import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Camera } from "../cameras/Camera";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Vector4 } from "../main";
import { Compute } from "../materials/Compute";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { GBuffer } from "../postprocessing/GBuffer";

/**
 * Configuration options for the WebGPU renderer.
 * @interface RendererOptions
 * @property {boolean} [antialias] - Enable antialiasing
 * @property {boolean} [premultipliedAlpha] - Enable premultiplied alpha
 * @property {GPUCanvasAlphaMode} [alphaMode] - Canvas alpha mode configuration
 * @property {Vector4} [clearColor] - Clear color for the renderer
 */
export interface RendererOptions {
    antialias?: boolean;
    premultipliedAlpha?: boolean;
    alphaMode?: GPUCanvasAlphaMode;
    clearColor?: Vector4;
    width?: number;
    height?: number;
    sampleCount?: number;
    devicePixelRatio?: number;
}

/**
 * Core WebGPU renderer class that handles initialization, rendering, and compute operations.
 * @class Renderer
 */
class Renderer {
    /**
     * Creates a new Renderer instance.
     * @constructor
     * @param {RendererOptions} options - Configuration options for the renderer
     * @throws {Error} Throws if WebGPU is not supported in the browser
     */

    public canvas: HTMLCanvasElement;
    public context: GPUCanvasContext | null;
    private device?: GPUDevice;
    private _presentationFormat?: GPUTextureFormat;
    private sampleCount: number = 4;
    private devicePixelRatio: number = window.devicePixelRatio;
    private colorTexture?: GPUTexture;
    private depthTexture?: GPUTexture;
    private width: number = 320;
    private height: number = 240;
    private clearColor: Vector4 = new Vector4(0, 0, 0, 0);

    // ── Public read-only accessors ──────────────────────────────────────────
    /** The initialised GPU device. Undefined before initialize() resolves. */
    public get gpuDevice(): GPUDevice { return this.device!; }
    /** Canvas colour format negotiated with the platform. */
    public get presentationFormat(): GPUTextureFormat { return this._presentationFormat!; }
    /** Render width in physical pixels (includes devicePixelRatio). */
    public get renderWidth(): number { return this.width; }
    /** Render height in physical pixels (includes devicePixelRatio). */
    public get renderHeight(): number { return this.height; }
    /** The shared per-object mesh bind group (normal + world matrices, dynamic offsets). */
    public get sharedMeshBindGroup(): GPUBindGroup | null { return this._sharedMeshBG; }
    /** Layout used for the shared mesh bind group. */
    public get sharedMeshBindGroupLayout(): GPUBindGroupLayout | null { return this._sharedMeshBGLayout; }
    /** Alignment stride for the shared matrix buffers (device minimum, typically 256). */
    public get matrixAlignment(): number { return this._matrixAlignment; }

    // Pre-recorded bundle for the standard canvas render pass.
    private _renderBundle: GPURenderBundle | null = null;
    private _lastObjectCount: number = -1;

    // Separate bundle for off-screen GBuffer rendering (rgba16float + depth32float).
    private _gbufferBundle: GPURenderBundle | null = null;
    private _gbufferLastObjectCount: number = -1;
    private _gbufferLastSampleCount: number = -1;

    // Depth-copy pipeline: resolves MSAA depth from depthMSAATexture → depthTexture.
    // A fullscreen render pass reads texture_depth_multisampled_2d (sample 0) and
    // writes it via @builtin(frag_depth) into the non-MSAA depth attachment so that
    // compute shaders can sample it as texture_depth_2d.
    private _depthCopyPipeline: GPURenderPipeline | null = null;
    private _depthCopyBGL: GPUBindGroupLayout | null = null;
    private _depthCopyBG: GPUBindGroup | null = null;
    private _depthCopyBGSource: GPUTexture | null = null;

    // Shared matrix buffers — all objects' world and normal matrices packed into
    // two large GPU buffers (one per type) with 256-byte aligned strides.
    // The renderer uploads all matrices in exactly 2 writeBuffer calls per frame
    // instead of 2×N individual calls.
    private _matrixAlignment: number = 256;  // device.limits.minUniformBufferOffsetAlignment
    private _worldMatricesBuf: GPUBuffer | null = null;
    private _normalMatricesBuf: GPUBuffer | null = null;
    private _worldMatricesStaging: Float32Array | null = null;
    private _normalMatricesStaging: Float32Array | null = null;
    private _sharedMeshBGLayout: GPUBindGroupLayout | null = null;
    private _sharedMeshBG: GPUBindGroup | null = null;
    private _sharedMeshObjectCount: number = -1;

    constructor(
        private options: RendererOptions = {}
    ) {
        this.canvas = document.createElement('canvas');
        this.context = this.canvas.getContext('webgpu');
        this.sampleCount = this.options.sampleCount || this.sampleCount;
        this.devicePixelRatio = this.options.devicePixelRatio || this.devicePixelRatio;
        this.width = this.options.width || this.width;
        this.height = this.options.height || this.height;
        this.clearColor = this.options.clearColor || this.clearColor;

        if (!this.context || navigator.gpu == null) {
            throw new Error('WebGPU is not supported');
        }
    }

    private async getDevice(): Promise<GPUDevice> {
        const adapter = await navigator.gpu.requestAdapter();
        if (adapter == null) {
            throw new Error("No WebGPU adapter found");
        }

        return adapter.requestDevice();
    }

    /**
     * Initializes the WebGPU device and context.
     * @async
     * @returns {Promise<void>}
     */
    public async initialize(): Promise<void> {
        if (!this.device) {
            const device = await this.getDevice();
            if (!this.device) {
                this.device = device;
                this._presentationFormat = navigator.gpu.getPreferredCanvasFormat();
                this.context?.configure({
                    device: this.device,
                    format: this._presentationFormat,
                    alphaMode: this.options?.alphaMode || "opaque",
                });

                this.setSize(
                    this.options.width || this.canvas.width,
                    this.options.height || this.canvas.height
                );
            }
        }
        return Promise.resolve();
    }

    /**
     * Sets the size of the rendering canvas and updates related resources.
     * @param {number} width - Canvas width in pixels
     * @param {number} height - Canvas height in pixels
     */
    public setSize(width: number, height: number) {
        this.width = width * this.devicePixelRatio;
        this.height = height * this.devicePixelRatio;
        this.canvas.width = this.width;
        this.canvas.height = this.height;

        this.depthTexture?.destroy();
        this.depthTexture = this.device!.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: this.sampleCount,
            dimension: '2d',
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        this.colorTexture?.destroy();
        this.colorTexture = this.device!.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: this.sampleCount,
            format: this.presentationFormat!,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    /**
     * Call this whenever the scene composition changes (objects added/removed,
     * material or pipeline reassigned) to force the render bundle to be rebuilt
     * on the next frame.
     */
    public invalidateBundle() {
        this._renderBundle = null;
        this._gbufferBundle = null;
    }

    /**
     * Creates or resizes the shared per-object matrix GPU buffers.
     *
     * All objects' world and normal matrices are packed into two large GPU
     * buffers (one per type) with 256-byte aligned strides so every object's
     * slice is accessible via a dynamic uniform buffer offset.  This lets the
     * renderer upload ALL matrices in exactly 2 writeBuffer calls per frame.
     */
    private _ensureSharedMeshResources(objectCount: number) {
        if (objectCount === this._sharedMeshObjectCount && this._sharedMeshBG !== null) return;

        const alignment = this.device!.limits.minUniformBufferOffsetAlignment as number ?? 256;
        this._matrixAlignment = alignment;

        const bufferSize = Math.max(objectCount * alignment, alignment); // at least one slot
        const floatsPerSlot = alignment / 4; // 64 floats for 256-byte alignment

        this._worldMatricesBuf?.destroy();
        this._normalMatricesBuf?.destroy();

        this._worldMatricesBuf = this.device!.createBuffer({
            label: 'WorldMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._normalMatricesBuf = this.device!.createBuffer({
            label: 'NormalMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Pre-zero the staging arrays — padding bytes stay zero every frame.
        this._worldMatricesStaging = new Float32Array(objectCount * floatsPerSlot);
        this._normalMatricesStaging = new Float32Array(objectCount * floatsPerSlot);

        // Create the layout once; all subsequent bind groups reuse it.
        if (!this._sharedMeshBGLayout) {
            this._sharedMeshBGLayout = this.device!.createBindGroupLayout({
                label: 'SharedMesh BindGroupLayout',
                entries: [
                    { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                    { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                ],
            });
        }

        // Bind each buffer's first 64 bytes; the dynamic offset shifts that
        // window to the i-th object's slot at draw time.
        this._sharedMeshBG = this.device!.createBindGroup({
            label: 'SharedMesh BindGroup',
            layout: this._sharedMeshBGLayout,
            entries: [
                { binding: 0, resource: { buffer: this._normalMatricesBuf, size: 64 } },
                { binding: 1, resource: { buffer: this._worldMatricesBuf,  size: 64 } },
            ],
        });

        this._sharedMeshObjectCount = objectCount;
        this._renderBundle = null;     // force bundle rebuild with new bind group
        this._gbufferBundle = null;
    }

    /**
     * Renders a stack using the specified camera.
     *
     * Each frame has three phases:
     *  1. Update — compute all matrices on the CPU, copy them into aligned
     *     staging arrays, then upload via exactly 2 writeBuffer calls.
     *  2. Bundle — pre-record all draw commands once into a GPURenderBundle
     *     (dynamic offsets into the shared buffers are baked per object).
     *  3. Execute — replay the bundle with a single executeBundles() call.
     *
     * @param {Scene} stack - The stack to render
     * @param {Camera} camera - The camera to use for rendering
     */
    public render(stack: Scene, camera: Camera) {
        stack.prepare(camera);
        camera.updateViewMatrix();

        const orderedObjects = stack.getOrderedObjects();
        const cameraBindGroup = camera.getBindGroup(this.device!);

        // Allocate / resize shared matrix buffers to match the current object count.
        this._ensureSharedMeshResources(orderedObjects.length);

        const alignment = this._matrixAlignment;
        const floatsPerSlot = alignment / 4; // stride in the staging Float32Array

        // Phase 1 — update all CPU matrices and write them into the staging arrays.
        // No per-object writeBuffer calls happen here.
        for (let i = 0; i < orderedObjects.length; i++) {
            const renderable = orderedObjects[i];

            if (!renderable.geometry.initialized) {
                renderable.geometry.initialize(this.device!);
            }
            if (!renderable.material.initialized) {
                renderable.material.initialize(
                    this.device!,
                    renderable.geometry.vertexBuffersDescriptors!,
                    this.presentationFormat!,
                    this.sampleCount
                );
            }
            if (renderable.geometry.isInstancedGeometry) {
                const geo = renderable.geometry as InstancedGeometry;
                for (const extraBuffer of geo.extraBuffers) {
                    if (!extraBuffer.initialized) extraBuffer.initialize(this.device!);
                }
            }

            renderable.updateModelMatrix();
            renderable.updateNormalMatrix(camera.viewMatrix);

            // Copy the 16-float matrices into their aligned slots in the staging arrays.
            // internalMat4 is the live Float32Array — no intermediate copy needed.
            this._worldMatricesStaging!.set(renderable.worldMatrix.internalMat4,  i * floatsPerSlot);
            this._normalMatricesStaging!.set(renderable.normalMatrix.internalMat4, i * floatsPerSlot);

            // Flush any dirty material-level buffers (textures, material uniforms).
            renderable.material.getBindGroup(this.device!);
        }

        // Upload ALL matrices to the GPU in exactly 2 writeBuffer calls.
        if (orderedObjects.length > 0) {
            this.device!.queue.writeBuffer(this._worldMatricesBuf!,  0, this._worldMatricesStaging!.buffer as ArrayBuffer);
            this.device!.queue.writeBuffer(this._normalMatricesBuf!, 0, this._normalMatricesStaging!.buffer as ArrayBuffer);
        }

        // Invalidate the bundle if any object had its material swapped this frame.
        for (const renderable of orderedObjects) {
            if (renderable.materialDirty) {
                this._renderBundle = null;
                renderable.materialDirty = false;
            }
        }

        // Phase 2 — (re-)record the render bundle when the scene composition changes.
        if (!this._renderBundle || this._lastObjectCount !== orderedObjects.length) {
            this._renderBundle = this._buildRenderBundle(orderedObjects, cameraBindGroup);
            this._lastObjectCount = orderedObjects.length;
        }

        // Phase 3 — execute the pre-recorded bundle in a fresh render pass.
        const commandRenderEncoder = this.device!.createCommandEncoder();
        const textureView = this.context!.getCurrentTexture().createView();

        const renderPassDescriptor = {
            colorAttachments: [
                {
                    view: this.sampleCount > 1 ? this.colorTexture!.createView() : textureView,
                    resolveTarget: this.sampleCount > 1 ? textureView : undefined,
                    clearValue: {
                        r: this.options.clearColor?.x || 0.0,
                        g: this.options.clearColor?.y || 0.0,
                        b: this.options.clearColor?.z || 0.0,
                        a: this.options.clearColor?.w || 0.0
                    },
                    loadOp: 'clear',
                    storeOp: 'store',
                },
            ],
            depthStencilAttachment: {
                view: this.depthTexture!.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        } as GPURenderPassDescriptor;

        const passRenderEncoder = commandRenderEncoder.beginRenderPass(renderPassDescriptor);
        passRenderEncoder.executeBundles([this._renderBundle!]);
        passRenderEncoder.end();
        this.device!.queue.submit([commandRenderEncoder.finish()]);
    }

    // ── Depth-copy helpers ───────────────────────────────────────────────────

    /**
     * Builds the depth-copy render pipeline on first use.
     * The pipeline draws a fullscreen triangle, reads sample 0 from a
     * texture_depth_multisampled_2d, and writes it as @builtin(frag_depth)
     * into a non-MSAA depth32float attachment.
     */
    private _ensureDepthCopyPipeline(): void {
        if (this._depthCopyPipeline) return;

        const shader = /* wgsl */`
            @group(0) @binding(0) var msaaDepth: texture_depth_multisampled_2d;

            @vertex
            fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4f {
                const pos = array<vec2f, 3>(
                    vec2f(-1.0, -1.0),
                    vec2f( 3.0, -1.0),
                    vec2f(-1.0,  3.0),
                );
                return vec4f(pos[vi], 0.0, 1.0);
            }

            struct DepthOut { @builtin(frag_depth) depth: f32 }

            @fragment
            fn fs(@builtin(position) fragPos: vec4f) -> DepthOut {
                let coord = vec2i(i32(fragPos.x), i32(fragPos.y));
                return DepthOut(textureLoad(msaaDepth, coord, 0));
            }
        `;

        const module = this.device!.createShaderModule({ code: shader });

        this._depthCopyBGL = this.device!.createBindGroupLayout({
            label: 'DepthCopy/BGL',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.FRAGMENT,
                texture: { sampleType: 'depth', multisampled: true },
            }],
        });

        this._depthCopyPipeline = this.device!.createRenderPipeline({
            label: 'DepthCopy/Pipeline',
            layout: this.device!.createPipelineLayout({ bindGroupLayouts: [this._depthCopyBGL] }),
            vertex: { module, entryPoint: 'vs' },
            fragment: { module, entryPoint: 'fs', targets: [] },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'always',
            },
            primitive: { topology: 'triangle-list' },
        });
    }

    /** Returns (and lazily creates) the depth-copy bind group for the given MSAA depth texture. */
    private _getDepthCopyBindGroup(msaaDepth: GPUTexture): GPUBindGroup {
        if (this._depthCopyBGSource !== msaaDepth) {
            this._depthCopyBG = this.device!.createBindGroup({
                label: 'DepthCopy/BindGroup',
                layout: this._depthCopyBGL!,
                entries: [{ binding: 0, resource: msaaDepth.createView() }],
            });
            this._depthCopyBGSource = msaaDepth;
        }
        return this._depthCopyBG!;
    }

    /**
     * Renders the scene into a GBuffer for post-processing.
     *
     * This is a drop-in replacement for render() when a PostProcessingVolume is in use.
     * It performs the same three-phase matrix-upload / bundle-record / execute loop but
     * targets the GBuffer's rgba16float colour texture and depth32float depth texture at
     * sampleCount=1 (no MSAA — post-processing handles aliasing via FXAA etc.).
     *
     * @param stack   - The scene to render.
     * @param camera  - The camera to use.
     * @param gbuffer - The GBuffer to write colour and depth data into.
     */
    public renderToGBuffer(stack: Scene, camera: Camera, gbuffer: GBuffer): void {
        stack.prepare(camera);
        camera.updateViewMatrix();

        const orderedObjects = stack.getOrderedObjects();
        const cameraBindGroup = camera.getBindGroup(this.device!);

        this._ensureSharedMeshResources(orderedObjects.length);

        const alignment = this._matrixAlignment;
        const floatsPerSlot = alignment / 4;

        // Phase 1 — identical to render(): upload matrices.
        for (let i = 0; i < orderedObjects.length; i++) {
            const renderable = orderedObjects[i];

            if (!renderable.geometry.initialized) {
                renderable.geometry.initialize(this.device!);
            }
            // Ensure the material has a pipeline compiled for the GBuffer config.
            renderable.material.getPipelineForConfig(
                this.device!,
                renderable.geometry.vertexBuffersDescriptors,
                'rgba16float',
                gbuffer.msaaSampleCount,
                'depth32float'
            );
            if (!renderable.material.initialized) {
                renderable.material.initialize(
                    this.device!,
                    renderable.geometry.vertexBuffersDescriptors,
                    this._presentationFormat!,
                    this.sampleCount
                );
            }
            if (renderable.geometry.isInstancedGeometry) {
                const geo = renderable.geometry as InstancedGeometry;
                for (const extraBuffer of geo.extraBuffers) {
                    if (!extraBuffer.initialized) extraBuffer.initialize(this.device!);
                }
            }

            renderable.updateModelMatrix();
            renderable.updateNormalMatrix(camera.viewMatrix);

            this._worldMatricesStaging!.set(renderable.worldMatrix.internalMat4,  i * floatsPerSlot);
            this._normalMatricesStaging!.set(renderable.normalMatrix.internalMat4, i * floatsPerSlot);

            renderable.material.getBindGroup(this.device!);
        }

        if (orderedObjects.length > 0) {
            this.device!.queue.writeBuffer(this._worldMatricesBuf!,  0, this._worldMatricesStaging!.buffer as ArrayBuffer);
            this.device!.queue.writeBuffer(this._normalMatricesBuf!, 0, this._normalMatricesStaging!.buffer as ArrayBuffer);
        }

        for (const renderable of orderedObjects) {
            if (renderable.materialDirty) {
                this._gbufferBundle = null;
                renderable.materialDirty = false;
            }
        }

        // Phase 2 — build GBuffer bundle if stale.
        if (!this._gbufferBundle ||
            this._gbufferLastObjectCount !== orderedObjects.length ||
            this._gbufferLastSampleCount !== gbuffer.msaaSampleCount) {
            this._gbufferBundle = this._buildRenderBundle(
                orderedObjects, cameraBindGroup, 'rgba16float', gbuffer.msaaSampleCount, 'depth32float'
            );
            this._gbufferLastObjectCount = orderedObjects.length;
            this._gbufferLastSampleCount = gbuffer.msaaSampleCount;
        }

        // Phase 3 — execute into the GBuffer render pass.
        const commandEncoder = this.device!.createCommandEncoder();

        const clearColor = {
            r: this.options.clearColor?.x || 0.0,
            g: this.options.clearColor?.y || 0.0,
            b: this.options.clearColor?.z || 0.0,
            a: this.options.clearColor?.w || 0.0,
        };

        let passDescriptor: GPURenderPassDescriptor;
        if (gbuffer.msaaSampleCount > 1 && gbuffer.colorMSAATexture && gbuffer.depthMSAATexture) {
            // MSAA path: render into multi-sample textures, resolve colour automatically.
            // MSAA depth is stored so the depth-copy pass can read it next.
            passDescriptor = {
                colorAttachments: [{
                    view: gbuffer.colorMSAATexture.createView(),
                    resolveTarget: gbuffer.colorTexture.createView(),
                    clearValue: clearColor,
                    loadOp: 'clear',
                    storeOp: 'discard', // MSAA samples discarded after resolve
                }],
                depthStencilAttachment: {
                    view: gbuffer.depthMSAATexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store', // keep MSAA depth for the depth-copy pass
                },
            };
        } else {
            // Non-MSAA path (original behaviour).
            passDescriptor = {
                colorAttachments: [{
                    view: gbuffer.colorTexture.createView(),
                    clearValue: clearColor,
                    loadOp: 'clear',
                    storeOp: 'store',
                }],
                depthStencilAttachment: {
                    view: gbuffer.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            };
        }

        const pass = commandEncoder.beginRenderPass(passDescriptor);
        pass.executeBundles([this._gbufferBundle!]);
        pass.end();

        // Depth-copy pass: resolve MSAA depth → non-MSAA depthTexture for compute shaders.
        if (gbuffer.msaaSampleCount > 1 && gbuffer.depthMSAATexture) {
            this._ensureDepthCopyPipeline();
            const depthCopyPass = commandEncoder.beginRenderPass({
                colorAttachments: [],
                depthStencilAttachment: {
                    view: gbuffer.depthTexture.createView(),
                    depthClearValue: 1.0,
                    depthLoadOp: 'clear',
                    depthStoreOp: 'store',
                },
            });
            depthCopyPass.setPipeline(this._depthCopyPipeline!);
            depthCopyPass.setBindGroup(0, this._getDepthCopyBindGroup(gbuffer.depthMSAATexture));
            depthCopyPass.draw(3);
            depthCopyPass.end();
        }

        this.device!.queue.submit([commandEncoder.finish()]);
    }

    /**
     * Records all draw commands into a GPURenderBundle.
     *
     * @param colorFormat   Target colour attachment format. Defaults to the canvas presentation format.
     * @param sampleCount   MSAA sample count of the render pass. Defaults to the renderer's sampleCount.
     * @param depthFormat   Depth-stencil format. Defaults to 'depth24plus'.
     */
    private _buildRenderBundle(
        orderedObjects: Renderable[],
        cameraBindGroup: GPUBindGroup,
        colorFormat: GPUTextureFormat = this._presentationFormat!,
        sampleCount: number = this.sampleCount,
        depthFormat: GPUTextureFormat = 'depth24plus'
    ): GPURenderBundle {
        const encoder = this.device!.createRenderBundleEncoder({
            colorFormats: [colorFormat],
            depthStencilFormat: depthFormat,
            sampleCount,
        });

        let currentPipeline: GPURenderPipeline | null = null;
        let currentMaterialBindGroup: GPUBindGroup | null = null;
        let currentIndexBuffer: GPUBuffer | null = null;
        let currentVertexBuffer: GPUBuffer | null = null;

        const alignment = this._matrixAlignment;

        // Camera bind group is the same for every object — set once.
        encoder.setBindGroup(2, cameraBindGroup);

        for (let i = 0; i < orderedObjects.length; i++) {
            const renderable = orderedObjects[i];
            // For off-screen passes request the pipeline for the specific config.
            const pipeline = renderable.material.getPipelineForConfig(
                this.device!,
                renderable.geometry.vertexBuffersDescriptors,
                colorFormat,
                sampleCount,
                depthFormat
            );
            if (!pipeline || !renderable.geometry.initialized) continue;

            if (pipeline !== currentPipeline) {
                encoder.setPipeline(pipeline);
                currentPipeline = pipeline;
                currentMaterialBindGroup = null;
            }

            if (renderable.geometry.indexBuffer !== currentIndexBuffer) {
                encoder.setIndexBuffer(renderable.geometry.indexBuffer!, renderable.geometry.indexFormat!);
                currentIndexBuffer = renderable.geometry.indexBuffer!;
            }

            if (renderable.geometry.vertexBuffer !== currentVertexBuffer) {
                encoder.setVertexBuffer(0, renderable.geometry.vertexBuffer!);
                currentVertexBuffer = renderable.geometry.vertexBuffer!;
            }

            if (renderable.geometry.isInstancedGeometry) {
                const geo = renderable.geometry as InstancedGeometry;
                let idx = 1;
                for (const extraBuffer of geo.extraBuffers) {
                    encoder.setVertexBuffer(idx++, extraBuffer.resource.buffer);
                }
            }

            const materialBindGroup = renderable.material.getBindGroup(this.device!);
            if (materialBindGroup !== currentMaterialBindGroup) {
                encoder.setBindGroup(0, materialBindGroup);
                currentMaterialBindGroup = materialBindGroup;
            }

            // Bake the per-object dynamic offset: both bindings (normalMatrix,
            // worldMatrix) live in separate buffers but share the same stride.
            const offset = i * alignment;
            encoder.setBindGroup(1, this._sharedMeshBG!, [offset, offset]);

            if (renderable.geometry.isInstancedGeometry) {
                const geo = renderable.geometry as InstancedGeometry;
                encoder.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
            } else {
                encoder.drawIndexed(renderable.geometry.vertexCount);
            }
        }

        return encoder.finish();
    }

    /**
     * Executes a compute shader with the specified workgroup configuration.
     * @async
     * @param {Compute} compute - The compute shader to execute
     * @param {number} [workgroupsX=64] - Number of workgroups in X dimension
     * @param {number} [workgroupsY=1] - Number of workgroups in Y dimension
     * @param {number} [workgroupsZ=1] - Number of workgroups in Z dimension
     * @returns {Promise<void>}
     */
    public async compute(compute: Compute, workgroupsX: number = 64, workgroupsY: number = 1, workgroupsZ: number = 1): Promise<void> {
        const commandEncoder = this.device!.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        if (!compute.initialized) {
            compute.initialize(this.device!);
        }
        const bindGroup = compute.getBindGroup(this.device!);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.setPipeline(compute.pipeline!);
        passEncoder.dispatchWorkgroups(workgroupsX, workgroupsY, workgroupsZ);
        passEncoder.end();
        const commands = commandEncoder.finish();

        this.device!.queue.submit([commands]);

        // Wait for the compute work to complete
        return this.device!.queue.onSubmittedWorkDone();
    }

    /**
     * Reads data back from a compute buffer.
     * @async
     * @template T
     * @param {ComputeBuffer} buffer - The buffer to read from
     * @param {new (buffer: ArrayBuffer) => T} ArrayType - The type of array to create
     * @returns {Promise<T>} The buffer data
     */
    public async readBackBuffer<T extends Float32Array | Uint32Array | Int32Array>(
        buffer: ComputeBuffer,
        ArrayType: new (buffer: ArrayBuffer) => T
    ): Promise<T> {
        const stagingBuffer = this.device!.createBuffer({
            size: buffer.resource.buffer!.size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        const commandEncoder = this.device!.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            buffer.resource.buffer,
            0, // Source offset
            stagingBuffer,
            0, // Destination offset
            buffer.resource.buffer!.size
        );

        this.device!.queue.submit([commandEncoder.finish()]);

        await stagingBuffer.mapAsync(
            GPUMapMode.READ,
            0, // Offset
            buffer.resource.buffer!.size // Length
        );
        const copyArrayBuffer = stagingBuffer.getMappedRange(0, buffer.resource.buffer!.size);
        const data = new ArrayType(copyArrayBuffer.slice(0));
        stagingBuffer.unmap();

        return data;
    }
}

export { Renderer };
