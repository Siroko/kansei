import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Camera } from "../cameras/Camera";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Vector4 } from "../main";
import { Compute } from "../materials/Compute";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";

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
    private presentationFormat?: GPUTextureFormat;
    private sampleCount: number = 4;
    private devicePixelRatio: number = window.devicePixelRatio;
    private colorTexture?: GPUTexture;
    private depthTexture?: GPUTexture;
    private width: number = 320;
    private height: number = 240;
    private clearColor: Vector4 = new Vector4(0, 0, 0, 0);

    // Pre-recorded bundle of all draw commands. Re-built only when the scene
    // composition changes (objects added/removed, pipeline reassigned, etc.).
    private _renderBundle: GPURenderBundle | null = null;
    private _lastObjectCount: number = -1;

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
                this.presentationFormat = navigator.gpu.getPreferredCanvasFormat();
                this.context?.configure({
                    device: this.device,
                    format: this.presentationFormat,
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
        this._renderBundle = null; // force bundle rebuild with new bind group
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

    /**
     * Records all draw commands into a GPURenderBundle.
     *
     * Each object uses the shared mesh bind group with a baked dynamic offset
     * that selects its 64-byte matrix slice from the large shared buffers.
     * The bundle is recorded once and replayed cheaply every frame.
     */
    private _buildRenderBundle(
        orderedObjects: Renderable[],
        cameraBindGroup: GPUBindGroup
    ): GPURenderBundle {
        const encoder = this.device!.createRenderBundleEncoder({
            colorFormats: [this.presentationFormat!],
            depthStencilFormat: 'depth24plus',
            sampleCount: this.sampleCount,
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
            if (!renderable.material.pipeline || !renderable.geometry.initialized) continue;

            if (renderable.material.pipeline !== currentPipeline) {
                encoder.setPipeline(renderable.material.pipeline);
                currentPipeline = renderable.material.pipeline;
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
