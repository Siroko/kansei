import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Camera } from "../cameras/Camera";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { Vector4 } from "../main";
import { Compute } from "../materials/Compute";
import { Object3D } from "../objects/Object3D";
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

    // Per-pass state tracking — reset at the start of each render() call so
    // redundant setPipeline / setBindGroup / setIndexBuffer / setVertexBuffer
    // commands are skipped when consecutive objects share the same state.
    private _currentPipeline: GPURenderPipeline | null = null;
    private _currentMaterialBindGroup: GPUBindGroup | null = null;
    private _currentIndexBuffer: GPUBuffer | null = null;
    private _currentVertexBuffer: GPUBuffer | null = null;

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
     * Renders a stack using the specified camera.
     * @param {Scene} stack - The stack to render
     * @param {Camera} camera - The camera to use for rendering
     */
    public render(stack: Scene, camera: Camera) {

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
        const passRenderEncoder = commandRenderEncoder!.beginRenderPass(renderPassDescriptor);

        // Prepare stack (sort transparent objects)
        stack.prepare(camera);

        camera.updateViewMatrix();
        const cameraBindGroup = camera.getBindGroup(this.device!);

        // Reset per-pass state so the first object always sets everything
        this._currentPipeline = null;
        this._currentMaterialBindGroup = null;
        this._currentIndexBuffer = null;
        this._currentVertexBuffer = null;

        // Camera bind group (slot 2) is identical for every object — set once
        passRenderEncoder.setBindGroup(2, cameraBindGroup);

        // Render objects in order (opaque first, then transparent)
        for (const object of stack.getOrderedObjects()) {
            this.renderObject(object, passRenderEncoder, camera);
        }
        passRenderEncoder!.end();
        this.device!.queue.submit([commandRenderEncoder!.finish()]);
        // Wait for the render work to complete
        // return this.device!.queue.onSubmittedWorkDone();
    }

    /**
     * Renders a single object, skipping redundant pipeline/buffer/bind-group
     * state changes when consecutive objects share the same state.
     * @private
     * @param {Object3D} object - The object to render
     * @param {GPURenderPassEncoder} passRenderEncoder - The current render pass encoder
     * @param {Camera} camera - The camera used for rendering
     */
    private renderObject(
        object: Object3D,
        passRenderEncoder: GPURenderPassEncoder,
        camera: Camera
    ) {
        if (!object.isRenderable) return;

        const renderable = object as Renderable;
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

        // Pipeline — skip if unchanged; a new pipeline may have a different
        // layout at slot 0 so force the material bind group to be re-set.
        if (renderable.material.pipeline !== this._currentPipeline) {
            passRenderEncoder.setPipeline(renderable.material.pipeline!);
            this._currentPipeline = renderable.material.pipeline!;
            this._currentMaterialBindGroup = null;
        }

        // Index buffer — skip if same geometry
        if (renderable.geometry.indexBuffer !== this._currentIndexBuffer) {
            passRenderEncoder.setIndexBuffer(renderable.geometry.indexBuffer!, renderable.geometry.indexFormat!);
            this._currentIndexBuffer = renderable.geometry.indexBuffer!;
        }

        // Primary vertex buffer — skip if same geometry
        if (renderable.geometry.vertexBuffer !== this._currentVertexBuffer) {
            passRenderEncoder.setVertexBuffer(0, renderable.geometry.vertexBuffer!);
            this._currentVertexBuffer = renderable.geometry.vertexBuffer!;
        }

        // Extra per-instance vertex buffers (instanced geometry only)
        if (renderable.geometry.isInstancedGeometry) {
            const geo = renderable.geometry as InstancedGeometry;
            let vertexBufferIndex = 1;
            for (const extraBuffer of geo.extraBuffers) {
                if (!extraBuffer.initialized) {
                    extraBuffer.initialize(this.device!);
                }
                passRenderEncoder.setVertexBuffer(vertexBufferIndex, extraBuffer.resource.buffer);
                vertexBufferIndex++;
            }
        }

        renderable.updateModelMatrix();
        renderable.updateNormalMatrix(camera.viewMatrix);

        // Material bind group (slot 0) — skip if same material
        const materialBindGroup = renderable.material.getBindGroup(this.device!);
        if (materialBindGroup !== this._currentMaterialBindGroup) {
            passRenderEncoder.setBindGroup(0, materialBindGroup);
            this._currentMaterialBindGroup = materialBindGroup;
        }

        // Mesh bind group (slot 1) — unique per object, always set
        passRenderEncoder.setBindGroup(1, renderable.getBindGroup(this.device!));

        // Slot 2 (camera) is set once in render() — no per-object call needed

        if (renderable.geometry.isInstancedGeometry) {
            const geo = renderable.geometry as InstancedGeometry;
            passRenderEncoder.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
        } else {
            passRenderEncoder.drawIndexed(renderable.geometry.vertexCount);
        }
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
