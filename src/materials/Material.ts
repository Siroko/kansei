import { BindGroupDescriptor, BindableGroup } from "./BindableGroup";
import { parseIncludes } from "./shaders/ShaderUtils";

/**
 * Represents a material used in rendering, encapsulating shader modules and pipeline configurations.
 */
class Material {

    public shaderRenderModule?: GPUShaderModule;
    public pipeline?: GPURenderPipeline;
    public initialized: boolean = false;
    public uuid: string;
    public transparent: boolean = false;
    public outputsEmissive: boolean = false;
    /** When true, the renderer draws this object AFTER opaque geometry and AFTER
     *  snapshotting the opaque result into GBuffer.backgroundTexture — so a
     *  downstream post-processing effect can sample the undistorted background
     *  to compute screen-space refraction. */
    public transmissive: boolean = false;

    // Cache of render pipelines keyed by "colorFormat:sampleCount:depthFormat".
    // Allows the same material to be used in multiple render passes with different
    // target configurations (e.g. canvas MSAA pass vs. off-screen GBuffer pass).
    private _pipelineCache: Map<string, GPURenderPipeline> = new Map();

    private bindableGroup: BindableGroup;
    private depthWriteEnabled: boolean = true;
    private depthCompare: GPUCompareFunction = 'less';
    private cullMode: GPUCullMode = 'back';
    private topology: GPUPrimitiveTopology = 'triangle-list';
    private depthStencilFormat: GPUTextureFormat = 'depth24plus';

    /**
     * Constructs a new Material instance.
     * 
     * @param shaderCode - The shader code to be used for this material.
     * @param options - Configuration options for the material
     * @param options.bindings - Array of uniform descriptors for the material's bind group
     * @param options.transparent - Whether the material is transparent. Affects depth writing and culling. Defaults to false
     * @param options.depthWriteEnabled - Whether depth writing is enabled. Defaults to true
     * @param options.depthCompare - Depth comparison function. Defaults to 'less'
     * @param options.cullMode - Face culling mode. Defaults to 'back'
     * @param options.topology - Primitive topology type. Defaults to 'triangle-list'
     * @param options.depthStencilFormat - Format of the depth/stencil buffer. Defaults to 'depth24plus'
     */
    constructor(
        private shaderCode: string,
        private options: {
            bindings?: BindGroupDescriptor[],
            transparent?: boolean,
            transmissive?: boolean,
            depthWriteEnabled?: boolean,
            depthCompare?: GPUCompareFunction,
            cullMode?: GPUCullMode,
            topology?: GPUPrimitiveTopology,
            depthStencilFormat?: GPUTextureFormat,
            outputsEmissive?: boolean,
        }
    ) {
        this.bindableGroup = new BindableGroup(this.options.bindings || []);
        this.uuid = crypto.randomUUID();

        this.transparent = this.options.transparent || false;
        this.transmissive = this.options.transmissive || false;
        this.outputsEmissive = this.options.outputsEmissive || false;
        this.depthWriteEnabled = this.options.depthWriteEnabled ?? true;
        this.depthCompare = this.options.depthCompare || 'less';
        this.cullMode = this.options.cullMode || 'back';
        this.topology = this.options.topology || 'triangle-list';
        this.depthStencilFormat = this.options.depthStencilFormat || 'depth24plus';
    }

    /**
     * Creates a shader module from the provided shader code.
     * 
     * @param gpuDevice - The GPU device used to create the shader module.
     */
    private createShaderModule(gpuDevice: GPUDevice) {
        this.shaderRenderModule = gpuDevice.createShaderModule({
            code: parseIncludes(this.shaderCode)
        });
    }

    /**
     * Ensures the shader module and shared bind group layouts are created exactly once.
     */
    private _ensureSharedResources(gpuDevice: GPUDevice): void {
        if (!this.shaderRenderModule) {
            this.createShaderModule(gpuDevice);
        }

        if (!this.bindableGroup.bindGroupLayout) {
            this.bindableGroup.createRenderingBindGroupLayout(gpuDevice);
            this.bindableGroup.createBindGroupLayout(gpuDevice);

            this.bindableGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
                label: "Render Pipeline Layout",
                bindGroupLayouts: [
                    this.bindableGroup.bindGroupLayout!,
                    this.bindableGroup.cameraBindablesGroupLayout!,
                    this.bindableGroup.meshBindablesGroupLayout!,
                    this.bindableGroup.shadowBindablesGroupLayout!,
                ]
            });
        }
    }

    /**
     * Builds a GPURenderPipeline for the given (colorFormat, sampleCount, depthFormat) combination.
     */
    private _buildPipeline(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        colorFormat: GPUTextureFormat,
        sampleCount: number,
        depthFormat: GPUTextureFormat,
        colorTargetCount: number = 1,
        colorFormats?: GPUTextureFormat[]
    ): GPURenderPipeline {
        const targets: GPUColorTargetState[] = [];

        if (colorFormats && colorFormats.length > 0) {
            // Mixed-format MRT: build one target per entry in colorFormats.
            // Blend is only applied to the first target (@location(0)) for
            // transparent materials — additional targets receive raw values.
            for (const fmt of colorFormats) {
                const target: GPUColorTargetState = { format: fmt };
                if (targets.length === 0 && this.transparent) {
                    target.blend = {
                        color: {
                            operation: 'add',
                            srcFactor: 'src-alpha',
                            dstFactor: 'one-minus-src-alpha'
                        },
                        alpha: {
                            operation: 'add',
                            srcFactor: 'one',
                            dstFactor: 'one-minus-src-alpha'
                        }
                    };
                }
                targets.push(target);
            }
        } else {
            // Single-format path (backwards compatible).
            const colorTarget: GPUColorTargetState = { format: colorFormat };
            if (this.transparent) {
                colorTarget.blend = {
                    color: {
                        operation: 'add',
                        srcFactor: 'src-alpha',
                        dstFactor: 'one-minus-src-alpha'
                    },
                    alpha: {
                        operation: 'add',
                        srcFactor: 'one',
                        dstFactor: 'one-minus-src-alpha'
                    }
                };
            }
            targets.push(colorTarget);
            for (let t = 1; t < colorTargetCount; t++) {
                targets.push({ format: colorFormat });
            }
        }

        const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
            layout: this.bindableGroup.pipelineBindGroupLayout!,
            label: "Render Pipeline",
            multisample: { count: sampleCount },
            vertex: {
                module: this.shaderRenderModule!,
                entryPoint: 'vertex_main',
                buffers: vertexBuffersDescriptors
            } as GPUVertexState,
            fragment: {
                module: this.shaderRenderModule!,
                entryPoint: 'fragment_main',
                targets,
            } as GPUFragmentState,
            primitive: {
                topology: this.topology,
                cullMode: this.transparent ? 'none' : this.cullMode,
            } as GPUPrimitiveState,
            depthStencil: {
                depthWriteEnabled: this.options.depthWriteEnabled ?? (this.transparent ? false : this.depthWriteEnabled),
                depthCompare: this.depthCompare,
                format: depthFormat,
            },
        };
        return gpuDevice.createRenderPipeline(renderPipelineDescriptor);
    }

    /**
     * Returns a cached pipeline for the given configuration, creating it if it does not exist.
     *
     * This allows the same material to be used in multiple render passes that differ in
     * color format, sample count, or depth format — e.g. the canvas MSAA pass and an
     * off-screen GBuffer pass — without re-compiling the shader.
     *
     * @param gpuDevice - The GPU device.
     * @param vertexBuffersDescriptors - Vertex buffer layout descriptors (only used on first call per key).
     * @param colorFormat - Target color attachment format.
     * @param sampleCount - MSAA sample count of the render pass.
     * @param depthFormat - Depth-stencil attachment format. Defaults to 'depth24plus'.
     */
    public getPipelineForConfig(
        gpuDevice: GPUDevice,
        vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
        colorFormat: GPUTextureFormat,
        sampleCount: number,
        depthFormat: GPUTextureFormat = 'depth24plus',
        colorTargetCount: number = 1,
        colorFormats?: GPUTextureFormat[]
    ): GPURenderPipeline {
        this._ensureSharedResources(gpuDevice);
        const key = colorFormats
            ? `${colorFormats.join(',')}:${sampleCount}:${depthFormat}`
            : `${colorFormat}:${sampleCount}:${depthFormat}:${colorTargetCount}`;
        if (!this._pipelineCache.has(key)) {
            this._pipelineCache.set(
                key,
                this._buildPipeline(gpuDevice, vertexBuffersDescriptors, colorFormat, sampleCount, depthFormat, colorTargetCount, colorFormats)
            );
        }
        return this._pipelineCache.get(key)!;
    }

    /**
     * Initializes the material by creating the shader module, bind group layouts, and render pipeline.
     *
     * @param gpuDevice - The GPU device used for initialization.
     * @param vertexBuffersDescriptors - Descriptors for the vertex buffers.
     * @param presentationFormat - The format of the presentation surface.
     * @param sampleCount - MSAA sample count for the render pass.
     */
    public initialize(gpuDevice: GPUDevice, vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>, presentationFormat: GPUTextureFormat, sampleCount: number) {
        this._ensureSharedResources(gpuDevice);
        this.pipeline = this.getPipelineForConfig(
            gpuDevice, vertexBuffersDescriptors, presentationFormat, sampleCount, this.depthStencilFormat
        );
        this.initialized = true;
    }

    /**
     * Retrieves the bind group for the material.
     * 
     * @param gpuDevice - The GPU device used to get the bind group.
     * @returns The bind group associated with this material.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.bindableGroup.getBindGroup(gpuDevice);
        return this.bindableGroup.bindGroup!;
    }
}

export { Material }
