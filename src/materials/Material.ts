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

    private uniformGroup: BindableGroup;
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
     * @param options.uniforms - Array of uniform descriptors for the material's bind group
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
            uniforms?: BindGroupDescriptor[],
            transparent?: boolean,
            depthWriteEnabled?: boolean,
            depthCompare?: GPUCompareFunction,
            cullMode?: GPUCullMode,
            topology?: GPUPrimitiveTopology,
            depthStencilFormat?: GPUTextureFormat,
        }
    ) {
        this.uniformGroup = new BindableGroup(this.options.uniforms || []);
        this.uuid = crypto.randomUUID();

        this.transparent = this.options.transparent || false;
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
     * Initializes the material by creating the shader module, bind group layouts, and render pipeline.
     * 
     * @param gpuDevice - The GPU device used for initialization.
     * @param vertexBuffersDescriptors - Descriptors for the vertex buffers.
     * @param presentationFormat - The format of the presentation surface.
     */
    public initialize(gpuDevice: GPUDevice, vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>, presentationFormat: GPUTextureFormat, sampleCount: number) {
        if (!this.shaderRenderModule) {
            this.createShaderModule(gpuDevice);
        }

        this.uniformGroup.createRenderingBindGroupLayout(gpuDevice);
        this.uniformGroup.createBindGroupLayout(gpuDevice);

        this.uniformGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
            label: "Render Pipeline Layout",
            bindGroupLayouts: [
                this.uniformGroup.bindGroupLayout!,
                this.uniformGroup.cameraBindablesGroupLayout!,
                this.uniformGroup.meshBindablesGroupLayout!
            ]
        });

        const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
            layout: this.uniformGroup.pipelineBindGroupLayout,
            label: "Render Pipeline",
            multisample: {
                count: sampleCount
            },
            vertex: {
                module: this.shaderRenderModule,
                entryPoint: 'vertex_main',
                buffers: vertexBuffersDescriptors
            } as GPUVertexState,
            fragment: {
                module: this.shaderRenderModule,
                entryPoint: 'fragment_main',
                targets: [
                    {
                        format: presentationFormat,
                        blend: {
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
                        }
                    }
                ]
            } as GPUFragmentState,
            primitive: {
                topology: this.topology,
                cullMode: this.transparent ? 'none' : this.cullMode,
            } as GPUPrimitiveState,

            depthStencil: {
                depthWriteEnabled: this.transparent ? false : this.depthWriteEnabled,
                depthCompare: this.depthCompare,
                format: this.depthStencilFormat,
            },
        }
        this.pipeline = gpuDevice.createRenderPipeline(renderPipelineDescriptor);

        this.initialized = true;
    }

    /**
     * Retrieves the bind group for the material.
     * 
     * @param gpuDevice - The GPU device used to get the bind group.
     * @returns The bind group associated with this material.
     */
    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.uniformGroup.getBindGroup(gpuDevice);
        return this.uniformGroup.bindGroup!;
    }
}

export { Material }
