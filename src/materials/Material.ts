import { BindGroupDescriptor, UniformGroup } from "./UniformGroup";
import { parseIncludes } from "./shaders/ShaderUtils";

/**
 * Represents a material used in rendering, encapsulating shader modules and pipeline configurations.
 */
class Material {

    public shaderRenderModule?: GPUShaderModule;
    public pipeline?: GPURenderPipeline;
    public initialized: boolean = false;

    public uuid: string;

    private uniformGroup: UniformGroup;

    /**
     * Constructs a new Material instance.
     * 
     * @param shaderCode - The shader code to be used for this material.
     * @param uniforms - The descriptor for the bind group uniforms.
     */
    constructor(
        private shaderCode: string,
        public uniforms: BindGroupDescriptor[],
        public transparent: boolean = false,
        public depthWriteEnabled: boolean = true,
        public depthCompare: GPUCompareFunction = 'less',
        public cullMode: GPUCullMode = 'back',
        public topology: GPUPrimitiveTopology = 'triangle-list',
        public depthStencilFormat: GPUTextureFormat = 'depth24plus',
    ) {
        this.uniformGroup = new UniformGroup(uniforms);
        this.uuid = crypto.randomUUID();
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

        this.uniformGroup.createRenderingMaterialUniformsBindGroupLayout(gpuDevice);
        this.uniformGroup.createBindGroupLayout(gpuDevice);

        this.uniformGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
            label: "Render Pipeline Layout",
            bindGroupLayouts: [
                this.uniformGroup.bindGroupLayout!,
                this.uniformGroup.cameraUniformsGroupLayout!,
                this.uniformGroup.meshUniformsGroupLayout!
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
                topology: 'triangle-list',
                cullMode: this.transparent ? 'none' : 'back',
            } as GPUPrimitiveState,

            depthStencil: {
                depthWriteEnabled: !this.transparent,
                depthCompare: 'less',
                format: 'depth24plus',
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
