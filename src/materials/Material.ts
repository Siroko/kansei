import { BindGroupDescriptor, UniformGroup } from "./UniformGroup";

class Material {

    public shaderRenderModule?: GPUShaderModule;
    public pipeline?: GPURenderPipeline;
    public initialized: boolean = false;

    public uuid: string;

    private uniformGroup: UniformGroup;

    constructor(
        private shaderCode: string,
        public uniforms: BindGroupDescriptor[]
    ) {
        this.uniformGroup = new UniformGroup(uniforms);
        this.uuid = crypto.randomUUID();
    }

    private createShaderModule(gpuDevice: GPUDevice) {
        this.shaderRenderModule = gpuDevice.createShaderModule({
            code: this.shaderCode
        });
    }

    public initialize(gpuDevice: GPUDevice, vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>, presentationFormat: GPUTextureFormat) {
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
                    },
                ],
            } as GPUFragmentState,
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            } as GPUPrimitiveState,

            depthStencil: {
                depthWriteEnabled: true,
                depthCompare: 'less',
                format: 'depth24plus',
            },
        }
        this.pipeline = gpuDevice.createRenderPipeline(renderPipelineDescriptor);

        this.initialized = true;
    }

    public getBindGroup(gpuDevice: GPUDevice): GPUBindGroup {
        this.uniformGroup.getBindGroup(gpuDevice);
        return this.uniformGroup.bindGroup!;
    }
}

export { Material }