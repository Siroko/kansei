import { IBindable } from "../buffers/IBindable";
import { UniformGroup } from "./UniformGroup";

export class BindGroupDescriptor {
    binding?: number;
    visibility?: GPUFlagsConstant;
    value?: IBindable;
}

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

        if (!this.uniformGroup.bindGroupLayout) {
            this.uniformGroup.createBindGroupLayout(gpuDevice);
        }

        const renderPipelineDescriptor: GPURenderPipelineDescriptor = {
            layout: "auto",
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

    public async getBindGroup(gpuDevice: GPUDevice): Promise<GPUBindGroup> {

        await this.uniformGroup.getBindGroup(gpuDevice, this.pipeline!);
        return new Promise((resolve) => {
            resolve(this.uniformGroup.bindGroup!);
        });
    }
}

export { Material }