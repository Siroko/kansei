import { BindGroupDescriptor, UniformGroup } from "./UniformGroup";
class Compute {

    public shaderComputeModule?: GPUShaderModule;
    public pipeline?: GPUComputePipeline;
    public bindGroupLayout?: GPUBindGroupLayout;
    public bindGroup?: GPUBindGroup;
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
        this.shaderComputeModule = gpuDevice.createShaderModule({
            code: this.shaderCode
        });
    }

    public initialize(gpuDevice: GPUDevice) {
        if (!this.shaderComputeModule) {
            this.createShaderModule(gpuDevice);
        }

        this.uniformGroup.createBindGroupLayout(gpuDevice);
        this.uniformGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
            label: "Compute Pipeline Layout",
            bindGroupLayouts: [this.uniformGroup.bindGroupLayout!]
        });
        const computePipelineDescriptor: GPUComputePipelineDescriptor = {
            layout: this.uniformGroup.pipelineBindGroupLayout,
            label: "Compute Pipeline",
            compute: {
                module: this.shaderComputeModule!,
                entryPoint: 'main'
            }
        }
        this.pipeline = gpuDevice.createComputePipeline(computePipelineDescriptor);

        this.initialized = true;
    }

    public getBindGroup(gpuDevice: GPUDevice) {
        this.uniformGroup.getBindGroup(gpuDevice);
        return this.uniformGroup.bindGroup!;
    }
}

export { Compute }