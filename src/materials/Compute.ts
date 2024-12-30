import { parseIncludes } from "./shaders/ShaderUtils";
import { BindGroupDescriptor, BindableGroup } from "./BindableGroup";

/**
 * Represents a compute shader and its associated pipeline and resources.
 */
class Compute {

    /** The shader module for the compute shader. */
    public shaderComputeModule?: GPUShaderModule;

    /** The compute pipeline associated with the shader. */
    public pipeline?: GPUComputePipeline;

    /** The layout of the bind group for the compute shader. */
    public bindGroupLayout?: GPUBindGroupLayout;

    /** The bind group for the compute shader. */
    public bindGroup?: GPUBindGroup;

    /** Indicates whether the compute shader has been initialized. */
    public initialized: boolean = false;

    /** A unique identifier for the compute shader instance. */
    public uuid: string;

    /** The uniform group associated with the compute shader. */
    private bindableGroup: BindableGroup;

    /**
     * Constructs a new Compute instance.
     * 
     * @param shaderCode - The GLSL or WGSL code for the compute shader.
     * @param bindings - The descriptor for the bind group bindings.
     */
    constructor(
        private shaderCode: string,
        public bindings: BindGroupDescriptor[]
    ) {
        this.bindableGroup = new BindableGroup(bindings);
        this.uuid = crypto.randomUUID();
    }

    /**
     * Creates the shader module from the provided shader code.
     * 
     * @param gpuDevice - The GPU device used to create the shader module.
     */
    private createShaderModule(gpuDevice: GPUDevice) {
        this.shaderComputeModule = gpuDevice.createShaderModule({
            code: parseIncludes(this.shaderCode)
        });
    }

    /**
     * Initializes the compute pipeline and associated resources.
     * 
     * @param gpuDevice - The GPU device used to initialize the pipeline.
     */
    public initialize(gpuDevice: GPUDevice) {
        if (!this.shaderComputeModule) {
            this.createShaderModule(gpuDevice);
        }

        this.bindableGroup.createBindGroupLayout(gpuDevice);
        this.bindableGroup.pipelineBindGroupLayout = gpuDevice.createPipelineLayout({
            label: "Compute Pipeline Layout",
            bindGroupLayouts: [this.bindableGroup.bindGroupLayout!]
        });
        const computePipelineDescriptor: GPUComputePipelineDescriptor = {
            layout: this.bindableGroup.pipelineBindGroupLayout,
            label: "Compute Pipeline",
            compute: {
                module: this.shaderComputeModule!,
                entryPoint: 'main'
            }
        }
        this.pipeline = gpuDevice.createComputePipeline(computePipelineDescriptor);

        this.initialized = true;
    }

    /**
     * Retrieves the bind group for the compute shader.
     * 
     * @param gpuDevice - The GPU device used to get the bind group.
     * @returns The bind group for the compute shader.
     */
    public getBindGroup(gpuDevice: GPUDevice) {
        this.bindableGroup.getBindGroup(gpuDevice);
        return this.bindableGroup.bindGroup!;
    }
}

export { Compute }
