import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Geometry } from "../buffers/Geometry";

/**
 * Represents a geometry that can be instanced multiple times.
 * Extends the base Geometry class to support instancing.
 */
class InstancedGeometry extends Geometry {
    /** Indicates that this is an instanced geometry. */
    public isInstancedGeometry: boolean = true;

    /**
     * Constructs an InstancedGeometry object.
     * 
     * @param geometry - The base geometry to be instanced.
     * @param instanceCount - The number of instances to create. Defaults to 1.
     * @param extraBuffers - Additional compute buffers for instancing.
     */
    constructor(
        public geometry: Geometry,
        public instanceCount: number = 1,
        public extraBuffers: ComputeBuffer[] = []
    ) {
        super();
        this.vertexCount = this.geometry.vertexCount;
        this.vertices = this.geometry.vertices;
        this.indices = this.geometry.indices;
    }

    /**
     * Initializes the instanced geometry with the given GPU device.
     * 
     * @param gpuDevice - The GPU device used for initialization.
     */
    public initialize(gpuDevice: GPUDevice) {
        super.initialize(gpuDevice);
        for (const extraBuffer of this.extraBuffers) {
            const attrs: GPUVertexAttribute[] = extraBuffer.attributes
                ? extraBuffer.attributes.map(a => ({
                    shaderLocation: a.shaderLocation as GPUIndex32,
                    offset: a.offset as GPUSize64,
                    format: a.format as GPUVertexFormat,
                }))
                : [{
                    shaderLocation: extraBuffer.shaderLocation! as GPUIndex32,
                    offset: (extraBuffer.offset ?? 0) as GPUSize64,
                    format: extraBuffer.format! as GPUVertexFormat,
                }];

            (this.vertexBuffersDescriptors! as Array<GPUVertexBufferLayout>).push({
                attributes: attrs,
                arrayStride: extraBuffer.stride!,
                stepMode: "instance" as GPUVertexStepMode,
            });
        }

        this.initialized = true;
    }
}

export { InstancedGeometry };
