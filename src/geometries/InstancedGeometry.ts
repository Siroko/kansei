import { ComputeBuffer } from "../buffers/ComputeBuffer";
import { Geometry } from "../buffers/Geometry";

class InstancedGeometry extends Geometry {
    public isInstancedGeometry: boolean = true;
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

    public initialize(gpuDevice: GPUDevice) {
        super.initialize(gpuDevice);
        for (const extraBuffer of this.extraBuffers) {
            (this.vertexBuffersDescriptors! as Array<GPUVertexBufferLayout>).push(
                {
                    attributes: [
                        {
                            shaderLocation: extraBuffer.shaderLocation! as GPUIndex32,
                            offset: extraBuffer.offset! as GPUSize64,
                            format: extraBuffer.format! as GPUVertexFormat
                        }
                    ] as Iterable<GPUVertexAttribute>,
                    arrayStride: extraBuffer.stride!,
                    stepMode: "instance" as GPUVertexStepMode
                });
        }

        console.log(this.vertexBuffersDescriptors);
        this.initialized = true;
    }
}

export { InstancedGeometry };
