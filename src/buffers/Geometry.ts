class Geometry {
    public vertexBuffer?: GPUBuffer;
    public indexBuffer?: GPUBuffer;
    public vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null> = [];
    public initialized: boolean = false;

    public vertexCount: number = 0;

    // Interleaved vertices, normals, uvs
    protected vertices?: Float32Array;
    protected indices?: Uint16Array;

    constructor() {
    }

    public initialize(gpuDevice: GPUDevice) {
        this.vertexBuffer = gpuDevice.createBuffer({
            label: 'vertex buffer',
            size: this.vertices!.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Float32Array(this.vertexBuffer.getMappedRange()).set(this.vertices!);
        this.vertexBuffer.unmap();

        this.indexBuffer = gpuDevice.createBuffer({
            label: 'index buffer',
            size: this.indices!.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        new Uint16Array(this.indexBuffer.getMappedRange()).set(this.indices!);
        this.indexBuffer.unmap();

        this.vertexBuffersDescriptors = [{
            attributes: [
                {
                    shaderLocation: 0 as GPUIndex32,
                    offset: 0 as GPUSize64,
                    format: "float32x4" as GPUVertexFormat
                },
                {
                    shaderLocation: 1 as GPUIndex32,
                    offset: 4 * 4 as GPUSize64,
                    format: "float32x3" as GPUVertexFormat
                },
                {
                    shaderLocation: 2 as GPUIndex32,
                    offset: 4 * 4 + 4 * 3 as GPUSize64,
                    format: "float32x2" as GPUVertexFormat
                }
            ] as Iterable<GPUVertexAttribute>,
            arrayStride: 4 * 4 + 4 * 3 + 4 * 2 as GPUSize32,
            stepMode: "vertex" as GPUVertexStepMode
        },
        {
            attributes: [
                {
                    shaderLocation: 3 as GPUIndex32,
                    offset: 0 as GPUSize64,
                    format: "float32x4" as GPUVertexFormat
                }
            ] as Iterable<GPUVertexAttribute>,
            arrayStride: 4 * 4 as GPUSize32,
            stepMode: "instance" as GPUVertexStepMode
        }
        ] as Iterable<GPUVertexBufferLayout | null>;

        this.initialized = true;
    }
}

export { Geometry }
