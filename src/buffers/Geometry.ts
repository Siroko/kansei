/**
 * Represents a geometric mesh with vertex and index buffers for WebGPU rendering.
 */
class Geometry {
    /** Indicates if this geometry is used for instanced rendering */
    public isInstancedGeometry: boolean = false;

    /** WebGPU buffer containing vertex data */
    public vertexBuffer?: GPUBuffer;

    /** WebGPU buffer containing index data */
    public indexBuffer?: GPUBuffer;

    /** Format of the index buffer data */
    public indexFormat: GPUIndexFormat = "uint16";

    /** Collection of vertex buffer layout descriptors */
    public vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null> = [];

    /** Indicates if the geometry has been initialized with GPU buffers */
    public initialized: boolean = false;

    /** Number of vertices in the geometry */
    public vertexCount: number = 0;

    /** Raw vertex data containing interleaved positions, normals, and UVs */
    public vertices?: Float32Array;

    /** Raw index data for defining triangles */
    public indices?: Uint16Array;

    constructor() { }

    /**
     * Initializes the geometry by creating GPU buffers and setting up vertex layouts.
     * @param gpuDevice - The WebGPU device to create buffers on
     */
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

        (this.vertexBuffersDescriptors as Array<GPUVertexBufferLayout>).push({
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
        });

        this.initialized = true;
    }
}

export { Geometry }
