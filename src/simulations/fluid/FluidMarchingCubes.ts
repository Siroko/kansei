import { Renderer } from '../../renderers/Renderer';
import { shaderCode as resetShader } from './shaders/marching-cubes-reset.wgsl';
import { shaderCode as extractShader } from './shaders/marching-cubes-extract.wgsl';
import { shaderCode as classicExtractShader } from './shaders/marching-cubes-classic.wgsl';
import { shaderCode as finalizeShader } from './shaders/marching-cubes-finalize.wgsl';
import { EDGE_TABLE, TRI_TABLE } from './marching-cubes-tables';
import { FluidDensityField } from './FluidDensityField';

export interface MarchingCubesOptions {
    maxTriangles?: number;
    isoLevel?: number;
    /** When true, use the classic Paul Bourke lookup-table marching cubes (smooth surfaces).
     *  When false, use the voxel-shell approach (axis-aligned faces). Default: false. */
    useClassic?: boolean;
}

/** Bytes per emitted vertex (matches engine standard Vertex layout: pos vec4 + normal vec3 + uv vec2). */
export const MC_VERTEX_STRIDE = 36;

export class FluidMarchingCubes {
    private _renderer: Renderer;
    private _device: GPUDevice;
    private _maxTriangles: number;
    private _isoLevel: number;
    private _useClassic: boolean;
    private _sampler: GPUSampler;
    private _paramsBuffer: GPUBuffer;
    private _triangleCounterBuffer: GPUBuffer;
    private _vertexBuffer: GPUBuffer;
    private _indexBuffer: GPUBuffer;
    private _indirectArgsBuffer: GPUBuffer;
    /** Paul Bourke edge-intersection bitmask table (256 × u32). Uploaded once. */
    private _edgeTableBuffer: GPUBuffer;
    /** Paul Bourke triangle lookup table (4096 × i32, sentinel -1). Uploaded once. */
    private _triTableBuffer: GPUBuffer;
    private _resetPipeline: GPUComputePipeline;
    private _extractPipeline: GPUComputePipeline;
    private _classicExtractPipeline: GPUComputePipeline;
    private _finalizePipeline: GPUComputePipeline;
    private _resetBG: GPUBindGroup;
    private _finalizeBG: GPUBindGroup;
    private _extractBGL: GPUBindGroupLayout;

    constructor(renderer: Renderer, options?: MarchingCubesOptions) {
        this._renderer = renderer;
        this._device = renderer.gpuDevice;
        const device = this._device;
        this._maxTriangles = Math.max(options?.maxTriangles ?? 262144, 1);
        this._isoLevel = options?.isoLevel ?? 1.0;
        this._useClassic = options?.useClassic ?? false;

        const maxVertices = this._maxTriangles * 3;
        const maxIndices = this._maxTriangles * 3;
        this._paramsBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/Params',
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._triangleCounterBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/TriangleCounter',
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        this._vertexBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/Vertices',
            size: maxVertices * MC_VERTEX_STRIDE,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_SRC,
        });
        this._indexBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/Indices',
            size: maxIndices * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDEX | GPUBufferUsage.COPY_SRC,
        });
        this._indirectArgsBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/IndirectArgs',
            size: 20,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });

        // Marching-cubes lookup tables — uploaded once as uniform buffers so
        // Chrome/Dawn routes the lookups through the GPU's constant cache
        // (much faster than storage memory for random-access tables). The
        // classic MC shader declares them as `array<vec4<u32>, 64>` /
        // `array<vec4<i32>, 1024>` so the 16-byte uniform-array stride
        // exactly matches the natural vec4 packing of the flat u32/i32
        // arrays — no CPU-side repacking needed.
        this._edgeTableBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/EdgeTable',
            size: EDGE_TABLE.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this._edgeTableBuffer, 0, EDGE_TABLE);
        this._triTableBuffer = device.createBuffer({
            label: 'FluidMarchingCubes/TriTable',
            size: TRI_TABLE.byteLength,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this._triTableBuffer, 0, TRI_TABLE);

        this._sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        // Shared explicit BGL for both extract pipelines (so the bind group is interchangeable).
        // Bindings 6 and 7 are classic-MC-only lookup tables; the voxel shell shader
        // simply ignores them, which is allowed by WGSL.
        this._extractBGL = device.createBindGroupLayout({
            label: 'FluidMarchingCubes/ExtractBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 6, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 7, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const extractLayout = device.createPipelineLayout({
            label: 'FluidMarchingCubes/ExtractLayout',
            bindGroupLayouts: [this._extractBGL],
        });

        this._resetPipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/ResetPipeline',
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: resetShader }), entryPoint: 'main' },
        });
        this._extractPipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/ExtractPipeline',
            layout: extractLayout,
            compute: { module: device.createShaderModule({ code: extractShader }), entryPoint: 'main' },
        });
        this._classicExtractPipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/ClassicExtractPipeline',
            layout: extractLayout,
            compute: { module: device.createShaderModule({ code: classicExtractShader }), entryPoint: 'main' },
        });
        this._finalizePipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/FinalizePipeline',
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: finalizeShader }), entryPoint: 'main' },
        });

        this._resetBG = device.createBindGroup({
            label: 'FluidMarchingCubes/ResetBG',
            layout: this._resetPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this._triangleCounterBuffer } },
                { binding: 1, resource: { buffer: this._indirectArgsBuffer } },
            ],
        });
        this._finalizeBG = device.createBindGroup({
            label: 'FluidMarchingCubes/FinalizeBG',
            layout: this._finalizePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this._triangleCounterBuffer } },
                { binding: 1, resource: { buffer: this._indirectArgsBuffer } },
                { binding: 2, resource: { buffer: this._paramsBuffer } },
            ],
        });
    }

    get vertexBuffer(): GPUBuffer { return this._vertexBuffer; }
    get indexBuffer(): GPUBuffer { return this._indexBuffer; }
    get indirectArgsBuffer(): GPUBuffer { return this._indirectArgsBuffer; }
    get triangleCounterBuffer(): GPUBuffer { return this._triangleCounterBuffer; }
    get maxTriangles(): number { return this._maxTriangles; }
    get isoLevel(): number { return this._isoLevel; }
    get useClassic(): boolean { return this._useClassic; }

    set isoLevel(v: number) { this._isoLevel = v; }
    set useClassic(v: boolean) { this._useClassic = v; }

    /** Zero the indirect-draw args so the mesh Renderable draws nothing
     *  until the next `update()` call repopulates them. */
    clear(): void {
        const zero = new Uint32Array([0, 1, 0, 0, 0]);
        this._device.queue.writeBuffer(this._indirectArgsBuffer, 0, zero);
    }

    createBindGroup(densityField: FluidDensityField): GPUBindGroup {
        return this._device.createBindGroup({
            label: 'FluidMarchingCubes/ExtractBG',
            layout: this._extractBGL,
            entries: [
                { binding: 0, resource: densityField.densityView },
                { binding: 1, resource: this._sampler },
                { binding: 2, resource: { buffer: this._paramsBuffer } },
                { binding: 3, resource: { buffer: this._vertexBuffer } },
                { binding: 4, resource: { buffer: this._indexBuffer } },
                { binding: 5, resource: { buffer: this._triangleCounterBuffer } },
                { binding: 6, resource: { buffer: this._edgeTableBuffer } },
                { binding: 7, resource: { buffer: this._triTableBuffer } },
            ],
        });
    }

    /**
     * Run reset → extract → finalize compute passes.
     * If `commandEncoder` is omitted, the engine creates and submits its own.
     */
    update(extractBindGroup: GPUBindGroup, densityField: FluidDensityField, commandEncoder?: GPUCommandEncoder): void {
        if (!commandEncoder) {
            const enc = this._renderer.createCommandEncoder('FluidMarchingCubes/Encoder');
            this._record(enc, extractBindGroup, densityField);
            this._renderer.submit([enc.finish()]);
            return;
        }
        this._record(commandEncoder, extractBindGroup, densityField);
    }

    private _record(commandEncoder: GPUCommandEncoder, extractBindGroup: GPUBindGroup, densityField: FluidDensityField): void {
        const dims = densityField.texDims;
        const bmin = densityField.boundsMin;
        const bmax = densityField.boundsMax;

        const paramsU32 = new Uint32Array(12);
        const paramsF32 = new Float32Array(paramsU32.buffer);
        paramsU32[0] = Math.max(dims[0], 1);
        paramsU32[1] = Math.max(dims[1], 1);
        paramsU32[2] = Math.max(dims[2], 1);
        paramsU32[3] = this._maxTriangles;
        paramsF32[4] = bmin[0];
        paramsF32[5] = bmin[1];
        paramsF32[6] = bmin[2];
        paramsF32[7] = this._isoLevel;
        paramsF32[8] = bmax[0];
        paramsF32[9] = bmax[1];
        paramsF32[10] = bmax[2];
        paramsF32[11] = 0.0;
        this._device.queue.writeBuffer(this._paramsBuffer, 0, paramsF32);

        const resetPass = commandEncoder.beginComputePass({ label: 'FluidMarchingCubes/Reset' });
        resetPass.setPipeline(this._resetPipeline);
        resetPass.setBindGroup(0, this._resetBG);
        resetPass.dispatchWorkgroups(1, 1, 1);
        resetPass.end();

        const extractPass = commandEncoder.beginComputePass({ label: 'FluidMarchingCubes/Extract' });
        const pipeline = this._useClassic ? this._classicExtractPipeline : this._extractPipeline;
        extractPass.setPipeline(pipeline);
        extractPass.setBindGroup(0, extractBindGroup);
        extractPass.dispatchWorkgroups(
            Math.ceil(Math.max(dims[0], 1) / 4),
            Math.ceil(Math.max(dims[1], 1) / 4),
            Math.ceil(Math.max(dims[2], 1) / 4),
        );
        extractPass.end();

        const finalizePass = commandEncoder.beginComputePass({ label: 'FluidMarchingCubes/Finalize' });
        finalizePass.setPipeline(this._finalizePipeline);
        finalizePass.setBindGroup(0, this._finalizeBG);
        finalizePass.dispatchWorkgroups(1, 1, 1);
        finalizePass.end();
    }
}
