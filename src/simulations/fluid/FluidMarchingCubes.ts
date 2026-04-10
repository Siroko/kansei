import { shaderCode as resetShader } from './shaders/marching-cubes-reset.wgsl';
import { shaderCode as extractShader } from './shaders/marching-cubes-extract.wgsl';
import { shaderCode as finalizeShader } from './shaders/marching-cubes-finalize.wgsl';
import { FluidDensityField } from './FluidDensityField';

export interface MarchingCubesOptions {
    maxTriangles?: number;
    isoLevel?: number;
}

export class FluidMarchingCubes {
    private _device: GPUDevice;
    private _maxTriangles: number;
    private _isoLevel: number;
    private _sampler: GPUSampler;
    private _paramsBuffer: GPUBuffer;
    private _triangleCounterBuffer: GPUBuffer;
    private _vertexBuffer: GPUBuffer;
    private _indexBuffer: GPUBuffer;
    private _indirectArgsBuffer: GPUBuffer;
    private _resetPipeline: GPUComputePipeline;
    private _extractPipeline: GPUComputePipeline;
    private _finalizePipeline: GPUComputePipeline;
    private _resetBG: GPUBindGroup;
    private _finalizeBG: GPUBindGroup;
    private _extractBGL: GPUBindGroupLayout;

    constructor(device: GPUDevice, options?: MarchingCubesOptions) {
        this._device = device;
        this._maxTriangles = Math.max(options?.maxTriangles ?? 262144, 1);
        this._isoLevel = options?.isoLevel ?? 1.0;

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
            size: maxVertices * 32,
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
        this._sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        this._resetPipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/ResetPipeline',
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: resetShader }), entryPoint: 'main' },
        });
        this._extractPipeline = device.createComputePipeline({
            label: 'FluidMarchingCubes/ExtractPipeline',
            layout: 'auto',
            compute: { module: device.createShaderModule({ code: extractShader }), entryPoint: 'main' },
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
        this._extractBGL = this._extractPipeline.getBindGroupLayout(0);
    }

    get vertexBuffer(): GPUBuffer { return this._vertexBuffer; }
    get indexBuffer(): GPUBuffer { return this._indexBuffer; }
    get indirectArgsBuffer(): GPUBuffer { return this._indirectArgsBuffer; }
    get triangleCounterBuffer(): GPUBuffer { return this._triangleCounterBuffer; }
    get maxTriangles(): number { return this._maxTriangles; }
    get isoLevel(): number { return this._isoLevel; }

    set isoLevel(v: number) {
        this._isoLevel = v;
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
            ],
        });
    }

    update(commandEncoder: GPUCommandEncoder, extractBindGroup: GPUBindGroup, densityField: FluidDensityField): void {
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
        extractPass.setPipeline(this._extractPipeline);
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
