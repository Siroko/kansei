import { gpuPass } from '../profiling/Profiler';
import { assemble } from '../materials/shaders/ShaderUtils';
import froxelCommon from '../../rust/kansei-core/src/shaders/froxel_common.wgsl?raw';
import froxelAccumulate from '../../rust/kansei-core/src/shaders/froxel_accumulate.wgsl?raw';
import froxelTemporal from '../../rust/kansei-core/src/shaders/froxel_temporal.wgsl?raw';

/** Bytes of the WGSL `TemporalParams` (`froxel_temporal.wgsl`). */
const TEMPORAL_PARAMS_BYTES = 176;

export interface FroxelGridOptions {
    gridW?: number;  // default 160
    gridH?: number;  // default 90
    gridD?: number;  // default 64
    near?: number;   // default 0.1
    far?: number;    // default 1000
    temporal?: boolean;     // default false
    blendFactor?: number;   // default 0.05
}

class FroxelGrid {
    private _device: GPUDevice;
    private _gridW: number;
    private _gridH: number;
    private _gridD: number;
    private _near: number;
    private _far: number;

    private _scatterExtinctionTex!: GPUTexture;
    private _accumTex!: GPUTexture;

    private _accumPipeline: GPUComputePipeline | null = null;
    private _accumBG: GPUBindGroup | null = null;
    private _gridParamsBuffer: GPUBuffer;

    // Temporal reprojection state
    private _temporal: boolean;
    private _blendFactor: number;
    private _historyTex!: [GPUTexture, GPUTexture];
    private _temporalSampler: GPUSampler | null = null;
    private _temporalPipeline: GPUComputePipeline | null = null;
    private _temporalBG!: [GPUBindGroup, GPUBindGroup];
    private _temporalParamsBuffer: GPUBuffer | null = null;
    private _accumBGTemporal!: [GPUBindGroup, GPUBindGroup];
    private _prevVP = new Float32Array(16);
    private _frameIdx = 0;
    private _hasPrevFrame = false;

    /**
     * WGSL helpers for exponential depth slicing and froxel <-> world conversion, in the camera's
     * [0,1] depth convention (`sliceDepth`, `depthToSlice`, `linearToNdcDepth`,
     * `ndcToLinearDepth`, `froxelToWorld`). Prepend to consumer shaders that read the grid. Rust:
     * `froxels::FROXEL_WGSL_HELPERS` (`froxel_common.wgsl`).
     */
    static readonly WGSL_HELPERS: string = froxelCommon;

    constructor(device: GPUDevice, options?: FroxelGridOptions) {
        this._device = device;
        this._gridW = options?.gridW ?? 160;
        this._gridH = options?.gridH ?? 90;
        this._gridD = options?.gridD ?? 64;
        this._near  = options?.near ?? 0.1;
        this._far   = options?.far ?? 1000;
        this._temporal = options?.temporal ?? false;
        this._blendFactor = options?.blendFactor ?? 0.05;

        this._createTextures();

        // Grid params uniform (for accumulation shader)
        this._gridParamsBuffer = device.createBuffer({
            label: 'FroxelGrid/Params',
            size: 32, // near(4) + far(4) + gridW(4) + gridH(4) + gridD(4) + pad(12) = 32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._uploadGridParams();
        this._createAccumPipeline();

        if (this._temporal) {
            this._createTemporalResources();
        }
    }

    get scatterExtinctionTex(): GPUTexture { return this._scatterExtinctionTex; }
    get accumTex(): GPUTexture { return this._accumTex; }
    get gridW(): number { return this._gridW; }
    get gridH(): number { return this._gridH; }
    get gridD(): number { return this._gridD; }
    get near(): number { return this._near; }
    get far(): number { return this._far; }
    /** Whether injected froxels blend with reprojected history (`temporal`). */
    get isTemporal(): boolean { return this._temporal; }
    /** Weight of the current frame when `temporal` is on. */
    get blendFactor(): number { return this._blendFactor; }

    /**
     * Forget the temporal history, so the next frame uses only its own injection. Call on a
     * camera cut, or reprojection smears the previous shot's fog into the new one.
     */
    resetHistory(): void {
        this._hasPrevFrame = false;
    }

    private _createTextures(): void {
        const texUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const size: GPUExtent3D = [this._gridW, this._gridH, this._gridD];

        this._scatterExtinctionTex = this._device.createTexture({
            label: 'FroxelGrid/ScatterExtinction',
            size,
            dimension: '3d',
            format: 'rgba16float',
            usage: texUsage,
        });

        this._accumTex = this._device.createTexture({
            label: 'FroxelGrid/Accum',
            size,
            dimension: '3d',
            format: 'rgba16float',
            usage: texUsage,
        });
    }

    private _uploadGridParams(): void {
        const data = new Float32Array(8); // 32 bytes
        data[0] = this._near;
        data[1] = this._far;
        new Uint32Array(data.buffer, 8, 1)[0] = this._gridW;
        new Uint32Array(data.buffer, 12, 1)[0] = this._gridH;
        new Uint32Array(data.buffer, 16, 1)[0] = this._gridD;
        // [5..7] padding
        this._device.queue.writeBuffer(this._gridParamsBuffer, 0, data.buffer as ArrayBuffer);
    }

    private _createAccumPipeline(): void {
        const shaderCode = assemble([froxelCommon, froxelAccumulate]);

        const module = this._device.createShaderModule({
            label: 'FroxelGrid/AccumShader',
            code: shaderCode,
        });

        const bgl = this._device.createBindGroupLayout({
            label: 'FroxelGrid/Accum BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._accumPipeline = this._device.createComputePipeline({
            label: 'FroxelGrid/AccumPipeline',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._accumBG = this._device.createBindGroup({
            label: 'FroxelGrid/Accum BG',
            layout: bgl,
            entries: [
                { binding: 0, resource: this._scatterExtinctionTex.createView() },
                { binding: 1, resource: this._accumTex.createView() },
                { binding: 2, resource: { buffer: this._gridParamsBuffer } },
            ],
        });
    }

    // ── Temporal blend shader ──────────────────────────────────────────────

    private static _TEMPORAL_SHADER = assemble([froxelCommon, froxelTemporal]);

    private _createTemporalResources(): void {
        const device = this._device;
        const size: GPUExtent3D = [this._gridW, this._gridH, this._gridD];
        const texUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;

        // Ping-pong history textures
        this._historyTex = [
            device.createTexture({
                label: 'FroxelGrid/HistoryTex0',
                size, dimension: '3d', format: 'rgba16float', usage: texUsage,
            }),
            device.createTexture({
                label: 'FroxelGrid/HistoryTex1',
                size, dimension: '3d', format: 'rgba16float', usage: texUsage,
            }),
        ];

        this._temporalSampler = device.createSampler({
            label: 'FroxelGrid/TemporalSampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        this._temporalParamsBuffer = device.createBuffer({
            label: 'FroxelGrid/TemporalParams',
            size: TEMPORAL_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Temporal blend pipeline
        const module = device.createShaderModule({
            label: 'FroxelGrid/TemporalShader',
            code: FroxelGrid._TEMPORAL_SHADER,
        });

        const bgl = device.createBindGroupLayout({
            label: 'FroxelGrid/Temporal BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._temporalPipeline = device.createComputePipeline({
            label: 'FroxelGrid/TemporalPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        // Ping-pong bind groups: [0] reads history0, writes history1; [1] reads history1, writes history0
        this._temporalBG = [
            device.createBindGroup({
                label: 'FroxelGrid/Temporal BG 0',
                layout: bgl,
                entries: [
                    { binding: 0, resource: this._scatterExtinctionTex.createView() },
                    { binding: 1, resource: this._historyTex[0].createView() },
                    { binding: 2, resource: this._temporalSampler },
                    { binding: 3, resource: this._historyTex[1].createView() },
                    { binding: 4, resource: { buffer: this._temporalParamsBuffer } },
                ],
            }),
            device.createBindGroup({
                label: 'FroxelGrid/Temporal BG 1',
                layout: bgl,
                entries: [
                    { binding: 0, resource: this._scatterExtinctionTex.createView() },
                    { binding: 1, resource: this._historyTex[1].createView() },
                    { binding: 2, resource: this._temporalSampler },
                    { binding: 3, resource: this._historyTex[0].createView() },
                    { binding: 4, resource: { buffer: this._temporalParamsBuffer } },
                ],
            }),
        ];

        // Accumulation bind groups that read from history textures instead of scatterExtinctionTex
        const accumBGL = this._accumPipeline!.getBindGroupLayout(0);
        this._accumBGTemporal = [
            device.createBindGroup({
                label: 'FroxelGrid/Accum BG Temporal 0',
                layout: accumBGL,
                entries: [
                    { binding: 0, resource: this._historyTex[0].createView() },
                    { binding: 1, resource: this._accumTex.createView() },
                    { binding: 2, resource: { buffer: this._gridParamsBuffer } },
                ],
            }),
            device.createBindGroup({
                label: 'FroxelGrid/Accum BG Temporal 1',
                layout: accumBGL,
                entries: [
                    { binding: 0, resource: this._historyTex[1].createView() },
                    { binding: 1, resource: this._accumTex.createView() },
                    { binding: 2, resource: { buffer: this._gridParamsBuffer } },
                ],
            }),
        ];

        this._prevVP.fill(0);
        this._frameIdx = 0;
        this._hasPrevFrame = false;
    }

    /**
     * Temporal reprojection blend pass.
     * Call after injection and before accumulation.
     * No-op if temporal is disabled.
     */
    temporalBlend(
        encoder: GPUCommandEncoder,
        currentInvVP: Float32Array,
        currentVP: Float32Array,
        cameraNear: number,
        cameraFar: number
    ): void {
        if (!this._temporal) return;

        const device = this._device;
        const readIdx = this._frameIdx;
        const writeIdx = 1 - this._frameIdx;

        const buf = new ArrayBuffer(TEMPORAL_PARAMS_BYTES);
        const f32 = new Float32Array(buf);
        const u32 = new Uint32Array(buf);

        f32.set(currentInvVP, 0);       // currentInvVP: mat4x4f (0..15)
        f32.set(this._prevVP, 16);       // prevVP: mat4x4f (16..31)
        f32[32] = this._near;            // gridNear
        f32[33] = this._far;             // gridFar
        f32[34] = cameraNear;            // cameraNear
        f32[35] = cameraFar;             // cameraFar
        u32[36] = this._gridW;           // gridW
        u32[37] = this._gridH;           // gridH
        u32[38] = this._gridD;           // gridD
        f32[39] = this._blendFactor;     // blendFactor
        u32[40] = this._hasPrevFrame ? 1 : 0; // hasPrevFrame

        device.queue.writeBuffer(this._temporalParamsBuffer!, 0, buf);

        // Dispatch temporal blend
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/TemporalBlend', timestampWrites: gpuPass('FroxelGrid/TemporalBlend') });
        pass.setPipeline(this._temporalPipeline!);
        pass.setBindGroup(0, this._temporalBG[readIdx]);
        pass.dispatchWorkgroups(
            Math.ceil(this._gridW / 4),
            Math.ceil(this._gridH / 4),
            Math.ceil(this._gridD / 4)
        );
        pass.end();

        // Store current VP as previous for next frame
        this._prevVP.set(currentVP);
        this._hasPrevFrame = true;
        this._frameIdx = writeIdx;
    }

    /**
     * Front-to-back accumulation pass.
     * Call after the injection pass (and temporal blend if enabled).
     */
    accumulate(encoder: GPUCommandEncoder): void {
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/Accumulate', timestampWrites: gpuPass('FroxelGrid/Accumulate') });
        pass.setPipeline(this._accumPipeline!);

        if (this._temporal && this._hasPrevFrame) {
            // Read from the history texture that was just written
            // _frameIdx was already toggled, so the last write was to (1 - _frameIdx)
            const lastWriteIdx = 1 - this._frameIdx;
            pass.setBindGroup(0, this._accumBGTemporal[lastWriteIdx]);
        } else {
            pass.setBindGroup(0, this._accumBG!);
        }

        pass.dispatchWorkgroups(
            Math.ceil(this._gridW / 8),
            Math.ceil(this._gridH / 8)
        );
        pass.end();
    }

    resize(gridW: number, gridH: number, gridD: number): void {
        this._gridW = gridW;
        this._gridH = gridH;
        this._gridD = gridD;
        this._scatterExtinctionTex?.destroy();
        this._accumTex?.destroy();
        this._createTextures();
        this._uploadGridParams();
        // Rebuild accumulation bind group with new textures
        this._accumBG = null;
        this._createAccumPipeline();

        if (this._temporal) {
            this._historyTex[0]?.destroy();
            this._historyTex[1]?.destroy();
            this._createTemporalResources();
        }
    }

    destroy(): void {
        this._scatterExtinctionTex?.destroy();
        this._accumTex?.destroy();
        this._gridParamsBuffer?.destroy();
        if (this._temporal) {
            this._historyTex[0]?.destroy();
            this._historyTex[1]?.destroy();
            this._temporalParamsBuffer?.destroy();
        }
    }
}

export { FroxelGrid };
