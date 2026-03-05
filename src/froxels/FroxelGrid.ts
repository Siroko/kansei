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
     * WGSL helper functions for exponential depth slicing and coordinate conversion.
     * Inject into consumer shaders via string concatenation.
     */
    static readonly WGSL_HELPERS = /* wgsl */`
        fn sliceDepth(i: f32, near: f32, far: f32, numSlices: f32) -> f32 {
            return near * pow(far / near, i / numSlices);
        }

        fn depthToSlice(linearDepth: f32, near: f32, far: f32, numSlices: f32) -> f32 {
            return numSlices * log(linearDepth / near) / log(far / near);
        }

        fn froxelToWorld(coord: vec3f, invViewProj: mat4x4f, near: f32, far: f32, gridSize: vec3f) -> vec3f {
            let uv = (coord.xy + 0.5) / gridSize.xy;
            let linearD = sliceDepth(coord.z + 0.5, near, far, gridSize.z);
            // Reverse perspective: linear depth -> NDC Z (WebGPU [0,1] range)
            let ndcZ = far * (linearD - near) / (linearD * (far - near));
            let ndcX = uv.x * 2.0 - 1.0;
            let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
            let world = invViewProj * vec4f(ndcX, ndcY, ndcZ, 1.0);
            return world.xyz / world.w;
        }
    `;

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
        const shaderCode = FroxelGrid.WGSL_HELPERS + /* wgsl */`

            struct GridParams {
                near  : f32,
                far   : f32,
                gridW : u32,
                gridH : u32,
                gridD : u32,
                _pad0 : f32,
                _pad1 : f32,
                _pad2 : f32,
            }

            @group(0) @binding(0) var scatterExtTex : texture_3d<f32>;
            @group(0) @binding(1) var accumOut       : texture_storage_3d<rgba16float, write>;
            @group(0) @binding(2) var<uniform> gp    : GridParams;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid : vec3u) {
                let x = gid.x;
                let y = gid.y;
                if (x >= gp.gridW || y >= gp.gridH) { return; }

                var transmittance = 1.0;
                var accLight = vec3f(0.0);

                for (var z = 0u; z < gp.gridD; z++) {
                    let data = textureLoad(scatterExtTex, vec3u(x, y, z), 0);
                    let scatter    = data.rgb;
                    let extinction = data.a;

                    let d0 = sliceDepth(f32(z), gp.near, gp.far, f32(gp.gridD));
                    let d1 = sliceDepth(f32(z + 1u), gp.near, gp.far, f32(gp.gridD));
                    let thickness = d1 - d0;

                    let sliceT = exp(-extinction * thickness);

                    // Integrate in-scattered light over this slice
                    accLight += transmittance * scatter * (1.0 - sliceT) / max(extinction, 0.0001);
                    transmittance *= sliceT;

                    textureStore(accumOut, vec3u(x, y, z), vec4f(accLight, transmittance));
                }
            }
        `;

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

    private static _TEMPORAL_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct TemporalParams {
            currentInvVP : mat4x4f,
            prevVP       : mat4x4f,
            gridNear     : f32,
            gridFar      : f32,
            cameraNear   : f32,
            cameraFar    : f32,
            gridW        : u32,
            gridH        : u32,
            gridD        : u32,
            blendFactor  : f32,
            hasPrevFrame : u32,
        }

        @group(0) @binding(0) var currentTex  : texture_3d<f32>;
        @group(0) @binding(1) var historyIn   : texture_3d<f32>;
        @group(0) @binding(2) var historySamp  : sampler;
        @group(0) @binding(3) var historyOut  : texture_storage_3d<rgba16float, write>;
        @group(0) @binding(4) var<uniform> tp : TemporalParams;

        @compute @workgroup_size(4, 4, 4)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            if (gid.x >= tp.gridW || gid.y >= tp.gridH || gid.z >= tp.gridD) { return; }

            let current = textureLoad(currentTex, gid, 0);

            // Compute world position of this froxel
            let linearD = sliceDepth(f32(gid.z) + 0.5, tp.gridNear, tp.gridFar, f32(tp.gridD));
            let n = tp.cameraNear;
            let f = tp.cameraFar;
            let ndcZ = ((f + n) * linearD - 2.0 * f * n) / ((f - n) * linearD);
            let uv = (vec2f(f32(gid.x), f32(gid.y)) + 0.5) / vec2f(f32(tp.gridW), f32(tp.gridH));
            let ndcX = uv.x * 2.0 - 1.0;
            let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
            let world = tp.currentInvVP * vec4f(ndcX, ndcY, ndcZ, 1.0);
            let worldPos = world.xyz / world.w;

            // Reproject into previous frame's clip space
            let prevClip = tp.prevVP * vec4f(worldPos, 1.0);
            let prevNDC = prevClip.xyz / prevClip.w;

            // Convert to froxel UVW in previous frame
            let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 0.5 - prevNDC.y * 0.5);
            let prevLinearD = 2.0 * n * f / ((f + n) - prevNDC.z * (f - n));
            let prevSlice = depthToSlice(prevLinearD, tp.gridNear, tp.gridFar, f32(tp.gridD));
            let prevUVW = vec3f(prevUV, (prevSlice + 0.5) / f32(tp.gridD));

            // Bounds check
            let valid = all(prevUVW >= vec3f(0.0)) && all(prevUVW <= vec3f(1.0)) && tp.hasPrevFrame != 0u;

            // Sample history with trilinear filtering
            let history = textureSampleLevel(historyIn, historySamp, prevUVW, 0.0);

            // Exponential blend: 100% current when invalid/first frame
            let alpha = select(1.0, tp.blendFactor, valid);
            let result = mix(history, current, alpha);

            textureStore(historyOut, gid, result);
        }
    `;

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
            size: 256,
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

        // Upload temporal params (256 bytes)
        const buf = new ArrayBuffer(256);
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
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/TemporalBlend' });
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
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/Accumulate' });
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
