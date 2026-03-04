export interface FroxelGridOptions {
    gridW?: number;  // default 160
    gridH?: number;  // default 90
    gridD?: number;  // default 64
    near?: number;   // default 0.1
    far?: number;    // default 1000
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

        this._createTextures();

        // Grid params uniform (for accumulation shader)
        this._gridParamsBuffer = device.createBuffer({
            label: 'FroxelGrid/Params',
            size: 32, // near(4) + far(4) + gridW(4) + gridH(4) + gridD(4) + pad(12) = 32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._uploadGridParams();
        this._createAccumPipeline();
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

    /**
     * Front-to-back accumulation pass.
     * Call after the injection pass has written to scatterExtinctionTex.
     */
    accumulate(encoder: GPUCommandEncoder): void {
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/Accumulate' });
        pass.setPipeline(this._accumPipeline!);
        pass.setBindGroup(0, this._accumBG!);
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
    }

    destroy(): void {
        this._scatterExtinctionTex?.destroy();
        this._accumTex?.destroy();
        this._gridParamsBuffer?.destroy();
    }
}

export { FroxelGrid };
