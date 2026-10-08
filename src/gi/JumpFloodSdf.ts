import { Texture } from '../buffers/Texture';
import { gpuPass } from '../profiling/Profiler';
import { JUMP_FLOOD_WGSL } from './GiWGSL';
import type { VolumeLayout } from './VoxelVolume';

/**
 * Where a `JumpFloodSdf`'s seeds come from. Rust: `gi::SdfSeeds`.
 * - `surfaces`: a mesh voxelizer's surface buffers (`MeshVoxelizer.staticSurfaces` and
 *   `dynamicSurfaces`, `SURFACE_WORDS_PER_VOXEL` u32 a voxel): a voxel holding a surface is a seed;
 * - `opacity`: a volume's mip 0 (`VoxelVolume.view`): a voxel at least `threshold` opaque is a
 *   seed (particles, analytic boxes).
 */
export type SdfSeeds =
    | { kind: 'surfaces' }
    | { kind: 'opacity', radiance: GPUTextureView, threshold: number };

/** Bytes of the WGSL `SdfParams` (jump_flood.wgsl; Rust `SdfParamsGpu`). */
export const SDF_PARAMS_BYTES = 32;

/** A seed field's two ping-pong textures and the passes that fill them (seed, then floods). */
interface Chain {
    textures: [GPUTexture, GPUTexture];
    views: [GPUTextureView, GPUTextureView];
    /** (entry index, bind group) per pass, in order. */
    passes: [number, GPUBindGroup][];
}

/**
 * A distance field over a voxel volume by jump flooding (miaumiau.cat/?p=1457's distance field,
 * on WebGPU compute; Rong and Tan 2006): the unsigned distance in metres from each voxel to the
 * nearest occupied one, in an `r32float` 3D texture over the same `VolumeLayout`, read with
 * `SDF_WGSL` (`sdfDistance`, `sdfSoftShadow`, `sdfSurfaceShadow`, `sdfAo`). Sampling it linearly
 * needs the device's `float32-filterable` (the renderer requests it by default).
 *
 * Seeds come from a mesh voxelizer's surfaces or a volume's opacity (`SdfSeeds`). With surfaces,
 * the static renderables' seeds flood only when they change (`encodeStatic`) and the dynamic
 * ones' every frame they exist (`encodeDynamic`); the distance pass takes the nearer of the two.
 * A flood is log2 of the volume's side passes plus two (JFA+2), 26 texel reads a voxel each.
 * Rust: `gi::JumpFloodSdf`.
 */
export class JumpFloodSdf {
    /** The field (metres), to bind as `texture_3d<f32>` and sample with the volume's sampler. */
    public readonly view: GPUTextureView;
    public readonly texture: GPUTexture;

    private readonly pipelines: GPUComputePipeline[];
    private readonly bgl: GPUBindGroupLayout;
    /**
     * One per pass (the flood steps differ; each pass reads its own, so they never share a buffer
     * a frame writes twice), written once.
     */
    private readonly params: GPUBuffer[];
    private readonly distanceParams: [GPUBuffer, GPUBuffer];
    private readonly steps: number[];
    private readonly staticChain: Chain;
    private dynamicChain: Chain | null = null;
    private readonly distanceStorage: GPUTextureView;
    /** The distance passes: static seeds only, and with dynamic seeds. */
    private distanceGroups: [GPUBindGroup | null, GPUBindGroup | null] = [null, null];
    private readonly dummyBuffer: GPUBuffer;
    private readonly dummyVolume: GPUTexture;

    /**
     * A field over `layout`. With `surfaces` seeds give the static surfaces now (`staticSurfaces`)
     * and the dynamic ones with `setDynamicSurfaces` when they appear.
     */
    constructor(private readonly device: GPUDevice, public readonly layout: VolumeLayout, seeds: SdfSeeds, staticSurfaces?: GPUBuffer) {
        const visibility = GPUShaderStage.COMPUTE;
        const uintTexture: GPUTextureBindingLayout = { sampleType: 'uint', viewDimension: '3d' };
        this.bgl = device.createBindGroupLayout({
            label: 'VoxelGI/SdfBGL',
            entries: [
                { binding: 0, visibility, buffer: { type: 'uniform' } },
                { binding: 1, visibility, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 3, visibility, texture: uintTexture },
                { binding: 4, visibility, storageTexture: { access: 'write-only', format: 'r32uint', viewDimension: '3d' } },
                { binding: 5, visibility, texture: uintTexture },
                { binding: 6, visibility, storageTexture: { access: 'write-only', format: 'r32float', viewDimension: '3d' } },
            ],
        });
        const module = device.createShaderModule({ label: 'VoxelGI/JumpFlood', code: JUMP_FLOOD_WGSL });
        const pipelineLayout = device.createPipelineLayout({ label: 'VoxelGI/JumpFlood', bindGroupLayouts: [this.bgl] });
        this.pipelines = ['seed', 'flood', 'distance'].map((entryPoint) =>
            device.createComputePipeline({ label: 'VoxelGI/JumpFlood', layout: pipelineLayout, compute: { module, entryPoint } }));

        const seedMode = seeds.kind === 'surfaces' ? 0 : 1;
        const threshold = seeds.kind === 'opacity' ? seeds.threshold : 0;
        const opacity = seeds.kind === 'opacity' ? seeds.radiance : null;
        // the flood's steps: half the largest side (a power of two) down to 1, then 2 and 1 again
        const side = nextPowerOfTwo(Math.max(...layout.dims, 1));
        this.steps = [];
        for (let s = Math.max(side / 2, 1); ; s /= 2) {
            this.steps.push(s);
            if (s <= 1) break;
        }
        this.steps.push(2, 1);
        const paramsWith = (step: number, hasDynamic: number) => {
            const data = new ArrayBuffer(SDF_PARAMS_BYTES);
            const u32 = new Uint32Array(data);
            const f32 = new Float32Array(data);
            u32.set(layout.dims, 0);
            u32[3] = step;
            f32[4] = layout.voxelSize;
            f32[5] = threshold;
            u32[6] = seedMode;
            u32[7] = hasDynamic;
            const buffer = device.createBuffer({ label: 'VoxelGI/SdfParams', size: SDF_PARAMS_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
            device.queue.writeBuffer(buffer, 0, data);
            return buffer;
        };
        // the seed pass's at 0, then one per flood step
        this.params = [0, ...this.steps].map((s) => paramsWith(s, 0));
        this.distanceParams = [paramsWith(0, 0), paramsWith(0, 1)];

        this.dummyBuffer = device.createBuffer({ label: 'VoxelGI/SdfNoSurfaces', size: 16, usage: GPUBufferUsage.STORAGE });
        this.dummyVolume = device.createTexture({
            label: 'VoxelGI/SdfNoVolume',
            size: [1, 1, 1],
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING,
        });
        this.texture = volumeTexture(device, layout, 'VoxelGI/Sdf', 'r32float');
        this.view = this.texture.createView();
        this.distanceStorage = this.texture.createView();
        this.staticChain = newChain(device, layout, 'VoxelGI/SdfStaticSeeds');
        this.staticChain.passes = this.chainPasses(this.staticChain, staticSurfaces ?? null, opacity);
        this.rebuildDistanceGroups();
    }

    /** The seed pass then the floods, ping-ponging between the chain's textures. */
    private chainPasses(chain: Chain, surfaces: GPUBuffer | null, opacity: GPUTextureView | null): [number, GPUBindGroup][] {
        const group = (params: GPUBuffer, read: number, write: number) => this.device.createBindGroup({
            label: 'VoxelGI/SdfPass',
            layout: this.bgl,
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: surfaces ?? this.dummyBuffer } },
                { binding: 2, resource: opacity ?? this.dummyVolume.createView() },
                { binding: 3, resource: chain.views[read] },
                { binding: 4, resource: chain.views[write] },
                { binding: 5, resource: chain.views[read] },
                { binding: 6, resource: this.distanceStorage },
            ],
        });
        // the seed pass writes 0 (it reads nothing; 1 stands in), each flood reads what the one
        // before wrote
        const passes: [number, GPUBindGroup][] = [[0, group(this.params[0], 1, 0)]];
        this.steps.forEach((_, k) => {
            const [read, write] = k % 2 === 0 ? [0, 1] : [1, 0];
            passes.push([1, group(this.params[k + 1], read, write)]);
        });
        return passes;
    }

    /** Which texture of a chain holds its result after its passes. */
    private resultIndex(): number {
        // the seed pass writes 0; flood k writes 1 when k is even
        return this.steps.length % 2 === 1 ? 1 : 0;
    }

    private rebuildDistanceGroups(): void {
        const result = this.resultIndex();
        const statics = this.staticChain.views;
        const group = (params: GPUBuffer, dynamic: GPUTextureView) => this.device.createBindGroup({
            label: 'VoxelGI/SdfDistance',
            layout: this.bgl,
            entries: [
                { binding: 0, resource: { buffer: params } },
                { binding: 1, resource: { buffer: this.dummyBuffer } },
                { binding: 2, resource: this.dummyVolume.createView() },
                { binding: 3, resource: statics[result] },
                // (unwritten: the other static texture)
                { binding: 4, resource: statics[1 - result] },
                { binding: 5, resource: dynamic },
                { binding: 6, resource: this.distanceStorage },
            ],
        });
        this.distanceGroups = [
            group(this.distanceParams[0], statics[result]),
            this.dynamicChain ? group(this.distanceParams[1], this.dynamicChain.views[result]) : null,
        ];
    }

    /**
     * Flood the dynamic renderables' seeds from `surfaces` (`MeshVoxelizer.dynamicSurfaces`),
     * making their textures the first time; null drops them.
     */
    public setDynamicSurfaces(surfaces: GPUBuffer | null): void {
        if (this.dynamicChain) for (const t of this.dynamicChain.textures) t.destroy();
        this.dynamicChain = null;
        if (surfaces) {
            const chain = newChain(this.device, this.layout, 'VoxelGI/SdfDynamicSeeds');
            chain.passes = this.chainPasses(chain, surfaces, null);
            this.dynamicChain = chain;
        }
        this.rebuildDistanceGroups();
    }

    public get hasDynamic(): boolean {
        return this.dynamicChain !== null;
    }

    private encodeChain(encoder: GPUCommandEncoder, chain: Chain, label: string): void {
        const [w, h, d] = this.layout.dims;
        const pass = encoder.beginComputePass({ label, timestampWrites: gpuPass(label) });
        for (const [entry, group] of chain.passes) {
            pass.setPipeline(this.pipelines[entry]);
            pass.setBindGroup(0, group);
            pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        }
        pass.end();
    }

    /** Record the static seeds' flood (when they changed; then `encodeDistance`). */
    public encodeStatic(encoder: GPUCommandEncoder): void {
        this.encodeChain(encoder, this.staticChain, 'VoxelGI/SdfStaticFlood');
    }

    /** Record the dynamic seeds' flood (every frame there are any; then `encodeDistance`). */
    public encodeDynamic(encoder: GPUCommandEncoder): void {
        if (this.dynamicChain) this.encodeChain(encoder, this.dynamicChain, 'VoxelGI/SdfDynamicFlood');
    }

    /** Record the distance pass: the nearer of the static and (if any) dynamic seeds. */
    public encodeDistance(encoder: GPUCommandEncoder): void {
        const group = this.distanceGroups[1] ?? this.distanceGroups[0];
        if (!group) return;
        const [w, h, d] = this.layout.dims;
        const pass = encoder.beginComputePass({ label: 'VoxelGI/SdfDistance', timestampWrites: gpuPass('VoxelGI/SdfDistance') });
        pass.setPipeline(this.pipelines[2]);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        pass.end();
    }

    /** Record everything for a volume seeded by its opacity (`opacity` seeds), every frame. */
    public encode(encoder: GPUCommandEncoder): void {
        this.encodeStatic(encoder);
        this.encodeDistance(encoder);
    }

    /**
     * The field as a `Texture` (shares the GPU texture), to attach to a material that reads it
     * with `SDF_WGSL`: bind with `BindingLayouts.texture3d()`.
     */
    public asTexture(): Texture {
        return Texture.fromView('VoxelGI/Sdf', this.texture, this.view, '3d');
    }

    /** Flood passes per rebuild (seed and distance aside). */
    public get floodPasses(): number {
        return this.steps.length;
    }

    /** Bytes on the GPU: the field and the seed textures. */
    public memoryBytes(): number {
        return this.layout.voxelCount() * 4 * (1 + 2 + (this.dynamicChain ? 2 : 0));
    }

    public destroy(): void {
        for (const t of [this.texture, ...this.staticChain.textures, ...(this.dynamicChain?.textures ?? []), this.dummyVolume]) t.destroy();
        for (const b of [...this.params, ...this.distanceParams, this.dummyBuffer]) b.destroy();
    }
}

function nextPowerOfTwo(n: number): number {
    let p = 1;
    while (p < n) p *= 2;
    return p;
}

/** A 3D texture over `layout`'s voxels, written by storage and read by sampling. */
function volumeTexture(device: GPUDevice, layout: VolumeLayout, label: string, format: GPUTextureFormat): GPUTexture {
    return device.createTexture({
        label,
        size: layout.dims,
        dimension: '3d',
        format,
        // (COPY_SRC: readable in checks)
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC,
    });
}

function newChain(device: GPUDevice, layout: VolumeLayout, label: string): Chain {
    const textures: [GPUTexture, GPUTexture] = [volumeTexture(device, layout, label, 'r32uint'), volumeTexture(device, layout, label, 'r32uint')];
    return { textures, views: [textures[0].createView(), textures[1].createView()], passes: [] };
}
