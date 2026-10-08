import type { Vec3 } from './VoxelVolume';

/** The most levels a `VoxelClipmap` has. Rust: `gi::MAX_CLIPMAP_LEVELS`. */
export const MAX_CLIPMAP_LEVELS = 6;

/** Bytes of the WGSL `VoxelClipmap` (clipmap.wgsl; Rust `VoxelClipmapGpu`). */
export const VOXEL_CLIPMAP_BYTES = 32 + 16 * MAX_CLIPMAP_LEVELS;

/** Rust's `f32::round`: halves away from zero. */
function roundHalfAway(x: number): number {
    return Math.sign(x) * Math.round(Math.abs(x));
}

/**
 * Where a voxel clipmap's levels lie: `levels` boxes of `dims` voxels each, centred on a point that
 * moves (the camera), the finest of voxels `voxelSize` metres wide and each next one twice as
 * coarse, so each covers twice the extent of the one before. A level's voxels are cells of a fixed
 * world lattice (voxel `c` spans `c * size .. (c + 1) * size`), and a level keeps a window of
 * `dims` of them, stored toroidally: voxel `c` in texel `c mod dims`. Moving the window moves its
 * origin by whole voxels and rewrites only the slabs that came into it. Rust: `gi::ClipmapLayout`.
 */
export class ClipmapLayout {
    public readonly levels: number;
    /** Voxels of each level, multiples of 8. */
    public readonly dims: Vec3;
    /** The finest level's voxels, metres. */
    public readonly voxelSize: number;

    /**
     * `levels` (at most `MAX_CLIPMAP_LEVELS`) of `dims` voxels (each rounded up to a multiple of
     * 8), the finest `voxelSize` metres.
     */
    constructor(levels: number, dims: Vec3, voxelSize: number) {
        this.levels = Math.min(Math.max(Math.floor(levels), 1), MAX_CLIPMAP_LEVELS);
        this.dims = dims.map((d) => Math.ceil(Math.max(d, 8) / 8) * 8) as Vec3;
        this.voxelSize = Math.max(voxelSize, 1e-4);
    }

    /** Level `level`'s voxels, metres. */
    public levelVoxelSize(level: number): number {
        return this.voxelSize * 2 ** level;
    }

    /** Level `level`'s extent, metres. */
    public levelExtent(level: number): Vec3 {
        const size = this.levelVoxelSize(level);
        return this.dims.map((d) => d * size) as Vec3;
    }

    /** Voxels of one level. */
    public voxelCount(): number {
        return this.dims[0] * this.dims[1] * this.dims[2];
    }

    /** The radiance of every level, and the scratch level the injection writes into. */
    public radianceBytes(): number {
        return (this.levels + 1) * this.voxelCount() * 8;
    }

    /**
     * The origin (first voxel) of level `level`'s window centred on `eye`, a multiple of `snap`
     * voxels.
     */
    public centredOrigin(level: number, eye: Vec3, snap: number): Vec3 {
        const s = Math.max(Math.floor(snap), 1);
        const size = this.levelVoxelSize(level);
        return [0, 1, 2].map((a) => {
            const cell = Math.floor(eye[a] / size);
            // the snapped voxel nearest the eye's
            return roundHalfAway(cell / s) * s - Math.trunc(this.dims[a] / 2);
        }) as Vec3;
    }

    /**
     * Where level `level`'s window at `origin` goes for an eye at `eye`: it stays while the eye is
     * less than `snap` voxels from its centre along each axis, and otherwise moves by whole
     * multiples of `snap` toward the eye (so an eye moving to and fro across a step does not move
     * it back and forth).
     */
    public follow(level: number, origin: Vec3, eye: Vec3, snap: number): Vec3 {
        const s = Math.max(Math.floor(snap), 1);
        const size = this.levelVoxelSize(level);
        return [0, 1, 2].map((a) => {
            const centre = origin[a] + this.dims[a] * 0.5;
            // whole steps toward the eye, rounding toward zero (`+ 0`: no -0)
            return origin[a] + Math.trunc((eye[a] / size - centre) / s) * s + 0;
        }) as Vec3;
    }

    /**
     * The level whose voxels a footprint `diameter` metres wide reads, before the levels that
     * don't hold the point: `log2(diameter / voxelSize)`, clamped to the levels.
     */
    public levelFor(diameter: number): number {
        return Math.min(Math.floor(Math.log2(Math.max(diameter / this.voxelSize, 1))), this.levels - 1);
    }
}

/**
 * A voxel clipmap of the scene's light (the outdoor counterpart of `VoxelVolume`, which covers one
 * fixed box): `ClipmapLayout.levels` nested windows around a moving point, each its own
 * `rgba16float` 3D texture (premultiplied radiance in rgb, opacity across one voxel in a, as in
 * `VoxelVolume`), stored toroidally and sampled with a repeating sampler, so a world position maps
 * to its texel without an offset (`CLIPMAP_WGSL`'s `clipSample`). The levels stand in for a
 * volume's mips: a cone reads the level whose voxels are as wide as it is, or the next coarser one
 * that holds the point.
 *
 * Each level's window has an origin once it holds valid data (`origin`); readers ignore levels
 * without one. A producer (`SceneVoxelClipmap`) writes a level's texture through a scratch level
 * (`scratchView`), copied over it, so a pass that writes a level can read every level.
 * Rust: `gi::VoxelClipmap`.
 */
export class VoxelClipmap {
    /** The `VoxelClipmap` uniform. */
    public readonly uniform: GPUBuffer;
    /** Trilinear, repeating: world positions wrap onto the toroidal levels. */
    public readonly sampler: GPUSampler;
    /** The scratch level a producer writes, then copies into a level (`copyScratchTo`). */
    public readonly scratchView: GPUTextureView;

    private readonly textures: GPUTexture[];
    private readonly views: GPUTextureView[];
    private readonly scratch: GPUTexture;
    /** Bound for the levels a layout lacks. */
    private readonly missing: GPUTexture;
    private readonly missingView: GPUTextureView;
    /** Per level: its window's origin, or null before it holds data. */
    private readonly origins: (Vec3 | null)[] = Array.from({ length: MAX_CLIPMAP_LEVELS }, () => null);
    private _radianceScale: number;
    /** The uniform as last written (`upload`). */
    private written: Uint8Array | null = null;

    constructor(private readonly device: GPUDevice, public readonly layout: ClipmapLayout, radianceScale: number = 1) {
        this._radianceScale = Math.max(radianceScale, 1e-6);
        const texture = (label: string, size: Vec3, usage: GPUTextureUsageFlags) => device.createTexture({
            label,
            size,
            dimension: '3d',
            format: 'rgba16float',
            usage,
        });
        const usage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC;
        this.textures = Array.from({ length: layout.levels }, () => texture('VoxelClipmap/Level', layout.dims, usage));
        this.views = this.textures.map((t) => t.createView());
        this.scratch = texture('VoxelClipmap/Scratch', layout.dims, GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC);
        this.scratchView = this.scratch.createView();
        this.missing = texture('VoxelClipmap/NoLevel', [1, 1, 1], GPUTextureUsage.TEXTURE_BINDING);
        this.missingView = this.missing.createView();
        this.uniform = device.createBuffer({ label: 'VoxelClipmap/Params', size: VOXEL_CLIPMAP_BYTES, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        this.sampler = device.createSampler({
            label: 'VoxelClipmap/LinearRepeat',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
            addressModeW: 'repeat',
            magFilter: 'linear',
            minFilter: 'linear',
        });
    }

    /** Level `level`'s origin (its window's first voxel, in its lattice), once it holds data. */
    public origin(level: number): Vec3 | null {
        return this.origins[level] ?? null;
    }

    /** Set level `level`'s origin (null: no valid data). Written to the uniform by `upload`. */
    public setOrigin(level: number, origin: Vec3 | null): void {
        this.origins[level] = origin ? [...origin] as Vec3 : null;
    }

    /** World box of level `level`'s window, if it holds data. */
    public levelBounds(level: number): [Vec3, Vec3] | null {
        const o = this.origin(level);
        if (!o) return null;
        const size = this.layout.levelVoxelSize(level);
        return [o.map((v) => v * size) as Vec3, o.map((v, a) => (v + this.layout.dims[a]) * size) as Vec3];
    }

    public get radianceScale(): number {
        return this._radianceScale;
    }

    /** Change the reference radiance is stored against (as `VoxelVolume.setRadianceScale`). */
    public set radianceScale(scale: number) {
        this._radianceScale = Math.max(scale, 1e-6);
    }

    /** Write the uniform if it changed (once a frame, after the origins are set). */
    public upload(): void {
        const data = new ArrayBuffer(VOXEL_CLIPMAP_BYTES);
        const u32 = new Uint32Array(data);
        const i32 = new Int32Array(data);
        const f32 = new Float32Array(data);
        u32.set(this.layout.dims, 0);
        u32[3] = this.layout.levels;
        f32[4] = this.layout.voxelSize;
        f32[5] = this._radianceScale;
        this.origins.forEach((o, k) => {
            if (!o) return;
            i32.set(o, 8 + 4 * k);
            u32[11 + 4 * k] = 1;
        });
        const bytes = new Uint8Array(data);
        if (this.written && this.written.every((b, i) => b === bytes[i])) return;
        this.device.queue.writeBuffer(this.uniform, 0, data);
        this.written = bytes;
    }

    /** Level `level`'s texture. */
    public texture(level: number): GPUTexture {
        return this.textures[level];
    }

    /** Level `level`'s view (sampled, or written as storage by a producer). */
    public view(level: number): GPUTextureView {
        return this.views[level];
    }

    /**
     * The `MAX_CLIPMAP_LEVELS` views `CLIPMAP_WGSL` binds (a 1-texel stand-in past the layout's
     * levels).
     */
    public levelViews(): GPUTextureView[] {
        return Array.from({ length: MAX_CLIPMAP_LEVELS }, (_, k) => this.views[k] ?? this.missingView);
    }

    /** Record copying the scratch level over level `level`. */
    public copyScratchTo(encoder: GPUCommandEncoder, level: number): void {
        encoder.copyTextureToTexture({ texture: this.scratch }, { texture: this.textures[level] }, this.layout.dims);
    }

    /** The radiance of every level and the scratch level. */
    public memoryBytes(): number {
        return this.layout.radianceBytes();
    }

    public destroy(): void {
        for (const t of [...this.textures, this.scratch, this.missing]) t.destroy();
        this.uniform.destroy();
    }
}

/**
 * Bind group layout entries for `CLIPMAP_WGSL`'s group 0 bindings 50-57 (the uniform, the levels,
 * the sampler), visible to `visibility`. Rust: `gi::clipmap::clipmap_layout_entries`.
 */
export function clipmapLayoutEntries(visibility: GPUShaderStageFlags = GPUShaderStage.COMPUTE): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [{ binding: 50, visibility, buffer: { type: 'uniform' } }];
    for (let k = 0; k < MAX_CLIPMAP_LEVELS; k++) entries.push({ binding: 51 + k, visibility, texture: { sampleType: 'float', viewDimension: '3d' } });
    entries.push({ binding: 57, visibility, sampler: { type: 'filtering' } });
    return entries;
}

/** The entries binding `clipmap` at `clipmapLayoutEntries`' bindings. Rust: `gi::clipmap::clipmap_entries`. */
export function clipmapEntries(clipmap: VoxelClipmap): GPUBindGroupEntry[] {
    return [
        { binding: 50, resource: { buffer: clipmap.uniform } },
        ...clipmap.levelViews().map((view, k) => ({ binding: 51 + k, resource: view })),
        { binding: 57, resource: clipmap.sampler },
    ];
}
