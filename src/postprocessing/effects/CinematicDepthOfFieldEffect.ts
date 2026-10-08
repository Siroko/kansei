import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { assemble } from '../../materials/shaders/ShaderUtils';
// The Rust engine's shaders, unchanged: `cargo test -p kansei-core` validates them with naga and
// checks DofParams, HighlightParams and Sprite against the sizes packed below.
import COMMON from '../../../rust/kansei-core/src/shaders/cinematic_dof_common.wgsl?raw';
import PREFILTER from '../../../rust/kansei-core/src/shaders/cinematic_dof_prefilter.wgsl?raw';
import TILES from '../../../rust/kansei-core/src/shaders/cinematic_dof_tiles.wgsl?raw';
import GATHER from '../../../rust/kansei-core/src/shaders/cinematic_dof_gather.wgsl?raw';
import POSTFILTER from '../../../rust/kansei-core/src/shaders/cinematic_dof_postfilter.wgsl?raw';
import DOWNSAMPLE from '../../../rust/kansei-core/src/shaders/cinematic_dof_downsample.wgsl?raw';
import HIGHLIGHTS from '../../../rust/kansei-core/src/shaders/cinematic_dof_highlights.wgsl?raw';
import SPRITES from '../../../rust/kansei-core/src/shaders/cinematic_dof_sprites.wgsl?raw';
import COMPOSITE from '../../../rust/kansei-core/src/shaders/cinematic_dof_composite.wgsl?raw';

/** Bytes per scattered highlight (the WGSL `Sprite`). */
const SPRITE_BYTES = 32;
/** The sprite bins' size in half-resolution pixels and the sprites each lists (the WGSL `BIN` and `BIN_CAPACITY`). */
const BIN = 16;
const BIN_CAPACITY = 64;
/** Half-resolution texels per tile side (the shaders' TILE). */
const TILE = 8;
/** Levels of the half-resolution chain the gather reads (the gather shader's LEVELS). */
const LEVELS = 3;
/** Levels of the background chain, which also fills what the near field hides: all of them, down to a single texel. */
const FILL_LEVELS = 16;
/** The WGSL `DofParams` and `HighlightParams`, bytes. */
const PARAMS_BYTES = 48;
const HIGHLIGHT_PARAMS_BYTES = 16;

const LAYER_FORMAT: GPUTextureFormat = 'rgba32float';

const DEG = Math.PI / 180;
const divCeil = (a: number, b: number) => Math.ceil(a / b);

export interface CameraLensOptions {
    /** Focal length, mm. `null` (the default) derives it from the camera's field of view and the filmback, so the blur always matches the picture being rendered. */
    focalLengthMm?: number | null;
    /** Aperture as an f-number (focal length / aperture diameter). Default 2.8 */
    fStop?: number;
    /** Distance to the plane in focus, metres of view depth. Default 10 */
    focusDistanceM?: number;
    /** Filmback width, mm; its full width spans the image width. Default 23.76 (Unreal's) */
    sensorWidthMm?: number;
    /** Aperture blades: 3 or more gives polygonal bokeh, fewer a round one. Default 0 */
    bladeCount?: number;
    /** Default 0 */
    bladeRotationDeg?: number;
}

/**
 * A physical camera lens, as Unreal's CineCamera: the circle of confusion follows from the
 * thin-lens equation.
 */
class CameraLens {
    focalLengthMm: number | null;
    fStop: number;
    focusDistanceM: number;
    sensorWidthMm: number;
    bladeCount: number;
    bladeRotationDeg: number;

    constructor(options: CameraLensOptions = {}) {
        this.focalLengthMm    = options.focalLengthMm    ?? null;
        this.fStop            = options.fStop            ?? 2.8;
        this.focusDistanceM   = options.focusDistanceM   ?? 10.0;
        this.sensorWidthMm    = options.sensorWidthMm    ?? 23.76;
        this.bladeCount       = options.bladeCount       ?? 0;
        this.bladeRotationDeg = options.bladeRotationDeg ?? 0.0;
    }

    /** Focal length for a horizontal field of view on this filmback, mm. */
    focalLengthForHfov(hfovRad: number): number {
        return 0.5 * this.sensorWidthMm / Math.max(Math.tan(0.5 * hfovRad), 1e-6);
    }

    /** The focal length in use with `camera` (its vertical fov in degrees and aspect), mm. */
    focalLength(camera: Camera): number {
        if (this.focalLengthMm !== null) return this.focalLengthMm;
        const hfov = 2.0 * Math.atan(Math.tan(camera.fov * DEG * 0.5) * camera.aspect);
        return this.focalLengthForHfov(hfov);
    }

    /**
     * CoC radius in pixels of a point at infinity, for an image `widthPx` wide: the thin-lens CoC
     * diameter A f (S2 - S1) / (S2 (S1 - f)) as S2 -> infinity, with A = f / N.
     */
    cocScale(focalLengthMm: number, widthPx: number): number {
        const f = Math.max(focalLengthMm, 1e-3);
        const s1 = Math.max(this.focusDistanceM * 1000.0, f * 1.001);
        const aperture = f / Math.max(this.fStop, 0.1);
        const diameterMm = aperture * f / (s1 - f);
        return 0.5 * diameterMm * widthPx / Math.max(this.sensorWidthMm, 1e-3);
    }

    /** Signed CoC radius in pixels of a point at `viewDepthM` (negative in front of the focus plane), as the shaders compute it before clamping. */
    cocRadiusPx(focalLengthMm: number, widthPx: number, viewDepthM: number): number {
        return this.cocScale(focalLengthMm, widthPx) * (1.0 - this.focusDistanceM / Math.max(viewDepthM, 1e-4));
    }
}

/** Which pixels scatter their light as bokeh sprites. */
export interface HighlightOptions {
    /** Default true */
    enabled: boolean;
    /** A half-resolution pixel scatters what exceeds this multiple of its neighbours' mean luminance. Default 3 */
    contrast: number;
    /** Only pixels blurred by more than this CoC radius (full-resolution pixels) scatter. Default 4 */
    minCocPx: number;
    /** Sprites per frame; brighter pixels beyond it are gathered as usual. Read once, when the effect initializes. Default 8192 */
    maxSprites: number;
}

export interface CinematicDepthOfFieldOptions {
    lens?: CameraLens | CameraLensOptions;
    /**
     * Largest CoC radius as a fraction of the image width (bigger blur is clamped to it), so the
     * picture looks the same at any resolution. Default 0.025, Unreal's
     * (`r.DOF.Kernel.MaxBackgroundRadius` and `MaxForegroundRadius`).
     */
    maxCocFraction?: number;
    /**
     * Largest CoC radius, full-resolution pixels, whatever the width: a ceiling for cost and
     * sampling density (the gather samples discs up to 96 px at full density). The tighter of this
     * and `maxCocFraction` applies. Default 96
     */
    maxCocPx?: number;
    /**
     * Gather samples per half-resolution pixel. Discs wider than 12 half-resolution pixels read a
     * coarser level of the half-resolution image, so this count holds their density too. Default 72
     */
    sampleCount?: number;
    /** Rotate the sample pattern every frame, for a TAA after the DoF to average away. Leave off (the default) without one. */
    temporalNoise?: boolean;
    /** Scatter bright highlights as crisp, aperture-shaped bokeh sprites instead of gathering them (a gather leaves small bright sources grainy). */
    highlights?: Partial<HighlightOptions>;
}

/** What the effect outputs: the image, or one of its layers for inspection. */
enum DofDebugView {
    None = 0,
    /** The background layer (in focus and behind), with what the near field hides filled in. */
    Background = 1,
    /** The near-field layer's colour. */
    Near = 2,
    /** The near-field layer's coverage: how much of the aperture it hides at each pixel. */
    NearAlpha = 3,
    /** The CoC: red in front of the focus plane, blue behind, green in focus. */
    Coc = 4,
}

/** A half-resolution layer: its chain of levels (sampled, all levels) and each level alone. */
interface Layer {
    raw: GPUTextureView;
    chain: GPUTextureView;
    levels: GPUTextureView[];
}

interface Targets {
    width: number;
    height: number;
    textures: GPUTexture[];
    near: Layer;
    far: Layer;
    tiles: GPUTextureView;
    tilesDilated: GPUTextureView;
    bg: GPUTextureView;
    fg: GPUTextureView;
    bgFiltered: GPUTextureView;
    fgFiltered: GPUTextureView;
    /** Per sprite bin: how many sprites reach it, and their indices. */
    binCount: GPUBuffer;
    binList: GPUBuffer;
    // the bind groups that read only the effect's own targets
    highlights: GPUBindGroup;
    nearDownsamples: GPUBindGroup[];
    farDownsamples: GPUBindGroup[];
    dilate: GPUBindGroup;
    gatherNear: GPUBindGroup;
    gatherFar: GPUBindGroup;
    postfilter: GPUBindGroup;
}

interface Pipelines {
    prefilter: GPUComputePipeline;
    highlights: GPUComputePipeline;
    downsampleNear: GPUComputePipeline;
    downsampleFar: GPUComputePipeline;
    dilate: GPUComputePipeline;
    gatherNear: GPUComputePipeline;
    gatherFar: GPUComputePipeline;
    postfilter: GPUComputePipeline;
    composite: GPUComputePipeline;
}

interface Layouts {
    prefilter: GPUBindGroupLayout;
    highlights: GPUBindGroupLayout;
    downsample: GPUBindGroupLayout;
    tiles: GPUBindGroupLayout;
    gather: GPUBindGroupLayout;
    postfilter: GPUBindGroupLayout;
    composite: GPUBindGroupLayout;
}

const source = (pass: string) => assemble([COMMON, pass]);
/** A pass that reads or writes the scattered highlights. */
const spriteSource = (pass: string) => assemble([COMMON, SPRITES, pass]);

/**
 * Physically based depth of field, a port of the Rust engine's `CinematicDepthOfFieldEffect`
 * running its shaders. The circle of confusion comes from the camera's focal length, f-stop,
 * focus distance and filmback (`CameraLens`, as Unreal's CineCamera), and the blur is gathered as
 * scattered bokeh at half resolution in two layers, as Unreal's DiaphragmDOF:
 * - the **near field** (in front of the focus plane) is gathered with its coverage and composited
 *   over everything behind it, so a blurred foreground spreads softly past its own silhouette, and
 *   a porous one (leaves, a fence) shows the background through it;
 * - the **background** (in focus and behind) never bleeds over a sharper surface in front of it,
 *   so there are no halos at depth edges; what the near field hides is filled from its chain;
 * - every 2x2 block is split between the layers by CoC before anything is averaged, so no pixel
 *   is mixed across depth and each keeps its energy: a small highlight becomes a large, dimmer
 *   disc in the aperture's shape (round, or polygonal with `bladeCount`), and bright ones are
 *   scattered as crisp sprites (`HighlightOptions`);
 * - in-focus pixels stay the full-resolution image.
 *
 * Focus, aperture and focal length are public and can change every frame (focus pulls). The
 * simpler `DepthOfFieldEffect` (a focus range and a blur in pixels) stays alongside it.
 */
class CinematicDepthOfFieldEffect extends PostProcessingEffect {
    lens: CameraLens;
    maxCocFraction: number;
    maxCocPx: number;
    sampleCount: number;
    temporalNoise: boolean;
    highlights: HighlightOptions;
    /** Output a layer instead of the image, for tuning and debugging. */
    debugView: DofDebugView = DofDebugView.None;

    private _frame = 0;
    private _device: GPUDevice | null = null;
    private _params: GPUBuffer | null = null;
    private _highlightParams: GPUBuffer | null = null;
    private _sprites: GPUBuffer | null = null;
    /** The extraction pass counts its sprites here. */
    private _spriteCount: GPUBuffer | null = null;
    private _maxSprites = 1;
    private _pipelines: Pipelines | null = null;
    private _layouts: Layouts | null = null;
    private _targets: Targets | null = null;

    // the bind groups that read the chain's textures, rebuilt when those change
    private _prefilter: GPUBindGroup | null = null;
    private _composite: GPUBindGroup | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    private readonly _paramsData = new ArrayBuffer(PARAMS_BYTES);
    private readonly _paramsF32 = new Float32Array(this._paramsData);
    private readonly _paramsU32 = new Uint32Array(this._paramsData);
    private readonly _highlightData = new ArrayBuffer(HIGHLIGHT_PARAMS_BYTES);
    private readonly _highlightF32 = new Float32Array(this._highlightData);
    private readonly _highlightU32 = new Uint32Array(this._highlightData);

    constructor(options: CinematicDepthOfFieldOptions = {}) {
        super();
        this.lens = options.lens instanceof CameraLens ? options.lens : new CameraLens(options.lens);
        this.maxCocFraction = options.maxCocFraction ?? 0.025;
        this.maxCocPx       = options.maxCocPx       ?? 96.0;
        this.sampleCount    = options.sampleCount    ?? 72;
        this.temporalNoise  = options.temporalNoise  ?? false;
        this.highlights = { enabled: true, contrast: 3.0, minCocPx: 4.0, maxSprites: 8192, ...options.highlights };
    }

    /** Largest CoC radius in pixels for an image `widthPx` wide. */
    maxCocRadiusPx(widthPx: number): number {
        return Math.max(Math.min(this.maxCocFraction * widthPx, this.maxCocPx), 0.0);
    }

    /** Signed CoC radius in pixels of a point at `viewDepthM` for this camera and image width. */
    cocRadiusPx(camera: Camera, widthPx: number, viewDepthM: number): number {
        const r = this.lens.cocRadiusPx(this.lens.focalLength(camera), widthPx, viewDepthM);
        const max = this.maxCocRadiusPx(widthPx);
        return Math.min(Math.max(r, -max), max);
    }

    // ========================================================================
    // PostProcessingEffect interface
    // ========================================================================

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const compute = GPUShaderStage.COMPUTE;
        const tex = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, texture: { sampleType: 'unfilterable-float' } });
        const depth = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, texture: { sampleType: 'depth' } });
        const storageFormat = (binding: number, format: GPUTextureFormat): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, storageTexture: { access: 'write-only', format } });
        const storage = (binding: number) => storageFormat(binding, 'rgba16float');
        const layerStorage = (binding: number) => storageFormat(binding, LAYER_FORMAT);
        const uniform = (binding: number): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, buffer: { type: 'uniform' } });
        const buffer = (binding: number, readOnly: boolean): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: compute, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } });
        const bgl = (label: string, entries: GPUBindGroupLayoutEntry[]) => device.createBindGroupLayout({ label, entries });

        const layouts: Layouts = {
            prefilter: bgl('CinematicDoF/PrefilterBGL', [tex(0), depth(1), layerStorage(2), uniform(3), layerStorage(4), storage(5)]),
            highlights: bgl('CinematicDoF/HighlightsBGL', [
                tex(0), tex(1), layerStorage(2), layerStorage(3), uniform(4),
                buffer(5, false), buffer(6, false), uniform(7), buffer(8, false), buffer(9, false),
            ]),
            downsample: bgl('CinematicDoF/DownsampleBGL', [tex(0), layerStorage(1), uniform(2)]),
            tiles: bgl('CinematicDoF/TilesBGL', [tex(0), storage(1), uniform(2)]),
            gather: bgl('CinematicDoF/GatherBGL', [tex(0), tex(1), storage(2), tex(3), uniform(4)]),
            postfilter: bgl('CinematicDoF/PostfilterBGL', [
                tex(0), tex(1), storage(2), storage(3), uniform(4), buffer(5, true), buffer(6, true), buffer(7, true),
            ]),
            composite: bgl('CinematicDoF/CompositeBGL', [tex(0), depth(1), tex(2), tex(3), storage(4), uniform(5)]),
        };
        const modules = new Map<string, GPUShaderModule>();
        const pipeline = (label: string, code: string, entryPoint: string, layout: GPUBindGroupLayout) => {
            let module = modules.get(code);
            if (!module) {
                module = device.createShaderModule({ label, code });
                modules.set(code, module);
            }
            return device.createComputePipeline({
                label,
                layout: device.createPipelineLayout({ label, bindGroupLayouts: [layout] }),
                compute: { module, entryPoint },
            });
        };
        this._pipelines = {
            prefilter: pipeline('CinematicDoF/Prefilter', source(PREFILTER), 'main', layouts.prefilter),
            highlights: pipeline('CinematicDoF/Highlights', spriteSource(HIGHLIGHTS), 'main', layouts.highlights),
            downsampleNear: pipeline('CinematicDoF/DownsampleNear', source(DOWNSAMPLE), 'downsampleNear', layouts.downsample),
            downsampleFar: pipeline('CinematicDoF/DownsampleFar', source(DOWNSAMPLE), 'downsampleFar', layouts.downsample),
            dilate: pipeline('CinematicDoF/Dilate', source(TILES), 'dilate', layouts.tiles),
            gatherNear: pipeline('CinematicDoF/GatherNear', source(GATHER), 'gatherNear', layouts.gather),
            gatherFar: pipeline('CinematicDoF/GatherFar', source(GATHER), 'gatherFar', layouts.gather),
            postfilter: pipeline('CinematicDoF/Postfilter', spriteSource(POSTFILTER), 'main', layouts.postfilter),
            composite: pipeline('CinematicDoF/Composite', source(COMPOSITE), 'main', layouts.composite),
        };
        this._layouts = layouts;

        this._maxSprites = Math.max(Math.floor(this.highlights.maxSprites), 1);
        const createBuffer = (label: string, size: number, usage: number) => device.createBuffer({ label, size, usage });
        this._params = createBuffer('CinematicDoF/Params', PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this._highlightParams = createBuffer('CinematicDoF/HighlightParams', HIGHLIGHT_PARAMS_BYTES, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        this._sprites = createBuffer('CinematicDoF/Sprites', this._maxSprites * SPRITE_BYTES, GPUBufferUsage.STORAGE);
        this._spriteCount = createBuffer('CinematicDoF/SpriteCount', 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number
    ): void {
        if (!this._pipelines) return;
        const device = this._device!;
        const targets = this._ensureTargets(width, height);
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput || !this._prefilter) {
            this._buildChainBindGroups(input, depth, output);
        }

        const focal = this.lens.focalLength(camera);
        const f = this._paramsF32, u = this._paramsU32;
        f[0] = this.lens.cocScale(focal, width);
        f[1] = Math.max(this.lens.focusDistanceM, 1e-3);
        f[2] = this.maxCocRadiusPx(width);
        f[3] = camera.near;
        f[4] = camera.far;
        u[5] = width;
        u[6] = height;
        u[7] = Math.max(Math.floor(this.sampleCount), 1);
        u[8] = Math.max(Math.floor(this.lens.bladeCount), 0);
        f[9] = this.lens.bladeRotationDeg * DEG;
        u[10] = this.temporalNoise ? (this._frame % 64) + 1 : 0;
        u[11] = this.debugView;
        this._frame = (this._frame + 1) >>> 0;
        device.queue.writeBuffer(this._params!, 0, this._paramsData);

        const hf = this._highlightF32, hu = this._highlightU32;
        hf[0] = Math.max(this.highlights.contrast, 1.0);
        hf[1] = Math.max(this.highlights.minCocPx, 0.5) * 0.5;
        hu[2] = this._maxSprites;
        hu[3] = this.highlights.enabled ? 1 : 0;
        device.queue.writeBuffer(this._highlightParams!, 0, this._highlightData);

        const p = this._pipelines;
        const hw = divCeil(width, 2), hh = divCeil(height, 2);
        const tw = divCeil(hw, TILE), th = divCeil(hh, TILE);
        const passes: [GPUComputePipeline, GPUBindGroup, number, number][] = [
            [p.prefilter, this._prefilter!, hw, hh],
            [p.highlights, targets.highlights, hw, hh],
        ];
        targets.farDownsamples.forEach((far, l) => {
            const sw = Math.max(hw >> (l + 1), 1), sh = Math.max(hh >> (l + 1), 1);
            const near = targets.nearDownsamples[l];
            if (near) passes.push([p.downsampleNear, near, sw, sh]);
            passes.push([p.downsampleFar, far, sw, sh]);
        });
        passes.push(
            [p.dilate, targets.dilate, tw, th],
            [p.gatherNear, targets.gatherNear, hw, hh],
            [p.gatherFar, targets.gatherFar, hw, hh],
            [p.postfilter, targets.postfilter, hw, hh],
            [p.composite, this._composite!, width, height],
        );

        // the highlights are counted and binned afresh every frame
        commandEncoder.clearBuffer(this._spriteCount!);
        commandEncoder.clearBuffer(targets.binCount);
        const pass = commandEncoder.beginComputePass({ label: 'CinematicDoF' });
        for (const [pipeline, bindGroup, x, y] of passes) {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bindGroup);
            pass.dispatchWorkgroups(divCeil(x, 8), divCeil(y, 8));
        }
        pass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // the targets follow render()'s size; the chain's textures were recreated
        this._prefilter = null;
        this._composite = null;
    }

    destroy(): void {
        this._destroyTargets();
        this._params?.destroy();
        this._highlightParams?.destroy();
        this._sprites?.destroy();
        this._spriteCount?.destroy();
        this._params = this._highlightParams = this._sprites = this._spriteCount = null;
        this._pipelines = null;
        this._layouts = null;
        this._prefilter = this._composite = null;
        this._currentInput = this._currentDepth = this._currentOutput = null;
        this.initialized = false;
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    private _ensureTargets(width: number, height: number): Targets {
        if (this._targets && this._targets.width === width && this._targets.height === height) return this._targets;
        this._destroyTargets();
        const device = this._device!;
        const layouts = this._layouts!;
        const textures: GPUTexture[] = [];
        const texture = (label: string, w: number, h: number, mipLevelCount: number, format: GPUTextureFormat) => {
            const t = device.createTexture({
                label,
                size: [Math.max(w, 1), Math.max(h, 1)],
                mipLevelCount,
                format,
                usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
            });
            textures.push(t);
            return t;
        };
        const target = (label: string, w: number, h: number) => texture(label, w, h, 1, 'rgba16float').createView();
        const hw = divCeil(width, 2), hh = divCeil(height, 2);
        const tw = divCeil(hw, TILE), th = divCeil(hh, TILE);
        const bins = divCeil(hw, BIN) * divCeil(hh, BIN);
        const layer = (label: string, levels: number): Layer => {
            const count = Math.min(levels, Math.floor(Math.log2(Math.max(hw, hh, 1))) + 1);
            const chainTexture = texture(label, hw, hh, count, LAYER_FORMAT);
            const rawTexture = texture(`${label}Raw`, hw, hh, 1, LAYER_FORMAT);
            return {
                raw: rawTexture.createView(),
                chain: chainTexture.createView(),
                levels: Array.from({ length: count }, (_, level) => chainTexture.createView({ baseMipLevel: level, mipLevelCount: 1 })),
            };
        };
        const near = layer('CinematicDoF/Near', LEVELS);
        const far = layer('CinematicDoF/Far', FILL_LEVELS);
        const tiles = target('CinematicDoF/Tiles', tw, th);
        const tilesDilated = target('CinematicDoF/TilesDilated', tw, th);
        const bg = target('CinematicDoF/Background', hw, hh);
        const fg = target('CinematicDoF/Foreground', hw, hh);
        const bgFiltered = target('CinematicDoF/BackgroundFiltered', hw, hh);
        const fgFiltered = target('CinematicDoF/ForegroundFiltered', hw, hh);
        const binCount = device.createBuffer({ label: 'CinematicDoF/SpriteBinCount', size: bins * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        const binList = device.createBuffer({ label: 'CinematicDoF/SpriteBinList', size: bins * BIN_CAPACITY * 4, usage: GPUBufferUsage.STORAGE });

        const params: GPUBindingResource = { buffer: this._params! };
        const group = (layout: GPUBindGroupLayout, resources: GPUBindingResource[]) => device.createBindGroup({
            label: 'CinematicDoF/BG',
            layout,
            entries: resources.map((resource, binding) => ({ binding, resource })),
        });
        const downsamples = (l: Layer) =>
            l.levels.slice(1).map((dst, i) => group(layouts.downsample, [l.levels[i], dst, params]));

        this._targets = {
            width, height, textures, near, far, tiles, tilesDilated, bg, fg, bgFiltered, fgFiltered, binCount, binList,
            highlights: group(layouts.highlights, [
                near.raw, far.raw, near.levels[0], far.levels[0], params,
                { buffer: this._sprites! }, { buffer: this._spriteCount! }, { buffer: this._highlightParams! },
                { buffer: binCount }, { buffer: binList },
            ]),
            nearDownsamples: downsamples(near),
            farDownsamples: downsamples(far),
            dilate: group(layouts.tiles, [tiles, tilesDilated, params]),
            gatherNear: group(layouts.gather, [near.chain, tilesDilated, fg, near.levels[0], params]),
            gatherFar: group(layouts.gather, [far.chain, tilesDilated, bg, near.levels[0], params]),
            postfilter: group(layouts.postfilter, [
                bg, fg, bgFiltered, fgFiltered, params, { buffer: this._sprites! }, { buffer: binCount }, { buffer: binList },
            ]),
        };
        // the prefilter writes, and the composite reads, the new targets
        this._prefilter = null;
        this._composite = null;
        return this._targets;
    }

    private _buildChainBindGroups(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const device = this._device!;
        const layouts = this._layouts!;
        const t = this._targets!;
        const params: GPUBindingResource = { buffer: this._params! };
        const group = (layout: GPUBindGroupLayout, resources: GPUBindingResource[]) => device.createBindGroup({
            label: 'CinematicDoF/BG',
            layout,
            entries: resources.map((resource, binding) => ({ binding, resource })),
        });
        this._prefilter = group(layouts.prefilter, [input.createView(), depth.createView(), t.near.raw, params, t.far.raw, t.tiles]);
        this._composite = group(layouts.composite, [input.createView(), depth.createView(), t.bgFiltered, t.fgFiltered, output.createView(), params]);
        this._currentInput = input;
        this._currentDepth = depth;
        this._currentOutput = output;
    }

    private _destroyTargets(): void {
        if (!this._targets) return;
        for (const texture of this._targets.textures) texture.destroy();
        this._targets.binCount.destroy();
        this._targets.binList.destroy();
        this._targets = null;
    }
}

export { CinematicDepthOfFieldEffect, CameraLens, DofDebugView };
