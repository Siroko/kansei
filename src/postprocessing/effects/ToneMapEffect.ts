import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { LOCAL_EXPOSURE_WGSL, TONEMAP_WGSL } from '../../materials/shaders/SharedWGSL';
import { gpuPass } from '../../profiling/Profiler';

// The display transform of the Rust engine (`rust/kansei-core/src/postprocessing/effects/tonemap.rs`),
// on the same WGSL (`shaders/tonemap.wgsl`, `tonemap_params.wgsl`, `local_exposure.wgsl`).

/**
 * The lens attenuation q of Unreal Engine 5 (`r.EyeAdaptation.LensAttenuation`), with which
 * 1 cd/m² exposes to 1.0 at EV100 0: the ISO 12232 saturation constant 0.78 over q.
 */
export const LENS_ATTENUATION_UE5 = 0.78;
/**
 * Unreal Engine 4's (and Lagarde & de Rousiers', Moving Frostbite to PBR, 2014): 1 cd/m²
 * exposes to 1/1.2 at EV100 0.
 */
export const LENS_ATTENUATION_UE4 = 0.65;

/**
 * The linear exposure for a manual EV100 through a lens of attenuation q: `q / 0.78 / 2^ev100`
 * (Unreal's `LuminanceMaxFromLensAttenuation`). The default, `LENS_ATTENUATION_UE5`, exposes as
 * Unreal Engine 5 does, `1 / 2^ev100`: physical light values (cd/m²) times this land in the tone
 * curve's range; EV100 3.9 (dusk) gives 0.067. `LENS_ATTENUATION_UE4` gives `1 / (1.2 * 2^ev100)`.
 */
export function exposureFromEV100(ev100: number, lensAttenuation: number = LENS_ATTENUATION_UE5): number {
    return Math.max(lensAttenuation, 0.01) / 0.78 / Math.pow(2, ev100);
}

/**
 * EV100 of a physical camera: `log2(N² / t * 100 / ISO)`, for an f-number, a shutter time in
 * seconds and an ISO sensitivity.
 */
export function ev100FromCamera(fNumber: number, shutterSeconds: number, iso: number): number {
    return Math.log2(fNumber * fNumber / shutterSeconds * 100 / iso);
}

/** The filmic tone curve that maps exposed scene-linear light to display-linear [0, 1]. */
export const ToneMapper = {
    /** Clamp only. */
    None: 0,
    /** ACES RRT + sRGB ODT, Stephen Hill's fit. UE's default filmic curve is tuned to match it. */
    AcesFitted: 1,
    /** AgX (Blender 4 default): graceful highlight desaturation, no hue skews in saturated lights. */
    AgX: 2,
    /** AgX with Blender's "Punchy" look. */
    AgXPunchy: 3,
    /** Khronos PBR Neutral: keeps base colours as authored up to 0.76, then a soft shoulder. */
    KhronosNeutral: 4,
    /**
     * Unreal Engine's filmic display transform, as its tonemapper LUT computes it for an sRGB
     * display: Unreal's white balance, the colour into AP1 with its gamut expansion, the grade as
     * Unreal's ColorCorrectAll in AP1, blue correction, `FilmToneMap` with `unrealFilm`'s curve,
     * and back to sRGB. For scenes matched to Unreal.
     */
    UnrealFilmic: 5,
    /** `1 - exp(-x)`: a plain exponential shoulder, for scenes tuned to it in their own shaders. */
    Exponential: 6,
} as const;
export type ToneMapper = typeof ToneMapper[keyof typeof ToneMapper];

type Vec3 = [number, number, number];

/** Colour grading on exposed scene-linear light, before the tone curve (where UE grades too). The defaults are neutral. */
export interface ColorGrade {
    /**
     * Colour temperature (K) of the light that should read as white. 6500 is neutral; lower
     * values neutralise warm light (the picture turns cooler).
     */
    whiteTemperature: number;
    /**
     * Offset of that white from the Planckian locus, -1..1 (±0.02 Δuv). Positive treats a
     * greener white as neutral, so the picture turns magenta.
     */
    whiteTint: number;
    /** Power about middle grey (0.18): above 1 more contrast. */
    contrast: number;
    /**
     * Per channel, as UE's ColorSaturation (its rgb times its w): each channel's distance from
     * the luminance is scaled, so (0.9, 0.95, 1.0) desaturates reds most.
     */
    saturation: Vec3;
    /** Per-channel multiplier. */
    gain: Vec3;
    shadowGain: Vec3;
    shadowSaturation: Vec3;
    highlightGain: Vec3;
    highlightSaturation: Vec3;
    /** Luminance below which a pixel counts as shadow (fading out toward it). */
    shadowsMax: number;
    /** Luminance above which a pixel starts to count as highlight (fully at 1). */
    highlightsMin: number;
}

export function defaultColorGrade(): ColorGrade {
    return {
        whiteTemperature: 6500,
        whiteTint: 0,
        contrast: 1,
        saturation: [1, 1, 1],
        gain: [1, 1, 1],
        shadowGain: [1, 1, 1],
        shadowSaturation: [1, 1, 1],
        highlightGain: [1, 1, 1],
        highlightSaturation: [1, 1, 1],
        shadowsMax: 0.09,
        highlightsMin: 0.5,
    };
}

/**
 * Unreal's filmic curve and what surrounds it (its post-process settings `FilmSlope`, `FilmToe`,
 * `FilmShoulder`, `FilmBlackClip`, `FilmWhiteClip`, `BlueCorrection`, `ExpandGamut`), for
 * `ToneMapper.UnrealFilmic`. The defaults are Unreal's.
 */
export interface UnrealFilm {
    slope: number;
    toe: number;
    shoulder: number;
    blackClip: number;
    whiteClip: number;
    /** How much of the blue correction for bright blue lights (0..1). */
    blueCorrection: number;
    /** How far bright saturated colours are pushed out of the sRGB gamut toward AP1 (0..1). */
    expandGamut: number;
}

export function defaultUnrealFilm(): UnrealFilm {
    return { slope: 0.88, toe: 0.55, shoulder: 0.26, blackClip: 0, whiteClip: 0.04, blueCorrection: 0.6, expandGamut: 1 };
}

/**
 * Unreal Engine 5's local exposure, its bilateral method (post-process settings
 * `LocalExposure*`): a factor on each pixel's light that scales the contrast of its
 * surroundings' luminance about middle grey, while keeping its detail against them. Bright
 * surroundings (a sky) come down and dark ones come up, as a photographer dodges and burns.
 *
 * Each frame it builds Unreal's inputs from the picture: a bilateral grid of log luminance (cells
 * of 128 x 128 pixels, 32 bins of luminance), and the log luminance at 1/32 of the picture's size,
 * blurred. Middle grey is 0.18 of exposed light. The defaults are Unreal's, which change nothing;
 * `unrealLocalExposure(highlight, shadow)` sets the two contrasts.
 */
export interface LocalExposure {
    /** Contrast of the surroundings brighter than middle grey (1 keeps it, 0 flattens it). */
    highlightContrast: number;
    /** Contrast of the surroundings darker than middle grey. */
    shadowContrast: number;
    /** How much of a pixel's detail against its surroundings is kept. */
    detailStrength: number;
    /** How much the surroundings are the blurred luminance rather than the bilateral grid's. */
    blurredLuminanceBlend: number;
    /** The blurred luminance's kernel, as a share of the picture's width (%). */
    blurredLuminanceKernelPercent: number;
    /** Stops on middle grey. */
    middleGreyBias: number;
    /** The log2 scene luminance the grid's bins span (Unreal's histogram range, -10 to 20). */
    logLuminanceRange: [number, number];
}

/** Unreal's local exposure defaults with these highlight and shadow contrasts. */
export function unrealLocalExposure(highlightContrast: number = 1, shadowContrast: number = 1): LocalExposure {
    return {
        highlightContrast,
        shadowContrast,
        detailStrength: 1,
        blurredLuminanceBlend: 0.6,
        blurredLuminanceKernelPercent: 50,
        middleGreyBias: 0,
        logLuminanceRange: [-10, 20],
    };
}

export interface ToneMapOptions {
    tonemapper: ToneMapper;
    /** The curve of `ToneMapper.UnrealFilmic`. */
    unrealFilm: UnrealFilm;
    /** Linear multiplier on scene light; see `exposureFromEV100`. */
    exposure: number;
    /** Stops on top of `exposure` (per-shot trims). */
    exposureCompensation: number;
    grade: ColorGrade;
    /**
     * Natural (cos⁴) vignetting, 0 = off: Unreal's `VignetteIntensity` and its formula, so 0.5
     * darkens the corners to 0.44.
     */
    vignette: number;
    /** Lateral chromatic aberration, 0 = off; at 1 red and blue differ in magnification by 1 %. */
    chromaticAberration: number;
    /** Film grain amplitude, 0 = off (UE's GrainIntensity scale). */
    grain: number;
    /** Grain size in pixels. */
    grainSize: number;
    /**
     * Write sRGB-encoded values. Needed when the canvas format is not sRGB (browsers give
     * `bgra8unorm`), since the blit copies the chain's last texture as is; see `toneMapOptionsForSurface`.
     */
    encodeSrgb: boolean;
    /** Triangular dither of one 8-bit step against banding. */
    dither: boolean;
    /** Unreal's local exposure; off when null. */
    localExposure: LocalExposure | null;
    /**
     * The aspect (width / height) of the frame the picture is the centre crop of, when it is
     * letterboxed: the vignette is the frame's. null: the picture's own.
     */
    frameAspect: number | null;
}

export function defaultToneMapOptions(): ToneMapOptions {
    return {
        tonemapper: ToneMapper.AcesFitted,
        unrealFilm: defaultUnrealFilm(),
        exposure: 1,
        exposureCompensation: 0,
        grade: defaultColorGrade(),
        vignette: 0,
        chromaticAberration: 0,
        grain: 0,
        grainSize: 1.6,
        encodeSrgb: true,
        dither: true,
        localExposure: null,
        frameAspect: null,
    };
}

/**
 * Defaults with `encodeSrgb` set for a canvas of this format (`renderer.presentationFormat`):
 * encode unless the format is sRGB already.
 */
export function toneMapOptionsForSurface(format: GPUTextureFormat): ToneMapOptions {
    return { ...defaultToneMapOptions(), encodeSrgb: !format.endsWith('-srgb') };
}

// ── 3x3 matrices, column-major as glam's: m[c][r] ──

type Mat3 = [Vec3, Vec3, Vec3];

const mulVec = (m: Mat3, v: Vec3): Vec3 => [
    m[0][0] * v[0] + m[1][0] * v[1] + m[2][0] * v[2],
    m[0][1] * v[0] + m[1][1] * v[1] + m[2][1] * v[2],
    m[0][2] * v[0] + m[1][2] * v[1] + m[2][2] * v[2],
];

const mul = (a: Mat3, b: Mat3): Mat3 => [mulVec(a, b[0]), mulVec(a, b[1]), mulVec(a, b[2])];

const diagonal = (v: Vec3): Mat3 => [[v[0], 0, 0], [0, v[1], 0], [0, 0, v[2]]];

function inverse(m: Mat3): Mat3 {
    const [[a, b, c], [d, e, f], [g, h, i]] = m;
    const A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g;
    const det = a * A + b * B + c * C;
    const s = 1 / det;
    // the inverse's columns are the cofactor matrix's rows over the determinant
    return [
        [A * s, -(b * i - c * h) * s, (b * f - c * e) * s],
        [B * s, (a * i - c * g) * s, -(a * f - c * d) * s],
        [C * s, -(a * h - b * g) * s, (a * e - b * d) * s],
    ];
}

const BRADFORD: Mat3 = [[0.8951, -0.7502, 0.0389], [0.2664, 1.7135, -0.0685], [-0.1614, 0.0367, 1.0296]];

// ── White balance ──

/** CIE 1931 xy of the Planckian locus at `kelvin` (Kim et al. 2002 cubic fit, 1667-25000 K). */
function planckianXY(kelvin: number): [number, number] {
    const t = Math.min(Math.max(kelvin, 1667), 25000);
    const t2 = t * t, t3 = t2 * t;
    const x = t <= 4000
        ? -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
        : -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390;
    const x2 = x * x, x3 = x2 * x;
    const y = t <= 2222
        ? -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
        : t <= 4000
            ? -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
            : 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
    return [x, y];
}

function xyToUV([x, y]: [number, number]): [number, number] {
    const d = -2 * x + 12 * y + 3;
    return [4 * x / d, 6 * y / d];
}

function uvToXY([u, v]: [number, number]): [number, number] {
    const d = 2 * u - 8 * v + 4;
    return [3 * u / d, 2 * v / d];
}

const xyToXYZ = ([x, y]: [number, number]): Vec3 => [x / y, 1, (1 - x - y) / y];

/**
 * White point for a temperature and a tint: the Planckian locus, moved `tint * 0.02` along its
 * normal in CIE 1960 uv (toward green for positive tints).
 */
function whitePointXY(kelvin: number, tint: number): [number, number] {
    const uv = xyToUV(planckianXY(kelvin));
    const a = xyToUV(planckianXY(kelvin * 1.01)), b = xyToUV(planckianXY(kelvin * 0.99));
    // toward lower temperatures the locus runs to +u, +v; its left normal points to +v (green)
    const tangent = [b[0] - a[0], b[1] - a[1]];
    const len = Math.hypot(tangent[0], tangent[1]);
    const t = len > 0 ? [tangent[0] / len, tangent[1] / len] : [0, 0];
    return uvToXY([uv[0] - t[1] * tint * 0.02, uv[1] + t[0] * tint * 0.02]);
}

/** Linear sRGB (D65) to CIE XYZ. */
const SRGB_TO_XYZ: Mat3 = [[0.4124564, 0.2126729, 0.0193339], [0.3575761, 0.7151522, 0.119192], [0.1804375, 0.072175, 0.950304]];

/**
 * Linear-sRGB matrix (column-major) that adapts white(`kelvin`, `tint`) to white(6500 K, 0) with
 * the Bradford transform, so 6500/0 is the identity.
 */
export function whiteBalanceMatrix(kelvin: number, tint: number): Mat3 {
    const src = mulVec(BRADFORD, xyToXYZ(whitePointXY(kelvin, tint)));
    const dst = mulVec(BRADFORD, xyToXYZ(whitePointXY(6500, 0)));
    const scale = diagonal([dst[0] / src[0], dst[1] / src[1], dst[2] / src[2]]);
    return mul(mul(mul(mul(inverse(SRGB_TO_XYZ), inverse(BRADFORD)), scale), BRADFORD), SRGB_TO_XYZ);
}

// ── Unreal's white balance (TonemapCommon.ush) ──

function unrealDIlluminantXY(temperature: number): [number, number] {
    const t = temperature * 1.4388 / 1.438;
    const o = 1 / t;
    const x = t <= 7000
        ? 0.244063 + (0.09911e3 + (2.9678e6 - 4.6070e9 * o) * o) * o
        : 0.237040 + (0.24748e3 + (1.9018e6 - 2.0064e9 * o) * o) * o;
    return [x, -3 * x * x + 2.87 * x - 0.275];
}

function unrealPlanckianUV(t: number): [number, number] {
    const u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (1 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    const v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (1 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    return [u, v];
}

/** The Planckian locus at `t`, moved `tint * 0.05` along its isotherm (Unreal's `PlanckianIsothermal`). */
function unrealPlanckianIsothermalXY(t: number, tint: number): [number, number] {
    const uv = unrealPlanckianUV(t);
    const ud = (-1.13758118e9 - 1.91615621e6 * t - 1.53177 * t * t) / Math.pow(1.41213984e6 + 1189.62 * t + t * t, 2);
    const vd = (1.97471536e9 - 705674.0 * t - 308.607 * t * t) / Math.pow(6.19363586e6 - 179.456 * t + t * t, 2);
    const len = Math.hypot(ud, vd);
    const n = [ud / len, vd / len];
    return uvToXY([uv[0] + n[1] * tint * 0.05, uv[1] - n[0] * tint * 0.05]);
}

/**
 * Linear-sRGB matrix (column-major) of Unreal's temperature white balance (`WhiteBalance`): the
 * white of `kelvin` (a daylight illuminant from 4000 K, the Planckian locus below), moved along
 * its isotherm by `tint`, adapted to D65 with the Bradford transform. Its tint runs the same way
 * as `whiteBalanceMatrix`'s, 0.05 in CIE 1960 uv per unit rather than 0.02.
 */
export function unrealWhiteBalanceMatrix(kelvin: number, tint: number): Mat3 {
    const locus = uvToXY(unrealPlanckianUV(kelvin));
    const base = kelvin < 4000 ? locus : unrealDIlluminantXY(kelvin);
    const iso = unrealPlanckianIsothermalXY(kelvin, tint);
    const src: [number, number] = [base[0] + iso[0] - locus[0], base[1] + iso[1] - locus[1]];
    const bradfordInv: Mat3 = [[0.9869929, 0.4323053, -0.0085287], [-0.1470543, 0.5183603, 0.0400428], [0.1599627, 0.0492912, 0.9684867]];
    const s = mulVec(BRADFORD, xyToXYZ(src));
    const d = mulVec(BRADFORD, xyToXYZ([0.31270, 0.32900]));
    const cat = mul(mul(bradfordInv, diagonal([d[0] / s[0], d[1] / s[1], d[2] / s[2]])), BRADFORD);
    const srgbToXYZ: Mat3 = [[0.4123907993, 0.2126390059, 0.0193308187], [0.3575843394, 0.7151686788, 0.1191947798], [0.1804807884, 0.0721923154, 0.9505321522]];
    const xyzToSrgb: Mat3 = [[3.2409699419, -0.9692436363, 0.0556300797], [-1.5373831776, 1.8759675015, -0.2039769589], [-0.4986107603, 0.0415550574, 1.0569715142]];
    return mul(mul(xyzToSrgb, cat), srgbToXYZ);
}

// ── GPU layout (ToneMapParams in tonemap_params.wgsl: 64 words) ──

const PARAMS_BYTES = 256;
const FLAG_ENCODE_SRGB = 1;
const FLAG_DITHER = 2;
const FLAG_LOCAL_EXPOSURE = 4;
/** Half-resolution texels per side of a bilateral grid cell (local_exposure.wgsl's LOCAL_CELL). */
const LOCAL_CELL = 64;
/** Bins of the grid. */
const LOCAL_BINS = 32;

const divCeil = (a: number, b: number) => Math.ceil(a / b);

/** The bilateral grid's cells and the blurred luminance's texels for a picture size. */
function localSizes(width: number, height: number): [[number, number], [number, number]] {
    const half = [divCeil(width, 2), divCeil(height, 2)];
    return [[divCeil(half[0], LOCAL_CELL), divCeil(half[1], LOCAL_CELL)], [divCeil(width, 32), divCeil(height, 32)]];
}

/**
 * The display transform: physical exposure, lens (chromatic aberration, vignette), a
 * scene-linear grade, a filmic tone curve, film grain, dither and sRGB encoding. Optionally
 * Unreal's local exposure.
 *
 * It turns scene-linear HDR into the display signal, so it goes last in the HDR chain: after
 * fog, depth of field and bloom, which all want linear light. A display-space
 * `ColorGradingEffect` may follow it.
 *
 * ```ts
 * const tonemap = new ToneMapEffect({
 *     ...toneMapOptionsForSurface(renderer.presentationFormat),
 *     exposure: exposureFromEV100(3.9),
 *     vignette: 0.5, grain: 0.22, chromaticAberration: 0.25,
 * });
 * const volume = new PostProcessingVolume(renderer, [fog, bloom, tonemap]);
 * ```
 */
class ToneMapEffect extends PostProcessingEffect {
    public options: ToneMapOptions;

    private _frame = 0;
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _params: GPUBuffer | null = null;
    private _sampler: GPUSampler | null = null;
    private readonly _paramsData = new ArrayBuffer(PARAMS_BYTES);

    // local exposure: pipelines, and textures for a picture size (1-texel stand-ins until it is on)
    private _localPipelines: { grid: GPUComputePipeline; logLuminance: GPUComputePipeline; blurX: GPUComputePipeline; blurY: GPUComputePipeline } | null = null;
    private _localSize: [number, number] = [0, 0];
    private _localTextures: GPUTexture[] = [];
    private _gridView: GPUTextureView | null = null;
    /** The log luminance, and the blur's intermediate. */
    private _logViews: GPUTextureView[] = [];

    // bind groups, rebuilt when what they bind changes
    private _bindGroup: GPUBindGroup | null = null;
    private _localBindGroups: GPUBindGroup[] | null = null;
    private _boundInput: GPUTexture | null = null;
    private _boundOutput: GPUTexture | null = null;
    private _localBoundInput: GPUTexture | null = null;

    constructor(options: Partial<ToneMapOptions> = {}) {
        super();
        this.options = { ...defaultToneMapOptions(), ...options };
    }

    /**
     * The linear factor applied to scene light: exposure times 2^compensation (what a bloom
     * threshold in tonemapper units would use).
     */
    public totalExposure(): number {
        return this.options.exposure * Math.pow(2, this.options.exposureCompensation);
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const texture2d = { sampleType: 'float', viewDimension: '2d' } as const;
        const bgl = device.createBindGroupLayout({
            label: 'ToneMap/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: texture2d },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, texture: texture2d },
            ],
        });
        this._pipeline = device.createComputePipeline({
            label: 'ToneMap/Pipeline',
            layout: device.createPipelineLayout({ label: 'ToneMap/Layout', bindGroupLayouts: [bgl] }),
            compute: { module: device.createShaderModule({ label: 'ToneMap/Shader', code: TONEMAP_WGSL }), entryPoint: 'main' },
        });
        this._params = device.createBuffer({
            label: 'ToneMap/Params',
            size: PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._sampler = device.createSampler({
            label: 'ToneMap/Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
        });
        const module = device.createShaderModule({ label: 'ToneMap/LocalExposure', code: LOCAL_EXPOSURE_WGSL });
        const local = (entryPoint: string) => device.createComputePipeline({ label: `ToneMap/LocalExposure/${entryPoint}`, layout: 'auto', compute: { module, entryPoint } });
        this._localPipelines = { grid: local('grid'), logLuminance: local('logLuminance'), blurX: local('blurX'), blurY: local('blurY') };
        this._createLocalTextures([1, 1, 1], [1, 1]);
        this.initialized = true;
    }

    /** Local exposure's grid (`cells` and its bins) and the blurred luminance's two textures. */
    private _createLocalTextures(grid: [number, number, number], blurred: [number, number]): void {
        for (const t of this._localTextures) t.destroy();
        const texture = (label: string, size: GPUExtent3DStrict, dimension: GPUTextureDimension) => this._device!.createTexture({
            label,
            size,
            dimension,
            format: 'rgba16float',
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        const flat = { width: blurred[0], height: blurred[1] };
        this._localTextures = [
            texture('ToneMap/LocalExposureGrid', { width: grid[0], height: grid[1], depthOrArrayLayers: grid[2] }, '3d'),
            texture('ToneMap/LocalExposureLog', flat, '2d'),
            texture('ToneMap/LocalExposureBlur', flat, '2d'),
        ];
        this._gridView = this._localTextures[0].createView();
        this._logViews = [this._localTextures[1].createView(), this._localTextures[2].createView()];
        this._bindGroup = null;
        this._localBindGroups = null;
    }

    /** The params for a picture size, written into `_paramsData`. */
    private _writeParams(width: number, height: number): void {
        const o = this.options;
        const g = o.grade;
        const f32 = new Float32Array(this._paramsData);
        const u32 = new Uint32Array(this._paramsData);
        f32.fill(0);
        const wb = o.tonemapper === ToneMapper.UnrealFilmic
            ? unrealWhiteBalanceMatrix(g.whiteTemperature, g.whiteTint)
            : whiteBalanceMatrix(g.whiteTemperature, g.whiteTint);
        wb.forEach((column, c) => f32.set(column, c * 4));
        const vec3AndScalar = (offset: number, v: Vec3, s: number) => { f32.set(v, offset); f32[offset + 3] = s; };
        vec3AndScalar(12, g.gain, this.totalExposure());
        vec3AndScalar(16, g.shadowGain, g.contrast);
        vec3AndScalar(20, g.highlightGain, Math.max(g.shadowsMax, 1e-4));
        vec3AndScalar(24, g.saturation, Math.min(g.highlightsMin, 0.999));
        vec3AndScalar(28, g.shadowSaturation, Math.max(o.vignette, 0));
        vec3AndScalar(32, g.highlightSaturation, Math.max(o.chromaticAberration, 0));
        f32[36] = Math.max(o.grain, 0);
        f32[37] = Math.max(o.grainSize, 1);
        u32[38] = width;
        u32[39] = height;
        u32[40] = this._frame;
        u32[41] = o.tonemapper;
        u32[42] = (o.encodeSrgb ? FLAG_ENCODE_SRGB : 0) | (o.dither ? FLAG_DITHER : 0) | (o.localExposure ? FLAG_LOCAL_EXPOSURE : 0);
        f32[43] = Math.max(o.frameAspect ?? 0, 0);
        const film = o.unrealFilm;
        f32.set([film.slope, film.toe, film.shoulder, film.blackClip], 44);
        f32.set([film.whiteClip, Math.min(Math.max(film.blueCorrection, 0), 1), Math.max(film.expandGamut, 0), 1], 48);

        const le = o.localExposure;
        if (!le) return;
        const [logMin, logMax] = le.logLuminanceRange;
        const scale = 1 / Math.max(logMax - logMin, 1e-3);
        const half = [divCeil(width, 2), divCeil(height, 2)];
        const [cells, blurred] = localSizes(width, height);
        // Unreal's Gaussian (PostProcessWeightedSampleSum): a radius of half the kernel's share of
        // the blurred texture's width, at most 31 texels
        const radius = Math.min(Math.max(blurred[0] * le.blurredLuminanceKernelPercent * 0.01 * 0.5, 1e-3), 31);
        f32.set([le.highlightContrast, le.shadowContrast, le.detailStrength, Math.min(Math.max(le.blurredLuminanceBlend, 0), 1)], 52);
        f32.set([Math.log2(0.18) + le.middleGreyBias, scale, -logMin * scale, logMin], 56);
        f32.set([half[0] / LOCAL_CELL / cells[0], half[1] / LOCAL_CELL / cells[1], radius, Math.min(Math.ceil(radius), 31)], 60);
    }

    /**
     * Local exposure's grid and blurred luminance for this frame's input (params already
     * written): the grid, the log luminance at 1/32, and its blur across then down.
     */
    private _encodeLocalExposure(commandEncoder: GPUCommandEncoder, input: GPUTexture, width: number, height: number): void {
        const device = this._device!;
        const [cells, blurred] = localSizes(width, height);
        if (this._localSize[0] !== width || this._localSize[1] !== height) {
            this._createLocalTextures([cells[0], cells[1], LOCAL_BINS], blurred);
            this._localSize = [width, height];
        }
        const p = this._localPipelines!;
        if (!this._localBindGroups || this._localBoundInput !== input) {
            const group = (pipeline: GPUComputePipeline, entries: [number, GPUBindingResource][]) => device.createBindGroup({
                label: 'ToneMap/LocalExposureBG',
                layout: pipeline.getBindGroupLayout(0),
                entries: entries.map(([binding, resource]) => ({ binding, resource })),
            });
            const inputView = input.createView();
            const params = { buffer: this._params! };
            const sampler = this._sampler!;
            this._localBindGroups = [
                group(p.grid, [[0, inputView], [1, params], [2, sampler], [3, this._gridView!]]),
                group(p.logLuminance, [[0, inputView], [1, params], [2, sampler], [4, this._logViews[0]]]),
                group(p.blurX, [[1, params], [5, this._logViews[0]], [4, this._logViews[1]]]),
                group(p.blurY, [[1, params], [5, this._logViews[1]], [4, this._logViews[0]]]),
            ];
            this._localBoundInput = input;
        }
        const blurGroups: [number, number] = [divCeil(blurred[0], 8), divCeil(blurred[1], 8)];
        const passes: [GPUComputePipeline, [number, number]][] = [[p.grid, cells], [p.logLuminance, blurred], [p.blurX, blurGroups], [p.blurY, blurGroups]];
        const pass = commandEncoder.beginComputePass({ label: 'ToneMap/LocalExposure', timestampWrites: gpuPass('ToneMap/LocalExposure') });
        passes.forEach(([pipeline, [x, y]], i) => {
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, this._localBindGroups![i]);
            pass.dispatchWorkgroups(x, y);
        });
        pass.end();
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        _depth: GPUTexture,
        output: GPUTexture,
        _camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._pipeline) return;
        const device = this._device!;
        this._writeParams(width, height);
        this._frame = (this._frame + 1) >>> 0;
        device.queue.writeBuffer(this._params!, 0, this._paramsData);
        if (this.options.localExposure) {
            this._encodeLocalExposure(commandEncoder, input, width, height);
        }

        if (!this._bindGroup || this._boundInput !== input || this._boundOutput !== output) {
            this._bindGroup = device.createBindGroup({
                label: 'ToneMap/BG',
                layout: this._pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: input.createView() },
                    { binding: 1, resource: output.createView() },
                    { binding: 2, resource: { buffer: this._params! } },
                    { binding: 3, resource: this._sampler! },
                    { binding: 4, resource: this._gridView! },
                    { binding: 5, resource: this._logViews[0] },
                ],
            });
            this._boundInput = input;
            this._boundOutput = output;
        }
        const pass = commandEncoder.beginComputePass({ label: 'ToneMap', timestampWrites: gpuPass('ToneMap') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        pass.dispatchWorkgroups(divCeil(width, 8), divCeil(height, 8));
        pass.end();
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        // the GBuffer's textures are new: bind groups follow the next render's input and output
        this._bindGroup = null;
        this._localBindGroups = null;
    }

    destroy(): void {
        this._params?.destroy();
        for (const t of this._localTextures) t.destroy();
        this._params = null;
        this._localTextures = [];
        this._pipeline = null;
        this._localPipelines = null;
        this._bindGroup = null;
        this._localBindGroups = null;
        this._localSize = [0, 0];
        this.initialized = false;
    }
}

export { ToneMapEffect };
