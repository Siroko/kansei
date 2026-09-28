use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::math::Vec3;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = concat!(include_str!("../../shaders/tonemap_params.wgsl"), include_str!("../../shaders/tonemap.wgsl"));
const LOCAL_EXPOSURE_WGSL: &str = concat!(include_str!("../../shaders/tonemap_params.wgsl"), include_str!("../../shaders/local_exposure.wgsl"));

/// The lens attenuation q of Unreal Engine 5 (`r.EyeAdaptation.LensAttenuation`), with which
/// 1 cd/m² exposes to 1.0 at EV100 0: the ISO 12232 saturation constant 0.78 over q.
pub const LENS_ATTENUATION_UE5: f32 = 0.78;
/// Unreal Engine 4's (and Lagarde & de Rousiers', Moving Frostbite to PBR, 2014): 1 cd/m²
/// exposes to 1/1.2 at EV100 0.
pub const LENS_ATTENUATION_UE4: f32 = 0.65;

/// The linear exposure for a manual EV100, as Unreal Engine 5 exposes: `1 / 2^ev100`. Physical
/// light values (cd/m²) times this land in the tone curve's range; EV100 3.9 (dusk) gives 0.067.
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    exposure_from_ev100_lens(ev100, LENS_ATTENUATION_UE5)
}

/// The linear exposure for a manual EV100 through a lens of attenuation q: `q / 0.78 / 2^ev100`
/// (Unreal's `LuminanceMaxFromLensAttenuation`). `LENS_ATTENUATION_UE4` gives the older
/// `1 / (1.2 * 2^ev100)`.
pub fn exposure_from_ev100_lens(ev100: f32, lens_attenuation: f32) -> f32 {
    lens_attenuation.max(0.01) / 0.78 / 2f32.powf(ev100)
}

/// EV100 of a physical camera: `log2(N² / t * 100 / ISO)`, for an f-number, a shutter time in
/// seconds and an ISO sensitivity.
pub fn ev100_from_camera(f_number: f32, shutter_seconds: f32, iso: f32) -> f32 {
    (f_number * f_number / shutter_seconds * 100.0 / iso).log2()
}

/// The filmic tone curve that maps exposed scene-linear light to display-linear [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapper {
    /// Clamp only.
    None,
    /// ACES RRT + sRGB ODT, Stephen Hill's fit. UE's default filmic curve is tuned to match it.
    AcesFitted,
    /// AgX (Blender 4 default): graceful highlight desaturation, no hue skews in saturated lights.
    AgX,
    /// AgX with Blender's "Punchy" look.
    AgXPunchy,
    /// Khronos PBR Neutral: keeps base colours as authored up to 0.76, then a soft shoulder.
    KhronosNeutral,
    /// Unreal Engine's filmic display transform, as its tonemapper LUT computes it for an sRGB
    /// display (PostProcessCombineLUTs.usf, TonemapCommon.ush): Unreal's white balance, the
    /// colour into AP1 with its gamut expansion, the grade as Unreal's ColorCorrectAll in AP1 (with
    /// AP1 luma and its shadow, midtone and highlight weights), blue correction, `FilmToneMap`
    /// with `ToneMapOptions::unreal_film`'s curve, and back to sRGB. For scenes matched to Unreal.
    UnrealFilmic,
}

impl ToneMapper {
    fn gpu_id(self) -> u32 {
        match self {
            ToneMapper::None => 0,
            ToneMapper::AcesFitted => 1,
            ToneMapper::AgX => 2,
            ToneMapper::AgXPunchy => 3,
            ToneMapper::KhronosNeutral => 4,
            ToneMapper::UnrealFilmic => 5,
        }
    }
}

/// Colour grading on exposed scene-linear light, before the tone curve (where UE grades too).
/// The defaults are neutral.
#[derive(Debug, Clone, Copy)]
pub struct ColorGrade {
    /// Colour temperature (K) of the light that should read as white. 6500 is neutral; lower
    /// values neutralise warm light (the picture turns cooler).
    pub white_temperature: f32,
    /// Offset of that white from the Planckian locus, -1..1 (±0.02 Δuv). Positive treats a
    /// greener white as neutral, so the picture turns magenta.
    pub white_tint: f32,
    /// Power about middle grey (0.18): above 1 more contrast.
    pub contrast: f32,
    /// Per channel, as UE's ColorSaturation (its rgb times its w): each channel's distance from
    /// the luminance is scaled, so (0.9, 0.95, 1.0) desaturates reds most.
    pub saturation: Vec3,
    /// Per-channel multiplier.
    pub gain: Vec3,
    pub shadow_gain: Vec3,
    pub shadow_saturation: Vec3,
    pub highlight_gain: Vec3,
    pub highlight_saturation: Vec3,
    /// Luminance below which a pixel counts as shadow (fading out toward it).
    pub shadows_max: f32,
    /// Luminance above which a pixel starts to count as highlight (fully at 1).
    pub highlights_min: f32,
}

impl Default for ColorGrade {
    fn default() -> Self {
        Self {
            white_temperature: 6500.0,
            white_tint: 0.0,
            contrast: 1.0,
            saturation: Vec3::new(1.0, 1.0, 1.0),
            gain: Vec3::new(1.0, 1.0, 1.0),
            shadow_gain: Vec3::new(1.0, 1.0, 1.0),
            shadow_saturation: Vec3::new(1.0, 1.0, 1.0),
            highlight_gain: Vec3::new(1.0, 1.0, 1.0),
            highlight_saturation: Vec3::new(1.0, 1.0, 1.0),
            shadows_max: 0.09,
            highlights_min: 0.5,
        }
    }
}

/// Unreal's filmic curve and what surrounds it (its post-process settings `FilmSlope`, `FilmToe`,
/// `FilmShoulder`, `FilmBlackClip`, `FilmWhiteClip`, `BlueCorrection`, `ExpandGamut`), for
/// `ToneMapper::UnrealFilmic`. The defaults are Unreal's.
#[derive(Debug, Clone, Copy)]
pub struct UnrealFilm {
    pub slope: f32,
    pub toe: f32,
    pub shoulder: f32,
    pub black_clip: f32,
    pub white_clip: f32,
    /// How much of the blue correction for bright blue lights (0..1).
    pub blue_correction: f32,
    /// How far bright saturated colours are pushed out of the sRGB gamut toward AP1 (0..1).
    pub expand_gamut: f32,
}

impl Default for UnrealFilm {
    fn default() -> Self {
        Self { slope: 0.88, toe: 0.55, shoulder: 0.26, black_clip: 0.0, white_clip: 0.04, blue_correction: 0.6, expand_gamut: 1.0 }
    }
}

/// Unreal Engine 5's local exposure, its bilateral method (post-process settings
/// `LocalExposure*`): a factor on each pixel's light that scales the contrast of its
/// surroundings' luminance about middle grey, while keeping its detail against them. Bright
/// surroundings (a sky) come down and dark ones come up, as a photographer dodges and burns.
///
/// Each frame it builds Unreal's inputs from the picture: a bilateral grid of log luminance (cells
/// of 128 x 128 pixels, 32 bins of luminance), so the surroundings are the nearby pixels about as
/// bright as the pixel; and the log luminance at 1/32 of the picture's size, blurred. Middle grey
/// is 0.18 of exposed light, as with Unreal's manual metering. The defaults are Unreal's, which
/// change nothing; `unreal(highlight, shadow)` sets the two contrasts, as a project's
/// `r.DefaultFeature.LocalExposure.*ContrastScale` do.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LocalExposure {
    /// Contrast of the surroundings brighter than middle grey (1 keeps it, 0 flattens it).
    pub highlight_contrast: f32,
    /// Contrast of the surroundings darker than middle grey.
    pub shadow_contrast: f32,
    /// How much of a pixel's detail against its surroundings is kept.
    pub detail_strength: f32,
    /// How much the surroundings are the blurred luminance rather than the bilateral grid's.
    pub blurred_luminance_blend: f32,
    /// The blurred luminance's kernel, as a share of the picture's width (%).
    pub blurred_luminance_kernel_percent: f32,
    /// Stops on middle grey.
    pub middle_grey_bias: f32,
    /// The log2 scene luminance the grid's bins span (Unreal's histogram range, -10 to 20 with its
    /// extended luminance range).
    pub log_luminance_range: (f32, f32),
}

impl Default for LocalExposure {
    fn default() -> Self {
        Self {
            highlight_contrast: 1.0,
            shadow_contrast: 1.0,
            detail_strength: 1.0,
            blurred_luminance_blend: 0.6,
            blurred_luminance_kernel_percent: 50.0,
            middle_grey_bias: 0.0,
            log_luminance_range: (-10.0, 20.0),
        }
    }
}

impl LocalExposure {
    /// Unreal's defaults with these highlight and shadow contrasts.
    pub fn unreal(highlight_contrast: f32, shadow_contrast: f32) -> Self {
        Self { highlight_contrast, shadow_contrast, ..Default::default() }
    }
}

pub struct ToneMapOptions {
    pub tonemapper: ToneMapper,
    /// The curve of `ToneMapper::UnrealFilmic`.
    pub unreal_film: UnrealFilm,
    /// Linear multiplier on scene light; see [`exposure_from_ev100`].
    pub exposure: f32,
    /// Stops on top of `exposure` (per-shot trims).
    pub exposure_compensation: f32,
    pub grade: ColorGrade,
    /// Natural (cos⁴) vignetting, 0 = off: Unreal's `VignetteIntensity` and its formula, the
    /// tangent off the axis growing with the intensity to sqrt(2) times it at the frame's
    /// corners, so 0.5 darkens the corners to 0.44.
    pub vignette: f32,
    /// Lateral chromatic aberration, 0 = off; at 1 red and blue differ in magnification by 1 %.
    pub chromatic_aberration: f32,
    /// Film grain amplitude, 0 = off (UE's GrainIntensity scale).
    pub grain: f32,
    /// Grain size in pixels.
    pub grain_size: f32,
    /// Write sRGB-encoded values. Needed when the surface format is not sRGB (browsers give
    /// `Bgra8Unorm`), since the blit copies the chain's last texture as is; see [`Self::for_surface`].
    pub encode_srgb: bool,
    /// Triangular dither of one 8-bit step against banding.
    pub dither: bool,
    /// Unreal's local exposure; off when `None`.
    pub local_exposure: Option<LocalExposure>,
    /// The aspect (width / height) of the frame the picture is the centre crop of, when it is
    /// letterboxed: the vignette is the frame's, as Unreal's is its view's when a widget draws
    /// the letterbox over it (the Midsommar intro renders 16:9 behind a 2.39:1 one). `None`: the
    /// picture's own.
    pub frame_aspect: Option<f32>,
}

impl Default for ToneMapOptions {
    fn default() -> Self {
        Self {
            tonemapper: ToneMapper::AcesFitted,
            unreal_film: UnrealFilm::default(),
            exposure: 1.0,
            exposure_compensation: 0.0,
            grade: ColorGrade::default(),
            vignette: 0.0,
            chromatic_aberration: 0.0,
            grain: 0.0,
            grain_size: 1.6,
            encode_srgb: true,
            dither: true,
            local_exposure: None,
            frame_aspect: None,
        }
    }
}

impl ToneMapOptions {
    /// Defaults with `encode_srgb` set for a surface of this format
    /// (`renderer.presentation_format()`): encode unless the surface is sRGB already.
    pub fn for_surface(format: wgpu::TextureFormat) -> Self {
        Self { encode_srgb: !format.is_srgb(), ..Default::default() }
    }
}

// ── White balance ──

/// CIE 1931 xy of the Planckian locus at `kelvin` (Kim et al. 2002 cubic fit, 1667-25000 K).
fn planckian_xy(kelvin: f32) -> glam::Vec2 {
    let t = kelvin.clamp(1667.0, 25000.0) as f64;
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    glam::Vec2::new(x as f32, y as f32)
}

fn xy_to_uv(xy: glam::Vec2) -> glam::Vec2 {
    let d = -2.0 * xy.x + 12.0 * xy.y + 3.0;
    glam::Vec2::new(4.0 * xy.x / d, 6.0 * xy.y / d)
}

fn uv_to_xy(uv: glam::Vec2) -> glam::Vec2 {
    let d = 2.0 * uv.x - 8.0 * uv.y + 4.0;
    glam::Vec2::new(3.0 * uv.x / d, 2.0 * uv.y / d)
}

/// White point for a temperature and a tint: the Planckian locus, moved `tint * 0.02` along its
/// normal in CIE 1960 uv (toward green for positive tints).
fn white_point_xy(kelvin: f32, tint: f32) -> glam::Vec2 {
    let uv = xy_to_uv(planckian_xy(kelvin));
    let tangent = xy_to_uv(planckian_xy(kelvin * 1.01)) - xy_to_uv(planckian_xy(kelvin * 0.99));
    // toward lower temperatures the locus runs to +u, +v; its left normal points to +v (green)
    let t = -tangent.normalize_or_zero();
    let normal = glam::Vec2::new(-t.y, t.x);
    uv_to_xy(uv + normal * (tint * 0.02))
}

/// Linear sRGB (D65) to CIE XYZ.
const SRGB_TO_XYZ: glam::Mat3 = glam::Mat3::from_cols(
    glam::Vec3::new(0.412_456_4, 0.212_672_9, 0.019_333_9),
    glam::Vec3::new(0.357_576_1, 0.715_152_2, 0.119_192),
    glam::Vec3::new(0.180_437_5, 0.072_175, 0.950_304),
);

fn xy_to_xyz(xy: glam::Vec2) -> glam::Vec3 {
    glam::Vec3::new(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y)
}

/// Linear-sRGB matrix that adapts white(`kelvin`, `tint`) to white(6500 K, 0) with the Bradford
/// transform, so 6500/0 is the identity.
pub fn white_balance_matrix(kelvin: f32, tint: f32) -> glam::Mat3 {
    let srgb_to_xyz = SRGB_TO_XYZ;
    let bradford = glam::Mat3::from_cols(
        glam::Vec3::new(0.8951, -0.7502, 0.0389),
        glam::Vec3::new(0.2664, 1.7135, -0.0685),
        glam::Vec3::new(-0.1614, 0.0367, 1.0296),
    );
    let src = bradford * xy_to_xyz(white_point_xy(kelvin, tint));
    let dst = bradford * xy_to_xyz(white_point_xy(6500.0, 0.0));
    let scale = glam::Mat3::from_diagonal(dst / src);
    srgb_to_xyz.inverse() * bradford.inverse() * scale * bradford * srgb_to_xyz
}

// ── Unreal's white balance (TonemapCommon.ush) ──

fn unreal_d_illuminant_xy(t: f64) -> glam::DVec2 {
    let t = t * 1.4388 / 1.438;
    let o = 1.0 / t;
    let x = if t <= 7000.0 {
        0.244063 + (0.09911e3 + (2.9678e6 - 4.6070e9 * o) * o) * o
    } else {
        0.237040 + (0.24748e3 + (1.9018e6 - 2.0064e9 * o) * o) * o
    };
    glam::DVec2::new(x, -3.0 * x * x + 2.87 * x - 0.275)
}

fn unreal_planckian_uv(t: f64) -> glam::DVec2 {
    let u = (0.860117757 + 1.54118254e-4 * t + 1.28641212e-7 * t * t) / (1.0 + 8.42420235e-4 * t + 7.08145163e-7 * t * t);
    let v = (0.317398726 + 4.22806245e-5 * t + 4.20481691e-8 * t * t) / (1.0 - 2.89741816e-5 * t + 1.61456053e-7 * t * t);
    glam::DVec2::new(u, v)
}

fn unreal_uv_to_xy(uv: glam::DVec2) -> glam::DVec2 {
    let d = 2.0 * uv.x - 8.0 * uv.y + 4.0;
    glam::DVec2::new(3.0 * uv.x / d, 2.0 * uv.y / d)
}

/// The Planckian locus at `t`, moved `tint * 0.05` along its isotherm (Unreal's
/// `PlanckianIsothermal`).
fn unreal_planckian_isothermal_xy(t: f64, tint: f64) -> glam::DVec2 {
    let uv = unreal_planckian_uv(t);
    let ud = (-1.13758118e9 - 1.91615621e6 * t - 1.53177 * t * t) / (1.41213984e6 + 1189.62 * t + t * t).powi(2);
    let vd = (1.97471536e9 - 705674.0 * t - 308.607 * t * t) / (6.19363586e6 - 179.456 * t + t * t).powi(2);
    let n = glam::DVec2::new(ud, vd).normalize();
    unreal_uv_to_xy(uv + glam::DVec2::new(n.y, -n.x) * (tint * 0.05))
}

/// Linear-sRGB matrix of Unreal's temperature white balance (`WhiteBalance`): the white of
/// `kelvin` (a daylight illuminant from 4000 K, the Planckian locus below), moved along its
/// isotherm by `tint`, adapted to D65 with the Bradford transform. Its tint runs the same way as
/// `white_balance_matrix`'s (negative: the picture turns green), 0.05 in CIE 1960 uv per unit
/// rather than 0.02.
pub fn unreal_white_balance_matrix(kelvin: f32, tint: f32) -> glam::Mat3 {
    let (t, tint) = (kelvin as f64, tint as f64);
    let locus = unreal_uv_to_xy(unreal_planckian_uv(t));
    let src = if t < 4000.0 { locus } else { unreal_d_illuminant_xy(t) } + (unreal_planckian_isothermal_xy(t, tint) - locus);
    let xyz = |xy: glam::DVec2| glam::DVec3::new(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y);
    let bradford = glam::DMat3::from_cols(
        glam::DVec3::new(0.8951, -0.7502, 0.0389),
        glam::DVec3::new(0.2664, 1.7135, -0.0685),
        glam::DVec3::new(-0.1614, 0.0367, 1.0296),
    );
    let bradford_inv = glam::DMat3::from_cols(
        glam::DVec3::new(0.9869929, 0.4323053, -0.0085287),
        glam::DVec3::new(-0.1470543, 0.5183603, 0.0400428),
        glam::DVec3::new(0.1599627, 0.0492912, 0.9684867),
    );
    let (s, d) = (bradford * xyz(src), bradford * xyz(glam::DVec2::new(0.31270, 0.32900)));
    let cat = bradford_inv * glam::DMat3::from_diagonal(d / s) * bradford;
    let srgb_to_xyz = glam::DMat3::from_cols(
        glam::DVec3::new(0.4123907993, 0.2126390059, 0.0193308187),
        glam::DVec3::new(0.3575843394, 0.7151686788, 0.1191947798),
        glam::DVec3::new(0.1804807884, 0.0721923154, 0.9505321522),
    );
    let xyz_to_srgb = glam::DMat3::from_cols(
        glam::DVec3::new(3.2409699419, -0.9692436363, 0.0556300797),
        glam::DVec3::new(-1.5373831776, 1.8759675015, -0.2039769589),
        glam::DVec3::new(-0.4986107603, 0.0415550574, 1.0569715142),
    );
    (xyz_to_srgb * cat * srgb_to_xyz).as_mat3()
}

// ── GPU layout (must match ToneMapParams in tonemap.wgsl) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ToneMapParamsGpu {
    white_balance: [[f32; 4]; 3],
    gain: [f32; 3],
    exposure: f32,
    shadow_gain: [f32; 3],
    contrast: f32,
    highlight_gain: [f32; 3],
    shadows_max: f32,
    saturation: [f32; 3],
    highlights_min: f32,
    shadow_saturation: [f32; 3],
    vignette: f32,
    highlight_saturation: [f32; 3],
    chromatic_aberration: f32,
    grain: f32,
    grain_size: f32,
    width: u32,
    height: u32,
    frame: u32,
    tonemapper: u32,
    flags: u32,
    frame_aspect: f32,
    film: [f32; 4],
    film2: [f32; 4],
    local_exposure: [f32; 4],
    local_exposure2: [f32; 4],
    local_exposure3: [f32; 4],
}

const FLAG_ENCODE_SRGB: u32 = 1;
const FLAG_DITHER: u32 = 2;
const FLAG_LOCAL_EXPOSURE: u32 = 4;
/// Half-resolution texels per side of a bilateral grid cell (local_exposure.wgsl's LOCAL_CELL).
const LOCAL_CELL: u32 = 64;
/// Bins of the grid.
const LOCAL_BINS: u32 = 32;

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    local: LocalGpu,
}

/// Local exposure's pipelines, and its textures for a picture size (a 1-texel stand-in until it
/// is on).
struct LocalGpu {
    grid: wgpu::ComputePipeline,
    log_luminance: wgpu::ComputePipeline,
    blur_x: wgpu::ComputePipeline,
    blur_y: wgpu::ComputePipeline,
    size: (u32, u32),
    grid_view: wgpu::TextureView,
    /// The log luminance, and the blur's intermediate
    log: [wgpu::TextureView; 2],
}

/// The bilateral grid's cells and the blurred luminance's texels for a picture size.
fn local_sizes(width: u32, height: u32) -> ((u32, u32), (u32, u32)) {
    let half = (width.div_ceil(2), height.div_ceil(2));
    ((half.0.div_ceil(LOCAL_CELL), half.1.div_ceil(LOCAL_CELL)), (width.div_ceil(32), height.div_ceil(32)))
}

/// The display transform (K2): physical exposure, lens (chromatic aberration, vignette), a
/// scene-linear grade, a filmic tone curve, film grain, dither and sRGB encoding.
///
/// It turns scene-linear HDR into the display signal, so it goes last in the HDR chain: after
/// fog, anti-aliasing, depth of field and bloom, which all want linear light. A display-space
/// `ColorGradingEffect` may follow it.
///
/// ```ignore
/// let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
/// options.exposure = exposure_from_ev100(3.9);
/// options.vignette = 0.5;
/// options.grain = 0.22;
/// options.chromatic_aberration = 0.25;
/// let volume = PostProcessingVolume::new(&renderer, vec![Box::new(fog), Box::new(bloom), Box::new(ToneMapEffect::new(options))]);
/// ```
pub struct ToneMapEffect {
    pub options: ToneMapOptions,
    frame: u32,
    gpu: Option<Gpu>,
}

impl ToneMapEffect {
    pub fn new(options: ToneMapOptions) -> Self {
        Self { options, frame: 0, gpu: None }
    }

    /// The linear factor applied to scene light: exposure times 2^compensation. Pass it to
    /// `BloomEffect::exposure` so the bloom threshold is in the same units.
    pub fn total_exposure(&self) -> f32 {
        self.options.exposure * 2f32.powf(self.options.exposure_compensation)
    }

    fn params(&self, width: u32, height: u32) -> ToneMapParamsGpu {
        let o = &self.options;
        let g = &o.grade;
        let wb = if o.tonemapper == ToneMapper::UnrealFilmic {
            unreal_white_balance_matrix(g.white_temperature, g.white_tint)
        } else {
            white_balance_matrix(g.white_temperature, g.white_tint)
        };
        let f = &o.unreal_film;
        let col = |v: glam::Vec3| [v.x, v.y, v.z, 0.0];
        let v3 = |v: Vec3| [v.x, v.y, v.z];
        ToneMapParamsGpu {
            white_balance: [col(wb.x_axis), col(wb.y_axis), col(wb.z_axis)],
            gain: v3(g.gain),
            exposure: self.total_exposure(),
            shadow_gain: v3(g.shadow_gain),
            contrast: g.contrast,
            highlight_gain: v3(g.highlight_gain),
            shadows_max: g.shadows_max.max(1e-4),
            saturation: v3(g.saturation),
            highlights_min: g.highlights_min.min(0.999),
            shadow_saturation: v3(g.shadow_saturation),
            vignette: o.vignette.max(0.0),
            highlight_saturation: v3(g.highlight_saturation),
            chromatic_aberration: o.chromatic_aberration.max(0.0),
            grain: o.grain.max(0.0),
            grain_size: o.grain_size.max(1.0),
            width,
            height,
            frame: self.frame,
            tonemapper: o.tonemapper.gpu_id(),
            flags: if o.encode_srgb { FLAG_ENCODE_SRGB } else { 0 } | if o.dither { FLAG_DITHER } else { 0 } | if o.local_exposure.is_some() { FLAG_LOCAL_EXPOSURE } else { 0 },
            frame_aspect: o.frame_aspect.unwrap_or(0.0).max(0.0),
            film: [f.slope, f.toe, f.shoulder, f.black_clip],
            film2: [f.white_clip, f.blue_correction.clamp(0.0, 1.0), f.expand_gamut.max(0.0), 1.0],
            ..self.local_params(width, height)
        }
    }

    /// Local exposure's parameters (and its flag) for a picture size; zeros when it is off.
    fn local_params(&self, width: u32, height: u32) -> ToneMapParamsGpu {
        let Some(le) = self.options.local_exposure else { return ToneMapParamsGpu::zeroed() };
        let (log_min, log_max) = le.log_luminance_range;
        let scale = 1.0 / (log_max - log_min).max(1e-3);
        let half = (width.div_ceil(2) as f32, height.div_ceil(2) as f32);
        let (cells, blurred) = local_sizes(width, height);
        // Unreal's Gaussian (PostProcessWeightedSampleSum): a radius of half the kernel's share of
        // the blurred texture's width, at most 31 texels
        let radius = blurred.0 as f32 * le.blurred_luminance_kernel_percent * 0.01 * 0.5;
        let radius = radius.clamp(1e-3, 31.0);
        ToneMapParamsGpu {
            flags: FLAG_LOCAL_EXPOSURE,
            local_exposure: [le.highlight_contrast, le.shadow_contrast, le.detail_strength, le.blurred_luminance_blend.clamp(0.0, 1.0)],
            local_exposure2: [(0.18f32).log2() + le.middle_grey_bias, scale, -log_min * scale, log_min],
            local_exposure3: [
                half.0 / LOCAL_CELL as f32 / cells.0 as f32,
                half.1 / LOCAL_CELL as f32 / cells.1 as f32,
                radius,
                radius.ceil().min(31.0),
            ],
            ..ToneMapParamsGpu::zeroed()
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToneMap/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(2, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(4, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                }),
                entry(5, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ToneMap/Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ToneMap/Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ToneMap/Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ToneMap/Params"),
            size: std::mem::size_of::<ToneMapParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ToneMap/LocalExposure"), source: wgpu::ShaderSource::Wgsl(LOCAL_EXPOSURE_WGSL.into()) });
        let local_pipeline = |entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ToneMap/LocalExposure"),
                layout: None,
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let (grid_view, log) = Self::local_textures(device, (1, 1, 1), (1, 1));
        let local = LocalGpu {
            grid: local_pipeline("grid"),
            log_luminance: local_pipeline("logLuminance"),
            blur_x: local_pipeline("blurX"),
            blur_y: local_pipeline("blurY"),
            size: (0, 0),
            grid_view,
            log,
        };
        self.gpu = Some(Gpu { pipeline, bgl, params, sampler, local });
    }

    /// Local exposure's grid (`cells` and its bins) and the blurred luminance's two textures.
    fn local_textures(device: &wgpu::Device, grid: (u32, u32, u32), blurred: (u32, u32)) -> (wgpu::TextureView, [wgpu::TextureView; 2]) {
        let texture = |label: &str, size: wgpu::Extent3d, dimension| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let flat = wgpu::Extent3d { width: blurred.0, height: blurred.1, depth_or_array_layers: 1 };
        (
            texture("ToneMap/LocalExposureGrid", wgpu::Extent3d { width: grid.0, height: grid.1, depth_or_array_layers: grid.2 }, wgpu::TextureDimension::D3),
            [texture("ToneMap/LocalExposureLog", flat, wgpu::TextureDimension::D2), texture("ToneMap/LocalExposureBlur", flat, wgpu::TextureDimension::D2)],
        )
    }

    /// Local exposure's grid and blurred luminance for this frame's input (`params` already
    /// written): the grid, the log luminance at 1/32, and its blur across then down.
    fn encode_local_exposure(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, input: &wgpu::TextureView, width: u32, height: u32) {
        let gpu = self.gpu.as_mut().unwrap();
        let (cells, blurred) = local_sizes(width, height);
        if gpu.local.size != (width, height) {
            (gpu.local.grid_view, gpu.local.log) = Self::local_textures(device, (cells.0, cells.1, LOCAL_BINS), blurred);
            gpu.local.size = (width, height);
        }
        let local = &gpu.local;
        let group = |pipeline: &wgpu::ComputePipeline, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<_> = entries.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ToneMap/LocalExposureBG"), layout: &pipeline.get_bind_group_layout(0), entries: &entries })
        };
        let tex = wgpu::BindingResource::TextureView;
        let params = gpu.params.as_entire_binding();
        let sampler = wgpu::BindingResource::Sampler(&gpu.sampler);
        let passes = [
            (&local.grid, group(&local.grid, &[(0, tex(input)), (1, params.clone()), (2, sampler.clone()), (3, tex(&local.grid_view))]), cells),
            (&local.log_luminance, group(&local.log_luminance, &[(0, tex(input)), (1, params.clone()), (2, sampler), (4, tex(&local.log[0]))]), blurred),
            (&local.blur_x, group(&local.blur_x, &[(1, params.clone()), (5, tex(&local.log[0])), (4, tex(&local.log[1]))]), (blurred.0.div_ceil(8), blurred.1.div_ceil(8))),
            (&local.blur_y, group(&local.blur_y, &[(1, params), (5, tex(&local.log[1])), (4, tex(&local.log[0]))]), (blurred.0.div_ceil(8), blurred.1.div_ceil(8))),
        ];
        let stamp = crate::profiling::gpu_pass("ToneMap/LocalExposure");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ToneMap/LocalExposure"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        for (pipeline, bind_group, (x, y)) in &passes {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*x, *y, 1);
        }
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        WGSL
    }

    #[cfg(test)]
    pub(crate) fn local_exposure_source() -> &'static str {
        LOCAL_EXPOSURE_WGSL
    }
}

impl PostProcessingEffect for ToneMapEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        _camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        let params = self.params(width, height);
        self.frame = self.frame.wrapping_add(1);
        queue.write_buffer(&self.gpu.as_ref().unwrap().params, 0, bytemuck::bytes_of(&params));
        if self.options.local_exposure.is_some() {
            self.encode_local_exposure(device, encoder, input, width, height);
        }
        let gpu = self.gpu.as_ref().unwrap();

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ToneMap/BG"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&gpu.local.grid_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&gpu.local.log[0]) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ToneMap"), timestamp_writes: crate::profiling::gpu_pass("ToneMap").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_params_layout_matches() {
        let code = ToneMapEffect::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "ToneMapParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<ToneMapParamsGpu>());
        let code = ToneMapEffect::local_exposure_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("local_exposure: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("local_exposure: {e:?}"));
    }

    /// Local exposure's GPU cost at 1920 x 1080, by wall clock (ten frames per submit, the two
    /// alternated): `cargo test -p kansei-core --lib time_local_exposure -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_local_exposure() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (1920u32, 1080u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let output = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
        let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        let mut effects = [None, Some(LocalExposure::unreal(0.8, 0.8))].map(|local| ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::UnrealFilmic, local_exposure: local, ..Default::default() }));
        let wall = |fx: &mut ToneMapEffect| {
            let mut e = device.create_command_encoder(&Default::default());
            for _ in 0..10 {
                fx.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
            }
            let t = std::time::Instant::now();
            queue.submit([e.finish()]);
            device.poll(wgpu::Maintain::Wait);
            t.elapsed().as_secs_f64() * 1e3 / 10.0
        };
        let mut times = [Vec::new(), Vec::new()];
        for round in 0..40 {
            for (k, fx) in effects.iter_mut().enumerate() {
                let ms = wall(fx);
                if round >= 5 {
                    times[k].push(ms);
                }
            }
        }
        for t in &mut times {
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
        }
        eprintln!("tonemap {:.3} ms, with local exposure {:.3} ms (medians per frame, {w}x{h})", times[0][times[0].len() / 2], times[1][times[1].len() / 2]);
    }

    /// Unreal's local exposure on a sky 2 stops over middle grey beside ground 4 stops under it.
    /// With the bilateral grid alone each side's surroundings are itself, up to their shared edge,
    /// so each is scaled by 2^((contrast - 1) x its stops from middle grey). With the blurred
    /// luminance blended in, as Unreal's defaults do, the surroundings follow a fine reference of
    /// the blur.
    #[test]
    fn local_exposure_follows_unreal_s() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (256u32, 128u32);
        // exposure 1: exposed light is scene light; the inputs as the half floats they are stored as
        let (sky, ground) = (f16_to_f32(f32_to_f16(0.18 * 4.0)), f16_to_f32(f32_to_f16(0.18 / 16.0)));
        let value = |x: u32| if x < w / 2 { sky } else { ground };
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING);
        let halves: Vec<u16> = (0..w * h).flat_map(|i| { let v = f32_to_f16(value(i % w)); [v, v, v, f32_to_f16(1.0)] }).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let input = input_tex.create_view(&Default::default());
        let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let output = output_tex.create_view(&Default::default());
        let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        let run = |local: LocalExposure| -> Vec<f32> {
            let mut fx = ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::None, encode_srgb: false, dither: false, local_exposure: Some(local), ..Default::default() });
            let mut e = device.create_command_encoder(&Default::default());
            fx.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (w * h * 8) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            // the green channel of each pixel
            bytemuck::cast_slice::<u8, u16>(&data).chunks(4).map(|px| f16_to_f32(px[1])).collect()
        };
        let middle_grey = 0.18f32.log2();
        let factor = |y: f32, base: f32, contrast: f32| (middle_grey + (base - middle_grey) * contrast + (y - base) - y).exp2();
        let columns = [4u32, 100, 126, 127, 128, 129, 160, 250];

        // the bilateral grid alone
        let out = run(LocalExposure { blurred_luminance_blend: 0.0, ..LocalExposure::unreal(0.8, 0.6) });
        for x in columns {
            let (v, contrast) = if x < w / 2 { (sky, 0.8) } else { (ground, 0.6) };
            let want = v * factor(v.log2(), v.log2(), contrast);
            let got = out[(64 * w + x) as usize];
            assert!((got / want - 1.0).abs() < 0.01, "grid alone, column {x}: {got} vs {want}");
        }

        // Unreal's blend of the blurred luminance: its reference, the log luminance at 1/32 (8 x 4
        // texels, each a side's own), blurred across then down with Unreal's Gaussian (radius 2
        // texels, 2 taps), mirrored, then sampled bilinearly at each pixel
        let (bw, bh) = (w / 32, h / 32);
        let mut log: Vec<f32> = (0..bw * bh).map(|i| value((i % bw) * 32).log2()).collect();
        let radius = bw as f32 * 50.0 * 0.01 * 0.5;
        let mirror = |c: i32, n: i32| (if c < 0 { -c - 1 } else if c >= n { 2 * n - c - 1 } else { c }).clamp(0, n - 1);
        for axis in [(1i32, 0i32), (0, 1)] {
            log = (0..bw * bh)
                .map(|i| {
                    let (x, y) = ((i % bw) as i32, (i / bw) as i32);
                    let (mut sum, mut weights) = (0.0, 0.0);
                    for k in -2i32..=2 {
                        let wk = (-16.7 * (k as f32 / radius).powi(2)).exp();
                        let (sx, sy) = (mirror(x + axis.0 * k, bw as i32), mirror(y + axis.1 * k, bh as i32));
                        sum += wk * log[(sy as u32 * bw + sx as u32) as usize];
                        weights += wk;
                    }
                    sum / weights
                })
                .collect();
        }
        let blurred_at = |x: u32, y: u32| {
            let (u, v) = ((x as f32 + 0.5) / w as f32 * bw as f32 - 0.5, (y as f32 + 0.5) / h as f32 * bh as f32 - 0.5);
            let at = |i: i32, j: i32| log[(j.clamp(0, bh as i32 - 1) as u32 * bw + i.clamp(0, bw as i32 - 1) as u32) as usize];
            let (i, j) = (u.floor() as i32, v.floor() as i32);
            let (fu, fv) = (u - u.floor(), v - v.floor());
            (at(i, j) * (1.0 - fu) + at(i + 1, j) * fu) * (1.0 - fv) + (at(i, j + 1) * (1.0 - fu) + at(i + 1, j + 1) * fu) * fv
        };
        let out = run(LocalExposure::unreal(0.8, 0.6));
        for x in columns {
            let v = value(x);
            let base = v.log2() + (blurred_at(x, 64) - v.log2()) * 0.6;
            let contrast = if base > middle_grey { 0.8 } else { 0.6 };
            let want = v * factor(v.log2(), base, contrast);
            let got = out[(64 * w + x) as usize];
            eprintln!("column {x}: {got:.5} (reference {want:.5}, input {v:.5})");
            assert!((got / want - 1.0).abs() < 0.015, "blended, column {x}: {got} vs {want}");
        }

        // Unreal's defaults change nothing
        let out = run(LocalExposure::default());
        for x in columns {
            assert!((out[(64 * w + x) as usize] / value(x) - 1.0).abs() < 2e-3, "defaults, column {x}");
        }
    }

    #[test]
    fn exposure_matches_ue_calibration() {
        // UE 5's manual exposure at EV100 3.9 (the Midsommar intro): 1 / 2^3.9; 1 cd/m^2 is 1.0 at 0
        assert!((exposure_from_ev100(3.9) - 0.066986).abs() < 1e-5);
        assert!((exposure_from_ev100(0.0) - 1.0).abs() < 1e-6);
        // UE 4's lens: 1 / (1.2 * 2^3.9)
        assert!((exposure_from_ev100_lens(3.9, LENS_ATTENUATION_UE4) - 0.05583).abs() < 1e-4);
        // f/1.4, 1/60 s, ISO 100 is EV100 ~6.9
        assert!((ev100_from_camera(1.4, 1.0 / 60.0, 100.0) - 6.878).abs() < 0.001);
        // doubling ISO opens one stop
        let a = ev100_from_camera(8.0, 1.0 / 125.0, 100.0);
        let b = ev100_from_camera(8.0, 1.0 / 125.0, 200.0);
        assert!((a - b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn white_balance_is_identity_at_6500_and_neutralises_warm_white() {
        let id = white_balance_matrix(6500.0, 0.0);
        assert!(id.abs_diff_eq(glam::Mat3::IDENTITY, 1e-5), "{id:?}");

        // the linear-sRGB colour of a 3200 K white, scaled to luminance 1
        let xyz = xy_to_xyz(white_point_xy(3200.0, 0.0));
        let srgb_to_xyz = SRGB_TO_XYZ;
        let warm = srgb_to_xyz.inverse() * xyz;
        assert!(warm.x > warm.z * 1.5, "3200 K white should be orange: {warm:?}");
        // balanced for 3200 K it becomes the 6500 K white, which is near-neutral in sRGB
        let balanced = white_balance_matrix(3200.0, 0.0) * warm;
        let neutral = srgb_to_xyz.inverse() * xy_to_xyz(white_point_xy(6500.0, 0.0));
        assert!((balanced - neutral).abs().max_element() < 1e-3, "{balanced:?} vs {neutral:?}");
        assert!((neutral.x / neutral.z - 1.0).abs() < 0.1, "{neutral:?}");
    }

    #[test]
    fn tint_moves_white_off_the_locus_toward_green() {
        let on = white_point_xy(5000.0, 0.0);
        let green = white_point_xy(5000.0, 1.0);
        let magenta = white_point_xy(5000.0, -1.0);
        assert!(green.y > on.y && magenta.y < on.y, "{green:?} {on:?} {magenta:?}");
        let duv = (xy_to_uv(green) - xy_to_uv(on)).length();
        assert!((duv - 0.02).abs() < 1e-4);
    }

    /// The vignette is Unreal's (PostProcessCommon.ush's VignetteSpace and ComputeVignetteMask,
    /// transcribed): a 16:9 frame's corners fall to 0.44 at intensity 0.5, and a 2.39:1 picture
    /// cropped from that frame shows the frame's vignette.
    #[test]
    fn vignette_is_unreal_s() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // Unreal: the frame's viewport position in [-1, 1], height / width a
        let unreal = |x: f32, y: f32, a: f32, intensity: f32| {
            let scale = 2f32.sqrt() / (1.0 + a * a).sqrt();
            let (px, py) = (x * scale * intensity, y * a * scale * intensity);
            (1.0 / (px * px + py * py + 1.0)).powi(2)
        };
        let camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        for (w, h, frame) in [(160u32, 90u32, None), (239, 100, Some(16.0f32 / 9.0))] {
            let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
            let input_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING);
            let halves: Vec<u16> = (0..w * h).flat_map(|_| [f32_to_f16(0.5), f32_to_f16(0.5), f32_to_f16(0.5), f32_to_f16(1.0)]).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                bytemuck::cast_slice(&halves),
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            let input = input_tex.create_view(&Default::default());
            let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
            let output = output_tex.create_view(&Default::default());
            let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
            let gbuffer = GBuffer::new(&device, w, h, 1);
            let mut fx = ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::None, vignette: 0.5, encode_srgb: false, dither: false, frame_aspect: frame, ..Default::default() });
            let mut e = device.create_command_encoder(&Default::default());
            fx.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
            let row = (w * 8).div_ceil(256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            let px: &[u16] = bytemuck::cast_slice(&data);
            let frame_aspect = frame.unwrap_or(w as f32 / h as f32);
            let band = frame_aspect / (w as f32 / h as f32);
            for (x, y) in [(w / 2, h / 2), (w - 1, h / 2), (w - 1, 0), (0, h - 1), (w / 4, h / 4)] {
                let got = f16_to_f32(px[((y * row / 2) + x * 4) as usize]) / 0.5;
                let (nx, ny) = (((x as f32 + 0.5) / w as f32) * 2.0 - 1.0, (((y as f32 + 0.5) / h as f32) * 2.0 - 1.0) * band);
                let want = unreal(nx, ny, 1.0 / frame_aspect, 0.5);
                assert!((got - want).abs() < 2e-3, "{w}x{h}, frame {frame:?}, pixel ({x}, {y}): {got} vs Unreal's {want}");
            }
            if frame.is_none() {
                // the frame's corner itself
                assert!((unreal(1.0, 1.0, h as f32 / w as f32, 0.5) - 0.4444).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn params_pack_options() {
        let mut fx = ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::AgX, exposure: 0.5, exposure_compensation: 1.0, ..ToneMapOptions::for_surface(wgpu::TextureFormat::Bgra8UnormSrgb) });
        let p = fx.params(640, 360);
        assert_eq!(p.exposure, 1.0);
        assert_eq!(p.tonemapper, 2);
        assert_eq!(p.flags, FLAG_DITHER); // sRGB surface: no encode
        fx.options.encode_srgb = true;
        fx.options.dither = false;
        assert_eq!(fx.params(640, 360).flags, FLAG_ENCODE_SRGB);
    }

    /// Unreal's white balance matrix, against a transcription of TonemapCommon.ush's
    /// `WhiteBalance` (the midsommar-twilight-sky scout's ue_display.py).
    #[test]
    fn unreal_white_balance_matches_unreal_s() {
        let cases: [(f32, f32, [[f32; 3]; 3]); 2] = [
            (6500.0, -0.18, [[0.963995, -0.08145, -0.00001], [0.004105, 1.039241, 0.002786], [-0.001346, -0.008964, 0.918815]]),
            (5000.0, 0.3, [[0.986126, 0.015607, -0.027377], [-0.001241, 0.968687, -0.015809], [0.015463, 0.064156, 1.692414]]),
        ];
        for (kelvin, tint, rows) in cases {
            let m = unreal_white_balance_matrix(kelvin, tint);
            for (i, row) in rows.iter().enumerate() {
                for (j, want) in row.iter().enumerate() {
                    let got = m.col(j)[i];
                    assert!((got - want).abs() < 2e-5, "{kelvin} K, tint {tint}: [{i}][{j}] {got} vs {want}");
                }
            }
        }
        assert!((unreal_white_balance_matrix(6500.0, 0.0) - glam::Mat3::IDENTITY).abs_diff_eq(glam::Mat3::ZERO, 2e-3));
        // a negative tint turns the picture green, as white_balance_matrix's does, more strongly
        let (unreal, kansei) = (unreal_white_balance_matrix(6500.0, -0.18) * glam::Vec3::ONE, white_balance_matrix(6500.0, -0.18) * glam::Vec3::ONE);
        assert!(unreal.y > unreal.x && unreal.y > unreal.z && kansei.y > kansei.x && kansei.y > kansei.z);
        assert!(unreal.y - unreal.x > kansei.y - kansei.x, "{unreal} {kansei}");
    }

    /// `ToneMapper::UnrealFilmic` displays scene colours as Unreal's tonemapper LUT does, neutral
    /// and with the Midsommar intro's grade, against a transcription of PostProcessCombineLUTs.usf
    /// and TonemapCommon.ush (the midsommar-twilight-sky scout's ue_display.py).
    #[test]
    fn unreal_filmic_matches_unreal_s_lut() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let inputs: [[f32; 3]; 10] = [
            [0.005; 3], [0.02; 3], [0.1; 3], [0.18; 3], [1.0; 3], [4.0; 3],
            [0.0, 0.0, 0.3], [0.3, 0.05, 0.02], [0.05, 0.2, 0.08], [2.0, 1.0, 0.3],
        ];
        let neutral: [[f32; 3]; 10] = [
            [0.000511, 0.000511, 0.000511], [0.005355, 0.005355, 0.005355], [0.075913, 0.075912, 0.075913], [0.180001, 0.179999, 0.18],
            [0.723365, 0.723357, 0.723359], [0.944162, 0.944152, 0.944155], [0.0, 0.0, 0.300718], [0.267275, 0.027444, 0.008443],
            [0.02144, 0.200903, 0.060531], [0.915832, 0.742783, 0.408637],
        ];
        let intro: [[f32; 3]; 10] = [
            [0.000141, 0.000384, 0.000325], [0.001811, 0.004662, 0.003921], [0.044192, 0.076682, 0.062223], [0.117805, 0.190485, 0.158558],
            [0.680856, 0.75487, 0.736746], [0.937129, 0.961538, 0.955643], [0.0, 0.0, 0.208206], [0.169866, 0.040124, 0.015121],
            [0.013544, 0.198433, 0.065892], [0.777586, 0.783623, 0.719583],
        ];
        // the intro's grade (create_intro_scene.py): Unreal's vec4s as their rgb times their w
        let intro_grade = ColorGrade {
            white_temperature: 6500.0,
            white_tint: -0.18,
            contrast: 1.06,
            saturation: Vec3::new(0.9 * 0.78, 0.95 * 0.78, 0.78),
            gain: Vec3::new(0.88, 1.0, 0.98),
            shadow_gain: Vec3::new(0.9, 1.0, 1.04),
            shadow_saturation: Vec3::new(1.0, 1.0, 1.0),
            highlight_gain: Vec3::new(0.97, 1.0, 1.0),
            highlight_saturation: Vec3::new(0.4, 0.4, 0.4),
            shadows_max: 0.09,
            highlights_min: 0.5,
        };
        let (w, h) = (inputs.len() as u32, 1u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let texels: Vec<f32> = inputs.iter().flat_map(|c| [c[0], c[1], c[2], 1.0]).collect();
        // the effect's input is filterable: half floats
        let half_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let halves: Vec<u16> = texels.iter().map(|&v| f32_to_f16(v)).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &half_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&halves),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 8), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let input = half_tex.create_view(&Default::default());
        let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let output = output_tex.create_view(&Default::default());
        let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let camera = Camera::new(60.0, 0.1, 100.0, 1.0);
        for (name, grade, want) in [("neutral", ColorGrade::default(), neutral), ("intro", intro_grade, intro)] {
            let mut fx = ToneMapEffect::new(ToneMapOptions {
                tonemapper: ToneMapper::UnrealFilmic,
                grade,
                encode_srgb: false,
                dither: false,
                ..Default::default()
            });
            let mut e = device.create_command_encoder(&Default::default());
            fx.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 256, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            let px: Vec<u16> = bytemuck::cast_slice(&data[..(w * 8) as usize]).to_vec();
            for (i, (input, want)) in inputs.iter().zip(want).enumerate() {
                let got = [f16_to_f32(px[i * 4]), f16_to_f32(px[i * 4 + 1]), f16_to_f32(px[i * 4 + 2])];
                for c in 0..3 {
                    // the half-float input and output, and the display's 8-bit-ish tolerance
                    assert!((got[c] - want[c]).abs() < 2e-3 + 0.01 * want[c], "{name} {input:?}: {got:?} vs Unreal {want:?}");
                }
            }
        }
    }

    fn f32_to_f16(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let mant = bits & 0x7f_ffff;
        if exp <= 0 {
            if exp < -10 { return sign; }
            let m = (mant | 0x80_0000) >> (1 - exp);
            return sign | ((m + 0x1000) >> 13) as u16;
        }
        if exp >= 31 { return sign | 0x7c00; }
        sign | ((exp as u16) << 10) | (((mant + 0x1000) >> 13) as u16)
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let frac = (h & 0x3ff) as f32;
        sign * match exp {
            0 => frac * 2f32.powi(-24),
            31 => f32::INFINITY,
            _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
        }
    }
}
