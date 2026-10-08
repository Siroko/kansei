//! `RtDiffuseGiEffect`: diffuse global illumination from one ray a pixel traced through the ray
//! tracing grid from the GBuffer's surfaces, the hits lit by their direct light and the voxels,
//! denoised by SVGF; and a reference path tracer through the same grid.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::grid::{RtGrid, RtGridHandle};
use crate::cameras::Camera;
use crate::gi::{VoxelClipmap, VoxelVolume};
use crate::lights::Light;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;
use crate::shadows::compute_shadows::ComputeShadows;
use crate::shadows::{CascadedShadowMap, CubeMapShadowMap, ShadowMap, SpotShadowAtlas};

/// The voxel source's functions for the trace (`srcVoxelSize`, `srcHitRadiance`, `srcCone`,
/// `srcSurfaceCone`), over a clipmap (`CLIPMAP_WGSL`'s group 0 bindings 50-57).
const CLIPMAP_SOURCE_WGSL: &str = r#"
fn srcVoxelSize(p: vec3f) -> f32 {
    return clipVoxelSize(min(clipLevelAt(p, 0u, 1.0), clipmap.levelCount - 1u));
}
// The light leaving a surface at p (face normal nf) as the voxels hold it: the finest level's
// sample a quarter voxel out of the surface (or in, where nothing is out), by its coverage.
fn srcHitRadiance(p: vec3f, nf: vec3f) -> vec3f {
    let k = clipLevelAt(p, 0u, 1.0);
    if (k >= clipmap.levelCount) { return vec3f(0.0); }
    let size = clipVoxelSize(k);
    var s = clipSample(k, p + nf * (0.25 * size));
    if (s.a < 0.05) { s = clipSample(k, p - nf * (0.25 * size)); }
    return s.rgb / max(s.a, 0.05) * clipmap.radianceScale;
}
fn srcCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return clipConeTrace(origin, dir, n, tanHalf, srcVoxelSize(origin), startDist, maxDist, steps);
}
// a cone leaving a surface (clipConeTrace lifts its samples off it)
fn srcSurfaceCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return srcCone(origin, dir, n, tanHalf, startDist, maxDist, steps);
}
"#;

/// The same over a voxel volume (group 0 bindings 60-62, its anisotropic mips 40-45).
const VOLUME_SOURCE_WGSL: &str = r#"
@group(0) @binding(60) var<uniform> vol : VoxelVolume;
@group(0) @binding(61) var volTex : texture_3d<f32>;
@group(0) @binding(62) var volSampler : sampler;
fn srcVoxelSize(p: vec3f) -> f32 {
    return vol.voxelSize;
}
fn srcHitRadiance(p: vec3f, nf: vec3f) -> vec3f {
    var s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p + nf * (0.25 * vol.voxelSize)), 0.0);
    if (s.a < 0.05) { s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p - nf * (0.25 * vol.voxelSize)), 0.0); }
    return s.rgb / max(s.a, 0.05) * vol.radianceScale;
}
fn srcCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return voxelConeTrace(vol, volTex, volSampler, origin, dir, tanHalf, startDist, maxDist, steps);
}
// a cone leaving a surface: voxel GI's own, through the anisotropic mips, its samples lifted off
// the surface (voxel_irradiance.wgsl); the isotropic mips leak through thin walls once it widens
fn srcSurfaceCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return voxelSurfaceConeTrace(vol, volTex, volSampler, origin, dir, n, tanHalf, startDist, maxDist, steps, 1.0);
}
"#;

/// The alpha test's texture and sampler (group 1 bindings 3 and 4), and the default
/// `kansei_rt_covered`: the texture's alpha at the uv, at least a half.
const ALPHA_BINDINGS_WGSL: &str = "@group(1) @binding(3) var kansei_rt_alpha_texture : texture_2d<f32>;\n@group(1) @binding(4) var kansei_rt_alpha_sampler : sampler;\n";
const DEFAULT_COVERED_WGSL: &str = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {\n    return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5;\n}\n";
const COMMON_WGSL: &str = include_str!("shaders/rt_gi_common.wgsl");
const PARAMS_WGSL: &str = "@group(0) @binding(20) var<uniform> gp : RtGiParams;\n";

/// The trace's WGSL over a clipmap or a volume, with `covered` (`kansei_rt_covered`).
pub(crate) fn trace_wgsl(clipmap: bool, covered: &str) -> String {
    let source = if clipmap {
        format!("{}{CLIPMAP_SOURCE_WGSL}", crate::gi::CLIPMAP_WGSL)
    } else {
        format!("{}{}{VOLUME_SOURCE_WGSL}", crate::gi::VOXEL_CONES_WGSL, include_str!("../gi/shaders/voxel_irradiance.wgsl"))
    };
    format!(
        "{}\n{}\n{}\n{COMMON_WGSL}\n{PARAMS_WGSL}\n{}\n{}\n{ALPHA_BINDINGS_WGSL}{covered}\n{source}\n{}",
        crate::atmosphere::SKY_LIGHTING_WGSL,
        crate::lights::SPOT_LIGHT_TYPES_WGSL,
        include_str!("../shaders/compute_shadows.wgsl"),
        super::RT_GRID_WGSL,
        RtGrid::bindings_wgsl(1, 0),
        include_str!("shaders/rt_gi_trace.wgsl")
    )
}

pub(crate) fn svgf_wgsl() -> String {
    format!("{COMMON_WGSL}\n{}", include_str!("shaders/rt_gi_svgf.wgsl"))
}

pub(crate) fn composite_wgsl() -> String {
    format!("{}\n{COMMON_WGSL}\n{}", crate::atmosphere::SKY_LIGHTING_WGSL, include_str!("shaders/rt_gi_composite.wgsl"))
}

/// What the rays are traced at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiResolution {
    /// One ray for every pixel.
    Full,
    /// One ray for each 2 x 2 block (a quarter of the rays), denoised there and upsampled by
    /// depth and normal. About a quarter of the cost; what real-time scenes use.
    #[default]
    Half,
}

impl RtGiResolution {
    fn downscale(self) -> u32 {
        match self {
            Self::Full => 1,
            Self::Half => 2,
        }
    }
}

/// What lights a ray's hit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiHitLighting {
    /// The voxels' radiance there (the voxels as the surface cache): no shadow rays, cheapest,
    /// but light leaks into contacts the voxels are too coarse for.
    Voxels,
    /// Its exact direct light (the lights passed to `update_lights` and `set_spot_lights`,
    /// shadowed as `RtGiShadows` says) plus the indirect light round it from one voxel cone in a
    /// cosine-distributed direction: the further bounces.
    #[default]
    Direct,
}

/// What shadows a hit's direct light (with `RtGiHitLighting::Direct`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiShadows {
    /// Shadow rays through the grid (a spot light's toward a point of its disk, a soft shadow
    /// over frames); past the grid's box, the directional shadow map.
    #[default]
    Rays,
    /// The renderer's shadow maps (`set_shadow_map`, `set_cascaded_shadow_map`,
    /// `set_point_shadows`, `set_spot_lights`' atlas).
    Maps,
}

/// The denoiser.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiDenoise {
    /// The raw 1 spp signal.
    Off,
    /// SVGF's temporal accumulation alone.
    Temporal,
    /// SVGF (Schied et al. 2017): temporal accumulation, a variance estimate, then
    /// `atrous_iterations` of the edge-avoiding a-trous wavelet.
    #[default]
    Svgf,
}

/// The a-trous wavelet's kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiKernel {
    /// 3 x 3 taps (1-2-1): half the cost of the 5 x 5, and as good on the scenes measured.
    #[default]
    Three,
    /// 5 x 5 taps (the B3 spline, SVGF's).
    Five,
}

/// What the rays compute.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiMode {
    /// One bounce traced through the grid, the voxels past it (the hybrid).
    #[default]
    Hybrid,
    /// A path tracer through the grid alone (`reference_bounces` vertices, the direct light at
    /// each, Russian roulette past the second), the same voxel cone past the grid's box. With
    /// `accumulate`, the ground truth the hybrid converges toward. A debug and quality tool: it
    /// costs several times the hybrid.
    Reference,
}

/// What the screen shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtGiView {
    /// The lit image with the GI.
    #[default]
    Lit,
    /// The light the GI adds alone (albedo times the signal).
    Indirect,
    /// The signal (the incoming light's irradiance over pi), no albedo.
    Signal,
    /// The variance the wavelet starts from, relative to the signal (`heat_scale`).
    Variance,
    /// Frames of history the temporal pass holds, of `max_history` (`heat_scale`).
    History,
    /// The rays' cost (cells visited and triangles tested), blue to red over 0-400
    /// (`heat_scale`).
    Cost,
}

/// `name` and `from_name` for the settings enums: the names a panel or URL parameter uses.
macro_rules! named {
    ($ty:ident { $($variant:ident => $name:literal),* $(,)? }) => {
        impl $ty {
            #[doc = concat!("The setting's name, as `from_name` reads it: ", $("`", $name, "` "),*)]
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

named!(RtGiResolution { Full => "full", Half => "half" });
named!(RtGiHitLighting { Voxels => "voxels", Direct => "direct" });
named!(RtGiShadows { Rays => "rays", Maps => "maps" });
named!(RtGiDenoise { Off => "off", Temporal => "temporal", Svgf => "svgf" });
named!(RtGiKernel { Three => "3x3", Five => "5x5" });
named!(RtGiMode { Hybrid => "hybrid", Reference => "reference" });
named!(RtGiView { Lit => "lit", Indirect => "indirect", Signal => "signal", Variance => "variance", History => "history", Cost => "cost" });

/// What `RtDiffuseGiEffect` sets up. The defaults are the measured real-time ones: half
/// resolution, exact direct light at hits with ray shadows, one anisotropic voxel cone at the
/// hit, SVGF with five 3 x 3 iterations.
#[derive(Clone, Debug)]
pub struct RtDiffuseGiOptions {
    pub resolution: RtGiResolution,
    pub hit_lighting: RtGiHitLighting,
    pub shadows: RtGiShadows,
    pub denoise: RtGiDenoise,
    pub kernel: RtGiKernel,
    /// Iterations of the wavelet (steps 1, 2, 4 ...; at most 8).
    pub atrous_iterations: u32,
    /// Metres a ray looks, through the grid then the voxels.
    pub max_distance: f32,
    /// Metres a ray walks the grid before the voxel cone takes over (and a sun's shadow ray
    /// before the shadow map); 0 walks the grid's whole box. In a large outdoor box 4-8 m halves
    /// the trace for some energy lost under thin occluders the voxels are too coarse for.
    pub near_distance: f32,
    /// The voxel cone a ray takes past the grid (tan of its half-angle) and its steps.
    pub cone_tan: f32,
    pub cone_steps: u32,
    /// The cone a hit's indirect light is read through (`RtGiHitLighting::Direct`); 0 steps
    /// lights hits by their direct light alone (one bounce).
    pub hit_cone_tan: f32,
    pub hit_cone_steps: u32,
    /// Scale of the sky past the voxels (`set_sky_lighting`).
    pub sky_scale: f32,
    /// Scale of the light the GI adds.
    pub intensity: f32,
    /// Share of the material's own sky ambient the GI replaces (with `set_sky_lighting`), as
    /// `VoxelGIOptions::ambient` does.
    pub ambient: f32,
    /// SVGF's temporal blend floors: the weight of a new frame in the colour and in the
    /// luminance moments once the history is long.
    pub temporal_alpha: f32,
    pub moments_alpha: f32,
    /// SVGF's edge stops: luminance (in the variance's standard deviations), normal (an
    /// exponent of their cosine), depth (in pixels' footprints off the centre's tangent plane).
    pub phi_color: f32,
    pub phi_normal: f32,
    pub phi_depth: f32,
    /// Frames the temporal history holds at most.
    pub max_history: f32,
    /// Alpha-test the grid's alpha-tested triangles (`RtSurface::with_alpha_layer`); off, they
    /// are solid.
    pub alpha_test: bool,
    /// WGSL defining `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, which may sample
    /// `kansei_rt_alpha_texture` with `kansei_rt_alpha_sampler` (`set_alpha_texture`). None: the
    /// texture's alpha is at least a half (a white texture until one is set: every hit).
    pub covered_wgsl: Option<String>,
    /// `RtGiMode::Reference`'s path vertices.
    pub reference_bounces: u32,
}

impl Default for RtDiffuseGiOptions {
    fn default() -> Self {
        Self {
            resolution: RtGiResolution::Half,
            hit_lighting: RtGiHitLighting::Direct,
            shadows: RtGiShadows::Rays,
            denoise: RtGiDenoise::Svgf,
            kernel: RtGiKernel::Three,
            atrous_iterations: 5,
            max_distance: 1e4,
            near_distance: 0.0,
            cone_tan: 0.1,
            cone_steps: 48,
            hit_cone_tan: 0.577,
            hit_cone_steps: 16,
            sky_scale: 1.0,
            intensity: 1.0,
            ambient: 1.0,
            temporal_alpha: 0.2,
            moments_alpha: 0.2,
            phi_color: 4.0,
            phi_normal: 128.0,
            phi_depth: 1.0,
            max_history: 32.0,
            alpha_test: true,
            covered_wgsl: None,
            reference_bounces: 4,
        }
    }
}

/// The trace's counters of a recent frame (`RtDiffuseGiEffect::collect_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtGiStats {
    /// Surface texels traced, and those whose ray hit a triangle of the grid.
    pub rays: u32,
    pub hits: u32,
    /// Cells visited plus triangles tested, by all the rays (shadow rays included).
    pub cost: u32,
    /// The most one texel's rays cost.
    pub max_cost: u32,
    pub shadow_rays: u32,
}

/// The WGSL `RtGiParams` (rt_gi_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct RtGiParamsGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    prev_view_proj: [f32; 16],
    full_size: [f32; 2],
    trace_size: [f32; 2],
    frame: u32,
    downscale: u32,
    flags: u32,
    view: u32,
    max_distance: f32,
    cone_tan: f32,
    cone_steps: u32,
    sky_scale: f32,
    intensity: f32,
    ambient: f32,
    hit_cone_tan: f32,
    hit_cone_steps: u32,
    bounces: u32,
    accum_count: u32,
    alpha_color: f32,
    alpha_moments: f32,
    phi_color: f32,
    phi_normal: f32,
    phi_depth: f32,
    num_dir_lights: u32,
    num_point_lights: u32,
    has_shadow_map: u32,
    pixel_size: f32,
    heat_scale: f32,
    max_history: f32,
    atrous_radius: u32,
    near_distance: f32,
    _pad: u32,
}

/// The WGSL `AtrousParams` (rt_gi_svgf.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct AtrousParamsGpu {
    step: u32,
    feedback: u32,
    last: u32,
    _pad: u32,
}

// rt_gi_common.wgsl's RT_GI_*
const FLAG_ALPHA: u32 = 1;
const FLAG_STATS: u32 = 2;
const FLAG_HISTORY: u32 = 4;
const FLAG_GRID: u32 = 8;
const FLAG_HIT_DIRECT: u32 = 16;
const FLAG_SHADOW_RAY: u32 = 32;
const FLAG_REFERENCE: u32 = 64;
const FLAG_ACCUMULATE: u32 = 128;
const FLAG_HAS_SKY: u32 = 256;
const MAX_ATROUS: usize = 8;

enum Source {
    Clipmap { uniform: wgpu::Buffer, levels: Vec<wgpu::TextureView>, sampler: wgpu::Sampler },
    Volume { view: wgpu::TextureView, anisotropic: Vec<wgpu::TextureView>, uniform: wgpu::Buffer, sampler: wgpu::Sampler },
}

/// The targets at the trace resolution: 8 bytes a texel for each texture, 16 for each wavelet
/// buffer.
struct Targets {
    size: (u32, u32),
    trace_size: (u32, u32),
    /// this frame's raw signal
    trace: wgpu::TextureView,
    /// the temporal pass's output (rgb, variance)
    integrated: wgpu::Texture,
    integrated_view: wgpu::TextureView,
    /// the wavelet's ping-pong (guide, colour and variance packed): the variance pass writes the
    /// second, the iterations alternate from there
    ping: [wgpu::Buffer; 2],
    /// the wavelet's result for the composite
    denoised: wgpu::TextureView,
    color_hist: [wgpu::Texture; 2],
    color_hist_view: [wgpu::TextureView; 2],
    moments: [wgpu::TextureView; 2],
    guide: [wgpu::TextureView; 2],
    /// the running sums (`accumulate`), made when first asked for
    accum: Option<wgpu::Buffer>,
}

struct Gpu {
    params: wgpu::Buffer,
    /// one per iteration, last or not
    atrous_params: Vec<wgpu::Buffer>,
    trace_bgl: wgpu::BindGroupLayout,
    grid_bgl: wgpu::BindGroupLayout,
    temporal_bgl: wgpu::BindGroupLayout,
    variance_bgl: wgpu::BindGroupLayout,
    atrous_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    variance: wgpu::ComputePipeline,
    atrous: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    no_sky: wgpu::Buffer,
    /// bound for the running sums while not accumulating
    no_accum: wgpu::Buffer,
    white: wgpu::TextureView,
    alpha_sampler: wgpu::Sampler,
    stats: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// the grid's group, with the grid generation and the alpha texture it was made with
    grid_group: Option<(u64, Option<wgpu::TextureView>, wgpu::BindGroup)>,
    targets: Option<Targets>,
}

/// Diffuse GI traced through the renderer's ray tracing grid (`Renderer::enable_rt_grid`;
/// `SceneRtGrid::handle`): one ray a pixel (by default one for each 2 x 2) a frame from the
/// GBuffer's surface in a cosine-distributed direction, the hits lit by their exact direct light
/// and one voxel cone's indirect light, the voxel GI's clipmap or volume (`with_clipmap`,
/// `with_volume`) past the grid's box, then the sky (`set_sky_lighting`). SVGF denoises the 1 spp
/// signal at the trace resolution; the composite upsamples it by depth and normal and adds
/// albedo times it to the lit colour, taking out the material's own sky ambient as
/// `VoxelGIEffect` does.
///
/// It is one GI path among voxel cones (`VoxelGIEffect`) and screen-space GI
/// (`ScreenSpaceGIEffect`), on the same scene setup: closest to a path-traced reference
/// (contacts, thin walls, sky through foliage), at a cost between theirs and a path tracer's.
/// Put it where `VoxelGIEffect` would go (before reflections, the atmosphere and TAA), and each
/// frame pass it the scene's lights (`update_lights`). The voxel GI it reads still updates
/// (`SceneVoxelGi`, `SceneVoxelClipmap`); the grid must hold the renderables the rays should hit
/// (`Renderable::rt`).
///
/// `mode` and `accumulate` turn it into a reference: a path tracer through the grid, and a
/// running mean of the raw signal while the view and settings stay put. `view` shows the
/// indirect light, the signal, SVGF's variance and history, or the rays' cost.
pub struct RtDiffuseGiEffect {
    pub enabled: bool,
    pub view: RtGiView,
    pub mode: RtGiMode,
    /// Show the running mean of the raw signal instead of the denoised one, restarted when the
    /// camera or the settings change (call `reset_history` after changing the scene or lights).
    pub accumulate: bool,
    pub hit_lighting: RtGiHitLighting,
    pub shadows: RtGiShadows,
    pub denoise: RtGiDenoise,
    pub kernel: RtGiKernel,
    pub atrous_iterations: u32,
    /// Trace the grid of triangles (on by default); off, every ray is a voxel cone from the
    /// surface (for comparison).
    pub trace_grid: bool,
    pub alpha_test: bool,
    pub max_distance: f32,
    pub near_distance: f32,
    pub cone_tan: f32,
    pub cone_steps: u32,
    pub hit_cone_tan: f32,
    pub hit_cone_steps: u32,
    pub sky_scale: f32,
    pub intensity: f32,
    pub ambient: f32,
    pub temporal_alpha: f32,
    pub moments_alpha: f32,
    pub phi_color: f32,
    pub phi_normal: f32,
    pub phi_depth: f32,
    pub max_history: f32,
    pub reference_bounces: u32,
    /// Scale of the debug views' colours (to suit the tone mapping after the effect).
    pub heat_scale: f32,
    /// Count the rays' work (`stats`), a few atomics a ray.
    pub collect_stats: bool,
    resolution: RtGiResolution,
    covered_wgsl: Option<String>,
    grid: RtGridHandle,
    source: Source,
    lights: ComputeShadows,
    sky_lighting: Option<wgpu::Buffer>,
    alpha_texture: Option<wgpu::TextureView>,
    frame: u32,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
    /// frames in the running sums, and what they were taken with
    accum_count: u32,
    accum_key: Option<(glam::Mat4, [u32; 12])>,
    /// the counters copied last frame, mapping, and the last read
    copied: bool,
    mapping: Option<Arc<AtomicU8>>,
    stats: Option<RtGiStats>,
    gpu: Option<Gpu>,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

impl RtDiffuseGiEffect {
    /// GI whose hits and far field read a voxel clipmap (`SceneVoxelClipmap::clipmap`): large
    /// and outdoor scenes.
    pub fn with_clipmap(clipmap: &VoxelClipmap, grid: RtGridHandle, options: RtDiffuseGiOptions) -> Self {
        let levels = clipmap.level_views().into_iter().cloned().collect();
        Self::from_source(Source::Clipmap { uniform: clipmap.uniform().clone(), levels, sampler: clipmap.sampler().clone() }, grid, options)
    }

    /// GI whose hits and far field read a voxel volume with anisotropic mips
    /// (`SceneVoxelGi::volume`): a room.
    pub fn with_volume(volume: &VoxelVolume, grid: RtGridHandle, options: RtDiffuseGiOptions) -> Self {
        let anisotropic = volume.anisotropic_views().expect("RtDiffuseGiEffect reads a volume with anisotropic mips (SceneVoxelGi's has them)").to_vec();
        Self::from_source(Source::Volume { view: volume.view().clone(), anisotropic, uniform: volume.uniform().clone(), sampler: volume.sampler().clone() }, grid, options)
    }

    fn from_source(source: Source, grid: RtGridHandle, o: RtDiffuseGiOptions) -> Self {
        Self {
            enabled: true,
            view: RtGiView::Lit,
            mode: RtGiMode::Hybrid,
            accumulate: false,
            hit_lighting: o.hit_lighting,
            shadows: o.shadows,
            denoise: o.denoise,
            kernel: o.kernel,
            atrous_iterations: o.atrous_iterations.min(MAX_ATROUS as u32),
            trace_grid: true,
            alpha_test: o.alpha_test,
            max_distance: o.max_distance,
            near_distance: o.near_distance,
            cone_tan: o.cone_tan,
            cone_steps: o.cone_steps,
            hit_cone_tan: o.hit_cone_tan,
            hit_cone_steps: o.hit_cone_steps,
            sky_scale: o.sky_scale,
            intensity: o.intensity,
            ambient: o.ambient,
            temporal_alpha: o.temporal_alpha,
            moments_alpha: o.moments_alpha,
            phi_color: o.phi_color,
            phi_normal: o.phi_normal,
            phi_depth: o.phi_depth,
            max_history: o.max_history,
            reference_bounces: o.reference_bounces,
            heat_scale: 1.0,
            collect_stats: false,
            resolution: o.resolution,
            covered_wgsl: o.covered_wgsl,
            grid,
            source,
            lights: ComputeShadows::new(),
            sky_lighting: None,
            alpha_texture: None,
            frame: 0,
            prev_view_proj: None,
            last_camera_frame: None,
            accum_count: 0,
            accum_key: None,
            copied: false,
            mapping: None,
            stats: None,
            gpu: None,
        }
    }

    /// The scene's directional, point and area lights (`Scene::lights`), each frame they may
    /// change: what lights the hits.
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        self.lights.update_lights(lights, false);
    }

    /// The renderer's spot lights (`Renderer::spot_lights_buffer`) and their shadow atlas.
    pub fn set_spot_lights(&mut self, lights: Option<&wgpu::Buffer>, atlas: Option<&SpotShadowAtlas>) {
        self.lights.set_spot_lights(lights, atlas);
    }

    /// The directional shadow map: the sun's shadows past the grid's box (or everywhere with
    /// `RtGiShadows::Maps`).
    pub fn set_shadow_map(&mut self, shadow_map: Option<&ShadowMap>) {
        self.lights.set_shadow_map(shadow_map);
    }

    /// Cascaded shadows in place of `set_shadow_map`'s (`Renderer::cascaded_shadow_map`).
    pub fn set_cascaded_shadow_map(&mut self, csm: Option<&CascadedShadowMap>) {
        self.lights.set_cascaded_shadow_map(csm);
    }

    /// Point lights' cube shadows (`RtGiShadows::Maps`).
    pub fn set_point_shadows(&mut self, cube: Option<&CubeMapShadowMap>) {
        self.lights.set_point_shadows(cube);
    }

    /// The sky past the voxels, and the ambient the composite takes out
    /// (`SkyAtmosphereBindings::sky_lighting`, or any `SkyLighting` uniform); black without one.
    pub fn set_sky_lighting(&mut self, sky: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky.cloned();
    }

    /// The texture `kansei_rt_covered` reads (`kansei_rt_alpha_texture`).
    pub fn set_alpha_texture(&mut self, view: Option<&wgpu::TextureView>) {
        self.alpha_texture = view.cloned();
    }

    /// Trace at another resolution (the targets are made anew, the history restarts).
    pub fn set_resolution(&mut self, resolution: RtGiResolution) {
        if resolution != self.resolution {
            self.resolution = resolution;
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.targets = None;
            }
            self.reset_history();
        }
    }

    pub fn resolution(&self) -> RtGiResolution {
        self.resolution
    }

    /// Start the denoiser's history and the running mean over (after a cut, or a change of the
    /// scene or its lights the history should not blend through).
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
        self.accum_key = None;
        self.accum_count = 0;
    }

    /// Frames in the running mean (`accumulate`).
    pub fn accumulated(&self) -> u32 {
        if self.accumulate { self.accum_count } else { 0 }
    }

    /// The counters of a recent frame, while `collect_stats` is on (they arrive a few frames late).
    pub fn stats(&self) -> Option<RtGiStats> {
        self.stats
    }

    /// Bytes of the effect's targets (about 104 a trace texel, 16 more while accumulating).
    pub fn memory_bytes(&self) -> u64 {
        let Some(t) = self.gpu.as_ref().and_then(|g| g.targets.as_ref()) else { return 0 };
        let texels = (t.trace_size.0 * t.trace_size.1) as u64;
        // trace, integrated, denoised, 2 colour, 2 moments, 2 guides at 8 bytes; 2 ping at 16
        texels * (9 * 8 + 2 * 16) + t.accum.as_ref().map_or(0, |a| a.size())
    }

    fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage_rw = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let storage_ro = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let d2 = wgpu::TextureViewDimension::D2;
        let tex = |filterable| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: d2, multisampled: false };
        let depth = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false };
        let storage_tex = |format| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: d2 };
        let f16 = wgpu::TextureFormat::Rgba16Float;
        let guide_format = wgpu::TextureFormat::Rg32Uint;
        let uint = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Uint, view_dimension: d2, multisampled: false };
        let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });

        let clipmap = matches!(self.source, Source::Clipmap { .. });
        let mut trace_entries = ComputeShadows::layout_entries();
        trace_entries.extend([entry(20, uniform), entry(21, depth), entry(22, tex(false)), entry(23, storage_tex(f16)), entry(24, uniform), entry(25, storage_rw)]);
        if clipmap {
            trace_entries.extend(crate::gi::clipmap_layout_entries(compute));
        } else {
            let d3 = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false };
            trace_entries.extend([entry(60, uniform), entry(61, d3), entry(62, filtering)]);
            trace_entries.extend((40..46).map(|b| entry(b, d3)));
        }
        let trace_bgl = layout("RtGi/Trace", &trace_entries);
        let mut grid_entries = RtGrid::layout_entries(0, compute).to_vec();
        grid_entries.extend([entry(3, tex(true)), entry(4, filtering), entry(5, storage_rw)]);
        let grid_bgl = layout("RtGi/Grid", &grid_entries);
        let temporal_bgl = layout(
            "RtGi/Temporal",
            &[
                entry(20, uniform),
                entry(21, tex(false)),
                entry(22, depth),
                entry(23, tex(false)),
                entry(24, tex(false)),
                entry(25, tex(false)),
                entry(26, tex(false)),
                entry(27, uint),
                entry(28, storage_tex(f16)),
                entry(29, storage_tex(f16)),
                entry(30, storage_tex(guide_format)),
            ],
        );
        let variance_bgl = layout("RtGi/Variance", &[entry(20, uniform), entry(31, tex(false)), entry(32, tex(false)), entry(33, uint), entry(34, storage_rw)]);
        let atrous_bgl = layout("RtGi/Atrous", &[entry(20, uniform), entry(35, uniform), entry(36, storage_ro), entry(37, storage_rw), entry(38, storage_tex(f16)), entry(39, storage_tex(f16))]);
        let composite_bgl = layout(
            "RtGi/Composite",
            &[
                entry(20, uniform),
                entry(21, tex(false)),
                entry(22, depth),
                entry(23, tex(false)),
                entry(24, tex(false)),
                entry(25, tex(false)),
                entry(26, uint),
                entry(27, storage_ro),
                entry(28, storage_tex(f16)),
                entry(29, uniform),
                entry(30, tex(false)),
                entry(31, tex(false)),
            ],
        );
        let pipeline = |label: &str, code: &str, entry_point: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let covered = self.covered_wgsl.clone().unwrap_or_else(|| DEFAULT_COVERED_WGSL.into());
        let trace = pipeline("RtGi/Trace", &trace_wgsl(clipmap, &covered), "main", &[&trace_bgl, &grid_bgl]);
        let svgf = svgf_wgsl();
        let temporal = pipeline("RtGi/Temporal", &svgf, "temporal", &[&temporal_bgl]);
        let variance = pipeline("RtGi/Variance", &svgf, "variance", &[&variance_bgl]);
        let atrous = pipeline("RtGi/Atrous", &svgf, "atrous", &[&atrous_bgl]);
        let composite = pipeline("RtGi/Composite", &composite_wgsl(), "main", &[&composite_bgl]);
        use wgpu::util::DeviceExt;
        let white = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("RtGi/White"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &[255; 4],
            )
            .create_view(&Default::default());
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let atrous_params = (0..MAX_ATROUS * 2)
            .map(|k| {
                let (i, last) = (k / 2, k % 2);
                let p = AtrousParamsGpu { step: 1 << i, feedback: (i == 0) as u32, last: last as u32, _pad: 0 };
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("RtGi/AtrousParams"), contents: bytemuck::bytes_of(&p), usage: wgpu::BufferUsages::UNIFORM })
            })
            .collect();
        self.gpu = Some(Gpu {
            params: buffer("RtGi/Params", std::mem::size_of::<RtGiParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            atrous_params,
            trace_bgl,
            grid_bgl,
            temporal_bgl,
            variance_bgl,
            atrous_bgl,
            composite_bgl,
            trace,
            temporal,
            variance,
            atrous,
            composite,
            no_sky: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("RtGi/NoSky"), contents: &[0u8; 256], usage: wgpu::BufferUsages::UNIFORM }),
            no_accum: buffer("RtGi/NoAccumulation", 16, wgpu::BufferUsages::STORAGE),
            white,
            alpha_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("RtGi/Alpha"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                ..Default::default()
            }),
            stats: buffer("RtGi/Stats", 32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            staging: buffer("RtGi/StatsReadback", 32, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            grid_group: None,
            targets: None,
        });
    }

    fn make_targets(device: &wgpu::Device, size: (u32, u32), trace_size: (u32, u32)) -> Targets {
        let texture = |label, format, extra: wgpu::TextureUsages| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: trace_size.0, height: trace_size.1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | extra,
                view_formats: &[],
            })
        };
        let f16 = wgpu::TextureFormat::Rgba16Float;
        let none = wgpu::TextureUsages::empty();
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let packed_buffer = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (trace_size.0 * trace_size.1) as u64 * 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let integrated = texture("RtGi/Integrated", f16, wgpu::TextureUsages::COPY_SRC);
        let color_hist = [texture("RtGi/ColorHistory", f16, wgpu::TextureUsages::COPY_DST), texture("RtGi/ColorHistory", f16, wgpu::TextureUsages::COPY_DST)];
        Targets {
            size,
            trace_size,
            trace: view(&texture("RtGi/Trace", f16, none)),
            integrated_view: view(&integrated),
            integrated,
            ping: [packed_buffer("RtGi/Ping"), packed_buffer("RtGi/Pong")],
            denoised: view(&texture("RtGi/Denoised", f16, none)),
            color_hist_view: [view(&color_hist[0]), view(&color_hist[1])],
            color_hist,
            moments: [view(&texture("RtGi/Moments", f16, none)), view(&texture("RtGi/Moments", f16, none))],
            guide: [view(&texture("RtGi/Guide", wgpu::TextureFormat::Rg32Uint, none)), view(&texture("RtGi/Guide", wgpu::TextureFormat::Rg32Uint, none))],
            accum: None,
        }
    }

    /// The settings the running sums depend on.
    fn accum_settings(&self) -> [u32; 12] {
        [
            self.mode as u32,
            self.hit_lighting as u32,
            self.shadows as u32,
            self.trace_grid as u32 | (self.alpha_test as u32) << 1,
            self.reference_bounces,
            self.cone_steps,
            self.cone_tan.to_bits(),
            self.hit_cone_tan.to_bits(),
            self.hit_cone_steps,
            self.sky_scale.to_bits(),
            self.max_distance.to_bits(),
            self.near_distance.to_bits(),
        ]
    }
}

impl PostProcessingEffect for RtDiffuseGiEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device, queue);
        }
        self.poll_stats(device);
        let downscale = self.resolution.downscale();
        let (tw, th) = (width.div_ceil(downscale), height.div_ceil(downscale));
        {
            let gpu = self.gpu.as_mut().unwrap();
            if gpu.targets.as_ref().is_none_or(|t| t.size != (width, height) || t.trace_size != (tw, th)) {
                gpu.targets = Some(Self::make_targets(device, (width, height), (tw, th)));
                self.prev_view_proj = None;
                self.accum_key = None;
            }
        }
        // a frame skipped (the effect was off, a cut): no history
        let camera_frame = camera.frame();
        if self.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
            self.accum_key = None;
        }
        self.last_camera_frame = Some(camera_frame);
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let view_proj = proj * view;
        // the running sums restart when the view or the settings change
        let key = (view_proj, self.accum_settings());
        if self.accumulate {
            if self.accum_key != Some(key) {
                self.accum_count = 0;
                self.accum_key = Some(key);
            }
            let targets = self.gpu.as_mut().unwrap().targets.as_mut().unwrap();
            targets.accum.get_or_insert_with(|| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtGi/Accumulation"), size: (tw * th) as u64 * 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false })
            });
        } else {
            self.accum_key = None;
            self.accum_count = 0;
        }
        self.lights.prepare(device, queue);
        let mut flags = 0;
        for (on, flag) in [
            (self.alpha_test, FLAG_ALPHA),
            (self.collect_stats, FLAG_STATS),
            (self.prev_view_proj.is_some(), FLAG_HISTORY),
            (self.trace_grid, FLAG_GRID),
            (self.hit_lighting == RtGiHitLighting::Direct, FLAG_HIT_DIRECT),
            (self.shadows == RtGiShadows::Rays, FLAG_SHADOW_RAY),
            (self.mode == RtGiMode::Reference, FLAG_REFERENCE),
            (self.accumulate, FLAG_ACCUMULATE),
            (self.sky_lighting.is_some(), FLAG_HAS_SKY),
        ] {
            if on {
                flags |= flag;
            }
        }
        let inv_proj = proj.inverse();
        let params = RtGiParamsGpu {
            inv_proj: inv_proj.to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            full_size: [width as f32, height as f32],
            trace_size: [tw as f32, th as f32],
            frame: self.frame,
            downscale,
            flags,
            view: self.view as u32,
            max_distance: self.max_distance.max(0.0),
            cone_tan: self.cone_tan.max(1e-3),
            cone_steps: self.cone_steps,
            sky_scale: self.sky_scale.max(0.0),
            intensity: self.intensity.max(0.0),
            ambient: self.ambient,
            hit_cone_tan: self.hit_cone_tan.max(1e-3),
            hit_cone_steps: self.hit_cone_steps,
            bounces: self.reference_bounces.max(1),
            accum_count: self.accum_count,
            alpha_color: self.temporal_alpha.clamp(0.01, 1.0),
            alpha_moments: self.moments_alpha.clamp(0.01, 1.0),
            phi_color: self.phi_color.max(1e-3),
            phi_normal: self.phi_normal,
            phi_depth: self.phi_depth.max(1e-3),
            num_dir_lights: self.lights.dir.len() as u32,
            num_point_lights: self.lights.point.len() as u32,
            has_shadow_map: self.lights.has_shadow_map() as u32,
            pixel_size: 2.0 * inv_proj.y_axis.y / height as f32,
            heat_scale: self.heat_scale,
            max_history: self.max_history.max(1.0),
            atrous_radius: match self.kernel {
                RtGiKernel::Three => 1,
                RtGiKernel::Five => 2,
            },
            near_distance: self.near_distance.max(0.0),
            _pad: 0,
        };
        let gpu = self.gpu.as_mut().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let cur = (self.frame % 2) as usize;
        let prev = 1 - cur;
        let t = gpu.targets.as_ref().unwrap();
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.no_sky);
        let accum = t.accum.as_ref().filter(|_| self.accumulate).unwrap_or(&gpu.no_accum);
        let b = |binding, resource| wgpu::BindGroupEntry { binding, resource };

        // the trace's group: the lights, the GBuffer, its targets, the voxel source
        let mut entries = self.lights.entries();
        entries.extend([b(20, buf(&gpu.params)), b(21, tex(depth)), b(22, tex(&gbuffer.normal_view)), b(23, tex(&t.trace)), b(24, buf(sky)), b(25, buf(accum))]);
        match &self.source {
            Source::Clipmap { uniform, levels, sampler } => {
                entries.push(b(50, buf(uniform)));
                entries.extend(levels.iter().enumerate().map(|(k, v)| b(51 + k as u32, tex(v))));
                entries.push(b(57, wgpu::BindingResource::Sampler(sampler)));
            }
            Source::Volume { view, anisotropic, uniform, sampler } => {
                entries.push(b(60, buf(uniform)));
                entries.push(b(61, tex(view)));
                entries.push(b(62, wgpu::BindingResource::Sampler(sampler)));
                entries.extend(anisotropic.iter().enumerate().map(|(i, v)| b(40 + i as u32, tex(v))));
            }
        }
        let trace_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("RtGi/Trace"), layout: &gpu.trace_bgl, entries: &entries });
        // the grid's group, made anew when the grid's buffers or the alpha texture change
        let (buffers, generation) = self.grid.buffers();
        if gpu.grid_group.as_ref().is_none_or(|(g, alpha, _)| *g != generation || *alpha != self.alpha_texture) {
            let alpha = self.alpha_texture.as_ref().unwrap_or(&gpu.white);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtGi/Grid"),
                layout: &gpu.grid_bgl,
                entries: &[b(0, buf(&buffers[0])), b(1, buf(&buffers[1])), b(2, buf(&buffers[2])), b(3, tex(alpha)), b(4, wgpu::BindingResource::Sampler(&gpu.alpha_sampler)), b(5, buf(&gpu.stats))],
            });
            gpu.grid_group = Some((generation, self.alpha_texture.clone(), group));
        }
        let temporal_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGi/Temporal"),
            layout: &gpu.temporal_bgl,
            entries: &[
                b(20, buf(&gpu.params)),
                b(21, tex(&t.trace)),
                b(22, tex(depth)),
                b(23, tex(&gbuffer.normal_view)),
                b(24, tex(&gbuffer.velocity_view)),
                b(25, tex(&t.color_hist_view[prev])),
                b(26, tex(&t.moments[prev])),
                b(27, tex(&t.guide[prev])),
                b(28, tex(&t.integrated_view)),
                b(29, tex(&t.moments[cur])),
                b(30, tex(&t.guide[cur])),
            ],
        });
        let svgf = self.denoise == RtGiDenoise::Svgf;
        let iterations = if svgf { self.atrous_iterations.min(MAX_ATROUS as u32) as usize } else { 0 };
        // the signal the composite shows
        let signal = match (self.view, self.denoise) {
            (RtGiView::Cost, _) | (_, RtGiDenoise::Off) => &t.trace,
            (_, RtGiDenoise::Temporal) => &t.integrated_view,
            (_, RtGiDenoise::Svgf) if iterations == 0 => &t.integrated_view,
            (_, RtGiDenoise::Svgf) => &t.denoised,
        };
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGi/Composite"),
            layout: &gpu.composite_bgl,
            entries: &[
                b(20, buf(&gpu.params)),
                b(21, tex(input)),
                b(22, tex(depth)),
                b(23, tex(&gbuffer.normal_view)),
                b(24, tex(&gbuffer.albedo_view)),
                b(25, tex(signal)),
                b(26, tex(&t.guide[cur])),
                b(27, buf(accum)),
                b(28, tex(output)),
                b(29, buf(sky)),
                b(30, tex(&t.moments[cur])),
                b(31, tex(&t.integrated_view)),
            ],
        });
        let read_stats = self.collect_stats && !self.copied && self.mapping.is_none();
        if read_stats {
            encoder.clear_buffer(&gpu.stats, 0, None);
        }
        let (gx, gy) = (tw.div_ceil(8), th.div_ceil(8));
        {
            let stamp = crate::profiling::gpu_pass("RtGi/Trace");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtGi/Trace"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.trace);
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, &gpu.grid_group.as_ref().unwrap().2, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if read_stats {
            encoder.copy_buffer_to_buffer(&gpu.stats, 0, &gpu.staging, 0, 32);
            self.copied = true;
        }
        {
            let stamp = crate::profiling::gpu_pass("RtGi/Temporal");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtGi/Temporal"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.temporal);
            pass.set_bind_group(0, &temporal_group, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        if svgf {
            // the variance pass writes the second ping buffer; iteration i reads the other
            let variance_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtGi/Variance"),
                layout: &gpu.variance_bgl,
                entries: &[b(20, buf(&gpu.params)), b(31, tex(&t.integrated_view)), b(32, tex(&t.moments[cur])), b(33, tex(&t.guide[cur])), b(34, buf(&t.ping[1]))],
            });
            {
                let stamp = crate::profiling::gpu_pass("RtGi/Variance");
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtGi/Variance"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
                pass.set_pipeline(&gpu.variance);
                pass.set_bind_group(0, &variance_group, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
            const LABELS: [&str; MAX_ATROUS] = ["RtGi/Atrous1", "RtGi/Atrous2", "RtGi/Atrous3", "RtGi/Atrous4", "RtGi/Atrous5", "RtGi/Atrous6", "RtGi/Atrous7", "RtGi/Atrous8"];
            for (i, label) in LABELS.iter().enumerate().take(iterations) {
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RtGi/Atrous"),
                    layout: &gpu.atrous_bgl,
                    entries: &[
                        b(20, buf(&gpu.params)),
                        b(35, buf(&gpu.atrous_params[i * 2 + (i + 1 == iterations) as usize])),
                        b(36, buf(&t.ping[(i + 1) % 2])),
                        b(37, buf(&t.ping[i % 2])),
                        b(38, tex(&t.color_hist_view[cur])),
                        b(39, tex(&t.denoised)),
                    ],
                });
                let stamp = crate::profiling::gpu_pass(label);
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
                pass.set_pipeline(&gpu.atrous);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
            }
        }
        if iterations == 0 {
            // the history is the temporal pass's output
            let extent = wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 };
            encoder.copy_texture_to_texture(t.integrated.as_image_copy(), t.color_hist[cur].as_image_copy(), extent);
        }
        {
            let stamp = crate::profiling::gpu_pass("RtGi/Composite");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtGi/Composite"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.composite);
            pass.set_bind_group(0, &composite_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);
        if self.accumulate {
            self.accum_count += 1;
        }
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn tex(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

fn buf(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buffer.as_entire_binding()
}

impl RtDiffuseGiEffect {
    /// The counters' readback: map last frame's copy (its frame has been submitted since), and take
    /// a finished map.
    fn poll_stats(&mut self, device: &wgpu::Device) {
        let Some(gpu) = self.gpu.as_ref() else { return };
        #[cfg(not(target_arch = "wasm32"))]
        if self.mapping.is_some() {
            device.poll(wgpu::Maintain::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        if let Some(state) = self.mapping.take_if(|s| s.load(Ordering::Acquire) != MAPPING) {
            if state.load(Ordering::Acquire) == MAPPED {
                {
                    let bytes = gpu.staging.slice(..).get_mapped_range();
                    let w: &[u32] = bytemuck::cast_slice(&bytes);
                    self.stats = Some(RtGiStats { rays: w[0], hits: w[1], cost: w[3], max_cost: w[4], shadow_rays: w[5] });
                }
                gpu.staging.unmap();
            }
        } else if std::mem::take(&mut self.copied) {
            let state = Arc::new(AtomicU8::new(MAPPING));
            let done = state.clone();
            gpu.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
            self.mapping = Some(state);
        }
        if !self.collect_stats {
            self.stats = None;
        }
    }
}
