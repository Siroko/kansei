//! `RtReflectionsEffect`: sharp and glossy reflections traced through the ray tracing grid, lit
//! by the voxel GI.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::grid::{RtGrid, RtGridHandle};
use crate::cameras::Camera;
use crate::gi::{VoxelClipmap, VoxelVolume};
use crate::lights::Light;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

/// The voxel source's functions for the trace (`srcVoxelSize`, `srcHitRadiance`, `srcCone`), over
/// a clipmap (`CLIPMAP_WGSL`'s group 0 bindings 50-57).
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
// The irradiance a surface at p (normal n) receives from the voxels and the sky past them, as
// voxel GI's composite gathers it (its six cones).
fn srcIrradiance(p: vec3f, n: vec3f) -> vec3f {
    let size = srcVoxelSize(p);
    return clipIrradiance(sky, rp.skyScale, p, n, 0.0, size, 1.5 * size, rp.maxDistance, 16u).rgb;
}
"#;

/// The same over a voxel volume (group 0 bindings 6-8).
const VOLUME_SOURCE_WGSL: &str = r#"
@group(0) @binding(6) var<uniform> vol : VoxelVolume;
@group(0) @binding(7) var volTex : texture_3d<f32>;
@group(0) @binding(8) var volSampler : sampler;
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
// The irradiance a surface at p (normal n) receives from the voxels and the sky past them, as
// voxel GI's composite gathers it: its six cones through the anisotropic mips (bindings 40-45,
// RT_REFLECT_ANISO), else five through the isotropic ones.
fn srcIrradiance(p: vec3f, n: vec3f) -> vec3f {
    if ((rp.flags & RT_REFLECT_ANISO) != 0u) {
        return voxelIrradiance(vol, volTex, volSampler, sky, rp.skyScale, p, n, 0.0, vol.voxelSize, rp.maxDistance, 32u, 1.0).rgb;
    }
    var e = vec3f(0.0);
    for (var k = 0u; k < VOXEL_HEMISPHERE_CONES; k++) {
        let cone = voxelHemisphereCone(n, k);
        let c = voxelConeTrace(vol, volTex, volSampler, p, cone.xyz, VOXEL_HEMISPHERE_TAN, 1.5 * vol.voxelSize, rp.maxDistance, 16u);
        e += cone.w * (c.rgb + c.a * rp.skyScale * skyRadiance(sky, cone.xyz));
    }
    return e;
}
"#;

/// The alpha test's texture and sampler (group 1 bindings 3 and 4), and the default
/// `kansei_rt_covered`: the texture's alpha at the uv, at least a half.
const ALPHA_BINDINGS_WGSL: &str = "@group(1) @binding(3) var kansei_rt_alpha_texture : texture_2d<f32>;\n@group(1) @binding(4) var kansei_rt_alpha_sampler : sampler;\n";
const DEFAULT_COVERED_WGSL: &str = "fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {\n    return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5;\n}\n";

const COMMON_WGSL: &str = include_str!("shaders/rt_reflect_common.wgsl");

/// The trace's WGSL over a clipmap or a volume, with `covered` (`kansei_rt_covered`).
pub(crate) fn trace_wgsl(clipmap: bool, covered: &str) -> String {
    traced_wgsl(clipmap, covered, include_str!("shaders/rt_reflect_trace.wgsl"))
}

/// The glass pass's WGSL, prefixed as the trace.
pub(crate) fn glass_wgsl(clipmap: bool, covered: &str) -> String {
    traced_wgsl(clipmap, covered, include_str!("shaders/rt_glass.wgsl"))
}

fn traced_wgsl(clipmap: bool, covered: &str, main: &str) -> String {
    let source = if clipmap {
        format!("{}{CLIPMAP_SOURCE_WGSL}", crate::gi::CLIPMAP_WGSL)
    } else {
        format!("{}{}{VOLUME_SOURCE_WGSL}", crate::gi::VOXEL_CONES_WGSL, include_str!("../gi/shaders/voxel_irradiance.wgsl"))
    };
    format!(
        "{}\n{}\n{COMMON_WGSL}\n{}\n{}\n{ALPHA_BINDINGS_WGSL}{covered}\n{source}\n{}\n{main}",
        crate::atmosphere::SKY_LIGHTING_WGSL,
        include_str!("../shaders/spot_light_types.wgsl"),
        super::RT_GRID_WGSL,
        RtGrid::bindings_wgsl(1, 0),
        include_str!("shaders/rt_reflect_hit.wgsl"),
    )
}

pub(crate) fn resolve_wgsl() -> String {
    format!("{COMMON_WGSL}\n{}", include_str!("shaders/rt_reflect_resolve.wgsl"))
}

/// What the reflections trace at: one pixel of each block this wide a frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RtTraceResolution {
    /// One pixel of each 2 x 2 a frame.
    Half,
    /// One pixel of each 4 x 4 a frame.
    Quarter,
}

impl RtTraceResolution {
    fn downscale(self) -> u32 {
        match self {
            Self::Half => 2,
            Self::Quarter => 4,
        }
    }
}

/// What the screen shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtReflectionsView {
    /// The lit image with the reflections.
    #[default]
    Lit,
    /// The light the reflections add alone (Fresnel applied).
    Reflection,
    /// What the reflection rays see, without the Fresnel.
    Mirror,
    /// The rays' cost (cells visited and triangles tested), blue to red over 0-400.
    Cost,
}

/// What `RtReflectionsEffect` sets up.
#[derive(Clone, Debug)]
pub struct RtReflectionsOptions {
    pub resolution: RtTraceResolution,
    /// Scale of the reflection (1 is physical).
    pub intensity: f32,
    /// Metres a reflection looks, through the grid then the voxels.
    pub max_distance: f32,
    /// tan of the half-angle of the cone rays take through the voxels past the grid (rough
    /// surfaces widen it to their lobe).
    pub cone_tan: f32,
    pub cone_steps: u32,
    /// Weight of each new frame in the accumulated reflection.
    pub temporal_blend: f32,
    /// Scale of the sky past the voxels.
    pub sky_scale: f32,
    /// Alpha-test the grid's alpha-tested triangles (`RtSurface::with_alpha_layer`); off, they are
    /// solid.
    pub alpha_test: bool,
    /// WGSL defining `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, which may sample
    /// `kansei_rt_alpha_texture` with `kansei_rt_alpha_sampler` (`set_alpha_texture`). None: the
    /// texture's alpha is at least a half (a white texture until one is set: every hit).
    pub covered_wgsl: Option<String>,
}

impl Default for RtReflectionsOptions {
    fn default() -> Self {
        Self { resolution: RtTraceResolution::Half, intensity: 1.0, max_distance: 400.0, cone_tan: 0.04, cone_steps: 96, temporal_blend: 0.25, sky_scale: 1.0, alpha_test: true, covered_wgsl: None }
    }
}

/// Glass (`RtReflectionsEffect::set_glass`): the pixels whose material writes glass
/// (`StandardLitOptions::glass`, or GBUFFER_OUT_WGSL's `kansei_gbuffer_out_glass`: its tint, index
/// of refraction and roughness) show the light it reflects and the light through it, refracted at
/// every surface of the grid's glass (`RtSurface::glass`) it crosses: true entry and exit, with
/// total internal reflection. Other rays (diffuse GI, shadows) pass through glass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtGlass {
    /// The most surfaces a ray through the glass crosses or reflects off inside.
    pub interfaces: u32,
    /// Paths a frosted (rough) glass pixel traces a frame, averaged with the pixel's history
    /// (reprojected by the surface); clear glass traces one.
    pub samples: u32,
}

impl Default for RtGlass {
    fn default() -> Self {
        Self { interfaces: 8, samples: 4 }
    }
}

/// The trace's counters of a recent frame (`RtReflectionsEffect::collect_stats`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtReflectionStats {
    /// Reflective pixels traced, and those whose ray hit a triangle of the grid.
    pub rays: u32,
    pub hits: u32,
    /// Cells visited and triangles tested, by all the rays.
    pub cells: u32,
    pub tests: u32,
    /// The most cells and triangles one ray visited.
    pub max_cost: u32,
    /// Glass pixels, and the cells and triangles their rays visited.
    pub glass_pixels: u32,
    pub glass_cost: u32,
}

/// The WGSL `RtReflectParams` (rt_reflect_common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct RtReflectParamsGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    prev_view_proj: [f32; 16],
    full_size: [f32; 2],
    trace_size: [f32; 2],
    frame: u32,
    downscale: u32,
    flags: u32,
    view: u32,
    cone_tan: f32,
    cone_steps: u32,
    max_distance: f32,
    intensity: f32,
    sky_scale: f32,
    blend: f32,
    heat_scale: f32,
    _pad: u32,
    view_proj: [f32; 16],
    glass_interfaces: u32,
    glass_samples: u32,
    _pad2: [u32; 2],
}

enum Source {
    Clipmap { uniform: wgpu::Buffer, levels: Vec<wgpu::TextureView>, sampler: wgpu::Sampler },
    Volume { view: wgpu::TextureView, anisotropic: Option<Vec<wgpu::TextureView>>, uniform: wgpu::Buffer, sampler: wgpu::Sampler },
}

struct Targets {
    size: (u32, u32),
    trace: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
    /// the input with the glass drawn (what the reflections then composite over)
    glassed: wgpu::TextureView,
    /// frosted glass's history (last frame's and this frame's, by turns)
    glass_history: [wgpu::TextureView; 2],
}

struct Gpu {
    params: wgpu::Buffer,
    trace_bgl: wgpu::BindGroupLayout,
    grid_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    glass: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    no_sky: wgpu::Buffer,
    /// no spot lights (a count of 0)
    no_spots: wgpu::Buffer,
    /// the directional and point lights (`update_lights`; rt_reflect_hit.wgsl's RtReflectLights)
    lights: wgpu::Buffer,
    white: wgpu::TextureView,
    alpha_sampler: wgpu::Sampler,
    linear: wgpu::Sampler,
    stats: wgpu::Buffer,
    staging: wgpu::Buffer,
    /// the grid's group, with the grid generation and the alpha texture it was made with
    grid_group: Option<(u64, Option<wgpu::TextureView>, wgpu::BindGroup)>,
    targets: Option<Targets>,
}

/// Sharp and glossy reflections, traced through the renderer's ray tracing grid
/// (`Renderer::enable_rt_grid`; `SceneRtGrid::handle`) on the surfaces whose material writes an
/// F0 (`StandardLitOptions::mirror`, `StandardTraced::Reflective`, or GBUFFER_OUT_WGSL's
/// `kansei_gbuffer_out_specular`), and glass (`set_glass`). The hits are lit by the voxel GI's
/// clipmap or volume (`with_clipmap`, `with_volume`): the light leaving the surface there, the
/// voxels as the surface cache; with `screen_hits`, by the lit image where the camera sees the
/// same point, and with `set_spot_lights`, the hits it doesn't see by the spot lights (shadow rays
/// through the grid) plus the voxels' irradiance. Rays leaving the grid's box go on as a narrow
/// voxel cone, then the sky (`set_sky_lighting`). It needs the voxel GI running whatever GI the
/// image shows (`hit_indirect` off where that is none or screen space).
///
/// It traces one pixel of each 2 x 2 (or 4 x 4) block a frame, each in turn, and accumulates them
/// at full resolution: the frame's traced pixels upsampled by depth and normal, blended with the
/// history reprojected by the surface and clamped to their range. The composite is the lit colour
/// times 1 - F plus F times the reflection, F Schlick's Fresnel of the F0 (lessened on rough
/// surfaces, whose rays jitter over their lobe). Put it after the GI and before the atmosphere and
/// TAA.
pub struct RtReflectionsEffect {
    pub enabled: bool,
    pub view: RtReflectionsView,
    pub intensity: f32,
    pub max_distance: f32,
    pub cone_tan: f32,
    pub cone_steps: u32,
    pub temporal_blend: f32,
    pub sky_scale: f32,
    pub alpha_test: bool,
    /// Trace the grid of triangles (on by default); off, every reflection is the voxel cone from
    /// the surface (for comparison, or where the grid is too costly).
    pub trace_grid: bool,
    /// Scale of the cost view's colours.
    pub heat_scale: f32,
    /// Count the rays' work (`stats`), a few atomics a ray.
    pub collect_stats: bool,
    /// Light the hits by the lit image where the camera sees the same point (sharp, with the
    /// direct light), the voxels elsewhere; off, always the voxels.
    pub screen_hits: bool,
    /// The hits the camera doesn't see (with `set_spot_lights`) get the voxels' indirect light too;
    /// off where the image shows no GI, so that what the mirror and the glass show matches it.
    pub hit_indirect: bool,
    /// Glass drawn before the reflections (`set_glass`).
    glass: Option<RtGlass>,
    /// The spot lights the hits the camera doesn't see are lit by (`set_spot_lights`).
    spot_lights: Option<wgpu::Buffer>,
    /// The directional and point lights they are lit by (`update_lights`): how many directional
    /// ones, then (a, b) per light as RtReflectLight holds them, and whether they changed.
    lights: (u32, Vec<[f32; 8]>),
    lights_dirty: bool,
    resolution: RtTraceResolution,
    covered_wgsl: Option<String>,
    grid: RtGridHandle,
    source: Source,
    sky_lighting: Option<wgpu::Buffer>,
    alpha_texture: Option<wgpu::TextureView>,
    frame: u32,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
    /// the counters copied last frame, mapping, and the last read
    copied: bool,
    mapping: Option<Arc<AtomicU8>>,
    stats: Option<RtReflectionStats>,
    gpu: Option<Gpu>,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

impl RtReflectionsEffect {
    /// Reflections lit by a voxel clipmap (`SceneVoxelClipmap::clipmap`).
    pub fn with_clipmap(clipmap: &VoxelClipmap, grid: RtGridHandle, options: RtReflectionsOptions) -> Self {
        let levels = clipmap.level_views().into_iter().cloned().collect();
        Self::from_source(Source::Clipmap { uniform: clipmap.uniform().clone(), levels, sampler: clipmap.sampler().clone() }, grid, options)
    }

    /// Reflections lit by a voxel volume (`SceneVoxelGi::volume`).
    pub fn with_volume(volume: &VoxelVolume, grid: RtGridHandle, options: RtReflectionsOptions) -> Self {
        let anisotropic = volume.anisotropic_views().map(|v| v.to_vec());
        Self::from_source(Source::Volume { view: volume.view().clone(), anisotropic, uniform: volume.uniform().clone(), sampler: volume.sampler().clone() }, grid, options)
    }

    fn from_source(source: Source, grid: RtGridHandle, o: RtReflectionsOptions) -> Self {
        Self {
            enabled: true,
            view: RtReflectionsView::Lit,
            intensity: o.intensity,
            max_distance: o.max_distance,
            cone_tan: o.cone_tan,
            cone_steps: o.cone_steps,
            temporal_blend: o.temporal_blend,
            sky_scale: o.sky_scale,
            alpha_test: o.alpha_test,
            trace_grid: true,
            heat_scale: 1.0,
            collect_stats: false,
            screen_hits: false,
            hit_indirect: true,
            glass: None,
            spot_lights: None,
            lights: (0, Vec::new()),
            lights_dirty: false,
            resolution: o.resolution,
            covered_wgsl: o.covered_wgsl,
            grid,
            source,
            sky_lighting: None,
            alpha_texture: None,
            frame: 0,
            prev_view_proj: None,
            last_camera_frame: None,
            copied: false,
            mapping: None,
            stats: None,
            gpu: None,
        }
    }

    /// The sky past the voxels (`SkyAtmosphereBindings::sky_lighting`, or any `SkyLighting`
    /// uniform); black without one.
    pub fn set_sky_lighting(&mut self, sky: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky.cloned();
    }

    /// The texture `kansei_rt_covered` reads (`kansei_rt_alpha_texture`).
    pub fn set_alpha_texture(&mut self, view: Option<&wgpu::TextureView>) {
        self.alpha_texture = view.cloned();
    }

    /// Light the hits the camera doesn't see by these spot lights (`Renderer::spot_lights_buffer`),
    /// shadowed by rays through the grid, plus a voxel cone along the normal, as the hybrid GI
    /// lights its hits; None: by the voxels' radiance there (the surface cache).
    pub fn set_spot_lights(&mut self, lights: Option<&wgpu::Buffer>) {
        self.spot_lights = lights.cloned();
    }

    /// Light the hits the camera doesn't see by these directional, point (and area, as point)
    /// lights too, shadowed by rays through the grid, a point light falling off as
    /// (1 - d / radius)^2 (`Scene::lights`; call it when they change, or each frame).
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        let (mut dir, mut point) = (Vec::new(), Vec::new());
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let c = l.effective_color();
                    dir.push([l.direction.x, l.direction.y, l.direction.z, 0.0, c.x, c.y, c.z, 0.0]);
                }
                Light::Point(l) => {
                    let c = l.effective_color();
                    point.push([l.position.x, l.position.y, l.position.z, l.radius, c.x, c.y, c.z, 0.0]);
                }
                Light::Area(l) => {
                    let c = l.effective_color();
                    point.push([l.position.x, l.position.y, l.position.z, l.radius, c.x, c.y, c.z, 0.0]);
                }
                Light::Spot(_) => {}
            }
        }
        let num_dir = dir.len() as u32;
        dir.extend(point);
        if (num_dir, &dir) != (self.lights.0, &self.lights.1) {
            self.lights = (num_dir, dir);
            self.lights_dirty = true;
        }
    }

    /// Draw glass (or none): see `RtGlass`.
    pub fn set_glass(&mut self, glass: Option<RtGlass>) {
        self.glass = glass;
    }

    pub fn glass(&self) -> Option<RtGlass> {
        self.glass
    }

    /// Trace at another resolution (the targets are made anew).
    pub fn set_resolution(&mut self, resolution: RtTraceResolution) {
        if resolution != self.resolution {
            self.resolution = resolution;
            if let Some(gpu) = self.gpu.as_mut() {
                gpu.targets = None;
            }
            self.reset_history();
        }
    }

    pub fn resolution(&self) -> RtTraceResolution {
        self.resolution
    }

    /// Start the accumulation over (after a cut, or a change the history should not blend through).
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }

    /// The counters of a recent frame, while `collect_stats` is on (they arrive a few frames late).
    pub fn stats(&self) -> Option<RtReflectionStats> {
        self.stats
    }

    fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let texture = |filterable, view_dimension| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension, multisampled: false };
        let d2 = wgpu::TextureViewDimension::D2;
        let depth = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false };
        let storage_tex = wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: d2 };
        let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let mut trace_entries = vec![entry(0, uniform), entry(1, depth), entry(2, texture(false, d2)), entry(3, texture(false, d2)), entry(4, storage_tex), entry(5, uniform), entry(10, texture(false, d2)), entry(11, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None }), entry(12, texture(true, d2)), entry(13, storage_tex), entry(14, filtering), entry(15, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None })];
        let clipmap = matches!(self.source, Source::Clipmap { .. });
        if clipmap {
            trace_entries.extend(crate::gi::clipmap_layout_entries(compute));
        } else {
            trace_entries.extend([entry(6, uniform), entry(7, texture(true, wgpu::TextureViewDimension::D3)), entry(8, filtering)]);
            // the anisotropic mips (the volume itself where it has none)
            trace_entries.extend((40..46).map(|b| entry(b, texture(true, wgpu::TextureViewDimension::D3))));
        }
        let trace_bgl = layout("RtReflections/Trace", &trace_entries);
        let mut grid_entries = RtGrid::layout_entries(0, compute).to_vec();
        grid_entries.extend([
            entry(3, texture(true, d2)),
            entry(4, filtering),
            entry(5, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }),
        ]);
        let grid_bgl = layout("RtReflections/Grid", &grid_entries);
        let resolve_bgl = layout(
            "RtReflections/Resolve",
            &[
                entry(0, uniform),
                entry(1, depth),
                entry(2, texture(false, d2)),
                entry(3, texture(false, d2)),
                entry(4, texture(false, d2)),
                entry(5, texture(false, d2)),
                entry(6, texture(true, d2)),
                entry(7, storage_tex),
                entry(8, storage_tex),
                entry(9, filtering),
            ],
        );
        let pipeline = |label: &str, code: String, layouts: &[&wgpu::BindGroupLayout]| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None })
        };
        let covered = self.covered_wgsl.clone().unwrap_or_else(|| DEFAULT_COVERED_WGSL.into());
        let trace = pipeline("RtReflections/Trace", trace_wgsl(clipmap, &covered), &[&trace_bgl, &grid_bgl]);
        let glass = pipeline("RtReflections/Glass", glass_wgsl(clipmap, &covered), &[&trace_bgl, &grid_bgl]);
        let resolve = pipeline("RtReflections/Resolve", resolve_wgsl(), &[&resolve_bgl]);
        use wgpu::util::DeviceExt;
        let white = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("RtReflections/White"),
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
        let sampler = |label, address| device.create_sampler(&wgpu::SamplerDescriptor { label: Some(label), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, address_mode_u: address, address_mode_v: address, ..Default::default() });
        self.gpu = Some(Gpu {
            params: buffer("RtReflections/Params", std::mem::size_of::<RtReflectParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            trace_bgl,
            grid_bgl,
            resolve_bgl,
            trace,
            glass,
            resolve,
            no_sky: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("RtReflections/NoSky"), contents: &[0u8; 256], usage: wgpu::BufferUsages::UNIFORM }),
            no_spots: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("RtReflections/NoSpots"), contents: &[0u8; 256], usage: wgpu::BufferUsages::STORAGE }),
            lights: lights_buffer(device, 0),
            white,
            alpha_sampler: sampler("RtReflections/Alpha", wgpu::AddressMode::Repeat),
            linear: sampler("RtReflections/Linear", wgpu::AddressMode::ClampToEdge),
            stats: buffer("RtReflections/Stats", 32, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC),
            staging: buffer("RtReflections/StatsReadback", 32, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST),
            grid_group: None,
            targets: None,
        });
        // (the lights into the new buffer)
        self.lights_dirty = true;
    }
}

impl PostProcessingEffect for RtReflectionsEffect {
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
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_none_or(|t| t.size != (width, height)) {
            let texture = |label, w, h| {
                device
                    .create_texture(&wgpu::TextureDescriptor {
                        label: Some(label),
                        size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba16Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                        view_formats: &[],
                    })
                    .create_view(&Default::default())
            };
            gpu.targets = Some(Targets {
                size: (width, height),
                trace: texture("RtReflections/Trace", tw, th),
                history: [texture("RtReflections/History", width, height), texture("RtReflections/History", width, height)],
                glassed: texture("RtReflections/Glassed", width, height),
                glass_history: [texture("RtReflections/GlassHistory", width, height), texture("RtReflections/GlassHistory", width, height)],
            });
            self.prev_view_proj = None;
        }
        // a frame skipped (the effect was off, a cut): no history
        let camera_frame = camera.frame();
        if self.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
        }
        self.last_camera_frame = Some(camera_frame);
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let view_proj = proj * view;
        // the directional and point lights, uploaded when they changed (into a larger buffer when
        // they outgrew it)
        if std::mem::take(&mut self.lights_dirty) {
            let mut words: Vec<f32> = vec![f32::from_bits(self.lights.0), f32::from_bits(self.lights.1.len() as u32 - self.lights.0), 0.0, 0.0];
            words.extend(self.lights.1.iter().flatten());
            let bytes: &[u8] = bytemuck::cast_slice(&words);
            if gpu.lights.size() < bytes.len() as u64 {
                gpu.lights = lights_buffer(device, self.lights.1.len());
            }
            queue.write_buffer(&gpu.lights, 0, bytes);
        }
        let direct = self.spot_lights.is_some() || !self.lights.1.is_empty();
        let mut flags = 0;
        for (on, flag) in [(self.alpha_test, 1), (self.collect_stats, 2), (self.prev_view_proj.is_some(), 4), (self.trace_grid, 8), (self.screen_hits, 16), (direct, 32), (matches!(&self.source, Source::Volume { anisotropic: Some(_), .. }), 64), (!self.hit_indirect, 128), (self.glass.is_some(), 256)] {
            if on {
                flags |= flag;
            }
        }
        let params = RtReflectParamsGpu {
            inv_proj: proj.inverse().to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            full_size: [width as f32, height as f32],
            trace_size: [tw as f32, th as f32],
            frame: self.frame,
            downscale,
            flags,
            view: self.view as u32,
            cone_tan: self.cone_tan.max(1e-3),
            cone_steps: self.cone_steps,
            max_distance: self.max_distance.max(0.0),
            intensity: self.intensity.max(0.0),
            sky_scale: self.sky_scale.max(0.0),
            blend: self.temporal_blend.clamp(0.01, 1.0),
            heat_scale: self.heat_scale,
            _pad: 0,
            view_proj: view_proj.to_cols_array(),
            glass_interfaces: self.glass.map_or(0, |g| g.interfaces.max(1)),
            glass_samples: self.glass.map_or(1, |g| g.samples.max(1)),
            _pad2: [0; 2],
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);
        let t = gpu.targets.as_ref().unwrap();
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.no_sky);
        // the glass first (its pass writes `glassed`), then the reflections over it
        let glass_on = self.glass.is_some();
        let lit = if glass_on { &t.glassed } else { input };
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: buf(&gpu.params) },
            wgpu::BindGroupEntry { binding: 1, resource: tex(depth) },
            wgpu::BindGroupEntry { binding: 2, resource: tex(&gbuffer.normal_view) },
            wgpu::BindGroupEntry { binding: 3, resource: tex(&gbuffer.albedo_view) },
            wgpu::BindGroupEntry { binding: 5, resource: buf(sky) },
            wgpu::BindGroupEntry { binding: 11, resource: buf(self.spot_lights.as_ref().unwrap_or(&gpu.no_spots)) },
            wgpu::BindGroupEntry { binding: 15, resource: buf(&gpu.lights) },
        ];
        match &self.source {
            Source::Clipmap { uniform, levels, sampler } => {
                entries.push(wgpu::BindGroupEntry { binding: 50, resource: buf(uniform) });
                entries.extend(levels.iter().enumerate().map(|(k, v)| wgpu::BindGroupEntry { binding: 51 + k as u32, resource: tex(v) }));
                entries.push(wgpu::BindGroupEntry { binding: 57, resource: wgpu::BindingResource::Sampler(sampler) });
            }
            Source::Volume { view, anisotropic, uniform, sampler } => {
                entries.push(wgpu::BindGroupEntry { binding: 6, resource: buf(uniform) });
                entries.push(wgpu::BindGroupEntry { binding: 7, resource: tex(view) });
                entries.push(wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(sampler) });
                entries.extend((0..6).map(|i| wgpu::BindGroupEntry { binding: 40 + i as u32, resource: tex(anisotropic.as_ref().map_or(view, |a| &a[i])) }));
            }
        }
        let group = |label, out: &wgpu::TextureView, screen: &wgpu::TextureView| {
            let mut e = entries.clone();
            e.push(wgpu::BindGroupEntry { binding: 4, resource: tex(out) });
            e.push(wgpu::BindGroupEntry { binding: 10, resource: tex(screen) });
            e.push(wgpu::BindGroupEntry { binding: 12, resource: tex(&t.glass_history[1 - current]) });
            e.push(wgpu::BindGroupEntry { binding: 13, resource: tex(&t.glass_history[current]) });
            e.push(wgpu::BindGroupEntry { binding: 14, resource: wgpu::BindingResource::Sampler(&gpu.linear) });
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout: &gpu.trace_bgl, entries: &e })
        };
        let trace_group = group("RtReflections/Trace", &t.trace, lit);
        let glass_group = glass_on.then(|| group("RtReflections/Glass", &t.glassed, input));
        // the grid's group, made anew when the grid's buffers or the alpha texture change
        let (buffers, generation) = self.grid.buffers();
        if gpu.grid_group.as_ref().is_none_or(|(g, alpha, _)| *g != generation || *alpha != self.alpha_texture) {
            let alpha = self.alpha_texture.as_ref().unwrap_or(&gpu.white);
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtReflections/Grid"),
                layout: &gpu.grid_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buf(&buffers[0]) },
                    wgpu::BindGroupEntry { binding: 1, resource: buf(&buffers[1]) },
                    wgpu::BindGroupEntry { binding: 2, resource: buf(&buffers[2]) },
                    wgpu::BindGroupEntry { binding: 3, resource: tex(alpha) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&gpu.alpha_sampler) },
                    wgpu::BindGroupEntry { binding: 5, resource: buf(&gpu.stats) },
                ],
            });
            gpu.grid_group = Some((generation, self.alpha_texture.clone(), group));
        }
        let resolve_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtReflections/Resolve"),
            layout: &gpu.resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: buf(&gpu.params) },
                wgpu::BindGroupEntry { binding: 1, resource: tex(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: tex(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: tex(&gbuffer.albedo_view) },
                wgpu::BindGroupEntry { binding: 4, resource: tex(lit) },
                wgpu::BindGroupEntry { binding: 5, resource: tex(&t.trace) },
                wgpu::BindGroupEntry { binding: 6, resource: tex(&t.history[1 - current]) },
                wgpu::BindGroupEntry { binding: 7, resource: tex(output) },
                wgpu::BindGroupEntry { binding: 8, resource: tex(&t.history[current]) },
                wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&gpu.linear) },
            ],
        });
        let read_stats = self.collect_stats && !self.copied && self.mapping.is_none();
        if read_stats {
            encoder.clear_buffer(&gpu.stats, 0, None);
        }
        if let Some(glass_group) = &glass_group {
            let stamp = crate::profiling::gpu_pass("Rt/Glass");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Rt/Glass"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.glass);
            pass.set_bind_group(0, glass_group, &[]);
            pass.set_bind_group(1, &gpu.grid_group.as_ref().unwrap().2, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        {
            let stamp = crate::profiling::gpu_pass("Rt/Trace");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Rt/Trace"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.trace);
            pass.set_bind_group(0, &trace_group, &[]);
            pass.set_bind_group(1, &gpu.grid_group.as_ref().unwrap().2, &[]);
            pass.dispatch_workgroups(tw.div_ceil(8), th.div_ceil(8), 1);
        }
        if read_stats {
            encoder.copy_buffer_to_buffer(&gpu.stats, 0, &gpu.staging, 0, 32);
            self.copied = true;
        }
        let stamp = crate::profiling::gpu_pass("Rt/Resolve");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Rt/Resolve"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.resolve);
        pass.set_bind_group(0, &resolve_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
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

/// Room for `n` lights after RtReflectLights' header (at least 8).
fn lights_buffer(device: &wgpu::Device, n: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtReflections/Lights"), size: 16 + 32 * n.max(8) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
}

fn tex(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

fn buf(buffer: &wgpu::Buffer) -> wgpu::BindingResource<'_> {
    buffer.as_entire_binding()
}

impl RtReflectionsEffect {
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
                    self.stats = Some(RtReflectionStats { rays: w[0], hits: w[1], cells: w[2], tests: w[3], max_cost: w[4], glass_pixels: w[5], glass_cost: w[6] });
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
