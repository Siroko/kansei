use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::froxels::{FroxelGrid, FroxelGridOptions};
use crate::lights::Light;
use crate::math::{Mat4, Vec3};
use crate::postprocessing::PostProcessingEffect;
use crate::reflections::{flip_x, mirrored_view, PlanarReflection, ReflectionFog, ReflectionFogParamsGpu};
use crate::renderers::{GBuffer, Renderer};
use crate::shadows::{CascadedShadowMap, CubeMapShadowMap, ShadowMap, SpotShadowAtlas};

const INJECT_WGSL: &str = concat!(
    include_str!("../../shaders/froxel_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_inject.wgsl"),
    include_str!("../../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../../shaders/sky_occlusion.wgsl"),
    include_str!("../../shaders/volumetric_fog_media.wgsl"),
    include_str!("../../shaders/spot_light_types.wgsl"),
    include_str!("../../shaders/volumetric_fog_spot.wgsl"),
);
const COMPOSITE_WGSL: &str = concat!(
    include_str!("../../shaders/froxel_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_composite.wgsl"),
);
/// The injection's shader with the shafts' entry point (`shafts`) appended.
const SHAFTS_WGSL: &str = concat!(
    include_str!("../../shaders/froxel_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_inject.wgsl"),
    include_str!("../../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../../shaders/sky_occlusion.wgsl"),
    include_str!("../../shaders/volumetric_fog_media.wgsl"),
    include_str!("../../shaders/spot_light_types.wgsl"),
    include_str!("../../shaders/volumetric_fog_spot.wgsl"),
    include_str!("../../shaders/volumetric_fog_shafts_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_shafts.wgsl"),
);
const SHAFTS_TEMPORAL_WGSL: &str = concat!(
    include_str!("../../shaders/volumetric_fog_shafts_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_shafts_temporal.wgsl"),
);

/// How the fog scatters the spot lights' light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SpotScattering {
    /// In the froxel grid with the fog's other light (the default). Cheap, but the grid's slices
    /// are metres deep where the beams are, so a shadow thinner than that (a trunk or grass
    /// shadowing a beam that crosses the view) fades into the lit fog around it.
    #[default]
    Froxels,
    /// Raymarched along each view ray at half resolution, `steps` samples per light over the part
    /// of the ray inside its cone, each through the light's shadow map; filtered over frames and
    /// upsampled by depth. The shadows of thin things stay sharp in the beams. The froxels keep the
    /// medium and the other lights (and a planar reflection's fog keeps the spots in its grid).
    Raymarched { steps: u32 },
}

const NO_SHADOW: u32 = u32::MAX;

pub struct VolumetricFogOptions {
    /// Froxel grid resolution, depth range and temporal settings.
    pub grid: FroxelGridOptions,
    /// Scattering density at `fog_height` (per metre).
    pub base_density: f32,
    /// Exponential falloff of density with height above `fog_height` (per metre).
    pub height_falloff: f32,
    /// Height below which density stays at `base_density`.
    pub fog_height: f32,
    /// Extinction = density * extinction_coeff (1 = no absorption).
    pub extinction_coeff: f32,
    /// Henyey-Greenstein g: 0 isotropic, > 0 forward scattering.
    pub anisotropy: f32,
    /// View distance before which there is no fog (UE's fog start distance).
    pub start_distance: f32,
    /// View depth past which the froxels hold no fog (Unreal's `VolumetricFogDistance`), at most
    /// the grid's `far`; 0: the grid's `far`. See `VolumetricFogEffect::max_distance`.
    pub max_distance: f32,
    /// Density field drift, in metres per second of `time`.
    pub wind_direction: Vec3,
    /// Radiance of a uniform sky around the fog (scatters as density * ambient). Zero matches the
    /// TS effect; set it so fog stays lit with no direct light (dusk, overcast). Far fog tends to it.
    pub ambient: Vec3,
    /// Scattering albedo: scattering = density * albedo (extinction = density * extinction_coeff).
    /// Unreal's volumetric fog maps as albedo = its albedo * its extinction scale,
    /// extinction_coeff = its extinction scale.
    pub albedo: Vec3,
    /// Scales the sky's light on the fog once a sky is bound (`set_sky_lighting`); 1 is physical.
    pub sky_ambient_scale: f32,
    /// How the spot lights scatter: in the froxels, or raymarched per pixel for sharp shafts.
    pub spot_scattering: SpotScattering,
}

impl Default for VolumetricFogOptions {
    fn default() -> Self {
        Self {
            grid: FroxelGridOptions::default(),
            base_density: 0.02,
            height_falloff: 0.1,
            fog_height: 0.0,
            extinction_coeff: 1.0,
            anisotropy: 0.6,
            start_distance: 0.0,
            max_distance: 0.0,
            wind_direction: Vec3::ZERO,
            ambient: Vec3::ZERO,
            albedo: Vec3::new(1.0, 1.0, 1.0),
            sky_ambient_scale: 1.0,
            spot_scattering: SpotScattering::Froxels,
        }
    }
}

// ── GPU layouts (must match the WGSL structs) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FogParamsGpu {
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    base_density: f32,
    wind_offset: [f32; 3],
    height_falloff: f32,
    ambient: [f32; 3],
    fog_height: f32,
    grid_near: f32,
    grid_far: f32,
    camera_near: f32,
    camera_far: f32,
    grid_w: u32,
    grid_h: u32,
    grid_d: u32,
    num_dir_lights: u32,
    num_point_lights: u32,
    has_shadow_map: u32,
    has_point_shadows: u32,
    extinction_coeff: f32,
    anisotropy: f32,
    start_distance: f32,
    jitter_frame: u32,
    skip_spots: u32,
    clip_plane: [f32; 4],
    max_distance: f32,
    _pad: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DirLightGpu {
    direction: [f32; 3],
    shadowed: u32,
    color: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PointLightGpu {
    position: [f32; 3],
    radius: f32,
    color: [f32; 3],
    shadow_layer: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeParamsGpu {
    camera_near: f32,
    camera_far: f32,
    grid_near: f32,
    grid_far: f32,
    grid_d: f32,
    screen_width: f32,
    screen_height: f32,
    shafts: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShaftParamsGpu {
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    frame: u32,
    view_forward: [f32; 3],
    steps: u32,
    size: [u32; 2],
    full_size: [u32; 2],
    grid_near: f32,
    grid_far: f32,
    grid_d: f32,
    blend: f32,
    camera_near: f32,
    camera_far: f32,
    history_valid: u32,
    _pad: f32,
}

/// The shape of a [`LocalFogVolume`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LocalFogShape {
    #[default]
    Ellipsoid,
    Box,
}

/// A local fog volume: an ellipsoid or box of mist (over a lake, in a hollow) injected into the
/// fog's froxels, after Unreal's `LocalFogVolume`. In the volume's unit shape `q`, with `r` = |q|
/// (the largest |q_i| for a box) below 1, the extinction is `radial_extinction * (1 - r^2) +
/// height_extinction * exp(-height_falloff * max(q.y - height_offset, 0))`, faded to zero over
/// the outer `edge_fade` of the radius. It scatters the fog's lights and sky with its own albedo;
/// wind and start distance leave it alone.
///
/// Unreal's `radial_fog_extinction`, `height_fog_extinction`, `height_fog_falloff`,
/// `height_fog_offset` and `fog_albedo` carry over; the shapes of the two terms are kansei's
/// own, so the look may need a trim.
#[derive(Debug, Clone, Copy)]
pub struct LocalFogVolume {
    pub shape: LocalFogShape,
    pub center: Vec3,
    /// Semi-axes of the ellipsoid, or half extents of the box, metres.
    pub radii: Vec3,
    /// Rotation about +Y, radians (as `Object3D::rotation.y`).
    pub yaw: f32,
    /// Extinction per metre at the centre, falling to zero at the surface.
    pub radial_extinction: f32,
    /// Extinction per metre at and below `height_offset`, falling off above it.
    pub height_extinction: f32,
    /// Exponential falloff per unit of the volume's half height.
    pub height_falloff: f32,
    /// Height in the unit sphere (-1 bottom, 1 top) below which the height term is at full strength.
    pub height_offset: f32,
    pub albedo: Vec3,
    /// Fraction of the radius over which the fog fades out at the surface (0: a hard edge).
    pub edge_fade: f32,
}

impl LocalFogVolume {
    /// An axis-aligned ellipsoid of `radius` across and `half_height` up and down, with Unreal's
    /// defaults otherwise: radial extinction 1, no height term, a soft edge.
    pub fn new(center: Vec3, radius: f32, half_height: f32) -> Self {
        Self {
            shape: LocalFogShape::Ellipsoid,
            center,
            radii: Vec3::new(radius, half_height, radius),
            yaw: 0.0,
            radial_extinction: 1.0,
            height_extinction: 0.0,
            height_falloff: 1000.0,
            height_offset: 0.0,
            albedo: Vec3::new(1.0, 1.0, 1.0),
            edge_fade: 0.25,
        }
    }

    /// A box of `half_extents`, with radial extinction 1, no height term and a soft edge.
    pub fn new_box(center: Vec3, half_extents: Vec3) -> Self {
        Self { shape: LocalFogShape::Box, radii: half_extents, ..Self::new(center, 1.0, 1.0) }
    }

    /// Extinction per metre at a world-space point, as the fog shader computes it.
    pub fn extinction_at(&self, p: Vec3) -> f32 {
        let g = self.gpu();
        let d = glam::Vec3::new(p.x - g.center[0], p.y - g.center[1], p.z - g.center[2]);
        let q = glam::Vec3::new(g.cos_yaw * d.x - g.sin_yaw * d.z, d.y, g.sin_yaw * d.x + g.cos_yaw * d.z) * glam::Vec3::from(g.inv_radii);
        let r = if g.shape == 1 { q.abs().max_element() } else { q.length() };
        if r >= 1.0 {
            return 0.0;
        }
        let r2 = r * r;
        let edge = if g.edge_fade > 0.0 {
            let t = ((1.0 - r) / g.edge_fade).clamp(0.0, 1.0);
            t * t * (3.0 - 2.0 * t)
        } else {
            1.0
        };
        let radial = g.radial_extinction * (1.0 - r2);
        let height = g.height_extinction * (-g.height_falloff * (q.y - g.height_offset).max(0.0)).exp();
        (radial + height) * edge
    }

    fn gpu(&self) -> LocalFogVolumeGpu {
        let inv = |r: f32| 1.0 / r.max(1e-3);
        LocalFogVolumeGpu {
            center: [self.center.x, self.center.y, self.center.z],
            radial_extinction: self.radial_extinction.max(0.0),
            inv_radii: [inv(self.radii.x), inv(self.radii.y), inv(self.radii.z)],
            height_extinction: self.height_extinction.max(0.0),
            albedo: [self.albedo.x, self.albedo.y, self.albedo.z],
            height_falloff: self.height_falloff,
            cos_yaw: self.yaw.cos(),
            sin_yaw: self.yaw.sin(),
            height_offset: self.height_offset,
            edge_fade: self.edge_fade.clamp(0.0, 1.0),
            shape: (self.shape == LocalFogShape::Box) as u32,
            _pad: [0; 3],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct LocalFogVolumeGpu {
    center: [f32; 3],
    radial_extinction: f32,
    inv_radii: [f32; 3],
    height_extinction: f32,
    albedo: [f32; 3],
    height_falloff: f32,
    cos_yaw: f32,
    sin_yaw: f32,
    height_offset: f32,
    edge_fade: f32,
    shape: u32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FogMediaParamsGpu {
    albedo: [f32; 3],
    sky_ambient_scale: f32,
    num_volumes: u32,
    has_sky_lighting: u32,
    _pad: [u32; 2],
}

/// A half-resolution shafts target (rgb light, a linear depth).
fn shafts_target(device: &wgpu::Device, label: &str, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

/// Size of the WGSL `SkyLighting` struct (atmosphere::SKY_LIGHTING_WGSL).
const SKY_LIGHTING_BYTES: u64 = std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64;

struct Gpu {
    grid: FroxelGrid,
    inject_pipeline: wgpu::ComputePipeline,
    inject_bgl: wgpu::BindGroupLayout,
    inject_bg: Option<wgpu::BindGroup>,
    composite_pipeline: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    fog_params: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    dir_lights: wgpu::Buffer,
    point_lights: wgpu::Buffer,
    accum_sampler: wgpu::Sampler,
    dummy_depth: wgpu::TextureView,
    dummy_atlas: wgpu::TextureView,
    dummy_vp: wgpu::Buffer,
    media_params: wgpu::Buffer,
    volumes: wgpu::Buffer,
    dummy_sky_lighting: wgpu::Buffer,
    /// Stand-ins for the sky occlusion: a 1-texel volume, and its parameters off
    dummy_occlusion_volume: wgpu::TextureView,
    dummy_occlusion_params: wgpu::Buffer,
    dummy_spot_lights: wgpu::Buffer,
    dummy_spot_atlas: wgpu::TextureView,
    spot_sampler: wgpu::Sampler,
    reflection: Option<ReflectionGpu>,
    shafts: ShaftsGpu,
}

/// The raymarched spot-light shafts (SpotScattering::Raymarched).
struct ShaftsGpu {
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    /// Bound in the composite while the shafts are off.
    none: wgpu::TextureView,
    /// (width, height, trace, history A and B), at half the image's resolution.
    targets: Option<(u32, u32, wgpu::TextureView, [wgpu::TextureView; 2])>,
    frame: u32,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
}

/// The fog as a planar reflection sees it: a second froxel grid, built from the mirrored camera
/// with the fog below the mirror left out.
struct ReflectionGpu {
    grid: FroxelGrid,
    fog_params: wgpu::Buffer,
    inject_bg: Option<wgpu::BindGroup>,
    /// The volume's lookup parameters, staged here each frame and copied into the shared buffer
    /// after the volume is built, so the reflection reads parameters and volume from one frame.
    staging: wgpu::Buffer,
    shared: ReflectionFog,
}

/// Froxel volumetric fog, ported from the TS `VolumetricFogEffect`.
///
/// Per frame it injects height fog lit by the scene's volumetric lights (with directional and
/// point shadows) into a [`FroxelGrid`], optionally blends it with reprojected history,
/// integrates it front to back, and composites it over the scene by depth.
///
/// ```ignore
/// let mut fog = VolumetricFogEffect::new(VolumetricFogOptions { base_density: 0.03, ..Default::default() });
/// fog.set_shadow_map(renderer.shadow_map());          // optional: shafts from the sun
/// fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas()); // beams
/// fog.update_lights(scene.lights());                  // each frame, or when lights change
/// ```
///
/// With a temporal grid, the injection samples a different point of each froxel every frame, so
/// the history resolves shadow detail (shafts, the shadows of trunks in beams) finer than the grid,
/// across the view. Along the view a froxel is a few metres deep at a few tens of metres, too deep
/// for the shadow of a trunk or of grass in a beam: `spot_scattering:
/// SpotScattering::Raymarched` marches the spot lights' beams per pixel instead.
pub struct VolumetricFogEffect {
    pub base_density: f32,
    pub height_falloff: f32,
    pub fog_height: f32,
    pub extinction_coeff: f32,
    pub anisotropy: f32,
    pub start_distance: f32,
    /// View depth past which the froxels hold no fog (Unreal's `VolumetricFogDistance`, which its
    /// shots change): at most the grid's `far`, 0 for the grid's `far`. Change it any frame; the
    /// grid stays. Start a `HeightFogEffect` there (`volumetric_fog_distance = reach()`), as
    /// Unreal's analytic fog takes over where its volumetric fog ends.
    pub max_distance: f32,
    pub wind_direction: Vec3,
    pub ambient: Vec3,
    pub albedo: Vec3,
    pub sky_ambient_scale: f32,
    /// Local fog volumes, uploaded every frame (keep it to tens).
    pub local_volumes: Vec<LocalFogVolume>,
    /// Seconds, drives the wind offset. The effect has no clock of its own; set it per frame.
    pub time: f32,
    /// How the spot lights scatter: in the froxels, or raymarched per pixel for sharp shafts.
    pub spot_scattering: SpotScattering,
    /// Temporal jitter index of the injection (1..=1024), advanced every frame.
    frame: u32,
    grid_options: FroxelGridOptions,
    dir_data: Vec<DirLightGpu>,
    point_data: Vec<PointLightGpu>,
    shadow_map: Option<(wgpu::TextureView, wgpu::Buffer)>,
    point_shadows: Option<wgpu::TextureView>,
    sky_lighting: Option<wgpu::Buffer>,
    /// The sky occlusion's volume and parameters (`set_sky_occlusion`)
    sky_occlusion: Option<(wgpu::TextureView, wgpu::Buffer)>,
    spot_lights: Option<wgpu::Buffer>,
    spot_shadows: Option<wgpu::TextureView>,
    /// The mirror plane (unit normal, d) of the reflection fog, if any.
    reflection_plane: Option<(glam::Vec3, f32)>,
    lights_dirty: bool,
    bindings_dirty: bool,
    gpu: Option<Gpu>,
}

impl VolumetricFogEffect {
    pub fn new(options: VolumetricFogOptions) -> Self {
        Self {
            base_density: options.base_density,
            height_falloff: options.height_falloff,
            fog_height: options.fog_height,
            extinction_coeff: options.extinction_coeff,
            anisotropy: options.anisotropy,
            start_distance: options.start_distance,
            max_distance: options.max_distance,
            wind_direction: options.wind_direction,
            ambient: options.ambient,
            albedo: options.albedo,
            sky_ambient_scale: options.sky_ambient_scale,
            local_volumes: Vec::new(),
            time: 0.0,
            spot_scattering: options.spot_scattering,
            frame: 1,
            grid_options: options.grid,
            dir_data: Vec::new(),
            point_data: Vec::new(),
            shadow_map: None,
            point_shadows: None,
            sky_lighting: None,
            sky_occlusion: None,
            spot_lights: None,
            spot_shadows: None,
            reflection_plane: None,
            lights_dirty: true,
            bindings_dirty: true,
            gpu: None,
        }
    }

    /// The froxel grid (available after the effect's first frame).
    /// The view depth the froxels hold fog to: `max_distance`, or the grid's `far` when it is 0 or
    /// beyond it. For `HeightFogEffect::volumetric_fog_distance`.
    pub fn reach(&self) -> f32 {
        if self.max_distance > 0.0 { self.max_distance.min(self.grid_options.far) } else { self.grid_options.far }
    }

    pub fn froxel_grid(&self) -> Option<&FroxelGrid> {
        self.gpu.as_ref().map(|g| &g.grid)
    }

    /// Drop the temporal history; call on camera cuts. No-op without a temporal grid.
    pub fn reset_history(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.grid.reset_history();
            g.shafts.prev_view_proj = None;
            if let Some(r) = &mut g.reflection {
                r.grid.reset_history();
            }
        }
    }

    /// Collect the volumetric lights. Directional lights cast shafts through the renderer's
    /// shadow map if they are the scene's first directional light and `cast_shadow` is set
    /// (that is the light the renderer's `ShadowMap` follows); the first shadow-casting point
    /// light uses the cube shadow atlas. Area lights are treated as point lights at their
    /// position, as in the TS effect. Spot lights are not collected here: the fog reads them,
    /// with their shadows, from the renderer (`set_spot_lights`).
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        self.dir_data.clear();
        self.point_data.clear();
        let mut seen_directional = false;
        let mut point_shadow_assigned = false;
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let first = !seen_directional;
                    seen_directional = true;
                    if l.volumetric {
                        let c = l.effective_color();
                        self.dir_data.push(DirLightGpu {
                            direction: [l.direction.x, l.direction.y, l.direction.z],
                            shadowed: (first && l.cast_shadow) as u32,
                            color: [c.x, c.y, c.z],
                            _pad: 0.0,
                        });
                    }
                }
                Light::Point(l) => {
                    let shadow_layer = if l.cast_shadow && !point_shadow_assigned {
                        point_shadow_assigned = true;
                        0 // CubeMapShadowMap renders the first shadow-casting light into layers 0..6
                    } else {
                        NO_SHADOW
                    };
                    let c = if l.volumetric { l.effective_color() } else { Vec3::ZERO };
                    self.point_data.push(PointLightGpu {
                        position: [l.position.x, l.position.y, l.position.z],
                        radius: l.radius,
                        color: [c.x, c.y, c.z],
                        shadow_layer,
                    });
                }
                Light::Area(l) => {
                    let c = l.effective_color();
                    self.point_data.push(PointLightGpu {
                        position: [l.position.x, l.position.y, l.position.z],
                        radius: l.radius,
                        color: [c.x, c.y, c.z],
                        shadow_layer: NO_SHADOW,
                    });
                }
                // spot lights come from the renderer's buffer (set_spot_lights), with its shadows
                Light::Spot(_) => {}
            }
        }
        self.lights_dirty = true;
    }

    /// Use the renderer's directional shadow map (`Renderer::shadow_map()`) for light shafts.
    /// The light's view-projection is read from the map's own uniform buffer, so it is always the
    /// one the renderer computed this frame.
    pub fn set_shadow_map(&mut self, shadow_map: Option<&ShadowMap>) {
        self.shadow_map = shadow_map.and_then(|sm| Some((sm.depth_view.clone()?, sm.light_vp_buf.clone()?)));
        self.bindings_dirty = true;
    }

    /// Shafts from the renderer's cascaded shadow map (`Renderer::cascaded_shadow_map()`): its
    /// widest cascade, whose matrix is rewritten every frame. Use instead of `set_shadow_map`.
    pub fn set_cascaded_shadow_map(&mut self, csm: Option<&CascadedShadowMap>) {
        self.shadow_map = csm.map(|c| (c.far_view.clone(), c.far_view_proj.clone()));
        self.bindings_dirty = true;
    }

    /// Use the renderer's point-light cube shadows (`Renderer::cubemap_shadow_map()`).
    pub fn set_point_shadows(&mut self, cube: Option<&CubeMapShadowMap>) {
        self.point_shadows = cube.map(|c| c.distance_view.clone());
        self.bindings_dirty = true;
    }

    /// Light the fog with a sky: `SkyAtmosphere::bindings().sky_lighting`. The sky's radiance,
    /// convolved with the fog's phase function, scatters in every froxel (times
    /// `sky_ambient_scale`), on top of `ambient`, so the fog takes the sky's colour and stays lit
    /// at dusk. Put the `AtmosphereEffect` before the fog in the chain, so the fog lies in front of
    /// the sky and its aerial perspective.
    pub fn set_sky_lighting(&mut self, sky_lighting: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky_lighting.cloned();
        self.bindings_dirty = true;
    }

    /// Dim the sky's light on the fog by how much of the sky each froxel sees
    /// (`Renderer::sky_occlusion`, `shadows::SKY_OCCLUSION_WGSL`'s `skyVisibility`), as Unreal's
    /// Lumen occludes the sky light its volumetric fog receives: under the canopy, and where the
    /// trees round a clearing hide the horizon. The fog's other lights are not affected.
    pub fn set_sky_occlusion(&mut self, sky_occlusion: Option<&crate::shadows::SkyOcclusion>) {
        self.sky_occlusion = sky_occlusion.map(|s| (s.volume.clone(), s.params.clone()));
        self.bindings_dirty = true;
    }

    /// The fog as `reflection` sees it, for `PlanarReflection::set_fog`: every frame the effect
    /// also builds a froxel volume from the camera mirrored in the reflection's plane, with the
    /// same grid, lights and media but only the fog above the plane, and the reflection
    /// composites it over what it saw. A lake then mirrors the glow of beams and lamps in the mist
    /// (the main fog already covers the camera's path to the water). It costs about one more
    /// injection, and nothing while the reflection is disabled (`PlanarReflection::enabled`) or
    /// the camera is under its plane. Call again if the plane moves.
    ///
    /// ```ignore
    /// let fog_in_reflection = fog.reflection_fog(&renderer, &reflection);
    /// reflection.set_fog(&renderer, Some(&fog_in_reflection));
    /// renderer.add_planar_reflection(reflection);
    /// ```
    pub fn reflection_fog(&mut self, renderer: &Renderer, reflection: &PlanarReflection) -> ReflectionFog {
        self.mirrored_fog(renderer.device(), renderer.queue(), reflection.plane())
    }

    /// The froxel grid the reflection fog is built in (available after `reflection_fog`).
    pub fn reflection_froxel_grid(&self) -> Option<&FroxelGrid> {
        self.gpu.as_ref().and_then(|g| g.reflection.as_ref()).map(|r| &r.grid)
    }

    /// `reflection_fog` for the plane `n·p + d = 0` (`n` unit length).
    pub(crate) fn mirrored_fog(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, plane: (glam::Vec3, f32)) -> ReflectionFog {
        if self.gpu.is_none() {
            self.init_gpu(device, queue);
        }
        self.reflection_plane = Some(plane);
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.reflection.is_none() {
            let buffer = |label: &str, size: usize, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage, mapped_at_creation: false })
            };
            let params_size = std::mem::size_of::<ReflectionFogParamsGpu>();
            let grid = FroxelGrid::new(device, queue, &self.grid_options);
            let shared = ReflectionFog {
                drawn: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true)),
                volume: grid.accum_view().clone(),
                params: buffer("VolumetricFog/ReflectionFogParams", params_size, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            };
            gpu.reflection = Some(ReflectionGpu {
                grid,
                fog_params: buffer("VolumetricFog/ReflectionParams", std::mem::size_of::<FogParamsGpu>(), wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
                inject_bg: None,
                staging: buffer("VolumetricFog/ReflectionFogStaging", params_size, wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST),
                shared,
            });
            self.bindings_dirty = true;
        }
        gpu.reflection.as_ref().unwrap().shared.clone()
    }

    /// Scatter the renderer's spot lights (`Renderer::spot_lights_buffer()`, rewritten every
    /// frame) in their cones, shadowed by its spot shadow atlas (`Renderer::spot_shadow_atlas()`).
    /// Call again after `Renderer::enable_spot_shadows`. Each light's `volumetric_scale` scales
    /// its scattering; 0 keeps it out of the fog.
    pub fn set_spot_lights(&mut self, lights: Option<&wgpu::Buffer>, shadow_atlas: Option<&SpotShadowAtlas>) {
        self.spot_lights = lights.cloned();
        self.spot_shadows = shadow_atlas.map(|a| a.array_view.clone());
        self.bindings_dirty = true;
    }

    fn init_gpu(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let grid = FroxelGrid::new(device, queue, &self.grid_options);

        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let texture_2d = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let storage_2d = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let spot_atlas_entry = wgpu::BindGroupLayoutEntry {
            binding: 8,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Depth,
                view_dimension: wgpu::TextureViewDimension::D2Array,
                multisampled: false,
            },
            count: None,
        };
        let spot_sampler_entry = wgpu::BindGroupLayoutEntry { binding: 9, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison), count: None };
        // the shafts: the injection's medium, lights and parameters, the scene's depth, the
        // accumulated grid (for the transmittance), their target and parameters
        let shafts_trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog/ShaftsBGL"),
            entries: &[
                uniform(2),
                storage(7),
                spot_atlas_entry,
                spot_sampler_entry,
                uniform(10),
                storage(11),
                uniform(12),
                wgpu::BindGroupLayoutEntry {
                    binding: 13,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 15, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                storage_2d(16),
                uniform(17),
            ],
        });
        let shafts_temporal_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog/ShaftsTemporalBGL"),
            entries: &[
                uniform(0),
                texture_2d(1, false),
                texture_2d(2, true),
                storage_2d(3),
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
            ],
        });
        let inject_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog/InjectBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D3,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform(2),
                storage(3),
                storage(4),
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                uniform(6),
                uniform(10),
                storage(11),
                uniform(12),
                storage(7),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                    count: None,
                },
                // the sky occlusion (volumetric_fog_media.wgsl)
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry { binding: 19, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                uniform(20),
            ],
        });
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VolumetricFog/CompositeBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: compute,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform(5),
                texture_2d(6, false),
            ],
        });

        let entry_pipeline = |label: &str, code: &str, entry: &str, bgl: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline = |label: &str, code: &str, bgl: &wgpu::BindGroupLayout| entry_pipeline(label, code, "main", bgl);
        let buffer = |label: &str, size: usize, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: usage | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        // Fallbacks bound when no shadow map is set: a 1x1 depth texture (never sampled, the
        // lookups are gated by flags), a 1x1x6 distance atlas, and an identity light VP.
        let dummy_depth = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VolumetricFog/DummyShadowDepth"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let dummy_atlas = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VolumetricFog/DummyPointShadow"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let dummy_vp = buffer("VolumetricFog/DummyLightVP", 64, wgpu::BufferUsages::UNIFORM);
        queue.write_buffer(&dummy_vp, 0, bytemuck::cast_slice(Mat4::identity().as_slice()));
        // no spot lights: a buffer whose count is 0 (zero-initialised), and a 1x1 atlas
        let dummy_spot_lights = buffer(
            "VolumetricFog/DummySpotLights",
            16 + std::mem::size_of::<crate::lights::spot_lights_gpu::SpotLightGpu>(),
            wgpu::BufferUsages::STORAGE,
        );
        let dummy_spot_atlas = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VolumetricFog/DummySpotShadow"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: SpotShadowAtlas::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let spot_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("VolumetricFog/SpotShadowSampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let volumes = buffer("VolumetricFog/LocalVolumes", std::mem::size_of::<LocalFogVolumeGpu>(), wgpu::BufferUsages::STORAGE);
        let dummy_sky_lighting = buffer("VolumetricFog/NoSkyLighting", SKY_LIGHTING_BYTES as usize, wgpu::BufferUsages::UNIFORM);
        // no sky occlusion: its parameters zero (off, so skyVisibility is 1) and a 1-texel volume
        let dummy_occlusion_params = buffer("VolumetricFog/NoSkyOcclusion", 32, wgpu::BufferUsages::UNIFORM);
        let dummy_occlusion_volume = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("VolumetricFog/NoSkyOcclusionVolume"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D3,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let media_params = buffer("VolumetricFog/MediaParams", std::mem::size_of::<FogMediaParamsGpu>(), wgpu::BufferUsages::UNIFORM);

        self.gpu = Some(Gpu {
            grid,
            inject_pipeline: pipeline("VolumetricFog/Inject", INJECT_WGSL, &inject_bgl),
            composite_pipeline: pipeline("VolumetricFog/Composite", COMPOSITE_WGSL, &composite_bgl),
            inject_bgl,
            inject_bg: None,
            composite_bgl,
            fog_params: buffer("VolumetricFog/Params", std::mem::size_of::<FogParamsGpu>(), wgpu::BufferUsages::UNIFORM),
            composite_params: buffer("VolumetricFog/CompositeParams", std::mem::size_of::<CompositeParamsGpu>(), wgpu::BufferUsages::UNIFORM),
            dir_lights: buffer("VolumetricFog/DirLights", std::mem::size_of::<DirLightGpu>(), wgpu::BufferUsages::STORAGE),
            point_lights: buffer("VolumetricFog/PointLights", std::mem::size_of::<PointLightGpu>(), wgpu::BufferUsages::STORAGE),
            accum_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("VolumetricFog/AccumSampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            dummy_depth,
            dummy_atlas,
            dummy_vp,
            media_params,
            volumes,
            dummy_sky_lighting,
            dummy_occlusion_volume,
            dummy_occlusion_params,
            dummy_spot_lights,
            dummy_spot_atlas,
            spot_sampler,
            reflection: None,
            shafts: ShaftsGpu {
                trace: entry_pipeline("VolumetricFog/Shafts", SHAFTS_WGSL, "shafts", &shafts_trace_bgl),
                trace_bgl: shafts_trace_bgl,
                temporal: pipeline("VolumetricFog/ShaftsTemporal", SHAFTS_TEMPORAL_WGSL, &shafts_temporal_bgl),
                temporal_bgl: shafts_temporal_bgl,
                params: buffer("VolumetricFog/ShaftParams", std::mem::size_of::<ShaftParamsGpu>(), wgpu::BufferUsages::UNIFORM),
                none: shafts_target(device, "VolumetricFog/NoShafts", 1, 1),
                targets: None,
                frame: 0,
                prev_view_proj: None,
                last_camera_frame: None,
            },
        });
        self.lights_dirty = true;
        self.bindings_dirty = true;
    }

    /// Grow the light storage buffers if needed and upload the packed lights.
    fn upload_lights(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(gpu) = &mut self.gpu else { return };
        fn fit<T: Pod>(device: &wgpu::Device, buf: &mut wgpu::Buffer, data: &[T], label: &str) -> bool {
            let needed = (std::mem::size_of_val(data) as u64).max(std::mem::size_of::<T>() as u64);
            let grown = needed > buf.size();
            if grown {
                *buf = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(label),
                    size: needed.next_power_of_two(),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
            }
            grown
        }
        let grew_dir = fit(device, &mut gpu.dir_lights, &self.dir_data, "VolumetricFog/DirLights");
        let grew_point = fit(device, &mut gpu.point_lights, &self.point_data, "VolumetricFog/PointLights");
        if grew_dir || grew_point {
            self.bindings_dirty = true;
        }
        if !self.dir_data.is_empty() {
            queue.write_buffer(&gpu.dir_lights, 0, bytemuck::cast_slice(&self.dir_data));
        }
        if !self.point_data.is_empty() {
            queue.write_buffer(&gpu.point_lights, 0, bytemuck::cast_slice(&self.point_data));
        }
        self.lights_dirty = false;
    }

    /// Upload the media parameters and the local volumes, growing the volume buffer if needed.
    fn upload_media(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(gpu) = &mut self.gpu else { return };
        let volumes: Vec<LocalFogVolumeGpu> = self.local_volumes.iter().map(LocalFogVolume::gpu).collect();
        let needed = std::mem::size_of_val(volumes.as_slice()) as u64;
        if needed > gpu.volumes.size() {
            gpu.volumes = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("VolumetricFog/LocalVolumes"),
                size: needed.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.bindings_dirty = true;
        }
        if !volumes.is_empty() {
            queue.write_buffer(&gpu.volumes, 0, bytemuck::cast_slice(&volumes));
        }
        let params = FogMediaParamsGpu {
            albedo: [self.albedo.x, self.albedo.y, self.albedo.z],
            sky_ambient_scale: self.sky_ambient_scale,
            num_volumes: volumes.len() as u32,
            has_sky_lighting: self.sky_lighting.is_some() as u32,
            _pad: [0; 2],
        };
        queue.write_buffer(&gpu.media_params, 0, bytemuck::bytes_of(&params));
    }

    fn rebuild_inject_bind_group(&mut self, device: &wgpu::Device) {
        let Some(gpu) = &mut self.gpu else { return };
        let (depth, vp) = match &self.shadow_map {
            Some((view, buf)) => (view, buf),
            None => (&gpu.dummy_depth, &gpu.dummy_vp),
        };
        let atlas = self.point_shadows.as_ref().unwrap_or(&gpu.dummy_atlas);
        let sky_lighting = self.sky_lighting.as_ref().unwrap_or(&gpu.dummy_sky_lighting);
        let spot_lights = self.spot_lights.as_ref().unwrap_or(&gpu.dummy_spot_lights);
        let spot_atlas = self.spot_shadows.as_ref().unwrap_or(&gpu.dummy_spot_atlas);
        let (occlusion_volume, occlusion_params) = match &self.sky_occlusion {
            Some((volume, params)) => (volume, params),
            None => (&gpu.dummy_occlusion_volume, &gpu.dummy_occlusion_params),
        };
        let group = |label: &str, output: &wgpu::TextureView, params: &wgpu::Buffer| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &gpu.inject_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(output) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                    wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: gpu.dir_lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 4, resource: gpu.point_lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(atlas) },
                    wgpu::BindGroupEntry { binding: 6, resource: vp.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 10, resource: gpu.media_params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 11, resource: gpu.volumes.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 12, resource: sky_lighting.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 7, resource: spot_lights.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::TextureView(spot_atlas) },
                    wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::Sampler(&gpu.spot_sampler) },
                    wgpu::BindGroupEntry { binding: 18, resource: wgpu::BindingResource::TextureView(occlusion_volume) },
                    wgpu::BindGroupEntry { binding: 19, resource: wgpu::BindingResource::Sampler(&gpu.accum_sampler) },
                    wgpu::BindGroupEntry { binding: 20, resource: occlusion_params.as_entire_binding() },
                ],
            })
        };
        let main = group("VolumetricFog/InjectBG", gpu.grid.scatter_extinction_view(), &gpu.fog_params);
        let reflection = gpu.reflection.as_ref().map(|r| group("VolumetricFog/ReflectionInjectBG", r.grid.scatter_extinction_view(), &r.fog_params));
        gpu.inject_bg = Some(main);
        if let Some(r) = &mut gpu.reflection {
            r.inject_bg = reflection;
        }
        self.bindings_dirty = false;
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 4] {
        [
            ("volumetric_fog_inject", INJECT_WGSL),
            ("volumetric_fog_composite", COMPOSITE_WGSL),
            ("volumetric_fog_shafts", SHAFTS_WGSL),
            ("volumetric_fog_shafts_temporal", SHAFTS_TEMPORAL_WGSL),
        ]
    }
}

impl PostProcessingEffect for VolumetricFogEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        // GPU resources need the queue too; they are created on the first render().
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer,
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
        if self.lights_dirty {
            self.upload_lights(device, queue);
        }
        self.upload_media(device, queue);
        if self.bindings_dirty {
            self.rebuild_inject_bind_group(device);
        }
        // the spots raymarched per pixel instead of in the froxels (only with spot lights bound)
        let raymarch_steps = match self.spot_scattering {
            SpotScattering::Raymarched { steps } if self.spot_lights.is_some() => Some(steps.max(1)),
            _ => None,
        };
        let gpu = self.gpu.as_mut().unwrap();

        let vp = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let inv_vp = vp.inverse();
        let cam = camera.inverse_view_matrix.to_glam().w_axis;
        let wind = self.wind_direction * self.time;
        let grid = &gpu.grid;
        let params = FogParamsGpu {
            inv_view_proj: inv_vp.to_cols_array(),
            camera_pos: [cam.x, cam.y, cam.z],
            base_density: self.base_density,
            wind_offset: [wind.x, wind.y, wind.z],
            height_falloff: self.height_falloff,
            ambient: [self.ambient.x, self.ambient.y, self.ambient.z],
            fog_height: self.fog_height,
            grid_near: grid.near(),
            grid_far: grid.far(),
            camera_near: camera.near,
            camera_far: camera.far,
            grid_w: grid.grid_w(),
            grid_h: grid.grid_h(),
            grid_d: grid.grid_d(),
            num_dir_lights: self.dir_data.len() as u32,
            num_point_lights: self.point_data.len() as u32,
            has_shadow_map: self.shadow_map.is_some() as u32,
            has_point_shadows: self.point_shadows.is_some() as u32,
            extinction_coeff: self.extinction_coeff,
            anisotropy: self.anisotropy,
            start_distance: self.start_distance,
            jitter_frame: if grid.is_temporal() { self.frame } else { 0 },
            skip_spots: raymarch_steps.is_some() as u32,
            clip_plane: [0.0, 0.0, 0.0, 1.0],
            // the fog's reach (`reach`)
            max_distance: if self.max_distance > 0.0 { self.max_distance.min(grid.far()) } else { grid.far() },
            _pad: [0.0; 3],
        };
        self.frame = self.frame % 1024 + 1;
        queue.write_buffer(&gpu.fog_params, 0, bytemuck::bytes_of(&params));
        let composite = CompositeParamsGpu {
            camera_near: camera.near,
            camera_far: camera.far,
            grid_near: grid.near(),
            grid_far: grid.far(),
            grid_d: grid.grid_d() as f32,
            screen_width: width as f32,
            screen_height: height as f32,
            shafts: if raymarch_steps.is_some() { 1.0 } else { 0.0 },
        };
        queue.write_buffer(&gpu.composite_params, 0, bytemuck::bytes_of(&composite));

        // 1. inject density + lighting into the froxels
        {
            let (w, h, d) = (grid.grid_w(), grid.grid_h(), grid.grid_d());
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/Inject"), timestamp_writes: crate::profiling::gpu_pass("VolumetricFog/Inject").as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.inject_pipeline);
            pass.set_bind_group(0, gpu.inject_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }

        // 2. temporal reprojection (no-op unless temporal), 3. front-to-back accumulation
        gpu.grid.temporal_blend(queue, encoder, &Mat4::from(inv_vp), &Mat4::from(vp), camera.near, camera.far);
        gpu.grid.accumulate(encoder);

        // the same from the camera mirrored in the reflection's plane, above the plane only
        if let (Some(r), Some((n, d))) = (gpu.reflection.as_mut(), self.reflection_plane) {
            // skipped whenever the reflection is not drawn (disabled, or the camera under its plane)
            let above = n.dot(glam::Vec3::new(cam.x, cam.y, cam.z)) + d > 0.0 && r.shared.drawn.load(std::sync::atomic::Ordering::Relaxed);
            let view = mirrored_view(camera.view_matrix.to_glam(), n, d);
            let m_vp = flip_x() * camera.projection_matrix.to_glam() * view;
            let m_inv = m_vp.inverse();
            let m_cam = view.inverse().w_axis;
            let mut mirrored = params;
            mirrored.inv_view_proj = m_inv.to_cols_array();
            mirrored.camera_pos = [m_cam.x, m_cam.y, m_cam.z];
            mirrored.clip_plane = [n.x, n.y, n.z, d];
            // the reflection keeps the spots in its grid
            mirrored.skip_spots = 0;
            queue.write_buffer(&r.fog_params, 0, bytemuck::bytes_of(&mirrored));
            if above {
                let (gw, gh, gd) = (r.grid.grid_w(), r.grid.grid_h(), r.grid.grid_d());
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/ReflectionInject"), timestamp_writes: crate::profiling::gpu_pass("VolumetricFog/ReflectionInject").as_ref().map(crate::profiling::PassStamp::compute) });
                pass.set_pipeline(&gpu.inject_pipeline);
                pass.set_bind_group(0, r.inject_bg.as_ref().unwrap(), &[]);
                pass.dispatch_workgroups(gw.div_ceil(4), gh.div_ceil(4), gd.div_ceil(4));
                drop(pass);
                r.grid.temporal_blend(queue, encoder, &Mat4::from(m_inv), &Mat4::from(m_vp), camera.near, camera.far);
                r.grid.accumulate(encoder);
            }
            let lookup = ReflectionFogParamsGpu {
                view_proj: m_vp.to_cols_array(),
                grid_near: r.grid.near(),
                grid_far: r.grid.far(),
                grid_d: r.grid.grid_d() as f32,
                enabled: above as u32,
            };
            queue.write_buffer(&r.staging, 0, bytemuck::bytes_of(&lookup));
            encoder.copy_buffer_to_buffer(&r.staging, 0, &r.shared.params, 0, std::mem::size_of::<ReflectionFogParamsGpu>() as u64);
        }

        // 4. the spot lights' shafts, raymarched per pixel at half resolution
        let mut shafts_view = None;
        if let Some(steps) = raymarch_steps {
            let (sw, sh) = (width.div_ceil(2), height.div_ceil(2));
            let sg = &mut gpu.shafts;
            if sg.targets.as_ref().is_none_or(|t| t.0 != sw || t.1 != sh) {
                let history = [shafts_target(device, "VolumetricFog/ShaftsHistoryA", sw, sh), shafts_target(device, "VolumetricFog/ShaftsHistoryB", sw, sh)];
                sg.targets = Some((sw, sh, shafts_target(device, "VolumetricFog/ShaftsTrace", sw, sh), history));
                sg.prev_view_proj = None;
            }
            // a gap in the camera's frames (a cut, or the shafts were off) invalidates the history
            let camera_frame = camera.frame();
            if sg.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
                sg.prev_view_proj = None;
            }
            sg.last_camera_frame = Some(camera_frame);
            let forward = -camera.inverse_view_matrix.to_glam().z_axis.truncate().normalize();
            let shaft_params = ShaftParamsGpu {
                inv_view_proj: inv_vp.to_cols_array(),
                prev_view_proj: sg.prev_view_proj.unwrap_or(vp).to_cols_array(),
                camera_pos: [cam.x, cam.y, cam.z],
                frame: sg.frame,
                view_forward: forward.to_array(),
                steps,
                size: [sw, sh],
                full_size: [width, height],
                grid_near: composite.grid_near,
                grid_far: composite.grid_far,
                grid_d: composite.grid_d,
                blend: self.grid_options.blend_factor.clamp(0.05, 1.0),
                camera_near: camera.near,
                camera_far: camera.far,
                history_valid: sg.prev_view_proj.is_some() as u32,
                _pad: 0.0,
            };
            queue.write_buffer(&sg.params, 0, bytemuck::bytes_of(&shaft_params));
            let current = (sg.frame % 2) as usize;
            sg.frame = sg.frame.wrapping_add(1);
            sg.prev_view_proj = Some(vp);
            let (_, _, trace, history) = sg.targets.as_ref().unwrap();
            let tex = wgpu::BindingResource::TextureView;
            let spot_lights = self.spot_lights.as_ref().unwrap_or(&gpu.dummy_spot_lights);
            let spot_atlas = self.spot_shadows.as_ref().unwrap_or(&gpu.dummy_spot_atlas);
            let sky_lighting = self.sky_lighting.as_ref().unwrap_or(&gpu.dummy_sky_lighting);
            let group = |label: &str, layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]| {
                let entries: Vec<_> = entries.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
            };
            let trace_bg = group(
                "VolumetricFog/ShaftsBG",
                &sg.trace_bgl,
                &[
                    (2, gpu.fog_params.as_entire_binding()),
                    (7, spot_lights.as_entire_binding()),
                    (8, tex(spot_atlas)),
                    (9, wgpu::BindingResource::Sampler(&gpu.spot_sampler)),
                    (10, gpu.media_params.as_entire_binding()),
                    (11, gpu.volumes.as_entire_binding()),
                    (12, sky_lighting.as_entire_binding()),
                    (13, tex(depth)),
                    (14, tex(gpu.grid.accum_view())),
                    (15, wgpu::BindingResource::Sampler(&gpu.accum_sampler)),
                    (16, tex(trace)),
                    (17, sg.params.as_entire_binding()),
                ],
            );
            let temporal_bg = group(
                "VolumetricFog/ShaftsTemporalBG",
                &sg.temporal_bgl,
                &[
                    (0, sg.params.as_entire_binding()),
                    (1, tex(trace)),
                    (2, tex(&history[1 - current])),
                    (3, tex(&history[current])),
                    (4, wgpu::BindingResource::Sampler(&gpu.accum_sampler)),
                ],
            );
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/Shafts"), timestamp_writes: crate::profiling::gpu_pass("VolumetricFog/Shafts").as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&sg.trace);
            pass.set_bind_group(0, &trace_bg, &[]);
            pass.dispatch_workgroups(sw.div_ceil(8), sh.div_ceil(8), 1);
            pass.set_pipeline(&sg.temporal);
            pass.set_bind_group(0, &temporal_bg, &[]);
            pass.dispatch_workgroups(sw.div_ceil(8), sh.div_ceil(8), 1);
            drop(pass);
            shafts_view = Some(history[current].clone());
        }

        // 5. composite over the scene
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VolumetricFog/CompositeBG"),
            layout: &gpu.composite_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(gpu.grid.accum_view()) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&gpu.accum_sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: gpu.composite_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(shafts_view.as_ref().unwrap_or(&gpu.shafts.none)) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/Composite"), timestamp_writes: crate::profiling::gpu_pass("VolumetricFog/Composite").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.composite_pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {
        // The grid is resolution-independent; composite params are written every frame.
    }

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Validate every froxel/fog WGSL module with naga, and check that the Rust uniform structs
    /// have exactly the size of their WGSL counterparts.
    #[test]
    fn shaders_validate_and_uniform_layouts_match() {
        let sources = FroxelGrid::shader_sources().into_iter().chain(VolumetricFogEffect::shader_sources());
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in sources {
            let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["FogParams"], std::mem::size_of::<FogParamsGpu>());
        assert_eq!(sizes["DirLightData"], std::mem::size_of::<DirLightGpu>());
        assert_eq!(sizes["PointLightData"], std::mem::size_of::<PointLightGpu>());
        assert_eq!(sizes["CompositeParams"], std::mem::size_of::<CompositeParamsGpu>());
        assert_eq!(sizes["GridParams"], std::mem::size_of::<crate::froxels::froxel_grid::GridParamsGpu>());
        assert_eq!(sizes["TemporalParams"], std::mem::size_of::<crate::froxels::froxel_grid::TemporalParamsGpu>());
        assert_eq!(sizes["FogMediaParams"], std::mem::size_of::<FogMediaParamsGpu>());
        assert_eq!(sizes["LocalFogVolume"], std::mem::size_of::<LocalFogVolumeGpu>());
        assert_eq!(sizes["SkyLighting"] as u64, SKY_LIGHTING_BYTES);
        assert_eq!(sizes["ShaftParams"], std::mem::size_of::<ShaftParamsGpu>());
    }

    #[test]
    fn local_fog_volumes_are_ellipsoids_with_a_low_lying_height_term() {
        // the Midsommar lake mist: 230 m across, 8 m up and down, height extinction 0.05
        let mut lake = LocalFogVolume::new(Vec3::new(320.0, 3.0, -50.0), 230.0, 8.0);
        lake.radial_extinction = 0.01;
        lake.height_extinction = 0.05;
        lake.height_falloff = 3.0;
        let at = |x: f32, y: f32, z: f32| lake.extinction_at(Vec3::new(x, y, z));
        // at the water: both terms; 4 m up the height term has fallen by e^-1.5
        assert!((at(320.0, 3.0, -50.0) - 0.06).abs() < 1e-6);
        let up = at(320.0, 7.0, -50.0);
        let expected = 0.01 * 0.75 + 0.05 * (-1.5f32).exp();
        assert!((up - expected).abs() < 1e-5, "{up} vs {expected}");
        // outside the ellipsoid, and fading out toward its rim
        assert_eq!(at(320.0, 11.5, -50.0), 0.0);
        assert_eq!(at(551.0, 3.0, -50.0), 0.0);
        assert!(at(540.0, 3.0, -50.0) < at(450.0, 3.0, -50.0));

        // yaw turns the long axis: a quarter turn about +Y puts local x along world z
        let mut bar = LocalFogVolume::new(Vec3::ZERO, 10.0, 2.0);
        bar.radii = Vec3::new(10.0, 2.0, 1.0);
        bar.yaw = std::f32::consts::FRAC_PI_2;
        assert!(bar.extinction_at(Vec3::new(0.0, 0.0, 8.0)) > 0.0);
        assert_eq!(bar.extinction_at(Vec3::new(8.0, 0.0, 0.0)), 0.0);

        // a box fills its corners, where the ellipsoid of the same extents is empty
        let corner = Vec3::new(8.0, 8.0, 8.0);
        let mut cube = LocalFogVolume::new_box(Vec3::ZERO, Vec3::new(10.0, 10.0, 10.0));
        cube.edge_fade = 0.1;
        assert!(cube.extinction_at(corner) > 0.0);
        assert_eq!(LocalFogVolume::new(Vec3::ZERO, 10.0, 10.0).extinction_at(corner), 0.0);
        assert_eq!(cube.extinction_at(Vec3::new(10.5, 0.0, 0.0)), 0.0);
    }

    #[test]
    fn update_lights_assigns_shadows_to_the_lights_the_renderer_shadows() {
        use crate::lights::{DirectionalLight, PointLight};
        let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 2.0);
        sun.cast_shadow = true;
        let mut moon = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0);
        moon.cast_shadow = true;
        let mut lamp = PointLight::new(Vec3::ZERO, Vec3::new(1.0, 0.5, 0.0), 1.0, 10.0);
        lamp.cast_shadow = true;
        let mut lamp2 = PointLight::new(Vec3::ZERO, Vec3::new(1.0, 0.5, 0.0), 1.0, 10.0);
        lamp2.cast_shadow = true;
        let mut dark = PointLight::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), 1.0, 10.0);
        dark.volumetric = false;
        let lights = [Light::Directional(sun), Light::Directional(moon), Light::Point(lamp), Light::Point(lamp2), Light::Point(dark)];

        let mut fog = VolumetricFogEffect::new(Default::default());
        fog.update_lights(lights.iter());
        assert_eq!(fog.dir_data.iter().map(|d| d.shadowed).collect::<Vec<_>>(), [1, 0]);
        assert_eq!(fog.dir_data[0].color, [2.0, 2.0, 2.0]);
        assert_eq!(fog.point_data.iter().map(|p| p.shadow_layer).collect::<Vec<_>>(), [0, NO_SHADOW, NO_SHADOW]);
        assert_eq!(fog.point_data[2].color, [0.0, 0.0, 0.0]);
    }

    fn f16_to_f32(h: u16) -> f32 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let exp = ((h >> 10) & 0x1f) as i32;
        let frac = (h & 0x3ff) as f32;
        sign * match exp {
            0 => frac * 2f32.powi(-24),
            31 => f32::INFINITY,
            e => (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
        }
    }

    fn read_volume(device: &wgpu::Device, queue: &wgpu::Queue, grid: &FroxelGrid) -> Vec<[f32; 4]> {
        let (w, h, d) = (grid.grid_w(), grid.grid_h(), grid.grid_d());
        let row = (w * 8).div_ceil(256) * 256;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: grid.accum_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: d },
        );
        queue.submit(std::iter::once(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = buffer.slice(..).get_mapped_range();
        let mut out = Vec::with_capacity((w * h * d) as usize);
        for z in 0..d {
            for y in 0..h {
                for x in 0..w {
                    let o = ((z * h + y) * row + x * 8) as usize;
                    let c = |i: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * i], data[o + 2 * i + 1]]));
                    out.push([c(0), c(1), c(2), c(3)]);
                }
            }
        }
        out
    }

    /// In a uniform fog every slice's transmittance is exp(-sigma (depth - near)), the thick far
    /// ones too; with `max_distance` the froxels hold fog only to that depth: the far slices'
    /// transmittance is the reach's, not the grid's, and the slices past it add nothing more.
    #[test]
    fn the_fog_ends_at_its_reach() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (64u32, 32u32);
        let tex = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
                .create_view(&Default::default())
        };
        let input = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
        let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
        let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let mut camera = Camera::new(60.0, 0.5, 1000.0, w as f32 / h as f32);
        camera.set_position(0.0, 5.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 5.0, -10.0));
        camera.update_view_matrix();
        let sigma = 0.01f32;
        let volume = |max_distance: f32| {
            let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
                grid: FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 32, near: 0.5, far: 200.0, temporal: false, blend_factor: 1.0 },
                base_density: sigma,
                height_falloff: 0.0,
                fog_height: 100.0,
                ambient: Vec3::new(1.0, 1.0, 1.0),
                max_distance,
                ..Default::default()
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            fog.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
            queue.submit(std::iter::once(encoder.finish()));
            (read_volume(&device, &queue, fog.froxel_grid().unwrap()), fog.reach())
        };
        // the centre column's transmittance at slice z
        let t = |v: &[[f32; 4]], z: u32| v[((z * 16 + 8) * 16 + 8) as usize][3];
        let (all, reach_all) = volume(0.0);
        // the whole grid holds the fog, its thick far slices too (the clip plane's fade once
        // thinned them when no plane was set)
        for z in [16u32, 24, 28] {
            let d1 = 0.5 * 400f32.powf((z + 1) as f32 / 32.0);
            assert!((t(&all, z) - (-sigma * (d1 - 0.5)).exp()).abs() < 2e-3, "slice {z}: {}", t(&all, z));
        }
        let (short, reach) = volume(60.0);
        assert_eq!((reach_all, reach), (200.0, 60.0));
        assert_eq!(volume(500.0).1, 200.0, "the reach is at most the grid's far");
        let (want_all, want) = ((-sigma * 199.5).exp(), (-sigma * 59.5).exp());
        eprintln!("far transmittance: grid {:.4} (want {want_all:.4}), reach 60 m {:.4} (want {want:.4})", t(&all, 31), t(&short, 31));
        assert!((t(&all, 31) - want_all).abs() < 0.02, "grid: {}", t(&all, 31));
        assert!((t(&short, 31) - want).abs() < 0.02, "reach: {}", t(&short, 31));
        // past the reach nothing more (60 m is slice 32 ln(120) / ln(400) = 25.6): slices 27 and 31
        // agree; before it the two volumes agree
        assert!((t(&short, 27) - t(&short, 31)).abs() < 1e-3, "past the reach: {} vs {}", t(&short, 27), t(&short, 31));
        assert!((t(&short, 20) - t(&all, 20)).abs() < 1e-3, "before the reach: {} vs {}", t(&short, 20), t(&all, 20));
    }

    /// The reflection's volume, built from the camera mirrored in the water, holds only the fog
    /// above the water: a mirrored ray below the plane crosses no fog, one rising through it does.
    #[test]
    fn the_reflection_fog_lies_only_above_the_mirror() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (64u32, 32u32);
        let tex = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
                .create_view(&Default::default())
        };
        let input = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
        let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
        let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let gbuffer = GBuffer::new(&device, w, h, 1);
        // uniform fog lit by a uniform sky, the camera 5 m above still water looking along it
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 32, near: 0.5, far: 200.0, temporal: false, blend_factor: 1.0 },
            base_density: 0.02,
            height_falloff: 0.0,
            fog_height: 100.0,
            ambient: Vec3::new(1.0, 1.0, 1.0),
            ..Default::default()
        });
        let _ = fog.mirrored_fog(&device, &queue, (glam::Vec3::Y, 0.0));
        let mut camera = Camera::new(60.0, 0.5, 1000.0, w as f32 / h as f32);
        camera.set_position(0.0, 5.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 5.0, -10.0));
        camera.update_view_matrix();
        let mut encoder = device.create_command_encoder(&Default::default());
        fog.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
        queue.submit(std::iter::once(encoder.finish()));
        let main = read_volume(&device, &queue, fog.froxel_grid().unwrap());
        let mirrored = read_volume(&device, &queue, fog.reflection_froxel_grid().unwrap());
        let at = |v: &[[f32; 4]], x: u32, y: u32, z: u32| v[((z * 16 + y) * 16 + x) as usize];
        let last = 31;
        // the main view: fog all along its middle row
        let m = at(&main, 8, 8, last);
        assert!(m[3] < 0.5 && m[1] > 0.1, "main view, far slice: {m:?}");
        // the mirrored camera sits 5 m below the water, and its image is the reflection's: what
        // lies above the water shows in its lower rows. Its upper rows never rise above the water
        let below = at(&mirrored, 8, 4, last);
        assert!(below[3] > 0.99 && below[1] < 1e-3, "mirrored view below the water: {below:?}");
        // its bottom row rises through the water into the fog beyond it
        let rising = at(&mirrored, 8, 15, last);
        assert!(rising[3] < 0.95 && rising[1] > 0.01, "mirrored view rising through the water: {rising:?}");
        // and nothing before it crosses the plane, 5 m / tan(28 degrees) = 9.2 m out, about slice
        // 15 of 32 between 0.5 and 200 m
        let near = at(&mirrored, 8, 15, 10);
        assert!(near[3] > 0.99 && near[1] < 1e-3, "mirrored view before the water: {near:?}");
        eprintln!("main {m:?}, mirrored below {below:?}, rising {rising:?}, before the water {near:?}");
    }

    /// With a sky occlusion bound, each froxel's sky light is dimmed by the sky it sees: a fog lit
    /// only by a uniform sky, half of it seen over the world's left (x < 0) and all of it over the
    /// right, scatters half as much light down the picture's left columns and as much down its
    /// right ones.
    #[test]
    fn the_sky_light_on_the_fog_follows_the_sky_occlusion() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (64u32, 32u32);
        let tex = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
                .create_view(&Default::default())
        };
        let input = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
        let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
        let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let gbuffer = GBuffer::new(&device, w, h, 1);
        // a uniform sky of radiance 1: its SH is band 0 only
        let sky = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: SKY_LIGHTING_BYTES, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let band0 = 0.282095 * 4.0 * std::f32::consts::PI;
        queue.write_buffer(&sky, 0, bytemuck::cast_slice(&[band0, band0, band0, 0.0f32]));
        // the sky occlusion's volume by hand: half the sky seen over x < 0, all of it beyond
        let shared = crate::renderers::SharedLayouts::new(&device);
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4096, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false });
        let options = crate::shadows::SkyOcclusionOptions { extent_m: 400.0, volume_size: (32, 8), min_height_m: -10.0, max_height_m: 30.0, ..Default::default() };
        let occlusion = crate::shadows::SkyOcclusion::new(&device, &shared.camera_bgl, &light_buf, options);
        let (side, layers) = options.volume_size;
        let texels: Vec<u8> = (0..side * layers * side).flat_map(|i| [if i % side < side / 2 { 128 } else { 255 }, 0, 0, 255]).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: occlusion.volume_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &texels,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(side * 4), rows_per_image: Some(layers) },
            wgpu::Extent3d { width: side, height: layers, depth_or_array_layers: side },
        );
        // SkyOcclusionParams: centre (0, 0), 1 / extent, min height, 1 / height span, on
        let params = [0.0f32, 0.0, 1.0 / options.extent_m, options.min_height_m, 1.0 / (options.max_height_m - options.min_height_m), 1.0, 0.0, 0.0];
        queue.write_buffer(&occlusion.params, 0, bytemuck::cast_slice(&params));
        let mut camera = Camera::new(60.0, 0.5, 1000.0, w as f32 / h as f32);
        camera.set_position(0.0, 5.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 5.0, -10.0));
        camera.update_view_matrix();
        let scatter = |occluded: bool| {
            let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
                grid: FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 32, near: 0.5, far: 100.0, temporal: false, blend_factor: 1.0 },
                base_density: 0.02,
                height_falloff: 0.0,
                fog_height: 100.0,
                ambient: Vec3::new(0.0, 0.0, 0.0),
                ..Default::default()
            });
            fog.set_sky_lighting(Some(&sky));
            fog.set_sky_occlusion(occluded.then_some(&occlusion));
            let mut encoder = device.create_command_encoder(&Default::default());
            fog.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
            queue.submit(std::iter::once(encoder.finish()));
            read_volume(&device, &queue, fog.froxel_grid().unwrap())
        };
        let (open, occluded) = (scatter(false), scatter(true));
        let at = |v: &[[f32; 4]], x: u32| v[((31 * 16 + 8) * 16 + x) as usize][1];
        let (left, right) = (at(&occluded, 2) / at(&open, 2), at(&occluded, 13) / at(&open, 13));
        eprintln!("the fog's sky light with the occlusion: left {left:.3}, right {right:.3} of without");
        assert!(at(&open, 2) > 0.01, "no sky light on the fog: {:?}", at(&open, 2));
        // each column's first metres lie within a voxel (12.5 m) of x = 0, where the volume blends
        // the two halves
        assert!((left - 128.0 / 255.0).abs() < 0.04, "left: {left}");
        assert!((right - 1.0).abs() < 0.03, "right: {right}");
    }

    /// While the reflection is not drawn (disabled, or the camera under its plane) the fog builds
    /// no volume for it.
    #[test]
    fn a_reflection_not_drawn_gets_no_fog_volume() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (64u32, 32u32);
        let tex = |format, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] })
                .create_view(&Default::default())
        };
        let input = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING);
        let output = tex(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING);
        let depth = tex(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 32, near: 0.5, far: 200.0, temporal: false, blend_factor: 1.0 },
            base_density: 0.02,
            height_falloff: 0.0,
            fog_height: 100.0,
            ambient: Vec3::new(1.0, 1.0, 1.0),
            ..Default::default()
        });
        let seen = fog.mirrored_fog(&device, &queue, (glam::Vec3::Y, 0.0));
        seen.drawn.store(false, std::sync::atomic::Ordering::Relaxed);
        let mut camera = Camera::new(60.0, 0.5, 1000.0, w as f32 / h as f32);
        camera.set_position(0.0, 5.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 5.0, -10.0));
        camera.update_view_matrix();
        let mut encoder = device.create_command_encoder(&Default::default());
        fog.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
        queue.submit(std::iter::once(encoder.finish()));
        // never written: the zero-initialised texture
        let mirrored = read_volume(&device, &queue, fog.reflection_froxel_grid().unwrap());
        assert!(mirrored.iter().all(|v| *v == [0.0; 4]), "a volume was built for a reflection that was not drawn");
    }

    /// A fog scene for the spot-light tests: spot lights packed as the renderer packs them, a
    /// shadow atlas where the first light's layer holds a thin vertical card (the plane
    /// z = card.0, card.1 < x < card.2), nothing but fog before a black background, and the camera
    /// at (0, 1, 0) looking down -z.
    struct SpotFogScene {
        light_buf: wgpu::Buffer,
        atlas: wgpu::TextureView,
        input: wgpu::TextureView,
        output_tex: wgpu::Texture,
        output: wgpu::TextureView,
        depth: wgpu::TextureView,
        gbuffer: GBuffer,
        camera: Camera,
    }

    fn spot_fog_scene(device: &wgpu::Device, queue: &wgpu::Queue, (w, h): (u32, u32), lights: &[Light], card: (f32, f32, f32), res: u32) -> SpotFogScene {
        use glam::Vec3 as V;
        let mut packed = crate::lights::spot_lights_gpu::SpotLightsGpu::new();
        packed.pack(lights.iter(), lights.len() as u32, res);
        let light_buf = { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: packed.as_bytes(), usage: wgpu::BufferUsages::STORAGE }) };
        let layers = lights.len() as u32;
        let atlas_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: res, height: res, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SpotShadowAtlas::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let atlas = atlas_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        for (layer, light) in packed.lights.iter().enumerate() {
            // the card's depth wherever it covers a texel of the first light's map
            let light_vp = glam::Mat4::from_cols_array(&light.view_proj);
            let inv = light_vp.inverse();
            let lamp = V::from(light.position);
            let mut depths = vec![1.0f32; (res * res) as usize];
            if layer == 0 {
                for y in 0..res {
                    for x in 0..res {
                        let ndc = glam::Vec2::new((x as f32 + 0.5) / res as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / res as f32 * 2.0);
                        let dir = (inv.project_point3(V::new(ndc.x, ndc.y, 1.0)) - lamp).normalize();
                        let t = (card.0 - lamp.z) / dir.z;
                        let hit = lamp + dir * t;
                        if t > 0.0 && hit.x > card.1 && hit.x < card.2 {
                            depths[(y * res + x) as usize] = light_vp.project_point3(hit).z;
                        }
                    }
                }
            }
            let layer_view = atlas_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_array_layer: layer as u32, array_layer_count: Some(1), ..Default::default() });
            let depth_buf = { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&depths), usage: wgpu::BufferUsages::STORAGE }) };
            let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
                @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
                @fragment fn fs(@builtin(position) pos : vec4f) -> @builtin(frag_depth) f32 {{ return d[u32(pos.y) * {res}u + u32(pos.x)]; }}");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: None,
                layout: None,
                vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
                fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState { format: SpotShadowAtlas::FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            });
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() }] });
            let mut e = device.create_command_encoder(&Default::default());
            {
                let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: None,
                    color_attachments: &[],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &layer_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                    timestamp_writes: None,
                    occlusion_query_set: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.draw(0..3, 0..1);
            }
            queue.submit([e.finish()]);
        }
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let output = output_tex.create_view(&Default::default());
        let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        {
            let mut e = device.create_command_encoder(&Default::default());
            e.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[], depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }), timestamp_writes: None, occlusion_query_set: None });
            queue.submit([e.finish()]);
        }
        let gbuffer = GBuffer::new(device, w, h, 1);
        let mut camera = Camera::new(60.0, 0.1, 100.0, w as f32 / h as f32);
        camera.set_position(0.0, 1.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 1.0, -1.0));
        camera.update_view_matrix();
        SpotFogScene { light_buf, atlas, input, output_tex, output, depth, gbuffer, camera }
    }

    /// A headlamp shines through fog toward the camera past a thin card just in front of it, whose
    /// shadow wedge reaches the camera (the film's beams through grass and trunks). Along the
    /// lamp's row, the raymarched shafts follow a CPU raymarch of the same fog, light and card
    /// (the card's shadow takes about 30 % of the light there); the froxels, at the film's 12
    /// pixels per froxel, stray much further. The columns next to the lamp are left out: its glow
    /// changes too fast there for the half-resolution march.
    #[test]
    fn raymarched_spot_shafts_keep_thin_shadows() {
        use glam::Vec3 as V;
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (256u32, 128u32);
        const DENSITY: f32 = 0.05;
        const G: f32 = 0.3;
        let lamp = V::new(0.0, 1.0, -20.0);
        let mut spot = crate::lights::SpotLight::new(Vec3::new(lamp.x, lamp.y, lamp.z), Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 1.0, 1.0), 5000.0, 40.0, 10f32.to_radians(), 20f32.to_radians());
        spot.cast_shadow = true;
        spot.volumetric_scale = 1.0;
        spot.source_radius = 0.05;
        // the card: the plane z = -18.5, 0.2 < x < 0.35 (1.5 m in front of the lamp, beside its axis)
        let (card_z, card_x0, card_x1) = (-18.5f32, 0.2f32, 0.35f32);
        let blocked = |p: V| -> bool {
            let d = lamp - p;
            if d.z.abs() < 1e-6 { return false; }
            let t = (card_z - p.z) / d.z;
            if !(0.0..=1.0).contains(&t) { return false; }
            let x = p.x + d.x * t;
            x > card_x0 && x < card_x1
        };
        let scene = spot_fog_scene(&device, &queue, (w, h), &[Light::Spot(spot)], (card_z, card_x0, card_x1), 512);
        let SpotFogScene { light_buf, atlas: atlas_array, input, output_tex, output, depth, gbuffer, camera } = &scene;
        let run = |scattering: SpotScattering| -> Vec<[f32; 4]> {
            let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
                grid: FroxelGridOptions { grid_w: 22, grid_h: 11, grid_d: 48, near: 0.5, far: 40.0, temporal: true, blend_factor: 0.1 },
                base_density: DENSITY,
                height_falloff: 0.0,
                anisotropy: G,
                spot_scattering: scattering,
                ..Default::default()
            });
            fog.spot_lights = Some(light_buf.clone());
            fog.spot_shadows = Some(atlas_array.clone());
            for _ in 0..48 {
                let mut e = device.create_command_encoder(&Default::default());
                fog.render(&device, &queue, &mut e, gbuffer, input, depth, output, camera, w, h);
                queue.submit([e.finish()]);
            }
            let row = (w * 8).div_ceil(256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            (0..w * h)
                .map(|i| {
                    let o = ((i / w) * row + (i % w) * 8) as usize;
                    let c = |k: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * k], data[o + 2 * k + 1]]));
                    [c(0), c(1), c(2), c(3)]
                })
                .collect()
        };
        let froxels = run(SpotScattering::Froxels);
        let raymarched = run(SpotScattering::Raymarched { steps: 32 });

        // the reference: the same fog (uniform, albedo 1), light and card, marched finely on the CPU
        let inv_vp = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
        let eye = V::new(0.0, 1.0, 0.0);
        let (cos_outer, cos_inner) = (20f32.to_radians().cos(), 10f32.to_radians().cos());
        let truth = |x: u32, y: u32, shadowed: bool| -> f32 {
            let ndc = glam::Vec2::new((x as f32 + 0.5) / w as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / h as f32 * 2.0);
            let rd = (inv_vp.project_point3(V::new(ndc.x, ndc.y, 1.0)) - eye).normalize();
            let (n, t_max) = (8000, 40.0f32);
            let dt = t_max / n as f32;
            let mut sum = 0.0;
            for k in 0..n {
                let t = (k as f32 + 0.5) * dt;
                let p = eye + rd * t;
                let d = lamp - p;
                let dist2 = d.length_squared().max(0.05 * 0.05);
                let l = d / d.length();
                let r = d.length_squared() / (40.0 * 40.0);
                let window = (1.0 - r * r).clamp(0.0, 1.0);
                let cone = ((-l.z - cos_outer) / (cos_inner - cos_outer)).clamp(0.0, 1.0);
                if window * cone <= 0.0 || (shadowed && blocked(p)) { continue; }
                let hg = (1.0 - G * G) / (4.0 * std::f32::consts::PI * (1.0 + G * G - 2.0 * G * rd.dot(l)).powf(1.5));
                sum += DENSITY * 5000.0 * window * window * cone * cone / dist2 * hg * (-DENSITY * t).exp() * dt;
            }
            sum
        };
        // along the lamp's row, right of it: where the card's shadow darkens the truth
        let y = h / 2;
        let (mut worst_ray, mut worst_frox, mut streak) = (0.0f32, 0.0f32, 0);
        for x in (w / 2 + 6..w).step_by(2) {
            let lit = truth(x, y, false);
            let want = truth(x, y, true);
            if lit < 0.02 * truth(w / 2 + 2, y, false) { continue; }
            let i = (y * w + x) as usize;
            let (ray, frox) = (raymarched[i][0], froxels[i][0]);
            let in_streak = want < 0.8 * lit;
            if in_streak { streak += 1; }
            worst_ray = worst_ray.max((ray / want - 1.0).abs());
            worst_frox = worst_frox.max((frox / want - 1.0).abs());
        }
        eprintln!("worst relative error: raymarched {worst_ray:.3}, froxels {worst_frox:.3}; {streak} shadowed columns");
        assert!(streak >= 10, "the card's shadow did not cross the row");
        assert!(worst_ray < 0.1, "raymarched shafts off the reference by {worst_ray}");
        assert!(worst_frox > 2.0 * worst_ray, "the froxels were as close ({worst_frox} vs {worst_ray})");
    }

    /// The GPU time of the fog at the film's resolution and grid with its two headlamps shining at
    /// the camera (every pixel's ray crosses both cones: the worst case), froxels against
    /// raymarched shafts: each frame submitted and waited for, less an empty submit's time.
    /// Other GPU work on the machine inflates it, so the minimum is the figure to read. Run by
    /// hand: `cargo test -p kansei-core --lib time_spot_shafts -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_spot_shafts() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (1920u32, 803u32);
        let lamps: Vec<Light> = [-0.62f32, 0.62]
            .iter()
            .map(|&x| {
                let mut s = crate::lights::SpotLight::new(Vec3::new(x, 0.7, -25.0), Vec3::new(0.0, -0.02, 1.0), Vec3::new(1.0, 0.95, 0.85), 22000.0, 70.0, 10f32.to_radians(), 30f32.to_radians());
                s.cast_shadow = true;
                s.volumetric_scale = 1.0;
                s.source_radius = 0.08;
                Light::Spot(s)
            })
            .collect();
        let scene = spot_fog_scene(&device, &queue, (w, h), &lamps, (-23.5, -0.3, -0.1), 1024);
        let wall = |encode: &mut dyn FnMut(&mut wgpu::CommandEncoder)| -> f64 {
            let mut e = device.create_command_encoder(&Default::default());
            encode(&mut e);
            let t = std::time::Instant::now();
            queue.submit([e.finish()]);
            device.poll(wgpu::Maintain::Wait);
            t.elapsed().as_secs_f64() * 1e3
        };
        let mut empty: Vec<f64> = (0..100).map(|_| wall(&mut |_| {})).collect();
        empty.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for scattering in [SpotScattering::Froxels, SpotScattering::Raymarched { steps: 8 }, SpotScattering::Raymarched { steps: 16 }, SpotScattering::Raymarched { steps: 24 }] {
            let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
                grid: FroxelGridOptions { grid_w: 160, grid_h: 67, grid_d: 48, near: 0.5, far: 260.0, temporal: true, blend_factor: 0.1 },
                base_density: 0.02,
                height_falloff: 0.05,
                spot_scattering: scattering,
                ..Default::default()
            });
            fog.spot_lights = Some(scene.light_buf.clone());
            fog.spot_shadows = Some(scene.atlas.clone());
            let mut times: Vec<f64> = (0..120)
                .map(|_| wall(&mut |e| fog.render(&device, &queue, e, &scene.gbuffer, &scene.input, &scene.depth, &scene.output, &scene.camera, w, h)))
                .skip(20)
                .collect();
            times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            eprintln!("{scattering:?}: fog {:.2} ms min, {:.2} ms median ({w}x{h}, an empty submit {:.2} ms taken off)", times[0] - empty[0], times[times.len() / 2] - empty[0], empty[0]);
        }
    }
}

