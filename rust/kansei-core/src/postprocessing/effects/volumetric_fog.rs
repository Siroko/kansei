use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::froxels::{FroxelGrid, FroxelGridOptions};
use crate::lights::Light;
use crate::math::{Mat4, Vec3};
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;
use crate::shadows::{CubeMapShadowMap, ShadowMap};

const INJECT_WGSL: &str = concat!(
    include_str!("../../shaders/froxel_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_inject.wgsl"),
    include_str!("../../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../../shaders/volumetric_fog_media.wgsl"),
);
const COMPOSITE_WGSL: &str = concat!(
    include_str!("../../shaders/froxel_common.wgsl"),
    include_str!("../../shaders/volumetric_fog_composite.wgsl"),
);

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
            wind_direction: Vec3::ZERO,
            ambient: Vec3::ZERO,
            albedo: Vec3::new(1.0, 1.0, 1.0),
            sky_ambient_scale: 1.0,
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
    _pad: [f32; 2],
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
    _pad: f32,
}

/// A local fog volume: an ellipsoid of mist (over a lake, in a hollow) injected into the fog's
/// froxels, after Unreal's `LocalFogVolume`. In the volume's unit sphere `q` (|q| < 1) the
/// extinction is `radial_extinction * (1 - |q|^2) + height_extinction * exp(-height_falloff *
/// max(q.y - height_offset, 0))`, faded to zero over the outer `edge_fade` of the radius. It
/// scatters the fog's lights and sky with its own albedo; wind and start distance leave it alone.
///
/// Unreal's `radial_fog_extinction`, `height_fog_extinction`, `height_fog_falloff`,
/// `height_fog_offset` and `fog_albedo` carry over; the shapes of the two terms are kansei's
/// own, so the look may need a trim.
#[derive(Debug, Clone, Copy)]
pub struct LocalFogVolume {
    pub center: Vec3,
    /// Semi-axes of the ellipsoid, metres.
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

    /// Extinction per metre at a world-space point, as the fog shader computes it.
    pub fn extinction_at(&self, p: Vec3) -> f32 {
        let g = self.gpu();
        let d = glam::Vec3::new(p.x - g.center[0], p.y - g.center[1], p.z - g.center[2]);
        let q = glam::Vec3::new(g.cos_yaw * d.x - g.sin_yaw * d.z, d.y, g.sin_yaw * d.x + g.cos_yaw * d.z) * glam::Vec3::from(g.inv_radii);
        let r2 = q.length_squared();
        if r2 >= 1.0 {
            return 0.0;
        }
        let edge = if g.edge_fade > 0.0 {
            let t = ((1.0 - r2.sqrt()) / g.edge_fade).clamp(0.0, 1.0);
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
/// fog.update_lights(scene.lights());                  // each frame, or when lights change
/// ```
pub struct VolumetricFogEffect {
    pub base_density: f32,
    pub height_falloff: f32,
    pub fog_height: f32,
    pub extinction_coeff: f32,
    pub anisotropy: f32,
    pub start_distance: f32,
    pub wind_direction: Vec3,
    pub ambient: Vec3,
    pub albedo: Vec3,
    pub sky_ambient_scale: f32,
    /// Local fog volumes, uploaded every frame (keep it to tens).
    pub local_volumes: Vec<LocalFogVolume>,
    /// Seconds, drives the wind offset. The effect has no clock of its own; set it per frame.
    pub time: f32,
    grid_options: FroxelGridOptions,
    dir_data: Vec<DirLightGpu>,
    point_data: Vec<PointLightGpu>,
    shadow_map: Option<(wgpu::TextureView, wgpu::Buffer)>,
    point_shadows: Option<wgpu::TextureView>,
    sky_lighting: Option<wgpu::Buffer>,
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
            wind_direction: options.wind_direction,
            ambient: options.ambient,
            albedo: options.albedo,
            sky_ambient_scale: options.sky_ambient_scale,
            local_volumes: Vec::new(),
            time: 0.0,
            grid_options: options.grid,
            dir_data: Vec::new(),
            point_data: Vec::new(),
            shadow_map: None,
            point_shadows: None,
            sky_lighting: None,
            lights_dirty: true,
            bindings_dirty: true,
            gpu: None,
        }
    }

    /// The froxel grid (available after the effect's first frame).
    pub fn froxel_grid(&self) -> Option<&FroxelGrid> {
        self.gpu.as_ref().map(|g| &g.grid)
    }

    /// Drop the temporal history; call on camera cuts. No-op without a temporal grid.
    pub fn reset_history(&mut self) {
        if let Some(g) = &mut self.gpu {
            g.grid.reset_history();
        }
    }

    /// Collect the volumetric lights. Directional lights cast shafts through the renderer's
    /// shadow map if they are the scene's first directional light and `cast_shadow` is set
    /// (that is the light the renderer's `ShadowMap` follows); the first shadow-casting point
    /// light uses the cube shadow atlas. Area lights are treated as point lights at their
    /// position, as in the TS effect.
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
            ],
        });

        let pipeline = |label: &str, code: &str, bgl: &wgpu::BindGroupLayout| {
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
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
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

        let volumes = buffer("VolumetricFog/LocalVolumes", std::mem::size_of::<LocalFogVolumeGpu>(), wgpu::BufferUsages::STORAGE);
        let dummy_sky_lighting = buffer("VolumetricFog/NoSkyLighting", SKY_LIGHTING_BYTES as usize, wgpu::BufferUsages::UNIFORM);
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
        gpu.inject_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("VolumetricFog/InjectBG"),
            layout: &gpu.inject_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(gpu.grid.scatter_extinction_view()) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.fog_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.dir_lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: gpu.point_lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(atlas) },
                wgpu::BindGroupEntry { binding: 6, resource: vp.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: gpu.media_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 11, resource: gpu.volumes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: sky_lighting.as_entire_binding() },
            ],
        }));
        self.bindings_dirty = false;
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 2] {
        [("volumetric_fog_inject", INJECT_WGSL), ("volumetric_fog_composite", COMPOSITE_WGSL)]
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
            _pad: [0.0; 2],
        };
        queue.write_buffer(&gpu.fog_params, 0, bytemuck::bytes_of(&params));
        let composite = CompositeParamsGpu {
            camera_near: camera.near,
            camera_far: camera.far,
            grid_near: grid.near(),
            grid_far: grid.far(),
            grid_d: grid.grid_d() as f32,
            screen_width: width as f32,
            screen_height: height as f32,
            _pad: 0.0,
        };
        queue.write_buffer(&gpu.composite_params, 0, bytemuck::bytes_of(&composite));

        // 1. inject density + lighting into the froxels
        {
            let (w, h, d) = (grid.grid_w(), grid.grid_h(), grid.grid_d());
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/Inject"), ..Default::default() });
            pass.set_pipeline(&gpu.inject_pipeline);
            pass.set_bind_group(0, gpu.inject_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }

        // 2. temporal reprojection (no-op unless temporal), 3. front-to-back accumulation
        gpu.grid.temporal_blend(queue, encoder, &Mat4::from(inv_vp), &Mat4::from(vp), camera.near, camera.far);
        gpu.grid.accumulate(encoder);

        // 4. composite over the scene
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
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VolumetricFog/Composite"), ..Default::default() });
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
}
