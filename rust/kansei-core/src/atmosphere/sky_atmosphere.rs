use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::math::Vec3;
use crate::postprocessing::effects::{HeightFogEffect, HeightFogLayer};

use super::params::{AtmosphereGpu, AtmosphereParams, CelestialLight, SkyFrameGpu, SkyLightingGpu};

pub(crate) const COMMON_WGSL: &str = include_str!("shaders/common.wgsl");
pub(crate) const FRAME_WGSL: &str = include_str!("shaders/frame.wgsl");
pub(crate) const LOOKUP_TRANSMITTANCE_WGSL: &str = include_str!("shaders/lookup_transmittance.wgsl");
pub(crate) const LOOKUP_MULTI_SCATTERING_WGSL: &str = include_str!("shaders/lookup_multi_scattering.wgsl");
pub(crate) const SCATTERING_WGSL: &str = include_str!("shaders/scattering.wgsl");
pub(crate) const SKY_LOOKUP_WGSL: &str = include_str!("shaders/sky_lookup.wgsl");
pub(crate) const AERIAL_PERSPECTIVE_LOOKUP_WGSL: &str = include_str!("shaders/aerial_perspective_lookup.wgsl");

const TRANSMITTANCE_WGSL: &str = include_str!("shaders/transmittance_lut.wgsl");
const MULTI_SCATTERING_WGSL: &str = include_str!("shaders/multi_scattering_lut.wgsl");
const SKY_VIEW_WGSL: &str = include_str!("shaders/sky_view_lut.wgsl");
const AERIAL_PERSPECTIVE_WGSL: &str = include_str!("shaders/aerial_perspective_lut.wgsl");
pub(crate) const SKY_LIGHTING_WGSL: &str = include_str!("shaders/sky_lighting.wgsl");
pub(crate) const CLOUD_MAP_WGSL: &str = include_str!("shaders/cloud_map.wgsl");
const SKY_CAPTURE_WGSL: &str = include_str!("shaders/sky_capture.wgsl");
/// The cloud map's size (cloud_map.wgsl): azimuth by zenith angle.
pub(crate) const CLOUD_MAP_SIZE: (u32, u32) = (128, 64);
pub(crate) use super::CLOUD_SHADOW_WGSL;
/// The cloud shadow map's texels per side.
pub(crate) const CLOUD_SHADOW_SIZE: u32 = 256;
const SKY_LIGHTING_PASS_WGSL: &str = include_str!("shaders/sky_lighting_pass.wgsl");
const ENVIRONMENT_PASS_WGSL: &str = include_str!("shaders/sky_environment_pass.wgsl");

/// GGX samples per texel for the environment's rough mips.
const ENVIRONMENT_SAMPLES: u32 = 96;

pub(crate) const LUT_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// Limb darkening of the sun's disk (the moon's is flat); the composite shader uses the same.
pub(crate) const SUN_LIMB_DARKENING: f32 = 0.6;

/// Lowest camera altitude the LUTs are built for, km: below it the horizon maths loses precision.
const MIN_CAMERA_ALTITUDE_KM: f32 = 0.005;

fn shader(parts: &[&str]) -> String {
    parts.concat()
}

pub(crate) fn transmittance_source() -> String {
    shader(&[COMMON_WGSL, TRANSMITTANCE_WGSL])
}

pub(crate) fn multi_scattering_source() -> String {
    shader(&[COMMON_WGSL, LOOKUP_TRANSMITTANCE_WGSL, MULTI_SCATTERING_WGSL])
}

pub(crate) fn sky_view_source() -> String {
    shader(&[COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, LOOKUP_MULTI_SCATTERING_WGSL, SCATTERING_WGSL, SKY_VIEW_WGSL])
}

pub(crate) fn sky_lighting_source() -> String {
    shader(&[COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, SKY_LOOKUP_WGSL, SKY_LIGHTING_WGSL, CLOUD_MAP_WGSL, SKY_CAPTURE_WGSL, SKY_LIGHTING_PASS_WGSL])
}

pub(crate) fn environment_source() -> String {
    shader(&[COMMON_WGSL, FRAME_WGSL, SKY_LOOKUP_WGSL, SKY_LIGHTING_WGSL, CLOUD_MAP_WGSL, SKY_CAPTURE_WGSL, ENVIRONMENT_PASS_WGSL])
}

pub(crate) fn aerial_perspective_source() -> String {
    shader(&[COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, LOOKUP_MULTI_SCATTERING_WGSL, SCATTERING_WGSL, AERIAL_PERSPECTIVE_WGSL])
}

/// LUT resolutions. The defaults are Hillaire 2020's, except the sky-view LUT, which spans the
/// full world azimuth (so the sun and the moon can share it) and has more columns for that.
#[derive(Debug, Clone, Copy)]
pub struct SkyAtmosphereOptions {
    pub transmittance_size: (u32, u32),
    pub multi_scattering_size: u32,
    pub sky_view_size: (u32, u32),
    /// Aerial-perspective volume: screen-aligned columns and depth slices.
    pub aerial_perspective_size: (u32, u32, u32),
    /// Distance covered by the aerial-perspective volume, km (farther surfaces use its last slice).
    pub aerial_perspective_distance_km: f32,
    /// Face size of the sky environment cubemap; its mips are GGX-prefiltered (roughness 0 to 1).
    pub environment_size: u32,
}

impl Default for SkyAtmosphereOptions {
    fn default() -> Self {
        Self {
            transmittance_size: (256, 64),
            multi_scattering_size: 32,
            sky_view_size: (256, 128),
            aerial_perspective_size: (32, 32, 32),
            aerial_perspective_distance_km: 32.0,
            environment_size: 64,
        }
    }
}

/// The GPU handles an effect or a material needs to read the atmosphere. Cheap to clone.
/// The scene's exponential height fog as the sky lighting and the environment cubemap capture it
/// (`SkyAtmosphere::capture_fog`): composited at infinite distance over the sky and the clouds, as
/// seen from `capture_height_m`, as Unreal's real-time sky-light capture does with its
/// ExponentialHeightFog. Seen from low down it covers the horizon and, opaque below it, replaces
/// the ground, so the ambient light and the reflections take the fog's colour where the fog is.
#[derive(Debug, Clone, Copy)]
pub struct SkyCaptureFog {
    /// The fog's layers (as `HeightFogEffect::layers`); the second is off while its density is 0.
    pub layers: [HeightFogLayer; 2],
    /// Its colour at full opacity, cd/m^2 (as `HeightFogEffect::inscattering`).
    pub inscattering: Vec3,
    /// At most this opaque (as `HeightFogEffect::max_opacity`).
    pub max_opacity: f32,
    /// World height of the point the sky is captured from, metres (Unreal: its SkyLight actor).
    pub capture_height_m: f32,
}

impl SkyCaptureFog {
    /// The fog a `HeightFogEffect` draws (its layers, colour and opacity), captured from
    /// `capture_height_m`.
    pub fn from_height_fog(fog: &HeightFogEffect, capture_height_m: f32) -> Self {
        Self { layers: fog.layers, inscattering: fog.inscattering, max_opacity: fog.max_opacity, capture_height_m }
    }
}

/// What the sky lighting and the environment cubemap see below the horizon.
#[derive(Debug, Clone, Copy, Default)]
pub enum SkyLowerHemisphere {
    /// A Lambertian ground of `sky_light_ground_albedo` lit by the sky and the lights, behind the
    /// air; under a capture fog, the fog, which is opaque below the horizon (Unreal's capture with
    /// `bLowerHemisphereIsBlack` off).
    #[default]
    Ground,
    /// This radiance, whatever the fog (Unreal's `bLowerHemisphereIsBlack` with its
    /// `LowerHemisphereColor`; black is `Color(Vec3::ZERO)`).
    Color(Vec3),
}

/// The WGSL `SkyCapture` struct (sky_capture.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkyCaptureGpu {
    layer0: [f32; 4],
    layer1: [f32; 4],
    inscattering: [f32; 3],
    fog_on: u32,
    lower_color: [f32; 3],
    lower_mode: u32,
    capture_height: f32,
    max_opacity: f32,
    _pad: [f32; 2],
}

#[derive(Clone)]
pub struct SkyAtmosphereBindings {
    /// `Atmosphere` uniform (the WGSL struct in `ATMOSPHERE_WGSL`).
    pub atmosphere: wgpu::Buffer,
    /// `SkyFrame` uniform, rewritten by every `SkyAtmosphere::update`.
    pub frame: wgpu::Buffer,
    pub transmittance: wgpu::TextureView,
    pub multi_scattering: wgpu::TextureView,
    pub sky_view: wgpu::TextureView,
    /// Aerial-perspective volume (3D): in-scattered light toward the camera, and transmittance.
    pub ap_scattering: wgpu::TextureView,
    pub ap_transmittance: wgpu::TextureView,
    /// `SkyLighting` (see `SKY_LIGHTING_WGSL`): the sky's radiance as order-2 SH and the sun and
    /// the moon at the camera, rewritten by every update. Usable as a uniform or a storage buffer.
    pub sky_lighting: wgpu::Buffer,
    /// The sky around the camera as a cube (no sun disk), mip m prefiltered for GGX roughness
    /// m / (mips - 1): sample it with `environment_sampler` and `SKY_ENVIRONMENT_WGSL`.
    pub environment: wgpu::TextureView,
    /// Trilinear, for the environment cubemap's mips.
    pub environment_sampler: wgpu::Sampler,
    /// Linear, clamping: for the transmittance and multiple-scattering LUTs.
    pub lut_sampler: wgpu::Sampler,
    /// Linear, repeating in u (the azimuth): for the sky-view LUT.
    pub sky_view_sampler: wgpu::Sampler,
    /// The clouds around the camera (rgba16float, azimuth by zenith angle; rgb their light, a
    /// their opacity), written by `VolumetricCloudsEffect` and read by the sky lighting and the
    /// environment cubemap. Empty (a clear sky) without clouds.
    pub cloud_map: wgpu::TextureView,
    /// The camera frame (plus one; 0 never) the clouds last wrote `cloud_map` in. The sky lighting
    /// reads the map only while it is that fresh, so clouds taken out of the chain leave no trace.
    pub(crate) cloud_map_frame: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// The clouds' shadow on the scene (rgba8unorm, r the cloud layer's transmittance toward the
    /// sun), written by `VolumetricCloudsEffect`: bind it with `lut_sampler` and
    /// `cloud_shadow_params` and multiply the sun's light by `CLOUD_SHADOW_WGSL`'s
    /// `cloudShadow`. Off (1 everywhere) without clouds.
    pub cloud_shadow: wgpu::TextureView,
    /// The shadow map's placement (WGSL `CloudShadowParams`), from the same frame as the map.
    pub cloud_shadow_params: wgpu::Buffer,
    /// As `cloud_map_frame`, for the shadow map.
    pub(crate) cloud_shadow_frame: std::sync::Arc<std::sync::atomic::AtomicU32>,
    /// The sun's direction at the last update, for the clouds' shadow map.
    pub(crate) sun_direction: std::sync::Arc<std::sync::Mutex<glam::Vec3>>,
}

struct Pipelines {
    transmittance: wgpu::ComputePipeline,
    transmittance_bg: wgpu::BindGroup,
    multi_scattering: wgpu::ComputePipeline,
    multi_scattering_bg: wgpu::BindGroup,
    sky_view: wgpu::ComputePipeline,
    sky_view_bg: wgpu::BindGroup,
    aerial_perspective: wgpu::ComputePipeline,
    aerial_perspective_bg: wgpu::BindGroup,
    sky_lighting: wgpu::ComputePipeline,
    // with the cloud map, and without it (no clouds wrote it last frame)
    sky_lighting_bgs: [wgpu::BindGroup; 2],
    environment: wgpu::ComputePipeline,
    /// One per mip, each with its own storage view and parameters.
    environment_bgs: Vec<[wgpu::BindGroup; 2]>,
}

/// A physically based sky and atmosphere after Hillaire 2020, as the LUTs the sky, the aerial
/// perspective and the sky lighting are rendered from:
/// - **transmittance** (256x64): transmittance to space by altitude and zenith angle;
/// - **multiple scattering** (32x32): all scattering orders >= 2, per unit illuminance;
/// - **sky view** (256x128): the sky's luminance around the camera, every frame;
/// - **aerial perspective** (32x32x32 camera froxels, to 32 km): the light scattered toward the
///   camera and the transmittance in front of every surface, every frame;
/// - **sky lighting**: the sky's radiance as order-2 spherical harmonics (with light bounced off
///   the ground below the horizon) and the sun and the moon at the camera, every frame, for
///   materials (`SKY_LIGHTING_WGSL`) so a scene's ambient light comes from its sky;
/// - **environment** (64x64 cube): the sky as a GGX-prefiltered cubemap, every frame, for the
///   specular reflection of the sky (`SKY_ENVIRONMENT_WGSL`).
///
/// The first two depend only on [`AtmosphereParams`] and are rebuilt when they change. The sun can
/// be anywhere, including below the horizon at dusk, where the sky is lit only by the light
/// scattered high above the planet's shadow. A moon (off by default) scatters as a second light.
///
/// World space is metres, Y up, with the planet's surface at `y = -origin_altitude_m` and its
/// centre straight below the world origin.
///
/// ```ignore
/// let mut sky = SkyAtmosphere::new(renderer.device(), SkyAtmosphereOptions::default());
/// sky.sun.direction = direction_from_elevation_bearing(-2.5, 140.0);
/// sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0); // lux
/// let volume = PostProcessingVolume::new(&renderer, vec![Box::new(AtmosphereEffect::new(&sky)), ...]);
/// // each frame, once the camera is placed:
/// sky.update(renderer.device(), renderer.queue(), &mut camera);
/// renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
/// ```
pub struct SkyAtmosphere {
    /// The scene's height fog as the sky lighting and the environment capture it, in front of the
    /// sky and the clouds (Unreal's real-time sky-light capture); `None` captures the sky alone.
    pub capture_fog: Option<SkyCaptureFog>,
    /// What the sky lighting and the environment see below the horizon.
    pub lower_hemisphere: SkyLowerHemisphere,
    pub params: AtmosphereParams,
    pub sun: CelestialLight,
    pub moon: CelestialLight,
    /// Altitude of the world origin above the planet's surface, metres.
    pub origin_altitude_m: f32,
    /// Albedo of the ground below the horizon in the sky lighting (the light it bounces up);
    /// `None` uses `params.ground_albedo`, zero leaves the lower hemisphere black.
    pub sky_light_ground_albedo: Option<Vec3>,
    bindings: SkyAtmosphereBindings,
    /// The capture's fog and lower hemisphere (WGSL `SkyCapture`), written every update.
    capture: wgpu::Buffer,
    /// Transmittance, multiple-scattering and sky-view LUT textures (the views are in `bindings`).
    luts: [wgpu::Texture; 3],
    /// Aerial-perspective scattering and transmittance volumes.
    ap_volumes: [wgpu::Texture; 2],
    environment: wgpu::Texture,
    cloud_shadow: wgpu::Texture,
    pipelines: Pipelines,
    transmittance_size: (u32, u32),
    multi_scattering_size: u32,
    sky_view_size: (u32, u32),
    ap_size: (u32, u32, u32),
    ap_distance_km: f32,
    /// Atmosphere the static LUTs were last built for.
    built_for: Option<AtmosphereGpu>,
}

fn lut_2d(device: &wgpu::Device, label: &str, (w, h): (u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn volume_3d(device: &wgpu::Device, label: &str, (w, h, d): (u32, u32, u32)) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: d.max(1) },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: LUT_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

pub(crate) fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub(crate) fn texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) fn texture_3d_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D3,
            multisampled: false,
        },
        count: None,
    }
}

pub(crate) fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
        count: None,
    }
}

pub(crate) fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    storage_entry_dim(binding, wgpu::TextureViewDimension::D2)
}

fn storage_entry_dim(binding: u32, view_dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: LUT_FORMAT, view_dimension },
        count: None,
    }
}

pub(crate) fn compute_pipeline(device: &wgpu::Device, label: &str, code: &str, bgl: &wgpu::BindGroupLayout) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn bind_group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, resources: &[wgpu::BindingResource]) -> wgpu::BindGroup {
    let entries: Vec<_> = resources
        .iter()
        .enumerate()
        .map(|(i, r)| wgpu::BindGroupEntry { binding: i as u32, resource: r.clone() })
        .collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

impl SkyAtmosphere {
    pub fn new(device: &wgpu::Device, options: SkyAtmosphereOptions) -> Self {
        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sampler = |label: &str, u: wgpu::AddressMode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: u,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            })
        };
        let ms = options.multi_scattering_size.max(2);
        let luts = [
            lut_2d(device, "SkyAtmosphere/TransmittanceLUT", options.transmittance_size),
            lut_2d(device, "SkyAtmosphere/MultiScatteringLUT", (ms, ms)),
            lut_2d(device, "SkyAtmosphere/SkyViewLUT", options.sky_view_size),
        ];
        let ap_size = options.aerial_perspective_size;
        let ap_volumes = [
            volume_3d(device, "SkyAtmosphere/AerialPerspectiveScattering", ap_size),
            volume_3d(device, "SkyAtmosphere/AerialPerspectiveTransmittance", ap_size),
        ];
        let env_size = options.environment_size.max(1);
        let env_mips = env_size.ilog2() + 1;
        let environment = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SkyAtmosphere/Environment"),
            size: wgpu::Extent3d { width: env_size, height: env_size, depth_or_array_layers: 6 },
            mip_level_count: env_mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: LUT_FORMAT,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let cloud_shadow_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SkyAtmosphere/CloudShadow"),
            size: wgpu::Extent3d { width: CLOUD_SHADOW_SIZE, height: CLOUD_SHADOW_SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let bindings = SkyAtmosphereBindings {
            atmosphere: uniform("SkyAtmosphere/Atmosphere", std::mem::size_of::<AtmosphereGpu>()),
            frame: uniform("SkyAtmosphere/Frame", std::mem::size_of::<SkyFrameGpu>()),
            transmittance: view(&luts[0]),
            multi_scattering: view(&luts[1]),
            sky_view: view(&luts[2]),
            ap_scattering: view(&ap_volumes[0]),
            ap_transmittance: view(&ap_volumes[1]),
            sky_lighting: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SkyAtmosphere/SkyLighting"),
                size: std::mem::size_of::<SkyLightingGpu>() as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            environment: environment.create_view(&wgpu::TextureViewDescriptor {
                label: Some("SkyAtmosphere/EnvironmentCube"),
                dimension: Some(wgpu::TextureViewDimension::Cube),
                ..Default::default()
            }),
            environment_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SkyAtmosphere/EnvironmentSampler"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            lut_sampler: sampler("SkyAtmosphere/LutSampler", wgpu::AddressMode::ClampToEdge),
            sky_view_sampler: sampler("SkyAtmosphere/SkyViewSampler", wgpu::AddressMode::Repeat),
            // zero-initialised: no clouds until the clouds write it
            cloud_map: view(&lut_2d(device, "SkyAtmosphere/CloudMap", CLOUD_MAP_SIZE)),
            cloud_map_frame: Default::default(),
            cloud_shadow: cloud_shadow_texture.create_view(&Default::default()),
            // zero: shadows off until the clouds write them
            cloud_shadow_params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SkyAtmosphere/CloudShadowParams"),
                size: std::mem::size_of::<crate::atmosphere::params::CloudShadowParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            cloud_shadow_frame: Default::default(),
            sun_direction: Default::default(),
        };
        // bound in place of the cloud map while no clouds write it
        let no_clouds = view(&lut_2d(device, "SkyAtmosphere/NoClouds", (1, 1)));
        let capture = uniform("SkyAtmosphere/Capture", std::mem::size_of::<SkyCaptureGpu>());

        let b = &bindings;
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let transmittance_bgl = bgl("SkyAtmosphere/TransmittanceBGL", &[uniform_entry(0), storage_entry(1)]);
        let multi_scattering_bgl = bgl(
            "SkyAtmosphere/MultiScatteringBGL",
            &[uniform_entry(0), texture_entry(1), sampler_entry(2), storage_entry(3)],
        );
        let sky_view_bgl = bgl(
            "SkyAtmosphere/SkyViewBGL",
            &[uniform_entry(0), uniform_entry(1), texture_entry(2), texture_entry(3), sampler_entry(4), storage_entry(5)],
        );
        let aerial_perspective_bgl = bgl(
            "SkyAtmosphere/AerialPerspectiveBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                texture_entry(3),
                sampler_entry(4),
                storage_entry_dim(5, wgpu::TextureViewDimension::D3),
                storage_entry_dim(6, wgpu::TextureViewDimension::D3),
            ],
        );
        let sky_lighting_bgl = bgl(
            "SkyAtmosphere/SkyLightingBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                texture_entry(3),
                sampler_entry(4),
                sampler_entry(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_entry(7),
                uniform_entry(8),
            ],
        );
        let environment_bgl = bgl(
            "SkyAtmosphere/EnvironmentBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                sampler_entry(3),
                uniform_entry(4),
                storage_entry_dim(5, wgpu::TextureViewDimension::D2Array),
                uniform_entry(6),
                texture_entry(7),
                uniform_entry(8),
            ],
        );
        let environment_bgs = (0..env_mips)
            .map(|mip| {
                let target = environment.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("SkyAtmosphere/EnvironmentMip"),
                    dimension: Some(wgpu::TextureViewDimension::D2Array),
                    base_mip_level: mip,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                // each mip has its own parameters, written once: a shared buffer rewritten per
                // dispatch would hold only the last write by the time the passes run
                let roughness = if env_mips > 1 { mip as f32 / (env_mips - 1) as f32 } else { 0.0 };
                let params = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("SkyAtmosphere/EnvironmentMipParams"),
                    size: 16,
                    usage: wgpu::BufferUsages::UNIFORM,
                    mapped_at_creation: true,
                });
                params.slice(..).get_mapped_range_mut().copy_from_slice(bytemuck::cast_slice(&[
                    roughness.to_bits(),
                    ENVIRONMENT_SAMPLES,
                    0,
                    0,
                ]));
                params.unmap();
                [&b.cloud_map, &no_clouds].map(|clouds| {
                    bind_group(
                        device,
                        "SkyAtmosphere/EnvironmentBG",
                        &environment_bgl,
                        &[
                            b.atmosphere.as_entire_binding(),
                            b.frame.as_entire_binding(),
                            wgpu::BindingResource::TextureView(&b.sky_view),
                            wgpu::BindingResource::Sampler(&b.sky_view_sampler),
                            b.sky_lighting.as_entire_binding(),
                            wgpu::BindingResource::TextureView(&target),
                            params.as_entire_binding(),
                            wgpu::BindingResource::TextureView(clouds),
                            capture.as_entire_binding(),
                        ],
                    )
                })
            })
            .collect();
        let tex = wgpu::BindingResource::TextureView;
        let pipelines = Pipelines {
            transmittance: compute_pipeline(device, "SkyAtmosphere/Transmittance", &transmittance_source(), &transmittance_bgl),
            transmittance_bg: bind_group(
                device,
                "SkyAtmosphere/TransmittanceBG",
                &transmittance_bgl,
                &[b.atmosphere.as_entire_binding(), tex(&b.transmittance)],
            ),
            multi_scattering: compute_pipeline(device, "SkyAtmosphere/MultiScattering", &multi_scattering_source(), &multi_scattering_bgl),
            multi_scattering_bg: bind_group(
                device,
                "SkyAtmosphere/MultiScatteringBG",
                &multi_scattering_bgl,
                &[
                    b.atmosphere.as_entire_binding(),
                    tex(&b.transmittance),
                    wgpu::BindingResource::Sampler(&b.lut_sampler),
                    tex(&b.multi_scattering),
                ],
            ),
            sky_view: compute_pipeline(device, "SkyAtmosphere/SkyView", &sky_view_source(), &sky_view_bgl),
            sky_view_bg: bind_group(
                device,
                "SkyAtmosphere/SkyViewBG",
                &sky_view_bgl,
                &[
                    b.atmosphere.as_entire_binding(),
                    b.frame.as_entire_binding(),
                    tex(&b.transmittance),
                    tex(&b.multi_scattering),
                    wgpu::BindingResource::Sampler(&b.lut_sampler),
                    tex(&b.sky_view),
                ],
            ),
            aerial_perspective: compute_pipeline(device, "SkyAtmosphere/AerialPerspective", &aerial_perspective_source(), &aerial_perspective_bgl),
            aerial_perspective_bg: bind_group(
                device,
                "SkyAtmosphere/AerialPerspectiveBG",
                &aerial_perspective_bgl,
                &[
                    b.atmosphere.as_entire_binding(),
                    b.frame.as_entire_binding(),
                    tex(&b.transmittance),
                    tex(&b.multi_scattering),
                    wgpu::BindingResource::Sampler(&b.lut_sampler),
                    tex(&b.ap_scattering),
                    tex(&b.ap_transmittance),
                ],
            ),
            sky_lighting: compute_pipeline(device, "SkyAtmosphere/SkyLighting", &sky_lighting_source(), &sky_lighting_bgl),
            sky_lighting_bgs: [&b.cloud_map, &no_clouds].map(|clouds| {
                bind_group(
                    device,
                    "SkyAtmosphere/SkyLightingBG",
                    &sky_lighting_bgl,
                    &[
                        b.atmosphere.as_entire_binding(),
                        b.frame.as_entire_binding(),
                        tex(&b.transmittance),
                        tex(&b.sky_view),
                        wgpu::BindingResource::Sampler(&b.lut_sampler),
                        wgpu::BindingResource::Sampler(&b.sky_view_sampler),
                        b.sky_lighting.as_entire_binding(),
                        tex(clouds),
                        capture.as_entire_binding(),
                    ],
                )
            }),
            environment: compute_pipeline(device, "SkyAtmosphere/Environment", &environment_source(), &environment_bgl),
            environment_bgs,
        };

        Self {
            params: AtmosphereParams::earth(),
            sun: CelestialLight::sun(),
            moon: CelestialLight::moon(),
            origin_altitude_m: 0.0,
            sky_light_ground_albedo: None,
            capture_fog: None,
            lower_hemisphere: SkyLowerHemisphere::Ground,
            bindings,
            capture,
            luts,
            ap_volumes,
            environment,
            cloud_shadow: cloud_shadow_texture,
            pipelines,
            transmittance_size: options.transmittance_size,
            multi_scattering_size: ms,
            sky_view_size: options.sky_view_size,
            ap_size,
            ap_distance_km: options.aerial_perspective_distance_km.max(1e-3),
            built_for: None,
        }
    }

    /// GPU handles for effects and materials that read the atmosphere.
    pub fn bindings(&self) -> &SkyAtmosphereBindings {
        &self.bindings
    }

    /// Planet-frame position (km) of a world-space point (metres).
    pub fn to_planet_km(&self, world: Vec3) -> glam::Vec3 {
        glam::Vec3::new(world.x, world.y + self.origin_altitude_m, world.z) * 0.001
            + glam::Vec3::new(0.0, self.params.bottom_radius_km, 0.0)
    }

    /// Transmittance of the atmosphere from a world-space point toward the sun: 0 when the sun is
    /// below that point's horizon. Multiply the sun's illuminance by it to light the scene with
    /// the colour the sky is rendered with.
    pub fn sun_transmittance(&self, world: Vec3) -> Vec3 {
        let p = self.to_planet_km(world);
        let r = p.length().max(self.params.bottom_radius_km);
        let mu = (p / p.length()).dot(self.sun.direction.to_glam().normalize_or(glam::Vec3::Y));
        // the part of the disk above the horizon, as the shaders' horizonVisibility
        let bottom = self.params.bottom_radius_km;
        let horizon = -((r - bottom) * (r + bottom)).max(0.0).sqrt() / r;
        let w = self.sun.angular_radius();
        let x = ((mu - horizon + w) / (2.0 * w)).clamp(0.0, 1.0);
        let visibility = x * x * (3.0 - 2.0 * x);
        let t = self.params.transmittance_to_space(r - bottom, mu.max(horizon + 1e-5)) * visibility;
        Vec3::new(t.x, t.y, t.z)
    }

    /// The sun's illuminance after the atmosphere at a world-space point: the colour (times
    /// intensity) of a directional light that agrees with the sky.
    pub fn sun_illuminance_at(&self, world: Vec3) -> Vec3 {
        let t = self.sun_transmittance(world);
        Vec3::new(self.sun.illuminance.x * t.x, self.sun.illuminance.y * t.y, self.sun.illuminance.z * t.z)
    }

    fn frame(&self, camera: &Camera) -> SkyFrameGpu {
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let eye = camera.inverse_view_matrix.to_glam().w_axis;
        let bottom = self.params.bottom_radius_km;
        let top = self.params.top_radius_km();
        // keep the camera inside the atmosphere, above the ground by a few metres
        let p = self.to_planet_km(Vec3::new(eye.x, eye.y, eye.z));
        let r = p.length().clamp(bottom + MIN_CAMERA_ALTITUDE_KM, top - 1e-3);
        let camera_pos = p.normalize_or(glam::Vec3::Y) * r;
        let origin = self.to_planet_km(Vec3::ZERO);
        let dir = |l: &CelestialLight| l.direction.to_glam().normalize_or(glam::Vec3::Y).to_array();
        let rgb = |v: Vec3| [v.x, v.y, v.z];
        let f = self.params.sky_luminance_factor;
        SkyFrameGpu {
            inv_view_proj: view_proj.inverse().to_cols_array(),
            camera_pos: camera_pos.to_array(),
            ap_distance: self.ap_distance_km,
            world_origin: origin.to_array(),
            ap_start_depth: self.params.aerial_perspective_start_depth_km.max(0.0),
            camera_world: [eye.x, eye.y, eye.z],
            ap_distance_scale: self.params.aerial_perspective_view_distance_scale.max(0.0),
            sun_direction: dir(&self.sun),
            sun_angular_radius: self.sun.angular_radius(),
            sun_illuminance: rgb(self.sun.illuminance),
            sun_disk_luminance: self.sun.disk_luminance(SUN_LIMB_DARKENING),
            moon_direction: dir(&self.moon),
            moon_angular_radius: self.moon.angular_radius(),
            moon_illuminance: rgb(self.moon.illuminance),
            moon_disk_luminance: self.moon.disk_luminance(0.0),
            sky_luminance_factor: [f.x, f.y, f.z],
            _pad0: 0.0,
            sky_light_ground_albedo: rgb(self.sky_light_ground_albedo.unwrap_or(self.params.ground_albedo)),
            _pad1: 0.0,
        }
    }

    /// Record this frame's LUT passes: the transmittance and multiple-scattering LUTs when the
    /// atmosphere changed, then the sky-view LUT for the camera. Uses the camera's current view
    /// and projection matrices, so place the camera first.
    pub fn encode(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, camera: &Camera) {
        // the cloud map (0) if the clouds wrote it this frame or the last, else none (1)
        let fresh = |stamp: &std::sync::atomic::AtomicU32| {
            let written = stamp.load(std::sync::atomic::Ordering::Relaxed);
            written != 0 && camera.frame().wrapping_sub(written - 1) <= 1
        };
        let clouds = if fresh(&self.bindings.cloud_map_frame) { 0 } else { 1 };
        if let Ok(mut sun) = self.bindings.sun_direction.lock() {
            *sun = self.sun.direction.to_glam().normalize_or(glam::Vec3::Y);
        }
        // no clouds drawn lately: their shadows are off (the clouds copy the parameters in when drawn)
        if !fresh(&self.bindings.cloud_shadow_frame) {
            queue.write_buffer(&self.bindings.cloud_shadow_params, 0, bytemuck::bytes_of(&<crate::atmosphere::params::CloudShadowParamsGpu as bytemuck::Zeroable>::zeroed()));
        }
        let atmosphere = self.params.gpu_layout();
        let b = &self.bindings;
        let p = &self.pipelines;
        if self.built_for != Some(atmosphere) {
            queue.write_buffer(&b.atmosphere, 0, bytemuck::bytes_of(&atmosphere));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SkyAtmosphere/StaticLUTs"), ..Default::default() });
            pass.set_pipeline(&p.transmittance);
            pass.set_bind_group(0, &p.transmittance_bg, &[]);
            pass.dispatch_workgroups(self.transmittance_size.0.div_ceil(8), self.transmittance_size.1.div_ceil(8), 1);
            // one workgroup per texel
            pass.set_pipeline(&p.multi_scattering);
            pass.set_bind_group(0, &p.multi_scattering_bg, &[]);
            pass.dispatch_workgroups(self.multi_scattering_size, self.multi_scattering_size, 1);
            self.built_for = Some(atmosphere);
        }

        queue.write_buffer(&b.frame, 0, bytemuck::bytes_of(&self.frame(camera)));
        queue.write_buffer(&self.capture, 0, bytemuck::bytes_of(&self.capture_gpu()));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SkyAtmosphere/SkyView"), ..Default::default() });
        pass.set_pipeline(&p.sky_view);
        pass.set_bind_group(0, &p.sky_view_bg, &[]);
        pass.dispatch_workgroups(self.sky_view_size.0.div_ceil(8), self.sky_view_size.1.div_ceil(8), 1);
        pass.set_pipeline(&p.aerial_perspective);
        pass.set_bind_group(0, &p.aerial_perspective_bg, &[]);
        pass.dispatch_workgroups(self.ap_size.0.div_ceil(8), self.ap_size.1.div_ceil(8), 1);
        pass.set_pipeline(&p.sky_lighting);
        pass.set_bind_group(0, &p.sky_lighting_bgs[clouds], &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&p.environment);
        for (mip, bg) in p.environment_bgs.iter().map(|bgs| &bgs[clouds]).enumerate() {
            let size = (self.environment.width() >> mip).max(1);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 6);
        }
    }

    /// `encode` on a fresh encoder, submitted at once. Call every frame after placing the camera
    /// and before rendering: the scene's materials and the post effects read what it writes.
    pub fn update(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, camera: &mut Camera) {
        camera.update_view_matrix();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("SkyAtmosphere") });
        self.encode(queue, &mut encoder, camera);
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The LUT textures: transmittance, multiple scattering, sky view (rgba16float).
    pub fn lut_textures(&self) -> &[wgpu::Texture; 3] {
        &self.luts
    }

    /// The aerial-perspective volumes: scattering, transmittance (rgba16float, 3D).
    pub fn aerial_perspective_textures(&self) -> &[wgpu::Texture; 2] {
        &self.ap_volumes
    }

    /// The sky environment cubemap texture (6 layers, GGX-prefiltered mips).
    pub fn environment_texture(&self) -> &wgpu::Texture {
        &self.environment
    }

    fn capture_gpu(&self) -> SkyCaptureGpu {
        let layer = |l: &HeightFogLayer| [l.density.max(0.0), l.height_falloff, l.height, 0.0];
        let rgb = |v: Vec3| [v.x, v.y, v.z];
        let (lower_color, lower_mode) = match self.lower_hemisphere {
            SkyLowerHemisphere::Ground => ([0.0; 3], 0),
            SkyLowerHemisphere::Color(c) => (rgb(c), 1),
        };
        match &self.capture_fog {
            Some(fog) => SkyCaptureGpu {
                layer0: layer(&fog.layers[0]),
                layer1: layer(&fog.layers[1]),
                inscattering: rgb(fog.inscattering),
                fog_on: 1,
                lower_color,
                lower_mode,
                capture_height: fog.capture_height_m,
                max_opacity: fog.max_opacity.clamp(0.0, 1.0),
                _pad: [0.0; 2],
            },
            None => SkyCaptureGpu { lower_color, lower_mode, ..Zeroable::zeroed() },
        }
    }

    /// The clouds' shadow map (`SkyAtmosphereBindings::cloud_shadow`), for binding it in a material.
    pub fn cloud_shadow_texture(&self) -> &wgpu::Texture {
        &self.cloud_shadow
    }

    /// Force the transmittance and multiple-scattering LUTs to be rebuilt on the next update.
    pub fn invalidate(&mut self) {
        self.built_for = None;
    }
}
