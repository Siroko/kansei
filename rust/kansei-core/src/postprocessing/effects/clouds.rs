use bytemuck::{Pod, Zeroable};

use crate::atmosphere::sky_atmosphere::{
    compute_pipeline, sampler_entry, texture_3d_entry, texture_entry, uniform_entry, AERIAL_PERSPECTIVE_LOOKUP_WGSL, CLOUD_MAP_SIZE,
    CLOUD_MAP_WGSL, CLOUD_SHADOW_SIZE, CLOUD_SHADOW_WGSL, COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, SKY_LIGHTING_WGSL,
    SKY_LOOKUP_WGSL,
};
use crate::atmosphere::params::CloudShadowParamsGpu;
use crate::atmosphere::{SkyAtmosphere, SkyAtmosphereBindings};
use crate::cameras::Camera;
use crate::math::Vec3;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const NOISE_WGSL: &str = include_str!("../../atmosphere/shaders/clouds_noise.wgsl");
const MARCH_WGSL: &str = include_str!("../../atmosphere/shaders/clouds_march.wgsl");
const COMPOSITE_WGSL: &str = include_str!("../../atmosphere/shaders/clouds_composite.wgsl");

/// Texels of the 3D shape noise per side, of the 3D detail noise, and of the 2D weather map.
const SHAPE_SIZE: u32 = 128;
const DETAIL_SIZE: u32 = 32;
const WEATHER_SIZE: u32 = 256;

fn march_source() -> String {
    [COMMON_WGSL, FRAME_WGSL, LOOKUP_TRANSMITTANCE_WGSL, AERIAL_PERSPECTIVE_LOOKUP_WGSL, SKY_LOOKUP_WGSL, SKY_LIGHTING_WGSL, CLOUD_MAP_WGSL, CLOUD_SHADOW_WGSL, MARCH_WGSL].concat()
}

fn composite_source() -> String {
    [FRAME_WGSL, COMPOSITE_WGSL].concat()
}

/// A layer of cloud around the planet, between two altitudes. Distances are in metres.
#[derive(Debug, Clone, Copy)]
pub struct CloudLayer {
    /// Altitude of the layer's base, above the planet's surface.
    pub bottom_m: f32,
    /// Altitude of its top.
    pub top_m: f32,
    /// How much of the sky it covers: 0 clear, 1 overcast.
    pub coverage: f32,
    /// 0 flat stratus, 0.5 stratocumulus, 1 towering cumulus (the weather map varies it).
    pub cloud_type: f32,
    /// Extinction per metre at full density (real clouds: about 0.02 to 0.1).
    pub extinction: f32,
    /// Single-scattering albedo (water droplets: 0.99 and above).
    pub albedo: Vec3,
    /// Wind, metres per second: the clouds drift by `wind * time`.
    pub wind: Vec3,
    /// Size of one repeat of the shape noise (the billows are a quarter of it).
    pub shape_size_m: f32,
    /// Size of one repeat of the detail noise, which frays the edges.
    pub detail_size_m: f32,
    /// Size of one repeat of the weather map, which decides where clouds form.
    pub weather_size_m: f32,
}

impl Default for CloudLayer {
    fn default() -> Self {
        Self {
            bottom_m: 1500.0,
            top_m: 4000.0,
            coverage: 0.5,
            cloud_type: 0.7,
            extinction: 0.05,
            albedo: Vec3::new(0.99, 0.99, 0.99),
            wind: Vec3::new(10.0, 0.0, 3.0),
            shape_size_m: 9000.0,
            detail_size_m: 700.0,
            weather_size_m: 40000.0,
        }
    }
}

pub struct VolumetricCloudsOptions {
    pub layer: CloudLayer,
    /// Resolution of the march relative to the image (0.5: a quarter of the pixels).
    pub resolution_scale: f32,
    /// Steps along each view ray through the layer, and toward the sun from each step.
    pub steps: u32,
    pub light_steps: u32,
    /// Farthest the march goes into the layer, metres (the horizon's clouds beyond it fade into
    /// the aerial perspective).
    pub max_distance_m: f32,
    /// Weight of each new frame in the accumulated clouds (lower: smoother, slower to follow).
    pub temporal_blend: f32,
}

impl Default for VolumetricCloudsOptions {
    fn default() -> Self {
        Self { layer: CloudLayer::default(), resolution_scale: 0.5, steps: 64, light_steps: 6, max_distance_m: 60000.0, temporal_blend: 0.1 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CloudParamsGpu {
    prev_view_proj: [f32; 16],
    wind_offset: [f32; 3],
    coverage: f32,
    bottom_km: f32,
    top_km: f32,
    extinction: f32,
    cloud_type: f32,
    albedo: [f32; 3],
    shape_scale: f32,
    detail_scale: f32,
    weather_scale: f32,
    max_distance: f32,
    frame: u32,
    size: [u32; 2],
    history_valid: u32,
    steps: u32,
    light_steps: u32,
    blend: f32,
    _pad: [f32; 2],
}

struct Targets {
    width: u32,
    height: u32,
    /// Ping-pong: each frame writes one and reads the other as its history.
    color: [wgpu::TextureView; 2],
    depth: wgpu::TextureView,
}

struct Gpu {
    params: wgpu::Buffer,
    march: wgpu::ComputePipeline,
    march_bgl: wgpu::BindGroupLayout,
    // the cloud map for the sky lighting (SkyAtmosphereBindings::cloud_map)
    sky_map: wgpu::ComputePipeline,
    sky_map_bgl: wgpu::BindGroupLayout,
    // the shadow map (SkyAtmosphereBindings::cloud_shadow), and its parameters staged for it and
    // copied to the bindings after it is written
    shadow_map: wgpu::ComputePipeline,
    shadow_map_bgl: wgpu::BindGroupLayout,
    shadow_staging: wgpu::Buffer,
    composite: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    shape: wgpu::TextureView,
    detail: wgpu::TextureView,
    weather: wgpu::TextureView,
    noise_sampler: wgpu::Sampler,
    noise_ready: bool,
    #[allow(dead_code)]
    noise_textures: [wgpu::Texture; 3],
    noise: (wgpu::ComputePipeline, wgpu::ComputePipeline, wgpu::ComputePipeline, wgpu::BindGroup),
    targets: Option<Targets>,
}

/// Volumetric clouds over a [`SkyAtmosphere`] (`CloudLayer`): a layer of cloud between two
/// altitudes, shaped by a weather map, a height profile and Perlin-Worley noise, lit by the sun
/// through the atmosphere and through the cloud (with an approximation of multiple scattering)
/// and by the sky, and seen through the atmosphere in front of it. They are marched at a reduced
/// resolution with a jittered start, accumulated over frames by reprojection, and composited over
/// the sky and over any surface they are in front of.
///
/// Put it right after the `AtmosphereEffect` (before the fogs), and set `time` every frame for
/// the wind. `layer` can change every frame (coverage for the weather of a shot). The clouds are
/// part of the sky: `AtmosphereParams::sky_luminance_factor` scales the light they send like the
/// sky's.
///
/// They also light the scene: each frame they march a small map of themselves all around the
/// camera (`SkyAtmosphereBindings::cloud_map`), and the next frame's sky lighting (the SH the
/// materials and fogs take their ambient light from) and environment cubemap see the sky through
/// it, so the ambient light is occluded and tinted by the cloud layer. The clouds themselves stay
/// lit by the clear sky above them. `lights_sky = false` leaves the sky lighting clear, as does
/// taking the effect out of the chain.
pub struct VolumetricCloudsEffect {
    pub layer: CloudLayer,
    /// Whether the clouds occlude and tint the sky lighting and the environment (default true).
    pub lights_sky: bool,
    /// Whether the clouds shadow the scene from the sun (`SkyAtmosphereBindings::cloud_shadow`,
    /// default true).
    pub casts_shadows: bool,
    /// The side of the shadow map (m), centred under the camera; beyond it there are no cloud
    /// shadows. 256 texels, so 16 km gives 62.5 m per texel.
    pub shadow_size_m: f32,
    /// Seconds, drives the wind.
    pub time: f32,
    pub resolution_scale: f32,
    pub steps: u32,
    pub light_steps: u32,
    pub max_distance_m: f32,
    pub temporal_blend: f32,
    sky: SkyAtmosphereBindings,
    prev_view_proj: Option<glam::Mat4>,
    frame: u32,
    gpu: Option<Gpu>,
}

impl VolumetricCloudsEffect {
    pub fn new(sky: &SkyAtmosphere, options: VolumetricCloudsOptions) -> Self {
        Self {
            layer: options.layer,
            lights_sky: true,
            casts_shadows: true,
            shadow_size_m: 16_000.0,
            time: 0.0,
            resolution_scale: options.resolution_scale,
            steps: options.steps,
            light_steps: options.light_steps,
            max_distance_m: options.max_distance_m,
            temporal_blend: options.temporal_blend,
            sky: sky.bindings().clone(),
            prev_view_proj: None,
            frame: 0,
            gpu: None,
        }
    }

    /// Drop the accumulated frames; call on camera cuts.
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        vec![
            ("clouds_noise", NOISE_WGSL.to_string()),
            ("clouds_march", march_source()),
            ("clouds_composite", composite_source()),
            ("cloud_shadow", CLOUD_SHADOW_WGSL.to_string()),
        ]
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let storage = |binding, format, view_dimension| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension },
            count: None,
        };
        let unfiltered = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let depth_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });

        // the noise textures, generated once
        let texture = |label: &str, size: wgpu::Extent3d, dimension| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
        };
        let cube = |n| wgpu::Extent3d { width: n, height: n, depth_or_array_layers: n };
        let shape_texture = texture("Clouds/Shape", cube(SHAPE_SIZE), wgpu::TextureDimension::D3);
        let detail_texture = texture("Clouds/Detail", cube(DETAIL_SIZE), wgpu::TextureDimension::D3);
        let weather_texture = texture("Clouds/Weather", wgpu::Extent3d { width: WEATHER_SIZE, height: WEATHER_SIZE, depth_or_array_layers: 1 }, wgpu::TextureDimension::D2);
        let (shape, detail, weather) = (shape_texture.create_view(&Default::default()), detail_texture.create_view(&Default::default()), weather_texture.create_view(&Default::default()));
        let noise_bgl = bgl(
            "Clouds/NoiseBGL",
            &[
                storage(0, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureViewDimension::D3),
                storage(1, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureViewDimension::D3),
                storage(2, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureViewDimension::D2),
            ],
        );
        let noise_pipeline = |entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Clouds/Noise"), source: wgpu::ShaderSource::Wgsl(NOISE_WGSL.into()) });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Clouds/Noise"), bind_group_layouts: &[&noise_bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clouds/Noise"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let noise_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clouds/NoiseBG"),
            layout: &noise_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&shape) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&detail) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&weather) },
            ],
        });

        let march_bgl = bgl(
            "Clouds/MarchBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                sampler_entry(3),
                texture_3d_entry(4),
                texture_3d_entry(5),
                uniform_entry(6),
                depth_entry(7),
                texture_3d_entry(8),
                texture_3d_entry(9),
                texture_entry(10),
                sampler_entry(11),
                texture_entry(12),
                storage(13, wgpu::TextureFormat::Rgba16Float, wgpu::TextureViewDimension::D2),
                storage(14, wgpu::TextureFormat::R32Float, wgpu::TextureViewDimension::D2),
                uniform_entry(15),
            ],
        );
        let sky_map_bgl = bgl(
            "Clouds/SkyMapBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_entry(2),
                sampler_entry(3),
                uniform_entry(6),
                texture_3d_entry(8),
                texture_3d_entry(9),
                texture_entry(10),
                sampler_entry(11),
                uniform_entry(15),
                texture_entry(16),
                sampler_entry(17),
                storage(18, wgpu::TextureFormat::Rgba16Float, wgpu::TextureViewDimension::D2),
            ],
        );
        let sky_map = {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Clouds/SkyMap"), source: wgpu::ShaderSource::Wgsl(march_source().into()) });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Clouds/SkyMap"), bind_group_layouts: &[&sky_map_bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clouds/SkyMap"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("skyMap"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let shadow_map_bgl = bgl(
            "Clouds/ShadowMapBGL",
            &[
                uniform_entry(0),
                uniform_entry(1),
                texture_3d_entry(8),
                texture_3d_entry(9),
                texture_entry(10),
                sampler_entry(11),
                uniform_entry(15),
                storage(19, wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureViewDimension::D2),
                uniform_entry(20),
            ],
        );
        let shadow_map = {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Clouds/ShadowMap"), source: wgpu::ShaderSource::Wgsl(march_source().into()) });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Clouds/ShadowMap"), bind_group_layouts: &[&shadow_map_bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Clouds/ShadowMap"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("shadowMap"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let composite_bgl = bgl(
            "Clouds/CompositeBGL",
            &[
                uniform_entry(0),
                unfiltered(1),
                depth_entry(2),
                unfiltered(3),
                unfiltered(4),
                storage(5, GBuffer::COLOR_FORMAT, wgpu::TextureViewDimension::D2),
            ],
        );
        self.gpu = Some(Gpu {
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Clouds/Params"),
                size: std::mem::size_of::<CloudParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            march: compute_pipeline(device, "Clouds/March", &march_source(), &march_bgl),
            march_bgl,
            sky_map,
            sky_map_bgl,
            shadow_map,
            shadow_map_bgl,
            shadow_staging: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Clouds/ShadowParamsStaging"),
                size: std::mem::size_of::<CloudShadowParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            composite: compute_pipeline(device, "Clouds/Composite", &composite_source(), &composite_bgl),
            composite_bgl,
            shape,
            detail,
            weather,
            noise_sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Clouds/NoiseSampler"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                address_mode_w: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            noise_ready: false,
            noise_textures: [shape_texture, detail_texture, weather_texture],
            noise: (noise_pipeline("shape"), noise_pipeline("detail"), noise_pipeline("weather"), noise_bg),
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (w, h) = (((width as f32 * self.resolution_scale).ceil() as u32).max(1), ((height as f32 * self.resolution_scale).ceil() as u32).max(1));
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == w && t.height == h) {
            return;
        }
        let texture = |label: &str, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let color = [texture("Clouds/ColorA", wgpu::TextureFormat::Rgba16Float), texture("Clouds/ColorB", wgpu::TextureFormat::Rgba16Float)];
        let cloud_depth = texture("Clouds/Depth", wgpu::TextureFormat::R32Float);
        gpu.targets = Some(Targets { width: w, height: h, color, depth: cloud_depth });
        self.prev_view_proj = None;
    }
}

impl PostProcessingEffect for VolumetricCloudsEffect {
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
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        self.ensure_targets(device, width, height);
        let gpu = self.gpu.as_mut().unwrap();
        if !gpu.noise_ready {
            let (shape, detail, weather, bg) = &gpu.noise;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Clouds/Noise"), ..Default::default() });
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(shape);
            pass.dispatch_workgroups(SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
            pass.set_pipeline(detail);
            pass.dispatch_workgroups(DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
            pass.set_pipeline(weather);
            pass.dispatch_workgroups(WEATHER_SIZE / 8, WEATHER_SIZE / 8, 1);
            gpu.noise_ready = true;
        }
        let t = gpu.targets.as_ref().unwrap();
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let l = &self.layer;
        let km = |m: f32| m * 0.001;
        let wind = l.wind * (self.time * 0.001);
        let params = CloudParamsGpu {
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            wind_offset: [wind.x, wind.y, wind.z],
            coverage: l.coverage.clamp(0.0, 1.0),
            bottom_km: km(l.bottom_m),
            top_km: km(l.top_m.max(l.bottom_m + 1.0)),
            extinction: l.extinction.max(0.0) * 1000.0,
            cloud_type: l.cloud_type.clamp(0.0, 1.0),
            albedo: [l.albedo.x, l.albedo.y, l.albedo.z],
            shape_scale: 1.0 / km(l.shape_size_m.max(1.0)),
            detail_scale: 1.0 / km(l.detail_size_m.max(1.0)),
            weather_scale: 1.0 / km(l.weather_size_m.max(1.0)),
            max_distance: km(self.max_distance_m.max(1.0)),
            frame: self.frame,
            size: [t.width, t.height],
            history_valid: self.prev_view_proj.is_some() as u32,
            steps: self.steps.max(1),
            light_steps: self.light_steps.max(1),
            blend: self.temporal_blend.clamp(0.01, 1.0),
            _pad: [0.0; 2],
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);

        let tex = wgpu::BindingResource::TextureView;
        let s = &self.sky;
        let march_bg = {
            let resources = [
                s.atmosphere.as_entire_binding(),
                s.frame.as_entire_binding(),
                tex(&s.transmittance),
                wgpu::BindingResource::Sampler(&s.lut_sampler),
                tex(&s.ap_scattering),
                tex(&s.ap_transmittance),
                s.sky_lighting.as_entire_binding(),
                tex(depth),
                tex(&gpu.shape),
                tex(&gpu.detail),
                tex(&gpu.weather),
                wgpu::BindingResource::Sampler(&gpu.noise_sampler),
                tex(&t.color[1 - current]),
                tex(&t.color[current]),
                tex(&t.depth),
                gpu.params.as_entire_binding(),
            ];
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Clouds/MarchBG"), layout: &gpu.march_bgl, entries: &entries })
        };
        let sky_map_bg = {
            let sampler = wgpu::BindingResource::Sampler;
            let entries = [
                (0, s.atmosphere.as_entire_binding()),
                (1, s.frame.as_entire_binding()),
                (2, tex(&s.transmittance)),
                (3, sampler(&s.lut_sampler)),
                (6, s.sky_lighting.as_entire_binding()),
                (8, tex(&gpu.shape)),
                (9, tex(&gpu.detail)),
                (10, tex(&gpu.weather)),
                (11, sampler(&gpu.noise_sampler)),
                (15, gpu.params.as_entire_binding()),
                (16, tex(&s.sky_view)),
                (17, sampler(&s.sky_view_sampler)),
                (18, tex(&s.cloud_map)),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource });
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Clouds/SkyMapBG"), layout: &gpu.sky_map_bgl, entries: &entries })
        };
        let composite_bg = {
            let resources = [self.sky.frame.as_entire_binding(), tex(input), tex(depth), tex(&t.color[current]), tex(&t.depth), tex(output)];
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Clouds/CompositeBG"), layout: &gpu.composite_bgl, entries: &entries })
        };
        let shadow_bg = if self.casts_shadows {
            let sun = self.sky.sun_direction.lock().map(|s| *s).unwrap_or(glam::Vec3::Y);
            let eye = camera.inverse_view_matrix.to_glam().w_axis;
            let size = self.shadow_size_m.max(100.0);
            // centred under the camera, snapped to whole texels so the shadows don't crawl
            let texel = size / CLOUD_SHADOW_SIZE as f32;
            let center = [(eye.x / texel).round() * texel, (eye.z / texel).round() * texel];
            let shadow = CloudShadowParamsGpu { center, inv_size: 1.0 / size, plane_y: eye.y, sun_dir: sun.to_array(), enabled: (sun.y > 0.01) as u32 as f32 };
            queue.write_buffer(&gpu.shadow_staging, 0, bytemuck::bytes_of(&shadow));
            let entries = [
                (0, s.atmosphere.as_entire_binding()),
                (1, s.frame.as_entire_binding()),
                (8, tex(&gpu.shape)),
                (9, tex(&gpu.detail)),
                (10, tex(&gpu.weather)),
                (11, wgpu::BindingResource::Sampler(&gpu.noise_sampler)),
                (15, gpu.params.as_entire_binding()),
                (19, tex(&s.cloud_shadow)),
                (20, gpu.shadow_staging.as_entire_binding()),
            ]
            .map(|(binding, resource)| wgpu::BindGroupEntry { binding, resource });
            Some(device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Clouds/ShadowMapBG"), layout: &gpu.shadow_map_bgl, entries: &entries }))
        } else {
            None
        };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Clouds"), ..Default::default() });
        pass.set_pipeline(&gpu.march);
        pass.set_bind_group(0, &march_bg, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        // the clouds all around, for the sky lighting and the environment of the next frame
        if self.lights_sky {
            pass.set_pipeline(&gpu.sky_map);
            pass.set_bind_group(0, &sky_map_bg, &[]);
            pass.dispatch_workgroups(CLOUD_MAP_SIZE.0.div_ceil(8), CLOUD_MAP_SIZE.1.div_ceil(8), 1);
            self.sky.cloud_map_frame.store(camera.frame().wrapping_add(1), std::sync::atomic::Ordering::Relaxed);
        }
        // the shadow on the scene from the sun, for the next frame's materials
        let sun = self.sky.sun_direction.lock().map(|s| *s).unwrap_or(glam::Vec3::Y);
        let shadows = self.casts_shadows && sun.y > 0.01;
        if shadows {
            pass.set_pipeline(&gpu.shadow_map);
            pass.set_bind_group(0, shadow_bg.as_ref().unwrap(), &[]);
            pass.dispatch_workgroups(CLOUD_SHADOW_SIZE.div_ceil(8), CLOUD_SHADOW_SIZE.div_ceil(8), 1);
        }
        pass.set_pipeline(&gpu.composite);
        pass.set_bind_group(0, &composite_bg, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        drop(pass);
        if self.casts_shadows {
            // the parameters of the map just written (or its shadows off, with the sun down)
            encoder.copy_buffer_to_buffer(&gpu.shadow_staging, 0, &self.sky.cloud_shadow_params, 0, std::mem::size_of::<CloudShadowParamsGpu>() as u64);
            self.sky.cloud_shadow_frame.store(camera.frame().wrapping_add(1), std::sync::atomic::Ordering::Relaxed);
        }
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

    /// The cloud shaders parse and validate with naga, and `CloudParams` matches its Rust layout.
    #[test]
    fn shaders_validate_and_the_params_layout_matches() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in VolumetricCloudsEffect::shader_sources() {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["CloudParams"], std::mem::size_of::<CloudParamsGpu>());
        assert_eq!(sizes["CloudShadowParams"], std::mem::size_of::<CloudShadowParamsGpu>());
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
    }

    /// The generated noises use their whole range, so coverage and erosion thresholds carve them
    /// evenly (raw Perlin and Worley sums span only a narrow band).
    #[test]
    fn the_cloud_noises_span_their_range() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let sky = crate::atmosphere::SkyAtmosphere::new(&device, Default::default());
        let mut fx = VolumetricCloudsEffect::new(&sky, Default::default());
        fx.init_gpu(&device);
        let gpu = fx.gpu.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let (shape, detail, weather, bg) = &gpu.noise;
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_bind_group(0, bg, &[]);
            pass.set_pipeline(shape);
            pass.dispatch_workgroups(SHAPE_SIZE / 4, SHAPE_SIZE / 4, SHAPE_SIZE / 4);
            pass.set_pipeline(detail);
            pass.dispatch_workgroups(DETAIL_SIZE / 4, DETAIL_SIZE / 4, DETAIL_SIZE / 4);
            pass.set_pipeline(weather);
            pass.dispatch_workgroups(WEATHER_SIZE / 8, WEATHER_SIZE / 8, 1);
        }
        queue.submit([encoder.finish()]);
        for (i, (name, n, d, channels)) in [("shape", SHAPE_SIZE, SHAPE_SIZE, 4usize), ("detail", DETAIL_SIZE, DETAIL_SIZE, 3), ("weather", WEATHER_SIZE, 1, 2)].into_iter().enumerate() {
            let row = (n * 4).div_ceil(256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * n * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &gpu.noise_textures[i], mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(n) } },
                wgpu::Extent3d { width: n, height: n, depth_or_array_layers: d },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            for c in 0..channels {
                let mut v: Vec<u8> = Vec::new();
                for z in 0..d {
                    for y in 0..n {
                        for x in 0..n {
                            v.push(data[((z * n + y) * row + x * 4) as usize + c]);
                        }
                    }
                }
                v.sort_unstable();
                let q = |f: f32| v[((v.len() - 1) as f32 * f) as usize] as f32 / 255.0;
                let (lo, hi) = (q(0.05), q(0.95));
                assert!(lo < 0.4 && hi > 0.9 && hi - lo > 0.55, "{name} channel {c}: 5% {lo:.2}, 95% {hi:.2}");
            }
        }
    }

    /// With no coverage the image passes through untouched; an overcast deck seen from below under
    /// a 100 000 lux sun is grey, as real overcast bases are (about 1 000 to 10 000 cd/m2), not
    /// black: light diffuses through thick cloud.
    #[test]
    fn a_clear_sky_passes_through_and_an_overcast_one_covers_it() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let (w, h) = (64u32, 32u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        // a uniform background of 1000 behind everything
        let input_tex = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let sky_value: Vec<f32> = (0..w * h).flat_map(|_| [1000.0f32, 1000.0, 1000.0, 1.0]).collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &input_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&sky_value),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 16), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        let input = input_tex.create_view(&Default::default());
        let output_tex = texture(GBuffer::COLOR_FORMAT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let output = output_tex.create_view(&Default::default());
        let depth_tex = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = depth_tex.create_view(&Default::default());
        {
            let mut e = device.create_command_encoder(&Default::default());
            e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            queue.submit([e.finish()]);
        }
        let gbuffer = GBuffer::new(&device, w, h, 1);
        // looking straight up from the ground at midday
        let mut camera = Camera::new(60.0, 0.1, 5000.0, w as f32 / h as f32);
        camera.set_position(0.0, 2.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 100.0, -1.0));
        camera.update_view_matrix();
        let mut sky = crate::atmosphere::SkyAtmosphere::new(&device, Default::default());
        sky.sun.direction = crate::atmosphere::direction_from_elevation_bearing(60.0, 180.0);
        sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);
        let run = |sky: &mut crate::atmosphere::SkyAtmosphere, coverage: f32| -> Vec<[f32; 4]> {
            let mut fx = VolumetricCloudsEffect::new(sky, VolumetricCloudsOptions { layer: CloudLayer { coverage, ..Default::default() }, ..Default::default() });
            for _ in 0..30 {
                let mut encoder = device.create_command_encoder(&Default::default());
                sky.encode(&queue, &mut encoder, &camera);
                fx.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
                queue.submit([encoder.finish()]);
            }
            let row = (w * 8).div_ceil(256) * 256;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: &output_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
                wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            );
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let data = buf.slice(..).get_mapped_range();
            let half = |o: usize| {
                let h = u16::from_le_bytes([data[o], data[o + 1]]);
                let e = ((h >> 10) & 0x1f) as i32;
                let m = (h & 0x3ff) as f32;
                if e == 0 { m * 2f32.powi(-24) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
            };
            (0..h).flat_map(|y| (0..w).map(move |x| (y, x))).map(|(y, x)| {
                let o = (y * row + x * 8) as usize;
                [half(o), half(o + 2), half(o + 4), half(o + 6)]
            }).collect()
        };
        // the sky lighting after one more update, which reads the cloud map the clouds left
        let lighting_at = |sky: &mut crate::atmosphere::SkyAtmosphere, camera: &Camera| -> Vec<f32> {
            let size = std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            sky.encode(&queue, &mut e, camera);
            e.copy_buffer_to_buffer(&sky.bindings().sky_lighting, 0, &buf, 0, size);
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let v = bytemuck::cast_slice::<u8, f32>(&buf.slice(..).get_mapped_range()).to_vec();
            v
        };
        let lighting = |sky: &mut crate::atmosphere::SkyAtmosphere| lighting_at(sky, &camera);
        // irradiance on an upward surface from the SH (skyIrradiance at +Y), and the clear sky's
        let up = |l: &[f32]| -> glam::Vec3 {
            let c = |i: usize, k: usize| l[i * 4 + k];
            let e = |k: usize| c(0, k) * 0.282095 * 3.141593 + c(1, k) * 0.488603 * 2.094395 - c(6, k) * 0.315392 * 0.785398 - c(8, k) * 0.546274 * 0.785398;
            glam::Vec3::new(e(0), e(1), e(2))
        };
        let clear_up = |l: &[f32]| glam::Vec3::new(l[13 * 4], l[13 * 4 + 1], l[13 * 4 + 2]);
        let none = lighting(&mut sky);

        let clear = run(&mut sky, 0.0);
        let worst = clear.iter().map(|c| (c[0] - 1000.0).abs().max((c[2] - 1000.0).abs())).fold(0.0, f32::max);
        assert!(worst < 1.0, "a clear sky changed by {worst}");
        // no clouds leave the sky lighting as it was
        let lit_clear = lighting(&mut sky);
        let drift = none.iter().zip(&lit_clear).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0, f32::max);
        assert!(drift < 1e-3, "a clear layer changed the sky lighting by {drift}");
        let overcast = run(&mut sky, 1.0);
        // a closed deck takes the blue sky out of the ambient light and puts its grey base in (under
        // a high sun about as bright as the clear sky's diffuse light, as a medium overcast is);
        // the clouds themselves stay lit by the clear sky above them
        let lit_overcast = lighting(&mut sky);
        let (e_clear, e_overcast) = (up(&none), up(&lit_overcast));
        let ratio = e_overcast / e_clear;
        eprintln!("irradiance up: clear sky {e_clear:?}, overcast {e_overcast:?}, ratio {ratio:?}");
        assert!(ratio.y > 0.3 && ratio.y < 1.5, "overcast / clear up-irradiance {ratio:?}");
        let blue = |e: glam::Vec3| e.z / e.x;
        assert!(blue(e_overcast) < 0.7 * blue(e_clear), "the overcast ambient is about as blue as the clear sky's: {ratio:?}");
        let kept = (clear_up(&lit_overcast) - clear_up(&none)).abs().max_element() / clear_up(&none).max_element();
        assert!(kept < 1e-3, "the clear sky's irradiance changed by {kept}");
        // frames later with no clouds drawn (the effect out of the chain), the sky lighting is clear
        let mut later = Camera::new(60.0, 0.1, 5000.0, w as f32 / h as f32);
        later.set_position(0.0, 2.0, 0.0);
        later.look_at(&Vec3::new(0.0, 100.0, -1.0));
        later.update_view_matrix();
        for _ in 0..3 { later.end_frame(); }
        let gone = lighting_at(&mut sky, &later);
        let drift = none.iter().zip(&gone).map(|(a, b)| (a - b).abs() / a.abs().max(1.0)).fold(0.0, f32::max);
        assert!(drift < 1e-3, "clouds no longer drawn still change the sky lighting by {drift}");
        assert!(overcast.iter().all(|c| c.iter().all(|v| v.is_finite())), "non-finite cloud light");
        let mean: f32 = overcast.iter().map(|c| c[1]).sum::<f32>() / overcast.len() as f32;
        // the background barely shows through a closed deck
        assert!(mean > 500.0 && mean < 20_000.0, "overcast base luminance {mean}");
        eprintln!("overcast base seen from below: {mean:.0} cd/m2 under a 100 000 lux sun");

        // the clouds follow the sky's luminance factor, as the sky they cover does
        sky.params.sky_luminance_factor = Vec3::new(0.25, 0.25, 0.25);
        let scaled = run(&mut sky, 1.0);
        let scaled_mean: f32 = scaled.iter().map(|c| c[1]).sum::<f32>() / scaled.len() as f32;
        let ratio = scaled_mean / mean;
        assert!((ratio - 0.25).abs() < 0.03, "with the sky scaled by 0.25 the clouds scale by {ratio}");
    }

    /// By hand: how much an overcast darkens the sky lighting at twilight, by default with
    /// midsommar-web's settings (sun 2.5 degrees below the horizon, haze 1.7, a layer from 1.2 to
    /// 5.2 km at coverage 0.9 and density 1.2; SUN=, COV=, DEN= change them). Prints the SH's
    /// irradiance and radiance with and without the clouds.
    /// `cargo test -p kansei-core --lib twilight_overcast -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn twilight_overcast_sky_lighting() {
        let Some((device, queue)) = gpu() else { return };
        let (w, h) = (64u32, 32u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let output = texture(GBuffer::COLOR_FORMAT, wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
        let depth_tex = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = depth_tex.create_view(&Default::default());
        {
            let mut e = device.create_command_encoder(&Default::default());
            e.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[], depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }), timestamp_writes: None, occlusion_query_set: None });
            queue.submit([e.finish()]);
        }
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let mut camera = Camera::new(60.0, 0.1, 5000.0, w as f32 / h as f32);
        camera.set_position(0.0, 2.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 100.0, -1.0));
        camera.update_view_matrix();
        let env = |k: &str, d: f32| std::env::var(k).map(|v| v.parse().unwrap()).unwrap_or(d);
        let mut sky = crate::atmosphere::SkyAtmosphere::new(&device, Default::default());
        sky.sun.direction = crate::atmosphere::direction_from_elevation_bearing(env("SUN", -2.5), 140.0);
        sky.sun.illuminance = Vec3::new(100_000.0, 73_000.0, 55_200.0);
        sky.params.mie_scattering_scale = 0.003996 * 1.7;
        let lighting = |sky: &mut crate::atmosphere::SkyAtmosphere| -> Vec<f32> {
            let size = std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64;
            let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            sky.encode(&queue, &mut e, &camera);
            e.copy_buffer_to_buffer(&sky.bindings().sky_lighting, 0, &buf, 0, size);
            queue.submit([e.finish()]);
            buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let v = bytemuck::cast_slice::<u8, f32>(&buf.slice(..).get_mapped_range()).to_vec();
            v
        };
        let eval = |l: &[f32], d: glam::Vec3, band: [f32; 3]| -> glam::Vec3 {
            let c = |i: usize| glam::Vec3::new(l[i * 4], l[i * 4 + 1], l[i * 4 + 2]);
            c(0) * (0.282095 * band[0]) + (c(1) * d.y + c(2) * d.z + c(3) * d.x) * (0.488603 * band[1])
                + (c(4) * (1.092548 * d.x * d.y) + c(5) * (1.092548 * d.y * d.z) + c(6) * (0.315392 * (3.0 * d.z * d.z - 1.0)) + c(7) * (1.092548 * d.x * d.z) + c(8) * (0.546274 * (d.x * d.x - d.y * d.y))) * band[2]
        };
        let none = lighting(&mut sky);
        let mut fx = VolumetricCloudsEffect::new(&sky, VolumetricCloudsOptions {
            layer: CloudLayer { bottom_m: 1200.0, top_m: 5200.0, coverage: env("COV", 0.9), cloud_type: 0.5, extinction: CloudLayer::default().extinction * env("DEN", 1.2), ..Default::default() },
            max_distance_m: 50_000.0,
            ..Default::default()
        });
        for _ in 0..30 {
            let mut encoder = device.create_command_encoder(&Default::default());
            sky.encode(&queue, &mut encoder, &camera);
            fx.render(&device, &queue, &mut encoder, &gbuffer, &input, &depth, &output, &camera, w, h);
            queue.submit([encoder.finish()]);
        }
        let over = lighting(&mut sky);
        let up = glam::Vec3::Y;
        let horizon = glam::Vec3::new(0.0, 0.17, 0.98).normalize();
        let irr = [3.141593, 2.094395, 0.785398];
        eprintln!("sun {} deg, coverage {}, density {}", env("SUN", -2.5), env("COV", 0.9), env("DEN", 1.2));
        eprintln!("irradiance up: clear {:?} overcast {:?} ratio {:?}", eval(&none, up, irr), eval(&over, up, irr), eval(&over, up, irr) / eval(&none, up, irr));
        eprintln!("radiance zenith (low-pass): clear {:?} overcast {:?} ratio {:?}", eval(&none, up, [1.0; 3]), eval(&over, up, [1.0; 3]), eval(&over, up, [1.0; 3]) / eval(&none, up, [1.0; 3]));
        eprintln!("radiance 10 deg up: clear {:?} overcast {:?}", eval(&none, horizon, [1.0; 3]), eval(&over, horizon, [1.0; 3]));
        eprintln!("clear sky up (clouds' ambient) {:?}; sun at camera {:?}", &over[13 * 4..13 * 4 + 3], &over[9 * 4..9 * 4 + 4]);
    }

    /// The clouds' shadow map: a clear sky casts none, an overcast one a deep shadow, broken cloud
    /// patches of both; it lies under the camera, and is off with the sun down or the clouds gone.
    #[test]
    fn the_clouds_shadow_the_scene_from_the_sun() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let (w, h) = (64u32, 32u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let output = texture(GBuffer::COLOR_FORMAT, wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
        let depth = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT).create_view(&Default::default());
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let mut camera = Camera::new(60.0, 0.1, 5000.0, w as f32 / h as f32);
        camera.set_position(300.0, 2.0, -700.0);
        camera.look_at(&Vec3::new(300.0, 2.0, -800.0));
        camera.update_view_matrix();
        let mut sky = crate::atmosphere::SkyAtmosphere::new(&device, Default::default());
        sky.sun.direction = crate::atmosphere::direction_from_elevation_bearing(50.0, 180.0);
        sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);
        // the map (transmittance per texel) and its parameters after a few frames at a coverage
        let run = |sky: &mut crate::atmosphere::SkyAtmosphere, coverage: f32| -> (Vec<f32>, CloudShadowParamsGpu) {
            let mut fx = VolumetricCloudsEffect::new(sky, VolumetricCloudsOptions { layer: CloudLayer { coverage, ..Default::default() }, ..Default::default() });
            for _ in 0..3 {
                let mut e = device.create_command_encoder(&Default::default());
                sky.encode(&queue, &mut e, &camera);
                fx.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
                queue.submit([e.finish()]);
            }
            let n = CLOUD_SHADOW_SIZE;
            let size = std::mem::size_of::<CloudShadowParamsGpu>() as u64;
            let map_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (n * n * 4) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let params_buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let mut e = device.create_command_encoder(&Default::default());
            e.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo { texture: sky.cloud_shadow_texture(), mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                wgpu::TexelCopyBufferInfo { buffer: &map_buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * 4), rows_per_image: Some(n) } },
                wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
            );
            e.copy_buffer_to_buffer(&sky.bindings().cloud_shadow_params, 0, &params_buf, 0, size);
            queue.submit([e.finish()]);
            map_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            params_buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let map = map_buf.slice(..).get_mapped_range().chunks_exact(4).map(|t| t[0] as f32 / 255.0).collect();
            let params = *bytemuck::from_bytes::<CloudShadowParamsGpu>(&params_buf.slice(..).get_mapped_range());
            (map, params)
        };
        let mean = |m: &[f32]| m.iter().sum::<f32>() / m.len() as f32;
        let (clear, params) = run(&mut sky, 0.0);
        assert!(clear.iter().all(|&t| t > 0.99), "a clear sky casts a shadow: min {}", clear.iter().cloned().fold(1.0, f32::min));
        assert!(params.enabled > 0.5, "shadows off with the sun up");
        let texel = params.inv_size.recip() / CLOUD_SHADOW_SIZE as f32;
        assert!((params.center[0] - 300.0).abs() <= texel && (params.center[1] + 700.0).abs() <= texel && (params.plane_y - 2.0).abs() < 1e-3, "{:?} {}", params.center, params.plane_y);
        let (overcast, _) = run(&mut sky, 1.0);
        let (broken, _) = run(&mut sky, 0.5);
        let (lo, hi) = (broken.iter().cloned().fold(1.0, f32::min), broken.iter().cloned().fold(0.0, f32::max));
        eprintln!("sun transmitted: overcast {:.3} mean; broken {:.3} mean, {lo:.3}..{hi:.3}", mean(&overcast), mean(&broken));
        assert!(mean(&overcast) < 0.1, "an overcast lets {} of the sun through", mean(&overcast));
        assert!(lo < 0.3 && hi > 0.9, "broken cloud casts no patches: {lo}..{hi}");

        // the sun down: the shadows are off
        sky.sun.direction = crate::atmosphere::direction_from_elevation_bearing(-3.0, 180.0);
        let (_, down) = run(&mut sky, 1.0);
        assert!(down.enabled < 0.5, "shadows on with the sun down");
        // frames later with no clouds drawn: off
        sky.sun.direction = crate::atmosphere::direction_from_elevation_bearing(50.0, 180.0);
        let _ = run(&mut sky, 1.0);
        let mut later = Camera::new(60.0, 0.1, 5000.0, w as f32 / h as f32);
        for _ in 0..3 { later.end_frame(); }
        let mut e = device.create_command_encoder(&Default::default());
        sky.encode(&queue, &mut e, &later);
        let size = std::mem::size_of::<CloudShadowParamsGpu>() as u64;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        e.copy_buffer_to_buffer(&sky.bindings().cloud_shadow_params, 0, &buf, 0, size);
        queue.submit([e.finish()]);
        buf.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let gone = *bytemuck::from_bytes::<CloudShadowParamsGpu>(&buf.slice(..).get_mapped_range());
        assert!(gone.enabled < 0.5, "clouds no longer drawn still cast shadows");
    }
}
