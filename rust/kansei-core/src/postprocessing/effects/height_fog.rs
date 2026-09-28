use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::math::Vec3;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = concat!(
    include_str!("../../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("../../shaders/height_fog.wgsl"),
);

/// One exponential layer: extinction `density` per metre at `height`, falling by e every
/// 1 / `height_falloff` metres above it (and rising below it).
#[derive(Debug, Clone, Copy, Default)]
pub struct HeightFogLayer {
    pub density: f32,
    pub height_falloff: f32,
    pub height: f32,
}

impl HeightFogLayer {
    /// From Unreal's `FogDensity`, `FogHeightFalloff` and the fog actor's height in metres, so the
    /// fog is as opaque as Unreal draws it. Unreal takes both coefficients per 1000 cm and in base
    /// 2 (`SceneCore.cpp`: `/ 1000`), and its line integral, `(1 - 2^-F) / F` times the density
    /// and the ray's length (`HeightFogCommon.ush`), is ln 2 times the base-2 integral; its
    /// transmittance is `2^-integral`. Per metre in base e: the falloff is x 0.1 x ln 2, the
    /// density x 0.1 x (ln 2)^2.
    pub fn from_unreal(fog_density: f32, fog_height_falloff: f32, height_m: f32) -> Self {
        let ln2 = std::f32::consts::LN_2;
        Self { density: fog_density * 0.1 * ln2 * ln2, height_falloff: fog_height_falloff * 0.1 * ln2, height: height_m }
    }

    /// Optical depth along `origin + dir * t`, t in [t0, t1] (dir unit), as the shader computes it.
    pub fn optical_depth(&self, origin: glam::Vec3, dir: glam::Vec3, t0: f32, t1: f32) -> f32 {
        if self.density <= 0.0 || t1 <= t0 {
            return 0.0;
        }
        let start = self.density * (-self.height_falloff * (origin.y + dir.y * t0 - self.height)).min(80.0).exp();
        let k = self.height_falloff * dir.y;
        let len = t1 - t0;
        let x = k * len;
        let shape = if x.abs() > 1e-4 { (1.0 - (-x).min(80.0).exp()) / k } else { len };
        start * shape
    }
}

/// Analytic exponential height fog, after Unreal's `ExponentialHeightFog`: per pixel, the line
/// integral of one or two exponential layers from the camera (from `start_distance`) to the
/// surface, or to `sky_distance` for the sky, fading the scene toward a fog colour. The colour is
/// `inscattering` plus the sky's distant light when a sky is bound (`set_sky_lighting`, times
/// `sky_ambient_scale`), with a lobe of `directional_inscattering` toward the sun.
///
/// It is the far fog of a scene: start it where the volumetric fog's froxels end (their `far`)
/// and put it after the `AtmosphereEffect` and before the `VolumetricFogEffect` in the chain, so
/// near fog lies in front of far fog, which lies in front of the aerial perspective and the sky.
pub struct HeightFogEffect {
    /// The second layer is off while its density is 0.
    pub layers: [HeightFogLayer; 2],
    /// Fog luminance at full opacity (Unreal's fog inscattering luminance).
    pub inscattering: Vec3,
    /// Scales the sky's light on the fog once a sky is bound.
    pub sky_ambient_scale: f32,
    /// Luminance of the lobe toward the light; zero switches it off.
    pub directional_inscattering: Vec3,
    pub directional_exponent: f32,
    pub directional_start_distance: f32,
    /// Toward the light, for the lobe when no sky is bound (the sky's sun otherwise).
    pub light_direction: Vec3,
    /// Metres before which there is no fog.
    pub start_distance: f32,
    /// Nothing farther than this gets fog (0: no cutoff).
    pub cutoff_distance: f32,
    pub max_opacity: f32,
    /// Distance the sky is fogged as, metres.
    pub sky_distance: f32,
    sky_lighting: Option<wgpu::Buffer>,
    gpu: Option<Gpu>,
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    no_sky: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct HeightFogParamsGpu {
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    start_distance: f32,
    inscattering: [f32; 3],
    cutoff_distance: f32,
    directional: [f32; 3],
    directional_exponent: f32,
    light_direction: [f32; 3],
    directional_start: f32,
    layer0: [f32; 4],
    layer1: [f32; 4],
    max_opacity: f32,
    sky_ambient_scale: f32,
    sky_distance: f32,
    has_sky_lighting: u32,
}

impl Default for HeightFogEffect {
    fn default() -> Self {
        Self::new(HeightFogLayer { density: 0.002, height_falloff: 0.01, height: 0.0 })
    }
}

impl HeightFogEffect {
    pub fn new(layer: HeightFogLayer) -> Self {
        Self {
            layers: [layer, HeightFogLayer::default()],
            inscattering: Vec3::ZERO,
            sky_ambient_scale: 1.0,
            directional_inscattering: Vec3::ZERO,
            directional_exponent: 4.0,
            directional_start_distance: 0.0,
            light_direction: Vec3::UP,
            start_distance: 0.0,
            cutoff_distance: 0.0,
            max_opacity: 1.0,
            sky_distance: 100_000.0,
            sky_lighting: None,
            gpu: None,
        }
    }

    /// Colour the fog with a sky (`SkyAtmosphere::bindings().sky_lighting`): its distant light (the
    /// mean radiance all round from 6 km up, `SkyLighting.distantSkyLight`, as Unreal's fog adds),
    /// and its sun for the directional lobe (off while the sun is below the horizon).
    pub fn set_sky_lighting(&mut self, sky_lighting: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky_lighting.cloned();
    }

    /// Transmittance of the fog between `origin` and `point` (world space), as the shader.
    pub fn transmittance(&self, origin: Vec3, point: Vec3) -> f32 {
        let (o, p) = (origin.to_glam(), point.to_glam());
        let dist = o.distance(p);
        if dist <= self.start_distance || (self.cutoff_distance > 0.0 && dist > self.cutoff_distance) {
            return 1.0;
        }
        let dir = (p - o) / dist;
        let depth: f32 = self.layers.iter().map(|l| l.optical_depth(o, dir, self.start_distance, dist)).sum();
        1.0 - (1.0 - (-depth).exp()).min(self.max_opacity)
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("HeightFog/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(2, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: GBuffer::COLOR_FORMAT,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(3, uniform),
                entry(4, uniform),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("HeightFog"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("HeightFog"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("HeightFog"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let sky_bytes = std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64;
        self.gpu = Some(Gpu {
            pipeline,
            bgl,
            params: buffer("HeightFog/Params", std::mem::size_of::<HeightFogParamsGpu>() as u64),
            no_sky: buffer("HeightFog/NoSkyLighting", sky_bytes),
        });
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        WGSL
    }
}

impl PostProcessingEffect for HeightFogEffect {
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
        let gpu = self.gpu.as_ref().unwrap();
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let eye = camera.inverse_view_matrix.to_glam().w_axis;
        let rgb = |v: Vec3| [v.x, v.y, v.z];
        let layer = |l: &HeightFogLayer| [l.density.max(0.0), l.height_falloff, l.height, 0.0];
        let params = HeightFogParamsGpu {
            inv_view_proj: view_proj.inverse().to_cols_array(),
            camera_pos: [eye.x, eye.y, eye.z],
            start_distance: self.start_distance.max(0.0),
            inscattering: rgb(self.inscattering),
            cutoff_distance: self.cutoff_distance.max(0.0),
            directional: rgb(self.directional_inscattering),
            directional_exponent: self.directional_exponent.max(0.0),
            light_direction: rgb(self.light_direction),
            directional_start: self.directional_start_distance.max(0.0),
            layer0: layer(&self.layers[0]),
            layer1: layer(&self.layers[1]),
            max_opacity: self.max_opacity.clamp(0.0, 1.0),
            sky_ambient_scale: self.sky_ambient_scale,
            sky_distance: self.sky_distance.max(0.0),
            has_sky_lighting: self.sky_lighting.is_some() as u32,
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.no_sky);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HeightFog/BG"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: sky.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("HeightFog"), ..Default::default() });
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

    /// `from_unreal` fogs a ray as Unreal's shader does: its transmittance is `2^-I`, with
    /// I = density x 2^(-falloff (z0 - height)) x (1 - 2^-F) / F x length, F = falloff x the ray's
    /// rise, all in centimetres, the coefficients / 1000 (HeightFogCommon.ush, SceneCore.cpp).
    #[test]
    fn from_unreal_matches_unreal_s_line_integral() {
        let (fog_density, falloff, height) = (0.03f32, 0.1f32, -20.0f32);
        let layer = HeightFogLayer::from_unreal(fog_density, falloff, height);
        let unreal = |origin: glam::Vec3, dir: glam::Vec3, len_m: f32| -> f32 {
            let (rho, k) = (fog_density as f64 / 1000.0, falloff as f64 / 1000.0);
            let origin_terms = rho * 2f64.powf(-k * (origin.y as f64 - height as f64) * 100.0);
            let f = k * dir.y as f64 * len_m as f64 * 100.0;
            let integral = if f.abs() > 1e-9 { (1.0 - 2f64.powf(-f)) / f } else { std::f64::consts::LN_2 };
            2f64.powf(-origin_terms * integral * len_m as f64 * 100.0) as f32
        };
        for (origin, dir, len) in [
            (glam::Vec3::new(0.0, 6.0, 0.0), glam::Vec3::Y, 5000.0),
            (glam::Vec3::new(0.0, 6.0, 0.0), glam::Vec3::new(0.0, 0.1, 1.0).normalize(), 3000.0),
            (glam::Vec3::new(0.0, 200.0, 0.0), glam::Vec3::new(0.3, -0.2, 0.9).normalize(), 800.0),
            (glam::Vec3::new(0.0, 1.0, 0.0), glam::Vec3::X, 300.0),
        ] {
            let kansei = (-layer.optical_depth(origin, dir, 0.0, len)).exp();
            let want = unreal(origin, dir, len);
            assert!((kansei - want).abs() < 1e-3 * want.max(1e-3), "{origin} {dir} {len} m: kansei {kansei}, Unreal {want}");
        }
        // the film's fog (0.03, 0.1) is 1.44e-3 per metre at its height
        assert!((HeightFogLayer::from_unreal(0.03, 0.1, 0.0).density - 1.4414e-3).abs() < 1e-6);
    }

    #[test]
    fn shader_validates_and_the_params_layout_matches() {
        let code = HeightFogEffect::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "HeightFogParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<HeightFogParamsGpu>());
    }

    #[test]
    fn closed_form_optical_depth_matches_numerical_integration() {
        let layer = HeightFogLayer::from_unreal(0.03, 0.1, 2.0);
        let origin = glam::Vec3::new(0.0, 12.0, 0.0);
        for dir in [
            glam::Vec3::new(1.0, 0.0, 0.0),
            glam::Vec3::new(1.0, 0.05, 0.0).normalize(),
            glam::Vec3::new(1.0, -0.01, 0.2).normalize(),
            glam::Vec3::new(0.0, 1.0, 0.0),
        ] {
            let (t0, t1) = (120.0, 3000.0);
            let n = 200_000;
            let dt = (t1 - t0) / n as f32;
            let numeric: f64 = (0..n)
                .map(|i| {
                    let h = origin.y + dir.y * (t0 + (i as f32 + 0.5) * dt);
                    (layer.density * (-layer.height_falloff * (h - layer.height)).exp() * dt) as f64
                })
                .sum();
            let analytic = layer.optical_depth(origin, dir, t0, t1) as f64;
            assert!((analytic - numeric).abs() <= 1e-3 * numeric.max(1e-6), "{dir}: {analytic} vs {numeric}");
        }
    }

    #[test]
    fn fog_starts_and_cuts_off_where_asked() {
        let mut fog = HeightFogEffect::new(HeightFogLayer::from_unreal(0.03, 0.1, 0.0));
        fog.start_distance = 120.0;
        let eye = Vec3::new(0.0, 2.0, 0.0);
        assert_eq!(fog.transmittance(eye, Vec3::new(100.0, 2.0, 0.0)), 1.0);
        let far = fog.transmittance(eye, Vec3::new(2000.0, 2.0, 0.0));
        assert!(far < 0.1 && far > 0.0, "{far}");
        // up in the sky the layer is thin
        assert!(fog.transmittance(eye, Vec3::new(0.0, 2000.0, 0.0)) > 0.5);
        fog.cutoff_distance = 1500.0;
        assert_eq!(fog.transmittance(eye, Vec3::new(2000.0, 2.0, 0.0)), 1.0);
        fog.cutoff_distance = 0.0;
        fog.max_opacity = 0.7;
        assert!((fog.transmittance(eye, Vec3::new(20000.0, 2.0, 0.0)) - 0.3).abs() < 1e-5);
    }
}
