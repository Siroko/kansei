use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::materials::GradientSkyOptions;
use crate::math::Vec3;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = include_str!("../../shaders/distance_fog.wgsl");

/// What a [`DistanceFogEffect`] fades toward.
#[derive(Debug, Clone, Copy)]
pub enum DistanceFogColor {
    /// One radiance (cd/m², linear rgb).
    Constant([f32; 3]),
    /// The [`Material::gradient_sky`](crate::materials::Material::gradient_sky) dome of `sky`,
    /// centred at `centre` with `radius` (its mesh's, unscaled): each pixel fades toward what the
    /// dome shows behind it, and the fade ends where its ray meets the dome, so a floor reaching
    /// the dome meets it with no edge wherever the camera is inside it.
    GradientSky { sky: GradientSkyOptions, centre: Vec3, radius: f32 },
}

/// Distance fog: each surface fades toward a colour by its distance from the camera, from nothing
/// at `start` to `max_opacity` of it at `end` (metres, a smoothstep between). Cheap (one read of
/// colour and depth a pixel), for the far edge of a scene: a floor meeting a backdrop, a set that
/// should end in haze rather than a line. The sky (depth 1) is left as it is. Put it before the
/// volumetric fog, which then lies over it.
pub struct DistanceFogEffect {
    pub start: f32,
    pub end: f32,
    pub color: DistanceFogColor,
    pub max_opacity: f32,
    /// Measure each surface's distance from this point instead of the camera: a set that stays
    /// clear round its centre wherever the camera goes, its floor fading out round it (a camera's
    /// fade crowds a low view's floor into a few rows under the horizon, and hazes the set from
    /// afar).
    pub from_point: Option<Vec3>,
    gpu: Option<Gpu>,
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DistanceFogParamsGpu {
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    start: f32,
    zenith: [f32; 4],
    horizon: [f32; 4],
    ground: [f32; 4],
    centre: [f32; 3],
    radius: f32,
    end: f32,
    max_opacity: f32,
    use_point: f32,
    _pad: f32,
    point: [f32; 4],
}

impl DistanceFogEffect {
    pub fn new(start: f32, end: f32, color: DistanceFogColor) -> Self {
        Self { start, end, color, max_opacity: 1.0, from_point: None, gpu: None }
    }

    /// How much of the fog colour a surface at `point` takes, seen from `eye` (world space), as
    /// the shader computes it.
    pub fn amount(&self, eye: Vec3, point: Vec3) -> f32 {
        let (o, p) = (eye.to_glam(), point.to_glam());
        let dir = (p - o) / o.distance(p).max(1e-6);
        let from = self.from_point.map_or(o, |f| f.to_glam());
        let dist = from.distance(p);
        let mut end = self.end;
        if let DistanceFogColor::GradientSky { centre, radius, .. } = self.color {
            if let Some(t) = dome_exit(o - centre.to_glam(), dir, radius) {
                end = end.min(from.distance(o + dir * t));
            }
        }
        let start = self.start.min(end);
        let t = if end > start {
            let x = ((dist - start) / (end - start)).clamp(0.0, 1.0);
            x * x * (3.0 - 2.0 * x)
        } else {
            (dist >= end) as u32 as f32
        };
        t * self.max_opacity
    }

    fn params(&self, camera: &Camera) -> DistanceFogParamsGpu {
        let view_proj = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let eye = camera.inverse_view_matrix.to_glam().w_axis;
        let (sky, centre, radius) = match self.color {
            DistanceFogColor::Constant(c) => (GradientSkyOptions { zenith: c, horizon: c, ground: c, curve: 1.0 }, Vec3::ZERO, 0.0),
            DistanceFogColor::GradientSky { sky, centre, radius } => (sky, centre, radius.max(0.0)),
        };
        let rgba = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
        DistanceFogParamsGpu {
            inv_view_proj: view_proj.inverse().to_cols_array(),
            camera_pos: [eye.x, eye.y, eye.z],
            start: self.start.max(0.0),
            zenith: rgba(sky.zenith, sky.curve),
            horizon: rgba(sky.horizon, 0.0),
            ground: rgba(sky.ground, 0.0),
            centre: [centre.x, centre.y, centre.z],
            radius,
            end: self.end.max(0.0),
            max_opacity: self.max_opacity.clamp(0.0, 1.0),
            use_point: self.from_point.is_some() as u32 as f32,
            _pad: 0.0,
            point: self.from_point.map_or([0.0; 4], |f| [f.x, f.y, f.z, 0.0]),
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DistanceFog/BGL"),
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
                entry(3, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("DistanceFog"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("DistanceFog"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DistanceFog"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DistanceFog/Params"),
            size: std::mem::size_of::<DistanceFogParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some(Gpu { pipeline, bgl, params });
    }
}

/// Where a ray from `oc` (relative to the dome's centre) along `dir` leaves a dome of `radius`.
fn dome_exit(oc: glam::Vec3, dir: glam::Vec3, radius: f32) -> Option<f32> {
    if radius <= 0.0 {
        return None;
    }
    let b = oc.dot(dir);
    let h = b * b - (oc.dot(oc) - radius * radius);
    (h >= 0.0).then(|| -b + h.sqrt()).filter(|t| *t > 0.0)
}

impl PostProcessingEffect for DistanceFogEffect {
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
        let params = self.params(camera);
        let gpu = self.gpu.as_ref().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DistanceFog/BG"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 3, resource: gpu.params.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DistanceFog"), timestamp_writes: crate::profiling::gpu_pass("DistanceFog").as_ref().map(crate::profiling::PassStamp::compute) });
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
    fn shader_validates_and_the_params_layout_matches() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "DistanceFogParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<DistanceFogParamsGpu>());
    }

    /// Nothing before `start`, all of it from `end`, eased between; inside a dome, all of it where
    /// the ray meets the dome even short of `end`.
    #[test]
    fn the_fade_runs_from_start_to_end_and_ends_at_the_dome() {
        let eye = Vec3::new(0.0, 1.5, 0.0);
        let at = |d: f32| Vec3::new(0.0, 1.5, -d);
        let mut fog = DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::Constant([1.0; 3]));
        assert_eq!(fog.amount(eye, at(5.0)), 0.0);
        assert!((fog.amount(eye, at(20.0)) - 0.5).abs() < 1e-6);
        assert_eq!(fog.amount(eye, at(30.0)), 1.0);
        fog.max_opacity = 0.8;
        assert!((fog.amount(eye, at(40.0)) - 0.8).abs() < 1e-6);
        // a dome of 25 m round the origin: the fade ends at it
        fog = DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::GradientSky { sky: GradientSkyOptions::default(), centre: Vec3::ZERO, radius: 25.0 });
        let dome = (25.0f32 * 25.0 - 1.5 * 1.5).sqrt();
        assert!(fog.amount(eye, at(dome)) > 0.9999);
        assert!(fog.amount(eye, at(15.0)) > DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::Constant([1.0; 3])).amount(eye, at(15.0)));
        // a start past the dome: a step at the dome
        fog.start = 40.0;
        assert_eq!(fog.amount(eye, at(20.0)), 0.0);
        assert_eq!(fog.amount(eye, at(dome)), 1.0);
        // from a point: what is near it stays clear however far the camera is, and the fade
        // still ends at the dome
        fog.start = 10.0;
        fog.from_point = Some(Vec3::ZERO);
        let far_eye = Vec3::new(0.0, 1.5, 20.0);
        assert_eq!(fog.amount(far_eye, at(5.0)), 0.0);
        assert!(DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::Constant([1.0; 3])).amount(far_eye, at(5.0)) > 0.5);
        assert!((fog.amount(eye, Vec3::new(0.0, 0.0, -17.5)) - 0.5).abs() < 1e-6);
        assert!(fog.amount(far_eye, at(dome)) > 0.9999);
    }

    /// On the GPU: a black wall straight ahead, at 2 m (before the fog), 20 m (halfway) and 40 m
    /// (past a 30 m dome centred on the camera, so all fog: the dome's horizon colour).
    #[test]
    fn the_shader_fades_as_the_cpu_does() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let (w, h) = (9u32, 9u32);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let output_tex = texture(wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let output = output_tex.create_view(&Default::default());
        let depth_tex = texture(GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth = depth_tex.create_view(&Default::default());
        let gbuffer = GBuffer::new(&device, w, h, 1);
        let mut camera = Camera::new(40.0, 0.1, 100.0, 1.0);
        camera.set_position(0.0, 0.0, 0.0);
        camera.look_at(&Vec3::new(0.0, 0.0, -1.0));
        camera.update_view_matrix();
        camera.update_projection_matrix();
        let sky = GradientSkyOptions { zenith: [0.0, 0.0, 4.0], horizon: [2.0, 1.0, 0.5], ground: [0.0, 3.0, 0.0], curve: 1.0 };
        let mut fog = DistanceFogEffect::new(10.0, 30.0, DistanceFogColor::GradientSky { sky, centre: Vec3::ZERO, radius: 30.0 });
        let row = (w * 8).div_ceil(256) * 256;
        let half = |b: u16| -> f32 {
            let (s, e, m) = ((b >> 15) as i32, ((b >> 10) & 0x1f) as i32, (b & 0x3ff) as f32);
            (if s == 1 { -1.0 } else { 1.0 }) * if e == 0 { m / 1024.0 * 2f32.powi(-14) } else { (1.0 + m / 1024.0) * 2f32.powi(e - 15) }
        };
        for (distance, want) in [(2.0f32, 0.0f32), (20.0, 0.5), (40.0, 1.0)] {
            let clip = camera.projection_matrix.to_glam() * glam::Vec4::new(0.0, 0.0, -distance, 1.0);
            let mut e = device.create_command_encoder(&Default::default());
            e.begin_render_pass(&wgpu::RenderPassDescriptor { label: None, color_attachments: &[], depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(clip.z / clip.w), store: wgpu::StoreOp::Store }), stencil_ops: None }), timestamp_writes: None, occlusion_query_set: None });
            fog.render(&device, &queue, &mut e, &gbuffer, &input, &depth, &output, &camera, w, h);
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
            // the centre pixel looks along -z: the dome's horizon
            let px = &bytemuck::cast_slice::<u8, u16>(&data[(4 * row) as usize..])[16..19];
            let got = [half(px[0]), half(px[1]), half(px[2])];
            let cpu = fog.amount(Vec3::ZERO, Vec3::new(0.0, 0.0, -distance));
            assert!((cpu - want).abs() < 1e-3, "{distance} m: cpu {cpu}, want {want}");
            for k in 0..3 {
                assert!((got[k] - sky.horizon[k] * want).abs() < 0.01, "{distance} m: {got:?}, want {want} of {:?}", sky.horizon);
            }
        }
        // past the dome the fog has its full colour even before `end`
        fog.end = 60.0;
        assert_eq!(fog.amount(Vec3::ZERO, Vec3::new(0.0, 0.0, -40.0)), 1.0);
    }
}
