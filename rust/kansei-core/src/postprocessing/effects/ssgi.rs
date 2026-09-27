use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const COMMON_WGSL: &str = include_str!("../../shaders/ssgi_common.wgsl");
const NORMAL_WGSL: &str = include_str!("../../shaders/ssgi_normal.wgsl");
const TRACE_WGSL: &str = include_str!("../../shaders/ssgi_trace.wgsl");
const TEMPORAL_WGSL: &str = include_str!("../../shaders/ssgi_temporal.wgsl");
const COMPOSITE_WGSL: &str = include_str!("../../shaders/ssgi_composite.wgsl");

fn trace_source() -> String {
    [COMMON_WGSL, NORMAL_WGSL, TRACE_WGSL].concat()
}

fn temporal_source() -> String {
    [COMMON_WGSL, TEMPORAL_WGSL].concat()
}

fn composite_source() -> String {
    [COMMON_WGSL, NORMAL_WGSL, crate::atmosphere::SKY_LIGHTING_WGSL, COMPOSITE_WGSL].concat()
}

/// How much work the global illumination does per frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GiQuality {
    /// A quarter of the resolution each way, 2 slices of 6 steps a side.
    Low,
    /// Half resolution, 2 slices of 8 steps.
    #[default]
    Medium,
    /// Half resolution, 4 slices of 12 steps.
    High,
    /// Full resolution, 4 slices of 16 steps.
    Ultra,
}

impl GiQuality {
    /// (resolution scale, slices, steps per side)
    fn settings(self) -> (f32, u32, u32) {
        match self {
            GiQuality::Low => (0.25, 2, 6),
            GiQuality::Medium => (0.5, 2, 8),
            GiQuality::High => (0.5, 4, 12),
            GiQuality::Ultra => (1.0, 4, 16),
        }
    }
}

pub struct ScreenSpaceGIOptions {
    pub quality: GiQuality,
    /// Metres searched around each point for surfaces that bounce light onto it.
    pub radius_m: f32,
    /// Metres assumed behind each depth sample (thin things let light past them).
    pub thickness_m: f32,
    /// Scale of the bounce (1 is physical).
    pub intensity: f32,
    /// How much of the sky's ambient light is taken out where the sky is hidden (needs
    /// `set_sky_lighting`, and materials lit by the sky's SH as `SKY_LIGHTING_WGSL` does).
    pub ambient_occlusion: f32,
    /// Weight of each new frame in the accumulated result.
    pub temporal_blend: f32,
}

impl Default for ScreenSpaceGIOptions {
    fn default() -> Self {
        Self { quality: GiQuality::Medium, radius_m: 3.0, thickness_m: 0.5, intensity: 1.0, ambient_occlusion: 1.0, temporal_blend: 0.1 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SsgiParamsGpu {
    proj: [f32; 16],
    inv_proj: [f32; 16],
    view: [f32; 16],
    inv_view: [f32; 16],
    prev_view_proj: [f32; 16],
    full_size: [f32; 2],
    trace_size: [f32; 2],
    radius: f32,
    thickness: f32,
    intensity: f32,
    ao_strength: f32,
    slices: u32,
    steps: u32,
    frame: u32,
    history_valid: u32,
    max_radius_px: f32,
    blend: f32,
    has_sky: u32,
    _pad: f32,
}

struct Targets {
    width: u32,
    height: u32,
    trace: wgpu::TextureView,
    history: [wgpu::TextureView; 2],
}

struct Gpu {
    params: wgpu::Buffer,
    trace: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    temporal: wgpu::ComputePipeline,
    temporal_bgl: wgpu::BindGroupLayout,
    composite: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    no_sky: wgpu::Buffer,
    targets: Option<Targets>,
}

/// Screen-space global illumination: one bounce of the light on screen onto every diffuse
/// surface, and ambient occlusion of the sky's light, from the GBuffer (after Therrien et al.
/// 2023, visibility bitmasks). It works on any scene, with no preprocessing: foliage, instanced
/// geometry and alpha-tested cards included. Light from surfaces off screen or hidden from the
/// camera is not seen.
///
/// Opt-in: add it first in the chain (before the atmosphere and the fogs, so the bounce lies on
/// the surfaces under the aerial perspective). `enabled = false` skips it at no cost, and
/// `quality` trades resolution and samples for time. It adds light only where a material writes
/// the GBuffer's albedo (and uses its normal when written, rebuilding one from depth otherwise).
pub struct ScreenSpaceGIEffect {
    pub enabled: bool,
    pub quality: GiQuality,
    pub radius_m: f32,
    pub thickness_m: f32,
    pub intensity: f32,
    pub ambient_occlusion: f32,
    pub temporal_blend: f32,
    sky_lighting: Option<wgpu::Buffer>,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
    frame: u32,
    gpu: Option<Gpu>,
}

impl ScreenSpaceGIEffect {
    pub fn new(options: ScreenSpaceGIOptions) -> Self {
        Self {
            enabled: true,
            quality: options.quality,
            radius_m: options.radius_m,
            thickness_m: options.thickness_m,
            intensity: options.intensity,
            ambient_occlusion: options.ambient_occlusion,
            temporal_blend: options.temporal_blend,
            sky_lighting: None,
            prev_view_proj: None,
            last_camera_frame: None,
            frame: 0,
            gpu: None,
        }
    }

    /// The sky's lighting (`SkyAtmosphereBindings::sky_lighting`), for ambient occlusion.
    pub fn set_sky_lighting(&mut self, sky_lighting: Option<&wgpu::Buffer>) {
        self.sky_lighting = sky_lighting.cloned();
    }

    /// Drop the accumulated frames; call on camera cuts.
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        vec![("ssgi_trace", trace_source()), ("ssgi_temporal", temporal_source()), ("ssgi_composite", composite_source())]
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = |binding, filterable| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let depth = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba16Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let sampler_entry = |binding| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None };
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let trace_bgl = bgl("SSGI/TraceBGL", &[uniform(0), texture(1, false), depth(2), texture(3, false), storage(4)]);
        let temporal_bgl = bgl("SSGI/TemporalBGL", &[uniform(0), texture(1, false), texture(2, true), depth(3), storage(4), sampler_entry(5)]);
        let composite_bgl = bgl("SSGI/CompositeBGL", &[uniform(0), texture(1, false), depth(2), texture(3, false), texture(4, false), texture(5, false), uniform(6), storage(7)]);
        let pipeline = |label: &str, code: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[layout], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        self.gpu = Some(Gpu {
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSGI/Params"),
                size: std::mem::size_of::<SsgiParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            trace: pipeline("SSGI/Trace", &trace_source(), &trace_bgl),
            trace_bgl,
            temporal: pipeline("SSGI/Temporal", &temporal_source(), &temporal_bgl),
            temporal_bgl,
            composite: pipeline("SSGI/Composite", &composite_source(), &composite_bgl),
            composite_bgl,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("SSGI/Linear"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                ..Default::default()
            }),
            no_sky: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSGI/NoSky"),
                size: std::mem::size_of::<crate::atmosphere::params::SkyLightingGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            }),
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (scale, _, _) = self.quality.settings();
        let (w, h) = (((width as f32 * scale).ceil() as u32).max(1), ((height as f32 * scale).ceil() as u32).max(1));
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == w && t.height == h) {
            return;
        }
        let target = |label: &str| {
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
        gpu.targets = Some(Targets { width: w, height: h, trace: target("SSGI/Trace"), history: [target("SSGI/HistoryA"), target("SSGI/HistoryB")] });
        self.prev_view_proj = None;
    }
}

impl PostProcessingEffect for ScreenSpaceGIEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
    }

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
            self.init_gpu(device);
        }
        self.ensure_targets(device, width, height);
        // a gap in the camera's frames (the effect was off, or a cut) invalidates the history
        let camera_frame = camera.frame();
        if self.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
        }
        self.last_camera_frame = Some(camera_frame);
        let gpu = self.gpu.as_ref().unwrap();
        let t = gpu.targets.as_ref().unwrap();
        let (_, slices, steps) = self.quality.settings();
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let view_proj = proj * view;
        let params = SsgiParamsGpu {
            proj: proj.to_cols_array(),
            inv_proj: proj.inverse().to_cols_array(),
            view: view.to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            full_size: [width as f32, height as f32],
            trace_size: [t.width as f32, t.height as f32],
            radius: self.radius_m.max(0.01),
            thickness: self.thickness_m.max(0.0),
            intensity: self.intensity.max(0.0),
            ao_strength: self.ambient_occlusion.clamp(0.0, 1.0),
            slices,
            steps,
            frame: self.frame,
            history_valid: self.prev_view_proj.is_some() as u32,
            max_radius_px: height as f32 * 0.25,
            blend: self.temporal_blend.clamp(0.01, 1.0),
            has_sky: self.sky_lighting.is_some() as u32,
            _pad: 0.0,
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);

        let tex = wgpu::BindingResource::TextureView;
        let group = |label: &str, layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>| {
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        let p = || gpu.params.as_entire_binding();
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.no_sky);
        let trace = group("SSGI/TraceBG", &gpu.trace_bgl, vec![p(), tex(input), tex(depth), tex(&gbuffer.normal_view), tex(&t.trace)]);
        let temporal = group(
            "SSGI/TemporalBG",
            &gpu.temporal_bgl,
            vec![p(), tex(&t.trace), tex(&t.history[1 - current]), tex(depth), tex(&t.history[current]), wgpu::BindingResource::Sampler(&gpu.sampler)],
        );
        let composite = group(
            "SSGI/CompositeBG",
            &gpu.composite_bgl,
            vec![p(), tex(input), tex(depth), tex(&t.history[current]), tex(&gbuffer.albedo_view), tex(&gbuffer.normal_view), sky.as_entire_binding(), tex(output)],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SSGI"), ..Default::default() });
        pass.set_pipeline(&gpu.trace);
        pass.set_bind_group(0, &trace, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        pass.set_pipeline(&gpu.temporal);
        pass.set_bind_group(0, &temporal, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        pass.set_pipeline(&gpu.composite);
        pass.set_bind_group(0, &composite, &[]);
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

    /// The SSGI shaders parse and validate with naga, and `SsgiParams` matches its Rust layout.
    #[test]
    fn shaders_validate_and_the_params_layout_matches() {
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in ScreenSpaceGIEffect::shader_sources() {
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
        assert_eq!(sizes["SsgiParams"], std::mem::size_of::<SsgiParamsGpu>());
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
    }

    const W: u32 = 384;
    const H: u32 = 256;

    /// Renders a scene of up to two planes into the GBuffer's depth (a fullscreen pass writing
    /// frag_depth), fills colour and albedo per pixel, runs the effect for a few frames, and
    /// returns the output.
    fn run(device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera, wall_z: f32, color: impl Fn(glam::Vec3) -> [f32; 3], albedo: f32, fx: &mut ScreenSpaceGIEffect) -> Vec<[f32; 4]> {
        let gbuffer = GBuffer::new(device, W, H, 1);
        let inv_vp = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
        let vp = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        // the scene per pixel: the floor y = 0 and a wall z = wall_z, whichever the ray meets first
        let mut depth = vec![1.0f32; (W * H) as usize];
        let mut rgba = vec![0.0f32; (W * H * 4) as usize];
        for y in 0..H {
            for x in 0..W {
                let ndc = glam::Vec2::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0);
                let far = inv_vp * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
                let dir = (far.truncate() / far.w - eye).normalize();
                let mut t = f32::INFINITY;
                if dir.y < 0.0 { t = t.min(-eye.y / dir.y); }
                if dir.z < 0.0 { t = t.min((wall_z - eye.z) / dir.z); }
                if t.is_finite() {
                    let p = eye + dir * t;
                    let clip = vp * p.extend(1.0);
                    depth[(y * W + x) as usize] = clip.z / clip.w;
                    let c = color(p);
                    rgba[((y * W + x) * 4) as usize..((y * W + x) * 4 + 3) as usize].copy_from_slice(&c);
                }
            }
        }
        // depth and albedo through a fullscreen pass that writes them from storage buffers
        let a = albedo;
        let albedo_px: Vec<f32> = depth.iter().map(|&d| if d < 1.0 { a } else { 0.0 }).collect();
        let buffer = |data: &[f32]| { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE }) };
        let (depth_buf, albedo_buf) = (buffer(&depth), buffer(&albedo_px));
        let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
            @group(0) @binding(1) var<storage, read> a : array<f32>;\n\
            @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
            struct O {{ @builtin(frag_depth) depth : f32, @location(0) albedo : vec4f }};\n\
            @fragment fn fs(@builtin(position) pos : vec4f) -> O {{ var o : O; let i = u32(pos.y) * {W}u + u32(pos.x); o.depth = d[i]; o.albedo = vec4f(vec3f(a[i]), 1.0); return o; }}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: GBuffer::ALBEDO_FORMAT, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: GBuffer::DEPTH_FORMAT, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: albedo_buf.as_entire_binding() }],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &gbuffer.albedo_view,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &gbuffer.depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        let texture = |format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 }, mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let input = texture(wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let extent = wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 };
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &input, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&rgba),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(W * 16), rows_per_image: Some(H) },
            extent,
        );
        let output = texture(GBuffer::COLOR_FORMAT, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let (input_view, output_view) = (input.create_view(&Default::default()), output.create_view(&Default::default()));
        for _ in 0..24 {
            let mut encoder = device.create_command_encoder(&Default::default());
            fx.render(device, queue, &mut encoder, &gbuffer, &input_view, &gbuffer.depth_view, &output_view, camera, W, H);
            queue.submit([encoder.finish()]);
        }
        let row = (W * 8).div_ceil(256) * 256;
        let buf = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut e = device.create_command_encoder(&Default::default());
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &output, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &buf, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
            extent,
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
        (0..H * W).map(|i| { let o = ((i / W) * row + (i % W) * 8) as usize; [half(o), half(o + 2), half(o + 4), half(o + 6)] }).collect()
    }

    fn camera() -> Camera {
        let mut camera = Camera::new(60.0, 0.1, 100.0, W as f32 / H as f32);
        camera.set_position(0.0, 1.5, 4.0);
        camera.look_at(&crate::math::Vec3::new(0.0, 0.8, -2.0));
        camera.update_view_matrix();
        camera
    }

    /// Without an albedo in the GBuffer (a material that writes none) the image is untouched;
    /// an open floor with no wall near gets no bounce.
    #[test]
    fn surfaces_without_albedo_or_occluders_are_left_alone() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let camera = camera();
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, ..Default::default() });
        let out = run(&device, &queue, &camera, -1000.0, |_| [0.5, 0.5, 0.5], 0.0, &mut fx);
        let worst = out.iter().map(|c| (c[0] - 0.5).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "changed without albedo: {worst}");
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, ..Default::default() });
        let out = run(&device, &queue, &camera, -1000.0, |_| [0.5, 0.5, 0.5], 0.5, &mut fx);
        let worst = out.iter().map(|c| (c[0] - 0.5).abs()).fold(0.0, f32::max);
        assert!(worst < 0.01, "an open floor changed by {worst}");
    }

    /// A black floor meets a white wall: near the corner the floor receives about half the wall's
    /// light (the view factor of a wall from the floor is 1/2), times its albedo; beyond the search
    /// radius, none.
    #[test]
    fn a_floor_by_a_bright_wall_receives_its_light() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let camera = camera();
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, radius_m: 3.0, ..Default::default() });
        let wall_z = -2.0;
        let out = run(&device, &queue, &camera, wall_z, |p| if p.z <= wall_z + 1e-3 { [1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0] }, 0.5, &mut fx);
        // floor pixels by their distance to the wall
        let inv_vp = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let (mut near, mut n_near, mut far, mut n_far) = (0.0f32, 0, 0.0f32, 0);
        for y in 0..H {
            for x in 0..W {
                let ndc = glam::Vec2::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0);
                let farp = inv_vp * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
                let dir = (farp.truncate() / farp.w - eye).normalize();
                if dir.y >= 0.0 { continue; }
                let t = -eye.y / dir.y;
                let p = eye + dir * t;
                if p.z <= wall_z || (dir.z < 0.0 && (wall_z - eye.z) / dir.z < t) { continue; }
                let d = p.z - wall_z;
                let v = out[(y * W + x) as usize][0];
                if d > 0.1 && d < 0.4 { near += v; n_near += 1; }
                if d > 3.5 { far += v; n_far += 1; }
            }
        }
        let (near, far) = (near / n_near.max(1) as f32, far / n_far.max(1) as f32);
        eprintln!("floor 0.1-0.4 m from the wall: {near:.3} ({n_near} px); beyond 3.5 m: {far:.3} ({n_far} px); ideal near 0.5 x 0.5 = 0.25");
        assert!(n_near > 10 && n_far > 10, "{n_near} {n_far}");
        assert!(near > 0.12 && near < 0.3, "the floor by the wall receives {near}");
        assert!(far < 0.02, "the floor far from the wall receives {far}");
    }
}

