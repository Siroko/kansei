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

/// The search radius on screen at most, as a share of the image's height. A quarter held the
/// radius to about 0.3 times the depth in metres whatever `radius_m` asked, and a Cornell box got
/// 10-40 % of its bounce; the full height costs no more (the steps stay as many) and gets 70-95 %.
const MAX_RADIUS_SCREEN: f32 = 1.0;

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

#[derive(Clone, Copy, Debug)]
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
    debug: u32,
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
    /// Debug view: output only the light the bounce adds (albedo / pi times its irradiance),
    /// black elsewhere, in place of the lit image.
    pub show_indirect: bool,
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
            show_indirect: false,
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

    /// Record the trace and the temporal filter, not the composite: the accumulated bounce (rgb
    /// its irradiance, a the share of the hemisphere left open) and its size, for an effect that
    /// composites it with light from elsewhere (`gi::VoxelGIEffect`'s near field).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn encode_trace(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, [u32; 2]) {
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
            max_radius_px: height as f32 * MAX_RADIUS_SCREEN,
            blend: self.temporal_blend.clamp(0.01, 1.0),
            has_sky: self.sky_lighting.is_some() as u32,
            debug: self.show_indirect as u32,
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let current = (self.frame % 2) as usize;
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);

        let tex = wgpu::BindingResource::TextureView;
        let p = || gpu.params.as_entire_binding();
        let trace = group(device, "SSGI/TraceBG", &gpu.trace_bgl, vec![p(), tex(input), tex(depth), tex(&gbuffer.normal_view), tex(&t.trace)]);
        let temporal = group(
            device,
            "SSGI/TemporalBG",
            &gpu.temporal_bgl,
            vec![p(), tex(&t.trace), tex(&t.history[1 - current]), tex(depth), tex(&t.history[current]), wgpu::BindingResource::Sampler(&gpu.sampler)],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SSGI"), timestamp_writes: crate::profiling::gpu_pass("SSGI").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.trace);
        pass.set_bind_group(0, &trace, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        pass.set_pipeline(&gpu.temporal);
        pass.set_bind_group(0, &temporal, &[]);
        pass.dispatch_workgroups(t.width.div_ceil(8), t.height.div_ceil(8), 1);
        (t.history[current].clone(), [t.width, t.height])
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
        let (gi, _) = self.encode_trace(device, queue, encoder, gbuffer, input, depth, camera, width, height);
        let gpu = self.gpu.as_ref().unwrap();
        let tex = wgpu::BindingResource::TextureView;
        let sky = self.sky_lighting.as_ref().unwrap_or(&gpu.no_sky);
        let composite = group(
            device,
            "SSGI/CompositeBG",
            &gpu.composite_bgl,
            vec![gpu.params.as_entire_binding(), tex(input), tex(depth), tex(&gi), tex(&gbuffer.albedo_view), tex(&gbuffer.normal_view), sky.as_entire_binding(), tex(output)],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("SSGI/Composite"), timestamp_writes: crate::profiling::gpu_pass("SSGI/Composite").as_ref().map(crate::profiling::PassStamp::compute) });
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

/// A bind group of `resources` at bindings 0, 1, 2...
fn group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>) -> wgpu::BindGroup {
    let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
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

    /// What a camera ray meets: the lit colour there, the material's albedo (none: no albedo
    /// written) and its normal (none: rebuilt from depth).
    struct Hit {
        t: f32,
        color: glam::Vec3,
        albedo: glam::Vec3,
        normal: Option<glam::Vec3>,
    }

    /// The world ray through a pixel's centre.
    fn pixel_ray(camera: &Camera, x: u32, y: u32) -> (glam::Vec3, glam::Vec3) {
        let inv_vp = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let ndc = glam::Vec2::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0);
        let far = inv_vp * glam::Vec4::new(ndc.x, ndc.y, 1.0, 1.0);
        (eye, (far.truncate() / far.w - eye).normalize())
    }

    /// Renders an analytic scene (`scene(origin, direction)`) into the GBuffer: its depth, normal
    /// and albedo through a fullscreen pass, its lit colour as the input. Runs the effect for a
    /// few frames and returns the output.
    fn run(device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera, scene: impl Fn(glam::Vec3, glam::Vec3) -> Option<Hit>, fx: &mut ScreenSpaceGIEffect) -> Vec<[f32; 4]> {
        let gbuffer = GBuffer::new(device, W, H, 1);
        let vp = camera.projection_matrix.to_glam() * camera.view_matrix.to_glam();
        let mut depth = vec![1.0f32; (W * H) as usize];
        let mut rgba = vec![0.0f32; (W * H * 4) as usize];
        let mut normal = vec![0.0f32; (W * H * 4) as usize];
        let mut albedo = vec![0.0f32; (W * H * 4) as usize];
        for y in 0..H {
            for x in 0..W {
                let (eye, dir) = pixel_ray(camera, x, y);
                let Some(hit) = scene(eye, dir) else { continue };
                let i = (y * W + x) as usize;
                let clip = vp * (eye + dir * hit.t).extend(1.0);
                depth[i] = clip.z / clip.w;
                rgba[i * 4..i * 4 + 3].copy_from_slice(&hit.color.to_array());
                albedo[i * 4..i * 4 + 3].copy_from_slice(&hit.albedo.to_array());
                if let Some(n) = hit.normal {
                    normal[i * 4..i * 4 + 3].copy_from_slice(&(n * 0.5 + 0.5).to_array());
                }
            }
        }
        let buffer = |data: &[f32]| { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(data), usage: wgpu::BufferUsages::STORAGE }) };
        let (depth_buf, normal_buf, albedo_buf) = (buffer(&depth), buffer(&normal), buffer(&albedo));
        let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
            @group(0) @binding(1) var<storage, read> n : array<vec4f>;\n\
            @group(0) @binding(2) var<storage, read> a : array<vec4f>;\n\
            @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
            struct O {{ @builtin(frag_depth) depth : f32, @location(0) normal : vec4f, @location(1) albedo : vec4f }};\n\
            @fragment fn fs(@builtin(position) pos : vec4f) -> O {{ var o : O; let i = u32(pos.y) * {W}u + u32(pos.x); o.depth = d[i]; o.normal = n[i]; o.albedo = a[i]; return o; }}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let target = |format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                targets: &[target(GBuffer::MRT_FORMATS[2]), target(GBuffer::ALBEDO_FORMAT)],
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
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: depth_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: normal_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: albedo_buf.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let attachment = |view| Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT), store: wgpu::StoreOp::Store } });
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[attachment(&gbuffer.normal_view), attachment(&gbuffer.albedo_view)],
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

    /// The floor y = 0 and a wall z = wall_z facing +z, coloured by `color(point)`, with a grey
    /// albedo (0: none written) and normals rebuilt from depth.
    fn floor_and_wall(wall_z: f32, color: impl Fn(glam::Vec3) -> [f32; 3], albedo: f32) -> impl Fn(glam::Vec3, glam::Vec3) -> Option<Hit> {
        move |eye, dir| {
            let mut t = f32::INFINITY;
            if dir.y < 0.0 { t = t.min(-eye.y / dir.y); }
            if dir.z < 0.0 { t = t.min((wall_z - eye.z) / dir.z); }
            t.is_finite().then(|| Hit { t, color: glam::Vec3::from(color(eye + dir * t)), albedo: glam::Vec3::splat(albedo), normal: None })
        }
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
        let out = run(&device, &queue, &camera, floor_and_wall(-1000.0, |_| [0.5, 0.5, 0.5], 0.0), &mut fx);
        let worst = out.iter().map(|c| (c[0] - 0.5).abs()).fold(0.0, f32::max);
        assert!(worst < 2e-3, "changed without albedo: {worst}");
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, ..Default::default() });
        let out = run(&device, &queue, &camera, floor_and_wall(-1000.0, |_| [0.5, 0.5, 0.5], 0.5), &mut fx);
        let worst = out.iter().map(|c| (c[0] - 0.5).abs()).fold(0.0, f32::max);
        assert!(worst < 0.01, "an open floor changed by {worst}");
    }

    /// A black floor meets a white wall: 0.1-0.4 m from it the floor sees the wall over 0.446 of
    /// its cosine-weighted hemisphere within the 3 m radius (by Monte Carlo), so it receives
    /// 0.446 x 0.5 (its albedo) = 0.223 of the wall's light; beyond the search radius, none.
    #[test]
    fn a_floor_by_a_bright_wall_receives_its_light() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let camera = camera();
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, radius_m: 3.0, thickness_m: std::env::var("T").map(|v| v.parse().unwrap()).unwrap_or(0.5), ..Default::default() });
        let wall_z = -2.0;
        let out = run(&device, &queue, &camera, floor_and_wall(wall_z, |p| if p.z <= wall_z + 1e-3 { [1.0, 1.0, 1.0] } else { [0.0, 0.0, 0.0] }, 0.5), &mut fx);
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
        eprintln!("floor 0.1-0.4 m from the wall: {near:.3} ({n_near} px, exact 0.223); beyond 3.5 m: {far:.3} ({n_far} px)");
        assert!(n_near > 10 && n_far > 10, "{n_near} {n_far}");
        assert!(near > 0.19 && near < 0.24, "the floor by the wall receives {near}");
        assert!(far < 0.02, "the floor far from the wall receives {far}");
    }

    /// A 2 m Cornell box, open toward the camera: white floor, ceiling and back wall, a red wall
    /// on the left and a green one on the right, lit by a downlight under the ceiling.
    const BOX_LIGHT: glam::Vec3 = glam::Vec3::new(0.0, 1.95, -1.0);

    fn box_hit(o: glam::Vec3, d: glam::Vec3) -> Option<(f32, glam::Vec3, glam::Vec3)> {
        use glam::Vec3;
        let white = Vec3::splat(0.73);
        let faces = [
            (Vec3::Y, 0.0, white),
            (-Vec3::Y, -2.0, white),
            (Vec3::X, -1.0, Vec3::new(0.63, 0.065, 0.05)),
            (-Vec3::X, -1.0, Vec3::new(0.14, 0.45, 0.09)),
            (Vec3::Z, -2.0, white),
        ];
        let mut best: Option<(f32, Vec3, Vec3)> = None;
        for (n, c, albedo) in faces {
            let dn = n.dot(d);
            if dn >= 0.0 { continue; }
            let t = (c - n.dot(o)) / dn;
            if t <= 1e-4 || best.is_some_and(|b| b.0 <= t) { continue; }
            let p = o + d * t;
            if p.x.abs() > 1.001 || p.y < -0.001 || p.y > 2.001 || p.z < -2.001 || p.z > 0.001 { continue; }
            best = Some((t, n, albedo));
        }
        best
    }

    /// The light a point of the box sends out: a 10 cd downlight whose intensity falls off with
    /// the cosine from straight down, so the ceiling gets none directly.
    fn box_radiance(p: glam::Vec3, n: glam::Vec3, albedo: glam::Vec3) -> glam::Vec3 {
        let l = BOX_LIGHT - p;
        let d2 = l.length_squared();
        let l = l / d2.sqrt();
        albedo / std::f32::consts::PI * (10.0 * l.y.max(0.0) * n.dot(l).max(0.0) / d2)
    }

    fn cornell_box(eye: glam::Vec3, dir: glam::Vec3) -> Option<Hit> {
        box_hit(eye, dir).map(|(t, n, albedo)| Hit { t, color: box_radiance(eye + dir * t, n, albedo), albedo, normal: Some(n) })
    }

    fn rnd(i: u32) -> f32 {
        let mut x = i.wrapping_mul(0x9E37_79B9) ^ 0x85EB_CA6B;
        x ^= x >> 16;
        x = x.wrapping_mul(0x7FEB_352D);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846C_A68B);
        x ^= x >> 16;
        (x >> 8) as f32 / (1u32 << 24) as f32
    }

    /// The irradiance the box's lit surfaces send onto p (normal n) in one bounce, by Monte Carlo.
    fn box_one_bounce(p: glam::Vec3, n: glam::Vec3, seed: u32) -> glam::Vec3 {
        let (t, b) = n.any_orthonormal_pair();
        let samples = 1024u32;
        let mut sum = glam::Vec3::ZERO;
        for j in 0..samples {
            let (u1, u2) = (rnd(seed.wrapping_mul(7919) + j * 2), rnd(seed.wrapping_mul(7919) + j * 2 + 1));
            let (r, phi) = (u1.sqrt(), 2.0 * std::f32::consts::PI * u2);
            let d = t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1.0 - u1).max(0.0).sqrt();
            if let Some((th, hn, albedo)) = box_hit(p + n * 1e-4, d) {
                sum += box_radiance(p + n * 1e-4 + d * th, hn, albedo);
            }
        }
        sum * (std::f32::consts::PI / samples as f32)
    }

    /// In the Cornell box, the bounce the effect adds (as irradiance) against the one-bounce
    /// reference: the colour bleeding from the red and green walls onto the floor and the back
    /// wall, and the ceiling lit only by the bounce. What is missing is what the screen cannot
    /// see (the search radius, the gaps between depth slabs, the open front).
    #[test]
    fn a_cornell_box_gets_most_of_its_one_bounce() {
        let Some((device, queue)) = gpu() else { return eprintln!("no GPU adapter: skipping") };
        let mut camera = Camera::new(60.0, 0.1, 100.0, W as f32 / H as f32);
        camera.set_position(0.0, 1.0, 0.9);
        camera.look_at(&crate::math::Vec3::new(0.0, 1.0, -1.0));
        camera.update_view_matrix();
        let mut fx = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality: GiQuality::Ultra, ..Default::default() });
        let out = run(&device, &queue, &camera, cornell_box, &mut fx);
        let regions: [(&str, fn(glam::Vec3) -> bool); 4] = [
            ("floor by the red wall", |p| p.y < 1e-3 && p.x < -0.6 && p.z > -1.6 && p.z < -0.4),
            ("floor by the green wall", |p| p.y < 1e-3 && p.x > 0.6 && p.z > -1.6 && p.z < -0.4),
            ("ceiling", |p| p.y > 2.0 - 1e-3 && p.x.abs() < 0.6),
            ("back wall", |p| p.z < -2.0 + 1e-3 && p.x.abs() < 0.6 && p.y > 0.4 && p.y < 1.6),
        ];
        for (name, inside) in regions {
            let (mut got, mut want, mut pixels) = (glam::Vec3::ZERO, glam::Vec3::ZERO, 0);
            for y in (0..H).step_by(5) {
                for x in (0..W).step_by(5) {
                    let (eye, dir) = pixel_ray(&camera, x, y);
                    let Some((t, n, albedo)) = box_hit(eye, dir) else { continue };
                    let p = eye + dir * t;
                    if !inside(p) { continue; }
                    let i = (y * W + x) as usize;
                    // the albedo as the GBuffer stores it (8 bits)
                    let a = (albedo * 255.0).round() / 255.0;
                    let added = glam::Vec3::new(out[i][0], out[i][1], out[i][2]) - box_radiance(p, n, albedo);
                    got += added * std::f32::consts::PI / a;
                    want += box_one_bounce(p, n, i as u32);
                    pixels += 1;
                }
            }
            let ratio = got / want;
            eprintln!("{name}: {pixels} px, bounce {:.3} {:.3} {:.3} of the reference {:.3} {:.3} {:.3}: {:.2} {:.2} {:.2}",
                got.x / pixels as f32, got.y / pixels as f32, got.z / pixels as f32, want.x / pixels as f32, want.y / pixels as f32, want.z / pixels as f32, ratio.x, ratio.y, ratio.z);
            assert!(pixels > 20, "{name}: {pixels} px");
            // the colour bled from the nearest wall arrives almost in full; the rest of the light
            // mostly, less what comes from the box's parts off screen
            assert!(ratio.min_element() > 0.6 && ratio.max_element() < 1.05, "{name}: {ratio}");
        }
    }
}

