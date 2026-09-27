use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = include_str!("../../shaders/taa_resolve.wgsl");
const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub struct TemporalAAOptions {
    /// History weight when the image is still (0.95: about 20 frames of accumulation).
    pub feedback_max: f32,
    /// History weight at 8+ pixels of motion per frame.
    pub feedback_min: f32,
    /// Scene multiplier of the resolve's working space; set it to the tonemapper's exposure
    /// (`ToneMapEffect::total_exposure()`) so its luma weighting matches what is displayed.
    pub exposure: f32,
    /// Half-size of the history clip box in standard deviations of the neighbourhood (lower:
    /// less ghosting, more flicker).
    pub variance_gamma: f32,
}

impl Default for TemporalAAOptions {
    fn default() -> Self {
        Self { feedback_max: 0.95, feedback_min: 0.85, exposure: 1.0, variance_gamma: 1.0 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TaaParamsGpu {
    inv_view_proj: [f32; 16],
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    jitter_px: [f32; 2],
    size: [f32; 2],
    feedback_min: f32,
    feedback_max: f32,
    exposure: f32,
    variance_gamma: f32,
    has_history: u32,
    has_velocity: u32,
    input_size: [f32; 2],
}

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// Ping-pong history, `read` is last frame's.
    history: [wgpu::TextureView; 2],
    read: usize,
    size: (u32, u32),
}

/// Temporal anti-aliasing: the renderer jitters the camera's projection by a sub-pixel Halton
/// offset every frame while this effect is in the chain, and the resolve accumulates the
/// jittered frames into a history reprojected by motion vectors (materials with
/// `outputs_velocity`, drawn in the renderer's velocity pass) or by depth (everything else,
/// camera motion only), clipped to the current neighbourhood so moving things don't ghost.
///
/// Put it after the volumetric fog and before depth of field, bloom and the tonemapper (it works
/// on linear light). On camera cuts call `reset_history` here and `Camera::reset_motion`, so
/// nothing is reprojected across the cut.
///
/// History is also dropped per pixel where it saw another surface: its view depth is kept in the
/// history's alpha and compared with where this pixel's surface was last frame, which keeps
/// swaying foliage (which uncovers and covers itself every frame) from smearing.
///
/// It is also the chain's temporal upscaler: with `Renderer::set_render_scale` below 1 it reads
/// the GBuffer-size frame and writes (and keeps its history at) the display size, placing each
/// jittered sample where it fell among the display's pixels, so the history gathers detail
/// finer than a rendered pixel over the frames. At scale 1 it is the plain 1:1 resolve.
pub struct TemporalAAEffect {
    pub options: TemporalAAOptions,
    has_history: bool,
    gpu: Option<Gpu>,
}

impl TemporalAAEffect {
    pub fn new(options: TemporalAAOptions) -> Self {
        Self { options, has_history: false, gpu: None }
    }

    /// Start over from the current frame (camera cuts, teleports).
    pub fn reset_history(&mut self) {
        self.has_history = false;
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let texture = |filterable| wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let storage = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: HISTORY_FORMAT,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TAA/BGL"),
            entries: &[
                entry(0, texture(false)),
                entry(1, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(2, texture(false)),
                entry(3, texture(true)),
                entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(5, storage),
                entry(6, storage),
                entry(7, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("TAA/Shader"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("TAA/Layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("TAA/Resolve"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TAA/Params"),
            size: std::mem::size_of::<TaaParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("TAA/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let history = [Self::history_texture(device, 1, 1), Self::history_texture(device, 1, 1)];
        self.gpu = Some(Gpu {
            pipeline,
            bgl,
            params,
            sampler,
            history,
            read: 0,
            size: (1, 1),
        });
        self.has_history = false;
    }

    fn history_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("TAA/History"),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: HISTORY_FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default())
    }

    /// Resolve parameters for an output of `width` x `height` from a frame rendered at `input`.
    fn params(&self, camera: &Camera, input: (u32, u32), width: u32, height: u32, has_velocity: bool) -> TaaParamsGpu {
        let view = camera.view_matrix.to_glam();
        let jittered = camera.jittered_projection().to_glam() * view;
        let view_proj = camera.view_projection().to_glam();
        let o = &self.options;
        TaaParamsGpu {
            inv_view_proj: jittered.inverse().to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            prev_view_proj: camera.previous_view_projection().map(|m| m.to_glam()).unwrap_or(view_proj).to_cols_array(),
            jitter_px: [camera.jitter[0] * input.0 as f32 * 0.5, -camera.jitter[1] * input.1 as f32 * 0.5],
            size: [width as f32, height as f32],
            feedback_min: o.feedback_min.clamp(0.0, 0.99),
            feedback_max: o.feedback_max.clamp(0.0, 0.99),
            exposure: o.exposure.max(1e-8),
            variance_gamma: o.variance_gamma.max(0.1),
            has_history: self.has_history as u32,
            has_velocity: has_velocity as u32,
            input_size: [input.0 as f32, input.1 as f32],
        }
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        WGSL
    }
}

impl PostProcessingEffect for TemporalAAEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

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
        if self.gpu.as_ref().unwrap().size != (width, height) {
            let gpu = self.gpu.as_mut().unwrap();
            gpu.history = [Self::history_texture(device, width, height), Self::history_texture(device, width, height)];
            gpu.size = (width, height);
            self.has_history = false;
        }
        // the input is at the GBuffer's size (the effects before the upscaler run at it)
        let params = self.params(camera, (gbuffer.width, gbuffer.height), width, height, true);
        let gpu = self.gpu.as_mut().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let velocity = &gbuffer.velocity_view;
        let (read, write) = (gpu.read, 1 - gpu.read);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TAA/BG"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(velocity) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gpu.history[read]) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&gpu.history[write]) },
                wgpu::BindGroupEntry { binding: 7, resource: gpu.params.as_entire_binding() },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("TAA/Resolve"), ..Default::default() });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        gpu.read = write;
        self.has_history = true;
    }

    // The history is at the output size and starts over when that changes (see `render`), so a
    // new render scale keeps it.
    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn wants_jitter(&self) -> bool {
        true
    }

    fn upscales_to_display(&self) -> bool {
        true
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

    #[test]
    fn shader_validates_and_params_layout_matches() {
        let code = TemporalAAEffect::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "TaaParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<TaaParamsGpu>());
    }

    /// MOTION_VECTORS_WGSL works in a material that writes the velocity target, and its camera
    /// struct matches what `Camera::upload` writes.
    #[test]
    fn motion_vector_chunk_validates_in_a_material() {
        let shader = format!(
            "{}
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
             @group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
             @group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;
             struct VOut {{ @builtin(position) clip: vec4f, @location(0) curr: vec4f, @location(1) prev: vec4f }};
             struct FOut {{ @location(0) color: vec4f, @location(4) velocity: vec2f }};
             @vertex fn vertex_main(@location(0) position: vec4f) -> VOut {{
                     let world = mesh.world * position;
                     var out: VOut;
                     out.clip = projection_matrix * view_matrix * world;
                     out.curr = kansei_camera_temporal.viewProj * world;
                     out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * position);
                     return out;
}}
             @fragment fn fragment_main(in: VOut) -> FOut {{
                     return FOut(vec4f(1.0), kansei_motion_vector(in.curr, in.prev));
}}
",
            crate::cameras::MOTION_VECTORS_WGSL
        );
        let module = naga::front::wgsl::parse_str(&shader).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&shader)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "KanseiCameraTemporal" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<crate::cameras::camera::CameraTemporalGpu>());
    }

    /// The jitter in pixels the resolve uses matches where the jittered projection moves a point.
    #[test]
    fn jitter_in_pixels_matches_the_jittered_projection() {
        let mut camera = Camera::new(50.0, 0.1, 100.0, 16.0 / 9.0);
        camera.jitter = [0.3 * 2.0 / 1280.0, -0.2 * 2.0 / 720.0]; // +0.3 px right, +0.2 px down
        let p = TemporalAAEffect::new(Default::default()).params(&camera, (1280, 720), 2560, 1440, false);
        assert!((p.jitter_px[0] - 0.3).abs() < 1e-5 && (p.jitter_px[1] - 0.2).abs() < 1e-5, "{:?}", p.jitter_px);
        // a point at the screen centre moves by the same amount under the jittered projection
        let point = glam::Vec4::new(0.0, 0.0, -10.0, 1.0);
        let clip = camera.jittered_projection().to_glam() * point;
        let px = ((clip.x / clip.w) * 640.0, -(clip.y / clip.w) * 360.0);
        assert!((px.0 - 0.3).abs() < 1e-4 && (px.1 - 0.2).abs() < 1e-4, "{px:?}");
    }
}
