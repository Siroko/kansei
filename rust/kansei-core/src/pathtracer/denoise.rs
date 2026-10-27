use crate::cameras::Camera;
use crate::renderers::Renderer;

use super::PathTracer;

/// Iterations of the wavelet at most.
pub const MAX_ATROUS_ITERATIONS: u32 = 8;

const PARAMS_SIZE: u64 = std::mem::size_of::<SvgfParamsGpu>() as u64;

// SvgfParams flags (svgf.wgsl's SVGF_*)
const SVGF_HISTORY: u32 = 1;
const SVGF_FILTER: u32 = 2;
const SVGF_SPATIAL: u32 = 4;
const SVGF_TAA_HISTORY: u32 = 8;

/// What `PathTracerDenoise` does.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PathTracerDenoiseMode {
    /// The tracer's 1 frame, unfiltered.
    Off,
    /// SVGF's temporal accumulation alone.
    Temporal,
    /// SVGF (Schied et al. 2017): temporal accumulation, a variance estimate, then `iterations`
    /// of the edge-avoiding a-trous wavelet.
    #[default]
    Svgf,
}

/// The a-trous wavelet's kernel.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PathTracerDenoiseKernel {
    /// 3 x 3 taps (1-2-1).
    Three,
    /// 5 x 5 taps (the B3 spline, SVGF's).
    #[default]
    Five,
}

/// What `PathTracerDenoise` shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum PathTracerDenoiseView {
    /// The denoised image.
    #[default]
    Final,
    /// The tracer's frame, unfiltered.
    Noisy,
    /// The first hit's diffuse albedo (the base colour times what its specular lobe leaves).
    DiffuseAlbedo,
    /// The first hit's specular albedo (Fresnel at the view; 1 on glass).
    SpecularAlbedo,
    /// The first hit's world normal.
    Normal,
    /// The first hit's linear depth, blue to red over `heat_scale` metres.
    Depth,
    /// Pixels moved since last frame, |x| red, |y| green (a pixel at `heat_scale` 1).
    Motion,
    /// The diffuse light over its albedo, unfiltered.
    Diffuse,
    /// The specular light over its albedo, unfiltered.
    Specular,
    /// The standard deviation the wavelet ends with, relative to the light (diffuse red,
    /// specular green), times `heat_scale`.
    Variance,
    /// Frames of history the temporal pass holds, of `max_history`.
    History,
    /// The temporal accumulation alone.
    Temporal,
}

/// `name` and `from_name` for the settings enums: the names a panel or URL parameter uses.
macro_rules! named {
    ($ty:ident { $($variant:ident => $name:literal),* $(,)? }) => {
        impl $ty {
            #[doc = concat!("The setting's name, as `from_name` reads it: ", $("`", $name, "` "),*)]
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }

            pub fn from_name(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$variant),)*
                    _ => None,
                }
            }
        }
    };
}

named!(PathTracerDenoiseMode { Off => "off", Temporal => "temporal", Svgf => "svgf" });
named!(PathTracerDenoiseKernel { Three => "3x3", Five => "5x5" });
named!(PathTracerDenoiseView {
    Final => "final",
    Noisy => "noisy",
    DiffuseAlbedo => "diffuse-albedo",
    SpecularAlbedo => "specular-albedo",
    Normal => "normal",
    Depth => "depth",
    Motion => "motion",
    Diffuse => "diffuse",
    Specular => "specular",
    Variance => "variance",
    History => "history",
    Temporal => "temporal",
});

impl PathTracerDenoiseView {
    fn index(self) -> u32 {
        self as u32
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SvgfParamsGpu {
    view_matrix: [f32; 16],
    size: [f32; 2],
    proj_scale: [f32; 2],
    alpha_color: f32,
    alpha_moments: f32,
    max_history: f32,
    clamp_sigma: f32,
    phi_color: f32,
    phi_normal: f32,
    phi_depth: f32,
    pixel_size: f32,
    view: u32,
    flags: u32,
    heat_scale: f32,
    radius: u32,
    jitter: [f32; 2],
    taa_feedback: f32,
    taa_gamma: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct AtrousParamsGpu {
    step: u32,
    feedback: u32,
    _pad: [u32; 2],
}

/// The svgf.wgsl source.
pub(crate) const SVGF_WGSL: &str = include_str!("shaders/svgf.wgsl");

/// One texel of the wavelet's buffers (svgf.wgsl's `Texel`): six words.
const TEXEL_BYTES: u64 = 24;

struct Targets {
    width: u32,
    height: u32,
    integrated_diffuse: wgpu::Texture,
    integrated_specular: wgpu::Texture,
    history_diffuse: wgpu::Texture,
    history_specular: wgpu::Texture,
    moments: [wgpu::TextureView; 2],
    surface: [wgpu::TextureView; 2],
    texels: [wgpu::Buffer; 2],
    output: wgpu::TextureView,
    /// the composite before the resolve, and the resolve's history (ping-ponged)
    current: wgpu::TextureView,
    taa_history: [wgpu::TextureView; 2],
    views: [wgpu::TextureView; 4],
}

impl Targets {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let texture = |label, format, copy: bool| {
            let mut usage = wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING;
            if copy {
                usage |= wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST;
            }
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let view = |t: &wgpu::Texture| t.create_view(&Default::default());
        let f16 = wgpu::TextureFormat::Rgba16Float;
        let f32 = wgpu::TextureFormat::Rgba32Float;
        let u32 = wgpu::TextureFormat::Rgba32Uint;
        let integrated_diffuse = texture("PathTracerDenoise/IntegratedDiffuse", f16, true);
        let integrated_specular = texture("PathTracerDenoise/IntegratedSpecular", f16, true);
        let history_diffuse = texture("PathTracerDenoise/HistoryDiffuse", f16, true);
        let history_specular = texture("PathTracerDenoise/HistorySpecular", f16, true);
        let texels = |label| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: (width * height) as u64 * TEXEL_BYTES, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        Self {
            width,
            height,
            views: [view(&integrated_diffuse), view(&integrated_specular), view(&history_diffuse), view(&history_specular)],
            integrated_diffuse,
            integrated_specular,
            history_diffuse,
            history_specular,
            moments: [view(&texture("PathTracerDenoise/Moments0", f32, false)), view(&texture("PathTracerDenoise/Moments1", f32, false))],
            surface: [view(&texture("PathTracerDenoise/Surface0", u32, false)), view(&texture("PathTracerDenoise/Surface1", u32, false))],
            texels: [texels("PathTracerDenoise/Texels0"), texels("PathTracerDenoise/Texels1")],
            output: view(&texture("PathTracerDenoise/Output", f16, false)),
            current: view(&texture("PathTracerDenoise/Current", f16, false)),
            taa_history: [view(&texture("PathTracerDenoise/TaaHistory0", f16, false)), view(&texture("PathTracerDenoise/TaaHistory1", f16, false))],
        }
    }
}

/// The path tracer's denoiser: SVGF (Schied et al. 2017) over the guide buffers the tracer
/// writes (`PathTracer::set_guides`), the diffuse and the specular light filtered apart, each
/// over its albedo, then multiplied back by it. The light the tracer marks as bypassing (the sky,
/// emitters seen directly) is added unfiltered.
///
/// Each frame, after the trace: a temporal pass reprojects last frame's history along the
/// tracer's motion vectors (camera and object motion), keeping each bilinear tap only where last
/// frame's surface (depth, normal) is this one's, holds the history within the neighbourhood's
/// spread (`clamp_sigma`), and blends the new frame in; a variance pass estimates the variance
/// spatially where few frames are seen; `iterations` a-trous passes (steps 1, 2, 4 ...) filter
/// with depth, normal and luminance stops, the luminance's scaled by the variance prefiltered
/// 3 x 3; iteration `feedback_iteration`'s output is next frame's history; a composite
/// multiplies the albedos back (or shows a guide buffer, `view`); and (`taa`) a temporal
/// anti-aliasing resolve, as SVGF ends: the light is demodulated by the albedo each frame's
/// jittered sample hit, so edges only average over frames there.
///
/// ```ignore
/// tracer.set_guides(true);
/// let mut denoise = PathTracerDenoise::new(&renderer);
/// // each frame
/// tracer.trace_frame(&renderer, &bvh, &tlas, &camera, light_count);
/// denoise.denoise_frame(&renderer, &tracer, &camera);
/// denoise.present(&mut renderer, &camera, &mut volume);
/// ```
///
/// Call `reset` when the history no longer holds (a cut, a light or material changed), as
/// `PathTracer::reset_accumulation` is called. Every pass is labelled for the profiler
/// (`PathTracerDenoise/*`).
pub struct PathTracerDenoise {
    pub mode: PathTracerDenoiseMode,
    pub view: PathTracerDenoiseView,
    pub kernel: PathTracerDenoiseKernel,
    /// Iterations of the wavelet (steps 1, 2, 4 ...; at most `MAX_ATROUS_ITERATIONS`).
    pub iterations: u32,
    /// The iteration whose output is next frame's history (SVGF's: the first); at or past
    /// `iterations`, the temporal accumulation alone is.
    pub feedback_iteration: u32,
    /// The temporal blend floors: the weight of a new frame in the light and in the luminance
    /// moments once the history is long.
    pub alpha_color: f32,
    pub alpha_moments: f32,
    /// Frames the temporal history holds at most.
    pub max_history: f32,
    /// The history is held within the 3 x 3 neighbourhood's mean plus or minus this many
    /// standard deviations (of this frame's samples); 0 leaves it unclamped.
    pub clamp_sigma: f32,
    /// The edge stops: luminance (in the variance's standard deviations), normal (an exponent of
    /// their cosine), depth (in pixels' footprints off the centre's tangent plane).
    pub phi_color: f32,
    pub phi_normal: f32,
    pub phi_depth: f32,
    /// Scale of the debug views (`view`).
    pub heat_scale: f32,
    /// The temporal anti-aliasing resolve after the composite (the final view only).
    pub taa: bool,
    /// Its history's weight when still (less in motion), and its clip box's half-size in
    /// standard deviations of the 3 x 3 neighbourhood.
    pub taa_feedback: f32,
    pub taa_gamma: f32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    temporal_bgl: wgpu::BindGroupLayout,
    variance_bgl: wgpu::BindGroupLayout,
    atrous_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    resolve_bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    temporal: wgpu::ComputePipeline,
    variance: wgpu::ComputePipeline,
    atrous: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    atrous_params: Vec<wgpu::Buffer>,
    targets: Option<Targets>,
    /// which of the ping-ponged moments and surfaces this frame writes
    parity: usize,
    history_valid: bool,
    taa_valid: bool,
}

impl PathTracerDenoise {
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let tex = |sample_type| wgpu::BindingType::Texture { sample_type, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false };
        let float = tex(wgpu::TextureSampleType::Float { filterable: false });
        let uint = tex(wgpu::TextureSampleType::Uint);
        let storage_tex = |format| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D2 };
        let f16 = storage_tex(wgpu::TextureFormat::Rgba16Float);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let temporal_bgl = layout(
            "PathTracerDenoise/Temporal",
            &[
                entry(0, uniform),
                entry(1, uint),
                entry(2, uint),
                entry(3, float),
                entry(4, float),
                entry(5, float),
                entry(6, uint),
                entry(7, f16),
                entry(8, f16),
                entry(9, storage_tex(wgpu::TextureFormat::Rgba32Float)),
                entry(10, storage_tex(wgpu::TextureFormat::Rgba32Uint)),
            ],
        );
        let variance_bgl = layout("PathTracerDenoise/Variance", &[entry(0, uniform), entry(11, float), entry(12, float), entry(13, float), entry(14, uint), entry(15, storage(false)), entry(35, uint)]);
        let atrous_bgl = layout("PathTracerDenoise/Atrous", &[entry(0, uniform), entry(16, uniform), entry(17, storage(true)), entry(18, storage(false)), entry(19, f16), entry(20, f16), entry(36, uint)]);
        let composite_bgl = layout(
            "PathTracerDenoise/Composite",
            &[entry(0, uniform), entry(21, storage(true)), entry(22, uint), entry(23, uint), entry(24, uint), entry(25, float), entry(26, float), entry(27, uint), entry(28, f16)],
        );
        let resolve_bgl = layout(
            "PathTracerDenoise/Resolve",
            &[
                entry(0, uniform),
                entry(29, float),
                entry(30, uint),
                entry(31, tex(wgpu::TextureSampleType::Float { filterable: true })),
                entry(32, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(33, f16),
                entry(34, f16),
            ],
        );
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PathTracerDenoise/Linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("PathTracerDenoise"), source: wgpu::ShaderSource::Wgsl(SVGF_WGSL.into()) });
        let pipeline = |label, entry_point, bgl: &wgpu::BindGroupLayout| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let temporal = pipeline("PathTracerDenoise/Temporal", "temporal", &temporal_bgl);
        let variance = pipeline("PathTracerDenoise/Variance", "variance", &variance_bgl);
        let atrous = pipeline("PathTracerDenoise/Atrous", "atrous", &atrous_bgl);
        let composite = pipeline("PathTracerDenoise/Composite", "composite", &composite_bgl);
        let resolve = pipeline("PathTracerDenoise/Resolve", "resolve", &resolve_bgl);
        let buffer = |label, size| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let params = buffer("PathTracerDenoise/Params", PARAMS_SIZE);
        // one per iteration: several writes to one buffer before a submit leave only the last
        let atrous_params = (0..MAX_ATROUS_ITERATIONS).map(|_| buffer("PathTracerDenoise/AtrousParams", std::mem::size_of::<AtrousParamsGpu>() as u64)).collect();
        Self {
            mode: PathTracerDenoiseMode::Svgf,
            view: PathTracerDenoiseView::Final,
            kernel: PathTracerDenoiseKernel::Five,
            iterations: 5,
            feedback_iteration: 0,
            alpha_color: 0.2,
            alpha_moments: 0.2,
            max_history: 32.0,
            clamp_sigma: 0.0,
            phi_color: 4.0,
            phi_normal: 128.0,
            phi_depth: 1.0,
            heat_scale: 1.0,
            taa: true,
            taa_feedback: 0.9,
            taa_gamma: 1.0,
            device,
            queue,
            temporal_bgl,
            variance_bgl,
            atrous_bgl,
            composite_bgl,
            resolve_bgl,
            sampler,
            temporal,
            variance,
            atrous,
            composite,
            resolve,
            params,
            atrous_params,
            targets: None,
            parity: 0,
            history_valid: false,
            taa_valid: false,
        }
    }

    /// Drop the history: the next frame starts again from its own samples.
    pub fn reset(&mut self) {
        self.history_valid = false;
        self.taa_valid = false;
    }

    /// The image the last `denoise` wrote (rgba16float, HDR), once one ran.
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.targets.as_ref().map(|t| &t.output)
    }

    /// Denoise the frame `tracer` last traced (its encoder submitted, or earlier in `encoder`)
    /// from `camera`, in an encoder of its own (submitted). Then `present` shows it.
    pub fn denoise_frame(&mut self, renderer: &Renderer, tracer: &PathTracer, camera: &Camera) {
        let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("PathTracerDenoise/Frame") });
        self.denoise(&mut encoder, tracer, camera);
        renderer.submit(std::iter::once(encoder.finish()));
    }

    /// Show the denoised image on the surface through `volume`'s post-processing chain (see
    /// `PathTracer::present`). Nothing happens before the first `denoise`.
    pub fn present(&self, renderer: &mut Renderer, camera: &Camera, volume: &mut crate::postprocessing::PostProcessingVolume) {
        if let Some(view) = self.output_view() {
            renderer.present_through(view, camera, volume);
        }
    }

    /// Record the denoiser's passes for the frame `tracer` traced from `camera` (before them in
    /// `encoder`, or submitted already); returns the output, or None while the tracer writes no
    /// guides (`PathTracer::set_guides`).
    pub fn denoise(&mut self, encoder: &mut wgpu::CommandEncoder, tracer: &PathTracer, camera: &Camera) -> Option<&wgpu::TextureView> {
        if !self.record(encoder, tracer, camera, None) {
            return None;
        }
        self.output_view()
    }

    /// `denoise` into `output` (an rgba16float storage texture the tracer's size) instead of its
    /// own output; false (nothing written) while the tracer writes no guides.
    pub fn denoise_into(&mut self, encoder: &mut wgpu::CommandEncoder, tracer: &PathTracer, camera: &Camera, output: &wgpu::TextureView) -> bool {
        self.record(encoder, tracer, camera, Some(output))
    }

    fn record(&mut self, encoder: &mut wgpu::CommandEncoder, tracer: &PathTracer, camera: &Camera, output: Option<&wgpu::TextureView>) -> bool {
        let Some(guides) = tracer.guides() else { return false };
        let (width, height) = (guides.width, guides.height);
        if self.targets.as_ref().is_none_or(|t| t.width != width || t.height != height) {
            self.targets = Some(Targets::new(&self.device, width, height));
            self.history_valid = false;
            self.taa_valid = false;
        }
        let targets = self.targets.as_ref().unwrap();
        let iterations = match self.mode {
            PathTracerDenoiseMode::Svgf => self.iterations.min(MAX_ATROUS_ITERATIONS),
            _ => 0,
        };
        let filter = self.mode != PathTracerDenoiseMode::Off;
        let taa = self.taa && filter && self.view == PathTracerDenoiseView::Final;

        let p = &camera.projection_matrix.as_slice();
        let proj_scale = [1.0 / p[0], 1.0 / p[5]];
        let mut flags = 0;
        if self.history_valid {
            flags |= SVGF_HISTORY;
        }
        if filter {
            flags |= SVGF_FILTER;
        }
        if self.mode == PathTracerDenoiseMode::Svgf {
            flags |= SVGF_SPATIAL;
        }
        if taa && self.taa_valid {
            flags |= SVGF_TAA_HISTORY;
        }
        let params = SvgfParamsGpu {
            view_matrix: *camera.view_matrix.as_slice(),
            size: [width as f32, height as f32],
            proj_scale,
            alpha_color: self.alpha_color.clamp(0.001, 1.0),
            alpha_moments: self.alpha_moments.clamp(0.001, 1.0),
            max_history: self.max_history.max(1.0),
            clamp_sigma: self.clamp_sigma.max(0.0),
            phi_color: self.phi_color,
            phi_normal: self.phi_normal,
            phi_depth: self.phi_depth,
            pixel_size: 2.0 * proj_scale[1] / height as f32,
            view: self.view.index(),
            flags,
            heat_scale: self.heat_scale,
            radius: match self.kernel {
                PathTracerDenoiseKernel::Three => 1,
                PathTracerDenoiseKernel::Five => 2,
            },
            jitter: tracer.jitter(),
            taa_feedback: self.taa_feedback.clamp(0.0, 0.99),
            taa_gamma: self.taa_gamma.max(0.1),
        };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));

        let cur = self.parity;
        let prev = 1 - cur;
        let [integrated_diffuse, integrated_specular, history_diffuse, history_specular] = &targets.views;
        let view = wgpu::BindingResource::TextureView;
        let bind = |label, layout: &wgpu::BindGroupLayout, entries: &[(u32, wgpu::BindingResource)]| {
            let entries: Vec<_> = entries.iter().map(|(binding, resource)| wgpu::BindGroupEntry { binding: *binding, resource: resource.clone() }).collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
        };
        let groups = (width.div_ceil(8), height.div_ceil(8));
        let dispatch = |encoder: &mut wgpu::CommandEncoder, label: &'static str, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup| {
            let stamp = crate::profiling::gpu_pass(label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        };

        if filter {
            let bg = bind(
                "PathTracerDenoise/Temporal",
                &self.temporal_bgl,
                &[
                    (0, self.params.as_entire_binding()),
                    (1, view(guides.signal)),
                    (2, view(guides.surface)),
                    (3, view(history_diffuse)),
                    (4, view(history_specular)),
                    (5, view(&targets.moments[prev])),
                    (6, view(&targets.surface[prev])),
                    (7, view(integrated_diffuse)),
                    (8, view(integrated_specular)),
                    (9, view(&targets.moments[cur])),
                    (10, view(&targets.surface[cur])),
                ],
            );
            dispatch(encoder, "PathTracerDenoise/Temporal", &self.temporal, &bg);

            let bg = bind(
                "PathTracerDenoise/Variance",
                &self.variance_bgl,
                &[
                    (0, self.params.as_entire_binding()),
                    (11, view(integrated_diffuse)),
                    (12, view(integrated_specular)),
                    (13, view(&targets.moments[cur])),
                    (14, view(&targets.surface[cur])),
                    (15, targets.texels[0].as_entire_binding()),
                    (35, view(guides.albedo)),
                ],
            );
            dispatch(encoder, "PathTracerDenoise/Variance", &self.variance, &bg);

            for i in 0..iterations {
                let ap = AtrousParamsGpu { step: 1 << i, feedback: (i == self.feedback_iteration) as u32, _pad: [0; 2] };
                self.queue.write_buffer(&self.atrous_params[i as usize], 0, bytemuck::bytes_of(&ap));
                let bg = bind(
                    "PathTracerDenoise/Atrous",
                    &self.atrous_bgl,
                    &[
                        (0, self.params.as_entire_binding()),
                        (16, self.atrous_params[i as usize].as_entire_binding()),
                        (17, targets.texels[(i % 2) as usize].as_entire_binding()),
                        (18, targets.texels[((i + 1) % 2) as usize].as_entire_binding()),
                        (19, view(history_diffuse)),
                        (20, view(history_specular)),
                        (36, view(guides.albedo)),
                    ],
                );
                dispatch(encoder, "PathTracerDenoise/Atrous", &self.atrous, &bg);
            }
            // no iteration fed the history back: the temporal accumulation is it
            if self.feedback_iteration >= iterations {
                let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
                encoder.copy_texture_to_texture(targets.integrated_diffuse.as_image_copy(), targets.history_diffuse.as_image_copy(), size);
                encoder.copy_texture_to_texture(targets.integrated_specular.as_image_copy(), targets.history_specular.as_image_copy(), size);
            }
        }

        let bg = bind(
            "PathTracerDenoise/Composite",
            &self.composite_bgl,
            &[
                (0, self.params.as_entire_binding()),
                (21, targets.texels[(iterations % 2) as usize].as_entire_binding()),
                (22, view(guides.albedo)),
                (23, view(guides.signal)),
                (24, view(guides.surface)),
                (25, view(integrated_diffuse)),
                (26, view(integrated_specular)),
                (27, view(&targets.surface[cur])),
                (28, view(if taa { &targets.current } else { output.unwrap_or(&targets.output) })),
            ],
        );
        dispatch(encoder, "PathTracerDenoise/Composite", &self.composite, &bg);

        if taa {
            let bg = bind(
                "PathTracerDenoise/Resolve",
                &self.resolve_bgl,
                &[
                    (0, self.params.as_entire_binding()),
                    (29, view(&targets.current)),
                    (30, view(guides.surface)),
                    (31, view(&targets.taa_history[prev])),
                    (32, wgpu::BindingResource::Sampler(&self.sampler)),
                    (33, view(output.unwrap_or(&targets.output))),
                    (34, view(&targets.taa_history[cur])),
                ],
            );
            dispatch(encoder, "PathTracerDenoise/Resolve", &self.resolve, &bg);
        }
        self.taa_valid = taa;

        if filter {
            self.parity = prev;
            self.history_valid = true;
        } else {
            self.history_valid = false;
        }
        true
    }
}
