use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const COMMON: &str = include_str!("../../shaders/motion_blur_common.wgsl");
const PREPARE: &str = include_str!("../../shaders/motion_blur_prepare.wgsl");
const NEIGHBOURS: &str = include_str!("../../shaders/motion_blur_neighbours.wgsl");
const GATHER: &str = include_str!("../../shaders/motion_blur_gather.wgsl");

/// Pixels per tile side (the shaders' TILE, the prepare pass' workgroup size).
const TILE: u32 = 16;

#[derive(Debug, Clone, Copy)]
pub struct MotionBlurOptions {
    /// Fraction of the frame interval the shutter is open, as Unreal's Motion Blur Amount: 0.5
    /// is a 180-degree shutter, 0 turns the blur off.
    pub amount: f32,
    /// Largest blur, as a fraction of the image width, measured from the pixel to the end of
    /// its streak (Unreal's Motion Blur Max / 100; its default is 5 %). Faster motion is
    /// clamped to it, so the picture looks the same at any resolution.
    pub max: f32,
    /// Gather samples per pixel, in mirrored pairs (rounded up to even).
    pub sample_count: u32,
    /// Unreal's Motion Blur Target FPS: blur as if the frame rate were this, so the streaks
    /// don't depend on the rate the browser runs at. Needs the frame's duration each frame
    /// (`set_frame_time`); `None` blurs by each rendered frame's motion.
    pub target_fps: Option<f32>,
}

impl Default for MotionBlurOptions {
    fn default() -> Self {
        Self { amount: 0.5, max: 0.05, sample_count: 16, target_fps: None }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct MotionBlurParamsGpu {
    inv_view_proj: [f32; 16],
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    size: [f32; 2],
    scale: f32,
    max_radius: f32,
    steps: u32,
    frame: u32,
    enabled: u32,
    _pad: u32,
}

struct Targets {
    width: u32,
    height: u32,
    /// Per pixel: blur vector (px) and view depth.
    motion: wgpu::TextureView,
    /// Per tile: its longest blur vector, then the longest that reaches it.
    tiles: wgpu::TextureView,
    neighbours: wgpu::TextureView,
}

struct Gpu {
    params: wgpu::Buffer,
    prepare: wgpu::ComputePipeline,
    prepare_bgl: wgpu::BindGroupLayout,
    neighbours: wgpu::ComputePipeline,
    neighbours_bgl: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
    gather_bgl: wgpu::BindGroupLayout,
    targets: Option<Targets>,
}

/// Camera and object motion blur, as Unreal's (McGuire et al. 2012, Jimenez 2014): each pixel's
/// motion is the GBuffer's velocity where a material writes one (`outputs_velocity`), else the
/// camera's reprojection of its depth. The longest motion per 16x16 tile, spread to the tiles it
/// reaches, gives each pixel a dominant direction; the pixel gathers jittered samples along it
/// (and along its own motion), weighted by depth and by how far each streak reaches, so moving
/// objects smear over what is behind them, the background shows through their smeared edges,
/// and nothing static bleeds over a sharp foreground. Tiles without motion copy the input and
/// tiles where everything moves alike take a plain average, so a still frame costs three light
/// passes.
///
/// It works at the size it is given (after `TemporalAAEffect` under a render scale, the display
/// size): each output pixel reads the depth and velocity texel under it, blur lengths and the
/// cap are in output pixels, and the tiles cover the output.
///
/// A cut is not motion: the effect copies the input for a frame after `Camera::reset_motion`
/// (as the TAA reprojects nothing across it) or after `reset`.
///
/// Put it after `TemporalAAEffect` and before `CinematicDepthOfFieldEffect`, bloom and the
/// tonemapper:
/// - after the TAA, whose history must stay sharp (reprojecting blurred frames smears them
///   again and breaks its neighbourhood clamp), and whose resolve gives the blur an unjittered,
///   anti-aliased image;
/// - before the depth of field, while the colour still lines up with the depth and velocity
///   the weights classify it by (the DoF spreads colour past the depth silhouettes);
/// - before bloom and the tonemapper, on scene-linear light, so highlights streak with their
///   full energy as they do on film.
pub struct MotionBlurEffect {
    pub options: MotionBlurOptions,
    frame_time: f32,
    skip_next: bool,
    frame: u32,
    gpu: Option<Gpu>,
}

fn source(pass: &str) -> String {
    format!("{COMMON}\n{pass}")
}

impl MotionBlurEffect {
    pub fn new(options: MotionBlurOptions) -> Self {
        Self { options, frame_time: 0.0, skip_next: false, frame: 0, gpu: None }
    }

    /// No blur next frame (camera cuts, teleports). `Camera::reset_motion` does this too.
    pub fn reset(&mut self) {
        self.skip_next = true;
    }

    /// The duration of the frame about to be rendered, seconds (for `target_fps`).
    pub fn set_frame_time(&mut self, seconds: f32) {
        self.frame_time = seconds;
    }

    /// Per-frame motion in pixels -> blur radius in pixels: half the shutter's share of the
    /// motion (the streak runs both ways from the pixel), rescaled to `target_fps` if set.
    pub fn blur_scale(&self) -> f32 {
        let time_scale = match self.options.target_fps {
            Some(fps) if fps > 0.0 && self.frame_time > 1e-4 => 1.0 / (fps * self.frame_time),
            _ => 1.0,
        };
        0.5 * self.options.amount.max(0.0) * time_scale
    }

    /// Largest blur radius in pixels for an image `width_px` wide.
    pub fn max_radius_px(&self, width_px: u32) -> f32 {
        self.options.max.max(0.0) * width_px as f32
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        vec![("prepare", source(PREPARE)), ("neighbours", source(NEIGHBOURS)), ("gather", source(GATHER))]
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        use wgpu::*;
        let compute = ShaderStages::COMPUTE;
        let tex = |binding| BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: BindingType::Texture { sample_type: TextureSampleType::Float { filterable: false }, view_dimension: TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let depth = |binding| BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: BindingType::Texture { sample_type: TextureSampleType::Depth, view_dimension: TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let storage = |binding| BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: BindingType::StorageTexture { access: StorageTextureAccess::WriteOnly, format: TextureFormat::Rgba16Float, view_dimension: TextureViewDimension::D2 },
            count: None,
        };
        let uniform = |binding| BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: BindingType::Buffer { ty: BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = |label: &str, entries: &[BindGroupLayoutEntry]| device.create_bind_group_layout(&BindGroupLayoutDescriptor { label: Some(label), entries });
        let prepare_bgl = bgl("MotionBlur/PrepareBGL", &[depth(0), tex(1), storage(2), storage(3), uniform(4)]);
        let neighbours_bgl = bgl("MotionBlur/NeighboursBGL", &[tex(0), storage(1), uniform(2)]);
        let gather_bgl = bgl("MotionBlur/GatherBGL", &[tex(0), tex(1), tex(2), storage(3), uniform(4)]);
        let pipeline = |label: &str, code: &str, bgl: &BindGroupLayout| {
            let module = device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(code.into()) });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        self.gpu = Some(Gpu {
            params: device.create_buffer(&BufferDescriptor {
                label: Some("MotionBlur/Params"),
                size: std::mem::size_of::<MotionBlurParamsGpu>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            prepare: pipeline("MotionBlur/Prepare", &source(PREPARE), &prepare_bgl),
            neighbours: pipeline("MotionBlur/Neighbours", &source(NEIGHBOURS), &neighbours_bgl),
            gather: pipeline("MotionBlur/Gather", &source(GATHER), &gather_bgl),
            prepare_bgl,
            neighbours_bgl,
            gather_bgl,
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return;
        }
        let target = |label: &str, w: u32, h: u32| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let (tw, th) = (width.div_ceil(TILE), height.div_ceil(TILE));
        gpu.targets = Some(Targets {
            width,
            height,
            motion: target("MotionBlur/Motion", width, height),
            tiles: target("MotionBlur/Tiles", tw, th),
            neighbours: target("MotionBlur/Neighbours", tw, th),
        });
    }

    fn params(&self, camera: &Camera, width: u32, height: u32, enabled: bool) -> MotionBlurParamsGpu {
        let view = camera.view_matrix.to_glam();
        let jittered = camera.jittered_projection().to_glam() * view;
        let view_proj = camera.view_projection().to_glam();
        MotionBlurParamsGpu {
            inv_view_proj: jittered.inverse().to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            prev_view_proj: camera.previous_view_projection().map(|m| m.to_glam()).unwrap_or(view_proj).to_cols_array(),
            size: [width as f32, height as f32],
            scale: self.blur_scale(),
            max_radius: self.max_radius_px(width),
            steps: self.options.sample_count.div_ceil(2).max(1),
            frame: self.frame,
            enabled: enabled as u32,
            _pad: 0,
        }
    }
}

impl PostProcessingEffect for MotionBlurEffect {
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
        // a cut (no previous view) or an explicit reset: nothing moved as far as the shutter saw
        let enabled = !std::mem::take(&mut self.skip_next)
            && camera.previous_view_projection().is_some()
            && self.blur_scale() > 0.0
            && self.max_radius_px(width) >= 0.5;
        let params = self.params(camera, width, height, enabled);
        self.frame = self.frame.wrapping_add(1);
        let gpu = self.gpu.as_ref().unwrap();
        let t = gpu.targets.as_ref().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));

        let group = |layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>| {
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("MotionBlur/BG"), layout, entries: &entries })
        };
        let view = wgpu::BindingResource::TextureView;
        let params_res = || gpu.params.as_entire_binding();
        let gather = group(&gpu.gather_bgl, vec![view(input), view(&t.motion), view(&t.neighbours), view(output), params_res()]);
        let (tw, th) = (width.div_ceil(TILE), height.div_ceil(TILE));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("MotionBlur"), ..Default::default() });
        if enabled {
            let prepare = group(&gpu.prepare_bgl, vec![view(depth), view(&gbuffer.velocity_view), view(&t.motion), view(&t.tiles), params_res()]);
            let neighbours = group(&gpu.neighbours_bgl, vec![view(&t.tiles), view(&t.neighbours), params_res()]);
            pass.set_pipeline(&gpu.prepare);
            pass.set_bind_group(0, &prepare, &[]);
            pass.dispatch_workgroups(tw, th, 1);
            pass.set_pipeline(&gpu.neighbours);
            pass.set_bind_group(0, &neighbours, &[]);
            pass.dispatch_workgroups(tw.div_ceil(8), th.div_ceil(8), 1);
        }
        pass.set_pipeline(&gpu.gather);
        pass.set_bind_group(0, &gather, &[]);
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
    fn shaders_validate_and_the_params_layout_matches() {
        for (name, code) in MotionBlurEffect::shader_sources() {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "MotionBlurParams" => Some(*span as usize),
                _ => None,
            });
            assert_eq!(span, Some(std::mem::size_of::<MotionBlurParamsGpu>()), "{name}");
        }
    }

    #[test]
    fn the_blur_scales_like_unreals() {
        // amount 0.5: a 180-degree shutter, the streak half the frame's motion, its radius a quarter
        let mut fx = MotionBlurEffect::new(MotionBlurOptions { amount: 0.5, ..Default::default() });
        assert_eq!(fx.blur_scale(), 0.25);
        // at 60 fps with a 30 fps target, each frame's (half as long) motion counts twice
        fx.options.target_fps = Some(30.0);
        fx.set_frame_time(1.0 / 60.0);
        assert!((fx.blur_scale() - 0.5).abs() < 1e-5);
        fx.set_frame_time(1.0 / 30.0);
        assert!((fx.blur_scale() - 0.25).abs() < 1e-5);
        // the cap is the same fraction of any picture
        fx.options.max = 0.04;
        assert!((fx.max_radius_px(1920) - 76.8).abs() < 1e-3 && (fx.max_radius_px(1280) - 51.2).abs() < 1e-3);
    }
}
