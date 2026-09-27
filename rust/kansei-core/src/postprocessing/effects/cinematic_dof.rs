use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const COMMON: &str = include_str!("../../shaders/cinematic_dof_common.wgsl");
const PREFILTER: &str = include_str!("../../shaders/cinematic_dof_prefilter.wgsl");
const TILES: &str = include_str!("../../shaders/cinematic_dof_tiles.wgsl");
const GATHER: &str = include_str!("../../shaders/cinematic_dof_gather.wgsl");
const POSTFILTER: &str = include_str!("../../shaders/cinematic_dof_postfilter.wgsl");
const DOWNSAMPLE: &str = include_str!("../../shaders/cinematic_dof_downsample.wgsl");
const COMPOSITE: &str = include_str!("../../shaders/cinematic_dof_composite.wgsl");

/// Half-resolution texels per tile side (the shaders' TILE).
const TILE: u32 = 8;
/// Levels of the half-resolution chain the gather reads (the gather shader's LEVELS).
const LEVELS: u32 = 3;

/// A physical camera lens, as Unreal's CineCamera: the circle of confusion follows from the
/// thin-lens equation.
#[derive(Debug, Clone, Copy)]
pub struct CameraLens {
    /// Focal length, mm. `None` derives it from the camera's field of view and the filmback, so
    /// the blur always matches the picture being rendered.
    pub focal_length_mm: Option<f32>,
    /// Aperture as an f-number (focal length / aperture diameter).
    pub f_stop: f32,
    /// Distance to the plane in focus, metres of view depth.
    pub focus_distance_m: f32,
    /// Filmback width, mm; its full width spans the image width (Unreal's default 23.76).
    pub sensor_width_mm: f32,
    /// Aperture blades: 3 or more gives polygonal bokeh, fewer a round one.
    pub blade_count: u32,
    pub blade_rotation_deg: f32,
}

impl Default for CameraLens {
    fn default() -> Self {
        Self { focal_length_mm: None, f_stop: 2.8, focus_distance_m: 10.0, sensor_width_mm: 23.76, blade_count: 0, blade_rotation_deg: 0.0 }
    }
}

impl CameraLens {
    /// Focal length for a horizontal field of view on this filmback, mm.
    pub fn focal_length_for_hfov(&self, hfov_rad: f32) -> f32 {
        0.5 * self.sensor_width_mm / (0.5 * hfov_rad).tan().max(1e-6)
    }

    /// The focal length in use with `camera` (its vertical fov in degrees and aspect), mm.
    pub fn focal_length(&self, camera: &Camera) -> f32 {
        self.focal_length_mm.unwrap_or_else(|| {
            let hfov = 2.0 * ((camera.fov.to_radians() * 0.5).tan() * camera.aspect).atan();
            self.focal_length_for_hfov(hfov)
        })
    }

    /// CoC radius in pixels of a point at infinity, for an image `width_px` wide: the thin-lens
    /// CoC diameter A f (S2 - S1) / (S2 (S1 - f)) as S2 -> infinity, with A = f / N.
    pub fn coc_scale(&self, focal_length_mm: f32, width_px: u32) -> f32 {
        let f = focal_length_mm.max(1e-3);
        let s1 = (self.focus_distance_m * 1000.0).max(f * 1.001);
        let aperture = f / self.f_stop.max(0.1);
        let diameter_mm = aperture * f / (s1 - f);
        0.5 * diameter_mm * width_px as f32 / self.sensor_width_mm.max(1e-3)
    }

    /// Signed CoC radius in pixels of a point at `view_depth_m` (negative in front of the focus
    /// plane), as the shaders compute it before clamping.
    pub fn coc_radius_px(&self, focal_length_mm: f32, width_px: u32, view_depth_m: f32) -> f32 {
        self.coc_scale(focal_length_mm, width_px) * (1.0 - self.focus_distance_m / view_depth_m.max(1e-4))
    }
}

pub struct CinematicDepthOfFieldOptions {
    pub lens: CameraLens,
    /// Largest CoC radius as a fraction of the image width (bigger blur is clamped to it), so
    /// the picture looks the same at any resolution. 0.025 is Unreal's default
    /// (`r.DOF.Kernel.MaxBackgroundRadius` and `MaxForegroundRadius`).
    pub max_coc_fraction: f32,
    /// Largest CoC radius, full-resolution pixels, whatever the width: a ceiling for cost and
    /// sampling density (the gather samples discs up to 96 px at full density). The tighter of
    /// this and `max_coc_fraction` applies.
    pub max_coc_px: f32,
    /// Gather samples per half-resolution pixel. Discs wider than 12 half-resolution pixels read
    /// a coarser level of the half-resolution image, so this count holds their density too.
    pub sample_count: u32,
    /// Rotate the sample pattern every frame, for a temporal resolve after the DoF to average
    /// away. Leave off when the DoF runs after TAA (as `TemporalAAEffect` expects).
    pub temporal_noise: bool,
}

impl Default for CinematicDepthOfFieldOptions {
    fn default() -> Self {
        Self { lens: CameraLens::default(), max_coc_fraction: 0.025, max_coc_px: 96.0, sample_count: 72, temporal_noise: false }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct DofParamsGpu {
    coc_scale: f32,
    focus_distance: f32,
    max_coc: f32,
    camera_near: f32,
    camera_far: f32,
    width: u32,
    height: u32,
    sample_count: u32,
    blade_count: u32,
    blade_rotation: f32,
    frame: u32,
    _pad: u32,
}

struct Targets {
    width: u32,
    height: u32,
    /// The half-resolution chain, all levels (sampled).
    half: wgpu::TextureView,
    /// Each level alone, for writing it (storage) and for reading it into the next.
    half_levels: Vec<wgpu::TextureView>,
    tiles: wgpu::TextureView,
    tiles_dilated: wgpu::TextureView,
    bg: wgpu::TextureView,
    fg: wgpu::TextureView,
    bg_filtered: wgpu::TextureView,
    fg_filtered: wgpu::TextureView,
}

struct Gpu {
    params: wgpu::Buffer,
    prefilter: wgpu::ComputePipeline,
    prefilter_bgl: wgpu::BindGroupLayout,
    downsample: wgpu::ComputePipeline,
    downsample_bgl: wgpu::BindGroupLayout,
    tiles: wgpu::ComputePipeline,
    dilate: wgpu::ComputePipeline,
    tiles_bgl: wgpu::BindGroupLayout,
    gather: wgpu::ComputePipeline,
    gather_bgl: wgpu::BindGroupLayout,
    postfilter: wgpu::ComputePipeline,
    postfilter_bgl: wgpu::BindGroupLayout,
    composite: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    targets: Option<Targets>,
}

/// Physically based depth of field, always on like a real lens: the circle of confusion comes
/// from the camera's focal length, f-stop, focus distance and filmback (`CameraLens`, as
/// Unreal's CineCamera), and the blur is gathered as scattered bokeh at half resolution:
/// - bokeh keep their energy (a small highlight becomes a large, dimmer disc) and take the
///   aperture's shape (round, or polygonal with `blade_count`);
/// - a blurred foreground spills over whatever is behind it, while a blurred background never
///   bleeds over a sharper object in front of it (no halos at depth edges);
/// - in-focus pixels stay the full-resolution image, so fine detail such as alpha-tested
///   foliage stays crisp.
///
/// Focus, aperture and focal length are public and can change every frame (focus pulls). Put it
/// after the fog and the TAA resolve, before bloom and the tonemapper.
pub struct CinematicDepthOfFieldEffect {
    pub lens: CameraLens,
    pub max_coc_fraction: f32,
    pub max_coc_px: f32,
    pub sample_count: u32,
    pub temporal_noise: bool,
    frame: u32,
    gpu: Option<Gpu>,
}

fn source(pass: &str) -> String {
    format!("{COMMON}\n{pass}")
}

impl CinematicDepthOfFieldEffect {
    pub fn new(options: CinematicDepthOfFieldOptions) -> Self {
        Self {
            lens: options.lens,
            max_coc_fraction: options.max_coc_fraction,
            max_coc_px: options.max_coc_px,
            sample_count: options.sample_count,
            temporal_noise: options.temporal_noise,
            frame: 0,
            gpu: None,
        }
    }

    /// Largest CoC radius in pixels for an image `width_px` wide.
    pub fn max_coc_radius_px(&self, width_px: u32) -> f32 {
        (self.max_coc_fraction * width_px as f32).min(self.max_coc_px).max(0.0)
    }

    /// Signed CoC radius in pixels of a point at `view_depth_m` for this camera and image width.
    pub fn coc_radius_px(&self, camera: &Camera, width_px: u32, view_depth_m: f32) -> f32 {
        let r = self.lens.coc_radius_px(self.lens.focal_length(camera), width_px, view_depth_m);
        let max = self.max_coc_radius_px(width_px);
        r.clamp(-max, max)
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        vec![
            ("prefilter", source(PREFILTER)),
            ("downsample", source(DOWNSAMPLE)),
            ("tiles", source(TILES)),
            ("gather", source(GATHER)),
            ("postfilter", source(POSTFILTER)),
            ("composite", source(COMPOSITE)),
        ]
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
        let prefilter_bgl = bgl("CinematicDoF/PrefilterBGL", &[tex(0), depth(1), storage(2), uniform(3)]);
        let downsample_bgl = bgl("CinematicDoF/DownsampleBGL", &[tex(0), storage(1), uniform(2)]);
        let tiles_bgl = bgl("CinematicDoF/TilesBGL", &[tex(0), storage(1), uniform(2)]);
        let gather_bgl = bgl("CinematicDoF/GatherBGL", &[tex(0), tex(1), storage(2), storage(3), uniform(4)]);
        let postfilter_bgl = bgl("CinematicDoF/PostfilterBGL", &[tex(0), tex(1), storage(2), storage(3), uniform(4)]);
        let composite_bgl = bgl("CinematicDoF/CompositeBGL", &[tex(0), depth(1), tex(2), tex(3), storage(4), uniform(5)]);
        let pipeline = |label: &str, code: &str, entry: &str, bgl: &BindGroupLayout| {
            let module = device.create_shader_module(ShaderModuleDescriptor { label: Some(label), source: ShaderSource::Wgsl(code.into()) });
            let layout = device.create_pipeline_layout(&PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let tiles_src = source(TILES);
        self.gpu = Some(Gpu {
            params: device.create_buffer(&BufferDescriptor {
                label: Some("CinematicDoF/Params"),
                size: std::mem::size_of::<DofParamsGpu>() as u64,
                usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            prefilter: pipeline("CinematicDoF/Prefilter", &source(PREFILTER), "main", &prefilter_bgl),
            downsample: pipeline("CinematicDoF/Downsample", &source(DOWNSAMPLE), "main", &downsample_bgl),
            downsample_bgl,
            tiles: pipeline("CinematicDoF/Tiles", &tiles_src, "tiles", &tiles_bgl),
            dilate: pipeline("CinematicDoF/Dilate", &tiles_src, "dilate", &tiles_bgl),
            gather: pipeline("CinematicDoF/Gather", &source(GATHER), "main", &gather_bgl),
            postfilter: pipeline("CinematicDoF/Postfilter", &source(POSTFILTER), "main", &postfilter_bgl),
            postfilter_bgl,
            composite: pipeline("CinematicDoF/Composite", &source(COMPOSITE), "main", &composite_bgl),
            prefilter_bgl,
            tiles_bgl,
            gather_bgl,
            composite_bgl,
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return;
        }
        let target_format = |label: &str, w: u32, h: u32, format: wgpu::TextureFormat| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: w.max(1), height: h.max(1), depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let target = |label: &str, w: u32, h: u32| target_format(label, w, h, wgpu::TextureFormat::Rgba16Float);
        let (hw, hh) = (width.div_ceil(2), height.div_ceil(2));
        let (tw, th) = (hw.div_ceil(TILE), hh.div_ceil(TILE));
        let half = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CinematicDoF/Half"),
            size: wgpu::Extent3d { width: hw.max(1), height: hh.max(1), depth_or_array_layers: 1 },
            mip_level_count: LEVELS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let half_levels = (0..LEVELS)
            .map(|level| half.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() }))
            .collect();
        gpu.targets = Some(Targets {
            width,
            height,
            half: half.create_view(&Default::default()),
            half_levels,
            tiles: target("CinematicDoF/Tiles", tw, th),
            tiles_dilated: target("CinematicDoF/TilesDilated", tw, th),
            bg: target("CinematicDoF/Background", hw, hh),
            fg: target("CinematicDoF/Foreground", hw, hh),
            bg_filtered: target("CinematicDoF/BackgroundFiltered", hw, hh),
            fg_filtered: target("CinematicDoF/ForegroundFiltered", hw, hh),
        });
    }
}

impl PostProcessingEffect for CinematicDepthOfFieldEffect {
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
        let focal = self.lens.focal_length(camera);
        let params = DofParamsGpu {
            coc_scale: self.lens.coc_scale(focal, width),
            focus_distance: self.lens.focus_distance_m.max(1e-3),
            max_coc: self.max_coc_radius_px(width),
            camera_near: camera.near,
            camera_far: camera.far,
            width,
            height,
            sample_count: self.sample_count.max(1),
            blade_count: self.lens.blade_count,
            blade_rotation: self.lens.blade_rotation_deg.to_radians(),
            frame: if self.temporal_noise { self.frame % 64 + 1 } else { 0 },
            _pad: 0,
        };
        self.frame = self.frame.wrapping_add(1);
        let gpu = self.gpu.as_ref().unwrap();
        let t = gpu.targets.as_ref().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));

        let group = |layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>| {
            let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("CinematicDoF/BG"), layout, entries: &entries })
        };
        let view = wgpu::BindingResource::TextureView;
        let params_res = || gpu.params.as_entire_binding();
        let prefilter = group(&gpu.prefilter_bgl, vec![view(input), view(depth), view(&t.half_levels[0]), params_res()]);
        let downsamples: Vec<_> = (1..LEVELS as usize)
            .map(|l| group(&gpu.downsample_bgl, vec![view(&t.half_levels[l - 1]), view(&t.half_levels[l]), params_res()]))
            .collect();
        let tiles = group(&gpu.tiles_bgl, vec![view(&t.half), view(&t.tiles), params_res()]);
        let dilate = group(&gpu.tiles_bgl, vec![view(&t.tiles), view(&t.tiles_dilated), params_res()]);
        let gather = group(&gpu.gather_bgl, vec![view(&t.half), view(&t.tiles_dilated), view(&t.bg), view(&t.fg), params_res()]);
        let postfilter = group(&gpu.postfilter_bgl, vec![view(&t.bg), view(&t.fg), view(&t.bg_filtered), view(&t.fg_filtered), params_res()]);
        let composite = group(&gpu.composite_bgl, vec![view(input), view(depth), view(&t.bg_filtered), view(&t.fg_filtered), view(output), params_res()]);

        let (hw, hh) = (width.div_ceil(2), height.div_ceil(2));
        let (tw, th) = (hw.div_ceil(TILE), hh.div_ceil(TILE));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("CinematicDoF"), ..Default::default() });
        let mut passes = vec![(&gpu.prefilter, &prefilter, (hw, hh))];
        for (l, bind_group) in downsamples.iter().enumerate() {
            passes.push((&gpu.downsample, bind_group, ((hw >> (l + 1)).max(1), (hh >> (l + 1)).max(1))));
        }
        for (pipeline, bind_group, (x, y)) in passes.into_iter().chain([
            (&gpu.tiles, &tiles, (tw, th)),
            (&gpu.dilate, &dilate, (tw, th)),
            (&gpu.gather, &gather, (hw, hh)),
            (&gpu.postfilter, &postfilter, (hw, hh)),
            (&gpu.composite, &composite, (width, height)),
        ]) {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(x.div_ceil(8), y.div_ceil(8), 1);
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

    #[test]
    fn shaders_validate_and_the_params_layout_matches() {
        for (name, code) in CinematicDepthOfFieldEffect::shader_sources() {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "DofParams" => Some(*span as usize),
                _ => None,
            });
            assert_eq!(span, Some(std::mem::size_of::<DofParamsGpu>()), "{name}");
        }
    }

    #[test]
    fn coc_follows_the_thin_lens_equation() {
        // 85 mm f/2 focused at 5 m, full frame 36 mm across 3600 px (100 px per mm)
        let lens = CameraLens { focal_length_mm: Some(85.0), f_stop: 2.0, focus_distance_m: 5.0, sensor_width_mm: 36.0, ..Default::default() };
        let thin_lens = |s2_m: f32| {
            let (f, s1, s2) = (85.0f32, 5000.0f32, s2_m * 1000.0);
            let a = f / 2.0;
            a * f * (s2 - s1) / (s2 * (s1 - f)) // diameter, mm, signed
        };
        for depth in [2.0f32, 4.0, 5.0, 8.0, 50.0, 1e6] {
            let expected = 0.5 * thin_lens(depth) * 100.0;
            let got = lens.coc_radius_px(85.0, 3600, depth);
            assert!((got - expected).abs() <= 1e-3 * expected.abs().max(1.0), "{depth} m: {got} vs {expected}");
        }
        // in focus at the focus distance, negative in front, positive behind
        assert!(lens.coc_radius_px(85.0, 3600, 5.0).abs() < 1e-4);
        assert!(lens.coc_radius_px(85.0, 3600, 3.0) < 0.0 && lens.coc_radius_px(85.0, 3600, 9.0) > 0.0);
        // stopping down shrinks it in proportion; a wide lens at f/8 is nearly all in focus
        let f8 = CameraLens { f_stop: 8.0, ..lens };
        assert!((f8.coc_scale(85.0, 3600) * 4.0 - lens.coc_scale(85.0, 3600)).abs() < 1e-3);
        let wide = CameraLens { focal_length_mm: Some(26.0), f_stop: 8.0, focus_distance_m: 20.0, sensor_width_mm: 23.76, ..Default::default() };
        assert!(wide.coc_scale(26.0, 1920) < 0.5);
    }

    #[test]
    fn the_blur_cap_is_the_same_fraction_of_any_picture() {
        // 45 mm wide open, focused at 0.4 m: distant bokeh bigger than the cap
        let lens = CameraLens { focal_length_mm: Some(45.0), f_stop: 2.8, focus_distance_m: 0.4, ..Default::default() };
        let camera = Camera::new(20.0, 0.1, 1000.0, 2.39);
        let fx = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions { lens, ..Default::default() });
        for width in [1280u32, 1920, 2560, 3024] {
            let r = fx.coc_radius_px(&camera, width, 100.0);
            assert!((r / width as f32 - 0.025).abs() < 1e-6, "{width} px: {r}");
        }
        // up to the pixel ceiling
        assert_eq!(fx.max_coc_radius_px(7680), 96.0);
        // below the cap, the lens' own blur, in proportion to the width
        let f8 = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
            lens: CameraLens { f_stop: 8.0, focus_distance_m: 5.0, ..lens },
            ..Default::default()
        });
        let (a, b) = (f8.coc_radius_px(&camera, 1280, 100.0), f8.coc_radius_px(&camera, 2560, 100.0));
        assert!(a > 1.0 && a < 0.025 * 1280.0 && (b - 2.0 * a).abs() < 1e-4, "{a} {b}");
    }

    #[test]
    fn focal_length_follows_the_camera() {
        // Unreal's 23.76 mm filmback at 35 mm spans about 37.5 degrees horizontally
        let lens = CameraLens::default();
        let hfov = 2.0 * (23.76f32 / 70.0).atan();
        assert!((lens.focal_length_for_hfov(hfov) - 35.0).abs() < 1e-3);
        let aspect = 2.39;
        let vfov = 2.0 * ((hfov * 0.5).tan() / aspect).atan();
        let camera = Camera::new(vfov.to_degrees(), 0.1, 1000.0, aspect);
        assert!((lens.focal_length(&camera) - 35.0).abs() < 1e-2);
    }
}
