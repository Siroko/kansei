use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::math::Vec3;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = include_str!("../../shaders/tonemap.wgsl");

/// Lagarde & de Rousiers' calibration constant (Moving Frostbite to PBR, 2014), as UE uses:
/// the luminance that saturates the sensor at EV100 0 is 1.2 cd/m².
const SATURATION_LUMINANCE_AT_EV0: f32 = 1.2;

/// The linear exposure for a manual EV100: `1 / (1.2 * 2^ev100)`. Physical light values (cd/m²)
/// times this land in the tone curve's range; EV100 3.9 (dusk) gives 0.0558.
pub fn exposure_from_ev100(ev100: f32) -> f32 {
    1.0 / (SATURATION_LUMINANCE_AT_EV0 * 2f32.powf(ev100))
}

/// EV100 of a physical camera: `log2(N² / t * 100 / ISO)`, for an f-number, a shutter time in
/// seconds and an ISO sensitivity.
pub fn ev100_from_camera(f_number: f32, shutter_seconds: f32, iso: f32) -> f32 {
    (f_number * f_number / shutter_seconds * 100.0 / iso).log2()
}

/// The filmic tone curve that maps exposed scene-linear light to display-linear [0, 1].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToneMapper {
    /// Clamp only.
    None,
    /// ACES RRT + sRGB ODT, Stephen Hill's fit. UE's default filmic curve is tuned to match it.
    AcesFitted,
    /// AgX (Blender 4 default): graceful highlight desaturation, no hue skews in saturated lights.
    AgX,
    /// AgX with Blender's "Punchy" look.
    AgXPunchy,
    /// Khronos PBR Neutral: keeps base colours as authored up to 0.76, then a soft shoulder.
    KhronosNeutral,
}

impl ToneMapper {
    fn gpu_id(self) -> u32 {
        match self {
            ToneMapper::None => 0,
            ToneMapper::AcesFitted => 1,
            ToneMapper::AgX => 2,
            ToneMapper::AgXPunchy => 3,
            ToneMapper::KhronosNeutral => 4,
        }
    }
}

/// Colour grading on exposed scene-linear light, before the tone curve (where UE grades too).
/// The defaults are neutral.
#[derive(Debug, Clone, Copy)]
pub struct ColorGrade {
    /// Colour temperature (K) of the light that should read as white. 6500 is neutral; lower
    /// values neutralise warm light (the picture turns cooler).
    pub white_temperature: f32,
    /// Offset of that white from the Planckian locus, -1..1 (±0.02 Δuv). Positive treats a
    /// greener white as neutral, so the picture turns magenta.
    pub white_tint: f32,
    /// Power about middle grey (0.18): above 1 more contrast.
    pub contrast: f32,
    /// Per channel, as UE's ColorSaturation (its rgb times its w): each channel's distance from
    /// the luminance is scaled, so (0.9, 0.95, 1.0) desaturates reds most.
    pub saturation: Vec3,
    /// Per-channel multiplier.
    pub gain: Vec3,
    pub shadow_gain: Vec3,
    pub shadow_saturation: Vec3,
    pub highlight_gain: Vec3,
    pub highlight_saturation: Vec3,
    /// Luminance below which a pixel counts as shadow (fading out toward it).
    pub shadows_max: f32,
    /// Luminance above which a pixel starts to count as highlight (fully at 1).
    pub highlights_min: f32,
}

impl Default for ColorGrade {
    fn default() -> Self {
        Self {
            white_temperature: 6500.0,
            white_tint: 0.0,
            contrast: 1.0,
            saturation: Vec3::new(1.0, 1.0, 1.0),
            gain: Vec3::new(1.0, 1.0, 1.0),
            shadow_gain: Vec3::new(1.0, 1.0, 1.0),
            shadow_saturation: Vec3::new(1.0, 1.0, 1.0),
            highlight_gain: Vec3::new(1.0, 1.0, 1.0),
            highlight_saturation: Vec3::new(1.0, 1.0, 1.0),
            shadows_max: 0.09,
            highlights_min: 0.5,
        }
    }
}

pub struct ToneMapOptions {
    pub tonemapper: ToneMapper,
    /// Linear multiplier on scene light; see [`exposure_from_ev100`].
    pub exposure: f32,
    /// Stops on top of `exposure` (per-shot trims).
    pub exposure_compensation: f32,
    pub grade: ColorGrade,
    /// Natural (cos⁴) vignetting, 0 = off. UE's VignetteIntensity scale; 0.5 darkens the
    /// corners to about 0.64.
    pub vignette: f32,
    /// Lateral chromatic aberration, 0 = off; at 1 red and blue differ in magnification by 1 %.
    pub chromatic_aberration: f32,
    /// Film grain amplitude, 0 = off (UE's GrainIntensity scale).
    pub grain: f32,
    /// Grain size in pixels.
    pub grain_size: f32,
    /// Write sRGB-encoded values. Needed when the surface format is not sRGB (browsers give
    /// `Bgra8Unorm`), since the blit copies the chain's last texture as is; see [`Self::for_surface`].
    pub encode_srgb: bool,
    /// Triangular dither of one 8-bit step against banding.
    pub dither: bool,
}

impl Default for ToneMapOptions {
    fn default() -> Self {
        Self {
            tonemapper: ToneMapper::AcesFitted,
            exposure: 1.0,
            exposure_compensation: 0.0,
            grade: ColorGrade::default(),
            vignette: 0.0,
            chromatic_aberration: 0.0,
            grain: 0.0,
            grain_size: 1.6,
            encode_srgb: true,
            dither: true,
        }
    }
}

impl ToneMapOptions {
    /// Defaults with `encode_srgb` set for a surface of this format
    /// (`renderer.presentation_format()`): encode unless the surface is sRGB already.
    pub fn for_surface(format: wgpu::TextureFormat) -> Self {
        Self { encode_srgb: !format.is_srgb(), ..Default::default() }
    }
}

// ── White balance ──

/// CIE 1931 xy of the Planckian locus at `kelvin` (Kim et al. 2002 cubic fit, 1667-25000 K).
fn planckian_xy(kelvin: f32) -> glam::Vec2 {
    let t = kelvin.clamp(1667.0, 25000.0) as f64;
    let (t2, t3) = (t * t, t * t * t);
    let x = if t <= 4000.0 {
        -0.2661239e9 / t3 - 0.2343589e6 / t2 + 0.8776956e3 / t + 0.179910
    } else {
        -3.0258469e9 / t3 + 2.1070379e6 / t2 + 0.2226347e3 / t + 0.240390
    };
    let (x2, x3) = (x * x, x * x * x);
    let y = if t <= 2222.0 {
        -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683
    } else if t <= 4000.0 {
        -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867
    } else {
        3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483
    };
    glam::Vec2::new(x as f32, y as f32)
}

fn xy_to_uv(xy: glam::Vec2) -> glam::Vec2 {
    let d = -2.0 * xy.x + 12.0 * xy.y + 3.0;
    glam::Vec2::new(4.0 * xy.x / d, 6.0 * xy.y / d)
}

fn uv_to_xy(uv: glam::Vec2) -> glam::Vec2 {
    let d = 2.0 * uv.x - 8.0 * uv.y + 4.0;
    glam::Vec2::new(3.0 * uv.x / d, 2.0 * uv.y / d)
}

/// White point for a temperature and a tint: the Planckian locus, moved `tint * 0.02` along its
/// normal in CIE 1960 uv (toward green for positive tints).
fn white_point_xy(kelvin: f32, tint: f32) -> glam::Vec2 {
    let uv = xy_to_uv(planckian_xy(kelvin));
    let tangent = xy_to_uv(planckian_xy(kelvin * 1.01)) - xy_to_uv(planckian_xy(kelvin * 0.99));
    // toward lower temperatures the locus runs to +u, +v; its left normal points to +v (green)
    let t = -tangent.normalize_or_zero();
    let normal = glam::Vec2::new(-t.y, t.x);
    uv_to_xy(uv + normal * (tint * 0.02))
}

/// Linear sRGB (D65) to CIE XYZ.
const SRGB_TO_XYZ: glam::Mat3 = glam::Mat3::from_cols(
    glam::Vec3::new(0.412_456_4, 0.212_672_9, 0.019_333_9),
    glam::Vec3::new(0.357_576_1, 0.715_152_2, 0.119_192),
    glam::Vec3::new(0.180_437_5, 0.072_175, 0.950_304),
);

fn xy_to_xyz(xy: glam::Vec2) -> glam::Vec3 {
    glam::Vec3::new(xy.x / xy.y, 1.0, (1.0 - xy.x - xy.y) / xy.y)
}

/// Linear-sRGB matrix that adapts white(`kelvin`, `tint`) to white(6500 K, 0) with the Bradford
/// transform, so 6500/0 is the identity.
pub fn white_balance_matrix(kelvin: f32, tint: f32) -> glam::Mat3 {
    let srgb_to_xyz = SRGB_TO_XYZ;
    let bradford = glam::Mat3::from_cols(
        glam::Vec3::new(0.8951, -0.7502, 0.0389),
        glam::Vec3::new(0.2664, 1.7135, -0.0685),
        glam::Vec3::new(-0.1614, 0.0367, 1.0296),
    );
    let src = bradford * xy_to_xyz(white_point_xy(kelvin, tint));
    let dst = bradford * xy_to_xyz(white_point_xy(6500.0, 0.0));
    let scale = glam::Mat3::from_diagonal(dst / src);
    srgb_to_xyz.inverse() * bradford.inverse() * scale * bradford * srgb_to_xyz
}

// ── GPU layout (must match ToneMapParams in tonemap.wgsl) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ToneMapParamsGpu {
    white_balance: [[f32; 4]; 3],
    gain: [f32; 3],
    exposure: f32,
    shadow_gain: [f32; 3],
    contrast: f32,
    highlight_gain: [f32; 3],
    shadows_max: f32,
    saturation: [f32; 3],
    highlights_min: f32,
    shadow_saturation: [f32; 3],
    vignette: f32,
    highlight_saturation: [f32; 3],
    chromatic_aberration: f32,
    grain: f32,
    grain_size: f32,
    width: u32,
    height: u32,
    frame: u32,
    tonemapper: u32,
    flags: u32,
    _pad: u32,
}

const FLAG_ENCODE_SRGB: u32 = 1;
const FLAG_DITHER: u32 = 2;

struct Gpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    sampler: wgpu::Sampler,
}

/// The display transform (K2): physical exposure, lens (chromatic aberration, vignette), a
/// scene-linear grade, a filmic tone curve, film grain, dither and sRGB encoding.
///
/// It turns scene-linear HDR into the display signal, so it goes last in the HDR chain: after
/// fog, anti-aliasing, depth of field and bloom, which all want linear light. A display-space
/// `ColorGradingEffect` may follow it.
///
/// ```ignore
/// let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
/// options.exposure = exposure_from_ev100(3.9);
/// options.vignette = 0.5;
/// options.grain = 0.22;
/// options.chromatic_aberration = 0.25;
/// let volume = PostProcessingVolume::new(&renderer, vec![Box::new(fog), Box::new(bloom), Box::new(ToneMapEffect::new(options))]);
/// ```
pub struct ToneMapEffect {
    pub options: ToneMapOptions,
    frame: u32,
    gpu: Option<Gpu>,
}

impl ToneMapEffect {
    pub fn new(options: ToneMapOptions) -> Self {
        Self { options, frame: 0, gpu: None }
    }

    /// The linear factor applied to scene light: exposure times 2^compensation. Pass it to
    /// `BloomEffect::exposure` so the bloom threshold is in the same units.
    pub fn total_exposure(&self) -> f32 {
        self.options.exposure * 2f32.powf(self.options.exposure_compensation)
    }

    fn params(&self, width: u32, height: u32) -> ToneMapParamsGpu {
        let o = &self.options;
        let g = &o.grade;
        let wb = white_balance_matrix(g.white_temperature, g.white_tint);
        let col = |v: glam::Vec3| [v.x, v.y, v.z, 0.0];
        let v3 = |v: Vec3| [v.x, v.y, v.z];
        ToneMapParamsGpu {
            white_balance: [col(wb.x_axis), col(wb.y_axis), col(wb.z_axis)],
            gain: v3(g.gain),
            exposure: self.total_exposure(),
            shadow_gain: v3(g.shadow_gain),
            contrast: g.contrast,
            highlight_gain: v3(g.highlight_gain),
            shadows_max: g.shadows_max.max(1e-4),
            saturation: v3(g.saturation),
            highlights_min: g.highlights_min.min(0.999),
            shadow_saturation: v3(g.shadow_saturation),
            vignette: o.vignette.max(0.0),
            highlight_saturation: v3(g.highlight_saturation),
            chromatic_aberration: o.chromatic_aberration.max(0.0),
            grain: o.grain.max(0.0),
            grain_size: o.grain_size.max(1.0),
            width,
            height,
            frame: self.frame,
            tonemapper: o.tonemapper.gpu_id(),
            flags: if o.encode_srgb { FLAG_ENCODE_SRGB } else { 0 } | if o.dither { FLAG_DITHER } else { 0 },
            _pad: 0,
        }
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ToneMap/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(2, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(3, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ToneMap/Shader"),
            source: wgpu::ShaderSource::Wgsl(WGSL.into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ToneMap/Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ToneMap/Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ToneMap/Params"),
            size: std::mem::size_of::<ToneMapParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ToneMap/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        self.gpu = Some(Gpu { pipeline, bgl, params, sampler });
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        WGSL
    }
}

impl PostProcessingEffect for ToneMapEffect {
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
        _depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        _camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        let params = self.params(width, height);
        self.frame = self.frame.wrapping_add(1);
        let gpu = self.gpu.as_ref().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ToneMap/BG"),
            layout: &gpu.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 2, resource: gpu.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("ToneMap"), timestamp_writes: crate::profiling::gpu_pass("ToneMap").as_ref().map(crate::profiling::PassStamp::compute) });
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
    fn shader_validates_and_params_layout_matches() {
        let code = ToneMapEffect::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "ToneMapParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<ToneMapParamsGpu>());
    }

    #[test]
    fn exposure_matches_ue_calibration() {
        // UE manual exposure at EV100 3.9 (the Midsommar intro): 1 / (1.2 * 2^3.9)
        assert!((exposure_from_ev100(3.9) - 0.05583).abs() < 1e-4);
        assert!((exposure_from_ev100(0.0) - 1.0 / 1.2).abs() < 1e-6);
        // f/1.4, 1/60 s, ISO 100 is EV100 ~6.9
        assert!((ev100_from_camera(1.4, 1.0 / 60.0, 100.0) - 6.878).abs() < 0.001);
        // doubling ISO opens one stop
        let a = ev100_from_camera(8.0, 1.0 / 125.0, 100.0);
        let b = ev100_from_camera(8.0, 1.0 / 125.0, 200.0);
        assert!((a - b - 1.0).abs() < 1e-5);
    }

    #[test]
    fn white_balance_is_identity_at_6500_and_neutralises_warm_white() {
        let id = white_balance_matrix(6500.0, 0.0);
        assert!(id.abs_diff_eq(glam::Mat3::IDENTITY, 1e-5), "{id:?}");

        // the linear-sRGB colour of a 3200 K white, scaled to luminance 1
        let xyz = xy_to_xyz(white_point_xy(3200.0, 0.0));
        let srgb_to_xyz = SRGB_TO_XYZ;
        let warm = srgb_to_xyz.inverse() * xyz;
        assert!(warm.x > warm.z * 1.5, "3200 K white should be orange: {warm:?}");
        // balanced for 3200 K it becomes the 6500 K white, which is near-neutral in sRGB
        let balanced = white_balance_matrix(3200.0, 0.0) * warm;
        let neutral = srgb_to_xyz.inverse() * xy_to_xyz(white_point_xy(6500.0, 0.0));
        assert!((balanced - neutral).abs().max_element() < 1e-3, "{balanced:?} vs {neutral:?}");
        assert!((neutral.x / neutral.z - 1.0).abs() < 0.1, "{neutral:?}");
    }

    #[test]
    fn tint_moves_white_off_the_locus_toward_green() {
        let on = white_point_xy(5000.0, 0.0);
        let green = white_point_xy(5000.0, 1.0);
        let magenta = white_point_xy(5000.0, -1.0);
        assert!(green.y > on.y && magenta.y < on.y, "{green:?} {on:?} {magenta:?}");
        let duv = (xy_to_uv(green) - xy_to_uv(on)).length();
        assert!((duv - 0.02).abs() < 1e-4);
    }

    #[test]
    fn params_pack_options() {
        let mut fx = ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::AgX, exposure: 0.5, exposure_compensation: 1.0, ..ToneMapOptions::for_surface(wgpu::TextureFormat::Bgra8UnormSrgb) });
        let p = fx.params(640, 360);
        assert_eq!(p.exposure, 1.0);
        assert_eq!(p.tonemapper, 2);
        assert_eq!(p.flags, FLAG_DITHER); // sRGB surface: no encode
        fx.options.encode_srgb = true;
        fx.options.dither = false;
        assert_eq!(fx.params(640, 360).flags, FLAG_ENCODE_SRGB);
    }
}
