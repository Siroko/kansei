//! `HdrEnvironment`: an equirectangular HDR image as an environment source.

use bytemuck::{Pod, Zeroable};

use super::{EnvironmentLighting, EnvironmentMap};
use crate::gi::SkyLightingData;

const PREPARE_WGSL: &str = include_str!("shaders/hdr_environment.wgsl");

pub(crate) fn prepare_source() -> String {
    format!("{}\n{PREPARE_WGSL}", crate::atmosphere::SKY_LIGHTING_WGSL)
}

/// An equirectangular image of linear radiance (rgb, row 0 the top: straight up).
#[derive(Clone, Debug, PartialEq)]
pub struct HdrImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<[f32; 3]>,
}

impl HdrImage {
    /// Decode a Radiance `.hdr` (RGBE) file, or any other floating-point image the `image` crate
    /// reads (OpenEXR among them), from its bytes: natively from a file, on the web from a fetch.
    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let image = image::load_from_memory(bytes).map_err(|e| format!("HdrImage: {e}"))?.into_rgb32f();
        let (width, height) = image.dimensions();
        let pixels = image.pixels().map(|p| p.0).collect();
        Ok(Self { width, height, pixels })
    }

    /// An image from its pixels, row by row from the top.
    pub fn from_pixels(width: u32, height: u32, pixels: Vec<[f32; 3]>) -> Self {
        assert_eq!(pixels.len(), (width * height) as usize, "HdrImage: {width} x {height} needs that many pixels");
        Self { width, height, pixels }
    }

    /// Half the size each way, each pixel the mean of four.
    fn halved(&self) -> Self {
        let (w, h) = ((self.width / 2).max(1), (self.height / 2).max(1));
        let at = |x: u32, y: u32| self.pixels[(y.min(self.height - 1) * self.width + x.min(self.width - 1)) as usize];
        let mut pixels = Vec::with_capacity((w * h) as usize);
        for y in 0..h {
            for x in 0..w {
                let q = [at(2 * x, 2 * y), at(2 * x + 1, 2 * y), at(2 * x, 2 * y + 1), at(2 * x + 1, 2 * y + 1)];
                pixels.push(std::array::from_fn(|c| q.iter().map(|p| p[c]).sum::<f32>() * 0.25));
            }
        }
        Self { width: w, height: h, pixels }
    }

    /// The pixels as rgba16float texels.
    fn to_rgba16(&self) -> Vec<u16> {
        self.pixels.iter().flat_map(|p| [f32_to_f16(p[0]), f32_to_f16(p[1]), f32_to_f16(p[2]), f32_to_f16(1.0)]).collect()
    }
}

/// A half float, rounded to nearest; negatives and NaN to 0 and anything past its range to its
/// largest finite value (a radiance is never negative, and an infinity would spread through the
/// filtering).
pub(crate) fn f32_to_f16(v: f32) -> u16 {
    let v = if v.is_nan() { 0.0 } else { v.clamp(0.0, 65504.0) };
    let bits = v.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x7f_ffff;
    if exp <= 0 {
        // subnormal (or zero): the mantissa with its implicit bit, shifted into place
        if exp < -10 {
            return 0;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - exp) as u32;
        let half = m >> shift;
        let round = (m >> (shift - 1)) & 1;
        return (half + round) as u16;
    }
    let half = ((exp as u32) << 10) | (mant >> 13);
    // round to nearest (a carry into the exponent is still the right value)
    (half + ((mant >> 12) & 1)).min(0x7bff) as u16
}

/// What an `HdrEnvironment` shows: changing any of it prepares the environment again.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HdrEnvironmentSettings {
    /// Scale of the image's radiance (2^EV for an exposure).
    pub intensity: f32,
    /// Turn about +y, radians: positive turns the environment from -z toward +x.
    pub rotation: f32,
    /// Multiplies the radiance (linear rgb).
    pub tint: [f32; 3],
}

impl Default for HdrEnvironmentSettings {
    fn default() -> Self {
        Self { intensity: 1.0, rotation: 0.0, tint: [1.0; 3] }
    }
}

/// How an `HdrEnvironment` is prepared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HdrEnvironmentOptions {
    /// Face size of the cubemaps, texels; 0 picks a quarter of the image's width (its texels at
    /// the equator), as a power of two from 32 to `max_face_size`.
    pub face_size: u32,
    pub max_face_size: u32,
    /// GGX samples a texel of each rough mip takes.
    pub samples: u32,
}

impl Default for HdrEnvironmentOptions {
    fn default() -> Self {
        Self { face_size: 0, max_face_size: 512, samples: 64 }
    }
}

/// The WGSL `HdrPrepare`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct HdrPrepareGpu {
    rotation: [f32; 4],
    tint: [f32; 4],
}

/// The WGSL `HdrMipPass`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct HdrMipPassGpu {
    roughness: f32,
    samples: u32,
    _pad: [u32; 2],
}

struct Pipelines {
    to_cube: wgpu::ComputePipeline,
    downsample: wgpu::ComputePipeline,
    prefilter: wgpu::ComputePipeline,
    project_sh: wgpu::ComputePipeline,
    to_cube_group: wgpu::BindGroup,
    /// mip m of the radiance from mip m - 1 (m from 1)
    downsample_groups: Vec<wgpu::BindGroup>,
    /// mip m of the environment (m from 1)
    prefilter_groups: Vec<wgpu::BindGroup>,
    project_sh_group: wgpu::BindGroup,
}

/// An HDR environment from an equirectangular image (`HdrImage`, e.g. a studio HDRI from Poly
/// Haven): the third environment source beside the gradient (`GradientEnvironment`) and the
/// atmosphere (`SkyAtmosphere::lighting`), with an intensity, a turn about the vertical and a tint
/// (`set_settings`). Its `lighting` has
/// - the `SkyLighting` uniform: the radiance as order-2 SH, the diffuse light for the GI's misses,
///   the voxels' sky, the fog and the materials' ambient;
/// - the environment cubemap: the radiance at mip 0 (the background, `EnvironmentBackgroundEffect`,
///   and sharp reflections), GGX-prefiltered at mip m for roughness m / (mips - 1)
///   (`SKY_ENVIRONMENT_WGSL`: glossy reflections and the materials' specular ambient).
///
/// It is prepared on the GPU (`prepare`): the image into a radiance cubemap, its mips, the
/// prefiltered levels and the SH, once at load and again only when the settings change.
pub struct HdrEnvironment {
    settings: HdrEnvironmentSettings,
    /// what the cubemaps and the SH hold (None: nothing yet)
    prepared: Option<HdrEnvironmentSettings>,
    face_size: u32,
    image_size: (u32, u32),
    radiance: wgpu::Texture,
    environment: wgpu::Texture,
    prepare: wgpu::Buffer,
    lighting: EnvironmentLighting,
    pipelines: Pipelines,
}

impl HdrEnvironment {
    /// Upload `image` and make the GPU work that prepares it; the first `prepare` does it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, image: &HdrImage, options: HdrEnvironmentOptions) -> Self {
        // an image wider than the device holds, halved until it fits
        let max = device.limits().max_texture_dimension_2d;
        let mut halved = None;
        while halved.as_ref().unwrap_or(image).width > max {
            halved = Some(halved.as_ref().unwrap_or(image).halved());
        }
        let image = halved.as_ref().unwrap_or(image);
        let face_size = match options.face_size {
            0 => (image.width / 4).max(1).next_power_of_two().clamp(32, options.max_face_size.max(32)),
            n => n.max(1),
        };
        let mips = face_size.ilog2() + 1;
        use wgpu::util::DeviceExt;
        let equirect = device
            .create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("HdrEnvironment/Equirect"),
                    size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                bytemuck::cast_slice(&image.to_rgba16()),
            )
            .create_view(&Default::default());
        let cube = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: face_size, height: face_size, depth_or_array_layers: 6 },
                mip_level_count: mips,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let radiance = cube("HdrEnvironment/Radiance");
        let environment = cube("HdrEnvironment/Environment");
        let layer_view = |texture: &wgpu::Texture, mip: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                base_mip_level: mip,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let cube_view = |texture: &wgpu::Texture| texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::Cube), ..Default::default() });
        let sampler = |label, address_mode_u| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            })
        };
        let equirect_sampler = sampler("HdrEnvironment/EquirectSampler", wgpu::AddressMode::Repeat);
        let cube_sampler = sampler("HdrEnvironment/CubeSampler", wgpu::AddressMode::ClampToEdge);
        let sky_lighting = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HdrEnvironment/SkyLighting"),
            size: std::mem::size_of::<SkyLightingData>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let prepare = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HdrEnvironment/Prepare"),
            size: std::mem::size_of::<HdrPrepareGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // each rough mip's roughness in a slot of its own (the dispatches share a submit)
        let stride = (std::mem::size_of::<HdrMipPassGpu>() as u64).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as u64);
        let mut slots = vec![0u8; (stride * mips as u64) as usize];
        for mip in 0..mips {
            let pass = HdrMipPassGpu { roughness: mip as f32 / (mips - 1).max(1) as f32, samples: options.samples.max(1), _pad: [0; 2] };
            let at = (stride * mip as u64) as usize;
            slots[at..at + std::mem::size_of::<HdrMipPassGpu>()].copy_from_slice(bytemuck::bytes_of(&pass));
        }
        let mip_passes = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("HdrEnvironment/MipPasses"), contents: &slots, usage: wgpu::BufferUsages::UNIFORM });

        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let texture = |filterable, view_dimension| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension, multisampled: false };
        let filtering = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        let cube_out = wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2Array };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let cube_in = texture(true, wgpu::TextureViewDimension::Cube);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("HdrEnvironment"), source: wgpu::ShaderSource::Wgsl(prepare_source().into()) });
        let pipeline = |label: &str, entry_point: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            });
            (pipeline, bgl)
        };
        let (to_cube, to_cube_bgl) = pipeline(
            "HdrEnvironment/ToCube",
            "to_cube",
            &[entry(0, uniform), entry(1, texture(true, wgpu::TextureViewDimension::D2)), entry(2, filtering), entry(3, cube_out)],
        );
        let (downsample, downsample_bgl) = pipeline("HdrEnvironment/Downsample", "downsample", &[entry(3, cube_out), entry(4, texture(false, wgpu::TextureViewDimension::D2Array))]);
        let (prefilter, prefilter_bgl) = pipeline("HdrEnvironment/Prefilter", "prefilter", &[entry(3, cube_out), entry(5, cube_in), entry(6, filtering), entry(7, uniform)]);
        let (project_sh, project_sh_bgl) = pipeline("HdrEnvironment/ProjectSh", "project_sh", &[entry(5, cube_in), entry(6, filtering), entry(8, storage)]);
        let group = |label, layout, entries: &[wgpu::BindGroupEntry]| device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries });
        fn tex(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
            wgpu::BindingResource::TextureView(view)
        }
        let radiance_cube = cube_view(&radiance);
        let to_cube_group = group(
            "HdrEnvironment/ToCube",
            &to_cube_bgl,
            &[
                wgpu::BindGroupEntry { binding: 0, resource: prepare.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tex(&equirect) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&equirect_sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: tex(&layer_view(&radiance, 0)) },
            ],
        );
        let downsample_groups = (1..mips)
            .map(|mip| {
                group(
                    "HdrEnvironment/Downsample",
                    &downsample_bgl,
                    &[wgpu::BindGroupEntry { binding: 3, resource: tex(&layer_view(&radiance, mip)) }, wgpu::BindGroupEntry { binding: 4, resource: tex(&layer_view(&radiance, mip - 1)) }],
                )
            })
            .collect();
        let prefilter_groups = (1..mips)
            .map(|mip| {
                let slot = wgpu::BufferBinding { buffer: &mip_passes, offset: stride * mip as u64, size: wgpu::BufferSize::new(std::mem::size_of::<HdrMipPassGpu>() as u64) };
                group(
                    "HdrEnvironment/Prefilter",
                    &prefilter_bgl,
                    &[
                        wgpu::BindGroupEntry { binding: 3, resource: tex(&layer_view(&environment, mip)) },
                        wgpu::BindGroupEntry { binding: 5, resource: tex(&radiance_cube) },
                        wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&cube_sampler) },
                        wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Buffer(slot) },
                    ],
                )
            })
            .collect();
        let project_sh_group = group(
            "HdrEnvironment/ProjectSh",
            &project_sh_bgl,
            &[
                wgpu::BindGroupEntry { binding: 5, resource: tex(&radiance_cube) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::Sampler(&cube_sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: sky_lighting.as_entire_binding() },
            ],
        );
        let lighting = EnvironmentLighting { sky_lighting, environment: Some(EnvironmentMap { view: cube_view(&environment), sampler: cube_sampler, mips }) };
        Self {
            settings: HdrEnvironmentSettings::default(),
            prepared: None,
            face_size,
            image_size: (image.width, image.height),
            radiance,
            environment,
            prepare,
            lighting,
            pipelines: Pipelines { to_cube, downsample, prefilter, project_sh, to_cube_group, downsample_groups, prefilter_groups, project_sh_group },
        }
    }

    /// Decode an `.hdr` (or `.exr`) file's bytes and upload it (`HdrImage::decode`, `new`).
    pub fn from_bytes(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8], options: HdrEnvironmentOptions) -> Result<Self, String> {
        Ok(Self::new(device, queue, &HdrImage::decode(bytes)?, options))
    }

    pub fn settings(&self) -> HdrEnvironmentSettings {
        self.settings
    }

    /// Change the intensity, the turn or the tint: the next `prepare` works them in.
    pub fn set_settings(&mut self, settings: HdrEnvironmentSettings) {
        self.settings = settings;
    }

    /// Whether the cubemaps and the SH hold the current settings.
    pub fn is_prepared(&self) -> bool {
        self.prepared == Some(self.settings)
    }

    /// Prepare the environment on the GPU for the current settings, unless it holds them already
    /// (call it every frame before rendering: it costs nothing then). Returns whether it ran. Its
    /// passes are labelled `Environment/*` for the profiler.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> bool {
        if self.is_prepared() {
            return false;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("HdrEnvironment/Prepare") });
        self.encode(queue, &mut encoder);
        queue.submit(std::iter::once(encoder.finish()));
        true
    }

    /// `prepare`'s work into `encoder`, whatever was prepared before.
    pub fn encode(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let s = self.settings;
        let gpu = HdrPrepareGpu { rotation: [s.rotation.cos(), s.rotation.sin(), s.intensity.max(0.0), 0.0], tint: [s.tint[0].max(0.0), s.tint[1].max(0.0), s.tint[2].max(0.0), 0.0] };
        queue.write_buffer(&self.prepare, 0, bytemuck::bytes_of(&gpu));
        let p = &self.pipelines;
        let groups = |size: u32| size.div_ceil(8);
        let pass = |encoder: &mut wgpu::CommandEncoder, label: &'static str, work: &mut dyn FnMut(&mut wgpu::ComputePass)| {
            let stamp = crate::profiling::gpu_pass(label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            work(&mut pass);
        };
        let face = self.face_size;
        pass(encoder, "Environment/Cube", &mut |pass| {
            pass.set_pipeline(&p.to_cube);
            pass.set_bind_group(0, &p.to_cube_group, &[]);
            pass.dispatch_workgroups(groups(face), groups(face), 6);
            pass.set_pipeline(&p.downsample);
            for (k, group) in p.downsample_groups.iter().enumerate() {
                let size = (face >> (k + 1)).max(1);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(groups(size), groups(size), 6);
            }
        });
        // the environment's mirror mip is the radiance's
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo { texture: &self.radiance, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyTextureInfo { texture: &self.environment, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::Extent3d { width: face, height: face, depth_or_array_layers: 6 },
        );
        pass(encoder, "Environment/Prefilter", &mut |pass| {
            pass.set_pipeline(&p.prefilter);
            for (k, group) in p.prefilter_groups.iter().enumerate() {
                let size = (face >> (k + 1)).max(1);
                pass.set_bind_group(0, group, &[]);
                pass.dispatch_workgroups(groups(size), groups(size), 6);
            }
        });
        pass(encoder, "Environment/SH", &mut |pass| {
            pass.set_pipeline(&p.project_sh);
            pass.set_bind_group(0, &p.project_sh_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });
        self.prepared = Some(self.settings);
    }

    /// The environment for its consumers: the SH (`SkyLighting`) and the cubemap. The handles stay
    /// the same for the environment's life: settings change what they hold, not them.
    pub fn lighting(&self) -> EnvironmentLighting {
        self.lighting.clone()
    }

    /// Face size of the cubemaps, texels.
    pub fn face_size(&self) -> u32 {
        self.face_size
    }

    /// The image's size as uploaded (halved while wider than the device holds).
    pub fn image_size(&self) -> (u32, u32) {
        self.image_size
    }

    /// The environment cubemap (6 layers; mip m GGX-prefiltered for roughness m / (mips - 1)).
    pub fn environment_texture(&self) -> &wgpu::Texture {
        &self.environment
    }

    /// The radiance cubemap (6 layers; box-filtered mips), the source of the prefiltering.
    pub fn radiance_texture(&self) -> &wgpu::Texture {
        &self.radiance
    }

    /// Bytes on the GPU: the image and the two cubemaps with their mips.
    pub fn memory_bytes(&self) -> u64 {
        let cube: u64 = (0..self.radiance.mip_level_count()).map(|m| ((self.face_size >> m).max(1) as u64).pow(2) * 6 * 8).sum();
        self.image_size.0 as u64 * self.image_size.1 as u64 * 8 + 2 * cube
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_round_and_clamp() {
        assert_eq!(f32_to_f16(0.0), 0);
        assert_eq!(f32_to_f16(1.0), 0x3c00);
        assert_eq!(f32_to_f16(0.5), 0x3800);
        assert_eq!(f32_to_f16(65504.0), 0x7bff);
        assert_eq!(f32_to_f16(1e9), 0x7bff);
        assert_eq!(f32_to_f16(-3.0), 0);
        assert_eq!(f32_to_f16(f32::NAN), 0);
        // the smallest subnormal, and 1 + 2^-10
        assert_eq!(f32_to_f16(2f32.powi(-24)), 1);
        assert_eq!(f32_to_f16(1.0 + 2f32.powi(-10)), 0x3c01);
    }

    #[test]
    fn halving_averages_four_pixels() {
        let image = HdrImage::from_pixels(2, 2, vec![[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 2.0, 4.0]]);
        let half = image.halved();
        assert_eq!((half.width, half.height), (1, 1));
        assert_eq!(half.pixels[0], [1.0, 1.0, 1.0]);
    }
}
