//! A stylised contact darkening over the lit image (`contact_ao.wgsl`). It began as the
//! caustics-box demo's ambient occlusion, which the showroom's room
//! (`rust/kansei-wasm/demos/motion-matching`) and the Kansei Marbles game
//! (Siroko/kansei-marbles, `crates/marble/src/render/ao.rs`, which keeps what surfaces emit out of
//! the darkening) carried on.

use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const CONTACT_AO_WGSL: &str = include_str!("../../shaders/contact_ao.wgsl");

/// The most samples a pixel takes.
const MAX_SAMPLES: u32 = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactAoOptions {
    /// How far the occlusion reaches, in world units: a few times the size of the creases and
    /// gaps that should darken (0.6 in a room in metres, 0.06 on a tabletop of marbles).
    /// [`ContactAoOptions::for_scale`] gives the defaults in other units.
    pub radius: f32,
    /// The exponent on the unoccluded share: 0 leaves the image as it is, higher darkens the
    /// creases harder.
    pub strength: f32,
    /// How tight the darkening hugs the contacts: the exponent on each sample's weight by its
    /// reach (0 every sample alike, higher the near ones count most).
    pub falloff: f32,
    /// Samples a pixel, 1 to 32 (the pattern turns every frame for TAA to resolve).
    pub samples: u32,
    /// How much nearer than a sample a surface must be to occlude it, as a share of `radius`.
    pub bias: f32,
}

impl Default for ContactAoOptions {
    /// The showroom's room, in metres: a reach of 0.6, strength 2.5, falloff 1, 12 samples and a
    /// bias of 1% of the reach.
    fn default() -> Self {
        Self { radius: 0.6, strength: 2.5, falloff: 1.0, samples: 12, bias: 0.01 }
    }
}

impl ContactAoOptions {
    /// The defaults for a world of `units_per_metre` units a metre (100 in centimetres): the
    /// reach scaled, the rest as it is (the bias is a share of the reach already).
    pub fn for_scale(units_per_metre: f32) -> Self {
        let d = Self::default();
        Self { radius: d.radius * units_per_metre, ..d }
    }
}

/// `ContactAoParams` in contact_ao.wgsl (240 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ContactAoParams {
    view: [f32; 16],
    proj: [f32; 16],
    inv_proj: [f32; 16],
    size: [f32; 2],
    radius: f32,
    strength: f32,
    falloff: f32,
    samples: u32,
    frame: u32,
    debug: u32,
    white: f32,
    bias: f32,
    _pad: [f32; 2],
}

struct Gpu {
    params: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

/// Contact ambient occlusion: a game's darkening in creases and where objects meet, multiplied
/// into the lit colour. Each pixel casts a hemisphere of samples round its GBuffer normal, up to
/// [`ContactAoOptions::radius`] world units out, against the depth buffer; the share left open,
/// to the power [`ContactAoOptions::strength`], scales the light the surface reflects. What it
/// emits (the GBuffer's emissive target) keeps its glow, and the sky, glass and the fluid's
/// surface pass through.
///
/// Unlike [`super::SSAOEffect`], which takes the occluded share out of the ambient light alone
/// and goes first in the chain, this darkens all the reflected light, direct and indirect: a
/// look rather than a measure, for scenes lit mostly by direct light whose GI barely shows
/// contact. Put it after the GI, the reflections and the glass ([`crate::rt::RtReflectionsEffect`],
/// [`super::ScreenSpaceGlassEffect`]) and before the fog, the distance fade and TAA, which
/// resolves its noise. It takes one pass ("ContactAO" in the profiler).
pub struct ContactAoEffect {
    pub options: ContactAoOptions,
    /// Debug view: the occlusion alone, `debug_white` where nothing occludes, in place of the
    /// image.
    pub show_occlusion: bool,
    /// The radiance (cd/m²) the debug view shows where nothing occludes: about mid grey after
    /// the tone map at `0.5 / exposure`.
    pub debug_white: f32,
    frame: u32,
    gpu: Option<Gpu>,
}

impl ContactAoEffect {
    pub fn new(options: ContactAoOptions) -> Self {
        Self { options, show_occlusion: false, debug_white: 1.0, frame: 0, gpu: None }
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        CONTACT_AO_WGSL
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ContactAO/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: compute,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: compute,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                texture(2),
                texture(3),
                texture(4),
                texture(5),
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: compute,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: GBuffer::COLOR_FORMAT, view_dimension: wgpu::TextureViewDimension::D2 },
                    count: None,
                },
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ContactAO"), source: wgpu::ShaderSource::Wgsl(CONTACT_AO_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ContactAO"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ContactAO"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ContactAO/Params"),
            size: std::mem::size_of::<ContactAoParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.gpu = Some(Gpu { params, pipeline, bgl });
    }

    /// The parameters the shader gets for the next frame: the options clamped to what it takes.
    fn params(&self, camera: &Camera, width: u32, height: u32) -> ContactAoParams {
        let o = &self.options;
        let radius = o.radius.max(f32::MIN_POSITIVE);
        let proj = camera.projection_matrix.to_glam();
        ContactAoParams {
            view: camera.view_matrix.to_glam().to_cols_array(),
            proj: proj.to_cols_array(),
            inv_proj: proj.inverse().to_cols_array(),
            size: [width as f32, height as f32],
            radius,
            strength: o.strength.max(0.0),
            falloff: o.falloff.max(0.0),
            samples: o.samples.clamp(1, MAX_SAMPLES),
            frame: self.frame,
            debug: self.show_occlusion as u32,
            white: self.debug_white,
            bias: o.bias.max(0.0) * radius,
            _pad: [0.0; 2],
        }
    }
}

impl PostProcessingEffect for ContactAoEffect {
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
        self.frame = self.frame.wrapping_add(1);
        let params = self.params(camera, width, height);
        let gpu = self.gpu.as_ref().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let tex = wgpu::BindingResource::TextureView;
        let resources = [gpu.params.as_entire_binding(), tex(depth), tex(&gbuffer.normal_view), tex(&gbuffer.albedo_view), tex(&gbuffer.emissive_view), tex(input), tex(output)];
        let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
        let group = crate::renderers::cached_bind_group(device, &wgpu::BindGroupDescriptor { label: Some("ContactAO/BindGroup"), layout: &gpu.bgl, entries: &entries });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("ContactAO"),
            timestamp_writes: crate::profiling::gpu_pass("ContactAO").as_ref().map(crate::profiling::PassStamp::compute),
        });
        pass.set_pipeline(&gpu.pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}
    fn destroy(&mut self) {
        self.gpu = None;
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shader parses and validates with naga, and its ContactAoParams is the 240 bytes the
    /// effect writes.
    #[test]
    fn shaders_validate_and_the_params_match() {
        let code = ContactAoEffect::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "ContactAoParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<ContactAoParams>());
        assert_eq!(std::mem::size_of::<ContactAoParams>(), 240);
    }

    /// The defaults are the showroom's, `for_scale` scales only the reach, and the parameters
    /// clamp to what the shader takes, the bias following the reach.
    #[test]
    fn defaults_scale_and_clamp() {
        let d = ContactAoOptions::default();
        assert_eq!(d, ContactAoOptions { radius: 0.6, strength: 2.5, falloff: 1.0, samples: 12, bias: 0.01 });
        let cm = ContactAoOptions::for_scale(100.0);
        assert!((cm.radius - 60.0).abs() < 1e-4);
        assert_eq!(ContactAoOptions { radius: d.radius, ..cm }, d);

        let mut camera = Camera::new(45.0, 0.1, 100.0, 1.0);
        camera.update_projection_matrix();
        let mut effect = ContactAoEffect::new(ContactAoOptions { radius: 0.06, strength: -1.0, falloff: -2.0, samples: 0, bias: 0.01 });
        let p = effect.params(&camera, 64, 32);
        assert_eq!((p.size, p.strength, p.falloff, p.samples, p.debug), ([64.0, 32.0], 0.0, 0.0, 1, 0));
        assert!((p.bias - 0.0006).abs() < 1e-7);
        effect.options.samples = 100;
        effect.options.radius = 0.0;
        effect.show_occlusion = true;
        let p = effect.params(&camera, 64, 32);
        assert_eq!((p.samples, p.debug), (MAX_SAMPLES, 1));
        assert!(p.radius > 0.0);
    }
}
