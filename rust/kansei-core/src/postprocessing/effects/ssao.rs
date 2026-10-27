use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const COMMON_WGSL: &str = include_str!("../../shaders/ssao_common.wgsl");
const AO_WGSL: &str = include_str!("../../shaders/ssao.wgsl");
const BLUR_WGSL: &str = include_str!("../../shaders/ssao_blur.wgsl");
const BLUR_X_WGSL: &str = include_str!("../../shaders/ssao_blur_x.wgsl");
const COMPOSITE_WGSL: &str = include_str!("../../shaders/ssao_composite.wgsl");

fn ao_source() -> String {
    [COMMON_WGSL, AO_WGSL].concat()
}

fn blur_source() -> String {
    [COMMON_WGSL, BLUR_WGSL, BLUR_X_WGSL].concat()
}

fn composite_source() -> String {
    [COMMON_WGSL, crate::atmosphere::SKY_LIGHTING_WGSL, BLUR_WGSL, COMPOSITE_WGSL].concat()
}

/// The most hemisphere samples a pixel takes (the kernel's size).
const MAX_KERNEL: usize = 32;

/// The sampling radius on screen at most, as a share of the image's height: past it the radius
/// shrinks (a close-up's samples would otherwise spread over the whole screen).
const MAX_RADIUS_SCREEN: f32 = 0.1;

/// The occlusion targets' format: the share left open, and the point's distance for the blur.
const AO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg32Float;

/// The ambient light the materials add to the colour, which [`SSAOEffect`] takes the occluded
/// share of: the occlusion darkens it alone, never the direct light or the emission.
#[derive(Clone)]
pub enum SSAOAmbient {
    /// The albedo times a hemisphere of sky blended by the normal's height,
    /// `mix(down, up, n.y * 0.5 + 0.5)`: what `Material::standard_lit` adds from its `sky_up`
    /// and `sky_down` (give the same radiances). `up == down` for a constant ambient (albedo x
    /// ambient).
    Hemisphere { up: [f32; 3], down: [f32; 3] },
    /// The albedo times the irradiance of the sky's SH around the normal, over pi: materials lit
    /// by the sky as `atmosphere::SKY_LIGHTING_WGSL`'s `skyIrradiance` (a `SkyLighting` uniform:
    /// `SkyAtmosphereBindings::sky_lighting`, or `gi::gradient_sky_lighting` written to a
    /// buffer).
    Sky(wgpu::Buffer),
    /// The whole colour less the GBuffer's emissive part: unlit materials, whose colour is all
    /// ambient (a texture shown as it is, with no lights).
    Unlit,
}

#[derive(Clone)]
pub struct SSAOOptions {
    /// Metres around each point searched for occluders: a few times the size of the creases
    /// that should darken (centimetres on a tabletop, half a metre in a room).
    pub radius: f32,
    /// How much nearer than a sample a surface must be to occlude it, per metre of the point's
    /// distance from the camera (a slanted surface's own depth steps grow with distance).
    pub bias: f32,
    /// Hemisphere samples a pixel, 1 to 32.
    pub kernel_size: u32,
    /// How much of the occluded ambient is taken out (1: all of it).
    pub strength: f32,
    /// The ambient light it occludes.
    pub ambient: SSAOAmbient,
}

impl SSAOOptions {
    /// The defaults (0.5 m, a bias of 3 mm a metre, 16 samples, full strength) occluding
    /// `ambient`.
    pub fn new(ambient: SSAOAmbient) -> Self {
        Self { radius: 0.5, bias: 0.003, kernel_size: 16, strength: 1.0, ambient }
    }
}

/// `SSAOParams` in ssao_common.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SSAOParams {
    proj: [f32; 16],
    inv_proj: [f32; 16],
    view: [f32; 16],
    inv_view: [f32; 16],
    sky_up: [f32; 4],
    sky_down: [f32; 4],
    size: [f32; 2],
    radius: f32,
    bias: f32,
    kernel_size: u32,
    strength: f32,
    max_radius_px: f32,
    ambient: u32,
    debug: u32,
    _pad: [u32; 3],
}

struct Targets {
    width: u32,
    height: u32,
    /// The occlusion, then blurred across.
    ao: [wgpu::TextureView; 2],
}

struct Gpu {
    params: wgpu::Buffer,
    kernel: wgpu::Buffer,
    no_sky: wgpu::Buffer,
    ao: wgpu::ComputePipeline,
    ao_bgl: wgpu::BindGroupLayout,
    blur: wgpu::ComputePipeline,
    blur_bgl: wgpu::BindGroupLayout,
    composite: wgpu::ComputePipeline,
    composite_bgl: wgpu::BindGroupLayout,
    targets: Option<Targets>,
}

/// Screen-space ambient occlusion: a hemisphere of samples around each point's normal (the
/// GBuffer's, or one rebuilt from depth where the material wrote none), turned per pixel and
/// projected back to the screen against the depth there, blurred by depth, then taken out of
/// the ambient light alone ([`SSAOAmbient`]): flat open surfaces keep their light, creases and
/// contacts lose some of their ambient, and the direct light and emission stay as they are.
///
/// Put it first in the chain. The GI effects occlude the light they add themselves
/// (`ScreenSpaceGIEffect`'s visibility, the voxel cones and `sdf_ao`, the RT GI's rays): with
/// one that replaces the materials' ambient, leave this off, or give it only the ambient that
/// effect leaves.
pub struct SSAOEffect {
    pub options: SSAOOptions,
    /// Debug view: output the occlusion (white open, black occluded) in place of the image.
    pub show_occlusion: bool,
    gpu: Option<Gpu>,
}

impl SSAOEffect {
    pub fn new(options: SSAOOptions) -> Self {
        Self { options, show_occlusion: false, gpu: None }
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> Vec<(&'static str, String)> {
        vec![("ssao", ao_source()), ("ssao_blur", blur_source()), ("ssao_composite", composite_source())]
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let texture = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let depth = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
            count: None,
        };
        let storage = |binding, format| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: wgpu::TextureViewDimension::D2 },
            count: None,
        };
        let bgl = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        // bindings 0-2 of every pass: the params, the GBuffer's depth and normal
        let ao_bgl = bgl("SSAO/BGL", &[uniform(0), depth(1), texture(2), uniform(3), storage(4, AO_FORMAT)]);
        let blur_bgl = bgl("SSAO/BlurBGL", &[uniform(0), depth(1), texture(2), texture(3), storage(4, AO_FORMAT)]);
        let composite_bgl = bgl(
            "SSAO/CompositeBGL",
            &[uniform(0), depth(1), texture(2), texture(3), texture(4), texture(5), texture(6), uniform(7), storage(8, GBuffer::COLOR_FORMAT)],
        );
        let pipeline = |label: &str, code: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[Some(layout)], immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        use wgpu::util::DeviceExt;
        self.gpu = Some(Gpu {
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSAO/Params"),
                size: std::mem::size_of::<SSAOParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            kernel: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("SSAO/Kernel"), contents: bytemuck::cast_slice(&kernel()), usage: wgpu::BufferUsages::UNIFORM }),
            no_sky: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("SSAO/NoSky"),
                size: std::mem::size_of::<crate::gi::SkyLightingData>() as u64,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            }),
            ao: pipeline("SSAO", &ao_source(), &ao_bgl),
            ao_bgl,
            blur: pipeline("SSAO/Blur", &blur_source(), &blur_bgl),
            blur_bgl,
            composite: pipeline("SSAO/Composite", &composite_source(), &composite_bgl),
            composite_bgl,
            targets: None,
        });
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let gpu = self.gpu.as_mut().unwrap();
        if gpu.targets.as_ref().is_some_and(|t| t.width == width && t.height == height) {
            return;
        }
        let target = |label: &str| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: AO_FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        gpu.targets = Some(Targets { width, height, ao: [target("SSAO/Occlusion"), target("SSAO/OcclusionBlurred")] });
    }
}

/// The hemisphere kernel: `MAX_KERNEL` points in the +z half of the unit ball, more of them near
/// the centre (scaled by lerp(0.1, 1, (i / n)²)) and none within 8.6 degrees of the base, whose
/// samples would lie in the surface's own plane, from a fixed seed.
fn kernel() -> [[f32; 4]; MAX_KERNEL] {
    let mut seed = 0x9e37_79b9u32;
    let mut random = move || {
        // xorshift32
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        (seed >> 8) as f32 / (1u32 << 24) as f32
    };
    std::array::from_fn(|i| {
        let phi = random() * std::f32::consts::TAU;
        let cos_theta = 0.15 + 0.85 * random();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let t = i as f32 / MAX_KERNEL as f32;
        let scale = 0.1 + t * t * 0.9;
        [phi.cos() * sin_theta * scale, phi.sin() * sin_theta * scale, cos_theta * scale, 0.0]
    })
}

impl PostProcessingEffect for SSAOEffect {
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
        let gpu = self.gpu.as_ref().unwrap();
        let t = gpu.targets.as_ref().unwrap();
        let o = &self.options;
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let (ambient, [up, down]) = match &o.ambient {
            SSAOAmbient::Hemisphere { up, down } => (0, [*up, *down]),
            SSAOAmbient::Sky(_) => (1, [[0.0; 3]; 2]),
            SSAOAmbient::Unlit => (2, [[0.0; 3]; 2]),
        };
        let params = SSAOParams {
            proj: proj.to_cols_array(),
            inv_proj: proj.inverse().to_cols_array(),
            view: view.to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            sky_up: [up[0], up[1], up[2], 0.0],
            sky_down: [down[0], down[1], down[2], 0.0],
            size: [width as f32, height as f32],
            radius: o.radius.max(1e-4),
            bias: o.bias.max(0.0),
            kernel_size: o.kernel_size.clamp(1, MAX_KERNEL as u32),
            strength: o.strength.max(0.0),
            max_radius_px: height as f32 * MAX_RADIUS_SCREEN,
            ambient,
            debug: self.show_occlusion as u32,
            _pad: [0; 3],
        };
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        let sky = match &o.ambient {
            SSAOAmbient::Sky(buffer) => buffer,
            _ => &gpu.no_sky,
        };

        let tex = wgpu::BindingResource::TextureView;
        let p = || gpu.params.as_entire_binding();
        let ao = group(device, "SSAO/BindGroup", &gpu.ao_bgl, vec![p(), tex(depth), tex(&gbuffer.normal_view), gpu.kernel.as_entire_binding(), tex(&t.ao[0])]);
        let blur = group(device, "SSAO/BlurBindGroup", &gpu.blur_bgl, vec![p(), tex(depth), tex(&gbuffer.normal_view), tex(&t.ao[0]), tex(&t.ao[1])]);
        let composite = group(
            device,
            "SSAO/CompositeBindGroup",
            &gpu.composite_bgl,
            vec![p(), tex(depth), tex(&gbuffer.normal_view), tex(&t.ao[1]), tex(input), tex(&gbuffer.albedo_view), tex(&gbuffer.emissive_view), sky.as_entire_binding(), tex(output)],
        );
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SSAO"),
            timestamp_writes: crate::profiling::gpu_pass("SSAO").as_ref().map(crate::profiling::PassStamp::compute),
        });
        let groups = (width.div_ceil(8), height.div_ceil(8));
        for (pipeline, bind_group) in [(&gpu.ao, &ao), (&gpu.blur, &blur), (&gpu.composite, &composite)] {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, 1);
        }
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

/// A bind group of `resources` at bindings 0, 1, 2...
fn group(device: &wgpu::Device, label: &str, layout: &wgpu::BindGroupLayout, resources: Vec<wgpu::BindingResource>) -> wgpu::BindGroup {
    let entries: Vec<_> = resources.into_iter().enumerate().map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource }).collect();
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(label), layout, entries: &entries })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The SSAO shaders parse and validate with naga, and their SSAOParams is the 336 bytes the
    /// effect writes.
    #[test]
    fn shaders_validate_and_the_params_match() {
        for (name, code) in SSAOEffect::shader_sources() {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            let span = module
                .types
                .iter()
                .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                    (Some(n), naga::TypeInner::Struct { span, .. }) if n == "SSAOParams" => Some(*span as usize),
                    _ => None,
                })
                .unwrap();
            assert_eq!(span, std::mem::size_of::<SSAOParams>(), "{name}");
        }
        assert_eq!(std::mem::size_of::<SSAOParams>(), 336);
    }

    /// The kernel lies in the +z hemisphere of the unit ball, nearer the centre first, and
    /// clear of the base plane.
    #[test]
    fn kernel_is_a_hemisphere() {
        let k = kernel();
        for (i, p) in k.iter().enumerate() {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let t = i as f32 / MAX_KERNEL as f32;
            assert!(p[2] >= 0.15 * len - 1e-6, "{i}: {p:?}");
            assert!(len <= 0.1 + t * t * 0.9 + 1e-5, "{i}: {len}");
        }
    }

    /// The normal rebuilt from depth, as ssao_common.wgsl's `ssaoViewNormal` takes it (uv down
    /// the screen, the right and upper neighbours), faces the camera: +z for a wall facing it,
    /// and up and toward the camera for a floor seen from above. The old `cross(right - left,
    /// down - up)` pointed into the surface.
    #[test]
    fn the_rebuilt_normal_faces_the_camera() {
        let (w, h) = (64.0f32, 48.0f32);
        let proj = glam::Mat4::perspective_rh(60f32.to_radians(), w / h, 0.1, 100.0);
        let inv_proj = proj.inverse();
        let view_pos = |px: glam::Vec2, depth: f32| {
            let uv = (px + 0.5) / glam::Vec2::new(w, h);
            let p = inv_proj * glam::Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
            p.truncate() / p.w
        };
        // the depth of a view-space plane (normal n through point q) under each pixel
        let depth_of = |n: glam::Vec3, q: glam::Vec3| {
            move |px: glam::Vec2| {
                let uv = (px + 0.5) / glam::Vec2::new(w, h);
                let far = inv_proj * glam::Vec4::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 1.0, 1.0);
                let dir = (far.truncate() / far.w).normalize();
                let t = n.dot(q) / n.dot(dir);
                let clip = proj * (dir * t).extend(1.0);
                clip.z / clip.w
            }
        };
        let floor_n = glam::Vec3::new(0.0, 1.0, 0.3).normalize();
        for (n, q) in [(glam::Vec3::Z, glam::Vec3::new(0.0, 0.0, -5.0)), (floor_n, glam::Vec3::new(0.0, -1.0, -5.0))] {
            let depth = depth_of(n, q);
            let at = |px: glam::Vec2| view_pos(px, depth(px));
            let c = glam::Vec2::new(32.0, 24.0);
            let p = at(c);
            let dx = at(c + glam::Vec2::X) - p;
            let dy = at(c - glam::Vec2::Y) - p;
            let rebuilt = dx.cross(dy).normalize();
            assert!(rebuilt.dot(n) > 0.999, "{rebuilt} for {n}");
            assert!(rebuilt.dot(-p) > 0.0);
            let old = (at(c + glam::Vec2::X) - at(c - glam::Vec2::X)).cross(at(c + glam::Vec2::Y) - at(c - glam::Vec2::Y)).normalize();
            assert!(old.dot(n) < -0.999, "the old reconstruction: {old}");
        }
    }
}
