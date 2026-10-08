use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

const WGSL: &str = include_str!("../../shaders/ssao.wgsl");

/// The most hemisphere samples a pixel takes (the kernel's size).
const MAX_KERNEL: usize = 32;

pub struct SSAOOptions {
    /// World-space sampling radius around each point.
    pub radius: f32,
    /// Depth bias against self-occlusion.
    pub bias: f32,
    /// Hemisphere samples a pixel, 1 to 32.
    pub kernel_size: u32,
    /// How much of the occlusion darkens the colour (1: all of it).
    pub strength: f32,
}

impl Default for SSAOOptions {
    fn default() -> Self {
        Self { radius: 0.5, bias: 0.025, kernel_size: 16, strength: 1.0 }
    }
}

/// `SSAOParams` in ssao.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SSAOParams {
    proj: [f32; 16],
    inv_proj: [f32; 16],
    width: f32,
    height: f32,
    radius: f32,
    bias: f32,
    kernel_size: u32,
    strength: f32,
    _pad: [f32; 2],
}

/// Screen-space ambient occlusion from depth alone, the TS engine's `SSAOEffect`: view positions
/// and normals are rebuilt from the depth buffer (no normal target needed), a hemisphere of
/// samples around each point is projected back to the screen and compared with the depth there,
/// and the colour is multiplied by what is left unoccluded. It darkens the whole lit colour,
/// direct light included, as a look rather than a physical term; for sky occlusion of materials
/// lit by the sky's SH, see `ScreenSpaceGIEffect::ambient_occlusion`. Put it first in the chain.
pub struct SSAOEffect {
    pub options: SSAOOptions,
    pipeline: Option<wgpu::ComputePipeline>,
    bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    kernel_buf: Option<wgpu::Buffer>,
}

impl SSAOEffect {
    pub fn new(options: SSAOOptions) -> Self {
        Self { options, pipeline: None, bgl: None, params_buf: None, kernel_buf: None }
    }
}

/// The hemisphere kernel: `MAX_KERNEL` points in the +z half of the unit ball, more of them near
/// the centre (scaled by lerp(0.1, 1, (i / n)²)), from a fixed seed.
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
        let cos_theta = random();
        let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();
        let t = i as f32 / MAX_KERNEL as f32;
        let scale = 0.1 + t * t * 0.9;
        [phi.cos() * sin_theta * scale, phi.sin() * sin_theta * scale, cos_theta * scale, 0.0]
    })
}

impl PostProcessingEffect for SSAOEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.pipeline.is_some() {
            return;
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("SSAO/Shader"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let uniform = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SSAO/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                uniform(3),
                uniform(4),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("SSAO/Layout"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        self.pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SSAO/Pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        }));
        self.params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SSAO/Params"),
            size: std::mem::size_of::<SSAOParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.kernel_buf = Some(wgpu::util::DeviceExt::create_buffer_init(
            device,
            &wgpu::util::BufferInitDescriptor { label: Some("SSAO/Kernel"), contents: bytemuck::cast_slice(&kernel()), usage: wgpu::BufferUsages::UNIFORM },
        ));
        self.bgl = Some(bgl);
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
        let (Some(pipeline), Some(bgl), Some(params_buf), Some(kernel_buf)) = (&self.pipeline, &self.bgl, &self.params_buf, &self.kernel_buf) else { return };
        let o = &self.options;
        let params = SSAOParams {
            proj: camera.projection_matrix.to_cols_array(),
            inv_proj: camera.projection_matrix.inverse().to_cols_array(),
            width: width as f32,
            height: height as f32,
            radius: o.radius,
            bias: o.bias,
            kernel_size: o.kernel_size.clamp(1, MAX_KERNEL as u32),
            strength: o.strength,
            _pad: [0.0; 2],
        };
        queue.write_buffer(params_buf, 0, bytemuck::bytes_of(&params));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SSAO/BindGroup"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 3, resource: params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: kernel_buf.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("SSAO"),
            timestamp_writes: crate::profiling::gpu_pass("SSAO").as_ref().map(crate::profiling::PassStamp::compute),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}
    fn destroy(&mut self) {
        self.pipeline = None;
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

    /// ssao.wgsl validates, and its SSAOParams is the 160 bytes the effect writes.
    #[test]
    fn shaders_validate_and_the_params_match() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "SSAOParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<SSAOParams>());
        assert_eq!(span, 160);
    }

    /// The kernel lies in the +z hemisphere of the unit ball, nearer the centre first.
    #[test]
    fn kernel_is_a_hemisphere() {
        let k = kernel();
        for (i, p) in k.iter().enumerate() {
            let len = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
            let t = i as f32 / MAX_KERNEL as f32;
            assert!(p[2] >= 0.0, "{i}: {p:?}");
            assert!(len <= 0.1 + t * t * 0.9 + 1e-5, "{i}: {len}");
        }
    }
}
