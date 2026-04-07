use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};

/// Uniform parameters uploaded per composite dispatch.
///
/// Layout must match the WGSL `CompositeParams` struct exactly:
///   u32  width        (4 bytes)
///   u32  height       (4 bytes)
///   u32  rasterDirect (4 bytes)
///   u32  _pad         (4 bytes)
/// Total: 16 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct CompositeParams {
    width: u32,
    height: u32,
    raster_direct: u32,
    _pad: u32,
}

/// Composite pass — combines path-traced GI with the rasterized scene.
///
/// Computes: `output = GI * albedo + direct + emissive` (raster-direct mode)
/// or `output = GI * albedo + emissive` (full path-tracer mode), followed by
/// Reinhard tone mapping.
///
/// Bindings (group 0):
///   0 — inputTex    (texture_2d<f32>) rasterized direct-light buffer
///   1 — denoisedGI  (texture_2d<f32>) denoised indirect GI (sampled with giSampler)
///   2 — albedoTex   (texture_2d<f32>) GBuffer albedo
///   3 — outputTex   (texture_storage_2d<rgba16float, write>) HDR output
///   4 — params      (uniform CompositeParams)
///   5 — giSampler   (sampler, filtering)
///   6 — emissiveTex (texture_2d<f32>) GBuffer emissive
pub struct Compositor {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    sampler: wgpu::Sampler,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl Compositor {
    /// Create the composite compute pipeline and associated resources.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        let shader_src = include_str!("shaders/composite.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Compositor Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compositor BGL"),
            entries: &[
                // binding 0: inputTex (rasterized direct light, texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 1: denoisedGI (texture_2d<f32>, filterable for giSampler)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 2: albedoTex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: outputTex (storage texture, rgba16float, write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 4: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(
                            std::mem::size_of::<CompositeParams>() as u64,
                        ),
                    },
                    count: None,
                },
                // binding 5: giSampler (filtering sampler for denoisedGI)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // binding 6: emissiveTex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compositor PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Compositor Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("composite_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Compositor Params"),
            size: std::mem::size_of::<CompositeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Compositor GI Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        Self {
            pipeline,
            bgl,
            params_buf,
            sampler,
            device,
            queue,
        }
    }

    /// Dispatch the composite compute shader.
    ///
    /// - `gi_view`     — denoised GI texture (indirect lighting)
    /// - `albedo_view` — GBuffer albedo
    /// - `direct_view` — rasterized direct-light buffer (or a dummy when
    ///                   `raster_direct` is `false`)
    /// - `emissive_view` — GBuffer emissive
    /// - `output_view` — rgba16float storage texture to write the final HDR image
    /// - `raster_direct` — when `true`, blend rasterized direct light with GI indirect;
    ///                     when `false`, use GI as complete radiance
    pub fn composite(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        gi_view: &wgpu::TextureView,
        albedo_view: &wgpu::TextureView,
        direct_view: &wgpu::TextureView,
        emissive_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        width: u32,
        height: u32,
        raster_direct: bool,
    ) {
        let params = CompositeParams {
            width,
            height,
            raster_direct: raster_direct as u32,
            _pad: 0,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compositor BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(direct_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(gi_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::TextureView(emissive_view),
                },
            ],
        });

        let wg_x = (width + 7) / 8;
        let wg_y = (height + 7) / 8;

        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compositor"),
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}
