use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};

/// Parameters uploaded per A-trous spatial denoise iteration.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SpatialParams {
    step_size: u32,
    sigma_depth: f32,
    sigma_normal: f32,
    sigma_lum: f32,
    width: u32,
    height: u32,
    use_svgf: u32,
    _pad: u32,
}

/// A-trous wavelet spatial denoiser with variance-guided edge stopping.
///
/// Runs a configurable number of iterations (default 3) of a 5×5 a-trous
/// filter, doubling the step size each iteration (`stepSize = 2^i`). The
/// filter uses edge-stopping weights based on depth, normal, and luminance
/// (with optional SVGF variance guidance from a moments texture).
///
/// Ping-pongs between two scratch textures (`scratch_a` / `scratch_b`).
/// Iteration 0 reads from the provided input and writes to `scratch_a`.
/// Subsequent iterations alternate reads/writes between the scratch pair.
pub struct SpatialDenoise {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    scratch_a: Option<wgpu::Texture>,
    scratch_a_view: Option<wgpu::TextureView>,
    scratch_b: Option<wgpu::Texture>,
    scratch_b_view: Option<wgpu::TextureView>,
    /// Number of A-trous iterations (default 3).
    pub iterations: u32,
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl SpatialDenoise {
    /// Create the spatial denoise pipeline and parameter buffer.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        let shader_src = include_str!("shaders/denoise-spatial.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SpatialDenoise Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpatialDenoise BGL"),
            entries: &[
                // binding 0: inputGI (texture_2d<f32>)
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
                // binding 1: outputGI (storage texture, rgba16float, write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // binding 2: depthTex (texture_depth_2d)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // binding 3: normalTex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
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
                            std::mem::size_of::<SpatialParams>() as u64,
                        ),
                    },
                    count: None,
                },
                // binding 5: momentsTex (texture_2d<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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
            label: Some("SpatialDenoise PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SpatialDenoise Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("spatial_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SpatialDenoise Params"),
            size: std::mem::size_of::<SpatialParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline,
            bgl,
            params_buf,
            scratch_a: None,
            scratch_a_view: None,
            scratch_b: None,
            scratch_b_view: None,
            iterations: 3,
            width: 0,
            height: 0,
            device,
            queue,
        }
    }

    /// Recreate scratch textures when the viewport changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        let create_scratch = |label: &str| -> (wgpu::Texture, wgpu::TextureView) {
            let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };

        let (a_tex, a_view) = create_scratch("SpatialDenoise/ScratchA");
        let (b_tex, b_view) = create_scratch("SpatialDenoise/ScratchB");

        self.scratch_a = Some(a_tex);
        self.scratch_a_view = Some(a_view);
        self.scratch_b = Some(b_tex);
        self.scratch_b_view = Some(b_view);
    }

    /// Run N iterations of the A-trous spatial filter, ping-ponging between
    /// scratch textures. Returns a reference to the texture view containing
    /// the final denoised result.
    ///
    /// Iteration layout (for 3 iterations):
    ///   - iter 0: input    -> scratch_a  (step=1)
    ///   - iter 1: scratch_a -> scratch_b (step=2)
    ///   - iter 2: scratch_b -> scratch_a (step=4)
    ///   Result is in scratch_a.
    pub fn denoise(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        moments_view: &wgpu::TextureView,
    ) -> &wgpu::TextureView {
        let scratch_a_view = self
            .scratch_a_view
            .as_ref()
            .expect("SpatialDenoise::resize() must be called before denoise()");
        let scratch_b_view = self
            .scratch_b_view
            .as_ref()
            .expect("SpatialDenoise::resize() must be called before denoise()");

        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        for i in 0..self.iterations {
            let step_size = 1u32 << i;

            let params = SpatialParams {
                step_size,
                sigma_depth: 1.0,
                sigma_normal: 128.0,
                sigma_lum: 4.0,
                width: self.width,
                height: self.height,
                use_svgf: 1,
                _pad: 0,
            };
            self.queue
                .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

            // Determine read/write views for this iteration.
            let (read_view, write_view) = match i {
                0 => (input_view, scratch_a_view),
                _ if i % 2 == 1 => (scratch_a_view, scratch_b_view),
                _ => (scratch_b_view, scratch_a_view),
            };

            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("SpatialDenoise BG iter {i}")),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(read_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(write_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(normal_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: self.params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::TextureView(moments_view),
                    },
                ],
            });

            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&format!("SpatialDenoise iter {i} step={step_size}")),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Return the view that was last written to.
        // iter 0 -> scratch_a, iter 1 -> scratch_b, iter 2 -> scratch_a, ...
        if self.iterations == 0 {
            // No filtering — caller should use input directly. We return
            // scratch_a as a fallback (it will be stale / uninitialised).
            scratch_a_view
        } else if (self.iterations - 1) % 2 == 0 {
            // Last write was to scratch_a (iterations 1, 3, 5, ...)
            scratch_a_view
        } else {
            // Last write was to scratch_b (iterations 2, 4, 6, ...)
            scratch_b_view
        }
    }
}
