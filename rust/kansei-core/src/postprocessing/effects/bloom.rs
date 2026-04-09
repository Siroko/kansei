use crate::cameras::Camera;
use crate::renderers::GBuffer;
use crate::postprocessing::PostProcessingEffect;

pub struct BloomOptions {
    pub threshold: f32,
    pub knee: f32,
    pub intensity: f32,
    pub radius: f32,
}

impl Default for BloomOptions {
    fn default() -> Self {
        Self { threshold: 1.0, knee: 0.1, intensity: 0.8, radius: 1.0 }
    }
}

const MIP_COUNT: usize = 6;

pub struct BloomEffect {
    pub options: BloomOptions,
    // Pipelines
    downsample_pipeline: Option<wgpu::ComputePipeline>,
    upsample_pipeline: Option<wgpu::ComputePipeline>,
    composite_pipeline: Option<wgpu::ComputePipeline>,
    // Bind group layouts
    downsample_bgl: Option<wgpu::BindGroupLayout>,
    upsample_bgl: Option<wgpu::BindGroupLayout>,
    composite_bgl: Option<wgpu::BindGroupLayout>,
    // Mip chain textures (for downsample output / upsample input)
    mip_textures: Vec<wgpu::Texture>,
    mip_views: Vec<wgpu::TextureView>,
    // Upsample output textures (separate to avoid read-write hazard)
    upsample_textures: Vec<wgpu::Texture>,
    upsample_views: Vec<wgpu::TextureView>,
    // Param buffers (one per pass)
    downsample_params: Vec<wgpu::Buffer>,
    upsample_params: Vec<wgpu::Buffer>,
    composite_params: Option<wgpu::Buffer>,
    // Sampler for upsample/composite
    linear_sampler: Option<wgpu::Sampler>,
    // Dimensions
    width: u32,
    height: u32,
    initialized: bool,
}

impl BloomEffect {
    pub fn new(options: BloomOptions) -> Self {
        Self {
            options,
            downsample_pipeline: None,
            upsample_pipeline: None,
            composite_pipeline: None,
            downsample_bgl: None,
            upsample_bgl: None,
            composite_bgl: None,
            mip_textures: Vec::new(),
            mip_views: Vec::new(),
            upsample_textures: Vec::new(),
            upsample_views: Vec::new(),
            downsample_params: Vec::new(),
            upsample_params: Vec::new(),
            composite_params: None,
            linear_sampler: None,
            width: 0,
            height: 0,
            initialized: false,
        }
    }

    fn create_mip_textures(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.mip_textures.clear();
        self.mip_views.clear();
        self.upsample_textures.clear();
        self.upsample_views.clear();

        let mut w = width;
        let mut h = height;
        for i in 0..MIP_COUNT {
            w = (w / 2).max(1);
            h = (h / 2).max(1);

            let mip_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom/Mip{}", i)),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            self.mip_views.push(mip_tex.create_view(&Default::default()));
            self.mip_textures.push(mip_tex);

            let up_tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(&format!("Bloom/Upsample{}", i)),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            });
            self.upsample_views.push(up_tex.create_view(&Default::default()));
            self.upsample_textures.push(up_tex);
        }

        self.width = width;
        self.height = height;
    }
}

impl PostProcessingEffect for BloomEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        if self.initialized { return; }

        // --- Shaders ---
        let ds_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom/DownsampleShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom_downsample.wgsl").into()),
        });
        let us_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom/UpsampleShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom_upsample.wgsl").into()),
        });
        let comp_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Bloom/CompositeShader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../../shaders/bloom_composite.wgsl").into()),
        });

        // --- Downsample BGL: texture_2d(0) + storage_texture(1) + uniform(2) ---
        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom/DownsampleBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // --- Upsample BGL: texture_2d(0) + sampler(1) + texture_2d(2) + storage_texture(3) + uniform(4) ---
        let upsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom/UpsampleBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // --- Composite BGL: texture_2d(0) + texture_2d(1) + sampler(2) + storage_texture(3) + uniform(4) ---
        let composite_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bloom/CompositeBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2, multisampled: false,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    }, count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false, min_binding_size: None,
                    }, count: None,
                },
            ],
        });

        // --- Pipelines ---
        let ds_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&downsample_bgl], push_constant_ranges: &[],
        });
        self.downsample_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bloom/DownsamplePipeline"), layout: Some(&ds_layout), module: &ds_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        let us_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&upsample_bgl], push_constant_ranges: &[],
        });
        self.upsample_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bloom/UpsamplePipeline"), layout: Some(&us_layout), module: &us_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        let comp_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&composite_bgl], push_constant_ranges: &[],
        });
        self.composite_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Bloom/CompositePipeline"), layout: Some(&comp_layout), module: &comp_shader,
            entry_point: Some("main"), compilation_options: Default::default(), cache: None,
        }));

        // --- Sampler ---
        self.linear_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Bloom/LinearSampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        }));

        // --- Param buffers: 6 downsample + 5 upsample + 1 composite (each 32 bytes) ---
        let buf_usage = wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST;
        for i in 0..MIP_COUNT {
            self.downsample_params.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Bloom/DSParams{}", i)),
                size: 32, usage: buf_usage, mapped_at_creation: false,
            }));
        }
        for i in 0..(MIP_COUNT - 1) {
            self.upsample_params.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Bloom/USParams{}", i)),
                size: 32, usage: buf_usage, mapped_at_creation: false,
            }));
        }
        self.composite_params = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Bloom/CompositeParams"),
            size: 32, usage: buf_usage, mapped_at_creation: false,
        }));

        self.downsample_bgl = Some(downsample_bgl);
        self.upsample_bgl = Some(upsample_bgl);
        self.composite_bgl = Some(composite_bgl);
        self.initialized = true;
    }

    fn render(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::TextureView, _depth: &wgpu::TextureView, output: &wgpu::TextureView,
        _camera: &Camera, width: u32, height: u32,
    ) {
        if !self.initialized { return; }

        // Recreate mip textures if dimensions changed
        if width != self.width || height != self.height {
            self.create_mip_textures(device, width, height);
        }

        if self.mip_views.is_empty() { return; }

        let ds_pipeline = self.downsample_pipeline.as_ref().unwrap();
        let us_pipeline = self.upsample_pipeline.as_ref().unwrap();
        let comp_pipeline = self.composite_pipeline.as_ref().unwrap();
        let ds_bgl = self.downsample_bgl.as_ref().unwrap();
        let us_bgl = self.upsample_bgl.as_ref().unwrap();
        let comp_bgl = self.composite_bgl.as_ref().unwrap();
        let sampler = self.linear_sampler.as_ref().unwrap();

        let o = &self.options;

        // ---- Downsample (6 passes) ----
        {
            let mut src_w = width;
            let mut src_h = height;
            for level in 0..MIP_COUNT {
                let dst_w = (src_w / 2).max(1);
                let dst_h = (src_h / 2).max(1);

                let data: [f32; 8] = [
                    o.threshold, o.knee, o.intensity, o.radius,
                    src_w as f32, src_h as f32,
                    f32::from_bits(level as u32), // level as u32 bits
                    0.0, // _pad
                ];
                queue.write_buffer(&self.downsample_params[level], 0, bytemuck::cast_slice(&data));

                let src_view = if level == 0 { input } else { &self.mip_views[level - 1] };
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: ds_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.mip_views[level]) },
                        wgpu::BindGroupEntry { binding: 2, resource: self.downsample_params[level].as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Bloom/Downsample"), ..Default::default()
                });
                pass.set_pipeline(ds_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);

                src_w = dst_w;
                src_h = dst_h;
            }
        }

        // ---- Upsample (5 passes, from level 4 down to 0) ----
        {
            for (pass_idx, level) in (0..MIP_COUNT - 1).rev().enumerate() {
                let dst_w = self.mip_textures[level].size().width;
                let dst_h = self.mip_textures[level].size().height;

                let smaller_view = if level == MIP_COUNT - 2 {
                    &self.mip_views[MIP_COUNT - 1]
                } else {
                    &self.upsample_views[level + 1]
                };

                let smaller_w = if level == MIP_COUNT - 2 {
                    self.mip_textures[MIP_COUNT - 1].size().width
                } else {
                    self.upsample_textures[level + 1].size().width
                };
                let smaller_h = if level == MIP_COUNT - 2 {
                    self.mip_textures[MIP_COUNT - 1].size().height
                } else {
                    self.upsample_textures[level + 1].size().height
                };

                let data: [f32; 8] = [
                    o.threshold, o.knee, o.intensity, o.radius,
                    smaller_w as f32, smaller_h as f32,
                    f32::from_bits(level as u32),
                    0.0,
                ];
                queue.write_buffer(&self.upsample_params[pass_idx], 0, bytemuck::cast_slice(&data));

                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None, layout: us_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(smaller_view) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.mip_views[level]) },
                        wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&self.upsample_views[level]) },
                        wgpu::BindGroupEntry { binding: 4, resource: self.upsample_params[pass_idx].as_entire_binding() },
                    ],
                });

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Bloom/Upsample"), ..Default::default()
                });
                pass.set_pipeline(us_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups((dst_w + 7) / 8, (dst_h + 7) / 8, 1);
            }
        }

        // ---- Composite (1 pass) ----
        {
            let data: [f32; 8] = [
                o.threshold, o.knee, o.intensity, o.radius,
                width as f32, height as f32,
                0.0, 0.0,
            ];
            queue.write_buffer(self.composite_params.as_ref().unwrap(), 0, bytemuck::cast_slice(&data));

            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None, layout: comp_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&self.upsample_views[0]) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(output) },
                    wgpu::BindGroupEntry { binding: 4, resource: self.composite_params.as_ref().unwrap().as_entire_binding() },
                ],
            });

            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bloom/Composite"), ..Default::default()
            });
            pass.set_pipeline(comp_pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
        }
    }

    fn resize(&mut self, width: u32, height: u32, _gbuffer: &GBuffer) {
        // Mip textures will be recreated lazily in render() when dimensions change
        if width != self.width || height != self.height {
            self.mip_textures.clear();
            self.mip_views.clear();
            self.upsample_textures.clear();
            self.upsample_views.clear();
            self.width = 0;
            self.height = 0;
        }
    }

    fn destroy(&mut self) {
        self.mip_textures.clear();
        self.mip_views.clear();
        self.upsample_textures.clear();
        self.upsample_views.clear();
        self.downsample_params.clear();
        self.upsample_params.clear();
        self.initialized = false;
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}
