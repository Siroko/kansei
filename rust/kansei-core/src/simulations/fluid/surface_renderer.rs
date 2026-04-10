const SURFACE_WGSL: &str = include_str!("shaders/fluid-surface.wgsl");
use crate::math::Mat4;

/// Screen-space ray-march renderer for the fluid density field.
pub struct FluidSurfaceRenderer {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    params_buffer: wgpu::Buffer,

    // Stored GPU handles (cheap Arc clones)
    device: wgpu::Device,
    queue: wgpu::Queue,

    pub fluid_color: [f32; 3],
    pub absorption: f32,
    pub density_scale: f32,
    pub density_threshold: f32,
    pub step_count: u32,
}

impl FluidSurfaceRenderer {
    pub fn new(renderer: &crate::renderers::Renderer) -> Self {
        let device = renderer.device();
        let queue = renderer.queue();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidSurface/Shader"),
            source: wgpu::ShaderSource::Wgsl(SURFACE_WGSL.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidSurface/BGL"),
            entries: &[
                // 0: inputTex (scene color)
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // 1: depthTex
                wgpu::BindGroupLayoutEntry { binding: 1, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false }, count: None },
                // 2: outputTex (storage write)
                wgpu::BindGroupLayoutEntry { binding: 2, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D2 }, count: None },
                // 3: densityTex (3D)
                wgpu::BindGroupLayoutEntry { binding: 3, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false }, count: None },
                // 4: sampler
                wgpu::BindGroupLayoutEntry { binding: 4, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering), count: None },
                // 5: params
                wgpu::BindGroupLayoutEntry { binding: 5, visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("FluidSurface/Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidSurface/Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("FluidSurface/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidSurface/Params"),
            size: 144, // SurfaceParams struct
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipeline, bgl, sampler, params_buffer,
            device: device.clone(), queue: queue.clone(),
            fluid_color: [0.77, 0.96, 1.0],
            absorption: 2.5,
            density_scale: 1.1,
            density_threshold: 1.09,
            step_count: 128,
        }
    }

    /// Build a bind group for the current set of textures.
    pub fn create_bind_group(
        &self,
        input_view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
        output_view: &wgpu::TextureView,
        density_view: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidSurface/BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(input_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(output_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(density_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 5, resource: self.params_buffer.as_entire_binding() },
            ],
        })
    }

    /// Dispatch the ray-march pass.
    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        inv_view_proj: &Mat4,
        camera_pos: [f32; 3],
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        width: u32,
        height: u32,
    ) {
        // Pack SurfaceParams (144 bytes = 36 floats)
        let mut data = [0.0f32; 36];
        let _u32_view: &mut [u32] = bytemuck::cast_slice_mut(&mut data);

        data[..16].copy_from_slice(inv_view_proj.as_slice());
        data[16] = camera_pos[0]; data[17] = camera_pos[1]; data[18] = camera_pos[2];
        data[19] = self.density_threshold;
        data[20] = bounds_min[0]; data[21] = bounds_min[1]; data[22] = bounds_min[2];
        data[23] = self.absorption;
        data[24] = bounds_max[0]; data[25] = bounds_max[1]; data[26] = bounds_max[2];
        data[27] = self.density_scale;
        data[28] = self.fluid_color[0]; data[29] = self.fluid_color[1]; data[30] = self.fluid_color[2];
        let u32_view: &mut [u32] = bytemuck::cast_slice_mut(&mut data[31..]);
        u32_view[0] = self.step_count;
        u32_view[1] = width;
        u32_view[2] = height;

        self.queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&data));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FluidSurface/March"), timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups((width + 7) / 8, (height + 7) / 8, 1);
    }
}
