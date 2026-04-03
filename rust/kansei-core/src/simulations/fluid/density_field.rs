const CLEAR_WGSL: &str = include_str!("shaders/density-field-clear.wgsl");
const SPLAT_WGSL: &str = include_str!("shaders/density-field-splat.wgsl");
const COPY_WGSL: &str = include_str!("shaders/density-field-copy.wgsl");

/// 3D density field — splatting particle density into a volume texture.
pub struct FluidDensityField {
    pub density_texture: wgpu::Texture,
    pub density_view: wgpu::TextureView,
    accum_buffer: wgpu::Buffer,
    params_buffer: wgpu::Buffer,
    params_data: [f32; 12],
    tex_dims: [u32; 3],
    pub kernel_scale: f32,

    clear_pipeline: wgpu::ComputePipeline,
    clear_bg: wgpu::BindGroup,
    splat_pipeline: wgpu::ComputePipeline,
    splat_bg: wgpu::BindGroup,
    copy_pipeline: wgpu::ComputePipeline,
    copy_bg: wgpu::BindGroup,
}

pub struct DensityFieldOptions {
    pub resolution: u32,
    pub kernel_scale: f32,
}

impl Default for DensityFieldOptions {
    fn default() -> Self {
        Self { resolution: 64, kernel_scale: 1.0 }
    }
}

impl FluidDensityField {
    pub fn new(
        device: &wgpu::Device,
        positions_buffer: &wgpu::Buffer,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        options: DensityFieldOptions,
    ) -> Self {
        let res = options.resolution;
        // Proportional dims based on bounds aspect ratio
        let extents = [
            (bounds_max[0] - bounds_min[0]).abs().max(0.001),
            (bounds_max[1] - bounds_min[1]).abs().max(0.001),
            (bounds_max[2] - bounds_min[2]).abs().max(0.001),
        ];
        let max_ext = extents[0].max(extents[1]).max(extents[2]);
        let tex_dims = [
            ((extents[0] / max_ext) * res as f32).round().max(1.0) as u32,
            ((extents[1] / max_ext) * res as f32).round().max(1.0) as u32,
            ((extents[2] / max_ext) * res as f32).round().max(1.0) as u32,
        ];
        let total_voxels = (tex_dims[0] * tex_dims[1] * tex_dims[2]) as usize;

        // 3D texture
        let density_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("DensityField/Tex"),
            size: wgpu::Extent3d { width: tex_dims[0], height: tex_dims[1], depth_or_array_layers: tex_dims[2] },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let density_view = density_texture.create_view(&Default::default());

        // Atomic accumulation buffer
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DensityField/Accum"),
            size: (total_voxels * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        // Params uniform (48 bytes)
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("DensityField/Params"),
            size: 48,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // === Clear pipeline ===
        let clear_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DensityField/Clear"), source: wgpu::ShaderSource::Wgsl(CLEAR_WGSL.into()),
        });
        let clear_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DensityField/ClearPipeline"), layout: None,
            module: &clear_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let clear_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DensityField/ClearBG"),
            layout: &clear_pipeline.get_bind_group_layout(0),
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&density_view) }],
        });

        // === Splat pipeline ===
        let splat_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DensityField/Splat"), source: wgpu::ShaderSource::Wgsl(SPLAT_WGSL.into()),
        });
        let splat_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DensityField/SplatPipeline"), layout: None,
            module: &splat_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let splat_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DensityField/SplatBG"),
            layout: &splat_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        // === Copy pipeline ===
        let copy_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DensityField/Copy"), source: wgpu::ShaderSource::Wgsl(COPY_WGSL.into()),
        });
        let copy_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("DensityField/CopyPipeline"), layout: None,
            module: &copy_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DensityField/CopyBG"),
            layout: &copy_pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: params_buffer.as_entire_binding() },
            ],
        });

        Self {
            density_texture, density_view, accum_buffer, params_buffer,
            params_data: [0.0; 12], tex_dims, kernel_scale: options.kernel_scale,
            clear_pipeline, clear_bg, splat_pipeline, splat_bg, copy_pipeline, copy_bg,
        }
    }

    pub fn tex_dims(&self) -> [u32; 3] { self.tex_dims }

    pub fn update(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        particle_count: u32,
        smoothing_radius: f32,
    ) {
        // Upload params
        let p = &mut self.params_data;
        // u32 fields via transmute
        let td = self.tex_dims;
        p[0] = f32::from_ne_bytes(td[0].to_ne_bytes());
        p[1] = f32::from_ne_bytes(td[1].to_ne_bytes());
        p[2] = f32::from_ne_bytes(td[2].to_ne_bytes());
        p[3] = f32::from_ne_bytes(particle_count.to_ne_bytes());
        p[4] = bounds_min[0]; p[5] = bounds_min[1]; p[6] = bounds_min[2];
        p[7] = smoothing_radius;
        p[8] = bounds_max[0]; p[9] = bounds_max[1]; p[10] = bounds_max[2];
        p[11] = self.kernel_scale;
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&self.params_data));

        let [w, h, d] = self.tex_dims;

        // Clear
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Clear"), timestamp_writes: None });
            pass.set_pipeline(&self.clear_pipeline);
            pass.set_bind_group(0, &self.clear_bg, &[]);
            pass.dispatch_workgroups((w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
        // Splat
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Splat"), timestamp_writes: None });
            pass.set_pipeline(&self.splat_pipeline);
            pass.set_bind_group(0, &self.splat_bg, &[]);
            pass.dispatch_workgroups((particle_count + 63) / 64, 1, 1);
        }
        // Copy
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Copy"), timestamp_writes: None });
            pass.set_pipeline(&self.copy_pipeline);
            pass.set_bind_group(0, &self.copy_bg, &[]);
            pass.dispatch_workgroups((w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
    }
}
