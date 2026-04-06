const CLEAR_WGSL: &str = include_str!("shaders/density-field-clear.wgsl");
const SPLAT_WGSL: &str = include_str!("shaders/density-field-splat.wgsl");
const COPY_WGSL: &str = include_str!("shaders/density-field-copy.wgsl");
use crate::buffers::{BufferType, ComputeBuffer};
use crate::buffers::Texture;
use crate::materials::{Binding, BindingResource, Compute};

/// 3D density field — splatting particle density into a volume texture.
pub struct FluidDensityField {
    pub density_texture: Texture,
    pub density_view: wgpu::TextureView,
    accum_buffer: ComputeBuffer,
    params_buffer: ComputeBuffer,
    params_data: [f32; 12],
    tex_dims: [u32; 3],
    pub kernel_scale: f32,

    // Stored GPU handles (cheap Arc clones)
    device: wgpu::Device,
    queue: wgpu::Queue,

    clear: Compute,
    splat: Compute,
    copy: Compute,
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
    /// Create from a Renderer reference (preferred for user-facing code).
    pub fn from_renderer(
        renderer: &crate::renderers::Renderer,
        positions_buffer: &wgpu::Buffer,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        options: DensityFieldOptions,
    ) -> Self {
        Self::new(renderer.device(), renderer.queue(), positions_buffer, bounds_min, bounds_max, options)
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
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
        let mut density_texture = Texture::new_3d(
            "DensityField/Tex",
            tex_dims[0],
            tex_dims[1],
            tex_dims[2],
            wgpu::TextureFormat::Rgba16Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
        );
        density_texture.initialize(device);
        let density_view = density_texture
            .view()
            .expect("DensityField/Tex view should exist after initialization")
            .clone();

        // Atomic accumulation buffer
        let mut accum_buffer = ComputeBuffer::new(
            "DensityField/Accum",
            BufferType::Storage,
            wgpu::BufferUsages::STORAGE,
            vec![0u8; total_voxels * 4],
        );
        accum_buffer.initialize(device);

        // Params uniform (48 bytes)
        let mut params_buffer = ComputeBuffer::new(
            "DensityField/Params",
            BufferType::Uniform,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            vec![0u8; 48],
        );
        params_buffer.initialize(device);

        let mut clear = Compute::new(
            "DensityField/Clear",
            CLEAR_WGSL,
            vec![Binding::storage_texture_3d(0, wgpu::ShaderStages::COMPUTE, wgpu::TextureFormat::Rgba16Float)],
        );
        clear.initialize(device);
        clear.set_bind_group(
            device,
            &[(0, BindingResource::StorageTexture(&density_view))],
        );

        let mut splat = Compute::new(
            "DensityField/Splat",
            SPLAT_WGSL,
            vec![
                Binding::storage(0, wgpu::ShaderStages::COMPUTE, false),
                Binding::storage(1, wgpu::ShaderStages::COMPUTE, false),
                Binding::uniform(2, wgpu::ShaderStages::COMPUTE),
            ],
        );
        splat.initialize(device);
        splat.set_bind_group(
            device,
            &[
                (0, BindingResource::Buffer { buffer: positions_buffer, offset: 0, size: None }),
                (1, BindingResource::Buffer { buffer: accum_buffer.gpu_buffer().unwrap(), offset: 0, size: None }),
                (2, BindingResource::Buffer { buffer: params_buffer.gpu_buffer().unwrap(), offset: 0, size: None }),
            ],
        );

        let mut copy = Compute::new(
            "DensityField/Copy",
            COPY_WGSL,
            vec![
                Binding::storage(0, wgpu::ShaderStages::COMPUTE, false),
                Binding::storage_texture_3d(1, wgpu::ShaderStages::COMPUTE, wgpu::TextureFormat::Rgba16Float),
                Binding::uniform(2, wgpu::ShaderStages::COMPUTE),
            ],
        );
        copy.initialize(device);
        copy.set_bind_group(
            device,
            &[
                (0, BindingResource::Buffer { buffer: accum_buffer.gpu_buffer().unwrap(), offset: 0, size: None }),
                (1, BindingResource::StorageTexture(&density_view)),
                (2, BindingResource::Buffer { buffer: params_buffer.gpu_buffer().unwrap(), offset: 0, size: None }),
            ],
        );

        Self {
            density_texture, density_view, accum_buffer, params_buffer,
            params_data: [0.0; 12], tex_dims, kernel_scale: options.kernel_scale,
            device: device.clone(), queue: queue.clone(),
            clear, splat, copy,
        }
    }

    pub fn tex_dims(&self) -> [u32; 3] { self.tex_dims }

    fn upload_params(&mut self, bounds_min: [f32; 3], bounds_max: [f32; 3], particle_count: u32, smoothing_radius: f32) {
        let p = &mut self.params_data;
        let td = self.tex_dims;
        p[0] = f32::from_ne_bytes(td[0].to_ne_bytes());
        p[1] = f32::from_ne_bytes(td[1].to_ne_bytes());
        p[2] = f32::from_ne_bytes(td[2].to_ne_bytes());
        p[3] = f32::from_ne_bytes(particle_count.to_ne_bytes());
        p[4] = bounds_min[0]; p[5] = bounds_min[1]; p[6] = bounds_min[2];
        p[7] = smoothing_radius;
        p[8] = bounds_max[0]; p[9] = bounds_max[1]; p[10] = bounds_max[2];
        p[11] = self.kernel_scale;
        self.params_buffer.write(&self.params_data);
        self.params_buffer.update(&self.queue);
    }

    pub fn update_with_encoder(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        particle_count: u32,
        smoothing_radius: f32,
    ) {
        self.upload_params(bounds_min, bounds_max, particle_count, smoothing_radius);

        let [w, h, d] = self.tex_dims;

        // Clear
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Clear"), timestamp_writes: None });
            self.clear.dispatch(&mut pass, (w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
        // Splat
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Splat"), timestamp_writes: None });
            self.splat.dispatch(&mut pass, (particle_count + 63) / 64, 1, 1);
        }
        // Copy
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Copy"), timestamp_writes: None });
            self.copy.dispatch(&mut pass, (w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
    }

    pub fn update(
        &mut self,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        particle_count: u32,
        smoothing_radius: f32,
    ) {
        self.upload_params(bounds_min, bounds_max, particle_count, smoothing_radius);

        let [w, h, d] = self.tex_dims;
        let device = &self.device;
        let queue = &self.queue;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("DensityField/Update") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Clear"), timestamp_writes: None });
            self.clear.dispatch(&mut pass, (w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Splat"), timestamp_writes: None });
            self.splat.dispatch(&mut pass, (particle_count + 63) / 64, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("DensityField/Copy"), timestamp_writes: None });
            self.copy.dispatch(&mut pass, (w + 3) / 4, (h + 3) / 4, (d + 3) / 4);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }
}
