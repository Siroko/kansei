/// A GPU texture resource.
pub struct Texture {
    label: String,
    gpu_texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    dimension: wgpu::TextureDimension,
    mip_levels: u32,
}

impl Texture {
    pub fn new_2d(label: &str, width: u32, height: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        Self {
            label: label.to_string(),
            gpu_texture: None,
            view: None,
            format,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            usage,
            dimension: wgpu::TextureDimension::D2,
            mip_levels: 1,
        }
    }

    pub fn new_3d(label: &str, width: u32, height: u32, depth: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        Self {
            label: label.to_string(),
            gpu_texture: None,
            view: None,
            format,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: depth },
            usage,
            dimension: wgpu::TextureDimension::D3,
            mip_levels: 1,
        }
    }

    pub fn initialize(&mut self, device: &wgpu::Device) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&self.label),
            size: self.size,
            mip_level_count: self.mip_levels,
            sample_count: 1,
            dimension: self.dimension,
            format: self.format,
            usage: self.usage,
            view_formats: &[],
        });
        self.view = Some(texture.create_view(&wgpu::TextureViewDescriptor::default()));
        self.gpu_texture = Some(texture);
    }

    pub fn gpu_texture(&self) -> Option<&wgpu::Texture> {
        self.gpu_texture.as_ref()
    }

    pub fn view(&self) -> Option<&wgpu::TextureView> {
        self.view.as_ref()
    }

    pub fn format(&self) -> wgpu::TextureFormat {
        self.format
    }

    pub fn size(&self) -> wgpu::Extent3d {
        self.size
    }
}
