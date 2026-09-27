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
    /// Optional initial data (RGBA bytes). Written to the GPU texture on first
    /// `initialize_with_data` call, then discarded.
    initial_data: Option<Vec<u8>>,
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
            initial_data: None,
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
            initial_data: None,
        }
    }

    /// Create a 2D RGBA texture from raw byte data. The data is stored and
    /// uploaded to the GPU on the first `initialize_with_data()` call (which
    /// the renderer triggers automatically when this texture is used as a
    /// material bindable).
    pub fn from_rgba(label: &str, width: u32, height: u32, data: &[u8]) -> Self {
        Self {
            label: label.to_string(),
            gpu_texture: None,
            view: None,
            format: wgpu::TextureFormat::Rgba8Unorm,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            dimension: wgpu::TextureDimension::D2,
            mip_levels: 1,
            initial_data: Some(data.to_vec()),
        }
    }

    /// Wrap a texture created elsewhere (a render target, a LUT, a cubemap) with the view to bind,
    /// so it can be attached to a material like any other Texture.
    pub fn from_view(label: &str, texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        Self {
            label: label.to_string(),
            format: texture.format(),
            size: texture.size(),
            usage: texture.usage(),
            dimension: texture.dimension(),
            mip_levels: texture.mip_level_count(),
            gpu_texture: Some(texture),
            view: Some(view),
            initial_data: None,
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

    /// Initialize the texture and upload any initial data. Called automatically
    /// by the renderer when this texture is used as a material bindable.
    pub fn initialize_with_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu_texture.is_some() { return; }
        self.initialize(device);
        if let Some(data) = self.initial_data.take() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: self.gpu_texture.as_ref().unwrap(),
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size.width * 4),
                    rows_per_image: None,
                },
                self.size,
            );
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.gpu_texture.is_some()
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

impl super::Bindable for Texture {
    fn ensure_ready(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.initialize_with_data(device, queue);
    }
    fn binding_resource(&self) -> Option<crate::materials::BindingResource> {
        self.view().map(crate::materials::BindingResource::TextureView)
    }
}
