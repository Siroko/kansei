/// A GPU texture resource.
pub struct Texture {
    label: String,
    gpu_texture: Option<wgpu::Texture>,
    view: Option<wgpu::TextureView>,
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    usage: wgpu::TextureUsages,
    dimension: wgpu::TextureDimension,
    /// The bound view's dimension; `None` lets wgpu pick it from the texture.
    view_dimension: Option<wgpu::TextureViewDimension>,
    mip_levels: u32,
    /// Optional initial data, one entry per mip level from level 0 (tightly packed rows or block
    /// rows, layer after layer). Written to the GPU texture on first `initialize_with_data` call,
    /// then discarded.
    initial_data: Option<Vec<Vec<u8>>>,
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
            view_dimension: None,
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
            view_dimension: None,
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
            view_dimension: None,
            mip_levels: 1,
            initial_data: Some(vec![data.to_vec()]),
        }
    }

    /// A 2D texture array of `layers` equal layers, bound as `texture_2d_array` (see
    /// `Binding::texture_2d_array`), for example terrain materials packed into one binding.
    pub fn new_2d_array(label: &str, width: u32, height: u32, layers: u32, format: wgpu::TextureFormat, usage: wgpu::TextureUsages) -> Self {
        Self {
            label: label.to_string(),
            gpu_texture: None,
            view: None,
            format,
            size: wgpu::Extent3d { width, height, depth_or_array_layers: layers.max(1) },
            usage,
            dimension: wgpu::TextureDimension::D2,
            view_dimension: Some(wgpu::TextureViewDimension::D2Array),
            mip_levels: 1,
            initial_data: None,
        }
    }

    /// An RGBA8 2D texture array from one `width` x `height` image per layer, uploaded on first
    /// use like `from_rgba`.
    pub fn from_rgba_layers(label: &str, width: u32, height: u32, layers: &[&[u8]]) -> Self {
        let mut texture = Self::new_2d_array(
            label,
            width,
            height,
            layers.len() as u32,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        texture.initial_data = Some(vec![layers.concat()]);
        texture
    }

    /// A 2D texture with its whole mip chain given, level 0 first, in any format including the
    /// block-compressed ones (each level's blocks tightly packed, rounded up to whole blocks as
    /// `ktx2::transcode` produces them). Uploaded on first use like `from_rgba`.
    pub fn from_levels(label: &str, format: wgpu::TextureFormat, width: u32, height: u32, levels: Vec<Vec<u8>>) -> Self {
        let mut texture = Self::new_2d(label, width, height, format, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        texture.mip_levels = levels.len().max(1) as u32;
        texture.initial_data = Some(levels);
        texture
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
            view_dimension: None,
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
        self.view = Some(texture.create_view(&wgpu::TextureViewDescriptor { dimension: self.view_dimension, ..Default::default() }));
        self.gpu_texture = Some(texture);
    }

    /// Initialize the texture and upload any initial data. Called automatically
    /// by the renderer when this texture is used as a material bindable.
    pub fn initialize_with_data(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu_texture.is_some() { return; }
        self.initialize(device);
        if let Some(levels) = self.initial_data.take() {
            let texture = self.gpu_texture.as_ref().unwrap();
            let (bw, bh) = self.format.block_dimensions();
            let block_bytes = self.format.block_copy_size(None).unwrap_or(4);
            for (level, data) in levels.iter().enumerate() {
                let level = level as u32;
                // compressed copies cover whole blocks: the level's physical size
                let blocks_x = (self.size.width >> level).max(1).div_ceil(bw);
                let blocks_y = (self.size.height >> level).max(1).div_ceil(bh);
                let depth = if self.dimension == wgpu::TextureDimension::D3 {
                    (self.size.depth_or_array_layers >> level).max(1)
                } else {
                    self.size.depth_or_array_layers
                };
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: level,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(blocks_x * block_bytes),
                        rows_per_image: Some(blocks_y),
                    },
                    wgpu::Extent3d { width: blocks_x * bw, height: blocks_y * bh, depth_or_array_layers: depth },
                );
            }
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
