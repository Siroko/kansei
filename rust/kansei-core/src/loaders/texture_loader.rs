/// Loads images from disk into GPU textures.
pub struct TextureLoader;

impl TextureLoader {
    /// Load a texture from a file path.
    pub fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        path: &str,
    ) -> Result<LoadedTexture, String> {
        let img = image::open(path)
            .map_err(|e| format!("Failed to load image '{}': {}", path, e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Self::from_rgba8(device, queue, &rgba, width, height, Some(path))
    }

    /// Load a texture from raw bytes (PNG, JPG, etc).
    pub fn load_bytes(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bytes: &[u8],
        label: Option<&str>,
    ) -> Result<LoadedTexture, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("Failed to decode image: {}", e))?;
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Self::from_rgba8(device, queue, &rgba, width, height, label)
    }

    /// Create a GPU texture from raw RGBA8 pixel data.
    pub fn from_rgba8(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        data: &[u8],
        width: u32,
        height: u32,
        label: Option<&str>,
    ) -> Result<LoadedTexture, String> {
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(4 * width),
                rows_per_image: Some(height),
            },
            size,
        );
        let view = texture.create_view(&Default::default());
        Ok(LoadedTexture { texture, view, width, height })
    }
}

/// A loaded GPU texture with its view and dimensions.
pub struct LoadedTexture {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}
