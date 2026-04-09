/// MRT GBuffer: color, emissive, normal, albedo + depth.
pub struct GBuffer {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub color_texture: wgpu::Texture,
    pub color_view: wgpu::TextureView,
    /// Snapshot of color after opaque rendering, before refractive/indirect objects.
    pub background_texture: wgpu::Texture,
    pub background_view: wgpu::TextureView,
    pub emissive_texture: wgpu::Texture,
    pub emissive_view: wgpu::TextureView,
    pub normal_texture: wgpu::Texture,
    pub normal_view: wgpu::TextureView,
    pub albedo_texture: wgpu::Texture,
    pub albedo_view: wgpu::TextureView,
    pub depth_texture: wgpu::Texture,
    pub depth_view: wgpu::TextureView,
    pub output_texture: wgpu::Texture,
    pub output_view: wgpu::TextureView,
    pub ping_pong_texture: wgpu::Texture,
    pub ping_pong_view: wgpu::TextureView,
    // MSAA intermediates
    pub color_msaa_view: Option<wgpu::TextureView>,
    pub emissive_msaa_view: Option<wgpu::TextureView>,
    pub normal_msaa_view: Option<wgpu::TextureView>,
    pub albedo_msaa_view: Option<wgpu::TextureView>,
    pub depth_msaa_view: Option<wgpu::TextureView>,
    // Keep textures alive
    _msaa_textures: Vec<wgpu::Texture>,
}

impl GBuffer {
    pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
    pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    pub const ALBEDO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
    pub const MRT_FORMATS: [wgpu::TextureFormat; 4] = [
        Self::COLOR_FORMAT, Self::COLOR_FORMAT, Self::COLOR_FORMAT, Self::ALBEDO_FORMAT,
    ];

    pub fn new(device: &wgpu::Device, width: u32, height: u32, sample_count: u32) -> Self {
        let mk = |label: &str, fmt: wgpu::TextureFormat, usage: wgpu::TextureUsages, sc: u32| {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: sc,
                dimension: wgpu::TextureDimension::D2, format: fmt, usage, view_formats: &[],
            });
            let view = tex.create_view(&Default::default());
            (tex, view)
        };

        let tex_usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let storage_usage = tex_usage | wgpu::TextureUsages::STORAGE_BINDING;

        let (color_texture, color_view) = mk("GBuffer/Color", Self::COLOR_FORMAT,
            storage_usage | wgpu::TextureUsages::COPY_SRC, 1);
        let (background_texture, background_view) = mk("GBuffer/Background", Self::COLOR_FORMAT,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST, 1);
        let (emissive_texture, emissive_view) = mk("GBuffer/Emissive", Self::COLOR_FORMAT, storage_usage, 1);
        let (normal_texture, normal_view) = mk("GBuffer/Normal", Self::COLOR_FORMAT, tex_usage, 1);
        let (albedo_texture, albedo_view) = mk("GBuffer/Albedo", Self::ALBEDO_FORMAT, tex_usage, 1);
        let (depth_texture, depth_view) = mk("GBuffer/Depth", Self::DEPTH_FORMAT, tex_usage, 1);
        let (output_texture, output_view) = mk("GBuffer/Output", Self::COLOR_FORMAT, storage_usage, 1);
        let (ping_pong_texture, ping_pong_view) = mk("GBuffer/PingPong", Self::COLOR_FORMAT, storage_usage, 1);

        let mut _msaa_textures = Vec::new();
        let (color_msaa_view, emissive_msaa_view, normal_msaa_view, albedo_msaa_view, depth_msaa_view) =
            if sample_count > 1 {
                let mu = wgpu::TextureUsages::RENDER_ATTACHMENT;
                let du = mu | wgpu::TextureUsages::TEXTURE_BINDING;
                let (ct, cv) = mk("GBuffer/ColorMSAA", Self::COLOR_FORMAT, mu, sample_count);
                let (et, ev) = mk("GBuffer/EmissiveMSAA", Self::COLOR_FORMAT, mu, sample_count);
                let (nt, nv) = mk("GBuffer/NormalMSAA", Self::COLOR_FORMAT, mu, sample_count);
                let (at, av) = mk("GBuffer/AlbedoMSAA", Self::ALBEDO_FORMAT, mu, sample_count);
                let (dt, dv) = mk("GBuffer/DepthMSAA", Self::DEPTH_FORMAT, du, sample_count);
                _msaa_textures.extend([ct, et, nt, at, dt]);
                (Some(cv), Some(ev), Some(nv), Some(av), Some(dv))
            } else {
                (None, None, None, None, None)
            };

        Self {
            width, height, sample_count,
            color_texture, color_view, background_texture, background_view,
            emissive_texture, emissive_view,
            normal_texture, normal_view, albedo_texture, albedo_view,
            depth_texture, depth_view, output_texture, output_view,
            ping_pong_texture, ping_pong_view,
            color_msaa_view, emissive_msaa_view, normal_msaa_view,
            albedo_msaa_view, depth_msaa_view, _msaa_textures,
        }
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        *self = Self::new(device, width, height, self.sample_count);
    }
}
