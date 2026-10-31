use crate::cameras::Camera;

/// Perspective shadow maps for spot lights: one layer of a depth-texture array per shadowed
/// light, rendered each frame by the renderer through the casters' own vertex shaders. The
/// casters that stand still are kept in a second array (`Renderer::set_spot_shadow_cache`),
/// drawn again only when they or the light change, and copied into the layer before the casters
/// that move are drawn over them.
pub struct SpotShadowAtlas {
    pub resolution: u32,
    pub layers: u32,
    pub texture: wgpu::Texture,
    /// All layers, for sampling (`texture_depth_2d_array`).
    pub array_view: wgpu::TextureView,
    /// One view per layer, for rendering.
    layer_views: Vec<wgpu::TextureView>,
    /// One camera per layer: its bind group carries the light's view and projection into the
    /// casters' vertex shaders.
    cameras: Vec<Camera>,
    /// The still casters' depth, a layer each (made on first use), and what each layer holds
    /// (the light's matrices and its casters, hashed; None: nothing yet).
    cache: Option<(wgpu::Texture, Vec<wgpu::TextureView>)>,
    pub(crate) cache_keys: Vec<Option<u64>>,
}

impl SpotShadowAtlas {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

    /// Depth bias of the shadow pipelines (the shaders add a normal offset on top).
    pub(crate) const DEPTH_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState { constant: 0, slope_scale: 1.5, clamp: 0.0 };

    pub(crate) fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout, light_buf: &wgpu::Buffer, resolution: u32, layers: u32) -> Self {
        let layers = layers.max(1);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SpotShadowAtlas"),
            size: wgpu::Extent3d { width: resolution, height: resolution, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            // (COPY_SRC: read back by the tests; COPY_DST: the still casters' cache copied in)
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("SpotShadowAtlas/Array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let layer_views = (0..layers)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("SpotShadowAtlas/Layer"),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    base_array_layer: layer,
                    array_layer_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();
        let cameras = (0..layers)
            .map(|_| {
                let mut camera = Camera::new(60.0, 0.05, 100.0, 1.0);
                camera.gpu_initialize(device, camera_bgl, light_buf);
                camera
            })
            .collect();
        Self { resolution, layers, texture, array_view, layer_views, cameras, cache: None, cache_keys: vec![None; layers as usize] }
    }

    /// Layer `layer` of the still casters' cache, made with the cache on first use.
    pub(crate) fn cache_view(&mut self, device: &wgpu::Device, layer: u32) -> &wgpu::TextureView {
        let (resolution, layers) = (self.resolution, self.layers);
        let (_, views) = self.cache.get_or_insert_with(|| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("SpotShadowAtlas/StillCasters"),
                size: wgpu::Extent3d { width: resolution, height: resolution, depth_or_array_layers: layers },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Self::FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let views = (0..layers)
                .map(|layer| {
                    texture.create_view(&wgpu::TextureViewDescriptor {
                        label: Some("SpotShadowAtlas/StillCasters"),
                        dimension: Some(wgpu::TextureViewDimension::D2),
                        base_array_layer: layer,
                        array_layer_count: Some(1),
                        ..Default::default()
                    })
                })
                .collect();
            (texture, views)
        });
        &views[layer as usize]
    }

    /// Copy layer `layer` of the still casters' cache into the atlas.
    pub(crate) fn copy_cache(&self, encoder: &mut wgpu::CommandEncoder, layer: u32) {
        let Some((cache, _)) = &self.cache else { return };
        let at = |texture| wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All };
        encoder.copy_texture_to_texture(at(cache), at(&self.texture), wgpu::Extent3d { width: self.resolution, height: self.resolution, depth_or_array_layers: 1 });
    }

    /// Drop the cache (`Renderer::set_spot_shadow_cache(false)`), and what it held.
    pub(crate) fn drop_cache(&mut self) {
        if let Some((texture, _)) = self.cache.take() {
            texture.destroy();
        }
        self.cache_keys.fill(None);
    }

    /// Bytes of the still casters' cache on the GPU.
    pub fn cache_bytes(&self) -> u64 {
        self.cache.as_ref().map_or(0, |_| self.resolution as u64 * self.resolution as u64 * self.layers as u64 * 4)
    }

    pub(crate) fn layer_view(&self, layer: u32) -> &wgpu::TextureView {
        &self.layer_views[layer as usize]
    }

    /// Set layer `layer`'s camera to the light's matrices and upload them.
    pub(crate) fn update_camera(&mut self, queue: &wgpu::Queue, layer: u32, view: glam::Mat4, projection: glam::Mat4) {
        let camera = &mut self.cameras[layer as usize];
        camera.view_matrix = view.into();
        camera.inverse_view_matrix = view.inverse().into();
        camera.projection_matrix = projection.into();
        camera.upload(queue);
    }

    pub(crate) fn camera(&self, layer: u32) -> &Camera {
        &self.cameras[layer as usize]
    }
}
