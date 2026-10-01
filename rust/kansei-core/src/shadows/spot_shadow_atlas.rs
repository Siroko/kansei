use crate::cameras::Camera;

/// Perspective shadow maps for spot lights: one layer of a depth-texture array per shadowed
/// light, rendered each frame by the renderer through the casters' own vertex shaders.
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
            // (COPY_SRC: read back by the tests)
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
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
        Self { resolution, layers, texture, array_view, layer_views, cameras }
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
