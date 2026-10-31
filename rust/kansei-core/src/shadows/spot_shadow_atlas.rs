use crate::cameras::Camera;

/// Perspective shadow maps for spot lights: one layer of a depth-texture array per shadowed
/// light, rendered each frame by the renderer through the casters' own vertex shaders. The
/// casters that stand still are kept in a second array (`Renderer::set_spot_shadow_cache`),
/// drawn again only when they or the light change. The layer then keeps the cache's depth but
/// where the casters that move were drawn: each frame puts the cache back there (a scissored
/// pass writing its depth) and draws them again.
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
    cache: Option<Cache>,
    pub(crate) cache_keys: Vec<Option<u64>>,
    /// What each layer holds besides its cache: unknown (None: put it all back), or the texels
    /// the moving casters were drawn in (`[x0, y0, x1, y1)`, None: none, the layer is its cache).
    pub(crate) dirty: Vec<Option<Option<[u32; 4]>>>,
}

/// The still casters' depth a layer each, and the pass that writes it back into the atlas.
struct Cache {
    texture: wgpu::Texture,
    views: Vec<wgpu::TextureView>,
    restore: wgpu::RenderPipeline,
    groups: Vec<wgpu::BindGroup>,
}

/// Writes the cache layer's depth where the scissor lets it.
const RESTORE_WGSL: &str = "
@group(0) @binding(0) var cache: texture_depth_2d;
@vertex
fn vertex_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(uv * 2.0 - 1.0, 0.0, 1.0);
}
@fragment
fn fragment_main(@builtin(position) p: vec4f) -> @builtin(frag_depth) f32 {
    return textureLoad(cache, vec2i(p.xy), 0);
}
";

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
        let n = layers as usize;
        Self { resolution, layers, texture, array_view, layer_views, cameras, cache: None, cache_keys: vec![None; n], dirty: vec![None; n] }
    }

    /// Make the still casters' cache, on first use.
    pub(crate) fn ensure_cache(&mut self, device: &wgpu::Device) {
        if self.cache.is_some() {
            return;
        }
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("SpotShadowAtlas/StillCasters"),
            size: wgpu::Extent3d { width: self.resolution, height: self.resolution, depth_or_array_layers: self.layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let views: Vec<wgpu::TextureView> = (0..self.layers)
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
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SpotShadowAtlas/Restore"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false },
                count: None,
            }],
        });
        let groups = views
            .iter()
            .map(|v| device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("SpotShadowAtlas/Restore"), layout: &bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(v) }] }))
            .collect();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("SpotShadowAtlas/Restore"), source: wgpu::ShaderSource::Wgsl(RESTORE_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("SpotShadowAtlas/Restore"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let restore = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SpotShadowAtlas/Restore"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vertex_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fragment_main"), targets: &[], compilation_options: Default::default() }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: Self::FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        self.cache = Some(Cache { texture, views, restore, groups });
        self.dirty.fill(None);
    }

    /// Layer `layer` of the still casters' cache (`ensure_cache` first).
    pub(crate) fn cache_view(&self, layer: u32) -> &wgpu::TextureView {
        &self.cache.as_ref().expect("ensure_cache first").views[layer as usize]
    }

    /// In a pass on layer `layer`, write its cache's depth back within `rect` (`[x0, y0, x1, y1)`
    /// texels), leaving the scissor the whole layer.
    pub(crate) fn restore(&self, pass: &mut wgpu::RenderPass<'_>, layer: u32, rect: [u32; 4]) {
        let Some(cache) = &self.cache else { return };
        let [x0, y0, x1, y1] = rect;
        if x1 <= x0 || y1 <= y0 {
            return;
        }
        pass.set_scissor_rect(x0, y0, x1 - x0, y1 - y0);
        pass.set_pipeline(&cache.restore);
        pass.set_bind_group(0, &cache.groups[layer as usize], &[]);
        pass.draw(0..3, 0..1);
        pass.set_scissor_rect(0, 0, self.resolution, self.resolution);
    }

    /// Drop the cache (`Renderer::set_spot_shadow_cache(false)`), and what it held.
    pub(crate) fn drop_cache(&mut self) {
        if let Some(cache) = self.cache.take() {
            cache.texture.destroy();
        }
        self.cache_keys.fill(None);
        self.dirty.fill(None);
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
