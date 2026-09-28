use crate::math::Vec4;
use crate::cameras::Camera;
use crate::geometries::Vertex;
use crate::lights::{Light, LightUniforms, LIGHT_UNIFORM_BYTES};
use crate::materials::{ComputePass, Material};
use crate::objects::Scene;
use super::compute_batch::ComputeBatch;
use super::gbuffer::GBuffer;
use super::shared_layouts::SharedLayouts;

/// Device limits the renderer requests from the adapter.
#[derive(Debug, Clone, Default)]
pub enum RequiredLimits {
    /// WebGPU's default limits, which every adapter supports (for example 16 sampled textures
    /// and 8 storage buffers per shader stage, 4 bind groups).
    #[default]
    Default,
    /// Everything the adapter supports: more sampled textures and storage buffers per shader
    /// stage, larger textures and buffers. Query what was granted with `Renderer::limits()`.
    Adapter,
    /// Exactly these limits; device creation fails if the adapter cannot meet them. Start from
    /// `wgpu::Limits::default()` and raise what you need.
    Custom(wgpu::Limits),
}

impl RequiredLimits {
    /// The limits to request from an adapter that supports `adapter`.
    pub fn resolve(&self, adapter: &wgpu::Limits) -> wgpu::Limits {
        match self {
            RequiredLimits::Default => wgpu::Limits::default(),
            RequiredLimits::Adapter => adapter.clone(),
            RequiredLimits::Custom(limits) => limits.clone(),
        }
    }
}

/// Core WebGPU renderer configuration.
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f32,
    pub sample_count: u32,
    pub clear_color: Vec4,
    pub present_mode: wgpu::PresentMode,
    /// Device limits to request (WebGPU's defaults unless raised).
    pub required_limits: RequiredLimits,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            device_pixel_ratio: 1.0,
            sample_count: 4,
            clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
            present_mode: wgpu::PresentMode::Fifo,
            required_limits: RequiredLimits::Default,
        }
    }
}

/// A pass's cached render bundles: the visible static renderables (`is_bundled`), opaque then
/// transparent, so the dynamic ones can be drawn live between them. `None` where there was
/// nothing to record.
struct SceneBundles {
    opaque: Option<wgpu::RenderBundle>,
    transparent: Option<wgpu::RenderBundle>,
    /// The renderables recorded (`bundle_key`); the bundles are re-recorded when it changes.
    key: Vec<usize>,
}

/// The main GPU renderer.
pub struct Renderer {
    pub config: RendererConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    presentation_format: wgpu::TextureFormat,
    // Depth texture for the canvas render path
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    // MSAA color texture (if sample_count > 1)
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    // Shared per-object matrix buffers (dynamic offset uniform)
    world_matrices_buf: Option<wgpu::Buffer>,
    normal_matrices_buf: Option<wgpu::Buffer>,
    world_matrices_staging: Vec<f32>,
    normal_matrices_staging: Vec<f32>,
    matrix_alignment: u32,
    last_object_count: usize,
    // Shared bind group layouts
    shared_layouts: Option<SharedLayouts>,
    // Mesh bind group (group 1) — dynamic offset into matrix buffers
    mesh_bind_group: Option<wgpu::BindGroup>,
    // Light uniform buffer (packed into camera bind group, binding 2)
    light_buf: Option<wgpu::Buffer>,
    light_uniforms: LightUniforms,
    // Shadow resources
    shadow_map: Option<crate::shadows::ShadowMap>,
    shadow_uniform_buf: Option<wgpu::Buffer>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_comparison_sampler: Option<wgpu::Sampler>,
    shadow_dummy_depth_tex: Option<wgpu::Texture>,
    shadow_dummy_depth_view: Option<wgpu::TextureView>,
    cube_dummy_tex: Option<wgpu::Texture>,
    cube_dummy_view: Option<wgpu::TextureView>,
    cube_shadow_sampler: Option<wgpu::Sampler>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_light_vp_bgl: Option<wgpu::BindGroupLayout>,
    shadow_light_vp_bg: Option<wgpu::BindGroup>,
    shadows_enabled: bool,
    // Cubemap shadow resources (point lights)
    cubemap_shadow_map: Option<crate::shadows::CubeMapShadowMap>,
    // Spot lights: storage buffer, shadow atlas and comparison sampler (group 3, bindings 5-7)
    spot_lights: crate::lights::spot_lights_gpu::SpotLightsGpu,
    spot_light_buf: Option<wgpu::Buffer>,
    spot_shadow_atlas: Option<crate::shadows::SpotShadowAtlas>,
    spot_dummy_atlas_view: Option<wgpu::TextureView>,
    spot_shadow_sampler: Option<wgpu::Sampler>,
    // Cascaded sun/moon shadows (group 3, bindings 10-12), and a count-0 stand-in without them
    cascaded_shadows: Option<crate::shadows::CascadedShadowMap>,
    cascade_dummy_buf: Option<wgpu::Buffer>,
    // Clustered light lists (group 3, bindings 8-9)
    light_clusters: Option<crate::lights::light_clusters::LightClusters>,
    clustered_lights: bool,
    // GPU instance culling (renderables with `instance_culling`)
    cull_pipeline: Option<crate::culling::CullPipeline>,
    // Planar reflections, drawn after the shadow maps and before the main pass
    planar_reflections: Vec<crate::reflections::PlanarReflection>,
    // Render bundle caching (the static renderables; dynamic ones are drawn live)
    render_bundle: Option<SceneBundles>,
    gbuffer_bundle: Option<SceneBundles>,
    gbuffer_last_sample_count: u32,
    // Depth-copy pass (resolve MSAA depth for compute shaders)
    depth_copy_pipeline: Option<wgpu::RenderPipeline>,
    depth_copy_bgl: Option<wgpu::BindGroupLayout>,
    // Fraction of the surface size the post-processing path renders the scene at
    render_scale: f32,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            device: None,
            queue: None,
            surface: None,
            surface_config: None,
            presentation_format: wgpu::TextureFormat::Bgra8Unorm,
            depth_texture: None,
            depth_view: None,
            msaa_texture: None,
            msaa_view: None,
            world_matrices_buf: None,
            normal_matrices_buf: None,
            world_matrices_staging: Vec::new(),
            normal_matrices_staging: Vec::new(),
            matrix_alignment: 256,
            last_object_count: 0,
            shared_layouts: None,
            mesh_bind_group: None,
            light_buf: None,
            light_uniforms: LightUniforms::new(),
            shadow_map: None,
            shadow_uniform_buf: None,
            shadow_bind_group: None,
            shadow_comparison_sampler: None,
            shadow_dummy_depth_tex: None,
            shadow_dummy_depth_view: None,
            cube_dummy_tex: None,
            cube_dummy_view: None,
            cube_shadow_sampler: None,
            shadow_pipeline: None,
            shadow_light_vp_bgl: None,
            shadow_light_vp_bg: None,
            shadows_enabled: false,
            cubemap_shadow_map: None,
            spot_lights: crate::lights::spot_lights_gpu::SpotLightsGpu::new(),
            spot_light_buf: None,
            spot_shadow_atlas: None,
            spot_dummy_atlas_view: None,
            spot_shadow_sampler: None,
            cascaded_shadows: None,
            cascade_dummy_buf: None,
            light_clusters: None,
            clustered_lights: true,
            cull_pipeline: None,
            planar_reflections: Vec::new(),
            render_bundle: None,
            gbuffer_bundle: None,
            gbuffer_last_sample_count: 0,
            depth_copy_pipeline: None,
            depth_copy_bgl: None,
            render_scale: 1.0,
        }
    }

    /// Create and initialize a Renderer from a wgpu `SurfaceTarget`.
    ///
    /// Handles `Instance`, `Surface`, and `Adapter` creation internally so that
    /// user code does not need to touch raw wgpu bootstrap.
    ///
    /// **Native (winit):**
    /// ```ignore
    /// let renderer = pollster::block_on(Renderer::create(config, window.clone()));
    /// ```
    ///
    /// **WASM (canvas):**
    /// ```ignore
    /// Initialize from an HTML canvas element (WASM only).
    /// ```ignore
    /// renderer.initialize_with_canvas(canvas).await;
    /// ```
    #[cfg(target_arch = "wasm32")]
    pub async fn initialize_with_canvas(&mut self, canvas: web_sys::HtmlCanvasElement) {
        self.initialize_with_target(wgpu::SurfaceTarget::Canvas(canvas)).await;
    }

    /// Initialize the Renderer from a platform surface target.
    /// Handles Instance, Surface, Adapter, Device creation internally.
    ///
    /// Native: `renderer.initialize_with_target(window.clone()).await`
    /// WASM: `renderer.initialize_with_canvas(canvas).await`
    pub async fn initialize_with_target(&mut self, target: impl Into<wgpu::SurfaceTarget<'static>>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let surface = instance
            .create_surface(target)
            .expect("Failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .expect("No suitable GPU adapter found");
        self.initialize(surface, &adapter).await;
    }

    /// Low-level initialization with a pre-created surface and adapter.
    ///
    /// Prefer [`Renderer::create`] which handles Instance/Surface/Adapter
    /// creation automatically.
    #[doc(hidden)]
    pub async fn initialize(&mut self, surface: wgpu::Surface<'static>, adapter: &wgpu::Adapter) {
        // Optional features: requested only where the adapter offers them, so devices without
        // them still initialize. TIMESTAMP_QUERY lets apps time GPU passes (perf HUDs).
        let optional_features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Kansei Device"),
                required_features: wgpu::Features::FLOAT32_FILTERABLE | optional_features,
                required_limits: self.config.required_limits.resolve(&adapter.limits()),
                memory_hints: wgpu::MemoryHints::default(),
            }, None)
            .await
            .expect("Failed to create device");
        let limits = device.limits();
        log::info!(
            "device limits: {} sampled textures, {} samplers, {} storage buffers per shader stage; textures up to {}",
            limits.max_sampled_textures_per_shader_stage,
            limits.max_samplers_per_shader_stage,
            limits.max_storage_buffers_per_shader_stage,
            limits.max_texture_dimension_2d
        );

        self.matrix_alignment = device.limits().min_uniform_buffer_offset_alignment;

        let surface_caps = surface.get_capabilities(adapter);
        let format = surface_caps.formats.iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Pick best available present mode: prefer requested, fallback to Mailbox, then Fifo
        let present_mode = if surface_caps.present_modes.contains(&self.config.present_mode) {
            self.config.present_mode
        } else if surface_caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
            log::info!("PresentMode::{:?} not supported, using Mailbox", self.config.present_mode);
            wgpu::PresentMode::Mailbox
        } else {
            log::info!("Using PresentMode::Fifo (vsync)");
            wgpu::PresentMode::Fifo
        };
        log::info!("Present mode: {:?} (available: {:?})", present_mode, surface_caps.present_modes);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.config.width,
            height: self.config.height,
            present_mode,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        self.presentation_format = format;
        self._create_depth_texture(&device);

        // Create shared bind group layouts
        let shared = SharedLayouts::new(&device);

        // Create light uniform buffer
        let light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/Lights"),
            size: LIGHT_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Shadow resources — dummy depth texture + comparison sampler + uniform buffer
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/DummyDepth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let dummy_depth_view = dummy_depth.create_view(&Default::default());

        let comparison_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/ShadowSampler"),
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        // Dummy cubemap distance texture (1x1x6 r32float)
        let cube_dummy_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/DummyCubeShadow"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 6 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cube_dummy_view = cube_dummy_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let cube_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/CubeShadowSampler"),
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let shadow_uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/ShadowUniforms"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Spot lights: a fixed-capacity storage buffer (so bind groups never go stale), a 1x1
        // dummy atlas until spot shadows are enabled, and the atlas' comparison sampler
        let spot_light_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/SpotLights"),
            size: crate::lights::spot_lights_gpu::SPOT_LIGHTS_BUFFER_BYTES as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let spot_dummy_atlas_view = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/DummySpotShadowAtlas"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: crate::shadows::SpotShadowAtlas::FORMAT,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let spot_shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/SpotShadowSampler"),
            compare: Some(wgpu::CompareFunction::LessEqual),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        self.shared_layouts = Some(shared);
        self.light_buf = Some(light_buf);
        self.shadow_dummy_depth_tex = Some(dummy_depth);
        self.shadow_dummy_depth_view = Some(dummy_depth_view);
        self.shadow_comparison_sampler = Some(comparison_sampler);
        self.shadow_uniform_buf = Some(shadow_uniform_buf);
        self.cube_dummy_tex = Some(cube_dummy_tex);
        self.cube_dummy_view = Some(cube_dummy_view);
        self.cube_shadow_sampler = Some(cube_shadow_sampler);
        self.light_clusters = Some(crate::lights::light_clusters::LightClusters::new(&device, &spot_light_buf));
        // zero-initialised: cascade count 0, everything lit
        self.cascade_dummy_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/NoCascades"),
            size: 384,
            usage: wgpu::BufferUsages::UNIFORM,
            mapped_at_creation: false,
        }));
        self.spot_light_buf = Some(spot_light_buf);
        self.spot_dummy_atlas_view = Some(spot_dummy_atlas_view);
        self.spot_shadow_sampler = Some(spot_shadow_sampler);

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
        self.rebuild_shadow_bind_group();
    }

    /// (Re)create the shared shadow bind group (group 3) from the enabled shadow resources, with
    /// 1x1 dummies standing in for the others.
    fn rebuild_shadow_bind_group(&mut self) {
        let device = self.device.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        // the cascaded map's widest cascade stands in for the single directional map
        let dir_view = self
            .cascaded_shadows
            .as_ref()
            .map(|c| &c.far_view)
            .or_else(|| self.shadow_map.as_ref().and_then(|sm| sm.depth_view.as_ref()))
            .unwrap_or_else(|| self.shadow_dummy_depth_view.as_ref().unwrap());
        let (cascade_view, cascade_buf) = match &self.cascaded_shadows {
            Some(c) => (&c.array_view, &c.uniform),
            None => (self.spot_dummy_atlas_view.as_ref().unwrap(), self.cascade_dummy_buf.as_ref().unwrap()),
        };
        let cube_view = self
            .cubemap_shadow_map
            .as_ref()
            .map(|c| &c.distance_view)
            .unwrap_or_else(|| self.cube_dummy_view.as_ref().unwrap());
        let spot_view = self
            .spot_shadow_atlas
            .as_ref()
            .map(|a| &a.array_view)
            .unwrap_or_else(|| self.spot_dummy_atlas_view.as_ref().unwrap());
        let view = wgpu::BindingResource::TextureView;
        let sampler = wgpu::BindingResource::Sampler;
        let clusters = self.light_clusters.as_ref().unwrap();
        self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/ShadowBG"),
            layout: &shared.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: view(dir_view) },
                wgpu::BindGroupEntry { binding: 1, resource: sampler(self.shadow_comparison_sampler.as_ref().unwrap()) },
                wgpu::BindGroupEntry { binding: 2, resource: self.shadow_uniform_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: view(cube_view) },
                wgpu::BindGroupEntry { binding: 4, resource: sampler(self.cube_shadow_sampler.as_ref().unwrap()) },
                wgpu::BindGroupEntry { binding: 5, resource: view(spot_view) },
                wgpu::BindGroupEntry { binding: 6, resource: self.spot_light_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: sampler(self.spot_shadow_sampler.as_ref().unwrap()) },
                wgpu::BindGroupEntry { binding: 8, resource: clusters.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 9, resource: clusters.lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 10, resource: view(cascade_view) },
                wgpu::BindGroupEntry { binding: 11, resource: cascade_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 12, resource: sampler(self.spot_shadow_sampler.as_ref().unwrap()) },
            ],
        }));
        self.invalidate_bundle();
    }

    /// Returns a reference to the underlying wgpu device.
    ///
    /// Prefer using higher-level APIs (e.g. `material.set_uniform_bindable()`)
    /// instead of accessing the device directly. This accessor will be removed
    /// in a future release once all subsystems manage their own GPU resources.
    #[doc(hidden)]
    pub fn device(&self) -> &wgpu::Device {
        self.device.as_ref().expect("Renderer not initialized")
    }

    /// Returns a reference to the underlying wgpu queue.
    ///
    /// Prefer using higher-level APIs instead of accessing the queue directly.
    /// This accessor will be removed in a future release.
    #[doc(hidden)]
    /// The limits the device was created with (see `RendererConfig::required_limits`).
    pub fn limits(&self) -> wgpu::Limits {
        self.device().limits()
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref().expect("Renderer not initialized")
    }

    pub fn shared_layouts(&self) -> &SharedLayouts {
        self.shared_layouts.as_ref().expect("Renderer not initialized")
    }

    pub(crate) fn raw_device(&self) -> &wgpu::Device {
        self.device()
    }

    pub(crate) fn raw_queue(&self) -> &wgpu::Queue {
        self.queue()
    }

    pub fn create_command_encoder(&self, desc: &wgpu::CommandEncoderDescriptor) -> wgpu::CommandEncoder {
        self.device().create_command_encoder(desc)
    }

    pub fn submit(&self, command_buffers: impl IntoIterator<Item = wgpu::CommandBuffer>) {
        self.queue().submit(command_buffers);
    }

    /// Create a 2D RGBA texture from raw bytes, ready for use as a material binding.
    /// The texture is initialized on the GPU and the data is uploaded immediately.
    pub fn create_texture_from_rgba(&self, label: &str, width: u32, height: u32, data: &[u8]) -> crate::buffers::Texture {
        let device = self.device();
        let queue = self.queue();
        let mut tex = crate::buffers::Texture::new_2d(
            label, width, height,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        tex.initialize(device);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: tex.gpu_texture().unwrap(),
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        );
        tex
    }

    /// Create a linear-filtering sampler for texture sampling.
    pub fn create_sampler_linear(&self) -> wgpu::Sampler {
        self.device().create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        })
    }

    /// Build a material's bind group using this renderer's shared layouts.
    /// Convenience wrapper so user code doesn't need to access `device()` or
    /// `shared_layouts()` directly.
    pub fn build_material_bind_group(
        &self,
        material: &mut Material,
        resources: &[(u32, crate::materials::BindingResource)],
    ) {
        material.create_bind_group(self.device(), self.shared_layouts(), resources);
    }

    pub fn compute(&self, pass: &ComputePass, workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) {
        ComputeBatch::submit(self.device(), self.queue(), &[(pass, workgroups_x, workgroups_y, workgroups_z)]);
    }

    pub fn compute_batch(&self, passes: &[(&ComputePass, u32, u32, u32)]) {
        ComputeBatch::submit(self.device(), self.queue(), passes);
    }

    pub fn presentation_format(&self) -> wgpu::TextureFormat {
        self.presentation_format
    }

    pub fn surface(&self) -> Option<&wgpu::Surface<'static>> {
        self.surface.as_ref()
    }

    pub fn width(&self) -> u32 { self.config.width }
    pub fn height(&self) -> u32 { self.config.height }

    /// Render the scene at `scale` times the surface size (clamped to 0.25..=1) in
    /// `render_with_postprocessing`. The GBuffer and every scene pass run at `render_size()`;
    /// the post-processing chain's upscaler (`TemporalAAEffect`) reconstructs the surface size
    /// from the jittered frames, and the effects after it run at the surface size. Without an
    /// upscaler in the chain the blit stretches the image to the surface.
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = if scale.is_finite() { scale.clamp(0.25, 1.0) } else { 1.0 };
    }

    pub fn render_scale(&self) -> f32 {
        self.render_scale
    }

    /// The size the post-processing path renders the scene at: the surface size times the
    /// render scale, rounded.
    pub fn render_size(&self) -> (u32, u32) {
        scaled_size(self.config.width, self.config.height, self.render_scale)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        if self.device.is_none() { return; }
        // Configure surface
        if let (Some(ref surface), Some(ref mut config)) = (&self.surface, &mut self.surface_config) {
            config.width = width;
            config.height = height;
            surface.configure(self.device.as_ref().unwrap(), config);
        }
        // Recreate depth/MSAA (separate borrow scope)
        self._recreate_size_dependent();
    }

    fn _recreate_size_dependent(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let sc = self.config.sample_count;
        let fmt = self.presentation_format;
        let device = self.device.as_ref().unwrap();

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if sc > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1, sample_count: sc,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT, view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    fn _create_depth_texture(&mut self, device: &wgpu::Device) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.config.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if self.config.sample_count > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.config.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.presentation_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    /// Ensure shared per-object matrix buffers are large enough.
    fn _ensure_matrix_buffers(&mut self, count: usize) {
        if count <= self.last_object_count && self.world_matrices_buf.is_some() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;
        let total_floats = count * floats_per_slot;
        let total_bytes = (total_floats * 4) as u64;

        self.world_matrices_staging.resize(total_floats, 0.0);
        self.normal_matrices_staging.resize(total_floats, 0.0);

        let world_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/WorldMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let normal_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/NormalMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Rebuild mesh bind group (group 1) to point at the new buffers
        let shared = self.shared_layouts.as_ref().unwrap();
        self.mesh_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/MeshBG"),
            layout: &shared.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &normal_buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
                // world matrix, then last frame's (for motion vectors): shaders may declare
                // either a mat4x4 or `KanseiMeshTransforms` (cameras::MOTION_VECTORS_WGSL)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &world_buf,
                        offset: 0,
                        size: std::num::NonZeroU64::new(128),
                    }),
                },
            ],
        }));

        self.world_matrices_buf = Some(world_buf);
        self.normal_matrices_buf = Some(normal_buf);
        self.last_object_count = count;

        // Bind group changed — cached bundles are stale
        self.render_bundle = None;
        self.gbuffer_bundle = None;
    }

    /// Invalidate all cached render bundles.
    ///
    /// Call this when scene objects are added/removed, materials change,
    /// or shadow resources are recreated.
    pub fn invalidate_bundle(&mut self) {
        self.render_bundle = None;
        self.gbuffer_bundle = None;
    }

    /// Record the scene's static renderables for a pass into its opaque and transparent bundles.
    fn build_scene_bundles(
        &self,
        scene: &Scene,
        camera: &Camera,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        key: Vec<usize>,
    ) -> SceneBundles {
        let bundle = |transparent| self.build_render_bundle(scene, camera, color_formats, depth_format, sample_count, transparent);
        SceneBundles { opaque: bundle(false), transparent: bundle(true), key }
    }

    /// Pre-record the draws of the visible static renderables (`is_bundled`), the opaque or the
    /// transparent ones, into a reusable `RenderBundle`; `None` when none of them is drawn.
    fn build_render_bundle(
        &self,
        scene: &Scene,
        camera: &Camera,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        transparent: bool,
    ) -> Option<wgpu::RenderBundle> {
        let device = self.device.as_ref().unwrap();
        let alignment = self.matrix_alignment;

        let formats: Vec<Option<wgpu::TextureFormat>> =
            color_formats.iter().map(|f| Some(*f)).collect();
        let mut encoder =
            device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
                label: Some("RenderBundle"),
                color_formats: &formats,
                depth_stencil: Some(wgpu::RenderBundleDepthStencil {
                    format: depth_format,
                    depth_read_only: false,
                    stencil_read_only: true,
                }),
                sample_count,
                multiview: None,
            });

        // Set camera bind group (group 1) — same for all objects
        encoder.set_bind_group(1, camera.bind_group().unwrap(), &[]);

        // Set shadow bind group (group 3) — same for all objects
        if let Some(ref bg) = self.shadow_bind_group {
            encoder.set_bind_group(3, bg, &[]);
        }

        // State tracking for dedup
        let mut current_pipeline_ptr: usize = 0;
        let mut current_material_bg_ptr: usize = 0;
        let mut draws = 0;

        for scene_idx in scene.ordered_indices() {
            let r = match scene.get_renderable(scene_idx) {
                Some(r) => r,
                None => continue,
            };
            if !r.visible || !r.geometry.initialized || !is_bundled(r) || r.is_transparent() != transparent {
                continue;
            }

            // Get pipeline
            let num_vb = 1 + r.geometry.instance_buffers.len();
            let key = crate::materials::PipelineKey {
                color_formats: color_formats.to_vec(),
                depth_format,
                sample_count,
                num_vertex_buffers: num_vb,
            };
            let pipeline = match r.material.pipeline_cache.get(&key) {
                Some(p) => p,
                None => continue,
            };

            // Set pipeline (skip if same)
            let pipeline_ptr = pipeline as *const _ as usize;
            if pipeline_ptr != current_pipeline_ptr {
                encoder.set_pipeline(pipeline);
                current_pipeline_ptr = pipeline_ptr;
                current_material_bg_ptr = 0; // reset material bg tracking
            }

            // Set material bind group (group 0, skip if same)
            if let Some(bg) = r.material.bind_group() {
                let bg_ptr = bg as *const _ as usize;
                if bg_ptr != current_material_bg_ptr {
                    encoder.set_bind_group(0, bg, &[]);
                    current_material_bg_ptr = bg_ptr;
                }
            }

            // Set mesh bind group (group 2) with dynamic offsets
            let offset = mesh_offset(scene_idx, alignment);
            encoder.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

            // Vertex/index buffers and the draw (the camera's culled instances, if culled)
            draw_geometry(&mut encoder, r, MAIN_VIEW);
            draws += 1;
        }

        (draws > 0).then(|| encoder.finish(&Default::default()))
    }

    /// Draw the visible dynamic renderables (`Renderable::dynamic`), the opaque or the
    /// transparent ones, directly in a live render pass, after the bundle of the same kind.
    /// They are never recorded into a bundle, so each frame's draw reads the matrices uploaded
    /// for that frame.
    #[allow(clippy::too_many_arguments)]
    fn draw_dynamic_renderables<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a Scene,
        camera: &'a Camera,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
        transparent: bool,
    ) {
        let mesh_bg = self.mesh_bind_group.as_ref().unwrap();
        // executing a bundle clears the pass's state: bind the shared groups again
        let mut shared_bound = false;
        for scene_idx in scene.ordered_indices() {
            let Some(r) = scene.get_renderable(scene_idx) else { continue };
            if !r.visible || !r.dynamic || !r.geometry.initialized || r.geometry.is_indirect() || r.is_transparent() != transparent {
                continue;
            }
            let key = crate::materials::PipelineKey {
                color_formats: color_formats.to_vec(),
                depth_format,
                sample_count,
                num_vertex_buffers: 1 + r.geometry.instance_buffers.len(),
            };
            let Some(pipeline) = r.material.pipeline_cache.get(&key) else { continue };
            if !shared_bound {
                pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
                if let Some(bg) = &self.shadow_bind_group {
                    pass.set_bind_group(3, bg, &[]);
                }
                shared_bound = true;
            }
            pass.set_pipeline(pipeline);
            if let Some(bg) = r.material.bind_group() {
                pass.set_bind_group(0, bg, &[]);
            }
            let offset = mesh_offset(scene_idx, self.matrix_alignment);
            pass.set_bind_group(2, mesh_bg, &[offset, offset]);
            draw_geometry(pass, r, MAIN_VIEW);
        }
    }

    /// Draw a pass's scene renderables (all but the indirect ones): its cached bundles with the
    /// dynamic renderables drawn live after each, opaque then transparent.
    #[allow(clippy::too_many_arguments)]
    fn draw_scene_renderables<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        bundles: &'a SceneBundles,
        scene: &'a Scene,
        camera: &'a Camera,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) {
        for (bundle, transparent) in [(&bundles.opaque, false), (&bundles.transparent, true)] {
            if let Some(bundle) = bundle {
                pass.execute_bundles(std::iter::once(bundle));
            }
            self.draw_dynamic_renderables(pass, scene, camera, color_formats, depth_format, sample_count, transparent);
        }
    }

    /// Draw indirect renderables (GPU-driven geometry like marching cubes)
    /// directly in a live render pass. Called after executing the render bundle.
    fn draw_indirect_renderables<'a>(
        &'a self,
        pass: &mut wgpu::RenderPass<'a>,
        scene: &'a Scene,
        camera: &'a Camera,
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) {
        let alignment = self.matrix_alignment;

        pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
        if let Some(ref bg) = self.shadow_bind_group {
            pass.set_bind_group(3, bg, &[]);
        }

        for scene_idx in scene.ordered_indices() {
            let r = match scene.get_renderable(scene_idx) {
                Some(r) => r,
                None => continue,
            };
            if !r.visible || !r.geometry.initialized || !r.geometry.is_indirect() {
                continue;
            }

            let num_vb = 1 + r.geometry.instance_buffers.len();
            let key = crate::materials::PipelineKey {
                color_formats: color_formats.to_vec(),
                depth_format,
                sample_count,
                num_vertex_buffers: num_vb,
            };
            let pipeline = match r.material.pipeline_cache.get(&key) {
                Some(p) => p,
                None => {
                    log::warn!("Indirect renderable '{}' missing pipeline for key {:?}",
                        r.geometry.label, key);
                    continue;
                }
            };

            pass.set_pipeline(pipeline);

            if let Some(bg) = r.material.bind_group() {
                pass.set_bind_group(0, bg, &[]);
            }

            let offset = mesh_offset(scene_idx, alignment);
            pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);
            pass.set_vertex_buffer(0, r.geometry.active_vertex_buffer().unwrap().slice(..));
            pass.set_index_buffer(
                r.geometry.active_index_buffer().unwrap().slice(..),
                wgpu::IndexFormat::Uint32,
            );
            pass.draw_indexed_indirect(
                r.geometry.active_indirect_buffer().unwrap(),
                0,
            );
        }
    }

    /// Upload camera + per-object matrices to GPU.
    fn upload_all(&mut self, scene: &Scene, camera: &Camera) {
        let queue = self.queue.as_ref().unwrap();

        // Upload camera matrices (camera owns its buffers)
        camera.upload(queue);

        // Upload lights
        let lights_vec: Vec<&Light> = scene.lights().collect();
        self.light_uniforms.pack_refs(&lights_vec);
        if let Some(ref buf) = self.light_buf {
            queue.write_buffer(buf, 0, self.light_uniforms.as_bytes());
        }
        let (spot_layers, spot_resolution) = self.spot_shadow_atlas.as_ref().map(|a| (a.layers, a.resolution)).unwrap_or((0, 0));
        self.spot_lights.pack(scene.lights(), spot_layers, spot_resolution);
        if let Some(ref buf) = self.spot_light_buf {
            queue.write_buffer(buf, 0, self.spot_lights.as_bytes());
        }

        // Upload per-object matrices: a slot per scene child, at its scene index (`mesh_offset`)
        let count = scene.children_len();
        self._ensure_matrix_buffers(count);

        let alignment = self.matrix_alignment as usize;
        debug_assert!(alignment >= 128, "mesh slots hold two matrices");
        write_mesh_slots(scene, &mut self.world_matrices_staging, &mut self.normal_matrices_staging, alignment / 4);

        let queue = self.queue.as_ref().unwrap();
        if !scene.is_empty() {
            if let Some(ref buf) = self.world_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.world_matrices_staging));
            }
            if let Some(ref buf) = self.normal_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.normal_matrices_staging));
            }
        }
    }

    /// Upload all scene object matrices to shared GPU buffers.
    /// Public wrapper around upload_all for backward compatibility.
    pub fn upload_matrices(&mut self, scene: &Scene, camera: &Camera) {
        self.upload_all(scene, camera);
    }

    /// Enable directional shadow mapping.
    pub fn enable_shadows(&mut self, resolution: u32) {
        let device = self.device.as_ref().unwrap();
        let mut sm = crate::shadows::ShadowMap::new(resolution);
        sm.initialize(device);

        let shared = self.shared_layouts.as_ref().unwrap();

        // Shadow pipeline (depth-only, no fragment)
        let shadow_light_vp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/LightVPBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow/PipelineLayout"),
            bind_group_layouts: &[&shadow_light_vp_bgl, &shared.camera_bgl, &shared.mesh_bgl],
            push_constant_ranges: &[],
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_vs.wgsl").into()),
        });

        self.shadow_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow/Pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("shadow_vs"),
                buffers: &[crate::geometries::Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        }));

        // Light VP bind group
        self.shadow_light_vp_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/LightVPBG"),
            layout: &shadow_light_vp_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sm.light_vp_buf.as_ref().unwrap().as_entire_binding(),
            }],
        }));

        self.shadow_light_vp_bgl = Some(shadow_light_vp_bgl);
        self.shadow_map = Some(sm);
        self.shadows_enabled = true;
        self.rebuild_shadow_bind_group();
    }

    /// The directional shadow map, once `enable_shadows` has been called.
    pub fn shadow_map(&self) -> Option<&crate::shadows::ShadowMap> {
        self.shadow_map.as_ref()
    }

    /// The point-light cube shadow atlas, once `enable_point_shadows` has been called.
    pub fn cubemap_shadow_map(&self) -> Option<&crate::shadows::CubeMapShadowMap> {
        self.cubemap_shadow_map.as_ref()
    }

    /// Enable cubemap shadow mapping for point lights.
    pub fn enable_point_shadows(&mut self, resolution: u32, max_lights: u32) {
        let csm = crate::shadows::CubeMapShadowMap::new(self, resolution, max_lights);

        self.cubemap_shadow_map = Some(csm);
        self.rebuild_shadow_bind_group();
    }

    /// Enable cascaded shadow maps for the scene's first directional light, when it has
    /// `cast_shadow` (the sun, or the moon at night): stable cascades, drawn through the casters' own vertex shaders and
    /// culled per cascade, sampled with contact-hardening PCSS by materials that include
    /// `shadows::CASCADED_SHADOWS_WGSL`. It replaces `enable_shadows`: the widest cascade also
    /// serves shaders that read the single directional map (group 3 binding 0).
    pub fn enable_cascaded_shadows(&mut self, options: crate::shadows::CascadedShadowOptions) {
        let device = self.device.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        self.cascaded_shadows = Some(crate::shadows::CascadedShadowMap::new(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap(), options));
        self.rebuild_shadow_bind_group();
    }

    /// The cascaded shadow map, once `enable_cascaded_shadows` has been called.
    pub fn cascaded_shadow_map(&self) -> Option<&crate::shadows::CascadedShadowMap> {
        self.cascaded_shadows.as_ref()
    }

    /// Fit the cascades to the camera for the first directional light (if it casts shadows), and point
    /// the single-map shadow uniforms at the widest cascade (before culling, which culls for the
    /// cascades too).
    fn update_cascaded_shadows(&mut self, scene: &Scene, camera: &Camera) {
        let Some(csm) = self.cascaded_shadows.as_mut() else { return };
        let queue = self.queue.as_ref().unwrap();
        // the scene's first directional light, when it casts shadows (as the fog assumes)
        let sun = scene
            .lights()
            .find_map(|l| match l {
                crate::lights::Light::Directional(d) => Some(d),
                _ => None,
            })
            .filter(|d| d.cast_shadow);
        let Some(sun) = sun else {
            csm.disable(queue);
            if let Some(buf) = &self.shadow_uniform_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&[0.0f32; 24]));
            }
            return;
        };
        let dir = glam::Vec3::new(sun.direction.x, sun.direction.y, sun.direction.z);
        let color = sun.effective_color();
        csm.fit(camera, dir);
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        csm.upload(queue, dir, glam::Vec3::new(color.x, color.y, color.z), eye);
        if let (Some(buf), Some(vp)) = (&self.shadow_uniform_buf, csm.far_view_projection()) {
            let mut data = [0.0f32; 24];
            data[..16].copy_from_slice(&vp.to_cols_array());
            data[16] = 0.0005; // bias (the single-map path of basic_lit.wgsl)
            data[17] = 2.0 * csm.options.max_distance / csm.options.resolution as f32; // normal bias
            data[18] = 1.0; // shadowEnabled
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&data));
        }
    }

    /// Cull view of cascade `index`, after the camera, the spot layers and the reflections.
    fn cascade_view(&self, index: usize) -> usize {
        self.reflection_view(self.planar_reflections.len()) + index
    }

    /// Render each cascade: every visible shadow caster, culled to the cascade.
    fn run_cascade_shadow_pass(&mut self, scene: &Scene) {
        let Some(csm) = self.cascaded_shadows.as_ref() else { return };
        if csm.slots.is_empty() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let mesh_bg = self.mesh_bind_group.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/CascadedShadows") });
        for cascade in 0..csm.slots.len() {
            let view = self.cascade_view(cascade);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/CascadePass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: csm.layer_view(cascade),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(1, csm.camera(cascade).bind_group().unwrap(), &[]);
            for scene_idx in scene.ordered_indices() {
                let Some(r) = scene.get_renderable(scene_idx) else { continue };
                if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                    continue;
                }
                let key = crate::materials::DepthPipelineKey::new(
                    crate::shadows::CascadedShadowMap::FORMAT,
                    1 + r.geometry.instance_buffers.len(),
                    crate::shadows::CascadedShadowMap::DEPTH_BIAS,
                );
                let Some(pipeline) = r.material.depth_pipeline_cache.get(&key) else { continue };
                pass.set_pipeline(pipeline);
                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }
                let offset = mesh_offset(scene_idx, self.matrix_alignment);
                pass.set_bind_group(2, mesh_bg, &[offset, offset]);
                draw_geometry(&mut pass, r, view);
            }
        }
        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
    }

    /// Shade spot lights through the clustered light lists (the default): each fragment visits
    /// only the lights whose range and cone reach its cluster. Off, every fragment visits every
    /// light (for comparisons and debugging).
    pub fn set_clustered_lights(&mut self, enabled: bool) {
        self.clustered_lights = enabled;
    }

    /// Enable perspective shadow maps for spot lights: each frame, the first `max_lights` spot
    /// lights with `cast_shadow` (in scene order) render a `resolution`² layer of the spot shadow
    /// atlas, drawing every caster through its material's own vertex shader.
    pub fn enable_spot_shadows(&mut self, resolution: u32, max_lights: u32) {
        let device = self.device.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        let atlas = crate::shadows::SpotShadowAtlas::new(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap(), resolution, max_lights);
        self.spot_shadow_atlas = Some(atlas);
        self.rebuild_shadow_bind_group();
    }

    /// The spot-light shadow atlas, once `enable_spot_shadows` has been called.
    pub fn spot_shadow_atlas(&self) -> Option<&crate::shadows::SpotShadowAtlas> {
        self.spot_shadow_atlas.as_ref()
    }

    /// The storage buffer holding the scene's spot lights (`KanseiSpotLights` in
    /// `lights::SPOT_LIGHT_TYPES_WGSL`), rewritten every frame. Materials see it at group 3
    /// binding 6; effects such as the volumetric fog bind it themselves.
    pub fn spot_lights_buffer(&self) -> &wgpu::Buffer {
        self.spot_light_buf.as_ref().expect("Renderer not initialized")
    }

    /// Create a camera's GPU resources (bind group with this renderer's lights).
    pub(crate) fn init_camera(&self, camera: &mut Camera) {
        let shared = self.shared_layouts.as_ref().expect("Renderer not initialized");
        camera.gpu_initialize(self.device(), &shared.camera_bgl, self.light_buf.as_ref().unwrap());
    }

    /// Register a planar reflection; the renderer draws it every frame, after the shadow maps
    /// and before the main pass. Returns its index for `planar_reflection(_mut)`.
    pub fn add_planar_reflection(&mut self, reflection: crate::reflections::PlanarReflection) -> usize {
        self.planar_reflections.push(reflection);
        self.planar_reflections.len() - 1
    }

    pub fn planar_reflection(&self, index: usize) -> Option<&crate::reflections::PlanarReflection> {
        self.planar_reflections.get(index)
    }

    pub fn planar_reflection_mut(&mut self, index: usize) -> Option<&mut crate::reflections::PlanarReflection> {
        self.planar_reflections.get_mut(index)
    }

    /// Point every planar reflection's mirrored camera for this frame (before culling, which
    /// culls for them too).
    fn update_planar_reflection_cameras(&mut self, camera: &Camera) {
        let queue = self.queue.as_ref().unwrap();
        for reflection in &mut self.planar_reflections {
            reflection.update_camera(queue, camera);
        }
    }

    /// Cull view of planar reflection `index`, after the camera and the spot shadow layers.
    fn reflection_view(&self, index: usize) -> usize {
        1 + self.spot_shadow_atlas.as_ref().map_or(0, |a| a.layers as usize) + index
    }

    /// Draw every planar reflection: the scene from the mirrored camera (only renderables on the
    /// reflection's layer mask) with the materials' GBuffer pipelines, then its resolve and mips.
    fn render_planar_reflections(&mut self, scene: &Scene) {
        if self.planar_reflections.is_empty() {
            return;
        }
        let queue = self.queue.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let mesh_bg = self.mesh_bind_group.as_ref().unwrap();
        let alignment = self.matrix_alignment;
        let cc = &self.config.clear_color;
        let clear = wgpu::Color { r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64 };
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/PlanarReflections") });
        for (index, reflection) in self.planar_reflections.iter().enumerate().filter(|(_, r)| r.is_active()) {
            let view = self.reflection_view(index);
            {
                let targets = reflection.color_attachments();
                let attachment = |view, load| Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                });
                let black = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Renderer/PlanarReflectionPass"),
                    color_attachments: &[
                        attachment(targets[0], wgpu::LoadOp::Clear(clear)),
                        attachment(targets[1], black),
                        attachment(targets[2], black),
                        attachment(targets[3], black),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: reflection.depth_attachment(),
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                pass.set_bind_group(1, reflection.camera().bind_group().unwrap(), &[]);
                if let Some(bg) = &self.shadow_bind_group {
                    pass.set_bind_group(3, bg, &[]);
                }
                for scene_idx in scene.ordered_indices() {
                    let Some(r) = scene.get_renderable(scene_idx) else { continue };
                    if !r.visible || !r.geometry.initialized || r.layers & reflection.layer_mask == 0 {
                        continue;
                    }
                    let key = crate::materials::PipelineKey {
                        color_formats: GBuffer::MRT_FORMATS.to_vec(),
                        depth_format: GBuffer::DEPTH_FORMAT,
                        sample_count: 1,
                        num_vertex_buffers: 1 + r.geometry.instance_buffers.len(),
                    };
                    let Some(pipeline) = r.material.pipeline_cache.get(&key) else { continue };
                    pass.set_pipeline(pipeline);
                    if let Some(bg) = r.material.bind_group() {
                        pass.set_bind_group(0, bg, &[]);
                    }
                    let offset = mesh_offset(scene_idx, alignment);
                    pass.set_bind_group(2, mesh_bg, &[offset, offset]);
                    // culled against the mirrored view, whose near plane is the water
                    draw_geometry(&mut pass, r, view);
                }
            }
            reflection.resolve(queue, &mut encoder);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The velocity pass: clear the GBuffer's velocity texture to `NO_VELOCITY`, then redraw the
    /// opaque renderables whose material has `outputs_velocity` with its velocity pipeline (the
    /// same shader, only @location(4) kept), depth-tested against the GBuffer.
    fn draw_velocity(&self, encoder: &mut wgpu::CommandEncoder, scene: &Scene, camera: &Camera, gbuffer: &GBuffer) {
        let no_velocity = GBuffer::NO_VELOCITY as f64;
        let mut attachments: [Option<wgpu::RenderPassColorAttachment>; GBuffer::VELOCITY_TARGET + 1] = Default::default();
        attachments[GBuffer::VELOCITY_TARGET] = Some(wgpu::RenderPassColorAttachment {
            view: &gbuffer.velocity_view,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color { r: no_velocity, g: no_velocity, b: 0.0, a: 0.0 }),
                store: wgpu::StoreOp::Store,
            },
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Renderer/VelocityPass"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &gbuffer.depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
        if let Some(bg) = &self.shadow_bind_group {
            pass.set_bind_group(3, bg, &[]);
        }
        let mesh_bg = self.mesh_bind_group.as_ref().unwrap();
        for scene_idx in scene.ordered_indices() {
            let Some(r) = scene.get_renderable(scene_idx) else { continue };
            if !r.visible || !r.geometry.initialized || !r.material.options.outputs_velocity || r.is_transparent() {
                continue;
            }
            let Some(pipeline) = r.material.velocity_pipeline_cache.get(&(1 + r.geometry.instance_buffers.len())) else { continue };
            pass.set_pipeline(pipeline);
            if let Some(bg) = r.material.bind_group() {
                pass.set_bind_group(0, bg, &[]);
            }
            let offset = mesh_offset(scene_idx, self.matrix_alignment);
            pass.set_bind_group(2, mesh_bg, &[offset, offset]);
            draw_geometry(&mut pass, r, MAIN_VIEW);
        }
    }

    /// Render each shadowed spot light's depth into its atlas layer: every visible shadow caster,
    /// with its material's depth pipeline and the light's camera in group 1.
    fn run_spot_shadow_pass(&mut self, scene: &Scene) {
        let Some(atlas) = self.spot_shadow_atlas.as_mut() else { return };
        if self.spot_lights.shadows.is_empty() {
            return;
        }
        let queue = self.queue.as_ref().unwrap();
        for slot in &self.spot_lights.shadows {
            atlas.update_camera(queue, slot.layer, slot.view, slot.projection);
        }
        let atlas = self.spot_shadow_atlas.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();
        let mesh_bg = self.mesh_bind_group.as_ref().unwrap();
        let alignment = self.matrix_alignment;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/SpotShadows") });
        for slot in &self.spot_lights.shadows {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/SpotShadowPass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: atlas.layer_view(slot.layer),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_bind_group(1, atlas.camera(slot.layer).bind_group().unwrap(), &[]);
            for scene_idx in scene.ordered_indices() {
                let Some(r) = scene.get_renderable(scene_idx) else { continue };
                if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                    continue;
                }
                let key = crate::materials::DepthPipelineKey::new(
                    crate::shadows::SpotShadowAtlas::FORMAT,
                    1 + r.geometry.instance_buffers.len(),
                    crate::shadows::SpotShadowAtlas::DEPTH_BIAS,
                );
                let Some(pipeline) = r.material.depth_pipeline_cache.get(&key) else { continue };
                pass.set_pipeline(pipeline);
                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }
                let offset = mesh_offset(scene_idx, alignment);
                pass.set_bind_group(2, mesh_bg, &[offset, offset]);
                // culled against this light's frustum, not the camera's
                draw_geometry(&mut pass, r, spot_view(slot.layer));
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// The views instance culling runs for, at fixed indices: `MAIN_VIEW`, then `spot_view(l)`
    /// for every layer of the spot shadow atlas (`None` when no light uses it this frame), then
    /// `reflection_view(r)` for every planar reflection, then `cascade_view(c)` for every cascade.
    fn cull_views(&self, camera: &Camera) -> Vec<Option<crate::culling::CullView>> {
        let mut views = vec![Some(crate::culling::CullView { view_proj: camera.projection_matrix.to_glam() * camera.view_matrix.to_glam(), casters_only: false })];
        if let Some(atlas) = &self.spot_shadow_atlas {
            views.resize(1 + atlas.layers as usize, None);
            for slot in &self.spot_lights.shadows {
                views[spot_view(slot.layer)] = Some(crate::culling::CullView { view_proj: slot.projection * slot.view, casters_only: true });
            }
        }
        // then planar reflections (`reflection_view`): the mirrored camera, near plane at the water
        views.extend(self.planar_reflections.iter().map(|r| {
            r.is_active().then(|| crate::culling::CullView {
                view_proj: r.camera().projection_matrix.to_glam() * r.camera().view_matrix.to_glam(),
                casters_only: false,
            })
        }));
        // then the cascades (`cascade_view`)
        if let Some(csm) = &self.cascaded_shadows {
            views.extend(csm.slots.iter().map(|s| Some(crate::culling::CullView { view_proj: s.projection * s.view, casters_only: true })));
        }
        views
    }

    /// Cull every renderable with `instance_culling` for every view (after the frame's uploads,
    /// before its shadow and main passes).
    fn run_instance_culling(&mut self, scene: &mut Scene, camera: &Camera) {
        let culled: Vec<usize> = scene
            .ordered_indices()
            .filter(|&i| scene.get_renderable(i).is_some_and(|r| r.instance_culling.is_some() && r.geometry.initialized))
            .collect();
        if culled.is_empty() {
            return;
        }
        let views = self.cull_views(camera);
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let pipeline = self.cull_pipeline.get_or_insert_with(|| crate::culling::CullPipeline::new(device));
        let lod_origin = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let mut stale_bundles = false;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Renderer/InstanceCulling") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Renderer/InstanceCulling"), ..Default::default() });
            pass.set_pipeline(&pipeline.pipeline);
            for idx in culled {
                let r = scene.get_renderable_mut(idx).unwrap();
                let (world, index_count, casts) = (r.world_matrix.to_glam(), r.geometry.index_count(), r.cast_shadow);
                let culling = r.instance_culling.as_mut().unwrap();
                stale_bundles |= culling.ensure_views(device, &pipeline.bgl, views.len());
                for (slot, view) in views.iter().enumerate() {
                    let Some(view) = view else { continue };
                    if view.casters_only && !casts {
                        continue;
                    }
                    let params = culling.params(view, world, lod_origin);
                    culling.dispatch(queue, &mut pass, slot, &params, index_count);
                }
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
        if stale_bundles {
            self.invalidate_bundle();
        }
    }

    /// Run the cubemap shadow pass for point lights.
    ///
    /// This is extracted into its own method to avoid borrow conflicts in render().
    fn run_cubemap_shadow_pass(&mut self, scene: &Scene) {
        // Collect shadow-casting point lights
        let shadow_point_lights: Vec<_> = scene
            .lights()
            .filter_map(|l| match l {
                crate::lights::Light::Point(pl) if pl.cast_shadow => {
                    Some((pl.position, pl.radius))
                }
                _ => None,
            })
            .collect();

        let csm = match self.cubemap_shadow_map.as_mut() {
            Some(csm) => csm,
            None => return,
        };

        let max = csm.max_lights as usize;
        let shadow_point_lights: Vec<_> = shadow_point_lights.into_iter().take(max).collect();

        if shadow_point_lights.is_empty() {
            return;
        }

        let (first_light_pos, first_light_radius) = shadow_point_lights[0];
        let light_pos = [first_light_pos.x, first_light_pos.y, first_light_pos.z];

        let queue = self.queue.as_ref().unwrap();
        let device = self.device.as_ref().unwrap();

        // Upload face uniforms for first shadow-casting point light
        csm.upload_face_uniforms(queue, 0, &light_pos, first_light_radius);

        // Ensure mesh buffers hold a slot per scene child (slots follow the scene index, as in
        // the renderer's own matrix buffers)
        csm.ensure_mesh_buffers(device, scene.children_len());

        // Upload mesh matrices to cubemap shadow's own buffers
        let csm_alignment = csm.matrix_alignment();
        let floats_per_slot = csm_alignment as usize / 4;

        for idx in scene.ordered_indices() {
            if let Some(r) = scene.get_renderable(idx) {
                let offset = idx * floats_per_slot;
                if offset + 16 <= csm.world_staging_len() {
                    csm.write_world_matrix(idx, r.world_matrix.as_slice());
                    csm.write_normal_matrix(idx, r.normal_matrix.as_slice());
                }
            }
        }
        csm.upload_mesh_matrices(queue);

        let shadow_far = csm.shadow_far;
        let csm_uniform_alignment = csm.uniform_alignment();

        // Render 6 faces
        for face in 0..6u32 {
            let face_slot = face as usize; // light 0, face N
            let color_view = csm.face_color_view(face_slot);

            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("CubemapShadow"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &color_view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: shadow_far as f64,
                                g: 0.0,
                                b: 0.0,
                                a: 0.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: csm.scratch_depth_view(),
                        depth_ops: Some(wgpu::Operations {
                            load: wgpu::LoadOp::Clear(1.0),
                            store: wgpu::StoreOp::Store,
                        }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });

                pass.set_pipeline(csm.pipeline());
                let light_offset = face * csm_uniform_alignment;
                pass.set_bind_group(0, csm.light_uniform_bg(), &[light_offset]);

                for scene_idx in scene.ordered_indices() {
                    if let Some(r) = scene.get_renderable(scene_idx) {
                        if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                            continue;
                        }

                        let offset = mesh_offset(scene_idx, csm_alignment);
                        pass.set_bind_group(1, csm.mesh_bg(), &[offset, offset]);
                        pass.set_vertex_buffer(
                            0,
                            r.geometry.active_vertex_buffer().unwrap().slice(..),
                        );
                        pass.set_index_buffer(
                            r.geometry.active_index_buffer().unwrap().slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                    }
                }
            }
            queue.submit(std::iter::once(encoder.finish()));
        }

        // Upload point shadow params to shadow uniform buffer
        let mut shadow_data = [0.0f32; 24];
        // Preserve existing directional shadow data if present
        if let Some(vp) = self.cascaded_shadows.as_ref().and_then(|c| c.far_view_projection()) {
            let csm = self.cascaded_shadows.as_ref().unwrap();
            shadow_data[..16].copy_from_slice(&vp.to_cols_array());
            shadow_data[16] = 0.0005;
            shadow_data[17] = 2.0 * csm.options.max_distance / csm.options.resolution as f32;
            shadow_data[18] = 1.0; // shadowEnabled
        } else if self.shadows_enabled {
            if let Some(ref sm) = self.shadow_map {
                shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                shadow_data[16] = sm.bias;
                shadow_data[17] = sm.normal_bias;
                shadow_data[18] = 1.0; // shadowEnabled
            }
        }
        shadow_data[19] = 1.0; // pointShadowEnabled
        shadow_data[20] = light_pos[0];
        shadow_data[21] = light_pos[1];
        shadow_data[22] = light_pos[2];
        shadow_data[23] = first_light_radius;

        if let Some(ref buf) = self.shadow_uniform_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
        }
    }

    /// Render the scene to the canvas surface (simple path).
    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera) {
        // Phase 0: Update transforms, prepare scene
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Phase 0.5: Initialize geometries, instance buffers, and pre-warm pipelines
        {
            let device = self.device.as_ref().unwrap();
            let queue = self.queue.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            let format = self.presentation_format;
            let sample_count = self.config.sample_count;
            let depth_format = wgpu::TextureFormat::Depth24Plus;
            let spot_shadows = self.spot_shadow_atlas.is_some();
            let cascades = self.cascaded_shadows.is_some();
            let reflections = !self.planar_reflections.is_empty();

            let ordered_indices: Vec<usize> = scene.ordered_indices().collect();
            for idx in ordered_indices {
                let r = scene.get_renderable_mut(idx).expect("ordered scene index should exist");
                if !r.geometry.initialized {
                    r.geometry.initialize(device);
                }
                for cb in &mut r.geometry.instance_buffers {
                    cb.ensure_ready(device, queue);
                }
                r.material.ensure_bindables_initialized(self);
                r.material.initialize(device, shared);

                let instance_layouts: Vec<_> = r.geometry.instance_buffers.iter()
                    .filter_map(|cb| cb.vertex_layout())
                    .collect();
                let mut layouts = vec![Vertex::LAYOUT];
                for il in &instance_layouts {
                    layouts.push(il.as_layout());
                }

                r.material.get_pipeline(
                    device, &layouts,
                    &[format], depth_format, sample_count,
                );
                if spot_shadows && r.cast_shadow {
                    r.material.get_depth_pipeline(device, &layouts, crate::shadows::SpotShadowAtlas::FORMAT, crate::shadows::SpotShadowAtlas::DEPTH_BIAS);
                }
                if cascades && r.cast_shadow {
                    r.material.get_depth_pipeline(device, &layouts, crate::shadows::CascadedShadowMap::FORMAT, crate::shadows::CascadedShadowMap::DEPTH_BIAS);
                }
                if reflections {
                    r.material.get_pipeline(device, &layouts, &GBuffer::MRT_FORMATS, GBuffer::DEPTH_FORMAT, 1);
                }
            }
        }

        // Initialize camera GPU resources if needed
        if !camera.initialized {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            camera.gpu_initialize(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap());
        }

        // Phase 1: Upload camera + per-object matrices
        self.upload_all(scene, camera);

        // GPU instance culling for every view (camera, spot shadows, reflections, cascades)
        self.update_planar_reflection_cameras(camera);
        self.update_cascaded_shadows(scene, camera);
        self.run_instance_culling(scene, camera);

        // Shadow pass
        if self.shadows_enabled && self.cascaded_shadows.is_none() {
            if let Some(ref mut sm) = self.shadow_map {
                // Find first directional light
                let dir_light_dir = scene.lights().find_map(|l| {
                    if let crate::lights::Light::Directional(dl) = l { Some(dl.direction) } else { None }
                });

                if let Some(light_dir) = dir_light_dir {
                    sm.compute_light_vp(camera, &light_dir);
                    sm.upload(self.queue.as_ref().unwrap());

                    // Upload shadow uniforms (96 bytes)
                    let mut shadow_data = [0.0f32; 24];
                    shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                    shadow_data[16] = sm.bias;
                    shadow_data[17] = sm.normal_bias;
                    shadow_data[18] = 1.0; // shadowEnabled
                    shadow_data[19] = 0.0; // pointShadowEnabled
                    shadow_data[20] = 0.0; // pointLightPos.x
                    shadow_data[21] = 0.0; // pointLightPos.y
                    shadow_data[22] = 0.0; // pointLightPos.z
                    shadow_data[23] = 0.0; // pointShadowFar
                    if let Some(ref buf) = self.shadow_uniform_buf {
                        self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
                    }

                    // Render shadow depth pass
                    let device = self.device.as_ref().unwrap();
                    let mut shadow_encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = shadow_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Renderer/ShadowPass"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: sm.depth_view.as_ref().unwrap(),
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });

                        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.shadow_light_vp_bg.as_ref().unwrap(), &[]);
                        pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);

                        let alignment = self.matrix_alignment;
                        for scene_idx in scene.ordered_indices() {
                            let r = scene.get_renderable(scene_idx).unwrap();
                            if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                                continue;
                            }

                            let offset = mesh_offset(scene_idx, alignment);
                            pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                            pass.set_vertex_buffer(0, r.geometry.active_vertex_buffer().unwrap().slice(..));
                            pass.set_index_buffer(r.geometry.active_index_buffer().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                        }
                    }
                    self.queue.as_ref().unwrap().submit(std::iter::once(shadow_encoder.finish()));
                }
            }
        }

        // Upload disabled shadow uniforms when shadows are off
        if !self.shadows_enabled && self.cubemap_shadow_map.is_none() && self.cascaded_shadows.is_none() {
            let shadow_data = [0.0f32; 24];
            if let Some(ref buf) = self.shadow_uniform_buf {
                self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
            }
        }

        // Cubemap shadow pass (point lights)
        if self.cubemap_shadow_map.is_some() {
            self.run_cubemap_shadow_pass(scene);
        }

        // Spot light shadow maps
        self.run_spot_shadow_pass(scene);
        // Cascaded sun/moon shadows
        self.run_cascade_shadow_pass(scene);

        // Planar reflections (they sample this frame's shadow maps), shaded with every light,
        // then the light clusters for the camera's passes
        if !self.planar_reflections.is_empty() {
            self.light_clusters.as_ref().unwrap().disable(self.queue.as_ref().unwrap());
        }
        self.render_planar_reflections(scene);
        let clusters = self.light_clusters.as_ref().unwrap();
        if self.clustered_lights {
            clusters.build(self.device.as_ref().unwrap(), self.queue.as_ref().unwrap(), camera, self.config.width, self.config.height);
        } else {
            clusters.disable(self.queue.as_ref().unwrap());
        }

        // Check material dirty flags → invalidate bundle
        for idx in scene.ordered_indices() {
            if let Some(r) = scene.get_renderable(idx) {
                if r.material_dirty {
                    self.render_bundle = None;
                    break;
                }
            }
        }

        // Build render bundle if needed
        let format = self.presentation_format;
        let sample_count = self.config.sample_count;
        let depth_format = wgpu::TextureFormat::Depth24Plus;
        let key = bundle_key(scene);
        if self.render_bundle.as_ref().is_none_or(|b| b.key != key) {
            self.render_bundle = Some(self.build_scene_bundles(
                scene,
                camera,
                &[format],
                depth_format,
                sample_count,
                key,
            ));
        }

        // Clear material_dirty flags
        let ordered: Vec<usize> = scene.ordered_indices().collect();
        for idx in ordered {
            if let Some(r) = scene.get_renderable_mut(idx) {
                r.material_dirty = false;
            }
        }

        // Phase 2+3: Create render pass and execute bundle
        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().expect("Failed to get surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/RenderEncoder"),
        });

        let cc = &self.config.clear_color;

        {
            let color_view = if sample_count > 1 {
                self.msaa_view.as_ref().unwrap()
            } else {
                &canvas_view
            };
            let resolve = if sample_count > 1 { Some(&canvas_view) } else { None };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/MainPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Static renderables from the bundles, dynamic ones drawn live
            self.draw_scene_renderables(
                &mut pass, self.render_bundle.as_ref().unwrap(), scene, camera,
                &[format], depth_format, sample_count,
            );

            // Draw GPU-driven indirect renderables in the same pass
            let fmt = self.presentation_format;
            self.draw_indirect_renderables(
                &mut pass, scene, camera,
                &[fmt], wgpu::TextureFormat::Depth24Plus,
                self.config.sample_count,
            );
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
        output.present();
        camera.end_frame();
    }

    /// Render scene with post-processing effects.
    pub fn render_with_postprocessing(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        volume: &mut crate::postprocessing::PostProcessingVolume,
    ) {
        let width = self.config.width;
        let height = self.config.height;
        let (render_width, render_height) = self.render_size();
        volume.ensure_gbuffer(render_width, render_height);

        // sub-pixel jitter for temporal anti-aliasing, in rendered pixels: a Halton (2, 3)
        // sequence of 8 phases per displayed pixel, so each one still sees 8 samples when
        // rendering below the display size
        camera.jitter = if volume.wants_jitter() {
            let i = camera.frame() % jitter_phases(self.render_scale) + 1;
            [(2.0 * halton(i, 2) - 1.0) / render_width as f32, (2.0 * halton(i, 3) - 1.0) / render_height as f32]
        } else {
            [0.0, 0.0]
        };

        // Render scene to GBuffer (actually draw into it)
        {
            let gbuffer = volume.gbuffer().unwrap();
            self.render_scene_to_gbuffer(scene, camera, gbuffer);
        }

        // Get surface texture for blit
        let surface = self.surface.as_ref().unwrap();
        let output = surface.get_current_texture().expect("Surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        // Run post-processing chain + blit
        volume.render(camera, &canvas_view, width, height);

        output.present();
        camera.end_frame();
    }

    /// Private: draw scene into GBuffer MRT (non-MSAA, sample_count=1).
    fn render_scene_to_gbuffer(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        gbuffer: &GBuffer,
    ) {
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Initialize camera GPU resources if needed
        if !camera.initialized {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            camera.gpu_initialize(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap());
        }

        // Initialize geometries + pre-warm pipelines for GBuffer formats
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        let depth_format = GBuffer::DEPTH_FORMAT;
        let sample_count = gbuffer.sample_count;
        let spot_shadows = self.spot_shadow_atlas.is_some();
            let cascades = self.cascaded_shadows.is_some();

        let ordered_indices: Vec<usize> = scene.ordered_indices().collect();
        for idx in ordered_indices {
            let r = scene.get_renderable_mut(idx).expect("ordered scene index should exist");
            if !r.geometry.initialized {
                r.geometry.initialize(device);
            }
            for cb in &mut r.geometry.instance_buffers {
                cb.ensure_ready(device, queue);
            }
            r.material.ensure_bindables_initialized(self);
            r.material.initialize(device, shared);

            let instance_layouts: Vec<_> = r.geometry.instance_buffers.iter()
                .filter_map(|cb| cb.vertex_layout())
                .collect();
            let mut layouts = vec![Vertex::LAYOUT];
            for il in &instance_layouts {
                layouts.push(il.as_layout());
            }

            r.material.get_pipeline(
                device, &layouts,
                &GBuffer::MRT_FORMATS, depth_format, sample_count,
            );
            if spot_shadows && r.cast_shadow {
                r.material.get_depth_pipeline(device, &layouts, crate::shadows::SpotShadowAtlas::FORMAT, crate::shadows::SpotShadowAtlas::DEPTH_BIAS);
            }
            if cascades && r.cast_shadow {
                r.material.get_depth_pipeline(device, &layouts, crate::shadows::CascadedShadowMap::FORMAT, crate::shadows::CascadedShadowMap::DEPTH_BIAS);
            }
            if r.material.options.outputs_velocity {
                r.material.get_velocity_pipeline(device, &layouts);
            }
        }

        // Upload camera + per-object matrices
        self.upload_all(scene, camera);

        // GPU instance culling for every view (camera, spot shadows, reflections, cascades)
        self.update_planar_reflection_cameras(camera);
        self.update_cascaded_shadows(scene, camera);
        self.run_instance_culling(scene, camera);

        // Shadow pass (if enabled)
        if self.shadows_enabled && self.cascaded_shadows.is_none() {
            if let Some(ref mut sm) = self.shadow_map {
                let dir_light_dir = scene.lights().find_map(|l| {
                    if let crate::lights::Light::Directional(dl) = l { Some(dl.direction) } else { None }
                });

                if let Some(light_dir) = dir_light_dir {
                    sm.compute_light_vp(camera, &light_dir);
                    sm.upload(self.queue.as_ref().unwrap());

                    let mut shadow_data = [0.0f32; 24];
                    shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                    shadow_data[16] = sm.bias;
                    shadow_data[17] = sm.normal_bias;
                    shadow_data[18] = 1.0; // shadowEnabled
                    if let Some(ref buf) = self.shadow_uniform_buf {
                        self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
                    }

                    let device = self.device.as_ref().unwrap();
                    let mut shadow_encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = shadow_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Shadow/GBufferPath"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: sm.depth_view.as_ref().unwrap(),
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });

                        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.shadow_light_vp_bg.as_ref().unwrap(), &[]);
                        pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);

                        let alignment = self.matrix_alignment;
                        for scene_idx in scene.ordered_indices() {
                            let r = scene.get_renderable(scene_idx).unwrap();
                            if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                                continue;
                            }

                            let offset = mesh_offset(scene_idx, alignment);
                            pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);
                            pass.set_vertex_buffer(0, r.geometry.active_vertex_buffer().unwrap().slice(..));
                            pass.set_index_buffer(r.geometry.active_index_buffer().unwrap().slice(..), wgpu::IndexFormat::Uint32);
                            pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                        }
                    }
                    self.queue.as_ref().unwrap().submit(std::iter::once(shadow_encoder.finish()));
                }
            }
        } else if self.cubemap_shadow_map.is_none() && self.cascaded_shadows.is_none() {
            let shadow_data = [0.0f32; 24];
            if let Some(ref buf) = self.shadow_uniform_buf {
                self.queue.as_ref().unwrap().write_buffer(buf, 0, bytemuck::cast_slice(&shadow_data));
            }
        }

        // Cubemap shadow pass (point lights)
        if self.cubemap_shadow_map.is_some() {
            self.run_cubemap_shadow_pass(scene);
        }

        // Spot light shadow maps
        self.run_spot_shadow_pass(scene);
        // Cascaded sun/moon shadows
        self.run_cascade_shadow_pass(scene);

        // Planar reflections (they sample this frame's shadow maps), shaded with every light,
        // then the light clusters for the camera's passes
        if !self.planar_reflections.is_empty() {
            self.light_clusters.as_ref().unwrap().disable(self.queue.as_ref().unwrap());
        }
        self.render_planar_reflections(scene);
        let clusters = self.light_clusters.as_ref().unwrap();
        if self.clustered_lights {
            // tiles of the GBuffer's pixels (the render size)
            clusters.build(self.device.as_ref().unwrap(), self.queue.as_ref().unwrap(), camera, gbuffer.width, gbuffer.height);
        } else {
            clusters.disable(self.queue.as_ref().unwrap());
        }

        // Check material dirty flags → invalidate gbuffer bundle
        for idx in scene.ordered_indices() {
            if let Some(r) = scene.get_renderable(idx) {
                if r.material_dirty {
                    self.gbuffer_bundle = None;
                    break;
                }
            }
        }

        // Build GBuffer render bundle if needed
        let key = bundle_key(scene);
        if self.gbuffer_bundle.as_ref().is_none_or(|b| b.key != key)
            || self.gbuffer_last_sample_count != gbuffer.sample_count
        {
            self.gbuffer_bundle = Some(self.build_scene_bundles(
                scene,
                camera,
                &GBuffer::MRT_FORMATS,
                GBuffer::DEPTH_FORMAT,
                gbuffer.sample_count,
                key,
            ));
            self.gbuffer_last_sample_count = gbuffer.sample_count;
        }

        // Clear material_dirty flags
        let ordered: Vec<usize> = scene.ordered_indices().collect();
        for idx in ordered {
            if let Some(r) = scene.get_renderable_mut(idx) {
                r.material_dirty = false;
            }
        }

        // GBuffer MRT render pass
        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/GBufferDraw"),
        });
        let cc = &self.config.clear_color;
        let clear = wgpu::Color { r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64 };
        let black = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

        // Pass 1: Opaque objects (render bundle)
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferOpaquePass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.color_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.emissive_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo_view, resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            // Static renderables from the bundles, dynamic ones drawn live
            self.draw_scene_renderables(
                &mut pass, self.gbuffer_bundle.as_ref().unwrap(), scene, camera,
                &GBuffer::MRT_FORMATS, GBuffer::DEPTH_FORMAT, gbuffer.sample_count,
            );
        }

        // Copy opaque color → background texture (for refractive objects to sample)
        encoder.copy_texture_to_texture(
            gbuffer.color_texture.as_image_copy(),
            gbuffer.background_texture.as_image_copy(),
            wgpu::Extent3d { width: gbuffer.width, height: gbuffer.height, depth_or_array_layers: 1 },
        );

        // Pass 2: Indirect/refractive objects (Load existing MRT + depth)
        {
            let load = wgpu::LoadOp::Load;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferIndirectPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.color_view, resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.emissive_view, resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal_view, resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo_view, resolve_target: None,
                        ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            self.draw_indirect_renderables(
                &mut pass, scene, camera,
                &GBuffer::MRT_FORMATS, GBuffer::DEPTH_FORMAT, sample_count,
            );
        }

        // Pass 3: motion vectors of the materials that write them, against the GBuffer depth;
        // the rest of the velocity texture keeps NO_VELOCITY
        self.draw_velocity(&mut encoder, scene, camera, gbuffer);

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
    }

    /// Render scene into a GBuffer for post-processing.
    pub fn render_to_gbuffer(&mut self, scene: &mut Scene, camera: &mut Camera, gbuffer: &GBuffer) {
        camera.update_view_matrix();
        scene.prepare(camera.position());
        self.upload_all(scene, camera);

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/GBufferEncoder"),
        });

        let cc = &self.config.clear_color;
        let clear = wgpu::Color { r: cc.x as f64, g: cc.y as f64, b: cc.z as f64, a: cc.w as f64 };
        let black = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

        if gbuffer.sample_count > 1 {
            // MSAA path — clear only for now (GBuffer draw loop comes in Plan 2)
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.color_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.color_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.emissive_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.emissive_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.normal_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.normal_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: gbuffer.albedo_msaa_view.as_ref().unwrap(),
                        resolve_target: Some(&gbuffer.albedo_view),
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: gbuffer.depth_msaa_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        } else {
            // Non-MSAA path — clear only for now (GBuffer draw loop comes in Plan 2)
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/GBufferPass"),
                color_attachments: &[
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.color_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.emissive_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.normal_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                    Some(wgpu::RenderPassColorAttachment {
                        view: &gbuffer.albedo_view,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &gbuffer.depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
        }

        self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
    }

    /// Read data back from a GPU buffer to CPU.
    /// Creates a staging buffer, copies, maps, and returns the data.
    pub fn read_back_buffer_sync<T: bytemuck::Pod + Clone>(&self, buffer: &wgpu::Buffer, size: u64) -> Vec<T> {
        let device = self.device.as_ref().unwrap();
        let queue = self.queue.as_ref().unwrap();

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Readback/Staging"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Readback/Encoder"),
        });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).ok();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let result: Vec<T> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging.unmap();
        result
    }

    fn ensure_depth_copy_pipeline(&mut self) {
        if self.depth_copy_pipeline.is_some() { return; }
        let device = self.device.as_ref().unwrap();
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DepthCopy"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/depth_copy.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("DepthCopy/BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: true,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None, bind_group_layouts: &[&bgl], push_constant_ranges: &[],
        });
        self.depth_copy_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("DepthCopy"),
            layout: Some(&layout),
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        }));
        self.depth_copy_bgl = Some(bgl);
    }

    pub fn copy_msaa_depth(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        msaa_depth_view: &wgpu::TextureView,
        resolved_depth_view: &wgpu::TextureView,
    ) {
        self.ensure_depth_copy_pipeline();
        let device = self.device.as_ref().unwrap();
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: self.depth_copy_bgl.as_ref().unwrap(),
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(msaa_depth_view),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DepthCopy"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: resolved_depth_view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_pipeline(self.depth_copy_pipeline.as_ref().unwrap());
        pass.set_bind_group(0, &bg, &[]);
        pass.draw(0..3, 0..1);
    }

    /// Initialize and update a compute system.
    pub fn run_system(&self, system: &mut dyn crate::systems::ComputeSystem, dt: f32) {
        if !system.is_initialized() {
            let device = self.device.as_ref().unwrap();
            let queue = self.queue.as_ref().unwrap();
            system.initialize(device, queue);
        }
        system.update(dt);
    }
}

/// Cull view of the camera; spot shadow layer `l` is `spot_view(l)`.
const MAIN_VIEW: usize = 0;

fn spot_view(layer: u32) -> usize {
    1 + layer as usize
}

/// Byte offset of scene child `scene_idx`'s slot in the per-object matrix buffers (group 2's
/// dynamic offsets). Slots follow the scene index, not the draw order, so a renderable keeps its
/// slot, and a recorded render bundle its data, whatever else is shown or hidden.
fn mesh_offset(scene_idx: usize, alignment: u32) -> u32 {
    scene_idx as u32 * alignment
}

/// Write every visible renderable's world matrix, last frame's world matrix (which it then
/// replaces) and normal matrix into its slot of the staging arrays, which hold `floats_per_slot`
/// floats per scene child; slot `i` is scene child `i` (`mesh_offset`).
fn write_mesh_slots(scene: &Scene, world_staging: &mut [f32], normal_staging: &mut [f32], floats_per_slot: usize) {
    for idx in scene.ordered_indices() {
        if let Some(renderable) = scene.get_renderable(idx) {
            let offset = idx * floats_per_slot;
            let world = renderable.world_matrix;
            let previous = renderable.previous_world_matrix.replace(Some(world)).unwrap_or(world);
            world_staging[offset..offset + 16].copy_from_slice(world.as_slice());
            world_staging[offset + 16..offset + 32].copy_from_slice(previous.as_slice());
            normal_staging[offset..offset + 16].copy_from_slice(renderable.normal_matrix.as_slice());
        }
    }
}

/// Whether the cached render bundles record `r` when it is visible. Dynamic renderables are
/// drawn live after them, indirect ones in `draw_indirect_renderables`.
fn is_bundled(r: &crate::objects::Renderable) -> bool {
    !r.dynamic && !r.geometry.is_indirect()
}

/// The renderables a pass's bundles hold: the scene indices of the visible bundled ones, sorted.
/// Showing, hiding, adding or un-marking one changes it, and the bundles are re-recorded.
fn bundle_key(scene: &Scene) -> Vec<usize> {
    let mut key: Vec<usize> = scene
        .ordered_indices()
        .filter(|&i| scene.get_renderable(i).is_some_and(|r| r.visible && is_bundled(r)))
        .collect();
    key.sort_unstable();
    key
}

/// Bind a renderable's vertex and index buffers and draw it for cull view `view`: its culled,
/// compacted instances (indirect) when it has `InstanceCulling`, otherwise its geometry as is.
fn draw_geometry<'a>(enc: &mut impl wgpu::util::RenderEncoder<'a>, r: &'a crate::objects::Renderable, view: usize) {
    let culled = r.instance_culling.as_ref().and_then(|c| c.view(view));
    enc.set_vertex_buffer(0, r.geometry.active_vertex_buffer().unwrap().slice(..));
    for (i, cb) in r.geometry.instance_buffers.iter().enumerate() {
        let buffer = match culled {
            Some((instances, _)) if i == 0 => Some(instances),
            _ => cb.gpu_buffer(),
        };
        if let Some(buffer) = buffer {
            enc.set_vertex_buffer(i as u32 + 1, buffer.slice(..));
        }
    }
    enc.set_index_buffer(r.geometry.active_index_buffer().unwrap().slice(..), wgpu::IndexFormat::Uint32);
    if let Some((_, args)) = culled {
        enc.draw_indexed_indirect(args, 0);
    } else if r.geometry.is_indirect() {
        enc.draw_indexed_indirect(r.geometry.active_indirect_buffer().unwrap(), 0);
    } else {
        enc.draw_indexed(0..r.geometry.index_count(), 0, 0..r.geometry.instance_count);
    }
}

/// `width` x `height` times `scale`, rounded, at least 1 x 1.
fn scaled_size(width: u32, height: u32, scale: f32) -> (u32, u32) {
    let scaled = |n: u32| ((n as f32 * scale).round() as u32).max(1);
    (scaled(width), scaled(height))
}

/// Jitter phases for a render scale: 8 per displayed pixel.
fn jitter_phases(scale: f32) -> u32 {
    (8.0 / (scale * scale)).round() as u32
}

/// The `index`-th element (from 1) of the Halton sequence in `base`, in [0, 1).
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0;
    let mut f = 1.0;
    while index > 0 {
        f /= base as f32;
        result += f * (index % base) as f32;
        index /= base;
    }
    result
}

#[cfg(test)]
mod tests {
    #[test]
    fn halton_sequence() {
        assert_eq!(super::halton(1, 2), 0.5);
        assert_eq!(super::halton(2, 2), 0.25);
        assert_eq!(super::halton(3, 2), 0.75);
        assert!((super::halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((super::halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
        assert!((super::halton(4, 3) - 4.0 / 9.0).abs() < 1e-6);
    }

    #[test]
    fn render_size_and_jitter_phases_follow_the_scale() {
        assert_eq!(super::scaled_size(1920, 1080, 1.0), (1920, 1080));
        assert_eq!(super::scaled_size(1920, 1080, 0.5), (960, 540));
        assert_eq!(super::scaled_size(1920, 1080, 0.67), (1286, 724));
        assert_eq!(super::scaled_size(1, 1, 0.25), (1, 1));
        assert_eq!(super::jitter_phases(1.0), 8);
        assert_eq!(super::jitter_phases(0.5), 32);
    }

    use crate::geometries::BoxGeometry;
    use crate::materials::{Material, MaterialOptions};
    use crate::math::Vec3;
    use crate::objects::{Object3D, Renderable, Scene, SceneNode};

    /// A scene of a transform node, then three boxes at x = 1, 2, 3 (the second one dynamic).
    fn three_boxes() -> Scene {
        let mut scene = Scene::new();
        scene.add(SceneNode::Transform(Object3D::new()));
        for x in 1..=3 {
            let material = Material::new("Box", "", vec![], MaterialOptions::default());
            let mut r = Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), material);
            r.set_position(x as f32, 0.0, 0.0);
            r.dynamic = x == 2;
            scene.add(SceneNode::Renderable(r));
        }
        scene.prepare(&Vec3::ZERO);
        scene
    }

    /// The x translation written into each of the scene's slots (0 where nothing was written).
    fn slot_translations(scene: &Scene) -> Vec<f32> {
        const FLOATS_PER_SLOT: usize = 64; // 256-byte slots
        let mut world = vec![0.0; scene.children_len() * FLOATS_PER_SLOT];
        let mut normal = world.clone();
        super::write_mesh_slots(scene, &mut world, &mut normal, FLOATS_PER_SLOT);
        world.chunks(FLOATS_PER_SLOT).map(|slot| slot[12]).collect()
    }

    #[test]
    fn mesh_slots_follow_the_scene_index_whatever_is_hidden() {
        let mut scene = three_boxes();
        assert_eq!(slot_translations(&scene), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(super::mesh_offset(3, 256), 768);

        // hiding the first box leaves every other box in its slot
        scene.get_renderable_mut(1).unwrap().visible = false;
        scene.prepare(&Vec3::ZERO);
        assert_eq!(slot_translations(&scene), [0.0, 0.0, 2.0, 3.0]);
    }

    #[test]
    fn bundles_hold_the_visible_static_renderables() {
        let mut scene = three_boxes();
        assert_eq!(super::bundle_key(&scene), [1, 3]);

        // showing or hiding a dynamic renderable keeps the bundles
        scene.get_renderable_mut(2).unwrap().visible = false;
        scene.prepare(&Vec3::ZERO);
        assert_eq!(super::bundle_key(&scene), [1, 3]);

        // hiding a static one, or making it dynamic, re-records them
        scene.get_renderable_mut(1).unwrap().visible = false;
        scene.prepare(&Vec3::ZERO);
        assert_eq!(super::bundle_key(&scene), [3]);
        scene.get_renderable_mut(3).unwrap().dynamic = true;
        assert!(super::bundle_key(&scene).is_empty());
    }
}
