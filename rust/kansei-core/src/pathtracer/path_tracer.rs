use crate::cameras::Camera;
use crate::lights::Light;
use crate::objects::Scene;
use crate::renderers::Renderer;

use super::blue_noise::generate_blue_noise;
use super::bvh_builder::GPUBVHData;
use super::material::PathTracerMaterial;

/// Size of the TraceParams uniform in bytes (52 floats * 4 bytes).
const TRACE_PARAMS_SIZE: u64 = 208;

// TraceParams flags (trace.wgsl's PT_*)
const PT_GUIDES: u32 = 1;
const PT_PREV_INSTANCES: u32 = 2;

/// The guide buffers a trace writes for a denoiser (`PathTracer::set_guides`), all
/// `rgba32uint`, laid out as trace.wgsl's bindings 20-22 describe: the first hit's diffuse and
/// specular light each over its albedo (`signal`), those albedos (`albedo`), the light no filter
/// should touch (sky, emitters seen directly, in both), and the surface: linear depth, normal,
/// motion and last frame's depth (`surface`).
pub struct PathTracerGuides<'a> {
    pub signal: &'a wgpu::TextureView,
    pub albedo: &'a wgpu::TextureView,
    pub surface: &'a wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

struct GuideTextures {
    signal: wgpu::TextureView,
    albedo: wgpu::TextureView,
    surface: wgpu::TextureView,
}

impl GuideTextures {
    fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let view = |label| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba32Uint,
                    usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&wgpu::TextureViewDescriptor::default())
        };
        Self { signal: view("PathTracer/GuideSignal"), albedo: view("PathTracer/GuideAlbedo"), surface: view("PathTracer/GuideSurface") }
    }
}

/// The `index`-th element (from 1) of the Halton sequence in `base`, in [0, 1).
fn halton(mut index: u32, base: u32) -> f32 {
    let mut f = 1.0;
    let mut r = 0.0;
    while index > 0 {
        f /= base as f32;
        r += f * (index % base) as f32;
        index /= base;
    }
    r
}

/// Owns the path trace compute pipeline and dispatches trace passes.
///
/// The trace shader is formed by concatenating `intersection.wgsl`,
/// `traversal.wgsl`, a binding preamble for the BVH storage buffers,
/// and `trace.wgsl`. The resulting pipeline uses a single bind group
/// (group 0) with 9 bindings:
///
///   0  TraceParams uniform
///   1  output_tex  storage texture (rgba16float, write)
///   2  materials   storage buffer  (read)
///   3  scene_lights storage buffer (read)
///   4  blue_noise  storage buffer  (read)
///   5  triangles   storage buffer  (read) — array<f32>
///   6  bvh4_nodes  storage buffer  (read) — array<vec4<f32>>
///   7  tlas_bvh4_nodes storage buffer (read) — array<vec4<f32>>
///   8  instances   storage buffer  (read) — array of Instance
///   9  prev_frame  texture_2d (read) — previous frame for temporal accumulation
///   20-22 guide buffers (rgba32uint, write) — `PathTracerGuides`, written with `set_guides`
///   23 prev_instances storage buffer (read) — last frame's instances, for the guides' motion
pub struct PathTracer {
    trace_pipeline: Option<wgpu::ComputePipeline>,
    trace_bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    prev_texture: Option<wgpu::Texture>,
    prev_view: Option<wgpu::TextureView>,
    materials_buf: Option<wgpu::Buffer>,
    lights_buf: Option<wgpu::Buffer>,
    blue_noise_buf: Option<wgpu::Buffer>,
    frame_index: u32,
    /// the sampler's frame: advances every trace, never reset (`frame_index` counts the
    /// accumulated frames)
    seed: u32,
    jitter: bool,
    /// the last trace's sub-pixel offset, pixels
    last_jitter: [f32; 2],
    /// full-size guide buffers while `set_guides(true)`, else 1x1 placeholders to bind
    guides: Option<GuideTextures>,
    guides_enabled: bool,
    placeholder_guides: GuideTextures,
    /// last frame's instances (the guides' object motion), and whether it holds them
    prev_instances: Option<wgpu::Buffer>,
    prev_instances_valid: bool,
    /// last frame's view-projection, for the guides' motion
    prev_view_proj: Option<[f32; 16]>,
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    spp: u32,
    max_bounces: u32,
    use_blue_noise: bool,
}

impl PathTracer {
    /// Create a new `PathTracer` from an initialized renderer.
    ///
    /// This immediately builds the compute pipeline and uploads the
    /// blue noise LUT. Call [`resize`] before the first [`trace`].
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        let shader_code = trace_wgsl();

        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PathTracer/trace"),
            source: wgpu::ShaderSource::Wgsl(shader_code.into()),
        });

        // ── Bind group layout ──────────────────────────────────────
        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PathTracer/BGL"),
            entries: &[
                // 0: TraceParams uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(TRACE_PARAMS_SIZE),
                    },
                    count: None,
                },
                // 1: output texture (rgba16float, write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba16Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // 2: materials
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 3: scene_lights
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 4: blue_noise
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 5: triangles (array<f32>)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 6: bvh4_nodes (array<vec4f>)
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 7: tlas_bvh4_nodes (array<vec4f>)
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 8: instances (array<Instance>)
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // 9: prev_frame (previous frame for temporal accumulation)
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // 20-22: guide buffers (rgba32uint, write)
                guide_entry(20),
                guide_entry(21),
                guide_entry(22),
                // 23: prev_instances (array<Instance>)
                wgpu::BindGroupLayoutEntry {
                    binding: 23,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // ── Pipeline ───────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PathTracer/PipelineLayout"),
            bind_group_layouts: &[Some(&trace_bgl)],
            immediate_size: 0,
        });

        let trace_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PathTracer/Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("trace_main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // ── TraceParams uniform buffer ─────────────────────────────
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PathTracer/TraceParams"),
            size: TRACE_PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ── Blue noise LUT ─────────────────────────────────────────
        let noise_data = generate_blue_noise();
        let blue_noise_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PathTracer/BlueNoise"),
            size: (noise_data.len() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&blue_noise_buf, 0, bytemuck::cast_slice(&noise_data));

        Self {
            trace_pipeline: Some(trace_pipeline),
            trace_bgl: Some(trace_bgl),
            params_buf: Some(params_buf),
            output_texture: None,
            output_view: None,
            prev_texture: None,
            prev_view: None,
            materials_buf: None,
            lights_buf: None,
            blue_noise_buf: Some(blue_noise_buf),
            frame_index: 0,
            seed: 0,
            jitter: true,
            last_jitter: [0.0; 2],
            guides: None,
            guides_enabled: false,
            placeholder_guides: GuideTextures::new(&device, 1, 1),
            prev_instances: None,
            prev_instances_valid: false,
            prev_view_proj: None,
            width: 0,
            height: 0,
            device,
            queue,
            spp: 1,
            max_bounces: 4,
            use_blue_noise: true,
        }
    }

    // ── Configuration ──────────────────────────────────────────────

    /// Set samples-per-pixel for the trace dispatch.
    pub fn set_spp(&mut self, spp: u32) {
        self.spp = spp.max(1);
    }

    /// Set maximum bounce depth.
    pub fn set_max_bounces(&mut self, bounces: u32) {
        self.max_bounces = bounces;
    }

    /// Enable/disable blue noise sampling (falls back to PCG RNG).
    pub fn set_use_blue_noise(&mut self, enabled: bool) {
        self.use_blue_noise = enabled;
    }

    /// Offset the primary rays within their pixels, a Halton (2, 3) point a frame (default on):
    /// accumulated or denoised frames anti-alias. Off, every ray goes through its pixel's centre.
    pub fn set_jitter(&mut self, enabled: bool) {
        self.jitter = enabled;
    }

    /// Write the guide buffers a denoiser reads (`guides`; `PathTracerDenoise` turns this on).
    pub fn set_guides(&mut self, enabled: bool) {
        if enabled != self.guides_enabled {
            self.guides_enabled = enabled;
            self.guides = (enabled && self.width > 0).then(|| GuideTextures::new(&self.device, self.width, self.height));
            self.prev_instances_valid = false;
            self.prev_view_proj = None;
        }
    }

    /// The guide buffers the last trace wrote, while `set_guides` is on (after the first `resize`).
    pub fn guides(&self) -> Option<PathTracerGuides<'_>> {
        self.guides.as_ref().map(|g| PathTracerGuides { signal: &g.signal, albedo: &g.albedo, surface: &g.surface, width: self.width, height: self.height })
    }

    /// The last trace's sub-pixel offset of its (first) samples from the pixel centres, in pixels
    /// (+x right, +y down).
    pub fn jitter(&self) -> [f32; 2] {
        self.last_jitter
    }

    /// The sampler's frame: it advances every trace and `reset_accumulation` leaves it, so each
    /// frame draws new samples.
    pub fn seed(&self) -> u32 {
        self.seed
    }

    // ── Resize / recreate output texture ───────────────────────────

    /// Recreate the output storage texture when the viewport changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;
        self.frame_index = 0;

        let tex_desc = wgpu::TextureDescriptor {
            label: Some("PathTracer/Output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        };
        let texture = self.device.create_texture(&tex_desc);
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.output_texture = Some(texture);
        self.output_view = Some(view);

        let prev = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PathTracer/PrevFrame"),
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ..tex_desc
        });
        let prev_view = prev.create_view(&wgpu::TextureViewDescriptor::default());
        self.prev_texture = Some(prev);
        self.prev_view = Some(prev_view);
        if self.guides_enabled {
            self.guides = Some(GuideTextures::new(&self.device, width, height));
            self.prev_view_proj = None;
        }
    }

    // ── Material upload ────────────────────────────────────────────

    /// Upload material data to the GPU. Creates or resizes the buffer as needed.
    pub fn set_materials(&mut self, materials: &[PathTracerMaterial]) {
        let byte_len = (materials.len() * std::mem::size_of::<PathTracerMaterial>()) as u64;
        let needs_recreate = match self.materials_buf {
            Some(ref buf) => buf.size() < byte_len,
            None => true,
        };
        if needs_recreate {
            self.materials_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PathTracer/Materials"),
                size: byte_len.max(PathTracerMaterial::GPU_STRIDE as u64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.queue.write_buffer(
            self.materials_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(materials),
        );
    }

    // ── Light upload ───────────────────────────────────────────────

    /// Upload `lights` as the tracer's lights; returns how many it took (the `light_count` for
    /// `trace`). Directional, point and area lights are traced; spot lights are not (skipped).
    pub fn set_lights(&mut self, lights: &[Light]) -> u32 {
        let data = pack_lights(lights.iter());
        self.set_lights_raw(&data);
        (data.len() / 16) as u32
    }

    /// `set_lights` with the scene's lights.
    pub fn set_lights_from_scene(&mut self, scene: &Scene) -> u32 {
        let data = pack_lights(scene.lights());
        self.set_lights_raw(&data);
        (data.len() / 16) as u32
    }

    /// Trace one frame of the scene `bvh_data` and `tlas` hold, from `camera`, in an encoder of
    /// its own (submitted), accumulating onto the frames before; `light_count` is what
    /// `set_lights` returned. Then `present` shows it.
    pub fn trace_frame(&mut self, renderer: &Renderer, bvh_data: &GPUBVHData, tlas: &super::TLASBuilder, camera: &Camera, light_count: u32) {
        let Some(tlas_buf) = tlas.tlas_nodes_buf.as_ref() else { return };
        let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("PathTracer/Frame") });
        self.trace(&mut encoder, bvh_data, tlas_buf, camera, light_count);
        renderer.submit(std::iter::once(encoder.finish()));
    }

    /// Show the traced image (after `trace`, its encoder submitted) on the surface through
    /// `volume`'s post-processing chain, e.g. one `ToneMapEffect` for exposure, a tone curve and
    /// the display's encoding. Nothing happens before the first `resize`.
    pub fn present(&self, renderer: &mut Renderer, camera: &Camera, volume: &mut crate::postprocessing::PostProcessingVolume) {
        if let Some(view) = self.output_view.as_ref() {
            renderer.present_through(view, camera, volume);
        }
    }

    /// Upload raw light data (array of f32) to the GPU.
    pub fn set_lights_raw(&mut self, data: &[f32]) {
        let byte_len = (data.len() * std::mem::size_of::<f32>()) as u64;
        let needs_recreate = match self.lights_buf {
            Some(ref buf) => buf.size() < byte_len,
            None => true,
        };
        if needs_recreate {
            self.lights_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PathTracer/Lights"),
                // at least one 64-byte LightData, the shader's minimum binding size
                size: byte_len.max(64),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        self.queue.write_buffer(
            self.lights_buf.as_ref().unwrap(),
            0,
            bytemuck::cast_slice(data),
        );
    }

    // ── Trace dispatch ─────────────────────────────────────────────

    /// Dispatch the path trace compute shader.
    ///
    /// Requires:
    ///   - `resize()` called with non-zero dimensions
    ///   - `set_materials()` called with at least one material
    ///   - `bvh_data` from `BVHBuilder::upload_to_gpu()`
    ///   - `tlas_buf` from `TLASBuilder` (the TLAS BVH4 node buffer)
    ///   - `camera` with up-to-date view/projection matrices
    ///   - `light_count` number of lights uploaded via `set_lights_raw`
    pub fn trace(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        bvh_data: &GPUBVHData,
        tlas_buf: &wgpu::Buffer,
        camera: &Camera,
        light_count: u32,
    ) {
        // Guard: all resources must be present
        let (Some(pipeline), Some(bgl), Some(params_buf), Some(output_view), Some(prev_view)) = (
            self.trace_pipeline.as_ref(),
            self.trace_bgl.as_ref(),
            self.params_buf.as_ref(),
            self.output_view.as_ref(),
            self.prev_view.as_ref(),
        ) else {
            return;
        };
        let Some(materials_buf) = self.materials_buf.as_ref() else {
            return;
        };
        let Some(blue_noise_buf) = self.blue_noise_buf.as_ref() else {
            return;
        };

        // Create a placeholder lights buffer if none exists
        if self.lights_buf.is_none() {
            self.lights_buf = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PathTracer/Lights/Empty"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let lights_buf = self.lights_buf.as_ref().unwrap();

        // last frame's instances, for the guides' object motion (a copy of the instances after
        // each trace; the same buffer, unread, until there is one)
        if self.guides_enabled {
            let size = bvh_data.instances_buf.size();
            if self.prev_instances.as_ref().is_none_or(|b| b.size() != size) {
                self.prev_instances = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("PathTracer/PrevInstances"),
                    size,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
                self.prev_instances_valid = false;
            }
        }
        let prev_instances = match (self.guides_enabled, self.prev_instances.as_ref()) {
            (true, Some(buf)) => buf,
            _ => &bvh_data.instances_buf,
        };
        let guides = self.guides.as_ref().unwrap_or(&self.placeholder_guides);

        // ── Upload TraceParams ─────────────────────────────────────
        let params = self.pack_params(camera, light_count);
        self.last_jitter = [params[32], params[33]];
        self.queue
            .write_buffer(params_buf, 0, bytemuck::cast_slice(&params));

        // ── Create bind group ──────────────────────────────────────
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PathTracer/BG"),
            layout: bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: materials_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: lights_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: blue_noise_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: bvh_data.triangles_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: bvh_data.bvh4_nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: tlas_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: bvh_data.instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(prev_view),
                },
                wgpu::BindGroupEntry { binding: 20, resource: wgpu::BindingResource::TextureView(&guides.signal) },
                wgpu::BindGroupEntry { binding: 21, resource: wgpu::BindingResource::TextureView(&guides.albedo) },
                wgpu::BindGroupEntry { binding: 22, resource: wgpu::BindingResource::TextureView(&guides.surface) },
                wgpu::BindGroupEntry { binding: 23, resource: prev_instances.as_entire_binding() },
            ],
        });

        // ── Dispatch ───────────────────────────────────────────────
        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PathTracer/Trace"),
            timestamp_writes: crate::profiling::gpu_pass("PathTracer/Trace").as_ref().map(crate::profiling::PassStamp::compute),
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
        drop(pass);

        // Copy output → prev_frame for next frame's accumulation
        if let (Some(output_tex), Some(prev_tex)) =
            (self.output_texture.as_ref(), self.prev_texture.as_ref())
        {
            let size = wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            };
            encoder.copy_texture_to_texture(
                output_tex.as_image_copy(),
                prev_tex.as_image_copy(),
                size,
            );
        }

        if self.guides_enabled {
            if let Some(prev) = self.prev_instances.as_ref() {
                encoder.copy_buffer_to_buffer(&bvh_data.instances_buf, 0, prev, 0, prev.size());
                self.prev_instances_valid = true;
            }
            self.prev_view_proj = Some(*camera.projection_matrix.mul(&camera.view_matrix).as_slice());
        }

        // Advance frame index (for temporal jitter / accumulation)
        self.frame_index += 1;
        self.seed = self.seed.wrapping_add(1);
    }

    // ── Output accessor ────────────────────────────────────────────

    /// Returns the trace output texture view for compositing / display.
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output_view.as_ref()
    }

    /// Current accumulated frame count.
    pub fn frame_index(&self) -> u32 {
        self.frame_index
    }

    /// Reset frame accumulation (e.g. after camera move).
    pub fn reset_accumulation(&mut self) {
        self.frame_index = 0;
    }

    // ── TraceParams packing ────────────────────────────────────────

    /// Pack camera + render parameters into a 208-byte (52 x f32) array
    /// matching the WGSL `TraceParams` struct layout.
    fn pack_params(&self, camera: &Camera, light_count: u32) -> [f32; 52] {
        // inv_view_proj = (P * V)^-1
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        let inv_vp = vp.inverse();

        let mut params = [0.0f32; 52];

        // bytes 0-63: mat4x4f inv_view_proj (16 floats, column-major)
        params[0..16].copy_from_slice(inv_vp.as_slice());

        // bytes 64-75: vec3f camera_pos
        let pos = camera.position();
        params[16] = pos.x;
        params[17] = pos.y;
        params[18] = pos.z;

        // bytes 76-79: u32 frame_index
        params[19] = f32::from_bits(self.frame_index);

        // bytes 80-83: u32 width
        params[20] = f32::from_bits(self.width);

        // bytes 84-87: u32 height
        params[21] = f32::from_bits(self.height);

        // bytes 88-91: u32 light_count
        params[22] = f32::from_bits(light_count);

        // bytes 92-95: u32 spp
        params[23] = f32::from_bits(self.spp);

        // bytes 96-99: u32 use_blue_noise
        params[24] = f32::from_bits(if self.use_blue_noise { 1u32 } else { 0u32 });

        // bytes 100-103: u32 fixed_seed (0 = normal temporal jitter)
        params[25] = f32::from_bits(0u32);

        // bytes 104-107: u32 max_bounces
        params[26] = f32::from_bits(self.max_bounces);

        // bytes 108-111: u32 seed
        params[27] = f32::from_bits(self.seed);

        // bytes 112-123: vec3f ambient_color (default dark gray sky)
        params[28] = 0.05;
        params[29] = 0.07;
        params[30] = 0.1;

        // bytes 124-127: u32 flags
        let mut flags = 0;
        if self.guides_enabled {
            flags |= PT_GUIDES;
            if self.prev_instances_valid {
                flags |= PT_PREV_INSTANCES;
            }
        }
        params[31] = f32::from_bits(flags);

        // bytes 128-135: vec2f jitter, in pixels within [-0.5, 0.5)
        if self.jitter {
            let i = self.seed % 65536 + 1;
            params[32] = halton(i, 2) - 0.5;
            params[33] = halton(i, 3) - 0.5;
        }

        // bytes 144-207: mat4x4f prev_view_proj (this frame's on the first)
        params[36..52].copy_from_slice(self.prev_view_proj.as_ref().unwrap_or(vp.as_slice()));

        params
    }
}

/// The trace shader's source.
///
/// The traversal shader references `triangles`, `bvh4_nodes`, `tlas_bvh4_nodes`, and
/// `instances` without declaring their bindings (they are marked "external"). A small preamble
/// between traversal.wgsl and trace.wgsl declares bindings 5-8 matching the TypeScript engine.
pub(crate) fn trace_wgsl() -> String {
    let intersection_src = include_str!("shaders/intersection.wgsl");
    let traversal_src = include_str!("shaders/traversal.wgsl");
    let trace_src = include_str!("shaders/trace.wgsl");

    let bvh_bindings = "\
// ── BVH storage bindings (injected by PathTracer) ───────────────────
@group(0) @binding(5) var<storage, read> triangles       : array<f32>;
@group(0) @binding(6) var<storage, read> bvh4_nodes      : array<vec4f>;
@group(0) @binding(7) var<storage, read> tlas_bvh4_nodes : array<vec4f>;
@group(0) @binding(8) var<storage, read> instances       : array<Instance>;
";

    format!("{}\n{}\n{}\n{}", intersection_src, traversal_src, bvh_bindings, trace_src)
}

fn guide_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba32Uint,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// The tracer's `LightData` records (16 floats each: position or direction and the type, colour
/// and intensity, the facing normal, extra sizes) for the lights it supports.
fn pack_lights<'a>(lights: impl Iterator<Item = &'a Light>) -> Vec<f32> {
    let mut data = Vec::new();
    for light in lights {
        let record: [f32; 16] = match light {
            Light::Directional(l) => {
                let d = l.direction.to_glam().normalize_or_zero();
                [d.x, d.y, d.z, f32::from_bits(1), l.color.x, l.color.y, l.color.z, l.intensity, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            }
            Light::Area(l) => {
                let n = (l.target - l.position).to_glam().normalize_or_zero();
                [l.position.x, l.position.y, l.position.z, f32::from_bits(2), l.color.x, l.color.y, l.color.z, l.intensity, n.x, n.y, n.z, 0.0, l.width, l.height, 0.0, 0.0]
            }
            Light::Point(l) => [l.position.x, l.position.y, l.position.z, f32::from_bits(3), l.color.x, l.color.y, l.color.z, l.intensity, 0.0, 0.0, 0.0, 0.0, l.radius, 0.0, 0.0, 0.0],
            Light::Spot(_) => continue,
        };
        data.extend_from_slice(&record);
    }
    data
}

#[cfg(test)]
mod shader_tests {
    use super::*;
    use crate::pathtracer::denoise::{AtrousParamsGpu, SvgfParamsGpu, SVGF_WGSL};

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        // (unpack2x16float counts as f16-in-f32 in naga 30, a capability every WebGPU device has)
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32)
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        module
    }

    fn struct_span(module: &naga::Module, name: &str) -> u64 {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as u64),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no struct {name}"))
    }

    /// The trace and the denoiser validate, and their uniforms are the size the CPU packs.
    #[test]
    fn shaders_validate_and_the_params_match() {
        let trace = validate("trace", &trace_wgsl());
        assert_eq!(struct_span(&trace, "TraceParams"), TRACE_PARAMS_SIZE);
        let svgf = validate("svgf", SVGF_WGSL);
        assert_eq!(struct_span(&svgf, "SvgfParams"), std::mem::size_of::<SvgfParamsGpu>() as u64);
        assert_eq!(struct_span(&svgf, "AtrousParams"), std::mem::size_of::<AtrousParamsGpu>() as u64);
        assert_eq!(struct_span(&svgf, "Texel"), 24);
    }

    /// Jitter offsets cover the pixel evenly: Halton (2, 3)'s first points.
    #[test]
    fn halton_points() {
        assert_eq!(halton(1, 2), 0.5);
        assert_eq!(halton(2, 2), 0.25);
        assert_eq!(halton(3, 2), 0.75);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod light_tests {
    use super::*;
    use crate::lights::{AreaLight, DirectionalLight, PointLight, SpotLight};
    use crate::math::Vec3;

    /// Each supported light becomes one 16-float LightData record with trace.wgsl's type tag
    /// (1 directional, 2 area, 3 point); spot lights are left out.
    #[test]
    fn scene_lights_pack_into_light_data_records() {
        let lights = [
            Light::Directional(DirectionalLight::new(Vec3::new(0.0, -2.0, 0.0), Vec3::new(1.0, 0.9, 0.8), 3.0)),
            Light::Spot(SpotLight::new(Vec3::new(0.0, 3.0, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 100.0, 10.0, 0.3, 0.5)),
            Light::Area(AreaLight::new(Vec3::new(0.0, 5.0, 0.0), Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 20.0, 2.0, 1.5)),
            Light::Point(PointLight::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.5, 0.5, 1.0), 4.0, 9.0)),
        ];
        let data = pack_lights(lights.iter());
        assert_eq!(data.len(), 3 * 16);
        let tag = |r: usize| data[r * 16 + 3].to_bits();
        assert_eq!((tag(0), tag(1), tag(2)), (1, 2, 3));
        assert_eq!(&data[0..3], &[0.0, -1.0, 0.0]);
        assert_eq!(&data[4..8], &[1.0, 0.9, 0.8, 3.0]);
        // the area light faces its target, sized width x height
        assert_eq!(&data[16 + 8..16 + 11], &[0.0, -1.0, 0.0]);
        assert_eq!(&data[16 + 12..16 + 14], &[2.0, 1.5]);
        // the point light's radius
        assert_eq!(data[32 + 12], 9.0);
    }
}
