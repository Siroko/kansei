use crate::cameras::Camera;
use crate::renderers::Renderer;

use super::blue_noise::generate_blue_noise;
use super::bvh_builder::GPUBVHData;
use super::material::PathTracerMaterial;

/// Size of the TraceParams uniform in bytes (48 floats * 4 bytes).
const TRACE_PARAMS_SIZE: u64 = 192;

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
pub struct PathTracer {
    trace_pipeline: Option<wgpu::ComputePipeline>,
    trace_bgl: Option<wgpu::BindGroupLayout>,
    params_buf: Option<wgpu::Buffer>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    materials_buf: Option<wgpu::Buffer>,
    lights_buf: Option<wgpu::Buffer>,
    blue_noise_buf: Option<wgpu::Buffer>,
    frame_index: u32,
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
    spp: u32,
    max_bounces: u32,
}

impl PathTracer {
    /// Create a new `PathTracer` from an initialized renderer.
    ///
    /// This immediately builds the compute pipeline and uploads the
    /// blue noise LUT. Call [`resize`] before the first [`trace`].
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        // ── Build concatenated shader source ────────────────────────
        //
        // The traversal shader references `triangles`, `bvh4_nodes`,
        // `tlas_bvh4_nodes`, and `instances` without declaring their
        // bindings (they are marked "external"). We inject a small
        // preamble between traversal.wgsl and trace.wgsl that declares
        // bindings 5-8 matching the TypeScript engine.
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

        let shader_code = format!(
            "{}\n{}\n{}\n{}",
            intersection_src, traversal_src, bvh_bindings, trace_src,
        );

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
            ],
        });

        // ── Pipeline ───────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PathTracer/PipelineLayout"),
            bind_group_layouts: &[&trace_bgl],
            push_constant_ranges: &[],
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
            materials_buf: None,
            lights_buf: None,
            blue_noise_buf: Some(blue_noise_buf),
            frame_index: 0,
            width: 0,
            height: 0,
            device,
            queue,
            spp: 1,
            max_bounces: 4,
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

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
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
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.output_texture = Some(texture);
        self.output_view = Some(view);
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
                size: byte_len.max(16),
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
        let (Some(pipeline), Some(bgl), Some(params_buf), Some(output_view)) = (
            self.trace_pipeline.as_ref(),
            self.trace_bgl.as_ref(),
            self.params_buf.as_ref(),
            self.output_view.as_ref(),
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
                size: 16,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let lights_buf = self.lights_buf.as_ref().unwrap();

        // ── Upload TraceParams ─────────────────────────────────────
        let params = self.pack_params(camera, light_count);
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
            ],
        });

        // ── Dispatch ───────────────────────────────────────────────
        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PathTracer/Trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(wg_x, wg_y, 1);

        // Advance frame index (for temporal jitter / accumulation)
        self.frame_index += 1;
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

    /// Pack camera + render parameters into a 192-byte (48 x f32) array
    /// matching the WGSL `TraceParams` struct layout.
    fn pack_params(&self, camera: &Camera, light_count: u32) -> [f32; 48] {
        // inv_view_proj = (P * V)^-1
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        let inv_vp = vp.inverse();

        let mut params = [0.0f32; 48];

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

        // bytes 96-99: u32 use_blue_noise (always enabled)
        params[24] = f32::from_bits(1u32);

        // bytes 100-103: u32 fixed_seed (0 = normal temporal jitter)
        params[25] = f32::from_bits(0u32);

        // bytes 104-107: u32 max_bounces
        params[26] = f32::from_bits(self.max_bounces);

        // bytes 108-111: _pad0

        // bytes 112-123: vec3f ambient_color (default dark gray sky)
        params[28] = 0.05;
        params[29] = 0.07;
        params[30] = 0.1;

        // bytes 124-127: _pad1
        // bytes 128-191: padding vec4f * 4 (zeroed)

        params
    }
}
