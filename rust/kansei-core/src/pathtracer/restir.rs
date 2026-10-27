use crate::cameras::Camera;
use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};

use super::bvh_builder::GPUBVHData;
use crate::rt::RtBackend;

/// Uniform parameters for the ReSTIR DI shader.
///
/// Layout matches the WGSL `ReSTIRParams` struct:
///   mat4x4f inv_view_proj   (64 bytes)
///   mat4x4f prev_view_proj  (64 bytes)
///   vec3f   camera_pos       (12 bytes)
///   u32     frame_index      (4 bytes)
///   u32     width            (4 bytes)
///   u32     height           (4 bytes)
///   u32     light_count      (4 bytes)
///   u32     max_history      (4 bytes)
/// Total: 160 bytes
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct ReSTIRParams {
    inv_view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    frame_index: u32,
    width: u32,
    height: u32,
    light_count: u32,
    max_history: u32,
}

const PARAMS_SIZE: u64 = std::mem::size_of::<ReSTIRParams>() as u64;

/// Reservoir stride: 3 vec4f per pixel (48 bytes).
const RESERVOIR_VEC4_PER_PIXEL: u32 = 3;

/// The ReSTIR passes for one backend of the scene.
struct ReSTIRPipelines {
    backend: RtBackend,
    generate_pipeline: wgpu::ComputePipeline,
    spatial_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

/// The ReSTIR shader for the scene's `backend`.
pub(crate) fn restir_wgsl(backend: RtBackend) -> String {
    super::accel::assemble(format!("{}\n{}", super::accel::traversal_wgsl(backend, 4, None), include_str!("shaders/restir-di.wgsl")))
}

impl ReSTIRPipelines {
    fn new(device: &wgpu::Device, backend: RtBackend) -> Self {
        let shader_module = super::accel::shader_module(device, backend, "ReSTIR/Shader", &restir_wgsl(backend));

        // ── Bind group layout ────────────────────────────────────────
        //
        //  0: depth_tex           texture_depth_2d
        //  1: normal_tex          texture_2d<f32>
        //  2: params              uniform
        //  3: scene_lights        storage (read)
        //  4-7: the scene        (`GPUBVHData::layout_entries`: triangles, BLAS nodes,
        //                         TLAS nodes or the acceleration structure, instances)
        //  8: materials           storage (read)  -- unused by the shader
        //  9: materials           storage (read)  -- from restir-di.wgsl binding 9
        // 10: reservoir_prev      storage (read)
        // 11: reservoir_cur       storage (read_write)
        // 12: direct_light_out    storage texture (write) -- spatial pass only
        //
        // Note: The restir-di.wgsl declares its own bindings 0-3 and 9-12;
        // `accel::traversal_wgsl` declares 4-7.

        let mut entries = vec![
            // 0: depth_tex
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 1: normal_tex
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // 2: params
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(PARAMS_SIZE),
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
            // 8: materials (from preamble)
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
            // 9: materials (from restir-di.wgsl)
            wgpu::BindGroupLayoutEntry {
                binding: 9,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 10: reservoir_prev
            wgpu::BindGroupLayoutEntry {
                binding: 10,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 11: reservoir_cur
            wgpu::BindGroupLayoutEntry {
                binding: 11,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // 12: direct_light_out (storage texture, write)
            wgpu::BindGroupLayoutEntry {
                binding: 12,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
        ];
        // 4-7: the scene
        entries.extend(super::accel::traversal_layout_entries(backend, 4));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("ReSTIR/BGL"), entries: &entries });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReSTIR/PipelineLayout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let generate_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ReSTIR/Generate"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("restir_generate"),
                compilation_options: Default::default(),
                cache: None,
            });

        let spatial_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("ReSTIR/Spatial"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("restir_spatial"),
                compilation_options: Default::default(),
                cache: None,
            });

        Self { backend, generate_pipeline, spatial_pipeline, bgl }
    }
}

/// ReSTIR Direct Illumination.
///
/// Two-pass compute pipeline:
///   1. **Generate** — streams all lights through RIS to build initial
///      reservoirs, then merges with the previous frame's reservoir
///      (temporal reuse).
///   2. **Spatial** — merges with 5 random neighbor reservoirs that pass
///      depth/normal similarity checks, then performs a shadow test and
///      evaluates the final direct lighting contribution.
///
/// The reservoir data is double-buffered (ping-pong) for temporal reuse.
pub struct ReSTIR {
    /// the passes for each backend traced so far
    pipelines: Vec<ReSTIRPipelines>,
    reservoir_a: Option<wgpu::Buffer>,
    reservoir_b: Option<wgpu::Buffer>,
    output_tex: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    params_buf: wgpu::Buffer,
    ping: bool,
    width: u32,
    height: u32,
    /// Maximum temporal history length (default 20). Higher values give
    /// smoother results but slower convergence when lighting changes.
    pub history_cap: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl ReSTIR {
    /// Create the ReSTIR DI pipelines and parameter buffer.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReSTIR/Params"),
            size: PARAMS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            pipelines: Vec::new(),
            reservoir_a: None,
            reservoir_b: None,
            output_tex: None,
            output_view: None,
            params_buf,
            ping: false,
            width: 0,
            height: 0,
            history_cap: 20,
            device,
            queue,
        }
    }

    /// Recreate reservoir buffers and output texture for the given resolution.
    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        if width == self.width && height == self.height {
            return;
        }
        self.width = width;
        self.height = height;

        // 3 vec4f per pixel, each vec4f = 16 bytes
        let reservoir_size =
            (width * height * RESERVOIR_VEC4_PER_PIXEL) as u64 * 16;

        let make_reservoir = |label: &str| {
            self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: reservoir_size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };

        self.reservoir_a = Some(make_reservoir("ReSTIR/ReservoirA"));
        self.reservoir_b = Some(make_reservoir("ReSTIR/ReservoirB"));

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ReSTIR/Output"),
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
                | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.output_tex = Some(texture);
        self.output_view = Some(view);

        self.ping = false;
    }

    /// Dispatch the generate (temporal) + spatial reuse passes.
    ///
    /// # Arguments
    ///
    /// * `encoder`      - Active command encoder
    /// * `depth_view`   - GBuffer resolved depth texture view
    /// * `normal_view`  - GBuffer normal texture view
    /// * `lights_buf`   - Buffer containing `LightData` array
    /// * `light_count`  - Number of lights
    /// * `materials_buf`- Buffer containing `MaterialData` array
    /// * `camera`       - Camera with current view/projection matrices
    /// * `prev_vp`      - Previous frame's view-projection (column-major f32x16)
    /// * `bvh_data`     - GPU BVH data from `BVHBuilder`
    /// * `tlas_buf`     - TLAS BVH4 node buffer from `TLASBuilder` (unused on the
    ///                    hardware backend)
    /// * `frame_index`  - Current frame number
    #[allow(clippy::too_many_arguments)]
    pub fn run(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        depth_view: &wgpu::TextureView,
        normal_view: &wgpu::TextureView,
        lights_buf: &wgpu::Buffer,
        light_count: u32,
        materials_buf: &wgpu::Buffer,
        camera: &Camera,
        prev_vp: &[f32; 16],
        bvh_data: &GPUBVHData,
        tlas_buf: &wgpu::Buffer,
        frame_index: u32,
    ) {
        let (Some(output_view), Some(res_a), Some(res_b)) = (
            self.output_view.as_ref(),
            self.reservoir_a.as_ref(),
            self.reservoir_b.as_ref(),
        ) else {
            return;
        };

        // Compute inverse VP
        let vp = camera.projection_matrix.mul(&camera.view_matrix);
        let inv_vp = vp.inverse();

        let pos = camera.position();
        let params = ReSTIRParams {
            inv_view_proj: *inv_vp.as_slice(),
            prev_view_proj: *prev_vp,
            camera_pos: [pos.x, pos.y, pos.z],
            frame_index,
            width: self.width,
            height: self.height,
            light_count,
            max_history: self.history_cap,
        };
        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&params));

        // Determine ping-pong: prev reads from one, cur writes to other
        let (prev_buf, cur_buf) = if self.ping {
            (res_b, res_a)
        } else {
            (res_a, res_b)
        };

        let backend = bvh_data.backend();
        if !self.pipelines.iter().any(|p| p.backend == backend) {
            self.pipelines.push(ReSTIRPipelines::new(&self.device, backend));
        }
        let pipelines = self.pipelines.iter().find(|p| p.backend == backend).unwrap();
        let mut entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(depth_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(normal_view),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: self.params_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: lights_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: materials_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: materials_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: prev_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: cur_buf.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: wgpu::BindingResource::TextureView(output_view),
            },
        ];
        entries.extend(bvh_data.bind_group_entries(tlas_buf, 4));
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReSTIR/BG"),
            layout: &pipelines.bgl,
            entries: &entries,
        });

        let wg_x = (self.width + 7) / 8;
        let wg_y = (self.height + 7) / 8;

        // Pass 1: Generate (initial reservoir + temporal reuse)
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReSTIR/Generate"),
                timestamp_writes: crate::profiling::gpu_pass("ReSTIR/Generate").as_ref().map(crate::profiling::PassStamp::compute),
            });
            pass.set_pipeline(&pipelines.generate_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Pass 2: Spatial reuse + final shading
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReSTIR/Spatial"),
                timestamp_writes: crate::profiling::gpu_pass("ReSTIR/Spatial").as_ref().map(crate::profiling::PassStamp::compute),
            });
            pass.set_pipeline(&pipelines.spatial_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(wg_x, wg_y, 1);
        }

        // Flip ping-pong for next frame
        self.ping = !self.ping;
    }

    /// Returns the direct lighting output texture view (available after `run()`).
    pub fn output_view(&self) -> Option<&wgpu::TextureView> {
        self.output_view.as_ref()
    }
}
