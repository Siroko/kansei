use crate::cameras::Camera;
use crate::lights::Light;
use crate::objects::Scene;
use crate::renderers::Renderer;

use crate::rt::RtBackend;

use super::accel;
use super::blue_noise::generate_blue_noise;
use super::bvh_builder::GPUBVHData;
use super::material::PathTracerMaterial;

/// Size of the TraceParams uniform in bytes (48 floats * 4 bytes).
const TRACE_PARAMS_SIZE: u64 = 192;

/// Owns the path trace compute pipeline and dispatches trace passes.
///
/// The trace shader is formed by concatenating `intersection.wgsl`, `instances.wgsl`, the
/// scene's traversal for its backend (`traversal.wgsl`'s BVH4s, or `traversal_query.wgsl`'s
/// hardware ray queries; `GPUBVHData::backend`) with its bindings, and `trace.wgsl`. The pipeline
/// uses a single bind group (group 0), made for each backend the first time it traces one:
///
///   0  TraceParams uniform
///   1  output_tex  storage texture (rgba16float, write)
///   2  materials   storage buffer  (read)
///   3  scene_lights storage buffer (read)
///   4  blue_noise  storage buffer  (read)
///   5  triangles   storage buffer  (read) — array<f32>
///   6  bvh4_nodes  storage buffer  (read) — array<vec4<f32>> (software)
///   7  tlas_bvh4_nodes storage buffer (read) — array<vec4<f32>> (software), or the scene's
///      acceleration structure (hardware)
///   8  instances   storage buffer  (read) — array of Instance
///   9  prev_frame  texture_2d (read) — previous frame for temporal accumulation
///  10  the ray counter, while counting rays (`set_count_rays`)
pub struct PathTracer {
    /// the trace pipelines made so far, by backend and whether they count rays
    pipelines: Vec<TracePipeline>,
    count_rays: bool,
    ray_counter: Option<wgpu::Buffer>,
    params_buf: Option<wgpu::Buffer>,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    prev_texture: Option<wgpu::Texture>,
    prev_view: Option<wgpu::TextureView>,
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
    use_blue_noise: bool,
}

struct TracePipeline {
    backend: RtBackend,
    count_rays: bool,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
}

/// The ray counter's binding.
const COUNTER_BINDING: u32 = 10;

/// The trace shader for `backend`, counting rays or not.
pub(crate) fn trace_wgsl(backend: RtBackend, count_rays: bool) -> String {
    accel::assemble(format!(
        "{}\n{}",
        accel::traversal_wgsl(backend, 5, count_rays.then_some(COUNTER_BINDING)),
        include_str!("shaders/trace.wgsl")
    ))
}

impl TracePipeline {
    fn new(device: &wgpu::Device, backend: RtBackend, count_rays: bool) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("PathTracer/trace"),
            source: wgpu::ShaderSource::Wgsl(trace_wgsl(backend, count_rays).into()),
        });

        // ── Bind group layout ──────────────────────────────────────
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let mut entries = vec![
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
            // 2: materials, 3: scene_lights, 4: blue_noise
            storage(2),
            storage(3),
            storage(4),
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
        ];
        // 5-8: the scene
        entries.extend(accel::traversal_layout_entries(backend, 5));
        if count_rays {
            entries.push(accel::counter_layout_entry(COUNTER_BINDING));
        }
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("PathTracer/BGL"), entries: &entries });

        // ── Pipeline ───────────────────────────────────────────────
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("PathTracer/PipelineLayout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("PathTracer/Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("trace_main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { backend, count_rays, layout, pipeline }
    }
}

impl PathTracer {
    /// Create a new `PathTracer` from an initialized renderer.
    ///
    /// This uploads the blue noise LUT; the trace pipeline is made at the first `trace`, for the
    /// scene's backend. Call [`resize`] before the first [`trace`].
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();

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
            pipelines: Vec::new(),
            count_rays: false,
            ray_counter: None,
            params_buf: Some(params_buf),
            output_texture: None,
            output_view: None,
            prev_texture: None,
            prev_view: None,
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

    /// Count the rays the traces cast (each closest-hit and any-hit query, an atomic add in the
    /// shader: slower, for measuring rays per second); `take_ray_count` reads them.
    pub fn set_count_rays(&mut self, enabled: bool) {
        self.count_rays = enabled;
    }

    /// The rays the traces cast since counting started or the last take (a 32-bit count: take it
    /// every frame or so at 1080p), waiting for the GPU; then counting again from 0.
    pub fn take_ray_count(&mut self, renderer: &Renderer) -> u32 {
        let Some(counter) = self.ray_counter.as_ref() else { return 0 };
        let count: Vec<u32> = renderer.read_back_buffer_sync(counter, 4);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("PathTracer/RayCounter") });
        encoder.clear_buffer(counter, 0, None);
        self.queue.submit(Some(encoder.finish()));
        count[0]
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
    ///   - `tlas_buf` from `TLASBuilder` (the TLAS BVH4 node buffer; unused when
    ///     `bvh_data.backend()` is hardware)
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
        // The pipeline for the scene's backend
        let backend = bvh_data.backend();
        let count_rays = self.count_rays;
        let index = match self.pipelines.iter().position(|p| p.backend == backend && p.count_rays == count_rays) {
            Some(index) => index,
            None => {
                self.pipelines.push(TracePipeline::new(&self.device, backend, count_rays));
                self.pipelines.len() - 1
            }
        };
        if count_rays && self.ray_counter.is_none() {
            self.ray_counter = Some(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PathTracer/RayCounter"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        // Guard: all resources must be present
        let TracePipeline { layout: bgl, pipeline, .. } = &self.pipelines[index];
        let (Some(params_buf), Some(output_view), Some(prev_view)) = (
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

        // ── Upload TraceParams ─────────────────────────────────────
        let params = self.pack_params(camera, light_count);
        self.queue
            .write_buffer(params_buf, 0, bytemuck::cast_slice(&params));

        // ── Create bind group ──────────────────────────────────────
        let mut entries = vec![
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
                binding: 9,
                resource: wgpu::BindingResource::TextureView(prev_view),
            },
        ];
        entries.extend(bvh_data.bind_group_entries(tlas_buf, 5));
        if let (true, Some(counter)) = (count_rays, self.ray_counter.as_ref()) {
            entries.push(wgpu::BindGroupEntry { binding: COUNTER_BINDING, resource: counter.as_entire_binding() });
        }
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PathTracer/BG"),
            layout: bgl,
            entries: &entries,
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

        // bytes 96-99: u32 use_blue_noise
        params[24] = f32::from_bits(if self.use_blue_noise { 1u32 } else { 0u32 });

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
