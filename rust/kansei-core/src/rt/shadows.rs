//! Ray-traced direct light and shadows: [`RtShadowsEffect`].

use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;
use crate::lights::Light;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

use super::{RtGrid, RtGridHandle, RT_GRID_WGSL, RT_OPAQUE_WGSL};

/// The most lights one `RtShadowsEffect` shades (the first directional and spot lights).
pub const RT_SHADOW_MAX_LIGHTS: usize = 8;

/// GBuffer output for the surfaces `RtShadowsEffect` lights: `kansei_gbuffer_out_rt_lit`. Prepend
/// `materials::GBUFFER_OUT_WGSL`.
pub const RT_SHADOWS_GBUFFER_WGSL: &str = include_str!("shaders/rt_shadows_gbuffer.wgsl");

const COMMON_WGSL: &str = include_str!("shaders/rt_shadows.wgsl");

fn trace_wgsl() -> String {
    format!("{RT_GRID_WGSL}\n{}\n{RT_OPAQUE_WGSL}\n{COMMON_WGSL}\n{}", RtGrid::bindings_wgsl(1, 0), include_str!("shaders/rt_shadows_trace.wgsl"))
}

fn filter_wgsl() -> String {
    format!("{COMMON_WGSL}\n{}", include_str!("shaders/rt_shadows_filter.wgsl"))
}

fn composite_wgsl() -> String {
    format!("{COMMON_WGSL}\n{}", include_str!("shaders/rt_shadows_composite.wgsl"))
}

/// What the effect shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RtShadowsView {
    /// The lit image.
    #[default]
    Lit,
    /// The visibility of one light (`RtShadowsEffect::debug_light`), white lit, black shadowed.
    Visibility,
    /// The direct light alone.
    Direct,
}

/// The WGSL `RtShadowParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    view_proj: [f32; 16],
    prev_view_proj: [f32; 16],
    full_size: [f32; 2],
    trace_size: [f32; 2],
    frame: u32,
    downscale: u32,
    num_lights: u32,
    flags: u32,
    contact_length: f32,
    contact_thickness: f32,
    temporal_alpha: f32,
    max_distance: f32,
    view: u32,
    debug_light: u32,
    step_width: u32,
    last_step: u32,
    phi_depth: f32,
    phi_normal: f32,
    intensity: f32,
    max_history: f32,
}

/// The WGSL `RtShadowLight`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct LightGpu {
    position: [f32; 3],
    kind: u32,
    direction: [f32; 3],
    range: f32,
    color: [f32; 3],
    cos_outer: f32,
    axis_u: [f32; 3],
    cos_inner: f32,
    axis_v: [f32; 3],
    radius: f32,
}

const FLAG_HISTORY: u32 = 1;
const FLAG_CONTACT: u32 = 2;
const ITERATIONS: usize = 3;

struct Targets {
    size: (u32, u32),
    trace_size: (u32, u32),
    /// this frame's raw visibility (lights 0-3, 4-7) and guide
    raw: [wgpu::TextureView; 3],
    /// the accumulated visibility and guide (with the history length), ping-ponged by frame
    history: [[wgpu::TextureView; 3]; 2],
    /// the wavelet's ping-pong (lights 0-3, 4-7)
    filtered: [[wgpu::TextureView; 2]; 2],
}

struct Gpu {
    /// one per pass that reads its own step: the trace and temporal (0), each wavelet step
    params: Vec<wgpu::Buffer>,
    lights: wgpu::Buffer,
    trace_bgl: wgpu::BindGroupLayout,
    grid_bgl: wgpu::BindGroupLayout,
    temporal_bgl: wgpu::BindGroupLayout,
    atrous_bgl: wgpu::BindGroupLayout,
    composite_bgl: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    temporal: wgpu::ComputePipeline,
    atrous: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    grid_group: Option<(u64, wgpu::BindGroup)>,
    targets: Option<Targets>,
}

/// Direct light with ray-traced shadows, for the surfaces whose material asks for it
/// (`RT_SHADOWS_GBUFFER_WGSL`'s `kansei_gbuffer_out_rt_lit`: such a surface leaves the sun and the
/// spot lights out of its own colour). For each pixel and light (the scene's directional and spot
/// lights, `update_lights`, at most [`RT_SHADOW_MAX_LIGHTS`]) a shadow ray goes through the
/// renderer's grid of triangles (`Renderer::enable_rt_grid`) toward a point of the emitter: a
/// cone of `sun_angle` round a directional light, a disk of the spot light's `source_radius`, or a
/// rectangle (`set_rect_emitter`); a new point every frame, so the penumbrae are soft where the
/// emitter is large and sharpen at contacts. A short screen-space ray (`contact_length`) catches
/// what the grid's cells are too coarse for: feet, small props. Traced at half resolution (or
/// full), accumulated over frames (reprojected by the velocity target, else by depth) and filtered
/// by an a-trous wavelet with depth and normal edge stops, then upsampled the same way and lit:
/// Lambert on the albedo plus GGX toward the emitter's point closest to the reflected ray, except
/// on reflective surfaces, whose highlights `RtReflectionsEffect` traces.
///
/// Put it first in the chain, before the GI (which adds the indirect light on top). Light sources
/// the rays should pass (lamp shades, light panels) stay out of the grid, and glass lets them
/// through. The shadow maps still serve what needs them (fog, voxel GI's light injection, other
/// materials).
pub struct RtShadowsEffect {
    pub enabled: bool,
    pub view: RtShadowsView,
    pub debug_light: u32,
    /// Trace one texel in each 2 x 2 (true, the default) or every pixel.
    pub half_resolution: bool,
    /// Metres the screen-space contact rays walk (0: none), and how thick a depth sample counts.
    pub contact_length: f32,
    pub contact_thickness: f32,
    /// The sun's angular radius (radians; the real one is 0.0047): the softness of its shadows.
    pub sun_angle: f32,
    /// How far a ray toward the sun looks (m).
    pub max_distance: f32,
    /// The temporal blend's floor, the longest history, and the wavelet's edge stops.
    pub temporal_alpha: f32,
    pub max_history: f32,
    pub phi_depth: f32,
    pub phi_normal: f32,
    /// Wavelet steps (0-3).
    pub iterations: u32,
    /// Scale of the direct light (1 is physical).
    pub intensity: f32,
    grid: RtGridHandle,
    lights: Vec<LightGpu>,
    rects: Vec<(usize, [f32; 2])>,
    frame: u32,
    prev_view_proj: Option<glam::Mat4>,
    last_camera_frame: Option<u32>,
    gpu: Option<Gpu>,
}

impl RtShadowsEffect {
    /// Shadows traced through `grid` (`SceneRtGrid::handle`).
    pub fn new(grid: RtGridHandle) -> Self {
        Self {
            enabled: true,
            view: RtShadowsView::Lit,
            debug_light: 0,
            half_resolution: true,
            contact_length: 0.25,
            contact_thickness: 0.12,
            sun_angle: 0.02,
            max_distance: 60.0,
            temporal_alpha: 0.12,
            max_history: 24.0,
            phi_depth: 1.0,
            phi_normal: 32.0,
            iterations: 2,
            intensity: 1.0,
            grid,
            lights: Vec::new(),
            rects: Vec::new(),
            frame: 0,
            prev_view_proj: None,
            last_camera_frame: None,
            gpu: None,
        }
    }

    /// The `index`th spot light (in `update_lights`' order, counting spot lights only) emits from a
    /// rectangle `width` x `height` (m) facing along its axis, instead of its disk.
    pub fn set_rect_emitter(&mut self, index: usize, width: f32, height: f32) {
        self.rects.retain(|r| r.0 != index);
        self.rects.push((index, [width, height]));
    }

    /// The scene's lights (`Scene::lights`), each frame they may change: its directional lights,
    /// then its spot lights, up to [`RT_SHADOW_MAX_LIGHTS`] in all; the slots' order is the
    /// visibility view's `debug_light`.
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        let (mut dir, mut spot) = (Vec::new(), Vec::new());
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let c = l.effective_color();
                    let d = glam::Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or_zero();
                    dir.push(LightGpu { kind: 0, direction: d.to_array(), color: [c.x, c.y, c.z], radius: self.sun_angle, ..Default::default() });
                }
                Light::Spot(l) => {
                    let c = l.effective_color();
                    let d = glam::Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or_zero();
                    let (inner, outer) = l.cone();
                    let mut g = LightGpu {
                        position: [l.position.x, l.position.y, l.position.z],
                        kind: 1,
                        direction: d.to_array(),
                        range: l.range.max(1e-3),
                        color: [c.x, c.y, c.z],
                        cos_outer: outer.cos(),
                        cos_inner: inner.cos(),
                        radius: l.source_radius.max(0.0),
                        ..Default::default()
                    };
                    if let Some((_, [w, h])) = self.rects.iter().find(|r| r.0 == spot.len()) {
                        let any = if d.y.abs() < 0.9 { glam::Vec3::Y } else { glam::Vec3::X };
                        let u = d.cross(any).normalize();
                        let v = d.cross(u).normalize();
                        g.axis_u = (u * w * 0.5).to_array();
                        g.axis_v = (v * h * 0.5).to_array();
                    }
                    spot.push(g);
                }
                Light::Point(_) | Light::Area(_) => {}
            }
        }
        dir.extend(spot);
        dir.truncate(RT_SHADOW_MAX_LIGHTS);
        self.lights = dir;
    }

    /// Start the accumulation over (after a cut).
    pub fn reset_history(&mut self) {
        self.prev_view_proj = None;
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage_ro = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: true }, has_dynamic_offset: false, min_binding_size: None };
        let d2 = wgpu::TextureViewDimension::D2;
        let tex = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: d2, multisampled: false };
        let depth = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false };
        let storage = |format| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format, view_dimension: d2 };
        let (f16, f32x4) = (wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Rgba32Float);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let trace_bgl = layout(
            "RtShadows/Trace",
            &[entry(0, uniform), entry(1, storage_ro), entry(2, depth), entry(3, tex), entry(4, tex), entry(5, storage(f16)), entry(6, storage(f16)), entry(7, storage(f32x4))],
        );
        let grid_bgl = layout("RtShadows/Grid", &RtGrid::layout_entries(0, compute));
        let temporal_bgl = layout(
            "RtShadows/Temporal",
            &[
                entry(0, uniform),
                entry(2, depth),
                entry(3, tex),
                entry(4, tex),
                entry(5, tex),
                entry(6, tex),
                entry(7, tex),
                entry(8, tex),
                entry(9, tex),
                entry(10, storage(f16)),
                entry(11, storage(f16)),
                entry(12, storage(f32x4)),
            ],
        );
        let atrous_bgl = layout("RtShadows/Atrous", &[entry(0, uniform), entry(13, tex), entry(14, tex), entry(15, tex), entry(16, storage(f16)), entry(17, storage(f16))]);
        let composite_bgl = layout(
            "RtShadows/Composite",
            &[
                entry(0, uniform),
                entry(1, storage_ro),
                entry(2, depth),
                entry(3, tex),
                entry(4, tex),
                entry(5, tex),
                entry(6, tex),
                entry(7, tex),
                entry(8, tex),
                entry(9, tex),
                entry(10, storage(f16)),
            ],
        );
        let pipeline = |label: &str, code: &str, entry_point: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let filter = filter_wgsl();
        let trace = pipeline("RtShadows/Trace", &trace_wgsl(), "main", &[&trace_bgl, &grid_bgl]);
        let temporal = pipeline("RtShadows/Temporal", &filter, "temporal", &[&temporal_bgl]);
        let atrous = pipeline("RtShadows/Atrous", &filter, "atrous", &[&atrous_bgl]);
        let composite = pipeline("RtShadows/Composite", &composite_wgsl(), "main", &[&composite_bgl]);
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let params = (0..=ITERATIONS).map(|_| buffer("RtShadows/Params", std::mem::size_of::<ParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST)).collect();
        let lights = buffer("RtShadows/Lights", (RT_SHADOW_MAX_LIGHTS * std::mem::size_of::<LightGpu>()) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        self.gpu = Some(Gpu { params, lights, trace_bgl, grid_bgl, temporal_bgl, atrous_bgl, composite_bgl, trace, temporal, atrous, composite, grid_group: None, targets: None });
    }

    fn make_targets(device: &wgpu::Device, size: (u32, u32), trace_size: (u32, u32)) -> Targets {
        let texture = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: trace_size.0, height: trace_size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let (f16, f32x4) = (wgpu::TextureFormat::Rgba16Float, wgpu::TextureFormat::Rgba32Float);
        let set = |label| [texture(label, f16), texture(label, f16), texture(label, f32x4)];
        Targets {
            size,
            trace_size,
            raw: set("RtShadows/Raw"),
            history: [set("RtShadows/History"), set("RtShadows/History")],
            filtered: [[texture("RtShadows/Filtered", f16), texture("RtShadows/Filtered", f16)], [texture("RtShadows/Filtered", f16), texture("RtShadows/Filtered", f16)]],
        }
    }
}

impl PostProcessingEffect for RtShadowsEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        gbuffer: &GBuffer,
        input: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        output: &wgpu::TextureView,
        camera: &Camera,
        width: u32,
        height: u32,
    ) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        let downscale = if self.half_resolution { 2 } else { 1 };
        let (tw, th) = (width.div_ceil(downscale), height.div_ceil(downscale));
        {
            let gpu = self.gpu.as_mut().unwrap();
            if gpu.targets.as_ref().is_none_or(|t| t.size != (width, height) || t.trace_size != (tw, th)) {
                gpu.targets = Some(Self::make_targets(device, (width, height), (tw, th)));
                self.prev_view_proj = None;
            }
        }
        let camera_frame = camera.frame();
        if self.last_camera_frame.is_some_and(|f| camera_frame != f && camera_frame != f.wrapping_add(1)) {
            self.prev_view_proj = None;
        }
        self.last_camera_frame = Some(camera_frame);
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let view_proj = proj * view;
        let mut flags = 0;
        if self.prev_view_proj.is_some() {
            flags |= FLAG_HISTORY;
        }
        if self.contact_length > 0.0 {
            flags |= FLAG_CONTACT;
        }
        let iterations = (self.iterations as usize).min(ITERATIONS);
        let base = ParamsGpu {
            inv_proj: proj.inverse().to_cols_array(),
            inv_view: view.inverse().to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            prev_view_proj: self.prev_view_proj.unwrap_or(view_proj).to_cols_array(),
            full_size: [width as f32, height as f32],
            trace_size: [tw as f32, th as f32],
            frame: self.frame,
            downscale,
            num_lights: self.lights.len() as u32,
            flags,
            contact_length: self.contact_length.max(0.0),
            contact_thickness: self.contact_thickness.max(0.01),
            temporal_alpha: self.temporal_alpha.clamp(0.02, 1.0),
            max_distance: self.max_distance.max(0.1),
            view: self.view as u32,
            debug_light: self.debug_light,
            step_width: 1,
            last_step: 0,
            phi_depth: self.phi_depth.max(1e-3),
            phi_normal: self.phi_normal.max(0.0),
            intensity: self.intensity.max(0.0),
            max_history: self.max_history.max(1.0),
        };
        let gpu = self.gpu.as_mut().unwrap();
        for (k, buffer) in gpu.params.iter().enumerate() {
            let p = ParamsGpu { step_width: 1 << k.saturating_sub(1), last_step: (k == iterations) as u32, ..base };
            queue.write_buffer(buffer, 0, bytemuck::bytes_of(&p));
        }
        let mut lights = self.lights.clone();
        lights.resize(RT_SHADOW_MAX_LIGHTS, LightGpu::default());
        queue.write_buffer(&gpu.lights, 0, bytemuck::cast_slice(&lights));

        let (buffers, generation) = self.grid.buffers();
        if gpu.grid_group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtShadows/Grid"),
                layout: &gpu.grid_bgl,
                entries: &[b(0, buffers[0].as_entire_binding()), b(1, buffers[1].as_entire_binding()), b(2, buffers[2].as_entire_binding())],
            });
            gpu.grid_group = Some((generation, group));
        }
        let cur = (self.frame % 2) as usize;
        let prev = 1 - cur;
        let t = gpu.targets.as_ref().unwrap();
        let p0 = gpu.params[0].as_entire_binding();
        let trace_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtShadows/Trace"),
            layout: &gpu.trace_bgl,
            entries: &[
                b(0, p0.clone()),
                b(1, gpu.lights.as_entire_binding()),
                b(2, tex(depth)),
                b(3, tex(&gbuffer.normal_view)),
                b(4, tex(&gbuffer.emissive_view)),
                b(5, tex(&t.raw[0])),
                b(6, tex(&t.raw[1])),
                b(7, tex(&t.raw[2])),
            ],
        });
        let temporal_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtShadows/Temporal"),
            layout: &gpu.temporal_bgl,
            entries: &[
                b(0, p0.clone()),
                b(2, tex(depth)),
                b(3, tex(&gbuffer.velocity_view)),
                b(4, tex(&t.raw[0])),
                b(5, tex(&t.raw[1])),
                b(6, tex(&t.raw[2])),
                b(7, tex(&t.history[prev][0])),
                b(8, tex(&t.history[prev][1])),
                b(9, tex(&t.history[prev][2])),
                b(10, tex(&t.history[cur][0])),
                b(11, tex(&t.history[cur][1])),
                b(12, tex(&t.history[cur][2])),
            ],
        });
        let (gx, gy) = (tw.div_ceil(8), th.div_ceil(8));
        let pass = |encoder: &mut wgpu::CommandEncoder, label: &'static str, pipeline: &wgpu::ComputePipeline, groups: &[&wgpu::BindGroup], size: (u32, u32)| {
            let stamp = crate::profiling::gpu_pass(label);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(pipeline);
            for (i, g) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, *g, &[]);
            }
            pass.dispatch_workgroups(size.0, size.1, 1);
        };
        pass(encoder, "RtShadows/Trace", &gpu.trace, &[&trace_group, &gpu.grid_group.as_ref().unwrap().1], (gx, gy));
        pass(encoder, "RtShadows/Temporal", &gpu.temporal, &[&temporal_group], (gx, gy));
        // the wavelet: from the history into the ping-pong
        let mut source: [&wgpu::TextureView; 2] = [&t.history[cur][0], &t.history[cur][1]];
        const LABELS: [&str; ITERATIONS] = ["RtShadows/Atrous1", "RtShadows/Atrous2", "RtShadows/Atrous3"];
        for (i, label) in LABELS.iter().enumerate().take(iterations) {
            let dst = &t.filtered[i % 2];
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtShadows/Atrous"),
                layout: &gpu.atrous_bgl,
                entries: &[b(0, gpu.params[i + 1].as_entire_binding()), b(13, tex(source[0])), b(14, tex(source[1])), b(15, tex(&t.history[cur][2])), b(16, tex(&dst[0])), b(17, tex(&dst[1]))],
            });
            pass(encoder, label, &gpu.atrous, &[&group], (gx, gy));
            source = [&dst[0], &dst[1]];
        }
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtShadows/Composite"),
            layout: &gpu.composite_bgl,
            entries: &[
                b(0, p0),
                b(1, gpu.lights.as_entire_binding()),
                b(2, tex(depth)),
                b(3, tex(&gbuffer.normal_view)),
                b(4, tex(&gbuffer.albedo_view)),
                b(5, tex(&gbuffer.emissive_view)),
                b(6, tex(input)),
                b(7, tex(source[0])),
                b(8, tex(source[1])),
                b(9, tex(&t.history[cur][2])),
                b(10, tex(output)),
            ],
        });
        pass(encoder, "RtShadows/Composite", &gpu.composite, &[&composite_group], (width.div_ceil(8), height.div_ceil(8)));
        self.frame = self.frame.wrapping_add(1);
        self.prev_view_proj = Some(view_proj);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

fn tex(view: &wgpu::TextureView) -> wgpu::BindingResource<'_> {
    wgpu::BindingResource::TextureView(view)
}

fn b(binding: u32, resource: wgpu::BindingResource<'_>) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        module
    }

    fn struct_span(module: &naga::Module, name: &str) -> usize {
        module.types.iter().find(|(_, t)| t.name.as_deref() == Some(name)).map(|(_, t)| match &t.inner {
            naga::TypeInner::Struct { span, .. } => *span as usize,
            _ => panic!("{name} is not a struct"),
        }).unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn shaders_validate_rt_shadows() {
        let trace = validate("rt_shadows_trace", &trace_wgsl());
        assert_eq!(struct_span(&trace, "RtShadowParams"), std::mem::size_of::<ParamsGpu>());
        assert_eq!(struct_span(&trace, "RtShadowLight"), std::mem::size_of::<LightGpu>());
        validate("rt_shadows_filter", &filter_wgsl());
        validate("rt_shadows_composite", &composite_wgsl());
        validate("rt_shadows_gbuffer", &format!("{}\n{RT_SHADOWS_GBUFFER_WGSL}", crate::materials::GBUFFER_OUT_WGSL));
    }
}
