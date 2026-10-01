//! Occlusion culling: a dense procedural forest of spruces (60 000 by default, three LODs) on
//! terrain with a ridge and a hill. Each LOD is a renderable culled on the GPU per view by frustum
//! and LOD band (`InstanceCulling`), and for the camera also by occlusion
//! (`InstanceCulling::with_occlusion`): two phases against a depth pyramid. The terrain is an
//! ordinary mesh; it writes the depth the pyramid is built from, with the trees seen last frame.
//!
//! The HUD shows the camera's culling stats (`Renderer::culling_stats`) and the GPU time
//! (timestamp queries when the adapter has them, else the interval between frames: run the
//! browser without vsync). Keys: O toggles occlusion culling, F freezes the camera's culling (then
//! move on to see what it culled), C cycles the camera.
//!
//! URL parameters: `cam=valley|forest|ridge|high|fly|sky|edge` (sky: nothing in view, so occlusion's
//! overhead alone), `occlusion=0`, `freeze=1`, `trees=<n>`, `bounds=sphere` (the spheres round
//! the trees' bases, as before tighter bounds, instead of boxes), `scale=<render scale>`, `taa=0`,
//! `t=<seconds>` (the fly path's time, frozen),
//! `size=<w>x<h>` (canvas pixels), `bench=1` (alternate occlusion on and off every 3 s, 8 times,
//! and report the mean GPU time and frame interval of each), `skyocc=1` (the trees occlude the
//! sky's light, `Renderer::enable_sky_occlusion`: the sky ambient is dimmed under the canopy;
//! `skyocc=show` shows the sky visibility, `skyocc=rebuild` starts a rebuild every frame, to time
//! a rebuild's first frame: the top-down pass, the pyramid and a quarter of the volume).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, Sampler, Texture, VertexFormat};
use kansei_core::cameras::{Camera, MOTION_VECTORS_WGSL};
use kansei_core::culling::{CullStats, InstanceCulling};
use kansei_core::geometries::{Geometry, InstancedGeometry, SphereGeometry, Vertex};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::shadows::{SkyOcclusion, SkyOcclusionOptions, SKY_OCCLUSION_WGSL};

/// A diffuse surface in sunlight and haze, writing motion vectors. TREE_* (string replaced) place
/// a unit tree: instance vec4 (xyz base, w height), vec4 (yaw, tint, -, -). SKY_* bind the sky
/// occlusion and dim the sky's light by it.
const SURFACE_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
SKY_BINDINGS
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, TREE_INPUT };
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) tint: f32,
    @location(3) curr: vec4<f32>,
    @location(4) prev: vec4<f32>,
};
struct FOut { @location(0) color: vec4<f32>, @location(4) velocity: vec2<f32> };

@vertex
fn vertex_main(v: VIn) -> VOut {
    var local = v.position.xyz;
    var n = v.normal;
    var tint = 0.5;
    TREE_PLACE
    let world = mesh.world * vec4<f32>(local, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (normal_matrix * vec4<f32>(n, 0.0)).xyz;
    out.world = world.xyz;
    out.tint = tint;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4<f32>(local, 1.0));
    return out;
}

@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal);
    let sun = normalize(vec3<f32>(-0.4, 0.55, -0.7));
    let visibility = SKY_VISIBILITY;
    let sky = mix(vec3<f32>(60.0, 55.0, 45.0), vec3<f32>(900.0, 1100.0, 1500.0), n.y * 0.5 + 0.5) * visibility;
    let base = surface.base_color.rgb * (0.7 + 0.6 * in.tint);
    var lit = base * (sky + vec3<f32>(9000.0, 7600.0, 6000.0) * max(dot(n, sun), 0.0));
    SKY_SHOW
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let haze = 1.0 - exp(-distance(in.world, eye) * 0.0025);
    let color = mix(lit, vec3<f32>(1500.0, 1700.0, 2000.0), haze);
    return FOut(vec4<f32>(color, 1.0), kansei_motion_vector(in.curr, in.prev));
}
"#;

const SKY_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let up = saturate(normalize(in.dir).y);
    return vec4<f32>(mix(vec3<f32>(1500.0, 1700.0, 2000.0), vec3<f32>(700.0, 1100.0, 2200.0), sqrt(up)), 1.0);
}
"#;

/// `skyocc=`: off, on, or showing the sky visibility.
#[derive(Clone, Copy, PartialEq)]
enum SkyOcc {
    Off,
    On,
    Show,
}

/// The layer the trees are on (as well as the default one), the only one occluding the sky: the
/// terrain is solid ground, not canopy.
const TREE_LAYER: u32 = 1 << 1;

fn surface_material(label: &str, base: [f32; 3], tree: bool, sky: Option<(&SkyOcclusion, SkyOcc)>) -> Material {
    let (input, place) = if tree {
        (
            "@location(3) inst: vec4<f32>, @location(4) extra: vec4<f32>,",
            "let c = cos(v.extra.x); let s = sin(v.extra.x); \
             local = vec3<f32>(c * local.x + s * local.z, local.y, -s * local.x + c * local.z) * v.inst.w + v.inst.xyz; \
             n = vec3<f32>(c * n.x + s * n.z, n.y, -s * n.x + c * n.z); \
             tint = v.extra.y;",
        )
    } else {
        ("", "")
    };
    let mut shader = SURFACE_WGSL.replace("TREE_INPUT", input).replace("TREE_PLACE", place);
    let mut bindings = vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)];
    if let Some((_, mode)) = sky {
        shader = format!("{SKY_OCCLUSION_WGSL}\n{shader}")
            .replace(
                "SKY_BINDINGS",
                "@group(0) @binding(1) var sky_volume: texture_3d<f32>;\n\
                 @group(0) @binding(2) var sky_sampler: sampler;\n\
                 @group(0) @binding(3) var<uniform> sky_occlusion: SkyOcclusionParams;",
            )
            .replace("SKY_VISIBILITY", "skyVisibility(sky_volume, sky_sampler, sky_occlusion, in.world)")
            .replace("SKY_SHOW", if mode == SkyOcc::Show { "lit = vec3<f32>(visibility * 2000.0);" } else { "" });
        bindings.extend([Binding::texture_3d(1, ShaderStages::FRAGMENT), Binding::sampler(2, ShaderStages::FRAGMENT), Binding::uniform(3, ShaderStages::FRAGMENT)]);
    } else {
        shader = shader.replace("SKY_BINDINGS", "").replace("SKY_VISIBILITY", "1.0").replace("SKY_SHOW", "");
    }
    let mut m = Material::new(
        label,
        &format!("{MOTION_VECTORS_WGSL}\n{shader}"),
        bindings,
        MaterialOptions { outputs_velocity: true, ..Default::default() },
    );
    m.set_uniform_bindable(0, label, &[base[0], base[1], base[2], 1.0, 0.0, 0.0, 0.0, 0.0f32]);
    if let Some((sky, _)) = sky {
        m.set_bindable(1, Texture::from_view("SkyOcclusion", sky.volume_texture().clone(), sky.volume.clone()));
        m.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
        m.set_bindable(3, ComputeBuffer::from_external("SkyOcclusionParams", sky.params.clone(), BufferType::Uniform));
    }
    m
}

/// Deterministic 0..1 hash.
fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// Half the side of the square the forest covers, in metres.
const EXTENT: f32 = 300.0;

/// Terrain height: a ridge across the valley's north, a hill to its east, rolling ground.
fn ground(x: f32, z: f32) -> f32 {
    let ridge = 42.0 * (-((z + 140.0) / 45.0).powi(2)).exp() * (1.0 - 0.25 * (x * 0.01).sin());
    let hill = 55.0 * (-((x - 170.0).powi(2) + (z + 20.0).powi(2)) / (2.0 * 60.0f32.powi(2))).exp();
    let roll = 2.5 * (x * 0.031).sin() * (z * 0.027).cos() + 1.5 * (x * 0.083 + z * 0.061).sin();
    ridge + hill + roll
}

fn vertex(p: [f32; 3], n: [f32; 3], uv: [f32; 2]) -> Vertex {
    Vertex { position: [p[0], p[1], p[2], 1.0], normal: n, uv }
}

/// The terrain: a `cells` x `cells` grid over the forest's square.
fn terrain(cells: u32) -> Geometry {
    let step = 2.0 * EXTENT / cells as f32;
    let mut vertices = Vec::new();
    for j in 0..=cells {
        for i in 0..=cells {
            let (x, z) = (-EXTENT + i as f32 * step, -EXTENT + j as f32 * step);
            let e = 0.5;
            let n = glam::Vec3::new(ground(x - e, z) - ground(x + e, z), 2.0 * e, ground(x, z - e) - ground(x, z + e)).normalize();
            vertices.push(vertex([x, ground(x, z), z], n.to_array(), [i as f32 / cells as f32, j as f32 / cells as f32]));
        }
    }
    let mut indices = Vec::new();
    let at = |i: u32, j: u32| j * (cells + 1) + i;
    for j in 0..cells {
        for i in 0..cells {
            let (p00, p10, p01, p11) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
            indices.extend_from_slice(&[p00, p01, p10, p10, p01, p11]);
        }
    }
    Geometry::new("Terrain", vertices, indices)
}

/// A closed truncated cone round the y axis, from radius `r0` at `y0` to `r1` at `y1`, in
/// `rings` bands of `segments` quads, with a cap underneath.
fn frustum(vertices: &mut Vec<Vertex>, indices: &mut Vec<u32>, (y0, y1): (f32, f32), (r0, r1): (f32, f32), segments: u32, rings: u32) {
    let base = vertices.len() as u32;
    for k in 0..=rings {
        let f = k as f32 / rings as f32;
        let (y, r) = (y0 + (y1 - y0) * f, r0 + (r1 - r0) * f);
        for s in 0..=segments {
            let a = s as f32 / segments as f32 * std::f32::consts::TAU;
            let n = glam::Vec3::new(a.cos() * (y1 - y0), r0 - r1, a.sin() * (y1 - y0)).normalize();
            vertices.push(vertex([r * a.cos(), y, r * a.sin()], n.to_array(), [s as f32 / segments as f32, f]));
        }
    }
    let row = segments + 1;
    for k in 0..rings {
        for s in 0..segments {
            let (a, b, c, d) = (base + k * row + s, base + k * row + s + 1, base + (k + 1) * row + s, base + (k + 1) * row + s + 1);
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }
    let centre = vertices.len() as u32;
    vertices.push(vertex([0.0, y0, 0.0], [0.0, -1.0, 0.0], [0.5, 0.5]));
    for s in 0..=segments {
        let a = s as f32 / segments as f32 * std::f32::consts::TAU;
        vertices.push(vertex([r0 * a.cos(), y0, r0 * a.sin()], [0.0, -1.0, 0.0], [0.5, 0.5]));
    }
    for s in 0..segments {
        indices.extend_from_slice(&[centre, centre + 1 + s, centre + 2 + s]);
    }
}

/// A spruce of height 1: a trunk and `cones` stacked cones of `segments` x `rings` quads.
fn spruce(segments: u32, rings: u32, cones: u32, label: &str) -> Geometry {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    frustum(&mut vertices, &mut indices, (0.0, 0.3), (0.035, 0.025), segments.min(8), 1);
    for k in 0..cones {
        let f = k as f32 / cones as f32;
        let y0 = 0.15 + 0.62 * f;
        let y1 = if k + 1 == cones { 1.0 } else { y0 + 0.42 - 0.12 * f };
        frustum(&mut vertices, &mut indices, (y0, y1), (0.24 * (1.0 - 0.55 * f), 0.0), segments, rings);
    }
    Geometry::new(label, vertices, indices)
}

/// Camera presets: position and target.
const CAMS: [&str; 7] = ["valley", "forest", "ridge", "high", "fly", "sky", "edge"];

fn eye(x: f32, z: f32, up: f32) -> glam::Vec3 {
    glam::Vec3::new(x, ground(x, z) + up, z)
}

fn place_camera(camera: &mut Camera, cam: &str, t: f32) {
    let (from, to) = match cam {
        // from the meadow: the ridge hides most of the forest behind it
        "valley" => (eye(0.0, 100.0, 1.7), glam::Vec3::new(-10.0, 30.0, -200.0)),
        // among the trees: the nearest trunks and crowns hide the rest
        "forest" => (eye(-120.0, 60.0, 1.7), eye(-160.0, -40.0, 4.0)),
        // on the ridge, looking over the far forest: little is hidden
        "ridge" => (eye(-40.0, -130.0, 6.0), eye(-40.0, -300.0, 0.0)),
        "sky" => (eye(0.0, 100.0, 1.7), eye(0.0, 100.0, 1.7) + glam::Vec3::new(0.001, 100.0, 0.0)),
        // at the meadow's eastern rim, the forest's edge close by
        "edge" => (eye(52.0, 70.0, 1.7), eye(90.0, 40.0, 8.0)),
        // high above: nothing is hidden, occlusion only costs
        "high" => (glam::Vec3::new(-260.0, 220.0, 260.0), glam::Vec3::new(0.0, 0.0, -60.0)),
        // a loop through the valley and over the ridge
        _ => {
            let a = t * 0.05;
            let (x, z) = (140.0 * a.cos(), -60.0 + 170.0 * a.sin());
            let ahead = (a + 0.3, 0.0);
            let (tx, tz) = (140.0 * ahead.0.cos(), -60.0 + 170.0 * ahead.0.sin());
            (eye(x, z, 3.0), eye(tx, tz, 2.0))
        }
    };
    camera.set_position(from.x, from.y, from.z);
    camera.look_at(&Vec3::new(to.x, to.y, to.z));
}

/// GPU frame time from timestamp queries: a no-op compute pass before and after the frame (on
/// Metal an empty pass resolves its timestamps to zero). The browser resolves readbacks late, so
/// a ring of them is in flight.
struct GpuTimer {
    noop: wgpu::ComputePipeline,
    period_ns: f64,
    slots: Vec<TimerSlot>,
    armed: Option<usize>,
    /// measurements (ms) not yet taken
    results: Arc<std::sync::Mutex<Vec<f64>>>,
}

struct TimerSlot {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

impl GpuTimer {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("GpuTimer"), source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()) });
        let buffer = |usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some("GpuTimer"), size: 16, usage, mapped_at_creation: false });
        let slots = (0..16)
            .map(|_| TimerSlot {
                set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("GpuTimer"), ty: wgpu::QueryType::Timestamp, count: 2 }),
                resolve: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                readback: buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
                busy: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Some(Self {
            noop: device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuTimer"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            }),
            period_ns: queue.get_timestamp_period() as f64,
            slots,
            armed: None,
            results: Arc::new(std::sync::Mutex::new(Vec::new())),
        })
    }

    fn stamp(&self, encoder: &mut wgpu::CommandEncoder, slot: usize, index: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GpuTimer"),
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: &self.slots[slot].set,
                beginning_of_pass_write_index: (index == 0).then_some(0),
                end_of_pass_write_index: (index == 1).then_some(1),
            }),
        });
        pass.set_pipeline(&self.noop);
        pass.dispatch_workgroups(1, 1, 1);
    }

    fn begin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.armed = self.slots.iter().position(|s| !s.busy.load(Ordering::Acquire));
        if let Some(slot) = self.armed {
            let mut encoder = device.create_command_encoder(&Default::default());
            self.stamp(&mut encoder, slot, 0);
            queue.submit(Some(encoder.finish()));
        }
    }

    fn end(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(k) = self.armed.take() else { return };
        let slot = &self.slots[k];
        slot.busy.store(true, Ordering::Release);
        let mut encoder = device.create_command_encoder(&Default::default());
        self.stamp(&mut encoder, k, 1);
        encoder.resolve_query_set(&slot.set, 0..2, &slot.resolve, 0);
        encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, 16);
        queue.submit(Some(encoder.finish()));
        let (busy, results, readback, period) = (slot.busy.clone(), self.results.clone(), slot.readback.clone(), self.period_ns);
        slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                let t: [u64; 2] = bytemuck::pod_read_unaligned(&readback.slice(..).get_mapped_range()[..16]);
                readback.unmap();
                if t[1] > t[0] {
                    results.lock().unwrap().push((t[1] - t[0]) as f64 * period / 1.0e6);
                }
            }
            busy.store(false, Ordering::Release);
        });
    }

    /// The measurements (ms) that arrived since the last call.
    fn take(&self) -> Vec<f64> {
        std::mem::take(&mut *self.results.lock().unwrap())
    }
}

/// `bench=1`: alternate occlusion on and off, averaging the GPU time and the frame interval of
/// each (after a warm-up, and ignoring the start of each phase).
struct Bench {
    start: f64,
    /// (GPU ms, GPU samples, frame intervals ms, frames) with occlusion off and on
    sums: [(f64, u32, f64, u32); 2],
    last_frame: f64,
    report: Option<String>,
}

const BENCH_WARMUP_MS: f64 = 3000.0;
const BENCH_PHASE_MS: f64 = 3000.0;
const BENCH_SETTLE_MS: f64 = 500.0;
const BENCH_PHASES: u32 = 8;

impl Bench {
    /// (occlusion on, measuring) at `now`, or None when done.
    fn phase(&self, now: f64) -> Option<(bool, bool)> {
        let t = now - self.start - BENCH_WARMUP_MS;
        if t < 0.0 {
            return Some((true, false));
        }
        let phase = (t / BENCH_PHASE_MS) as u32;
        (phase < BENCH_PHASES).then_some((phase.is_multiple_of(2), t % BENCH_PHASE_MS >= BENCH_SETTLE_MS))
    }

    fn record(&mut self, gpu: &[f64], now: f64) {
        if let Some((on, true)) = self.phase(now) {
            let sum = &mut self.sums[on as usize];
            sum.0 += gpu.iter().sum::<f64>();
            sum.1 += gpu.len() as u32;
            sum.2 += now - self.last_frame;
            sum.3 += 1;
        }
        self.last_frame = now;
        if self.phase(now).is_none() && self.report.is_none() {
            let mean = |(sum, n, _, _): (f64, u32, f64, u32)| if n > 0 { format!("{:.2} ms GPU ({n} samples)", sum / n as f64) } else { "no GPU timestamps".into() };
            let interval = |(_, _, sum, n): (f64, u32, f64, u32)| format!("{:.2} ms/frame ({n} frames)", sum / n.max(1) as f64);
            self.report = Some(format!("bench: occlusion on {}, {} | off {}, {}", mean(self.sums[1]), interval(self.sums[1]), mean(self.sums[0]), interval(self.sums[0])));
        }
    }
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    timer: Option<GpuTimer>,
    bench: Option<Bench>,
    cam: usize,
    start: f64,
    frozen_t: Option<f32>,
    trees: u32,
    frame: u32,
    last_frame: f64,
    interval_ms: f64,
    gpu_ms: f64,
    cpu_ms: f64,
    keys: Rc<RefCell<Vec<String>>>,
    /// `skyocc=rebuild`: start a rebuild of the sky occlusion every frame
    rebuild_sky: bool,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

fn checkbox(id: &str) -> Option<web_sys::HtmlInputElement> {
    web_sys::window()?.document()?.get_element_by_id(id)?.dyn_into().ok()
}

fn set_text(id: &str, text: &str) {
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(id)) {
        el.set_text_content(Some(text));
    }
}

fn thousands(n: impl Into<u64>) -> String {
    let s = n.into().to_string();
    let mut out = String::new();
    for (k, c) in s.chars().enumerate() {
        if k > 0 && (s.len() - k).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

fn hud(st: &State) -> String {
    let occlusion = st.renderer.occlusion_culling();
    let frozen = checkbox("freeze").is_some_and(|c| c.checked());
    let mut text = format!(
        "{} trees x 3 LODs · camera {} (C) · occlusion {} (O) · culling {} (F)\n",
        thousands(st.trees),
        CAMS[st.cam],
        if occlusion { "on" } else { "off" },
        if frozen { "frozen" } else { "live" }
    );
    if let Some(stats) = st.renderer.culling_stats() {
        let c: CullStats = stats.camera();
        text += &format!(
            "camera: {} tested · {} outside their LOD band · {} outside the frustum · {} occluded · {} drawn ({} triangles)\n",
            thousands(c.tested),
            thousands(c.lod_culled),
            thousands(c.frustum_culled),
            thousands(c.occlusion_culled),
            thousands(c.drawn),
            thousands(c.triangles)
        );
    }
    let (w, h) = st.renderer.render_size();
    text += &format!(
        "{w} x {h} rendered · GPU {:.2} ms{} · CPU {:.2} ms in render · {:.1} ms between frames",
        st.gpu_ms,
        if st.timer.is_some() { "" } else { " (no timestamps)" },
        st.cpu_ms,
        st.interval_ms
    );
    text
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let (width, height) = query_param("size")
        .and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?))))
        .unwrap_or((canvas.client_width() as u32, canvas.client_height() as u32));
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;
    if let Some(scale) = query_param("scale").and_then(|v| v.parse().ok()) {
        renderer.set_render_scale(scale);
    }
    renderer.set_culling_stats(true);
    renderer.set_occlusion_culling(query_param("occlusion").as_deref() != Some("0"));
    let sky_occ = match query_param("skyocc").as_deref() {
        Some("1") | Some("rebuild") => SkyOcc::On,
        Some("show") => SkyOcc::Show,
        _ => SkyOcc::Off,
    };
    let rebuild_sky = query_param("skyocc").as_deref() == Some("rebuild");
    if sky_occ != SkyOcc::Off {
        // the ground and the trees' tops (up to 26 m over the hill's 57 m) lie in the volume; the
        // top-down view draws coarser LODs (the last one reaches any distance)
        renderer.enable_sky_occlusion(SkyOcclusionOptions {
            min_height_m: -10.0,
            max_height_m: 90.0,
            volume_size: (128, 32),
            lod_distance_scale: 4.0,
            layer_mask: TREE_LAYER,
            ..Default::default()
        });
    }

    let mut scene = Scene::new();
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    scene.add(SceneNode::Renderable(Renderable::new(SphereGeometry::new(1500.0, 32, 16), sky)));
    // the terrain: an ordinary mesh, and the main occluder
    let sky = renderer.sky_occlusion().map(|s| (s, sky_occ));
    scene.add(SceneNode::Renderable(Renderable::new(terrain(240), surface_material("Terrain", [0.09, 0.1, 0.05], false, sky))));

    // the forest: base xyz and height, then yaw and tint, 32 bytes a tree; everywhere but the
    // valley's meadow and clearings where the cameras stand
    let trees: u32 = query_param("trees").and_then(|v| v.parse().ok()).unwrap_or(60_000);
    let mut data: Vec<f32> = Vec::with_capacity(trees as usize * 8);
    let clearings = [(-120.0, 60.0, 4.0), (-40.0, -130.0, 16.0)];
    let meadow = |x: f32, z: f32| x.abs() < 70.0 && (-60.0..140.0).contains(&z);
    let mut i = 0u32;
    while (data.len() as u32) < trees * 8 {
        i += 1;
        let (x, z) = (-EXTENT + hash(i) * 2.0 * EXTENT, -EXTENT + hash(i ^ 0x5bd1e995) * 2.0 * EXTENT);
        if meadow(x, z) || clearings.iter().any(|&(cx, cz, r)| (x - cx).powi(2) + (z - cz).powi(2) < r * r) {
            continue;
        }
        let h = 14.0 + 12.0 * hash(i.wrapping_mul(3) + 7);
        data.extend_from_slice(&[x, ground(x, z) - 0.3, z, h, hash(i + 11) * std::f32::consts::TAU, hash(i + 23), 0.0, 0.0]);
    }
    let source = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Forest"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        })
    };
    // three LODs by distance, each culled per view on the GPU; with occlusion for the camera
    let lods = [(spruce(48, 6, 6, "Spruce/LOD0"), 0.0, 80.0), (spruce(20, 2, 5, "Spruce/LOD1"), 80.0, 220.0), (spruce(8, 1, 3, "Spruce/LOD2"), 220.0, f32::INFINITY)];
    let spheres = query_param("bounds").as_deref() == Some("sphere");
    for (geometry, near, far) in lods {
        let instances = ComputeBuffer::from_external("Forest", source.clone(), BufferType::Storage).with_vertex_layout(
            32,
            vec![
                InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32x4 },
            ],
        );
        let label = geometry.label.clone();
        let mut r = Renderable::new(InstancedGeometry::new(geometry, trees, vec![instances]), surface_material(&label, [0.05, 0.09, 0.05], true, sky));
        r.layers |= TREE_LAYER;
        // a sphere round the base reaching the top (x height), or a box from the ground to the
        // top, as wide as the lowest cone
        let mut culling = InstanceCulling::new(source.clone(), trees, 32, 0, 1.0).with_radius_scale(12).with_lod_range(near, far).with_occlusion(true);
        if !spheres {
            culling = culling.with_bounds_shift(glam::Vec3::new(0.0, 0.5, 0.0)).with_bounds_box(glam::Vec3::new(0.25, 0.5, 0.25));
        }
        r.instance_culling = Some(culling);
        scene.add(SceneNode::Renderable(r));
    }

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(11.0);
        ToneMapEffect::new(options)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if query_param("taa").as_deref() != Some("0") {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(50.0, 0.3, 3000.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    let cam = CAMS.iter().position(|&c| Some(c) == query_param("cam").as_deref()).unwrap_or(0);
    if query_param("freeze").as_deref() == Some("1") {
        if let Some(c) = checkbox("freeze") {
            c.set_checked(true);
        }
    }
    if let Some(c) = checkbox("occlusion") {
        c.set_checked(renderer.occlusion_culling());
    }
    let timer = GpuTimer::new(renderer.device(), renderer.queue());
    let bench = (query_param("bench").as_deref() == Some("1")).then(|| Bench { start: now_ms(), sums: [(0.0, 0, 0.0, 0); 2], last_frame: now_ms(), report: None });
    log::info!("Kansei — Occlusion culling (WASM) ready: {trees} trees, timestamps {}", timer.is_some());

    let keys = Rc::new(RefCell::new(Vec::new()));
    {
        let keys = keys.clone();
        let on_key = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| keys.borrow_mut().push(e.key().to_lowercase()));
        window.add_event_listener_with_callback("keydown", on_key.as_ref().unchecked_ref())?;
        on_key.forget();
    }

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, timer, bench, cam, start: now_ms(), frozen_t, trees, frame: 0, last_frame: now_ms(), interval_ms: 0.0, gpu_ms: 0.0, cpu_ms: 0.0, keys, rebuild_sky }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut guard = state.borrow_mut();
            let st = &mut *guard;
            // keys and the HUD's checkboxes
            for key in st.keys.borrow_mut().drain(..) {
                let toggle = |id: &str| {
                    if let Some(c) = checkbox(id) {
                        c.set_checked(!c.checked());
                    }
                };
                match key.as_str() {
                    "o" => toggle("occlusion"),
                    "f" => toggle("freeze"),
                    "c" => st.cam = (st.cam + 1) % CAMS.len(),
                    _ => {}
                }
            }
            let occlusion = match st.bench.as_ref().and_then(|b| b.phase(now_ms())) {
                Some((on, _)) => on,
                None => checkbox("occlusion").is_none_or(|c| c.checked()),
            };
            st.renderer.set_occlusion_culling(occlusion);
            st.renderer.set_freeze_culling(checkbox("freeze").is_some_and(|c| c.checked()));

            let t = st.frozen_t.unwrap_or(((now_ms() - st.start) / 1000.0) as f32);
            place_camera(&mut st.camera, CAMS[st.cam], t);

            if let Some(timer) = st.timer.as_mut() {
                timer.begin(st.renderer.device(), st.renderer.queue());
            }
            if st.rebuild_sky {
                if let Some(sky) = st.renderer.sky_occlusion_mut() {
                    sky.refresh();
                }
            }
            let before = now_ms();
            st.renderer.render_with_postprocessing(&mut st.scene, &mut st.camera, &mut st.volume);
            st.cpu_ms += (now_ms() - before - st.cpu_ms) * 0.05;
            if let Some(timer) = st.timer.as_mut() {
                timer.end(st.renderer.device(), st.renderer.queue());
            }

            let now = now_ms();
            st.interval_ms += (now - st.last_frame - st.interval_ms) * 0.05;
            st.last_frame = now;
            let gpu = st.timer.as_ref().map(|t| t.take()).unwrap_or_default();
            for ms in &gpu {
                st.gpu_ms += (ms - st.gpu_ms) * 0.05;
            }
            if let Some(bench) = st.bench.as_mut() {
                let was_done = bench.report.is_some();
                bench.record(&gpu, now);
                if let (false, Some(report)) = (was_done, &bench.report) {
                    log::info!("{report}");
                    set_text("bench", report);
                }
            }
            st.frame += 1;
            if st.frame.is_multiple_of(10) {
                set_text("hud", &hud(st));
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
