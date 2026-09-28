//! Impostors: a lake ringed by a forest of spruces (40 000 by default), mirrored in the water
//! (a planar reflection). The spruces have two mesh LODs and, beyond `far` metres, an octahedral
//! impostor baked at start-up from the nearest LOD (`Renderer::bake_impostor`): a billboard per
//! tree whose material (`impostors::IMPOSTOR_WGSL`) reads the albedo, normal and depth of the
//! three baked views nearest its view direction, shades them as the mesh's material does and
//! writes the depth of the surface it found. The reflection sees the far shore from below: the
//! impostor is baked from the whole sphere of directions.
//!
//! The HUD shows the instances drawn for the camera and the reflection (`Renderer::culling_stats`)
//! and the GPU time (timestamp queries when the adapter has them). Keys: I toggles the impostors
//! (off: the coarser mesh LOD reaches the horizon), C cycles the camera.
//!
//! URL parameters: `cam=shore|low|high|fly`, `impostors=0`, `trees=<n>`, `far=<metres>` (where
//! the impostors start), `depth=1` (the impostors write the depth of the surface they find),
//! `frames=<n>` and `frame=<texels>` (the bake), `t=<seconds>` (the fly
//! path's time, frozen), `size=<w>x<h>`, `bench=1` (alternate the impostors on and off every 3 s,
//! 8 times, and report the mean GPU time and frame interval of each).

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, Sampler, VertexFormat};
use kansei_core::cameras::Camera;
use kansei_core::culling::{CullViewKind, InstanceCulling};
use kansei_core::geometries::{Geometry, InstancedGeometry, PlaneGeometry, SphereGeometry, Vertex};
use kansei_core::impostors::{billboard_geometry, ImpostorOptions, IMPOSTOR_WGSL};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::reflections::{PlanarReflection, PlanarReflectionOptions, PLANAR_REFLECTION_WGSL};
use kansei_core::renderers::{Renderer, RendererConfig};

const WATER_LAYER: u32 = 2;

/// Sunlight, sky and haze, shared by the meshes and the impostors so they match.
const SHADE_WGSL: &str = r#"
fn eye_of(view: mat4x4<f32>) -> vec3<f32> {
    let view3 = mat3x3<f32>(view[0].xyz, view[1].xyz, view[2].xyz);
    return -(transpose(view3) * view[3].xyz);
}

fn shade(albedo: vec3<f32>, n: vec3<f32>, world: vec3<f32>, eye: vec3<f32>) -> vec3<f32> {
    let sun = normalize(vec3<f32>(-0.4, 0.55, -0.7));
    let sky = mix(vec3<f32>(60.0, 55.0, 45.0), vec3<f32>(900.0, 1100.0, 1500.0), n.y * 0.5 + 0.5);
    let lit = albedo * (sky + vec3<f32>(9000.0, 7600.0, 6000.0) * max(dot(n, sun), 0.0));
    let haze = 1.0 - exp(-distance(world, eye) * 0.0015);
    return mix(lit, vec3<f32>(1500.0, 1700.0, 2000.0), haze);
}

// a unit spruce placed by its instance: base xyz and height, then yaw and tint
fn place(local: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3<f32>(c * local.x + s * local.z, local.y, -s * local.x + c * local.z) * inst.w + inst.xyz;
}

fn turn(v: vec3<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn unplace(world: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    return turn((world - inst.xyz) / inst.w, -yaw);
}

struct FOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

fn surface_out(color: vec3<f32>, n: vec3<f32>, albedo: vec3<f32>) -> FOut {
    return FOut(vec4<f32>(color, 1.0), vec4<f32>(0.0), vec4<f32>(normalize(n) * 0.5 + 0.5, 1.0), vec4<f32>(albedo, 1.0));
}
"#;

/// The terrain and the spruce meshes: albedo from the material (trunks brown), lit by `shade`.
const SURFACE_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, params: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, TREE_INPUT };
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) world: vec3<f32>,
    @location(2) albedo: vec3<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    var world = (world_matrix * vec4<f32>(v.position.xyz, 1.0)).xyz;
    var n = v.normal;
    var albedo = surface.base_color.rgb;
    TREE_PLACE
    var out: VOut;
    out.clip = projection_matrix * view_matrix * vec4<f32>(world, 1.0);
    out.normal = n;
    out.world = world;
    out.albedo = albedo;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal);
    return surface_out(shade(in.albedo, n, in.world, eye_of(view_matrix)), n, in.albedo);
}
"#;

/// The impostor: a billboard per spruce, read from the baked atlases, shaded as the meshes.
const IMPOSTOR_MATERIAL_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> impostor: KanseiImpostor;
@group(0) @binding(1) var albedo_atlas: texture_2d<f32>;
@group(0) @binding(2) var normal_depth_atlas: texture_2d<f32>;
@group(0) @binding(3) var atlas_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
    @location(3) inst: vec4<f32>, @location(4) extra: vec4<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) local: vec3<f32>,
    @location(1) eye: vec3<f32>,
    @location(2) inst: vec4<f32>,
    @location(3) extra: vec4<f32>,
};
struct ImpostorOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
    DEPTH_OUTPUT
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    // the camera in the tree's own space, the billboard there, then placed as the tree
    let eye = unplace(eye_of(view_matrix), v.inst, v.extra.x);
    let local = kansei_impostor_corner(impostor, v.position.xy, eye);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * vec4<f32>(place(local, v.inst, v.extra.x), 1.0);
    out.local = local;
    out.eye = eye;
    out.inst = v.inst;
    out.extra = v.extra;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> ImpostorOut {
    let s = kansei_impostor_sample(impostor, albedo_atlas, normal_depth_atlas, atlas_sampler, in.local, in.eye);
    if (s.alpha < 0.5) {
        discard;
    }
    let world = place(s.position, in.inst, in.extra.x);
    let n = turn(s.normal, in.extra.x);
    // baked with the tint neutral: this tree's own
    let albedo = s.albedo * (0.7 + 0.6 * in.extra.y);
    let base = surface_out(shade(albedo, n, world, eye_of(view_matrix)), n, albedo);
    let clip = projection_matrix * view_matrix * vec4<f32>(world, 1.0);
    return ImpostorOut(base.color, base.emissive, base.normal, base.albedo DEPTH_VALUE);
}
"#;

const SKY_WGSL: &str = r#"
struct Sky { params: vec4<f32> };
@group(0) @binding(0) var<uniform> sky: Sky;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * vec4<f32>(position.xyz, 1.0);
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let d = normalize(in.dir);
    return vec4<f32>(mix(vec3<f32>(1500.0, 1700.0, 2000.0), vec3<f32>(500.0, 800.0, 1500.0), saturate(d.y * 3.0)), 1.0);
}
"#;

/// The lake: a Fresnel mix of a dark body colour and the reflection.
const WATER_WGSL: &str = r#"
@group(0) @binding(0) var reflection_tex: texture_2d<f32>;
@group(0) @binding(1) var reflection_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) pixel: vec4<f32>, @location(0) world: vec3<f32>, @location(1) clip: vec4<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.pixel = out.clip;
    out.world = world.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let v = normalize(-(transpose(view3) * view_matrix[3].xyz) - in.world);
    let r = kansei_planar_reflection(reflection_tex, reflection_sampler, kansei_screen_uv(in.clip), vec2<f32>(0.0), 0.02);
    let fresnel = 0.02 + 0.98 * pow(1.0 - saturate(v.y), 5.0);
    return vec4<f32>(mix(vec3<f32>(20.0, 30.0, 30.0), r.rgb, max(fresnel, 0.35)), 1.0);
}
"#;

fn surface_material(label: &str, base: [f32; 3], tree: bool) -> Material {
    let (input, place) = if tree {
        (
            "@location(3) inst: vec4<f32>, @location(4) extra: vec4<f32>,",
            // trunks brown; each tree tinted
            "albedo = select(albedo, vec3<f32>(0.09, 0.06, 0.04), length(v.position.xz) < 0.04); \
             albedo *= 0.7 + 0.6 * v.extra.y; \
             world = place(v.position.xyz, v.inst, v.extra.x); \
             n = turn(v.normal, v.extra.x);",
        )
    } else {
        ("", "")
    };
    let shader = SURFACE_WGSL.replace("TREE_INPUT", input).replace("TREE_PLACE", place);
    let mut m = Material::new(
        label,
        &format!("{SHADE_WGSL}\n{shader}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    m.set_uniform_bindable(0, label, &[base[0], base[1], base[2], 1.0, 0.0, 0.0, 0.0, 0.0f32]);
    m
}

/// Deterministic 0..1 hash.
fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// Half the side of the square the forest covers, in metres.
const EXTENT: f32 = 700.0;
/// The lake's radius (roughly: its shore follows the terrain) and level.
const LAKE: f32 = 170.0;
const LAKE_LEVEL: f32 = 0.0;

/// Terrain height: a basin holding the lake, rising to hills round it.
fn ground(x: f32, z: f32) -> f32 {
    let r = (x * x + z * z).sqrt();
    let basin = -6.0 + 16.0 * ((r - LAKE + 40.0) / 120.0).clamp(0.0, 1.0).powi(2);
    let hills = 30.0 * ((r - 350.0) / 300.0).clamp(0.0, 1.0) * (0.6 + 0.4 * (x * 0.004).sin() * (z * 0.005).cos());
    let roll = 2.5 * (x * 0.031).sin() * (z * 0.027).cos() + 1.5 * (x * 0.083 + z * 0.061).sin();
    basin + hills + roll * ((r - LAKE) / 60.0).clamp(0.0, 1.0)
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

const CAMS: [&str; 4] = ["shore", "low", "high", "fly"];

fn place_camera(camera: &mut Camera, cam: &str, t: f32) {
    let (from, to) = match cam {
        // on the south shore, looking across the lake at the far treeline and its reflection
        "shore" => (glam::Vec3::new(0.0, 3.0, LAKE + 10.0), glam::Vec3::new(0.0, 4.0, -LAKE)),
        // just above the water: most of the view is the mirrored forest
        "low" => (glam::Vec3::new(-60.0, 0.6, 60.0), glam::Vec3::new(40.0, 2.0, -LAKE)),
        "high" => (glam::Vec3::new(-300.0, 160.0, 320.0), glam::Vec3::new(0.0, 0.0, -80.0)),
        // round the lake, over the water
        _ => {
            let a = t * 0.04;
            let r = LAKE * 0.7;
            (glam::Vec3::new(r * a.cos(), 4.0, r * a.sin()), glam::Vec3::new(r * (a + 0.9).cos(), 3.0, r * (a + 0.9).sin()))
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

    fn take(&self) -> Vec<f64> {
        std::mem::take(&mut *self.results.lock().unwrap())
    }
}

/// `bench=1`: alternate the impostors on and off, averaging the GPU time and the frame interval
/// of each (after a warm-up, and ignoring the start of each phase).
struct Bench {
    start: f64,
    /// (GPU ms, GPU samples, frame intervals ms, frames) with the impostors off and on
    sums: [(f64, u32, f64, u32); 2],
    last_frame: f64,
    report: Option<String>,
}

const BENCH_WARMUP_MS: f64 = 3000.0;
const BENCH_PHASE_MS: f64 = 3000.0;
const BENCH_SETTLE_MS: f64 = 500.0;
const BENCH_PHASES: u32 = 8;

impl Bench {
    /// (impostors on, measuring) at `now`, or None when done.
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
            self.report = Some(format!("bench: impostors on {}, {} | off {}, {}", mean(self.sums[1]), interval(self.sums[1]), mean(self.sums[0]), interval(self.sums[0])));
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
    /// scene indices of the coarse mesh LOD and the impostors, and where the impostors start
    lod1: usize,
    impostor: usize,
    far: f32,
    bake_ms: f64,
    frame: u32,
    last_frame: f64,
    interval_ms: f64,
    gpu_ms: f64,
    keys: Rc<RefCell<Vec<String>>>,
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

fn thousands(n: u32) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (k, c) in s.chars().enumerate() {
        if k > 0 && (s.len() - k).is_multiple_of(3) {
            out.push(' ');
        }
        out.push(c);
    }
    out
}

/// Switch the far band between the impostors and the coarse mesh LOD.
fn set_impostors(st: &mut State, on: bool) {
    let far = if on { st.far } else { f32::INFINITY };
    if let Some(culling) = st.scene.get_renderable_mut(st.lod1).and_then(|r| r.instance_culling.as_mut()) {
        culling.lod_range.1 = far;
    }
    if let Some(r) = st.scene.get_renderable_mut(st.impostor) {
        r.visible = on;
    }
}

fn hud(st: &State, impostors: bool) -> String {
    let mut text = format!(
        "{} spruces · impostors {} beyond {} m (I) · camera {} (C) · baked in {:.0} ms\n",
        thousands(st.trees),
        if impostors { "on" } else { "off" },
        st.far,
        CAMS[st.cam],
        st.bake_ms
    );
    if let Some(stats) = st.renderer.culling_stats() {
        let reflection = stats.view(CullViewKind::Reflection(0)).map_or("-".into(), |s| thousands(s.drawn));
        text += &format!("drawn: {} for the camera, {} for the reflection (meshes and impostors)\n", thousands(stats.camera().drawn), reflection);
    }
    let (w, h) = st.renderer.render_size();
    text += &format!("{w} x {h} · GPU {:.2} ms{} · {:.1} ms between frames", st.gpu_ms, if st.timer.is_some() { "" } else { " (no timestamps)" }, st.interval_ms);
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
    renderer.set_culling_stats(true);

    let mut scene = Scene::new();
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    scene.add(SceneNode::Renderable(Renderable::new(SphereGeometry::new(2500.0, 32, 16), sky)));
    scene.add(SceneNode::Renderable(Renderable::new(terrain(280), surface_material("Terrain", [0.09, 0.1, 0.05], false))));

    // the forest round the lake: base xyz and height, then yaw and tint, 32 bytes a tree
    let trees: u32 = query_param("trees").and_then(|v| v.parse().ok()).unwrap_or(40_000);
    let mut data: Vec<f32> = Vec::with_capacity(trees as usize * 8);
    let mut i = 0u32;
    while (data.len() as u32) < trees * 8 {
        i += 1;
        let (x, z) = (-EXTENT + hash(i) * 2.0 * EXTENT, -EXTENT + hash(i ^ 0x5bd1e995) * 2.0 * EXTENT);
        if ground(x, z) < LAKE_LEVEL + 1.0 {
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
    let instances = || {
        ComputeBuffer::from_external("Forest", source.clone(), BufferType::Storage).with_vertex_layout(
            32,
            vec![
                InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32x4 },
            ],
        )
    };
    // a box from the ground to the top of a tree (x its height), as wide as its lowest cone
    let culling = |near: f32, far: f32| {
        InstanceCulling::new(source.clone(), trees, 32, 0, 1.0)
            .with_radius_scale(12)
            .with_lod_range(near, far)
            .with_bounds_shift(glam::Vec3::new(0.0, 0.5, 0.0))
            .with_bounds_box(glam::Vec3::new(0.25, 0.5, 0.25))
    };
    let far: f32 = query_param("far").and_then(|v| v.parse().ok()).unwrap_or(120.0);
    let mut lod_indices = Vec::new();
    for (geometry, near, band_far) in [(spruce(48, 6, 6, "Spruce/LOD0"), 0.0, 50.0), (spruce(16, 2, 5, "Spruce/LOD1"), 50.0, far)] {
        let label = geometry.label.clone();
        let mut r = Renderable::new(InstancedGeometry::new(geometry, trees, vec![instances()]), surface_material(&label, [0.05, 0.09, 0.05], true));
        r.instance_culling = Some(culling(near, band_far));
        lod_indices.push(scene.add(SceneNode::Renderable(r)));
    }

    // the impostor, baked from the nearest LOD with a neutral instance (at the origin, height
    // 1, unturned, the tint that leaves the albedo as it is)
    let frames: u32 = query_param("frames").and_then(|v| v.parse().ok()).unwrap_or(12);
    let frame_size: u32 = query_param("frame").and_then(|v| v.parse().ok()).unwrap_or(128);
    let before = now_ms();
    let impostor = renderer.bake_impostor(
        &mut scene,
        &lod_indices[..1],
        &ImpostorOptions { frames, frame_size, instance: bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0]).to_vec(), ..Default::default() },
    );
    let bake_ms = now_ms() - before;
    // depth=1: the impostors write the depth of the surface they find (it costs: see IMPOSTOR_WGSL)
    let impostor_shader = if query_param("depth").as_deref() == Some("1") {
        IMPOSTOR_MATERIAL_WGSL.replace("DEPTH_OUTPUT", "@builtin(frag_depth) depth: f32,").replace("DEPTH_VALUE", ", clip.z / clip.w")
    } else {
        IMPOSTOR_MATERIAL_WGSL.replace("DEPTH_OUTPUT", "").replace("DEPTH_VALUE", "")
    };
    let mut material = Material::new(
        "Spruce/Impostor",
        &format!("{IMPOSTOR_WGSL}\n{SHADE_WGSL}\n{impostor_shader}"),
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::texture_2d(2, ShaderStages::FRAGMENT),
            Binding::sampler(3, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { mrt_output_count: Some(4), cull_mode: CullMode::None, ..Default::default() },
    );
    material.set_uniform_bindable(0, "Impostor", &[impostor.params()]);
    material.set_bindable(1, impostor.albedo_texture());
    material.set_bindable(2, impostor.normal_depth_texture());
    material.set_bindable(3, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    let mut billboards = Renderable::new(InstancedGeometry::new(billboard_geometry("Spruce/Impostor"), trees, vec![instances()]), material);
    billboards.instance_culling = Some(culling(far, f32::INFINITY));
    let impostor_index = scene.add(SceneNode::Renderable(billboards));

    // the lake, mirroring everything but itself
    let reflection = PlanarReflection::new(
        &renderer,
        Vec3::new(0.0, LAKE_LEVEL, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        PlanarReflectionOptions { width: width / 2, height: height / 2, layer_mask: !WATER_LAYER, ..Default::default() },
    );
    let mut water_material = Material::new(
        "Water",
        &format!("{PLANAR_REFLECTION_WGSL}\n{WATER_WGSL}"),
        vec![Binding::texture_2d(0, ShaderStages::FRAGMENT), Binding::sampler(1, ShaderStages::FRAGMENT)],
        MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
    );
    water_material.set_bindable(0, reflection.material_texture());
    water_material.set_bindable(1, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    renderer.add_planar_reflection(reflection);
    let mut lake = Renderable::new(PlaneGeometry::new(2.0 * LAKE + 200.0, 2.0 * LAKE + 200.0), water_material);
    lake.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    lake.object.set_position(0.0, LAKE_LEVEL, 0.0);
    lake.layers = WATER_LAYER;
    lake.cast_shadow = false;
    scene.add(SceneNode::Renderable(lake));

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

    let mut camera = Camera::new(45.0, 0.3, 4000.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    let cam = CAMS.iter().position(|&c| Some(c) == query_param("cam").as_deref()).unwrap_or(0);
    if let Some(c) = checkbox("impostors") {
        c.set_checked(query_param("impostors").as_deref() != Some("0"));
    }
    let timer = GpuTimer::new(renderer.device(), renderer.queue());
    let bench = (query_param("bench").as_deref() == Some("1")).then(|| Bench { start: now_ms(), sums: [(0.0, 0, 0.0, 0); 2], last_frame: now_ms(), report: None });
    log::info!("Kansei — Impostors (WASM) ready: {trees} trees, impostor {frames}x{frames} frames of {frame_size} texels baked in {bake_ms:.0} ms");

    let keys = Rc::new(RefCell::new(Vec::new()));
    {
        let keys = keys.clone();
        let on_key = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| keys.borrow_mut().push(e.key().to_lowercase()));
        window.add_event_listener_with_callback("keydown", on_key.as_ref().unchecked_ref())?;
        on_key.forget();
    }

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        volume,
        timer,
        bench,
        cam,
        start: now_ms(),
        frozen_t,
        trees,
        lod1: lod_indices[1],
        impostor: impostor_index,
        far,
        bake_ms,
        frame: 0,
        last_frame: now_ms(),
        interval_ms: 0.0,
        gpu_ms: 0.0,
        keys,
    }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut guard = state.borrow_mut();
            let st = &mut *guard;
            for key in st.keys.borrow_mut().drain(..) {
                match key.as_str() {
                    "i" => {
                        if let Some(c) = checkbox("impostors") {
                            c.set_checked(!c.checked());
                        }
                    }
                    "c" => st.cam = (st.cam + 1) % CAMS.len(),
                    _ => {}
                }
            }
            let impostors = match st.bench.as_ref().and_then(|b| b.phase(now_ms())) {
                Some((on, _)) => on,
                None => checkbox("impostors").is_none_or(|c| c.checked()),
            };
            set_impostors(st, impostors);

            let t = st.frozen_t.unwrap_or(((now_ms() - st.start) / 1000.0) as f32);
            place_camera(&mut st.camera, CAMS[st.cam], t);

            if let Some(timer) = st.timer.as_mut() {
                timer.begin(st.renderer.device(), st.renderer.queue());
            }
            st.renderer.render_with_postprocessing(&mut st.scene, &mut st.camera, &mut st.volume);
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
                set_text("hud", &hud(st, impostors));
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
