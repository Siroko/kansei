//! Cluster LOD: a field of rocks (24 x 24 by default, 81 920 triangles each), drawn three ways
//! with one material. `clusters`: one renderable with cluster LOD, each rock's cut picked per
//! cluster every frame. `lods`: four discrete LODs cut from the same cluster graph, each at the
//! error budget from its band's near edge, switched per rock by distance. `full`: the mesh as is.
//! The HUD shows the GPU time of each frame (timestamp queries when the adapter has them) and
//! the frame interval.
//!
//! URL parameters: `mode=clusters|lods|full`, `n=<rocks per side>`, `sub=<subdivisions>`,
//! `tau=<pixels>` (the error budget), `size=<w>x<h>`, `t=<seconds>` (the camera's time, frozen),
//! `profile=1` (log each pass's GPU time every 2 s), `bench=lods|full` (alternate clusters and
//! that mode every 3 s, 8 times, with the camera still through each pair, and report the mean GPU
//! time and frame interval of each).

use std::collections::HashMap;
use std::sync::Arc;
use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform, LodView};
use kansei_core::culling::InstanceCulling;
use kansei_core::geometries::{Geometry, InstancedGeometry, PlaneGeometry, Vertex};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FrameTimer;
use kansei_core::postprocessing::{effects::{exposure_from_ev100, ToneMapEffect, ToneMapOptions}, PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{flag, now, param, param_or, Canvas};

/// Rocks placed by records of position + scale, then yaw (8 floats), lit by a low sun. Prefixed
/// with GBUFFER_OUT_WGSL.
const ROCK_WGSL: &str = r#"
struct Surface { color: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) yaw: f32 };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world: vec3<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let world = world_matrix * vec4<f32>(turn(v.position.xyz * v.place.w, v.yaw) + v.place.xyz, 1.0);
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.yaw), 0.0)).xyz;
    out.world = world.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let n = normalize(in.normal);
    let sun = normalize(vec3<f32>(0.4, 0.6, 0.3));
    let light = vec3<f32>(1.0, 0.95, 0.85) * max(dot(n, sun), 0.0) * 3.0 + vec3<f32>(0.25, 0.3, 0.4) * (0.6 + 0.4 * n.y);
    let color = surface.color.rgb * light;
    return kansei_gbuffer_out(color, vec3<f32>(0.0), n, surface.color.rgb);
}
"#;

/// Distances (metres) where the discrete LODs start.
const BANDS: [f32; 4] = [0.0, 8.0, 24.0, 72.0];
const SPACING: f32 = 4.0;
const MODES: [&str; 3] = ["clusters", "lods", "full"];

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// A noisy icosphere of `subdivisions`, about a metre across.
fn rock(subdivisions: u32) -> Geometry {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut p: Vec<glam::Vec3> = [[-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0], [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t], [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0]]
        .iter()
        .map(|v| glam::Vec3::from(*v).normalize())
        .collect();
    let mut f: Vec<[u32; 3]> = vec![[0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]];
    for _ in 0..subdivisions {
        let mut mid = HashMap::new();
        let mut next = Vec::with_capacity(f.len() * 4);
        for [a, b, c] in f {
            let mut m = |x: u32, y: u32| {
                *mid.entry((x.min(y), x.max(y))).or_insert_with(|| {
                    p.push(((p[x as usize] + p[y as usize]) * 0.5).normalize());
                    p.len() as u32 - 1
                })
            };
            let (ab, bc, ca) = (m(a, b), m(b, c), m(c, a));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        f = next;
    }
    let height = |v: glam::Vec3| 1.0 + 0.12 * (5.0 * v.x).sin() * (4.0 * v.y).cos() + 0.06 * (13.0 * v.z + 2.0 * v.x).sin() + 0.02 * (37.0 * v.y).sin() * (31.0 * v.x).cos();
    let vertices: Vec<Vertex> = p
        .iter()
        .map(|&v| {
            // the normal of the displaced surface, from two nearby points on it
            let (a, b) = (v.any_orthonormal_vector(), v.cross(v.any_orthonormal_vector()));
            let at = |d: glam::Vec3| (d.normalize()) * height(d.normalize());
            let n = (at(v + a * 1e-3) - at(v - a * 1e-3)).cross(at(v + b * 1e-3) - at(v - b * 1e-3)).normalize();
            let q = v * height(v);
            Vertex { position: [q.x, q.y * 0.7, q.z, 1.0], normal: (n * glam::Vec3::new(0.7, 1.0, 0.7)).normalize().to_array(), uv: [v.x * 0.5 + 0.5, v.y * 0.5 + 0.5] }
        })
        .collect();
    Geometry::new("Rock", vertices, f.into_iter().flatten().collect())
}

/// The cut of `mesh` seen from `distance` away at `tau` pixels (`ppr` pixels per radian), as a
/// mesh: a discrete LOD for a band starting there (for the largest rock, `scale`).
fn lod_mesh(mesh: &ClusterMesh, distance: f32, scale: f32, ppr: f32, tau: f32) -> Geometry {
    let view = LodView { eye: glam::Vec3::new(0.0, 0.0, distance / scale), pixels_per_radian: ppr, near: 0.1 / scale, threshold: tau, orthographic: false };
    let indices: Vec<u32> = mesh.select(&view).into_iter().flat_map(|c| mesh.triangles(c).flatten().collect::<Vec<_>>()).collect();
    Geometry::new("Rock/LOD", mesh.vertices.clone(), indices)
}

fn material() -> Material {
    let mut m = Material::new("Rock", &format!("{GBUFFER_OUT_WGSL}\n{ROCK_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Rock", &[[0.42f32, 0.4, 0.37, 1.0]]);
    m
}

/// `bench=<mode>`: alternate clusters and `mode`, averaging each one's GPU time and frame
/// interval (after a warm-up, and ignoring the start of each phase).
struct Bench {
    other: usize,
    start: f64,
    /// (GPU ms, GPU samples, frame intervals ms, frames) of clusters and the other mode
    sums: [(f64, u32, f64, u32); 2],
    last_frame: f64,
    report: Option<String>,
}

const BENCH_WARMUP_MS: f64 = 3000.0;
const BENCH_PHASE_MS: f64 = 3000.0;
const BENCH_SETTLE_MS: f64 = 500.0;
const BENCH_PHASES: u32 = 8;

impl Bench {
    /// (mode, measuring) at `now`, or None when done.
    fn phase(&self, now: f64) -> Option<(usize, bool)> {
        let t = now - self.start - BENCH_WARMUP_MS;
        if t < 0.0 {
            return Some((0, false));
        }
        let phase = (t / BENCH_PHASE_MS) as u32;
        (phase < BENCH_PHASES).then_some((if phase.is_multiple_of(2) { 0 } else { self.other }, t % BENCH_PHASE_MS >= BENCH_SETTLE_MS))
    }

    /// The camera's time at `now`: still through each pair of phases (both modes see the same
    /// view), a different stretch of the loop for each pair.
    fn view_time(&self, now: f64) -> f32 {
        let pair = ((now - self.start - BENCH_WARMUP_MS).max(0.0) / BENCH_PHASE_MS) as u32 / 2;
        5.0 + 9.0 * pair as f32
    }

    fn record(&mut self, gpu: &[f64], now: f64) {
        if let Some((mode, true)) = self.phase(now) {
            let sum = &mut self.sums[(mode != 0) as usize];
            sum.0 += gpu.iter().sum::<f64>();
            sum.1 += gpu.len() as u32;
            sum.2 += now - self.last_frame;
            sum.3 += 1;
        }
        self.last_frame = now;
        if self.phase(now).is_none() && self.report.is_none() {
            let mean = |(sum, n, _, _): (f64, u32, f64, u32)| if n > 0 { format!("{:.2} ms GPU ({n} samples)", sum / n as f64) } else { "no GPU timestamps".into() };
            let interval = |(_, _, sum, n): (f64, u32, f64, u32)| format!("{:.2} ms/frame ({n} frames)", sum / n.max(1) as f64);
            self.report = Some(format!("bench: clusters {}, {} | {} {}, {}", mean(self.sums[0]), interval(self.sums[0]), MODES[self.other], mean(self.sums[1]), interval(self.sums[1])));
        }
    }
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    timer: FrameTimer,
    bench: Option<Bench>,
    /// scene indices of each mode's renderables
    modes: [Vec<usize>; 3],
    mode: usize,
    rocks: u32,
    triangles: usize,
    build_ms: f64,
    frozen_t: Option<f32>,
    profile: bool,
    frame: u32,
    interval_ms: f64,
    gpu_ms: f64,
}

fn set_text(id: &str, text: &str) {
    if let Some(el) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id(id)) {
        el.set_text_content(Some(text));
    }
}

fn set_mode(st: &mut State, mode: usize) {
    for (m, indices) in st.modes.iter().enumerate() {
        for &i in indices {
            if let Some(r) = st.scene.get_renderable_mut(i) {
                r.visible = m == mode;
            }
        }
    }
    st.mode = mode;
}

/// A loop through the field, low over the rocks, looking ahead and down.
fn place_camera(camera: &mut Camera, extent: f32, t: f32) {
    let a = t * 0.08;
    let r = extent * 0.35;
    let (x, z) = (r * a.cos(), r * a.sin());
    camera.set_position(x, 2.2 + 0.8 * (t * 0.3).sin(), z);
    camera.look_at(&Vec3::new(x - 12.0 * a.sin(), 0.0, z + 12.0 * a.cos()));
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let mut canvas = Canvas::find(canvas_id)?;
    if let Some((w, h)) = param("size").and_then(|s| s.split_once('x').and_then(|(w, h)| Some((w.parse().ok()?, h.parse().ok()?)))) {
        canvas = canvas.with_size(w, h);
    }
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.45, 0.55, 0.7, 1.0), ..Default::default() }).await;
    let tau: f32 = param_or("tau", 1.0);
    renderer.set_cluster_error_threshold(tau);
    let profile = flag("profile", false);
    renderer.set_profiling(profile);

    // the rocks and their graph
    let n: u32 = param_or("n", 24);
    let sub: u32 = param_or("sub", 6);
    let geometry = rock(sub);
    let triangles = geometry.indices.len() / 3;
    let before = now();
    let mesh = Arc::new(ClusterMesh::build(&geometry, &ClusterOptions::default()));
    let build_ms = (now() - before) * 1000.0;

    // the field: position + scale, yaw (8 floats a rock)
    let extent = n as f32 * SPACING;
    let mut data: Vec<f32> = Vec::with_capacity((n * n * 8) as usize);
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let x = -extent / 2.0 + (i as f32 + 0.2 + 0.6 * hash(k)) * SPACING;
        let z = -extent / 2.0 + (j as f32 + 0.2 + 0.6 * hash(k + 7919)) * SPACING;
        data.extend_from_slice(&[x, 0.2, z, 0.7 + 0.6 * hash(k + 104729), hash(k + 3) * std::f32::consts::TAU, 0.0, 0.0, 0.0]);
    }
    let source = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Rocks"), contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE })
    };
    let rocks = n * n;
    let instances = || {
        ComputeBuffer::from_external("Rocks", source.clone(), BufferType::Storage).with_vertex_layout(
            32,
            vec![InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 }, InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32 }],
        )
    };
    let culling = |near: f32, far: f32| InstanceCulling::new(source.clone(), rocks, 32, 0, 1.2).with_radius_scale(12).with_lod_range(near, far);

    let mut scene = Scene::new();
    // the ground: lit by an even sky alone (the rocks' sun is their own)
    let ground_material = Material::standard_lit("Ground", &StandardLitOptions { base_color: [0.2, 0.22, 0.16], roughness: 0.9, sky_up: [1.6; 3], sky_down: [1.6; 3], ..Default::default() });
    let mut ground = Renderable::new(PlaneGeometry::new(extent * 2.0, extent * 2.0), ground_material);
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    let mut modes: [Vec<usize>; 3] = Default::default();
    // clusters
    let mut r = Renderable::new(InstancedGeometry::new(Geometry::new("Rock", geometry.vertices.clone(), geometry.indices.clone()), rocks, vec![instances()]), material());
    r.instance_culling = Some(culling(0.0, f32::INFINITY));
    r.clusters = Some(ClusterLod::new(mesh.clone()).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None }));
    modes[0].push(scene.add(SceneNode::Renderable(r)));
    // discrete LODs cut from the graph at the same budget, for the largest rock (1.3), at the
    // starting height (they are not re-cut when the canvas resizes)
    let ppr = canvas.size().1 as f32 / (2.0 * (45f32.to_radians() / 2.0).tan());
    for (k, &near) in BANDS.iter().enumerate() {
        let far = BANDS.get(k + 1).copied().unwrap_or(f32::INFINITY);
        // cut from a rock's reach nearer than the band's start: within the budget from any side
        let lod = lod_mesh(&mesh, (near - 1.6).max(0.5), 1.3, ppr, tau);
        let mut r = Renderable::new(InstancedGeometry::new(lod, rocks, vec![instances()]), material());
        r.instance_culling = Some(culling(near, far));
        modes[1].push(scene.add(SceneNode::Renderable(r)));
    }
    // the full mesh
    let mut r = Renderable::new(InstancedGeometry::new(geometry, rocks, vec![instances()]), material());
    r.instance_culling = Some(culling(0.0, f32::INFINITY));
    modes[2].push(scene.add(SceneNode::Renderable(r)));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(1.0);
        ToneMapEffect::new(options)
    };
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(tonemap)];
    let volume = PostProcessingVolume::new(&renderer, effects);
    let mut camera = Camera::new(45.0, 0.1, 1000.0, canvas.aspect());
    camera.update_projection_matrix();

    let mode = MODES.iter().position(|&m| Some(m) == param("mode").as_deref()).unwrap_or(0);
    let bench = param("bench").and_then(|b| MODES.iter().position(|&m| m == b).filter(|&m| m != 0)).map(|other| Bench { other, start: now() * 1000.0, sums: [(0.0, 0, 0.0, 0); 2], last_frame: now() * 1000.0, report: None });
    let timer = FrameTimer::new(renderer.device(), renderer.queue());
    log::info!("Kansei — Cluster LOD (WASM) ready: {rocks} rocks of {triangles} triangles, {} clusters over {} levels built in {build_ms:.0} ms", mesh.clusters.len(), mesh.levels().len());

    let mut state = State { renderer, scene, camera, volume, timer, bench, modes, mode, rocks, triangles, build_ms, frozen_t: param("t").and_then(|v| v.parse().ok()), profile, frame: 0, interval_ms: 0.0, gpu_ms: 0.0 };
    set_mode(&mut state, mode);
    kansei_wasm::run(&canvas, move |frame| {
        let st = &mut state;
        frame.resize(&mut st.renderer, &mut st.camera);
        // the bench's clock, in milliseconds
        let now_ms = now() * 1000.0;
        if let Some((mode, _)) = st.bench.as_ref().and_then(|b| b.phase(now_ms)) {
            if mode != st.mode {
                set_mode(st, mode);
            }
        }
        let t = match (&st.bench, st.frozen_t) {
            (Some(bench), _) => bench.view_time(now_ms),
            (None, Some(t)) => t,
            (None, None) => frame.time as f32,
        };
        place_camera(&mut st.camera, extent, t);
        st.timer.begin();
        st.renderer.render_with_postprocessing(&mut st.scene, &mut st.camera, &mut st.volume);
        st.timer.end();

        let now_ms = now() * 1000.0;
        st.interval_ms += (frame.dt as f64 * 1000.0 - st.interval_ms) * 0.05;
        let gpu = st.timer.take();
        for ms in &gpu {
            st.gpu_ms += (ms - st.gpu_ms) * 0.05;
        }
        if let Some(bench) = st.bench.as_mut() {
            let was_done = bench.report.is_some();
            bench.record(&gpu, now_ms);
            if let (false, Some(report)) = (was_done, &bench.report) {
                log::info!("{report}");
                set_text("bench", report);
            }
        }
        st.frame += 1;
        if st.profile && st.frame.is_multiple_of(240) {
            log::info!("{}: {}", MODES[st.mode], st.renderer.take_profile().report());
        }
        if st.frame.is_multiple_of(10) {
            let (w, h) = st.renderer.render_size();
            set_text(
                "hud",
                &format!(
                    "{} · {} rocks of {} triangles · graph built in {:.0} ms · budget {} px\n{w} x {h} · GPU {:.2} ms · {:.1} ms between frames",
                    MODES[st.mode], st.rocks, st.triangles, st.build_ms, st.renderer.cluster_error_threshold(), st.gpu_ms, st.interval_ms
                ),
            );
        }
    });
    Ok(())
}
