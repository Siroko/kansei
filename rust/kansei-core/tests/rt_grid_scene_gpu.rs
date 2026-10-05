//! The renderer's ray tracing grid gathers its renderables on the GPU, on a real GPU: a single
//! mesh, instances culled for the grid's box (with crossfades, so their compacted records carry
//! a fade), and card crowns with cluster LOD by their cut at a cell of error. The grid holds
//! exactly the triangles the CPU places (a box away from the grid, instances out of it and a LOD
//! whose band leaves it out add none), rebuilds when the box moves and only then, and every
//! frame while a dynamic renderable is in it. Skipped (passes) when no adapter is available.

use glam::Vec3 as GVec3;
use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform, LodView};
use kansei_core::culling::InstanceCulling;
use kansei_core::geometries::{BoxGeometry, Geometry, InstancedGeometry, Vertex};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtPlacement, RtSurface, SceneRtGridOptions};

const W: u32 = 96;
const H: u32 = 64;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// A surface writing its albedo and normal; `placed`: placed by records of position + scale,
/// then yaw, and the crossfade's fade (36 bytes as culled).
fn material(placed: bool) -> Material {
    let vin = if placed {
        "struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) extra: vec4<f32>, @location(5) fade: f32 };"
    } else {
        "struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32> };"
    };
    let local = if placed { "turn(v.position.xyz * v.place.w, v.extra.x) + v.place.xyz" } else { "v.position.xyz" };
    let code = format!(
        r#"
struct Surface {{ albedo: vec4<f32> }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
{vin}
struct VOut {{ @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> }};
struct FOut {{ @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> }};
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {{
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}}
@vertex
fn vertex_main(v: VIn) -> VOut {{
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>({local}, 1.0);
    out.normal = (world_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}}
@fragment
fn fragment_main(in: VOut) -> FOut {{
    return FOut(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0), vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0), surface.albedo);
}}
"#
    );
    let mut m = Material::new("Surface", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), cull_mode: CullMode::None, ..Default::default() });
    m.set_uniform_bindable(0, "Surface", &[0.5f32, 0.5, 0.5, 1.0]);
    m
}

const PLACEMENT: InstanceTransform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None };

/// `mesh` placed on each of `records` (position xyz + scale, yaw), culled per view with
/// crossfades, in the grid with the matching placement.
fn placed(renderer: &Renderer, mesh: Geometry, records: &[[f32; 5]]) -> Renderable {
    use wgpu::util::DeviceExt;
    let data: Vec<f32> = records.iter().flat_map(|p| [p[0], p[1], p[2], p[3], p[4], 0.0, 0.0, 0.0]).collect();
    let source = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE });
    let culled = ComputeBuffer::from_external("Placed", source.clone(), BufferType::Storage).with_vertex_layout(
        36,
        vec![
            InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: 5, offset: 32, format: VertexFormat::Float32 },
        ],
    );
    let count = records.len() as u32;
    let mut r = Renderable::new(InstancedGeometry::new(mesh, count, vec![culled]), material(true)).with_rt(RtSurface::new([0.5; 3])).with_rt_placement(RtPlacement::Instance(PLACEMENT));
    r.instance_culling = Some(InstanceCulling::new(source, count, 32, 0, 1.0).with_radius_scale(12).with_crossfade(2.0));
    r
}

/// Where a record puts mesh point `p` (the materials' `turn`).
fn place(record: &[f32; 5], p: GVec3) -> GVec3 {
    let a = record[4];
    let q = p * record[3];
    GVec3::new(a.cos() * q.x + a.sin() * q.z, q.y, -a.sin() * q.x + a.cos() * q.z) + GVec3::new(record[0], record[1], record[2])
}

/// A crown of `count` cards (0.4 m quads) over a cone 4 m tall, as one mesh (cluster LOD's cards).
fn crown(count: u32) -> Geometry {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let hash = |i: u32| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    for k in 0..count {
        let h = hash(k * 3);
        let a = hash(k * 3 + 1) * std::f32::consts::TAU;
        let r = 1.2 * (1.0 - h) * hash(k * 3 + 2).sqrt();
        let c = GVec3::new(r * a.cos(), 0.3 + 3.6 * h, r * a.sin());
        let u = GVec3::new(-a.sin(), 0.0, a.cos()) * 0.2;
        let v = (GVec3::new(a.cos(), -0.3, a.sin())).normalize() * 0.2;
        let n = u.cross(v).normalize();
        let base = vertices.len() as u32;
        for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            let p = c + u * s + v * t;
            vertices.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [(s + 1.0) * 0.5, (t + 1.0) * 0.5] });
        }
        indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
    Geometry::new("Crown", vertices, indices)
}

fn camera_at(eye: [f32; 3], target: [f32; 3]) -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 200.0, W as f32 / H as f32);
    camera.set_position(eye[0], eye[1], eye[2]);
    camera.look_at(&Vec3::new(target[0], target[1], target[2]));
    camera.update_projection_matrix();
    camera
}

/// The grid's triangles, as world vertices.
fn grid_triangles(renderer: &mut Renderer) -> Vec<[GVec3; 3]> {
    let device = renderer.device().clone();
    renderer.rt_grid_mut().unwrap().wait_readback(&device);
    let rt = renderer.rt_grid().unwrap();
    let count = rt.stats().grid.triangles as usize;
    let buffer = rt.grid().triangles_buffer().clone();
    let words = renderer.read_back_buffer_sync::<u32>(&buffer, buffer.size());
    (0..count)
        .map(|id| {
            let f = |k: usize| f32::from_bits(words[id * 16 + k]);
            let v0 = GVec3::new(f(0), f(1), f(2));
            [v0, v0 + GVec3::new(f(4), f(5), f(6)), v0 + GVec3::new(f(8), f(9), f(10))]
        })
        .collect()
}

/// Every expected triangle matched by one of the grid's, one to one (vertices within 1 mm, in
/// order); returns the grid's unmatched ones.
fn match_triangles(expected: &[[GVec3; 3]], got: &[[GVec3; 3]]) -> (usize, usize) {
    let key = |t: &[GVec3; 3]| {
        let c = (t[0] + t[1] + t[2]) / 3.0;
        ((c.x * 100.0).round() as i64, (c.y * 100.0).round() as i64, (c.z * 100.0).round() as i64)
    };
    let mut buckets: std::collections::HashMap<(i64, i64, i64), Vec<usize>> = std::collections::HashMap::new();
    for (k, t) in got.iter().enumerate() {
        buckets.entry(key(t)).or_default().push(k);
    }
    let mut used = vec![false; got.len()];
    let mut missing = 0;
    for t in expected {
        let (x, y, z) = key(t);
        let found = (-1..=1).flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| (x + dx, y + dy, z + dz)))).find_map(|b| {
            buckets.get(&b)?.iter().copied().find(|&k| !used[k] && (0..3).all(|i| got[k][i].distance(t[i]) < 1e-3))
        });
        match found {
            Some(k) => used[k] = true,
            None => missing += 1,
        }
    }
    (missing, used.iter().filter(|u| !**u).count())
}

#[test]
fn the_grid_holds_exactly_what_the_renderables_place_in_its_box() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    let cell = 0.25;
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell, below: 0.25, ..Default::default() }, cluster_error_cells: 1.0, rebuild_every_frame: false });
    let mut scene = Scene::new();
    // a single box, and one far from the grid
    let cube = || BoxGeometry::new(1.0, 1.0, 1.0);
    let mut single = Renderable::new(cube(), material(false)).with_rt(RtSurface::new([0.8, 0.2, 0.2]));
    single.object.set_position(-3.0, 0.5, -4.0);
    let single = scene.add(SceneNode::Renderable(single));
    let mut far = Renderable::new(cube(), material(false)).with_rt(RtSurface::new([0.8, 0.2, 0.2]));
    far.object.set_position(60.0, 0.5, 0.0);
    scene.add(SceneNode::Renderable(far));
    // 9 cubes in the box, turned and scaled, and 4 far out of it
    let mut cubes: Vec<[f32; 5]> = (0..9).map(|k| [(k % 3) as f32 * 2.0 - 2.0, 0.5, (k / 3) as f32 * 2.0 - 4.0, 0.6 + 0.05 * k as f32, k as f32 * 0.3]).collect();
    let inside = cubes.clone();
    cubes.extend((0..4).map(|k| [40.0 + k as f32 * 2.0, 0.5, 0.0, 1.0, 0.0]));
    scene.add(SceneNode::Renderable(placed(&renderer, cube(), &cubes)));
    // the same cubes raised 1.5 m, left out of the grid by an empty rt band
    let raised: Vec<[f32; 5]> = inside.iter().map(|c| [c[0], c[1] + 1.5, c[2], c[3], c[4]]).collect();
    let mut out = placed(&renderer, cube(), &raised);
    out.instance_culling = out.instance_culling.map(|c| c.with_rt_lod_range(0.0, 0.0));
    scene.add(SceneNode::Renderable(out));
    // two crowns of cards, by their cut at a cell of error
    let cards = crown(400);
    let mesh = std::sync::Arc::new(ClusterMesh::build(&cards, &ClusterOptions { cards: true, ..Default::default() }));
    let crowns = [[3.0f32, 0.0, -3.0, 1.0, 0.0], [5.0, 0.0, -5.0, 1.0, 1.3]];
    let mut trees = placed(&renderer, crown(400), &crowns);
    trees.clusters = Some(ClusterLod::new(mesh.clone()).with_transform(PLACEMENT).with_cone_culling(false));
    trees.rt_placement = None;
    scene.add(SceneNode::Renderable(trees));

    let mut camera = camera_at([0.0, 2.0, 2.0], [0.0, 1.0, -3.0]);
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
    let got = grid_triangles(&mut renderer);
    let stats = renderer.rt_grid().unwrap().stats();
    let (lo, hi) = renderer.rt_grid().unwrap().grid().bounds();
    eprintln!("box {lo} .. {hi}: {} triangles from {} sources, {} references", stats.grid.triangles, stats.sources, stats.grid.references);

    // what the CPU places: the single box, the 9 cubes, and the crowns' cut
    let corners = |m: &glam::Mat4, g: &Geometry| -> Vec<[GVec3; 3]> {
        g.indices.chunks(3).map(|t| [0, 1, 2].map(|i| m.transform_point3(GVec3::from_slice(&g.vertices[t[i] as usize].position[..3])))).collect()
    };
    let cube = cube();
    let mut expected = corners(&glam::Mat4::from_translation(GVec3::new(-3.0, 0.5, -4.0)), &cube);
    for c in &inside {
        expected.extend(cube.indices.chunks(3).map(|t| [0, 1, 2].map(|i| place(c, GVec3::from_slice(&cube.vertices[t[i] as usize].position[..3])))));
    }
    let view = LodView { eye: GVec3::ZERO, pixels_per_radian: 1.0 / cell, near: 0.01, threshold: 1.0, orthographic: true };
    let cut = mesh.select(&view);
    assert!(cut.iter().all(|&c| mesh.clusters[c].level > 0), "the cut is coarser than the cards ({} clusters)", cut.len());
    for c in &crowns {
        for &k in &cut {
            expected.extend(mesh.triangles(k).map(|t| t.map(|v| place(c, GVec3::from_slice(&mesh.vertices[v as usize].position[..3])))));
        }
    }
    // (every expected triangle is in the box)
    assert!(expected.iter().flatten().all(|p| p.cmpge(lo).all() && p.cmple(hi).all()));
    let (missing, extra) = match_triangles(&expected, &got);
    assert_eq!((missing, extra), (0, 0), "{} expected, {} gathered", expected.len(), got.len());
    assert_eq!(stats.sources, 4, "the single box in the box, the cubes, the raised cubes (culled to none) and the crowns");

    // still frames: the cuts' first frame rebuilds it once more (their draw lists were bound
    // then), and after that nothing does
    let mut rebuilt = Vec::new();
    for _ in 0..4 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        rebuilt.push(renderer.rt_grid().unwrap().stats().rebuilt);
    }
    assert_eq!(rebuilt, [true, false, false, false], "still frames keep the grid");
    assert_eq!(match_triangles(&expected, &grid_triangles(&mut renderer)), (0, 0));
    // the camera goes 30 m away: the box follows, empty but for nothing
    let mut away = camera_at([30.0, 2.0, 2.0], [30.0, 1.0, -3.0]);
    renderer.render_scene_offscreen(&mut scene, &mut away, &gbuffer);
    let got = grid_triangles(&mut renderer);
    assert!(renderer.rt_grid().unwrap().stats().rebuilt);
    assert!(got.is_empty(), "nothing within 8 m of x = 30 ({} triangles)", got.len());
    // back, with the single box dynamic: rebuilt every frame
    scene.get_renderable_mut(single).unwrap().dynamic = true;
    for _ in 0..3 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        assert!(renderer.rt_grid().unwrap().stats().rebuilt, "a dynamic renderable rebuilds the grid every frame");
    }
    let got = grid_triangles(&mut renderer);
    let (missing, extra) = match_triangles(&expected, &got);
    assert_eq!((missing, extra), (0, 0));
}
