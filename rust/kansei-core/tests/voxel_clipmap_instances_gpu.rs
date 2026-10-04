//! Voxel GI's clipmap draws a forest as the camera sees it, on a real GPU: instances culled with
//! crossfades (whose compacted records carry a fade the material reads) voxelize through the
//! clipmap's own cull view, a LOD with an empty GI band stays out, and card foliage with cluster
//! LOD voxelizes by its cut for the region, its cards' area summed into the voxels. Skipped
//! (passes) when no adapter is available.

use glam::{IVec3, UVec3, Vec3 as GVec3};
use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform};
use kansei_core::culling::InstanceCulling;
use kansei_core::geometries::{BoxGeometry, Geometry, InstancedGeometry, Vertex};
use kansei_core::gi::{GiSurface, SceneVoxelClipmapOptions, CLIP_SURFACE_WORDS};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 96;
const H: u32 = 64;
const WORDS: usize = CLIP_SURFACE_WORDS as usize;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// Placed by records of position + scale, then yaw, and the crossfade's fade (36 bytes as
/// culled); the fade only drops pixels on screen. Writes its albedo and normal.
const PLACED_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(3) place: vec4<f32>, @location(4) extra: vec4<f32>, @location(5) fade: f32 };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) fade: f32 };
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
fn turn(v: vec3<f32>, a: f32) -> vec3<f32> {
    return vec3<f32>(cos(a) * v.x + sin(a) * v.z, v.y, -sin(a) * v.x + cos(a) * v.z);
}
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let local = turn(v.position.xyz * v.place.w, v.extra.x) + v.place.xyz;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(local, 1.0);
    out.normal = (world_matrix * vec4<f32>(turn(v.normal, v.extra.x), 0.0)).xyz;
    out.fade = v.fade;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> FOut {
    if (in.fade < 0.0) { discard; }
    let n = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0);
    return FOut(vec4<f32>(0.0, 0.0, 0.0, 1.0), vec4<f32>(0.0), n, surface.albedo);
}
"#;

fn placed_material(albedo: [f32; 3]) -> Material {
    let mut m = Material::new("Placed", PLACED_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), cull_mode: CullMode::None, ..Default::default() });
    m.set_uniform_bindable(0, "Placed", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
    m
}

/// `mesh` placed on each of `records` (position xyz + scale, yaw), culled per view with
/// crossfades: the compacted records 36 bytes, the source's 32.
fn placed(renderer: &Renderer, mesh: Geometry, records: &[[f32; 5]], albedo: [f32; 3]) -> Renderable {
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
    let mut r = Renderable::new(InstancedGeometry::new(mesh, count, vec![culled]), placed_material(albedo)).with_gi(GiSurface::new(albedo));
    r.instance_culling = Some(InstanceCulling::new(source, count, 32, 0, 1.0).with_radius_scale(12).with_crossfade(2.0));
    r
}

fn camera_at(eye: [f32; 3], target: [f32; 3]) -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 200.0, W as f32 / H as f32);
    camera.set_position(eye[0], eye[1], eye[2]);
    camera.look_at(&Vec3::new(target[0], target[1], target[2]));
    camera.update_projection_matrix();
    camera
}

fn texel(c: IVec3, dims: [u32; 3]) -> usize {
    let t = c.rem_euclid(UVec3::from(dims).as_ivec3()).as_uvec3();
    ((t.z * dims[1] + t.y) * dims[0] + t.x) as usize
}

/// Level 0's voxels holding a surface, as world points (their centres), with their area words.
fn occupied(renderer: &Renderer) -> Vec<(GVec3, u32)> {
    let gi = renderer.voxel_clipmap().unwrap();
    let layout = *gi.clipmap().layout();
    let origin = gi.clipmap().origin(0).unwrap();
    let buffer = gi.voxelizer().static_surfaces(0);
    let words = renderer.read_back_buffer_sync::<u32>(buffer, buffer.size());
    let mut out = Vec::new();
    for z in 0..layout.dims[2] as i32 {
        for y in 0..layout.dims[1] as i32 {
            for x in 0..layout.dims[0] as i32 {
                let c = origin + IVec3::new(x, y, z);
                let w = &words[WORDS * texel(c, layout.dims)..][..WORDS];
                if w[0] >> 24 != 0 {
                    out.push(((c.as_vec3() + 0.5) * layout.voxel_size, w[4]));
                }
            }
        }
    }
    out
}

/// Instances culled with crossfades voxelize through the region's own cull view, in the layout
/// their material reads (the source's records would not fit it): every cube's surface is in the
/// voxels, nowhere else. A LOD whose GI band is empty stays out.
#[test]
fn crossfaded_instances_voxelize_through_their_culled_layout() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 1, resolution: 64, height_resolution: 32, voxel_size: 0.125, ..Default::default() });
    let cubes: Vec<[f32; 5]> = (0..9).map(|k| [(k % 3) as f32 * 2.0 - 2.0, 0.5, (k / 3) as f32 * 2.0 - 2.0, 0.6 + 0.05 * k as f32, k as f32 * 0.3]).collect();
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(placed(&renderer, BoxGeometry::new(1.0, 1.0, 1.0).into(), &cubes, [0.6, 0.3, 0.2])));
    // the same cubes 1.2 m up, kept out of the voxels by an empty GI band
    let raised: Vec<[f32; 5]> = cubes.iter().map(|c| [c[0], c[1] + 1.2, c[2], c[3], c[4]]).collect();
    let mut out = placed(&renderer, BoxGeometry::new(1.0, 1.0, 1.0).into(), &raised, [0.1, 0.9, 0.1]);
    out.instance_culling = out.instance_culling.map(|c| c.with_gi_lod_range(0.0, 0.0));
    scene.add(SceneNode::Renderable(out));
    let mut camera = camera_at([0.2, 1.0, 0.3], [0.0, 0.0, -2.0]);
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    for _ in 0..2 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
    }
    let voxels = occupied(&renderer);
    // each voxel near some cube's surface (a cube turned about y: within its circumscribed
    // cylinder, its half height), and each cube with voxels
    let near = |p: GVec3, c: &[f32; 5]| (GVec3::new(p.x - c[0], 0.0, p.z - c[2]).length() < c[3] * 0.75 + 0.2) && (p.y - c[1]).abs() < c[3] * 0.5 + 0.2;
    let stray = voxels.iter().filter(|(p, _)| !cubes.iter().any(|c| near(*p, c))).count();
    let per_cube: Vec<usize> = cubes.iter().map(|c| voxels.iter().filter(|(p, _)| near(*p, c)).count()).collect();
    eprintln!("{} voxels, {stray} away from the cubes; per cube {per_cube:?}", voxels.len());
    assert!(per_cube.iter().all(|&n| n > 50), "every cube voxelized: {per_cube:?}");
    assert_eq!(stray, 0, "voxels away from the cubes (the raised ones are out of the voxels)");
}

/// A crown of `count` cards (0.4 m quads) over a cone 4 m tall, 1.2 m wide at its base, as one
/// mesh: small, open and flat, so cluster LOD treats them as cards.
fn crown(count: u32) -> Geometry {
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let hash = |i: u32| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    for k in 0..count {
        let h = hash(k * 3);
        let a = hash(k * 3 + 1) * std::f32::consts::TAU;
        let r = 1.2 * (1.0 - h) * hash(k * 3 + 2).sqrt();
        let c = glam::Vec3::new(r * a.cos(), 0.3 + 3.6 * h, r * a.sin());
        // a card turned about y toward the trunk, tipped down a little
        let u = glam::Vec3::new(-a.sin(), 0.0, a.cos()) * 0.2;
        let v = (glam::Vec3::new(a.cos(), -0.3, a.sin())).normalize() * 0.2;
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

/// Card foliage with cluster LOD voxelizes by its cut for the region (the cards pruned to a voxel
/// of error, each grown to stand for those it replaces): its voxels sum about the cards' area,
/// and hold the crowns.
#[test]
fn card_clusters_voxelize_by_their_cut() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 1, resolution: 64, height_resolution: 64, voxel_size: 0.125, cluster_error_voxels: 1.0, ..Default::default() });
    let count = 600;
    let cards = crown(count);
    let card_area = count as f32 * 0.4 * 0.4;
    let mesh = ClusterMesh::build(&cards, &ClusterOptions { cards: true, ..Default::default() });
    let crowns = [[-1.6f32, 0.0, 0.0, 1.0, 0.0], [1.6, 0.0, 0.0, 1.0, 1.3]];
    let mut r = placed(&renderer, cards, &crowns, [0.1, 0.3, 0.1]);
    r.clusters = Some(ClusterLod::new(mesh).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None }).with_cone_culling(false));
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(r));
    let mut camera = camera_at([0.0, 2.0, 2.5], [0.0, 2.0, 0.0]);
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    // the fill, then the cut's readbacks and the region voxelized again once it holds them all
    let mut areas = Vec::new();
    for _ in 0..16 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        renderer.device().poll(wgpu::Maintain::Wait);
        areas.push(occupied(&renderer).iter().map(|(_, a)| *a as f32 / 256.0).sum::<f32>() * 0.125 * 0.125);
    }
    let voxels = occupied(&renderer);
    let area = *areas.last().unwrap();
    let in_crown = |p: GVec3| crowns.iter().any(|c| GVec3::new(p.x - c[0], 0.0, p.z - c[2]).length() < 1.6 && p.y > -0.1 && p.y < 4.4);
    let stray = voxels.iter().filter(|(p, _)| !in_crown(*p)).count();
    eprintln!("{} voxels, {stray} outside the crowns; area per frame {areas:?} m^2 of the cards' {} m^2", voxels.len(), 2.0 * card_area);
    assert!(voxels.len() > 400, "the crowns are in the voxels");
    assert_eq!(stray, 0);
    // pruned levels keep about their cards' area (cards overlap the more, the more are kept)
    assert!((area / (2.0 * card_area) - 1.0).abs() < 0.3, "the voxels sum {area} m^2 of the cards' {}", 2.0 * card_area);
}

/// A region whose cut needs more than its cluster draw holds at first (its buffers grow from the
/// cull's readbacks, frames later) is voxelized again once the cut has grown: the voxels end up
/// with every card's area, which the first pass could not draw.
#[test]
fn a_region_is_voxelized_again_once_its_cut_has_grown() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    // a fine budget: every card of every crown, past what a cut first makes room for
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 1, resolution: 64, height_resolution: 32, voxel_size: 0.5, cluster_error_voxels: 0.02, ..Default::default() });
    let count = 600;
    let cards = crown(count);
    let mesh = ClusterMesh::build(&cards, &ClusterOptions { cards: true, ..Default::default() });
    let crowns: Vec<[f32; 5]> = (0..400).map(|k| [(k % 20) as f32 * 1.4 - 13.3, 0.0, (k / 20) as f32 * 1.4 - 13.3, 1.0, k as f32]).collect();
    let mut r = placed(&renderer, cards, &crowns, [0.1, 0.3, 0.1]);
    r.clusters = Some(ClusterLod::new(mesh).with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None }).with_cone_culling(false));
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(r));
    let mut camera = camera_at([0.0, 2.0, 0.5], [0.0, 2.0, -3.0]);
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let want = crowns.len() as f32 * count as f32 * 0.4 * 0.4;
    let mut areas = Vec::new();
    for _ in 0..24 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        renderer.device().poll(wgpu::Maintain::Wait);
        areas.push(occupied(&renderer).iter().map(|(_, a)| *a as f32 / 256.0).sum::<f32>() * 0.25);
    }
    eprintln!("area per frame {:?} m^2 of {want} m^2", areas.iter().map(|a| a.round()).collect::<Vec<_>>());
    assert!(areas[0] < 0.8 * want, "the first pass already drew every card");
    assert!((areas.last().unwrap() / want - 1.0).abs() < 0.1, "the voxels end with {} m^2 of {want}", areas.last().unwrap());
}
