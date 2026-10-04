//! Voxel GI's irradiance probes (`SdfProbes`) on a real GPU: an open sky gives pi times its
//! radiance, probes move off nearby surfaces and those inside geometry are left out, the depth
//! moments stop light leaking through a wall, the grid follows the camera by whole cells keeping
//! the history of the probes that stay, and on screen the probes light a closed glowing box as the
//! cones do. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfProbeOptions, SdfProbes, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, PROBES_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use wgpu::util::DeviceExt;

const W: u32 = 128;
const H: u32 = 96;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
struct GBufferOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@vertex fn vertex_main(@location(0) p: vec4<f32>, @location(1) n: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * p;
    out.normal = (normal_matrix * vec4<f32>(n, 0.0)).xyz;
    return out;
}
@fragment fn fragment_main(in: VOut) -> GBufferOut {
    var out: GBufferOut;
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(surface.albedo.rgb, 1.0);
    return out;
}
"#;

fn material(albedo: [f32; 3]) -> Material {
    let mut m = Material::new("Surface", SURFACE_WGSL, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Surface", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
    m
}

fn block(size: [f32; 3], at: [f32; 3], surface: GiSurface) -> Renderable {
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material(surface.albedo)).with_gi(surface);
    r.object.set_position(at[0], at[1], at[2]);
    r
}

fn camera_at(eye: [f32; 3], target: [f32; 3]) -> Camera {
    let mut camera = Camera::new(60.0, 0.05, 50.0, W as f32 / H as f32);
    camera.set_position(eye[0], eye[1], eye[2]);
    camera.look_at(&Vec3::new(target[0], target[1], target[2]));
    camera.update_projection_matrix();
    camera
}

fn read_buffer(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: source.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let out = bytemuck::cast_slice(&buffer.slice(..).get_mapped_range()).to_vec();
    out
}

/// A probe's irradiance toward `n` from its 9 SH coefficients (red).
fn sh_irradiance(sh: &[f32], slot: usize, n: [f32; 3]) -> f32 {
    let [x, y, z] = n;
    let basis = [0.282095, 0.488603 * y, 0.488603 * z, 0.488603 * x, 1.092548 * x * y, 1.092548 * y * z, 0.315392 * (3.0 * z * z - 1.0), 1.092548 * x * z, 0.546274 * (x * x - y * y)];
    basis.iter().enumerate().map(|(i, b)| sh[(slot * 9 + i) * 4] * b).sum::<f32>().max(0.0)
}

/// The storage slot of grid position `local` with the grid's first probe at lattice cell `base`.
fn slot(dims: [u32; 3], base: [i32; 3], local: [i32; 3]) -> usize {
    let w: [i32; 3] = std::array::from_fn(|i| (base[i] + local[i]).rem_euclid(dims[i] as i32));
    ((w[2] * dims[1] as i32 + w[1]) * dims[0] as i32 + w[0]) as usize
}

/// Under a uniform sky and with nothing in the volume, every probe sees the sky all round: its
/// irradiance is pi times the sky's radiance toward every normal, and no probe is left out.
#[test]
fn an_open_sky_gives_pi_times_its_radiance() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    renderer.voxel_gi().unwrap().set_sky_gradient(renderer.queue(), [0.7, 0.5, 0.3], [0.7, 0.5, 0.3]);
    renderer.voxel_gi_mut().unwrap().enable_probes(SdfProbeOptions::default());
    // (the renderer draws at least one renderable) one without GI, far outside the volume
    let mut scene = Scene::new();
    let mut away = Renderable::new(BoxGeometry::new(0.1, 0.1, 0.1), material([0.5; 3]));
    away.object.set_position(0.0, 10.0, 0.0);
    scene.add(SceneNode::Renderable(away));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.0, 0.0, 3.0], [0.0; 3]);
    for _ in 0..3 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
    }
    let probes = renderer.voxel_gi().unwrap().probes().unwrap();
    let sh = read_buffer(renderer.device(), renderer.queue(), probes.sh_buffer());
    let state = read_buffer(renderer.device(), renderer.queue(), probes.state_buffer());
    let want = std::f32::consts::PI * 0.7;
    let mut worst = 0.0f32;
    for p in 0..probes.probe_count() as usize {
        for n in [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, -1.0]] {
            worst = worst.max((sh_irradiance(&sh, p, n) - want).abs() / want);
        }
        assert!(state[p * 8 + 3] <= 0.0, "probe {p} saw back faces in an empty volume");
    }
    eprintln!("{} probes ({:?}): worst irradiance error {:.4}", probes.probe_count(), probes.dims(), worst);
    assert!(worst < 0.01, "irradiance off by {worst}");
}

/// A closed block around one probe and a wall just beside another: the enclosed probe's rays all
/// meet the block's insides (back faces), so it is left out; the one beside the wall moves away
/// from it; one in the open stays where it is.
#[test]
fn probes_move_off_surfaces_and_those_inside_are_left_out() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    renderer.voxel_gi_mut().unwrap().enable_probes(SdfProbeOptions::default());
    // 64 voxels over 2 m, a probe every 8: at -0.875 + 0.25 k
    let mut scene = Scene::new();
    // around the probe at (0.375, 0.375, 0.375)
    scene.add(SceneNode::Renderable(block([0.4; 3], [0.375; 3], GiSurface::new([0.5; 3]))));
    // a wall whose face lies 3 cm from the probe at x = -0.375 (y, z = -0.375)
    scene.add(SceneNode::Renderable(block([0.2, 0.6, 0.6], [-0.245, -0.375, -0.375], GiSurface::new([0.5; 3]))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.0, 0.0, 3.0], [0.0; 3]);
    for _ in 0..4 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
    }
    let probes = renderer.voxel_gi().unwrap().probes().unwrap();
    let (dims, spacing) = (probes.dims(), probes.spacing());
    assert_eq!(dims, [8, 8, 8]);
    let state = read_buffer(renderer.device(), renderer.queue(), probes.state_buffer());
    let at = |k: [i32; 3]| {
        let s = slot(dims, [0; 3], k) * 8;
        ([state[s], state[s + 1], state[s + 2]], state[s + 3])
    };
    let (inside_offset, inside_backs) = at([5, 5, 5]);
    let (beside_offset, beside_backs) = at([2, 2, 2]);
    let (open_offset, open_backs) = at([1, 6, 6]);
    eprintln!("inside: offset {inside_offset:?} back faces {inside_backs:.2}; beside the wall: offset {beside_offset:?} back faces {beside_backs:.2}; open: {open_offset:?} {open_backs:.2}");
    assert!(inside_backs > 0.25, "the enclosed probe is still used ({inside_backs} back faces)");
    assert!(beside_backs <= 0.25 && open_backs <= 0.25, "a probe outside geometry was left out");
    assert!(beside_offset[0] < -0.01 && beside_offset.iter().all(|o| o.abs() <= 0.45 * spacing + 1e-4), "the probe beside the wall moved {beside_offset:?}");
    assert!(open_offset.iter().all(|o| o.abs() < 1e-6), "the probe in the open moved {open_offset:?}");
}

/// `kansei_gi_irradiance` at `points` (with `normals`), through `PROBES_WGSL` as a material reads it.
fn irradiance_at(renderer: &Renderer, probes: &SdfProbes, points: &[[f32; 3]], normals: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let device = renderer.device();
    let code = format!(
        "{PROBES_WGSL}\n{}\n@group(0) @binding(4) var<storage, read> points: array<vec4f>;\n@group(0) @binding(5) var<storage, read> normals: array<vec4f>;\n@group(0) @binding(6) var<storage, read_write> out: array<vec4f>;\n\
         @compute @workgroup_size(1) fn main(@builtin(global_invocation_id) id: vec3u) {{ out[id.x] = vec4f(kansei_gi_irradiance(points[id.x].xyz, normals[id.x].xyz), 0.0); }}",
        SdfProbes::bindings_wgsl(0, 0)
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
    let pad = |v: &[[f32; 3]]| v.iter().flat_map(|p| [p[0], p[1], p[2], 0.0]).collect::<Vec<f32>>();
    let input = |v: &[[f32; 3]]| device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&pad(v)), usage: wgpu::BufferUsages::STORAGE });
    let (p, n) = (input(points), input(normals));
    let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 * points.len() as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
    let mut entries = probes.bind_group_entries(0).to_vec();
    entries.extend([
        wgpu::BindGroupEntry { binding: 4, resource: p.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 5, resource: n.as_entire_binding() },
        wgpu::BindGroupEntry { binding: 6, resource: out.as_entire_binding() },
    ]);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(points.len() as u32, 1, 1);
    }
    renderer.queue().submit(Some(encoder.finish()));
    read_buffer(device, renderer.queue(), &out).chunks(4).map(|c| [c[0], c[1], c[2]]).collect()
}

/// A wall splits the volume: a glowing block lights the +x side, the -x side is dark. Read on the
/// dark face of the wall, the probes on the lit side (inside the trilinear footprint, weighed down
/// but not out for lying behind the surface) leak some of their light without visibility; their
/// depth moments (they see the wall nearer than the point) stop it. The lit face is lit either way.
/// (With the default bias, 3 voxels, the dark face's lookups already land past the lit probes'
/// footprint here; a 1-voxel bias keeps them in it, to test the visibility alone.)
#[test]
fn depth_moments_stop_light_leaking_through_a_wall() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    renderer.voxel_gi_mut().unwrap().enable_probes(SdfProbeOptions { normal_bias_voxels: 1.0, ..Default::default() });
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(block([0.1, 2.2, 2.2], [0.0, 0.0, 0.0], GiSurface::new([0.5; 3]))));
    scene.add(SceneNode::Renderable(block([0.2, 1.6, 1.6], [0.75, 0.0, 0.0], GiSurface::new([0.5; 3]).with_emission([4.0, 4.0, 4.0]))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.5, 0.0, 3.0], [0.0; 3]);
    let points: Vec<[f32; 3]> = [-0.3, -0.1, 0.1, 0.3].iter().flat_map(|&y| [-0.3, 0.0, 0.3].map(|z| [y, z])).flat_map(|[y, z]| [[-0.06, y, z], [0.06, y, z]]).collect();
    let normals: Vec<[f32; 3]> = points.iter().map(|p| [p[0].signum(), 0.0, 0.0]).collect();
    let mut read = |renderer: &mut Renderer, visibility: bool| {
        let probes = renderer.voxel_gi_mut().unwrap().probes_mut().unwrap();
        probes.options.visibility = visibility;
        for _ in 0..40 {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        }
        let e = irradiance_at(renderer, renderer.voxel_gi().unwrap().probes().unwrap(), &points, &normals);
        let mean = |lit: bool| e.iter().zip(&points).filter(|(_, p)| (p[0] > 0.0) == lit).map(|(e, _)| e[0]).sum::<f32>() / (points.len() / 2) as f32;
        (mean(false), mean(true))
    };
    let (dark_off, lit_off) = read(&mut renderer, false);
    let (dark_on, lit_on) = read(&mut renderer, true);
    eprintln!("dark face: {dark_off:.4} without visibility, {dark_on:.4} with; lit face: {lit_off:.4} / {lit_on:.4}");
    assert!(lit_on > 0.5 && lit_off > 0.5, "the lit face is dark");
    assert!(dark_off > 0.005 * lit_off, "no leak to stop: {dark_off} of {lit_off}");
    assert!(dark_on < 0.25 * dark_off, "visibility kept {dark_on} of the leak {dark_off}");
    assert!((lit_on - lit_off).abs() < 0.35 * lit_off, "visibility changed the lit face: {lit_off} -> {lit_on}");
}

/// A grid smaller than the volume follows the camera by whole cells: after a step of one cell the
/// probes still in the grid keep their slots and history, the new ones start over, and every slot
/// records the lattice cell it holds.
#[test]
fn the_grid_follows_the_camera_by_whole_cells() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-2.0; 3], bounds_max: [2.0; 3], ..Default::default() });
    renderer.voxel_gi_mut().unwrap().enable_probes(SdfProbeOptions { max_probes_per_axis: 4, ..Default::default() });
    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(block([0.5; 3], [0.0; 3], GiSurface::new([0.5; 3]))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    // 64 voxels over 4 m, a probe every 8 (0.5 m): 8 lattice cells per axis, from -1.75
    let mut first = camera_at([-0.25, 0.0, 0.0], [-0.25, 0.0, -1.0]);
    for _ in 0..3 {
        renderer.render_scene_offscreen(&mut scene, &mut first, &gbuffer);
    }
    let origin = renderer.voxel_gi().unwrap().probes().unwrap().grid_origin();
    let mut second = camera_at([0.25, 0.0, 0.0], [0.25, 0.0, -1.0]);
    renderer.render_scene_offscreen(&mut scene, &mut second, &gbuffer);
    let probes = renderer.voxel_gi().unwrap().probes().unwrap();
    let moved = probes.grid_origin();
    eprintln!("grid origin {origin:?} -> {moved:?}");
    assert!((moved[0] - origin[0] - probes.spacing()).abs() < 1e-5 && moved[1] == origin[1] && moved[2] == origin[2], "the grid moved {origin:?} -> {moved:?}");
    let dims = probes.dims();
    let base: [i32; 3] = std::array::from_fn(|i| ((moved[i] - (-1.75)) / probes.spacing()).round() as i32);
    let state = read_buffer(renderer.device(), renderer.queue(), probes.state_buffer());
    let words: &[u32] = bytemuck::cast_slice(&state);
    for z in 0..dims[2] as i32 {
        for y in 0..dims[1] as i32 {
            for x in 0..dims[0] as i32 {
                let s = slot(dims, base, [x, y, z]) * 8;
                let cell = [words[s + 4] as i32, words[s + 5] as i32, words[s + 6] as i32];
                assert_eq!(cell, [base[0] + x, base[1] + y, base[2] + z], "slot {s} holds the wrong cell");
                let frames = words[s + 7];
                // the grid's last column entered with the step; the rest were there for 4 frames
                let want = if x == dims[0] as i32 - 1 { 1 } else { 4 };
                assert_eq!(frames, want, "probe {:?} has {frames} frames", [x, y, z]);
            }
        }
    }
}

/// Voxel GI's light on screen (the indirect view) inside a closed glowing box, from the cones or
/// from the probes.
fn glowing_box(probes: bool) -> Option<Vec<[f32; 4]>> {
    let mut renderer = renderer()?;
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-1.7; 3], bounds_max: [1.7; 3], ..Default::default() });
    if probes {
        renderer.voxel_gi_mut().unwrap().enable_probes(SdfProbeOptions::default());
    }
    let mut scene = Scene::new();
    let walls: [([f32; 3], [f32; 3]); 6] = [
        ([3.3, 0.15, 3.3], [0.0, -1.575, 0.0]),
        ([3.3, 0.15, 3.3], [0.0, 1.575, 0.0]),
        ([0.15, 3.3, 3.3], [-1.575, 0.0, 0.0]),
        ([0.15, 3.3, 3.3], [1.575, 0.0, 0.0]),
        ([3.3, 3.3, 0.15], [0.0, 0.0, -1.575]),
        ([3.3, 3.3, 0.15], [0.0, 0.0, 1.575]),
    ];
    for (size, at) in walls {
        scene.add(SceneNode::Renderable(block(size, at, GiSurface::new([0.5; 3]).with_emission([1.0, 0.5, 0.25]))));
    }
    scene.add(SceneNode::Renderable(block([0.6, 0.6, 0.6], [0.5, -1.2, -0.3], GiSurface::new([0.5; 3]))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.0, 0.5, 1.2], [0.0, -1.5, -0.2]);
    let mut effect = VoxelGIEffect::new(renderer.voxel_gi().unwrap().volume(), VoxelGIOptions { quality: VoxelGiQuality::Medium, ..Default::default() });
    effect.show_indirect = true;
    let output = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = output.create_view(&Default::default());
    for _ in 0..60 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        effect.set_probes(renderer.voxel_gi().unwrap().probes());
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &view, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    Some(read_image(renderer.device(), renderer.queue(), &output))
}

/// In a closed glowing box every surface sees glowing walls all round: the probes give about the
/// cones' light, pixel by pixel.
#[test]
fn probes_light_the_screen_as_the_cones_do() {
    let Some(cones) = glowing_box(false) else { return eprintln!("no GPU adapter: skipping") };
    let probes = glowing_box(true).unwrap();
    let mean = |img: &[[f32; 4]]| img.iter().map(|p| p[0]).sum::<f32>() / img.len() as f32;
    let (c, p) = (mean(&cones), mean(&probes));
    let within = cones.iter().zip(&probes).filter(|(c, p)| (p[0] - c[0]).abs() < 0.2 * c[0].max(1e-3)).count() as f32 / cones.len() as f32;
    eprintln!("mean light: cones {c:.4}, probes {p:.4}; {within:.3} of pixels within 20 %");
    assert!(c > 0.05, "the cones see no light");
    assert!((p - c).abs() < 0.15 * c, "probes {p} vs cones {c}");
    assert!(within > 0.7, "only {within} of the pixels agree within 20 %");
}

fn read_image(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let size = texture.size();
    let (w, h) = (size.width, size.height);
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        size,
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    (0..w * h)
        .map(|i| {
            let o = ((i / w) * row + (i % w) * 8) as usize;
            std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]])))
        })
        .collect()
}

fn f16_value(h: u16) -> f32 {
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        31 => sign * f32::INFINITY,
        e => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(e - 15),
    }
}
