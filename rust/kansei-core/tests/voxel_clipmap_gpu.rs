//! Voxel GI through a clipmap round the camera, on a real GPU: a box voxelizes into every level
//! that holds it, each voxel in its toroidal texel; a window that followed the camera holds what
//! one built where it stopped holds; a sun lights the voxels, cones through the clipmap shadow
//! them where no shadow map reaches; and the screen cones see the sky through an empty clipmap.
//! Skipped (passes) when no adapter is available.

use glam::{IVec3, UVec3, Vec3 as GVec3};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelClipmapOptions, VoxelGIEffect, VoxelGIOptions, CLIP_SURFACE_WORDS, VOXEL_WRITE_WGSL};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 128;
const H: u32 = 96;
const WORDS: usize = CLIP_SURFACE_WORDS as usize;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// An unlit surface that writes its albedo and normal into the GBuffer.
const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * v.position;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    var out: GBufferOut;
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(normalize(in.normal) * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(surface.albedo.rgb, 1.0);
    return out;
}
"#;

fn surface_material(albedo: [f32; 3]) -> Material {
    let mut material = Material::new("Surface", SURFACE_WGSL, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    material.set_uniform_bindable(0, "Surface", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
    material
}

/// A box of `size` centred at `centre`, in voxel GI with `albedo`.
fn gi_box(size: [f32; 3], centre: [f32; 3], albedo: [f32; 3]) -> SceneNode {
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), surface_material(albedo)).with_gi(GiSurface::new(albedo));
    r.object.set_position(centre[0], centre[1], centre[2]);
    SceneNode::Renderable(r)
}

fn camera_at(eye: [f32; 3], target: [f32; 3]) -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 200.0, W as f32 / H as f32);
    camera.set_position(eye[0], eye[1], eye[2]);
    camera.look_at(&Vec3::new(target[0], target[1], target[2]));
    camera.update_projection_matrix();
    camera
}

fn read_words(renderer: &Renderer, buffer: &wgpu::Buffer) -> Vec<u32> {
    renderer.read_back_buffer_sync::<u32>(buffer, buffer.size())
}

fn unpack8(v: u32) -> [f32; 4] {
    [(v & 255) as f32, ((v >> 8) & 255) as f32, ((v >> 16) & 255) as f32, (v >> 24) as f32]
}

/// The texel of lattice voxel `c` in a window of `dims`.
fn texel(c: IVec3, dims: [u32; 3]) -> usize {
    let d = UVec3::from(dims).as_ivec3();
    let t = c.rem_euclid(d).as_uvec3();
    ((t.z * dims[1] + t.y) * dims[0] + t.x) as usize
}

fn f16_value(h: u16) -> f32 {
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

/// A 3D rgba16float texture's texels, x fastest.
fn read_radiance(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let size = texture.size();
    let (w, h, d) = (size.width, size.height, size.depth_or_array_layers);
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
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
    let mut out = Vec::with_capacity((w * h * d) as usize);
    for z in 0..d {
        for y in 0..h {
            for x in 0..w {
                let o = ((z * h + y) * row + x * 8) as usize;
                out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
            }
        }
    }
    out
}

/// Signed distance from `p` to the box `[lo, hi]` (negative inside).
fn box_distance(p: GVec3, lo: GVec3, hi: GVec3) -> f32 {
    let centre = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let q = (p - centre).abs() - half;
    q.max(GVec3::ZERO).length() + q.max_element().min(0.0)
}

/// Render `frames` frames of `scene` from `camera`.
fn run(renderer: &mut Renderer, scene: &mut Scene, camera: &mut Camera, frames: u32) {
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    for _ in 0..frames {
        renderer.render_scene_offscreen(scene, camera, &gbuffer);
    }
}

/// A box is voxelized into every level whose window holds it, at that level's voxels: each voxel
/// its surface crosses holds it, in the texel its lattice voxel wraps to, with the box's albedo;
/// no voxel away from the surface does.
#[test]
fn a_box_voxelizes_into_every_level_that_holds_it() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 3, resolution: 32, height_resolution: 32, voxel_size: 0.125, ..Default::default() });
    let (size, centre) = (GVec3::new(1.1, 0.53, 0.77), GVec3::new(0.13, -0.07, 0.04));
    let albedo = [0.8, 0.4, 0.2];
    let mut scene = Scene::new();
    scene.add(gi_box(size.to_array(), centre.to_array(), albedo));
    let mut camera = camera_at([0.3, 0.6, 1.2], [0.0, 0.0, 0.0]);
    // a level filled a frame
    run(&mut renderer, &mut scene, &mut camera, 3);

    let gi = renderer.voxel_clipmap().unwrap();
    let layout = *gi.clipmap().layout();
    let (lo, hi) = (centre - size * 0.5, centre + size * 0.5);
    for level in 0..layout.levels {
        let origin = gi.clipmap().origin(level).expect("filled");
        let vs = layout.level_voxel_size(level);
        let words = read_words(&renderer, gi.voxelizer().static_surfaces(level));
        let (mut shell, mut covered, mut stray, mut occupied) = (0, 0, 0, 0);
        for z in 0..layout.dims[2] as i32 {
            for y in 0..layout.dims[1] as i32 {
                for x in 0..layout.dims[0] as i32 {
                    let c = origin + IVec3::new(x, y, z);
                    let corner = c.as_vec3() * vs;
                    let crosses = corner.cmple(hi).all() && (corner + vs).cmpge(lo).all() && (corner.cmplt(lo).any() || (corner + vs).cmpgt(hi).any());
                    let a = unpack8(words[WORDS * texel(c, layout.dims)]);
                    let held = a[3] > 0.0;
                    if crosses {
                        shell += 1;
                        covered += held as u32;
                    }
                    if !held {
                        continue;
                    }
                    occupied += 1;
                    if box_distance(corner + vs * 0.5, lo, hi).abs() > 1.5 * vs {
                        stray += 1;
                    }
                    for (ch, want) in albedo.iter().enumerate() {
                        assert!((a[ch] / 255.0 - want).abs() < 2.0 / 255.0, "level {level}: albedo {a:?}");
                    }
                }
            }
        }
        eprintln!("level {level} at {origin}: shell {shell} voxels, {covered} covered, {occupied} occupied, {stray} stray");
        assert!(shell >= 8, "level {level}: the window holds the box");
        assert!(covered as f32 >= 0.97 * shell as f32, "level {level}: {covered} of {shell} surface voxels");
        assert_eq!(stray, 0, "level {level}: voxels away from the surface");
    }
}

/// The static surfaces of a clipmap whose windows followed a moving camera, slab by slab, hold
/// what those of one built where the camera stopped hold, voxel for voxel over the part of the
/// lattice both windows cover.
#[test]
fn a_window_that_followed_the_camera_matches_one_built_where_it_stopped() {
    let Some(mut moving) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    let mut fresh = renderer().unwrap();
    let options = SceneVoxelClipmapOptions { levels: 3, resolution: 32, height_resolution: 16, voxel_size: 0.25, snap_voxels: 4, ..Default::default() };
    moving.enable_voxel_clipmap(options);
    fresh.enable_voxel_clipmap(options);
    let scene = || {
        let mut scene = Scene::new();
        scene.add(gi_box([40.0, 0.2, 40.0], [0.0, -0.59, 0.0], [0.3, 0.3, 0.3]));
        for k in 0..12 {
            let (x, z) = ((k % 4) as f32 * 4.3 - 6.0, (k / 4) as f32 * 3.7 - 2.0);
            scene.add(gi_box([0.6 + 0.1 * k as f32, 1.0 + 0.2 * (k % 3) as f32, 0.7], [x, 0.0, z], [0.2 + 0.05 * k as f32, 0.6, 0.4]));
        }
        scene
    };
    let (mut moving_scene, mut fresh_scene) = (scene(), scene());
    let gbuffer = GBuffer::new(moving.device(), W, H, 1);
    // across 12 m in x and 5 m in z, a little each frame, then still while the levels catch up
    let path = |t: f32| [-6.0 + 12.0 * t, 1.5, -1.0 + 5.0 * t];
    let mut camera = camera_at(path(0.0), [0.0, 0.0, 0.0]);
    for frame in 0..80 {
        let eye = path((frame as f32 / 60.0).min(1.0));
        camera.set_position(eye[0], eye[1], eye[2]);
        camera.look_at(&Vec3::new(eye[0] + 1.0, 0.0, eye[2]));
        moving.render_scene_offscreen(&mut moving_scene, &mut camera, &gbuffer);
    }
    let end = path(1.0);
    let mut still = camera_at(end, [end[0] + 1.0, 0.0, end[2]]);
    run(&mut fresh, &mut fresh_scene, &mut still, 6);

    let (a, b) = (moving.voxel_clipmap().unwrap(), fresh.voxel_clipmap().unwrap());
    let layout = *a.clipmap().layout();
    let dims = UVec3::from(layout.dims).as_ivec3();
    for level in 0..layout.levels {
        let (oa, ob) = (a.clipmap().origin(level).unwrap(), b.clipmap().origin(level).unwrap());
        let (wa, wb) = (read_words(&moving, a.voxelizer().static_surfaces(level)), read_words(&fresh, b.voxelizer().static_surfaces(level)));
        let (lo, hi) = (oa.max(ob), (oa + dims).min(ob + dims));
        let (mut compared, mut occupied, mut differ) = (0, 0, 0);
        for z in lo.z..hi.z {
            for y in lo.y..hi.y {
                for x in lo.x..hi.x {
                    let c = IVec3::new(x, y, z);
                    let (va, vb) = (unpack8(wa[WORDS * texel(c, layout.dims)]), unpack8(wb[WORDS * texel(c, layout.dims)]));
                    compared += 1;
                    occupied += (vb[3] > 0.0) as u32;
                    if (va[3] > 0.0) != (vb[3] > 0.0) || (0..3).any(|ch| (va[ch] - vb[ch]).abs() > 2.0) {
                        differ += 1;
                    }
                }
            }
        }
        eprintln!("level {level}: moved to {oa}, built at {ob}; {compared} voxels compared, {occupied} occupied, {differ} differ");
        assert!(compared as f32 > 0.6 * dims.as_vec3().element_product(), "level {level}: the windows overlap");
        assert!(occupied > 100);
        assert!(differ as f32 <= 0.01 * occupied as f32, "level {level}: {differ} of {occupied} voxels differ");
    }
}

/// A sun straight above lights a floor's voxels with albedo / pi times its illuminance; under a
/// roof, with no shadow map, the cones toward the sun through the clipmap find the roof and the
/// floor there stays dark (direct light only).
#[test]
fn a_sun_lights_the_voxels_and_cones_shadow_them_past_the_maps() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 2, resolution: 48, height_resolution: 32, voxel_size: 0.25, ..Default::default() });
    renderer.voxel_clipmap_mut().unwrap().settings.bounce = 0.0;
    let albedo = [0.5, 0.5, 0.5];
    let mut scene = Scene::new();
    // the floor's top at y = -0.49 (inside a voxel layer), a roof over x, z in [-2, 2]
    scene.add(gi_box([12.0, 0.2, 12.0], [0.0, -0.59, 0.0], albedo));
    scene.add(gi_box([4.0, 0.3, 4.0], [0.0, 2.0, 0.0], albedo));
    let sun = 5.0;
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), sun))));
    let mut camera = camera_at([0.0, 1.0, 4.0], [0.0, 0.0, 0.0]);
    // fill the levels, then a frame for the cones to see the roof's voxels lit (opaque)
    run(&mut renderer, &mut scene, &mut camera, 5);

    let gi = renderer.voxel_clipmap().unwrap();
    let layout = *gi.clipmap().layout();
    let origin = gi.clipmap().origin(0).unwrap();
    let vs = layout.level_voxel_size(0);
    let radiance = read_radiance(&renderer, gi.clipmap().texture(0));
    let scale = gi.clipmap().radiance_scale();
    let want = 0.5 / std::f32::consts::PI * sun;
    let y = (-0.49f32 / vs).floor() as i32;
    let (mut lit, mut shadowed) = (Vec::new(), Vec::new());
    for z in origin.z..origin.z + layout.dims[2] as i32 {
        for x in origin.x..origin.x + layout.dims[0] as i32 {
            let p = (GVec3::new(x as f32, 0.0, z as f32) + 0.5) * vs;
            let r = radiance[texel(IVec3::new(x, y, z), layout.dims)];
            if p.x.abs() < 1.0 && p.z.abs() < 1.0 {
                shadowed.push(r[0] * scale);
            } else if (p.x.abs() > 3.0 || p.z.abs() > 3.0) && p.x.abs() < 5.5 && p.z.abs() < 5.5 {
                lit.push(r[0] * scale);
            }
        }
    }
    let mean = lit.iter().sum::<f32>() / lit.len() as f32;
    let darkest_lit = lit.iter().cloned().fold(f32::MAX, f32::min);
    let brightest_shadowed = shadowed.iter().cloned().fold(0.0, f32::max);
    eprintln!("{} lit voxels (mean {mean}, darkest {darkest_lit}, expected {want}), {} under the roof (brightest {brightest_shadowed})", lit.len(), shadowed.len());
    assert!(lit.len() > 100 && shadowed.len() > 40);
    assert!((mean / want - 1.0).abs() < 0.05, "the open floor leaves {mean}, expected {want}");
    assert!(darkest_lit > 0.9 * want, "an open voxel leaves {darkest_lit}");
    assert!(brightest_shadowed < 0.05 * want, "a voxel under the roof leaves {brightest_shadowed}");
}

/// Through an empty clipmap every cone escapes to a uniform sky of radiance L, so a surface of
/// albedo a gains a * L.
#[test]
fn the_screen_cones_see_the_sky_through_an_empty_clipmap() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 3, resolution: 32, height_resolution: 16, voxel_size: 0.25, ..Default::default() });
    let albedo = [0.5, 0.25, 1.0];
    let mut scene = Scene::new();
    // a floor, not in the clipmap (no GI surface)
    let mut floor = Renderable::new(BoxGeometry::new(30.0, 0.2, 30.0), surface_material(albedo));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.0, 1.5, 3.0], [0.0, 0.0, 0.0]);
    let sky = [2.0, 1.0, 0.5];
    let mut effect = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
    effect.sky_gradient = (sky, sky);
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
    let output_view = output.create_view(&Default::default());
    for _ in 0..6 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let pixels = read_radiance_2d(&renderer, &output);
    let (mut sum, mut count) = ([0.0f32; 3], 0);
    for y in H * 2 / 3..H - 4 {
        for x in W / 4..W * 3 / 4 {
            let p = pixels[(y * W + x) as usize];
            for c in 0..3 {
                sum[c] += p[c];
            }
            count += 1;
        }
    }
    let got = sum.map(|s| s / count as f32);
    let want: [f32; 3] = std::array::from_fn(|c| (albedo[c] * 255.0).round() / 255.0 * sky[c]);
    eprintln!("the floor gains {got:?}, expected {want:?}");
    for c in 0..3 {
        assert!((got[c] / want[c] - 1.0).abs() < 0.03, "channel {c} gains {} of {}", got[c], want[c]);
    }
}

/// A 2D rgba16float texture's texels, row by row.
fn read_radiance_2d(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let size = texture.size();
    let row = (size.width * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * size.height) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(size.height) } },
        size,
    );
    queue.submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((size.width * size.height) as usize);
    for y in 0..size.height {
        for x in 0..size.width {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// A material whose voxel entry cuts out the texels of every other quarter of u (an alpha-tested
/// card), keeping the coverage of the samples elsewhere.
const CUTOUT_VOXEL_WGSL: &str = r#"
struct CutVOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn cut_vertex(v: VIn) -> CutVOut {
    var out: CutVOut;
    out.clip = projection_matrix * view_matrix * world_matrix * v.position;
    out.uv = v.uv;
    return out;
}
@fragment
fn voxel_main(in: CutVOut, @builtin(front_facing) front: bool) {
    let kept = fract(in.uv.x * 4.0) < 0.5;
    kansei_voxel_write_coverage(in.clip, front, surface.albedo.rgb, vec3<f32>(0.0), select(0.0, 1.0, kept));
}
"#;

fn cutout_material(albedo: [f32; 3]) -> Material {
    // the voxel pass draws `vertex_main`: make it the cut-out's, with the uv
    let shader = format!("{VOXEL_WRITE_WGSL}\n{}\n{CUTOUT_VOXEL_WGSL}", SURFACE_WGSL.replace("fn vertex_main(", "fn plain_vertex(")).replace("fn cut_vertex(", "fn vertex_main(").replace("fn fragment_main(in: VOut)", "fn plain_fragment(in: VOut)");
    let shader = shader + "\n@fragment fn fragment_main(in: CutVOut) -> GBufferOut { var out: GBufferOut; out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0); out.emissive = vec4<f32>(0.0); out.normal = vec4<f32>(0.5, 1.0, 0.5, 1.0); out.albedo = vec4<f32>(surface.albedo.rgb, 1.0); return out; }\n";
    let mut material = Material::new(
        "Cutout",
        &shader,
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: Some("voxel_main"), cull_mode: kansei_core::materials::CullMode::None, ..Default::default() },
    );
    material.set_uniform_bindable(0, "Cutout", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
    material
}

/// The area a level's voxels sum, in square metres (word 4 of each voxel, voxel faces / 256).
fn summed_area(words: &[u32], voxel: f32) -> f32 {
    words.chunks(WORDS).map(|w| w[4] as f32 / 256.0).sum::<f32>() * voxel * voxel
}

/// A clipmap's voxels sum the area of the surfaces in them, whatever their tilt; an alpha-tested
/// surface's cut-out texels add none; a flat slab's voxels are opaque (its two faces in a voxel
/// agree on their axis), those of a sheet with `GiSurface::opacity` that much.
#[test]
fn voxels_sum_the_surface_area_that_makes_them_opaque() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 1, resolution: 64, height_resolution: 64, voxel_size: 0.125, ..Default::default() });
    renderer.voxel_clipmap_mut().unwrap().settings.bounce = 0.0;
    // (the level's window: 8 m round the camera)
    let mut camera = camera_at([0.3, 0.4, 1.0], [0.0, 0.0, 0.0]);
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let level_area = |renderer: &Renderer| {
        let gi = renderer.voxel_clipmap().unwrap();
        summed_area(&read_words(renderer, gi.voxelizer().static_surfaces(0)), gi.clipmap().layout().voxel_size)
    };
    // a 2 x 1.5 m plate tilted about two axes: 3 m^2
    let mut scene = Scene::new();
    let mut plate = Renderable::new(BoxGeometry::new(2.0, 0.0, 1.5), surface_material([0.5; 3])).with_gi(GiSurface::new([0.5; 3]));
    plate.object.rotation.x = 0.5;
    plate.object.rotation.z = 0.35;
    plate.material.options.cull_mode = kansei_core::materials::CullMode::None;
    scene.add(SceneNode::Renderable(plate));
    run(&mut renderer, &mut scene, &mut camera, 2);
    let tilted = level_area(&renderer);
    // the same plate cut out every other quarter of u: half of it
    renderer.voxel_clipmap_mut().unwrap().invalidate();
    let mut cut_scene = Scene::new();
    let mut cut = Renderable::new(kansei_core::geometries::PlaneGeometry::new(2.0, 1.5), cutout_material([0.5; 3])).with_gi(GiSurface::new([0.5; 3]));
    cut.object.rotation.x = 0.5;
    cut.object.rotation.z = 0.35;
    cut_scene.add(SceneNode::Renderable(cut));
    for _ in 0..2 {
        renderer.render_scene_offscreen(&mut cut_scene, &mut camera, &gbuffer);
    }
    let cut_area = level_area(&renderer);
    eprintln!("tilted plate: {tilted} m^2 summed (both faces of a box of no thickness: 6); cut out: {cut_area} m^2 (1.5)");
    // (a box of zero height: its top and bottom faces, 2 x 3 m^2; the plane: one face)
    assert!((tilted / 6.0 - 1.0).abs() < 0.06, "the plate sums {tilted} m^2");
    assert!((cut_area / 1.5 - 1.0).abs() < 0.1, "the cut-out plane sums {cut_area} m^2");

    // a level slab and a plate of opacity 0.4, both flat: their voxels' opacity
    renderer.voxel_clipmap_mut().unwrap().invalidate();
    let mut flat = Scene::new();
    flat.add(gi_box([3.0, 0.06, 3.0], [-1.6, -0.94, 0.0], [0.5; 3]));
    // (one face: a slab's two faces in one voxel would add up)
    let mut light = Renderable::new(kansei_core::geometries::PlaneGeometry::new(3.0, 3.0), surface_material([0.5; 3])).with_gi(GiSurface::new([0.5; 3]).with_opacity(0.4));
    light.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    light.object.set_position(1.6, -0.94, 0.0);
    flat.add(SceneNode::Renderable(light));
    flat.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 1.0))));
    for _ in 0..3 {
        renderer.render_scene_offscreen(&mut flat, &mut camera, &gbuffer);
    }
    let gi = renderer.voxel_clipmap().unwrap();
    let layout = *gi.clipmap().layout();
    let origin = gi.clipmap().origin(0).unwrap();
    let radiance = read_radiance(&renderer, gi.clipmap().texture(0));
    let y = (-0.94f32 / layout.voxel_size).floor() as i32;
    let (mut opaque, mut partial) = (Vec::new(), Vec::new());
    for z in origin.z..origin.z + layout.dims[2] as i32 {
        for x in origin.x..origin.x + layout.dims[0] as i32 {
            let p = (GVec3::new(x as f32, 0.0, z as f32) + 0.5) * layout.voxel_size;
            if p.z.abs() > 1.2 || (p.x.abs() - 1.6).abs() > 1.2 {
                continue;
            }
            let a = radiance[texel(IVec3::new(x, y, z), layout.dims)][3];
            if p.x < 0.0 { opaque.push(a) } else { partial.push(a) }
        }
    }
    let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
    eprintln!("slab voxels: opacity {} (of {}); with opacity 0.4: {} (of {})", mean(&opaque), opaque.len(), mean(&partial), partial.len());
    assert!(opaque.len() > 100 && partial.len() > 100);
    assert!(opaque.iter().all(|&a| a > 0.97), "a flat slab's voxels are opaque: {:?}", opaque.iter().cloned().fold(1.0, f32::min));
    assert!((mean(&partial) - 0.4).abs() < 0.06, "the light slab's voxels: {}", mean(&partial));
}



/// An open floor that is in the clipmap itself sees the whole sky through it: its own voxels (at
/// every level, the coarse ones thick) don't hide the sky from its cones.
#[test]
fn an_open_floor_in_the_clipmap_sees_the_sky() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 4, resolution: 32, height_resolution: 16, voxel_size: 0.25, ..Default::default() });
    renderer.voxel_clipmap_mut().unwrap().settings.bounce = 0.0;
    let albedo = [0.5, 0.5, 0.5];
    let mut scene = Scene::new();
    scene.add(gi_box([120.0, 0.4, 120.0], [0.0, -0.2, 0.0], albedo));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera_at([0.0, 1.5, 3.0], [0.0, 0.0, -2.0]);
    let sky = [1.0, 1.0, 1.0];
    let mut effect = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
    effect.sky_gradient = (sky, sky);
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
    let output_view = output.create_view(&Default::default());
    for _ in 0..8 {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let pixels = read_radiance_2d(&renderer, &output);
    // rows from near the camera to the far floor (the top third is sky)
    let rows: Vec<f32> = [H - 6, H * 3 / 4, H * 3 / 5, H / 2, H * 2 / 5].iter().map(|&y| (W / 4..W * 3 / 4).map(|x| pixels[(y * W + x) as usize][0]).sum::<f32>() / (W / 2) as f32).collect();
    let want = (0.5f32 * 255.0).round() / 255.0;
    eprintln!("the floor gains {rows:?} from near to far, expected {want}");
    for (k, got) in rows.iter().enumerate() {
        assert!((got / want - 1.0).abs() < 0.1, "row {k}: the floor gains {got} of {want}");
    }
}

/// The floor's gain through `effect` (`show_indirect`) after `frames` frames of `scene`: the mean
/// of the image's rows `rows` (fractions of its height), its middle half across.
fn floor_gain(renderer: &mut Renderer, scene: &mut Scene, camera: &mut Camera, effect: &mut VoxelGIEffect, frames: u32, rows: &[f32]) -> Vec<f32> {
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
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
    let output_view = output.create_view(&Default::default());
    for _ in 0..frames {
        renderer.render_scene_offscreen(scene, camera, &gbuffer);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let pixels = read_radiance_2d(renderer, &output);
    rows.iter().map(|&f| {
        let y = ((H as f32 * f) as u32).min(H - 1);
        (W / 4..W * 3 / 4).map(|x| pixels[(y * W + x) as usize][0]).sum::<f32>() / (W / 2) as f32
    }).collect()
}

/// Lit through the clipmap's probes, an open floor in the clipmap gains what the cones give it,
/// albedo times the sky; under a roof, most of the sky is gone.
#[test]
fn the_probes_light_an_open_floor_and_a_roof_hides_the_sky() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 3, resolution: 32, height_resolution: 16, voxel_size: 0.25, ..Default::default() });
    let gi = renderer.voxel_clipmap_mut().unwrap();
    gi.settings.bounce = 0.0;
    gi.enable_probes(kansei_core::gi::ClipmapProbeOptions { probes_per_frame: 1 << 16, hysteresis: 0.5, ..Default::default() });
    // (the probes see the clipmap's sky)
    renderer.voxel_clipmap().unwrap().set_sky_gradient(renderer.queue(), [1.0; 3], [1.0; 3]);
    let mut scene = Scene::new();
    scene.add(gi_box([120.0, 0.4, 120.0], [0.0, -0.2, 0.0], [0.5; 3]));
    let mut camera = camera_at([0.0, 1.5, 3.0], [0.0, 0.0, -2.0]);
    let mut effect = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
    effect.set_clipmap_probes(renderer.voxel_clipmap().unwrap().probes());
    assert!(effect.uses_clipmap_probes());
    effect.sky_gradient = ([1.0; 3], [1.0; 3]);
    effect.show_indirect = true;
    let rows = [0.95, 0.8, 0.65, 0.55];
    let open = floor_gain(&mut renderer, &mut scene, &mut camera, &mut effect, 10, &rows);
    let want = (0.5f32 * 255.0).round() / 255.0;
    eprintln!("open floor through the probes: {open:?}, expected {want}");
    for got in &open {
        assert!((got / want - 1.0).abs() < 0.08, "the open floor gains {got} of {want}");
    }
    // a roof 1.5 m up over the floor the camera sees, 12 m wide
    scene.add(gi_box([12.0, 0.3, 12.0], [0.0, 1.5, -2.0], [0.5; 3]));
    let mut low = camera_at([0.0, 0.8, 1.0], [0.0, 0.0, -2.0]);
    let roofed = floor_gain(&mut renderer, &mut scene, &mut low, &mut effect, 12, &rows);
    eprintln!("under the roof: {roofed:?}");
    for got in &roofed {
        assert!(*got < 0.25 * want, "under the roof the floor gains {got}");
    }
}

/// A probe inside a block (the clipmap's voxels opaque round it) is left out; those round the
/// block in the open are traced.
#[test]
fn probes_inside_surfaces_are_left_out() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 1, resolution: 32, height_resolution: 32, voxel_size: 0.25, ..Default::default() });
    renderer.voxel_clipmap_mut().unwrap().enable_probes(kansei_core::gi::ClipmapProbeOptions { probes_per_frame: 1 << 16, ..Default::default() });
    let mut scene = Scene::new();
    // a solid block: its walls 0.25 m thick on every side round a 0.5 m core... the voxelizer
    // fills only surfaces, so make the inside a stack of slabs
    for k in 0..8 {
        scene.add(gi_box([2.0, 0.24, 2.0], [0.0, -0.9 + k as f32 * 0.25, 0.0], [0.5; 3]));
    }
    let mut camera = camera_at([3.0, 1.0, 3.0], [0.0, 0.0, 0.0]);
    run(&mut renderer, &mut scene, &mut camera, 3);
    let gi = renderer.voxel_clipmap().unwrap();
    let probes = gi.probes().unwrap();
    let layout = *probes.layout();
    let origin = probes.origin(0).unwrap();
    let words = renderer.read_back_buffer_sync::<f32>(probes.probe_buffer(), probes.probe_buffer().size());
    let (mut inside, mut left_out, mut open, mut traced) = (0, 0, 0, 0);
    for z in 0..layout.dims[2] as i32 {
        for y in 0..layout.dims[1] as i32 {
            for x in 0..layout.dims[0] as i32 {
                let c = origin + IVec3::new(x, y, z);
                let p = c.as_vec3() * layout.voxel_size;
                let a = words[16 * texel(c, layout.dims) + 3];
                if p.x.abs() < 0.6 && p.z.abs() < 0.6 && p.y > -0.7 && p.y < 0.8 {
                    inside += 1;
                    left_out += (a == -1.0) as u32;
                } else if p.x.abs() > 1.6 || p.z.abs() > 1.6 || p.y > 1.4 {
                    open += 1;
                    traced += (a >= 0.0) as u32;
                }
            }
        }
    }
    eprintln!("{left_out} of {inside} probes inside the block left out; {traced} of {open} in the open traced");
    assert!(inside > 8 && left_out == inside);
    assert!(open > 100 && traced == open);
}

/// Down a long corridor open to the sky, the floor's centre sees the sky through the slot between
/// the walls: sin(atan(w / h)) of a uniform sky's irradiance (the view factor of an infinite slot
/// of half-width w between walls h high), through the cones traced per pixel and through the
/// probes alike.
#[test]
fn the_cones_and_the_probes_see_the_sky_down_a_corridor() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_clipmap(SceneVoxelClipmapOptions { levels: 5, resolution: 32, height_resolution: 16, voxel_size: 0.25, ..Default::default() });
    let gi = renderer.voxel_clipmap_mut().unwrap();
    gi.settings.bounce = 0.0;
    // (the default history: it averages the probes' sparse samples)
    gi.enable_probes(kansei_core::gi::ClipmapProbeOptions { probes_per_frame: 1 << 16, ..Default::default() });
    renderer.voxel_clipmap().unwrap().set_sky_gradient(renderer.queue(), [1.0; 3], [1.0; 3]);
    // walls 6 m high, 1.5 m either side of the corridor's axis (along z), 200 m long, black
    let (w, h) = (1.5f32, 6.0f32);
    let mut scene = Scene::new();
    scene.add(gi_box([200.0, 0.4, 200.0], [0.0, -0.2, 0.0], [0.5; 3]));
    scene.add(gi_box([1.0, h, 200.0], [-w - 0.5, h * 0.5, 0.0], [0.0; 3]));
    scene.add(gi_box([1.0, h, 200.0], [w + 0.5, h * 0.5, 0.0], [0.0; 3]));
    let want = (0.5f32 * 255.0).round() / 255.0 * (w / h).atan().sin();
    let mut camera = camera_at([0.0, 1.2, 2.0], [0.0, 0.0, -2.0]);
    let rows = [0.95, 0.85];
    for probes in [false, true] {
        let mut effect = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
        if probes {
            effect.set_clipmap_probes(renderer.voxel_clipmap().unwrap().probes());
        }
        effect.sky_gradient = ([1.0; 3], [1.0; 3]);
        effect.show_indirect = true;
        let got = floor_gain(&mut renderer, &mut scene, &mut camera, &mut effect, 40, &rows);
        let label = if probes { "probes" } else { "cones" };
        eprintln!("{label}: the corridor's floor gains {got:?}, the slot's view factor gives {want}");
        // (cones as wide as the slot would fill it: they read voxels finer than they are wide)
        let tolerance = if probes { 0.25 } else { 0.2 };
        for g in &got {
            assert!((g / want - 1.0).abs() < tolerance, "{label}: the corridor's floor gains {g} of {want}");
        }
    }
}
