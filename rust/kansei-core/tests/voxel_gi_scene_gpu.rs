//! Voxel GI of a scene's meshes on a real GPU: a box voxelizes into its surface with outward
//! normals, a material's voxel entry gives textured albedo, a spot light lights the voxels
//! through its shadow map, and the screen-space cones see the sky through an empty volume.
//! Skipped (passes) when no adapter is available.

use kansei_core::buffers::{Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, VOXEL_WRITE_WGSL};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::ScreenSpaceGIOptions;
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};

const W: u32 = 128;
const H: u32 = 96;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// An unlit surface that writes its albedo and normal into the GBuffer, and optionally a voxel
/// entry that reads its albedo from a texture across u (`textured`).
const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};
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
    out.uv = v.uv;
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

const TEXTURED_VOXEL_WGSL: &str = r#"
@group(0) @binding(1) var albedo_texture: texture_2d<f32>;
@group(0) @binding(2) var albedo_sampler: sampler;

@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, textureSample(albedo_texture, albedo_sampler, in.uv).rgb, vec3<f32>(0.0));
}
"#;

fn surface_material(albedo: [f32; 3]) -> Material {
    let mut material = Material::new(
        "Surface",
        SURFACE_WGSL,
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    material.set_uniform_bindable(0, "Surface", &[albedo[0], albedo[1], albedo[2], 1.0f32]);
    material
}

/// A material whose voxels take a 2x1 texture's colours: `left` for u < 0.5, `right` above.
fn textured_material(left: [u8; 3], right: [u8; 3]) -> Material {
    let mut material = Material::new(
        "Textured",
        &format!("{VOXEL_WRITE_WGSL}\n{SURFACE_WGSL}\n{TEXTURED_VOXEL_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: Some("voxel_main"), ..Default::default() },
    );
    material.set_uniform_bindable(0, "Textured", &[0.5f32, 0.5, 0.5, 1.0]);
    let texels = [left[0], left[1], left[2], 255, right[0], right[1], right[2], 255];
    material.set_bindable(1, Texture::from_rgba("Halves", 2, 1, &texels));
    material.set_bindable(2, Sampler::new(wgpu::FilterMode::Nearest, wgpu::FilterMode::Nearest));
    material
}

fn camera() -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 1.5, 3.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    camera
}

fn read_words(renderer: &Renderer, buffer: &wgpu::Buffer) -> Vec<u32> {
    renderer.read_back_buffer_sync::<u32>(buffer, buffer.size())
}

fn unpack8(v: u32) -> [f32; 4] {
    [(v & 255) as f32, ((v >> 8) & 255) as f32, ((v >> 16) & 255) as f32, (v >> 24) as f32]
}

/// A voxel's surface: (albedo 0..1, unit normal), or None where nothing was drawn.
fn surface_at(words: &[u32], index: usize) -> Option<([f32; 3], glam::Vec3)> {
    let a = unpack8(words[3 * index]);
    if a[3] == 0.0 {
        return None;
    }
    let n = unpack8(words[3 * index + 1]);
    let normal = glam::Vec3::new(n[0], n[1], n[2]) / 255.0 * 2.0 - 1.0;
    Some(([a[0] / 255.0, a[1] / 255.0, a[2] / 255.0], normal.normalize_or_zero()))
}

/// Signed distance from `p` to the box `[lo, hi]` (negative inside).
fn box_distance(p: glam::Vec3, lo: glam::Vec3, hi: glam::Vec3) -> f32 {
    let centre = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let q = (p - centre).abs() - half;
    q.max(glam::Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// A box drawn into voxels covers exactly its surface: every voxel the surface crosses holds it,
/// no voxel away from it does, each face's voxels hold that face's outward normal and the
/// renderable's albedo.
#[test]
fn a_voxelized_box_has_its_surface_and_its_normals() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    let (size, centre) = (glam::Vec3::new(1.1, 0.53, 0.77), glam::Vec3::new(0.13, -0.07, 0.04));
    let albedo = [0.8, 0.4, 0.2];
    let mut scene = Scene::new();
    let mut boxed = Renderable::new(BoxGeometry::new(size.x, size.y, size.z), surface_material(albedo)).with_gi(GiSurface::new(albedo));
    boxed.object.set_position(centre.x, centre.y, centre.z);
    scene.add(SceneNode::Renderable(boxed));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);

    let gi = renderer.voxel_gi().unwrap();
    let layout = *gi.volume().layout();
    let words = read_words(&renderer, gi.voxelizer().static_surfaces());
    let (lo, hi) = (centre - size * 0.5, centre + size * 0.5);
    let vs = layout.voxel_size;
    let [dx, dy, dz] = layout.dims;
    let (mut shell, mut covered, mut stray, mut occupied) = (0, 0, 0, 0);
    let mut faces = [(0, 0); 6];
    let axes = [glam::Vec3::X, glam::Vec3::NEG_X, glam::Vec3::Y, glam::Vec3::NEG_Y, glam::Vec3::Z, glam::Vec3::NEG_Z];
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let index = ((z * dy + y) * dx + x) as usize;
                let corner = glam::Vec3::from(layout.origin) + glam::UVec3::new(x, y, z).as_vec3() * vs;
                let centre_v = corner + vs * 0.5;
                let d = box_distance(centre_v, lo, hi);
                // the surface crosses the voxel when it overlaps the box without lying inside it
                let crosses = corner.cmple(hi).all() && (corner + vs).cmpge(lo).all() && (corner.cmplt(lo).any() || (corner + vs).cmpgt(hi).any());
                let surface = surface_at(&words, index);
                if crosses {
                    shell += 1;
                    covered += surface.is_some() as u32;
                }
                let Some((a, n)) = surface else { continue };
                occupied += 1;
                // a little past the surface: multisampled coverage reaches into the next voxel
                if d.abs() > 1.5 * vs {
                    stray += 1;
                }
                for c in 0..3 {
                    assert!((a[c] - albedo[c]).abs() < 2.0 / 255.0, "albedo {a:?}");
                }
                // well inside one face: its outward normal
                let p = centre_v - centre;
                let rel = p / (size * 0.5);
                let (axis, along) = rel.abs().to_array().into_iter().enumerate().fold((0, 0.0f32), |m, (i, v)| if v > m.1 { (i, v) } else { m });
                let others_inside = (0..3).filter(|&i| i != axis).all(|i| rel[i].abs() < 1.0 - 3.0 * vs / (size[i] * 0.5));
                if others_inside && along > 0.0 && d.abs() < vs {
                    let face = 2 * axis + (rel[axis] < 0.0) as usize;
                    faces[face].0 += 1;
                    if n.dot(axes[face]) > 0.95 {
                        faces[face].1 += 1;
                    }
                }
            }
        }
    }
    eprintln!("shell {shell} voxels, {covered} covered; {occupied} occupied, {stray} stray; faces (voxels, outward) {faces:?}");
    assert!(covered as f32 >= 0.97 * shell as f32, "{covered} of {shell} surface voxels");
    assert!(stray == 0, "{stray} voxels away from the surface");
    for (face, (count, outward)) in faces.iter().enumerate() {
        assert!(*count > 50, "face {face}: {count} voxels");
        assert!(*outward as f32 >= 0.98 * *count as f32, "face {face}: {outward} of {count} face outward");
    }
}

/// A material's voxel entry puts its texture into the voxels: the box's top face, whose u runs
/// along x, is red on its left half and blue on its right; the same box without the entry has its
/// constant albedo everywhere.
#[test]
fn a_material_voxel_entry_gives_textured_albedo() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    let constant = [0.3, 0.6, 0.9];
    let mut scene = Scene::new();
    let mut textured = Renderable::new(BoxGeometry::new(1.2, 0.4, 0.6), textured_material([255, 0, 0], [0, 0, 255])).with_gi(GiSurface::new(constant));
    textured.object.set_position(0.0, 0.31, -0.4);
    scene.add(SceneNode::Renderable(textured));
    let mut plain = Renderable::new(BoxGeometry::new(1.2, 0.4, 0.6), surface_material(constant)).with_gi(GiSurface::new(constant));
    plain.object.set_position(0.0, -0.5, 0.4);
    scene.add(SceneNode::Renderable(plain));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);

    let gi = renderer.voxel_gi().unwrap();
    let layout = *gi.volume().layout();
    let words = read_words(&renderer, gi.voxelizer().static_surfaces());
    let vs = layout.voxel_size;
    let [dx, dy, dz] = layout.dims;
    // the top faces' voxels (y within a voxel of each box's top), away from their edges
    let (mut left, mut right, mut plain_top) = (Vec::new(), Vec::new(), Vec::new());
    for z in 0..dz {
        for y in 0..dy {
            for x in 0..dx {
                let p = glam::Vec3::from(layout.origin) + (glam::UVec3::new(x, y, z).as_vec3() + 0.5) * vs;
                let Some((a, n)) = surface_at(&words, ((z * dy + y) * dx + x) as usize) else { continue };
                if n.y < 0.9 || p.x.abs() > 0.5 || p.x.abs() < 0.1 {
                    continue;
                }
                if (p.y - 0.51).abs() < vs && (p.z + 0.4).abs() < 0.2 {
                    if p.x < 0.0 { left.push(a) } else { right.push(a) }
                } else if (p.y + 0.3).abs() < vs && (p.z - 0.4).abs() < 0.2 {
                    plain_top.push(a);
                }
            }
        }
    }
    eprintln!("top voxels: {} left, {} right, {} on the plain box", left.len(), right.len(), plain_top.len());
    assert!(left.len() > 30 && right.len() > 30 && plain_top.len() > 30);
    let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|c| (a[c] - b[c]).abs() < 3.0 / 255.0);
    assert!(left.iter().all(|&a| close(a, [1.0, 0.0, 0.0])), "left half: {:?}", left.iter().find(|&&a| !close(a, [1.0, 0.0, 0.0])));
    assert!(right.iter().all(|&a| close(a, [0.0, 0.0, 1.0])), "right half: {:?}", right.iter().find(|&&a| !close(a, [0.0, 0.0, 1.0])));
    assert!(plain_top.iter().all(|&a| close(a, constant)), "plain box: {:?}", plain_top.iter().find(|&&a| !close(a, constant)));
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

/// Mip 0 of the volume, rgba per voxel.
fn read_radiance(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let (device, queue) = (renderer.device(), renderer.queue());
    let size = texture.size();
    let (w, h, d) = (size.width, size.height, size.depth_or_array_layers);
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * h * d) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
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

/// A shadowed downlight over a floor with a block between them: the floor's voxels in the light
/// leave albedo / pi times the light's illuminance on them, those in the block's shadow nothing
/// (direct light only).
#[test]
fn voxels_are_lit_through_the_lights_shadow_map() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_spot_shadows(512, 1);
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-1.0; 3], bounds_max: [1.0; 3], ..Default::default() });
    renderer.voxel_gi_mut().unwrap().settings.bounce = 0.0;
    let albedo = [0.5, 0.5, 0.5];
    let mut scene = Scene::new();
    // the floor's top at y = -0.49 (inside a voxel layer, not on its boundary), a block over its
    // left side
    let mut floor = Renderable::new(BoxGeometry::new(1.8, 0.2, 1.8), surface_material(albedo)).with_gi(GiSurface::new(albedo));
    floor.object.set_position(0.0, -0.59, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut block = Renderable::new(BoxGeometry::new(0.5, 0.1, 0.5), surface_material(albedo)).with_gi(GiSurface::new(albedo));
    block.object.set_position(-0.4, 0.2, 0.0);
    scene.add(SceneNode::Renderable(block));
    let light_pos = glam::Vec3::new(0.0, 0.9, 0.0);
    let mut lamp = SpotLight::new(Vec3::new(light_pos.x, light_pos.y, light_pos.z), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 10.0, 20.0, 60f32.to_radians(), 70f32.to_radians());
    lamp.cast_shadow = true;
    lamp.source_radius = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
    renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);

    let gi = renderer.voxel_gi().unwrap();
    let layout = *gi.volume().layout();
    let radiance = read_radiance(&renderer, gi.volume().texture());
    let scale = gi.volume().radiance_scale();
    let vs = layout.voxel_size;
    let [dx, dy, dz] = layout.dims;
    // the floor's top voxel layer
    let y = ((-0.49 - layout.origin[1]) / vs) as u32;
    let (mut lit, mut shadowed) = (Vec::new(), Vec::new());
    for z in 0..dz {
        for x in 0..dx {
            let p = glam::Vec3::from(layout.origin) + (glam::UVec3::new(x, y, z).as_vec3() + 0.5) * vs;
            let top = glam::Vec3::new(p.x, -0.49, p.z);
            let r = radiance[((z * dy + y) * dx + x) as usize];
            // under the block's middle (its shadow, the light being straight above), or in the
            // open on the right
            if (p.x + 0.4 * 1.39 / 0.7).abs() < 0.1 && p.z.abs() < 0.1 {
                shadowed.push(r[0] * scale);
            } else if p.x > 0.15 && p.x < 0.5 && p.z.abs() < 0.3 {
                // illuminance from the spot's falloff (spot_light_types.wgsl): I / d^2 times the
                // range window, the cone full this close to the axis, times N.L
                let d = light_pos - top;
                let d2 = d.length_squared();
                let window = (1.0 - (d2 / 400.0).powi(2)).clamp(0.0, 1.0);
                let e = 10.0 / d2 * window * window * (d.y / d2.sqrt());
                lit.push((r[0] * scale, 0.5 / std::f32::consts::PI * e, r[3]));
            }
        }
    }
    eprintln!("{} lit voxels, {} shadowed: {:?} ... {:?}", lit.len(), shadowed.len(), lit.first(), shadowed.first());
    assert!(lit.len() > 20 && shadowed.len() > 4);
    for (got, want, opacity) in &lit {
        assert!((got / want - 1.0).abs() < 0.08, "lit voxel leaves {got}, expected {want}");
        assert_eq!(*opacity, 1.0);
    }
    let brightest_shadowed = shadowed.iter().cloned().fold(0.0, f32::max);
    assert!(brightest_shadowed < 0.05 * lit[0].1, "a shadowed voxel leaves {brightest_shadowed}");
}

/// Through an empty volume every cone escapes to a uniform sky of radiance L, so a surface of
/// albedo a gains a * L (its irradiance is pi L): with the voxels alone, and with screen-space GI
/// in front (an open floor hides no direction from it).
#[test]
fn the_screen_cones_see_the_sky_through_an_empty_volume() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Low, bounds_min: [-2.0; 3], bounds_max: [2.0; 3], ..Default::default() });
    let albedo = [0.5, 0.25, 1.0];
    let mut scene = Scene::new();
    // a floor, not in the volume (no GI surface)
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), surface_material(albedo));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
    let sky = [2.0, 1.0, 0.5];
    for near_field in [None, Some(ScreenSpaceGIOptions::default())] {
        let mut effect = VoxelGIEffect::new(renderer.voxel_gi().unwrap().volume(), VoxelGIOptions { near_field, ..Default::default() });
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
        for _ in 0..4 {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        let row = (W * 8).next_multiple_of(256);
        let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            output.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
            output.size(),
        );
        renderer.queue().submit(Some(encoder.finish()));
        buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        renderer.device().poll(wgpu::Maintain::Wait);
        let bytes = buffer.slice(..).get_mapped_range();
        // the image's lower middle is floor
        let (mut sum, mut pixels) = ([0.0f32; 3], 0);
        for y in H * 2 / 3..H - 4 {
            for x in W / 4..W * 3 / 4 {
                let o = (y * row + x * 8) as usize;
                for (c, s) in sum.iter_mut().enumerate() {
                    *s += f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]));
                }
                pixels += 1;
            }
        }
        let got = sum.map(|s| s / pixels as f32);
        let want: [f32; 3] = std::array::from_fn(|c| (albedo[c] * 255.0).round() / 255.0 * sky[c]);
        eprintln!("near field {}: the floor gains {got:?}, expected {want:?}", near_field.is_some());
        for c in 0..3 {
            assert!((got[c] / want[c] - 1.0).abs() < 0.03, "near field {}: channel {c} gains {} of {}", near_field.is_some(), got[c], want[c]);
        }
    }
}
