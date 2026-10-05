//! Ray-traced reflections on a real GPU: a mirror-like floor (its material's F0 through
//! `kansei_gbuffer_out_specular`) under a box that glows red in the voxel GI. In the mirror view,
//! every floor pixel whose reflected ray the CPU finds hitting the box is red, and every one whose
//! ray misses it is not: the GBuffer's F0, the reflected rays, the grid's hits lit by the voxels,
//! the upsampling and the accumulation together. The lit view leaves surfaces with no F0 as they
//! were, and the voxel cone alone smears the box. Skipped (passes) when no adapter is available.

use glam::{Vec3 as GVec3, Vec4Swizzles};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtReflectionsEffect, RtReflectionsOptions, RtReflectionsView, RtSurface, SceneRtGridOptions};

const W: u32 = 160;
const H: u32 = 120;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// An unlit surface: its colour, normal, albedo, and an F0 and roughness for the reflections.
fn material(color: [f32; 3], f0: f32) -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
struct Surface {{ color: vec4<f32> }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut {{ @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> }};
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {{
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    return kansei_gbuffer_out_specular(surface.color.rgb, vec3<f32>(0.0), normalize(in.normal), surface.color.rgb, surface.color.w, 0.0);
}}
"#
    );
    let mut m = Material::new("Surface", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Surface", &[color[0], color[1], color[2], f0]);
    m
}

fn camera() -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 1.6, 3.0);
    camera.look_at(&Vec3::new(0.0, 0.0, -0.4));
    camera.update_projection_matrix();
    camera
}

const BOX_CENTRE: GVec3 = GVec3::new(0.0, 0.45, -0.7);
const BOX_HALF: f32 = 0.3;

/// Where the box is hit by a ray, if it is: the slab test against the box grown by `grow`.
fn hits_box(o: GVec3, d: GVec3, grow: f32) -> Option<f32> {
    let lo = BOX_CENTRE - GVec3::splat(BOX_HALF + grow);
    let hi = BOX_CENTRE + GVec3::splat(BOX_HALF + grow);
    let inv = d.recip();
    let (a, b) = ((lo - o) * inv, (hi - o) * inv);
    let (near, far) = (a.min(b).max_element(), a.max(b).min_element());
    (near <= far && far > 0.0).then_some(near.max(0.0))
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

fn read_rgba16(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let row = (W * 8).next_multiple_of(256);
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
        texture.size(),
    );
    renderer.queue().submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    renderer.device().poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// What the floor's pixels reflect, as the CPU finds it: Some(true) where the reflected ray hits
/// the box (well inside it), Some(false) where it misses it (well clear), None near its edges and
/// off the floor.
fn expected(camera: &Camera) -> Vec<Option<bool>> {
    let inv = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
    let eye = GVec3::new(0.0, 1.6, 3.0);
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let ndc = glam::Vec4::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0, 1.0, 1.0);
            let far = inv * ndc;
            let d = (far.xyz() / far.w - eye).normalize();
            // the floor's top at y = 0, unless the box is in front of it
            let t = -eye.y / d.y;
            let p = eye + d * t;
            if d.y >= 0.0 || p.x.abs() > 2.5 || p.z.abs() > 2.5 || hits_box(eye, d, 0.05).is_some_and(|b| b < t) {
                out.push(None);
                continue;
            }
            let r = GVec3::new(d.x, -d.y, d.z);
            out.push(match (hits_box(p, r, -0.06), hits_box(p, r, 0.06)) {
                (Some(_), _) => Some(true),
                (None, None) => Some(false),
                _ => None,
            });
        }
    }
    out
}

#[test]
fn a_mirror_floor_reflects_the_box_where_its_rays_hit_it() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-2.0, -0.5, -2.0], bounds_max: [2.0, 1.5, 2.0], ..Default::default() });
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 4.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-2.0, -0.5, -2.0)), ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    // a dark mirror-like floor, in the grid but not in the voxels; a box glowing red in the voxels
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), material([0.02, 0.02, 0.02], 1.0)).with_rt(RtSurface::new([0.02; 3]));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut glow = Renderable::new(BoxGeometry::new(2.0 * BOX_HALF, 2.0 * BOX_HALF, 2.0 * BOX_HALF), material([1.0, 0.1, 0.1], 0.0))
        .with_gi(GiSurface::new([0.8, 0.1, 0.1]).with_emission([3.0, 0.1, 0.1]))
        .with_rt(RtSurface::new([0.8, 0.1, 0.1]));
    glow.object.set_position(BOX_CENTRE.x, BOX_CENTRE.y, BOX_CENTRE.z);
    scene.add(SceneNode::Renderable(glow));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let expect = expected(&camera());
    let mut camera = camera();
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
    let handle = renderer.rt_grid().unwrap().handle();
    let mut effect = RtReflectionsEffect::with_volume(renderer.voxel_gi().unwrap().volume(), handle, RtReflectionsOptions { max_distance: 10.0, ..Default::default() });
    let mut run = |renderer: &mut Renderer, effect: &mut RtReflectionsEffect, frames: usize| {
        for _ in 0..frames {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        read_rgba16(renderer, &output)
    };
    let red = |c: [f32; 4]| c[0] > 0.3 && c[0] > 4.0 * c[1].max(c[2]);
    // the mirror image, once the voxels hold the glow and every pixel of each 2 x 2 was traced
    effect.view = RtReflectionsView::Mirror;
    let mirror = run(&mut renderer, &mut effect, 12);
    let (mut hit, mut hit_red, mut miss, mut miss_red) = (0, 0, 0, 0);
    for (e, c) in expect.iter().zip(&mirror) {
        match e {
            Some(true) => {
                hit += 1;
                hit_red += red(*c) as u32;
            }
            Some(false) => {
                miss += 1;
                miss_red += red(*c) as u32;
            }
            None => {}
        }
    }
    eprintln!("floor pixels reflecting the box: {hit_red} of {hit} red; missing it: {miss_red} of {miss} red");
    assert!(hit > 200 && miss > 2000, "the view frames the reflection ({hit} hit, {miss} miss)");
    assert_eq!(hit_red, hit, "every pixel whose ray hits the box reflects it");
    assert_eq!(miss_red, 0, "no pixel whose ray misses the box reflects it");
    // the lit view: the box (no F0) as the GBuffer had it
    effect.view = RtReflectionsView::Lit;
    let lit = run(&mut renderer, &mut effect, 2);
    let colour = read_rgba16(&renderer, &gbuffer.color_texture);
    let (cx, cy) = {
        let c = self::camera();
        let clip = c.projection_matrix.to_glam() * c.view_matrix.to_glam() * BOX_CENTRE.extend(1.0);
        (((clip.x / clip.w * 0.5 + 0.5) * W as f32) as usize, ((0.5 - clip.y / clip.w * 0.5) * H as f32) as usize)
    };
    let i = cy * W as usize + cx;
    assert!(red(colour[i]) && (0..3).all(|c| (lit[i][c] - colour[i][c]).abs() < 1e-3), "the box unchanged: {:?} vs {:?}", lit[i], colour[i]);
    // the voxel cone alone: the box's reflection smeared past where the rays hit it
    effect.view = RtReflectionsView::Mirror;
    effect.trace_grid = false;
    effect.reset_history();
    let cone = run(&mut renderer, &mut effect, 12);
    let smeared = expect.iter().zip(&cone).filter(|(e, c)| **e == Some(false) && red(**c)).count();
    eprintln!("voxel cone alone: {smeared} of {miss} missing pixels red");
    assert!(smeared > 0, "the voxel cone blurs the box past its edges");
}
