//! Ray-traced shadows on a real GPU: a floor whose material leaves its direct light to
//! `RtShadowsEffect` (`kansei_gbuffer_out_rt_lit`) under a box and a slanted sun. Every floor pixel
//! whose ray toward the sun the CPU finds hitting the box is black in the visibility view, and
//! every one whose ray misses it is white; the lit view adds the sun's Lambert light to the lit
//! pixels and none to the shadowed ones; with the denoiser and contact rays on, the shadow is
//! still where the CPU puts it. A slab of glass a little above the floor casts none: the grid's
//! rays and the screen-space contact rays pass through it. Through the grid's cells and, where the adapter has ray queries,
//! through hardware ray queries (`RtBackend::Hardware`). Skipped (passes) when no adapter is
//! available.

use glam::{Vec3 as GVec3, Vec4Swizzles};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtGridOptions, RtShadowsEffect, RtShadowsView, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

const W: u32 = 160;
const H: u32 = 120;
const ALBEDO: f32 = 0.8;
/// The sun's illuminance (lux) and the way its light travels.
const SUN: f32 = 2.0;
const SUN_DIR: GVec3 = GVec3::new(0.4, -1.0, 1.2);

/// A headless renderer on the fast adapter of `WGPU_BACKEND`'s backend (the discrete GPU beside an
/// integrated one, with its ray queries where it has them), named on stderr.
fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let info = adapter.get_info();
    eprintln!("{} ({:?})", info.name, info.backend);
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// Clear glass (`kansei_gbuffer_out_glass`), which `RtShadowsEffect` leaves unlit.
fn glass_material() -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
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
    return kansei_gbuffer_out_glass(normalize(in.normal), vec3<f32>(0.9), 1.5, 0.0);
}}
"#
    );
    Material::new("Glass", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

/// A surface lit by `RtShadowsEffect` alone: no colour of its own, its albedo for the effect.
fn material() -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
{RT_SHADOWS_GBUFFER_WGSL}
struct Surface {{ albedo: vec4<f32> }};
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
    return kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), normalize(in.normal), surface.albedo.rgb, 1.0, 0.04, false);
}}
"#
    );
    let mut m = Material::new("RtLit", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "RtLit", &[ALBEDO, ALBEDO, ALBEDO, 1.0]);
    m
}

const EYE: GVec3 = GVec3::new(0.0, 2.2, 2.6);

fn camera() -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(EYE.x, EYE.y, EYE.z);
    camera.look_at(&Vec3::new(0.0, 0.0, -0.5));
    camera.update_projection_matrix();
    camera
}

const BOX_CENTRE: GVec3 = GVec3::new(-0.2, 0.55, -0.6);
const BOX_HALF: f32 = 0.25;
/// The glass: 13 cm above the floor, so the contact rays from under it meet it on screen.
const GLASS_CENTRE: GVec3 = GVec3::new(0.6, 0.15, 0.8);
const GLASS_HALF: GVec3 = GVec3::new(0.35, 0.02, 0.3);

/// Where the box is hit by a ray, if it is: the slab test against the box grown by `grow`.
fn hits_box(o: GVec3, d: GVec3, grow: f32) -> Option<f32> {
    hits(BOX_CENTRE, GVec3::splat(BOX_HALF + grow), o, d)
}

/// Where a box of `half` extents about `centre` is hit by a ray, if it is.
fn hits(centre: GVec3, half: GVec3, o: GVec3, d: GVec3) -> Option<f32> {
    let lo = centre - half;
    let hi = centre + half;
    let inv = d.recip();
    let (a, b) = ((lo - o) * inv, (hi - o) * inv);
    let (near, far) = (a.min(b).max_element(), a.max(b).min_element());
    (near <= far && far > 0.0).then_some(near.max(0.0))
}

/// What the floor's pixels see of the sun, as the CPU finds it: Some(false) where the ray toward
/// it hits the box (well inside it), Some(true) where it misses it (well clear), None near the
/// shadow's edge and off the floor. `margin` (m) is how far from the edge a pixel must be.
fn expected(camera: &Camera, margin: f32) -> Vec<Option<bool>> {
    let inv = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
    let to_sun = -SUN_DIR.normalize();
    let mut out = Vec::new();
    for y in 0..H {
        for x in 0..W {
            let ndc = glam::Vec4::new((x as f32 + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y as f32 + 0.5) / H as f32 * 2.0, 1.0, 1.0);
            let far = inv * ndc;
            let d = (far.xyz() / far.w - EYE).normalize();
            // the floor's top at y = 0, unless the box is in front of it
            let t = -EYE.y / d.y;
            let p = EYE + d * t;
            let glass = hits(GLASS_CENTRE, GLASS_HALF + 0.03, EYE, d).is_some_and(|b| b < t);
            if d.y >= 0.0 || p.x.abs() > 1.8 || p.z.abs() > 1.8 || glass || hits_box(EYE, d, 0.05).is_some_and(|b| b < t) {
                out.push(None);
                continue;
            }
            out.push(match (hits_box(p, to_sun, -margin), hits_box(p, to_sun, margin)) {
                (Some(_), _) => Some(false),
                (None, None) => Some(true),
                _ => None,
            });
        }
    }
    out
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
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let bytes = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// The floor pixels expected lit whose ray toward the sun passes through the glass within the
/// contact rays' reach.
fn under_glass(camera: &Camera, expect: &[Option<bool>]) -> Vec<usize> {
    let inv = (camera.projection_matrix.to_glam() * camera.view_matrix.to_glam()).inverse();
    let to_sun = -SUN_DIR.normalize();
    (0..(W * H) as usize)
        .filter(|&i| {
            if expect[i] != Some(true) {
                return false;
            }
            let (x, y) = ((i as u32 % W) as f32, (i as u32 / W) as f32);
            let ndc = glam::Vec4::new((x + 0.5) / W as f32 * 2.0 - 1.0, 1.0 - (y + 0.5) / H as f32 * 2.0, 1.0, 1.0);
            let far = inv * ndc;
            let d = (far.xyz() / far.w - EYE).normalize();
            let p = EYE + d * (-EYE.y / d.y);
            hits(GLASS_CENTRE, GLASS_HALF - 0.03, p, to_sun).is_some_and(|t| t < 0.24)
        })
        .collect()
}

/// Pixels the CPU expects lit and shadowed, and of each how many `test` passes.
fn tally(expect: &[Option<bool>], image: &[[f32; 4]], lit: impl Fn([f32; 4]) -> bool, dark: impl Fn([f32; 4]) -> bool) -> (u32, u32, u32, u32) {
    let (mut lit_n, mut lit_ok, mut dark_n, mut dark_ok) = (0, 0, 0, 0);
    for (e, c) in expect.iter().zip(image) {
        match e {
            Some(true) => {
                lit_n += 1;
                lit_ok += lit(*c) as u32;
            }
            Some(false) => {
                dark_n += 1;
                dark_ok += dark(*c) as u32;
            }
            None => {}
        }
    }
    (lit_n, lit_ok, dark_n, dark_ok)
}

fn shadows_match_the_cpu(backend: RtBackend) {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 4.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-2.0, -0.5, -2.0)), backend, ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(4.0, 0.2, 4.0), material()).with_rt(RtSurface::new([ALBEDO; 3]));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut blocker = Renderable::new(BoxGeometry::new(2.0 * BOX_HALF, 2.0 * BOX_HALF, 2.0 * BOX_HALF), material()).with_rt(RtSurface::new([ALBEDO; 3]));
    blocker.object.set_position(BOX_CENTRE.x, BOX_CENTRE.y, BOX_CENTRE.z);
    scene.add(SceneNode::Renderable(blocker));
    let mut glass = Renderable::new(BoxGeometry::new(2.0 * GLASS_HALF.x, 2.0 * GLASS_HALF.y, 2.0 * GLASS_HALF.z), glass_material()).with_rt(RtSurface::glass([0.9; 3]));
    glass.object.set_position(GLASS_CENTRE.x, GLASS_CENTRE.y, GLASS_CENTRE.z);
    scene.add(SceneNode::Renderable(glass));
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(SUN_DIR.x, SUN_DIR.y, SUN_DIR.z), Vec3::new(1.0, 1.0, 1.0), SUN))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
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
    eprintln!("asked for {backend:?}, traced with {:?}", handle.backend());
    let mut effect = RtShadowsEffect::new(handle);
    effect.update_lights(scene.lights());
    let mut run = |renderer: &mut Renderer, effect: &mut RtShadowsEffect, frames: usize| {
        for _ in 0..frames {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        read_rgba16(renderer, &output)
    };

    // exact: every pixel traced, a sun of a tenth of a degree, no wavelet, no contact rays
    effect.half_resolution = false;
    effect.iterations = 0;
    effect.contact_length = 0.0;
    effect.sun_angle = 0.002;
    effect.view = RtShadowsView::Visibility;
    let expect = expected(&self::camera(), 0.03);
    let vis = run(&mut renderer, &mut effect, 4);
    let (lit_n, lit_ok, dark_n, dark_ok) = tally(&expect, &vis, |c| c[0] > 0.99, |c| c[0] < 0.01);
    eprintln!("visibility: {lit_ok} of {lit_n} lit pixels white, {dark_ok} of {dark_n} shadowed ones black");
    assert!(lit_n > 2000 && dark_n > 200, "the view frames the shadow ({lit_n} lit, {dark_n} shadowed)");
    assert_eq!((lit_ok, dark_ok), (lit_n, dark_n), "every pixel's visibility is the CPU's");

    // the lit view: the sun's Lambert light (albedo / pi x E x cos) where it reaches, none in the
    // shadow; the material's own colour is black
    effect.view = RtShadowsView::Lit;
    let lit = run(&mut renderer, &mut effect, 2);
    let lambert = ALBEDO / std::f32::consts::PI * SUN * SUN_DIR.normalize().y.abs();
    // GGX at roughness 1 adds a little
    let (lit_n, lit_ok, dark_n, dark_ok) = tally(&expect, &lit, |c| c[0] > lambert * 0.98 && c[0] < lambert * 1.25, |c| c[0] < 1e-3);
    eprintln!("lit: {lit_ok} of {lit_n} lit pixels at the sun's light ({lambert:.3}), {dark_ok} of {dark_n} shadowed ones black");
    assert_eq!((lit_ok, dark_ok), (lit_n, dark_n), "the lit view adds the sun's light where it reaches and only there");

    // the defaults (half resolution, accumulated, filtered, a softer sun, contact rays): the shadow
    // still where the CPU puts it, away from its penumbra
    let defaults = RtShadowsEffect::new(renderer.rt_grid().unwrap().handle());
    (effect.half_resolution, effect.iterations, effect.contact_length, effect.sun_angle) = (defaults.half_resolution, defaults.iterations, defaults.contact_length, defaults.sun_angle);
    effect.view = RtShadowsView::Visibility;
    effect.reset_history();
    let expect = expected(&self::camera(), 0.12);
    let vis = run(&mut renderer, &mut effect, 24);
    let (lit_n, lit_ok, dark_n, dark_ok) = tally(&expect, &vis, |c| c[0] > 0.9, |c| c[0] < 0.1);
    eprintln!("defaults: {lit_ok} of {lit_n} lit pixels white, {dark_ok} of {dark_n} shadowed ones black");
    let under = under_glass(&self::camera(), &expect);
    let under_white = under.iter().filter(|&&i| vis[i][0] > 0.9).count();
    eprintln!("under the glass: {under_white} of {} white", under.len());
    assert!(under.len() > 100 && under_white == under.len(), "the glass casts no shadow");
    assert!(lit_ok as f32 >= 0.995 * lit_n as f32 && dark_ok as f32 >= 0.995 * dark_n as f32, "the denoised shadow is the CPU's");
}

#[test]
fn shadows_through_the_grid_match_the_cpu() {
    shadows_match_the_cpu(RtBackend::Software);
}

#[test]
fn shadows_through_hardware_ray_queries_match_the_cpu() {
    shadows_match_the_cpu(RtBackend::Hardware);
}
