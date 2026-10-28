//! Ray-traced caustics with the ray-traced shadows they correct, on a real GPU: a 3 cm glass marble
//! on a floor whose direct light `RtShadowsEffect` gives (its shadow rays pass through glass), under
//! a spot light with a 5 cm emitter. The light at a floor pixel is RtShadowsEffect's plus the
//! caustics' change there; what goes wrong is where the two disagree:
//!
//! - an opaque slab between the lamp and the marble, its edge swept across the emitter as the
//!   marble sees it: the change never takes away more light than RtShadowsEffect put there (the
//!   light before the composite's clamp stays above zero, but for photon noise);
//! - a jar beside the marble, its shadow across the marble's: where the jar keeps the light off the
//!   floor, the marble's shadow takes nothing (it merges into the jar's shadow, not stack on it),
//!   and its caustics add only the light the marble sends there (its reflection);
//! - a marble round an opaque core, rolling: RtShadowsEffect's history leaves no trail of the
//!   core's shadow behind it, and a still scene's shadows stay as steady as they were;
//! - a marble round an opaque core, resting: the light its shell focuses into the core's shadow
//!   (where RtShadowsEffect finds the lamp hidden) is all there, as without the visibility.
//!
//! Through the grid's cells and, where the adapter has ray queries, through hardware ray queries.
//! Skipped (passes) when no adapter is available. `KANSEI_CAUSTICS_PPM=<dir>` writes the light
//! before the clamp as pictures (red below zero).

use glam::Vec3 as GVec3;
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, MaterialOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtCausticsEffect, RtCausticsOptions, RtCausticsView, RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

const W: u32 = 320;
const H: u32 = 240;
const RADIUS: f32 = 0.015;
const ALBEDO: f32 = 0.8;
/// The lamp's intensity (cd) and its emitter's radius (m).
const CANDELA: f32 = 20.0;
const EMITTER: f32 = 0.05;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let info = adapter.get_info();
    eprintln!("{} ({:?})", info.name, info.backend);
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

const VERTEX: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
"#;

fn glass_material() -> Material {
    let code = format!("{GBUFFER_OUT_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_glass(normalize(in.normal), vec3<f32>(1.0), 1.5, 0.0); }}\n");
    Material::new("Glass", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

/// A surface lit by `RtShadowsEffect` alone, of albedo `ALBEDO`.
fn lit_material() -> Material {
    let code = format!(
        "{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), normalize(in.normal), vec3<f32>({ALBEDO:?}), 1.0, 0.04, false); }}\n"
    );
    Material::new("RtLit", &code, vec![], MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
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

/// An rgba16float texture's green channel.
fn read_green(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<f32> {
    let row = (W * 8).next_multiple_of(256);
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
        wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
    );
    renderer.queue().submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let bytes = buffer.slice(..).get_mapped_range().unwrap();
    (0..W * H).map(|i| (((i / W) * row + (i % W) * 8 + 2) as usize, i)).map(|(o, _)| f16_value(u16::from_le_bytes([bytes[o], bytes[o + 1]]))).collect()
}

/// A scene: the marble's centre at `height` (on the floor: `RADIUS`), with an opaque core of half
/// its radius or not; opaque boxes (centre, size); a spot light at `lamp` toward `aim`; the camera.
struct Rig {
    height: f32,
    core: bool,
    boxes: Vec<(GVec3, GVec3)>,
    lamp: GVec3,
    aim: GVec3,
    eye: GVec3,
    target: GVec3,
}

/// What the last frame shows: RtShadowsEffect's lit image, the caustics' change over a white
/// surface (radiance) and the light before the composite's clamp (lit + albedo x change; the
/// image the screen shows is that clamped at zero), green.
struct Shot {
    lit: Vec<f32>,
    change: Vec<f32>,
    pre: Vec<f32>,
    /// RtShadowsEffect's image's change from frame to frame over the last 8 frames, pixel by
    /// pixel (its mean over them, a share of its light)
    flicker: f32,
}

/// `rig` with the marble at x = `path[k]` in frame k (moving: `dynamic`), RtShadowsEffect's temporal
/// blend floor `alpha`; `fresh`: the caustics' tile cleared for the last frame (one frame of
/// photons, as a moving marble's tile holds); `weighted`: the caustics read RtShadowsEffect's
/// visibility.
fn shoot(backend: RtBackend, rig: &Rig, path: &[f32], alpha: f32, fresh: bool, weighted: bool) -> Option<Shot> {
    let mut renderer = renderer()?;
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 0.4 / 64.0, fixed_origin: Some(GVec3::new(-0.2, -0.05, -0.2)), backend, ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(0.4, 0.02, 0.4), lit_material()).with_rt(RtSurface::new([ALBEDO; 3]));
    floor.object.set_position(0.0, -0.01, 0.0);
    scene.add(SceneNode::Renderable(floor));
    for &(centre, size) in &rig.boxes {
        let mut b = Renderable::new(BoxGeometry::new(size.x, size.y, size.z), lit_material()).with_rt(RtSurface::new([ALBEDO; 3]));
        b.object.set_position(centre.x, centre.y, centre.z);
        scene.add(SceneNode::Renderable(b));
    }
    let mut ids = vec![scene.add(SceneNode::Renderable(Renderable::new(IcosphereGeometry::new(RADIUS, 5), glass_material()).with_rt(RtSurface::glass([1.0; 3]))))];
    if rig.core {
        ids.push(scene.add(SceneNode::Renderable(Renderable::new(IcosphereGeometry::new(0.5 * RADIUS, 3), lit_material()).with_rt(RtSurface::new([0.8, 0.3, 0.1])))));
    }
    let d = (rig.aim - rig.lamp).normalize();
    let mut spot = SpotLight::new(Vec3::new(rig.lamp.x, rig.lamp.y, rig.lamp.z), Vec3::new(d.x, d.y, d.z), Vec3::new(1.0, 1.0, 1.0), CANDELA, 100.0, 50f32.to_radians(), 60f32.to_radians());
    spot.source_radius = EMITTER;
    scene.add(SceneNode::Light(Light::Spot(spot)));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.01, 10.0, W as f32 / H as f32);
    camera.set_position(rig.eye.x, rig.eye.y, rig.eye.z);
    camera.look_at(&Vec3::new(rig.target.x, rig.target.y, rig.target.z));
    camera.update_projection_matrix();
    let texture = |usage| {
        renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: usage | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let (lit_tex, out_tex) = (texture(wgpu::TextureUsages::TEXTURE_BINDING), texture(wgpu::TextureUsages::empty()));
    let (lit, output) = (lit_tex.create_view(&Default::default()), out_tex.create_view(&Default::default()));
    let handle = renderer.rt_grid().unwrap().handle();
    eprintln!("traced with {:?}", handle.backend());
    let mut shadows = RtShadowsEffect::new(handle.clone());
    shadows.temporal_alpha = alpha;
    shadows.update_lights(scene.lights());
    let mut caustics = RtCausticsEffect::new(renderer.device(), handle, RtCausticsOptions { texel: 0.0005, reach: rig.height, ..Default::default() });
    caustics.set_glass(1.5, [1.0; 3]);
    caustics.update_lights(scene.lights());
    caustics.set_visibility(weighted.then(|| shadows.visibility()));
    let mut last = Vec::new();
    for (k, &x) in path.iter().enumerate() {
        let moved = k > 0 && path[k - 1] != x;
        for &i in &ids {
            let r = scene.get_renderable_mut(i).unwrap();
            r.object.set_position(x, rig.height, 0.0);
            r.dynamic = moved;
        }
        caustics.set_casters(&[(GVec3::new(x - RADIUS, rig.height - RADIUS, -RADIUS), GVec3::new(x + RADIUS, rig.height + RADIUS, RADIUS))]);
        // the last frame shows the change, over 40 (unclamped below it)
        caustics.view = if k + 1 == path.len() { RtCausticsView::Caustics } else { RtCausticsView::Lit };
        if fresh && k + 1 == path.len() {
            caustics.clear();
        }
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        shadows.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &lit, &camera, W, H);
        caustics.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &lit, &gbuffer.depth_view, &output, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
        if k + 9 >= path.len() && path[path.len() - 9..].iter().all(|&p| p == x) {
            last.push(read_green(&renderer, &lit_tex));
        }
        // a moving marble's frames as the screen shows them (KANSEI_CAUSTICS_PPM)
        if moved && k + 1 < path.len() && std::env::var("KANSEI_CAUSTICS_PPM").is_ok() {
            let top = read_green(&renderer, &lit_tex).iter().copied().fold(0.0, f32::max);
            picture(&format!("rolling-frame-{k:02}"), &read_green(&renderer, &out_tex), top);
        }
    }
    let flicker = last.windows(2).map(|w| w[0].iter().zip(&w[1]).map(|(a, b)| (a - b).abs()).sum::<f32>() / w[1].iter().sum::<f32>().max(1e-6)).sum::<f32>() / last.len().saturating_sub(1).max(1) as f32;
    if let Some((min, texel, dims, history)) = caustics.tile(0) {
        eprintln!("the tile: from {min:?}, {dims:?} texels of {:.2} mm, {history} frames", texel * 1e3);
    }
    let lit = read_green(&renderer, &lit_tex);
    let change: Vec<f32> = read_green(&renderer, &out_tex).iter().map(|v| v - 40.0).collect();
    let pre = lit.iter().zip(&change).map(|(l, c)| l + ALBEDO * c).collect();
    Some(Shot { lit, change, pre, flicker })
}

/// `KANSEI_CAUSTICS_PPM`: the light before the clamp as a picture named `name`, grey by `top`, red
/// below zero.
fn picture(name: &str, pre: &[f32], top: f32) {
    let Ok(dir) = std::env::var("KANSEI_CAUSTICS_PPM") else { return };
    let mut bytes = format!("P6 {W} {H} 255\n").into_bytes();
    for v in pre {
        let g = ((v / top.max(1e-6)).clamp(0.0, 1.0).powf(1.0 / 2.2) * 255.0) as u8;
        bytes.extend_from_slice(&if *v < -0.05 { [255, 0, 0] } else { [g, g, g] });
    }
    std::fs::write(std::path::Path::new(&dir).join(format!("{name}.ppm")), bytes).unwrap();
}

/// The light missing below zero over the lit light, and the pixels below zero (past `noise`).
fn below_zero(shot: &Shot, noise: f32) -> (f32, usize) {
    let missing: f32 = shot.pre.iter().map(|v| (-v).max(0.0)).sum();
    (missing / shot.lit.iter().sum::<f32>().max(1e-6), shot.pre.iter().filter(|v| **v < -noise).count())
}

/// The lamp 0.5 m over the marble, a slab 4 mm thick at 0.1 m covering x < `edge`.
fn slab_rig(edge: f32) -> Rig {
    Rig {
        height: RADIUS,
        core: false,
        boxes: vec![(GVec3::new(edge - 0.2, 0.1, 0.0), GVec3::new(0.4, 0.004, 0.4))],
        lamp: GVec3::new(0.0, 0.5, 0.0),
        aim: GVec3::ZERO,
        eye: GVec3::new(0.0, 0.08, 0.08),
        target: GVec3::ZERO,
    }
}

/// The slab's edge swept across the emitter as the marble sees it (from wholly hidden to wholly
/// seen): the light before the clamp falls below zero by no more than photon noise.
fn penumbra(backend: RtBackend) {
    // the edge's reach at the slab: the emitter's radius scaled from the lamp to the marble
    let reach = EMITTER * (0.1 - RADIUS) / (0.5 - RADIUS) * 1.2;
    let mut worst = (0.0f32, 0usize);
    for k in 0..=8 {
        let edge = -reach + 2.0 * reach * k as f32 / 8.0;
        let Some(shot) = shoot(backend, &slab_rig(edge), &[0.0; 32], 0.12, false, true) else { return eprintln!("no GPU adapter: skipping") };
        let (missing, negative) = below_zero(&shot, 1.0);
        let least = shot.pre.iter().copied().fold(f32::MAX, f32::min);
        eprintln!("slab edge {:+5.1} mm: least {least:+6.2}, {negative} pixels below -1, missing {missing:.5} of the lit light", edge * 1e3);
        picture(&format!("penumbra-{k}"), &shot.pre, shot.lit.iter().copied().fold(0.0, f32::max));
        worst = (worst.0.max(missing), worst.1.max(negative));
    }
    assert!(worst.0 < 2e-4, "the caustics take away light that isn't there ({:.5} of the lit light missing below zero)", worst.0);
    assert!(worst.1 < 40, "{} pixels below zero", worst.1);
}

#[test]
fn penumbra_light_stays_above_zero_through_the_grid() {
    penumbra(RtBackend::Software);
}

#[test]
fn penumbra_light_stays_above_zero_through_hardware_ray_queries() {
    penumbra(RtBackend::Hardware);
}

/// The lamp low to the left, a jar (a box) beside the marble whose shadow falls across the
/// marble's: the share of the marble's change taking and adding light where the jar keeps the
/// light off the floor (RtShadowsEffect's light there under 2% of the floor's brightest), and the
/// light below zero.
fn beside_a_jar(backend: RtBackend) {
    let rig = Rig {
        height: RADIUS,
        core: false,
        boxes: vec![(GVec3::new(-0.05, 0.04, 0.03), GVec3::new(0.04, 0.08, 0.04))],
        lamp: GVec3::new(-0.3, 0.4, 0.0),
        aim: GVec3::ZERO,
        eye: GVec3::new(0.03, 0.12, 0.09),
        target: GVec3::new(0.01, 0.0, 0.0),
    };
    let Some(shot) = shoot(backend, &rig, &[0.0; 32], 0.12, false, true) else { return eprintln!("no GPU adapter: skipping") };
    let top = shot.lit.iter().copied().fold(0.0, f32::max);
    let dark: Vec<bool> = shot.lit.iter().map(|v| *v < 0.02 * top).collect();
    let all: f32 = shot.change.iter().map(|c| c.abs()).sum();
    let in_shadow = |f: fn(f32) -> f32| shot.change.iter().zip(&dark).filter(|(_, d)| **d).map(|(c, _)| f(*c)).sum::<f32>();
    let (taken, added) = (in_shadow(|c| (-c).max(0.0)), in_shadow(|c| c.max(0.0)));
    let (missing, negative) = below_zero(&shot, 1.0);
    eprintln!(
        "beside a jar: of the marble's change, {:.4} taken and {:.4} added in the jar's shadow ({} dark pixels); missing {missing:.5} below zero, {negative} pixels; the shadows' frame against the next {:.5}",
        taken / all.max(1e-6),
        added / all.max(1e-6),
        dark.iter().filter(|d| **d).count(),
        shot.flicker
    );
    picture("jar", &shot.pre, top);
    assert!(all > 0.0, "the marble changes the light");
    assert!(taken < 0.005 * all, "the marble's shadow stacks on the jar's shadow ({:.4} of its change there)", taken / all);
    // the light added there is the marble's own, reflected off its side into the jar's shadow
    // beside it (0.0095 of its change; 0.0004 with the first surface's reflections left out)
    assert!(added < 0.02 * all, "the marble's caustics spill into the jar's shadow ({:.4} of its change there)", added / all);
    assert!(missing < 2e-4 && negative < 40, "and take away light that isn't there ({missing:.5}, {negative} pixels)");
    // RtShadowsEffect's history held near its rays: the jar's wide penumbra as steady as with the
    // old history (0.0045 of the light from frame to frame, pixel by pixel)
    assert!(shot.flicker < 0.0049, "the jar's penumbra flickers ({:.5} of the light from frame to frame)", shot.flicker);
}

#[test]
fn a_marbles_change_merges_into_another_shadow_through_the_grid() {
    beside_a_jar(RtBackend::Software);
}

#[test]
fn a_marbles_change_merges_into_another_shadow_through_hardware_ray_queries() {
    beside_a_jar(RtBackend::Hardware);
}

/// The marble round its core, the lamp behind it so its shadow falls toward the camera.
fn core_rig() -> Rig {
    Rig {
        height: RADIUS,
        core: true,
        boxes: Vec::new(),
        lamp: GVec3::new(0.0, 0.5, -0.25),
        aim: GVec3::new(0.0, 0.0, 0.0),
        eye: GVec3::new(0.0, 0.12, 0.06),
        target: GVec3::new(0.0, 0.0, 0.01),
    }
}

/// The marble rolling 4 mm a frame for 12 frames to x = 0, against one that rested where it
/// started and then at x = 0 (its tile placed as a rolling one's and fed one frame at the end, its
/// pattern a moving marble's), the images compared after a 9 x 9 box (which keeps a trail and
/// averages the noise out): where the rolling marble's image is darker (a shadow left behind it,
/// RtShadowsEffect's or the caustics') and where it is lighter (the shadow's soft front, still
/// settling), each a share of the light, a few times a resting frame's change at most (the old
/// history: 0.0021 and 0.0021; with the caustics 0.0015 and 0.0021). And a still scene's shadows
/// change from frame to frame no more than with the old history (0.00007 of the light, pixel by
/// pixel).
fn rolling(backend: RtBackend) {
    let rig = core_rig();
    let Some(steady) = shoot(backend, &rig, &[0.0; 44], 0.12, false, true) else { return eprintln!("no GPU adapter: skipping") };
    let rest = |frames: usize| {
        let mut path = vec![-0.048; 32];
        path.extend(std::iter::repeat_n(0.0, frames));
        shoot(backend, &rig, &path, 0.12, true, true)
    };
    let (still, other) = (rest(12).unwrap(), rest(13).unwrap());
    let mut path = vec![-0.048; 32];
    path.extend((1..=12).map(|k| -0.048 + 0.004 * k as f32));
    let moving = shoot(backend, &rig, &path, 0.12, false, true).unwrap();
    let base: f32 = still.lit.iter().sum();
    let blur = |img: &[f32]| -> Vec<f32> {
        let at = |x: i32, y: i32| img[(y.clamp(0, H as i32 - 1) as u32 * W + x.clamp(0, W as i32 - 1) as u32) as usize];
        (0..(W * H) as i32).map(|i| (-4..=4).flat_map(|dy| (-4..=4).map(move |dx| (dx, dy))).map(|(dx, dy)| at(i % W as i32 + dx, i / W as i32 + dy)).sum::<f32>() / 81.0).collect()
    };
    // (darker, lighter) than `b`, over `b`'s light
    let apart = |a: &[f32], b: &[f32]| {
        let (a, b) = (blur(a), blur(b));
        let darker = a.iter().zip(&b).map(|(x, y)| (y - x).max(0.0)).sum::<f32>() / base;
        let lighter = a.iter().zip(&b).map(|(x, y)| (x - y).max(0.0)).sum::<f32>() / base;
        (darker, lighter)
    };
    let shown = |s: &Shot| s.pre.iter().map(|v| v.max(0.0)).collect::<Vec<f32>>();
    let noise = apart(&other.lit, &still.lit);
    let noise = noise.0.max(noise.1);
    let (trail, front) = apart(&moving.lit, &still.lit);
    let (shown_trail, shown_front) = apart(&shown(&moving), &shown(&still));
    eprintln!(
        "rolling to rest against resting: RtShadows' image darker by {trail:.4} of its light, lighter by {front:.4}; with the caustics {shown_trail:.4}, {shown_front:.4}; a resting frame against the next {noise:.4}; a still scene's frame against the next, pixel by pixel, {:.5}",
        steady.flicker
    );
    let top = still.lit.iter().copied().fold(0.0, f32::max);
    picture("rolling-still", &still.pre, top);
    picture("rolling-moving", &moving.pre, top);
    picture("rolling-still-shadows", &still.lit, top);
    picture("rolling-moving-shadows", &moving.lit, top);
    assert!(trail < 0.0008 && shown_trail < 0.0008, "the core's shadow trails behind the rolling marble ({trail:.4} of the light, with the caustics {shown_trail:.4}, against {noise:.4} resting)");
    assert!(front < 0.0012 && shown_front < 0.0012, "the shadow's front lags ({front:.4}, with the caustics {shown_front:.4})");
    assert!(steady.flicker < 0.0001, "a still scene's shadows flicker ({:.5} of the light from frame to frame)", steady.flicker);
}

#[test]
fn a_rolling_marble_leaves_no_trail_through_the_grid() {
    rolling(RtBackend::Software);
}

#[test]
fn a_rolling_marble_leaves_no_trail_through_hardware_ray_queries() {
    rolling(RtBackend::Hardware);
}

/// The marble round its core at rest, with and without RtShadowsEffect's visibility: the light the
/// shell focuses lands in the core's shadow, where the straight ray to the lamp is blocked, so the
/// visibility there is about zero. Weighed by it (the light added and taken alike) the focus would
/// all but go (#262: the light added in the core's shadow 0.014 of the unweighted); only the light taken
/// away is weighed, so it stays: the light added there, the brightest pixel, and the light below
/// zero as without the visibility.
fn focus(backend: RtBackend) {
    let rig = core_rig();
    let Some(weighted) = shoot(backend, &rig, &[0.0; 32], 0.12, false, true) else { return eprintln!("no GPU adapter: skipping") };
    let unweighted = shoot(backend, &rig, &[0.0; 32], 0.12, false, false).unwrap();
    // the core's shadow on the floor: RtShadowsEffect's light under a tenth of the floor's
    let top = unweighted.lit.iter().copied().fold(0.0, f32::max);
    let shade: Vec<bool> = unweighted.lit.iter().map(|v| *v < 0.1 * top).collect();
    let gain = |s: &Shot| s.change.iter().zip(&shade).filter(|(_, d)| **d).map(|(c, _)| c.max(0.0)).sum::<f32>();
    let brightest = |s: &Shot| s.pre.iter().copied().fold(0.0, f32::max);
    let (g, u) = (gain(&weighted), gain(&unweighted));
    let (b, bu) = (brightest(&weighted), brightest(&unweighted));
    let (missing, negative) = below_zero(&weighted, 1.0);
    eprintln!(
        "the focus in the core's shadow ({} pixels): the light added {g:.0} weighted, {u:.0} unweighted ({:.3}); the brightest pixel {b:.1} and {bu:.1} (the lit floor's {top:.1}); missing {missing:.5} below zero, {negative} pixels",
        shade.iter().filter(|d| **d).count(),
        g / u.max(1e-6)
    );
    picture("focus-weighted", &weighted.pre, bu);
    picture("focus-unweighted", &unweighted.pre, bu);
    assert!(u > 0.0 && bu > 1.5 * top, "the shell focuses light into the core's shadow ({u:.0}, the brightest {bu:.1} over {top:.1})");
    assert!(g > 0.85 * u, "the visibility takes the focus away ({g:.0} of {u:.0})");
    assert!(b > 0.85 * bu, "and its brightest spot ({b:.1} of {bu:.1})");
    assert!(missing < 2e-4 && negative < 40, "and takes away light that isn't there ({missing:.5}, {negative} pixels)");
}

#[test]
fn the_focus_round_an_opaque_core_stays_through_the_grid() {
    focus(RtBackend::Software);
}

#[test]
fn the_focus_round_an_opaque_core_stays_through_hardware_ray_queries() {
    focus(RtBackend::Hardware);
}
