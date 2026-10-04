//! Temporal anti-aliasing: thin trunks and power lines (edge aliasing), a meadow of
//! alpha-tested grass cards swaying in the wind, and a car crossing the frame, under a moving
//! camera. The grass and the car write motion vectors (`MaterialOptions::outputs_velocity` with
//! `cameras::MOTION_VECTORS_WGSL`); everything else is reprojected by depth.
//!
//! URL parameters: `taa=0` (off), `vel=0` (no motion vectors: the car and grass reproject by
//! depth only), `wind=<scale>`, `t=<seconds>` (freeze the camera; the car and the wind keep
//! moving, the jitter keeps accumulating), `scale=<0.25..1>` (render the scene at that fraction
//! of the canvas and let the TAA upscale it), `stats=1` (log the interval between frames).
//!
//! Motion blur: `mblur=<amount>` (e.g. 0.5, a 180-degree shutter) adds a `MotionBlurEffect`
//! after the TAA, scaled to 30 fps and capped at 4 % of the width as the Midsommar intro's
//! (`mbfps=<fps>`, 0 for per-frame; `mbmax=<fraction>`). `pan=<deg/s>` swings the camera,
//! `car=<m/s>` sets the car's speed (9), and `step=<seconds>` with `t` alternates the camera and
//! the car between t and t + step every frame: a still picture that is moving, for comparing
//! the blur on and off.

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::{Camera, MOTION_VECTORS_WGSL};
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, MotionBlurEffect, MotionBlurOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, param, param_or, Canvas};

/// A diffuse surface under a sky and a low sun, writing motion vectors. Variants (string
/// replaced): INSTANCE_* for instanced trunks/grass (vec4: xyz position, w yaw or height), WIND
/// for grass cards bending with height, ALPHA_TEST for the cards' blade cut-out.
const SURFACE_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, params: vec4<f32> };  // params: time, previous time, wind
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, INSTANCE_INPUT };
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) curr: vec4<f32>,
    @location(3) prev: vec4<f32>,
};
struct FOut { @location(0) color: vec4<f32>, @location(4) velocity: vec2<f32> };

// the card at time t: instance placement, then wind bending the top
fn place(v: VIn, t: f32) -> vec3<f32> {
    var local = v.position.xyz;
    var offset = vec3<f32>(0.0);
    INSTANCE_PLACE
    WIND
    return local + offset;
}

@vertex
fn vertex_main(v: VIn) -> VOut {
    let p = place(v, surface.params.x);
    let prev_p = place(v, surface.params.y);
    let world = mesh.world * vec4<f32>(p, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv = v.uv;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4<f32>(prev_p, 1.0));
    return out;
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> FOut {
    ALPHA_TEST
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    let sun = normalize(vec3<f32>(-0.5, 0.35, -0.8));
    let sky = mix(vec3<f32>(60.0, 55.0, 45.0), vec3<f32>(900.0, 1100.0, 1500.0), n.y * 0.5 + 0.5);
    let lit = surface.base_color.rgb * (sky + vec3<f32>(9000.0, 7600.0, 6000.0) * max(dot(n, sun), 0.0));
    return FOut(vec4<f32>(lit, 1.0), kansei_motion_vector(in.curr, in.prev));
}
"#;

const SKY_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let up = saturate(normalize(in.dir).y);
    return vec4<f32>(mix(vec3<f32>(2600.0, 2800.0, 3100.0), vec3<f32>(900.0, 1300.0, 2400.0), sqrt(up)), 1.0);
}
"#;

#[derive(Clone, Copy, PartialEq)]
enum Kind {
    Plain,
    Trunks,
    Grass,
}

fn surface_material(label: &str, base: [f32; 3], kind: Kind, velocity: bool) -> Material {
    let mut shader = SURFACE_WGSL.to_string();
    let (input, place, wind, alpha) = match kind {
        Kind::Plain => ("", "", "", ""),
        // vec4: position, height
        Kind::Trunks => ("@location(3) instance: vec4<f32>,", "local.y = (local.y + 0.5) * v.instance.w; offset = v.instance.xyz;", "", ""),
        // vec4: position, yaw; a 0.5 x 0.7 card standing on the ground
        Kind::Grass => (
            "@location(3) instance: vec4<f32>,",
            "let c = cos(v.instance.w); let s = sin(v.instance.w); \
             let up = local.y + 0.35; local = vec3<f32>(c * local.x, up, -s * local.x); offset = v.instance.xyz;",
            "let h = saturate(1.0 - v.uv.y); \
             let gust = sin(t * 1.7 + v.instance.x * 0.9 + v.instance.z * 0.6) + 0.4 * sin(t * 4.1 + v.instance.x * 2.3); \
             local.x += gust * 0.12 * h * h * surface.params.z; local.z += gust * 0.05 * h * h * surface.params.z;",
            // five tapering blades per card
            "let bx = fract(in.uv.x * 5.0) - 0.5; let width = 0.45 * (in.uv.y * 0.9 + 0.1); \
             if (abs(bx) > width) { discard; }",
        ),
    };
    shader = shader.replace("INSTANCE_INPUT", input).replace("INSTANCE_PLACE", place).replace("WIND", wind).replace("ALPHA_TEST", alpha);
    let mut m = Material::new(
        label,
        &format!("{MOTION_VECTORS_WGSL}\n{shader}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions {
            cull_mode: if kind == Kind::Grass { CullMode::None } else { CullMode::Back },
            outputs_velocity: velocity,
            ..Default::default()
        },
    );
    m.set_uniform_bindable(0, label, &[base[0], base[1], base[2], 1.0, 0.0, 0.0, 0.0, 0.0f32]);
    m
}

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    if let Some(scale) = param("scale").and_then(|v| v.trim().parse().ok()) {
        renderer.set_render_scale(scale);
    }
    let velocity = flag("vel", true);

    let mut scene = Scene::new();
    let mut animated = Vec::new();
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    scene.add(SceneNode::Renderable(Renderable::new(kansei_core::geometries::SphereGeometry::new(500.0, 32, 16), sky)));

    let mut ground = Renderable::new(PlaneGeometry::new(400.0, 400.0), surface_material("Ground", [0.08, 0.1, 0.05], Kind::Plain, velocity));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // thin trunks, and power lines: sub-pixel edges that crawl without AA
    let mut trunks = Vec::new();
    for i in 0..400u32 {
        let x = -60.0 + hash(i) * 120.0;
        let z = -12.0 - hash(i + 3) * 90.0;
        trunks.extend_from_slice(&[x, 0.0, z, 8.0 + hash(i + 9) * 10.0]);
    }
    let n = trunks.len() as u32 / 4;
    let trunk_buf = ComputeBuffer::from_slice("Trunks", BufferType::Storage, BufferUsage::VERTEX, &trunks).with_vertex_vec4(3);
    scene.add(SceneNode::Renderable(Renderable::new(
        InstancedGeometry::new(BoxGeometry::new(0.18, 1.0, 0.18), n, vec![trunk_buf]),
        surface_material("Trunks", [0.12, 0.1, 0.08], Kind::Trunks, velocity),
    )));
    for k in 0..3 {
        let mut wire = Renderable::new(BoxGeometry::new(200.0, 0.03, 0.03), surface_material("Wire", [0.02, 0.02, 0.02], Kind::Plain, velocity));
        wire.object.set_position(0.0, 7.5 + k as f32 * 0.6, -14.0);
        scene.add(SceneNode::Renderable(wire));
    }

    // the meadow: 12 000 alpha-tested grass cards in the wind
    let mut grass = Vec::new();
    for i in 0..12000u32 {
        let x = -25.0 + hash(i + 101) * 50.0;
        let z = 2.0 - hash(i + 211) * 22.0;
        grass.extend_from_slice(&[x, 0.0, z, hash(i + 307) * std::f32::consts::TAU]);
    }
    let grass_buf = ComputeBuffer::from_slice("Grass", BufferType::Storage, BufferUsage::VERTEX, &grass).with_vertex_vec4(3);
    let meadow = scene.add(SceneNode::Renderable(Renderable::new(
        InstancedGeometry::new(PlaneGeometry::new(0.5, 0.7), grass.len() as u32 / 4, vec![grass_buf]),
        surface_material("Grass", [0.1, 0.16, 0.04], Kind::Grass, velocity),
    )));
    animated.push(meadow);

    // a car crossing the frame
    let car = scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(4.4, 1.3, 1.8), surface_material("Car", [0.3, 0.02, 0.02], Kind::Plain, velocity))));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(11.0);
        ToneMapEffect::new(options)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if flag("taa", true) {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    let mblur: f32 = param_or("mblur", 0.0);
    if mblur > 0.0 {
        let fps: f32 = param_or("mbfps", 30.0);
        effects.push(Box::new(MotionBlurEffect::new(MotionBlurOptions {
            amount: mblur,
            max: param_or("mbmax", 0.04),
            target_fps: (fps > 0.0).then_some(fps),
            ..Default::default()
        })));
    }
    effects.push(Box::new(tonemap));
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(40.0, 0.1, 1000.0, canvas.aspect());
    camera.update_projection_matrix();

    let (render_width, render_height) = renderer.render_size();
    let (width, height) = canvas.size();
    log::info!("Kansei — Temporal AA (WASM) ready: motion vectors {velocity}, motion blur {mblur}, rendering {render_width}x{render_height} for {width}x{height}");

    let frozen_t: Option<f32> = param("t").and_then(|v| v.trim().parse().ok());
    let step: Option<f32> = param("step").and_then(|v| v.trim().parse().ok()).filter(|_| frozen_t.is_some());
    let wind: f32 = param_or("wind", 1.0);
    // `stats=1`: frames in the current window and when it started
    let mut stats = flag("stats", false).then(|| (0u32, kansei_wasm::now()));
    let pan: f32 = param_or("pan", 0.0);
    let car_speed: f32 = param_or("car", 9.0);
    let mut last_t = 0.0f32;
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let mut clock = frame.time as f32;
        let mut t = frozen_t.unwrap_or(clock);
        // a still picture in motion: alternate between t and t + step
        let mut frame_time = frame.dt;
        if let Some(step) = step {
            t += (frame.index % 2) as f32 * step;
            clock = t;
            frame_time = step;
        }
        for effect in volume.effects.iter_mut() {
            if let Some(mb) = effect.as_any_mut().downcast_mut::<MotionBlurEffect>() {
                mb.set_frame_time(frame_time);
            }
        }

        // wind time, this frame's and last frame's
        for &idx in &animated {
            if let Some(r) = scene.get_renderable_mut(idx) {
                if let Some(buf) = r.material.bindable_buffer(0) {
                    renderer.queue().write_buffer(&buf, 16, bytemuck::cast_slice(&[clock, last_t, wind]));
                }
            }
        }
        last_t = clock;

        if let Some(r) = scene.get_renderable_mut(car) {
            r.object.set_position(-30.0 + (clock * car_speed) % 60.0, 0.85, -8.0);
        }
        // swinging left and right at up to `pan` degrees a second
        let yaw = pan.to_radians() / 0.8 * (0.8 * t).sin();
        camera.set_position(-2.0 + t * 0.4, 1.3, 6.0);
        camera.look_at(&Vec3::new(-2.0 + t * 0.4 + 26.0 * yaw.sin(), 2.2, 6.0 - 26.0 * yaw.cos()));

        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
        if let Some((frames, window_start)) = stats.as_mut() {
            *frames += 1;
            if *frames == 240 {
                let now = kansei_wasm::now();
                log::info!("frame interval: {:.2} ms", (now - *window_start) * 1000.0 / 240.0);
                *frames = 0;
                *window_start = now;
            }
        }
    });
    Ok(())
}
