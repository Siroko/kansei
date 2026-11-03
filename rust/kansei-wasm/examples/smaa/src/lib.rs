//! SMAA: thin, high-contrast things moving fast, where aliasing crawls and a temporal resolve
//! smears. A wire fence in a ±45° lattice across the view, square-mesh fences down both sides, a
//! lattice tower against the sky and a rotor of eight thin bars turning twice a second, under a
//! camera that swings left and right at up to `pan` degrees a second.
//!
//! The chain is `TemporalAAEffect`, `ToneMapEffect`, `SmaaEffect`; the mode runs one of the two
//! anti-aliasings (`PostProcessingVolume::set_enabled`) or neither. Every surface is the stock
//! `Material::standard_lit` with `outputs_velocity` (the TAA's motion vectors).
//!
//! URL parameters: `aa=off|smaa|smaa3|taa` (the starting mode; `smaa3` is SMAA's three-pass
//! reference), `reach=<px>` (the three passes' reach), `threshold=<luma>`, `t=<seconds>` (freeze
//! the scene), `pan=<deg/s>`, `size=<w>x<h>` (the drawing buffer's size), `bench=1` (SMAA, its
//! three passes and TAA in turn, see `bench.rs`).

// natively only the bench's tests build (the page's entry points are wasm32's)
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

mod bench;

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use glam::{Mat4, Quat, Vec3 as V3};
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, Geometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{GradientSkyOptions, Material, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, FilmGrain, SmaaEffect, SmaaMode, SmaaOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{flag, now, param, param_or, set_text, Canvas};

use bench::Bench;

/// The low sun behind the fences: the direction it travels, and its colour times illuminance
/// (lux); the wires stand dark against a bright sky.
const SUN_DIR: [f32; 3] = [0.4, -0.35, 0.85];
const SUN: [f32; 3] = [9000.0 * std::f32::consts::PI, 7600.0 * std::f32::consts::PI, 6000.0 * std::f32::consts::PI];
/// The sky hemisphere's radiance from straight up and straight down (cd/m²).
const SKY_UP: [f32; 3] = [900.0, 1100.0, 1500.0];
const SKY_DOWN: [f32; 3] = [60.0, 55.0, 45.0];
/// The rotor's turns a second.
const ROTOR_SPEED: f32 = 2.0;
/// The camera's swing, rad/s: its yaw is (pan / SWING) sin(SWING t), so it turns at up to `pan`.
const SWING: f32 = 2.5;

/// The anti-aliasing a frame runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Aa {
    Off,
    /// SMAA's single pass.
    Smaa,
    /// SMAA's three-pass reference.
    Smaa3,
    Taa,
}

impl Aa {
    fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "off" | "0" | "none" => Some(Aa::Off),
            "smaa" => Some(Aa::Smaa),
            "smaa3" => Some(Aa::Smaa3),
            "taa" => Some(Aa::Taa),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Aa::Off => "off",
            Aa::Smaa => "smaa",
            Aa::Smaa3 => "smaa3",
            Aa::Taa => "taa",
        }
    }
}

/// A diffuse surface under the sun and the sky (the stock standard material), writing motion
/// vectors for the TAA.
fn surface(label: &str, base: [f32; 3]) -> Material {
    Material::standard_lit(label, &StandardLitOptions { base_color: base, roughness: 0.9, sky_up: SKY_UP, sky_down: SKY_DOWN, outputs_velocity: true, ..Default::default() })
}

/// A unit box (`BoxGeometry::new(1, 1, 1)`) moved onto a square bar `thickness` thick from `a`
/// to `b`.
fn bar(a: V3, b: V3, thickness: f32) -> Mat4 {
    let d = b - a;
    Mat4::from_scale_rotation_translation(V3::new(thickness, d.length(), thickness), Quat::from_rotation_arc(V3::Y, d.normalize()), (a + b) * 0.5)
}

/// Bars as one geometry.
fn bars(label: &str, bars: &[Mat4]) -> Geometry {
    let unit = BoxGeometry::new(1.0, 1.0, 1.0);
    let parts: Vec<(&Geometry, Mat4)> = bars.iter().map(|m| (&unit, *m)).collect();
    Geometry::merged(label, &parts)
}

/// A wire fence in a ±45° lattice along x at depth `z`, `x0..x1` wide and `height` high, its
/// wires `wire` thick and `spacing` apart, with posts and rails.
fn diagonal_fence(x0: f32, x1: f32, z: f32, height: f32, spacing: f32, wire: f32) -> Vec<Mat4> {
    let mut out = Vec::new();
    let mut c = x0 - height;
    while c <= x1 + height {
        // y = x - c, and y = c - x, clipped to the panel
        let (a, b) = (c.max(x0), (c + height).min(x1));
        if b > a {
            out.push(bar(V3::new(a, a - c, z), V3::new(b, b - c, z), wire));
        }
        let (a, b) = ((c - height).max(x0), c.min(x1));
        if b > a {
            out.push(bar(V3::new(a, c - a, z), V3::new(b, c - b, z), wire));
        }
        c += spacing;
    }
    let mut x = x0;
    while x <= x1 + 1e-3 {
        out.push(bar(V3::new(x, 0.0, z), V3::new(x, height + 0.15, z), 0.07));
        x += 3.5;
    }
    for y in [0.05, height] {
        out.push(bar(V3::new(x0, y, z), V3::new(x1, y, z), 0.04));
    }
    out
}

/// A square-mesh wire fence along z at `x`, from `z0` to `z1`, `height` high.
fn square_fence(x: f32, z0: f32, z1: f32, height: f32, spacing: f32, wire: f32) -> Vec<Mat4> {
    let mut out = Vec::new();
    let mut y = spacing;
    while y <= height {
        out.push(bar(V3::new(x, y, z0), V3::new(x, y, z1), wire));
        y += spacing;
    }
    let mut z = z0;
    while z >= z1 {
        out.push(bar(V3::new(x, 0.0, z), V3::new(x, height, z), wire));
        z -= spacing;
    }
    let mut z = z0;
    while z >= z1 {
        out.push(bar(V3::new(x, 0.0, z), V3::new(x, height + 0.2, z), 0.08));
        z -= 3.0;
    }
    out
}

/// A lattice tower (a pylon) standing at `base`: four tapering legs, a ring and an X of braces on
/// each face every level, and two cross-arms near the top.
fn tower(base: V3, height: f32, foot: f32, top: f32, member: f32) -> Vec<Mat4> {
    let levels = 9;
    let corner = |level: usize, k: usize| {
        let t = level as f32 / levels as f32;
        let half = foot + (top - foot) * t;
        let (sx, sz) = [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)][k];
        base + V3::new(sx * half, height * t, sz * half)
    };
    let mut out = Vec::new();
    for level in 0..levels {
        for k in 0..4 {
            let n = (k + 1) % 4;
            out.push(bar(corner(level, k), corner(level + 1, k), member * 1.6));
            out.push(bar(corner(level + 1, k), corner(level + 1, n), member));
            out.push(bar(corner(level, k), corner(level + 1, n), member));
            out.push(bar(corner(level, n), corner(level + 1, k), member));
        }
    }
    for (y, span) in [(0.78, 7.0), (0.92, 5.0)] {
        let at = base + V3::new(0.0, height * y, 0.0);
        out.push(bar(at - V3::X * span, at + V3::X * span, member * 1.2));
        for s in [-1.0, 1.0] {
            out.push(bar(at + V3::X * span * s, at + V3::new(0.0, height * 0.06, 0.0) + V3::X * top * s, member));
        }
    }
    out
}

/// The rotor's bars, about its hub at the origin in the xy plane.
fn rotor(count: u32, radius: f32, thickness: f32) -> Vec<Mat4> {
    (0..count)
        .map(|k| {
            let a = k as f32 / count as f32 * std::f32::consts::TAU;
            bar(V3::new(a.cos(), a.sin(), 0.0) * 0.15, V3::new(a.cos(), a.sin(), 0.0) * radius, thickness)
        })
        .chain(std::iter::once(bar(V3::new(0.0, 0.0, -0.1), V3::new(0.0, 0.0, 0.1), 0.25)))
        .collect()
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    rotor: usize,
    aa: Aa,
    pan: f32,
    /// The scene's time, held (`t=`; the bench's).
    frozen_t: Option<f32>,
    bench: Option<Bench>,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

#[allow(dead_code)]
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

/// Run `aa`: SMAA (single pass or three) or TAA, or neither, and hand SMAA the tonemapper's
/// signal.
fn apply(st: &mut State, aa: Aa) {
    let taa = aa == Aa::Taa;
    if taa && !st.volume.is_enabled::<TemporalAAEffect>() {
        if let Some(fx) = st.volume.effect_mut::<TemporalAAEffect>() {
            fx.reset_history();
        }
    }
    st.volume.set_enabled::<TemporalAAEffect>(taa);
    st.volume.set_enabled::<SmaaEffect>(matches!(aa, Aa::Smaa | Aa::Smaa3));
    let encode_srgb = st.volume.effect::<ToneMapEffect>().is_some_and(|t| t.options.encode_srgb);
    if let Some(smaa) = st.volume.effect_mut::<SmaaEffect>() {
        smaa.set_mode(if aa == Aa::Smaa3 { SmaaMode::ThreePass } else { SmaaMode::SinglePass });
        smaa.set_film(FilmGrain { encode_srgb, ..Default::default() });
    }
    st.aa = aa;
}

impl State {
    fn frame(&mut self, time: f32) {
        let now_ms = now() * 1000.0;
        if let Some(bench) = self.bench.as_mut() {
            if let Some(report) = bench.frame(&self.renderer, now_ms) {
                log::info!("{report}");
                set_text("bench", &report);
            }
            if let Some(mode) = bench.mode(now_ms) {
                if mode != self.aa {
                    apply(self, mode);
                }
            }
        }
        let t = self.frozen_t.unwrap_or(time);
        if let Some(r) = self.scene.get_renderable_mut(self.rotor) {
            r.object.rotation.z = -(t * ROTOR_SPEED).fract() * std::f32::consts::TAU;
        }
        // swinging left and right at up to `pan` degrees a second
        let yaw = self.pan.to_radians() / SWING * (SWING * t).sin();
        let eye = V3::new(0.0, 1.5, 4.5);
        let target = eye + V3::new(26.0 * yaw.sin(), 3.2, -26.0 * yaw.cos());
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::new(target.x, target.y, target.z));

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if self.bench.as_ref().is_none_or(|b| b.mode(now_ms).is_none()) {
            kansei_wasm::perf::profile(&mut self.renderer);
        }
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let bench = flag("bench", false);
    let mut canvas = Canvas::find(canvas_id)?;
    let size = param("size").and_then(|v| {
        let (w, h) = v.trim().split_once('x')?;
        Some((w.parse::<u32>().ok()?, h.parse::<u32>().ok()?))
    });
    if let Some((w, h)) = size.or(bench.then_some((1920, 1080))) {
        canvas = canvas.with_size(w, h);
    }
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;

    let mut scene = Scene::new();
    let horizon = [2600.0, 2800.0, 3100.0];
    let sky = Material::gradient_sky("Sky", &GradientSkyOptions { zenith: [900.0, 1300.0, 2400.0], horizon, ground: horizon, curve: 0.5 });
    let mut sky = Renderable::new(SphereGeometry::new(500.0, 32, 16), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::from(SUN_DIR), Vec3::from(SUN), 1.0))));

    let mut ground = Renderable::new(PlaneGeometry::new(400.0, 400.0), surface("Ground", [0.16, 0.17, 0.12]));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // wires 1-3 px wide at 1080p: 1.2 cm at 7 m, 1.5 cm on the sides, 12 cm members 40 m away
    let wire = [0.03, 0.03, 0.035];
    scene.add(SceneNode::Renderable(Renderable::new(bars("DiagonalFence", &diagonal_fence(-15.0, 15.0, -7.0, 3.6, 0.4, 0.012)), surface("DiagonalFence", wire))));
    let mut sides = square_fence(-10.0, 2.0, -24.0, 3.0, 0.22, 0.015);
    sides.extend(square_fence(10.0, 2.0, -24.0, 3.0, 0.22, 0.015));
    scene.add(SceneNode::Renderable(Renderable::new(bars("SquareFences", &sides), surface("SquareFences", wire))));
    let mut pylons = tower(V3::new(5.0, 0.0, -40.0), 30.0, 3.2, 0.9, 0.12);
    pylons.extend(tower(V3::new(-22.0, 0.0, -55.0), 30.0, 3.2, 0.9, 0.12));
    scene.add(SceneNode::Renderable(Renderable::new(bars("Towers", &pylons), surface("Towers", [0.05, 0.05, 0.055]))));

    // the rotor on its mast, behind the diagonal fence and above it
    let hub = V3::new(-3.5, 5.2, -10.5);
    scene.add(SceneNode::Renderable(Renderable::new(bars("Mast", &[bar(V3::new(hub.x, 0.0, hub.z - 0.3), V3::new(hub.x, hub.y, hub.z - 0.3), 0.16)]), surface("Mast", [0.08, 0.08, 0.08]))));
    let mut blades = Renderable::new(bars("Rotor", &rotor(8, 2.8, 0.04)), surface("Rotor", [0.55, 0.06, 0.03]));
    blades.object.set_position(hub.x, hub.y, hub.z);
    let rotor = scene.add(SceneNode::Renderable(blades));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(11.0);
        ToneMapEffect::new(options)
    };
    let smaa = SmaaEffect::new(SmaaOptions {
        threshold: param_or("threshold", 0.1),
        three_pass_reach: param_or("reach", SmaaEffect::REACH),
        ..Default::default()
    });
    let effects: Vec<Box<dyn PostProcessingEffect>> =
        vec![Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })), Box::new(tonemap), Box::new(smaa)];
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(50.0, 0.1, 1000.0, canvas.aspect());
    camera.update_projection_matrix();

    let (width, height) = canvas.size();
    let aa = param("aa").and_then(|v| Aa::parse(&v)).unwrap_or(Aa::Smaa);
    let frozen_t = param("t").and_then(|v| v.trim().parse().ok()).or(bench.then_some(2.0));
    let bench = bench.then(|| {
        renderer.set_profiling(true);
        Bench::new(now() * 1000.0, 4, (width, height))
    });
    log::info!("Kansei — SMAA (WASM) ready: {width}x{height}, aa={}, {:?}", aa.name(), renderer.presentation_format());

    let mut st = State { renderer, scene, camera, volume, rotor, aa, pan: param_or("pan", 90.0), frozen_t, bench };
    apply(&mut st, aa);
    let state = Rc::new(RefCell::new(st));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| {
        let mut st = state.borrow_mut();
        let st = &mut *st;
        frame.resize(&mut st.renderer, &mut st.camera);
        st.frame(frame.time as f32);
    });
    Ok(())
}
