//! Volumetric smoke through the volumetric fog: a dark room with one window, a low sun through
//! it in a shaft that faint fog makes visible, a table in the shaft and steam off a cup on it.
//! The steam is a `SmokeSimulation` volume marched and lit by the `VolumetricFogEffect`
//! (`set_smoke`), so the window frame's shadow cuts through it as it does through the fog.
//! See README.md for the URL parameters.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, CylinderGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FixedStep;
use kansei_core::postprocessing::effects::{TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::smoke::{SmokeMotion, SmokeMouse, SmokeSimulation, SmokeSource, SmokeVolume, SmokeVolumeId};
use kansei_wasm::{flag, param, Canvas};

/// The room's inside: x and z half extents, and its height (metres).
const ROOM: [f32; 3] = [3.0, 3.0, 2.5];
/// The window in the +x wall: z half width, and its sill and head heights.
const WINDOW: [f32; 3] = [0.6, 1.0, 2.2];
/// The table's top centre and the cup on it.
const TABLE: [f32; 3] = [1.2, 0.75, 0.0];
const CUP_HEIGHT: f32 = 0.09;
/// The low sun: the direction it travels (in through the window, from +x), and its radiance.
const SUN_DIR: [f32; 3] = [-0.88, -0.31, 0.12];
const SUN: f32 = 6.0;
/// The fog's default density, per metre.
const FOG_DENSITY: f32 = 0.03;

fn surface(label: &str, base: [f32; 3], velocity: bool) -> Material {
    Material::standard_lit(label, &StandardLitOptions {
        base_color: base,
        roughness: 0.85,
        sky_up: [0.02, 0.022, 0.026],
        sky_down: [0.01, 0.009, 0.008],
        outputs_velocity: velocity,
        ..Default::default()
    })
}

/// A box of `size` centred at `at`.
fn slab(scene: &mut Scene, label: &str, size: [f32; 3], at: [f32; 3], base: [f32; 3], velocity: bool) {
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), surface(label, base, velocity));
    r.object.set_position(at[0], at[1], at[2]);
    scene.add(SceneNode::Renderable(r));
}

/// The room, its window with a cross-shaped frame, the table and the cup.
fn build_room(scene: &mut Scene, velocity: bool) {
    let [hx, h, hz] = ROOM;
    let t = 0.1;
    let wall = [0.16, 0.15, 0.14];
    slab(scene, "Floor", [2.0 * hx + 2.0 * t, t, 2.0 * hz + 2.0 * t], [0.0, -t * 0.5, 0.0], [0.22, 0.16, 0.11], velocity);
    slab(scene, "Ceiling", [2.0 * hx + 2.0 * t, t, 2.0 * hz + 2.0 * t], [0.0, h + t * 0.5, 0.0], wall, velocity);
    slab(scene, "WallBack", [t, h + 2.0 * t, 2.0 * hz + 2.0 * t], [-hx - t * 0.5, h * 0.5, 0.0], wall, velocity);
    for z in [-1.0f32, 1.0] {
        slab(scene, "WallSide", [2.0 * hx + 2.0 * t, h + 2.0 * t, t], [0.0, h * 0.5, z * (hz + t * 0.5)], wall, velocity);
    }
    // the window wall, round its opening
    let [wz, sill, head] = WINDOW;
    let x = hx + t * 0.5;
    slab(scene, "WallWindowBelow", [t, sill + t, 2.0 * hz + 2.0 * t], [x, (sill - t) * 0.5, 0.0], wall, velocity);
    slab(scene, "WallWindowAbove", [t, h - head + t, 2.0 * hz + 2.0 * t], [x, (head + h + t) * 0.5, 0.0], wall, velocity);
    for z in [-1.0f32, 1.0] {
        let width = hz + t - wz;
        slab(scene, "WallWindowSide", [t, head - sill, width], [x, (sill + head) * 0.5, z * (wz + width * 0.5)], wall, velocity);
    }
    // the frame: a cross
    let frame = [0.05, 0.045, 0.04];
    slab(scene, "FrameVertical", [0.06, head - sill, 0.05], [x, (sill + head) * 0.5, 0.0], frame, velocity);
    slab(scene, "FrameHorizontal", [0.06, 0.05, 2.0 * wz], [x, (sill + head) * 0.5, 0.0], frame, velocity);
    // the table and the cup
    let [tx, ty, tz] = TABLE;
    let wood = [0.3, 0.18, 0.1];
    slab(scene, "TableTop", [1.2, 0.05, 0.8], [tx, ty - 0.025, tz], wood, velocity);
    for (dx, dz) in [(-0.55f32, -0.35f32), (0.55, -0.35), (-0.55, 0.35), (0.55, 0.35)] {
        slab(scene, "TableLeg", [0.05, ty - 0.05, 0.05], [tx + dx, (ty - 0.05) * 0.5, tz + dz], wood, velocity);
    }
    let mut cup = Renderable::new(CylinderGeometry::new(0.038, 0.045, CUP_HEIGHT, 32, 1), surface("Cup", [0.75, 0.73, 0.7], velocity));
    cup.object.set_position(tx, ty, tz);
    scene.add(SceneNode::Renderable(cup));
}

/// Where the steam leaves the cup.
fn cup_top() -> Vec3 {
    Vec3::new(TABLE[0], TABLE[1] + CUP_HEIGHT, TABLE[2])
}

/// The steam's look: how dense the plume reads.
const STEAM_DENSITY: f32 = 8.0;

/// The plume over the cup: steam rising from it, 0.8 m tall, at `motion`.
fn plume(motion: SmokeMotion) -> SmokeVolume {
    let source = SmokeSource { position: cup_top(), radius: 0.03, ..Default::default() };
    let mut v = SmokeVolume::rising_from(source, 0.8, motion);
    v.look.density_scale = STEAM_DENSITY;
    v
}

/// `stats=1`: frames and sim steps in the current window, and when it started.
struct Stats {
    frames: u32,
    steps: u32,
    start: f64,
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    smoke: SmokeSimulation,
    #[allow(dead_code)]
    plume: SmokeVolumeId,
    fixed: FixedStep,
    stats: Option<Stats>,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

impl State {
    fn fog(&mut self) -> Option<&mut VolumetricFogEffect> {
        self.volume.effect_mut::<VolumetricFogEffect>()
    }

    fn frame(&mut self, dt: f32, time: f32) {
        self.controls.update(&mut self.camera, dt);
        let steps = self.fixed.advance(dt as f64);
        for _ in 0..steps {
            self.smoke.step(self.fixed.step as f32, &SmokeMouse::NONE);
        }
        if let Some(fog) = self.fog() {
            fog.time = time;
        }
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        kansei_wasm::perf::profile(&mut self.renderer);
        if let Some(stats) = self.stats.as_mut() {
            stats.frames += 1;
            stats.steps += steps;
            if stats.frames == 240 {
                let now = kansei_wasm::now();
                let (w, h) = self.renderer.render_size();
                log::info!(
                    "smoke stats: {w}x{h}, {:.2} ms between frames, {:.2} sim steps a frame\n{}",
                    (now - stats.start) * 1000.0 / 240.0,
                    stats.steps as f32 / 240.0,
                    self.renderer.take_profile().report()
                );
                *stats = Stats { frames: 0, steps: 0, start: now };
            }
        }
    }

    fn view_json(&mut self) -> String {
        let fog_density = self.fog().map_or(0.0, |f| f.base_density);
        let t = self.controls.look_target();
        let r2 = |v: f32| (v * 1000.0).round() / 1000.0;
        format!(
            "{{\"fogDensity\":{},\"radius\":{},\"azimuth\":{},\"elevation\":{},\"target\":[{},{},{}]}}",
            r2(fog_density),
            r2(self.controls.radius),
            r2(self.controls.azimuth().to_degrees()),
            r2(self.controls.elevation().to_degrees()),
            r2(t.x),
            r2(t.y),
            r2(t.z)
        )
    }

    fn set(&mut self, key: &str, value: f32) {
        match key {
            "fogDensity" => {
                if let Some(fog) = self.fog() {
                    fog.base_density = value.max(0.0);
                }
            }
            "radius" | "azimuth" | "elevation" | "targetX" | "targetY" | "targetZ" => {
                let (mut t, mut r, mut a, mut e) = (self.controls.look_target(), self.controls.radius, self.controls.azimuth(), self.controls.elevation());
                match key {
                    "radius" => r = value.max(0.2),
                    "azimuth" => a = value.to_radians(),
                    "elevation" => e = value.to_radians(),
                    "targetX" => t.x = value,
                    "targetY" => t.y = value,
                    _ => t.z = value,
                }
                self.controls.set_view(t, r, a, e);
            }
            _ => log::warn!("smoke: no setting {key}"),
        }
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_shadows(2048);
    let taa = param("aa").as_deref() != Some("off");

    let mut scene = Scene::new();
    build_room(&mut scene, taa);
    let mut sun = DirectionalLight::new(Vec3::from(SUN_DIR).normalize(), Vec3::new(1.0, 0.9, 0.75), SUN);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    let mut smoke = SmokeSimulation::new(renderer.device(), renderer.queue());
    let motion = match param("motion").as_deref() {
        Some("waves") => SmokeMotion::waves(),
        Some("curl") => SmokeMotion::curl_noise(),
        _ => SmokeMotion::fluid(),
    };
    let plume = smoke.add(plume(motion));
    let stats = flag("stats", false).then(|| Stats { frames: 0, steps: 0, start: kansei_wasm::now() });
    renderer.set_profiling(stats.is_some());

    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { far: 20.0, temporal: true, ..Default::default() },
        base_density: FOG_DENSITY,
        height_falloff: 0.0,
        anisotropy: 0.6,
        ambient: Vec3::new(0.004, 0.0045, 0.005),
        ..Default::default()
    });
    fog.set_shadow_map(renderer.shadow_map());
    fog.update_lights(scene.lights());
    fog.set_smoke(Some(&smoke.handle()));

    let tonemap = ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()));
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(fog)];
    if taa {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    // drag to orbit, right-drag or shift-drag to pan, the wheel to zoom: looking at the steam
    // from the room, the window on the right
    let mut camera = Camera::new(45.0, 0.05, 15.0, canvas.aspect());
    camera.update_projection_matrix();
    let target = Vec3::new(TABLE[0] - 0.1, TABLE[1] + 0.35, TABLE[2]);
    let mut controls = CameraControls::from_canvas(canvas.element(), target, 2.2).with_mouse_pan(canvas.element());
    controls.set_view(target, 2.2, -0.25, 0.12);

    log::info!("Kansei — Volumetric Smoke (WASM) ready, TAA {taa}");
    let fixed = FixedStep::new(1.0 / 60.0);
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, controls, volume, smoke, plume, fixed, stats }));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| {
        let mut st = state.borrow_mut();
        let st = &mut *st;
        frame.resize(&mut st.renderer, &mut st.camera);
        st.frame(frame.dt, frame.time as f32);
    });
    Ok(())
}

/// The current settings and view as JSON.
#[wasm_bindgen]
pub fn view() -> String {
    with_state(|s| s.view_json()).unwrap_or_default()
}

/// Set one value by its `view` key (`fogDensity`; the view's `radius`, `azimuth` and
/// `elevation` in degrees, `targetX`/`Y`/`Z`).
#[wasm_bindgen]
pub fn set(key: &str, value: f32) {
    with_state(|s| s.set(key, value));
}
