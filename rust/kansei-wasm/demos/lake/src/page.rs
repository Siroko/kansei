//! The lake page: the world without a character, seen from an orbit camera. The cannon is always
//! ready (E, the gamepad's X, or a click on its prompt fires it as it stands; R or Y drains the
//! lake), the mill turns, and the P panel tweaks the water.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::math::Vec3;
use kansei_core::objects::Scene;
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_wasm::{flag, set_text, Canvas, Gamepad, Keys};

use crate::lake::CENTER;
use crate::panel::{register, Host};
use crate::world::{renderer, World, WorldInput, WorldOptions};

struct State {
    renderer: Renderer,
    scene: Scene,
    world: World,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    keys: Keys,
    pad: Gamepad,
    last: f64,
    frame: u32,
    fps: f32,
    /// `profile=1`: when the profile was last logged (0: not profiling).
    profile_since: f64,
}

impl Host for State {
    fn parts(&mut self) -> (&mut World, &mut PostProcessingVolume, &Renderer) {
        (&mut self.world, &mut self.volume, &self.renderer)
    }
}

impl State {
    fn frame(&mut self) {
        let now = kansei_wasm::now();
        let dt = ((now - self.last) as f32).clamp(1e-4, 1.0 / 15.0);
        self.last = now;
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;

        // the cannon's trigger (E, the gamepad's X) and the drain (R, Y)
        let (mut fire_pressed, mut fire_held, mut drain) = (false, false, false);
        fire_held |= self.keys.held("e");
        for key in self.keys.take_pressed() {
            match key.as_str() {
                "e" => fire_pressed = true,
                "r" => drain = true,
                _ => {}
            }
        }
        if self.pad.poll() {
            let [x, y] = self.pad.right_stick();
            self.controls.rotate(-x * 2.5 * dt, y * 1.5 * dt);
            fire_pressed |= self.pad.pressed(Gamepad::X);
            drain |= self.pad.pressed(Gamepad::Y);
            fire_held |= self.pad.held(Gamepad::X);
        }

        self.controls.update(&mut self.camera, dt);
        let view_proj = self.camera.view_projection().to_glam();
        let State { world, scene, volume, renderer, .. } = &mut *self;
        world.update(WorldInput { fire: (fire_pressed, fire_held), drain, ..Default::default() }, dt, scene, volume, renderer, view_proj);

        // HUD, a few times a second
        if self.frame % 10 == 0 {
            set_text("hud", &format!(
                "{:.0} fps\n{}\nE / X fire the cannon (hold to pour) · R / Y drain the lake\ndrag / right stick orbit · right drag pan · wheel zoom",
                self.fps,
                self.world.status(""),
            ));
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if self.profile_since > 0.0 && now - self.profile_since > 3.0 {
            self.profile_since = now;
            log::info!("profile ({:.1} ms/frame)\n{}", 1000.0 / self.fps, self.renderer.take_profile().report());
        }
    }
}

/// Start the lake page on canvas `canvas_id`.
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let renderer = renderer(&canvas).await;
    let mut scene = Scene::new();
    // no character: the cannon fires from wherever the camera is
    let mut world = World::new(&renderer, &mut scene, &WorldOptions { cannon_reach: false, ..WorldOptions::from_url() });
    let volume = world.post_processing(&renderer, &mut scene);

    let mut camera = Camera::new(45.0, 0.1, 1200.0, canvas.aspect());
    camera.update_projection_matrix();
    // from the south, a little west: the cannon on the left firing across, the mill beyond
    let target = if world.lake.is_some() { Vec3::new(CENTER[0], 0.0, CENTER[1]) } else { Vec3::new(0.0, 0.0, 0.0) };
    let mut controls = CameraControls::from_canvas(canvas.element(), target, 16.0).with_mouse_pan(canvas.element());
    controls.set_view(target, 16.0, std::f32::consts::PI + 0.25, 0.4);

    log::info!("Kansei — Lake (WASM) ready");
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        world,
        camera,
        controls,
        volume,
        keys: Keys::listen(),
        pad: Gamepad::new(),
        last: kansei_wasm::now(),
        frame: 0,
        fps: 60.0,
        profile_since: 0.0,
    }));
    if flag("profile", false) {
        let mut s = state.borrow_mut();
        s.renderer.set_profiling(true);
        s.profile_since = kansei_wasm::now();
    }
    register(state.clone());
    kansei_wasm::run(&canvas, move |frame| {
        let mut s = state.borrow_mut();
        let State { renderer, camera, .. } = &mut *s;
        frame.resize(renderer, camera);
        s.frame();
    });
    Ok(())
}
