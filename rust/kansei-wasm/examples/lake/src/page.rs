//! The lake page: the world without a character, seen from an orbit camera. The cannon is always
//! ready (E, the gamepad's X, or a click on its prompt fires it as it stands; R or Y drains the
//! lake), the mill turns, and the P panel tweaks the water.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::math::Vec3;
use kansei_core::objects::Scene;
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_wasm::{flag, Canvas};

use crate::lake::CENTER;
use crate::panel::{register, Host};
use crate::world::{renderer, World, WorldInput, WorldOptions};

/// Keys held and pressed since last frame, from the page's key events.
#[derive(Default)]
struct Keys {
    held: HashSet<String>,
    pressed: Vec<String>,
}

struct State {
    renderer: Renderer,
    scene: Scene,
    world: World,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    keys: Rc<RefCell<Keys>>,
    /// Gamepad buttons held last frame (to act on a press, not while held).
    pad_held: Vec<bool>,
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

/// The right stick and the pressed state of each button of the first connected gamepad.
fn gamepad() -> Option<([f32; 2], Vec<bool>)> {
    let pads = web_sys::window()?.navigator().get_gamepads().ok()?;
    let pad: web_sys::Gamepad = (0..pads.length()).find_map(|i| pads.get(i).dyn_into().ok())?;
    let axes: Vec<f32> = pad.axes().iter().map(|a| a.as_f64().unwrap_or(0.0) as f32).collect();
    let (x, y) = (axes.get(2).copied().unwrap_or(0.0), axes.get(3).copied().unwrap_or(0.0));
    let m = (x * x + y * y).sqrt();
    let right = if m < 0.15 { [0.0, 0.0] } else { let s = ((m - 0.15) / 0.85).min(1.0) / m; [x * s, y * s] };
    let buttons = pad.buttons().iter().map(|b| b.dyn_into::<web_sys::GamepadButton>().is_ok_and(|b| b.pressed())).collect();
    Some((right, buttons))
}

fn set_hud(text: &str) {
    if let Some(hud) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id("hud")) {
        hud.set_text_content(Some(text));
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
        {
            let mut keys = self.keys.borrow_mut();
            fire_held |= keys.held.contains("e");
            for key in keys.pressed.drain(..) {
                match key.as_str() {
                    "e" => fire_pressed = true,
                    "r" => drain = true,
                    _ => {}
                }
            }
        }
        if let Some((right, buttons)) = gamepad() {
            self.controls.rotate(-right[0] * 2.5 * dt, right[1] * 1.5 * dt);
            let pressed = |i: usize| buttons.get(i).copied().unwrap_or(false);
            let was = |i: usize| self.pad_held.get(i).copied().unwrap_or(false);
            fire_pressed |= pressed(2) && !was(2);
            drain |= pressed(3) && !was(3);
            fire_held |= pressed(2);
            self.pad_held = buttons;
        }

        self.controls.update(&mut self.camera, dt);
        let view_proj = self.camera.view_projection().to_glam();
        let State { world, scene, volume, renderer, .. } = &mut *self;
        world.update(WorldInput { fire: (fire_pressed, fire_held), drain, ..Default::default() }, dt, scene, volume, renderer, view_proj);

        // HUD, a few times a second
        if self.frame % 10 == 0 {
            set_hud(&format!(
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
    let window = web_sys::window().unwrap();
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

    let keys = Rc::new(RefCell::new(Keys::default()));
    {
        let down = keys.clone();
        let on_down = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            let key = e.key().to_lowercase();
            let mut keys = down.borrow_mut();
            if !e.repeat() {
                keys.pressed.push(key.clone());
            }
            keys.held.insert(key);
        });
        window.add_event_listener_with_callback("keydown", on_down.as_ref().unchecked_ref())?;
        on_down.forget();
        let up = keys.clone();
        let on_up = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            up.borrow_mut().held.remove(&e.key().to_lowercase());
        });
        window.add_event_listener_with_callback("keyup", on_up.as_ref().unchecked_ref())?;
        on_up.forget();
        // a key released while the page is in the background never sends keyup
        let blur = keys.clone();
        let on_blur = Closure::<dyn FnMut()>::new(move || blur.borrow_mut().held.clear());
        window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref())?;
        on_blur.forget();
    }

    log::info!("Kansei — Lake (WASM) ready");
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        world,
        camera,
        controls,
        volume,
        keys,
        pad_held: Vec::new(),
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
