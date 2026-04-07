use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::math::{Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::objects::{CornellBox, Scene};
use kansei_core::renderers::{Renderer, RendererConfig};

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;

    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;

    // Scene: CornellBox
    let mut scene = Scene::new();
    let cornell_box = CornellBox::new([-4.0, -1.5, -4.0], [4.0, 4.0, 4.0]);
    cornell_box.add_to_scene(&mut scene);

    // Camera
    let mut camera = Camera::new(45.0, 0.1, 100.0, width as f32 / height as f32);
    camera.set_position(0.0, 2.0, 6.0);
    camera.look_at(&Vec3::ZERO);
    camera.update_projection_matrix();

    log::info!("Kansei — Spinning Box (WASM) ready");

    // Animation loop
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
    }));

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let s = state.clone();

    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = s.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera } = *st;
            renderer.render(scene, camera);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
