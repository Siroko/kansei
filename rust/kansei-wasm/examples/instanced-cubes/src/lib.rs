use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::InstanceBuffer;
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Binding, Material, MaterialOptions};
use kansei_core::math::{Mat4, Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};

const INSTANCED_WGSL: &str = include_str!("../../../../kansei-core/src/shaders/basic_instanced.wgsl");

const GRID_SIZE: usize = 10;
const INSTANCE_COUNT: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const SPACING: f32 = 3.0;

fn build_instance_matrices(time: f32) -> Vec<f32> {
    let mut data = vec![0.0f32; INSTANCE_COUNT * 16];
    let offset = (GRID_SIZE as f32 - 1.0) * SPACING * 0.5;
    let mut idx = 0;
    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let px = x as f32 * SPACING - offset;
                let py = y as f32 * SPACING - offset;
                let pz = z as f32 * SPACING - offset;
                let angle = time * 0.5 + (x + y + z) as f32 * 0.3;
                let m = Mat4::from_translation(Vec3::new(px, py, pz))
                    * Mat4::from_rotation_y(angle);
                data[idx..idx + 16].copy_from_slice(&m.to_cols_array());
                idx += 16;
            }
        }
    }
    data
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    start_ms: f64,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window()
        .unwrap()
        .request_animation_frame(f.as_ref().unchecked_ref())
        .unwrap();
}

fn now_secs() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
        / 1000.0
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

    let renderer = Renderer::create(
        RendererConfig {
            width,
            height,
            sample_count: 1,
            clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0),
            ..Default::default()
        },
        wgpu::SurfaceTarget::Canvas(canvas.clone()),
    )
    .await;

    // Geometry: 1x1x1 box
    let geometry = BoxGeometry::new(1.0, 1.0, 1.0);

    // Material with color uniform
    let color_data: [f32; 4] = [0.4, 0.7, 0.9, 1.0];
    let mut material = Material::new(
        "InstancedMaterial",
        INSTANCED_WGSL,
        vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
        MaterialOptions::default(),
    );
    material.set_uniform_bindable(0, "ColorUniform", &color_data);

    // Instance buffer
    let matrices = build_instance_matrices(0.0);
    let instance_buf = InstanceBuffer::from_mat4("InstanceMatrices", &matrices, 3);

    let renderable = Renderable::new_instanced(
        geometry,
        material,
        INSTANCE_COUNT as u32,
        vec![instance_buf],
    );

    let mut scene = Scene::new();
    scene.add(SceneNode::Renderable(renderable));

    // Camera: orbit from a distance
    let mut camera = Camera::new(45.0, 0.1, 500.0, width as f32 / height as f32);
    camera.set_position(20.0, 15.0, 40.0);
    camera.look_at(&Vec3::ZERO);
    camera.update_projection_matrix();

    log::info!(
        "Kansei — Instanced Cubes (WASM): {} instances",
        INSTANCE_COUNT
    );

    let start_ms = now_secs();
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        start_ms,
    }));

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    let s = state.clone();

    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = s.borrow_mut();
            let State {
                ref mut renderer,
                ref mut scene,
                ref mut camera,
                ref start_ms,
            } = *st;
            let t = (now_secs() - *start_ms) as f32;

            // Update instance matrices
            if let Some(r) = scene.get_renderable_mut(0) {
                let matrices = build_instance_matrices(t);
                if let Some(ib) = r.instance_buffers.get(0) {
                    ib.update(bytemuck::cast_slice(&matrices));
                }
            }

            // Slowly orbit camera
            let angle = t * 0.1;
            let dist = 50.0;
            camera.set_position(
                angle.sin() * dist,
                15.0,
                angle.cos() * dist,
            );
            camera.look_at(&Vec3::ZERO);

            renderer.render(scene, camera);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
