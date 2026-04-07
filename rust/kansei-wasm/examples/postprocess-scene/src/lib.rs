use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::{Binding, Material, MaterialOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingVolume,
    effects::{BloomEffect, BloomOptions, ColorGradingEffect, ColorGradingOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};

const LIT_WGSL: &str = include_str!("../../../../kansei-core/src/shaders/basic_lit.wgsl");

fn create_lit_material(label: &str, color: [f32; 4], specular: [f32; 4]) -> Material {
    let mat_data: [f32; 8] = [
        color[0], color[1], color[2], color[3],
        specular[0], specular[1], specular[2], specular[3],
    ];
    let mut material = Material::new(
        label,
        LIT_WGSL,
        vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
        MaterialOptions::default(),
    );
    material.set_uniform_bindable(0, label, &mat_data);
    material
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
    volume: PostProcessingVolume,
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

    // sample_count=1 since post-processing GBuffer is non-MSAA
    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_target(wgpu::SurfaceTarget::Canvas(canvas.clone())).await;

    let mut scene = Scene::new();

    // Floor
    let floor_geo = PlaneGeometry::new(20.0, 20.0);
    let floor_mat = create_lit_material("FloorMat", [0.5, 0.5, 0.5, 1.0], [0.2, 0.2, 0.2, 0.1]);
    let mut floor = Renderable::new(floor_geo, floor_mat);
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.object.update_model_matrix();
    floor.object.update_world_matrix(None);

    // Bright emissive box (HDR to trigger bloom)
    let box_geo = BoxGeometry::new(2.0, 2.0, 2.0);
    let box_mat = create_lit_material(
        "BrightBoxMat",
        [3.0, 3.0, 2.5, 1.0],
        [1.0, 1.0, 1.0, 0.8],
    );
    let mut box_obj = Renderable::new(box_geo, box_mat);
    box_obj.object.set_position(0.0, 1.0, 0.0);
    box_obj.object.update_model_matrix();
    box_obj.object.update_world_matrix(None);

    // Sphere
    let sphere_geo = SphereGeometry::new(1.0, 32, 16);
    let sphere_mat = create_lit_material("SphereMat", [0.2, 0.4, 0.9, 1.0], [0.8, 0.8, 0.8, 0.6]);
    let mut sphere = Renderable::new(sphere_geo, sphere_mat);
    sphere.object.set_position(3.5, 1.0, 0.0);
    sphere.object.update_model_matrix();
    sphere.object.update_world_matrix(None);

    scene.add(SceneNode::Renderable(floor));
    scene.add(SceneNode::Renderable(box_obj));
    scene.add(SceneNode::Renderable(sphere));

    // Lights
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(
        Vec3::new(-0.5, -1.0, -0.3),
        Vec3::new(1.0, 0.95, 0.8),
        0.8,
    ))));
    scene.add(SceneNode::Light(Light::Point(PointLight::new(
        Vec3::new(3.0, 3.0, 2.0),
        Vec3::new(1.0, 0.6, 0.2),
        3.0,
        15.0,
    ))));
    scene.add(SceneNode::Light(Light::Point(PointLight::new(
        Vec3::new(-3.0, 2.0, -2.0),
        Vec3::new(0.3, 0.5, 1.0),
        2.0,
        12.0,
    ))));

    // Camera
    let mut camera = Camera::new(45.0, 0.1, 100.0, width as f32 / height as f32);
    camera.set_position(8.0, 5.0, 10.0);
    camera.look_at(&Vec3::new(0.0, 1.0, 0.0));
    camera.update_projection_matrix();

    // Post-processing: bloom + color grading
    let volume = PostProcessingVolume::new(
        &renderer,
        vec![
            Box::new(BloomEffect::new(BloomOptions {
                threshold: 0.8,
                intensity: 0.6,
                ..Default::default()
            })),
            Box::new(ColorGradingEffect::new(ColorGradingOptions {
                contrast: 1.2,
                temperature: 0.15,
                ..Default::default()
            })),
        ],
    );

    log::info!("Kansei — Post-Processing Scene (WASM) ready");

    let start_ms = now_secs();
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        volume,
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
                ref mut volume,
                ref start_ms,
            } = *st;
            let t = (now_secs() - *start_ms) as f32;

            // Rotate the box slowly
            if let Some(r) = scene.get_renderable_mut(1) {
                r.object.rotation.y = t * 0.3;
                r.object.update_model_matrix();
                r.object.update_world_matrix(None);
            }

            // Update normal matrices
            for i in 0..scene.children_len() {
                if let Some(r) = scene.get_renderable_mut(i) {
                    r.object.update_normal_matrix();
                }
            }

            // Slowly orbit camera
            let angle = t * 0.15;
            let dist = 12.0;
            camera.set_position(
                angle.sin() * dist,
                5.0,
                angle.cos() * dist,
            );
            camera.look_at(&Vec3::new(0.0, 1.0, 0.0));

            renderer.render_with_postprocessing(scene, camera, volume);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    Ok(())
}
