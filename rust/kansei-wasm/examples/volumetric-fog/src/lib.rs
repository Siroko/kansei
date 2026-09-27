//! Froxel volumetric fog: a grove of pillars under a low sun (shadowed, so its light comes
//! through in shafts), a warm lamp with cube shadows and a cool one without, and a dim ambient
//! sky term, rendered through the GBuffer path with VolumetricFogEffect.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingVolume,
    effects::{VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};

const LIT_WGSL: &str = include_str!("../../../../kansei-core/src/shaders/basic_lit.wgsl");

fn lit_material(label: &str, color: [f32; 4], specular: [f32; 4]) -> Material {
    let data: [f32; 8] = [color[0], color[1], color[2], color[3], specular[0], specular[1], specular[2], specular[3]];
    let mut material = Material::new(label, LIT_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    material.set_uniform_bindable(0, label, &data);
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
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_secs() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    // sample_count 1: the post-processing GBuffer is not multisampled
    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.01, 0.012, 0.02, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;
    renderer.enable_shadows(2048);
    renderer.enable_point_shadows(512, 1);

    let mut scene = Scene::new();

    let mut floor = Renderable::new(PlaneGeometry::new(60.0, 60.0), lit_material("Floor", [0.35, 0.35, 0.33, 1.0], [0.05, 0.05, 0.05, 0.05]));
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.cast_shadow = false;
    scene.add(SceneNode::Renderable(floor));

    // a loose grid of pillars, jittered so the shafts don't line up
    for i in 0..9 {
        for j in 0..9 {
            let h = ((i * 7 + j * 13) % 5) as f32;
            let (x, z) = ((i as f32 - 4.0) * 3.2 + (j % 3) as f32 * 0.6, (j as f32 - 4.0) * 3.2 + (i % 2) as f32 * 0.9);
            if x.abs() < 2.0 && z.abs() < 2.0 {
                continue; // a clearing in the middle
            }
            let height_m = 6.0 + h;
            let mut pillar = Renderable::new(
                BoxGeometry::new(0.45, height_m, 0.45),
                lit_material("Pillar", [0.3, 0.27, 0.24, 1.0], [0.05, 0.05, 0.05, 0.1]),
            );
            pillar.object.set_position(x, height_m * 0.5, z);
            scene.add(SceneNode::Renderable(pillar));
        }
    }

    // low sun, shadowed: the fog shows its light in shafts between the pillars
    let mut sun = DirectionalLight::new(Vec3::new(-0.8, -0.35, -0.45), Vec3::new(1.0, 0.85, 0.65), 1.6);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    // warm lamp in the clearing with cube shadows, cool one unshadowed
    let mut lamp = PointLight::new(Vec3::new(0.0, 1.6, 0.0), Vec3::new(1.0, 0.55, 0.25), 6.0, 14.0);
    lamp.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Point(lamp)));
    scene.add(SceneNode::Light(Light::Point(PointLight::new(Vec3::new(9.0, 2.5, -8.0), Vec3::new(0.3, 0.55, 1.0), 4.0, 12.0))));

    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { far: 80.0, temporal: true, ..Default::default() },
        base_density: 0.1,
        height_falloff: 0.12,
        anisotropy: 0.55,
        wind_direction: Vec3::new(0.6, 0.0, 0.2),
        ambient: Vec3::new(0.012, 0.016, 0.03),
        ..Default::default()
    });
    fog.set_shadow_map(renderer.shadow_map());
    fog.set_point_shadows(renderer.cubemap_shadow_map());
    fog.update_lights(scene.lights());
    let volume = PostProcessingVolume::new(&renderer, vec![Box::new(fog)]);

    let mut camera = Camera::new(50.0, 0.1, 200.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    log::info!("Kansei — Volumetric Fog (WASM) ready");

    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, start_ms: now_secs() }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref start_ms } = *st;
            let t = (now_secs() - *start_ms) as f32;

            if let Some(fog) = volume.effects[0].as_any_mut().downcast_mut::<VolumetricFogEffect>() {
                fog.time = t;
            }

            // start on the far side of the grove from the sun, looking into the shafts
            let angle = -2.08 + t * 0.06;
            camera.set_position(angle.sin() * 22.0, 4.0, angle.cos() * 22.0);
            camera.look_at(&Vec3::new(0.0, 3.0, 0.0));

            renderer.render_with_postprocessing(scene, camera, volume);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
