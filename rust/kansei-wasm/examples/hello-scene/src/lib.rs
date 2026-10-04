//! Hello scene: the smallest complete Kansei scene. A floor, a spinning box and a sphere in the
//! stock lit material (`Material::basic_lit`), a sun and two point lights, shadows from the sun
//! and one lamp, and an orbit camera. Drag to orbit, wheel or pinch to zoom.
//!
//! URL parameters: `shadows=0` (no shadow maps), `post=1` (render through a post-processing
//! volume: bloom on the bright box, then a colour grade).

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::Material;
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{BloomEffect, BloomOptions, ColorGradingEffect, ColorGradingOptions};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, Canvas};

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let shadows = flag("shadows", true);
    let post = flag("post", false);

    let canvas = Canvas::find(canvas_id)?;
    // the post-processing GBuffer is not multisampled
    let config = RendererConfig { sample_count: if post { 1 } else { 4 }, clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0), ..Default::default() };
    let mut renderer = canvas.renderer(config).await;
    if shadows {
        renderer.enable_shadows(2048);
        renderer.enable_point_shadows(512, 1);
    }

    let mut scene = Scene::new();
    let mut floor = Renderable::new(PlaneGeometry::new(20.0, 20.0), Material::basic_lit("Floor", [0.6, 0.6, 0.6, 1.0], [0.2, 0.2, 0.2, 0.1]));
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.cast_shadow = false;
    scene.add(SceneNode::Renderable(floor));

    // bright enough (above 1) for the bloom to catch it with `post=1`
    let mut cube = Renderable::new(BoxGeometry::new(1.6, 1.6, 1.6), Material::basic_lit("Box", [1.6, 0.35, 0.2, 1.0], [0.5, 0.5, 0.5, 0.4]));
    cube.object.set_position(-1.2, 0.8, 0.0);
    let cube = scene.add(SceneNode::Renderable(cube));

    let mut sphere = Renderable::new(SphereGeometry::new(1.0, 48, 24), Material::basic_lit("Sphere", [0.2, 0.4, 0.9, 1.0], [0.8, 0.8, 0.8, 0.6]));
    sphere.object.set_position(1.4, 1.0, 0.4);
    scene.add(SceneNode::Renderable(sphere));

    let mut sun = DirectionalLight::new(Vec3::new(-0.4, -1.0, -0.5), Vec3::new(0.75, 0.72, 0.65), 1.0);
    sun.cast_shadow = shadows;
    scene.add(SceneNode::Light(Light::Directional(sun)));
    let mut lamp = PointLight::new(Vec3::new(-3.0, 3.5, 2.5), Vec3::new(0.7, 0.5, 0.3), 2.0, 14.0);
    lamp.cast_shadow = shadows;
    scene.add(SceneNode::Light(Light::Point(lamp)));
    scene.add(SceneNode::Light(Light::Point(PointLight::new(Vec3::new(3.5, 2.0, -2.5), Vec3::new(0.25, 0.4, 0.75), 1.5, 12.0))));

    let mut camera = Camera::new(45.0, 0.1, 100.0, canvas.aspect());
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.8, 0.0), 11.0);
    controls.set_elevation(0.35);

    let mut volume = post.then(|| {
        PostProcessingVolume::new(
            &renderer,
            vec![
                Box::new(BloomEffect::new(BloomOptions { threshold: 0.9, intensity: 0.6, ..Default::default() })),
                Box::new(ColorGradingEffect::new(ColorGradingOptions { contrast: 1.15, temperature: 0.1, ..Default::default() })),
            ],
        )
    });

    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        if let Some(r) = scene.get_renderable_mut(cube) {
            r.object.rotation.y = frame.time as f32 * 0.6;
        }
        controls.update(&mut camera, frame.dt);
        match &mut volume {
            Some(volume) => renderer.render_with_postprocessing(&mut scene, &mut camera, volume),
            None => renderer.render(&mut scene, &mut camera),
        }
    });
    Ok(())
}
