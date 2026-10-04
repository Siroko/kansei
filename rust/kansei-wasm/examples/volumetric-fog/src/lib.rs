//! Froxel volumetric fog: a grove of pillars under a low sun (shadowed, so its light comes
//! through in shafts), a warm lamp with cube shadows and a cool one without, and a dim ambient
//! sky term, rendered through the GBuffer path with VolumetricFogEffect.

use wasm_bindgen::prelude::*;

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
use kansei_core::renderers::RendererConfig;
use kansei_wasm::Canvas;

const LIT_WGSL: &str = include_str!("../../../../kansei-core/src/shaders/basic_lit.wgsl");

fn lit_material(label: &str, color: [f32; 4], specular: [f32; 4]) -> Material {
    let data: [f32; 8] = [color[0], color[1], color[2], color[3], specular[0], specular[1], specular[2], specular[3]];
    let mut material = Material::new(label, LIT_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    material.set_uniform_bindable(0, label, &data);
    material
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    // sample_count 1: the post-processing GBuffer is not multisampled
    let mut renderer = canvas
        .renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.01, 0.012, 0.02, 1.0), ..Default::default() })
        .await;
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
    let mut volume = PostProcessingVolume::new(&renderer, vec![Box::new(fog)]);

    let mut camera = Camera::new(50.0, 0.1, 200.0, canvas.aspect());
    camera.update_projection_matrix();

    log::info!("Kansei — Volumetric Fog (WASM) ready");

    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let t = frame.time as f32;

        if let Some(fog) = volume.effects[0].as_any_mut().downcast_mut::<VolumetricFogEffect>() {
            fog.time = t;
        }

        // start on the far side of the grove from the sun, looking into the shafts
        let angle = -2.08 + t * 0.06;
        camera.set_position(angle.sin() * 22.0, 4.0, angle.cos() * 22.0);
        camera.look_at(&Vec3::new(0.0, 3.0, 0.0));

        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    });
    Ok(())
}
