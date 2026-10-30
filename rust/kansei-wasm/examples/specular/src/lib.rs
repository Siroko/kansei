//! Specular factor: the same pieces twice, side by side under a low sun ahead of the camera. On
//! the left they have the default specular factor (1: a dielectric's Fresnel, F0 0.04 rising to
//! 1 at grazing angles, the highlight and the sheen on the board), on the right a factor of 0:
//! matte, Lambert alone (unfinished wood, chalk, cloth). Each side has a board, three balls of
//! roughness 0.35, 0.6 and 0.85 and a sloped plank. Drag to orbit, wheel or pinch to zoom.
//!
//! URL parameters: `lighting=rt` (the sun's light added by `RtShadowsEffect` from the GBuffer,
//! with ray-traced shadows, instead of the stock material's own lighting and shadow map),
//! `matte=<factor>` (the right side's factor, 0 by default).

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, Geometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{ToneMapEffect, ToneMapOptions};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::RendererConfig;
use kansei_core::rt::{RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};
use kansei_wasm::{param, param_or, Canvas};

const WOOD: [f32; 3] = [0.55, 0.3, 0.17];
const GROUND: [f32; 3] = [0.16, 0.16, 0.17];
/// The hemisphere of sky every surface takes (cd/m²): from above, and the ground's bounce.
const SKY_UP: [f32; 3] = [0.22, 0.27, 0.36];
const SKY_DOWN: [f32; 3] = [0.05, 0.045, 0.04];

/// A surface whose sun light `RtShadowsEffect` adds: its sky light only, its albedo, roughness and
/// specular factor in the GBuffer (`kansei_gbuffer_out_rt_lit`, then
/// `kansei_gbuffer_with_specular`).
fn rt_material(color: [f32; 3], roughness: f32, specular: f32) -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
{RT_SHADOWS_GBUFFER_WGSL}
struct Surface {{ albedo: vec4f, skyUp: vec4f, skyDown: vec4f }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4f;
struct VOut {{ @builtin(position) clip: vec4f, @location(0) normal: vec3f }};
@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f) -> VOut {{
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    return out;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    let n = normalize(in.normal);
    let base = surface.albedo.rgb;
    let sky = mix(surface.skyDown.rgb, surface.skyUp.rgb, n.y * 0.5 + 0.5);
    // roughness and specular factor in the albedo's and the sky's alphas
    let out = kansei_gbuffer_out_rt_lit(base * sky, vec3f(0.0), n, base, surface.albedo.a, 0.04, false);
    return kansei_gbuffer_with_specular(out, surface.skyUp.a);
}}
"#
    );
    let mut m = Material::new("RtLit", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    let ([r, g, b], [ur, ug, ub], [dr, dg, db]) = (color, SKY_UP, SKY_DOWN);
    m.set_uniform_bindable(0, "RtLit/Surface", &[r, g, b, roughness, ur, ug, ub, specular, dr, dg, db, 0.0]);
    m
}

/// The stock material lit by the sun itself (through the shadow map) and the sky.
fn standard_material(color: [f32; 3], roughness: f32, specular: f32) -> Material {
    Material::standard_lit("Standard", &StandardLitOptions { base_color: color, roughness, specular, sky_up: SKY_UP, sky_down: SKY_DOWN, ..Default::default() })
}

/// The scene: the ground, then each side's board, balls and plank, the default on the left (−x)
/// and `matte`'s factor on the right; `rt` for `RtShadowsEffect`'s materials and grid.
fn scene(rt: bool, matte: f32) -> Scene {
    let mut scene = Scene::new();
    let add = |scene: &mut Scene, geometry: Geometry, color: [f32; 3], roughness: f32, specular: f32, at: [f32; 3], tilt: f32| {
        let material = if rt { rt_material(color, roughness, specular) } else { standard_material(color, roughness, specular) };
        let mut r = Renderable::new(geometry, material).with_rt(RtSurface::new(color));
        r.object.set_position(at[0], at[1], at[2]);
        r.object.rotation.x = tilt;
        scene.add(SceneNode::Renderable(r));
    };
    add(&mut scene, BoxGeometry::new(12.0, 0.1, 12.0), GROUND, 0.9, 1.0, [0.0, -0.05, -1.0], 0.0);
    for (x, specular) in [(-1.45, 1.0), (1.45, matte)] {
        add(&mut scene, BoxGeometry::new(2.5, 0.12, 4.6), WOOD, 0.6, specular, [x, 0.06, -1.0], 0.0);
        for (k, roughness) in [0.35, 0.6, 0.85].into_iter().enumerate() {
            add(&mut scene, SphereGeometry::new(0.36, 48, 24), WOOD, roughness, specular, [x - 0.55, 0.48, 0.4 - 1.3 * k as f32], 0.0);
        }
        // a plank sloped toward the sun, its top at a grazing angle to the camera
        add(&mut scene, BoxGeometry::new(0.7, 0.08, 3.2), WOOD, 0.6, specular, [x + 0.6, 0.45, -1.0], -0.22);
    }
    // low, ahead of the camera: its light travels toward it
    let mut sun = DirectionalLight::new(Vec3::new(-0.25, -0.32, 1.0), Vec3::new(1.0, 0.95, 0.88), 3.0);
    sun.cast_shadow = !rt;
    scene.add(SceneNode::Light(Light::Directional(sun)));
    scene
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let rt = param("lighting").as_deref() == Some("rt");
    let matte = param_or("matte", 0.0f32);

    let canvas = Canvas::find(canvas_id)?;
    let config = RendererConfig { sample_count: 1, clear_color: Vec4::new(0.02, 0.02, 0.03, 1.0), ..Default::default() };
    let mut renderer = canvas.renderer(config).await;
    if rt {
        renderer.enable_rt_grid(SceneRtGridOptions {
            grid: RtGridOptions { dims: [128, 32, 128], cell: 12.0 / 128.0, fixed_origin: Some(Vec3::new(-6.0, -0.2, -7.0).to_glam()), ..Default::default() },
            ..Default::default()
        });
    } else {
        renderer.enable_shadows(2048);
    }
    let mut scene = scene(rt, matte);

    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(grid) = renderer.rt_grid() {
        let mut shadows = RtShadowsEffect::new(grid.handle());
        shadows.update_lights(scene.lights());
        effects.push(Box::new(shadows));
    }
    effects.push(Box::new(ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()))));
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(40.0, 0.1, 100.0, canvas.aspect());
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.3, -0.8), 7.5);
    controls.set_elevation(0.32);

    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        controls.update(&mut camera, frame.dt);
        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    });
    Ok(())
}
