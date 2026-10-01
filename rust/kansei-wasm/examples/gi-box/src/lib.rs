//! GI box: a Cornell box (white floor, ceiling and back wall, a red wall on the left, a green one
//! on the right, two white blocks) under one shadowed downlight, to show what the screen-space
//! global illumination adds. Without it the scene has direct light only, so the ceiling and the
//! shadows are black; with it the floor's light reaches the ceiling and the walls bleed their
//! colour onto the floor and the blocks.
//!
//! URL parameters: `gi=off|low|medium|high|ultra` (default high), `view=indirect` (only the light
//! the bounce adds, 2 stops brighter), `stats=1` (log the frame interval, which is the GPU time
//! when the browser runs without vsync).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, GiQuality, ScreenSpaceGIEffect, ScreenSpaceGIOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};

/// A diffuse surface lit by the spot lights only (no ambient), writing the normal and albedo the
/// global illumination reads (GBuffer targets 2 and 3).
const LIT_WGSL: &str = r#"
struct Surface { base_color: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = surface.base_color.rgb;
    var out: GBufferOut;
    out.color = vec4<f32>(kansei_spot_lights_radiance(in.world, n, v, base, 1.0, 0.0, in.clip.xy), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}
"#;

fn lit_material(label: &str, base_color: [f32; 3]) -> Material {
    let mut material = Material::new(
        label,
        &format!("{SPOT_LIGHTS_WGSL}\n{LIT_WGSL}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    material.set_uniform_bindable(0, label, &[base_color[0], base_color[1], base_color[2], 1.0f32]);
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
    stats: Option<(f64, u32)>,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
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

    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;
    renderer.enable_spot_shadows(2048, 1);

    // the room: 4 m wide, high and deep, open toward the camera (z = 0), walls 0.2 m thick
    let mut scene = Scene::new();
    let white = [0.73, 0.73, 0.73];
    let slabs: [(&str, [f32; 3], [f32; 3], [f32; 3]); 5] = [
        ("Floor", [4.4, 0.2, 4.2], [0.0, -0.1, -2.1], white),
        ("Ceiling", [4.4, 0.2, 4.2], [0.0, 4.1, -2.1], white),
        ("Back", [4.4, 4.4, 0.2], [0.0, 2.0, -4.1], white),
        ("Left", [0.2, 4.4, 4.2], [-2.1, 2.0, -2.1], [0.63, 0.065, 0.05]),
        ("Right", [0.2, 4.4, 4.2], [2.1, 2.0, -2.1], [0.14, 0.45, 0.09]),
    ];
    for (label, size, position, color) in slabs {
        let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), lit_material(label, color));
        slab.object.set_position(position[0], position[1], position[2]);
        scene.add(SceneNode::Renderable(slab));
    }
    // a tall block near the red wall and a short one near the green wall
    let mut tall = Renderable::new(BoxGeometry::new(1.2, 2.4, 1.2), lit_material("Tall", white));
    tall.object.set_position(-0.75, 1.2, -2.6);
    tall.object.rotation.y = 0.33;
    scene.add(SceneNode::Renderable(tall));
    let mut short = Renderable::new(BoxGeometry::new(1.2, 1.2, 1.2), lit_material("Short", white));
    short.object.set_position(0.8, 0.6, -1.5);
    short.object.rotation.y = -0.3;
    scene.add(SceneNode::Renderable(short));

    // the key light: a shadowed downlight just under the ceiling's centre, wide enough to light
    // the tops of the side walls
    let mut lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    lamp.cast_shadow = true;
    lamp.source_radius = 0.15;
    lamp.volumetric_scale = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    let quality = match query_param("gi").as_deref() {
        Some("off") => None,
        Some("low") => Some(GiQuality::Low),
        Some("medium") => Some(GiQuality::Medium),
        Some("ultra") => Some(GiQuality::Ultra),
        _ => Some(GiQuality::High),
    };
    let indirect = query_param("view").as_deref() == Some("indirect");
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(quality) = quality {
        let mut gi = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, radius_m: 4.0, ..Default::default() });
        gi.show_indirect = indirect;
        effects.push(Box::new(gi));
    }
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    // the indirect view alone is 2 stops brighter
    tone.exposure = exposure_from_ev100(5.0) * if indirect { 4.0 } else { 1.0 };
    effects.push(Box::new(ToneMapEffect::new(tone)));
    let volume = PostProcessingVolume::new(&renderer, effects);

    // framing the box's opening, as the Cornell box photographs do
    let mut camera = Camera::new(38.0, 0.1, 100.0, width as f32 / height as f32);
    camera.set_position(0.0, 2.0, 6.3);
    camera.look_at(&Vec3::new(0.0, 2.0, -2.0));
    camera.update_projection_matrix();

    log::info!("Kansei — GI box (WASM) ready: gi {quality:?}, view {}", if indirect { "indirect" } else { "lit" });

    let stats = (query_param("stats").as_deref() == Some("1")).then_some((now_ms(), 0));
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, stats }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref mut stats } = *st;
            renderer.render_with_postprocessing(scene, camera, volume);
            if let Some((start, frames)) = stats {
                *frames += 1;
                if *frames == 240 {
                    log::info!("frame interval {:.2} ms", (now_ms() - *start) / 240.0);
                    *start = now_ms();
                    *frames = 0;
                }
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
