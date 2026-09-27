//! Physically based sky (Hillaire 2020) from noon to the Midsommar intro's dusk: a clearing in a
//! ring of dark spruce proxies, with hills out to 15 km behind it, under a SkyAtmosphere with
//! aerial perspective, lit by the sun the sky is rendered with (`SkyAtmosphere::sun_illuminance_at`),
//! in physical units (lux, cd/m^2) exposed by EV100.
//! See www/index.html for the URL parameters.

mod display;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions};
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{effects::AtmosphereEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};

use display::DisplayEffect;

/// Lambertian surfaces lit by the scene's directional lights (in lux) with the renderer's
/// shadow map: radiance = albedo / pi * E * cos.
const SURFACE_WGSL: &str = r#"
struct MaterialUniforms { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> material: MaterialUniforms;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct DirLight { direction: vec3<f32>, _pad0: f32, color: vec3<f32>, intensity: f32 };
struct PtLight { position: vec3<f32>, radius: f32, color: vec3<f32>, intensity: f32 };
struct LightUniforms { num_directional: u32, num_point: u32, _pad0: u32, _pad1: u32,
                       directional: array<DirLight, 4>, point: array<PtLight, 8> };
@group(1) @binding(2) var<uniform> lights: LightUniforms;

struct ShadowUniforms { light_view_proj: mat4x4<f32>, bias: f32, normal_bias: f32, shadow_enabled: f32,
                        point_shadow_enabled: f32, point_light_pos: vec3<f32>, point_shadow_far: f32 };
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;

struct VertexInput { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
struct VertexOutput { @builtin(position) clip_position: vec4<f32>, @location(0) world_position: vec3<f32>,
                      @location(1) world_normal: vec3<f32> };

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    return out;
}

fn sun_shadow(world_pos: vec3<f32>, n: vec3<f32>) -> f32 {
    if (shadow_uniforms.shadow_enabled < 0.5) { return 1.0; }
    let ls = shadow_uniforms.light_view_proj * vec4<f32>(world_pos + n * shadow_uniforms.normal_bias, 1.0);
    let ndc = ls.xyz / ls.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    let inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0) * step(ndc.z, 1.0);
    let texel = 1.0 / vec2<f32>(textureDimensions(shadow_depth_tex));
    var s = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let suv = clamp(uv + vec2<f32>(f32(x), f32(y)) * texel, vec2<f32>(0.0), vec2<f32>(1.0));
            s += textureSampleCompare(shadow_depth_tex, shadow_sampler, suv, ndc.z - shadow_uniforms.bias);
        }
    }
    return mix(1.0, s / 9.0, inside);
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    let shadow = sun_shadow(input.world_position, n);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < lights.num_directional; i++) {
        let l = lights.directional[i];
        e += l.color * max(dot(n, -normalize(l.direction)), 0.0) * select(1.0, shadow, i == 0u);
    }
    return vec4<f32>(material.albedo.rgb / 3.14159265 * e, 1.0);
}
"#;

fn surface(label: &str, albedo: [f32; 3]) -> Material {
    let mut m = Material::new(label, SURFACE_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    m.set_uniform_bindable(0, label, &[albedo[0], albedo[1], albedo[2], 1.0]);
    m
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct Settings {
    /// Fixed sun elevation, or None for a day cycle.
    elevation: Option<f32>,
    bearing: f32,
    look: Option<f32>,
    pitch: f32,
    height: f32,
    ev: Option<f32>,
}

fn settings() -> (Settings, web_sys::UrlSearchParams) {
    let search = web_sys::window().unwrap().location().search().unwrap_or_default();
    let q = web_sys::UrlSearchParams::new_with_str(&search).unwrap();
    let num = |k: &str| q.get(k).and_then(|v| v.parse::<f32>().ok());
    let s = Settings { elevation: num("elevation"), bearing: num("bearing").unwrap_or(140.0), look: num("look"), pitch: num("pitch").unwrap_or(6.0), height: num("height").unwrap_or(1.7), ev: num("ev") };
    (s, q)
}

/// EV100 for a sun elevation, keyed to the sky this atmosphere renders (EV100 = log2(L * 100 /
/// 12.5) puts a luminance L at middle grey): clear daylight 15, twilight at -2.5 degrees ~8.
fn auto_ev100(elevation: f32) -> f32 {
    const CURVE: [(f32, f32); 8] = [(-8.0, 5.0), (-4.0, 7.3), (-2.5, 8.2), (0.0, 9.8), (2.0, 11.2), (6.0, 12.8), (15.0, 14.3), (40.0, 15.0)];
    if elevation <= CURVE[0].0 {
        return CURVE[0].1;
    }
    for w in CURVE.windows(2) {
        let ((e0, v0), (e1, v1)) = (w[0], w[1]);
        if elevation <= e1 {
            return v0 + (v1 - v0) * (elevation - e0) / (e1 - e0);
        }
    }
    CURVE[CURVE.len() - 1].1
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    sky: SkyAtmosphere,
    volume: PostProcessingVolume,
    sun_light: usize,
    settings: Settings,
    start: f64,
}

fn now_secs() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

impl State {
    fn frame(&mut self, t: f32) {
        let s = &self.settings;
        let elevation = s.elevation.unwrap_or_else(|| 12.5 + 20.0 * (t * std::f32::consts::TAU / 60.0).cos());
        let sun_dir = direction_from_elevation_bearing(elevation, s.bearing);
        self.sky.sun.direction = sun_dir;

        let look = s.look.unwrap_or(s.bearing);
        let eye = Vec3::new(0.0, s.height, 0.0);
        let d = direction_from_elevation_bearing(s.pitch, look);
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::new(eye.x + d.x, eye.y + d.y, eye.z + d.z));

        // the sun light the sky is rendered with
        if let Some(Light::Directional(l)) = self.scene.get_light_mut(self.sun_light) {
            l.direction = Vec3::new(-sun_dir.x, -sun_dir.y, -sun_dir.z);
            l.color = self.sky.sun_illuminance_at(eye);
            l.intensity = 1.0;
        }
        if let Some(display) = self.volume.effects[1].as_any_mut().downcast_mut::<DisplayEffect>() {
            display.ev100 = s.ev.unwrap_or_else(|| auto_ev100(elevation));
        }

        self.sky.update(self.renderer.device(), self.renderer.queue(), &mut self.camera);
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let (width, height) = (canvas.client_width().max(1) as u32, canvas.client_height().max(1) as u32);
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas).await;
    renderer.enable_shadows(2048);
    let encode_srgb = !renderer.presentation_format().is_srgb();

    let (settings, q) = settings();
    let mut sky = SkyAtmosphere::new(renderer.device(), SkyAtmosphereOptions::default());
    sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);
    if let Some(haze) = q.get("haze").and_then(|v| v.parse::<f32>().ok()) {
        sky.params.mie_scattering_scale *= haze;
    }
    if let Some(ozone) = q.get("ozone").and_then(|v| v.parse::<f32>().ok()) {
        sky.params.other_absorption_scale *= ozone;
    }
    if q.get("moon").is_some() {
        sky.moon.direction = direction_from_elevation_bearing(18.0, settings.bearing + 150.0);
        sky.moon.illuminance = Vec3::new(0.25, 0.25, 0.25);
    }

    let mut scene = Scene::new();
    let mut ground = Renderable::new(PlaneGeometry::new(40000.0, 40000.0), surface("Ground", [0.08, 0.1, 0.06]));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // a clearing in a ring of spruce-sized proxies, jittered so the treeline is ragged
    let hash = |i: u32| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    for i in 0..220u32 {
        let a = i as f32 / 220.0 * std::f32::consts::TAU + hash(i) * 0.02;
        let r = 55.0 + hash(i + 1000) * 70.0;
        let h = 12.0 + hash(i + 2000) * 14.0;
        let w = 1.5 + hash(i + 3000) * 2.5;
        let mut tree = Renderable::new(BoxGeometry::new(w, h, w), surface("Spruce", [0.04, 0.06, 0.035]));
        tree.object.set_position(a.sin() * r, h * 0.5, -a.cos() * r);
        tree.object.rotation.y = hash(i + 4000) * 3.0;
        scene.add(SceneNode::Renderable(tree));
    }
    let mut stone = Renderable::new(BoxGeometry::new(2.0, 1.2, 1.4), surface("Stone", [0.3, 0.3, 0.28]));
    let d = direction_from_elevation_bearing(0.0, settings.look.unwrap_or(settings.bearing));
    stone.object.set_position(d.x * 9.0 - 1.5, 0.6, d.z * 9.0);
    stone.object.rotation.y = 0.6;
    scene.add(SceneNode::Renderable(stone));

    // hills from 1.5 to 15 km away, half buried ellipsoids: aerial perspective fades them into the sky
    for i in 0..28u32 {
        let a = i as f32 / 28.0 * std::f32::consts::TAU + hash(i + 5000) * 0.2;
        let r = 1500.0 * 10f32.powf(hash(i + 6000));
        let (rx, ry) = (r * (0.12 + hash(i + 7000) * 0.2), r * (0.02 + hash(i + 8000) * 0.03));
        let mut hill = Renderable::new(SphereGeometry::new(1.0, 48, 24), surface("Hill", [0.05, 0.08, 0.045]));
        hill.object.set_position(a.sin() * r, 0.0, -a.cos() * r);
        hill.object.scale = Vec3::new(rx, ry, rx * 0.6);
        hill.object.rotation.y = a;
        scene.add(SceneNode::Renderable(hill));
    }

    let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 1.0);
    sun.cast_shadow = true;
    let sun_light = scene.add(SceneNode::Light(Light::Directional(sun)));

    let volume = PostProcessingVolume::new(&renderer, vec![Box::new(AtmosphereEffect::new(&sky)), Box::new(DisplayEffect::new(encode_srgb))]);
    let camera = Camera::new(62.0, 0.5, 60000.0, width as f32 / height as f32);

    log::info!("Kansei — Sky Atmosphere (WASM) ready, {:?}", renderer.presentation_format());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, sky, volume, sun_light, settings, start: now_secs() }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let t = (now_secs() - st.start) as f32;
            st.frame(t);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
