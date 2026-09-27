//! Physical depth of field: a forest-like depth set (trunks from 3 to 120 m, alpha-tested leaf
//! cards near and far, strings of small lights behind the subject) under a low sun, seen through
//! a CameraLens on Unreal's 23.76 mm filmback. The circle of confusion follows from the focal
//! length, f-stop and focus distance; bokeh keep their energy and the aperture's shape. See
//! www/index.html for the URL parameters.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions, SKY_LIGHTING_WGSL};
use kansei_core::buffers::{BufferType, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100, AtmosphereEffect, CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, DofDebugView, HighlightOptions,
    TemporalAAEffect,
    TemporalAAOptions, ToneMapEffect, ToneMapOptions,
};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::{Renderer, RendererConfig};

/// Unreal's default filmback width, mm.
const SENSOR_MM: f32 = 23.76;

/// Shared by the materials: camera, lights, shadows, and Lambertian lighting by the sun (with the
/// renderer's shadow map) and the sky (SkyLighting SH). Prefixed with SKY_LIGHTING_WGSL.
const COMMON_WGSL: &str = r#"
@group(0) @binding(1) var<uniform> sky: SkyLighting;
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
                      @location(1) world_normal: vec3<f32>, @location(2) uv: vec2<f32> };

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_position = world_pos.xyz;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    out.uv = input.uv;
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

// Lambertian: albedo / pi * (sun * cos * shadow + sky irradiance)
fn lambert(albedo: vec3<f32>, world_pos: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let shadow = sun_shadow(world_pos, n);
    var e = skyIrradiance(sky, n);
    for (var i = 0u; i < lights.num_directional; i++) {
        let l = lights.directional[i];
        e += l.color * max(dot(n, -normalize(l.direction)), 0.0) * select(1.0, shadow, i == 0u);
    }
    return albedo / 3.14159265 * e;
}
"#;

const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> material: Surface;
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(lambert(material.albedo.rgb, input.world_position, normalize(input.world_normal)), 1.0);
}
"#;

/// Alpha-tested leaf cards: a grid of rotated elliptical leaves cut out of the card with discard,
/// lit from both sides.
const LEAVES_WGSL: &str = r#"
struct Surface { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> material: Surface;
fn hash(p: vec2<f32>) -> f32 { return fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453); }
@fragment
fn fragment_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let cells = 6.0;
    let g = input.uv * cells;
    let cell = floor(g);
    var covered = false;
    // leaves overlap their neighbours: test this cell and the ones around it
    for (var dy = -1.0; dy <= 1.0; dy += 1.0) {
        for (var dx = -1.0; dx <= 1.0; dx += 1.0) {
            let c = cell + vec2<f32>(dx, dy);
            let a = hash(c) * 6.2831;
            let centre = c + 0.5 + (vec2<f32>(hash(c + 7.0), hash(c + 13.0)) - 0.5) * 0.6;
            var q = g - centre;
            q = vec2<f32>(cos(a) * q.x + sin(a) * q.y, -sin(a) * q.x + cos(a) * q.y);
            if ((q.x * q.x) / 0.36 + (q.y * q.y) / 0.08 < 1.0 && hash(c + 3.0) > 0.25) { covered = true; }
        }
    }
    if (!covered) { discard; }
    var n = normalize(input.world_normal);
    if (!front) { n = -n; }
    let tint = 0.8 + 0.4 * hash(cell + 21.0);
    return vec4<f32>(lambert(material.albedo.rgb * tint, input.world_position, n), 1.0);
}
"#;

/// Small lights: constant luminance (cd/m^2).
const EMISSIVE_WGSL: &str = r#"
struct Surface { luminance: vec4<f32> };
@group(0) @binding(0) var<uniform> material: Surface;
@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(material.luminance.rgb, 1.0);
}
"#;

fn material(label: &str, body: &str, rgb: [f32; 3], sky: &SkyAtmosphere, cull: CullMode) -> Material {
    let mut m = Material::new(
        label,
        &format!("{SKY_LIGHTING_WGSL}\n{COMMON_WGSL}\n{body}"),
        vec![Binding::uniform(0, ShaderStages::FRAGMENT), Binding::uniform(1, ShaderStages::FRAGMENT)],
        MaterialOptions { cull_mode: cull, ..Default::default() },
    );
    m.set_uniform_bindable(0, label, &[rgb[0], rgb[1], rgb[2], 1.0]);
    m.set_bindable(1, ComputeBuffer::from_external("SkyLighting", sky.bindings().sky_lighting.clone(), BufferType::Uniform));
    m
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct Settings {
    focal_mm: f32,
    f_stop: f32,
    focus_m: f32,
    blades: u32,
    rack: bool,
    ev: f32,
    off: bool,
    taa: bool,
    dof_after_taa: bool,
    /// DofDebugView: 1 background, 2 near layer, 3 near alpha, 4 CoC.
    debug: u32,
    scatter: bool,
    samples: u32,
    /// Render scale (the TAA upscales to the canvas).
    scale: f32,
    /// Fixed time for the rack focus, seconds (screenshots).
    time: Option<f32>,
}

fn settings() -> Settings {
    let search = web_sys::window().unwrap().location().search().unwrap_or_default();
    let q = web_sys::UrlSearchParams::new_with_str(&search).unwrap();
    let num = |k: &str| q.get(k).and_then(|v| v.parse::<f32>().ok());
    Settings {
        focal_mm: num("focal").unwrap_or(50.0),
        f_stop: num("fstop").unwrap_or(1.8),
        focus_m: num("focus").unwrap_or(8.0),
        blades: num("blades").unwrap_or(0.0) as u32,
        rack: q.get("rack").is_some(),
        ev: num("ev").unwrap_or(10.8),
        off: q.get("dof").as_deref() == Some("0"),
        taa: q.get("taa").as_deref() != Some("0"),
        dof_after_taa: q.get("order").as_deref() == Some("after"),
        debug: num("debug").unwrap_or(0.0) as u32,
        scatter: q.get("scatter").as_deref() != Some("0"),
        samples: num("samples").unwrap_or(72.0) as u32,
        scale: num("scale").unwrap_or(1.0),
        time: num("t"),
    }
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
        // a rack focus between the leaves at 2 m and the trunks at 30 m, or a fixed focus
        let focus = if s.rack { 2.0 * 15f32.powf(0.5 - 0.5 * (t * std::f32::consts::TAU / 8.0).cos()) } else { s.focus_m };
        for effect in &mut self.volume.effects {
            if let Some(dof) = effect.as_any_mut().downcast_mut::<CinematicDepthOfFieldEffect>() {
                dof.lens.focus_distance_m = focus;
            }
        }
        let eye = Vec3::new(0.0, 1.5, 0.0);
        self.camera.set_position(eye.x, eye.y, eye.z);
        self.camera.look_at(&Vec3::new(0.0, 1.35, -10.0));
        if let Some(Light::Directional(l)) = self.scene.get_light_mut(self.sun_light) {
            let d = self.sky.sun.direction;
            l.direction = Vec3::new(-d.x, -d.y, -d.z);
            l.color = self.sky.sun_illuminance_at(eye);
            l.intensity = 1.0;
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
    let settings = settings();
    renderer.set_render_scale(settings.scale);

    // a low sun ahead, backlighting the set
    let mut sky = SkyAtmosphere::new(renderer.device(), SkyAtmosphereOptions::default());
    sky.sun.direction = direction_from_elevation_bearing(4.0, 12.0);
    sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);

    let mut scene = Scene::new();
    let hash = |i: u32| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    let mut ground = Renderable::new(PlaneGeometry::new(4000.0, 4000.0), material("Ground", SURFACE_WGSL, [0.07, 0.08, 0.05], &sky, CullMode::Back));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // trunks from 12 to 120 m, and a few wide of the lens nearer in
    for i in 0..70u32 {
        let z = -14.0 - 106.0 * hash(i).powf(1.2);
        let x = (hash(i + 500) - 0.5) * (8.0 + (-z) * 1.6);
        let (w, h) = (0.25 + hash(i + 900) * 0.35, 9.0 + hash(i + 1300) * 8.0);
        let mut trunk = Renderable::new(BoxGeometry::new(w, h, w), material("Trunk", SURFACE_WGSL, [0.16, 0.12, 0.09], &sky, CullMode::Back));
        trunk.object.set_position(x, h * 0.5, z);
        trunk.object.rotation.y = hash(i + 1700) * 3.0;
        scene.add(SceneNode::Renderable(trunk));
    }
    for (x, z) in [(-3.2f32, -6.0f32), (3.6, -9.0), (-4.4, -10.5)] {
        let mut trunk = Renderable::new(BoxGeometry::new(0.45, 14.0, 0.45), material("Trunk", SURFACE_WGSL, [0.16, 0.12, 0.09], &sky, CullMode::Back));
        trunk.object.set_position(x, 7.0, z);
        scene.add(SceneNode::Renderable(trunk));
    }

    // the subject on the 8 m focus plane, leaves beside it, leaves close to the lens and far off
    let mut subject = Renderable::new(BoxGeometry::new(0.7, 1.3, 0.7), material("Subject", SURFACE_WGSL, [0.5, 0.45, 0.4], &sky, CullMode::Back));
    subject.object.set_position(-0.55, 0.65, -8.0);
    subject.object.rotation.y = 0.5;
    scene.add(SceneNode::Renderable(subject));
    let leaves = [(0.3, 1.3, -1.5, 0.45), (-0.34, 1.72, -1.8, 0.5), (1.0, 1.4, -8.2, 1.4), (-1.9, 1.9, -8.8, 1.4), (-2.5, 2.0, -20.0, 2.5), (3.2, 1.6, -30.0, 3.0)];
    for (i, &(x, y, z, size)) in leaves.iter().enumerate() {
        let mut card = Renderable::new(PlaneGeometry::new(size, size), material("Leaves", LEAVES_WGSL, [0.12, 0.22, 0.06], &sky, CullMode::None));
        card.object.set_position(x, y, z);
        card.object.rotation.y = (hash(i as u32 + 40) - 0.5) * 0.8;
        card.object.rotation.z = hash(i as u32 + 60) * 3.0;
        scene.add(SceneNode::Renderable(card));
    }

    // a string of small warm lights sagging between 25 and 40 m, and a few close to the lens
    for i in 0..48u32 {
        let u = i as f32 / 47.0;
        let x = (u - 0.5) * 26.0;
        let z = -25.0 - 15.0 * (u * 3.0).sin().abs();
        let y = 1.5 - 1.0 * (1.0 - (2.0 * u - 1.0).powi(2)) + 0.2 * hash(i + 2200);
        let mut light = Renderable::new(SphereGeometry::new(0.08, 8, 6), material("Light", EMISSIVE_WGSL, [20000.0, 11000.0, 4000.0], &sky, CullMode::Back));
        light.object.set_position(x, y, z);
        scene.add(SceneNode::Renderable(light));
    }
    for (i, &(x, y)) in [(-0.22, 1.28), (0.18, 1.62), (-0.05, 1.18)].iter().enumerate() {
        let mut light = Renderable::new(SphereGeometry::new(0.006, 8, 6), material("NearLight", EMISSIVE_WGSL, [5000.0, 6000.0, 8000.0], &sky, CullMode::Back));
        light.object.set_position(x, y, -1.2 - 0.25 * i as f32);
        scene.add(SceneNode::Renderable(light));
    }

    let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 1.0);
    sun.cast_shadow = true;
    let sun_light = scene.add(SceneNode::Light(Light::Directional(sun)));

    // the camera's field of view comes from the lens on the filmback
    let aspect = width as f32 / height as f32;
    let hfov = 2.0 * (SENSOR_MM / (2.0 * settings.focal_mm)).atan();
    let vfov = 2.0 * ((hfov * 0.5).tan() / aspect).atan();
    let camera = Camera::new(vfov.to_degrees(), 0.1, 4000.0, aspect);

    // the chain: sky and aerial perspective, depth of field on the jittered frame (each pixel's
    // colour and depth still agree), TAA resolving both, then exposure and tonemapping.
    // order=after puts the DoF after TAA instead, for comparison.
    // The alpha and CoC debug views are [0, 1] values rather than radiance: show them unexposed.
    let exposure = if matches!(settings.debug, 3 | 4) { 1.0 } else { exposure_from_ev100(settings.ev) };
    let tonemap = ToneMapEffect::new(ToneMapOptions { exposure, ..ToneMapOptions::for_surface(renderer.presentation_format()) });
    let mut effects: Vec<Box<dyn kansei_core::postprocessing::PostProcessingEffect>> = vec![Box::new(AtmosphereEffect::new(&sky))];
    let taa = || Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() }));
    if settings.taa && settings.dof_after_taa {
        effects.push(taa());
    }
    if !settings.off {
        let mut dof = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
            lens: CameraLens {
                focal_length_mm: None, // from the camera: the blur matches the picture
                f_stop: settings.f_stop,
                focus_distance_m: settings.focus_m,
                sensor_width_mm: SENSOR_MM,
                blade_count: settings.blades,
                blade_rotation_deg: 15.0,
            },
            sample_count: settings.samples,
            highlights: HighlightOptions { enabled: settings.scatter, ..Default::default() },
            // resolved by the TAA after it
            temporal_noise: settings.taa && !settings.dof_after_taa,
            ..Default::default()
        });
        dof.debug_view = match settings.debug {
            1 => DofDebugView::Background,
            2 => DofDebugView::Near,
            3 => DofDebugView::NearAlpha,
            4 => DofDebugView::Coc,
            _ => DofDebugView::None,
        };
        effects.push(Box::new(dof));
    }
    if settings.taa && !settings.dof_after_taa {
        effects.push(taa());
    }
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    log::info!("Kansei — Depth of Field (WASM) ready, {:?}", renderer.presentation_format());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, sky, volume, sun_light, settings, start: now_secs() }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let t = st.settings.time.unwrap_or((now_secs() - st.start) as f32);
            st.frame(t);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
