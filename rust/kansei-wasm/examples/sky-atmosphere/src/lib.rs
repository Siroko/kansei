//! Physically based sky (Hillaire 2020) from noon to the Midsommar intro's dusk: a clearing in a
//! ring of dark spruce proxies, with hills out to 15 km behind it, under a SkyAtmosphere with
//! aerial perspective. Surfaces are lit by the sun the sky is rendered with
//! (`SkyAtmosphere::sun_illuminance_at`) and by the sky itself (`SkyLighting` SH), in physical units
//! (lux, cd/m^2) exposed by EV100. With `fog=`, froxel height fog lit by the same sun and sky
//! lies in front of the atmosphere; with `mist=1`, a local fog volume fills the clearing. A chrome
//! ball, a rough metal ball and a pond reflect the sky through the prefiltered environment cubemap.
//! See www/index.html for the URL parameters.

mod display;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::atmosphere::{
    direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions, CLOUD_SHADOW_WGSL, SKY_ENVIRONMENT_WGSL, SKY_LIGHTING_WGSL,
};
use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::postprocessing::effects::{
    AtmosphereEffect, CloudLayer, GiQuality, HeightFogEffect, HeightFogLayer, LocalFogVolume, ScreenSpaceGIEffect, ScreenSpaceGIOptions,
    VolumetricCloudsEffect, VolumetricCloudsOptions, VolumetricFogEffect, VolumetricFogOptions,
};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::{Renderer, RendererConfig};

use display::DisplayEffect;

/// Lambertian surfaces lit by the scene's directional lights (in lux) with the renderer's shadow
/// map, and by the sky: radiance = albedo / pi * (E_sun * cos * shadow + E_sky(n)). Prefixed with
/// SKY_LIGHTING_WGSL.
const SURFACE_WGSL: &str = r#"
struct MaterialUniforms { albedo: vec4<f32> };
@group(0) @binding(0) var<uniform> material: MaterialUniforms;
@group(0) @binding(1) var<uniform> sky: SkyLighting;
@group(0) @binding(2) var cloud_shadow_map: texture_2d<f32>;
@group(0) @binding(3) var cloud_shadow_sampler: sampler;
@group(0) @binding(4) var<uniform> cloud_shadow_params: CloudShadowParams;
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

// the lit colour, and the normal and albedo the screen-space GI reads (GBuffer targets 2 and 3)
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fragment_main(input: VertexOutput) -> GBufferOut {
    let n = normalize(input.world_normal);
    let shadow = sun_shadow(input.world_position, n);
    var e = vec3<f32>(0.0);
    for (var i = 0u; i < lights.num_directional; i++) {
        let l = lights.directional[i];
        // the sun (light 0) through its shadow map and the clouds
        let clouds = cloudShadow(cloud_shadow_map, cloud_shadow_sampler, cloud_shadow_params, input.world_position);
        e += l.color * max(dot(n, -normalize(l.direction)), 0.0) * select(1.0, shadow * clouds, i == 0u);
    }
    var out: GBufferOut;
    out.color = vec4<f32>(material.albedo.rgb / 3.14159265 * (e + skyIrradiance(sky, n)), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(material.albedo.rgb, 1.0);
    return out;
}
"#;

/// Reflective surfaces: the sky's prefiltered environment (split sum) and a GGX sun highlight
/// over a diffuse base. Prefixed with SKY_LIGHTING_WGSL and SKY_ENVIRONMENT_WGSL.
const REFLECTIVE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32>, f0_roughness: vec4<f32> };
@group(0) @binding(0) var<uniform> material: Surface;
@group(0) @binding(1) var<uniform> sky: SkyLighting;
@group(0) @binding(2) var env: texture_cube<f32>;
@group(0) @binding(3) var env_sampler: sampler;
@group(0) @binding(4) var cloud_shadow_map: texture_2d<f32>;
@group(0) @binding(5) var cloud_shadow_sampler: sampler;
@group(0) @binding(6) var<uniform> cloud_shadow_params: CloudShadowParams;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct DirLight { direction: vec3<f32>, _pad0: f32, color: vec3<f32>, intensity: f32 };
struct PtLight { position: vec3<f32>, radius: f32, color: vec3<f32>, intensity: f32 };
struct LightUniforms { num_directional: u32, num_point: u32, _pad0: u32, _pad1: u32,
                       directional: array<DirLight, 4>, point: array<PtLight, 8> };
@group(1) @binding(2) var<uniform> lights: LightUniforms;

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

struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@fragment
fn fragment_main(input: VertexOutput) -> GBufferOut {
    let n = normalize(input.world_normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - input.world_position);
    let nv = max(dot(n, v), 1e-3);
    let f0 = material.f0_roughness.rgb;
    let roughness = max(material.f0_roughness.a, 0.02);

    // the sky: diffuse from the SH, specular from the prefiltered cubemap (split sum)
    let spec_brdf = skyEnvironmentBrdf(f0, roughness, nv);
    var color = material.albedo.rgb / 3.14159265 * skyIrradiance(sky, n) * (1.0 - spec_brdf)
              + skyEnvironment(env, env_sampler, reflect(-v, n), roughness) * spec_brdf;

    // the sun: Lambert + GGX with Schlick's Fresnel and a Smith visibility approximation
    let a2 = roughness * roughness * roughness * roughness;
    for (var i = 0u; i < lights.num_directional; i++) {
        let l = -normalize(lights.directional[i].direction);
        let nl = max(dot(n, l), 0.0);
        let h = normalize(l + v);
        let nh = max(dot(n, h), 0.0);
        let d = a2 / (3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
        let f = f0 + (1.0 - f0) * pow(1.0 - max(dot(v, h), 0.0), 5.0);
        let k = roughness * roughness * 0.5;
        let vis = 0.25 / ((nl * (1.0 - k) + k) * (nv * (1.0 - k) + k));
        let clouds = select(1.0, cloudShadow(cloud_shadow_map, cloud_shadow_sampler, cloud_shadow_params, input.world_position), i == 0u);
        color += lights.directional[i].color * (nl * clouds) * (material.albedo.rgb / 3.14159265 * (1.0 - f) + d * f * vis);
    }
    var out: GBufferOut;
    out.color = vec4<f32>(color, 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(material.albedo.rgb * (1.0 - spec_brdf), 1.0);
    return out;
}
"#;

fn reflective(label: &str, albedo: [f32; 3], f0: [f32; 3], roughness: f32, sky: &SkyAtmosphere) -> Material {
    let mut m = Material::new(
        label,
        &format!("{SKY_LIGHTING_WGSL}\n{SKY_ENVIRONMENT_WGSL}\n{CLOUD_SHADOW_WGSL}\n{REFLECTIVE_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::FRAGMENT),
            Binding::uniform(1, ShaderStages::FRAGMENT),
            Binding::texture_cube(2, ShaderStages::FRAGMENT),
            Binding::sampler(3, ShaderStages::FRAGMENT),
            Binding::texture_2d(4, ShaderStages::FRAGMENT),
            Binding::sampler(5, ShaderStages::FRAGMENT),
            Binding::uniform(6, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    m.set_uniform_bindable(0, label, &[albedo[0], albedo[1], albedo[2], 1.0, f0[0], f0[1], f0[2], roughness]);
    m.set_bindable(1, ComputeBuffer::from_external("SkyLighting", sky.bindings().sky_lighting.clone(), BufferType::Uniform));
    m.set_bindable(2, Texture::from_view("SkyEnvironment", sky.environment_texture().clone(), sky.bindings().environment.clone()));
    m.set_bindable(3, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    bind_cloud_shadow(&mut m, 4, sky);
    m
}

/// The clouds' shadow (CLOUD_SHADOW_WGSL) at bindings `first` (map), +1 (sampler), +2 (params).
fn bind_cloud_shadow(m: &mut Material, first: u32, sky: &SkyAtmosphere) {
    let b = sky.bindings();
    m.set_bindable(first, Texture::from_view("CloudShadow", sky.cloud_shadow_texture().clone(), b.cloud_shadow.clone()));
    m.set_bindable(first + 1, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    m.set_bindable(first + 2, ComputeBuffer::from_external("CloudShadowParams", b.cloud_shadow_params.clone(), BufferType::Uniform));
}

fn surface(label: &str, albedo: [f32; 3], sky: &SkyAtmosphere) -> Material {
    let mut m = Material::new(
        label,
        &format!("{SKY_LIGHTING_WGSL}\n{CLOUD_SHADOW_WGSL}\n{SURFACE_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::FRAGMENT),
            Binding::uniform(1, ShaderStages::FRAGMENT),
            Binding::texture_2d(2, ShaderStages::FRAGMENT),
            Binding::sampler(3, ShaderStages::FRAGMENT),
            Binding::uniform(4, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    m.set_uniform_bindable(0, label, &[albedo[0], albedo[1], albedo[2], 1.0]);
    m.set_bindable(1, ComputeBuffer::from_external("SkyLighting", sky.bindings().sky_lighting.clone(), BufferType::Uniform));
    bind_cloud_shadow(&mut m, 2, sky);
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
    // preset=midsommar: the Unreal intro's light block (sun 2.5 degrees down, EV100 3.9)
    let midsommar = q.get("preset").as_deref() == Some("midsommar");
    let s = Settings {
        elevation: num("elevation").or(midsommar.then_some(-2.5)),
        bearing: num("bearing").unwrap_or(140.0),
        look: num("look"),
        pitch: num("pitch").unwrap_or(6.0),
        height: num("height").unwrap_or(1.7),
        ev: num("ev").or(midsommar.then_some(3.9)),
    };
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
        let ev100 = s.ev.unwrap_or_else(|| auto_ev100(elevation));
        for effect in &mut self.volume.effects {
            if let Some(display) = effect.as_any_mut().downcast_mut::<DisplayEffect>() {
                display.ev100 = ev100;
            } else if let Some(fog) = effect.as_any_mut().downcast_mut::<VolumetricFogEffect>() {
                fog.update_lights(self.scene.lights());
                fog.time = t;
            } else if let Some(clouds) = effect.as_any_mut().downcast_mut::<VolumetricCloudsEffect>() {
                clouds.time = t;
            }
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
    let mut ground = Renderable::new(PlaneGeometry::new(40000.0, 40000.0), surface("Ground", [0.08, 0.1, 0.06], &sky));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // a clearing in a ring of spruce-sized proxies, jittered so the treeline is ragged
    let hash = |i: u32| ((i.wrapping_mul(2654435761) >> 8) & 0xffff) as f32 / 65535.0;
    for i in 0..220u32 {
        let a = i as f32 / 220.0 * std::f32::consts::TAU + hash(i) * 0.02;
        let r = 55.0 + hash(i + 1000) * 70.0;
        let h = 12.0 + hash(i + 2000) * 14.0;
        let w = 1.5 + hash(i + 3000) * 2.5;
        let mut tree = Renderable::new(BoxGeometry::new(w, h, w), surface("Spruce", [0.04, 0.06, 0.035], &sky));
        tree.object.set_position(a.sin() * r, h * 0.5, -a.cos() * r);
        tree.object.rotation.y = hash(i + 4000) * 3.0;
        scene.add(SceneNode::Renderable(tree));
    }
    let mut stone = Renderable::new(BoxGeometry::new(2.0, 1.2, 1.4), surface("Stone", [0.3, 0.3, 0.28], &sky));
    let d = direction_from_elevation_bearing(0.0, settings.look.unwrap_or(settings.bearing));
    stone.object.set_position(d.x * 9.0 - 1.5, 0.6, d.z * 9.0);
    stone.object.rotation.y = 0.6;
    scene.add(SceneNode::Renderable(stone));

    // sky reflections: a chrome ball, a rough metal ball and a still pond
    let side = Vec3::new(-d.z, 0.0, d.x);
    let place = |ahead: f32, across: f32| (d.x * ahead + side.x * across, d.z * ahead + side.z * across);
    let (x, z) = place(7.0, 2.2);
    let mut chrome = Renderable::new(SphereGeometry::new(0.8, 64, 32), reflective("Chrome", [0.0, 0.0, 0.0], [0.95, 0.93, 0.88], 0.03, &sky));
    chrome.object.set_position(x, 0.8, z);
    scene.add(SceneNode::Renderable(chrome));
    let (x, z) = place(8.0, 4.2);
    let mut rough = Renderable::new(SphereGeometry::new(0.8, 64, 32), reflective("RoughMetal", [0.0, 0.0, 0.0], [0.9, 0.7, 0.45], 0.45, &sky));
    rough.object.set_position(x, 0.8, z);
    scene.add(SceneNode::Renderable(rough));
    let (x, z) = place(16.0, -1.0);
    let mut pond = Renderable::new(PlaneGeometry::new(16.0, 9.0), reflective("Pond", [0.01, 0.012, 0.012], [0.02, 0.02, 0.02], 0.04, &sky));
    pond.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    pond.object.set_position(x, 0.03, z);
    scene.add(SceneNode::Renderable(pond));

    // hills from 1.5 to 15 km away, half buried ellipsoids: aerial perspective fades them into the sky
    for i in 0..28u32 {
        let a = i as f32 / 28.0 * std::f32::consts::TAU + hash(i + 5000) * 0.2;
        let r = 1500.0 * 10f32.powf(hash(i + 6000));
        let (rx, ry) = (r * (0.12 + hash(i + 7000) * 0.2), r * (0.02 + hash(i + 8000) * 0.03));
        let mut hill = Renderable::new(SphereGeometry::new(1.0, 48, 24), surface("Hill", [0.05, 0.08, 0.045], &sky));
        hill.object.set_position(a.sin() * r, 0.0, -a.cos() * r);
        hill.object.scale = Vec3::new(rx, ry, rx * 0.6);
        hill.object.rotation.y = a;
        scene.add(SceneNode::Renderable(hill));
    }

    let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 1.0);
    sun.cast_shadow = true;
    let sun_light = scene.add(SceneNode::Light(Light::Directional(sun)));

    // the chain: the sky and its aerial perspective, the fog in front of them, the display transform
    // gi=low|medium|high|ultra: screen-space global illumination (off by default), first in the
    // chain so the bounce lies on the surfaces under the aerial perspective
    let mut effects: Vec<Box<dyn kansei_core::postprocessing::PostProcessingEffect>> = Vec::new();
    let quality = match q.get("gi").as_deref() {
        Some("low") => Some(GiQuality::Low),
        Some("medium") | Some("1") => Some(GiQuality::Medium),
        Some("high") => Some(GiQuality::High),
        Some("ultra") => Some(GiQuality::Ultra),
        _ => None,
    };
    if let Some(quality) = quality {
        let mut gi = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, ..Default::default() });
        gi.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        effects.push(Box::new(gi));
    }
    effects.push(Box::new(AtmosphereEffect::new(&sky)));
    // clouds=<coverage 0..1> (clouds=0 none), cloudtype=<0 stratus .. 1 cumulus>, cloudbase=<m>,
    // cloudshadows=0 (no cloud shadows on the scene)
    let clouds = q.get("clouds").and_then(|v| v.parse::<f32>().ok()).unwrap_or(0.45);
    if clouds > 0.0 {
        let num = |k: &str, d: f32| q.get(k).and_then(|v| v.parse::<f32>().ok()).unwrap_or(d);
        let base = num("cloudbase", 1500.0);
        let layer = CloudLayer { coverage: clouds, cloud_type: num("cloudtype", 0.7), bottom_m: base, top_m: base + num("cloudthick", 2500.0), ..Default::default() };
        let mut fx = VolumetricCloudsEffect::new(&sky, VolumetricCloudsOptions { layer, ..Default::default() });
        // cloudshadows=0: the clouds cast no shadows on the scene
        fx.casts_shadows = q.get("cloudshadows").as_deref() != Some("0");
        effects.push(Box::new(fx));
    }
    let midsommar = q.get("preset").as_deref() == Some("midsommar");
    if midsommar {
        // intro_scene.json's light block, as create_intro_scene.py applies it in Unreal
        sky.params.mie_scattering_scale = 0.003996 * 1.7; // haze
        sky.params.other_absorption_scale = 0.8; // ozone, set absolute by the script
        // 100 000 lux, light colour (255, 222, 196) in linear
        sky.sun.illuminance = Vec3::new(100_000.0, 73_000.0, 55_200.0);
        let layer = HeightFogLayer::from_unreal(0.03, 0.1, 0.0);
        // far: Unreal's exponential height fog beyond its volumetric fog distance (120 m)
        let mut height_fog = HeightFogEffect::new(layer);
        height_fog.inscattering = Vec3::new(1.2, 1.45, 1.8);
        height_fog.sky_ambient_scale = 1.0;
        height_fog.start_distance = 120.0;
        height_fog.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        effects.push(Box::new(height_fog));
        // near: volumetric fog in the same layer, extinction scale 1.2, albedo (0.85, 0.88, 0.93)
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 120.0, grid_d: 48, temporal: true, blend_factor: 0.1, ..Default::default() },
            base_density: layer.density,
            height_falloff: layer.height_falloff,
            extinction_coeff: 1.2,
            albedo: Vec3::new(0.85 * 1.2, 0.88 * 1.2, 0.93 * 1.2),
            anisotropy: 0.3,
            ..Default::default()
        });
        fog.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        fog.set_shadow_map(renderer.shadow_map());
        effects.push(Box::new(fog));
    }
    let fog_density = q.get("fog").and_then(|v| v.parse::<f32>().ok());
    if !midsommar && (fog_density.is_some() || q.get("mist").is_some()) {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 1.0, far: 2500.0, temporal: true, blend_factor: 0.1, ..Default::default() },
            base_density: fog_density.unwrap_or(0.0),
            height_falloff: 0.04,
            anisotropy: 0.6,
            albedo: Vec3::new(0.9, 0.92, 0.95),
            ..Default::default()
        });
        fog.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        fog.set_shadow_map(renderer.shadow_map());
        if q.get("mist").is_some() {
            // mist lying in the clearing, thickest on the ground
            let mut mist = if q.get("mist").as_deref() == Some("box") {
                LocalFogVolume::new_box(Vec3::new(0.0, 0.0, 0.0), Vec3::new(30.0, 5.0, 30.0))
            } else {
                LocalFogVolume::new(Vec3::new(0.0, 0.0, 0.0), 60.0, 5.0)
            };
            mist.radial_extinction = 0.01;
            mist.height_extinction = 0.08;
            mist.height_falloff = 3.0;
            mist.albedo = Vec3::new(0.8, 0.84, 0.9);
            fog.local_volumes.push(mist);
        }
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(DisplayEffect::new(encode_srgb)));
    let volume = PostProcessingVolume::new(&renderer, effects);
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
