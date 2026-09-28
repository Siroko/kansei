//! Planar reflection: the Midsommar lake shot in miniature. A still lake at dusk mirrors the far
//! shore's treeline, a red cottage with lit windows and the sky, through a PlanarReflection
//! (mirrored camera, oblique clip plane at the water, the water itself left out by its layer).
//! The water material adds Fresnel, wind ripples that displace the lookup and a roughness that
//! picks the reflection's mips. A lamp on the far bank throws a beam through the mist, and the
//! volumetric fog is composited into the reflection too (`VolumetricFogEffect::reflection_fog`),
//! so the lake mirrors the beam's glow.
//!
//! URL parameters: `ripples=<strength>` (0 = mirror), `rough=<0..1>`, `t=<seconds>` (freeze),
//! `fogrefl=0` (no fog in the reflection: the water fogs the reflected path with a flat colour),
//! `occlusion=1` (the treeline occlusion-culled, for the camera and in the mirror
//! (`PlanarReflection::occlusion_culling`); the culling stats logged every 2 s), `screen=1` (the
//! reflection from the screen, `PlanarReflection::screen_space`, instead of the mirrored view).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler};
use kansei_core::culling::InstanceCulling;
use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::reflections::{PlanarReflection, PlanarReflectionOptions, PLANAR_REFLECTION_WGSL};
use kansei_core::renderers::{Renderer, RendererConfig};

const WATER_LAYER: u32 = 2;
const LAKE_LEVEL: f32 = 0.0;

/// Diffuse surface under the dusk sky (a hemispherical sky term, cd/m²), optionally instanced
/// (vec4 per instance: xyz offset, w height scale), plus an emissive term for lit windows.
const SURFACE_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, emissive: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, INSTANCE_INPUT };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) local_y: f32 };

@vertex
fn vertex_main(v: VIn) -> VOut {
    var local = v.position.xyz;
    var offset = vec3<f32>(0.0);
    INSTANCE_OFFSET
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(local + offset, 1.0);
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    out.local_y = v.position.y;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    // dusk: 40 cd/m² sky from above and the west, dark ground bounce
    let sky = mix(vec3<f32>(0.4, 0.45, 0.5), vec3<f32>(22.0, 26.0, 36.0), saturate(n.y * 0.5 + 0.5));
    let window = select(0.0, 1.0, surface.emissive.w > 0.0 && abs(in.local_y) < surface.emissive.w);
    return vec4<f32>(surface.base_color.rgb * sky + surface.emissive.rgb * window, 1.0);
}
"#;

/// Dusk sky dome: horizon glow toward the (set) sun in the north-west, deep blue overhead.
const SKY_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let d = normalize(in.dir);
    let up = saturate(d.y);
    let glow = pow(saturate(dot(d, normalize(vec3<f32>(-0.6, 0.0, -0.8))) * 0.5 + 0.5), 6.0);
    let horizon = mix(vec3<f32>(60.0, 62.0, 70.0), vec3<f32>(240.0, 150.0, 90.0), glow);
    let zenith = vec3<f32>(18.0, 28.0, 55.0);
    return vec4<f32>(mix(horizon, zenith, pow(up, 0.45)), 1.0);
}
"#;

/// The lake: Fresnel mix of a dark body colour and the planar reflection, displaced by wind
/// ripples and fogged over the reflected path.
const WATER_WGSL: &str = r#"
struct Water { params: vec4<f32>, fog: vec4<f32>, body: vec4<f32> };  // params: time, ripples, roughness, -
@group(0) @binding(0) var<uniform> water: Water;
@group(0) @binding(1) var reflection_tex: texture_2d<f32>;
@group(0) @binding(2) var reflection_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut { @builtin(position) pixel: vec4<f32>, @location(0) world: vec3<f32>, @location(1) clip: vec4<f32> };

@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.pixel = out.clip;
    out.world = world.xyz;
    return out;
}

// gradient of a few travelling sine waves: faint wind ripples
fn ripple_slope(p: vec2<f32>, t: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    let dirs = array<vec2<f32>, 4>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.3, 0.95), vec2<f32>(0.99, -0.14), vec2<f32>(-0.7, -0.7));
    let freq = array<f32, 4>(1.3, 2.1, 3.7, 5.3);
    for (var i = 0; i < 4; i++) {
        let k = freq[i];
        g += dirs[i] * cos(dot(dirs[i], p) * k + t * sqrt(9.81 * k)) * (0.35 / k);
    }
    return g;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let slope = ripple_slope(in.world.xz, water.params.x) * water.params.y;
    let n = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let to_eye = camera_pos - in.world;
    let v = normalize(to_eye);

    let uv = kansei_screen_uv(in.clip);
    let offset = kansei_reflection_offset(view_matrix, up, n, 0.6);
    let r = kansei_planar_reflection(reflection_tex, reflection_sampler, uv, offset, water.params.z);
    // fog over the part of the path beyond the water (the main fog covers camera -> water)
    let beyond = max(r.a - length(to_eye), 0.0);
    let t = exp(-water.fog.w * beyond);
    let reflected = r.rgb * t + water.fog.rgb * (1.0 - t);

    let fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(n, v)), 5.0);
    return vec4<f32>(mix(water.body.rgb, reflected, fresnel), 1.0);
}
"#;

fn surface_material(label: &str, base: [f32; 3], emissive: [f32; 3], window_band: f32, instanced: bool) -> Material {
    let shader = if instanced {
        SURFACE_WGSL
            .replace("INSTANCE_INPUT", "@location(3) instance: vec4<f32>,")
            .replace("INSTANCE_OFFSET", "local.y *= v.instance.w; offset = v.instance.xyz;")
    } else {
        SURFACE_WGSL.replace("INSTANCE_INPUT", "").replace("INSTANCE_OFFSET", "")
    };
    let data: [f32; 8] = [base[0], base[1], base[2], 1.0, emissive[0], emissive[1], emissive[2], window_band];
    let mut m = Material::new(label, &shader, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions::default());
    m.set_uniform_bindable(0, label, &data);
    m
}

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
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
    water: usize,
    water_params: [f32; 12],
    start_ms: f64,
    frozen_t: Option<f32>,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_secs() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
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

    let mut scene = Scene::new();
    // (a pipeline layout's group 0 needs a bind group, so the sky gets an unused uniform)
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    scene.add(SceneNode::Renderable(Renderable::new(SphereGeometry::new(900.0, 48, 24), sky)));

    // far shore: a bank rising out of the lake 120-400 m away, with a treeline on it
    let mut bank = Renderable::new(BoxGeometry::new(1600.0, 6.0, 300.0), surface_material("Bank", [0.05, 0.06, 0.04], [0.0; 3], 0.0, false));
    bank.object.set_position(0.0, 1.0, -270.0);
    scene.add(SceneNode::Renderable(bank));
    let mut trees: Vec<f32> = Vec::new();
    for i in 0..1600u32 {
        let x = -700.0 + hash(i) * 1400.0;
        let z = -125.0 - hash(i + 5) * 200.0;
        let h = 12.0 + hash(i + 11) * 14.0;
        trees.extend_from_slice(&[x, 4.0 + h * 0.5, z, h]);
    }
    let count = trees.len() as u32 / 4;
    let all_trees = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Trees"),
            contents: bytemuck::cast_slice(&trees),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        })
    };
    let instances = ComputeBuffer::from_external("Trees", all_trees.clone(), BufferType::Storage).with_vertex_vec4(3);
    // spruce-like silhouettes: narrow tall boxes (a stand-in; the app has real trees), culled on
    // the GPU for the camera and, separately, for the mirrored view
    let treeline = InstancedGeometry::new(BoxGeometry::new(2.6, 1.0, 2.6), count, vec![instances]);
    let mut treeline = Renderable::new(treeline, surface_material("Trees", [0.03, 0.04, 0.03], [0.0; 3], 0.0, true));
    let occlusion = query_param("occlusion").as_deref() == Some("1");
    treeline.instance_culling = Some(InstanceCulling::new(all_trees, count, 16, 0, 0.6).with_radius_scale(12).with_occlusion(occlusion));
    scene.add(SceneNode::Renderable(treeline));

    // the red cottage on the shore, windows glowing at 160 cd/m² (the intro's window_glow)
    let mut cottage = Renderable::new(BoxGeometry::new(9.0, 5.0, 6.0), surface_material("Cottage", [0.35, 0.05, 0.03], [160.0, 110.0, 60.0], 0.6, false));
    cottage.object.set_position(-18.0, 6.5, -128.0);
    scene.add(SceneNode::Renderable(cottage));
    let mut roof = Renderable::new(BoxGeometry::new(9.6, 1.2, 6.6), surface_material("Roof", [0.04, 0.04, 0.04], [0.0; 3], 0.0, false));
    roof.object.set_position(-18.0, 9.6, -128.0);
    scene.add(SceneNode::Renderable(roof));

    // near shore under the camera, so the lake has an edge
    let mut near_bank = Renderable::new(BoxGeometry::new(1600.0, 4.0, 60.0), surface_material("NearBank", [0.04, 0.05, 0.03], [0.0; 3], 0.0, false));
    near_bank.object.set_position(0.0, -1.2, 44.0);
    scene.add(SceneNode::Renderable(near_bank));

    // a searchlight on the far bank beside the cottage (5e7 cd), its beam rising across the lake through
    // the mist
    let lamp_pos = Vec3::new(-32.0, 11.0, -126.0);
    let mut lamp = SpotLight::new(
        lamp_pos,
        Vec3::new(40.0 - lamp_pos.x, 45.0 - lamp_pos.y, -50.0 - lamp_pos.z).normalize(),
        Vec3::new(1.0, 0.82, 0.6),
        5.0e7,
        300.0,
        3f32.to_radians(),
        7f32.to_radians(),
    );
    lamp.volumetric_scale = 1.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    // the reflection: half resolution, everything but the water
    let mut reflection = PlanarReflection::new(
        &renderer,
        Vec3::new(0.0, LAKE_LEVEL, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        PlanarReflectionOptions { width: width / 2, height: height / 2, layer_mask: !WATER_LAYER, ..Default::default() },
    );
    reflection.occlusion_culling = occlusion;
    reflection.screen_space = query_param("screen").as_deref() == Some("1");
    renderer.set_culling_stats(occlusion);
    let ripples: f32 = query_param("ripples").and_then(|v| v.parse().ok()).unwrap_or(0.03);
    let roughness: f32 = query_param("rough").and_then(|v| v.parse().ok()).unwrap_or(0.05);
    // the fog in the reflection (fogrefl=0 leaves it out, and the water fogs the reflected path
    // with a flat colour instead, as before)
    let fog_in_reflection = query_param("fogrefl").as_deref() != Some("0");
    // time, ripples, roughness, - | fog colour, fog density | deep body colour
    let water_fog = if fog_in_reflection { 0.0 } else { 0.0012 };
    let water_params: [f32; 12] = [0.0, ripples, roughness, 0.0, 45.0, 48.0, 58.0, water_fog, 0.2, 0.3, 0.3, 1.0];
    let mut water_material = Material::new(
        "Water",
        &format!("{PLANAR_REFLECTION_WGSL}\n{WATER_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
    );
    water_material.set_uniform_bindable(0, "Water", &water_params);
    water_material.set_bindable(1, reflection.material_texture());
    water_material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));

    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { near: 1.0, far: 800.0, temporal: true, ..Default::default() },
        base_density: 0.0012,
        height_falloff: 0.04,
        anisotropy: 0.3,
        ambient: Vec3::new(38.0, 40.0, 50.0),
        ..Default::default()
    });
    fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    if fog_in_reflection {
        let seen = fog.reflection_fog(&renderer, &reflection);
        reflection.set_fog(&renderer, Some(&seen));
    }
    renderer.add_planar_reflection(reflection);
    let mut lake = Renderable::new(PlaneGeometry::new(1600.0, 400.0), water_material);
    lake.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    lake.object.set_position(0.0, LAKE_LEVEL, -100.0);
    lake.layers = WATER_LAYER;
    lake.cast_shadow = false;
    let water = scene.add(SceneNode::Renderable(lake));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(6.5);
        options.vignette = 0.5;
        options.grain = 0.22;
        ToneMapEffect::new(options)
    };
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![
        Box::new(fog),
        Box::new(BloomEffect::new(BloomOptions { threshold: 0.0, intensity: 0.04, ..Default::default() }).with_exposure(tonemap.total_exposure())),
        Box::new(tonemap),
    ];
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(28.0, 0.5, 2000.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    log::info!("Kansei — Planar Reflection (WASM) ready: ripples {ripples}, roughness {roughness}");

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, water, water_params, start_ms: now_secs(), frozen_t }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, water, ref mut water_params, ref start_ms, frozen_t } = *st;
            let clock = (now_secs() - *start_ms) as f32;
            let t = frozen_t.unwrap_or(clock);

            if let Some(fog) = volume.effects[0].as_any_mut().downcast_mut::<VolumetricFogEffect>() {
                fog.time = clock;
            }
            water_params[0] = clock;
            if let Some(r) = scene.get_renderable_mut(water) {
                if let Some(buf) = r.material.bindable_buffer(0) {
                    renderer.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&water_params[..]));
                }
            }

            // low over the near shore, panning slowly along the far treeline
            let yaw = -0.25 + 0.12 * (t * 0.05).sin();
            camera.set_position(6.0, 2.2, 16.0);
            camera.look_at(&Vec3::new(6.0 + yaw.sin() * 100.0, 3.0, 16.0 - yaw.cos() * 100.0));

            renderer.render_with_postprocessing(scene, camera, volume);
            if camera.frame() % 120 == 0 {
                if let Some(stats) = renderer.culling_stats() {
                    log::info!("culling: {:?}", stats.views);
                }
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
