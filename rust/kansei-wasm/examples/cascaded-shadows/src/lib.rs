//! Cascaded shadows: a forest under a low afternoon sun, 4 stable cascades of 2048² out to
//! 250 m with contact-hardening (PCSS) penumbrae, the trees instanced and GPU-culled per cascade,
//! through TAA and the tonemapper. `csm=0` uses the single 2048² directional map instead, for
//! comparison; `debug=1` tints each cascade.
//!
//! URL parameters: `csm=0`, `debug=1`, `fog=1` (volumetric fog with shafts from the widest
//! cascade), `far=<metres>` (camera far plane, which the single map
//! is fitted to), `t=<seconds>` (freeze the camera).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::culling::InstanceCulling;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::shadows::{CascadedShadowOptions, CASCADED_SHADOWS_WGSL};

/// Sunlit diffuse surface: sun (lux) times the shadow, plus a sky hemisphere (cd/m²). INSTANCE_*
/// replaced for instanced trunks and crowns (vec4: position, scale).
const SURFACE_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, sun_dir: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };  // base_color.w: debug
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
// the single directional map (group 3, bindings 0-2), for `csm=0`
struct LegacyShadow { view_proj: mat4x4<f32>, bias: f32, normal_bias: f32, enabled: f32, _p: f32, _q: vec4<f32> };
@group(3) @binding(0) var legacy_depth: texture_depth_2d;
@group(3) @binding(1) var legacy_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> legacy: LegacyShadow;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, INSTANCE_INPUT };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };

@vertex
fn vertex_main(v: VIn) -> VOut {
    var local = v.position.xyz;
    var offset = vec3<f32>(0.0);
    INSTANCE_PLACE
    let world = world_matrix * vec4<f32>(local + offset, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

fn legacy_shadow(world: vec3<f32>, n: vec3<f32>) -> f32 {
    if (legacy.enabled < 0.5) { return 1.0; }
    let clip = legacy.view_proj * vec4<f32>(world + n * legacy.normal_bias, 1.0);
    let uv = vec2<f32>(clip.x, -clip.y) * 0.5 + 0.5;
    let texel = 1.0 / vec2<f32>(textureDimensions(legacy_depth));
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(legacy_depth, legacy_sampler, uv + vec2<f32>(f32(x), f32(y)) * texel, clip.z - legacy.bias);
        }
    }
    return lit / 9.0;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = -surface.sun_dir.xyz;
    var shadow = 1.0;
    if (kansei_cascades.count > 0u) {
        shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    } else {
        shadow = legacy_shadow(in.world, n);
    }
    var base = surface.base_color.rgb;
    if (surface.base_color.w > 0.5) {
        let tints = array<vec3<f32>, 5>(vec3<f32>(1.0, 0.4, 0.4), vec3<f32>(0.4, 1.0, 0.4), vec3<f32>(0.4, 0.6, 1.0), vec3<f32>(1.0, 1.0, 0.4), vec3<f32>(1.0));
        base *= tints[min(kansei_sun_cascade(in.world), 4u)];
    }
    let sky = mix(surface.sky.rgb * 0.15, surface.sky.rgb, n.y * 0.5 + 0.5);
    let lit = base / 3.14159265 * surface.sun.rgb * max(dot(n, l), 0.0) * shadow + base * sky;
    return vec4<f32>(lit, 1.0);
}
"#;

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
    let up = saturate(normalize(in.dir).y);
    return vec4<f32>(mix(vec3<f32>(9000.0, 9500.0, 10500.0), vec3<f32>(3000.0, 5000.0, 9000.0), sqrt(up)), 1.0);
}
"#;

/// Low afternoon sun (lux) and its travel direction.
const SUN_DIR: [f32; 3] = [-0.62, -0.42, -0.66];
const SUN: [f32; 3] = [80000.0, 70000.0, 56000.0];
const SKY: [f32; 3] = [4000.0, 5000.0, 7000.0];

fn surface_material(label: &str, base: [f32; 3], instanced: bool, debug: bool) -> Material {
    let shader = if instanced {
        SURFACE_WGSL
            .replace("INSTANCE_INPUT", "@location(3) instance: vec4<f32>,")
            .replace("INSTANCE_PLACE", "local = local * v.instance.w; offset = v.instance.xyz;")
    } else {
        SURFACE_WGSL.replace("INSTANCE_INPUT", "").replace("INSTANCE_PLACE", "")
    };
    let d = SUN_DIR;
    let data: [f32; 16] = [
        base[0], base[1], base[2], debug as u32 as f32,
        d[0], d[1], d[2], 0.0,
        SUN[0], SUN[1], SUN[2], 0.0,
        SKY[0], SKY[1], SKY[2], 0.0,
    ];
    let mut m = Material::new(
        label,
        &format!("{CASCADED_SHADOWS_WGSL}\n{shader}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions::default(),
    );
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

/// An instanced renderable of `geometry` at (position, scale) instances, GPU-culled per view.
fn instanced(renderer: &Renderer, label: &str, geometry: kansei_core::geometries::Geometry, instances: &[f32], radius: f32, material: Material) -> Renderable {
    use wgpu::util::DeviceExt;
    let buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some(label),
        contents: bytemuck::cast_slice(instances),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
    });
    let count = instances.len() as u32 / 4;
    let layout = ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage).with_vertex_vec4(3);
    let mut r = Renderable::new(InstancedGeometry::new(geometry, count, vec![layout]), material);
    r.instance_culling = Some(InstanceCulling::new(buffer, count, 16, 0, radius).with_radius_scale(12));
    r
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

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;
    let csm = query_param("csm").as_deref() != Some("0");
    let debug = query_param("debug").as_deref() == Some("1");
    if csm {
        renderer.enable_cascaded_shadows(CascadedShadowOptions::default());
    } else {
        renderer.enable_shadows(2048);
    }

    let mut scene = Scene::new();
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));

    let mut ground = Renderable::new(PlaneGeometry::new(1600.0, 1600.0), surface_material("Ground", [0.16, 0.18, 0.1], false, debug));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    ground.cast_shadow = false;
    scene.add(SceneNode::Renderable(ground));

    // a forest: trunks (unit boxes scaled by height) and crowns (spheres), 4000 trees out to
    // 500 m, clear around the path the camera walks
    let (mut trunks, mut crowns) = (Vec::new(), Vec::new());
    for i in 0..4000u32 {
        let (x, z) = (-500.0 + hash(i) * 1000.0, -500.0 + hash(i + 7) * 1000.0);
        if x.abs() < 4.0 {
            continue;
        }
        let h = 8.0 + hash(i + 13) * 10.0;
        trunks.extend_from_slice(&[x, h * 0.5, z, h]);
        crowns.extend_from_slice(&[x, h, z, 1.6 + hash(i + 17) * 1.4]);
    }
    scene.add(SceneNode::Renderable(instanced(&renderer, "Trunks", BoxGeometry::new(0.035, 1.0, 0.035), &trunks, 0.6, surface_material("Trunk", [0.2, 0.15, 0.1], true, debug))));
    scene.add(SceneNode::Renderable(instanced(&renderer, "Crowns", SphereGeometry::new(1.0, 12, 8), &crowns, 1.0, surface_material("Crown", [0.06, 0.12, 0.05], true, debug))));
    // a fence along the path: thin, close shadows
    for k in 0..60 {
        let mut post = Renderable::new(BoxGeometry::new(0.08, 1.2, 0.08), surface_material("Post", [0.35, 0.3, 0.25], false, debug));
        post.object.set_position(2.5, 0.6, 10.0 - k as f32 * 2.0);
        scene.add(SceneNode::Renderable(post));
    }

    let mut sun = DirectionalLight::new(Vec3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]), Vec3::new(1.0, 0.875, 0.7), 80000.0);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    let tonemap = {
        let mut o = ToneMapOptions::for_surface(renderer.presentation_format());
        o.exposure = exposure_from_ev100(14.5);
        o.vignette = 0.4;
        ToneMapEffect::new(o)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if query_param("fog").as_deref() == Some("1") {
        // shafts through the canopy, shadowed by the widest cascade
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 300.0, temporal: true, ..Default::default() },
            base_density: 0.01,
            height_falloff: 0.05,
            anisotropy: 0.6,
            ambient: Vec3::new(SKY[0], SKY[1], SKY[2]) * 0.5,
            ..Default::default()
        });
        if csm {
            fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        } else {
            fog.set_shadow_map(renderer.shadow_map());
        }
        fog.update_lights(scene.lights());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);
    let far: f32 = query_param("far").and_then(|v| v.parse().ok()).unwrap_or(1200.0);
    let camera = Camera::new(50.0, 0.3, far, width as f32 / height as f32);

    log::info!("Kansei — Cascaded Shadows (WASM) ready: {} trees, cascades {csm}", trunks.len() / 4);

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, start_ms: now_secs(), frozen_t }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref start_ms, frozen_t } = *st;
            let t = frozen_t.unwrap_or((now_secs() - *start_ms) as f32);
            // walk down the path, looking along it and slightly toward the sun
            let z = 8.0 - (t * 1.2) % 60.0;
            camera.set_position(0.0, 1.7, z);
            camera.look_at(&Vec3::new(4.0, 1.2, z - 20.0));
            renderer.render_with_postprocessing(scene, camera, volume);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
