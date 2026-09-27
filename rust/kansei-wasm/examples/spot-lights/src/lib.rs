//! Spot lights: a car with the Midsommar intro's headlights (22 000 cd, 10°/30° cones, 70 m)
//! shines into an instanced forest at night. The trunks shadow the beams on the ground and in
//! the volumetric fog (spot shadow atlas + cone injection); surfaces use a GGX material that
//! includes `lights::SPOT_LIGHTS_WGSL`. Exposure is the intro's EV100 3.9, through ToneMapEffect.
//!
//! URL parameters: `cam=front|behind|top`, `drive=1`, `t=<seconds>` (freeze), `shadows=0`, `fog=0`.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};

/// GGX surface lit by the spot lights plus a dim hemispherical night sky. `INSTANCE_INPUT` and
/// `INSTANCE_OFFSET` are replaced for the instanced variant (a vec4 per trunk: xyz offset, w
/// height scale).
const LIT_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, params: vec4<f32> };   // params: roughness, metallic
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    INSTANCE_INPUT
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    var local = v.position.xyz;
    var offset = vec3<f32>(0.0);
    INSTANCE_OFFSET
    let world = world_matrix * vec4<f32>(local + offset, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = surface.base_color.rgb;
    // night sky 0.25 cd/m² above, the ground bouncing a tenth of it
    let sky = mix(vec3<f32>(0.02, 0.025, 0.03), vec3<f32>(0.15, 0.2, 0.3), n.y * 0.5 + 0.5);
    let spots = kansei_spot_lights_radiance(in.world, n, v, base, surface.params.x, surface.params.y, in.clip.xy);
    return vec4<f32>(base * sky + spots, 1.0);
}
"#;

/// Unlit radiance (cd/m²), for the headlight lenses.
const EMISSIVE_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> radiance: vec4<f32>;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> @builtin(position) vec4<f32> {
    return projection_matrix * view_matrix * world_matrix * position;
}
@fragment
fn fragment_main() -> @location(0) vec4<f32> {
    return vec4<f32>(radiance.rgb, 1.0);
}
"#;

fn lit_material(label: &str, base_color: [f32; 3], roughness: f32, instanced: bool) -> Material {
    let shader = if instanced {
        LIT_WGSL
            .replace("INSTANCE_INPUT", "@location(3) instance: vec4<f32>,")
            .replace("INSTANCE_OFFSET", "local.y *= v.instance.w; offset = v.instance.xyz;")
    } else {
        LIT_WGSL.replace("INSTANCE_INPUT", "").replace("INSTANCE_OFFSET", "")
    };
    let data: [f32; 8] = [base_color[0], base_color[1], base_color[2], 1.0, roughness, 0.0, 0.0, 0.0];
    let mut material = Material::new(
        label,
        &format!("{SPOT_LIGHTS_WGSL}\n{shader}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions::default(),
    );
    material.set_uniform_bindable(0, label, &data);
    material
}

fn emissive_material(label: &str, radiance: [f32; 3]) -> Material {
    let mut material = Material::new(label, EMISSIVE_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    material.set_uniform_bindable(0, label, &[radiance[0], radiance[1], radiance[2], 0.0f32]);
    material
}

/// Deterministic 0..1 hash.
fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

struct Car {
    body: usize,
    lenses: [usize; 2],
    lights: [usize; 2],
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    car: Car,
    start_ms: f64,
    frozen_t: Option<f32>,
    cam: String,
    drive: bool,
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

/// Car position along Z: parked at the forest edge, or creeping forward and back with `drive=1`.
fn car_z(t: f32, drive: bool) -> f32 {
    if drive { 2.0 * (t * 0.3).sin() } else { 0.0 }
}

fn place_car(scene: &mut Scene, car: &Car, z: f32) {
    if let Some(r) = scene.get_renderable_mut(car.body) {
        r.object.set_position(1.75, 0.75, z);
    }
    for (k, side) in [-0.62f32, 0.62].into_iter().enumerate() {
        let lamp = Vec3::new(1.75 + side, 0.7, z - 2.3);
        if let Some(r) = scene.get_renderable_mut(car.lenses[k]) {
            r.object.set_position(lamp.x, lamp.y, lamp.z - 0.02);
        }
        if let Some(Light::Spot(s)) = scene.get_light_mut(car.lights[k]) {
            s.position = lamp;
            // dipped beams: aimed at the ground 25 m ahead, spreading slightly apart
            s.look_at(Vec3::new(lamp.x + side * 2.0, 0.0, lamp.z - 25.0));
        }
    }
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
    let shadows = query_param("shadows").as_deref() != Some("0");
    if shadows {
        renderer.enable_spot_shadows(1024, 2);
    }

    let mut scene = Scene::new();

    let mut ground = Renderable::new(PlaneGeometry::new(200.0, 200.0), lit_material("Ground", [0.12, 0.11, 0.09], 0.9, false));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // the forest: one instanced draw of unit trunks, everywhere but a small clearing around the
    // car, which stands at the forest edge with its lights into the trees
    let mut trunks: Vec<f32> = Vec::new();
    for i in 0..1400u32 {
        let x = -45.0 + hash(i) * 90.0;
        let z = -90.0 + hash(i + 13) * 110.0;
        if (x - 1.75).abs() < 3.5 && z > -9.0 {
            continue;
        }
        let h = 9.0 + hash(i + 29) * 8.0;
        trunks.extend_from_slice(&[x, h * 0.5, z, h]);
    }
    let count = trunks.len() as u32 / 4;
    let instances = ComputeBuffer::from_slice("Trunks", BufferType::Storage, BufferUsage::VERTEX, &trunks).with_vertex_vec4(3);
    let forest = InstancedGeometry::new(BoxGeometry::new(0.35, 1.0, 0.35), count, vec![instances]);
    scene.add(SceneNode::Renderable(Renderable::new(forest, lit_material("Trunks", [0.22, 0.18, 0.15], 0.8, true))));

    // the car: a dark body, two lens emitters (4000 cd/m², the intro's headlight_lens) and two
    // 22 000 cd dipped beams, 10° inner / 30° outer, 70 m; both shadowed
    let body = scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(1.8, 1.1, 4.4), lit_material("Car", [0.05, 0.05, 0.06], 0.3, false))));
    let mut lenses = [0; 2];
    let mut lights = [0; 2];
    for k in 0..2 {
        let mut lens = Renderable::new(BoxGeometry::new(0.3, 0.15, 0.04), emissive_material("Lens", [4000.0, 3800.0, 3400.0]));
        lens.cast_shadow = false;
        lenses[k] = scene.add(SceneNode::Renderable(lens));
        let mut beam = SpotLight::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 0.95, 0.85), 22000.0, 70.0, 10f32.to_radians(), 30f32.to_radians());
        beam.cast_shadow = true;
        beam.source_radius = 0.08;
        lights[k] = scene.add(SceneNode::Light(Light::Spot(beam)));
    }
    let car = Car { body, lenses, lights };

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(3.9);
        options.vignette = 0.5;
        options.grain = 0.22;
        options.chromatic_aberration = 0.25;
        ToneMapEffect::new(options)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if query_param("fog").as_deref() != Some("0") {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 150.0, grid_d: 96, temporal: true, ..Default::default() },
            base_density: 0.015,
            height_falloff: 0.05,
            anisotropy: 0.3,
            ambient: Vec3::new(0.25, 0.32, 0.45),
            ..Default::default()
        });
        fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(BloomEffect::new(BloomOptions { threshold: 0.0, intensity: 0.05, exposure: tonemap.total_exposure(), ..Default::default() })));
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(40.0, 0.1, 400.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    log::info!("Kansei — Spot Lights (WASM) ready: {count} instanced trunks, shadows {shadows}");

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let cam = query_param("cam").unwrap_or_default();
    let drive = query_param("drive").as_deref() == Some("1");
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, car, start_ms: now_secs(), frozen_t, cam, drive }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref car, ref start_ms, frozen_t, ref cam, ref drive } = *st;
            let clock = (now_secs() - *start_ms) as f32;
            let t = frozen_t.unwrap_or(clock);

            if let Some(fog) = volume.effects[0].as_any_mut().downcast_mut::<VolumetricFogEffect>() {
                fog.time = clock;
            }

            let z = car_z(t, *drive);
            place_car(scene, car, z);
            match cam.as_str() {
                // high behind the car, looking down the beams into the forest
                "behind" => {
                    camera.set_position(1.75, 5.0, z + 10.0);
                    camera.look_at(&Vec3::new(1.75, 1.0, z - 20.0));
                }
                // above the canopy, looking down on the beams through the trees
                "top" => {
                    camera.set_position(-10.0, 24.0, z + 6.0);
                    camera.look_at(&Vec3::new(2.0, 0.0, z - 16.0));
                }
                // low, facing the lamps from inside the forest: glare, beams and the trunks'
                // shadows radiating from the car
                _ => {
                    camera.set_position(-1.0, 1.8, z - 25.0);
                    camera.look_at(&Vec3::new(1.75, 1.2, z));
                }
            }

            renderer.render_with_postprocessing(scene, camera, volume);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
