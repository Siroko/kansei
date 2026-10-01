//! Spot lights: a car with the Midsommar intro's headlights (22 000 cd, 10°/30° cones, 70 m)
//! shines into an instanced forest at night. The trunks shadow the beams on the ground and in
//! the volumetric fog (spot shadow atlas + cone injection); surfaces use a GGX material that
//! includes `lights::SPOT_LIGHTS_WGSL`. Exposure is the intro's EV100 3.9, through ToneMapEffect.
//!
//! URL parameters: `cam=front|behind|top|wall`, `cull=main` (CPU-cull the trunks to the camera
//! only, the bug GPU per-view culling avoids), `drive=1`, `t=<seconds>` (freeze), `shadows=0`,
//! `fog=0`, `stats=1` (log the CPU time of the render call and the frame interval, which is the
//! GPU time when the browser runs without vsync), `casters=<n>` (n more
//! renderables), `lamps=<n>` (n small downlights), `clusters=0` (every light at every pixel),
//! `shafts=<steps>` (the beams raymarched per pixel with that many samples per light, instead of
//! in the fog's froxels).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::culling::{frustum_planes, InstanceCulling};
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, SpotScattering, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
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
    /// `cull=main`: (trunks, their buffer, scene index) to cull on the CPU against the camera.
    cpu_cull: Option<(Vec<f32>, wgpu::Buffer, usize)>,
    /// `stats=1`: CPU milliseconds spent in render_with_postprocessing, summed over a window.
    stats: Option<(f64, u32)>,
    /// `stats=1`: when the window started, for the frame interval (GPU-bound without vsync).
    window_start: f64,
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
    let cam = query_param("cam").unwrap_or_default();
    // `cam=wall`: a corridor to a pale wall 20 m ahead, and a row of poles right by the lamps
    // that the camera (between them and the wall) doesn't see; their shadows on the wall show
    // that shadow casters are culled per light, not to the camera
    let wall_test = cam == "wall";
    let mut trunks: Vec<f32> = Vec::new();
    for i in 0..1400u32 {
        let x = -45.0 + hash(i) * 90.0;
        let z = -90.0 + hash(i + 13) * 110.0;
        if (x - 1.75).abs() < 3.5 && z > -9.0 || wall_test && (x - 1.75).abs() < 9.0 && z > -21.0 {
            continue;
        }
        let h = 9.0 + hash(i + 29) * 8.0;
        trunks.extend_from_slice(&[x, h * 0.5, z, h]);
    }
    if wall_test {
        for k in 0..6 {
            trunks.extend_from_slice(&[-1.0 + k as f32 * 1.1, 1.5, -5.0, 3.0]);
        }
        let mut wall = Renderable::new(BoxGeometry::new(16.0, 6.0, 0.3), lit_material("Wall", [0.6, 0.6, 0.6], 0.9, false));
        wall.object.set_position(1.75, 3.0, -20.0);
        scene.add(SceneNode::Renderable(wall));
    }
    let count = trunks.len() as u32 / 4;
    // all trunks, in a buffer the GPU culls per view: the camera, and each shadowed spot light's
    // own frustum, so trunks outside the picture still shadow the beams in it
    let all_trunks = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Trunks"),
            contents: bytemuck::cast_slice(&trunks),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    // `cull=main` instead culls on the CPU against the camera only (as an app would): the bug
    // this avoids, where trunks leaving the frame stop casting shadows
    let cull_main = query_param("cull").as_deref() == Some("main");
    let instances = ComputeBuffer::from_external("Trunks", all_trunks.clone(), BufferType::Storage).with_vertex_vec4(3);
    let mut forest = Renderable::new(
        InstancedGeometry::new(BoxGeometry::new(0.35, 1.0, 0.35), count, vec![instances]),
        lit_material("Trunks", [0.22, 0.18, 0.15], 0.8, true),
    );
    if !cull_main {
        // sphere at the trunk's middle (position.y = h/2) with radius h/2 + a margin: scale by h
        forest.instance_culling = Some(InstanceCulling::new(all_trunks.clone(), count, 16, 0, 0.55).with_radius_scale(12));
    }
    let forest = scene.add(SceneNode::Renderable(forest));

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

    // `lamps=N`: N small unshadowed downlights over the forest, in random colours (clustered
    // light culling keeps them cheap; `clusters=0` shades every pixel with every light)
    let lamps: u32 = query_param("lamps").and_then(|v| v.parse().ok()).unwrap_or(0);
    for i in 0..lamps {
        let pos = Vec3::new(-30.0 + hash(i + 501) * 60.0, 2.5 + hash(i + 503) * 2.0, -70.0 + hash(i + 507) * 75.0);
        let hue = hash(i + 509) * 6.0;
        let color = Vec3::new((hue - 3.0).abs() - 1.0, 2.0 - (hue - 2.0).abs(), 2.0 - (hue - 4.0).abs());
        let color = Vec3::new(color.x.clamp(0.0, 1.0), color.y.clamp(0.0, 1.0), color.z.clamp(0.0, 1.0));
        let mut lamp = SpotLight::new(pos, Vec3::new(0.0, -1.0, 0.0), color, 800.0, 7.0, 25f32.to_radians(), 50f32.to_radians());
        lamp.volumetric_scale = 0.0;
        scene.add(SceneNode::Light(Light::Spot(lamp)));
    }
    renderer.set_clustered_lights(query_param("clusters").as_deref() != Some("0"));

    // `casters=N`: N more renderables (one draw each), to measure per-draw CPU cost
    let extra: u32 = query_param("casters").and_then(|v| v.parse().ok()).unwrap_or(0);
    for i in 0..extra {
        let mut post = Renderable::new(BoxGeometry::new(0.2, 2.0, 0.2), lit_material("Post", [0.3, 0.3, 0.3], 0.8, false));
        post.object.set_position(-3.0 + (i % 10) as f32 * 0.9, 1.0, -12.0 - (i / 10) as f32 * 1.5);
        scene.add(SceneNode::Renderable(post));
    }

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
            spot_scattering: match query_param("shafts").and_then(|v| v.parse().ok()) {
                Some(steps) if steps > 0 => SpotScattering::Raymarched { steps },
                _ => SpotScattering::Froxels,
            },
            ..Default::default()
        });
        fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(BloomEffect::new(BloomOptions { threshold: 0.0, intensity: 0.05, ..Default::default() }).with_exposure(tonemap.total_exposure())));
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(40.0, 0.1, 400.0, width as f32 / height as f32);
    camera.update_projection_matrix();

    log::info!("Kansei — Spot Lights (WASM) ready: {count} instanced trunks, shadows {shadows}");

    let frozen_t = query_param("t").and_then(|v| v.parse().ok());
    let drive = query_param("drive").as_deref() == Some("1");
    let cpu_cull = cull_main.then(|| (trunks.clone(), all_trunks.clone(), forest));
    let stats = (query_param("stats").as_deref() == Some("1")).then_some((0.0, 0));
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, car, start_ms: now_secs(), frozen_t, cam, drive, cpu_cull, stats, window_start: now_secs() }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref car, ref start_ms, frozen_t, ref cam, ref drive, ref cpu_cull, ref mut stats, ref mut window_start } = *st;
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
                // facing the wall, with the poles behind the camera
                "wall" => {
                    camera.set_position(1.75, 2.2, z - 9.0);
                    camera.look_at(&Vec3::new(1.75, 2.0, z - 20.0));
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

            if let Some((all, buffer, forest)) = cpu_cull {
                camera.update_view_matrix();
                let planes = frustum_planes(camera.projection_matrix.to_glam() * camera.view_matrix.to_glam());
                let visible: Vec<f32> = all
                    .chunks_exact(4)
                    .filter(|t| planes.iter().all(|p| p.x * t[0] + p.y * t[1] + p.z * t[2] + p.w >= -t[3] * 0.55))
                    .flatten()
                    .copied()
                    .collect();
                renderer.queue().write_buffer(buffer, 0, bytemuck::cast_slice(&visible));
                if let Some(r) = scene.get_renderable_mut(*forest) {
                    r.geometry.instance_count = visible.len() as u32 / 4;
                }
                renderer.invalidate_bundle();
            }
            let before = now_secs();
            renderer.render_with_postprocessing(scene, camera, volume);
            if let Some((sum, frames)) = stats {
                *sum += (now_secs() - before) * 1000.0;
                *frames += 1;
                if *frames == 240 {
                    let interval = (now_secs() - *window_start) * 1000.0 / 240.0;
                    log::info!("frame: {:.2} ms CPU in render_with_postprocessing, {interval:.2} ms between frames", *sum / 240.0);
                    *sum = 0.0;
                    *frames = 0;
                    *window_start = now_secs();
                }
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
