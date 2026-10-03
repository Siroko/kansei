//! Voxel GI on particles: Hector Arellano's "indirect lighting on particles" (miaumiau.cat,
//! p=1476) on Kansei's `gi` module. A fluid sloshes in an open-topped room (white floor and back
//! wall, a red wall on the left, a green one on the right) under a sun and a blue sky. Every
//! frame the particles splat their density and emission into a voxel volume, with the walls as
//! analytic boxes; each particle then cone traces its incoming light (six 90-degree cones,
//! escaping to the sky) and its sun visibility (one narrow cone). So the fluid shades itself:
//! its body darkens inside, the walls' colours bleed onto it, glowing particles light their
//! neighbours and it casts a soft shadow. The walls trace the same volume: the fluid shadows the
//! floor and its glow lights it.
//!
//! URL parameters: `gi=on|off` (default on), `view=indirect` (only the light the volume and the
//! sky bring, without the sun's direct light, a stop brighter), `quality=low|medium|high`
//! (default medium, low on phones, which also keep the volume within 24 MiB), `particles=N`
//! (default 32768, 16384 on phones), `stats=1` (log the frame interval).

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler};
use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::gi::{GiBox, ParticleEmission, ParticleGi, ParticleGiOptions, VoxelGiQuality, VOXEL_CONES_WGSL};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{FluidSimulation, FluidSimulationOptions};

/// The room's inside (the fluid's bounds), metres; the walls are `WALL` thick outside it and the
/// top and the front (+z, toward the camera) are open.
const ROOM_MIN: [f32; 3] = [-14.0, 0.0, -7.0];
const ROOM_MAX: [f32; 3] = [14.0, 22.0, 7.0];
const WALL: f32 = 2.0;

/// Light shared by every material (WGSL `SceneParams`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SceneParams {
    to_sun: [f32; 3],
    exposure: f32,
    sun_illuminance: [f32; 3],
    gi_on: f32,
    sky_up: [f32; 3],
    view: f32,
    sky_down: [f32; 3],
    sun_cone_tan: f32,
    cone_steps: f32,
    box_sky_scale: f32,
    _pad: [f32; 2],
}

const SCENE_WGSL: &str = r#"
struct SceneParams {
    toSun          : vec3f,
    exposure       : f32,
    sunIlluminance : vec3f,
    giOn           : f32,
    skyUp          : vec3f,
    view           : f32,    // 1: indirect light only
    skyDown        : vec3f,
    sunConeTan     : f32,
    coneSteps      : f32,
    boxSkyScale    : f32,
    _pad0          : f32,
    _pad1          : f32,
}

const PI: f32 = 3.14159265;

// the sky as gi::gradient_sky_lighting makes it: linear in the direction's height
fn skyRad(s: SceneParams, d: vec3f) -> vec3f {
    return mix(s.skyDown, s.skyUp, d.y * 0.5 + 0.5);
}
fn skyIrr(s: SceneParams, n: vec3f) -> vec3f {
    return PI * (s.skyUp + s.skyDown) * 0.5 + (2.0 * PI / 3.0) * (s.skyUp - s.skyDown) * 0.5 * n.y;
}
fn tonemap(s: SceneParams, c: vec3f) -> vec3f {
    let k = select(1.0, 2.0, s.view > 0.5);
    return 1.0 - exp(-c * s.exposure * k);
}
"#;

/// The walls: lit by the sun and the sky, and with voxel GI on, shadowed by the fluid (a sun
/// cone) and lit by the volume (five cones over the hemisphere: the other walls' bounce, the
/// fluid's glow, the sky it leaves).
const ROOM_WGSL: &str = r#"
struct Surface { albedo: vec4f };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(0) @binding(1) var<uniform> scene: SceneParams;
@group(0) @binding(2) var<uniform> vol: VoxelVolume;
@group(0) @binding(3) var radiance: texture_3d<f32>;
@group(0) @binding(4) var linearClamp: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
};

@vertex
fn vertex_main(@location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f) -> VOut {
    let world = world_matrix * vec4f(position.xyz, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(normal, 0.0)).xyz;
    return out;
}

// irradiance over the hemisphere around n from one cone along it and four at 60 degrees, each
// 60 degrees wide (Crassin et al. 2011), the sky past them
fn hemisphere(o: vec3f, n: vec3f, start: f32) -> vec3f {
    let t = normalize(select(cross(n, vec3f(0.0, 1.0, 0.0)), cross(n, vec3f(1.0, 0.0, 0.0)), abs(n.y) > 0.9));
    let b = cross(n, t);
    var dirs = array<vec3f, 5>(n, 0.5 * n + 0.866 * t, 0.5 * n - 0.866 * t, 0.5 * n + 0.866 * b, 0.5 * n - 0.866 * b);
    var sum = vec3f(0.0);
    for (var k = 0u; k < 5u; k++) {
        let c = voxelConeTrace(vol, radiance, linearClamp, o, dirs[k], 0.577, start, 1e4, u32(scene.coneSteps));
        sum += select(0.15, 0.25, k == 0u) * (c.rgb + c.a * skyRad(scene, dirs[k]));
    }
    return PI * sum / 0.85;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let albedo = surface.albedo.rgb;
    let l = normalize(scene.toSun);
    var sunVis = 1.0;
    var irradiance = scene.boxSkyScale * skyIrr(scene, n);
    if (scene.giOn > 0.5) {
        // start out of the wall's own voxels
        let o = in.world + n * vol.voxelSize;
        sunVis = voxelConeTrace(vol, radiance, linearClamp, o, l, scene.sunConeTan, 0.5 * vol.voxelSize, 1e4, u32(scene.coneSteps)).a;
        irradiance = hemisphere(o, n, vol.voxelSize);
    }
    let direct = scene.sunIlluminance * max(dot(n, l), 0.0) * sunVis;
    var color = albedo / PI * (direct + irradiance);
    if (scene.view > 0.5) {
        color = albedo / PI * irradiance;
    }
    return vec4f(tonemap(scene, color), 1.0);
}
"#;

/// The particles: camera-facing spheres lit by what the GI gathered for them (instance
/// attributes 4 and 5: incoming light and sun visibility, emission).
const PARTICLE_WGSL: &str = r#"
struct Particles { albedo: vec4f, size: f32, _p0: f32, _p1: f32, _p2: f32 };
@group(0) @binding(0) var<uniform> particles: Particles;
@group(0) @binding(1) var<uniform> scene: SceneParams;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> _world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) center: vec4f,
    @location(4) lighting: vec4f,
    @location(5) emission: vec4f,
};
struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) corner: vec2f,
    @location(1) lighting: vec4f,
    @location(2) emission: vec3f,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let world = v.center.xyz + (right * v.position.x + up * v.position.y) * particles.size;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * vec4f(world, 1.0);
    out.corner = v.position.xy * 2.0;
    out.lighting = v.lighting;
    out.emission = v.emission.rgb;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let r2 = dot(in.corner, in.corner);
    if (r2 > 1.0) { discard; }
    // the sphere's normal, from view space
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let back = vec3f(view_matrix[0][2], view_matrix[1][2], view_matrix[2][2]);
    let n = normalize(right * in.corner.x + up * in.corner.y + back * sqrt(1.0 - r2));
    let albedo = particles.albedo.rgb;
    let direct = scene.sunIlluminance * max(dot(n, normalize(scene.toSun)), 0.0) * in.lighting.a;
    var color = albedo * (in.lighting.rgb + direct / PI) + in.emission;
    if (scene.view > 0.5) {
        color = albedo * in.lighting.rgb;
    }
    return vec4f(tonemap(scene, color), 1.0);
}
"#;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
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

/// Toward the sun at `elevation` degrees up, `bearing` degrees from +z (the open front) toward +x.
fn sun_direction(elevation: f32, bearing: f32) -> [f32; 3] {
    let (e, b) = (elevation.to_radians(), bearing.to_radians());
    [e.cos() * b.sin(), e.sin(), e.cos() * b.cos()]
}

fn is_phone() -> bool {
    let agent = web_sys::window().and_then(|w| w.navigator().user_agent().ok()).unwrap_or_default();
    ["Mobi", "Android", "iPhone", "iPad"].iter().any(|k| agent.contains(k))
}

/// A dam: particles on a jittered lattice filling the room's left half.
fn initial_positions(count: usize) -> Vec<f32> {
    let lo = [ROOM_MIN[0] + 0.5, ROOM_MIN[1] + 0.5, ROOM_MIN[2] + 0.5];
    let hi = [-1.0, 16.0, ROOM_MAX[2] - 0.5];
    let size: [f32; 3] = std::array::from_fn(|i| hi[i] - lo[i]);
    let spacing = (size[0] * size[1] * size[2] / count as f32).cbrt();
    let cells: [usize; 3] = std::array::from_fn(|i| ((size[i] / spacing).floor() as usize).max(1));
    let mut rng: u32 = 12345;
    let mut jitter = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0 - 0.5) * spacing * 0.3
    };
    let mut positions = Vec::with_capacity(count * 4);
    'fill: for layer in 0.. {
        for z in 0..cells[2] {
            for y in 0..cells[1] {
                for x in 0..cells[0] {
                    if positions.len() >= count * 4 {
                        break 'fill;
                    }
                    let base = [x, y, z].map(|c| (c as f32 + 0.5) * spacing);
                    // a further layer, if the lattice falls short, sits half a cell above
                    let lift = layer as f32 * spacing * 0.5;
                    positions.extend_from_slice(&[lo[0] + base[0] + jitter(), lo[1] + base[1] + lift + jitter(), lo[2] + base[2] + jitter(), 1.0]);
                }
            }
        }
    }
    positions
}

struct Wall {
    label: &'static str,
    min: [f32; 3],
    max: [f32; 3],
    albedo: [f32; 3],
    normal: [f32; 3],
}

fn walls() -> [Wall; 4] {
    let [x0, y0, z0] = ROOM_MIN;
    let [x1, y1, z1] = ROOM_MAX;
    let white = [0.75, 0.75, 0.75];
    [
        Wall { label: "Floor", min: [x0 - WALL, y0 - WALL, z0 - WALL], max: [x1 + WALL, y0, z1], albedo: white, normal: [0.0, 1.0, 0.0] },
        Wall { label: "Back", min: [x0 - WALL, y0, z0 - WALL], max: [x1 + WALL, y1, z0], albedo: white, normal: [0.0, 0.0, 1.0] },
        Wall { label: "Left", min: [x0 - WALL, y0, z0], max: [x0, y1, z1], albedo: [0.75, 0.08, 0.06], normal: [1.0, 0.0, 0.0] },
        Wall { label: "Right", min: [x1, y0, z0], max: [x1 + WALL, y1, z1], albedo: [0.1, 0.6, 0.12], normal: [-1.0, 0.0, 0.0] },
    ]
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    sim: FluidSimulation,
    gi: ParticleGi,
    scene_params: SceneParams,
    scene_buffer: wgpu::Buffer,
    particles_index: usize,
    initial: Vec<f32>,
    sim_accumulator: f64,
    last_time: f64,
    stats: Option<(f64, u32)>,
    frame_ms: f64,
    paused: bool,
}

impl State {
    fn frame(&mut self) {
        let now = now_ms();
        let dt = ((now - self.last_time) * 0.001).clamp(0.0, 0.1);
        self.last_time = now;
        self.frame_ms = self.frame_ms * 0.95 + dt * 1000.0 * 0.05;

        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(dt.max(1e-3) as f32);
        self.camera.update_projection_matrix();
        self.camera.update_view_matrix();

        // the fluid, at a fixed step
        let view = self.camera.view_matrix.to_glam();
        let proj = self.camera.projection_matrix.to_glam();
        let inv_view = self.camera.inverse_view_matrix.to_glam();
        self.sim.set_camera_matrices(&view.to_cols_array(), &proj.to_cols_array(), &inv_view.to_cols_array(), &glam::Mat4::IDENTITY.to_cols_array());
        let step = 1.0 / 60.0;
        self.sim_accumulator = (self.sim_accumulator + dt).min(step * 3.0);
        if self.paused {
            self.sim_accumulator = 0.0;
        }
        while self.sim_accumulator >= step {
            let strength = self.mouse.strength.min(1.0);
            self.sim.update_batched(step as f32, strength, [self.mouse.position.x, self.mouse.position.y], [self.mouse.direction.x, self.mouse.direction.y]);
            self.sim_accumulator -= step;
        }

        // the GI: the volume and the particles' light (or, off, the sky alone)
        self.scene_params.gi_on = if self.gi.settings.cones.use_volume { 1.0 } else { 0.0 };
        self.renderer.queue().write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&self.scene_params));
        let mut encoder = self.renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("VoxelGIParticles/GI") });
        let count = self.sim.particle_count();
        self.gi.encode(self.renderer.queue(), &mut encoder, count);
        self.renderer.submit(std::iter::once(encoder.finish()));

        self.renderer.render(&mut self.scene, &mut self.camera);

        if let Some((start, frames)) = &mut self.stats {
            *frames += 1;
            if *frames == 240 {
                log::info!("frame interval {:.2} ms", (now_ms() - *start) / 240.0);
                *start = now_ms();
                *frames = 0;
            }
        }
    }

    fn set_particle_size(&mut self, size: f32) {
        if let Some(r) = self.scene.get_renderable_mut(self.particles_index) {
            if let Some(buffer) = r.material.bindable_buffer(0) {
                self.renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&[0.85f32, 0.88, 0.92, 1.0, size, 0.0, 0.0, 0.0]));
            }
        }
    }
}

thread_local! { static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) }; }

fn with_state<F: FnOnce(&mut State)>(f: F) {
    STATE.with(|s| {
        if let Some(rc) = s.borrow().as_ref() {
            if let Ok(mut st) = rc.try_borrow_mut() {
                f(&mut st);
            }
        }
    });
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document()
        .unwrap()
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let width = canvas.client_width().max(1) as u32;
    let height = canvas.client_height().max(1) as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 4, clear_color: Vec4::new(0.32, 0.42, 0.6, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;

    let phone = is_phone();
    let quality = query_param("quality").as_deref().and_then(VoxelGiQuality::from_name).unwrap_or(if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium });
    let count: usize = query_param("particles").and_then(|v| v.parse().ok()).unwrap_or(if phone { 16_384 } else { 32_768 }).clamp(1024, 262_144);
    let gi_on = query_param("gi").as_deref() != Some("off");
    let indirect = query_param("view").as_deref() == Some("indirect");

    // the fluid
    let initial = initial_positions(count);
    let mut sim = FluidSimulation::new(
        &renderer,
        FluidSimulationOptions {
            max_particles: count as u32,
            dimensions: 3,
            smoothing_radius: 1.0,
            pressure_multiplier: 46.5,
            near_pressure_multiplier: 20.0,
            density_target: 8.6,
            viscosity: 1.0,
            damping: 1.0,
            gravity: [0.0, -9.8, 0.0],
            mouse_force: 1600.0,
            substeps: 2,
            world_bounds_padding: 0.3,
            ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
        },
        &initial,
    );
    sim.world_bounds_min = ROOM_MIN;
    sim.world_bounds_max = ROOM_MAX;
    sim.rebuild_grid();
    let positions = sim.positions_buffer().unwrap().clone();
    let velocities = sim.velocities_buffer().cloned();

    // the GI over the room and its walls
    let bounds_min = [ROOM_MIN[0] - WALL, ROOM_MIN[1] - WALL, ROOM_MIN[2] - WALL];
    let bounds_max = [ROOM_MAX[0] + WALL, ROOM_MAX[1], ROOM_MAX[2]];
    let options = ParticleGiOptions { quality, bounds_min, bounds_max, capacity: count as u32, radiance_scale: 1.0, budget_bytes: if phone { 24 << 20 } else { 0 } };
    let mut gi = ParticleGi::new(renderer.device(), options, &positions, velocities.as_ref());
    // low from the right, so the red wall is lit and the fluid shadows itself and the back wall
    let to_sun = sun_direction(40.0, 70.0);
    let sun = [2.6, 2.45, 2.2];
    let (sky_up, sky_down) = ([0.25, 0.35, 0.55], [0.04, 0.04, 0.04]);
    gi.set_sky_gradient(renderer.queue(), sky_up, sky_down);
    gi.settings.set_sun(to_sun, sun);
    gi.settings.splat.extinction = 0.6;
    gi.settings.splat.box_sky_scale = 1.0;
    gi.settings.emission = ParticleEmission { color: [6.0, 2.2, 0.5], share: 0.03, speed_color: [0.0; 3], ..Default::default() };
    gi.settings.cones.use_volume = gi_on;
    let walls = walls();
    let boxes: Vec<GiBox> = walls.iter().map(|w| GiBox::new(w.min, w.max, w.albedo, w.normal)).collect();
    gi.set_boxes(renderer.queue(), &boxes);
    let layout = *gi.volume().layout();
    log::info!(
        "voxel GI {:?}: {:?} voxels of {:.2} m, {:.1} MiB, {} particles",
        gi.quality(),
        layout.dims,
        layout.voxel_size,
        gi.volume().memory_bytes() as f64 / (1 << 20) as f64,
        count
    );

    let scene_params = SceneParams {
        to_sun,
        exposure: 1.2,
        sun_illuminance: sun,
        gi_on: gi_on as u32 as f32,
        sky_up,
        view: indirect as u32 as f32,
        sky_down,
        sun_cone_tan: gi.settings.cones.sun_cone_tan,
        cone_steps: gi.settings.cones.max_steps as f32,
        box_sky_scale: gi.settings.splat.box_sky_scale,
        _pad: [0.0; 2],
    };
    use wgpu::util::DeviceExt;
    let scene_buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VoxelGIParticles/Scene"),
        contents: bytemuck::bytes_of(&scene_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let scene_uniform = || ComputeBuffer::from_external("Scene", scene_buffer.clone(), BufferType::Uniform);

    let mut scene = Scene::new();
    let fragment = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    for wall in &walls {
        let mut material = Material::new(
            wall.label,
            &format!("{VOXEL_CONES_WGSL}\n{SCENE_WGSL}\n{ROOM_WGSL}"),
            vec![Binding::uniform(0, fragment), Binding::uniform(1, fragment), Binding::uniform(2, fragment), Binding::texture_3d(3, fragment), Binding::sampler(4, fragment)],
            MaterialOptions::default(),
        );
        material.set_uniform_bindable(0, wall.label, &[wall.albedo[0], wall.albedo[1], wall.albedo[2], 1.0f32]);
        material.set_bindable(1, scene_uniform());
        material.set_bindable(2, ComputeBuffer::from_external("VoxelVolume", gi.volume().uniform().clone(), BufferType::Uniform));
        material.set_bindable(3, gi.volume().as_texture());
        material.set_bindable(4, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
        let size: [f32; 3] = std::array::from_fn(|i| wall.max[i] - wall.min[i]);
        let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material);
        slab.object.set_position((wall.min[0] + wall.max[0]) * 0.5, (wall.min[1] + wall.max[1]) * 0.5, (wall.min[2] + wall.max[2]) * 0.5);
        scene.add(SceneNode::Renderable(slab));
    }

    let particle_size = 0.45;
    let mut particle_material = Material::new(
        "Particles",
        &format!("{SCENE_WGSL}\n{PARTICLE_WGSL}"),
        vec![Binding::uniform(0, fragment), Binding::uniform(1, fragment)],
        MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
    );
    particle_material.set_uniform_bindable(0, "Particles", &[0.85f32, 0.88, 0.92, 1.0, particle_size, 0.0, 0.0, 0.0]);
    particle_material.set_bindable(1, scene_uniform());
    let instances = InstancedGeometry::new(
        PlaneGeometry::new(1.0, 1.0),
        count as u32,
        vec![sim.positions_as_compute_buffer(3).unwrap(), gi.lighting_instance_buffer(4)],
    );
    let particles_index = scene.add(SceneNode::Renderable(Renderable::new(instances, particle_material)));

    let mut camera = Camera::new(40.0, 0.5, 600.0, width as f32 / height as f32);
    camera.update_projection_matrix();
    // far enough that the room fits across a portrait screen too
    let aspect = width as f32 / height as f32;
    let mut controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 8.0, 0.0), 58.0 * (1.6 / aspect).max(1.0));
    controls.set_elevation(0.45);
    controls.set_azimuth(0.18);
    let mouse = MouseVectors::from_canvas(&canvas);

    let stats = (query_param("stats").as_deref() == Some("1")).then(|| (now_ms(), 0));
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        controls,
        mouse,
        sim,
        gi,
        scene_params,
        scene_buffer,
        particles_index,
        initial,
        sim_accumulator: 0.0,
        last_time: now_ms(),
        stats,
        frame_ms: 16.7,
        paused: false,
    }));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));

    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        state.borrow_mut().frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

/// The tier in use, the volume's size and memory, the particles and the frame time, as JSON.
#[wasm_bindgen]
pub fn info() -> String {
    let mut out = String::from("{}");
    with_state(|s| {
        let layout = s.gi.volume().layout();
        out = format!(
            r#"{{"quality":"{:?}","dims":[{},{},{}],"voxel_m":{:.3},"mib":{:.2},"particles":{},"gi":{},"frame_ms":{:.2}}}"#,
            s.gi.quality(),
            layout.dims[0],
            layout.dims[1],
            layout.dims[2],
            layout.voxel_size,
            s.gi.volume().memory_bytes() as f64 / (1 << 20) as f64,
            s.sim.particle_count(),
            s.gi.settings.cones.use_volume,
            s.frame_ms
        );
    });
    out
}

#[wasm_bindgen]
pub fn set_gi(on: bool) {
    with_state(|s| {
        if s.gi.settings.cones.use_volume != on {
            s.gi.settings.cones.use_volume = on;
            s.gi.reset_history();
        }
    });
}

/// 0: lit, 1: indirect light only.
#[wasm_bindgen]
pub fn set_view(view: u32) {
    with_state(|s| s.scene_params.view = view.min(1) as f32);
}

#[wasm_bindgen]
pub fn set_extinction(v: f32) {
    with_state(|s| s.gi.settings.splat.extinction = v.max(0.0));
}

/// The diffuse cones' full aperture, degrees (90: six cones tiling the sphere).
#[wasm_bindgen]
pub fn set_cone_aperture(degrees: f32) {
    with_state(|s| s.gi.settings.cones.diffuse_cone_tan = (degrees.clamp(5.0, 150.0).to_radians() * 0.5).tan());
}

/// The sun cone's full aperture, degrees: the softness of the fluid's shadow.
#[wasm_bindgen]
pub fn set_sun_cone(degrees: f32) {
    with_state(|s| {
        let tan = (degrees.clamp(0.5, 60.0).to_radians() * 0.5).tan();
        s.gi.settings.cones.sun_cone_tan = tan;
        s.scene_params.sun_cone_tan = tan;
    });
}

#[wasm_bindgen]
pub fn set_emissive_share(share: f32) {
    with_state(|s| s.gi.settings.emission.share = share.clamp(0.0, 1.0));
}

#[wasm_bindgen]
pub fn set_emissive_intensity(v: f32) {
    with_state(|s| s.gi.settings.emission.color = [6.0, 2.2, 0.5].map(|c| c * v.max(0.0) / 6.0));
}

/// Glow per unit of speed (a blue added to every particle).
#[wasm_bindgen]
pub fn set_speed_glow(v: f32) {
    with_state(|s| s.gi.settings.emission.speed_color = [0.02, 0.06, 0.15].map(|c| c * v.max(0.0)));
}

#[wasm_bindgen]
pub fn set_temporal_blend(v: f32) {
    with_state(|s| s.gi.settings.cones.temporal_blend = v.clamp(0.01, 1.0));
}

#[wasm_bindgen]
pub fn set_particle_size(v: f32) {
    with_state(|s| s.set_particle_size(v.max(0.01)));
}

/// The sun's elevation and bearing, degrees.
#[wasm_bindgen]
pub fn set_sun(elevation: f32, bearing: f32) {
    with_state(|s| {
        let to_sun = sun_direction(elevation, bearing);
        s.scene_params.to_sun = to_sun;
        let illuminance = s.scene_params.sun_illuminance;
        s.gi.settings.set_sun(to_sun, illuminance);
    });
}

#[wasm_bindgen]
pub fn set_exposure(v: f32) {
    with_state(|s| s.scene_params.exposure = v.max(0.0));
}

/// Freeze the fluid (the GI keeps running), for comparing views of one moment.
#[wasm_bindgen]
pub fn set_paused(paused: bool) {
    with_state(|s| s.paused = paused);
}

/// Pour the dam again.
#[wasm_bindgen]
pub fn reset() {
    with_state(|s| {
        s.sim.reset_particles(&s.initial);
        s.gi.reset_history();
    });
}
