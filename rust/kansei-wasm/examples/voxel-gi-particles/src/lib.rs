//! Voxel GI on particles: Hector Arellano's "indirect lighting on particles" (miaumiau.cat,
//! p=1476) on Kansei's `gi` module. Every frame the particles of an SPH fluid splat their density
//! and emission into a voxel volume, with the walls as analytic boxes; each particle then cone
//! traces its incoming light (six 90-degree cones) and its visibility of the key light (one
//! narrow cone). So the particles shade each other: the pile darkens inside, glowing particles
//! light their neighbours, and the walls, tracing the same volume, take the pile's shadow and glow.
//!
//! Two scenes:
//! - the lightbox (default), the article's look: a closed white room lit by an emissive panel in
//!   its ceiling, on black, over a glossy floor that reflects it. The particles are spheres of
//!   varied size, charcoal and brown with a share glowing orange to yellow, raining down into a
//!   pile. `rt=on` is the article's bonus: some particles turn to mirrors and glass, their rays
//!   walking the fluid's own neighbour grid;
//! - `scene=cornell`: the fluid dam-breaks in an open-topped Cornell room (red and green side
//!   walls) under a sun and a blue sky.
//!
//! URL parameters: `scene=lightbox|cornell`, `gi=on|off` (default on), `view=indirect` (only the
//! light the volume brings, without the direct light, a stop brighter), `rt=on` (lightbox),
//! `quality=low|medium|high` (default medium, low on phones, which also keep the volume within
//! 24 MiB), `particles=N` (default 12288 in the lightbox, 8192 ray traced, 32768 in the Cornell
//! room, half on phones), `stats=1` (log the frame interval).
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Look {
    Lightbox,
    Cornell,
}

impl Look {
    fn name(self) -> &'static str {
        match self {
            Look::Lightbox => "lightbox",
            Look::Cornell => "cornell",
        }
    }

    /// The room's inside (the fluid's bounds), metres.
    fn room(self) -> ([f32; 3], [f32; 3]) {
        match self {
            Look::Lightbox => ([-14.0, 0.0, -7.0], [14.0, 15.0, 7.0]),
            Look::Cornell => ([-14.0, 0.0, -7.0], [14.0, 22.0, 7.0]),
        }
    }

    /// The box the fluid is kept in: the room, but in the lightbox short of the open front, leaving
    /// a strip of floor where the pile's glow pools in view.
    fn fluid(self) -> ([f32; 3], [f32; 3]) {
        let (min, mut max) = self.room();
        if self == Look::Lightbox {
            max[2] -= 1.5;
        }
        (min, max)
    }

    /// The walls' thickness, outside the room.
    fn wall(self) -> f32 {
        match self {
            Look::Lightbox => 0.8,
            Look::Cornell => 2.0,
        }
    }
}

/// The lightbox's panel, in the ceiling: x and z extents, metres.
const PANEL_X: [f32; 2] = [-11.5, 11.5];
const PANEL_Z: [f32; 2] = [-4.0, 4.0];
/// Its radiance at intensity 1, a cool white.
const PANEL_RADIANCE: [f32; 3] = [7.4, 7.9, 8.2];
/// The lightbox's walls, a pale green-grey.
const LIGHTBOX_ALBEDO: [f32; 3] = [0.72, 0.79, 0.77];
/// The particles' glow (scene radiance), tinted per particle by the material.
const GLOW: [f32; 3] = [24.0, 8.8, 2.0];

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
    ao_distance: f32,
    ao_strength: f32,
    panel_min: [f32; 3],
    panel_cone_tan: f32,
    panel_max: [f32; 3],
    mirror_y: f32,
    panel_radiance: [f32; 3],
    reflectivity: f32,
    room_min: [f32; 3],
    reflect_fade: f32,
    room_max: [f32; 3],
    ior: f32,
    wall_albedo: [f32; 3],
    rt_on: f32,
    rt_bounces: f32,
    mirror_share: f32,
    glass_share: f32,
    _pad: f32,
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
    aoDistance     : f32,    // lightbox: how far the walls' occlusion cones look, metres
    aoStrength     : f32,
    panelMin       : vec3f,  // lightbox: the ceiling panel's corners (its emitting face at panelMin.y)
    panelConeTan   : f32,    // tan of the half aperture of the cones toward it
    panelMax       : vec3f,
    mirrorY        : f32,    // the glossy floor under the box
    panelRadiance  : vec3f,
    reflectivity   : f32,
    roomMin        : vec3f,
    reflectFade    : f32,    // metres over which the reflection fades
    roomMax        : vec3f,
    ior            : f32,    // the glass particles' (ray traced)
    wallAlbedo     : vec3f,
    rtOn           : f32,
    rtBounces      : f32,
    mirrorShare    : f32,
    glassShare     : f32,
    _pad0          : f32,
}

const PI: f32 = 3.14159265;

// the sky as gi::gradient_sky_lighting makes it: linear in the direction's height (in the
// lightbox, the ceiling and the open front the volume leaves out)
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

const ROOM_COMMON_WGSL: &str = include_str!("shaders/room_common.wgsl");
const ROOM_WALLS_WGSL: &str = include_str!("shaders/room_walls.wgsl");
const ROOM_SPHERES_WGSL: &str = include_str!("shaders/room_spheres.wgsl");

fn lightbox_wall_shader() -> String {
    format!("{VOXEL_CONES_WGSL}\n{SCENE_WGSL}\n{ROOM_COMMON_WGSL}\n{ROOM_WALLS_WGSL}")
}

fn lightbox_sphere_shader() -> String {
    format!("{VOXEL_CONES_WGSL}\n{SCENE_WGSL}\n{ROOM_COMMON_WGSL}\n{ROOM_SPHERES_WGSL}")
}

fn cornell_wall_shader() -> String {
    format!("{VOXEL_CONES_WGSL}\n{SCENE_WGSL}\n{CORNELL_WALL_WGSL}")
}

fn cornell_particle_shader() -> String {
    format!("{SCENE_WGSL}\n{CORNELL_PARTICLE_WGSL}")
}

/// The Cornell room's walls: lit by the sun and the sky, and with voxel GI on, shadowed by the
/// fluid (a sun cone) and lit by the volume (five cones over the hemisphere: the other walls'
/// bounce, the fluid's glow, the sky it leaves).
const CORNELL_WALL_WGSL: &str = r#"
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

/// The Cornell room's particles: camera-facing discs lit by what the GI gathered for them
/// (instance attributes 4 and 5: incoming light and sun visibility, emission).
const CORNELL_PARTICLE_WGSL: &str = r#"
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

/// The lightbox's panel hangs this far under the ceiling (its emitting face).
const PANEL_DROP: f32 = 0.04;
/// tan of the half aperture of the cones toward the panel (the particles' and the walls'): the
/// softness of the pile's shadow.
const PANEL_CONE_TAN: f32 = 0.4;
/// The lightbox's "sky" (what the volume leaves out: the ceiling around the panel above, the open
/// front level), per unit of the panel's radiance.
const SKY_UP: f32 = 0.18;
const SKY_DOWN: f32 = 0.05;

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

/// `count` particles on a jittered lattice filling `lo`..`hi`.
fn lattice(count: usize, lo: [f32; 3], hi: [f32; 3]) -> Vec<f32> {
    let size: [f32; 3] = std::array::from_fn(|i| hi[i] - lo[i]);
    let spacing = (size[0] * size[1] * size[2] / count as f32).cbrt();
    let cells: [usize; 3] = std::array::from_fn(|i| ((size[i] / spacing).floor() as usize).max(1));
    let mut rng: u32 = 12345;
    let mut jitter = || {
        rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
        ((rng >> 8) as f32 / 16777216.0 - 0.5) * spacing * 0.3
    };
    let mut positions = Vec::with_capacity(count * 4);
    // front (+z, toward the camera) first: particles keep their order, so they are drawn
    // roughly front to back and the depth test spares the shading of those behind
    'fill: for layer in 0.. {
        for z in (0..cells[2]).rev() {
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

/// The lightbox: a rain over the whole room that settles into a pile. The Cornell room: a dam
/// filling its left half.
fn initial_positions(look: Look, count: usize) -> Vec<f32> {
    let (min, max) = look.fluid();
    match look {
        Look::Lightbox => lattice(count, [min[0] + 0.5, 2.0, min[2] + 0.5], [max[0] - 0.5, max[1] - 0.5, max[2] - 0.5]),
        Look::Cornell => lattice(count, [min[0] + 0.5, min[1] + 0.5, min[2] + 0.5], [-1.0, 16.0, max[2] - 0.5]),
    }
}

struct Wall {
    label: &'static str,
    min: [f32; 3],
    max: [f32; 3],
    albedo: [f32; 3],
    normal: [f32; 3],
    /// Drawn (the lightbox's front wall is not: the camera looks in through it).
    render: bool,
    /// In the volume (the lightbox's ceiling is not: the volume stops under it, and the sky
    /// gradient stands in for it).
    gi: bool,
    /// The lightbox's emissive panel.
    panel: bool,
}

fn walls(look: Look) -> Vec<Wall> {
    let ([x0, y0, z0], [x1, y1, z1]) = look.room();
    let w = look.wall();
    let wall = |label, min, max, albedo, normal| Wall { label, min, max, albedo, normal, render: true, gi: true, panel: false };
    match look {
        Look::Lightbox => {
            let a = LIGHTBOX_ALBEDO;
            vec![
                wall("Floor", [x0 - w, y0 - w, z0 - w], [x1 + w, y0, z1], a, [0.0, 1.0, 0.0]),
                wall("Back", [x0 - w, y0, z0 - w], [x1 + w, y1, z0], a, [0.0, 0.0, 1.0]),
                wall("Left", [x0 - w, y0, z0], [x0, y1, z1], a, [1.0, 0.0, 0.0]),
                wall("Right", [x1, y0, z0], [x1 + w, y1, z1], a, [-1.0, 0.0, 0.0]),
                Wall { gi: false, ..wall("Ceiling", [x0 - w, y1, z0 - w], [x1 + w, y1 + w, z1], a, [0.0, -1.0, 0.0]) },
                Wall { gi: false, panel: true, ..wall("Panel", [PANEL_X[0], y1 - PANEL_DROP, PANEL_Z[0]], [PANEL_X[1], y1, PANEL_Z[1]], [0.0; 3], [0.0, -1.0, 0.0]) },
            ]
        }
        Look::Cornell => {
            let white = [0.75, 0.75, 0.75];
            vec![
                wall("Floor", [x0 - w, y0 - w, z0 - w], [x1 + w, y0, z1], white, [0.0, 1.0, 0.0]),
                wall("Back", [x0 - w, y0, z0 - w], [x1 + w, y1, z0], white, [0.0, 0.0, 1.0]),
                wall("Left", [x0 - w, y0, z0], [x0, y1, z1], [0.75, 0.08, 0.06], [1.0, 0.0, 0.0]),
                wall("Right", [x1, y0, z0], [x1 + w, y1, z1], [0.1, 0.6, 0.12], [-1.0, 0.0, 0.0]),
            ]
        }
    }
}

/// room_common.wgsl's `panelIrradiance` per unit of the panel's radiance (its face at height `y`).
fn panel_irradiance(p: glam::Vec3, n: glam::Vec3, y: f32) -> f32 {
    let corners = [
        glam::Vec3::new(PANEL_X[0], y, PANEL_Z[0]),
        glam::Vec3::new(PANEL_X[1], y, PANEL_Z[0]),
        glam::Vec3::new(PANEL_X[1], y, PANEL_Z[1]),
        glam::Vec3::new(PANEL_X[0], y, PANEL_Z[1]),
    ];
    let mut f = 0.0;
    for k in 0..4 {
        let a = (corners[k] - p).normalize();
        let b = (corners[(k + 1) % 4] - p).normalize();
        let g = b.cross(a);
        if g.length() > 1e-6 {
            f += a.dot(b).clamp(-1.0, 1.0).acos() * g.normalize().dot(n);
        }
    }
    (0.5 * f).max(0.0)
}

/// The panel's mean irradiance over the part of `wall`'s lit face inside the room, per unit of its
/// radiance: what the volume stores the wall as (its light bounces from there; the walls draw it
/// per pixel, shadowed).
fn mean_panel_irradiance(wall: &Wall, look: Look) -> f32 {
    let (lo, hi) = look.room();
    let n = glam::Vec3::from(wall.normal);
    let axis = (0..3).find(|&i| wall.normal[i] != 0.0).unwrap_or(1);
    let face = if wall.normal[axis] > 0.0 { wall.max[axis] } else { wall.min[axis] };
    let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
    let span = |i: usize, t: f32| {
        let (s0, s1) = (wall.min[i].max(lo[i]), wall.max[i].min(hi[i]));
        s0 + (s1 - s0) * t
    };
    let mut sum = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            let mut p = [0.0; 3];
            p[axis] = face;
            p[a] = span(a, (i as f32 + 0.5) / 8.0);
            p[b] = span(b, (j as f32 + 0.5) / 8.0);
            sum += panel_irradiance(glam::Vec3::from(p), n, hi[1] - PANEL_DROP);
        }
    }
    sum / 64.0
}

/// The fluid's neighbour grid, for the ray-traced spheres (room_spheres.wgsl's `Grid`).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    origin: [f32; 3],
    cell_size: f32,
    dims: [u32; 3],
    count: u32,
}

struct State {
    look: Look,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    sim: FluidSimulation,
    gi: ParticleGi,
    scene_params: SceneParams,
    scene_buffer: wgpu::Buffer,
    walls: Vec<Wall>,
    /// The particles' renderables, and whether each is the floor's reflection.
    particles: Vec<(usize, bool)>,
    /// The lightbox's panel renderables, likewise.
    panels: Vec<(usize, bool)>,
    panel_intensity: f32,
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
        for &(index, mirrored) in &self.particles {
            if let Some(buffer) = self.scene.get_renderable_mut(index).and_then(|r| r.material.bindable_buffer(0)) {
                self.renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&[0.85f32, 0.88, 0.92, 1.0, size, mirrored as u32 as f32, 0.0, 0.0]));
            }
        }
    }

    /// The lightbox's light at `panel_intensity`: the panel, the walls as the volume holds them
    /// (lit by it) and the sky standing in for the ceiling.
    fn apply_panel(&mut self) {
        if self.look != Look::Lightbox {
            return;
        }
        let radiance = PANEL_RADIANCE.map(|c| c * self.panel_intensity);
        let (up, down) = (radiance.map(|c| c * SKY_UP), radiance.map(|c| c * SKY_DOWN));
        self.scene_params.panel_radiance = radiance;
        self.scene_params.sky_up = up;
        self.scene_params.sky_down = down;
        let queue = self.renderer.queue();
        self.gi.set_sky_gradient(queue, up, down);
        let boxes: Vec<GiBox> = self
            .walls
            .iter()
            .filter(|w| w.gi)
            .map(|w| {
                let e = mean_panel_irradiance(w, self.look) / std::f32::consts::PI;
                GiBox::new(w.min, w.max, w.albedo, w.normal).with_emission(std::array::from_fn(|c| w.albedo[c] * e * radiance[c]))
            })
            .collect();
        self.gi.set_boxes(queue, &boxes);
        for &(index, mirrored) in &self.panels {
            if let Some(buffer) = self.scene.get_renderable_mut(index).and_then(|r| r.material.bindable_buffer(0)) {
                let surface = [0.0, 0.0, 0.0, 1.0, radiance[0], radiance[1], radiance[2], 1.0, mirrored as u32 as f32, 0.0, 0.0, 0.0];
                self.renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&surface));
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

// browser only (the controls read the canvas); the shaders' tests run natively
#[cfg(target_arch = "wasm32")]
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

    let look = if query_param("scene").as_deref() == Some("cornell") { Look::Cornell } else { Look::Lightbox };
    let lightbox = look == Look::Lightbox;
    let clear_color = if lightbox { Vec4::new(0.0, 0.0, 0.0, 1.0) } else { Vec4::new(0.32, 0.42, 0.6, 1.0) };
    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 4, clear_color, ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;

    let phone = is_phone();
    let quality = query_param("quality").as_deref().and_then(VoxelGiQuality::from_name).unwrap_or(if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium });
    let rt = lightbox && query_param("rt").as_deref() == Some("on");
    // ray traced, fewer and larger, as in the article's bonus
    let default_count = (if rt { 8_192 } else if lightbox { 12_288 } else { 32_768 }) / (if phone { 2 } else { 1 });
    let count: usize = query_param("particles").and_then(|v| v.parse().ok()).unwrap_or(default_count).clamp(1024, 262_144);
    let gi_on = query_param("gi").as_deref() != Some("off");
    let indirect = query_param("view").as_deref() == Some("indirect");
    let (room_min, room_max) = look.room();
    let wall = look.wall();

    // the fluid; in the lightbox damped and viscous, so the rain settles into a pile
    let initial = initial_positions(look, count);
    let mut sim = FluidSimulation::new(
        &renderer,
        FluidSimulationOptions {
            max_particles: count as u32,
            dimensions: 3,
            smoothing_radius: 1.0,
            pressure_multiplier: 46.5,
            near_pressure_multiplier: 20.0,
            density_target: 8.6,
            viscosity: if lightbox { 3.0 } else { 1.0 },
            damping: if lightbox { 0.994 } else { 1.0 },
            gravity: [0.0, -9.8, 0.0],
            mouse_force: 1600.0,
            substeps: 2,
            world_bounds_padding: 0.3,
            ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
        },
        &initial,
    );
    (sim.world_bounds_min, sim.world_bounds_max) = look.fluid();
    sim.rebuild_grid();
    let positions = sim.positions_buffer().unwrap().clone();
    let velocities = sim.velocities_buffer().cloned();

    // the GI over the room and its walls, up to the ceiling and the open front (in the lightbox a
    // little past it, so the frame's front faces gather the pile's glow)
    let bounds_min = [room_min[0] - wall, room_min[1] - wall, room_min[2] - wall];
    let bounds_max = [room_max[0] + wall, room_max[1], room_max[2] + if lightbox { 1.5 } else { 0.0 }];
    let options = ParticleGiOptions { quality, bounds_min, bounds_max, capacity: count as u32, radiance_scale: 1.0, budget_bytes: if phone { 24 << 20 } else { 0 } };
    let mut gi = ParticleGi::new(renderer.device(), options, &positions, velocities.as_ref());
    gi.settings.splat.extinction = 0.6;
    gi.settings.cones.use_volume = gi_on;
    let walls = walls(look);
    let mut scene_params = SceneParams {
        to_sun: [0.0, 1.0, 0.0],
        exposure: 1.2,
        sun_illuminance: [0.0; 3],
        gi_on: gi_on as u32 as f32,
        sky_up: [0.0; 3],
        view: indirect as u32 as f32,
        sky_down: [0.0; 3],
        sun_cone_tan: gi.settings.cones.sun_cone_tan,
        cone_steps: gi.settings.cones.max_steps as f32,
        box_sky_scale: 1.0,
        ao_distance: 5.0,
        ao_strength: 0.85,
        panel_min: [PANEL_X[0], room_max[1] - PANEL_DROP, PANEL_Z[0]],
        panel_cone_tan: PANEL_CONE_TAN,
        panel_max: [PANEL_X[1], room_max[1], PANEL_Z[1]],
        mirror_y: room_min[1] - wall,
        panel_radiance: [0.0; 3],
        reflectivity: 0.4,
        room_min,
        reflect_fade: 7.0,
        room_max,
        ior: 1.5,
        wall_albedo: LIGHTBOX_ALBEDO,
        rt_on: rt as u32 as f32,
        rt_bounces: 2.0,
        mirror_share: 0.3,
        glass_share: 0.35,
        _pad: 0.0,
    };
    if lightbox {
        // the panel above: no sun, the walls lit through their emission (apply_panel), the
        // particles' narrow cone looking up at the panel
        gi.settings.splat.sun_illuminance = [0.0; 3];
        gi.settings.splat.box_sky_scale = 0.0;
        gi.settings.cones.to_sun = [0.0, 1.0, 0.0];
        gi.settings.cones.sun_cone_tan = PANEL_CONE_TAN;
        gi.settings.emission = ParticleEmission { color: GLOW, share: if rt { 0.0 } else { 0.25 }, speed_color: [0.0; 3], ..Default::default() };
        scene_params.exposure = 1.3;
    } else {
        // low from the right, so the red wall is lit and the fluid shadows itself and the back wall
        let to_sun = sun_direction(40.0, 70.0);
        let sun = [2.6, 2.45, 2.2];
        let (sky_up, sky_down) = ([0.25, 0.35, 0.55], [0.04, 0.04, 0.04]);
        gi.set_sky_gradient(renderer.queue(), sky_up, sky_down);
        gi.settings.set_sun(to_sun, sun);
        gi.settings.splat.box_sky_scale = 1.0;
        gi.settings.emission = ParticleEmission { color: GLOW.map(|c| c * 0.25), share: 0.03, speed_color: [0.0; 3], ..Default::default() };
        let boxes: Vec<GiBox> = walls.iter().map(|w| GiBox::new(w.min, w.max, w.albedo, w.normal)).collect();
        gi.set_boxes(renderer.queue(), &boxes);
        scene_params.to_sun = to_sun;
        scene_params.sun_illuminance = sun;
        scene_params.sky_up = sky_up;
        scene_params.sky_down = sky_down;
        scene_params.sun_cone_tan = gi.settings.cones.sun_cone_tan;
    }
    let layout = *gi.volume().layout();
    log::info!(
        "voxel GI {:?} ({}): {:?} voxels of {:.2} m, {:.1} MiB, {} particles",
        gi.quality(),
        look.name(),
        layout.dims,
        layout.voxel_size,
        gi.volume().memory_bytes() as f64 / (1 << 20) as f64,
        count
    );

    use wgpu::util::DeviceExt;
    let scene_buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("VoxelGIParticles/Scene"),
        contents: bytemuck::bytes_of(&scene_params),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let scene_uniform = || ComputeBuffer::from_external("Scene", scene_buffer.clone(), BufferType::Uniform);
    let volume_uniform = || ComputeBuffer::from_external("VoxelVolume", gi.volume().uniform().clone(), BufferType::Uniform);
    let linear_clamp = || Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge);

    let mut scene = Scene::new();
    let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    let fragment = ShaderStages::FRAGMENT;
    let mut panels = Vec::new();
    // the walls; in the lightbox each again mirrored under the floor
    let mirrors: &[bool] = if lightbox { &[false, true] } else { &[false] };
    let wall_shader = if lightbox { lightbox_wall_shader() } else { cornell_wall_shader() };
    for wall in walls.iter().filter(|w| w.render) {
        for &mirrored in mirrors {
            let cull_mode = if mirrored { CullMode::None } else { CullMode::Back };
            let mut material = Material::new(
                wall.label,
                &wall_shader,
                vec![Binding::uniform(0, both), Binding::uniform(1, both), Binding::uniform(2, both), Binding::texture_3d(3, both), Binding::sampler(4, both)],
                MaterialOptions { cull_mode, ..Default::default() },
            );
            let [r, g, b] = wall.albedo;
            if lightbox {
                // the panel's radiance follows set_panel (apply_panel)
                let [er, eg, eb] = if wall.panel { PANEL_RADIANCE } else { [0.0; 3] };
                material.set_uniform_bindable(0, wall.label, &[r, g, b, 1.0f32, er, eg, eb, 1.0, mirrored as u32 as f32, 0.0, 0.0, 0.0]);
            } else {
                material.set_uniform_bindable(0, wall.label, &[r, g, b, 1.0f32]);
            }
            material.set_bindable(1, scene_uniform());
            material.set_bindable(2, volume_uniform());
            material.set_bindable(3, gi.volume().as_texture());
            material.set_bindable(4, linear_clamp());
            let size: [f32; 3] = std::array::from_fn(|i| wall.max[i] - wall.min[i]);
            let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material);
            slab.object.set_position((wall.min[0] + wall.max[0]) * 0.5, (wall.min[1] + wall.max[1]) * 0.5, (wall.min[2] + wall.max[2]) * 0.5);
            let index = scene.add(SceneNode::Renderable(slab));
            if wall.panel {
                panels.push((index, mirrored));
            }
        }
    }

    // the particles: in the lightbox ray-cast spheres reading the GI's light and the fluid's grid
    // as storage, twice (the second the reflection); in the Cornell room discs reading it as
    // instance attributes
    let particle_size = if rt { 0.8 } else if lightbox { 0.55 } else { 0.45 };
    let mut particles = Vec::new();
    if lightbox {
        let grid = GridParams { origin: sim.grid_origin(), cell_size: sim.cell_size(), dims: sim.grid_dims(), count: sim.particle_count() };
        let grid_buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("VoxelGIParticles/Grid"),
            contents: bytemuck::bytes_of(&grid),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let storage = |label: &str, buffer: &wgpu::Buffer| ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage);
        let shader = lightbox_sphere_shader();
        for mirrored in [false, true] {
            let mut material = Material::new(
                "Spheres",
                &shader,
                vec![
                    Binding::uniform(0, both),
                    Binding::uniform(1, both),
                    Binding::uniform(2, both),
                    Binding::storage(3, fragment, true),
                    Binding::storage(4, fragment, true),
                    Binding::storage(5, fragment, true),
                    Binding::storage(6, fragment, true),
                    Binding::uniform(7, fragment),
                    Binding::texture_3d(8, fragment),
                    Binding::sampler(9, fragment),
                ],
                MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
            );
            material.set_uniform_bindable(0, "Particles", &[0.85f32, 0.88, 0.92, 1.0, particle_size, mirrored as u32 as f32, 0.0, 0.0]);
            material.set_bindable(1, scene_uniform());
            material.set_bindable(2, ComputeBuffer::from_external("Grid", grid_buffer.clone(), BufferType::Uniform));
            material.set_bindable(3, storage("SortedPositions", sim.sorted_positions_buffer().unwrap()));
            material.set_bindable(4, storage("CellOffsets", sim.cell_offsets_buffer().unwrap()));
            material.set_bindable(5, storage("SortedIndices", sim.sorted_indices_buffer().unwrap()));
            material.set_bindable(6, storage("ParticleLighting", gi.lighting_buffer()));
            material.set_bindable(7, volume_uniform());
            material.set_bindable(8, gi.volume().as_texture());
            material.set_bindable(9, linear_clamp());
            let instances = InstancedGeometry::new(PlaneGeometry::new(1.0, 1.0), count as u32, vec![sim.positions_as_compute_buffer(3).unwrap()]);
            particles.push((scene.add(SceneNode::Renderable(Renderable::new(instances, material))), mirrored));
        }
    } else {
        let mut material = Material::new(
            "Particles",
            &cornell_particle_shader(),
            vec![Binding::uniform(0, both), Binding::uniform(1, both)],
            MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
        );
        material.set_uniform_bindable(0, "Particles", &[0.85f32, 0.88, 0.92, 1.0, particle_size, 0.0, 0.0, 0.0]);
        material.set_bindable(1, scene_uniform());
        let instances = InstancedGeometry::new(PlaneGeometry::new(1.0, 1.0), count as u32, vec![sim.positions_as_compute_buffer(3).unwrap(), gi.lighting_instance_buffer(4)]);
        particles.push((scene.add(SceneNode::Renderable(Renderable::new(instances, material))), false));
    }

    let mut camera = Camera::new(40.0, 0.5, 600.0, width as f32 / height as f32);
    camera.update_projection_matrix();
    // far enough that the room fits across a portrait screen too
    let aspect = width as f32 / height as f32;
    let fit = (1.6 / aspect).max(1.0);
    let mut controls = if lightbox {
        // straight on, a little low, with the reflection under the box in view
        let mut c = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 5.0, 0.0), 37.0 * fit);
        c.set_elevation(0.04);
        c
    } else {
        let mut c = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 8.0, 0.0), 58.0 * fit);
        c.set_elevation(0.45);
        c.set_azimuth(0.18);
        c
    };
    controls.update(&mut camera, 0.0);
    let mouse = MouseVectors::from_canvas(&canvas);

    let stats = (query_param("stats").as_deref() == Some("1")).then(|| (now_ms(), 0));
    let mut state = State {
        look,
        renderer,
        scene,
        camera,
        controls,
        mouse,
        sim,
        gi,
        scene_params,
        scene_buffer,
        walls,
        particles,
        panels,
        panel_intensity: 1.0,
        initial,
        sim_accumulator: 0.0,
        last_time: now_ms(),
        stats,
        frame_ms: 16.7,
        paused: false,
    };
    state.apply_panel();
    let state = Rc::new(RefCell::new(state));
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

/// The scene, the tier in use, the volume's size and memory, the particles and the frame time,
/// as JSON.
#[wasm_bindgen]
pub fn info() -> String {
    let mut out = String::from("{}");
    with_state(|s| {
        let layout = s.gi.volume().layout();
        out = format!(
            r#"{{"scene":"{}","rt":{},"quality":"{:?}","dims":[{},{},{}],"voxel_m":{:.3},"mib":{:.2},"particles":{},"gi":{},"frame_ms":{:.2}}}"#,
            s.look.name(),
            s.scene_params.rt_on > 0.5,
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

/// The full aperture of the cone toward the key light (the sun, or the lightbox's panel),
/// degrees: the softness of the particles' shadow.
#[wasm_bindgen]
pub fn set_sun_cone(degrees: f32) {
    with_state(|s| {
        let tan = (degrees.clamp(0.5, 120.0).to_radians() * 0.5).tan();
        s.gi.settings.cones.sun_cone_tan = tan;
        s.scene_params.sun_cone_tan = tan;
        s.scene_params.panel_cone_tan = tan;
    });
}

#[wasm_bindgen]
pub fn set_emissive_share(share: f32) {
    with_state(|s| s.gi.settings.emission.share = share.clamp(0.0, 1.0));
}

#[wasm_bindgen]
pub fn set_emissive_intensity(v: f32) {
    with_state(|s| s.gi.settings.emission.color = GLOW.map(|c| c * v.max(0.0) / GLOW[0]));
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

/// The sun's elevation and bearing, degrees (the Cornell room).
#[wasm_bindgen]
pub fn set_sun(elevation: f32, bearing: f32) {
    with_state(|s| {
        if s.look != Look::Cornell {
            return;
        }
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

/// The lightbox's panel, relative to its default.
#[wasm_bindgen]
pub fn set_panel(intensity: f32) {
    with_state(|s| {
        s.panel_intensity = intensity.max(0.0);
        s.apply_panel();
    });
}

/// How much of the box the glossy floor under it reflects (the lightbox).
#[wasm_bindgen]
pub fn set_reflectivity(v: f32) {
    with_state(|s| s.scene_params.reflectivity = v.clamp(0.0, 1.0));
}

/// How much the walls' short cones darken corners and the pile's surroundings (the lightbox).
#[wasm_bindgen]
pub fn set_ao(strength: f32) {
    with_state(|s| s.scene_params.ao_strength = strength.clamp(0.0, 1.0));
}

/// Ray-traced mirror and glass particles (the lightbox).
#[wasm_bindgen]
pub fn set_rt(on: bool) {
    with_state(|s| s.scene_params.rt_on = on as u32 as f32);
}

/// The shares of mirror and glass particles when ray traced; the rest stay matte.
#[wasm_bindgen]
pub fn set_rt_mix(mirror: f32, glass: f32) {
    with_state(|s| {
        let glass = glass.clamp(0.0, 1.0);
        s.scene_params.glass_share = glass;
        s.scene_params.mirror_share = mirror.clamp(0.0, 1.0 - glass);
    });
}

/// 1: a mirror or glass particle's rays see the others as matte; 2: they reflect and refract
/// once more.
#[wasm_bindgen]
pub fn set_rt_bounces(bounces: u32) {
    with_state(|s| s.scene_params.rt_bounces = bounces.clamp(1, 2) as f32);
}

/// Freeze the fluid (the GI keeps running), for comparing views of one moment.
#[wasm_bindgen]
pub fn set_paused(paused: bool) {
    with_state(|s| s.paused = paused);
}

/// Pour again.
#[wasm_bindgen]
pub fn reset() {
    with_state(|s| {
        s.sim.reset_particles(&s.initial);
        s.gi.reset_history();
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        module
    }

    fn struct_size(module: &naga::Module, name: &str) -> usize {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                _ => None,
            })
            .unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn shaders_validate_and_the_uniforms_match() {
        for (name, code) in [("lightbox walls", lightbox_wall_shader()), ("cornell walls", cornell_wall_shader()), ("cornell particles", cornell_particle_shader())] {
            let module = validate(name, &code);
            assert_eq!(struct_size(&module, "SceneParams"), std::mem::size_of::<SceneParams>(), "{name}");
        }
        let spheres = validate("lightbox spheres", &lightbox_sphere_shader());
        assert_eq!(struct_size(&spheres, "SceneParams"), std::mem::size_of::<SceneParams>());
        assert_eq!(struct_size(&spheres, "Grid"), std::mem::size_of::<GridParams>());
        assert_eq!(struct_size(&spheres, "Particles"), 32);
        assert_eq!(struct_size(&validate("lightbox walls", &lightbox_wall_shader()), "Surface"), 48);
    }

    #[test]
    fn the_panel_lights_the_floor_under_it_and_not_the_ceiling() {
        let y = 15.0 - PANEL_DROP;
        let under = panel_irradiance(glam::Vec3::new(0.0, 0.0, 0.0), glam::Vec3::Y, y);
        // a 23 x 8 m panel 15 m up: about its projected solid angle
        assert!(under > 0.5 && under < 1.0, "{under}");
        assert_eq!(panel_irradiance(glam::Vec3::new(0.0, 14.0, 0.0), -glam::Vec3::Y, y), 0.0);
        // a wall facing it gets less than the floor below it
        let wall = panel_irradiance(glam::Vec3::new(-14.0, 7.0, 0.0), glam::Vec3::X, y);
        assert!(wall > 0.0 && wall < under, "{wall}");
    }
}
