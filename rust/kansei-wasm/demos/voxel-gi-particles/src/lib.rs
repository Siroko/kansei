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
//! room, half on phones), `stats=1` (log the frame interval), `profile=1` (time each pass:
//! `profile_report`, with `set_layers` to time the walls, particles and reflection apart).
//!
//! Depth of field (`dof=1`, off by default): the engine's `CinematicDepthOfFieldEffect`. With it
//! on, the materials write HDR light instead of tone mapping it themselves, and a
//! `PostProcessingVolume` runs the DoF and then the same `1 - exp(-x)` curve
//! (`ToneMapper::Exponential`); the volume's GBuffer is single-sampled, so that path has no MSAA.
//! `focus=` sets the focus distance in metres (default: where the view axis enters the fluid's box,
//! the front of the particle cloud, so panning refocuses) and `fstop=` the aperture (default 1). The room is 28 m wide and seen from 37 m, where a real
//! lens blurs nothing, so the lens sees it as a 1:100 tabletop model (`DOF_MODEL_SCALE`).
#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use kansei_core::atmosphere::{direction_from_elevation_bearing, SKY_LIGHTING_WGSL};
use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler};
use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::gi::{GiBox, ParticleEmission, ParticleGi, ParticleGiOptions, VoxelGiQuality, VOXEL_CONES_WGSL};
use kansei_core::materials::{Binding, BindingResource, Compute, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FixedStep;
use kansei_core::postprocessing::effects::{CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, ToneMapEffect, ToneMapOptions, ToneMapper};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{flag, is_phone, now, param, param_or, Canvas, Frame};
use kansei_core::simulations::fluid::{fill_box, FluidSimulation, FluidSimulationOptions};
use kansei_core::simulations::grid::NEIGHBOUR_GRID_WGSL;

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
    view: f32,
    sun_cone_tan: f32,
    cone_steps: f32,
    box_sky_scale: f32,
    ao_distance: f32,
    ao_strength: f32,
    _pad0: [f32; 2],
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
    /// 1: write exposed HDR light, the depth of field's volume tone mapping it after.
    hdr: f32,
}

const SCENE_WGSL: &str = r#"
struct SceneParams {
    toSun          : vec3f,
    exposure       : f32,
    sunIlluminance : vec3f,
    giOn           : f32,
    view           : f32,    // 1: indirect light only
    sunConeTan     : f32,
    coneSteps      : f32,
    boxSkyScale    : f32,
    aoDistance     : f32,    // lightbox: how far the walls' occlusion cones look, metres
    aoStrength     : f32,
    _pad0          : vec2f,
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
    hdr            : f32,    // 1: exposed HDR out, tone mapped after the depth of field
}

const PI: f32 = 3.14159265;

fn tonemap(s: SceneParams, c: vec3f) -> vec3f {
    let k = select(1.0, 2.0, s.view > 0.5);
    let exposed = c * s.exposure * k;
    // with depth of field the volume applies the same curve (ToneMapper::Exponential) after it
    if (s.hdr > 0.5) { return exposed; }
    return 1.0 - exp(-exposed);
}
"#;

const ROOM_COMMON_WGSL: &str = include_str!("shaders/room_common.wgsl");
const ROOM_WALLS_WGSL: &str = include_str!("shaders/room_walls.wgsl");
const ROOM_SPHERES_WGSL: &str = include_str!("shaders/room_spheres.wgsl");
const ROOM_SPHERES_SHADE_WGSL: &str = include_str!("shaders/room_spheres_shade.wgsl");
const ROOM_SPHERES_DEPTH_WGSL: &str = include_str!("shaders/room_spheres_depth.wgsl");

fn lightbox_wall_shader() -> String {
    format!("{VOXEL_CONES_WGSL}\n{SKY_LIGHTING_WGSL}\n{SCENE_WGSL}\n{ROOM_COMMON_WGSL}\n{ROOM_WALLS_WGSL}")
}

/// The spheres' shader with `entry` (ROOM_SPHERES_SHADE_WGSL or ROOM_SPHERES_DEPTH_WGSL).
fn sphere_shader(entry: &str) -> String {
    format!("{VOXEL_CONES_WGSL}\n{SKY_LIGHTING_WGSL}\n{SCENE_WGSL}\n{NEIGHBOUR_GRID_WGSL}\n{ROOM_COMMON_WGSL}\n{ROOM_SPHERES_WGSL}\n{entry}")
}

fn pile_top_shader() -> String {
    format!("{NEIGHBOUR_GRID_WGSL}\n{}", include_str!("shaders/pile_top.wgsl"))
}

fn cornell_wall_shader() -> String {
    format!("{VOXEL_CONES_WGSL}\n{SKY_LIGHTING_WGSL}\n{SCENE_WGSL}\n{CORNELL_WALL_WGSL}")
}

fn cornell_particle_shader() -> String {
    format!("{SCENE_WGSL}\n{CORNELL_PARTICLE_WGSL}")
}

/// The Cornell room's walls: lit by the sun and the sky (`ParticleGi::sky_buffer`, the gradient
/// the particles' cones escape to), and with voxel GI on, shadowed by the fluid (a sun cone) and
/// lit by the volume (five cones over the hemisphere: the other walls' bounce, the fluid's glow,
/// the sky it leaves).
const CORNELL_WALL_WGSL: &str = r#"
struct Surface { albedo: vec4f };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(0) @binding(1) var<uniform> scene: SceneParams;
@group(0) @binding(2) var<uniform> vol: VoxelVolume;
@group(0) @binding(3) var radiance: texture_3d<f32>;
@group(0) @binding(4) var linearClamp: sampler;
@group(0) @binding(5) var<uniform> sky: SkyLighting;
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

// irradiance over the hemisphere around n from VOXEL_CONES_WGSL's five cones (one along it and
// four at 60 degrees, each 60 degrees wide), the sky past them
fn hemisphere(o: vec3f, n: vec3f, start: f32) -> vec3f {
    var sum = vec3f(0.0);
    for (var k = 0u; k < VOXEL_HEMISPHERE_CONES; k++) {
        let cone = voxelHemisphereCone(n, k);
        let c = voxelConeTrace(vol, radiance, linearClamp, o, cone.xyz, VOXEL_HEMISPHERE_TAN, start, 1e4, u32(scene.coneSteps));
        sum += cone.w * (c.rgb + c.a * skyRadiance(sky, cone.xyz));
    }
    return sum;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    let n = normalize(in.normal);
    let albedo = surface.albedo.rgb;
    let l = normalize(scene.toSun);
    var sunVis = 1.0;
    var irradiance = scene.boxSkyScale * skyIrradiance(sky, n);
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

/// Toward the sun at `elevation` degrees up, `bearing` degrees from +z (the open front) toward +x:
/// a compass bearing (clockwise from north, -z) of 180 - `bearing`.
fn sun_direction(elevation: f32, bearing: f32) -> [f32; 3] {
    direction_from_elevation_bearing(elevation, 180.0 - bearing).to_glam().to_array()
}

/// The lightbox: a rain over the whole room that settles into a pile. The Cornell room: a dam
/// filling its left half.
fn initial_positions(look: Look, count: usize) -> Vec<f32> {
    let (min, max) = look.fluid();
    let mut positions = match look {
        Look::Lightbox => fill_box(count, [min[0] + 0.5, 2.0, min[2] + 0.5], [max[0] - 0.5, max[1] - 0.5, max[2] - 0.5], 0.3),
        Look::Cornell => fill_box(count, [min[0] + 0.5, min[1] + 0.5, min[2] + 0.5], [-1.0, 16.0, max[2] - 0.5], 0.3),
    };
    if look == Look::Lightbox {
        set_radius_shares(&mut positions);
    }
    positions
}

/// Particle `i`'s radius as a share of the particle size, 0.45 to 1.25: the lightbox's spheres
/// read it from their position's w (which the fluid keeps), so the ray walk tests a sphere with
/// one load.
fn radius_share(i: u32) -> f32 {
    // PCG (Jarzynski & Olano 2020), as room_spheres.wgsl's hash01
    let x = i.wrapping_mul(8).wrapping_add(1);
    let state = x.wrapping_mul(747796405).wrapping_add(2891336453);
    let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
    let h = (((word >> 22) ^ word) >> 8) as f32 / 16777216.0;
    0.45 + 0.8 * h
}

fn set_radius_shares(positions: &mut [f32]) {
    for (i, p) in positions.chunks_exact_mut(4).enumerate() {
        p[3] = radius_share(i as u32);
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

/// The highest particle centre, each frame (pile_top.wgsl): rays leaving the pile upward stop
/// walking the grid above it.
struct PileTop {
    buffer: wgpu::Buffer,
    compute: Compute,
    count: u32,
}

impl PileTop {
    fn new(device: &wgpu::Device, grid: &wgpu::Buffer, sorted_positions: &wgpu::Buffer, count: u32) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("VoxelGIParticles/PileTop"),
            size: 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = wgpu::ShaderStages::COMPUTE;
        let mut compute = Compute::new("VoxelGIParticles/PileTop", &pile_top_shader(), vec![Binding::uniform(0, stage), Binding::storage(1, stage, true), Binding::storage(2, stage, false)]);
        compute.initialize(device);
        let whole = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        compute.set_bind_group(device, &[(0, whole(grid)), (1, whole(sorted_positions)), (2, whole(&buffer))]);
        Self { buffer, compute, count }
    }

    fn encode(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&0u32));
        let stamp = kansei_core::profiling::gpu_pass("VoxelGIParticles/PileTop");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelGIParticles/PileTop"), timestamp_writes: stamp.as_ref().map(kansei_core::profiling::PassStamp::compute) });
        self.compute.dispatch(&mut pass, self.count.div_ceil(64), 1, 1);
    }
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
    /// Every wall's renderable, likewise.
    slabs: Vec<(usize, bool)>,
    pile_top: Option<PileTop>,
    panel_intensity: f32,
    initial: Vec<f32>,
    /// The fluid's fixed step (60 Hz, at most 3 a frame).
    sim_step: FixedStep,
    stats: Option<(f64, u32)>,
    frame_ms: f64,
    paused: bool,
    dof: Dof,
    /// The depth of field's chain, made the first time it is turned on.
    volume: Option<PostProcessingVolume>,
    /// The background as the direct path shows it (display values).
    clear_color: Vec4,
}

/// The depth of field's settings.
#[derive(Clone, Copy)]
struct Dof {
    on: bool,
    /// Focus distance, metres; None focuses on the orbit target.
    focus: Option<f32>,
    f_stop: f32,
}

const DOF_F_STOP: f32 = 1.0;
/// The lens sees the room as a model this many times smaller: its filmback is this many times
/// Unreal's 23.76 mm, which blurs as a 23.76 mm one would on the scene scaled down (the field of
/// view, and so the picture, stay the camera's).
const DOF_MODEL_SCALE: f32 = 100.0;

/// The depth of field and then the materials' own curve: the materials write HDR light with their
/// exposure applied (`SceneParams::hdr`), and the curve's output is the value they would have
/// written to the screen (so no sRGB encoding on top).
fn dof_effects(dof: Dof, focus: f32) -> Vec<Box<dyn PostProcessingEffect>> {
    vec![
        Box::new(CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
            lens: CameraLens { f_stop: dof.f_stop, focus_distance_m: focus, sensor_width_mm: 23.76 * DOF_MODEL_SCALE, ..Default::default() },
            // no TAA here to average a rotating pattern
            temporal_noise: false,
            ..Default::default()
        })),
        Box::new(ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::Exponential, encode_srgb: false, ..Default::default() })),
    ]
}

impl State {
    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let dt = (frame.dt as f64).clamp(0.0, 0.1);
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
        let steps = if self.paused {
            self.sim_step.reset();
            0
        } else {
            self.sim_step.advance(dt)
        };
        for _ in 0..steps {
            let strength = self.mouse.strength.min(1.0);
            self.sim.update_batched(self.sim_step.step as f32, strength, [self.mouse.position.x, self.mouse.position.y], [self.mouse.direction.x, self.mouse.direction.y]);
        }

        // the GI: the volume and the particles' light (or, off, the sky alone)
        self.scene_params.hdr = self.dof.on as u32 as f32;
        self.scene_params.gi_on = if self.gi.settings.cones.use_volume { 1.0 } else { 0.0 };
        self.renderer.queue().write_buffer(&self.scene_buffer, 0, bytemuck::bytes_of(&self.scene_params));
        let mut encoder = self.renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("VoxelGIParticles/GI") });
        let count = self.sim.particle_count();
        self.gi.encode(self.renderer.queue(), &mut encoder, count);
        if let Some(top) = &self.pile_top {
            top.encode(self.renderer.queue(), &mut encoder);
        }
        self.renderer.submit(std::iter::once(encoder.finish()));

        if self.dof.on {
            let focus = self.focus_distance();
            let volume = self.volume.get_or_insert_with(|| PostProcessingVolume::new(&self.renderer, dof_effects(self.dof, focus)));
            if let Some(effect) = volume.effect_mut::<CinematicDepthOfFieldEffect>() {
                effect.lens.focus_distance_m = focus;
                effect.lens.f_stop = self.dof.f_stop;
            }
            // the background in HDR terms: what the curve maps back to the direct path's
            let c = self.clear_color;
            let hdr = |v: f32| -(1.0 - v.min(0.999)).ln();
            self.renderer.config.clear_color = Vec4::new(hdr(c.x), hdr(c.y), hdr(c.z), c.w);
            self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, volume);
        } else {
            self.renderer.config.clear_color = self.clear_color;
            self.renderer.render(&mut self.scene, &mut self.camera);
        }

        if let Some((start, frames)) = &mut self.stats {
            *frames += 1;
            if *frames == 240 {
                log::info!("frame interval {:.2} ms", (now() - *start) * 1000.0 / 240.0);
                *start = now();
                *frames = 0;
            }
        }
    }

    /// The depth of field's focus distance: the set one, or where the view axis (toward the orbit
    /// target) enters the fluid's box, the front of the particle cloud; the target's distance if
    /// the axis misses the box or starts inside it.
    fn focus_distance(&self) -> f32 {
        self.dof.focus.unwrap_or_else(|| {
            let eye = glam::Vec3::from(self.camera.object.position);
            let to_target = glam::Vec3::from(self.controls.look_target()) - eye;
            let axis = to_target.normalize_or_zero();
            let (lo, hi) = self.look.fluid();
            let (t0, t1) = (glam::Vec3::from(lo) - eye, glam::Vec3::from(hi) - eye);
            let inv = axis.recip();
            let (near, far) = ((t0 * inv).min(t1 * inv).max_element(), (t0 * inv).max(t1 * inv).min_element());
            if near > 0.0 && near <= far { near } else { to_target.length() }.max(0.1)
        })
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
    let page = Canvas::find(canvas_id)?;
    let canvas = page.element();

    let look = if param("scene").as_deref() == Some("cornell") { Look::Cornell } else { Look::Lightbox };
    let lightbox = look == Look::Lightbox;
    let clear_color = if lightbox { Vec4::new(0.0, 0.0, 0.0, 1.0) } else { Vec4::new(0.32, 0.42, 0.6, 1.0) };
    let mut renderer = page.renderer(RendererConfig { sample_count: 4, clear_color, ..Default::default() }).await;

    let phone = is_phone();
    let quality = param("quality").as_deref().and_then(VoxelGiQuality::from_name).unwrap_or(if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium });
    let rt = lightbox && flag("rt", false);
    // ray traced, fewer and larger, as in the article's bonus
    let default_count = (if rt { 8_192 } else if lightbox { 12_288 } else { 32_768 }) / (if phone { 2 } else { 1 });
    let count: usize = param_or("particles", default_count).clamp(1024, 262_144);
    let gi_on = flag("gi", true);
    let indirect = param("view").as_deref() == Some("indirect");
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
        view: indirect as u32 as f32,
        sun_cone_tan: gi.settings.cones.sun_cone_tan,
        cone_steps: gi.settings.cones.max_steps as f32,
        box_sky_scale: 1.0,
        ao_distance: 5.0,
        ao_strength: 0.85,
        _pad0: [0.0; 2],
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
        hdr: 0.0,
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
        scene_params.sun_cone_tan = gi.settings.cones.sun_cone_tan;
    }
    let layout = *gi.volume().layout();
    log::info!(
        "voxel GI {} ({}): {:?} voxels of {:.2} m, {:.1} MiB, {} particles",
        gi.quality().name(),
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
    let sky_uniform = || ComputeBuffer::from_external("SkyLighting", gi.sky_buffer().clone(), BufferType::Uniform);
    let linear_clamp = || Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge);

    let mut scene = Scene::new();
    let both = ShaderStages::VERTEX | ShaderStages::FRAGMENT;
    let fragment = ShaderStages::FRAGMENT;
    let mut panels = Vec::new();
    let mut slabs = Vec::new();
    // the walls; in the lightbox each again mirrored under the floor
    let mirrors: &[bool] = if lightbox { &[false, true] } else { &[false] };
    let wall_shader = if lightbox { lightbox_wall_shader() } else { cornell_wall_shader() };
    for wall in walls.iter().filter(|w| w.render) {
        for &mirrored in mirrors {
            let cull_mode = if mirrored { CullMode::None } else { CullMode::Back };
            let mut material = Material::new(
                wall.label,
                &wall_shader,
                vec![Binding::uniform(0, both), Binding::uniform(1, both), Binding::uniform(2, both), Binding::texture_3d(3, both), Binding::sampler(4, both), Binding::uniform(5, fragment)],
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
            material.set_bindable(5, sky_uniform());
            let size: [f32; 3] = std::array::from_fn(|i| wall.max[i] - wall.min[i]);
            let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material);
            slab.object.set_position((wall.min[0] + wall.max[0]) * 0.5, (wall.min[1] + wall.max[1]) * 0.5, (wall.min[2] + wall.max[2]) * 0.5);
            let index = scene.add(SceneNode::Renderable(slab));
            slabs.push((index, mirrored));
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
    let mut pile_top = None;
    if lightbox {
        let grid_buffer = sim.grid().unwrap().params_buffer().clone();
        let storage = |label: &str, buffer: &wgpu::Buffer| ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage);
        let top = PileTop::new(renderer.device(), &grid_buffer, sim.sorted_positions_buffer().unwrap(), sim.particle_count());
        // a depth prepass, then the shading on equal depth: the shading (and the ray tracing)
        // runs for the nearest sphere alone, not for every sphere of the pile behind it
        let depth = sphere_shader(ROOM_SPHERES_DEPTH_WGSL);
        let shade = sphere_shader(ROOM_SPHERES_SHADE_WGSL);
        let options = |equal: bool| {
            if equal {
                MaterialOptions { cull_mode: CullMode::None, depth_compare: wgpu::CompareFunction::Equal, depth_write: Some(false), ..Default::default() }
            } else {
                MaterialOptions { cull_mode: CullMode::None, ..Default::default() }
            }
        };
        for (label, shader, equal) in [("SpheresDepth", &depth, false), ("Spheres", &shade, true)] {
            for mirrored in [false, true] {
                let mut material = Material::new(
                    label,
                    shader,
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
                        Binding::storage(10, fragment, true),
                        Binding::uniform(11, fragment),
                    ],
                    options(equal),
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
                material.set_bindable(10, storage("PileTop", &top.buffer));
                material.set_bindable(11, sky_uniform());
                let instances = InstancedGeometry::new(PlaneGeometry::new(1.0, 1.0), count as u32, vec![sim.positions_as_compute_buffer(3).unwrap()]);
                particles.push((scene.add(SceneNode::Renderable(Renderable::new(instances, material))), mirrored));
            }
        }
        pile_top = Some(top);
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

    let aspect = page.aspect();
    let mut camera = Camera::new(40.0, 0.5, 600.0, aspect);
    camera.update_projection_matrix();
    // far enough that the room fits across a portrait screen too
    let fit = (1.6 / aspect).max(1.0);
    let mut controls = if lightbox {
        // straight on, a little low, with the reflection under the box in view
        let mut c = CameraControls::from_canvas(canvas, Vec3::new(0.0, 5.0, 0.0), 37.0 * fit).with_mouse_pan(canvas);
        c.set_elevation(0.04);
        c
    } else {
        let mut c = CameraControls::from_canvas(canvas, Vec3::new(0.0, 8.0, 0.0), 58.0 * fit).with_mouse_pan(canvas);
        c.set_elevation(0.45);
        c.set_azimuth(0.18);
        c
    };
    controls.update(&mut camera, 0.0);
    let mouse = MouseVectors::from_canvas(canvas);

    let stats = flag("stats", false).then(|| (now(), 0));
    if flag("profile", false) {
        renderer.set_profiling(true);
    }
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
        slabs,
        pile_top,
        panel_intensity: 1.0,
        initial,
        sim_step: FixedStep::new(1.0 / 60.0).with_max_steps(3),
        stats,
        frame_ms: 16.7,
        paused: false,
        dof: Dof {
            on: flag("dof", false),
            focus: param("focus").and_then(|v| v.parse::<f32>().ok()).filter(|&f| f > 0.0),
            f_stop: param_or("fstop", DOF_F_STOP).max(0.1),
        },
        volume: None,
        clear_color,
    };
    state.apply_panel();
    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));

    kansei_wasm::run(&page, move |frame| state.borrow_mut().frame(frame));
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
            r#"{{"scene":"{}","rt":{},"quality":"{}","dims":[{},{},{}],"voxel_m":{:.3},"mib":{:.2},"particles":{},"gi":{},"frame_ms":{:.2},"dof":{},"focus":{},"focus_m":{:.2},"fstop":{}}}"#,
            s.look.name(),
            s.scene_params.rt_on > 0.5,
            s.gi.quality().name(),
            layout.dims[0],
            layout.dims[1],
            layout.dims[2],
            layout.voxel_size,
            s.gi.volume().memory_bytes() as f64 / (1 << 20) as f64,
            s.sim.particle_count(),
            s.gi.settings.cones.use_volume,
            s.frame_ms,
            s.dof.on,
            s.dof.focus.map_or("null".into(), |f| f.to_string()),
            s.focus_distance(),
            s.dof.f_stop
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

/// The GPU time of each pass and the CPU sections since the last call, per frame (with
/// `profile=1`; see `Renderer::take_profile`).
#[wasm_bindgen]
pub fn profile_report() -> String {
    let mut out = String::new();
    with_state(|s| out = s.renderer.take_profile().report());
    out
}

/// Draw or hide the walls, the particles and the floor's reflection of both, for timing each.
#[wasm_bindgen]
pub fn set_layers(walls: bool, particles: bool, reflection: bool) {
    with_state(|s| {
        let layers = s.slabs.iter().map(|&(i, m)| (i, m, walls)).chain(s.particles.iter().map(|&(i, m)| (i, m, particles))).collect::<Vec<_>>();
        for (index, mirrored, on) in layers {
            if let Some(r) = s.scene.get_renderable_mut(index) {
                r.visible = on && (reflection || !mirrored);
            }
        }
    });
}

/// Put the particles at `positions` (x, y, z, w each), still, and sort the neighbour grid on them
/// without moving them: with `set_paused`, a fixed state to compare renders of.
#[wasm_bindgen]
pub fn set_positions(positions: &[f32]) {
    with_state(|s| {
        let mut positions = positions.to_vec();
        if s.look == Look::Lightbox {
            set_radius_shares(&mut positions);
        }
        s.sim.reset_particles(&positions);
        s.sim.update_batched(0.0, 0.0, [0.0; 2], [0.0; 2]);
        s.gi.reset_history();
    });
}

/// Voxels the particles' cones' start moves by each frame (0: not at all, so that with a temporal
/// blend of 1 every frame of a still fluid is alike).
#[wasm_bindgen]
pub fn set_cone_jitter(voxels: f32) {
    with_state(|s| s.gi.settings.cones.jitter_voxels = voxels.max(0.0));
}

/// Freeze the fluid (the GI keeps running), for comparing views of one moment.
#[wasm_bindgen]
pub fn set_paused(paused: bool) {
    with_state(|s| s.paused = paused);
}

/// Depth of field on or off (`dof=1`).
#[wasm_bindgen]
pub fn set_dof(on: bool) {
    with_state(|s| s.dof.on = on);
}

/// The depth of field's focus distance, metres; 0 or less focuses on the orbit target.
#[wasm_bindgen]
pub fn set_dof_focus(metres: f32) {
    with_state(|s| s.dof.focus = (metres > 0.0).then_some(metres));
}

/// The depth of field's aperture, an f-number.
#[wasm_bindgen]
pub fn set_dof_fstop(f_stop: f32) {
    with_state(|s| s.dof.f_stop = f_stop.max(0.1));
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
        let sky = std::mem::size_of::<kansei_core::gi::SkyLightingData>();
        for (name, code) in [("lightbox walls", lightbox_wall_shader()), ("cornell walls", cornell_wall_shader()), ("cornell particles", cornell_particle_shader())] {
            let module = validate(name, &code);
            assert_eq!(struct_size(&module, "SceneParams"), std::mem::size_of::<SceneParams>(), "{name}");
            if name != "cornell particles" {
                assert_eq!(struct_size(&module, "SkyLighting"), sky, "{name}");
            }
        }
        validate("lightbox sphere depth", &sphere_shader(ROOM_SPHERES_DEPTH_WGSL));
        let spheres = validate("lightbox spheres", &sphere_shader(ROOM_SPHERES_SHADE_WGSL));
        assert_eq!(struct_size(&spheres, "SceneParams"), std::mem::size_of::<SceneParams>());
        assert_eq!(struct_size(&spheres, "SkyLighting"), sky);
        assert_eq!(struct_size(&spheres, "NeighbourGrid"), std::mem::size_of::<kansei_core::simulations::grid::GpuNeighbourGrid>());
        assert_eq!(struct_size(&validate("pile top", &pile_top_shader()), "NeighbourGrid"), std::mem::size_of::<kansei_core::simulations::grid::GpuNeighbourGrid>());
        assert_eq!(struct_size(&spheres, "Particles"), 32);
        assert_eq!(struct_size(&validate("lightbox walls", &lightbox_wall_shader()), "Surface"), 48);
    }

    #[test]
    fn the_sun_keeps_the_panels_bearing() {
        // bearing 0 looks out of the open front (+z), 90 toward +x
        let close = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5);
        assert!(close(sun_direction(0.0, 0.0), [0.0, 0.0, 1.0]));
        assert!(close(sun_direction(0.0, 90.0), [1.0, 0.0, 0.0]));
        let (e, b) = (40f32.to_radians(), 70f32.to_radians());
        assert!(close(sun_direction(40.0, 70.0), [e.cos() * b.sin(), e.sin(), e.cos() * b.cos()]));
    }

    #[test]
    fn radius_shares_span_the_sizes_and_stay_under_the_bound() {
        let shares: Vec<f32> = (0..4096).map(radius_share).collect();
        // room_spheres.wgsl bounds the largest radius by 0.625 * size (half of 1.25) for the pile top
        assert!(shares.iter().all(|&w| (0.45..1.25).contains(&w)));
        assert!(shares.iter().any(|&w| w < 0.5) && shares.iter().any(|&w| w > 1.2));
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
