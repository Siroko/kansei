//! Glass particles: two emitters pour particles, each a small box turned to the way it moves,
//! through a scatter of glass spheres, boxes and pyramids in a mist under a spot light from above.
//!
//! - The particles live on the GPU (`src/shaders/particles_sim.wgsl`, a compute pass a frame): a
//!   dead one respawns at its emitter while the emitter's budget for the frame (its rate) lasts,
//!   a live one is carried by curl noise (the curl of a noise potential: a flow with vorticity and
//!   no sinks), pushed off the repellers along the gradients of their distance fields and kept out
//!   of them, and pushed by the pointer along its motion.
//! - The repellers are distance fields in that pass (a sphere, a box, a pyramid each) and drawn as
//!   the engine's ray-traced glass (`StandardLitOptions::glass`, `RtSurface::glass`,
//!   `RtReflectionsEffect::set_glass`): they refract the particles behind them, which go into the
//!   ray tracing grid each frame with their own albedo and glow (`RtSurface::record_emission`),
//!   the lamp's face, and the floor's grid (`RtReflectionsOptions::albedo_wgsl`) and reflection of
//!   the environment (`RtSurface::with_specular`). Or, cheaper, in screen space
//!   (`ScreenSpaceGlassEffect`, the glass's material transmissive).
//! - Only the live particles are drawn, in every pass: the sim compacts them first.
//! - A share of the particles glow; the bloom reads only the GBuffer's emissive target
//!   (`BloomOptions::use_emissive`), so the lit glass and the mist don't bloom.
//! - Volumetric fog lit by the spot light, raymarched through its shadow map, so the particles'
//!   and the glass's shadows streak the beam; SSAO; a polished floor with a technical grid that
//!   reflects it all (ray traced, `reflections` off to save the trace).
//! - One environment (`sky_up`, `sky_down`) lights the particles' and the floor's ambient, the
//!   fog's, and what the glass sees past the grid; lighting presets set it with the spot light,
//!   the exposure and the mist.
//! - Then TAA (the particles write motion vectors), the cinematic depth of field (focused on the
//!   orbit's target unless set), the bloom and the tone mapper.
//!
//! URL parameters: `n=<particles>` (default 32768, 4096..262144, split between the emitters),
//! `palette=<name>`, `lighting=spotlight|studio|overcast` (default studio), `ev=<EV100>` (over the
//! preset's), `fog=0`, `dof=0`, `scale=<0.25..1>` (render scale, TAA upscaling), `glassmode=1`
//! (the glass in screen space), `glass=0` (the repellers drawn as plain grey instead: no ray
//! tracing), `perf=0` (no performance overlay: the passes go untimed), `ui=0` (the page hides its
//! panel).

#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::{Camera, MOTION_VECTORS_WGSL};
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, CylinderGeometry, Geometry, InstancedGeometry, PlaneGeometry, SphereGeometry, Vertex};
use kansei_core::gi::{gradient_sky_lighting, GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, BindingResource, Compute, GradientSkyOptions, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100, BloomEffect, BloomOptions, CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, ContactAoEffect, ContactAoOptions, DistanceFogColor, DistanceFogEffect, SSAOAmbient, SSAOEffect, SSAOOptions, ScreenSpaceGlassEffect, ScreenSpaceGlassOptions, SpotScattering, TemporalAAEffect, TemporalAAOptions,
    ToneMapEffect, ToneMapOptions, ToneMapper, VolumetricFogEffect, VolumetricFogOptions,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::{pack_emission, RtGlass, RtGridOptions, RtPlacement, RtReflectionsEffect, RtReflectionsOptions, RtSurface, RtTraceResolution, SceneRtGridOptions, RT_EMISSION_WGSL};
use kansei_wasm::{flag, now, param, param_or, Canvas};

const SIM_WGSL: &str = include_str!("shaders/particles_sim.wgsl");
const DRAW_WGSL: &str = include_str!("shaders/particles_draw.wgsl");
const CLEAR_WGSL: &str = include_str!("shaders/particles_clear.wgsl");
const FLOOR_WGSL: &str = include_str!("shaders/floor.wgsl");
const FLOOR_GRID_WGSL: &str = include_str!("shaders/floor_grid.wgsl");

fn sim_shader() -> String {
    format!("{RT_EMISSION_WGSL}\n{SIM_WGSL}")
}

fn draw_shader() -> String {
    format!("{SPOT_LIGHTS_WGSL}\n{MOTION_VECTORS_WGSL}\n{RT_EMISSION_WGSL}\n{DRAW_WGSL}")
}

fn floor_shader() -> String {
    format!("{SPOT_LIGHTS_WGSL}\n{GBUFFER_OUT_WGSL}\n{FLOOR_GRID_WGSL}\n{FLOOR_WGSL}")
}

/// The ray traced hits' albedo (`RtReflectionsOptions::albedo_wgsl`): the floor's grid where a ray
/// meets the floor (renderable `floor`, `grid` baked in), over a footprint of about a pixel at the
/// hit's distance; every other triangle's own.
fn hit_albedo_wgsl(floor: usize, grid: &FloorGpu) -> String {
    let v = |c: [f32; 4]| format!("vec4f({:?}, {:?}, {:?}, {:?})", c[0], c[1], c[2], c[3]);
    format!(
        "{FLOOR_GRID_WGSL}\nconst KANSEI_FLOOR_GRID = FloorGrid({}, {}, {}, {}, {}, {}, {}, {});\n\
         fn kansei_rt_hit_albedo(triangle: u32, p: vec3f, bary: vec2f) -> vec3f {{\n\
             if (kansei_rt_source(triangle) == {}u) {{\n\
                 return floorGridAlbedo(KANSEI_FLOOR_GRID, p.xz, vec2f(0.0015 * distance(rp.invView[3].xyz, p)));\n\
             }}\n\
             return kansei_rt_albedo(triangle);\n\
         }}\n",
        v(grid.base), v(grid.minor), v(grid.major), v(grid.cross), v(grid.widths), v(grid.specular), v(grid.sky_up), v(grid.sky_down),
        floor & 0xfff
    )
}

/// The glass and the floor's reflections (`RtReflectionsEffect`), through the renderer's grid.
fn rt_effect(renderer: &Renderer, scene: &Scene, floor: usize, settings: &Settings, sky: &wgpu::Buffer) -> Option<RtReflectionsEffect> {
    let (gi, grid) = (renderer.voxel_gi()?, renderer.rt_grid()?);
    let options = RtReflectionsOptions { max_distance: 12.0, albedo_wgsl: Some(hit_albedo_wgsl(floor, &floor_grid(settings))), ..Default::default() };
    let mut rt = RtReflectionsEffect::with_volume(gi.volume(), grid.handle(), options);
    rt.set_spot_lights(Some(renderer.spot_lights_buffer()));
    // (the screen-space glass draws it itself)
    rt.set_glass((settings.glass_pass && settings.glass_mode == 0).then_some(RtGlass { interfaces: settings.glass_interfaces, ..Default::default() }));
    rt.set_sky_lighting(Some(sky));
    rt.reflections = settings.reflections;
    rt.update_lights(scene.lights());
    Some(rt)
}

/// Where an instance record (particles_sim.wgsl's `Instance`) puts the box's points, as the
/// draw's vertex stage does: sized, stretched along z, turned, placed.
const PLACE_WGSL: &str = r#"
fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {
    let placement = kansei_rt_record_vec4(record, 0u);
    let q = kansei_rt_record_vec4(record, 4u);
    let stretch = kansei_rt_record_f32(record, 11u);
    let local = p * vec3f(placement.w, placement.w, placement.w * stretch);
    return placement.xyz + local + 2.0 * cross(q.xyz, cross(q.xyz, local) + q.w * local);
}
"#;

/// Bytes of an instance record; its albedo and glow words (the grid reads them).
const INSTANCE_BYTES: u64 = 64;
const ALBEDO_WORD: u32 = 12;
const EMISSION_WORD: u32 = 13;
const MAX_COLLIDERS: usize = 16;
/// The particles live in this box (metres); the floor is its bottom.
const BOUNDS_MIN: [f32; 3] = [-4.0, 0.0, -4.0];
const BOUNDS_MAX: [f32; 3] = [4.0, 5.0, 4.0];
/// The spot light's height over the middle of the floor, and the radius of its lamp's face.
const LIGHT_HEIGHT: f32 = 5.5;
const LAMP_RADIUS: f32 = 0.2;
/// The nozzle drawn behind each emitter.
const NOZZLE_LENGTH: f32 = 0.18;
/// The glass's tint (the light left after a metre inside) and index of refraction.
const GLASS_TINT: [f32; 3] = [0.92, 0.96, 0.98];
/// The repellers with the glass's pass off: polished grey solids.
const SOLID_REPELLER: StandardLitOptions = StandardLitOptions {
    base_color: [0.55, 0.57, 0.6],
    roughness: 0.18,
    metallic: 0.0,
    specular: 1.0,
    emissive: [0.0; 3],
    sky_up: [0.0; 3],
    sky_down: [0.0; 3],
    instancing: None,
    outputs_velocity: false,
    traced: kansei_core::materials::StandardTraced::None,
};
/// The tone mapper's curves, by `Settings::tone_mapper`.
const TONE_MAPPERS: [ToneMapper; 5] = [ToneMapper::AcesFitted, ToneMapper::AgX, ToneMapper::AgXPunchy, ToneMapper::KhronosNeutral, ToneMapper::Exponential];
/// The floor's polish (GGX roughness) and reflectance (F0), as the reflections trace it and the
/// rays that meet it shade it.
const FLOOR_ROUGHNESS: f32 = 0.22;
const FLOOR_F0: f32 = 0.04;

// ── GPU layouts (particles_sim.wgsl, particles_draw.wgsl) ──

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct EmitterGpu {
    axis_x: [f32; 4],
    axis_y: [f32; 4],
    axis_z: [f32; 4],
    position: [f32; 4],
    launch: [f32; 4],
    jitter: [f32; 4],
    pool: [u32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct ColliderGpu {
    row0: [f32; 4],
    row1: [f32; 4],
    row2: [f32; 4],
    center: [f32; 4],
    size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct SimGpu {
    dt: f32,
    time: f32,
    curl_frequency: f32,
    curl_speed: f32,
    drag: f32,
    gravity: f32,
    repel_range: f32,
    repel_strength: f32,
    collider_count: u32,
    frame: u32,
    glow: f32,
    stretch: f32,
    curl_evolve: f32,
    bounce: f32,
    hit_speed: f32,
    _pad0: f32,
    bounds_min: [f32; 4],
    bounds_max: [f32; 4],
    mouse_origin: [f32; 4],
    mouse_ray: [f32; 4],
    mouse_motion: [f32; 4],
    base: [[f32; 4]; 4],
    emissive: [[f32; 4]; 2],
    emitters: [EmitterGpu; 2],
    colliders: [ColliderGpu; MAX_COLLIDERS],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FloorGpu {
    base: [f32; 4],
    minor: [f32; 4],
    major: [f32; 4],
    cross: [f32; 4],
    widths: [f32; 4],
    specular: [f32; 4],
    sky_up: [f32; 4],
    sky_down: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LookGpu {
    roughness: f32,
    metallic: f32,
    dt: f32,
    _pad: f32,
    sky_up: [f32; 4],
    sky_down: [f32; 4],
}

// ── What the panel sets ──

/// An emitter: where it is and points (its local +y), the shape it emits from, how fast and how
/// many.
#[derive(Clone, Copy, Debug)]
struct Emitter {
    position: [f32; 3],
    /// Euler angles (degrees), applied x then y then z, as `Object3D::rotation`.
    rotation: [f32; 3],
    /// 0 a point, 1 a ball, 2 a disc across its axis.
    shape: u32,
    radius: f32,
    /// Launch speed (m/s), and the cone it launches into (half angle, degrees).
    speed: f32,
    spread: f32,
    lifetime: f32,
    /// Particles a second (at most its pool over the lifetime are alive).
    rate: f32,
    size: f32,
    speed_jitter: f32,
    life_jitter: f32,
    /// The share that glows.
    glow_share: f32,
    enabled: bool,
}

impl Emitter {
    fn rotation_matrix(&self) -> glam::Mat3 {
        let [x, y, z] = self.rotation.map(f32::to_radians);
        glam::Mat3::from_euler(glam::EulerRot::ZYX, z, y, x)
    }

    fn json(&self) -> String {
        format!(
            "{{\"x\":{},\"y\":{},\"z\":{},\"rx\":{},\"ry\":{},\"rz\":{},\"shape\":{},\"radius\":{},\"speed\":{},\"spread\":{},\"lifetime\":{},\"rate\":{},\"size\":{},\"speedJitter\":{},\"lifeJitter\":{},\"glowShare\":{},\"enabled\":{}}}",
            self.position[0], self.position[1], self.position[2], self.rotation[0], self.rotation[1], self.rotation[2], self.shape, self.radius, self.speed, self.spread,
            self.lifetime, self.rate, self.size, self.speed_jitter, self.life_jitter, self.glow_share, self.enabled
        )
    }

    fn set(&mut self, key: &str, v: f32) -> bool {
        match key {
            "x" => self.position[0] = v,
            "y" => self.position[1] = v,
            "z" => self.position[2] = v,
            "rx" => self.rotation[0] = v,
            "ry" => self.rotation[1] = v,
            "rz" => self.rotation[2] = v,
            "shape" => self.shape = (v.max(0.0) as u32).min(2),
            "radius" => self.radius = v.max(0.0),
            "speed" => self.speed = v.max(0.0),
            "spread" => self.spread = v.clamp(0.0, 180.0),
            "lifetime" => self.lifetime = v.max(0.1),
            "rate" => self.rate = v.max(0.0),
            "size" => self.size = v.max(0.001),
            "speedJitter" => self.speed_jitter = v.clamp(0.0, 1.0),
            "lifeJitter" => self.life_jitter = v.clamp(0.0, 1.0),
            "glowShare" => self.glow_share = v.clamp(0.0, 1.0),
            "enabled" => self.enabled = v != 0.0,
            _ => return false,
        }
        true
    }
}

const EMITTERS: [Emitter; 2] = [
    // low on the left, aimed up and right across the middle
    Emitter {
        position: [-2.2, 0.7, 0.4],
        rotation: [0.0, 0.0, -58.0],
        shape: 1,
        radius: 0.06,
        speed: 3.2,
        spread: 9.0,
        lifetime: 7.0,
        rate: 2200.0,
        size: 0.022,
        speed_jitter: 0.25,
        life_jitter: 0.35,
        glow_share: 0.06,
        enabled: true,
    },
    // high on the right, aimed down and left
    Emitter {
        position: [2.2, 3.3, -0.5],
        rotation: [0.0, 0.0, 118.0],
        shape: 2,
        radius: 0.12,
        speed: 2.6,
        spread: 14.0,
        lifetime: 7.0,
        rate: 2200.0,
        size: 0.022,
        speed_jitter: 0.25,
        life_jitter: 0.35,
        glow_share: 0.06,
        enabled: true,
    },
];

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    Sphere,
    Box,
    Pyramid,
}

/// A repeller: a sphere (`size[0]` its radius), a box (`size` its sides) or a pyramid (`size[0]`
/// its base's side, `size[1]` its height; `center` the middle of its base), turned by `rotation`
/// (degrees, as `Object3D::rotation`).
#[derive(Clone, Copy, Debug)]
struct Collider {
    kind: Kind,
    center: [f32; 3],
    rotation: [f32; 3],
    size: [f32; 3],
}

impl Collider {
    fn rotation_matrix(&self) -> glam::Mat3 {
        let [x, y, z] = self.rotation.map(f32::to_radians);
        glam::Mat3::from_euler(glam::EulerRot::ZYX, z, y, x)
    }

    fn gpu(&self) -> ColliderGpu {
        // world to local is the rotation's transpose: its rows are the rotation's columns
        let r = self.rotation_matrix();
        let kind = match self.kind {
            Kind::Sphere => 0.0,
            Kind::Box => 1.0,
            Kind::Pyramid => 2.0,
        };
        let size = match self.kind {
            Kind::Sphere => [self.size[0], 0.0, 0.0, 0.0],
            Kind::Box => [self.size[0] * 0.5, self.size[1] * 0.5, self.size[2] * 0.5, 0.0],
            Kind::Pyramid => [self.size[0], self.size[1], 0.0, 0.0],
        };
        ColliderGpu {
            row0: [r.x_axis.x, r.x_axis.y, r.x_axis.z, kind],
            row1: [r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0],
            row2: [r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0],
            center: [self.center[0], self.center[1], self.center[2], 0.0],
            size,
        }
    }

    fn geometry(&self) -> Geometry {
        match self.kind {
            Kind::Sphere => SphereGeometry::new(self.size[0], 64, 48),
            Kind::Box => BoxGeometry::new(self.size[0], self.size[1], self.size[2]),
            Kind::Pyramid => pyramid(self.size[0], self.size[1]),
        }
    }
}

/// The repellers, spread through the middle of the room where the two streams cross.
fn colliders() -> Vec<Collider> {
    use Kind::*;
    let c = |kind, center, rotation, size| Collider { kind, center, rotation, size };
    vec![
        c(Sphere, [0.0, 1.55, 0.0], [0.0; 3], [0.5, 0.0, 0.0]),
        c(Box, [-1.2, 1.0, 0.7], [24.0, 38.0, 12.0], [0.7, 0.7, 0.7]),
        // (those on the floor 2 mm over it: a base in its plane meets the rays as it does, by turns)
        c(Pyramid, [1.1, 0.002, 0.3], [0.0, 20.0, 0.0], [1.0, 1.1, 0.0]),
        c(Sphere, [1.25, 2.35, 0.9], [0.0; 3], [0.3, 0.0, 0.0]),
        c(Box, [0.4, 0.55, -1.2], [0.0, -30.0, 0.0], [1.3, 0.25, 0.55]),
        c(Pyramid, [-0.9, 2.9, -0.8], [180.0, 15.0, 10.0], [0.75, 0.8, 0.0]),
        c(Sphere, [-1.8, 2.0, -0.3], [0.0; 3], [0.24, 0.0, 0.0]),
        c(Box, [0.3, 2.7, -1.4], [40.0, 10.0, 45.0], [0.45, 0.45, 0.45]),
        c(Pyramid, [-0.3, 0.002, 1.6], [0.0, 45.0, 0.0], [0.6, 0.55, 0.0]),
        c(Sphere, [1.9, 1.2, -1.1], [0.0; 3], [0.35, 0.0, 0.0]),
    ]
}

/// A square pyramid with flat faces: base side `side` on y = 0 round the origin, apex at `height`.
fn pyramid(side: f32, height: f32) -> Geometry {
    let h = side * 0.5;
    let apex = glam::Vec3::new(0.0, height, 0.0);
    let base = [glam::Vec3::new(-h, 0.0, -h), glam::Vec3::new(h, 0.0, -h), glam::Vec3::new(h, 0.0, h), glam::Vec3::new(-h, 0.0, h)];
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let mut face = |points: &[glam::Vec3], normal: glam::Vec3| {
        let first = vertices.len() as u32;
        for (k, p) in points.iter().enumerate() {
            vertices.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: normal.to_array(), uv: [k as f32 * 0.5, (k % 2) as f32] });
        }
        for k in 1..points.len() as u32 - 1 {
            indices.extend_from_slice(&[first, first + k, first + k + 1]);
        }
    };
    // the sides, counter-clockwise seen from outside
    for k in 0..4 {
        let (a, b) = (base[k], base[(k + 1) % 4]);
        let normal = (apex - a).cross(b - a).normalize();
        let outward = if normal.dot((a + b) * 0.5) < 0.0 { -normal } else { normal };
        let (p, q) = if (b - a).cross(apex - a).dot(outward) > 0.0 { (a, b) } else { (b, a) };
        face(&[p, q, apex], outward);
    }
    // the base, facing down
    face(&[base[0], base[1], base[2], base[3]], glam::Vec3::NEG_Y);
    Geometry::new("Pyramid", vertices, indices)
}

/// A palette: the particles' albedos, the glowing particles' colours, the light's colour.
struct Palette {
    name: &'static str,
    base: [[f32; 3]; 4],
    glow: [[f32; 3]; 2],
    light: [f32; 3],
}

const PALETTES: [Palette; 6] = [
    Palette { name: "ember", base: [[0.05, 0.045, 0.04], [0.35, 0.3, 0.26], [0.6, 0.3, 0.12], [0.8, 0.72, 0.6]], glow: [[1.0, 0.42, 0.08], [1.0, 0.72, 0.28]], light: [1.0, 0.86, 0.72] },
    Palette { name: "glacier", base: [[0.85, 0.9, 0.95], [0.45, 0.6, 0.75], [0.12, 0.2, 0.32], [0.95, 0.97, 1.0]], glow: [[0.25, 0.7, 1.0], [0.7, 0.95, 1.0]], light: [0.8, 0.9, 1.0] },
    Palette { name: "neon", base: [[0.06, 0.04, 0.1], [0.35, 0.1, 0.55], [0.08, 0.15, 0.5], [0.9, 0.85, 0.95]], glow: [[1.0, 0.12, 0.55], [0.1, 0.85, 1.0]], light: [0.85, 0.8, 1.0] },
    Palette { name: "bauhaus", base: [[0.75, 0.08, 0.06], [0.95, 0.75, 0.08], [0.05, 0.15, 0.6], [0.9, 0.88, 0.82]], glow: [[1.0, 0.82, 0.2], [1.0, 0.3, 0.08]], light: [1.0, 0.97, 0.92] },
    Palette { name: "moss", base: [[0.08, 0.2, 0.1], [0.3, 0.45, 0.2], [0.6, 0.65, 0.4], [0.12, 0.12, 0.1]], glow: [[0.5, 1.0, 0.3], [0.9, 1.0, 0.5]], light: [0.95, 1.0, 0.88] },
    Palette { name: "mono", base: [[0.04, 0.04, 0.04], [0.25, 0.25, 0.25], [0.6, 0.6, 0.6], [0.9, 0.9, 0.9]], glow: [[1.0, 1.0, 1.0], [0.85, 0.92, 1.0]], light: [1.0, 1.0, 1.0] },
];

fn palette_index(name: &str) -> Option<usize> {
    PALETTES.iter().position(|p| p.name == name)
}

/// The flow, the repellers, the pointer and the look.
#[derive(Clone, Copy, Debug)]
struct Settings {
    curl_frequency: f32,
    curl_speed: f32,
    curl_evolve: f32,
    drag: f32,
    gravity: f32,
    repel_range: f32,
    repel_strength: f32,
    bounce: f32,
    stretch: f32,
    /// The glowing particles' radiance (cd/m²).
    glow: f32,
    palette: usize,
    /// How fast the particles near the pointer take up its motion (1/s), and how far round it.
    mouse_force: f32,
    mouse_radius: f32,
    roughness: f32,
    metallic: f32,
    /// The spot light (cd) and its cone (outer half angle, degrees).
    light: f32,
    cone: f32,
    ev100: f32,
    fog_density: f32,
    fog_anisotropy: f32,
    /// Depth of field (off: out of the chain).
    dof: bool,
    /// The scene's passes (the glass, the mist, the GBuffer) at this share of the canvas's size,
    /// the TAA upscaling to it (`Renderer::set_render_scale`).
    render_scale: f32,
    /// Focus on the orbit's target, else at `focus` (m).
    autofocus: bool,
    focus: f32,
    f_stop: f32,
    bloom: f32,
    bloom_threshold: f32,
    glass_ior: f32,
    glass_roughness: f32,
    /// Whether the glass casts shadows (real glass's are faint, and bright where it focuses).
    glass_shadows: bool,
    /// How the glass is drawn: 0 ray traced (`RtReflectionsEffect`'s glass, every surface it
    /// crosses), 1 in screen space (`ScreenSpaceGlassEffect`, its material transmissive: a
    /// fraction of the cost, one surface deep).
    glass_mode: u32,
    /// The environment's radiance from straight up and straight down (cd/m²): the hemisphere of
    /// ambient light on the particles and the floor, the fog's ambient, and what the glass shows
    /// past the ray tracing grid's triangles (the fog, composited at the glass's surface, can't
    /// put itself behind it).
    sky_up: f32,
    sky_down: f32,
    /// The floor's ray-traced reflections (the glass is traced either way).
    reflections: bool,
    /// The post-processing passes, each on or off (`PostProcessingVolume::set_enabled`) with its
    /// own settings. SSAO: its strength, radius (m), depth bias (m a metre of distance) and
    /// samples.
    ssao: bool,
    ao: f32,
    ao_radius: f32,
    ao_bias: f32,
    ao_samples: u32,
    /// The contact darkening after the glass (`ContactAoEffect`): its reach (m), strength (the
    /// exponent on the open share), falloff, samples, and the occlusion alone in place of the
    /// image.
    contact_ao: bool,
    cao_radius: f32,
    cao_strength: f32,
    cao_falloff: f32,
    cao_samples: u32,
    cao_view: bool,
    /// The glass's pass (off: the repellers drawn as polished grey solids); the screen-space
    /// glass's ball (m), its depth behind (m) and dispersion; the ray traced glass's surfaces.
    glass_pass: bool,
    ss_thickness: f32,
    ss_behind: f32,
    ss_dispersion: f32,
    glass_interfaces: u32,
    /// The floor's reflections: their scale, reach (m), a pixel of each 4 x 4 (else 2 x 2) a
    /// frame, and the weight of each new frame.
    refl_intensity: f32,
    refl_distance: f32,
    refl_quarter: bool,
    refl_blend: f32,
    /// The mist: on, its falloff with height (1/m), its ambient (a share of the environment's)
    /// and the beam's samples a pixel (0: in the froxels).
    fog: bool,
    fog_falloff: f32,
    fog_ambient: f32,
    fog_steps: u32,
    /// TAA: the history's weight still and moving, and its clip box (standard deviations).
    taa: bool,
    taa_feedback_max: f32,
    taa_feedback_min: f32,
    taa_gamma: f32,
    /// Depth of field: its samples a pixel, the aperture's blades (0 round) and the largest
    /// blur (a share of the image's width).
    dof_samples: u32,
    dof_blades: u32,
    dof_max_coc: f32,
    /// Bloom: on, its knee, its spread and whether it reads the emissive target only.
    bloom_on: bool,
    bloom_knee: f32,
    bloom_radius: f32,
    bloom_emissive: bool,
    /// The tone mapper: on (off: the light as it is, clamped), its curve (0 ACES fitted, 1 AgX,
    /// 2 AgX punchy, 3 Khronos neutral, 4 1 - exp(-x)), the exposure's compensation (stops),
    /// vignetting, grain and fringing.
    tone: bool,
    tone_mapper: u32,
    exposure_comp: f32,
    vignette: f32,
    grain: f32,
    chromatic: f32,
    /// The floor's grid: its colours (linear albedo), the minor lines' spacing (m), minor cells
    /// per major cell, the lines' and the crosses' widths (m), the crosses' arms (share of a
    /// major cell).
    grid_base: [f32; 3],
    grid_minor: [f32; 3],
    grid_major: [f32; 3],
    grid_cross: [f32; 3],
    grid_spacing: f32,
    grid_major_every: f32,
    grid_minor_width: f32,
    grid_major_width: f32,
    grid_cross_width: f32,
    grid_cross_arm: f32,
    paused: bool,
    /// The fade of the far floor into the backdrop (`DistanceFogEffect`), from and to (m from the
    /// scene's centre, so the set stays clear wherever the camera is).
    fade: bool,
    fade_start: f32,
    fade_end: f32,
}

impl Default for Settings {
    fn default() -> Self {
        let mut s = Self::unlit();
        s.lighting(&LIGHTINGS[1]);
        s
    }
}

impl Settings {
    /// The settings before a lighting preset.
    fn unlit() -> Self {
        Self {
            curl_frequency: 0.55,
            curl_speed: 0.9,
            curl_evolve: 0.25,
            drag: 0.9,
            gravity: -0.15,
            repel_range: 0.35,
            repel_strength: 9.0,
            bounce: 0.35,
            stretch: 10.0,
            glow: 40.0,
            palette: 0,
            mouse_force: 6.0,
            mouse_radius: 0.6,
            roughness: 0.35,
            metallic: 0.0,
            light: 3000.0,
            cone: 32.0,
            ev100: 2.5,
            fog_density: 0.12,
            fog_anisotropy: 0.45,
            dof: true,
            render_scale: 1.0,
            autofocus: true,
            focus: 6.0,
            f_stop: 1.8,
            bloom: 0.25,
            bloom_threshold: 1.0,
            glass_ior: 1.5,
            glass_roughness: 0.0,
            glass_shadows: true,
            glass_mode: 0,
            sky_up: 0.15,
            sky_down: 0.05,
            reflections: true,
            ssao: true,
            ao: 0.6,
            ao_radius: 0.25,
            ao_bias: 0.003,
            ao_samples: 16,
            contact_ao: false,
            cao_radius: 0.3,
            cao_strength: 2.5,
            cao_falloff: 1.0,
            cao_samples: 12,
            cao_view: false,
            glass_pass: true,
            ss_thickness: 0.8,
            ss_behind: 2.0,
            ss_dispersion: 0.02,
            glass_interfaces: 8,
            refl_intensity: 1.0,
            refl_distance: 12.0,
            refl_quarter: false,
            refl_blend: 0.25,
            fog: true,
            fog_falloff: 0.08,
            fog_ambient: 1.0,
            fog_steps: 32,
            taa: true,
            taa_feedback_max: 0.95,
            taa_feedback_min: 0.85,
            taa_gamma: 1.0,
            dof_samples: 48,
            dof_blades: 7,
            dof_max_coc: 0.025,
            bloom_on: true,
            bloom_knee: 0.5,
            bloom_radius: 1.0,
            bloom_emissive: true,
            tone: true,
            tone_mapper: 0,
            exposure_comp: 0.0,
            vignette: 0.35,
            grain: 0.0,
            chromatic: 0.0,
            grid_base: [0.035, 0.036, 0.038],
            grid_minor: [0.08, 0.082, 0.085],
            grid_major: [0.17, 0.174, 0.18],
            grid_cross: [0.34, 0.345, 0.35],
            grid_spacing: 0.1,
            grid_major_every: 10.0,
            grid_minor_width: 0.0035,
            grid_major_width: 0.011,
            grid_cross_width: 0.016,
            grid_cross_arm: 0.07,
            paused: false,
            fade: true,
            fade_start: 7.0,
            fade_end: 30.0,
        }
    }
}

macro_rules! settings_json {
    ($s:expr, $($key:literal => $field:ident),* $(,)?) => {{
        let mut out = String::from("{");
        $( out.push_str(&format!("\"{}\":{},", $key, $s.$field)); )*
        out.push_str(&format!("\"palette\":\"{}\"}}", PALETTES[$s.palette].name));
        out
    }};
}

/// A lighting preset: the spot light (cd) and its cone (degrees), the environment (cd/m² up and
/// down), the exposure (EV100) and the mist (1/m).
struct Lighting {
    name: &'static str,
    light: f32,
    cone: f32,
    sky: [f32; 2],
    ev100: f32,
    fog: f32,
}

const LIGHTINGS: [Lighting; 3] = [
    // the spot light alone in a dark room, its beam in the mist
    Lighting { name: "spotlight", light: 3000.0, cone: 32.0, sky: [0.15, 0.05], ev100: 2.5, fog: 0.12 },
    // a brighter, wider key over a soft studio fill: everything reads, the beam still shows
    Lighting { name: "studio", light: 6000.0, cone: 42.0, sky: [6.0, 2.5], ev100: 3.4, fog: 0.025 },
    // daylight from a white sky, the spot a faint accent
    Lighting { name: "overcast", light: 2000.0, cone: 55.0, sky: [40.0, 14.0], ev100: 4.6, fog: 0.02 },
];

impl Settings {
    fn lighting(&mut self, preset: &Lighting) {
        self.light = preset.light;
        self.cone = preset.cone;
        [self.sky_up, self.sky_down] = preset.sky;
        self.ev100 = preset.ev100;
        self.fog_density = preset.fog;
    }

    fn json(&self) -> String {
        settings_json!(self,
            "curlFrequency" => curl_frequency, "curlSpeed" => curl_speed, "curlEvolve" => curl_evolve, "drag" => drag, "gravity" => gravity,
            "repelRange" => repel_range, "repelStrength" => repel_strength, "bounce" => bounce, "stretch" => stretch, "glow" => glow,
            "mouseForce" => mouse_force, "mouseRadius" => mouse_radius, "roughness" => roughness, "metallic" => metallic, "light" => light, "cone" => cone,
            "ev100" => ev100, "dof" => dof, "renderScale" => render_scale, "fogDensity" => fog_density, "fogAnisotropy" => fog_anisotropy, "autofocus" => autofocus, "focus" => focus, "fStop" => f_stop,
            "bloom" => bloom, "bloomThreshold" => bloom_threshold, "glassIor" => glass_ior, "glassRoughness" => glass_roughness, "glassShadows" => glass_shadows, "glassMode" => glass_mode,
            "skyUp" => sky_up, "skyDown" => sky_down, "reflections" => reflections, "ao" => ao, "gridSpacing" => grid_spacing, "gridMajorEvery" => grid_major_every, "gridMinorWidth" => grid_minor_width, "gridMajorWidth" => grid_major_width,
            "gridCrossWidth" => grid_cross_width, "gridCrossArm" => grid_cross_arm, "paused" => paused,
            "fade" => fade, "fadeStart" => fade_start, "fadeEnd" => fade_end,
            "ssao" => ssao, "aoRadius" => ao_radius, "aoBias" => ao_bias, "aoSamples" => ao_samples, "contactAo" => contact_ao, "caoRadius" => cao_radius,
            "caoStrength" => cao_strength, "caoFalloff" => cao_falloff, "caoSamples" => cao_samples, "caoView" => cao_view, "glassPass" => glass_pass, "ssThickness" => ss_thickness,
            "ssBehind" => ss_behind, "ssDispersion" => ss_dispersion, "glassInterfaces" => glass_interfaces, "reflIntensity" => refl_intensity,
            "reflDistance" => refl_distance, "reflQuarter" => refl_quarter, "reflBlend" => refl_blend, "fog" => fog, "fogFalloff" => fog_falloff,
            "fogAmbient" => fog_ambient, "fogSteps" => fog_steps, "taa" => taa, "taaFeedbackMax" => taa_feedback_max, "taaFeedbackMin" => taa_feedback_min,
            "taaGamma" => taa_gamma, "dofSamples" => dof_samples, "dofBlades" => dof_blades, "dofMaxCoc" => dof_max_coc, "bloomOn" => bloom_on,
            "bloomKnee" => bloom_knee, "bloomRadius" => bloom_radius, "bloomEmissive" => bloom_emissive, "tone" => tone, "toneMapper" => tone_mapper,
            "exposureComp" => exposure_comp, "vignette" => vignette, "grain" => grain, "chromatic" => chromatic,
        )
        .replacen('{', &format!("{{\"gridBase\":{:?},\"gridMinor\":{:?},\"gridMajor\":{:?},\"gridCross\":{:?},", self.grid_base, self.grid_minor, self.grid_major, self.grid_cross), 1)
    }

    fn set(&mut self, key: &str, v: f32) -> bool {
        match key {
            "curlFrequency" => self.curl_frequency = v.max(0.01),
            "curlSpeed" => self.curl_speed = v,
            "curlEvolve" => self.curl_evolve = v,
            "drag" => self.drag = v.max(0.0),
            "gravity" => self.gravity = v,
            "repelRange" => self.repel_range = v.max(0.01),
            "repelStrength" => self.repel_strength = v.max(0.0),
            "bounce" => self.bounce = v.clamp(0.0, 1.0),
            "stretch" => self.stretch = v.max(0.0),
            "glow" => self.glow = v.max(0.0),
            "mouseForce" => self.mouse_force = v.max(0.0),
            "mouseRadius" => self.mouse_radius = v.max(0.01),
            "roughness" => self.roughness = v.clamp(0.02, 1.0),
            "metallic" => self.metallic = v.clamp(0.0, 1.0),
            "light" => self.light = v.max(0.0),
            "cone" => self.cone = v.clamp(5.0, 80.0),
            "ev100" => self.ev100 = v,
            "fogDensity" => self.fog_density = v.max(0.0),
            "fogAnisotropy" => self.fog_anisotropy = v.clamp(-0.95, 0.95),
            "dof" => self.dof = v != 0.0,
            "renderScale" => self.render_scale = v.clamp(0.25, 1.0),
            "autofocus" => self.autofocus = v != 0.0,
            "focus" => self.focus = v.max(0.1),
            "fStop" => self.f_stop = v.max(0.5),
            "bloom" => self.bloom = v.max(0.0),
            "bloomThreshold" => self.bloom_threshold = v.max(0.01),
            "glassIor" => self.glass_ior = v.max(1.0),
            "glassRoughness" => self.glass_roughness = v.clamp(0.0, 1.0),
            "glassShadows" => self.glass_shadows = v != 0.0,
            "glassMode" => self.glass_mode = (v.max(0.0) as u32).min(1),
            "skyUp" => self.sky_up = v.max(0.0),
            "skyDown" => self.sky_down = v.max(0.0),
            "reflections" => self.reflections = v != 0.0,
            "ao" => self.ao = v.clamp(0.0, 1.0),
            "gridSpacing" => self.grid_spacing = v.max(0.005),
            "gridMajorEvery" => self.grid_major_every = v.round().max(1.0),
            "gridMinorWidth" => self.grid_minor_width = v.max(0.0),
            "gridMajorWidth" => self.grid_major_width = v.max(0.0),
            "gridCrossWidth" => self.grid_cross_width = v.max(0.0),
            "gridCrossArm" => self.grid_cross_arm = v.clamp(0.0, 0.5),
            "paused" => self.paused = v != 0.0,
            "fade" => self.fade = v != 0.0,
            "fadeStart" => self.fade_start = v.max(0.0),
            "fadeEnd" => self.fade_end = v.max(0.0),
            "ssao" => self.ssao = v != 0.0,
            "aoRadius" => self.ao_radius = v.max(0.01),
            "aoBias" => self.ao_bias = v.max(0.0),
            "aoSamples" => self.ao_samples = (v.round() as u32).clamp(1, 32),
            "contactAo" => self.contact_ao = v != 0.0,
            "caoRadius" => self.cao_radius = v.max(0.005),
            "caoStrength" => self.cao_strength = v.max(0.0),
            "caoFalloff" => self.cao_falloff = v.max(0.0),
            "caoSamples" => self.cao_samples = (v.round() as u32).clamp(1, 32),
            "caoView" => self.cao_view = v != 0.0,
            "glassPass" => self.glass_pass = v != 0.0,
            "ssThickness" => self.ss_thickness = v.max(0.01),
            "ssBehind" => self.ss_behind = v.max(0.0),
            "ssDispersion" => self.ss_dispersion = v.max(0.0),
            "glassInterfaces" => self.glass_interfaces = (v.round() as u32).clamp(1, 16),
            "reflIntensity" => self.refl_intensity = v.max(0.0),
            "reflDistance" => self.refl_distance = v.max(0.5),
            "reflQuarter" => self.refl_quarter = v != 0.0,
            "reflBlend" => self.refl_blend = v.clamp(0.02, 1.0),
            "fog" => self.fog = v != 0.0,
            "fogFalloff" => self.fog_falloff = v.max(0.0),
            "fogAmbient" => self.fog_ambient = v.max(0.0),
            "fogSteps" => self.fog_steps = (v.round() as u32).min(128),
            "taa" => self.taa = v != 0.0,
            "taaFeedbackMax" => self.taa_feedback_max = v.clamp(0.0, 0.99),
            "taaFeedbackMin" => self.taa_feedback_min = v.clamp(0.0, 0.99),
            "taaGamma" => self.taa_gamma = v.clamp(0.25, 4.0),
            "dofSamples" => self.dof_samples = (v.round() as u32).clamp(8, 128),
            "dofBlades" => self.dof_blades = (v.round() as u32).min(12),
            "dofMaxCoc" => self.dof_max_coc = v.clamp(0.002, 0.08),
            "bloomOn" => self.bloom_on = v != 0.0,
            "bloomKnee" => self.bloom_knee = v.max(0.01),
            "bloomRadius" => self.bloom_radius = v.max(0.1),
            "bloomEmissive" => self.bloom_emissive = v != 0.0,
            "tone" => self.tone = v != 0.0,
            "toneMapper" => self.tone_mapper = (v.max(0.0) as u32).min(4),
            "exposureComp" => self.exposure_comp = v.clamp(-6.0, 6.0),
            "vignette" => self.vignette = v.clamp(0.0, 2.0),
            "grain" => self.grain = v.clamp(0.0, 1.0),
            "chromatic" => self.chromatic = v.clamp(0.0, 2.0),
            _ => return false,
        }
        true
    }
}

// ── The particles on the GPU ──

/// A staging buffer the events are copied into, and whether its map came back.
struct Readback {
    buffer: wgpu::Buffer,
    /// 0 free, 1 copied (map pending), 2 mapped
    state: Arc<AtomicU8>,
    /// The frames whose events it holds, and how many respawns the budgets had allowed in all
    /// when it was copied.
    frames: u32,
    spawn_total: u64,
}

/// The events per second over the last window (about half a second), for the panel.
#[derive(Clone, Copy, Default)]
struct Rates {
    /// The window so far: its start (s), its events (as `Particles::totals`) and frames.
    start: f64,
    counts: [u64; 8],
    frames: u64,
    /// The last full window's: respawns, brushes with the glass and hits on the floor a
    /// second, and the particles alive.
    emitted: f64,
    brushes: f64,
    floor_hits: f64,
    alive: f64,
}

struct Particles {
    count: u32,
    /// Particles of emitter 0; the rest are emitter 1's.
    split: u32,
    sim: wgpu::Buffer,
    spawned: wgpu::Buffer,
    events: wgpu::Buffer,
    instances: wgpu::Buffer,
    compute: Compute,
    /// Clears the instances past the live ones (particles_clear.wgsl).
    clear: Compute,
    readbacks: Vec<Readback>,
    /// Respawns the budgets have allowed in all, and the live particles a readback last saw with
    /// that total then: the live ones now are at most those plus the respawns since.
    spawn_total: u64,
    live_seen: (u32, u64),
    /// Whether the events buffer was copied last frame (cleared before this one's pass then).
    copied: bool,
    /// The events read back since the page last took them.
    totals: [u64; 8],
    /// Frames stepped since the events were last copied.
    pending_frames: u32,
    rates: Rates,
}

impl Particles {
    fn new(device: &wgpu::Device, count: u32) -> Self {
        let storage = |label, size: u64, extra| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra, mapped_at_creation: false });
        // every particle dead (its age at its lifetime) until its emitter spawns it
        let state: Vec<[f32; 8]> = (0..count).map(|_| [0.0, -1000.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]).collect();
        let particles = wgpu::util::DeviceExt::create_buffer_init(device, &wgpu::util::BufferInitDescriptor {
            label: Some("GlassParticles/State"),
            contents: bytemuck::cast_slice(&state),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let instances = storage("GlassParticles/Instances", count as u64 * INSTANCE_BYTES, wgpu::BufferUsages::VERTEX);
        let spawned = storage("GlassParticles/Spawned", 16, wgpu::BufferUsages::COPY_SRC);
        let events = storage("GlassParticles/Events", 32, wgpu::BufferUsages::COPY_SRC);
        let sim = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlassParticles/Sim"),
            size: std::mem::size_of::<SimGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stage = ShaderStages::COMPUTE;
        let mut compute = Compute::new(
            "GlassParticles/Sim",
            &sim_shader(),
            vec![Binding::uniform(0, stage), Binding::storage(1, stage, false), Binding::storage(2, stage, false), Binding::storage(3, stage, false), Binding::storage(4, stage, false)],
        );
        compute.initialize(device);
        let whole = |buffer| BindingResource::Buffer { buffer, offset: 0, size: None };
        compute.set_bind_group(device, &[(0, whole(&sim)), (1, whole(&particles)), (2, whole(&instances)), (3, whole(&spawned)), (4, whole(&events))]);
        let mut clear = Compute::new("GlassParticles/Clear", CLEAR_WGSL, vec![Binding::storage(0, stage, true), Binding::storage(1, stage, false)]);
        clear.initialize(device);
        clear.set_bind_group(device, &[(0, whole(&spawned)), (1, whole(&instances))]);
        let readbacks = (0..3)
            .map(|k| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("GlassParticles/EventsReadback{k}")),
                    size: 48,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(0)),
                frames: 0,
                spawn_total: 0,
            })
            .collect();
        let rates = Rates { start: now(), ..Default::default() };
        Self { count, split: count / 2, sim, spawned, events, instances, compute, clear, readbacks, spawn_total: 0, live_seen: (0, 0), copied: false, totals: [0; 8], pending_frames: 0, rates }
    }

    /// The instances to draw: the live particles, or a few more (a bound from a readback a few
    /// frames old and the respawns since; the rest are cleared to nothing).
    fn drawn(&self) -> u32 {
        let (live, then) = self.live_seen;
        (live as u64 + (self.spawn_total - then)).min(self.count as u64) as u32
    }

    /// The instance buffer as the boxes' vertex source (locations 3-6).
    fn instance_buffer(&self) -> ComputeBuffer {
        Self::records(&self.instances, "GlassParticles/Instances")
    }

    /// `buffer`'s instance records (particles_sim.wgsl's `Instance`) as a vertex source.
    fn records(buffer: &wgpu::Buffer, label: &str) -> ComputeBuffer {
        let attribute = |shader_location, offset, format| InstanceAttribute { shader_location, offset, format };
        ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage).with_vertex_layout(
            INSTANCE_BYTES,
            vec![
                attribute(3, 0, VertexFormat::Float32x4),
                attribute(4, 16, VertexFormat::Float32x4),
                attribute(5, 32, VertexFormat::Float32x4),
                attribute(6, 48, VertexFormat::Uint32x4),
            ],
        )
    }

    /// Run the frame's step (its parameters in `sim`), and copy its events into a free readback.
    fn step(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sim: &SimGpu) {
        queue.write_buffer(&self.sim, 0, bytemuck::bytes_of(sim));
        queue.write_buffer(&self.spawned, 0, bytemuck::cast_slice(&[0u32; 4]));
        self.spawn_total += sim.emitters.iter().map(|e| e.pool[2] as u64).sum::<u64>();
        if self.copied {
            queue.write_buffer(&self.events, 0, bytemuck::cast_slice(&[0u32; 8]));
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("GlassParticles/Step") });
        {
            let stamp = kansei_core::profiling::gpu_pass("GlassParticles/Sim");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GlassParticles/Sim"), timestamp_writes: stamp.as_ref().map(kansei_core::profiling::PassStamp::compute) });
            self.compute.dispatch(&mut pass, self.count.div_ceil(64), 1, 1);
        }
        {
            let stamp = kansei_core::profiling::gpu_pass("GlassParticles/Clear");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("GlassParticles/Clear"), timestamp_writes: stamp.as_ref().map(kansei_core::profiling::PassStamp::compute) });
            self.clear.dispatch(&mut pass, self.count.div_ceil(64), 1, 1);
        }
        // the events add up until a readback is free to take them
        self.copied = false;
        self.pending_frames += 1;
        if let Some(r) = self.readbacks.iter_mut().find(|r| r.state.load(Ordering::Relaxed) == 0) {
            encoder.copy_buffer_to_buffer(&self.events, 0, &r.buffer, 0, 32);
            encoder.copy_buffer_to_buffer(&self.spawned, 0, &r.buffer, 32, 16);
            r.state.store(1, Ordering::Relaxed);
            r.frames = std::mem::take(&mut self.pending_frames);
            r.spawn_total = self.spawn_total;
            self.copied = true;
            queue.submit(Some(encoder.finish()));
            let state = r.state.clone();
            r.buffer.slice(..).map_async(wgpu::MapMode::Read, move |result| state.store(if result.is_ok() { 2 } else { 0 }, Ordering::Relaxed));
        } else {
            queue.submit(Some(encoder.finish()));
        }
    }

    /// Add up the readbacks that came back.
    fn collect(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        for r in &self.readbacks {
            if r.state.load(Ordering::Relaxed) != 2 {
                continue;
            }
            {
                let data = r.buffer.slice(..).get_mapped_range().unwrap();
                let words: &[u32] = bytemuck::cast_slice(&data);
                for ((total, window), &w) in self.totals.iter_mut().zip(self.rates.counts.iter_mut()).zip(&words[..8]) {
                    *total += w as u64;
                    *window += w as u64;
                }
                self.rates.frames += r.frames as u64;
                // the live particles of the frame it was copied in (the newest reading wins)
                if r.spawn_total >= self.live_seen.1 {
                    self.live_seen = (words[10], r.spawn_total);
                }
            }
            r.buffer.unmap();
            r.state.store(0, Ordering::Relaxed);
        }
        let t = now();
        let rates = &mut self.rates;
        if t - rates.start >= 0.5 && rates.frames > 0 {
            let seconds = t - rates.start;
            let c = rates.counts;
            rates.emitted = (c[0] + c[1]) as f64 / seconds;
            rates.brushes = (c[2] + c[3] + c[4]) as f64 / seconds;
            rates.floor_hits = c[5] as f64 / seconds;
            rates.alive = c[7] as f64 / rates.frames as f64;
            rates.start = t;
            rates.counts = [0; 8];
            rates.frames = 0;
        }
    }
}

// ── The demo ──

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    volume: PostProcessingVolume,
    particles: Particles,
    particle_node: usize,
    lamp_node: usize,
    /// The environment's dome.
    backdrop: usize,
    floor: usize,
    settings: Settings,
    emitters: [Emitter; 2],
    /// Respawns owed to each emitter (the fraction its rate left over).
    owed: [f32; 2],
    nozzles: [usize; 2],
    spot: usize,
    colliders: Vec<Collider>,
    glass_nodes: Vec<usize>,
    glass: bool,
    /// The reflections' and the glass's sky (`SkyLighting`): the environment past the grid.
    sky: wgpu::Buffer,
    /// The lamp's instance record (a particle's layout: the boxes' shader draws it).
    lamp: wgpu::Buffer,
    /// The floor grid the ray traced hits see (baked into the reflections' shader).
    traced_grid: FloorGpu,
    /// The glass mode the scene and the chain were last brought to (`Settings::glass_mode`).
    applied_glass_mode: Option<u32>,
    time: f32,
    frame: u32,
    dt: f32,
    /// Frame interval, ms (smoothed).
    frame_ms: f64,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

/// The spot light, straight down over the middle.
fn spot_light(settings: &Settings) -> SpotLight {
    let colour = PALETTES[settings.palette].light;
    let outer = settings.cone.to_radians();
    let mut spot = SpotLight::new(Vec3::new(0.0, LIGHT_HEIGHT, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(colour[0], colour[1], colour[2]), settings.light, 14.0, outer * 0.6, outer);
    spot.cast_shadow = true;
    spot.source_radius = 0.04;
    spot
}

/// The lamp's face, a disc under the spot light, as an instance record: on its axis a flat
/// Lambertian emitter of the light's intensity over its area has that luminance (I = L A), so the
/// glass and the floor reflect a lamp as bright as the light it casts.
fn lamp_record(settings: &Settings) -> [u32; 16] {
    let luminance = settings.light / (std::f32::consts::PI * LAMP_RADIUS * LAMP_RADIUS);
    let glow = PALETTES[settings.palette].light.map(|c| c * luminance);
    let f = f32::to_bits;
    [f(0.0), f(LIGHT_HEIGHT + 0.03), f(0.0), f(1.0), f(0.0), f(0.0), f(0.0), f(1.0), f(0.0), f(0.0), f(0.0), f(1.0), 0xff0d0d0d, pack_emission(glow), 0, 0]
}

fn glass_material(settings: &Settings) -> StandardLitOptions {
    StandardLitOptions::glass(GLASS_TINT, settings.glass_ior, settings.glass_roughness)
}

/// A material's uniform at binding 0: written in place once it is on the GPU (the scene's bind
/// groups keep its buffer), made before.
fn write_uniform<T: bytemuck::Pod>(renderer: &Renderer, material: &mut Material, label: &str, data: &T) {
    match material.bindable_buffer(0) {
        Some(buffer) => renderer.queue().write_buffer(&buffer, 0, bytemuck::bytes_of(data)),
        None => material.set_uniform_bindable(0, label, std::slice::from_ref(data)),
    }
}

/// The floor's grid uniform (floor.wgsl's `FloorGrid`).
fn floor_grid(s: &Settings) -> FloorGpu {
    let rgb = |c: [f32; 3], w: f32| [c[0], c[1], c[2], w];
    FloorGpu {
        base: rgb(s.grid_base, 0.0),
        minor: rgb(s.grid_minor, s.grid_spacing),
        major: rgb(s.grid_major, s.grid_major_every),
        cross: rgb(s.grid_cross, s.grid_cross_arm),
        widths: [s.grid_minor_width, s.grid_major_width, s.grid_cross_width, FLOOR_ROUGHNESS],
        specular: [FLOOR_F0, 0.0, 0.0, 0.0],
        sky_up: sky_rgb(s.sky_up),
        sky_down: sky_rgb(s.sky_down),
    }
}

/// The environment drawn round the scene (`Material::gradient_sky`'s uniform): straight up, at the
/// horizon (the mean, so it runs linearly in the height as `gi::gradient_sky_lighting` does) and
/// below.
fn backdrop(s: &Settings) -> [f32; 12] {
    let sky = backdrop_sky(s);
    let ([zr, zg, zb], [hr, hg, hb], [gr, gg, gb]) = (sky.zenith, sky.horizon, sky.ground);
    [zr, zg, zb, sky.curve, hr, hg, hb, 0.0, gr, gg, gb, 0.0]
}

fn backdrop_sky(s: &Settings) -> GradientSkyOptions {
    let (up, down) = (sky_rgb(s.sky_up), sky_rgb(s.sky_down));
    let rgb = |c: [f32; 4]| [c[0], c[1], c[2]];
    let mid = [0, 1, 2].map(|k| 0.5 * (up[k] + down[k]));
    GradientSkyOptions { zenith: rgb(up), horizon: mid, ground: rgb(down), curve: 1.0 }
}

/// The backdrop's radius (m), round the origin.
const DOME_RADIUS: f32 = 45.0;

/// The environment's colour: a cool white.
fn sky_rgb(radiance: f32) -> [f32; 4] {
    [radiance * 0.96, radiance, radiance * 1.06, 0.0]
}

/// SSAO's settings: it occludes the environment's hemisphere, the ambient the particles and the
/// floor add.
fn ssao_options(s: &Settings) -> SSAOOptions {
    let rgb = |c: [f32; 4]| [c[0], c[1], c[2]];
    let ambient = SSAOAmbient::Hemisphere { up: rgb(sky_rgb(s.sky_up)), down: rgb(sky_rgb(s.sky_down)) };
    SSAOOptions { radius: s.ao_radius, bias: s.ao_bias, kernel_size: s.ao_samples, strength: s.ao, ambient }
}

/// The contact darkening's settings.
fn contact_ao_options(s: &Settings) -> ContactAoOptions {
    ContactAoOptions { radius: s.cao_radius, strength: s.cao_strength, falloff: s.cao_falloff, samples: s.cao_samples, ..Default::default() }
}

fn look(settings: &Settings, dt: f32) -> LookGpu {
    LookGpu { roughness: settings.roughness, metallic: settings.metallic, dt, _pad: 0.0, sky_up: sky_rgb(settings.sky_up), sky_down: sky_rgb(settings.sky_down) }
}

impl State {
    /// The pointer as the sim takes it: the ray from the eye through it, and its motion over the
    /// view plane at the orbit's target, in m/s.
    fn mouse_gpu(&self, dt: f32) -> ([f32; 4], [f32; 4], [f32; 4]) {
        // MouseVectors gives NDC with y down, and its direction is the previous position minus
        // the current one
        let (x, y) = (self.mouse.position.x, -self.mouse.position.y);
        let inverse = (self.camera.projection_matrix * self.camera.view_matrix).inverse().to_glam();
        let near = inverse.project_point3(glam::Vec3::new(x, y, 0.0));
        let far = inverse.project_point3(glam::Vec3::new(x, y, 1.0));
        let ray = (far - near).normalize_or_zero();
        let m = &self.camera.inverse_view_matrix.data;
        let (right, up) = (glam::Vec3::new(m[0], m[1], m[2]), glam::Vec3::new(m[4], m[5], m[6]));
        // NDC to metres at the target's distance
        let half_height = self.controls.radius * (0.5 * self.camera.fov.to_radians()).tan();
        let (dx, dy) = (-self.mouse.direction.x * half_height * self.camera.aspect, self.mouse.direction.y * half_height);
        let velocity = (right * dx + up * dy) / dt.max(1e-3);
        let strength = (self.mouse.strength * 40.0).min(1.0);
        (
            [near.x, near.y, near.z, strength],
            [ray.x, ray.y, ray.z, self.settings.mouse_radius],
            [velocity.x, velocity.y, velocity.z, self.settings.mouse_force],
        )
    }

    fn sim_gpu(&mut self, dt: f32) -> SimGpu {
        let s = &self.settings;
        let palette = &PALETTES[s.palette];
        let rgb = |c: [f32; 3]| [c[0], c[1], c[2], 0.0];
        let mut emitters = [EmitterGpu::default(); 2];
        for (k, e) in self.emitters.iter().enumerate() {
            let pool = if k == 0 { self.particles.split } else { self.particles.count - self.particles.split };
            let first = if k == 0 { 0 } else { self.particles.split };
            // the respawns its rate owes this frame, at most its pool
            self.owed[k] = if e.enabled && !s.paused { (self.owed[k] + e.rate * dt).min(pool as f32) } else { 0.0 };
            let spawn = self.owed[k].floor();
            self.owed[k] -= spawn;
            let r = e.rotation_matrix();
            emitters[k] = EmitterGpu {
                axis_x: [r.x_axis.x, r.x_axis.y, r.x_axis.z, 0.0],
                axis_y: [r.y_axis.x, r.y_axis.y, r.y_axis.z, 0.0],
                axis_z: [r.z_axis.x, r.z_axis.y, r.z_axis.z, 0.0],
                position: [e.position[0], e.position[1], e.position[2], e.shape as f32],
                launch: [e.radius, e.speed, e.spread.to_radians(), e.lifetime],
                jitter: [e.size, e.speed_jitter, e.life_jitter, e.glow_share],
                pool: [first, pool, spawn as u32, e.enabled as u32],
            };
        }
        let mut colliders = [ColliderGpu::default(); MAX_COLLIDERS];
        for (slot, c) in colliders.iter_mut().zip(&self.colliders) {
            *slot = c.gpu();
        }
        let (mouse_origin, mouse_ray, mouse_motion) = self.mouse_gpu(dt);
        let s = &self.settings;
        SimGpu {
            dt: if s.paused { 0.0 } else { dt },
            time: self.time,
            curl_frequency: s.curl_frequency,
            curl_speed: s.curl_speed,
            drag: s.drag,
            gravity: s.gravity,
            repel_range: s.repel_range,
            repel_strength: s.repel_strength,
            collider_count: self.colliders.len().min(MAX_COLLIDERS) as u32,
            frame: self.frame,
            glow: s.glow,
            stretch: s.stretch,
            curl_evolve: s.curl_evolve,
            bounce: s.bounce,
            hit_speed: 0.4,
            _pad0: 0.0,
            bounds_min: [BOUNDS_MIN[0], BOUNDS_MIN[1], BOUNDS_MIN[2], 0.0],
            bounds_max: [BOUNDS_MAX[0], BOUNDS_MAX[1], BOUNDS_MAX[2], 0.0],
            mouse_origin,
            mouse_ray,
            mouse_motion,
            base: palette.base.map(rgb),
            emissive: palette.glow.map(rgb),
            emitters,
            colliders,
        }
    }

    /// Bring the scene's lights, nozzles, glass and effects to the settings.
    fn apply(&mut self) {
        let s = self.settings;
        if let Some(Light::Spot(spot)) = self.scene.get_light_mut(self.spot) {
            *spot = spot_light(&s);
        }
        for (k, e) in self.emitters.iter().enumerate() {
            if let Some(r) = self.scene.get_renderable_mut(self.nozzles[k]) {
                let axis = e.rotation_matrix().y_axis;
                let base = glam::Vec3::from(e.position) - axis * NOZZLE_LENGTH;
                r.object.set_position(base.x, base.y, base.z);
                r.object.rotation = Vec3::new(e.rotation[0].to_radians(), e.rotation[1].to_radians(), e.rotation[2].to_radians());
                r.visible = e.enabled;
            }
        }
        // the glass: ray traced, in screen space, or (its pass off) polished grey solids
        let glass_on = self.glass && s.glass_pass;
        let screen_glass = glass_on && s.glass_mode == 1;
        let traced_glass = glass_on && s.glass_mode == 0;
        for &node in &self.glass_nodes {
            if let Some(r) = self.scene.get_renderable_mut(node) {
                if self.glass {
                    r.material.set_standard_lit(&self.renderer, &if glass_on { glass_material(&s) } else { SOLID_REPELLER });
                    r.rt = Some(if glass_on { RtSurface::glass(GLASS_TINT) } else { RtSurface::new(SOLID_REPELLER.base_color).with_smooth_normals() });
                }
                r.cast_shadow = s.glass_shadows || !glass_on;
                // in screen space the glass draws after the background copy it refracts
                r.material.options.transmissive = screen_glass;
            }
        }
        let glass_state = (glass_on as u32) << 8 | s.glass_mode << 4 | s.glass_interfaces;
        if self.glass && self.applied_glass_mode != Some(glass_state) {
            self.applied_glass_mode = Some(glass_state);
            // the bundles hold the glass in the set it was drawn in
            self.renderer.invalidate_bundle();
            if let Some(rt) = self.volume.effect_mut::<RtReflectionsEffect>() {
                rt.set_glass(traced_glass.then_some(RtGlass { interfaces: s.glass_interfaces, ..Default::default() }));
                rt.reset_history();
            }
        }
        self.volume.set_enabled::<ScreenSpaceGlassEffect>(screen_glass);
        if let Some(ss) = self.volume.effect_mut::<ScreenSpaceGlassEffect>() {
            ss.options.thickness = s.ss_thickness;
            ss.options.behind = s.ss_behind;
            ss.options.dispersion = s.ss_dispersion;
        }
        // the particles and the lamp go into the ray tracing grid only while a ray can meet them
        let traced = traced_glass || (self.glass && s.reflections);
        for node in [self.particle_node, self.lamp_node] {
            if let Some(r) = self.scene.get_renderable_mut(node) {
                r.rt = traced.then(|| RtSurface::new([0.5; 3]).with_record_surface(Some(ALBEDO_WORD), Some(EMISSION_WORD)));
            }
        }
        if let Some(grid) = self.renderer.rt_grid_mut() {
            grid.set_rebuild_every_frame(traced);
        }
        let grid = floor_grid(&s);
        if let Some(r) = self.scene.get_renderable_mut(self.backdrop) {
            write_uniform(&self.renderer, &mut r.material, "Backdrop/Sky", &backdrop(&s));
        }
        if let Some(r) = self.scene.get_renderable_mut(self.floor) {
            write_uniform(&self.renderer, &mut r.material, "Floor/Grid", &grid);
            r.rt = self.glass.then(|| RtSurface::new(s.grid_base).with_specular(FLOOR_F0));
        }
        // the rays see the grid as their effect's shader bakes it: a new effect for a new grid
        let traced = FloorGpu { sky_up: [0.0; 4], sky_down: [0.0; 4], ..grid };
        if self.glass && bytemuck::bytes_of(&traced) != bytemuck::bytes_of(&self.traced_grid) {
            self.traced_grid = traced;
            if let (Some(slot), Some(rt)) = (
                self.volume.effects.iter().position(|e| e.as_any().is::<RtReflectionsEffect>()),
                rt_effect(&self.renderer, &self.scene, self.floor, &s, &self.sky),
            ) {
                self.volume.effects[slot] = Box::new(rt);
            }
        }
        self.renderer.queue().write_buffer(&self.lamp, 0, bytemuck::cast_slice(&lamp_record(&s)));
        let rgb = |c: [f32; 4]| [c[0], c[1], c[2]];
        let sky = gradient_sky_lighting(rgb(sky_rgb(s.sky_up)), rgb(sky_rgb(s.sky_down)));
        self.renderer.queue().write_buffer(&self.sky, 0, bytemuck::cast_slice(&sky));
        // the passes: each on or off in the chain, with its settings; a temporal one coming back
        // starts its history afresh
        self.volume.set_enabled::<SSAOEffect>(s.ssao);
        if let Some(ao) = self.volume.effect_mut::<SSAOEffect>() {
            ao.options = ssao_options(&s);
        }
        let exposure = exposure_from_ev100(s.ev100);
        self.volume.set_enabled::<ToneMapEffect>(s.tone);
        if let Some(tone) = self.volume.effect_mut::<ToneMapEffect>() {
            let o = &mut tone.options;
            o.exposure = exposure;
            o.exposure_compensation = s.exposure_comp;
            o.tonemapper = TONE_MAPPERS[s.tone_mapper.min(TONE_MAPPERS.len() as u32 - 1) as usize];
            o.vignette = s.vignette;
            o.grain = s.grain;
            o.chromatic_aberration = s.chromatic;
        }
        let total = self.volume.effect::<ToneMapEffect>().map_or(exposure, |t| t.total_exposure());
        // the contact darkening; its occlusion alone shown about mid grey at the exposure, with
        // the passes that would cloud it (the fade, the mist, depth of field, bloom) left out
        let cao_view = s.contact_ao && s.cao_view;
        self.volume.set_enabled::<ContactAoEffect>(s.contact_ao);
        if let Some(cao) = self.volume.effect_mut::<ContactAoEffect>() {
            cao.options = contact_ao_options(&s);
            cao.show_occlusion = cao_view;
            cao.debug_white = 0.5 / total.max(1e-9);
        }
        let taa_back = s.taa && !self.volume.is_enabled::<TemporalAAEffect>();
        self.volume.set_enabled::<TemporalAAEffect>(s.taa);
        if let Some(taa) = self.volume.effect_mut::<TemporalAAEffect>() {
            taa.options = TemporalAAOptions { feedback_max: s.taa_feedback_max, feedback_min: s.taa_feedback_min.min(s.taa_feedback_max), exposure: total, variance_gamma: s.taa_gamma };
            if taa_back {
                taa.reset_history();
            }
        }
        self.volume.set_enabled::<BloomEffect>(s.bloom_on && !cao_view);
        if let Some(bloom) = self.volume.effect_mut::<BloomEffect>() {
            bloom.exposure = total;
            bloom.options = BloomOptions { threshold: s.bloom_threshold, knee: s.bloom_knee, intensity: s.bloom, radius: s.bloom_radius, use_emissive: s.bloom_emissive };
        }
        self.volume.set_enabled::<DistanceFogEffect>(s.fade && !cao_view);
        if let Some(fade) = self.volume.effect_mut::<DistanceFogEffect>() {
            fade.start = s.fade_start;
            fade.end = s.fade_end;
            fade.color = DistanceFogColor::GradientSky { sky: backdrop_sky(&s), centre: Vec3::ZERO, radius: DOME_RADIUS };
            fade.from_point = Some(Vec3::ZERO);
        }
        let fog_on = s.fog && !cao_view;
        let fog_back = fog_on && !self.volume.is_enabled::<VolumetricFogEffect>();
        self.volume.set_enabled::<VolumetricFogEffect>(fog_on);
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.base_density = s.fog_density;
            fog.anisotropy = s.fog_anisotropy;
            fog.height_falloff = s.fog_falloff;
            // a uniform sky of the environment's mean radiance, scaled
            let [r, g, b, _] = sky_rgb(0.5 * (s.sky_up + s.sky_down) * s.fog_ambient);
            fog.ambient = Vec3::new(r, g, b);
            fog.spot_scattering = if s.fog_steps == 0 { SpotScattering::Froxels } else { SpotScattering::Raymarched { steps: s.fog_steps } };
            if fog_back {
                fog.reset_history();
            }
            fog.update_lights(self.scene.lights());
        }
        self.renderer.set_render_scale(s.render_scale);
        self.volume.set_enabled::<CinematicDepthOfFieldEffect>(s.dof && !cao_view);
        if let Some(dof) = self.volume.effect_mut::<CinematicDepthOfFieldEffect>() {
            dof.lens.f_stop = s.f_stop;
            dof.lens.blade_count = s.dof_blades;
            dof.sample_count = s.dof_samples;
            dof.max_coc_fraction = s.dof_max_coc;
        }
        if let Some(ss) = self.volume.effect_mut::<ScreenSpaceGlassEffect>() {
            ss.set_sky_lighting(Some(&self.sky));
        }
        if let Some(rt) = self.volume.effect_mut::<RtReflectionsEffect>() {
            rt.update_lights(self.scene.lights());
            rt.intensity = s.refl_intensity;
            rt.max_distance = s.refl_distance;
            rt.temporal_blend = s.refl_blend;
            let resolution = if s.refl_quarter { RtTraceResolution::Quarter } else { RtTraceResolution::Half };
            if rt.reflections != s.reflections || rt.resolution() != resolution {
                rt.reflections = s.reflections;
                rt.set_resolution(resolution);
                rt.reset_history();
            }
        }
    }

    fn frame(&mut self, dt: f32) {
        // a hidden tab's long frame is not simulated as one step
        let dt = dt.clamp(0.0, 1.0 / 30.0);
        self.dt = dt;
        self.controls.update(&mut self.camera, dt);
        self.camera.update_view_matrix();
        self.mouse.update(dt);
        if !self.settings.paused {
            self.time += dt;
        }
        self.frame = self.frame.wrapping_add(1);
        let sim = self.sim_gpu(dt);
        let (device, queue) = (self.renderer.device().clone(), self.renderer.queue().clone());
        self.particles.step(&device, &queue, &sim);
        // every pass draws (and the grid gathers) the live particles, compacted first
        let drawn = self.particles.drawn();
        if let Some(r) = self.scene.get_renderable_mut(self.particle_node) {
            r.geometry.instance_count = drawn;
            // (none alive: no draw at all)
            r.visible = drawn > 0;
        }
        if let Some(r) = self.scene.get_renderable_mut(self.particle_node) {
            write_uniform(&self.renderer, &mut r.material, "GlassParticles/Look", &look(&self.settings, sim.dt));
        }

        let (time, focus) = (self.time, if self.settings.autofocus { self.controls.radius } else { self.settings.focus });
        if self.settings.autofocus {
            self.settings.focus = focus;
        }
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.time = time;
        }
        if let Some(dof) = self.volume.effect_mut::<CinematicDepthOfFieldEffect>() {
            dof.lens.focus_distance_m = focus;
        }
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        self.particles.collect(&device);
        // the performance overlay's GPU passes (it profiles only while it shows)
        kansei_wasm::perf::profile(&mut self.renderer);
    }

    fn info(&self) -> String {
        let palettes: Vec<String> = PALETTES.iter().map(|p| format!("\"{}\"", p.name)).collect();
        let r = &self.particles.rates;
        format!(
            "{{\"drawn\":{},\"alive\":{:.0},\"emitted_per_s\":{:.0},\"brushes_per_s\":{:.0},\"floor_hits_per_s\":{:.0},\"perf\":{},\"lightings\":[{}],\"particles\":{},\"frame_ms\":{:.2},\"glass\":{},\"colliders\":{},\"palettes\":[{}],\"settings\":{},\"emitters\":[{},{}]}}",
            self.particles.drawn(),
            r.alive,
            r.emitted,
            r.brushes,
            r.floor_hits,
            kansei_wasm::perf::visible(),
            LIGHTINGS.iter().map(|l| format!("\"{}\"", l.name)).collect::<Vec<_>>().join(","),
            self.particles.count,
            self.frame_ms,
            self.glass,
            self.colliders.len(),
            palettes.join(","),
            self.settings.json(),
            self.emitters[0].json(),
            self.emitters[1].json()
        )
    }
}

// browser only (the controls read the canvas); the shaders' tests run natively
#[cfg(target_arch = "wasm32")]
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    // at most 1.5 drawing-buffer pixels per CSS pixel (the ray traced glass and the mist cost by
    // the pixel, and the lens blurs most of the frame); ?dpr= overrides it
    let canvas = Canvas::find(canvas_id)?.with_max_pixel_ratio(1.5);
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    let count = param_or("n", 32768u32).clamp(4096, 262144) & !1;
    let glass = flag("glass", true);
    let mut settings = Settings::default();
    if let Some(p) = param("palette").and_then(|p| palette_index(&p)) {
        settings.palette = p;
    }
    if let Some(l) = param("lighting").and_then(|n| LIGHTINGS.iter().find(|l| l.name == n)) {
        settings.lighting(l);
    }
    settings.ev100 = param_or("ev", settings.ev100);
    settings.dof = flag("dof", true);
    settings.fog = flag("fog", true);
    settings.fade = flag("fade", true);
    settings.contact_ao = flag("contactao", false);
    settings.cao_view = flag("aoview", false);
    settings.glass_mode = param_or("glassmode", 0u32).min(1);
    settings.render_scale = param_or("scale", 1.0f32).clamp(0.25, 1.0);

    renderer.enable_spot_shadows(2048, 1);
    if glass {
        // the ray traced glass and floor read the voxels past the grid (and need them running)
        renderer.enable_voxel_gi(SceneVoxelGiOptions {
            quality: VoxelGiQuality::Low,
            bounds_min: [BOUNDS_MIN[0], BOUNDS_MIN[1] - 0.2, BOUNDS_MIN[2]],
            bounds_max: BOUNDS_MAX,
            radiance_scale: 1.0,
            budget_bytes: 0,
        });
        // the room's triangles, the particles' among them, rebuilt every frame (they all move)
        let cell = (BOUNDS_MAX[0] - BOUNDS_MIN[0]) / 96.0;
        renderer.enable_rt_grid(SceneRtGridOptions {
            grid: RtGridOptions { dims: [96, 64, 96], cell, fixed_origin: Some(glam::Vec3::new(BOUNDS_MIN[0], BOUNDS_MIN[1] - 0.2, BOUNDS_MIN[2])), ..Default::default() },
            cluster_error_cells: 1.0,
            rebuild_every_frame: true,
        });
    }

    let mut scene = Scene::new();

    // a dark polished floor with a technical grid, reflecting the room (ray traced) and bouncing
    // the light (voxels)
    let mut floor_material = Material::new("Floor", &floor_shader(), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    floor_material.set_uniform_bindable(0, "Floor/Grid", &[floor_grid(&settings)]);
    let mut floor = Renderable::new(PlaneGeometry::new(120.0, 120.0), floor_material);
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.cast_shadow = false;
    floor.gi = Some(GiSurface::new(settings.grid_base));
    floor.rt = glass.then(|| RtSurface::new(settings.grid_base).with_specular(FLOOR_F0));
    let floor = scene.add(SceneNode::Renderable(floor));

    // the environment round it all: the sky the glass and the floor reflect, drawn
    let mut dome = Renderable::new(SphereGeometry::new(DOME_RADIUS, 48, 24), Material::gradient_sky("Backdrop", &GradientSkyOptions { curve: 1.0, ..Default::default() }));
    dome.cast_shadow = false;
    let backdrop = scene.add(SceneNode::Renderable(dome));

    // the repellers: glass (clear to the shadow maps and the voxels)
    let collider_list = colliders();
    let mut glass_nodes = Vec::new();
    for c in &collider_list {
        let material = if glass { Material::standard_lit("Glass", &glass_material(&settings)) } else { Material::standard_lit("Repeller", &StandardLitOptions { base_color: [0.5; 3], roughness: 0.3, ..Default::default() }) };
        let mut r = Renderable::new(c.geometry(), material);
        r.object.set_position(c.center[0], c.center[1], c.center[2]);
        r.object.rotation = Vec3::new(c.rotation[0].to_radians(), c.rotation[1].to_radians(), c.rotation[2].to_radians());
        if glass {
            r.rt = Some(RtSurface::glass(GLASS_TINT));
        }
        r.cast_shadow = settings.glass_shadows;
        glass_nodes.push(scene.add(SceneNode::Renderable(r)));
    }

    // the emitters' nozzles: short dark metal cones, open along the emitter's axis
    let mut nozzles = [0; 2];
    for nozzle in &mut nozzles {
        let geometry = CylinderGeometry::new(0.07, 0.035, NOZZLE_LENGTH, 24, 1);
        let mut r = Renderable::new(geometry, Material::standard_lit("Nozzle", &StandardLitOptions { base_color: [0.6, 0.6, 0.62], roughness: 0.3, metallic: 1.0, ..Default::default() }));
        r.dynamic = true;
        *nozzle = scene.add(SceneNode::Renderable(r));
    }

    // the particles: a unit box per instance, placed by the sim's records
    let particles = Particles::new(renderer.device(), count);
    let mut material = Material::new(
        "GlassParticles/Boxes",
        &draw_shader(),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() },
    );
    material.set_uniform_bindable(0, "GlassParticles/Look", &[look(&settings, 1.0 / 60.0)]);
    let mut boxes = Renderable::new(InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), 0, vec![particles.instance_buffer()]), material);
    // drawn live, as many as are alive (`Particles::drawn`), not from a bundle
    boxes.dynamic = true;
    if glass {
        boxes.rt = Some(RtSurface::new([0.5; 3]).with_record_surface(Some(ALBEDO_WORD), Some(EMISSION_WORD)));
        boxes.rt_placement = Some(RtPlacement::Wgsl(PLACE_WGSL.into()));
    }
    let particle_node = scene.add(SceneNode::Renderable(boxes));

    // the lamp's face: one record of the particles' layout, drawn and traced as they are
    let lamp = wgpu::util::DeviceExt::create_buffer_init(renderer.device(), &wgpu::util::BufferInitDescriptor {
        label: Some("GlassParticles/Lamp"),
        contents: bytemuck::cast_slice(&lamp_record(&settings)),
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut lamp_material = Material::new(
        "GlassParticles/Lamp",
        &draw_shader(),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() },
    );
    lamp_material.set_uniform_bindable(0, "GlassParticles/LampLook", &[look(&settings, 0.0)]);
    let lamp_records = Particles::records(&lamp, "GlassParticles/Lamp");
    let mut lamp_face = Renderable::new(InstancedGeometry::new(CylinderGeometry::new(LAMP_RADIUS, LAMP_RADIUS, 0.03, 48, 1), 1, vec![lamp_records]), lamp_material);
    lamp_face.cast_shadow = false;
    if glass {
        lamp_face.rt = Some(RtSurface::new([0.05; 3]).with_record_surface(Some(ALBEDO_WORD), Some(EMISSION_WORD)));
        lamp_face.rt_placement = Some(RtPlacement::Wgsl(PLACE_WGSL.into()));
    }
    let lamp_node = scene.add(SceneNode::Renderable(lamp_face));

    let spot = scene.add(SceneNode::Light(Light::Spot(spot_light(&settings))));

    // the chain: SSAO, the ray traced glass and reflections, the screen-space glass, the contact
    // darkening, the distance fade, the mist, TAA, depth of field, bloom, the tone mapper; each
    // pass on or off (`set_enabled`) by `apply`
    let tone = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(settings.ev100);
        options.vignette = 0.35;
        ToneMapEffect::new(options)
    };
    let exposure = tone.total_exposure();
    let sky = wgpu::util::DeviceExt::create_buffer_init(renderer.device(), &wgpu::util::BufferInitDescriptor {
        label: Some("GlassParticles/Sky"),
        contents: bytemuck::cast_slice(&gradient_sky_lighting([0.0; 3], [0.0; 3])),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(SSAOEffect::new(ssao_options(&settings)))];
    if let Some(rt) = rt_effect(&renderer, &scene, floor, &settings, &sky).filter(|_| glass) {
        effects.push(Box::new(rt));
    }
    effects.push(Box::new(ScreenSpaceGlassEffect::new(ScreenSpaceGlassOptions::default())));
    effects.push(Box::new(ContactAoEffect::new(contact_ao_options(&settings))));
    // the far floor fades into the backdrop behind it, so it never ends on a line
    effects.push(Box::new(DistanceFogEffect::new(settings.fade_start, settings.fade_end, DistanceFogColor::Constant([0.0; 3]))));
    {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.1, far: 24.0, temporal: true, ..Default::default() },
            base_density: settings.fog_density,
            height_falloff: 0.08,
            anisotropy: settings.fog_anisotropy,
            ambient: Vec3::new(0.02, 0.022, 0.026),
            spot_scattering: SpotScattering::Raymarched { steps: 32 },
            ..Default::default()
        });
        fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        fog.update_lights(scene.lights());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure, ..Default::default() })));
    effects.push(Box::new(CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
        lens: CameraLens { focal_length_mm: None, f_stop: settings.f_stop, focus_distance_m: settings.focus, sensor_width_mm: 23.76, blade_count: 7, blade_rotation_deg: 0.0 },
        ..Default::default()
    })));
    effects.push(Box::new(
        BloomEffect::new(BloomOptions { threshold: settings.bloom_threshold, knee: 0.5, intensity: settings.bloom, radius: 1.0, use_emissive: true }).with_exposure(exposure),
    ));
    effects.push(Box::new(tone));
    let volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(38.0, 0.05, 60.0, canvas.aspect());
    camera.update_projection_matrix();
    let target = Vec3::new(0.0, 1.5, 0.0);
    let mut controls = CameraControls::from_canvas(canvas.element(), target, 7.2).with_mouse_pan(canvas.element());
    controls.set_view(target, 7.2, 0.35, 0.16);
    let mouse = MouseVectors::from_canvas(canvas.element());

    // kansei-wasm's performance overlay, shown unless ?perf=0 (the backquote key toggles it)
    kansei_wasm::perf::set_visible(flag("perf", true));
    let mut state = State {
        renderer,
        scene,
        camera,
        controls,
        mouse,
        volume,
        particles,
        particle_node,
        lamp_node,
        backdrop,
        floor,
        settings,
        emitters: EMITTERS,
        owed: [0.0; 2],
        nozzles,
        spot,
        colliders: collider_list,
        glass_nodes,
        glass,
        applied_glass_mode: None,
        traced_grid: FloorGpu { sky_up: [0.0; 4], sky_down: [0.0; 4], ..floor_grid(&settings) },
        sky,
        lamp,
        time: 0.0,
        frame: 0,
        dt: 1.0 / 60.0,
        frame_ms: 16.7,
    };
    state.apply();
    log::info!("Kansei — Glass particles (WASM) ready: {count} particles, glass {glass}");

    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| {
        let mut st = state.borrow_mut();
        let st = &mut *st;
        frame.resize(&mut st.renderer, &mut st.camera);
        st.frame_ms = st.frame_ms * 0.95 + frame.dt as f64 * 1000.0 * 0.05;
        st.frame(frame.dt);
    });
    Ok(())
}

/// The particle count, the frame interval, the palettes, the settings and the emitters (JSON).
#[wasm_bindgen]
pub fn info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// One setting by its `info().settings` key (booleans as 0 or 1).
#[wasm_bindgen]
pub fn set(key: &str, value: f32) {
    with_state(|s| {
        if s.settings.set(key, value) {
            s.apply();
        } else {
            log::warn!("glass-particles: no setting {key}");
        }
    });
}

/// One of the floor grid's colours (`base`, `minor`, `major` or `cross`), linear rgb.
#[wasm_bindgen]
pub fn set_grid_color(which: &str, r: f32, g: f32, b: f32) {
    with_state(|s| {
        let c = [r, g, b].map(|x| x.clamp(0.0, 1.0));
        match which {
            "base" => s.settings.grid_base = c,
            "minor" => s.settings.grid_minor = c,
            "major" => s.settings.grid_major = c,
            "cross" => s.settings.grid_cross = c,
            _ => return,
        }
        s.apply();
    });
}

/// The palette by name (`info().palettes`).
#[wasm_bindgen]
pub fn set_palette(name: &str) {
    with_state(|s| {
        if let Some(p) = palette_index(name) {
            s.settings.palette = p;
            s.apply();
        }
    });
}

/// One of emitter `index`'s settings by its `info().emitters[index]` key (`x`, `y`, `z`; `rx`,
/// `ry`, `rz` in degrees; `shape` 0 point, 1 ball, 2 disc; `radius`, `speed`, `spread` (degrees),
/// `lifetime`, `rate` (a second), `size`, `speedJitter`, `lifeJitter`, `glowShare`, `enabled`).
#[wasm_bindgen]
pub fn set_emitter(index: u32, key: &str, value: f32) {
    with_state(|s| {
        let Some(e) = s.emitters.get_mut(index as usize) else { return };
        if e.set(key, value) {
            s.apply();
        } else {
            log::warn!("glass-particles: no emitter setting {key}");
        }
    });
}

/// The sim's events since the last call (JSON): respawns of each emitter, brushes with spheres,
/// boxes and pyramids (a particle coming within half their reach) and hits on the floor, and their
/// mean speed toward the surface (m/s). Read back from the GPU a few frames late, in bulk.
#[wasm_bindgen]
pub fn take_events() -> String {
    with_state(|s| {
        let t = std::mem::take(&mut s.particles.totals);
        let hits = t[2] + t[3] + t[4] + t[5];
        let speed = if hits > 0 { t[6] as f64 / 100.0 / hits as f64 } else { 0.0 };
        format!("{{\"spawn\":[{},{}],\"hits\":[{},{},{},{}],\"speed\":{:.3}}}", t[0], t[1], t[2], t[3], t[4], t[5], speed)
    })
    .unwrap_or_default()
}

/// kansei-wasm's performance overlay on or off (`kansei_wasm::perf`; also the backquote key).
#[wasm_bindgen]
pub fn set_perf(on: bool) {
    kansei_wasm::perf::set_visible(on);
}

/// A lighting preset by name (`info().lightings`): the spot light, the environment, the exposure
/// and the mist together, each still editable after.
#[wasm_bindgen]
pub fn set_lighting(name: &str) {
    with_state(|s| {
        if let Some(l) = LIGHTINGS.iter().find(|l| l.name == name) {
            s.settings.lighting(l);
            s.apply();
        }
    });
}

/// The orbit (JSON: `target`, `radius`, `azimuth` and `elevation` in radians).
#[wasm_bindgen]
pub fn camera() -> String {
    with_state(|s| {
        let t = s.controls.look_target();
        format!("{{\"target\":[{:.3},{:.3},{:.3}],\"radius\":{:.3},\"azimuth\":{:.4},\"elevation\":{:.4}}}", t.x, t.y, t.z, s.controls.radius, s.controls.azimuth(), s.controls.elevation())
    })
    .unwrap_or_default()
}

/// Frame the orbit (for scripted captures).
#[wasm_bindgen]
pub fn set_camera(x: f32, y: f32, z: f32, radius: f32, azimuth: f32, elevation: f32) {
    with_state(|s| s.controls.set_view(Vec3::new(x, y, z), radius.max(0.5), azimuth, elevation));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::empty())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        module
    }

    fn span(module: &naga::Module, name: &str) -> usize {
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
    fn shaders_validate_and_match_their_layouts() {
        let sim = validate("sim", &sim_shader());
        assert_eq!(span(&sim, "Sim"), std::mem::size_of::<SimGpu>());
        assert_eq!(span(&sim, "Emitter"), std::mem::size_of::<EmitterGpu>());
        assert_eq!(span(&sim, "Collider"), std::mem::size_of::<ColliderGpu>());
        assert_eq!(span(&sim, "Instance") as u64, INSTANCE_BYTES);
        let floor = validate("floor", &floor_shader());
        // the albedo hook, with the names the reflections' shader gives it
        validate(
            "hit albedo",
            &format!(
                "struct P {{ invView: mat4x4f }}\n@group(0) @binding(0) var<uniform> rp: P;\n@group(0) @binding(1) var<storage, read> tris: array<u32>;\n\
                 fn kansei_rt_source(id: u32) -> u32 {{ return tris[id] >> 20u; }}\nfn kansei_rt_albedo(id: u32) -> vec3f {{ return unpack4x8unorm(tris[id]).rgb; }}\n{}\n\
                 @group(0) @binding(2) var<storage, read_write> out: array<vec4f>;\n@compute @workgroup_size(1) fn main() {{ out[0] = vec4f(kansei_rt_hit_albedo(0u, vec3f(0.3, 0.0, 0.7), vec2f(0.0)), 0.0); }}",
                hit_albedo_wgsl(0, &floor_grid(&Settings::default()))
            ),
        );
        assert_eq!(span(&floor, "FloorGrid"), std::mem::size_of::<FloorGpu>());
        let draw = validate("draw", &draw_shader());
        assert_eq!(span(&draw, "ParticleLook"), std::mem::size_of::<LookGpu>());
        // the grid's placement, as the gather compiles it
        validate(
            "placement",
            &format!(
                "@group(0) @binding(0) var<storage, read> records: array<u32>;\nfn kansei_rt_record_f32(record: u32, word: u32) -> f32 {{ return bitcast<f32>(records[record * 16u + word]); }}\n\
                 fn kansei_rt_record_vec4(record: u32, word: u32) -> vec4f {{ return vec4f(kansei_rt_record_f32(record, word), kansei_rt_record_f32(record, word + 1u), kansei_rt_record_f32(record, word + 2u), kansei_rt_record_f32(record, word + 3u)); }}\n\
                 {PLACE_WGSL}\n@group(0) @binding(1) var<storage, read_write> out: array<vec4f>;\n@compute @workgroup_size(1) fn main() {{ out[0] = vec4f(kansei_rt_place(0u, vec3f(0.5)), 0.0); }}"
            ),
        );
    }

    #[test]
    fn the_pyramid_faces_out() {
        let g = pyramid(1.0, 1.2);
        for t in g.indices.chunks_exact(3) {
            let p = |i: u32| glam::Vec4::from(g.vertices[i as usize].position).truncate();
            let (a, b, c) = (p(t[0]), p(t[1]), p(t[2]));
            let n = (b - a).cross(c - a);
            let centre = (a + b + c) / 3.0 - glam::Vec3::new(0.0, 0.3, 0.0);
            assert!(n.dot(centre) > 0.0, "a face winds inward: {a} {b} {c}");
            assert!(n.normalize().dot(glam::Vec3::from(g.vertices[t[0] as usize].normal)) > 0.99, "its normal is its face's");
        }
    }

    #[test]
    fn the_settings_round_trip_their_keys() {
        let mut s = Settings::default();
        assert!(s.set("curlSpeed", 2.0) && s.curl_speed == 2.0);
        assert!(!s.set("nothing", 1.0));
        let mut e = EMITTERS[0];
        assert!(e.set("rz", 30.0) && e.rotation[2] == 30.0);
        // the emitter points along its rotated +y: -58 degrees about z leans it toward +x
        let up = EMITTERS[0].rotation_matrix().y_axis;
        assert!(up.x > 0.8 && up.y > 0.5, "{up}");
        assert!(s.json().contains("\"palette\":\"ember\""));
    }
}
