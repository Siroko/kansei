//! A small lake beside the course: an irregular outline, a bed that shelves from the waterline
//! down to about 0.6 m in the middle, and a low shore around it. The water is the engine's SPH
//! fluid (`simulations::fluid`, as in the fluid clock), held by a `FluidContainer` whose walls
//! follow the outline a strip of shore outside it and whose floor is the bed, and pushed by the
//! character's legs through `FluidColliders`. The character wades: the bed is in the collision
//! world, so it walks down into the shallows and out again, splashing.
//!
//! The simulation runs 11 times the world's size (a particle every 5 cm, with the fluid clock's
//! tuned constants at a smoothing radius of 1) and so √11 times faster than real time, which keeps
//! gravity-driven motion (waves, splashes) at its real pace.

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb, Shape, TriangleMesh};
use kansei_core::geometries::{Geometry, Vertex};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{FluidSurfaceEffect, FluidSurfaceOptions};
use kansei_core::renderers::Renderer;
use kansei_core::shadows::CASCADED_SHADOWS_WGSL;
use kansei_core::simulations::fluid::{
    DensityFieldOptions, FluidCapsule, FluidColliders, FluidCollidersOptions, FluidContainer,
    FluidContainerOptions, FluidDensityField, FluidMarchingCubes, FluidSimulation, FluidSimulationOptions,
    FluidSubstepPass, MarchingCubesOptions, PlanarContainerShape, DEFAULT_OPTIONS,
};

use crate::{SKY, SUN, SUN_DIR};

/// Simulation units per metre.
const SIM_SCALE: f32 = 11.0;
/// Simulated seconds per real second: √SIM_SCALE, so gravity acts at its real pace.
const TIME_SCALE: f32 = 3.316_625;
/// Real seconds per simulation step (up to `MAX_STEPS` a frame).
const STEP: f32 = 1.0 / 60.0;
const MAX_STEPS: u32 = 2;
/// The lake's middle (x, z) and half-size along x and z.
const CENTER: [f32; 2] = [21.0, -1.0];
const HALF: [f32; 2] = [5.2, 3.4];
/// The still water's height, a little under the ground's.
const WATER: f32 = -0.1;
/// The bed: its depth in the middle, and how far in from the outline it gets there, steepest at
/// the outline (so the water's edge is short, not a long film a particle thick).
const DEPTH: f32 = 0.6;
const SHELF: f32 = 2.6;
/// The shore: the container's walls stand this far outside the outline, on a low bank that
/// drains back into the lake and goes down to the ground past them.
const SHORE: f32 = 1.2;
const BANK: f32 = 0.1;
const BANK_OUT: f32 = 1.6;
/// A landing's splash: the sphere at the feet (m), how long it pushes (s), and its push outward
/// per m/s of the fall.
const SPLASH_RADIUS: f32 = 0.3;
const SPLASH_TIME: f32 = 0.12;
const SPLASH_PUSH: f32 = 1.2;
/// How the water's surface is extracted from the particles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SurfaceSettings {
    /// A surface field (droplets, flat layers) rather than the splatted density.
    pub surface_field: bool,
    /// Voxels along the field's longest side.
    pub resolution: u32,
    /// The splat's kernel radius (simulation units).
    pub kernel: f32,
    /// The surface field's particle radius (simulation units).
    pub particle_radius: f32,
    /// The iso level: 1 for a surface field; a density for the density field.
    pub iso: f32,
    /// Interpolated marching cubes (else voxel faces).
    pub interpolate: bool,
}

impl SurfaceSettings {
    /// The default: a surface field, spray as droplets 8 cm across, at the density build's cost.
    pub const DROPLETS: Self = Self { surface_field: true, resolution: 256, kernel: 1.5, particle_radius: 0.45, iso: 1.0, interpolate: true };
    /// The splatted density at an iso level, wide and smooth: spray is blobby.
    pub const SMOOTH: Self = Self { surface_field: false, resolution: 256, kernel: 1.6, particle_radius: 0.45, iso: 0.5, interpolate: true };
    /// A coarser surface field, for slower GPUs.
    pub const PERFORMANCE: Self = Self { surface_field: true, resolution: 192, kernel: 1.5, particle_radius: 0.55, iso: 1.0, interpolate: true };

    fn density_options(&self) -> DensityFieldOptions {
        DensityFieldOptions {
            resolution: self.resolution,
            // surface field: 1 over the kernel weight of the bulk (particles per unit³ × 0.638 h³)
            kernel_scale: if self.surface_field { 1.0 / (SPACING.powi(-3) * 0.638 * self.kernel.powi(3)) } else { 0.6 },
            particle_radius: self.surface_field.then_some(self.particle_radius),
        }
    }
}
/// Lattice spacing of the particles at the fluid clock's rest density (simulation units).
const SPACING: f32 = 0.537;

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The waterline: a wobbly ellipse.
fn outline() -> Vec<[f32; 2]> {
    (0..256)
        .map(|k| {
            let a = k as f32 / 256.0 * std::f32::consts::TAU;
            let r = 1.0 + 0.11 * (2.0 * a + 0.6).sin() + 0.07 * (3.0 * a + 2.1).sin() + 0.04 * (5.0 * a + 0.3).sin();
            [CENTER[0] + HALF[0] * r * a.cos(), CENTER[1] + HALF[1] * r * a.sin()]
        })
        .collect()
}

/// The ground's height at signed distance `d` from the waterline (negative in the lake).
fn height(d: f32) -> f32 {
    if d < 0.0 {
        let t = (-d / SHELF).min(1.0);
        -DEPTH * (1.0 - (1.0 - t) * (1.0 - t))
    } else if d < SHORE {
        BANK * smoothstep(0.0, SHORE, d)
    } else {
        BANK * (1.0 - smoothstep(SHORE, SHORE + BANK_OUT, d))
    }
}

/// Lake bed and shore: grey ground with the course's metre grid, turning to wet silt below the
/// water and darker with depth; sun (cascade-shadowed) and sky. `surface.base_color.w` is the
/// water's height.
const TERRAIN_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, sun_dir: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = normal;
    return out;
}
fn grid(p: vec2<f32>, spacing: f32, width: f32) -> f32 {
    let q = p / spacing;
    let d = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(d.x, d.y) / width, 1.0);
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let lines = max(grid(in.world.xz, 1.0, 1.0) * 0.35, grid(in.world.xz, 5.0, 1.5) * 0.6);
    let under = in.world.y - surface.base_color.w;
    // wet up to a damp margin over the still water (the water settles a few cm over the height
    // it is filled to), so no dry band shows through the shallows
    let wet = smoothstep(0.08, 0.04, under);
    let silt = mix(vec3<f32>(0.3, 0.26, 0.19), vec3<f32>(0.16, 0.15, 0.11), smoothstep(0.0, -0.5, under));
    let base = mix(surface.base_color.rgb, silt, wet) * (1.0 - lines * mix(1.0, 0.4, wet));
    let l = -normalize(surface.sun_dir.xyz);
    let lit = base / 3.14159265 * surface.sun.rgb * max(dot(n, l), 0.0) * shadow + base * surface.sky.rgb * (0.6 + 0.4 * n.y);
    return vec4<f32>(lit, 1.0);
}
"#;

/// The water's surface (the marching-cubes mesh, in simulation space): writes its world normal to
/// the GBuffer, where `FluidSurfaceEffect` finds it and composites the refraction and reflection.
const WATER_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> color: vec4<f32>;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(position.xyz, 1.0);
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
struct FOut { @location(0) color: vec4<f32>, @location(1) emissive: vec4<f32>, @location(2) normal: vec4<f32>, @location(3) albedo: vec4<f32> };
@fragment
fn fragment_main(in: VOut) -> FOut {
    let n = normalize(in.normal);
    return FOut(color, vec4<f32>(0.0), vec4<f32>(n, 1.0), color);
}
"#;

/// A grid of vertices over `[min, max]` (x, z), `cells` across, at `height(x, z)`, with normals
/// from the heights around; triangles wound counter-clockwise seen from above.
fn heightfield(min: [f32; 2], max: [f32; 2], cells: [usize; 2], height: &dyn Fn(f32, f32) -> f32) -> (Vec<GVec3>, Vec<[f32; 3]>, Vec<u32>) {
    let step = [(max[0] - min[0]) / cells[0] as f32, (max[1] - min[1]) / cells[1] as f32];
    let (mut positions, mut normals) = (Vec::new(), Vec::new());
    for j in 0..=cells[1] {
        for i in 0..=cells[0] {
            let (x, z) = (min[0] + i as f32 * step[0], min[1] + j as f32 * step[1]);
            positions.push(GVec3::new(x, height(x, z), z));
            let e = 0.1;
            let n = GVec3::new(height(x - e, z) - height(x + e, z), 2.0 * e, height(x, z - e) - height(x, z + e)).normalize();
            normals.push(n.to_array());
        }
    }
    let row = cells[0] as u32 + 1;
    let mut indices = Vec::new();
    for j in 0..cells[1] as u32 {
        for i in 0..cells[0] as u32 {
            let (a, b, c, d) = (j * row + i, j * row + i + 1, (j + 1) * row + i + 1, (j + 1) * row + i);
            // (x, z) → (x, z + 1) → (x + 1, z + 1) turns counter-clockwise seen from +y
            indices.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }
    (positions, normals, indices)
}

fn geometry(label: &str, positions: &[GVec3], normals: &[[f32; 3]], indices: Vec<u32>) -> Geometry {
    let vertices = positions.iter().zip(normals).map(|(p, n)| Vertex { position: [p.x, p.y, p.z, 1.0], normal: *n, uv: [p.x, p.z] }).collect();
    Geometry::new(label, vertices, indices)
}

/// The ground outside a rectangle `[min, max]` (x, z), out to `extent` all round: four flat quads
/// at height 0, with the course's ground material.
fn ground_around(scene: &mut Scene, world: &mut CollisionWorld, material: Material, min: [f32; 2], max: [f32; 2], extent: f32) {
    let quads = [
        ([-extent, -extent], [extent, min[1]]),
        ([-extent, max[1]], [extent, extent]),
        ([-extent, min[1]], [min[0], max[1]]),
        ([max[0], min[1]], [extent, max[1]]),
    ];
    let (mut positions, mut indices) = (Vec::new(), Vec::new());
    for (a, b) in quads {
        let k = positions.len() as u32;
        positions.extend([GVec3::new(a[0], 0.0, a[1]), GVec3::new(b[0], 0.0, a[1]), GVec3::new(b[0], 0.0, b[1]), GVec3::new(a[0], 0.0, b[1])]);
        indices.extend_from_slice(&[k, k + 3, k + 2, k, k + 2, k + 1]);
        world.add_box(Obb::from_min_max(GVec3::new(a[0], -1.0, a[1]), GVec3::new(b[0], 0.0, b[1])));
    }
    let normals = vec![[0.0, 1.0, 0.0]; positions.len()];
    let mut r = Renderable::new(geometry("Ground", &positions, &normals, indices), material);
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r));
}

/// The lake: its shape, the simulation's passes, and the character's legs in it.
pub struct Lake {
    container: FluidContainer,
    colliders: FluidColliders,
    /// The effect's index in the post-processing chain (it owns the simulation).
    pub effect: usize,
    accumulator: f32,
    /// Last frame's capsule ends (world), for their velocities.
    previous: Vec<[GVec3; 2]>,
    particles: u32,
    /// A landing's splash: where (world), how fast it came down, and for how long it has pushed.
    splash: Option<(GVec3, f32, f32)>,
    /// The waterline's bounds (x, z): legs within 2 m of them push the water.
    near: ([f32; 2], [f32; 2]),
    /// Simulated seconds per real second.
    pub time_scale: f32,
    /// A landing splash's push outward per m/s of the fall.
    pub splash_push: f32,
    /// The particles as they started, for a reset.
    initial: Vec<f32>,
    surface: SurfaceSettings,
}

/// Blur the floor of `shape` (a box filter `radius` nodes each way, along x then z, twice): the
/// distance to the outline creases outside its concave stretches, and so would a bank shaped by
/// it.
fn smooth_floor(shape: &mut PlanarContainerShape, radius: i32) {
    let [w, h] = shape.dims.map(|d| d as i32);
    for _ in 0..2 {
        for axis in 0..2 {
            let floor: Vec<f32> = shape.nodes.iter().map(|n| n[1]).collect();
            for j in 0..h {
                for i in 0..w {
                    let (mut sum, mut n) = (0.0, 0.0);
                    for k in -radius..=radius {
                        let (x, z) = if axis == 0 { (i + k, j) } else { (i, j + k) };
                        if x >= 0 && x < w && z >= 0 && z < h {
                            sum += floor[(z * w + x) as usize];
                            n += 1.0;
                        }
                    }
                    shape.nodes[(j * w + i) as usize][1] = sum / n;
                }
            }
        }
    }
}

impl Lake {
    /// Build the lake into `scene` and `world`: its terrain, the ground around it (with
    /// `ground_material`), and its surface effect, for the post-processing chain at index
    /// `effect`.
    pub fn new(renderer: &Renderer, scene: &mut Scene, world: &mut CollisionWorld, ground_material: Material, effect: usize) -> (Self, FluidSurfaceEffect) {
        let outline = outline();
        // the terrain's heights on a grid over the lake and its banks, smoothed; flat ground
        // around it
        let mut terrain = PlanarContainerShape::from_outline(&outline, 0.1, SHORE + BANK_OUT + 0.5, |_, _, d| height(d));
        smooth_floor(&mut terrain, 4);
        let (min, max) = terrain.bounds();
        ground_around(scene, world, ground_material, min, max, 200.0);
        let ground = |x: f32, z: f32| terrain.floor(x, z);

        // the terrain: a fine mesh to draw, a coarser one to walk on
        let cells = |step: f32| [((max[0] - min[0]) / step).ceil() as usize, ((max[1] - min[1]) / step).ceil() as usize];
        let (positions, normals, indices) = heightfield(min, max, cells(0.2), &ground);
        let mut material = Material::new("Lake/Terrain", &format!("{CASCADED_SHADOWS_WGSL}\n{TERRAIN_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
        let d = SUN_DIR;
        material.set_uniform_bindable(0, "Lake/Terrain", &[0.32f32, 0.32, 0.3, WATER, d[0], d[1], d[2], 0.0, SUN[0], SUN[1], SUN[2], 0.0, SKY[0], SKY[1], SKY[2], 0.0]);
        let mut r = Renderable::new(geometry("Lake/Terrain", &positions, &normals, indices), material);
        r.cast_shadow = false;
        scene.add(SceneNode::Renderable(r));
        let (positions, _, indices) = heightfield(min, max, cells(0.5), &ground);
        world.add(Shape::Mesh(TriangleMesh::from_indexed(&positions, &indices, Mat4::IDENTITY)), 1);

        // the container, in simulation space: walls a shore's width outside the waterline, the
        // bed as its floor
        let shape = PlanarContainerShape::from_outline(&outline, 0.1, SHORE, |x, z, _| terrain.floor(x, z)).scaled(SIM_SCALE);

        // the water: a lattice at rest density from the bed up to the still water's height
        let (lo, hi) = shape.bounds();
        let mut particles = Vec::new();
        let top = WATER * SIM_SCALE;
        let mut z = lo[1];
        while z <= hi[1] {
            let mut x = lo[0];
            while x <= hi[0] {
                let [d, floor] = shape.sample(x, z);
                if d < 0.0 {
                    let mut y = floor + SPACING * 0.5;
                    while y < top {
                        particles.extend_from_slice(&[x, y, z, 1.0]);
                        y += SPACING;
                    }
                }
                x += SPACING;
            }
            z += SPACING;
        }
        let count = (particles.len() / 4) as u32;
        log::info!("lake: {count} particles");

        // the fluid clock's tuning at 80K particles, h = 1 (see its `tuning_for`), less viscous
        let mut sim = FluidSimulation::new(renderer, FluidSimulationOptions {
            max_particles: count,
            dimensions: 3,
            smoothing_radius: 1.0,
            pressure_multiplier: 46.5,
            near_pressure_multiplier: 20.0,
            density_target: 8.6,
            // a fifth of the clock's viscosity: the water runs and splashes freely
            viscosity: 0.15,
            damping: 1.0,
            gravity: [0.0, -9.8, 0.0],
            // 4 substeps keep each one near the clock's stable 16 ms at this time scale
            substeps: 4,
            // 0.6 of the pull under the rest density: fewer filaments than the clock's full
            // cohesion. Not much less: near pressure always pushes, so sparse water with little
            // pull spreads like a gas, climbs the bank as a film and stays there (at 0.2 a
            // minute's splashing stranded 4% of the lake on the shore; at 0.6 and 1 it drains
            // back within 8 s)
            negative_pressure_scale: 0.6,
            ..DEFAULT_OPTIONS
        }, &particles);
        sim.world_bounds_min = [lo[0], (-DEPTH - 0.1) * SIM_SCALE, lo[1]];
        sim.world_bounds_max = [hi[0], 1.3 * SIM_SCALE, hi[1]];
        sim.rebuild_grid();
        let container = FluidContainer::new(&sim, shape, FluidContainerOptions { margin: 0.1, restitution: 0.05, friction: 0.002 });
        let colliders = FluidColliders::new(&sim, 16, FluidCollidersOptions { restitution: 0.3, drag: 0.15 });

        // its surface: a surface field (the distance to the weighted mean of the particles within
        // 1.5 units, less a particle radius of 0.45) polygonised at its iso level of 1, where the bulk
        // is inside whatever the mean. Averaging flattens the layers the particles settle in along
        // the sloping bed, while a lone particle stays a droplet 8 cm across and a jet of them a
        // thin one. The voxels are 0.6 units (5.5 cm): the kernel reaches 3 of them each way.
        let settings = SurfaceSettings::DROPLETS;
        let density = FluidDensityField::new(renderer, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, settings.density_options());
        let mut marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: 600_000, iso_level: 1.0 });
        // interpolated marching cubes (the default extraction draws voxel faces)
        marching_cubes.set_use_classic(true);
        let marching_cubes_bg = marching_cubes.create_bind_group(renderer, &density.density_view);
        let mut surface = FluidSurfaceEffect::new(sim, density, marching_cubes, marching_cubes_bg, FluidSurfaceOptions {
            ior: 1.33,
            chromatic_aberration: 0.02,
            tint_strength: 0.75,
            fresnel_power: 5.0,
            roughness: 0.12,
            thickness: 1.2,
            color: [0.35, 0.55, 0.6, 1.0],
            light_direction: SUN_DIR,
            light_intensity: 1.0,
            light_color: SUN,
            rim: 0.0,
            // the sky shader's colour a little above the horizon
            sky_color: [7000.0, 8000.0, 10000.0],
            sky_reflection: 1.0,
        });
        surface.splat_radius = Some(settings.kernel);

        let (bmin, bmax) = (outline.iter().fold([f32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]), outline.iter().fold([f32::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]));
        (Self { container, colliders, effect, accumulator: 0.0, previous: Vec::new(), particles: count, splash: None, near: (bmin, bmax), time_scale: TIME_SCALE, splash_push: SPLASH_PUSH, initial: particles, surface: settings }, surface)
    }

    /// The water's surface renderable, drawing the effect's marching-cubes mesh (the effect must
    /// be boxed at its final address: the geometry points at its buffers).
    pub fn add_surface(scene: &mut Scene, surface: &FluidSurfaceEffect) {
        let mut geometry = Geometry::new_indirect_placeholder("Lake/Water");
        let mc = &surface.marching_cubes;
        // SAFETY: the effect lives, boxed, in the post-processing chain for the page's life
        unsafe { geometry.set_external_buffers(mc.vertex_buffer(), mc.index_buffer(), Some(mc.indirect_args_buffer())) };
        let mut material = Material::new("Lake/Water", WATER_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, mrt_output_count: Some(4), ..Default::default() });
        material.set_uniform_bindable(0, "Lake/Water", &[0.2f32, 0.3, 0.3, 1.0]);
        let mut r = Renderable::new(geometry, material);
        r.cast_shadow = false;
        let s = 1.0 / SIM_SCALE;
        r.object.scale = kansei_core::math::Vec3::new(s, s, s);
        scene.add(SceneNode::Renderable(r));
    }

    pub fn particles(&self) -> u32 {
        self.particles
    }

    /// Place the character's leg capsules (world ends and radii), and a landing's splash (where
    /// the feet came down and how fast, m/s), and step the water by `dt` real seconds.
    pub fn update(&mut self, surface: &mut FluidSurfaceEffect, legs: &[(GVec3, GVec3, f32)], landing: Option<(GVec3, f32)>, dt: f32) {
        // legs count only near the lake
        let (lo, hi) = self.near;
        let near = legs.first().is_some_and(|(a, _, _)| a.x > lo[0] - 2.0 && a.x < hi[0] + 2.0 && a.z > lo[1] - 2.0 && a.z < hi[1] + 2.0);
        let legs: &[(GVec3, GVec3, f32)] = if near { legs } else { &[] };
        if self.previous.len() != legs.len() {
            self.previous = legs.iter().map(|(a, b, _)| [*a, *b]).collect();
        }
        let time_scale = self.time_scale;
        let velocity = |now: GVec3, then: GVec3| {
            let v = (now - then) / dt.max(1e-3);
            // a teleport (`at=`, a switch of body) is no kick
            let v = if v.length() > 15.0 { GVec3::ZERO } else { v };
            (v * SIM_SCALE / time_scale).to_array()
        };
        let mut capsules: Vec<FluidCapsule> = legs
            .iter()
            .zip(&self.previous)
            .map(|((a, b, r), [pa, pb])| FluidCapsule::new((*a * SIM_SCALE).to_array(), (*b * SIM_SCALE).to_array(), r * SIM_SCALE, velocity(*a, *pa), velocity(*b, *pb)))
            .collect();
        self.previous = legs.iter().map(|(a, b, _)| [*a, *b]).collect();
        // a landing in the water: a body-sized sphere at the feet that pushes the water out all
        // round for a moment, as fast as the fall, throwing a crown of spray
        if let Some((at, speed)) = landing {
            let inside = self.container.shape().distance(at.x * SIM_SCALE, at.z * SIM_SCALE) < 0.0;
            if inside && at.y < WATER + 0.15 && speed > 1.0 {
                self.splash = Some((at, speed.min(8.0), 0.0));
            }
        }
        if let Some((at, speed, age)) = self.splash {
            if age < SPLASH_TIME {
                let c = ((at + GVec3::Y * 0.1) * SIM_SCALE).to_array();
                let up = [0.0, speed * 0.4 * SIM_SCALE / self.time_scale, 0.0];
                capsules.push(FluidCapsule { expansion: speed * self.splash_push * SIM_SCALE / self.time_scale, ..FluidCapsule::new(c, c, SPLASH_RADIUS * SIM_SCALE, up, up) });
                self.splash = Some((at, speed, age + dt));
            } else {
                self.splash = None;
            }
        }
        self.colliders.set(&capsules);

        self.accumulator = (self.accumulator + dt).min(STEP * MAX_STEPS as f32);
        while self.accumulator >= STEP {
            surface.sim.update_batched_with(STEP * self.time_scale, 0.0, [0.0; 2], [0.0; 2], &[&self.colliders as &dyn FluidSubstepPass, &self.container]);
            self.accumulator -= STEP;
        }
    }

    /// Change a setting of the water by name (the page's tweak panel); false for an unknown one.
    pub fn set(&mut self, surface: &mut FluidSurfaceEffect, key: &str, value: f32) -> bool {
        let p = &mut surface.sim.params;
        match key {
            "viscosity" => p.viscosity = value,
            "negativePressure" => p.negative_pressure_scale = value,
            "pressure" => p.pressure_multiplier = value,
            "nearPressure" => p.near_pressure_multiplier = value,
            "restDensity" => p.density_target = value,
            "substeps" => p.substeps = value.round().clamp(1.0, 8.0) as u32,
            "timeScale" => self.time_scale = value.max(0.1),
            "drag" => self.colliders.options.drag = value,
            "splash" => self.splash_push = value,
            "friction" | "restitution" => {
                let o = &mut self.container.options;
                if key == "friction" { o.friction = value } else { o.restitution = value }
                self.container.upload();
            }
            _ => return false,
        }
        true
    }

    /// Put the water back as it started, still.
    pub fn reset(&mut self, surface: &FluidSurfaceEffect) {
        let queue = surface.sim.gpu().1;
        if let (Some(positions), Some(velocities)) = (surface.sim.positions_buffer(), surface.sim.velocities_buffer()) {
            queue.write_buffer(positions, 0, bytemuck::cast_slice(&self.initial));
            queue.write_buffer(velocities, 0, bytemuck::cast_slice(&vec![0.0f32; self.initial.len()]));
        }
        self.splash = None;
        self.accumulator = 0.0;
    }

    pub fn surface_settings(&self) -> SurfaceSettings {
        self.surface
    }

    /// Extract the surface as `settings` say: a new field when its kind or resolution changes,
    /// else the kernel, radius, iso level and interpolation in place.
    pub fn set_surface(&mut self, renderer: &Renderer, surface: &mut FluidSurfaceEffect, settings: SurfaceSettings) {
        let old = self.surface;
        if settings.surface_field != old.surface_field || settings.resolution != old.resolution {
            let sim = &surface.sim;
            let field = FluidDensityField::new(renderer, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, settings.density_options());
            surface.marching_cubes_bg = surface.marching_cubes.create_bind_group(renderer, &field.density_view);
            surface.density_field = field;
        } else {
            surface.density_field.kernel_scale = settings.density_options().kernel_scale;
            surface.density_field.set_particle_radius(settings.particle_radius);
        }
        surface.splat_radius = Some(settings.kernel);
        surface.marching_cubes.set_iso_level(settings.iso);
        surface.marching_cubes.set_use_classic(settings.interpolate);
        self.surface = settings;
    }

    pub fn drag(&self) -> f32 {
        self.colliders.options.drag
    }

    pub fn friction(&self) -> f32 {
        self.container.options.friction
    }

    /// Particles by region, from positions read back (simulation space, 4 floats each): in the
    /// lake (inside the waterline), on the bank (the shore strip's inner part), in the band
    /// against the walls (the strip's outer 0.3 m), and past the walls; with each region's mean
    /// height (m).
    pub fn regions(&self, positions: &[f32]) -> [(u32, f32); 4] {
        let shape = self.container.shape();
        let mut out = [(0u32, 0.0f32); 4];
        for p in positions.chunks_exact(4) {
            let d = shape.distance(p[0], p[2]) / SIM_SCALE;
            let k = if d < 0.0 { 0 } else if d < SHORE - 0.3 { 1 } else if d <= SHORE + 0.01 { 2 } else { 3 };
            out[k].0 += 1;
            out[k].1 += p[1] / SIM_SCALE;
        }
        out.map(|(n, y)| (n, if n > 0 { y / n as f32 } else { 0.0 }))
    }

}
