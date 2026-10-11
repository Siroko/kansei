//! The box of water: the engine's fluid (SPH or Position Based Fluids, `simulations::fluid`) in a
//! white tank standing on the studio floor, which is its bed. The simulation's bounds are the
//! tank's inside (no container pass: a box is what the bounds already are), with room above the
//! walls for splashes. The water's surface (marching cubes over a surface field) is drawn as
//! ray-traced glass (`FluidSurfaceEffect::surface_renderable_glass`): `RtReflectionsEffect`'s glass
//! pass refracts the bed and the walls through it and reflects the studio off it. Its height field
//! (`FluidHeightfield`) puts a coarse mesh of it in the ray tracing grid, and the caustics' photons
//! refract at the fine heights, so the caustics on the bed follow the waves the mouse makes.
//! `fluid_render = screen` draws it the old way (`FluidSurfaceEffect`'s screen-space composite).
//!
//! The mouse pushes it (`push::FluidPush`): a column of water round the point under the cursor
//! follows the cursor's motion over the water. A glass sphere stands on the floor beside the tank,
//! for the glass's caustics beside the water's (in the water, the glass pass, which refracts
//! through one glass at a time, drew it as a white disc without the caustics' map).

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::geometries::{BoxGeometry, Geometry, SphereGeometry, Vertex};
use kansei_core::gi::GiSurface;
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{FluidGlassSurface, FluidMask, FluidSurfaceEffect, FluidSurfaceOptions};
use kansei_core::renderers::Renderer;
use kansei_core::rt::{RtCausticWater, RtSurface};
use kansei_core::simulations::fluid::{
    lattice_density, DensityFieldOptions, FluidActivity, FluidDensityField, FluidHeightfield, FluidHeightfieldOptions,
    FluidMarchingCubes, FluidSimulation, FluidSimulationOptions, FluidSleepOptions, FluidSolver, FluidStepper, FluidSubstepPass, MarchingCubesOptions, PbfOptions, WorldScale,
    DEFAULT_OPTIONS,
};

use crate::push::{FluidPush, PushGpu};
use crate::settings::Settings;
use crate::studio;

/// Lattice spacing of the particles at rest density (simulation units), the lake's tuning.
const SPACING: f32 = 0.537;
const STEP: f32 = 1.0 / 60.0;
const MAX_STEPS: u32 = 2;
/// The tank's walls: thickness (m), and the room over their tops the water may splash into.
pub const WALL: f32 = 0.05;
pub const WALL_ROUGHNESS: f32 = 0.7;
/// The bed's thickness (m): the water stands on it.
const BED: f32 = 0.004;
const HEADROOM: f32 = 0.18;
/// The water's light left after a metre at `fluid_tint` 1 (none at 0): a pool's turquoise.
const WATER_TINT: [f32; 3] = [0.12, 0.45, 0.5];
/// Frames between the grid's copies of the surface while the water moves.
const GRID_EVERY: u32 = 2;
/// The particles the simulation holds at most: the first box's fill times this (a larger box
/// later fills only to it).
const CAPACITY_SCALE: f32 = 2.5;
/// The fastest a particle may move (m/s) for the water to sleep.
pub const SETTLE_SPEED: f32 = 0.08;
/// The glass sphere: radius (m), and how far (m) beside the tank's right wall it stands, and
/// toward the camera.
const SPHERE_RADIUS: f32 = 0.2;
const SPHERE_AT: [f32; 2] = [0.35, 0.25];
/// The bounds' wireframe: its edges' thickness (m) and glow (cd/m²).
const EDGE: f32 = 0.012;
const EDGE_RADIANCE: [f32; 3] = [9000.0, 6000.0, 1200.0];

/// What the water and its surface are built at: rebuilt (`Water::rebuild`) when one changes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Build {
    resolution: u32,
    max_triangles: u32,
    rt_spacing: f32,
    /// the tank's inside (m: x, z, wall height) and the water's depth
    size: [f32; 3],
    depth: f32,
}

impl Build {
    fn from(s: &Settings) -> Self {
        Self {
            resolution: s.fluid_res.round().clamp(32.0, 512.0) as u32,
            max_triangles: s.fluid_mc_max.round().clamp(20_000.0, 4_000_000.0) as u32,
            rt_spacing: s.fluid_rt_spacing.clamp(0.03, 0.5),
            size: [s.box_w.clamp(0.4, 8.0), s.box_d.clamp(0.4, 8.0), s.box_h.clamp(0.15, 2.0)],
            depth: s.fluid_depth.clamp(0.04, s.box_h.clamp(0.15, 2.0) - 0.02),
        }
    }

    /// The simulation's bounds (world m): the tank's inside, from its bed's top to over its walls.
    fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        let (hx, hz) = (self.size[0] * 0.5, self.size[1] * 0.5);
        ([-hx, BED, -hz], [hx, self.size[2] + HEADROOM, hz])
    }
}

/// The surface field's kernel over a lattice of particles at rest (see `DensityFieldOptions`).
fn kernel_scale() -> f32 {
    1.0 / (SPACING.powi(-3) * 0.638 * 1.5f32.powi(3))
}

fn density_field(renderer: &Renderer, sim: &FluidSimulation, resolution: u32, drop: f32) -> FluidDensityField {
    FluidDensityField::new(renderer, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, DensityFieldOptions { resolution, kernel_scale: kernel_scale(), particle_radius: Some(drop) })
}

/// The water at rest in the tank to `depth` (simulation units, 4 floats a particle).
fn fill(build: &Build, scale: f32) -> Vec<f32> {
    let (lo, _) = build.bounds();
    let (lo, hi) = (GVec3::from(lo) * scale, GVec3::new(-lo[0], build.depth, -lo[2]) * scale);
    let mut particles = Vec::new();
    let half = SPACING * 0.5;
    let mut y = lo.y + half;
    while y < hi.y {
        let mut z = lo.z + half;
        while z < hi.z - half * 0.5 {
            let mut x = lo.x + half;
            while x < hi.x - half * 0.5 {
                particles.extend_from_slice(&[x, y, z, 1.0]);
                x += SPACING;
            }
            z += SPACING;
        }
        y += SPACING;
    }
    particles
}

/// The twelve edges of the box `min`-`max`: each a centre and a size.
fn box_edges(min: GVec3, max: GVec3) -> [(GVec3, GVec3); 12] {
    let (c, d) = ((min + max) * 0.5, max - min);
    let mut edges = [(GVec3::ZERO, GVec3::ZERO); 12];
    let mut k = 0;
    for axis in 0..3 {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        for (sa, sb) in [(-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
            let mut at = c;
            at[a] += d[a] * sa;
            at[b] += d[b] * sb;
            let mut size = GVec3::splat(EDGE);
            size[axis] = d[axis] + EDGE;
            edges[k] = (at, size);
            k += 1;
        }
    }
    edges
}

/// The tank's four walls (centre, size), standing on the floor round the inside `size`.
fn walls(size: [f32; 3]) -> [(GVec3, GVec3); 4] {
    let (hx, hz, h) = (size[0] * 0.5, size[1] * 0.5, size[2]);
    let (x, z) = (hx + WALL * 0.5, hz + WALL * 0.5);
    [
        (GVec3::new(-x, h * 0.5, 0.0), GVec3::new(WALL, h, size[1] + 2.0 * WALL)),
        (GVec3::new(x, h * 0.5, 0.0), GVec3::new(WALL, h, size[1] + 2.0 * WALL)),
        (GVec3::new(0.0, h * 0.5, -z), GVec3::new(size[0], h, WALL)),
        (GVec3::new(0.0, h * 0.5, z), GVec3::new(size[0], h, WALL)),
    ]
}

/// The tank's bed: a thin white plate over the floor inside the walls (centre, size).
fn bed(size: [f32; 3]) -> (GVec3, GVec3) {
    (GVec3::new(0.0, BED * 0.5, 0.0), GVec3::new(size[0], BED, size[1]))
}

/// The tank's white, lit by the scene's lights and shadows itself, with `ambient` (cd/m²) from
/// the white sky.
fn tank_material(label: &str, ambient: f32) -> Material {
    Material::standard_lit(label, &tank_options(ambient))
}

fn tank_options(ambient: f32) -> StandardLitOptions {
    StandardLitOptions { base_color: [studio::WHITE; 3], roughness: WALL_ROUGHNESS, sky_up: [ambient; 3], sky_down: [ambient * studio::WHITE; 3], outputs_velocity: true, ..Default::default() }
}

/// The water's tint after a metre at `strength` (0 clear, 1 `WATER_TINT`).
fn water_tint(strength: f32) -> [f32; 3] {
    let k = strength.clamp(0.0, 1.5);
    WATER_TINT.map(|t| (1.0 - k * (1.0 - t)).clamp(0.02, 1.0))
}

/// The glass's tint at `strength` (0 clear, 1 a pale aqua).
pub fn glass_tint(strength: f32) -> [f32; 3] {
    let k = strength.clamp(0.0, 1.0);
    [1.0 - 0.5 * k, 1.0 - 0.12 * k, 1.0 - 0.18 * k]
}

/// The mouse over the water: where the cursor's ray meets the still water (world), and the
/// velocity it moves at, eased.
#[derive(Clone, Copy, Debug)]
struct Stroke {
    at: GVec3,
    velocity: GVec3,
}

pub struct Water {
    scale: WorldScale,
    stepper: FluidStepper,
    push: FluidPush,
    stroke: Option<Stroke>,
    sphere: usize,
    sphere_shown: bool,
    count: u32,
    capacity: u32,
    /// the water's surface renderables: the screen-space composite's and the glass one, and its
    /// copy in the ray tracing grid
    surface: usize,
    glass_surface: usize,
    grid_water: usize,
    /// the walls and the bed
    walls: Vec<usize>,
    edges: Vec<usize>,
    rt: bool,
    glass: FluidGlassSurface,
    frames: u32,
    simulate: bool,
    show: bool,
    poke: bool,
    build: Build,
    /// frames left to extract the surface whatever the water's state (after a rebuild)
    extract_frames: u32,
}

impl Water {
    /// The tank, its water and the sphere into `scene`, and the water's surface effect for the
    /// post-processing chain. `scale`: simulation units a metre (the particles `0.537 / scale` m
    /// apart; time runs √`scale` times faster, so the water keeps its tuning at any size).
    pub fn new(renderer: &Renderer, scene: &mut Scene, s: &Settings, scale: f32) -> (Self, FluidSurfaceEffect) {
        let scale = WorldScale { length: scale, time: scale.sqrt() };
        let l = scale.length;
        let build = Build::from(s);
        let (lo, hi) = build.bounds();
        let sphere_at = Self::sphere_center(&build);
        let particles = fill(&build, l);
        let count = (particles.len() / 4) as u32;
        let capacity = ((count as f32 * CAPACITY_SCALE) as u32).clamp(count, 600_000);
        log::info!("caustics box: {count} particles ({:.1} cm apart), room for {capacity}", SPACING / l * 100.0);
        let mut sim = FluidSimulation::with_capacity(
            renderer,
            FluidSimulationOptions {
                max_particles: capacity,
                dimensions: 3,
                smoothing_radius: 1.0,
                pressure_multiplier: 46.5,
                near_pressure_multiplier: 20.0,
                density_target: 8.6,
                viscosity: 0.15,
                damping: 1.0,
                gravity: [0.0, -9.8, 0.0],
                substeps: 4,
                negative_pressure_scale: 0.6,
                pbf: PbfOptions { rest_density: lattice_density(SPACING, 1.0), max_speed: scale.speed_to_sim(12.0), ..PbfOptions::DEFAULT },
                ..DEFAULT_OPTIONS
            },
            &particles,
            capacity,
        );
        sim.world_bounds_min = lo.map(|v| v * l);
        sim.world_bounds_max = hi.map(|v| v * l);
        sim.rebuild_grid();
        let push = FluidPush::new(&sim);

        // its surface: droplets' surface field, interpolated marching cubes
        let density = density_field(renderer, &sim, build.resolution, s.fluid_drop * l);
        let mut marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: build.max_triangles, iso_level: 1.0 });
        marching_cubes.set_use_classic(true);
        let marching_cubes_bg = marching_cubes.create_bind_group(renderer, &density.density_view);
        let mut surface = FluidSurfaceEffect::new(
            sim,
            density,
            marching_cubes,
            marching_cubes_bg,
            FluidSurfaceOptions {
                ior: 1.33,
                chromatic_aberration: 0.02,
                tint_strength: 0.3,
                fresnel_power: 5.0,
                roughness: 0.02,
                thickness: 1.2,
                color: [0.3, 0.5, 0.55, 1.0],
                light_direction: [0.0, -1.0, 0.0],
                light_intensity: 1.0,
                light_color: [1000.0, 1000.0, 1000.0],
                rim: 0.0,
                sky_color: [1800.0; 3],
                sky_reflection: 0.6,
            },
        );
        surface.splat_radius = Some(s.fluid_splat * l);
        // the studio's materials write normals too: the fluid is where its material marks the
        // emissive alpha
        surface.mask = FluidMask::EmissiveAlpha;
        let world = kansei_core::math::Vec3::new(1.0 / l, 1.0 / l, 1.0 / l);
        let mut r = surface.surface_renderable([0.2, 0.3, 0.3, 1.0]);
        r.object.scale = world;
        r.visible = false;
        let surface_index = scene.add(SceneNode::Renderable(r));
        let glass = FluidGlassSurface { tint: water_tint(s.fluid_tint), ior: s.fluid_ior, roughness: s.fluid_rough, ripples: s.fluid_ripples, time: 0.0 };
        let mut r = surface.surface_renderable_glass(&glass);
        r.object.scale = world;
        let glass_surface = scene.add(SceneNode::Renderable(r));
        let heightfield = FluidHeightfield::new(renderer.device(), renderer.queue(), &surface.density_field, surface.sim.world_bounds_min, surface.sim.world_bounds_max, FluidHeightfieldOptions { iso: 1.0, world_scale: 1.0 / l, rt_spacing: build.rt_spacing });
        // (a triangle for the renderer to draw, out of view; the grid takes `rt_mesh`)
        let placeholder = Geometry::new("Water/Grid", vec![Vertex { position: [0.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0], uv: [0.0; 2] }; 3], vec![0, 1, 2]);
        let mut r = crate::page::grid_only(placeholder, RtSurface::glass(glass.tint).with_approximate());
        r.rt_mesh = Some(heightfield.rt_mesh().clone());
        let grid_water = scene.add(SceneNode::Renderable(r));
        surface.heightfield = Some(heightfield);
        surface.composite = false;
        let stepper = FluidStepper::new(STEP, MAX_STEPS, scale).with_rest(&surface.sim, FluidSleepOptions { cull_after: 1.5, settle_speed: SETTLE_SPEED, settle_after: 1.0 });

        // the tank's walls and its bed, white: lit by their own material (the sun through the
        // cascades), as the water's glass pass sees them in the GBuffer's background
        // (`RtSurface::with_self_lit`), where the deferred studio floor has no direct light
        let mut wall_indices = Vec::new();
        for (k, (at, size)) in walls(build.size).into_iter().chain([bed(build.size)]).enumerate() {
            let label = if k < 4 { "Tank/Wall" } else { "Tank/Bed" };
            let mut r = Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), tank_material(label, 0.0)).with_gi(GiSurface::new([studio::WHITE; 3]));
            r.rt = Some(RtSurface::new([studio::WHITE; 3]).with_self_lit());
            r.object.set_position(at.x, at.y, at.z);
            r.object.scale = kansei_core::math::Vec3::new(size.x, size.y, size.z);
            wall_indices.push(scene.add(SceneNode::Renderable(r)));
        }
        // the glass sphere beside the tank
        let mut r = Renderable::new(SphereGeometry::new(SPHERE_RADIUS, 64, 32), Material::standard_lit("Tank/Sphere", &StandardLitOptions::glass(glass_tint(s.glass_tint), s.glass_ior, s.glass_rough)));
        r.rt = Some(RtSurface::glass(glass_tint(s.glass_tint)).with_smooth_normals());
        r.cast_shadow = false;
        r.object.set_position(sphere_at.x, sphere_at.y, sphere_at.z);
        let sphere = scene.add(SceneNode::Renderable(r));
        // the bounds' wireframe, hidden until asked for
        let mut edges = Vec::new();
        for _ in 0..12 {
            let mut r = Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), Material::emissive("Water/Bounds", EDGE_RADIANCE));
            r.cast_shadow = false;
            r.visible = false;
            edges.push(scene.add(SceneNode::Renderable(r)));
        }

        let mut water = Self {
            scale,
            stepper,
            push,
            stroke: None,
            sphere,
            sphere_shown: s.sphere,
            count,
            capacity,
            surface: surface_index,
            glass_surface,
            grid_water,
            walls: wall_indices,
            edges,
            rt: true,
            glass,
            frames: 0,
            simulate: true,
            show: true,
            poke: false,
            build,
            extract_frames: 2,
        };
        water.place(scene);
        (water, surface)
    }

    /// Where the sphere stands beside a tank built at `build`.
    fn sphere_center(build: &Build) -> GVec3 {
        GVec3::new(build.size[0] * 0.5 + WALL + SPHERE_AT[0], SPHERE_RADIUS, SPHERE_AT[1])
    }

    /// The walls, the sphere and the wireframe to the build.
    fn place(&mut self, scene: &mut Scene) {
        for (&index, (at, size)) in self.walls.iter().zip(walls(self.build.size).into_iter().chain([bed(self.build.size)])) {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.object.set_position(at.x, at.y, at.z);
                r.object.scale = kansei_core::math::Vec3::new(size.x, size.y, size.z);
            }
        }
        let c = Self::sphere_center(&self.build);
        if let Some(r) = scene.get_renderable_mut(self.sphere) {
            r.object.set_position(c.x, c.y, c.z);
            r.visible = self.sphere_shown;
        }
        let (lo, hi) = self.build.bounds();
        for (&index, (at, size)) in self.edges.iter().zip(box_edges(GVec3::from(lo), GVec3::from(hi))) {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.object.set_position(at.x, at.y, at.z);
                r.object.scale = kansei_core::math::Vec3::new(size.x, size.y, size.z);
            }
        }
    }

    /// The flat light the tank's white adds itself (cd/m²; none while the RT GI lights it).
    pub fn set_ambient(&self, renderer: &Renderer, scene: &mut Scene, ambient: f32) {
        for &index in &self.walls {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.material.set_standard_lit(renderer, &tank_options(ambient));
            }
        }
    }

    pub fn particles(&self) -> u32 {
        self.count
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Particles apart (cm).
    pub fn spacing_cm(&self) -> f32 {
        SPACING / self.scale.length * 100.0
    }

    pub fn speed(&self) -> Option<(f32, u32)> {
        self.stepper.speed().map(|s| (s.max, s.above))
    }

    /// The state for the page: frozen or hidden by the settings, else the stepper's.
    pub fn state_name(&self) -> &'static str {
        match (self.show, self.simulate) {
            (false, _) => "hidden",
            (true, false) => "frozen",
            _ => self.stepper.state().name(),
        }
    }

    /// The tank's inside and the water's still level (world m).
    pub fn size(&self) -> [f32; 3] {
        self.build.size
    }

    pub fn level(&self) -> f32 {
        self.build.depth
    }

    pub fn sphere_box(&self) -> Option<(GVec3, GVec3)> {
        let c = Self::sphere_center(&self.build);
        self.sphere_shown.then(|| (c - GVec3::splat(SPHERE_RADIUS), c + GVec3::splat(SPHERE_RADIUS)))
    }

    /// The box the water fills (for the caustics' photons): the tank's inside to a little over
    /// the still level.
    pub fn water_box(&self) -> (GVec3, GVec3) {
        let [w, d, _] = self.build.size;
        (GVec3::new(-w * 0.5, 0.0, -d * 0.5), GVec3::new(w * 0.5, self.build.depth + 0.12, d * 0.5))
    }

    /// The live settings (all but the build's, `rebuild`): a change wakes the water to show it.
    pub fn apply(&mut self, renderer: &Renderer, scene: &mut Scene, surface: &mut FluidSurfaceEffect, s: &Settings) {
        self.simulate = s.fluid;
        self.show = s.fluid_show;
        self.rt = s.fluid_render != "screen";
        for (index, shown) in [(self.surface, !self.rt), (self.glass_surface, self.rt), (self.grid_water, self.rt)] {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.visible = s.fluid_show && shown;
            }
        }
        surface.composite = !self.rt;
        self.glass = FluidGlassSurface { tint: water_tint(s.fluid_tint), ior: s.fluid_ior.max(1.0), roughness: s.fluid_rough.clamp(0.0, 1.0), ripples: s.fluid_ripples.max(0.0), time: self.glass.time };
        if let Some(r) = scene.get_renderable_mut(self.grid_water) {
            let rt = Some(RtSurface::glass(self.glass.tint).with_approximate());
            if r.rt != rt {
                r.rt = rt;
            }
        }
        // the sphere's glass
        let tint = glass_tint(s.glass_tint);
        if let Some(r) = scene.get_renderable_mut(self.sphere) {
            r.material.set_standard_lit(renderer, &StandardLitOptions::glass(tint, s.glass_ior, s.glass_rough));
            let rt = Some(RtSurface::glass(tint).with_smooth_normals());
            if r.rt != rt {
                r.rt = rt;
            }
        }
        if self.sphere_shown != s.sphere {
            self.sphere_shown = s.sphere;
            self.place(scene);
        }
        let sim = &mut surface.sim;
        let p = &mut sim.params;
        p.solver = if s.fluid_solver == "pbf" { FluidSolver::Pbf } else { FluidSolver::Sph };
        p.substeps = s.fluid_substeps.round().clamp(1.0, 12.0) as u32;
        p.viscosity = s.fluid_viscosity.max(0.0);
        p.pressure_multiplier = s.fluid_pressure.max(0.0);
        p.near_pressure_multiplier = s.fluid_near.max(0.0);
        p.negative_pressure_scale = s.fluid_cohesion.clamp(0.0, 1.0);
        p.damping = s.fluid_damping.clamp(0.0, 1.0);
        // (the scale's length over time squared is 1: m/s² as they are)
        p.gravity = [0.0, -s.fluid_gravity, 0.0];
        p.pbf.iterations = s.fluid_pbf_iter.round().clamp(1.0, 10.0) as u32;
        p.pbf.xsph = s.fluid_xsph.clamp(0.0, 1.0);
        self.stepper.set_time_scale(sim, self.scale.time * s.fluid_speed.clamp(0.05, 4.0));
        self.stepper.set_rest_enabled(s.fluid_rest);
        let l = self.scale.length;
        surface.density_field.set_particle_radius(s.fluid_drop.clamp(0.005, 0.3) * l);
        surface.splat_radius = Some(s.fluid_splat.clamp(0.01, 0.5) * l);
        for &index in &self.edges {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.visible = s.fluid_bounds_show;
            }
        }
        let o = &mut surface.options;
        o.ior = s.fluid_ior;
        o.tint_strength = s.fluid_tint * 0.6;
        o.roughness = s.fluid_rough;
        o.sky_color = [s.sky_radiance; 3];
        self.poke = true;
    }

    /// Bring the build to the settings (the field's resolution, the marching cubes' cap, the
    /// grid's mesh spacing, the tank and the water's depth): what changed is made again, live; a
    /// new tank or depth fills it again. False when nothing changed.
    pub fn rebuild(&mut self, renderer: &Renderer, scene: &mut Scene, surface: &mut FluidSurfaceEffect, s: &Settings) -> bool {
        let want = Build::from(s);
        let old = self.build;
        if want == old {
            return false;
        }
        let l = self.scale.length;
        let refill = want.size != old.size || (want.depth - old.depth).abs() > 1e-4;
        self.build = want;
        if want.size != old.size {
            let (lo, hi) = want.bounds();
            surface.sim.world_bounds_min = lo.map(|v| v * l);
            surface.sim.world_bounds_max = hi.map(|v| v * l);
            surface.sim.rebuild_grid();
            self.place(scene);
        }
        if refill {
            self.reset(surface);
        }
        let drop = surface.density_field.particle_radius().unwrap_or(0.03 * l);
        let field = density_field(renderer, &surface.sim, want.resolution, drop);
        if want.max_triangles != old.max_triangles {
            let classic = surface.marching_cubes.use_classic();
            surface.marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: want.max_triangles, iso_level: 1.0 });
            surface.marching_cubes.set_use_classic(classic);
            let mc = &surface.marching_cubes;
            for (index, label) in [(self.surface, "FluidSurface/Mesh"), (self.glass_surface, "FluidSurface/GlassMesh")] {
                if let Some(r) = scene.get_renderable_mut(index) {
                    r.geometry = Geometry::from_gpu_buffers(label, mc.vertex_buffer().clone(), mc.index_buffer().clone(), Some(mc.indirect_args_buffer().clone()));
                }
            }
        }
        surface.marching_cubes_bg = surface.marching_cubes.create_bind_group(renderer, &field.density_view);
        let heightfield = FluidHeightfield::new(renderer.device(), renderer.queue(), &field, surface.sim.world_bounds_min, surface.sim.world_bounds_max, FluidHeightfieldOptions { iso: 1.0, world_scale: 1.0 / l, rt_spacing: want.rt_spacing });
        if let Some(r) = scene.get_renderable_mut(self.grid_water) {
            r.rt_mesh = Some(heightfield.rt_mesh().clone());
            r.geometry.revision = r.geometry.revision.wrapping_add(1);
        }
        surface.heightfield = Some(heightfield);
        surface.density_field = field;
        self.extract_frames = 2;
        self.poke = true;
        true
    }

    /// Fill the tank again, still.
    pub fn reset(&mut self, surface: &mut FluidSurfaceEffect) {
        let mut particles = fill(&self.build, self.scale.length);
        particles.truncate(self.capacity as usize * 4);
        surface.sim.reset_particles(&particles);
        self.count = surface.sim.particle_count();
        self.stroke = None;
        self.poke = true;
        self.stepper.wake();
    }

    /// The water for the caustics (`RtCausticsEffect::set_water`), when drawn as ray-traced glass:
    /// its photons refract at the waves alone (the wind ripples' fine focus is the lines',
    /// `RtCausticsEffect::water_lines`).
    pub fn caustic_water(&self, surface: &FluidSurfaceEffect) -> Option<RtCausticWater> {
        let heightfield = surface.heightfield.as_ref().filter(|_| self.rt && self.show)?;
        let (min, size) = heightfield.rect();
        Some(RtCausticWater { heights: heightfield.height_view().clone(), min, size, dims: heightfield.height_dims(), source: self.grid_water as u32, ior: self.glass.ior, tint: self.glass.tint, ripples: 0.0, level: self.build.depth })
    }

    /// The glass's uniform for this frame (the ripples' time), and the grid's copy read again
    /// every few frames while the surface is extracted.
    pub fn update_surface(&mut self, scene: &mut Scene, queue: &wgpu::Queue, surface: &FluidSurfaceEffect, time: f32) {
        self.glass.time = time;
        if let Some(buffer) = scene.get_renderable(self.glass_surface).and_then(|r| r.material.bindable_buffer(0)) {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.glass.uniform()));
        }
        self.frames = self.frames.wrapping_add(1);
        if surface.extract && surface.active && self.rt && self.frames.is_multiple_of(GRID_EVERY) {
            if let Some(r) = scene.get_renderable_mut(self.grid_water) {
                r.geometry.revision = r.geometry.revision.wrapping_add(1);
            }
        }
    }

    /// The mouse over the water this frame: the cursor's ray (world origin and direction) while
    /// the push button is held, or none. Returns the point it pushes at.
    pub fn stroke(&mut self, ray: Option<(GVec3, GVec3)>, dt: f32, s: &Settings) -> Option<GVec3> {
        let level = self.build.depth;
        let [w, d, _] = self.build.size;
        let hit = ray.and_then(|(o, dir)| {
            if dir.y.abs() < 1e-4 {
                return None;
            }
            let t = (level - o.y) / dir.y;
            (t > 0.0).then(|| o + dir * t)
        });
        let Some(hit) = hit else {
            self.stroke = None;
            return None;
        };
        let margin = s.push_radius * 0.3;
        let at = GVec3::new(hit.x.clamp(-w * 0.5 + margin, w * 0.5 - margin), level, hit.z.clamp(-d * 0.5 + margin, d * 0.5 - margin));
        let max = s.push_max_speed.max(0.1);
        self.stroke = Some(match self.stroke {
            None => Stroke { at, velocity: GVec3::ZERO },
            Some(prev) => {
                let v = ((at - prev.at) / dt.max(1e-3)).clamp_length_max(max);
                // eased over about 50 ms: the cursor's motion between frames is jittery
                let k = 1.0 - (-dt / 0.05).exp();
                Stroke { at, velocity: prev.velocity + (v - prev.velocity) * k }
            }
        });
        Some(at)
    }

    /// Step the water by `dt` (with the mouse's push), unless it rests (out of the view
    /// `view_proj`, or settled with nothing pushing it).
    pub fn update(&mut self, surface: &mut FluidSurfaceEffect, dt: f32, view_proj: Mat4, s: &Settings) {
        let l = self.scale.length;
        let push = self.stroke.map(|st| PushGpu {
            center: [st.at.x * l, st.at.z * l],
            bottom: -1.0,
            top: (self.build.size[2] + HEADROOM) * l,
            velocity: self.scale.velocity_to_sim(st.velocity.to_array()),
            radius: s.push_radius.max(0.02) * l,
            strength: s.push_strength.clamp(0.0, 1.0),
            ..Default::default()
        });
        self.push.set(push.filter(|p| p.strength > 0.0));
        let disturbed = self.push.active() || std::mem::take(&mut self.poke);
        let (min, max) = surface.sim.bounds(0.0);
        let in_view = kansei_core::culling::aabb_in_frustum(&kansei_core::culling::frustum_planes(view_proj), min / l, max / l);
        let rebuilt = self.extract_frames > 0 && in_view;
        if rebuilt {
            self.extract_frames -= 1;
        }
        if !self.simulate || !self.show {
            surface.set_activity(if in_view && self.show { FluidActivity::Asleep } else { FluidActivity::Culled });
            surface.extract |= rebuilt && self.show;
            return;
        }
        let state = self.stepper.update_rest(&surface.sim, dt, in_view, disturbed);
        surface.set_activity(state);
        surface.extract |= rebuilt;
        let steps = self.stepper.advance(dt);
        for _ in 0..steps {
            surface.sim.update_batched_with(self.stepper.step_dt(), 0.0, [0.0; 2], [0.0; 2], &[&self.push as &dyn FluidSubstepPass]);
        }
        self.stepper.stepped(&surface.sim, steps);
        self.count = surface.sim.particle_count();
    }

    /// The water's surface as built, for the page (`info().fluid`): the field's voxels and their
    /// size (cm), the marching cubes' cap, the grid's mesh, the droplet and splat radii (cm).
    pub fn surface_json(&self, surface: &FluidSurfaceEffect) -> String {
        let dims = surface.density_field.tex_dims();
        let (lo, hi) = self.build.bounds();
        let cell: Vec<String> = (0..3).map(|k| format!("{:.2}", (hi[k] - lo[k]) / dims[k] as f32 * 100.0)).collect();
        let l = self.scale.length;
        format!(
            "{{\"dims\":[{},{},{}],\"cell_cm\":[{}],\"max_triangles\":{},\"rt_spacing_cm\":{:.1},\"rt_mesh\":{:?},\"drop_cm\":{:.2},\"splat_cm\":{:.2},\"spacing_cm\":{:.2},\"particles\":{},\"capacity\":{}}}",
            dims[0],
            dims[1],
            dims[2],
            cell.join(","),
            surface.marching_cubes.max_triangles(),
            self.build.rt_spacing * 100.0,
            surface.heightfield.as_ref().map_or([0, 0], |h| h.mesh_dims()),
            surface.density_field.particle_radius().unwrap_or(0.0) / l * 100.0,
            surface.splat_radius.unwrap_or(0.0) / l * 100.0,
            self.spacing_cm(),
            self.count,
            self.capacity,
        )
    }
}
