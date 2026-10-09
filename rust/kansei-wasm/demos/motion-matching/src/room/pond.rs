//! The pond in the middle of the room: the lake demo's water (`kansei_wasm_lake::lake`, the
//! engine's SPH fluid at the same scale and tuning) in a smaller outline round a stone plinth, the
//! glass dragon's island. The bed is the container's floor and a mesh in the collision world, so
//! the character wades in, its legs pushing the water through `FluidColliders`. The plinth stands
//! on the bed as one more collider, a still upright capsule the water flows round (as part of the
//! container's floor, a step that steep kept the water churning: it never settled).
//!
//! The bed is drawn into the GBuffer (wet below the waterline) and lit by the room's spot lights,
//! for the GI and the reflections; the water's surface refracts it and reflects the room
//! (`FluidSurfaceEffect`, screen space). Like the lake, the water rests when it can: culled out of
//! view, asleep once settled with no legs near it.

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Shape, TriangleMesh};
use kansei_core::geometries::{CylinderGeometry, Geometry, Vertex};
use kansei_core::gi::GiSurface;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{FluidSurfaceEffect, FluidSurfaceOptions};
use kansei_core::renderers::Renderer;
use kansei_core::rt::RtSurface;
use kansei_core::simulations::fluid::{
    lattice_density, DensityFieldOptions, FluidActivity, FluidCapsule, FluidColliders, FluidCollidersOptions, FluidContainer, FluidContainerOptions, FluidDensityField,
    FluidMarchingCubes, FluidSimulation, FluidSimulationOptions, FluidSleepOptions, FluidSolver, FluidStepper, FluidSubstepPass, MarchingCubesOptions, PbfOptions, PlanarContainerShape,
    WorldScale, DEFAULT_OPTIONS,
};

/// The lake's scale: 11 simulation units a metre, simulated time √11 times real time.
const SCALE: WorldScale = WorldScale { length: 11.0, time: 3.316_625 };
const SIM_SCALE: f32 = SCALE.length;
const STEP: f32 = 1.0 / 60.0;
const MAX_STEPS: u32 = 2;
/// The outline's half-size along x and z (round the room's centre).
const HALF: [f32; 2] = [4.2, 6.0];
/// The still water's height, a little under the floor's; the bed's depth and shelf; the shore
/// strip the container's walls stand on, and the low bank down to the floor (as the lake's).
const WATER: f32 = -0.1;
const DEPTH: f32 = 0.4;
const SHELF: f32 = 0.6;
const SHORE: f32 = 1.2;
/// The plinth: its radius, its top's height, and the stone's colour.
pub const PLINTH_RADIUS: f32 = 1.35;
pub const PLINTH_TOP: f32 = 0.5;
const STONE: [f32; 3] = [0.75, 0.75, 0.74];
/// A landing's splash, as the lake's.
const SPLASH_RADIUS: f32 = 0.3;
const SPLASH_TIME: f32 = 0.12;
const SPLASH_PUSH: f32 = 1.2;
const COLLIDERS: usize = 16;
const WAKE_DISTANCE: f32 = 2.0;
/// The fastest a particle may move (m/s) for the water to sleep: the last few that creep along the
/// bed for a minute after the fill move nothing the surface shows.
pub const SETTLE_SPEED: f32 = 0.15;
/// Lattice spacing of the particles at rest density (simulation units).
const SPACING: f32 = 0.537;

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The waterline: a long pool under the skylight, its corners rounded.
fn outline() -> Vec<[f32; 2]> {
    (0..256)
        .map(|k| {
            let a = k as f32 / 256.0 * std::f32::consts::TAU;
            // a superellipse (|x / a|^4 + |z / b|^4 = 1): a rectangle with rounded corners
            let (c, s) = (a.cos(), a.sin());
            [HALF[0] * c.signum() * c.abs().sqrt(), HALF[1] * s.signum() * s.abs().sqrt()]
        })
        .collect()
}

/// The ground's height at signed distance `d` from the waterline (negative in the pond).
fn height(_x: f32, _z: f32, d: f32) -> f32 {
    // a pool's: its floor flat, its sides a short slope, its coping flush with the room's floor
    if d < 0.0 {
        -DEPTH * smoothstep(0.0, SHELF, -d)
    } else {
        0.0
    }
}

/// How far the black stone coping reaches past the waterline (m): the terrain's edge.
const COPING: f32 = 1.6;

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
            indices.extend_from_slice(&[a, d, c, a, c, b]);
        }
    }
    (positions, normals, indices)
}

pub struct Pond {
    container: FluidContainer,
    colliders: FluidColliders,
    /// The plinth: still, neither bouncing the water nor dragging it (so it settles against it).
    plinth: FluidColliders,
    stepper: FluidStepper,
    previous: Vec<[GVec3; 2]>,
    splash: Option<(GVec3, f32, f32)>,
    /// The waterline's bounds (x, z): legs within `WAKE_DISTANCE` of them push the water.
    near: ([f32; 2], [f32; 2]),
    count: u32,
    /// The full fill, lowest particles first (a partial fill is its first part: lower water).
    fill: Vec<f32>,
    /// The water's surface renderable (hidden with the effect).
    surface: usize,
    /// Stepped at all (off: frozen, its last surface drawn); a landing's push; wake next update.
    simulate: bool,
    show: bool,
    splash_push: f32,
    poke: bool,
}

impl Pond {
    /// The rectangle (x, z min and max) the pond's terrain covers: the floor stays out of it.
    pub fn terrain_bounds() -> ([f32; 2], [f32; 2]) {
        let shape = PlanarContainerShape::from_outline(&outline(), 0.1, COPING, height);
        shape.bounds()
    }

    /// Build the pond into `scene` and `world`: its bed, the plinth, and the water's surface
    /// effect for the post-processing chain (its renderable added).
    pub fn new(renderer: &Renderer, scene: &mut Scene, world: &mut CollisionWorld, assets: &super::assets::Assets, deferred: bool) -> (Self, FluidSurfaceEffect) {
        let outline = outline();
        let mut terrain = PlanarContainerShape::from_outline(&outline, 0.1, COPING, height);
        terrain.smooth_floor(4);
        let (min, max) = terrain.bounds();
        let ground = |x: f32, z: f32| terrain.floor(x, z);

        // the bed: a fine mesh to draw (and trace), a coarser one to walk on
        let cells = |step: f32| [((max[0] - min[0]) / step).ceil() as usize, ((max[1] - min[1]) / step).ceil() as usize];
        let (positions, normals, indices) = heightfield(min, max, cells(0.15), &ground);
        let vertices = positions.iter().zip(&normals).map(|(p, n)| Vertex { position: [p.x, p.y, p.z, 1.0], normal: *n, uv: [p.x, p.z] }).collect();
        // black marble, polished, lit by the lights itself (the water refracts the image from before
        // the effects, so the bed can't leave its light to RtShadowsEffect)
        let mut bed = super::layout::surface([1.0, 1.0, 1.0], 2.0, [0.6, 0.0], true, false);
        bed.flags[0] = 1.0;
        let material = super::pbr::material("Pond/Bed", assets.surface("blackmarble", "Pond/Bed"), &bed, false);
        let mut r = Renderable::new(Geometry::new("Pond/Bed", vertices, indices), material).with_gi(GiSurface::new([0.05, 0.05, 0.05]));
        r.rt = Some(RtSurface::new([0.05, 0.05, 0.05]));
        r.cast_shadow = false;
        scene.add(SceneNode::Renderable(r));
        let (positions, _, indices) = heightfield(min, max, cells(0.4), &ground);
        world.add(Shape::Mesh(TriangleMesh::from_indexed(&positions, &indices, Mat4::IDENTITY)), 1);

        // the plinth: a stone drum from below the bed to its top, and the same in the collision
        // world (to wade round, or climb onto)
        let drum = CylinderGeometry::new(PLINTH_RADIUS, PLINTH_RADIUS, PLINTH_TOP + DEPTH, 48, 1);
        let place = Mat4::from_translation(GVec3::new(0.0, (PLINTH_TOP - DEPTH) * 0.5, 0.0));
        let corners: Vec<GVec3> = drum.vertices.iter().map(|v| GVec3::new(v.position[0], v.position[1], v.position[2])).collect();
        world.add(Shape::Mesh(TriangleMesh::from_indexed(&corners, &drum.indices, place)), 1);
        let stone = super::pbr::material("Plinth", assets.surface("marble", "Plinth"), &super::layout::surface([1.0; 3], 1.5, [1.0, 0.0], true, deferred), false);
        let mut r = Renderable::new(drum, stone).with_gi(GiSurface::new(STONE));
        r.rt = Some(RtSurface::new(STONE));
        r.object.set_position(0.0, (PLINTH_TOP - DEPTH) * 0.5, 0.0);
        scene.add(SceneNode::Renderable(r));

        // the container in simulation space: walls a shore's width outside the waterline, the bed
        // (and the plinth) its floor
        let shape = PlanarContainerShape::from_outline(&outline, 0.1, SHORE, |x, z, _| terrain.floor(x, z)).scaled(SIM_SCALE);
        let (lo, hi) = shape.bounds();
        // (none inside the plinth)
        let keep = (PLINTH_RADIUS + 0.05) * SIM_SCALE;
        let mut cells: Vec<&[f32]> = Vec::new();
        let lattice = shape.lattice(SPACING, WATER * SIM_SCALE, |d| d < 0.0);
        cells.extend(lattice.chunks_exact(4).filter(|p| p[0] * p[0] + p[2] * p[2] > keep * keep));
        // lowest first, so a partial fill (`reset`) is shallower water
        cells.sort_by(|a, b| a[1].total_cmp(&b[1]));
        let particles: Vec<f32> = cells.into_iter().flatten().copied().collect();
        let count = (particles.len() / 4) as u32;
        log::info!("pond: {count} particles");
        // the lake's tuning (see `kansei_wasm_lake::lake`)
        let mut sim = FluidSimulation::with_capacity(
            renderer,
            FluidSimulationOptions {
                max_particles: count,
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
                pbf: PbfOptions { rest_density: lattice_density(SPACING, 1.0), max_speed: SCALE.speed_to_sim(12.0), ..PbfOptions::DEFAULT },
                ..DEFAULT_OPTIONS
            },
            &particles,
            count,
        );
        sim.world_bounds_min = [lo[0], (-DEPTH - 0.1) * SIM_SCALE, lo[1]];
        sim.world_bounds_max = [hi[0], 1.3 * SIM_SCALE, hi[1]];
        sim.rebuild_grid();
        let container = FluidContainer::new(&sim, shape, FluidContainerOptions { margin: 0.1, restitution: 0.05, friction: 0.002 });
        let colliders = FluidColliders::new(&sim, COLLIDERS, FluidCollidersOptions { restitution: 0.3, drag: 0.6 });
        // an upright capsule from under the bed to over the plinth's top
        let mut plinth = FluidColliders::new(&sim, 1, FluidCollidersOptions { restitution: 0.0, drag: 0.0 });
        let r = PLINTH_RADIUS * SIM_SCALE;
        plinth.set(&[FluidCapsule::new([0.0, -(DEPTH + 0.5) * SIM_SCALE - r, 0.0], [0.0, (PLINTH_TOP + 0.3) * SIM_SCALE, 0.0], r, [0.0; 3], [0.0; 3])]);

        // its surface: the lake's droplets surface field, interpolated marching cubes
        let density = FluidDensityField::new(
            renderer,
            sim.positions_buffer().unwrap(),
            sim.world_bounds_min,
            sim.world_bounds_max,
            DensityFieldOptions { resolution: 256, kernel_scale: 1.0 / (SPACING.powi(-3) * 0.638 * 1.5f32.powi(3)), particle_radius: Some(0.45) },
        );
        let mut marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: 500_000, iso_level: 1.0 });
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
                tint_strength: 0.6,
                fresnel_power: 5.0,
                roughness: 0.08,
                thickness: 1.2,
                color: [0.3, 0.5, 0.55, 1.0],
                // the panel overhead: its light straight down, about the illuminance it gives the
                // pond (lux)
                light_direction: [0.0, -1.0, 0.0],
                light_intensity: 1.0,
                light_color: [170.0, 165.0, 155.0],
                rim: 0.0,
                // what the reflection shows where the screen has nothing: the ceiling's light
                sky_color: [6.0, 6.0, 6.2],
                sky_reflection: 0.6,
            },
        );
        surface.splat_radius = Some(1.5);
        // other materials write the GBuffer's normals too (for the GI): the fluid is where its
        // material marks the emissive alpha
        surface.mask = kansei_core::postprocessing::effects::FluidMask::EmissiveAlpha;
        let mut r = surface.surface_renderable([0.2, 0.3, 0.3, 1.0]);
        let s = 1.0 / SIM_SCALE;
        r.object.scale = kansei_core::math::Vec3::new(s, s, s);
        let surface_index = scene.add(SceneNode::Renderable(r));
        let stepper = FluidStepper::new(STEP, MAX_STEPS, SCALE).with_rest(&surface.sim, FluidSleepOptions { cull_after: 1.5, settle_speed: SETTLE_SPEED, settle_after: 1.0 });

        let (bmin, bmax) = (outline.iter().fold([f32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]), outline.iter().fold([f32::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]));
        let pond = Self { container, colliders, plinth, stepper, previous: Vec::new(), splash: None, near: (bmin, bmax), count, fill: particles, surface: surface_index, simulate: true, show: true, splash_push: SPLASH_PUSH, poke: false };
        (pond, surface)
    }

    pub fn particles(&self) -> u32 {
        self.count
    }

    /// The fastest particle (m/s) and how many move faster than the water settles at, as last
    /// read (while it runs).
    pub fn speed(&self) -> Option<(f32, u32)> {
        self.stepper.speed().map(|s| (s.max, s.above))
    }

    pub fn state(&self) -> FluidActivity {
        self.stepper.state()
    }

    /// The state for the page: `state`'s name, or frozen or hidden by the settings.
    pub fn state_name(&self) -> &'static str {
        match (self.show, self.simulate) {
            (false, _) => "hidden",
            (true, false) => "frozen",
            _ => self.state().name(),
        }
    }

    /// The page's water settings (`settings::Settings`' `fluid*`): all live but `fluid_fill`
    /// (`reset`). A change wakes the water to show it.
    pub fn apply(&mut self, scene: &mut Scene, surface: &mut FluidSurfaceEffect, s: &super::settings::Settings) {
        self.simulate = s.fluid;
        self.show = s.fluid_show;
        if let Some(r) = scene.get_renderable_mut(self.surface) {
            r.visible = s.fluid_show;
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
        // the scale's length over time squared is 1: m/s² as they are
        p.gravity = [0.0, -s.fluid_gravity, 0.0];
        p.pbf.iterations = s.fluid_pbf_iter.round().clamp(1.0, 10.0) as u32;
        p.pbf.xsph = s.fluid_xsph.clamp(0.0, 1.0);
        self.stepper.set_time_scale(sim, SCALE.time * s.fluid_speed.clamp(0.05, 4.0));
        self.stepper.set_rest_enabled(s.fluid_rest);
        self.colliders.options = FluidCollidersOptions { restitution: s.fluid_bounce.clamp(0.0, 1.0), drag: s.fluid_push.clamp(0.0, 1.0) };
        self.colliders.upload_uniform();
        self.splash_push = s.fluid_splash.max(0.0);
        surface.marching_cubes.set_use_classic(s.fluid_mesh != "voxels");
        let o = &mut surface.options;
        o.ior = s.fluid_ior;
        o.tint_strength = s.fluid_tint;
        o.roughness = s.fluid_rough;
        o.thickness = s.fluid_thickness;
        o.sky_reflection = s.fluid_reflect;
        o.chromatic_aberration = s.fluid_chromatic;
        self.poke = true;
    }

    /// Fill the pond again, still: `fraction` of the full fill (its lowest particles).
    pub fn reset(&mut self, surface: &mut FluidSurfaceEffect, fraction: f32) {
        let n = ((self.fill.len() / 4) as f32 * fraction.clamp(0.05, 1.0)) as usize;
        surface.sim.reset_particles(&self.fill[..n * 4]);
        self.count = surface.sim.particle_count();
        self.previous.clear();
        self.splash = None;
        self.poke = true;
    }

    pub fn set_rest(&mut self, rest: bool) {
        self.stepper.set_rest_enabled(rest);
    }

    /// Step the water by `dt` with the legs in it (world capsules) and a landing's splash, unless
    /// it rests (out of the view `view_proj`, or settled with nothing near).
    pub fn update(&mut self, surface: &mut FluidSurfaceEffect, legs: &[(GVec3, GVec3, f32)], landing: Option<(GVec3, f32)>, dt: f32, view_proj: Mat4) {
        let (lo, hi) = self.near;
        let w = WAKE_DISTANCE;
        let near = legs.iter().any(|(a, b, r)| {
            let (min, max) = (a.min(*b) - *r, a.max(*b) + *r);
            max.x > lo[0] - w && min.x < hi[0] + w && max.z > lo[1] - w && min.z < hi[1] + w
        });
        let legs: &[(GVec3, GVec3, f32)] = if near { legs } else { &[] };
        if self.previous.len() != legs.len() {
            self.previous = legs.iter().map(|(a, b, _)| [*a, *b]).collect();
        }
        let scale = self.stepper.scale();
        let velocity = |now: GVec3, then: GVec3| {
            let v = (now - then) / dt.max(1e-3);
            let v = if v.length() > 15.0 { GVec3::ZERO } else { v };
            scale.velocity_to_sim(v.to_array())
        };
        let mut capsules: Vec<FluidCapsule> =
            legs.iter().zip(&self.previous).map(|((a, b, r), [pa, pb])| FluidCapsule::new((*a * SIM_SCALE).to_array(), (*b * SIM_SCALE).to_array(), r * SIM_SCALE, velocity(*a, *pa), velocity(*b, *pb))).collect();
        self.previous = legs.iter().map(|(a, b, _)| [*a, *b]).collect();
        if let Some((at, speed)) = landing {
            let inside = self.container.shape().distance(at.x * SIM_SCALE, at.z * SIM_SCALE) < 0.0;
            if inside && at.y < WATER + 0.15 && speed > 1.0 {
                self.splash = Some((at, speed.min(8.0), 0.0));
            }
        }
        if let Some((at, speed, age)) = self.splash {
            if age < SPLASH_TIME {
                let c = ((at + GVec3::Y * 0.1) * SIM_SCALE).to_array();
                let up = [0.0, scale.speed_to_sim(speed * 0.4), 0.0];
                let radius = SPLASH_RADIUS * (age / SPLASH_TIME).clamp(0.2, 1.0);
                capsules.push(FluidCapsule { expansion: scale.speed_to_sim(speed * self.splash_push), ..FluidCapsule::new(c, c, radius * SIM_SCALE, up, up) });
                self.splash = Some((at, speed, age + dt));
            } else {
                self.splash = None;
            }
        }
        let disturbed = !capsules.is_empty() || std::mem::take(&mut self.poke);
        self.colliders.set(&capsules);
        let (min, max) = surface.sim.bounds(0.0);
        let in_view = kansei_core::culling::aabb_in_frustum(&kansei_core::culling::frustum_planes(view_proj), min / SIM_SCALE, max / SIM_SCALE);
        if !self.simulate || !self.show {
            // frozen: its last surface, no steps; hidden: nothing at all
            surface.set_activity(if in_view && self.show { FluidActivity::Asleep } else { FluidActivity::Culled });
            return;
        }
        let state = self.stepper.update_rest(&surface.sim, dt, in_view, disturbed);
        surface.set_activity(state);
        let steps = self.stepper.advance(dt);
        for _ in 0..steps {
            surface.sim.update_batched_with(self.stepper.step_dt(), 0.0, [0.0; 2], [0.0; 2], &[&self.colliders as &dyn FluidSubstepPass, &self.plinth, &self.container]);
        }
        self.stepper.stepped(&surface.sim, steps);
        self.count = surface.sim.particle_count();
    }
}
