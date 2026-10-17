//! The aquarium: the lake demo's water (`kansei_wasm_lake::lake`, the engine's SPH fluid at the
//! same scale and tuning) in a glass tank by the west windows, the low sun through it. The tank is
//! the room glb's `AQUA_Water` box (`kansei: aquarium`): the container is that box (a rectangle
//! for walls, its bottom for a floor), filled to `fill` of its height. The glass panes round it are
//! the room's (`kansei: glass`); the character's legs still push the water through
//! `FluidColliders` were it to reach in.
//!
//! The water's surface (the marching-cubes mesh) is drawn as glass
//! (`FluidSurfaceEffect::surface_renderable_glass`): `RtReflectionsEffect`'s glass pass refracts
//! through the tank's panes and the water to what is behind them in the grid (Beer-Lambert in the
//! water) and reflects the room off it, with small ripples on its normal. The water is in the grid
//! too, as glass (`FluidHeightfield`'s coarse mesh of the surface), so the mirrors and the dragon
//! show it, and the caustics' photons refract at it (`RtCausticsEffect`, the fine height field and
//! the same ripples): the sun through the tank throws them across the floor. `fluid_render =
//! screen` draws it the old way, refracting the image on screen (`FluidSurfaceEffect`'s
//! composite). Like the lake, the water rests when it can: culled out of view, asleep once settled
//! with nothing near it.

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::geometries::{BoxGeometry, Geometry, Vertex};
use kansei_core::materials::Material;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{FluidGlassSurface, FluidSurfaceEffect, FluidSurfaceOptions};
use kansei_core::renderers::Renderer;
use kansei_core::rt::{RtCausticWater, RtSurface};
use kansei_core::simulations::fluid::{
    lattice_density, DensityFieldOptions, FluidActivity, FluidCapsule, FluidColliders, FluidCollidersOptions, FluidContainer, FluidContainerOptions, FluidDensityField,
    FluidHeightfield, FluidHeightfieldOptions, FluidMarchingCubes, FluidSimulation, FluidSimulationOptions, FluidSleepOptions, FluidSolver, FluidStepper, FluidSubstepPass, MarchingCubesOptions, PbfOptions,
    PlanarContainerShape,
    WorldScale, DEFAULT_OPTIONS,
};

/// The lake's scale: 11 simulation units a metre, simulated time √11 times real time.
const SCALE: WorldScale = WorldScale { length: 11.0, time: 3.316_625 };
const SIM_SCALE: f32 = SCALE.length;
const STEP: f32 = 1.0 / 60.0;
const MAX_STEPS: u32 = 2;
/// The tank as the room's glb places it (`layout`: `AQUA_Water`): the water's box (world metres)
/// and how full it is.
#[derive(Clone, Copy, Debug)]
pub struct Tank {
    pub min: GVec3,
    pub max: GVec3,
    pub fill: f32,
}

impl Default for Tank {
    fn default() -> Self {
        Self { min: GVec3::new(-13.0, 0.85, -2.6), max: GVec3::new(-12.0, 1.85, 1.4), fill: 0.85 }
    }
}

impl Tank {
    /// The still water's height.
    pub fn level(&self) -> f32 {
        self.min.y + (self.max.y - self.min.y) * self.fill.clamp(0.05, 1.0)
    }
}

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
/// The water's light left after a metre through it, at `fluid_tint` 1 (none at 0): deeper than
/// clear water's: the tank reads aqua against the room.
const WATER_TINT: [f32; 3] = [0.12, 0.45, 0.5];
/// Metres between the vertices of the water's mesh in the ray tracing grid, and the frames between
/// its updates while the water moves.
const GRID_SPACING: f32 = 0.3;
const GRID_EVERY: u32 = 4;
/// The surface field's voxels along the box's longest side (the tank's 4 m: 3.2 cm), the
/// marching cubes' triangles at most, a lone particle's droplet radius and the particles' splat radius (simulation units: 4.1
/// and 13.6 cm): the surface's defaults (`settings::Settings::fluid_res`, `fluid_mc_max`,
/// `fluid_drop`, `fluid_splat`).
pub const RESOLUTION: u32 = 128;
pub const MAX_TRIANGLES: u32 = 500_000;
const DROP_RADIUS: f32 = 0.45;
const SPLAT_RADIUS: f32 = 1.5;

/// The bounds' wireframe: its edges' thickness (m) and glow (cd/m²).
const EDGE: f32 = 0.025;
const EDGE_RADIANCE: [f32; 3] = [600.0, 380.0, 60.0];

/// What the water's surface is built at: rebuilt (`Aquarium::rebuild_surface`) when one changes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct SurfaceBuild {
    resolution: u32,
    max_triangles: u32,
    rt_spacing: f32,
    /// the fluid's box, world metres
    min: [f32; 3],
    max: [f32; 3],
}

/// The surface field's kernel over a lattice of particles at rest (see `DensityFieldOptions`).
fn kernel_scale() -> f32 {
    1.0 / (SPACING.powi(-3) * 0.638 * 1.5f32.powi(3))
}

/// The surface field over `sim`'s bounds, `resolution` voxels along the longest side.
fn density_field(renderer: &Renderer, sim: &FluidSimulation, resolution: u32, drop: f32) -> FluidDensityField {
    FluidDensityField::new(renderer, sim.positions_buffer().unwrap(), sim.world_bounds_min, sim.world_bounds_max, DensityFieldOptions { resolution, kernel_scale: kernel_scale(), particle_radius: Some(drop) })
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

/// The water's tint after a metre at `strength` (`fluid_tint`: 0 clear, 1 `WATER_TINT`).
fn water_tint(strength: f32) -> [f32; 3] {
    let k = strength.clamp(0.0, 1.5);
    WATER_TINT.map(|t| (1.0 - k * (1.0 - t)).clamp(0.02, 1.0))
}

/// The tank's walls: its box's rectangle (x, z), counter-clockwise.
fn outline(tank: &Tank) -> Vec<[f32; 2]> {
    let (a, b) = (tank.min, tank.max);
    vec![[a.x, a.z], [b.x, a.z], [b.x, b.z], [a.x, b.z]]
}

pub struct Aquarium {
    container: FluidContainer,
    colliders: FluidColliders,
    stepper: FluidStepper,
    previous: Vec<[GVec3; 2]>,
    splash: Option<(GVec3, f32, f32)>,
    /// The tank's bounds (x, z): legs within `WAKE_DISTANCE` of them push the water.
    near: ([f32; 2], [f32; 2]),
    count: u32,
    /// The full fill, lowest particles first (a partial fill is its first part: lower water).
    fill: Vec<f32>,
    /// The water's surface renderables: the screen-space composite's and the glass one (hidden
    /// with the effect), and its copy in the ray tracing grid.
    surface: usize,
    glass_surface: usize,
    grid_water: usize,
    /// Drawn as ray-traced glass (`fluid_render`), and the glass's look.
    rt: bool,
    glass: FluidGlassSurface,
    frames: u32,
    tank: Tank,
    /// Stepped at all (off: frozen, its last surface drawn); a landing's push; wake next update.
    simulate: bool,
    show: bool,
    splash_push: f32,
    poke: bool,
    /// What the surface is built at, and frames left to extract it whatever the water's state
    /// (after a rebuild, the new mesh is empty until it is extracted).
    build: SurfaceBuild,
    extract_frames: u32,
    /// The fluid's box drawn as a wireframe (`fluid_bounds_show`): its edges' renderables.
    edges: Vec<usize>,
}

impl Aquarium {
    /// Build the aquarium into `scene`: the water in `tank`, and its surface effect for the
    /// post-processing chain (its renderable added).
    pub fn new(renderer: &Renderer, scene: &mut Scene, tank: Tank) -> (Self, FluidSurfaceEffect) {
        let outline = outline(&tank);
        let level = tank.level();

        // the container in simulation space: the tank's rectangle its walls, its bottom the floor
        let bottom = tank.min.y;
        let shape = PlanarContainerShape::from_outline(&outline, 0.05, 0.0, |_, _, _| bottom).scaled(SIM_SCALE);
        let (lo, hi) = shape.bounds();
        let lattice = shape.lattice(SPACING, level * SIM_SCALE, |d| d < 0.0);
        let mut cells: Vec<&[f32]> = lattice.chunks_exact(4).collect();
        // lowest first, so a partial fill (`reset`) is shallower water
        cells.sort_by(|a, b| a[1].total_cmp(&b[1]));
        let particles: Vec<f32> = cells.into_iter().flatten().copied().collect();
        let count = (particles.len() / 4) as u32;
        log::info!("aquarium: {count} particles");
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
        sim.world_bounds_min = [lo[0], (tank.min.y - 0.05) * SIM_SCALE, lo[1]];
        sim.world_bounds_max = [hi[0], (tank.max.y + 0.3) * SIM_SCALE, hi[1]];
        sim.rebuild_grid();
        let container = FluidContainer::new(&sim, shape, FluidContainerOptions { margin: 0.1, restitution: 0.05, friction: 0.002 });
        let colliders = FluidColliders::new(&sim, COLLIDERS, FluidCollidersOptions { restitution: 0.3, drag: 0.6 });

        // its surface: the lake's droplets surface field, interpolated marching cubes
        let density = density_field(renderer, &sim, RESOLUTION, DROP_RADIUS);
        let mut marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: MAX_TRIANGLES, iso_level: 1.0 });
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
                // (the screen-space look's light: straight down, about the illuminance at the tank)
                light_direction: [0.0, -1.0, 0.0],
                light_intensity: 1.0,
                light_color: [170.0, 165.0, 155.0],
                rim: 0.0,
                // what the reflection shows where the screen has nothing: the ceiling's light
                sky_color: [6.0, 6.0, 6.2],
                sky_reflection: 0.6,
            },
        );
        surface.splat_radius = Some(SPLAT_RADIUS);
        // other materials write the GBuffer's normals too (for the GI): the fluid is where its
        // material marks the emissive alpha
        surface.mask = kansei_core::postprocessing::effects::FluidMask::EmissiveAlpha;
        let mut r = surface.surface_renderable([0.2, 0.3, 0.3, 1.0]);
        let s = 1.0 / SIM_SCALE;
        r.object.scale = kansei_core::math::Vec3::new(s, s, s);
        r.visible = false;
        let surface_index = scene.add(SceneNode::Renderable(r));
        // the same mesh as glass, and the surface as a height field (for the caustics) and a
        // coarse mesh in the grid
        let glass = FluidGlassSurface { tint: water_tint(0.6), ior: 1.33, roughness: 0.0, ripples: 1.0, time: 0.0 };
        let mut r = surface.surface_renderable_glass(&glass);
        r.object.scale = kansei_core::math::Vec3::new(s, s, s);
        let glass_surface = scene.add(SceneNode::Renderable(r));
        let (sim_min, sim_max) = (surface.sim.world_bounds_min, surface.sim.world_bounds_max);
        let heightfield = FluidHeightfield::new(renderer.device(), renderer.queue(), &surface.density_field, sim_min, sim_max, FluidHeightfieldOptions { iso: 1.0, world_scale: s, rt_spacing: GRID_SPACING });
        // (a triangle for the renderer to draw, out of view; the grid takes `rt_mesh`)
        let placeholder = Geometry::new("Aquarium/WaterGrid", vec![Vertex { position: [0.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0], uv: [0.0; 2] }; 3], vec![0, 1, 2]);
        let mut r = super::look::grid_only(placeholder, RtSurface::glass(glass.tint).with_approximate());
        r.rt_mesh = Some(heightfield.rt_mesh().clone());
        let grid_water = scene.add(SceneNode::Renderable(r));
        surface.heightfield = Some(heightfield);
        surface.composite = false;
        let stepper = FluidStepper::new(STEP, MAX_STEPS, SCALE).with_rest(&surface.sim, FluidSleepOptions { cull_after: 1.5, settle_speed: SETTLE_SPEED, settle_after: 1.0 });

        // the box's wireframe, hidden until asked for
        let build = SurfaceBuild { resolution: RESOLUTION, max_triangles: MAX_TRIANGLES, rt_spacing: GRID_SPACING, min: sim_min.map(|v| v * s), max: sim_max.map(|v| v * s) };
        let mut edges = Vec::new();
        for (at, size) in box_edges(GVec3::from(build.min), GVec3::from(build.max)) {
            let mut r = Renderable::new(BoxGeometry::new(1.0, 1.0, 1.0), Material::emissive("Aquarium/Bounds", EDGE_RADIANCE));
            r.object.set_position(at.x, at.y, at.z);
            r.object.scale = kansei_core::math::Vec3::new(size.x, size.y, size.z);
            r.cast_shadow = false;
            r.visible = false;
            edges.push(scene.add(SceneNode::Renderable(r)));
        }

        let (bmin, bmax) = (outline.iter().fold([f32::MAX; 2], |m, p| [m[0].min(p[0]), m[1].min(p[1])]), outline.iter().fold([f32::MIN; 2], |m, p| [m[0].max(p[0]), m[1].max(p[1])]));
        let aquarium = Self {
            container,
            colliders,
            stepper,
            previous: Vec::new(),
            splash: None,
            near: (bmin, bmax),
            count,
            fill: particles,
            surface: surface_index,
            glass_surface,
            grid_water,
            rt: true,
            glass,
            frames: 0,
            tank,
            simulate: true,
            show: true,
            splash_push: SPLASH_PUSH,
            poke: false,
            build,
            extract_frames: 0,
            edges,
        };
        (aquarium, surface)
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
        self.rt = s.fluid_render != "screen";
        for (index, shown) in [(self.surface, !self.rt), (self.glass_surface, self.rt), (self.grid_water, self.rt)] {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.visible = s.fluid_show && shown;
            }
        }
        surface.composite = !self.rt;
        self.glass = FluidGlassSurface { tint: water_tint(s.fluid_tint), ior: s.fluid_ior.max(1.0), roughness: s.fluid_rough.clamp(0.0, 1.0), ripples: s.fluid_ripples.max(0.0), time: self.glass.time };
        if let Some(r) = scene.get_renderable_mut(self.grid_water) {
            r.rt = Some(RtSurface::glass(self.glass.tint).with_approximate());
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
        // the surface's droplets and splats (the rest of it, `rebuild_surface`)
        surface.density_field.set_particle_radius(s.fluid_drop.clamp(0.005, 0.5) * SIM_SCALE);
        surface.splat_radius = Some(s.fluid_splat.clamp(0.02, 1.0) * SIM_SCALE);
        for &index in &self.edges {
            if let Some(r) = scene.get_renderable_mut(index) {
                r.visible = s.fluid_bounds_show;
            }
        }
        let o = &mut surface.options;
        o.ior = s.fluid_ior;
        // (the screen-space look's tint, as it was at its default)
        o.tint_strength = s.fluid_tint * 0.6;
        o.roughness = s.fluid_rough;
        o.thickness = s.fluid_thickness;
        o.sky_reflection = s.fluid_reflect;
        o.chromatic_aberration = s.fluid_chromatic;
        self.poke = true;
    }

    /// The fluid's box (world metres) as built.
    pub fn bounds(&self) -> ([f32; 3], [f32; 3]) {
        (self.build.min, self.build.max)
    }

    /// The surface as built, for the page (`room.info()`'s `fluid`): the field's voxels a side and
    /// their size (cm), the marching cubes' cap, the grid's mesh spacing (cm), the droplet and
    /// splat radii (cm), the box (m).
    pub fn surface_json(&self, surface: &FluidSurfaceEffect) -> String {
        let dims = surface.density_field.tex_dims();
        let (min, max) = (self.build.min, self.build.max);
        let cell: Vec<String> = (0..3).map(|k| format!("{:.2}", (max[k] - min[k]) / dims[k] as f32 * 100.0)).collect();
        format!(
            "{{\"dims\":[{},{},{}],\"cell_cm\":[{}],\"max_triangles\":{},\"rt_spacing_cm\":{:.1},\"rt_mesh\":{:?},\"drop_cm\":{:.2},\"splat_cm\":{:.2},\"min\":[{}],\"max\":[{}]}}",
            dims[0],
            dims[1],
            dims[2],
            cell.join(","),
            surface.marching_cubes.max_triangles(),
            self.build.rt_spacing * 100.0,
            surface.heightfield.as_ref().map_or([0, 0], |h| h.mesh_dims()),
            surface.density_field.particle_radius().unwrap_or(0.0) / SIM_SCALE * 100.0,
            surface.splat_radius.unwrap_or(0.0) / SIM_SCALE * 100.0,
            min.map(|v| format!("{v:.3}")).join(","),
            max.map(|v| format!("{v:.3}")).join(","),
        )
    }

    /// Bring the surface's build to the settings (`fluid_res`, `fluid_mc_max`, `fluid_rt_spacing`,
    /// the box `fluid_min_*`/`fluid_max_*`): what changed is made again (the field, the marching
    /// cubes and their renderables' meshes, the height field and the grid's copy), live; a new box
    /// moves the simulation's bounds too (its neighbour grid refitted) and fills the tank again
    /// with the water within it (pushed in, the water outside piled up against the box and blew
    /// up). False when nothing changed.
    pub fn rebuild_surface(&mut self, renderer: &Renderer, scene: &mut Scene, surface: &mut FluidSurfaceEffect, s: &super::settings::Settings) -> bool {
        let min = [s.fluid_min_x, s.fluid_min_y, s.fluid_min_z];
        let max = [s.fluid_max_x, s.fluid_max_y, s.fluid_max_z];
        // (an empty or inverted box keeps the old one)
        let (min, max) = if (0..3).all(|k| max[k] - min[k] > 0.2) { (min, max) } else { (self.build.min, self.build.max) };
        let want = SurfaceBuild {
            resolution: s.fluid_res.round().clamp(16.0, 512.0) as u32,
            max_triangles: s.fluid_mc_max.round().clamp(10_000.0, 4_000_000.0) as u32,
            rt_spacing: s.fluid_rt_spacing.clamp(0.05, 1.0),
            min,
            max,
        };
        let near = |a: [f32; 3], b: [f32; 3]| (0..3).all(|k| (a[k] - b[k]).abs() < 1e-3);
        let old = self.build;
        let moved = !near(old.min, want.min) || !near(old.max, want.max);
        if old.resolution == want.resolution && old.max_triangles == want.max_triangles && (old.rt_spacing - want.rt_spacing).abs() < 1e-4 && !moved {
            return false;
        }
        let sim = &mut surface.sim;
        if moved {
            sim.world_bounds_min = want.min.map(|v| v * SIM_SCALE);
            sim.world_bounds_max = want.max.map(|v| v * SIM_SCALE);
            sim.rebuild_grid();
            for (&index, (at, size)) in self.edges.iter().zip(box_edges(GVec3::from(want.min), GVec3::from(want.max))) {
                if let Some(r) = scene.get_renderable_mut(index) {
                    r.object.set_position(at.x, at.y, at.z);
                    r.object.scale = kansei_core::math::Vec3::new(size.x, size.y, size.z);
                }
            }
        }
        if moved {
            self.reset(surface, s.fluid_fill);
        }
        let drop = surface.density_field.particle_radius().unwrap_or(DROP_RADIUS);
        let field = density_field(renderer, &surface.sim, want.resolution, drop);
        if want.max_triangles != old.max_triangles {
            let classic = surface.marching_cubes.use_classic();
            surface.marching_cubes = FluidMarchingCubes::new(renderer, MarchingCubesOptions { max_triangles: want.max_triangles, iso_level: 1.0 });
            surface.marching_cubes.set_use_classic(classic);
            // the renderables draw the new mesh
            let mc = &surface.marching_cubes;
            for (index, label) in [(self.surface, "FluidSurface/Mesh"), (self.glass_surface, "FluidSurface/GlassMesh")] {
                if let Some(r) = scene.get_renderable_mut(index) {
                    r.geometry = Geometry::from_gpu_buffers(label, mc.vertex_buffer().clone(), mc.index_buffer().clone(), Some(mc.indirect_args_buffer().clone()));
                }
            }
        }
        surface.marching_cubes_bg = surface.marching_cubes.create_bind_group(renderer, &field.density_view);
        let heightfield = FluidHeightfield::new(renderer.device(), renderer.queue(), &field, surface.sim.world_bounds_min, surface.sim.world_bounds_max, FluidHeightfieldOptions { iso: 1.0, world_scale: 1.0 / SIM_SCALE, rt_spacing: want.rt_spacing });
        if let Some(r) = scene.get_renderable_mut(self.grid_water) {
            r.rt_mesh = Some(heightfield.rt_mesh().clone());
            r.geometry.revision = r.geometry.revision.wrapping_add(1);
        }
        surface.heightfield = Some(heightfield);
        surface.density_field = field;
        self.build = want;
        self.extract_frames = 2;
        self.poke = true;
        true
    }

    /// Fill the tank again, still: `fraction` of the full fill (its lowest particles), those
    /// within the fluid's box.
    pub fn reset(&mut self, surface: &mut FluidSurfaceEffect, fraction: f32) {
        let n = ((self.fill.len() / 4) as f32 * fraction.clamp(0.05, 1.0)) as usize;
        let (lo, hi) = (surface.sim.world_bounds_min, surface.sim.world_bounds_max);
        let inside = |p: &[f32]| (0..3).all(|k| p[k] > lo[k] + 0.1 && p[k] < hi[k] - 0.1);
        let fill: Vec<f32> = self.fill[..n * 4].chunks_exact(4).filter(|p| inside(p)).flatten().copied().collect();
        surface.sim.reset_particles(&fill);
        self.count = surface.sim.particle_count();
        self.previous.clear();
        self.splash = None;
        self.poke = true;
    }

    /// The water for the caustics (`RtCausticsEffect::set_water`), drawn as ray-traced glass and
    /// shown; and the box it fills.
    pub fn caustic_water(&self, surface: &FluidSurfaceEffect) -> Option<RtCausticWater> {
        let heightfield = surface.heightfield.as_ref().filter(|_| self.rt && self.show)?;
        let (min, size) = heightfield.rect();
        Some(RtCausticWater { heights: heightfield.height_view().clone(), min, size, dims: heightfield.height_dims(), source: self.grid_water as u32, ior: self.glass.ior, tint: self.glass.tint, ripples: self.glass.ripples, level: self.tank.level() })
    }

    /// The water's box (world metres): the caustics aim their photons at it.
    pub fn water_box(&self) -> (GVec3, GVec3) {
        (self.tank.min - 0.05, GVec3::new(self.tank.max.x, self.tank.level() + 0.15, self.tank.max.z) + 0.05)
    }

    /// The glass's uniform for this frame (the ripples' time `time`), and the grid's copy read
    /// again when the surface was extracted.
    pub fn update_surface(&mut self, scene: &mut Scene, queue: &wgpu::Queue, surface: &FluidSurfaceEffect, time: f32) {
        self.glass.time = time;
        if let Some(buffer) = scene.get_renderable(self.glass_surface).and_then(|r| r.material.bindable_buffer(0)) {
            queue.write_buffer(&buffer, 0, bytemuck::cast_slice(&self.glass.uniform()));
        }
        // (every few frames while the water moves: each time the whole grid is rebuilt; the
        // caustics' photons refract at the height field, which follows every frame)
        self.frames = self.frames.wrapping_add(1);
        if surface.extract && surface.active && self.rt && self.frames.is_multiple_of(GRID_EVERY) {
            if let Some(r) = scene.get_renderable_mut(self.grid_water) {
                r.geometry.revision = r.geometry.revision.wrapping_add(1);
            }
        }
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
            if inside && at.y < self.tank.level() + 0.15 && speed > 1.0 {
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
        // (a rebuilt surface is extracted once whatever the water does)
        let rebuilt = self.extract_frames > 0 && in_view;
        if rebuilt {
            self.extract_frames -= 1;
        }
        if !self.simulate || !self.show {
            // frozen: its last surface, no steps; hidden: nothing at all
            surface.set_activity(if in_view && self.show { FluidActivity::Asleep } else { FluidActivity::Culled });
            surface.extract |= rebuilt && self.show;
            return;
        }
        let state = self.stepper.update_rest(&surface.sim, dt, in_view, disturbed);
        surface.set_activity(state);
        surface.extract |= rebuilt;
        let steps = self.stepper.advance(dt);
        for _ in 0..steps {
            surface.sim.update_batched_with(self.stepper.step_dt(), 0.0, [0.0; 2], [0.0; 2], &[&self.colliders as &dyn FluidSubstepPass, &self.container]);
        }
        self.stepper.stepped(&surface.sim, steps);
        self.count = surface.sim.particle_count();
    }
}
