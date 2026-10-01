//! Adding fluid at runtime on a real GPU: a simulation with spare capacity takes particles from
//! a nozzle (`FluidSimulation::emit`), and they join it. With either solver (SPH, PBF), poured
//! into a basin of resting water, the new particles end up in the pool among the old ones (the
//! neighbour search sees them: none overlaps another), the pool's surface rises, the spare slots
//! past the live count are never touched (no pass runs on them), the emission stops at the
//! capacity, and a reset puts the initial fill back. The speed probe counts the live particles.
//! Skipped (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidContainer, FluidContainerOptions, FluidNozzle, FluidSimulation, FluidSimulationOptions, FluidSolver,
    FluidSpeedProbe, FluidSubstepPass, PbfOptions, PlanarContainerShape, DEFAULT_OPTIONS,
};

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig::default());
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn read(sim: &FluidSimulation, buffer: &wgpu::Buffer) -> Vec<[f32; 4]> {
    let (device, queue) = sim.gpu();
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    data
}

const SPACING: f32 = 0.537;

/// A square basin 6 across with a flat floor at 0, walls on its outline, and a layer of water 4
/// particles deep in it.
fn basin() -> (PlanarContainerShape, Vec<f32>) {
    let outline = [[-3.0, -3.0], [3.0, -3.0], [3.0, 3.0], [-3.0, 3.0]];
    let shape = PlanarContainerShape::from_outline(&outline, 0.25, 0.0, |_, _, _| 0.0);
    let mut positions = Vec::new();
    for i in 0..10 {
        for j in 0..4 {
            for k in 0..10 {
                positions.extend_from_slice(&[(i as f32 - 4.5) * SPACING, 0.3 + j as f32 * SPACING, (k as f32 - 4.5) * SPACING, 1.0]);
            }
        }
    }
    (shape, positions)
}

fn simulation(renderer: &Renderer, solver: FluidSolver, positions: &[f32], capacity: u32) -> FluidSimulation {
    let mut sim = FluidSimulation::with_capacity(renderer, FluidSimulationOptions {
        max_particles: capacity, dimensions: 3, smoothing_radius: 1.0,
        pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6, viscosity: 1.0,
        damping: 1.0, substeps: if solver == FluidSolver::Pbf { 2 } else { 4 },
        solver,
        pbf: PbfOptions { iterations: 4, rest_density: 1.0 / SPACING.powi(3), ..PbfOptions::DEFAULT },
        ..DEFAULT_OPTIONS
    }, positions, capacity);
    sim.world_bounds_min = [-4.0, -1.0, -4.0];
    sim.world_bounds_max = [4.0, 12.0, 4.0];
    sim.rebuild_grid();
    sim
}

fn poured_water_joins_the_pool(solver: FluidSolver) {
    let Some(renderer) = renderer() else { return };
    let (shape, initial) = basin();
    let n0 = (initial.len() / 4) as u32;
    let capacity = n0 + 1500;
    let mut sim = simulation(&renderer, solver, &initial, capacity);
    assert_eq!((sim.particle_count(), sim.capacity()), (n0, capacity));
    let container = FluidContainer::new(&sim, shape.clone(), FluidContainerOptions { margin: 0.05, restitution: 0.0, friction: 0.0 });
    let step = |sim: &mut FluidSimulation| sim.update_batched_with(1.0 / 60.0, 0.0, [0.0; 2], [0.0; 2], &[&container as &dyn FluidSubstepPass]);

    // a stream from above one corner, slanting down into the pool, until 1000 have gone in
    let mut nozzle = FluidNozzle::new([-1.5, 5.0, -1.5], [0.5, -1.0, 0.5], 0.8, 6.0, SPACING);
    let poured = 1000;
    while sim.particle_count() < n0 + poured {
        let room = n0 + poured - sim.particle_count();
        let (p, v) = nozzle.flow(1.0 / 60.0);
        let take = (p.len() as u32).min(room) as usize;
        sim.emit(&p[..take], &v[..take]);
        step(&mut sim);
    }
    assert_eq!(sim.particle_count(), n0 + poured);
    for _ in 0..360 {
        step(&mut sim);
    }

    let p = read(&sim, sim.positions_buffer().unwrap());
    let live = &p[..sim.particle_count() as usize];
    // the spare slots were never stepped: still zeros (the container would have lifted them to
    // its floor's margin)
    assert!(p[sim.particle_count() as usize..].iter().all(|q| *q == [0.0; 4]), "a pass ran past the live count");
    // every live particle in the basin, none overlapping another (the new ones are in the
    // neighbour grid: the old ones push them apart)
    for q in live {
        assert!(q.iter().all(|c| c.is_finite()) && shape.distance(q[0], q[2]) <= 0.02 && q[1] >= 0.0, "outside: {q:?}");
    }
    let mut closest = f32::MAX;
    for (a, pa) in live.iter().enumerate().skip(n0 as usize) {
        for (b, pb) in live.iter().enumerate() {
            if a != b {
                closest = closest.min(((pa[0] - pb[0]).powi(2) + (pa[1] - pb[1]).powi(2) + (pa[2] - pb[2]).powi(2)).sqrt());
            }
        }
    }
    // the poured particles settled into the pool (not left in the air)
    let top = |ps: &[[f32; 4]]| {
        let mut y: Vec<f32> = ps.iter().map(|q| q[1]).collect();
        y.sort_by(f32::total_cmp);
        y[y.len() * 95 / 100]
    };
    let new_top = top(live);
    let highest_new = live[n0 as usize..].iter().map(|q| q[1]).fold(f32::MIN, f32::max);
    let initial_top = top(bytemuck::cast_slice(&initial));
    eprintln!("{solver:?}: closest pair {closest:.3}, surface {initial_top:.2} -> {new_top:.2}, highest poured {highest_new:.2}");
    assert!(closest > 0.25 * SPACING, "particles overlap: {closest}");
    // 1000 particles over a 6 × 6 floor at the rest spacing: about 4 layers more (up to 2.1)
    assert!(new_top > initial_top + 1.0, "the surface did not rise: {initial_top} -> {new_top}");
    assert!(highest_new < new_top + 1.5, "poured water left above the pool: {highest_new}");

    // the stream stops at the capacity
    let mut added = 0;
    for _ in 0..200 {
        added += nozzle.emit_into(&mut sim, 0.1);
    }
    assert_eq!(added, capacity - n0 - poured);
    assert_eq!(sim.particle_count(), capacity);
    assert_eq!(nozzle.emit_into(&mut sim, 1.0), 0);

    // a reset: the initial fill, at rest
    sim.reset_particles(&initial);
    assert_eq!(sim.particle_count(), n0);
    let p = read(&sim, sim.positions_buffer().unwrap());
    assert_eq!(bytemuck::cast_slice::<[f32; 4], f32>(&p[..n0 as usize]), initial.as_slice());
    step(&mut sim);
    assert_eq!(sim.particle_count(), n0);
}

#[test]
fn poured_water_joins_an_sph_pool() {
    poured_water_joins_the_pool(FluidSolver::Sph);
}

#[test]
fn poured_water_joins_a_pbf_pool() {
    poured_water_joins_the_pool(FluidSolver::Pbf);
}

#[test]
fn the_speed_probe_counts_the_emitted_particles() {
    let Some(renderer) = renderer() else { return };
    let (_, initial) = basin();
    let n0 = (initial.len() / 4) as u32;
    let mut sim = simulation(&renderer, FluidSolver::Sph, &initial, n0 + 100);
    let mut probe = FluidSpeedProbe::new(&sim, 1.0);
    // the old particles at rest, 64 new ones at 5 units/s
    let positions: Vec<[f32; 3]> = (0..64).map(|k| [(k % 8) as f32 * SPACING - 2.0, 4.0, (k / 8) as f32 * SPACING - 2.0]).collect();
    assert_eq!(sim.emit(&positions, &[[3.0, 0.0, 4.0]]), 64);
    probe.measure(&sim);
    let speed = loop {
        if let Some(s) = probe.take(sim.gpu().0) {
            break s;
        }
    };
    assert_eq!(speed.above, 64);
    assert!((speed.max - 5.0).abs() < 1e-4, "{}", speed.max);
}
