//! Position Based Fluids on a real GPU: a block of particles dropped into a basin settles (it
//! comes to rest), stays in its container, and keeps its density: the bulk's density, measured on
//! the CPU with the solver's kernel, stays within a few percent of the rest density. Skipped
//! (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidContainer, FluidContainerOptions, FluidSimulation, FluidSimulationOptions, FluidSolver, FluidSubstepPass,
    PbfOptions, PlanarContainerShape, DEFAULT_OPTIONS,
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

#[test]
fn a_pbf_basin_settles_and_keeps_its_density() {
    let Some(renderer) = renderer() else { return };
    // a square basin 5 across with a flat floor at 0, walls on its outline
    let outline = [[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]];
    let shape = PlanarContainerShape::from_outline(&outline, 0.25, 0.0, |_, _, _| 0.0);
    // a block 8 x 12 x 8 particles at the rest spacing, dropped from 1 unit up
    let spacing = 0.537f32;
    let rest = 1.0 / spacing.powi(3);
    let mut positions = Vec::new();
    for i in 0..8 {
        for j in 0..12 {
            for k in 0..8 {
                positions.extend_from_slice(&[(i as f32 - 3.5) * spacing, 1.0 + j as f32 * spacing, (k as f32 - 3.5) * spacing, 1.0]);
            }
        }
    }
    let mut sim = FluidSimulation::new(&renderer, FluidSimulationOptions {
        max_particles: (positions.len() / 4) as u32, dimensions: 3, smoothing_radius: 1.0, damping: 1.0, substeps: 2,
        solver: FluidSolver::Pbf,
        pbf: PbfOptions { iterations: 4, rest_density: rest, ..PbfOptions::DEFAULT },
        ..DEFAULT_OPTIONS
    }, &positions);
    sim.world_bounds_min = [-3.5, -1.0, -3.5];
    sim.world_bounds_max = [3.5, 10.0, 3.5];
    sim.rebuild_grid();
    let container = FluidContainer::new(&sim, shape.clone(), FluidContainerOptions { margin: 0.05, restitution: 0.0, friction: 0.0 });
    for _ in 0..360 {
        sim.update_batched_with(1.0 / 60.0, 0.0, [0.0; 2], [0.0; 2], &[&container as &dyn FluidSubstepPass]);
    }
    let p = read(&sim, sim.positions_buffer().unwrap());
    let v = read(&sim, sim.velocities_buffer().unwrap());

    // in the basin, and at rest
    for q in &p {
        assert!(q.iter().all(|c| c.is_finite()), "{q:?}");
        assert!(shape.distance(q[0], q[2]) <= 0.02 && q[1] >= 0.0, "outside: {q:?}");
    }
    let speed = v.iter().map(|v| (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()).sum::<f32>() / v.len() as f32;
    assert!(speed < 0.2, "still moving: mean speed {speed}");

    // the bulk's density (particles with a full neighbourhood: a smoothing radius from the free
    // surface and the walls), with the solver's poly6 kernel
    let h = 1.0f32;
    let poly6 = 315.0 / (64.0 * std::f32::consts::PI * h.powi(9));
    let top = p.iter().map(|q| q[1]).fold(f32::MIN, f32::max);
    let mut densities = Vec::new();
    for a in &p {
        if a[1] < 0.05 + h || a[1] > top - h || a[0].abs() > 2.5 - h || a[2].abs() > 2.5 - h {
            continue;
        }
        let rho: f32 = p.iter().map(|b| {
            let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
            if d2 < h * h { poly6 * (h * h - d2).powi(3) } else { 0.0 }
        }).sum();
        densities.push(rho / rest);
    }
    assert!(densities.len() > 20, "too few bulk particles: {}", densities.len());
    let mean = densities.iter().sum::<f32>() / densities.len() as f32;
    let max = densities.iter().cloned().fold(0.0, f32::max);
    eprintln!("bulk density / rest: mean {mean:.3}, max {max:.3}, over {} particles; mean speed {speed:.3}", densities.len());
    assert!((mean - 1.0).abs() < 0.05, "bulk density off rest: {mean}");
    assert!(max < 1.1, "compressed: {max}");
}
