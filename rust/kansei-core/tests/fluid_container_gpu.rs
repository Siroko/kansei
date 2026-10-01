//! The fluid's planar container and moving colliders on a real GPU: a block of particles
//! dropped into a round basin with a sloping floor stays inside its walls and on its floor, and a
//! capsule swept through the pool pushes the particles out of it and along with it. Skipped
//! (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidCapsule, FluidColliders, FluidCollidersOptions, FluidContainer, FluidContainerOptions, FluidSimulation,
    FluidSimulationOptions, FluidSubstepPass, PlanarContainerShape, DEFAULT_OPTIONS,
};

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig::default());
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn read_positions(sim: &FluidSimulation) -> Vec<[f32; 4]> {
    let (device, queue) = sim.gpu();
    let buffer = sim.positions_buffer().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    data
}

/// A 16-sided basin 10 across, its floor 2 deep in the middle, rising to 0 at the outline; the
/// walls half a unit outside it.
fn basin() -> PlanarContainerShape {
    let outline: Vec<[f32; 2]> = (0..16).map(|k| {
        let a = k as f32 / 16.0 * std::f32::consts::TAU;
        [5.0 * a.cos(), 5.0 * a.sin()]
    }).collect();
    PlanarContainerShape::from_outline(&outline, 0.25, 0.5, |_, _, d| (d * 0.6).clamp(-2.0, 0.0))
}

#[test]
fn particles_stay_in_the_container_and_out_of_the_colliders() {
    let Some(renderer) = renderer() else { return };
    let shape = basin();
    // a block of particles above the middle, about the fluid clock's rest density
    let mut positions = Vec::new();
    let spacing = 0.55;
    for i in 0..12 {
        for j in 0..8 {
            for k in 0..12 {
                positions.extend_from_slice(&[(i as f32 - 5.5) * spacing, 0.5 + j as f32 * spacing, (k as f32 - 5.5) * spacing, 1.0]);
            }
        }
    }
    let mut sim = FluidSimulation::new(&renderer, FluidSimulationOptions {
        max_particles: (positions.len() / 4) as u32, dimensions: 3, smoothing_radius: 1.0,
        pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6, viscosity: 1.0,
        damping: 1.0, substeps: 2, ..DEFAULT_OPTIONS
    }, &positions);
    sim.world_bounds_min = [-7.0, -3.0, -7.0];
    sim.world_bounds_max = [7.0, 8.0, 7.0];
    sim.rebuild_grid();
    let container = FluidContainer::new(&sim, shape.clone(), FluidContainerOptions::default());
    let mut colliders = FluidColliders::new(&sim, 4, FluidCollidersOptions::default());

    // settle, then sweep a vertical capsule through the pool along +x at 3 units/s
    let dt = 1.0 / 60.0;
    for _ in 0..120 {
        sim.update_batched_with(dt, 0.0, [0.0; 2], [0.0; 2], &[&colliders as &dyn FluidSubstepPass, &container]);
    }
    let before = read_positions(&sim);
    let radius = 0.6;
    let mut x = -3.0;
    let capsule = |x: f32| FluidCapsule::new([x, -3.0, 0.0], [x, 3.0, 0.0], radius, [3.0, 0.0, 0.0], [3.0, 0.0, 0.0]);
    for _ in 0..60 {
        x += 3.0 * dt;
        colliders.set(&[capsule(x)]);
        sim.update_batched_with(dt, 0.0, [0.0; 2], [0.0; 2], &[&colliders as &dyn FluidSubstepPass, &container]);
    }
    let after = read_positions(&sim);

    let margin = 0.05;
    for p in before.iter().chain(&after) {
        assert!(p.iter().all(|v| v.is_finite()), "{p:?}");
        assert!(shape.distance(p[0], p[2]) <= shape.wall_offset + 0.02, "outside the walls: {p:?}");
        assert!(p[1] >= shape.floor(p[0], p[2]) + margin - 0.05, "under the floor: {p:?} (floor {})", shape.floor(p[0], p[2]));
    }
    // nothing left inside the capsule, and the fluid ahead of it pushed forward
    for p in &after {
        let d = ((p[0] - x).powi(2) + p[2].powi(2)).sqrt();
        assert!(d >= radius - 0.02 || p[1] > 3.0 + radius, "inside the capsule: {p:?}");
    }
    let mean_x = |ps: &[[f32; 4]]| ps.iter().map(|p| p[0]).sum::<f32>() / ps.len() as f32;
    assert!(mean_x(&after) > mean_x(&before) + 0.05, "the sweep moved the fluid: {} -> {}", mean_x(&before), mean_x(&after));
}
