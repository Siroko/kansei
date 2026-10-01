//! The fluid's speed probe on a real GPU: it reads back the fastest particle's speed and how many
//! move faster than its threshold, asynchronously. Skipped (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{FluidSimulation, FluidSimulationOptions, FluidSpeed, FluidSpeedProbe, DEFAULT_OPTIONS};

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig::default());
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// The probe's measurement, polling until it lands.
fn wait(probe: &mut FluidSpeedProbe, sim: &FluidSimulation) -> Option<FluidSpeed> {
    for _ in 0..1000 {
        sim.gpu().0.poll(wgpu::Maintain::Wait);
        if let Some(s) = probe.take(sim.gpu().0) {
            return Some(s);
        }
    }
    None
}

#[test]
fn the_probe_reads_back_the_fastest_speed_and_counts_the_fast_particles() {
    let Some(renderer) = renderer() else { return };
    // 1000 particles (four workgroups, the last part full)
    let n = 1000;
    let positions: Vec<f32> = (0..n).flat_map(|i| [(i % 10) as f32, (i / 100) as f32, ((i / 10) % 10) as f32, 1.0]).collect();
    let sim = FluidSimulation::new(&renderer, FluidSimulationOptions { max_particles: n as u32, dimensions: 3, ..DEFAULT_OPTIONS }, &positions);
    // most at rest, a few slow, three fast (the fastest in the last workgroup)
    let mut velocities = vec![0.0f32; n * 4];
    for i in (0..n).step_by(7) {
        velocities[i * 4] = 0.02;
    }
    for (i, v) in [(5, [3.0, 0.0, 0.0]), (300, [0.0, -2.0, 0.0]), (999, [3.0, 4.0, 0.0])] {
        velocities[i * 4..i * 4 + 3].copy_from_slice(&v);
    }
    sim.gpu().1.write_buffer(sim.velocities_buffer().unwrap(), 0, bytemuck::cast_slice(&velocities));

    let mut probe = FluidSpeedProbe::new(&sim, 0.5);
    probe.measure(&sim);
    let s = wait(&mut probe, &sim).expect("a measurement");
    assert!((s.max - 5.0).abs() < 1e-5, "{s:?}");
    assert_eq!(s.above, 3);
    assert_eq!(probe.take(sim.gpu().0), None, "each measurement once");

    // a lower threshold counts the slow ones too; a measurement forgotten is never returned
    probe.set_threshold(sim.gpu().1, 0.01);
    probe.measure(&sim);
    probe.forget();
    assert_eq!(wait(&mut probe, &sim), None);
    probe.measure(&sim);
    let s = wait(&mut probe, &sim).expect("a measurement");
    assert_eq!(s.above, 3 + (0..n).step_by(7).filter(|i| ![5, 300, 999].contains(i)).count() as u32);

    // all at rest
    sim.gpu().1.write_buffer(sim.velocities_buffer().unwrap(), 0, bytemuck::cast_slice(&vec![0.0f32; n * 4]));
    probe.measure(&sim);
    assert_eq!(wait(&mut probe, &sim), Some(FluidSpeed { max: 0.0, above: 0 }));
}
