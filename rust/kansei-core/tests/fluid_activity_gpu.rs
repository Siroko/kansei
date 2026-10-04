//! The fluid's speed probe on a real GPU: it reads back the fastest particle's speed and how many
//! move faster than its threshold, asynchronously; and a `FluidStepper` resting by it, at a world
//! scale. Skipped (passes) when no adapter is available.

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    FluidActivity, FluidSimulation, FluidSimulationOptions, FluidSleepOptions, FluidSpeed, FluidSpeedProbe, FluidStepper, WorldScale, DEFAULT_OPTIONS,
};

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

#[test]
fn a_stepper_sleeps_once_settled_and_culls_out_of_view_reading_speeds_in_metres() {
    let Some(renderer) = renderer() else { return };
    let n = 256;
    let positions: Vec<f32> = (0..n).flat_map(|i| [(i % 8) as f32, (i / 64) as f32, ((i / 8) % 8) as f32, 1.0]).collect();
    let sim = FluidSimulation::new(&renderer, FluidSimulationOptions { max_particles: n as u32, dimensions: 3, ..DEFAULT_OPTIONS }, &positions);
    let scale = WorldScale::with_real_gravity(11.0);
    let options = FluidSleepOptions { cull_after: 0.5, settle_speed: 0.05, settle_after: 0.1 };
    let mut stepper = FluidStepper::new(1.0 / 60.0, 2, scale).with_rest(&sim, options);
    assert!((stepper.step_dt() - scale.time / 60.0).abs() < 1e-6);

    // one particle at 1 unit per simulated second: 0.3 m/s in the world, over the settle speed
    let mut velocities = vec![0.0f32; n * 4];
    velocities[0] = 1.0;
    sim.gpu().1.write_buffer(sim.velocities_buffer().unwrap(), 0, bytemuck::cast_slice(&velocities));
    let frame = |stepper: &mut FluidStepper, in_view: bool| {
        sim.gpu().0.poll(wgpu::Maintain::Wait);
        let state = stepper.update_rest(&sim, 1.0 / 60.0, in_view, false);
        let steps = stepper.advance(1.0 / 60.0);
        stepper.stepped(&sim, steps);
        (state, steps)
    };
    for _ in 0..30 {
        assert_eq!(frame(&mut stepper, true), (FluidActivity::Running, 1));
    }
    let speed = stepper.speed().expect("a speed read");
    assert!((speed.max - scale.speed_to_world(1.0)).abs() < 1e-5 && speed.above == 1, "{speed:?}");

    // at rest it falls asleep and stops stepping; a wake runs it again
    sim.gpu().1.write_buffer(sim.velocities_buffer().unwrap(), 0, bytemuck::cast_slice(&vec![0.0f32; n * 4]));
    stepper.wake();
    let asleep = (0..120).position(|_| frame(&mut stepper, true) == (FluidActivity::Asleep, 0)).expect("asleep");
    assert!(asleep >= 6, "slept after {asleep} frames, before settle_after");
    stepper.wake();
    assert_eq!(frame(&mut stepper, true).0, FluidActivity::Running);

    // out of view for cull_after: culled
    let culled = (0..120).position(|_| frame(&mut stepper, false) == (FluidActivity::Culled, 0)).expect("culled");
    assert!((28..=31).contains(&culled), "culled after {culled} frames");
}
