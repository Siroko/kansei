//! Native readback verification for the glyph attractor.
//!
//! Spawns a small cloud of particles offset from glyph slot 0's box, runs
//! the SPH solver + glyph attractor compute pass for ~120 steps, then reads
//! the positions buffer back to the CPU and asserts the cloud's mean
//! distance to the slot's box center has shrunk substantially — i.e. the
//! attractor is actually pulling particles in. (Checking the *velocity*
//! direction instead was fragile: once the pull is strong enough to arrive
//! within the step budget, particles slosh around the glyph.)
//!
//! Env knobs: COUNT (64), STEPS (300), MAXSPEED (10). Keep MAXSPEED small
//! relative to the 4-unit slot box: at 20 u/s a particle crosses the whole
//! stroke in a step or two and never settles. The count is sized to what the
//! box holds; a bigger cloud just overflows it (see glyph_form_test).
//!
//! This is the real GPU execution gate for `GlyphAttractor` (Task 5): the
//! WGSL compute shader was never actually dispatched before this example.

use std::sync::Arc;

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{
    FluidSimulation, FluidSimulationOptions, GlyphAttractor, SlotLayout, DEFAULT_OPTIONS,
};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const FONT: &[u8] = include_bytes!("../../kansei-core/tests/fixtures/L10-medium.arfont");

fn count() -> usize { std::env::var("COUNT").ok().and_then(|v| v.parse().ok()).unwrap_or(64) }
fn max_speed() -> f32 { std::env::var("MAXSPEED").ok().and_then(|v| v.parse().ok()).unwrap_or(10.0) }
fn steps() -> u32 { std::env::var("STEPS").ok().and_then(|v| v.parse().ok()).unwrap_or(300) }

struct App {
    window: Option<Arc<Window>>,
    ran: bool,
}

impl App {
    fn new() -> Self {
        Self { window: None, ran: false }
    }

    fn run_test(&mut self, window: Arc<Window>) {
        let size = window.inner_size();

        // ── Renderer ─────────────────────────────────────────────────────
        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 1,
            ..Default::default()
        });
        pollster::block_on(renderer.initialize_with_target(window.clone()));

        // ── Font + glyph volumes + slot layout ──────────────────────────
        let font = FontAtlas::parse(FONT).expect("parse font");
        let set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);

        let mut layout = SlotLayout::hh_mm_ss(4.0, 1.0);
        layout.set_time(11, 11, 11); // slot 0 shows glyph '1'

        let slot0_min = layout.slots[0].world_min;
        let slot0_size = layout.slots[0].world_size;
        let center = [
            slot0_min[0] + 0.5 * slot0_size[0],
            slot0_min[1] + 0.5 * slot0_size[1],
            slot0_min[2] + 0.5 * slot0_size[2],
        ];
        log::info!("Slot 0 box: min={:?} size={:?} center={:?}", slot0_min, slot0_size, center);

        // ── Spawn particles in a small cloud offset from the slot center ──
        // A few units to the side and below, so a correct attractor
        // produces a clear net velocity toward the center.
        let spawn_center = [center[0] - 3.0, center[1] - 3.0, center[2]];
        let spawn_half = [0.5f32, 0.5, 0.5];

        let mut positions = vec![0.0f32; count() * 4];
        let mut rng: u64 = 987654321;
        for i in 0..count() {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let ux = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uy = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uz = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            positions[i * 4] = spawn_center[0] + ux * spawn_half[0];
            positions[i * 4 + 1] = spawn_center[1] + uy * spawn_half[1];
            positions[i * 4 + 2] = spawn_center[2] + uz * spawn_half[2];
            positions[i * 4 + 3] = 1.0;
        }

        // ── Sim + attractor ─────────────────────────────────────────────
        let mut sim = FluidSimulation::new(
            &renderer,
            FluidSimulationOptions {
                max_particles: count() as u32,
                dimensions: 3,
                smoothing_radius: 0.3,
                substeps: 1,
                world_bounds_padding: 1.0,
                ..DEFAULT_OPTIONS
            },
            &positions,
        );
        // Bounds default to the spawn cloud + padding, which would wall the
        // particles off from the slot box; contain both, as the example does.
        sim.world_bounds_min = [-22.0, -10.0, -6.0];
        sim.world_bounds_max = [22.0, 20.0, 6.0];
        sim.rebuild_grid();

        let attractor = GlyphAttractor::new(&renderer, &set, count() as u32);
        attractor.set_tags(&vec![0i32; count()]);
        attractor.set_slots(&layout);
        attractor.set_params(0.016, 40.0, max_speed(), 5.0, 0.1, 3.0);

        // ── Run steps: sim.update_batched then attractor.dispatch in a
        //    separate encoder, per the attractor's documented contract. ──
        for _ in 0..steps() {
            sim.update_batched(0.016, 0.0, [0.0, 0.0], [0.0, 0.0]);

            let positions_buf = sim.positions_buffer().expect("positions buffer");
            let velocities_buf = sim.velocities_buffer().expect("velocities buffer");
            let mut encoder = renderer
                .device()
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Attractor/Frame"),
                });
            attractor.dispatch(&mut encoder, positions_buf, velocities_buf);
            renderer.queue().submit(std::iter::once(encoder.finish()));
        }

        // ── Readback positions ───────────────────────────────────────────
        let device = renderer.device();
        let queue = renderer.queue();
        let size_bytes = count() as u64 * 16;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            sim.positions_buffer().unwrap(),
            0,
            &staging,
            0,
            size_bytes,
        );
        queue.submit(std::iter::once(enc.finish()));

        let vstaging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback-vel"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(sim.velocities_buffer().unwrap(), 0, &vstaging, 0, size_bytes);
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let vslice = vstaging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        let tx2 = tx.clone();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        vslice.map_async(wgpu::MapMode::Read, move |r| {
            tx2.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let pos: &[f32] = bytemuck::cast_slice(&data);
        let vdata = vslice.get_mapped_range();
        let vel: &[f32] = bytemuck::cast_slice(&vdata);
        let mut mean_speed = 0.0f64;
        let mut max_speed = 0.0f64;
        for i in 0..count() {
            let v = &vel[i * 4..i * 4 + 3];
            let sp = ((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) as f64).sqrt();
            mean_speed += sp;
            max_speed = max_speed.max(sp);
        }
        mean_speed /= count() as f64;
        println!("  mean_speed={mean_speed:.2} max_speed={max_speed:.2}");
        drop(vdata);
        vstaging.unmap();

        let dist_to_center = |p: &[f32]| -> f64 {
            let dx = (p[0] - center[0]) as f64;
            let dy = (p[1] - center[1]) as f64;
            let dz = (p[2] - center[2]) as f64;
            (dx * dx + dy * dy + dz * dz).sqrt()
        };
        let mut mean_dist = 0.0f64;
        let mut nan = 0usize;
        let mut mean_pos = [0.0f64; 3];
        let mut in_box = 0usize;
        for i in 0..count() {
            let p = &pos[i * 4..i * 4 + 3];
            if p.iter().any(|v| !v.is_finite()) {
                nan += 1;
                continue;
            }
            mean_dist += dist_to_center(p);
            for d in 0..3 {
                mean_pos[d] += p[d] as f64;
            }
            let inside = (0..3).all(|d| p[d] >= slot0_min[d] && p[d] <= slot0_min[d] + slot0_size[d]);
            if inside {
                in_box += 1;
            }
        }
        mean_dist /= (count() - nan).max(1) as f64;
        for d in 0..3 {
            mean_pos[d] /= (count() - nan).max(1) as f64;
        }
        println!(
            "  steps={} spawn_center={:?} mean_pos={:?} in_box={} world_bounds={:?}..{:?}",
            steps(), spawn_center, mean_pos, in_box, sim.world_bounds_min, sim.world_bounds_max
        );
        drop(data);
        staging.unmap();

        let mut initial_dist = 0.0f64;
        for i in 0..count() {
            initial_dist += dist_to_center(&positions[i * 4..i * 4 + 3]);
        }
        initial_dist /= count() as f64;

        let ratio = mean_dist / initial_dist;
        log::info!("initial_dist={initial_dist} final_dist={mean_dist} ratio={ratio} nan={nan}");

        // The GPU sim is not bit-deterministic and the initial pressure blast
        // amplifies that: identical runs land anywhere in ~0.55–0.8. The
        // failures this guards against (walled off, oscillating through the
        // box, never pulled in) all sit at or above 1.0.
        if nan == 0 && ratio < 0.85 {
            println!(
                "ATTRACTOR TEST: PASS  (mean distance to slot center {initial_dist:.3} -> {mean_dist:.3}, ratio={ratio:.3})"
            );
            std::process::exit(0);
        } else {
            println!(
                "ATTRACTOR TEST: FAIL  (mean distance to slot center {initial_dist:.3} -> {mean_dist:.3}, ratio={ratio:.3}, nan={nan})"
            );
            std::process::exit(1);
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Kansei \u{2014} Attractor Test")
                    .with_inner_size(winit::dpi::LogicalSize::new(400, 300)),
            )
            .unwrap(),
        );
        window.request_redraw();
        self.window = Some(window);
    }

    fn window_event(&mut self, _el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        if self.ran {
            return;
        }
        if let WindowEvent::RedrawRequested = event {
            self.ran = true;
            let window = self.window.clone().unwrap();
            self.run_test(window);
        } else if let Some(ref w) = self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei — Attractor Test");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
