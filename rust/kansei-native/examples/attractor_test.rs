//! Native readback verification for the glyph attractor.
//!
//! Spawns a small cloud of particles offset from glyph slot 0's box, runs
//! the SPH solver + glyph attractor compute pass for ~120 steps, then reads
//! the velocities buffer back to the CPU and asserts the mean velocity of
//! the tagged particles points toward the slot's box center — i.e. the
//! attractor is actually pulling particles in.
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

const PARTICLE_COUNT: usize = 4096;
const STEPS: u32 = 120;

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

        let mut positions = vec![0.0f32; PARTICLE_COUNT * 4];
        let mut rng: u64 = 987654321;
        for i in 0..PARTICLE_COUNT {
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
                max_particles: PARTICLE_COUNT as u32,
                dimensions: 3,
                smoothing_radius: 0.3,
                substeps: 1,
                world_bounds_padding: 1.0,
                ..DEFAULT_OPTIONS
            },
            &positions,
        );
        sim.rebuild_grid();

        let attractor = GlyphAttractor::new(&renderer, &set, PARTICLE_COUNT as u32);
        attractor.set_tags(&vec![0i32; PARTICLE_COUNT]);
        attractor.set_slots(&layout);
        attractor.set_params(0.016, 40.0, 20.0, 5.0);

        // ── Run steps: sim.update_batched then attractor.dispatch in a
        //    separate encoder, per the attractor's documented contract. ──
        for _ in 0..STEPS {
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

        // ── Readback velocities ──────────────────────────────────────────
        let device = renderer.device();
        let queue = renderer.queue();
        let count = PARTICLE_COUNT as u64;
        let size_bytes = count * 16;

        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(
            sim.velocities_buffer().unwrap(),
            0,
            &staging,
            0,
            size_bytes,
        );
        queue.submit(std::iter::once(enc.finish()));

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).unwrap();
        });
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let vels: &[f32] = bytemuck::cast_slice(&data);

        let mut mean_vel = [0.0f64; 3];
        for i in 0..PARTICLE_COUNT {
            mean_vel[0] += vels[i * 4] as f64;
            mean_vel[1] += vels[i * 4 + 1] as f64;
            mean_vel[2] += vels[i * 4 + 2] as f64;
        }
        for d in 0..3 {
            mean_vel[d] /= PARTICLE_COUNT as f64;
        }
        drop(data);
        staging.unmap();

        // Direction from the spawn cloud's center to slot 0's box center.
        let mut dir = [
            (center[0] - spawn_center[0]) as f64,
            (center[1] - spawn_center[1]) as f64,
            (center[2] - spawn_center[2]) as f64,
        ];
        let dir_len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
        for d in dir.iter_mut() {
            *d /= dir_len;
        }

        let dot = mean_vel[0] * dir[0] + mean_vel[1] * dir[1] + mean_vel[2] * dir[2];

        log::info!(
            "mean_vel={:?} dir_to_center={:?} dot={}",
            mean_vel, dir, dot
        );

        if dot > 0.0 {
            println!(
                "ATTRACTOR TEST: PASS  (dot={:.6}, mean_vel={:?}, dir_to_center={:?})",
                dot, mean_vel, dir
            );
            std::process::exit(0);
        } else {
            println!(
                "ATTRACTOR TEST: FAIL  (dot={:.6}, mean_vel={:?}, dir_to_center={:?})",
                dot, mean_vel, dir
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
