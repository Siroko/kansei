//! Native readback verification for the GPU-resident particle tagging passes
//! (`GlyphAttractor::retag`): recruit, release, and cooldown.
//!
//! Spawns exactly `BUDGET` particles inside slot 0's capture box and a large
//! population far outside all slot boxes, then:
//!   1. Runs a recruit-only retag (nothing changed) and asserts exactly
//!      `BUDGET` particles get tagged to slot 0 — precisely the in-box ones.
//!   2. Runs a retag with slot 0 marked "changed" (its digit flipped) and
//!      asserts the tagged particles are released (tag -> -1, cooldown set)
//!      and, since the in-box population is exactly the budget and all of it
//!      just went on cooldown, none can be re-recruited this pass — slot 0's
//!      count must be 0.
//!
//! This is the real GPU execution gate for the tagging WGSL (Task 2): the
//! `clear_fill` / `count_fill` / `release` / `recruit` compute passes were
//! never actually dispatched on a device before this example.

use std::sync::Arc;

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{GlyphAttractor, SlotLayout, RetagParams};

use wgpu::util::DeviceExt;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const FONT: &[u8] = include_bytes!("../../kansei-core/tests/fixtures/L10-medium.arfont");

/// Exactly the per-slot recruit budget. The in-box cluster is sized to match
/// so the release assertion is unambiguous (no in-box stragglers left to
/// re-recruit after release).
const BUDGET: u32 = 100;
const OUTSIDE_COUNT: u32 = 400;
const COOLDOWN_FRAMES: u32 = 30;

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

        let mut layout = SlotLayout::hh_mm_ss(4.0, 1.5);
        layout.set_time(11, 11, 11);

        let s0 = layout.slots[0];
        let center0 = [
            s0.world_min[0] + 0.5 * s0.world_size[0],
            s0.world_min[1] + 0.5 * s0.world_size[1],
            s0.world_min[2] + 0.5 * s0.world_size[2],
        ];
        log::info!("Slot 0 box: min={:?} size={:?} center={:?}", s0.world_min, s0.world_size, center0);

        // ── Spawn: BUDGET particles inside slot 0's box, OUTSIDE_COUNT far away ──
        let total: u32 = BUDGET + OUTSIDE_COUNT;
        let mut positions = vec![0.0f32; (total as usize) * 4];
        let mut rng: u64 = 424242;
        let mut next_unit = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };

        // In-box cluster: jitter within ±0.4 * world_size of each axis around
        // the box center, comfortably inside the ±0.5 * world_size half-box.
        for i in 0..BUDGET as usize {
            let ux = next_unit();
            let uy = next_unit();
            let uz = next_unit();
            positions[i * 4] = center0[0] + ux * 0.4 * s0.world_size[0];
            positions[i * 4 + 1] = center0[1] + uy * 0.4 * s0.world_size[1];
            positions[i * 4 + 2] = center0[2] + uz * 0.4 * s0.world_size[2];
            positions[i * 4 + 3] = 1.0;
        }
        // Far-outside cluster: nowhere near any of the 8 slot boxes (which sit
        // close to the origin), so it can never be recruited.
        for i in 0..OUTSIDE_COUNT as usize {
            let idx = BUDGET as usize + i;
            positions[idx * 4] = 1000.0 + i as f32;
            positions[idx * 4 + 1] = 0.0;
            positions[idx * 4 + 2] = 0.0;
            positions[idx * 4 + 3] = 1.0;
        }

        let device = renderer.device();
        let queue = renderer.queue();

        let positions_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("TaggerTest/Positions"),
            contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // Velocities are only touched by retag when emitting; a zeroed buffer suffices here.
        let velocities_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("velocities"),
            contents: bytemuck::cast_slice(&vec![0.0f32; total as usize * 4]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // ── Attractor ────────────────────────────────────────────────────
        let attractor = GlyphAttractor::new(&renderer, &set, total);
        attractor.set_slots(&layout);
        attractor.set_tags(&vec![-1i32; total as usize]);

        let read_tags = |device: &wgpu::Device, queue: &wgpu::Queue| -> Vec<i32> {
            let size_bytes = (total as u64) * 4;
            let staging = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("TaggerTest/Readback"),
                size: size_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut enc = device.create_command_encoder(&Default::default());
            enc.copy_buffer_to_buffer(attractor.tags_buffer(), 0, &staging, 0, size_bytes);
            queue.submit(std::iter::once(enc.finish()));

            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                tx.send(r).unwrap();
            });
            device.poll(wgpu::Maintain::Wait);
            rx.recv().unwrap().unwrap();

            let data = slice.get_mapped_range();
            let tags: Vec<i32> = bytemuck::cast_slice(&data).to_vec();
            drop(data);
            staging.unmap();
            tags
        };

        // ── Phase 1: recruit (nothing changed yet) ─────────────────────────
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TaggerTest/Recruit"),
            });
            attractor.retag(&mut enc, &positions_buf, &velocities_buf, &RetagParams { changed_mask: 0, per_slot_count: BUDGET, cooldown_frames: COOLDOWN_FRAMES, capture_scale: 1.0, ..Default::default() });
            queue.submit(std::iter::once(enc.finish()));
        }
        let tags_after_recruit = read_tags(device, queue);

        let slot0_tagged: Vec<usize> =
            (0..total as usize).filter(|&i| tags_after_recruit[i] == 0).collect();
        let in_box_indices: Vec<usize> = (0..BUDGET as usize).collect();
        let outside_all_untagged =
            (BUDGET as usize..total as usize).all(|i| tags_after_recruit[i] == -1);

        let assert_a_ok = slot0_tagged.len() == BUDGET as usize
            && slot0_tagged == in_box_indices
            && outside_all_untagged;

        log::info!(
            "Assert A: slot0_tagged.len()={} expected={} matches_in_box={} outside_all_untagged={}",
            slot0_tagged.len(),
            BUDGET,
            slot0_tagged == in_box_indices,
            outside_all_untagged
        );

        if !assert_a_ok {
            println!(
                "TAGGER TEST: FAIL  Assert A (recruit) failed: tagged_count={} expected={} matches_in_box_set={} all_outside_untagged={}",
                slot0_tagged.len(),
                BUDGET,
                slot0_tagged == in_box_indices,
                outside_all_untagged
            );
            std::process::exit(1);
        }

        // ── Phase 2: release (slot 0's digit "changed") ────────────────────
        {
            let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TaggerTest/Release"),
            });
            attractor.retag(&mut enc, &positions_buf, &velocities_buf, &RetagParams { changed_mask: 1 << 0, per_slot_count: BUDGET, cooldown_frames: COOLDOWN_FRAMES, capture_scale: 1.0, ..Default::default() });
            queue.submit(std::iter::once(enc.finish()));
        }
        let tags_after_release = read_tags(device, queue);
        let slot0_count_after_release =
            tags_after_release.iter().filter(|&&t| t == 0).count();

        log::info!("Assert B: slot0_count_after_release={}", slot0_count_after_release);

        let assert_b_ok = slot0_count_after_release == 0;

        if !assert_b_ok {
            println!(
                "TAGGER TEST: FAIL  Assert B (release) failed: slot0_count_after_release={} expected=0",
                slot0_count_after_release
            );
            std::process::exit(1);
        }

        println!(
            "TAGGER TEST: PASS  (recruited={}, slot0_after_release={})",
            slot0_tagged.len(),
            slot0_count_after_release
        );
        std::process::exit(0);
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
                    .with_title("Kansei \u{2014} Tagger Test")
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
    log::info!("Kansei — Tagger Test");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
