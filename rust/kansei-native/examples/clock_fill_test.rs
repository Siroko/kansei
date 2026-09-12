//! Headless reproduction of the fluid_clock WASM example's *full* clock
//! configuration: 60K-particle pool, 8-slot layout, per-frame retag
//! (recruitment) + attractor, stepped the way the page steps it (time scale
//! applied per step, ~2 steps per frame). Reads tags + positions back and
//! reports per-slot fill, how well each slot's particles sit in its stroke,
//! where the pool surface is, and CPU-side timings for sim vs attractor.
//!
//! Env knobs: FRAMES (300), COUNT (60000), SCALE (1.9), GLYPH_Y (3),
//! PER_SLOT (400), CAPTURE (1.15), MAXSPEED (12), STIFF (90), ZHALF (6),
//! CELL (10), DEPTH (4).

use std::sync::Arc;
use std::time::Instant;

use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{
    ClockState, FluidSimulation, FluidSimulationOptions, GlyphAttractor, RetagParams, SlotLayout, DEFAULT_OPTIONS,
};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const FONT: &[u8] = include_bytes!("../../kansei-core/tests/fixtures/L10-medium.arfont");

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn readback<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, bytes: u64) -> Vec<T> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut enc = device.create_command_encoder(&Default::default());
    enc.copy_buffer_to_buffer(src, 0, &staging, 0, bytes);
    queue.submit(std::iter::once(enc.finish()));
    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::Maintain::Wait);
    rx.recv().unwrap().unwrap();
    let out: Vec<T> = bytemuck::cast_slice(&slice.get_mapped_range()).to_vec();
    staging.unmap();
    out
}

struct App {
    window: Option<Arc<Window>>,
    ran: bool,
}

impl App {
    fn run_test(&mut self, window: Arc<Window>) {
        let size = window.inner_size();
        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 1,
            ..Default::default()
        });
        pollster::block_on(renderer.initialize_with_target(window.clone()));

        let frames = env_u32("FRAMES", 300);
        let stacked = env_u32("STACKED", 1) != 0;
        let count = env_u32("COUNT", 80_000) as usize;
        let scale = env_f32("SCALE", 1.9);
        let glyph_y = env_f32("GLYPH_Y", if stacked { 29.0 } else { 3.0 });
        let per_slot = env_u32("PER_SLOT", if stacked { 2000 } else { 1000 });
        let capture = env_f32("CAPTURE", 1.15);
        let capture_below = env_f32("CAPTURE_BELOW", if stacked { 45.0 } else { 8.0 });
        let emit_height = env_f32("EMIT_HEIGHT", if stacked { 1.0 } else { 4.0 });
        let emit_spread = env_f32("EMIT_SPREAD", if stacked { 2.5 } else { 3.0 });
        let emit_rate = env_u32("EMIT_RATE", if stacked { 100 } else { 50 });
        let max_speed = env_f32("MAXSPEED", 12.0);
        let stiffness = env_f32("STIFF", 90.0);
        let zhalf = env_f32("ZHALF", if stacked { 10.0 } else { 6.0 });
        let cell = env_f32("CELL", if stacked { 14.0 } else { 10.0 });
        let depth = env_f32("DEPTH", 6.0);
        let bold = env_f32("BOLD", 0.35);
        // Tank + spawn overrides (defaults = the fluid_clock example).
        let xhalf = env_f32("XHALF", if stacked { 25.0 } else { 50.0 });
        let ymin = env_f32("YMIN", -10.0);
        let ymax = env_f32("YMAX", if stacked { 60.0 } else { 30.0 });
        let zmin = env_f32("ZMIN", -zhalf);
        let zmax = env_f32("ZMAX", zhalf);
        let attract_on = env_u32("ATTRACT", 1) != 0;
        // Frame at which the seconds digit flips (12:34:56 -> 12:34:57), to
        // exercise release + recruitment; 0 disables.
        let change_at = env_u32("CHANGE_AT", 150);
        let checkpoint = env_u32("CHECKPOINT", 0); // print per-slot summary every N frames (0 = off)
        let gravity_y = env_f32("GRAVITY", -9.8);

        // ── Same pool as the example ──
        let center = [env_f32("SPAWN_CX", 0.0), env_f32("SPAWN_CY", if stacked { -3.0 } else { -4.0 }), env_f32("SPAWN_CZ", 0.0)];
        let half = [env_f32("SPAWN_HX", if stacked { 23.0 } else { 45.0 }), env_f32("SPAWN_HY", if stacked { 6.0 } else { 5.0 }), env_f32("SPAWN_HZ", if stacked { 9.0 } else { 4.0f32.min(zhalf) })];
        let mut positions = vec![0.0f32; count * 4];
        let mut rng: u64 = 12345;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            (rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        for i in 0..count {
            let (ux, uy, uz) = (next(), next(), next());
            positions[i * 4] = center[0] + ux * half[0];
            positions[i * 4 + 1] = center[1] + uy * half[1];
            positions[i * 4 + 2] = center[2] + uz * half[2];
            positions[i * 4 + 3] = 1.0;
        }
        let mut sim = FluidSimulation::new(
            &renderer,
            FluidSimulationOptions {
                max_particles: count as u32,
                dimensions: 3,
                smoothing_radius: env_f32("RADIUS", 1.0),
                pressure_multiplier: env_f32("PRESSURE", 46.5),
                near_pressure_multiplier: env_f32("NEAR", 20.0),
                density_target: env_f32("TARGET", 8.6),
                viscosity: env_f32("VISC", 1.0),
                damping: 1.0,
                gravity: [0.0, gravity_y, 0.0],
                mouse_force: 1600.0,
                substeps: 2,
                world_bounds_padding: 2.0,
                ..DEFAULT_OPTIONS
            },
            &positions,
        );
        sim.world_bounds_min = [-xhalf, ymin, zmin];
        sim.world_bounds_max = [xhalf, ymax, zmax];
        sim.rebuild_grid();

        // ── Layout + attractor, as in the example ──
        let font = FontAtlas::parse(FONT).expect("parse font");
        let res_xy = env_u32("RES", 256);
        let set = GlyphVolumeSet::for_clock_with_threshold(&font, res_xy, 8, 0.5, bold);
        let attractor = GlyphAttractor::new(&renderer, &set, count as u32);
        let mut layout = if stacked { SlotLayout::stacked_hh_mm_ss(cell, depth, 1.3) } else { SlotLayout::hh_mm_ss(cell, depth) };
        for s in layout.slots.iter_mut() {
            if stacked {
                s.world_min[1] += glyph_y; // stack offset
            } else {
                s.world_min[1] = glyph_y - s.world_size[1] * 0.5;
            }
        }
        let mut clock = ClockState::new();
        let _ = clock.update(&mut layout, 12, 34, 56);
        for s in layout.slots.iter_mut() {
            s.budget = if s.glyph_id == 10 { (per_slot as f32 * 0.45) as u32 } else { 0 };
        }
        attractor.set_slots(&layout);
        attractor.set_tags(&vec![-1i32; count]);

        // ── Frame loop, stepping like the page does ──
        let step_dt = 1.0f32 / 60.0;
        let scaled_dt = step_dt * scale;
        let mut acc = 0.0f64;
        let mut sim_ms = 0.0f64;
        let mut attr_ms = 0.0f64;
        let mut total_steps = 0u32;
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        for frame in 0..frames {
            acc += step_dt as f64 * scale as f64;
            let mut steps = 0u32;
            let t0 = Instant::now();
            while acc >= step_dt as f64 && steps < 4 {
                sim.update_batched(scaled_dt, 0.0, [0.0, 0.0], [0.0, 0.0]);
                acc -= step_dt as f64;
                steps += 1;
            }
            if acc > step_dt as f64 {
                acc = step_dt as f64;
            }
            device.poll(wgpu::Maintain::Wait);
            sim_ms += t0.elapsed().as_secs_f64() * 1000.0;
            total_steps += steps;

            if !attract_on {
                continue;
            }
            let t1 = Instant::now();
            let dt_frame = steps as f32 * scaled_dt;
            let mut changed_mask = 0u32;
            if change_at > 0 && frame == change_at {
                let changed = clock.update(&mut layout, 12, 34, 57);
                changed_mask = ClockState::changed_mask(&changed);
                for s in layout.slots.iter_mut() {
                    s.budget = if s.glyph_id == 10 { (per_slot as f32 * 0.45) as u32 } else { 0 };
                }
                attractor.set_slots(&layout);
                println!("  frame {frame}: digit change, mask {changed_mask:#b}");
            }
            attractor.set_params(dt_frame, stiffness, max_speed, 5.0, 0.1, 3.0);
            let pos_buf = sim.positions_buffer().unwrap();
            let vel_buf = sim.velocities_buffer().unwrap();
            let mut enc = device.create_command_encoder(&Default::default());
            attractor.retag(&mut enc, pos_buf, vel_buf, &RetagParams { changed_mask, per_slot_count: per_slot, cooldown_frames: 45, capture_scale: capture, capture_below, emit_height, emit_spread, emit_rate });
            attractor.dispatch(&mut enc, pos_buf, vel_buf);
            queue.submit(std::iter::once(enc.finish()));
            device.poll(wgpu::Maintain::Wait);
            attr_ms += t1.elapsed().as_secs_f64() * 1000.0;

            if checkpoint > 0 && (frame + 1) % checkpoint == 0 {
                let tags: Vec<i32> = readback(&device, &queue, attractor.tags_buffer(), (count * 4) as u64);
                let pos: Vec<f32> = readback(&device, &queue, sim.positions_buffer().unwrap(), (count * 16) as u64);
                let mut n = [0usize; 8];
                let mut sy = [0.0f64; 8];
                let mut pool_ys: Vec<f32> = Vec::new();
                for i in 0..count {
                    let t = tags[i];
                    let y = pos[i * 4 + 1];
                    if !y.is_finite() { continue; }
                    if t >= 0 && t < 8 { n[t as usize] += 1; sy[t as usize] += y as f64; } else { pool_ys.push(y); }
                }
                pool_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p95 = pool_ys[((pool_ys.len() as f32 - 1.0) * 0.95) as usize];
                let mut line = format!("  f{:>4} pool p95 {:.2}:", frame + 1, p95);
                for k in 0..8 {
                    line.push_str(&format!(" s{k}[{} y{:+.1}]", n[k], sy[k] / n[k].max(1) as f64));
                }
                println!("{line}");
            }
        }

        // ── Readback + analysis ──
        let tags: Vec<i32> = readback(&device, &queue, attractor.tags_buffer(), (count * 4) as u64);
        let pos: Vec<f32> = readback(&device, &queue, sim.positions_buffer().unwrap(), (count * 16) as u64);

        let res_z = 8u32;
        let mut slot_n = [0usize; 8];
        let mut slot_in_stroke = [0usize; 8];
        let mut slot_in_box = [0usize; 8];
        let mut slot_mean_y = [0.0f64; 8];
        let mut pool_ys: Vec<f32> = Vec::with_capacity(count);
        let mut nan = 0usize;
        for i in 0..count {
            let p = [pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]];
            if p.iter().any(|v| !v.is_finite()) {
                nan += 1;
                continue;
            }
            let t = tags[i];
            if t < 0 || t >= 8 {
                pool_ys.push(p[1]);
                continue;
            }
            let k = t as usize;
            slot_n[k] += 1;
            slot_mean_y[k] += p[1] as f64;
            let sl = &layout.slots[k];
            let l = [
                (p[0] - sl.world_min[0]) / sl.world_size[0],
                (p[1] - sl.world_min[1]) / sl.world_size[1],
                (p[2] - sl.world_min[2]) / sl.world_size[2],
            ];
            if l.iter().any(|v| *v < 0.0 || *v > 1.0) {
                continue;
            }
            slot_in_box[k] += 1;
            let vol = match set.volumes().get(sl.glyph_id as usize) {
                Some(Some(v)) => v,
                _ => continue,
            };
            let vx = ((l[0] * (res_xy as f32 - 1.0)) as u32).min(res_xy - 1);
            let vy = ((l[1] * (res_xy as f32 - 1.0)) as u32).min(res_xy - 1);
            let vz = ((l[2] * (res_z as f32 - 1.0)) as u32).min(res_z - 1);
            let s = vol.data[(((vz * res_xy) + vy) * res_xy + vx) as usize];
            if s > 0.0 {
                slot_in_stroke[k] += 1;
            }
        }
        pool_ys.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // Height profile of the untagged pool, 1-unit bins from the floor.
        let nb = ((ymax - ymin).ceil() as usize).max(1);
        let mut ybins = vec![0usize; nb];
        for y in pool_ys.iter() {
            let b = (((*y - ymin).max(0.0)) as usize).min(nb - 1);
            ybins[b] += 1;
        }
        let pct = |q: f32| pool_ys[((pool_ys.len() as f32 - 1.0) * q) as usize];

        println!(
            "CLOCK FILL TEST frames={frames} count={count} scale={scale} glyph_y={glyph_y} per_slot={per_slot} capture={capture} capture_below={capture_below} emit={emit_height}/{emit_spread}/{emit_rate} max_speed={max_speed} tank x±{xhalf} y[{ymin},{ymax}] z[{zmin},{zmax}] cell={cell} stacked={stacked} attract={attract_on}"
        );
        println!(
            "  timing (CPU wall incl. GPU wait): sim {:.2} ms/frame ({:.2} steps/frame), attractor+retag {:.2} ms/frame",
            sim_ms / frames as f64,
            total_steps as f64 / frames as f64,
            attr_ms / frames as f64
        );
        println!(
            "  pool: {} untagged, y p50={:.2} p95={:.2} max={:.2}; nan={nan}",
            pool_ys.len(),
            pct(0.5),
            pct(0.95),
            pool_ys[pool_ys.len() - 1]
        );
        let mut prof = String::new();
        for (b, n) in ybins.iter().enumerate() {
            if *n > 0 {
                prof.push_str(&format!(" y{:+.0}:{}", ymin + b as f32, n));
            }
        }
        println!("  pool height profile (1-unit bins):{prof}");
        for k in 0..8 {
            let sl = &layout.slots[k];
            let cap = if sl.budget > 0 { sl.budget } else { per_slot };
            println!(
                "  slot {k} glyph {:>2} box y[{:.1},{:.1}] cap {cap:>4}: tagged {:>4}, in_box {:>4}, in_stroke {:>4} ({:.0}%), mean_y {:.2}",
                sl.glyph_id,
                sl.world_min[1],
                sl.world_min[1] + sl.world_size[1],
                slot_n[k],
                slot_in_box[k],
                slot_in_stroke[k],
                100.0 * slot_in_stroke[k] as f32 / slot_n[k].max(1) as f32,
                slot_mean_y[k] / slot_n[k].max(1) as f64
            );
        }
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
                    .with_title("Kansei \u{2014} Clock Fill Test")
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
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App { window: None, ran: false }).unwrap();
}
