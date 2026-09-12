//! Headless glyph *formation* check for the attractor force model.
//!
//! Mirrors the fluid_clock WASM test mode: a few thousand particles, all
//! tagged to slot 0, which shows one big glyph. Runs SPH + attractor for
//! `STEPS` frames, reads positions back, and reports how well the particles
//! fill the stroke (not just whether they drift toward the box, which is
//! what `attractor_test` checks). Prints an XY occupancy map next to the
//! SDF sign map so the shape can be eyeballed in a terminal.
//!
//! Env knobs: GLYPH (digit, default 8), STEPS (default 600), STIFF (90),
//! TARGET (0.1), BASIN (5), DRAG (3), COUNT (3000), GRAVITY (-9.8), W (14), D (6).

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

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}
fn env_u32(k: &str, d: u32) -> u32 {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
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

        let glyph = env_u32("GLYPH", 8).min(9);
        let steps = env_u32("STEPS", 600);
        let stiffness = env_f32("STIFF", 90.0);
        let target = env_f32("TARGET", 0.1);
        let basin = env_f32("BASIN", 5.0);
        let count = env_u32("COUNT", 3000) as usize;
        let gravity_y = env_f32("GRAVITY", -9.8);
        let drag = env_f32("DRAG", 3.0);
        let max_speed = env_f32("MAXSPEED", 20.0);

        // ── Glyph volumes + one big test slot (same as fluid_clock test mode) ──
        let font = FontAtlas::parse(FONT).expect("parse font");
        let res_xy = env_u32("RES", 64);
        let res_z = 8u32;
        let bold = env_f32("BOLD", 0.5); // inside threshold on atlas alpha; lower = bolder
        let t_sdf = std::time::Instant::now();
        let set = GlyphVolumeSet::for_clock_with_threshold(&font, res_xy, res_z, 0.5, bold);
        println!("  built 11 glyph volumes at {res_xy}x{res_xy}x{res_z} in {:.0} ms", t_sdf.elapsed().as_secs_f64() * 1000.0);
        let vol = set.volume_for_digit(glyph).expect("glyph volume");

        let glyph_y = 2.0f32;
        let w = env_f32("W", 14.0); // glyph box width/height (real clock cell: 4)
        let d = env_f32("D", 6.0); // glyph box depth (real clock: 1.5)
        let mut layout = SlotLayout::hh_mm_ss(4.0, 1.5);
        layout.slots[0].glyph_id = glyph as i32;
        layout.slots[0].world_size = [w, w, d];
        layout.slots[0].world_min = [-w * 0.5, -w * 0.5 + glyph_y, -d * 0.5];
        for i in 1..layout.slots.len() {
            layout.slots[i].glyph_id = -1;
        }
        let bmin = layout.slots[0].world_min;
        let bsize = layout.slots[0].world_size;

        // ── Spawn: same box as the WASM example ──
        let center = [0.0f32, 0.0, 0.0];
        let half = [18.0f32, 8.0, 2.0];
        let mut positions = vec![0.0f32; count * 4];
        let mut rng: u64 = 0x9E3779B97F4A7C15;
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
                smoothing_radius: 1.0,
                pressure_multiplier: 46.5,
                near_pressure_multiplier: 20.0,
                density_target: 8.6,
                viscosity: 1.0,
                damping: 1.0,
                gravity: [0.0, gravity_y, 0.0],
                mouse_force: 1600.0,
                substeps: 2,
                world_bounds_padding: 2.0,
                ..DEFAULT_OPTIONS
            },
            &positions,
        );
        sim.world_bounds_min = [-22.0, -10.0, -12.0];
        sim.world_bounds_max = [22.0, 20.0, 12.0];
        sim.rebuild_grid();

        let attractor = GlyphAttractor::new(&renderer, &set, count as u32);
        attractor.set_tags(&vec![0i32; count]);
        attractor.set_slots(&layout);
        attractor.set_params(1.0 / 60.0, stiffness, max_speed, basin, target, drag);

        for _ in 0..steps {
            sim.update_batched(1.0 / 60.0, 0.0, [0.0, 0.0], [0.0, 0.0]);
            let pos_buf = sim.positions_buffer().unwrap();
            let vel_buf = sim.velocities_buffer().unwrap();
            let mut enc = renderer
                .device()
                .create_command_encoder(&Default::default());
            attractor.dispatch(&mut enc, pos_buf, vel_buf);
            renderer.queue().submit(std::iter::once(enc.finish()));
        }

        // ── Readback positions ──
        let device = renderer.device();
        let queue = renderer.queue();
        let size_bytes = (count * 16) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(sim.positions_buffer().unwrap(), 0, &staging, 0, size_bytes);
        queue.submit(std::iter::once(enc.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let pos: &[f32] = bytemuck::cast_slice(&data);

        // ── Analyse: sample the SDF the same way the shader does ──
        let sdf_at = |x: i32, y: i32, z: i32| -> f32 {
            let cx = x.clamp(0, res_xy as i32 - 1) as u32;
            let cy = y.clamp(0, res_xy as i32 - 1) as u32;
            let cz = z.clamp(0, res_z as i32 - 1) as u32;
            vol.data[(((cz * res_xy) + cy) * res_xy + cx) as usize]
        };
        let map_w = res_xy as usize;
        let map_h = (res_xy / 2) as usize; // terminal chars are ~2:1
        let mut occ = vec![0u32; map_w * map_h];
        let mut outside_box = 0usize;
        let mut nan = 0usize;
        let mut inside_stroke = 0usize;
        let mut above_target = 0usize;
        let mut sum_s = 0.0f64;
        let mut in_box = 0usize;
        let mut hist = [0u32; 8]; // s in [-1,-0.5),[-0.5,-0.2),[-0.2,-0.1),[-0.1,0),[0,0.05),[0.05,0.1),[0.1,0.2),[0.2,1]
        for i in 0..count {
            let p = [pos[i * 4], pos[i * 4 + 1], pos[i * 4 + 2]];
            if p.iter().any(|v| !v.is_finite()) {
                nan += 1;
                continue;
            }
            let l = [
                (p[0] - bmin[0]) / bsize[0],
                (p[1] - bmin[1]) / bsize[1],
                (p[2] - bmin[2]) / bsize[2],
            ];
            if l.iter().any(|v| *v < 0.0 || *v > 1.0) {
                outside_box += 1;
                continue;
            }
            in_box += 1;
            let vx = (l[0] * (res_xy as f32 - 1.0)) as i32;
            let vy = (l[1] * (res_xy as f32 - 1.0)) as i32;
            let vz = (l[2] * (res_z as f32 - 1.0)) as i32;
            let s = sdf_at(vx, vy, vz);
            sum_s += s as f64;
            if s > 0.0 {
                inside_stroke += 1;
            }
            if s > target {
                above_target += 1;
            }
            let b = if s < -0.5 { 0 } else if s < -0.2 { 1 } else if s < -0.1 { 2 } else if s < 0.0 { 3 }
                else if s < 0.05 { 4 } else if s < 0.1 { 5 } else if s < 0.2 { 6 } else { 7 };
            hist[b] += 1;
            let mx = ((l[0] * (map_w as f32 - 1.0)) as usize).min(map_w - 1);
            let my = ((l[1] * (map_h as f32 - 1.0)) as usize).min(map_h - 1);
            occ[my * map_w + mx] += 1;
        }
        drop(data);
        staging.unmap();

        println!(
            "GLYPH FORM TEST glyph={glyph} steps={steps} stiff={stiffness} target={target} basin={basin} drag={drag} max_speed={max_speed} bold={bold} w={w} d={d} count={count} gravity={gravity_y}"
        );
        println!(
            "  nan={nan} outside_box={outside_box} ({:.1}%) in_box={in_box}",
            100.0 * outside_box as f32 / count as f32
        );
        println!(
            "  of in_box: inside_stroke={inside_stroke} ({:.1}%) above_target={above_target} ({:.1}%) mean_s={:.3}",
            100.0 * inside_stroke as f32 / in_box.max(1) as f32,
            100.0 * above_target as f32 / in_box.max(1) as f32,
            sum_s / in_box.max(1) as f64
        );
        println!("  s histogram [<-.5, -.5..-.2, -.2..-.1, -.1..0 | 0..0.05, .05..0.1, 0.1..0.2, >0.2]: {:?}", hist);

        // Occupancy map (top row = glyph top) with the SDF sign map beside it.
        let zmid = (res_z / 2) as i32;
        let max_occ = occ.iter().cloned().max().unwrap_or(1).max(1);
        println!("  occupancy (left, ' .:-=+*#%@' by density)   |   SDF sign (right, # inside)");
        for my in (0..map_h).rev() {
            let mut left = String::new();
            for mx in 0..map_w {
                let c = occ[my * map_w + mx];
                let ch = if c == 0 { ' ' } else {
                    let ramp = b".:-=+*#%@";
                    let t = (c as f32 / max_occ as f32 * (ramp.len() - 1) as f32).round() as usize;
                    ramp[t.min(ramp.len() - 1)] as char
                };
                left.push(ch);
            }
            let mut right = String::new();
            let vy = ((my as f32 + 0.5) / map_h as f32 * res_xy as f32) as i32;
            for mx in 0..map_w {
                let s = sdf_at(mx as i32, vy, zmid);
                right.push(if s > 0.0 { '#' } else if s > -0.1 { '.' } else { ' ' });
            }
            println!("  |{left}| |{right}|");
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
                    .with_title("Kansei \u{2014} Glyph Form Test")
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
