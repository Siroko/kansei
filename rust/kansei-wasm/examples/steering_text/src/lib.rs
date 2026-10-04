use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{ComputeBuffer, BufferType};
use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{PlaneGeometry, InstancedGeometry};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{Canvas, Frame};

mod steering_sim;
mod text_data;

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let (width, height) = canvas.size();
    let renderer = canvas.renderer(RendererConfig {
        sample_count: 4,
        clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
        ..Default::default()
    }).await;

    let camera = Camera::new(45.0, 0.1, 1000.0, canvas.aspect());
    let controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.0, 0.0), 150.0);
    let mouse = MouseVectors::from_canvas(canvas.element());

    let state = Rc::new(RefCell::new(State {
        renderer, camera, controls, mouse,
        width, height,
        scene: Scene::new(),
        simulation: None,
        steering_params: steering_sim::SteeringParams::default(),
        colors_buffer: None,
        auto_rotate_speed: 0.1,
        elapsed_time: 0.0,
        frame_count: 0, frame_time_sum: 0.0,
        current_fps: 0.0, current_frame_ms: 0.0,
        particle_count: 0, word_count: 0,
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().render_frame(frame));

    log::info!("Kansei WASM — Steering Text running");
    Ok(())
}

struct State {
    renderer: Renderer,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    width: u32,
    height: u32,
    scene: Scene,
    simulation: Option<steering_sim::SteeringSimulation>,
    steering_params: steering_sim::SteeringParams,
    colors_buffer: Option<wgpu::Buffer>,
    auto_rotate_speed: f32,
    elapsed_time: f32,
    frame_count: u32,
    frame_time_sum: f64,
    current_fps: f64,
    current_frame_ms: f64,
    particle_count: u32,
    word_count: u32,
}

impl State {
    fn render_frame(&mut self, frame: &Frame) {
        if let Some((width, height)) = frame.resized {
            frame.resize(&mut self.renderer, &mut self.camera);
            self.width = width;
            self.height = height;
        }
        let frame_ms = frame.dt as f64 * 1000.0;
        let dt = (frame_ms * 0.001).max(1.0 / 1000.0);

        // FPS tracking
        self.frame_time_sum += frame_ms;
        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            let avg = self.frame_time_sum / 60.0;
            self.current_frame_ms = avg;
            self.current_fps = 1000.0 / avg;
            self.frame_time_sum = 0.0;
        }

        // Auto-rotate + controls
        if self.auto_rotate_speed > 0.001 {
            let az = self.controls.azimuth() + self.auto_rotate_speed * dt as f32;
            self.controls.set_azimuth(az);
        }
        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(dt as f32);
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();

        // Step the steering simulation before rendering
        if let Some(ref mut sim) = self.simulation {
            let clamped_dt = (dt as f32).min(0.05); // cap to avoid explosions on tab return
            self.elapsed_time += clamped_dt;
            let mouse = steering_sim::MouseState {
                strength: self.mouse.strength.min(1.0),
                pos: [self.mouse.position.x, self.mouse.position.y],
                dir: [self.mouse.direction.x, self.mouse.direction.y],
            };
            sim.update(&self.steering_params, clamped_dt, self.elapsed_time, &mouse);

            // Debug: log every 120 frames
            if self.frame_count % 120 == 0 {
                log::info!("Sim: t={:.2} dt={:.4} vehicles={}", self.elapsed_time, clamped_dt, sim.vehicle_count());
            }
        }

        // Render scene — engine handles surface acquire, clear, instanced draw, present.
        // Skip until init_text has added renderables (avoids 0-size buffer warning).
        if self.particle_count > 0 {
            self.renderer.render(&mut self.scene, &mut self.camera);
        }
    }
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }

#[wasm_bindgen]
pub fn init_text(
    words_json: &str,
    glyphs_json: &str,
    msdf_rgba: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) {
    let words: Vec<String> = serde_json::from_str(words_json)
        .expect("Failed to parse words JSON");
    let glyphs: Vec<text_data::GlyphMetrics> = serde_json::from_str(glyphs_json)
        .expect("Failed to parse glyphs JSON");

    let palette: Vec<[f32; 4]> = vec![
        [1.0, 0.75, 0.80, 1.0],  // soft pink
        [0.70, 0.82, 1.0, 1.0],  // soft blue
        [0.75, 1.0, 0.82, 1.0],  // soft green
        [1.0, 0.95, 0.70, 1.0],  // soft yellow
        [0.85, 0.72, 1.0, 1.0],  // soft purple
        [1.0, 0.82, 0.72, 1.0],  // soft coral
    ];
    let data = text_data::build_particle_data(
        &words, &glyphs, 4.0, &palette, 100.0, 1.5,
    );

    log::info!("Steering Text: {} words, {} total particles, atlas {}x{}, {} glyph metrics",
        data.total_words, data.total_particles, atlas_width, atlas_height, glyphs.len());

    with_state(|st| {
        st.particle_count = data.total_particles;
        st.word_count = data.total_words;

        // 1. Create steering simulation (owns the authoritative positions buffer)
        let steering_params = steering_sim::SteeringParams::default();
        let sim = steering_sim::SteeringSimulation::new(&st.renderer, &data, &steering_params);

        // 2. Wrap the sim's positions buffer as a vertex source for the instanced mesh.
        //    The sim writes positions via compute; the mesh reads them as vertex input.
        //    Other instance buffers (image_bounds, plane_bounds, colors) are static.
        let pos_buf = ComputeBuffer::from_external(
            "SteeringText/Positions", sim.positions_buffer().clone(), BufferType::Storage,
        ).with_vertex_vec4(3);

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let img_buf = ComputeBuffer::from_slice(
            "SteeringText/ImageBounds", BufferType::Storage, usage, &data.image_bounds,
        ).with_vertex_vec4(4);
        let plane_buf = ComputeBuffer::from_slice(
            "SteeringText/PlaneBounds", BufferType::Storage, usage, &data.plane_bounds,
        ).with_vertex_vec4(5);
        // Colors buffer: create via device so we keep a handle for live palette updates
        use wgpu::util::DeviceExt;
        let colors_gpu = st.renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringText/Colors"),
            contents: bytemuck::cast_slice(&data.colors),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let col_buf = ComputeBuffer::from_external(
            "SteeringText/Colors", colors_gpu.clone(), BufferType::Storage,
        ).with_vertex_vec4(6);

        // 3. InstancedGeometry
        let base = PlaneGeometry::new(1.0, 1.0);
        let instanced = InstancedGeometry::new(
            base, data.total_particles, vec![pos_buf, img_buf, plane_buf, col_buf],
        );

        // 4. MSDF atlas texture + sampler + material
        let atlas = kansei_core::buffers::Texture::from_rgba("MSDF/Atlas", atlas_width, atlas_height, msdf_rgba);
        let atlas_sampler = kansei_core::buffers::Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear)
            .with_anisotropy(8);
        const MSDF_WGSL: &str = include_str!("shaders/msdf_text.wgsl");
        let mut mat = Material::new(
            "SteeringText/MSDF",
            MSDF_WGSL,
            vec![
                Binding::texture_2d(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
                Binding::sampler(1, ShaderStages::FRAGMENT),
            ],
            MaterialOptions {
                transparent: true,
                depth_write: Some(true),
                cull_mode: CullMode::None,
                ..Default::default()
            },
        );
        mat.set_bindable(0, atlas);
        mat.set_bindable(1, atlas_sampler);

        // 5. Add renderable to the scene
        let renderable = Renderable::new(instanced, mat);
        st.scene.add(SceneNode::Renderable(renderable));

        // 6. Store simulation + colors buffer for live palette updates
        st.simulation = Some(sim);
        st.steering_params = steering_params;
        st.colors_buffer = Some(colors_gpu);
        st.elapsed_time = 0.0;
    });
}

// ── Tweakpane setters ──
#[wasm_bindgen] pub fn set_separation_strength(v: f32) { with_state(|s| s.steering_params.separation_strength = v); }
#[wasm_bindgen] pub fn set_separation_radius(v: f32)   { with_state(|s| s.steering_params.separation_radius = v); }
#[wasm_bindgen] pub fn set_wander_strength(v: f32)     { with_state(|s| s.steering_params.wander_strength = v); }
#[wasm_bindgen] pub fn set_wander_speed(v: f32)        { with_state(|s| s.steering_params.wander_speed = v); }
#[wasm_bindgen] pub fn set_mouse_force(v: f32)         { with_state(|s| s.steering_params.mouse_force = v); }
#[wasm_bindgen] pub fn set_max_speed(v: f32)           { with_state(|s| s.steering_params.max_speed = v); }
#[wasm_bindgen] pub fn set_max_force(v: f32)           { with_state(|s| s.steering_params.max_force = v); }
#[wasm_bindgen] pub fn set_damping(v: f32)             { with_state(|s| s.steering_params.damping = v); }
#[wasm_bindgen] pub fn set_bounds_size(v: f32)         { with_state(|s| s.steering_params.bounds_size = v); }
#[wasm_bindgen] pub fn set_cohesion_strength(v: f32)   { with_state(|s| s.steering_params.cohesion_strength = v); }
#[wasm_bindgen] pub fn set_alignment_strength(v: f32)  { with_state(|s| s.steering_params.alignment_strength = v); }
#[wasm_bindgen] pub fn set_attractor_pos(x: f32, y: f32, z: f32) { with_state(|s| s.steering_params.attractor_pos = [x, y, z]); }
#[wasm_bindgen] pub fn set_attractor_strength(v: f32)  { with_state(|s| s.steering_params.attractor_strength = v); }
#[wasm_bindgen] pub fn set_repulsion_strength(v: f32)  { with_state(|s| s.steering_params.repulsion_strength = v); }
#[wasm_bindgen] pub fn set_repulsion_radius(v: f32)    { with_state(|s| s.steering_params.repulsion_radius = v); }
#[wasm_bindgen] pub fn set_max_per_cell(v: u32)        { with_state(|s| s.steering_params.max_per_cell = v); }
#[wasm_bindgen] pub fn set_verlet_iterations(v: u32)   { with_state(|s| s.steering_params.verlet_iterations = v); }
#[wasm_bindgen] pub fn set_auto_rotate_speed(v: f32)   { with_state(|s| s.auto_rotate_speed = v); }

/// Update color palette (6 colors × RGBA). Rewrites the entire colors buffer
/// so every word picks up the new palette on the next frame.
#[wasm_bindgen]
pub fn set_palette(palette_flat: &[f32]) {
    with_state(|st| {
        if let Some(ref colors_buf) = st.colors_buffer {
            // palette_flat = [r,g,b,a, r,g,b,a, ...] for 6 colors
            let palette_len = palette_flat.len() / 4;
            if palette_len == 0 { return; }
            // Rebuild per-particle colors cycling through the palette
            let total = st.particle_count as usize;
            // We need word_id per particle — reconstruct from word_count
            // Simple: particle i belongs to word (i * word_count / total) approximately.
            // Better: just cycle by particle index / avg_word_len. But simplest:
            // re-derive from the simulation's word structure. Since we don't store
            // word_meta here, use a simpler heuristic: cycle colors per ~7 particles
            // (average word length). Good enough for visual variety.
            let avg_word_len = if st.word_count > 0 { total / st.word_count as usize } else { 7 };
            let avg_word_len = avg_word_len.max(1);
            let mut colors = Vec::with_capacity(total * 4);
            for i in 0..total {
                let word_id = i / avg_word_len;
                let ci = (word_id % palette_len) * 4;
                colors.push(palette_flat[ci]);
                colors.push(palette_flat[ci + 1]);
                colors.push(palette_flat[ci + 2]);
                colors.push(palette_flat[ci + 3]);
            }
            st.renderer.queue().write_buffer(colors_buf, 0, bytemuck::cast_slice(&colors));
        }
    });
}
