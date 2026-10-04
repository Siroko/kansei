use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{ComputeBuffer, BufferType};
use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{PlaneGeometry, InstancedGeometry};
use kansei_core::materials::Material;
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, MsdfTextOptions};
use kansei_wasm::{fetch_bytes, Canvas, Frame};

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
        word_meta: Vec::new(),
        auto_rotate_speed: 0.1,
        elapsed_time: 0.0,
        frame_count: 0, frame_time_sum: 0.0,
        current_fps: 0.0, current_frame_ms: 0.0,
        particle_count: 0,
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
    /// Per particle: word id, letter index, word length, the word's first particle.
    word_meta: Vec<u32>,
    auto_rotate_speed: f32,
    elapsed_time: f32,
    frame_count: u32,
    frame_time_sum: f64,
    current_fps: f64,
    current_frame_ms: f64,
    particle_count: u32,
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
        let mouse = self.mouse_in_world();
        if let Some(sim) = self.simulation.as_mut() {
            let clamped_dt = (dt as f32).min(0.05); // cap to avoid explosions on tab return
            self.elapsed_time += clamped_dt;
            sim.update(&self.steering_params, clamped_dt, self.elapsed_time, &mouse);
        }

        // Render scene — engine handles surface acquire, clear, instanced draw, present.
        // Skip until init_text has added renderables (avoids 0-size buffer warning).
        if self.particle_count > 0 {
            self.renderer.render(&mut self.scene, &mut self.camera);
        }
    }

    /// The cursor as the steering shader takes it: the ray from the camera through it (MouseVectors
    /// gives NDC with y down) and its motion in the camera's right/up plane, in NDC units as before.
    fn mouse_in_world(&self) -> steering_sim::MouseState {
        let (x, y) = (self.mouse.position.x, -self.mouse.position.y);
        let inverse_view_projection = (self.camera.projection_matrix * self.camera.view_matrix).inverse().to_glam();
        let near = Vec3::from(inverse_view_projection.project_point3(Vec3::new(x, y, 0.0).to_glam()));
        let far = Vec3::from(inverse_view_projection.project_point3(Vec3::new(x, y, 1.0).to_glam()));
        let ray = Vec3::new(far.x - near.x, far.y - near.y, far.z - near.z).normalize();
        // direction is the previous position minus the current one: the motion is (-x, +y) once y
        // points up
        let m = &self.camera.inverse_view_matrix.data;
        let (right, up) = ([m[0], m[1], m[2]], [m[4], m[5], m[6]]);
        let (dx, dy) = (-self.mouse.direction.x, self.mouse.direction.y);
        steering_sim::MouseState {
            strength: self.mouse.strength.min(1.0),
            ray_origin: [near.x, near.y, near.z],
            ray_dir: [ray.x, ray.y, ray.z],
            dir: [0, 1, 2].map(|i| right[i] * dx + up[i] * dy),
        }
    }
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }

const PALETTE: [[f32; 4]; 6] = [
    [1.0, 0.75, 0.80, 1.0],  // soft pink
    [0.70, 0.82, 1.0, 1.0],  // soft blue
    [0.75, 1.0, 0.82, 1.0],  // soft green
    [1.0, 0.95, 0.70, 1.0],  // soft yellow
    [0.85, 0.72, 1.0, 1.0],  // soft purple
    [1.0, 0.82, 0.72, 1.0],  // soft coral
];

/// Load the font and build the words (a JSON array of strings) into the scene.
#[wasm_bindgen]
pub async fn init_text(words_json: String) -> Result<(), JsValue> {
    let words: Vec<String> = serde_json::from_str(&words_json)
        .map_err(|e| JsValue::from_str(&format!("words JSON: {e}")))?;
    let font = fetch_bytes("assets/fonts/L10-medium.arfont").await?;
    let atlas = FontAtlas::parse(&font).map_err(|e| JsValue::from_str(&format!("font: {e:?}")))?;

    let data = text_data::build_particle_data(&words, &atlas, 4.0, &PALETTE, 100.0, 1.5);

    log::info!("Steering Text: {} words, {} total particles, atlas {}x{}, {} glyph metrics",
        data.total_words, data.total_particles, atlas.width, atlas.height, atlas.glyphs.len());

    with_state(|st| {
        st.particle_count = data.total_particles;

        // 1. Create steering simulation (owns the authoritative positions buffer)
        let steering_params = steering_sim::SteeringParams::default();
        let sim = steering_sim::SteeringSimulation::new(&st.renderer, &data, &steering_params);

        // 2. Wrap the sim's positions buffer as a vertex source for the instanced mesh.
        //    The sim writes positions via compute; the mesh reads them as vertex input.
        //    The glyph rects are static; colours change with the palette.
        let pos_buf = ComputeBuffer::from_external(
            "SteeringText/Positions", sim.positions_buffer().clone(), BufferType::Storage,
        ).with_vertex_vec4(3);

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let atlas_buf = ComputeBuffer::from_slice(
            "SteeringText/AtlasRects", BufferType::Storage, usage, &data.atlas_rects,
        ).with_vertex_vec4(4);
        let plane_buf = ComputeBuffer::from_slice(
            "SteeringText/PlaneRects", BufferType::Storage, usage, &data.plane_rects,
        ).with_vertex_vec4(5);
        // Colors buffer: create via device so we keep a handle for live palette updates
        use wgpu::util::DeviceExt;
        let colors_gpu = st.renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SteeringText/Colors"),
            contents: bytemuck::cast_slice(&data.colors),
            usage,
        });
        let col_buf = ComputeBuffer::from_external(
            "SteeringText/Colors", colors_gpu.clone(), BufferType::Storage,
        ).with_vertex_vec4(6);

        // 3. InstancedGeometry
        let base = PlaneGeometry::new(1.0, 1.0);
        let instanced = InstancedGeometry::new(
            base, data.total_particles, vec![pos_buf, atlas_buf, plane_buf, col_buf],
        );

        // 4. MSDF glyph material (positions' w is 1: no per-glyph turn). Letters write depth
        //    so the 3D cloud occludes itself, and the atlas is sampled anisotropically.
        let mat = Material::msdf_text("SteeringText/MSDF", &atlas, MsdfTextOptions::default());

        // 5. Add renderable to the scene
        let renderable = Renderable::new(instanced, mat);
        st.scene.add(SceneNode::Renderable(renderable));

        // 6. Store simulation + colors buffer for live palette updates
        st.simulation = Some(sim);
        st.steering_params = steering_params;
        st.colors_buffer = Some(colors_gpu);
        st.word_meta = data.word_meta;
        st.elapsed_time = 0.0;
    });
    Ok(())
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

/// Set the background (the renderer's clear colour).
#[wasm_bindgen]
pub fn set_background(r: f32, g: f32, b: f32) {
    with_state(|s| s.renderer.config.clear_color = Vec4::new(r, g, b, 1.0));
}

/// Update the colour palette (RGBA per entry): every letter takes its word's entry, as at start.
#[wasm_bindgen]
pub fn set_palette(palette_flat: &[f32]) {
    with_state(|st| {
        let palette: Vec<[f32; 4]> = palette_flat.chunks_exact(4).map(|c| [c[0], c[1], c[2], c[3]]).collect();
        if let (Some(colors_buf), false) = (&st.colors_buffer, palette.is_empty()) {
            let colors = text_data::word_colors(&st.word_meta, &palette);
            st.renderer.queue().write_buffer(colors_buf, 0, bytemuck::cast_slice(&colors));
        }
    });
}
