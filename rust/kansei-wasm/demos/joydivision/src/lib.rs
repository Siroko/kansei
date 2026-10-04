use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{ComputeBuffer, BufferType};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{Geometry, PlaneGeometry, InstancedGeometry};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, MsdfTextOptions};
use kansei_wasm::{fetch_bytes, Canvas, Frame};

mod fft_compute;
mod text_layout;

/// The displacement the page's panel starts at.
const FFT_AMPLITUDE: f32 = 17.0;

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let (width, height) = canvas.size();
    let renderer = canvas.renderer(RendererConfig {
        sample_count: 4,
        clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    }).await;

    // Set up camera — perspective looking straight at the text plane
    let mut camera = Camera::new(45.0, 0.1, 2000.0, canvas.aspect());
    // Position camera far enough back to see text filling the width.
    // With font_size=2.5 and ~50 chars max, line width ~ 50*2.5*0.5 ~ 62 units.
    // At FOV 45, half-width at distance d = d * tan(22.5 deg) ~ d * 0.414
    // We want half-width ~ 35 => d ~ 35 / 0.414 ~ 85
    camera.set_position(0.0, 0.0, 100.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();

    let controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.0, 0.0), 100.0);

    let state = Rc::new(RefCell::new(State {
        renderer, camera, controls,
        width, height,
        scene: Scene::new(),
        fft_compute: None,
        fft_amplitude: FFT_AMPLITUDE,
        noise_dt: 0.0,
        particle_count: 0,
        text_scene_idx: None,
        text_offset: [0.0, 0.0, 1.0],
        glyph_rot_x: 90.0_f32.to_radians(),
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().render_frame(frame));

    log::info!("Kansei WASM — Joy Division running");
    Ok(())
}

struct State {
    renderer: Renderer,
    camera: Camera,
    controls: CameraControls,
    width: u32,
    height: u32,
    scene: Scene,
    fft_compute: Option<fft_compute::FftCompute>,
    fft_amplitude: f32,
    /// Seconds rendered since the last update_fft: how far the noise field moves on the next.
    noise_dt: f32,
    particle_count: u32,
    text_scene_idx: Option<usize>,
    text_offset: [f32; 3],
    glyph_rot_x: f32,
}

impl State {
    fn render_frame(&mut self, frame: &Frame) {
        if let Some((width, height)) = frame.resized {
            frame.resize(&mut self.renderer, &mut self.camera);
            self.width = width;
            self.height = height;
        }
        self.noise_dt += frame.dt;
        self.controls.update(&mut self.camera, 0.0);
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();

        if self.particle_count > 0 {
            self.renderer.render(&mut self.scene, &mut self.camera);
        }
    }
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            f(&mut rc.borrow_mut());
        }
    });
}

/// Load the font and lay out the lyrics: `lines_json` and `timestamps_json` are JSON arrays of
/// each line's text and start time (seconds).
#[wasm_bindgen]
pub async fn init_text(lines_json: String, timestamps_json: String) -> Result<(), JsValue> {
    let json_error = |e: serde_json::Error| JsValue::from_str(&format!("lyrics JSON: {e}"));
    let lines: Vec<String> = serde_json::from_str(&lines_json).map_err(json_error)?;
    let timestamps: Vec<f32> = serde_json::from_str(&timestamps_json).map_err(json_error)?;
    let font = fetch_bytes("assets/fonts/L10-medium.arfont").await?;
    let atlas = FontAtlas::parse(&font).map_err(|e| JsValue::from_str(&format!("font: {e:?}")))?;

    let font_size: f32 = 2.5;
    let line_spacing: f32 = 3.5;

    let data = text_layout::build_lyrics_particles(
        &lines, &timestamps, &atlas, font_size, line_spacing,
    );

    log::info!("Joy Division: {} lines, {} total particles, atlas {}x{}, {} glyph metrics",
        data.total_lines, data.total_particles, atlas.width, atlas.height, atlas.glyphs.len());

    with_state(|st| {
        st.particle_count = data.total_particles;

        // 1. Create FFT compute (owns the positions buffer)
        let fft = fft_compute::FftCompute::new(&st.renderer, &data, line_spacing, st.fft_amplitude);

        // 2. Wrap the compute's positions buffer as vertex source
        let pos_buf = ComputeBuffer::from_external(
            "JoyDiv/Positions", fft.positions_buffer().clone(), BufferType::Storage,
        ).with_vertex_vec4(3);

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let atlas_buf = ComputeBuffer::from_slice(
            "JoyDiv/AtlasRects", BufferType::Storage, usage, &data.atlas_rects,
        ).with_vertex_vec4(4);
        let plane_buf = ComputeBuffer::from_slice(
            "JoyDiv/PlaneRects", BufferType::Storage, usage, &data.plane_rects,
        ).with_vertex_vec4(5);
        let col_buf = ComputeBuffer::from_slice(
            "JoyDiv/Colors", BufferType::Storage, usage, &data.colors,
        ).with_vertex_vec4(6);

        // 3. InstancedGeometry
        let base = PlaneGeometry::new(1.0, 1.0);
        let instanced = InstancedGeometry::new(
            base, data.total_particles, vec![pos_buf, atlas_buf, plane_buf, col_buf],
        );

        // 4. MSDF glyph material: fft_displace puts each glyph's turn about x in position.w.
        //    Glyphs write depth, and the atlas is sampled anisotropically.
        let mat = Material::msdf_text("JoyDiv/MSDF", &atlas, MsdfTextOptions { rotate_x_by_w: true });

        // 5. Add text renderable to scene
        let renderable = Renderable::new(instanced, mat);
        let text_idx = st.scene.add(SceneNode::Renderable(renderable));
        st.text_scene_idx = Some(text_idx);

        // 6. Wave fill mesh — solid black occlusion curtain (renders FIRST for depth)
        {
            let line_count = fft.wave_line_count();
            let verts_per_line = fft.wave_verts_per_line();
            // Same index pattern as wave lines (quads from pairs of verts)
            let segs_per_line = (verts_per_line - 1) as usize;
            let total_indices = line_count as usize * segs_per_line * 6;
            let mut fill_indices: Vec<u32> = Vec::with_capacity(total_indices);
            for line in 0..line_count {
                let base = line * verts_per_line * 2;
                for seg in 0..segs_per_line as u32 {
                    let top_l = base + seg * 2;
                    let bot_l = top_l + 1;
                    let top_r = base + (seg + 1) * 2;
                    let bot_r = top_r + 1;
                    fill_indices.push(top_l);
                    fill_indices.push(bot_l);
                    fill_indices.push(top_r);
                    fill_indices.push(bot_l);
                    fill_indices.push(bot_r);
                    fill_indices.push(top_r);
                }
            }

            // the compute pass writes the vertices
            let fill_geo = Geometry::from_gpu_vertices("JoyDiv/WaveFill", fft.fill_positions_buffer().clone(), fill_indices);

            const WAVE_LINE_WGSL: &str = include_str!("shaders/wave_line.wgsl");
            let mut fill_mat = Material::new(
                "JoyDiv/WaveFill",
                WAVE_LINE_WGSL,
                vec![
                    Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
                ],
                MaterialOptions {
                    transparent: false,
                    depth_write: Some(true),
                    cull_mode: CullMode::None,
                    ..Default::default()
                },
            );
            fill_mat.set_uniform_bindable(0, "JoyDiv/WaveFillColor", &[0.0f32, 0.0, 0.0, 1.0]);

            let fill_renderable = Renderable::new(fill_geo, fill_mat);
            st.scene.add(SceneNode::Renderable(fill_renderable));
        }

        // 7. Wave line mesh — tessellated ribbon lines displaced by elevation map
        {
            let line_count = fft.wave_line_count();
            let verts_per_line = fft.wave_verts_per_line();
            // Build index buffer for TriangleList topology:
            // Each segment = 2 triangles (quad) connecting 4 vertices:
            //   top_i, bottom_i, top_i+1, bottom_i, bottom_i+1, top_i+1
            let segs_per_line = (verts_per_line - 1) as usize;
            let total_indices = line_count as usize * segs_per_line * 6;
            let mut indices: Vec<u32> = Vec::with_capacity(total_indices);
            for line in 0..line_count {
                let base = line * verts_per_line * 2; // 2 verts per sample
                for seg in 0..segs_per_line as u32 {
                    let top_l = base + seg * 2;
                    let bot_l = top_l + 1;
                    let top_r = base + (seg + 1) * 2;
                    let bot_r = top_r + 1;
                    // Triangle 1
                    indices.push(top_l);
                    indices.push(bot_l);
                    indices.push(top_r);
                    // Triangle 2
                    indices.push(bot_l);
                    indices.push(bot_r);
                    indices.push(top_r);
                }
            }

            // the compute pass writes the vertices
            let wave_geo = Geometry::from_gpu_vertices("JoyDiv/WaveLines", fft.wave_positions_buffer().clone(), indices);

            // Wave material: solid white triangles
            const WAVE_LINE_WGSL: &str = include_str!("shaders/wave_line.wgsl");
            let mut wave_mat = Material::new(
                "JoyDiv/WaveLine",
                WAVE_LINE_WGSL,
                vec![
                    Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
                ],
                MaterialOptions {
                    transparent: false,
                    depth_write: Some(true),
                    cull_mode: CullMode::None,
                    ..Default::default()
                },
            );
            wave_mat.set_uniform_bindable(0, "JoyDiv/WaveLineColor", &[1.0f32, 1.0, 1.0, 1.0]);

            let wave_renderable = Renderable::new(wave_geo, wave_mat);
            st.scene.add(SceneNode::Renderable(wave_renderable));
        }

        // 8. Store FFT compute
        st.fft_compute = Some(fft);
    });
    Ok(())
}

/// Called from JS each frame with FFT frequency data.
#[wasm_bindgen]
pub fn update_fft(fft_data: &[u8], current_time: f32) {
    with_state(|st| {
        if let Some(ref mut fft) = st.fft_compute {
            fft.set_fft_amplitude(st.fft_amplitude);
            fft.update(fft_data, current_time, std::mem::take(&mut st.noise_dt));
        }
    });
}

#[wasm_bindgen]
pub fn set_noise_scale(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_noise_scale(v); } }); }
#[wasm_bindgen]
pub fn set_noise_strength(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_noise_strength(v); } }); }
#[wasm_bindgen]
pub fn set_noise_speed(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_noise_speed(v); } }); }
#[wasm_bindgen]
pub fn set_noise_zoom(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_noise_zoom(v); } }); }
#[wasm_bindgen]
pub fn set_noise_mix(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_noise_mix(v); } }); }
#[wasm_bindgen]
pub fn set_wave_thickness(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_wave_thickness(v); } }); }
#[wasm_bindgen]
pub fn set_peak_exponent(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_peak_exponent(v); } }); }
#[wasm_bindgen]
pub fn set_temporal_blend(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_temporal_blend(v); } }); }
#[wasm_bindgen]
pub fn set_fft_pow(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_fft_pow(v); } }); }
#[wasm_bindgen]
pub fn set_peak_min(v: f32) { with_state(|s| { if let Some(ref mut f) = s.fft_compute { f.set_peak_min(v); } }); }

#[wasm_bindgen]
pub fn set_text_visible(v: bool) {
    with_state(|s| {
        if let Some(idx) = s.text_scene_idx {
            if let Some(r) = s.scene.get_renderable_mut(idx) {
                r.visible = v;
            }
        }
    });
}

#[wasm_bindgen]
pub fn set_text_offset(x: f32, y: f32, z: f32) {
    with_state(|s| {
        s.text_offset = [x, y, z];
        if let Some(ref mut f) = s.fft_compute { f.set_text_offset([x, y, z]); }
    });
}

#[wasm_bindgen]
pub fn set_glyph_rot_x(degrees: f32) {
    with_state(|s| {
        s.glyph_rot_x = degrees.to_radians();
        if let Some(ref mut f) = s.fft_compute { f.set_glyph_rot_x(degrees.to_radians()); }
    });
}

#[wasm_bindgen]
pub fn set_fft_amplitude(v: f32) { with_state(|s| s.fft_amplitude = v); }
