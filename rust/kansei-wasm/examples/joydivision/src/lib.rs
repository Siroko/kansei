use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{ComputeBuffer, BufferType};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{Geometry, PlaneGeometry, InstancedGeometry, Vertex};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};

mod fft_compute;
mod text_layout;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("Kansei WASM (Joy Division) initialized");
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = canvas_id;
        return Err(JsValue::from_str("kansei-wasm start() is only supported on wasm32"));
    }

    #[cfg(target_arch = "wasm32")]
    {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;

    let dpr = window.device_pixel_ratio().min(2.0);
    let width = (canvas.client_width() as f64 * dpr) as u32;
    let height = (canvas.client_height() as f64 * dpr) as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig {
        width, height,
        sample_count: 4,
        clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;

    // Set up camera — perspective looking straight at the text plane
    let aspect = width as f32 / height as f32;
    let mut camera = Camera::new(45.0, 0.1, 2000.0, aspect);
    // Position camera far enough back to see text filling the width.
    // With font_size=2.5 and ~50 chars max, line width ~ 50*2.5*0.5 ~ 62 units.
    // At FOV 45, half-width at distance d = d * tan(22.5 deg) ~ d * 0.414
    // We want half-width ~ 35 => d ~ 35 / 0.414 ~ 85
    camera.set_position(0.0, 0.0, 100.0);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();

    let controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 0.0, 0.0), 100.0);

    let state = Rc::new(RefCell::new(State {
        renderer, camera, controls,
        width, height,
        scene: Scene::new(),
        fft_compute: None,
        fft_amplitude: 6.0,
        particle_count: 0,
        text_scene_idx: None,
        text_offset: [0.0, 0.0, 1.0],
        glyph_rot_x: 90.0_f32.to_radians(),
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    // Animation loop
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone(); let s = state.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        s.borrow_mut().render_frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());

    log::info!("Kansei WASM — Joy Division running");
    Ok(())
    }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
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
    particle_count: u32,
    text_scene_idx: Option<usize>,
    text_offset: [f32; 3],
    glyph_rot_x: f32,
}

impl State {
    fn render_frame(&mut self) {
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

#[wasm_bindgen]
pub fn init_text(
    words_json: &str,
    glyphs_json: &str,
    msdf_rgba: &[u8],
    atlas_width: u32,
    atlas_height: u32,
    timestamps_json: &str,
) {
    let lines: Vec<String> = serde_json::from_str(words_json)
        .expect("Failed to parse lines JSON");
    let glyphs: Vec<text_layout::GlyphMetrics> = serde_json::from_str(glyphs_json)
        .expect("Failed to parse glyphs JSON");
    let timestamps: Vec<f32> = serde_json::from_str(timestamps_json)
        .expect("Failed to parse timestamps JSON");

    let font_size: f32 = 2.5;
    let line_spacing: f32 = 3.5;
    let fft_amplitude: f32 = 6.0;

    let data = text_layout::build_lyrics_particles(
        &lines, &timestamps, &glyphs, font_size, line_spacing,
    );

    log::info!("Joy Division: {} lines, {} total particles, atlas {}x{}, {} glyph metrics",
        data.total_lines, data.total_particles, atlas_width, atlas_height, glyphs.len());

    with_state(|st| {
        st.particle_count = data.total_particles;

        // 1. Create FFT compute (owns the positions buffer)
        let fft = fft_compute::FftCompute::new(&st.renderer, &data, line_spacing, fft_amplitude);

        // 2. Wrap the compute's positions buffer as vertex source
        let pos_buf = ComputeBuffer::from_external(
            "JoyDiv/Positions", fft.positions_buffer().clone(), BufferType::Storage,
        ).with_vertex_vec4(3);

        let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST;
        let img_buf = ComputeBuffer::from_slice(
            "JoyDiv/ImageBounds", BufferType::Storage, usage, &data.image_bounds,
        ).with_vertex_vec4(4);
        let plane_buf = ComputeBuffer::from_slice(
            "JoyDiv/PlaneBounds", BufferType::Storage, usage, &data.plane_bounds,
        ).with_vertex_vec4(5);
        let col_buf = ComputeBuffer::from_slice(
            "JoyDiv/Colors", BufferType::Storage, usage, &data.colors,
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
            "JoyDiv/MSDF",
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

        // 5. Add text renderable to scene
        let renderable = Renderable::new(instanced, mat);
        let text_idx = st.scene.add(SceneNode::Renderable(renderable));
        st.text_scene_idx = Some(text_idx);

        // 6. Wave fill mesh — solid black occlusion curtain (renders FIRST for depth)
        {
            let line_count = fft.wave_line_count();
            let verts_per_line = fft.wave_verts_per_line();
            let total_verts = (line_count * verts_per_line * 2) as usize;

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

            let placeholder_verts: Vec<Vertex> = vec![
                Vertex {
                    position: [0.0, 0.0, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                };
                total_verts
            ];
            let mut fill_geo = Geometry::new("JoyDiv/WaveFill", placeholder_verts, fill_indices);
            fill_geo.initialize(st.renderer.device());
            fill_geo.vertex_buffer = Some(fft.fill_positions_buffer().clone());

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
            // 2 vertices per sample (top + bottom of ribbon)
            let total_verts = (line_count * verts_per_line * 2) as usize;

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

            // Create Geometry with placeholder vertices (compute shader overwrites)
            let placeholder_verts: Vec<Vertex> = vec![
                Vertex {
                    position: [0.0, 0.0, 0.0, 1.0],
                    normal: [0.0, 0.0, 1.0],
                    uv: [0.0, 0.0],
                };
                total_verts
            ];
            let mut wave_geo = Geometry::new("JoyDiv/WaveLines", placeholder_verts, indices);
            wave_geo.initialize(st.renderer.device());
            wave_geo.vertex_buffer = Some(fft.wave_positions_buffer().clone());

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

        // 7. Store FFT compute
        st.fft_compute = Some(fft);
    });
}

/// Called from JS each frame with FFT frequency data.
#[wasm_bindgen]
pub fn update_fft(fft_data: &[u8], current_time: f32) {
    with_state(|st| {
        if let Some(ref mut fft) = st.fft_compute {
            fft.set_fft_amplitude(st.fft_amplitude);
            fft.update(fft_data, current_time);
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
