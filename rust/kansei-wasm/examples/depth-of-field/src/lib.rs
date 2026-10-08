//! Physical depth of field on a scrolling field of noise-driven columns, the Rust twin of the TS
//! `examples/index_dof_terrain.html`: the same 150 x 150 columns, the same glowing column tops
//! (small bright sources that scatter as bokeh), camera, lens and panel. The circle of confusion
//! follows from a CameraLens on Unreal's 23.76 mm filmback; a long lens (450 mm at f/1.4) gives
//! the terrain the shallow focus of a miniature. See www/index.html for the URL parameters.

use std::cell::RefCell;
use std::rc::Rc;

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, DepthOfFieldEffect, DepthOfFieldOptions, DofDebugView, HighlightOptions,
    SSAOEffect, SSAOOptions, ToneMapEffect, ToneMapOptions, ToneMapper,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{param, param_or, Canvas};

/// Unreal's default filmback width, mm.
const SENSOR_MM: f32 = 23.76;

// ── Scene parameters (as the TS example's) ──
/// Columns per side (GRID_SIZE² in all).
const GRID_SIZE: u32 = 150;
/// World-space gap between column centres.
const GRID_SPACING: f32 = 1.0;
/// Noise frequency: lower gives broader hills.
const NOISE_ZOOM: f64 = 0.07;
/// The tallest column.
const SCALE_Y_STRENGTH: f64 = 60.0;
/// World units per second along Z.
const SCROLL_SPEED: f32 = 35.0;
/// The share of columns whose top glows.
const LIGHT_SHARE: f64 = 0.03;
/// Their tops' brightness (the white columns read about 1).
const LIGHT_RADIANCE: f32 = 12.0;

const GRID_HALF: f32 = (GRID_SIZE - 1) as f32 * GRID_SPACING * 0.5;
const GRID_SPAN: f32 = GRID_SIZE as f32 * GRID_SPACING;

/// Lambert from one light at the top right of the scene, baked as constants, written to the
/// GBuffer as the TS shader writes it. The whole field is one instanced draw: each instance is a
/// column (x, start z, height, glow of its top), and the vertex shader scrolls it along Z by the
/// shared offset, wrapping it to the back. Prefixed with GBUFFER_OUT_WGSL and the grid constants.
const COLUMNS_WGSL: &str = r#"
const LIGHT_POS: vec3<f32> = vec3<f32>(80.0, 150.0, -60.0);
const AMBIENT: f32 = 0.12;
const BASE_COLOR: vec3<f32> = vec3<f32>(1.0, 1.0, 1.0);
const LIGHT_COLOR: vec3<f32> = vec3<f32>(1.0, 0.62, 0.3);

// scroll along Z in world units, kept in [0, GRID_SPAN)
@group(0) @binding(0) var<uniform> scroll_offset: vec4<f32>;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) glow: f32,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) column: vec4<f32>, // x, start z, height (y scale), glow of the top face
) -> VertexOutput {
    // a column that scrolls past the front edge wraps to the back of the grid
    var z = column.y + scroll_offset.x;
    if (z > MAX_Z) { z -= GRID_SPAN; }
    let local = vec4<f32>(position.x + column.x, position.y * column.z, position.z + z, 1.0);
    let world_pos = world_matrix * local;
    var out: VertexOutput;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_position = world_pos.xyz;
    // inverse-transpose of the column's Y scale, then the renderable's normal matrix
    let column_normal = normal * vec3<f32>(1.0, 1.0 / column.z, 1.0);
    out.world_normal = normalize((normal_matrix * vec4<f32>(column_normal, 0.0)).xyz);
    out.glow = select(0.0, column.w, normal.y > 0.5);
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> KanseiGBufferOut {
    let n = normalize(input.world_normal);
    let l = normalize(LIGHT_POS - input.world_position);
    let lighting = AMBIENT + (1.0 - AMBIENT) * max(dot(n, l), 0.0);
    let glow = LIGHT_COLOR * input.glow;
    return kansei_gbuffer_out(BASE_COLOR * lighting + glow, glow, n, BASE_COLOR);
}
"#;

// ── Smooth value noise (2D, 4 octaves), in f64 as the TS page computes it ──
fn hash2(x: f64, y: f64) -> f64 {
    let n = (x * 127.1 + y * 311.7).sin() * 43758.5453;
    n - n.floor()
}

fn value_noise(x: f64, y: f64) -> f64 {
    let (ix, iy) = (x.floor(), y.floor());
    let (fx, fy) = (x - ix, y - iy);
    let ux = fx * fx * (3.0 - 2.0 * fx);
    let uy = fy * fy * (3.0 - 2.0 * fy);
    let (a, b) = (hash2(ix, iy), hash2(ix + 1.0, iy));
    let (c, d) = (hash2(ix, iy + 1.0), hash2(ix + 1.0, iy + 1.0));
    a + (b - a) * ux + (c - a) * uy + ((d - c) - (b - a)) * ux * uy
}

/// Normalised to [0, 1].
fn fbm(x: f64, y: f64) -> f64 {
    let (mut v, mut amp, mut freq, mut norm) = (0.0, 0.5, 1.0, 0.0);
    for _ in 0..4 {
        v += value_noise(x * freq, y * freq) * amp;
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    v / norm
}

/// One vec4 per column: x, start z, height, glow of its top (a fixed scatter of glowing tops).
fn columns() -> Vec<f32> {
    let mut data = Vec::with_capacity((GRID_SIZE * GRID_SIZE * 4) as usize);
    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let x = col as f32 * GRID_SPACING - GRID_HALF;
            let z = row as f32 * GRID_SPACING - GRID_HALF;
            let height = fbm(x as f64 * NOISE_ZOOM, z as f64 * NOISE_ZOOM) * SCALE_Y_STRENGTH + 0.1;
            let glow = if hash2(col as f64 + 0.37, row as f64 + 0.71) < LIGHT_SHARE { LIGHT_RADIANCE } else { 0.0 };
            data.extend_from_slice(&[x, z, height as f32, glow]);
        }
    }
    data
}

#[derive(Clone, Copy, PartialEq)]
enum DofMode {
    Cinematic,
    Simple,
    Off,
}

/// The opening view and lens, as the TS page's `DEFAULT_VIEW` (angles in degrees: azimuth around
/// the target, 0 on the +Z side and 90 on the +X side; elevation above the ground plane).
struct View {
    radius: f32,
    azimuth: f32,
    elevation: f32,
    target: Vec3,
    fov: f32,
}

const DEFAULT_VIEW: View = View { radius: 150.0, azimuth: 318.0, elevation: 22.0, target: Vec3::ZERO, fov: 26.0 };

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    mode: DofMode,
    scroll_buffer: wgpu::Buffer,
    scroll: f32,
    /// Fixed scroll time, seconds (stills).
    time: Option<f32>,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

impl State {
    fn frame(&mut self, dt: f32) {
        // wrapped on the CPU so the offset keeps full float precision on the GPU
        self.scroll = match self.time {
            Some(t) => (SCROLL_SPEED * t) % GRID_SPAN,
            None => (self.scroll + SCROLL_SPEED * dt) % GRID_SPAN,
        };
        self.renderer.queue().write_buffer(&self.scroll_buffer, 0, bytemuck::cast_slice(&[self.scroll, 0.0, 0.0, 0.0]));
        self.controls.update(&mut self.camera, dt);
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
    }

    fn cinematic(&mut self) -> Option<&mut CinematicDepthOfFieldEffect> {
        self.volume.effect_mut::<CinematicDepthOfFieldEffect>()
    }

    fn simple(&mut self) -> Option<&mut DepthOfFieldEffect> {
        self.volume.effect_mut::<DepthOfFieldEffect>()
    }

    /// The view as the TS page's 'Copy view' JSON, so a view copied on either page opens on both.
    fn view_json(&mut self) -> String {
        let (radius, azimuth, elevation, target, fov) =
            (self.controls.radius, self.controls.azimuth().to_degrees(), self.controls.elevation().to_degrees(), self.controls.look_target(), self.camera.fov);
        let simple = self
            .simple()
            .map(|d| (d.options.focus_distance, d.options.focus_range, d.options.max_blur))
            .unwrap_or((101.5, 15.0, 24.0));
        let mode = self.mode;
        let dof = self.cinematic();
        let (focus, f_stop, focal, blades, rotation, samples, scatter, debug) = match dof {
            Some(d) => (
                d.lens.focus_distance_m,
                d.lens.f_stop,
                d.lens.focal_length_mm.unwrap_or(450.0),
                d.lens.blade_count,
                d.lens.blade_rotation_deg,
                d.sample_count,
                d.highlights.enabled,
                d.debug_view as u32,
            ),
            None => (simple.0, 1.4, 450.0, 0, 15.0, 72, true, 0),
        };
        let focus = if mode == DofMode::Simple { simple.0 } else { focus };
        let r2 = |v: f32| (v * 100.0).round() / 100.0;
        format!(
            "{{\"radius\":{},\"azimuth\":{},\"elevation\":{},\"target\":[{},{},{}],\"fov\":{},\"focusDistance\":{},\"focusRange\":{},\"maxBlur\":{},\
             \"fStop\":{},\"focalLength\":{},\"blades\":{},\"bladeRotation\":{},\"samples\":{},\"scatter\":{},\"debug\":{},\"mode\":\"{}\"}}",
            r2(radius),
            r2(azimuth.rem_euclid(360.0)),
            r2(elevation),
            r2(target.x),
            r2(target.y),
            r2(target.z),
            fov,
            focus,
            simple.1,
            simple.2,
            f_stop,
            focal,
            blades,
            rotation,
            samples,
            scatter,
            debug,
            match mode {
                DofMode::Cinematic => "cinematic",
                DofMode::Simple => "simple",
                DofMode::Off => "off",
            }
        )
    }

    fn set(&mut self, key: &str, value: f32) {
        let deg = value.to_radians();
        match key {
            "radius" => {
                let (t, a, e) = (self.controls.look_target(), self.controls.azimuth(), self.controls.elevation());
                self.controls.set_view(t, value.max(1.0), a, e);
            }
            "azimuth" => self.controls.set_azimuth(deg),
            "elevation" => self.controls.set_elevation(deg),
            "targetX" | "targetY" | "targetZ" => {
                let mut t = self.controls.look_target();
                match key {
                    "targetX" => t.x = value,
                    "targetY" => t.y = value,
                    _ => t.z = value,
                }
                let (r, a, e) = (self.controls.radius, self.controls.azimuth(), self.controls.elevation());
                self.controls.set_view(t, r, a, e);
            }
            "fov" => {
                self.camera.fov = value.clamp(1.0, 170.0);
                self.camera.update_projection_matrix();
            }
            "focusDistance" => {
                if let Some(d) = self.cinematic() {
                    d.lens.focus_distance_m = value.max(0.01);
                }
                if let Some(d) = self.simple() {
                    d.options.focus_distance = value;
                }
            }
            "focusRange" => {
                if let Some(d) = self.simple() {
                    d.options.focus_range = value;
                }
            }
            "maxBlur" => {
                if let Some(d) = self.simple() {
                    d.options.max_blur = value;
                }
            }
            _ => {
                let Some(d) = self.cinematic() else { return };
                match key {
                    "fStop" => d.lens.f_stop = value.max(0.1),
                    "focalLength" => d.lens.focal_length_mm = Some(value.max(1.0)),
                    "blades" => d.lens.blade_count = value.max(0.0).round() as u32,
                    "bladeRotation" => d.lens.blade_rotation_deg = value,
                    "samples" => d.sample_count = value.max(1.0).round() as u32,
                    "scatter" => d.highlights.enabled = value != 0.0,
                    "debug" => d.debug_view = debug_view(value as u32),
                    _ => log::warn!("depth-of-field: no setting {key}"),
                }
            }
        }
    }
}

fn debug_view(n: u32) -> DofDebugView {
    match n {
        1 => DofDebugView::Background,
        2 => DofDebugView::Near,
        3 => DofDebugView::NearAlpha,
        4 => DofDebugView::Coc,
        _ => DofDebugView::None,
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    // one drawing-buffer pixel per CSS pixel, as the TS page (the CoC is in pixels); ?dpr= overrides it
    let canvas = Canvas::find(canvas_id)?.with_max_pixel_ratio(1.0);
    let renderer = canvas.renderer(RendererConfig { sample_count: 4, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    let mode = match param("dof").as_deref() {
        Some("simple") => DofMode::Simple,
        Some("0") => DofMode::Off,
        _ => DofMode::Cinematic,
    };
    let time = param("t").and_then(|v| v.trim().parse().ok());

    // the field of columns: one instanced draw, one vec4 per column
    let mut scene = Scene::new();
    let column_buffer = ComputeBuffer::from_slice("Columns", BufferType::Storage, BufferUsage::VERTEX, &columns()).with_vertex_vec4(3);
    let geometry = InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), GRID_SIZE * GRID_SIZE, vec![column_buffer]);
    let scroll_buffer = wgpu::util::DeviceExt::create_buffer_init(
        renderer.device(),
        &wgpu::util::BufferInitDescriptor {
            label: Some("ScrollOffset"),
            contents: bytemuck::cast_slice(&[0.0f32; 4]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        },
    );
    let constants = format!("const MAX_Z: f32 = {:.4};\nconst GRID_SPAN: f32 = {:.4};\n", GRID_HALF + GRID_SPACING, GRID_SPAN);
    let mut material = Material::new(
        "Columns",
        &format!("{GBUFFER_OUT_WGSL}\n{constants}{COLUMNS_WGSL}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    material.set_bindable(0, ComputeBuffer::from_external("ScrollOffset", scroll_buffer.clone(), BufferType::Uniform));
    scene.add(SceneNode::Renderable(Renderable::new(geometry, material)));

    // drag to orbit, scroll to zoom
    let mut camera = Camera::new(DEFAULT_VIEW.fov, 0.1, 1000.0, canvas.aspect());
    camera.update_projection_matrix();
    let mut controls = CameraControls::from_canvas(canvas.element(), DEFAULT_VIEW.target, DEFAULT_VIEW.radius);
    controls.set_view(DEFAULT_VIEW.target, DEFAULT_VIEW.radius, DEFAULT_VIEW.azimuth.to_radians(), DEFAULT_VIEW.elevation.to_radians());

    // The cinematic DoF's circle of confusion follows from a physical lens (thin-lens equation).
    // A 150-unit terrain seen through a lens that frames it would be nearly all in focus, so the
    // lens is a long one, set apart from the view's field of view: the shallow focus of a miniature.
    let focus = param_or("focus", 101.5f32);
    let ssao = SSAOEffect::new(SSAOOptions { radius: 0.8, bias: 0.02, kernel_size: 32, strength: 0.5 });
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(ssao)];
    match mode {
        DofMode::Cinematic => {
            let mut dof = CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions {
                lens: CameraLens {
                    focal_length_mm: Some(param_or("focal", 450.0)),
                    f_stop: param_or("fstop", 1.4),
                    focus_distance_m: focus,
                    sensor_width_mm: SENSOR_MM,
                    blade_count: param_or("blades", 0.0f32).max(0.0).round() as u32,
                    blade_rotation_deg: 15.0,
                },
                sample_count: param_or("samples", 72.0f32).max(1.0).round() as u32,
                highlights: HighlightOptions { enabled: param("scatter").as_deref() != Some("0"), ..Default::default() },
                ..Default::default()
            });
            dof.debug_view = debug_view(param_or("debug", 0.0f32) as u32);
            effects.push(Box::new(dof));
        }
        DofMode::Simple => effects.push(Box::new(DepthOfFieldEffect::new(DepthOfFieldOptions { focus_distance: focus, focus_range: 15.0, max_blur: 24.0 }))),
        DofMode::Off => {}
    }
    // The TS page shows the chain's last image as it is, clamped: no exposure, no curve and no
    // sRGB encode (its shader's values are display values). The same here, on a non-sRGB surface.
    let tonemap = ToneMapEffect::new(ToneMapOptions { tonemapper: ToneMapper::None, encode_srgb: false, ..Default::default() });
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);

    log::info!("Kansei — Depth of Field (WASM) ready, {:?}", renderer.presentation_format());
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, controls, volume, mode, scroll_buffer, scroll: 0.0, time }));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| {
        let mut st = state.borrow_mut();
        let st = &mut *st;
        frame.resize(&mut st.renderer, &mut st.camera);
        st.frame(frame.dt);
    });
    Ok(())
}

/// The current view and lens, as the TS page's view JSON (plus the cinematic DoF's other settings).
#[wasm_bindgen]
pub fn view() -> String {
    with_state(|s| s.view_json()).unwrap_or_default()
}

/// Set one view or lens value by its view JSON key (`radius`, `azimuth` and `elevation` in
/// degrees, `targetX`/`Y`/`Z`, `fov`, `focusDistance`, `focusRange`, `maxBlur`, `fStop`,
/// `focalLength`, `blades`, `bladeRotation`, `samples`, `scatter` 0 or 1, `debug` 0 to 4).
#[wasm_bindgen]
pub fn set(key: &str, value: f32) {
    with_state(|s| s.set(key, value));
}
