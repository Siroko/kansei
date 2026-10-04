use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{PARTICLE_BILLBOARD_WGSL, Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Mat4, Vec3};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FixedStep;
use kansei_core::postprocessing::{PostProcessingVolume, effects::{
    DepthOfFieldEffect, DepthOfFieldOptions,
    FluidSurfaceEffect, FluidSurfaceOptions,
}};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{
    ClockState, DensityFieldOptions, FluidDensityField, FluidMarchingCubes, FluidSimulation,
    FluidSimulationOptions, GlyphAttractor, MarchingCubesOptions, RaymarchingRenderable, SlotLayout, RetagParams,
    DEFAULT_OPTIONS};
use kansei_wasm::{fetch_bytes, Canvas, Frame};

const FONT: &[u8] = include_bytes!("../assets/L10-medium.arfont");

// ── Op-art stripe shader (matches engine bind group layout) ──
const STRIPE_WGSL: &str = r#"
// Group 1: Camera (engine layout — separate bindings)
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// Group 2: Mesh (engine layout — separate bindings with dynamic offsets)
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// Group 0: Material params
struct StripeParams {
    color_a: vec4<f32>,
    color_b: vec4<f32>,
    thickness_a: f32,
    thickness_b: f32,
    _pad0: f32,
    _pad1: f32,
    light_dir: vec4<f32>,   // xyz = direction light travels, w = intensity
    light_color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> params: StripeParams;

struct VOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
};

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> VOut {
    let world_pos = (world_matrix * vec4<f32>(position.xyz, 1.0)).xyz;
    let world_normal = normalize((normal_matrix * vec4<f32>(normal, 0.0)).xyz);
    var out: VOut;
    out.clip_pos = projection_matrix * view_matrix * vec4<f32>(world_pos, 1.0);
    out.world_pos = world_pos;
    out.world_normal = world_normal;
    return out;
}

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    let period = params.thickness_a + params.thickness_b;
    let t = ((v.world_pos.x % period) + period) % period;
    let in_a = t < params.thickness_a;
    let base = select(params.color_b.rgb, params.color_a.rgb, in_a);

    let light = normalize(-params.light_dir.xyz);
    let ndotl = max(dot(normalize(v.world_normal), light), 0.0);
    // intensity 2 reproduces the old fixed 0.3 + 0.7·ndotl
    let lit = base * (0.3 + ndotl * 0.35 * params.light_dir.w) * params.light_color.rgb;
    return vec4<f32>(lit, 1.0);
}
"#;

/// Everything that has to move together when the particle count changes. All of it is
/// derived from the tuned 80K @ h=1.0 setup by keeping the fluid volume constant:
/// radius ∝ count^(-1/3), density_target ∝ n/h (kernels integrate to 2n/h), near
/// pressure ∝ h, glyph budgets ∝ count. Pressure multiplier and sim time scale are
/// the measured stable pairs per count band (clock_fill_test, ≥7 s of sim time):
/// the stable substep shrinks ~1/sqrt(k), and near pressure — not k — provides the
/// incompressibility, so k can drop without changing the pool shape.
#[derive(Clone, Copy)]
pub struct SimTuning {
    pub count: u32, pub radius: f32, pub pressure: f32, pub near_pressure: f32,
    pub density_target: f32, pub viscosity: f32, pub time_scale: f32,
    pub per_slot: u32, pub emit_rate: u32, pub particle_size: f32, pub kernel_scale: f32,
}

/// The sim as tuned for `BASE_COUNT` particles at h = 1.0, which `tuning_for` scales to a count.
const BASE_COUNT: u32 = 80_000;
const BASE_OPTIONS: FluidSimulationOptions = FluidSimulationOptions {
    max_particles: BASE_COUNT, dimensions: 3, smoothing_radius: 1.0,
    pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6,
    viscosity: 1.0, damping: 1.0, gravity: [0.0, -9.8, 0.0],
    mouse_force: 1600.0, substeps: 2, world_bounds_padding: 2.0,
    ..DEFAULT_OPTIONS
};

/// The sim options for `tuning`: the base scaled to its count, at its band's pressure.
fn sim_options(tuning: &SimTuning) -> FluidSimulationOptions {
    FluidSimulationOptions {
        pressure_multiplier: tuning.pressure,
        ..BASE_OPTIONS.scaled_to_count(BASE_COUNT, tuning.count)
    }
}

pub fn tuning_for(count: u32) -> SimTuning {
    let ratio = count as f32 / BASE_COUNT as f32;
    let scaled = BASE_OPTIONS.scaled_to_count(BASE_COUNT, count);
    let radius = scaled.smoothing_radius;
    let (pressure, time_scale) = if count <= 100_000 { (46.5, 1.9) }
        else if count <= 200_000 { (12.0, 1.9) }
        else if count <= 300_000 { (12.0, 1.4) }
        else { (6.0, 1.0) };
    SimTuning {
        count, radius, pressure,
        near_pressure: scaled.near_pressure_multiplier,
        density_target: scaled.density_target,
        viscosity: scaled.viscosity,
        time_scale,
        per_slot: (2000.0 * ratio).round() as u32,
        emit_rate: (100.0 * ratio).round() as u32,
        particle_size: 0.15 * radius,
        kernel_scale: 0.6 / ratio,
    }
}

thread_local! { static TUNING: std::cell::Cell<Option<SimTuning>> = const { std::cell::Cell::new(None) }; }

/// The effective sim configuration as a JS object (for the page's tweakpane defaults).
#[wasm_bindgen]
pub fn sim_config() -> JsValue {
    let t = TUNING.with(|c| c.get()).unwrap_or_else(|| tuning_for(500_000));
    let o = js_sys::Object::new();
    let set = |k: &str, v: f64| { let _ = js_sys::Reflect::set(&o, &JsValue::from_str(k), &JsValue::from_f64(v)); };
    set("count", t.count as f64); set("radius", t.radius as f64); set("pressure", t.pressure as f64);
    set("nearPressure", t.near_pressure as f64); set("densityTarget", t.density_target as f64);
    set("viscosity", t.viscosity as f64); set("timeScale", t.time_scale as f64);
    set("perSlot", t.per_slot as f64); set("emitRate", t.emit_rate as f64);
    set("particleSize", t.particle_size as f64); set("kernelScale", t.kernel_scale as f64);
    o.into()
}

/// `count` = 0 selects the default (500K).
#[wasm_bindgen]
pub async fn start(canvas_id: &str, count: u32) -> Result<(), JsValue> {
    let tuning = tuning_for(if count == 0 { 500_000 } else { count.clamp(10_000, 2_000_000) });
    TUNING.with(|c| c.set(Some(tuning)));
    // Render at device resolution (CSS size × devicePixelRatio, capped at 2×, which the
    // runtime's Canvas does): the page stretches the canvas to the viewport, so a CSS-pixel
    // backing store gets upscaled by the browser on Retina displays and every edge looks aliased.
    let canvas = Canvas::find(canvas_id)?;
    let (width, height) = canvas.size();
    let renderer = canvas.renderer(RendererConfig { sample_count: 4, ..Default::default() }).await;
    let format = renderer.presentation_format();

    // ── Particles ──
    // Portrait tank for the stacked HH / MM / SS layout (rows of two 14-unit
    // glyphs). Whatever the count, the particles fill the same ~12500 units³
    // as the original 80K @ h=1.0 (see `tuning_for`): ~12.5 units deep on the
    // 50×20 floor, surface near y ≈ 2.5 (clock_fill_test, equal sim time).
    let count = tuning.count as usize;
    let center = [0.0f32, -1.5, 0.0];
    let half = [23.0f32, 7.5, 9.0]; // spawn box at rest density: x∈[-23,23], y∈[-9,6], z∈[-9,9]
    let mut positions = vec![0.0f32; count * 4];
    let mut rng: u64 = 12345;
    for i in 0..count {
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let ux = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let uy = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
        let uz = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
        positions[i*4]   = center[0] + ux * half[0];
        positions[i*4+1] = center[1] + uy * half[1];
        positions[i*4+2] = center[2] + uz * half[2];
        positions[i*4+3] = 1.0;
    }

    // ── Sim ──
    let mut sim = FluidSimulation::new(&renderer, sim_options(&tuning), &positions);
    // World bounds contain the spawn box (clock band + pool) plus margin for
    // falling/settling under gravity.
    sim.world_bounds_min = [-25.0, -10.0, -10.0];
    sim.world_bounds_max = [25.0, 60.0, 10.0];
    sim.rebuild_grid();

    // ── Glyph attractor: GPU tagging + clock-driven slot layout ──
    let font = FontAtlas::parse(FONT).expect("parse font");
    // Inside threshold 0.35 (< 0.5 = bolder strokes) and a 6-deep box: the
    // glyph holds ~2x the fluid of the plain outline at depth 4.
    let glyph_set = GlyphVolumeSet::for_clock_with_threshold(&font, 256, 8, 0.5, GLYPH_BOLD);
    let attractor = GlyphAttractor::new(&renderer, &glyph_set, count as u32);

    // 10-unit cells, 4 deep: holds ~400 particles per digit at rest density
    // (measured with kansei-native's glyph_form_test; 4-unit cells hold ~50).
    let mut slot_layout = SlotLayout::stacked_hh_mm_ss(CLOCK_CELL, CLOCK_DEPTH, ROW_SPACING);
    let slot_base_y: [f32; 8] = std::array::from_fn(|i| slot_layout.slots[i].world_min[1]);
    // Raise the stack so the SS row's box bottom sits ~1.5 units under the
    // pool surface (~2.5): SS center = -1.3·14 = -18.2, bottom = -25.2 → +29.
    let glyph_y = 28.0f32;
    apply_glyph_y(&mut slot_layout, &slot_base_y, glyph_y);
    let per_slot_count = tuning.per_slot; // 2000 @ 80K (~1800 fit a bold 14-unit, 6-deep stroke, glyph_form_test), ∝ count
    apply_budgets(&mut slot_layout, per_slot_count);
    let mut clock = ClockState::new();
    let (h0, m0, s0) = now_hms();
    let _ = clock.update(&mut slot_layout, h0, m0, s0);
    attractor.set_slots(&slot_layout);
    attractor.set_tags(&vec![-1i32; count]); // start all ordinary fluid

    // ── Density field ──
    let density_field = FluidDensityField::new(&renderer, sim.positions_buffer().unwrap(),
        sim.world_bounds_min, sim.world_bounds_max,
        DensityFieldOptions { resolution: 128, kernel_scale: tuning.kernel_scale, ..Default::default() }); // max-axis cells; ~0.55 units/cell on the 70-tall tank (192/256 looked the same)

    // ── Marching cubes (compute only — render via standard Renderable) ──
    let marching_cubes = FluidMarchingCubes::new(&renderer, MarchingCubesOptions {
        max_triangles: 500_000,
        iso_level: 0.05,
    });
    let marching_cubes_bg = marching_cubes.create_bind_group(&renderer, &density_field.density_view);

    // The effect owns the sim, density field and marching cubes: density + MC compute + refraction
    let mut fluid = FluidSurfaceEffect::new(
        sim, density_field, marching_cubes, marching_cubes_bg,
        FluidSurfaceOptions::default(),
    );
    // Surface field is splatted at the h=1.0 radius regardless of the sim
    // radius (0.543 < one 0.55-unit voxel would give a sparse, shattered
    // field); kernel scale divided by the 6.25x particle count keeps the
    // field values — and the tuned iso level — where they were at 80K.
    fluid.splat_radius = Some(SURFACE_SPLAT_RADIUS);

    // ── Raymarch mode: the density field raymarched offscreen and blitted to the canvas ──
    let raymarch = RaymarchingRenderable::new(&renderer, format, width, height,
        &fluid.density_field, fluid.sim.world_bounds_min, fluid.sim.world_bounds_max);

    // ── Build scene with standard Renderables ──
    let mut scene = Scene::new();

    // Dome mesh (GLB) — loaded via GLTFLoader, added as standard Renderable
    let dome_loaded = fetch_bytes("assets/dome.glb").await.ok().and_then(|bytes| GLTFLoader::load_glb(&bytes).ok());
    let mut dome_scene_index = None;
    if let Some(result) = dome_loaded {
        log::info!("Loaded dome: {} renderables", result.renderables.len());
        for gr in result.renderables {
            let s = 20.0;
            let mut opts = MaterialOptions::default();
            opts.cull_mode = CullMode::Front;

            let stripe_data = stripe_uniform(&DEFAULT_STRIPES, [-0.13, 0.22, 0.5], 4.0, [1.0, 1.0, 1.0]);
            let mut mat = Material::new(
                "Dome/Stripes", STRIPE_WGSL,
                vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
                opts,
            );
            mat.set_uniform_bindable(0, "Dome/StripeParams", &stripe_data);

            let mut r = Renderable::new(gr.geometry, mat);
            r.object.position = Vec3::new(gr.position.x, gr.position.y + 3.0, gr.position.z);
            r.object.rotation = gr.rotation;
            r.object.scale = Vec3::new(gr.scale.x * s, gr.scale.y * s, gr.scale.z * s);
            r.object.update_model_matrix();
            r.object.update_world_matrix(None);
            dome_scene_index = Some(scene.add(SceneNode::Renderable(r)));
        }
    }

    // MC surface in the GBuffer (colour + world normal), where FluidSurfaceEffect refracts it
    let mut mc_renderable = fluid.surface_renderable([0.77, 0.96, 1.0, 1.0]);
    mc_renderable.visible = false;
    let mc_scene_index = scene.add(SceneNode::Renderable(mc_renderable));

    // Light
    let sun = DirectionalLight::new(
        Vec3::new(-0.13, 0.22, 0.5).normalize(),
        Vec3::new(1.0, 1.0, 1.0), 4.0,
    );
    let light_scene_index = scene.add(SceneNode::Light(Light::Directional(sun)));

    // ── Camera ──
    let mut camera = Camera::new(45.0, 0.1, 1000.0, canvas.aspect());
    camera.set_position(0.0, 24.0, 95.0);
    camera.look_at(&Vec3::new(0.0, 22.0, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();

    // ── Particle renderable ──
    let positions_cb = fluid.sim.positions_as_compute_buffer(3).expect("sim should be initialized");
    let billboard_geo = PlaneGeometry::new(1.0, 1.0);
    let instanced_geo = InstancedGeometry::new(billboard_geo, count as u32, vec![positions_cb]);

    let particle_params: [f32; 12] = [
        0.15, -8.0, 8.0, 0.0,            // size, height_min, height_max, pad
        0.1, 0.3, 0.8, 1.0,              // color_low (blue)
        0.8, 0.95, 1.0, 1.0,             // color_high (white-blue)
    ];
    let mut particle_mat = Material::new(
        "Particles/Billboard", PARTICLE_BILLBOARD_WGSL,
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { cull_mode: CullMode::None, ..Default::default() },
    );
    particle_mat.set_uniform_bindable(0, "ParticleParams", &particle_params);

    let particle_renderable = Renderable::new(instanced_geo, particle_mat);
    let particle_scene_index = scene.add(SceneNode::Renderable(particle_renderable));

    // ── Post-processing: FluidSurface (compute + refraction) → DoF ──
    let volume = PostProcessingVolume::new(
        &renderer,
        vec![
            Box::new(fluid),
            Box::new(DepthOfFieldEffect::new(DepthOfFieldOptions {
                focus_distance: 90.0,
                focus_range: 37.0,
                max_blur: 7.0,
            })),
        ],
    );

    // Camera: back view aligned with long X axis (azimuth = π), radius wide
    // enough to see the full ~34-unit clock band (slots span x ∈ [-17, 17]).
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 22.0, 0.0), 95.0);
    controls.set_azimuth(std::f32::consts::PI);
    let mouse = MouseVectors::from_canvas(canvas.element());

    let state = Rc::new(RefCell::new(State {
        renderer, scene, camera, controls, mouse, volume,
        mc_scene_index, dome_scene_index, light_scene_index,
        particle_scene_index,
        raymarch,
        width, height,
        particle_size: tuning.particle_size, show_particles: true, render_mode: 0, mc_iso_level: 0.05,
        use_batched_sim: true,
        sim_step: FixedStep::new(1.0 / 60.0).with_max_steps(4), sim_time_scale: tuning.time_scale,
        max_render_fps: 0.0, render_accumulator: 0.0,
        frame_count: 0, frame_time_sum: 0.0,
        current_fps: 0.0, current_frame_ms: 0.0,
        attractor, slot_layout, clock, glyph_y, slot_base_y, stripes: DEFAULT_STRIPES,
        per_slot_count,
        cooldown_frames: 45,
        capture_scale: 1.15,
        capture_below: 45.0, // recruit columns reach from the top row down into the pool
        emit_height: 1.0,    // recruits appear in the gap just above their glyph and fall in
        emit_spread: 2.5,
        emit_rate: tuning.emit_rate, // per slot per frame: the budget over ~20 frames
        attr_stiffness: 90.0,
        attr_max_speed: 12.0, // ~0.2 units/step: must stay small vs. the stroke width
        attr_basin: 5.0,
        attr_target: 0.34,
        attr_drag: 3.0,
        audio_ctx: None,
        sound: false,
        last_second: -1,
    }));

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().render_frame(frame));

    log::info!("Kansei WASM — {} particles, [toggle via tweakpane]", count);
    Ok(())
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    volume: PostProcessingVolume,
    mc_scene_index: usize,
    dome_scene_index: Option<usize>,
    light_scene_index: usize,
    particle_scene_index: usize,
    raymarch: RaymarchingRenderable,
    width: u32, height: u32,
    particle_size: f32, show_particles: bool, render_mode: u32, mc_iso_level: f32,
    use_batched_sim: bool,
    /// The sim's fixed step (clamped to 1/240..1/20 s) and its cap of steps a frame.
    sim_step: FixedStep, sim_time_scale: f32,
    max_render_fps: f64, render_accumulator: f64,
    frame_count: u32, frame_time_sum: f64,
    current_fps: f64, current_frame_ms: f64,
    attractor: GlyphAttractor,
    slot_layout: SlotLayout,
    clock: ClockState,
    glyph_y: f32,
    slot_base_y: [f32; 8],
    stripes: [f32; 8],
    per_slot_count: u32,
    cooldown_frames: u32,
    capture_scale: f32,
    capture_below: f32,
    emit_height: f32,
    emit_spread: f32,
    emit_rate: u32,
    attr_stiffness: f32,
    attr_max_speed: f32,
    attr_basin: f32,
    attr_target: f32,
    attr_drag: f32,
    audio_ctx: Option<web_sys::AudioContext>,
    /// Beep each second; off until the page's sound box is ticked (a user gesture).
    sound: bool,
    last_second: i32,
}

impl State {
    fn render_frame(&mut self, frame: &Frame) {
        if let Some((width, height)) = frame.resized {
            frame.resize(&mut self.renderer, &mut self.camera);
            self.width = width;
            self.height = height;
            self.raymarch.resize(&self.renderer, width, height);
        }

        let frame_s = frame.dt as f64;
        self.render_accumulator += frame_s;
        let target_render_dt = if self.max_render_fps > 0.0 { 1.0 / self.max_render_fps } else { 0.0 };
        if target_render_dt > 0.0 && self.render_accumulator < target_render_dt { return; }
        let render_dt = if target_render_dt > 0.0 {
            let dt = self.render_accumulator; self.render_accumulator = 0.0; dt
        } else { frame_s };

        self.frame_time_sum += render_dt * 1000.0;
        self.frame_count += 1;
        if self.frame_count % 60 == 0 {
            let avg = self.frame_time_sum / 60.0;
            self.current_frame_ms = avg;
            self.current_fps = 1000.0 / avg;
            self.frame_time_sum = 0.0;
        }

        let frame_dt = render_dt.max(1.0 / 1000.0);
        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(frame_dt as f32);
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();

        let eye = *self.camera.position();
        let view = self.camera.view_matrix.to_glam();
        let proj = self.camera.projection_matrix.to_glam();
        let inv_view = self.camera.inverse_view_matrix.to_glam();
        let inv_vp = (proj * view).inverse();

        let mouse_ndc = [self.mouse.position.x, self.mouse.position.y];
        let mouse_dir = [self.mouse.direction.x, self.mouse.direction.y];
        let mouse_strength = self.mouse.strength.min(1.0);

        // Step fluid simulation (owned by FluidSurfaceEffect in the volume) at a fixed step, so
        // it evolves the same at any frame rate: each step simulates `step * sim_time_scale`.
        // `update_batched` encodes all substeps into one submit, so a slow frame's extra steps
        // cost GPU time but almost no CPU; past the step cap the backlog is dropped.
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        // Sim time advanced this frame (the attractor integrates over it).
        let mut sim_dt_frame = 0.0f32;
        if let Some(fse) = self.volume.effect_mut::<FluidSurfaceEffect>()
        {
            fse.sim.set_camera_matrices(&view.to_cols_array(), &proj.to_cols_array(), &inv_view.to_cols_array(), &identity);
            let scale = self.sim_time_scale.clamp(0.1, 4.0);
            let scaled_dt = self.sim_step.step as f32 * scale;
            let steps = self.sim_step.advance(frame_dt * scale as f64);
            for _ in 0..steps {
                fse.step_simulation(scaled_dt, mouse_strength, mouse_ndc, mouse_dir, self.use_batched_sim);
            }
            sim_dt_frame = steps as f32 * scaled_dt;
        }

        // ── Clock → GPU tagging → attractor (all GPU; no readback) ──────
        let (h, m, s) = now_hms();
        let changed = self.clock.update(&mut self.slot_layout, h, m, s);
        let changed_mask = ClockState::changed_mask(&changed);
        if !changed.is_empty() {
            self.attractor.set_slots(&self.slot_layout);
        }
        if s as i32 != self.last_second {
            if self.sound {
                self.beep_for_change(h, m, s);
            }
            self.last_second = s as i32;
        }
        self.attractor.set_params(sim_dt_frame, self.attr_stiffness, self.attr_max_speed, self.attr_basin, self.attr_target, self.attr_drag);

        if let Some(fse) = self.volume.effect::<FluidSurfaceEffect>()
        {
            if let (Some(pos_buf), Some(vel_buf)) = (fse.sim.positions_buffer(), fse.sim.velocities_buffer()) {
                let mut enc = self.renderer.device().create_command_encoder(&Default::default());
                self.attractor.retag(&mut enc, pos_buf, vel_buf, &RetagParams {
                    changed_mask, per_slot_count: self.per_slot_count, cooldown_frames: self.cooldown_frames,
                    capture_scale: self.capture_scale, capture_below: self.capture_below,
                    emit_height: self.emit_height, emit_spread: self.emit_spread, emit_rate: self.emit_rate,
                });
                self.attractor.dispatch(&mut enc, pos_buf, vel_buf);
                self.renderer.queue().submit(std::iter::once(enc.finish()));
            }
        }

        // ── Render mode 0: Particles (standard engine path via InstancedGeometry) ──
        if self.render_mode == 0 {
            if let Some(r) = self.scene.get_renderable(self.particle_scene_index) {
                if let Some(buf) = r.material.bindable_buffer(0) {
                    let params: [f32; 12] = [
                        self.particle_size, -8.0, 8.0, 0.0,
                        0.1, 0.3, 0.8, 1.0,
                        0.8, 0.95, 1.0, 1.0,
                    ];
                    self.renderer.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&params));
                }
            }
            self.renderer.render(&mut self.scene, &mut self.camera);

        // ── Render mode 1: Raymarch (custom compute + blit) ──
        } else if self.render_mode == 1 {
            let output = self.renderer.surface().unwrap().get_current_texture().unwrap();
            let canvas_view = output.texture.create_view(&Default::default());
            let mut encoder = self.renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("WasmFluid/Raymarch"),
            });

            let fse = self.volume.effect_mut::<FluidSurfaceEffect>().unwrap();
            fse.density_field.update_with_encoder(&mut encoder,
                fse.sim.world_bounds_min, fse.sim.world_bounds_max,
                fse.sim.particle_count(), SURFACE_SPLAT_RADIUS);
            self.raymarch.bounds_min = fse.sim.world_bounds_min;
            self.raymarch.bounds_max = fse.sim.world_bounds_max;

            // Clear offscreen color + depth (raymarch reads depth to know geometry)
            {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: self.raymarch.input_color_view(), resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 }),
                            store: wgpu::StoreOp::Store },
                    })],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: self.raymarch.input_depth_view(),
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }), ..Default::default()
                });
            }

            self.raymarch.render(&mut encoder, &canvas_view, &Mat4::from(inv_vp),
                [eye.x, eye.y, eye.z], self.width, self.height);
            self.renderer.submit(std::iter::once(encoder.finish()));
            output.present();

        // ── Render mode 2/3: MC surface via standard Renderer ──
        // FluidSurfaceEffect handles density + MC compute + refraction composite.
        // DoF runs after. All orchestrated by render_with_postprocessing.
        } else {
            self.renderer.render_with_postprocessing(
                &mut self.scene, &mut self.camera, &mut self.volume,
            );
        }
    }

    /// Beep once per second: hour rollover (m==0 && s==0) highest, minute
    /// rollover (s==0) higher, ordinary second base.
    fn beep_for_change(&self, _h: u32, m: u32, s: u32) {
        let Some(ctx) = self.audio_ctx.as_ref() else { return; };
        let freq: f32 = if m == 0 && s == 0 { 880.0 } else if s == 0 { 660.0 } else { 440.0 };
        let (Ok(osc), Ok(gain)) = (ctx.create_oscillator(), ctx.create_gain()) else { return; };
        osc.set_type(web_sys::OscillatorType::Sine);
        osc.frequency().set_value(freq);
        let now = ctx.current_time();
        gain.gain().set_value(0.0001);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.12, now + 0.01);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.0001, now + 0.13);
        let _ = osc.connect_with_audio_node(&gain);
        let _ = gain.connect_with_audio_node(&ctx.destination());
        let _ = osc.start_with_when(now);
        let _ = osc.stop_with_when(now + 0.14);
    }
}

fn now_hms() -> (u32, u32, u32) {
    let d = js_sys::Date::new_0();
    (d.get_hours(), d.get_minutes(), d.get_seconds())
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }

/// Surface density-field splat radius and kernel scale (see FluidSurfaceEffect::splat_radius).
const SURFACE_SPLAT_RADIUS: f32 = 1.0;
const CLOCK_CELL: f32 = 14.0;
const CLOCK_DEPTH: f32 = 6.0;
/// Row pitch of the stacked layout as a multiple of the cell (0.3·cell gap).
const ROW_SPACING: f32 = 1.3;
/// Atlas-alpha inside threshold: 0.5 = true outline, lower = bolder.
const GLYPH_BOLD: f32 = 0.35;
/// Fraction of a digit's budget the `:` slots get (their stroke area is
/// under half a digit's).
const COLON_BUDGET_SCALE: f32 = 0.45;

/// Shift the whole stack vertically by `offset` from its layout-time position.
fn apply_glyph_y(layout: &mut SlotLayout, base_y: &[f32; 8], offset: f32) {
    for (s, b) in layout.slots.iter_mut().zip(base_y.iter()) {
        s.world_min[1] = b + offset;
    }
}
/// Digits use the global per-slot count (budget 0); colons get a smaller cap.
fn apply_budgets(layout: &mut SlotLayout, per_slot: u32) {
    for s in layout.slots.iter_mut() {
        s.budget = if s.glyph_id == 10 { ((per_slot as f32) * COLON_BUDGET_SCALE) as u32 } else { 0 };
    }
}
fn with_fluid<F: FnOnce(&mut FluidSurfaceEffect)>(f: F) {
    with_state(|s| {
        if let Some(fse) = s.volume.effect_mut::<FluidSurfaceEffect>() { f(fse); }
    });
}

#[wasm_bindgen] pub fn get_fps() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            rc.try_borrow().map(|s| s.current_fps).unwrap_or(0.0)
        } else { 0.0 }
    })
}
#[wasm_bindgen] pub fn get_frame_time() -> f64 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            rc.try_borrow().map(|s| s.current_frame_ms).unwrap_or(0.0)
        } else { 0.0 }
    })
}

#[wasm_bindgen] pub fn set_pressure(v: f32) { with_fluid(|f| f.sim.params.pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_near_pressure(v: f32) { with_fluid(|f| f.sim.params.near_pressure_multiplier = v); }
#[wasm_bindgen] pub fn set_density_target(v: f32) { with_fluid(|f| f.sim.params.density_target = v); }
#[wasm_bindgen] pub fn set_viscosity(v: f32) { with_fluid(|f| f.sim.params.viscosity = v); }
#[wasm_bindgen] pub fn set_damping(v: f32) { with_fluid(|f| f.sim.params.damping = v); }
#[wasm_bindgen] pub fn set_gravity_y(v: f32) { with_fluid(|f| f.sim.params.gravity[1] = v); }
#[wasm_bindgen] pub fn set_radial_gravity(enabled: bool) {
    with_fluid(|f| f.sim.params.radial_gravity = enabled);
}
#[wasm_bindgen] pub fn set_gravity_center(x: f32, y: f32, z: f32) {
    with_fluid(|f| f.sim.params.gravity_center = [x, y, z]);
}
#[wasm_bindgen] pub fn set_mouse_force(v: f32) { with_fluid(|f| f.sim.params.mouse_force = v); }
#[wasm_bindgen] pub fn set_mouse_radius(v: f32) { with_fluid(|f| f.sim.params.mouse_radius = v); }
#[wasm_bindgen] pub fn set_particle_size(v: f32) { with_state(|s| s.particle_size = v); }
#[wasm_bindgen] pub fn set_substeps(v: u32) { with_fluid(|f| f.sim.params.substeps = v); }
#[wasm_bindgen] pub fn set_show_particles(v: bool) {
    with_state(|s| {
        s.show_particles = v;
        s.render_mode = if v { 0 } else { 2 };
        if let Some(r) = s.scene.get_renderable_mut(s.particle_scene_index) {
            r.visible = v;
        }
    });
}
#[wasm_bindgen] pub fn set_render_mode(v: u32) {
    with_state(|s| {
        s.render_mode = v.min(3);
        s.show_particles = s.render_mode == 0;
        if let Some(fse) = s.volume.effect_mut::<FluidSurfaceEffect>() {
            fse.marching_cubes.set_use_classic(s.render_mode == 3);
        }
        if let Some(r) = s.scene.get_renderable_mut(s.mc_scene_index) {
            r.visible = s.render_mode >= 2;
        }
        if let Some(r) = s.scene.get_renderable_mut(s.particle_scene_index) {
            r.visible = s.render_mode == 0;
        }
    });
}
#[wasm_bindgen] pub fn set_mc_iso_level(v: f32) {
    with_state(|s| {
        let iso = v.max(0.0);
        s.mc_iso_level = iso;
        s.raymarch.surface_renderer_mut().density_threshold = iso;
        if let Some(fse) = s.volume.effect_mut::<FluidSurfaceEffect>() {
            fse.marching_cubes.set_iso_level(iso);
        }
    });
}
#[wasm_bindgen] pub fn set_mc_resolution(v: u32) {
    with_fluid(|f| {
        use kansei_core::simulations::fluid::MarchingCubesGridSizing;
        let mut p = f.marching_cubes.params();
        p.grid_sizing = if v == 0 { MarchingCubesGridSizing::FromSource } else { MarchingCubesGridSizing::MaxAxis(v) };
        f.marching_cubes.set_params(p);
    });
}
#[wasm_bindgen] pub fn set_use_batched_sim(v: bool) { with_state(|s| s.use_batched_sim = v); }
#[wasm_bindgen] pub fn set_sim_dt_step(v: f32) { with_state(|s| s.sim_step.step = v.max(1.0 / 1000.0).clamp(1.0 / 240.0, 1.0 / 20.0) as f64); }
#[wasm_bindgen] pub fn set_sim_time_scale(v: f32) { with_state(|s| s.sim_time_scale = v.max(0.01)); }
#[wasm_bindgen] pub fn set_max_render_fps(v: f32) { with_state(|s| s.max_render_fps = v.max(0.0) as f64); }
#[wasm_bindgen] pub fn set_density_scale(v: f32) { with_state(|s| s.raymarch.surface_renderer_mut().density_scale = v); }
#[wasm_bindgen] pub fn set_density_threshold(v: f32) {
    with_state(|s| { s.raymarch.surface_renderer_mut().density_threshold = v; s.mc_iso_level = v.max(0.0); });
}
#[wasm_bindgen] pub fn set_absorption(v: f32) { with_state(|s| s.raymarch.surface_renderer_mut().absorption = v); }
#[wasm_bindgen] pub fn set_step_count(v: u32) { with_state(|s| s.raymarch.surface_renderer_mut().step_count = v); }
#[wasm_bindgen] pub fn set_kernel_scale(v: f32) { with_fluid(|f| f.density_field.kernel_scale = v); }
#[wasm_bindgen] pub fn set_bounds(min_x: f32, min_y: f32, min_z: f32, max_x: f32, max_y: f32, max_z: f32) {
    with_fluid(|f| {
        f.sim.world_bounds_min = [min_x, min_y, min_z];
        f.sim.world_bounds_max = [max_x, max_y, max_z];
        f.sim.rebuild_grid();
    });
}
/// Dome stripe settings: color_a rgb, color_b rgb, thickness_a, thickness_b.
const DEFAULT_STRIPES: [f32; 8] = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Pack stripe settings + key light into the dome's `StripeParams` uniform.
fn stripe_uniform(st: &[f32; 8], light_dir: [f32; 3], intensity: f32, light_color: [f32; 3]) -> [f32; 20] {
    [
        st[0], st[1], st[2], 1.0,
        st[3], st[4], st[5], 1.0,
        st[6], st[7], 0.0, 0.0,
        light_dir[0], light_dir[1], light_dir[2], intensity,
        light_color[0], light_color[1], light_color[2], 1.0,
    ]
}

/// Push the scene's directional light into everything that shades with it
/// but isn't an engine-lit material: the dome stripes and the fluid surface.
fn sync_light(s: &mut State) {
    let (dir, intensity, color) = match s.scene.get_light_mut(s.light_scene_index) {
        Some(Light::Directional(dl)) => ([dl.direction.x, dl.direction.y, dl.direction.z], dl.intensity, [dl.color.x, dl.color.y, dl.color.z]),
        _ => return,
    };
    if let Some(idx) = s.dome_scene_index {
        if let Some(r) = s.scene.get_renderable_mut(idx) {
            let data = stripe_uniform(&s.stripes, dir, intensity, color);
            r.material.set_uniform_bindable(0, "Dome/StripeParams", &data);
            r.material_dirty = true;
        }
    }
    if let Some(fse) = s.volume.effect_mut::<FluidSurfaceEffect>()
    {
        fse.options.light_direction = dir;
        fse.options.light_intensity = intensity;
        fse.options.light_color = color;
    }
}

#[wasm_bindgen] pub fn set_stripe_params(
    r1: f32, g1: f32, b1: f32,
    r2: f32, g2: f32, b2: f32,
    thick_a: f32, thick_b: f32,
) {
    with_state(|s| {
        s.stripes = [r1, g1, b1, r2, g2, b2, thick_a, thick_b];
        sync_light(s);
    });
}
#[wasm_bindgen] pub fn set_light_direction(x: f32, y: f32, z: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.direction = Vec3::new(x, y, z).normalize();
        }
        sync_light(s);
    });
}
#[wasm_bindgen] pub fn set_light_intensity(v: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.intensity = v;
        }
        sync_light(s);
    });
}
#[wasm_bindgen] pub fn set_light_color(r: f32, g: f32, b: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.color = Vec3::new(r, g, b);
        }
        sync_light(s);
    });
}
#[wasm_bindgen] pub fn set_dof_focus_distance(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effect_mut::<DepthOfFieldEffect>() {
            d.options.focus_distance = v;
        }
    });
}
#[wasm_bindgen] pub fn set_dof_focus_range(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effect_mut::<DepthOfFieldEffect>() {
            d.options.focus_range = v;
        }
    });
}
#[wasm_bindgen] pub fn set_dof_max_blur(v: f32) {
    with_state(|s| {
        if let Some(d) = s.volume.effect_mut::<DepthOfFieldEffect>() {
            d.options.max_blur = v;
        }
    });
}
#[wasm_bindgen] pub fn set_transmission_params(
    ior: f32, chromatic_aberration: f32, tint_strength: f32,
    fresnel_power: f32, roughness: f32, thickness: f32,
    r: f32, g: f32, b: f32,
) {
    with_fluid(|f| {
        f.options.ior = ior;
        f.options.chromatic_aberration = chromatic_aberration;
        f.options.tint_strength = tint_strength;
        f.options.fresnel_power = fresnel_power;
        f.options.roughness = roughness;
        f.options.thickness = thickness;
        f.options.color = [r, g, b, 1.0];
    });
}

#[wasm_bindgen] pub fn set_attr_stiffness(v: f32) { with_state(|s| s.attr_stiffness = v); }
#[wasm_bindgen] pub fn set_attr_max_speed(v: f32) { with_state(|s| s.attr_max_speed = v); }
#[wasm_bindgen] pub fn set_attr_basin(v: f32) { with_state(|s| s.attr_basin = v); }
#[wasm_bindgen] pub fn set_attr_target(v: f32) { with_state(|s| s.attr_target = v); }
#[wasm_bindgen] pub fn set_attr_drag(v: f32) { with_state(|s| s.attr_drag = v); }
#[wasm_bindgen] pub fn set_per_slot_count(v: u32) { with_state(|s| { s.per_slot_count = v; apply_budgets(&mut s.slot_layout, v); s.attractor.set_slots(&s.slot_layout); }); }
#[wasm_bindgen] pub fn set_capture_scale(v: f32) { with_state(|s| s.capture_scale = v); }
#[wasm_bindgen] pub fn set_capture_below(v: f32) { with_state(|s| s.capture_below = v); }
#[wasm_bindgen] pub fn set_emit_height(v: f32) { with_state(|s| s.emit_height = v); }
#[wasm_bindgen] pub fn set_emit_spread(v: f32) { with_state(|s| s.emit_spread = v); }
#[wasm_bindgen] pub fn set_emit_rate(v: u32) { with_state(|s| s.emit_rate = v); }
#[wasm_bindgen] pub fn set_glyph_y(v: f32) { with_state(|s| { s.glyph_y = v; let base = s.slot_base_y; apply_glyph_y(&mut s.slot_layout, &base, v); s.attractor.set_slots(&s.slot_layout); }); }
#[wasm_bindgen] pub fn set_cooldown_frames(v: u32) { with_state(|s| s.cooldown_frames = v); }
/// Turn the per-second beep on or off. Call it from a user gesture: the first `true` creates the
/// `AudioContext`, which browsers only let start from one.
#[wasm_bindgen] pub fn set_sound(on: bool) {
    with_state(|s| {
        s.sound = on;
        if on && s.audio_ctx.is_none() {
            s.audio_ctx = web_sys::AudioContext::new().ok();
        }
        if let Some(c) = s.audio_ctx.as_ref() {
            let _ = if on { c.resume() } else { c.suspend() };
        }
    });
}
