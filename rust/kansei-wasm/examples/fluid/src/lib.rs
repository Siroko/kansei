use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::geometries::{InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{PARTICLE_BILLBOARD_WGSL, Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FixedStep;
use kansei_core::postprocessing::{PostProcessingVolume, effects::{
    DepthOfFieldEffect, DepthOfFieldOptions,
    FluidSurfaceEffect, FluidSurfaceOptions,
}};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::simulations::fluid::{
    DensityFieldOptions, FluidDensityField, FluidMarchingCubes, FluidSimulation, FluidSimulationOptions,
    MarchingCubesOptions, RaymarchingRenderable,
};
use kansei_wasm::{fetch_bytes, param, param_or, Canvas, Frame};


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

    let light = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normalize(v.world_normal), light), 0.0);
    let lit = base * (0.3 + ndotl * 0.7);
    return vec4<f32>(lit, 1.0);
}
"#;

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let (width, height) = canvas.size();
    let renderer = canvas.renderer(RendererConfig { sample_count: 4, ..Default::default() }).await;
    let format = renderer.presentation_format();

    // ?n=<particles> (50k by default). ?match=ts takes the TypeScript original's scene
    // (examples/index_fluid.html: its bounds, spread, gravity, damping and mouse force; the page
    // sets the last three), to compare the two side by side.
    let match_ts = param("match").as_deref() == Some("ts");
    // ── Particles ──
    // Spread in an ellipsoid centered on the sim bounds (~70% of bounds extent).
    let count = param_or("n", 50_000usize).max(1000);
    let (bounds_min, bounds_max) = if match_ts { ([-25.0f32, -8.0, -16.0], [25.0f32, 30.0, 16.0]) } else { ([-25.0, -8.0, -8.0], [25.0, 30.0, 16.0]) };
    let center = [0.0f32, 11.0, (bounds_min[2] + bounds_max[2]) / 2.0]; // (min+max)/2 of bounds
    // ~90% of bounds half-extent (more spread → less pressure); the original's 88%
    let half = if match_ts { [22.0f32, 17.0, 14.0] } else { [22.0f32, 17.0, 10.0] };
    let mut positions = vec![0.0f32; count * 4];
    let mut rng: u64 = 12345;
    for i in 0..count {
        loop {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let ux = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uy = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            let uz = (rng as f32 / u64::MAX as f32) * 2.0 - 1.0;
            if ux*ux + uy*uy + uz*uz <= 1.0 {
                positions[i*4]   = center[0] + ux * half[0];
                positions[i*4+1] = center[1] + uy * half[1];
                positions[i*4+2] = center[2] + uz * half[2];
                positions[i*4+3] = 1.0;
                break;
            }
        }
    }

    // ── Sim ──
    let mut sim = FluidSimulation::new(&renderer, FluidSimulationOptions {
        max_particles: count as u32, dimensions: 3, smoothing_radius: 1.0,
        pressure_multiplier: 46.5, near_pressure_multiplier: 20.0, density_target: 8.6,
        viscosity: 1.0, damping: 1.0, gravity: [0.0, -9.8, 0.0],
        mouse_force: 1600.0, substeps: 2, world_bounds_padding: 0.3,
        ..kansei_core::simulations::fluid::DEFAULT_OPTIONS
    }, &positions);
    sim.world_bounds_min = bounds_min;
    sim.world_bounds_max = bounds_max;
    sim.rebuild_grid();

    // ── Density field ──
    let density_field = FluidDensityField::new(&renderer, sim.positions_buffer().unwrap(),
        sim.world_bounds_min, sim.world_bounds_max,
        DensityFieldOptions { resolution: 128, kernel_scale: 0.6, ..Default::default() });

    // ── Marching cubes (compute only — render via standard Renderable) ──
    let marching_cubes = FluidMarchingCubes::new(&renderer, MarchingCubesOptions {
        max_triangles: 500_000,
        iso_level: 0.05,
    });
    let marching_cubes_bg = marching_cubes.create_bind_group(&renderer, &density_field.density_view);

    // The effect owns the sim, density field and marching cubes: density + MC compute + refraction
    let fluid = FluidSurfaceEffect::new(
        sim, density_field, marching_cubes, marching_cubes_bg,
        FluidSurfaceOptions::default(),
    );

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

            let stripe_data: [f32; 12] = [
                0.0, 0.0, 0.0, 1.0,   // color_a (black)
                1.0, 1.0, 1.0, 1.0,   // color_b (white)
                1.0, 1.0,             // thickness_a, thickness_b
                0.0, 0.0,             // padding
            ];
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
        Vec3::new(0.3, -1.0, 0.5).normalize(),
        Vec3::new(1.0, 1.0, 1.0), 2.0,
    );
    let light_scene_index = scene.add(SceneNode::Light(Light::Directional(sun)));

    // ── Camera ──
    let mut camera = Camera::new(45.0, 0.1, 1000.0, canvas.aspect());
    camera.set_position(0.0, 20.0, 75.0);
    camera.look_at(&Vec3::new(0.0, 3.0, 0.0));
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
                focus_distance: 42.0,
                focus_range: 37.0,
                max_blur: 7.0,
            })),
        ],
    );

    // Camera: back view aligned with long X axis (azimuth = π), radius 30
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 3.0, 0.0), 30.0);
    controls.set_azimuth(std::f32::consts::PI);
    let mouse = MouseVectors::from_canvas(canvas.element());

    let state = Rc::new(RefCell::new(State {
        renderer, scene, camera, controls, mouse, volume,
        mc_scene_index, dome_scene_index, light_scene_index,
        particle_scene_index,
        raymarch,
        width, height,
        particle_size: 0.15, show_particles: true, render_mode: 0, mc_iso_level: 0.05,
        use_batched_sim: true,
        sim_step: FixedStep::new(1.0 / 60.0).with_max_steps(4), sim_time_scale: 1.0,
        max_render_fps: 0.0, render_accumulator: 0.0,
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


        let frame_dt = render_dt.max(1.0 / 1000.0);
        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(frame_dt as f32);
        self.camera.aspect = self.width as f32 / self.height as f32;
        self.camera.update_projection_matrix();

        let view = self.camera.view_matrix.to_glam();
        let proj = self.camera.projection_matrix.to_glam();
        let inv_view = self.camera.inverse_view_matrix.to_glam();

        let mouse_ndc = [self.mouse.position.x, self.mouse.position.y];
        let mouse_dir = [self.mouse.direction.x, self.mouse.direction.y];
        let mouse_strength = self.mouse.strength.min(1.0);

        // Step fluid simulation (owned by FluidSurfaceEffect in the volume) at a fixed step, so
        // it evolves the same at any frame rate: each step simulates `step * sim_time_scale`.
        // `update_batched` encodes all substeps into one submit, so a slow frame's extra steps
        // cost GPU time but almost no CPU; past the step cap the backlog is dropped.
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        if let Some(fse) = self.volume.effect_mut::<FluidSurfaceEffect>()
        {
            fse.sim.set_camera_matrices(&view.to_cols_array(), &proj.to_cols_array(), &inv_view.to_cols_array(), &identity);
            let scale = self.sim_time_scale.clamp(0.1, 4.0);
            let scaled_dt = self.sim_step.step as f32 * scale;
            let steps = self.sim_step.advance(frame_dt * scale as f64);
            for _ in 0..steps {
                fse.step_simulation(scaled_dt, mouse_strength, mouse_ndc, mouse_dir, self.use_batched_sim);
            }
            LAST_SIM_STEPS.with(|n| n.set(steps));
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
            let fse = self.volume.effect_mut::<FluidSurfaceEffect>().unwrap();
            let splat = fse.sim.params.smoothing_radius;
            self.raymarch.render_frame(&self.renderer, &mut fse.density_field, &fse.sim, splat, &self.camera, wgpu::Color { r: 0.02, g: 0.02, b: 0.04, a: 1.0 });

        // ── Render mode 2/3: MC surface via standard Renderer ──
        // FluidSurfaceEffect handles density + MC compute + refraction composite.
        // DoF runs after. All orchestrated by render_with_postprocessing.
        } else {
            self.renderer.render_with_postprocessing(
                &mut self.scene, &mut self.camera, &mut self.volume,
            );
        }
    }
}

// ── JS interop ──
thread_local! { static LAST_SIM_STEPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) }; }
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) { GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } }); }
fn with_fluid<F: FnOnce(&mut FluidSurfaceEffect)>(f: F) {
    with_state(|s| {
        if let Some(fse) = s.volume.effect_mut::<FluidSurfaceEffect>() { f(fse); }
    });
}


/// The sim steps the last rendered frame ran, for the page's HUD (which asks mid-frame, while
/// the state is borrowed).
#[wasm_bindgen] pub fn last_sim_steps() -> u32 { LAST_SIM_STEPS.with(|n| n.get()) }
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
#[wasm_bindgen] pub fn set_stripe_params(
    r1: f32, g1: f32, b1: f32,
    r2: f32, g2: f32, b2: f32,
    thick_a: f32, thick_b: f32,
) {
    with_state(|s| {
        if let Some(idx) = s.dome_scene_index {
            if let Some(r) = s.scene.get_renderable_mut(idx) {
                let data: [f32; 12] = [
                    r1, g1, b1, 1.0,
                    r2, g2, b2, 1.0,
                    thick_a, thick_b,
                    0.0, 0.0,
                ];
                r.material.set_uniform_bindable(0, "Dome/StripeParams", &data);
                r.material_dirty = true;
            }
        }
    });
}
#[wasm_bindgen] pub fn set_light_direction(x: f32, y: f32, z: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.direction = Vec3::new(x, y, z).normalize();
        }
    });
}
#[wasm_bindgen] pub fn set_light_intensity(v: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.intensity = v;
        }
    });
}
#[wasm_bindgen] pub fn set_light_color(r: f32, g: f32, b: f32) {
    with_state(|s| {
        if let Some(Light::Directional(dl)) = s.scene.get_light_mut(s.light_scene_index) {
            dl.color = Vec3::new(r, g, b);
        }
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
