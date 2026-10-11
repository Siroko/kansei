//! The page itself (`start` and the exports the panel calls): the renderer, the chain, the two
//! scenes, the pointer and the settings applied.

use crate::{input, settings, studio, water};

use std::cell::RefCell;
use std::rc::Rc;

use glam::{Mat4, Vec3 as GVec3, Vec4Swizzles};
use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, LodView};
use kansei_core::controls::CameraControls;
use kansei_core::geometries::Geometry;
use kansei_core::gi::{SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::Light;
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Material, MaterialOptions, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100, BloomEffect, BloomOptions, FluidSurfaceEffect, HeightFogEffect, HeightFogLayer, SSAOEffect, SSAOOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, ToneMapper,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::{
    RtCausticsEffect, RtCausticsOptions, RtCausticsView, RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiResolution, RtGiShadows, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions,
    RtShadowsEffect, RtSurface, RtTraceResolution, SceneRtGridOptions,
};
use kansei_core::shadows::CascadedShadowOptions;
use kansei_wasm::{fetch_bytes, param_or, set_text, Canvas, Frame};

use input::{Drag, Pointer};
use settings::Settings;
use studio::Studio;
use water::Water;

/// The studio's box for voxel GI (m): 12 x 4 x 12 m round the subject.
const BOUNDS_MIN: [f32; 3] = [-6.0, -0.2, -6.0];
const BOUNDS_MAX: [f32; 3] = [6.0, 3.8, 6.0];

/// The horizon's fade: metres from the camera before it starts, and its extinction a metre past
/// them (the floor 30 m off about half faded).
const HORIZON_START: f32 = 9.0;
const HORIZON_DENSITY: f32 = 0.03;

/// The scene's pixels under the automatic render scale (`render_scale=0`): about 1520 x 860.
const AUTO_PIXELS: f32 = 1.3e6;

/// The lit white floor's level of display white under `auto_ev` (the Khronos neutral tone map
/// keeps it, AgX's shoulder greys it a little).
const AUTO_LEVEL: f32 = 0.92;

/// Never drawn: its vertices land outside the clip volume, so it costs a draw and nothing more.
/// For renderables that exist only for the grid (the water's coarse copy).
const HIDDEN_WGSL: &str = r#"
@vertex
fn vertex_main(@location(0) position: vec4f) -> @builtin(position) vec4f {
    return vec4f(0.0, 0.0, 2.0, 1.0);
}
@fragment
fn fragment_main() -> @location(0) vec4f {
    return vec4f(0.0);
}
"#;

/// A renderable only the ray tracing grid sees: no camera, no shadow map, no voxel GI.
pub(crate) fn grid_only(geometry: Geometry, surface: RtSurface) -> Renderable {
    let mut material = Material::new("Caustics/GridOnly", HIDDEN_WGSL, Vec::new(), MaterialOptions::default());
    material.options.cull_mode = kansei_core::materials::CullMode::None;
    let mut r = Renderable::new(geometry, material);
    r.cast_shadow = false;
    r.rt = Some(surface);
    r
}

/// An effect that can be switched off (skipped, costing nothing) at run time.
struct Switch<E> {
    effect: E,
    on: bool,
}

impl<E: PostProcessingEffect + 'static> PostProcessingEffect for Switch<E> {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &kansei_core::renderers::GBuffer, camera: &Camera) {
        self.effect.initialize(device, gbuffer, camera);
    }
    #[allow(clippy::too_many_arguments)]
    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, gbuffer: &kansei_core::renderers::GBuffer, input: &wgpu::TextureView, depth: &wgpu::TextureView, output: &wgpu::TextureView, camera: &Camera, width: u32, height: u32) {
        self.effect.render(device, queue, encoder, gbuffer, input, depth, output, camera, width, height);
    }
    fn resize(&mut self, width: u32, height: u32, gbuffer: &kansei_core::renderers::GBuffer) {
        self.effect.resize(width, height, gbuffer);
    }
    fn is_active(&self) -> bool {
        self.on && self.effect.is_active()
    }
    fn wants_jitter(&self) -> bool {
        self.on && self.effect.wants_jitter()
    }
    fn upscales_to_display(&self) -> bool {
        self.on && self.effect.upscales_to_display()
    }
    fn destroy(&mut self) {
        self.effect.destroy();
    }
    fn name(&self) -> &'static str {
        self.effect.name()
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// The chain: the water's surface (the box), SSAO (when chosen), the ray-traced direct light and
/// shadows, the hybrid RT GI, the caustics over `caustics_rect` (x, z), the reflections and the
/// glass, TAA, bloom and the tone map. `apply` sets them to the settings.
fn build_effects(renderer: &Renderer, surface: Option<FluidSurfaceEffect>, sky: &wgpu::Buffer, caustics_rect: ([f32; 2], [f32; 2])) -> Vec<Box<dyn PostProcessingEffect>> {
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(surface) = surface {
        effects.push(Box::new(surface));
    }
    // (over the flat ambient only: the direct light comes after it)
    effects.push(Box::new(Switch { effect: SSAOEffect::new(SSAOOptions::default()), on: false }));
    let scene_gi = renderer.voxel_gi().expect("the studio enables voxel GI");
    let grid = renderer.rt_grid().expect("the studio builds the grid");
    effects.push(Box::new(RtShadowsEffect::new(grid.handle())));
    // the white sky's light and occlusion, the floor's and the walls' bounce (the materials add no
    // ambient of their own while it runs)
    let mut gi = RtDiffuseGiEffect::with_volume(scene_gi.volume(), grid.handle(), RtDiffuseGiOptions { max_distance: 30.0, ambient: 0.0, shadows: RtGiShadows::Rays, ..Default::default() });
    gi.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    gi.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
    gi.set_sky_lighting(Some(sky));
    effects.push(Box::new(gi));
    let (lo, hi) = caustics_rect;
    let caustics = RtCausticsEffect::new(renderer.device(), grid.handle(), RtCausticsOptions { min: lo, max: hi, texel: 0.012, height_tolerance: 0.15, ..Default::default() });
    let (map, lookup) = caustics.map();
    let caustics_map = (map.clone(), lookup.clone());
    effects.push(Box::new(Switch { effect: caustics, on: true }));
    let mut reflections = RtReflectionsEffect::with_volume(scene_gi.volume(), grid.handle(), RtReflectionsOptions { resolution: RtTraceResolution::Half, max_distance: 30.0, ..Default::default() });
    reflections.screen_hits = true;
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    reflections.set_glass(Some(RtGlass { reflections_in_trace: true, ..Default::default() }));
    reflections.set_caustics(Some((&caustics_map.0, &caustics_map.1)));
    reflections.set_sky_lighting(Some(sky));
    effects.push(Box::new(reflections));
    // the far floor faded into the backdrop: the studio's horizon gone (a uniform layer from
    // `HORIZON_START` on, its colour the backdrop's, set by `apply`)
    let mut horizon = HeightFogEffect::new(HeightFogLayer { density: HORIZON_DENSITY, height_falloff: 1e-4, height: 0.0 });
    horizon.start_distance = HORIZON_START;
    effects.push(Box::new(horizon));
    effects.push(Box::new(Switch { effect: TemporalAAEffect::new(TemporalAAOptions::default()), on: true }));
    effects.push(Box::new(Switch { effect: BloomEffect::new(BloomOptions::default()), on: true }));
    effects.push(Box::new(ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()))));
    effects
}

/// The white sky as `SkyLighting` (`atmosphere::SKY_LIGHTING_WGSL`) for the rays that leave the
/// grid: radiance `up` overhead, a little less toward the horizon, the floor's `down` below.
fn sky_lighting(up: f32, down: f32) -> [f32; 60] {
    // L(d) = a + b d.y, as order-2 SH (the constant and vertical terms)
    let (a, b) = ((up + down) * 0.5, (up - down) * 0.5);
    let mut out = [0.0f32; 60];
    for c in 0..3 {
        out[c] = a * 3.544_908;
        out[4 + c] = b * 2.046_653;
    }
    out
}

fn tonemapper(name: &str) -> ToneMapper {
    match name {
        "aces" => ToneMapper::AcesFitted,
        "agx_punchy" => ToneMapper::AgXPunchy,
        "neutral" => ToneMapper::KhronosNeutral,
        "none" => ToneMapper::None,
        _ => ToneMapper::AgX,
    }
}

/// The first of `urls` that loads (the page's `assets/` links the gi-box example's files; the
/// published site has them there, beside this page's folder).
async fn fetch_first(urls: &[String]) -> Option<Vec<u8>> {
    for url in urls {
        match fetch_bytes(url).await {
            Ok(b) if b.len() > 1024 => return Some(b),
            Ok(_) => log::warn!("nothing at {url}"),
            Err(e) => log::warn!("nothing at {url}: {e:?}"),
        }
    }
    None
}

/// `assets/<name>`, then the gi-box example's.
fn asset_urls(name: &str) -> [String; 2] {
    [format!("assets/{name}"), format!("../gi-box/assets/{name}")]
}

/// The glass Stanford dragon (CC-BY-NC-4.0, credited on the page), 2 m long, on the floor: the
/// full scan (`dragon=full`, the default: 871k triangles, a 24 MB download) or its 19k-triangle
/// decimation (`dragon=light`), drawn by its cluster LOD (full detail up close, `dragon_lod`) or
/// as the whole mesh. The grid holds a copy of it of its own (`grid_only`): a cut of its cluster
/// graph at a uniform `dragon_rt_mm` of error, compacted, so the rays inside the glass and the
/// caustics' photons see about the shape the camera does at a fixed cost (the grid's own cut of
/// the full scan cost 4-8 ms of glass pass at 1600 x 813 and left the decimation 900 triangles).
/// Returns the dragon's renderable, the grid's copy's and its box.
async fn add_dragon(scene: &mut Scene, s: &Settings) -> Option<(usize, usize, (GVec3, GVec3))> {
    let loaded = if s.dragon == "light" {
        GLTFLoader::load_glb(&fetch_first(&asset_urls("stanford_dragon_pbr.glb")).await?).ok()?
    } else {
        let json = fetch_first(&asset_urls("scene.gltf")).await?;
        let bin = fetch_first(&asset_urls("scene.bin")).await?;
        GLTFLoader::load_gltf_with_buffers(&json, vec![bin]).ok()?
    };
    let geometry = loaded.merged_geometry("Dragon").fit(GVec3::new(2.0, f32::INFINITY, 2.0));
    let mesh = ClusterMesh::build(&geometry, &ClusterOptions::default());
    // the grid's copy: every cluster within the error (metres, seen "orthographically": one
    // pixel a metre), its vertices compacted
    let view = LodView { eye: GVec3::ZERO, pixels_per_radian: 1.0, near: 1.0, threshold: s.dragon_rt_mm.max(0.1) * 1e-3, orthographic: true };
    let cut = mesh.cut_geometry("Dragon/Grid", &view);
    let mut remap = vec![u32::MAX; cut.vertices.len()];
    let (mut vertices, mut indices) = (Vec::new(), Vec::with_capacity(cut.indices.len()));
    for &i in &cut.indices {
        let r = &mut remap[i as usize];
        if *r == u32::MAX {
            *r = vertices.len() as u32;
            vertices.push(cut.vertices[i as usize]);
        }
        indices.push(*r);
    }
    log::info!("dragon: {} triangles drawn, {} in the ray tracing grid ({} mm)", geometry.indices.len() / 3, indices.len() / 3, s.dragon_rt_mm);
    let tint = water::glass_tint(s.glass_tint);
    let mut dragon = Renderable::new(geometry, Material::standard_lit("Dragon", &StandardLitOptions::glass(tint, s.glass_ior, s.glass_rough)));
    if s.dragon_lod {
        dragon.clusters = Some(ClusterLod::new(mesh));
    }
    dragon.cast_shadow = false;
    dragon.object.rotation.y = DRAGON_TURN;
    let (lo, hi) = dragon.geometry.bounds();
    let placed = kansei_core::rt::transform_box(&Mat4::from_rotation_y(DRAGON_TURN), lo, hi);
    let mut copy = grid_only(Geometry::new("Dragon/Grid", vertices, indices), RtSurface::glass(tint).with_smooth_normals());
    copy.object.rotation.y = DRAGON_TURN;
    Some((scene.add(SceneNode::Renderable(dragon)), scene.add(SceneNode::Renderable(copy)), placed))
}

/// The dragon's turn on the floor (radians): its head toward the camera's left.
const DRAGON_TURN: f32 = 0.5;

/// The renderer's profile over the last second (`stats`).
#[derive(Default)]
struct Stats {
    frames: u32,
    since: f64,
    frame_ms: f64,
    passes: Vec<(&'static str, f64)>,
    cpu: Vec<(&'static str, f64)>,
    gpu_ms: f64,
    gpu_span_ms: f64,
}

struct State {
    canvas: Canvas,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    pointer: Pointer,
    volume: PostProcessingVolume,
    studio: Studio,
    water: Option<Water>,
    /// the dragon, its copy in the grid, its box
    dragon: Option<(usize, usize, (GVec3, GVec3))>,
    sky_lighting: wgpu::Buffer,
    settings: Settings,
    applied: Option<Settings>,
    stats: Option<Stats>,
    time: f32,
    last: f64,
    frame: u32,
    fps: f32,
    /// where the mouse pushes the water now
    pushing: Option<GVec3>,
}

impl State {
    /// The camera's start for the scene.
    fn home(&mut self) {
        let (target, radius, azimuth, elevation) = if self.settings.scene == "dragon" { (Vec3::new(0.0, 0.5, 0.0), 4.0, 0.35, 0.26) } else { (Vec3::new(0.0, 0.1, 0.25), 4.3, 0.2, 0.72) };
        self.controls.set_view(target, radius, azimuth, elevation);
    }

    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        if frame.resized.is_some() {
            self.renderer.set_render_scale(self.render_scale());
        }
        let now = kansei_wasm::now();
        let dt = ((now - self.last) as f32).clamp(1e-4, 1.0 / 15.0);
        self.last = now;
        self.time += dt;
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;

        // the pointer: the left drag's action, orbit, the light, the wheel's zoom
        let p = self.pointer.take();
        if p.wheel != 0.0 {
            let r = self.controls.radius;
            self.controls.zoom(r - (r * (1.0 + p.wheel * 0.001)).clamp(0.8, 40.0));
        }
        let action = match p.drag {
            Some(Drag::Primary) => self.settings.drag.as_str(),
            Some(Drag::Orbit) => "orbit",
            Some(Drag::Light) => "light",
            None => "",
        };
        let mut ray = None;
        match action {
            "orbit" => self.controls.rotate(p.moved[0] * -0.005, p.moved[1] * 0.005),
            "light" if p.moved != [0.0; 2] => {
                let s = &mut self.settings;
                s.light_azim = (s.light_azim + p.moved[0] * 0.3 + 180.0).rem_euclid(360.0) - 180.0;
                s.light_elev = (s.light_elev - p.moved[1] * 0.25).clamp(5.0, 89.0);
                // (the light, the backdrop, the horizon and the exposure follow it)
                self.apply();
            }
            "push" => {
                let inv = (self.camera.projection_matrix.to_glam() * self.camera.view_matrix.to_glam()).inverse();
                let near = inv * glam::Vec4::new(p.ndc[0], p.ndc[1], 0.0, 1.0);
                let far = inv * glam::Vec4::new(p.ndc[0], p.ndc[1], 1.0, 1.0);
                let (near, far) = (near.xyz() / near.w, far.xyz() / far.w);
                ray = Some((near, (far - near).normalize()));
            }
            _ => {}
        }
        self.controls.update(&mut self.camera, dt);

        let view_proj = self.camera.view_projection().to_glam();
        let mut water_caustics = None;
        if let Some(w) = &mut self.water {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                self.pushing = w.stroke(ray, dt, &self.settings);
                w.update(surface, dt, view_proj, &self.settings);
                w.update_surface(&mut self.scene, self.renderer.queue(), surface, self.time);
                water_caustics = w.caustic_water(surface);
            }
        }
        let State { scene, volume, .. } = self;
        let lights: Vec<&Light> = scene.lights().collect();
        if let Some(e) = volume.effect_mut::<RtShadowsEffect>() {
            e.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<RtDiffuseGiEffect>() {
            e.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<RtReflectionsEffect>() {
            e.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<Switch<RtCausticsEffect>>() {
            let x = &mut e.effect;
            x.update_lights(lights.iter().copied());
            x.set_time(self.time);
            let mut casters: Vec<(GVec3, GVec3)> = self.dragon.iter().map(|d| d.2).collect();
            if let Some(w) = &self.water {
                casters.extend(w.sphere_box());
                if water_caustics.is_some() && x.water_photons {
                    casters.push(w.water_box());
                }
            }
            x.set_casters(&casters);
            x.set_water(water_caustics);
            x.interval = self.settings.caustics_every.round().clamp(1.0, 16.0) as u32;
        }

        if self.frame.is_multiple_of(10) {
            set_text("hud", &self.status());
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        kansei_wasm::perf::profile(&mut self.renderer);
        if let Some(stats) = &mut self.stats {
            stats.frames += 1;
            let ms = now * 1000.0;
            if ms - stats.since >= 1000.0 {
                stats.frame_ms = (ms - stats.since) / stats.frames as f64;
                stats.frames = 0;
                stats.since = ms;
                let profile = self.renderer.take_profile();
                if profile.gpu_frames > 0 {
                    stats.passes = profile.top_passes(usize::MAX);
                    stats.gpu_ms = profile.gpu_ms;
                    stats.gpu_span_ms = profile.gpu_span_ms;
                }
                let mut cpu = profile.cpu.clone();
                cpu.sort_by(|a, b| b.1.total_cmp(&a.1));
                stats.cpu = cpu;
                let (w, h) = (self.renderer.width(), self.renderer.height());
                let mut text = format!("{w} x {h}   frame {:.2} ms   GPU {:.2} ms (span {:.2})\n", stats.frame_ms, stats.gpu_ms, stats.gpu_span_ms);
                for (label, ms) in stats.passes.iter().take(24) {
                    text.push_str(&format!("{ms:6.2}  {label}\n"));
                }
                text.push_str("CPU\n");
                for (label, ms) in stats.cpu.iter().take(8) {
                    text.push_str(&format!("{ms:6.2}  {label}\n"));
                }
                set_text("stats", &text);
            }
        }
    }

    /// The scene's share of the canvas: `render_scale`, or (0) what keeps it at about
    /// `AUTO_PIXELS` (a Retina canvas's four times the pixels would cost the ray traced passes
    /// four times as much; TAA upscales).
    fn render_scale(&self) -> f32 {
        let s = self.settings.render_scale;
        if s > 0.0 {
            return s.clamp(0.25, 1.0);
        }
        let pixels = (self.renderer.width() as f32 * self.renderer.height() as f32).max(1.0);
        (AUTO_PIXELS / pixels).sqrt().clamp(0.5, 1.0)
    }

    /// The flat ambient the surfaces add themselves (cd/m²): the white sky's, unless the RT GI
    /// lights them with it.
    fn ambient(&self) -> f32 {
        if self.settings.gi == "rt" { 0.0 } else { self.settings.sky_radiance }
    }

    fn status(&self) -> String {
        let s = &self.settings;
        let mut text = format!("{:.0} fps · {} · light {} {:.0}° / {:.0}° · GI {}", self.fps, if s.scene == "dragon" { "glass dragon" } else { "box of water" }, s.light_type, s.light_azim, s.light_elev, s.gi);
        if let Some(w) = &self.water {
            let speed = w.speed().filter(|_| w.state_name() == "running").map_or(String::new(), |(max, _)| format!(", fastest {max:.2} m/s"));
            text.push_str(&format!("\nwater  {} particles {:.1} cm apart, {}{speed}", kansei_wasm::thousands(w.particles()), w.spacing_cm(), w.state_name()));
        }
        let left = match s.drag.as_str() {
            "push" => "drag: push the water",
            "light" => "drag: move the light",
            _ => "drag: orbit",
        };
        text.push_str(&format!("\n{left} · right drag orbit · shift drag light · wheel zoom · P panel · ` perf"));
        text
    }

    /// Bring everything to the settings: what changed since the last time.
    fn apply(&mut self) {
        let s = self.settings.clone();
        let old = self.applied.clone();
        let changed = |f: fn(&Settings) -> String| old.as_ref().is_none_or(|o| f(o) != f(&s));
        let ambient = self.ambient();
        let queue = self.renderer.queue().clone();

        self.studio.apply(&self.renderer, &mut self.scene, &s, ambient);
        // the tank's walls' ambient
        if let Some(w) = &self.water {
            w.set_look(&self.renderer, &mut self.scene, &s, ambient);
        }
        // the checkerboard's texture for the reflections' and the glass's hits
        if std::mem::take(&mut self.studio.checker_changed) {
            if let Some(e) = self.volume.effect_mut::<RtReflectionsEffect>() {
                e.set_albedo_texture(self.studio.checker_view.as_ref());
            }
        }
        if changed(|s| format!("{} {} {}", s.sky_radiance, s.floor, s.checker_lines)) {
            let up = s.sky_radiance;
            queue.write_buffer(&self.sky_lighting, 0, bytemuck::cast_slice(&sky_lighting(up, up * studio::floor_surface(&s, 0.0).mean())));
        }

        // the glass
        let tint = water::glass_tint(s.glass_tint);
        if let Some((index, copy, _)) = self.dragon {
            if let Some(r) = self.scene.get_renderable_mut(index) {
                r.material.set_standard_lit(&self.renderer, &StandardLitOptions::glass(tint, s.glass_ior, s.glass_rough));
            }
            if let Some(r) = self.scene.get_renderable_mut(copy) {
                let rt = Some(RtSurface::glass(tint).with_smooth_normals());
                if r.rt != rt {
                    r.rt = rt;
                }
            }
        }

        // the water
        let fluid = |s: &Settings| {
            let j = s.json();
            j.split(",\"").filter(|f| f.starts_with("fluid") || f.starts_with("box_") || f.starts_with("sphere") || f.starts_with("glass") || f.starts_with("sky")).collect::<Vec<_>>().join(",")
        };
        if let (Some(w), true) = (&mut self.water, changed(fluid)) {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                w.rebuild(&self.renderer, &mut self.scene, surface, &s);
                w.apply(&self.renderer, &mut self.scene, surface, &s);
            }
        }

        // the light's shadows, the GI, the caustics, the reflections
        if let Some(e) = self.volume.effect_mut::<RtShadowsEffect>() {
            e.half_resolution = s.shadow_res != "full";
            e.contact_length = if s.contact { s.contact_length } else { 0.0 };
            // (the sun's angular radius)
            e.sun_angle = (s.light_size.clamp(0.0, 40.0) * 0.5).to_radians();
        }
        if let Some(e) = self.volume.effect_mut::<Switch<SSAOEffect>>() {
            e.on = s.gi == "ssao";
            e.effect.options.radius = s.ssao_radius.max(0.01);
            e.effect.options.strength = s.ssao_strength.clamp(0.0, 2.0);
        }
        if let Some(e) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            e.enabled = s.gi == "rt";
            e.set_resolution(if s.gi_res == "full" { RtGiResolution::Full } else { RtGiResolution::Half });
        }
        if let Some(e) = self.volume.effect_mut::<Switch<RtCausticsEffect>>() {
            e.on = s.caustics;
            e.effect.set_enabled(&queue, s.caustics);
            e.effect.photons = s.caustics_photons.round().clamp(8.0, 1024.0) as u32;
            e.effect.blend = s.caustics_blend.clamp(0.01, 1.0);
            e.effect.intensity = s.caustics_intensity.max(0.0);
            e.effect.view = if s.caustics_view { RtCausticsView::Caustics } else { RtCausticsView::Lit };
            e.effect.water_photons = s.caustics_water_photons;
            e.effect.water_lines = Some(s.caustics_lines.clamp(0.0, 1.0));
            e.effect.water_span = s.caustics_smooth.round().clamp(1.0, 8.0) as u32;
            e.effect.area_emitters = s.caustics_area;
            e.effect.focus_gain = s.caustics_glass_gain.max(0.0);
            e.effect.focus_sharpen = s.caustics_sharpen.clamp(0.1, 1.0);
            e.effect.set_glass(s.glass_ior, tint);
        }
        if let Some(e) = self.volume.effect_mut::<RtReflectionsEffect>() {
            e.set_glass(Some(RtGlass { reflections_in_trace: s.glass_refl_shared, checkerboard: s.glass_checker, ..Default::default() }));
        }

        // the horizon: fades to the backdrop
        let bg = self.studio.backdrop(&s);
        if let Some(e) = self.volume.effect_mut::<HeightFogEffect>() {
            e.inscattering = Vec3::new(bg, bg, bg);
            e.layers[0].density = if s.horizon { HORIZON_DENSITY } else { 0.0 };
        }

        // the post-processing
        let ev = if s.auto_ev { Studio::auto_ev(&s, AUTO_LEVEL) } else { s.ev };
        let mut total = exposure_from_ev100(ev);
        if let Some(t) = self.volume.effect_mut::<ToneMapEffect>() {
            let o = &mut t.options;
            o.tonemapper = tonemapper(&s.tonemap);
            o.exposure = exposure_from_ev100(ev);
            o.exposure_compensation = s.exposure_comp;
            o.vignette = s.vignette;
            o.grain = 0.0;
            o.chromatic_aberration = 0.0;
            total = t.total_exposure();
        }
        if let Some(e) = self.volume.effect_mut::<Switch<TemporalAAEffect>>() {
            e.on = s.taa;
            e.effect.options.exposure = total;
        }
        if let Some(e) = self.volume.effect_mut::<Switch<BloomEffect>>() {
            e.on = s.bloom;
            e.effect.options.intensity = s.bloom_intensity;
            e.effect.exposure = total;
        }

        // the resolution: the canvas's height (or the screen's pixels times the ratio), the
        // scene's share of it (TAA upscales)
        if changed(|s| format!("{} {}", s.resolution, s.dpr)) {
            self.canvas.set_render_height(s.resolution.trim_end_matches('p').parse::<u32>().ok());
            self.canvas.set_pixel_ratio((s.dpr > 0.0).then_some(s.dpr as f64));
        }
        self.renderer.set_render_scale(self.render_scale());
        if changed(|s| format!("{}", s.perf)) {
            kansei_wasm::perf::set_visible(s.perf);
        }
        if changed(|s| format!("{}", s.stats)) {
            self.renderer.set_profiling(s.stats);
            self.stats = s.stats.then(|| Stats { since: kansei_wasm::now() * 1000.0, ..Default::default() });
            if !s.stats {
                set_text("stats", "");
            }
        }
        self.applied = Some(s);
    }

    /// The state as JSON: the settings, the water, the grid, the caustics, and (with stats) the
    /// frame and each pass's time.
    fn info(&self) -> String {
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let cpu: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.cpu.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let grid = self.renderer.rt_grid().map(|g| g.stats());
        let caustics = self.volume.effect::<Switch<RtCausticsEffect>>().map_or((0, 0), |e| e.effect.photons_sent());
        format!(
            "{{\"settings\":{},\"fluid\":{},\"state\":\"{}\",\"pushing\":{},\"grid_triangles\":{},\"grid_rebuilds\":{},\"caustics\":[{},{}],\"size\":[{},{}],\"fps\":{:.1},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"passes\":[{}],\"cpu\":[{}]}}",
            self.settings.json(),
            match (&self.water, self.volume.effect::<FluidSurfaceEffect>()) {
                (Some(w), Some(surface)) => w.surface_json(surface),
                _ => "null".into(),
            },
            self.water.as_ref().map_or("none", Water::state_name),
            self.pushing.map_or("null".into(), |p| format!("[{:.3},{:.3},{:.3}]", p.x, p.y, p.z)),
            grid.map_or(0, |g| g.grid.triangles),
            grid.map_or(0, |g| g.rebuilds),
            caustics.0,
            caustics.1,
            self.renderer.width(),
            self.renderer.height(),
            self.fps,
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            passes.join(","),
            cpu.join(","),
        )
    }
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        Some(f(&mut state))
    })
}

/// The page on canvas `canvas_id`.
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    kansei_wasm::init();
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(1.0, 1.0, 1.0, 1.0), ..Default::default() }).await;
    // the key light's shadow maps feed voxel GI's light (the RT shadows are what shows)
    renderer.enable_spot_shadows(1024, 1);
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: 20.0, ..Default::default() });
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: BOUNDS_MIN, bounds_max: BOUNDS_MAX, radiance_scale: 1.0, budget_bytes: 0 });
    // the grid: 140 x 48 x 140 cells round the subject (`grid_cell=`: 3.5 cm, 4.9 x 1.7 x 4.9 m;
    // the glass's cost is the triangles a ray meets in a cell)
    let cell: f32 = param_or("grid_cell", 0.035f32).clamp(0.02, 0.3);
    let origin = GVec3::new(-70.0 * cell, -0.05, -70.0 * cell);
    renderer.enable_rt_grid(SceneRtGridOptions {
        grid: RtGridOptions { dims: [140, 48, 140], cell, fixed_origin: Some(origin), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    });
    let settings = Settings::from_url();
    let dragon_scene = settings.scene == "dragon";
    set_text("hud", if dragon_scene { "Loading the dragon …" } else { "Filling the box …" });

    let mut scene = Scene::new();
    let subject = if dragon_scene { GVec3::new(0.0, 0.6, 0.0) } else { GVec3::new(0.0, 0.2, 0.0) };
    let studio = Studio::new(&mut scene, &settings, subject);
    // (`scale=`: simulation units a metre, the particles 0.537 / scale m apart)
    let scale: f32 = param_or("scale", 16.0f32).clamp(4.0, 40.0);
    let (water, surface) = if dragon_scene {
        (None, None)
    } else {
        let (w, s) = Water::new(&renderer, &mut scene, &settings, scale);
        (Some(w), Some(s))
    };
    let dragon = if dragon_scene { add_dragon(&mut scene, &settings).await } else { None };
    if dragon_scene && dragon.is_none() {
        set_text("hud", "The dragon didn't load (assets/: scene.gltf and scene.bin, or stanford_dragon_pbr.glb with dragon=light).");
    }
    let sky_lighting = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: Some("Caustics/SkyLighting"), size: 240, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    if let Some(gi) = renderer.voxel_gi_mut() {
        gi.use_sky_lighting(&sky_lighting);
    }
    // the caustics' map: round the tank (and the sphere at its right) or the dragon, 1.2 cm
    // texels
    let rect = match &water {
        Some(w) => {
            let [x, z, _] = w.size();
            ([-x * 0.5 - 0.6, -z * 0.5 - 0.6], [x * 0.5 + 1.2, z * 0.5 + 0.6])
        }
        None => ([-3.0, -3.0], [3.0, 3.0]),
    };
    let volume = PostProcessingVolume::new(&renderer, build_effects(&renderer, surface, &sky_lighting, rect));

    let mut camera = Camera::new(40.0, 0.05, 2500.0, canvas.aspect());
    camera.update_projection_matrix();
    let controls = CameraControls::new(Vec3::new(0.0, 0.0, 0.0), 4.0);
    let pointer = Pointer::listen(canvas.element());
    let mut state = State {
        canvas: canvas.clone(),
        renderer,
        scene,
        camera,
        controls,
        pointer,
        volume,
        studio,
        water,
        dragon,
        sky_lighting,
        settings,
        applied: None,
        stats: None,
        time: 0.0,
        last: kansei_wasm::now(),
        frame: 0,
        fps: 60.0,
        pushing: None,
    };
    state.home();
    state.apply();
    log::info!("Kansei — Caustics ({}) ready", state.settings.scene);
    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The state as JSON (with `stats`, each pass's GPU time): for scripted captures.
#[wasm_bindgen]
pub fn caustics_info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// Every setting as JSON: the panel's starting values.
#[wasm_bindgen]
pub fn caustics_settings() -> String {
    with_state(|s| s.settings.json()).unwrap_or_default()
}

/// Change a setting (the URL's names) live; false for an unknown one.
#[wasm_bindgen]
pub fn caustics_set(key: &str, value: &str) -> bool {
    with_state(|s| {
        let ok = s.settings.set(key, value);
        if ok {
            s.apply();
        }
        ok
    })
    .unwrap_or(false)
}

/// The quality preset (`game` or `high`) for the scene, live; the settings it sets, as JSON.
#[wasm_bindgen]
pub fn caustics_quality(name: &str) -> String {
    with_state(|s| {
        s.settings.apply_quality(name);
        s.apply();
        format!("[{}]", Settings::QUALITY_KEYS.iter().map(|k| format!("\"{k}\"")).collect::<Vec<_>>().join(","))
    })
    .unwrap_or_default()
}

/// Fill the tank again, still.
#[wasm_bindgen]
pub fn caustics_reset() {
    with_state(|s| {
        if let (Some(w), Some(surface)) = (&mut s.water, s.volume.effect_mut::<FluidSurfaceEffect>()) {
            w.reset(surface);
        }
    });
}

/// The camera back to the scene's start.
#[wasm_bindgen]
pub fn caustics_home() {
    with_state(State::home);
}
