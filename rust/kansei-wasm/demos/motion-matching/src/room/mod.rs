//! The room page (`www/room.html`, [`start_room`]): the motion-matched character in a 40 x 40 m
//! loft lit by ray tracing.
//!
//! - The light: the low sun through three tall steel windows to the west (warm, its shadows ray
//!   traced through a cone), a skylight over the pond (a cool area light: a spot light whose
//!   emitter the ray-traced shadows sample as the opening's rectangle), and warm lamps under glowing
//!   shades. A sky behind the windows lights what the rays find through them.
//! - Ray-traced direct light and shadows (`rt::RtShadowsEffect`, `shadows=rt`, the default): soft
//!   where the emitter is large, sharp at contacts, screen-space contact rays for feet and props;
//!   `shadows=maps` falls back to the shadow maps. The GI: the hybrid (`RtDiffuseGiEffect`, `gi=rt`),
//!   voxel cones, screen space or none.
//! - Two mirrors, a polished floor (white marble slabs or an oak herringbone) and a glass Stanford
//!   dragon on a black marble plinth in a pond, all ray traced (`RtReflectionsEffect`); the pond is
//!   the engine's SPH water the character wades into (`pond`).
//! - CC0 furniture (Poly Haven) and surfaces (ambientCG), KTX2 (`assets`, `tools/room_assets.py`);
//!   concrete and oak blocks to vault, mantle and climb (`layout`).
//! - Subtle volumetric fog lit by the same lights, and dust motes from a compute shader drawn with
//!   sprites it generates at start (`dust`).
//! - Post-processing: AgX (or another curve), bloom, a cinematic depth of field focused on the
//!   character, vignette, grain, chromatic aberration, TAA, motion blur.
//! - The walls and the ceiling are one-sided: O swings the camera outside.
//!
//! Every setting is a URL parameter (its initial value) and a control in the page's panel (live):
//! see `settings`. The character is the lake page's (`character`, `player`), drawn by
//! `look::RoomLit` and standing in the grid as capsules on its bones (`look::Proxy`).

mod assets;
mod dust;
mod layout;
mod look;
mod pbr;
mod pond;
mod settings;

use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec3 as GVec3;
use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions};
use kansei_core::controls::CameraControls;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::SphereGeometry;
use kansei_core::gi::{SceneVoxelGiOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
use kansei_core::lights::Light;
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{GradientSkyOptions, Material, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100, BloomEffect, BloomOptions, CameraLens, CinematicDepthOfFieldEffect, CinematicDepthOfFieldOptions, FluidSurfaceEffect, LocalExposure, LocalFogVolume,
    MotionBlurEffect, MotionBlurOptions, ScreenSpaceGIEffect, ScreenSpaceGIOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, ToneMapper,
    VolumetricFogEffect, VolumetricFogOptions,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::{
    RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiResolution, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions, RtShadowsEffect, RtShadowsView, RtSurface,
    RtTraceResolution, SceneRtGridOptions,
};
use kansei_core::shadows::CascadedShadowOptions;
use kansei_wasm::{param, param_or, set_text, Canvas, Frame};

use crate::player::Player;
use crate::{fetch_bytes, load_character, set_page, Page};
use assets::Assets;
use dust::Dust;
use layout::{Layout, HALF, HEIGHT, START, START_HEADING};
use look::{Proxy, RoomLit};
use pond::Pond;
use settings::{kelvin, Settings};

/// The renderer's profile over the last second (`stats`).
#[derive(Default)]
struct Stats {
    frames: u32,
    since: f64,
    frame_ms: f64,
    /// (label, ms per frame), most expensive first
    passes: Vec<(&'static str, f64)>,
    /// (label, ms per frame) of the CPU sections, most expensive first
    cpu: Vec<(&'static str, f64)>,
    gpu_ms: f64,
    gpu_span_ms: f64,
}

/// The glass dragon's tint (the light left after a metre inside).
const GLASS_TINT: [f32; 3] = [0.95, 0.98, 0.97];

struct RoomState {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    player: Player,
    assets: Assets,
    layout: Layout,
    pond: Option<Pond>,
    dust: Option<Dust>,
    proxy: Option<Proxy>,
    dragon: Option<usize>,
    /// The character's bodies or the stand-in (for the deferred switch).
    bodies: Vec<usize>,
    sky: usize,
    sky_lighting: wgpu::Buffer,
    settings: Settings,
    /// The settings as last applied (what changed since).
    applied: Option<Settings>,
    /// A fixed view (`VIEWS`), or following the character.
    view: Option<&'static str>,
    stats: Option<Stats>,
    profile_since: f64,
    time: f32,
    dof_focus: f32,
}

impl Page for RoomState {
    fn player(&mut self) -> &mut Player {
        &mut self.player
    }
}

/// The room's box, for voxel GI and the grid: the walls, the floor (and the pond's bed under it)
/// and the ceiling with the skylight's well, with a margin.
const BOUNDS_MIN: [f32; 3] = [-HALF - 0.5, -1.0, -HALF - 0.5];
const BOUNDS_MAX: [f32; 3] = [HALF + 0.5, HEIGHT + 1.5, HALF + 0.5];

/// The grid of the room's triangles: 30 cm cells over the room (140 x 36 x 140, within the grid's
/// million cells), fixed.
fn grid_options() -> SceneRtGridOptions {
    SceneRtGridOptions {
        grid: RtGridOptions { dims: [140, 36, 140], cell: 0.3, fixed_origin: Some(glam::Vec3::new(-21.0, -1.2, -21.0)), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    }
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

/// The effect chain: the pond's surface first (on the lit scene), the ray-traced direct light, the
/// three GI effects (one enabled), the reflections and the glass, the fog, TAA, motion blur, the
/// depth of field, bloom and the tone map. `apply` sets them all to the settings.
fn build_effects(renderer: &Renderer, surface: Option<FluidSurfaceEffect>, sky: &wgpu::Buffer) -> Vec<Box<dyn PostProcessingEffect>> {
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(surface) = surface {
        effects.push(Box::new(surface));
    }
    let scene_gi = renderer.voxel_gi().expect("the room enables voxel GI");
    let grid = renderer.rt_grid().expect("the room builds the grid");
    let mut shadows = RtShadowsEffect::new(grid.handle());
    // the skylight is the first spot light: its emitter is the opening
    shadows.set_rect_emitter(0, layout::SKYLIGHT[0] * 2.0, layout::SKYLIGHT[1] * 2.0);
    effects.push(Box::new(Switch { effect: shadows, on: true }));
    let mut hybrid = RtDiffuseGiEffect::with_volume(scene_gi.volume(), grid.handle(), RtDiffuseGiOptions { max_distance: 60.0, ..Default::default() });
    hybrid.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    hybrid.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
    hybrid.set_sky_lighting(Some(sky));
    effects.push(Box::new(hybrid));
    let mut voxel = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), ..Default::default() });
    voxel.set_sky_lighting(Some(sky));
    effects.push(Box::new(voxel));
    let mut ssgi = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { radius_m: 6.0, ..Default::default() });
    ssgi.set_sky_lighting(Some(sky));
    effects.push(Box::new(ssgi));
    // the mirrors, the floor and the dragon's glass
    let mut reflections = RtReflectionsEffect::with_volume(scene_gi.volume(), grid.handle(), RtReflectionsOptions { resolution: RtTraceResolution::Half, max_distance: 60.0, ..Default::default() });
    reflections.screen_hits = true;
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    reflections.set_glass(Some(RtGlass::default()));
    reflections.set_sky_lighting(Some(sky));
    effects.push(Box::new(Switch { effect: reflections, on: true }));
    // a faint, even haze filling the room (a box volume; none outside), lit by the sun through the
    // cascades and the spot lights through their shadow maps
    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions { grid: FroxelGridOptions { near: 0.3, far: 90.0, temporal: true, ..Default::default() }, base_density: 0.0, ..Default::default() });
    let mut haze = LocalFogVolume::new_box(Vec3::new(0.0, HEIGHT * 0.5, 0.0), Vec3::new(HALF, HEIGHT * 0.5, HALF));
    haze.radial_extinction = 0.0;
    haze.height_offset = -1.0;
    haze.height_falloff = 0.35;
    haze.edge_fade = 0.02;
    haze.albedo = Vec3::new(0.9, 0.9, 0.9);
    fog.local_volumes.push(haze);
    fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
    effects.push(Box::new(Switch { effect: fog, on: true }));
    effects.push(Box::new(Switch { effect: TemporalAAEffect::new(TemporalAAOptions::default()), on: true }));
    effects.push(Box::new(Switch { effect: MotionBlurEffect::new(MotionBlurOptions::default()), on: false }));
    effects.push(Box::new(Switch { effect: CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions::default()), on: false }));
    effects.push(Box::new(Switch { effect: BloomEffect::new(BloomOptions::default()), on: true }));
    effects.push(Box::new(ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()))));
    effects
}

/// The sky the windows and the skylight show, as `SkyLighting` (`atmosphere::SKY_LIGHTING_WGSL`)
/// for the rays that leave through them: its radiance as order-2 SH (the constant and vertical
/// terms) of a gradient from the horizon to the zenith, darker below.
fn sky_lighting(zenith: [f32; 3], horizon: [f32; 3]) -> [f32; 60] {
    // L(d) = a + b d.y
    let a: [f32; 3] = std::array::from_fn(|c| (zenith[c] + horizon[c]) * 0.5);
    let b: [f32; 3] = std::array::from_fn(|c| (zenith[c] - horizon[c]) * 0.5 + horizon[c] * 0.4);
    let mut out = [0.0f32; 60];
    for c in 0..3 {
        // sh0: a * 0.282095 * 4 pi; sh1 (y): b * 0.488603 * 4 pi / 3
        out[c] = a[c] * 3.544_908;
        out[4 + c] = b[c] * 2.046_653;
    }
    out
}

/// The glass dragon on the plinth: the Stanford dragon (`www/assets`, CC-BY-NC-4.0, credited on the
/// page) about 2.6 m long, clear glass, in the grid by its cluster LOD's cut with its vertex normals,
/// casting no shadow (light passes through glass), outside voxel GI.
async fn add_dragon(scene: &mut Scene) -> Option<usize> {
    let bytes = match fetch_bytes("assets/stanford_dragon_pbr.glb").await {
        Ok(b) => b,
        Err(e) => {
            log::warn!("no dragon: {e}");
            return None;
        }
    };
    let geometry = GLTFLoader::load_glb(&bytes).ok()?.merged_geometry("Dragon").fit(glam::Vec3::new(2.6, f32::INFINITY, 2.6));
    let mesh = ClusterMesh::build(&geometry, &ClusterOptions::default());
    let mut dragon = Renderable::new(geometry, Material::standard_lit("Dragon", &StandardLitOptions::glass(GLASS_TINT, 1.5, 0.0)));
    dragon.clusters = Some(ClusterLod::new(mesh));
    dragon.rt = Some(RtSurface::glass(GLASS_TINT));
    dragon.cast_shadow = false;
    dragon.object.set_position(0.0, pond::PLINTH_TOP, 0.0);
    dragon.object.rotation.y = 0.7;
    Some(scene.add(SceneNode::Renderable(dragon)))
}

/// Fixed views (`cam=<name>`): (name, target, distance, azimuth, elevation); the camera stands at
/// the target plus (sin, cos) of the azimuth times the distance.
const VIEWS: [(&str, [f32; 3], f32, f32, f32); 9] = [
    // from the south-east, above the walls' tops, looking in through them and the ceiling
    ("outside", [0.0, 1.5, 0.0], 52.0, 0.65, 0.42),
    // the whole room from its south-east, the windows and the sun opposite
    ("hall", [-2.0, 2.2, -2.0], 26.0, 0.78, 0.16),
    // the glass dragon on its plinth, the big mirror behind it
    ("dragon", [0.0, 1.3, 0.0], 6.5, 0.25, 0.12),
    // the big mirror on the north wall at an angle, the pond and the dragon in it
    ("mirror", [0.0, 2.2, -19.5], 14.8, 0.42, 0.09),
    // the windows and the sun coming through them
    ("windows", [-16.0, 2.5, 0.0], 15.0, 1.25, 0.08),
    // the crates and platforms in the north-east corner
    ("parkour", [12.0, 1.0, -12.0], 12.0, -0.7, 0.3),
    // the living corner and its lamp
    ("living", [13.0, 0.8, 13.0], 8.5, -0.9, 0.22),
    // the dining table under its pendant
    ("dining", [-13.0, 1.0, 13.0], 6.5, 0.7, 0.25),
    // the library: shelves, the reading chair
    ("library", [-13.0, 1.2, -16.0], 7.0, 0.6, 0.18),
];

/// Without a character: the pond from beside the stand-in, a little off its axis.
fn stand_in_view(controls: &mut CameraControls) {
    controls.set_view(Vec3::new(START[0], 1.3, START[2] - 5.0), 11.0, 0.4, 0.16);
}

fn tonemapper(name: &str) -> ToneMapper {
    match name {
        "aces" => ToneMapper::AcesFitted,
        "agx_punchy" => ToneMapper::AgXPunchy,
        "neutral" => ToneMapper::KhronosNeutral,
        "unreal" => ToneMapper::UnrealFilmic,
        "none" => ToneMapper::None,
        _ => ToneMapper::AgX,
    }
}

impl RoomState {
    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let now = kansei_wasm::now();
        let dt = ((now - self.player.last_frame(now)) as f32).clamp(1e-4, 1.0 / 15.0);
        self.time += dt;
        self.player.follow = self.view.is_none();
        let RoomState { renderer, scene, controls, player, layout, .. } = self;
        let out = {
            let _t = kansei_core::profiling::cpu_scope("Room/Character");
            player.update(now, dt, scene, renderer, controls, &layout.collision)
        };
        let mut changed = false;
        for key in &out.keys {
            changed |= match key.as_str() {
                "o" => {
                    self.settings.cam = if self.view.is_some() { "follow".into() } else { "outside".into() };
                    true
                }
                "g" => {
                    let next = match self.settings.gi.as_str() {
                        "rt" => "voxel",
                        "voxel" => "ssgi",
                        "ssgi" => "off",
                        _ => "rt",
                    };
                    self.settings.gi = next.into();
                    true
                }
                "f" => {
                    self.settings.fog = if self.settings.fog > 0.0 { 0.0 } else { Settings::default().fog };
                    true
                }
                "n" => {
                    self.settings.dust = !self.settings.dust;
                    true
                }
                "t" => {
                    self.settings.shadows = if self.settings.shadows == "rt" { "maps".into() } else { "rt".into() };
                    true
                }
                _ => false,
            };
        }
        if changed {
            self.apply();
        }
        self.controls.update(&mut self.camera, dt);

        let world_scope = kansei_core::profiling::cpu_scope("Room/World");
        // the character's capsules in the grid, the water, the dust, the lens
        if let (Some(proxy), Some(c)) = (&mut self.proxy, &self.player.character) {
            let visible = self.scene.get_renderable(c.bodies[c.showing].index).is_some_and(|r| r.visible);
            proxy.update(&mut self.scene, c, visible);
        }
        let view_proj = self.camera.view_projection().to_glam();
        if let Some(pond) = &mut self.pond {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                pond.update(surface, &out.legs, out.landing, dt, view_proj);
            }
        }
        let eye = *self.camera.position();
        let eye = GVec3::new(eye.x, eye.y, eye.z);
        self.update_focus(eye, out.at, dt);
        let lens = self.dust_lens();
        if let Some(dust) = &mut self.dust {
            let mut capsules = out.legs.clone();
            if let Some(&(hips, _, _)) = out.legs.last() {
                capsules.insert(0, (hips, hips + GVec3::Y * 0.65, 0.25));
            }
            let height = self.renderer.height() as f32;
            dust.update(&self.renderer, &mut self.scene, eye, &capsules, dt, height, lens);
        }
        let RoomState { scene, volume, .. } = self;
        let lights: Vec<&Light> = scene.lights().collect();
        if let Some(e) = volume.effect_mut::<Switch<RtShadowsEffect>>() {
            e.effect.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<RtDiffuseGiEffect>() {
            e.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<Switch<RtReflectionsEffect>>() {
            e.effect.update_lights(lights.iter().copied());
        }
        if let Some(e) = volume.effect_mut::<Switch<VolumetricFogEffect>>() {
            e.effect.update_lights(lights.iter().copied());
            e.effect.time = self.time;
        }
        if let Some(e) = volume.effect_mut::<Switch<MotionBlurEffect>>() {
            e.effect.set_frame_time(dt);
        }
        drop(world_scope);

        // HUD, a few times a second
        if self.player.frame.is_multiple_of(10) {
            let room = self.status();
            let hint = " · O inside / outside · G GI · T shadows · F fog · N dust · P panel";
            match self.player.hud(&room, hint) {
                Some(text) => set_text("hud", &text),
                // no pack: the page's message stays, the room's lines under it
                None => set_text("hud-room", &format!("{room}\ndrag orbit · wheel zoom{hint}")),
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
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
        } else if self.profile_since > 0.0 && now - self.profile_since > 3.0 {
            self.profile_since = now;
            log::info!("profile ({:.1} ms/frame)\n{}", 1000.0 / self.player.fps, self.renderer.take_profile().report());
        }
    }

    /// The depth of field's focus: on the character (a metre up), on what the screen's centre looks
    /// at (a ray through the collision world), or a set distance; eased.
    fn update_focus(&mut self, eye: GVec3, at: Option<GVec3>, dt: f32) {
        let s = &self.settings;
        let target = match s.dof_focus.as_str() {
            "manual" => s.dof_distance,
            "centre" => {
                let forward = (-self.camera.inverse_view_matrix.to_glam().z_axis.truncate()).normalize();
                self.layout.collision.raycast(eye, forward, 200.0, u32::MAX).map_or(s.dof_distance, |h| h.distance)
            }
            _ => match at {
                Some(p) if self.view.is_none() => eye.distance(p + GVec3::Y * 1.0),
                _ => self.controls_distance(),
            },
        };
        self.dof_focus += (target.max(0.2) - self.dof_focus) * (1.0 - (-dt * 6.0).exp());
        let focus = self.dof_focus;
        if let Some(e) = self.volume.effect_mut::<Switch<CinematicDepthOfFieldEffect>>() {
            e.effect.lens.focus_distance_m = focus;
        }
    }

    fn controls_distance(&self) -> f32 {
        let target = self.controls.look_target();
        let eye = self.camera.position();
        ((target.x - eye.x).powi(2) + (target.y - eye.y).powi(2) + (target.z - eye.z).powi(2)).sqrt()
    }

    /// The lens for the dust's bokeh: focus distance and the circle (px) per unit of
    /// |1 - focus / z|, from the f-number and the focal length.
    fn dust_lens(&self) -> dust::DustLens {
        let s = &self.settings;
        if !s.dof {
            return None;
        }
        let sensor = 23.76f32;
        let fov = self.camera.fov.to_radians();
        let aspect = self.renderer.width() as f32 / self.renderer.height().max(1) as f32;
        let hfov = 2.0 * ((fov * 0.5).tan() * aspect).atan();
        let f = if s.focal_mm > 0.0 { s.focal_mm } else { 0.5 * sensor / (0.5 * hfov).tan() };
        let focus_mm = self.dof_focus.max(0.2) * 1000.0;
        // the blur circle on the sensor for an object at distance z: f^2 / (N (focus - f)) |1 - focus / z|
        let c_mm = f * f / (s.f_stop.max(0.5) * (focus_mm - f).max(1.0));
        Some((self.dof_focus, c_mm / sensor * self.renderer.width() as f32))
    }

    /// The HUD's lines about the room.
    fn status(&self) -> String {
        let s = &self.settings;
        let pond = self.pond.as_ref().map_or(String::new(), |p| format!("\npond   {} particles, {}", p.particles(), p.state().name()));
        format!(
            "room   GI {} · shadows {} · floor {} · fog {} · dust {} · camera {}{}\n",
            s.gi,
            if s.shadows == "rt" { "ray traced" } else { "shadow maps" },
            s.floor,
            if s.fog > 0.0 { "on" } else { "off" },
            self.dust.as_ref().filter(|_| s.dust).map_or("off".to_string(), |d| format!("{} motes", (d.count() as f32 * s.dust_amount.clamp(0.0, 1.0)) as u32)),
            self.view.unwrap_or("following"),
            pond,
        )
    }

    /// A fixed view by name, or back to following the character.
    fn set_view(&mut self, name: &str) {
        self.view = VIEWS.iter().find(|v| v.0 == name).map(|v| v.0);
        if let Some(&(_, target, distance, azimuth, elevation)) = VIEWS.iter().find(|v| Some(v.0) == self.view) {
            self.controls.set_view(Vec3::new(target[0], target[1], target[2]), distance, azimuth, elevation);
        } else if let Some(c) = &self.player.character {
            let at = c.controller.matcher.character();
            self.controls.set_view(Vec3::new(at.translation.x, 0.9, at.translation.z), 4.5, std::f32::consts::PI + kansei_core::animation::motion_matching::yaw_of(at.rotation), 0.25);
        } else {
            stand_in_view(&mut self.controls);
        }
    }

    /// Bring everything to the settings: what changed since the last time.
    fn apply(&mut self) {
        let s = self.settings.clone();
        let old = self.applied.clone();
        let changed = |f: fn(&Settings) -> String| old.as_ref().is_none_or(|o| f(o) != f(&s));
        let queue = self.renderer.queue().clone();
        let deferred = s.shadows == "rt";

        // the surfaces: deferred or forward, the floor
        if changed(|s| s.shadows.clone()) {
            for surface in &mut self.layout.surfaces {
                surface.params.ambient[3] = deferred as u32 as f32;
                if let Some(buffer) = self.scene.get_renderable(surface.index).and_then(|r| r.material.bindable_buffer(0)) {
                    queue.write_buffer(&buffer, 60, bytemuck::bytes_of(&surface.params.ambient[3]));
                }
            }
            for &body in &self.bodies {
                if let Some(buffer) = self.scene.get_renderable(body).and_then(|r| r.material.bindable_buffer(0)) {
                    queue.write_buffer(&buffer, 28, bytemuck::bytes_of(&(deferred as u32 as f32)));
                }
            }
        }
        if changed(|s| s.floor.clone()) || changed(|s| format!("{}", s.floor_rough)) {
            let rebuild = old.is_some() && changed(|s| s.floor.clone());
            for surface in self.layout.surfaces.iter_mut().filter(|f| f.floor) {
                let params = layout::floor_params(&s.floor, s.floor_rough, deferred);
                surface.params = params;
                let Some(r) = self.scene.get_renderable_mut(surface.index) else { continue };
                if rebuild {
                    r.material = pbr::material("Floor", self.assets.surface(&s.floor, "Floor"), &params, false);
                    r.material_dirty = true;
                    r.gi = Some(kansei_core::gi::GiSurface::new(layout::mean(&s.floor, [1.0; 3])));
                    r.rt = Some(RtSurface::new(layout::mean(&s.floor, [1.0; 3])));
                } else if let Some(buffer) = r.material.bindable_buffer(0) {
                    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&params));
                }
            }
            if rebuild {
                if let Some(gi) = self.renderer.voxel_gi_mut() {
                    gi.invalidate();
                }
            }
        }

        // the lights
        let sun_dir = {
            let (e, a) = (s.sun_elev.to_radians(), s.sun_azim.to_radians());
            // from the west (light travelling +x), turned toward +z by the azimuth
            GVec3::new(e.cos() * a.cos(), -e.sin(), e.cos() * a.sin()).normalize()
        };
        let sun_color = kelvin(s.sun_temp);
        if let Some(Light::Directional(l)) = self.scene.get_light_mut(self.layout.lights.sun) {
            l.direction = Vec3::new(sun_dir.x, sun_dir.y, sun_dir.z);
            l.color = Vec3::new(sun_color[0], sun_color[1], sun_color[2]);
            l.intensity = if s.sun { s.sun_lux } else { 0.0 };
        }
        let sky_color = kelvin(s.sky_temp);
        if let Some(Light::Spot(l)) = self.scene.get_light_mut(self.layout.lights.skylight) {
            l.color = Vec3::new(sky_color[0], sky_color[1], sky_color[2]);
            l.intensity = if s.sky { s.sky_cd } else { 0.0 };
        }
        let lamp_color = kelvin(s.lamp_temp);
        for &lamp in &self.layout.lights.lamps {
            if let Some(Light::Spot(l)) = self.scene.get_light_mut(lamp) {
                l.color = Vec3::new(lamp_color[0], lamp_color[1], lamp_color[2]);
                l.intensity = if s.lamps { s.lamp_cd } else { 0.0 };
            }
        }
        let shade = lamp_color.map(|c| c * layout::SHADE_RADIANCE * if s.lamps { s.lamp_cd / layout::LAMP_CD } else { 0.02 });
        for &index in &self.layout.lights.shades {
            if let Some(r) = self.scene.get_renderable_mut(index) {
                r.material.set_standard_lit(&self.renderer, &StandardLitOptions::emissive(shade));
            }
        }
        // the sky behind the windows, and what the rays leaving through them find
        if changed(|s| format!("{}", s.sky_radiance)) {
            let horizon = [s.sky_radiance, s.sky_radiance * 0.82, s.sky_radiance * 0.62];
            let zenith = [s.sky_radiance * 0.45, s.sky_radiance * 0.62, s.sky_radiance];
            if let Some(r) = self.scene.get_renderable_mut(self.sky) {
                r.material = Material::gradient_sky("Sky", &GradientSkyOptions { zenith, horizon, ground: horizon.map(|c| c * 0.25), curve: 0.5 });
                r.material_dirty = true;
            }
            queue.write_buffer(&self.sky_lighting, 0, bytemuck::cast_slice(&sky_lighting(zenith, horizon)));
        }

        // shadows, GI, reflections
        let gi = s.gi.as_str();
        if let Some(e) = self.volume.effect_mut::<Switch<RtShadowsEffect>>() {
            e.on = deferred;
            let x = &mut e.effect;
            x.half_resolution = s.shadow_res != "full";
            x.contact_length = if s.contact { s.contact_length } else { 0.0 };
            x.sun_angle = s.sun_soft.to_radians();
            x.iterations = s.shadow_steps.round().clamp(0.0, 3.0) as u32;
            x.view = match s.rt_view.as_str() {
                "visibility" => RtShadowsView::Visibility,
                "direct" => RtShadowsView::Direct,
                "mask" => RtShadowsView::Mask,
                _ => RtShadowsView::Lit,
            };
            x.debug_light = s.debug_light.max(0.0) as u32;
        }
        if let Some(e) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            e.enabled = gi == "rt";
            e.set_resolution(if s.rtgi_res == "full" { RtGiResolution::Full } else { RtGiResolution::Half });
        }
        if let Some(e) = self.volume.effect_mut::<VoxelGIEffect>() {
            e.enabled = gi == "voxel";
        }
        if let Some(e) = self.volume.effect_mut::<ScreenSpaceGIEffect>() {
            e.enabled = gi == "ssgi";
        }
        if let Some(e) = self.volume.effect_mut::<Switch<RtReflectionsEffect>>() {
            e.on = s.reflections;
            e.effect.hit_indirect = gi != "off";
        }
        for (k, &mirror) in self.layout.mirrors.iter().enumerate() {
            let roughness = if k == 0 { s.mirror_rough } else { s.mirror2_rough };
            if let Some(r) = self.scene.get_renderable_mut(mirror) {
                r.material.set_standard_lit(&self.renderer, &StandardLitOptions::mirror([0.92; 3], roughness));
            }
        }
        if let Some(r) = self.dragon.and_then(|d| self.scene.get_renderable_mut(d)) {
            r.material.set_standard_lit(&self.renderer, &StandardLitOptions::glass(GLASS_TINT, s.glass_ior, s.glass_rough));
        }

        // the air
        if let Some(e) = self.volume.effect_mut::<Switch<VolumetricFogEffect>>() {
            e.on = s.fog > 0.0;
            e.effect.anisotropy = s.fog_aniso;
            if let Some(v) = e.effect.local_volumes.first_mut() {
                v.height_extinction = s.fog.max(0.0);
            }
        }
        if let Some(d) = &mut self.dust {
            d.set_visible(&mut self.scene, s.dust);
            d.look = dust::DustLook { size: s.dust_size, brightness: s.dust_bright, opacity: s.dust_opacity, speed: s.dust_speed, mix: s.dust_mix, amount: s.dust_amount.clamp(0.0, 1.0) };
        }

        // the post-processing
        let mut total = exposure_from_ev100(s.ev);
        if let Some(t) = self.volume.effect_mut::<ToneMapEffect>() {
            let o = &mut t.options;
            o.tonemapper = tonemapper(&s.tonemap);
            o.exposure = exposure_from_ev100(s.ev);
            o.exposure_compensation = s.exposure_comp;
            o.local_exposure = s.local_exposure.then(|| LocalExposure::unreal(0.8, 0.85));
            o.grade.white_temperature = s.white_temp;
            o.grade.white_tint = s.tint;
            o.grade.contrast = s.contrast;
            o.grade.saturation = Vec3::new(s.saturation, s.saturation, s.saturation);
            o.grade.shadow_gain = Vec3::new(s.lift, s.lift, s.lift);
            o.grade.gain = Vec3::new(s.gain, s.gain, s.gain);
            o.grade.highlight_gain = Vec3::new(s.highlights, s.highlights, s.highlights);
            o.vignette = s.vignette;
            o.grain = s.grain;
            o.chromatic_aberration = s.chromatic;
            total = t.total_exposure();
        }
        if let Some(e) = self.volume.effect_mut::<Switch<TemporalAAEffect>>() {
            e.on = s.taa;
            e.effect.options.exposure = total;
        }
        if let Some(e) = self.volume.effect_mut::<Switch<MotionBlurEffect>>() {
            e.on = s.motion_blur;
            e.effect.options.amount = s.motion_amount;
        }
        if let Some(e) = self.volume.effect_mut::<Switch<CinematicDepthOfFieldEffect>>() {
            e.on = s.dof;
            let lens: &mut CameraLens = &mut e.effect.lens;
            lens.f_stop = s.f_stop.max(0.5);
            lens.focal_length_mm = (s.focal_mm > 0.0).then_some(s.focal_mm);
            lens.blade_count = s.blades.round().max(0.0) as u32;
            lens.blade_rotation_deg = s.blade_rot;
            e.effect.max_coc_fraction = s.max_coc.max(0.001);
        }
        if let Some(e) = self.volume.effect_mut::<Switch<BloomEffect>>() {
            e.on = s.bloom;
            e.effect.options.intensity = s.bloom_intensity;
            e.effect.options.threshold = s.bloom_threshold;
            e.effect.options.radius = s.bloom_radius;
            e.effect.exposure = total;
        }

        // the camera, the stats
        if changed(|s| s.cam.clone()) {
            let cam = s.cam.clone();
            self.set_view(&cam);
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

    /// The state as JSON: the settings in use and (with stats) the frame and each pass's time.
    fn info(&self) -> String {
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let cpu: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.cpu.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let grid = self.renderer.rt_grid().map(|g| g.stats());
        format!(
            "{{\"settings\":{},\"camera\":\"{}\",\"character\":{},\"pond\":{},\"grid_triangles\":{},\"grid_rebuilds\":{},\"size\":[{},{}],\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"passes\":[{}],\"cpu\":[{}]}}",
            self.settings.json(),
            self.view.unwrap_or("follow"),
            self.player.character.is_some(),
            self.pond.as_ref().map_or(0, Pond::particles),
            grid.map_or(0, |g| g.grid.triangles),
            grid.map_or(0, |g| g.rebuilds),
            self.renderer.width(),
            self.renderer.height(),
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            passes.join(","),
            cpu.join(","),
        )
    }
}

thread_local! {
    static ROOM: RefCell<Option<Rc<RefCell<RoomState>>>> = const { RefCell::new(None) };
}

fn with_room<R>(f: impl FnOnce(&mut RoomState) -> R) -> Option<R> {
    ROOM.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        Some(f(&mut state))
    })
}

/// The room page on canvas `canvas_id`.
#[wasm_bindgen]
pub async fn start_room(canvas_id: &str) -> Result<(), JsValue> {
    start_room_with_loader(canvas_id, |url: String| async move { fetch_bytes(&url).await }).await
}

/// `start_room`, with the packs' bytes from `load` (see `start_with_loader`).
pub async fn start_room_with_loader<L, F>(canvas_id: &str, load: L) -> Result<(), JsValue>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    // the skylight and four lamps cast shadows (the maps feed the fog, the voxels and the fallback)
    renderer.enable_spot_shadows(2048, 5);
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: 70.0, ..Default::default() });
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: BOUNDS_MIN, bounds_max: BOUNDS_MAX, radiance_scale: 1.0, budget_bytes: 0 });
    renderer.enable_rt_grid(grid_options());
    let settings = Settings::from_url();
    let deferred = settings.shadows == "rt";

    set_text("hud", "Loading the room …");
    let assets = Assets::load(&renderer).await;
    let mut scene = Scene::new();
    let with_pond = kansei_wasm::flag("pond", true);
    let mut layout = layout::build(&mut scene, &assets, with_pond.then(Pond::terrain_bounds), &settings.floor, settings.floor_rough, deferred);
    let (pond, surface) = match with_pond {
        true => {
            let (mut p, s) = Pond::new(&renderer, &mut scene, &mut layout.collision, &assets, deferred);
            // `rest=0`: the water never rests (always stepped and drawn)
            p.set_rest(kansei_wasm::flag("rest", true));
            (Some(p), Some(s))
        }
        false => (None, None),
    };
    let dragon = if kansei_wasm::flag("dragon", true) { add_dragon(&mut scene).await } else { None };
    // the sky behind the windows and the skylight (its colours set by `apply`)
    let mut sky = Renderable::new(SphereGeometry::new(300.0, 32, 16), Material::gradient_sky("Sky", &GradientSkyOptions { zenith: [1.0; 3], horizon: [1.0; 3], ground: [0.2; 3], curve: 0.5 }));
    sky.cast_shadow = false;
    let sky = scene.add(SceneNode::Renderable(sky));
    let sky_lighting = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: Some("Room/SkyLighting"), size: 240, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });

    // the character (or, without a pack, a capsule standing in for it), facing the pond
    let page_note = "\n\nThe room renders without one: a capsule stands in for the character.";
    let mut character = load_character(&renderer, &mut scene, &layout.collision, &RoomLit { deferred }, &load, page_note).await;
    let mut bodies = Vec::new();
    if let Some(c) = &mut character {
        if param("at").is_none() {
            c.controller.matcher.teleport(GVec3::from(START), START_HEADING);
        }
        bodies.extend(c.bodies.iter().map(|b| b.index));
    } else {
        bodies.push(look::stand_in(&mut scene, GVec3::from(START), deferred));
    }
    let proxy = character.as_ref().filter(|_| kansei_wasm::flag("rt_body", true)).map(|c| Proxy::new(&mut scene, c));

    let volume = PostProcessingVolume::new(&renderer, build_effects(&renderer, surface, &sky_lighting));
    let dust_count: u32 = param_or("dust_count", 32768);
    let dust = (dust_count > 0).then(|| Dust::new(&renderer, &mut scene, dust_count, GVec3::from(START)));

    let mut camera = Camera::new(45.0, 0.1, 700.0, canvas.aspect());
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let at = if character.is_some() { start.translation } else { GVec3::from(START) };
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(at.x, 0.9, at.z), 4.5).with_mouse_pan(canvas.element());
    controls.set_elevation(0.25);
    let heading = if character.is_some() { kansei_core::animation::motion_matching::yaw_of(start.rotation) } else { START_HEADING };
    controls.set_azimuth(std::f32::consts::PI + heading + param_or("view", 0.0f32).to_radians());
    if character.is_none() {
        stand_in_view(&mut controls);
    }

    let mut player = Player::new(character);
    player.follow = true;
    let mut state = RoomState {
        renderer,
        scene,
        camera,
        controls,
        volume,
        player,
        assets,
        layout,
        pond,
        dust,
        proxy,
        dragon,
        bodies,
        sky,
        sky_lighting,
        settings,
        applied: None,
        view: None,
        stats: None,
        profile_since: 0.0,
        time: 0.0,
        dof_focus: 5.0,
    };
    state.apply();
    if kansei_wasm::flag("profile", false) && !state.settings.stats {
        state.renderer.set_profiling(true);
        state.profile_since = kansei_wasm::now();
    }
    log::info!("Kansei — Motion Matching room (WASM) ready: character {}, gi {}, shadows {}", state.player.character.is_some(), state.settings.gi, state.settings.shadows);
    let state = Rc::new(RefCell::new(state));
    set_page(state.clone());
    ROOM.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The room's state as JSON (with `stats`, each pass's GPU time): for scripted captures.
#[wasm_bindgen]
pub fn room_info() -> String {
    with_room(|s| s.info()).unwrap_or_default()
}

/// Every setting as JSON (`settings`): the panel's starting values.
#[wasm_bindgen]
pub fn room_settings() -> String {
    with_room(|s| s.settings.json()).unwrap_or_default()
}

/// The fixed views' names, as JSON (`follow` first).
#[wasm_bindgen]
pub fn room_views() -> String {
    let names: Vec<String> = VIEWS.iter().map(|v| format!("\"{}\"", v.0)).collect();
    format!("[\"follow\",{}]", names.join(","))
}

/// Change a setting (`settings`' names, as the URL's) live; false for an unknown one.
#[wasm_bindgen]
pub fn room_set(key: &str, value: &str) -> bool {
    with_room(|s| {
        let ok = s.settings.set(key, value);
        if ok {
            s.apply();
        }
        ok
    })
    .unwrap_or(false)
}

/// Whether the furniture meant for parkour is traversable as meant (`layout::check_traversals`),
/// as JSON: `[[name, null or what went wrong], ...]`.
#[wasm_bindgen]
pub fn room_check() -> String {
    with_room(|s| {
        let rows: Vec<String> = layout::check_traversals(&s.layout.collision)
            .into_iter()
            .map(|(name, r)| format!("[\"{name}\",{}]", r.err().map_or("null".to_string(), |e| format!("\"{e}\""))))
            .collect();
        format!("[{}]", rows.join(","))
    })
    .unwrap_or_default()
}
