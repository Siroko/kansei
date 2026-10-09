//! The room page (`www/room.html`, [`start_room`]): the motion-matched character in a 40 x 40 m
//! room lit by ray-traced global illumination.
//!
//! - The light: a panel in the ceiling, top-down, its light a spot light straight down with a
//!   wide PCSS emitter (soft, contact-hardening shadows, as an area light's), and three floor
//!   lamps, warm shadowed downlights under glowing shades (`layout`).
//! - The GI: the hybrid (`RtDiffuseGiEffect`, `gi=rt`, the default: a ray for each 2 x 2 pixels
//!   through a grid of the room's triangles, SVGF 3 x 3), voxel cones (`gi=voxel`), screen space
//!   (`gi=ssgi`) or none (`gi=off`); G switches at run time.
//! - Two mirrors (one sharp, one brushed) and a glass Stanford dragon on an island in a pond in the
//!   middle, ray traced through the same grid (`RtReflectionsEffect`, with its glass pass); the
//!   pond is the engine's SPH water the character wades into (`pond`).
//! - Furniture: sofas, tables, shelves, crates and platforms, the solid ones in the collision world
//!   the character's traversals read (vault, mantle, climb, jump the gap).
//! - Subtle volumetric fog in the room, lit by the same lights through their shadows, and dust
//!   motes stepped by a compute shader that catch the light in the beams (`dust`).
//! - The walls and the ceiling are one-sided: O swings the camera outside, where it looks in
//!   through them (`layout`'s header says how the grid sees them).
//!
//! The character is the lake page's (`character`, `player`), drawn into the GBuffer by
//! `look::RoomLit` and standing in the grid as capsules on its bones (`look::Proxy`), so the
//! mirrors show it. Without a pack a capsule stands in for it.
//!
//! URL parameters (with the lake page's character ones: `pack`, `hero`, `char`, `gait`, `walk`,
//! `run`, `at`, `drive`, `play`, `view`): `gi=rt|voxel|ssgi|off`, `cam=outside|dragon|mirror|
//! parkour|living` (a fixed view), `fog=<extinction per metre>` (0: none), `dust=<count>` (0:
//! none; default 32768) and `dust_size`, `dust_bright`, `dust_opacity`, `dust_speed`, `pond=0`,
//! `rest=0`, `dragon=0`, `taa=0`, `ev=<EV100>` (exposure, default 4.0), `rt_body=0` (the character
//! out of the grid), `stats=1` (each pass's GPU time and the CPU sections on the page, and in
//! `room_info()`), `profile=1` (the profile in the console every 3 s).

mod dust;
mod layout;
mod look;
mod pond;

use std::cell::RefCell;
use std::rc::Rc;

use glam::Vec3 as GVec3;
use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions};
use kansei_core::controls::CameraControls;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::gi::{SceneVoxelGiOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100, FluidSurfaceEffect, LocalFogVolume, ScreenSpaceGIEffect, ScreenSpaceGIOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions,
    VolumetricFogEffect, VolumetricFogOptions,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions, RtSurface, RtTraceResolution, SceneRtGridOptions};
use kansei_wasm::{flag, param, param_or, set_text, Canvas, Frame};

use crate::player::Player;
use crate::{fetch_bytes, load_character, set_page, Page};
use layout::{HALF, HEIGHT, START, START_HEADING};
use look::{Proxy, RoomLit};
use pond::Pond;

/// The diffuse GI on screen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Gi {
    Rt,
    Voxel,
    Ssgi,
    Off,
}

impl Gi {
    const ALL: [Gi; 4] = [Gi::Rt, Gi::Voxel, Gi::Ssgi, Gi::Off];

    fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|g| g.name() == name)
    }

    fn name(self) -> &'static str {
        match self {
            Gi::Rt => "rt",
            Gi::Voxel => "voxel",
            Gi::Ssgi => "ssgi",
            Gi::Off => "off",
        }
    }

    fn label(self) -> &'static str {
        match self {
            Gi::Rt => "hybrid RT",
            Gi::Voxel => "voxel cones",
            Gi::Ssgi => "screen space",
            Gi::Off => "off",
        }
    }
}

/// The renderer's profile over the last second (`stats=1`).
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

struct RoomState {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    player: Player,
    collision: kansei_core::collision::CollisionWorld,
    pond: Option<Pond>,
    dust: Option<Dust>,
    proxy: Option<Proxy>,
    gi: Gi,
    fog: bool,
    /// A fixed view (`VIEWS`: O swings to "outside" and back), or following the character.
    view: Option<&'static str>,
    stats: Option<Stats>,
    profile_since: f64,
    time: f32,
}

use dust::Dust;

impl Page for RoomState {
    fn player(&mut self) -> &mut Player {
        &mut self.player
    }
}

/// The room's box, for voxel GI and the grid: the walls, the floor (and the pond's bed under it)
/// and the ceiling, with a margin.
const BOUNDS_MIN: [f32; 3] = [-HALF - 0.5, -1.0, -HALF - 0.5];
const BOUNDS_MAX: [f32; 3] = [HALF + 0.5, HEIGHT + 0.5, HALF + 0.5];

/// The grid of the room's triangles: 30 cm cells over the room (140 x 36 x 140, within the
/// grid's million cells), fixed.
fn grid_options() -> SceneRtGridOptions {
    let cell = 0.3;
    SceneRtGridOptions {
        grid: RtGridOptions { dims: [140, 36, 140], cell, fixed_origin: Some(glam::Vec3::new(-21.0, -1.2, -21.0)), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    }
}

/// The effect chain: the pond's surface first (on the lit scene), the three GI effects (the one
/// for `gi` enabled), the reflections and the glass, the fog, TAA and the tone map.
fn build_effects(renderer: &Renderer, surface: Option<FluidSurfaceEffect>, gi: Gi, fog_on: bool, stats: bool) -> Vec<Box<dyn PostProcessingEffect>> {
    let ev: f32 = param_or("ev", 4.0);
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    tone.exposure = exposure_from_ev100(ev);
    tone.vignette = 0.25;
    let tonemap = ToneMapEffect::new(tone);
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(surface) = surface {
        effects.push(Box::new(surface));
    }
    let scene_gi = renderer.voxel_gi().expect("the room enables voxel GI");
    let grid = renderer.rt_grid().expect("the room builds the grid");
    let heat = 0.5 / exposure_from_ev100(ev);
    // the hybrid: half resolution, SVGF 3 x 3, the hits lit by the spot lights (shadow rays) and a
    // voxel cone
    let mut hybrid = RtDiffuseGiEffect::with_volume(scene_gi.volume(), grid.handle(), RtDiffuseGiOptions { max_distance: 60.0, ..Default::default() });
    hybrid.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    hybrid.heat_scale = heat;
    hybrid.collect_stats = stats;
    hybrid.enabled = gi == Gi::Rt;
    effects.push(Box::new(hybrid));
    let mut voxel = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), ..Default::default() });
    voxel.enabled = gi == Gi::Voxel;
    effects.push(Box::new(voxel));
    let mut ssgi = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { radius_m: 6.0, ..Default::default() });
    ssgi.enabled = gi == Gi::Ssgi;
    effects.push(Box::new(ssgi));
    // the mirrors and the dragon's glass, whatever the GI
    let mut reflections = RtReflectionsEffect::with_volume(scene_gi.volume(), grid.handle(), RtReflectionsOptions { resolution: RtTraceResolution::Half, max_distance: 60.0, ..Default::default() });
    reflections.screen_hits = true;
    reflections.hit_indirect = gi != Gi::Off;
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    reflections.set_glass(Some(RtGlass::default()));
    reflections.heat_scale = heat;
    reflections.collect_stats = stats;
    effects.push(Box::new(reflections));
    {
        // a faint, even haze filling the room (a box volume; none outside), lit by the panel and
        // the lamps through their shadow maps
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.3, far: 90.0, temporal: true, ..Default::default() },
            base_density: 0.0,
            anisotropy: 0.55,
            ..Default::default()
        });
        let mut haze = LocalFogVolume::new_box(Vec3::new(0.0, HEIGHT * 0.5, 0.0), Vec3::new(HALF, HEIGHT * 0.5, HALF));
        haze.radial_extinction = 0.0;
        haze.height_extinction = param_or("fog", 0.004f32).max(0.0);
        haze.height_offset = -1.0;
        haze.height_falloff = 0.35;
        haze.edge_fade = 0.02;
        haze.albedo = Vec3::new(0.9, 0.9, 0.9);
        fog.local_volumes.push(haze);
        fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        effects.push(Box::new(Switch { effect: fog, on: fog_on }));
    }
    if flag("taa", true) {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    effects.push(Box::new(tonemap));
    effects
}

/// The glass dragon on the plinth: the Stanford dragon (`www/assets`, CC-BY-NC-4.0, credited on the
/// page) about 2.6 m long, clear glass (index 1.5), in the grid by its cluster LOD's cut with its
/// vertex normals, casting no shadow (light passes through glass), outside voxel GI.
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
    let tint = [0.95, 0.98, 0.97];
    let mut dragon = Renderable::new(geometry, Material::standard_lit("Dragon", &StandardLitOptions::glass(tint, 1.5, 0.0)));
    dragon.clusters = Some(ClusterLod::new(mesh));
    dragon.rt = Some(RtSurface::glass(tint));
    dragon.cast_shadow = false;
    dragon.object.set_position(0.0, pond::PLINTH_TOP, 0.0);
    dragon.object.rotation.y = 0.7;
    Some(scene.add(SceneNode::Renderable(dragon)))
}

/// Fixed views (`cam=<name>`, `room_set('cam', name)`): (name, target, distance, azimuth,
/// elevation); the camera stands at the target plus (sin, cos) of the azimuth times the distance.
const VIEWS: [(&str, [f32; 3], f32, f32, f32); 5] = [
    // from the south-east, above the walls' tops, looking in through them and the ceiling
    ("outside", [0.0, 1.5, 0.0], 52.0, 0.65, 0.42),
    // the glass dragon on its plinth, the big mirror behind it
    ("dragon", [0.0, 1.3, 0.0], 6.5, 0.25, 0.12),
    // the big mirror on the north wall at an angle, the pond and the dragon in it
    ("mirror", [0.0, 2.2, -19.5], 14.8, 0.42, 0.09),
    // the crates and platforms in the north-east corner
    ("parkour", [12.0, 1.0, -12.0], 12.0, -0.7, 0.3),
    // the living corner and its lamp
    ("living", [13.0, 0.8, 12.8], 9.0, -0.9, 0.25),
];

impl RoomState {
    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let now = kansei_wasm::now();
        let dt = ((now - self.player.last_frame(now)) as f32).clamp(1e-4, 1.0 / 15.0);
        self.time += dt;
        let RoomState { renderer, scene, controls, player, collision, .. } = self;
        player.follow = self.view.is_none();
        let out = {
            let _t = kansei_core::profiling::cpu_scope("Room/Character");
            player.update(now, dt, scene, renderer, controls, collision)
        };
        for key in &out.keys {
            match key.as_str() {
                "o" => self.set_view(if self.view.is_some() { None } else { Some("outside") }),
                "g" => {
                    let next = Gi::ALL[(Gi::ALL.iter().position(|g| *g == self.gi).unwrap_or(0) + 1) % Gi::ALL.len()];
                    self.set_gi(next);
                }
                "f" => self.set_fog(!self.fog),
                "n" => {
                    if let Some(d) = &self.dust {
                        let on = !d.visible(&self.scene);
                        d.set_visible(&mut self.scene, on);
                    }
                }
                _ => {}
            }
        }
        self.controls.update(&mut self.camera, dt);

        // the character's capsules in the grid, the water, the dust
        let _t = kansei_core::profiling::cpu_scope("Room/World");
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
        if let Some(dust) = &mut self.dust {
            // the legs, and the body above the hips
            let mut capsules = out.legs.clone();
            if let Some(&(hips, _, _)) = out.legs.last() {
                capsules.insert(0, (hips, hips + GVec3::Y * 0.65, 0.25));
            }
            let eye = self.camera.position();
            let height = self.renderer.height() as f32;
            dust.update(&self.renderer, &mut self.scene, GVec3::new(eye.x, eye.y, eye.z), &capsules, dt, height);
        }
        if let Some(gi) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            gi.update_lights(self.scene.lights());
        }
        if let Some(fog) = self.volume.effect_mut::<Switch<VolumetricFogEffect>>() {
            fog.effect.update_lights(self.scene.lights());
            fog.effect.time = self.time;
        }

        drop(_t);

        // HUD, a few times a second
        if self.player.frame.is_multiple_of(10) {
            let room = self.status();
            let hint = " · O inside / outside · G GI · F fog · N dust";
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
                    let mut cpu = profile.cpu.clone();
                    cpu.sort_by(|a, b| b.1.total_cmp(&a.1));
                    stats.cpu = cpu;
                    stats.gpu_ms = profile.gpu_ms;
                    stats.gpu_span_ms = profile.gpu_span_ms;
                }
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

    /// The HUD's lines about the room.
    fn status(&self) -> String {
        let pond = self.pond.as_ref().map_or(String::new(), |p| format!("\npond   {} particles, {}, fastest {:.2} m/s, {} over 0.05", p.particles(), p.state().name(), p.speed().0, p.speed().1));
        format!(
            "room   GI {} · fog {} · dust {} · camera {}{}\n",
            self.gi.label(),
            if self.fog { "on" } else { "off" },
            self.dust.as_ref().map_or("none".to_string(), |d| if d_visible(&self.scene, d) { format!("{} motes", d.count()) } else { "off".into() }),
            self.view.unwrap_or("following"),
            pond,
        )
    }

    /// A fixed view by name, or (None, or an unknown name) back to following the character.
    fn set_view(&mut self, name: Option<&str>) {
        self.view = name.and_then(|n| VIEWS.iter().find(|v| v.0 == n)).map(|v| v.0);
        if let Some(&(_, target, distance, azimuth, elevation)) = VIEWS.iter().find(|v| Some(v.0) == self.view) {
            self.controls.set_view(Vec3::new(target[0], target[1], target[2]), distance, azimuth, elevation);
        } else if let Some(c) = &self.player.character {
            let at = c.controller.matcher.character();
            self.controls.set_view(Vec3::new(at.translation.x, 0.9, at.translation.z), 4.5, std::f32::consts::PI + kansei_core::animation::motion_matching::yaw_of(at.rotation), 0.25);
        } else {
            stand_in_view(&mut self.controls);
        }
    }

    fn set_gi(&mut self, gi: Gi) {
        self.gi = gi;
        if let Some(e) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            e.enabled = gi == Gi::Rt;
        }
        if let Some(e) = self.volume.effect_mut::<VoxelGIEffect>() {
            e.enabled = gi == Gi::Voxel;
        }
        if let Some(e) = self.volume.effect_mut::<ScreenSpaceGIEffect>() {
            e.enabled = gi == Gi::Ssgi;
        }
        if let Some(e) = self.volume.effect_mut::<RtReflectionsEffect>() {
            e.hit_indirect = gi != Gi::Off;
        }
    }

    fn set_fog(&mut self, on: bool) {
        self.fog = on;
        if let Some(e) = self.volume.effect_mut::<Switch<VolumetricFogEffect>>() {
            e.on = on;
        }
    }

    /// The state as JSON: the settings and (with stats) the frame and each pass's GPU time.
    fn info(&self) -> String {
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let cpu: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.cpu.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let grid = self.renderer.rt_grid().map(|g| g.stats());
        format!(
            "{{\"gi\":\"{}\",\"fog\":{},\"dust\":{},\"camera\":\"{}\",\"character\":{},\"pond\":{},\"grid_triangles\":{},\"grid_rebuilds\":{},\"size\":{:?},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"passes\":[{}],\"cpu\":[{}]}}",
            self.gi.name(),
            self.fog,
            self.dust.as_ref().map_or(0, |d| if d_visible(&self.scene, d) { d.count() } else { 0 }),
            self.view.unwrap_or("follow"),
            self.player.character.is_some(),
            self.pond.as_ref().map_or(0, Pond::particles),
            grid.map_or(0, |g| g.grid.triangles),
            grid.map_or(0, |g| g.rebuilds),
            [self.renderer.width(), self.renderer.height()],
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            passes.join(","),
            cpu.join(","),
        )
    }
}

/// Without a character: the pond from beside the stand-in, a little off its axis.
fn stand_in_view(controls: &mut CameraControls) {
    controls.set_view(Vec3::new(START[0], 1.3, START[2] - 5.0), 11.0, 0.4, 0.16);
}

fn d_visible(scene: &Scene, dust: &Dust) -> bool {
    dust.visible(scene)
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
        self.effect.wants_jitter()
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
    // the panel and the three lamps cast shadows
    renderer.enable_spot_shadows(2048, 4);
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: BOUNDS_MIN, bounds_max: BOUNDS_MAX, radiance_scale: 1.0, budget_bytes: 0 });
    renderer.enable_rt_grid(grid_options());

    let mut scene = Scene::new();
    let with_pond = flag("pond", true);
    let mut layout = layout::build(&mut scene, with_pond.then(Pond::terrain_bounds));
    let (pond, surface) = match with_pond {
        true => {
            let (mut p, s) = Pond::new(&renderer, &mut scene, &mut layout.collision);
            // `rest=0`: the water never rests (always stepped and drawn)
            p.set_rest(flag("rest", true));
            (Some(p), Some(s))
        }
        false => (None, None),
    };
    if flag("dragon", true) {
        add_dragon(&mut scene).await;
    }

    // the character (or, without a pack, a capsule standing in for it), facing the pond
    let page_note = "\n\nThe room renders without one: a capsule stands in for the character.";
    let mut character = load_character(&renderer, &mut scene, &layout.collision, &RoomLit, &load, page_note).await;
    if let Some(c) = &mut character {
        if param("at").is_none() {
            c.controller.matcher.teleport(GVec3::from(START), START_HEADING);
        }
    } else {
        look::stand_in(&mut scene, GVec3::from(START));
    }
    let proxy = character.as_ref().filter(|_| flag("rt_body", true)).map(|c| Proxy::new(&mut scene, c));

    let gi = param("gi").and_then(|g| Gi::from_name(&g)).unwrap_or(Gi::Rt);
    let fog = param("fog").is_none_or(|v| v != "0" && v != "off");
    let stats = flag("stats", false);
    let volume = PostProcessingVolume::new(&renderer, build_effects(&renderer, surface, gi, fog, stats));
    let dust_count: u32 = param_or("dust", 32768);
    let dust = (dust_count > 0).then(|| Dust::new(&renderer, &mut scene, dust_count, GVec3::from(START)));

    let mut camera = Camera::new(45.0, 0.1, 400.0, canvas.aspect());
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

    if stats {
        renderer.set_profiling(true);
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
        collision: layout.collision,
        pond,
        dust,
        proxy,
        gi,
        fog,
        view: None,
        stats: stats.then(|| Stats { since: kansei_wasm::now() * 1000.0, ..Default::default() }),
        profile_since: 0.0,
        time: 0.0,
    };
    if let Some(name) = param("cam") {
        state.set_view(Some(&name));
    }
    if let Some(d) = &mut state.dust {
        d.size = param_or("dust_size", d.size);
        d.brightness = param_or("dust_bright", d.brightness);
        d.opacity = param_or("dust_opacity", d.opacity);
        d.speed = param_or("dust_speed", d.speed);
    }
    if flag("profile", false) && !stats {
        state.renderer.set_profiling(true);
        state.profile_since = kansei_wasm::now();
    }
    log::info!("Kansei — Motion Matching room (WASM) ready: character {}, gi {}", state.player.character.is_some(), gi.name());
    let state = Rc::new(RefCell::new(state));
    set_page(state.clone());
    ROOM.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The room's state as JSON (with `stats=1`, each pass's GPU time): for scripted captures.
#[wasm_bindgen]
pub fn room_info() -> String {
    with_room(|s| s.info()).unwrap_or_default()
}

/// Whether the furniture meant for parkour is traversable as meant (`layout::check_traversals`),
/// as JSON: `[[name, null or what went wrong], ...]`.
#[wasm_bindgen]
pub fn room_check() -> String {
    let rows: Vec<String> = layout::check_traversals()
        .into_iter()
        .map(|(name, r)| format!("[\"{name}\",{}]", r.err().map_or("null".to_string(), |e| format!("\"{e}\""))))
        .collect();
    format!("[{}]", rows.join(","))
}

/// Switch the GI (`gi`: `rt`, `voxel`, `ssgi`, `off`), the fog, the dust, the outside camera
/// (`outside`) or a fixed view (`cam`: `outside`, `dragon`, `mirror`, `parkour`, `living`; any other
/// name follows the character).
#[wasm_bindgen]
pub fn room_set(key: &str, value: &str) -> bool {
    with_room(|s| {
        let on = matches!(value, "1" | "true" | "on");
        match key {
            "gi" => match Gi::from_name(value) {
                Some(g) => s.set_gi(g),
                None => return false,
            },
            "fog" => s.set_fog(on),
            "dust" => {
                if let Some(d) = &s.dust {
                    d.set_visible(&mut s.scene, on);
                }
            }
            "outside" => s.set_view(on.then_some("outside")),
            "cam" => s.set_view(Some(value)),
            _ => return false,
        }
        true
    })
    .unwrap_or(false)
}
