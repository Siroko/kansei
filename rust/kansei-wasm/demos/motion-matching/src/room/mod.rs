//! The room page (`www/room.html`, [`start_room`]): the motion-matched character in a 32 x 28 m
//! loft gallery lit by ray tracing, authored in Blender and loaded from `www/assets/room/room.glb`
//! (`layout`: its names and custom properties place the lights, views, mirrors, aquarium, dragon
//! and parkour; see the README's "The room's source").
//!
//! - The light: the low sun through four tall steel windows to the west (warm, its shadows ray
//!   traced through a cone), a skylight over the dragon (a cool area light: a spot light whose
//!   emitter the ray-traced shadows sample as the opening's rectangle), and warm lamps under glowing
//!   shades. A sky behind the windows lights what the rays find through them.
//! - Ray-traced direct light and shadows (`rt::RtShadowsEffect`, `shadows=rt`, the default): soft
//!   where the emitter is large, sharp at contacts, screen-space contact rays for feet and props;
//!   `shadows=maps` falls back to the shadow maps. The GI: the hybrid (`RtDiffuseGiEffect`, `gi=rt`),
//!   voxel cones, screen space or none.
//! - Two mirrors and a glass Stanford dragon on a marble drum, ray traced (`RtReflectionsEffect`);
//!   an aquarium of the engine's SPH water by the windows, its surface ray-traced glass, the sun
//!   through it throwing caustics on the floor (`aquarium`, `RtCausticsEffect`).
//! - CC0 furniture (Poly Haven) and surfaces (ambientCG), KTX2; blocks to vault, mantle and climb,
//!   and a course tagged for moves to come (`layout::check_traversals`, `room_parkour`).
//! - Subtle volumetric fog lit by the same lights, and dust motes from a compute shader drawn with
//!   sprites it generates at start (`dust`).
//! - Post-processing: AgX (or another curve), bloom, a cinematic depth of field focused on the
//!   character, vignette, grain, chromatic aberration, TAA, motion blur.
//! - The walls and the ceiling are one-sided: O swings the camera outside.
//!
//! Every setting is a URL parameter (its initial value) and a control in the page's panel (live):
//! see `settings`. The character is the lake page's (`character`, `player`), drawn by
//! `look::RoomLit` and standing in the grid as capsules on its bones (`look::Proxy`).

mod ao;
mod assets;
mod dust;
mod layout;
mod look;
mod pbr;
mod probe;
mod aquarium;
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
use kansei_core::gi::{SceneVoxelGiOptions, SdfProbeOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
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
    RtCausticsEffect, RtCausticsOptions, RtCausticsView, RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiResolution, RtGiShadows, RtGlass, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions,
    RtShadowsEffect, RtShadowsView, RtSurface, RtTraceResolution, SceneRtGridOptions,
};
use kansei_core::animation::motion_matching::traversal::features::ParkourFeatures;
use kansei_core::shadows::CascadedShadowOptions;
use kansei_wasm::{param, param_or, set_text, Canvas, Frame};

use crate::player::Player;
use crate::{fetch_bytes, load_character, set_page, Page};
use dust::Dust;
use layout::Layout;
use look::{Proxy, RoomLit};
use aquarium::Aquarium;
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

/// A page on the room's renderer: the loft (`room.html`) or the showroom (`showroom.html`).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Venue {
    Room,
    Showroom,
}

impl Venue {
    /// The glbs it is built from: the showroom's architecture, then its furniture if there (kept
    /// out of the repository: `www/assets/showroom/private/`, a link to the owner's exports;
    /// `pieces=0` leaves it out, for captures that may be published).
    fn sources(self) -> Vec<layout::Source<'static>> {
        match self {
            Venue::Room => vec![layout::Source { url: "assets/room/room.glb", required: true, reflective: false }],
            Venue::Showroom => {
                let mut sources = vec![layout::Source { url: "assets/showroom/showroom.glb", required: true, reflective: false }];
                if kansei_wasm::flag("pieces", true) {
                    sources.push(layout::Source { url: "assets/showroom/private/pieces.glb", required: false, reflective: true });
                }
                sources
            }
        }
    }

    /// The body shown first (`char=` another).
    fn body(self) -> &'static str {
        match self {
            Venue::Room => "hero",
            Venue::Showroom => "mannequin",
        }
    }

    fn settings(self) -> Settings {
        match self {
            Venue::Room => Settings::default(),
            Venue::Showroom => Settings::showroom(),
        }
    }

    /// `RtGlass::absorption_scale`: the room's glass (the dragon, the water) is tinted by the
    /// metre, the showroom's thin green vase by the 2 cm.
    fn glass_scale(self) -> f32 {
        match self {
            Venue::Room => 1.0,
            Venue::Showroom => 50.0,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Venue::Room => "room",
            Venue::Showroom => "showroom",
        }
    }
}

/// The glass dragon's tint (the light left after a metre inside).
const GLASS_TINT: [f32; 3] = [0.95, 0.98, 0.97];

struct RoomState {
    canvas: Canvas,
    /// The probes' options while a probe mode is on.
    probes_applied: Option<(f32, f32)>,
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    player: Player,
    layout: Layout,
    aquarium: Option<Aquarium>,
    dust: Option<Dust>,
    proxy: Option<Proxy>,
    dragon: Option<usize>,
    /// The boxes the glass fills, the dragon's and the glb's (the caustics aim their photons at
    /// them and the water).
    glass_boxes: Vec<(GVec3, GVec3)>,
    venue: Venue,
    /// The character's bodies or the stand-in (for the deferred switch).
    bodies: Vec<usize>,
    sky: usize,
    sky_lighting: wgpu::Buffer,
    settings: Settings,
    /// The settings as last applied (what changed since).
    applied: Option<Settings>,
    /// A fixed view (the glb's `CAM_<name>`, `layout.views`' index), or following the character.
    view: Option<usize>,
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

/// The room's box, for voxel GI and the grid: the walls, the floor and the ceiling with the
/// skylight's well, with a margin.
fn room_bounds(layout: &Layout) -> ([f32; 3], [f32; 3]) {
    let (lo, hi) = (layout.min, layout.max);
    ([lo.x - 0.5, lo.y - 1.0, lo.z - 0.5], [hi.x + 0.5, hi.y + 1.5, hi.z + 0.5])
}

/// The grid of the room's triangles: 30 cm cells over the room's box (within the grid's million
/// cells), fixed; the dragon's cut in it within `lod_cm` of its surface (a cell's error would
/// trace a 2.6 m dragon as a lump through the glass).
fn grid_options(lod_cm: f32, (lo, hi): ([f32; 3], [f32; 3])) -> SceneRtGridOptions {
    let cell = 0.3;
    let origin = glam::Vec3::from(lo) - 0.2;
    // (in whole macro cells of 4)
    let dims: [u32; 3] = std::array::from_fn(|k| ((hi[k] - lo[k] + 0.4) / cell / 4.0).ceil() as u32 * 4);
    SceneRtGridOptions {
        grid: RtGridOptions { dims, cell, fixed_origin: Some(origin), ..Default::default() },
        cluster_error_cells: (lod_cm.max(0.0) * 0.01 / cell).max(1e-4),
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

/// `render=forward`'s flat fill in place of the GI (cd/m² per unit albedo).
const FORWARD_FILL: [f32; 3] = [220.0, 225.0, 240.0];

/// The effect chain: the aquarium's surface first (on the lit scene), the ray-traced direct light, the
/// three GI effects (one enabled), the reflections and the glass, the fog, TAA, motion blur, the
/// depth of field, bloom and the tone map. `apply` sets them all to the settings.
fn build_effects(renderer: &Renderer, surface: Option<FluidSurfaceEffect>, sky: &wgpu::Buffer, layout: &Layout, forward: bool, glass_scale: f32) -> Vec<Box<dyn PostProcessingEffect>> {
    let [rect_z, rect_x] = layout.lights.skylight_rect;
    // (the sky lighting lights only the rays that leave through the windows and the skylight: the
    // room's materials add no sky ambient of their own for the GI to take out, so `ambient` 0)
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if let Some(surface) = surface {
        effects.push(Box::new(surface));
    }
    if forward {
        // the post-processing only
        post_effects(renderer, &mut effects);
        return effects;
    }
    let scene_gi = renderer.voxel_gi().expect("the room enables voxel GI");
    let grid = renderer.rt_grid().expect("the room builds the grid");
    let mut shadows = RtShadowsEffect::new(grid.handle());
    // the skylight is the first spot light: its emitter is the opening (the rectangle's width lies
    // along z for a light pointing down, `set_rect_emitter`'s axes)
    shadows.set_rect_emitter(0, rect_z, rect_x);
    effects.push(Box::new(Switch { effect: shadows, on: true }));
    let mut hybrid = RtDiffuseGiEffect::with_volume(scene_gi.volume(), grid.handle(), RtDiffuseGiOptions { max_distance: 60.0, ambient: 0.0, ..Default::default() });
    hybrid.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    hybrid.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
    hybrid.set_sky_lighting(Some(sky));
    effects.push(Box::new(hybrid));
    // voxel GI twice: the cones (or the probes) alone, and with screen-space GI in front for the
    // near field (`voxel+ssgi`, `probes+ssgi`; it is fixed when the effect is made)
    let near = ScreenSpaceGIOptions { radius_m: 6.0, ambient_occlusion: 0.0, ..Default::default() };
    let mut voxel = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), material_ambient: 0.0, ..Default::default() });
    voxel.set_sky_lighting(Some(sky));
    effects.push(Box::new(voxel));
    let mut near_voxel = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), material_ambient: 0.0, near_field: Some(near), ..Default::default() });
    near_voxel.set_sky_lighting(Some(sky));
    effects.push(Box::new(Switch { effect: near_voxel, on: false }));
    let mut ssgi = ScreenSpaceGIEffect::new(near);
    ssgi.set_sky_lighting(Some(sky));
    effects.push(Box::new(ssgi));
    // the dragon's and the water's caustics, over the floor round them (the glb's caustics area),
    // 3 cm texels; the skylight emits from its opening
    let (lo, hi) = layout.caustics;
    let mut caustics = RtCausticsEffect::new(renderer.device(), grid.handle(), RtCausticsOptions { min: lo, max: hi, texel: 0.03, ..Default::default() });
    caustics.set_rect_emitter(0, rect_z, rect_x);
    let (map, lookup) = caustics.map();
    let caustics_map = (map.clone(), lookup.clone());
    effects.push(Box::new(Switch { effect: caustics, on: true }));
    // the mirrors, the floor and the dragon's glass
    let mut reflections = RtReflectionsEffect::with_volume(scene_gi.volume(), grid.handle(), RtReflectionsOptions { resolution: RtTraceResolution::Half, max_distance: 60.0, ..Default::default() });
    reflections.screen_hits = true;
    reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
    // (the glass's reflections traced with the floor's and the mirrors', at their resolution: the
    // water can fill much of the screen)
    reflections.set_glass(Some(RtGlass { reflections_in_trace: true, absorption_scale: glass_scale, ..Default::default() }));
    reflections.set_caustics(Some((&caustics_map.0, &caustics_map.1)));
    reflections.set_sky_lighting(Some(sky));
    effects.push(Box::new(Switch { effect: reflections, on: true }));
    // a faint, even haze filling the room (a box volume; none outside), lit by the sun through the
    // cascades and the spot lights through their shadow maps
    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions { grid: FroxelGridOptions { near: 0.3, far: 90.0, temporal: true, ..Default::default() }, base_density: 0.0, ..Default::default() });
    let (c, h) = ((layout.min + layout.max) * 0.5, (layout.max - layout.min) * 0.5);
    let mut haze = LocalFogVolume::new_box(Vec3::new(c.x, c.y, c.z), Vec3::new(h.x, h.y, h.z));
    haze.radial_extinction = 0.0;
    haze.height_offset = -1.0;
    haze.height_falloff = 0.35;
    haze.edge_fade = 0.02;
    haze.albedo = Vec3::new(0.9, 0.9, 0.9);
    fog.local_volumes.push(haze);
    fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
    effects.push(Box::new(Switch { effect: fog, on: true }));
    post_effects(renderer, &mut effects);
    effects
}

/// The ambient occlusion over the lit image, TAA, motion blur, depth of field, bloom and the tone
/// map.
fn post_effects(renderer: &Renderer, effects: &mut Vec<Box<dyn PostProcessingEffect>>) {
    effects.push(Box::new(ao::AoEffect::new()));
    effects.push(Box::new(Switch { effect: TemporalAAEffect::new(TemporalAAOptions::default()), on: true }));
    effects.push(Box::new(Switch { effect: MotionBlurEffect::new(MotionBlurOptions::default()), on: false }));
    effects.push(Box::new(Switch { effect: CinematicDepthOfFieldEffect::new(CinematicDepthOfFieldOptions::default()), on: false }));
    effects.push(Box::new(Switch { effect: BloomEffect::new(BloomOptions::default()), on: true }));
    effects.push(Box::new(ToneMapEffect::new(ToneMapOptions::for_surface(renderer.presentation_format()))));
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

/// The glass dragon on the plinth (`MARK_Dragon`: where, its yaw and length): the Stanford dragon
/// (`www/assets`, CC-BY-NC-4.0, credited on the page), clear glass, in the grid by its cluster
/// LOD's cut with its vertex normals, casting no shadow (light passes through glass), outside
/// voxel GI.
async fn add_dragon(scene: &mut Scene, (at, yaw, length): (GVec3, f32, f32)) -> Option<(usize, (GVec3, GVec3))> {
    let bytes = match fetch_bytes("assets/stanford_dragon_pbr.glb").await {
        Ok(b) => b,
        Err(e) => {
            log::warn!("no dragon: {e}");
            return None;
        }
    };
    let geometry = GLTFLoader::load_glb(&bytes).ok()?.merged_geometry("Dragon").fit(glam::Vec3::new(length, f32::INFINITY, length));
    let mesh = ClusterMesh::build(&geometry, &ClusterOptions::default());
    let mut dragon = Renderable::new(geometry, Material::standard_lit("Dragon", &StandardLitOptions::glass(GLASS_TINT, 1.5, 0.0)));
    dragon.clusters = Some(ClusterLod::new(mesh));
    dragon.rt = Some(RtSurface::glass(GLASS_TINT));
    dragon.cast_shadow = false;
    dragon.object.set_position(at.x, at.y, at.z);
    dragon.object.rotation.y = yaw;
    let (lo, hi) = dragon.geometry.bounds();
    let placed = kansei_core::rt::transform_box(&(glam::Mat4::from_translation(at) * glam::Mat4::from_rotation_y(yaw)), lo, hi);
    Some((scene.add(SceneNode::Renderable(dragon)), placed))
}

/// Without a character: the room from beside the stand-in (at `spawn`, facing `heading`), a little
/// off its axis.
fn stand_in_view(controls: &mut CameraControls, (spawn, heading): (GVec3, f32)) {
    let at = spawn + GVec3::new(heading.sin(), 0.0, heading.cos()) * 5.0;
    controls.set_view(Vec3::new(at.x, 1.3, at.z), 11.0, heading + std::f32::consts::PI + 0.4, 0.16);
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
                        "voxel" => "voxel+ssgi",
                        "voxel+ssgi" => "probes",
                        "probes" => "probes+ssgi",
                        "probes+ssgi" => "ssgi",
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
            proxy.update(self.renderer.queue(), &mut self.scene, c, visible);
        }
        let view_proj = self.camera.view_projection().to_glam();
        if let Some(aquarium) = &mut self.aquarium {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                aquarium.update(surface, &out.legs, out.landing, dt, view_proj);
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
        // the water's glass (its ripples) and its copy in the grid
        let mut water = None;
        if let Some(aquarium) = &mut self.aquarium {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                aquarium.update_surface(&mut self.scene, self.renderer.queue(), surface, self.time);
                water = aquarium.caustic_water(surface);
            }
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
        // the caustics' casters
        if let Some(e) = volume.effect_mut::<Switch<RtCausticsEffect>>() {
            let x = &mut e.effect;
            x.update_lights(lights.iter().copied());
            x.set_time(self.time);
            let water_box = self.aquarium.as_ref().map(Aquarium::water_box);
            let mut casters: Vec<(GVec3, GVec3)> = self.glass_boxes.clone();
            if let (Some(b), true) = (water_box, water.is_some() && x.water_photons) {
                casters.push(b);
            }
            x.set_casters(&casters);
            x.set_water(water);
            // fed every `caustics_every` frames, four times as seldom with the tank and the dragon
            // out of view (the mirrors may still show them)
            let boxes: Vec<(GVec3, GVec3)> = water_box.into_iter().chain(self.glass_boxes.iter().copied()).collect();
            let (lo, hi) = boxes.iter().fold((GVec3::splat(f32::MAX), GVec3::splat(f32::MIN)), |(lo, hi), (a, b)| (lo.min(*a), hi.max(*b)));
            let seen = kansei_core::culling::aabb_in_frustum(&kansei_core::culling::frustum_planes(view_proj), lo, hi);
            let every = self.settings.caustics_every.round().clamp(1.0, 16.0) as u32;
            x.interval = if seen { every } else { every * 4 };
        }
        drop(world_scope);

        // HUD, a few times a second
        if self.player.frame.is_multiple_of(10) {
            let room = self.status();
            let hint = " · O inside / outside · G GI · T shadows · F fog · N dust · P panel · ` perf";
            match self.player.hud(&room, hint) {
                Some(text) => set_text("hud", &text),
                // no pack: the page's message stays, the room's lines under it
                None => set_text("hud-room", &format!("{room}\ndrag orbit · wheel zoom{hint}")),
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        // the performance overlay's passes (while it shows)
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
                let (rw, rh) = self.renderer.render_size();
                let scaled = if (rw, rh) != (w, h) { format!(" (scene {rw} x {rh})") } else { String::new() };
                let mut text = format!("{w} x {h}{scaled}   frame {:.2} ms   GPU {:.2} ms (span {:.2})\n", stats.frame_ms, stats.gpu_ms, stats.gpu_span_ms);
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
        let aquarium = self.aquarium.as_ref().map_or(String::new(), |p| {
            let speed = p.speed().filter(|_| p.state_name() == "running").map_or(String::new(), |(max, above)| format!(" (fastest {max:.2} m/s, {above} over {:.2})", aquarium::SETTLE_SPEED));
            format!("\naquarium {} particles, {}{speed}", p.particles(), p.state_name())
        });
        format!(
            "{}   {} · GI {} · shadows {} · fog {} · dust {} · camera {}{}\n",
            self.venue.name(),
            if s.render == "forward" { "forward (no ray tracing, no GI: render=hybrid for the full room)" } else { "hybrid" },
            s.gi,
            if s.shadows == "rt" { "ray traced" } else { "shadow maps" },
            if s.fog > 0.0 && s.render != "forward" { "on" } else { "off" },
            self.dust.as_ref().filter(|_| s.dust).map_or("off".to_string(), |d| format!("{} motes", (d.count() as f32 * s.dust_amount.clamp(0.0, 1.0)) as u32)),
            self.view_name().unwrap_or("following"),
            aquarium,
        )
    }

    /// The fixed view's name, if one.
    fn view_name(&self) -> Option<&str> {
        self.view.map(|k| self.layout.views[k].name.as_str())
    }

    /// A fixed view by name, or back to following the character.
    fn set_view(&mut self, name: &str) {
        self.view = self.layout.views.iter().position(|v| v.name == name);
        if let Some(v) = self.view.map(|k| &self.layout.views[k]) {
            self.controls.set_view(Vec3::new(v.target.x, v.target.y, v.target.z), v.distance, v.azimuth, v.elevation);
        } else if let Some(c) = &self.player.character {
            let at = c.controller.matcher.character();
            self.controls.set_view(Vec3::new(at.translation.x, 0.9, at.translation.z), 4.5, std::f32::consts::PI + kansei_core::animation::motion_matching::yaw_of(at.rotation), 0.25);
        } else {
            stand_in_view(&mut self.controls, self.layout.spawn);
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
        // the floor's roughness: its material's, scaled
        if changed(|s| format!("{}", s.floor_rough)) {
            for surface in self.layout.surfaces.iter_mut().filter(|f| f.floor) {
                surface.params.surface[1] = surface.roughness * s.floor_rough.max(0.0);
                if let Some(buffer) = self.scene.get_renderable(surface.index).and_then(|r| r.material.bindable_buffer(0)) {
                    queue.write_buffer(&buffer, 0, bytemuck::bytes_of(&surface.params));
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
                // ray traced, the lamps need no shadow map (only the fog and the GI's hits read
                // them, and the skylight's is the one that shows there): four fewer redraws of
                // the room a frame
                // (forward, for speed, only the sun and the skylight cast shadows)
                l.cast_shadow = !deferred && s.render != "forward";
            }
        }
        if let Some(Light::Spot(l)) = self.scene.get_light_mut(self.layout.lights.display) {
            l.intensity = if s.display_spot { s.display_cd } else { 0.0 };
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
            // the light at the rays' hits shadowed by more rays, or by the maps (the sun's
            // cascades and the skylight's; the lamps' then unshadowed); past `gi_near` metres the
            // voxel cone takes over (0: the whole grid)
            e.shadows = if s.gi_shadows == "maps" { RtGiShadows::Maps } else { RtGiShadows::Rays };
            e.near_distance = s.gi_near.max(0.0);
        }
        // the probes (traced in voxel GI's distance field) only while a probe mode shows them
        let probes = gi.starts_with("probes");
        let near = gi.ends_with("+ssgi");
        if let Some(scene_gi) = self.renderer.voxel_gi_mut() {
            // the probes blend the volume's light over tens of frames: lit every third frame, it
            // costs a third (the cone modes read it every frame)
            scene_gi.lighting_interval = if probes { 3 } else { 1 };
            let wanted = probes.then_some((s.probe_spacing.clamp(2.0, 24.0), s.probes_per_frame.max(0.0)));
            if wanted != self.probes_applied {
                match wanted {
                    // the lattice puts probes in the volume's margin outside the one-sided walls and above
                    // the ceiling, where they see the sky, and the lookups' normal bias reaches
                    // them through the 1-voxel shell: a band of sky light along the top of the
                    // walls at the default 0.25. Those probes see the shell's back faces, so a
                    // stricter limit drops them (0.1: no band, nothing inside lost)
                    Some((spacing, per_frame)) => scene_gi.enable_probes(SdfProbeOptions { spacing_voxels: spacing, probes_per_frame: per_frame as u32, backface_limit: 0.1, ..Default::default() }),
                    None => {
                        scene_gi.disable_probes();
                        scene_gi.disable_sdf();
                    }
                }
                self.probes_applied = wanted;
            }
        }
        let field = self.renderer.voxel_gi().and_then(|g| g.probes()).filter(|_| probes);
        if let Some(e) = self.volume.effect_mut::<VoxelGIEffect>() {
            e.enabled = !near && (gi == "voxel" || probes);
            e.set_probes(field.filter(|_| !near));
        }
        if let Some(e) = self.volume.effect_mut::<Switch<VoxelGIEffect>>() {
            e.on = near && (gi == "voxel+ssgi" || probes);
            e.effect.set_probes(field.filter(|_| near));
        }
        if let Some(e) = self.volume.effect_mut::<ScreenSpaceGIEffect>() {
            e.enabled = gi == "ssgi";
        }
        if let Some(e) = self.volume.effect_mut::<Switch<RtReflectionsEffect>>() {
            e.on = s.reflections;
            e.effect.hit_indirect = gi != "off";
            e.effect.screen_hits = s.refl_screen;
            e.effect.intensity = s.refl_intensity.max(0.0);
            // the voxels' irradiance at the hits the camera doesn't see, brought to the hybrid
            // RT GI's level (`refl_indirect`) so a mirror's far floor matches the floor in view
            e.effect.hit_indirect_scale = if gi == "rt" { s.refl_indirect } else { 1.0 };
            e.effect.set_glass(Some(RtGlass { reflections_in_trace: s.glass_refl_shared, checkerboard: s.glass_checker, absorption_scale: self.venue.glass_scale(), ..Default::default() }));
        }
        if let Some(e) = self.volume.effect_mut::<Switch<RtCausticsEffect>>() {
            // (off, the reflections' and the water's lookups of it too)
            e.effect.set_enabled(self.renderer.queue(), s.caustics);
            e.effect.photons = s.caustics_photons.round().clamp(8.0, 1024.0) as u32;
            e.effect.blend = s.caustics_blend.clamp(0.01, 1.0);
            e.effect.intensity = s.caustics_intensity.max(0.0);
            e.effect.view = if s.caustics_view { RtCausticsView::Caustics } else { RtCausticsView::Lit };
            e.effect.water_photons = s.caustics_water_photons;
            e.effect.area_emitters = s.caustics_area;
            e.effect.focus_gain = s.caustics_glass_gain.max(0.0);
            e.effect.focus_sharpen = s.caustics_sharpen.clamp(0.1, 1.0);
            // (the dragon's glass, else the glb's: the showroom's green vase)
            let (tint, scale) = if self.dragon.is_some() { (GLASS_TINT, 1.0) } else { (self.layout.glass_tint.unwrap_or(GLASS_TINT), self.venue.glass_scale()) };
            e.effect.set_glass(s.glass_ior, tint);
            e.effect.glass_absorption_scale = scale;
        }
        for (k, &(mirror, _)) in self.layout.mirrors.iter().enumerate() {
            let roughness = if k == 0 { s.mirror_rough } else { s.mirror2_rough };
            if let Some(r) = self.scene.get_renderable_mut(mirror) {
                r.material.set_standard_lit(&self.renderer, &StandardLitOptions::mirror([0.92; 3], roughness));
            }
        }
        if let Some(r) = self.dragon.and_then(|d| self.scene.get_renderable_mut(d)) {
            r.material.set_standard_lit(&self.renderer, &StandardLitOptions::glass(GLASS_TINT, s.glass_ior, s.glass_rough));
        }

        // the water
        let fluid = |s: &Settings| {
            let j = s.json();
            j.split(",\"").filter(|f| f.starts_with("fluid") && !f.starts_with("fluid_fill")).collect::<Vec<_>>().join(",")
        };
        if let (Some(aquarium), true) = (&mut self.aquarium, changed(fluid)) {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                aquarium.rebuild_surface(&self.renderer, &mut self.scene, surface, &s);
                aquarium.apply(&mut self.scene, surface, &s);
            }
        }
        if let (Some(aquarium), true) = (&mut self.aquarium, old.as_ref().is_some_and(|o| o.fluid_fill != s.fluid_fill)) {
            if let Some(surface) = self.volume.effect_mut::<FluidSurfaceEffect>() {
                aquarium.reset(surface, s.fluid_fill);
            }
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
        if let Some(e) = self.volume.effect_mut::<ao::AoEffect>() {
            e.enabled = s.ao;
            e.radius = s.ao_radius.clamp(0.02, 4.0);
            e.intensity = s.ao_intensity.clamp(0.0, 6.0);
            e.falloff = s.ao_falloff.clamp(0.0, 8.0);
            e.samples = s.ao_samples.round().clamp(1.0, 32.0) as u32;
            e.debug = s.ao_view;
            // (the occlusion alone shown about mid grey at the exposure)
            e.white = 0.5 / total.max(1e-9);
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

        // the resolution: the canvas's height (or the screen's pixels times the ratio), the
        // scene's share of it (TAA upscales)
        if changed(|s| format!("{} {}", s.resolution, s.dpr)) {
            self.canvas.set_render_height(s.resolution.trim_end_matches('p').parse::<u32>().ok());
            self.canvas.set_pixel_ratio((s.dpr > 0.0).then_some(s.dpr as f64));
        }
        self.renderer.set_render_scale(s.render_scale);

        // the camera, the stats
        if changed(|s| s.cam.clone()) {
            let cam = s.cam.clone();
            self.set_view(&cam);
        }
        if changed(|s| format!("{}", s.lod_px)) {
            self.renderer.set_cluster_error_threshold(s.lod_px.max(0.0));
        }
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

    /// The state as JSON: the settings in use and (with stats) the frame and each pass's time.
    fn info(&self) -> String {
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let cpu: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.cpu.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        let grid = self.renderer.rt_grid().map(|g| g.stats());
        format!(
            "{{\"settings\":{},\"camera\":\"{}\",\"character\":{},\"room_triangles\":{{{}}},\"aquarium\":{},\"fluid\":{},\"proxy_triangles\":{:?},\"grid_triangles\":{},\"grid_rebuilds\":{},\"dragon\":{},\"caustics\":[{},{}],\"size\":[{},{}],\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"passes\":[{}],\"cpu\":[{}]}}",
            self.settings.json(),
            self.view_name().unwrap_or("follow"),
            self.player.character.is_some(),
            self.layout.triangles.iter().map(|(k, n)| format!("\"{k}\":{n}")).collect::<Vec<_>>().join(","),
            self.aquarium.as_ref().map_or(0, Aquarium::particles),
            match (&self.aquarium, self.volume.effect::<FluidSurfaceEffect>()) {
                (Some(aquarium), Some(surface)) => aquarium.surface_json(surface),
                _ => "null".into(),
            },
            self.proxy.as_ref().map_or(Vec::new(), |p| p.triangles(&self.scene)),
            grid.map_or(0, |g| g.grid.triangles),
            grid.map_or(0, |g| g.rebuilds),
            // its mesh's triangles, its cut's on screen and in every view (the grid's...)
            self.dragon.and_then(|i| self.scene.get_renderable(i)).map_or("null".into(), |r| {
                let (camera, all) = self.renderer.cluster_triangles(r).unwrap_or((0, 0));
                format!("{{\"mesh\":{},\"camera\":{camera},\"all\":{all}}}", r.geometry.index_count() / 3)
            }),
            self.volume.effect::<Switch<RtCausticsEffect>>().map_or(0, |e| e.effect.photons_sent().0),
            self.volume.effect::<Switch<RtCausticsEffect>>().map_or(0, |e| e.effect.photons_sent().1),
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
    start_venue(canvas_id, load, Venue::Room).await
}

/// The showroom page (`www/showroom.html`) on canvas `canvas_id`: the room's renderer in a white
/// gallery of furniture (`assets/showroom/showroom.glb`, the pieces from the private
/// `assets/showroom/private/pieces.glb`), the mannequin shown first.
#[wasm_bindgen]
pub async fn start_showroom(canvas_id: &str) -> Result<(), JsValue> {
    start_showroom_with_loader(canvas_id, |url: String| async move { fetch_bytes(&url).await }).await
}

/// `start_showroom`, with the packs' bytes from `load` (see `start_with_loader`).
pub async fn start_showroom_with_loader<L, F>(canvas_id: &str, load: L) -> Result<(), JsValue>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    start_venue(canvas_id, load, Venue::Showroom).await
}

async fn start_venue<L, F>(canvas_id: &str, load: L, venue: Venue) -> Result<(), JsValue>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    // the skylight, the lamps (four in the room, five in the showroom) and the display spot cast
    // shadows (the maps feed the fog, the voxels and the fallback)
    renderer.enable_spot_shadows(2048, if venue == Venue::Showroom { 7 } else { 6 });
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: 70.0, ..Default::default() });
    let mut settings = Settings::from_url(venue.settings());
    let deferred = settings.shadows == "rt";
    // forward (`render=forward`): no voxel GI or ray tracing grid at all, a flat fill for the GI
    let forward = settings.render == "forward";
    layout::set_fill(if forward { FORWARD_FILL } else { [0.0; 3] });
    // forward's reflections: the room's probe, bound now, captured once the room stands
    let probe = forward.then(|| probe::Probe::new(&renderer));
    pbr::set_probe(probe.as_ref().map(probe::Probe::parts));

    // the room (`www/assets/room/room.glb`, from Blender), then voxel GI and the grid over its box
    set_text("hud", &format!("Loading the {} …", venue.name()));
    let mut scene = Scene::new();
    let layout = layout::load(&renderer, &mut scene, deferred, &venue.sources(), venue.glass_scale(), settings.glass_density).await.map_err(|e| {
        set_text("hud", &format!("The {} didn't load: {e}", venue.name()));
        JsValue::from_str(&e)
    })?;
    let bounds = room_bounds(&layout);
    if !forward {
        renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: bounds.0, bounds_max: bounds.1, radiance_scale: 1.0, budget_bytes: 0 });
        renderer.enable_rt_grid(grid_options(settings.grid_lod_cm, bounds));
    }
    // the sun where the room's Blender sun points, unless the URL says
    let d = layout.lights.sun_direction;
    if param("sun_elev").is_none() {
        settings.sun_elev = (-d.y).clamp(-1.0, 1.0).asin().to_degrees();
    }
    if param("sun_azim").is_none() {
        settings.sun_azim = d.z.atan2(d.x).to_degrees();
    }
    let (aquarium, surface) = match (kansei_wasm::flag("aquarium", true), layout.tank) {
        (true, Some(tank)) => {
            let (p, s) = Aquarium::new(&renderer, &mut scene, tank);
            // the fluid's box: the tank's own, unless the URL names it
            let (min, max) = p.bounds();
            for (name, value) in ["fluid_min_x", "fluid_min_y", "fluid_min_z", "fluid_max_x", "fluid_max_y", "fluid_max_z"].iter().zip(min.iter().chain(&max)) {
                if kansei_wasm::param(name).is_none() {
                    settings.set(name, &format!("{:.3}", value));
                }
            }
            (Some(p), Some(s))
        }
        _ => (None, None),
    };
    let (dragon, dragon_box) = match (kansei_wasm::flag("dragon", true), layout.dragon) {
        (true, Some(at)) => add_dragon(&mut scene, at).await.map_or((None, None), |(d, b)| (Some(d), Some(b))),
        _ => (None, None),
    };
    let glass_boxes: Vec<(GVec3, GVec3)> = dragon_box.into_iter().chain(layout.glass.iter().copied()).collect();
    // the sky behind the windows and the skylight (its colours set by `apply`)
    let mut sky = Renderable::new(SphereGeometry::new(300.0, 32, 16), Material::gradient_sky("Sky", &GradientSkyOptions { zenith: [1.0; 3], horizon: [1.0; 3], ground: [0.2; 3], curve: 0.5 }));
    sky.cast_shadow = false;
    let sky = scene.add(SceneNode::Renderable(sky));
    let sky_lighting = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: Some("Room/SkyLighting"), size: 240, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    // the sky the voxel volume's light (its bounce cones, the probes) sees past it: through the
    // windows and the skylight
    if kansei_wasm::flag("vox_sky", true) {
        if let Some(gi) = renderer.voxel_gi_mut() {
            gi.use_sky_lighting(&sky_lighting);
        }
    }

    // the character (or, without a pack, a capsule standing in for it), at the spawn marker
    let page_note = "\n\nThe room renders without one: a capsule stands in for the character.";
    let mut character = load_character(&renderer, &mut scene, &layout.collision, &RoomLit { deferred }, &load, page_note, venue.body()).await;
    let mut bodies = Vec::new();
    if let Some(c) = &mut character {
        // the parkour the room's tagged pieces give: hangs and shimmies on its ledges, wall runs,
        // bars, jumps aimed along its platforms
        c.controller.features = ParkourFeatures::from_pieces(&layout::parkour_pieces(&layout.traversals), &layout.collision);
        log::info!("room: parkour {} edges, {} bars, {} wall faces, {} platforms", c.controller.features.edges.len(), c.controller.features.bars.len(), c.controller.features.faces.len(), c.controller.features.platforms.len());
        if param("at").is_none() {
            c.controller.matcher.teleport(layout.spawn.0, layout.spawn.1);
        }
        bodies.extend(c.bodies.iter().map(|b| b.index));
    } else {
        bodies.push(look::stand_in(&mut scene, layout.spawn.0, deferred));
    }
    let body_layers = layout.albedo_layers.len() as u8;
    let proxy = character.as_ref().filter(|_| !forward && kansei_wasm::flag("rt_body", true)).map(|c| Proxy::new(&renderer, &mut scene, c, deferred, body_layers));

    if let Some(probe) = &probe {
        // from the middle of the room at eye height, the character left out
        let (lo, hi) = (layout.min, layout.max);
        let at = GVec3::new((lo.x + hi.x) * 0.5, lo.y + 1.4, (lo.z + hi.z) * 0.5);
        probe.capture(&mut renderer, &mut scene, at, &bodies);
        log::info!("{}: forward probe captured at {at:?}", venue.name());
    }
    let mut volume = PostProcessingVolume::new(&renderer, build_effects(&renderer, surface, &sky_lighting, &layout, forward, venue.glass_scale()));
    // the albedo the reflections' hits read: the surfaces' colour maps and the bodies' (each
    // textured body a layer after the surfaces', in order: `Proxy::new`)
    let body_maps: Vec<kansei_core::buffers::Texture> = character.iter().flat_map(|c| c.bodies.iter().filter_map(|b| b.base_color.as_ref())).map(|(img, _)| {
        let mut t = kansei_core::buffers::Texture::from_image("Room/BodyAlbedo", img, true);
        t.initialize_with_data(renderer.device(), renderer.queue());
        t
    }).collect();
    let sources: Vec<&wgpu::TextureView> = layout.albedo_layers.iter().map(|(v, _)| v).chain(body_maps.iter().filter_map(|t| t.view())).collect();
    let (albedo_texture, albedo) = assets::albedo_array(&renderer, &sources);
    if let Some(e) = volume.effect_mut::<Switch<RtReflectionsEffect>>() {
        e.effect.set_albedo_texture(Some(&albedo));
    }
    // the surfaces' albedo in the grid and voxel GI: their colour maps' measured means times their
    // tints, in place of the guesses the layout made before the textures were drawn
    let means = assets::albedo_means(&renderer, &albedo_texture).await;
    let indices: Vec<usize> = scene.ordered_indices().collect();
    for idx in indices {
        let Some(r) = scene.get_renderable_mut(idx) else { continue };
        let Some(layer) = r.rt.and_then(|rt| rt.albedo_layer).map(usize::from).filter(|&l| l < layout.albedo_layers.len()) else { continue };
        let tint = layout.albedo_layers[layer].1;
        let albedo: [f32; 3] = std::array::from_fn(|c| (means[layer][c] * tint[c]).min(1.0));
        if let Some(rt) = r.rt.as_mut() {
            rt.albedo = albedo;
        }
        if let Some(gi) = r.gi.as_mut() {
            gi.albedo = albedo;
        }
    }
    let dust_count: u32 = param_or("dust_count", 65536);
    let (lo, hi) = (layout.min, layout.max);
    let dust = (dust_count > 0 && (!forward || settings.dust)).then(|| Dust::new(&renderer, &mut scene, dust_count, GVec3::ZERO, (hi.x - lo.x).min(hi.z - lo.z) * 0.5, hi.y - lo.y));

    let mut camera = Camera::new(45.0, 0.1, 700.0, canvas.aspect());
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let at = if character.is_some() { start.translation } else { layout.spawn.0 };
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(at.x, 0.9, at.z), 4.5).with_mouse_pan(canvas.element());
    controls.set_elevation(0.25);
    let heading = if character.is_some() { kansei_core::animation::motion_matching::yaw_of(start.rotation) } else { layout.spawn.1 };
    controls.set_azimuth(std::f32::consts::PI + heading + param_or("view", 0.0f32).to_radians());
    if character.is_none() {
        stand_in_view(&mut controls, layout.spawn);
    }

    let mut player = Player::new(character);
    player.follow = true;
    let mut state = RoomState {
        canvas: canvas.clone(),
        probes_applied: None,
        renderer,
        scene,
        camera,
        controls,
        volume,
        player,
        layout,
        aquarium,
        dust,
        proxy,
        dragon,
        glass_boxes,
        venue,
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
    log::info!("Kansei — Motion Matching {} (WASM) ready: character {}, gi {}, shadows {}", venue.name(), state.player.character.is_some(), state.settings.gi, state.settings.shadows);
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
    let names: Vec<String> = with_room(|s| s.layout.views.iter().map(|v| format!("\"{}\"", v.name)).collect()).unwrap_or_default();
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

/// Fill the aquarium again, still (`fluid_fill` of it).
#[wasm_bindgen]
pub fn room_aquarium_reset() {
    with_room(|s| {
        let fill = s.settings.fluid_fill;
        if let (Some(aquarium), Some(surface)) = (&mut s.aquarium, s.volume.effect_mut::<FluidSurfaceEffect>()) {
            aquarium.reset(surface, fill);
        }
    });
}

/// The pieces of the room tagged for parkour (`kansei_traversal` and its properties), as JSON:
/// `[{"name", "kind", "min", "max", "extras"}, ...]`, for the controller's moves to come.
#[wasm_bindgen]
pub fn room_parkour() -> String {
    with_room(|s| {
        let rows: Vec<String> = s
            .layout
            .traversals
            .iter()
            .map(|t| format!("{{\"name\":\"{}\",\"kind\":\"{}\",\"min\":{:?},\"max\":{:?},\"extras\":{}}}", t.name, t.want, t.min.to_array(), t.max.to_array(), t.extras))
            .collect();
        format!("[{}]", rows.join(","))
    })
    .unwrap_or_default()
}

/// Whether the furniture meant for parkour is traversable as meant (`layout::check_traversals`),
/// as JSON: `[[name, null or what went wrong], ...]`.
#[wasm_bindgen]
pub fn room_check() -> String {
    with_room(|s| {
        let rows: Vec<String> = layout::check_traversals(&s.layout.collision, &s.layout.traversals)
            .into_iter()
            .map(|(name, r)| format!("[\"{name}\",{}]", r.err().map_or("null".to_string(), |e| format!("\"{e}\""))))
            .collect();
        format!("[{}]", rows.join(","))
    })
    .unwrap_or_default()
}
