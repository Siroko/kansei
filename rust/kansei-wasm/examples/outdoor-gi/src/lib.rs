//! Outdoor GI: the Raggare intro's forest (raggare.kansei.graphics) under a low sun, lit by the sun
//! (cascaded shadows) and an atmosphere's sky, with voxel GI through a clipmap round the camera
//! (`Renderer::enable_voxel_clipmap`): the sky's light reaches the forest floor only where the
//! canopy lets it through, and the sunlit ground and trees light what is in their shade. The GI
//! modes compare what lights the shade: the materials' own sky light (`gi=off`), dimmed by the
//! top-down sky occlusion the film uses (`skyocc`) or by the clipmap's sky visibility
//! (`visibility`, read in the material), voxel GI on screen from cones per pixel (`cones`) or the
//! clipmap's probes (`probes`), screen-space GI (`ssgi`), or the hybrid (`rt`,
//! `RtDiffuseGiEffect`): rays through a grid of the scene's triangles near the camera, the clipmap
//! past them.
//!
//! The scene is the film's, loaded at start from where raggare.kansei.graphics serves it
//! (`data.rs`): its terrain, splat and road, its 26 390 spruces and birches and its verge, with
//! the CC0 ground and asphalt scans. The trees, the ground cover and the materials are the film
//! renderer's own procedural ones (`tree_*.rs`, `cover_*.rs`, examples/forest/*.wgsl, shared
//! with the TS page): per species three LODs, culled on the GPU per view with dithered crossfades,
//! the spruces' foliage as card clusters near the camera and an impostor far off (`forest.rs`).
//! The clipmap voxelizes them through its own cull view. With `reflect=1` the road and the lake
//! are wet and reflect the forest, traced through the grid (`Renderer::enable_rt_grid`,
//! `RtReflectionsEffect`). See README.md for the URL parameters.

mod canvas;
mod cover_meshes;
mod cover_textures;
mod data;
mod forest;
mod tree_meshes;
mod tree_textures;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::gi::{ClipmapProbeOptions, ConeShadows, SceneVoxelClipmapOptions, VoxelGIEffect, VoxelGIOptions};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Scene, SceneNode};
use kansei_core::postprocessing::effects::{
    exposure_from_ev100_lens, AtmosphereEffect, ScreenSpaceGIEffect, ScreenSpaceGIOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, ToneMapper, VolumetricFogEffect,
    VolumetricFogOptions, LENS_ATTENUATION_UE4,
};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::rt::{
    RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiHitLighting, RtGiKernel, RtGiMode, RtGiResolution, RtGiShadows, RtGiView, RtGridOptions, RtReflectionsEffect, RtReflectionsOptions,
    RtReflectionsView, RtTraceResolution, SceneRtGridOptions,
};
use kansei_core::shadows::{CascadedShadowOptions, SkyOcclusionOptions};
use kansei_wasm::{flag, now, param, param_or, Canvas, Frame};

use data::{SceneData, Shot};
use forest::{AmbientSources, Build, Forest, TREE_LAYER};

/// What lights the surfaces besides the sun.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gi {
    /// The materials' own sky light: the whole sky, as if no tree stood over them.
    Off,
    /// Their sky light dimmed by the top-down sky occlusion (`Renderer::enable_sky_occlusion`),
    /// as the film does today.
    SkyOcc,
    /// Their sky light dimmed by the sky visibility the clipmap's probes measure
    /// (`kansei_clipmap_sky_visibility`, in the material).
    Visibility,
    /// Voxel GI on screen, cones traced per pixel through the clipmap: the sky past the canopy
    /// and the bounces.
    Cones,
    /// Voxel GI on screen from the clipmap's probes: the same light, cheaper and smoother.
    Probes,
    /// Screen-space GI (`ScreenSpaceGIEffect`): the bounces between what is on screen, and the sky
    /// taken out where what is on screen hides it.
    Ssgi,
    /// The hybrid (`RtDiffuseGiEffect`): one ray for each 2 x 2 pixels through the grid of
    /// triangles near the camera, the hits lit by the sun and the clipmap, the clipmap past them.
    Rt,
}

impl Gi {
    const ALL: [Gi; 7] = [Gi::Off, Gi::SkyOcc, Gi::Visibility, Gi::Cones, Gi::Probes, Gi::Ssgi, Gi::Rt];

    fn name(self) -> &'static str {
        match self {
            Gi::Off => "off",
            Gi::SkyOcc => "skyocc",
            Gi::Visibility => "visibility",
            Gi::Cones => "cones",
            Gi::Probes => "probes",
            Gi::Ssgi => "ssgi",
            Gi::Rt => "rt",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Gi::ALL.into_iter().find(|g| g.name() == name)
    }

    /// AMBIENT_WGSL's mode.
    fn ambient_mode(self) -> u32 {
        match self {
            Gi::SkyOcc => 1,
            Gi::Visibility => 2,
            _ => 0,
        }
    }
}

/// What the screen shows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum View {
    Lit,
    /// The light the GI adds alone.
    Indirect,
    /// The clipmap's voxels and their light.
    Voxels,
}

impl View {
    fn name(self) -> &'static str {
        match self {
            View::Lit => "lit",
            View::Indirect => "indirect",
            View::Voxels => "voxels",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "lit" => Some(View::Lit),
            "indirect" => Some(View::Indirect),
            "voxels" => Some(View::Voxels),
            _ => None,
        }
    }
}

/// The reflections' view by name: `lit`, `reflection`, `mirror` or `cost`.
fn reflection_view(name: &str) -> Option<RtReflectionsView> {
    match name {
        "lit" => Some(RtReflectionsView::Lit),
        "reflection" => Some(RtReflectionsView::Reflection),
        "mirror" => Some(RtReflectionsView::Mirror),
        "cost" => Some(RtReflectionsView::Cost),
        _ => None,
    }
}

fn reflection_view_name(view: RtReflectionsView) -> &'static str {
    match view {
        RtReflectionsView::Lit => "lit",
        RtReflectionsView::Reflection => "reflection",
        RtReflectionsView::Mirror => "mirror",
        RtReflectionsView::Cost => "cost",
    }
}

/// The hybrid's settings (`gi=rt`): the URL's `rtgi_*` parameters and `set_rtgi`'s keys.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RtGi {
    resolution: RtGiResolution,
    denoise: RtGiDenoise,
    kernel: RtGiKernel,
    hit: RtGiHitLighting,
    shadows: RtGiShadows,
    mode: RtGiMode,
    accumulate: bool,
    view: RtGiView,
    /// Metres a ray walks the grid before the clipmap takes over (0: the grid's box).
    near: f32,
}

impl RtGi {
    const KEYS: [&'static str; 9] = ["res", "denoise", "kernel", "hit", "shadows", "mode", "accum", "view", "near"];

    fn from_url() -> Self {
        // an 8 m near field: half the trace of the whole 64 m box, for some light lost under the
        // canopy past it, which the clipmap's voxels are too coarse to hold
        let mut r = Self {
            resolution: RtGiResolution::Half,
            denoise: RtGiDenoise::Svgf,
            kernel: RtGiKernel::Three,
            hit: RtGiHitLighting::Direct,
            shadows: RtGiShadows::Rays,
            mode: RtGiMode::Hybrid,
            accumulate: false,
            view: RtGiView::Lit,
            near: 8.0,
        };
        for key in Self::KEYS {
            if let Some(v) = param(&format!("rtgi_{key}")) {
                r.set(key, &v);
            }
        }
        r
    }

    /// One setting by its key and name; false if either is unknown.
    fn set(&mut self, key: &str, v: &str) -> bool {
        match key {
            "res" => RtGiResolution::from_name(v).map(|x| self.resolution = x),
            "denoise" => RtGiDenoise::from_name(v).map(|x| self.denoise = x),
            "kernel" => RtGiKernel::from_name(v).map(|x| self.kernel = x),
            "hit" => RtGiHitLighting::from_name(v).map(|x| self.hit = x),
            "shadows" => RtGiShadows::from_name(v).map(|x| self.shadows = x),
            "mode" => RtGiMode::from_name(v).map(|x| self.mode = x),
            "accum" => {
                self.accumulate = v == "1" || v == "true";
                Some(())
            }
            "view" => RtGiView::from_name(v).map(|x| self.view = x),
            "near" => v.parse::<f32>().ok().map(|x| self.near = x.max(0.0)),
            _ => None,
        }
        .is_some()
    }

    /// Bring `e` to these settings (its history restarts).
    fn apply_to(&self, e: &mut RtDiffuseGiEffect) {
        e.set_resolution(self.resolution);
        e.denoise = self.denoise;
        e.kernel = self.kernel;
        e.hit_lighting = self.hit;
        e.shadows = self.shadows;
        e.mode = self.mode;
        e.accumulate = self.accumulate;
        e.near_distance = self.near;
        e.reset_history();
    }

    /// The settings as JSON members (for `info`).
    fn json(&self) -> String {
        format!(
            "\"res\":\"{}\",\"denoise\":\"{}\",\"kernel\":\"{}\",\"hit\":\"{}\",\"shadows\":\"{}\",\"mode\":\"{}\",\"accum\":{},\"view\":\"{}\",\"near\":{}",
            self.resolution.name(),
            self.denoise.name(),
            self.kernel.name(),
            self.hit.name(),
            self.shadows.name(),
            self.mode.name(),
            self.accumulate,
            self.view.name(),
            self.near
        )
    }
}

/// The film's shots as camera presets, by a short name and the shot's id in scene.json.
const SHOTS: [(&str, &str); 7] = [
    ("canopy", "Shot01_Canopy"),
    ("trunks", "Shot02_Trunks"),
    ("branches", "Shot03_Branches"),
    ("lake", "Shot04_Lake"),
    ("headlights", "Shot05_Headlights"),
    ("departure", "Shot06_Departure"),
    ("rise", "Shot07_Rise"),
];

/// The field of view (vertical, degrees) of the free camera and the drive.
const FREE_FOV: f32 = 55.0;

/// Where a shot's camera is `local` seconds into it: its position, direction and horizontal field
/// of view (radians) on the film's filmback, its keys interpolated linearly as the film does.
fn shot_pose(shot: &Shot, filmback_mm: [f32; 2], local: f32) -> (glam::Vec3, glam::Vec3, f32) {
    let keys = &shot.camera;
    let i = keys.partition_point(|k| k[0] <= local).clamp(1, keys.len().max(2) - 1).min(keys.len() - 1);
    let k = if keys.len() < 2 {
        keys[0]
    } else {
        let (a, b) = (keys[i - 1], keys[i]);
        let u = ((local - a[0]) / (b[0] - a[0]).max(1e-6)).clamp(0.0, 1.0);
        std::array::from_fn(|c| a[c] + (b[c] - a[c]) * u)
    };
    let (sb, cb) = k[4].sin_cos();
    let (sp, cp) = k[5].sin_cos();
    let hfov = 2.0 * (filmback_mm[0] * 0.5 / shot.lens_mm).atan();
    (glam::Vec3::new(k[1], k[2], k[3]), glam::Vec3::new(sb * cp, sp, -cb * cp).normalize(), hfov)
}

/// How the camera moves.
#[derive(Clone, Copy, Debug, PartialEq)]
enum CameraMode {
    /// Orbit controls (a shot's preset starts them at its camera).
    Orbit,
    /// The film's shots in turn, on its timeline (looping).
    Film,
    /// Along the road at 12 m/s, eyes 1.4 m up.
    Drive,
}

#[derive(Default)]
struct Stats {
    since: f64,
    frames: u32,
    frame_ms: f64,
    gpu_ms: f64,
    gpu_span_ms: f64,
    passes: Vec<(&'static str, f64)>,
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    sky: SkyAtmosphere,
    volume: PostProcessingVolume,
    /// The voxels' debug view: the GI effect's view of them and the display transform alone, no
    /// atmosphere over them.
    debug_volume: PostProcessingVolume,
    sun_light: usize,
    gi: Gi,
    /// ambient.wgsl's mode, read by every material.
    ambient_mode: wgpu::Buffer,
    view: View,
    elevation: f32,
    bearing: f32,
    mode: CameraMode,
    camera_name: String,
    /// Seconds into the film (`CameraMode::Film`) or the drive.
    clock: f32,
    time: f32,
    forest: Forest,
    /// How many times nearer than the film's the trees' LODs switch.
    lod_scale: f32,
    /// How many times nearer than the camera's the shadow maps' LODs switch.
    shadow_lod: f32,
    shots: Vec<Shot>,
    filmback_mm: [f32; 2],
    /// The road's `[s, X, Y, Z, tX, tZ]` rows.
    road: Vec<f32>,
    road_stride: usize,
    data_mib: f64,
    stats: Option<Stats>,
    /// The hybrid's settings (its effect exists with the grid: `gi=rt`, `rt=1` or `reflect=1`).
    rtgi: RtGi,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

/// Exposure for the forest under a sun `elevation` degrees up: the open landscape's (the
/// sky-atmosphere example's curve), opened up for a forest's shade.
fn forest_ev100(elevation: f32) -> f32 {
    auto_ev100(elevation) - 2.3
}

/// Exposure for a sun `elevation` degrees up (sky-atmosphere's curve).
fn auto_ev100(elevation: f32) -> f32 {
    const CURVE: [(f32, f32); 8] = [(-8.0, 5.0), (-4.0, 7.3), (-2.5, 8.2), (0.0, 9.8), (2.0, 11.2), (6.0, 12.8), (15.0, 14.3), (40.0, 15.0)];
    if elevation <= CURVE[0].0 {
        return CURVE[0].1;
    }
    for w in CURVE.windows(2) {
        let ((e0, v0), (e1, v1)) = (w[0], w[1]);
        if elevation <= e1 {
            return v0 + (v1 - v0) * (elevation - e0) / (e1 - e0);
        }
    }
    CURVE[CURVE.len() - 1].1
}

impl State {
    fn apply(&mut self) {
        // the clipmap runs for its GI and its probes (and the hybrid's far field); the effects show
        // the GI on screen
        let clipmap = matches!(self.gi, Gi::Visibility | Gi::Cones | Gi::Probes | Gi::Rt);
        let on_screen = matches!(self.gi, Gi::Cones | Gi::Probes);
        if let Some(gi) = self.renderer.voxel_clipmap_mut() {
            gi.settings.enabled = clipmap;
        }
        self.renderer.queue().write_buffer(&self.ambient_mode, 0, &[self.gi.ambient_mode().to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat());
        let probes = self.renderer.voxel_clipmap().and_then(|g| g.probes()).filter(|_| self.gi == Gi::Probes);
        // the fog: lit by the probes with the clipmap, by the sky dimmed by the sky occlusion with it
        let fog_probes = self.renderer.voxel_clipmap().and_then(|g| g.probes()).filter(|_| clipmap);
        let fog_occlusion = self.renderer.sky_occlusion().filter(|_| self.gi == Gi::SkyOcc);
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.set_clipmap_probes(fog_probes);
            fog.set_sky_occlusion(fog_occlusion);
        }
        let indirect = self.view == View::Indirect;
        if let Some(effect) = self.volume.effect_mut::<VoxelGIEffect>() {
            effect.enabled = on_screen;
            effect.set_clipmap_probes(probes);
            effect.show_indirect = indirect;
            effect.reset_history();
        }
        if let Some(effect) = self.volume.effect_mut::<ScreenSpaceGIEffect>() {
            effect.enabled = self.gi == Gi::Ssgi;
            effect.show_indirect = indirect;
        }
        let rtgi = self.rtgi;
        if let Some(effect) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            effect.enabled = self.gi == Gi::Rt;
            rtgi.apply_to(effect);
            effect.view = if indirect { RtGiView::Indirect } else { rtgi.view };
        }
    }

    fn reset_history(&mut self) {
        self.camera.reset_motion();
        for volume in [&mut self.volume, &mut self.debug_volume] {
            if let Some(effect) = volume.effect_mut::<VoxelGIEffect>() {
                effect.reset_history();
            }
        }
        if let Some(effect) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            effect.reset_history();
        }
    }

    /// A shot's name (`SHOTS`), `film` or `drive` (`fly`).
    fn set_camera(&mut self, name: &str) {
        self.clock = 0.0;
        self.camera_name = name.to_string();
        self.mode = match name {
            "film" => CameraMode::Film,
            "drive" | "fly" => CameraMode::Drive,
            _ => CameraMode::Orbit,
        };
        self.camera.fov = FREE_FOV;
        if self.mode == CameraMode::Orbit {
            let id = SHOTS.iter().find(|s| s.0 == name).unwrap_or(&SHOTS[1]).1;
            if let Some(shot) = self.shots.iter().find(|s| s.id == id) {
                // the shot's camera halfway through it, as an orbit 8 m round what it looks at
                let (eye, dir, hfov) = shot_pose(shot, self.filmback_mm, shot.duration * 0.5);
                self.camera.fov = vertical_fov(hfov, self.camera.aspect).to_degrees();
                let r = 8.0;
                let target = eye + dir * r;
                self.controls.set_view(Vec3::new(target.x, target.y, target.z), r, (-dir.x).atan2(-dir.z), (-dir.y).asin());
            }
        }
        self.camera.update_projection_matrix();
        self.reset_history();
    }

    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let now = now() * 1000.0;
        let dt = frame.dt.clamp(0.0, 0.1);
        self.time += dt;
        self.clock += dt;
        match self.mode {
            CameraMode::Film => {
                let length = self.shots.iter().map(|s| s.start + s.duration).fold(0.0, f32::max).max(1.0);
                let t = self.clock % length;
                let index = self.shots.iter().rposition(|s| t >= s.start).unwrap_or(0);
                let shot = &self.shots[index];
                let (eye, dir, hfov) = shot_pose(shot, self.filmback_mm, t - shot.start);
                let fov = vertical_fov(hfov, self.camera.aspect).to_degrees();
                if (fov - self.camera.fov).abs() > 1e-3 {
                    // a cut: the GI's history is of another place
                    self.camera.fov = fov;
                    self.camera.update_projection_matrix();
                    self.reset_history();
                }
                self.camera.set_position(eye.x, eye.y, eye.z);
                let at = eye + dir;
                self.camera.look_at(&Vec3::new(at.x, at.y, at.z));
            }
            CameraMode::Drive => {
                // along the road, eyes 1.4 m up, looking 25 m ahead
                let n = self.road.len() / self.road_stride;
                let row = |i: usize| &self.road[(i % n) * self.road_stride..];
                let i = (self.clock * 12.0) as usize % n.max(1);
                let (p, ahead) = (row(i), row((i + 25).min(n - 1)));
                self.camera.set_position(p[1], p[2] + 1.4, p[3]);
                self.camera.look_at(&Vec3::new(ahead[1], ahead[2] + 1.2, ahead[3]));
            }
            CameraMode::Orbit => self.controls.update(&mut self.camera, dt),
        }
        self.forest.update(&mut self.scene, (self.camera.fov.to_radians() * 0.5).tan(), self.lod_scale, self.shadow_lod);
        let sun_dir = direction_from_elevation_bearing(self.elevation, self.bearing);
        self.sky.sun.direction = sun_dir;
        let eye = *self.camera.position();
        if let Some(Light::Directional(l)) = self.scene.get_light_mut(self.sun_light) {
            l.direction = Vec3::new(-sun_dir.x, -sun_dir.y, -sun_dir.z);
            l.color = self.sky.sun_illuminance_at(eye);
            l.intensity = 1.0;
        }
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.update_lights(self.scene.lights());
            fog.time = self.time;
        }
        if let Some(gi) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            gi.update_lights(self.scene.lights());
        }
        self.sky.update(self.renderer.device(), self.renderer.queue(), &mut self.camera);
        let volume = if self.view == View::Voxels && self.renderer.voxel_clipmap().is_some_and(|g| g.settings.enabled) { &mut self.debug_volume } else { &mut self.volume };
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, volume);
        if let Some(stats) = &mut self.stats {
            stats.frames += 1;
            if now - stats.since >= 1000.0 {
                stats.frame_ms = (now - stats.since) / stats.frames as f64;
                stats.frames = 0;
                stats.since = now;
                let profile = self.renderer.take_profile();
                if profile.gpu_frames > 0 {
                    stats.passes = profile.top_passes(usize::MAX);
                    stats.gpu_ms = profile.gpu_ms;
                    stats.gpu_span_ms = profile.gpu_span_ms;
                }
            }
        }
    }

    /// The ray tracing grid's figures as JSON (null without it).
    fn rt_info(&self) -> String {
        let Some(rt) = self.renderer.rt_grid() else { return "null".into() };
        let s = rt.stats();
        let (lo, hi) = rt.grid().bounds();
        format!(
            "{{\"triangles\":{},\"references\":{},\"big\":{},\"sources\":{},\"rebuilt\":{},\"rebuilds\":{},\"cpu_ms\":{:.3},\"mib\":{:.1},\"box\":[[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}]]}}",
            s.grid.triangles, s.grid.references, s.grid.big_triangles, s.sources, s.rebuilt, s.rebuilds, s.cpu_ms, rt.memory_bytes() as f64 / (1 << 20) as f64, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
        )
    }

    /// The reflections' settings and counters as JSON (null without them).
    fn reflect_info(&self) -> String {
        let Some(r) = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtReflectionsEffect>()) else { return "null".into() };
        let s = r.stats().unwrap_or_default();
        format!(
            "{{\"enabled\":{},\"view\":\"{}\",\"res\":\"{}\",\"alpha\":{},\"grid\":{},\"rays\":{},\"hits\":{},\"cells\":{},\"tests\":{},\"max_cost\":{}}}",
            r.enabled,
            reflection_view_name(r.view),
            if r.resolution() == RtTraceResolution::Quarter { "quarter" } else { "half" },
            r.alpha_test,
            r.trace_grid,
            s.rays,
            s.hits,
            s.cells,
            s.tests,
            s.max_cost
        )
    }

    /// The hybrid's settings, and with its effect its counters, memory and accumulated frames
    /// (null without the effect).
    fn rtgi_info(&self) -> String {
        let Some(e) = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtDiffuseGiEffect>()) else { return "null".into() };
        let s = e.stats().unwrap_or_default();
        format!(
            "{{{},\"on\":{},\"accumulated\":{},\"mib\":{:.1},\"rays\":{},\"hits\":{},\"cost\":{},\"shadow_rays\":{}}}",
            self.rtgi.json(),
            e.enabled,
            e.accumulated(),
            e.memory_bytes() as f64 / (1 << 20) as f64,
            s.rays,
            s.hits,
            s.cost,
            s.shadow_rays
        )
    }

    fn info(&self) -> String {
        let gi = self.renderer.voxel_clipmap();
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        // (+ 0.0: an empty sum is -0)
        let sum = |prefix: &str| self.stats.as_ref().map_or(0.0, |s| s.passes.iter().filter(|p| p.0.starts_with(prefix)).map(|p| p.1).sum::<f64>()) + 0.0;
        let eye = self.camera.position();
        let layout = gi.map(|g| *g.clipmap().layout());
        format!(
            "{{\"rtgi\":{},\"rtgi_ms\":{:.3},\"rt\":{},\"rt_ms\":{:.3},\"reflect\":{},\"reflect_ms\":{:.3},\"gi\":\"{}\",\"view\":\"{}\",\"camera\":\"{}\",\"levels\":{},\"dims\":{},\"voxel\":{},\"mib\":{:.1},\"filling\":{},\"trees\":{},\"triangles\":{},\"data_mib\":{:.1},\"elevation\":{},\"bearing\":{},\"eye\":[{:.1},{:.1},{:.1}],\"stats\":{},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"voxelize_ms\":{:.3},\"inject_ms\":{:.3},\"screen_ms\":{:.3},\"ssgi_ms\":{:.3},\"passes\":[{}]}}",
            self.rtgi_info(),
            sum("RtGi/"),
            self.rt_info(),
            sum("Rt/Gather") + sum("Rt/Grid"),
            self.reflect_info(),
            sum("Rt/Trace") + sum("Rt/Resolve"),
            self.gi.name(),
            self.view.name(),
            self.camera_name,
            layout.map_or(0, |l| l.levels),
            layout.map_or("null".into(), |l| format!("{:?}", l.dims)),
            layout.map_or(0.0, |l| l.voxel_size),
            gi.map_or(0.0, |g| g.memory_bytes() as f64 / (1 << 20) as f64),
            gi.is_some_and(|g| g.filling()),
            self.forest.tree_count,
            self.forest.triangles,
            self.data_mib,
            self.elevation,
            self.bearing,
            eye.x,
            eye.y,
            eye.z,
            self.stats.is_some(),
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            sum("VoxelClipmap/Voxelize") + sum("VoxelClipmap/Dynamic"),
            sum("VoxelClipmap/Inject"),
            sum("VoxelGI/Screen"),
            sum("SSGI"),
            passes.join(","),
        )
    }
}

/// Vertical field of view (radians) for a picture of `aspect` (width / height) at horizontal
/// field of view `hfov`, at most the film's own (2.39:1 letterbox) height: a narrow window shows
/// the shot's width.
fn vertical_fov(hfov: f32, aspect: f32) -> f32 {
    2.0 * ((hfov * 0.5).tan() / aspect.max(1.0)).atan()
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    // the film's scene, from raggare.kansei.graphics (data=<base URL> for another copy)
    let base = param("data").map(|b| if b.ends_with('/') { b } else { format!("{b}/") }).unwrap_or_else(|| data::DEFAULT_BASE.into());
    let started = now();
    let data: SceneData = data::load(&base).await?;
    log::info!("Outdoor GI: {:.1} MB of the Raggare intro's data from {base} in {:.0} ms", data.bytes as f64 / 1e6, now() - started);

    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: param_or("shadow_far", 160.0), caster_distance: 120.0, resolution: param_or("shadow_res", 1024), ..Default::default() });
    // levels=<n>, res=<voxels across>, vox=<finest voxel, metres>
    let options = SceneVoxelClipmapOptions {
        levels: param_or("levels", 5),
        resolution: param_or("res", 64),
        height_resolution: param_or("res", 64u32) / 2,
        voxel_size: param_or("vox", 0.5),
        ..Default::default()
    };
    renderer.enable_voxel_clipmap(options);
    // the film's sky occlusion, for comparison: the trees seen from above
    renderer.enable_sky_occlusion(SkyOcclusionOptions { extent_m: 320.0, min_height_m: -10.0, max_height_m: 90.0, volume_size: (128, 32), layer_mask: TREE_LAYER, ..Default::default() });
    // rt=1: a ray tracing grid of the scene's triangles round the camera, 64 x 32 x 64 m of 0.5 m
    // cells, its trees by their cluster cut at a cell of error; rt_cell=<m> (the box stays 64 m
    // across), rt_rebuild=1 rebuilds it every frame
    // reflect=1: ray-traced reflections on the road and the lake, wet (wet=all: everywhere),
    // through the grid; gi=rt: the hybrid GI's rays through it
    let reflect = flag("reflect", false);
    let gi_mode = param("gi").and_then(|g| Gi::from_name(&g)).unwrap_or(Gi::Rt);
    let rt = flag("rt", false) || reflect || gi_mode == Gi::Rt;
    // the wet surfaces' F0 (the road's, the rest's) and roughness, 0 when dry (the default)
    let wet = match (reflect, param("wet").as_deref()) {
        (false, _) => [0.0; 3],
        (true, Some("all")) => [param_or("wet_f0", 0.04), param_or("wet_f0", 0.04), param_or("wet_rough", 0.1)],
        (true, _) => [param_or("wet_f0", 0.04), 0.0, param_or("wet_rough", 0.1)],
    };
    if rt {
        let cell: f32 = param_or("rt_cell", 0.5);
        let across = (64.0 / cell / 4.0).round() as u32 * 4;
        renderer.enable_rt_grid(SceneRtGridOptions {
            grid: RtGridOptions { dims: [across, across / 2, across], cell, below: 0.25, ..Default::default() },
            cluster_error_cells: 1.0,
            rebuild_every_frame: flag("rt_rebuild", false),
        });
    }

    let mut sky = SkyAtmosphere::new(renderer.device(), SkyAtmosphereOptions::default());
    sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);
    let gi = renderer.voxel_clipmap_mut().unwrap();
    gi.use_sky_lighting(&sky.bindings().sky_lighting);
    // lit=<levels a frame> (0: all), probes=<probes a frame>
    gi.settings.levels_per_frame = param_or("lit", gi.settings.levels_per_frame);
    // coneshadows=off|fallback|always, bounce=<share>, shadowsteps=<n>
    if let Some(mode) = param("coneshadows").and_then(|m| ConeShadows::from_name(&m)) {
        gi.settings.cone_shadows = mode;
    }
    gi.settings.bounce = param_or("bounce", gi.settings.bounce);
    gi.settings.cone_shadow_steps = param_or("shadowsteps", gi.settings.cone_shadow_steps);
    gi.enable_probes(ClipmapProbeOptions { probes_per_frame: param_or("probes", ClipmapProbeOptions::default().probes_per_frame), ..Default::default() });
    let ambient_mode = renderer.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("AmbientMode"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // the spruces' card clusters prune at an error of 16 render pixels (card_error=<px>; the far
    // crowns are the impostors'), four times that in the shadow maps, as the film's
    renderer.set_cluster_error_threshold(param_or("card_error", 16.0));
    renderer.set_shadow_cluster_error_scale(4.0);

    let mut scene = Scene::new();
    // the sun first: the impostors bake with the renderer's lights
    let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 1.0);
    sun.cast_shadow = true;
    let sun_light = scene.add(SceneNode::Light(Light::Directional(sun)));
    let ambient = AmbientSources::new(&sky, &ambient_mode, renderer.sky_occlusion().unwrap(), renderer.voxel_clipmap().unwrap().probes().unwrap());
    let started = now();
    let forest = Forest::build(&mut renderer, &mut scene, &data, &Build { ambient, rt, wet })?;
    // hide=<labels>: leave parts out, to measure them (`set_hidden`)
    hide(&mut scene, param("hide").as_deref().unwrap_or(""));
    if !flag("impostor_shadows", true) {
        for i in 0..scene.children_len() {
            if let Some(r) = scene.get_renderable_mut(i).filter(|r| r.geometry.label == "TreeImpostor") {
                r.cast_shadow = false;
            }
        }
    }
    log::info!("Outdoor GI: built the forest ({} trees) in {:.0} ms", forest.tree_count, now() - started);

    // the chain: the GI first (it lies on the surfaces under the aerial perspective), the
    // atmosphere, the fog, TAA, the display transform
    let elevation: f32 = param_or("elevation", 14.0);
    let bearing: f32 = param_or("bearing", 110.0);
    let ev = param_or("ev", forest_ev100(elevation));
    let tonemap = ToneMapEffect::new(ToneMapOptions {
        exposure: exposure_from_ev100_lens(ev, LENS_ATTENUATION_UE4),
        exposure_compensation: 1.0,
        tonemapper: ToneMapper::AcesFitted,
        ..ToneMapOptions::for_surface(renderer.presentation_format())
    });
    let mut gi = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions { intensity: param_or("intensity", 1.0), ..Default::default() });
    gi.set_sky_lighting(Some(&sky.bindings().sky_lighting));
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(gi)];
    // screen-space GI (gi=ssgi): the bounces between what is on screen, the sky taken out where it
    // is hidden
    let mut ssgi = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { radius_m: param_or("ssgi_radius", 6.0), ..Default::default() });
    ssgi.set_sky_lighting(Some(&sky.bindings().sky_lighting));
    effects.push(Box::new(ssgi));
    // with the grid, the hybrid (on in gi=rt): rays through the grid, the foliage alpha-tested, the
    // hits lit by the sun (shadow rays, the cascades past the grid) and the clipmap, the clipmap
    // and the sky past the grid
    if rt {
        let mut hybrid = RtDiffuseGiEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), renderer.rt_grid().unwrap().handle(), RtDiffuseGiOptions::default());
        hybrid.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        hybrid.set_alpha_texture(Some(forest.atlases.foliage_view()));
        hybrid.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        hybrid.heat_scale = exposure_from_ev100_lens(ev, LENS_ATTENUATION_UE4).recip() * 0.5;
        hybrid.collect_stats = flag("stats", false);
        effects.push(Box::new(hybrid));
    }
    // the reflections: after the GI (they reflect its light), under the aerial perspective
    if reflect {
        let options = RtReflectionsOptions {
            resolution: if param("rt_res").as_deref() == Some("quarter") { RtTraceResolution::Quarter } else { RtTraceResolution::Half },
            alpha_test: flag("rt_alpha", true),
            ..Default::default()
        };
        let mut reflections = RtReflectionsEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), renderer.rt_grid().unwrap().handle(), options);
        reflections.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        reflections.set_alpha_texture(Some(forest.atlases.foliage_view()));
        reflections.view = param("rt_view").and_then(|v| reflection_view(&v)).unwrap_or_default();
        reflections.heat_scale = exposure_from_ev100_lens(ev, LENS_ATTENUATION_UE4).recip() * 0.5;
        reflections.collect_stats = flag("stats", false);
        reflections.trace_grid = param("rt_trace").as_deref() != Some("voxels");
        effects.push(Box::new(reflections));
    }
    effects.push(Box::new(AtmosphereEffect::new(&sky)));
    // fog=<density>: mist in the forest, lit by the sun through the cascades and by the sky, or by
    // the clipmap's probes in its modes (the light of the sunlit road, the sky past the trees)
    let fog_density: f32 = param_or("fog", 0.0015);
    if fog_density > 0.0 {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 240.0, temporal: true, blend_factor: 0.1, ..Default::default() },
            base_density: fog_density,
            height_falloff: 0.06,
            anisotropy: 0.55,
            albedo: Vec3::new(0.85, 0.88, 0.93),
            ..Default::default()
        });
        fog.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);
    let mut voxels = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
    voxels.show_voxels = true;
    let debug_tonemap = ToneMapEffect::new(ToneMapOptions {
        exposure: exposure_from_ev100_lens(ev, LENS_ATTENUATION_UE4),
        exposure_compensation: 1.0,
        tonemapper: ToneMapper::AcesFitted,
        ..ToneMapOptions::for_surface(renderer.presentation_format())
    });
    let debug_volume = PostProcessingVolume::new(&renderer, vec![Box::new(voxels), Box::new(debug_tonemap)]);
    let camera = Camera::new(FREE_FOV, 0.2, 4000.0, canvas.aspect());
    let controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.0, 0.0), 10.0).with_mouse_pan(canvas.element());

    let stats = flag("stats", false).then(|| Stats { since: now() * 1000.0, ..Default::default() });
    if stats.is_some() {
        renderer.set_profiling(true);
    }
    let mut state = State {
        renderer,
        scene,
        camera,
        controls,
        sky,
        volume,
        debug_volume,
        sun_light,
        gi: gi_mode,
        ambient_mode,
        view: param("view").and_then(|v| View::from_name(&v)).unwrap_or(View::Lit),
        elevation,
        bearing,
        mode: CameraMode::Orbit,
        camera_name: String::new(),
        clock: 0.0,
        time: 0.0,
        forest,
        lod_scale: param_or("lod", 1.5),
        shadow_lod: param_or("shadow_lod", 2.0),
        shots: data.scene.film.shots.clone(),
        filmback_mm: data.scene.film.filmback_mm,
        road: data.road.clone(),
        road_stride: data.scene.road.stride,
        data_mib: data.bytes as f64 / (1 << 20) as f64,
        stats,
        rtgi: RtGi::from_url(),
    };
    drop(data);
    state.set_camera(param("cam").as_deref().unwrap_or("departure"));
    state.apply();
    log::info!("Kansei — Outdoor GI (WASM) ready: {}", state.info());

    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The state as JSON: the settings, the clipmap, the scene and (with stats) the times.
#[wasm_bindgen]
pub fn info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// `off`, `skyocc`, `visibility`, `cones`, `probes`, `ssgi` or `rt` (`Gi`; `rt` only where the
/// page built the grid: `gi=rt`, the default, `rt=1` or `reflect=1`).
#[wasm_bindgen]
pub fn set_gi(name: &str) {
    with_state(|s| {
        if let Some(gi) = Gi::from_name(name) {
            if gi == Gi::Rt && s.volume.effect_mut::<RtDiffuseGiEffect>().is_none() {
                return;
            }
            s.gi = gi;
            s.apply();
        }
    });
}

/// One of the hybrid's settings at run time, by its `rtgi_*` URL parameter's key (`res`,
/// `denoise`, `kernel`, `hit`, `shadows`, `mode`, `accum`, `view`, `near`) and value.
#[wasm_bindgen]
pub fn set_rtgi(key: &str, value: &str) {
    with_state(|s| {
        if s.rtgi.set(key, value) {
            s.apply();
        }
    });
}

/// `lit`, `indirect` or `voxels`.
#[wasm_bindgen]
pub fn set_view(name: &str) {
    with_state(|s| {
        if let Some(view) = View::from_name(name) {
            s.view = view;
            s.apply();
        }
    });
}

/// A shot of the film as a starting view (`canopy`, `trunks`, `branches`, `lake`, `headlights`,
/// `departure`, `rise`), `film` (the shots in turn) or `drive` (along the road).
#[wasm_bindgen]
pub fn set_camera(name: &str) {
    with_state(|s| s.set_camera(name));
}

/// The sun's elevation, degrees (the exposure follows).
#[wasm_bindgen]
pub fn set_elevation(degrees: f32) {
    with_state(|s| {
        s.elevation = degrees;
        for volume in [&mut s.volume, &mut s.debug_volume] {
            if let Some(tonemap) = volume.effect_mut::<ToneMapEffect>() {
                tonemap.options.exposure = exposure_from_ev100_lens(forest_ev100(degrees), LENS_ATTENUATION_UE4);
            }
        }
    });
}

/// The sun's bearing, degrees.
#[wasm_bindgen]
pub fn set_bearing(degrees: f32) {
    with_state(|s| s.bearing = degrees);
}

/// Leave out the renderables whose label contains one of the comma-separated `labels` (Terrain,
/// Road, Lake, Spruce, Birch, Grass, Flowers, Shrubs, TreeImpostor; /Bark/, /Foliage/, /Cards
/// match within the trees' labels), and draw the rest: to measure what each part costs.
#[wasm_bindgen]
pub fn set_hidden(labels: &str) {
    with_state(|s| hide(&mut s.scene, labels));
}

fn hide(scene: &mut Scene, labels: &str) {
    for i in 0..scene.children_len() {
        if let Some(r) = scene.get_renderable_mut(i) {
            r.visible = !labels.split(',').any(|h| !h.is_empty() && r.geometry.label.contains(h));
        }
    }
}

/// Per-pass GPU times in `info()`.
#[wasm_bindgen]
pub fn set_stats(on: bool) {
    with_state(|s| {
        s.renderer.set_profiling(on);
        s.stats = on.then(|| Stats { since: now() * 1000.0, ..Default::default() });
    });
}

fn with_reflections(f: impl FnOnce(&mut RtReflectionsEffect)) {
    with_state(|s| {
        if let Some(r) = s.volume.effect_mut::<RtReflectionsEffect>() {
            f(r);
        }
    });
}

/// The ray-traced reflections on or off (with `reflect=1`).
#[wasm_bindgen]
pub fn set_reflections(on: bool) {
    with_reflections(|r| {
        r.enabled = on;
        r.reset_history();
    });
}

/// `lit`, `reflection` (the light they add), `mirror` (what the rays see) or `cost`.
#[wasm_bindgen]
pub fn set_reflection_view(name: &str) {
    with_reflections(|r| r.view = reflection_view(name).unwrap_or_default());
}

/// The foliage's alpha test in the reflections (off: the cards are solid).
#[wasm_bindgen]
pub fn set_reflection_alpha(on: bool) {
    with_reflections(|r| r.alpha_test = on);
}

/// Trace the grid of triangles, or (off) the voxel cone alone.
#[wasm_bindgen]
pub fn set_reflection_grid(on: bool) {
    with_reflections(|r| r.trace_grid = on);
}

/// `half` or `quarter`: one pixel of each 2 x 2 or 4 x 4 traced a frame.
#[wasm_bindgen]
pub fn set_reflection_resolution(name: &str) {
    with_reflections(|r| r.set_resolution(if name == "quarter" { RtTraceResolution::Quarter } else { RtTraceResolution::Half }));
}
