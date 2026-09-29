//! Motion matching: a skinned character walking, running, stopping and turning under keyboard
//! or gamepad control, animated by searching a database of animation frames
//! (`kansei_core::animation::motion_matching`), on a sunlit ground plane with cascaded shadows
//! and TAA (the character writes motion vectors from last frame's bone palette).
//!
//! The animation comes from a motion-matching pack (`.kmm`) baked with `kansei-anim-bake`. None
//! ships with Kansei: the page loads `pack/locomotion.kmm` next to `index.html` (or `pack=<url>`)
//! and says how to make one when it is missing. See this example's README.
//!
//! Controls: WASD or arrows move relative to the camera, Shift runs, Q toggles strafing (face the
//! camera's direction), mouse drag orbits and the wheel zooms. Gamepad: left stick moves (tilt
//! sets the pace), right stick orbits, A or the right trigger runs, the left bumper toggles
//! strafing. B toggles the trajectory overlay and HUD, K the skeleton, M the mesh, L foot locking.
//!
//! URL parameters: `pack=<url>`, `gait=0` (search every clip whatever the gait, instead of
//! idle + walk or idle + run by the pack's tags), `taa=0`, `walk=<m/s>`, `run=<m/s>` (forward
//! paces; sideways and backward scale with them).

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3 as GVec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::animation::motion_matching::pack::MotionPack;
use kansei_core::animation::motion_matching::{yaw_of, Database, MotionInput, MotionMatcher, MotionMatchingSettings};
use kansei_core::animation::{skinned_lit_material, BonePalette, SkinnedLitParams, SkinnedMesh, PALETTE_BINDING};
use kansei_core::buffers::{BufferType, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    effects::{exposure_from_ev100, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
    PostProcessingEffect, PostProcessingVolume,
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::shadows::{CascadedShadowOptions, CASCADED_SHADOWS_WGSL};

/// The sun's travel direction, its illuminance (lux) and the sky's luminance (cd/m²).
const SUN_DIR: [f32; 3] = [-0.45, -0.6, -0.66];
const SUN: [f32; 3] = [80000.0, 72000.0, 60000.0];
const SKY: [f32; 3] = [4000.0, 5000.0, 7000.0];
/// Paces of the walk and run loops (m/s) forward, sideways and backward: strafing moves at the
/// pace of the loop for its direction relative to the facing. `walk=` and `run=` scale them.
const WALK: [f32; 3] = [2.0, 1.8, 1.5];
const RUN: [f32; 3] = [5.0, 3.5, 3.0];

/// The pace for moving along `direction` (unit, local to the facing: x sideways, y forward) on
/// an ellipse through the forward, sideways and backward paces.
fn pace(paces: [f32; 3], direction: [f32; 2]) -> f32 {
    let along = if direction[1] >= 0.0 { paces[0] } else { paces[2] };
    1.0 / ((direction[1] / along).powi(2) + (direction[0] / paces[1]).powi(2)).sqrt().max(1e-6)
}

/// Ground: grey with a metre grid and a darker 5 m grid, lit by the sun (cascade-shadowed) and the
/// sky, writing no motion (it does not move).
const GROUND_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, sun_dir: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    return out;
}
fn grid(p: vec2<f32>, spacing: f32, width: f32) -> f32 {
    let q = p / spacing;
    let d = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(d.x, d.y) / width, 1.0);
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = vec3<f32>(0.0, 1.0, 0.0);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let lines = max(grid(in.world.xz, 1.0, 1.0) * 0.35, grid(in.world.xz, 5.0, 1.5) * 0.6);
    let base = surface.base_color.rgb * (1.0 - lines);
    let l = -normalize(surface.sun_dir.xyz);
    let lit = base / 3.14159265 * surface.sun.rgb * max(dot(n, l), 0.0) * shadow + base * surface.sky.rgb;
    return vec4<f32>(lit, 1.0);
}
"#;

const SKY_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let up = saturate(normalize(in.dir).y);
    return vec4<f32>(mix(vec3<f32>(9000.0, 9500.0, 10500.0), vec3<f32>(3000.0, 5000.0, 9000.0), sqrt(up)), 1.0);
}
"#;

/// Debug markers: unit boxes placed by a per-instance matrix, one flat colour.
const MARKER_WGSL: &str = r#"
struct Marker { color: vec4<f32> };
@group(0) @binding(0) var<uniform> marker: Marker;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
struct VIn {
    @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>,
    @location(3) m0: vec4<f32>, @location(4) m1: vec4<f32>, @location(5) m2: vec4<f32>, @location(6) m3: vec4<f32>,
};
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(v: VIn) -> VOut {
    let m = mat4x4<f32>(v.m0, v.m1, v.m2, v.m3);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * m * v.position;
    out.normal = normalize((m * vec4<f32>(v.normal, 0.0)).xyz);
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let shade = 0.65 + 0.35 * max(in.normal.y, 0.0);
    return vec4<f32>(marker.color.rgb * shade, 1.0);
}
"#;

fn surface_params(base: [f32; 3]) -> [f32; 16] {
    let d = SUN_DIR;
    [base[0], base[1], base[2], 0.0, d[0], d[1], d[2], 0.0, SUN[0], SUN[1], SUN[2], 0.0, SKY[0], SKY[1], SKY[2], 0.0]
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_secs() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now() / 1000.0
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

/// Show `text` in the page's HUD element.
fn set_hud(text: &str) {
    if let Some(hud) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id("hud")) {
        hud.set_text_content(Some(text));
    }
}

/// Fetch `url` as bytes; the error says what went wrong in words for the page.
async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    let window = web_sys::window().ok_or("no window")?;
    let response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await.map_err(|_| format!("could not fetch {url}"))?;
    let response: web_sys::Response = response.dyn_into().map_err(|_| "not a response".to_string())?;
    if !response.ok() {
        return Err(format!("{url}: HTTP {}", response.status()));
    }
    let buffer = wasm_bindgen_futures::JsFuture::from(response.array_buffer().map_err(|_| "no body".to_string())?).await.map_err(|_| format!("could not read {url}"))?;
    Ok(js_sys::Uint8Array::new(&buffer).to_vec())
}

/// Instanced unit boxes placed by matrices the app rewrites each frame.
struct Markers {
    buffer: wgpu::Buffer,
    index: usize,
    matrices: Vec<Mat4>,
}

impl Markers {
    fn new(renderer: &Renderer, scene: &mut Scene, label: &str, count: usize, color: [f32; 3], x_ray: bool) -> Self {
        use wgpu::util::DeviceExt;
        let matrices = vec![Mat4::ZERO; count];
        let buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&matrices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let instances = ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage).with_vertex_mat4(3);
        let options = if x_ray {
            // drawn over everything, after the opaque pass
            MaterialOptions { transparent: true, depth_write: Some(false), depth_compare: wgpu::CompareFunction::Always, ..Default::default() }
        } else {
            MaterialOptions::default()
        };
        let mut material = Material::new(label, MARKER_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], options);
        material.set_uniform_bindable(0, label, &[color[0], color[1], color[2], 1.0f32]);
        let mut r = Renderable::new(InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), count as u32, vec![instances]), material);
        r.dynamic = true;
        r.cast_shadow = false;
        r.render_order = 10;
        let index = scene.add(SceneNode::Renderable(r));
        Self { buffer, index, matrices }
    }

    fn upload(&self, renderer: &Renderer) {
        renderer.queue().write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.matrices));
    }

    fn set_visible(&self, scene: &mut Scene, visible: bool) {
        if let Some(r) = scene.get_renderable_mut(self.index) {
            r.visible = visible;
        }
    }
}

/// A box from `a` to `b`, `thickness` across.
fn segment(a: GVec3, b: GVec3, thickness: f32) -> Mat4 {
    let d = b - a;
    let length = d.length();
    if length < 1e-5 {
        return Mat4::ZERO;
    }
    let rotation = Quat::from_rotation_arc(GVec3::Y, d / length);
    Mat4::from_scale_rotation_translation(GVec3::new(thickness, length, thickness), rotation, (a + b) * 0.5)
}

/// The character: its database, matcher, mesh and palette, and the debug markers.
struct Character {
    db: Database,
    matcher: MotionMatcher,
    mesh: SkinnedMesh,
    palette: BonePalette,
    index: usize,
    bones: Markers,
    trajectory: Markers,
    /// Tag bits the search may use while walking and while running (all when the pack has no
    /// gait tags or `gait=0`).
    walk_tags: u32,
    run_tags: u32,
    source: String,
}

impl Character {
    fn new(renderer: &Renderer, scene: &mut Scene, pack: MotionPack, gait: bool) -> Result<Self, String> {
        let tags: Vec<String> = pack.meta("tags").unwrap_or("").split(',').map(str::to_string).collect();
        let source = format!("{}\n{}", pack.meta("source").unwrap_or("(no source noted)"), pack.meta("license").unwrap_or("(no licence noted)"));
        let MotionPack { database: db, meshes, .. } = pack;
        let first = meshes.into_iter().next().ok_or("the pack has no mesh")?;
        let mesh = first.mesh;
        let mut palette = BonePalette::new(mesh.skin_joints.len());
        palette.update(&mesh, &db.skeleton.rest_model());
        let params = SkinnedLitParams {
            base_color: first.color,
            sun_direction: [SUN_DIR[0], SUN_DIR[1], SUN_DIR[2], 0.0],
            sun: [SUN[0], SUN[1], SUN[2], 0.0],
            sky: [SKY[0], SKY[1], SKY[2], 0.0],
        };
        let mut r = Renderable::new(mesh.geometry(), skinned_lit_material("Character", params, &mesh, &palette));
        r.dynamic = true;
        let index = scene.add(SceneNode::Renderable(r));
        let bones = Markers::new(renderer, scene, "Bones", db.joint_count(), [30000.0, 20000.0, 4000.0], true);
        bones.set_visible(scene, false);
        // the simulation now and its 3 predicted samples, and each foot's target
        let trajectory = Markers::new(renderer, scene, "Trajectory", 6, [300.0, 1600.0, 3000.0], false);
        let bit = |name: &str| tags.iter().position(|t| t == name).map_or(0, |b| 1u32 << b);
        let (idle, walk, run) = (bit("idle"), bit("walk"), bit("run"));
        let (walk_tags, run_tags) = if gait && walk != 0 && run != 0 { (idle | walk, idle | run) } else { (u32::MAX, u32::MAX) };
        let matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), GVec3::ZERO, 0.0);
        log::info!("motion pack: {} clips, {} frames, {} joints; {} vertices", db.clips.len(), db.frame_count(), db.joint_count(), mesh.vertices.len());
        Ok(Self { db, matcher, mesh, palette, index, bones, trajectory, walk_tags, run_tags, source })
    }
}

/// Keys held and pressed since last frame, from the page's key events.
#[derive(Default)]
struct Keys {
    held: HashSet<String>,
    pressed: Vec<String>,
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    character: Option<Character>,
    keys: Rc<RefCell<Keys>>,
    last: f64,
    frame: u32,
    strafe: bool,
    overlay: bool,
    /// Gamepad buttons held last frame (to toggle on a press, not while held).
    pad_held: Vec<bool>,
    /// Walk and run paces: forward, sideways, backward.
    speeds: ([f32; 3], [f32; 3]),
    fps: f32,
    searches: u32,
    switches: u32,
    counted_since: f64,
    rates: (f32, f32),
}

/// Left stick, right stick, and the pressed state of each button of the first connected gamepad.
fn gamepad() -> Option<([f32; 2], [f32; 2], Vec<(bool, f32)>)> {
    let pads = web_sys::window()?.navigator().get_gamepads().ok()?;
    let pad: web_sys::Gamepad = (0..pads.length()).find_map(|i| pads.get(i).dyn_into().ok())?;
    let axes: Vec<f32> = pad.axes().iter().map(|a| a.as_f64().unwrap_or(0.0) as f32).collect();
    let axis = |i: usize| axes.get(i).copied().unwrap_or(0.0);
    let dead = |x: f32, y: f32| {
        let m = (x * x + y * y).sqrt();
        if m < 0.15 { [0.0, 0.0] } else { let s = ((m - 0.15) / 0.85).min(1.0) / m; [x * s, y * s] }
    };
    let buttons = pad.buttons().iter().map(|b| b.dyn_into::<web_sys::GamepadButton>().map(|b| (b.pressed(), b.value() as f32)).unwrap_or((false, 0.0))).collect();
    Some((dead(axis(0), axis(1)), dead(axis(2), axis(3)), buttons))
}

impl State {
    fn frame(&mut self) {
        let now = now_secs();
        let dt = ((now - self.last) as f32).clamp(1e-4, 1.0 / 15.0);
        self.last = now;
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;

        // input: keyboard, then the gamepad on top
        let (mut stick, mut run, mut toggles) = ([0.0f32; 2], false, Vec::new());
        {
            let mut keys = self.keys.borrow_mut();
            let held = |k: &[&str]| k.iter().any(|k| keys.held.contains(*k));
            stick[0] = held(&["d", "arrowright"]) as i32 as f32 - held(&["a", "arrowleft"]) as i32 as f32;
            stick[1] = held(&["w", "arrowup"]) as i32 as f32 - held(&["s", "arrowdown"]) as i32 as f32;
            run = held(&["shift"]);
            toggles.append(&mut keys.pressed);
        }
        let length = (stick[0] * stick[0] + stick[1] * stick[1]).sqrt();
        if length > 1.0 {
            stick = [stick[0] / length, stick[1] / length];
        }
        if let Some((left, right, buttons)) = gamepad() {
            if left != [0.0, 0.0] {
                stick = [left[0], -left[1]];
            }
            self.controls.rotate(-right[0] * 2.5 * dt, right[1] * 1.5 * dt);
            let pressed = |i: usize| buttons.get(i).is_some_and(|b| b.0);
            run |= pressed(0) || buttons.get(7).is_some_and(|b| b.1 > 0.3);
            let was = |i: usize| self.pad_held.get(i).copied().unwrap_or(false);
            if pressed(4) && !was(4) {
                toggles.push("q".into());
            }
            self.pad_held = buttons.iter().map(|b| b.0).collect();
        }
        for key in toggles {
            match key.as_str() {
                "q" => self.strafe = !self.strafe,
                "b" => self.overlay = !self.overlay,
                "k" | "m" | "l" => {
                    if let Some(c) = &mut self.character {
                        match key.as_str() {
                            "k" => {
                                let visible = self.scene.get_renderable(c.bones.index).is_some_and(|r| r.visible);
                                c.bones.set_visible(&mut self.scene, !visible);
                            }
                            "m" => {
                                if let Some(r) = self.scene.get_renderable_mut(c.index) {
                                    r.visible = !r.visible;
                                }
                            }
                            _ => c.matcher.settings.foot_lock = !c.matcher.settings.foot_lock,
                        }
                    }
                }
                _ => {}
            }
        }

        // the camera's heading on the ground: the stick moves relative to it
        let azimuth = self.controls.azimuth();
        let forward = GVec3::new(-azimuth.sin(), 0.0, -azimuth.cos());
        let right = GVec3::new(azimuth.cos(), 0.0, -azimuth.sin());
        if let Some(c) = &mut self.character {
            let paces = if run { self.speeds.1 } else { self.speeds.0 };
            // facing the way it moves, the character walks its loops forward; strafing, the pace
            // follows the direction
            let tilt = (stick[0] * stick[0] + stick[1] * stick[1]).sqrt();
            let speed = if self.strafe && tilt > 1e-3 { pace(paces, [stick[0] / tilt, stick[1] / tilt]) } else { paces[0] };
            let velocity = (forward * stick[1] + right * stick[0]) * speed;
            let facing = self.strafe.then(|| forward.x.atan2(forward.z));
            c.matcher.settings.filter.tags = if run { c.run_tags } else { c.walk_tags };
            c.matcher.update(&c.db, &MotionInput { velocity, facing }, dt);
            let search = c.matcher.last_search();
            self.searches += search.searched as u32;
            self.switches += search.switched as u32;

            let character = c.matcher.character();
            if let Some(r) = self.scene.get_renderable_mut(c.index) {
                r.object.set_position(character.translation.x, character.translation.y, character.translation.z);
                r.object.rotation.y = yaw_of(character.rotation);
                c.palette.update(&c.mesh, c.matcher.model());
                if let Some(buffer) = r.material.bindable_buffer(PALETTE_BINDING) {
                    c.palette.upload(self.renderer.queue(), &buffer);
                }
            }
            // follow the character's hips
            let hips = character.transform_point(c.matcher.model()[c.db.roles.hips].translation);
            let target = GVec3::new(character.translation.x, hips.y * 0.9, character.translation.z);
            let t = 1.0 - (-dt * 8.0).exp();
            let current = GVec3::new(self.controls.target.x, self.controls.target.y, self.controls.target.z);
            let followed = current.lerp(target, t);
            self.controls.target = Vec3::new(followed.x, followed.y, followed.z);

            if self.overlay {
                // the simulation and its predicted samples: flat boxes pointing where they face
                let s = c.matcher.simulation();
                let mut k = 0;
                for (p, q, size) in std::iter::once((s.position, s.rotation, 0.16)).chain(c.matcher.trajectory().iter().map(|t| (t.translation, t.rotation, 0.1))) {
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::new(size, 0.02, size * 2.0), q, p + GVec3::Y * 0.01);
                    k += 1;
                }
                // each foot's target, raised when not planted
                let model = c.matcher.model();
                for (side, locked) in c.matcher.feet_locked().iter().enumerate() {
                    let p = character.transform_point(model[c.db.roles.feet[side]].translation);
                    let size = if *locked { 0.09 } else { 0.04 };
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::splat(size), Quat::IDENTITY, GVec3::new(p.x, 0.02, p.z));
                    k += 1;
                }
            } else {
                c.trajectory.matrices.fill(Mat4::ZERO);
            }
            c.trajectory.upload(&self.renderer);
            if self.scene.get_renderable(c.bones.index).is_some_and(|r| r.visible) {
                let model = c.matcher.model();
                for (j, parent) in c.db.skeleton.parents.iter().enumerate() {
                    c.bones.matrices[j] = match parent {
                        Some(p) if *p != c.db.roles.root => segment(character.transform_point(model[*p].translation), character.transform_point(model[j].translation), 0.015),
                        _ => Mat4::ZERO,
                    };
                }
                c.bones.upload(&self.renderer);
            }
        }
        self.controls.update(&mut self.camera, dt);

        // HUD, a few times a second
        if now - self.counted_since > 1.0 {
            let span = (now - self.counted_since) as f32;
            self.rates = (self.searches as f32 / span, self.switches as f32 / span);
            self.searches = 0;
            self.switches = 0;
            self.counted_since = now;
        }
        if self.frame % 10 == 0 {
            match &self.character {
                Some(c) if self.overlay => {
                    let (clip, frame) = c.matcher.playing();
                    let info = &c.db.clips[clip];
                    let s = c.matcher.last_search();
                    let feet = c.matcher.feet_locked();
                    let speed = c.matcher.simulation().velocity.length();
                    set_hud(&format!(
                        "{:.0} fps   {} {}{}\nclip   {}\nframe  {:.0} / {}{}\nsearch {:.0}/s, switch {:.1}/s, cost {:.3}\nfeet   {} {}  (lock {})\n\n{}\n\nWASD / left stick move · Shift / A run · Q / LB strafe\ndrag / right stick orbit · B overlay · K skeleton · M mesh · L foot lock",
                        self.fps,
                        if run { "run" } else { "walk" },
                        format_args!("{speed:.1} m/s"),
                        if self.strafe { "  strafe" } else { "" },
                        info.name,
                        frame,
                        info.playable(),
                        if info.looping { " (loop)" } else { "" },
                        self.rates.0,
                        self.rates.1,
                        s.cost,
                        if feet[0] { "L planted" } else { "L free" },
                        if feet[1] { "R planted" } else { "R free" },
                        if c.matcher.settings.foot_lock { "on" } else { "off" },
                        c.source,
                    ));
                }
                Some(_) => set_hud(""),
                None => {}
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let canvas = document.get_element_by_id(canvas_id).ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;
    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig { width, height, sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() });
    renderer.initialize_with_canvas(canvas.clone()).await;
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: 60.0, ..Default::default() });

    let mut scene = Scene::new();
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));
    let mut ground_material = Material::new("Ground", &format!("{CASCADED_SHADOWS_WGSL}\n{GROUND_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    ground_material.set_uniform_bindable(0, "Ground", &surface_params([0.32, 0.32, 0.3]));
    let mut ground = Renderable::new(PlaneGeometry::new(400.0, 400.0), ground_material);
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    ground.cast_shadow = false;
    scene.add(SceneNode::Renderable(ground));
    let mut sun = DirectionalLight::new(Vec3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]), Vec3::new(1.0, 0.9, 0.75), 80000.0);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    // the character, from a pack outside the repository
    let url = query_param("pack").unwrap_or_else(|| "pack/locomotion.kmm".to_string());
    set_hud(&format!("Loading motion pack {url} …"));
    let gait = query_param("gait").as_deref() != Some("0");
    let character = match fetch_bytes(&url).await.and_then(|bytes| MotionPack::from_bytes(&bytes)) {
        Ok(pack) => match Character::new(&renderer, &mut scene, pack, gait) {
            Ok(c) => Some(c),
            Err(e) => {
                set_hud(&format!("The motion pack {url} can't be used: {e}."));
                None
            }
        },
        Err(e) => {
            log::warn!("no motion pack: {e}");
            set_hud(&format!(
                "No motion pack ({e}).\n\nThis example animates a character from a motion-matching pack (.kmm), and Kansei ships none:\n\
                 bake one from your own glTF clips with kansei-anim-bake, then put it at\n\
                 rust/kansei-wasm/examples/motion-matching/www/pack/locomotion.kmm\n\
                 (or open this page with ?pack=<url>). See this example's README."
            ));
            None
        }
    };

    let tonemap = {
        let mut o = ToneMapOptions::for_surface(renderer.presentation_format());
        o.exposure = exposure_from_ev100(14.0);
        o.vignette = 0.3;
        ToneMapEffect::new(o)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if query_param("taa").as_deref() != Some("0") {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);
    let mut camera = Camera::new(45.0, 0.1, 1200.0, width as f32 / height as f32);
    camera.update_projection_matrix();
    let mut controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 0.9, 0.0), 4.5);
    controls.set_elevation(0.25);
    controls.set_azimuth(std::f32::consts::PI);

    let keys = Rc::new(RefCell::new(Keys::default()));
    {
        let down = keys.clone();
        let on_down = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            let key = e.key().to_lowercase();
            let mut keys = down.borrow_mut();
            if !e.repeat() {
                keys.pressed.push(key.clone());
            }
            keys.held.insert(key);
            if e.key().starts_with("Arrow") {
                e.prevent_default();
            }
        });
        window.add_event_listener_with_callback("keydown", on_down.as_ref().unchecked_ref())?;
        on_down.forget();
        let up = keys.clone();
        let on_up = Closure::<dyn FnMut(web_sys::KeyboardEvent)>::new(move |e: web_sys::KeyboardEvent| {
            up.borrow_mut().held.remove(&e.key().to_lowercase());
        });
        window.add_event_listener_with_callback("keyup", on_up.as_ref().unchecked_ref())?;
        on_up.forget();
        // a key released while the page is in the background never sends keyup
        let blur = keys.clone();
        let on_blur = Closure::<dyn FnMut()>::new(move || blur.borrow_mut().held.clear());
        window.add_event_listener_with_callback("blur", on_blur.as_ref().unchecked_ref())?;
        on_blur.forget();
    }

    let scaled = |paces: [f32; 3], name: &str| query_param(name).and_then(|v| v.parse::<f32>().ok()).map_or(paces, |forward| paces.map(|p| p * forward / paces[0]));
    let speeds = (scaled(WALK, "walk"), scaled(RUN, "run"));
    log::info!("Kansei — Motion Matching (WASM) ready: character {}", character.is_some());
    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        controls,
        volume,
        character,
        keys,
        last: now_secs(),
        frame: 0,
        strafe: false,
        overlay: true,
        pad_held: Vec::new(),
        speeds,
        fps: 60.0,
        searches: 0,
        switches: 0,
        counted_since: now_secs(),
        rates: (0.0, 0.0),
    }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        state.borrow_mut().frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
