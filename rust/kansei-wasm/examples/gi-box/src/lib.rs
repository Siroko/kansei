//! GI box: a Cornell box (white floor, ceiling and back wall, a red wall on the left, a green one
//! on the right, two white blocks, a rug half orange and half blue, optionally the Stanford
//! dragon) under one shadowed downlight, to compare global illumination methods. Without GI the
//! scene has direct light only, so the ceiling and the shadows are black; with it the floor's
//! light reaches the ceiling and the walls bleed their colour onto the floor and the blocks.
//!
//! - Screen-space GI (`gi=low|medium|high|ultra`) sees only what is on screen.
//! - Voxel GI (`gi=voxel`) traces cones through a voxel volume of the room: the renderables are
//!   voxelized through their own vertex shaders and lit through the lamp's shadow map, with the
//!   bounces adding up over frames, so light from off screen (the open front's side of the walls,
//!   the backs of the blocks) arrives too. The rug's voxels take its texture through its
//!   material's voxel entry (`albedo=constant`: its mean colour instead).
//! - `gi=voxel+ssgi`: screen-space GI in front for contact detail, the voxels for the rest.
//!
//! Drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan. The panel
//! (and `window.kansei`) switches everything at run time.
//!
//! URL parameters (a `preset` first, the others over it):
//! - `preset=off|ssgi|voxel|best|indirect|voxels|phone|dragon` (see `PRESETS`);
//! - `gi=off|low|medium|high|ultra|voxel|voxel+ssgi` (default high);
//! - `voxels=low|medium|high`: the volume's resolution (default medium; low on phones, which also
//!   keep it within 24 MiB);
//! - `view=indirect` (only the light GI adds, 2 stops brighter) or `view=voxels` (with voxel GI:
//!   the lit voxels themselves);
//! - `cam=front|corner|low`;
//! - `dragon=1|full`: the Stanford dragon (CC-BY-NC-4.0, see `www/assets/license.txt`): `1` the
//!   decimated `.glb` (19k triangles), `full` the whole scan (871k triangles, 24 MB);
//! - `animate=1`: the dragon (or, without it, the tall block) turns and slides, re-voxelized each
//!   frame;
//! - `albedo=constant`, `rug=off` (the box without its rug), `ui=0` (no panel);
//! - `stats=1`: triangles, frame interval and the GPU time of each pass (the renderer's profiling).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, VOXEL_WRITE_WGSL};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, GiQuality, ScreenSpaceGIEffect, ScreenSpaceGIOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};

/// A diffuse surface lit by the spot lights only (no ambient), writing the normal and albedo the
/// global illumination reads (GBuffer targets 2 and 3).
const LIT_WGSL: &str = r#"
struct Surface { base_color: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
};
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = surface.base_color.rgb;
    var out: GBufferOut;
    out.color = vec4<f32>(kansei_spot_lights_radiance(in.world, n, v, base, 1.0, 0.0, in.clip.xy), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}
"#;

/// The rug: `LIT_WGSL` with its base colour from a texture, and a voxel entry that gives voxel GI
/// the same texture.
const RUG_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> surface: vec4<f32>;
@group(0) @binding(1) var rug_texture: texture_2d<f32>;
@group(0) @binding(2) var rug_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv = v.uv;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = textureSample(rug_texture, rug_sampler, in.uv).rgb;
    var out: GBufferOut;
    out.color = vec4<f32>(kansei_spot_lights_radiance(in.world, n, v, base, 1.0, 0.0, in.clip.xy), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}

// voxel GI's voxelizer: the rug's texture into its voxels
@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, textureSample(rug_texture, rug_sampler, in.uv).rgb, vec3<f32>(0.0));
}
"#;

/// The rug's two halves (linear albedo), split across its width.
const RUG_LEFT: [f32; 3] = [0.85, 0.35, 0.05];
const RUG_RIGHT: [f32; 3] = [0.05, 0.25, 0.8];

/// The rug's texture: orange on the left half, blue on the right, with a thin pale border.
fn rug_texture() -> Texture {
    const N: u32 = 64;
    let srgb = |c: f32| (if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 } * 255.0).round() as u8;
    let mut texels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let border = x < 3 || y < 3 || x >= N - 3 || y >= N - 3;
            let c = if border { [0.8, 0.75, 0.6] } else if x < N / 2 { RUG_LEFT } else { RUG_RIGHT };
            texels.extend([srgb(c[0]), srgb(c[1]), srgb(c[2]), 255]);
        }
    }
    Texture::from_rgba("Rug", N, N, &texels)
}

/// The rug's material; `voxel_entry` false voxelizes it with its constant `GiSurface` instead.
fn rug_material(voxel_entry: bool) -> Material {
    let mut material = Material::new(
        "Rug",
        &format!("{SPOT_LIGHTS_WGSL}
{VOXEL_WRITE_WGSL}
{RUG_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: voxel_entry.then_some("voxel_main"), ..Default::default() },
    );
    material.set_uniform_bindable(0, "Rug", &[1.0f32, 1.0, 1.0, 1.0]);
    material.set_bindable(1, rug_texture());
    material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear));
    material
}

fn lit_material(label: &str, base_color: [f32; 3]) -> Material {
    let mut material = Material::new(
        label,
        &format!("{SPOT_LIGHTS_WGSL}\n{LIT_WGSL}"),
        vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)],
        MaterialOptions { mrt_output_count: Some(4), ..Default::default() },
    );
    material.set_uniform_bindable(0, label, &[base_color[0], base_color[1], base_color[2], 1.0f32]);
    material
}

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// The global illumination asked for (`gi=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gi {
    Off,
    Screen(GiQuality),
    Voxel,
    VoxelAndScreen,
}

impl Gi {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "off" => Gi::Off,
            "low" => Gi::Screen(GiQuality::Low),
            "medium" => Gi::Screen(GiQuality::Medium),
            "high" | "ssgi" => Gi::Screen(GiQuality::High),
            "ultra" => Gi::Screen(GiQuality::Ultra),
            "voxel" => Gi::Voxel,
            // a raw '+' in the query, or one decoded to a space or escaped
            "voxel+ssgi" | "voxel ssgi" | "voxel%2Bssgi" | "voxel%2bssgi" => Gi::VoxelAndScreen,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Gi::Off => "off",
            Gi::Screen(GiQuality::Low) => "low",
            Gi::Screen(GiQuality::Medium) => "medium",
            Gi::Screen(GiQuality::High) => "high",
            Gi::Screen(GiQuality::Ultra) => "ultra",
            Gi::Voxel => "voxel",
            Gi::VoxelAndScreen => "voxel+ssgi",
        }
    }

    fn voxels(self) -> bool {
        matches!(self, Gi::Voxel | Gi::VoxelAndScreen)
    }
}

/// Which Stanford dragon (`dragon=`): both are the same Sketchfab scan (CC-BY-NC-4.0,
/// `www/assets/license.txt`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dragon {
    Off,
    /// The decimated `.glb` (19k triangles).
    Light,
    /// The full scan, `scene.gltf` + `scene.bin` (871k triangles, a 24 MB download).
    Full,
}

impl Dragon {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "0" | "off" | "false" => Dragon::Off,
            "1" | "on" | "true" | "light" => Dragon::Light,
            "full" => Dragon::Full,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Dragon::Off => "off",
            Dragon::Light => "light",
            Dragon::Full => "full",
        }
    }

    /// Its slot in `State::dragons`.
    fn slot(self) -> Option<usize> {
        match self {
            Dragon::Off => None,
            Dragon::Light => Some(0),
            Dragon::Full => Some(1),
        }
    }
}

/// What the image shows (`view=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum View {
    Lit,
    /// Only the light GI adds, two stops brighter.
    Indirect,
    /// Voxel GI's lit voxels.
    Voxels,
}

impl View {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "lit" => View::Lit,
            "indirect" => View::Indirect,
            "voxels" => View::Voxels,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            View::Lit => "lit",
            View::Indirect => "indirect",
            View::Voxels => "voxels",
        }
    }
}

fn tier_name(q: VoxelGiQuality) -> &'static str {
    match q {
        VoxelGiQuality::Low => "low",
        VoxelGiQuality::Medium => "medium",
        VoxelGiQuality::High => "high",
    }
}

/// Everything the panel and the URL set.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    gi: Gi,
    /// The voxel tier asked for (the device may hold less).
    voxels: VoxelGiQuality,
    view: View,
    dragon: Dragon,
    animate: bool,
    rug: bool,
    /// The rug's voxels from its texture (its material's voxel entry) or its mean colour.
    textured: bool,
}

/// The presets: (name, label, gi, voxels (None: the device's default), view, dragon).
const PRESETS: [(&str, &str, &str, Option<VoxelGiQuality>, View, Dragon); 8] = [
    ("off", "Off (direct light)", "off", None, View::Lit, Dragon::Off),
    ("ssgi", "SSGI", "high", None, View::Lit, Dragon::Off),
    ("voxel", "Voxel", "voxel", None, View::Lit, Dragon::Off),
    ("best", "Voxel + SSGI (best)", "voxel+ssgi", None, View::Lit, Dragon::Off),
    ("indirect", "Indirect only", "voxel+ssgi", None, View::Indirect, Dragon::Off),
    ("voxels", "Voxels (debug)", "voxel", None, View::Voxels, Dragon::Off),
    ("phone", "Phone (low)", "voxel+ssgi", Some(VoxelGiQuality::Low), View::Lit, Dragon::Off),
    ("dragon", "Dragon, 871k tris (voxel + SSGI)", "voxel+ssgi", None, View::Lit, Dragon::Full),
];

/// The camera presets: (name, target, distance, azimuth, elevation).
const CAMERAS: [(&str, [f32; 3], f32, f32, f32); 3] = [
    // the box's opening, framed as the Cornell box photographs are
    ("front", [0.0, 2.0, -2.0], 8.3, 0.0, 0.0),
    // inside, from the upper front-left corner toward the short block and the green wall
    ("corner", [1.0, 0.8, -3.0], 4.98, -0.777, 0.634),
    // half a metre above the floor in front of the opening, looking up into the box
    ("low", [0.0, 1.4, -2.5], 7.5, 0.0, -0.12),
];

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    config: Config,
    phone: bool,
    /// The voxel tier the renderer's voxel GI was enabled at.
    enabled_voxels: Option<VoxelGiQuality>,
    tall: usize,
    rug: usize,
    /// The dragons loaded so far (`Dragon::slot`).
    dragons: [Option<usize>; 2],
    /// (position, yaw) the animated renderables rest at.
    tall_rest: (Vec3, f32),
    dragon_rest: (Vec3, f32),
    time: f32,
    last_ms: f64,
    stats: Option<Stats>,
}

/// The `stats=1` overlay's numbers, refreshed every second.
#[derive(Default)]
struct Stats {
    frames: u32,
    since: f64,
    frame_ms: f64,
    /// (label, ms per frame), most expensive first
    passes: Vec<(String, f64)>,
    gpu_ms: f64,
    gpu_span_ms: f64,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn now_ms() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

fn is_phone() -> bool {
    let agent = web_sys::window().and_then(|w| w.navigator().user_agent().ok()).unwrap_or_default();
    ["Mobi", "Android", "iPhone", "iPad"].iter().any(|k| agent.contains(k))
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

fn default_voxels(phone: bool) -> VoxelGiQuality {
    if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium }
}

/// `config` with preset `name` applied (unknown names change nothing).
fn with_preset(config: Config, name: &str, phone: bool) -> Config {
    let Some(&(_, _, gi, voxels, view, dragon)) = PRESETS.iter().find(|p| p.0 == name) else { return config };
    Config { gi: Gi::from_name(gi).unwrap(), voxels: voxels.unwrap_or(default_voxels(phone)), view, dragon, ..config }
}

/// The configuration the URL asks for: its preset, then its other parameters over it.
fn config_from_url(phone: bool) -> Config {
    let mut c = Config {
        gi: Gi::Screen(GiQuality::High),
        voxels: default_voxels(phone),
        view: View::Lit,
        dragon: Dragon::Off,
        animate: false,
        rug: true,
        textured: true,
    };
    if let Some(preset) = query_param("preset") {
        c = with_preset(c, &preset, phone);
    }
    if let Some(gi) = query_param("gi").as_deref().and_then(Gi::from_name) {
        c.gi = gi;
    }
    if let Some(q) = query_param("voxels").as_deref().and_then(VoxelGiQuality::from_name) {
        c.voxels = q;
    }
    if let Some(view) = query_param("view").as_deref().and_then(View::from_name) {
        c.view = view;
    }
    if let Some(dragon) = query_param("dragon").as_deref().and_then(Dragon::from_name) {
        c.dragon = dragon;
    }
    c.animate = query_param("animate").map_or(c.animate, |v| v == "1" || v == "on" || v == "true");
    c.rug = query_param("rug").as_deref() != Some("off");
    c.textured = query_param("albedo").as_deref() != Some("constant");
    c
}

/// The post-processing chain for `config`: its GI effect (none when off) and the tone mapping.
fn build_effects(renderer: &Renderer, config: &Config, phone: bool) -> Vec<Box<dyn PostProcessingEffect>> {
    let indirect = config.view == View::Indirect;
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    let screen = ScreenSpaceGIOptions { radius_m: 4.0, ..Default::default() };
    match config.gi {
        Gi::Off => {}
        Gi::Screen(quality) => {
            let mut effect = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, ..screen });
            effect.show_indirect = indirect;
            effects.push(Box::new(effect));
        }
        Gi::Voxel | Gi::VoxelAndScreen => {
            let scene_gi = renderer.voxel_gi().expect("voxel GI is enabled for the voxel modes");
            let near_quality = if phone || scene_gi.quality() == VoxelGiQuality::Low { GiQuality::Low } else { GiQuality::High };
            let near_field = (config.gi == Gi::VoxelAndScreen).then_some(ScreenSpaceGIOptions { quality: near_quality, ..screen });
            let mut effect = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), near_field, ..Default::default() });
            effect.show_indirect = indirect;
            effect.show_voxels = config.view == View::Voxels;
            effects.push(Box::new(effect));
        }
    }
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    // the indirect view alone is 2 stops brighter
    tone.exposure = exposure_from_ev100(5.0) * if indirect { 4.0 } else { 1.0 };
    effects.push(Box::new(ToneMapEffect::new(tone)));
    effects
}

/// Fetch `url`'s bytes.
async fn fetch_bytes(url: &str) -> Option<Vec<u8>> {
    let window = web_sys::window()?;
    let resp = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await.ok()?;
    let resp: web_sys::Response = resp.dyn_into().ok()?;
    if !resp.ok() {
        return None;
    }
    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer().ok()?).await.ok()?;
    Some(js_sys::Uint8Array::new(&buf).to_vec())
}

/// A Stanford dragon from `www/assets` as one geometry about 1.1 m long, standing on y = 0 and
/// centred on its origin, its parts merged with their glTF transforms baked in (None if it can't
/// be loaded).
async fn load_dragon(model: Dragon) -> Option<kansei_core::geometries::Geometry> {
    let result = match model {
        Dragon::Off => return None,
        Dragon::Light => GLTFLoader::load_glb(&fetch_bytes("assets/stanford_dragon_pbr.glb").await?).ok()?,
        Dragon::Full => {
            let json = fetch_bytes("assets/scene.gltf").await?;
            let bin = fetch_bytes("assets/scene.bin").await?;
            GLTFLoader::load_gltf_with_buffers(&json, vec![bin]).ok()?
        }
    };
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    let (mut lo, mut hi) = (glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN));
    for part in result.renderables {
        let (r, s) = (part.rotation, part.scale);
        let rotation = glam::Mat4::from_rotation_z(r.z) * glam::Mat4::from_rotation_y(r.y) * glam::Mat4::from_rotation_x(r.x);
        let node = glam::Mat4::from_translation(glam::Vec3::new(part.position.x, part.position.y, part.position.z)) * rotation * glam::Mat4::from_scale(glam::Vec3::new(s.x, s.y, s.z));
        let base = vertices.len() as u32;
        for mut v in part.geometry.vertices {
            let p = node.transform_point3(glam::Vec3::new(v.position[0], v.position[1], v.position[2]));
            v.position = [p.x, p.y, p.z, 1.0];
            v.normal = rotation.transform_vector3(glam::Vec3::from(v.normal)).normalize_or_zero().to_array();
            lo = lo.min(p);
            hi = hi.max(p);
            vertices.push(v);
        }
        indices.extend(part.geometry.indices.iter().map(|i| i + base));
    }
    let extent = hi - lo;
    let k = 1.1 / extent.x.max(extent.z);
    let centre = glam::Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
    for v in &mut vertices {
        let p = (glam::Vec3::new(v.position[0], v.position[1], v.position[2]) - centre) * k;
        v.position = [p.x, p.y, p.z, 1.0];
    }
    let geometry = kansei_core::geometries::Geometry::new("Dragon", vertices, indices);
    log::info!("dragon ({}): {} triangles, {:.2} x {:.2} x {:.2} m", model.name(), geometry.index_count() / 3, extent.x * k, extent.y * k, extent.z * k);
    Some(geometry)
}

impl State {
    /// Bring the renderer, the chain and the scene to `config`.
    fn apply(&mut self, config: Config) {
        let previous = self.config;
        self.config = config;
        // voxel GI: enabled at the asked tier for the voxel modes, freed for the others
        if config.gi.voxels() {
            if self.enabled_voxels != Some(config.voxels) {
                self.renderer.enable_voxel_gi(SceneVoxelGiOptions {
                    quality: config.voxels,
                    bounds_min: [-2.3, -0.3, -4.3],
                    bounds_max: [2.3, 4.3, 0.3],
                    radiance_scale: 1.0,
                    budget_bytes: if self.phone { 24 << 20 } else { 0 },
                });
                self.enabled_voxels = Some(config.voxels);
            }
        } else if self.enabled_voxels.is_some() {
            self.renderer.disable_voxel_gi();
            self.enabled_voxels = None;
        }
        self.volume.effects = build_effects(&self.renderer, &config, self.phone);

        if let Some(r) = self.scene.get_renderable_mut(self.rug) {
            r.visible = config.rug;
        }
        if config.textured != previous.textured {
            if let Some(r) = self.scene.get_renderable_mut(self.rug) {
                r.material = rug_material(config.textured);
                r.material_dirty = true;
            }
            // the voxelizer can't see a material change
            if let Some(gi) = self.renderer.voxel_gi_mut() {
                gi.invalidate();
            }
        }
        for (slot, index) in self.dragons.iter().enumerate() {
            if let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) {
                r.visible = config.dragon.slot() == Some(slot);
            }
        }
        // the animated renderable is drawn (and voxelized) live; the others rest
        let animated = self.animated();
        let rests = [(Some(self.tall), self.tall_rest), (self.dragons[0], self.dragon_rest), (self.dragons[1], self.dragon_rest)];
        for (index, rest) in rests {
            let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) else { continue };
            r.dynamic = config.animate && index == Some(animated);
            if !r.dynamic {
                r.object.position = rest.0;
                r.object.rotation.y = rest.1;
            }
        }
    }

    /// The renderable `animate` moves: the dragon when it is shown, the tall block otherwise.
    fn animated(&self) -> usize {
        self.config.dragon.slot().and_then(|slot| self.dragons[slot]).unwrap_or(self.tall)
    }

    fn frame(&mut self) {
        let now = now_ms();
        let dt = ((now - self.last_ms) / 1000.0).clamp(0.0, 0.1) as f32;
        self.last_ms = now;
        if self.config.animate {
            self.time += dt;
            let animated = self.animated();
            let rest = if animated == self.tall { self.tall_rest } else { self.dragon_rest };
            let t = self.time;
            if let Some(r) = self.scene.get_renderable_mut(animated) {
                // turn, and slide back and forth across the box
                r.object.position = Vec3::new(rest.0.x + 0.45 * (t * 0.7).sin(), rest.0.y, rest.0.z + 0.25 * (t * 0.45).sin());
                r.object.rotation.y = rest.1 + t * 0.6;
            }
        }
        self.controls.update(&mut self.camera, dt);
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if let Some(stats) = &mut self.stats {
            stats.frames += 1;
            if now - stats.since >= 1000.0 {
                stats.frame_ms = (now - stats.since) / stats.frames as f64;
                stats.frames = 0;
                stats.since = now;
                let profile = self.renderer.take_profile();
                if profile.gpu_frames > 0 {
                    let mut passes: Vec<(String, f64)> = profile.gpu.iter().map(|p| (p.label.to_string(), p.exclusive_ms)).collect();
                    passes.sort_by(|a, b| b.1.total_cmp(&a.1));
                    stats.passes = passes;
                    stats.gpu_ms = profile.gpu_ms;
                    stats.gpu_span_ms = profile.gpu_span_ms;
                }
            }
        }
    }

    /// Triangles drawn per view: every visible renderable's.
    fn triangles(&self) -> u64 {
        self.scene.ordered_indices().filter_map(|i| self.scene.get_renderable(i)).filter(|r| r.visible).map(|r| r.geometry.index_count() as u64 / 3 * r.geometry.instance_count.max(1) as u64).sum()
    }

    fn info(&self) -> String {
        let gi = self.renderer.voxel_gi();
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        // (+ 0.0: an empty sum is -0)
        let sum = |prefix: &str| self.stats.as_ref().map_or(0.0, |s| s.passes.iter().filter(|p| p.0.starts_with(prefix)).map(|p| p.1).sum::<f64>()) + 0.0;
        format!(
            "{{\"gi\":\"{}\",\"view\":\"{}\",\"voxels\":\"{}\",\"voxel_tier\":{},\"dims\":{},\"mib\":{:.1},\"dragon\":\"{}\",\"animate\":{},\"rug\":{},\"textured\":{},\"triangles\":{},\"stats\":{},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"voxelize_ms\":{:.3},\"inject_ms\":{:.3},\"mips_ms\":{:.3},\"screen_ms\":{:.3},\"ssgi_ms\":{:.3},\"passes\":[{}]}}",
            self.config.gi.name(),
            self.config.view.name(),
            tier_name(self.config.voxels),
            gi.map_or("null".into(), |g| format!("\"{}\"", tier_name(g.quality()))),
            gi.map_or("null".into(), |g| format!("{:?}", g.volume().dims())),
            gi.map_or(0.0, |g| g.memory_bytes() as f64 / (1 << 20) as f64),
            self.config.dragon.name(),
            self.config.animate,
            self.config.rug,
            self.config.textured,
            self.triangles(),
            self.stats.is_some(),
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            sum("VoxelGI/Voxelize"),
            sum("VoxelGI/Inject"),
            sum("VoxelGI/Mips") + sum("VoxelGI/AnisotropicMips"),
            sum("VoxelGI/Screen"),
            sum("SSGI"),
            passes.join(","),
        )
    }
}

/// Add the dragon to the scene (hidden until the config shows it), returning its index.
fn add_dragon(scene: &mut Scene, geometry: kansei_core::geometries::Geometry, rest: (Vec3, f32)) -> usize {
    let albedo = [0.75, 0.62, 0.42];
    let mut dragon = Renderable::new(geometry, lit_material("Dragon", albedo)).with_gi(GiSurface::new(albedo));
    dragon.object.position = rest.0;
    dragon.object.rotation.y = rest.1;
    dragon.visible = false;
    scene.add(SceneNode::Renderable(dragon))
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let width = canvas.client_width() as u32;
    let height = canvas.client_height() as u32;
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;
    renderer.enable_spot_shadows(2048, 1);
    let phone = is_phone();
    let config = config_from_url(phone);

    // the room: 4 m wide, high and deep, open toward the camera (z = 0), walls 0.2 m thick
    let mut scene = Scene::new();
    let white = [0.73, 0.73, 0.73];
    let slabs: [(&str, [f32; 3], [f32; 3], [f32; 3]); 5] = [
        ("Floor", [4.4, 0.2, 4.2], [0.0, -0.1, -2.1], white),
        ("Ceiling", [4.4, 0.2, 4.2], [0.0, 4.1, -2.1], white),
        ("Back", [4.4, 4.4, 0.2], [0.0, 2.0, -4.1], white),
        ("Left", [0.2, 4.4, 4.2], [-2.1, 2.0, -2.1], [0.63, 0.065, 0.05]),
        ("Right", [0.2, 4.4, 4.2], [2.1, 2.0, -2.1], [0.14, 0.45, 0.09]),
    ];
    for (label, size, position, color) in slabs {
        let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), lit_material(label, color)).with_gi(GiSurface::new(color));
        slab.object.set_position(position[0], position[1], position[2]);
        scene.add(SceneNode::Renderable(slab));
    }
    // a tall block near the red wall and a short one near the green wall
    let tall_rest = (Vec3::new(-0.75, 1.2, -2.6), 0.33);
    let mut tall = Renderable::new(BoxGeometry::new(1.2, 2.4, 1.2), lit_material("Tall", white)).with_gi(GiSurface::new(white));
    tall.object.position = tall_rest.0;
    tall.object.rotation.y = tall_rest.1;
    let tall = scene.add(SceneNode::Renderable(tall));
    let mut short = Renderable::new(BoxGeometry::new(1.2, 1.2, 1.2), lit_material("Short", white)).with_gi(GiSurface::new(white));
    short.object.set_position(0.8, 0.6, -1.5);
    short.object.rotation.y = -0.3;
    scene.add(SceneNode::Renderable(short));
    // a textured rug on the floor in front of the blocks, orange before the tall one and blue
    // before the short one, whose sides facing the camera only bounces light; its constant
    // surface (its mean colour) is used where the material's voxel entry is not
    let mean: [f32; 3] = std::array::from_fn(|c| (RUG_LEFT[c] + RUG_RIGHT[c]) * 0.5);
    let mut rug = Renderable::new(BoxGeometry::new(2.4, 0.02, 0.8), rug_material(config.textured)).with_gi(GiSurface::new(mean));
    rug.object.set_position(0.0, 0.01, -0.55);
    let rug = scene.add(SceneNode::Renderable(rug));
    // the dragon, in the free corner at the front left, when asked for
    let dragon_rest = (Vec3::new(-1.15, 0.0, -1.25), 0.5);
    let mut dragons = [None, None];
    if let Some(slot) = config.dragon.slot() {
        dragons[slot] = load_dragon(config.dragon).await.map(|g| add_dragon(&mut scene, g, dragon_rest));
    }

    // the key light: a shadowed downlight just under the ceiling's centre, wide enough to light
    // the tops of the side walls
    let mut lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    lamp.cast_shadow = true;
    lamp.source_radius = 0.15;
    lamp.volumetric_scale = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    let camera = Camera::new(38.0, 0.1, 100.0, width as f32 / height as f32);
    let (_, target, distance, azimuth, elevation) = CAMERAS.iter().find(|c| Some(c.0) == query_param("cam").as_deref()).copied().unwrap_or(CAMERAS[0]);
    let mut controls = CameraControls::from_canvas(&canvas, Vec3::new(target[0], target[1], target[2]), distance).with_mouse_pan(&canvas);
    controls.set_view(Vec3::new(target[0], target[1], target[2]), distance, azimuth, elevation);

    let stats = (query_param("stats").as_deref() == Some("1")).then(|| Stats { since: now_ms(), ..Default::default() });
    if stats.is_some() {
        renderer.set_profiling(true);
    }
    let volume = PostProcessingVolume::new(&renderer, Vec::new());
    let mut state = State {
        renderer,
        scene,
        camera,
        controls,
        volume,
        config,
        phone,
        enabled_voxels: None,
        tall,
        rug,
        dragons,
        tall_rest,
        dragon_rest,
        time: 0.0,
        last_ms: now_ms(),
        stats,
    };
    state.apply(config);
    log::info!("Kansei — GI box (WASM) ready: {}", state.info());

    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        state.borrow_mut().frame();
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}

/// The state as JSON: the configuration, the volume, the triangles and (with stats) the times.
#[wasm_bindgen]
pub fn info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// The presets, as JSON `[[name, label], ...]`.
#[wasm_bindgen]
pub fn presets() -> String {
    let items: Vec<String> = PRESETS.iter().map(|p| format!("[\"{}\",\"{}\"]", p.0, p.1)).collect();
    format!("[{}]", items.join(","))
}

/// Apply preset `name` (`PRESETS`): GI mode, voxel tier, view and the dragon together.
#[wasm_bindgen]
pub async fn set_preset(name: String) {
    let Some(config) = with_state(|s| with_preset(s.config, &name, s.phone)) else { return };
    apply_with_dragon(config).await;
}

/// `gi=` at run time.
#[wasm_bindgen]
pub fn set_gi(name: &str) {
    if let Some(gi) = Gi::from_name(name) {
        with_state(|s| s.apply(Config { gi, ..s.config }));
    }
}

/// `voxels=` at run time.
#[wasm_bindgen]
pub fn set_voxels(name: &str) {
    if let Some(voxels) = VoxelGiQuality::from_name(name) {
        with_state(|s| s.apply(Config { voxels, ..s.config }));
    }
}

/// `view=lit|indirect|voxels` at run time.
#[wasm_bindgen]
pub fn set_view(name: &str) {
    if let Some(view) = View::from_name(name) {
        with_state(|s| s.apply(Config { view, ..s.config }));
    }
}

/// `dragon=off|light|full`: hide the dragon, or show the light (19k triangles) or full (871k)
/// one, loading it the first time.
#[wasm_bindgen]
pub async fn set_dragon(name: String) {
    let Some(dragon) = Dragon::from_name(&name) else { return };
    let Some(config) = with_state(|s| Config { dragon, ..s.config }) else { return };
    apply_with_dragon(config).await;
}

/// Apply `config`, loading its dragon first if it isn't loaded.
async fn apply_with_dragon(config: Config) {
    if let Some(slot) = config.dragon.slot() {
        if with_state(|s| s.dragons[slot].is_none()) == Some(true) {
            if let Some(geometry) = load_dragon(config.dragon).await {
                with_state(|s| {
                    if s.dragons[slot].is_none() {
                        s.dragons[slot] = Some(add_dragon(&mut s.scene, geometry, s.dragon_rest));
                    }
                });
            }
        }
    }
    with_state(|s| s.apply(config));
}

/// Move the dragon (or the tall block) around the box, re-voxelized every frame.
#[wasm_bindgen]
pub fn set_animate(on: bool) {
    with_state(|s| s.apply(Config { animate: on, ..s.config }));
}

#[wasm_bindgen]
pub fn set_rug(on: bool) {
    with_state(|s| s.apply(Config { rug: on, ..s.config }));
}

/// The rug's voxels from its texture (true) or its mean colour.
#[wasm_bindgen]
pub fn set_textured(on: bool) {
    with_state(|s| s.apply(Config { textured: on, ..s.config }));
}

/// Camera preset `front|corner|low` (`CAMERAS`).
#[wasm_bindgen]
pub fn set_camera(name: &str) {
    if let Some(&(_, t, distance, azimuth, elevation)) = CAMERAS.iter().find(|c| c.0 == name) {
        with_state(|s| s.controls.set_view(Vec3::new(t[0], t[1], t[2]), distance, azimuth, elevation));
    }
}

/// Profile every pass (the stats in `info`).
#[wasm_bindgen]
pub fn set_stats(on: bool) {
    with_state(|s| {
        s.renderer.set_profiling(on);
        s.stats = on.then(|| Stats { since: now_ms(), ..Default::default() });
    });
}
