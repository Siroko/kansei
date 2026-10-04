//! GI box: a Cornell box (white floor, ceiling and back wall, a red wall on the left, a green one
//! on the right, two white blocks, a rug half orange and half blue) under one shadowed downlight,
//! to compare global illumination methods. Without GI the scene has direct light only, so the
//! ceiling and the shadows are black; with it the floor's light reaches the ceiling and the walls
//! bleed their colour onto the floor and the blocks.
//!
//! - Screen-space GI (`gi=low|medium|high|ultra`) sees only what is on screen.
//! - Voxel GI (`gi=voxel`) traces cones through a voxel volume of the room: the renderables are
//!   voxelized through their own vertex shaders and lit through the lamp's shadow map, with the
//!   bounces adding up over frames, so light from off screen (the open front's side of the walls,
//!   the backs of the blocks) arrives too. The rug's voxels take its texture through its
//!   material's voxel entry (`albedo=constant`: its mean colour instead).
//! - `gi=voxel+ssgi`: screen-space GI in front for contact detail, the voxels for the rest.
//!
//! URL parameters: `gi=off|low|medium|high|ultra|voxel|voxel+ssgi` (default high), `voxels=low|
//! medium|high` (the volume's resolution: default medium, low on phones, which also keep it within
//! 24 MiB), `albedo=constant`, `rug=off` (the box without its rug), `view=indirect` (only the
//! light GI adds, 2 stops brighter), `view=voxels` (with voxel GI: the lit voxels themselves),
//! `stats=1` (log the frame interval, which is the GPU time when the browser runs without vsync).

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, VOXEL_WRITE_WGSL};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
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

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    volume: PostProcessingVolume,
    stats: Option<(f64, u32)>,
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

/// The global illumination asked for (`gi=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gi {
    Off,
    Screen(GiQuality),
    Voxel,
    VoxelAndScreen,
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
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

    // the room: 4 m wide, high and deep, open toward the camera (z = 0), walls 0.2 m thick
    let gi = match query_param("gi").as_deref() {
        Some("off") => Gi::Off,
        Some("low") => Gi::Screen(GiQuality::Low),
        Some("medium") => Gi::Screen(GiQuality::Medium),
        Some("ultra") => Gi::Screen(GiQuality::Ultra),
        Some("voxel") => Gi::Voxel,
        // a raw '+' in the query, or one decoded to a space or escaped
        Some("voxel+ssgi" | "voxel ssgi" | "voxel%2Bssgi" | "voxel%2bssgi") => Gi::VoxelAndScreen,
        _ => Gi::Screen(GiQuality::High),
    };
    let voxels = matches!(gi, Gi::Voxel | Gi::VoxelAndScreen);
    let phone = is_phone();
    if voxels {
        // the room and a margin; phones take the low tier within 24 MiB
        let quality = query_param("voxels").as_deref().and_then(VoxelGiQuality::from_name).unwrap_or(if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium });
        renderer.enable_voxel_gi(SceneVoxelGiOptions {
            quality,
            bounds_min: [-2.3, -0.3, -4.3],
            bounds_max: [2.3, 4.3, 0.3],
            radiance_scale: 1.0,
            budget_bytes: if phone { 24 << 20 } else { 0 },
        });
    }
    let textured_voxels = query_param("albedo").as_deref() != Some("constant");

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
    let mut tall = Renderable::new(BoxGeometry::new(1.2, 2.4, 1.2), lit_material("Tall", white)).with_gi(GiSurface::new(white));
    tall.object.set_position(-0.75, 1.2, -2.6);
    tall.object.rotation.y = 0.33;
    scene.add(SceneNode::Renderable(tall));
    let mut short = Renderable::new(BoxGeometry::new(1.2, 1.2, 1.2), lit_material("Short", white)).with_gi(GiSurface::new(white));
    short.object.set_position(0.8, 0.6, -1.5);
    short.object.rotation.y = -0.3;
    scene.add(SceneNode::Renderable(short));
    // a textured rug on the floor in front of the blocks, orange before the tall one and blue
    // before the short one, whose sides facing the camera only bounces light; its constant
    // surface (its mean colour) is used where the material's voxel entry is not
    let mean: [f32; 3] = std::array::from_fn(|c| (RUG_LEFT[c] + RUG_RIGHT[c]) * 0.5);
    if query_param("rug").as_deref() != Some("off") {
        let mut rug = Renderable::new(BoxGeometry::new(2.4, 0.02, 0.8), rug_material(textured_voxels)).with_gi(GiSurface::new(mean));
        rug.object.set_position(0.0, 0.01, -0.55);
        scene.add(SceneNode::Renderable(rug));
    }

    // the key light: a shadowed downlight just under the ceiling's centre, wide enough to light
    // the tops of the side walls
    let mut lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    lamp.cast_shadow = true;
    lamp.source_radius = 0.15;
    lamp.volumetric_scale = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    let indirect = query_param("view").as_deref() == Some("indirect");
    let show_voxels = query_param("view").as_deref() == Some("voxels");
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    let screen = ScreenSpaceGIOptions { radius_m: 4.0, ..Default::default() };
    match gi {
        Gi::Off => {}
        Gi::Screen(quality) => {
            let mut effect = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, ..screen });
            effect.show_indirect = indirect;
            effects.push(Box::new(effect));
        }
        Gi::Voxel | Gi::VoxelAndScreen => {
            let scene_gi = renderer.voxel_gi().unwrap();
            let near_field = (gi == Gi::VoxelAndScreen).then_some(ScreenSpaceGIOptions { quality: if phone { GiQuality::Low } else { GiQuality::High }, ..screen });
            let mut effect = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), near_field, ..Default::default() });
            effect.show_indirect = indirect;
            effect.show_voxels = show_voxels;
            effects.push(Box::new(effect));
        }
    }
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    // the indirect view alone is 2 stops brighter
    tone.exposure = exposure_from_ev100(5.0) * if indirect { 4.0 } else { 1.0 };
    effects.push(Box::new(ToneMapEffect::new(tone)));
    let volume = PostProcessingVolume::new(&renderer, effects);

    // framing the box's opening, as the Cornell box photographs do
    let mut camera = Camera::new(38.0, 0.1, 100.0, width as f32 / height as f32);
    camera.set_position(0.0, 2.0, 6.3);
    camera.look_at(&Vec3::new(0.0, 2.0, -2.0));
    camera.update_projection_matrix();

    log::info!(
        "Kansei — GI box (WASM) ready: gi {gi:?}{}, view {}",
        renderer.voxel_gi().map(|g| format!(" ({:?} voxels, {:?}, {:.1} MiB)", g.quality(), g.volume().dims(), g.memory_bytes() as f64 / (1 << 20) as f64)).unwrap_or_default(),
        if indirect { "indirect" } else if show_voxels { "voxels" } else { "lit" }
    );

    let stats = (query_param("stats").as_deref() == Some("1")).then_some((now_ms(), 0));
    let state = Rc::new(RefCell::new(State { renderer, scene, camera, volume, stats }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, ref mut volume, ref mut stats } = *st;
            renderer.render_with_postprocessing(scene, camera, volume);
            if let Some((start, frames)) = stats {
                *frames += 1;
                if *frames == 240 {
                    log::info!("frame interval {:.2} ms", (now_ms() - *start) / 240.0);
                    *start = now_ms();
                    *frames = 0;
                }
            }
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(())
}
