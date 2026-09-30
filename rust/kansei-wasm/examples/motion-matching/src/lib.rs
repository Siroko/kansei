//! Motion matching: a skinned character walking, running, stopping and turning under keyboard
//! or gamepad control, animated by searching a database of animation frames
//! (`kansei_core::animation::motion_matching`), on a sunlit ground plane with cascaded shadows
//! and TAA (the character writes motion vectors from last frame's bone palette).
//!
//! The animation comes from a motion-matching pack (`.kmm`) baked with `kansei-anim-bake`. None
//! ships with Kansei: the page loads `pack/locomotion.kmm` next to `index.html` (or `pack=<url>`)
//! and says how to make one when it is missing. See this example's README.
//!
//! A character pack (`pack/hero.kmm`, or `hero=<url>`), when there is one, is a second body for
//! the same animation: a mesh rigged to the same skeleton with its own proportions and textures,
//! the pose retargeted onto it (`animation::retarget`). It is shown by default; C switches
//! between it and the motion pack's own mesh (`char=hero` or `char=mannequin` to choose).
//!
//! A course of boxes stands around the start: low rails to hurdle, boxes to vault, blocks to mantle
//! onto, walls to climb, long narrow beams and stacked blocks, some turned. Space in front of one
//! traverses it (`motion_matching::traversal`): the kind from its shape, the clip from the pace,
//! the clip's root motion warped onto its ledge. With nothing to traverse ahead, Space jumps: a
//! jump clip for the pace up to its take-off, then a ballistic flight that lands wherever it
//! comes down, box tops included. Walking off a top falls and lands. The pack needs action clips
//! for that (`kansei-anim-bake`'s `actions`); without them Space does nothing.
//!
//! Controls: WASD or arrows move relative to the camera, Shift runs, Space jumps or traverses, Q
//! toggles strafing (face the camera's direction), mouse drag orbits and the wheel zooms. Gamepad:
//! left stick moves (tilt sets the pace), right stick orbits, A jumps or traverses, B or the right
//! trigger runs, the left bumper toggles strafing. Keys B, K, M, L and C toggle the trajectory
//! overlay and HUD, the skeleton, the mesh, foot locking and the character.
//!
//! A small lake lies east of the course (`lake`): SPH water in a container shaped like the lake,
//! which the character wades into, its legs pushing the water (wakes, splashes and ripples).
//!
//! URL parameters: `pack=<url>`, `gait=0` (search every clip whatever the gait, instead of
//! idle + walk or idle + run by the pack's tags), `taa=0`, `walk=<m/s>`, `run=<m/s>` (forward
//! paces; sideways and backward scale with them), `course=0` (no boxes), `lake=0` (no lake),
//! `at=<x>,<z>,<heading in degrees>` (where the character starts; `at=14,-1,90` at the lake).

mod lake;

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use glam::{Mat4, Quat, Vec3 as GVec3};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use kansei_core::animation::motion_matching::pack::{CharacterPack, MotionPack};
use kansei_core::animation::retarget::Retarget;
use kansei_core::animation::motion_matching::traversal::{CharacterController, CharacterState};
use kansei_core::animation::motion_matching::{yaw_of, Database, MotionInput, MotionMatcher, MotionMatchingSettings, ACTION_TAG};
use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::animation::{skin_buffer, skinned_lit_material, BonePalette, Skeleton, SkinnedLitParams, SkinnedMesh, PALETTE_BINDING, SKINNING_WGSL, SKIN_BINDING};
use kansei_core::buffers::{Bindable, BufferType, ComputeBuffer, Sampler};
use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::materials::BindingResource;
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    effects::{exposure_from_ev100, FluidSurfaceEffect, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions},
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

/// Course boxes: flat colour with a half-metre grid on every face, sun (cascade-shadowed) and sky.
const BOX_WGSL: &str = r#"
struct Surface { base_color: vec4<f32>, sun_dir: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
fn lines(p: vec2<f32>) -> f32 {
    let q = p / 0.5;
    let d = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(d.x, d.y), 1.0);
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let a = abs(n);
    var p = in.world.xz;
    if (a.x > a.y && a.x > a.z) { p = in.world.zy; } else if (a.z > a.y) { p = in.world.xy; }
    let base = surface.base_color.rgb * (1.0 - 0.3 * lines(p));
    let l = -normalize(surface.sun_dir.xyz);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let sky = mix(surface.sky.rgb * 0.25, surface.sky.rgb, n.y * 0.5 + 0.5);
    let lit = base / 3.14159265 * surface.sun.rgb * max(dot(n, l), 0.0) * shadow + base * sky;
    return vec4<f32>(lit, 1.0);
}
"#;

/// The course: (x, base height, z) of a box's bottom centre, its width, height and depth, its
/// heading (radians) and colour.
const COURSE: [([f32; 3], [f32; 3], f32, [f32; 3]); 20] = [
    // low rails to hurdle, one long and turned
    ([0.0, 0.0, 5.0], [3.0, 0.5, 0.25], 0.0, [0.55, 0.35, 0.2]),
    ([-6.0, 0.0, 4.0], [4.0, 0.8, 0.25], 0.5, [0.55, 0.35, 0.2]),
    ([6.5, 0.0, 3.0], [2.5, 1.0, 0.3], -0.4, [0.55, 0.35, 0.2]),
    // boxes to vault
    ([0.0, 0.0, 10.0], [2.0, 1.0, 0.8], 0.0, [0.25, 0.4, 0.55]),
    ([-9.0, 0.0, 9.0], [2.2, 0.9, 1.0], 0.8, [0.25, 0.4, 0.55]),
    // blocks to mantle onto
    ([7.0, 0.0, 9.5], [3.0, 1.2, 3.0], 0.3, [0.45, 0.45, 0.4]),
    ([-3.5, 0.0, -6.0], [3.0, 1.5, 2.5], 0.0, [0.45, 0.45, 0.4]),
    ([4.0, 0.0, -5.0], [2.5, 1.35, 2.5], -0.7, [0.45, 0.45, 0.4]),
    // walls to climb, with room on top
    ([0.0, 0.0, 16.0], [4.0, 2.0, 3.0], 0.0, [0.5, 0.3, 0.3]),
    ([-10.0, 0.0, -2.0], [3.0, 2.4, 3.5], std::f32::consts::FRAC_PI_2, [0.5, 0.3, 0.3]),
    // long narrow beams: hurdle across, too narrow to stand along
    ([10.0, 0.0, -3.0], [0.35, 0.6, 7.0], 0.0, [0.3, 0.3, 0.3]),
    ([-6.0, 0.0, 13.0], [6.0, 0.7, 0.35], -0.2, [0.3, 0.3, 0.3]),
    // stacked: mantle onto the first, then onto the second (set back: room to stand in front
    // of it, not on the other sides)
    ([12.0, 0.0, 12.0], [3.0, 1.2, 3.0], 0.2, [0.45, 0.45, 0.4]),
    ([12.11, 1.2, 12.54], [1.8, 1.1, 1.8], 0.2, [0.55, 0.5, 0.35]),
    ([-12.0, 0.0, 16.0], [3.5, 1.0, 3.5], -0.5, [0.45, 0.45, 0.4]),
    ([-12.22, 1.0, 16.39], [1.8, 1.3, 1.8], -0.5, [0.55, 0.5, 0.35]),
    // a low step onto a platform, and a thin wall at an angle
    ([0.0, 0.0, -10.0], [5.0, 0.3, 3.0], 0.0, [0.4, 0.42, 0.45]),
    ([8.0, 0.0, 16.0], [3.0, 1.1, 0.3], 0.9, [0.55, 0.35, 0.2]),
    // two platforms with a gap to jump across (a running jump; a walking one falls short)
    ([-14.0, 0.0, 3.0], [3.0, 1.2, 6.0], 0.0, [0.4, 0.5, 0.45]),
    ([-14.0, 0.0, 9.5], [3.0, 1.2, 4.0], 0.0, [0.4, 0.5, 0.45]),
];

/// The course's boxes as colliders and renderables.
fn build_course(scene: &mut Scene, world: &mut CollisionWorld) {
    for (base, size, yaw, color) in COURSE {
        let center = GVec3::new(base[0], base[1] + size[1] * 0.5, base[2]);
        world.add_box(Obb::new(center, GVec3::from(size) * 0.5, Quat::from_rotation_y(yaw)));
        let mut material = Material::new("Box", &format!("{CASCADED_SHADOWS_WGSL}\n{BOX_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
        material.set_uniform_bindable(0, "Box", &surface_params(color));
        let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), material);
        r.object.set_position(center.x, center.y, center.z);
        r.object.rotation.y = yaw;
        scene.add(SceneNode::Renderable(r));
    }
}

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

/// The textured character: `SkinnedLitParams` (base_color tints the colour texture), then its
/// colour (sRGB), normal (tangent space, +Y up) and occlusion/roughness/metallic textures.
const HERO_WGSL: &str = r#"
struct Surface { tint: vec4<f32>, sun_direction: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(0) @binding(3) var base_color_texture: texture_2d<f32>;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var orm_texture: texture_2d<f32>;
@group(0) @binding(6) var texture_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @builtin(vertex_index) vertex: u32, @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) curr: vec4<f32>,
    @location(4) prev: vec4<f32>,
};
struct FOut { @location(0) color: vec4<f32>, @location(4) velocity: vec2<f32> };

@vertex
fn vertex_main(v: VIn) -> VOut {
    let s = kansei_skin(v.vertex, v.position.xyz, v.normal);
    let world = mesh.world * vec4<f32>(s.position, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(s.normal, 0.0)).xyz;
    out.uv = v.uv;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4<f32>(s.prev_position, 1.0));
    return out;
}

// The normal map in the frame of the surface's position and uv derivatives (no stored tangents).
// glTF uv run down the image while the map's +Y is up, hence -B.
fn mapped_normal(n: vec3<f32>, p: vec3<f32>, uv: vec2<f32>, m: vec3<f32>) -> vec3<f32> {
    let dp1 = dpdx(p);
    let dp2 = dpdy(p);
    let duv1 = dpdx(uv);
    let duv2 = dpdy(uv);
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return normalize(t * scale * m.x - b * scale * m.y + n * m.z);
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> FOut {
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    let albedo = textureSample(base_color_texture, texture_sampler, in.uv).rgb * surface.tint.rgb;
    let orm = textureSample(orm_texture, texture_sampler, in.uv).rgb;
    let m = textureSample(normal_texture, texture_sampler, in.uv).xyz * 2.0 - 1.0;
    n = mapped_normal(n, in.world, in.uv, m);
    let l = -normalize(surface.sun_direction.xyz);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let view = normalize(-(transpose(mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz)) * view_matrix[3].xyz) - in.world);
    let h = normalize(l + view);
    let roughness = clamp(orm.g, 0.08, 1.0);
    let a2 = roughness * roughness * roughness * roughness;
    let nh = max(dot(n, h), 0.0);
    let d = a2 / (3.14159265 * pow(nh * nh * (a2 - 1.0) + 1.0, 2.0));
    let nl = max(dot(n, l), 0.0);
    let specular = mix(vec3<f32>(0.04), albedo, orm.b) * d * 0.25;
    let sky = mix(surface.sky.rgb * 0.2, surface.sky.rgb, n.y * 0.5 + 0.5);
    let lit = (albedo * (1.0 - orm.b) / 3.14159265 + specular) * surface.sun.rgb * nl * shadow + albedo * sky * orm.r;
    return FOut(vec4<f32>(lit, 1.0), kansei_motion_vector(in.curr, in.prev));
}
"#;

/// A texture already on the GPU, bound as it is.
struct GpuTexture {
    view: wgpu::TextureView,
}

impl Bindable for GpuTexture {
    fn ensure_ready(&mut self, _: &wgpu::Device, _: &wgpu::Queue) {}
    fn binding_resource(&self) -> Option<BindingResource<'_>> {
        Some(BindingResource::TextureView(&self.view))
    }
}

/// An image with its mip chain on the GPU.
fn upload_texture(renderer: &Renderer, label: &str, image: image::RgbaImage, srgb: bool) -> GpuTexture {
    let mut levels = vec![image];
    while levels.last().is_some_and(|l| l.width() > 1 || l.height() > 1) {
        let l = levels.last().unwrap();
        let next = image::imageops::resize(l, (l.width() / 2).max(1), (l.height() / 2).max(1), image::imageops::FilterType::Triangle);
        levels.push(next);
    }
    let (width, height) = (levels[0].width(), levels[0].height());
    let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip, level) in levels.iter().enumerate() {
        renderer.queue().write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: mip as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            level.as_raw(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(4 * level.width()), rows_per_image: None },
            wgpu::Extent3d { width: level.width(), height: level.height(), depth_or_array_layers: 1 },
        );
    }
    GpuTexture { view: texture.create_view(&Default::default()) }
}

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

/// A mesh the character can be shown as: the motion pack's own, on the database's skeleton, or a
/// character pack's, the pose retargeted onto its skeleton.
struct Body {
    name: &'static str,
    mesh: SkinnedMesh,
    palette: BonePalette,
    index: usize,
    display: Option<(Skeleton, Retarget)>,
}

/// A character pack as a body: its textured mesh, hidden until shown.
fn hero_body(renderer: &Renderer, scene: &mut Scene, pack: CharacterPack, db: &Database) -> Result<Body, String> {
    let retarget = Retarget::new(&db.skeleton, &pack.skeleton, &Retarget::UNREAL_KEEP);
    let texture = |name: &str, srgb: bool, fallback: [u8; 4]| -> Result<GpuTexture, String> {
        let image = match pack.image(name) {
            Some(i) => image::load_from_memory(&i.bytes).map_err(|e| format!("texture {name}: {e}"))?.to_rgba8(),
            None => image::RgbaImage::from_pixel(1, 1, image::Rgba(fallback)),
        };
        Ok(upload_texture(renderer, name, image, srgb))
    };
    let (base_color, normal, orm) = (texture("base_color", true, [200, 200, 200, 255])?, texture("normal", false, [128, 128, 255, 255])?, texture("orm", false, [255, 160, 0, 255])?);
    let first = pack.meshes.into_iter().next().ok_or("the character pack has no mesh")?;
    let mesh = first.mesh;
    let mut palette = BonePalette::new(mesh.skin_joints.len());
    palette.update(&mesh, &pack.skeleton.rest_model());
    let params = SkinnedLitParams { base_color: first.color, sun_direction: [SUN_DIR[0], SUN_DIR[1], SUN_DIR[2], 0.0], sun: [SUN[0], SUN[1], SUN[2], 0.0], sky: [SKY[0], SKY[1], SKY[2], 0.0] };
    let shader = format!("{SKINNING_WGSL}\n{MOTION_VECTORS_WGSL}\n{CASCADED_SHADOWS_WGSL}\n{HERO_WGSL}");
    let mut material = Material::new(
        "Hero",
        &shader,
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::storage(PALETTE_BINDING, ShaderStages::VERTEX, true),
            Binding::storage(SKIN_BINDING, ShaderStages::VERTEX, true),
            Binding::texture_2d(3, ShaderStages::FRAGMENT),
            Binding::texture_2d(4, ShaderStages::FRAGMENT),
            Binding::texture_2d(5, ShaderStages::FRAGMENT),
            Binding::sampler(6, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { outputs_velocity: true, ..Default::default() },
    );
    material.set_uniform_bindable(0, "Hero/Params", &[params]);
    material.set_bindable(PALETTE_BINDING, palette.buffer("Hero/Palette"));
    material.set_bindable(SKIN_BINDING, skin_buffer("Hero/Skin", &mesh));
    material.set_bindable(3, base_color);
    material.set_bindable(4, normal);
    material.set_bindable(5, orm);
    material.set_bindable(6, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
    let mut r = Renderable::new(mesh.geometry(), material);
    r.dynamic = true;
    r.visible = false;
    let index = scene.add(SceneNode::Renderable(r));
    log::info!("character pack: {} joints, {} vertices, {} triangles", pack.skeleton.len(), mesh.vertices.len(), mesh.indices.len() / 3);
    Ok(Body { name: "hero", mesh, palette, index, display: Some((pack.skeleton, retarget)) })
}

/// The character: its database, matcher, bodies, and the debug markers.
struct Character {
    db: Database,
    controller: CharacterController,
    bodies: Vec<Body>,
    /// The last obstacle found, marked at its ledge.
    ledge: Markers,
    showing: usize,
    bones: Markers,
    trajectory: Markers,
    /// Tag bits the search may use while walking and while running (all when the pack has no
    /// gait tags or `gait=0`).
    walk_tags: u32,
    run_tags: u32,
}

impl Character {
    fn new(renderer: &Renderer, scene: &mut Scene, pack: MotionPack, hero: Option<CharacterPack>, gait: bool) -> Result<Self, String> {
        let tags: Vec<String> = pack.meta("tags").unwrap_or("").split(',').map(str::to_string).collect();
        let MotionPack { database: db, meshes, actions, .. } = pack;
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
        let mut bodies = vec![Body { name: "mannequin", mesh, palette, index, display: None }];
        if let Some(pack) = hero {
            match hero_body(renderer, scene, pack, &db) {
                Ok(b) => bodies.push(b),
                Err(e) => log::warn!("character pack left out: {e}"),
            }
        }
        let joints = bodies.iter().map(|b| b.display.as_ref().map_or(db.joint_count(), |d| d.0.len())).max().unwrap_or(0);
        let bones = Markers::new(renderer, scene, "Bones", joints, [30000.0, 20000.0, 4000.0], true);
        bones.set_visible(scene, false);
        // the simulation now and its 3 predicted samples, and each foot's target
        let trajectory = Markers::new(renderer, scene, "Trajectory", 6, [300.0, 1600.0, 3000.0], false);
        let bit = |name: &str| tags.iter().position(|t| t == name).map_or(0, |b| 1u32 << b);
        let (idle, walk, run) = (bit("idle"), bit("walk"), bit("run"));
        let (walk_tags, run_tags) = if gait && walk != 0 && run != 0 { (idle | walk, idle | run) } else { (!ACTION_TAG, !ACTION_TAG) };
        let matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), GVec3::ZERO, 0.0);
        log::info!("{} action clips", actions.len());
        let controller = CharacterController::new(matcher, actions);
        let ledge = Markers::new(renderer, scene, "Ledge", 2, [3000.0, 400.0, 200.0], true);
        log::info!("motion pack: {} clips, {} frames, {} joints", db.clips.len(), db.frame_count(), db.joint_count());
        Ok(Self { db, controller, bodies, ledge, showing: 0, bones, trajectory, walk_tags, run_tags })
    }

    /// Show body `which` (its mesh, the pose on its skeleton).
    fn show(&mut self, scene: &mut Scene, which: usize) {
        self.showing = which % self.bodies.len();
        for (i, b) in self.bodies.iter_mut().enumerate() {
            if let Some(r) = scene.get_renderable_mut(b.index) {
                r.visible = i == self.showing;
                r.reset_motion();
            }
            b.palette.reset_motion();
        }
        self.controller.matcher.set_display(&self.db, self.bodies[self.showing].display.clone());
    }

    /// The output skeleton's joint named as database joint `j`.
    fn joint(&self, j: usize) -> usize {
        self.controller.matcher.output_skeleton(&self.db).find(&self.db.skeleton.names[j]).unwrap_or(0)
    }

    /// The legs as capsules (world ends, radius): each thigh, shin and foot (hip to knee to ankle
    /// to toe, up the foot's parents and down to its first child), and the hips.
    fn leg_capsules(&self) -> Vec<(GVec3, GVec3, f32)> {
        let matcher = &self.controller.matcher;
        let (character, model, skeleton) = (matcher.character(), matcher.model(), matcher.output_skeleton(&self.db));
        let at = |j: usize| character.transform_point(model[j].translation);
        let mut legs = Vec::new();
        for side in 0..2 {
            let foot = self.joint(self.db.roles.feet[side]);
            let Some(knee) = skeleton.parents[foot] else { continue };
            let Some(hip) = skeleton.parents[knee] else { continue };
            let toe = skeleton.parents.iter().position(|p| *p == Some(foot)).map_or(at(foot) + character.rotation * GVec3::Z * 0.15, at);
            legs.extend([(at(hip), at(knee), 0.085), (at(knee), at(foot), 0.06), (at(foot), toe, 0.05)]);
        }
        let hips = at(self.joint(self.db.roles.hips));
        legs.push((hips, hips, 0.14));
        legs
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
    world: CollisionWorld,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    character: Option<Character>,
    lake: Option<lake::Lake>,
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
    /// Whether the character was in the air last frame, and the fastest it has fallen since
    /// (m/s); its height last frame.
    air: (bool, f32),
    last_y: f32,
    /// `profile=1`: when the profile was last logged (0: not profiling).
    profile_since: f64,
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
            run |= pressed(1) || buttons.get(7).is_some_and(|b| b.1 > 0.3);
            let was = |i: usize| self.pad_held.get(i).copied().unwrap_or(false);
            if pressed(0) && !was(0) {
                toggles.push(" ".into());
            }
            if pressed(4) && !was(4) {
                toggles.push("q".into());
            }
            self.pad_held = buttons.iter().map(|b| b.0).collect();
        }
        let mut traverse = false;
        for key in toggles {
            match key.as_str() {
                " " => traverse = true,
                "q" => self.strafe = !self.strafe,
                "b" => self.overlay = !self.overlay,
                "c" => {
                    if let Some(c) = &mut self.character {
                        let next = c.showing + 1;
                        c.show(&mut self.scene, next);
                    }
                }
                "k" | "m" | "l" => {
                    if let Some(c) = &mut self.character {
                        match key.as_str() {
                            "k" => {
                                let visible = self.scene.get_renderable(c.bones.index).is_some_and(|r| r.visible);
                                c.bones.set_visible(&mut self.scene, !visible);
                            }
                            "m" => {
                                if let Some(r) = self.scene.get_renderable_mut(c.bodies[c.showing].index) {
                                    r.visible = !r.visible;
                                }
                            }
                            _ => c.controller.matcher.settings.foot_lock = !c.controller.matcher.settings.foot_lock,
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
            c.controller.matcher.settings.filter.tags = if run { c.run_tags } else { c.walk_tags };
            c.controller.update(&c.db, &self.world, &MotionInput { velocity, facing }, dt);
            if traverse {
                // an obstacle ahead: traverse it (pressed a little early, once in reach); else jump
                let _ = c.controller.request_traverse_or_jump(&c.db, &self.world, 1.0);
            }
            // the obstacle last looked at: its ledge, and a post down to the floor
            c.ledge.matrices.fill(Mat4::ZERO);
            if let (true, Some(o)) = (self.overlay, c.controller.last_obstacle) {
                let side = o.normal.cross(GVec3::Y);
                c.ledge.matrices[0] = segment(o.ledge - side * 0.4, o.ledge + side * 0.4, 0.04);
                c.ledge.matrices[1] = segment(o.ledge, o.ledge - GVec3::Y * o.height, 0.02);
            }
            c.ledge.upload(&self.renderer);
            let search = c.controller.matcher.last_search();
            self.searches += search.searched as u32;
            self.switches += search.switched as u32;

            let character = c.controller.matcher.character();
            let body = &mut c.bodies[c.showing];
            if let Some(r) = self.scene.get_renderable_mut(body.index) {
                r.object.set_position(character.translation.x, character.translation.y, character.translation.z);
                r.object.rotation.y = yaw_of(character.rotation);
                body.palette.update(&body.mesh, c.controller.matcher.model());
                if let Some(buffer) = r.material.bindable_buffer(PALETTE_BINDING) {
                    body.palette.upload(self.renderer.queue(), &buffer);
                }
            }
            // follow the character's hips
            let hips = character.transform_point(c.controller.matcher.model()[c.joint(c.db.roles.hips)].translation);
            let target = GVec3::new(character.translation.x, hips.y * 0.9, character.translation.z);
            let t = 1.0 - (-dt * 8.0).exp();
            let current = GVec3::new(self.controls.target.x, self.controls.target.y, self.controls.target.z);
            let followed = current.lerp(target, t);
            self.controls.target = Vec3::new(followed.x, followed.y, followed.z);

            if self.overlay {
                // the simulation and its predicted samples: flat boxes pointing where they face
                let s = c.controller.matcher.simulation();
                let mut k = 0;
                for (p, q, size) in std::iter::once((s.position, s.rotation, 0.16)).chain(c.controller.matcher.trajectory().iter().map(|t| (t.translation, t.rotation, 0.1))) {
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::new(size, 0.02, size * 2.0), q, p + GVec3::Y * 0.01);
                    k += 1;
                }
                // each foot's target, raised when not planted
                let model = c.controller.matcher.model();
                for (side, locked) in c.controller.matcher.feet_locked().iter().enumerate() {
                    let p = character.transform_point(model[c.joint(c.db.roles.feet[side])].translation);
                    let size = if *locked { 0.09 } else { 0.04 };
                    c.trajectory.matrices[k] = Mat4::from_scale_rotation_translation(GVec3::splat(size), Quat::IDENTITY, GVec3::new(p.x, character.translation.y + 0.02, p.z));
                    k += 1;
                }
            } else {
                c.trajectory.matrices.fill(Mat4::ZERO);
            }
            c.trajectory.upload(&self.renderer);
            if self.scene.get_renderable(c.bones.index).is_some_and(|r| r.visible) {
                let model = c.controller.matcher.model();
                let root = c.joint(c.db.roles.root);
                let parents = c.controller.matcher.output_skeleton(&c.db).parents.clone();
                c.bones.matrices.fill(Mat4::ZERO);
                for (j, parent) in parents.iter().enumerate() {
                    if let Some(p) = parent.filter(|p| *p != root) {
                        c.bones.matrices[j] = segment(character.transform_point(model[p].translation), character.transform_point(model[j].translation), 0.015);
                    }
                }
                c.bones.upload(&self.renderer);
            }
        }
        if let Some(lake) = &mut self.lake {
            let legs = self.character.as_ref().map(Character::leg_capsules).unwrap_or_default();
            // a landing: back on its feet after being in the air, at the fastest it came down
            let mut landing = None;
            if let Some(c) = &self.character {
                let y = c.controller.matcher.character().translation.y;
                let airborne = matches!(c.controller.state(), CharacterState::Jumping | CharacterState::Falling(_));
                let (was, fall) = &mut self.air;
                if airborne {
                    *fall = fall.max((self.last_y - y) / dt);
                } else if *was {
                    landing = Some((c.controller.matcher.character().translation, *fall));
                    *fall = 0.0;
                }
                *was = airborne;
                self.last_y = y;
            }
            if let Some(surface) = self.volume.effects.get_mut(lake.effect).and_then(|e| e.as_any_mut().downcast_mut::<FluidSurfaceEffect>()) {
                lake.update(surface, &legs, landing, dt);
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
                    let (clip, frame) = c.controller.matcher.playing();
                    let info = &c.db.clips[clip];
                    let s = c.controller.matcher.last_search();
                    let feet = c.controller.matcher.feet_locked();
                    let speed = c.controller.matcher.simulation().velocity.length();
                    set_hud(&format!(
                        "{:.0} fps   {} {}{}\nclip   {}\nframe  {:.0} / {}{}\nsearch {:.0}/s, switch {:.1}/s, cost {:.3}\nfeet   {} {}  (lock {})\n{}\n\n{}\n\nWASD / left stick move · Shift / B run · Space / A jump, traverse · Q / LB strafe\ndrag / right stick orbit · B overlay · K skeleton · M mesh · L foot lock",
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
                        if c.controller.matcher.settings.foot_lock { "on" } else { "off" },
                        format_args!(
                            "state  {}{}",
                            match c.controller.state() {
                                CharacterState::Grounded => "on the ground".to_string(),
                                CharacterState::Traversing(k) => ["hurdling", "vaulting", "mantling", "climbing", "falling", "landing", "jumping"][k as usize].to_string(),
                                CharacterState::Jumping => "jumping".to_string(),
                                CharacterState::Falling(t) => format!("in the air {t:.1} s"),
                                CharacterState::Landing => "landing".to_string(),
                            },
                            match c.controller.last_result {
                                Some(Ok(k)) => format!("   last Space: {}", k.name()),
                                Some(Err(e)) => format!("   last Space: {}", e.describe()),
                                None => String::new(),
                            }
                        ),
                        format_args!(
                            "{}character: {} (C to switch)",
                            self.lake.as_ref().map_or(String::new(), |l| format!("lake   {} particles, east of the course (at=14,-1,90), P tweaks\n", l.particles())),
                            c.bodies[c.showing].name,
                        ),
                    ));
                }
                Some(_) => set_hud(""),
                None => {}
            }
        }

        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if self.profile_since > 0.0 && now - self.profile_since > 3.0 {
            self.profile_since = now;
            log::info!("profile ({:.1} ms/frame)\n{}", 1000.0 / self.fps, self.renderer.take_profile().report());
        }
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
    // the floor and the course, for collision
    let mut world = CollisionWorld::new();
    let with_lake = query_param("lake").as_deref() != Some("0");
    if !with_lake {
        world.add_box(Obb::from_min_max(GVec3::new(-200.0, -1.0, -200.0), GVec3::new(200.0, 0.0, 200.0)));
    }
    if query_param("course").as_deref() != Some("0") {
        build_course(&mut scene, &mut world);
    }
    let mut sky = Material::new("Sky", SKY_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { cull_mode: CullMode::None, ..Default::default() });
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));
    let mut ground_material = Material::new("Ground", &format!("{CASCADED_SHADOWS_WGSL}\n{GROUND_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    ground_material.set_uniform_bindable(0, "Ground", &surface_params([0.32, 0.32, 0.3]));
    // with the lake, the ground has a hole the lake's terrain fills
    let lake = if with_lake {
        Some(lake::Lake::new(&renderer, &mut scene, &mut world, ground_material, 0))
    } else {
        let mut ground = Renderable::new(PlaneGeometry::new(400.0, 400.0), ground_material);
        ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
        ground.cast_shadow = false;
        scene.add(SceneNode::Renderable(ground));
        None
    };
    let mut sun = DirectionalLight::new(Vec3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]), Vec3::new(1.0, 0.9, 0.75), 80000.0);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    // the character, from a pack outside the repository
    let url = query_param("pack").unwrap_or_else(|| "pack/locomotion.kmm".to_string());
    set_hud(&format!("Loading motion pack {url} …"));
    let gait = query_param("gait").as_deref() != Some("0");
    let motion = fetch_bytes(&url).await.and_then(|bytes| MotionPack::from_bytes(&bytes));
    // a second body, optional
    let hero_url = query_param("hero").unwrap_or_else(|| "pack/hero.kmm".to_string());
    let hero = match &motion {
        Ok(_) => match fetch_bytes(&hero_url).await.and_then(|bytes| CharacterPack::from_bytes(&bytes)) {
            Ok(h) => Some(h),
            Err(e) => {
                log::info!("no character pack: {e}");
                None
            }
        },
        Err(_) => None,
    };
    let character = match motion {
        Ok(pack) => match Character::new(&renderer, &mut scene, pack, hero, gait) {
            Ok(mut c) => {
                let wanted = match query_param("char").as_deref() {
                    Some("mannequin") => 0,
                    _ => c.bodies.len() - 1,
                };
                c.show(&mut scene, wanted);
                if let Some(at) = query_param("at") {
                    let v: Vec<f32> = at.split(',').filter_map(|x| x.parse().ok()).collect();
                    if v.len() >= 2 {
                        // on whatever is there (a box top)
                        let y = world.ground_height(GVec3::new(v[0], 0.0, v[1]), 50.0, 50.0, u32::MAX).unwrap_or(0.0);
                        c.controller.matcher.teleport(GVec3::new(v[0], y, v[1]), v.get(2).copied().unwrap_or(0.0).to_radians());
                    }
                }
                Some(c)
            }
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
    // the water's refraction and reflection come first, on the lit scene (the lake's `effect` 0)
    let lake = lake.map(|(lake, surface)| {
        effects.push(Box::new(surface));
        lake
    });
    if query_param("taa").as_deref() != Some("0") {
        effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    }
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);
    if let Some(surface) = lake.as_ref().and_then(|l| volume.effects[l.effect].as_any().downcast_ref::<FluidSurfaceEffect>()) {
        lake::Lake::add_surface(&mut scene, surface);
    }
    let mut camera = Camera::new(45.0, 0.1, 1200.0, width as f32 / height as f32);
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let mut controls = CameraControls::from_canvas(&canvas, Vec3::new(start.translation.x, 0.9, start.translation.z), 4.5);
    controls.set_elevation(0.25);
    controls.set_azimuth(std::f32::consts::PI + yaw_of(start.rotation));

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
            if e.key().starts_with("Arrow") || e.key() == " " {
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
        world,
        scene,
        camera,
        controls,
        volume,
        character,
        lake,
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
        profile_since: 0.0,
        air: (false, 0.0),
        last_y: 0.0,
    }));
    if query_param("profile").as_deref() == Some("1") {
        let mut s = state.borrow_mut();
        s.renderer.set_profiling(true);
        s.profile_since = now_secs();
    }
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

thread_local! {
    /// The page's state, for the exports the tweak panel calls between frames.
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

/// Run `f` on the lake, its surface effect and the renderer, when there is a lake.
fn with_lake<R>(f: impl FnOnce(&mut lake::Lake, &mut FluidSurfaceEffect, &Renderer) -> R) -> Option<R> {
    STATE.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        let State { lake, volume, renderer, .. } = &mut *state;
        let lake = lake.as_mut()?;
        let surface = volume.effects.get_mut(lake.effect)?.as_any_mut().downcast_mut::<FluidSurfaceEffect>()?;
        Some(f(lake, surface, renderer))
    })
}

fn surface_js(s: lake::SurfaceSettings) -> JsValue {
    let o = js_sys::Object::new();
    let set = |k: &str, v: JsValue| {
        let _ = js_sys::Reflect::set(&o, &k.into(), &v);
    };
    set("surfaceField", s.surface_field.into());
    set("resolution", s.resolution.into());
    set("kernel", s.kernel.into());
    set("particleRadius", s.particle_radius.into());
    set("iso", s.iso.into());
    set("interpolate", s.interpolate.into());
    o.into()
}

/// The lake's current settings, for the tweak panel: the simulation's and the surface's.
#[wasm_bindgen]
pub fn lake_settings() -> JsValue {
    with_lake(|lake, surface, _| {
        let p = &surface.sim.params;
        let o: js_sys::Object = surface_js(lake.surface_settings()).into();
        for (k, v) in [
            ("viscosity", p.viscosity),
            ("negativePressure", p.negative_pressure_scale),
            ("pressure", p.pressure_multiplier),
            ("nearPressure", p.near_pressure_multiplier),
            ("restDensity", p.density_target),
            ("substeps", p.substeps as f32),
            ("timeScale", lake.time_scale),
            ("drag", lake.drag()),
            ("splash", lake.splash_push),
            ("friction", lake.friction()),
        ] {
            let _ = js_sys::Reflect::set(&o, &k.into(), &v.into());
        }
        o.into()
    })
    .unwrap_or(JsValue::NULL)
}

/// Set one of the lake's simulation settings by name (see `lake::Lake::set`).
#[wasm_bindgen]
pub fn lake_set(key: &str, value: f32) -> bool {
    with_lake(|lake, surface, _| lake.set(surface, key, value)).unwrap_or(false)
}

/// Put the lake's water back as it started.
#[wasm_bindgen]
pub fn lake_reset() {
    with_lake(|lake, surface, _| lake.reset(surface));
}

/// Extract the lake's surface with these settings.
#[wasm_bindgen]
pub fn lake_surface(surface_field: bool, resolution: u32, kernel: f32, particle_radius: f32, iso: f32, interpolate: bool) {
    let settings = lake::SurfaceSettings { surface_field, resolution: resolution.clamp(32, 384), kernel: kernel.max(0.5), particle_radius, iso, interpolate };
    with_lake(|lake, surface, renderer| lake.set_surface(renderer, surface, settings));
}

/// Apply a surface preset ("droplets", "smooth" or "performance") and return its settings.
#[wasm_bindgen]
pub fn lake_surface_preset(name: &str) -> JsValue {
    let settings = match name {
        "smooth" => lake::SurfaceSettings::SMOOTH,
        "performance" => lake::SurfaceSettings::PERFORMANCE,
        _ => lake::SurfaceSettings::DROPLETS,
    };
    with_lake(|lake, surface, renderer| lake.set_surface(renderer, surface, settings));
    surface_js(settings)
}

/// Debugging the lake: its particles by region, read back from the GPU ("lake n (y) · bank ·
/// wall band · outside").
#[wasm_bindgen]
pub async fn lake_regions() -> JsValue {
    let Some((buffer, device, queue)) = with_lake(|_, surface, renderer| (surface.sim.positions_buffer().cloned(), renderer.device().clone(), renderer.queue().clone())) else { return JsValue::NULL };
    let Some(buffer) = buffer else { return JsValue::NULL };
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Lake/Readback"), size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    let (tx, rx) = (Rc::new(RefCell::new(None::<js_sys::Function>)), ());
    let _ = rx;
    let promise = {
        let tx = tx.clone();
        js_sys::Promise::new(&mut move |resolve, _| *tx.borrow_mut() = Some(resolve))
    };
    staging.slice(..).map_async(wgpu::MapMode::Read, move |_| {
        if let Some(resolve) = tx.borrow_mut().take() {
            let _ = resolve.call0(&JsValue::NULL);
        }
    });
    let _ = wasm_bindgen_futures::JsFuture::from(promise).await;
    let positions: Vec<f32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    with_lake(|lake, _, _| {
        let r = lake.regions(&positions);
        JsValue::from_str(&format!("lake {} ({:.3}) · bank {} ({:.3}) · wall band {} ({:.3}) · outside {} ({:.3})", r[0].0, r[0].1, r[1].0, r[1].1, r[2].0, r[2].1, r[3].0, r[3].1))
    })
    .unwrap_or(JsValue::NULL)
}
