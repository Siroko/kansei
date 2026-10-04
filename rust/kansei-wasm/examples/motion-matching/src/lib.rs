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
//! A small lake lies east of the course: SPH water in a container shaped like the lake, which the
//! character wades into, its legs pushing the water (wakes, splashes and ripples). A water cannon
//! on its west bank pours more water in when the character stands by it and presses E (X on a
//! gamepad, or a click on its prompt), raising the lake's level; R (Y) by it drains the lake back.
//! A water mill in the lake turns its paddles through the water. The course, the lake and its
//! props are the lake example's world (`kansei_wasm_lake::World`, with its P panel's exports);
//! this crate is the character on it.
//!
//! URL parameters: `pack=<url>`, `hero=<url>` (`hero=none`: no character pack), `gait=0`
//! (search every clip whatever the gait, instead of idle + walk or idle + run by the pack's tags),
//! `taa=0`, `walk=<m/s>`, `run=<m/s>` (forward paces; sideways and backward scale with them),
//! `course=0` (no boxes), `lake=0` (no lake), `rest=0` (the lake's water never rests: always
//! stepped and drawn), `mill=0` (the mill stands still), `profile=1` (log the renderer's GPU/CPU
//! profile every 3 s), `debug=1` (allows `lake_regions()`, a GPU readback),
//! `at=<x>,<z>,<heading in degrees>` (where the character starts; `at=14,-1,90` at the lake),
//! `drive=1` (a fixed route instead of the player, for side-by-side captures; `demo`),
//! `play=<pattern>` (the pack's clips whose names start with it, `*` any run, one after another),
//! `view=<degrees>` (the camera turned round the character from behind it).

mod demo;

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
use kansei_core::animation::{skin_buffer, skinned_lit_material, BonePalette, Skeleton, SkinnedLitParams, SkinnedMesh, PALETTE_BINDING, SKINNING_WGSL, SKIN_BINDING};
use kansei_core::buffers::{Bindable, BufferType, ComputeBuffer, Sampler};
use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::materials::BindingResource;
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingVolume;
use kansei_core::renderers::Renderer;
use kansei_core::shadows::CASCADED_SHADOWS_WGSL;
use kansei_wasm::{flag, param, param_or, Canvas};
use kansei_wasm_lake::{World, WorldInput, WorldOptions, SKY, SUN, SUN_DIR};

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

/// The page URL's parameter `name`, percent-decoded ([`kansei_wasm::param`]).
pub fn query_param(name: &str) -> Option<String> {
    param(name)
}

/// Show `text` in the page's HUD element.
fn set_hud(text: &str) {
    if let Some(hud) = web_sys::window().and_then(|w| w.document()).and_then(|d| d.get_element_by_id("hud")) {
        hud.set_text_content(Some(text));
    }
}

/// Fetch `url` as bytes; the error says what went wrong in words for the page.
pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
    kansei_wasm::fetch_bytes(url).await.map_err(|e| e.as_string().unwrap_or_else(|| format!("could not fetch {url}")))
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
        // the pack's own mesh, when it has one (a pack may ship without, for a character pack's body)
        let mut bodies = Vec::new();
        if let Some(first) = meshes.into_iter().next() {
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
            bodies.push(Body { name: "the pack's mesh", mesh, palette, index, display: None });
        }
        if let Some(pack) = hero {
            match hero_body(renderer, scene, pack, &db) {
                Ok(b) => bodies.push(b),
                Err(e) => log::warn!("character pack left out: {e}"),
            }
        }
        if bodies.is_empty() {
            return Err("the pack has no mesh, and there is no character pack to show it on".into());
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
    /// The course, the lake and its props, and their collision.
    world: World,
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
    /// Whether the character was in the air last frame, and the fastest it has fallen since
    /// (m/s); its height last frame.
    air: (bool, f32),
    last_y: f32,
    /// `profile=1`: when the profile was last logged (0: not profiling).
    profile_since: f64,
    /// `drive=1`: the scripted route; `play=<pattern>`: the clips played in turn.
    drive: Option<demo::Drive>,
    player: Option<demo::ClipPlayer>,
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
        let now = kansei_wasm::now();
        let dt = ((now - self.last) as f32).clamp(1e-4, 1.0 / 15.0);
        self.last = now;
        self.frame += 1;
        self.fps = self.fps * 0.95 + 0.05 / dt;

        // input: keyboard, then the gamepad on top
        let (mut stick, mut run, mut toggles) = ([0.0f32; 2], false, Vec::new());
        // the cannon's trigger: E, the gamepad's X (a click on the prompt adds to it in the world)
        let (mut fire_pressed, mut fire_held) = (false, false);
        {
            let mut keys = self.keys.borrow_mut();
            let held = |k: &[&str]| k.iter().any(|k| keys.held.contains(*k));
            stick[0] = held(&["d", "arrowright"]) as i32 as f32 - held(&["a", "arrowleft"]) as i32 as f32;
            stick[1] = held(&["w", "arrowup"]) as i32 as f32 - held(&["s", "arrowdown"]) as i32 as f32;
            run = held(&["shift"]);
            fire_held |= held(&["e"]);
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
            if pressed(2) && !was(2) {
                toggles.push("e".into());
            }
            if pressed(3) && !was(3) {
                toggles.push("r".into());
            }
            fire_held |= pressed(2);
            self.pad_held = buttons.iter().map(|b| b.0).collect();
        }
        let (mut traverse, mut drain) = (false, false);
        for key in toggles {
            match key.as_str() {
                " " => traverse = true,
                "e" => fire_pressed = true,
                "r" => drain = true,
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
        let step = self.drive.as_ref().map(|d| d.step(now));
        if let Some(step) = &step {
            run = step.run;
            self.strafe = step.facing.is_some();
        }
        if let Some(c) = &mut self.character {
            let paces = if run { self.speeds.1 } else { self.speeds.0 };
            // facing the way it moves, the character walks its loops forward; strafing, the pace
            // follows the direction
            let tilt = (stick[0] * stick[0] + stick[1] * stick[1]).sqrt();
            let speed = if self.strafe && tilt > 1e-3 { pace(paces, [stick[0] / tilt, stick[1] / tilt]) } else { paces[0] };
            let mut velocity = (forward * stick[1] + right * stick[0]) * speed;
            let mut facing = self.strafe.then(|| forward.x.atan2(forward.z));
            if let Some(step) = &step {
                // the route's direction, at the pace for it relative to the facing held
                velocity = step.direction.map_or(GVec3::ZERO, |d| {
                    let f = step.facing.unwrap_or(d.x.atan2(d.z));
                    let (front, side) = (GVec3::new(f.sin(), 0.0, f.cos()), GVec3::new(-f.cos(), 0.0, f.sin()));
                    d * pace(paces, [d.dot(side), d.dot(front)])
                });
                facing = step.facing;
            }
            c.controller.matcher.settings.filter.tags = if run { c.run_tags } else { c.walk_tags };
            if let Some(player) = &mut self.player {
                // clips as they are, one after another, from the start point
                if let Some(action) = player.due(&c.db, now) {
                    c.controller.matcher.teleport(player.home.0, player.home.1);
                    c.controller.matcher.start_action(&c.db, action);
                }
                c.controller.matcher.update(&c.db, &MotionInput { velocity: GVec3::ZERO, facing: None }, dt);
            } else {
                c.controller.update(&c.db, &self.world.collision, &MotionInput { velocity, facing }, dt);
            }
            if traverse {
                // an obstacle ahead: traverse it (pressed a little early, once in reach); else jump
                let _ = c.controller.request_traverse_or_jump(&c.db, &self.world.collision, 1.0);
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
        self.controls.update(&mut self.camera, dt);
        if self.world.lake.is_some() {
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
            // the props: the mill turns, the cannon fires when the character stands by it
            let at = self.character.as_ref().map(|c| c.controller.matcher.character().translation);
            let view_proj = self.camera.view_projection().to_glam();
            let State { world, scene, volume, renderer, .. } = &mut *self;
            world.update(WorldInput { legs: &legs, landing, at, fire: (fire_pressed, fire_held), drain }, dt, scene, volume, renderer, view_proj);
        }

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
                        "{:.0} fps   {} {}{}\nclip   {}\nframe  {:.0} / {}{}\nsearch {:.0}/s, switch {:.1}/s, cost {:.3}\nfeet   {} {}  (lock {})\n{}\n\n{}\n\nWASD / left stick move · Shift / B run · Space / A jump, traverse · Q / LB strafe\ndrag / right stick orbit · B overlay · K skeleton · M mesh · L foot lock · E / X fire the cannon (by it)",
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
                            "{}{}",
                            self.world.status("east of the course (at=14,-1,90), "),
                            if c.bodies.len() > 1 { format!("character: {} (C to switch)", c.bodies[c.showing].name) } else { String::new() },
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
    start_with_loader(canvas_id, |url: String| async move { fetch_bytes(&url).await }).await
}

/// `start`, with the packs' bytes from `load` instead of a plain fetch: it gets each pack's URL
/// (`pack/locomotion.kmm`, `pack/hero.kmm`, or `pack=`/`hero=`; never asked for the character
/// pack with `hero=none`) and returns the `.kmm` bytes, or why there are none. For an app that
/// stores its packs another way, e.g. encrypted.
pub async fn start_with_loader<L, F>(canvas_id: &str, load: L) -> Result<(), JsValue>
where
    L: Fn(String) -> F,
    F: std::future::Future<Output = Result<Vec<u8>, String>>,
{
    let window = web_sys::window().unwrap();
    let canvas = Canvas::find(canvas_id)?;
    let renderer = kansei_wasm_lake::renderer(&canvas).await;

    let mut scene = Scene::new();
    // the course, the lake and its props (`course=0`, `lake=0`, `rest=0`, `mill=0`, `taa=0`)
    let mut world = World::new(&renderer, &mut scene, &WorldOptions::from_url());

    // the character, from a pack outside the repository
    let url = param("pack").unwrap_or_else(|| "pack/locomotion.kmm".to_string());
    set_hud(&format!("Loading motion pack {url} …"));
    let gait = flag("gait", true);
    let motion = load(url.clone()).await.and_then(|bytes| MotionPack::from_bytes(&bytes));
    // a second body, optional
    let hero_url = param("hero").unwrap_or_else(|| "pack/hero.kmm".to_string());
    let hero = match &motion {
        Ok(_) if hero_url == "none" => {
            log::info!("no character pack: none asked for (hero=none)");
            None
        }
        Ok(_) => match load(hero_url).await.and_then(|bytes| CharacterPack::from_bytes(&bytes)) {
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
                let wanted = match param("char").as_deref() {
                    Some("mannequin") => 0,
                    _ => c.bodies.len() - 1,
                };
                c.show(&mut scene, wanted);
                if let Some(at) = param("at") {
                    let v: Vec<f32> = at.split(',').filter_map(|x| x.parse().ok()).collect();
                    if v.len() >= 2 {
                        // on whatever is there (a box top)
                        let y = world.collision.ground_height(GVec3::new(v[0], 0.0, v[1]), 50.0, 50.0, u32::MAX).unwrap_or(0.0);
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

    let volume = world.post_processing(&renderer, &mut scene);
    let mut camera = Camera::new(45.0, 0.1, 1200.0, canvas.aspect());
    camera.update_projection_matrix();
    let start = character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(start.translation.x, 0.9, start.translation.z), 4.5);
    controls.set_elevation(0.25);
    // behind the character, or turned round it by `view=<degrees>` (90: its left side)
    let view = param_or("view", 0.0f32).to_radians();
    controls.set_azimuth(std::f32::consts::PI + yaw_of(start.rotation) + view);

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

    let scaled = |paces: [f32; 3], name: &str| param(name).and_then(|v| v.parse::<f32>().ok()).map_or(paces, |forward| paces.map(|p| p * forward / paces[0]));
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
        keys,
        last: kansei_wasm::now(),
        frame: 0,
        strafe: false,
        overlay: true,
        pad_held: Vec::new(),
        speeds,
        fps: 60.0,
        searches: 0,
        switches: 0,
        counted_since: kansei_wasm::now(),
        rates: (0.0, 0.0),
        profile_since: 0.0,
        air: (false, 0.0),
        last_y: 0.0,
        drive: None,
        player: None,
    }));
    if flag("profile", false) {
        let mut s = state.borrow_mut();
        s.renderer.set_profiling(true);
        s.profile_since = kansei_wasm::now();
    }
    {
        let mut s = state.borrow_mut();
        let home = s.character.as_ref().map(|c| {
            let at = c.controller.matcher.character();
            (at.translation, yaw_of(at.rotation))
        });
        if let Some(home) = home {
            if flag("drive", false) {
                s.drive = Some(demo::Drive { start: kansei_wasm::now(), home });
            }
            if let Some(prefix) = param("play") {
                let player = s.character.as_ref().map(|c| demo::ClipPlayer::new(&c.db, &prefix, home));
                s.player = player;
            }
        }
    }
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    // the lake's panel and the cannon's prompt act on this world
    kansei_wasm_lake::register(state.clone());
    kansei_wasm::run(&canvas, move |frame| {
        let mut s = state.borrow_mut();
        let State { renderer, camera, .. } = &mut *s;
        frame.resize(renderer, camera);
        s.frame();
    });
    Ok(())
}

impl kansei_wasm_lake::Host for State {
    fn parts(&mut self) -> (&mut World, &mut PostProcessingVolume, &Renderer) {
        (&mut self.world, &mut self.volume, &self.renderer)
    }
}

thread_local! {
    /// The page's state, for the exports the clip tools call between frames.
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

/// Run `f` on the page's state, once it has started.
fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    STATE.with(|s| {
        let state = s.borrow().clone()?;
        let mut state = state.borrow_mut();
        Some(f(&mut state))
    })
}

/// Start the `drive=1` route (or the `play=` clips) over, the character back where it started:
/// to line up recordings of different packs.
#[wasm_bindgen]
pub fn drive_restart() {
    with_state(|s| {
        if let (Some(d), Some(c)) = (&mut s.drive, &mut s.character) {
            d.start = kansei_wasm::now();
            c.controller.matcher.teleport(d.home.0, d.home.1);
        }
        if let Some(p) = &mut s.player {
            p.restart();
        }
    });
}

/// The motion pack's clip names, in its order (empty before it has loaded): for a page's clip
/// browser.
#[wasm_bindgen]
pub fn clip_names() -> Vec<String> {
    with_state(|s| s.character.as_ref().map(|c| c.db.clips.iter().map(|c| c.name.clone()).collect())).flatten().unwrap_or_default()
}

/// Play the clips whose names start with `pattern` one after another, as `play=` does, from where
/// the character stands now; `""` hands it back to the player (the clip playing finishes first).
/// Returns how many clips match.
#[wasm_bindgen]
pub fn play_clips(pattern: &str) -> usize {
    with_state(|s| {
        if pattern.is_empty() {
            s.player = None;
            return 0;
        }
        let Some(c) = &s.character else { return 0 };
        let at = c.controller.matcher.character();
        let player = demo::ClipPlayer::new(&c.db, pattern, (at.translation, yaw_of(at.rotation)));
        let n = player.clips.len();
        s.player = (n > 0).then_some(player);
        n
    })
    .unwrap_or(0)
}

/// Drive the `drive=1` route from where the character stands now, or hand it back to the player.
#[wasm_bindgen]
pub fn set_drive(on: bool) {
    with_state(|s| {
        s.drive = on.then(|| {
            let at = s.character.as_ref().map(|c| c.controller.matcher.character()).unwrap_or_default();
            demo::Drive { start: kansei_wasm::now(), home: (at.translation, yaw_of(at.rotation)) }
        });
        if !on {
            s.strafe = false;
        }
    });
}
