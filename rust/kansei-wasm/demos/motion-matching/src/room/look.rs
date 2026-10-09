//! The character in the room: its bodies drawn into the GBuffer (so the global illumination and
//! the reflections see their normal and albedo) lit by the room's spot lights and their shadows,
//! a stand-in when there is no pack, and the capsules that stand for it in the ray tracing grid.

use glam::{Mat4, Quat, Vec3 as GVec3};

use kansei_core::animation::{skin_buffer, skinned_material, BonePalette, SkinTextures, SkinnedMesh, PALETTE_BINDING, SKINNING_WGSL, SKIN_BINDING};
use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::geometries::{CylinderGeometry, Geometry, SphereGeometry};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::rt::RtSurface;

use crate::character::{BodyLook, Character};

/// The bodies' uniform: albedo (or the tint of a textured one's colour) and the light the room
/// gives every surface where the GI is off (cd/m², a little; the GI adds the real bounce); its w is
/// 1 when `RtShadowsEffect` adds the direct light (ray-traced shadows), 0 when the material does
/// (the shadow maps).
const SURFACE_WGSL: &str = r#"
struct RoomSkin { base_color: vec4f, ambient: vec4f };
@group(0) @binding(0) var<uniform> surface: RoomSkin;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @builtin(vertex_index) vertex: u32, @location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f };
struct VOut {
    @builtin(position) @invariant clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) curr: vec4f,
    @location(4) prev: vec4f,
};
// the GBuffer's four targets (as materials::GBUFFER_OUT_WGSL writes them) and the motion
struct FOut {
    @location(0) color: vec4f,
    @location(1) emissive: vec4f,
    @location(2) normal: vec4f,
    @location(3) albedo: vec4f,
    @location(4) velocity: vec2f,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let s = kansei_skin(v.vertex, v.position.xyz, v.normal);
    let world = mesh.world * vec4f(s.position, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(s.normal, 0.0)).xyz;
    out.uv = v.uv;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(s.prev_position, 1.0));
    return out;
}

// lit by the sun and the spot lights (their shadow maps) and the ambient, into the GBuffer; or,
// deferred, the ambient alone, its roughness marked for RtShadowsEffect (rt_shadows_gbuffer.wgsl)
fn shade(in: VOut, n: vec3f, albedo: vec3f, roughness: f32, metallic: f32, occlusion: f32) -> FOut {
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(eye - in.world);
    let deferred = surface.ambient.w > 0.5;
    var radiance = albedo * (1.0 - metallic) * surface.ambient.rgb * occlusion;
    if (!deferred) {
        radiance += pbr_direct(in.world, n, v, albedo, roughness, metallic, in.clip.xy);
    }
    var out: FOut;
    out.color = vec4f(radiance, 1.0);
    out.emissive = vec4f(0.0, 0.0, 0.0, select(0.0, 0.05 + 0.4 * clamp(roughness, 0.0, 1.0), deferred));
    out.normal = vec4f(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4f(albedo * (1.0 - metallic) * occlusion, 1.0);
    out.velocity = kansei_motion_vector(in.curr, in.prev);
    return out;
}
"#;

const PLAIN_WGSL: &str = r#"
@fragment
fn fragment_main(in: VOut) -> FOut {
    return shade(in, normalize(in.normal), surface.base_color.rgb, 0.6, 0.0, 1.0);
}
"#;

/// A character pack's colour, normal (+Y up, no stored tangents) and occlusion/roughness/metallic
/// maps, as `animation::SKINNED_LIT_TEXTURED_WGSL` reads them.
const TEXTURED_WGSL: &str = r#"
@group(0) @binding(3) var base_color_texture: texture_2d<f32>;
@group(0) @binding(4) var normal_texture: texture_2d<f32>;
@group(0) @binding(5) var orm_texture: texture_2d<f32>;
@group(0) @binding(6) var texture_sampler: sampler;

fn mapped_normal(n: vec3f, p: vec3f, uv: vec2f, m: vec3f) -> vec3f {
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
    let albedo = textureSample(base_color_texture, texture_sampler, in.uv).rgb * surface.base_color.rgb;
    let orm = textureSample(orm_texture, texture_sampler, in.uv).rgb;
    let m = textureSample(normal_texture, texture_sampler, in.uv).xyz * 2.0 - 1.0;
    n = mapped_normal(n, in.world, in.uv, m);
    return shade(in, n, albedo, clamp(orm.g, 0.08, 1.0), orm.b, orm.r);
}
"#;

/// The ambient the bodies' uniform carries (cd/m²): a faint fill where nothing else lights them.
const AMBIENT: [f32; 3] = [0.6, 0.6, 0.65];

fn shader(body: &str) -> String {
    format!("{SKINNING_WGSL}\n{MOTION_VECTORS_WGSL}\n{}\n{SURFACE_WGSL}\n{body}", super::pbr::lit_wgsl())
}

fn options() -> MaterialOptions {
    MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() }
}

/// The room's look for the character's bodies: `deferred` leaves their direct light to
/// `RtShadowsEffect`.
pub struct RoomLit {
    pub deferred: bool,
}

impl RoomLit {
    fn uniform(&self, color: [f32; 4]) -> [f32; 8] {
        [color[0], color[1], color[2], 1.0, AMBIENT[0], AMBIENT[1], AMBIENT[2], self.deferred as u32 as f32]
    }
}

impl BodyLook for RoomLit {
    fn plain(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette) -> Material {
        let uniform = self.uniform(color);
        skinned_material(label, &shader(PLAIN_WGSL), &uniform, mesh, palette, options())
    }

    fn textured(&self, label: &str, color: [f32; 4], mesh: &SkinnedMesh, palette: &BonePalette, textures: SkinTextures) -> Material {
        let mut material = Material::new(
            label,
            &shader(TEXTURED_WGSL),
            vec![
                Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
                Binding::storage(PALETTE_BINDING, ShaderStages::VERTEX, true),
                Binding::storage(SKIN_BINDING, ShaderStages::VERTEX, true),
                Binding::texture_2d(3, ShaderStages::FRAGMENT),
                Binding::texture_2d(4, ShaderStages::FRAGMENT),
                Binding::texture_2d(5, ShaderStages::FRAGMENT),
                Binding::sampler(6, ShaderStages::FRAGMENT),
            ],
            options(),
        );
        let uniform = self.uniform(color);
        material.set_uniform_bindable(0, &format!("{label}/Params"), &uniform);
        material.set_bindable(PALETTE_BINDING, palette.buffer(&format!("{label}/Palette")));
        material.set_bindable(SKIN_BINDING, skin_buffer(&format!("{label}/Skin"), mesh));
        material.set_bindable(3, textures.base_color);
        material.set_bindable(4, textures.normal);
        material.set_bindable(5, textures.orm);
        material.set_bindable(6, kansei_core::buffers::Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
        material
    }

    /// The room is exposed about 10 stops brighter than the lake's daylight.
    fn overlay_scale(&self) -> f32 {
        1.0 / 1024.0
    }
}

/// The colour of the stand-in and of the capsules that stand for the character in the grid.
pub const STAND_IN_COLOR: [f32; 3] = [0.55, 0.5, 0.45];

/// Without a pack: a capsule 1.8 m tall where the character would stand, drawn by the same
/// skinned material (one joint, at rest) and in the grid as it is, so it shows in the mirrors.
pub fn stand_in(scene: &mut Scene, at: GVec3, deferred: bool) -> usize {
    let (radius, height) = (0.28, 1.8);
    // a sphere stretched into a capsule: its upper and lower halves moved apart
    let sphere = SphereGeometry::new(radius, 24, 16);
    let mut vertices = sphere.vertices.clone();
    for v in &mut vertices {
        let up = v.position[1] >= 0.0;
        v.position[1] += if up { height - radius } else { radius };
    }
    let mesh = SkinnedMesh {
        name: "StandIn".into(),
        joints: vec![[0; 4]; vertices.len()],
        weights: vec![[1.0, 0.0, 0.0, 0.0]; vertices.len()],
        vertices,
        indices: sphere.indices.clone(),
        skin_joints: vec![0],
        inverse_bind: vec![Mat4::IDENTITY],
        material: None,
    };
    let palette = BonePalette::new(1);
    let color = STAND_IN_COLOR;
    let mut r = Renderable::new(mesh.geometry(), RoomLit { deferred }.plain("StandIn", [color[0], color[1], color[2], 1.0], &mesh, &palette));
    r.object.set_position(at.x, at.y, at.z);
    r.rt = Some(RtSurface::new(color).with_smooth_normals());
    scene.add(SceneNode::Renderable(r))
}

/// Never drawn: its vertices land outside the clip volume, so it costs a draw and nothing more,
/// in every pass (its shadows are the body's). For renderables that exist only for the grid.
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

/// How thick each bone's capsule is, by the joint's name (Unreal's and most rigs' names): the
/// trunk and the head thick, the limbs thinner, the fingers and toes left out.
fn bone_radius(name: &str) -> Option<f32> {
    let n = name.to_ascii_lowercase();
    let has = |parts: &[&str]| parts.iter().any(|p| n.contains(p));
    if has(&["finger", "thumb", "index", "middle", "ring", "pinky", "toe", "ball", "twist", "ik_", "_ik", "eye", "jaw", "ear"]) {
        return None;
    }
    Some(if has(&["head"]) {
        0.11
    } else if has(&["neck"]) {
        0.06
    } else if has(&["spine", "chest", "pelvis", "hips", "torso", "abdomen"]) {
        0.14
    } else if has(&["clavicle", "shoulder"]) {
        0.06
    } else if has(&["thigh", "upleg", "upperleg"]) {
        0.085
    } else if has(&["calf", "shin", "leg", "knee"]) {
        0.06
    } else if has(&["upperarm", "uparm", "arm"]) {
        0.05
    } else if has(&["forearm", "lowerarm"]) {
        0.045
    } else if has(&["hand", "wrist"]) {
        0.04
    } else {
        // the feet and anything else
        0.05
    })
}

/// The character in the ray tracing grid: a capsule along each of its bones (a cylinder, scaled),
/// moved with the pose. They are in the grid only (see `HIDDEN_WGSL`), so the mirrors and the GI's
/// rays see the character's shape; the camera and the shadow maps see its real mesh.
///
/// Moving them rebuilds the grid (about 2 ms of GPU at 1080p), so they are static renderables
/// (the grid rebuilds only when one changes, not every frame as for a `dynamic` one) and move only
/// once a bone has moved `MOVE` from where they stand: a character standing still costs nothing.
pub struct Proxy {
    /// (joint, radius, renderable) for each bone that has a capsule.
    capsules: Vec<(usize, f32, usize)>,
    /// The bones' ends where the capsules were last put, and whether they were shown.
    placed: Vec<(GVec3, GVec3)>,
    shown: bool,
}

/// How far a bone moves (m) before the capsules follow.
const MOVE: f32 = 0.03;

impl Proxy {
    pub fn new(scene: &mut Scene, character: &Character) -> Self {
        // one unit capsule shape: a cylinder of radius 1 and length 1 along y, from 0 to 1, with
        // half spheres at its ends (scaled by the bone's length, they flatten: fine at this size)
        let mut capsules = Vec::new();
        for (joint, _, _) in character.bones_world() {
            let Some(radius) = bone_radius(character.joint_name(joint)) else { continue };
            let mut material = Material::new("Room/BodyProxy", HIDDEN_WGSL, Vec::new(), MaterialOptions::default());
            material.options.cull_mode = kansei_core::materials::CullMode::None;
            let mut r = Renderable::new(unit_capsule(), material);
            r.cast_shadow = false;
            r.rt = Some(RtSurface::new(STAND_IN_COLOR).with_smooth_normals());
            capsules.push((joint, radius, scene.add(SceneNode::Renderable(r))));
        }
        log::info!("room: {} capsules stand for the character in the grid", capsules.len());
        Self { capsules, placed: Vec::new(), shown: false }
    }

    /// Put the capsules on the character's bones as posed now, once one has moved `MOVE` (hidden
    /// with the body: `visible`).
    pub fn update(&mut self, scene: &mut Scene, character: &Character, visible: bool) {
        let bones = character.bones_world();
        let ends: Vec<(GVec3, GVec3)> = self.capsules.iter().map(|&(joint, _, _)| bones.iter().find(|(j, _, _)| *j == joint).map_or((GVec3::ZERO, GVec3::ZERO), |&(_, a, b)| (a, b))).collect();
        let moved = self.placed.len() != ends.len() || ends.iter().zip(&self.placed).any(|((a, b), (pa, pb))| a.distance(*pa) > MOVE || b.distance(*pb) > MOVE);
        if !moved && visible == self.shown {
            return;
        }
        self.placed = ends.clone();
        self.shown = visible;
        for (&(_, radius, index), &(a, b)) in self.capsules.iter().zip(&ends) {
            let Some(r) = scene.get_renderable_mut(index) else { continue };
            let along = b - a;
            let length = along.length();
            r.visible = visible && length > 0.02;
            if !r.visible {
                continue;
            }
            // (Object3D turns by z * y * x)
            let (z, y, x) = Quat::from_rotation_arc(GVec3::Y, along / length).to_euler(glam::EulerRot::ZYX);
            r.object.set_position(a.x, a.y, a.z);
            r.object.rotation = kansei_core::math::Vec3::new(x, y, z);
            r.object.scale = kansei_core::math::Vec3::new(radius, length, radius);
        }
    }
}

/// A cylinder of radius 1 from y = 0 to 1, capped (the capsules' shape before scaling).
fn unit_capsule() -> Geometry {
    let mut g = CylinderGeometry::new(1.0, 1.0, 1.0, 10, 1);
    for v in &mut g.vertices {
        v.position[1] += 0.5;
    }
    g
}
