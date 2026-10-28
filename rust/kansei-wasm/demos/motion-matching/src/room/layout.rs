//! The room: a loft gallery authored in Blender and exported as `www/assets/room/room.glb`
//! (`tools/room_glb.py`; the .blend lives outside the repository, see the README's "The room").
//! Everything solid is in the collision world, voxel GI and the ray tracing grid.
//!
//! The glb is read by the names and custom properties (glTF extras) the README's contract lists:
//! meshes are merged by material (and by how they are drawn) into one renderable each, so the
//! room is a draw per material; `<name>_COL` boxes are the collision world, `<name>_SHADOW` meshes
//! what their parent casts into the shadow maps, `_CUT` meshes Blender's boolean cutters (skipped);
//! `kansei` names a role (`shell`, `mirror`, `glass`, `aquarium`, `caustics`, `dragon`, `spawn`,
//! `preview`); lights carry `kansei_light` (`sun`, `skylight`, `lamp`,
//! `display`); cameras named `CAM_<view>` are the page's fixed views.
//!
//! One-sided: the shell (`kansei: shell`: walls, floor, ceiling) is drawn with back faces discarded
//! on screen, so from outside (the orbit camera past a wall or above the ceiling) the camera looks
//! through it into the room, while the shadow maps draw both sides (the sun outside casts the
//! room's shadow into it). The grid's rays hit a triangle from either side, so the GI's and the
//! mirrors' rays meet the walls as they should.

use std::collections::HashMap;

use glam::{Mat3, Mat4, Vec3 as GVec3};

use kansei_core::animation::motion_matching::traversal::features::ParkourPiece;
use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::geometries::{Geometry, Vertex};
use kansei_core::gi::GiSurface;
use kansei_core::lights::{DirectionalLight, Light, SpotLight};
use kansei_core::loaders::{GLTFLoader, GLTFResult};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;
use kansei_core::rt::RtSurface;
use serde_json::Value;

use super::assets::{Shared, SharedMaps};
use super::pbr::{self, PbrParams};
use super::aquarium::Tank;

/// Default light levels: the sun's illuminance (lux), the skylight's and a lamp's intensity (cd),
/// a shade's radiance (cd/m²), the display spot's intensity (cd).
pub const SUN_LUX: f32 = 9000.0;
pub const SKYLIGHT_CD: f32 = 32000.0;
pub const LAMP_CD: f32 = 700.0;
pub const SHADE_RADIANCE: f32 = 900.0;
pub const DISPLAY_CD: f32 = 1_500_000.0;
const SUN_COLOR: [f32; 3] = [1.0, 0.74, 0.5];
const LAMP_COLOR: [f32; 3] = [1.0, 0.68, 0.4];
const DISPLAY_COLOR: [f32; 3] = [1.0, 0.86, 0.72];
/// cd/m² for a unit of Blender's emission strength (its shades' 6 are `SHADE_RADIANCE`).
const EMISSION_SCALE: f32 = 150.0;
const MIRROR: [f32; 3] = [0.92, 0.92, 0.92];
const MIRROR_GI: [f32; 3] = [0.03, 0.03, 0.03];

/// The lights as built (scene indices): the sun, the skylight's area light (the first spot light:
/// `RtShadowsEffect::set_rect_emitter(0, ..)`) and its rectangle, the lamps, the glowing shades,
/// the display spot over the dragon (the last spot light).
pub struct Lights {
    pub sun: usize,
    pub sun_direction: GVec3,
    pub skylight: usize,
    /// The skylight's emitter: its extent along z and along x (m), as `set_rect_emitter` takes it
    /// for a light pointing down.
    pub skylight_rect: [f32; 2],
    pub lamps: Vec<usize>,
    pub shades: Vec<usize>,
    pub display: usize,
}

/// A PBR surface in the room, for the page to change (deferred lighting, the floor's roughness).
pub struct Surface {
    pub index: usize,
    pub params: PbrParams,
    /// The floor's (its roughness follows `floor_rough`), and its roughness factor as authored.
    pub floor: bool,
    pub roughness: f32,
}

/// A fixed view (`cam=<name>`): the orbit camera's target, distance, azimuth and elevation.
pub struct View {
    pub name: String,
    pub target: GVec3,
    pub distance: f32,
    pub azimuth: f32,
    pub elevation: f32,
}

/// A piece meant for parkour (`kansei_traversal`): the traversal expected walking at it from the
/// south, from feet `from` metres in front of it at height `from_y`, and all its properties (the
/// course's `wallrun`, `bar`, `ledge` and `platform` carry what the controller to come needs).
pub struct Traversal {
    pub name: String,
    pub min: GVec3,
    pub max: GVec3,
    pub from: f32,
    pub from_y: f32,
    pub want: String,
    pub extras: String,
}

pub struct Layout {
    /// The floor, the walls and the furniture, for the character.
    pub collision: CollisionWorld,
    pub surfaces: Vec<Surface>,
    pub lights: Lights,
    /// The mirrors' renderables and their authored roughness, by name.
    pub mirrors: Vec<(usize, f32)>,
    pub views: Vec<View>,
    /// Where the character starts and its heading.
    pub spawn: (GVec3, f32),
    /// The dragon's place on the plinth, its yaw and its length.
    pub dragon: (GVec3, f32, f32),
    /// The room's inside (the shell's bounds).
    pub min: GVec3,
    pub max: GVec3,
    /// The aquarium's water (`kansei: aquarium`), and the rectangle (x, z) the caustics' map
    /// covers (`kansei: caustics`, else round the tank, where the sun throws them, and the dragon).
    pub tank: Option<Tank>,
    pub caustics: ([f32; 2], [f32; 2]),
    /// The colour maps the reflections' hits read (`assets::albedo_array`, a layer each) and the
    /// tint each is drawn with.
    pub albedo_layers: Vec<(wgpu::TextureView, [f32; 3])>,
    pub traversals: Vec<Traversal>,
    /// Triangles drawn, by role, for the log and the README.
    pub triangles: Vec<(String, usize)>,
}

thread_local! {
    /// The flat fill surfaces get with no GI (`render=forward`), cd/m² per unit albedo.
    static FILL: std::cell::Cell<[f32; 3]> = const { std::cell::Cell::new([0.0; 3]) };
}

/// Give the surfaces made from now on a flat fill in place of the GI (`render=forward`).
pub fn set_fill(fill: [f32; 3]) {
    FILL.with(|f| f.set(fill));
}

/// The fill `set_fill` gave.
pub fn fill() -> [f32; 3] {
    FILL.with(|f| f.get())
}

/// How a surface is drawn: a texture `size` metres across in the world (box mapped), its
/// roughness scaled and biased, reflective (traced) or not, deferred (`RtShadowsEffect`) or not.
pub fn surface(tint: [f32; 3], size: f32, roughness: [f32; 2], reflective: bool, deferred: bool) -> PbrParams {
    PbrParams {
        tint: [tint[0], tint[1], tint[2], 1.0],
        surface: [size, roughness[0], roughness[1], reflective as u32 as f32],
        ambient: { let f = fill(); [f[0], f[1], f[2], deferred as u32 as f32] },
        emissive: [0.0, 0.0, 0.0, 1.0],
        ..Default::default()
    }
}

/// A material of the glb as the room draws it.
struct RoomMaterial {
    name: String,
    maps: SharedMaps,
    params: PbrParams,
    /// Its colour map's layer in the albedo array.
    layer: Option<u8>,
    tint: [f32; 3],
    double_sided: bool,
    cutout: bool,
    shade: bool,
    emission: [f32; 3],
    roughness: f32,
}

/// A group of meshes drawn as one renderable.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Key {
    material: usize,
    shell: bool,
    self_lit: bool,
}

#[derive(Default)]
struct Group {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    /// What the members cast into the shadow maps (their `_SHADOW` meshes, else themselves), and
    /// whether any member has a `_SHADOW` mesh (else the group casts its own geometry).
    shadow: (Vec<Vertex>, Vec<u32>),
    proxies: bool,
}

fn extras(raw: &Option<Box<serde_json::value::RawValue>>) -> Value {
    raw.as_ref().and_then(|r| serde_json::from_str(r.get()).ok()).unwrap_or(Value::Null)
}

fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v.get(key).and_then(Value::as_str).unwrap_or("")
}

fn number(v: &Value, key: &str) -> Option<f32> {
    v.get(key).and_then(Value::as_f64).map(|x| x as f32)
}

/// A primitive's triangles in the world (placed by `world`, wound counter-clockwise).
fn primitive(primitive: &gltf::Primitive, buffers: &[gltf::buffer::Data], world: Mat4, out: &mut (Vec<Vertex>, Vec<u32>)) {
    let reader = primitive.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
    let Some(positions) = reader.read_positions() else { return };
    let positions: Vec<[f32; 3]> = positions.collect();
    let normals: Vec<[f32; 3]> = reader.read_normals().map(|n| n.collect()).unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; positions.len()]);
    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_else(|| vec![[0.0; 2]; positions.len()]);
    let indices: Vec<u32> = reader.read_indices().map(|i| i.into_u32().collect()).unwrap_or_else(|| (0..positions.len() as u32).collect());
    let normal_matrix = Mat3::from_mat4(world).inverse().transpose();
    let flip = Mat3::from_mat4(world).determinant() < 0.0;
    let base = out.0.len() as u32;
    for k in 0..positions.len() {
        let p = world.transform_point3(GVec3::from(positions[k]));
        let n = (normal_matrix * GVec3::from(normals[k])).normalize_or_zero();
        out.0.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: uvs[k] });
    }
    for t in indices.chunks_exact(3) {
        if flip {
            out.1.extend_from_slice(&[base + t[0], base + t[2], base + t[1]]);
        } else {
            out.1.extend_from_slice(&[base + t[0], base + t[1], base + t[2]]);
        }
    }
}

/// A mesh's bounds in its own space.
fn local_bounds(mesh: &gltf::Mesh, buffers: &[gltf::buffer::Data]) -> (GVec3, GVec3) {
    let (mut lo, mut hi) = (GVec3::splat(f32::MAX), GVec3::splat(f32::MIN));
    for p in mesh.primitives() {
        let reader = p.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
        for q in reader.read_positions().into_iter().flatten() {
            lo = lo.min(GVec3::from(q));
            hi = hi.max(GVec3::from(q));
        }
    }
    (lo, hi)
}

/// `local`'s box placed by `world`, as world bounds.
fn world_bounds(world: Mat4, (lo, hi): (GVec3, GVec3)) -> (GVec3, GVec3) {
    kansei_core::rt::transform_box(&world, lo, hi)
}

/// The materials, their maps uploaded once each.
fn materials(renderer: &Renderer, doc: &gltf::Document, result: &GLTFResult, deferred: bool, layers: &mut Vec<(wgpu::TextureView, [f32; 3])>) -> Vec<RoomMaterial> {
    let support = renderer.compression_support();
    let mut uploaded: HashMap<usize, (wgpu::Texture, wgpu::TextureView)> = HashMap::new();
    let mut texture = |t: Option<&kansei_core::loaders::GLTFTextureRef>, flat: [u8; 4], label: &str| -> Shared {
        let Some(t) = t else { return Shared::upload(renderer, kansei_core::buffers::Texture::from_rgba(label, 1, 1, &flat)) };
        if let Some((tex, view)) = uploaded.get(&t.image) {
            return Shared::from_parts(tex.clone(), view.clone());
        }
        match result.load_texture(t, support) {
            Ok(x) => {
                let s = Shared::upload(renderer, x.into_texture());
                uploaded.insert(t.image, s.parts());
                s
            }
            Err(e) => {
                log::warn!("room: {label}: {e}");
                Shared::upload(renderer, kansei_core::buffers::Texture::from_rgba(label, 1, 1, &flat))
            }
        }
    };
    doc.materials()
        .zip(&result.materials)
        .map(|(m, info)| {
            let x = extras(m.extras());
            let color = texture(info.base_color_texture.as_ref(), [255; 4], &info.name);
            let normal = texture(info.normal_texture.as_ref(), [128, 128, 255, 255], &info.name);
            let textured_orm = info.metallic_roughness_texture.is_some();
            let orm = texture(info.metallic_roughness_texture.as_ref().or(info.occlusion_texture.as_ref()), [255, (info.roughness * 255.0) as u8, (info.metallic * 255.0) as u8, 255], &info.name);
            let tint = [info.base_color[0], info.base_color[1], info.base_color[2]];
            let cutout = m.alpha_mode() != gltf::material::AlphaMode::Opaque;
            let emissive = m.emissive_factor();
            let strength = m.emissive_strength().unwrap_or(1.0);
            let emission = emissive.map(|c| c * strength * EMISSION_SCALE);
            let roughness = if textured_orm { info.roughness } else { 1.0 };
            let reflective = x.get("kansei_reflective").and_then(Value::as_f64).unwrap_or(0.0) > 0.5;
            let params = PbrParams {
                tint: [tint[0], tint[1], tint[2], 0.0],
                surface: [1.0, roughness, 0.0, reflective as u32 as f32],
                pattern: [0.0, 1.0, m.normal_texture().map_or(1.0, |n| n.scale()), if cutout { m.alpha_cutoff().unwrap_or(0.5) } else { 0.0 }],
                ambient: { let f = fill(); [f[0], f[1], f[2], deferred as u32 as f32] },
                emissive: [emission[0], emission[1], emission[2], number(&x, "kansei_macro").unwrap_or(0.0)],
                flags: [0.0, 1.0, 0.0, 0.0],
            };
            let layer = info.base_color_texture.as_ref().map(|_| {
                layers.push((color.view().clone(), tint));
                (layers.len() - 1) as u8
            });
            RoomMaterial {
                name: info.name.clone(),
                maps: SharedMaps { color, normal, orm },
                params,
                layer,
                tint,
                double_sided: m.double_sided() || cutout,
                cutout,
                shade: x.get("kansei_shade").and_then(Value::as_f64).unwrap_or(0.0) > 0.5,
                emission,
                roughness,
            }
        })
        .collect()
}

/// Fetch `assets/room/room.glb` and build the room into `scene`: `deferred` leaves the direct
/// light to `RtShadowsEffect`.
pub async fn load(renderer: &Renderer, scene: &mut Scene, deferred: bool) -> Result<Layout, String> {
    let bytes = crate::fetch_bytes("assets/room/room.glb").await?;
    let gltf = gltf::Gltf::from_slice(&bytes).map_err(|e| format!("room.glb: {e}"))?;
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob.clone()).map_err(|e| format!("room.glb buffers: {e}"))?;
    let result = GLTFLoader::load_glb(&bytes)?;
    let doc = &gltf.document;
    let mut albedo_layers = Vec::new();
    let mats = materials(renderer, doc, &result, deferred, &mut albedo_layers);

    // every node, placed in the world
    let mut placed: Vec<(gltf::Node, Mat4, Option<usize>)> = Vec::new();
    fn walk<'a>(node: gltf::Node<'a>, parent: Mat4, parent_index: Option<usize>, out: &mut Vec<(gltf::Node<'a>, Mat4, Option<usize>)>) {
        let world = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        let me = node.index();
        for child in node.children() {
            walk(child, world, Some(me), out);
        }
        out.push((node, world, parent_index));
    }
    let root = doc.default_scene().or_else(|| doc.scenes().next()).ok_or("room.glb: no scene")?;
    for node in root.nodes() {
        walk(node, Mat4::IDENTITY, None, &mut placed);
    }

    let mut collision = CollisionWorld::new();
    let mut groups: HashMap<Key, Group> = HashMap::new();
    let mut shadows: HashMap<usize, (Vec<Vertex>, Vec<u32>)> = HashMap::new();
    // the same `_SHADOW` meshes by parent and material, and what the ray tracing grid traces of
    // each material: a member's `_SHADOW` mesh where it has one (the grid's cost is triangles a
    // cell: the furniture's full meshes cost the rays half again), else its mesh
    let mut proxy_parts: HashMap<(usize, usize), (Vec<Vertex>, Vec<u32>)> = HashMap::new();
    let mut grid: HashMap<(usize, bool), (Vec<Vertex>, Vec<u32>)> = HashMap::new();
    let mut mirrors: Vec<(String, Geometry, f32)> = Vec::new();
    let mut glass: Vec<(String, Geometry, [f32; 3])> = Vec::new();
    let mut views = Vec::new();
    let mut traversals = Vec::new();
    let mut spawn = (GVec3::new(0.0, 0.0, 11.0), std::f32::consts::PI);
    let mut dragon = (GVec3::new(0.0, 0.5, 0.0), 0.7, 2.6);
    let mut tank = None;
    let mut colliders = 0;
    let mut caustics = None;
    let (mut shell_lo, mut shell_hi) = (GVec3::splat(f32::MAX), GVec3::splat(f32::MIN));
    let (mut sun, mut skylight, mut lamps, mut display) = (None, None, Vec::new(), None);
    let mut triangles: HashMap<String, usize> = HashMap::new();

    // the `_SHADOW` meshes first, keyed by their parent
    for (node, world, parent) in &placed {
        let name = node.name().unwrap_or("");
        if let (Some(mesh), Some(parent), true) = (node.mesh(), parent, name.ends_with("_SHADOW")) {
            let out = shadows.entry(*parent).or_default();
            for p in mesh.primitives() {
                primitive(&p, &buffers, *world, out);
                let material = p.material().index().unwrap_or(0).min(mats.len().saturating_sub(1));
                primitive(&p, &buffers, *world, proxy_parts.entry((*parent, material)).or_default());
            }
        }
    }
    for (node, world, parent) in &placed {
        let name = node.name().unwrap_or("").to_string();
        let x = extras(node.extras());
        let role = text(&x, "kansei").to_string();
        let (_, rotation, translation) = world.to_scale_rotation_translation();
        if name.ends_with("_CUT") || name.ends_with("_SHADOW") || role == "preview" {
            continue;
        }
        if node.camera().is_some() {
            let forward = rotation * -GVec3::Z;
            let distance = number(&x, "distance").unwrap_or(8.0).max(0.5);
            let target = translation + forward * distance;
            let d = translation - target;
            views.push(View { name: name.trim_start_matches("CAM_").to_string(), target, distance, azimuth: d.x.atan2(d.z), elevation: (d.y / distance).clamp(-1.0, 1.0).asin() });
            continue;
        }
        let light_role = text(&x, "kansei_light");
        if !light_role.is_empty() {
            let forward = (rotation * -GVec3::Z).normalize();
            let at = Vec3::new(translation.x, translation.y, translation.z);
            let dir = Vec3::new(forward.x, forward.y, forward.z);
            let cone = node.light().and_then(|l| match l.kind() {
                gltf::khr_lights_punctual::Kind::Spot { inner_cone_angle, outer_cone_angle } => Some((inner_cone_angle, outer_cone_angle)),
                _ => None,
            });
            match light_role {
                "sun" => sun = Some(forward),
                "skylight" => skylight = Some((at, dir, [number(&x, "length").unwrap_or(12.0), number(&x, "width").unwrap_or(6.0)])),
                "lamp" => lamps.push((at, dir, cone.unwrap_or((55f32.to_radians(), 84f32.to_radians())))),
                "display" => display = Some((at, dir, cone.unwrap_or((7f32.to_radians(), 11f32.to_radians())))),
                other => log::warn!("room: {name}: unknown kansei_light {other}"),
            }
            continue;
        }
        match role.as_str() {
            "spawn" => {
                let facing = rotation * GVec3::Y;
                spawn = (translation, facing.x.atan2(facing.z));
                continue;
            }
            "dragon" => {
                let facing = rotation * GVec3::Z;
                dragon = (translation, facing.x.atan2(facing.z), number(&x, "length").unwrap_or(2.6));
                continue;
            }
            _ => {}
        }
        let Some(mesh) = node.mesh() else { continue };
        if name.ends_with("_COL") {
            // a bar's tube (`<bar>_COL`) is not in the collision world: the body swinging up to it
            // must not be pushed off (its posts are)
            let tube = parent.and_then(|p| doc.nodes().nth(p)).is_some_and(|p| text(&extras(p.extras()), "kansei_traversal") == "bar" && name == format!("{}_COL", p.name().unwrap_or("")));
            if tube {
                continue;
            }
            let (lo, hi) = local_bounds(&mesh, &buffers);
            let (scale, rotation, _) = world.to_scale_rotation_translation();
            let centre = world.transform_point3((lo + hi) * 0.5);
            collision.add_box(Obb::new(centre, (hi - lo) * 0.5 * scale, rotation));
            colliders += 1;
            continue;
        }
        let bounds = world_bounds(*world, local_bounds(&mesh, &buffers));
        match role.as_str() {
            "aquarium" => {
                tank = Some(Tank { min: bounds.0, max: bounds.1, fill: number(&x, "fill").unwrap_or(0.85) });
                continue;
            }
            "caustics" => {
                caustics = Some(([bounds.0.x, bounds.0.z], [bounds.1.x, bounds.1.z]));
                continue;
            }
            "shell" => {
                shell_lo = shell_lo.min(bounds.0);
                shell_hi = shell_hi.max(bounds.1);
            }
            _ => {}
        }
        if let Some(kind) = x.get("kansei_traversal").and_then(Value::as_str) {
            traversals.push(Traversal { name: name.clone(), min: bounds.0, max: bounds.1, from: number(&x, "kansei_from").unwrap_or(1.1), from_y: number(&x, "kansei_from_y").unwrap_or(0.0), want: kind.to_string(), extras: x.to_string() });
        }
        if role == "mirror" || role == "glass" {
            let mut part = (Vec::new(), Vec::new());
            for p in mesh.primitives() {
                primitive(&p, &buffers, *world, &mut part);
            }
            let geometry = Geometry::new(&name, part.0, part.1);
            if role == "mirror" {
                mirrors.push((name, geometry, number(&x, "kansei_roughness").unwrap_or(0.0)));
            } else {
                let tint = mesh.primitives().next().and_then(|p| p.material().index()).map_or([0.95; 3], |m| mats[m].tint);
                glass.push((name, geometry, tint));
            }
            continue;
        }
        let proxy = shadows.get(&node.index());
        for p in mesh.primitives() {
            let material = p.material().index().unwrap_or(0).min(mats.len().saturating_sub(1));
            let key = Key { material, shell: role == "shell", self_lit: x.get("kansei_self_lit").and_then(Value::as_f64).unwrap_or(0.0) > 0.5 };
            let group = groups.entry(key).or_default();
            let mut part = (Vec::new(), Vec::new());
            primitive(&p, &buffers, *world, &mut part);
            let base = group.vertices.len() as u32;
            group.vertices.extend_from_slice(&part.0);
            group.indices.extend(part.1.iter().map(|i| i + base));
            match proxy {
                // (a node's proxy is cast once, by its first primitive's group)
                Some(s) if p.index() == 0 => {
                    let base = group.shadow.0.len() as u32;
                    group.shadow.0.extend_from_slice(&s.0);
                    group.shadow.1.extend(s.1.iter().map(|i| i + base));
                    group.proxies = true;
                }
                Some(_) => {}
                None => {
                    let base = group.shadow.0.len() as u32;
                    group.shadow.0.extend_from_slice(&part.0);
                    group.shadow.1.extend(part.1.iter().map(|i| i + base));
                }
            }
            let traced = grid.entry((material, key.self_lit)).or_default();
            let source = if proxy.is_some() { proxy_parts.get(&(node.index(), material)) } else { Some(&part) };
            if let Some((v, i)) = source {
                let base = traced.0.len() as u32;
                traced.0.extend_from_slice(v);
                traced.1.extend(i.iter().map(|k| k + base));
            }
            let what = if role.is_empty() { "furniture".to_string() } else { role.clone() };
            *triangles.entry(what).or_default() += part.1.len() / 3;
        }
    }
    if shell_lo.x > shell_hi.x {
        return Err("room.glb: no shell (kansei: shell)".into());
    }

    // the renderables, a draw per material and way of drawing
    let mut surfaces = Vec::new();
    let mut shades = Vec::new();
    let mut grid_triangles = 0;
    let mut keys: Vec<Key> = groups.keys().copied().collect();
    keys.sort_by_key(|k| (k.material, k.shell, k.self_lit));
    for key in keys {
        let group = groups.remove(&key).unwrap();
        let m = &mats[key.material];
        let label = format!("Room/{}", m.name);
        let geometry = Geometry::new(&label, group.vertices, group.indices);
        let albedo = m.tint.map(|c| c * 0.6);
        if m.shade {
            let mut r = Renderable::new(geometry, Material::emissive("LampShade", m.emission)).with_gi(GiSurface::new([0.8; 3]).with_emission(m.emission.map(|c| c * 0.02)));
            r.cast_shadow = false;
            shades.push(scene.add(SceneNode::Renderable(r)));
            continue;
        }
        let mut params = m.params;
        params.flags[0] = key.shell as u32 as f32;
        let material = pbr::material(&label, m.maps.maps(&label), &params, m.double_sided);
        let r = Renderable::new(geometry, material).with_gi(GiSurface::new(albedo));
        // in the grid (not cut-out leaves): its members as `grid` gathered them
        if let Some((v, i)) = grid.remove(&(key.material, key.self_lit)).filter(|_| !m.cutout) {
            let mut rt = RtSurface::new(albedo);
            if let Some(layer) = m.layer {
                rt = rt.with_albedo_layer(layer);
            }
            if key.self_lit {
                rt = rt.with_self_lit();
            }
            grid_triangles += i.len() / 3;
            scene.add(SceneNode::Renderable(super::look::grid_only(Geometry::new(&format!("{label}/Grid"), v, i), rt)));
        }
        let mut r = r;
        if group.proxies {
            r.shadow_geometry = Some(Geometry::new(&format!("{label}/Shadow"), group.shadow.0, group.shadow.1));
        }
        // (the floor casts nothing the room can see)
        let floor = m.name == "Floor";
        r.cast_shadow = !floor;
        let index = scene.add(SceneNode::Renderable(r));
        surfaces.push(Surface { index, params, floor, roughness: m.roughness });
    }
    let mut mirrors_out = Vec::new();
    mirrors.sort_by(|a, b| a.0.cmp(&b.0));
    // (the north mirror first: `mirror_rough`, then `mirror2_rough`)
    mirrors.sort_by_key(|m| !m.0.contains("North"));
    for (_, geometry, roughness) in mirrors {
        let mut r = Renderable::new(geometry, Material::standard_lit("Mirror", &StandardLitOptions::mirror(MIRROR, roughness))).with_gi(GiSurface::new(MIRROR_GI));
        r.rt = Some(RtSurface::new(MIRROR_GI));
        mirrors_out.push((scene.add(SceneNode::Renderable(r)), roughness));
    }
    for (name, geometry, tint) in glass {
        let mut r = Renderable::new(geometry, Material::standard_lit(&name, &StandardLitOptions::glass(tint, 1.5, 0.0)));
        r.rt = Some(RtSurface::glass(tint));
        r.cast_shadow = false;
        scene.add(SceneNode::Renderable(r));
    }

    // the shell in the collision world: the floor, thick walls outside
    let (lo, hi) = (shell_lo, shell_hi);
    collision.add_box(Obb::from_min_max(GVec3::new(lo.x, lo.y - 1.0, lo.z), GVec3::new(hi.x, lo.y, hi.z)));
    for (min, max) in [([lo.x - 1.0, lo.z - 1.0], [hi.x + 1.0, lo.z]), ([lo.x - 1.0, hi.z], [hi.x + 1.0, hi.z + 1.0]), ([lo.x - 1.0, lo.z], [lo.x, hi.z]), ([hi.x, lo.z], [hi.x + 1.0, hi.z])] {
        collision.add_box(Obb::from_min_max(GVec3::new(min[0], lo.y - 1.0, min[1]), GVec3::new(max[0], hi.y + 1.0, max[1])));
    }

    // the lights: the sun, the skylight (the first spot light), the lamps, the display spot (the last)
    let sun_direction = sun.unwrap_or(GVec3::new(0.86, -0.4, 0.32)).normalize();
    let d = sun_direction;
    let mut light = DirectionalLight::new(Vec3::new(d.x, d.y, d.z), Vec3::new(SUN_COLOR[0], SUN_COLOR[1], SUN_COLOR[2]), SUN_LUX);
    light.cast_shadow = true;
    let sun = scene.add(SceneNode::Light(Light::Directional(light)));
    let (at, dir, skylight_rect) = skylight.unwrap_or((Vec3::new(0.0, hi.y + 0.7, 0.0), Vec3::new(0.0, -1.0, 0.0), [12.0, 6.0]));
    let mut sky = SpotLight::new(at, dir, Vec3::new(0.78, 0.86, 1.0), SKYLIGHT_CD, 48.0, 62f32.to_radians(), 86f32.to_radians());
    sky.cast_shadow = true;
    sky.source_radius = 1.5;
    sky.volumetric_scale = 1.0;
    let skylight = scene.add(SceneNode::Light(Light::Spot(sky)));
    let lamps = lamps
        .into_iter()
        .map(|(at, dir, (inner, outer))| {
            let mut light = SpotLight::new(at, dir, Vec3::new(LAMP_COLOR[0], LAMP_COLOR[1], LAMP_COLOR[2]), LAMP_CD, 10.0, inner, outer);
            light.cast_shadow = true;
            light.source_radius = 0.12;
            light.volumetric_scale = 1.0;
            scene.add(SceneNode::Light(Light::Spot(light)))
        })
        .collect();
    let (at, dir, (inner, outer)) = display.unwrap_or((Vec3::new(0.0, hi.y - 0.6, -10.5), Vec3::new(0.0, -0.54, 0.84), (7f32.to_radians(), 11f32.to_radians())));
    let mut spot = SpotLight::new(at, dir, Vec3::new(DISPLAY_COLOR[0], DISPLAY_COLOR[1], DISPLAY_COLOR[2]), DISPLAY_CD, 40.0, inner, outer);
    spot.cast_shadow = true;
    spot.source_radius = 0.03;
    spot.volumetric_scale = 1.0;
    let display = scene.add(SceneNode::Light(Light::Spot(spot)));

    // the caustics' map: as authored, else round the tank and where the sun through it lands on
    // the floor, and round the dragon
    let caustics = caustics.unwrap_or_else(|| {
        let mut points = vec![dragon.0 - GVec3::splat(2.0), dragon.0 + GVec3::splat(2.0)];
        if let Some(t) = tank {
            for corner in [t.min, t.max, GVec3::new(t.min.x, t.max.y, t.max.z), GVec3::new(t.max.x, t.max.y, t.min.z)] {
                points.push(corner - GVec3::splat(0.8));
                points.push(corner + GVec3::splat(0.8));
                let along = corner.y / (-sun_direction.y).max(0.1);
                points.push(corner + sun_direction * along);
            }
        }
        let (a, b) = points.iter().fold((GVec3::splat(f32::MAX), GVec3::splat(f32::MIN)), |(a, b), p| (a.min(*p), b.max(*p)));
        ([a.x.max(lo.x), a.z.max(lo.z)], [b.x.min(hi.x), b.z.min(hi.z)])
    });
    triangles.insert("grid".into(), grid_triangles);
    let mut triangles: Vec<(String, usize)> = triangles.into_iter().collect();
    triangles.sort();
    log::info!("room: {} draws from {} materials, triangles {:?}, {} views, {} colliders", surfaces.len() + shades.len() + mirrors_out.len(), mats.len(), triangles, views.len(), colliders);
    Ok(Layout {
        collision,
        surfaces,
        lights: Lights { sun, sun_direction, skylight, skylight_rect, lamps, shades, display },
        mirrors: mirrors_out,
        views,
        spawn,
        dragon,
        min: lo,
        max: hi,
        tank,
        caustics,
        albedo_layers,
        traversals,
        triangles,
    })
}

/// What today's traversal does at a piece of the course tagged for a move it hasn't yet: a platform
/// is mantled onto, a ledge tower climbed; walls to run along and bars to swing from it doesn't
/// take (`None`: not checked).
fn today(kind: &str) -> Option<&str> {
    match kind {
        "platform" => Some("mantle"),
        "ledge" => Some("climb"),
        "wallrun" | "bar" => None,
        other => Some(other),
    }
}

/// The pieces meant for parkour as the character's traversal sees them: from feet in front of each
/// (south of it, `Traversal::from` away, at `from_y`), walking north at it, the probe
/// (`traversal::detect_obstacle`) finds it and picks the kind of traversal (`traversal_kind`, the
/// default rules). Each piece's name and whether it got the kind meant (for the course's new
/// kinds, what today's traversal makes of them: `today`), or what it got: `room_check()` on the
/// page.
/// The tagged pieces as the controller's parkour reads them (`ParkourPiece::from_extras`): each
/// piece's box and its `kansei_*` extras.
pub fn parkour_pieces(pieces: &[Traversal]) -> Vec<ParkourPiece> {
    pieces
        .iter()
        .filter_map(|t| {
            let x: Value = serde_json::from_str(&t.extras).unwrap_or(Value::Null);
            let entries: Vec<(String, String)> = x.as_object().map_or(Vec::new(), |o| o.iter().map(|(k, v)| (k.clone(), v.as_str().map_or(v.to_string(), str::to_string))).collect());
            ParkourPiece::from_extras(&t.name, Obb::from_min_max(t.min, t.max), entries.iter().map(|(k, v)| (k.as_str(), v.as_str())))
        })
        .collect()
}

pub fn check_traversals(world: &CollisionWorld, pieces: &[Traversal]) -> Vec<(String, Result<(), String>)> {
    use kansei_core::animation::motion_matching::traversal::{detect_obstacle, traversal_kind, DetectionSettings, TraversalRules};
    let (settings, rules) = (DetectionSettings::default(), TraversalRules::default());
    pieces
        .iter()
        .filter_map(|t| Some((t, today(&t.want)?)))
        .map(|(t, want)| {
            let feet = GVec3::new((t.min.x + t.max.x) * 0.5, t.from_y, t.max.z + t.from);
            let got = match detect_obstacle(world, feet, GVec3::NEG_Z, 2.0, &settings) {
                None => Err("nothing found".to_string()),
                Some(o) => traversal_kind(world, &o, feet, &rules, u32::MAX).map_err(|e| format!("{} (height {:.2}, depth {:?}, half width {:.2})", e.describe(), o.height, o.depth, o.half_width)),
            };
            (t.name.clone(), match got {
                Ok(kind) if kind.name().eq_ignore_ascii_case(want) => Ok(()),
                Ok(kind) => Err(format!("{} where {} was meant", kind.name(), want)),
                Err(e) => Err(e),
            })
        })
        .collect()
}
