//! The room: 40 x 40 m and 9 m high. A marble or oak herringbone floor, plaster walls (a
//! terracotta one to the south, board-formed concrete behind the big mirror to the north), three
//! tall steel-framed windows to the west that the low sun comes through, a skylight in the
//! concrete ceiling over the pond with oak beams crossing it, four concrete columns, and the
//! furniture (CC0 models, `assets`): a living corner, a dining table under a pendant, a library,
//! and blocks of concrete and oak to vault, mantle and climb. Everything solid is in the collision
//! world, voxel GI and the ray tracing grid.
//!
//! One-sided: the shell (walls, floor, ceiling) is drawn with back faces discarded on screen, so
//! from outside (the orbit camera past a wall or above the ceiling) the camera looks through it
//! into the room, while the shadow maps draw both sides (the sun outside casts the room's shadow
//! into it). The grid's rays hit a triangle from either side, so the GI's and the mirrors' rays
//! meet the walls as they should.

use glam::{Quat, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::geometries::{BoxGeometry, CylinderGeometry, Geometry, PlaneGeometry};
use kansei_core::gi::GiSurface;
use kansei_core::lights::{DirectionalLight, Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::rt::RtSurface;

use super::assets::Assets;
use super::pbr::{self, PbrParams};

/// Half the room's width and depth, and its height (m).
pub const HALF: f32 = 20.0;
pub const HEIGHT: f32 = 9.0;

/// Where the character starts: south of the pond, facing it (north, -z).
pub const START: [f32; 3] = [0.0, 0.0, 12.0];
pub const START_HEADING: f32 = std::f32::consts::PI;

/// The skylight's opening (x and z half sizes, m), over the pond.
pub const SKYLIGHT: [f32; 2] = [3.0, 6.0];
/// The west windows: their centres along z, half width, sill and head heights.
const WINDOWS: [f32; 3] = [-10.0, 0.0, 10.0];
const WINDOW_HALF: f32 = 2.2;
const SILL: f32 = 0.9;
const HEAD: f32 = 7.2;

/// The lights as built (scene indices): the sun, the skylight's area light, the lamps (the
/// skylight is the first spot light: `RtShadowsEffect::set_rect_emitter(0, ..)`), and the lamps'
/// shades.
pub struct Lights {
    pub sun: usize,
    pub skylight: usize,
    pub lamps: Vec<usize>,
    pub shades: Vec<usize>,
}

/// Default light levels: the sun's illuminance (lux), the skylight's and a lamp's intensity (cd),
/// a shade's radiance (cd/m²).
pub const SUN_LUX: f32 = 9000.0;
pub const SKYLIGHT_CD: f32 = 32000.0;
pub const LAMP_CD: f32 = 700.0;
pub const SHADE_RADIANCE: f32 = 900.0;
pub const SUN_DIR: [f32; 3] = [0.86, -0.4, 0.32];
pub const SUN_COLOR: [f32; 3] = [1.0, 0.74, 0.5];
pub const LAMP_COLOR: [f32; 3] = [1.0, 0.68, 0.4];

/// A PBR surface in the room, for the page to change (deferred lighting, roughness, floors).
pub struct Surface {
    pub index: usize,
    pub params: PbrParams,
    /// One of the floor's quads (rebuilt when the floor changes).
    pub floor: bool,
}

pub struct Layout {
    /// The floor, the walls and the furniture, for the character.
    pub collision: CollisionWorld,
    pub surfaces: Vec<Surface>,
    pub lights: Lights,
    /// The mirrors' renderables.
    pub mirrors: Vec<usize>,
}

/// Linear tints over the textures.
const PLASTER: [f32; 3] = [0.92, 0.9, 0.86];
const TERRACOTTA: [f32; 3] = [0.82, 0.45, 0.32];
const STEEL: [f32; 3] = [0.09, 0.09, 0.1];
const MIRROR: [f32; 3] = [0.92, 0.92, 0.92];
const MIRROR_GI: [f32; 3] = [0.03, 0.03, 0.03];

/// How a surface is drawn: a texture `size` metres across in the world, its roughness scaled and
/// biased, reflective (traced) or not, deferred (`RtShadowsEffect`) or not.
pub fn surface(tint: [f32; 3], size: f32, roughness: [f32; 2], reflective: bool, deferred: bool) -> PbrParams {
    PbrParams {
        tint: [tint[0], tint[1], tint[2], 1.0],
        surface: [size, roughness[0], roughness[1], reflective as u32 as f32],
        ambient: [0.0, 0.0, 0.0, deferred as u32 as f32],
        emissive: [0.0, 0.0, 0.0, 1.0],
        ..Default::default()
    }
}

/// The floor's surface for `floor` ("marble": slabs of 1.2 m, each its own piece of the stone, or
/// "wood": the herringbone as it is), its roughness scaled by `roughness`.
pub fn floor_params(floor: &str, roughness: f32, deferred: bool) -> PbrParams {
    let mut p = surface([1.0, 1.0, 1.0], if floor == "wood" { 2.0 } else { 2.4 }, [roughness, 0.0], true, deferred);
    p.flags[0] = 1.0;
    if floor != "wood" {
        p.pattern = [1.0, 1.8, 1.0, 0.0];
    }
    p
}

/// A rough mean albedo for voxel GI and the grid: the tint times the surface's typical colour.
pub fn mean(name: &str, tint: [f32; 3]) -> [f32; 3] {
    let base = match name {
        "marble" => [0.78, 0.78, 0.78],
        "wood" => [0.18, 0.1, 0.06],
        "plaster" => [0.78, 0.77, 0.74],
        "concrete" => [0.42, 0.42, 0.41],
        "ceiling" => [0.6, 0.6, 0.58],
        "oak" => [0.5, 0.36, 0.22],
        "rug" => [0.08, 0.1, 0.18],
        "blackmarble" => [0.05, 0.05, 0.05],
        "gravel" => [0.35, 0.34, 0.32],
        _ => [0.5, 0.5, 0.5],
    };
    [base[0] * tint[0], base[1] * tint[1], base[2] * tint[2]]
}

struct Builder<'a> {
    scene: &'a mut Scene,
    collision: CollisionWorld,
    assets: &'a Assets,
    deferred: bool,
    surfaces: Vec<Surface>,
}

impl Builder<'_> {
    /// `geometry` drawn with `name`'s maps as `params` says, at `center` turned by `yaw`, in voxel
    /// GI as `albedo`, and in the grid when `grid`.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, label: &str, geometry: Geometry, name: &str, mut params: PbrParams, albedo: [f32; 3], center: [f32; 3], yaw: f32, grid: bool) -> usize {
        params.ambient[3] = self.deferred as u32 as f32;
        let material = pbr::material(label, self.assets.surface(name, label), &params, false);
        let mut r = Renderable::new(geometry, material).with_gi(GiSurface::new(albedo));
        r.rt = grid.then(|| RtSurface::new(albedo));
        r.object.set_position(center[0], center[1], center[2]);
        r.object.rotation.y = yaw;
        let index = self.scene.add(SceneNode::Renderable(r));
        self.surfaces.push(Surface { index, params, floor: false });
        index
    }

    /// A box of `size` centred at `center`, turned by `yaw`, in the collision world when `collide`.
    #[allow(clippy::too_many_arguments)]
    fn solid(&mut self, label: &str, size: [f32; 3], center: [f32; 3], yaw: f32, name: &str, params: PbrParams, albedo: [f32; 3], collide: bool) -> usize {
        if collide {
            self.collision.add_box(Obb::new(GVec3::from(center), GVec3::from(size) * 0.5, Quat::from_rotation_y(yaw)));
        }
        self.add(label, BoxGeometry::new(size[0], size[1], size[2]), name, params, albedo, center, yaw, true)
    }

    /// A box standing on `base` (x, height of its bottom, z), as the course's: `size` wide, high
    /// and deep, turned by `yaw`, in the collision world.
    fn block(&mut self, label: &str, base: [f32; 3], size: [f32; 3], yaw: f32, name: &str, tint: [f32; 3]) -> usize {
        let albedo = mean(name, tint);
        let params = surface(tint, 1.6, [1.0, 0.0], false, self.deferred);
        self.solid(label, size, [base[0], base[1] + size[1] * 0.5, base[2]], yaw, name, params, albedo, true)
    }

    /// A one-sided plane `size` (width, height) at `center`, its front toward the room (turned by
    /// `rotation`, radians about x and y).
    #[allow(clippy::too_many_arguments)]
    fn shell(&mut self, label: &str, size: [f32; 2], center: [f32; 3], rotation: [f32; 2], name: &str, mut params: PbrParams, albedo: [f32; 3]) -> usize {
        params.flags[0] = 1.0;
        let index = self.add(label, PlaneGeometry::new(size[0], size[1]), name, params, albedo, center, 0.0, true);
        if let Some(r) = self.scene.get_renderable_mut(index) {
            r.object.rotation.x = rotation[0];
            r.object.rotation.y = rotation[1];
        }
        index
    }

    /// Model `name` standing on the floor (or `y` up) at `at`, turned by `yaw` and scaled by
    /// `scale`: a renderable for each part, a box round it in the collision world (`collide`), in
    /// the grid (`grid`; cut-out leaves never) and voxel GI as `albedo`. Returns the box (world
    /// centre, half size).
    #[allow(clippy::too_many_arguments)]
    fn model(&mut self, name: &str, at: [f32; 2], y: f32, yaw: f32, scale: f32, collide: bool, grid: bool, albedo: [f32; 3]) -> Option<(GVec3, GVec3)> {
        let model = self.assets.models.get(name)?;
        let lift = y - model.min.y * scale;
        for part in &model.parts {
            let m = &model.materials[part.material];
            let mut params = PbrParams {
                tint: [m.color[0], m.color[1], m.color[2], 0.0],
                surface: [1.0, 1.0, 0.0, 0.0],
                pattern: [0.0, 1.0, 1.0, if m.cutout { 0.5 } else { 0.0 }],
                ambient: [0.0, 0.0, 0.0, self.deferred as u32 as f32],
                emissive: [0.0; 4],
                ..Default::default()
            };
            params.flags[1] = 1.0;
            let material = pbr::material(name, m.maps.maps(name), &params, m.double_sided);
            let mut r = Renderable::new(Geometry::new(name, part.geometry.vertices.clone(), part.geometry.indices.clone()), material).with_gi(GiSurface::new(albedo));
            r.rt = (grid && !m.cutout).then(|| RtSurface::new(albedo));
            r.object.set_position(at[0], lift, at[1]);
            r.object.rotation.y = yaw;
            r.object.scale = Vec3::new(scale, scale, scale);
            let index = self.scene.add(SceneNode::Renderable(r));
            self.surfaces.push(Surface { index, params, floor: false });
        }
        let (lo, hi) = (model.min * scale, model.max * scale);
        let local = GVec3::new((lo.x + hi.x) * 0.5, 0.0, (lo.z + hi.z) * 0.5);
        let centre = GVec3::new(at[0], y + (hi.y - lo.y) * 0.5, at[1]) + Quat::from_rotation_y(yaw) * local;
        let half = (hi - lo) * 0.5;
        if collide {
            self.collision.add_box(Obb::new(centre, half, Quat::from_rotation_y(yaw)));
        }
        Some((centre, half))
    }
}

/// The floor, around the rectangle `hole` (x, z min and max) the pond's terrain fills.
fn floor(b: &mut Builder, hole: Option<([f32; 2], [f32; 2])>, name: &str, roughness: f32) {
    let h = HALF;
    let quads: Vec<([f32; 2], [f32; 2])> = match hole {
        None => vec![([-h, -h], [h, h])],
        Some((min, max)) => vec![([-h, -h], [h, min[1]]), ([-h, max[1]], [h, h]), ([-h, min[1]], [min[0], max[1]]), ([max[0], min[1]], [h, max[1]])],
    };
    for (a, c) in quads {
        let params = floor_params(name, roughness, b.deferred);
        let index = b.shell("Floor", [c[0] - a[0], c[1] - a[1]], [(a[0] + c[0]) * 0.5, 0.0, (a[1] + c[1]) * 0.5], [-std::f32::consts::FRAC_PI_2, 0.0], name, params, mean(name, [1.0; 3]));
        b.surfaces.last_mut().unwrap().floor = true;
        if let Some(r) = b.scene.get_renderable_mut(index) {
            r.cast_shadow = false;
        }
        b.collision.add_box(Obb::from_min_max(GVec3::new(a[0], -1.0, a[1]), GVec3::new(c[0], 0.0, c[1])));
    }
}

/// A floor lamp at `at`: a steel base and pole, a glowing shade, a warm shadowed downlight under
/// it. Returns the light and the shade.
fn floor_lamp(b: &mut Builder, at: [f32; 2], height: f32) -> (usize, usize) {
    let steel = surface(STEEL, 0.5, [0.6, 0.0], false, b.deferred);
    b.solid("LampBase", [0.36, 0.03, 0.36], [at[0], 0.015, at[1]], 0.0, "metal", steel, STEEL, false);
    b.add("LampPole", CylinderGeometry::new(0.018, 0.018, height, 10, 1), "metal", steel, STEEL, [at[0], height * 0.5, at[1]], 0.0, false);
    b.collision.add_box(Obb::new(GVec3::new(at[0], height * 0.5, at[1]), GVec3::new(0.18, height * 0.5, 0.18), Quat::IDENTITY));
    let shade = shade(b.scene, [at[0], height + 0.1, at[1]], 0.26, 0.17, 0.34);
    (lamp_light(b.scene, [at[0], height - 0.1, at[1]]), shade)
}

/// A glowing shade: a frustum of radii `bottom` and `top`, `height` tall, centred at `at`.
fn shade(scene: &mut Scene, at: [f32; 3], bottom: f32, top: f32, height: f32) -> usize {
    let radiance = LAMP_COLOR.map(|c| c * SHADE_RADIANCE);
    let mut r = Renderable::new(CylinderGeometry::new(bottom, top, height, 32, 1), Material::emissive("LampShade", radiance)).with_gi(GiSurface::new([0.8; 3]).with_emission(radiance.map(|c| c * 0.02)));
    r.object.set_position(at[0], at[1], at[2]);
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r))
}

/// A lamp's light: a warm, wide downlight, shadowed (soft).
fn lamp_light(scene: &mut Scene, at: [f32; 3]) -> usize {
    let mut light = SpotLight::new(Vec3::new(at[0], at[1], at[2]), Vec3::new(0.0, -1.0, 0.0), Vec3::new(LAMP_COLOR[0], LAMP_COLOR[1], LAMP_COLOR[2]), LAMP_CD, 10.0, 55f32.to_radians(), 84f32.to_radians());
    light.cast_shadow = true;
    light.source_radius = 0.12;
    light.volumetric_scale = 1.0;
    scene.add(SceneNode::Light(Light::Spot(light)))
}

/// A mirror on a wall: a sheet `size` at `center` turned by `yaw` (0: facing +z), in a steel
/// frame; `roughness` 0 is a mirror, more a brushed metal.
fn mirror(b: &mut Builder, center: [f32; 3], size: [f32; 2], yaw: f32, roughness: f32) -> usize {
    let (s, c) = yaw.sin_cos();
    let back = [center[0] - s * 0.03, center[1], center[2] - c * 0.03];
    b.solid("MirrorFrame", [size[0] + 0.16, size[1] + 0.16, 0.05], back, yaw, "metal", surface(STEEL, 0.5, [0.5, 0.0], false, b.deferred), STEEL, false);
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], 0.02), Material::standard_lit("Mirror", &StandardLitOptions::mirror(MIRROR, roughness))).with_gi(GiSurface::new(MIRROR_GI));
    r.rt = Some(RtSurface::new(MIRROR_GI));
    r.object.set_position(center[0], center[1], center[2]);
    r.object.rotation.y = yaw;
    b.scene.add(SceneNode::Renderable(r))
}

/// The west wall (x = -HALF, facing +x) with its three windows: the wall round them, their reveals
/// and their steel frames (a 3 x 3 grid of panes).
fn west_wall(b: &mut Builder) {
    use std::f32::consts::FRAC_PI_2;
    let (h, top) = (HALF, HEIGHT);
    let wall = surface(PLASTER, 2.0, [1.0, 0.0], false, b.deferred);
    let albedo = mean("plaster", PLASTER);
    let x = -h;
    let face = [0.0, FRAC_PI_2];
    // below the sills and above the heads, the whole length
    b.shell("WallWest", [2.0 * h, SILL], [x, SILL * 0.5, 0.0], face, "plaster", wall, albedo);
    b.shell("WallWest", [2.0 * h, top - HEAD], [x, (HEAD + top) * 0.5, 0.0], face, "plaster", wall, albedo);
    // the piers between the windows
    let mut edges = vec![-h];
    for z in WINDOWS {
        edges.push(z - WINDOW_HALF);
        edges.push(z + WINDOW_HALF);
    }
    edges.push(h);
    for pair in edges.chunks(2) {
        let (a, c) = (pair[0], pair[1]);
        b.shell("WallWest", [c - a, HEAD - SILL], [x, (SILL + HEAD) * 0.5, (a + c) * 0.5], face, "plaster", wall, albedo);
    }
    let steel = surface(STEEL, 0.5, [0.45, 0.0], false, b.deferred);
    let depth = 0.35;
    for z in WINDOWS {
        // the reveals: sill, head and jambs, a wall's depth into it
        let (w, hh) = (2.0 * WINDOW_HALF, HEAD - SILL);
        b.solid("Reveal", [depth, 0.04, w], [x - depth * 0.5, SILL - 0.02, z], 0.0, "plaster", wall, albedo, false);
        b.solid("Reveal", [depth, 0.04, w], [x - depth * 0.5, HEAD + 0.02, z], 0.0, "plaster", wall, albedo, false);
        b.solid("Reveal", [depth, hh, 0.04], [x - depth * 0.5, (SILL + HEAD) * 0.5, z - WINDOW_HALF - 0.02], 0.0, "plaster", wall, albedo, false);
        b.solid("Reveal", [depth, hh, 0.04], [x - depth * 0.5, (SILL + HEAD) * 0.5, z + WINDOW_HALF + 0.02], 0.0, "plaster", wall, albedo, false);
        // the frame: mullions and transoms
        let fx = x - 0.12;
        for k in 0..=3 {
            let zz = z - WINDOW_HALF + 2.0 * WINDOW_HALF * k as f32 / 3.0;
            b.solid("WindowFrame", [0.06, hh, 0.07], [fx, (SILL + HEAD) * 0.5, zz], 0.0, "metal", steel, STEEL, false);
        }
        for k in 0..=3 {
            let y = SILL + hh * k as f32 / 3.0;
            b.solid("WindowFrame", [0.06, 0.07, w], [fx, y, z], 0.0, "metal", steel, STEEL, false);
        }
    }
}

/// The ceiling (y = HEIGHT, facing down) round the skylight, the skylight's well and its steel
/// grid, the oak beams across the room.
fn ceiling(b: &mut Builder) {
    use std::f32::consts::{FRAC_PI_2, PI};
    let (h, top) = (HALF, HEIGHT);
    let [sx, sz] = SKYLIGHT;
    let ceil = surface([0.95, 0.95, 0.93], 3.0, [1.0, 0.05], false, b.deferred);
    let down = [FRAC_PI_2, 0.0];
    for (min, max) in [([-h, -h], [h, -sz]), ([-h, sz], [h, h]), ([-h, -sz], [-sx, sz]), ([sx, -sz], [h, sz])] {
        b.shell("Ceiling", [max[0] - min[0], max[1] - min[1]], [(min[0] + max[0]) * 0.5, top, (min[1] + max[1]) * 0.5], down, "ceiling", ceil, mean("ceiling", [1.0; 3]));
    }
    // the well: four faces looking into the opening, 0.8 m up
    let well = 0.8;
    let wall = surface(PLASTER, 2.0, [1.0, 0.0], false, b.deferred);
    let albedo = mean("plaster", PLASTER);
    let y = top + well * 0.5;
    b.shell("SkylightWell", [2.0 * sx, well], [0.0, y, -sz], [0.0, 0.0], "plaster", wall, albedo);
    b.shell("SkylightWell", [2.0 * sx, well], [0.0, y, sz], [0.0, PI], "plaster", wall, albedo);
    b.shell("SkylightWell", [2.0 * sz, well], [-sx, y, 0.0], [0.0, FRAC_PI_2], "plaster", wall, albedo);
    b.shell("SkylightWell", [2.0 * sz, well], [sx, y, 0.0], [0.0, -FRAC_PI_2], "plaster", wall, albedo);
    // its steel grid, at the top of the well
    let steel = surface(STEEL, 0.5, [0.45, 0.0], false, b.deferred);
    let gy = top + well - 0.05;
    for k in 0..=4 {
        let x = -sx + 2.0 * sx * k as f32 / 4.0;
        b.solid("SkylightGrid", [0.08, 0.1, 2.0 * sz], [x, gy, 0.0], 0.0, "metal", steel, STEEL, false);
    }
    for k in 0..=8 {
        let z = -sz + 2.0 * sz * k as f32 / 8.0;
        b.solid("SkylightGrid", [2.0 * sx, 0.1, 0.08], [0.0, gy, z], 0.0, "metal", steel, STEEL, false);
    }
    // oak beams across the room, under the ceiling (three cross the skylight)
    let tint = [0.85, 0.75, 0.65];
    let oak = surface(tint, 2.0, [1.0, 0.0], false, b.deferred);
    for k in 0..9 {
        let z = -16.0 + 4.0 * k as f32;
        b.solid("Beam", [2.0 * h, 0.5, 0.32], [0.0, top - 0.25, z], 0.0, "oak", oak, mean("oak", tint), false);
    }
}

/// Build the room into `scene` from `assets`; `pond_hole` leaves the floor open where the pond's
/// terrain goes; the floor is `floor_name` with its roughness scaled by `floor_roughness`;
/// `deferred` leaves the direct light to `RtShadowsEffect`.
pub fn build(scene: &mut Scene, assets: &Assets, pond_hole: Option<([f32; 2], [f32; 2])>, floor_name: &str, floor_roughness: f32, deferred: bool) -> Layout {
    use std::f32::consts::{FRAC_PI_2, PI};
    let mut b = Builder { scene, collision: CollisionWorld::new(), assets, deferred, surfaces: Vec::new() };
    let (h, top) = (HALF, HEIGHT);

    // the shell: floor, walls, ceiling, and thick colliders outside the walls
    floor(&mut b, pond_hole, floor_name, floor_roughness);
    let wall = surface(PLASTER, 2.0, [1.0, 0.0], false, deferred);
    let concrete = surface([1.0, 1.0, 1.0], 2.5, [1.0, 0.0], false, deferred);
    b.shell("WallNorth", [2.0 * h, top], [0.0, top * 0.5, -h], [0.0, 0.0], "concrete", concrete, mean("concrete", [1.0; 3]));
    b.shell("WallSouth", [2.0 * h, top], [0.0, top * 0.5, h], [0.0, PI], "plaster", surface(TERRACOTTA, 2.0, [1.0, 0.0], false, deferred), mean("plaster", TERRACOTTA));
    b.shell("WallEast", [2.0 * h, top], [h, top * 0.5, 0.0], [0.0, -FRAC_PI_2], "plaster", wall, mean("plaster", PLASTER));
    west_wall(&mut b);
    ceiling(&mut b);
    for (min, max) in [([-h - 1.0, -h - 1.0], [h + 1.0, -h]), ([-h - 1.0, h], [h + 1.0, h + 1.0]), ([-h - 1.0, -h], [-h, h]), ([h, -h], [h + 1.0, h])] {
        b.collision.add_box(Obb::from_min_max(GVec3::new(min[0], -1.0, min[1]), GVec3::new(max[0], top + 1.0, max[1])));
    }
    // oak baseboards along the walls
    let tint = [0.85, 0.75, 0.65];
    let oak = surface(tint, 1.5, [1.0, 0.0], false, deferred);
    for (size, c) in [([2.0 * h, 0.12, 0.02], [0.0, 0.06, -h + 0.01]), ([2.0 * h, 0.12, 0.02], [0.0, 0.06, h - 0.01]), ([0.02, 0.12, 2.0 * h], [h - 0.01, 0.06, 0.0]), ([0.02, 0.12, 2.0 * h], [-h + 0.01, 0.06, 0.0])] {
        b.solid("Baseboard", size, c, 0.0, "oak", oak, mean("oak", tint), false);
    }
    // four concrete columns round the pond
    for (x, z) in [(-9.5, -9.5), (9.5, -9.5), (-9.5, 9.5), (9.5, 9.5)] {
        b.solid("Column", [0.7, top, 0.7], [x, top * 0.5, z], 0.0, "concrete", concrete, mean("concrete", [1.0; 3]), true);
    }

    // the light: the low sun through the west windows (cascade-shadowed), the skylight's area light
    // (a spot above its well, the RT shadows sampling the opening), lamps
    let d = GVec3::from(SUN_DIR).normalize();
    let mut sun = DirectionalLight::new(Vec3::new(d.x, d.y, d.z), Vec3::new(SUN_COLOR[0], SUN_COLOR[1], SUN_COLOR[2]), SUN_LUX);
    sun.cast_shadow = true;
    let sun = b.scene.add(SceneNode::Light(Light::Directional(sun)));
    let mut sky = SpotLight::new(Vec3::new(0.0, top + 0.7, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.78, 0.86, 1.0), SKYLIGHT_CD, 48.0, 62f32.to_radians(), 86f32.to_radians());
    sky.cast_shadow = true;
    sky.source_radius = 1.5;
    sky.volumetric_scale = 1.0;
    let skylight = b.scene.add(SceneNode::Light(Light::Spot(sky)));

    // two mirrors: a large one on the north wall, a brushed one on the east wall
    let mirrors = vec![mirror(&mut b, [0.0, 2.6, -h + 0.04], [9.0, 4.2], 0.0, 0.0), mirror(&mut b, [h - 0.04, 2.2, -1.0], [5.0, 3.4], -FRAC_PI_2, 0.1)];

    // the living corner (south-east): a sofa, two lounge chairs, a coffee table on a rug
    let rug = surface([1.0, 1.0, 1.0], 1.0, [1.0, 0.0], false, deferred);
    let r = b.add("Rug", BoxGeometry::new(6.4, 0.012, 5.0), "rug", rug, mean("rug", [1.0; 3]), [13.0, 0.006, 13.2], 0.0, true);
    if let Some(r) = b.scene.get_renderable_mut(r) {
        r.cast_shadow = false;
    }
    b.model("sofa_02", [13.0, 16.4], 0.0, PI, 1.0, true, true, [0.06, 0.06, 0.06]);
    b.model("throw_pillows_01", [13.0, 16.2], 0.42, PI, 1.0, false, false, [0.5, 0.3, 0.2]);
    b.model("mid_century_lounge_chair", [10.7, 10.4], 0.0, 0.4, 1.0, true, true, [0.25, 0.14, 0.08]);
    b.model("mid_century_lounge_chair", [15.3, 10.4], 0.0, -0.4, 1.0, true, true, [0.25, 0.14, 0.08]);
    b.model("modern_coffee_table_01", [13.0, 13.3], 0.0, 0.0, 1.0, true, true, [0.4, 0.38, 0.35]);
    b.model("Ottoman_01", [16.4, 13.6], 0.0, 0.3, 1.0, true, true, [0.08, 0.06, 0.05]);
    b.model("potted_plant_02", [17.8, 17.8], 0.0, 0.0, 1.4, true, false, [0.1, 0.2, 0.06]);
    b.model("hanging_picture_frame_02", [13.0, h - 0.03], 1.4, PI, 2.6, false, true, [0.5, 0.45, 0.4]);
    b.model("modern_wooden_cabinet", [h - 0.3, 5.5], 0.0, -FRAC_PI_2, 1.2, true, true, [0.2, 0.12, 0.07]);
    let (mut lamps, mut shades) = (Vec::new(), Vec::new());
    let (l, s) = floor_lamp(&mut b, [17.3, 15.0], 1.6);
    lamps.push(l);
    shades.push(s);

    // the dining table (south-west) under a pendant, four chairs round it
    if let Some((_, half)) = b.model("round_wooden_table_01", [-13.0, 13.0], 0.0, 0.0, 1.0, true, true, [0.18, 0.09, 0.05]) {
        b.model("ceramic_vase_01", [-13.0, 13.0], half.y * 2.0, 0.0, 1.0, false, false, [0.8, 0.8, 0.78]);
    }
    for k in 0..4 {
        let a = k as f32 * FRAC_PI_2 + 0.3;
        b.model("dining_chair_02", [-13.0 + a.sin() * 1.05, 13.0 + a.cos() * 1.05], 0.0, a + PI, 1.0, false, true, [0.15, 0.1, 0.07]);
    }
    if let Some((_, half)) = b.model("modern_ceiling_lamp_01", [-13.0, 13.0], 2.3, 0.0, 1.0, false, false, [0.8, 0.8, 0.8]) {
        let rod = top - (2.3 + half.y * 2.0);
        let steel = surface(STEEL, 0.5, [0.5, 0.0], false, deferred);
        if rod > 0.05 {
            b.add("PendantRod", CylinderGeometry::new(0.006, 0.006, rod, 6, 1), "metal", steel, STEEL, [-13.0, top - rod * 0.5, 13.0], 0.0, false);
        }
    }
    lamps.push(lamp_light(b.scene, [-13.0, 2.25, 13.0]));

    // the library (north-west): steel shelves of books, a reading chair and its lamp
    for x in [-16.2, -12.0] {
        if let Some((c, half)) = b.model("steel_frame_shelves_03", [x, -h + 0.25], 0.0, 0.0, 1.25, true, true, [0.2, 0.15, 0.1]) {
            for level in [0.36, 0.62] {
                b.model("book_encyclopedia_set_01", [x, c.z], half.y * 2.0 * level, 0.0, 1.0, false, false, [0.3, 0.2, 0.15]);
            }
        }
    }
    b.model("wooden_display_shelves_01", [-7.6, -h + 0.25], 0.0, 0.0, 1.0, true, true, [0.5, 0.4, 0.3]);
    b.model("modern_arm_chair_01", [-13.6, -16.8], 0.0, 0.5, 1.0, true, true, [0.1, 0.07, 0.05]);
    b.model("potted_plant_04", [-17.6, -17.6], 0.0, 0.0, 1.6, true, false, [0.2, 0.3, 0.1]);
    let (l, s) = floor_lamp(&mut b, [-15.4, -16.6], 1.7);
    lamps.push(l);
    shades.push(s);

    // the parkour corner (north-east): crates to hurdle and vault, blocks to mantle, a stack, a
    // platform to climb, two ledges with a gap to jump; a sideboard and an island (north-west); a
    // bench on the pond's south shore
    let white = [1.0, 1.0, 1.0];
    b.block("Crate", [9.5, 0.0, -7.0], [2.0, 1.0, 0.8], 0.0, "oak", white);
    b.block("Rail", [13.5, 0.0, -5.0], [3.0, 0.55, 0.25], 0.3, "oak", white);
    b.block("Block", [14.5, 0.0, -9.5], [2.5, 1.3, 2.5], -0.4, "concrete", white);
    b.block("Stack", [10.5, 0.0, -14.0], [3.0, 1.2, 3.0], 0.2, "concrete", white);
    b.block("Stack", [10.61, 1.2, -14.54], [1.8, 1.1, 1.8], 0.2, "oak", white);
    b.block("Platform", [16.0, 0.0, -16.0], [4.0, 2.2, 3.5], 0.0, "concrete", white);
    b.block("Step", [16.0, 0.0, -12.6], [4.0, 0.3, 2.0], 0.0, "concrete", [0.8, 0.8, 0.8]);
    b.block("Ledge", [6.5, 0.0, -16.0], [3.0, 1.2, 4.0], 0.0, "concrete", [0.9, 0.9, 0.9]);
    b.block("Ledge", [6.5, 0.0, -10.5], [3.0, 1.2, 3.5], 0.0, "concrete", [0.9, 0.9, 0.9]);
    b.block("Sideboard", [-12.0, 0.0, -14.0], [2.6, 1.25, 0.9], 0.0, "oak", white);
    b.block("Island", [-11.0, 0.0, -7.5], [2.4, 1.0, 0.9], 0.35, "marble", [0.95, 0.95, 0.95]);
    b.block("Bench", [0.0, 0.0, 8.8], [3.2, 0.45, 0.5], 0.0, "oak", white);
    // a lamp by the east mirror
    let (l, s) = floor_lamp(&mut b, [18.4, 2.6], 1.7);
    lamps.push(l);
    shades.push(s);

    Layout { collision: b.collision, surfaces: b.surfaces, lights: Lights { sun, skylight, lamps, shades }, mirrors }
}

/// The furniture meant for parkour as the character's traversal sees it: from feet in front of each
/// piece, walking at it, the probe (`traversal::detect_obstacle`) finds it and picks the kind of
/// traversal (`traversal_kind`, the default rules). Each piece's name and whether it got the kind
/// meant, or what it got: `room_check()` on the page.
pub fn check_traversals(world: &CollisionWorld) -> Vec<(&'static str, Result<(), String>)> {
    use kansei_core::animation::motion_matching::traversal::{detect_obstacle, traversal_kind, ActionKind, DetectionSettings, TraversalRules};
    let cases: [(&str, [f32; 3], [f32; 3], ActionKind); 10] = [
        ("crate", [9.5, 0.0, -5.6], [0.0, 0.0, -1.0], ActionKind::Vault),
        ("rail", [13.5, 0.0, -3.8], [0.0, 0.0, -1.0], ActionKind::Hurdle),
        ("block", [14.5, 0.0, -6.9], [0.0, 0.0, -1.0], ActionKind::Mantle),
        ("stack", [10.5, 0.0, -11.6], [0.0, 0.0, -1.0], ActionKind::Mantle),
        ("stack's top", [10.6, 1.2, -13.0], [0.0, 0.0, -1.0], ActionKind::Mantle),
        ("platform, from the step", [16.0, 0.3, -13.0], [0.0, 0.0, -1.0], ActionKind::Climb),
        ("ledge", [6.5, 0.0, -7.6], [0.0, 0.0, -1.0], ActionKind::Mantle),
        ("sideboard", [-12.0, 0.0, -12.4], [0.0, 0.0, -1.0], ActionKind::Vault),
        ("island", [-11.4, 0.0, -6.0], [0.0, 0.0, -1.0], ActionKind::Vault),
        ("bench", [0.0, 0.0, 10.2], [0.0, 0.0, -1.0], ActionKind::Hurdle),
    ];
    let (settings, rules) = (DetectionSettings::default(), TraversalRules::default());
    cases
        .into_iter()
        .map(|(name, feet, direction, want)| {
            let feet = GVec3::from(feet);
            let got = match detect_obstacle(world, feet, GVec3::from(direction), 2.0, &settings) {
                None => Err("nothing found".to_string()),
                Some(o) => traversal_kind(world, &o, feet, &rules, u32::MAX).map_err(|e| format!("{} (height {:.2}, depth {:?}, half width {:.2})", e.describe(), o.height, o.depth, o.half_width)),
            };
            (name, match got {
                Ok(kind) if kind == want => Ok(()),
                Ok(kind) => Err(format!("{} where {} was meant", kind.name(), want.name())),
                Err(e) => Err(e),
            })
        })
        .collect()
}
