//! The room: 40 x 40 m and 9 m high, its walls, floor and ceiling one-sided planes facing in, the
//! light panel and the lamps, two mirrors, and the furniture: a living corner, a dining table,
//! shelves, and crates and platforms to vault, mantle and climb. Everything solid is a box in the
//! collision world (the walls thick boxes outside the planes), in voxel GI and in the ray tracing
//! grid.
//!
//! One-sided: the planes are drawn with back faces culled, so from outside (the orbit camera past
//! a wall or above the ceiling) the camera looks through them into the room, while from inside
//! they are solid. The grid's rays hit a triangle from either side, so the GI's rays and the
//! mirrors' from inside the room meet the walls as they should; nothing outside lights the room.

use glam::{Quat, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::geometries::{BoxGeometry, CylinderGeometry, Geometry, IcosphereGeometry, PlaneGeometry};
use kansei_core::gi::GiSurface;
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::rt::RtSurface;

/// Half the room's width and depth, and its height (m).
pub const HALF: f32 = 20.0;
pub const HEIGHT: f32 = 9.0;

/// Where the character starts: south of the pond, facing it (north, -z).
pub const START: [f32; 3] = [0.0, 0.0, 12.0];
pub const START_HEADING: f32 = std::f32::consts::PI;

/// The light panel's side (m) and the spot light under it that stands for it: its intensity (cd)
/// and the radius of its PCSS emitter (m), which softens the shadows as an area light's are.
const PANEL: f32 = 7.0;
const PANEL_INTENSITY: f32 = 14000.0;
const PANEL_SOURCE_RADIUS: f32 = 0.9;
/// The panel's radiance (cd/m²): a Lambertian emitter of its area with the spot's intensity.
const PANEL_RADIANCE: f32 = PANEL_INTENSITY / (PANEL * PANEL);
/// The lamps' warm white, their intensity (cd) and their shades' radiance (cd/m²).
const LAMP_COLOR: [f32; 3] = [1.0, 0.72, 0.45];
const LAMP_INTENSITY: f32 = 260.0;
const SHADE_RADIANCE: f32 = 450.0;

/// Linear albedos.
const FLOOR: [f32; 3] = [0.42, 0.3, 0.2];
const WALL: [f32; 3] = [0.74, 0.71, 0.66];
const WALL_SOUTH: [f32; 3] = [0.6, 0.27, 0.17];
const WALL_WEST: [f32; 3] = [0.36, 0.47, 0.36];
const CEILING: [f32; 3] = [0.78, 0.78, 0.76];
const WOOD: [f32; 3] = [0.24, 0.13, 0.07];
const PLY: [f32; 3] = [0.58, 0.44, 0.27];
const CONCRETE: [f32; 3] = [0.46, 0.46, 0.44];
const FRAME: [f32; 3] = [0.06, 0.05, 0.04];
/// The mirrors' reflectance, and what the GI's rays and voxels see of them (dark: they don't
/// mirror).
const MIRROR: [f32; 3] = [0.92, 0.92, 0.92];
const MIRROR_GI: [f32; 3] = [0.03, 0.03, 0.03];

/// A matte surface of `color` (roughness `roughness`).
pub fn matte(label: &str, color: [f32; 3], roughness: f32) -> Material {
    Material::standard_lit(label, &StandardLitOptions { base_color: color, roughness, ..Default::default() })
}

/// The room, as built.
pub struct Layout {
    /// The floor, the walls and the furniture, for the character.
    pub collision: CollisionWorld,
}

/// Something solid: a box of `size` (m) centred at `center`, turned by `yaw`, drawn with
/// `material`, in voxel GI and the grid as `albedo`, and (with `collide`) in the collision world.
#[allow(clippy::too_many_arguments)]
fn solid(scene: &mut Scene, collision: &mut CollisionWorld, label: &str, size: [f32; 3], center: [f32; 3], yaw: f32, color: [f32; 3], roughness: f32, collide: bool) -> usize {
    if collide {
        collision.add_box(Obb::new(GVec3::from(center), GVec3::from(size) * 0.5, Quat::from_rotation_y(yaw)));
    }
    add(scene, BoxGeometry::new(size[0], size[1], size[2]), matte(label, color, roughness), color, center, yaw)
}

/// `geometry` at `center` turned by `yaw`, in voxel GI and the grid as `albedo`.
fn add(scene: &mut Scene, geometry: Geometry, material: Material, albedo: [f32; 3], center: [f32; 3], yaw: f32) -> usize {
    let mut r = Renderable::new(geometry, material).with_gi(GiSurface::new(albedo));
    r.rt = Some(RtSurface::new(albedo));
    r.object.set_position(center[0], center[1], center[2]);
    r.object.rotation.y = yaw;
    scene.add(SceneNode::Renderable(r))
}

/// A box standing on `base` (x, height of its bottom, z), as the course's: `size` wide, high and
/// deep, turned by `yaw`, in the collision world.
fn block(scene: &mut Scene, collision: &mut CollisionWorld, label: &str, base: [f32; 3], size: [f32; 3], yaw: f32, color: [f32; 3]) -> usize {
    solid(scene, collision, label, size, [base[0], base[1] + size[1] * 0.5, base[2]], yaw, color, 0.75, true)
}

/// A point `(x, z)` turned by `yaw` about the origin and moved to `at` (x, z).
fn place(at: [f32; 2], yaw: f32, x: f32, z: f32) -> [f32; 2] {
    let (s, c) = yaw.sin_cos();
    [at[0] + x * c + z * s, at[1] - x * s + z * c]
}

/// A sofa at `at` (x, z) facing `yaw` (0: +z): seat, back and arms, each a collider (the back
/// to vault over, the arms to step on).
fn sofa(scene: &mut Scene, collision: &mut CollisionWorld, at: [f32; 2], yaw: f32, color: [f32; 3]) {
    let parts: [([f32; 3], [f32; 3]); 4] = [
        // (size, centre in the sofa's frame: x across, y up, z forward)
        ([2.4, 0.45, 0.95], [0.0, 0.225, 0.05]),
        ([2.4, 0.9, 0.25], [0.0, 0.45, -0.5]),
        ([0.22, 0.65, 1.2], [-1.31, 0.325, -0.02]),
        ([0.22, 0.65, 1.2], [1.31, 0.325, -0.02]),
    ];
    for (size, c) in parts {
        let p = place(at, yaw, c[0], c[2]);
        solid(scene, collision, "Sofa", size, [p[0], c[1], p[1]], yaw, color, 0.9, true);
    }
}

/// A table at `at` (x, z) turned by `yaw`: a top `size` (width, height to its top, depth) on four
/// legs, one box in the collision world (vault it, or mantle onto it).
fn table(scene: &mut Scene, collision: &mut CollisionWorld, at: [f32; 2], yaw: f32, size: [f32; 3], color: [f32; 3]) {
    let (w, h, d) = (size[0], size[1], size[2]);
    collision.add_box(Obb::new(GVec3::new(at[0], h * 0.5, at[1]), GVec3::new(w, h, d) * 0.5, Quat::from_rotation_y(yaw)));
    let mut no = CollisionWorld::new();
    solid(scene, &mut no, "Table", [w, 0.06, d], [at[0], h - 0.03, at[1]], yaw, color, 0.5, false);
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let p = place(at, yaw, sx * (w * 0.5 - 0.1), sz * (d * 0.5 - 0.1));
        solid(scene, &mut no, "TableLeg", [0.07, h - 0.06, 0.07], [p[0], (h - 0.06) * 0.5, p[1]], yaw, color, 0.5, false);
    }
}

/// A chair at `at` facing `yaw`: a seat and a back (not colliders: the character walks round them).
fn chair(scene: &mut Scene, at: [f32; 2], yaw: f32, color: [f32; 3]) {
    let mut no = CollisionWorld::new();
    solid(scene, &mut no, "Chair", [0.45, 0.05, 0.45], [at[0], 0.46, at[1]], yaw, color, 0.6, false);
    let back = place(at, yaw, 0.0, -0.21);
    solid(scene, &mut no, "Chair", [0.45, 0.5, 0.04], [back[0], 0.73, back[1]], yaw, color, 0.6, false);
    for (sx, sz) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
        let p = place(at, yaw, sx * 0.19, sz * 0.19);
        solid(scene, &mut no, "ChairLeg", [0.04, 0.44, 0.04], [p[0], 0.22, p[1]], yaw, color, 0.6, false);
    }
}

/// A floor lamp at `at` (x, z): a base, a pole and a glowing shade, with a shadowed spot light
/// under the shade pointing down.
fn floor_lamp(scene: &mut Scene, collision: &mut CollisionWorld, at: [f32; 2], height: f32) {
    let mut no = CollisionWorld::new();
    solid(scene, &mut no, "LampBase", [0.4, 0.04, 0.4], [at[0], 0.02, at[1]], 0.0, FRAME, 0.4, false);
    add(scene, CylinderGeometry::new(0.025, 0.025, height, 8, 1), matte("LampPole", FRAME, 0.4), FRAME, [at[0], height * 0.5, at[1]], 0.0);
    collision.add_box(Obb::new(GVec3::new(at[0], height * 0.5, at[1]), GVec3::new(0.2, height * 0.5, 0.2), Quat::IDENTITY));
    let shade = [LAMP_COLOR[0] * SHADE_RADIANCE, LAMP_COLOR[1] * SHADE_RADIANCE, LAMP_COLOR[2] * SHADE_RADIANCE];
    let mut r = Renderable::new(CylinderGeometry::new(0.3, 0.2, 0.36, 24, 1), Material::emissive("LampShade", shade)).with_gi(GiSurface::new([0.8; 3]).with_emission(scale(shade, 0.05)));
    r.object.set_position(at[0], height + 0.12, at[1]);
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r));
    lamp_light(scene, [at[0], height - 0.08, at[1]]);
}

fn scale(c: [f32; 3], s: f32) -> [f32; 3] {
    c.map(|v| v * s)
}

/// A lamp's light: a warm, wide downlight, shadowed (soft).
fn lamp_light(scene: &mut Scene, at: [f32; 3]) {
    let mut light = SpotLight::new(Vec3::new(at[0], at[1], at[2]), Vec3::new(0.0, -1.0, 0.0), Vec3::new(LAMP_COLOR[0], LAMP_COLOR[1], LAMP_COLOR[2]), LAMP_INTENSITY, 9.0, 55f32.to_radians(), 82f32.to_radians());
    light.cast_shadow = true;
    light.source_radius = 0.12;
    light.volumetric_scale = 1.0;
    scene.add(SceneNode::Light(Light::Spot(light)));
}

/// A mirror on a wall: a sheet `size` (width, height) at `center`, its face turned by `yaw` (0:
/// toward +z), in a dark frame; `roughness` 0 is a mirror, more a brushed metal.
fn mirror(scene: &mut Scene, center: [f32; 3], size: [f32; 2], yaw: f32, roughness: f32) {
    let mut no = CollisionWorld::new();
    // the frame: a slab a little larger, behind the glass
    let back = place([center[0], center[2]], yaw, 0.0, -0.03);
    solid(scene, &mut no, "MirrorFrame", [size[0] + 0.24, size[1] + 0.24, 0.05], [back[0], center[1], back[1]], yaw, FRAME, 0.5, false);
    let mut r = Renderable::new(BoxGeometry::new(size[0], size[1], 0.02), Material::standard_lit("Mirror", &StandardLitOptions::mirror(MIRROR, roughness))).with_gi(GiSurface::new(MIRROR_GI));
    r.rt = Some(RtSurface::new(MIRROR_GI));
    r.object.set_position(center[0], center[1], center[2]);
    r.object.rotation.y = yaw;
    scene.add(SceneNode::Renderable(r));
}

/// A plant in a pot: a cylinder pot and a few green spheres.
fn plant(scene: &mut Scene, collision: &mut CollisionWorld, at: [f32; 2], size: f32) {
    let pot = [0.32, 0.18, 0.12];
    add(scene, CylinderGeometry::new(0.28 * size, 0.38 * size, 0.6 * size, 16, 1), matte("Pot", pot, 0.8), pot, [at[0], 0.3 * size, at[1]], 0.0);
    collision.add_box(Obb::new(GVec3::new(at[0], 0.3 * size, at[1]), GVec3::new(0.3, 0.3, 0.3) * size, Quat::IDENTITY));
    let leaf = [0.1, 0.28, 0.07];
    for (k, (x, y, z, r)) in [(0.0, 1.1, 0.0, 0.45), (0.25, 1.45, 0.1, 0.32), (-0.2, 1.5, -0.15, 0.3), (0.05, 1.8, -0.05, 0.25)].into_iter().enumerate() {
        add(scene, IcosphereGeometry::new(r * size, 1), matte("Leaves", leaf, 0.9), leaf, [at[0] + x * size, y * size, at[1] + z * size], k as f32);
    }
}

/// The floor: the room's whole floor, or (with the pond) the part around the rectangle `hole`
/// (x, z min and max) the pond's terrain fills; and its collider.
fn floor(scene: &mut Scene, collision: &mut CollisionWorld, hole: Option<([f32; 2], [f32; 2])>) {
    let h = HALF;
    let quads: Vec<([f32; 2], [f32; 2])> = match hole {
        None => vec![([-h, -h], [h, h])],
        Some((min, max)) => vec![([-h, -h], [h, min[1]]), ([-h, max[1]], [h, h]), ([-h, min[1]], [min[0], max[1]]), ([max[0], min[1]], [h, max[1]])],
    };
    for (a, b) in quads {
        let (w, d) = (b[0] - a[0], b[1] - a[1]);
        let mut r = Renderable::new(PlaneGeometry::new(w, d), matte("Floor", FLOOR, 0.55)).with_gi(GiSurface::new(FLOOR));
        r.rt = Some(RtSurface::new(FLOOR));
        r.object.set_position((a[0] + b[0]) * 0.5, 0.0, (a[1] + b[1]) * 0.5);
        r.object.rotation.x = -std::f32::consts::FRAC_PI_2;
        r.cast_shadow = false;
        scene.add(SceneNode::Renderable(r));
        collision.add_box(Obb::from_min_max(GVec3::new(a[0], -1.0, a[1]), GVec3::new(b[0], 0.0, b[1])));
    }
}

/// A one-sided plane `size` (width, height) at `center`, its front toward the room (turned by
/// `rotation`, radians about x and y), in GI and the grid.
fn plane(scene: &mut Scene, label: &str, size: [f32; 2], center: [f32; 3], rotation: [f32; 2], color: [f32; 3]) -> usize {
    let mut r = Renderable::new(PlaneGeometry::new(size[0], size[1]), matte(label, color, 0.85)).with_gi(GiSurface::new(color));
    r.rt = Some(RtSurface::new(color));
    r.object.set_position(center[0], center[1], center[2]);
    r.object.rotation.x = rotation[0];
    r.object.rotation.y = rotation[1];
    // (the light is inside: the walls cast no shadow it needs, and from outside they would)
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r))
}

/// Build the room into `scene`; `pond_hole` leaves the floor open where the pond's terrain goes.
pub fn build(scene: &mut Scene, pond_hole: Option<([f32; 2], [f32; 2])>) -> Layout {
    use std::f32::consts::{FRAC_PI_2, PI};
    let mut collision = CollisionWorld::new();
    let (h, top) = (HALF, HEIGHT);

    // the shell: floor, ceiling and walls facing in, and thick colliders outside the walls
    floor(scene, &mut collision, pond_hole);
    plane(scene, "Ceiling", [2.0 * h, 2.0 * h], [0.0, top, 0.0], [FRAC_PI_2, 0.0], CEILING);
    plane(scene, "WallNorth", [2.0 * h, top], [0.0, top * 0.5, -h], [0.0, 0.0], WALL);
    plane(scene, "WallSouth", [2.0 * h, top], [0.0, top * 0.5, h], [0.0, PI], WALL_SOUTH);
    plane(scene, "WallWest", [2.0 * h, top], [-h, top * 0.5, 0.0], [0.0, FRAC_PI_2], WALL_WEST);
    plane(scene, "WallEast", [2.0 * h, top], [h, top * 0.5, 0.0], [0.0, -FRAC_PI_2], WALL);
    for (min, max) in [([-h - 1.0, -h - 1.0], [h + 1.0, -h]), ([-h - 1.0, h], [h + 1.0, h + 1.0]), ([-h - 1.0, -h], [-h, h]), ([h, -h], [h + 1.0, h])] {
        collision.add_box(Obb::from_min_max(GVec3::new(min[0], -1.0, min[1]), GVec3::new(max[0], top + 1.0, max[1])));
    }

    // the light panel in the ceiling, and the spot light standing for it: straight down, its
    // cone wide enough for the whole floor, its emitter as wide as PCSS lets a penumbra grow
    let panel = [PANEL_RADIANCE, PANEL_RADIANCE * 0.97, PANEL_RADIANCE * 0.92];
    // (one-sided, facing down: from above the ceiling the camera looks past it)
    let mut r = Renderable::new(PlaneGeometry::new(PANEL, PANEL), Material::emissive("LightPanel", panel)).with_gi(GiSurface::new([0.8; 3]));
    r.object.set_position(0.0, top - 0.03, 0.0);
    r.object.rotation.x = FRAC_PI_2;
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r));
    // its frame: four thin strips round it
    let mut no = CollisionWorld::new();
    let (edge, side) = (PANEL * 0.5 + 0.08, PANEL + 0.32);
    for (size, x, z) in [([side, 0.06, 0.16], 0.0, -edge), ([side, 0.06, 0.16], 0.0, edge), ([0.16, 0.06, side], -edge, 0.0), ([0.16, 0.06, side], edge, 0.0)] {
        solid(scene, &mut no, "PanelFrame", size, [x, top - 0.04, z], 0.0, FRAME, 0.5, false);
    }
    let mut light = SpotLight::new(Vec3::new(0.0, top - 0.25, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.96, 0.9), PANEL_INTENSITY, 45.0, 62f32.to_radians(), 84f32.to_radians());
    light.cast_shadow = true;
    light.source_radius = PANEL_SOURCE_RADIUS;
    light.volumetric_scale = 1.0;
    scene.add(SceneNode::Light(Light::Spot(light)));

    // two mirrors: a large one on the north wall, a brushed one on the east wall
    mirror(scene, [0.0, 2.6, -h + 0.04], [9.0, 4.2], 0.0, 0.0);
    mirror(scene, [h - 0.04, 2.2, -1.0], [5.0, 3.4], -FRAC_PI_2, 0.1);

    // the living corner (south-east): two sofas face each other over a coffee table, on a rug
    sofa(scene, &mut collision, [13.0, 10.0], 0.0, [0.1, 0.18, 0.4]);
    sofa(scene, &mut collision, [13.0, 15.6], PI, [0.62, 0.42, 0.1]);
    table(scene, &mut collision, [13.0, 12.8], 0.0, [1.5, 0.45, 0.8], WOOD);
    let rug = [0.5, 0.12, 0.1];
    let mut r = Renderable::new(BoxGeometry::new(6.0, 0.012, 5.0), matte("Rug", rug, 0.95)).with_gi(GiSurface::new(rug));
    r.rt = Some(RtSurface::new(rug));
    r.object.set_position(13.0, 0.006, 12.8);
    r.cast_shadow = false;
    scene.add(SceneNode::Renderable(r));
    floor_lamp(scene, &mut collision, [17.6, 9.0], 1.65);

    // the dining table (south-west), its chairs, and a lamp in the corner
    table(scene, &mut collision, [-13.0, 13.0], 0.0, [2.8, 0.78, 1.1], WOOD);
    for (x, z, yaw) in [(-14.0, 12.0, PI), (-13.0, 12.0, PI), (-12.0, 12.0, PI), (-14.0, 14.0, 0.0), (-13.0, 14.0, 0.0), (-12.0, 14.0, 0.0)] {
        chair(scene, [x, z], yaw, [0.3, 0.2, 0.12]);
    }
    floor_lamp(scene, &mut collision, [-17.8, 17.8], 1.7);

    // shelves along the west wall (north-west), a sideboard to mantle onto, an island to vault
    for k in 0..3 {
        let z = -14.0 + k as f32 * 4.2;
        block(scene, &mut collision, "Shelf", [-h + 0.25, 0.0, z], [0.45, 2.4, 3.8], 0.0, WOOD);
        for shelf in 1..5 {
            let mut no = CollisionWorld::new();
            let color = [[0.5, 0.15, 0.1], [0.12, 0.25, 0.4], [0.55, 0.5, 0.35], [0.2, 0.35, 0.18]][(k + shelf) % 4];
            solid(scene, &mut no, "Books", [0.3, 0.3, 3.2], [-h + 0.55, shelf as f32 * 0.48 + 0.15, z], 0.0, color, 0.8, false);
        }
    }
    block(scene, &mut collision, "Sideboard", [-12.0, 0.0, -14.0], [2.6, 1.25, 0.9], 0.0, PLY);
    block(scene, &mut collision, "Island", [-11.0, 0.0, -7.5], [2.4, 1.0, 0.9], 0.35, CONCRETE);
    floor_lamp(scene, &mut collision, [-17.8, -3.0], 1.7);

    // the parkour corner (north-east): crates to hurdle and vault, blocks to mantle, a stack, a
    // platform to climb, two platforms with a gap to jump
    block(scene, &mut collision, "Crate", [9.5, 0.0, -7.0], [2.0, 1.0, 0.8], 0.0, PLY);
    block(scene, &mut collision, "Rail", [13.5, 0.0, -5.0], [3.0, 0.55, 0.25], 0.3, WOOD);
    block(scene, &mut collision, "Block", [14.5, 0.0, -9.5], [2.5, 1.3, 2.5], -0.4, CONCRETE);
    block(scene, &mut collision, "Stack", [10.5, 0.0, -14.0], [3.0, 1.2, 3.0], 0.2, PLY);
    block(scene, &mut collision, "Stack", [10.61, 1.2, -14.54], [1.8, 1.1, 1.8], 0.2, [0.55, 0.5, 0.35]);
    block(scene, &mut collision, "Platform", [16.0, 0.0, -16.0], [4.0, 2.2, 3.5], 0.0, CONCRETE);
    block(scene, &mut collision, "Step", [16.0, 0.0, -12.6], [4.0, 0.3, 2.0], 0.0, [0.4, 0.42, 0.45]);
    block(scene, &mut collision, "Ledge", [6.5, 0.0, -16.0], [3.0, 1.2, 4.0], 0.0, [0.4, 0.5, 0.45]);
    block(scene, &mut collision, "Ledge", [6.5, 0.0, -10.5], [3.0, 1.2, 3.5], 0.0, [0.4, 0.5, 0.45]);
    // a bench on the pond's south shore, low enough to hurdle
    block(scene, &mut collision, "Bench", [0.0, 0.0, 8.8], [3.2, 0.45, 0.5], 0.0, WOOD);

    // plants in the corners
    plant(scene, &mut collision, [17.5, 17.5], 1.2);
    plant(scene, &mut collision, [-17.5, -17.5], 1.4);
    plant(scene, &mut collision, [17.5, 5.5], 1.0);
    plant(scene, &mut collision, [-6.0, 17.6], 1.1);

    Layout { collision }
}

/// The furniture meant for parkour as the character's traversal sees it: from feet in front of each
/// piece, walking at it, the probe (`traversal::detect_obstacle`) finds it and picks the kind of
/// traversal (`traversal_kind`, the default rules). Each piece's name and whether it got the kind
/// meant, or what it got: `room_check()` on the page.
pub fn check_traversals() -> Vec<(&'static str, Result<(), String>)> {
    use kansei_core::animation::motion_matching::traversal::{detect_obstacle, traversal_kind, ActionKind, DetectionSettings, TraversalRules};
    let layout = build(&mut Scene::new(), None);
    let world = &layout.collision;
    let cases: [(&str, [f32; 3], [f32; 3], ActionKind); 12] = [
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
        ("dining table", [-13.0, 0.0, 11.0], [0.0, 0.0, 1.0], ActionKind::Vault),
        ("sofa's back", [13.0, 0.0, 8.3], [0.0, 0.0, 1.0], ActionKind::Hurdle),
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
