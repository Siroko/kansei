//! The course of boxes around the start, the ground and the sky, in a blockout look: flat colours
//! with grid lines, lit by the sun (cascade-shadowed) and the sky.

use glam::{Quat, Vec3 as GVec3};

use kansei_core::collision::{CollisionWorld, Obb};
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::shadows::CASCADED_SHADOWS_WGSL;

use crate::{SKY, SUN, SUN_DIR};

/// Ground: grey with a metre grid and a darker 5 m grid, lit by the sun (cascade-shadowed) and the
/// sky, writing no motion (it does not move).
pub const GROUND_WGSL: &str = r#"
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
pub const BOX_WGSL: &str = r#"
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

/// The sky: a gradient from the horizon up, on a sphere round everything.
pub const SKY_WGSL: &str = r#"
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

/// The course: (x, base height, z) of a box's bottom centre, its width, height and depth, its
/// heading (radians) and colour.
pub const COURSE: [([f32; 3], [f32; 3], f32, [f32; 3]); 20] = [
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
pub fn build_course(scene: &mut Scene, world: &mut CollisionWorld) {
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

/// The ground and box shaders' uniform: a base colour, the sun's direction and illuminance, and
/// the sky's luminance.
pub fn surface_params(base: [f32; 3]) -> [f32; 16] {
    let d = SUN_DIR;
    [base[0], base[1], base[2], 0.0, d[0], d[1], d[2], 0.0, SUN[0], SUN[1], SUN[2], 0.0, SKY[0], SKY[1], SKY[2], 0.0]
}
