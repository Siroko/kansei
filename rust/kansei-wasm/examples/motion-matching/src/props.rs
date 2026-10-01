//! Blockout props by the lake (the cannon, the water mill), in the course's prototype look: meshes
//! merged from boxes and cylinders, each part one of the course boxes' flat colours with the same
//! half-metre grid lines (in the prop's own space, so they turn with it), lit by the sun
//! (cascade-shadowed) and the sky as the boxes are.

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::geometries::{Geometry, Vertex};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::shadows::CASCADED_SHADOWS_WGSL;

use crate::{SKY, SUN, SUN_DIR};

/// The props' colours, picked per vertex (`uv.x`): the course's (`COURSE` in `lib.rs`), and black.
#[derive(Clone, Copy)]
pub enum Paint {
    /// The rails'.
    Orange = 0,
    /// The vault boxes'.
    Blue = 1,
    /// The blocks'.
    Grey = 2,
    /// The beams'.
    Dark = 3,
    Black = 4,
}

const PROP_WGSL: &str = r#"
struct Light { sun_dir: vec4<f32>, sun: vec4<f32>, sky: vec4<f32> };
@group(0) @binding(0) var<uniform> light: Light;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) local: vec3<f32>,
    @location(3) local_normal: vec3<f32>,
    @location(4) @interpolate(flat) paint: u32,
};
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.local = position.xyz;
    out.local_normal = normal;
    out.paint = u32(uv.x + 0.5);
    return out;
}
fn lines(p: vec2<f32>) -> f32 {
    let q = p / 0.5;
    let d = abs(fract(q - 0.5) - 0.5) / fwidth(q);
    return 1.0 - min(min(d.x, d.y), 1.0);
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    var colors = array<vec3<f32>, 5>(
        vec3<f32>(0.55, 0.35, 0.2),
        vec3<f32>(0.25, 0.4, 0.55),
        vec3<f32>(0.45, 0.45, 0.4),
        vec3<f32>(0.3, 0.3, 0.3),
        vec3<f32>(0.03, 0.03, 0.03),
    );
    // the course boxes' grid, on the face's plane in the prop's own space
    let a = abs(in.local_normal);
    var p = in.local.xz;
    if (a.x > a.y && a.x > a.z) { p = in.local.zy; } else if (a.z > a.y) { p = in.local.xy; }
    let base = colors[min(in.paint, 4u)] * (1.0 - 0.3 * lines(p));
    let n = normalize(in.normal);
    let l = -normalize(light.sun_dir.xyz);
    let shadow = kansei_sun_shadow(in.world, n, in.clip.xy);
    let sky = mix(light.sky.rgb * 0.25, light.sky.rgb, n.y * 0.5 + 0.5);
    let lit = base / 3.14159265 * light.sun.rgb * max(dot(n, l), 0.0) * shadow + base * sky;
    return vec4<f32>(lit, 1.0);
}
"#;

/// The props' material.
pub fn material(label: &str) -> Material {
    let mut material = Material::new(label, &format!("{CASCADED_SHADOWS_WGSL}\n{PROP_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions::default());
    let d = SUN_DIR;
    material.set_uniform_bindable(0, label, &[d[0], d[1], d[2], 0.0, SUN[0], SUN[1], SUN[2], 0.0, SKY[0], SKY[1], SKY[2], 0.0]);
    material
}

/// A mesh built from parts, each placed by a transform.
#[derive(Default)]
pub struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    fn push(&mut self, at: Mat4, paint: Paint, positions: &[GVec3], normals: &[GVec3], indices: &[u32]) {
        let k = self.vertices.len() as u32;
        let normal_matrix = at.inverse().transpose();
        for (p, n) in positions.iter().zip(normals) {
            let p = at.transform_point3(*p);
            let n = normal_matrix.transform_vector3(*n).normalize();
            self.vertices.push(Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [paint as u32 as f32, 0.0] });
        }
        self.indices.extend(indices.iter().map(|i| i + k));
    }

    /// A box of `size` centred at the origin, placed by `at`.
    pub fn cuboid(&mut self, at: Mat4, size: GVec3, paint: Paint) {
        let h = size * 0.5;
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        for axis in 0..3 {
            for sign in [-1.0f32, 1.0] {
                let mut n = GVec3::ZERO;
                n[axis] = sign;
                // two axes across the face, wound counter-clockwise seen from outside
                let (u, v) = {
                    let mut u = GVec3::ZERO;
                    let mut v = GVec3::ZERO;
                    u[(axis + 1) % 3] = 1.0;
                    v[(axis + 2) % 3] = 1.0;
                    if sign < 0.0 { (v, u) } else { (u, v) }
                };
                let k = positions.len() as u32;
                for (a, b) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                    positions.push((n + u * a + v * b) * h);
                    normals.push(n);
                }
                indices.extend_from_slice(&[k, k + 1, k + 2, k, k + 2, k + 3]);
            }
        }
        self.push(at, paint, &positions, &normals, &indices);
    }

    /// A cylinder of `radius` along +y from 0 to `length`, `sides` around, capped, placed by `at`.
    pub fn cylinder(&mut self, at: Mat4, radius: f32, length: f32, sides: u32, paint: Paint) {
        let (mut positions, mut normals, mut indices) = (Vec::new(), Vec::new(), Vec::new());
        for s in 0..=sides {
            let a = s as f32 / sides as f32 * std::f32::consts::TAU;
            let n = GVec3::new(a.cos(), 0.0, -a.sin());
            positions.extend([n * radius, n * radius + GVec3::Y * length]);
            normals.extend([n, n]);
        }
        for s in 0..sides {
            let k = s * 2;
            indices.extend_from_slice(&[k, k + 2, k + 3, k, k + 3, k + 1]);
        }
        for (y, n) in [(0.0, -GVec3::Y), (length, GVec3::Y)] {
            let c = positions.len() as u32;
            positions.push(GVec3::Y * y);
            normals.push(n);
            for s in 0..=sides {
                let a = s as f32 / sides as f32 * std::f32::consts::TAU;
                positions.push(GVec3::new(a.cos() * radius, y, -a.sin() * radius));
                normals.push(n);
            }
            for s in 0..sides {
                let (p, q) = (c + 1 + s, c + 2 + s);
                if n.y > 0.0 { indices.extend_from_slice(&[c, p, q]) } else { indices.extend_from_slice(&[c, q, p]) }
            }
        }
        self.push(at, paint, &positions, &normals, &indices);
    }

    /// A box from `a` to `b`, `thickness` square across.
    pub fn beam(&mut self, a: GVec3, b: GVec3, thickness: f32, paint: Paint) {
        let d = b - a;
        let at = Mat4::from_rotation_translation(glam::Quat::from_rotation_arc(GVec3::Y, d.normalize()), (a + b) * 0.5);
        self.cuboid(at, GVec3::new(thickness, d.length(), thickness), paint);
    }

    /// A cylinder from `a` to `b`.
    pub fn rod(&mut self, a: GVec3, b: GVec3, radius: f32, sides: u32, paint: Paint) {
        let d = b - a;
        let at = Mat4::from_rotation_translation(glam::Quat::from_rotation_arc(GVec3::Y, d.normalize()), a);
        self.cylinder(at, radius, d.length(), sides, paint);
    }

    pub fn geometry(self, label: &str) -> Geometry {
        Geometry::new(label, self.vertices, self.indices)
    }
}
