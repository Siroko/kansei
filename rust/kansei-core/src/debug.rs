//! Debug drawing: [`DebugBoxes`], a set of flat-coloured boxes the app places by matrix each
//! frame (one instanced draw), for trajectories, bones, contact points and other markers.

use glam::{Mat4, Quat, Vec3};

use crate::buffers::{BufferType, ComputeBuffer};
use crate::geometries::{BoxGeometry, InstancedGeometry};
use crate::materials::{Binding, Material, MaterialOptions, ShaderStages};
use crate::objects::{Renderable, Scene, SceneNode};
use crate::renderers::Renderer;

/// Unit boxes placed by a per-instance matrix, one flat colour lit a little from above.
pub const DEBUG_BOXES_WGSL: &str = r#"
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

/// `count` unit boxes (centred, 1 across) in one colour, each placed by its matrix in
/// [`matrices`](Self::matrices): set them, then [`upload`](Self::upload) once a frame. A zero
/// matrix hides its box. They cast no shadow and draw after the opaque scene; `x_ray` draws them
/// over everything (no depth test).
pub struct DebugBoxes {
    /// One matrix per box (unit box to world).
    pub matrices: Vec<Mat4>,
    buffer: wgpu::Buffer,
    index: usize,
}

impl DebugBoxes {
    /// Add `count` boxes of `color` (linear rgb, in the scene's units of radiance) to `scene`,
    /// all hidden (zero matrices).
    pub fn new(renderer: &Renderer, scene: &mut Scene, label: &str, count: usize, color: [f32; 3], x_ray: bool) -> Self {
        use wgpu::util::DeviceExt;
        let matrices = vec![Mat4::ZERO; count];
        let buffer = renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(&matrices),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });
        let instances = ComputeBuffer::from_external(label, buffer.clone(), BufferType::Storage).with_vertex_mat4(3);
        let options = if x_ray {
            MaterialOptions { transparent: true, depth_write: Some(false), depth_compare: wgpu::CompareFunction::Always, ..Default::default() }
        } else {
            MaterialOptions::default()
        };
        let mut material = Material::new(label, DEBUG_BOXES_WGSL, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], options);
        material.set_uniform_bindable(0, label, &[color[0], color[1], color[2], 1.0f32]);
        let mut r = Renderable::new(InstancedGeometry::new(BoxGeometry::new(1.0, 1.0, 1.0), count as u32, vec![instances]), material);
        r.dynamic = true;
        r.cast_shadow = false;
        r.render_order = 10;
        let index = scene.add(SceneNode::Renderable(r));
        Self { matrices, buffer, index }
    }

    /// Write `matrices` to the GPU (one write).
    pub fn upload(&self, renderer: &Renderer) {
        renderer.queue().write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.matrices));
    }

    /// Hide every box (zero matrices; upload to show it).
    pub fn clear(&mut self) {
        self.matrices.fill(Mat4::ZERO);
    }

    /// The boxes' renderable in the scene.
    pub fn index(&self) -> usize {
        self.index
    }

    pub fn visible(&self, scene: &Scene) -> bool {
        scene.get_renderable(self.index).is_some_and(|r| r.visible)
    }

    pub fn set_visible(&self, scene: &mut Scene, visible: bool) {
        if let Some(r) = scene.get_renderable_mut(self.index) {
            r.visible = visible;
        }
    }
}

/// A box's matrix from `a` to `b`, `thickness` across (zero, hidden, when they meet): a bone, a
/// ray, an edge.
pub fn segment(a: Vec3, b: Vec3, thickness: f32) -> Mat4 {
    let d = b - a;
    let length = d.length();
    if length < 1e-5 {
        return Mat4::ZERO;
    }
    let rotation = Quat::from_rotation_arc(Vec3::Y, d / length);
    Mat4::from_scale_rotation_translation(Vec3::new(thickness, length, thickness), rotation, (a + b) * 0.5)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_validates() {
        let module = naga::front::wgsl::parse_str(DEBUG_BOXES_WGSL).expect("parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("validates");
    }

    #[test]
    fn a_segment_spans_its_ends() {
        let (a, b) = (Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 2.0));
        let m = segment(a, b, 0.1);
        assert!(m.transform_point3(Vec3::new(0.0, -0.5, 0.0)).distance(a) < 1e-5);
        assert!(m.transform_point3(Vec3::new(0.0, 0.5, 0.0)).distance(b) < 1e-5);
        assert_eq!(segment(a, a, 0.1), Mat4::ZERO);
    }
}
