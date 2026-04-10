use super::{Object3D, Renderable};
use crate::geometries::PlaneGeometry;
use crate::materials::{Binding, CullMode, Material, MaterialOptions};
use crate::math::Vec3;

const CORNELL_WGSL: &str = include_str!("../shaders/basic_lit.wgsl");

pub struct CornellBox {
    pub object: Object3D,
    faces: Vec<Renderable>,
}

impl CornellBox {
    pub fn new(bounds_min: [f32; 3], bounds_max: [f32; 3]) -> Self {
        let [x0, y0, z0] = bounds_min;
        let [x1, y1, z1] = bounds_max;
        let sx = (x1 - x0).abs().max(0.0001);
        let sy = (y1 - y0).abs().max(0.0001);
        let sz = (z1 - z0).abs().max(0.0001);
        let cx = (x0 + x1) * 0.5;
        let cy = (y0 + y1) * 0.5;
        let cz = (z0 + z1) * 0.5;

        let mut object = Object3D::new();
        object.set_position(cx, cy, cz);

        let mut faces = Vec::with_capacity(6);
        let mut push_face =
            |name: &str, color: [f32; 4], pos: Vec3, rot: Vec3, scale: Vec3| {
                let geometry = PlaneGeometry::new(1.0, 1.0);
                let material = Material::new(
                    &format!("CornellBox/{name}/Material"),
                    CORNELL_WGSL,
                    vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
                    MaterialOptions {
                        cull_mode: CullMode::Front,
                        ..Default::default()
                    },
                );
                let mut renderable = Renderable::new(geometry, material);
                renderable.object.position = pos;
                renderable.object.rotation = rot;
                renderable.object.scale = scale;

                // basic_lit expects color + specular vec4.
                let uniform: [f32; 8] = [
                    color[0], color[1], color[2], color[3],
                    0.15, 0.15, 0.15, 0.5,
                ];
                renderable.material.set_uniform_bindable(
                    0,
                    &format!("CornellBox/{name}/Color"),
                    &uniform,
                );

                faces.push(renderable);
            };

        // Floor (+Y normal)
        push_face(
            "Floor",
            [0.72, 0.72, 0.72, 1.0],
            Vec3::new(cx, y0, cz),
            Vec3::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            Vec3::new(sx, sz, 1.0),
        );
        // Ceiling (-Y normal)
        push_face(
            "Ceiling",
            [0.72, 0.72, 0.72, 1.0],
            Vec3::new(cx, y1, cz),
            Vec3::new(std::f32::consts::FRAC_PI_2, 0.0, 0.0),
            Vec3::new(sx, sz, 1.0),
        );
        // Back (+Z normal)
        push_face(
            "Back",
            [0.72, 0.72, 0.72, 1.0],
            Vec3::new(cx, cy, z0),
            Vec3::ZERO,
            Vec3::new(sx, sy, 1.0),
        );
        // Front (-Z normal)
        push_face(
            "Front",
            [0.72, 0.72, 0.72, 1.0],
            Vec3::new(cx, cy, z1),
            Vec3::new(0.0, std::f32::consts::PI, 0.0),
            Vec3::new(sx, sy, 1.0),
        );
        // Left (+X normal)
        push_face(
            "Left",
            [0.80, 0.15, 0.10, 1.0],
            Vec3::new(x0, cy, cz),
            Vec3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0),
            Vec3::new(sz, sy, 1.0),
        );
        // Right (-X normal)
        push_face(
            "Right",
            [0.15, 0.80, 0.10, 1.0],
            Vec3::new(x1, cy, cz),
            Vec3::new(0.0, -std::f32::consts::FRAC_PI_2, 0.0),
            Vec3::new(sz, sy, 1.0),
        );

        Self {
            object,
            faces,
        }
    }

}

impl CornellBox {
    /// Add all faces to the scene as SceneNode::Renderable children.
    pub fn add_to_scene(self, scene: &mut super::Scene) -> Vec<usize> {
        let mut indices = Vec::with_capacity(self.faces.len());
        for face in self.faces {
            indices.push(scene.add_renderable_internal(face, true));
        }
        if let Some(&root_idx) = indices.first() {
            for &child_idx in &indices[1..] {
                scene.attach_child(root_idx, child_idx);
            }
        }
        indices
    }
}

