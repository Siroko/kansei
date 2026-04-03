use crate::math::{Vec3, Mat4};

/// Base 3D object with position, rotation, scale, and parent-child hierarchy.
/// Mirrors the TS Object3D class.
pub struct Object3D {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub world_matrix: Mat4,
    pub normal_matrix: Mat4,
    children: Vec<usize>, // indices into a scene-level arena
    parent: Option<usize>,
    dirty: bool,
}

impl Object3D {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::new(1.0, 1.0, 1.0),
            world_matrix: Mat4::identity(),
            normal_matrix: Mat4::identity(),
            children: Vec::new(),
            parent: None,
            dirty: true,
        }
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position.set(x, y, z);
        self.dirty = true;
    }

    /// Recompute the world matrix from position/rotation/scale.
    pub fn update_model_matrix(&mut self) {
        let t = glam::Mat4::from_translation(self.position.to_glam());
        let rx = glam::Mat4::from_rotation_x(self.rotation.x);
        let ry = glam::Mat4::from_rotation_y(self.rotation.y);
        let rz = glam::Mat4::from_rotation_z(self.rotation.z);
        let s = glam::Mat4::from_scale(self.scale.to_glam());
        self.world_matrix = Mat4::from(t * rz * ry * rx * s);
        self.dirty = false;
    }

    /// Compute normal matrix (inverse transpose of model-view).
    pub fn update_normal_matrix(&mut self, view_matrix: &Mat4) {
        let mv = view_matrix.to_glam() * self.world_matrix.to_glam();
        let normal = mv.inverse().transpose();
        self.normal_matrix = Mat4::from(normal);
    }

    pub fn look_at(&mut self, target: &Vec3) {
        let forward = (*target - self.position).normalize();
        self.rotation.y = forward.x.atan2(forward.z);
        self.rotation.x = (-forward.y).asin();
        self.dirty = true;
    }

    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub fn children(&self) -> &[usize] {
        &self.children
    }

    pub fn add_child(&mut self, child_index: usize) {
        self.children.push(child_index);
    }
}

impl Default for Object3D {
    fn default() -> Self {
        Self::new()
    }
}
