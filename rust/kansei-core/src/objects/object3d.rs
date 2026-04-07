use crate::math::{Vec3, Mat4};

/// Base 3D object with position, rotation, scale, and parent-child hierarchy.
/// Mirrors the TS Object3D class.
pub struct Object3D {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub model_matrix: Mat4,     // local T * Rz * Ry * Rx * S
    pub world_matrix: Mat4,     // parent.world * model (or just model if root)
    pub normal_matrix: Mat4,    // transpose(inverse(world_matrix)) — world-space
    children: Vec<usize>, // indices into a scene-level arena
    parent: Option<usize>,
    dirty: bool,
    last_position: Vec3,
    last_rotation: Vec3,
    last_scale: Vec3,
}

impl Object3D {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            rotation: Vec3::ZERO,
            scale: Vec3::new(1.0, 1.0, 1.0),
            model_matrix: Mat4::identity(),
            world_matrix: Mat4::identity(),
            normal_matrix: Mat4::identity(),
            children: Vec::new(),
            parent: None,
            dirty: true,
            last_position: Vec3::ZERO,
            last_rotation: Vec3::ZERO,
            last_scale: Vec3::new(1.0, 1.0, 1.0),
        }
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.position.set(x, y, z);
        self.dirty = true;
    }

    /// Recompute the local model matrix from position/rotation/scale.
    pub fn update_model_matrix(&mut self) {
        // Auto-detect dirty from direct field mutation
        if !self.dirty {
            if self.position.x != self.last_position.x || self.position.y != self.last_position.y || self.position.z != self.last_position.z
                || self.rotation.x != self.last_rotation.x || self.rotation.y != self.last_rotation.y || self.rotation.z != self.last_rotation.z
                || self.scale.x != self.last_scale.x || self.scale.y != self.last_scale.y || self.scale.z != self.last_scale.z
            {
                self.dirty = true;
            }
        }
        if !self.dirty { return; }

        let t = glam::Mat4::from_translation(self.position.to_glam());
        let rx = glam::Mat4::from_rotation_x(self.rotation.x);
        let ry = glam::Mat4::from_rotation_y(self.rotation.y);
        let rz = glam::Mat4::from_rotation_z(self.rotation.z);
        let s = glam::Mat4::from_scale(self.scale.to_glam());
        self.model_matrix = Mat4::from(t * rz * ry * rx * s);

        self.last_position = self.position;
        self.last_rotation = self.rotation;
        self.last_scale = self.scale;
        self.dirty = false;
    }

    /// Recompute world matrix: parent.world * model (or just model if root).
    pub fn update_world_matrix(&mut self, parent_world: Option<&Mat4>) {
        match parent_world {
            Some(pw) => self.world_matrix = pw.mul(&self.model_matrix),
            None => self.world_matrix = self.model_matrix,
        }
    }

    /// Compute world-space normal matrix: transpose(inverse(world_matrix)).
    pub fn update_normal_matrix(&mut self) {
        self.normal_matrix = Mat4::from(self.world_matrix.to_glam().inverse().transpose());
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
