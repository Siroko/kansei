use crate::math::{Vec3, Mat4};
use crate::objects::Object3D;

/// Perspective camera with view/projection matrices.
pub struct Camera {
    pub object: Object3D,
    pub fov: f32,       // degrees
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub view_matrix: Mat4,
    pub inverse_view_matrix: Mat4,
    pub projection_matrix: Mat4,
    look_at_target: Option<Vec3>,
}

impl Camera {
    pub fn new(fov: f32, near: f32, far: f32, aspect: f32) -> Self {
        let fov_rad = fov * std::f32::consts::PI / 180.0;
        let projection = Mat4::perspective(fov_rad, aspect, near, far);
        Self {
            object: Object3D::new(),
            fov,
            near,
            far,
            aspect,
            view_matrix: Mat4::identity(),
            inverse_view_matrix: Mat4::identity(),
            projection_matrix: projection,
            look_at_target: None,
        }
    }

    pub fn update_projection_matrix(&mut self) {
        let fov_rad = self.fov * std::f32::consts::PI / 180.0;
        self.projection_matrix = Mat4::perspective(fov_rad, self.aspect, self.near, self.far);
    }

    pub fn update_view_matrix(&mut self) {
        if let Some(target) = self.look_at_target {
            self.view_matrix = Mat4::look_at(&self.object.position, &target, &Vec3::UP);
        } else {
            self.object.update_model_matrix();
            self.view_matrix = self.object.world_matrix.inverse();
        }
        self.inverse_view_matrix = self.view_matrix.inverse();
    }

    pub fn look_at(&mut self, target: &Vec3) {
        self.look_at_target = Some(*target);
        self.view_matrix = Mat4::look_at(&self.object.position, target, &Vec3::UP);
        self.inverse_view_matrix = self.view_matrix.inverse();
        self.object.world_matrix = self.inverse_view_matrix;
    }

    pub fn position(&self) -> &Vec3 {
        &self.object.position
    }

    pub fn set_position(&mut self, x: f32, y: f32, z: f32) {
        self.object.set_position(x, y, z);
    }
}

impl std::ops::Deref for Camera {
    type Target = Object3D;
    fn deref(&self) -> &Object3D {
        &self.object
    }
}

impl std::ops::DerefMut for Camera {
    fn deref_mut(&mut self) -> &mut Object3D {
        &mut self.object
    }
}
