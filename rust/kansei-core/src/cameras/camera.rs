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
    // GPU resources (owned by Camera, created by Renderer on first render)
    view_buf: Option<wgpu::Buffer>,
    proj_buf: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
    pub initialized: bool,
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
            view_buf: None,
            proj_buf: None,
            camera_bind_group: None,
            initialized: false,
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
            self.object.update_world_matrix(None);
            self.view_matrix = self.object.world_matrix.inverse();
        }
        self.inverse_view_matrix = self.view_matrix.inverse();
    }

    pub fn look_at(&mut self, target: &Vec3) {
        self.look_at_target = Some(*target);
        self.view_matrix = Mat4::look_at(&self.object.position, target, &Vec3::UP);
        self.inverse_view_matrix = self.view_matrix.inverse();
        self.object.world_matrix = self.inverse_view_matrix;
        self.object.model_matrix = self.object.world_matrix; // keep in sync
    }

    /// Called by Renderer during first render. Creates GPU buffers and bind group.
    pub fn gpu_initialize(
        &mut self,
        device: &wgpu::Device,
        camera_bgl: &wgpu::BindGroupLayout,
        light_buf: &wgpu::Buffer,
    ) {
        if self.initialized {
            return;
        }
        self.view_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera/View"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.proj_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera/Proj"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera/BG"),
            layout: camera_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.view_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.proj_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: light_buf.as_entire_binding(),
                },
            ],
        }));
        self.initialized = true;
    }

    /// Upload current view + projection matrices to GPU.
    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(ref buf) = self.view_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.view_matrix.as_slice()));
        }
        if let Some(ref buf) = self.proj_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.projection_matrix.as_slice()));
        }
    }

    /// Get the camera's bind group (group 1).
    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.camera_bind_group.as_ref()
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
