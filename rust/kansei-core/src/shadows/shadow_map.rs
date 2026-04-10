use crate::math::{Vec3, Mat4};
use crate::cameras::Camera;

/// Directional light shadow map.
pub struct ShadowMap {
    pub resolution: u32,
    pub depth_texture: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub light_vp: Mat4,
    pub light_vp_buf: Option<wgpu::Buffer>,
    pub bias: f32,
    pub normal_bias: f32,
    initialized: bool,
}

impl ShadowMap {
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            depth_texture: None,
            depth_view: None,
            light_vp: Mat4::identity(),
            light_vp_buf: None,
            bias: 0.001,
            normal_bias: 0.02,
            initialized: false,
        }
    }

    pub fn initialize(&mut self, device: &wgpu::Device) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowMap/Depth"),
            size: wgpu::Extent3d {
                width: self.resolution,
                height: self.resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        self.light_vp_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowMap/LightVP"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.initialized = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Compute tight orthographic light-view-projection matrix from camera frustum.
    pub fn compute_light_vp(&mut self, camera: &Camera, light_dir: &Vec3) {
        let g_light_dir = glam::Vec3::new(light_dir.x, light_dir.y, light_dir.z).normalize();

        let view = camera.view_matrix.to_glam();
        let proj = camera.projection_matrix.to_glam();
        let inv_vp = (proj * view).inverse();

        // 8 frustum corners in NDC (WebGPU: z in [0,1])
        let ndc_corners = [
            glam::Vec3::new(-1.0, -1.0, 0.0),
            glam::Vec3::new( 1.0, -1.0, 0.0),
            glam::Vec3::new(-1.0,  1.0, 0.0),
            glam::Vec3::new( 1.0,  1.0, 0.0),
            glam::Vec3::new(-1.0, -1.0, 1.0),
            glam::Vec3::new( 1.0, -1.0, 1.0),
            glam::Vec3::new(-1.0,  1.0, 1.0),
            glam::Vec3::new( 1.0,  1.0, 1.0),
        ];

        let mut world_corners = [glam::Vec3::ZERO; 8];
        let mut center = glam::Vec3::ZERO;
        for (i, ndc) in ndc_corners.iter().enumerate() {
            let clip = inv_vp * glam::Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
            world_corners[i] = clip.truncate() / clip.w;
            center += world_corners[i];
        }
        center /= 8.0;

        let eye = center - g_light_dir * 100.0;
        let up = if g_light_dir.y.abs() >= 0.99 { glam::Vec3::X } else { glam::Vec3::Y };
        let light_view = glam::Mat4::look_at_rh(eye, center, up);

        let mut min_ls = glam::Vec3::splat(f32::MAX);
        let mut max_ls = glam::Vec3::splat(f32::MIN);
        for corner in &world_corners {
            let ls = (light_view * glam::Vec4::new(corner.x, corner.y, corner.z, 1.0)).truncate();
            min_ls = min_ls.min(ls);
            max_ls = max_ls.max(ls);
        }

        let z_range = max_ls.z - min_ls.z;
        min_ls.z -= z_range * 2.0;

        let light_proj = glam::Mat4::orthographic_rh(
            min_ls.x, max_ls.x,
            min_ls.y, max_ls.y,
            -max_ls.z, -min_ls.z,
        );

        self.light_vp = Mat4::from(light_proj * light_view);
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(ref buf) = self.light_vp_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.light_vp.as_slice()));
        }
    }
}
