use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;

pub const MAX_CASCADES: usize = 4;

pub struct CascadedShadowOptions {
    /// Cascades, 1 to 4.
    pub cascades: u32,
    /// Resolution of each cascade's square map.
    pub resolution: u32,
    /// View distance the last cascade reaches; shadows fade out over its last tenth.
    pub max_distance: f32,
    /// Split scheme between uniform (0) and logarithmic (1) cascade depths.
    pub split_lambda: f32,
    /// Metres toward the light that casters are still drawn beyond each cascade's bounds, so
    /// tall things outside the view keep their shadows.
    pub caster_distance: f32,
    /// Apparent diameter of the light in radians, for contact-hardening (PCSS) penumbrae: the sun
    /// is 0.0093. 0 gives a fixed small PCF kernel.
    pub light_angular_diameter: f32,
    /// Receiver offset along the normal, in texels of the cascade it is looked up in.
    pub normal_bias: f32,
    /// Fraction of a cascade's half-width over which it dithers into the next one.
    pub blend: f32,
}

impl Default for CascadedShadowOptions {
    fn default() -> Self {
        Self {
            cascades: 4,
            resolution: 2048,
            max_distance: 250.0,
            split_lambda: 0.75,
            caster_distance: 200.0,
            light_angular_diameter: 0.0093,
            normal_bias: 1.5,
            blend: 0.15,
        }
    }
}

/// `KanseiCascade` in cascaded_shadows.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default, Debug)]
pub(crate) struct CascadeGpu {
    view_proj: [f32; 16],
    texel_world: f32,
    depth_range: f32,
    radius: f32,
    _pad: f32,
}

/// `KanseiCascades` in cascaded_shadows.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CascadesGpu {
    cascades: [CascadeGpu; MAX_CASCADES],
    light_direction: [f32; 3],
    count: u32,
    light_color: [f32; 3],
    tan_angular_radius: f32,
    normal_bias: f32,
    blend: f32,
    max_distance: f32,
    _pad0: f32,
    camera_pos: [f32; 3],
    _pad1: f32,
}

/// One cascade this frame: its light view and orthographic projection.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CascadeSlot {
    pub view: glam::Mat4,
    pub projection: glam::Mat4,
    pub radius: f32,
    pub depth_range: f32,
}

/// The view depths splitting `[near, max_distance]` into `count` cascades: a blend of uniform
/// and logarithmic splits (the "practical" scheme, Zhang et al. 2006). Returns count + 1 depths.
pub fn cascade_splits(near: f32, max_distance: f32, count: u32, lambda: f32) -> Vec<f32> {
    (0..=count)
        .map(|i| {
            let t = i as f32 / count as f32;
            let log = near * (max_distance / near).powf(t);
            let uniform = near + (max_distance - near) * t;
            lambda * log + (1.0 - lambda) * uniform
        })
        .collect()
}

/// The smallest sphere around the part of a view frustum between view depths `near` and `far`
/// (`tan_diag`: tangent of the half-angle to the frustum's corners): its centre's depth along
/// the view axis and its radius. It depends only on the depths and the lens, so it doesn't
/// change as the camera turns: the cascade keeps its size, and its texels stay put.
pub fn frustum_slice_sphere(near: f32, far: f32, tan_diag: f32) -> (f32, f32) {
    let t2 = tan_diag * tan_diag;
    let center = ((near + far) * (1.0 + t2) * 0.5).min(far);
    let radius = ((center - near).powi(2) + near * near * t2).sqrt().max(((far - center).powi(2) + far * far * t2).sqrt());
    (center, radius)
}

/// The cascades for `camera` and a light travelling along `light_dir`: per cascade, the
/// bounding sphere of its frustum slice, its centre snapped to whole texels in light space, an
/// orthographic box around it reaching `caster_distance` further toward the light.
pub(crate) fn fit_cascades(o: &CascadedShadowOptions, camera: &Camera, light_dir: glam::Vec3) -> Vec<CascadeSlot> {
    let dir = light_dir.normalize_or(glam::Vec3::NEG_Y);
    let up = if dir.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
    let rotation = glam::Mat4::look_to_rh(glam::Vec3::ZERO, dir, up);
    let inv_view = camera.inverse_view_matrix.to_glam();
    let eye = inv_view.w_axis.truncate();
    let forward = -inv_view.z_axis.truncate().normalize_or(glam::Vec3::NEG_Z);
    let tan_v = (camera.fov.to_radians() * 0.5).tan();
    let tan_diag = tan_v * (1.0 + camera.aspect * camera.aspect).sqrt();
    let splits = cascade_splits(camera.near, o.max_distance.max(camera.near * 2.0), o.cascades, o.split_lambda);
    (0..o.cascades as usize)
        .map(|c| {
            let (depth, radius) = frustum_slice_sphere(splits[c], splits[c + 1], tan_diag);
            let center = eye + forward * depth;
            let texel = 2.0 * radius / o.resolution as f32;
            let mut center_ls = rotation.transform_point3(center);
            center_ls.x = (center_ls.x / texel).floor() * texel;
            center_ls.y = (center_ls.y / texel).floor() * texel;
            // the eye sits `radius + caster_distance` toward the light from the centre
            let back = radius + o.caster_distance;
            let view = glam::Mat4::from_translation(-(center_ls + glam::Vec3::new(0.0, 0.0, back))) * rotation;
            let depth_range = back + radius;
            let projection = glam::Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, depth_range);
            CascadeSlot { view, projection, radius, depth_range }
        })
        .collect()
}

/// Cascaded shadow maps for the scene's first directional light when it casts shadows (the sun,
/// or the moon): stable cascades (bounding spheres that keep their size as the camera turns,
/// light-space centres snapped to whole texels, so shadows don't shimmer), rendered each frame
/// through the casters' own vertex shaders and culled per cascade, and looked up with
/// contact-hardening PCSS and dithered transitions (`shadows::CASCADED_SHADOWS_WGSL`).
///
/// The widest cascade also stands in for the single directional shadow map: materials that read
/// group 3 binding 0 (basic_lit.wgsl) and the volumetric fog (`set_cascaded_shadow_map`) use it.
pub struct CascadedShadowMap {
    pub options: CascadedShadowOptions,
    pub texture: wgpu::Texture,
    /// All cascades (`texture_depth_2d_array`).
    pub array_view: wgpu::TextureView,
    /// The widest cascade alone, as a `texture_depth_2d`.
    pub far_view: wgpu::TextureView,
    /// The widest cascade's view-projection (a `mat4x4<f32>` uniform), rewritten every frame.
    pub far_view_proj: wgpu::Buffer,
    pub(crate) uniform: wgpu::Buffer,
    layer_views: Vec<wgpu::TextureView>,
    cameras: Vec<Camera>,
    pub(crate) slots: Vec<CascadeSlot>,
}

impl CascadedShadowMap {
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
    /// Depth bias of the cascade pipelines (the shader adds a normal offset on top).
    pub(crate) const DEPTH_BIAS: wgpu::DepthBiasState = wgpu::DepthBiasState { constant: 0, slope_scale: 2.0, clamp: 0.0 };

    pub(crate) fn new(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout, light_buf: &wgpu::Buffer, options: CascadedShadowOptions) -> Self {
        let count = options.cascades.clamp(1, MAX_CASCADES as u32);
        let options = CascadedShadowOptions { cascades: count, ..options };
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("CascadedShadowMap"),
            size: wgpu::Extent3d { width: options.resolution, height: options.resolution, depth_or_array_layers: count },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            // (COPY_SRC: read back by the tests)
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let array_view = texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let layer = |l: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: l,
                array_layer_count: Some(1),
                ..Default::default()
            })
        };
        let layer_views = (0..count).map(layer).collect();
        let far_view = layer(count - 1);
        let buffer = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: size as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let cameras = (0..count)
            .map(|_| {
                let mut camera = Camera::new(60.0, 0.1, 100.0, 1.0);
                camera.gpu_initialize(device, camera_bgl, light_buf);
                camera
            })
            .collect();
        Self {
            options,
            texture,
            array_view,
            far_view,
            far_view_proj: buffer("CascadedShadowMap/FarViewProj", 64),
            uniform: buffer("CascadedShadowMap/Cascades", std::mem::size_of::<CascadesGpu>()),
            layer_views,
            cameras,
            slots: Vec::new(),
        }
    }

    /// Fit the cascades to `camera` for a light travelling along `light_dir`.
    pub(crate) fn fit(&mut self, camera: &Camera, light_dir: glam::Vec3) {
        self.slots = fit_cascades(&self.options, camera, light_dir);
    }

    /// Upload this frame's cascades, the wide cascade's matrix and the cascade cameras.
    pub(crate) fn upload(&mut self, queue: &wgpu::Queue, light_dir: glam::Vec3, light_color: glam::Vec3, camera_pos: glam::Vec3) {
        let o = &self.options;
        let mut cascades = [CascadeGpu::default(); MAX_CASCADES];
        for (c, slot) in self.slots.iter().enumerate() {
            cascades[c] = CascadeGpu {
                view_proj: (slot.projection * slot.view).to_cols_array(),
                texel_world: 2.0 * slot.radius / o.resolution as f32,
                depth_range: slot.depth_range,
                radius: slot.radius,
                _pad: 0.0,
            };
        }
        let data = CascadesGpu {
            cascades,
            light_direction: light_dir.normalize_or(glam::Vec3::NEG_Y).to_array(),
            count: self.slots.len() as u32,
            light_color: light_color.to_array(),
            tan_angular_radius: (o.light_angular_diameter.max(0.0) * 0.5).tan(),
            normal_bias: o.normal_bias,
            blend: o.blend.clamp(1e-3, 1.0),
            max_distance: o.max_distance,
            _pad0: 0.0,
            camera_pos: camera_pos.to_array(),
            _pad1: 0.0,
        };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&data));
        if let Some(far) = self.slots.last() {
            queue.write_buffer(&self.far_view_proj, 0, bytemuck::cast_slice(&(far.projection * far.view).to_cols_array()));
        }
        for (camera, slot) in self.cameras.iter_mut().zip(&self.slots) {
            camera.view_matrix = slot.view.into();
            camera.inverse_view_matrix = slot.view.inverse().into();
            camera.projection_matrix = slot.projection.into();
            camera.upload(queue);
        }
    }

    /// No shadowed directional light this frame: cascades off.
    pub(crate) fn disable(&mut self, queue: &wgpu::Queue) {
        self.slots.clear();
        let data = CascadesGpu { cascades: [CascadeGpu::default(); MAX_CASCADES], ..Zeroable::zeroed() };
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&data));
    }

    /// The widest cascade's view-projection this frame.
    pub fn far_view_projection(&self) -> Option<glam::Mat4> {
        self.slots.last().map(|s| s.projection * s.view)
    }

    pub(crate) fn layer_view(&self, cascade: usize) -> &wgpu::TextureView {
        &self.layer_views[cascade]
    }

    pub(crate) fn camera(&self, cascade: usize) -> &Camera {
        &self.cameras[cascade]
    }

    #[cfg(test)]
    pub(crate) fn gpu_size() -> usize {
        std::mem::size_of::<CascadesGpu>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_span_the_range_and_grow() {
        let s = cascade_splits(0.5, 250.0, 4, 0.75);
        assert_eq!(s.len(), 5);
        assert!((s[0] - 0.5).abs() < 1e-5 && (s[4] - 250.0).abs() < 1e-3);
        assert!(s.windows(2).all(|w| w[1] > w[0]));
        // lambda 1 is logarithmic: equal ratios
        let log = cascade_splits(1.0, 1000.0, 3, 1.0);
        assert!((log[1] - 10.0).abs() < 1e-3 && (log[2] - 100.0).abs() < 1e-2);
    }

    /// The sphere holds every corner of its frustum slice, and turning the camera doesn't change it.
    #[test]
    fn slice_sphere_bounds_the_slice() {
        for (near, far, tan_diag) in [(0.5, 10.0, 0.6), (10.0, 60.0, 0.6), (60.0, 250.0, 1.2), (1.0, 1.5, 0.3)] {
            let (c, r) = frustum_slice_sphere(near, far, tan_diag);
            for (z, half) in [(near, near * tan_diag), (far, far * tan_diag)] {
                assert!(((z - c) * (z - c) + half * half).sqrt() <= r + 1e-3, "{near} {far}: corner outside");
            }
            assert!(c >= near && c <= far);
        }
    }

    /// Turning and moving the camera a little keeps every cascade's size, and moves it by whole
    /// texels in light space, so its texels land where they were: no shimmering edges.
    #[test]
    fn cascades_are_stable_under_camera_motion() {
        let o = CascadedShadowOptions::default();
        let dir = glam::Vec3::new(-0.4, -0.6, -0.7);
        let mut camera = Camera::new(40.0, 0.5, 1000.0, 16.0 / 9.0);
        camera.set_position(0.0, 2.0, 0.0);
        camera.look_at(&crate::math::Vec3::new(10.0, 1.0, -20.0));
        let a = fit_cascades(&o, &camera, dir);
        camera.set_position(0.013, 2.0, -0.021);
        camera.look_at(&crate::math::Vec3::new(-5.0, 1.5, -20.0));
        let b = fit_cascades(&o, &camera, dir);
        assert_eq!(a.len(), 4);
        for (a, b) in a.iter().zip(&b) {
            assert!((a.radius - b.radius).abs() < 1e-4);
            let texel = 2.0 * a.radius / o.resolution as f32;
            let d = (b.view.w_axis - a.view.w_axis).truncate() / texel;
            assert!((d.x - d.x.round()).abs() < 1e-2 && (d.y - d.y.round()).abs() < 1e-2, "{d:?}");
        }
        // and each cascade holds its frustum slice's centre: its box contains the view axis point
        let eye = camera.inverse_view_matrix.to_glam().w_axis.truncate();
        let forward = -camera.inverse_view_matrix.to_glam().z_axis.truncate();
        let splits = cascade_splits(camera.near, o.max_distance, o.cascades, o.split_lambda);
        for (c, slot) in b.iter().enumerate() {
            let p = eye + forward * 0.5 * (splits[c] + splits[c + 1]);
            let clip = (slot.projection * slot.view).project_point3(p);
            assert!(clip.x.abs() < 1.0 && clip.y.abs() < 1.0 && clip.z > 0.0 && clip.z < 1.0, "{c}: {clip:?}");
        }
    }

    #[test]
    fn wgsl_validates_and_matches_the_gpu_layout() {
        let shader = format!(
            "{}\n@fragment fn fs(@builtin(position) p: vec4f) -> @location(0) vec4f {{\n    \
             return vec4f(kansei_sun_shadow(p.xyz, vec3f(0.0, 1.0, 0.0), p.xy) * kansei_cascades.lightColor, 1.0);\n}}\n",
            super::super::CASCADED_SHADOWS_WGSL
        );
        let module = naga::front::wgsl::parse_str(&shader).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&shader)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = |name: &str| {
            module
                .types
                .iter()
                .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                    (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                    _ => None,
                })
                .unwrap()
        };
        assert_eq!(span("KanseiCascade"), std::mem::size_of::<CascadeGpu>());
        assert_eq!(span("KanseiCascades"), CascadedShadowMap::gpu_size());
    }
}
