use bytemuck::{Pod, Zeroable};

use crate::buffers::Texture;
use crate::cameras::Camera;
use crate::math::Vec3;
use crate::renderers::{GBuffer, Renderer};

const RESOLVE_WGSL: &str = include_str!("../shaders/planar_reflection_resolve.wgsl");
const DOWNSAMPLE_WGSL: &str = include_str!("../shaders/planar_reflection_downsample.wgsl");

/// Reflection about the plane `n·p + d = 0` (`n` unit length).
pub fn reflection_matrix(n: glam::Vec3, d: f32) -> glam::Mat4 {
    glam::Mat4::from_cols(
        glam::Vec4::new(1.0 - 2.0 * n.x * n.x, -2.0 * n.x * n.y, -2.0 * n.x * n.z, 0.0),
        glam::Vec4::new(-2.0 * n.x * n.y, 1.0 - 2.0 * n.y * n.y, -2.0 * n.y * n.z, 0.0),
        glam::Vec4::new(-2.0 * n.x * n.z, -2.0 * n.y * n.z, 1.0 - 2.0 * n.z * n.z, 0.0),
        glam::Vec4::new(-2.0 * n.x * d, -2.0 * n.y * d, -2.0 * n.z * d, 1.0),
    )
}

/// Replace the near plane of a `[0, 1]`-depth perspective projection by `clip_plane` (view
/// space; points with `plane · (p, 1) >= 0` are kept), after Lengyel, "Oblique View Frustum
/// Depth Projection and Clipping" (2005). Clipping happens in the rasterizer, so no shader needs
/// a clip distance.
pub fn oblique_near_plane(projection: glam::Mat4, clip_plane: glam::Vec4) -> glam::Mat4 {
    // the frustum corner opposite the plane, which must stay on the far plane (z_ndc = 1)
    let q = projection.inverse() * glam::Vec4::new(clip_plane.x.signum(), clip_plane.y.signum(), 1.0, 1.0);
    let c = clip_plane * (projection.row(3).dot(q) / clip_plane.dot(q));
    let mut m = projection;
    m.x_axis.z = c.x;
    m.y_axis.z = c.y;
    m.z_axis.z = c.z;
    m.w_axis.z = c.w;
    m
}

pub struct PlanarReflectionOptions {
    /// Render-target size; half the canvas is typical.
    pub width: u32,
    pub height: u32,
    /// Draw only renderables whose `layers` intersect this mask (leave out the water itself,
    /// grass, small props).
    pub layer_mask: u32,
    /// Metres the clip plane sits below the reflecting plane, so geometry meeting the water
    /// (shores, posts) reflects without a gap.
    pub clip_bias: f32,
    /// Mip levels of the reflection texture for rough surfaces (1 = mirror only).
    pub mip_levels: u32,
}

impl Default for PlanarReflectionOptions {
    fn default() -> Self {
        Self { width: 960, height: 540, layer_mask: u32::MAX, clip_bias: 0.02, mip_levels: 6 }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveParamsGpu {
    inv_view_proj: [f32; 16],
    camera_pos: [f32; 3],
    _pad0: f32,
    size: [u32; 2],
    _pad1: [u32; 2],
}

/// A mirror view of the scene across a plane (a lake, a wet floor), for materials to sample
/// (K9). The renderer draws every registered reflection each frame, after its shadow maps and
/// before the main pass, from the camera mirrored in the plane, with an oblique near plane at
/// the surface and only the renderables on `layer_mask`. The result goes into a mip-mapped
/// texture: rgb radiance, a = the reflected path length for fog. Sample it in a material with
/// `reflections::PLANAR_REFLECTION_WGSL`.
///
/// ```ignore
/// let reflection = PlanarReflection::new(&renderer, Vec3::new(0.0, 3.0, 0.0), Vec3::UP, PlanarReflectionOptions {
///     width: w / 2, height: h / 2, layer_mask: !WATER_LAYER, ..Default::default()
/// });
/// water_material.set_bindable(1, reflection.material_texture());
/// let lake = renderer.add_planar_reflection(reflection);
/// ```
pub struct PlanarReflection {
    /// A point on the reflecting plane.
    pub plane_point: Vec3,
    /// The plane's normal, pointing to the side that is reflected.
    pub plane_normal: Vec3,
    pub layer_mask: u32,
    pub clip_bias: f32,
    /// Skip rendering (the texture keeps its last contents).
    pub enabled: bool,
    width: u32,
    height: u32,
    active: bool,
    camera: Camera,
    // MRT targets matching the GBuffer, so the materials' GBuffer pipelines draw into them
    color_view: wgpu::TextureView,
    extra_views: [wgpu::TextureView; 3],
    depth_view: wgpu::TextureView,
    // the sampled result, with its mip chain
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    // (the per-mip views are held by the resolve and downsample bind groups)
    resolve_pipeline: wgpu::ComputePipeline,
    resolve_bg: wgpu::BindGroup,
    resolve_params: wgpu::Buffer,
    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bgs: Vec<wgpu::BindGroup>,
}

impl PlanarReflection {
    pub fn new(renderer: &Renderer, plane_point: Vec3, plane_normal: Vec3, options: PlanarReflectionOptions) -> Self {
        let device = renderer.device();
        let (width, height) = (options.width.max(1), options.height.max(1));
        let max_mips = 32 - width.max(height).leading_zeros();
        let mip_levels = options.mip_levels.clamp(1, max_mips);
        let size = wgpu::Extent3d { width, height, depth_or_array_layers: 1 };
        let target = |label: &str, format: wgpu::TextureFormat, usage: wgpu::TextureUsages| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let formats = GBuffer::MRT_FORMATS;
        let color_view = target("PlanarReflection/Color", formats[0], wgpu::TextureUsages::TEXTURE_BINDING);
        let extra_views = [
            target("PlanarReflection/Emissive", formats[1], wgpu::TextureUsages::empty()),
            target("PlanarReflection/Normal", formats[2], wgpu::TextureUsages::empty()),
            target("PlanarReflection/Albedo", formats[3], wgpu::TextureUsages::empty()),
        ];
        let depth_view = target("PlanarReflection/Depth", GBuffer::DEPTH_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING);

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PlanarReflection/Texture"),
            size,
            mip_level_count: mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let mip_views: Vec<_> = (0..mip_levels)
            .map(|level| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("PlanarReflection/Mip"),
                    base_mip_level: level,
                    mip_level_count: Some(1),
                    ..Default::default()
                })
            })
            .collect();

        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba16Float,
            view_dimension: wgpu::TextureViewDimension::D2,
        };
        let resolve_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PlanarReflection/ResolveBGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(2, storage),
                entry(3, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            ],
        });
        let downsample_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PlanarReflection/DownsampleBGL"),
            entries: &[
                entry(0, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                }),
                entry(1, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(2, storage),
            ],
        });
        let pipeline = |label: &str, code: &str, bgl: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let resolve_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PlanarReflection/ResolveParams"),
            size: std::mem::size_of::<ResolveParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let resolve_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PlanarReflection/ResolveBG"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&color_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&depth_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&mip_views[0]) },
                wgpu::BindGroupEntry { binding: 3, resource: resolve_params.as_entire_binding() },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PlanarReflection/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let downsample_bgs = mip_views
            .windows(2)
            .map(|pair| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("PlanarReflection/DownsampleBG"),
                    layout: &downsample_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&pair[0]) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                        wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&pair[1]) },
                    ],
                })
            })
            .collect();

        let mut camera = Camera::new(60.0, 0.1, 1000.0, width as f32 / height as f32);
        renderer.init_camera(&mut camera);

        Self {
            plane_point,
            plane_normal,
            layer_mask: options.layer_mask,
            clip_bias: options.clip_bias,
            enabled: true,
            width,
            height,
            active: false,
            camera,
            color_view,
            extra_views,
            depth_view,
            texture,
            texture_view,
            resolve_pipeline: pipeline("PlanarReflection/Resolve", RESOLVE_WGSL, &resolve_bgl),
            resolve_bg,
            resolve_params,
            downsample_pipeline: pipeline("PlanarReflection/Downsample", DOWNSAMPLE_WGSL, &downsample_bgl),
            downsample_bgs,
        }
    }

    /// The reflection (all mips) as a material bindable: `texture_2d<f32>` in WGSL.
    pub fn material_texture(&self) -> Texture {
        Texture::from_view("PlanarReflection", self.texture.clone(), self.texture_view.clone())
    }

    pub fn set_plane(&mut self, point: Vec3, normal: Vec3) {
        self.plane_point = point;
        self.plane_normal = normal;
    }

    /// Whether the last frame rendered the reflection (the camera was on the reflected side).
    pub fn is_active(&self) -> bool {
        self.active
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// Point the mirrored camera for this frame. Returns false when the main camera is not on the
    /// reflected side of the plane (nothing to render).
    pub(crate) fn update_camera(&mut self, queue: &wgpu::Queue, main: &Camera) -> bool {
        let n = glam::Vec3::new(self.plane_normal.x, self.plane_normal.y, self.plane_normal.z).normalize_or(glam::Vec3::Y);
        let p = glam::Vec3::new(self.plane_point.x, self.plane_point.y, self.plane_point.z);
        let d = -n.dot(p);
        let view = main.view_matrix.to_glam();
        let cam_pos = view.inverse().w_axis.truncate();
        self.active = self.enabled && n.dot(cam_pos) + d > 0.0;
        if !self.active {
            return false;
        }
        let mirrored_view = view * reflection_matrix(n, d);
        // clip plane (lowered by the bias) in the mirrored view space: planes transform by the
        // inverse transpose
        let plane_world = glam::Vec4::new(n.x, n.y, n.z, d + self.clip_bias);
        let plane_view = mirrored_view.inverse().transpose() * plane_world;
        let projection = oblique_near_plane(main.projection_matrix.to_glam(), plane_view);
        // the mirror flips triangle winding; flipping x in clip space flips it back, so the
        // materials' back-face culling still works (samplers flip u back, see the WGSL helper)
        let flip_x = glam::Mat4::from_scale(glam::Vec3::new(-1.0, 1.0, 1.0));
        self.camera.view_matrix = mirrored_view.into();
        self.camera.inverse_view_matrix = mirrored_view.inverse().into();
        self.camera.projection_matrix = (flip_x * projection).into();
        self.camera.upload(queue);
        true
    }

    pub(crate) fn camera(&self) -> &Camera {
        &self.camera
    }

    pub(crate) fn color_attachments(&self) -> [&wgpu::TextureView; 4] {
        [&self.color_view, &self.extra_views[0], &self.extra_views[1], &self.extra_views[2]]
    }

    pub(crate) fn depth_attachment(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Resolve the render into mip 0 (with the path length in alpha), then build the mips.
    pub(crate) fn resolve(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        let view_proj = self.camera.projection_matrix.to_glam() * self.camera.view_matrix.to_glam();
        let cam = self.camera.inverse_view_matrix.to_glam().w_axis;
        let params = ResolveParamsGpu {
            inv_view_proj: view_proj.inverse().to_cols_array(),
            camera_pos: [cam.x, cam.y, cam.z],
            _pad0: 0.0,
            size: [self.width, self.height],
            _pad1: [0; 2],
        };
        queue.write_buffer(&self.resolve_params, 0, bytemuck::bytes_of(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PlanarReflection/Resolve"), ..Default::default() });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &self.resolve_bg, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        pass.set_pipeline(&self.downsample_pipeline);
        for (level, bg) in self.downsample_bgs.iter().enumerate() {
            let (w, h) = ((self.width >> (level + 1)).max(1), (self.height >> (level + 1)).max(1));
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 2] {
        [("planar_reflection_resolve", RESOLVE_WGSL), ("planar_reflection_downsample", DOWNSAMPLE_WGSL)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflection_matrix_mirrors_across_the_plane() {
        // the plane y = 3
        let m = reflection_matrix(glam::Vec3::Y, -3.0);
        assert!(m.transform_point3(glam::Vec3::new(1.0, 5.0, 2.0)).abs_diff_eq(glam::Vec3::new(1.0, 1.0, 2.0), 1e-6));
        assert!(m.transform_point3(glam::Vec3::new(4.0, 3.0, -1.0)).abs_diff_eq(glam::Vec3::new(4.0, 3.0, -1.0), 1e-6));
        assert!((m.determinant() + 1.0).abs() < 1e-6);
        // a tilted plane: reflecting twice is the identity
        let n = glam::Vec3::new(0.3, 0.9, -0.2).normalize();
        let r = reflection_matrix(n, 1.7);
        assert!((r * r).abs_diff_eq(glam::Mat4::IDENTITY, 1e-5));
    }

    /// With the mirrored camera below the plane y = 0 looking up and ahead, the oblique
    /// projection puts the plane at depth 0, keeps what is above it in [0, 1], clips what is
    /// below, and leaves x, y and w (so the image) unchanged.
    #[test]
    fn oblique_projection_clips_at_the_plane() {
        let camera_pos = glam::Vec3::new(0.0, 2.0, 5.0);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::new(0.0, 0.5, -10.0), glam::Vec3::Y);
        let n = glam::Vec3::Y;
        let mirrored = view * reflection_matrix(n, 0.0);
        let proj = glam::Mat4::perspective_rh(1.0, 16.0 / 9.0, 0.1, 500.0);
        let plane_view = mirrored.inverse().transpose() * glam::Vec4::new(0.0, 1.0, 0.0, 0.0);
        let oblique = oblique_near_plane(proj, plane_view);

        let ndc = |p: glam::Vec3| {
            let c = oblique * mirrored * p.extend(1.0);
            c.truncate() / c.w
        };
        let on_plane = ndc(glam::Vec3::new(0.5, 0.0, -8.0));
        assert!(on_plane.z.abs() < 1e-4, "{on_plane:?}");
        let above = ndc(glam::Vec3::new(0.5, 2.0, -20.0));
        assert!(above.z > 0.0 && above.z < 1.0, "{above:?}");
        let below = ndc(glam::Vec3::new(0.5, -1.0, -8.0));
        assert!(below.z < 0.0, "{below:?}");
        // same x/y as the unmodified projection
        let plain = proj * mirrored * glam::Vec3::new(0.5, 2.0, -20.0).extend(1.0);
        assert!(((plain.truncate() / plain.w).truncate() - above.truncate()).length() < 1e-4);
    }

    /// A point on the plane lands at the same screen position in the main view and, flipped
    /// left-right, in the mirrored view: the lookup the WGSL helper does.
    #[test]
    fn plane_points_project_to_mirrored_screen_positions() {
        let camera_pos = glam::Vec3::new(3.0, 4.0, 10.0);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::new(-2.0, 0.0, -10.0), glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(0.8, 1.5, 0.1, 500.0);
        let mirrored = view * reflection_matrix(glam::Vec3::Y, 0.0);
        let flip = glam::Mat4::from_scale(glam::Vec3::new(-1.0, 1.0, 1.0));
        let p = glam::Vec3::new(-1.0, 0.0, -4.0).extend(1.0);
        let main = proj * view * p;
        let refl = flip * proj * mirrored * p;
        assert!((main.x / main.w + refl.x / refl.w).abs() < 1e-5);
        assert!((main.y / main.w - refl.y / refl.w).abs() < 1e-5);
    }

    #[test]
    fn shaders_validate() {
        let helper = format!(
            "{}\n@group(0) @binding(0) var t: texture_2d<f32>;\n@group(0) @binding(1) var s: sampler;\n\
             @fragment fn fs(@location(0) clip: vec4f) -> @location(0) vec4f {{\n    \
             let off = kansei_reflection_offset(mat4x4f(), vec3f(0.0, 1.0, 0.0), vec3f(0.1, 1.0, 0.0), 0.05);\n    \
             return kansei_planar_reflection(t, s, kansei_screen_uv(clip), off, 0.3);\n}}\n",
            super::super::PLANAR_REFLECTION_WGSL
        );
        let sources = PlanarReflection::shader_sources().into_iter().map(|(n, c)| (n, c.to_string())).chain([("planar_reflection_sample", helper)]);
        let mut sizes = std::collections::HashMap::new();
        for (name, code) in sources {
            let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(&code)));
            naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            for (_, ty) in module.types.iter() {
                if let (Some(n), naga::TypeInner::Struct { span, .. }) = (&ty.name, &ty.inner) {
                    sizes.insert(n.clone(), *span as usize);
                }
            }
        }
        assert_eq!(sizes["ResolveParams"], std::mem::size_of::<ResolveParamsGpu>());
    }
}
