use bytemuck::{Pod, Zeroable};

use crate::buffers::Texture;
use crate::cameras::Camera;
use crate::math::Vec3;
use super::screen_space::{ScreenSpaceParamsGpu, ScreenSpaceProjection};
use crate::renderers::{GBuffer, Renderer};

const RESOLVE_WGSL: &str = concat!(include_str!("../shaders/froxel_common.wgsl"), include_str!("../shaders/planar_reflection_resolve.wgsl"));
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

/// The mirrored view of `view` across the plane `n·p + d = 0`, and the projection that renders
/// it with the winding flipped back (x negated in clip space), without the oblique near plane.
pub(crate) fn mirrored_view(view: glam::Mat4, n: glam::Vec3, d: f32) -> glam::Mat4 {
    view * reflection_matrix(n, d)
}

/// Where a box is on screen.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum ScreenRect {
    /// Wholly outside the view.
    Offscreen,
    /// Within this rectangle, in screen uv (x0, y0, x1, y1; y down).
    Rect([f32; 4]),
    /// Unknown (it reaches behind the eye): the whole screen.
    Unbounded,
}

/// The screen rectangle of the box `lo`..`hi` seen through `view_proj`, widened by `margin` (uv)
/// and clamped to the screen.
pub(crate) fn screen_rect(view_proj: glam::Mat4, lo: glam::Vec3, hi: glam::Vec3, margin: f32) -> ScreenRect {
    let (mut min, mut max) = (glam::Vec2::splat(f32::MAX), glam::Vec2::splat(f32::MIN));
    let mut behind = 0;
    for k in 0..8 {
        let corner = glam::Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        let clip = view_proj * corner.extend(1.0);
        if clip.w <= 1e-4 {
            behind += 1;
            continue;
        }
        let ndc = clip.truncate().truncate() / clip.w;
        min = min.min(ndc);
        max = max.max(ndc);
    }
    if behind == 8 {
        return ScreenRect::Offscreen;
    }
    if behind > 0 {
        return ScreenRect::Unbounded;
    }
    // ndc -> uv (y down), widened, clamped
    let (u0, u1) = (min.x * 0.5 + 0.5 - margin, max.x * 0.5 + 0.5 + margin);
    let (v0, v1) = (0.5 - max.y * 0.5 - margin, 0.5 - min.y * 0.5 + margin);
    if u1 <= 0.0 || u0 >= 1.0 || v1 <= 0.0 || v0 >= 1.0 {
        return ScreenRect::Offscreen;
    }
    ScreenRect::Rect([u0.max(0.0), v0.max(0.0), u1.min(1.0), v1.min(1.0)])
}

/// Clip-space crop mapping the ndc rectangle x in [x0, x1], y in [y0, y1] to the whole of it:
/// frustum planes of `crop * view_proj` bound that part of the view.
pub(crate) fn crop(x0: f32, x1: f32, y0: f32, y1: f32) -> glam::Mat4 {
    let (sx, sy) = (2.0 / (x1 - x0).max(1e-6), 2.0 / (y1 - y0).max(1e-6));
    glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::Z,
        glam::Vec4::new(-(x0 + x1) * sx * 0.5, -(y0 + y1) * sy * 0.5, 0.0, 1.0),
    )
}

pub(crate) fn flip_x() -> glam::Mat4 {
    glam::Mat4::from_scale(glam::Vec3::new(-1.0, 1.0, 1.0))
}

/// The volumetric fog as a planar reflection sees it (`VolumetricFogEffect::reflection_fog`): a
/// froxel volume built from the mirrored camera, holding only the fog above the mirror (the main
/// fog already covers the camera's path to the water). Give it to the reflection with
/// `PlanarReflection::set_fog`; its resolve composites the fog over what the mirror saw, so a
/// lake mirrors the glow of beams and lamps in the mist. The volume is the one the fog built in
/// the previous frame, looked up by world position, so it stays in place as the camera moves.
#[derive(Clone)]
pub struct ReflectionFog {
    pub(crate) volume: wgpu::TextureView,
    pub(crate) params: wgpu::Buffer,
    /// Whether the reflection was drawn this frame (it is drawn before the post chain), so the
    /// fog builds no volume for a reflection that is disabled or whose plane the camera is under.
    pub(crate) drawn: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

/// What the resolve needs to look up a `ReflectionFog` volume (the WGSL `ReflectionFogParams`).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ReflectionFogParamsGpu {
    pub view_proj: [f32; 16],
    pub grid_near: f32,
    pub grid_far: f32,
    pub grid_d: f32,
    pub enabled: u32,
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
    /// Scales the distances by which instanced renderables (`InstanceCulling`) choose their LOD
    /// in the mirrored view: below 1 it picks finer LODs than the camera. A mirror sees objects
    /// from below, where coarse LODs built to read from the side (flat cards, dropped detail)
    /// show; 1 (the default) picks the camera's.
    pub lod_distance_scale: f32,
    /// Occlusion culling in the mirrored view (off by default): instanced renderables with
    /// `InstanceCulling::with_occlusion` also skip the instances hidden behind the rest of what
    /// the mirror draws, in two phases as for the camera, against a depth pyramid of the mirror's
    /// own depth. It pays where much of what the mirror sees is hidden (a far shore behind a near
    /// one); it costs a second pass over the reflection's targets, a pyramid and a second cull.
    pub occlusion_culling: bool,
    /// World-space bounds of the reflecting surface (its min and max corners), if known. Materials
    /// sample the reflection by screen position, so it is then drawn only where the surface is
    /// on screen: its pass is scissored to the surface's rectangle (plus `screen_margin`), its
    /// instances are culled to that part of the view, and while the surface is off screen it is
    /// not drawn at all.
    pub surface_bounds: Option<(Vec3, Vec3)>,
    /// Margin round the surface's screen rectangle, in screen uv: room for lookups displaced by
    /// ripples and widened by roughness (0.05 by default).
    pub screen_margin: f32,
    /// Reflect what the camera saw instead of drawing the mirrored view (off by default): last
    /// frame's GBuffer, each pixel above the plane mirrored across it into this frame's view
    /// (pixel-projected reflections), in one compute pass whatever the scene's geometry. What
    /// the screen did not see (above its top edge, behind the camera, hidden from it) reads as
    /// sky, which materials fill from their environment; after a camera cut
    /// (`Camera::reset_motion`) the whole reflection does, for a frame. It needs the GBuffer:
    /// `render_with_postprocessing` or `render_to_gbuffer`, single-sampled.
    pub screen_space: bool,
    width: u32,
    height: u32,
    active: bool,
    // this frame's screen rectangle of the surface, in uv (x0, y0, x1, y1; y down), when bounded
    screen_rect: Option<[f32; 4]>,
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
    resolve_bgl: wgpu::BindGroupLayout,
    resolve_bg: wgpu::BindGroup,
    resolve_params: wgpu::Buffer,
    mip0_view: wgpu::TextureView,
    fog_sampler: wgpu::Sampler,
    // no fog: an empty volume and parameters that say so
    no_fog: ReflectionFog,
    // the attached fog's flag, told each frame whether the reflection is drawn
    fog_drawn: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    downsample_pipeline: wgpu::ComputePipeline,
    downsample_bgs: Vec<wgpu::BindGroup>,
    // the attached fog (the screen-space path binds it itself)
    fog: Option<ReflectionFog>,
    screen_space_projection: Option<ScreenSpaceProjection>,
}

#[allow(clippy::too_many_arguments)]
fn resolve_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    color: &wgpu::TextureView,
    depth: &wgpu::TextureView,
    mip0: &wgpu::TextureView,
    params: &wgpu::Buffer,
    fog_sampler: &wgpu::Sampler,
    fog: &ReflectionFog,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("PlanarReflection/ResolveBG"),
        layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color) },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
            wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(mip0) },
            wgpu::BindGroupEntry { binding: 3, resource: params.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&fog.volume) },
            wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::Sampler(fog_sampler) },
            wgpu::BindGroupEntry { binding: 6, resource: fog.params.as_entire_binding() },
        ],
    })
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
                entry(4, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                }),
                entry(5, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(6, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
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
        let fog_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PlanarReflection/FogSampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });
        let no_fog = ReflectionFog {
            volume: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("PlanarReflection/NoFog"),
                    size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D3,
                    format: wgpu::TextureFormat::Rgba16Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default()),
            // zeroed: enabled = 0
            params: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("PlanarReflection/NoFogParams"),
                size: std::mem::size_of::<ReflectionFogParamsGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM,
                mapped_at_creation: false,
            }),
            drawn: Default::default(),
        };
        let resolve_bg = resolve_bind_group(device, &resolve_bgl, &color_view, &depth_view, &mip_views[0], &resolve_params, &fog_sampler, &no_fog);
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
            lod_distance_scale: 1.0,
            occlusion_culling: false,
            surface_bounds: None,
            screen_margin: 0.05,
            screen_rect: None,
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
            resolve_bgl,
            resolve_bg,
            resolve_params,
            mip0_view: mip_views[0].clone(),
            fog_sampler,
            no_fog,
            fog_drawn: None,
            downsample_pipeline: pipeline("PlanarReflection/Downsample", DOWNSAMPLE_WGSL, &downsample_bgl),
            downsample_bgs,
            screen_space: false,
            fog: None,
            screen_space_projection: None,
        }
    }

    /// Composite a volumetric fog over the reflection (`VolumetricFogEffect::reflection_fog`), or
    /// none. Materials then fog only what lies beyond the fog's volume along the reflected path.
    pub fn set_fog(&mut self, renderer: &Renderer, fog: Option<&ReflectionFog>) {
        self.fog_drawn = fog.map(|f| f.drawn.clone());
        self.fog = fog.cloned();
        let fog = fog.unwrap_or(&self.no_fog);
        self.resolve_bg = resolve_bind_group(
            renderer.device(),
            &self.resolve_bgl,
            &self.color_view,
            &self.depth_view,
            &self.mip0_view,
            &self.resolve_params,
            &self.fog_sampler,
            fog,
        );
    }

    /// The plane as (unit normal, d) with `n·p + d = 0`.
    pub(crate) fn plane(&self) -> (glam::Vec3, f32) {
        let n = glam::Vec3::new(self.plane_normal.x, self.plane_normal.y, self.plane_normal.z).normalize_or(glam::Vec3::Y);
        let p = glam::Vec3::new(self.plane_point.x, self.plane_point.y, self.plane_point.z);
        (n, -n.dot(p))
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
        let (n, d) = self.plane();
        let view = main.view_matrix.to_glam();
        let cam_pos = view.inverse().w_axis.truncate();
        self.active = self.enabled && n.dot(cam_pos) + d > 0.0;
        // only the surface's part of the screen, and nothing while it is off screen
        self.screen_rect = None;
        if self.active {
            if let Some((lo, hi)) = self.surface_bounds {
                let view_proj = main.projection_matrix.to_glam() * view;
                match screen_rect(view_proj, lo.to_glam(), hi.to_glam(), self.screen_margin) {
                    ScreenRect::Offscreen => self.active = false,
                    ScreenRect::Rect(rect) => self.screen_rect = Some(rect),
                    ScreenRect::Unbounded => {}
                }
            }
        }
        if let Some(drawn) = &self.fog_drawn {
            drawn.store(self.active, std::sync::atomic::Ordering::Relaxed);
        }
        if !self.active {
            return false;
        }
        let mirrored_view = mirrored_view(view, n, d);
        // clip plane (lowered by the bias) in the mirrored view space: planes transform by the
        // inverse transpose
        let plane_world = glam::Vec4::new(n.x, n.y, n.z, d + self.clip_bias);
        let plane_view = mirrored_view.inverse().transpose() * plane_world;
        let projection = oblique_near_plane(main.projection_matrix.to_glam(), plane_view);
        // the mirror flips triangle winding; flipping x in clip space flips it back, so the
        // materials' back-face culling still works (samplers flip u back, see the WGSL helper)
        self.camera.view_matrix = mirrored_view.into();
        self.camera.inverse_view_matrix = mirrored_view.inverse().into();
        self.camera.projection_matrix = (flip_x() * projection).into();
        self.camera.upload(queue);
        true
    }

    pub(crate) fn camera(&self) -> &Camera {
        &self.camera
    }

    /// What occlusion culling projects the mirrored view's bounds with: its view and projection
    /// as rasterized, and the targets' size. Its depth pyramid holds view distances: with the
    /// near plane at the water, the depth of what is seen at a grazing angle hardly grows with
    /// its distance, which a depth pyramid could not tell apart.
    pub(crate) fn occlusion_view(&self) -> crate::culling::OcclusionView {
        crate::culling::OcclusionView {
            view: self.camera.view_matrix.to_glam(),
            proj: self.camera.projection_matrix.to_glam(),
            depth_size: (self.width, self.height),
            reverse_z: false,
            linear_depth: true,
        }
    }

    /// The view-projection to cull this frame's instances with: the mirrored camera's, cropped
    /// to the surface's part of the screen.
    pub(crate) fn cull_view_proj(&self) -> glam::Mat4 {
        let view_proj = self.camera.projection_matrix.to_glam() * self.camera.view_matrix.to_glam();
        match self.screen_rect {
            // screen uv -> the mirrored view's ndc: x flipped (see `flip_x`), y up
            Some([u0, v0, u1, v1]) => crop(-(2.0 * u1 - 1.0), -(2.0 * u0 - 1.0), 1.0 - 2.0 * v1, 1.0 - 2.0 * v0) * view_proj,
            None => view_proj,
        }
    }

    /// This frame's scissor rectangle in the render target (x, y, width, height), when bounded.
    pub(crate) fn scissor(&self) -> Option<[u32; 4]> {
        let [u0, v0, u1, v1] = self.screen_rect?;
        let (w, h) = (self.width as f32, self.height as f32);
        // the target is mirrored left-right
        let x0 = ((1.0 - u1) * w).floor().clamp(0.0, w - 1.0) as u32;
        let x1 = ((1.0 - u0) * w).ceil().clamp(x0 as f32 + 1.0, w) as u32;
        let y0 = (v0 * h).floor().clamp(0.0, h - 1.0) as u32;
        let y1 = (v1 * h).ceil().clamp(y0 as f32 + 1.0, h) as u32;
        Some([x0, y0, x1 - x0, y1 - y0])
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
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PlanarReflection/Resolve"), timestamp_writes: crate::profiling::gpu_pass("PlanarReflection/Resolve").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.resolve_pipeline);
        pass.set_bind_group(0, &self.resolve_bg, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
        self.build_mips(&mut pass);
    }

    fn build_mips(&self, pass: &mut wgpu::ComputePass) {
        pass.set_pipeline(&self.downsample_pipeline);
        for (level, bg) in self.downsample_bgs.iter().enumerate() {
            let (w, h) = ((self.width >> (level + 1)).max(1), (self.height >> (level + 1)).max(1));
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(w.div_ceil(8), h.div_ceil(8), 1);
        }
    }

    /// The screen-space path (`screen_space`): project `gbuffer`'s colour and depth, still last
    /// frame's (`camera`'s previous view), into mip 0, then build the mips.
    pub(crate) fn project_screen_space(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, camera: &Camera, gbuffer: &GBuffer) {
        let (width, height) = (self.width, self.height);
        let (n, d) = self.plane();
        let view_proj = camera.view_projection().to_glam();
        let camera_pos = camera.view_matrix.to_glam().inverse().w_axis.truncate();
        // no last frame to project (a camera cut, the first frame): an empty rectangle, all sky
        let (prev, rect) = match screen_space_source(camera) {
            Some(prev) => (prev, self.screen_rect.unwrap_or([0.0, 0.0, 1.0, 1.0])),
            None => (view_proj, [1.0, 1.0, 0.0, 0.0]),
        };
        let params = ScreenSpaceParamsGpu {
            prev_inv_view_proj: prev.inverse().to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            plane: [n.x, n.y, n.z, d],
            camera_pos: camera_pos.to_array(),
            min_height: self.clip_bias.max(0.0),
            src_size: [gbuffer.width, gbuffer.height],
            dst_size: [width, height],
            rect,
        };
        let projection = self.screen_space_projection.get_or_insert_with(|| ScreenSpaceProjection::new(device, width, height));
        let fog = self.fog.as_ref().unwrap_or(&self.no_fog);
        projection.run(device, queue, encoder, &gbuffer.color_view, &gbuffer.depth_view, &self.mip0_view, fog, &self.fog_sampler, &params);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("PlanarReflection/Mips"), timestamp_writes: None });
        self.build_mips(&mut pass);
    }

    #[cfg(test)]
    pub(crate) fn shader_sources() -> [(&'static str, &'static str); 2] {
        [("planar_reflection_resolve", RESOLVE_WGSL), ("planar_reflection_downsample", DOWNSAMPLE_WGSL)]
    }
}

/// The view last frame's GBuffer was drawn with (jittered, with TAA), which the screen-space path
/// reconstructs its pixels' world positions with; None without a last frame.
fn screen_space_source(camera: &Camera) -> Option<glam::Mat4> {
    camera.previous_jittered_view_projection().map(|vp| vp.to_glam())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Last frame's GBuffer was drawn jittered (TAA): its pixels are projected with that jittered
    /// view, or the whole reflection slides by the jitter from frame to frame (a flicker TAA
    /// can't settle).
    #[test]
    fn the_screen_is_projected_with_the_view_it_was_drawn_with() {
        let mut camera = Camera::new(60.0, 0.1, 100.0, 1.5);
        camera.update_projection_matrix();
        camera.update_view_matrix();
        assert!(screen_space_source(&camera).is_none(), "no last frame");
        camera.jitter = [0.002, -0.003];
        let drawn = camera.jittered_projection().to_glam() * camera.view_matrix.to_glam();
        camera.end_frame();
        camera.jitter = [-0.001, 0.004];
        assert!(screen_space_source(&camera).unwrap().abs_diff_eq(drawn, 1e-6));
    }

    /// The mirrored view cropped to a lake's screen rectangle keeps what the lake reflects, and
    /// culls what is reflected beside it.
    #[test]
    fn cropped_mirror_keeps_what_the_surface_reflects() {
        let camera_pos = glam::Vec3::new(0.0, 4.0, 20.0);
        let view = glam::Mat4::look_at_rh(camera_pos, glam::Vec3::new(0.0, 0.0, -10.0), glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(0.8, 1.5, 0.1, 500.0);
        // a lake at y = 0 from x -5..5, z -20..0, seen from above its near shore
        let rect = match screen_rect(proj * view, glam::Vec3::new(-5.0, 0.0, -20.0), glam::Vec3::new(5.0, 0.0, 0.0), 0.0) {
            ScreenRect::Rect(r) => r,
            other => panic!("{other:?}"),
        };
        let [u0, v0, u1, v1] = rect;
        // the mirror as update_camera builds it, cropped as cull_view_proj does
        let mirrored = view * reflection_matrix(glam::Vec3::Y, 0.0);
        let refl = flip_x() * proj * mirrored;
        let planes = crate::culling::frustum_planes(crop(-(2.0 * u1 - 1.0), -(2.0 * u0 - 1.0), 1.0 - 2.0 * v1, 1.0 - 2.0 * v0) * refl);
        let kept = |p: glam::Vec3| planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= 0.0);
        // the lake itself, and a tree top on the far shore whose reflection falls on the lake
        assert!(kept(glam::Vec3::new(0.0, 0.0, -10.0)));
        assert!(kept(glam::Vec3::new(2.0, 3.0, -24.0)));
        // a tree well off to the side, whose reflection falls beside the lake
        assert!(!kept(glam::Vec3::new(40.0, 3.0, -10.0)));
    }

    /// A box's screen rectangle: on screen, beside the view, behind the eye, straddling it; and
    /// the crop of that rectangle bounds exactly its part of the view.
    #[test]
    fn surface_rectangle_and_crop() {
        // looking down -z, 90 degrees: at z = -10 the view spans x, y in [-10, 10]
        let view_proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0) * glam::Mat4::look_at_rh(glam::Vec3::ZERO, glam::Vec3::NEG_Z, glam::Vec3::Y);
        let rect = |lo: [f32; 3], hi: [f32; 3]| screen_rect(view_proj, lo.into(), hi.into(), 0.0);
        let ScreenRect::Rect([u0, v0, u1, v1]) = rect([1.0, -1.0, -10.0], [3.0, 1.0, -10.0]) else { panic!() };
        for (got, want) in [(u0, 0.55), (u1, 0.65), (v0, 0.45), (v1, 0.55)] {
            assert!((got - want).abs() < 1e-5, "{got} vs {want}");
        }
        assert_eq!(rect([20.0, -1.0, -10.0], [30.0, 1.0, -10.0]), ScreenRect::Offscreen, "beside the view");
        assert_eq!(rect([-1.0, -1.0, 5.0], [1.0, 1.0, 10.0]), ScreenRect::Offscreen, "behind the eye");
        assert_eq!(rect([-1.0, -1.0, -10.0], [1.0, 1.0, 10.0]), ScreenRect::Unbounded, "round the eye");
        // the crop of ndc x in [0.1, 0.3], y in [-0.1, 0.1] keeps the box's part of the view only
        let planes = crate::culling::frustum_planes(crop(0.1, 0.3, -0.1, 0.1) * view_proj);
        let inside = |p: glam::Vec3| planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= 0.0);
        assert!(inside(glam::Vec3::new(2.0, 0.0, -10.0)));
        assert!(inside(glam::Vec3::new(4.0, 0.0, -20.0)), "farther along the same rays");
        assert!(!inside(glam::Vec3::new(0.0, 0.0, -10.0)));
        assert!(!inside(glam::Vec3::new(4.0, 0.0, -10.0)));
        assert!(!inside(glam::Vec3::new(2.0, 2.0, -10.0)));
    }

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
        assert_eq!(sizes["ReflectionFogParams"], std::mem::size_of::<ReflectionFogParamsGpu>());
    }
}
