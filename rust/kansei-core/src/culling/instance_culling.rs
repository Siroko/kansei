use bytemuck::{Pod, Zeroable};

use crate::buffers::{BufferType, ComputeBuffer};

const WGSL: &str = include_str!("../shaders/instance_cull.wgsl");
const NO_WORD: u32 = u32::MAX;

/// World-space frustum planes of a `[0, 1]`-depth view-projection (Gribb & Hartmann), as
/// `(normal, d)` with `normal · p + d >= 0` inside, normals unit length. Order: left, right,
/// bottom, top, near, far.
pub fn frustum_planes(view_proj: glam::Mat4) -> [glam::Vec4; 6] {
    let (r0, r1, r2, r3) = (view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| p / p.truncate().length().max(1e-12))
}

/// Whether the box `[min, max]` is at least partly inside the frustum `planes`
/// ([`frustum_planes`]): false only when it lies wholly outside one plane, so a box near a corner
/// of the frustum may be kept though nothing of it shows (conservative, as culling must be).
pub fn aabb_in_frustum(planes: &[glam::Vec4; 6], min: glam::Vec3, max: glam::Vec3) -> bool {
    planes.iter().all(|p| {
        // the box's corner furthest along the plane's normal
        let far = glam::Vec3::select(p.truncate().cmpge(glam::Vec3::ZERO), max, min);
        p.truncate().dot(far) + p.w >= 0.0
    })
}

/// Bytes of a view's indirect draw: `DrawIndexedIndirect`'s five words, then the instances culled
/// by the LOD band, the frustum and occlusion (counted when the renderer's culling stats are on).
pub(crate) const ARGS_BYTES: u64 = 32;

/// Per-view GPU culling for an instanced renderable (set `Renderable::instance_culling`).
///
/// `source` holds every instance, never culled: the per-instance vertex data of the geometry's
/// (single) instance buffer, `stride` bytes each. Every frame the renderer culls it on the GPU
/// for each view that draws the renderable: the main camera, and each shadowed spot light's own
/// frustum, so things outside the picture still cast into it. An instance is kept when its
/// bounding sphere (centre at `center_offset`, `radius` times the f32 at
/// `radius_scale_offset` if set, both in the renderable's object space) is inside the view, and
/// its distance from the main camera is within `lod_range`. The survivors are compacted, a
/// region per view, and drawn indirectly; the geometry's own instance buffer only lends its
/// vertex layout. One dispatch culls all of a renderable's views.
///
/// LOD: one renderable per LOD mesh, all sharing `source`, each with its distance band. Bands
/// are measured from the main camera in every view, so shadows match what is on screen. Shadow
/// maps, planar reflections and voxel GI can have bands of their own (`with_shadow_lod_range`,
/// `with_reflection_lod_range`, `with_gi_lod_range`): a far LOD (an impostor) nearer there than on
/// screen, say. Keep each kind of view's bands of a mesh's LODs adjoining, as the camera's.
///
/// Voxel GI's clipmap (`Renderer::enable_voxel_clipmap`) voxelizes a renderable with a
/// `Renderable::gi` surface through its own view too: the box of the region it voxelizes, its
/// instances compacted as the camera's are (crossfaded ones with their fade, the layout the
/// material reads).
///
/// Tighter bounds (`with_bounds_shift`, `with_bounds_box`) cull more, in every view, and matter
/// most for occlusion: a tree's sphere round its base reaches a tree's height below the ground
/// and to each side.
///
/// Occlusion (`with_occlusion(true)`): the camera's view, and planar reflections that ask for it
/// (`PlanarReflection::occlusion_culling`), also skip instances hidden behind the depth of the
/// rest of what they draw, in two phases per frame (see `Renderer::set_occlusion_culling`).
/// Shadow maps stay frustum-only (an instance hidden from the camera may still cast a shadow
/// into the picture).
///
/// ```ignore
/// // 32-byte instances: position xyz + height, then yaw, ...; spheres of 0.6 x height
/// let culling = InstanceCulling::new(all_trees.clone(), tree_count, 32, 0, 0.6)
///     .with_radius_scale(12)
///     .with_lod_range(0.0, 60.0);
/// lod0.instance_culling = Some(culling);
///
/// // the same trees, with a box from the ground to the top of a tree (x its height) and
/// // occlusion culling for the camera
/// let culling = InstanceCulling::new(all_trees.clone(), tree_count, 32, 0, 0.6)
///     .with_radius_scale(12)
///     .with_bounds_shift(glam::Vec3::new(0.0, 0.5, 0.0))
///     .with_bounds_box(glam::Vec3::new(0.25, 0.5, 0.25))
///     .with_occlusion(true);
/// ```
pub struct InstanceCulling {
    /// All instances: a handle to the geometry's instance buffer (a clone of the same
    /// `ComputeBuffer`), created on first use; needs `BufferUsages::STORAGE`.
    pub source: ComputeBuffer,
    /// Instances in `source` (at most the count it was created with).
    pub count: u32,
    /// Bytes per instance, a multiple of 4.
    pub stride: u32,
    /// Byte offset of the instance's centre (3 x f32) within an instance.
    pub center_offset: u32,
    /// Bounding radius, object space.
    pub radius: f32,
    /// Byte offset of an f32 the bounds are multiplied by (a per-instance scale or height).
    pub radius_scale_offset: Option<u32>,
    /// Distances from the main camera at which the instances draw here: [near, far).
    pub lod_range: (f32, f32),
    /// The band in shadow maps (spot lights, cascades), if not `lod_range`.
    pub shadow_lod_range: Option<(f32, f32)>,
    /// The band in planar reflections, if not `lod_range` (their `lod_distance_scale` applies to
    /// it as to `lod_range`).
    pub reflection_lod_range: Option<(f32, f32)>,
    /// The band in voxel GI's views (`Renderer::enable_voxel_clipmap`), if not `lod_range`: a
    /// coarse LOD voxelizes the forest at every distance, a fine one near the camera, and a LOD
    /// that should stay out of the voxels (an impostor) gets an empty band (`near >= far`: no
    /// instance, whatever the crossfades).
    pub gi_lod_range: Option<(f32, f32)>,
    /// Object-space offset of the bounds' centre from the instance's centre, times the per-instance
    /// scale: where the sphere (or box) sits. It moves the centre the LOD distance is measured
    /// from too. Instances' own rotations are not applied: use it along an axis they turn about.
    pub bounds_shift: glam::Vec3,
    /// Object-space half extents of a box (times the per-instance scale, round the shifted centre)
    /// to cull with instead of the sphere. Instances' own rotations are not applied: make it wide
    /// enough for them (for instances turned about y, equal x and z extents of the widest reach).
    pub bounds_box: Option<glam::Vec3>,
    /// Occlusion culling for the main camera, in two phases (off by default).
    pub occlusion: bool,
    /// Width of the crossfades at the LOD bands' edges, in LOD distance (0: none); see
    /// `with_crossfade`. Turning them on or off changes the compacted instances' layout.
    pub crossfade: f32,
    capacity: u32,
    shared: Option<Shared>,
    occlusion_slots: Vec<OcclusionSlot>,
    /// the views culled in two phases this frame
    two_phase: Vec<usize>,
}

/// Consecutive views culled by one dispatch: their compacted instances, a region of `capacity`
/// instances each in one buffer, and the bind group (with the chunk's parameters).
struct Chunk {
    first_view: usize,
    views: usize,
    instances: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

/// The renderable's culling state for every view: the instances' parameters (a copy per chunk,
/// then one per view for its occlusion phases, `params_stride` apart), written only when they
/// change; the indirect draws, `ARGS_BYTES` apart in one buffer that a frame resets with one
/// clear (each view's, then each view's second occlusion phase's); and the views' compacted
/// instances, in chunks as large as a storage binding allows (one, but for very many instances
/// and views).
struct Shared {
    params: wgpu::Buffer,
    params_stride: u64,
    written: Option<CullInstancesGpu>,
    args: wgpu::Buffer,
    views: usize,
    chunks: Vec<Chunk>,
    /// bytes per compacted instance the buffers were made for
    out_stride: u32,
}

/// Occlusion culling's per-renderable state for a view culled in two phases: which instances it
/// saw last frame, the first phase's bind group (the view's region, and `visibility`), and the
/// second phase's own instances and bind group.
struct OcclusionSlot {
    view: usize,
    visibility: wgpu::Buffer,
    early: wgpu::BindGroup,
    late_instances: wgpu::Buffer,
    late: wgpu::BindGroup,
}

/// A culled draw: the compacted instances (for the geometry's instance buffer, from
/// `instances_offset`) and the indirect draw, at `offset` in `args`.
#[derive(Clone, Copy)]
pub(crate) struct CulledDraw<'a> {
    pub instances: &'a wgpu::Buffer,
    pub instances_offset: u64,
    pub args: &'a wgpu::Buffer,
    pub offset: u64,
}

/// `CullInstances` in instance_cull.wgsl: a renderable's instances.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Pod, Zeroable)]
pub(crate) struct CullInstancesGpu {
    world: [f32; 16],
    shift: [f32; 3],
    lod_near: f32,
    box_half: [f32; 3],
    lod_far: f32,
    radius: f32,
    max_scale: f32,
    count: u32,
    stride_words: u32,
    center_word: u32,
    scale_word: u32,
    flags: u32,
    index_count: u32,
    first_view: u32,
    capacity: u32,
    late_slot: u32,
    layers: u32,
    shadow_lod: [f32; 2],
    reflection_lod: [f32; 2],
    occlusion_view: u32,
    crossfade: f32,
    gi_lod: [f32; 2],
}

/// `CullView` in instance_cull.wgsl: a view, shared by every renderable culled for it.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullViewGpu {
    planes: [[f32; 4]; 6],
    view: [f32; 16],
    proj: [f32; 16],
    lod_origin: [f32; 3],
    flags: u32,
    depth_size: [f32; 2],
    lod_scale: f32,
    layer_mask: u32,
}

const FLAG_STATS: u32 = 1;
const FLAG_BOX: u32 = 2;
const FLAG_REVERSE_Z: u32 = 4;
const FLAG_CASTS_SHADOW: u32 = 8;
const FLAG_TWO_PHASE: u32 = 16;
const FLAG_VIEW: u32 = 32;
const FLAG_CASTERS_ONLY: u32 = 64;
const FLAG_LAYERED: u32 = 128;
const FLAG_REFLECTION: u32 = 256;
const FLAG_OCCLUSION: u32 = 512;
const FLAG_LINEAR_DEPTH: u32 = 1024;
const FLAG_GI: u32 = 2048;
const FLAG_GI_SURFACE: u32 = 4096;

/// A view the renderer culls for: its view-projection, whether it only draws shadow casters (a
/// shadow map, with `shadow_lod_range`), whether it is a planar reflection (with
/// `reflection_lod_range`) or voxel GI's (renderables with a GI surface, with `gi_lod_range`),
/// the layers it draws if not all (a reflection's `layer_mask`), and how it scales the LOD
/// distances (below 1 a view picks finer LODs than the camera would).
#[derive(Clone, Copy, Debug)]
pub(crate) struct CullView {
    pub view_proj: glam::Mat4,
    pub casters_only: bool,
    pub reflection: bool,
    pub gi: bool,
    pub layer_mask: Option<u32>,
    pub lod_distance_scale: f32,
}

/// What occlusion culling projects bounds with: the view, its projection as rasterized
/// (jittered), and the depth buffer's size in pixels; and whether the depth pyramid holds view
/// distances (`DepthPyramid::build_linear`) rather than depths.
#[derive(Clone, Copy, Debug)]
pub(crate) struct OcclusionView {
    pub view: glam::Mat4,
    pub proj: glam::Mat4,
    pub depth_size: (u32, u32),
    pub reverse_z: bool,
    pub linear_depth: bool,
}

impl CullView {
    /// Whether it draws a renderable that casts shadows or not, on `layers`, with a GI surface or
    /// not (the cull shader's `drawnIn`, less the camera's two phases).
    pub(crate) fn draws(&self, casts_shadow: bool, layers: u32, gi_surface: bool) -> bool {
        (!self.casters_only || casts_shadow) && (!self.gi || gi_surface) && self.layer_mask.is_none_or(|mask| mask & layers != 0)
    }

    /// The view for the GPU: its frustum, the LOD origin, and for a view culled in two phases
    /// this frame what occlusion culling projects with.
    pub(crate) fn gpu(&self, lod_origin: glam::Vec3, occlusion: Option<&OcclusionView>, stats: bool) -> CullViewGpu {
        let mut flags = FLAG_VIEW;
        if self.casters_only {
            flags |= FLAG_CASTERS_ONLY;
        }
        if self.layer_mask.is_some() {
            flags |= FLAG_LAYERED;
        }
        if self.reflection {
            flags |= FLAG_REFLECTION;
        }
        if self.gi {
            flags |= FLAG_GI;
        }
        if stats {
            flags |= FLAG_STATS;
        }
        if occlusion.is_some_and(|o| o.reverse_z) {
            flags |= FLAG_REVERSE_Z;
        }
        if occlusion.is_some() {
            flags |= FLAG_OCCLUSION;
        }
        if occlusion.is_some_and(|o| o.linear_depth) {
            flags |= FLAG_LINEAR_DEPTH;
        }
        let (view, proj, depth_size) = match occlusion {
            Some(o) => (o.view, o.proj, [o.depth_size.0 as f32, o.depth_size.1 as f32]),
            None => (glam::Mat4::IDENTITY, glam::Mat4::IDENTITY, [1.0, 1.0]),
        };
        CullViewGpu {
            planes: frustum_planes(self.view_proj).map(|p| p.to_array()),
            view: view.to_cols_array(),
            proj: proj.to_cols_array(),
            lod_origin: lod_origin.to_array(),
            flags,
            depth_size,
            // a band [near, far) at distance x scale is the band [near, far) / scale at x
            lod_scale: self.lod_distance_scale.max(1e-3),
            layer_mask: self.layer_mask.unwrap_or(u32::MAX),
        }
    }
}

impl InstanceCulling {
    /// Culling for `count` instances of `stride` bytes in a GPU buffer created elsewhere (a
    /// simulation's). For an instance buffer of your own, `from_buffer` with the geometry's
    /// `ComputeBuffer`.
    pub fn new(source: wgpu::Buffer, count: u32, stride: u32, center_offset: u32, radius: f32) -> Self {
        Self::from_buffer(&ComputeBuffer::from_external("InstanceCulling/Source", source, BufferType::Storage), count, stride, center_offset, radius)
    }

    /// Culling for `count` instances of `stride` bytes in `source`, the `ComputeBuffer` the
    /// geometry reads them from (this keeps a handle to the same GPU buffer), each a sphere of
    /// `radius` around the 3 floats at `center_offset`.
    pub fn from_buffer(source: &ComputeBuffer, count: u32, stride: u32, center_offset: u32, radius: f32) -> Self {
        let source = source.clone();
        assert!(stride.is_multiple_of(4) && center_offset.is_multiple_of(4) && center_offset + 12 <= stride, "instance layout must be 4-byte words, with the centre inside");
        Self {
            source,
            count,
            stride,
            center_offset,
            radius,
            radius_scale_offset: None,
            lod_range: (0.0, f32::INFINITY),
            shadow_lod_range: None,
            reflection_lod_range: None,
            gi_lod_range: None,
            bounds_shift: glam::Vec3::ZERO,
            bounds_box: None,
            occlusion: false,
            crossfade: 0.0,
            capacity: count,
            shared: None,
            occlusion_slots: Vec::new(),
            two_phase: Vec::new(),
        }
    }

    /// Culling for instances of one vec4 each, position in xyz and a scale in w that multiplies
    /// `radius` (the layout of `materials::StandardInstancing`): `from_buffer(source, count, 16, 0,
    /// radius).with_radius_scale(12)`.
    pub fn for_vec4_instances(source: &ComputeBuffer, count: u32, radius: f32) -> Self {
        Self::from_buffer(source, count, 16, 0, radius).with_radius_scale(12)
    }

    pub fn with_radius_scale(mut self, offset: u32) -> Self {
        assert!(offset.is_multiple_of(4) && offset + 4 <= self.stride);
        self.radius_scale_offset = Some(offset);
        self
    }

    pub fn with_lod_range(mut self, near: f32, far: f32) -> Self {
        self.lod_range = (near, far);
        self
    }

    /// See `shadow_lod_range`.
    pub fn with_shadow_lod_range(mut self, near: f32, far: f32) -> Self {
        self.shadow_lod_range = Some((near, far));
        self
    }

    /// See `reflection_lod_range`.
    pub fn with_reflection_lod_range(mut self, near: f32, far: f32) -> Self {
        self.reflection_lod_range = Some((near, far));
        self
    }

    /// See `gi_lod_range`. An empty band (`near >= far`) leaves the renderable out of the voxels.
    pub fn with_gi_lod_range(mut self, near: f32, far: f32) -> Self {
        self.gi_lod_range = Some((near, far));
        self
    }

    /// See `bounds_shift`.
    pub fn with_bounds_shift(mut self, shift: glam::Vec3) -> Self {
        self.bounds_shift = shift;
        self
    }

    /// See `bounds_box`.
    pub fn with_bounds_box(mut self, half_extents: glam::Vec3) -> Self {
        self.bounds_box = Some(half_extents);
        self
    }

    /// See `occlusion`.
    pub fn with_occlusion(mut self, occlusion: bool) -> Self {
        self.occlusion = occlusion;
        self
    }

    /// Dithered crossfades between LODs, over `width` of LOD distance (metres, times the view's
    /// LOD distance scale): each band's edges widen by `width / 2`, and there the instances draw
    /// in both LODs, each keeping a complementary share of the pixels, so an instance moving
    /// across a band's edge dissolves from one LOD into the other instead of snapping, in every
    /// view (the camera, shadow maps, reflections), by that view's band. A band from 0 has no
    /// near crossfade, and one to infinity no far one: an impostor LOD fades in from the meshes.
    /// Give every LOD of a set the same width, narrower than its bands.
    ///
    /// Each compacted instance is then followed by its fade, an f32: declare the instance
    /// buffer's vertex layout `stride + 4` bytes wide with the fade as an attribute at offset
    /// `stride`, pass it to the fragment shader, and drop pixels with
    /// `culling::LOD_FADE_WGSL`'s `kansei_lod_fade_discard` (in the shadow fragment too, for the
    /// shadows to dissolve). An impostor baked from such a LOD draws it with that layout too: end
    /// `ImpostorOptions::instance` with a fade of 1.
    ///
    /// The fade depends only on the distance, so there is no switch to flip back and forth: an
    /// instance moving to and fro across a band's edge dissolves to and fro by as much as it
    /// moves, one held there stays part dissolved, and temporal antialiasing blends the dither.
    pub fn with_crossfade(mut self, width: f32) -> Self {
        self.crossfade = width.max(0.0);
        self
    }

    /// Bytes of a compacted instance: the source's, and with crossfades its fade.
    pub fn culled_stride(&self) -> u32 {
        self.stride + if self.crossfade > 0.0 { 4 } else { 0 }
    }

    /// The compacted instances and indirect draw of `view` (0 is the main camera), once culled;
    /// with occlusion, the first phase's.
    pub(crate) fn view(&self, view: usize) -> Option<CulledDraw<'_>> {
        let shared = self.shared.as_ref()?;
        let chunk = shared.chunks.iter().find(|c| (c.first_view..c.first_view + c.views).contains(&view))?;
        Some(CulledDraw {
            instances: &chunk.instances,
            instances_offset: ((view - chunk.first_view) as u64) * self.region_bytes(),
            args: &shared.args,
            offset: view as u64 * ARGS_BYTES,
        })
    }

    /// The second phase's compacted instances and indirect draw in `view`, when this frame culls
    /// it in two phases.
    pub(crate) fn late(&self, view: usize) -> Option<CulledDraw<'_>> {
        let shared = self.shared.as_ref()?;
        let slot = self.occlusion_slots.iter().find(|o| o.view == view).filter(|_| self.two_phase.contains(&view))?;
        Some(CulledDraw { instances: &slot.late_instances, instances_offset: 0, args: &shared.args, offset: (shared.views + view) as u64 * ARGS_BYTES })
    }

    /// Whether this frame culls `view` in two phases (set by the renderer).
    pub(crate) fn two_phase_in(&self, view: usize) -> bool {
        self.two_phase.contains(&view)
    }

    /// Cull `views` in two phases this frame (those with occlusion state), the others by frustum.
    pub(crate) fn set_two_phase(&mut self, views: &[usize]) {
        self.two_phase = views.iter().copied().filter(|v| self.occlusion_slots.iter().any(|o| o.view == *v)).collect();
    }

    /// The instances a dispatch tests.
    pub(crate) fn tested(&self) -> u32 {
        self.count.min(self.capacity)
    }

    /// Bytes of a view's region of compacted instances.
    fn region_bytes(&self) -> u64 {
        self.capacity.max(1) as u64 * self.culled_stride() as u64
    }

    fn instances_buffer(&self, device: &wgpu::Device, views: usize) -> wgpu::Buffer {
        // (COPY_SRC: readable for debugging and tests)
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("InstanceCulling/Instances"),
            size: views as u64 * self.region_bytes(),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// A bind group: chunk `chunk`'s parameters, the source, compacted instances `instances`,
    /// every draw, and with occlusion the visibility.
    fn bind_group(&self, device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, chunk: usize, instances: &wgpu::Buffer, visibility: Option<&wgpu::Buffer>) -> wgpu::BindGroup {
        let shared = self.shared.as_ref().expect("shared buffers first");
        let params = wgpu::BufferBinding { buffer: &shared.params, offset: chunk as u64 * shared.params_stride, size: std::num::NonZeroU64::new(std::mem::size_of::<CullInstancesGpu>() as u64) };
        let mut entries = vec![
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(params) },
            wgpu::BindGroupEntry { binding: 1, resource: self.source.gpu_buffer().expect("ensure_views creates the source").as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: instances.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 3, resource: shared.args.as_entire_binding() },
        ];
        if let Some(visibility) = visibility {
            entries.push(wgpu::BindGroupEntry { binding: 4, resource: visibility.as_entire_binding() });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("InstanceCulling/BG"), layout: bgl, entries: &entries })
    }

    /// Make sure there are GPU slots for `count` views; true if buffers were (re)created (render
    /// bundles that recorded the old ones are stale).
    pub(crate) fn ensure_views(&mut self, device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, count: usize) -> bool {
        let limits = device.limits();
        let max_chunk_bytes = (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size);
        self.ensure_views_within(device, bgl, count, max_chunk_bytes)
    }

    /// `ensure_views`, with at most `max_chunk_bytes` of compacted instances per chunk.
    fn ensure_views_within(&mut self, device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, count: usize, max_chunk_bytes: u64) -> bool {
        // the source, unless the geometry sharing it has been drawn already
        self.source.initialize(device);
        if self.count <= self.capacity && self.shared.as_ref().is_some_and(|s| s.views >= count && s.out_stride == self.culled_stride()) {
            return false;
        }
        // recreate everything: the draws hold a slot per view and the second phase's
        self.capacity = self.capacity.max(self.count);
        let count = count.max(self.shared.as_ref().map_or(0, |s| s.views)).max(1);
        self.occlusion_slots.clear();
        self.two_phase.clear();
        let per_chunk = (max_chunk_bytes / self.region_bytes()).clamp(1, count as u64) as usize;
        let chunks = count.div_ceil(per_chunk);
        let align = device.limits().min_uniform_buffer_offset_alignment as u64;
        let params_stride = (std::mem::size_of::<CullInstancesGpu>() as u64).div_ceil(align) * align;
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        self.shared = Some(Shared {
            params: buffer("InstanceCulling/Params", (chunks + count) as u64 * params_stride, wgpu::BufferUsages::UNIFORM),
            params_stride,
            written: None,
            // (COPY_SRC: the stats read them back)
            args: buffer("InstanceCulling/Args", 2 * count as u64 * ARGS_BYTES, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            views: count,
            chunks: Vec::new(),
            out_stride: self.culled_stride(),
        });
        let chunks: Vec<Chunk> = (0..chunks)
            .map(|k| {
                let first_view = k * per_chunk;
                let views = per_chunk.min(count - first_view);
                let instances = self.instances_buffer(device, views);
                let bind_group = self.bind_group(device, bgl, k, &instances, None);
                Chunk { first_view, views, instances, bind_group }
            })
            .collect();
        self.shared.as_mut().unwrap().chunks = chunks;
        true
    }

    /// Make sure there is occlusion state for each of `views` (after `ensure_views`); true if
    /// any was created.
    pub(crate) fn ensure_occlusion(&mut self, device: &wgpu::Device, pipeline: &CullPipeline, views: &[usize]) -> bool {
        let Some(shared) = self.shared.as_ref() else { return false };
        let chunk_count = shared.chunks.len();
        let mut created = false;
        for &view in views {
            if view >= shared.views || self.occlusion_slots.iter().any(|o| o.view == view) {
                continue;
            }
            // one word per instance, zero (nothing seen yet)
            let visibility = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("InstanceCulling/Visibility"),
                size: self.capacity.max(1) as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // the first phase fills the view's region (in its chunk), the second its own buffer;
            // both read the view's copy of the parameters
            let chunk = shared.chunks.iter().find(|c| (c.first_view..c.first_view + c.views).contains(&view)).unwrap();
            let early = self.bind_group(device, &pipeline.occlusion_bgl, chunk_count + view, &chunk.instances, Some(&visibility));
            let late_instances = self.instances_buffer(device, 1);
            let late = self.bind_group(device, &pipeline.occlusion_bgl, chunk_count + view, &late_instances, Some(&visibility));
            self.occlusion_slots.push(OcclusionSlot { view, visibility, early, late_instances, late });
            created = true;
        }
        if created {
            // the new views' copies of the parameters
            self.shared.as_mut().unwrap().written = None;
        }
        created
    }

    /// Forget which instances were visible (a camera cut): the next first phase draws none of
    /// them, and the second tests them all.
    pub(crate) fn reset_visibility(&self, encoder: &mut wgpu::CommandEncoder) {
        for o in &self.occlusion_slots {
            encoder.clear_buffer(&o.visibility, 0, None);
        }
    }

    /// Start a frame, before its cull pass (after `ensure_views` and `set_two_phase`): write the
    /// instances' parameters for a renderable with `world` matrix and `index_count` indices,
    /// drawn in shadow maps if `casts_shadow`, on `layers`, in voxel GI's views if `gi_surface`,
    /// if they changed, and clear every slot's draw (the cull sets the index count of those it
    /// culls into).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn begin_frame(&mut self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, world: glam::Mat4, index_count: u32, casts_shadow: bool, layers: u32, gi_surface: bool) {
        let scale = glam::Vec3::new(world.x_axis.truncate().length(), world.y_axis.truncate().length(), world.z_axis.truncate().length());
        let band = |(near, far): (f32, f32)| [near, far.min(f32::MAX)];
        let mut flags = 0;
        for (on, flag) in [(self.bounds_box.is_some(), FLAG_BOX), (casts_shadow, FLAG_CASTS_SHADOW), (!self.two_phase.is_empty(), FLAG_TWO_PHASE), (gi_surface, FLAG_GI_SURFACE)] {
            if on {
                flags |= flag;
            }
        }
        let params = CullInstancesGpu {
            world: world.to_cols_array(),
            shift: self.bounds_shift.to_array(),
            lod_near: self.lod_range.0,
            box_half: self.bounds_box.unwrap_or(glam::Vec3::ZERO).to_array(),
            lod_far: self.lod_range.1.min(f32::MAX),
            radius: self.radius,
            max_scale: scale.max_element(),
            count: self.tested(),
            stride_words: self.stride / 4,
            center_word: self.center_offset / 4,
            scale_word: self.radius_scale_offset.map_or(NO_WORD, |o| o / 4),
            flags,
            index_count,
            first_view: 0,
            capacity: self.capacity,
            late_slot: 0,
            layers,
            shadow_lod: band(self.shadow_lod_range.unwrap_or(self.lod_range)),
            reflection_lod: band(self.reflection_lod_range.unwrap_or(self.lod_range)),
            occlusion_view: 0,
            crossfade: self.crossfade,
            // an empty band is nowhere, crossfades and all
            gi_lod: band(match self.gi_lod_range {
                Some((near, far)) if near >= far => (f32::MAX, f32::MAX),
                band => band.unwrap_or(self.lod_range),
            }),
        };
        let shared = self.shared.as_mut().expect("ensure_views first");
        if shared.written != Some(params) {
            // a copy per chunk, each with its first view; then one per view culled in two phases,
            // with the view, its chunk's first view and its second phase's draw
            let stride = shared.params_stride as usize;
            let mut bytes = vec![0u8; (shared.chunks.len() + shared.views) * stride];
            let mut put = |k: usize, params: CullInstancesGpu| bytes[k * stride..][..std::mem::size_of::<CullInstancesGpu>()].copy_from_slice(bytemuck::bytes_of(&params));
            for (k, chunk) in shared.chunks.iter().enumerate() {
                put(k, CullInstancesGpu { first_view: chunk.first_view as u32, ..params });
            }
            for o in &self.occlusion_slots {
                let first_view = shared.chunks.iter().find(|c| (c.first_view..c.first_view + c.views).contains(&o.view)).unwrap().first_view;
                put(shared.chunks.len() + o.view, CullInstancesGpu { first_view: first_view as u32, occlusion_view: o.view as u32, late_slot: (shared.views + o.view) as u32, ..params });
            }
            queue.write_buffer(&shared.params, 0, &bytes);
            shared.written = Some(params);
        }
        encoder.clear_buffer(&shared.args, 0, None);
    }

    fn workgroups(&self) -> u32 {
        self.tested().div_ceil(64).max(1)
    }

    /// Cull into every view the renderable is drawn in, a dispatch per chunk (with the
    /// frustum-only pipeline and the views' group 1 set). The views culled in two phases this
    /// frame are left to `dispatch_early` and `dispatch_late`.
    pub(crate) fn dispatch(&self, pass: &mut wgpu::ComputePass) {
        for chunk in &self.shared.as_ref().expect("ensure_views first").chunks {
            pass.set_bind_group(0, &chunk.bind_group, &[]);
            pass.dispatch_workgroups(self.workgroups(), chunk.views as u32, 1);
        }
    }

    fn slot(&self, view: usize) -> &OcclusionSlot {
        self.occlusion_slots.iter().find(|o| o.view == view).expect("ensure_occlusion for the view first")
    }

    /// Occlusion's first phase in `view` (with the `early` pipeline and the views' group 1 set).
    pub(crate) fn dispatch_early(&self, pass: &mut wgpu::ComputePass, view: usize) {
        pass.set_bind_group(0, &self.slot(view).early, &[]);
        pass.dispatch_workgroups(self.workgroups(), 1, 1);
    }

    /// Occlusion's second phase in `view` (with the `late` pipeline, the views' group 1 and the
    /// view's pyramid's group 2 set).
    pub(crate) fn dispatch_late(&self, pass: &mut wgpu::ComputePass, view: usize) {
        pass.set_bind_group(0, &self.slot(view).late, &[]);
        pass.dispatch_workgroups(self.workgroups(), 1, 1);
    }
}

/// The cull compute pipelines, shared by every renderable: frustum and LOD only (`pipeline`), and
/// occlusion's two phases; and the frame's views, in one buffer (`set_views`, `view_bind_group`).
pub(crate) struct CullPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub early: wgpu::ComputePipeline,
    pub late: wgpu::ComputePipeline,
    /// params, source, compacted instances, indirect draw
    pub bgl: wgpu::BindGroupLayout,
    /// the same, and the visibility
    pub occlusion_bgl: wgpu::BindGroupLayout,
    /// group 1: the views
    view_bgl: wgpu::BindGroupLayout,
    /// group 2 of `late`: the depth pyramid
    pub pyramid_bgl: wgpu::BindGroupLayout,
    views: Option<(wgpu::Buffer, wgpu::BindGroup, usize)>,
}

impl CullPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let entries = [
            entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
            entry(1, storage(true)),
            entry(2, storage(false)),
            entry(3, storage(false)),
            entry(4, storage(false)),
        ];
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("InstanceCulling/BGL"), entries: &entries[..4] });
        let occlusion_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("InstanceCulling/OcclusionBGL"), entries: &entries });
        let view_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("InstanceCulling/ViewBGL"), entries: &[entry(0, storage(true))] });
        let pyramid_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("InstanceCulling/PyramidBGL"),
            entries: &[entry(0, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2, multisampled: false })],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("InstanceCulling"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let pipeline = |entry_point: &str, bgls: &[&wgpu::BindGroupLayout]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("InstanceCulling"), bind_group_layouts: bgls, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("InstanceCulling/{entry_point}")),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            pipeline: pipeline("main", &[&bgl, &view_bgl]),
            early: pipeline("early", &[&occlusion_bgl, &view_bgl]),
            late: pipeline("late", &[&occlusion_bgl, &view_bgl, &pyramid_bgl]),
            bgl,
            occlusion_bgl,
            view_bgl,
            pyramid_bgl,
            views: None,
        }
    }

    /// Upload the frame's views (one write), for `view_bind_group`.
    pub fn set_views(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, views: &[CullViewGpu]) {
        if self.views.as_ref().is_none_or(|(_, _, capacity)| *capacity < views.len()) {
            let capacity = views.len().next_power_of_two().max(8);
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("InstanceCulling/Views"),
                size: (capacity * std::mem::size_of::<CullViewGpu>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("InstanceCulling/Views"),
                layout: &self.view_bgl,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            });
            self.views = Some((buffer, bind_group, capacity));
        }
        queue.write_buffer(&self.views.as_ref().unwrap().0, 0, bytemuck::cast_slice(views));
    }

    /// Group 1 of every cull pipeline (after `set_views`).
    pub fn view_bind_group(&self) -> &wgpu::BindGroup {
        &self.views.as_ref().expect("set_views first").1
    }

    /// The `late` pipeline's group 2 for a depth pyramid.
    pub fn pyramid_bind_group(&self, device: &wgpu::Device, pyramid: &super::DepthPyramid) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("InstanceCulling/Pyramid"),
            layout: &self.pyramid_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(pyramid.view()) }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_planes_classify_points() {
        let view = glam::Mat4::look_at_rh(glam::Vec3::new(0.0, 0.0, 10.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let planes = frustum_planes(proj * view);
        let inside = |p: glam::Vec3, r: f32| planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= -r);
        assert!(inside(glam::Vec3::ZERO, 0.0));
        assert!(!inside(glam::Vec3::new(0.0, 0.0, 20.0), 0.0), "behind the camera");
        assert!(!inside(glam::Vec3::new(0.0, 0.0, -200.0), 0.0), "beyond the far plane");
        assert!(!inside(glam::Vec3::new(30.0, 0.0, 0.0), 1.0), "off to the side");
        // a sphere straddling the left plane is kept
        let edge_x = 10.0 * 0.5f32.tan();
        assert!(inside(glam::Vec3::new(-edge_x - 0.5, 0.0, 0.0), 1.0));
        assert!(!inside(glam::Vec3::new(-edge_x - 2.0, 0.0, 0.0), 1.0));
        // planes are normalized: distances are metres
        for p in planes {
            assert!((p.truncate().length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn aabb_in_frustum_keeps_boxes_that_touch_the_view() {
        let view = glam::Mat4::look_at_rh(glam::Vec3::new(0.0, 0.0, 10.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let planes = frustum_planes(glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0) * view);
        let in_view = |c: glam::Vec3, h: f32| aabb_in_frustum(&planes, c - glam::Vec3::splat(h), c + glam::Vec3::splat(h));
        assert!(in_view(glam::Vec3::ZERO, 1.0));
        assert!(!in_view(glam::Vec3::new(0.0, 0.0, 20.0), 1.0), "behind the camera");
        assert!(!in_view(glam::Vec3::new(30.0, 0.0, 0.0), 1.0), "off to the side");
        assert!(!in_view(glam::Vec3::new(0.0, 0.0, -200.0), 1.0), "beyond the far plane");
        // straddling the left plane: kept; a little further out: culled
        let edge_x = 10.0 * 0.5f32.tan();
        assert!(in_view(glam::Vec3::new(-edge_x - 0.5, 0.0, 0.0), 1.0));
        assert!(!in_view(glam::Vec3::new(-edge_x - 2.0, 0.0, 0.0), 1.0));
        // a box around the camera, and one far bigger than the view, are kept
        assert!(in_view(glam::Vec3::new(0.0, 0.0, 10.0), 1.0));
        assert!(in_view(glam::Vec3::ZERO, 1000.0));
    }

    #[test]
    fn shader_validates_and_params_layout_matches() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
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
        assert_eq!(span("CullInstances"), std::mem::size_of::<CullInstancesGpu>());
        assert_eq!(span("CullView"), std::mem::size_of::<CullViewGpu>());
    }

    /// On a real GPU: of a row of instances along x, a view keeps exactly those in its frustum
    /// and LOD band, copies their whole records, and counts them into the indirect draw; views
    /// the renderable is not drawn in (an unused one, a shadow it does not cast) draw nothing;
    /// all of it with every view in one dispatch, and split into chunks.
    #[test]
    fn gpu_culls_and_compacts_per_view() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // 8-word instances: centre xyz, scale, then an id and padding
        let mut data: Vec<f32> = Vec::new();
        for i in 0..100 {
            data.extend_from_slice(&[i as f32 - 50.0, 0.0, 0.0, 1.0, i as f32, 0.0, 0.0, 0.0]);
        }
        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let mut pipeline = CullPipeline::new(&device);
        // view 0: looking down -z from z = 20, 90 degrees wide: sees |x| < 20 at z = 0
        // view 1: the same from x = 40, sees 20 < x < 60 but the LOD band (from the origin) stops at 30
        // view 2: view 1 with a LOD distance scale of 0.5, so the band reaches 60
        // view 3: unused (no light this frame)
        // view 4: view 0 for shadow casters only, which this renderable is not
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let view = |x: f32, lod_distance_scale: f32, casters_only: bool, layer_mask: Option<u32>| CullView {
            view_proj: proj * glam::Mat4::look_at_rh(glam::Vec3::new(x, 0.0, 20.0), glam::Vec3::new(x, 0.0, 0.0), glam::Vec3::Y),
            casters_only,
            reflection: false,
            gi: false,
            layer_mask,
            lod_distance_scale,
        };
        // view 5: view 0 drawing layers 1 and 2 (the renderable is on layer 1, bit 0)
        // view 6: view 0 drawing layer 2 only
        let views = [
            Some(view(0.0, 1.0, false, None)),
            Some(view(40.0, 1.0, false, None)),
            Some(view(40.0, 0.5, false, None)),
            None,
            Some(view(0.0, 1.0, true, None)),
            Some(view(0.0, 1.0, false, Some(0b11))),
            Some(view(0.0, 1.0, false, Some(0b10))),
        ];
        pipeline.set_views(&device, &queue, &views.map(|v| v.map_or(bytemuck::Zeroable::zeroed(), |v| v.gpu(glam::Vec3::ZERO, None, false))));
        // every view in one chunk, then 3 and 1 per chunk
        for per_chunk in [u64::MAX, 3, 1] {
            let mut culling = InstanceCulling::new(source.clone(), 100, 32, 0, 0.5).with_radius_scale(12).with_lod_range(0.0, 30.0);
            culling.ensure_views_within(&device, &pipeline.bgl, views.len(), per_chunk.saturating_mul(100 * 32));
            let chunks = culling.shared.as_ref().unwrap().chunks.len();
            assert_eq!(chunks, (views.len() as u64).div_ceil(per_chunk.min(views.len() as u64)) as usize);
            let mut encoder = device.create_command_encoder(&Default::default());
            culling.begin_frame(&queue, &mut encoder, glam::Mat4::IDENTITY, 36, false, 1, false);
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline.pipeline);
                pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
                culling.dispatch(&mut pass);
            }
            queue.submit(Some(encoder.finish()));
            // each view's draw, ARGS_BYTES apart in the args, and its region of the instances
            let args = read_words(&device, &queue, culling.view(0).unwrap().args);
            let ids = |v: usize| -> (u32, Vec<i32>) {
                let draw = culling.view(v).unwrap();
                let a = &args[draw.offset as usize / 4..][..8];
                let inst = &read_words(&device, &queue, draw.instances)[draw.instances_offset as usize / 4..];
                let mut ids: Vec<i32> = (0..a[1] as usize).map(|k| f32::from_bits(inst[k * 8 + 4]) as i32 - 50).collect();
                ids.sort();
                (a[0], ids)
            };
            let label = format!("{per_chunk} views per chunk");
            // |x| <= 20 (spheres of 0.5 straddle the planes at +-20.x)
            assert_eq!(ids(0), (36, (-20..=20).collect::<Vec<_>>()), "{label}: view 0");
            // 20 <= x < 30: in view 1's frustum and within 30 of the LOD origin
            assert_eq!(ids(1), (36, (20..=29).collect::<Vec<_>>()), "{label}: view 1");
            // and with the LOD distances halved, all of them to the last instance (49)
            assert_eq!(ids(2), (36, (20..=49).collect::<Vec<_>>()), "{label}: view 2");
            // not drawn in: untouched (cleared) draws
            assert_eq!(ids(3), (0, vec![]), "{label}: unused view");
            assert_eq!(ids(4), (0, vec![]), "{label}: casters only");
            assert_eq!(ids(5), ids(0), "{label}: on a layer the view draws");
            assert_eq!(ids(6), (0, vec![]), "{label}: on no layer the view draws");
            for (v, drawn) in views.iter().zip([true, true, true, false, false, true, false]) {
                assert_eq!(v.is_some_and(|v| v.draws(false, 1, false)), drawn, "{label}: CullView::draws agrees");
            }
        }
    }

    /// On a real GPU: each kind of view culls by its own LOD band, the camera by `lod_range`, a
    /// shadow map by `shadow_lod_range` and a reflection by `reflection_lod_range`.
    #[test]
    fn gpu_lod_bands_per_kind_of_view() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // a row of instances along x, |x| metres from the LOD origin; ids from x
        let data: Vec<f32> = (0..100).flat_map(|i| [i as f32 - 50.0, 0.0, 0.0, 1.0, i as f32, 0.0, 0.0, 0.0]).collect();
        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE });
        let mut pipeline = CullPipeline::new(&device);
        // three views seeing the whole row: the camera, a shadow map and a reflection
        let view_proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0) * glam::Mat4::look_at_rh(glam::Vec3::new(0.0, 0.0, 200.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let view = |casters_only, reflection| CullView { view_proj, casters_only, reflection, gi: false, layer_mask: None, lod_distance_scale: 1.0 };
        let views = [view(false, false), view(true, false), view(false, true)];
        pipeline.set_views(&device, &queue, &views.map(|v| v.gpu(glam::Vec3::ZERO, None, false)));
        let mut culling = InstanceCulling::new(source, 100, 32, 0, 0.1)
            .with_radius_scale(12)
            .with_lod_range(0.0, 10.0)
            .with_shadow_lod_range(0.0, 5.0)
            .with_reflection_lod_range(20.0, 30.0);
        culling.ensure_views(&device, &pipeline.bgl, views.len());
        let mut encoder = device.create_command_encoder(&Default::default());
        culling.begin_frame(&queue, &mut encoder, glam::Mat4::IDENTITY, 36, true, 1, false);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
            culling.dispatch(&mut pass);
        }
        queue.submit(Some(encoder.finish()));
        let args = read_words(&device, &queue, culling.view(0).unwrap().args);
        let ids = |v: usize| -> Vec<i32> {
            let draw = culling.view(v).unwrap();
            let count = args[draw.offset as usize / 4 + 1] as usize;
            let inst = &read_words(&device, &queue, draw.instances)[draw.instances_offset as usize / 4..];
            let mut ids: Vec<i32> = (0..count).map(|k| f32::from_bits(inst[k * 8 + 4]) as i32 - 50).collect();
            ids.sort();
            ids
        };
        assert_eq!(ids(0), (-9..=9).collect::<Vec<_>>(), "the camera's band");
        assert_eq!(ids(1), (-4..=4).collect::<Vec<_>>(), "the shadow map's band");
        assert_eq!(ids(2), (-29..=-20).chain(20..=29).collect::<Vec<_>>(), "the reflection's band");
    }

    /// The GPU time of the cull pass (timestamps) for a forest the size of the midsommar film's:
    /// 48 renderables (24 sets of 4700 instances, two LOD bands each) in 4 views, each
    /// renderable's views in one dispatch against one dispatch per view (1-view chunks). Other
    /// GPU work on the machine inflates it, so the median is the figure to read. Run by hand:
    /// `cargo test --release -p kansei-core --lib time_cull_pass -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn time_cull_pass() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        if !adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return eprintln!("no timestamp queries: skipping");
        }
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: wgpu::Features::TIMESTAMP_QUERY, ..Default::default() }, None)).unwrap();
        use wgpu::util::DeviceExt;
        let mut seed = 1u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let sources: Vec<wgpu::Buffer> = (0..24)
            .map(|_| {
                let data: Vec<f32> = (0..4700).flat_map(|_| [rnd() * 600.0 - 300.0, 0.0, rnd() * 600.0 - 300.0, 0.5 + rnd(), 0.0, 0.0, 0.0, 0.0]).collect();
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE })
            })
            .collect();
        let mut pipeline = CullPipeline::new(&device);
        let look = |eye: glam::Vec3, at: glam::Vec3, fov: f32, far: f32| CullView {
            view_proj: glam::Mat4::perspective_rh(fov.to_radians(), 16.0 / 9.0, 0.1, far) * glam::Mat4::look_at_rh(eye, at, glam::Vec3::Y),
            casters_only: false,
            reflection: false,
            gi: false,
            layer_mask: None,
            lod_distance_scale: 1.0,
        };
        let views = [
            look(glam::Vec3::new(0.0, 2.0, 0.0), glam::Vec3::new(0.0, 1.5, -10.0), 60.0, 1000.0),
            look(glam::Vec3::new(-0.8, 1.0, -2.0), glam::Vec3::new(-0.8, 0.0, -30.0), 40.0, 60.0),
            look(glam::Vec3::new(0.8, 1.0, -2.0), glam::Vec3::new(0.8, 0.0, -30.0), 40.0, 60.0),
            look(glam::Vec3::new(0.0, -2.0, 0.0), glam::Vec3::new(0.0, -1.5, -10.0), 60.0, 1000.0),
        ];
        pipeline.set_views(&device, &queue, &views.map(|v| v.gpu(glam::Vec3::new(0.0, 2.0, 0.0), None, false)));
        let make = |per_view: bool| -> Vec<InstanceCulling> {
            sources
                .iter()
                .flat_map(|source| [(0.0, 30.0), (30.0, 200.0)].map(|(near, far)| {
                    let mut c = InstanceCulling::new(source.clone(), 4700, 32, 0, 1.0).with_radius_scale(12).with_lod_range(near, far);
                    c.ensure_views_within(&device, &pipeline.bgl, views.len(), if per_view { 4700 * 32 } else { u64::MAX });
                    c
                }))
                .collect()
        };
        const FRAMES: u32 = 200;
        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor { label: None, ty: wgpu::QueryType::Timestamp, count: 2 * FRAMES });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 * FRAMES as u64, usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 16 * FRAMES as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let run = |cullings: &mut [InstanceCulling]| -> (f64, f64) {
            for frame in 0..FRAMES {
                let mut encoder = device.create_command_encoder(&Default::default());
                for c in cullings.iter_mut() {
                    c.begin_frame(&queue, &mut encoder, glam::Mat4::IDENTITY, 36, true, 1, false);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites { query_set: &query_set, beginning_of_pass_write_index: Some(2 * frame), end_of_pass_write_index: Some(2 * frame + 1) }),
                    });
                    pass.set_pipeline(&pipeline.pipeline);
                    pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
                    for c in cullings.iter() {
                        c.dispatch(&mut pass);
                    }
                }
                queue.submit(Some(encoder.finish()));
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.resolve_query_set(&query_set, 0..2 * FRAMES, &resolve, 0);
            encoder.copy_buffer_to_buffer(&resolve, 0, &staging, 0, resolve.size());
            queue.submit(Some(encoder.finish()));
            staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let stamps: Vec<u64> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
            staging.unmap();
            let period = queue.get_timestamp_period() as f64;
            // (after a warm-up; a pass whose stamps did not resolve reads 0)
            let mut ms: Vec<f64> = stamps.chunks(2).skip(20).filter(|t| t[0] > 0 && t[1] > t[0]).map(|t| (t[1] - t[0]) as f64 * period / 1e6).collect();
            ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            (ms[0], ms[ms.len() / 2])
        };
        let (mut merged, mut per_view) = (make(false), make(true));
        for round in 0..3 {
            let (a_min, a_med) = run(&mut merged);
            let (b_min, b_med) = run(&mut per_view);
            eprintln!("round {round}: a dispatch per renderable {a_min:.3} ms min, {a_med:.3} ms median; a dispatch per view {b_min:.3} ms min, {b_med:.3} ms median");
        }
    }

    /// On a real GPU: two LODs meeting at 50 m with a 10 m crossfade both draw the instances from
    /// 45 to 55 m, with fades that add up to 1 (the nearer LOD's share falling, the farther's
    /// rising), and nothing else changes; a shadow map fades at its own band's edge (30 m), a
    /// reflection at the camera's band scaled by its LOD distance scale.
    #[test]
    fn gpu_crossfades_between_lods() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // a row of instances along x, x metres from the LOD origin (0 to 99); ids from x
        let data: Vec<f32> = (0..100).flat_map(|i| [i as f32, 0.0, 0.0, 1.0, i as f32, 0.0, 0.0, 0.0]).collect();
        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE });
        let mut pipeline = CullPipeline::new(&device);
        // the camera, a shadow map and a reflection (LOD distances x 2), all seeing the whole row
        let view_proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 1000.0) * glam::Mat4::look_at_rh(glam::Vec3::new(50.0, 0.0, 200.0), glam::Vec3::new(50.0, 0.0, 0.0), glam::Vec3::Y);
        let view = |casters_only, reflection, lod_distance_scale| CullView { view_proj, casters_only, reflection, gi: false, layer_mask: None, lod_distance_scale };
        let views = [view(false, false, 1.0), view(true, false, 1.0), view(false, true, 2.0)];
        pipeline.set_views(&device, &queue, &views.map(|v| v.gpu(glam::Vec3::ZERO, None, false)));
        let lod = |near: f32, far: f32, shadow: (f32, f32)| InstanceCulling::new(source.clone(), 100, 32, 0, 0.1).with_lod_range(near, far).with_shadow_lod_range(shadow.0, shadow.1).with_crossfade(10.0);
        let mut lods = [lod(0.0, 50.0, (0.0, 30.0)), lod(50.0, f32::INFINITY, (30.0, f32::INFINITY))];
        let mut encoder = device.create_command_encoder(&Default::default());
        for culling in &mut lods {
            assert_eq!(culling.culled_stride(), 36);
            culling.ensure_views(&device, &pipeline.bgl, views.len());
            culling.begin_frame(&queue, &mut encoder, glam::Mat4::IDENTITY, 36, true, 1, false);
        }
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline.pipeline);
            pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
            for culling in &lods {
                culling.dispatch(&mut pass);
            }
        }
        queue.submit(Some(encoder.finish()));
        // each LOD's (id, fade) in view v, by id
        let culled = |culling: &InstanceCulling, v: usize| -> Vec<(u32, f32)> {
            let draw = culling.view(v).unwrap();
            let count = read_words(&device, &queue, draw.args)[draw.offset as usize / 4 + 1] as usize;
            let words = &read_words(&device, &queue, draw.instances)[draw.instances_offset as usize / 4..];
            let mut out: Vec<(u32, f32)> = (0..count).map(|k| (f32::from_bits(words[k * 9 + 4]) as u32, f32::from_bits(words[k * 9 + 8]))).collect();
            out.sort_by_key(|&(id, _)| id);
            out
        };
        // (view, its band's edge in LOD distance, its LOD distance scale)
        for (v, edge, scale, what) in [(0usize, 50.0f32, 1.0f32, "camera"), (1, 30.0, 1.0, "shadow"), (2, 50.0, 2.0, "reflection")] {
            let (near, far) = (culled(&lods[0], v), culled(&lods[1], v));
            let ids = |c: &[(u32, f32)]| c.iter().map(|&(id, _)| id).collect::<Vec<_>>();
            // the nearer LOD to edge + 5 (exclusive), the farther from edge - 5, in LOD distance
            let (lo, hi) = (((edge - 5.0) / scale).ceil() as u32, ((edge + 5.0) / scale).ceil() as u32);
            assert_eq!(ids(&near), (0..hi).collect::<Vec<_>>(), "{what}: the nearer LOD");
            assert_eq!(ids(&far), (lo..100).collect::<Vec<_>>(), "{what}: the farther LOD");
            for &(id, fade) in &near {
                let d = id as f32 * scale;
                let want = if d <= edge - 5.0 { 1.0 } else { (edge + 5.0 - d) / 10.0 };
                assert!((fade - want).abs() < 1e-5, "{what}: nearer LOD, instance {id}: fade {fade}, want {want}");
            }
            for &(id, fade) in &far {
                let d = id as f32 * scale;
                if d >= edge + 5.0 {
                    assert_eq!(fade, 1.0, "{what}: farther LOD, instance {id}");
                } else {
                    // fading in, and the two shares add up to 1
                    let other = near.iter().find(|&&(n, _)| n == id).map(|&(_, f)| f).unwrap();
                    assert!(fade <= 0.0 && (other - fade - 1.0).abs() < 1e-5, "{what}: instance {id}: fades {other} and {fade}");
                }
            }
        }
    }

    /// The material side (LOD_FADE_WGSL): of 64 x 64 pixels, a LOD fading out with a share f and
    /// its partner fading in with 1 - f keep complementary pixels (every pixel exactly once), the
    /// first about f of them, on every frame.
    #[test]
    fn lod_fade_dither_is_complementary() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let code = format!(
            "{}\n@group(0) @binding(0) var<storage, read_write> out : array<u32>;\n\
             @group(0) @binding(1) var<uniform> job : vec4f;\n\
             @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) g : vec3u) {{\n\
                 let p = vec2f(g.xy) + 0.5;\n\
                 let keepOut = !kansei_lod_fade_discard(job.x, p, u32(job.z));\n\
                 let keepIn = !kansei_lod_fade_discard(job.y, p, u32(job.z));\n\
                 out[g.y * 64u + g.x] = select(0u, 1u, keepOut) | select(0u, 2u, keepIn);\n\
             }}",
            crate::culling::LOD_FADE_WGSL
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        use wgpu::util::DeviceExt;
        let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 64 * 64 * 4, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        for f in [0.1f32, 0.35, 0.5, 0.8] {
            for frame in [0u32, 1, 17] {
                let job = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&[f, -(1.0 - f), frame as f32, 0.0]), usage: wgpu::BufferUsages::UNIFORM });
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[wgpu::BindGroupEntry { binding: 0, resource: out.as_entire_binding() }, wgpu::BindGroupEntry { binding: 1, resource: job.as_entire_binding() }],
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline);
                    pass.set_bind_group(0, &bg, &[]);
                    pass.dispatch_workgroups(8, 8, 1);
                }
                queue.submit(Some(encoder.finish()));
                let words = read_words(&device, &queue, &out);
                assert!(words.iter().all(|&w| w == 1 || w == 2), "f {f}, frame {frame}: a pixel kept by neither or both");
                let share = words.iter().filter(|&&w| w == 1).count() as f32 / words.len() as f32;
                assert!((share - f).abs() < 0.03, "f {f}, frame {frame}: the fading-out LOD keeps {share}");
            }
        }
    }

    fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
        queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let words = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        words
    }

    /// A 64 x 64 depth buffer, cleared to the far plane, with a wall at `wall_depth` over its
    /// left half (or none).
    fn wall_depth(device: &wgpu::Device, queue: &wgpu::Queue, wall_depth: Option<f32>) -> wgpu::Texture {
        let depth = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: 64, height: 64, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let source = format!(
            "@vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{
                 let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
                 return vec4f(uv * 2.0 - 1.0, 0.5, 1.0);
             }}
             @fragment fn fs(@builtin(position) p : vec4f) -> @builtin(frag_depth) f32 {{
                 if (p.x < 32.0) {{ return {:?}; }}
                 return 1.0;
             }}",
            wall_depth.unwrap_or(1.0)
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(source.into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth.create_view(&Default::default()),
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        queue.submit(Some(encoder.finish()));
        depth
    }

    /// On a real GPU, frame by frame: the first phase draws the instances seen last frame, the
    /// second those it did not that are visible against the pyramid; hidden instances are
    /// culled, and drawn again the frame they are uncovered; a reset forgets the history; spheres
    /// and boxes both work, and the counters add up.
    #[test]
    fn gpu_culls_occluded_instances_in_two_phases() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // the camera at the origin looking down -z, 90 degrees; a wall 10 m away over the left half
        let view = glam::Mat4::look_at_rh(glam::Vec3::ZERO, glam::Vec3::NEG_Z, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let wall = proj.project_point3(glam::Vec3::new(0.0, 0.0, -10.0)).z;
        // 8-word instances: centre xyz, scale, id; bounds of 0.5 x scale
        const HIDDEN: u32 = 0; // behind the wall
        const RIGHT: u32 = 1; // beside it
        const FRONT: u32 = 2; // in front of it
        const EDGE: u32 = 3; // straddling its edge
        // (4: beyond the far plane)
        let spheres: [[f32; 4]; 5] = [[-5.0, 0.0, -20.0, 1.0], [5.0, 0.0, -20.0, 1.0], [-3.0, 0.0, -5.0, 1.0], [-1.0, 0.0, -20.0, 3.0], [0.0, 0.0, -200.0, 1.0]];
        let mut data: Vec<f32> = Vec::new();
        for (id, s) in spheres.iter().enumerate() {
            data.extend_from_slice(&[s[0], s[1], s[2], s[3], id as f32, 0.0, 0.0, 0.0]);
        }
        // the same, for a box shifted 1 x scale up from centres 1 x scale lower
        for (id, s) in spheres.iter().enumerate() {
            data.extend_from_slice(&[s[0], s[1] - s[3], s[2], s[3], id as f32, 0.0, 0.0, 0.0]);
        }
        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&data), usage: wgpu::BufferUsages::STORAGE });
        // culled in two phases in view 0 alone, then in view 1 while view 0 (the same camera)
        // stays frustum-only: the renderer's camera, and a reflection with occlusion culling;
        // then with a pyramid of view distances (as reflections have), to the same effect
        for (occluded, linear) in [(0usize, false), (1, false), (1, true)] {
            let mut pipeline = CullPipeline::new(&device);
            let mut sphere = InstanceCulling::new(source.clone(), 5, 32, 0, 0.5).with_radius_scale(12).with_occlusion(true);
            // all ten records as boxes: the first five then reach from their centre up, the second
            // five are centred where the spheres are; either way each id is hidden, beside, ...
            let mut boxed = InstanceCulling::new(source.clone(), 10, 32, 0, 0.5)
                .with_radius_scale(12)
                .with_bounds_shift(glam::Vec3::Y)
                .with_bounds_box(glam::Vec3::new(0.5, 1.0, 0.5))
                .with_occlusion(true);
            for culling in [&mut sphere, &mut boxed] {
                culling.ensure_views(&device, &pipeline.bgl, occluded + 1);
                assert!(culling.ensure_occlusion(&device, &pipeline, &[occluded]));
                culling.set_two_phase(&[occluded]);
            }
            let cull_view = CullView { view_proj: proj * view, casters_only: false, reflection: false, gi: false, layer_mask: None, lod_distance_scale: 1.0 };
            let occlusion = OcclusionView { view, proj, depth_size: (64, 64), reverse_z: false, linear_depth: linear };
            let views: Vec<_> = (0..=occluded).map(|v| cull_view.gpu(glam::Vec3::ZERO, (v == occluded).then_some(&occlusion), true)).collect();
            pipeline.set_views(&device, &queue, &views);
            let mut pyramid = super::super::DepthPyramid::new(&device, 64, 64, super::super::DepthReduction::Max);

            // one frame of both phases; the ids each drew, (lod, frustum, occluded) culled, and
            // the ids the frustum-only view drew (if any)
            let mut frame = |culling: &mut InstanceCulling, wall_at: Option<f32>, reset: bool| -> (Vec<u32>, Vec<u32>, [u32; 3], Vec<u32>) {
                let depth = wall_depth(&device, &queue, wall_at);
                pyramid.resize(&device, 64, 64);
                let bind_group = pipeline.pyramid_bind_group(&device, &pyramid);
                let mut encoder = device.create_command_encoder(&Default::default());
                if reset {
                    culling.reset_visibility(&mut encoder);
                }
                culling.begin_frame(&queue, &mut encoder, glam::Mat4::IDENTITY, 36, false, 1, false);
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    // as the renderer: every view, which leaves the one culled in two phases to them
                    pass.set_pipeline(&pipeline.pipeline);
                    pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
                    culling.dispatch(&mut pass);
                    pass.set_pipeline(&pipeline.early);
                    pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
                    culling.dispatch_early(&mut pass, occluded);
                }
                if linear {
                    pyramid.build_linear(&device, &queue, &mut encoder, &depth.create_view(&Default::default()), proj.inverse());
                } else {
                    pyramid.build(&device, &mut encoder, &depth.create_view(&Default::default()));
                }
                {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(&pipeline.late);
                    pass.set_bind_group(1, pipeline.view_bind_group(), &[]);
                    pass.set_bind_group(2, &bind_group, &[]);
                    culling.dispatch_late(&mut pass, occluded);
                }
                queue.submit(Some(encoder.finish()));
                let (early, late) = (culling.view(occluded).unwrap(), culling.late(occluded).unwrap());
                let ids = |draw: CulledDraw, args: &[u32]| -> Vec<u32> {
                    let words = &read_words(&device, &queue, draw.instances)[draw.instances_offset as usize / 4..];
                    let mut ids: Vec<u32> = (0..args[1] as usize).map(|k| f32::from_bits(words[k * 8 + 4]) as u32).collect();
                    ids.sort();
                    ids
                };
                let args = |draw: CulledDraw| read_words(&device, &queue, draw.args)[draw.offset as usize / 4..][..8].to_vec();
                let (a0, a1) = (args(early), args(late));
                assert_eq!((a0[0], a1[0]), (36, 36), "index counts");
                let frustum_only = if occluded > 0 {
                    let draw = culling.view(0).unwrap();
                    ids(draw, &args(draw))
                } else {
                    Vec::new()
                };
                (ids(early, &a0), ids(late, &a1), [a0[5] + a1[5], a0[6] + a1[6], a0[7] + a1[7]], frustum_only)
            };

            for culling in [&mut sphere, &mut boxed] {
                let label = format!("{} in view {occluded}{}", if culling.bounds_box.is_some() { "box" } else { "sphere" }, if linear { ", view distances" } else { "" });
                // frame 1: nothing seen yet; the second phase draws the visible ones
                let (early, late, culled, frustum_only) = frame(&mut *culling, Some(wall), true);
                if culling.count == 5 {
                    assert_eq!(early, Vec::<u32>::new(), "{label}: frame 1 early");
                    assert_eq!(late, vec![RIGHT, FRONT, EDGE], "{label}: frame 1 late");
                    assert_eq!(culled, [0, 1, 1], "{label}: frame 1 (lod, frustum, occluded)");
                    if occluded > 0 {
                        assert_eq!(frustum_only, vec![HIDDEN, RIGHT, FRONT, EDGE], "{label}: the frustum-only view");
                    }
                    // frame 2: the first phase draws them; the hidden one stays culled
                    let (early, late, culled, _) = frame(&mut *culling, Some(wall), false);
                    assert_eq!((early, late, culled), (vec![RIGHT, FRONT, EDGE], vec![], [0, 1, 1]), "{label}: frame 2");
                    // frame 3: the wall is gone; the second phase draws the uncovered one
                    let (early, late, culled, _) = frame(&mut *culling, None, false);
                    assert_eq!((early, late, culled), (vec![RIGHT, FRONT, EDGE], vec![HIDDEN], [0, 1, 0]), "{label}: frame 3");
                    // frame 4: all four seen; a reset forgets them
                    let (early, _, _, _) = frame(&mut *culling, None, false);
                    assert_eq!(early, vec![HIDDEN, RIGHT, FRONT, EDGE], "{label}: frame 4");
                    let (early, late, _, _) = frame(&mut *culling, Some(wall), true);
                    assert_eq!((early, late), (vec![], vec![RIGHT, FRONT, EDGE]), "{label}: after a reset");
                } else {
                    assert_eq!(early, Vec::<u32>::new(), "{label}: frame 1 early");
                    assert_eq!(late, vec![RIGHT, RIGHT, FRONT, FRONT, EDGE, EDGE], "{label}: frame 1 late");
                    assert_eq!(culled, [0, 2, 2], "{label}: frame 1 (lod, frustum, occluded)");
                }
            }
        }
    }
}

