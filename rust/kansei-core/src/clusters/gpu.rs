//! A `ClusterMesh` on the GPU, and the camera's per-frame cut of it.

use super::{ClusterMesh, Sphere};
use crate::geometries::Vertex;
use bytemuck::{Pod, Zeroable};

/// The packed mesh's layout (`ClusterMesh::gpu_words`, read by cluster_mesh.wgsl).
pub(crate) const HEADER_WORDS: usize = 8;
pub(crate) const VERTEX_WORDS: usize = 9;
pub(crate) const CLUSTER_WORDS: usize = 28;
pub(crate) const LEVEL_WORDS: usize = 8;
/// The error of a missing parent (∞): shaders never see infinities.
pub(crate) const NO_PARENT: f32 = -1.0;

pub(crate) const CLUSTER_MESH_WGSL: &str = include_str!("../shaders/cluster_mesh.wgsl");

const _: () = assert!(std::mem::size_of::<Vertex>() == VERTEX_WORDS * 4);

impl ClusterMesh {
    /// The most triangles in one cluster: what every cluster is drawn as.
    pub fn max_triangles(&self) -> u32 {
        self.clusters.iter().map(|c| c.triangle_count).max().unwrap_or(0)
    }

    /// The mesh as the GPU reads it, in one buffer. A header of where each section starts (in
    /// words), the cluster and level counts, and `max_triangles`; then the vertices (`Vertex` as
    /// is), the clusters' vertices, their triangles (3 local indices in a word's low 3 bytes),
    /// the cluster records and the level records (see cluster_mesh.wgsl).
    pub fn gpu_words(&self) -> Vec<u32> {
        let levels = self.levels();
        let vertices = HEADER_WORDS;
        let cluster_vertices = vertices + self.vertices.len() * VERTEX_WORDS;
        let triangles = cluster_vertices + self.cluster_vertices.len();
        let clusters = triangles + self.cluster_triangles.len() / 3;
        let level_records = clusters + self.clusters.len() * CLUSTER_WORDS;
        let mut words = Vec::with_capacity(level_records + levels.len() * LEVEL_WORDS);
        words.extend([vertices, cluster_vertices, triangles, clusters, level_records, self.clusters.len(), levels.len(), self.max_triangles() as usize].map(|w| w as u32));
        words.extend_from_slice(bytemuck::cast_slice(&self.vertices));
        words.extend_from_slice(&self.cluster_vertices);
        words.extend(self.cluster_triangles.chunks(3).map(|t| t[0] as u32 | (t[1] as u32) << 8 | (t[2] as u32) << 16));
        let parent = |e: f32| if e.is_finite() { e } else { NO_PARENT };
        let sphere = |s: Sphere| [s.center.x, s.center.y, s.center.z, s.radius].map(f32::to_bits);
        for c in &self.clusters {
            words.extend([c.vertex_offset, c.triangle_offset, c.triangle_count, c.level]);
            words.extend(sphere(c.bounds));
            words.extend([c.cone_apex.x, c.cone_apex.y, c.cone_apex.z, c.cone_cutoff].map(f32::to_bits));
            words.extend([c.cone_axis.x, c.cone_axis.y, c.cone_axis.z, c.error].map(f32::to_bits));
            words.extend(sphere(c.lod_bounds));
            words.extend(sphere(c.parent_bounds));
            words.extend([parent(c.parent_error).to_bits(), 0, 0, 0]);
        }
        let finite = |x: f32| if x.is_finite() { x } else { 0.0 };
        for l in &levels {
            words.extend([l.first, l.count, l.min_error.to_bits(), parent(l.max_parent_error).to_bits(), finite(l.near_reach).to_bits(), finite(l.far_reach).to_bits(), 0, 0]);
        }
        words
    }
}

pub(crate) const CLUSTER_CULL_WGSL: &str = include_str!("../shaders/cluster_cull.wgsl");
pub(crate) const NO_WORD: u32 = u32::MAX;
const KIND_NONE: u32 = 0;
const KIND_PLACEMENT: u32 = 1;
const KIND_MATRIX: u32 = 2;
const FLAG_CONE: u32 = 1;
/// Entries of a draw list by default, at most (32 MB).
pub(crate) const DEFAULT_MAX_DRAWN: u32 = 1 << 22;
/// Entries a draw list starts with, until the cull says what its cut needs (128 KB; each entry
/// stands for a cluster's triangles in the index buffer, ~1.5 KB).
pub(crate) const INITIAL_DRAWN: u32 = 1 << 14;
/// Entries a draw list keeps at least (8 KB).
pub(crate) const MIN_DRAWN: u32 = 1 << 10;
/// Readbacks in a row a cut must need at most a quarter of its list before the list shrinks.
pub(crate) const SHRINK_AFTER: u32 = 64;
/// Triangles an index buffer starts with, until the cull says what its cut needs (3 MB).
pub(crate) const INITIAL_TRIANGLES: u32 = 1 << 18;
/// Triangles an index buffer keeps room for at least (48 KB).
pub(crate) const MIN_TRIANGLES: u32 = 1 << 12;
/// Bytes of a cluster draw: `DrawIndexedIndirect`'s five words (the index count, one instance,
/// zeros), then the visible instances, the clusters claimed and listed (drawn), the triangles
/// claimed and listed, and two pads.
pub(crate) const DRAW_ARGS_BYTES: u64 = 48;
/// Word of the draw: the clusters claimed, drawn or not.
pub(crate) const CLAIMED_WORD: u64 = 6;
/// Word of the draw: the triangles claimed, drawn or not.
pub(crate) const TRIANGLES_CLAIMED_WORD: u64 = 8;

/// Where an instance record places the mesh, as far as the cluster test needs it: spheres,
/// errors and cones follow the instance. It should say what the material's vertex stage does
/// with the record. Offsets are in bytes into the record, multiples of 4.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InstanceTransform {
    /// A column-major 4x4 matrix of f32.
    Matrix { offset: u32 },
    /// A position (3 x f32), then optionally a uniform scale (f32), a turn about +y (f32 times
    /// `yaw_scale` radians, as `glam::Mat3::from_rotation_y`: -1 for a bearing that turns a mesh
    /// by minus itself) and a rotation (a unit quaternion, x y z w):
    /// `position + yaw * rotation * (scale * p)`.
    Placement { position: u32, scale: Option<u32>, yaw: Option<u32>, yaw_scale: f32, rotation: Option<u32> },
}

/// Where the cull reads a renderable's instances: none (the mesh once, where the renderable
/// is), every record of a buffer, or a view's compacted records (`InstanceCulling`), as many
/// as word `count_word` of its draws says.
#[derive(Clone, Copy)]
pub(crate) enum InstanceSource<'a> {
    None,
    All { records: &'a wgpu::Buffer, count: u32 },
    Culled { records: &'a wgpu::Buffer, first_record: u32, capacity: u32, args: &'a wgpu::Buffer, count_word: u32 },
}

impl InstanceSource<'_> {
    /// Instances it can hold, which sizes the draw list.
    pub(crate) fn capacity(&self) -> u32 {
        match self {
            Self::None => 1,
            Self::All { count, .. } => *count,
            Self::Culled { capacity, .. } => *capacity,
        }
    }
}

/// `ClusterCull` in cluster_cull.wgsl: a renderable's parameters.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ClusterCullGpu {
    world: [f32; 16],
    kind: u32,
    position_word: u32,
    scale_word: u32,
    yaw_word: u32,
    rotation_word: u32,
    stride_words: u32,
    first_record: u32,
    instance_count: u32,
    count_word: u32,
    capacity: u32,
    vertex_count: u32,
    flags: u32,
    yaw_scale: f32,
    stretch: f32,
    /// the cut's view, in `ClusterCulling::set_views` (set by `ClusterGpu::bind`)
    view: u32,
    /// the triangles its index buffer holds (set by `ClusterGpu::bind`)
    triangle_capacity: u32,
}

impl ClusterCullGpu {
    /// A renderable's parameters. Its world matrix; how its records (`stride` bytes each) place
    /// the mesh, and where they come from; the draw list's capacity; the draw's vertices
    /// (3 x the mesh's max triangles); whether to test the cones; and how much further the
    /// material may stretch an instance (`ClusterLod::stretch`: no cone test past 1).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(world: glam::Mat4, transform: Option<InstanceTransform>, stride: u32, source: &InstanceSource, capacity: u32, vertex_count: u32, cone_culling: bool, stretch: f32) -> Self {
        let word = |offset: u32| offset / 4;
        let (kind, position_word, scale_word, yaw_word, yaw_scale, rotation_word) = match (source, transform) {
            (InstanceSource::None, _) | (_, None) => (KIND_NONE, 0, NO_WORD, NO_WORD, 1.0, NO_WORD),
            (_, Some(InstanceTransform::Matrix { offset })) => (KIND_MATRIX, word(offset), NO_WORD, NO_WORD, 1.0, NO_WORD),
            (_, Some(InstanceTransform::Placement { position, scale, yaw, yaw_scale, rotation })) => (KIND_PLACEMENT, word(position), scale.map_or(NO_WORD, word), yaw.map_or(NO_WORD, word), yaw_scale, rotation.map_or(NO_WORD, word)),
        };
        let (first_record, instance_count, count_word) = match *source {
            InstanceSource::None => (0, 1, NO_WORD),
            InstanceSource::All { count, .. } => (0, count, NO_WORD),
            InstanceSource::Culled { first_record, count_word, .. } => (first_record, 0, count_word),
        };
        Self {
            world: world.to_cols_array(),
            kind,
            position_word,
            scale_word,
            yaw_word,
            rotation_word,
            stride_words: stride / 4,
            first_record,
            instance_count,
            count_word,
            capacity,
            vertex_count,
            flags: if cone_culling && stretch <= 1.0 { FLAG_CONE } else { 0 },
            yaw_scale,
            stretch: stretch.max(1.0),
            view: 0,
            triangle_capacity: 0,
        }
    }
}

/// `ClusterView` in cluster_cull.wgsl: the view every renderable is cut for.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ClusterViewGpu {
    planes: [[f32; 4]; 6],
    eye: [f32; 3],
    pixels_per_radian: f32,
    near: f32,
    threshold: f32,
    orthographic: u32,
    _pad: f32,
}

impl ClusterViewGpu {
    /// A view: its view-projection's frustum, the eye errors are measured from, pixels per
    /// radian (per metre when `orthographic`), the distance errors are clamped to, and the budget
    /// in pixels.
    pub(crate) fn new(view_proj: glam::Mat4, eye: glam::Vec3, pixels_per_unit: f32, near: f32, threshold: f32, orthographic: bool) -> Self {
        Self { planes: crate::culling::frustum_planes(view_proj).map(|p| p.to_array()), eye: eye.to_array(), pixels_per_radian: pixels_per_unit, near, threshold, orthographic: orthographic as u32, _pad: 0.0 }
    }
}

/// Views a frame's cull holds at most (the camera, the spot shadow layers, the reflections, the
/// cascades and the sky: far fewer).
pub(crate) const MAX_VIEWS: usize = 64;

/// The cluster cull's pipelines and the frame's views (one per renderer).
pub(crate) struct ClusterCulling {
    cull_bgl: wgpu::BindGroupLayout,
    prepare_bgl: wgpu::BindGroupLayout,
    prepare: wgpu::ComputePipeline,
    cull: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    view: wgpu::Buffer,
    view_bind_group: wgpu::BindGroup,
    feedback: Feedback,
}

impl ClusterCulling {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let uniform = |binding| buffer_entry(binding, wgpu::BufferBindingType::Uniform);
        let storage = |binding, read_only| buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only });
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let cull_bgl = layout("ClusterCulling/Cull", &[uniform(0), storage(1, true), storage(2, true), storage(3, false), storage(4, false), storage(5, false)]);
        let view_bgl = layout("ClusterCulling/View", &[storage(0, true)]);
        let prepare_bgl = layout("ClusterCulling/Prepare", &[uniform(10), storage(11, true), storage(12, false), storage(13, false)]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ClusterCulling"), source: wgpu::ShaderSource::Wgsl(format!("{CLUSTER_CULL_WGSL}\n{CLUSTER_MESH_WGSL}").into()) });
        let pipeline = |entry: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ClusterCulling"), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&format!("ClusterCulling/{entry}")), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let view = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ClusterCulling/Views"), size: (MAX_VIEWS * std::mem::size_of::<ClusterViewGpu>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ClusterCulling/View"), layout: &view_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view.as_entire_binding() }] });
        Self { prepare: pipeline("prepare", &[&prepare_bgl]), cull: pipeline("cull", &[&cull_bgl, &view_bgl]), finish: pipeline("finish", &[&prepare_bgl]), cull_bgl, prepare_bgl, view, view_bind_group, feedback: Feedback::default() }
    }

    /// The frame's views, by index (`ClusterGpu::bind`'s `view`), in one write: a write per view
    /// would leave every cut with the last (`queue.write_buffer` lands before the frame's work).
    pub(crate) fn set_views(&self, queue: &wgpu::Queue, views: &[ClusterViewGpu]) {
        assert!(views.len() <= MAX_VIEWS, "{} cluster views, at most {MAX_VIEWS}", views.len());
        queue.write_buffer(&self.view, 0, bytemuck::cast_slice(views));
    }

    /// Start a frame's cull: collect the counts a finished readback holds (each read once, by
    /// the next `ClusterGpu::bind` of its cut).
    pub(crate) fn begin_frame(&mut self, device: &wgpu::Device) {
        self.feedback.begin_frame(device);
    }

    /// After the frame's cuts `(clusters, view)` are submitted: read back how many clusters each
    /// claimed (drawn or not), unless a readback is still in flight.
    pub(crate) fn read_back(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cuts: &[(&ClusterGpu, u32)]) {
        self.feedback.read_back(device, queue, cuts);
    }

    /// Run the cuts `(clusters, view)` (each bound with `ClusterGpu::bind`) in one compute pass:
    /// every prepare, then every cull (dispatched indirectly, a workgroup per visible instance),
    /// then every finish. A cull's dispatch buffer is never bound while it is dispatched.
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder, cuts: &[(&ClusterGpu, u32)]) {
        let bound: Vec<(&Cut, &Bound)> = cuts.iter().filter_map(|&(c, view)| {
            let cut = c.cut(view)?;
            Some((cut, cut.bound.as_ref()?))
        }).collect();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Renderer/ClusterCulling"), timestamp_writes: crate::profiling::gpu_pass("Renderer/ClusterCulling").as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.prepare);
        for (_, b) in &bound {
            pass.set_bind_group(0, &b.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_pipeline(&self.cull);
        pass.set_bind_group(1, &self.view_bind_group, &[]);
        for (c, b) in &bound {
            pass.set_bind_group(0, &b.cull_bind_group, &[]);
            pass.dispatch_workgroups_indirect(&c.dispatch, 0);
        }
        pass.set_pipeline(&self.finish);
        for (_, b) in &bound {
            pass.set_bind_group(0, &b.prepare_bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
    }
}

/// Cluster LOD for a renderable (`Renderable::clusters`). Each frame, every view that draws it
/// (the camera and its velocity pass, the spot and cascaded shadow maps, rendered planar
/// reflections and the sky occlusion's top-down view) draws the cut of `mesh` that view needs
/// (`Renderer::set_cluster_error_threshold`, times the view's scale:
/// `Renderer::set_shadow_cluster_error_scale`, `PlanarReflection::lod_error_scale`,
/// `SkyOcclusionOptions::lod_error_scale`) instead of the geometry, through a vertex stage
/// generated around the material's `vertex_main`. The geometry, which must be the mesh `mesh`
/// was built from, is still what impostor bakes and point-light (cubemap) shadows draw.
///
/// The instances are the geometry's one instance buffer (if any), culled by the renderable's
/// `InstanceCulling` when it has one (without occlusion phases) and placed as `transform` says.
/// Set it before the renderable is first drawn, or call `Renderer::invalidate_bundle` after.
pub struct ClusterLod {
    pub mesh: std::sync::Arc<ClusterMesh>,
    /// How an instance record places the mesh. None: the instances are drawn where the renderable
    /// is (or there are none).
    pub transform: Option<InstanceTransform>,
    /// Skip clusters whose every triangle faces away (on by default), where the material culls
    /// back faces and isn't transparent. Turn it off when the material turns instances in a way
    /// `transform` doesn't describe.
    pub cone_culling: bool,
    /// Clusters drawn per frame in each view, at most. By default every cluster of every
    /// instance, up to 4 194 304. Each view's cut keeps a draw list (8 bytes an entry) and an
    /// index buffer of its clusters' triangles (12 bytes each), drawn as one indexed draw, both
    /// sized to what it needs as the cull reads back: 16 384 entries and 262 144 triangles at
    /// first, then half again the need, growing at once and shrinking only after 64 readbacks at
    /// a quarter or less. A cut that suddenly needs more than they hold (at load, or past half
    /// again its recent need) leaves the clusters past them undrawn until the readback lands,
    /// 2-3 frames later.
    pub capacity: Option<u32>,
    /// How much further than `transform` the material may stretch or sway an instance about its
    /// origin (1 by default): no point moves more than `stretch - 1` times its distance from the
    /// origin. The cull's errors grow by it, its spheres by it and by how far their centres may
    /// move, and its cones are off past 1. The film's trees: widths up to 1.1x their height, and
    /// a sway.
    pub stretch: f32,
    pub(crate) gpu: Option<ClusterGpu>,
}

impl ClusterLod {
    pub fn new(mesh: impl Into<std::sync::Arc<ClusterMesh>>) -> Self {
        Self { mesh: mesh.into(), transform: None, cone_culling: true, capacity: None, stretch: 1.0, gpu: None }
    }

    pub fn with_transform(mut self, transform: InstanceTransform) -> Self {
        self.transform = Some(transform);
        self
    }

    pub fn with_cone_culling(mut self, on: bool) -> Self {
        self.cone_culling = on;
        self
    }

    pub fn with_stretch(mut self, stretch: f32) -> Self {
        self.stretch = stretch;
        self
    }

    pub fn with_capacity(mut self, clusters: u32) -> Self {
        self.capacity = Some(clusters);
        self
    }

    /// Ready this frame's cut for view `view`: the GPU state made once, the cut bound to `source`
    /// (records of `stride` bytes) with the parameters, and its vertex stage group 2 (`layout`)
    /// over the renderer's normal and world matrices. `back_faces_culled`: the material culls
    /// back faces (the cone test only removes what it would). True when what bundles recorded
    /// changed.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, layout: &wgpu::BindGroupLayout, matrices: (&wgpu::Buffer, &wgpu::Buffer), view: u32, source: InstanceSource, stride: u32, world: glam::Mat4, back_faces_culled: bool) -> bool {
        let gpu = self.gpu.get_or_insert_with(|| ClusterGpu::new(device, &self.mesh));
        let every = (source.capacity() as u64 * gpu.cluster_count as u64).min(DEFAULT_MAX_DRAWN as u64) as u32;
        let capacity = self.capacity.unwrap_or(every).max(1);
        let params = ClusterCullGpu::new(world, self.transform, stride, &source, capacity, gpu.vertex_count, self.cone_culling && back_faces_culled, self.stretch);
        let grown = gpu.bind(device, queue, culling, view, source, params);
        gpu.bind_draw(device, layout, view, matrices.0, matrices.1) || grown
    }
}

/// A renderable's cluster mesh on the GPU, and its cuts: one per view that draws it.
pub(crate) struct ClusterGpu {
    /// Names its cuts' readbacks (`Feedback`), whatever the scene does with the renderable.
    id: u64,
    mesh: wgpu::Buffer,
    vertex_count: u32,
    cluster_count: u32,
    /// bound in place of a missing instance buffer or count
    empty: wgpu::Buffer,
    /// by view (`ClusterCulling::set_views`); none for views that never drew it
    cuts: Vec<Option<Cut>>,
    /// the most triangles an index buffer holds (tests make it small)
    triangle_limit: u32,
}

/// A view's cut of a renderable: its parameters, the draw list and its indirect draw, and the
/// cull's indirect dispatch.
pub(crate) struct Cut {
    params: wgpu::Buffer,
    written: Option<ClusterCullGpu>,
    draws: wgpu::Buffer,
    /// `draws`' length in entries (0 before the first bind)
    capacity: u32,
    list: Sizer,
    /// the cut's triangles, 3 indices each (`entry << 8 | local vertex`)
    indices: wgpu::Buffer,
    /// `indices`' room in triangles (0 before the first bind)
    triangle_capacity: u32,
    triangles: Sizer,
    args: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    bound: Option<Bound>,
    /// the vertex stage's group 2, and the buffers it was made with
    draw: Option<(DrawKey, wgpu::BindGroup)>,
}

/// The buffers the vertex stage's group 2 holds: the normal and world matrices, the draw list and
/// the records.
type DrawKey = (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, Option<wgpu::Buffer>);

/// The cull's bind groups, and the instance buffers they were made with.
struct Bound {
    records: Option<wgpu::Buffer>,
    count: Option<wgpu::Buffer>,
    cull_bind_group: wgpu::BindGroup,
    prepare_bind_group: wgpu::BindGroup,
}

impl Cut {
    fn new(device: &wgpu::Device) -> Self {
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        Self {
            params: buffer("Clusters/Params", std::mem::size_of::<ClusterCullGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            written: None,
            draws: buffer("Clusters/Draws", 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            capacity: 0,
            list: Sizer::default(),
            indices: buffer("Clusters/Indices", 12, wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            triangle_capacity: 0,
            triangles: Sizer::default(),
            // (COPY_SRC: read back by the stats, the feedback and the tests)
            args: buffer("Clusters/Args", DRAW_ARGS_BYTES, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            dispatch: buffer("Clusters/Dispatch", 16, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE),
            bound: None,
            draw: None,
        }
    }

    /// The indirect draw (`DRAW_ARGS_BYTES`, see `DRAW_ARGS_BYTES` for its words).
    pub(crate) fn args(&self) -> &wgpu::Buffer {
        &self.args
    }

    /// Its index buffer (`DRAW_ARGS_BYTES`'s index count of it is drawn).
    pub(crate) fn indices(&self) -> &wgpu::Buffer {
        &self.indices
    }

    /// The vertex stage's group 2 for this cut, once `ClusterGpu::bind_draw` made it.
    pub(crate) fn draw_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.draw.as_ref().map(|(_, group)| group)
    }
}

impl ClusterGpu {
    pub(crate) fn new(device: &wgpu::Device, mesh: &ClusterMesh) -> Self {
        use wgpu::util::DeviceExt;
        static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        Self {
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            mesh: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Clusters/Mesh"), contents: bytemuck::cast_slice(&mesh.gpu_words()), usage: wgpu::BufferUsages::STORAGE }),
            vertex_count: 3 * mesh.max_triangles(),
            cluster_count: mesh.clusters.len() as u32,
            empty: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Empty"), size: 32, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            cuts: Vec::new(),
            triangle_limit: u32::MAX,
        }
    }

    /// View `view`'s cut, if it was ever bound.
    pub(crate) fn cut(&self, view: u32) -> Option<&Cut> {
        self.cuts.get(view as usize)?.as_ref()
    }

    /// Bind view `view`'s cut to `source`, with a draw list of at most `params.capacity` entries
    /// sized to what the cut needs (`Cut::sized`, from `culling`'s readbacks), and write the
    /// parameters if they changed. True when the draw list was remade: draws recorded with the
    /// old one are stale.
    pub(crate) fn bind(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, view: u32, source: InstanceSource, mut params: ClusterCullGpu) -> bool {
        params.view = view;
        if self.cuts.len() <= view as usize {
            self.cuts.resize_with(view as usize + 1, || None);
        }
        let (mesh, empty) = (&self.mesh, &self.empty);
        let cut = self.cuts[view as usize].get_or_insert_with(|| Cut::new(device));
        let max = params.capacity.max(1);
        let reading = culling.feedback.needed(self.id, view);
        let needed = reading.map(|(clusters, _)| clusters);
        let triangles_seen = reading.map(|(clusters, triangles)| triangles_needed(clusters, cut.capacity, triangles));
        let length = cut.list.sized(cut.capacity, INITIAL_DRAWN, MIN_DRAWN, max, needed);
        // (the shader lists no more than the list holds, nor than `max`)
        params.capacity = length.min(max);
        let mut grown = length != cut.capacity;
        if grown {
            cut.capacity = length;
            cut.draws = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Draws"), size: cut.capacity as u64 * 8, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            cut.bound = None;
        }
        // every cluster it may list at its largest, within what a binding holds
        let binding = (device.limits().max_storage_buffer_binding_size as u64).min(device.limits().max_buffer_size) / 12;
        let most = (max as u64 * (self.vertex_count / 3) as u64).min(binding).min(self.triangle_limit as u64).max(1) as u32;
        let triangles = cut.triangles.sized(cut.triangle_capacity, INITIAL_TRIANGLES, MIN_TRIANGLES, most, triangles_seen);
        params.triangle_capacity = triangles.min(most);
        if triangles != cut.triangle_capacity {
            cut.triangle_capacity = triangles;
            cut.indices = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Indices"), size: triangles as u64 * 12, usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            cut.bound = None;
            grown = true;
        }
        let (records, count) = match source {
            InstanceSource::None => (None, None),
            InstanceSource::All { records, .. } => (Some(records.clone()), None),
            InstanceSource::Culled { records, args, .. } => (Some(records.clone()), Some(args.clone())),
        };
        if cut.bound.as_ref().is_none_or(|b| b.records != records || b.count != count) {
            let cull_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Clusters/Cull"),
                layout: &culling.cull_bgl,
                entries: &[entry(0, &cut.params), entry(1, mesh), entry(2, records.as_ref().unwrap_or(empty)), entry(3, &cut.draws), entry(4, &cut.args), entry(5, &cut.indices)],
            });
            let prepare_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Clusters/Prepare"),
                layout: &culling.prepare_bgl,
                entries: &[entry(10, &cut.params), entry(11, count.as_ref().unwrap_or(empty)), entry(12, &cut.args), entry(13, &cut.dispatch)],
            });
            cut.bound = Some(Bound { records, count, cull_bind_group, prepare_bind_group });
        }
        if cut.written != Some(params) {
            queue.write_buffer(&cut.params, 0, bytemuck::bytes_of(&params));
            cut.written = Some(params);
        }
        grown
    }

    /// View `view`'s vertex stage group 2 (`SharedLayouts::cluster_mesh_bgl`): the renderer's
    /// normal and world matrices, the mesh, the cut's draw list and its bound records. Remade when
    /// any of them changed; true then (bundles recorded the old one). The cut must be bound.
    pub(crate) fn bind_draw(&mut self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, view: u32, normal: &wgpu::Buffer, world: &wgpu::Buffer) -> bool {
        let (mesh, empty) = (&self.mesh, &self.empty);
        let cut = self.cuts[view as usize].as_mut().expect("the cut is bound");
        let key = (normal.clone(), world.clone(), cut.draws.clone(), cut.bound.as_ref().and_then(|b| b.records.clone()));
        if cut.draw.as_ref().is_some_and(|(k, _)| *k == key) {
            return false;
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Clusters/Draw"),
            layout,
            entries: &[matrix(0, normal, 64), matrix(1, world, 128), entry(2, mesh), entry(3, &cut.draws), entry(4, key.3.as_ref().unwrap_or(empty))],
        });
        cut.draw = Some((key, group));
        true
    }

    /// View `view`'s indirect draw (the cut must be bound).
    #[cfg(test)]
    pub(crate) fn args(&self, view: u32) -> &wgpu::Buffer {
        &self.cut(view).expect("the cut is bound").args
    }

    /// View `view`'s index buffer.
    #[cfg(test)]
    pub(crate) fn indices(&self, view: u32) -> &wgpu::Buffer {
        &self.cut(view).expect("the cut is bound").indices
    }

    /// Room for at most `triangles` in every cut's index buffer.
    #[cfg(test)]
    pub(crate) fn limit_triangles(&mut self, triangles: u32) {
        self.triangle_limit = triangles;
    }

    /// View `view`'s draw list: (record, cluster) per drawn cluster.
    #[cfg(test)]
    pub(crate) fn draws(&self, view: u32) -> &wgpu::Buffer {
        &self.cut(view).expect("the cut is bound").draws
    }

    /// View `view`'s draw list's length, in entries.
    #[cfg(test)]
    pub(crate) fn capacity(&self, view: u32) -> u32 {
        self.cut(view).expect("the cut is bound").capacity
    }

    /// Vertices a cluster is drawn as.
    #[cfg(test)]
    pub(crate) fn vertex_count(&self) -> u32 {
        self.vertex_count
    }
}

/// The cull's counts read back: how many clusters each cut claimed, drawn or not (word 5 of its
/// draw), copied into one buffer a frame while none is in flight and read when mapped, a few
/// frames later, so the frame never waits.
#[derive(Default)]
struct Feedback {
    staging: Option<wgpu::Buffer>,
    /// the cuts (`ClusterGpu` id, view) of the copy in flight, and its state
    pending: Option<(Vec<(u64, u32)>, std::sync::Arc<std::sync::atomic::AtomicU8>)>,
    /// the last readback's counts (clusters, triangles), until the next frame's
    needed: std::collections::HashMap<(u64, u32), (u32, u32)>,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

impl Feedback {
    /// The clusters and triangles cut (`id`, `view`) last claimed, if a reading arrived.
    fn needed(&self, id: u64, view: u32) -> Option<(u32, u32)> {
        self.needed.get(&(id, view)).copied()
    }

    fn begin_frame(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        self.needed.clear();
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending.is_some() {
            device.poll(wgpu::Maintain::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let Some((cuts, state)) = self.pending.take_if(|(_, s)| s.load(Ordering::Acquire) != MAPPING) else { return };
        if state.load(Ordering::Acquire) == FAILED {
            return;
        }
        let staging = self.staging.as_ref().unwrap();
        {
            let bytes = staging.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            self.needed.extend(cuts.into_iter().zip(words.chunks(2).map(|w| (w[0], w[1]))));
        }
        staging.unmap();
    }

    fn read_back(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, cuts: &[(&ClusterGpu, u32)]) {
        if self.pending.is_some() {
            return;
        }
        let read: Vec<(&Cut, (u64, u32))> = cuts.iter().filter_map(|&(c, view)| Some((c.cut(view)?, (c.id, view)))).collect();
        if read.is_empty() {
            return;
        }
        let size = read.len() as u64 * 8;
        if self.staging.as_ref().is_none_or(|s| s.size() < size) {
            self.staging = Some(device.create_buffer(&wgpu::BufferDescriptor { label: Some("ClusterCulling/Feedback"), size: size.next_power_of_two().max(16), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }));
        }
        let staging = self.staging.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("ClusterCulling/Feedback") });
        for (k, (cut, _)) in read.iter().enumerate() {
            encoder.copy_buffer_to_buffer(&cut.args, CLAIMED_WORD * 4, staging, k as u64 * 8, 4);
            encoder.copy_buffer_to_buffer(&cut.args, TRIANGLES_CLAIMED_WORD * 4, staging, k as u64 * 8 + 4, 4);
        }
        queue.submit(Some(encoder.finish()));
        let state = std::sync::Arc::new(std::sync::atomic::AtomicU8::new(MAPPING));
        let done = state.clone();
        staging.slice(..).map_async(wgpu::MapMode::Read, move |result| done.store(if result.is_ok() { MAPPED } else { FAILED }, std::sync::atomic::Ordering::Release));
        self.pending = Some((read.into_iter().map(|(_, key)| key).collect(), state));
    }
}

/// The triangles a cut needs, from a reading of the clusters it `claimed` and the triangles it
/// claimed room for: only clusters with an entry in its draw list (`list` of them) take room, so
/// past a full list the triangles are scaled by the clusters claimed over those with an entry
/// (both buffers then grow on the same readback).
pub(crate) fn triangles_needed(claimed: u32, list: u32, triangles: u32) -> u32 {
    if claimed <= list || list == 0 {
        return triangles;
    }
    (triangles as u64 * claimed as u64 / list as u64).min(u32::MAX as u64) as u32
}

/// A buffer's length (a draw list's entries, an index buffer's triangles) sized to what its cut
/// needs, from the cull's readbacks.
#[derive(Default)]
struct Sizer {
    /// readbacks in a row that needed at most a quarter of the length
    low: u32,
}

impl Sizer {
    /// The length for `current` (0: none yet) given a reading of what the cut `needed` (if one
    /// arrived): `initial` at first; then half again the need, rounded up to an eighth of its
    /// power of two (at least `min`), at once when that is longer; and that when it has been at most a quarter of the
    /// length for `SHRINK_AFTER` readings in a row. It grows no longer than `max`, but a smaller
    /// `max` alone doesn't shrink it (the draw is capped instead).
    fn sized(&mut self, current: u32, initial: u32, min: u32, max: u32, needed: Option<u32>) -> u32 {
        if current == 0 {
            return initial.min(max);
        }
        let mut length = current;
        if let Some(needed) = needed {
            // (rounded up to an eighth of its power of two: steps small next to the length)
            let want = (needed as u64 * 3 / 2).max(1);
            let step = (want.next_power_of_two() / 8).max(1);
            let target = want.div_ceil(step).saturating_mul(step).min(max as u64).max(min.min(max) as u64) as u32;
            if target > length {
                length = target;
                self.low = 0;
            } else if target as u64 * 4 <= length as u64 {
                self.low += 1;
                if self.low >= SHRINK_AFTER {
                    length = target;
                    self.low = 0;
                }
            } else {
                self.low = 0;
            }
        }
        length
    }
}

/// A bind group entry for the whole of `buffer`.
fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

/// A bind group entry for a mesh matrix: `size` bytes of `buffer` at a dynamic offset.
fn matrix(binding: u32, buffer: &wgpu::Buffer, size: u64) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset: 0, size: std::num::NonZeroU64::new(size) }) }
}
