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
/// Bytes of a cluster draw: `DrawIndirect`'s four words, then the visible instances, the
/// clusters claimed (drawn up to the capacity), the triangles drawn, and a pad.
pub(crate) const DRAW_ARGS_BYTES: u64 = 32;

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
    _pad: u32,
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
            _pad: 0,
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
}

impl ClusterCulling {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let buffer_entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let uniform = |binding| buffer_entry(binding, wgpu::BufferBindingType::Uniform);
        let storage = |binding, read_only| buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only });
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let cull_bgl = layout("ClusterCulling/Cull", &[uniform(0), storage(1, true), storage(2, true), storage(3, false), storage(4, false)]);
        let view_bgl = layout("ClusterCulling/View", &[storage(0, true)]);
        let prepare_bgl = layout("ClusterCulling/Prepare", &[uniform(10), storage(11, true), storage(12, false), storage(13, false)]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("ClusterCulling"), source: wgpu::ShaderSource::Wgsl(format!("{CLUSTER_CULL_WGSL}\n{CLUSTER_MESH_WGSL}").into()) });
        let pipeline = |entry: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("ClusterCulling"), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(&format!("ClusterCulling/{entry}")), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let view = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ClusterCulling/Views"), size: (MAX_VIEWS * std::mem::size_of::<ClusterViewGpu>()) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let view_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("ClusterCulling/View"), layout: &view_bgl, entries: &[wgpu::BindGroupEntry { binding: 0, resource: view.as_entire_binding() }] });
        Self { prepare: pipeline("prepare", &[&prepare_bgl]), cull: pipeline("cull", &[&cull_bgl, &view_bgl]), finish: pipeline("finish", &[&prepare_bgl]), cull_bgl, prepare_bgl, view, view_bind_group }
    }

    /// The frame's views, by index (`ClusterGpu::bind`'s `view`), in one write: a write per view
    /// would leave every cut with the last (`queue.write_buffer` lands before the frame's work).
    pub(crate) fn set_views(&self, queue: &wgpu::Queue, views: &[ClusterViewGpu]) {
        assert!(views.len() <= MAX_VIEWS, "{} cluster views, at most {MAX_VIEWS}", views.len());
        queue.write_buffer(&self.view, 0, bytemuck::cast_slice(views));
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
    /// Clusters drawn per frame in each view, at most (each view's cut has a draw list this
    /// long: 8 bytes an entry). By default every cluster of every instance, up to 4 194 304.
    /// Clusters past it aren't drawn.
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
    mesh: wgpu::Buffer,
    vertex_count: u32,
    cluster_count: u32,
    /// bound in place of a missing instance buffer or count
    empty: wgpu::Buffer,
    /// by view (`ClusterCulling::set_views`); none for views that never drew it
    cuts: Vec<Option<Cut>>,
}

/// A view's cut of a renderable: its parameters, the draw list and its indirect draw, and the
/// cull's indirect dispatch.
pub(crate) struct Cut {
    params: wgpu::Buffer,
    written: Option<ClusterCullGpu>,
    draws: wgpu::Buffer,
    capacity: u32,
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
            // (COPY_SRC: read back by the stats and the tests)
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

    /// The vertex stage's group 2 for this cut, once `ClusterGpu::bind_draw` made it.
    pub(crate) fn draw_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.draw.as_ref().map(|(_, group)| group)
    }
}

impl ClusterGpu {
    pub(crate) fn new(device: &wgpu::Device, mesh: &ClusterMesh) -> Self {
        use wgpu::util::DeviceExt;
        Self {
            mesh: device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Clusters/Mesh"), contents: bytemuck::cast_slice(&mesh.gpu_words()), usage: wgpu::BufferUsages::STORAGE }),
            vertex_count: 3 * mesh.max_triangles(),
            cluster_count: mesh.clusters.len() as u32,
            empty: device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Empty"), size: 32, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false }),
            cuts: Vec::new(),
        }
    }

    /// View `view`'s cut, if it was ever bound.
    pub(crate) fn cut(&self, view: u32) -> Option<&Cut> {
        self.cuts.get(view as usize)?.as_ref()
    }

    /// Bind view `view`'s cut to `source`, with a draw list of at least `params.capacity` entries
    /// (it grows, never shrinks), and write the parameters if they changed. True when the draw
    /// list was remade: draws recorded with the old one are stale.
    pub(crate) fn bind(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, culling: &ClusterCulling, view: u32, source: InstanceSource, mut params: ClusterCullGpu) -> bool {
        params.view = view;
        if self.cuts.len() <= view as usize {
            self.cuts.resize_with(view as usize + 1, || None);
        }
        let (mesh, empty) = (&self.mesh, &self.empty);
        let cut = self.cuts[view as usize].get_or_insert_with(|| Cut::new(device));
        let grown = params.capacity > cut.capacity;
        if grown {
            cut.capacity = params.capacity;
            cut.draws = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Clusters/Draws"), size: cut.capacity as u64 * 8, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            cut.bound = None;
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
                entries: &[entry(0, &cut.params), entry(1, mesh), entry(2, records.as_ref().unwrap_or(empty)), entry(3, &cut.draws), entry(4, &cut.args)],
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

    /// View `view`'s draw list: (record, cluster) per drawn cluster.
    #[cfg(test)]
    pub(crate) fn draws(&self, view: u32) -> &wgpu::Buffer {
        &self.cut(view).expect("the cut is bound").draws
    }

    /// Vertices a cluster is drawn as.
    #[cfg(test)]
    pub(crate) fn vertex_count(&self) -> u32 {
        self.vertex_count
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
