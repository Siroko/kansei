//! `RtGrid`: a uniform grid of world triangles in a box, rebuilt on the GPU (see the module docs).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use crate::clusters::InstanceTransform;

pub(crate) const BUILD_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_build.wgsl"));
pub(crate) const GATHER_WGSL: &str = concat!(include_str!("shaders/rt_types.wgsl"), include_str!("shaders/rt_gather.wgsl"));

/// Cells a grid holds at most (the scan's two levels of 1024).
pub const RT_MAX_CELLS: u32 = 1 << 20;
/// Cells a macro cell spans each way.
const MACRO: u32 = 4;
/// Words of the counters before the wide triangles' list (one word a triangle): [0] triangles
/// claimed, [1] references needed, [2] wide triangles, the scan's block sums from 16.
const COUNTER_WORDS: u64 = 1056;
/// Bytes of a world triangle.
pub const RT_TRIANGLE_BYTES: u64 = 64;
/// Bytes between sources' parameters (WebGPU's dynamic uniform offset alignment, at most).
const SOURCE_STRIDE: u64 = 256;
const NO_WORD: u32 = u32::MAX;

/// The WGSL `KanseiRtGrid` (rt_types.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct RtGridGpu {
    origin: [f32; 3],
    cell: f32,
    dims: [u32; 3],
    flags: u32,
    macro_dims: [u32; 3],
    epsilon: f32,
    cell_count: u32,
    macro_base: u32,
    big_base: u32,
    refs_base: u32,
    ref_capacity: u32,
    triangle_capacity: u32,
    big_capacity: u32,
    big_cells: u32,
}

/// The WGSL `RtSource` (rt_gather.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct RtSourceGpu {
    world: [f32; 16],
    triangles: u32,
    stride_words: u32,
    first_record: u32,
    records: u32,
    count_word: u32,
    surface: u32,
    albedo: u32,
    source: u32,
    slot: u32,
    kind: u32,
    _pad: [u32; 2],
}

/// What `RtGrid::new` sets up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtGridOptions {
    /// Cells each way: multiples of 4, at most `RT_MAX_CELLS` in all.
    pub dims: [u32; 3],
    /// A cell's size, metres.
    pub cell: f32,
    /// Where the box sits round the eye it follows (`RtGrid::follow`): the share of its height
    /// below the eye, the rest above.
    pub below: f32,
    /// The box follows the eye in steps of this many cells (a multiple of 4 keeps the macro cells
    /// on the same world blocks).
    pub snap_cells: u32,
    /// A box that stays put from this corner (a room) instead of following the eye.
    pub fixed_origin: Option<Vec3>,
    /// World units every cell is widened by when triangles are listed, so one lying on a cell's
    /// face, or a rounding error away from it, is listed on both sides (0: a 1024th of a cell).
    pub epsilon: f32,
    /// A triangle whose footprint across its plane's dominant axis spans more columns of cells
    /// than this goes to the big list, which every ray tests before walking the cells
    /// (miaumiau.cat/?p=1457's "big triangles"): a wall saves its thousands of references and
    /// their build. (Triangles over 32 columns are scattered by a workgroup each, so one wall no
    /// longer keeps a thread walking thousands of cells either way.)
    pub big_triangle_cells: u32,
    /// Triangles the big list holds; past that they are scattered into the cells.
    pub big_triangle_capacity: u32,
    /// Leave empty 4³ blocks of cells in one step.
    pub macro_skip: bool,
    /// Triangles the buffers start with room for; it grows to what a build needed (read back a
    /// few frames later: until then the triangles past it are lost).
    pub triangle_capacity: u32,
    /// References (a triangle listed in a cell) the buffers start with room for; grows likewise.
    pub reference_capacity: u32,
}

impl Default for RtGridOptions {
    fn default() -> Self {
        Self {
            dims: [128, 64, 128],
            cell: 0.5,
            below: 0.25,
            snap_cells: 4,
            fixed_origin: None,
            epsilon: 0.0,
            big_triangle_cells: 1024,
            big_triangle_capacity: 64,
            macro_skip: true,
            triangle_capacity: 1 << 16,
            reference_capacity: 1 << 20,
        }
    }
}

impl RtGridOptions {
    /// The epsilon cells are widened by.
    pub fn effective_epsilon(&self) -> f32 {
        if self.epsilon > 0.0 {
            self.epsilon
        } else {
            self.cell / 1024.0
        }
    }

    /// The box's size, metres.
    pub fn extent(&self) -> Vec3 {
        Vec3::new(self.dims[0] as f32, self.dims[1] as f32, self.dims[2] as f32) * self.cell
    }

    fn cell_count(&self) -> u32 {
        self.dims.iter().product()
    }

    fn macro_dims(&self) -> [u32; 3] {
        self.dims.map(|d| d.div_ceil(MACRO))
    }
}

/// What the last build read back needed, and the room the buffers have.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtGridStats {
    /// Triangles gathered (meeting the box), whether they fitted or not.
    pub triangles: u32,
    /// References the cells needed.
    pub references: u32,
    /// Triangles too wide for the cells (those up to `big_triangle_capacity` in the big list).
    pub big_triangles: u32,
    pub triangle_capacity: u32,
    pub reference_capacity: u32,
    /// Builds recorded.
    pub builds: u64,
}

/// The buffers a pass traces an `RtGrid` through, which follow the grid when its buffers are made
/// anew (`RtGrid::handle`): an effect holds one and binds what it holds each frame.
#[derive(Clone)]
pub struct RtGridHandle {
    shared: Arc<std::sync::Mutex<GridBuffers>>,
}

#[derive(Clone)]
struct GridBuffers {
    uniform: wgpu::Buffer,
    triangles: wgpu::Buffer,
    cells: wgpu::Buffer,
    generation: u64,
}

impl RtGridHandle {
    /// The buffers of `RtGrid::bindings_wgsl`, in order, and the grid's generation (changed when
    /// they are made anew).
    pub fn buffers(&self) -> ([wgpu::Buffer; 3], u64) {
        let b = self.shared.lock().unwrap();
        ([b.uniform.clone(), b.triangles.clone(), b.cells.clone()], b.generation)
    }
}

/// Where a source's records place its mesh, before the source's `world` matrix.
#[derive(Clone, Debug, PartialEq)]
pub enum RtPlacement {
    /// Nowhere else: the mesh once per record, where `world` puts it.
    None,
    /// As an instance record says (`InstanceTransform`, as cluster LOD reads it).
    Instance(InstanceTransform),
    /// WGSL defining `fn kansei_rt_place(record: u32, p: vec3f) -> vec3f`, where record `record`
    /// puts mesh point `p` (the renderable's space), for a material whose vertex stage does more
    /// than an `InstanceTransform` says. Read the record with `kansei_rt_record_f32`,
    /// `kansei_rt_record_vec3` or `kansei_rt_record_vec4(record, word)`.
    Wgsl(String),
}

impl RtPlacement {
    /// Its `kansei_rt_place`.
    pub fn wgsl(&self) -> String {
        match self {
            Self::None => "fn kansei_rt_place(record: u32, p: vec3f) -> vec3f { return p; }\n".into(),
            Self::Wgsl(code) => code.clone(),
            Self::Instance(InstanceTransform::Matrix { offset }) => {
                let w = offset / 4;
                format!(
                    "fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {{\n    let m = mat4x4f(kansei_rt_record_vec4(record, {}u), kansei_rt_record_vec4(record, {}u), kansei_rt_record_vec4(record, {}u), kansei_rt_record_vec4(record, {}u));\n    return (m * vec4f(p, 1.0)).xyz;\n}}\n",
                    w,
                    w + 4,
                    w + 8,
                    w + 12
                )
            }
            Self::Instance(InstanceTransform::Placement { position, scale, yaw, yaw_scale, rotation }) => {
                // as cluster_cull.wgsl's placement: position + yaw * rotation * (scale * p)
                let mut body = String::from("    var q = p;\n");
                if let Some(s) = scale {
                    body += &format!("    q = q * kansei_rt_record_f32(record, {}u);\n", s / 4);
                }
                if let Some(r) = rotation {
                    body += &format!(
                        "    let r = kansei_rt_record_vec4(record, {}u);\n    q = q + 2.0 * cross(r.xyz, cross(r.xyz, q) + r.w * q);\n",
                        r / 4
                    );
                }
                if let Some(y) = yaw {
                    body += &format!(
                        "    let a = kansei_rt_record_f32(record, {}u) * {:?};\n    q = vec3f(cos(a) * q.x + sin(a) * q.z, q.y, -sin(a) * q.x + cos(a) * q.z);\n",
                        y / 4,
                        yaw_scale
                    );
                }
                format!("fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {{\n{body}    return q + kansei_rt_record_vec3(record, {}u);\n}}\n", position / 4)
            }
        }
    }
}

/// How a triangle's surface reads to a ray: its albedo (for lighting a hit), and whether it is
/// alpha tested (`alpha_layer`: the caller's `kansei_rt_covered(layer, uv)` decides where it is
/// there).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtSurface {
    pub albedo: [f32; 3],
    pub alpha_layer: Option<u8>,
}

impl RtSurface {
    pub fn new(albedo: [f32; 3]) -> Self {
        Self { albedo, alpha_layer: None }
    }

    /// Alpha tested: `kansei_rt_covered(layer, uv)` says where it is there.
    pub fn with_alpha_layer(mut self, layer: u8) -> Self {
        self.alpha_layer = Some(layer);
        self
    }

    pub(crate) fn surface_word(&self) -> u32 {
        self.alpha_layer.map_or(0, |layer| 1 | (layer as u32) << 8)
    }

    pub(crate) fn albedo_word(&self) -> u32 {
        let byte = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
        byte(self.albedo[0]) | byte(self.albedo[1]) << 8 | byte(self.albedo[2]) << 16 | 255 << 24
    }
}

/// A source of triangles for `RtGrid::gather`: a mesh (`RtMesh::create_buffer`) placed once per
/// record.
pub struct RtSource<'a> {
    /// The mesh's words (`RtMesh::gpu_words`), in a STORAGE buffer.
    pub mesh: &'a wgpu::Buffer,
    /// Its triangles.
    pub triangles: u32,
    /// Instance records `stride` bytes each (a STORAGE buffer), placed by `placement`; none: the
    /// mesh once.
    pub records: Option<(&'a wgpu::Buffer, u32)>,
    /// The first record and how many.
    pub first_record: u32,
    pub record_count: u32,
    /// Where the placed mesh goes (a renderable's world matrix).
    pub world: Mat4,
    pub placement: &'a RtPlacement,
    pub surface: RtSurface,
    /// Names the source in its triangles (`kansei_rt_source`, 12 bits).
    pub id: u32,
}

/// A source as the gather runs it: `RtSource`'s, or one the renderer feeds from its culled views.
pub(crate) struct GatherSource<'a> {
    pub mesh: GatherMesh<'a>,
    /// Instance records `stride` bytes each; none: the mesh once.
    pub records: Option<(&'a wgpu::Buffer, u32)>,
    pub first_record: u32,
    pub count: GatherCount<'a>,
    pub world: Mat4,
    /// Its `kansei_rt_place` (`RtPlacement::wgsl`).
    pub placement: String,
    pub surface: RtSurface,
    pub id: u32,
}

/// What a source places.
pub(crate) enum GatherMesh<'a> {
    /// An `RtMesh`'s words.
    Mesh { buffer: &'a wgpu::Buffer, triangles: u32 },
    /// A cluster LOD cut: the mesh's words (`ClusterMesh::gpu_words`) and the cut's draw list of
    /// (record, cluster), whose records are absolute (`first_record` is ignored).
    Clusters { buffer: &'a wgpu::Buffer, draws: &'a wgpu::Buffer },
}

/// How many records (or draw-list entries) a source has.
pub(crate) enum GatherCount<'a> {
    Fixed(u32),
    /// Word `word` of `args` says, at most `at_most`.
    Word { args: &'a wgpu::Buffer, word: u32, at_most: u32 },
}

/// A source's bind group: its mesh, records, count and draw list.
type SourceKey = (wgpu::Buffer, Option<wgpu::Buffer>, Option<wgpu::Buffer>, Option<wgpu::Buffer>);

struct Readback {
    staging: wgpu::Buffer,
    pending: Option<Arc<AtomicU8>>,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// A uniform grid of world triangles in a box round the camera (or a fixed one), rebuilt on the
/// GPU, for rays: see the module docs. Each rebuild, between `begin` and `finish`, one `gather`
/// appends the sources' triangles; after the frame's submit, `read_back` reads what it needed.
pub struct RtGrid {
    options: RtGridOptions,
    origin: Vec3,
    placed: bool,
    uniform: wgpu::Buffer,
    written: Option<RtGridGpu>,
    triangles: wgpu::Buffer,
    cells: wgpu::Buffer,
    counters: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    sources: wgpu::Buffer,
    empty: wgpu::Buffer,
    triangle_capacity: u32,
    reference_capacity: u32,
    build_bgl: wgpu::BindGroupLayout,
    prepare_bgl: wgpu::BindGroupLayout,
    gather_grid_bgl: wgpu::BindGroupLayout,
    gather_source_bgl: wgpu::BindGroupLayout,
    gather_prepare_bgl: wgpu::BindGroupLayout,
    gather_prepare: wgpu::ComputePipeline,
    /// the sources' indirect dispatches, and the bind group writing them
    gather_dispatch: wgpu::Buffer,
    gather_dispatch_group: Option<wgpu::BindGroup>,
    prepare: wgpu::ComputePipeline,
    prepare_wide: wgpu::ComputePipeline,
    count: wgpu::ComputePipeline,
    count_wide: wgpu::ComputePipeline,
    fill: wgpu::ComputePipeline,
    fill_wide: wgpu::ComputePipeline,
    scan_blocks: wgpu::ComputePipeline,
    scan_sums: wgpu::ComputePipeline,
    scan_add: wgpu::ComputePipeline,
    gather_layout: wgpu::PipelineLayout,
    gather_pipelines: HashMap<(String, bool), wgpu::ComputePipeline>,
    bind_groups: Option<(wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup)>,
    source_groups: HashMap<SourceKey, wgpu::BindGroup>,
    readback: Readback,
    stats: RtGridStats,
    generation: u64,
    handle: RtGridHandle,
    /// between `begin` and `finish`: whether this build gathered yet
    gathered: Option<bool>,
}

impl RtGrid {
    pub fn new(device: &wgpu::Device, options: RtGridOptions) -> Self {
        assert!(options.dims.iter().all(|&d| d > 0 && d.is_multiple_of(MACRO)), "grid dims must be multiples of {MACRO}");
        assert!(options.cell_count() <= RT_MAX_CELLS, "at most {RT_MAX_CELLS} cells");
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let buffer = |ty, dynamic| wgpu::BindingType::Buffer { ty, has_dynamic_offset: dynamic, min_binding_size: None };
        let uniform = buffer(wgpu::BufferBindingType::Uniform, false);
        let rw = buffer(wgpu::BufferBindingType::Storage { read_only: false }, false);
        let ro = buffer(wgpu::BufferBindingType::Storage { read_only: true }, false);
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let build_bgl = layout("RtGrid/Build", &[entry(0, uniform), entry(1, rw), entry(2, rw), entry(3, rw)]);
        let prepare_bgl = layout("RtGrid/Prepare", &[entry(0, uniform), entry(3, rw), entry(4, rw)]);
        let gather_grid_bgl = layout("RtGrid/GatherGrid", &[entry(0, uniform), entry(1, rw), entry(2, rw)]);
        let gather_source_bgl = layout(
            "RtGrid/GatherSource",
            &[entry(0, buffer(wgpu::BufferBindingType::Uniform, true)), entry(1, ro), entry(2, ro), entry(3, ro), entry(4, ro)],
        );
        let gather_prepare_bgl = layout("RtGrid/GatherPrepare", &[entry(5, rw)]);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("RtGrid/Build"), source: wgpu::ShaderSource::Wgsl(BUILD_WGSL.into()) });
        let pipeline = |entry_point: &str, bgl: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("RtGrid/Build"), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("RtGrid/{entry_point}")),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let gather_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RtGrid/Gather"),
            bind_group_layouts: &[&gather_grid_bgl, &gather_source_bgl],
            push_constant_ranges: &[],
        });
        let gather_prepare = {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                source: wgpu::ShaderSource::Wgsl(gather_wgsl(&RtPlacement::None.wgsl()).into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                bind_group_layouts: &[&gather_prepare_bgl, &gather_source_bgl],
                push_constant_ranges: &[],
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("prepare"),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let storage = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | usage, mapped_at_creation: false });
        let mut grid = Self {
            options,
            origin: options.fixed_origin.unwrap_or(Vec3::ZERO),
            placed: false,
            uniform: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Uniform"),
                size: std::mem::size_of::<RtGridGpu>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            written: None,
            triangles: storage("RtGrid/Triangles", 16, wgpu::BufferUsages::empty()),
            cells: storage("RtGrid/Cells", 16, wgpu::BufferUsages::empty()),
            counters: storage("RtGrid/Counters", 16, wgpu::BufferUsages::empty()),
            dispatch: storage("RtGrid/Dispatch", 32, wgpu::BufferUsages::INDIRECT),
            sources: device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtGrid/Sources"), size: SOURCE_STRIDE, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            empty: storage("RtGrid/Empty", 16, wgpu::BufferUsages::empty()),
            triangle_capacity: 0,
            reference_capacity: 0,
            prepare: pipeline("prepare", &prepare_bgl),
            prepare_wide: pipeline("prepare_wide", &prepare_bgl),
            count: pipeline("count", &build_bgl),
            count_wide: pipeline("count_wide", &build_bgl),
            fill: pipeline("fill", &build_bgl),
            fill_wide: pipeline("fill_wide", &build_bgl),
            scan_blocks: pipeline("scan_blocks", &build_bgl),
            scan_sums: pipeline("scan_sums", &build_bgl),
            scan_add: pipeline("scan_add", &build_bgl),
            build_bgl,
            prepare_bgl,
            gather_grid_bgl,
            gather_source_bgl,
            gather_prepare_bgl,
            gather_prepare,
            gather_dispatch: storage("RtGrid/GatherDispatch", 16, wgpu::BufferUsages::INDIRECT),
            gather_dispatch_group: None,
            gather_layout,
            gather_pipelines: HashMap::new(),
            bind_groups: None,
            source_groups: HashMap::new(),
            readback: Readback {
                staging: device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtGrid/Readback"), size: 16, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
                pending: None,
            },
            stats: RtGridStats::default(),
            generation: 0,
            handle: RtGridHandle { shared: Arc::new(std::sync::Mutex::new(GridBuffers { uniform: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()), triangles: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()), cells: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()), generation: 0 })) },
            gathered: None,
        };
        grid.resize(device, options.triangle_capacity.max(64), options.reference_capacity.max(1024));
        grid
    }

    pub fn options(&self) -> &RtGridOptions {
        &self.options
    }

    /// The box's corner.
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// The box: its least and greatest corners.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.origin, self.origin + self.options.extent())
    }

    /// What the last readback said, and the room the buffers have.
    pub fn stats(&self) -> RtGridStats {
        RtGridStats { triangle_capacity: self.triangle_capacity, reference_capacity: self.reference_capacity, ..self.stats }
    }

    /// Changes whenever the buffers `bind_group_entries` binds are made anew: bind groups made
    /// with the old ones are stale.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bytes on the GPU: the triangles, the cell words, the counters (with the wide triangles'
    /// list) and the sources' parameters.
    pub fn memory_bytes(&self) -> u64 {
        self.triangles.size() + self.cells.size() + self.counters.size() + self.sources.size()
    }

    /// Place the box round `eye` (in steps of `snap_cells` cells, `below` of its height under the
    /// eye), unless it is fixed. True when it moved: rebuild it.
    pub fn follow(&mut self, eye: Vec3) -> bool {
        let o = &self.options;
        let origin = o.fixed_origin.unwrap_or_else(|| {
            let snap = o.cell * o.snap_cells.max(1) as f32;
            let size = o.extent();
            let corner = eye - Vec3::new(size.x * 0.5, size.y * o.below, size.z * 0.5);
            (corner / snap).floor() * snap
        });
        let moved = !self.placed || origin != self.origin;
        self.origin = origin;
        self.placed = true;
        moved
    }

    fn cells_layout(&self) -> (u32, u32, u32) {
        let cells = self.options.cell_count();
        let macro_cells = self.options.macro_dims().iter().product::<u32>();
        let macro_base = cells;
        let big_base = macro_base + macro_cells;
        let refs_base = big_base + 1 + self.options.big_triangle_capacity;
        (macro_base, big_base, refs_base)
    }

    /// The most a binding holds, in bytes.
    fn binding_limit(device: &wgpu::Device) -> u64 {
        let limits = device.limits();
        (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size)
    }

    /// Make the buffers hold `triangles` and `references` (within what a binding holds); their
    /// contents are lost.
    fn resize(&mut self, device: &wgpu::Device, triangles: u32, references: u32) {
        let limit = Self::binding_limit(device);
        let (_, _, refs_base) = self.cells_layout();
        let triangles = (triangles as u64).min(limit / RT_TRIANGLE_BYTES) as u32;
        let references = (references as u64).min(limit / 4 - refs_base as u64) as u32;
        if triangles != self.triangle_capacity {
            self.triangle_capacity = triangles;
            self.triangles = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Triangles"),
                size: triangles as u64 * RT_TRIANGLE_BYTES,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            // (every triangle may be wide)
            self.counters = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Counters"),
                size: (COUNTER_WORDS + triangles as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        }
        if references != self.reference_capacity {
            self.reference_capacity = references;
            self.cells = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Cells"),
                size: (refs_base as u64 + references as u64) * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
        }
        self.bind_groups = None;
        self.generation += 1;
        *self.handle.shared.lock().unwrap() = GridBuffers { uniform: self.uniform.clone(), triangles: self.triangles.clone(), cells: self.cells.clone(), generation: self.generation };
    }

    /// A handle to the buffers passes trace the grid through, which follows the grid's.
    pub fn handle(&self) -> RtGridHandle {
        self.handle.clone()
    }

    /// Room for `triangles` this build (call between `begin` and `gather`, when the sources know
    /// how many they hold at most): no frame then loses triangles to a readback still in flight.
    pub fn reserve_triangles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, triangles: u32) {
        if triangles > self.triangle_capacity {
            self.resize(device, grow(triangles), self.reference_capacity);
            self.ensure_bind_groups(device);
            if self.gathered.is_some() {
                self.write_uniform(queue);
            }
        }
    }

    fn ensure_bind_groups(&mut self, device: &wgpu::Device) {
        if self.bind_groups.is_some() {
            return;
        }
        let build = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/Build"),
            layout: &self.build_bgl,
            entries: &[entry(0, &self.uniform), entry(1, &self.triangles), entry(2, &self.cells), entry(3, &self.counters)],
        });
        let prepare = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/Prepare"),
            layout: &self.prepare_bgl,
            entries: &[entry(0, &self.uniform), entry(3, &self.counters), entry(4, &self.dispatch)],
        });
        let gather = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/GatherGrid"),
            layout: &self.gather_grid_bgl,
            entries: &[entry(0, &self.uniform), entry(1, &self.triangles), entry(2, &self.counters)],
        });
        self.bind_groups = Some((build, prepare, gather));
    }

    /// Start a rebuild: take a finished readback (growing the buffers to what it needed), write
    /// the box, and clear the triangles and the cells.
    pub fn begin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        assert!(self.gathered.is_none(), "RtGrid::begin twice without finish");
        self.collect_readback(device);
        let need_triangles = self.stats.triangles;
        // (triangles that didn't fit listed nothing: their references in proportion)
        let fitted = need_triangles.min(self.triangle_capacity).max(1);
        let need_refs = (self.stats.references as u64 * need_triangles.max(1) as u64 / fitted as u64).min(u32::MAX as u64) as u32;
        if need_triangles > self.triangle_capacity || need_refs > self.reference_capacity {
            let triangles = if need_triangles > self.triangle_capacity { grow(need_triangles) } else { self.triangle_capacity };
            let refs = if need_refs > self.reference_capacity { grow(need_refs) } else { self.reference_capacity };
            self.resize(device, triangles, refs);
        }
        self.ensure_bind_groups(device);
        self.write_uniform(queue);
        let (_, big_base, _) = self.cells_layout();
        encoder.clear_buffer(&self.counters, 0, Some(16));
        // the cells' counts, the macro cells and the big list's count
        encoder.clear_buffer(&self.cells, 0, Some((big_base as u64 + 1) * 4));
        self.gathered = Some(false);
    }

    /// Write the box and the buffers' layout, if they changed.
    fn write_uniform(&mut self, queue: &wgpu::Queue) {
        let (macro_base, big_base, refs_base) = self.cells_layout();
        let o = self.options;
        let gpu = RtGridGpu {
            origin: self.origin.to_array(),
            cell: o.cell,
            dims: o.dims,
            flags: o.macro_skip as u32,
            macro_dims: o.macro_dims(),
            epsilon: o.effective_epsilon(),
            cell_count: o.cell_count(),
            macro_base,
            big_base,
            refs_base,
            ref_capacity: self.reference_capacity,
            triangle_capacity: self.triangle_capacity,
            big_capacity: o.big_triangle_capacity,
            big_cells: o.big_triangle_cells,
        };
        if self.written != Some(gpu) {
            queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&gpu));
            self.written = Some(gpu);
        }
    }

    /// The gather pipeline for a placement, of meshes or of cluster cuts.
    fn gather_pipeline(&mut self, device: &wgpu::Device, placement: &str, clusters: bool) -> wgpu::ComputePipeline {
        let key = (placement.to_string(), clusters);
        if let Some(p) = self.gather_pipelines.get(&key) {
            return p.clone();
        }
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RtGrid/Gather"),
            source: wgpu::ShaderSource::Wgsl(gather_wgsl(placement).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RtGrid/Gather"),
            layout: Some(&self.gather_layout),
            module: &module,
            entry_point: Some(if clusters { "gather_clusters" } else { "gather" }),
            compilation_options: Default::default(),
            cache: None,
        });
        self.gather_pipelines.insert(key, pipeline.clone());
        pipeline
    }

    /// Append the triangles of `sources` that meet the box (once a build, between `begin` and
    /// `finish`: their parameters go up in one write, which lands before the frame's work).
    pub fn gather(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, sources: &[RtSource]) {
        let sources: Vec<GatherSource> = sources
            .iter()
            .map(|s| GatherSource {
                mesh: GatherMesh::Mesh { buffer: s.mesh, triangles: s.triangles },
                records: s.records,
                first_record: s.first_record,
                count: GatherCount::Fixed(s.record_count),
                world: s.world,
                placement: s.placement.wgsl(),
                surface: s.surface,
                id: s.id,
            })
            .collect();
        self.gather_sources(device, queue, encoder, &sources);
    }

    /// `gather` for the crate's sources: each source's dispatch sized on the GPU from its count
    /// (`prepare`), then the gathers, dispatched indirectly.
    pub(crate) fn gather_sources(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, sources: &[GatherSource]) {
        assert_eq!(self.gathered, Some(false), "RtGrid::gather once a build, between begin and finish");
        self.gathered = Some(true);
        if sources.is_empty() {
            return;
        }
        let needed = sources.len() as u64 * SOURCE_STRIDE;
        if self.sources.size() < needed {
            self.sources = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Sources"),
                size: needed.next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.source_groups.clear();
        }
        if self.gather_dispatch.size() < sources.len() as u64 * 16 {
            self.gather_dispatch = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/GatherDispatch"),
                size: (sources.len() as u64 * 16).next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            self.gather_dispatch_group = None;
        }
        let mut bytes = vec![0u8; needed as usize];
        for (k, s) in sources.iter().enumerate() {
            let (triangles, kind) = match s.mesh {
                GatherMesh::Mesh { triangles, .. } => (triangles, 0),
                GatherMesh::Clusters { .. } => (0, 1),
            };
            let (records, count_word) = match s.count {
                GatherCount::Fixed(n) => (n, NO_WORD),
                GatherCount::Word { word, at_most, .. } => (at_most, word),
            };
            let gpu = RtSourceGpu {
                world: s.world.to_cols_array(),
                triangles,
                stride_words: s.records.map_or(0, |(_, stride)| stride / 4),
                first_record: s.first_record,
                records,
                count_word,
                surface: s.surface.surface_word(),
                albedo: s.surface.albedo_word(),
                source: (s.id & 0xfff) << 20,
                slot: k as u32,
                kind,
                _pad: [0; 2],
            };
            bytes[k * SOURCE_STRIDE as usize..][..std::mem::size_of::<RtSourceGpu>()].copy_from_slice(bytemuck::bytes_of(&gpu));
        }
        queue.write_buffer(&self.sources, 0, &bytes);
        let pipelines: Vec<wgpu::ComputePipeline> = sources.iter().map(|s| self.gather_pipeline(device, &s.placement, matches!(s.mesh, GatherMesh::Clusters { .. }))).collect();
        let keys: Vec<SourceKey> = sources.iter().map(source_key).collect();
        for (s, key) in sources.iter().zip(&keys) {
            let (bgl, uniform, empty) = (&self.gather_source_bgl, &self.sources, &self.empty);
            self.source_groups.entry(key.clone()).or_insert_with(|| {
                let size = std::num::NonZeroU64::new(std::mem::size_of::<RtSourceGpu>() as u64);
                let (mesh, draws) = match s.mesh {
                    GatherMesh::Mesh { buffer, .. } => (buffer, empty),
                    GatherMesh::Clusters { buffer, draws } => (buffer, draws),
                };
                let args = match s.count {
                    GatherCount::Word { args, .. } => args,
                    GatherCount::Fixed(_) => empty,
                };
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("RtGrid/Source"),
                    layout: bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: uniform, offset: 0, size }) },
                        entry(1, mesh),
                        entry(2, s.records.map_or(empty, |(r, _)| r)),
                        entry(3, args),
                        entry(4, draws),
                    ],
                })
            });
        }
        if self.gather_dispatch_group.is_none() {
            self.gather_dispatch_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                layout: &self.gather_prepare_bgl,
                entries: &[entry(5, &self.gather_dispatch)],
            }));
        }
        self.ensure_bind_groups(device);
        let gather = &self.bind_groups.as_ref().unwrap().2;
        let mut pass = timed_pass(encoder, "Rt/Gather");
        pass.set_pipeline(&self.gather_prepare);
        pass.set_bind_group(0, self.gather_dispatch_group.as_ref().unwrap(), &[]);
        for (k, key) in keys.iter().enumerate() {
            pass.set_bind_group(1, &self.source_groups[key], &[(k as u64 * SOURCE_STRIDE) as u32]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_bind_group(0, gather, &[]);
        for (k, (key, pipeline)) in keys.iter().zip(&pipelines).enumerate() {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &self.source_groups[key], &[(k as u64 * SOURCE_STRIDE) as u32]);
            pass.dispatch_workgroups_indirect(&self.gather_dispatch, k as u64 * 16);
        }
    }

    /// Build the cells from the gathered triangles: count (and the big list), scan, fill.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        assert!(self.gathered.is_some(), "RtGrid::finish without begin");
        self.gathered = None;
        self.stats.builds += 1;
        let (build, prepare, _) = self.bind_groups.as_ref().expect("begin first");
        {
            let mut p = timed_pass(encoder, "Rt/GridCount");
            p.set_pipeline(&self.prepare);
            p.set_bind_group(0, prepare, &[]);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.count);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch, 0);
            p.set_pipeline(&self.prepare_wide);
            p.set_bind_group(0, prepare, &[]);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.count_wide);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch, 16);
        }
        {
            let cells = self.options.cell_count();
            let mut p = timed_pass(encoder, "Rt/GridScan");
            p.set_bind_group(0, build, &[]);
            p.set_pipeline(&self.scan_blocks);
            p.dispatch_workgroups(cells.div_ceil(1024), 1, 1);
            p.set_pipeline(&self.scan_sums);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.scan_add);
            p.dispatch_workgroups(cells.div_ceil(256), 1, 1);
        }
        {
            let mut p = timed_pass(encoder, "Rt/GridFill");
            p.set_pipeline(&self.fill);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&self.dispatch, 0);
            p.set_pipeline(&self.fill_wide);
            p.dispatch_workgroups_indirect(&self.dispatch, 16);
        }
    }

    /// After the submit of a build: read back what it needed (the gathered triangles, the
    /// references, the big triangles), unless a readback is still in flight. `begin` takes it.
    pub fn read_back(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.readback.pending.is_some() {
            return;
        }
        let (_, big_base, _) = self.cells_layout();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RtGrid/Readback") });
        encoder.copy_buffer_to_buffer(&self.counters, 0, &self.readback.staging, 0, 8);
        encoder.copy_buffer_to_buffer(&self.cells, big_base as u64 * 4, &self.readback.staging, 8, 4);
        queue.submit(Some(encoder.finish()));
        let state = Arc::new(AtomicU8::new(MAPPING));
        let done = state.clone();
        self.readback.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
        self.readback.pending = Some(state);
    }

    /// Take a readback that finished into `stats` (`begin` does too).
    pub fn poll_readback(&mut self, device: &wgpu::Device) {
        self.collect_readback(device);
    }

    /// Take a finished readback into `stats`.
    fn collect_readback(&mut self, device: &wgpu::Device) {
        #[cfg(not(target_arch = "wasm32"))]
        if self.readback.pending.is_some() {
            device.poll(wgpu::Maintain::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let Some(state) = self.readback.pending.take_if(|s| s.load(Ordering::Acquire) != MAPPING) else { return };
        if state.load(Ordering::Acquire) == FAILED {
            return;
        }
        {
            let bytes = self.readback.staging.slice(..).get_mapped_range();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            self.stats.triangles = words[0];
            self.stats.references = words[1];
            self.stats.big_triangles = words[2];
        }
        self.readback.staging.unmap();
    }

    /// Wait for the readback in flight and take it (native; for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_readback(&mut self, device: &wgpu::Device) {
        if self.readback.pending.is_some() {
            device.poll(wgpu::Maintain::Wait);
        }
        self.collect_readback(device);
    }

    /// WGSL declaring the grid's buffers in group `group` from binding `first` (three bindings),
    /// for `RT_GRID_WGSL`.
    pub fn bindings_wgsl(group: u32, first: u32) -> String {
        format!(
            "@group({group}) @binding({}) var<uniform> kansei_rt_grid : KanseiRtGrid;\n\
             @group({group}) @binding({}) var<storage, read> kansei_rt_triangles : array<vec4f>;\n\
             @group({group}) @binding({}) var<storage, read> kansei_rt_cells : array<u32>;\n",
            first,
            first + 1,
            first + 2
        )
    }

    /// The layout entries for `bindings_wgsl(_, first)`.
    pub fn layout_entries(first: u32, visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 3] {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        [buffer(first, wgpu::BufferBindingType::Uniform), buffer(first + 1, ro), buffer(first + 2, ro)]
    }

    /// The buffers for `bindings_wgsl(_, first)`, in order.
    pub fn bind_group_entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry { binding: first, resource: self.uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 1, resource: self.triangles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 2, resource: self.cells.as_entire_binding() },
        ]
    }

    /// The world triangles (64 bytes each, rt_types.wgsl), for tests and debugging.
    pub fn triangles_buffer(&self) -> &wgpu::Buffer {
        &self.triangles
    }

    /// The cell words (rt_types.wgsl), for tests and debugging.
    pub fn cells_buffer(&self) -> &wgpu::Buffer {
        &self.cells
    }

    /// Where the macro cells, the big list and the references start in the cell words.
    pub fn cells_layout_words(&self) -> (u32, u32, u32) {
        self.cells_layout()
    }
}

/// The buffers of a source's bind group.
fn source_key(s: &GatherSource) -> SourceKey {
    let (mesh, draws) = match s.mesh {
        GatherMesh::Mesh { buffer, .. } => (buffer.clone(), None),
        GatherMesh::Clusters { buffer, draws } => (buffer.clone(), Some(draws.clone())),
    };
    let args = match s.count {
        GatherCount::Word { args, .. } => Some(args.clone()),
        GatherCount::Fixed(_) => None,
    };
    (mesh, s.records.map(|(r, _)| r.clone()), args, draws)
}

/// A bind group entry for the whole of `buffer`.
fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry { binding, resource: buffer.as_entire_binding() }
}

/// A compute pass timed by the profiler as `label`.
fn timed_pass<'e>(encoder: &'e mut wgpu::CommandEncoder, label: &'static str) -> wgpu::ComputePass<'e> {
    let stamp = crate::profiling::gpu_pass(label);
    encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) })
}

/// A capacity for a need: a quarter more, rounded up to a 64th of its power of two.
fn grow(need: u32) -> u32 {
    let want = (need as u64 * 5 / 4).max(64);
    let step = (want.next_power_of_two() / 64).max(1);
    want.div_ceil(step).saturating_mul(step).min(u32::MAX as u64) as u32
}

/// The gather's WGSL with a placement's `kansei_rt_place`.
pub(crate) fn gather_wgsl(placement: &str) -> String {
    format!("{GATHER_WGSL}\n{placement}")
}
