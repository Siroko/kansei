//! `RtGrid`: a uniform grid of world triangles in a box, rebuilt on the GPU (see the module docs).

use std::collections::HashMap;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3};

use super::query::{RtQuery, TlasState};
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
/// The layer flag (rt_types.wgsl `KANSEI_RT_CLIP`) of a build that keeps only what its uniform's
/// second layer's box meets too.
const CLIP: u32 = 2;

/// The WGSL `KanseiRtLayer` (rt_types.wgsl): a layer's box, cells and regions of the buffers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct RtLayerGpu {
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
    cells_base: u32,
    triangle_base: u32,
    pad: [u32; 2],
}

/// The WGSL `KanseiRtGrid` (rt_types.wgsl): the static layer, then the moving one.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub(crate) struct RtGridGpu {
    layer: RtLayerGpu,
    moving: RtLayerGpu,
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
    /// `RtSurface::record_albedo` and `record_emission`, plus one (0: none).
    albedo_record_word: u32,
    emission_record_word: u32,
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
    /// What `kansei_rt_trace` finds hits with: the cells (software) or hardware ray queries,
    /// picked once at `RtGrid::new` (`Auto`: hardware where the device has it and it is trusted).
    pub backend: RtBackend,
    /// Cells the moving layer (`RtGrid::begin_moving`) has at most: its box is cut into cells as
    /// fine as `cell` where they fit, coarser where its box is too big for them.
    pub moving_cells: u32,
    /// Triangles and references the moving layer's buffers start with room for; they grow too.
    pub moving_triangle_capacity: u32,
    pub moving_reference_capacity: u32,
}

/// What traces an `RtGrid`'s rays: its cells walked in a compute shader (`RT_GRID_WGSL`, the web's
/// only one), or hardware ray queries over an acceleration structure of the same triangles
/// (`RT_QUERY_WGSL`, native: wgpu's `EXPERIMENTAL_RAY_QUERY`, which the renderer's device asks
/// for where the adapter has it). Both return the same `KanseiRtHit` for the same triangle ids.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum RtBackend {
    /// Hardware on Vulkan, and on Metal on Apple GPUs with ray tracing hardware (M3, A17 on);
    /// software elsewhere: on the web, on M1 and M2, and on DX12, where NVIDIA's driver 617.42
    /// miscompiles a ray query. `KANSEI_RT_BACKEND=software|hardware|auto` overrides it on native.
    #[default]
    Auto,
    /// The cells, whatever the device has (comparisons, and the hardware path's fallback).
    Software,
    /// Hardware ray queries wherever the device has them, DX12 included; software where it hasn't.
    Hardware,
}

impl RtBackend {
    /// The backend this one traces with on `device`: `Software` or `Hardware`, by the rules
    /// above (`Auto`, `KANSEI_RT_BACKEND`, and software where the device has no ray queries).
    /// `RtGrid::new` resolves its options' backend so, and the path tracer's scene too
    /// (`pathtracer::BVHBuilder::set_backend`).
    pub fn resolve(self, device: &wgpu::Device) -> RtBackend {
        super::query::resolve(self, device)
    }

    /// Its `kansei_rt_trace` library (`Auto` has none: ask the grid, `RtGrid::backend`).
    pub fn wgsl(self) -> &'static str {
        match self {
            Self::Hardware => super::RT_QUERY_WGSL,
            _ => super::RT_GRID_WGSL,
        }
    }

    /// WGSL declaring its bindings in group `group` from binding `first` (three: the grid's
    /// uniform, its triangles, and its cells or its acceleration structure).
    pub fn bindings_wgsl(self, group: u32, first: u32) -> String {
        match self {
            Self::Hardware => format!(
                "@group({group}) @binding({}) var<uniform> kansei_rt_grid : KanseiRtGrid;\n\
                 @group({group}) @binding({}) var<storage, read> kansei_rt_triangles : array<vec4f>;\n\
                 @group({group}) @binding({}) var kansei_rt_tlas : acceleration_structure;\n",
                first,
                first + 1,
                first + 2
            ),
            _ => RtGrid::bindings_wgsl(group, first),
        }
    }

    /// The layout entries for `bindings_wgsl(_, first)`.
    pub fn layout_entries(self, first: u32, visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 3] {
        let mut entries = RtGrid::layout_entries(first, visibility);
        if self == Self::Hardware {
            entries[2].ty = wgpu::BindingType::AccelerationStructure { vertex_return: false };
        }
        entries
    }
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
            backend: RtBackend::Auto,
            moving_cells: 1 << 18,
            moving_triangle_capacity: 1 << 12,
            moving_reference_capacity: 1 << 15,
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
}

/// What the last build read back needed, and the room the buffers have: the static layer's, then
/// the moving layer's.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtGridStats {
    /// Triangles gathered (meeting the box), whether they fitted or not.
    pub triangles: u32,
    /// References the cells needed (0 with the hardware backend, which has none).
    pub references: u32,
    /// Triangles too wide for the cells (those up to `big_triangle_capacity` in the big list).
    pub big_triangles: u32,
    pub triangle_capacity: u32,
    pub reference_capacity: u32,
    /// Builds recorded.
    pub builds: u64,
    /// The triangles the hardware backend's BLAS held at the build read back (sized from the
    /// readback before, with room to grow): those gathered past it were left out, and
    /// `SceneRtGrid` rebuilds. 0 in software.
    pub accel_triangles: u32,
    /// The moving layer's triangles gathered, references needed and room.
    pub moving_triangles: u32,
    pub moving_references: u32,
    pub moving_triangle_capacity: u32,
    pub moving_reference_capacity: u32,
    pub moving_builds: u64,
    /// Its cells now (0: empty), and where its triangles start in the triangles' buffer.
    pub moving_cells: u32,
    pub moving_triangle_base: u32,
}

/// The buffers a pass traces an `RtGrid` through, which follow the grid when its buffers are made
/// anew (`RtGrid::handle`): an effect holds one and binds what it holds each frame.
#[derive(Clone)]
pub struct RtGridHandle {
    shared: Arc<std::sync::Mutex<GridBuffers>>,
}

#[derive(Clone)]
struct GridBuffers {
    bindings: RtGridBindings,
    generation: u64,
    /// the hardware backend's TLAS, built before the first trace after a rebuild
    tlas: Option<Arc<std::sync::Mutex<TlasState>>>,
}

/// What a pass binds to trace an `RtGrid` (`RtGridHandle::bindings`).
#[derive(Clone)]
pub struct RtGridBindings {
    pub backend: RtBackend,
    pub uniform: wgpu::Buffer,
    pub triangles: wgpu::Buffer,
    /// The cells (software), or the acceleration structure (hardware).
    pub cells: wgpu::Buffer,
    pub tlas: Option<wgpu::Tlas>,
}

impl RtGridBindings {
    /// The entries for `self.backend.bindings_wgsl(_, first)`, in order.
    pub fn entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        let third = match &self.tlas {
            Some(tlas) => wgpu::BindingResource::AccelerationStructure(tlas),
            None => self.cells.as_entire_binding(),
        };
        [
            wgpu::BindGroupEntry { binding: first, resource: self.uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 1, resource: self.triangles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 2, resource: third },
        ]
    }
}

impl RtGridHandle {
    /// What `RtGrid::bindings_wgsl` (or the hardware backend's) binds, and the grid's generation
    /// (changed when they are made anew).
    pub fn bindings(&self) -> (RtGridBindings, u64) {
        let b = self.shared.lock().unwrap();
        (b.bindings.clone(), b.generation)
    }

    /// Ready the grid for tracing after a rebuild (`RtGrid::prepare_trace`): the effects call it
    /// before their passes.
    pub fn prepare_trace(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let tlas = self.shared.lock().unwrap().tlas.clone();
        if let Some(t) = tlas {
            t.lock().unwrap().build(device, queue);
        }
    }

    /// The grid's backend (never `Auto`): the WGSL and layout a pass traces it with.
    pub fn backend(&self) -> RtBackend {
        self.shared.lock().unwrap().bindings.backend
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
    /// Its triangles carry their vertex normals (in place of their uvs), which
    /// `kansei_rt_shading_normal` interpolates at a hit: for the rays a curved surface bends or
    /// mirrors (glass, chrome). Its uvs are then gone: only for surfaces no `kansei_rt_covered`
    /// samples by uv.
    pub smooth_normals: bool,
    /// Glass (`RtSurface::glass`): rays pass straight through it (diffuse GI, shadow rays, any
    /// trace without `KANSEI_RT_GLASS_HITS`); `RtReflectionsEffect`'s glass refracts at it.
    pub glass: bool,
    /// Its material lights it on screen itself (not `RtShadowsEffect` or the GI's composites), so
    /// the GBuffer's background image (`GBuffer::background_view`, before the refractive pass)
    /// holds its light: `RtReflectionsEffect` takes a hit there from it where the camera sees the
    /// point only through what that pass draws (the bed under ray-traced water), not tracing its
    /// lights.
    pub self_lit: bool,
    /// Its triangles are a coarse copy of a surface the camera draws finely (a fluid's surface,
    /// `FluidHeightfield::rt_mesh`): `RtReflectionsEffect`'s glass paths that leave the fine
    /// surface pass through the copy within `RT_APPROXIMATE_SKIN` of where they leave it, so the
    /// copy lying a little above or below doesn't bend them twice.
    pub approximate: bool,
    /// A layer of the albedo texture array the reflections' hits read by uv
    /// (`RtReflectionsEffect::set_albedo_texture`), times `albedo`: a textured surface in a
    /// mirror. Needs the triangles' uvs (not `smooth_normals`).
    pub albedo_layer: Option<u8>,
    /// An instanced surface's albedo per instance, in place of `albedo`: the word of each instance
    /// record (0 the first) holding it as `pack4x8unorm` (alpha unused), for instances the GPU
    /// colours (particles).
    pub record_albedo: Option<u32>,
    /// The light an instanced surface emits per instance (cd/m²): the word of each instance
    /// record holding it packed RGBE (`RT_EMISSION_WGSL`'s `kansei_rt_pack_emission`). The hits
    /// the rays shade themselves (those the camera doesn't see) add it; `None`: no emission.
    pub record_emission: Option<u32>,
    /// Its reflectance at normal incidence (F0: 0.04 for a polished dielectric, 0 matte, the
    /// default): the hits the rays light themselves add its Fresnel reflection of the sky, the
    /// rest of their light scaled by what it leaves (`RtReflectionsEffect`'s and its glass's). A
    /// polished floor seen in a mirror or through glass reflects the sky at grazing angles, as it
    /// does on screen.
    pub specular: f32,
}

impl RtSurface {
    pub fn new(albedo: [f32; 3]) -> Self {
        Self { albedo, alpha_layer: None, smooth_normals: false, glass: false, albedo_layer: None, self_lit: false, approximate: false, record_albedo: None, record_emission: None, specular: 0.0 }
    }

    /// The surface of a glass object (the triangles `RtReflectionsEffect` refracts through, with
    /// smooth normals) whose material is `StandardLitOptions::glass`: `tint` its albedo, for the
    /// rays that pass through it.
    pub fn glass(tint: [f32; 3]) -> Self {
        Self { albedo: tint, alpha_layer: None, smooth_normals: true, glass: true, albedo_layer: None, self_lit: false, approximate: false, record_albedo: None, record_emission: None, specular: 0.0 }
    }

    /// See `approximate`.
    pub fn with_approximate(mut self) -> Self {
        self.approximate = true;
        self
    }

    /// See `self_lit`.
    pub fn with_self_lit(mut self) -> Self {
        self.self_lit = true;
        self
    }

    /// See `specular`.
    pub fn with_specular(mut self, f0: f32) -> Self {
        self.specular = f0.clamp(0.0, 1.0);
        self
    }

    /// See `record_albedo` and `record_emission` (word offsets in the instance record; `None`
    /// leaves that one as it is).
    pub fn with_record_surface(mut self, albedo_word: Option<u32>, emission_word: Option<u32>) -> Self {
        self.record_albedo = albedo_word.or(self.record_albedo);
        self.record_emission = emission_word.or(self.record_emission);
        self
    }

    /// See `smooth_normals`.
    pub fn with_smooth_normals(mut self) -> Self {
        self.smooth_normals = true;
        self
    }

    /// Alpha tested: `kansei_rt_covered(layer, uv)` says where it is there.
    pub fn with_alpha_layer(mut self, layer: u8) -> Self {
        self.alpha_layer = Some(layer);
        self
    }

    /// See `albedo_layer`.
    pub fn with_albedo_layer(mut self, layer: u8) -> Self {
        self.albedo_layer = Some(layer);
        self
    }

    pub(crate) fn surface_word(&self) -> u32 {
        self.alpha_layer.map_or(0, |layer| 1 | (layer as u32) << 8)
            | if self.smooth_normals { 4 } else { 0 }
            | if self.glass { 8 } else { 0 }
            | if self.self_lit { 32 } else { 0 }
            | if self.approximate { 64 } else { 0 }
            | self.albedo_layer.filter(|_| !self.smooth_normals).map_or(0, |layer| 16 | (layer as u32) << 16)
    }

    pub(crate) fn albedo_word(&self) -> u32 {
        let byte = |x: f32| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
        byte(self.albedo[0]) | byte(self.albedo[1]) << 8 | byte(self.albedo[2]) << 16 | byte(1.0 - self.specular) << 24
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
    /// each layer's hardware BLAS's triangles at the build in flight
    accel_triangles: [u32; 2],
    /// readbacks taken
    taken: u64,
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// The static layer, and the moving one.
const STATIC: usize = 0;
const MOVING: usize = 1;

/// One of the grid's two layers (rt_types.wgsl's `KanseiRtLayer`): its box and cells, its regions
/// of the triangles and the cell words, and its own counters, build uniform, sources and BLAS.
struct Layer {
    /// the box's corner, a cell's size and the cells each way: the static layer's from the
    /// options, the moving layer's fitted round what moves at each build (no cells: empty)
    origin: Vec3,
    cell: f32,
    dims: [u32; 3],
    /// the uniform its build reads, this layer as `KanseiRtGrid`'s own fields (the static layer's
    /// is the grid's uniform, which the traces read)
    uniform: wgpu::Buffer,
    written: Option<RtGridGpu>,
    counters: wgpu::Buffer,
    dispatch: wgpu::Buffer,
    sources: wgpu::Buffer,
    /// the sources' indirect dispatches, and the bind group writing them
    gather_dispatch: wgpu::Buffer,
    gather_dispatch_group: Option<wgpu::BindGroup>,
    source_groups: HashMap<SourceKey, wgpu::BindGroup>,
    bind_groups: Option<(wgpu::BindGroup, wgpu::BindGroup, wgpu::BindGroup)>,
    /// cells its region of the cell words has room for (a multiple of 64)
    cell_capacity: u32,
    triangle_capacity: u32,
    reference_capacity: u32,
    /// what the last readback said: triangles gathered, references needed, big triangles
    need: [u32; 3],
    builds: u64,
    /// the hardware backend's BLAS: rebuilt for speed of building or of tracing, whether the last
    /// gather held non-opaque sources (its second geometry), its triangles this build
    query: Option<RtQuery>,
    fast_build: bool,
    split: bool,
    accel_slots: u32,
    accel_read: u32,
}

impl Layer {
    fn new(device: &wgpu::Device, uniform: wgpu::Buffer, cell_capacity: u32, query: Option<RtQuery>) -> Self {
        let storage = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | usage, mapped_at_creation: false });
        Self {
            origin: Vec3::ZERO,
            cell: 1.0,
            dims: [0; 3],
            uniform,
            written: None,
            counters: storage("RtGrid/Counters", 16, wgpu::BufferUsages::empty()),
            dispatch: storage("RtGrid/Dispatch", 32, wgpu::BufferUsages::INDIRECT),
            sources: device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtGrid/Sources"), size: SOURCE_STRIDE, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
            gather_dispatch: storage("RtGrid/GatherDispatch", 16, wgpu::BufferUsages::INDIRECT),
            gather_dispatch_group: None,
            source_groups: HashMap::new(),
            bind_groups: None,
            cell_capacity,
            triangle_capacity: 0,
            reference_capacity: 0,
            need: [0; 3],
            builds: 0,
            query,
            fast_build: false,
            split: false,
            accel_slots: 0,
            accel_read: 0,
        }
    }

    fn cell_count(&self) -> u32 {
        self.dims.iter().product()
    }

    /// Words of its region of the cell words before the references: its cells, its macro cells
    /// and its big list.
    fn head_words(&self, big_capacity: u32) -> u32 {
        self.cell_capacity + self.cell_capacity / (MACRO * MACRO * MACRO) + 1 + big_capacity
    }
}

/// A uniform grid of world triangles in a box round the camera (or a fixed one), rebuilt on the
/// GPU, for rays: see the module docs. Each rebuild, between `begin` and `finish`, one `gather`
/// appends the sources' triangles; after the frame's submit, `read_back` reads what it needed.
///
/// It has two layers in the same buffers, which every trace tests, keeping the nearer hit: the
/// static one (`begin`), rebuilt only when what it holds changes, and the moving one
/// (`begin_moving`), a small grid fitted round what moves and rebuilt every frame, so the moving
/// triangles cost a frame what they hold rather than what the scene holds. `clear_moving` empties
/// it. A layer's buffers grow without losing the other's, except that the static layer's growing
/// empties the moving one until its next build.
pub struct RtGrid {
    options: RtGridOptions,
    origin: Vec3,
    placed: bool,
    triangles: wgpu::Buffer,
    cells: wgpu::Buffer,
    empty: wgpu::Buffer,
    layers: [Layer; 2],
    build_bgl: wgpu::BindGroupLayout,
    prepare_bgl: wgpu::BindGroupLayout,
    gather_grid_bgl: wgpu::BindGroupLayout,
    gather_source_bgl: wgpu::BindGroupLayout,
    gather_prepare_bgl: wgpu::BindGroupLayout,
    gather_prepare: wgpu::ComputePipeline,
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
    readback: Readback,
    generation: u64,
    handle: RtGridHandle,
    /// between `begin` (or `begin_moving`) and `finish`: the layer, and whether it gathered yet
    building: Option<(usize, bool)>,
    /// what traces it (never `Auto`), and the hardware backend's TLAS of the layers' BLASes
    backend: RtBackend,
    tlas_state: Option<Arc<std::sync::Mutex<TlasState>>>,
    /// its TLAS, to bind
    tlas: Option<wgpu::Tlas>,
}

impl RtGrid {
    pub fn new(device: &wgpu::Device, options: RtGridOptions) -> Self {
        assert!(options.dims.iter().all(|&d| d > 0 && d.is_multiple_of(MACRO)), "grid dims must be multiples of {MACRO}");
        assert!(options.cell_count() <= RT_MAX_CELLS, "at most {RT_MAX_CELLS} cells");
        let backend = super::query::resolve(options.backend, device);
        log::info!("rt grid: {backend:?} backend (asked for {:?})", options.backend);
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
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("RtGrid/Build"), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
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
            bind_group_layouts: &[Some(&gather_grid_bgl), Some(&gather_source_bgl)],
            immediate_size: 0,
        });
        let gather_prepare = {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                source: wgpu::ShaderSource::Wgsl(gather_wgsl(&RtPlacement::None.wgsl()).into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                bind_group_layouts: &[Some(&gather_prepare_bgl), Some(&gather_source_bgl)],
                immediate_size: 0,
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
        let uniform_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: std::mem::size_of::<RtGridGpu>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
        };
        let tlas_state = (backend == RtBackend::Hardware).then(|| RtQuery::new_tlas(device));
        let query = |slot| tlas_state.as_ref().map(|t| RtQuery::new(device, t.clone(), slot));
        let moving_cells = (options.moving_cells / 64 * 64).clamp(64, RT_MAX_CELLS);
        let mut layers = [Layer::new(device, uniform_buffer("RtGrid/Uniform"), options.cell_count(), query(STATIC)), Layer::new(device, uniform_buffer("RtGrid/MovingUniform"), moving_cells, query(MOVING))];
        layers[STATIC].cell = options.cell;
        layers[STATIC].dims = options.dims;
        let mut grid = Self {
            options,
            origin: options.fixed_origin.unwrap_or(Vec3::ZERO),
            placed: false,
            triangles: storage("RtGrid/Triangles", 16, wgpu::BufferUsages::empty()),
            cells: storage("RtGrid/Cells", 16, wgpu::BufferUsages::empty()),
            empty: storage("RtGrid/Empty", 16, wgpu::BufferUsages::empty()),
            layers,
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
            gather_layout,
            gather_pipelines: HashMap::new(),
            readback: Readback {
                staging: device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtGrid/Readback"), size: 32, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false }),
                pending: None,
                accel_triangles: [0; 2],
                taken: 0,
            },
            generation: 0,
            handle: RtGridHandle {
                shared: Arc::new(std::sync::Mutex::new(GridBuffers {
                    bindings: RtGridBindings {
                        backend,
                        uniform: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()),
                        triangles: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()),
                        cells: storage("RtGrid/Placeholder", 16, wgpu::BufferUsages::empty()),
                        tlas: None,
                    },
                    generation: 0,
                    tlas: None,
                })),
            },
            building: None,
            tlas: tlas_state.as_ref().map(|t| t.lock().unwrap().tlas().clone()),
            tlas_state,
            backend,
        };
        let (triangles, references) = (options.triangle_capacity.max(64), options.reference_capacity.max(1024));
        let (moving_triangles, moving_references) = (options.moving_triangle_capacity.max(64), options.moving_reference_capacity.max(1024));
        grid.resize(device, None, [triangles, moving_triangles], [references, moving_references]);
        grid
    }

    pub fn options(&self) -> &RtGridOptions {
        &self.options
    }

    /// What traces it (never `Auto`): the WGSL (`RtBackend::wgsl`, `bindings_wgsl`) and layout
    /// (`layout_entries`) a pass traces it with.
    pub fn backend(&self) -> RtBackend {
        self.backend
    }

    /// The hardware backend's next static BLAS builds fast rather than tracing fast: for a static
    /// layer rebuilt every frame (`SceneRtGrid` sets it while its box moves). The moving layer's
    /// always builds fast.
    pub fn set_prefer_fast_build(&mut self, on: bool) {
        self.layers[STATIC].fast_build = on;
    }

    /// The box's corner.
    pub fn origin(&self) -> Vec3 {
        self.origin
    }

    /// The box: its least and greatest corners.
    pub fn bounds(&self) -> (Vec3, Vec3) {
        (self.origin, self.origin + self.options.extent())
    }

    /// The moving layer's box (the static box while it is empty).
    pub fn moving_bounds(&self) -> (Vec3, Vec3) {
        let m = &self.layers[MOVING];
        if m.cell_count() == 0 {
            return self.bounds();
        }
        (m.origin, m.origin + Vec3::new(m.dims[0] as f32, m.dims[1] as f32, m.dims[2] as f32) * m.cell)
    }

    /// The box of the layer being built (between `begin` or `begin_moving` and `finish`), whose
    /// triangles a gather keeps; the static layer's otherwise.
    pub fn gather_bounds(&self) -> (Vec3, Vec3) {
        match self.building {
            Some((MOVING, _)) => self.moving_bounds(),
            _ => self.bounds(),
        }
    }

    /// What the last readback said, and the room the buffers have.
    pub fn stats(&self) -> RtGridStats {
        let (s, m) = (&self.layers[STATIC], &self.layers[MOVING]);
        RtGridStats {
            triangles: s.need[0],
            references: s.need[1],
            big_triangles: s.need[2],
            triangle_capacity: s.triangle_capacity,
            reference_capacity: s.reference_capacity,
            builds: s.builds,
            accel_triangles: s.accel_read,
            moving_triangles: m.need[0],
            moving_references: m.need[1],
            moving_triangle_capacity: m.triangle_capacity,
            moving_reference_capacity: m.reference_capacity,
            moving_builds: m.builds,
            moving_cells: m.cell_count(),
            moving_triangle_base: s.triangle_capacity,
        }
    }

    /// Whether the static layer's build read back gathered more triangles than it had room for
    /// (the buffers', or the hardware BLAS's): rebuild. (The moving layer, rebuilt every frame,
    /// grows by itself.)
    pub fn overflowed(&self) -> bool {
        let s = self.stats();
        s.triangles > s.triangle_capacity || s.references > s.reference_capacity || (self.tlas_state.is_some() && s.triangles > s.accel_triangles)
    }

    /// Changes whenever the buffers `bind_group_entries` binds are made anew: bind groups made
    /// with the old ones are stale.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Bytes on the GPU: the triangles, the cell words, the layers' counters (with the wide
    /// triangles' lists) and sources' parameters.
    pub fn memory_bytes(&self) -> u64 {
        self.triangles.size() + self.cells.size() + self.layers.iter().map(|l| l.counters.size() + l.sources.size() + l.query.as_ref().map_or(0, RtQuery::memory_bytes)).sum::<u64>()
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

    /// Where layer `k`'s cell words start, and its macro cells, big list and references.
    fn cells_layout_of(&self, k: usize) -> (u32, u32, u32, u32) {
        let big = self.options.big_triangle_capacity;
        let s = &self.layers[STATIC];
        let base = if k == STATIC { 0 } else { s.head_words(big) + s.reference_capacity };
        let l = &self.layers[k];
        let macro_base = base + l.cell_capacity;
        let big_base = macro_base + l.cell_capacity / (MACRO * MACRO * MACRO);
        (base, macro_base, big_base, big_base + 1 + big)
    }

    fn cells_layout(&self) -> (u32, u32, u32) {
        let (_, macro_base, big_base, refs_base) = self.cells_layout_of(STATIC);
        (macro_base, big_base, refs_base)
    }

    /// The most a binding holds, in bytes.
    fn binding_limit(device: &wgpu::Device) -> u64 {
        let limits = device.limits();
        (limits.max_storage_buffer_binding_size as u64).min(limits.max_buffer_size)
    }

    /// Give the layers room for `triangles` and `references` each (within what a binding holds).
    /// With `encoder`, a layer whose region stays put keeps its contents (copied into the new
    /// buffers: only the static layer's can, the moving layer's starting past it); without, or
    /// when the static layer grows, the moving layer is emptied until its next build.
    fn resize(&mut self, device: &wgpu::Device, mut encoder: Option<&mut wgpu::CommandEncoder>, triangles: [u32; 2], references: [u32; 2]) {
        let limit = Self::binding_limit(device);
        let big = self.options.big_triangle_capacity;
        let hardware = self.backend == RtBackend::Hardware;
        // the static layer's region before, which a copy keeps
        let (old_triangles, old_cells) = (self.layers[STATIC].triangle_capacity as u64 * RT_TRIANGLE_BYTES, self.cells_layout_of(MOVING).0 as u64 * 4);
        let triangles = {
            let static_max = (limit / RT_TRIANGLE_BYTES).saturating_sub(triangles[MOVING] as u64).max(64);
            [(triangles[STATIC] as u64).min(static_max) as u32, (triangles[MOVING] as u64).min(limit / RT_TRIANGLE_BYTES - (triangles[STATIC] as u64).min(static_max)) as u32]
        };
        // (the hardware backend lists no references: it keeps the 16-byte placeholder)
        let references = if hardware {
            [0, 0]
        } else {
            let heads = (self.layers[STATIC].head_words(big) + self.layers[MOVING].head_words(big)) as u64;
            let room = limit / 4 - heads;
            let s = (references[STATIC] as u64).min(room - (references[MOVING] as u64).min(room / 2));
            [s as u32, (references[MOVING] as u64).min(room - s) as u32]
        };
        let static_moves = triangles[STATIC] != self.layers[STATIC].triangle_capacity || (!hardware && references[STATIC] != self.layers[STATIC].reference_capacity);
        let mut changed = false;
        if triangles != [self.layers[STATIC].triangle_capacity, self.layers[MOVING].triangle_capacity] {
            let old = std::mem::replace(
                &mut self.triangles,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("RtGrid/Triangles"),
                    size: (triangles[STATIC] as u64 + triangles[MOVING] as u64) * RT_TRIANGLE_BYTES,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
            );
            if let (Some(e), false, true) = (encoder.as_deref_mut(), static_moves, old_triangles > 0) {
                e.copy_buffer_to_buffer(&old, 0, &self.triangles, 0, old_triangles);
            }
            for (layer, &n) in self.layers.iter_mut().zip(&triangles) {
                if layer.triangle_capacity != n {
                    layer.triangle_capacity = n;
                    // (every triangle may be wide)
                    layer.counters = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("RtGrid/Counters"),
                        size: (COUNTER_WORDS + n as u64) * 4,
                        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    });
                }
            }
            changed = true;
        }
        if !hardware && references != [self.layers[STATIC].reference_capacity, self.layers[MOVING].reference_capacity] {
            self.layers[STATIC].reference_capacity = references[STATIC];
            self.layers[MOVING].reference_capacity = references[MOVING];
            let (moving_base, _, _, moving_refs) = self.cells_layout_of(MOVING);
            let old = std::mem::replace(
                &mut self.cells,
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("RtGrid/Cells"),
                    size: (moving_refs as u64 + references[MOVING] as u64) * 4,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
            );
            if let (Some(e), false, true) = (encoder.as_deref_mut(), static_moves, old.size() > 16) {
                e.copy_buffer_to_buffer(&old, 0, &self.cells, 0, old_cells.min(moving_base as u64 * 4));
            }
            changed = true;
        }
        if !changed {
            return;
        }
        if static_moves || encoder.is_none() {
            // the static layer's region moved the moving one's
            self.layers[MOVING].dims = [0; 3];
        }
        for layer in &mut self.layers {
            layer.bind_groups = None;
            layer.written = None;
        }
        self.generation += 1;
        *self.handle.shared.lock().unwrap() = GridBuffers {
            bindings: RtGridBindings { backend: self.backend, uniform: self.layers[STATIC].uniform.clone(), triangles: self.triangles.clone(), cells: self.cells.clone(), tlas: self.tlas.clone() },
            generation: self.generation,
            tlas: self.tlas_state.clone(),
        };
    }

    /// A handle to the buffers passes trace the grid through, which follows the grid's.
    pub fn handle(&self) -> RtGridHandle {
        self.handle.clone()
    }

    /// Room for `triangles` in the layer being built (call between `begin` and `gather`, when the
    /// sources know how many they hold at most): no frame then loses triangles to a readback still
    /// in flight.
    pub fn reserve_triangles(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, triangles: u32) {
        let Some((k, _)) = self.building else { panic!("RtGrid::reserve_triangles between begin and gather") };
        if self.layers[k].query.is_some() && triangles > self.layers[k].accel_slots {
            // the BLAS holds them all this build
            self.layers[k].accel_slots = triangles;
            if triangles <= self.layers[k].triangle_capacity {
                self.prepare_query(device, queue, k);
            }
        }
        if triangles > self.layers[k].triangle_capacity {
            let mut want = [self.layers[STATIC].triangle_capacity, self.layers[MOVING].triangle_capacity];
            want[k] = grow(triangles);
            let refs = [self.layers[STATIC].reference_capacity, self.layers[MOVING].reference_capacity];
            self.resize(device, Some(encoder), want, refs);
            self.ensure_bind_groups(device, k);
            self.write_uniforms(queue);
            self.prepare_query(device, queue, k);
        }
    }

    /// Ready layer `k`'s hardware BLAS for this build.
    fn prepare_query(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, k: usize) {
        let base = if k == STATIC { 0 } else { self.layers[STATIC].triangle_capacity };
        // the filler under the box, a metre clear of it
        let below = (self.origin - Vec3::Y).to_array();
        let (triangles, layer) = (&self.triangles, &mut self.layers[k]);
        if let Some(q) = layer.query.as_mut() {
            layer.accel_slots = layer.accel_slots.clamp(64.min(layer.triangle_capacity), layer.triangle_capacity);
            q.prepare(device, queue, triangles, &layer.counters, base, layer.accel_slots, layer.split, layer.fast_build, below);
        }
    }

    fn ensure_bind_groups(&mut self, device: &wgpu::Device, k: usize) {
        let (triangles, cells, layer) = (&self.triangles, &self.cells, &mut self.layers[k]);
        if layer.bind_groups.is_some() {
            return;
        }
        let build = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/Build"),
            layout: &self.build_bgl,
            entries: &[entry(0, &layer.uniform), entry(1, triangles), entry(2, cells), entry(3, &layer.counters)],
        });
        let prepare = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/Prepare"),
            layout: &self.prepare_bgl,
            entries: &[entry(0, &layer.uniform), entry(3, &layer.counters), entry(4, &layer.dispatch)],
        });
        let gather = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtGrid/GatherGrid"),
            layout: &self.gather_grid_bgl,
            entries: &[entry(0, &layer.uniform), entry(1, triangles), entry(2, &layer.counters)],
        });
        layer.bind_groups = Some((build, prepare, gather));
    }

    /// Start a rebuild of the static layer: take a finished readback (growing the buffers to what
    /// it needed), write the box, and clear the layer's triangles and cells.
    pub fn begin(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder) {
        assert!(self.building.is_none(), "RtGrid::begin twice without finish");
        self.collect_readback(device);
        self.layers[STATIC].origin = self.origin;
        self.begin_layer(device, queue, encoder, STATIC);
    }

    /// Start a rebuild of the moving layer, a grid of its own round `bounds` (within the static
    /// layer's box; None: all of it) with cells as fine as the static layer's where
    /// `RtGridOptions::moving_cells` allows: as `begin`, for the sources that move.
    pub fn begin_moving(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, bounds: Option<(Vec3, Vec3)>) {
        assert!(self.building.is_none(), "RtGrid::begin_moving twice without finish");
        let (lo, hi) = self.bounds();
        let eps = Vec3::splat(self.options.effective_epsilon());
        let (a, b) = bounds.map_or((lo, hi), |(a, b)| ((a - eps).max(lo), (b + eps).min(hi)));
        let (a, b) = if a.cmple(b).all() { (a, b) } else { (lo, lo + Vec3::splat(self.options.cell * MACRO as f32)) };
        let capacity = self.layers[MOVING].cell_capacity;
        let size = b - a;
        let mut cell = self.options.cell.max(size.max_element() / (capacity as f32 / (MACRO * MACRO) as f32).max(1.0));
        let dims = loop {
            let dims = (size / cell).ceil().max(Vec3::ONE).to_array().map(|d| (d as u32).next_multiple_of(MACRO));
            if dims.iter().map(|&d| d as u64).product::<u64>() <= capacity as u64 {
                break dims;
            }
            cell *= 1.125;
        };
        let m = &mut self.layers[MOVING];
        (m.origin, m.cell, m.dims) = (a, cell, dims);
        m.fast_build = true;
        self.begin_layer(device, queue, encoder, MOVING);
    }

    /// Empty the moving layer (nothing moves): traces test only the static one.
    pub fn clear_moving(&mut self, queue: &wgpu::Queue) {
        assert!(self.building.is_none(), "RtGrid::clear_moving while building");
        let m = &mut self.layers[MOVING];
        if m.cell_count() == 0 {
            return;
        }
        m.dims = [0; 3];
        if let Some(q) = m.query.as_mut() {
            q.remove();
        }
        self.write_uniforms(queue);
    }

    fn begin_layer(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, k: usize) {
        let layer = &self.layers[k];
        let need_triangles = layer.need[0];
        // (triangles that didn't fit listed nothing: their references in proportion)
        let fitted = need_triangles.min(layer.triangle_capacity).max(1);
        let need_refs = (layer.need[1] as u64 * need_triangles.max(1) as u64 / fitted as u64).min(u32::MAX as u64) as u32;
        if need_triangles > layer.triangle_capacity || need_refs > layer.reference_capacity {
            let mut triangles = [self.layers[STATIC].triangle_capacity, self.layers[MOVING].triangle_capacity];
            let mut refs = [self.layers[STATIC].reference_capacity, self.layers[MOVING].reference_capacity];
            if need_triangles > triangles[k] {
                triangles[k] = grow(need_triangles);
            }
            if need_refs > refs[k] {
                refs[k] = grow(need_refs);
            }
            self.resize(device, Some(&mut *encoder), triangles, refs);
        }
        self.ensure_bind_groups(device, k);
        self.write_uniforms(queue);
        let taken = self.readback.taken;
        let layer = &mut self.layers[k];
        if layer.query.is_some() {
            // the BLAS from what the last readback gathered, with room to grow (grown past it, or
            // shrunk under half); the whole capacity before any readback. A BLAS's cost grows with
            // its triangles, and Metal's builds lost the real ones among many filler triangles.
            let need = layer.need[0];
            layer.accel_slots = if taken == 0 {
                layer.triangle_capacity
            } else if need > layer.accel_slots || need.saturating_mul(2) < layer.accel_slots {
                grow(need)
            } else {
                layer.accel_slots
            };
        }
        let (base, _, big_base, _) = self.cells_layout_of(k);
        encoder.clear_buffer(&self.layers[k].counters, 0, Some(16));
        if self.layers[k].query.is_none() {
            // the cells' counts, the macro cells and the big list's count
            encoder.clear_buffer(&self.cells, base as u64 * 4, Some((big_base - base + 1) as u64 * 4));
        }
        self.prepare_query(device, queue, k);
        self.building = Some((k, false));
    }

    /// Layer `k` as the WGSL `KanseiRtLayer`.
    fn layer_gpu(&self, k: usize) -> RtLayerGpu {
        let (cells_base, macro_base, big_base, refs_base) = self.cells_layout_of(k);
        let o = &self.options;
        let l = &self.layers[k];
        let dims = l.dims;
        RtLayerGpu {
            origin: l.origin.to_array(),
            cell: l.cell,
            dims,
            flags: o.macro_skip as u32,
            macro_dims: dims.map(|d| d.div_ceil(MACRO)),
            epsilon: o.effective_epsilon(),
            cell_count: l.cell_count(),
            macro_base,
            big_base,
            refs_base,
            ref_capacity: l.reference_capacity,
            triangle_capacity: l.triangle_capacity,
            big_capacity: o.big_triangle_capacity,
            big_cells: o.big_triangle_cells,
            cells_base,
            triangle_base: if k == STATIC { 0 } else { self.layers[STATIC].triangle_capacity },
            pad: [0; 2],
        }
    }

    /// Write the layers' uniforms, if they changed: the grid's (the static layer, then the moving
    /// one) and the moving layer's build's (the moving layer, then the static one).
    fn write_uniforms(&mut self, queue: &wgpu::Queue) {
        let (s, m) = (self.layer_gpu(STATIC), self.layer_gpu(MOVING));
        // (the moving layer's build keeps what the static layer's box would: its box rounds out to
        // whole macro cells)
        let clipped = RtLayerGpu { flags: m.flags | CLIP, ..m };
        for (k, gpu) in [(STATIC, RtGridGpu { layer: s, moving: m }), (MOVING, RtGridGpu { layer: clipped, moving: s })] {
            let layer = &mut self.layers[k];
            if layer.written != Some(gpu) {
                queue.write_buffer(&layer.uniform, 0, bytemuck::bytes_of(&gpu));
                layer.written = Some(gpu);
            }
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

    /// Append the triangles of `sources` that meet the layer's box (once a build, between `begin`
    /// and `finish`: their parameters go up in one write, which lands before the frame's work).
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
        let Some((k, false)) = self.building else { panic!("RtGrid::gather once a build, between begin and finish") };
        self.building = Some((k, true));
        if sources.is_empty() {
            return;
        }
        // alpha-tested or glass triangles: the hardware BLAS's second, non-opaque geometry
        self.layers[k].split = sources.iter().any(|s| s.surface.alpha_layer.is_some() || s.surface.glass);
        self.prepare_query(device, queue, k);
        let needed = sources.len() as u64 * SOURCE_STRIDE;
        let layer = &mut self.layers[k];
        if layer.sources.size() < needed {
            layer.sources = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/Sources"),
                size: needed.next_power_of_two(),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            layer.source_groups.clear();
        }
        if layer.gather_dispatch.size() < sources.len() as u64 * 16 {
            layer.gather_dispatch = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtGrid/GatherDispatch"),
                size: (sources.len() as u64 * 16).next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::INDIRECT,
                mapped_at_creation: false,
            });
            layer.gather_dispatch_group = None;
        }
        let mut bytes = vec![0u8; needed as usize];
        for (slot, s) in sources.iter().enumerate() {
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
                slot: slot as u32,
                kind,
                albedo_record_word: s.surface.record_albedo.map_or(0, |w| w + 1),
                emission_record_word: s.surface.record_emission.map_or(0, |w| w + 1),
            };
            bytes[slot * SOURCE_STRIDE as usize..][..std::mem::size_of::<RtSourceGpu>()].copy_from_slice(bytemuck::bytes_of(&gpu));
        }
        queue.write_buffer(&self.layers[k].sources, 0, &bytes);
        let pipelines: Vec<wgpu::ComputePipeline> = sources.iter().map(|s| self.gather_pipeline(device, &s.placement, matches!(s.mesh, GatherMesh::Clusters { .. }))).collect();
        let keys: Vec<SourceKey> = sources.iter().map(source_key).collect();
        self.ensure_bind_groups(device, k);
        let (bgl, empty, layer) = (&self.gather_source_bgl, &self.empty, &mut self.layers[k]);
        // (only this build's: a bind group holds its mesh's buffers, a scene unloaded since among them)
        layer.source_groups.retain(|key, _| keys.contains(key));
        for (s, key) in sources.iter().zip(&keys) {
            let uniform = &layer.sources;
            layer.source_groups.entry(key.clone()).or_insert_with(|| {
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
        if layer.gather_dispatch_group.is_none() {
            layer.gather_dispatch_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtGrid/GatherPrepare"),
                layout: &self.gather_prepare_bgl,
                entries: &[entry(5, &layer.gather_dispatch)],
            }));
        }
        let gather = &layer.bind_groups.as_ref().unwrap().2;
        let mut pass = timed_pass(encoder, if k == STATIC { "Rt/Gather" } else { "Rt/GatherMoving" });
        pass.set_pipeline(&self.gather_prepare);
        pass.set_bind_group(0, layer.gather_dispatch_group.as_ref().unwrap(), &[]);
        for (slot, key) in keys.iter().enumerate() {
            pass.set_bind_group(1, &layer.source_groups[key], &[(slot as u64 * SOURCE_STRIDE) as u32]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        pass.set_bind_group(0, gather, &[]);
        for (slot, (key, pipeline)) in keys.iter().zip(&pipelines).enumerate() {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(1, &layer.source_groups[key], &[(slot as u64 * SOURCE_STRIDE) as u32]);
            pass.dispatch_workgroups_indirect(&layer.gather_dispatch, slot as u64 * 16);
        }
    }

    /// Build the layer's cells from the gathered triangles: count (and the big list), scan, fill.
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let Some((k, _)) = self.building.take() else { panic!("RtGrid::finish without begin") };
        let layer = &mut self.layers[k];
        layer.builds += 1;
        if let Some(q) = layer.query.as_mut() {
            // the BLAS; the TLAS after the submit (`read_back`)
            q.record_blas(encoder);
            return;
        }
        let labels = if k == STATIC { ["Rt/GridCount", "Rt/GridScan", "Rt/GridFill"] } else { ["Rt/MovingCount", "Rt/MovingScan", "Rt/MovingFill"] };
        let (build, prepare, _) = layer.bind_groups.as_ref().expect("begin first");
        {
            let mut p = timed_pass(encoder, labels[0]);
            p.set_pipeline(&self.prepare);
            p.set_bind_group(0, prepare, &[]);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.count);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&layer.dispatch, 0);
            p.set_pipeline(&self.prepare_wide);
            p.set_bind_group(0, prepare, &[]);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.count_wide);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&layer.dispatch, 16);
        }
        {
            let cells = layer.cell_count();
            let mut p = timed_pass(encoder, labels[1]);
            p.set_bind_group(0, build, &[]);
            p.set_pipeline(&self.scan_blocks);
            p.dispatch_workgroups(cells.div_ceil(1024), 1, 1);
            p.set_pipeline(&self.scan_sums);
            p.dispatch_workgroups(1, 1, 1);
            p.set_pipeline(&self.scan_add);
            p.dispatch_workgroups(cells.div_ceil(256), 1, 1);
        }
        {
            let mut p = timed_pass(encoder, labels[2]);
            p.set_pipeline(&self.fill);
            p.set_bind_group(0, build, &[]);
            p.dispatch_workgroups_indirect(&layer.dispatch, 0);
            p.set_pipeline(&self.fill_wide);
            p.dispatch_workgroups_indirect(&layer.dispatch, 16);
        }
    }

    /// After the submit of a build: read back what each layer needed (the gathered triangles, the
    /// references, the big triangles), unless a readback is still in flight (`begin` takes it).
    /// With the hardware backend, the TLAS is then to be built over the new BLASes before the
    /// next trace (`prepare_trace`).
    pub fn read_back(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        for layer in &mut self.layers {
            if let Some(q) = layer.query.as_mut() {
                q.submitted(device, queue);
            }
        }
        if self.readback.pending.is_some() {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("RtGrid/Readback") });
        for k in [STATIC, MOVING] {
            let at = k as u64 * 12;
            if self.tlas_state.is_some() {
                // no cells: the counters' wide triangles (none) for the big ones
                encoder.copy_buffer_to_buffer(&self.layers[k].counters, 0, &self.readback.staging, at, 12);
            } else {
                let (_, _, big_base, _) = self.cells_layout_of(k);
                encoder.copy_buffer_to_buffer(&self.layers[k].counters, 0, &self.readback.staging, at, 8);
                encoder.copy_buffer_to_buffer(&self.cells, big_base as u64 * 4, &self.readback.staging, at + 8, 4);
            }
        }
        queue.submit(Some(encoder.finish()));
        self.readback.accel_triangles = self.layers.each_ref().map(|l| l.accel_slots);
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
            let _ = device.poll(wgpu::PollType::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let Some(state) = self.readback.pending.take_if(|s| s.load(Ordering::Acquire) != MAPPING) else { return };
        if state.load(Ordering::Acquire) == FAILED {
            return;
        }
        {
            let bytes = self.readback.staging.slice(..).get_mapped_range().unwrap();
            let words: &[u32] = bytemuck::cast_slice(&bytes);
            for (k, layer) in self.layers.iter_mut().enumerate() {
                layer.need = [words[k * 3], words[k * 3 + 1], words[k * 3 + 2]];
                layer.accel_read = self.readback.accel_triangles[k];
            }
        }
        self.readback.taken += 1;
        self.readback.staging.unmap();
    }

    /// Wait for the readback in flight and take it (native; for tests and tools).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_readback(&mut self, device: &wgpu::Device) {
        if self.readback.pending.is_some() {
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
        }
        self.collect_readback(device);
    }

    /// WGSL declaring the software grid's buffers in group `group` from binding `first` (three
    /// bindings), for `RT_GRID_WGSL` (`RtBackend::bindings_wgsl` for either backend's).
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

    /// The layout entries for `bindings_wgsl(_, first)` (`RtBackend::layout_entries` for either
    /// backend's).
    pub fn layout_entries(first: u32, visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 3] {
        let buffer = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility, ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None }, count: None };
        let ro = wgpu::BufferBindingType::Storage { read_only: true };
        [buffer(first, wgpu::BufferBindingType::Uniform), buffer(first + 1, ro), buffer(first + 2, ro)]
    }

    /// Ready the grid for tracing after a rebuild's submit and `read_back`, before the passes that
    /// trace it: the hardware backend builds its TLAS over the new BLASes (on Metal once the BLAS's
    /// submit has finished, which the CPU waits for: gfx-rs/wgpu#9215). Nothing in software.
    pub fn prepare_trace(&self, device: &wgpu::Device, queue: &wgpu::Queue) {
        self.handle.prepare_trace(device, queue);
    }

    /// The resources for `self.backend().bindings_wgsl(_, first)`, in order.
    pub fn bind_group_entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        let third = match &self.tlas {
            Some(tlas) => wgpu::BindingResource::AccelerationStructure(tlas),
            None => self.cells.as_entire_binding(),
        };
        [
            wgpu::BindGroupEntry { binding: first, resource: self.layers[STATIC].uniform.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 1, resource: self.triangles.as_entire_binding() },
            wgpu::BindGroupEntry { binding: first + 2, resource: third },
        ]
    }

    /// The world triangles (64 bytes each, rt_types.wgsl: the static layer's from 0, the moving
    /// layer's from `RtGridStats::moving_triangle_base`), for tests and debugging.
    pub fn triangles_buffer(&self) -> &wgpu::Buffer {
        &self.triangles
    }

    /// The cell words (rt_types.wgsl), for tests and debugging.
    pub fn cells_buffer(&self) -> &wgpu::Buffer {
        &self.cells
    }

    /// Where the static layer's macro cells, big list and references start in the cell words.
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
