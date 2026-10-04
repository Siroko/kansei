use bytemuck::{Pod, Zeroable};
use crate::buffers::ComputeBuffer;

/// A single vertex attribute descriptor.
#[derive(Debug, Clone)]
pub struct VertexAttribute {
    pub shader_location: u32,
    pub offset: u64,
    pub format: wgpu::VertexFormat,
}

/// Standard interleaved vertex: position(vec4) + normal(vec3) + uv(vec2).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Vertex {
    pub position: [f32; 4],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
}

impl Vertex {
    pub const LAYOUT: wgpu::VertexBufferLayout<'static> = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            // @location(0) position: vec4<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x4,
                offset: 0,
                shader_location: 0,
            },
            // @location(1) normal: vec3<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x3,
                offset: 16,
                shader_location: 1,
            },
            // @location(2) uv: vec2<f32>
            wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Float32x2,
                offset: 28,
                shader_location: 2,
            },
        ],
    };
}

/// A GPU geometry — vertex + index buffers, with optional instancing data.
///
/// Plain geometries have `instance_count == 1` and empty `instance_buffers`.
/// `InstancedGeometry::new()` produces a `Geometry` with those fields populated.
pub struct Geometry {
    pub label: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub vertex_buffer: Option<wgpu::Buffer>,
    pub index_buffer: Option<wgpu::Buffer>,
    /// Optional indirect draw args buffer (DrawIndexedIndirect: 5 × u32).
    pub indirect_args_buffer: Option<wgpu::Buffer>,
    pub initialized: bool,
    // External buffer pointers (for compute-generated geometry, e.g. marching cubes)
    ext_vertex_buffer: Option<*const wgpu::Buffer>,
    ext_index_buffer: Option<*const wgpu::Buffer>,
    ext_indirect_buffer: Option<*const wgpu::Buffer>,
    /// Number of instances to draw (1 = non-instanced).
    pub instance_count: u32,
    /// Per-instance vertex buffers (empty for non-instanced geometry).
    pub instance_buffers: Vec<ComputeBuffer>,
}

// SAFETY: wgpu::Buffer is internally Arc-based and Send+Sync.
// The raw pointers are only dereferenced on the same thread that created them.
unsafe impl Send for Geometry {}
unsafe impl Sync for Geometry {}

impl Geometry {
    /// Get the active vertex buffer (owned or external).
    pub fn active_vertex_buffer(&self) -> Option<&wgpu::Buffer> {
        self.vertex_buffer.as_ref().or_else(|| {
            self.ext_vertex_buffer.map(|p| unsafe { &*p })
        })
    }
    /// Get the active index buffer (owned or external).
    pub fn active_index_buffer(&self) -> Option<&wgpu::Buffer> {
        self.index_buffer.as_ref().or_else(|| {
            self.ext_index_buffer.map(|p| unsafe { &*p })
        })
    }
    /// Get the active indirect args buffer (owned or external).
    pub fn active_indirect_buffer(&self) -> Option<&wgpu::Buffer> {
        self.indirect_args_buffer.as_ref().or_else(|| {
            self.ext_indirect_buffer.map(|p| unsafe { &*p })
        })
    }
}

impl Geometry {
    /// The same geometry named `label` (in profiles and GPU captures): one of the stock
    /// generators' meshes, say, as `SpruceGeometry::new(8, 1, 3).with_label("Spruce/LOD2")`.
    pub fn with_label(mut self, label: &str) -> Self {
        self.label = label.to_string();
        self
    }

    pub fn new(label: &str, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            label: label.to_string(),
            vertices,
            indices,
            vertex_buffer: None,
            index_buffer: None,
            indirect_args_buffer: None,
            initialized: false,
            ext_vertex_buffer: None,
            ext_index_buffer: None,
            ext_indirect_buffer: None,
            instance_count: 1,
            instance_buffers: Vec::new(),
        }
    }

    /// A geometry whose vertices a compute pass writes into `vertex_buffer` (laid out as
    /// [`Vertex`], with VERTEX usage), drawn with `indices`. Nothing is uploaded for the vertices:
    /// the geometry keeps a handle to the buffer (its CPU `vertices` stay empty).
    pub fn from_gpu_vertices(label: &str, vertex_buffer: wgpu::Buffer, indices: Vec<u32>) -> Self {
        let mut geometry = Self::new(label, Vec::new(), indices);
        geometry.vertex_buffer = Some(vertex_buffer);
        geometry
    }

    /// A geometry drawn indirectly from buffers a compute pass fills (a marching-cubes mesh):
    /// vertices laid out as [`Vertex`], u32 indices and, if given, `DrawIndexedIndirect`
    /// arguments. The geometry keeps handles to the buffers, so they must not be replaced by
    /// new ones afterwards (writing into them is fine).
    pub fn from_gpu_buffers(label: &str, vertex_buffer: wgpu::Buffer, index_buffer: wgpu::Buffer, indirect_args_buffer: Option<wgpu::Buffer>) -> Self {
        let mut geometry = Self::new_indirect_placeholder(label);
        geometry.vertex_buffer = Some(vertex_buffer);
        geometry.index_buffer = Some(index_buffer);
        geometry.indirect_args_buffer = indirect_args_buffer;
        geometry
    }

    /// Several geometries as one, each moved by its matrix first (normals by its inverse
    /// transpose): a spruce from cones, a model from its glTF parts, props from boxes and
    /// cylinders. Instancing is not carried over.
    pub fn merged(label: &str, parts: &[(&Geometry, glam::Mat4)]) -> Self {
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        for (geometry, matrix) in parts {
            let normal_matrix = matrix.inverse().transpose();
            let base = vertices.len() as u32;
            vertices.extend(geometry.vertices.iter().map(|v| {
                let p = matrix.transform_point3(glam::Vec3::new(v.position[0], v.position[1], v.position[2]));
                let n = normal_matrix.transform_vector3(glam::Vec3::from(v.normal)).normalize_or_zero();
                Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: v.uv }
            }));
            indices.extend(geometry.indices.iter().map(|i| base + i));
        }
        Self::new(label, vertices, indices)
    }

    /// The axis-aligned bounds of the CPU vertices, (min, max); zero for none.
    pub fn bounds(&self) -> (glam::Vec3, glam::Vec3) {
        let mut points = self.vertices.iter().map(|v| glam::Vec3::new(v.position[0], v.position[1], v.position[2]));
        let Some(first) = points.next() else { return (glam::Vec3::ZERO, glam::Vec3::ZERO) };
        points.fold((first, first), |(lo, hi), p| (lo.min(p), hi.max(p)))
    }

    /// Scale the vertices uniformly to fit inside a box of `size` (use `f32::INFINITY` for an
    /// axis that may be any size) and move them so the bottom centre of their bounds sits at the
    /// origin: a model ready to stand on the ground.
    pub fn fit(mut self, size: glam::Vec3) -> Self {
        let (lo, hi) = self.bounds();
        let scale = (size / (hi - lo).max(glam::Vec3::splat(1e-9))).min_element();
        let anchor = glam::Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
        for v in &mut self.vertices {
            let p = (glam::Vec3::new(v.position[0], v.position[1], v.position[2]) - anchor) * scale;
            v.position = [p.x, p.y, p.z, 1.0];
        }
        self
    }

    /// Create a Geometry placeholder for externally-owned GPU buffers (zero readback).
    /// Initially has no buffer pointers — call `set_external_buffers()` once the
    /// owning struct is at its final heap address.
    pub fn new_indirect_placeholder(label: &str) -> Self {
        Self {
            label: label.to_string(),
            vertices: Vec::new(),
            indices: Vec::new(),
            vertex_buffer: None,
            index_buffer: None,
            indirect_args_buffer: None,
            initialized: true,
            ext_vertex_buffer: None,
            ext_index_buffer: None,
            ext_indirect_buffer: None,
            instance_count: 1,
            instance_buffers: Vec::new(),
        }
    }

    pub fn initialize(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;

        // a buffer handed in (`from_gpu_vertices`) is kept
        if self.vertex_buffer.is_none() {
            self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{}/Vertices", self.label)),
                contents: bytemuck::cast_slice(&self.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            }));
        }

        self.index_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}/Indices", self.label)),
            contents: bytemuck::cast_slice(&self.indices),
            usage: wgpu::BufferUsages::INDEX,
        }));

        self.initialized = true;
    }

    pub fn index_count(&self) -> u32 {
        self.indices.len() as u32
    }

    pub fn vertex_count(&self) -> u32 {
        self.vertices.len() as u32
    }

    /// Whether this geometry uses instanced rendering.
    pub fn is_instanced(&self) -> bool {
        !self.instance_buffers.is_empty()
    }

    /// Update external buffer pointers. Call after the owning struct is at its final
    /// heap address (e.g., after moving into Rc<RefCell<>>).
    ///
    /// # Safety
    /// Pointers must remain valid for the lifetime of this Geometry.
    #[deprecated(note = "use Geometry::from_gpu_buffers, which keeps handles to the buffers")]
    pub unsafe fn set_external_buffers(
        &mut self,
        vertex: *const wgpu::Buffer,
        index: *const wgpu::Buffer,
        indirect: Option<*const wgpu::Buffer>,
    ) {
        self.ext_vertex_buffer = Some(vertex);
        self.ext_index_buffer = Some(index);
        self.ext_indirect_buffer = indirect;
    }

    /// Returns true if this geometry uses GPU-driven indirect drawing.
    pub fn is_indirect(&self) -> bool {
        self.active_indirect_buffer().is_some()
    }
}
