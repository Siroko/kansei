use bytemuck::{Pod, Zeroable};

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

/// A GPU geometry — vertex + index buffers.
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
        }
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
        }
    }

    pub fn initialize(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;

        self.vertex_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}/Vertices", self.label)),
            contents: bytemuck::cast_slice(&self.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        }));

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

    /// Update external buffer pointers. Call after the owning struct is at its final
    /// heap address (e.g., after moving into Rc<RefCell<>>).
    ///
    /// # Safety
    /// Pointers must remain valid for the lifetime of this Geometry.
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
