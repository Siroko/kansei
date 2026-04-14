/// Vertex attribute format — engine abstraction over wgpu::VertexFormat.
/// Callers use this instead of wgpu types directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VertexFormat {
    Float32,
    Float32x2,
    Float32x3,
    Float32x4,
    Uint32,
    Uint32x2,
    Uint32x3,
    Uint32x4,
    Sint32,
    Sint32x2,
    Sint32x3,
    Sint32x4,
}

impl VertexFormat {
    pub(crate) fn to_wgpu(self) -> wgpu::VertexFormat {
        match self {
            Self::Float32   => wgpu::VertexFormat::Float32,
            Self::Float32x2 => wgpu::VertexFormat::Float32x2,
            Self::Float32x3 => wgpu::VertexFormat::Float32x3,
            Self::Float32x4 => wgpu::VertexFormat::Float32x4,
            Self::Uint32    => wgpu::VertexFormat::Uint32,
            Self::Uint32x2  => wgpu::VertexFormat::Uint32x2,
            Self::Uint32x3  => wgpu::VertexFormat::Uint32x3,
            Self::Uint32x4  => wgpu::VertexFormat::Uint32x4,
            Self::Sint32    => wgpu::VertexFormat::Sint32,
            Self::Sint32x2  => wgpu::VertexFormat::Sint32x2,
            Self::Sint32x3  => wgpu::VertexFormat::Sint32x3,
            Self::Sint32x4  => wgpu::VertexFormat::Sint32x4,
        }
    }
}

/// A single vertex attribute within an instance buffer.
#[derive(Debug, Clone)]
pub struct InstanceAttribute {
    pub shader_location: u32,
    pub offset: u64,
    pub format: VertexFormat,
}

/// Per-instance GPU buffer with vertex attribute descriptors.
/// Used for instanced rendering — data steps once per instance, not per vertex.
#[deprecated(note = "Use ComputeBuffer with .with_vertex_layout() instead")]
pub struct InstanceBuffer {
    pub label: String,
    pub data: Vec<u8>,
    pub stride: u64,
    pub attributes: Vec<InstanceAttribute>,
    pub gpu_buffer: Option<wgpu::Buffer>,
    queue: Option<wgpu::Queue>,
    pub initialized: bool,
}

#[allow(deprecated)]
impl InstanceBuffer {
    /// Create an instance buffer from raw byte data.
    pub fn new(label: &str, data: &[u8], stride: u64, attributes: Vec<InstanceAttribute>) -> Self {
        Self {
            label: label.to_string(),
            data: data.to_vec(),
            stride,
            attributes,
            gpu_buffer: None,
            queue: None,
            initialized: false,
        }
    }

    /// Convenience: create from f32 data with a single vec4 attribute.
    pub fn from_f32_vec4(label: &str, data: &[f32], shader_location: u32) -> Self {
        Self::new(
            label,
            bytemuck::cast_slice(data),
            16,
            vec![InstanceAttribute {
                shader_location,
                offset: 0,
                format: VertexFormat::Float32x4,
            }],
        )
    }

    /// Convenience: create from f32 data with mat4 as 4 consecutive vec4 attributes.
    pub fn from_mat4(label: &str, data: &[f32], base_shader_location: u32) -> Self {
        Self::new(
            label,
            bytemuck::cast_slice(data),
            64,
            vec![
                InstanceAttribute { shader_location: base_shader_location, offset: 0, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 1, offset: 16, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 2, offset: 32, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 3, offset: 48, format: VertexFormat::Float32x4 },
            ],
        )
    }

    /// Initialize the GPU buffer, storing the queue for later self-contained updates.
    pub fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        use wgpu::util::DeviceExt;
        self.gpu_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}/Buffer", self.label)),
            contents: &self.data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        self.queue = Some(queue.clone());
        self.initialized = true;
    }

    /// Update GPU buffer contents using the stored queue.
    pub fn update(&self, data: &[u8]) {
        if let (Some(ref buf), Some(ref queue)) = (&self.gpu_buffer, &self.queue) {
            queue.write_buffer(buf, 0, data);
        }
    }

    /// Build owned vertex layout data for this instance buffer.
    pub fn vertex_layout(&self) -> InstanceBufferLayout {
        let attrs: Vec<wgpu::VertexAttribute> = self.attributes.iter().map(|a| {
            wgpu::VertexAttribute {
                format: a.format.to_wgpu(),
                offset: a.offset,
                shader_location: a.shader_location,
            }
        }).collect();
        InstanceBufferLayout { stride: self.stride, attributes: attrs }
    }
}

/// Owned vertex buffer layout data for an instance buffer.
pub struct InstanceBufferLayout {
    pub stride: u64,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl InstanceBufferLayout {
    /// Borrow as a wgpu::VertexBufferLayout with step_mode: Instance.
    pub fn as_layout(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.stride,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &self.attributes,
        }
    }
}
