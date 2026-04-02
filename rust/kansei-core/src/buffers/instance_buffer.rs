/// A single vertex attribute within an instance buffer.
#[derive(Debug, Clone)]
pub struct InstanceAttribute {
    pub shader_location: u32,
    pub offset: u64,
    pub format: wgpu::VertexFormat,
}

/// Per-instance GPU buffer with vertex attribute descriptors.
/// Used for instanced rendering — data steps once per instance, not per vertex.
pub struct InstanceBuffer {
    pub label: String,
    pub data: Vec<u8>,
    pub stride: u64,
    pub attributes: Vec<InstanceAttribute>,
    pub gpu_buffer: Option<wgpu::Buffer>,
    pub initialized: bool,
}

impl InstanceBuffer {
    /// Create an instance buffer from raw byte data.
    pub fn new(label: &str, data: &[u8], stride: u64, attributes: Vec<InstanceAttribute>) -> Self {
        Self {
            label: label.to_string(),
            data: data.to_vec(),
            stride,
            attributes,
            gpu_buffer: None,
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
                format: wgpu::VertexFormat::Float32x4,
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
                InstanceAttribute { shader_location: base_shader_location, offset: 0, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 1, offset: 16, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 2, offset: 32, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 3, offset: 48, format: wgpu::VertexFormat::Float32x4 },
            ],
        )
    }

    /// Initialize the GPU buffer.
    pub fn initialize(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;
        self.gpu_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}/Buffer", self.label)),
            contents: &self.data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        self.initialized = true;
    }

    /// Update GPU buffer contents.
    pub fn update(&self, queue: &wgpu::Queue, data: &[u8]) {
        if let Some(ref buf) = self.gpu_buffer {
            queue.write_buffer(buf, 0, data);
        }
    }

    /// Build owned vertex layout data for this instance buffer.
    pub fn vertex_layout(&self) -> InstanceBufferLayout {
        let attrs: Vec<wgpu::VertexAttribute> = self.attributes.iter().map(|a| {
            wgpu::VertexAttribute {
                format: a.format,
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
