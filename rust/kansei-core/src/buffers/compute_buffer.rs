use super::instance_buffer::{InstanceAttribute, InstanceBufferLayout, VertexFormat};

/// Buffer type for bind group layout entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferType {
    Storage,
    Uniform,
    ReadOnlyStorage,
}

/// Buffer usage flags (mirrors wgpu::BufferUsages).
pub type BufferUsage = wgpu::BufferUsages;

/// A GPU buffer with CPU-side staging data and lazy initialization.
/// Can serve as both a compute storage buffer and an instance vertex source,
/// mirroring the TS `ComputeBuffer` which carries dual storage + vertex roles.
pub struct ComputeBuffer {
    label: String,
    buffer_type: BufferType,
    usage: BufferUsage,
    data: Vec<u8>,
    gpu_buffer: Option<wgpu::Buffer>,
    needs_update: bool,
    vertex_stride: Option<u64>,
    vertex_attributes: Vec<InstanceAttribute>,
}

impl ComputeBuffer {
    pub fn new(label: &str, buffer_type: BufferType, usage: BufferUsage, data: Vec<u8>) -> Self {
        Self {
            label: label.to_string(),
            buffer_type,
            usage,
            data,
            gpu_buffer: None,
            needs_update: false,
            vertex_stride: None,
            vertex_attributes: Vec::new(),
        }
    }

    pub fn from_slice<T: bytemuck::Pod>(label: &str, buffer_type: BufferType, usage: BufferUsage, data: &[T]) -> Self {
        Self::new(label, buffer_type, usage, bytemuck::cast_slice(data).to_vec())
    }

    /// Wrap a pre-existing `wgpu::Buffer` (e.g. one created by a simulation).
    /// CPU-side staging data is empty — the buffer is already on the GPU.
    /// Internal only — callers should not touch wgpu types directly.
    pub(crate) fn from_external(label: &str, buffer: wgpu::Buffer, buffer_type: BufferType) -> Self {
        Self {
            label: label.to_string(),
            buffer_type,
            usage: buffer.usage(),
            data: Vec::new(),
            gpu_buffer: Some(buffer),
            needs_update: false,
            vertex_stride: None,
            vertex_attributes: Vec::new(),
        }
    }

    /// Builder: attach vertex attribute metadata so this buffer can double as
    /// an instance-stepped vertex source (mirrors TS `ComputeBuffer`'s
    /// `shaderLocation` / `stride` / `format` / `attributes` fields).
    pub fn with_vertex_layout(mut self, stride: u64, attributes: Vec<InstanceAttribute>) -> Self {
        self.vertex_stride = Some(stride);
        self.vertex_attributes = attributes;
        self
    }

    /// Convenience: configure as a single vec4 instance attribute.
    pub fn with_vertex_vec4(self, shader_location: u32) -> Self {
        self.with_vertex_layout(16, vec![InstanceAttribute {
            shader_location,
            offset: 0,
            format: VertexFormat::Float32x4,
        }])
    }

    /// Convenience: configure as a mat4 instance attribute (4 consecutive vec4 locations).
    pub fn with_vertex_mat4(self, base_shader_location: u32) -> Self {
        self.with_vertex_layout(64, vec![
            InstanceAttribute { shader_location: base_shader_location, offset: 0, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: base_shader_location + 1, offset: 16, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: base_shader_location + 2, offset: 32, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: base_shader_location + 3, offset: 48, format: VertexFormat::Float32x4 },
        ])
    }

    /// If vertex metadata is set, return the instance-stepped layout for pipeline creation.
    pub(crate) fn vertex_layout(&self) -> Option<InstanceBufferLayout> {
        let stride = self.vertex_stride?;
        let attrs = self.vertex_attributes.iter().map(|a| wgpu::VertexAttribute {
            format: a.format.to_wgpu(),
            offset: a.offset,
            shader_location: a.shader_location,
        }).collect();
        Some(InstanceBufferLayout { stride, attributes: attrs })
    }

    /// Whether this buffer has vertex attribute metadata attached.
    pub fn has_vertex_layout(&self) -> bool {
        self.vertex_stride.is_some()
    }

    pub fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    pub(crate) fn gpu_buffer(&self) -> Option<&wgpu::Buffer> {
        self.gpu_buffer.as_ref()
    }

    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [u8] {
        self.needs_update = true;
        &mut self.data
    }

    /// Write typed data into the CPU staging buffer.
    pub fn write<T: bytemuck::Pod>(&mut self, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        self.data[..bytes.len()].copy_from_slice(bytes);
        self.needs_update = true;
    }

    /// Write typed data at a byte offset.
    pub fn write_at<T: bytemuck::Pod>(&mut self, byte_offset: usize, data: &[T]) {
        let bytes = bytemuck::cast_slice(data);
        self.data[byte_offset..byte_offset + bytes.len()].copy_from_slice(bytes);
        self.needs_update = true;
    }

    /// Initialize the GPU buffer (mappedAtCreation with initial data).
    pub(crate) fn initialize(&mut self, device: &wgpu::Device) {
        if self.gpu_buffer.is_some() {
            return;
        }
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&self.label),
            size: self.data.len() as u64,
            usage: self.usage | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: true,
        });
        buffer.slice(..).get_mapped_range_mut().copy_from_slice(&self.data);
        buffer.unmap();
        self.gpu_buffer = Some(buffer);
        self.needs_update = false;
    }

    /// Upload dirty CPU data to the GPU via queue.write_buffer.
    pub(crate) fn update(&mut self, queue: &wgpu::Queue) {
        if self.needs_update {
            if let Some(ref buf) = self.gpu_buffer {
                queue.write_buffer(buf, 0, &self.data);
            }
            self.needs_update = false;
        }
    }

    /// Ensure initialized, then upload if dirty.
    pub(crate) fn ensure_ready(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        if self.gpu_buffer.is_none() {
            self.initialize(device);
        } else {
            self.update(queue);
        }
    }

    pub fn byte_len(&self) -> usize {
        self.data.len()
    }
}
