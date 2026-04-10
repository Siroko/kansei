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
pub struct GpuBuffer {
    label: String,
    buffer_type: BufferType,
    usage: BufferUsage,
    data: Vec<u8>,
    gpu_buffer: Option<wgpu::Buffer>,
    needs_update: bool,
}

impl GpuBuffer {
    pub fn new(label: &str, buffer_type: BufferType, usage: BufferUsage, data: Vec<u8>) -> Self {
        Self {
            label: label.to_string(),
            buffer_type,
            usage,
            data,
            gpu_buffer: None,
            needs_update: false,
        }
    }

    pub fn from_slice<T: bytemuck::Pod>(label: &str, buffer_type: BufferType, usage: BufferUsage, data: &[T]) -> Self {
        Self::new(label, buffer_type, usage, bytemuck::cast_slice(data).to_vec())
    }

    pub fn buffer_type(&self) -> BufferType {
        self.buffer_type
    }

    pub fn gpu_buffer(&self) -> Option<&wgpu::Buffer> {
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
    pub fn initialize(&mut self, device: &wgpu::Device) {
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
    pub fn update(&mut self, queue: &wgpu::Queue) {
        if self.needs_update {
            if let Some(ref buf) = self.gpu_buffer {
                queue.write_buffer(buf, 0, &self.data);
            }
            self.needs_update = false;
        }
    }

    /// Ensure initialized, then upload if dirty.
    pub fn ensure_ready(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
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
