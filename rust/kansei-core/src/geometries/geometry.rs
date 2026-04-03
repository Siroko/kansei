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
    pub initialized: bool,
}

impl Geometry {
    pub fn new(label: &str, vertices: Vec<Vertex>, indices: Vec<u32>) -> Self {
        Self {
            label: label.to_string(),
            vertices,
            indices,
            vertex_buffer: None,
            index_buffer: None,
            initialized: false,
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
}
