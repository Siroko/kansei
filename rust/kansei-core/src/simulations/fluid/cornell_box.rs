use crate::geometries::PlaneGeometry;
use crate::math::{Mat4, Vec3};
use crate::renderers::Renderer;

const CORNELL_WGSL: &str = r#"
struct Params { view: mat4x4<f32>, proj: mat4x4<f32>, }
@group(0) @binding(0) var<uniform> p: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32>, @location(1) norm: vec3<f32>, }

@vertex fn vs(@location(0) position: vec3<f32>, @location(1) normal: vec3<f32>, @location(2) color: vec3<f32>) -> VOut {
    var o: VOut;
    o.pos = p.proj * p.view * vec4<f32>(position, 1.0);
    o.col = color;
    o.norm = normal;
    return o;
}

@fragment fn fs(v: VOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.3, 1.0, 0.5));
    let ndotl = max(dot(normalize(v.norm), light), 0.0);
    let ambient = 0.3;
    return vec4<f32>(v.col * (ambient + ndotl * 0.7), 1.0);
}
"#;

pub struct FluidCornellBox {
    pub pipeline: wgpu::RenderPipeline,
    pub depth_pipeline: wgpu::RenderPipeline,
    pub vertex_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub vertex_count: u32,
}

impl FluidCornellBox {
    fn new_with_device(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        depth_pass_color_format: wgpu::TextureFormat,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        sample_count: u32,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CornellBox"),
            source: wgpu::ShaderSource::Wgsl(CORNELL_WGSL.into()),
        });

        let [x0, y0, z0] = bounds_min;
        let [x1, y1, z1] = bounds_max;
        let mut vertices: Vec<f32> = Vec::new();
        let mut append_plane =
            |width: f32, height: f32, center: Vec3, u: Vec3, v: Vec3, n: Vec3, color: [f32; 3]| {
                let g = PlaneGeometry::new(width, height);
                for &idx in &g.indices {
                    let vv = g.vertices[idx as usize];
                    let lp = Vec3::new(vv.position[0], vv.position[1], vv.position[2]);
                    let ln = Vec3::new(vv.normal[0], vv.normal[1], vv.normal[2]);
                    let wp = center + u * lp.x + v * lp.y + n * lp.z;
                    let wn = (u * ln.x + v * ln.y + n * ln.z).normalize();
                    vertices.extend_from_slice(&[
                        wp.x, wp.y, wp.z, wn.x, wn.y, wn.z, color[0], color[1], color[2],
                    ]);
                }
            };

        let white = [0.7, 0.7, 0.7];
        append_plane(
            x1 - x0,
            z1 - z0,
            Vec3::new((x0 + x1) * 0.5, y0, (z0 + z1) * 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            white,
        );
        append_plane(
            x1 - x0,
            z1 - z0,
            Vec3::new((x0 + x1) * 0.5, y1, (z0 + z1) * 0.5),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, -1.0, 0.0),
            white,
        );
        append_plane(
            x1 - x0,
            y1 - y0,
            Vec3::new((x0 + x1) * 0.5, (y0 + y1) * 0.5, z0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            white,
        );
        append_plane(
            z1 - z0,
            y1 - y0,
            Vec3::new(x0, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            [0.8, 0.15, 0.1],
        );
        append_plane(
            z1 - z0,
            y1 - y0,
            Vec3::new(x1, (y0 + y1) * 0.5, (z0 + z1) * 0.5),
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            [0.15, 0.8, 0.1],
        );
        append_plane(
            x1 - x0,
            y1 - y0,
            Vec3::new((x0 + x1) * 0.5, (y0 + y1) * 0.5, z1),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, -1.0),
            white,
        );

        use wgpu::util::DeviceExt;
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CornellBox/Vertices"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CornellBox/Params"),
            size: 128,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let vertex_layout = [wgpu::VertexBufferLayout {
            array_stride: 36,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 12,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 24,
                    shader_location: 2,
                },
            ],
        }];
        let primitive = wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            cull_mode: Some(wgpu::Face::Front),
            ..Default::default()
        };

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CornellBox/BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CornellBox/Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CornellBox"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &vertex_layout,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: sample_count.max(1),
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        let depth_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CornellBox/Depth"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &vertex_layout,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: depth_pass_color_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive,
            depth_stencil: Some(wgpu::DepthStencilState {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                ..Default::default()
            },
            multiview: None,
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CornellBox/BG"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buf.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            depth_pipeline,
            vertex_buf,
            params_buf,
            bind_group,
            vertex_count: (vertices.len() / 9) as u32,
        }
    }

    pub fn new(
        renderer: &Renderer,
        color_format: wgpu::TextureFormat,
        depth_pass_color_format: wgpu::TextureFormat,
        bounds_min: [f32; 3],
        bounds_max: [f32; 3],
        sample_count: u32,
        depth_format: wgpu::TextureFormat,
    ) -> Self {
        Self::new_with_device(
            renderer.raw_device(),
            color_format,
            depth_pass_color_format,
            bounds_min,
            bounds_max,
            sample_count,
            depth_format,
        )
    }

    fn upload_with_queue(&self, queue: &wgpu::Queue, view: &Mat4, proj: &Mat4) {
        let mut data = [0.0f32; 32];
        data[..16].copy_from_slice(view.as_slice());
        data[16..32].copy_from_slice(proj.as_slice());
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&data));
    }

    pub fn upload(&self, renderer: &Renderer, view: &Mat4, proj: &Mat4) {
        self.upload_with_queue(renderer.raw_queue(), view, proj);
    }

}
