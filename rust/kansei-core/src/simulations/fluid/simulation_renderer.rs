use super::{MarchingCubesVertex, SimulationRenderableInputContract, SurfaceContractVersion};
use crate::math::Mat4;
use crate::renderers::Renderer;

pub struct SimulationRenderable {
    pipeline: wgpu::RenderPipeline,
    params_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

impl SimulationRenderable {
    fn new_with_device(
        device: &wgpu::Device,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SimulationRenderer/SurfaceMeshShader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct Params { view: mat4x4<f32>, proj: mat4x4<f32>, color: vec4<f32>, }
@group(0) @binding(0) var<uniform> p: Params;

struct VSIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
}
struct VSOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) normal: vec3<f32>,
}

@vertex
fn vs(v: VSIn) -> VSOut {
    var o: VSOut;
    o.pos = p.proj * p.view * vec4<f32>(v.position.xyz, 1.0);
    o.normal = normalize(v.normal.xyz);
    return o;
}

@fragment
fn fs(v: VSOut) -> @location(0) vec4<f32> {
    let light = normalize(vec3<f32>(0.25, 1.0, 0.4));
    let ndotl = max(dot(v.normal, light), 0.0);
    let base = p.color.rgb;
    let lit = base * (0.2 + ndotl * 0.8);
    return vec4<f32>(lit, p.color.a);
}
"#
                .into(),
            ),
        });
        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SimulationRenderer/Params"),
            size: 144,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SimulationRenderer/BGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SimulationRenderer/Layout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("SimulationRenderer/Pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<MarchingCubesVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 0,
                            shader_location: 0,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: 16,
                            shader_location: 1,
                        },
                    ],
                }],
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
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth24Plus,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SimulationRenderer/BG"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: params_buffer.as_entire_binding(),
            }],
        });
        Self {
            pipeline,
            params_buffer,
            bind_group,
        }
    }

    pub fn new(renderer: &Renderer, color_format: wgpu::TextureFormat, sample_count: u32) -> Self {
        Self::new_with_device(renderer.raw_device(), color_format, sample_count)
    }

    fn render_surface_mesh_with_queue(
        &self,
        queue: &wgpu::Queue,
        pass: &mut wgpu::RenderPass<'_>,
        input: SimulationRenderableInputContract<'_>,
        view: &Mat4,
        proj: &Mat4,
        color: [f32; 4],
    ) {
        if input.version != SurfaceContractVersion::V1 {
            return;
        }
        let mut d = [0.0f32; 36];
        d[..16].copy_from_slice(view.as_slice());
        d[16..32].copy_from_slice(proj.as_slice());
        d[32] = color[0];
        d[33] = color[1];
        d[34] = color[2];
        d[35] = color[3];
        queue.write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&d));

        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_vertex_buffer(0, input.mesh.vertex_buffer.slice(..));
        pass.set_index_buffer(input.mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed_indirect(input.mesh.indirect_args_buffer, 0);
    }

    pub fn render_surface_mesh(
        &self,
        renderer: &Renderer,
        pass: &mut wgpu::RenderPass<'_>,
        input: SimulationRenderableInputContract<'_>,
        view: &Mat4,
        proj: &Mat4,
        color: [f32; 4],
    ) {
        self.render_surface_mesh_with_queue(renderer.raw_queue(), pass, input, view, proj, color);
    }

}

pub type SimulationRenderer = SimulationRenderable;
