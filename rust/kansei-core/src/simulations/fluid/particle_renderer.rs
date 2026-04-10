use crate::math::Mat4;
use crate::renderers::Renderer;

const PARTICLE_WGSL: &str = r#"
struct P { view: mat4x4<f32>, proj: mat4x4<f32>, size: f32, }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> p: P;
struct V { @builtin(position) pos: vec4<f32>, @location(0) col: vec3<f32>, }
const Q: array<vec2<f32>,6> = array(vec2(-1.,-1.),vec2(1.,-1.),vec2(1.,1.),vec2(-1.,-1.),vec2(1.,1.),vec2(-1.,1.));
@vertex fn vs(@builtin(vertex_index) vi: u32) -> V {
    let pid=vi/6u; let c=Q[vi%6u]; let pos=positions[pid]; let s=p.size;
    let r=vec3<f32>(p.view[0][0],p.view[1][0],p.view[2][0]);
    let u=vec3<f32>(p.view[0][1],p.view[1][1],p.view[2][1]);
    let wp=pos.xyz+r*c.x*s+u*c.y*s;
    var o:V; o.pos=p.proj*p.view*vec4<f32>(wp,1.);
    let t=clamp((pos.y+8.)/16.,0.,1.);
    o.col=mix(vec3<f32>(0.1,0.3,0.8),vec3<f32>(0.8,0.95,1.0),t); return o;
}
@fragment fn fs(v:V)->@location(0) vec4<f32>{return vec4<f32>(v.col,1.);}
"#;

pub struct FluidParticleRenderer {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group: wgpu::BindGroup,
    pub params_buf: wgpu::Buffer,
    pub count: u32,
}

impl FluidParticleRenderer {
    fn new_with_device(
        device: &wgpu::Device,
        positions_buffer: &wgpu::Buffer,
        count: u32,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("FluidParticles"),
            source: wgpu::ShaderSource::Wgsl(PARTICLE_WGSL.into()),
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FluidParticles/Params"),
            size: 144,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("FluidParticles/Pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs"),
                buffers: &[],
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
            primitive: Default::default(),
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: sample_count.max(1), ..Default::default() },
            multiview: None,
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidParticles/BG"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });
        Self { pipeline, bind_group, params_buf, count }
    }

    pub fn new(
        renderer: &Renderer,
        positions_buffer: &wgpu::Buffer,
        count: u32,
        color_format: wgpu::TextureFormat,
        sample_count: u32,
        depth_format: Option<wgpu::TextureFormat>,
    ) -> Self {
        Self::new_with_device(
            renderer.raw_device(),
            positions_buffer,
            count,
            color_format,
            sample_count,
            depth_format,
        )
    }

    fn upload_with_queue(&self, queue: &wgpu::Queue, view: &Mat4, proj: &Mat4, size: f32) {
        let mut d = [0.0f32; 36];
        d[..16].copy_from_slice(view.as_slice());
        d[16..32].copy_from_slice(proj.as_slice());
        d[32] = size;
        queue.write_buffer(&self.params_buf, 0, bytemuck::cast_slice(&d));
    }

    pub fn upload(&self, renderer: &Renderer, view: &Mat4, proj: &Mat4, size: f32) {
        self.upload_with_queue(renderer.raw_queue(), view, proj, size);
    }

}
