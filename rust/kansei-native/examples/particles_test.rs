use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

struct App {
    window: Option<Arc<Window>>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    pipeline: Option<wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    params_buf: Option<wgpu::Buffer>,
    count: u32,
    format: wgpu::TextureFormat,
}

impl App {
    fn new() -> Self { Self { window: None, device: None, queue: None, surface: None, pipeline: None, bind_group: None, params_buf: None, count: 0, format: wgpu::TextureFormat::Bgra8Unorm } }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let win = Arc::new(el.create_window(Window::default_attributes().with_title("Particles Test")).unwrap());
        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(win.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface), ..Default::default()
        })).unwrap();
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let caps = surface.get_capabilities(&adapter);
        self.format = caps.formats[0];
        let size = win.inner_size();
        surface.configure(&device, &wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT, format: self.format,
            width: size.width, height: size.height, present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: caps.alpha_modes[0], view_formats: vec![], desired_maximum_frame_latency: 2,
        });

        // 100 particles in a grid
        let mut positions = Vec::new();
        for y in -5..5 {
            for x in -5..5 {
                positions.extend_from_slice(&[x as f32 * 0.5, y as f32 * 0.5, 0.0, 1.0]);
            }
        }
        self.count = (positions.len() / 4) as u32;

        let pos_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Positions"), contents: bytemuck::cast_slice(&positions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Params"), size: 144,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(r#"
struct Params { view: mat4x4<f32>, proj: mat4x4<f32>, size: f32, }

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<uniform> params: Params;

struct VOut { @builtin(position) pos: vec4<f32>, @location(0) color: vec3<f32>, }

const QUAD: array<vec2<f32>, 6> = array(
    vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
    vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
);

@vertex fn vs(@builtin(vertex_index) vi: u32) -> VOut {
    let pid = vi / 6u;
    let corner = QUAD[vi % 6u];
    let p = positions[pid];
    let s = params.size;
    let right = vec3<f32>(params.view[0][0], params.view[1][0], params.view[2][0]);
    let up    = vec3<f32>(params.view[0][1], params.view[1][1], params.view[2][1]);
    let wp = p.xyz + right * corner.x * s + up * corner.y * s;
    var out: VOut;
    out.pos = params.proj * params.view * vec4<f32>(wp, 1.0);
    out.color = vec3<f32>(0.3, 0.7, 1.0);
    return out;
}

@fragment fn fs(v: VOut) -> @location(0) vec4<f32> { return vec4<f32>(v.color, 1.0); }
"#.into()),
        });

        use wgpu::util::DeviceExt;
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None, layout: None,
            vertex: wgpu::VertexState { module: &shader, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &shader, entry_point: Some("fs"),
                targets: &[Some(wgpu::ColorTargetState { format: self.format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: Default::default(), depth_stencil: None,
            multisample: Default::default(), multiview: None, cache: None,
        });

        let bgl = pipeline.get_bind_group_layout(0);
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None, layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: pos_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: params_buf.as_entire_binding() },
            ],
        });

        self.pipeline = Some(pipeline);
        self.bind_group = Some(bg);
        self.params_buf = Some(params_buf);
        self.surface = Some(surface);
        self.device = Some(device);
        self.queue = Some(queue);
        self.window = Some(win);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::RedrawRequested => {
                let device = self.device.as_ref().unwrap();
                let queue = self.queue.as_ref().unwrap();

                // Camera at (0, 0, 10) looking at origin
                let view = glam::Mat4::look_at_rh(
                    glam::Vec3::new(0.0, 0.0, 10.0),
                    glam::Vec3::ZERO,
                    glam::Vec3::Y,
                );
                let size = self.window.as_ref().unwrap().inner_size();
                let aspect = size.width as f32 / size.height as f32;
                let proj = glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect, 0.1, 100.0);

                let mut data = [0.0f32; 36];
                data[..16].copy_from_slice(&view.to_cols_array());
                data[16..32].copy_from_slice(&proj.to_cols_array());
                data[32] = 0.15;
                queue.write_buffer(self.params_buf.as_ref().unwrap(), 0, bytemuck::cast_slice(&data));

                let output = self.surface.as_ref().unwrap().get_current_texture().unwrap();
                let tex_view = output.texture.create_view(&Default::default());
                let mut enc = device.create_command_encoder(&Default::default());
                {
                    let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &tex_view, resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.05, g: 0.05, b: 0.1, a: 1.0 }), store: wgpu::StoreOp::Store },
                        })],
                        ..Default::default()
                    });
                    pass.set_pipeline(self.pipeline.as_ref().unwrap());
                    pass.set_bind_group(0, self.bind_group.as_ref().unwrap(), &[]);
                    pass.draw(0..self.count * 6, 0..1);
                }
                queue.submit(std::iter::once(enc.finish()));
                output.present();
                self.window.as_ref().unwrap().request_redraw();
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
