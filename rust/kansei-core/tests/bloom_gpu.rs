//! Runs BloomEffect on a synthetic frame on a real GPU: two equally bright spots in the scene,
//! only the right one emissive. Read from the scene both bloom; with `use_emissive` the bloom
//! added is exactly the one a scene holding only the emissive spot gets, so the lit spot adds
//! none. Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::postprocessing::effects::{BloomEffect, BloomOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect};

const W: u32 = 128;
const H: u32 = 64;
/// The spots: 8 x 8 pixels round (32, 32) and (96, 32), radiance 20.
const LEFT: u32 = 32;
const RIGHT: u32 = 96;
const SPOT: f32 = 20.0;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    sign * match exp {
        0 => frac * 2f32.powi(-24),
        31 => f32::INFINITY,
        _ => (1.0 + frac / 1024.0) * 2f32.powi(exp - 15),
    }
}

/// Draws the spots into a COLOR_FORMAT target: `both` the two, `right` the right one only.
const SPOTS_WGSL: &str = r#"
const SPOT: f32 = 20.0;
@vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {
    let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(p * 2.0 - 1.0, 0.0, 1.0);
}
fn spot(pos: vec2f, x: f32) -> f32 {
    return select(0.0, SPOT, abs(pos.x - x) < 4.0 && abs(pos.y - 32.0) < 4.0);
}
@fragment fn both(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    return vec4f(vec3f(spot(pos.xy, 32.0) + spot(pos.xy, 96.0)), 1.0);
}
@fragment fn right(@builtin(position) pos : vec4f) -> @location(0) vec4f {
    return vec4f(vec3f(spot(pos.xy, 96.0)), 1.0);
}
"#;

struct Harness {
    device: wgpu::Device,
    queue: wgpu::Queue,
    gbuffer: GBuffer,
    /// The scene with both spots, and with the right one alone.
    both: wgpu::Texture,
    right: wgpu::Texture,
    output: wgpu::Texture,
}

impl Harness {
    fn new() -> Option<Self> {
        let (device, queue) = gpu()?;
        let gbuffer = GBuffer::new(&device, W, H, 1);
        let texture = |usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: GBuffer::COLOR_FORMAT,
                usage,
                view_formats: &[],
            })
        };
        let scene = || texture(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING);
        let (both, right) = (scene(), scene());
        let output = texture(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let harness = Self { device, queue, gbuffer, both, right, output };
        // the scenes; the emissive target: the right spot only
        harness.draw("both", &harness.both.create_view(&Default::default()));
        harness.draw("right", &harness.right.create_view(&Default::default()));
        harness.draw("right", &harness.gbuffer.emissive_view);
        Some(harness)
    }

    fn draw(&self, entry: &str, target: &wgpu::TextureView) {
        let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(SPOTS_WGSL.into()) });
        let pipeline = self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some(entry), targets: &[Some(GBuffer::COLOR_FORMAT.into())], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.draw(0..3, 0..1);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// The bloomed frame's red channel, row-major: scene `input`, bloomed from it or from the
    /// emissive target.
    fn bloom(&self, input: &wgpu::Texture, use_emissive: bool) -> Vec<f32> {
        let camera = Camera::new(45.0, 0.1, 100.0, W as f32 / H as f32);
        let mut bloom = BloomEffect::new(BloomOptions { threshold: 1.0, knee: 0.1, intensity: 1.0, radius: 1.0, use_emissive });
        bloom.initialize(&self.device, &self.gbuffer, &camera);
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let input = input.create_view(&Default::default());
        let output = self.output.create_view(&Default::default());
        bloom.render(&self.device, &self.queue, &mut encoder, &self.gbuffer, &input, &self.gbuffer.depth_view, &output, &camera, W, H);
        let row = W * 8;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * H) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_texture_to_buffer(
            self.output.as_image_copy(),
            wgpu::TexelCopyBufferInfo { buffer: &staging, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
            wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        );
        self.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        let data = staging.slice(..).get_mapped_range().unwrap();
        let halves: &[u16] = bytemuck::cast_slice(&data);
        halves.chunks_exact(4).map(|t| f16_to_f32(t[0])).collect()
    }
}

/// A scene's red channel: `SPOTS_WGSL`'s spots, the left one too with `both`.
fn scene(both: bool) -> Vec<f32> {
    let spot = |x: f32, y: f32, cx: f32| if (x - cx).abs() < 4.0 && (y - 32.0).abs() < 4.0 { SPOT } else { 0.0 };
    (0..H)
        .flat_map(|y| (0..W).map(move |x| (x as f32 + 0.5, y as f32 + 0.5)))
        .map(|(x, y)| spot(x, y, RIGHT as f32) + if both { spot(x, y, LEFT as f32) } else { 0.0 })
        .collect()
}

#[test]
fn bloom_from_emissive_blooms_only_what_emits() {
    let Some(harness) = Harness::new() else {
        eprintln!("no GPU adapter: skipped");
        return;
    };
    let added = |image: &[f32], base: &[f32]| image.iter().zip(base).map(|(a, b)| a - b).collect::<Vec<f32>>();
    let at = |image: &[f32], x: u32| image[(32 * W + x) as usize];

    // from the scene, each spot blooms: beside the left one there is its own halo
    let from_scene = added(&harness.bloom(&harness.both, false), &scene(true));
    // from the emissive target (the right spot only) the bloom added is the one a scene of the
    // right spot alone gets: none of the lit spot's
    let from_emissive = added(&harness.bloom(&harness.both, true), &scene(true));
    let right_alone = added(&harness.bloom(&harness.right, false), &scene(false));
    // (within half-float steps: 1/64 on the lit spot's 20)
    let both = scene(true);
    for (k, (a, b)) in from_emissive.iter().zip(&right_alone).enumerate() {
        assert!((a - b).abs() <= 0.01 + 0.01 * b.abs() + both[k] / 512.0, "pixel {k}: {a} from the emissive target, {b} from the right spot alone");
    }
    let (beside_right, beside_left) = (at(&from_emissive, RIGHT + 12), at(&from_emissive, LEFT + 12));
    assert!(beside_right > 0.05, "the emissive spot blooms: {beside_right}");
    assert!(at(&from_scene, LEFT + 12) > beside_left + 0.05, "the lit spot's halo is gone: {} from the scene, {beside_left} from the emissive target", at(&from_scene, LEFT + 12));
}
