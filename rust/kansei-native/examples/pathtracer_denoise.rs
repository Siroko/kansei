//! The path tracer's denoiser (`PathTracerDenoise`) measured headless, on the web path tracer's
//! scene (its Cornell box lit by a low sun through the open front, two boxes), alone (`cornell`)
//! or with the glass Stanford dragon (`dragon`):
//!
//! - `reference`: the view the other modes end on, accumulated over `--samples` jittered samples
//!   a pixel in f32, written to `<out>/<scene>/reference.f32`.
//! - `eval`: the camera orbits for `--frames` frames onto that view, then holds still for as
//!   many; at the end of each, the tracer's own frame (`noisy`) and each denoiser setting's
//!   output are written as `<out>/<scene>/spp<N>/<moving|still>/<setting>.f32`. Every setting
//!   denoises the same traced frames.
//! - `time`: the denoiser's passes, timed by the profiler over `--frames` frames of the orbit,
//!   with the trace's, at `--size`.
//!
//! Images are raw: width and height (u32), then RGBA f32 rows, linear HDR.
//!
//! ```text
//! cargo run --release -p kansei-native --example pathtracer_denoise -- reference|eval|time [--scene cornell|dragon] [--size WxH] [--spp N] [--samples N] [--frames N] [--out DIR]
//! ```

use std::path::{Path, PathBuf};

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::Material;
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, GPUBVHData, PathTracer, PathTracerDenoise, PathTracerDenoiseKernel, PathTracerDenoiseMode, PathTracerMaterial, TLASBuilder};
use kansei_core::renderers::{Renderer, RendererConfig};

struct Args {
    mode: String,
    scene: String,
    width: u32,
    height: u32,
    spp: u32,
    samples: u32,
    frames: u32,
    out: PathBuf,
}

fn args() -> Args {
    let mut a = Args { mode: String::new(), scene: "cornell".into(), width: 1920, height: 1080, spp: 1, samples: 4096, frames: 60, out: PathBuf::from("pathtracer-denoise") };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().expect("a value");
        match arg.as_str() {
            "--scene" => a.scene = value(),
            "--size" => {
                let v = value();
                let (w, h) = v.split_once('x').expect("WxH");
                (a.width, a.height) = (w.parse().unwrap(), h.parse().unwrap());
            }
            "--spp" => a.spp = value().parse().unwrap(),
            "--samples" => a.samples = value().parse().unwrap(),
            "--frames" => a.frames = value().parse().unwrap(),
            "--out" => a.out = value().into(),
            m => a.mode = m.into(),
        }
    }
    a
}

fn lit(name: &str, color: [f32; 3]) -> Material {
    Material::basic_lit(name, [color[0], color[1], color[2], 1.0], [0.15, 0.15, 0.15, 0.5])
}

/// The web path tracer's scene (rust/kansei-wasm/examples/pathtracer), with or without its dragon.
fn scene(dragon: bool) -> Scene {
    let mut scene = Scene::new();
    let mut add = |w: f32, h: f32, d: f32, pos: [f32; 3], color: [f32; 3]| {
        let mut r = Renderable::new(BoxGeometry::new(w, h, d), lit("Box", color));
        r.path_tracer_material = Some(PathTracerMaterial { albedo: color, ..Default::default() });
        r.object.set_position(pos[0], pos[1], pos[2]);
        scene.add(SceneNode::Renderable(r));
    };
    let white = [0.73, 0.73, 0.73];
    add(8.0, 0.1, 8.0, [0.0, -0.05, 0.0], white);
    add(8.0, 0.1, 8.0, [0.0, 6.05, 0.0], white);
    add(8.0, 6.0, 0.1, [0.0, 3.0, -4.05], white);
    add(0.1, 6.0, 8.0, [-4.05, 3.0, 0.0], [0.65, 0.05, 0.05]);
    add(0.1, 6.0, 8.0, [4.05, 3.0, 0.0], [0.12, 0.45, 0.15]);
    add(1.0, 2.0, 1.0, [-1.5, 1.0, 0.0], [0.9, 0.2, 0.2]);
    add(1.0, 1.0, 1.0, [1.5, 0.5, 0.0], [0.2, 0.2, 0.9]);
    if dragon {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../kansei-wasm/examples/pathtracer/www/assets/stanford_dragon_pbr.glb");
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let s = 0.03;
        for gr in GLTFLoader::load_glb(&bytes).expect("dragon").renderables {
            let mut r = Renderable::new(gr.geometry, lit("Dragon", [0.9, 0.9, 0.95]));
            r.path_tracer_material = Some(PathTracerMaterial::glass(1.5));
            r.object.position = Vec3::new(gr.position.x, gr.position.y, gr.position.z + 2.5);
            r.object.rotation = gr.rotation;
            r.object.scale = Vec3::new(gr.scale.x * s, gr.scale.y * s, gr.scale.z * s);
            r.object.update_model_matrix();
            r.object.update_world_matrix(None);
            scene.add(SceneNode::Renderable(r));
        }
    }
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(-0.3, -0.5, -1.0).normalize(), Vec3::new(1.0, 0.95, 0.9), 9.5))));
    scene
}

/// The orbit's frame `k` of `frames`: the camera swings 20 degrees about the box's centre onto
/// the view every mode ends on (12 degrees off the axis, a little above), at 0.33 degrees a frame.
fn place(camera: &mut Camera, k: u32, frames: u32) {
    let end = 12f32.to_radians();
    let yaw = end - (frames.saturating_sub(1).saturating_sub(k)) as f32 * (20f32 / 60.0).to_radians();
    let (r, h) = (13.0, 3.6);
    camera.set_position(r * yaw.sin(), h, r * yaw.cos());
    camera.look_at(&Vec3::new(0.0, 2.8, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();
}

struct Setup {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    tracer: PathTracer,
    bvh_data: GPUBVHData,
    tlas: TLASBuilder,
    light_count: u32,
}

fn setup(adapter: &wgpu::Adapter, args: &Args) -> Setup {
    let (w, h) = (args.width, args.height);
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(adapter));
    let mut scene = scene(args.scene == "dragon");
    let mut camera = Camera::new(45.0, 0.1, 100.0, w as f32 / h as f32);
    place(&mut camera, 0, 1);
    scene.prepare(&camera.position());
    let mut bvh = BVHBuilder::new();
    let mut tlas = TLASBuilder::new(&renderer);
    let bvh_data = bvh.build_full(&renderer, &scene, &mut tlas);
    let mut tracer = PathTracer::new(&renderer);
    tracer.resize(w, h);
    tracer.set_spp(args.spp);
    tracer.set_max_bounces(8);
    tracer.set_materials(&BVHBuilder::scene_materials(&scene));
    if std::env::var("PT_PCG").is_ok() {
        tracer.set_use_blue_noise(false);
    }
    if std::env::var("PT_NOJITTER").is_ok() {
        tracer.set_jitter(false);
    }
    let light_count = tracer.set_lights_from_scene(&scene);
    Setup { renderer, scene, camera, tracer, bvh_data, tlas, light_count }
}

/// Reads rgba16float textures back as f32 (or adds them into a sum), through a compute copy.
struct Reader {
    pipeline: wgpu::ComputePipeline,
    buffer: wgpu::Buffer,
    params: wgpu::Buffer,
    width: u32,
    height: u32,
}

impl Reader {
    fn new(renderer: &Renderer, width: u32, height: u32) -> Self {
        let device = renderer.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Reader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var src : texture_2d<f32>;
                @group(0) @binding(1) var<storage, read_write> dst : array<vec4f>;
                @group(0) @binding(2) var<uniform> add : vec4u;
                @compute @workgroup_size(8, 8) fn main(@builtin(global_invocation_id) id : vec3u) {
                    let size = textureDimensions(src);
                    if (id.x >= size.x || id.y >= size.y) { return; }
                    let i = id.y * size.x + id.x;
                    let c = textureLoad(src, id.xy, 0);
                    dst[i] = select(c, dst[i] + c, add.x != 0u);
                }
                "#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some("Reader"), layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Reader"), size: (width * height * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
        let params = device.create_buffer(&wgpu::BufferDescriptor { label: Some("Reader/Params"), size: 16, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self { pipeline, buffer, params, width, height }
    }

    /// Copy `view` into the buffer, or add it to what the buffer holds.
    fn record(&self, renderer: &Renderer, encoder: &mut wgpu::CommandEncoder, view: &wgpu::TextureView, add: bool) {
        renderer.queue().write_buffer(&self.params, 0, bytemuck::cast_slice(&[add as u32, 0, 0, 0]));
        let bg = renderer.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: self.buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.params.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(self.width.div_ceil(8), self.height.div_ceil(8), 1);
    }

    /// The buffer, scaled by `scale`, written as a raw image.
    fn save(&self, renderer: &Renderer, path: &Path, scale: f32) {
        let data: Vec<f32> = renderer.read_back_buffer_sync(&self.buffer, (self.width * self.height * 16) as u64);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        let mut bytes = Vec::with_capacity(8 + data.len() * 4);
        bytes.extend_from_slice(&self.width.to_le_bytes());
        bytes.extend_from_slice(&self.height.to_le_bytes());
        for v in data {
            bytes.extend_from_slice(&(v * scale).to_le_bytes());
        }
        std::fs::write(path, bytes).unwrap();
        println!("wrote {}", path.display());
    }
}

fn trace(s: &mut Setup, encoder: &mut wgpu::CommandEncoder) {
    // each frame's own samples in the tracer's output (the noisy frame)
    s.tracer.reset_accumulation();
    let tlas = s.tlas.tlas_nodes_buf.as_ref().unwrap();
    s.tracer.trace(encoder, &s.bvh_data, tlas, &s.camera, s.light_count);
}

fn reference(adapter: &wgpu::Adapter, args: &Args) {
    let mut s = setup(adapter, args);
    let reader = Reader::new(&s.renderer, args.width, args.height);
    let per_dispatch = 8;
    s.tracer.set_spp(per_dispatch);
    place(&mut s.camera, 0, 1);
    let rounds = args.samples.div_ceil(per_dispatch);
    let t0 = std::time::Instant::now();
    for k in 0..rounds {
        let mut encoder = s.renderer.device().create_command_encoder(&Default::default());
        trace(&mut s, &mut encoder);
        reader.record(&s.renderer, &mut encoder, s.tracer.output_view().unwrap(), k > 0);
        s.renderer.queue().submit(Some(encoder.finish()));
        let _ = s.renderer.device().poll(wgpu::PollType::wait_indefinitely());
        if k % 64 == 63 {
            eprintln!("  {} of {} samples, {:.1} s", (k + 1) * per_dispatch, rounds * per_dispatch, t0.elapsed().as_secs_f64());
        }
    }
    reader.save(&s.renderer, &args.out.join(&args.scene).join("reference.f32"), 1.0 / rounds as f32);
}

/// The denoiser settings `eval` compares, by name.
fn settings(renderer: &Renderer) -> Vec<(&'static str, PathTracerDenoise)> {
    let make = |f: &dyn Fn(&mut PathTracerDenoise)| {
        let mut d = PathTracerDenoise::new(renderer);
        f(&mut d);
        d
    };
    vec![
        ("temporal", make(&|d| d.mode = PathTracerDenoiseMode::Temporal)),
        ("svgf", make(&|_| {})),
        ("svgf-3x3", make(&|d| d.kernel = PathTracerDenoiseKernel::Three)),
        ("svgf-no-taa", make(&|d| d.taa = false)),
        ("svgf-alpha0.1", make(&|d| d.alpha_color = 0.1)),
        ("svgf-clamp2", make(&|d| d.clamp_sigma = 2.0)),
    ]
}

fn eval(adapter: &wgpu::Adapter, args: &Args) {
    let mut s = setup(adapter, args);
    s.tracer.set_guides(true);
    let reader = Reader::new(&s.renderer, args.width, args.height);
    let mut settings = settings(&s.renderer);
    let dir = args.out.join(&args.scene).join(format!("spp{}", args.spp));
    for (phase, still) in [("moving", false), ("still", true)] {
        for k in 0..args.frames {
            place(&mut s.camera, if still { args.frames - 1 } else { k }, args.frames);
            let mut encoder = s.renderer.device().create_command_encoder(&Default::default());
            trace(&mut s, &mut encoder);
            for (_, d) in settings.iter_mut() {
                d.denoise(&mut encoder, &s.tracer, &s.camera);
            }
            s.renderer.queue().submit(Some(encoder.finish()));
            let _ = s.renderer.device().poll(wgpu::PollType::wait_indefinitely());
        }
        let save = |view: &wgpu::TextureView, name: &str| {
            let mut encoder = s.renderer.device().create_command_encoder(&Default::default());
            reader.record(&s.renderer, &mut encoder, view, false);
            s.renderer.queue().submit(Some(encoder.finish()));
            reader.save(&s.renderer, &dir.join(phase).join(format!("{name}.f32")), 1.0);
        };
        save(s.tracer.output_view().unwrap(), "noisy");
        for (name, d) in settings.iter() {
            save(d.output_view().unwrap(), name);
        }
    }
}

fn time(adapter: &wgpu::Adapter, args: &Args) {
    let mut s = setup(adapter, args);
    s.tracer.set_guides(true);
    let mut denoise = PathTracerDenoise::new(&s.renderer);
    s.renderer.set_profiling(true);
    let warmup = 20;
    for k in 0..warmup + args.frames {
        if k == warmup {
            let _ = s.renderer.take_profile();
        }
        place(&mut s.camera, k % 60, 60);
        let mut encoder = s.renderer.device().create_command_encoder(&Default::default());
        trace(&mut s, &mut encoder);
        denoise.denoise(&mut encoder, &s.tracer, &s.camera);
        s.renderer.queue().submit(Some(encoder.finish()));
        s.renderer.end_profiled_frame();
        let _ = s.renderer.device().poll(wgpu::PollType::wait_indefinitely());
    }
    let profile = s.renderer.take_profile();
    let info = s.renderer.adapter_info();
    println!("{} ({:?}), {}x{}, scene {}, {} spp, {} frames", info.name, info.backend, args.width, args.height, args.scene, args.spp, profile.gpu_frames);
    let mut denoiser = 0.0;
    for p in &profile.gpu {
        println!("  {:<28} busy {:>7.3} ms  exclusive {:>7.3} ms  x{}", p.label, p.busy_ms, p.exclusive_ms, p.count);
        if p.label.starts_with("PathTracerDenoise/") {
            denoiser += p.busy_ms;
        }
    }
    println!("  denoiser, busy summed: {denoiser:.3} ms");
    let _ = &s.scene;
}

fn main() {
    env_logger::init();
    let args = args();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    match args.mode.as_str() {
        "reference" => reference(&adapter, &args),
        "eval" => eval(&adapter, &args),
        "time" => time(&adapter, &args),
        m => panic!("mode {m:?}: reference, eval or time"),
    }
    std::process::exit(0);
}
