use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::Material;
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pathtracer::{BVHBuilder, GPUBVHData, PathTracer, PathTracerMaterial, TLASBuilder};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{fetch_bytes, Canvas};


const BLIT_SHADER: &str = "
@group(0) @binding(0) var t_input: texture_2d<f32>;
@group(0) @binding(1) var s_input: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let x = f32(i32(idx & 1u) * 4 - 1);
    let y = f32(i32(idx & 2u) * 2 - 1);
    var out: VertexOutput;
    out.position = vec4f(x, y, 0.0, 1.0);
    out.uv = vec2f((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4f {
    let color = textureSample(t_input, s_input, in.uv);
    let mapped = color.rgb / (color.rgb + vec3f(1.0));
    return vec4f(mapped, 1.0);
}
";

fn make_basic_material(name: &str, color: [f32; 4]) -> Material {
    Material::basic_lit(name, color, [0.15, 0.15, 0.15, 0.5])
}

/// Blit pipeline resources for presenting the path tracer output to the canvas.
struct BlitResources {
    pipeline: wgpu::RenderPipeline,
    bgl: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl BlitResources {
    fn new(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Blit/Shader"),
            source: wgpu::ShaderSource::Wgsl(BLIT_SHADER.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Blit/BGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Blit/PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Blit/Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Blit/Sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        Self { pipeline, bgl, sampler }
    }

    fn blit(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        source_view: &wgpu::TextureView,
        target_view: &wgpu::TextureView,
    ) {
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Blit/BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Blit/Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    path_tracer: PathTracer,
    bvh: BVHBuilder,
    bvh_data: GPUBVHData,
    tlas: TLASBuilder,
    blit: BlitResources,
    // Interactive scene controls
    box_a_idx: usize,
    box_b_idx: usize,
    dragon_parts: Vec<(usize, Vec3)>,
    light_dir: Vec3,
    light_color: Vec3,
    light_intensity: f32,
    animate: bool,
    scene_dirty: bool,
}

fn directional_light_data(dir: Vec3, color: Vec3, intensity: f32) -> [f32; 16] {
    let d = if dir.length() > 1e-6 {
        dir.normalize()
    } else {
        Vec3::new(0.0, -1.0, 0.0)
    };
    // LightData: direction(vec3) + LIGHT_DIRECTIONAL=1, color(vec3) + intensity,
    // normal(vec3, unused) + pad, extra(vec4)
    [
        d.x, d.y, d.z, f32::from_bits(1u32),
        color.x, color.y, color.z, intensity,
        0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0,
    ]
}

fn move_object(scene: &mut Scene, idx: usize, pos: Vec3) {
    if let Some(r) = scene.get_renderable_mut(idx) {
        r.object.position = pos;
        r.object.update_model_matrix();
        r.object.update_world_matrix(None);
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let page = Canvas::find(canvas_id)?;
    let canvas = page.element();
    let (width, height) = page.size();
    let renderer = page
        .renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() })
        .await;

    // Build scene
    let mut scene = Scene::new();

    // Cornell box: 8 wide, 6 tall, 8 deep, centered at origin
    // Floor (white)
    let floor_geo = BoxGeometry::new(8.0, 0.1, 8.0);
    let floor_mat = make_basic_material("Floor", [0.73, 0.73, 0.73, 1.0]);
    let mut floor = Renderable::new(floor_geo, floor_mat);
    floor.object.set_position(0.0, -0.05, 0.0);
    scene.add(SceneNode::Renderable(floor));

    // Ceiling (white)
    let ceil_geo = BoxGeometry::new(8.0, 0.1, 8.0);
    let ceil_mat = make_basic_material("Ceiling", [0.73, 0.73, 0.73, 1.0]);
    let mut ceil = Renderable::new(ceil_geo, ceil_mat);
    ceil.object.set_position(0.0, 6.05, 0.0);
    scene.add(SceneNode::Renderable(ceil));

    // Back wall (white)
    let back_geo = BoxGeometry::new(8.0, 6.0, 0.1);
    let back_mat = make_basic_material("BackWall", [0.73, 0.73, 0.73, 1.0]);
    let mut back = Renderable::new(back_geo, back_mat);
    back.object.set_position(0.0, 3.0, -4.05);
    scene.add(SceneNode::Renderable(back));

    // Left wall (red)
    let left_geo = BoxGeometry::new(0.1, 6.0, 8.0);
    let left_mat = make_basic_material("LeftWall", [0.65, 0.05, 0.05, 1.0]);
    let mut left = Renderable::new(left_geo, left_mat);
    left.object.set_position(-4.05, 3.0, 0.0);
    scene.add(SceneNode::Renderable(left));

    // Right wall (green)
    let right_geo = BoxGeometry::new(0.1, 6.0, 8.0);
    let right_mat = make_basic_material("RightWall", [0.12, 0.45, 0.15, 1.0]);
    let mut right = Renderable::new(right_geo, right_mat);
    right.object.set_position(4.05, 3.0, 0.0);
    scene.add(SceneNode::Renderable(right));

    // Box A
    let box_a_geo = BoxGeometry::new(1.0, 2.0, 1.0);
    let box_a_mat = make_basic_material("BoxA", [0.9, 0.2, 0.2, 1.0]);
    let mut box_a = Renderable::new(box_a_geo, box_a_mat);
    box_a.object.set_position(-1.5, 1.0, 0.0);
    let box_a_idx = scene.add(SceneNode::Renderable(box_a));

    // Box B
    let box_b_geo = BoxGeometry::new(1.0, 1.0, 1.0);
    let box_b_mat = make_basic_material("BoxB", [0.2, 0.2, 0.9, 1.0]);
    let mut box_b = Renderable::new(box_b_geo, box_b_mat);
    box_b.object.set_position(1.5, 0.5, 0.0);
    let box_b_idx = scene.add(SceneNode::Renderable(box_b));

    // Stanford Dragon (glass) — fetch GLB via HTTP
    let dragon_loaded = fetch_bytes("assets/stanford_dragon_pbr.glb")
        .await
        .ok()
        .and_then(|bytes| GLTFLoader::load_glb(&bytes).ok());

    let mut dragon_parts: Vec<(usize, Vec3)> = Vec::new();
    if let Some(result) = dragon_loaded {
        log::info!("Loaded dragon: {} renderables", result.renderables.len());
        // GLB is ~100 units tall; scale to ~3 units to fit scene
        let s = 0.03;
        for gr in result.renderables {
            let mut r = Renderable::new(gr.geometry, make_basic_material("Dragon", [0.9, 0.9, 0.95, 1.0]));
            let base_pos = Vec3::new(gr.position.x, gr.position.y, gr.position.z + 2.5);
            r.object.position = base_pos;
            r.object.rotation = gr.rotation;
            r.object.scale = Vec3::new(gr.scale.x * s, gr.scale.y * s, gr.scale.z * s);
            r.object.update_model_matrix();
            r.object.update_world_matrix(None);
            let idx = scene.add(SceneNode::Renderable(r));
            dragon_parts.push((idx, base_pos));
        }
    } else {
        log::warn!("Could not load dragon model");
    }

    // Directional light
    // Low and from the front, so it shines in through the box's open side (the ceiling is solid)
    let sun = DirectionalLight::new(
        Vec3::new(-0.3, -0.5, -1.0).normalize(),
        Vec3::new(1.0, 0.95, 0.9),
        3.0,
    );
    scene.add(SceneNode::Light(Light::Directional(sun)));

    // Camera
    let mut camera = Camera::new(45.0, 0.1, 100.0, width as f32 / height as f32);
    // Back far enough to frame the whole box
    camera.set_position(0.0, 3.0, 14.0);
    camera.look_at(&Vec3::new(0.0, 3.0, 0.0));
    camera.update_projection_matrix();
    camera.update_view_matrix();
    scene.prepare(&camera.position());

    // Build BVH
    let mut bvh = BVHBuilder::new();
    let mut tlas = TLASBuilder::new(&renderer);
    let gpu_data = bvh.build_full(&renderer, &scene, &mut tlas);
    log::info!(
        "BVH built: {} triangles, {} BVH4 nodes, {} instances",
        gpu_data.triangle_count,
        gpu_data.node_count,
        gpu_data.instance_count,
    );

    // Create PathTracer
    let mut pt = PathTracer::new(&renderer);
    pt.resize(width, height);
    pt.set_spp(1);
    pt.set_max_bounces(8); // more bounces for glass refraction

    // Materials: floor=0, ceiling=1, back=2, left=3, right=4, boxA=5, boxB=6, dragon=7+
    let mut materials = vec![
        PathTracerMaterial { albedo: [0.73, 0.73, 0.73], ..Default::default() }, // floor
        PathTracerMaterial { albedo: [0.73, 0.73, 0.73], ..Default::default() }, // ceiling
        PathTracerMaterial { albedo: [0.73, 0.73, 0.73], ..Default::default() }, // back wall
        PathTracerMaterial { albedo: [0.65, 0.05, 0.05], ..Default::default() }, // left (red)
        PathTracerMaterial { albedo: [0.12, 0.45, 0.15], ..Default::default() }, // right (green)
        PathTracerMaterial { albedo: [0.9, 0.2, 0.2], ..Default::default() },    // box A
        PathTracerMaterial { albedo: [0.2, 0.2, 0.9], ..Default::default() },    // box B
    ];
    let dragon_count = gpu_data.instance_count as usize - 7;
    for _ in 0..dragon_count {
        materials.push(PathTracerMaterial::glass(1.5));
    }
    pt.set_materials(&materials);

    // Light data for path tracer
    let light_dir = Vec3::new(-0.3, -0.5, -1.0);
    let light_color = Vec3::new(1.0, 0.95, 0.9);
    let light_intensity = 3.0;
    pt.set_lights_raw(&directional_light_data(light_dir, light_color, light_intensity));

    // Blit pipeline
    let blit = BlitResources::new(renderer.device(), renderer.presentation_format());

    log::info!("Kansei — Path Tracer (WASM) ready");

    let controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 3.0, 0.0), 14.0);

    let state = Rc::new(RefCell::new(State {
        renderer,
        scene,
        camera,
        controls,
        path_tracer: pt,
        bvh,
        bvh_data: gpu_data,
        tlas,
        blit,
        box_a_idx,
        box_b_idx,
        dragon_parts,
        light_dir,
        light_color,
        light_intensity,
        animate: false,
        scene_dirty: false,
    }));

    let s = state.clone();
    kansei_wasm::run(&page, move |frame| {
        {
            let mut st = s.borrow_mut();
            let State {
                ref mut renderer,
                ref mut scene,
                ref mut camera,
                ref mut controls,
                ref mut path_tracer,
                ref mut bvh,
                ref bvh_data,
                ref mut tlas,
                ref blit,
                box_b_idx,
                animate,
                ref mut scene_dirty,
                ..
            } = *st;
            // The tracer's output (and the blit reading it) follow the canvas size.
            if let Some((width, height)) = frame.resized {
                frame.resize(renderer, camera);
                path_tracer.resize(width, height);
                path_tracer.reset_accumulation();
            }
            if controls.is_dirty() {
                path_tracer.reset_accumulation();
            }
            controls.update(camera, 0.0);

            // Animate box B in an orbit (mirrors the TS example's animated cube)
            if animate {
                let t = frame.time as f32;
                if let Some(r) = scene.get_renderable_mut(box_b_idx) {
                    r.object.position = Vec3::new(t.sin() * 2.5, t.cos() * 1.5 + 2.5, 0.0);
                    r.object.rotation = Vec3::new(t * 0.7, t * 1.1, t * 0.5);
                    r.object.update_model_matrix();
                    r.object.update_world_matrix(None);
                }
                *scene_dirty = true;
            }

            // Objects moved: re-pack instance transforms + rebuild TLAS (BLAS untouched)
            if *scene_dirty {
                bvh.refresh_transforms(renderer, scene, bvh_data, tlas);
                path_tracer.reset_accumulation();
                *scene_dirty = false;
            }

            // Get surface texture
            let surface = renderer.surface().unwrap();
            let output = surface.get_current_texture().expect("Failed to get surface texture");
            let canvas_view = output.texture.create_view(&Default::default());

            let mut encoder = renderer.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor {
                    label: Some("PathTracer/Frame"),
                },
            );

            let tlas_buf = tlas.tlas_nodes_buf.as_ref().unwrap();
            path_tracer.trace(&mut encoder, bvh_data, tlas_buf, camera, 1);

            if let Some(pt_view) = path_tracer.output_view() {
                let device = renderer.device();
                blit.blit(&mut encoder, device, pt_view, &canvas_view);
            }

            renderer.queue().submit(std::iter::once(encoder.finish()));
            output.present();
        }
    });

    GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });

    Ok(())
}

// ── JS interop ──
thread_local! { static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None); }
fn with_state<F: FnOnce(&mut State)>(f: F) {
    GLOBAL_STATE.with(|gs| { if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); } });
}

fn upload_lights(s: &mut State) {
    s.path_tracer.set_lights_raw(&directional_light_data(
        s.light_dir,
        s.light_color,
        s.light_intensity,
    ));
    s.path_tracer.reset_accumulation();
}

#[wasm_bindgen]
pub fn set_light_dir(x: f32, y: f32, z: f32) {
    with_state(|s| {
        s.light_dir = Vec3::new(x, y, z);
        upload_lights(s);
    });
}

#[wasm_bindgen]
pub fn set_light_color(r: f32, g: f32, b: f32) {
    with_state(|s| {
        s.light_color = Vec3::new(r, g, b);
        upload_lights(s);
    });
}

#[wasm_bindgen]
pub fn set_light_intensity(v: f32) {
    with_state(|s| {
        s.light_intensity = v;
        upload_lights(s);
    });
}

/// Move an object: 0 = box A, 1 = box B, 2 = dragon (offset from load position).
#[wasm_bindgen]
pub fn set_object_position(id: u32, x: f32, y: f32, z: f32) {
    with_state(|s| {
        match id {
            0 => move_object(&mut s.scene, s.box_a_idx, Vec3::new(x, y, z)),
            1 => move_object(&mut s.scene, s.box_b_idx, Vec3::new(x, y, z)),
            2 => {
                let parts = s.dragon_parts.clone();
                for (idx, base) in parts {
                    let pos = Vec3::new(base.x + x, base.y + y, base.z + z);
                    move_object(&mut s.scene, idx, pos);
                }
            }
            _ => return,
        }
        s.scene_dirty = true;
    });
}

#[wasm_bindgen]
pub fn set_animate(v: bool) {
    with_state(|s| {
        s.animate = v;
        s.scene_dirty = true;
    });
}

#[wasm_bindgen] pub fn set_spp(v: u32) { with_state(|s| { s.path_tracer.set_spp(v); s.path_tracer.reset_accumulation(); }); }
#[wasm_bindgen] pub fn set_max_bounces(v: u32) { with_state(|s| { s.path_tracer.set_max_bounces(v); s.path_tracer.reset_accumulation(); }); }
#[wasm_bindgen] pub fn set_use_blue_noise(v: bool) { with_state(|s| { s.path_tracer.set_use_blue_noise(v); s.path_tracer.reset_accumulation(); }); }
#[wasm_bindgen] pub fn reset_accumulation() { with_state(|s| s.path_tracer.reset_accumulation()); }
#[wasm_bindgen] pub fn get_frame_index() -> u32 {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() { rc.borrow().path_tracer.frame_index() } else { 0 }
    })
}
