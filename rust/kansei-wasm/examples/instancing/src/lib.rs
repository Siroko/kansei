//! Instancing: a clone of square.felixmartinez.dev (three.js) on Kansei. A carpet of 128 x 128
//! cubes, one instanced draw, that an orange ball ploughs through: a compute shader moves every
//! cube each step (pushed out of the ball's way within 6.5 m, drawn towards it beyond, and
//! springing back home) and writes its position into the instance buffer the cubes are drawn
//! from (a vec4 per cube at vertex location 3, `Material::standard_lit`'s `OffsetScale`
//! instancing), so the positions never leave the GPU. A shadowed spot light lights the scene;
//! the cubes shadow each other and the floor through their own vertex shader. Drag to orbit,
//! wheel or pinch to zoom.
//!
//! The look is the original's: a palette row picked at random each load colours the cubes (its
//! middle swatch) and the fog (its first, halved); the colours go through three.js's unmanaged
//! pipeline, sRGB numbers lit as they are, ACES-fitted and written without encoding, then a
//! linear distance fog and a radial vignette in display space (`FogVignetteEffect`). The
//! simulation steps at a fixed 60 Hz, where the original stepped once per frame.
//!
//! URL parameters: `n=<cubes per side>` (default 128, 8 to 512), `palette=<row>` (0 to 10: the
//! palette row, default random), `shadows=0` (no spot shadow map).

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{Bindable, BufferType, BufferUsage, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Binding, BindingResource, ComputePass, Material, ShaderStages, StandardInstancing, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::pacing::FixedStep;
use kansei_core::postprocessing::effects::{ToneMapEffect, ToneMapOptions};
use kansei_core::postprocessing::{GBuffer, PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, param_or, Canvas};

/// The original's palette (assets/color_palette.png, 5 x 11): a row per look.
const PALETTE: [[u32; 5]; 11] = [
    [0x1b1b3a, 0xdad1e9, 0xffb64d, 0xe2370c, 0x701523],
    [0x0d4631, 0x419e09, 0xf3dc6b, 0xfff2f1, 0xfb4861],
    [0x2c111a, 0xf5af06, 0x6c9ae6, 0xcef151, 0xf5f1ce],
    [0xf7c693, 0xff948f, 0xff6167, 0xa91832, 0x671427],
    [0x2f4a6f, 0xb663b3, 0xda8bbd, 0xaef5c6, 0xffffb6],
    [0x879be3, 0xfbfeff, 0xa4b2b6, 0x292e34, 0x27292b],
    [0x646b6c, 0x37112f, 0x9c0e57, 0xef0041, 0xff939b],
    [0x1e291a, 0xe8a11a, 0xfe5632, 0xebd5cb, 0x2981a5],
    [0xff6da3, 0x8d0e77, 0x1d42d0, 0x35deff, 0x98ffe2],
    [0x1b1c2d, 0x005ca9, 0x4dede0, 0xf9f15b, 0xff5440],
    [0x9d6163, 0xe4ceab, 0xe59fa3, 0x75b0d7, 0x3d97be],
];

fn rgb(hex: u32, scale: f32) -> [f32; 3] {
    [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255].map(|c| c as f32 / 255.0 * scale)
}

// The original's scene in metres (it measured in centimetres)
/// A cube's size and the carpet's pitch.
const SPACING: f32 = 0.1;
const BALL_RADIUS: f32 = 1.0;
/// The simulation's step, s.
const STEP: f64 = 1.0 / 60.0;
/// Where the ball is for the very first step: 100 m up, so the carpet leaps towards it and drops
/// back into place, as the original's does.
const BALL_START: [f32; 3] = [0.0, 100.0, 0.0];

/// The ball's path at `t` seconds: a figure eight that bobs a little.
fn ball_at(t: f32) -> [f32; 3] {
    [
        8.0 * (5.0 * t).sin() * t.cos(),
        0.3 * (5.0 * t).sin() * t.cos(),
        8.0 * (0.5 * t).cos() * (-t).sin(),
    ]
}

/// One step per cube, as the original's position shader: every cube is drawn back a tenth of the
/// way home and pushed along the ball's radius by 0.4 m x (1 - distance / 6.5 m), out of the
/// ball's way when it is near and towards it when it is far.
const STEP_WGSL: &str = r#"
struct Params { ball : vec4f, count : vec4f }
@group(0) @binding(0) var<uniform> params : Params;
@group(0) @binding(1) var<storage, read_write> positions : array<vec4f>;
@group(0) @binding(2) var<storage, read> homes : array<vec4f>;

const RETURN = 0.1;
const PUSH = 0.4;
const REACH = 6.5;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let i = id.x;
    if (i >= u32(params.count.x)) { return; }
    let home = homes[i].xyz;
    var p = positions[i].xyz;
    for (var s = 0u; s < u32(params.ball.w); s++) {
        let away = p - params.ball.xyz;
        let d = length(away);
        let push = select(vec3f(0.0), away / d, d > 1e-6) * (1.0 - d / REACH) * PUSH;
        p += (home - p) * RETURN + push;
    }
    positions[i] = vec4f(p, 1.0);
}
"#;

/// three.js's linear fog (a smoothstep over the view depth) after tone mapping, then its
/// vignette: darker with the distance from the middle (y scaled by the aspect), dithered.
const FOG_VIGNETTE_WGSL: &str = r#"
struct Params { fogColor : vec4f, range : vec4f, size : vec4f }
@group(0) @binding(0) var inputTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id : vec3u) {
    let size = params.size.xy;
    if (f32(id.x) >= size.x || f32(id.y) >= size.y) { return; }
    var color = textureLoad(inputTex, id.xy, 0).rgb;
    // view depth from the [0, 1] depth: range.zw are the camera's near and far
    let depth = textureLoad(depthTex, id.xy, 0);
    let near = params.range.z;
    let far = params.range.w;
    let z = near * far / (far - depth * (far - near));
    color = mix(color, params.fogColor.rgb, smoothstep(params.range.x, params.range.y, z));
    let pixel = vec2f(id.xy) + 0.5;
    var q = pixel / size - 0.5;
    q.y /= size.x / size.y;
    let vignette = smoothstep(0.0, 0.99, length(q));
    let dither = mix(-6.0 / 255.0, 6.0 / 255.0, fract(sin(dot(pixel + params.size.z, vec2f(12.9898, 78.233))) * 43758.5453));
    color = color * (1.0 - vignette) + dither * vignette;
    textureStore(outputTex, id.xy, vec4f(color, 1.0));
}
"#;

/// The original's fog and vignette, drawn on the tone-mapped picture.
struct FogVignetteEffect {
    color: [f32; 3],
    near: f32,
    far: f32,
    frame: u32,
    pass: ComputePass,
    params: ComputeBuffer,
}

impl FogVignetteEffect {
    fn new(color: [f32; 3], near: f32, far: f32) -> Self {
        let pass = ComputePass::new("FogVignette", FOG_VIGNETTE_WGSL, vec![
            Binding::texture_2d(0, ShaderStages::COMPUTE),
            Binding::texture_depth(1, ShaderStages::COMPUTE),
            Binding::storage_texture_2d(2, ShaderStages::COMPUTE, GBuffer::COLOR_FORMAT),
            Binding::uniform(3, ShaderStages::COMPUTE),
        ]);
        let params = ComputeBuffer::from_slice("FogVignette/Params", BufferType::Uniform, BufferUsage::UNIFORM, &[0f32; 12]);
        Self { color, near, far, frame: 0, pass, params }
    }
}

impl PostProcessingEffect for FogVignetteEffect {
    fn initialize(&mut self, device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {
        self.pass.initialize(device);
    }

    fn render(
        &mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder,
        _gbuffer: &GBuffer, input: &wgpu::TextureView, depth: &wgpu::TextureView, output: &wgpu::TextureView,
        camera: &Camera, width: u32, height: u32,
    ) {
        self.frame = (self.frame + 1) % 1024;
        let [r, g, b] = self.color;
        self.params.write(&[r, g, b, 1.0, self.near, self.far, camera.near, camera.far, width as f32, height as f32, self.frame as f32, 0.0]);
        Bindable::ensure_ready(&mut self.params, device, queue);
        let Some(params) = Bindable::binding_resource(&self.params) else { return };
        self.pass.set_bind_group(device, &[
            (0, BindingResource::TextureView(input)),
            (1, BindingResource::TextureView(depth)),
            (2, BindingResource::StorageTexture(output)),
            (3, params),
        ]);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FogVignette"), timestamp_writes: None });
        self.pass.dispatch(&mut pass, width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}
    fn destroy(&mut self) {}
    fn as_any(&self) -> &dyn std::any::Any { self }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any { self }
}

/// Surfaces lit by the spot light and a flat ambient of 0.15, the original's: three.js's legacy
/// lights give albedo x N·L x intensity, which the spot light matches at the carpet's middle. Its
/// cubes are Lambert and its ball Phong without highlights (roughness 1); its floor is Phong with
/// a broad dull highlight (shininess 30).
fn surface(label: &str, base_color: [f32; 3], roughness: f32, instanced: bool) -> Material {
    Material::standard_lit(label, &StandardLitOptions {
        base_color,
        roughness,
        sky_up: [0.15; 3],
        sky_down: [0.15; 3],
        instancing: instanced.then_some(StandardInstancing::OffsetScale),
        ..Default::default()
    })
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let n = param_or("n", 128usize).clamp(8, 512);
    let count = n * n;
    let row = PALETTE[param_or("palette", (js_sys::Math::random() * PALETTE.len() as f64) as usize).min(PALETTE.len() - 1)];
    let shadows = flag("shadows", true);

    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    if shadows {
        renderer.enable_spot_shadows(2048, 1);
    }

    // ── the cubes ──
    // Each cube's place on the carpet (its home), and its position now: xyz, w its scale.
    let half = n as f32 * SPACING * 0.5;
    let home: Vec<[f32; 4]> = (0..count).map(|i| [(i % n) as f32 * SPACING - half, 0.0, (i / n) as f32 * SPACING - half, 1.0]).collect();
    let mut homes = ComputeBuffer::from_slice("Homes", BufferType::ReadOnlyStorage, BufferUsage::STORAGE, &home);
    // the compute shader's output and the cubes' instance attribute: one buffer, shared by clones
    let positions = ComputeBuffer::from_slice("Cubes", BufferType::Storage, BufferUsage::STORAGE | BufferUsage::VERTEX, &home).with_vertex_vec4(3);
    // ball xyz and the steps to run (w); the cube count
    let mut params = ComputeBuffer::from_slice("Params", BufferType::Uniform, BufferUsage::UNIFORM, &[0f32; 8]);
    let mut step = ComputePass::new("CarpetStep", STEP_WGSL, vec![
        Binding::uniform(0, ShaderStages::COMPUTE),
        Binding::storage(1, ShaderStages::COMPUTE, false),
        Binding::storage(2, ShaderStages::COMPUTE, true),
    ]);
    {
        let (device, queue) = (renderer.device(), renderer.queue());
        let mut positions = positions.clone();
        for buffer in [&mut params, &mut positions, &mut homes] {
            Bindable::ensure_ready(buffer, device, queue);
        }
        step.initialize(device);
        step.set_bind_group(device, &[
            (0, Bindable::binding_resource(&params).expect("uploaded")),
            (1, Bindable::binding_resource(&positions).expect("uploaded")),
            (2, Bindable::binding_resource(&homes).expect("uploaded")),
        ]);
    }

    let mut scene = Scene::new();
    let cube = BoxGeometry::new(SPACING, SPACING, SPACING);
    scene.add(SceneNode::Renderable(Renderable::new(InstancedGeometry::new(cube, count as u32, vec![positions]), surface("Cubes", rgb(row[2], 1.3), 1.0, true))));

    let mut ball = Renderable::new(SphereGeometry::new(BALL_RADIUS, 32, 16), surface("Ball", rgb(0xff553f, 1.0), 1.0, false));
    ball.object.set_position(BALL_START[0], BALL_START[1], BALL_START[2]);
    let ball = scene.add(SceneNode::Renderable(ball));

    let mut floor = Renderable::new(PlaneGeometry::new(500.0, 500.0), surface("Floor", rgb(row[0], 0.3), 0.65, false));
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.object.set_position(0.0, -5.0, 0.0);
    floor.cast_shadow = false;
    scene.add(SceneNode::Renderable(floor));

    // the spot light, on the original's ray to the carpet's middle but twice as far, since the
    // original's light doesn't fall off with distance and Kansei's falls off with its square;
    // its intensity gives an illuminance of π at the middle (three.js's unit light)
    const LIGHT_DISTANCE: f32 = 2.0;
    let light_position = Vec3::new(15.0, 15.0, 0.01) * LIGHT_DISTANCE;
    let mut spot = SpotLight::new(light_position, Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 1.0, 1.0),
        std::f32::consts::PI * (15.0 * 15.0 * 2.0) * LIGHT_DISTANCE * LIGHT_DISTANCE, 300.0, 25f32.to_radians(), 30f32.to_radians());
    spot.look_at(Vec3::ZERO);
    spot.cast_shadow = shadows;
    spot.source_radius = 0.15;
    scene.add(SceneNode::Light(Light::Spot(spot)));

    // ── the picture: ACES-fitted at the original's exposure (three.js divides it by 0.6), not
    // sRGB-encoded, then the fog and the vignette ──
    let mut tonemap = ToneMapOptions::for_surface(renderer.presentation_format());
    tonemap.exposure = 1.5 / 0.6;
    tonemap.encode_srgb = false;
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![
        Box::new(ToneMapEffect::new(tonemap)),
        Box::new(FogVignetteEffect::new(rgb(row[0], 0.5), 12.0, 40.0)),
    ];
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    // the original's near and far planes (1 mm, 100 m): at the carpet's distance its depth
    // resolves about a fifth of a cube, so where the carpet is squeezed and its cubes overlap,
    // their faces fight along the edges: the carpet's woven grain
    let mut camera = Camera::new(60.0, 0.001, 100.0, canvas.aspect());
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::ZERO, 18.0);
    controls.set_azimuth(0.3422115 * std::f32::consts::TAU);
    controls.set_elevation(0.1249 * std::f32::consts::TAU);
    log::info!("Kansei — Instancing (WASM): {count} cubes");

    let mut fixed = FixedStep::new(STEP);
    let mut sim_time = 0.0f64;
    let mut started = false;
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let steps = fixed.advance(frame.dt as f64);
        if steps > 0 {
            // the ball where it was at the start of these steps, as the original pushes the
            // cubes with last frame's ball
            let [x, y, z] = if started { ball_at(sim_time as f32) } else { BALL_START };
            params.write(&[x, y, z, steps as f32, count as f32, 0.0, 0.0, 0.0]);
            Bindable::ensure_ready(&mut params, renderer.device(), renderer.queue());
            renderer.compute(&step, (count as u32).div_ceil(64), 1, 1);
            sim_time += steps as f64 * STEP;
            started = true;
            if let Some(r) = scene.get_renderable_mut(ball) {
                let [x, y, z] = ball_at(sim_time as f32);
                r.object.set_position(x, y, z);
            }
        }
        controls.update(&mut camera, frame.dt);
        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    });
    Ok(())
}
