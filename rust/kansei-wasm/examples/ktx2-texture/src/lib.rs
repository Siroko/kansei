//! KTX2 textures: an ETC1S colour texture, a UASTC colour texture with alpha and a UASTC normal
//! map, each transcoded to the best format this device samples. The page lists what each became
//! and its GPU memory; `?support=none|bc|astc|etc2` restricts the formats to show the fallbacks.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::Sampler;
use kansei_core::cameras::Camera;
use kansei_core::geometries::PlaneGeometry;
use kansei_core::loaders::ktx2::{self, CompressionSupport, Ktx2Options, TranscodedTexture};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::{Renderer, RendererConfig};

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
}

/// A textured quad: colour over a checkerboard (so alpha shows), or a normal map lit by a light
/// circling in front of it (`params.x` = 1).
const QUAD_WGSL: &str = r#"
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
@group(0) @binding(2) var<uniform> params: vec4<f32>; // x: normal map, y: time
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) uv: vec2<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.uv = uv;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    let texel = textureSample(tex, tex_sampler, in.uv);
    if (params.x > 0.5) {
        // tangent space = the quad's: +x right, +y up, +z towards the viewer
        let n = normalize(texel.xyz * 2.0 - 1.0);
        let t = params.y;
        let light = normalize(vec3<f32>(cos(t), sin(t), 0.8));
        let lit = max(dot(n, light), 0.0);
        return vec4<f32>(vec3<f32>(0.9, 0.85, 0.8) * (0.08 + lit), 1.0);
    }
    let cell = floor(in.uv * 16.0);
    let checker = select(0.2, 0.35, (cell.x + cell.y) % 2.0 == 0.0);
    return vec4<f32>(mix(vec3<f32>(checker), texel.rgb, texel.a), 1.0);
}
"#;

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    normal_quad: usize,
    start: f64,
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

fn query_param(name: &str) -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    search.trim_start_matches('?').split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == name).then(|| v.to_string())
    })
}

fn now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

async fn fetch_bytes(url: &str) -> Result<Vec<u8>, JsValue> {
    let window = web_sys::window().unwrap();
    let resp: web_sys::Response = wasm_bindgen_futures::JsFuture::from(window.fetch_with_str(url)).await?.dyn_into()?;
    if !resp.ok() {
        return Err(format!("{url}: HTTP {}", resp.status()).into());
    }
    let buf = wasm_bindgen_futures::JsFuture::from(resp.array_buffer()?).await?;
    Ok(js_sys::Uint8Array::new(&buf).to_vec())
}

fn quad(label: &str, texture: TranscodedTexture, normal_map: bool, x: f32) -> Renderable {
    let mut material = Material::new(
        label,
        QUAD_WGSL,
        vec![
            Binding::texture_2d(0, ShaderStages::FRAGMENT),
            Binding::sampler(1, ShaderStages::FRAGMENT),
            Binding::uniform(2, ShaderStages::FRAGMENT),
        ],
        MaterialOptions::default(),
    );
    material.set_bindable(0, texture.into_texture());
    material.set_bindable(1, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
    material.set_uniform_bindable(2, &format!("{label}/Params"), &[if normal_map { 1.0f32 } else { 0.0 }, 0.0, 0.0, 0.0]);
    let mut r = Renderable::new(PlaneGeometry::new(1.8, 1.8), material);
    r.object.set_position(x, 0.0, 0.0);
    r
}

/// Starts the page; resolves to the report lines (one per texture, plus the device's formats).
#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<String, JsValue> {
    let window = web_sys::window().unwrap();
    let canvas = window
        .document()
        .unwrap()
        .get_element_by_id(canvas_id)
        .ok_or("Canvas not found")?
        .dyn_into::<web_sys::HtmlCanvasElement>()?;
    let (width, height) = (canvas.client_width() as u32, canvas.client_height() as u32);
    canvas.set_width(width);
    canvas.set_height(height);

    let mut renderer = Renderer::new(RendererConfig {
        width,
        height,
        sample_count: 1,
        clear_color: Vec4::new(0.03, 0.03, 0.04, 1.0),
        ..Default::default()
    });
    renderer.initialize_with_canvas(canvas.clone()).await;

    let device_support = renderer.compression_support();
    let support = match query_param("support").as_deref() {
        Some("none") => CompressionSupport::NONE,
        Some("bc") => CompressionSupport { bc: device_support.bc, ..CompressionSupport::NONE },
        Some("astc") => CompressionSupport { astc: device_support.astc, ..CompressionSupport::NONE },
        Some("etc2") => CompressionSupport { etc2: device_support.etc2, ..CompressionSupport::NONE },
        _ => device_support,
    };
    let mut report = vec![format!(
        "device: texture-compression-bc {} · -astc {} · -etc2 {}{}",
        device_support.bc,
        device_support.astc,
        device_support.etc2,
        if support != device_support { format!("  (restricted to {support:?})") } else { String::new() },
    )];

    let textures = [
        ("assets/card_etc1s.ktx2", "ETC1S colour", Ktx2Options::color(), false),
        ("assets/badge_uastc.ktx2", "UASTC colour + alpha", Ktx2Options::color(), false),
        ("assets/bumps_normal.ktx2", "UASTC normal map", Ktx2Options::linear(), true),
    ];
    let mut scene = Scene::new();
    let mut normal_quad = 0;
    for (i, (url, label, options, normal_map)) in textures.into_iter().enumerate() {
        let bytes = fetch_bytes(url).await?;
        let texture = ktx2::transcode(label, &bytes, &options, support).map_err(|e| JsValue::from_str(&e.to_string()))?;
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        let line = format!(
            "{label}: {} {}x{}, {} mips, {:.0} KB file -> {:?} ({}), {:.2} MiB on the GPU vs {:.2} MiB as RGBA8",
            texture.codec,
            texture.width,
            texture.height,
            texture.levels.len(),
            bytes.len() as f64 / 1024.0,
            texture.format,
            texture.target.name(),
            mib(texture.gpu_bytes()),
            mib(texture.uncompressed_bytes()),
        );
        log::info!("{}", texture.summary());
        report.push(line);
        let index = scene.add(SceneNode::Renderable(quad(label, texture, normal_map, (i as f32 - 1.0) * 2.0)));
        if normal_map {
            normal_quad = index;
        }
    }

    // back far enough that the three quads (3 units either side) fit the width
    let aspect = width as f32 / height as f32;
    let mut camera = Camera::new(45.0, 0.1, 100.0, aspect);
    camera.set_position(0.0, 0.0, (3.2 / (22.5f32.to_radians().tan() * aspect)).max(5.0));
    camera.look_at(&Vec3::ZERO);
    camera.update_projection_matrix();

    let state = Rc::new(RefCell::new(State { renderer, scene, camera, normal_quad, start: now() }));
    let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
    let g = f.clone();
    *g.borrow_mut() = Some(Closure::new(move || {
        {
            let mut st = state.borrow_mut();
            let State { ref mut renderer, ref mut scene, ref mut camera, normal_quad, start } = *st;
            let t = ((now() - start) / 1000.0) as f32;
            if let Some(buf) = scene.get_renderable(normal_quad).and_then(|r| r.material.bindable_buffer(2)) {
                renderer.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&[1.0f32, t, 0.0, 0.0]));
            }
            renderer.render(scene, camera);
        }
        request_animation_frame(f.borrow().as_ref().unwrap());
    }));
    request_animation_frame(g.borrow().as_ref().unwrap());
    Ok(report.join("\n"))
}
