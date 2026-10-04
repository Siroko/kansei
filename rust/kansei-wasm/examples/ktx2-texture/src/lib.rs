//! KTX2 textures: an ETC1S colour texture, a UASTC colour texture with alpha and a UASTC normal
//! map, each transcoded to the best format this device samples. The page lists what each became
//! and its GPU memory; `?support=none|bc|astc|etc2` restricts the formats to show the fallbacks.

use wasm_bindgen::prelude::*;

use kansei_core::buffers::Sampler;
use kansei_core::cameras::Camera;
use kansei_core::geometries::PlaneGeometry;
use kansei_core::loaders::ktx2::{self, CompressionSupport, Ktx2Options, TranscodedTexture};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{fetch_bytes, param, Canvas};

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

/// Back far enough that the three quads (3 units either side) fit the width.
fn fit_camera(camera: &mut Camera) {
    camera.set_position(0.0, 0.0, (3.2 / (22.5f32.to_radians().tan() * camera.aspect)).max(5.0));
    camera.look_at(&Vec3::ZERO);
    camera.update_projection_matrix();
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
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas
        .renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.03, 0.03, 0.04, 1.0), ..Default::default() })
        .await;

    let device_support = renderer.compression_support();
    let support = match param("support").as_deref() {
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

    let mut camera = Camera::new(45.0, 0.1, 100.0, canvas.aspect());
    fit_camera(&mut camera);

    kansei_wasm::run(&canvas, move |frame| {
        if frame.resized.is_some() {
            frame.resize(&mut renderer, &mut camera);
            fit_camera(&mut camera);
        }
        if let Some(buf) = scene.get_renderable(normal_quad).and_then(|r| r.material.bindable_buffer(2)) {
            renderer.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&[1.0f32, frame.time as f32, 0.0, 0.0]));
        }
        renderer.render(&mut scene, &mut camera);
    });
    Ok(report.join("\n"))
}
