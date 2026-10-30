//! The specular factor (`StandardLitOptions::specular`, `kansei_gbuffer_with_specular`) on a real
//! GPU: a floor seen at a low angle under a low sun ahead of the camera, where a dielectric's
//! Fresnel shows most.
//!
//! - `RtShadowsEffect`: a surface passed through `kansei_gbuffer_with_specular` at 1 lights
//!   exactly as one without it (today's GGX at F0 0.04); at 0 every pixel is the sun's Lambert
//!   light alone, with no highlight or sheen at any angle.
//! - `Material::standard_lit`: the default is the BRDF it had before the factor (a material
//!   carrying that BRDF's old code, drawn beside it); `StandardLitOptions::matte` is Lambert alone.
//!
//! Skipped (passes) when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::lights::{DirectionalLight, Light, LIGHTS_WGSL};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtGridOptions, RtShadowsEffect, RtShadowsView, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

const W: u32 = 160;
const H: u32 = 120;
const ALBEDO: f32 = 0.8;
const ROUGHNESS: f32 = 0.5;
/// The sun's illuminance (lux), and the way its light travels: low, toward the camera.
const SUN: f32 = 2.0;
const SUN_DIR: [f32; 3] = [0.0, -0.35, 1.0];

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let info = adapter.get_info();
    eprintln!("{} ({:?})", info.name, info.backend);
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

fn camera() -> Camera {
    let mut camera = Camera::new(50.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(0.0, 0.9, 3.0);
    camera.look_at(&Vec3::new(0.0, 0.0, -0.5));
    camera.update_projection_matrix();
    camera
}

const VERTEX: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
"#;

/// A surface `RtShadowsEffect` lights: through `kansei_gbuffer_with_specular` at `specular`, or
/// as it was before the factor (`None`).
fn rt_material(specular: Option<f32>) -> Material {
    let out = format!("kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), normalize(in.normal), vec3<f32>({ALBEDO:?}), {ROUGHNESS:?}, 0.04, false)");
    let out = match specular {
        Some(s) => format!("kansei_gbuffer_with_specular({out}, {s:?})"),
        None => out,
    };
    let code = format!("{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return {out}; }}\n");
    Material::new("RtLit", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

/// The standard material's BRDF as it was before the specular factor, for its directional lights.
fn old_standard_material() -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
{LIGHTS_WGSL}
{VERTEX}
struct Surface {{ base: vec4<f32>, params: vec4<f32> }};
@group(0) @binding(0) var<uniform> surface: Surface;
const PI: f32 = 3.14159265;
fn old_brdf(N: vec3f, V: vec3f, L: vec3f, baseColor: vec3f, roughness: f32, metallic: f32) -> vec3f {{
    let NdotL = saturate(dot(N, L));
    if (NdotL <= 0.0) {{ return vec3f(0.0); }}
    let H = normalize(V + L);
    let NdotV = max(dot(N, V), 1e-4);
    let NdotH = saturate(dot(N, H));
    let VdotH = saturate(dot(V, H));
    let a = max(roughness * roughness, 2e-3);
    let a2 = a * a;
    let dd = NdotH * NdotH * (a2 - 1.0) + 1.0;
    let D = a2 / (PI * dd * dd);
    let vis = 0.5 / (NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2) + NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2));
    let F0 = mix(vec3f(0.04), baseColor, metallic);
    let F = F0 + (1.0 - F0) * pow(1.0 - VdotH, 5.0);
    let diffuse = (1.0 - metallic) * (1.0 - F) * baseColor / PI;
    return (diffuse + D * vis * F) * NdotL;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    let N = normalize(in.normal);
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let V = normalize(eye - in.world);
    var radiance = vec3f(0.0);
    for (var i = 0u; i < kansei_lights.num_directional; i++) {{
        let light = kansei_lights.directional[i];
        radiance += old_brdf(N, V, -normalize(light.direction), surface.base.rgb, surface.params.x, surface.params.y) * light.color;
    }}
    return kansei_gbuffer_out(radiance, vec3f(0.0), N, surface.base.rgb);
}}
"#
    );
    let mut m = Material::new("OldStandard", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "OldStandard", &[ALBEDO, ALBEDO, ALBEDO, 1.0, ROUGHNESS, 0.0, 0.0, 0.0]);
    m
}

fn standard(specular: f32) -> Material {
    Material::standard_lit("Standard", &StandardLitOptions { base_color: [ALBEDO; 3], roughness: ROUGHNESS, specular, ..Default::default() })
}

/// A floor of `material` under the sun, in the grid.
fn scene(material: Material) -> Scene {
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(6.0, 0.2, 6.0), material).with_rt(RtSurface::new([ALBEDO; 3]));
    floor.object.set_position(0.0, -0.1, -1.0);
    scene.add(SceneNode::Renderable(floor));
    let [x, y, z] = SUN_DIR;
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(x, y, z), Vec3::new(1.0, 1.0, 1.0), SUN))));
    scene
}

fn f16_value(h: u16) -> f32 {
    let exponent = ((h >> 10) & 0x1f) as i32;
    let mantissa = (h & 0x3ff) as f32;
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    match exponent {
        0 => sign * mantissa * 2f32.powi(-24),
        _ => sign * (1.0 + mantissa / 1024.0) * 2f32.powi(exponent - 15),
    }
}

fn read_rgba16(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 4]> {
    let row = (W * 8).next_multiple_of(256);
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * H) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(H) } },
        texture.size(),
    );
    renderer.queue().submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let bytes = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// The sun's Lambert light on the floor (its normal straight up): albedo / pi x E x cos.
fn lambert() -> f32 {
    let [x, y, z] = SUN_DIR;
    ALBEDO / std::f32::consts::PI * SUN * (-y / (x * x + y * y + z * z).sqrt())
}

/// The floor's pixels (lit at all: the sky reads 0).
fn floor_pixels(image: &[[f32; 4]]) -> Vec<usize> {
    (0..image.len()).filter(|&i| image[i][0] > 1e-3).collect()
}

#[test]
fn rt_shadows_light_a_matte_surface_with_lambert_alone_and_the_default_as_before() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 16, 64], cell: 7.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-3.5, -0.5, -4.5)), ..Default::default() }, ..Default::default() });
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let output = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let output_view = output.create_view(&Default::default());
    let mut camera = camera();
    // exact: every pixel traced, a sun of a tenth of a degree, no wavelet, no contact rays; the
    // direct light alone
    let mut lit = |renderer: &mut Renderer, specular: Option<f32>| {
        let mut scene = scene(rt_material(specular));
        let mut effect = RtShadowsEffect::new(renderer.rt_grid().unwrap().handle());
        effect.half_resolution = false;
        effect.iterations = 0;
        effect.contact_length = 0.0;
        effect.sun_angle = 0.002;
        effect.view = RtShadowsView::Lit;
        effect.update_lights(scene.lights());
        for _ in 0..3 {
            renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        read_rgba16(renderer, &output)
    };
    let before = lit(&mut renderer, None);
    let one = lit(&mut renderer, Some(1.0));
    let matte = lit(&mut renderer, Some(0.0));
    let floor = floor_pixels(&before);
    assert!(floor.len() > (W * H / 2) as usize, "the floor fills the view ({} pixels)", floor.len());

    // a factor of 1 is the surface as it was, to the bit
    assert!(before == one, "kansei_gbuffer_with_specular at 1 changes nothing");

    // at 0, the sun's Lambert light alone at every pixel
    let lambert = lambert();
    let worst = floor.iter().map(|&i| (matte[i][0] / lambert - 1.0).abs()).fold(0.0, f32::max);
    eprintln!("matte: {} floor pixels, the most any is off Lambert ({lambert:.4}): {:.3}%", floor.len(), worst * 100.0);
    assert!(worst < 2e-3, "a matte surface takes Lambert alone (off by {:.3}%)", worst * 100.0);

    // the default's GGX: its sheen toward the low sun, where the matte one has none
    let sheen = floor.iter().map(|&i| before[i][0] / lambert - 1.0).fold(0.0, f32::max);
    let shining = floor.iter().filter(|&&i| before[i][0] > lambert * 1.1).count();
    eprintln!("default: its brightest pixel {:.0}% over Lambert, {shining} pixels over 10%", sheen * 100.0);
    assert!(sheen > 0.5 && shining > 200, "the default keeps its specular ({:.0}% at most, {shining} pixels over 10%)", sheen * 100.0);
}

#[test]
fn the_standard_material_keeps_its_brdf_by_default_and_is_lambert_when_matte() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
    let mut draw = |renderer: &mut Renderer, material: Material| {
        let mut scene = scene(material);
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        read_rgba16(renderer, &gbuffer.color_texture)
    };
    let old = draw(&mut renderer, old_standard_material());
    let default = draw(&mut renderer, standard(1.0));
    let matte = draw(&mut renderer, standard(0.0));
    let floor = floor_pixels(&old);
    assert!(floor.len() > (W * H / 2) as usize, "the floor fills the view ({} pixels)", floor.len());

    // the default against the BRDF's old code: the same to f16's precision
    let worst = floor.iter().map(|&i| ((default[i][0] - old[i][0]) / old[i][0]).abs()).fold(0.0, f32::max);
    eprintln!("default: the most a pixel is off the old BRDF: {:.4}%", worst * 100.0);
    assert!(worst < 2e-3, "the default standard material is the old one (off by {:.4}%)", worst * 100.0);

    let lambert = lambert();
    let worst = floor.iter().map(|&i| (matte[i][0] / lambert - 1.0).abs()).fold(0.0, f32::max);
    eprintln!("matte: the most a pixel is off Lambert ({lambert:.4}): {:.3}%", worst * 100.0);
    assert!(worst < 2e-3, "a matte standard material is Lambert alone (off by {:.3}%)", worst * 100.0);
    let sheen = floor.iter().map(|&i| old[i][0] / lambert - 1.0).fold(0.0, f32::max);
    assert!(sheen > 0.5, "the default shows a sheen to remove ({:.0}% over Lambert at most)", sheen * 100.0);
}
