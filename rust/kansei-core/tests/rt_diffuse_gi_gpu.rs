//! Ray-traced diffuse GI on a real GPU: a white floor beside a red wall, both lit by the sun.
//! The floor's indirect light near the wall is red, and redder than far from it; the hybrid with
//! its hit cone off converges to the one-bounce reference path tracer; the cone adds the voxels'
//! further bounces; SVGF at half resolution keeps the converged signal's energy. Skipped (passes)
//! when no adapter is available.

use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, VoxelGiQuality};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiMode, RtGiResolution, RtGiView, RtGridOptions, RtSurface, SceneRtGridOptions};

const W: u32 = 160;
const H: u32 = 120;

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

/// An unlit surface writing its colour, normal and albedo.
fn material(albedo: [f32; 3]) -> Material {
    let code = format!(
        r#"{GBUFFER_OUT_WGSL}
struct Surface {{ color: vec4<f32> }};
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut {{ @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) world: vec3<f32> }};
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {{
    var out: VOut;
    let world = world_matrix * position;
    out.clip = projection_matrix * view_matrix * world;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.world = world.xyz;
    return out;
}}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {{
    // the colour (which the GI's signal view leaves out) tells the test where the pixel is: the
    // world x, the normal's y, and a 1 on every surface
    let n = normalize(in.normal);
    return kansei_gbuffer_out(vec3<f32>(in.world.x, n.y, 1.0), vec3<f32>(0.0), n, surface.color.rgb);
}}
"#
    );
    let mut m = Material::new("Surface", &code, vec![Binding::uniform(0, ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    m.set_uniform_bindable(0, "Surface", &[albedo[0], albedo[1], albedo[2], 1.0]);
    m
}

fn camera() -> Camera {
    let mut camera = Camera::new(55.0, 0.1, 50.0, W as f32 / H as f32);
    camera.set_position(1.6, 1.4, 2.4);
    camera.look_at(&Vec3::new(-0.4, 0.1, -0.2));
    camera.update_projection_matrix();
    camera
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
    renderer.device().poll(wgpu::Maintain::Wait);
    let bytes = buffer.slice(..).get_mapped_range();
    let mut out = Vec::with_capacity((W * H) as usize);
    for y in 0..H {
        for x in 0..W {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// The mean of `image` over the pixels `mask` keeps.
fn mean(image: &[[f32; 4]], mask: &[bool]) -> [f32; 3] {
    let n = mask.iter().filter(|m| **m).count().max(1) as f32;
    let mut sum = [0.0f32; 3];
    for (c, _) in image.iter().zip(mask).filter(|(_, m)| **m) {
        for k in 0..3 {
            sum[k] += c[k];
        }
    }
    sum.map(|s| s / n)
}

fn luminance(c: [f32; 3]) -> f32 {
    0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
}

#[test]
fn the_floor_beside_a_red_wall_is_lit_red_and_the_hybrid_converges() {
    let Some(mut renderer) = renderer() else { return eprintln!("no GPU adapter: skipping") };
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: [-2.0, -0.5, -2.0], bounds_max: [2.0, 1.5, 2.0], ..Default::default() });
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 4.0 / 64.0, fixed_origin: Some(glam::Vec3::new(-2.0, -0.5, -2.0)), ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    let white = [0.8, 0.8, 0.8];
    let red = [0.8, 0.05, 0.05];
    let mut floor = Renderable::new(BoxGeometry::new(3.6, 0.2, 3.6), material(white)).with_gi(GiSurface::new(white)).with_rt(RtSurface::new(white));
    floor.object.set_position(0.0, -0.1, 0.0);
    scene.add(SceneNode::Renderable(floor));
    // the wall's lit face at x = -0.9
    let mut wall = Renderable::new(BoxGeometry::new(0.2, 1.4, 3.0), material(red)).with_gi(GiSurface::new(red)).with_rt(RtSurface::new(red));
    wall.object.set_position(-1.0, 0.7, 0.0);
    scene.add(SceneNode::Renderable(wall));
    // the sun from above and +x: the floor and the wall's +x face lit
    let sun = scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(-0.6, -1.0, -0.2), Vec3::new(1.0, 1.0, 1.0), 3.0))));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = camera();
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
    let handle = renderer.rt_grid().unwrap().handle();
    let effect = |options: RtDiffuseGiOptions, renderer: &Renderer, scene: &Scene| {
        let mut e = RtDiffuseGiEffect::with_volume(renderer.voxel_gi().unwrap().volume(), handle.clone(), RtDiffuseGiOptions { max_distance: 10.0, ..options });
        e.update_lights(scene.lights());
        e.view = RtGiView::Signal;
        e
    };
    let mut run = |renderer: &mut Renderer, scene: &mut Scene, effect: &mut RtDiffuseGiEffect, frames: usize| {
        for _ in 0..frames {
            renderer.render_scene_offscreen(scene, &mut camera, &gbuffer);
            let mut encoder = renderer.device().create_command_encoder(&Default::default());
            effect.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &output_view, &camera, W, H);
            renderer.queue().submit(Some(encoder.finish()));
        }
        read_rgba16(renderer, &output)
    };
    // let the voxels take the sun's light (the hit cone and the far field read them)
    let mut warm = effect(RtDiffuseGiOptions::default(), &renderer, &scene);
    run(&mut renderer, &mut scene, &mut warm, 8);
    // the floor's pixels (normal up), near the wall's lit face and far from it
    let colour = read_rgba16(&renderer, &gbuffer.color_texture);
    let floor_at = |i: usize, lo: f32, hi: f32| colour[i][1] > 0.9 && (lo..hi).contains(&colour[i][0]);
    let near: Vec<bool> = (0..colour.len()).map(|i| floor_at(i, -0.85, -0.5)).collect();
    let far: Vec<bool> = (0..colour.len()).map(|i| floor_at(i, 0.6, 1.4)).collect();
    let surface: Vec<bool> = colour.iter().map(|c| c[2] > 0.5).collect();
    let (n_near, n_far) = (near.iter().filter(|m| **m).count(), far.iter().filter(|m| **m).count());
    assert!(n_near > 300 && n_far > 300, "the view frames the floor near the wall and away from it ({n_near}, {n_far} pixels)");

    // the converged one-bounce hybrid (no hit cone) against the reference path tracer at one vertex
    let mut converged = |mode: RtGiMode, cone_steps: u32, renderer: &mut Renderer, scene: &mut Scene| {
        let mut e = effect(RtDiffuseGiOptions { resolution: RtGiResolution::Full, denoise: RtGiDenoise::Off, hit_cone_steps: cone_steps, reference_bounces: 1, ..Default::default() }, renderer, scene);
        e.mode = mode;
        e.accumulate = true;
        let image = run(renderer, scene, &mut e, 96);
        assert_eq!(e.accumulated(), 96);
        image
    };
    let hybrid = converged(RtGiMode::Hybrid, 0, &mut renderer, &mut scene);
    let reference = converged(RtGiMode::Reference, 0, &mut renderer, &mut scene);
    let (h_near, h_far) = (mean(&hybrid, &near), mean(&hybrid, &far));
    eprintln!("one bounce: near the wall {h_near:?}, far from it {h_far:?}; reference near {:?}", mean(&reference, &near));
    assert!(h_near[0] > 4.0 * h_near[1].max(h_near[2]), "the wall's light on the floor is red: {h_near:?}");
    assert!(h_near[0] > 2.0 * h_far[0], "redder near the wall than far from it: {h_near:?} vs {h_far:?}");
    for (h, r) in [(mean(&hybrid, &surface), mean(&reference, &surface)), (h_near, mean(&reference, &near))] {
        let (lh, lr) = (luminance(h), luminance(r));
        assert!((lh - lr).abs() <= 0.02 * lr, "the one-bounce hybrid converges to the reference: {h:?} vs {r:?}");
    }

    // the hit cone adds the further bounces the voxels hold
    let two = converged(RtGiMode::Hybrid, 16, &mut renderer, &mut scene);
    let (one, more) = (luminance(mean(&hybrid, &surface)), luminance(mean(&two, &surface)));
    eprintln!("surface signal: one bounce {one}, with the hit cone {more}");
    assert!(more > 1.03 * one, "the hit cone adds light: {more} vs {one}");

    // SVGF at half resolution keeps the converged signal's energy
    let mut svgf = effect(RtDiffuseGiOptions::default(), &renderer, &scene);
    let denoised = run(&mut renderer, &mut scene, &mut svgf, 48);
    let (ld, lc) = (luminance(mean(&denoised, &surface)), luminance(mean(&two, &surface)));
    eprintln!("SVGF half resolution {ld} against converged {lc}");
    assert!((ld - lc).abs() <= 0.1 * lc, "SVGF keeps the energy within 10%: {ld} vs {lc}");
    assert!(svgf.memory_bytes() > 0 && svgf.memory_bytes() < (W as u64 * H as u64) * 40, "half resolution targets: {} bytes", svgf.memory_bytes());

    // a sun 100 000 times brighter (an outdoor scene's luminance, thousands, with no
    // pre-exposure): SVGF denoises as it did. (Its luminance's second moment and variance in f16
    // overflowed there, which turned its luminance stop off: it kept more energy than at low
    // luminance, and blurred across the light's edges.)
    if let Some(Light::Directional(l)) = scene.get_light_mut(sun) {
        l.intensity *= 1.0e5;
    }
    let mut warm = effect(RtDiffuseGiOptions::default(), &renderer, &scene);
    run(&mut renderer, &mut scene, &mut warm, 8);
    let mut svgf = effect(RtDiffuseGiOptions::default(), &renderer, &scene);
    let bright = run(&mut renderer, &mut scene, &mut svgf, 48);
    let mut reference = effect(RtDiffuseGiOptions { resolution: RtGiResolution::Full, denoise: RtGiDenoise::Off, ..Default::default() }, &renderer, &scene);
    reference.accumulate = true;
    let converged = run(&mut renderer, &mut scene, &mut reference, 96);
    let finite = bright.iter().zip(&surface).filter(|(_, m)| **m).all(|(c, _)| c.iter().all(|v| v.is_finite()));
    let (lb, lc) = (luminance(mean(&bright, &surface)), luminance(mean(&converged, &surface)));
    eprintln!("bright sun: SVGF {lb} against converged {lc}");
    assert!(finite, "SVGF's output stays finite at high luminance");
    assert!(lc > 1000.0, "the bright sun's signal is in the thousands: {lc}");
    let (low, high) = (ld / luminance(mean(&two, &surface)), lb / lc);
    assert!((high - low).abs() < 0.015, "SVGF keeps the same share of the energy at any luminance: {high} vs {low}");
}
