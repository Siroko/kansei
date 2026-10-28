//! Ray-traced caustics on a real GPU: a 3 cm glass marble resting on a floor whose material leaves
//! its direct light to `RtShadowsEffect` (whose shadow rays pass through glass), under a spot
//! light. The caustics' tile holds the change the glass makes to that direct light, so the light
//! over the tile with the glass is the light without it plus the tile's sum:
//!
//! - glass of index 1 (it bends nothing, absorbs nothing) changes nothing: every texel's change is
//!   a hair from zero, the light with the glass the light without it;
//! - clear glass of index 1.5 moves light (a ball lens on the floor: a spot many times the direct
//!   light under it, a dark ring round it) but loses no more than its two surfaces reflect away
//!   from the floor (about a tenth each way through a ball);
//! - a tinted core takes its colour from the light through it;
//! - a rolling marble's tile, cleared and fed one frame of photons each frame, is already smooth,
//!   and moving the marble a texel or a fraction of one moves its caustics without changing them
//!   (no crawling).
//!
//! Under a spot light straight above and under a slanting sun. Through the grid's cells and,
//! where the adapter has ray queries, through hardware ray queries (`RtBackend::Hardware`).
//! Skipped (passes) when no adapter is available.

use glam::Vec3 as GVec3;
use kansei_core::cameras::Camera;
use kansei_core::geometries::{BoxGeometry, IcosphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, SpotLight};
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtBackend, RtCausticsEffect, RtCausticsOptions, RtGridOptions, RtShadowsEffect, RtSurface, SceneRtGridOptions, RT_SHADOWS_GBUFFER_WGSL};

const W: u32 = 320;
const H: u32 = 240;
const RADIUS: f32 = 0.015;
/// The spot light straight above the marble: its height (m) and intensity (cd).
const LAMP: f32 = 0.5;
const CANDELA: f32 = 20.0;
/// The sun's illuminance (lux, facing it) and the way its light travels.
const SUN: f32 = 80.0;
const SUN_DIR: GVec3 = GVec3::new(0.25, -1.0, 0.1);

fn renderer() -> Option<Renderer> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).ok()?;
    let info = adapter.get_info();
    eprintln!("{} ({:?})", info.name, info.backend);
    let mut renderer = Renderer::new(RendererConfig { width: W, height: H, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    Some(renderer)
}

const VERTEX: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    return out;
}
"#;

fn glass_material() -> Material {
    let code = format!("{GBUFFER_OUT_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_glass(normalize(in.normal), vec3<f32>(1.0), 1.5, 0.0); }}\n");
    Material::new("Glass", &code, Vec::new(), MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
}

/// A surface lit by `RtShadowsEffect` alone.
fn floor_material() -> Material {
    let code = format!(
        "{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{VERTEX}\n@fragment fn fragment_main(in: VOut) -> KanseiGBufferOut {{ return kansei_gbuffer_out_rt_lit(vec3<f32>(0.0), vec3<f32>(0.0), normalize(in.normal), vec3<f32>(0.8), 1.0, 0.04, false); }}\n"
    );
    Material::new("RtLit", &code, vec![], MaterialOptions { mrt_output_count: Some(4), ..Default::default() })
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

/// The first `dims` texels of the map's layer `layer`.
fn read_layer(renderer: &Renderer, texture: &wgpu::Texture, layer: u32, dims: [u32; 2]) -> Vec<[f32; 4]> {
    let row = (dims[0] * 8).next_multiple_of(256);
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * dims[1]) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = renderer.device().create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y: 0, z: layer }, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(dims[1]) } },
        wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: 1 },
    );
    renderer.queue().submit(Some(encoder.finish()));
    buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    let _ = renderer.device().poll(wgpu::PollType::wait_indefinitely());
    let bytes = buffer.slice(..).get_mapped_range().unwrap();
    let mut out = Vec::new();
    for y in 0..dims[1] {
        for x in 0..dims[0] {
            let o = (y * row + x * 8) as usize;
            out.push(std::array::from_fn(|c| f16_value(u16::from_le_bytes([bytes[o + 2 * c], bytes[o + 2 * c + 1]]))));
        }
    }
    out
}

/// What a run of the caustics leaves in the marble's tile.
struct Tile {
    texel: f32,
    texels: Vec<[f32; 4]>,
    /// the history it holds
    history: u32,
    /// world xz of each texel's centre
    centres: Vec<[f32; 2]>,
}

/// The marble (index `ior`; `core`: a tinted sphere of half its radius inside it) on the floor at
/// x = `x`, `frames` frames of the shadows and the caustics.
fn run(backend: RtBackend, sun: bool, ior: f32, core: Option<[f32; 3]>, frames: usize, x: f32) -> Option<Tile> {
    let mut renderer = renderer()?;
    renderer.enable_rt_grid(SceneRtGridOptions { grid: RtGridOptions { dims: [64, 32, 64], cell: 0.4 / 64.0, fixed_origin: Some(glam::Vec3::new(-0.2, -0.05, -0.2)), backend, ..Default::default() }, ..Default::default() });
    let mut scene = Scene::new();
    let mut floor = Renderable::new(BoxGeometry::new(0.4, 0.02, 0.4), floor_material()).with_rt(RtSurface::new([0.8; 3]));
    floor.object.set_position(0.0, -0.01, 0.0);
    scene.add(SceneNode::Renderable(floor));
    let mut marble = Renderable::new(IcosphereGeometry::new(RADIUS, 5), glass_material()).with_rt(RtSurface::glass([1.0; 3]));
    marble.object.set_position(x, RADIUS, 0.0);
    scene.add(SceneNode::Renderable(marble));
    if let Some(tint) = core {
        let mut inner = Renderable::new(IcosphereGeometry::new(0.5 * RADIUS, 4), glass_material()).with_rt(RtSurface::glass(tint));
        inner.object.set_position(x, RADIUS, 0.0);
        scene.add(SceneNode::Renderable(inner));
    }
    let spot = SpotLight::new(Vec3::new(0.0, LAMP, 0.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), CANDELA, 100.0, 50f32.to_radians(), 60f32.to_radians());
    let d = SUN_DIR.normalize();
    let sunlight = DirectionalLight::new(Vec3::new(d.x, d.y, d.z), Vec3::new(1.0, 1.0, 1.0), SUN);
    scene.add(SceneNode::Light(if sun { Light::Directional(sunlight) } else { Light::Spot(spot) }));
    let gbuffer = GBuffer::new(renderer.device(), W, H, 1);
    let mut camera = Camera::new(50.0, 0.01, 10.0, W as f32 / H as f32);
    camera.set_position(0.0, 0.08, 0.08);
    camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
    camera.update_projection_matrix();
    let texture = |usage| {
        renderer.device().create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: W, height: H, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage,
            view_formats: &[],
        })
    };
    let lit = texture(wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
    let output = texture(wgpu::TextureUsages::STORAGE_BINDING).create_view(&Default::default());
    let handle = renderer.rt_grid().unwrap().handle();
    eprintln!("asked for {backend:?}, traced with {:?}", handle.backend());
    let mut shadows = RtShadowsEffect::new(handle.clone());
    shadows.update_lights(scene.lights());
    let mut caustics = RtCausticsEffect::new(renderer.device(), handle, RtCausticsOptions { texel: 0.0005, ..Default::default() });
    caustics.set_glass(ior, [1.0; 3]);
    // a centimetre: the core's tint is the light it leaves after one
    caustics.tint_distance = 0.01;
    caustics.update_lights(scene.lights());
    caustics.set_casters(&[(GVec3::new(x - RADIUS, 0.0, -RADIUS), GVec3::new(x + RADIUS, 2.0 * RADIUS, RADIUS))]);
    for _ in 0..frames {
        renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
        let mut encoder = renderer.device().create_command_encoder(&Default::default());
        shadows.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &gbuffer.color_view, &gbuffer.depth_view, &lit, &camera, W, H);
        caustics.render(renderer.device(), renderer.queue(), &mut encoder, &gbuffer, &lit, &gbuffer.depth_view, &output, &camera, W, H);
        renderer.queue().submit(Some(encoder.finish()));
    }
    let (min, texel, dims, history) = caustics.tile(0).expect("the marble's tile is on screen");
    let (sent, photons, tiles) = caustics.photons_sent();
    eprintln!("tile at {min:?}, {dims:?} texels of {:.2} mm, {history} frames; last frame {sent} lights x casters, {photons} photons, {tiles} tiles", texel * 1e3);
    let texels = read_layer(&renderer, caustics.map_texture(), 0, dims);
    let centres = (0..dims[0] * dims[1]).map(|i| [min[0] + ((i % dims[0]) as f32 + 0.5) * texel, min[1] + ((i / dims[0]) as f32 + 0.5) * texel]).collect();
    Some(Tile { texel, texels, history, centres })
}

/// The light's flux into the marble (lm): the spot light's intensity times the solid angle the
/// sphere fills, seen from the lamp (inside the cone's full intensity), or the sun's illuminance
/// times its cross-section.
fn flux_into_marble(sun: bool) -> f32 {
    if sun {
        return SUN * std::f32::consts::PI * RADIUS * RADIUS;
    }
    let d = LAMP - RADIUS;
    CANDELA * 2.0 * std::f32::consts::PI * (1.0 - (1.0 - (RADIUS / d).powi(2)).sqrt())
}

/// The direct light on the floor under the lamp or the sun (lux).
fn direct_under_lamp(sun: bool) -> f32 {
    if sun {
        return SUN * -SUN_DIR.normalize().y;
    }
    CANDELA / (LAMP * LAMP)
}

/// Where the marble's shadow is centred on the floor (x, z) for a marble at x: under it, or along
/// the sun's slant from its centre.
fn shadow_centre(sun: bool, x: f32) -> [f32; 2] {
    if !sun {
        return [x, 0.0];
    }
    let d = SUN_DIR.normalize();
    let s = RADIUS / -d.y;
    [x + d.x * s, d.z * s]
}

/// The tile's mean in 1 mm rings round `at` (x, z), and the spread of its texels from 6 to 12 mm out
/// (all shadow).
fn rings(tile: &Tile, at: [f32; 2]) -> ([f32; 16], f32) {
    let r = |c: &[f32; 2]| (c[0] - at[0]).hypot(c[1] - at[1]);
    let mut sums = [(0.0f32, 0u32); 16];
    for (t, c) in tile.texels.iter().zip(&tile.centres) {
        let k = ((r(c) / 0.001) as usize).min(15);
        sums[k].0 += t[1];
        sums[k].1 += 1;
    }
    let annulus: Vec<f32> = tile.texels.iter().zip(&tile.centres).filter(|(_, c)| (6e-3..12e-3).contains(&r(c))).map(|(t, _)| t[1]).collect();
    let mean = annulus.iter().sum::<f32>() / annulus.len() as f32;
    let rms = (annulus.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / annulus.len() as f32).sqrt();
    (sums.map(|(s, n)| s / n.max(1) as f32), rms)
}

fn conservation(backend: RtBackend, sun: bool) {
    eprintln!("{}", if sun { "under the sun" } else { "under the spot light" });
    // index 1: no change at all
    let Some(tile) = run(backend, sun, 1.0, None, 8, 0.0) else { return eprintln!("no GPU adapter: skipping") };
    assert_eq!(tile.history, 8, "a still marble's tile accumulates");
    let area = tile.texel * tile.texel;
    let sum: f32 = tile.texels.iter().map(|t| t[1]).sum::<f32>() * area;
    let worst = tile.texels.iter().map(|t| t[1].abs()).fold(0.0, f32::max);
    let flux = flux_into_marble(sun);
    eprintln!("index 1: the tile's sum {sum:.2e} lm of {flux:.2e} through the marble, its largest texel {worst:.3} lux of {:.1} direct", direct_under_lamp(sun));
    assert!(sum.abs() < 0.01 * flux, "index-matched glass changes no light ({sum} of {flux})");
    assert!(worst < 0.02 * direct_under_lamp(sun), "nor any texel's ({worst} lux)");

    // clear glass: the light moves, little is lost
    let tile = run(backend, sun, 1.5, None, 32, 0.0).unwrap();
    let area = tile.texel * tile.texel;
    let sum: f32 = tile.texels.iter().map(|t| t[1]).sum::<f32>() * area;
    let brightest = tile.texels.iter().map(|t| t[1]).fold(f32::MIN, f32::max);
    let darkest = tile.texels.iter().map(|t| t[1]).fold(f32::MAX, f32::min);
    let direct = direct_under_lamp(sun);
    let with = 1.0 + sum / flux;
    let (profile, rms) = rings(&tile, shadow_centre(sun, 0.0));
    eprintln!("index 1.5: the light through the marble's footprint with the glass {with:.3} of without; texels {darkest:.1}..{brightest:.1} lux over {direct:.1} direct");
    eprintln!("1 mm rings: {:?}; the shadow 6-12 mm out varies {rms:.2} lux", profile.map(|v| v.round()));
    assert!(sum <= 0.01 * flux && sum >= -0.2 * flux, "clear glass loses only what its surfaces reflect away ({sum} of {flux})");
    assert!(brightest > 5.0 * direct, "the marble focuses the light under it ({brightest} lux)");
    assert!(darkest < -0.9 * direct, "and leaves a shadow round the focus ({darkest} lux)");
    // the focus lies under the marble (within 3 mm of its axis), the light it took round it
    let mean = |r: std::ops::Range<usize>| profile[r.clone()].iter().sum::<f32>() / r.len() as f32;
    let (focus, shadow) = (mean(0..3), mean(5..14));
    assert!(focus > direct && shadow < -0.8 * direct, "the focus ({focus:.1} lux) inside the shadow ({shadow:.1} lux)");
    assert!(rms < 0.1 * direct, "the shadow is smooth ({rms:.2} lux from texel to texel)");

    // a red core: the light through it red
    let centre = shadow_centre(sun, 0.0);
    let tile = run(backend, sun, 1.5, Some([0.9, 0.05, 0.05]), 32, 0.0).unwrap();
    let (mut red, mut green) = (0.0, 0.0);
    for (t, c) in tile.texels.iter().zip(&tile.centres) {
        if (c[0] - centre[0]).hypot(c[1] - centre[1]) < 0.5 * RADIUS {
            red += t[0];
            green += t[1];
        }
    }
    eprintln!("red core: under the marble red {red:.1}, green {green:.1}");
    assert!(red > green + 0.1 * red.abs().max(1.0), "the core's colour reaches the floor ({red} red, {green} green)");

    // rolling: one frame of photons (as a moving marble's tile has), here and a texel and 0.37 of
    // one along: the same caustics about the marble wherever it is, ring by ring for a whole
    // texel; for a fraction of one (where the rings' texel centres sample the focus's steep edge
    // elsewhere) the light within 6.5 mm, its rim in the flat shadow
    let texel = tile.texel;
    let within = |tile: &Tile, x: f32| {
        let at = shadow_centre(sun, x);
        tile.texels.iter().zip(&tile.centres).filter(|(_, c)| (c[0] - at[0]).hypot(c[1] - at[1]) < 0.0065).map(|(t, _)| t[1]).sum::<f32>() * texel * texel
    };
    let still = run(backend, sun, 1.5, None, 32, 0.0).unwrap();
    let (still_rings, _) = rings(&still, shadow_centre(sun, 0.0));
    let still_within = within(&still, 0.0);
    for x in [0.0, texel, 0.37 * texel] {
        let tile = run(backend, sun, 1.5, None, 1, x).unwrap();
        let (profile, rms) = rings(&tile, shadow_centre(sun, x));
        let inside = within(&tile, x);
        eprintln!("one frame, the marble {:.2} mm along: 1 mm rings {:?}, within 6.5 mm {:.3e} lm (still {still_within:.3e}), the shadow varies {rms:.2} lux", x * 1e3, profile.map(|v| v.round()), inside);
        assert!(rms < 0.05 * direct, "a rolling marble's shadow is smooth ({rms:.2} lux from texel to texel)");
        assert!((inside - still_within).abs() < 0.02 * still_within.abs(), "and brings the still marble's light ({inside} lm of {still_within})");
        if x != 0.37 * texel {
            let worst = profile.iter().zip(&still_rings).map(|(a, b)| (a - b).abs() / (0.05 * direct + 0.05 * b.abs())).fold(0.0, f32::max);
            assert!(worst < 1.0, "its caustics are the still marble's, a texel along ({worst:.2} of the tolerance)");
        }
    }
}

#[test]
fn caustics_through_the_grid_conserve_light() {
    conservation(RtBackend::Software, false);
}

#[test]
fn caustics_through_hardware_ray_queries_conserve_light() {
    conservation(RtBackend::Hardware, false);
}

#[test]
fn sun_caustics_through_the_grid_conserve_light() {
    conservation(RtBackend::Software, true);
}

#[test]
fn sun_caustics_through_hardware_ray_queries_conserve_light() {
    conservation(RtBackend::Hardware, true);
}
