//! Planar reflection: the Midsommar lake shot in miniature. A still lake at dusk mirrors the far
//! shore's treeline, a red cottage with lit windows and the sky, through a PlanarReflection
//! (mirrored camera, oblique clip plane at the water, the water itself left out by its layer).
//! The water material adds Fresnel, wind ripples that displace the lookup and a roughness that
//! picks the reflection's mips. A lamp on the far bank throws a beam through the mist, and the
//! volumetric fog is composited into the reflection too (`VolumetricFogEffect::reflection_fog`),
//! so the lake mirrors the beam's glow.
//!
//! URL parameters: `ripples=<strength>` (0 = mirror), `rough=<0..1>`, `t=<seconds>` (freeze),
//! `fogrefl=0` (no fog in the reflection: the water fogs the reflected path with a flat colour),
//! `occlusion=1` (the treeline occlusion-culled, for the camera and in the mirror
//! (`PlanarReflection::occlusion_culling`); the culling stats logged every 2 s), `screen=1` (the
//! reflection from the screen, `PlanarReflection::screen_space`, instead of the mirrored view).

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler};
use kansei_core::culling::InstanceCulling;
use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages, StandardInstancing, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::reflections::{PlanarReflection, PlanarReflectionOptions, PLANAR_REFLECTION_WGSL};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{flag, param, param_or, Canvas};

const WATER_LAYER: u32 = 2;
const LAKE_LEVEL: f32 = 0.0;

/// The dusk sky's radiance from straight up and the ground's bounce from straight down (cd/m²):
/// 40 cd/m² from above and the west, a dark ground.
const SKY_UP: [f32; 3] = [22.0, 26.0, 36.0];
const SKY_DOWN: [f32; 3] = [0.4, 0.45, 0.5];

/// The cottage: diffuse under the dusk sky (the hemisphere of the stock material), and its
/// windows, a band of emission around the box's middle (|local y| < emissive.w).
/// Prefixed with GBUFFER_OUT_WGSL.
const COTTAGE_WGSL: &str = r#"
struct Cottage { base_color: vec4<f32>, emissive: vec4<f32>, sky_up: vec4<f32>, sky_down: vec4<f32> };
@group(0) @binding(0) var<uniform> cottage: Cottage;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) normal: vec3<f32>, @location(1) local_y: f32 };

@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * vec4<f32>(position.xyz, 1.0);
    out.normal = (normal_matrix * vec4<f32>(normal, 0.0)).xyz;
    out.local_y = position.y;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let n = normalize(in.normal);
    let sky = mix(cottage.sky_down.rgb, cottage.sky_up.rgb, n.y * 0.5 + 0.5);
    let window = select(0.0, 1.0, abs(in.local_y) < cottage.emissive.w);
    let emissive = cottage.emissive.rgb * window;
    return kansei_gbuffer_out(cottage.base_color.rgb * sky + emissive, emissive, n, cottage.base_color.rgb);
}
"#;

/// Dusk sky dome: horizon glow toward the (set) sun in the north-west, deep blue overhead.
/// Prefixed with GBUFFER_OUT_WGSL.
const SKY_WGSL: &str = r#"
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) dir: vec3<f32> };
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.dir = position.xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let d = normalize(in.dir);
    let up = saturate(d.y);
    let glow = pow(saturate(dot(d, normalize(vec3<f32>(-0.6, 0.0, -0.8))) * 0.5 + 0.5), 6.0);
    let horizon = mix(vec3<f32>(60.0, 62.0, 70.0), vec3<f32>(240.0, 150.0, 90.0), glow);
    let zenith = vec3<f32>(18.0, 28.0, 55.0);
    return kansei_gbuffer_out(mix(horizon, zenith, pow(up, 0.45)), vec3<f32>(0.0), -d, vec3<f32>(0.0));
}
"#;

/// The lake: Fresnel mix of a dark body colour and the planar reflection, displaced by wind
/// ripples and fogged over the reflected path. Prefixed with PLANAR_REFLECTION_WGSL and
/// GBUFFER_OUT_WGSL.
const WATER_WGSL: &str = r#"
struct Water { params: vec4<f32>, fog: vec4<f32>, body: vec4<f32> };  // params: time, ripples, roughness, -
@group(0) @binding(0) var<uniform> water: Water;
@group(0) @binding(1) var reflection_tex: texture_2d<f32>;
@group(0) @binding(2) var reflection_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut { @builtin(position) pixel: vec4<f32>, @location(0) world: vec3<f32>, @location(1) clip: vec4<f32> };

@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    let world = world_matrix * position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.pixel = out.clip;
    out.world = world.xyz;
    return out;
}

// gradient of a few travelling sine waves: faint wind ripples
fn ripple_slope(p: vec2<f32>, t: f32) -> vec2<f32> {
    var g = vec2<f32>(0.0);
    let dirs = array<vec2<f32>, 4>(vec2<f32>(0.8, 0.6), vec2<f32>(-0.3, 0.95), vec2<f32>(0.99, -0.14), vec2<f32>(-0.7, -0.7));
    let freq = array<f32, 4>(1.3, 2.1, 3.7, 5.3);
    for (var i = 0; i < 4; i++) {
        let k = freq[i];
        g += dirs[i] * cos(dot(dirs[i], p) * k + t * sqrt(9.81 * k)) * (0.35 / k);
    }
    return g;
}

@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let slope = ripple_slope(in.world.xz, water.params.x) * water.params.y;
    let n = normalize(vec3<f32>(-slope.x, 1.0, -slope.y));
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let to_eye = camera_pos - in.world;
    let v = normalize(to_eye);

    let uv = kansei_screen_uv(in.clip);
    let offset = kansei_reflection_offset(view_matrix, up, n, 0.6);
    let r = kansei_planar_reflection(reflection_tex, reflection_sampler, uv, offset, water.params.z);
    // fog over the part of the path beyond the water (the main fog covers camera -> water)
    let beyond = max(r.a - length(to_eye), 0.0);
    let t = exp(-water.fog.w * beyond);
    let reflected = r.rgb * t + water.fog.rgb * (1.0 - t);

    let fresnel = 0.02 + 0.98 * pow(1.0 - saturate(dot(n, v)), 5.0);
    return kansei_gbuffer_out(mix(water.body.rgb, reflected, fresnel), vec3<f32>(0.0), n, vec3<f32>(0.0));
}
"#;

/// A diffuse surface under the dusk sky (the stock standard material), instanced for the
/// treeline (vec4 per instance: xyz offset, w height scale).
fn surface_material(label: &str, base: [f32; 3], instanced: bool) -> Material {
    Material::standard_lit(label, &StandardLitOptions {
        base_color: base,
        roughness: 0.9,
        sky_up: SKY_UP,
        sky_down: SKY_DOWN,
        instancing: instanced.then_some(StandardInstancing::OffsetHeight),
        ..Default::default()
    })
}

/// The cottage, with windows glowing at `window` (cd/m²) where |local y| < `band`.
fn cottage_material(label: &str, base: [f32; 3], window: [f32; 3], band: f32) -> Material {
    let (u, d) = (SKY_UP, SKY_DOWN);
    let data: [f32; 16] = [base[0], base[1], base[2], 1.0, window[0], window[1], window[2], band, u[0], u[1], u[2], 0.0, d[0], d[1], d[2], 0.0];
    let options = MaterialOptions { mrt_output_count: Some(4), ..Default::default() };
    let mut m = Material::new(label, &format!("{GBUFFER_OUT_WGSL}\n{COTTAGE_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], options);
    m.set_uniform_bindable(0, label, &data);
    m
}

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

/// The reflection: half the canvas's resolution, everything but the water, with the fog's
/// mirrored froxels composited in when `fog` is given.
fn lake_reflection(renderer: &Renderer, (width, height): (u32, u32), occlusion: bool, screen_space: bool, fog: Option<&mut VolumetricFogEffect>) -> PlanarReflection {
    let mut reflection = PlanarReflection::new(
        renderer,
        Vec3::new(0.0, LAKE_LEVEL, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        PlanarReflectionOptions { width: (width / 2).max(1), height: (height / 2).max(1), layer_mask: !WATER_LAYER, ..Default::default() },
    );
    reflection.occlusion_culling = occlusion;
    reflection.screen_space = screen_space;
    if let Some(fog) = fog {
        let seen = fog.reflection_fog(renderer, &reflection);
        reflection.set_fog(renderer, Some(&seen));
    }
    reflection
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;

    let mut scene = Scene::new();
    // (a pipeline layout's group 0 needs a bind group, so the sky gets an unused uniform)
    let sky_options = MaterialOptions { cull_mode: CullMode::None, mrt_output_count: Some(4), ..Default::default() };
    let mut sky = Material::new("Sky", &format!("{GBUFFER_OUT_WGSL}\n{SKY_WGSL}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], sky_options);
    sky.set_uniform_bindable(0, "Sky", &[0.0f32; 4]);
    let mut sky = Renderable::new(SphereGeometry::new(900.0, 48, 24), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));

    // far shore: a bank rising out of the lake 120-400 m away, with a treeline on it
    let mut bank = Renderable::new(BoxGeometry::new(1600.0, 6.0, 300.0), surface_material("Bank", [0.05, 0.06, 0.04], false));
    bank.object.set_position(0.0, 1.0, -270.0);
    scene.add(SceneNode::Renderable(bank));
    let mut trees: Vec<f32> = Vec::new();
    for i in 0..1600u32 {
        let x = -700.0 + hash(i) * 1400.0;
        let z = -125.0 - hash(i + 5) * 200.0;
        let h = 12.0 + hash(i + 11) * 14.0;
        trees.extend_from_slice(&[x, 4.0 + h * 0.5, z, h]);
    }
    let count = trees.len() as u32 / 4;
    let all_trees = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Trees"),
            contents: bytemuck::cast_slice(&trees),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
        })
    };
    let instances = ComputeBuffer::from_external("Trees", all_trees.clone(), BufferType::Storage).with_vertex_vec4(3);
    // spruce-like silhouettes: narrow tall boxes (a stand-in; the app has real trees), culled on
    // the GPU for the camera and, separately, for the mirrored view
    let treeline = InstancedGeometry::new(BoxGeometry::new(2.6, 1.0, 2.6), count, vec![instances]);
    let mut treeline = Renderable::new(treeline, surface_material("Trees", [0.03, 0.04, 0.03], true));
    let occlusion = flag("occlusion", false);
    treeline.instance_culling = Some(InstanceCulling::new(all_trees, count, 16, 0, 0.6).with_radius_scale(12).with_occlusion(occlusion));
    scene.add(SceneNode::Renderable(treeline));

    // the red cottage on the shore, windows glowing at 160 cd/m² (the intro's window_glow)
    let mut cottage = Renderable::new(BoxGeometry::new(9.0, 5.0, 6.0), cottage_material("Cottage", [0.35, 0.05, 0.03], [160.0, 110.0, 60.0], 0.6));
    cottage.object.set_position(-18.0, 6.5, -128.0);
    scene.add(SceneNode::Renderable(cottage));
    let mut roof = Renderable::new(BoxGeometry::new(9.6, 1.2, 6.6), surface_material("Roof", [0.04, 0.04, 0.04], false));
    roof.object.set_position(-18.0, 9.6, -128.0);
    scene.add(SceneNode::Renderable(roof));

    // near shore under the camera, so the lake has an edge
    let mut near_bank = Renderable::new(BoxGeometry::new(1600.0, 4.0, 60.0), surface_material("NearBank", [0.04, 0.05, 0.03], false));
    near_bank.object.set_position(0.0, -1.2, 44.0);
    scene.add(SceneNode::Renderable(near_bank));

    // a searchlight on the far bank beside the cottage (5e7 cd), its beam rising across the lake through
    // the mist
    let lamp_pos = Vec3::new(-32.0, 11.0, -126.0);
    let mut lamp = SpotLight::new(
        lamp_pos,
        Vec3::new(40.0 - lamp_pos.x, 45.0 - lamp_pos.y, -50.0 - lamp_pos.z).normalize(),
        Vec3::new(1.0, 0.82, 0.6),
        5.0e7,
        300.0,
        3f32.to_radians(),
        7f32.to_radians(),
    );
    lamp.volumetric_scale = 1.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    let screen_space = flag("screen", false);
    renderer.set_culling_stats(occlusion);
    let ripples: f32 = param_or("ripples", 0.03);
    let roughness: f32 = param_or("rough", 0.05);
    // the fog in the reflection (fogrefl=0 leaves it out, and the water fogs the reflected path
    // with a flat colour instead, as before)
    let fog_in_reflection = flag("fogrefl", true);
    // time, ripples, roughness, - | fog colour, fog density | deep body colour
    let water_fog = if fog_in_reflection { 0.0 } else { 0.0012 };
    let mut water_params: [f32; 12] = [0.0, ripples, roughness, 0.0, 45.0, 48.0, 58.0, water_fog, 0.2, 0.3, 0.3, 1.0];
    let mut water_material = Material::new(
        "Water",
        &format!("{PLANAR_REFLECTION_WGSL}\n{GBUFFER_OUT_WGSL}\n{WATER_WGSL}"),
        vec![
            Binding::uniform(0, ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { cull_mode: CullMode::None, mrt_output_count: Some(4), ..Default::default() },
    );
    water_material.set_uniform_bindable(0, "Water", &water_params);
    water_material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));

    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { near: 1.0, far: 800.0, temporal: true, ..Default::default() },
        base_density: 0.0012,
        height_falloff: 0.04,
        anisotropy: 0.3,
        ambient: Vec3::new(38.0, 40.0, 50.0),
        ..Default::default()
    });
    fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
    let reflection = lake_reflection(&renderer, canvas.size(), occlusion, screen_space, fog_in_reflection.then_some(&mut fog));
    water_material.set_bindable(1, reflection.material_texture());
    let reflection_index = renderer.add_planar_reflection(reflection);
    let mut lake = Renderable::new(PlaneGeometry::new(1600.0, 400.0), water_material);
    lake.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    lake.object.set_position(0.0, LAKE_LEVEL, -100.0);
    lake.layers = WATER_LAYER;
    lake.cast_shadow = false;
    let water = scene.add(SceneNode::Renderable(lake));

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(6.5);
        options.vignette = 0.5;
        options.grain = 0.22;
        ToneMapEffect::new(options)
    };
    let effects: Vec<Box<dyn PostProcessingEffect>> = vec![
        Box::new(fog),
        Box::new(BloomEffect::new(BloomOptions { threshold: 0.0, intensity: 0.04, ..Default::default() }).with_exposure(tonemap.total_exposure())),
        Box::new(tonemap),
    ];
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(28.0, 0.5, 2000.0, canvas.aspect());
    camera.update_projection_matrix();

    log::info!("Kansei — Planar Reflection (WASM) ready: ripples {ripples}, roughness {roughness}");

    let frozen_t: Option<f32> = param("t").and_then(|v| v.trim().parse().ok());
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        if let Some(size) = frame.resized {
            // the reflection's target follows the canvas: a new one, wired to the fog and the water
            let fog = volume.effect_mut::<VolumetricFogEffect>().filter(|_| fog_in_reflection);
            let reflection = lake_reflection(&renderer, size, occlusion, screen_space, fog);
            if let Some(r) = scene.get_renderable_mut(water) {
                r.material.set_bindable(1, reflection.material_texture());
            }
            if let Some(slot) = renderer.planar_reflection_mut(reflection_index) {
                *slot = reflection;
            }
            renderer.invalidate_bundle();
        }
        let clock = frame.time as f32;
        let t = frozen_t.unwrap_or(clock);

        if let Some(fog) = volume.effect_mut::<VolumetricFogEffect>() {
            fog.time = clock;
        }
        water_params[0] = clock;
        if let Some(r) = scene.get_renderable_mut(water) {
            if let Some(buf) = r.material.bindable_buffer(0) {
                renderer.queue().write_buffer(&buf, 0, bytemuck::cast_slice(&water_params[..]));
            }
        }

        // low over the near shore, panning slowly along the far treeline
        let yaw = -0.25 + 0.12 * (t * 0.05).sin();
        camera.set_position(6.0, 2.2, 16.0);
        camera.look_at(&Vec3::new(6.0 + yaw.sin() * 100.0, 3.0, 16.0 - yaw.cos() * 100.0));

        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
        if camera.frame() % 120 == 0 {
            if let Some(stats) = renderer.culling_stats() {
                log::info!("culling: {:?}", stats.views);
            }
        }
    });
    Ok(())
}
