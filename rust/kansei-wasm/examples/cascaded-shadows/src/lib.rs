//! Cascaded shadows: a forest under a low afternoon sun, 4 stable cascades of 2048² out to
//! 250 m with contact-hardening (PCSS) penumbrae, the trees instanced and GPU-culled per cascade,
//! through TAA and the tonemapper. `csm=0` uses the single 2048² directional map instead, for
//! comparison; `debug=1` tints each cascade.
//!
//! URL parameters: `csm=0`, `debug=1`, `fog=1` (volumetric fog with shafts from the widest
//! cascade), `far=<metres>` (camera far plane, which the single map
//! is fitted to), `t=<seconds>` (freeze the camera).

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, GradientSkyOptions, Material, MaterialOptions, ShaderStages, StandardInstancing, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, param_or, Canvas};
use kansei_core::shadows::{CascadedShadowOptions, CASCADED_SHADOWS_WGSL};

/// `debug=1`: the sun's light only, each point tinted by the cascade that shadows it.
const CASCADE_TINT_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> base_color: vec4<f32>;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;
struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>, INSTANCE_INPUT };
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };
@vertex
fn vertex_main(v: VIn) -> VOut {
    var local = v.position.xyz;
    INSTANCE_PLACE
    let world = world_matrix * vec4<f32>(local, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let n = normalize(in.normal);
    let tints = array<vec3<f32>, 5>(vec3<f32>(1.0, 0.4, 0.4), vec3<f32>(0.4, 1.0, 0.4), vec3<f32>(0.4, 0.6, 1.0), vec3<f32>(1.0, 1.0, 0.4), vec3<f32>(1.0));
    let base = base_color.rgb * tints[min(kansei_sun_cascade(in.world), 4u)];
    let sun = kansei_cascades.lightColor * max(dot(n, -kansei_cascades.lightDirection), 0.0) * kansei_sun_shadow(in.world, n, in.clip.xy);
    return kansei_gbuffer_out(base / 3.14159265 * sun, vec3<f32>(0.0), n, base);
}
"#;

/// Low afternoon sun's travel direction, and the sky's radiance from straight up (cd/m²).
const SUN_DIR: [f32; 3] = [-0.62, -0.42, -0.66];
const SKY: [f32; 3] = [4000.0, 5000.0, 7000.0];

/// A sunlit diffuse surface (the stock standard material), instanced for trunks and crowns
/// (vec4: position, scale), or with `debug` the cascade tint.
fn surface_material(label: &str, base: [f32; 3], instanced: bool, debug: bool) -> Material {
    if debug {
        let (input, place) = if instanced { ("@location(3) instance: vec4<f32>,", "local = local * v.instance.w + v.instance.xyz;") } else { ("", "") };
        let shader = CASCADE_TINT_WGSL.replace("INSTANCE_INPUT", input).replace("INSTANCE_PLACE", place);
        let options = MaterialOptions { mrt_output_count: Some(4), ..Default::default() };
        let mut m = Material::new(label, &format!("{CASCADED_SHADOWS_WGSL}\n{GBUFFER_OUT_WGSL}\n{shader}"), vec![Binding::uniform(0, ShaderStages::FRAGMENT)], options);
        m.set_uniform_bindable(0, label, &[base[0], base[1], base[2], 1.0]);
        return m;
    }
    Material::standard_lit(label, &StandardLitOptions {
        base_color: base,
        roughness: 0.9,
        sky_up: SKY,
        sky_down: SKY.map(|c| c * 0.15),
        instancing: instanced.then_some(StandardInstancing::OffsetScale),
        ..Default::default()
    })
}

fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    let csm = flag("csm", true);
    let debug = flag("debug", false);
    if csm {
        renderer.enable_cascaded_shadows(CascadedShadowOptions::default());
    } else {
        renderer.enable_shadows(2048);
    }

    let mut scene = Scene::new();
    let sky = Material::gradient_sky("Sky", &GradientSkyOptions { zenith: [3000.0, 5000.0, 9000.0], horizon: [9000.0, 9500.0, 10500.0], ground: [9000.0, 9500.0, 10500.0], curve: 0.5 });
    let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), sky);
    sky.cast_shadow = false;
    scene.add(SceneNode::Renderable(sky));

    let mut ground = Renderable::new(PlaneGeometry::new(1600.0, 1600.0), surface_material("Ground", [0.16, 0.18, 0.1], false, debug));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    ground.cast_shadow = false;
    scene.add(SceneNode::Renderable(ground));

    // a forest: trunks (unit boxes scaled by height) and crowns (spheres), 4000 trees out to
    // 500 m, clear around the path the camera walks
    let (mut trunks, mut crowns) = (Vec::new(), Vec::new());
    for i in 0..4000u32 {
        let (x, z) = (-500.0 + hash(i) * 1000.0, -500.0 + hash(i + 7) * 1000.0);
        if x.abs() < 4.0 {
            continue;
        }
        let h = 8.0 + hash(i + 13) * 10.0;
        trunks.push([x, h * 0.5, z, h]);
        crowns.push([x, h, z, 1.6 + hash(i + 17) * 1.4]);
    }
    scene.add(SceneNode::Renderable(Renderable::instanced_culled("Trunks", BoxGeometry::new(0.035, 1.0, 0.035), &trunks, 0.6, surface_material("Trunk", [0.2, 0.15, 0.1], true, debug))));
    scene.add(SceneNode::Renderable(Renderable::instanced_culled("Crowns", SphereGeometry::new(1.0, 12, 8), &crowns, 1.0, surface_material("Crown", [0.06, 0.12, 0.05], true, debug))));
    // a fence along the path: thin, close shadows
    for k in 0..60 {
        let mut post = Renderable::new(BoxGeometry::new(0.08, 1.2, 0.08), surface_material("Post", [0.35, 0.3, 0.25], false, debug));
        post.object.set_position(2.5, 0.6, 10.0 - k as f32 * 2.0);
        scene.add(SceneNode::Renderable(post));
    }

    let mut sun = DirectionalLight::new(Vec3::new(SUN_DIR[0], SUN_DIR[1], SUN_DIR[2]), Vec3::new(1.0, 0.875, 0.7), 80000.0);
    sun.cast_shadow = true;
    scene.add(SceneNode::Light(Light::Directional(sun)));

    let tonemap = {
        let mut o = ToneMapOptions::for_surface(renderer.presentation_format());
        o.exposure = exposure_from_ev100(14.5);
        o.vignette = 0.4;
        ToneMapEffect::new(o)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if flag("fog", false) {
        // shafts through the canopy, shadowed by the widest cascade
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 300.0, temporal: true, ..Default::default() },
            base_density: 0.01,
            height_falloff: 0.05,
            anisotropy: 0.6,
            ambient: Vec3::new(SKY[0], SKY[1], SKY[2]) * 0.5,
            ..Default::default()
        });
        if csm {
            fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        } else {
            fog.set_shadow_map(renderer.shadow_map());
        }
        fog.update_lights(scene.lights());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    effects.push(Box::new(tonemap));
    let mut volume = PostProcessingVolume::new(&renderer, effects);
    let far: f32 = param_or("far", 1200.0);
    let mut camera = Camera::new(50.0, 0.3, far, canvas.aspect());

    log::info!("Kansei — Cascaded Shadows (WASM) ready: {} trees, cascades {csm}", trunks.len());

    let frozen_t: Option<f32> = kansei_wasm::param("t").and_then(|v| v.parse().ok());
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let t = frozen_t.unwrap_or(frame.time as f32);
        // walk down the path, looking along it and slightly toward the sun
        let z = 8.0 - (t * 1.2) % 60.0;
        camera.set_position(0.0, 1.7, z);
        camera.look_at(&Vec3::new(4.0, 1.2, z - 20.0));
        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    });
    Ok(())
}
