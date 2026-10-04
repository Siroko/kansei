//! Film look: a dusk scene in physical-ish units (cd/m²) through the full HDR chain: volumetric
//! fog, physically based bloom, then ToneMapEffect (EV100 exposure, filmic curve, scene-linear
//! grade, vignette, grain, chromatic aberration, sRGB encoding and dither).
//!
//! URL parameters: `tm=aces|agx|punchy|neutral|unreal|none` (curve), `ev=<EV100>` (exposure),
//! `film=0` (no vignette/grain/CA), `bloom=0`, `t=<seconds>` (freeze the camera), `le=<contrast>`
//! (Unreal's local exposure at this highlight and shadow contrast, e.g. 0.8).

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, PlaneGeometry, SphereGeometry};
use kansei_core::lights::{DirectionalLight, Light, PointLight};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, LocalExposure, ToneMapEffect, ToneMapOptions, ToneMapper, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, param, param_or, Canvas};

/// Unlit radiance (cd/m²), for lamp heads; `sky` mode is a horizon-to-zenith gradient.
const EMISSIVE_WGSL: &str = r#"
struct Emissive { radiance: vec4<f32>, zenith: vec4<f32> };
@group(0) @binding(0) var<uniform> emissive: Emissive;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VOut { @builtin(position) clip: vec4<f32>, @location(0) local: vec3<f32> };

@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VOut {
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world_matrix * position;
    out.local = position.xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> {
    // zenith.w > 0: sky gradient over the local up axis
    if (emissive.zenith.w > 0.0) {
        let up = clamp(normalize(in.local).y, 0.0, 1.0);
        return vec4<f32>(mix(emissive.radiance.rgb, emissive.zenith.rgb, pow(up, 0.5)), 1.0);
    }
    return vec4<f32>(emissive.radiance.rgb, 1.0);
}
"#;

fn emissive_material(label: &str, radiance: [f32; 3], zenith: Option<[f32; 3]>, cull_mode: CullMode) -> Material {
    let z = zenith.unwrap_or([0.0; 3]);
    let data: [f32; 8] = [radiance[0], radiance[1], radiance[2], 0.0, z[0], z[1], z[2], zenith.is_some() as u32 as f32];
    let mut material = Material::new(
        label,
        EMISSIVE_WGSL,
        vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
        MaterialOptions { cull_mode, ..Default::default() },
    );
    material.set_uniform_bindable(0, label, &data);
    material
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas
        .renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() })
        .await;
    renderer.enable_point_shadows(512, 1);

    let mut scene = Scene::new();

    // dusk sky: 6 cd/m² at the horizon, 1.2 at the zenith
    let sky = Renderable::new(SphereGeometry::new(180.0, 32, 16), emissive_material("Sky", [6.0, 5.2, 4.6], Some([0.8, 1.1, 1.8]), CullMode::None));
    scene.add(SceneNode::Renderable(sky));

    let mut floor = Renderable::new(PlaneGeometry::new(120.0, 120.0), Material::basic_lit("Floor", [0.3, 0.31, 0.3, 1.0], [0.04, 0.04, 0.04, 0.05]));
    floor.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    floor.cast_shadow = false;
    scene.add(SceneNode::Renderable(floor));

    // a grove of trunks either side of a path
    for i in 0..14 {
        for side in [-1.0f32, 1.0] {
            let seed = (i * 31 + if side > 0.0 { 7 } else { 0 }) % 11;
            let x = side * (3.5 + (seed % 4) as f32 * 1.7);
            let z = -30.0 + i as f32 * 4.3 + (seed % 3) as f32 * 0.8;
            let h = 7.0 + (seed % 5) as f32;
            let mut trunk = Renderable::new(BoxGeometry::new(0.4, h, 0.4), Material::basic_lit("Trunk", [0.25, 0.2, 0.16, 1.0], [0.03, 0.03, 0.03, 0.1]));
            trunk.object.set_position(x, h * 0.5, z);
            scene.add(SceneNode::Renderable(trunk));
        }
    }

    // lamps along the path: 4000 cd/m² heads, which bloom and clip; the first one casts shadows
    for (k, z) in [-18.0f32, -4.0, 10.0].into_iter().enumerate() {
        let x = if k % 2 == 0 { -1.8 } else { 1.8 };
        let mut pole = Renderable::new(BoxGeometry::new(0.12, 3.0, 0.12), Material::basic_lit("Pole", [0.1, 0.1, 0.1, 1.0], [0.1, 0.1, 0.1, 0.3]));
        pole.object.set_position(x, 1.5, z);
        scene.add(SceneNode::Renderable(pole));
        let mut head = Renderable::new(SphereGeometry::new(0.18, 16, 8), emissive_material("LampHead", [4000.0, 2600.0, 1400.0], None, CullMode::Back));
        head.object.set_position(x, 3.1, z);
        head.cast_shadow = false;
        scene.add(SceneNode::Renderable(head));
        let mut lamp = PointLight::new(Vec3::new(x, 3.1, z), Vec3::new(1.0, 0.62, 0.32), 60.0, 14.0);
        lamp.cast_shadow = k == 1;
        scene.add(SceneNode::Light(Light::Point(lamp)));
    }
    // cool fill from the dusk sky
    scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.3, -1.0, 0.2), Vec3::new(0.55, 0.7, 1.0), 1.5))));

    let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
        grid: FroxelGridOptions { far: 150.0, temporal: true, ..Default::default() },
        base_density: 0.018,
        height_falloff: 0.08,
        anisotropy: 0.5,
        ambient: Vec3::new(1.4, 1.7, 2.4),
        ..Default::default()
    });
    fog.set_point_shadows(renderer.cubemap_shadow_map());
    fog.update_lights(scene.lights());

    let tonemapper = match param("tm").as_deref() {
        Some("agx") => ToneMapper::AgX,
        Some("punchy") => ToneMapper::AgXPunchy,
        Some("neutral") => ToneMapper::KhronosNeutral,
        Some("unreal") => ToneMapper::UnrealFilmic,
        Some("none") => ToneMapper::None,
        _ => ToneMapper::AcesFitted,
    };
    let film = flag("film", true);
    let ev100 = param_or("ev", 3.9f32);
    let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
    options.tonemapper = tonemapper;
    options.exposure = exposure_from_ev100(ev100);
    if film {
        // the Unreal intro's lens and film settings
        options.vignette = 0.5;
        options.grain = 0.22;
        options.chromatic_aberration = 0.25;
    }
    options.local_exposure = param("le").and_then(|v| v.parse().ok()).map(|c: f32| LocalExposure::unreal(c, c));
    let tonemap = ToneMapEffect::new(options);

    let mut effects: Vec<Box<dyn kansei_core::postprocessing::PostProcessingEffect>> = vec![Box::new(fog)];
    if flag("bloom", true) {
        effects.push(Box::new(
            BloomEffect::new(BloomOptions {
                threshold: 0.0, // physically based: every light scatters a little
                intensity: 0.06,
                ..Default::default()
            })
            .with_exposure(tonemap.total_exposure()),
        ));
    }
    effects.push(Box::new(tonemap));
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(45.0, 0.1, 400.0, canvas.aspect());
    camera.update_projection_matrix();

    log::info!("Kansei — Film Look (WASM) ready: {tonemapper:?}, EV100 {ev100}, film {film}, surface {:?}", renderer.presentation_format());

    let frozen_t: Option<f32> = param("t").and_then(|v| v.parse().ok());
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let clock = frame.time as f32;
        let t = frozen_t.unwrap_or(clock);

        if let Some(fog) = volume.effects[0].as_any_mut().downcast_mut::<VolumetricFogEffect>() {
            fog.time = clock;
        }

        // walk slowly down the path toward the lamps
        let z = 26.0 - (t * 0.6) % 20.0;
        camera.set_position(0.4 * (t * 0.2).sin(), 1.7, z);
        camera.look_at(&Vec3::new(0.0, 2.2, z - 20.0));

        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
    });
    Ok(())
}
