//! Spot lights: a car with the Midsommar intro's headlights (22 000 cd, 10°/30° cones, 70 m)
//! shines into an instanced forest at night. The trunks shadow the beams on the ground and in
//! the volumetric fog (spot shadow atlas + cone injection); surfaces use the stock GGX material
//! (`Material::standard_lit`), lit by the scene's spot lights and their shadows.
//! Exposure is the intro's EV100 3.9, through ToneMapEffect.
//!
//! URL parameters: `cam=front|behind|top|wall`, `cull=main` (CPU-cull the trunks to the camera
//! only, the bug GPU per-view culling avoids), `drive=1`, `t=<seconds>` (freeze), `shadows=0`,
//! `fog=0`, `stats=1` (log the renderer's profile, each pass's GPU time and the CPU sections, and the
//! frame interval, which is the GPU time when the browser runs without vsync), `casters=<n>` (n more
//! renderables), `lamps=<n>` (n small downlights), `clusters=0` (every light at every pixel),
//! `shafts=<steps>` (the beams raymarched per pixel with that many samples per light, instead of
//! in the fog's froxels).

use wasm_bindgen::prelude::*;

use kansei_core::buffers::{BufferType, ComputeBuffer};
use kansei_core::cameras::Camera;
use kansei_core::culling::{frustum_planes, InstanceCulling};
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::geometries::{BoxGeometry, InstancedGeometry, PlaneGeometry};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardInstancing, StandardLitOptions};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, BloomEffect, BloomOptions, SpotScattering, ToneMapEffect, ToneMapOptions, VolumetricFogEffect, VolumetricFogOptions},
};
use kansei_core::renderers::RendererConfig;
use kansei_wasm::{flag, param, param_or, Canvas};

/// GGX surface lit by the spot lights plus a dim hemispherical night sky (0.25 cd/m² above, the
/// ground bouncing a tenth of it): the stock standard material, instanced for the trunks (a vec4
/// per trunk: xyz offset, w height scale).
fn lit_material(label: &str, base_color: [f32; 3], roughness: f32, instanced: bool) -> Material {
    Material::standard_lit(label, &StandardLitOptions {
        base_color,
        roughness,
        sky_up: [0.15, 0.2, 0.3],
        sky_down: [0.02, 0.025, 0.03],
        instancing: instanced.then_some(StandardInstancing::OffsetHeight),
        ..Default::default()
    })
}

/// Deterministic 0..1 hash.
fn hash(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0
}

struct Car {
    body: usize,
    lenses: [usize; 2],
    lights: [usize; 2],
}

/// Car position along Z: parked at the forest edge, or creeping forward and back with `drive=1`.
fn car_z(t: f32, drive: bool) -> f32 {
    if drive { 2.0 * (t * 0.3).sin() } else { 0.0 }
}

fn place_car(scene: &mut Scene, car: &Car, z: f32) {
    if let Some(r) = scene.get_renderable_mut(car.body) {
        r.object.set_position(1.75, 0.75, z);
    }
    for (k, side) in [-0.62f32, 0.62].into_iter().enumerate() {
        let lamp = Vec3::new(1.75 + side, 0.7, z - 2.3);
        if let Some(r) = scene.get_renderable_mut(car.lenses[k]) {
            r.object.set_position(lamp.x, lamp.y, lamp.z - 0.02);
        }
        if let Some(Light::Spot(s)) = scene.get_light_mut(car.lights[k]) {
            s.position = lamp;
            // dipped beams: aimed at the ground 25 m ahead, spreading slightly apart
            s.look_at(Vec3::new(lamp.x + side * 2.0, 0.0, lamp.z - 25.0));
        }
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    let shadows = flag("shadows", true);
    if shadows {
        renderer.enable_spot_shadows(1024, 2);
    }

    let mut scene = Scene::new();

    let mut ground = Renderable::new(PlaneGeometry::new(200.0, 200.0), lit_material("Ground", [0.12, 0.11, 0.09], 0.9, false));
    ground.object.rotation.x = -std::f32::consts::FRAC_PI_2;
    scene.add(SceneNode::Renderable(ground));

    // the forest: one instanced draw of unit trunks, everywhere but a small clearing around the
    // car, which stands at the forest edge with its lights into the trees
    let cam = param("cam").unwrap_or_default();
    // `cam=wall`: a corridor to a pale wall 20 m ahead, and a row of poles right by the lamps
    // that the camera (between them and the wall) doesn't see; their shadows on the wall show
    // that shadow casters are culled per light, not to the camera
    let wall_test = cam == "wall";
    let mut trunks: Vec<f32> = Vec::new();
    for i in 0..1400u32 {
        let x = -45.0 + hash(i) * 90.0;
        let z = -90.0 + hash(i + 13) * 110.0;
        if (x - 1.75).abs() < 3.5 && z > -9.0 || wall_test && (x - 1.75).abs() < 9.0 && z > -21.0 {
            continue;
        }
        let h = 9.0 + hash(i + 29) * 8.0;
        trunks.extend_from_slice(&[x, h * 0.5, z, h]);
    }
    if wall_test {
        for k in 0..6 {
            trunks.extend_from_slice(&[-1.0 + k as f32 * 1.1, 1.5, -5.0, 3.0]);
        }
        let mut wall = Renderable::new(BoxGeometry::new(16.0, 6.0, 0.3), lit_material("Wall", [0.6, 0.6, 0.6], 0.9, false));
        wall.object.set_position(1.75, 3.0, -20.0);
        scene.add(SceneNode::Renderable(wall));
    }
    let count = trunks.len() as u32 / 4;
    // all trunks, in a buffer the GPU culls per view: the camera, and each shadowed spot light's
    // own frustum, so trunks outside the picture still shadow the beams in it
    let all_trunks = {
        use wgpu::util::DeviceExt;
        renderer.device().create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Trunks"),
            contents: bytemuck::cast_slice(&trunks),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    // `cull=main` instead culls on the CPU against the camera only (as an app would): the bug
    // this avoids, where trunks leaving the frame stop casting shadows
    let cull_main = param("cull").as_deref() == Some("main");
    let instances = ComputeBuffer::from_external("Trunks", all_trunks.clone(), BufferType::Storage).with_vertex_vec4(3);
    let mut forest = Renderable::new(
        InstancedGeometry::new(BoxGeometry::new(0.35, 1.0, 0.35), count, vec![instances]),
        lit_material("Trunks", [0.22, 0.18, 0.15], 0.8, true),
    );
    if !cull_main {
        // sphere at the trunk's middle (position.y = h/2) with radius h/2 + a margin: scale by h
        forest.instance_culling = Some(InstanceCulling::new(all_trunks.clone(), count, 16, 0, 0.55).with_radius_scale(12));
    }
    let forest = scene.add(SceneNode::Renderable(forest));

    // the car: a dark body, two lens emitters (4000 cd/m², the intro's headlight_lens) and two
    // 22 000 cd dipped beams, 10° inner / 30° outer, 70 m; both shadowed
    let body = scene.add(SceneNode::Renderable(Renderable::new(BoxGeometry::new(1.8, 1.1, 4.4), lit_material("Car", [0.05, 0.05, 0.06], 0.3, false))));
    let mut lenses = [0; 2];
    let mut lights = [0; 2];
    for k in 0..2 {
        let mut lens = Renderable::new(BoxGeometry::new(0.3, 0.15, 0.04), Material::emissive("Lens", [4000.0, 3800.0, 3400.0]));
        lens.cast_shadow = false;
        lenses[k] = scene.add(SceneNode::Renderable(lens));
        let mut beam = SpotLight::new(Vec3::ZERO, Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 0.95, 0.85), 22000.0, 70.0, 10f32.to_radians(), 30f32.to_radians());
        beam.cast_shadow = true;
        beam.source_radius = 0.08;
        lights[k] = scene.add(SceneNode::Light(Light::Spot(beam)));
    }
    let car = Car { body, lenses, lights };

    // `lamps=N`: N small unshadowed downlights over the forest, in random colours (clustered
    // light culling keeps them cheap; `clusters=0` shades every pixel with every light)
    let lamps: u32 = param_or("lamps", 0);
    for i in 0..lamps {
        let pos = Vec3::new(-30.0 + hash(i + 501) * 60.0, 2.5 + hash(i + 503) * 2.0, -70.0 + hash(i + 507) * 75.0);
        let hue = hash(i + 509) * 6.0;
        let color = Vec3::new((hue - 3.0).abs() - 1.0, 2.0 - (hue - 2.0).abs(), 2.0 - (hue - 4.0).abs());
        let color = Vec3::new(color.x.clamp(0.0, 1.0), color.y.clamp(0.0, 1.0), color.z.clamp(0.0, 1.0));
        let mut lamp = SpotLight::new(pos, Vec3::new(0.0, -1.0, 0.0), color, 800.0, 7.0, 25f32.to_radians(), 50f32.to_radians());
        lamp.volumetric_scale = 0.0;
        scene.add(SceneNode::Light(Light::Spot(lamp)));
    }
    renderer.set_clustered_lights(flag("clusters", true));

    // `casters=N`: N more renderables (one draw each), to measure per-draw CPU cost
    let extra: u32 = param_or("casters", 0);
    for i in 0..extra {
        let mut post = Renderable::new(BoxGeometry::new(0.2, 2.0, 0.2), lit_material("Post", [0.3, 0.3, 0.3], 0.8, false));
        post.object.set_position(-3.0 + (i % 10) as f32 * 0.9, 1.0, -12.0 - (i / 10) as f32 * 1.5);
        scene.add(SceneNode::Renderable(post));
    }

    let tonemap = {
        let mut options = ToneMapOptions::for_surface(renderer.presentation_format());
        options.exposure = exposure_from_ev100(3.9);
        options.vignette = 0.5;
        options.grain = 0.22;
        options.chromatic_aberration = 0.25;
        ToneMapEffect::new(options)
    };
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if flag("fog", true) {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 150.0, grid_d: 96, temporal: true, ..Default::default() },
            base_density: 0.015,
            height_falloff: 0.05,
            anisotropy: 0.3,
            ambient: Vec3::new(0.25, 0.32, 0.45),
            spot_scattering: match param("shafts").and_then(|v| v.trim().parse().ok()) {
                Some(steps) if steps > 0 => SpotScattering::Raymarched { steps },
                _ => SpotScattering::Froxels,
            },
            ..Default::default()
        });
        fog.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(BloomEffect::new(BloomOptions { threshold: 0.0, intensity: 0.05, ..Default::default() }).with_exposure(tonemap.total_exposure())));
    effects.push(Box::new(tonemap));
    let mut volume = PostProcessingVolume::new(&renderer, effects);

    let mut camera = Camera::new(40.0, 0.1, 400.0, canvas.aspect());
    camera.update_projection_matrix();

    log::info!("Kansei — Spot Lights (WASM) ready: {count} instanced trunks, shadows {shadows}");

    let frozen_t: Option<f32> = param("t").and_then(|v| v.trim().parse().ok());
    let drive = flag("drive", false);
    // `cull=main`: (trunks, their buffer, scene index) to cull on the CPU against the camera
    let cpu_cull = cull_main.then(|| (trunks.clone(), all_trunks.clone(), forest));
    // `stats=1`: the renderer's profile (each pass's GPU time, the frame's CPU sections) and the
    // frame interval (GPU-bound without vsync), over windows of 240 frames
    let mut stats = flag("stats", false).then(|| (0u32, kansei_wasm::now()));
    renderer.set_profiling(stats.is_some());
    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        let clock = frame.time as f32;
        let t = frozen_t.unwrap_or(clock);

        if let Some(fog) = volume.effect_mut::<VolumetricFogEffect>() {
            fog.time = clock;
        }

        let z = car_z(t, drive);
        place_car(&mut scene, &car, z);
        match cam.as_str() {
            // high behind the car, looking down the beams into the forest
            "behind" => {
                camera.set_position(1.75, 5.0, z + 10.0);
                camera.look_at(&Vec3::new(1.75, 1.0, z - 20.0));
            }
            // facing the wall, with the poles behind the camera
            "wall" => {
                camera.set_position(1.75, 2.2, z - 9.0);
                camera.look_at(&Vec3::new(1.75, 2.0, z - 20.0));
            }
            // above the canopy, looking down on the beams through the trees
            "top" => {
                camera.set_position(-10.0, 24.0, z + 6.0);
                camera.look_at(&Vec3::new(2.0, 0.0, z - 16.0));
            }
            // low, facing the lamps from inside the forest: glare, beams and the trunks'
            // shadows radiating from the car
            _ => {
                camera.set_position(-1.0, 1.8, z - 25.0);
                camera.look_at(&Vec3::new(1.75, 1.2, z));
            }
        }

        if let Some((all, buffer, forest)) = &cpu_cull {
            camera.update_view_matrix();
            let planes = frustum_planes(camera.projection_matrix.to_glam() * camera.view_matrix.to_glam());
            let visible: Vec<f32> = all
                .chunks_exact(4)
                .filter(|t| planes.iter().all(|p| p.x * t[0] + p.y * t[1] + p.z * t[2] + p.w >= -t[3] * 0.55))
                .flatten()
                .copied()
                .collect();
            renderer.queue().write_buffer(buffer, 0, bytemuck::cast_slice(&visible));
            if let Some(r) = scene.get_renderable_mut(*forest) {
                r.geometry.instance_count = visible.len() as u32 / 4;
            }
            renderer.invalidate_bundle();
        }
        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
        if let Some((frames, window_start)) = stats.as_mut() {
            *frames += 1;
            if *frames == 240 {
                let now = kansei_wasm::now();
                log::info!("{:.2} ms between frames\n{}", (now - *window_start) * 1000.0 / 240.0, renderer.take_profile().report());
                *frames = 0;
                *window_start = now;
            }
        }
    });
    Ok(())
}
