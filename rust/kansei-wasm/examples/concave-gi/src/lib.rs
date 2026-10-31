//! Concave GI: a U channel, an arch and a bowl of wood on a desk, a few centimetres each, under a
//! warm key spot light and a studio HDRI, to compare a mid tier's diffuse GI on hollows smaller
//! than its voxels resolve well (1.35 cm voxels, the volume's `Medium` over the desk's 1.3 m).
//!
//! - `gi=probes`: the voxels' irradiance probes (`SdfProbes`, one every 4 voxels) read by
//!   `VoxelGIEffect::set_probes`, with the distance field's AO at 0.8 on their light. The probes
//!   hold the light where they sit, up to a spacing off the surface: in a hollow narrower than
//!   that they see more sky than its floor does, and the AO, which can't tell what they already
//!   see, darkens what they do see a second time.
//! - `gi=rt` (the default): the hybrid RT GI (`RtDiffuseGiEffect`), rays through the grid of the
//!   pieces' triangles (1.5 cm cells), the voxels for the hits' further bounces and past 12 cm,
//!   at `res=half` (a ray for each 2 x 2 pixels) or `res=quarter` (one for each 4 x 4, a
//!   sixteenth of the rays: the default, what a mid tier can afford).
//!
//! The materials have no sky ambient of their own: the GI brings the sky, so where it is wrong
//! the image shows it. `kansei-native`'s `concave_gi` example renders the same scene headless and
//! measures each GI against the half-resolution RT GI.
//!
//! URL parameters: `gi=probes|rt`, `res=half|quarter`, `cam=wide|channel|arch|bowl`,
//! `view=lit|indirect` (the GI's light alone), `sdf_ao=0..1` (the probes' AO, 0.8),
//! `spacing=` (voxels between probes, 4), the RT GI's `cone_steps=`, `cone_tan=` (the voxel cone
//! past 12 cm), `hit_cone_steps=` and `rt_shadows=rays|maps` (its hits' shadows), `env=hdri|gradient`,
//! `ev=` (EV100, 3.2), `perf=1` (the performance overlay, each pass's GPU time). Drag to orbit,
//! wheel to zoom, right-drag to pan.

use wasm_bindgen::prelude::*;

use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::environment::{EnvironmentLighting, GradientEnvironment, HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings, HdrImage};
use kansei_core::geometries::{BoxGeometry, Geometry, Vertex};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfProbeOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    effects::{exposure_from_ev100, EnvironmentBackgroundEffect, ToneMapEffect, ToneMapOptions},
    PostProcessingEffect, PostProcessingVolume,
};
use kansei_core::renderers::RendererConfig;
use kansei_core::rt::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiResolution, RtGiShadows, RtGiView, RtGridOptions, RtSurface, SceneRtGridOptions};
use kansei_wasm::{fetch_bytes, param, param_or, Canvas};

/// The pieces' wood and the desk's (linear).
const WOOD: [f32; 3] = [0.62, 0.38, 0.22];
const DESK: [f32; 3] = [0.42, 0.36, 0.30];
/// The voxel volume's and the grid's box: the desk and a little above.
const BOX_MIN: [f32; 3] = [-0.65, -0.05, -0.4];
const BOX_MAX: [f32; 3] = [0.65, 0.35, 0.4];
const GRID_CELL: f32 = 0.015;
/// gi-box's studio HDRI: where the site serves it beside this page, then where it sits in the
/// source tree (the examples' folder served), and its stops.
const HDRI: [&str; 2] = ["../gi-box/assets/studio_small_09_1k.hdr", "../../gi-box/www/assets/studio_small_09_1k.hdr"];
const HDRI_EV: f32 = 4.0;

/// The camera presets: (name, target, distance, azimuth, elevation).
const CAMERAS: [(&str, [f32; 3], f32, f32, f32); 4] = [
    ("wide", [0.0, 0.02, 0.0], 0.85, 0.0, 0.49),
    ("channel", [-0.38, 0.0, 0.0], 0.45, 0.0, 0.36),
    ("arch", [0.0, 0.05, 0.0], 0.36, 0.0, 0.14),
    ("bowl", [0.38, 0.0, 0.0], 0.41, 0.0, 0.75),
];

/// A mesh under construction: quads with their normal, wound to face it.
#[derive(Default)]
struct Mesh {
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
}

impl Mesh {
    /// The quad p0 p1 p2 p3 (in order round it), facing `n` (each corner its own normal when
    /// `normals` is given, for smooth bands).
    fn quad(&mut self, p: [glam::Vec3; 4], n: glam::Vec3, normals: Option<[glam::Vec3; 4]>) {
        let base = self.vertices.len() as u32;
        for (q, m) in p.iter().zip(normals.unwrap_or([n; 4])) {
            self.vertices.push(Vertex { position: [q.x, q.y, q.z, 1.0], normal: m.to_array(), uv: [0.0, 0.0] });
        }
        let geometric = (p[1] - p[0]).cross(p[2] - p[0]) + (p[2] - p[0]).cross(p[3] - p[0]);
        if geometric.dot(n) >= 0.0 {
            self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            self.indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }
}

/// A block `w` x `h` x `d` (x, y, z) standing on y = 0, with an arched tunnel along z through it:
/// `opening` wide, its sides straight up to `spring`, then a half circle.
fn arch(w: f32, h: f32, d: f32, opening: f32, spring: f32) -> Geometry {
    use glam::{vec2, vec3, Vec2};
    let a = opening * 0.5;
    let n = 64;
    // the opening's outline and the block's, left foot to right foot, by arc length
    let along = |total: f32, at: &dyn Fn(f32) -> Vec2| -> Vec<Vec2> { (0..=n).map(|i| at(i as f32 / n as f32 * total)).collect() };
    let arc = std::f32::consts::PI * a;
    let inner = along(2.0 * spring + arc, &|s| {
        if s <= spring {
            vec2(-a, s)
        } else if s <= spring + arc {
            let phi = std::f32::consts::PI * (1.0 - (s - spring) / arc);
            vec2(a * phi.cos(), spring + a * phi.sin())
        } else {
            vec2(a, spring - (s - spring - arc))
        }
    });
    let outer = along(2.0 * h + w, &|s| {
        if s <= h {
            vec2(-w * 0.5, s)
        } else if s <= h + w {
            vec2(-w * 0.5 + (s - h), h)
        } else {
            vec2(w * 0.5, h - (s - h - w))
        }
    });
    let mut m = Mesh::default();
    let (zf, zb) = (d * 0.5, -d * 0.5);
    for i in 0..n {
        let (i0, i1, o0, o1) = (inner[i], inner[i + 1], outer[i], outer[i + 1]);
        // the front and back faces, the strip between the outlines
        m.quad([vec3(i0.x, i0.y, zf), vec3(o0.x, o0.y, zf), vec3(o1.x, o1.y, zf), vec3(i1.x, i1.y, zf)], glam::Vec3::Z, None);
        m.quad([vec3(i0.x, i0.y, zb), vec3(o0.x, o0.y, zb), vec3(o1.x, o1.y, zb), vec3(i1.x, i1.y, zb)], -glam::Vec3::Z, None);
        // the tunnel's walls and vault (facing into it) and the block's sides and top
        let mid = (i0 + i1) * 0.5;
        let into = if mid.y > spring + 1e-4 { (vec2(0.0, spring) - mid).normalize() } else { vec2(-mid.x.signum(), 0.0) };
        m.quad([vec3(i0.x, i0.y, zf), vec3(i1.x, i1.y, zf), vec3(i1.x, i1.y, zb), vec3(i0.x, i0.y, zb)], vec3(into.x, into.y, 0.0), None);
        let t = (o1 - o0).normalize_or_zero();
        let out = vec2(-t.y, t.x);
        let out = if out.dot((o0 + o1) * 0.5 - vec2(0.0, h * 0.5)) < 0.0 { -out } else { out };
        m.quad([vec3(o0.x, o0.y, zf), vec3(o1.x, o1.y, zf), vec3(o1.x, o1.y, zb), vec3(o0.x, o0.y, zb)], vec3(out.x, out.y, 0.0), None);
    }
    // the feet
    for (x0, x1) in [(-w * 0.5, -a), (a, w * 0.5)] {
        m.quad([vec3(x0, 0.0, zf), vec3(x1, 0.0, zf), vec3(x1, 0.0, zb), vec3(x0, 0.0, zb)], -glam::Vec3::Y, None);
    }
    Geometry::new("Arch", m.vertices, m.indices)
}

/// A round block `radius` wide and `height` tall standing on y = 0, hollowed into a bowl: a
/// spherical cap `depth` deep opening `rim` short of its edge.
fn bowl(radius: f32, height: f32, depth: f32, rim: f32) -> Geometry {
    use glam::vec2;
    let a = radius - rim;
    let rs = (a * a + depth * depth) / (2.0 * depth);
    let bottom = height - depth;
    let rings = 24;
    let theta_max = (a / rs).asin();
    // the profile (r, y): the hollow from its centre out and up, the rim, the side down, the base in
    let mut profile: Vec<glam::Vec2> = (0..=rings).map(|i| i as f32 / rings as f32 * theta_max).map(|t| vec2(rs * t.sin(), bottom + rs * (1.0 - t.cos()))).collect();
    profile.extend([vec2(radius, height), vec2(radius, 0.0), vec2(0.0, 0.0)]);
    let segments = 72;
    let turn = |p: glam::Vec2, k: usize| {
        let phi = k as f32 / segments as f32 * std::f32::consts::TAU;
        glam::Vec3::new(p.x * phi.cos(), p.y, p.x * phi.sin())
    };
    let mut m = Mesh::default();
    for j in 0..profile.len() - 1 {
        let (p0, p1) = (profile[j], profile[j + 1]);
        let dir = p1 - p0;
        // walking out along the hollow, over the rim, down the side and in along the base, the
        // surface faces (-dy, dr); the hollow's normals smooth (the sphere's), the rest flat
        let flat = vec2(-dir.y, dir.x).normalize();
        let sphere = |p: glam::Vec2| (vec2(0.0, bottom + rs) - p).normalize();
        let (na, nb) = if j < rings { (sphere(p0), sphere(p1)) } else { (flat, flat) };
        for k in 0..segments {
            let ns = [turn(na, k), turn(nb, k), turn(nb, k + 1), turn(na, k + 1)];
            m.quad([turn(p0, k), turn(p1, k), turn(p1, k + 1), turn(p0, k + 1)], (ns[0] + ns[1] + ns[2] + ns[3]).normalize_or_zero(), Some(ns));
        }
    }
    Geometry::new("Bowl", m.vertices, m.indices)
}

fn piece(scene: &mut Scene, geometry: Geometry, at: [f32; 3], color: [f32; 3]) {
    let material = Material::standard_lit("Wood", &StandardLitOptions { base_color: color, roughness: 0.8, ..Default::default() });
    let mut r = Renderable::new(geometry, material).with_gi(GiSurface::new(color)).with_rt(RtSurface::new(color));
    r.object.set_position(at[0], at[1], at[2]);
    scene.add(SceneNode::Renderable(r));
}

/// The desk, the pieces and the warm key light.
fn scene() -> Scene {
    let mut scene = Scene::new();
    piece(&mut scene, BoxGeometry::new(1.3, 0.03, 0.8), [0.0, -0.015, 0.0], DESK);
    // a U channel along z: its floor and two walls
    let (cx, inner, wall, tall, len) = (-0.38, 0.056, 0.012, 0.045, 0.4);
    piece(&mut scene, BoxGeometry::new(inner + 2.0 * wall, 0.012, len), [cx, 0.006, 0.0], WOOD);
    for side in [-1.0, 1.0] {
        piece(&mut scene, BoxGeometry::new(wall, tall, len), [cx + side * (inner + wall) * 0.5, tall * 0.5, 0.0], WOOD);
    }
    piece(&mut scene, arch(0.16, 0.12, 0.1, 0.07, 0.05), [0.0, 0.0, 0.0], WOOD);
    piece(&mut scene, bowl(0.11, 0.06, 0.04, 0.012), [0.38, 0.0, 0.0], WOOD);
    let mut key = SpotLight::new(Vec3::new(0.45, 0.75, 0.55), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.72, 0.45), 9.0, 4.0, 40f32.to_radians(), 60f32.to_radians());
    key.look_at(Vec3::new(0.0, 0.0, 0.0));
    key.cast_shadow = true;
    key.source_radius = 0.03;
    scene.add(SceneNode::Light(Light::Spot(key)));
    scene
}

/// The studio HDRI, fetched from the first place that has it (None: the gradient stands in).
async fn hdri(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<HdrEnvironment> {
    for url in HDRI {
        let Ok(bytes) = fetch_bytes(url).await else { continue };
        let Ok(image) = HdrImage::decode(&bytes) else { continue };
        let mut env = HdrEnvironment::new(device, queue, &image, HdrEnvironmentOptions::default());
        env.set_settings(HdrEnvironmentSettings { intensity: HDRI_EV.exp2(), ..Default::default() });
        env.prepare(device, queue);
        return Some(env);
    }
    log::warn!("the studio HDRI is at none of {HDRI:?}: a studio gradient stands in");
    None
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, ..Default::default() }).await;
    renderer.enable_spot_shadows(2048, 1);
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: BOX_MIN, bounds_max: BOX_MAX, ..Default::default() });
    let extent = glam::Vec3::from(BOX_MAX) - glam::Vec3::from(BOX_MIN);
    let dims = (extent / GRID_CELL / 4.0).ceil().as_uvec3() * 4;
    renderer.enable_rt_grid(SceneRtGridOptions {
        grid: RtGridOptions { dims: dims.to_array(), cell: GRID_CELL, fixed_origin: Some(glam::Vec3::from(BOX_MIN)), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    });

    let gi = param("gi").unwrap_or_else(|| "rt".into());
    let indirect = param("view").as_deref() == Some("indirect");
    let (device, queue) = (renderer.device().clone(), renderer.queue().clone());
    let mut hdri = if param("env").as_deref() == Some("gradient") { None } else { hdri(&device, &queue).await };
    let gradient = GradientEnvironment::new(&device, [20.0, 20.0, 21.0], [6.0, 5.8, 5.6]);
    let lighting: EnvironmentLighting = hdri.as_ref().map_or_else(|| gradient.lighting(), |h| h.lighting());
    renderer.set_environment(Some(&lighting));

    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    if lighting.environment.is_some() {
        effects.push(Box::new(EnvironmentBackgroundEffect::new(&lighting)));
    }
    let scene_gi = renderer.voxel_gi().unwrap();
    if gi == "probes" {
        let mut e = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), material_ambient: 0.0, sdf_ao: param_or("sdf_ao", 0.8f32).clamp(0.0, 1.0), ..Default::default() });
        e.show_indirect = indirect;
        effects.push(Box::new(e));
    } else {
        let resolution = RtGiResolution::from_name(&param("res").unwrap_or_default()).unwrap_or(RtGiResolution::Quarter);
        let defaults = RtDiffuseGiOptions::default();
        let mut e = RtDiffuseGiEffect::with_volume(
            scene_gi.volume(),
            renderer.rt_grid().unwrap().handle(),
            RtDiffuseGiOptions {
                max_distance: 3.0,
                near_distance: 0.12,
                ambient: 0.0,
                resolution,
                cone_steps: param_or("cone_steps", defaults.cone_steps),
                cone_tan: param_or("cone_tan", defaults.cone_tan),
                hit_cone_steps: param_or("hit_cone_steps", defaults.hit_cone_steps),
                shadows: RtGiShadows::from_name(&param("rt_shadows").unwrap_or_default()).unwrap_or(defaults.shadows),
                ..defaults
            },
        );
        e.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
        if indirect {
            e.view = RtGiView::Indirect;
        }
        effects.push(Box::new(e));
    }
    for e in &mut effects {
        e.set_environment(&lighting);
    }
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    tone.exposure = exposure_from_ev100(param_or("ev", 3.2f32));
    effects.push(Box::new(ToneMapEffect::new(tone)));
    let mut volume = PostProcessingVolume::new(&renderer, effects);
    if gi == "probes" {
        let spacing = param_or("spacing", 4.0f32).max(1.0);
        let scene_gi = renderer.voxel_gi_mut().unwrap();
        scene_gi.enable_probes(SdfProbeOptions { spacing_voxels: spacing, ..Default::default() });
        let (probes, sdf) = (scene_gi.probes(), scene_gi.sdf());
        if let Some(e) = volume.effect_mut::<VoxelGIEffect>() {
            e.set_probes(probes);
            e.set_sdf(sdf);
        }
    }

    let mut scene = scene();
    let mut camera = Camera::new(40.0, 0.01, 20.0, canvas.aspect());
    let (_, target, distance, azimuth, elevation) = CAMERAS.iter().find(|c| Some(c.0) == param("cam").as_deref()).copied().unwrap_or(CAMERAS[0]);
    let target = Vec3::new(target[0], target[1], target[2]);
    let mut controls = CameraControls::from_canvas(canvas.element(), target, distance).with_mouse_pan(canvas.element());
    controls.set_view(target, distance, azimuth, elevation);
    log::info!("Kansei — Concave GI (WASM): gi={gi}, HDRI {}", if hdri.is_some() { "loaded" } else { "missing" });

    kansei_wasm::run(&canvas, move |frame| {
        frame.resize(&mut renderer, &mut camera);
        controls.update(&mut camera, frame.dt);
        if let Some(h) = &mut hdri {
            h.prepare(renderer.device(), renderer.queue());
        }
        if let Some(e) = volume.effect_mut::<RtDiffuseGiEffect>() {
            e.update_lights(scene.lights());
        }
        renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
        kansei_wasm::perf::profile(&mut renderer);
    });
    Ok(())
}
