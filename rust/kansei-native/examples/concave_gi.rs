//! A mid tier's diffuse GI on concave pieces, headless: a U channel, an arch and a bowl of wood on
//! a desk (`kansei-wasm`'s `concave-gi` example's scene), under a warm key spot light and the studio
//! HDRI (gi-box's `studio_small_09_1k.hdr`), the scene's voxel volume at `Medium` (96 voxels over
//! its 1.3 m: 1.35 cm) and the grid of triangles in 1.5 cm cells. The pieces' hollows are a few
//! voxels across.
//!
//! - `probes`: the voxels' irradiance probes (`gi::SdfProbes` read by `VoxelGIEffect::set_probes`)
//!   every 4 voxels, the distance field's AO on their light (`--sdf-ao`, 0.8);
//! - `voxel`: the voxels' cones per pixel, under the same AO;
//! - `rt`: the hybrid RT GI (`rt::RtDiffuseGiEffect`: a ray for each 2 x 2 pixels through the grid,
//!   the voxels past 12 cm and for the hits' further bounces, SVGF), what the others are measured
//!   against;
//! - `rt-quarter`: the same at a quarter of the resolution (`RtGiResolution::Quarter`, a ray for
//!   each 4 x 4 pixels);
//! - `reference`: `RtGiMode::Reference` accumulated (a path tracer through the grid), to check
//!   `rt`.
//!
//! The materials have no sky ambient: the GI brings the sky. Each view renders every mode from a
//! fresh start (the voxels' light and the probes settle over the first frames), then averages the
//! last frames' linear output (`--average`). It prints the GI's passes' GPU time, the mean colour of
//! each patch (the pixels of one face of a piece) per mode and its luminance's ratio to `rt`, and
//! writes each frame (exposed, `1 - exp(-x)`, sRGB) as PPM with the patches outlined. Debug views:
//! `--indirect` (the GI's light alone), `--probe-view`, `--voxel-view`, `--slice <m>` (the distance
//! field).
//!
//! ```text
//! cargo run --release -p kansei-native --example concave_gi -- [--modes probes,rt-quarter,rt] [--views all|wide,channel,arch,bowl] [--size WxH] [--frames N] [--average N] [--out DIR] [--spacing VOXELS] [--sdf-ao 0..1]
//! ```

use kansei_core::cameras::Camera;
use kansei_core::environment::{EnvironmentLighting, HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings, HdrImage};
use kansei_core::geometries::{BoxGeometry, Geometry, Vertex};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfProbeOptions, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality};
use kansei_core::lights::{Light, SpotLight};
use kansei_core::materials::{Material, StandardLitOptions};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::effects::EnvironmentBackgroundEffect;
use kansei_core::postprocessing::PostProcessingEffect;
use kansei_core::renderers::{GBuffer, Renderer, RendererConfig};
use kansei_core::rt::{RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiMode, RtGiResolution, RtGiView, RtGridOptions, RtSurface, SceneRtGridOptions};

/// The pieces' wood and the desk's (linear).
const WOOD: [f32; 3] = [0.62, 0.38, 0.22];
const DESK: [f32; 3] = [0.42, 0.36, 0.30];
/// The voxel volume's and the grid's box.
const BOX_MIN: [f32; 3] = [-0.65, -0.05, -0.4];
const BOX_MAX: [f32; 3] = [0.65, 0.35, 0.4];
const GRID_CELL: f32 = 0.015;
/// The HDRI's stops, and the exposure the images are written at.
const HDRI_EV: f32 = 4.0;
const EXPOSURE: f32 = 0.09;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Probes,
    Voxel,
    Rt,
    RtQuarter,
    Reference,
}

impl Mode {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "probes" => Some(Mode::Probes),
            "voxel" => Some(Mode::Voxel),
            "rt" => Some(Mode::Rt),
            "rt-quarter" => Some(Mode::RtQuarter),
            "reference" => Some(Mode::Reference),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Mode::Probes => "probes",
            Mode::Voxel => "voxel",
            Mode::Rt => "rt",
            Mode::RtQuarter => "rt-quarter",
            Mode::Reference => "reference",
        }
    }
}

/// A view: its name, the camera's position and target, and its patches (name, pixel rectangle
/// as fractions of the frame: x0, y0, x1, y1).
struct View {
    name: &'static str,
    eye: [f32; 3],
    target: [f32; 3],
    patches: &'static [(&'static str, [f32; 4])],
}

const VIEWS: [View; 4] = [
    View { name: "wide", eye: [0.0, 0.42, 0.75], target: [0.0, 0.02, 0.0], patches: &[] },
    View {
        name: "channel",
        eye: [-0.38, 0.16, 0.42],
        target: [-0.38, 0.0, 0.0],
        patches: &[
            ("channel floor", [0.45, 0.6, 0.52, 0.78]),
            ("channel inner wall", [0.545, 0.5, 0.565, 0.72]),
            ("channel end faces", [0.38, 0.68, 0.405, 0.82]),
            ("desk beside channel", [0.3, 0.75, 0.36, 0.85]),
            ("desk, open", [0.1, 0.85, 0.25, 0.95]),
        ],
    },
    View {
        name: "arch",
        eye: [0.0, 0.1, 0.36],
        target: [0.0, 0.05, 0.0],
        patches: &[
            ("arch face, top", [0.33, 0.25, 0.4, 0.35]),
            ("arch face, legs", [0.31, 0.55, 0.4, 0.7]),
            ("tunnel walls", [0.415, 0.5, 0.43, 0.65]),
            ("desk under arch", [0.45, 0.62, 0.55, 0.7]),
            ("desk at arch foot", [0.35, 0.75, 0.65, 0.8]),
            ("desk, open", [0.05, 0.85, 0.2, 0.95]),
        ],
    },
    View {
        name: "bowl",
        eye: [0.38, 0.28, 0.3],
        target: [0.38, 0.0, 0.0],
        patches: &[
            ("bowl hollow, far wall", [0.4, 0.2, 0.6, 0.28]),
            ("bowl hollow, floor", [0.4, 0.38, 0.6, 0.5]),
            ("bowl outer side", [0.4, 0.68, 0.6, 0.76]),
            ("desk at bowl foot", [0.4, 0.82, 0.6, 0.88]),
            ("desk beside bowl", [0.15, 0.55, 0.25, 0.7]),
            ("desk, open", [0.05, 0.85, 0.15, 0.95]),
        ],
    },
];

struct Args {
    modes: Vec<Mode>,
    views: Vec<&'static View>,
    size: (u32, u32),
    frames: usize,
    out: String,
    spacing: f32,
    /// The distance field's AO on the probes' light, and the debug views: the GI's light alone,
    /// the probes as balls.
    sdf_ao: f32,
    indirect: bool,
    probe_view: bool,
    /// Debug views of the voxels: the lit voxels, or the distance field's slice at a height.
    voxel_view: bool,
    slice: Option<f32>,
    /// The last frames averaged into the images and the patches (1: the last frame as it is).
    average: usize,
}

fn args() -> Args {
    let mut a = Args { modes: vec![Mode::Probes, Mode::Rt], views: VIEWS.iter().collect(), size: (960, 540), frames: 240, out: ".".into(), spacing: 4.0, sdf_ao: 0.8, indirect: false, probe_view: false, voxel_view: false, slice: None, average: 16 };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut value = || it.next().unwrap_or_default();
        match arg.as_str() {
            "--modes" => a.modes = value().split(',').filter_map(Mode::from_name).collect(),
            "--views" => {
                let v = value();
                if v != "all" {
                    a.views = VIEWS.iter().filter(|view| v.split(',').any(|n| n == view.name)).collect();
                }
            }
            "--frames" => a.frames = value().parse().unwrap_or(a.frames),
            "--out" => a.out = value(),
            "--spacing" => a.spacing = value().parse().unwrap_or(a.spacing),
            "--sdf-ao" => a.sdf_ao = value().parse().unwrap_or(a.sdf_ao),
            "--indirect" => a.indirect = true,
            "--probe-view" => a.probe_view = true,
            "--voxel-view" => a.voxel_view = true,
            "--slice" => a.slice = value().parse().ok(),
            "--average" => a.average = value().parse().unwrap_or(16).max(1),
            "--size" => {
                if let Some((w, h)) = value().split_once('x').map(|(w, h)| (w.parse().unwrap_or(960), h.parse().unwrap_or(540))) {
                    a.size = (w, h);
                }
            }
            other => panic!("unknown argument {other}"),
        }
    }
    a
}

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
        let ns = normals.unwrap_or([n; 4]);
        for (q, m) in p.iter().zip(ns) {
            self.vertices.push(Vertex { position: [q.x, q.y, q.z, 1.0], normal: m.to_array(), uv: [0.0, 0.0] });
        }
        let geometric = (p[1] - p[0]).cross(p[2] - p[0]) + (p[2] - p[0]).cross(p[3] - p[0]);
        if geometric.dot(n) >= 0.0 {
            self.indices.extend([base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            self.indices.extend([base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }

    fn geometry(self, label: &str) -> Geometry {
        Geometry::new(label, self.vertices, self.indices)
    }
}

/// A block `w` x `h` x `d` (x, y, z) standing on y = 0, with an arched tunnel along z through it:
/// `opening` wide, its sides straight up to `spring`, then a half circle.
fn arch(w: f32, h: f32, d: f32, opening: f32, spring: f32) -> Geometry {
    use glam::{vec2, vec3, Vec2};
    let a = opening * 0.5;
    let n = 64;
    // the opening's outline and the block's, left foot to right foot, by arc length
    let inner_path: Vec<Vec2> = {
        let legs = spring;
        let arc = std::f32::consts::PI * a;
        let total = 2.0 * legs + arc;
        (0..=n)
            .map(|i| {
                let s = i as f32 / n as f32 * total;
                if s <= legs {
                    vec2(-a, s)
                } else if s <= legs + arc {
                    let phi = std::f32::consts::PI * (1.0 - (s - legs) / arc);
                    vec2(a * phi.cos(), spring + a * phi.sin())
                } else {
                    vec2(a, spring - (s - legs - arc))
                }
            })
            .collect()
    };
    let outer_path: Vec<Vec2> = {
        let total = 2.0 * h + w;
        (0..=n)
            .map(|i| {
                let s = i as f32 / n as f32 * total;
                if s <= h {
                    vec2(-w * 0.5, s)
                } else if s <= h + w {
                    vec2(-w * 0.5 + (s - h), h)
                } else {
                    vec2(w * 0.5, h - (s - h - w))
                }
            })
            .collect()
    };
    let mut m = Mesh::default();
    let (zf, zb) = (d * 0.5, -d * 0.5);
    for i in 0..n {
        let (i0, i1, o0, o1) = (inner_path[i], inner_path[i + 1], outer_path[i], outer_path[i + 1]);
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
    m.geometry("Arch")
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
    let mut profile: Vec<glam::Vec2> = (0..=rings)
        .map(|i| {
            let t = i as f32 / rings as f32 * theta_max;
            vec2(rs * t.sin(), bottom + rs * (1.0 - t.cos()))
        })
        .collect();
    profile.extend([vec2(radius, height), vec2(radius, 0.0), vec2(0.0, 0.0)]);
    let segments = 72;
    let mut m = Mesh::default();
    let ring = |p: glam::Vec2, k: usize| {
        let phi = k as f32 / segments as f32 * std::f32::consts::TAU;
        glam::Vec3::new(p.x * phi.cos(), p.y, p.x * phi.sin())
    };
    let normal_at = |n2: glam::Vec2, k: usize| {
        let phi = k as f32 / segments as f32 * std::f32::consts::TAU;
        glam::Vec3::new(n2.x * phi.cos(), n2.y, n2.x * phi.sin())
    };
    for j in 0..profile.len() - 1 {
        let (p0, p1) = (profile[j], profile[j + 1]);
        let dir = p1 - p0;
        // walking out along the hollow, over the rim, down the side and in along the base, the
        // surface faces (-dy, dr)
        let n2 = vec2(-dir.y, dir.x).normalize();
        // the hollow's normals smooth (the sphere's), the rest flat
        let smooth = j < rings;
        let sphere = |p: glam::Vec2| (vec2(0.0, bottom + rs) - p).normalize();
        let (na, nb) = if smooth { (sphere(p0), sphere(p1)) } else { (n2, n2) };
        for k in 0..segments {
            let q = [ring(p0, k), ring(p1, k), ring(p1, k + 1), ring(p0, k + 1)];
            let ns = [normal_at(na, k), normal_at(nb, k), normal_at(nb, k + 1), normal_at(na, k + 1)];
            let face = (ns[0] + ns[1] + ns[2] + ns[3]).normalize_or_zero();
            m.quad(q, face, Some(ns));
        }
    }
    m.geometry("Bowl")
}

fn piece(scene: &mut Scene, geometry: Geometry, at: [f32; 3], color: [f32; 3]) {
    let material = Material::standard_lit("Wood", &StandardLitOptions { base_color: color, roughness: 0.8, ..Default::default() });
    let mut r = Renderable::new(geometry, material).with_gi(GiSurface::new(color)).with_rt(RtSurface::new(color));
    r.object.set_position(at[0], at[1], at[2]);
    scene.add(SceneNode::Renderable(r));
}

/// The desk and the pieces, under the warm key light.
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

fn effects(renderer: &Renderer, scene: &Scene, lighting: &EnvironmentLighting, mode: Mode, args: &Args) -> Vec<Box<dyn PostProcessingEffect>> {
    let gi = renderer.voxel_gi().unwrap();
    let mut out: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(EnvironmentBackgroundEffect::new(lighting))];
    match mode {
        Mode::Probes | Mode::Voxel => {
            let mut e = VoxelGIEffect::new(gi.volume(), VoxelGIOptions { quality: gi.quality(), material_ambient: 0.0, sdf_ao: args.sdf_ao, ..Default::default() });
            e.set_probes(gi.probes().filter(|_| mode != Mode::Voxel));
            e.set_sdf(gi.sdf());
            e.show_indirect = args.indirect;
            e.show_probes = args.probe_view;
            e.show_voxels = args.voxel_view;
            e.show_sdf_slice = args.slice;
            out.push(Box::new(e));
        }
        Mode::Rt | Mode::RtQuarter | Mode::Reference => {
            let grid = renderer.rt_grid().unwrap().handle();
            let mut e = RtDiffuseGiEffect::with_volume(gi.volume(), grid, RtDiffuseGiOptions { max_distance: 3.0, near_distance: 0.12, ambient: 0.0, ..Default::default() });
            e.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
            e.update_lights(scene.lights());
            if args.indirect {
                e.view = RtGiView::Indirect;
            }
            if mode == Mode::RtQuarter {
                e.set_resolution(RtGiResolution::Quarter);
            }
            if mode == Mode::Reference {
                e.mode = RtGiMode::Reference;
                e.accumulate = true;
            }
            out.push(Box::new(e));
        }
    }
    for e in &mut out {
        e.set_environment(lighting);
    }
    out
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f32;
    match exp {
        0 => sign * frac * 2f32.powi(-24),
        31 => if frac == 0.0 { sign * f32::INFINITY } else { f32::NAN },
        e => sign * (1.0 + frac / 1024.0) * 2f32.powi(e - 15),
    }
}

/// The texture's texels (rgb, linear).
fn read_rgb(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, w: u32, h: u32) -> Vec<[f32; 3]> {
    let row = (w * 8).next_multiple_of(256);
    let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: Some("ProbeGi/Readback"), size: (row * h) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(h) } },
        wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
    );
    queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, |_| {});
    let _ = device.poll(wgpu::PollType::wait_indefinitely());
    let data = slice.get_mapped_range().unwrap();
    let mut out = Vec::with_capacity((w * h) as usize);
    for y in 0..h {
        for x in 0..w {
            let o = (y * row + x * 8) as usize;
            let c = |k: usize| f16_to_f32(u16::from_le_bytes([data[o + 2 * k], data[o + 2 * k + 1]]));
            out.push([c(0), c(1), c(2)]);
        }
    }
    drop(data);
    buffer.unmap();
    out
}

/// Display value 0-255 of linear `v` at the images' exposure (`1 - exp(-x)`, sRGB).
fn display(v: f32) -> f32 {
    let x = 1.0 - (-v * EXPOSURE).exp();
    let s = if x <= 0.0031308 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).clamp(0.0, 255.0)
}

fn write_ppm(path: &str, pixels: &[[f32; 3]], w: u32, h: u32, patches: &[(&str, [f32; 4])]) {
    let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let edge = patches.iter().any(|(_, r)| {
                let inside = fx >= r[0] && fx <= r[2] && fy >= r[1] && fy <= r[3];
                let near = |a: f32, b: f32, s: u32| (a - b).abs() * s as f32 <= 1.0;
                inside && (near(fx, r[0], w) || near(fx, r[2], w) || near(fy, r[1], h) || near(fy, r[3], h))
            });
            let p = pixels[(y * w + x) as usize];
            if edge {
                bytes.extend([0u8, 255, 255]);
            } else {
                bytes.extend(p.map(|c| display(c) as u8));
            }
        }
    }
    std::fs::write(path, bytes).unwrap_or_else(|e| panic!("{path}: {e}"));
}

/// The mean of each patch: linear rgb, and the display value's rgb.
fn patch_means(pixels: &[[f32; 3]], w: u32, h: u32, rect: [f32; 4]) -> ([f32; 3], [f32; 3]) {
    let (x0, y0) = ((rect[0] * w as f32) as u32, (rect[1] * h as f32) as u32);
    let (x1, y1) = ((rect[2] * w as f32) as u32, (rect[3] * h as f32) as u32);
    let (mut lin, mut disp, mut n) = ([0.0f64; 3], [0.0f64; 3], 0.0f64);
    for y in y0..y1.min(h) {
        for x in x0..x1.min(w) {
            let p = pixels[(y * w + x) as usize];
            for k in 0..3 {
                lin[k] += p[k] as f64;
                disp[k] += display(p[k]) as f64;
            }
            n += 1.0;
        }
    }
    let n = n.max(1.0);
    (lin.map(|v| (v / n) as f32), disp.map(|v| (v / n) as f32))
}

fn main() {
    env_logger::init();
    let args = args();
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() })).expect("no adapter");
    let info = adapter.get_info();
    println!("adapter: {} ({:?})", info.name, info.backend);
    let (w, h) = args.size;
    let mut renderer = Renderer::new(RendererConfig { width: w, height: h, sample_count: 1, ..Default::default() });
    pollster::block_on(renderer.initialize_headless(&adapter));
    let (device, queue) = (renderer.device().clone(), renderer.queue().clone());
    renderer.enable_spot_shadows(2048, 1);
    renderer.enable_voxel_gi(SceneVoxelGiOptions { quality: VoxelGiQuality::Medium, bounds_min: BOX_MIN, bounds_max: BOX_MAX, ..Default::default() });
    let extent = glam::Vec3::from(BOX_MAX) - glam::Vec3::from(BOX_MIN);
    let dims = (extent / GRID_CELL / 4.0).ceil().as_uvec3() * 4;
    renderer.enable_rt_grid(SceneRtGridOptions {
        grid: RtGridOptions { dims: dims.to_array(), cell: GRID_CELL, fixed_origin: Some(glam::Vec3::from(BOX_MIN)), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    });
    let hdri = concat!(env!("CARGO_MANIFEST_DIR"), "/../kansei-wasm/examples/gi-box/www/assets/studio_small_09_1k.hdr");
    let image = HdrImage::decode(&std::fs::read(hdri).expect("the studio HDRI")).expect("decode");
    let mut env = HdrEnvironment::new(&device, &queue, &image, HdrEnvironmentOptions::default());
    env.set_settings(HdrEnvironmentSettings { intensity: HDRI_EV.exp2(), ..Default::default() });
    env.prepare(&device, &queue);
    let lighting = env.lighting();
    renderer.set_environment(Some(&lighting));
    let mut scene = scene();
    let gbuffer = GBuffer::new(&device, w, h, 1);
    let target = |label| {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    };
    let ping = [target("ProbeGi/A"), target("ProbeGi/B")];
    let views = [ping[0].create_view(&Default::default()), ping[1].create_view(&Default::default())];
    let average = args.average.min(args.frames);
    let vox = renderer.voxel_gi().unwrap().volume().voxel_size();
    println!("voxels {:.2} cm, probes every {} voxels ({:.2} cm), grid cell {:.1} cm, {w}x{h}, {} frames a view and mode (the last {average} averaged)", vox * 100.0, args.spacing, vox * args.spacing * 100.0, GRID_CELL * 100.0, args.frames);

    // (one camera for every view: the renderer keeps the first camera's bindings)
    let mut camera = Camera::new(40.0, 0.01, 20.0, w as f32 / h as f32);
    for view in &args.views {
        camera.set_position(view.eye[0], view.eye[1], view.eye[2]);
        camera.look_at(&Vec3::new(view.target[0], view.target[1], view.target[2]));
        camera.update_projection_matrix();
        let mut results: Vec<(Mode, Vec<([f32; 3], [f32; 3])>)> = Vec::new();
        for &mode in &args.modes {
            // the probes (and the field) for the probe mode only, each mode from a fresh start
            let gi = renderer.voxel_gi_mut().unwrap();
            gi.disable_probes();
            gi.disable_sdf();
            if mode == Mode::Voxel && args.sdf_ao > 0.0 {
                gi.enable_sdf();
            }
            if mode == Mode::Probes {
                gi.enable_probes(SdfProbeOptions { spacing_voxels: args.spacing, ..Default::default() });
            }
            gi.invalidate();
            let mut fx = effects(&renderer, &scene, &lighting, mode, &args);
            let mut sum = vec![[0.0f32; 3]; (w * h) as usize];
            renderer.set_profiling(true);
            for k in 0..args.frames {
                if k + 48 == args.frames {
                    let _ = renderer.take_profile();
                }
                renderer.render_scene_offscreen(&mut scene, &mut camera, &gbuffer);
                let mut encoder = device.create_command_encoder(&Default::default());
                let mut input = &gbuffer.color_view;
                let mut last = 0;
                for (e, effect) in fx.iter_mut().enumerate() {
                    effect.render(&device, &queue, &mut encoder, &gbuffer, input, &gbuffer.depth_view, &views[e % 2], &camera, w, h);
                    input = &views[e % 2];
                    last = e % 2;
                }
                queue.submit(Some(encoder.finish()));
                renderer.end_profiled_frame();
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
                if k + average >= args.frames {
                    for (s, p) in sum.iter_mut().zip(read_rgb(&device, &queue, &ping[last], w, h)) {
                        for c in 0..3 {
                            s[c] += p[c] / average as f32;
                        }
                    }
                }
            }
            // (readbacks arrive late: a few empty frames to let them in)
            for _ in 0..4 {
                renderer.end_profiled_frame();
                let _ = device.poll(wgpu::PollType::wait_indefinitely());
            }
            let profile = renderer.take_profile();
            let gi_passes: Vec<String> = profile
                .gpu
                .iter()
                .filter(|p| ["Probes/", "VoxelGI/", "RtGi/", "Sdf", "Voxel"].iter().any(|k| p.label.starts_with(k)))
                .map(|p| format!("{} {:.3}", p.label, p.busy_ms))
                .collect();
            let gi_ms: f64 = profile.gpu.iter().filter(|p| ["Probes/", "VoxelGI/", "RtGi/"].iter().any(|k| p.label.starts_with(k))).map(|p| p.busy_ms).sum();
            println!("{} {}: the GI's passes {gi_ms:.3} ms a frame ({})", view.name, mode.name(), gi_passes.join(", "));
            let path = format!("{}/{}-{}.ppm", args.out, view.name, mode.name());
            write_ppm(&path, &sum, w, h, view.patches);
            eprintln!("wrote {path}");
            results.push((mode, view.patches.iter().map(|(_, r)| patch_means(&sum, w, h, *r)).collect()));
        }
        if view.patches.is_empty() {
            continue;
        }
        println!("\n## {}\n| patch | {} | {} |", view.name, args.modes.iter().map(|m| format!("{} (display rgb)", m.name())).collect::<Vec<_>>().join(" | "), args.modes.iter().filter(|m| **m != Mode::Rt).map(|m| format!("{} / rt (luminance)", m.name())).collect::<Vec<_>>().join(" | "));
        println!("|---|{}{}", "---|".repeat(args.modes.len()), "---|".repeat(args.modes.iter().filter(|m| **m != Mode::Rt).count()));
        let rt = results.iter().find(|(m, _)| *m == Mode::Rt);
        let lum = |c: [f32; 3]| 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
        for (p, (name, _)) in view.patches.iter().enumerate() {
            let cells: Vec<String> = results.iter().map(|(_, r)| format!("{:.0} {:.0} {:.0}", r[p].1[0], r[p].1[1], r[p].1[2])).collect();
            let ratios: Vec<String> = results
                .iter()
                .filter(|(m, _)| *m != Mode::Rt)
                .map(|(_, r)| rt.map_or("—".into(), |(_, t)| format!("{:.2}x", lum(r[p].0) / lum(t[p].0).max(1e-6))))
                .collect();
            println!("| {name} | {} | {} |", cells.join(" | "), ratios.join(" | "));
        }
    }
    // (no teardown: on Vulkan a device may not finish dropping)
    std::mem::forget(renderer);
    std::process::exit(0);
}
