//! The studio both scenes stand in: a floor that reaches the horizon (white, grey, or a lab's
//! checkerboard), a backdrop as bright as the lit floor far off (so the horizon disappears), the
//! key light (a sun or a spot), and the material the floor and the tank are drawn with.
//!
//! The material writes the GBuffer for `RtShadowsEffect` (`kansei_gbuffer_out_rt_lit`): the key
//! light, its ray-traced soft shadows and the caustics are added by the effects, the material
//! itself adds only a flat ambient (the white sky's, when the hybrid RT GI isn't lighting it).
//! Forward (`Surface::forward`, the tank) it lights itself, as `standard_lit` does: the water's
//! glass pass reads what lies under it from the GBuffer's background, where a deferred surface has
//! no direct light. The checkerboard is procedural, filtered over each pixel's footprint (no
//! aliasing at grazing angles); the reflections' and the glass's hits read the same pattern from
//! `checker_texture` (`RtSurface::with_albedo_layer`).

use glam::Vec3 as GVec3;

use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::geometries::{Geometry, SphereGeometry, Vertex};
use kansei_core::gi::GiSurface;
use kansei_core::lights::{DirectionalLight, Light, SpotLight, LIGHTS_WGSL, SPOT_LIGHTS_WGSL};
use kansei_core::materials::{Binding, GradientSkyOptions, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;
use kansei_core::rt::{RtSurface, RT_SHADOWS_GBUFFER_WGSL};
use kansei_core::shadows::{CASCADED_SHADOWS_WGSL, SHADOW_MAP_WGSL};

use crate::settings::{kelvin, Settings};

/// The floor's albedo, white and grey (a photographer's 18% card).
pub const WHITE: f32 = 0.82;
pub const GREY: f32 = 0.18;
/// The floor's half-size (m): far enough for its edge to sit under the horizon.
const FLOOR_HALF: f32 = 400.0;

/// The checkerboard (linear albedo): its light and mid squares, the metre lines and their width
/// (m), the axes' width and colours (the line x = 0 blue, z = 0 red).
const CHECKER_LIGHT: f32 = 0.78;
const CHECKER_MID: f32 = 0.5;
const CHECKER_LINE: f32 = 0.2;
const LINE_WIDTH: f32 = 0.008;
const AXIS_WIDTH: f32 = 0.016;
const AXIS_X: [f32; 3] = [0.55, 0.1, 0.08];
const AXIS_Z: [f32; 3] = [0.08, 0.18, 0.55];

fn pattern_wgsl() -> String {
    format!(
        "const CHECKER_LIGHT : f32 = {CHECKER_LIGHT:?};\nconst CHECKER_MID : f32 = {CHECKER_MID:?};\nconst CHECKER_LINE : f32 = {CHECKER_LINE:?};\nconst LINE_WIDTH : f32 = {LINE_WIDTH:?};\nconst AXIS_WIDTH : f32 = {AXIS_WIDTH:?};\nconst AXIS_X : vec3f = vec3f({:?}, {:?}, {:?});\nconst AXIS_Z : vec3f = vec3f({:?}, {:?}, {:?});\n",
        AXIS_X[0], AXIS_X[1], AXIS_X[2], AXIS_Z[0], AXIS_Z[1], AXIS_Z[2]
    )
}

const SURFACE_WGSL: &str = r#"
// albedo (rgb) and roughness; the flat ambient (cd/m², times the albedo) and whether it reflects;
// lit by itself (x), the checkerboard's square (m, y; 0 plain), its metre lines (z) and axes (w)
struct StudioSurface { albedo: vec4f, ambient: vec4f, mode: vec4f };
@group(0) @binding(0) var<uniform> surface: StudioSurface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> mesh: KanseiMeshTransforms;

struct VIn { @location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f };
struct VOut {
    @builtin(position) @invariant clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) curr: vec4f,
    @location(3) prev: vec4f,
};
// the GBuffer's four targets and the motion
struct FOut {
    @location(0) color: vec4f,
    @location(1) emissive: vec4f,
    @location(2) normal: vec4f,
    @location(3) albedo: vec4f,
    @location(4) velocity: vec2f,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = mesh.world * vec4f(v.position.xyz, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(v.normal, 0.0)).xyz;
    out.curr = kansei_camera_temporal.viewProj * world;
    out.prev = kansei_camera_temporal.prevViewProj * (mesh.prevWorld * vec4f(v.position.xyz, 1.0));
    return out;
}

// The coverage of lines `width` (m) wide at distances d, over a pixel's footprint fw: sharp up
// close, fading to their share of the area where a pixel spans more than a line.
fn studio_lines(d: vec2f, fw: vec2f, width: f32) -> vec2f {
    return clamp((width * 0.5 - d) / fw + 0.5, vec2f(0.0), vec2f(1.0)) * min(vec2f(width) / fw, vec2f(1.0));
}

// The checkerboard at world xz `p`: squares box-filtered over the pixel's footprint (Inigo
// Quilez's filtered checkers), the metre lines and the axes over them.
fn studio_checker(p: vec2f) -> vec3f {
    let q = p / max(surface.mode.y, 1e-3);
    let w = max(fwidth(q), vec2f(1e-4));
    let i = 2.0 * (abs(fract((q - 0.5 * w) * 0.5) - 0.5) - abs(fract((q + 0.5 * w) * 0.5) - 0.5)) / w;
    var c = vec3f(mix(CHECKER_LIGHT, CHECKER_MID, 0.5 - 0.5 * i.x * i.y));
    let fw = max(fwidth(p), vec2f(1e-4));
    let metre = studio_lines(abs(fract(p + 0.5) - 0.5), fw, LINE_WIDTH);
    c = mix(c, vec3f(CHECKER_LINE), max(metre.x, metre.y) * surface.mode.z);
    let axes = studio_lines(abs(p), fw, AXIS_WIDTH) * surface.mode.w;
    c = mix(c, AXIS_Z, axes.x);
    c = mix(c, AXIS_X, axes.y);
    return c;
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> FOut {
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    // (the pattern's derivatives taken whatever the mode, in uniform control flow)
    let checker = studio_checker(in.world.xz);
    let albedo = select(surface.albedo.rgb, checker, surface.mode.y > 0.0 && n.y > 0.5);
    let roughness = surface.albedo.a;
    var g: KanseiGBufferOut;
    if (surface.mode.x > 0.5) {
        // lit by the scene's sun (through the cascades) and spot lights, as standard_lit is
        let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
        let eye = -(transpose(view3) * view_matrix[3].xyz);
        let V = normalize(eye - in.world);
        let cascades = kansei_cascades.count > 0u;
        let sunShadow = select(kansei_shadow_map(in.world, n), kansei_sun_shadow(in.world, n, in.clip.xy), cascades);
        var radiance = albedo * surface.ambient.rgb;
        for (var i = 0u; i < kansei_lights.num_directional; i++) {
            let light = kansei_lights.directional[i];
            let L = -normalize(light.direction);
            let isSun = select(i == 0u, dot(normalize(light.direction), normalize(kansei_cascades.lightDirection)) > 0.9999, cascades);
            radiance += kansei_brdf(n, V, L, albedo, roughness, 0.0) * light.color * select(1.0, sunShadow, isSun);
        }
        radiance += kansei_spot_lights_radiance(in.world, n, V, albedo, roughness, 0.0, in.clip.xy);
        g = kansei_gbuffer_out(radiance, vec3f(0.0), n, albedo);
    } else {
        g = kansei_gbuffer_out_rt_lit(albedo * surface.ambient.rgb, vec3f(0.0), n, albedo, roughness, 0.04, surface.ambient.w > 0.5);
    }
    var out: FOut;
    out.color = g.color;
    out.emissive = g.emissive;
    out.normal = g.normal;
    out.albedo = g.albedo;
    out.velocity = kansei_motion_vector(in.curr, in.prev);
    return out;
}
"#;

/// The checkerboard's look (`floor=checker`): its square (m), the metre lines, the axes.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Checker {
    pub size: f32,
    pub lines: bool,
    pub axes: bool,
}

impl Checker {
    /// The squares the panel offers (m): each divides the texture's period (`period`).
    pub const SIZES: [f32; 5] = [0.05, 0.1, 0.25, 0.5, 1.0];

    pub fn from(s: &Settings) -> Option<Self> {
        let size = Self::SIZES.iter().copied().min_by(|a, b| (a - s.checker_size).abs().total_cmp(&(b - s.checker_size).abs())).unwrap();
        (s.floor == "checker").then_some(Self { size, lines: s.checker_lines, axes: s.checker_axes })
    }

    /// The pattern's period (m): two squares, and whole metres for the lines.
    pub fn period(&self) -> f32 {
        (2.0 * self.size).max(1.0)
    }
}

/// A studio surface: its albedo (or the checkerboard on its upward faces), roughness, the flat
/// ambient it adds (cd/m²), and lit by itself (`forward`) or by `RtShadowsEffect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub albedo: f32,
    pub roughness: f32,
    pub ambient: f32,
    pub forward: bool,
    pub checker: Option<Checker>,
}

impl Surface {
    /// The material's uniform (`StudioSurface`); reflective (traced by `RtReflectionsEffect`)
    /// under a roughness of 0.6, when deferred.
    pub fn uniform(&self) -> [f32; 12] {
        let a = self.albedo;
        let c = self.checker;
        [
            a, a, a, self.roughness,
            self.ambient, self.ambient, self.ambient, (self.roughness < 0.6 && !self.forward) as u32 as f32,
            self.forward as u32 as f32, c.map_or(0.0, |c| c.size), c.is_some_and(|c| c.lines) as u32 as f32, c.is_some_and(|c| c.axes) as u32 as f32,
        ]
    }

    /// Its mean albedo (the checkerboard's averaged), for voxel GI and the grid.
    pub fn mean(&self) -> f32 {
        self.checker.map_or(self.albedo, |c| checker_mean(&c))
    }

    /// Its surface in the ray tracing grid: textured by the checkerboard's layer (0 of
    /// `checker_texture`) when it has one; `self_lit` where only the water's glass pass sees it.
    pub fn rt(&self, self_lit: bool) -> RtSurface {
        let mut rt = RtSurface::new([self.mean(); 3]);
        if self.checker.is_some() {
            rt = rt.with_albedo_layer(0);
        }
        if self_lit {
            rt = rt.with_self_lit();
        }
        rt
    }
}

pub fn material(label: &str, surface: &Surface) -> Material {
    let shader = format!("{LIGHTS_WGSL}\n{SHADOW_MAP_WGSL}\n{CASCADED_SHADOWS_WGSL}\n{SPOT_LIGHTS_WGSL}\n{MOTION_VECTORS_WGSL}\n{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{}\n{SURFACE_WGSL}", pattern_wgsl());
    let mut m = Material::new(label, &shader, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() });
    m.set_uniform_bindable(0, &format!("{label}/Surface"), &surface.uniform());
    m
}

/// A renderable drawn by `material`, in voxel GI and the ray tracing grid.
pub fn renderable(label: &str, geometry: Geometry, surface: &Surface, self_lit: bool) -> Renderable {
    let mut r = Renderable::new(geometry, material(label, surface)).with_gi(GiSurface::new([surface.mean(); 3]));
    r.rt = Some(surface.rt(self_lit));
    r
}

/// Bring a `renderable`'s surface to `surface`: its uniform (in place once drawn), and its
/// albedo in voxel GI and the grid when that changed (which revoxelizes it and rebuilds the grid).
pub fn set_surface(renderer: &Renderer, r: &mut Renderable, label: &str, surface: &Surface, self_lit: bool) {
    let u = surface.uniform();
    match r.material.bindable_buffer(0) {
        Some(buffer) => renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&u)),
        None => r.material.set_uniform_bindable(0, &format!("{label}/Surface"), &u),
    }
    let rt = Some(surface.rt(self_lit));
    if r.rt != rt {
        r.rt = rt;
        r.gi = Some(GiSurface::new([surface.mean(); 3]));
    }
}

/// The checkerboard over one period, `CHECKER_TEXELS` a side, linear albedo, filtered 4 x 4 a
/// texel (the metre lines and the squares as the shader draws them up close; no axes).
fn checker_texels(c: &Checker) -> Vec<f32> {
    let n = CHECKER_TEXELS as usize;
    let period = c.period();
    let mut out = vec![0.0; n * n];
    let sub = 4;
    for y in 0..n {
        for x in 0..n {
            let mut sum = 0.0;
            for sy in 0..sub {
                for sx in 0..sub {
                    let p = [(x as f32 + (sx as f32 + 0.5) / sub as f32) / n as f32 * period, (y as f32 + (sy as f32 + 0.5) / sub as f32) / n as f32 * period];
                    let odd = ((p[0] / c.size).floor() + (p[1] / c.size).floor()).rem_euclid(2.0) > 0.5;
                    let mut v = if odd { CHECKER_MID } else { CHECKER_LIGHT };
                    let near = |t: f32| (t - t.round()).abs() < LINE_WIDTH * 0.5;
                    if c.lines && (near(p[0]) || near(p[1])) {
                        v = CHECKER_LINE;
                    }
                    sum += v;
                }
            }
            out[y * n + x] = sum / (sub * sub) as f32;
        }
    }
    out
}

/// Texels a side of the checkerboard's texture (2 mm over a 1 m period).
const CHECKER_TEXELS: u32 = 512;

/// The checkerboard's mean albedo: its squares', and the metre lines' share of the area (both
/// ways, `LINE_WIDTH` a metre).
fn checker_mean(c: &Checker) -> f32 {
    let lines = if c.lines { 2.0 * LINE_WIDTH - LINE_WIDTH * LINE_WIDTH } else { 0.0 };
    0.5 * (CHECKER_LIGHT + CHECKER_MID) * (1.0 - lines) + CHECKER_LINE * lines
}

/// The checkerboard as `RtReflectionsEffect::set_albedo_texture` takes it: a one-layer 2D array
/// (sRGB) with its mips to 1 texel, the floor's uvs a period apart (`floor_geometry`).
pub fn checker_texture(renderer: &Renderer, c: &Checker) -> wgpu::TextureView {
    let n = CHECKER_TEXELS;
    let levels = n.trailing_zeros() + 1;
    let texture = renderer.device().create_texture(&wgpu::TextureDescriptor {
        label: Some("Studio/Checker"),
        size: wgpu::Extent3d { width: n, height: n, depth_or_array_layers: 1 },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let srgb = |v: f32| {
        let v = v.clamp(0.0, 1.0);
        ((if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }) * 255.0).round() as u8
    };
    let mut level = checker_texels(c);
    let mut size = n as usize;
    for mip in 0..levels {
        let bytes: Vec<u8> = level.iter().flat_map(|&v| { let b = srgb(v); [b, b, b, 255] }).collect();
        renderer.queue().write_texture(
            wgpu::TexelCopyTextureInfo { texture: &texture, mip_level: mip, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            &bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(size as u32 * 4), rows_per_image: Some(size as u32) },
            wgpu::Extent3d { width: size as u32, height: size as u32, depth_or_array_layers: 1 },
        );
        if size > 1 {
            let half = size / 2;
            level = (0..half * half).map(|i| { let (x, y) = (i % half * 2, i / half * 2); (level[y * size + x] + level[y * size + x + 1] + level[(y + 1) * size + x] + level[(y + 1) * size + x + 1]) * 0.25 }).collect();
            size = half;
        }
    }
    texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() })
}

/// The floor's albedo for `floor` ("white", "grey"; the checkerboard's light squares).
pub fn floor_albedo(floor: &str) -> f32 {
    match floor {
        "grey" => GREY,
        "checker" => CHECKER_LIGHT,
        _ => WHITE,
    }
}

/// A quad at y = 0, `half` each way, facing up, its uvs `period` (m) apart.
fn floor_geometry(half: f32, period: f32) -> Geometry {
    let v = |x: f32, z: f32| Vertex { position: [x, 0.0, z, 1.0], normal: [0.0, 1.0, 0.0], uv: [x / period, z / period] };
    Geometry::new("Studio/Floor", vec![v(-half, -half), v(half, -half), v(half, half), v(-half, half)], vec![0, 3, 2, 0, 2, 1])
}

/// The floor's surface for the settings.
pub fn floor_surface(s: &Settings, ambient: f32) -> Surface {
    Surface { albedo: floor_albedo(&s.floor), roughness: s.floor_rough, ambient, forward: false, checker: Checker::from(s) }
}

/// The key light's direction of travel (from the light toward the subject) for an azimuth and
/// elevation in degrees; azimuth 0 lights from +z (the camera's side), 90 from +x.
pub fn light_direction(azim: f32, elev: f32) -> GVec3 {
    let (a, e) = (azim.to_radians(), elev.clamp(2.0, 89.5).to_radians());
    -GVec3::new(e.cos() * a.sin(), e.sin(), e.cos() * a.cos()).normalize()
}

pub struct Studio {
    pub floor: usize,
    pub sky: usize,
    pub light: usize,
    /// what the light points at (the subject's centre)
    pub subject: GVec3,
    /// the checkerboard as last applied, and its texture for the reflections' hits (made anew,
    /// and `checker_changed` set, when its look changes)
    checker: Option<Checker>,
    pub checker_view: Option<wgpu::TextureView>,
    pub checker_changed: bool,
}

impl Studio {
    pub fn new(scene: &mut Scene, s: &Settings, subject: GVec3) -> Self {
        let surface = floor_surface(s, 0.0);
        let period = surface.checker.map_or(1.0, |c| c.period());
        let mut floor = renderable("Studio/Floor", floor_geometry(FLOOR_HALF, period), &surface, false);
        floor.cast_shadow = false;
        let floor = scene.add(SceneNode::Renderable(floor));
        let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), Material::gradient_sky("Studio/Backdrop", &GradientSkyOptions { zenith: [1.0; 3], horizon: [1.0; 3], ground: [1.0; 3], curve: 1.0 }));
        sky.cast_shadow = false;
        let sky = scene.add(SceneNode::Renderable(sky));
        let light = scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.0))));
        Self { floor, sky, light, subject, checker: None, checker_view: None, checker_changed: false }
    }

    /// The white floor's radiance in the open (cd/m²), lit by the key light (a sun's; a spot's at
    /// the subject) and the sky.
    pub fn white(s: &Settings) -> f32 {
        let d = light_direction(s.light_azim, s.light_elev);
        WHITE * (s.light_lux * -d.y + s.sky_radiance * std::f32::consts::PI) / std::f32::consts::PI
    }

    /// The backdrop's radiance: the floor far off (white, or the grey's or the checkerboard's light
    /// squares'), times `bg`.
    pub fn backdrop(&self, s: &Settings) -> f32 {
        let a = if s.bg_white { WHITE } else { floor_albedo(&s.floor) };
        Self::white(s) * a / WHITE * s.bg.max(0.0)
    }

    /// The exposure (EV100) that shows the lit white floor at `level` of display white (before
    /// the tone map's shoulder): a white studio.
    pub fn auto_ev(s: &Settings, level: f32) -> f32 {
        // exposure_from_ev100(ev) = 1 / (1.2 x 2^ev)
        (Self::white(s) / (1.2 * level)).max(1.0).log2()
    }

    /// The light, the floor and the backdrop to the settings; `ambient` is the flat light the
    /// surfaces add themselves (cd/m², none when the RT GI lights them).
    pub fn apply(&mut self, renderer: &Renderer, scene: &mut Scene, s: &Settings, ambient: f32) {
        let d = light_direction(s.light_azim, s.light_elev);
        let c = kelvin(s.light_temp);
        let color = Vec3::new(c[0], c[1], c[2]);
        let size = s.light_size.clamp(0.0, 40.0).to_radians();
        let light = if s.light_type == "spot" {
            let dist = s.light_dist.clamp(1.0, 50.0);
            let at = self.subject - d * dist;
            // the cone wide enough for the subject and its shadow; the illuminance at the subject
            let mut l = SpotLight::new(Vec3::new(at.x, at.y, at.z), Vec3::new(d.x, d.y, d.z), color, s.light_lux * dist * dist, dist * 4.0, 0.3, 0.55);
            l.source_radius = (size * 0.5).tan() * dist;
            l.cast_shadow = true;
            Light::Spot(l)
        } else {
            let mut l = DirectionalLight::new(Vec3::new(d.x, d.y, d.z), color, s.light_lux);
            l.cast_shadow = true;
            Light::Directional(l)
        };
        if let Some(slot) = scene.get_light_mut(self.light) {
            *slot = light;
        }
        // the floor: its surface, and with a new checkerboard its uvs (a period apart) and the
        // texture the reflections' hits read
        let surface = floor_surface(s, ambient);
        if surface.checker != self.checker {
            if let Some(c) = &surface.checker {
                if self.checker.is_none_or(|old| old.period() != c.period()) {
                    if let Some(r) = scene.get_renderable_mut(self.floor) {
                        let revision = r.geometry.revision;
                        r.geometry = floor_geometry(FLOOR_HALF, c.period());
                        r.geometry.revision = revision.wrapping_add(1);
                    }
                }
                self.checker_view = Some(checker_texture(renderer, c));
                self.checker_changed = true;
            }
            self.checker = surface.checker;
        }
        if let Some(r) = scene.get_renderable_mut(self.floor) {
            set_surface(renderer, r, "Studio/Floor", &surface, false);
        }
        // the backdrop's uniform (`Material::gradient_sky`'s: zenith, horizon, ground), or the
        // material made again before its first draw (its buffer is made then)
        let bg = self.backdrop(s);
        if let Some(r) = scene.get_renderable_mut(self.sky) {
            match r.material.bindable_buffer(0) {
                Some(buffer) => renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&[bg, bg, bg, 1.0, bg, bg, bg, 0.0, bg, bg, bg, 0.0])),
                None => {
                    r.material = Material::gradient_sky("Studio/Backdrop", &GradientSkyOptions { zenith: [bg; 3], horizon: [bg; 3], ground: [bg; 3], curve: 1.0 });
                    r.material_dirty = true;
                }
            }
        }
    }
}
