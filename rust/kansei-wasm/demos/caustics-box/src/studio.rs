//! The studio both scenes stand in: a floor that reaches the horizon (white or grey), a backdrop
//! as bright as the lit floor far off (so the horizon disappears), the key light (a sun or a
//! spot), and the material every opaque surface here is drawn with.
//!
//! The material writes the GBuffer for `RtShadowsEffect` (`kansei_gbuffer_out_rt_lit`): the key
//! light, its ray-traced soft shadows and the caustics are added by the effects, the material
//! itself adds only a flat ambient (the white sky's, when the hybrid RT GI isn't lighting it).

use glam::Vec3 as GVec3;

use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::geometries::{Geometry, SphereGeometry, Vertex};
use kansei_core::gi::GiSurface;
use kansei_core::lights::{DirectionalLight, Light, SpotLight};
use kansei_core::materials::{Binding, GradientSkyOptions, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::Vec3;
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::rt::{RtSurface, RT_SHADOWS_GBUFFER_WGSL};

use crate::settings::{kelvin, Settings};

/// The floor's albedo, white and grey (a photographer's 18% card).
pub const WHITE: f32 = 0.82;
pub const GREY: f32 = 0.18;
/// The floor's half-size (m): far enough for its edge to sit under the horizon.
const FLOOR_HALF: f32 = 400.0;

const SURFACE_WGSL: &str = r#"
// albedo (rgb) and roughness; the flat ambient (cd/m², times the albedo) and whether it reflects
struct StudioSurface { albedo: vec4f, ambient: vec4f };
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

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> FOut {
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    let albedo = surface.albedo.rgb;
    let g = kansei_gbuffer_out_rt_lit(albedo * surface.ambient.rgb, vec3f(0.0), n, albedo, surface.albedo.a, 0.04, surface.ambient.w > 0.5);
    var out: FOut;
    out.color = g.color;
    out.emissive = g.emissive;
    out.normal = g.normal;
    out.albedo = g.albedo;
    out.velocity = kansei_motion_vector(in.curr, in.prev);
    return out;
}
"#;

/// The material's uniform: albedo, roughness, the ambient (cd/m²), reflective (roughness under
/// 0.6: `RtReflectionsEffect` traces it).
pub fn uniform(albedo: [f32; 3], roughness: f32, ambient: [f32; 3]) -> [f32; 8] {
    [albedo[0], albedo[1], albedo[2], roughness, ambient[0], ambient[1], ambient[2], (roughness < 0.6) as u32 as f32]
}

pub fn material(label: &str, albedo: [f32; 3], roughness: f32, ambient: [f32; 3]) -> Material {
    let shader = format!("{MOTION_VECTORS_WGSL}\n{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{SURFACE_WGSL}");
    let mut m = Material::new(label, &shader, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)], MaterialOptions { mrt_output_count: Some(4), outputs_velocity: true, ..Default::default() });
    m.set_uniform_bindable(0, &format!("{label}/Surface"), &uniform(albedo, roughness, ambient));
    m
}

/// A surface of the studio: drawn by `material`, in voxel GI and the ray tracing grid. Not
/// `with_self_lit`: deferred, it has no direct light in the GBuffer's background, so the glass
/// pass lights its hits itself (with shadow rays and the caustics).
pub fn surface(label: &str, geometry: Geometry, albedo: [f32; 3], roughness: f32) -> Renderable {
    let mut r = Renderable::new(geometry, material(label, albedo, roughness, [0.0; 3])).with_gi(GiSurface::new(albedo));
    r.rt = Some(RtSurface::new(albedo));
    r
}

/// The floor's albedo for `floor` ("white" or "grey").
pub fn floor_albedo(floor: &str) -> f32 {
    if floor == "grey" { GREY } else { WHITE }
}

/// A quad at y = 0, `half` each way, facing up.
fn floor_geometry(half: f32) -> Geometry {
    let v = |x: f32, z: f32| Vertex { position: [x, 0.0, z, 1.0], normal: [0.0, 1.0, 0.0], uv: [x, z] };
    Geometry::new("Studio/Floor", vec![v(-half, -half), v(half, -half), v(half, half), v(-half, half)], vec![0, 3, 2, 0, 2, 1])
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
}

impl Studio {
    pub fn new(scene: &mut Scene, s: &Settings, subject: GVec3) -> Self {
        let a = floor_albedo(&s.floor);
        let mut floor = surface("Studio/Floor", floor_geometry(FLOOR_HALF), [a; 3], s.floor_rough);
        floor.cast_shadow = false;
        let floor = scene.add(SceneNode::Renderable(floor));
        let mut sky = Renderable::new(SphereGeometry::new(900.0, 32, 16), Material::gradient_sky("Studio/Backdrop", &GradientSkyOptions { zenith: [1.0; 3], horizon: [1.0; 3], ground: [1.0; 3], curve: 1.0 }));
        sky.cast_shadow = false;
        let sky = scene.add(SceneNode::Renderable(sky));
        let light = scene.add(SceneNode::Light(Light::Directional(DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 1.0, 1.0), 0.0))));
        Self { floor, sky, light, subject }
    }

    /// The white floor's radiance in the open (cd/m²), lit by the key light (a sun's; a spot's at
    /// the subject) and the sky.
    pub fn white(s: &Settings) -> f32 {
        let d = light_direction(s.light_azim, s.light_elev);
        WHITE * (s.light_lux * -d.y + s.sky_radiance * std::f32::consts::PI) / std::f32::consts::PI
    }

    /// The backdrop's radiance: the floor far off (white, or the grey's), times `bg`.
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
    pub fn apply(&self, renderer: &kansei_core::renderers::Renderer, scene: &mut Scene, s: &Settings, ambient: f32) {
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
        let a = floor_albedo(&s.floor);
        if let Some(r) = scene.get_renderable_mut(self.floor) {
            // (in place once drawn; before, the uniform it will be made with)
            let u = uniform([a; 3], s.floor_rough, [ambient; 3]);
            match r.material.bindable_buffer(0) {
                Some(buffer) => renderer.queue().write_buffer(&buffer, 0, bytemuck::cast_slice(&u)),
                None => r.material.set_uniform_bindable(0, "Studio/Floor/Surface", &u),
            }
            // (a change of albedo revoxelizes the floor and rebuilds the grid: only on a change)
            let rt = Some(RtSurface::new([a; 3]));
            if r.rt != rt {
                r.rt = rt;
                r.gi = Some(GiSurface::new([a; 3]));
            }
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
