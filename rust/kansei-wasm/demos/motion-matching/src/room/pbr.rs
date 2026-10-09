//! The room's surfaces: a textured PBR material (colour, normal and occlusion/roughness/metallic
//! maps, from KTX2 files or a glTF model's) drawn into the GBuffer, lit by the sun (cascades) and
//! the spot lights with their shadow maps, or (`deferred`) leaving the direct light to
//! `rt::RtShadowsEffect`, which lights it with ray-traced shadows.
//!
//! Architecture is mapped by world position (box mapping on the dominant axis of the normal), so a
//! texture keeps its size in metres whatever the box; models keep their own uvs. Large surfaces
//! hide the repetition: a slow noise varies colour and roughness across metres, and a floor of
//! slabs (marble) gives each slab its own offset and turn into the texture, with thin seams.

use kansei_core::buffers::{Sampler, Texture};
use kansei_core::lights::{LIGHTS_WGSL, SPOT_LIGHTS_WGSL};
use kansei_core::loaders::ktx2::{self, CompressionSupport, Ktx2Options};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::rt::RT_SHADOWS_GBUFFER_WGSL;
use kansei_core::shadows::{CASCADED_SHADOWS_WGSL, SHADOW_MAP_WGSL};

/// The material's uniform (`Pbr` in the shader).
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PbrParams {
    /// Colour multiplier (rgb); w: 0 the mesh's uvs, 1 world box mapping.
    pub tint: [f32; 4],
    /// x: metres per texture repeat (box mapping); y: roughness scale; z: roughness bias;
    /// w: reflective (1: its reflection traced by `RtReflectionsEffect`).
    pub surface: [f32; 4],
    /// x: pattern (0 none, 1 slabs); y: the slab's size (m); z: the normal map's strength; w: the
    /// alpha cutoff (0: opaque).
    pub pattern: [f32; 4],
    /// rgb: a faint ambient fill (cd/m² per unit albedo), w: deferred (1: direct light by
    /// `RtShadowsEffect`).
    pub ambient: [f32; 4],
    /// rgb: emitted radiance (cd/m²); w: the macro variation's strength.
    pub emissive: [f32; 4],
    /// x: one-sided (1: the camera sees only its front, the shadow maps both sides); y: specular
    /// strength on the forward path; zw unused.
    pub flags: [f32; 4],
}

impl Default for PbrParams {
    fn default() -> Self {
        Self { tint: [1.0, 1.0, 1.0, 1.0], surface: [2.0, 1.0, 0.0, 0.0], pattern: [0.0, 1.2, 1.0, 0.0], ambient: [0.0, 0.0, 0.0, 0.0], emissive: [0.0, 0.0, 0.0, 0.0], flags: [0.0, 1.0, 0.0, 0.0] }
    }
}

/// The direct light on a surface from the scene's lights, with their shadow maps: `pbr_direct`.
/// Prefixed with the light, shadow and spot light chunks (see `LIT_WGSL`).
pub const DIRECT_WGSL: &str = r#"
// the sun through the cascades, the point lights, the spot lights with their shadow maps
fn pbr_direct(world: vec3f, n: vec3f, v: vec3f, base: vec3f, roughness: f32, metallic: f32, pixel: vec2f) -> vec3f {
    var radiance = vec3f(0.0);
    let cascades = kansei_cascades.count > 0u;
    for (var i = 0u; i < kansei_lights.num_directional; i++) {
        let light = kansei_lights.directional[i];
        let l = -normalize(light.direction);
        let isSun = cascades && dot(normalize(light.direction), normalize(kansei_cascades.lightDirection)) > 0.9999;
        let shadow = select(1.0, kansei_sun_shadow(world, n, pixel), isSun);
        radiance += kansei_brdf(n, v, l, base, roughness, metallic) * light.color * shadow;
    }
    for (var i = 0u; i < kansei_lights.num_point; i++) {
        let light = kansei_lights.point[i];
        let l = normalize(light.position - world);
        radiance += kansei_brdf(n, v, l, base, roughness, metallic) * light.color * kansei_point_falloff(light, world);
    }
    return radiance + kansei_spot_lights_radiance(world, n, v, base, roughness, metallic, pixel);
}

"#;

/// The light chunks a forward-lit room material starts with, and `pbr_direct`.
pub fn lit_wgsl() -> String {
    format!("{LIGHTS_WGSL}\n{SHADOW_MAP_WGSL}\n{CASCADED_SHADOWS_WGSL}\n{SPOT_LIGHTS_WGSL}\n{DIRECT_WGSL}")
}

const PBR_WGSL: &str = r#"
struct Pbr { tint: vec4f, surface: vec4f, pattern: vec4f, ambient: vec4f, emissive: vec4f, flags: vec4f };
@group(0) @binding(0) var<uniform> pbr: Pbr;
@group(0) @binding(1) var color_map: texture_2d<f32>;
@group(0) @binding(2) var normal_map: texture_2d<f32>;
@group(0) @binding(3) var orm_map: texture_2d<f32>;
@group(0) @binding(4) var maps: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4f;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4f;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4f;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4f;

struct VIn { @location(0) position: vec4f, @location(1) normal: vec3f, @location(2) uv: vec2f };
struct VOut {
    @builtin(position) @invariant clip: vec4f,
    @location(0) world: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4f(v.normal, 0.0)).xyz;
    out.uv = v.uv;
    return out;
}

fn pbr_hash(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453);
}

fn pbr_hash2(p: vec2f) -> vec2f {
    return fract(sin(vec2f(dot(p, vec2f(127.1, 311.7)), dot(p, vec2f(269.5, 183.3)))) * 43758.5453);
}

// smooth value noise, two octaves: the slow variation that hides a texture's repeats
fn pbr_noise(p: vec2f) -> f32 {
    var total = 0.0;
    var q = p;
    var amp = 0.6;
    for (var o = 0; o < 2; o++) {
        let i = floor(q);
        let f = fract(q);
        let u = f * f * (3.0 - 2.0 * f);
        let n = mix(mix(pbr_hash(i), pbr_hash(i + vec2f(1.0, 0.0)), u.x), mix(pbr_hash(i + vec2f(0.0, 1.0)), pbr_hash(i + vec2f(1.0, 1.0)), u.x), u.y);
        total += amp * n;
        q = q * 2.7 + vec2f(5.2, 1.3);
        amp *= 0.4;
    }
    return total;
}

// The normal map in the frame of the surface's position and uv derivatives (no stored tangents).
fn pbr_mapped_normal(n: vec3f, dp1: vec3f, dp2: vec3f, duv1: vec2f, duv2: vec2f, m: vec3f) -> vec3f {
    let dp2perp = cross(dp2, n);
    let dp1perp = cross(n, dp1);
    let t = dp2perp * duv1.x + dp1perp * duv2.x;
    let b = dp2perp * duv1.y + dp1perp * duv2.y;
    let scale = inverseSqrt(max(max(dot(t, t), dot(b, b)), 1e-20));
    return normalize(t * scale * m.x - b * scale * m.y + n * m.z);
}

struct PbrSample { albedo: vec3f, alpha: f32, n: vec3f, occlusion: f32, roughness: f32, metallic: f32, seam: f32, plane: vec2f };

fn pbr_sample(in: VOut, front: bool) -> PbrSample {
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    // the texture's coordinates and their derivatives: the mesh's, or the world's on the
    // plane facing the normal
    var uv = in.uv;
    var plane = in.world.xz;
    if (pbr.tint.w > 0.5) {
        let a = abs(n);
        if (a.y >= a.x && a.y >= a.z) { plane = in.world.xz; } else if (a.x >= a.z) { plane = vec2f(in.world.z * sign(n.x), -in.world.y); } else { plane = vec2f(-in.world.x * sign(n.z), -in.world.y); }
        uv = plane / pbr.surface.x;
    }
    var duv1 = dpdx(uv);
    var duv2 = dpdy(uv);
    var seam = 0.0;
    if (pbr.pattern.x > 0.5) {
        // slabs: each its own offset and quarter turn into the texture, thin seams between
        let s = plane / pbr.pattern.y;
        let cell = floor(s);
        let local = fract(s);
        let turn = floor(pbr_hash(cell + 17.0) * 4.0);
        var l = local - 0.5;
        if (turn > 0.5) { l = vec2f(-l.y, l.x); }
        if (turn > 1.5) { l = vec2f(-l.y, l.x); }
        if (turn > 2.5) { l = vec2f(-l.y, l.x); }
        uv = ((l + 0.5) * pbr.pattern.y) / pbr.surface.x + pbr_hash2(cell) * 7.0;
        let edge = min(min(local.x, 1.0 - local.x), min(local.y, 1.0 - local.y)) * pbr.pattern.y;
        let width = 0.001 + length(fwidth(plane)) * 0.5;
        seam = 1.0 - smoothstep(width, width * 2.0, edge);
    }
    let color = textureSampleGrad(color_map, maps, uv, duv1, duv2);
    let orm = textureSampleGrad(orm_map, maps, uv, duv1, duv2).rgb;
    let m = (textureSampleGrad(normal_map, maps, uv, duv1, duv2).xyz * 2.0 - 1.0) * vec3f(pbr.pattern.z, pbr.pattern.z, 1.0);
    var s: PbrSample;
    // slow variation over metres
    let macro_v = pbr_noise(plane * 0.23) - 0.5;
    s.albedo = color.rgb * pbr.tint.rgb * (1.0 + macro_v * pbr.emissive.w * 0.35) * (1.0 - seam * 0.3);
    s.alpha = color.a;
    s.n = pbr_mapped_normal(n, dpdx(in.world), dpdy(in.world), duv1, duv2, normalize(m));
    s.occlusion = orm.r;
    s.roughness = clamp(orm.g * pbr.surface.y + pbr.surface.z + macro_v * pbr.emissive.w * 0.2 + seam * 0.4, 0.04, 1.0);
    s.metallic = orm.b;
    s.seam = seam;
    s.plane = plane;
    return s;
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> KanseiGBufferOut {
    if (pbr.flags.x > 0.5 && !front) { discard; }
    let s = pbr_sample(in, front);
    if (pbr.pattern.w > 0.0 && s.alpha < pbr.pattern.w) { discard; }
    let view3 = mat3x3f(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let eye = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(eye - in.world);
    let diffuse = s.albedo * (1.0 - s.metallic);
    let f0 = mix(0.04, max(s.albedo.r, max(s.albedo.g, s.albedo.b)), s.metallic);
    let fill = diffuse * pbr.ambient.rgb * s.occlusion + pbr.emissive.rgb;
    // traced reflections only where the surface is glossy: a rough lobe's few rays a pixel come out
    // as speckle, and its blur is what the lights' GGX (and the GI) give anyway
    let reflective = pbr.surface.w > 0.5 && s.roughness < 0.45;
    if (pbr.ambient.w > 0.5) {
        // direct light from rt::RtShadowsEffect (its roughness and F0 in the GBuffer)
        return kansei_gbuffer_out_rt_lit(fill, pbr.emissive.rgb, s.n, diffuse, s.roughness, f0, reflective);
    }
    let lit = pbr_direct(in.world, s.n, v, s.albedo, s.roughness, s.metallic, in.clip.xy) * mix(1.0, s.occlusion, 0.5) + fill;
    if (reflective) {
        return kansei_gbuffer_out_specular(lit, pbr.emissive.rgb, s.n, diffuse, f0, s.roughness);
    }
    return kansei_gbuffer_out(lit, pbr.emissive.rgb, s.n, diffuse);
}

// shadow passes: alpha-tested cards cut out
@fragment
fn shadow_main(in: VOut, @builtin(front_facing) front: bool) {
    let s = pbr_sample(in, front);
    if (pbr.pattern.w > 0.0 && s.alpha < pbr.pattern.w) { discard; }
}
"#;

/// The surface's three maps.
pub struct PbrMaps {
    pub color: Texture,
    pub normal: Texture,
    pub orm: Texture,
}

impl PbrMaps {
    /// Flat maps of one colour (no detail).
    pub fn flat(color: [u8; 4], roughness: u8, metallic: u8) -> Self {
        Self { color: Texture::from_rgba("Flat/Color", 1, 1, &color), normal: Texture::from_rgba("Flat/Normal", 1, 1, &[128, 128, 255, 255]), orm: Texture::from_rgba("Flat/Orm", 1, 1, &[255, roughness, metallic, 255]) }
    }

    /// `<name>_color.ktx2`, `_normal.ktx2` and `_orm.ktx2` from their bytes.
    pub fn from_ktx2(name: &str, color: &[u8], normal: &[u8], orm: &[u8], support: CompressionSupport) -> Result<Self, String> {
        let load = |suffix: &str, bytes: &[u8], options: &Ktx2Options| ktx2::transcode(&format!("{name}/{suffix}"), bytes, options, support).map(|t| t.into_texture()).map_err(|e| format!("{name}_{suffix}: {e}"));
        Ok(Self { color: load("color", color, &Ktx2Options::color())?, normal: load("normal", normal, &Ktx2Options::linear())?, orm: load("orm", orm, &Ktx2Options::linear())? })
    }
}

/// The shader: the light and GBuffer chunks, then the material.
fn shader() -> String {
    format!("{}\n{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{PBR_WGSL}", lit_wgsl())
}

/// A PBR material of `maps` drawn as `params` says (`double_sided` for leaves and cards; a
/// one-sided one, `flags.x`, is drawn with both sides too, its back faces discarded on screen).
pub fn material(label: &str, maps: PbrMaps, params: &PbrParams, double_sided: bool) -> Material {
    let cutout = params.pattern[3] > 0.0;
    let one_sided = params.flags[0] > 0.5;
    let options = MaterialOptions {
        mrt_output_count: Some(4),
        cull_mode: if double_sided || one_sided { CullMode::None } else { CullMode::Back },
        shadow_fragment_entry: cutout.then_some("shadow_main"),
        ..Default::default()
    };
    let mut material = Material::new(
        label,
        &shader(),
        vec![
            Binding::uniform(0, ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::texture_2d(2, ShaderStages::FRAGMENT),
            Binding::texture_2d(3, ShaderStages::FRAGMENT),
            Binding::sampler(4, ShaderStages::FRAGMENT),
        ],
        options,
    );
    material.set_uniform_bindable(0, &format!("{label}/Pbr"), &[*params]);
    material.set_bindable(1, maps.color);
    material.set_bindable(2, maps.normal);
    material.set_bindable(3, maps.orm);
    material.set_bindable(4, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
    material
}
