//! GI box: a Cornell box (white floor, ceiling and back wall, a red wall on the left, a green one
//! on the right, two white blocks, a rug half orange and half blue, optionally the Stanford
//! dragon) under one shadowed downlight, to compare global illumination methods. Without GI the
//! scene has direct light only, so the ceiling and the shadows are black; with it the floor's
//! light reaches the ceiling and the walls bleed their colour onto the floor and the blocks.
//!
//! - Screen-space GI (`gi=low|medium|high|ultra`) sees only what is on screen.
//! - Voxel GI (`gi=voxel`) traces cones through a voxel volume of the room: the renderables are
//!   voxelized through their own vertex shaders and lit through the lamp's shadow map, with the
//!   bounces adding up over frames, so light from off screen (the open front's side of the walls,
//!   the backs of the blocks) arrives too. The rug's voxels take its texture through its
//!   material's voxel entry (`albedo=constant`: its mean colour instead).
//! - `gi=voxel+ssgi`: screen-space GI in front for contact detail, the voxels for the rest.
//! - `gi=probes` and `gi=probes+ssgi`: the voxels' light reaches the screen through irradiance
//!   probes traced in the voxels' distance field (`SdfProbes`) instead of cones per pixel.
//! - `gi=rt` (the default): the hybrid (`RtDiffuseGiEffect`): one ray for each 2 x 2 pixels from the surface
//!   through a grid of the room's triangles, the hits lit by the lamp (shadow rays through the
//!   grid) and one cone through the voxels for the further bounces, denoised by SVGF; closest to
//!   the path-traced reference in the contacts and behind the blocks.
//!
//! A chrome ball stands on the short block and a glass ball on the rug, ray traced through the
//! same grid of triangles whatever the GI (`RtReflectionsEffect` with `set_glass`): the chrome
//! mirrors the room (`StandardLitOptions::mirror`, rough to frosted chrome), the glass refracts
//! it through both its surfaces (`StandardLitOptions::glass`, `RtSurface::glass`: its index of
//! refraction, tint and roughness, total internal reflection), and the GI's rays pass through it.
//!
//! Drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan. The panel
//! (and `window.kansei`) switches everything at run time.
//!
//! URL parameters (a `preset` first, the others over it):
//! - `preset=hybrid|off|ssgi|voxel|best|indirect|voxels|phone|dragon|sdf|sdf-dragon|slice|probes|probe-view|probes-dragon` (see `PRESETS`;
//!   `hybrid`, the hybrid path tracing (`gi=rt`), unless the URL names a preset or a `gi`; `best` is
//!   voxel + SSGI at the device's tier);
//! - `gi=off|low|medium|high|ultra|voxel|voxel+ssgi|probes|probes+ssgi|rt` (`rt`, or `rt=1`,
//!   builds the grid of triangles at load; the panel reloads the page to switch to it without);
//! - with `gi=rt`, the hybrid's settings (`RtGi`): `rtgi_res=half|full`,
//!   `rtgi_denoise=svgf|temporal|off`, `rtgi_kernel=3x3|5x5`, `rtgi_hit=direct|voxels`,
//!   `rtgi_shadows=rays|maps`, `rtgi_mode=hybrid|reference` (a path tracer through the grid),
//!   `rtgi_accum=1` (the running mean of the raw signal),
//!   `rtgi_view=lit|indirect|signal|variance|history|cost`;
//! - `voxels=low|medium|high`: the volume's resolution (default medium; low on phones, which also
//!   keep it within 24 MiB);
//! - `view=indirect` (only the light GI adds, 2 stops brighter), `view=voxels` (with voxel GI:
//!   the lit voxels themselves), `view=sdf` (a slice of voxel GI's distance field, `slice=`
//!   metres up, default 0.6) or `view=probes` (the probes, lit by their own irradiance);
//! - the distance field (it turns voxel GI's scene volume on whatever the mode): `sdf_ao=0..1`
//!   (its AO on the GI), `sdf_shadows=off|fallback|always` (the voxels' shadows through it, where
//!   no map covers them or always), `shadows=sdf` (the direct light's shadows through it, by
//!   the material helper `gi::SDF_WGSL`);
//! - the direct light's shadows: `shadows=maps` (the shadow atlas, PCSS: the default) or
//!   `shadows=rt` (`RtShadowsEffect`: rays through the grid of triangles toward a point of the
//!   lamp's disk, denoised; the materials leave the lamp to it through `RT_SHADOWS_GBUFFER_WGSL`),
//!   with `rts_res=half|full`, `rts_contact=<m>` (the screen-space contact rays, 0.25; 0 none) and
//!   `rts_view=lit|visibility|direct|mask`; its shadow rays pass through the glass ball, whose
//!   caustics (`RtCausticsEffect`: photons from the lamp through the ball into a tile round it)
//!   put back what the glass changes of that light, the light it focuses and the shadow it
//!   leaves: `caustics=0` without them, `caustics_view=caustics` (the light they add and take,
//!   over grey);
//! - `glass_shadow=1`: the glass ball casts into the shadow maps (with `shadows=maps`, the solid
//!   disc a shadow map gives glass); `roll=1`: the glass ball rolls back and forth across the rug;
//!   `glass_radius=` (m): the glass ball's size (0.015 and `cam=marble`: a 3 cm marble on the rug);
//! - the balls: `mirror=0`, `glass=0` (without either, and without `reflect`, `gi=rt` or `rt=1`, no
//!   grid of triangles is built), `mirror_rough=0..1`, `glass_ior=` (1.5), `glass_rough=0..1`
//!   (frosted), `glass_tint=r,g,b` (the light left after a metre inside), `glass_samples=` (paths
//!   a frosted pixel a frame, 4), `rt_screen=0` (their rays' hits lit by the lamp and the voxels
//!   even where the camera sees them), `rt_direct=0` (hits the camera doesn't see lit by the
//!   voxels alone);
//! - `cam=front|corner|low|floor|glass|chrome|marble`;
//! - `dragon=1|full`: the Stanford dragon (CC-BY-NC-4.0, see `www/assets/license.txt`): `1` the
//!   decimated `.glb` (19k triangles), `full` the whole scan (871k triangles, 24 MB);
//! - `animate=1`: the dragon (or, without it, the tall block) turns and slides, re-voxelized each
//!   frame;
//! - `reflect=1`: the floor polished, its reflections traced through a grid of the room's
//!   triangles (`Renderer::enable_rt_grid`, `RtReflectionsEffect`), lit by the voxels;
//!   `floor_f0`, `floor_rough`, `rt_view=lit|reflection|mirror|cost`, `rt_trace=voxels`,
//!   `rt_res=quarter`, `rt_lod=0` (the dragon into the grid whole, not by its cluster cut);
//! - the environment past the box's opening (`environment`): `env=none` (the default: black),
//!   `gradient` (a studio gradient, `GradientEnvironment`, on a dome) or `hdri` (Poly Haven's CC0
//!   `studio_small_09`, `HdrEnvironment`: the background, the balls' reflections and refraction, and
//!   the GI's rays past the box); `env_ev=` (stops over the default), `env_rotation=` (degrees about
//!   the vertical), `env_blur=0..1` (the HDRI background's softness), `env_spin=` (degrees a second:
//!   the HDRI prepared again every frame, to time it);
//! - `albedo=constant`, `rug=off` (the box without its rug), `ui=0` (no panel);
//! - `stats=1`: triangles, frame interval and the GPU time of each pass (the renderer's profiling);
//! - `rt_moving=0`: what moves (the rolling glass ball, the animated block or dragon) rebuilds the
//!   whole grid every frame instead of the grid's moving layer alone (`SceneRtGrid::set_moving_layer`,
//!   for comparisons; `set_rt_moving_layer` switches it live);
//! - `record_pipelines=1`: every pipeline the page makes recorded (`window.kanseiPipelines.download()`
//!   saves them as `pipelines.json`); `warm=1`: `www/pipelines.json`'s compiled before the device
//!   comes, in parallel (`kansei_wasm::pipelines`).

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::environment::{EnvironmentLighting, GradientEnvironment, HdrEnvironment, HdrEnvironmentOptions, HdrEnvironmentSettings};
use kansei_core::geometries::{BoxGeometry, SphereGeometry};
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfProbeOptions, SdfShadows, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, SDF_WGSL, VOXEL_WRITE_WGSL};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Binding, GradientSkyOptions, Material, MaterialOptions, ShaderStages, StandardLitOptions, GBUFFER_OUT_WGSL};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, EnvironmentBackgroundEffect, GiQuality, ScreenSpaceGIEffect, ScreenSpaceGIOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions};
use kansei_core::rt::{RtCausticsEffect, RtCausticsOptions, RtCausticsView, 
    RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiHitLighting, RtGiKernel, RtGiMode, RtGiResolution, RtGiShadows, RtGiView, RtGlass, RtGridOptions,
    RtReflectionsEffect, RtReflectionsOptions, RtReflectionsView, RtShadowsEffect, RtShadowsView, RtSurface, RtTraceResolution, SceneRtGridOptions,
    RT_SHADOWS_GBUFFER_WGSL,
};
use kansei_wasm::{fetch_bytes, flag, is_phone, now, param, param_or, Canvas, Frame};

/// A diffuse surface lit by the spot lights only (no ambient), writing the normal and albedo the
/// global illumination reads (GBuffer targets 2 and 3, through `materials::GBUFFER_OUT_WGSL`): its
/// albedo the uniform's colour times, for `TEXTURED_WGSL`, a texture; and a polished one's F0 and
/// roughness (`specular`, 0 by default), for the ray-traced reflections.
const SURFACE_WGSL: &str = r#"
struct Surface { albedo: vec4<f32>, specular: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
struct VOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};
@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    out.uv = v.uv;
    return out;
}

fn shade(in: VOut, base: vec3<f32>) -> KanseiGBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    return lit_out(lit_radiance(in.world, n, v, base, in.clip.xy), n, base, surface.specular.x, surface.specular.y);
}
"#;

const PLAIN_WGSL: &str = r#"
@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    return shade(in, surface.albedo.rgb);
}
"#;

/// The rug: its albedo from a texture, and a voxel entry that gives voxel GI the same texture.
const TEXTURED_WGSL: &str = r#"
@group(0) @binding(1) var rug_texture: texture_2d<f32>;
@group(0) @binding(2) var rug_sampler: sampler;

@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    return shade(in, surface.albedo.rgb * textureSample(rug_texture, rug_sampler, in.uv).rgb);
}

// voxel GI's voxelizer: the rug's texture into its voxels
@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, surface.albedo.rgb * textureSample(rug_texture, rug_sampler, in.uv).rgb, vec3<f32>(0.0));
}
"#;

/// The chrome ball: its reflectance (`StandardLitOptions::mirror`), and the albedo the GI's rays
/// and voxels see it with (dark: they don't mirror).
const MIRROR_COLOR: [f32; 3] = [0.95, 0.95, 0.95];
const MIRROR_GI_ALBEDO: [f32; 3] = [0.02, 0.02, 0.02];
/// The light the glass leaves after a metre inside (a faint green, as window glass).
const GLASS_TINT: [f32; 3] = [0.82, 0.93, 0.88];
const BALL_RADIUS: f32 = 0.4;
/// Where the glass ball rests on the rug (x, z), its bottom a little over the rug's top.
const GLASS_REST: [f32; 2] = [-0.1, -0.45];

/// The glass ball's centre for its radius `r` (`glass_radius=`), `x` along from its rest.
fn glass_centre(r: f32, x: f32) -> Vec3 {
    Vec3::new(GLASS_REST[0] + x, 0.02 + r * 1.0125, GLASS_REST[1])
}
/// The balls' tessellation (the grid traces their triangles, shaded by their vertex normals).
const BALL_SEGMENTS: u32 = 64;
const BALL_RINGS: u32 = 48;

/// `r,g,b`
fn parse_rgb(v: &str) -> Option<[f32; 3]> {
    let c: Vec<f32> = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    (c.len() == 3).then(|| [c[0], c[1], c[2]])
}

/// The rug's two halves (linear albedo), split across its width.
const RUG_LEFT: [f32; 3] = [0.85, 0.35, 0.05];
const RUG_RIGHT: [f32; 3] = [0.05, 0.25, 0.8];

/// The rug's texture: orange on the left half, blue on the right, with a thin pale border.
fn rug_texture() -> Texture {
    const N: u32 = 64;
    let srgb = |c: f32| (if c <= 0.0031308 { c * 12.92 } else { 1.055 * c.powf(1.0 / 2.4) - 0.055 } * 255.0).round() as u8;
    let mut texels = Vec::with_capacity((N * N * 4) as usize);
    for y in 0..N {
        for x in 0..N {
            let border = x < 3 || y < 3 || x >= N - 3 || y >= N - 3;
            let c = if border { [0.8, 0.75, 0.6] } else if x < N / 2 { RUG_LEFT } else { RUG_RIGHT };
            texels.extend([srgb(c[0]), srgb(c[1]), srgb(c[2]), 255]);
        }
    }
    Texture::from_rgba("Rug", N, N, &texels)
}

/// The rug's material; `voxel_entry` false voxelizes it with its constant `GiSurface` instead.
fn rug_material(voxel_entry: bool, sdf: Option<&SdfBinding>, rt: bool) -> Material {
    let (code, bindings) = lit_shader(
        &format!("{VOXEL_WRITE_WGSL}\n{SURFACE_WGSL}\n{TEXTURED_WGSL}"),
        sdf,
        rt,
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
    );
    let mut material = Material::new("Rug", &code, bindings, MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: voxel_entry.then_some("voxel_main"), ..Default::default() });
    bind_sdf(&mut material, sdf);
    material.set_uniform_bindable(0, "Rug", &[1.0f32, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    material.set_bindable(1, rug_texture());
    material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear));
    material
}

/// The GBuffer output of a surface that lights itself (`lit_out`): its light, normal and albedo,
/// and a polished one's F0 and roughness.
const SELF_LIT_OUT_WGSL: &str = r#"
fn lit_out(color: vec3<f32>, n: vec3<f32>, base: vec3<f32>, f0: f32, roughness: f32) -> KanseiGBufferOut {
    return kansei_gbuffer_out_specular(color, vec3<f32>(0.0), n, base, f0, roughness);
}
"#;

/// The spot lights' light on a surface (`lit_radiance`), shadowed by the shadow atlas.
const MAP_SHADOWS_WGSL: &str = r#"
fn lit_radiance(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return kansei_spot_lights_radiance(world, n, v, base, 1.0, 0.0, pixel);
}
"#;

/// No light of its own: the lamp's direct light is `RtShadowsEffect`'s, with ray-traced shadows
/// (`RT_SHADOWS_GBUFFER_WGSL`'s `kansei_gbuffer_out_rt_lit` marks the surface for it). Matte
/// surfaces take its GGX at roughness 1, as `kansei_brdf` lights them with the maps; a polished
/// one's highlight is the traced reflections'.
const RT_SHADOWS_WGSL: &str = r#"
fn lit_radiance(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return vec3<f32>(0.0);
}

fn lit_out(color: vec3<f32>, n: vec3<f32>, base: vec3<f32>, f0: f32, roughness: f32) -> KanseiGBufferOut {
    let polished = f0 > 0.0;
    return kansei_gbuffer_out_rt_lit(color, vec3<f32>(0.0), n, base, select(1.0, roughness, polished), f0, polished);
}
"#;

/// The same, shadowed through voxel GI's distance field instead (`gi::SDF_WGSL`, the material
/// helper): soft shadows, sphere-traced in the field, bound at 10-12 of the material's group.
const SDF_SHADOWS_WGSL: &str = r#"
@group(0) @binding(10) var<uniform> gi_volume: VoxelVolume;
@group(0) @binding(11) var gi_sdf: texture_3d<f32>;
@group(0) @binding(12) var gi_sampler: sampler;

fn lit_radiance(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    var radiance = vec3<f32>(0.0);
    for (var i = 0u; i < kansei_spot_lights.count; i++) {
        let light = kansei_spot_lights.lights[i];
        let s = kansei_spot_sample(light, world);
        if (max(s.illuminance.r, max(s.illuminance.g, s.illuminance.b)) <= 0.0) { continue; }
        let brdf = kansei_brdf(n, v, s.toLight, base, 1.0, 0.0);
        let p = world + n * gi_volume.voxelSize;
        // as hard as the lamp's disk makes it, the march stopping short of the lamp
        let shape = sdfLightShape(gi_volume, light.sourceRadius, length(light.position - p));
        let visibility = sdfSurfaceShadow(gi_volume, gi_sdf, gi_sampler, p, n, s.toLight, shape.x, shape.y);
        radiance += brdf * s.illuminance * visibility;
    }
    return radiance;
}
"#;

/// Voxel GI's distance field as a material binds it for `SDF_SHADOWS_WGSL`.
struct SdfBinding {
    volume: wgpu::Buffer,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl SdfBinding {
    fn of(gi: &kansei_core::gi::SceneVoxelGi) -> Option<Self> {
        let sdf = gi.sdf()?;
        Some(Self { volume: gi.volume().uniform().clone(), texture: sdf.texture().clone(), view: sdf.view().clone() })
    }
}

/// `body` (which calls `lit_radiance` and `lit_out`, through `SURFACE_WGSL`'s `shade`) after the
/// GBuffer output and the light chunk for `sdf` (shadows through the field), `rt` (the light left
/// to `RtShadowsEffect`) or the shadow maps, with its bindings: `bindings` plus the field's.
fn lit_shader(body: &str, sdf: Option<&SdfBinding>, rt: bool, mut bindings: Vec<Binding>) -> (String, Vec<Binding>) {
    match sdf {
        _ if rt => (format!("{GBUFFER_OUT_WGSL}\n{RT_SHADOWS_GBUFFER_WGSL}\n{RT_SHADOWS_WGSL}\n{body}"), bindings),
        None => (format!("{GBUFFER_OUT_WGSL}\n{SPOT_LIGHTS_WGSL}\n{MAP_SHADOWS_WGSL}\n{SELF_LIT_OUT_WGSL}\n{body}"), bindings),
        Some(_) => {
            bindings.extend([Binding::uniform(10, ShaderStages::FRAGMENT), Binding::texture_3d(11, ShaderStages::FRAGMENT), Binding::sampler(12, ShaderStages::FRAGMENT)]);
            (format!("{GBUFFER_OUT_WGSL}\n{SPOT_LIGHTS_WGSL}\n{SDF_WGSL}\n{SDF_SHADOWS_WGSL}\n{SELF_LIT_OUT_WGSL}\n{body}"), bindings)
        }
    }
}

/// Attach the field's resources for `SDF_SHADOWS_WGSL`.
fn bind_sdf(material: &mut Material, sdf: Option<&SdfBinding>) {
    if let Some(sdf) = sdf {
        material.set_bindable(10, ComputeBuffer::from_external("GiVolume", sdf.volume.clone(), BufferType::Uniform));
        material.set_bindable(11, Texture::from_view("GiSdf", sdf.texture.clone(), sdf.view.clone()));
        material.set_bindable(12, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    }
}

/// A lit material of `base_color`, polished as `specular` says (F0, roughness; zeros: matte), its
/// direct light ray traced by `RtShadowsEffect` with `rt`.
fn lit_material(label: &str, base_color: [f32; 3], sdf: Option<&SdfBinding>, rt: bool, specular: [f32; 2]) -> Material {
    let (code, bindings) = lit_shader(&format!("{SURFACE_WGSL}\n{PLAIN_WGSL}"), sdf, rt, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)]);
    let mut material = Material::new(label, &code, bindings, MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    material.set_uniform_bindable(0, label, &[base_color[0], base_color[1], base_color[2], 1.0f32, specular[0], specular[1], 0.0, 0.0]);
    bind_sdf(&mut material, sdf);
    material
}

/// The global illumination asked for (`gi=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gi {
    Off,
    Screen(GiQuality),
    Voxel,
    VoxelAndScreen,
    Probes,
    ProbesAndScreen,
    /// Diffuse GI traced through the grid of triangles (`RtDiffuseGiEffect`), the voxels for the
    /// hits' further bounces.
    Rt,
}

impl Gi {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "off" => Gi::Off,
            "low" => Gi::Screen(GiQuality::Low),
            "medium" => Gi::Screen(GiQuality::Medium),
            "high" | "ssgi" => Gi::Screen(GiQuality::High),
            "ultra" => Gi::Screen(GiQuality::Ultra),
            "voxel" => Gi::Voxel,
            // `param` decodes the query: a raw '+' arrives as a space
            "voxel+ssgi" | "voxel ssgi" => Gi::VoxelAndScreen,
            "probes" => Gi::Probes,
            "probes+ssgi" | "probes ssgi" => Gi::ProbesAndScreen,
            "rt" => Gi::Rt,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Gi::Off => "off",
            Gi::Screen(GiQuality::Low) => "low",
            Gi::Screen(GiQuality::Medium) => "medium",
            Gi::Screen(GiQuality::High) => "high",
            Gi::Screen(GiQuality::Ultra) => "ultra",
            Gi::Voxel => "voxel",
            Gi::VoxelAndScreen => "voxel+ssgi",
            Gi::Probes => "probes",
            Gi::ProbesAndScreen => "probes+ssgi",
            Gi::Rt => "rt",
        }
    }

    fn voxels(self) -> bool {
        matches!(self, Gi::Voxel | Gi::VoxelAndScreen | Gi::Probes | Gi::ProbesAndScreen | Gi::Rt)
    }

    fn probes(self) -> bool {
        matches!(self, Gi::Probes | Gi::ProbesAndScreen)
    }

    /// Screen-space GI in front of the voxels.
    fn near_field(self) -> bool {
        matches!(self, Gi::VoxelAndScreen | Gi::ProbesAndScreen)
    }
}

/// Which Stanford dragon (`dragon=`): both are the same Sketchfab scan (CC-BY-NC-4.0,
/// `www/assets/license.txt`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Dragon {
    Off,
    /// The decimated `.glb` (19k triangles).
    Light,
    /// The full scan, `scene.gltf` + `scene.bin` (871k triangles, a 24 MB download).
    Full,
}

impl Dragon {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "0" | "off" | "false" => Dragon::Off,
            "1" | "on" | "true" | "light" => Dragon::Light,
            "full" => Dragon::Full,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Dragon::Off => "off",
            Dragon::Light => "light",
            Dragon::Full => "full",
        }
    }

    /// Its slot in `State::dragons`.
    fn slot(self) -> Option<usize> {
        match self {
            Dragon::Off => None,
            Dragon::Light => Some(0),
            Dragon::Full => Some(1),
        }
    }
}

/// What the image shows (`view=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum View {
    Lit,
    /// Only the light GI adds, two stops brighter.
    Indirect,
    /// Voxel GI's lit voxels.
    Voxels,
    /// A horizontal slice of voxel GI's distance field (`slice=` metres up).
    Sdf,
    /// The probes as balls lit by their irradiance.
    Probes,
}

impl View {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "lit" => View::Lit,
            "indirect" => View::Indirect,
            "voxels" => View::Voxels,
            "sdf" => View::Sdf,
            "probes" => View::Probes,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            View::Lit => "lit",
            View::Indirect => "indirect",
            View::Voxels => "voxels",
            View::Sdf => "sdf",
            View::Probes => "probes",
        }
    }
}

/// The environment past the box's opening (`env=`).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Env {
    /// Black, as the box has always been.
    None,
    /// A studio gradient (`GradientEnvironment`), bright above, on a dome round the box.
    Gradient,
    /// The studio HDRI (`HdrEnvironment`), fetched the first time.
    Hdri,
}

impl Env {
    fn from_name(name: &str) -> Option<Self> {
        Some(match name {
            "none" | "off" | "0" => Env::None,
            "gradient" => Env::Gradient,
            "hdri" | "hdr" => Env::Hdri,
            _ => return None,
        })
    }

    fn name(self) -> &'static str {
        match self {
            Env::None => "none",
            Env::Gradient => "gradient",
            Env::Hdri => "hdri",
        }
    }
}

/// Poly Haven's "Studio Small 09" by Sergej Majboroda (CC0), 1024 x 512.
const HDRI_ASSET: &str = "assets/studio_small_09_1k.hdr";
/// The HDRI's stops at `env_ev=0`: its mean radiance is about 0.86, the lamp-lit walls' about 20.
const HDRI_EV: f32 = 4.0;
/// The studio gradient's radiance straight up and straight down at `env_ev=0` (cd/m²), about the
/// HDRI's own above and below.
const STUDIO_UP: [f32; 3] = [20.0, 20.0, 21.0];
const STUDIO_DOWN: [f32; 3] = [6.0, 5.8, 5.6];

/// The reflections' view by name: `lit`, `reflection`, `mirror` or `cost`.
fn reflection_view(name: &str) -> Option<RtReflectionsView> {
    match name {
        "lit" => Some(RtReflectionsView::Lit),
        "reflection" => Some(RtReflectionsView::Reflection),
        "mirror" => Some(RtReflectionsView::Mirror),
        "cost" => Some(RtReflectionsView::Cost),
        _ => None,
    }
}

fn reflection_view_name(view: RtReflectionsView) -> &'static str {
    match view {
        RtReflectionsView::Lit => "lit",
        RtReflectionsView::Reflection => "reflection",
        RtReflectionsView::Mirror => "mirror",
        RtReflectionsView::Cost => "cost",
    }
}

/// The floor's F0 and roughness with `reflect=1` (`floor_f0`, `floor_rough`): a polished floor.
fn floor_specular() -> [f32; 2] {
    [param_or("floor_f0", 0.3), param_or("floor_rough", 0.05)]
}

/// The hybrid's settings (`gi=rt`): the URL's `rtgi_*` parameters and `set_rtgi`'s keys.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RtGi {
    resolution: RtGiResolution,
    denoise: RtGiDenoise,
    kernel: RtGiKernel,
    hit: RtGiHitLighting,
    shadows: RtGiShadows,
    mode: RtGiMode,
    accumulate: bool,
    view: RtGiView,
}

impl RtGi {
    const KEYS: [&'static str; 8] = ["res", "denoise", "kernel", "hit", "shadows", "mode", "accum", "view"];

    fn from_url() -> Self {
        let mut r = Self::default();
        for key in Self::KEYS {
            if let Some(v) = param(&format!("rtgi_{key}")) {
                r.set(key, &v);
            }
        }
        r
    }

    /// One setting by its key and name; false if either is unknown.
    fn set(&mut self, key: &str, v: &str) -> bool {
        match key {
            "res" => RtGiResolution::from_name(v).map(|x| self.resolution = x),
            "denoise" => RtGiDenoise::from_name(v).map(|x| self.denoise = x),
            "kernel" => RtGiKernel::from_name(v).map(|x| self.kernel = x),
            "hit" => RtGiHitLighting::from_name(v).map(|x| self.hit = x),
            "shadows" => RtGiShadows::from_name(v).map(|x| self.shadows = x),
            "mode" => RtGiMode::from_name(v).map(|x| self.mode = x),
            "accum" => {
                self.accumulate = v == "1" || v == "true";
                Some(())
            }
            "view" => RtGiView::from_name(v).map(|x| self.view = x),
            _ => None,
        }
        .is_some()
    }

    /// The settings as JSON members (for `info`).
    fn json(&self) -> String {
        format!(
            "\"res\":\"{}\",\"denoise\":\"{}\",\"kernel\":\"{}\",\"hit\":\"{}\",\"shadows\":\"{}\",\"mode\":\"{}\",\"accum\":{},\"view\":\"{}\"",
            self.resolution.name(),
            self.denoise.name(),
            self.kernel.name(),
            self.hit.name(),
            self.shadows.name(),
            self.mode.name(),
            self.accumulate,
            self.view.name()
        )
    }
}

/// The room's grid of triangles for the reflections and the hybrid's rays: 96 cells each way over
/// the box (4.8 cm).
fn rt_grid_options() -> SceneRtGridOptions {
    let cell = 4.6 / 96.0;
    SceneRtGridOptions {
        grid: RtGridOptions { dims: [96, 96, 96], cell, fixed_origin: Some(glam::Vec3::new(-2.3, -0.3, -4.3)), ..Default::default() },
        cluster_error_cells: 1.0,
        rebuild_every_frame: false,
    }
}

/// The direct light's shadows (`shadows=`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Direct {
    /// The shadow atlas (PCSS, the lamp's disk).
    Maps,
    /// Voxel GI's distance field, through the material helper `gi::SDF_WGSL`.
    Sdf,
    /// Ray traced through the room's grid (`RtShadowsEffect`): the materials leave the lamp's
    /// light to it.
    Rt,
}

impl Direct {
    fn from_name(name: &str) -> Option<Self> {
        match name {
            "maps" | "map" => Some(Self::Maps),
            "sdf" => Some(Self::Sdf),
            "rt" => Some(Self::Rt),
            _ => None,
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Maps => "maps",
            Self::Sdf => "sdf",
            Self::Rt => "rt",
        }
    }
}

fn rt_shadows_view(name: &str) -> Option<RtShadowsView> {
    match name {
        "lit" => Some(RtShadowsView::Lit),
        "visibility" => Some(RtShadowsView::Visibility),
        "direct" => Some(RtShadowsView::Direct),
        "mask" => Some(RtShadowsView::Mask),
        _ => None,
    }
}

fn rt_shadows_view_name(view: RtShadowsView) -> &'static str {
    match view {
        RtShadowsView::Lit => "lit",
        RtShadowsView::Visibility => "visibility",
        RtShadowsView::Direct => "direct",
        RtShadowsView::Mask => "mask",
    }
}

/// `RtShadowsEffect`'s settings (`rts_*`, the panel's "RT shadows").
#[derive(Clone, Copy, Debug, PartialEq)]
struct RtShadows {
    view: RtShadowsView,
    /// a ray a pixel (else one for each 2 x 2)
    full: bool,
    /// metres the screen-space contact rays walk (0: none)
    contact: f32,
    /// the glass ball's caustics (`RtCausticsEffect`), and whether they show alone
    caustics: bool,
    caustics_view: bool,
}

impl RtShadows {
    fn from_url() -> Self {
        Self {
            view: param("rts_view").and_then(|v| rt_shadows_view(&v)).unwrap_or_default(),
            full: param("rts_res").as_deref() == Some("full"),
            // (the effect's default)
            contact: param_or("rts_contact", 0.25f32).clamp(0.0, 2.0),
            caustics: flag("caustics", true),
            caustics_view: param("caustics_view").as_deref() == Some("caustics"),
        }
    }

    fn set(&mut self, key: &str, v: &str) -> bool {
        match key {
            "view" => rt_shadows_view(v).map(|x| self.view = x),
            "res" => Some(self.full = v == "full"),
            "contact" => v.parse::<f32>().ok().map(|x| self.contact = x.clamp(0.0, 2.0)),
            "caustics" => Some(self.caustics = matches!(v, "1" | "true" | "on")),
            "caustics_view" => Some(self.caustics_view = v == "caustics"),
            _ => None,
        }
        .is_some()
    }

    fn json(&self) -> String {
        format!(
            "{{\"view\":\"{}\",\"res\":\"{}\",\"contact\":{},\"caustics\":{},\"caustics_view\":\"{}\"}}",
            rt_shadows_view_name(self.view),
            if self.full { "full" } else { "half" },
            self.contact,
            self.caustics,
            if self.caustics_view { "caustics" } else { "lit" }
        )
    }
}

/// Everything the panel and the URL set.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Config {
    gi: Gi,
    /// The voxel tier asked for (the device may hold less).
    voxels: VoxelGiQuality,
    view: View,
    dragon: Dragon,
    animate: bool,
    rug: bool,
    /// The rug's voxels from its texture (its material's voxel entry) or its mean colour.
    textured: bool,
    /// Strength of the distance field's AO on voxel GI (0: none).
    sdf_ao: f32,
    /// The voxels' shadows through the distance field.
    sdf_shadows: SdfShadows,
    /// The direct light's shadows: the shadow atlas, the distance field or ray traced.
    direct: Direct,
    rt_shadows: RtShadows,
    /// The SDF slice's height, metres.
    slice: f32,
    /// `reflect=1`: the polished floor and the grid of triangles its reflections trace (set at
    /// load), and whether the reflections run, what they show, how they trace.
    reflect: bool,
    reflections: bool,
    reflection_view: RtReflectionsView,
    reflection_grid: bool,
    reflection_quarter: bool,
    /// The grid of triangles exists (`reflect=1`, `gi=rt`, `rt=1` or a ball; set at load).
    grid: bool,
    rtgi: RtGi,
    /// The chrome ball on the short block and the glass ball on the rug (`mirror=`, `glass=`),
    /// drawn by the reflections effect whatever the GI.
    mirror: bool,
    glass: bool,
    mirror_rough: f32,
    glass_ior: f32,
    glass_rough: f32,
    glass_tint: [f32; 3],
    glass_samples: u32,
    /// The glass ball casts into the shadow maps (`glass_shadow=1`), and rolls (`roll=1`); its
    /// radius (`glass_radius=`, m, set at load).
    glass_shadow: bool,
    glass_radius: f32,
    roll: bool,
    /// The balls' (and the floor's) rays' hits lit by the lit image where the camera sees them
    /// (`rt_screen=`), else by the lamp and the voxels.
    screen_hits: bool,
    /// The environment past the opening, its stops over the default, its turn (degrees) and the
    /// HDRI background's blur.
    env: Env,
    env_ev: f32,
    env_rotation: f32,
    env_blur: f32,
}

impl Config {
    /// Whether anything reads the probes.
    fn needs_probes(&self) -> bool {
        self.gi.probes() || self.view == View::Probes
    }

    /// Whether anything reads the distance field.
    fn needs_sdf(&self) -> bool {
        self.sdf_ao > 0.0 || self.sdf_shadows != SdfShadows::Off || self.view == View::Sdf || self.direct == Direct::Sdf || self.needs_probes()
    }

    /// Whether the scene's voxel GI must run: for the voxel modes, the distance field, or the
    /// reflections (it lights their hits).
    fn needs_voxels(&self) -> bool {
        self.gi.voxels() || self.needs_sdf() || self.traces()
    }

    /// Whether the reflections effect runs: the polished floor, the chrome ball, the glass.
    fn traces(&self) -> bool {
        self.grid && ((self.reflect && self.reflections) || self.mirror || self.glass)
    }

    /// The balls' materials.
    fn mirror_material(&self) -> StandardLitOptions {
        StandardLitOptions::mirror(MIRROR_COLOR, self.mirror_rough)
    }

    fn glass_material(&self) -> StandardLitOptions {
        StandardLitOptions::glass(self.glass_tint, self.glass_ior, self.glass_rough)
    }

    /// The HDRI's settings.
    fn hdri_settings(&self) -> HdrEnvironmentSettings {
        HdrEnvironmentSettings { intensity: (HDRI_EV + self.env_ev).exp2(), rotation: self.env_rotation.to_radians(), tint: [1.0; 3] }
    }

    /// The studio gradient's radiance up and down.
    fn studio(&self) -> ([f32; 3], [f32; 3]) {
        let k = self.env_ev.exp2();
        (STUDIO_UP.map(|v| v * k), STUDIO_DOWN.map(|v| v * k))
    }
}

/// A preset: the GI mode, voxel tier (None: the device's default), view, dragon and the
/// distance field's uses (AO strength, the voxels' shadows, the direct light's).
struct Preset {
    name: &'static str,
    label: &'static str,
    gi: &'static str,
    voxels: Option<VoxelGiQuality>,
    view: View,
    dragon: Dragon,
    sdf_ao: f32,
    sdf_shadows: SdfShadows,
    direct: Direct,
}

const fn preset(name: &'static str, label: &'static str, gi: &'static str, voxels: Option<VoxelGiQuality>, view: View, dragon: Dragon) -> Preset {
    Preset { name, label, gi, voxels, view, dragon, sdf_ao: 0.0, sdf_shadows: SdfShadows::Off, direct: Direct::Maps }
}

const PRESETS: [Preset; 15] = [
    // the default: diffuse GI path traced through the grid (one ray for each 2 x 2 pixels, SVGF)
    preset("hybrid", "Hybrid path tracing (default)", "rt", None, View::Lit, Dragon::Off),
    preset("off", "Off (direct light)", "off", None, View::Lit, Dragon::Off),
    preset("ssgi", "SSGI", "high", None, View::Lit, Dragon::Off),
    preset("voxel", "Voxel", "voxel", None, View::Lit, Dragon::Off),
    preset("best", "Voxel + SSGI", "voxel+ssgi", None, View::Lit, Dragon::Off),
    preset("indirect", "Indirect only", "voxel+ssgi", None, View::Indirect, Dragon::Off),
    preset("voxels", "Voxels (debug)", "voxel", None, View::Voxels, Dragon::Off),
    preset("phone", "Phone (low)", "voxel+ssgi", Some(VoxelGiQuality::Low), View::Lit, Dragon::Off),
    preset("dragon", "Dragon, 871k tris (voxel + SSGI)", "voxel+ssgi", None, View::Lit, Dragon::Full),
    // the distance field: its AO on the GI and soft shadows for the voxels and the direct light
    Preset { sdf_ao: 0.8, sdf_shadows: SdfShadows::Always, direct: Direct::Sdf, ..preset("sdf", "SDF: AO + soft shadows", "voxel+ssgi", None, View::Lit, Dragon::Off) },
    Preset { sdf_ao: 0.8, sdf_shadows: SdfShadows::Always, direct: Direct::Sdf, ..preset("sdf-dragon", "SDF + dragon, 871k tris", "voxel+ssgi", None, View::Lit, Dragon::Full) },
    preset("slice", "SDF slice (debug)", "voxel", None, View::Sdf, Dragon::Off),
    // irradiance probes traced in the distance field, under SSGI
    preset("probes", "Probes + SSGI", "probes+ssgi", None, View::Lit, Dragon::Off),
    preset("probe-view", "Probes (debug)", "probes+ssgi", None, View::Probes, Dragon::Off),
    preset("probes-dragon", "Probes + dragon, 871k tris", "probes+ssgi", None, View::Lit, Dragon::Full),
];

/// The probes for this device: a probe every 8 voxels; phones update half of them a frame.
fn probe_options(phone: bool) -> SdfProbeOptions {
    SdfProbeOptions { probes_per_frame: if phone { 256 } else { 0 }, ..Default::default() }
}

/// The camera presets: (name, target, distance, azimuth, elevation).
const CAMERAS: [(&str, [f32; 3], f32, f32, f32); 7] = [
    // the box's opening, framed as the Cornell box photographs are
    ("front", [0.0, 2.0, -2.0], 8.3, 0.0, 0.0),
    // inside, from the upper front-left corner toward the short block and the green wall
    ("corner", [1.0, 0.8, -3.0], 4.98, -0.777, 0.634),
    // half a metre above the floor in front of the opening, looking up into the box
    ("low", [0.0, 1.4, -2.5], 7.5, 0.0, -0.12),
    // from the front, above, down onto the floor (its reflections with reflect=1)
    ("floor", [-0.3, 0.5, -2.0], 6.2, 0.3, 0.42),
    // close on the glass ball, the room and (with dragon=1) the dragon behind it
    ("glass", [-0.1, 0.45, -0.45], 1.8, 0.92, 0.12),
    // close on the chrome ball from in front of the opening: it mirrors what is behind the camera
    ("chrome", [0.8, 1.6, -1.5], 2.2, -0.35, 0.1),
    // close on the glass ball shrunk to a marble (glass_radius=0.015) and its caustics on the rug
    ("marble", [-0.1, 0.03, -0.45], 0.12, 0.92, 0.5),
];

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    volume: PostProcessingVolume,
    config: Config,
    phone: bool,
    /// The voxel tier the renderer's voxel GI was enabled at.
    enabled_voxels: Option<VoxelGiQuality>,
    /// Bumped whenever voxel GI or its field is made anew (materials that read the field rebind).
    gi_generation: u32,
    /// The direct shadows the lit materials were made for, and the generation of the field they
    /// read (with `Direct::Sdf`).
    materials: (Direct, Option<u32>),
    /// The lit renderables: (scene index, label, albedo); the rug is apart.
    lit: Vec<(usize, &'static str, [f32; 3])>,
    tall: usize,
    rug: usize,
    /// The chrome and glass balls (with the grid).
    mirror: Option<usize>,
    glass: Option<usize>,
    /// The dragons loaded so far (`Dragon::slot`).
    dragons: [Option<usize>; 2],
    /// (position, yaw) the animated renderables rest at.
    tall_rest: (Vec3, f32),
    dragon_rest: (Vec3, f32),
    time: f32,
    stats: Option<Stats>,
    /// The floor's F0 and roughness (zeros without `reflect=1`).
    floor_specular: [f32; 2],
    /// The environments: the studio gradient (and its dome) and the HDRI once fetched.
    gradient: GradientEnvironment,
    dome: usize,
    hdri: Option<HdrEnvironment>,
    /// Fetching and decoding the HDRI, and uploading it, ms.
    hdri_load_ms: (f64, f64),
    /// `env_spin`: degrees a second the HDRI turns.
    env_spin: f32,
    /// Seconds the glass ball has rolled (`roll=1`).
    roll_time: f32,
}

/// The `stats=1` overlay's numbers, refreshed every second.
#[derive(Default)]
struct Stats {
    frames: u32,
    since: f64,
    frame_ms: f64,
    /// (label, ms per frame), most expensive first
    passes: Vec<(&'static str, f64)>,
    gpu_ms: f64,
    gpu_span_ms: f64,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

fn default_voxels(phone: bool) -> VoxelGiQuality {
    if phone { VoxelGiQuality::Low } else { VoxelGiQuality::Medium }
}

/// The preset of a URL that names neither a preset nor a `gi`.
const DEFAULT_PRESET: &str = "hybrid";

/// `config` with preset `name` applied (unknown names change nothing).
fn with_preset(config: Config, name: &str, phone: bool) -> Config {
    let Some(p) = PRESETS.iter().find(|p| p.name == name) else { return config };
    Config {
        gi: Gi::from_name(p.gi).unwrap(),
        voxels: p.voxels.unwrap_or(default_voxels(phone)),
        view: p.view,
        dragon: p.dragon,
        sdf_ao: p.sdf_ao,
        sdf_shadows: p.sdf_shadows,
        direct: p.direct,
        ..config
    }
}

/// The configuration the URL asks for: its preset, then its other parameters over it.
fn config_from_url(phone: bool) -> Config {
    let mut c = Config {
        gi: Gi::Screen(GiQuality::High),
        voxels: default_voxels(phone),
        view: View::Lit,
        dragon: Dragon::Off,
        animate: false,
        rug: true,
        textured: true,
        sdf_ao: 0.0,
        sdf_shadows: SdfShadows::Off,
        direct: Direct::Maps,
        rt_shadows: RtShadows::from_url(),
        slice: 0.6,
        reflect: flag("reflect", false),
        reflections: true,
        reflection_view: param("rt_view").and_then(|v| reflection_view(&v)).unwrap_or_default(),
        reflection_grid: param("rt_trace").as_deref() != Some("voxels"),
        reflection_quarter: param("rt_res").as_deref() == Some("quarter"),
        grid: false,
        rtgi: RtGi::from_url(),
        mirror: flag("mirror", true),
        glass: flag("glass", true),
        mirror_rough: param_or("mirror_rough", 0.0f32).clamp(0.0, 1.0),
        glass_ior: param_or("glass_ior", 1.5f32).max(1.0),
        glass_rough: param_or("glass_rough", 0.0f32).clamp(0.0, 1.0),
        glass_tint: param("glass_tint").and_then(|v| parse_rgb(&v)).unwrap_or(GLASS_TINT),
        glass_samples: param_or("glass_samples", 4u32).clamp(1, 16),
        glass_shadow: flag("glass_shadow", false),
        glass_radius: param_or("glass_radius", BALL_RADIUS).clamp(0.005, BALL_RADIUS),
        roll: flag("roll", false),
        screen_hits: flag("rt_screen", true),
        env: param("env").as_deref().and_then(Env::from_name).unwrap_or(Env::None),
        env_ev: param_or("env_ev", 0.0f32),
        env_rotation: param_or("env_rotation", 0.0f32),
        env_blur: param_or("env_blur", 0.0f32).clamp(0.0, 1.0),
    };
    if let Some(preset) = param("preset").or_else(|| param("gi").is_none().then(|| DEFAULT_PRESET.to_string())) {
        c = with_preset(c, &preset, phone);
    }
    if let Some(gi) = param("gi").as_deref().and_then(Gi::from_name) {
        c.gi = gi;
    }
    if let Some(q) = param("voxels").as_deref().and_then(VoxelGiQuality::from_name) {
        c.voxels = q;
    }
    if let Some(view) = param("view").as_deref().and_then(View::from_name) {
        c.view = view;
    }
    if let Some(dragon) = param("dragon").as_deref().and_then(Dragon::from_name) {
        c.dragon = dragon;
    }
    c.animate = param("animate").map_or(c.animate, |v| v == "1" || v == "on" || v == "true");
    if let Some(ao) = param("sdf_ao").and_then(|v| v.parse::<f32>().ok()) {
        c.sdf_ao = ao.clamp(0.0, 1.0);
    }
    if let Some(shadows) = param("sdf_shadows").as_deref().and_then(SdfShadows::from_name) {
        c.sdf_shadows = shadows;
    }
    if let Some(direct) = param("shadows").as_deref().and_then(Direct::from_name) {
        c.direct = direct;
    }
    if let Some(slice) = param("slice").and_then(|v| v.parse::<f32>().ok()) {
        c.slice = slice;
    }
    c.rug = param("rug").as_deref() != Some("off");
    c.textured = param("albedo").as_deref() != Some("constant");
    c.grid = c.reflect || c.gi == Gi::Rt || c.direct == Direct::Rt || flag("rt", false) || c.mirror || c.glass;
    c
}

/// The post-processing chain for `config`: its GI effect (none when off) and the tone mapping.
/// With an `environment`, every effect is lit by it (`set_environment`) and its cubemap, where it
/// has one, is the background. The room's materials have no sky ambient of their own, so none is
/// taken out where the GI replaces it.
fn build_effects(renderer: &Renderer, config: &Config, phone: bool, stats: bool, environment: Option<&EnvironmentLighting>) -> Vec<Box<dyn PostProcessingEffect>> {
    let indirect = config.view == View::Indirect;
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    // the lamp's direct light with ray-traced shadows (shadows=rt): first, so the GI and the
    // reflections' screen hits see it (and its visibility weighs the caustics)
    let mut visibility = None;
    if let (Direct::Rt, Some(grid)) = (config.direct, renderer.rt_grid()) {
        let mut shadows = RtShadowsEffect::new(grid.handle());
        let r = config.rt_shadows;
        shadows.view = r.view;
        shadows.half_resolution = !r.full;
        shadows.contact_length = r.contact;
        visibility = Some(shadows.visibility());
        effects.push(Box::new(shadows));
        if r.view != RtShadowsView::Lit {
            // its debug views alone: the direct light exposed as the image is, the visibility and
            // the mask as they are
            let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
            tone.exposure = if r.view == RtShadowsView::Direct { exposure_from_ev100(5.0) } else { 1.0 };
            effects.push(Box::new(ToneMapEffect::new(tone)));
            return effects;
        }
    }
    let screen = ScreenSpaceGIOptions { radius_m: 4.0, ambient_occlusion: 0.0, ..Default::default() };
    match config.gi {
        Gi::Off if config.view != View::Sdf => {}
        Gi::Screen(quality) if config.view != View::Sdf => {
            let mut effect = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, ..screen });
            effect.show_indirect = indirect;
            effects.push(Box::new(effect));
        }
        Gi::Rt if matches!(config.view, View::Lit | View::Indirect) => {
            // one ray for each 2 x 2 pixels through the room's grid, the voxels for the hits'
            // further bounces and past the box; the lamp from the renderer's spot lights
            let scene_gi = renderer.voxel_gi().expect("gi=rt enables voxel GI");
            let grid = renderer.rt_grid().expect("gi=rt builds the grid");
            let r = config.rtgi;
            let options = RtDiffuseGiOptions { resolution: r.resolution, hit_lighting: r.hit, shadows: r.shadows, denoise: r.denoise, kernel: r.kernel, max_distance: 20.0, ambient: 0.0, ..Default::default() };
            let mut effect = RtDiffuseGiEffect::with_volume(scene_gi.volume(), grid.handle(), options);
            effect.mode = r.mode;
            effect.accumulate = r.accumulate;
            effect.view = if indirect { RtGiView::Indirect } else { r.view };
            effect.set_spot_lights(Some(renderer.spot_lights_buffer()), renderer.spot_shadow_atlas());
            effect.heat_scale = 0.5 / exposure_from_ev100(5.0);
            effect.collect_stats = stats;
            effects.push(Box::new(effect));
        }
        _ => {
            // (the slice view shows through voxel GI's effect whatever the mode)
            let scene_gi = renderer.voxel_gi().expect("voxel GI is enabled for the voxel modes and the field");
            let near_quality = if phone || scene_gi.quality() == VoxelGiQuality::Low { GiQuality::Low } else { GiQuality::High };
            let near_field = config.gi.near_field().then_some(ScreenSpaceGIOptions { quality: near_quality, ..screen });
            let mut effect = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), near_field, material_ambient: 0.0, ..Default::default() });
            effect.show_indirect = indirect;
            effect.show_voxels = config.view == View::Voxels;
            effect.set_sdf(scene_gi.sdf());
            effect.sdf_ao = config.sdf_ao;
            effect.show_sdf_slice = (config.view == View::Sdf).then_some(config.slice);
            // the probes: the far field in the probe modes (and what the debug view shows)
            effect.set_probes(scene_gi.probes().filter(|_| config.needs_probes()));
            effect.show_probes = config.view == View::Probes;
            effects.push(Box::new(effect));
        }
    }
    // the glass ball's caustics on the light RtShadowsEffect lets through it (the casters and the
    // lights each frame)
    if let (Direct::Rt, true, true, Some(grid)) = (config.direct, config.rt_shadows.caustics, config.glass, renderer.rt_grid()) {
        let mut caustics = RtCausticsEffect::new(renderer.device(), grid.handle(), RtCausticsOptions::default());
        // (its triangles carry its tint, the light left after a metre)
        caustics.set_glass(config.glass_ior, [1.0; 3]);
        caustics.view = if config.rt_shadows.caustics_view { RtCausticsView::Caustics } else { RtCausticsView::Lit };
        // the light they add and take only where the lamp reaches
        caustics.set_visibility(visibility);
        effects.push(Box::new(caustics));
    }
    // the polished floor's reflections, the chrome ball and the glass, through the room's grid
    if let (true, Some(scene_gi), Some(grid)) = (config.traces(), renderer.voxel_gi(), renderer.rt_grid()) {
        let options = RtReflectionsOptions {
            resolution: if config.reflection_quarter { RtTraceResolution::Quarter } else { RtTraceResolution::Half },
            max_distance: 20.0,
            ..Default::default()
        };
        let mut reflections = RtReflectionsEffect::with_volume(scene_gi.volume(), grid.handle(), options);
        reflections.view = config.reflection_view;
        reflections.trace_grid = config.reflection_grid;
        reflections.heat_scale = 0.5 / exposure_from_ev100(5.0);
        reflections.collect_stats = stats;
        // the hits the camera sees: the lit image there; the others lit by the lamp (shadow rays
        // through the grid) and the voxels, with no indirect light where the image shows none
        reflections.screen_hits = config.screen_hits;
        reflections.hit_indirect = !matches!(config.gi, Gi::Off | Gi::Screen(_));
        if flag("rt_direct", true) {
            reflections.set_spot_lights(Some(renderer.spot_lights_buffer()));
        }
        reflections.set_glass(config.glass.then_some(RtGlass { samples: config.glass_samples, ..Default::default() }));
        effects.push(Box::new(reflections));
    }
    if let Some(environment) = environment {
        for effect in &mut effects {
            effect.set_environment(environment);
        }
        if environment.environment.is_some() {
            let mut background = EnvironmentBackgroundEffect::new(environment);
            background.blur = config.env_blur;
            effects.insert(0, Box::new(background));
        }
    }
    let mut tone = ToneMapOptions::for_surface(renderer.presentation_format());
    // the indirect view alone is 2 stops brighter
    tone.exposure = exposure_from_ev100(5.0) * if indirect { 4.0 } else { 1.0 };
    effects.push(Box::new(ToneMapEffect::new(tone)));
    effects
}

/// A Stanford dragon from `www/assets` as one geometry about 1.1 m long, standing on y = 0 and
/// centred on its origin, its parts merged with their glTF transforms baked in (None if it can't
/// be loaded).
async fn load_dragon(model: Dragon) -> Option<kansei_core::geometries::Geometry> {
    let result = match model {
        Dragon::Off => return None,
        Dragon::Light => GLTFLoader::load_glb(&fetch_bytes("assets/stanford_dragon_pbr.glb").await.ok()?).ok()?,
        Dragon::Full => {
            let json = fetch_bytes("assets/scene.gltf").await.ok()?;
            let bin = fetch_bytes("assets/scene.bin").await.ok()?;
            GLTFLoader::load_gltf_with_buffers(&json, vec![bin]).ok()?
        }
    };
    // 1.1 m across its wider side, standing on the floor at the centre
    let geometry = result.merged_geometry("Dragon").fit(glam::Vec3::new(1.1, f32::INFINITY, 1.1));
    let (lo, hi) = geometry.bounds();
    let extent = hi - lo;
    log::info!("dragon ({}): {} triangles, {:.2} x {:.2} x {:.2} m", model.name(), geometry.index_count() / 3, extent.x, extent.y, extent.z);
    Some(geometry)
}

impl State {
    /// Bring the renderer, the chain and the scene to `config`.
    fn apply(&mut self, config: Config) {
        let previous = self.config;
        self.config = config;
        // voxel GI: enabled at the asked tier for the voxel modes (and the distance field), freed
        // for the others
        if config.needs_voxels() {
            if self.enabled_voxels != Some(config.voxels) {
                self.renderer.enable_voxel_gi(SceneVoxelGiOptions {
                    quality: config.voxels,
                    bounds_min: [-2.3, -0.3, -4.3],
                    bounds_max: [2.3, 4.3, 0.3],
                    radiance_scale: 1.0,
                    budget_bytes: if self.phone { 24 << 20 } else { 0 },
                });
                self.enabled_voxels = Some(config.voxels);
                self.gi_generation += 1;
            }
        } else if self.enabled_voxels.is_some() {
            self.renderer.disable_voxel_gi();
            self.enabled_voxels = None;
        }
        if let Some(gi) = self.renderer.voxel_gi_mut() {
            if config.needs_sdf() {
                if gi.sdf().is_none() {
                    gi.enable_sdf();
                    self.gi_generation += 1;
                }
            } else {
                gi.disable_sdf();
            }
            gi.settings.sdf_shadows = config.sdf_shadows;
            if config.needs_probes() {
                gi.enable_probes(probe_options(self.phone));
            } else {
                gi.disable_probes();
            }
        }
        let environment = self.environment();
        self.renderer.set_environment(environment.as_ref());
        self.volume.effects = build_effects(&self.renderer, &config, self.phone, self.stats.is_some(), environment.as_ref());
        if let Some(r) = self.scene.get_renderable_mut(self.dome) {
            r.visible = config.env == Env::Gradient;
        }
        self.set_env_settings(config);
        // the lit materials: shadowed through the field (bound to this one), by the maps, or
        // leaving the lamp to the ray-traced shadows
        let wanted = (config.direct, (config.direct == Direct::Sdf).then_some(self.gi_generation));
        if wanted != self.materials || config.textured != previous.textured {
            let sdf = (config.direct == Direct::Sdf).then(|| self.renderer.voxel_gi().and_then(SdfBinding::of)).flatten();
            let rt = config.direct == Direct::Rt;
            for &(index, label, albedo) in &self.lit {
                if let Some(r) = self.scene.get_renderable_mut(index) {
                    r.material = lit_material(label, albedo, sdf.as_ref(), rt, if label == "Floor" { self.floor_specular } else { [0.0; 2] });
                    r.material_dirty = true;
                }
            }
            if let Some(r) = self.scene.get_renderable_mut(self.rug) {
                r.material = rug_material(config.textured, sdf.as_ref(), rt);
                r.material_dirty = true;
            }
            self.materials = wanted;
        }

        if let Some(r) = self.scene.get_renderable_mut(self.rug) {
            r.visible = config.rug;
        }
        for (index, on) in [(self.mirror, config.mirror), (self.glass, config.glass)] {
            if let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) {
                r.visible = on;
            }
        }
        if let Some(r) = self.glass.and_then(|i| self.scene.get_renderable_mut(i)) {
            r.cast_shadow = config.glass_shadow;
            r.dynamic = config.roll;
        }
        self.set_ball_materials(config);
        if config.textured != previous.textured {
            // the voxelizer can't see a material change
            if let Some(gi) = self.renderer.voxel_gi_mut() {
                gi.invalidate();
            }
        }
        for (slot, index) in self.dragons.iter().enumerate() {
            if let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) {
                r.visible = config.dragon.slot() == Some(slot);
            }
        }
        // the animated renderable is drawn (and voxelized) live; the others rest
        let animated = self.animated();
        let rests = [(Some(self.tall), self.tall_rest), (self.dragons[0], self.dragon_rest), (self.dragons[1], self.dragon_rest)];
        for (index, rest) in rests {
            let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) else { continue };
            r.dynamic = config.animate && index == Some(animated);
            if !r.dynamic {
                r.object.position = rest.0;
                r.object.rotation.y = rest.1;
            }
        }
    }

    /// The environment `config.env` names (None: black; the HDRI's until it is fetched).
    fn environment(&self) -> Option<EnvironmentLighting> {
        match self.config.env {
            Env::None => None,
            Env::Gradient => Some(self.gradient.lighting()),
            Env::Hdri => self.hdri.as_ref().map(|h| h.lighting()),
        }
    }

    /// The environment's intensity, turn and blur to `config`'s (alone: the chain goes on as it
    /// is; the HDRI is prepared again before the next frame).
    fn set_env_settings(&mut self, config: Config) {
        self.config = config;
        let (up, down) = config.studio();
        self.gradient.set(self.renderer.queue(), up, down);
        if let Some(buffer) = self.scene.get_renderable(self.dome).and_then(|r| r.material.bindable_buffer(0)) {
            let bytes: Vec<u8> = dome_uniform(up, down).iter().flat_map(|v| v.to_le_bytes()).collect();
            self.renderer.queue().write_buffer(&buffer, 0, &bytes);
        }
        if let Some(hdri) = &mut self.hdri {
            hdri.set_settings(config.hdri_settings());
        }
        if let Some(background) = self.volume.effect_mut::<EnvironmentBackgroundEffect>() {
            background.blur = config.env_blur;
        }
    }

    /// The balls' materials to `config`'s (alone: the effects, and the frosted glass's history, go
    /// on as they are).
    fn set_ball_materials(&mut self, config: Config) {
        self.config = config;
        if let Some(caustics) = self.volume.effect_mut::<RtCausticsEffect>() {
            caustics.set_glass(config.glass_ior, [1.0; 3]);
        }
        for (index, material) in [(self.mirror, config.mirror_material()), (self.glass, config.glass_material())] {
            if let Some(r) = index.and_then(|i| self.scene.get_renderable_mut(i)) {
                r.material.set_standard_lit(&self.renderer, &material);
            }
        }
    }

    /// The renderable `animate` moves: the dragon when it is shown, the tall block otherwise.
    fn animated(&self) -> usize {
        self.config.dragon.slot().and_then(|slot| self.dragons[slot]).unwrap_or(self.tall)
    }

    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let now = now() * 1000.0;
        let dt = frame.dt.clamp(0.0, 0.1);
        if self.config.animate {
            self.time += dt;
            let animated = self.animated();
            let rest = if animated == self.tall { self.tall_rest } else { self.dragon_rest };
            let t = self.time;
            if let Some(r) = self.scene.get_renderable_mut(animated) {
                // turn, and slide back and forth across the box
                r.object.position = Vec3::new(rest.0.x + 0.45 * (t * 0.7).sin(), rest.0.y, rest.0.z + 0.25 * (t * 0.45).sin());
                r.object.rotation.y = rest.1 + t * 0.6;
            }
        }
        self.controls.update(&mut self.camera, dt);
        if self.env_spin != 0.0 && self.hdri.is_some() {
            let config = Config { env_rotation: (self.config.env_rotation + self.env_spin * dt) % 360.0, ..self.config };
            self.set_env_settings(config);
        }
        // the HDRI's cubemaps and SH, prepared again when its settings changed
        if let Some(hdri) = &mut self.hdri {
            hdri.prepare(self.renderer.device(), self.renderer.queue());
        }
        if let (true, Some(r)) = (self.config.roll, self.glass.and_then(|i| self.scene.get_renderable_mut(i))) {
            // back and forth across the rug, turning as it rolls
            self.roll_time += dt;
            let radius = self.config.glass_radius;
            let x = 0.5 * (radius / BALL_RADIUS).max(0.2) * (self.roll_time * 0.5).sin();
            r.object.position = glass_centre(radius, x);
            r.object.rotation.z = -x / radius;
        }
        if let Some(shadows) = self.volume.effect_mut::<RtShadowsEffect>() {
            shadows.update_lights(self.scene.lights());
        }
        let ball = self.glass.and_then(|i| self.scene.get_renderable(i)).filter(|r| r.visible).map(|r| {
            let c = glam::Vec3::new(r.object.position.x, r.object.position.y, r.object.position.z);
            (c - glam::Vec3::splat(self.config.glass_radius), c + glam::Vec3::splat(self.config.glass_radius))
        });
        if let Some(caustics) = self.volume.effect_mut::<RtCausticsEffect>() {
            caustics.update_lights(self.scene.lights());
            caustics.set_casters(ball.as_slice());
        }
        if let Some(gi) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            gi.update_lights(self.scene.lights());
        }
        // (the lamp is a spot light, which the reflections read from the renderer; this hands
        // them any other)
        if let Some(reflections) = self.volume.effect_mut::<RtReflectionsEffect>() {
            reflections.update_lights(self.scene.lights());
        }
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, &mut self.volume);
        if let Some(stats) = &mut self.stats {
            stats.frames += 1;
            if now - stats.since >= 1000.0 {
                stats.frame_ms = (now - stats.since) / stats.frames as f64;
                stats.frames = 0;
                stats.since = now;
                let profile = self.renderer.take_profile();
                if profile.gpu_frames > 0 {
                    stats.passes = profile.top_passes(usize::MAX);
                    stats.gpu_ms = profile.gpu_ms;
                    stats.gpu_span_ms = profile.gpu_span_ms;
                }
            }
        }
    }

    /// Triangles drawn per view: every visible renderable's.
    fn triangles(&self) -> u64 {
        self.scene.ordered_indices().filter_map(|i| self.scene.get_renderable(i)).filter(|r| r.visible).map(|r| r.geometry.index_count() as u64 / 3 * r.geometry.instance_count.max(1) as u64).sum()
    }

    /// The grid's figures as JSON (null without it).
    fn grid_info(&self) -> String {
        let Some(rt) = self.renderer.rt_grid() else { return "null".into() };
        let g = rt.stats();
        format!(
            "{{\"triangles\":{},\"references\":{},\"big\":{},\"sources\":{},\"rebuilt\":{},\"rebuilds\":{},\"cpu_ms\":{:.3},\"mib\":{:.1}}}",
            g.grid.triangles,
            g.grid.references,
            g.grid.big_triangles,
            g.sources,
            g.rebuilt,
            g.rebuilds,
            g.cpu_ms,
            rt.memory_bytes() as f64 / (1 << 20) as f64,
        )
    }

    /// The reflections' figures as JSON (null without `reflect=1`).
    fn reflect_info(&self) -> String {
        if !self.config.reflect {
            return "null".into();
        }
        let r = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtReflectionsEffect>());
        let s = r.and_then(|r| r.stats()).unwrap_or_default();
        format!(
            "{{\"enabled\":{},\"view\":\"{}\",\"grid\":{},\"res\":\"{}\",\"rays\":{},\"hits\":{},\"cells\":{},\"tests\":{},\"max_cost\":{}}}",
            self.config.reflections,
            reflection_view_name(self.config.reflection_view),
            self.config.reflection_grid,
            if self.config.reflection_quarter { "quarter" } else { "half" },
            s.rays,
            s.hits,
            s.cells,
            s.tests,
            s.max_cost
        )
    }

    /// The balls' settings and the glass pass's counters (null without the grid).
    fn balls_info(&self) -> String {
        if self.mirror.is_none() {
            return "null".into();
        }
        let r = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtReflectionsEffect>());
        let s = r.and_then(|r| r.stats()).unwrap_or_default();
        let c = &self.config;
        format!(
            "{{\"mirror\":{},\"glass\":{},\"mirror_rough\":{},\"glass_ior\":{},\"glass_rough\":{},\"glass_tint\":{:?},\"glass_samples\":{},\"screen_hits\":{},\"rays\":{},\"glass_pixels\":{},\"glass_cost\":{}}}",
            c.mirror, c.glass, c.mirror_rough, c.glass_ior, c.glass_rough, c.glass_tint, c.glass_samples, c.screen_hits, s.rays, s.glass_pixels, s.glass_cost
        )
    }

    /// The hybrid's settings, and while it runs its counters, memory and accumulated frames.
    fn rtgi_info(&self) -> String {
        let e = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtDiffuseGiEffect>());
        let s = e.and_then(|e| e.stats()).unwrap_or_default();
        format!(
            "{{{},\"on\":{},\"accumulated\":{},\"mib\":{:.1},\"rays\":{},\"hits\":{},\"cost\":{},\"shadow_rays\":{}}}",
            self.config.rtgi.json(),
            e.is_some(),
            e.map_or(0, |e| e.accumulated()),
            e.map_or(0.0, |e| e.memory_bytes() as f64 / (1 << 20) as f64),
            s.rays,
            s.hits,
            s.cost,
            s.shadow_rays
        )
    }

    /// The environment's settings, and the HDRI's size and load times (null until fetched).
    fn env_info(&self) -> String {
        let c = &self.config;
        let hdri = self.hdri.as_ref().map_or("null".into(), |h| {
            let (w, h2) = h.image_size();
            format!("{{\"image\":[{w},{h2}],\"face\":{},\"mib\":{:.1},\"fetch_decode_ms\":{:.1},\"upload_ms\":{:.1}}}", h.face_size(), h.memory_bytes() as f64 / (1 << 20) as f64, self.hdri_load_ms.0, self.hdri_load_ms.1)
        });
        format!("{{\"mode\":\"{}\",\"ev\":{},\"rotation\":{},\"blur\":{},\"spin\":{},\"hdri\":{hdri}}}", c.env.name(), c.env_ev, c.env_rotation, c.env_blur, self.env_spin)
    }

    /// The glass ball's caustics: the photons and the tile last frame (null when they don't run).
    fn caustics_info(&self) -> String {
        let Some(c) = self.volume.effect::<RtCausticsEffect>() else { return "null".into() };
        let (pairs, photons, fed) = c.photons_sent();
        let tile = c.tile(0).map_or("null".into(), |(_, texel, dims, history)| format!("{{\"texel_mm\":{:.2},\"dims\":{dims:?},\"history\":{history}}}", texel * 1e3));
        format!("{{\"lights\":{pairs},\"photons\":{photons},\"fed\":{fed},\"tile\":{tile}}}")
    }

    fn info(&self) -> String {
        let gi = self.renderer.voxel_gi();
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        // (+ 0.0: an empty sum is -0)
        let sum = |prefix: &str| self.stats.as_ref().map_or(0.0, |s| s.passes.iter().filter(|p| p.0.starts_with(prefix)).map(|p| p.1).sum::<f64>()) + 0.0;
        format!(
            "{{\"env\":{},\"env_prepare_ms\":{:.3},\"env_background_ms\":{:.3},\"grid\":{},\"rtgi\":{},\"rtgi_ms\":{:.3},\"reflect\":{},\"balls\":{},\"rt_ms\":{:.3},\"reflect_ms\":{:.3},\"glass_ms\":{:.3},\"gi\":\"{}\",\"view\":\"{}\",\"voxels\":\"{}\",\"voxel_tier\":{},\"dims\":{},\"mib\":{:.1},\"dragon\":\"{}\",\"animate\":{},\"rug\":{},\"textured\":{},\"sdf_ao\":{},\"sdf_shadows\":\"{}\",\"shadows\":\"{}\",\"rt_shadows\":{},\"rt_shadows_ms\":{:.3},\"caustics\":{},\"caustics_ms\":{:.3},\"slice\":{},\"sdf\":{},\"sdf_ms\":{:.3},\"probes\":{},\"probe_dims\":{},\"probes_ms\":{:.3},\"triangles\":{},\"stats\":{},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"voxelize_ms\":{:.3},\"inject_ms\":{:.3},\"mips_ms\":{:.3},\"screen_ms\":{:.3},\"ssgi_ms\":{:.3},\"passes\":[{}]}}",
            self.env_info(),
            sum("Environment/Cube") + sum("Environment/Prefilter") + sum("Environment/SH"),
            sum("Environment/Background"),
            self.grid_info(),
            self.rtgi_info(),
            sum("RtGi/"),
            self.reflect_info(),
            self.balls_info(),
            sum("Rt/Gather") + sum("Rt/Grid"),
            sum("Rt/Trace") + sum("Rt/Resolve"),
            sum("Rt/Glass"),
            self.config.gi.name(),
            self.config.view.name(),
            self.config.voxels.name(),
            gi.map_or("null".into(), |g| format!("\"{}\"", g.quality().name())),
            gi.map_or("null".into(), |g| format!("{:?}", g.volume().dims())),
            gi.map_or(0.0, |g| g.memory_bytes() as f64 / (1 << 20) as f64),
            self.config.dragon.name(),
            self.config.animate,
            self.config.rug,
            self.config.textured,
            self.config.sdf_ao,
            self.config.sdf_shadows.name(),
            self.config.direct.name(),
            self.config.rt_shadows.json(),
            sum("RtShadows/"),
            self.caustics_info(),
            sum("RtCaustics/"),
            self.config.slice,
            gi.and_then(|g| g.sdf()).is_some(),
            sum("VoxelGI/Sdf"),
            gi.and_then(|g| g.probes()).is_some(),
            gi.and_then(|g| g.probes()).map_or("null".into(), |p| format!("{:?}", p.dims())),
            sum("VoxelGI/Probes"),
            self.triangles(),
            self.stats.is_some(),
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            sum("VoxelGI/Voxelize"),
            sum("VoxelGI/Inject"),
            sum("VoxelGI/Mips") + sum("VoxelGI/AnisotropicMips"),
            sum("VoxelGI/Screen"),
            sum("SSGI"),
            passes.join(","),
        )
    }
}

/// The studio gradient's dome (`Material::gradient_sky`'s uniform): `up` at the zenith, `down`
/// below the horizon, their mean at it.
fn dome_uniform(up: [f32; 3], down: [f32; 3]) -> [f32; 12] {
    let o = GradientSkyOptions { zenith: up, horizon: std::array::from_fn(|c| (up[c] + down[c]) * 0.5), ground: down, curve: 0.5 };
    let ([zr, zg, zb], [hr, hg, hb], [gr, gg, gb]) = (o.zenith, o.horizon, o.ground);
    [zr, zg, zb, o.curve, hr, hg, hb, 0.0, gr, gg, gb, 0.0]
}

/// Fetch the studio HDRI and upload it: the environment, and the ms its fetch and decode took and
/// its upload (None if it can't be fetched or read).
async fn load_hdri(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<(HdrEnvironment, (f64, f64))> {
    let t = now();
    let bytes = fetch_bytes(HDRI_ASSET).await.map_err(|e| log::warn!("{HDRI_ASSET}: {e:?}")).ok()?;
    let image = kansei_core::environment::HdrImage::decode(&bytes).map_err(|e| log::warn!("{e}")).ok()?;
    let decoded = now();
    let hdri = HdrEnvironment::new(device, queue, &image, HdrEnvironmentOptions::default());
    let ms = ((decoded - t) * 1000.0, (now() - decoded) * 1000.0);
    log::info!("HDRI: {} x {}, fetched and decoded in {:.0} ms, uploaded in {:.1} ms; cube faces of {}", image.width, image.height, ms.0, ms.1, hdri.face_size());
    Some((hdri, ms))
}

/// Add the dragon to the scene (hidden until the config shows it), returning its index.
const DRAGON_ALBEDO: [f32; 3] = [0.75, 0.62, 0.42];

/// With `grid`, the dragon goes into the room's grid of triangles too, by the cut of its cluster
/// LOD at a cell of error (`rt_lod=0`: whole; the full scan's graph takes seconds to build).
fn add_dragon(scene: &mut Scene, geometry: kansei_core::geometries::Geometry, rest: (Vec3, f32), grid: bool) -> usize {
    let albedo = DRAGON_ALBEDO;
    let clusters = (grid && flag("rt_lod", true)).then(|| {
        let t = now();
        let mesh = ClusterMesh::build(&geometry, &ClusterOptions::default());
        log::info!("dragon: cluster LOD of {} triangles in {:.0} ms", geometry.index_count() / 3, (now() - t) * 1000.0);
        ClusterLod::new(mesh)
    });
    let mut dragon = Renderable::new(geometry, lit_material("Dragon", albedo, None, false, [0.0; 2])).with_gi(GiSurface::new(albedo));
    dragon.clusters = clusters;
    // its hits shaded with its vertex normals, not its cut's facets
    dragon.rt = grid.then(|| RtSurface::new(albedo).with_smooth_normals());
    dragon.object.position = rest.0;
    dragon.object.rotation.y = rest.1;
    dragon.visible = false;
    scene.add(SceneNode::Renderable(dragon))
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    // `warm=1`: the pipelines `www/pipelines.json` recorded (`record_pipelines=1`) compiled before
    // the device comes (kansei_wasm::pipelines)
    if param("warm").as_deref() == Some("1") {
        if let Ok(bytes) = kansei_wasm::fetch_bytes("pipelines.json").await {
            kansei_wasm::pipelines::warm(std::str::from_utf8(&bytes).unwrap_or("{}")).await;
        }
    }
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_spot_shadows(2048, 1);
    let phone = is_phone();
    let config = config_from_url(phone);
    // a grid of the room's triangles for the hybrid GI's rays (gi=rt, rt=1) and the reflections
    // (reflect=1: the floor polished)
    if config.grid {
        renderer.enable_rt_grid(rt_grid_options());
        renderer.rt_grid_mut().unwrap().set_moving_layer(param("rt_moving").as_deref() != Some("0"));
    }
    let floor_specular = if config.reflect { floor_specular() } else { [0.0; 2] };
    let rt = |albedo: [f32; 3]| config.grid.then(|| RtSurface::new(albedo));

    // the room: 4 m wide, high and deep, open toward the camera (z = 0), walls 0.2 m thick
    let mut scene = Scene::new();
    let mut lit = Vec::new();
    let white = [0.73, 0.73, 0.73];
    let slabs: [(&str, [f32; 3], [f32; 3], [f32; 3]); 5] = [
        ("Floor", [4.4, 0.2, 4.2], [0.0, -0.1, -2.1], white),
        ("Ceiling", [4.4, 0.2, 4.2], [0.0, 4.1, -2.1], white),
        ("Back", [4.4, 4.4, 0.2], [0.0, 2.0, -4.1], white),
        ("Left", [0.2, 4.4, 4.2], [-2.1, 2.0, -2.1], [0.63, 0.065, 0.05]),
        ("Right", [0.2, 4.4, 4.2], [2.1, 2.0, -2.1], [0.14, 0.45, 0.09]),
    ];
    for (label, size, position, color) in slabs {
        let specular = if label == "Floor" { floor_specular } else { [0.0; 2] };
        let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), lit_material(label, color, None, false, specular)).with_gi(GiSurface::new(color));
        slab.object.set_position(position[0], position[1], position[2]);
        slab.rt = rt(color);
        lit.push((scene.add(SceneNode::Renderable(slab)), label, color));
    }
    // a tall block near the red wall and a short one near the green wall
    let tall_rest = (Vec3::new(-0.75, 1.2, -2.6), 0.33);
    let mut tall = Renderable::new(BoxGeometry::new(1.2, 2.4, 1.2), lit_material("Tall", white, None, false, [0.0; 2])).with_gi(GiSurface::new(white));
    tall.rt = rt(white);
    tall.object.position = tall_rest.0;
    tall.object.rotation.y = tall_rest.1;
    let tall = scene.add(SceneNode::Renderable(tall));
    lit.push((tall, "Tall", white));
    let mut short = Renderable::new(BoxGeometry::new(1.2, 1.2, 1.2), lit_material("Short", white, None, false, [0.0; 2])).with_gi(GiSurface::new(white));
    short.rt = rt(white);
    short.object.set_position(0.8, 0.6, -1.5);
    short.object.rotation.y = -0.3;
    lit.push((scene.add(SceneNode::Renderable(short)), "Short", white));
    // a textured rug on the floor in front of the blocks, orange before the tall one and blue
    // before the short one, whose sides facing the camera only bounces light; its constant
    // surface (its mean colour) is used where the material's voxel entry is not
    let mean: [f32; 3] = std::array::from_fn(|c| (RUG_LEFT[c] + RUG_RIGHT[c]) * 0.5);
    let mut rug = Renderable::new(BoxGeometry::new(2.4, 0.02, 0.8), rug_material(config.textured, None, false)).with_gi(GiSurface::new(mean));
    rug.object.set_position(0.0, 0.01, -0.55);
    rug.rt = rt(mean);
    let rug = scene.add(SceneNode::Renderable(rug));
    // with the grid: a chrome ball on the short block and a glass ball on the rug before it
    let (mut mirror, mut glass) = (None, None);
    if config.grid {
        let sphere = || SphereGeometry::new(BALL_RADIUS, BALL_SEGMENTS, BALL_RINGS);
        // (dark to the GI's rays and the voxels: they don't mirror; its own rays follow its
        // vertex normals, as the hits on it do)
        let mut ball = Renderable::new(sphere(), Material::standard_lit("Chrome", &config.mirror_material())).with_gi(GiSurface::new(MIRROR_GI_ALBEDO));
        ball.rt = Some(RtSurface::new(MIRROR_GI_ALBEDO).with_smooth_normals());
        ball.object.set_position(0.8, 1.2 + BALL_RADIUS, -1.5);
        ball.visible = config.mirror;
        mirror = Some(scene.add(SceneNode::Renderable(ball)));
        // clear to the shadow maps, the voxels and the GI's rays; refracted by the glass pass
        let mut ball = Renderable::new(sphere(), Material::standard_lit("Glass", &config.glass_material()));
        ball.rt = Some(RtSurface::glass(config.glass_tint));
        ball.cast_shadow = false;
        ball.object.position = glass_centre(config.glass_radius, 0.0);
        let s = config.glass_radius / BALL_RADIUS;
        ball.object.scale = Vec3::new(s, s, s);
        ball.visible = config.glass;
        glass = Some(scene.add(SceneNode::Renderable(ball)));
    }
    // the dragon, in the free corner at the front left, when asked for
    let dragon_rest = (Vec3::new(-1.15, 0.0, -1.25), 0.5);
    let mut dragons = [None, None];
    if let Some(slot) = config.dragon.slot() {
        dragons[slot] = load_dragon(config.dragon).await.map(|g| add_dragon(&mut scene, g, dragon_rest, config.grid));
        if let Some(index) = dragons[slot] {
            lit.push((index, "Dragon", DRAGON_ALBEDO));
        }
    }

    // the key light: a shadowed downlight just under the ceiling's centre, wide enough to light
    // the tops of the side walls
    let mut lamp = SpotLight::new(Vec3::new(0.0, 3.85, -2.0), Vec3::new(0.0, -1.0, 0.0), Vec3::new(1.0, 0.92, 0.8), 1600.0, 12.0, 50f32.to_radians(), 76f32.to_radians());
    lamp.cast_shadow = true;
    lamp.source_radius = 0.15;
    lamp.volumetric_scale = 0.0;
    scene.add(SceneNode::Light(Light::Spot(lamp)));

    // the studio gradient's dome round the box (env=gradient), out of the GI, the grid and the
    // shadows
    let (up, down) = config.studio();
    let gradient = GradientEnvironment::new(renderer.device(), up, down);
    let dome_options = GradientSkyOptions { zenith: up, horizon: std::array::from_fn(|c| (up[c] + down[c]) * 0.5), ground: down, curve: 0.5 };
    let mut dome = Renderable::new(SphereGeometry::new(60.0, 48, 24), Material::gradient_sky("Dome", &dome_options));
    dome.cast_shadow = false;
    dome.visible = false;
    let dome = scene.add(SceneNode::Renderable(dome));
    let (hdri, hdri_load_ms) = match config.env {
        Env::Hdri => load_hdri(renderer.device(), renderer.queue()).await.map_or((None, (0.0, 0.0)), |(h, ms)| (Some(h), ms)),
        _ => (None, (0.0, 0.0)),
    };

    let camera = Camera::new(38.0, 0.1, 100.0, canvas.aspect());
    let (_, target, distance, azimuth, elevation) = CAMERAS.iter().find(|c| Some(c.0) == param("cam").as_deref()).copied().unwrap_or(CAMERAS[0]);
    let mut controls = CameraControls::from_canvas(canvas.element(), Vec3::new(target[0], target[1], target[2]), distance).with_mouse_pan(canvas.element());
    controls.set_view(Vec3::new(target[0], target[1], target[2]), distance, azimuth, elevation);

    let stats = flag("stats", false).then(|| Stats { since: now() * 1000.0, ..Default::default() });
    if stats.is_some() {
        renderer.set_profiling(true);
    }
    let volume = PostProcessingVolume::new(&renderer, Vec::new());
    let mut state = State {
        renderer,
        scene,
        camera,
        controls,
        volume,
        config,
        phone,
        enabled_voxels: None,
        gi_generation: 0,
        materials: (Direct::Maps, None),
        lit,
        tall,
        rug,
        mirror,
        glass,
        dragons,
        tall_rest,
        dragon_rest,
        time: 0.0,
        stats,
        floor_specular,
        gradient,
        dome,
        hdri,
        hdri_load_ms,
        env_spin: param_or("env_spin", 0.0f32),
        roll_time: 0.0,
    };
    state.apply(config);
    log::info!("Kansei — GI box (WASM) ready: {}", state.info());

    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The state as JSON: the configuration, the volume, the triangles and (with stats) the times.
#[wasm_bindgen]
pub fn info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// The presets, as JSON `[[name, label], ...]`.
#[wasm_bindgen]
pub fn presets() -> String {
    let items: Vec<String> = PRESETS.iter().map(|p| format!("[\"{}\",\"{}\"]", p.name, p.label)).collect();
    format!("[{}]", items.join(","))
}

/// Apply preset `name` (`PRESETS`): GI mode, voxel tier, view and the dragon together.
#[wasm_bindgen]
pub async fn set_preset(name: String) {
    // (the hybrid needs the grid, built at load: without it the GI stays as it is)
    let Some(config) = with_state(|s| {
        let c = with_preset(s.config, &name, s.phone);
        if c.gi == Gi::Rt && !s.config.grid { Config { gi: s.config.gi, ..c } } else { c }
    }) else {
        return;
    };
    apply_with_dragon(config).await;
}

/// `gi=` at run time.
#[wasm_bindgen]
pub fn set_gi(name: &str) {
    if let Some(gi) = Gi::from_name(name) {
        // rt needs the grid, built at load only (gi=rt, rt=1 or reflect=1 in the URL)
        with_state(|s| {
            if gi != Gi::Rt || s.config.grid {
                s.apply(Config { gi, ..s.config });
            }
        });
    }
}

/// One of the hybrid's settings at run time, by its `rtgi_*` URL parameter's key (`res`,
/// `denoise`, `kernel`, `hit`, `shadows`, `mode`, `accum`, `view`) and value.
#[wasm_bindgen]
pub fn set_rtgi(key: &str, value: &str) {
    with_state(|s| {
        let mut config = s.config;
        if config.rtgi.set(key, value) && config != s.config {
            s.apply(config);
        }
    });
}

/// `voxels=` at run time.
#[wasm_bindgen]
pub fn set_voxels(name: &str) {
    if let Some(voxels) = VoxelGiQuality::from_name(name) {
        with_state(|s| s.apply(Config { voxels, ..s.config }));
    }
}

/// `view=lit|indirect|voxels` at run time.
#[wasm_bindgen]
pub fn set_view(name: &str) {
    if let Some(view) = View::from_name(name) {
        with_state(|s| s.apply(Config { view, ..s.config }));
    }
}

/// `dragon=off|light|full`: hide the dragon, or show the light (19k triangles) or full (871k)
/// one, loading it the first time.
#[wasm_bindgen]
pub async fn set_dragon(name: String) {
    let Some(dragon) = Dragon::from_name(&name) else { return };
    let Some(config) = with_state(|s| Config { dragon, ..s.config }) else { return };
    apply_with_dragon(config).await;
}

/// Apply `config`, loading its dragon first if it isn't loaded.
async fn apply_with_dragon(config: Config) {
    if let Some(slot) = config.dragon.slot() {
        if with_state(|s| s.dragons[slot].is_none()) == Some(true) {
            if let Some(geometry) = load_dragon(config.dragon).await {
                with_state(|s| {
                    if s.dragons[slot].is_none() {
                        let index = add_dragon(&mut s.scene, geometry, s.dragon_rest, s.config.grid);
                        s.dragons[slot] = Some(index);
                        s.lit.push((index, "Dragon", DRAGON_ALBEDO));
                        // its material follows the others' at the next apply
                        s.materials.1 = Some(u32::MAX);
                    }
                });
            }
        }
    }
    with_state(|s| s.apply(config));
}

/// Move the dragon (or the tall block) around the box, re-voxelized every frame.
#[wasm_bindgen]
pub fn set_animate(on: bool) {
    with_state(|s| s.apply(Config { animate: on, ..s.config }));
}

#[wasm_bindgen]
pub fn set_rug(on: bool) {
    with_state(|s| s.apply(Config { rug: on, ..s.config }));
}

/// The rug's voxels from its texture (true) or its mean colour.
#[wasm_bindgen]
pub fn set_textured(on: bool) {
    with_state(|s| s.apply(Config { textured: on, ..s.config }));
}

/// Camera preset `front|corner|low|floor|glass|chrome` (`CAMERAS`).
#[wasm_bindgen]
pub fn set_camera(name: &str) {
    if let Some(&(_, t, distance, azimuth, elevation)) = CAMERAS.iter().find(|c| c.0 == name) {
        with_state(|s| s.controls.set_view(Vec3::new(t[0], t[1], t[2]), distance, azimuth, elevation));
    }
}

/// The distance field's AO on voxel GI, 0 (none) to 1.
#[wasm_bindgen]
pub fn set_sdf_ao(strength: f32) {
    with_state(|s| s.apply(Config { sdf_ao: strength.clamp(0.0, 1.0), ..s.config }));
}

/// The voxels' shadows through the distance field: `off|fallback|always`.
#[wasm_bindgen]
pub fn set_sdf_shadows(name: &str) {
    if let Some(sdf_shadows) = SdfShadows::from_name(name) {
        with_state(|s| s.apply(Config { sdf_shadows, ..s.config }));
    }
}

/// The direct light's shadows: `maps` (the shadow atlas), `sdf` (the distance field, through the
/// material helper `gi::SDF_WGSL`) or `rt` (ray traced through the grid by `RtShadowsEffect`;
/// with the grid only, built at load: the page reloads with `shadows=rt` otherwise).
#[wasm_bindgen]
pub fn set_shadows(name: &str) {
    let Some(direct) = Direct::from_name(name) else { return };
    with_state(|s| {
        if direct != Direct::Rt || s.config.grid {
            s.apply(Config { direct, ..s.config })
        }
    });
}

/// One of `RtShadowsEffect`'s settings (`view`: `lit|visibility|direct|mask`; `res`: `half|full`;
/// `contact`: metres) and value.
#[wasm_bindgen]
pub fn set_rt_shadows(key: &str, value: &str) {
    with_state(|s| {
        let mut config = s.config;
        if config.rt_shadows.set(key, value) {
            s.apply(config);
        }
    });
}

/// The SDF slice's height, metres (`view=sdf`).
#[wasm_bindgen]
pub fn set_slice(height: f32) {
    with_state(|s| s.apply(Config { slice: height, ..s.config }));
}

/// Profile every pass (the stats in `info`).
#[wasm_bindgen]
pub fn set_stats(on: bool) {
    with_state(|s| {
        s.renderer.set_profiling(on);
        s.stats = on.then(|| Stats { since: now() * 1000.0, ..Default::default() });
    });
}

/// What moves into the grid's moving layer (on) or the whole grid rebuilt every frame (off).
#[wasm_bindgen]
pub fn set_rt_moving_layer(on: bool) {
    with_state(|s| {
        if let Some(grid) = s.renderer.rt_grid_mut() {
            grid.set_moving_layer(on);
        }
    });
}

/// The polished floor's reflections on or off (with `reflect=1`).
#[wasm_bindgen]
pub fn set_reflections(on: bool) {
    with_state(|s| s.apply(Config { reflections: on, ..s.config }));
}

/// The chrome ball (only where the page built the grid).
#[wasm_bindgen]
pub fn set_mirror(on: bool) {
    with_state(|s| {
        if s.mirror.is_some() {
            s.apply(Config { mirror: on, ..s.config });
        }
    });
}

/// The glass ball (only where the page built the grid).
#[wasm_bindgen]
pub fn set_glass(on: bool) {
    with_state(|s| {
        if s.glass.is_some() {
            s.apply(Config { glass: on, ..s.config });
        }
    });
}

/// The chrome ball's roughness, 0 (a mirror) to 1.
#[wasm_bindgen]
pub fn set_mirror_rough(roughness: f32) {
    with_state(|s| s.set_ball_materials(Config { mirror_rough: roughness.clamp(0.0, 1.0), ..s.config }));
}

/// The glass's index of refraction (1: air, 1.5: window glass, 2.4: diamond).
#[wasm_bindgen]
pub fn set_glass_ior(ior: f32) {
    with_state(|s| s.set_ball_materials(Config { glass_ior: ior.max(1.0), ..s.config }));
}

/// Frosted glass, 0 (clear) to 1.
#[wasm_bindgen]
pub fn set_glass_rough(roughness: f32) {
    with_state(|s| s.set_ball_materials(Config { glass_rough: roughness.clamp(0.0, 1.0), ..s.config }));
}

/// The rays' hits lit by the lit image where the camera sees them (`rt_screen=`), or always by the
/// lamp and the voxels.
#[wasm_bindgen]
pub fn set_screen_hits(on: bool) {
    with_state(|s| s.apply(Config { screen_hits: on, ..s.config }));
}

/// `lit`, `reflection` (the light they add), `mirror` (what the rays see) or `cost`.
#[wasm_bindgen]
pub fn set_reflection_view(name: &str) {
    with_state(|s| s.apply(Config { reflection_view: reflection_view(name).unwrap_or_default(), ..s.config }));
}

/// Trace the grid of triangles, or (off) the voxel cone alone.
#[wasm_bindgen]
pub fn set_reflection_grid(on: bool) {
    with_state(|s| s.apply(Config { reflection_grid: on, ..s.config }));
}

/// `half` or `quarter`: one pixel of each 2 x 2 or 4 x 4 traced a frame.
#[wasm_bindgen]
pub fn set_reflection_resolution(name: &str) {
    with_state(|s| s.apply(Config { reflection_quarter: name == "quarter", ..s.config }));
}

/// `env=none|gradient|hdri`: the environment past the box's opening, fetching the HDRI the first
/// time.
#[wasm_bindgen]
pub async fn set_env(name: String) {
    let Some(env) = Env::from_name(&name) else { return };
    if env == Env::Hdri {
        if let Some((device, queue)) = with_state(|s| s.hdri.is_none().then(|| (s.renderer.device().clone(), s.renderer.queue().clone()))).flatten() {
            if let Some((hdri, ms)) = load_hdri(&device, &queue).await {
                with_state(|s| {
                    s.hdri = Some(hdri);
                    s.hdri_load_ms = ms;
                });
            }
        }
    }
    with_state(|s| s.apply(Config { env, ..s.config }));
}

/// The environment's stops over its default (the gradient's and the HDRI's alike).
#[wasm_bindgen]
pub fn set_env_ev(ev: f32) {
    with_state(|s| s.set_env_settings(Config { env_ev: ev, ..s.config }));
}

/// The HDRI's turn about the vertical, degrees.
#[wasm_bindgen]
pub fn set_env_rotation(degrees: f32) {
    with_state(|s| s.set_env_settings(Config { env_rotation: degrees, ..s.config }));
}

/// The HDRI background's blur, 0 (sharp) to 1.
#[wasm_bindgen]
pub fn set_env_blur(blur: f32) {
    with_state(|s| s.set_env_settings(Config { env_blur: blur.clamp(0.0, 1.0), ..s.config }));
}

/// Degrees a second the HDRI turns (0: still): prepared again every frame, to time it.
#[wasm_bindgen]
pub fn set_env_spin(degrees_per_second: f32) {
    with_state(|s| s.env_spin = degrees_per_second);
}
