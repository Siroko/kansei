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
//!
//! Drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan. The panel
//! (and `window.kansei`) switches everything at run time.
//!
//! URL parameters (a `preset` first, the others over it):
//! - `preset=off|ssgi|voxel|best|indirect|voxels|phone|dragon|sdf|sdf-dragon|slice|probes|probe-view|probes-dragon` (see `PRESETS`;
//!   `best`, voxel + SSGI at the device's tier, unless the URL names a preset or a `gi`);
//! - `gi=off|low|medium|high|ultra|voxel|voxel+ssgi|probes|probes+ssgi`;
//! - `voxels=low|medium|high`: the volume's resolution (default medium; low on phones, which also
//!   keep it within 24 MiB);
//! - `view=indirect` (only the light GI adds, 2 stops brighter), `view=voxels` (with voxel GI:
//!   the lit voxels themselves), `view=sdf` (a slice of voxel GI's distance field, `slice=`
//!   metres up, default 0.6) or `view=probes` (the probes, lit by their own irradiance);
//! - the distance field (it turns voxel GI's scene volume on whatever the mode): `sdf_ao=0..1`
//!   (its AO on the GI), `sdf_shadows=off|fallback|always` (the voxels' shadows through it, where
//!   no map covers them or always), `shadows=map|sdf` (the direct light's shadows through it, by
//!   the material helper `gi::SDF_WGSL`);
//! - `cam=front|corner|low`;
//! - `dragon=1|full`: the Stanford dragon (CC-BY-NC-4.0, see `www/assets/license.txt`): `1` the
//!   decimated `.glb` (19k triangles), `full` the whole scan (871k triangles, 24 MB);
//! - `animate=1`: the dragon (or, without it, the tall block) turns and slides, re-voxelized each
//!   frame;
//! - `albedo=constant`, `rug=off` (the box without its rug), `ui=0` (no panel);
//! - `stats=1`: triangles, frame interval and the GPU time of each pass (the renderer's profiling).

use wasm_bindgen::prelude::*;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::buffers::{BufferType, ComputeBuffer, Sampler, Texture};
use kansei_core::cameras::Camera;
use kansei_core::controls::CameraControls;
use kansei_core::geometries::BoxGeometry;
use kansei_core::gi::{GiSurface, SceneVoxelGiOptions, SdfProbeOptions, SdfShadows, VoxelGIEffect, VoxelGIOptions, VoxelGiQuality, SDF_WGSL, VOXEL_WRITE_WGSL};
use kansei_core::lights::{Light, SpotLight, SPOT_LIGHTS_WGSL};
use kansei_core::loaders::GLTFLoader;
use kansei_core::materials::{Binding, Material, MaterialOptions, ShaderStages};
use kansei_core::math::{Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::postprocessing::{
    PostProcessingEffect, PostProcessingVolume,
    effects::{exposure_from_ev100, GiQuality, ScreenSpaceGIEffect, ScreenSpaceGIOptions, ToneMapEffect, ToneMapOptions},
};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_wasm::{fetch_bytes, flag, is_phone, now, param, Canvas, Frame};

/// A diffuse surface lit by the spot lights only (no ambient), writing the normal and albedo the
/// global illumination reads (GBuffer targets 2 and 3).
const LIT_WGSL: &str = r#"
struct Surface { base_color: vec4<f32> };
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
};
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    let world = world_matrix * v.position;
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = surface.base_color.rgb;
    var out: GBufferOut;
    out.color = vec4<f32>(lit_radiance(in.world, n, v, base, in.clip.xy), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}
"#;

/// The rug: `LIT_WGSL` with its base colour from a texture, and a voxel entry that gives voxel GI
/// the same texture.
const RUG_WGSL: &str = r#"
@group(0) @binding(0) var<uniform> surface: vec4<f32>;
@group(0) @binding(1) var rug_texture: texture_2d<f32>;
@group(0) @binding(2) var rug_sampler: sampler;
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
struct GBufferOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) albedo: vec4<f32>,
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

@fragment
fn fragment_main(in: VOut) -> GBufferOut {
    let n = normalize(in.normal);
    let view3 = mat3x3<f32>(view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz);
    let camera_pos = -(transpose(view3) * view_matrix[3].xyz);
    let v = normalize(camera_pos - in.world);
    let base = textureSample(rug_texture, rug_sampler, in.uv).rgb;
    var out: GBufferOut;
    out.color = vec4<f32>(lit_radiance(in.world, n, v, base, in.clip.xy), 1.0);
    out.emissive = vec4<f32>(0.0);
    out.normal = vec4<f32>(n * 0.5 + 0.5, 1.0);
    out.albedo = vec4<f32>(base, 1.0);
    return out;
}

// voxel GI's voxelizer: the rug's texture into its voxels
@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, textureSample(rug_texture, rug_sampler, in.uv).rgb, vec3<f32>(0.0));
}
"#;

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
fn rug_material(voxel_entry: bool, sdf: Option<&SdfBinding>) -> Material {
    let (code, bindings) = lit_shader(
        &format!("{VOXEL_WRITE_WGSL}\n{RUG_WGSL}"),
        sdf,
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::texture_2d(1, ShaderStages::FRAGMENT),
            Binding::sampler(2, ShaderStages::FRAGMENT),
        ],
    );
    let mut material = Material::new("Rug", &code, bindings, MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: voxel_entry.then_some("voxel_main"), ..Default::default() });
    bind_sdf(&mut material, sdf);
    material.set_uniform_bindable(0, "Rug", &[1.0f32, 1.0, 1.0, 1.0]);
    material.set_bindable(1, rug_texture());
    material.set_bindable(2, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear));
    material
}

/// The spot lights' light on a surface (`lit_radiance`), shadowed by the shadow atlas.
const MAP_SHADOWS_WGSL: &str = r#"
fn lit_radiance(world: vec3<f32>, n: vec3<f32>, v: vec3<f32>, base: vec3<f32>, pixel: vec2<f32>) -> vec3<f32> {
    return kansei_spot_lights_radiance(world, n, v, base, 1.0, 0.0, pixel);
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

/// `body` (which calls `lit_radiance`) after the light chunk for `sdf` (shadows through the field)
/// or the shadow maps, with its bindings: `bindings` plus the field's.
fn lit_shader(body: &str, sdf: Option<&SdfBinding>, mut bindings: Vec<Binding>) -> (String, Vec<Binding>) {
    match sdf {
        None => (format!("{SPOT_LIGHTS_WGSL}\n{MAP_SHADOWS_WGSL}\n{body}"), bindings),
        Some(_) => {
            bindings.extend([Binding::uniform(10, ShaderStages::FRAGMENT), Binding::texture_3d(11, ShaderStages::FRAGMENT), Binding::sampler(12, ShaderStages::FRAGMENT)]);
            (format!("{SPOT_LIGHTS_WGSL}\n{SDF_WGSL}\n{SDF_SHADOWS_WGSL}\n{body}"), bindings)
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

fn lit_material(label: &str, base_color: [f32; 3], sdf: Option<&SdfBinding>) -> Material {
    let (code, bindings) = lit_shader(LIT_WGSL, sdf, vec![Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT)]);
    let mut material = Material::new(label, &code, bindings, MaterialOptions { mrt_output_count: Some(4), ..Default::default() });
    material.set_uniform_bindable(0, label, &[base_color[0], base_color[1], base_color[2], 1.0f32]);
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
            // a raw '+' in the query, or one decoded to a space or escaped
            "voxel+ssgi" | "voxel ssgi" | "voxel%2Bssgi" | "voxel%2bssgi" => Gi::VoxelAndScreen,
            "probes" => Gi::Probes,
            "probes+ssgi" | "probes ssgi" | "probes%2Bssgi" | "probes%2bssgi" => Gi::ProbesAndScreen,
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
        }
    }

    fn voxels(self) -> bool {
        matches!(self, Gi::Voxel | Gi::VoxelAndScreen | Gi::Probes | Gi::ProbesAndScreen)
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

fn tier_name(q: VoxelGiQuality) -> &'static str {
    match q {
        VoxelGiQuality::Low => "low",
        VoxelGiQuality::Medium => "medium",
        VoxelGiQuality::High => "high",
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
    /// The direct light's shadows through the distance field (the material helper) instead of
    /// the shadow atlas.
    direct_sdf: bool,
    /// The SDF slice's height, metres.
    slice: f32,
}

impl Config {
    /// Whether anything reads the probes.
    fn needs_probes(&self) -> bool {
        self.gi.probes() || self.view == View::Probes
    }

    /// Whether anything reads the distance field.
    fn needs_sdf(&self) -> bool {
        self.sdf_ao > 0.0 || self.sdf_shadows != SdfShadows::Off || self.view == View::Sdf || self.direct_sdf || self.needs_probes()
    }

    /// Whether the scene's voxel GI must run: for the voxel modes, or for the distance field.
    fn needs_voxels(&self) -> bool {
        self.gi.voxels() || self.needs_sdf()
    }
}

fn sdf_shadows_name(s: SdfShadows) -> &'static str {
    match s {
        SdfShadows::Off => "off",
        SdfShadows::Fallback => "fallback",
        SdfShadows::Always => "always",
    }
}

fn sdf_shadows_from_name(name: &str) -> Option<SdfShadows> {
    Some(match name {
        "off" => SdfShadows::Off,
        "fallback" => SdfShadows::Fallback,
        "always" => SdfShadows::Always,
        _ => return None,
    })
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
    direct_sdf: bool,
}

const fn preset(name: &'static str, label: &'static str, gi: &'static str, voxels: Option<VoxelGiQuality>, view: View, dragon: Dragon) -> Preset {
    Preset { name, label, gi, voxels, view, dragon, sdf_ao: 0.0, sdf_shadows: SdfShadows::Off, direct_sdf: false }
}

const PRESETS: [Preset; 14] = [
    preset("off", "Off (direct light)", "off", None, View::Lit, Dragon::Off),
    preset("ssgi", "SSGI", "high", None, View::Lit, Dragon::Off),
    preset("voxel", "Voxel", "voxel", None, View::Lit, Dragon::Off),
    preset("best", "Voxel + SSGI (best)", "voxel+ssgi", None, View::Lit, Dragon::Off),
    preset("indirect", "Indirect only", "voxel+ssgi", None, View::Indirect, Dragon::Off),
    preset("voxels", "Voxels (debug)", "voxel", None, View::Voxels, Dragon::Off),
    preset("phone", "Phone (low)", "voxel+ssgi", Some(VoxelGiQuality::Low), View::Lit, Dragon::Off),
    preset("dragon", "Dragon, 871k tris (voxel + SSGI)", "voxel+ssgi", None, View::Lit, Dragon::Full),
    // the distance field: its AO on the GI and soft shadows for the voxels and the direct light
    Preset { sdf_ao: 0.8, sdf_shadows: SdfShadows::Always, direct_sdf: true, ..preset("sdf", "SDF: AO + soft shadows", "voxel+ssgi", None, View::Lit, Dragon::Off) },
    Preset { sdf_ao: 0.8, sdf_shadows: SdfShadows::Always, direct_sdf: true, ..preset("sdf-dragon", "SDF + dragon, 871k tris", "voxel+ssgi", None, View::Lit, Dragon::Full) },
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
const CAMERAS: [(&str, [f32; 3], f32, f32, f32); 3] = [
    // the box's opening, framed as the Cornell box photographs are
    ("front", [0.0, 2.0, -2.0], 8.3, 0.0, 0.0),
    // inside, from the upper front-left corner toward the short block and the green wall
    ("corner", [1.0, 0.8, -3.0], 4.98, -0.777, 0.634),
    // half a metre above the floor in front of the opening, looking up into the box
    ("low", [0.0, 1.4, -2.5], 7.5, 0.0, -0.12),
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
    /// The generation the lit materials read the field of (None: they use the shadow maps).
    materials_sdf: Option<u32>,
    /// The lit renderables: (scene index, label, albedo); the rug is apart.
    lit: Vec<(usize, &'static str, [f32; 3])>,
    tall: usize,
    rug: usize,
    /// The dragons loaded so far (`Dragon::slot`).
    dragons: [Option<usize>; 2],
    /// (position, yaw) the animated renderables rest at.
    tall_rest: (Vec3, f32),
    dragon_rest: (Vec3, f32),
    time: f32,
    stats: Option<Stats>,
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
const DEFAULT_PRESET: &str = "best";

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
        direct_sdf: p.direct_sdf,
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
        direct_sdf: false,
        slice: 0.6,
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
    if let Some(shadows) = param("sdf_shadows").as_deref().and_then(sdf_shadows_from_name) {
        c.sdf_shadows = shadows;
    }
    if let Some(direct) = param("shadows") {
        c.direct_sdf = direct == "sdf";
    }
    if let Some(slice) = param("slice").and_then(|v| v.parse::<f32>().ok()) {
        c.slice = slice;
    }
    c.rug = param("rug").as_deref() != Some("off");
    c.textured = param("albedo").as_deref() != Some("constant");
    c
}

/// The post-processing chain for `config`: its GI effect (none when off) and the tone mapping.
fn build_effects(renderer: &Renderer, config: &Config, phone: bool) -> Vec<Box<dyn PostProcessingEffect>> {
    let indirect = config.view == View::Indirect;
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = Vec::new();
    let screen = ScreenSpaceGIOptions { radius_m: 4.0, ..Default::default() };
    match config.gi {
        Gi::Off if config.view != View::Sdf => {}
        Gi::Screen(quality) if config.view != View::Sdf => {
            let mut effect = ScreenSpaceGIEffect::new(ScreenSpaceGIOptions { quality, ..screen });
            effect.show_indirect = indirect;
            effects.push(Box::new(effect));
        }
        _ => {
            // (the slice view shows through voxel GI's effect whatever the mode)
            let scene_gi = renderer.voxel_gi().expect("voxel GI is enabled for the voxel modes and the field");
            let near_quality = if phone || scene_gi.quality() == VoxelGiQuality::Low { GiQuality::Low } else { GiQuality::High };
            let near_field = config.gi.near_field().then_some(ScreenSpaceGIOptions { quality: near_quality, ..screen });
            let mut effect = VoxelGIEffect::new(scene_gi.volume(), VoxelGIOptions { quality: scene_gi.quality(), near_field, ..Default::default() });
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
        self.volume.effects = build_effects(&self.renderer, &config, self.phone);
        // the lit materials: shadowed through the field (bound to this one) or by the maps
        let wanted = config.direct_sdf.then_some(self.gi_generation);
        if wanted != self.materials_sdf || config.textured != previous.textured {
            let sdf = config.direct_sdf.then(|| self.renderer.voxel_gi().and_then(SdfBinding::of)).flatten();
            for &(index, label, albedo) in &self.lit {
                if let Some(r) = self.scene.get_renderable_mut(index) {
                    r.material = lit_material(label, albedo, sdf.as_ref());
                    r.material_dirty = true;
                }
            }
            if let Some(r) = self.scene.get_renderable_mut(self.rug) {
                r.material = rug_material(config.textured, sdf.as_ref());
                r.material_dirty = true;
            }
            self.materials_sdf = wanted;
        }

        if let Some(r) = self.scene.get_renderable_mut(self.rug) {
            r.visible = config.rug;
        }
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

    fn info(&self) -> String {
        let gi = self.renderer.voxel_gi();
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        // (+ 0.0: an empty sum is -0)
        let sum = |prefix: &str| self.stats.as_ref().map_or(0.0, |s| s.passes.iter().filter(|p| p.0.starts_with(prefix)).map(|p| p.1).sum::<f64>()) + 0.0;
        format!(
            "{{\"gi\":\"{}\",\"view\":\"{}\",\"voxels\":\"{}\",\"voxel_tier\":{},\"dims\":{},\"mib\":{:.1},\"dragon\":\"{}\",\"animate\":{},\"rug\":{},\"textured\":{},\"sdf_ao\":{},\"sdf_shadows\":\"{}\",\"shadows\":\"{}\",\"slice\":{},\"sdf\":{},\"sdf_ms\":{:.3},\"probes\":{},\"probe_dims\":{},\"probes_ms\":{:.3},\"triangles\":{},\"stats\":{},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"voxelize_ms\":{:.3},\"inject_ms\":{:.3},\"mips_ms\":{:.3},\"screen_ms\":{:.3},\"ssgi_ms\":{:.3},\"passes\":[{}]}}",
            self.config.gi.name(),
            self.config.view.name(),
            tier_name(self.config.voxels),
            gi.map_or("null".into(), |g| format!("\"{}\"", tier_name(g.quality()))),
            gi.map_or("null".into(), |g| format!("{:?}", g.volume().dims())),
            gi.map_or(0.0, |g| g.memory_bytes() as f64 / (1 << 20) as f64),
            self.config.dragon.name(),
            self.config.animate,
            self.config.rug,
            self.config.textured,
            self.config.sdf_ao,
            sdf_shadows_name(self.config.sdf_shadows),
            if self.config.direct_sdf { "sdf" } else { "map" },
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

/// Add the dragon to the scene (hidden until the config shows it), returning its index.
const DRAGON_ALBEDO: [f32; 3] = [0.75, 0.62, 0.42];

fn add_dragon(scene: &mut Scene, geometry: kansei_core::geometries::Geometry, rest: (Vec3, f32)) -> usize {
    let albedo = DRAGON_ALBEDO;
    let mut dragon = Renderable::new(geometry, lit_material("Dragon", albedo, None)).with_gi(GiSurface::new(albedo));
    dragon.object.position = rest.0;
    dragon.object.rotation.y = rest.1;
    dragon.visible = false;
    scene.add(SceneNode::Renderable(dragon))
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_spot_shadows(2048, 1);
    let phone = is_phone();
    let config = config_from_url(phone);

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
        let mut slab = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), lit_material(label, color, None)).with_gi(GiSurface::new(color));
        slab.object.set_position(position[0], position[1], position[2]);
        lit.push((scene.add(SceneNode::Renderable(slab)), label, color));
    }
    // a tall block near the red wall and a short one near the green wall
    let tall_rest = (Vec3::new(-0.75, 1.2, -2.6), 0.33);
    let mut tall = Renderable::new(BoxGeometry::new(1.2, 2.4, 1.2), lit_material("Tall", white, None)).with_gi(GiSurface::new(white));
    tall.object.position = tall_rest.0;
    tall.object.rotation.y = tall_rest.1;
    let tall = scene.add(SceneNode::Renderable(tall));
    lit.push((tall, "Tall", white));
    let mut short = Renderable::new(BoxGeometry::new(1.2, 1.2, 1.2), lit_material("Short", white, None)).with_gi(GiSurface::new(white));
    short.object.set_position(0.8, 0.6, -1.5);
    short.object.rotation.y = -0.3;
    lit.push((scene.add(SceneNode::Renderable(short)), "Short", white));
    // a textured rug on the floor in front of the blocks, orange before the tall one and blue
    // before the short one, whose sides facing the camera only bounces light; its constant
    // surface (its mean colour) is used where the material's voxel entry is not
    let mean: [f32; 3] = std::array::from_fn(|c| (RUG_LEFT[c] + RUG_RIGHT[c]) * 0.5);
    let mut rug = Renderable::new(BoxGeometry::new(2.4, 0.02, 0.8), rug_material(config.textured, None)).with_gi(GiSurface::new(mean));
    rug.object.set_position(0.0, 0.01, -0.55);
    let rug = scene.add(SceneNode::Renderable(rug));
    // the dragon, in the free corner at the front left, when asked for
    let dragon_rest = (Vec3::new(-1.15, 0.0, -1.25), 0.5);
    let mut dragons = [None, None];
    if let Some(slot) = config.dragon.slot() {
        dragons[slot] = load_dragon(config.dragon).await.map(|g| add_dragon(&mut scene, g, dragon_rest));
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
        materials_sdf: None,
        lit,
        tall,
        rug,
        dragons,
        tall_rest,
        dragon_rest,
        time: 0.0,
        stats,
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
    let Some(config) = with_state(|s| with_preset(s.config, &name, s.phone)) else { return };
    apply_with_dragon(config).await;
}

/// `gi=` at run time.
#[wasm_bindgen]
pub fn set_gi(name: &str) {
    if let Some(gi) = Gi::from_name(name) {
        with_state(|s| s.apply(Config { gi, ..s.config }));
    }
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
                        let index = add_dragon(&mut s.scene, geometry, s.dragon_rest);
                        s.dragons[slot] = Some(index);
                        s.lit.push((index, "Dragon", DRAGON_ALBEDO));
                        // its material follows the others' at the next apply
                        s.materials_sdf = Some(u32::MAX);
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

/// Camera preset `front|corner|low` (`CAMERAS`).
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
    if let Some(sdf_shadows) = sdf_shadows_from_name(name) {
        with_state(|s| s.apply(Config { sdf_shadows, ..s.config }));
    }
}

/// The direct light's shadows: `map` (the shadow atlas) or `sdf` (the distance field, through the
/// material helper `gi::SDF_WGSL`).
#[wasm_bindgen]
pub fn set_shadows(name: &str) {
    with_state(|s| s.apply(Config { direct_sdf: name == "sdf", ..s.config }));
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
