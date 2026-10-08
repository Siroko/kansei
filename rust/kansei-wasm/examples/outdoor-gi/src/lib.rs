//! Outdoor GI: a forest valley under a low sun, lit by the sun (cascaded shadows) and an
//! atmosphere's sky, with voxel GI through a clipmap round the camera
//! (`Renderer::enable_voxel_clipmap`): the sky's light reaches the forest floor only where the
//! canopy lets it through, and the sunlit ground and trees light what is in their shade. The GI
//! modes compare what lights the shade: the materials' own sky light (`gi=off`), dimmed by the
//! top-down sky occlusion the film uses (`skyocc`) or by the clipmap's sky visibility
//! (`visibility`, read in the material), voxel GI on screen from cones per pixel (`cones`) or
//! the clipmap's probes (`probes`), or the hybrid (`rt`, `RtDiffuseGiEffect`): rays through a grid
//! of the scene's triangles near the camera, the clipmap past them.
//!
//! The terrain is tiles of a height field; the spruces (17 576) are instanced, culled on the GPU
//! per view with dithered crossfades between LODs, as a film's forest is: needle-spray cards with
//! cluster LOD near the camera, cone meshes farther. The clipmap voxelizes them through its own
//! cull view. With `reflect=1` the road is wet and reflects the forest, traced through a grid of
//! the scene's triangles (`Renderer::enable_rt_grid`, `RtReflectionsEffect`). See README.md for
//! the URL parameters.

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use kansei_core::atmosphere::{direction_from_elevation_bearing, SkyAtmosphere, SkyAtmosphereOptions, SKY_LIGHTING_WGSL};
use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer, InstanceAttribute, VertexFormat};
use kansei_core::cameras::{Camera, MOTION_VECTORS_WGSL};
use kansei_core::controls::CameraControls;
use kansei_core::culling::{InstanceCulling, LOD_FADE_WGSL};
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform};
use kansei_core::geometries::{BoxGeometry, CylinderGeometry, Geometry, HeightfieldGeometry, IcosphereGeometry, InstancedGeometry, SpruceGeometry};
use kansei_core::gi::{ClipmapProbeOptions, ClipmapProbes, ConeShadows, GiSurface, SceneVoxelClipmapOptions, VoxelGIEffect, VoxelGIOptions, CLIPMAP_PROBES_WGSL, VOXEL_WRITE_WGSL};
use kansei_core::lights::{DirectionalLight, Light};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::math::{hash01, Vec3, Vec4};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::froxels::FroxelGridOptions;
use kansei_core::postprocessing::effects::{exposure_from_ev100_lens, AtmosphereEffect, VolumetricFogEffect, VolumetricFogOptions, TemporalAAEffect, TemporalAAOptions, ToneMapEffect, ToneMapOptions, ToneMapper, LENS_ATTENUATION_UE4};
use kansei_core::postprocessing::{PostProcessingEffect, PostProcessingVolume};
use kansei_core::renderers::{Renderer, RendererConfig};
use kansei_core::shadows::{CascadedShadowOptions, SkyOcclusion, SkyOcclusionOptions, CASCADED_SHADOWS_WGSL, SKY_OCCLUSION_WGSL};
use kansei_core::buffers::{Sampler, Texture};
use kansei_core::rt::{
    RtDiffuseGiEffect, RtDiffuseGiOptions, RtGiDenoise, RtGiHitLighting, RtGiKernel, RtGiMode, RtGiResolution, RtGiShadows, RtGiView, RtGridOptions, RtPlacement, RtReflectionsEffect,
    RtReflectionsOptions, RtReflectionsView, RtSurface, RtTraceResolution, SceneRtGridOptions,
};
use kansei_wasm::{flag, now, param, param_or, Canvas, Frame};

/// The road's centre line across the valley: x at z (metres). Kept in step with ROAD_WGSL.
fn road_x(z: f32) -> f32 {
    22.0 * (z / 85.0).sin() + 9.0 * (z / 37.0 + 1.3).sin()
}

const ROAD_WGSL: &str = r#"
fn road_x(z: f32) -> f32 {
    return 22.0 * sin(z / 85.0) + 9.0 * sin(z / 37.0 + 1.3);
}
"#;

/// The clearing beside the road: its centre and radius.
const CLEARING: (f32, f32, f32) = (45.0, 60.0, 26.0);

/// The valley: hills rising away from the road, bumps everywhere but on the road's bed.
fn height(x: f32, z: f32) -> f32 {
    let u = x - road_x(z);
    let valley = 32.0 * (1.0 - (-(u / 210.0).powi(2)).exp());
    let bumps = 7.0 * (x / 97.0 + 0.4).sin() * (z / 131.0).cos() + 4.0 * ((x - z) / 61.0).sin() + 1.6 * (x / 23.0).sin() * (z / 29.0).sin();
    let bed = smoothstep(4.0, 22.0, u.abs());
    valley + bumps * (0.25 + 0.75 * bed)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// The layer the trees are on besides the default one: the canopy the sky occlusion sees.
const TREE_LAYER: u32 = 1 << 1;

/// Half the side of the terrain (metres) and its tiles per side.
const WORLD: f32 = 640.0;
const TILES: u32 = 8;

/// The sky light every surface receives, by the page's GI mode (`Gi`): the sky's SH whole (its
/// own ambient: off, and the voxel GI's modes, which replace it on screen), dimmed by the top-down
/// sky occlusion the film uses, or by the sky visibility the clipmap's probes measure. Group 0
/// bindings 1 (the sky) and 2-7. Prefixed with SKY_LIGHTING_WGSL, SKY_OCCLUSION_WGSL and
/// CLIPMAP_PROBES_WGSL.
const AMBIENT_WGSL: &str = r#"
struct Ambient { mode: u32, _pad0: u32, _pad1: u32, _pad2: u32 };
@group(0) @binding(1) var<uniform> sky: SkyLighting;
@group(0) @binding(2) var<uniform> ambient: Ambient;
@group(0) @binding(3) var sky_volume: texture_3d<f32>;
@group(0) @binding(4) var sky_sampler: sampler;
@group(0) @binding(5) var<uniform> sky_occlusion: SkyOcclusionParams;
@group(0) @binding(6) var<uniform> kansei_clip_probe_grid: ClipProbeGrid;
@group(0) @binding(7) var<storage, read> kansei_clip_probes: array<vec4<f32>>;

fn sky_light(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let e = skyIrradiance(sky, n);
    switch (ambient.mode) {
        case 1u: { return e * skyVisibility(sky_volume, sky_sampler, sky_occlusion, world); }
        case 2u: { return e * kansei_clipmap_sky_visibility(world, n); }
        default: { return e; }
    }
}
"#;

/// Lambertian surfaces lit by the sun (its cascades) and the sky (AMBIENT_WGSL's `sky_light`),
/// writing the GBuffer's albedo and normal for the GI. ALBEDO (string replaced) is a function
/// `surface_albedo(world, n) -> vec3f` and `surface_specular(world, n) -> vec2f`, a wet surface's
/// F0 and roughness (`surface.wet`; 0 when dry) for the ray-traced reflections; the voxel entry
/// writes the same albedo into the GI's voxels. Prefixed with AMBIENT_WGSL and its chunks, CASCADED_SHADOWS_WGSL, GBUFFER_OUT_WGSL,
/// VOXEL_WRITE_WGSL and ROAD_WGSL.
const GROUND_WGSL: &str = r#"
struct Surface { albedo: vec4<f32>, road: vec4<f32>, rock: vec4<f32>, wet: vec4<f32> };
@group(0) @binding(0) var<uniform> surface: Surface;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn { @location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32> };
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) world: vec3<f32>, @location(1) normal: vec3<f32> };

@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    let world = world_matrix * v.position;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(v.normal, 0.0)).xyz;
    return out;
}

ALBEDO

@fragment
fn fragment_main(in: VOut) -> KanseiGBufferOut {
    let n = normalize(in.normal);
    let albedo = surface_albedo(in.world, n);
    let sun = kansei_cascades.lightColor * max(dot(n, -kansei_cascades.lightDirection), 0.0) * kansei_sun_shadow(in.world, n, in.clip.xy);
    // a wet surface's reflectance and roughness, for the ray-traced reflections (F0 0: none)
    let wet = surface_specular(in.world, n);
    return kansei_gbuffer_out_specular(albedo / 3.14159265 * (sun + sky_light(in.world, n)), vec3<f32>(0.0), n, albedo, wet.x, wet.y);
}

@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    kansei_voxel_write(in.clip, front, surface_albedo(in.world, normalize(in.normal)), vec3<f32>(0.0));
}
"#;

/// The terrain's albedo: grass, the road's dirt within 3.5 m of its centre, rock on steep slopes.
const TERRAIN_ALBEDO_WGSL: &str = r#"
fn surface_albedo(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    let road = 1.0 - smoothstep(2.8, 4.0, abs(world.x - road_x(world.z)));
    let rock = smoothstep(0.75, 0.6, n.y);
    // a little variation in the grass
    let variation = 0.85 + 0.3 * fract(sin(dot(floor(world.xz / 3.0), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    return mix(mix(surface.albedo.rgb * variation, surface.road.rgb, road), surface.rock.rgb, rock);
}

// wet: the road's bed (`surface.wet.x`, F0) where it is level, the rest by `surface.wet.y`
fn surface_specular(world: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    let road = 1.0 - smoothstep(2.8, 4.0, abs(world.x - road_x(world.z)));
    let level = smoothstep(0.85, 0.95, n.y);
    return vec2<f32>(mix(surface.wet.y, surface.wet.x * level, road), surface.wet.z);
}
"#;

/// A constant albedo (rocks, the cabin).
const CONSTANT_ALBEDO_WGSL: &str = r#"
fn surface_albedo(world: vec3<f32>, n: vec3<f32>) -> vec3<f32> {
    return surface.albedo.rgb;
}

fn surface_specular(world: vec3<f32>, n: vec3<f32>) -> vec2<f32> {
    return surface.wet.yz;
}
"#;

/// Spruces placed by instance records of 32 bytes (base xyz, height; yaw, tint, -, -) plus the
/// crossfade's fade (36 bytes as culled), lit as GROUND_WGSL's surfaces; the fade drops pixels in
/// every pass but the voxels'. Two kinds (`bark.w`): `SpruceGeometry`'s cones, bark within 0.045
/// of the axis and needles outside it; or `card_spruce`'s cards, bark where u is past 1.5 and
/// needle sprays cut out of the cards by the needle texture's alpha, in every pass (the voxels'
/// too: their area is the sprays'). Prefixed with AMBIENT_WGSL and its chunks,
/// CASCADED_SHADOWS_WGSL, GBUFFER_OUT_WGSL, VOXEL_WRITE_WGSL, MOTION_VECTORS_WGSL and
/// LOD_FADE_WGSL.
const TREE_WGSL: &str = r#"
struct Tree { bark: vec4<f32>, needles: vec4<f32> };
@group(0) @binding(0) var<uniform> tree: Tree;
@group(0) @binding(8) var needle_texture: texture_2d<f32>;
@group(0) @binding(9) var needle_sampler: sampler;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) place: vec4<f32>,
    @location(4) extra: vec4<f32>,
    @location(5) fade: f32,
};
struct VOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) world: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) radius: f32,
    @location(3) tint: f32,
    @location(4) fade: f32,
    @location(5) up: f32,
    @location(6) uv: vec2<f32>,
};

@vertex
fn vertex_main(v: VIn) -> VOut {
    // a spruce `place.w` tall, a little wider or narrower by its tint, turned by its yaw
    let width = place_width(v.extra.y);
    let s = vec3<f32>(width, 1.0, width) * v.place.w;
    let c = cos(v.extra.x);
    let sn = sin(v.extra.x);
    let q = v.position.xyz * s;
    let local = vec3<f32>(c * q.x + sn * q.z, q.y, -sn * q.x + c * q.z) + v.place.xyz;
    let m = v.normal / vec3<f32>(width, 1.0, width);
    let n = vec3<f32>(c * m.x + sn * m.z, m.y, -sn * m.x + c * m.z);
    let world = world_matrix * vec4<f32>(local, 1.0);
    var out: VOut;
    out.clip = projection_matrix * view_matrix * world;
    out.world = world.xyz;
    out.normal = (normal_matrix * vec4<f32>(n, 0.0)).xyz;
    out.radius = length(v.position.xz);
    out.tint = v.extra.y;
    out.fade = v.fade;
    out.up = v.position.y;
    out.uv = v.uv;
    return out;
}

fn place_width(tint: f32) -> f32 {
    return 0.85 + 0.3 * tint;
}

fn is_bark(in: VOut) -> bool {
    return select((in.radius < 0.045), (in.uv.x > 1.5), (tree.bark.w > 0.5));
}

fn tree_albedo(in: VOut) -> vec3<f32> {
    if (is_bark(in)) { return tree.bark.rgb; }
    // lighter toward the tips of the whorls and the top
    return tree.needles.rgb * (0.8 + 0.4 * in.tint) * (0.8 + 0.35 * in.up);
}

// Whether the needle sprays cover this point of a card (always on bark and cones). `alpha` is the
// needle texture's there, sampled in uniform control flow.
fn covered(in: VOut, alpha: f32) -> bool {
    return tree.bark.w < 0.5 || in.uv.x > 1.5 || alpha >= 0.5;
}

@fragment
fn fragment_main(in: VOut, @builtin(front_facing) front: bool) -> KanseiGBufferOut {
    let alpha = textureSample(needle_texture, needle_sampler, in.uv).a;
    if (kansei_lod_fade_discard(in.fade, in.clip.xy, kansei_camera_temporal.frame) || !covered(in, alpha)) { discard; }
    var n = normalize(in.normal);
    if (!front) { n = -n; }
    let albedo = tree_albedo(in);
    // needles let a little light through: some of the sun on the far side
    let ndl = dot(n, -kansei_cascades.lightDirection);
    let wrap = select(max(ndl, 0.0), max(ndl, 0.0) * 0.8 + 0.2 * abs(ndl), in.radius >= 0.045);
    let sun = kansei_cascades.lightColor * wrap * kansei_sun_shadow(in.world, n, in.clip.xy);
    return kansei_gbuffer_out(albedo / 3.14159265 * (sun + sky_light(in.world, n)), vec3<f32>(0.0), n, albedo);
}

@fragment
fn shadow_fragment(in: VOut) {
    let alpha = textureSample(needle_texture, needle_sampler, in.uv).a;
    if (kansei_lod_fade_discard(in.fade, in.clip.xy, kansei_camera_temporal.frame) || !covered(in, alpha)) { discard; }
}

@fragment
fn voxel_main(in: VOut, @builtin(front_facing) front: bool) {
    // the sprays' area alone: the cut-out texels write nothing
    let alpha = textureSample(needle_texture, needle_sampler, in.uv).a;
    kansei_voxel_write_coverage(in.clip, front, tree_albedo(in), vec3<f32>(0.0), select(0.0, 1.0, covered(in, alpha)));
}
"#;

const GRASS: [f32; 3] = [0.1, 0.13, 0.055];
const ROAD: [f32; 3] = [0.22, 0.19, 0.15];
const ROCK: [f32; 3] = [0.24, 0.23, 0.21];
const BARK: [f32; 3] = [0.11, 0.085, 0.065];
const NEEDLES: [f32; 3] = [0.045, 0.075, 0.035];

/// A spruce LOD's mesh: needle-spray cards drawn with cluster LOD (`card_spruce`), or
/// `SpruceGeometry`'s cones (segments, rings, cones).
#[derive(Clone, Copy)]
enum TreeMesh {
    Cards,
    Cones(u32, u32, u32),
}

/// Instanced spruces: the LODs' meshes, their bands on screen and in voxel GI's views (from the
/// camera, metres), and the crossfade. Near the camera the cards, as the film's spruces, which the
/// GI voxelizes by their cut within 20 m (its finest levels); then the cones, the coarsest at
/// every distance past 60 m. With `cards=0` the finest cones stand in for the cards (and stay out
/// of the voxels: their detail is finer than them).
const LODS: [(TreeMesh, (f32, f32), (f32, f32)); 3] = [
    (TreeMesh::Cards, (0.0, 45.0), (0.0, 20.0)),
    (TreeMesh::Cones(8, 2, 6), (45.0, 160.0), (20.0, 60.0)),
    (TreeMesh::Cones(5, 1, 4), (160.0, f32::INFINITY), (60.0, f32::INFINITY)),
];
const CROSSFADE: f32 = 8.0;

/// How much wider than its height the tree material may make an instance (`place_width`), for
/// the cards' cluster cull.
const TREE_STRETCH: f32 = 1.15;

/// Where a spruce's record puts a point of its mesh, as TREE_WGSL's `vertex_main` does (a little
/// wider or narrower by its tint, which `InstanceTransform` can't say), for the ray tracing grid.
const TREE_PLACEMENT_WGSL: &str = r#"
fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {
    let place = kansei_rt_record_vec4(record, 0u);
    let extra = kansei_rt_record_vec4(record, 4u);
    let width = 0.85 + 0.3 * extra.y;
    let q = p * vec3f(width, 1.0, width) * place.w;
    let c = cos(extra.x);
    let s = sin(extra.x);
    return vec3f(c * q.x + s * q.z, q.y, -s * q.x + c * q.z) + place.xyz;
}
"#;

/// Where a spruce card's surface is, for the ray-traced reflections' alpha test (TREE_WGSL's
/// `covered`): bark on the trunk (u past 1.5), the needle sprays where the needles' alpha is.
const CARD_COVERED_WGSL: &str = r#"
fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {
    if (uv.x > 1.5) { return true; }
    return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5;
}
"#;

/// How much of the light the crowns' voxels stop for their area: the meshes are closed cones
/// standing for needles light passes between.
const CROWN_OPACITY: f32 = 0.35;

/// What every material's sky light reads (AMBIENT_WGSL): the sky, the page's GI mode, the sky
/// occlusion and the clipmap's probes.
struct AmbientSources<'a> {
    sky: &'a SkyAtmosphere,
    mode: &'a wgpu::Buffer,
    occlusion: &'a SkyOcclusion,
    probes: &'a ClipmapProbes,
}

impl AmbientSources<'_> {
    /// Bind them at group 0 bindings 1-7 of `m`.
    fn bind(&self, m: &mut Material) {
        m.set_bindable(1, ComputeBuffer::from_external("SkyLighting", self.sky.bindings().sky_lighting.clone(), BufferType::Uniform));
        m.set_bindable(2, ComputeBuffer::from_external("AmbientMode", self.mode.clone(), BufferType::Uniform));
        m.set_bindable(3, Texture::from_view("SkyOcclusion", self.occlusion.volume_texture().clone(), self.occlusion.volume.clone()));
        m.set_bindable(4, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
        m.set_bindable(5, ComputeBuffer::from_external("SkyOcclusionParams", self.occlusion.params.clone(), BufferType::Uniform));
        m.set_bindable(6, ComputeBuffer::from_external("ClipProbeGrid", self.probes.grid_buffer().clone(), BufferType::Uniform));
        m.set_bindable(7, ComputeBuffer::from_external("ClipProbes", self.probes.probe_buffer().clone(), BufferType::Storage));
    }

    /// The bindings 0 (the material's own uniform) to 7.
    fn bindings() -> Vec<Binding> {
        let f = ShaderStages::FRAGMENT;
        vec![
            Binding::uniform(0, f),
            Binding::uniform(1, f),
            Binding::uniform(2, f),
            Binding::texture_3d(3, f),
            Binding::sampler(4, f),
            Binding::uniform(5, f),
            Binding::uniform(6, f),
            Binding::storage(7, f, true),
        ]
    }

    /// The WGSL every material is prefixed with.
    fn wgsl() -> String {
        format!("{SKY_LIGHTING_WGSL}\n{SKY_OCCLUSION_WGSL}\n{CLIPMAP_PROBES_WGSL}\n{AMBIENT_WGSL}")
    }
}

fn ground_material(label: &str, albedo_wgsl: &str, albedo: [f32; 3], ambient: &AmbientSources, wet: [f32; 4]) -> Material {
    let shader = GROUND_WGSL.replace("ALBEDO", albedo_wgsl);
    let code = format!("{}\n{CASCADED_SHADOWS_WGSL}\n{GBUFFER_OUT_WGSL}\n{VOXEL_WRITE_WGSL}\n{ROAD_WGSL}\n{shader}", AmbientSources::wgsl());
    let options = MaterialOptions { mrt_output_count: Some(4), voxel_fragment_entry: Some("voxel_main"), ..Default::default() };
    let mut m = Material::new(label, &code, AmbientSources::bindings(), options);
    let mut u = [0.0f32; 16];
    for (k, c) in [albedo, ROAD, ROCK].iter().enumerate() {
        u[4 * k..4 * k + 3].copy_from_slice(c);
    }
    u[12..].copy_from_slice(&wet);
    m.set_uniform_bindable(0, label, &u);
    ambient.bind(&mut m);
    m
}

/// A spruce's material: `SpruceGeometry`'s cones, or `card_spruce`'s cards (`cards`), the needle
/// sprays cut out of them by `needle_texture`.
fn tree_material(label: &str, ambient: &AmbientSources, cards: bool) -> Material {
    let code = format!("{}\n{CASCADED_SHADOWS_WGSL}\n{GBUFFER_OUT_WGSL}\n{VOXEL_WRITE_WGSL}\n{MOTION_VECTORS_WGSL}\n{LOD_FADE_WGSL}\n{TREE_WGSL}", AmbientSources::wgsl());
    let options = MaterialOptions {
        mrt_output_count: Some(4),
        cull_mode: CullMode::None,
        shadow_fragment_entry: Some("shadow_fragment"),
        voxel_fragment_entry: Some("voxel_main"),
        ..Default::default()
    };
    let mut bindings = AmbientSources::bindings();
    bindings.extend([Binding::texture_2d(8, ShaderStages::FRAGMENT), Binding::sampler(9, ShaderStages::FRAGMENT)]);
    let mut m = Material::new(label, &code, bindings, options);
    m.set_uniform_bindable(0, label, &[BARK[0], BARK[1], BARK[2], cards as u32 as f32, NEEDLES[0], NEEDLES[1], NEEDLES[2], 1.0]);
    ambient.bind(&mut m);
    m.set_bindable(8, needle_texture());
    m.set_bindable(9, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
    m
}

/// A needle spray, 32 x 64 texels (u across the spray, v from its base to its tip): a twig down
/// the middle and needles slanting off it toward the tip, the spray tapering at both ends; alpha
/// 1 on them, 0 between. Its colour is white (the material tints it).
fn needle_texture() -> Texture {
    let (w, h) = (32u32, 64u32);
    let mut texels = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        for x in 0..w {
            let (u, v) = ((x as f32 + 0.5) / w as f32, (y as f32 + 0.5) / h as f32);
            let across = (u - 0.5).abs();
            let half_width = 0.46 * (std::f32::consts::PI * v.min(0.97)).sin().powf(0.6);
            let twig = across < 0.04;
            let needle = across < half_width && ((v * 18.0 + across * 2.2) % 1.0) < 0.42;
            let on = twig || needle;
            texels.extend_from_slice(&[255, 255, 255, if on { 255 } else { 0 }]);
        }
    }
    Texture::from_rgba("NeedleSpray", w, h, &texels)
}

/// A spruce 1 high of needle-spray cards, for near trees: a trunk (its u past 1.5, for the
/// material's bark) under whorls of sprays, each branch two crossed cards from the trunk out to
/// the crown's cone, drooping, the crown as `SpruceGeometry`'s (0.24 wide at its base, at 0.15
/// up, to a point at the top). Small, open and flat, the cards are what cluster LOD prunes
/// (`ClusterOptions::cards`).
fn card_spruce() -> Geometry {
    let mut parts = Vec::new();
    let mut trunk = CylinderGeometry::new(0.03, 0.008, 0.98, 8, 4);
    for vertex in &mut trunk.vertices {
        vertex.uv[0] += 2.0;
    }
    parts.push((trunk, glam::Mat4::IDENTITY));
    let (mut vertices, mut indices) = (Vec::new(), Vec::new());
    let whorls = 16;
    for k in 0..whorls {
        let f = k as f32 / whorls as f32;
        let y = 0.16 + 0.8 * f;
        let reach = 0.25 * (1.0 - f).powf(0.9) + 0.02;
        let branches = 7 - (k / 5);
        for b in 0..branches {
            let a = (b as f32 + 0.5 * (k % 2) as f32) / branches as f32 * std::f32::consts::TAU + hash01(k * 31 + b) * 0.4;
            let out = glam::Vec3::new(a.cos(), -0.25 - 0.2 * f, a.sin()).normalize();
            let side = glam::Vec3::new(-a.sin(), 0.0, a.cos());
            let width = 0.35 * reach + 0.02;
            // two crossed cards along the branch: flat, and tilted about it
            for (k2, tilt) in [0.0f32, 1.1].into_iter().enumerate() {
                let across = (side * tilt.cos() + out.cross(side).normalize() * tilt.sin()) * width;
                let n = out.cross(across).normalize();
                let base = vertices.len() as u32;
                let root = glam::Vec3::new(0.0, y, 0.0) + out * 0.01;
                for (s, t) in [(-1.0f32, 0.0f32), (1.0, 0.0), (1.0, 1.0), (-1.0, 1.0)] {
                    let p = root + out * (reach * t) + across * (s * 0.5) + glam::Vec3::Y * (0.02 * k2 as f32);
                    vertices.push(kansei_core::geometries::Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: [(s + 1.0) * 0.5, t] });
                }
                indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
            }
        }
    }
    let cards = Geometry::new("Cards", vertices, indices);
    let all: Vec<(&Geometry, glam::Mat4)> = parts.iter().map(|(g, m)| (g, *m)).chain(std::iter::once((&cards, glam::Mat4::IDENTITY))).collect();
    Geometry::merged("CardSpruce", &all)
}

/// The spruces' instance records: (base x, y, z, height), (yaw, tint, 0, 0), on a jittered
/// 7 m grid, off the road and the clearing, thinned in patches; at most `count`.
fn forest(count: usize) -> Vec<f32> {
    let mut records = Vec::new();
    let spacing = 7.0;
    let n = (2.0 * WORLD / spacing) as u32;
    for k in 0..n * n {
        let (i, j) = (k % n, k / n);
        let x = -WORLD + (i as f32 + 0.15 + 0.7 * hash01(k)) * spacing;
        let z = -WORLD + (j as f32 + 0.15 + 0.7 * hash01(k + 7919)) * spacing;
        if (x - road_x(z)).abs() < 7.5 + 3.0 * hash01(k + 31) || x.abs() > WORLD - 8.0 || z.abs() > WORLD - 8.0 {
            continue;
        }
        let (cx, cz, r) = CLEARING;
        if ((x - cx).powi(2) + (z - cz).powi(2)).sqrt() < r + 4.0 * hash01(k + 47) {
            continue;
        }
        // patches thinner and denser than the rest
        let density = 0.55 + 0.45 * ((x / 83.0).sin() * (z / 71.0).cos() + 0.4 * ((x + z) / 37.0).sin());
        if hash01(k + 101) > density {
            continue;
        }
        let h = 14.0 + 14.0 * hash01(k + 13).powf(0.7);
        records.extend_from_slice(&[x, height(x, z) - 0.3, z, h, hash01(k + 3) * std::f32::consts::TAU, hash01(k + 5), 0.0, 0.0]);
    }
    records.truncate(count * 8);
    records
}

/// What lights the surfaces besides the sun.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Gi {
    /// The materials' own sky light: the whole sky, as if no tree stood over them.
    Off,
    /// Their sky light dimmed by the top-down sky occlusion (`Renderer::enable_sky_occlusion`),
    /// as the film does today.
    SkyOcc,
    /// Their sky light dimmed by the sky visibility the clipmap's probes measure
    /// (`kansei_clipmap_sky_visibility`, in the material).
    Visibility,
    /// Voxel GI on screen, cones traced per pixel through the clipmap: the sky past the canopy
    /// and the bounces.
    Cones,
    /// Voxel GI on screen from the clipmap's probes: the same light, cheaper and smoother.
    Probes,
    /// The hybrid (`RtDiffuseGiEffect`): one ray for each 2 x 2 pixels through the grid of
    /// triangles near the camera, the hits lit by the sun and the clipmap, the clipmap past them.
    Rt,
}

impl Gi {
    const ALL: [Gi; 6] = [Gi::Off, Gi::SkyOcc, Gi::Visibility, Gi::Cones, Gi::Probes, Gi::Rt];

    fn name(self) -> &'static str {
        match self {
            Gi::Off => "off",
            Gi::SkyOcc => "skyocc",
            Gi::Visibility => "visibility",
            Gi::Cones => "cones",
            Gi::Probes => "probes",
            Gi::Rt => "rt",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        Gi::ALL.into_iter().find(|g| g.name() == name)
    }

    /// AMBIENT_WGSL's mode.
    fn ambient_mode(self) -> u32 {
        match self {
            Gi::SkyOcc => 1,
            Gi::Visibility => 2,
            _ => 0,
        }
    }
}

/// What the screen shows.
#[derive(Clone, Copy, Debug, PartialEq)]
enum View {
    Lit,
    /// The light the GI adds alone.
    Indirect,
    /// The clipmap's voxels and their light.
    Voxels,
}

impl View {
    fn name(self) -> &'static str {
        match self {
            View::Lit => "lit",
            View::Indirect => "indirect",
            View::Voxels => "voxels",
        }
    }

    fn from_name(name: &str) -> Option<Self> {
        match name {
            "lit" => Some(View::Lit),
            "indirect" => Some(View::Indirect),
            "voxels" => Some(View::Voxels),
            _ => None,
        }
    }
}

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

/// The hybrid's settings (`gi=rt`): the URL's `rtgi_*` parameters and `set_rtgi`'s keys.
#[derive(Clone, Copy, Debug, PartialEq)]
struct RtGi {
    resolution: RtGiResolution,
    denoise: RtGiDenoise,
    kernel: RtGiKernel,
    hit: RtGiHitLighting,
    shadows: RtGiShadows,
    mode: RtGiMode,
    accumulate: bool,
    view: RtGiView,
    /// Metres a ray walks the grid before the clipmap takes over (0: the grid's box).
    near: f32,
}

impl RtGi {
    const KEYS: [&'static str; 9] = ["res", "denoise", "kernel", "hit", "shadows", "mode", "accum", "view", "near"];

    fn from_url() -> Self {
        // an 8 m near field: half the trace of the whole 64 m box, for some light lost under the
        // canopy past it, which the clipmap's voxels are too coarse to hold
        let mut r = Self {
            resolution: RtGiResolution::Half,
            denoise: RtGiDenoise::Svgf,
            kernel: RtGiKernel::Three,
            hit: RtGiHitLighting::Direct,
            shadows: RtGiShadows::Rays,
            mode: RtGiMode::Hybrid,
            accumulate: false,
            view: RtGiView::Lit,
            near: 8.0,
        };
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
            "near" => v.parse::<f32>().ok().map(|x| self.near = x.max(0.0)),
            _ => None,
        }
        .is_some()
    }

    /// Bring `e` to these settings (its history restarts).
    fn apply_to(&self, e: &mut RtDiffuseGiEffect) {
        e.set_resolution(self.resolution);
        e.denoise = self.denoise;
        e.kernel = self.kernel;
        e.hit_lighting = self.hit;
        e.shadows = self.shadows;
        e.mode = self.mode;
        e.accumulate = self.accumulate;
        e.near_distance = self.near;
        e.reset_history();
    }

    /// The settings as JSON members (for `info`).
    fn json(&self) -> String {
        format!(
            "\"res\":\"{}\",\"denoise\":\"{}\",\"kernel\":\"{}\",\"hit\":\"{}\",\"shadows\":\"{}\",\"mode\":\"{}\",\"accum\":{},\"view\":\"{}\",\"near\":{}",
            self.resolution.name(),
            self.denoise.name(),
            self.kernel.name(),
            self.hit.name(),
            self.shadows.name(),
            self.mode.name(),
            self.accumulate,
            self.view.name(),
            self.near
        )
    }
}

/// Camera presets: name, target, distance, azimuth and elevation (radians).
const CAMERAS: [(&str, [f32; 3], f32, f32, f32); 4] = [
    ("road", [6.0, 1.6, -10.0], 9.0, 2.6, 0.05),
    ("clearing", [45.0, 2.0, 60.0], 24.0, 1.1, 0.12),
    ("forest", [-40.0, 2.0, 30.0], 6.0, 0.4, 0.15),
    ("high", [0.0, 10.0, 0.0], 120.0, 2.4, 0.55),
];

#[derive(Default)]
struct Stats {
    since: f64,
    frames: u32,
    frame_ms: f64,
    gpu_ms: f64,
    gpu_span_ms: f64,
    passes: Vec<(&'static str, f64)>,
}

struct State {
    renderer: Renderer,
    scene: Scene,
    camera: Camera,
    controls: CameraControls,
    sky: SkyAtmosphere,
    volume: PostProcessingVolume,
    /// The voxels' debug view: the GI effect's view of them and the display transform alone, no
    /// atmosphere over them.
    debug_volume: PostProcessingVolume,
    sun_light: usize,
    gi: Gi,
    /// AMBIENT_WGSL's mode, read by every material.
    ambient_mode: wgpu::Buffer,
    view: View,
    elevation: f32,
    bearing: f32,
    /// `fly=1`: the camera drives along the road, at this speed (m/s)
    fly: Option<f32>,
    time: f32,
    trees: usize,
    triangles: u64,
    stats: Option<Stats>,
    /// The hybrid's settings (its effect exists with the grid: `gi=rt`, `rt=1` or `reflect=1`).
    rtgi: RtGi,
}

thread_local! {
    static STATE: RefCell<Option<Rc<RefCell<State>>>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> Option<R> {
    let state = STATE.with(|s| s.borrow().clone())?;
    let mut st = state.borrow_mut();
    Some(f(&mut st))
}

/// Exposure for the forest under a sun `elevation` degrees up: the open landscape's (the
/// sky-atmosphere example's curve), opened up for a forest's shade.
fn forest_ev100(elevation: f32) -> f32 {
    auto_ev100(elevation) - 1.8
}

/// Exposure for a sun `elevation` degrees up (sky-atmosphere's curve).
fn auto_ev100(elevation: f32) -> f32 {
    const CURVE: [(f32, f32); 8] = [(-8.0, 5.0), (-4.0, 7.3), (-2.5, 8.2), (0.0, 9.8), (2.0, 11.2), (6.0, 12.8), (15.0, 14.3), (40.0, 15.0)];
    if elevation <= CURVE[0].0 {
        return CURVE[0].1;
    }
    for w in CURVE.windows(2) {
        let ((e0, v0), (e1, v1)) = (w[0], w[1]);
        if elevation <= e1 {
            return v0 + (v1 - v0) * (elevation - e0) / (e1 - e0);
        }
    }
    CURVE[CURVE.len() - 1].1
}

impl State {
    fn apply(&mut self) {
        // the clipmap runs for its GI and its probes; the effect shows the GI on screen
        let clipmap = matches!(self.gi, Gi::Visibility | Gi::Cones | Gi::Probes | Gi::Rt);
        let on_screen = matches!(self.gi, Gi::Cones | Gi::Probes);
        if let Some(gi) = self.renderer.voxel_clipmap_mut() {
            gi.settings.enabled = clipmap;
        }
        self.renderer.queue().write_buffer(&self.ambient_mode, 0, &[self.gi.ambient_mode().to_le_bytes(), [0; 4], [0; 4], [0; 4]].concat());
        let probes = self.renderer.voxel_clipmap().and_then(|g| g.probes()).filter(|_| self.gi == Gi::Probes);
        // the fog: lit by the probes with the clipmap, by the sky dimmed by the sky occlusion with it
        let fog_probes = self.renderer.voxel_clipmap().and_then(|g| g.probes()).filter(|_| clipmap);
        let fog_occlusion = self.renderer.sky_occlusion().filter(|_| self.gi == Gi::SkyOcc);
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.set_clipmap_probes(fog_probes);
            fog.set_sky_occlusion(fog_occlusion);
        }
        if let Some(effect) = self.volume.effect_mut::<VoxelGIEffect>() {
            effect.enabled = on_screen;
            effect.set_clipmap_probes(probes);
            effect.show_indirect = self.view == View::Indirect;
            effect.reset_history();
        }
        let (rtgi, indirect) = (self.rtgi, self.view == View::Indirect);
        if let Some(effect) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            effect.enabled = self.gi == Gi::Rt;
            rtgi.apply_to(effect);
            effect.view = if indirect { RtGiView::Indirect } else { rtgi.view };
        }
    }

    fn set_camera(&mut self, name: &str) {
        self.fly = (name == "fly").then_some(12.0);
        let (_, target, distance, azimuth, elevation) = CAMERAS.iter().find(|c| c.0 == name).copied().unwrap_or(CAMERAS[0]);
        let ground = height(target[0], target[2]);
        self.controls.set_view(Vec3::new(target[0], ground + target[1], target[2]), distance, azimuth, elevation);
        self.camera.reset_motion();
        for volume in [&mut self.volume, &mut self.debug_volume] {
            if let Some(effect) = volume.effect_mut::<VoxelGIEffect>() {
                effect.reset_history();
            }
        }
        if let Some(effect) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            effect.reset_history();
        }
    }

    fn frame(&mut self, frame: &Frame) {
        frame.resize(&mut self.renderer, &mut self.camera);
        let now = now() * 1000.0;
        let dt = frame.dt.clamp(0.0, 0.1);
        self.time += dt;
        match self.fly {
            Some(speed) => {
                // along the road, eyes 1.7 m up, looking ahead
                let z = 200.0 - (self.time * speed) % 400.0;
                let (x, ahead) = (road_x(z), z - 25.0);
                let y = height(x, z) + 1.7;
                self.camera.set_position(x, y, z);
                self.camera.look_at(&Vec3::new(road_x(ahead), height(road_x(ahead), ahead) + 1.2, ahead));
            }
            None => self.controls.update(&mut self.camera, dt),
        }
        let sun_dir = direction_from_elevation_bearing(self.elevation, self.bearing);
        self.sky.sun.direction = sun_dir;
        let eye = *self.camera.position();
        if let Some(Light::Directional(l)) = self.scene.get_light_mut(self.sun_light) {
            l.direction = Vec3::new(-sun_dir.x, -sun_dir.y, -sun_dir.z);
            l.color = self.sky.sun_illuminance_at(eye);
            l.intensity = 1.0;
        }
        if let Some(fog) = self.volume.effect_mut::<VolumetricFogEffect>() {
            fog.update_lights(self.scene.lights());
            fog.time = self.time;
        }
        if let Some(gi) = self.volume.effect_mut::<RtDiffuseGiEffect>() {
            gi.update_lights(self.scene.lights());
        }
        self.sky.update(self.renderer.device(), self.renderer.queue(), &mut self.camera);
        let volume = if self.view == View::Voxels && self.renderer.voxel_clipmap().is_some_and(|g| g.settings.enabled) { &mut self.debug_volume } else { &mut self.volume };
        self.renderer.render_with_postprocessing(&mut self.scene, &mut self.camera, volume);
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

    /// The ray tracing grid's figures as JSON (null without it).
    fn rt_info(&self) -> String {
        let Some(rt) = self.renderer.rt_grid() else { return "null".into() };
        let s = rt.stats();
        let (lo, hi) = rt.grid().bounds();
        format!(
            "{{\"triangles\":{},\"references\":{},\"big\":{},\"sources\":{},\"rebuilt\":{},\"rebuilds\":{},\"cpu_ms\":{:.3},\"mib\":{:.1},\"box\":[[{:.1},{:.1},{:.1}],[{:.1},{:.1},{:.1}]]}}",
            s.grid.triangles, s.grid.references, s.grid.big_triangles, s.sources, s.rebuilt, s.rebuilds, s.cpu_ms, rt.memory_bytes() as f64 / (1 << 20) as f64, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z
        )
    }

    /// The reflections' settings and counters as JSON (null without them).
    fn reflect_info(&self) -> String {
        let Some(r) = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtReflectionsEffect>()) else { return "null".into() };
        let s = r.stats().unwrap_or_default();
        format!(
            "{{\"enabled\":{},\"view\":\"{}\",\"res\":\"{}\",\"alpha\":{},\"grid\":{},\"rays\":{},\"hits\":{},\"cells\":{},\"tests\":{},\"max_cost\":{}}}",
            r.enabled,
            reflection_view_name(r.view),
            if r.resolution() == RtTraceResolution::Quarter { "quarter" } else { "half" },
            r.alpha_test,
            r.trace_grid,
            s.rays,
            s.hits,
            s.cells,
            s.tests,
            s.max_cost
        )
    }

    /// The hybrid's settings, and with its effect its counters, memory and accumulated frames
    /// (null without the effect).
    fn rtgi_info(&self) -> String {
        let Some(e) = self.volume.effects.iter().find_map(|e| e.as_any().downcast_ref::<RtDiffuseGiEffect>()) else { return "null".into() };
        let s = e.stats().unwrap_or_default();
        format!(
            "{{{},\"on\":{},\"accumulated\":{},\"mib\":{:.1},\"rays\":{},\"hits\":{},\"cost\":{},\"shadow_rays\":{}}}",
            self.rtgi.json(),
            e.enabled,
            e.accumulated(),
            e.memory_bytes() as f64 / (1 << 20) as f64,
            s.rays,
            s.hits,
            s.cost,
            s.shadow_rays
        )
    }

    fn info(&self) -> String {
        let gi = self.renderer.voxel_clipmap();
        let passes: Vec<String> = self.stats.as_ref().map_or(Vec::new(), |s| s.passes.iter().map(|(l, ms)| format!("[\"{l}\",{ms:.3}]")).collect());
        // (+ 0.0: an empty sum is -0)
        let sum = |prefix: &str| self.stats.as_ref().map_or(0.0, |s| s.passes.iter().filter(|p| p.0.starts_with(prefix)).map(|p| p.1).sum::<f64>()) + 0.0;
        let eye = self.camera.position();
        let layout = gi.map(|g| *g.clipmap().layout());
        format!(
            "{{\"rtgi\":{},\"rtgi_ms\":{:.3},\"rt\":{},\"rt_ms\":{:.3},\"reflect\":{},\"reflect_ms\":{:.3},\"gi\":\"{}\",\"view\":\"{}\",\"levels\":{},\"dims\":{},\"voxel\":{},\"mib\":{:.1},\"filling\":{},\"trees\":{},\"triangles\":{},\"elevation\":{},\"eye\":[{:.1},{:.1},{:.1}],\"stats\":{},\"frame_ms\":{:.2},\"gpu_ms\":{:.3},\"gpu_span_ms\":{:.3},\"voxelize_ms\":{:.3},\"inject_ms\":{:.3},\"screen_ms\":{:.3},\"passes\":[{}]}}",
            self.rtgi_info(),
            sum("RtGi/"),
            self.rt_info(),
            sum("Rt/Gather") + sum("Rt/Grid"),
            self.reflect_info(),
            sum("Rt/Trace") + sum("Rt/Resolve"),
            self.gi.name(),
            self.view.name(),
            layout.map_or(0, |l| l.levels),
            layout.map_or("null".into(), |l| format!("{:?}", l.dims)),
            layout.map_or(0.0, |l| l.voxel_size),
            gi.map_or(0.0, |g| g.memory_bytes() as f64 / (1 << 20) as f64),
            gi.is_some_and(|g| g.filling()),
            self.trees,
            self.triangles,
            self.elevation,
            eye.x,
            eye.y,
            eye.z,
            self.stats.is_some(),
            self.stats.as_ref().map_or(0.0, |s| s.frame_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_ms),
            self.stats.as_ref().map_or(0.0, |s| s.gpu_span_ms),
            sum("VoxelClipmap/Voxelize") + sum("VoxelClipmap/Dynamic"),
            sum("VoxelClipmap/Inject"),
            sum("VoxelGI/Screen"),
            passes.join(","),
        )
    }
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    let canvas = Canvas::find(canvas_id)?;
    let mut renderer = canvas.renderer(RendererConfig { sample_count: 1, clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0), ..Default::default() }).await;
    renderer.enable_cascaded_shadows(CascadedShadowOptions { max_distance: param_or("shadow_far", 160.0), caster_distance: 120.0, ..Default::default() });
    // levels=<n>, res=<voxels across>, vox=<finest voxel, metres>
    let options = SceneVoxelClipmapOptions {
        levels: param_or("levels", 5),
        resolution: param_or("res", 64),
        height_resolution: param_or("res", 64u32) / 2,
        voxel_size: param_or("vox", 0.5),
        ..Default::default()
    };
    renderer.enable_voxel_clipmap(options);
    // the film's sky occlusion, for comparison: the trees seen from above
    renderer.enable_sky_occlusion(SkyOcclusionOptions { extent_m: 320.0, min_height_m: -10.0, max_height_m: 90.0, volume_size: (128, 32), layer_mask: TREE_LAYER, ..Default::default() });
    // rt=1: a ray tracing grid of the scene's triangles round the camera, 64 x 32 x 64 m of 0.5 m
    // cells, its trees by their cluster cut at a cell of error; rt_cell=<m> (the box stays
    // 64 m across), rt_rebuild=1 rebuilds it every frame
    // reflect=1: ray-traced reflections on the road, wet (wet=all: everywhere), through the grid;
    // gi=rt: the hybrid GI's rays through it
    let reflect = flag("reflect", false);
    let gi_mode = param("gi").and_then(|g| Gi::from_name(&g)).unwrap_or(Gi::Cones);
    let rt = flag("rt", false) || reflect || gi_mode == Gi::Rt;
    // the wet surfaces' F0 (the road's, the rest's) and roughness, 0 when dry (the default)
    let wet = match (reflect, param("wet").as_deref()) {
        (false, _) => [0.0; 4],
        (true, Some("all")) => [param_or("wet_f0", 0.04), param_or("wet_f0", 0.04), param_or("wet_rough", 0.1), 0.0],
        (true, _) => [param_or("wet_f0", 0.04), 0.0, param_or("wet_rough", 0.1), 0.0],
    };
    if rt {
        let cell: f32 = param_or("rt_cell", 0.5);
        let across = (64.0 / cell / 4.0).round() as u32 * 4;
        renderer.enable_rt_grid(SceneRtGridOptions {
            grid: RtGridOptions { dims: [across, across / 2, across], cell, below: 0.25, ..Default::default() },
            cluster_error_cells: 1.0,
            rebuild_every_frame: flag("rt_rebuild", false),
        });
    }

    let mut sky = SkyAtmosphere::new(renderer.device(), SkyAtmosphereOptions::default());
    sky.sun.illuminance = Vec3::new(100_000.0, 100_000.0, 100_000.0);
    let gi = renderer.voxel_clipmap_mut().unwrap();
    gi.use_sky_lighting(&sky.bindings().sky_lighting);
    // lit=<levels a frame> (0: all), probes=<probes a frame>
    gi.settings.levels_per_frame = param_or("lit", gi.settings.levels_per_frame);
    // coneshadows=off|fallback|always, bounce=<share>, shadowsteps=<n>
    if let Some(mode) = param("coneshadows").and_then(|m| ConeShadows::from_name(&m)) {
        gi.settings.cone_shadows = mode;
    }
    gi.settings.bounce = param_or("bounce", gi.settings.bounce);
    gi.settings.cone_shadow_steps = param_or("shadowsteps", gi.settings.cone_shadow_steps);
    gi.enable_probes(ClipmapProbeOptions { probes_per_frame: param_or("probes", ClipmapProbeOptions::default().probes_per_frame), ..Default::default() });
    let ambient_mode = renderer.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("AmbientMode"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let ambient = AmbientSources { sky: &sky, mode: &ambient_mode, occlusion: renderer.sky_occlusion().unwrap(), probes: renderer.voxel_clipmap().unwrap().probes().unwrap() };

    let mut scene = Scene::new();
    let mut triangles = 0u64;
    // the terrain, in tiles (the voxelizer skips those away from what it voxelizes)
    let tile = 2.0 * WORLD / TILES as f32;
    for j in 0..TILES {
        for i in 0..TILES {
            let min = [-WORLD + i as f32 * tile, -WORLD + j as f32 * tile];
            let mut geometry = HeightfieldGeometry::new(min, [min[0] + tile, min[1] + tile], (80, 80), height);
            geometry.label = "Terrain".into();
            triangles += geometry.indices.len() as u64 / 3;
            let mut r = Renderable::new(geometry, ground_material("Terrain", TERRAIN_ALBEDO_WGSL, GRASS, &ambient, wet)).with_gi(GiSurface::new(GRASS));
            r.cast_shadow = true;
            r.rt = rt.then(|| RtSurface::new(GRASS));
            scene.add(SceneNode::Renderable(r));
        }
    }
    // rocks along the road and a cabin in the clearing
    for k in 0..40u32 {
        let z = -200.0 + k as f32 * 10.0 + 6.0 * hash01(k);
        let side = if hash01(k + 9) > 0.5 { 1.0 } else { -1.0 };
        let x = road_x(z) + side * (5.0 + 3.0 * hash01(k + 2));
        let size = 0.4 + 1.4 * hash01(k + 4).powi(2);
        let mut rock = Renderable::new(IcosphereGeometry::new(size, 2), ground_material("Rock", CONSTANT_ALBEDO_WGSL, ROCK, &ambient, wet)).with_gi(GiSurface::new(ROCK));
        rock.object.set_position(x, height(x, z) + 0.2 * size, z);
        rock.object.scale = Vec3::new(1.0, 0.6, 1.2);
        rock.object.rotation.y = hash01(k + 6) * 3.0;
        rock.rt = rt.then(|| RtSurface::new(ROCK));
        scene.add(SceneNode::Renderable(rock));
    }
    let (cx, cz, _) = CLEARING;
    let wall = [0.42, 0.18, 0.12];
    let ground = height(cx, cz);
    for (size, offset, albedo) in [([6.0, 3.2, 4.5], [0.0, 1.6, 0.0], wall), ([6.6, 0.3, 5.4], [0.0, 3.5, 0.0], [0.2, 0.2, 0.22])] {
        let mut part = Renderable::new(BoxGeometry::new(size[0], size[1], size[2]), ground_material("Cabin", CONSTANT_ALBEDO_WGSL, albedo, &ambient, wet)).with_gi(GiSurface::new(albedo));
        part.object.set_position(cx + offset[0], ground + offset[1] - 0.3, cz + offset[2]);
        part.object.rotation.y = 0.5;
        part.rt = rt.then(|| RtSurface::new(albedo));
        scene.add(SceneNode::Renderable(part));
    }

    // the forest: three LODs of one spruce sharing the instances, each culled per view with
    // crossfades at its bands' edges
    let records = forest(param_or("trees", 20_000usize));
    let trees = records.len() / 8;
    let source = ComputeBuffer::from_slice("Spruces", BufferType::Storage, BufferUsage::VERTEX | BufferUsage::STORAGE, &records);
    let cards = flag("cards", true);
    for (kind, (near, far), (gi_near, gi_far)) in LODS {
        let (kind, gi_band) = match (kind, cards) {
            (TreeMesh::Cards, false) => (TreeMesh::Cones(14, 5, 9), (0.0, 0.0)),
            _ => (kind, (gi_near, gi_far)),
        };
        let mesh = match kind {
            TreeMesh::Cards => card_spruce(),
            TreeMesh::Cones(segments, rings, cones) => SpruceGeometry::new(segments, rings, cones),
        };
        triangles += (mesh.indices.len() / 3 * trees) as u64;
        let clusters = matches!(kind, TreeMesh::Cards).then(|| {
            ClusterLod::new(ClusterMesh::build(&mesh, &ClusterOptions { cards: true, card_error_scale: 0.25, ..Default::default() }))
                .with_transform(InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: 1.0, rotation: None })
                .with_stretch(TREE_STRETCH)
        });
        let culled = source.clone().with_vertex_layout(
            36,
            vec![
                InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: 5, offset: 32, format: VertexFormat::Float32 },
            ],
        );
        // the cones stand for needles light passes between; the cards' sprays are the needles
        let opacity = if clusters.is_some() { 1.0 } else { param_or("crown_opacity", CROWN_OPACITY) };
        let mut r = Renderable::new(InstancedGeometry::new(Geometry::new("Spruce", mesh.vertices, mesh.indices), trees as u32, vec![culled]), tree_material("Spruce", &ambient, clusters.is_some()))
            .with_gi(GiSurface::new(NEEDLES).with_opacity(opacity));
        // in the ray tracing grid: the cards' sprays cut out by the needles' alpha (layer 0)
        if rt {
            let surface = RtSurface::new(NEEDLES);
            r.rt = Some(if clusters.is_some() { surface.with_alpha_layer(0) } else { surface });
            r.rt_placement = Some(RtPlacement::Wgsl(TREE_PLACEMENT_WGSL.into()));
        }
        r.clusters = clusters;
        // the canopy the sky occlusion sees from above (the terrain is not)
        r.layers = Renderable::DEFAULT_LAYERS | TREE_LAYER;
        r.instance_culling = Some(
            InstanceCulling::from_buffer(&source, trees as u32, 32, 0, 0.6)
                .with_radius_scale(12)
                .with_bounds_shift(glam::Vec3::new(0.0, 0.5, 0.0))
                .with_bounds_box(glam::Vec3::new(0.3, 0.5, 0.3))
                .with_lod_range(near, far)
                .with_gi_lod_range(gi_band.0, gi_band.1)
                .with_crossfade(CROSSFADE),
        );
        scene.add(SceneNode::Renderable(r));
    }

    let mut sun = DirectionalLight::new(Vec3::new(0.0, -1.0, 0.0), Vec3::ZERO, 1.0);
    sun.cast_shadow = true;
    let sun_light = scene.add(SceneNode::Light(Light::Directional(sun)));
    drop(ambient);

    // the chain: the GI first (it lies on the surfaces under the aerial perspective), the
    // atmosphere, TAA, the display transform
    let elevation: f32 = param_or("elevation", 16.0);
    let bearing: f32 = param_or("bearing", 160.0);
    let tonemap = ToneMapEffect::new(ToneMapOptions {
        exposure: exposure_from_ev100_lens(param_or("ev", forest_ev100(elevation)), LENS_ATTENUATION_UE4),
        exposure_compensation: 1.0,
        tonemapper: ToneMapper::AcesFitted,
        ..ToneMapOptions::for_surface(renderer.presentation_format())
    });
    let mut gi = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions { intensity: param_or("intensity", 1.0), ..Default::default() });
    gi.set_sky_lighting(Some(&sky.bindings().sky_lighting));
    let mut effects: Vec<Box<dyn PostProcessingEffect>> = vec![Box::new(gi)];
    // with the grid, the hybrid in voxel GI's place (on in gi=rt): rays through the grid, the
    // cards alpha-tested, the hits lit by the sun (shadow rays, the cascades past the grid) and
    // the clipmap, the clipmap and the sky past the grid
    if rt {
        let options = RtDiffuseGiOptions { covered_wgsl: Some(CARD_COVERED_WGSL.into()), ..Default::default() };
        let mut hybrid = RtDiffuseGiEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), renderer.rt_grid().unwrap().handle(), options);
        hybrid.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        let mut needles = needle_texture();
        needles.initialize_with_data(renderer.device(), renderer.queue());
        hybrid.set_alpha_texture(needles.view());
        hybrid.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        hybrid.heat_scale = exposure_from_ev100_lens(param_or("ev", forest_ev100(elevation)), LENS_ATTENUATION_UE4).recip() * 0.5;
        hybrid.collect_stats = flag("stats", false);
        effects.push(Box::new(hybrid));
    }
    // the reflections: after the GI (they reflect its light), under the aerial perspective
    if reflect {
        let options = RtReflectionsOptions {
            resolution: if param("rt_res").as_deref() == Some("quarter") { RtTraceResolution::Quarter } else { RtTraceResolution::Half },
            alpha_test: flag("rt_alpha", true),
            covered_wgsl: Some(CARD_COVERED_WGSL.into()),
            ..Default::default()
        };
        let mut reflections = RtReflectionsEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), renderer.rt_grid().unwrap().handle(), options);
        reflections.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        let mut needles = needle_texture();
        needles.initialize_with_data(renderer.device(), renderer.queue());
        reflections.set_alpha_texture(needles.view());
        reflections.view = param("rt_view").and_then(|v| reflection_view(&v)).unwrap_or_default();
        reflections.heat_scale = exposure_from_ev100_lens(param_or("ev", forest_ev100(elevation)), LENS_ATTENUATION_UE4).recip() * 0.5;
        reflections.collect_stats = flag("stats", false);
        reflections.trace_grid = param("rt_trace").as_deref() != Some("voxels");
        effects.push(Box::new(reflections));
    }
    effects.push(Box::new(AtmosphereEffect::new(&sky)));
    // fog=<density>: mist in the valley, lit by the sun through the cascades and by the sky, or by
    // the clipmap's probes in its modes (the light of the sunlit clearing, the sky past the trees)
    let fog_density: f32 = param_or("fog", 0.0);
    if fog_density > 0.0 {
        let mut fog = VolumetricFogEffect::new(VolumetricFogOptions {
            grid: FroxelGridOptions { near: 0.5, far: 240.0, temporal: true, blend_factor: 0.1, ..Default::default() },
            base_density: fog_density,
            height_falloff: 0.06,
            anisotropy: 0.55,
            albedo: Vec3::new(0.9, 0.92, 0.95),
            ..Default::default()
        });
        fog.set_sky_lighting(Some(&sky.bindings().sky_lighting));
        fog.set_cascaded_shadow_map(renderer.cascaded_shadow_map());
        effects.push(Box::new(fog));
    }
    effects.push(Box::new(TemporalAAEffect::new(TemporalAAOptions { exposure: tonemap.total_exposure(), ..Default::default() })));
    effects.push(Box::new(tonemap));
    let volume = PostProcessingVolume::new(&renderer, effects);
    let mut voxels = VoxelGIEffect::with_clipmap(renderer.voxel_clipmap().unwrap().clipmap(), VoxelGIOptions::default());
    voxels.show_voxels = true;
    let debug_tonemap = ToneMapEffect::new(ToneMapOptions {
        exposure: exposure_from_ev100_lens(param_or("ev", forest_ev100(elevation)), LENS_ATTENUATION_UE4),
        exposure_compensation: 1.0,
        tonemapper: ToneMapper::AcesFitted,
        ..ToneMapOptions::for_surface(renderer.presentation_format())
    });
    let debug_volume = PostProcessingVolume::new(&renderer, vec![Box::new(voxels), Box::new(debug_tonemap)]);
    let camera = Camera::new(55.0, 0.2, 4000.0, canvas.aspect());
    let controls = CameraControls::from_canvas(canvas.element(), Vec3::new(0.0, 0.0, 0.0), 10.0).with_mouse_pan(canvas.element());

    let stats = flag("stats", false).then(|| Stats { since: now() * 1000.0, ..Default::default() });
    if stats.is_some() {
        renderer.set_profiling(true);
    }
    let mut state = State {
        renderer,
        scene,
        camera,
        controls,
        sky,
        volume,
        debug_volume,
        sun_light,
        gi: gi_mode,
        ambient_mode,
        view: param("view").and_then(|v| View::from_name(&v)).unwrap_or(View::Lit),
        elevation,
        bearing,
        fly: None,
        time: 0.0,
        trees,
        triangles,
        stats,
        rtgi: RtGi::from_url(),
    };
    state.set_camera(param("cam").as_deref().unwrap_or("road"));
    state.apply();
    log::info!("Kansei — Outdoor GI (WASM) ready: {}", state.info());

    let state = Rc::new(RefCell::new(state));
    STATE.with(|s| *s.borrow_mut() = Some(state.clone()));
    kansei_wasm::run(&canvas, move |frame| state.borrow_mut().frame(frame));
    Ok(())
}

/// The state as JSON: the settings, the clipmap, the scene and (with stats) the times.
#[wasm_bindgen]
pub fn info() -> String {
    with_state(|s| s.info()).unwrap_or_default()
}

/// `off`, `skyocc`, `visibility`, `cones`, `probes` or `rt` (`Gi`; `rt` only where the page
/// built the grid: `gi=rt`, `rt=1` or `reflect=1`).
#[wasm_bindgen]
pub fn set_gi(name: &str) {
    with_state(|s| {
        if let Some(gi) = Gi::from_name(name) {
            if gi == Gi::Rt && s.volume.effect_mut::<RtDiffuseGiEffect>().is_none() {
                return;
            }
            s.gi = gi;
            s.apply();
        }
    });
}

/// One of the hybrid's settings at run time, by its `rtgi_*` URL parameter's key (`res`,
/// `denoise`, `kernel`, `hit`, `shadows`, `mode`, `accum`, `view`, `near`) and value.
#[wasm_bindgen]
pub fn set_rtgi(key: &str, value: &str) {
    with_state(|s| {
        if s.rtgi.set(key, value) {
            s.apply();
        }
    });
}

/// `lit`, `indirect` or `voxels`.
#[wasm_bindgen]
pub fn set_view(name: &str) {
    with_state(|s| {
        if let Some(view) = View::from_name(name) {
            s.view = view;
            s.apply();
        }
    });
}

/// A camera preset (`road`, `clearing`, `forest`, `high`) or `fly` (along the road).
#[wasm_bindgen]
pub fn set_camera(name: &str) {
    with_state(|s| s.set_camera(name));
}

/// The sun's elevation, degrees (the exposure follows).
#[wasm_bindgen]
pub fn set_elevation(degrees: f32) {
    with_state(|s| {
        s.elevation = degrees;
        if let Some(tonemap) = s.volume.effect_mut::<ToneMapEffect>() {
            tonemap.options.exposure = exposure_from_ev100_lens(forest_ev100(degrees), LENS_ATTENUATION_UE4);
        }
        if let Some(tonemap) = s.debug_volume.effect_mut::<ToneMapEffect>() {
            tonemap.options.exposure = exposure_from_ev100_lens(forest_ev100(degrees), LENS_ATTENUATION_UE4);
        }
    });
}

/// Per-pass GPU times in `info()`.
#[wasm_bindgen]
pub fn set_stats(on: bool) {
    with_state(|s| {
        s.renderer.set_profiling(on);
        s.stats = on.then(|| Stats { since: now() * 1000.0, ..Default::default() });
    });
}

fn with_reflections(f: impl FnOnce(&mut RtReflectionsEffect)) {
    with_state(|s| {
        if let Some(r) = s.volume.effect_mut::<RtReflectionsEffect>() {
            f(r);
        }
    });
}

/// The ray-traced reflections on or off (with `reflect=1`).
#[wasm_bindgen]
pub fn set_reflections(on: bool) {
    with_reflections(|r| {
        r.enabled = on;
        r.reset_history();
    });
}

/// `lit`, `reflection` (the light they add), `mirror` (what the rays see) or `cost`.
#[wasm_bindgen]
pub fn set_reflection_view(name: &str) {
    with_reflections(|r| r.view = reflection_view(name).unwrap_or_default());
}

/// The cards' alpha test in the reflections (off: the cards are solid).
#[wasm_bindgen]
pub fn set_reflection_alpha(on: bool) {
    with_reflections(|r| r.alpha_test = on);
}

/// Trace the grid of triangles, or (off) the voxel cone alone.
#[wasm_bindgen]
pub fn set_reflection_grid(on: bool) {
    with_reflections(|r| r.trace_grid = on);
}

/// `half` or `quarter`: one pixel of each 2 x 2 or 4 x 4 traced a frame.
#[wasm_bindgen]
pub fn set_reflection_resolution(name: &str) {
    with_reflections(|r| r.set_resolution(if name == "quarter" { RtTraceResolution::Quarter } else { RtTraceResolution::Half }));
}
