/**
 * The ray tracing grid's WGSL, imported from `rust/kansei-core/src/rt/shaders` (Vite `?raw`) as
 * `gi/GiWGSL.ts` imports voxel GI's, so both engines run the same source and
 * `cargo test -p kansei-core` validates it. The exported names match the Rust constants
 * (`rt::RT_GRID_WGSL`, `rt::RT_OPAQUE_WGSL`); the rest are the passes' own, assembled as Rust's
 * `concat!` and `format!`.
 */
import { assemble } from '../materials/shaders/ShaderUtils';
import rtTypes from '../../rust/kansei-core/src/rt/shaders/rt_types.wgsl?raw';
import rtGrid from '../../rust/kansei-core/src/rt/shaders/rt_grid.wgsl?raw';
import rtBuild from '../../rust/kansei-core/src/rt/shaders/rt_build.wgsl?raw';
import rtGather from '../../rust/kansei-core/src/rt/shaders/rt_gather.wgsl?raw';
import rtReflectCommon from '../../rust/kansei-core/src/rt/shaders/rt_reflect_common.wgsl?raw';
import rtReflectTrace from '../../rust/kansei-core/src/rt/shaders/rt_reflect_trace.wgsl?raw';
import rtReflectResolve from '../../rust/kansei-core/src/rt/shaders/rt_reflect_resolve.wgsl?raw';
import rtReflectHit from '../../rust/kansei-core/src/rt/shaders/rt_reflect_hit.wgsl?raw';
import rtGlass from '../../rust/kansei-core/src/rt/shaders/rt_glass.wgsl?raw';
import { SPOT_LIGHT_TYPES_WGSL } from '../materials/shaders/SharedWGSL';
import rtGiCommon from '../../rust/kansei-core/src/rt/shaders/rt_gi_common.wgsl?raw';
import rtGiTrace from '../../rust/kansei-core/src/rt/shaders/rt_gi_trace.wgsl?raw';
import rtGiSvgf from '../../rust/kansei-core/src/rt/shaders/rt_gi_svgf.wgsl?raw';
import rtGiComposite from '../../rust/kansei-core/src/rt/shaders/rt_gi_composite.wgsl?raw';
import voxelIrradiance from '../../rust/kansei-core/src/gi/shaders/voxel_irradiance.wgsl?raw';
import { CLIPMAP_WGSL, SKY_LIGHTING_WGSL, VOXEL_CONES_WGSL } from '../gi/GiWGSL';
import { COMPUTE_SHADOWS_WGSL } from '../shadows/ComputeShadows';

/**
 * Ray tracing through an `RtGrid` from a compute pass: `KanseiRtGrid`, `KanseiRtHit` and
 * `kansei_rt_trace(origin, dir, tMin, tMax, flags)`, the closest hit of a ray among the grid's
 * triangles (`KANSEI_RT_ANY_HIT`: any hit, for shadows; `KANSEI_RT_SOLID`: no alpha test), with
 * `kansei_rt_exit` (where a ray leaves the box), `kansei_rt_contains`, and a hit triangle's
 * `kansei_rt_albedo`, `kansei_rt_uv`, `kansei_rt_shading_normal` (`RtSurface.smoothNormals`),
 * `kansei_rt_source` and `kansei_rt_record`. Declare the buffers with `rtGridBindingsWgsl(group, first)` (bind `RtGrid.bindGroupEntries`), and define
 * `fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool`, whether an alpha-tested triangle
 * (`RtSurface.alphaLayer`) is there at `uv` (`RT_OPAQUE_WGSL`: everywhere). Rust: `rt::RT_GRID_WGSL`.
 */
export const RT_GRID_WGSL: string = assemble([rtTypes, rtGrid]);

/** A `kansei_rt_covered` for `RT_GRID_WGSL` that keeps every hit (no alpha test). */
export const RT_OPAQUE_WGSL = 'fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool { return true; }\n';

/**
 * WGSL declaring the grid's buffers in group `group` from binding `first` (three bindings), for
 * `RT_GRID_WGSL`. Rust: `RtGrid::bindings_wgsl`.
 */
export function rtGridBindingsWgsl(group: number, first: number): string {
    return `@group(${group}) @binding(${first}) var<uniform> kansei_rt_grid : KanseiRtGrid;\n`
        + `@group(${group}) @binding(${first + 1}) var<storage, read> kansei_rt_triangles : array<vec4f>;\n`
        + `@group(${group}) @binding(${first + 2}) var<storage, read> kansei_rt_cells : array<u32>;\n`;
}

/** The grid's build: count, scan, fill (`RtGrid`). */
export const RT_BUILD_WGSL: string = assemble([rtTypes, rtBuild]);

/** The gather's WGSL with a placement's `kansei_rt_place` (`RtPlacement`). */
export function rtGatherWgsl(placement: string): string {
    return `${assemble([rtTypes, rtGather])}\n${placement}`;
}

/**
 * The voxel source's functions for the reflections' trace over a voxel volume (group 0 bindings
 * 6-8; its anisotropic mips at 40-45 for `srcIrradiance`, after `voxel_irradiance.wgsl`).
 */
const VOLUME_SOURCE_WGSL = /* wgsl */`
@group(0) @binding(6) var<uniform> vol : VoxelVolume;
@group(0) @binding(7) var volTex : texture_3d<f32>;
@group(0) @binding(8) var volSampler : sampler;
fn srcVoxelSize(p: vec3f) -> f32 {
    return vol.voxelSize;
}
fn srcHitRadiance(p: vec3f, nf: vec3f) -> vec3f {
    var s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p + nf * (0.25 * vol.voxelSize)), 0.0);
    if (s.a < 0.05) { s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p - nf * (0.25 * vol.voxelSize)), 0.0); }
    return s.rgb / max(s.a, 0.05) * vol.radianceScale;
}
fn srcCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return voxelConeTrace(vol, volTex, volSampler, origin, dir, tanHalf, startDist, maxDist, steps);
}
// The irradiance a surface at p (normal n) receives from the voxels and the sky past them, as
// voxel GI's composite gathers it: its six cones through the anisotropic mips (bindings 40-45,
// RT_REFLECT_ANISO), else five through the isotropic ones.
fn srcIrradiance(p: vec3f, n: vec3f) -> vec3f {
    if ((rp.flags & RT_REFLECT_ANISO) != 0u) {
        return voxelIrradiance(vol, volTex, volSampler, sky, rp.skyScale, p, n, 0.0, vol.voxelSize, rp.maxDistance, 32u, 1.0).rgb;
    }
    var e = vec3f(0.0);
    for (var k = 0u; k < VOXEL_HEMISPHERE_CONES; k++) {
        let cone = voxelHemisphereCone(n, k);
        let c = voxelConeTrace(vol, volTex, volSampler, p, cone.xyz, VOXEL_HEMISPHERE_TAN, 1.5 * vol.voxelSize, rp.maxDistance, 16u);
        e += cone.w * (c.rgb + c.a * rp.skyScale * skyRadiance(sky, cone.xyz));
    }
    return e;
}
`;

/**
 * The voxel source's functions for the traces (`srcVoxelSize`, `srcHitRadiance`, `srcCone`), over a
 * clipmap (`CLIPMAP_WGSL`'s group 0 bindings 50-57); the reflections add `srcIrradiance`
 * (`CLIPMAP_IRRADIANCE_WGSL`).
 */
const CLIPMAP_SOURCE_WGSL = /* wgsl */`
fn srcVoxelSize(p: vec3f) -> f32 {
    return clipVoxelSize(min(clipLevelAt(p, 0u, 1.0), clipmap.levelCount - 1u));
}
// The light leaving a surface at p (face normal nf) as the voxels hold it: the finest level's
// sample a quarter voxel out of the surface (or in, where nothing is out), by its coverage.
fn srcHitRadiance(p: vec3f, nf: vec3f) -> vec3f {
    let k = clipLevelAt(p, 0u, 1.0);
    if (k >= clipmap.levelCount) { return vec3f(0.0); }
    let size = clipVoxelSize(k);
    var s = clipSample(k, p + nf * (0.25 * size));
    if (s.a < 0.05) { s = clipSample(k, p - nf * (0.25 * size)); }
    return s.rgb / max(s.a, 0.05) * clipmap.radianceScale;
}
fn srcCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return clipConeTrace(origin, dir, n, tanHalf, srcVoxelSize(origin), startDist, maxDist, steps);
}
`;

/**
 * The reflections' `srcIrradiance` over a clipmap (it reads their parameters, `rp`, which the
 * diffuse GI's trace doesn't declare).
 */
const CLIPMAP_IRRADIANCE_WGSL = /* wgsl */`
// The irradiance a surface at p (normal n) receives from the voxels and the sky past them, as
// voxel GI's composite gathers it (its six cones).
fn srcIrradiance(p: vec3f, n: vec3f) -> vec3f {
    let size = srcVoxelSize(p);
    return clipIrradiance(sky, rp.skyScale, p, n, 0.0, size, 1.5 * size, rp.maxDistance, 16u).rgb;
}
`;

/**
 * The alpha test's texture and sampler (group 1 bindings 3 and 4) for `RtReflectionsEffect`'s
 * trace, which a `coveredWgsl` may sample.
 */
const ALPHA_BINDINGS_WGSL = '@group(1) @binding(3) var kansei_rt_alpha_texture : texture_2d<f32>;\n@group(1) @binding(4) var kansei_rt_alpha_sampler : sampler;\n';

/** The default `kansei_rt_covered` of the reflections: the alpha texture's alpha at the uv, at least a half. */
export const RT_DEFAULT_COVERED_WGSL = 'fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {\n    return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5;\n}\n';

/**
 * `RtReflectionsEffect`'s trace over a voxel clipmap (`clipmap`) or volume, with `covered`
 * (`kansei_rt_covered`). Rust: `rt::effect::trace_wgsl(clipmap, covered)`.
 */
export function rtReflectTraceWgsl(covered: string, clipmap: boolean = false): string {
    return rtTracedWgsl(covered, clipmap, rtReflectTrace);
}

/**
 * `RtReflectionsEffect`'s glass pass, prefixed as the trace. Rust: `rt::effect::glass_wgsl(clipmap,
 * covered)`.
 */
export function rtGlassWgsl(covered: string, clipmap: boolean = false): string {
    return rtTracedWgsl(covered, clipmap, rtGlass);
}

function rtTracedWgsl(covered: string, clipmap: boolean, main: string): string {
    const source = clipmap ? `${CLIPMAP_WGSL}${CLIPMAP_SOURCE_WGSL}${CLIPMAP_IRRADIANCE_WGSL}` : `${VOXEL_CONES_WGSL}${voxelIrradiance}${VOLUME_SOURCE_WGSL}`;
    return `${SKY_LIGHTING_WGSL}\n${SPOT_LIGHT_TYPES_WGSL}\n${rtReflectCommon}\n${RT_GRID_WGSL}\n${rtGridBindingsWgsl(1, 0)}\n${ALPHA_BINDINGS_WGSL}${covered}\n${source}\n${rtReflectHit}\n${main}`;
}

/** `RtReflectionsEffect`'s resolve. */
export const RT_REFLECT_RESOLVE_WGSL = `${rtReflectCommon}\n${rtReflectResolve}`;

/**
 * The diffuse GI's voxel source over a clipmap (`CLIPMAP_WGSL`'s group 0 bindings 50-57): the
 * reflections' functions and `srcSurfaceCone`, a cone leaving a surface (`clipConeTrace` lifts its
 * samples off it). Rust: `rt::diffuse::CLIPMAP_SOURCE_WGSL`.
 */
const GI_CLIPMAP_SOURCE_WGSL = `${CLIPMAP_SOURCE_WGSL}// a cone leaving a surface (clipConeTrace lifts its samples off it)
fn srcSurfaceCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return srcCone(origin, dir, n, tanHalf, startDist, maxDist, steps);
}
`;

/**
 * The same over a voxel volume (group 0 bindings 60-62, its anisotropic mips 40-45, after
 * `voxel_irradiance.wgsl`). Rust: `rt::diffuse::VOLUME_SOURCE_WGSL`.
 */
const GI_VOLUME_SOURCE_WGSL = /* wgsl */`
@group(0) @binding(60) var<uniform> vol : VoxelVolume;
@group(0) @binding(61) var volTex : texture_3d<f32>;
@group(0) @binding(62) var volSampler : sampler;
fn srcVoxelSize(p: vec3f) -> f32 {
    return vol.voxelSize;
}
fn srcHitRadiance(p: vec3f, nf: vec3f) -> vec3f {
    var s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p + nf * (0.25 * vol.voxelSize)), 0.0);
    if (s.a < 0.05) { s = textureSampleLevel(volTex, volSampler, voxelUvw(vol, p - nf * (0.25 * vol.voxelSize)), 0.0); }
    return s.rgb / max(s.a, 0.05) * vol.radianceScale;
}
fn srcCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return voxelConeTrace(vol, volTex, volSampler, origin, dir, tanHalf, startDist, maxDist, steps);
}
// a cone leaving a surface: voxel GI's own, through the anisotropic mips, its samples lifted off
// the surface (voxel_irradiance.wgsl); the isotropic mips leak through thin walls once it widens
fn srcSurfaceCone(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, startDist: f32, maxDist: f32, steps: u32) -> vec4f {
    return voxelSurfaceConeTrace(vol, volTex, volSampler, origin, dir, n, tanHalf, startDist, maxDist, steps, 1.0);
}
`;

const RT_GI_PARAMS_WGSL = '@group(0) @binding(20) var<uniform> gp : RtGiParams;\n';

/**
 * `RtDiffuseGiEffect`'s trace over a voxel clipmap (`clipmap`) or volume, with `covered`
 * (`kansei_rt_covered`). Rust: `rt::diffuse::trace_wgsl(clipmap, covered)`.
 */
export function rtGiTraceWgsl(covered: string, clipmap: boolean = false): string {
    const source = clipmap ? `${CLIPMAP_WGSL}${GI_CLIPMAP_SOURCE_WGSL}` : `${VOXEL_CONES_WGSL}${voxelIrradiance}${GI_VOLUME_SOURCE_WGSL}`;
    return `${SKY_LIGHTING_WGSL}\n${COMPUTE_SHADOWS_WGSL}\n${rtGiCommon}\n${RT_GI_PARAMS_WGSL}\n${RT_GRID_WGSL}\n${rtGridBindingsWgsl(1, 0)}\n${ALPHA_BINDINGS_WGSL}${covered}\n${source}\n${rtGiTrace}`;
}

/** `RtDiffuseGiEffect`'s SVGF: its `temporal`, `variance` and `atrous` entry points. Rust: `rt::diffuse::svgf_wgsl`. */
export const RT_GI_SVGF_WGSL = `${rtGiCommon}\n${rtGiSvgf}`;

/** `RtDiffuseGiEffect`'s composite. Rust: `rt::diffuse::composite_wgsl`. */
export const RT_GI_COMPOSITE_WGSL = `${SKY_LIGHTING_WGSL}\n${rtGiCommon}\n${rtGiComposite}`;
