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
import { SKY_LIGHTING_WGSL, VOXEL_CONES_WGSL } from '../gi/GiWGSL';

/**
 * Ray tracing through an `RtGrid` from a compute pass: `KanseiRtGrid`, `KanseiRtHit` and
 * `kansei_rt_trace(origin, dir, tMin, tMax, flags)`, the closest hit of a ray among the grid's
 * triangles (`KANSEI_RT_ANY_HIT`: any hit, for shadows; `KANSEI_RT_SOLID`: no alpha test), with
 * `kansei_rt_exit` (where a ray leaves the box), `kansei_rt_contains`, and a hit triangle's
 * `kansei_rt_albedo`, `kansei_rt_uv`, `kansei_rt_source` and `kansei_rt_record`. Declare the
 * buffers with `rtGridBindingsWgsl(group, first)` (bind `RtGrid.bindGroupEntries`), and define
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

/** The voxel source's functions for the reflections' trace over a voxel volume (group 0 bindings 6-8). */
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
`;

/**
 * The alpha test's texture and sampler (group 1 bindings 3 and 4) for `RtReflectionsEffect`'s
 * trace, which a `coveredWgsl` may sample.
 */
const ALPHA_BINDINGS_WGSL = '@group(1) @binding(3) var kansei_rt_alpha_texture : texture_2d<f32>;\n@group(1) @binding(4) var kansei_rt_alpha_sampler : sampler;\n';

/** The default `kansei_rt_covered` of the reflections: the alpha texture's alpha at the uv, at least a half. */
export const RT_DEFAULT_COVERED_WGSL = 'fn kansei_rt_covered(layer: u32, uv: vec2f) -> bool {\n    return textureSampleLevel(kansei_rt_alpha_texture, kansei_rt_alpha_sampler, uv, 0.0).a >= 0.5;\n}\n';

/**
 * `RtReflectionsEffect`'s trace over a voxel volume, with `covered` (`kansei_rt_covered`). Rust:
 * `rt::effect::trace_wgsl(false, covered)` (the clipmap source comes with G-4).
 */
export function rtReflectTraceWgsl(covered: string): string {
    return `${SKY_LIGHTING_WGSL}\n${rtReflectCommon}\n${RT_GRID_WGSL}\n${rtGridBindingsWgsl(1, 0)}\n${ALPHA_BINDINGS_WGSL}${covered}\n${VOXEL_CONES_WGSL}${VOLUME_SOURCE_WGSL}\n${rtReflectTrace}`;
}

/** `RtReflectionsEffect`'s resolve. */
export const RT_REFLECT_RESOLVE_WGSL = `${rtReflectCommon}\n${rtReflectResolve}`;
