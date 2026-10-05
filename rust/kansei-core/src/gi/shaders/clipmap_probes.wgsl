// Irradiance probes of a voxel clipmap (gi::ClipmapProbes), for whoever reads them: materials,
// the volumetric fog, VoxelGIEffect's composite. A probe sits on each point of a lattice
// `spacing * 2^level` apart (level 0's finest); each level keeps a window of `dims` of them round
// the camera, stored toroidally (lattice point c in slot c mod dims), as the clipmap keeps its
// voxels. Per probe, 4 vec4f in `kansei_clip_probes` (from `levelBase(level) + slot`, 4 apart):
// order-1 spherical harmonics, in the real orthonormal basis (0.282095; 0.488603 x, y, z), of the
// radiance arriving at it (rgb) and of the share of each direction that reaches the sky past the
// clipmap (a): [0] the constant band, [1..3] the x, y and z bands. A probe never traced, or inside
// a surface, has a negative a in [0] and is left out.
//
// Declare the buffers with ClipmapProbes::bindings_wgsl(group, first) and bind them with
// ClipmapProbes::bind_group_entries; kansei_clipmap_light(p, n) is what a surface at p facing n
// receives.

const CLIP_PROBE_MAX_LEVELS : u32 = 6u;

struct ClipProbeLevel {
    origin : vec3i,   // the window's first lattice point
    valid  : u32,     // 1 once its probes are placed
}

struct ClipProbeGrid {
    dims       : vec3u,   // probes of each level's window
    levelCount : u32,
    spacing    : f32,     // metres between level 0's probes
    normalBias : f32,     // spacings a lookup moves off its surface along the normal
    _pad0      : f32,
    _pad1      : f32,
    levels     : array<ClipProbeLevel, 6>,
}

// The first slot of level k's probes.
fn kanseiClipProbeLevelBase(grid: ClipProbeGrid, k: u32) -> u32 {
    return k * grid.dims.x * grid.dims.y * grid.dims.z;
}

// The slot of level k's lattice point c (toroidal).
fn kanseiClipProbeSlot(grid: ClipProbeGrid, k: u32, c: vec3i) -> u32 {
    let d = vec3i(grid.dims);
    let w = ((c % d) + d) % d;
    return kanseiClipProbeLevelBase(grid, k) + u32((w.z * d.y + w.y) * d.x + w.x);
}

// What the eight probes of level k round q weigh in, by trilinear weight, by facing n (DDGI's
// smooth back face: probes behind the surface count less) and only where traced: the weighted
// sum of their 4 vec4f (in [0..3]) and the weights' sum (w of [4]).
struct ClipProbeBlend {
    sh : array<vec4f, 4>,
    weight : f32,
}

fn kanseiClipProbeBlend(grid: ClipProbeGrid, k: u32, q: vec3f, n: vec3f) -> ClipProbeBlend {
    var out: ClipProbeBlend;
    let s = grid.spacing * exp2(f32(k));
    let f = q / s;
    let base = floor(f);
    let t = f - base;
    for (var i = 0u; i < 8u; i++) {
        let o = vec3f(f32(i & 1u), f32((i >> 1u) & 1u), f32((i >> 2u) & 1u));
        let c = vec3i(base + o);
        let slot = kanseiClipProbeSlot(grid, k, c);
        let d0 = kansei_clip_probes[4u * slot];
        if (d0.w < 0.0) { continue; }
        let tri = mix(1.0 - t, t, o);
        let toProbe = (vec3f(c) * s) - q;
        let facing = (dot(toProbe, n) * inverseSqrt(max(dot(toProbe, toProbe), 1e-8)) + 1.0) * 0.5;
        let w = tri.x * tri.y * tri.z * (facing * facing + 0.2);
        out.sh[0] += w * d0;
        out.sh[1] += w * kansei_clip_probes[4u * slot + 1u];
        out.sh[2] += w * kansei_clip_probes[4u * slot + 2u];
        out.sh[3] += w * kansei_clip_probes[4u * slot + 3u];
        out.weight += w;
    }
    return out;
}

// Level k's window holds the lattice cell round q, `margin` cells in from its sides.
fn kanseiClipProbeHolds(grid: ClipProbeGrid, k: u32, q: vec3f, margin: f32) -> f32 {
    let level = grid.levels[k];
    if (level.valid == 0u) { return -1.0; }
    let f = q / (grid.spacing * exp2(f32(k))) - vec3f(level.origin);
    // probes from origin to origin + dims - 1: the cell round q from floor(f) to floor(f) + 1
    let room = min(f, vec3f(grid.dims) - 1.0 - f);
    return min(room.x, min(room.y, room.z)) - margin;
}

// The irradiance (rgb, scene units) a surface at p facing n receives from the probes, and the
// cosine-weighted share of its hemisphere that sees the sky past the clipmap (a), from the finest
// level whose window holds it, blended into the next one toward the window's edge. a is -1 where
// no level holds p (or none of its probes is traced yet): use the material's own sky light there.
fn kansei_clipmap_light(p: vec3f, n: vec3f) -> vec4f {
    let grid = kansei_clip_probe_grid;
    for (var k = 0u; k < grid.levelCount; k++) {
        let q = p + n * (grid.normalBias * grid.spacing * exp2(f32(k)));
        let room = kanseiClipProbeHolds(grid, k, q, 0.0);
        if (room < 0.0) { continue; }
        let b = kanseiClipProbeBlend(grid, k, q, n);
        if (b.weight <= 1e-5) { continue; }
        var sh = array<vec4f, 4>(b.sh[0] / b.weight, b.sh[1] / b.weight, b.sh[2] / b.weight, b.sh[3] / b.weight);
        // toward the window's edge, into the next level
        let edge = smoothstep(0.0, 3.0, room);
        if (edge < 1.0 && k + 1u < grid.levelCount) {
            let q1 = p + n * (grid.normalBias * grid.spacing * exp2(f32(k + 1u)));
            if (kanseiClipProbeHolds(grid, k + 1u, q1, 0.0) >= 0.0) {
                let c = kanseiClipProbeBlend(grid, k + 1u, q1, n);
                if (c.weight > 1e-5) {
                    for (var i = 0u; i < 4u; i++) {
                        sh[i] = mix(c.sh[i] / c.weight, sh[i], edge);
                    }
                }
            }
        }
        // E(n) = pi Y0 c0 + 2 pi / 3 * 0.488603 (c1 . n); the visibility likewise, over pi
        let band1 = sh[1] * n.x + sh[2] * n.y + sh[3] * n.z;
        let e = 3.14159265 * 0.282095 * sh[0] + 2.0943951 * 0.488603 * band1;
        return vec4f(max(e.rgb, vec3f(0.0)), clamp(e.a / 3.14159265, 0.0, 1.0));
    }
    return vec4f(0.0, 0.0, 0.0, -1.0);
}

// The light a medium at p (fog, mist) scatters toward the camera per unit scattering coefficient,
// seen along viewDir (camera to p), with a Henyey-Greenstein phase of anisotropy g: the probes'
// radiance round p, convolved with the phase (its bands times g^l), as SKY_LIGHTING_WGSL's
// skyInscatter does with the sky's: the sky past the clipmap and what the scene round p bounces.
// a is -1 where no level holds p.
fn kansei_clipmap_inscatter(p: vec3f, viewDir: vec3f, g: f32) -> vec4f {
    let grid = kansei_clip_probe_grid;
    for (var k = 0u; k < grid.levelCount; k++) {
        if (kanseiClipProbeHolds(grid, k, p, 0.0) < 0.0) { continue; }
        // (no normal: the probes round p by their trilinear weights alone)
        let b = kanseiClipProbeBlend(grid, k, p, vec3f(0.0));
        if (b.weight <= 1e-5) { continue; }
        let band1 = b.sh[1] * viewDir.x + b.sh[2] * viewDir.y + b.sh[3] * viewDir.z;
        let l = (0.282095 * b.sh[0] + g * 0.488603 * band1) / b.weight;
        return vec4f(max(l.rgb, vec3f(0.0)), 1.0);
    }
    return vec4f(0.0, 0.0, 0.0, -1.0);
}

// The cosine-weighted share of the sky a surface at p facing n sees past the trees and terrain in
// the clipmap (1 in the open): to dim a material's own sky light by (as SKY_OCCLUSION_WGSL's
// skyVisibility does); 1 where no probe holds p.
fn kansei_clipmap_sky_visibility(p: vec3f, n: vec3f) -> f32 {
    let light = kansei_clipmap_light(p, n);
    return select(light.a, 1.0, light.a < 0.0);
}
