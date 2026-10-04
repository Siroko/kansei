// Irradiance probes of a voxel clipmap, update (gi::ClipmapProbes): one thread per probe traces
// 16 cones through the clipmap (clipmap.wgsl) from its lattice point, turned by `rotation` (a new
// random turn each frame, so the history integrates a finer set), and projects what each brings
// (the radiance gathered plus the sky past the clipmap times what is left of it; and that share)
// onto order-1 spherical harmonics, blended into its history. Needs clipmap_probes.wgsl (with
// `kansei_clip_probes` declared read_write), clipmap.wgsl and SKY_LIGHTING_WGSL.
//
// Modes: 0, `count` probes of level `level` from slot `first` of its window (round robin, into the
// history); 1, the probes of the lattice box `lo`..`lo + size` (a slab the window moved into:
// fresh, no history); 2, `count` slots from `first` marked never traced (a window placed anew).

struct ClipProbeUpdate {
    rotation   : mat4x4f,
    level      : u32,
    mode       : u32,
    first      : u32,
    count      : u32,
    lo         : vec3i,
    hysteresis : f32,     // weight of the history
    size       : vec3u,
    skyScale   : f32,
    maxSteps   : u32,
    voxelLevel : u32,     // the clipmap level whose voxels are half the probes' spacing
    tanHalf    : f32,     // of the cones
    levelBias  : f32,     // levels finer than the cones' width they read (clipConeTraceNear)
    frame      : u32,
    _pad0      : u32,
    _pad1      : u32,
    _pad2      : u32,
}

@group(0) @binding(0) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;
@group(0) @binding(1) var<storage, read_write> kansei_clip_probes : array<vec4f>;
@group(0) @binding(2) var<uniform> up : ClipProbeUpdate;
@group(0) @binding(3) var<uniform> sky : SkyLighting;

const CLIP_PROBE_CONES : u32 = 16u;
const NEVER_TRACED : f32 = -2.0;
const INSIDE : f32 = -1.0;

// Point i of n on the unit sphere, evenly spread (spherical Fibonacci).
fn sphericalFibonacci(i: u32, n: u32) -> vec3f {
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = f32(i) * 2.39996323;
    return vec3f(r * cos(phi), r * sin(phi), z);
}

fn store(slot: u32, sh: array<vec4f, 4>) {
    for (var b = 0u; b < 4u; b++) {
        kansei_clip_probes[4u * slot + b] = sh[b];
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let i = gid.x;
    let grid = kansei_clip_probe_grid;
    let per_level = grid.dims.x * grid.dims.y * grid.dims.z;
    if (up.mode == 2u) {
        if (i >= up.count) { return; }
        store(kanseiClipProbeLevelBase(grid, up.level) + (up.first + i) % per_level, array<vec4f, 4>(vec4f(0.0, 0.0, 0.0, NEVER_TRACED), vec4f(0.0), vec4f(0.0), vec4f(0.0)));
        return;
    }
    let k = up.level;
    let level = grid.levels[k];
    let d = vec3i(grid.dims);
    var c: vec3i;
    if (up.mode == 0u) {
        if (i >= up.count) { return; }
        // the lattice point the slot holds: congruent to it, inside the window
        let local = (up.first + i) % per_level;
        let t = vec3i(i32(local % grid.dims.x), i32((local / grid.dims.x) % grid.dims.y), i32(local / (grid.dims.x * grid.dims.y)));
        c = level.origin + ((((t - level.origin) % d) + d) % d);
    } else {
        let s = up.size;
        if (i >= s.x * s.y * s.z) { return; }
        c = up.lo + vec3i(i32(i % s.x), i32((i / s.x) % s.y), i32(i / (s.x * s.y)));
    }
    let slot = kanseiClipProbeSlot(grid, k, c);
    let spacing = grid.spacing * exp2(f32(k));
    let p = vec3f(c) * spacing;
    let voxel = clipVoxelSize(up.voxelLevel);
    // a probe inside a surface sees nothing of the scene's light: left out
    let at = clipLevelAt(p, up.voxelLevel, 0.5);
    if (at < clipmap.levelCount && clipSample(at, p).a > 0.9) {
        store(slot, array<vec4f, 4>(vec4f(0.0, 0.0, 0.0, INSIDE), vec4f(0.0), vec4f(0.0), vec4f(0.0)));
        return;
    }
    var sh = array<vec4f, 4>(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    let w = 4.0 * 3.14159265 / f32(CLIP_PROBE_CONES);
    for (var j = 0u; j < CLIP_PROBE_CONES; j++) {
        let dir = normalize((up.rotation * vec4f(sphericalFibonacci(j, CLIP_PROBE_CONES), 0.0)).xyz);
        // a jittered start, so the sparse samples land elsewhere each update
        let jitter = giHash01(slot * 16u + j + up.frame * 7919u);
        let cone = clipConeTraceNear(p, dir, vec3f(0.0), up.tanHalf, voxel, (0.5 + jitter) * voxel, 1e4, up.maxSteps, up.levelBias);
        let light = vec4f(cone.rgb + cone.a * up.skyScale * skyRadiance(sky, dir), cone.a) * w;
        sh[0] += light * 0.282095;
        sh[1] += light * (0.488603 * dir.x);
        sh[2] += light * (0.488603 * dir.y);
        sh[3] += light * (0.488603 * dir.z);
    }
    let old = kansei_clip_probes[4u * slot];
    // a fresh probe (never traced, inside a surface before, or in a new slab) takes its trace
    let h = select(up.hysteresis, 0.0, old.w < 0.0 || up.mode == 1u);
    for (var b = 0u; b < 4u; b++) {
        sh[b] = mix(sh[b], select(kansei_clip_probes[4u * slot + b], old, b == 0u), h);
    }
    store(slot, sh);
}
