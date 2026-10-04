// The probes' update (gi::SdfProbes): one workgroup per probe, one ray per invocation.
// 1. The probe is moved off the surfaces near its lattice point, up the distance field's gradient.
// 2. Its 64 rays (a spherical Fibonacci set, turned at random each frame) sphere-trace the scene's
//    distance field. A hit reads the lit volume there, mip 0 composited over the anisotropic chain
//    the ray faces (the side of the surface it meets), so no material or shadow lookup is needed:
//    the injection did both. A miss leaves the volume and takes the sky.
// 3. The rays' radiance is projected onto order-2 SH and convolved with the cosine lobe (the
//    probe's irradiance), their distances onto an 8x8 octahedral map of depth moments, and both
//    blend into the probe's history (`hysteresis`), or replace it when the probe is new to its slot.
// 4. The share of rays that met the back of a surface (by the voxels' normals) is smoothed into the
//    probe's state: readers skip a probe above `backfaceLimit` (inside geometry).
// Needs voxel_volume.wgsl, voxel_cones.wgsl, voxel_irradiance.wgsl (the anisotropic chains at
// 40-45, the distance field at 46), sky_lighting.wgsl and probe_common.wgsl.

struct ProbeUpdate {
    rotation        : mat4x4f,   // this frame's turn of the rays (upper 3x3)
    firstProbe      : u32,       // the grid's probe the first workgroup updates
    hysteresis      : f32,       // of the irradiance
    depthHysteresis : f32,       // of the depth moments
    skyScale        : f32,
    maxSteps        : u32,       // per ray
    minClearance    : f32,       // metres a probe is moved off the surfaces
    depthSharpness  : f32,       // exponent of the rays' weight on a depth texel
    resetAll        : u32,       // 1: every probe starts over
    hasDynamic      : u32,
    _pad0           : u32,
    _pad1           : u32,
    _pad2           : u32,
}

@group(0) @binding(0) var<uniform> vol : VoxelVolume;
@group(0) @binding(1) var<uniform> grid : ProbeGrid;
@group(0) @binding(2) var<uniform> pu : ProbeUpdate;
@group(0) @binding(3) var radiance : texture_3d<f32>;
@group(0) @binding(4) var linearClamp : sampler;
@group(0) @binding(5) var<uniform> sky : SkyLighting;
@group(0) @binding(6) var<storage, read> staticSurfaces : array<u32>;
@group(0) @binding(7) var<storage, read> dynamicSurfaces : array<u32>;
@group(0) @binding(8) var<storage, read_write> probeSh : array<vec4f>;
@group(0) @binding(9) var<storage, read_write> probeState : array<vec4f>;
@group(0) @binding(10) var<storage, read_write> probeDepth : array<vec2f>;

const PROBE_RAYS : u32 = 64u;
const PROBE_PI : f32 = 3.14159265;

var<workgroup> wgOrigin : vec4f;     // xyz the probe, w 1 when it starts over
var<workgroup> wgOffset : vec4f;     // xyz its offset, w 1 when it lies inside geometry
var<workgroup> wgSlot : u32;
var<workgroup> wgCell : vec3i;
var<workgroup> wgRadiance : array<vec4f, PROBE_RAYS>;   // rgb, a the distance (capped)
var<workgroup> wgDir : array<vec4f, PROBE_RAYS>;        // xyz, w 1 at a back face

fn fieldAt(p: vec3f) -> f32 {
    let uvw = voxelUvw(vol, p);
    if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { return 1e4; }
    return textureSampleLevel(sdfField, linearClamp, uvw, 0.0).r;
}

// The field at `p` with the volume's edge clamped, for moving probes (outside it there are no
// surfaces to move off).
fn fieldClamped(p: vec3f) -> f32 {
    return textureSampleLevel(sdfField, linearClamp, clamp(voxelUvw(vol, p), vec3f(0.0), vec3f(1.0)), 0.0).r;
}

fn fieldGradient(p: vec3f) -> vec3f {
    let h = vol.voxelSize;
    return vec3f(
        fieldClamped(p + vec3f(h, 0.0, 0.0)) - fieldClamped(p - vec3f(h, 0.0, 0.0)),
        fieldClamped(p + vec3f(0.0, h, 0.0)) - fieldClamped(p - vec3f(0.0, h, 0.0)),
        fieldClamped(p + vec3f(0.0, 0.0, h)) - fieldClamped(p - vec3f(0.0, 0.0, h)));
}

fn fibonacciDir(i: u32, n: u32) -> vec3f {
    let golden = 2.39996323;
    let z = 1.0 - (2.0 * f32(i) + 1.0) / f32(n);
    let r = sqrt(max(1.0 - z * z, 0.0));
    let phi = f32(i) * golden;
    return vec3f(r * cos(phi), z, r * sin(phi));
}

// The surface a ray reached at `q` faces away from it: it travels along the voxel's outward normal.
// Voxels lit from both sides (sheets thinner than a voxel) have no back.
fn hitBackface(q: vec3f, d: vec3f) -> bool {
    let c = vec3i(floor((q + d * vol.voxelSize - vol.origin) / vol.voxelSize));
    if (any(c < vec3i(0)) || any(c >= vec3i(vol.dims))) { return false; }
    let idx = 4u * voxelLinearIndex(vol, vec3u(c)) + 1u;
    var word = staticSurfaces[idx];
    if (pu.hasDynamic != 0u && (dynamicSurfaces[idx] >> 24u) != 0u) { word = dynamicSurfaces[idx]; }
    if ((word >> 24u) == 0u) { return false; }
    let n = vec3f(f32(word & 255u), f32((word >> 8u) & 255u), f32((word >> 16u) & 255u)) / 255.0 * 2.0 - 1.0;
    let len = length(n);
    return len >= 0.35 && dot(n / len, d) > 0.1;
}

// The light leaving the surface a ray meets at `q`: mip 0 there, over the anisotropic chain the
// ray faces behind it, each undone of its coverage.
fn hitRadiance(q: vec3f, d: vec3f) -> vec3f {
    let uvw = voxelUvw(vol, q + d * (0.5 * vol.voxelSize));
    let s0 = textureSampleLevel(radiance, linearClamp, uvw, 0.0);
    let s1 = voxelAnisoSample(linearClamp, uvw, d, 0.0);
    let rgb = s0.rgb + (1.0 - s0.a) * s1.rgb;
    let a = s0.a + (1.0 - s0.a) * s1.a;
    return select(vec3f(0.0), rgb / a, a > 1e-3) * vol.radianceScale;
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_index) lid: u32) {
    if (lid == 0u) {
        let n = (pu.firstProbe + wg.x) % grid.probeCount;
        let local = probeLocal(grid, n);
        let cell = grid.base + vec3i(local);
        let slot = probeSlot(grid, cell);
        let stored = probeState[2u * slot + 1u];
        let fresh = pu.resetAll != 0u || any(bitcast<vec3i>(stored.xyz) != cell) || bitcast<u32>(stored.w) == 0u;
        // off the surfaces: up the field's gradient, at most 0.45 spacing from the lattice point
        let anchor = grid.origin + vec3f(local) * grid.spacing;
        var offset = vec3f(0.0);
        for (var i = 0u; i < 4u; i++) {
            let p = anchor + offset;
            let d = fieldClamped(p);
            if (d >= pu.minClearance) { break; }
            let g = fieldGradient(p);
            let len = length(g);
            if (len < 1e-6) { break; }
            offset = clamp(offset + g / len * (pu.minClearance - d), vec3f(-0.45 * grid.spacing), vec3f(0.45 * grid.spacing));
        }
        let inside = fieldAt(anchor + offset) < 0.5 * vol.voxelSize;
        wgOrigin = vec4f(anchor + offset, select(0.0, 1.0, fresh));
        wgOffset = vec4f(offset, select(0.0, 1.0, inside));
        wgSlot = slot;
        wgCell = cell;
    }
    workgroupBarrier();
    let origin = wgOrigin.xyz;
    let fresh = wgOrigin.w > 0.5;
    let slot = wgSlot;

    // this invocation's ray
    let d = normalize((pu.rotation * vec4f(fibonacciDir(lid, PROBE_RAYS), 0.0)).xyz);
    var t = 0.0;
    var hit = false;
    var outside = false;
    for (var i = 0u; i < pu.maxSteps; i++) {
        let q = origin + d * t;
        let uvw = voxelUvw(vol, q);
        if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { outside = true; break; }
        let s = textureSampleLevel(sdfField, linearClamp, uvw, 0.0).r;
        if (s < 0.5 * vol.voxelSize) { hit = true; break; }
        t += s;
    }
    var light: vec3f;
    var back = false;
    if (outside) {
        light = pu.skyScale * skyRadiance(sky, d);
        t = grid.maxDepth;
    } else {
        // a hit, or a ray that ran out of steps creeping along a surface
        let q = origin + d * t;
        light = hitRadiance(q, d);
        back = hit && hitBackface(q, d);
    }
    wgRadiance[lid] = vec4f(light, min(t, grid.maxDepth));
    wgDir[lid] = vec4f(d, select(0.0, 1.0, back));
    workgroupBarrier();

    // the irradiance: coefficient `lid` of the radiance's SH, times the cosine lobe's band factor
    if (lid < PROBE_SH_WORDS) {
        var c = vec3f(0.0);
        for (var r = 0u; r < PROBE_RAYS; r++) {
            let y = probeShBasis(wgDir[r].xyz);
            c += wgRadiance[r].rgb * y[lid];
        }
        let band = select(select(PROBE_PI / 4.0, 2.0 * PROBE_PI / 3.0, lid < 4u), PROBE_PI, lid == 0u);
        c *= band * (4.0 * PROBE_PI / f32(PROBE_RAYS));
        let idx = slot * PROBE_SH_WORDS + lid;
        let old = probeSh[idx].rgb;
        probeSh[idx] = vec4f(select(mix(c, old, pu.hysteresis), c, fresh), 0.0);
    }
    // the state: offset, the smoothed share of back faces, the cell and the frames
    if (lid == PROBE_SH_WORDS) {
        var backs = 0.0;
        for (var r = 0u; r < PROBE_RAYS; r++) { backs += wgDir[r].w; }
        let share = backs / f32(PROBE_RAYS);
        let old = probeState[2u * slot];
        var smoothed = select(mix(share, old.w, pu.hysteresis), share, fresh || old.w > 1.0);
        let inside = wgOffset.w > 0.5;
        let frames = select(bitcast<u32>(probeState[2u * slot + 1u].w) + 1u, 1u, fresh);
        // inside geometry it reads 2 (off for any limit), and starts over from the share once out
        probeState[2u * slot] = vec4f(wgOffset.xyz, select(smoothed, 2.0, inside));
        probeState[2u * slot + 1u] = vec4f(bitcast<vec3f>(wgCell), bitcast<f32>(frames));
    }
    // the depth moments: texel `lid` of the octahedral map, from the rays near its direction
    {
        let texel = vec2f(f32(lid % PROBE_DEPTH_SIDE), f32(lid / PROBE_DEPTH_SIDE));
        let dir = probeOctDecode((texel + 0.5) / f32(PROBE_DEPTH_SIDE));
        var m = vec2f(0.0);
        var w = 0.0;
        for (var r = 0u; r < PROBE_RAYS; r++) {
            let k = pow(max(dot(dir, wgDir[r].xyz), 0.0), pu.depthSharpness);
            let dist = wgRadiance[r].a;
            m += k * vec2f(dist, dist * dist);
            w += k;
        }
        let idx = slot * PROBE_DEPTH_TEXELS + lid;
        let old = probeDepth[idx];
        if (w > 1e-4) {
            m /= w;
            probeDepth[idx] = select(mix(m, old, pu.depthHysteresis), m, fresh);
        } else if (fresh) {
            probeDepth[idx] = vec2f(grid.maxDepth, grid.maxDepth * grid.maxDepth);
        }
    }
}
