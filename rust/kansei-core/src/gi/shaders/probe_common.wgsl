// Irradiance probes traced in the scene's distance field (gi::SdfProbes), shared by their update
// and their readers. The probes sit on a world lattice (`ProbeGrid`): the grid is a box of it,
// moved by whole cells to follow the camera, and each probe is stored at its lattice cell modulo
// the grid's size, so a probe that stays in the grid keeps its slot and its history.
//
// Per probe, in storage buffers:
// - `kansei_probe_sh`, 9 vec4f: the irradiance it receives as order-2 spherical harmonics (rgb;
//   the radiance it sees convolved with the cosine lobe, so E(n) = sum c_i Y_i(n)), in
//   SKY_LIGHTING_WGSL's basis;
// - `kansei_probe_state`, 2 vec4f: [0] the probe's offset from its lattice point (moved out of
//   the surfaces) and the smoothed share of its rays that met the back of a surface (w, 1 inside
//   geometry); [1] its lattice cell (xyz, bitcast i32) and the frames it has been updated (w,
//   bitcast u32);
// - `kansei_probe_depth`, 64 vec2f: the mean distance to the surfaces it sees and its square, on
//   an 8x8 octahedral map of directions, for Chebyshev visibility (DDGI, Majercik et al. 2019).

struct ProbeGrid {
    origin        : vec3f,   // world position of the grid's first probe (before its offset)
    spacing       : f32,     // metres between probes
    dims          : vec3u,   // probes per axis
    probeCount    : u32,
    base          : vec3i,   // lattice cell of the grid's first probe
    normalBias    : f32,     // metres a lookup moves off its surface along the normal
    backfaceLimit : f32,     // a probe whose rays meet more back faces than this share is off
    maxDepth      : f32,     // the depth moments' cap, metres
    visibility    : u32,     // 1: weight probes by their depth moments (Chebyshev)
    _pad          : u32,
}

const PROBE_DEPTH_SIDE : u32 = 8u;
const PROBE_DEPTH_TEXELS : u32 = 64u;
const PROBE_SH_WORDS : u32 = 9u;

// The storage slot of a lattice cell: its position modulo the grid.
fn probeSlot(grid: ProbeGrid, cell: vec3i) -> u32 {
    let d = vec3i(grid.dims);
    let w = ((cell % d) + d) % d;
    return u32((w.z * d.y + w.y) * d.x + w.x);
}

// The grid position (0..dims) of the n-th probe of the grid.
fn probeLocal(grid: ProbeGrid, n: u32) -> vec3u {
    return vec3u(n % grid.dims.x, (n / grid.dims.x) % grid.dims.y, n / (grid.dims.x * grid.dims.y));
}

fn probeSignNotZero(v: vec2f) -> vec2f {
    return select(vec2f(-1.0), vec2f(1.0), v >= vec2f(0.0));
}

// A unit direction on the octahedral square, [0, 1]^2.
fn probeOctEncode(d: vec3f) -> vec2f {
    var p = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
    if (d.y < 0.0) { p = (1.0 - abs(p.yx)) * probeSignNotZero(p); }
    return p * 0.5 + 0.5;
}

fn probeOctDecode(uv: vec2f) -> vec3f {
    let f = uv * 2.0 - 1.0;
    var d = vec3f(f.x, 1.0 - abs(f.x) - abs(f.y), f.y);
    if (d.y < 0.0) {
        let xz = (1.0 - abs(d.zx)) * probeSignNotZero(d.xz);
        d = vec3f(xz.x, d.y, xz.y);
    }
    return normalize(d);
}

// The real orthonormal SH basis of SKY_LIGHTING_WGSL, as weights for the coefficients 0..8.
fn probeShBasis(d: vec3f) -> array<f32, 9> {
    return array<f32, 9>(
        0.282095,
        0.488603 * d.y, 0.488603 * d.z, 0.488603 * d.x,
        1.092548 * d.x * d.y, 1.092548 * d.y * d.z, 0.315392 * (3.0 * d.z * d.z - 1.0),
        1.092548 * d.x * d.z, 0.546274 * (d.x * d.x - d.y * d.y));
}
