// Particles into the volume (gi::ParticleVoxelizer), p=1476's scatter as a compute pass: one
// thread per particle adds its density and its density-weighted emission into four u32
// accumulators per voxel, in fixed point (WebGPU has neither texture nor float atomics). Needs
// voxel_volume.wgsl and particle_emission.wgsl.
//
// The splat is conservative: each particle's weights sum to exactly 1 (in fixed point, to within
// rounding), so the volume's mass does not depend on where a particle sits among the voxels.
// A particle smaller than a voxel is split trilinearly over its 8 neighbours, so crossing a voxel
// boundary moves its weight continuously instead of popping (the flicker the article reports);
// a larger one spreads over its footprint with a smooth kernel, normalised to 1. Positions
// outside the volume deposit on its border voxels, keeping their mass. Fixed point keeps 1/4096
// of density and 1/1024 of stored radiance times density, so a voxel holds up to a million
// particles, or a radiance-times-density of four million, before its sums wrap.
struct SplatParams {
    particleCount      : u32,
    radiusVoxels       : f32,   // the particle's radius in voxels: under 1, trilinear
    densityPerParticle : f32,
    _pad               : f32,
}

const DENSITY_FIXED: f32 = 4096.0;
const EMISSION_FIXED: f32 = 1024.0;

@group(0) @binding(0) var<uniform> vol: VoxelVolume;
@group(0) @binding(1) var<uniform> splat: SplatParams;
@group(0) @binding(2) var<uniform> emission: ParticleEmission;
@group(0) @binding(3) var<storage, read> positions: array<vec4f>;
@group(0) @binding(4) var<storage, read> velocities: array<vec4f>;
// per voxel: emission r, g, b (stored radiance times density), density
@group(0) @binding(5) var<storage, read_write> accum: array<atomic<u32>>;

fn deposit(c: vec3i, w: f32, e: vec3f) {
    let cc = vec3u(clamp(c, vec3i(0), vec3i(vol.dims) - 1));
    let idx = 4u * voxelLinearIndex(vol, cc);
    let density = w * splat.densityPerParticle;
    let weighted = e * density * EMISSION_FIXED;
    if (any(weighted >= vec3f(0.5))) {
        atomicAdd(&accum[idx + 0u], u32(weighted.r + 0.5));
        atomicAdd(&accum[idx + 1u], u32(weighted.g + 0.5));
        atomicAdd(&accum[idx + 2u], u32(weighted.b + 0.5));
    }
    atomicAdd(&accum[idx + 3u], u32(density * DENSITY_FIXED + 0.5));
}

fn kernelWeight(d2: f32, r2: f32) -> f32 {
    let t = max(1.0 - d2 / r2, 0.0);
    return t * t * t;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= splat.particleCount) { return; }
    let e = particleEmission(emission, i, velocities[i].xyz) / vol.radianceScale;
    // voxel space: voxel c spans [c, c + 1)
    let g = (positions[i].xyz - vol.origin) / vol.voxelSize;
    if (splat.radiusVoxels < 1.0) {
        let h = g - 0.5;
        let base = vec3i(floor(h));
        let f = h - floor(h);
        for (var k = 0u; k < 8u; k++) {
            let o = vec3u(k & 1u, (k >> 1u) & 1u, (k >> 2u) & 1u);
            let w3 = select(1.0 - f, f, o == vec3u(1u));
            deposit(base + vec3i(o), w3.x * w3.y * w3.z, e);
        }
        return;
    }
    // the footprint: voxel centres within the radius (at most 7^3 voxels)
    let r = min(splat.radiusVoxels, 3.0);
    let r2 = r * r;
    let lo = vec3i(floor(g - r));
    let hi = vec3i(floor(g + r));
    var total = 0.0;
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let d = vec3f(f32(x), f32(y), f32(z)) + 0.5 - g;
                total += kernelWeight(dot(d, d), r2);
            }
        }
    }
    if (total <= 0.0) { return; }
    for (var z = lo.z; z <= hi.z; z++) {
        for (var y = lo.y; y <= hi.y; y++) {
            for (var x = lo.x; x <= hi.x; x++) {
                let d = vec3f(f32(x), f32(y), f32(z)) + 0.5 - g;
                let w = kernelWeight(dot(d, d), r2) / total;
                if (w > 0.0) { deposit(vec3i(x, y, z), w, e); }
            }
        }
    }
}
