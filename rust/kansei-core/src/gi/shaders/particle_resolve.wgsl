// Accumulators to the volume's mip 0 (gi::ParticleVoxelizer), clearing them for the next frame as
// density-field-copy.wgsl does, plus the analytic boxes (walls, containers, colliders: what the
// p=1476 demo hard-codes as its room). Needs voxel_volume.wgsl and SKY_LIGHTING_WGSL.
//
// Particles: opacity is Beer-Lambert of the splatted density, so a settled body is opaque and
// spray translucent; their radiance is the density-weighted mean of their emission times that
// opacity (a voxel full of glowing particles is as bright as one of them, not their sum).
// Boxes: each covers its share of the voxel (exact box overlap), lit on the face its normal
// gives by the sun (N.L, no shadow) and the sky, plus its emission.
struct ResolveParams {
    toSun          : vec3f,
    extinction     : f32,    // opacity per unit density: 1 - exp(-extinction * density)
    sunIlluminance : vec3f,  // scene units
    boxCount       : u32,
    boxSkyScale    : f32,    // how much of the sky's irradiance reaches the boxes
    _pad0          : f32,
    _pad1          : f32,
    _pad2          : f32,
}

// gi::GiBox
struct GiBox {
    boxMin   : vec3f,
    _pad0    : f32,
    boxMax   : vec3f,
    _pad1    : f32,
    albedo   : vec3f,
    _pad2    : f32,
    emission : vec3f,   // scene radiance
    _pad3    : f32,
    normal   : vec3f,   // the lit face's outward normal
    _pad4    : f32,
}

const DENSITY_FIXED: f32 = 4096.0;
const EMISSION_FIXED: f32 = 1024.0;
const PI: f32 = 3.14159265;

@group(0) @binding(0) var<uniform> vol: VoxelVolume;
@group(0) @binding(1) var<uniform> rp: ResolveParams;
@group(0) @binding(2) var<uniform> sky: SkyLighting;
@group(0) @binding(3) var<storage, read_write> accum: array<u32>;
@group(0) @binding(4) var<storage, read> boxes: array<GiBox>;
@group(0) @binding(5) var radiance: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (any(gid >= vol.dims)) { return; }
    let idx = 4u * voxelLinearIndex(vol, gid);
    let weighted = vec3f(f32(accum[idx]), f32(accum[idx + 1u]), f32(accum[idx + 2u])) / EMISSION_FIXED;
    let density = f32(accum[idx + 3u]) / DENSITY_FIXED;
    accum[idx] = 0u;
    accum[idx + 1u] = 0u;
    accum[idx + 2u] = 0u;
    accum[idx + 3u] = 0u;

    var opacity = 1.0 - exp(-rp.extinction * density);
    var color = weighted / max(density, 1e-6) * opacity;

    let lo = vol.origin + vec3f(gid) * vol.voxelSize;
    let hi = lo + vol.voxelSize;
    for (var b = 0u; b < rp.boxCount; b++) {
        let bx = boxes[b];
        let overlap = max(min(hi, bx.boxMax) - max(lo, bx.boxMin), vec3f(0.0));
        let coverage = overlap.x * overlap.y * overlap.z / (vol.voxelSize * vol.voxelSize * vol.voxelSize);
        if (coverage <= 0.0) { continue; }
        let n = normalize(bx.normal);
        let irradiance = rp.sunIlluminance * max(dot(n, rp.toSun), 0.0) + rp.boxSkyScale * skyIrradiance(sky, n);
        let outgoing = (bx.albedo / PI * irradiance + bx.emission) / vol.radianceScale;
        color += outgoing * coverage;
        opacity = 1.0 - (1.0 - opacity) * (1.0 - coverage);
    }
    textureStore(radiance, gid, vec4f(color, opacity));
}
