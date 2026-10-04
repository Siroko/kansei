// The irradiance a surface receives through a VoxelVolume: six 60-degree cones over the
// hemisphere around its normal, one along it and five tilted 45 degrees from it, each weighted by
// the cosine of its tilt (their share of the cosine-weighted hemisphere, the cones being equal),
// after Crassin et al. 2011. The cones read the volume's anisotropic mips (bindings 40-45,
// `VoxelVolume::anisotropic_views`). Light past the volume is the sky's. Needs voxel_volume.wgsl
// and SKY_LIGHTING_WGSL.

const VOXEL_GI_PI : f32 = 3.14159265;

// How far a surface cone's samples are lifted off the surface, in footprints: a trilinear sample
// reaches a voxel away, and the surface's own voxel may sit half of one above it, so 1.5 keeps
// the cone off the voxel it starts from (whose far face, in the anisotropic chains, is the
// surface's back: unlit, and opaque).
const LIFT : f32 = 1.5;

// the scene's distance field (gi::JumpFloodSdf, metres; a 1-texel stand-in without one), for the
// injection's soft shadows
@group(0) @binding(46) var sdfField : texture_3d<f32>;

// the anisotropic chains, by the direction a cone travels; level 0 is the volume's mip 1
@group(0) @binding(40) var anisoPosX : texture_3d<f32>;
@group(0) @binding(41) var anisoPosY : texture_3d<f32>;
@group(0) @binding(42) var anisoPosZ : texture_3d<f32>;
@group(0) @binding(43) var anisoNegX : texture_3d<f32>;
@group(0) @binding(44) var anisoNegY : texture_3d<f32>;
@group(0) @binding(45) var anisoNegZ : texture_3d<f32>;

// The anisotropic chains at `level` for a cone travelling along `dir`: the three it faces,
// weighted by the square of each axis' share of the direction (they sum to 1). Their opacities
// combine as optical depths (the transmittances multiply, each to the power of its weight), so
// a wall that one chain sees face on stays opaque for a cone crossing it at a slant, where the
// chain seeing it edge on is mostly empty; colours combine as their weighted mean at that
// opacity.
fn voxelAnisoSample(linearClamp: sampler, uvw: vec3f, dir: vec3f, level: f32) -> vec4f {
    let w = dir * dir;
    var s: array<vec4f, 3>;
    if (dir.x >= 0.0) { s[0] = textureSampleLevel(anisoPosX, linearClamp, uvw, level); }
    else { s[0] = textureSampleLevel(anisoNegX, linearClamp, uvw, level); }
    if (dir.y >= 0.0) { s[1] = textureSampleLevel(anisoPosY, linearClamp, uvw, level); }
    else { s[1] = textureSampleLevel(anisoNegY, linearClamp, uvw, level); }
    if (dir.z >= 0.0) { s[2] = textureSampleLevel(anisoPosZ, linearClamp, uvw, level); }
    else { s[2] = textureSampleLevel(anisoNegZ, linearClamp, uvw, level); }
    var rgb = vec3f(0.0);
    var meanOpacity = 0.0;
    var depth = 0.0;
    for (var i = 0u; i < 3u; i++) {
        rgb += w[i] * s[i].rgb;
        meanOpacity += w[i] * s[i].a;
        depth -= w[i] * log(max(1.0 - s[i].a, 1e-4));
    }
    let opacity = 1.0 - exp(-depth);
    return vec4f(select(rgb, rgb / meanOpacity * opacity, meanOpacity > 1e-6), opacity);
}

// voxelConeTrace (voxel_cones.wgsl) for a cone that leaves a surface (normal `n`), through the
// anisotropic mips: each sample is lifted off the surface until its footprint (a voxel of the
// mip it reads, about the cone's width there) clears it. Without that, a cone tilted toward the
// surface reads the surface's own voxels at coarse mips: it is occluded by where it starts and
// returns that surface's light, so a wall would light itself and hide the walls beside it.
// `anisoOffset` is the level of `vol` the anisotropic chains start at: 1 for the volume itself
// (below it, the isotropic mip 0 in `radiance`), 0 for a `vol` describing its mips 1.. alone.
fn voxelSurfaceConeTrace(
    vol: VoxelVolume,
    radiance: texture_3d<f32>,
    linearClamp: sampler,
    origin: vec3f,
    dir: vec3f,
    n: vec3f,
    tanHalf: f32,
    startDist: f32,
    maxDist: f32,
    maxSteps: u32,
    anisoOffset: f32,
) -> vec4f {
    var color = vec3f(0.0);
    var transmittance = 1.0;
    var dist = startDist;
    // no coarser than a quarter of the volume: a cell of the top mips holds most of the scene,
    // and its directional composite starts at the far side of it (the underside of the floor the
    // cone left), whatever lies between; past that width the cone goes on as a cylinder
    let maxLod = max(f32(vol.mipCount) - 3.0, 0.0);
    let maxDiameter = vol.voxelSize * exp2(maxLod);
    let rise = dot(dir, n);
    var previous = origin;
    for (var i = 0u; i < maxSteps; i++) {
        if (dist >= maxDist || transmittance < 0.01) { break; }
        let diameter = clamp(2.0 * tanHalf * dist, vol.voxelSize, maxDiameter);
        let p = origin + dir * dist + n * max(LIFT * diameter - dist * rise, 0.0);
        let uvw = voxelUvw(vol, p);
        if (any(uvw < vec3f(0.0)) || any(uvw > vec3f(1.0))) { break; }
        let lod = min(log2(diameter / vol.voxelSize), maxLod);
        var s: vec4f;
        if (lod >= anisoOffset) {
            s = voxelAnisoSample(linearClamp, uvw, dir, lod - anisoOffset);
        } else {
            s = mix(textureSampleLevel(radiance, linearClamp, uvw, 0.0), voxelAnisoSample(linearClamp, uvw, dir, 0.0), lod / anisoOffset);
        }
        // as voxelConeTrace: the step's share of the sampled voxel's opacity and radiance, for
        // the length the lifted samples actually cover (longer than the step along the axis)
        let step = 0.5 * diameter;
        let covered = select(length(p - previous), step, i == 0u);
        previous = p;
        let crossed = covered / (vol.voxelSize * exp2(lod));
        let a = 1.0 - pow(max(1.0 - s.a, 0.0), crossed);
        let share = select(crossed, a / s.a, s.a > 1e-4);
        color += transmittance * s.rgb * share;
        transmittance *= 1.0 - a;
        dist += step;
    }
    return vec4f(color * vol.radianceScale, transmittance);
}

// Returns the irradiance (rgb, scene units) and the share of the cosine-weighted hemisphere that
// sees past the volume (a). The tilted cones turn by `angle` about the normal (jitter it per
// pixel and frame and let a temporal filter integrate); `skyScale` scales the sky.
fn voxelIrradiance(
    vol: VoxelVolume,
    radiance: texture_3d<f32>,
    linearClamp: sampler,
    sky: SkyLighting,
    skyScale: f32,
    origin: vec3f,
    n: vec3f,
    angle: f32,
    startDist: f32,
    maxDist: f32,
    maxSteps: u32,
    anisoOffset: f32,
) -> vec4f {
    // a frame around the normal (Duff et al. 2017)
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    let t = vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
    let bt = vec3f(b, s + n.y * n.y * a, -n.y);
    let tanHalf = 0.57735027;   // 30 degrees
    let tilt = 0.70710678;      // cos and sin of 45 degrees
    var e = vec3f(0.0);
    var open = 0.0;
    for (var k = 0u; k < 6u; k++) {
        var dir = n;
        var w = 1.0;
        if (k > 0u) {
            let phi = angle + f32(k - 1u) * (2.0 * VOXEL_GI_PI / 5.0);
            dir = n * tilt + (t * cos(phi) + bt * sin(phi)) * tilt;
            w = tilt;
        }
        let c = voxelSurfaceConeTrace(vol, radiance, linearClamp, origin, dir, n, tanHalf, startDist, maxDist, maxSteps, anisoOffset);
        e += w * (c.rgb + c.a * skyScale * skyRadiance(sky, dir));
        open += w * c.a;
    }
    // the weights sum to 1 + 5 cos 45: E is pi times the cosine-weighted mean radiance
    let total = 1.0 + 5.0 * tilt;
    return vec4f(e * (VOXEL_GI_PI / total), open / total);
}
