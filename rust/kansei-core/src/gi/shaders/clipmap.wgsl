// A voxel clipmap of the scene's light (gi::VoxelClipmap): nested windows around the camera, the
// finest of `voxelSize` voxels and each next one twice as coarse, each `dims` voxels of a world
// lattice stored toroidally (voxel c in texel c mod dims) and sampled with a repeating sampler,
// so a world position needs no offset: uvw = p / (level voxel size * dims). rgb is the radiance
// leaving each voxel premultiplied by its coverage, divided by `radianceScale`; a its opacity
// across one voxel. The levels play the part of a volume's mips: a cone reads the level whose
// voxels are as wide as it is, or the next coarser one that holds the point. Bindings: group 0,
// 50 the uniform, 51-56 the levels (a 1-texel stand-in past `levelCount`), 57 the sampler.

const CLIP_MAX_LEVELS : u32 = 6u;

struct ClipLevel {
    origin : vec3i,   // the window's first voxel, in the level's lattice
    valid  : u32,     // 1 once the level holds data
}

struct VoxelClipmap {
    dims          : vec3u,   // voxels of each level
    levelCount    : u32,
    voxelSize     : f32,     // level 0's voxels, metres
    radianceScale : f32,     // stored radiance times this is scene radiance
    _pad0         : f32,
    _pad1         : f32,
    levels        : array<ClipLevel, 6>,
}

@group(0) @binding(50) var<uniform> clipmap : VoxelClipmap;
@group(0) @binding(51) var clipLevel0 : texture_3d<f32>;
@group(0) @binding(52) var clipLevel1 : texture_3d<f32>;
@group(0) @binding(53) var clipLevel2 : texture_3d<f32>;
@group(0) @binding(54) var clipLevel3 : texture_3d<f32>;
@group(0) @binding(55) var clipLevel4 : texture_3d<f32>;
@group(0) @binding(56) var clipLevel5 : texture_3d<f32>;
@group(0) @binding(57) var clipSampler : sampler;

fn clipVoxelSize(k: u32) -> f32 {
    return clipmap.voxelSize * exp2(f32(k));
}

// Whether level k holds data at world position p, `margin` voxels in from its window's sides (a
// trilinear sample reaches half a voxel: keep it off the voxels that wrap around).
fn clipContains(k: u32, p: vec3f, margin: f32) -> bool {
    let level = clipmap.levels[k];
    if (k >= clipmap.levelCount || level.valid == 0u) { return false; }
    let v = p / clipVoxelSize(k) - vec3f(level.origin);
    return all(v >= vec3f(margin)) && all(v <= vec3f(clipmap.dims) - margin);
}

// The finest level from `first` that holds p (`margin` voxels in), or levelCount if none does.
fn clipLevelAt(p: vec3f, first: u32, margin: f32) -> u32 {
    for (var k = first; k < clipmap.levelCount; k++) {
        if (clipContains(k, p, margin)) { return k; }
    }
    return clipmap.levelCount;
}

// Level k at world position p, trilinear (stored radiance, opacity).
fn clipSample(k: u32, p: vec3f) -> vec4f {
    let uvw = p / (clipVoxelSize(k) * vec3f(clipmap.dims));
    switch (k) {
        case 0u: { return textureSampleLevel(clipLevel0, clipSampler, uvw, 0.0); }
        case 1u: { return textureSampleLevel(clipLevel1, clipSampler, uvw, 0.0); }
        case 2u: { return textureSampleLevel(clipLevel2, clipSampler, uvw, 0.0); }
        case 3u: { return textureSampleLevel(clipLevel3, clipSampler, uvw, 0.0); }
        case 4u: { return textureSampleLevel(clipLevel4, clipSampler, uvw, 0.0); }
        default: { return textureSampleLevel(clipLevel5, clipSampler, uvw, 0.0); }
    }
}

// How far a surface cone's samples are lifted off the surface, in footprints (voxel_irradiance's
// LIFT): a trilinear sample reaches a voxel away, and the surface's own voxel may sit half of
// one above it.
const CLIP_LIFT : f32 = 1.5;

// A cone through the clipmap (voxelConeTrace's step, with levels for mips): each step reads the
// level whose voxels are as wide as the cone there (at least `minDiameter`: its first footprint),
// or the next coarser one that holds the point, blended toward the level above by the fraction
// of the width, and advances half the width. With a surface normal `n` (zero for none), each
// sample is lifted off the surface until the voxels it reads clear it (voxelSurfaceConeTrace).
// Wider than the coarsest level's voxels, the cone reads that level over the longer steps. It
// stops where it leaves the coarsest level. Returns the scene radiance gathered (rgb) and the
// transmittance left (a): add `a * sky` for the light from past the clipmap.
fn clipConeTrace(origin: vec3f, dir: vec3f, n: vec3f, tanHalf: f32, minDiameter: f32, startDist: f32, maxDist: f32, maxSteps: u32) -> vec4f {
    var color = vec3f(0.0);
    var transmittance = 1.0;
    var dist = startDist;
    let coarsest = clipVoxelSize(clipmap.levelCount - 1u);
    let rise = dot(dir, n);
    var previous = origin;
    for (var i = 0u; i < maxSteps; i++) {
        if (dist >= maxDist || transmittance < 0.01) { break; }
        let diameter = max(2.0 * tanHalf * dist, minDiameter);
        // the lift clears the voxels read, no wider than the coarsest level's
        let p = origin + dir * dist + n * max(CLIP_LIFT * min(diameter, max(coarsest, minDiameter)) - dist * rise, 0.0);
        let lod = max(log2(diameter / clipmap.voxelSize), 0.0);
        let k = clipLevelAt(p, min(u32(lod), clipmap.levelCount - 1u), 0.5);
        if (k >= clipmap.levelCount) { break; }
        var s = clipSample(k, p);
        // toward the next level by the share of the width past this one's voxels
        let t = lod - f32(k);
        if (t > 0.0 && clipContains(k + 1u, p, 0.5)) {
            s = mix(s, clipSample(k + 1u, p), min(t, 1.0));
        }
        // the voxel width sampled: the cone's, or the level's where the cone is finer
        let voxel = clipmap.voxelSize * exp2(max(lod, f32(k)));
        // as voxelConeTrace: the step's share of the sampled voxel's opacity and radiance, for
        // the length the (lifted) samples cover
        let step = 0.5 * diameter;
        let covered = select(length(p - previous), step, i == 0u);
        previous = p;
        let crossed = covered / voxel;
        let a = 1.0 - pow(max(1.0 - s.a, 0.0), crossed);
        let share = select(crossed, a / s.a, s.a > 1e-4);
        color += transmittance * s.rgb * share;
        transmittance *= 1.0 - a;
        dist += step;
    }
    return vec4f(color * clipmap.radianceScale, transmittance);
}

// The irradiance a surface (normal n) at `origin` receives through the clipmap: voxelIrradiance's
// six 60-degree cones (one along n, five tilted 45 degrees and turned by `angle`), with the sky
// (SKY_LIGHTING_WGSL's skyRadiance, times skyScale) past it. Returns the irradiance (rgb, scene
// units) and the share of the cosine-weighted hemisphere that sees past the clipmap (a).
fn clipIrradiance(sky: SkyLighting, skyScale: f32, origin: vec3f, n: vec3f, angle: f32, minDiameter: f32, startDist: f32, maxDist: f32, maxSteps: u32) -> vec4f {
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
            let phi = angle + f32(k - 1u) * (2.0 * 3.14159265 / 5.0);
            dir = n * tilt + (t * cos(phi) + bt * sin(phi)) * tilt;
            w = tilt;
        }
        let c = clipConeTrace(origin, dir, n, tanHalf, minDiameter, startDist, maxDist, maxSteps);
        e += w * (c.rgb + c.a * skyScale * skyRadiance(sky, dir));
        open += w * c.a;
    }
    // the weights sum to 1 + 5 cos 45: E is pi times the cosine-weighted mean radiance
    let total = 1.0 + 5.0 * tilt;
    return vec4f(e * (3.14159265 / total), open / total);
}
