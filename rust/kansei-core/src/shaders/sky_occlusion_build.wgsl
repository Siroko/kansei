// Sky occlusion, build (shadows::SkyOcclusion): from a depth map of the scene seen straight from
// above, a pyramid of the canopy's cover and height, then a volume of how much of the sky each
// point around the camera sees through it.
//   top:      per texel of the depth map, whether something is there and the height of its top;
//   down:     each level of the pyramid from the one below: the share of the area covered, the
//             cover-weighted mean height of the tops, and the highest top;
//   volume:   per voxel, 24 directions over the upper hemisphere (cosine-weighted) cone-traced
//             through the pyramid: where a ray is below the tops, the canopy (as much of it as
//             covers the cone's footprint) dims it by `extinction` per metre. A slab of layers
//             from `firstLayer` per dispatch, so the volume can be built over several frames.

struct OcclusionBuild {
    center     : vec2f,   // world xz of the map's centre
    extent     : f32,     // its side, metres
    minY       : f32,     // world heights the volume spans
    maxY       : f32,
    eyeY       : f32,     // the top-down view's eye height, near and far (depth to height)
    near       : f32,
    far        : f32,
    extinction : f32,     // per metre of fully covered canopy
    levels     : u32,     // of the pyramid
    level      : u32,     // being built (down)
    firstLayer : u32,     // the volume's first layer in this dispatch (volume)
}

@group(0) @binding(0) var<uniform> b : OcclusionBuild;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var srcLevel : texture_2d<f32>;
@group(0) @binding(3) var dstLevel : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var pyramid : texture_2d<f32>;
@group(0) @binding(5) var pyramidSampler : sampler;
@group(0) @binding(6) var volumeOut : texture_storage_3d<rgba8unorm, write>;

// r: covered (0 or 1), g: covered x top height, b: top height (minY where empty)
@compute @workgroup_size(8, 8)
fn top(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(dstLevel);
    if (any(gid.xy >= size)) { return; }
    let d = textureLoad(depthTex, gid.xy, 0);
    let covered = select(0.0, 1.0, d < 1.0);
    let height = b.eyeY - (b.near + d * (b.far - b.near));
    textureStore(dstLevel, gid.xy, vec4f(covered, covered * height, select(b.minY, height, d < 1.0), 1.0));
}

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(dstLevel);
    if (any(gid.xy >= size)) { return; }
    let src = vec2i(gid.xy) * 2;
    var sum = vec2f(0.0);
    var top = -1e9;
    for (var i = 0; i < 4; i++) {
        let s = textureLoad(srcLevel, src + vec2i(i & 1, i >> 1), 0);
        sum += s.rg;
        top = max(top, s.b);
    }
    textureStore(dstLevel, gid.xy, vec4f(sum * 0.25, top, 1.0));
}

const DIRECTIONS : u32 = 24u;
const STEPS : u32 = 20u;
// the half-angle of each direction's cone: 24 cones share the hemisphere's 2 pi sr
const CONE_TAN : f32 = 0.3;

@compute @workgroup_size(4, 4, 4)
fn volume(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(volumeOut);
    let voxel = gid + vec3u(0u, b.firstLayer, 0u);
    if (any(voxel >= size)) { return; }
    let f = (vec3f(voxel) + 0.5) / vec3f(size);
    let p = vec3f(b.center.x + (f.x - 0.5) * b.extent, mix(b.minY, b.maxY, f.y), b.center.y + (f.z - 0.5) * b.extent);
    let base = vec2f(textureDimensions(pyramid, 0));
    let texel = b.extent / base.x;
    let highest = textureSampleLevel(pyramid, pyramidSampler, vec2f(0.5), f32(b.levels - 1u)).b;
    var visibility = 0.0;
    for (var i = 0u; i < DIRECTIONS; i++) {
        // cosine-weighted directions (a Fibonacci spiral over the disc, lifted to the hemisphere)
        let u = (f32(i) + 0.5) / f32(DIRECTIONS);
        let r = sqrt(u);
        let phi = f32(i) * 2.399963;
        let d = vec3f(r * cos(phi), sqrt(1.0 - u), r * sin(phi));
        var tau = 0.0;
        var t = texel;
        for (var k = 0u; k < STEPS; k++) {
            let dt = t * 0.35 + texel;
            let q = p + d * (t + 0.5 * dt);
            if (q.y > highest) { break; }
            let uv = (q.xz - b.center) / b.extent + 0.5;
            if (any(uv < vec2f(0.0)) || any(uv > vec2f(1.0))) { break; }
            // the pyramid level whose texels match the cone's width here
            let lod = clamp(log2(max(2.0 * t * CONE_TAN, texel) / texel), 0.0, f32(b.levels - 1u));
            let s = textureSampleLevel(pyramid, pyramidSampler, uv, lod);
            let top = s.g / max(s.r, 1e-4);
            if (q.y < top) {
                tau += b.extinction * s.r * dt;
            }
            t += dt;
        }
        visibility += exp(-tau);
    }
    textureStore(volumeOut, voxel, vec4f(visibility / f32(DIRECTIONS), 0.0, 0.0, 1.0));
}
