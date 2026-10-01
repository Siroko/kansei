// Planar reflection, resolve: copy the mirrored view's colour into mip 0 of the reflection
// texture, with alpha = the length of the reflected path from the camera (metres), so materials
// can fog the reflection. Pixels the mirrored view saw no geometry in get SKY_DISTANCE. With a
// ReflectionFog, the volumetric fog above the mirror is composited over the colour first: the
// froxel volume the fog built from the mirrored camera, looked up at each pixel's world position.

struct ResolveParams {
    invViewProj : mat4x4f,   // of the mirrored camera
    cameraPos   : vec3f,     // the mirrored camera (the reflected path length is from here)
    _pad0       : f32,
    size        : vec2u,
    _pad1       : vec2u,
}

@group(0) @binding(0) var srcColor : texture_2d<f32>;
@group(0) @binding(1) var srcDepth : texture_depth_2d;
@group(0) @binding(2) var dst      : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var<uniform> rp : ResolveParams;

struct ReflectionFogParams {
    viewProj : mat4x4f,   // of the mirrored camera the volume was built from
    gridNear : f32,
    gridFar  : f32,
    gridD    : f32,
    enabled  : u32,
}

@group(0) @binding(4) var fogVolume  : texture_3d<f32>;
@group(0) @binding(5) var fogSampler : sampler;
@group(0) @binding(6) var<uniform> rf : ReflectionFogParams;

const SKY_DISTANCE : f32 = 65504.0;   // largest f16

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= rp.size)) { return; }
    var color = textureLoad(srcColor, gid.xy, 0).rgb;
    let depth = textureLoad(srcDepth, gid.xy, 0);
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(rp.size);
    let h = rp.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    let world = h.xyz / h.w;
    var dist = SKY_DISTANCE;
    if (depth < 1.0) {
        dist = min(length(world - rp.cameraPos), SKY_DISTANCE);
    }
    if (rf.enabled != 0u) {
        // the sky (depth 1, the far plane) lies beyond the volume: all of its fog
        let clip = rf.viewProj * vec4f(world, 1.0);
        let ndc = clip.xy / clip.w;
        let slice = depthToSlice(max(clip.w, rf.gridNear), rf.gridNear, rf.gridFar, rf.gridD);
        let fog = textureSampleLevel(fogVolume, fogSampler, vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, saturate(slice / rf.gridD)), 0.0);
        color = color * fog.a + fog.rgb;
    }
    textureStore(dst, gid.xy, vec4f(color, dist));
}
