// Planar reflection, resolve: copy the mirrored view's colour into mip 0 of the reflection
// texture, with alpha = the length of the reflected path from the camera (metres), so materials
// can fog the reflection. Pixels the mirrored view saw no geometry in get SKY_DISTANCE.

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

const SKY_DISTANCE : f32 = 65504.0;   // largest f16

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= rp.size)) { return; }
    let color = textureLoad(srcColor, gid.xy, 0);
    let depth = textureLoad(srcDepth, gid.xy, 0);
    var dist = SKY_DISTANCE;
    if (depth < 1.0) {
        let uv = (vec2f(gid.xy) + 0.5) / vec2f(rp.size);
        let world = rp.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        dist = min(length(world.xyz / world.w - rp.cameraPos), SKY_DISTANCE);
    }
    textureStore(dst, gid.xy, vec4f(color.rgb, dist));
}
