// Per pixel: the blur vector (the velocity target where a material wrote one, else the camera's
// reprojection of the depth), scaled by the shutter and clamped to the largest radius, and the
// view depth, all in this pass' (output) pixels. The depth and velocity are at the scene's
// render size, below the output's when a temporal upscaler runs before this effect: each
// output pixel reads the texel under it, and velocities (uv per frame) scale to output pixels.
// Per 16x16 tile (one workgroup): the longest of those vectors, and the shortest
// length.

@group(0) @binding(0) var depthTex    : texture_depth_2d;
@group(0) @binding(1) var velocityTex : texture_2d<f32>;
@group(0) @binding(2) var motionOut   : texture_storage_2d<rgba16float, write>;   // xy blur px, z view depth
@group(0) @binding(3) var tileOut     : texture_storage_2d<rgba16float, write>;   // xy longest blur px, z shortest
@group(0) @binding(4) var<uniform> p  : MotionBlurParams;

// velocities at or beyond this are the GBuffer's "none written" clear value
const NO_VELOCITY : f32 = 1000.0;

var<workgroup> longest : array<vec4f, 256>;   // xy longest blur, z its squared length, w shortest's

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid : vec3u,
    @builtin(local_invocation_index) li : u32,
    @builtin(workgroup_id) wid : vec3u,
) {
    var v = vec2f(0.0);
    if (all(gid.xy < vec2u(p.size))) {
        let dims = textureDimensions(depthTex);
        let q = min(vec2u((vec2f(gid.xy) + 0.5) * vec2f(dims) / p.size), dims - 1u);
        // reproject the rendered sample itself (its texel's centre)
        let uv = (vec2f(q) + 0.5) / vec2f(dims);
        let depth = textureLoad(depthTex, q, 0);
        let w = p.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let world = vec4f(w.xyz / w.w, 1.0);
        let curr = p.viewProj * world;
        var velocity = textureLoad(velocityTex, q, 0).xy;
        if (!(abs(velocity.x) < NO_VELOCITY)) {
            let prev = p.prevViewProj * world;
            velocity = (curr.xy / curr.w - prev.xy / prev.w) * vec2f(0.5, -0.5);
        }
        v = velocity * p.size * p.scale;
        let len = length(v);
        if (len > p.maxRadius) { v *= p.maxRadius / len; }
        if (any(v != v)) { v = vec2f(0.0); }   // NaN guard
        textureStore(motionOut, gid.xy, vec4f(v, min(curr.w, 65000.0), 0.0));
    }
    // pixels outside the image count as neither
    let len2 = dot(v, v);
    longest[li] = vec4f(v, len2, select(1e9, len2, all(gid.xy < vec2u(p.size))));
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if (li < s) {
            let a = longest[li];
            let b = longest[li + s];
            longest[li] = vec4f(select(a.xyz, b.xyz, b.z > a.z), min(a.w, b.w));
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let l = longest[0];
        textureStore(tileOut, wid.xy, vec4f(l.xy, sqrt(l.w), 0.0));
    }
}
