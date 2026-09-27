// Volumetric clouds, composite: the reduced-resolution clouds upsampled over the image, as
// color * transmittance + light. A texel only counts where its cloud lies in front of this
// pixel's surface, so clouds never spill over the scene's silhouettes.

@group(0) @binding(0) var<uniform> frame : SkyFrame;
@group(0) @binding(1) var inputTex : texture_2d<f32>;
@group(0) @binding(2) var depthTex : texture_depth_2d;
@group(0) @binding(3) var cloudTex : texture_2d<f32>;
@group(0) @binding(4) var cloudDepth : texture_2d<f32>;
@group(0) @binding(5) var outputTex : texture_storage_2d<rgba16float, write>;

fn unprojectComposite(uv: vec2f, depth: f32) -> vec3f {
    let p = frame.invViewProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
    return p.xyz / p.w;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(outputTex);
    if (any(gid.xy >= size)) { return; }
    let color = textureLoad(inputTex, gid.xy, 0);
    let dsize = vec2f(textureDimensions(depthTex));
    let uv = (vec2f(gid.xy) + 0.5) / vec2f(size);
    let depth = textureLoad(depthTex, min(vec2i(uv * dsize), vec2i(dsize) - 1), 0);
    var surfaceKm = 1.0e9;
    if (depth < 1.0) { surfaceKm = length(unprojectComposite(uv, depth) - frame.cameraWorld) * 0.001; }

    let csize = vec2i(textureDimensions(cloudTex));
    let pos = uv * vec2f(csize) - 0.5;
    let base = floor(pos);
    let f = pos - base;
    var sum = vec4f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let o = vec2f(f32(i & 1u), f32(i >> 1u));
        let t = clamp(vec2i(base + o), vec2i(0), csize - 1);
        let bilinear = select(1.0 - f.x, f.x, o.x > 0.5) * select(1.0 - f.y, f.y, o.y > 0.5);
        let w = bilinear * select(0.0, 1.0, textureLoad(cloudDepth, t, 0).r < surfaceKm);
        sum += textureLoad(cloudTex, t, 0) * w;
        weight += w;
    }
    var cloud = vec4f(0.0, 0.0, 0.0, 1.0);
    if (weight > 1e-4) { cloud = sum / weight; }
    textureStore(outputTex, gid.xy, vec4f(color.rgb * cloud.a + cloud.rgb, color.a));
}
