// Caustics (rt::RtCausticsEffect), the map: `resolve` turns this frame's photons (the atomics,
// cleared for the next) into irradiance and blends it into the map (rgb lux, a the height the
// photons landed at); `composite` adds the map's light to the image where the GBuffer shows an
// upward-facing surface under it, times its albedo / pi (not on glass, the sky or the fluid's
// screen-space surface). Prefixed with rt_caustics_lookup.wgsl.

struct ResolveParams {
    mapDims : vec2u,
    blend   : f32,
    _pad    : u32,
}

const CAUSTICS_SCALE : f32 = 64.0;
const CAUSTICS_WEIGHT : f32 = 4096.0;

@group(0) @binding(0) var<uniform> rsp : ResolveParams;
@group(0) @binding(1) var<storage, read_write> acc : array<atomic<i32>>;
@group(0) @binding(2) var mapIn : texture_2d<f32>;
@group(0) @binding(3) var mapOut : texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8)
fn resolve(@builtin(global_invocation_id) gid : vec3u) {
    if (any(gid.xy >= rsp.mapDims)) { return; }
    let at = (gid.y * rsp.mapDims.x + gid.x) * 5u;
    let e = vec3f(f32(atomicExchange(&acc[at], 0)), f32(atomicExchange(&acc[at + 1u], 0)), f32(atomicExchange(&acc[at + 2u], 0))) / CAUSTICS_SCALE;
    let w = f32(atomicExchange(&acc[at + 3u], 0)) / CAUSTICS_WEIGHT;
    let wy = f32(atomicExchange(&acc[at + 4u], 0)) / CAUSTICS_WEIGHT;
    let previous = textureLoad(mapIn, gid.xy, 0);
    var height = previous.a;
    if (w > 1e-3) {
        height = select(wy / w, mix(previous.a, wy / w, 0.5), previous.a != 0.0);
    }
    textureStore(mapOut, gid.xy, vec4f(mix(previous.rgb, e, rsp.blend), height));
}

struct CompositeParams {
    invProj : mat4x4f,
    invView : mat4x4f,
    size    : vec2f,
    // 0: lit; 1: the light the caustics add and take on a white surface, over a grey of 40 cd/m²
    view    : u32,
    _pad    : u32,
}

@group(0) @binding(0) var<uniform> cc : CompositeParams;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var normalTex : texture_2d<f32>;
@group(0) @binding(3) var albedoTex : texture_2d<f32>;
@group(0) @binding(4) var emissiveTex : texture_2d<f32>;
@group(0) @binding(5) var inputTex : texture_2d<f32>;
@group(0) @binding(6) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(7) var causticsMap : texture_2d<f32>;
@group(0) @binding(8) var causticsSampler : sampler;
@group(0) @binding(9) var<uniform> caustics : KanseiRtCaustics;

@compute @workgroup_size(8, 8)
fn composite(@builtin(global_invocation_id) gid : vec3u) {
    if (any(vec2f(gid.xy) >= cc.size)) { return; }
    let px = gid.xy;
    let color = textureLoad(inputTex, px, 0);
    let depth = textureLoad(depthTex, px, 0);
    let albedo = textureLoad(albedoTex, px, 0);
    // the sky, glass (its albedo's alpha 0.05-0.45) and the fluid's screen-space surface
    let skip = depth >= 1.0 || (albedo.a > 0.025 && albedo.a < 0.475) || textureLoad(emissiveTex, px, 0).a > 0.5;
    var e = vec3f(0.0);
    var white = vec3f(0.0);
    if (!skip) {
        let uv = (vec2f(px) + 0.5) / cc.size;
        let view = cc.invProj * vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth, 1.0);
        let world = (cc.invView * vec4f(view.xyz / view.w, 1.0)).xyz;
        let n = normalize(textureLoad(normalTex, px, 0).xyz * 2.0 - 1.0);
        white = kansei_rt_caustics_lookup(causticsMap, causticsSampler, caustics, world, n) / 3.14159265;
        e = white * albedo.rgb;
    }
    if (cc.view == 1u) {
        textureStore(outputTex, px, vec4f(max(vec3f(40.0) + white, vec3f(0.0)), color.a));
        return;
    }
    textureStore(outputTex, px, vec4f(max(color.rgb + e, vec3f(0.0)), color.a));
}
