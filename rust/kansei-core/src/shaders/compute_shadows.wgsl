// The renderer's shadowed lights for compute passes (shadows::ComputeShadows), shared by the
// volumetric fog's injection and voxel GI's light injection: group 0 bindings 1 and 3-9, the
// copy of group 3 that compute shaders can see (group 3 itself is fragment-only). Needs
// spot_light_types.wgsl.
//
// - directional lights (`dirLights`), the first of them shadowed by `shadowDepthTex` through
//   `shadowViewProj` (the single shadow map, or the cascaded map's widest cascade);
// - point lights (`ptLights`), the first shadow-casting one through the cube atlas;
// - the renderer's spot lights and their shadow atlas.

struct DirLightData {
    direction : vec3f,   // the direction the light travels
    shadowed  : u32,     // 1 = use the directional shadow map
    color     : vec3f,
    _pad      : f32,
}

struct PointLightData {
    position    : vec3f,
    radius      : f32,
    color       : vec3f,
    shadowLayer : u32,   // first cube-face layer in the point shadow atlas; 0xffffffff = none
}

@group(0) @binding(1) var shadowDepthTex   : texture_depth_2d;
@group(0) @binding(3) var<storage, read> dirLights : array<DirLightData>;
@group(0) @binding(4) var<storage, read> ptLights  : array<PointLightData>;
@group(0) @binding(5) var pointShadowAtlas : texture_2d_array<f32>;
@group(0) @binding(6) var<uniform> shadowViewProj : mat4x4f;
@group(0) @binding(7) var<storage, read> spotLights : KanseiSpotLights;
@group(0) @binding(8) var spotShadowAtlas : texture_depth_2d_array;
@group(0) @binding(9) var spotShadowSampler : sampler_comparison;

const NO_SHADOW : u32 = 0xffffffffu;

fn dirShadowLookup(worldPos: vec3f) -> f32 {
    let lightClip = shadowViewProj * vec4f(worldPos, 1.0);
    let lightNDC  = lightClip.xyz / lightClip.w;
    let shadowUV  = vec2f(lightNDC.x * 0.5 + 0.5, 1.0 - (lightNDC.y * 0.5 + 0.5));

    if (shadowUV.x < 0.0 || shadowUV.x > 1.0 || shadowUV.y < 0.0 || shadowUV.y > 1.0 || lightNDC.z > 1.0) {
        return 1.0;
    }

    let shadowDim = textureDimensions(shadowDepthTex, 0);
    let coordF = shadowUV * vec2f(shadowDim) - 0.5;
    let base = vec2i(floor(coordF));
    var pcf = 0.0;
    for (var dy = 0; dy <= 1; dy++) {
        for (var dx = 0; dx <= 1; dx++) {
            let sc = clamp(base + vec2i(dx, dy), vec2i(0), vec2i(shadowDim) - 1);
            let sd = textureLoad(shadowDepthTex, sc, 0);
            pcf += select(0.0, 1.0, lightNDC.z <= sd + 0.005);
        }
    }
    return pcf * 0.25;
}

// Whether the directional shadow map covers `worldPos` (dirShadowLookup gives 1 outside it).
fn dirShadowCovers(worldPos: vec3f) -> bool {
    let lightClip = shadowViewProj * vec4f(worldPos, 1.0);
    let lightNDC  = lightClip.xyz / lightClip.w;
    let shadowUV  = vec2f(lightNDC.x * 0.5 + 0.5, 1.0 - (lightNDC.y * 0.5 + 0.5));
    return all(shadowUV >= vec2f(0.0)) && all(shadowUV <= vec2f(1.0)) && lightNDC.z <= 1.0;
}

// Same face layout as CubeMapShadowMap and basic_lit.wgsl's calcPointShadow.
fn pointShadowLookup(worldPos: vec3f, lightPos: vec3f, firstLayer: u32) -> f32 {
    let toFrag = worldPos - lightPos;
    let dist = length(toFrag);
    let dir = toFrag / max(dist, 1e-6);
    let a = abs(dir);

    var face: u32;
    var uv: vec2f;
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x > 0.0) { face = 0u; uv = vec2f(-dir.z, -dir.y) / a.x; }
        else             { face = 1u; uv = vec2f( dir.z, -dir.y) / a.x; }
    } else if (a.y >= a.x && a.y >= a.z) {
        if (dir.y > 0.0) { face = 2u; uv = vec2f(dir.x,  dir.z) / a.y; }
        else             { face = 3u; uv = vec2f(dir.x, -dir.z) / a.y; }
    } else {
        if (dir.z > 0.0) { face = 4u; uv = vec2f( dir.x, -dir.y) / a.z; }
        else             { face = 5u; uv = vec2f(-dir.x, -dir.y) / a.z; }
    }
    uv = uv * 0.5 + 0.5;

    let texDim = vec2f(textureDimensions(pointShadowAtlas, 0));
    let tc = vec2i(clamp(uv * texDim, vec2f(0.0), texDim - 1.0));
    let storedDist = textureLoad(pointShadowAtlas, tc, i32(firstLayer + face), 0).r;
    let bias = 0.05 + dist * 0.002;
    return select(0.0, 1.0, dist <= storedDist + bias);
}
