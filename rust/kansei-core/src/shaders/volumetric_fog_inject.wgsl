// Fog injection: per froxel, height-falloff density and the light scattered toward the camera
// (Henyey-Greenstein phase) from directional and point lights, with shadows, plus an optional
// ambient term: the radiance of a uniform sky around the fog (phase integrates to 1 over the
// sphere, so it scatters as density * ambient). Distant fog tends to that colour, and it keeps
// fog lit when no direct light reaches it (dusk, overcast).

struct FogParams {
    invViewProj     : mat4x4f,
    cameraPos       : vec3f,
    baseDensity     : f32,
    windOffset      : vec3f,
    heightFalloff   : f32,
    ambient         : vec3f,
    fogHeight       : f32,
    gridNear        : f32,
    gridFar         : f32,
    cameraNear      : f32,
    cameraFar       : f32,
    gridW           : u32,
    gridH           : u32,
    gridD           : u32,
    numDirLights    : u32,
    numPointLights  : u32,
    hasShadowMap    : u32,
    hasPointShadows : u32,
    extinctionCoeff : f32,
    anisotropy      : f32,
    _pad0           : f32,
    _pad1           : f32,
    _pad2           : f32,
}

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

@group(0) @binding(0) var scatterExtTex    : texture_storage_3d<rgba16float, write>;
@group(0) @binding(1) var shadowDepthTex   : texture_depth_2d;
@group(0) @binding(2) var<uniform> params  : FogParams;
@group(0) @binding(3) var<storage, read> dirLights : array<DirLightData>;
@group(0) @binding(4) var<storage, read> ptLights  : array<PointLightData>;
@group(0) @binding(5) var pointShadowAtlas : texture_2d_array<f32>;
@group(0) @binding(6) var<uniform> shadowViewProj : mat4x4f;

const NO_SHADOW : u32 = 0xffffffffu;
const INV_4PI : f32 = 0.0795774715;

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) * INV_4PI / pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5);
}

fn smoothFalloff(dist: f32, radius: f32) -> f32 {
    let r = clamp(dist / radius, 0.0, 1.0);
    let f = 1.0 - r * r;
    return f * f;
}

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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= params.gridW || gid.y >= params.gridH || gid.z >= params.gridD) { return; }

    let gridSize = vec3f(f32(params.gridW), f32(params.gridH), f32(params.gridD));
    let linearD = sliceDepth(f32(gid.z) + 0.5, params.gridNear, params.gridFar, gridSize.z);
    if (linearD < params.cameraNear) {
        textureStore(scatterExtTex, gid, vec4f(0.0));
        return;
    }

    let worldPos = froxelToWorld(vec3f(f32(gid.x), f32(gid.y), f32(gid.z) + 0.5), gridSize,
                                 params.gridNear, params.gridFar, params.cameraNear, params.cameraFar,
                                 params.invViewProj);

    let samplePos = worldPos + params.windOffset;
    let density = params.baseDensity * exp(-params.heightFalloff * max(samplePos.y - params.fogHeight, 0.0));
    let extinction = density * params.extinctionCoeff;

    let viewDir = normalize(worldPos - params.cameraPos);
    var totalScatter = density * params.ambient;

    for (var di = 0u; di < params.numDirLights; di++) {
        let dl = dirLights[di];
        var visibility = 1.0;
        if (dl.shadowed != 0u && params.hasShadowMap != 0u) {
            visibility = dirShadowLookup(worldPos);
        }
        let phase = henyeyGreenstein(dot(viewDir, -normalize(dl.direction)), params.anisotropy);
        totalScatter += density * dl.color * visibility * phase;
    }

    for (var pi = 0u; pi < params.numPointLights; pi++) {
        let pl = ptLights[pi];
        let dist = length(pl.position - worldPos);
        if (dist > pl.radius) { continue; }

        var visibility = 1.0;
        if (pl.shadowLayer != NO_SHADOW && params.hasPointShadows != 0u) {
            visibility = pointShadowLookup(worldPos, pl.position, pl.shadowLayer);
        }
        let phase = henyeyGreenstein(dot(viewDir, normalize(pl.position - worldPos)), params.anisotropy);
        totalScatter += density * pl.color * smoothFalloff(dist, pl.radius) * visibility * phase;
    }

    textureStore(scatterExtTex, gid, vec4f(totalScatter, extinction));
}
