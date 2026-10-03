// Fog injection: per froxel, height-falloff density (plus local fog volumes) and the light
// scattered toward the camera (Henyey-Greenstein phase) from directional, point and spot lights, with
// shadows, plus ambient terms: the radiance of a uniform sky around the fog (phase integrates to 1
// over the sphere, so it scatters as density * ambient), and the actual sky's light when a
// SkyAtmosphere is bound (volumetric_fog_media.wgsl). Distant fog tends to that colour, and it
// keeps fog lit when no direct light reaches it (dusk, overcast). The lights and their shadow maps
// are compute_shadows.wgsl's (bindings 1 and 3-9).

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
    startDistance   : f32,
    jitterFrame     : u32,   // 0: sample froxel centres; n > 0: the n-th temporal jitter
    skipSpots       : u32,   // 1: the spot lights are raymarched instead (volumetric_fog_shafts.wgsl)
    // world plane (n, d): the fog lies only where n.p + d >= 0 (a planar reflection's view sees
    // the fog above its mirror); (0, 0, 0, 1) keeps all of it
    clipPlane       : vec4f,
    maxDistance     : f32,   // the view depth the fog ends at (its reach)
}

@group(0) @binding(0) var scatterExtTex    : texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params  : FogParams;

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

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= params.gridW || gid.y >= params.gridH || gid.z >= params.gridD) { return; }

    let gridSize = vec3f(f32(params.gridW), f32(params.gridH), f32(params.gridD));
    let linearD = sliceDepth(f32(gid.z) + 0.5, params.gridNear, params.gridFar, gridSize.z);
    if (linearD < params.cameraNear) {
        textureStore(scatterExtTex, gid, vec4f(0.0));
        return;
    }

    // with temporal accumulation, each frame samples a different point of the froxel (R2
    // sequence), so the history resolves detail finer than the grid: shadow edges in beams
    var jitter = vec3f(0.0);
    if (params.jitterFrame != 0u) {
        jitter = fract(vec3f(0.5) + f32(params.jitterFrame) * vec3f(0.8191725, 0.6710436, 0.5497005)) - 0.5;
    }
    let worldPos = froxelToWorld(vec3f(f32(gid.x) + jitter.x, f32(gid.y) + jitter.y, f32(gid.z) + 0.5 + jitter.z), gridSize,
                                 params.gridNear, params.gridFar, params.cameraNear, params.cameraFar,
                                 params.invViewProj);

    let samplePos = worldPos + params.windOffset;
    // no fog closer than startDistance (UE's fog start distance), faded in over one slice
    let sliceThickness = linearD * (pow(params.gridFar / params.gridNear, 1.0 / gridSize.z) - 1.0);
    let start = saturate((linearD - params.startDistance) / max(sliceThickness, 1e-3) + 0.5);
    let heightFog = start * params.baseDensity * exp(-params.heightFalloff * max(samplePos.y - params.fogHeight, 0.0));
    // the height fog plus local fog volumes (volumetric_fog_media.wgsl): the lights below scale
    // with the total density, and the media's albedo colours what they scatter
    let media = fogMedia(worldPos, heightFog);
    // faded in over a slice across the clip plane, when there is one (a reflection's): the
    // default (0, 0, 0, 1) keeps all the fog, whatever the slices' thickness
    let clipped = any(params.clipPlane.xyz != vec3f(0.0));
    let above = select(1.0, saturate((dot(params.clipPlane.xyz, worldPos) + params.clipPlane.w) / max(sliceThickness, 1e-3) + 0.5), clipped);
    // none past the fog's reach: the slice across it keeps the share of its depth before it
    let d0 = sliceDepth(f32(gid.z), params.gridNear, params.gridFar, gridSize.z);
    let d1 = sliceDepth(f32(gid.z) + 1.0, params.gridNear, params.gridFar, gridSize.z);
    let reach = saturate((params.maxDistance - d0) / max(d1 - d0, 1e-3));
    let density = media.density * above * reach;
    let extinction = media.extinction * above * reach;

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

    // spot lights: cones with shadows (volumetric_fog_spot.wgsl), unless raymarched per pixel
    if (params.skipSpots == 0u) {
        totalScatter += density * spotInScatter(worldPos, viewDir, max(sliceThickness, 0.25));
    }

    totalScatter = (totalScatter + density * skyAmbient(viewDir, worldPos)) * media.albedo;
    textureStore(scatterExtTex, gid, vec4f(totalScatter, extinction));
}
