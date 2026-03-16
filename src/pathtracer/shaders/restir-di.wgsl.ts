export const restirGenerateShader = /* wgsl */`
// ReSTIR DI — Generate initial reservoirs + temporal reuse
// Requires: intersection.wgsl, traversal.wgsl (concatenated before this)
//
// Reservoir layout in storage buffer: 3 vec4f per pixel
//   [pixIdx*3 + 0]: xyz = sample position on light, w = bitcast<f32>(lightIdx)
//   [pixIdx*3 + 1]: xyz = sample normal,             w = targetPdf
//   [pixIdx*3 + 2]: x = W, y = wSum, z = bitcast<f32>(M), w = unused

const RESERVOIR_STRIDE = 3u;

const LIGHT_DIRECTIONAL = 1u;
const LIGHT_AREA        = 2u;
const LIGHT_POINT       = 3u;

struct LightData {
    position   : vec3f,
    lightType  : u32,
    color      : vec3f,
    intensity  : f32,
    normal     : vec3f,
    _pad       : f32,
    extra      : vec4f,
}

struct MaterialData {
    albedo       : vec3f,
    roughness    : f32,
    metallic     : f32,
    ior          : f32,
    maxBounces   : f32,
    transmission : f32,
    absorptionColor : vec3f,
    absorptionDensity : f32,
    emissive     : vec3f,
    emissiveIntensity : f32,
}

struct ReSTIRParams {
    invViewProj   : mat4x4f,
    prevViewProj  : mat4x4f,
    cameraPos     : vec3f,
    frameIndex    : u32,
    width         : u32,
    height        : u32,
    lightCount    : u32,
    maxHistory    : u32,
}

@group(0) @binding(0)  var depthTex       : texture_depth_2d;
@group(0) @binding(1)  var normalTex      : texture_2d<f32>;
@group(0) @binding(2)  var<uniform> params : ReSTIRParams;
@group(0) @binding(3)  var<storage, read> sceneLights   : array<LightData>;
@group(0) @binding(4)  var<storage, read> triangles     : array<f32>;
@group(0) @binding(5)  var<storage, read> bvh4Nodes     : array<vec4f>;
@group(0) @binding(6)  var<storage, read> tlasBvh4Nodes : array<vec4f>;
@group(0) @binding(7)  var<storage, read> instances     : array<Instance>;
@group(0) @binding(8)  var<storage, read> materials     : array<MaterialData>;
@group(0) @binding(9)  var<storage, read> reservoirPrev : array<vec4f>;
@group(0) @binding(10) var<storage, read_write> reservoirCur : array<vec4f>;

// ── PCG RNG (no blue noise needed for reservoir sampling) ──
var<private> _rngState: u32;

fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn initRng(pixel: vec2u, frame: u32) {
    _rngState = pcgHash(pixel.x + pixel.y * 9999u + frame * 1000003u);
}

fn nextRandom() -> f32 {
    _rngState = pcgHash(_rngState);
    return f32(_rngState) / 4294967295.0;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ── Shadow test (skips transmissive surfaces) ──
fn traceShadow(origin: vec3f, dir: vec3f, maxDist: f32) -> bool {
    var ro = origin;
    var remaining = maxDist;
    for (var i = 0u; i < 4u; i++) {
        var sray: Ray;
        sray.origin = ro;
        sray.dir = dir;
        let hit = traceBVHInternal(sray, false, remaining);
        if (!hit.hit) { return false; }
        let hitMat = materials[hit.matIndex];
        if (hitMat.transmission < 0.5) { return true; }
        ro = ro + dir * (hit.t + 0.002);
        remaining -= hit.t + 0.002;
        if (remaining <= 0.0) { return false; }
    }
    return false;
}

// ── Target PDF: unshadowed contribution estimate ──
fn evalTargetPdf(hitPos: vec3f, hitNorm: vec3f, samplePos: vec3f, sampleNorm: vec3f,
                 lightColor: vec3f, lightIntensity: f32, lightType: u32) -> f32 {
    if (lightType == LIGHT_DIRECTIONAL) {
        let lightDir = normalize(-samplePos);
        let nDotL = max(dot(hitNorm, lightDir), 0.0);
        return luminance(lightColor * lightIntensity * nDotL);
    }

    let toLight = samplePos - hitPos;
    let dist2 = dot(toLight, toLight);
    if (dist2 < 1e-8) { return 0.0; }
    let dist = sqrt(dist2);
    let lightDir = toLight / dist;
    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return 0.0; }

    if (lightType == LIGHT_POINT) {
        return luminance(lightColor * lightIntensity * nDotL / dist2);
    }

    // Area light
    let lightCos = max(dot(sampleNorm, -lightDir), 0.0);
    if (lightCos <= 0.0) { return 0.0; }
    return luminance(lightColor * lightIntensity * nDotL * lightCos / dist2);
}

fn writeReservoir(pixIdx: u32, pos: vec3f, norm: vec3f, lightIdx: u32,
                  targetPdf: f32, W: f32, wSum: f32, M: u32) {
    let base = pixIdx * RESERVOIR_STRIDE;
    reservoirCur[base + 0u] = vec4f(pos, bitcast<f32>(lightIdx));
    reservoirCur[base + 1u] = vec4f(norm, targetPdf);
    reservoirCur[base + 2u] = vec4f(W, wSum, bitcast<f32>(M), 0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let pixIdx = coord.y * params.width + coord.x;
    initRng(coord, params.frameIndex);

    // Reconstruct world position from GBuffer
    let traceSize = vec2f(f32(params.width), f32(params.height));
    let uv = (vec2f(coord) + 0.5) / traceSize;
    let gbufDim = vec2u(textureDimensions(depthTex));
    let gbufCoord = min(vec2u(vec2f(gbufDim) * uv), gbufDim - vec2u(1u));
    let depth = textureLoad(depthTex, gbufCoord, 0);

    // Sky — write empty reservoir
    if (depth >= 1.0) {
        writeReservoir(pixIdx, vec3f(0.0), vec3f(0.0), 0u, 0.0, 0.0, 0.0, 0u);
        return;
    }

    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.invViewProj * ndc;
    let worldPos = wp.xyz / wp.w;
    let worldNormal = normalize(textureLoad(normalTex, vec2i(gbufCoord), 0).xyz * 2.0 - 1.0);

    // ── Generate initial reservoir: stream all lights ──
    var rPos = vec3f(0.0);
    var rNorm = vec3f(0.0);
    var rLightIdx = 0u;
    var rTargetPdf = 0.0;
    var rWSum = 0.0;
    var rM = 0u;

    let lightCount = params.lightCount;
    for (var li = 0u; li < lightCount; li++) {
        let light = sceneLights[li];
        var sPos: vec3f;
        var sNorm: vec3f;
        var sourcePdf: f32;

        if (light.lightType == LIGHT_AREA) {
            // Generate random point on area light
            let lr1 = nextRandom();
            let lr2 = nextRandom();
            let sizeX = light.extra.x;
            let sizeZ = light.extra.y;
            let ln = light.normal;
            let lup = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(ln.y) < 0.9);
            let lt = normalize(cross(lup, ln));
            let lb = cross(ln, lt);
            sPos = light.position + lt * (lr1 - 0.5) * sizeX + lb * (lr2 - 0.5) * sizeZ;
            sNorm = ln;
            sourcePdf = 1.0 / (sizeX * sizeZ * f32(lightCount));
        } else if (light.lightType == LIGHT_POINT) {
            sPos = light.position;
            sNorm = vec3f(0.0, 1.0, 0.0);
            sourcePdf = 1.0 / f32(lightCount);
        } else {
            // Directional: store direction in samplePos
            sPos = light.position;
            sNorm = vec3f(0.0, 1.0, 0.0);
            sourcePdf = 1.0 / f32(lightCount);
        }

        let pHat = evalTargetPdf(worldPos, worldNormal, sPos, sNorm,
                                  light.color, light.intensity, light.lightType);
        let w = pHat / max(sourcePdf, 1e-20);
        rWSum += w;
        rM += 1u;

        if (nextRandom() * rWSum < w) {
            rPos = sPos;
            rNorm = sNorm;
            rLightIdx = li;
            rTargetPdf = pHat;
        }
    }

    // Compute W for initial reservoir
    var rW = 0.0;
    if (rTargetPdf > 0.0 && rM > 0u) {
        rW = rWSum / (f32(rM) * rTargetPdf);
    }

    // ── Temporal reuse: merge with previous frame's reservoir ──
    if (params.frameIndex > 0u) {
        // Reproject to previous frame
        let prevClip = params.prevViewProj * vec4f(worldPos, 1.0);
        let prevNDC = prevClip.xyz / prevClip.w;
        let prevUV = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

        if (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0) {
            let prevPixCoord = vec2u(traceSize * prevUV);
            let prevPixIdx = min(prevPixCoord.y * params.width + prevPixCoord.x,
                                 params.width * params.height - 1u);

            let pBase = prevPixIdx * RESERVOIR_STRIDE;
            let pr0 = reservoirPrev[pBase + 0u];
            let pr1 = reservoirPrev[pBase + 1u];
            let pr2 = reservoirPrev[pBase + 2u];
            let pPos = pr0.xyz;
            let pNorm = pr1.xyz;
            let pLightIdx = bitcast<u32>(pr0.w);
            let pW = pr2.x;
            let pM = bitcast<u32>(pr2.z);

            if (pM > 0u && pLightIdx < lightCount) {
                // Cap history to prevent stale samples
                let cappedM = min(pM, params.maxHistory);

                // Re-evaluate target PDF at current surface
                let prevLight = sceneLights[pLightIdx];
                let pHatAtCur = evalTargetPdf(worldPos, worldNormal, pPos, pNorm,
                                               prevLight.color, prevLight.intensity, prevLight.lightType);

                let mergeW = pHatAtCur * pW * f32(cappedM);
                rWSum += mergeW;
                rM += cappedM;

                if (nextRandom() * rWSum < mergeW) {
                    rPos = pPos;
                    rNorm = pNorm;
                    rLightIdx = pLightIdx;
                    rTargetPdf = pHatAtCur;
                }

                if (rTargetPdf > 0.0 && rM > 0u) {
                    rW = rWSum / (f32(rM) * rTargetPdf);
                }
            }
        }
    }

    writeReservoir(pixIdx, rPos, rNorm, rLightIdx, rTargetPdf, rW, rWSum, rM);
}
`;

export const restirSpatialShader = /* wgsl */`
// ReSTIR DI — Spatial reuse + final shading
// Requires: intersection.wgsl, traversal.wgsl (concatenated before this)

const RESERVOIR_STRIDE = 3u;

const LIGHT_DIRECTIONAL = 1u;
const LIGHT_AREA        = 2u;
const LIGHT_POINT       = 3u;

struct LightData {
    position   : vec3f,
    lightType  : u32,
    color      : vec3f,
    intensity  : f32,
    normal     : vec3f,
    _pad       : f32,
    extra      : vec4f,
}

struct MaterialData {
    albedo       : vec3f,
    roughness    : f32,
    metallic     : f32,
    ior          : f32,
    maxBounces   : f32,
    transmission : f32,
    absorptionColor : vec3f,
    absorptionDensity : f32,
    emissive     : vec3f,
    emissiveIntensity : f32,
}

struct ReSTIRParams {
    invViewProj   : mat4x4f,
    prevViewProj  : mat4x4f,
    cameraPos     : vec3f,
    frameIndex    : u32,
    width         : u32,
    height        : u32,
    lightCount    : u32,
    maxHistory    : u32,
}

@group(0) @binding(0)  var depthTex       : texture_depth_2d;
@group(0) @binding(1)  var normalTex      : texture_2d<f32>;
@group(0) @binding(2)  var<uniform> params : ReSTIRParams;
@group(0) @binding(3)  var<storage, read> sceneLights   : array<LightData>;
@group(0) @binding(4)  var<storage, read> triangles     : array<f32>;
@group(0) @binding(5)  var<storage, read> bvh4Nodes     : array<vec4f>;
@group(0) @binding(6)  var<storage, read> tlasBvh4Nodes : array<vec4f>;
@group(0) @binding(7)  var<storage, read> instances     : array<Instance>;
@group(0) @binding(8)  var<storage, read> materials     : array<MaterialData>;
@group(0) @binding(9)  var<storage, read> reservoirIn   : array<vec4f>;
@group(0) @binding(10) var directLightOut : texture_storage_2d<rgba16float, write>;

// ── PCG RNG ──
var<private> _rngState: u32;

fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn initRng(pixel: vec2u, frame: u32) {
    _rngState = pcgHash(pixel.x + pixel.y * 9999u + (frame + 7777u) * 1000003u);
}

fn nextRandom() -> f32 {
    _rngState = pcgHash(_rngState);
    return f32(_rngState) / 4294967295.0;
}

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

fn traceShadow(origin: vec3f, dir: vec3f, maxDist: f32) -> bool {
    var ro = origin;
    var remaining = maxDist;
    for (var i = 0u; i < 4u; i++) {
        var sray: Ray;
        sray.origin = ro;
        sray.dir = dir;
        let hit = traceBVHInternal(sray, false, remaining);
        if (!hit.hit) { return false; }
        let hitMat = materials[hit.matIndex];
        if (hitMat.transmission < 0.5) { return true; }
        ro = ro + dir * (hit.t + 0.002);
        remaining -= hit.t + 0.002;
        if (remaining <= 0.0) { return false; }
    }
    return false;
}

fn evalTargetPdf(hitPos: vec3f, hitNorm: vec3f, samplePos: vec3f, sampleNorm: vec3f,
                 lightColor: vec3f, lightIntensity: f32, lightType: u32) -> f32 {
    if (lightType == LIGHT_DIRECTIONAL) {
        let lightDir = normalize(-samplePos);
        let nDotL = max(dot(hitNorm, lightDir), 0.0);
        return luminance(lightColor * lightIntensity * nDotL);
    }
    let toLight = samplePos - hitPos;
    let dist2 = dot(toLight, toLight);
    if (dist2 < 1e-8) { return 0.0; }
    let dist = sqrt(dist2);
    let lightDir = toLight / dist;
    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return 0.0; }
    if (lightType == LIGHT_POINT) {
        return luminance(lightColor * lightIntensity * nDotL / dist2);
    }
    let lightCos = max(dot(sampleNorm, -lightDir), 0.0);
    if (lightCos <= 0.0) { return 0.0; }
    return luminance(lightColor * lightIntensity * nDotL * lightCos / dist2);
}

fn evalContribution(hitPos: vec3f, hitNorm: vec3f, samplePos: vec3f, sampleNorm: vec3f,
                    lightColor: vec3f, lightIntensity: f32, lightType: u32) -> vec3f {
    if (lightType == LIGHT_DIRECTIONAL) {
        let lightDir = normalize(-samplePos);
        let nDotL = max(dot(hitNorm, lightDir), 0.0);
        return lightColor * lightIntensity * nDotL;
    }
    let toLight = samplePos - hitPos;
    let dist2 = dot(toLight, toLight);
    if (dist2 < 1e-8) { return vec3f(0.0); }
    let dist = sqrt(dist2);
    let lightDir = toLight / dist;
    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }
    if (lightType == LIGHT_POINT) {
        return lightColor * lightIntensity * nDotL / dist2;
    }
    let lightCos = max(dot(sampleNorm, -lightDir), 0.0);
    if (lightCos <= 0.0) { return vec3f(0.0); }
    return lightColor * lightIntensity * nDotL * lightCos / dist2;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= params.width || gid.y >= params.height) { return; }

    let coord = vec2u(gid.xy);
    let pixIdx = coord.y * params.width + coord.x;
    initRng(coord, params.frameIndex);

    let traceSize = vec2f(f32(params.width), f32(params.height));
    let uv = (vec2f(coord) + 0.5) / traceSize;
    let gbufDim = vec2u(textureDimensions(depthTex));
    let gbufCoord = min(vec2u(vec2f(gbufDim) * uv), gbufDim - vec2u(1u));
    let depth = textureLoad(depthTex, gbufCoord, 0);

    if (depth >= 1.0) {
        textureStore(directLightOut, coord, vec4f(0.0));
        return;
    }

    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = params.invViewProj * ndc;
    let worldPos = wp.xyz / wp.w;
    let worldNormal = normalize(textureLoad(normalTex, vec2i(gbufCoord), 0).xyz * 2.0 - 1.0);

    // Read current reservoir (post-temporal)
    let base = pixIdx * RESERVOIR_STRIDE;
    let r0 = reservoirIn[base + 0u];
    let r1 = reservoirIn[base + 1u];
    let r2 = reservoirIn[base + 2u];
    var rPos = r0.xyz;
    var rNorm = r1.xyz;
    var rLightIdx = bitcast<u32>(r0.w);
    var rTargetPdf = r1.w;
    var rW = r2.x;
    var rWSum = r2.y;
    var rM = bitcast<u32>(r2.z);

    // ── Spatial reuse: merge with 5 random neighbors ──
    let spatialRadius = 10.0;
    let spatialCount = 5u;

    for (var si = 0u; si < spatialCount; si++) {
        let angle = nextRandom() * 6.2831853;
        let radius = nextRandom() * spatialRadius;
        let offset = vec2i(vec2f(cos(angle), sin(angle)) * radius);
        let neighborCoord = vec2i(coord) + offset;

        // Bounds check
        if (neighborCoord.x < 0 || neighborCoord.x >= i32(params.width) ||
            neighborCoord.y < 0 || neighborCoord.y >= i32(params.height)) {
            continue;
        }

        // Validate neighbor similarity (depth + normal)
        let nUV = (vec2f(vec2u(neighborCoord)) + 0.5) / traceSize;
        let nGBuf = min(vec2u(vec2f(gbufDim) * nUV), gbufDim - vec2u(1u));
        let nDepth = textureLoad(depthTex, nGBuf, 0);
        if (abs(depth - nDepth) > 0.1) { continue; }
        let nNormal = normalize(textureLoad(normalTex, vec2i(nGBuf), 0).xyz * 2.0 - 1.0);
        if (dot(worldNormal, nNormal) < 0.9) { continue; }

        let nPixIdx = u32(neighborCoord.y) * params.width + u32(neighborCoord.x);
        let nBase = nPixIdx * RESERVOIR_STRIDE;
        let nr0 = reservoirIn[nBase + 0u];
        let nr1 = reservoirIn[nBase + 1u];
        let nr2 = reservoirIn[nBase + 2u];
        let nPos = nr0.xyz;
        let nNorm2 = nr1.xyz;
        let nLightIdx = bitcast<u32>(nr0.w);
        let nW = nr2.x;
        let nM = min(bitcast<u32>(nr2.z), params.maxHistory);

        if (nM == 0u || nLightIdx >= params.lightCount) { continue; }

        let nLight = sceneLights[nLightIdx];
        let pHatAtMe = evalTargetPdf(worldPos, worldNormal, nPos, nNorm2,
                                      nLight.color, nLight.intensity, nLight.lightType);

        let mergeW = pHatAtMe * nW * f32(nM);
        rWSum += mergeW;
        rM += nM;

        if (nextRandom() * rWSum < mergeW) {
            rPos = nPos;
            rNorm = nNorm2;
            rLightIdx = nLightIdx;
            rTargetPdf = pHatAtMe;
        }
    }

    // Final W
    if (rTargetPdf > 0.0 && rM > 0u) {
        rW = rWSum / (f32(rM) * rTargetPdf);
    } else {
        rW = 0.0;
    }

    // ── Final shading: visibility test + contribution ──
    var directLight = vec3f(0.0);
    if (rTargetPdf > 0.0 && rLightIdx < params.lightCount) {
        let light = sceneLights[rLightIdx];
        var lightDir: vec3f;
        var shadowDist: f32;

        if (light.lightType == LIGHT_DIRECTIONAL) {
            lightDir = normalize(-rPos);
            shadowDist = 1e30;
        } else {
            let toLight = rPos - worldPos;
            let dist = length(toLight);
            lightDir = toLight / dist;
            shadowDist = dist - 0.01;
        }

        if (!traceShadow(worldPos + worldNormal * 0.001, lightDir, shadowDist)) {
            directLight = evalContribution(worldPos, worldNormal, rPos, rNorm,
                                            light.color, light.intensity, light.lightType) * rW;
        }
    }

    textureStore(directLightOut, coord, vec4f(directLight, 1.0));
}
`;
