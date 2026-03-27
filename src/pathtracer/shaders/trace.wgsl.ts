// BVH buffer bindings for the trace pipeline (bindings 5-8)
export const traceBVHBindings = /* wgsl */`
@group(0) @binding(5)  var<storage, read> triangles     : array<f32>;
@group(0) @binding(6)  var<storage, read> bvh4Nodes     : array<vec4f>;
@group(0) @binding(7)  var<storage, read> tlasBvh4Nodes : array<vec4f>;
@group(0) @binding(8)  var<storage, read> instances     : array<Instance>;

// Stubs for cone tracing API (only used in voxel mode, never called in BVH mode)
struct VoxelGridParams {
    gridMin: vec3f, voxelSize: f32, gridMax: vec3f, gridRes: u32,
    numLevels: u32, _pad0: u32, _pad1: u32, _pad2: u32,
    mipOffsets: array<vec4u, 2>,
}
fn sampleVoxelMip(pos: vec3f, level: u32) -> vec4f { return vec4f(0.0); }
fn getVoxelParams() -> VoxelGridParams {
    var p: VoxelGridParams;
    return p;
}
`;

export const traceShader = /* wgsl */`
// Requires: intersection.wgsl, traversal.wgsl, storage bindings

struct MaterialData {
    albedo       : vec3f,
    roughness    : f32,
    metallic     : f32,
    ior          : f32,
    maxBounces   : f32,
    transmission : f32,  // 0.0 = opaque, 1.0 = fully transmissive
    absorptionColor : vec3f,
    absorptionDensity : f32,
    emissive     : vec3f,
    emissiveIntensity : f32,
}

struct TraceParams {
    invViewProj  : mat4x4f,
    cameraPos    : vec3f,
    frameIndex   : u32,
    width        : u32,
    height       : u32,
    lightCount   : u32,
    spp          : u32,
    useBlueNoise : u32,
    fixedSeed    : u32,
    maxBounces   : u32,
    useReSTIR    : u32,
    ambientColor : vec3f,
    useProbes    : u32,
    probeGridMin : vec3f,
    probeStepX   : f32,
    probeGridDims: vec3u,
    probeStepY   : f32,
    probeStepZ    : f32,
    rasterDirect  : u32,
    showVoxels       : u32,
    fogDensity       : f32,       // 0 = no fog, >0 = participating media
    fogAnisotropy    : f32,       // Henyey-Greenstein g (-1..1)
    fogHeightFalloff : f32,       // exponential height decay
    _pad3            : f32,
}

const LIGHT_DIRECTIONAL = 1u;
const LIGHT_AREA        = 2u;
const LIGHT_POINT       = 3u;

struct LightData {
    position   : vec3f,    // world pos (area/point) or direction (directional)
    lightType  : u32,      // 1=directional, 2=area, 3=point
    color      : vec3f,
    intensity  : f32,
    normal     : vec3f,    // area light facing direction
    _pad       : f32,
    extra      : vec4f,    // area: (sizeX, sizeZ, 0, 0), point: (radius, 0, 0, 0)
}

@group(0) @binding(0)  var          depthTex   : texture_depth_2d;
@group(0) @binding(1)  var          normalTex  : texture_2d<f32>;
@group(0) @binding(2)  var          albedoTex  : texture_2d<f32>;
@group(0) @binding(3)  var          giOutput   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4)  var<uniform> traceParams : TraceParams;
// Bindings 5-8: declared externally by the traversal module (BVH or Voxel DDA)
@group(0) @binding(9)  var<storage, read> materials  : array<MaterialData>;
@group(0) @binding(10) var<storage, read> sceneLights : array<LightData>;
@group(0) @binding(11) var<storage, read> blueNoise   : array<f32>;
@group(0) @binding(12) var          emissiveTex : texture_2d<f32>;
@group(0) @binding(13) var          restirDirectTex : texture_2d<f32>;
@group(0) @binding(14) var<storage, read> probeSH : array<vec4f>;

// ── Voxel-space indirect helpers ─────────────────────────────────────────
// Mip-interpolated color sample from the voxel color pyramid.
fn sampleVoxelMipLerp(pos: vec3f, mipF: f32) -> vec4f {
    let lo = u32(floor(mipF));
    let hi = min(lo + 1u, getVoxelParams().numLevels - 1u);
    let frac = mipF - f32(lo);
    return mix(sampleVoxelMip(pos, lo), sampleVoxelMip(pos, hi), frac);
}

// ── Irradiance Probe Grid (L2 SH) ──────────────────────────────────
const SH_STRIDE = 9u;

fn computeSHBasisTrace(d: vec3f) -> array<f32, 9> {
    let x = d.x; let y = d.y; let z = d.z;
    var sh: array<f32, 9>;
    sh[0] = 0.282095;
    sh[1] = 0.488603 * y;
    sh[2] = 0.488603 * z;
    sh[3] = 0.488603 * x;
    sh[4] = 1.092548 * x * y;
    sh[5] = 1.092548 * y * z;
    sh[6] = 0.315392 * (3.0*z*z - 1.0);
    sh[7] = 1.092548 * x * z;
    sh[8] = 0.546274 * (x*x - y*y);
    return sh;
}

fn evaluateProbeSH(probeIdx: u32, dir: vec3f) -> vec3f {
    let base = probeIdx * SH_STRIDE;
    let sh = computeSHBasisTrace(dir);
    var result = vec3f(0.0);
    for (var c = 0u; c < 9u; c++) {
        result += probeSH[base + c].xyz * sh[c];
    }
    return max(result, vec3f(0.0));
}

fn sampleProbeGrid(worldPos: vec3f, dir: vec3f) -> vec3f {
    let dims = traceParams.probeGridDims;
    let gridStep = vec3f(traceParams.probeStepX, traceParams.probeStepY, traceParams.probeStepZ);

    // Continuous probe coordinate
    let localPos = (worldPos - traceParams.probeGridMin) / gridStep;
    let baseCoord = vec3i(floor(localPos));
    let frac = localPos - vec3f(baseCoord);

    var totalIrradiance = vec3f(0.0);
    var totalWeight = 0.0;

    // Pure trilinear interpolation over 8 surrounding probes
    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let probeCoord = baseCoord + vec3i(dx, dy, dz);

                // Clamp to grid bounds
                let clamped = clamp(probeCoord, vec3i(0), vec3i(dims) - vec3i(1));
                let probeIdx = u32(clamped.z) * dims.x * dims.y + u32(clamped.y) * dims.x + u32(clamped.x);

                // Trilinear weight
                let t = vec3f(f32(dx), f32(dy), f32(dz));
                let weight = (1.0 - abs(t.x - frac.x)) * (1.0 - abs(t.y - frac.y)) * (1.0 - abs(t.z - frac.z));

                if (weight <= 0.0) { continue; }

                let irradiance = evaluateProbeSH(probeIdx, dir);
                totalIrradiance += irradiance * weight;
                totalWeight += weight;
            }
        }
    }

    if (totalWeight > 0.0) {
        return totalIrradiance / totalWeight;
    }
    return vec3f(0.0);
}

// ── Blue noise hybrid sampler ────────────────────────────────────────
// First 8 dimensions use blue noise (spatially decorrelated, golden-ratio
// animated per frame). Higher dimensions fall back to PCG.
const BN_SIZE : u32 = 128u;
const GOLDEN_RATIO : f32 = 0.6180339887;

var<private> _bnPixel : vec2u;
var<private> _bnFrame : u32;
var<private> _bnDim   : u32;
var<private> _rngState: u32;

fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn initSampler(pixel: vec2u, frame: u32, sampleIdx: u32) {
    _bnPixel = pixel;
    _bnFrame = select(frame * 16u + sampleIdx, sampleIdx, traceParams.fixedSeed != 0u);
    _bnDim = 0u;
    _rngState = pcgHash(pixel.x + pixel.y * 9999u + _bnFrame * 1000003u);
}

fn nextRandom() -> f32 {
    let dim = _bnDim;
    _bnDim += 1u;

    if (dim < 8u && traceParams.useBlueNoise != 0u) {
        // Blue noise: per-dimension spatial offset (coprime to BN_SIZE) + temporal golden ratio
        let px = (_bnPixel.x + dim * 17u) % BN_SIZE;
        let py = (_bnPixel.y + dim * 31u) % BN_SIZE;
        let bn = blueNoise[py * BN_SIZE + px];
        return fract(bn + f32(_bnFrame) * GOLDEN_RATIO);
    }

    // PCG fallback for higher dimensions
    _rngState = pcgHash(_rngState);
    return f32(_rngState) / 4294967295.0;
}

// Robust orthonormal basis from normal (Duff et al. 2017, Frisvad 2012 revised)
// Branchless, no cross products, numerically stable for all directions.
fn buildONB(n: vec3f) -> array<vec3f, 2> {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return array<vec3f, 2>(
        vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x),
        vec3f(b, s + n.y * n.y * a, -n.y),
    );
}

// Cosine-weighted hemisphere sampling
fn cosineSampleHemisphere(n: vec3f, r1: f32, r2: f32) -> vec3f {
    let phi = 2.0 * 3.14159265 * r1;
    let cosTheta = sqrt(1.0 - r2);
    let sinTheta = sqrt(r2);

    let tb = buildONB(n);
    let tangent = tb[0];
    let bitangent = tb[1];

    return normalize(
        tangent * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        n * cosTheta
    );
}

// GGX importance sampling for specular
fn sampleGGX(n: vec3f, roughness: f32, r1: f32, r2: f32) -> vec3f {
    let a = roughness * roughness;
    let phi = 2.0 * 3.14159265 * r1;
    let cosTheta = sqrt((1.0 - r2) / (1.0 + (a * a - 1.0) * r2));
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    let tb = buildONB(n);
    let tangent = tb[0];
    let bitangent = tb[1];

    let h = normalize(
        tangent * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        n * cosTheta
    );
    return h;
}

// Schlick Fresnel approximation
fn fresnelSchlick(cosTheta: f32, F0: vec3f) -> vec3f {
    return F0 + (vec3f(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// ── MIS (Multiple Importance Sampling) helpers ──────────────────────

const PI = 3.14159265;

fn D_GGX(NdotH: f32, a2: f32) -> f32 {
    let denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

fn powerHeuristic(pdfA: f32, pdfB: f32) -> f32 {
    let a2 = pdfA * pdfA;
    return a2 / max(a2 + pdfB * pdfB, 1e-10);
}

// Combined BSDF pdf: specular GGX + diffuse cosine hemisphere
fn pdfBSDF(N: vec3f, wo: vec3f, wi: vec3f, roughness: f32, metallic: f32, surfAlbedo: vec3f) -> f32 {
    let cosTheta = max(dot(N, wi), 0.0);
    if (cosTheta <= 0.0) { return 0.0; }

    let NdotV = max(dot(N, wo), 0.001);
    let F0 = mix(vec3f(0.04), surfAlbedo, metallic);
    let F = fresnelSchlick(NdotV, F0);
    let specAvg = (F.r + F.g + F.b) / 3.0;
    let diffW = (1.0 - specAvg) * (1.0 - metallic);
    let totalW = specAvg + diffW;
    let specProb = clamp(specAvg / max(totalW, 0.001), 0.05, 0.95);

    let pdfDiff = cosTheta / PI;

    let h = normalize(wo + wi);
    let NdotH = max(dot(N, h), 0.0);
    let VdotH = max(dot(wo, h), 0.001);
    let alpha = max(roughness, 0.02) * max(roughness, 0.02);
    let alpha2 = alpha * alpha;
    let pdfSpec = D_GGX(NdotH, alpha2) * NdotH / (4.0 * VdotH);

    return specProb * pdfSpec + (1.0 - specProb) * pdfDiff;
}

// getWorldTriArea is defined by the traversal module (BVH or Voxel DDA)

// ── Shadow test that skips refractive (transparent) surfaces ────────

fn traceShadow(origin: vec3f, dir: vec3f, maxDist: f32) -> bool {
    var ro = origin;
    var remaining = maxDist;
    for (var i = 0u; i < 4u; i++) {
        var sray: Ray;
        sray.origin = ro;
        sray.dir = dir;
        // Closest hit bounded by remaining distance — prunes far-away TLAS/BLAS nodes
        let hit = traceBVHInternal(sray, false, remaining);
        if (!hit.hit) { return false; }
        let hitMat = materials[hit.matIndex];
        if (hitMat.transmission < 0.5) { return true; }
        // Transmissive surface: advance past it
        ro = ro + dir * (hit.t + 0.002);
        remaining -= hit.t + 0.002;
        if (remaining <= 0.0) { return false; }
    }
    return false;
}

// ── Per-light-type evaluation ───────────────────────────────────────

fn evaluateDirectionalLight(hitPos: vec3f, hitNorm: vec3f, light: LightData) -> vec3f {
    let lightDir = normalize(-light.position); // position stores direction for directional
    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }

    if (traceShadow(hitPos + hitNorm * 0.001, lightDir, 1e30)) { return vec3f(0.0); }

    return light.color * light.intensity * nDotL;
}

fn evaluateAreaLight(hitPos: vec3f, hitNorm: vec3f, light: LightData,
                     wo: vec3f, roughness: f32, metallic: f32, surfAlbedo: vec3f) -> vec3f {
    let lr1 = nextRandom();
    let lr2 = nextRandom();
    let sizeX = light.extra.x;
    let sizeZ = light.extra.y;
    let area = sizeX * sizeZ;

    // Build tangent frame from light normal
    let ln = light.normal;
    let lup = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(ln.y) < 0.9);
    let lt = normalize(cross(lup, ln));
    let lb = cross(ln, lt);

    let lightPos = light.position
        + lt * (lr1 - 0.5) * sizeX
        + lb * (lr2 - 0.5) * sizeZ;

    let toLight = lightPos - hitPos;
    let dist2 = dot(toLight, toLight);
    let dist = sqrt(dist2);
    let lightDir = toLight / dist;

    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }

    let lightCos = max(dot(ln, -lightDir), 0.0);
    if (lightCos <= 0.0) { return vec3f(0.0); }

    if (traceShadow(hitPos + hitNorm * 0.001, lightDir, dist - 0.01)) { return vec3f(0.0); }

    // MIS: weight NEE against BSDF probability for this direction
    let pdfLight = dist2 / max(lightCos * area, 1e-8);
    let pdfBsdf = pdfBSDF(hitNorm, wo, lightDir, roughness, metallic, surfAlbedo);
    let misW = powerHeuristic(pdfLight, pdfBsdf);

    return light.color * light.intensity * nDotL * lightCos * area / dist2 * misW;
}

fn evaluatePointLight(hitPos: vec3f, hitNorm: vec3f, light: LightData) -> vec3f {
    let toLight = light.position - hitPos;
    let dist2 = dot(toLight, toLight);
    let dist = sqrt(dist2);
    let lightDir = toLight / dist;

    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }

    if (traceShadow(hitPos + hitNorm * 0.001, lightDir, dist - 0.01)) { return vec3f(0.0); }

    return light.color * light.intensity * nDotL / dist2;
}

// ── Evaluate all scene lights at a hit point ────────────────────────

fn evaluateLighting(hitPos: vec3f, hitNorm: vec3f, wo: vec3f,
                    roughness: f32, metallic: f32, surfAlbedo: vec3f) -> vec3f {
    var total = vec3f(0.0);
    let count = traceParams.lightCount;
    for (var i = 0u; i < count; i++) {
        let light = sceneLights[i];
        if (light.lightType == LIGHT_DIRECTIONAL) {
            total += evaluateDirectionalLight(hitPos, hitNorm, light);
        } else if (light.lightType == LIGHT_AREA) {
            total += evaluateAreaLight(hitPos, hitNorm, light, wo, roughness, metallic, surfAlbedo);
        } else if (light.lightType == LIGHT_POINT) {
            total += evaluatePointLight(hitPos, hitNorm, light);
        }
    }
    return total;
}

// ── Volumetric fog (voxel-space ray-marched) ─────────────────────────────
// Analytically integrated extinction with jittered in-scattering samples.
// SVO shadow rays at each sample provide volumetric shadows / god rays.

fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
}

fn computeVoxelFog(rayOrigin: vec3f, rayDir: vec3f, hitDist: f32) -> vec4f {
    let density = traceParams.fogDensity;
    if (density <= 0.0) { return vec4f(0.0, 0.0, 0.0, 1.0); }

    let h = traceParams.fogHeightFalloff;
    let g = traceParams.fogAnisotropy;

    // Analytical height-integrated optical depth (closed-form for exp height fog)
    let a = h * rayDir.y;
    let baseTerm = density * exp(-h * rayOrigin.y);
    var opticalDepth: f32;
    if (abs(a) > 0.001) {
        opticalDepth = baseTerm * (1.0 - exp(-a * hitDist)) / a;
    } else {
        opticalDepth = baseTerm * hitDist;
    }
    let transmittance = exp(-max(opticalDepth, 0.0));

    // In-scattering: 4 jittered samples with SVO volumetric shadow checks
    let fogSteps = 4u;
    let stepLen = hitDist / f32(fogSteps);
    var inScatter = vec3f(0.0);

    for (var fi = 0u; fi < fogSteps; fi++) {
        let jitter = nextRandom();
        let t = (f32(fi) + jitter) / f32(fogSteps) * hitDist;
        let samplePos = rayOrigin + rayDir * t;
        let localDensity = density * exp(-h * samplePos.y);

        // Transmittance from camera to this sample
        var sampleOD: f32;
        if (abs(a) > 0.001) {
            sampleOD = baseTerm * (1.0 - exp(-a * t)) / a;
        } else {
            sampleOD = baseTerm * t;
        }
        let sampleT = exp(-max(sampleOD, 0.0));

        // Evaluate in-scattering from each light
        for (var li = 0u; li < traceParams.lightCount; li++) {
            let light = sceneLights[li];
            var lDir: vec3f;
            var lColor = light.color * light.intensity;
            var maxDist = 1e30;

            if (light.lightType == LIGHT_DIRECTIONAL) {
                lDir = normalize(-light.position);
            } else {
                let toLight = light.position - samplePos;
                let dist2 = dot(toLight, toLight);
                let dist = sqrt(dist2);
                lDir = toLight / dist;
                lColor /= dist2;
                maxDist = dist;
            }

            let phase = henyeyGreenstein(dot(rayDir, lDir), g);

            // SVO shadow check — volumetric shadows / god rays
            var shadowRay: Ray;
            shadowRay.origin = samplePos;
            shadowRay.dir = lDir;
            if (!traceBVHShadow(shadowRay, maxDist)) {
                inScatter += sampleT * localDensity * lColor * phase * stepLen;
            }
        }
    }

    return vec4f(inScatter, transmittance);
}

// ── Multi-bounce path tracer with NEE ─────────────────────────────────
// Returns irradiance at startPos. At each vertex: NEE for direct light,
// then stochastic PBR bounce (specular or diffuse based on metallic/Fresnel).

fn tracePath(startPos: vec3f, startNorm: vec3f, skipFirstNEE: u32,
             startWo: vec3f, startRoughness: f32, startMetallic: f32,
             startAlbedo: vec3f) -> vec3f {
    var accumulated = vec3f(0.0);
    if (skipFirstNEE == 0u) {
        accumulated = evaluateLighting(startPos, startNorm, startWo,
                                       startRoughness, startMetallic, startAlbedo);
    }
    var throughput = vec3f(1.0);
    var pos = startPos;
    var norm = startNorm;
    var sampleSpec = false;
    var specRoughness = 1.0f;
    var incomingDir = vec3f(0.0);
    // Track previous-vertex material for BSDF pdf at emissive hits
    var prevWo = startWo;
    var prevRoughness = startRoughness;
    var prevMetallic = startMetallic;
    var prevAlbedo = startAlbedo;
    let maxBounces = traceParams.maxBounces;

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        // Probe fallback: after bounce 0, use probe grid instead of tracing further
        if (traceParams.useProbes != 0u && bounce >= 1u && !sampleSpec) {
            accumulated += throughput * sampleProbeGrid(pos + norm * 0.01, norm);
            break;
        }

        var bounceRay: Ray;
        bounceRay.origin = pos + norm * 0.001;

        if (sampleSpec) {
            let h = sampleGGX(norm, max(specRoughness, 0.02), nextRandom(), nextRandom());
            bounceRay.dir = reflect(incomingDir, h);
            if (dot(bounceRay.dir, norm) <= 0.0) { break; }
        } else {
            bounceRay.dir = cosineSampleHemisphere(norm, nextRandom(), nextRandom());
        }

        let hit = traceBVH(bounceRay);
        if (!hit.hit) {
            accumulated += throughput * traceParams.ambientColor;
            break;
        }

        let mat = materials[hit.matIndex];

        // MIS: when BSDF hits emissive geometry, evaluate emission with MIS weight
        if (mat.emissiveIntensity > 0.0) {
            let emission = mat.emissive * mat.emissiveIntensity;
            let hitCos = abs(dot(hit.worldNorm, -bounceRay.dir));
            let dist2 = hit.t * hit.t;
            let triArea = getWorldTriArea(hit.triIndex, hit.instanceId);
            let pdfL = dist2 / max(hitCos * triArea, 1e-8);
            let pdfB = pdfBSDF(norm, prevWo, bounceRay.dir, prevRoughness, prevMetallic, prevAlbedo);
            let w = powerHeuristic(pdfB, pdfL);
            accumulated += throughput * emission * w;
        }

        if (mat.emissiveIntensity > 0.0 || mat.transmission > 0.5) { break; }

        // PBR: decide next bounce from this surface
        let cosTheta = abs(dot(hit.worldNorm, -bounceRay.dir));
        let F0 = mix(vec3f(0.04), mat.albedo, mat.metallic);
        let F = fresnelSchlick(cosTheta, F0);
        let specAvg = (F.r + F.g + F.b) / 3.0;
        let diffW = (1.0 - specAvg) * (1.0 - mat.metallic);
        let totalW = specAvg + diffW;
        let specProb = clamp(specAvg / max(totalW, 0.001), 0.05, 0.95);

        let hitWo = -bounceRay.dir;

        if (nextRandom() < specProb) {
            throughput *= F / specProb;
            sampleSpec = true;
            specRoughness = mat.roughness;
        } else {
            throughput *= mat.albedo * (vec3f(1.0) - F) * (1.0 - mat.metallic) / (1.0 - specProb);
            sampleSpec = false;
        }

        accumulated += throughput * evaluateLighting(hit.worldPos, hit.worldNorm, hitWo,
                                                     mat.roughness, mat.metallic, mat.albedo);

        // Russian roulette after first bounce
        if (bounce > 0u) {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if (p < 0.01 || nextRandom() > p) { break; }
            throughput /= p;
        }

        incomingDir = bounceRay.dir;
        pos = hit.worldPos;
        norm = hit.worldNorm;
        prevWo = hitWo;
        prevRoughness = mat.roughness;
        prevMetallic = mat.metallic;
        prevAlbedo = mat.albedo;
    }

    return accumulated;
}

fn traceRefraction(
    inRay: Ray,
    firstHit: HitInfo,
    mat: MaterialData,
) -> vec3f {
    var throughput = vec3f(1.0);
    var ray = inRay;
    var currentHit = firstHit;
    let maxBounces = u32(abs(mat.maxBounces));
    var insideMedium = false;

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let n = currentHit.worldNorm;
        // Geometric entering: ensures faceNorm always faces toward incoming ray
        let geomEntering = dot(ray.dir, n) < 0.0;
        let faceNorm = select(-n, n, geomEntering);
        // Medium tracking for eta: more robust than normal direction for complex meshes
        let mediaEntering = !insideMedium;
        let eta = select(mat.ior, 1.0 / mat.ior, mediaEntering);

        // Fresnel at each interface
        let cosI = abs(dot(ray.dir, faceNorm));
        let f0 = pow((1.0 - mat.ior) / (1.0 + mat.ior), 2.0);
        let fresnel = f0 + (1.0 - f0) * pow(1.0 - cosI, 5.0);

        let refracted = refract(ray.dir, faceNorm, eta);
        let tir = length(refracted) < 0.001;

        if (tir || nextRandom() < fresnel) {
            // Total internal reflection or Fresnel reflection — medium state unchanged
            let rh = sampleGGX(faceNorm, max(mat.roughness, 0.02), nextRandom(), nextRandom());
            ray.dir = reflect(ray.dir, rh);
            if (dot(ray.dir, faceNorm) <= 0.0) {
                ray.dir = reflect(inRay.dir, faceNorm); // fallback to perfect reflect
            }
            ray.origin = currentHit.worldPos + faceNorm * 0.01;
        } else {
            // Refraction: toggle medium state
            insideMedium = !insideMedium;
            // Refraction with roughness perturbation
            if (mat.roughness > 0.01) {
                let rh = sampleGGX(faceNorm, mat.roughness, nextRandom(), nextRandom());
                let perturbedRefract = refract(ray.dir, rh, eta);
                if (length(perturbedRefract) > 0.001) {
                    ray.dir = perturbedRefract;
                } else {
                    ray.dir = refracted;
                }
            } else {
                ray.dir = refracted;
            }
            ray.origin = currentHit.worldPos - faceNorm * 0.01;
        }

        let nextHit = traceBVH(ray);
        if (!nextHit.hit) {
            return throughput * traceParams.ambientColor;
        }

        let dist = nextHit.t;
        if (insideMedium) {
            throughput *= exp(-mat.absorptionColor * mat.absorptionDensity * dist);
        }

        let nextMat = materials[nextHit.matIndex];
        if (nextMat.transmission < 0.5) {
            // Exited transmissive medium — PBR shading at the exit surface
            let exitNorm = nextHit.worldNorm;
            let exitCos = abs(dot(ray.dir, exitNorm));
            let exitF0 = mix(vec3f(0.04), nextMat.albedo, nextMat.metallic);
            let exitF = fresnelSchlick(exitCos, exitF0);
            let exitSpecAvg = (exitF.r + exitF.g + exitF.b) / 3.0;
            let exitDiffW = (1.0 - exitSpecAvg) * (1.0 - nextMat.metallic);
            let exitTotalW = exitSpecAvg + exitDiffW;
            let exitSpecProb = clamp(exitSpecAvg / max(exitTotalW, 0.001), 0.05, 0.95);

            if (nextRandom() < exitSpecProb) {
                // Specular reflection off the exit surface (GGX)
                var mRay: Ray;
                let mH = sampleGGX(exitNorm, max(nextMat.roughness, 0.02), nextRandom(), nextRandom());
                mRay.dir = reflect(ray.dir, mH);
                if (dot(mRay.dir, exitNorm) <= 0.0) {
                    return throughput * tracePath(nextHit.worldPos, exitNorm, 0u,
                        -ray.dir, nextMat.roughness, nextMat.metallic, nextMat.albedo) * nextMat.albedo;
                }
                mRay.origin = nextHit.worldPos + exitNorm * 0.001;
                var mHit = traceBVH(mRay);
                // Follow specular bounces
                for (var mb = 0u; mb < 4u; mb++) {
                    if (!mHit.hit) { break; }
                    let mbMat = materials[mHit.matIndex];
                    let mbCos = abs(dot(mRay.dir, mHit.worldNorm));
                    let mbF0 = mix(vec3f(0.04), mbMat.albedo, mbMat.metallic);
                    let mbSpecAvg = (fresnelSchlick(mbCos, mbF0).r + fresnelSchlick(mbCos, mbF0).g + fresnelSchlick(mbCos, mbF0).b) / 3.0;
                    if (mbSpecAvg > 0.3 && mbMat.transmission < 0.5) {
                        let mbH = sampleGGX(mHit.worldNorm, max(mbMat.roughness, 0.02), nextRandom(), nextRandom());
                        mRay.dir = reflect(mRay.dir, mbH);
                        if (dot(mRay.dir, mHit.worldNorm) <= 0.0) { break; }
                        mRay.origin = mHit.worldPos + mHit.worldNorm * 0.001;
                        mHit = traceBVH(mRay);
                    } else {
                        break;
                    }
                }
                if (mHit.hit) {
                    let mMat = materials[mHit.matIndex];
                    return throughput * exitF / exitSpecProb * tracePath(mHit.worldPos, mHit.worldNorm, 0u,
                        -mRay.dir, mMat.roughness, mMat.metallic, mMat.albedo) * mMat.albedo;
                }
                return throughput * exitF / exitSpecProb * traceParams.ambientColor;
            } else {
                // Diffuse at exit surface
                let exitKd = (vec3f(1.0) - exitF) * (1.0 - nextMat.metallic);
                return throughput * tracePath(nextHit.worldPos, exitNorm, 0u,
                    -ray.dir, nextMat.roughness, nextMat.metallic, nextMat.albedo) * nextMat.albedo * exitKd / (1.0 - exitSpecProb);
            }
        }
        currentHit = nextHit;
    }
    _ = nextRandom();
    return throughput * traceParams.ambientColor;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= traceParams.width || gid.y >= traceParams.height) { return; }

    let coord = vec2u(gid.x, gid.y);

    // Map trace-res coord to full-res GBuffer coord via UV
    let uv = (vec2f(coord) + 0.5) / vec2f(f32(traceParams.width), f32(traceParams.height));

    // ── Direct voxel visualization mode ─────────────────────────────────
    if (traceParams.showVoxels != 0u) {
        let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 0.0, 1.0);
        let farNdc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, 1.0, 1.0);
        let nearWp = traceParams.invViewProj * ndc;
        let farWp = traceParams.invViewProj * farNdc;
        let nearPos = nearWp.xyz / nearWp.w;
        let farPos = farWp.xyz / farWp.w;

        var voxRay: Ray;
        voxRay.origin = traceParams.cameraPos;
        voxRay.dir = normalize(farPos - nearPos);

        let primaryHit = traceBVH(voxRay);
        if (!primaryHit.hit) {
            textureStore(giOutput, coord, vec4f(traceParams.ambientColor, 1.0));
            return;
        }

        initSampler(coord, traceParams.frameIndex, 0u);

        // Path tracing loop for voxel view (supports bounces)
        var throughput = vec3f(1.0);
        var radiance = vec3f(0.0);
        var currentHit = primaryHit;
        var currentDir = voxRay.dir;
        let maxBounces = traceParams.maxBounces;

        for (var bounce = 0u; bounce <= maxBounces; bounce++) {
            let mat = materials[currentHit.matIndex];
            let hitAlbedo = mat.albedo;
            let norm = currentHit.worldNorm;
            let wo = -currentDir;

            // Add emissive
            radiance += throughput * mat.emissive * mat.emissiveIntensity;

            // Evaluate direct lights
            var directLight = vec3f(0.0);
            for (var li = 0u; li < traceParams.lightCount; li++) {
                let light = sceneLights[li];
                if (light.lightType == LIGHT_DIRECTIONAL) {
                    directLight += evaluateDirectionalLight(currentHit.worldPos, norm, light);
                } else if (light.lightType == LIGHT_AREA) {
                    directLight += evaluateAreaLight(currentHit.worldPos, norm, light,
                        wo, max(mat.roughness, 0.3), mat.metallic, hitAlbedo);
                } else if (light.lightType == LIGHT_POINT) {
                    directLight += evaluatePointLight(currentHit.worldPos, norm, light);
                }
            }
            radiance += throughput * hitAlbedo * directLight;

            // If last bounce, no need to trace further
            if (bounce >= maxBounces) { break; }

            // ── Single-ray voxel indirect ──
            // 1 ray per pixel, mip-accelerated SVO traversal, temporal accumulation.
            // Offset origin by voxelSize along normal to clear surface voxel.
            let vp = getVoxelParams();
            let bounceDir = cosineSampleHemisphere(norm, nextRandom(), nextRandom());
            let NdotL = max(dot(norm, bounceDir), 0.0);
            if (NdotL < 0.001) { break; }

            throughput *= hitAlbedo;

            // Russian roulette after bounce 1
            if (bounce > 0u) {
                let rr = max(max(throughput.r, throughput.g), throughput.b);
                if (nextRandom() > rr) { break; }
                throughput /= rr;
            }

            var bounceRay: Ray;
            bounceRay.origin = currentHit.worldPos + norm * vp.voxelSize;
            bounceRay.dir = bounceDir;
            let bounceHit = traceBVH(bounceRay);
            if (!bounceHit.hit) {
                radiance += throughput * traceParams.ambientColor;
                break;
            }
            currentHit = bounceHit;
            currentDir = bounceDir;
        }

        radiance += traceParams.ambientColor * materials[primaryHit.matIndex].albedo * 0.1;

        // Volumetric fog between camera and primary hit
        if (traceParams.fogDensity > 0.0) {
            let fogResult = computeVoxelFog(traceParams.cameraPos, voxRay.dir, primaryHit.t);
            radiance = radiance * fogResult.a + fogResult.rgb;
        }

        textureStore(giOutput, coord, vec4f(radiance, 1.0));
        return;
    }
    let gbufDim = vec2u(textureDimensions(depthTex));
    let gbufCoord = vec2u(vec2f(gbufDim) * uv);
    let gbufClamped = min(gbufCoord, gbufDim - vec2u(1u));

    let depth = textureLoad(depthTex, gbufClamped, 0);
    if (depth >= 1.0) {
        textureStore(giOutput, coord, vec4f(0.0));
        return;
    }

    // Reconstruct world position from depth
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = traceParams.invViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    let normalData = textureLoad(normalTex, vec2i(gbufClamped), 0);
    let worldNormal = normalize(normalData.xyz * 2.0 - 1.0);
    let roughness = normalData.w;

    let albedoData = textureLoad(albedoTex, vec2i(gbufClamped), 0);
    let albedo = albedoData.rgb;
    let metallic = albedoData.a;

    let emissiveData = textureLoad(emissiveTex, vec2i(gbufClamped), 0);
    let transmission = emissiveData.a;

    let spp = traceParams.spp;
    var accumulated = vec3f(0.0);

    // Unified PBR: energy partition based on metallic, Fresnel, and transmission
    let viewDir = normalize(traceParams.cameraPos - worldPos);
    let NdotV = max(dot(worldNormal, viewDir), 0.001);
    let F0 = mix(vec3f(0.04), albedo, metallic);
    let F = fresnelSchlick(NdotV, F0);
    let specAvg = (F.r + F.g + F.b) / 3.0;

    // Three energy lobes: specular, transmission, diffuse
    let specW  = specAvg;
    let transW = (1.0 - specAvg) * (1.0 - metallic) * transmission;
    let diffW  = (1.0 - specAvg) * (1.0 - metallic) * (1.0 - transmission);
    let totalW = specW + transW + diffW;
    let specProb  = clamp(specW  / max(totalW, 0.001), 0.02, 0.98);
    let transProb = clamp(transW / max(totalW, 0.001), 0.0, 0.98 - specProb);
    // diffProb = 1.0 - specProb - transProb (implicit)

    for (var s = 0u; s < spp; s++) {
        initSampler(coord, traceParams.frameIndex, s);

        if (traceParams.rasterDirect != 0u && traceParams.useProbes != 0u) {
            // ── Raster direct mode: no ray tracing at all ────────────
            // Rasterizer handles direct light + reflections + refractions.
            // Probes provide indirect illumination only.
            let probePos = worldPos + worldNormal * 0.01;
            if (transmission > 0.5 || metallic > 0.5) {
                // Metallic/transmissive: evaluate SH in reflection direction
                let reflDir = reflect(-viewDir, worldNormal);
                let specEnv = sampleProbeGrid(probePos, reflDir);
                accumulated += (specEnv + traceParams.ambientColor) * albedo * F;
            } else {
                let kd = (vec3f(1.0) - F) * (1.0 - metallic);
                let indirect = sampleProbeGrid(probePos, worldNormal);
                accumulated += (indirect + traceParams.ambientColor) * albedo * kd;
            }
            continue;
        }

        let r = nextRandom();
        if (r < specProb) {
            // ── Specular reflection (GGX) ────────────────────────────
            let halfVec = sampleGGX(worldNormal, max(roughness, 0.02), nextRandom(), nextRandom());
            let specDir = reflect(-viewDir, halfVec);

            if (dot(specDir, worldNormal) > 0.0) {
                var specRay: Ray;
                specRay.origin = worldPos + worldNormal * 0.001;
                specRay.dir = specDir;
                var specHit = traceBVH(specRay);
                // Follow metallic bounces (transmissive surfaces handled via traceRefraction)
                for (var sk = 0u; sk < 4u; sk++) {
                    if (!specHit.hit) { break; }
                    let skMat = materials[specHit.matIndex];
                    if (skMat.metallic > 0.5 && skMat.transmission < 0.5) {
                        let h2 = sampleGGX(specHit.worldNorm, max(skMat.roughness, 0.02), nextRandom(), nextRandom());
                        specRay.dir = reflect(specRay.dir, h2);
                        if (dot(specRay.dir, specHit.worldNorm) <= 0.0) { break; }
                        specRay.origin = specHit.worldPos + specHit.worldNorm * 0.001;
                        specHit = traceBVH(specRay);
                    } else { break; }
                }
                if (specHit.hit) {
                    let specMat = materials[specHit.matIndex];
                    if (specMat.transmission > 0.5) {
                        accumulated += traceRefraction(specRay, specHit, specMat) * F / specProb;
                    } else {
                        accumulated += tracePath(specHit.worldPos, specHit.worldNorm, 0u,
                            -specRay.dir, specMat.roughness, specMat.metallic, specMat.albedo) * specMat.albedo * F / specProb;
                    }
                } else {
                    accumulated += traceParams.ambientColor * F / specProb;
                }
            }
        } else if (r < specProb + transProb) {
            // ── Transmission / refraction ────────────────────────────
            var primaryRay: Ray;
            primaryRay.origin = traceParams.cameraPos;
            primaryRay.dir = normalize(worldPos - traceParams.cameraPos);

            let primaryHit = traceBVH(primaryRay);
            if (primaryHit.hit) {
                let mat = materials[primaryHit.matIndex];
                let weight = (1.0 - specAvg) * (1.0 - metallic) * transmission / transProb;
                if (mat.transmission > 0.5) {
                    accumulated += traceRefraction(primaryRay, primaryHit, mat) * weight;
                } else {
                    // Primary ray hit an opaque surface: treat as diffuse contribution
                    accumulated += tracePath(primaryHit.worldPos, primaryHit.worldNorm, 0u,
                        -primaryRay.dir, mat.roughness, mat.metallic, mat.albedo) * mat.albedo * weight;
                }
            } else {
                let weight = (1.0 - specAvg) * (1.0 - metallic) * transmission / transProb;
                accumulated += traceParams.ambientColor * weight;
            }
        } else {
            // ── Diffuse: multi-bounce path tracing with NEE ──────────
            let diffProb = 1.0 - specProb - transProb;
            let kd = (vec3f(1.0) - F) * (1.0 - metallic) * (1.0 - transmission);
            if (traceParams.useProbes != 0u) {
                // Probes: NEE for direct, probe grid for indirect — no bounce tracing needed
                let direct = evaluateLighting(worldPos, worldNormal, viewDir, roughness, metallic, albedo);
                let indirect = sampleProbeGrid(worldPos + worldNormal * 0.01, worldNormal);
                accumulated += (direct + indirect) * albedo * kd / max(diffProb, 0.01);
            } else if (traceParams.useReSTIR != 0u) {
                // ReSTIR provides primary-surface direct light; skip first NEE in tracePath
                let restirDirect = textureLoad(restirDirectTex, vec2i(coord), 0).rgb;
                let indirect = tracePath(worldPos, worldNormal, 1u,
                    viewDir, roughness, metallic, albedo);
                accumulated += (restirDirect + indirect) * albedo * kd / max(diffProb, 0.01);
            } else {
                accumulated += tracePath(worldPos, worldNormal, 0u,
                    viewDir, roughness, metallic, albedo) * albedo * kd / max(diffProb, 0.01);
            }
        }
    }

    textureStore(giOutput, coord, vec4f(accumulated / f32(spp), 1.0));
}
`;
