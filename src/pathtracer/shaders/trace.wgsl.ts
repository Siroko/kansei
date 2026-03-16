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
    _pad1        : u32,
    ambientColor : vec3f,
    _pad2        : u32,
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
@group(0) @binding(5)  var<storage, read> triangles  : array<f32>;
@group(0) @binding(6)  var<storage, read> bvh4Nodes  : array<vec4f>;
@group(0) @binding(7)  var<storage, read> tlasBvh4Nodes : array<vec4f>;
@group(0) @binding(8)  var<storage, read> instances  : array<Instance>;
@group(0) @binding(9)  var<storage, read> materials  : array<MaterialData>;
@group(0) @binding(10) var<storage, read> sceneLights : array<LightData>;
@group(0) @binding(11) var<storage, read> blueNoise   : array<f32>;
@group(0) @binding(12) var          emissiveTex : texture_2d<f32>;

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

// World-space triangle area for MIS on emissive geometry hits
fn getWorldTriArea(triIdx: u32, instId: u32) -> f32 {
    let base = triIdx * TRI_STRIDE;
    let v0 = vec3f(triangles[base], triangles[base+1u], triangles[base+2u]);
    let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
    let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);
    let inst = instances[instId];
    let w0 = transformPointToWorld(v0, inst);
    let w1 = transformPointToWorld(v1, inst);
    let w2 = transformPointToWorld(v2, inst);
    return 0.5 * length(cross(w1 - w0, w2 - w0));
}

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
    let maxBounces = u32(mat.maxBounces);
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
            accumulated += tracePath(worldPos, worldNormal, 0u,
                viewDir, roughness, metallic, albedo) * albedo * kd / max(diffProb, 0.01);
        }
    }

    // giAlpha=0: trace shader computes full outgoing radiance for all paths
    textureStore(giOutput, coord, vec4f(accumulated / f32(spp), 0.0));
}
`;
