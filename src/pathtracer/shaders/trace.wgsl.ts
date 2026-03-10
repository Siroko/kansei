export const traceShader = /* wgsl */`
// Requires: intersection.wgsl, traversal.wgsl, storage bindings

struct MaterialData {
    albedo       : vec3f,
    roughness    : f32,
    metallic     : f32,
    ior          : f32,
    maxBounces   : f32,
    flags        : f32,  // bit 0 = refractive
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
        if ((u32(hitMat.flags) & 1u) == 0u) { return true; }
        // Transparent surface: advance past it
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

fn evaluateAreaLight(hitPos: vec3f, hitNorm: vec3f, light: LightData) -> vec3f {
    let lr1 = nextRandom();
    let lr2 = nextRandom();
    let sizeX = light.extra.x;
    let sizeZ = light.extra.y;

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

    // Cosine at hit surface
    let nDotL = max(dot(hitNorm, lightDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }

    // Cosine at light surface
    let lightCos = max(dot(ln, -lightDir), 0.0);
    if (lightCos <= 0.0) { return vec3f(0.0); }

    if (traceShadow(hitPos + hitNorm * 0.001, lightDir, dist - 0.01)) { return vec3f(0.0); }

    // L_e * cos_hit * cos_light * A / d^2
    let area = sizeX * sizeZ;
    return light.color * light.intensity * nDotL * lightCos * area / dist2;
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

fn evaluateLighting(hitPos: vec3f, hitNorm: vec3f) -> vec3f {
    var total = vec3f(0.0);
    let count = traceParams.lightCount;
    for (var i = 0u; i < count; i++) {
        let light = sceneLights[i];
        if (light.lightType == LIGHT_DIRECTIONAL) {
            total += evaluateDirectionalLight(hitPos, hitNorm, light);
        } else if (light.lightType == LIGHT_AREA) {
            total += evaluateAreaLight(hitPos, hitNorm, light);
        } else if (light.lightType == LIGHT_POINT) {
            total += evaluatePointLight(hitPos, hitNorm, light);
        }
    }
    return total;
}

// ── Multi-bounce path tracer with NEE ─────────────────────────────────
// Returns irradiance at startPos (primary surface albedo applied by composite).
// At each path vertex: sample all lights (NEE), then cosine-weighted random
// bounce. Throughput accumulates bounce-surface albedos. Emissive surfaces
// hit by random bounces are skipped (already counted by NEE).

fn tracePath(startPos: vec3f, startNorm: vec3f) -> vec3f {
    // NEE at primary vertex
    var accumulated = evaluateLighting(startPos, startNorm);
    var throughput = vec3f(1.0);
    var pos = startPos;
    var norm = startNorm;
    let maxBounces = traceParams.maxBounces;

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let r1 = nextRandom();
        let r2 = nextRandom();
        var bounceRay: Ray;
        bounceRay.origin = pos + norm * 0.001;
        bounceRay.dir = cosineSampleHemisphere(norm, r1, r2);

        let hit = traceBVH(bounceRay);
        if (!hit.hit) {
            // Add ambient/sky contribution when ray escapes
            accumulated += throughput * traceParams.ambientColor;
            break;
        }

        let mat = materials[hit.matIndex];

        // Stop on emissive (NEE already samples), glass, or mirror —
        // tracing through these on indirect bounces costs extra BVH
        // traversals for negligible visual contribution
        if (mat.emissiveIntensity > 0.0 || (u32(mat.flags) & 1u) != 0u || mat.metallic > 0.5) { break; }

        throughput *= mat.albedo;
        accumulated += throughput * evaluateLighting(hit.worldPos, hit.worldNorm);

        // Russian roulette after first bounce
        if (bounce > 0u) {
            let p = max(throughput.r, max(throughput.g, throughput.b));
            if (p < 0.01 || nextRandom() > p) { break; }
            throughput /= p;
        }

        pos = hit.worldPos;
        norm = hit.worldNorm;
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

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let n = currentHit.worldNorm;
        let entering = dot(ray.dir, n) < 0.0;
        let faceNorm = select(-n, n, entering);
        let eta = select(mat.ior, 1.0 / mat.ior, entering);

        let refracted = refract(ray.dir, faceNorm, eta);
        if (length(refracted) < 0.001) {
            ray.origin = currentHit.worldPos + faceNorm * 0.001;
            ray.dir = reflect(ray.dir, faceNorm);
        } else {
            ray.origin = currentHit.worldPos - faceNorm * 0.001;
            ray.dir = refracted;
        }

        let nextHit = traceBVH(ray);
        if (!nextHit.hit) { break; }

        let dist = nextHit.t;
        throughput *= exp(-mat.absorptionColor * mat.absorptionDensity * dist);

        let nextMat = materials[nextHit.matIndex];
        if ((u32(nextMat.flags) & 1u) == 0u) {
            // Exited glass onto non-refractive surface
            if (nextMat.metallic > 0.9) {
                // Mirror: follow reflection bounces to show mirror reflection through glass
                var mRay: Ray;
                mRay.dir = reflect(ray.dir, nextHit.worldNorm);
                mRay.origin = nextHit.worldPos + nextHit.worldNorm * 0.001;
                var mHit = traceBVH(mRay);
                for (var mb = 0u; mb < 4u; mb++) {
                    if (!mHit.hit) { break; }
                    let mbMat = materials[mHit.matIndex];
                    if (mbMat.metallic > 0.9 && (u32(mbMat.flags) & 1u) == 0u) {
                        mRay.dir = reflect(mRay.dir, mHit.worldNorm);
                        mRay.origin = mHit.worldPos + mHit.worldNorm * 0.001;
                        mHit = traceBVH(mRay);
                    } else {
                        break;
                    }
                }
                if (mHit.hit) {
                    let mMat = materials[mHit.matIndex];
                    return throughput * tracePath(mHit.worldPos, mHit.worldNorm) * mMat.albedo;
                }
                return throughput * vec3f(0.0);
            }
            return throughput * tracePath(nextHit.worldPos, nextHit.worldNorm) * nextMat.albedo;
        }
        currentHit = nextHit;
    }
    _ = nextRandom();
    return throughput * vec3f(0.0);
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
    let isRefractive = albedoData.a < 0.02;
    let metallic = albedoData.a;

    let spp = traceParams.spp;
    var accumulated = vec3f(0.0);

    for (var s = 0u; s < spp; s++) {
        initSampler(coord, traceParams.frameIndex, s);

        if (isRefractive) {
            // Primary refraction: trace ray from camera through this surface
            var primaryRay: Ray;
            primaryRay.origin = traceParams.cameraPos;
            primaryRay.dir = normalize(worldPos - traceParams.cameraPos);

            let primaryHit = traceBVH(primaryRay);
            if (primaryHit.hit) {
                let mat = materials[primaryHit.matIndex];
                if ((u32(mat.flags) & 1u) != 0u) {
                    accumulated += traceRefraction(primaryRay, primaryHit, mat);
                } else {
                    accumulated += tracePath(primaryHit.worldPos, primaryHit.worldNorm) * mat.albedo;
                }
            }
        } else if (metallic > 0.9) {
            // Mirror surface: trace specular ray to find reflected object,
            // then evaluate it as a diffuse-lit surface (direct + indirect * albedo)
            // so reflections show the object's "raster + GI" appearance.
            let sr1 = nextRandom();
            let sr2 = nextRandom();
            let halfVec = sampleGGX(worldNormal, max(roughness, 0.04), sr1, sr2);
            let viewDir = normalize(traceParams.cameraPos - worldPos);
            let specDir = reflect(-viewDir, halfVec);

            if (dot(specDir, worldNormal) > 0.0) {
                var specRay: Ray;
                specRay.origin = worldPos + worldNormal * 0.001;
                specRay.dir = specDir;
                var specHit = traceBVH(specRay);
                // Skip through glass/mirrors to find the first diffuse-like surface
                for (var sk = 0u; sk < 4u; sk++) {
                    if (!specHit.hit) { break; }
                    let skMat = materials[specHit.matIndex];
                    if ((u32(skMat.flags) & 1u) != 0u) {
                        specRay.origin = specHit.worldPos + specRay.dir * 0.002;
                        specHit = traceBVH(specRay);
                    } else if (skMat.metallic > 0.9) {
                        specRay.dir = reflect(specRay.dir, specHit.worldNorm);
                        specRay.origin = specHit.worldPos + specHit.worldNorm * 0.001;
                        specHit = traceBVH(specRay);
                    } else { break; }
                }
                if (specHit.hit) {
                    let specMat = materials[specHit.matIndex];
                    if ((u32(specMat.flags) & 1u) != 0u) {
                        accumulated += traceRefraction(specRay, specHit, specMat);
                    } else {
                        accumulated += tracePath(specHit.worldPos, specHit.worldNorm) * specMat.albedo;
                    }
                }
            }
        } else {
            // Diffuse surface: multi-bounce path tracing with NEE
            accumulated += tracePath(worldPos, worldNormal);
        }
    }

    let giAlpha = select(1.0, 0.0, isRefractive || metallic > 0.9);
    textureStore(giOutput, coord, vec4f(accumulated / f32(spp), giAlpha));
}
`;
