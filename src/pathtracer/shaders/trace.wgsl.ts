export const traceShader = /* wgsl */`
// Requires: intersection.wgsl, traversal.wgsl, BVHNode struct, storage bindings

struct BVHNode {
    boundsMin  : vec3f,
    leftChild  : i32,
    boundsMax  : vec3f,
    rightChild : i32,
}

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
    sunDirection : vec2f, // packed as vec2 for alignment, actually vec3
    sunDirZ      : f32,
    sunColor     : vec3f,
    sunIntensity : f32,
}

@group(0) @binding(0)  var          depthTex   : texture_depth_2d;
@group(0) @binding(1)  var          normalTex  : texture_2d<f32>;
@group(0) @binding(2)  var          albedoTex  : texture_2d<f32>;
@group(0) @binding(3)  var          giOutput   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4)  var<uniform> traceParams : TraceParams;
@group(0) @binding(5)  var<storage, read> triangles  : array<f32>;
@group(0) @binding(6)  var<storage, read> blasNodes  : array<BVHNode>;
@group(0) @binding(7)  var<storage, read> tlasNodes  : array<BVHNode>;
@group(0) @binding(8)  var<storage, read> instances  : array<Instance>;
@group(0) @binding(9)  var<storage, read> materials  : array<MaterialData>;

// PCG hash for random number generation
fn pcgHash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn randomFloat(state: ptr<function, u32>) -> f32 {
    *state = pcgHash(*state);
    return f32(*state) / 4294967295.0;
}

// Cosine-weighted hemisphere sampling
fn cosineSampleHemisphere(n: vec3f, r1: f32, r2: f32) -> vec3f {
    let phi = 2.0 * 3.14159265 * r1;
    let cosTheta = sqrt(1.0 - r2);
    let sinTheta = sqrt(r2);

    // Build orthonormal basis from normal
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

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

    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let tangent = normalize(cross(up, n));
    let bitangent = cross(n, tangent);

    let h = normalize(
        tangent * cos(phi) * sinTheta +
        bitangent * sin(phi) * sinTheta +
        n * cosTheta
    );
    return h;
}

fn evaluateDirectLight(hitPos: vec3f, hitNorm: vec3f) -> vec3f {
    let sunDir = normalize(vec3f(traceParams.sunDirection, traceParams.sunDirZ));
    let nDotL = max(dot(hitNorm, -sunDir), 0.0);
    if (nDotL <= 0.0) { return vec3f(0.0); }

    // Shadow ray
    var shadowRay: Ray;
    shadowRay.origin = hitPos + hitNorm * 0.001;
    shadowRay.dir = -sunDir;
    let shadowHit = traceBVH(shadowRay);
    if (shadowHit.hit) { return vec3f(0.0); }

    return traceParams.sunColor * traceParams.sunIntensity * nDotL;
}

fn traceRefraction(
    inRay: Ray,
    firstHit: HitInfo,
    mat: MaterialData,
    rng: ptr<function, u32>
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
            // Total internal reflection
            ray.origin = currentHit.worldPos + faceNorm * 0.001;
            ray.dir = reflect(ray.dir, faceNorm);
        } else {
            ray.origin = currentHit.worldPos - faceNorm * 0.001;
            ray.dir = refracted;
        }

        let nextHit = traceBVH(ray);
        if (!nextHit.hit) { break; }

        // Beer's law absorption
        let dist = nextHit.t;
        throughput *= exp(-mat.absorptionColor * mat.absorptionDensity * dist);

        let nextMat = materials[nextHit.matIndex];
        if (u32(nextMat.flags) & 1u) == 0u {
            // Hit opaque surface — shade and return
            let lighting = evaluateDirectLight(nextHit.worldPos, nextHit.worldNorm);
            return throughput * lighting * nextMat.albedo;
        }
        currentHit = nextHit;
    }
    // Exhausted bounces — discard contribution
    _ = randomFloat(rng); // suppress unused warning
    return throughput * vec3f(0.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= traceParams.width || gid.y >= traceParams.height) { return; }

    let coord = vec2u(gid.x, gid.y);
    let depth = textureLoad(depthTex, coord, 0);
    if (depth >= 1.0) {
        textureStore(giOutput, coord, vec4f(0.0));
        return;
    }

    // Reconstruct world position from depth
    let uv = (vec2f(coord) + 0.5) / vec2f(f32(traceParams.width), f32(traceParams.height));
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = traceParams.invViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    let normalData = textureLoad(normalTex, coord, 0);
    let worldNormal = normalize(normalData.xyz * 2.0 - 1.0);
    let roughness = normalData.w;

    let albedoData = textureLoad(albedoTex, coord, 0);
    let metallic = albedoData.a;

    // RNG seed from pixel + frame
    var rng = pcgHash(gid.x + gid.y * traceParams.width + traceParams.frameIndex * 1000003u);

    // Cosine-weighted hemisphere sample for indirect lighting
    let r1 = randomFloat(&rng);
    let r2 = randomFloat(&rng);
    var ray: Ray;
    ray.origin = worldPos + worldNormal * 0.001;
    ray.dir = cosineSampleHemisphere(worldNormal, r1, r2);

    let hit = traceBVH(ray);

    var indirect = vec3f(0.0);

    // Sky/miss: light from escaped rays (dark for enclosed spaces).
    // Hit ambient: simulates multi-bounce fill — drives color bleed from unlit walls.
    let skyColor = vec3f(0.05);
    let hitAmbient = vec3f(0.4);

    if (hit.hit) {
        let mat = materials[hit.matIndex];

        // Check if refractive
        if ((u32(mat.flags) & 1u) != 0u) {
            indirect = traceRefraction(ray, hit, mat, &rng);
        } else {
            let hitDirect = evaluateDirectLight(hit.worldPos, hit.worldNorm);
            indirect = (hitDirect + hitAmbient) * mat.albedo + mat.emissive * mat.emissiveIntensity;
        }
    } else {
        indirect = skyColor;
    }

    // Specular ray for metallic/glossy surfaces
    if (metallic > 0.1 || roughness < 0.5) {
        let sr1 = randomFloat(&rng);
        let sr2 = randomFloat(&rng);
        let halfVec = sampleGGX(worldNormal, max(roughness, 0.04), sr1, sr2);
        let viewDir = normalize(traceParams.cameraPos - worldPos);
        let specDir = reflect(-viewDir, halfVec);

        if (dot(specDir, worldNormal) > 0.0) {
            var specRay: Ray;
            specRay.origin = worldPos + worldNormal * 0.001;
            specRay.dir = specDir;
            let specHit = traceBVH(specRay);
            if (specHit.hit) {
                let specMat = materials[specHit.matIndex];
                let specLighting = evaluateDirectLight(specHit.worldPos, specHit.worldNorm);
                let specular = specLighting * specMat.albedo;
                indirect = mix(indirect, specular, metallic);
            }
        }
    }

    textureStore(giOutput, coord, vec4f(indirect, 1.0));
}
`;
