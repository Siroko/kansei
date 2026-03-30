export const probeTraceShader = /* wgsl */`
// Requires: intersection.wgsl, traversal.wgsl concatenated before this

struct ProbeTraceParams {
    gridMin      : vec3f,
    gridStepX    : f32,
    gridDims     : vec3u,
    gridStepY    : f32,
    gridStepZ    : f32,
    raysPerProbe : u32,
    frameIndex   : u32,
    lightCount   : u32,
    maxDistance   : f32,
    _pad0        : f32,
    _pad1        : f32,
    _pad2        : f32,
}

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

@group(0) @binding(0) var<uniform> params       : ProbeTraceParams;
@group(0) @binding(1) var<storage, read> triangles     : array<f32>;
@group(0) @binding(2) var<storage, read> bvh4Nodes     : array<vec4f>;
@group(0) @binding(3) var<storage, read> tlasBvh4Nodes : array<vec4f>;
@group(0) @binding(4) var<storage, read> instances     : array<Instance>;
@group(0) @binding(5) var<storage, read> materials     : array<MaterialData>;
@group(0) @binding(6) var<storage, read> sceneLights   : array<LightData>;
@group(0) @binding(7) var<storage, read_write> rayResults : array<vec4f>;
@group(0) @binding(8) var<storage, read> probeSHPrev   : array<vec4f>;

// ── PCG random ──
var<private> rngState: u32;

fn pcgHash(input: u32) -> u32 {
    let state = input * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn nextRandom() -> f32 {
    rngState = pcgHash(rngState);
    return f32(rngState) / 4294967295.0;
}

// ── Spherical Fibonacci for uniform sphere sampling ──
const GOLDEN_RATIO = 1.6180339887;

fn sphericalFibonacci(idx: u32, total: u32) -> vec3f {
    let i = f32(idx) + 0.5;
    let n = f32(total);
    let phi = 2.0 * 3.14159265 * (i / GOLDEN_RATIO - floor(i / GOLDEN_RATIO));
    let cosTheta = 1.0 - 2.0 * i / n;
    let sinTheta = sqrt(max(1.0 - cosTheta * cosTheta, 0.0));
    return vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

fn probeWorldPos(ix: u32, iy: u32, iz: u32) -> vec3f {
    return params.gridMin + vec3f(
        f32(ix) * params.gridStepX,
        f32(iy) * params.gridStepY,
        f32(iz) * params.gridStepZ,
    );
}

fn probeIndexToCoord(idx: u32) -> vec3u {
    let x = idx % params.gridDims.x;
    let y = (idx / params.gridDims.x) % params.gridDims.y;
    let z = idx / (params.gridDims.x * params.gridDims.y);
    return vec3u(x, y, z);
}

// ── Probe grid sampling (previous frame's SH for multi-bounce) ──
const SH_STRIDE = 9u;

fn computeSHBasisProbe(d: vec3f) -> array<f32, 9> {
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

fn evaluateProbeSHPrev(probeIdx: u32, dir: vec3f) -> vec3f {
    let base = probeIdx * SH_STRIDE;
    let sh = computeSHBasisProbe(dir);
    var result = vec3f(0.0);
    for (var c = 0u; c < 9u; c++) {
        result += probeSHPrev[base + c].xyz * sh[c];
    }
    return max(result, vec3f(0.0));
}

fn sampleProbeGridPrev(worldPos: vec3f, normal: vec3f) -> vec3f {
    let dims = params.gridDims;
    let gridStep = vec3f(params.gridStepX, params.gridStepY, params.gridStepZ);
    let localPos = (worldPos - params.gridMin) / gridStep;
    let baseCoord = vec3i(floor(localPos));
    let frac = localPos - vec3f(baseCoord);

    var totalIrradiance = vec3f(0.0);
    var totalWeight = 0.0;

    for (var dz = 0; dz <= 1; dz++) {
        for (var dy = 0; dy <= 1; dy++) {
            for (var dx = 0; dx <= 1; dx++) {
                let probeCoord = baseCoord + vec3i(dx, dy, dz);
                let clamped = clamp(probeCoord, vec3i(0), vec3i(dims) - vec3i(1));
                let probeIdx = u32(clamped.z) * dims.x * dims.y + u32(clamped.y) * dims.x + u32(clamped.x);

                let t = vec3f(f32(dx), f32(dy), f32(dz));
                let trilinear = (1.0 - abs(t.x - frac.x)) * (1.0 - abs(t.y - frac.y)) * (1.0 - abs(t.z - frac.z));

                if (trilinear <= 0.0) { continue; }

                let irradiance = evaluateProbeSHPrev(probeIdx, normal);
                totalIrradiance += irradiance * trilinear;
                totalWeight += trilinear;
            }
        }
    }

    if (totalWeight > 0.0) {
        return totalIrradiance / totalWeight;
    }
    return vec3f(0.0);
}

// ── Simple direct lighting evaluation for probe rays ──
fn evaluateDirectLighting(hitPos: vec3f, hitNorm: vec3f) -> vec3f {
    var total = vec3f(0.0);
    for (var i = 0u; i < params.lightCount; i++) {
        let light = sceneLights[i];
        var lightDir: vec3f;
        var dist: f32;
        var intensity: vec3f;

        if (light.lightType == LIGHT_DIRECTIONAL) {
            lightDir = normalize(-light.position);
            dist = 1e30;
            intensity = light.color * light.intensity;
        } else if (light.lightType == LIGHT_AREA) {
            // Sample center of area light
            let toLight = light.position - hitPos;
            dist = length(toLight);
            lightDir = toLight / dist;
            let cosLight = max(dot(-lightDir, light.normal), 0.0);
            let area = light.extra.x * light.extra.y;
            intensity = light.color * light.intensity * cosLight * area / (dist * dist);
        } else {
            let toLight = light.position - hitPos;
            dist = length(toLight);
            lightDir = toLight / dist;
            intensity = light.color * light.intensity / (dist * dist);
        }

        let NdotL = max(dot(hitNorm, lightDir), 0.0);
        if (NdotL <= 0.0) { continue; }

        // Shadow test
        var shadowRay: Ray;
        shadowRay.origin = hitPos + hitNorm * 0.001;
        shadowRay.dir = lightDir;
        let shadowHit = traceBVH(shadowRay);
        if (shadowHit.hit && shadowHit.t < dist - 0.01) { continue; }

        total += intensity * NdotL;
    }
    return total;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let totalProbes = params.gridDims.x * params.gridDims.y * params.gridDims.z;
    let totalRays = totalProbes * params.raysPerProbe;
    let rayIdx = gid.x;
    if (rayIdx >= totalRays) { return; }

    let probeIdx = rayIdx / params.raysPerProbe;
    let localRay = rayIdx % params.raysPerProbe;

    // Initialize RNG
    rngState = pcgHash(rayIdx + params.frameIndex * 1000003u);

    let coord = probeIndexToCoord(probeIdx);
    let probePos = probeWorldPos(coord.x, coord.y, coord.z);

    // Randomized spherical Fibonacci: rotate base direction by frame-varying random
    let baseDir = sphericalFibonacci(localRay, params.raysPerProbe);
    // Apply random rotation per frame for temporal variation
    let angle = f32(params.frameIndex) * GOLDEN_RATIO * 2.0 * 3.14159265;
    let ca = cos(angle);
    let sa = sin(angle);
    let rotDir = vec3f(
        baseDir.x * ca - baseDir.z * sa,
        baseDir.y,
        baseDir.x * sa + baseDir.z * ca,
    );

    var ray: Ray;
    ray.origin = probePos;
    ray.dir = normalize(rotDir);

    // Trace with transparency: skip transmissive surfaces (e.g. glass particles)
    // so probes capture the actual scene indirect light, not dark particle occlusion
    var radiance = vec3f(0.0);
    var distance = params.maxDistance;
    var currentRay = ray;
    var bounces = 0u;

    loop {
        if (bounces >= 4u) { break; }
        let hit = traceBVH(currentRay);
        if (!hit.hit) { break; }

        let mat = materials[hit.matIndex];

        // Probe-invisible material (maxBounces < 0) or transmissive — skip through
        if (mat.maxBounces < 0.0 || mat.transmission >= 0.5) {
            currentRay.origin = hit.worldPos + currentRay.dir * 0.01;
            bounces++;
            continue;
        }

        distance = hit.t;

        // Emissive surfaces
        if (mat.emissiveIntensity > 0.0) {
            radiance = mat.emissive * mat.emissiveIntensity;
        }

        // Direct lighting at hit point (Lambertian)
        let direct = evaluateDirectLighting(hit.worldPos, hit.worldNorm);

        // Multi-bounce: sample previous frame's probe grid for indirect at hit point
        let indirect = sampleProbeGridPrev(hit.worldPos + hit.worldNorm * 0.01, hit.worldNorm);

        radiance += (direct + indirect) * mat.albedo;
        break;
    }

    // Store: [radiance.rgb, distance], [rayDir.xyz, 0]
    let outBase = rayIdx * 2u;
    rayResults[outBase + 0u] = vec4f(radiance, distance);
    rayResults[outBase + 1u] = vec4f(ray.dir, 0.0);
}
`;
