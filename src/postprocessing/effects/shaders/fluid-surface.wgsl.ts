export const shaderCode = /* wgsl */`

struct SurfaceParams {
    invViewProj : mat4x4<f32>,
    cameraPos   : vec3<f32>,
    densityThreshold: f32,
    boundsMin   : vec3<f32>,
    absorption  : f32,
    boundsMax   : vec3<f32>,
    densityScale: f32,
    fluidColor  : vec3<f32>,
    stepCount   : u32,
    screenWidth : u32,
    screenHeight: u32,
    _pad0       : u32,
    _pad1       : u32,
};

@group(0) @binding(0) var inputTex   : texture_2d<f32>;
@group(0) @binding(1) var depthTex   : texture_depth_2d;
@group(0) @binding(2) var outputTex  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var densityTex : texture_3d<f32>;
@group(0) @binding(4) var densitySamp: sampler;
@group(0) @binding(5) var<uniform> params: SurfaceParams;

fn worldToUVW(worldPos: vec3<f32>) -> vec3<f32> {
    return (worldPos - params.boundsMin) / (params.boundsMax - params.boundsMin);
}

fn sampleDensity(worldPos: vec3<f32>) -> f32 {
    let uvw = worldToUVW(worldPos);
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 0.0; }
    return textureSampleLevel(densityTex, densitySamp, uvw, 0.0).r * params.densityScale;
}

fn computeNormal(worldPos: vec3<f32>) -> vec3<f32> {
    let eps = length(params.boundsMax - params.boundsMin) / 128.0;
    let dx = sampleDensity(worldPos + vec3<f32>(eps, 0.0, 0.0)) - sampleDensity(worldPos - vec3<f32>(eps, 0.0, 0.0));
    let dy = sampleDensity(worldPos + vec3<f32>(0.0, eps, 0.0)) - sampleDensity(worldPos - vec3<f32>(0.0, eps, 0.0));
    let dz = sampleDensity(worldPos + vec3<f32>(0.0, 0.0, eps)) - sampleDensity(worldPos - vec3<f32>(0.0, 0.0, eps));
    let n = vec3<f32>(dx, dy, dz);
    let l = length(n);
    if (l < 0.0001) { return vec3<f32>(0.0, 1.0, 0.0); }
    return -n / l;
}

fn intersectAABB(rayOrigin: vec3<f32>, rayDir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    let invDir = 1.0 / rayDir;
    let t1 = (boxMin - rayOrigin) * invDir;
    let t2 = (boxMax - rayOrigin) * invDir;
    let tNear = min(t1, t2);
    let tFar  = max(t1, t2);
    let tMin = max(max(tNear.x, tNear.y), tNear.z);
    let tMax = min(min(tFar.x, tFar.y), tFar.z);
    return vec2<f32>(max(tMin, 0.0), tMax);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screenWidth || gid.y >= params.screenHeight) { return; }

    let coord = vec2<i32>(gid.xy);
    let sceneColor = textureLoad(inputTex, coord, 0);
    let sceneDepth = textureLoad(depthTex, coord, 0);

    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.screenWidth), f32(params.screenHeight));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let nearClip = params.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
    let farClip  = params.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
    let nearWorld = nearClip.xyz / nearClip.w;
    let farWorld  = farClip.xyz / farClip.w;
    let rayDir = normalize(farWorld - nearWorld);
    let rayOrigin = params.cameraPos;

    let tRange = intersectAABB(rayOrigin, rayDir, params.boundsMin, params.boundsMax);
    if (tRange.x >= tRange.y) {
        textureStore(outputTex, coord, sceneColor);
        return;
    }

    let maxT = min(tRange.y, length(farWorld - nearWorld) * sceneDepth);
    let stepSize = (tRange.y - tRange.x) / f32(params.stepCount);

    // Jitter start to break banding
    let hash = fract(sin(dot(vec2<f32>(gid.xy), vec2<f32>(12.9898, 78.233))) * 43758.5453);
    let tStart = tRange.x + hash * stepSize;

    var transmittance = 1.0;
    var hitPos = vec3<f32>(0.0);
    var hit = false;
    var prevDensity = 0.0;
    var prevT = tStart;

    for (var i = 0u; i < params.stepCount; i++) {
        let t = tStart + f32(i) * stepSize;
        if (t > maxT) { break; }

        let samplePos = rayOrigin + rayDir * t;
        let density = sampleDensity(samplePos);

        if (density > params.densityThreshold && prevDensity <= params.densityThreshold) {
            // Binary search refinement (5 iterations)
            var lo = prevT;
            var hi = t;
            for (var r = 0u; r < 5u; r++) {
                let mid = (lo + hi) * 0.5;
                let midD = sampleDensity(rayOrigin + rayDir * mid);
                if (midD > params.densityThreshold) { hi = mid; } else { lo = mid; }
            }
            hitPos = rayOrigin + rayDir * (lo + hi) * 0.5;
            hit = true;
            break;
        }

        transmittance *= exp(-density * params.absorption * stepSize);
        if (transmittance < 0.01) { break; }

        prevDensity = density;
        prevT = t;
    }

    var finalColor = sceneColor.rgb;

    if (hit) {
        let normal = computeNormal(hitPos);

        let lightDir = normalize(vec3<f32>(0.3, 1.0, 0.5));
        let diffuse = max(dot(normal, lightDir), 0.0);
        let ambient = 0.2;
        let specDir = reflect(-lightDir, normal);
        let viewDir = normalize(params.cameraPos - hitPos);
        let spec = pow(max(dot(specDir, viewDir), 0.0), 64.0) * 0.6;

        // Fresnel-like rim
        let fresnel = pow(1.0 - max(dot(normal, viewDir), 0.0), 3.0) * 0.4;

        let surfaceColor = params.fluidColor * (ambient + diffuse) + vec3<f32>(spec + fresnel);
        finalColor = mix(surfaceColor, sceneColor.rgb, transmittance);
    } else if (transmittance < 0.99) {
        // Grazing rays: use the same fluid color with ambient shading
        // so near-surface fog looks like the fluid, not black
        let fogColor = params.fluidColor * 0.35;
        finalColor = mix(fogColor, sceneColor.rgb, transmittance);
    }

    textureStore(outputTex, coord, vec4<f32>(finalColor, sceneColor.a));
}
`;
