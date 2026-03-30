export const probeDebugShader = /* wgsl */`
struct ProbeDebugParams {
    viewProj     : mat4x4f,
    invViewProj  : mat4x4f,
    cameraPos    : vec3f,
    stepX        : f32,
    gridMin      : vec3f,
    stepY        : f32,
    gridDims     : vec3u,
    stepZ        : f32,
    screenWidth  : u32,
    screenHeight : u32,
    probeRadius  : f32,
    _pad         : u32,
}

@group(0) @binding(0) var<uniform> params : ProbeDebugParams;
@group(0) @binding(1) var<storage, read> probeSH : array<vec4f>;
@group(0) @binding(2) var outputTex : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex  : texture_depth_2d;

const SH_STRIDE = 9u;

fn computeSHBasis(d: vec3f) -> array<f32, 9> {
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

fn evaluateSH(probeIdx: u32, dir: vec3f) -> vec3f {
    let base = probeIdx * SH_STRIDE;
    let sh = computeSHBasis(dir);
    var result = vec3f(0.0);
    for (var c = 0u; c < 9u; c++) {
        result += probeSH[base + c].xyz * sh[c];
    }
    return max(result, vec3f(0.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let totalProbes = params.gridDims.x * params.gridDims.y * params.gridDims.z;
    let probeIdx = gid.x;
    if (probeIdx >= totalProbes) { return; }

    // Compute probe world position from flat index
    let ix = probeIdx % params.gridDims.x;
    let iy = (probeIdx / params.gridDims.x) % params.gridDims.y;
    let iz = probeIdx / (params.gridDims.x * params.gridDims.y);
    let worldPos = params.gridMin + vec3f(
        f32(ix) * params.stepX,
        f32(iy) * params.stepY,
        f32(iz) * params.stepZ,
    );

    // Project to screen
    let clip = params.viewProj * vec4f(worldPos, 1.0);
    if (clip.w <= 0.0) { return; }
    let ndc = clip.xyz / clip.w;
    if (ndc.x < -1.0 || ndc.x > 1.0 || ndc.y < -1.0 || ndc.y > 1.0) { return; }

    let uv = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));
    let screenX = i32(uv.x * f32(params.screenWidth));
    let screenY = i32(uv.y * f32(params.screenHeight));

    // Camera-relative vectors for sphere shading
    let viewDir = normalize(params.cameraPos - worldPos);
    // Build tangent frame from viewDir (sphere faces camera)
    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(dot(viewDir, up)) > 0.99) { up = vec3f(1.0, 0.0, 0.0); }
    let right = normalize(cross(up, viewDir));
    let trueUp = normalize(cross(viewDir, right));

    // Draw sphere: for each pixel, compute world-space direction on sphere surface
    let r = i32(params.probeRadius);
    let r2 = r * r;
    let rf = f32(r);
    for (var dy = -r; dy <= r; dy++) {
        for (var dx = -r; dx <= r; dx++) {
            let d2 = dx * dx + dy * dy;
            if (d2 > r2) { continue; }
            let px = screenX + dx;
            let py = screenY + dy;
            if (px < 0 || py < 0 || px >= i32(params.screenWidth) || py >= i32(params.screenHeight)) { continue; }

            // Normalized position on disk [-1, 1]
            let nx = f32(dx) / rf;
            let ny = f32(-dy) / rf;  // flip Y: screen Y is down, world up is up

            // Map disk to sphere surface (hemisphere facing camera)
            let nz = sqrt(max(1.0 - nx * nx - ny * ny, 0.0));

            // World-space normal on sphere surface
            let sphereNormal = normalize(right * nx + trueUp * ny + viewDir * nz);

            // Evaluate SH irradiance in this direction
            let irradiance = evaluateSH(probeIdx, sphereNormal);

            // Tonemap
            let color = irradiance / (irradiance + vec3f(1.0));

            // Subtle sphere shading (darken edges)
            let edgeFade = nz * 0.5 + 0.5;

            // Thin outline for visibility
            let edge = f32(d2) / f32(r2);
            if (edge > 0.92) {
                textureStore(outputTex, vec2u(u32(px), u32(py)), vec4f(0.2, 0.2, 0.2, 1.0));
            } else {
                textureStore(outputTex, vec2u(u32(px), u32(py)), vec4f(color * edgeFade, 1.0));
            }
        }
    }
}
`;
