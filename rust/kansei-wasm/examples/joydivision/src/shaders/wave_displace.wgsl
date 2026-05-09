// Wave line displacement compute shader.
// Emits a RIBBON (2 vertices per sample: top + bottom) so lines have visible thickness.
// Each line has `vertsPerLine` samples → `vertsPerLine * 2` actual vertices.
// Writes full 9-float Vertex layout (pos4 + normal3 + uv2).

struct WaveParams {
    lineCount: u32,
    vertsPerLine: u32,
    halfWidth: f32,
    activeLineIdx: u32,
    lineSpacing: f32,
    fftAmplitude: f32,
    thickness: f32,
    noiseScale: f32,
    noiseStrength: f32,
    noiseSpeed: f32,
    noiseZoom: f32,
    noiseMix: f32,
    noiseTime: f32,
    peakExponent: f32,
    peakMin: f32,
    _pad3: u32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<f32>;
@group(0) @binding(1) var<storage, read> baseY: array<f32>;
@group(0) @binding(2) var elevationTex: texture_2d<f32>;
@group(0) @binding(3) var<uniform> params: WaveParams;
@group(0) @binding(4) var elevationSampler: sampler;

fn hash2d(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.x, p.y, p.x) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}
fn valueNoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2d(i), hash2d(i + vec2(1.0, 0.0)), u.x),
               mix(hash2d(i + vec2(0.0, 1.0)), hash2d(i + vec2(1.0, 1.0)), u.x), u.y);
}
fn fbm(p: vec2<f32>) -> f32 {
    var val = 0.0; var amp = 0.5; var pos = p;
    for (var i = 0; i < 4; i++) { val += amp * (valueNoise(pos) * 2.0 - 1.0); amp *= 0.5; pos *= 2.0; }
    return val;
}


@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let totalSamples = params.lineCount * params.vertsPerLine;
    if (idx >= totalSamples) { return; }

    let lineIdx = idx / params.vertsPerLine;
    let vertIdx = idx % params.vertsPerLine;

    // X: spread across line width
    let t = f32(vertIdx) / f32(params.vertsPerLine - 1u);
    let x = -params.halfWidth + t * params.halfWidth * 2.0;

    // Y: cyclic line reordering (matching text displacement)
    let N = params.lineCount;
    let activeLine = params.activeLineIdx;
    let visualSlot = (lineIdx + N - activeLine - 1u) % N;
    let totalHeight = f32(N - 1u) * params.lineSpacing;
    let yOffset = totalHeight * 0.5;
    let lastLine = f32(N - 1u);
    let y = -(lastLine - f32(visualSlot)) * params.lineSpacing + yOffset;

    // Z: sample elevation map with linear filtering
    let linesInMap = min(N, 256u);
    let mapY = visualSlot * (256u / linesInMap);
    let uv = vec2<f32>(t, f32(min(mapY, 255u)) / 255.0);
    let elevation = textureSampleLevel(elevationTex, elevationSampler, uv, 0.0).r;

    // Spatially coherent noise based on vertex grid position
    let noiseDist = (t - 0.5) * 2.0;
    let noisePeak = max(params.peakMin, exp(-params.peakExponent * noiseDist * noiseDist));
    let noiseCoord = vec2<f32>(t * 256.0, f32(mapY)) * params.noiseScale * params.noiseZoom
                   + vec2<f32>(0.0, params.noiseTime * params.noiseSpeed);
    let noise = (fbm(noiseCoord) * 0.5 + 0.5) * params.noiseStrength * noisePeak;
    let val = mix(elevation, noise, params.noiseMix);

    // Per-line Z layering: upper lines closer to camera
    let lineZ = f32(N - 1u - visualSlot) * 0.5;
    let z = lineZ + val * params.fftAmplitude;
    let halfThick = params.thickness * 0.5;

    // Write TWO vertices per sample (top + bottom of ribbon)
    // Each vertex = 9 floats, two vertices per sample = 18 floats
    let baseOff = idx * 18u;

    // Top vertex (y + halfThick)
    vertices[baseOff + 0u] = x;
    vertices[baseOff + 1u] = y + halfThick;
    vertices[baseOff + 2u] = z;
    vertices[baseOff + 3u] = 1.0;
    vertices[baseOff + 4u] = 0.0;
    vertices[baseOff + 5u] = 0.0;
    vertices[baseOff + 6u] = 1.0;
    vertices[baseOff + 7u] = t;
    vertices[baseOff + 8u] = 0.0;

    // Bottom vertex (y - halfThick)
    vertices[baseOff + 9u] = x;
    vertices[baseOff + 10u] = y - halfThick;
    vertices[baseOff + 11u] = z;
    vertices[baseOff + 12u] = 1.0;
    vertices[baseOff + 13u] = 0.0;
    vertices[baseOff + 14u] = 0.0;
    vertices[baseOff + 15u] = 1.0;
    vertices[baseOff + 16u] = t;
    vertices[baseOff + 17u] = 1.0;
}
