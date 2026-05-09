// FFT displacement compute shader.
// Runs per-particle: displaces Y position based on FFT frequency data.

struct Params {
    particleCount: u32,
    lineCount: u32,
    activeLineIdx: u32,
    scrollY: f32,
    fftAmplitude: f32,
    lineSpacing: f32,
    maxCharsPerLine: u32,
    halfWidth: f32,
    noiseScale: f32,
    noiseStrength: f32,
    noiseSpeed: f32,
    noiseZoom: f32,
    noiseMix: f32,
    noiseTime: f32,
    peakExponent: f32,
    peakMin: f32,
    textOffsetX: f32,
    textOffsetY: f32,
    textOffsetZ: f32,
    glyphRotX: f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> basePositions: array<vec4<f32>>;
@group(0) @binding(2) var elevationTex: texture_2d<f32>;
@group(0) @binding(3) var<storage, read> lineMeta: array<vec4<u32>>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var elevationSampler: sampler;

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
    if (idx >= params.particleCount) { return; }

    let base = basePositions[idx];
    let lm = lineMeta[idx];
    let lineIdx = lm.x;
    let charIdx = lm.y;
    let charsInLine = lm.z;

    // Cyclic line reordering: active line always at bottom,
    // previous lines scroll up and wrap around to the top.
    // visual_slot 0 = top, lineCount-1 = bottom.
    let N = params.lineCount;
    let activeLine = params.activeLineIdx;
    let visualSlot = (lineIdx + N - activeLine - 1u) % N;

    // Recompute Y from the visual slot (overrides base Y)
    let totalHeight = f32(N - 1u) * params.lineSpacing;
    let yOffset = totalHeight * 0.5;
    let lastLine = f32(N - 1u);
    var pos = base;
    pos.y = -(lastLine - f32(visualSlot)) * params.lineSpacing + yOffset;

    // FFT displacement on ALL lines, with 2D center falloff:
    // - Vertical: center lines of the block get full amplitude, top/bottom fade
    // - Horizontal: center characters get full amplitude, edges fade
    // This creates the Unknown Pleasures silhouette — a mountain that bulges
    // from the center of the text area.

    if (charsInLine > 0u) {
        // Map the character's PHYSICAL x position to [0..1] relative to the
        // wave line extent. This ensures the text samples the same elevation
        // values as the wave line geometry at the same screen position.
        // base.x is the character's x from layout (centered around 0).
        // The wave line spans [-halfWidth, +halfWidth].
        // We approximate halfWidth as lineSpacing * 10 (matching layout's padding).
        // Map character X to elevation map column
        let hw = params.halfWidth;
        let t = clamp((base.x + hw) / (hw * 2.0), 0.0, 1.0);
        let linesInMap = min(N, 256u);
        let mapY = visualSlot * (256u / linesInMap);
        let uv = vec2<f32>(t, f32(min(mapY, 255u)) / 255.0);
        // Per-line Z layering: upper lines closer to camera
        let lineZ = f32(N - 1u - visualSlot) * 0.5;
        let elevation = textureSampleLevel(elevationTex, elevationSampler, uv, 0.0).r;

        // Apply peak shape to noise (same bell curve as FFT in the elevation map)
        let noiseDist = (t - 0.5) * 2.0;
        let noisePeak = max(params.peakMin, exp(-params.peakExponent * noiseDist * noiseDist));
        let noiseCoord = vec2<f32>(t * 256.0, f32(mapY)) * params.noiseScale * params.noiseZoom
                       + vec2<f32>(0.0, params.noiseTime * params.noiseSpeed);
        let noise = (fbm(noiseCoord) * 0.5 + 0.5) * params.noiseStrength * noisePeak;
        let val = mix(elevation, noise, params.noiseMix);

        pos.z = lineZ + val * params.fftAmplitude;

        // Apply text offset
        pos.x += params.textOffsetX;
        pos.y += params.textOffsetY;
        pos.z += params.textOffsetZ;
    }

    // Pack glyph rotation in w (read by msdf_text.wgsl vertex shader)
    pos.w = params.glyphRotX;
    positions[idx] = pos;
}
