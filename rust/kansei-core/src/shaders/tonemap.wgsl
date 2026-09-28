// Display transform, the last effect of the HDR chain. Per pixel, in order:
//   lens:    lateral chromatic aberration (spectral taps), then physical exposure, Unreal's local
//            exposure (optional, from local_exposure.wgsl's grid) and a cos^4 vignette, all on
//            scene-linear light;
//   grade:   white balance, contrast about middle grey, and saturation/gain per tonal zone,
//            on exposed scene-linear light (UE applies its colour grading at this point too);
//   curve:   a filmic tone curve to display-linear [0, 1] (or, for UnrealFilmic, Unreal's own
//            grade and filmic curve in AP1, unrealDisplay);
//   film:    monochrome grain and a triangular dither, in the sRGB-encoded signal;
//   output:  sRGB-encoded for non-sRGB targets, decoded back to linear for sRGB ones.

@group(0) @binding(0) var inputTex      : texture_2d<f32>;
@group(0) @binding(1) var outputTex     : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> p    : ToneMapParams;
@group(0) @binding(3) var linearSampler : sampler;
// Unreal's local exposure (local_exposure.wgsl): the bilateral grid of log luminance, and the
// blurred log luminance
@group(0) @binding(4) var localGrid     : texture_3d<f32>;
@group(0) @binding(5) var localBlurred  : texture_2d<f32>;

const TONEMAP_NONE           : u32 = 0u;
const TONEMAP_ACES_FITTED    : u32 = 1u;
const TONEMAP_AGX            : u32 = 2u;
const TONEMAP_AGX_PUNCHY     : u32 = 3u;
const TONEMAP_KHRONOS_NEUTRAL: u32 = 4u;
const TONEMAP_UNREAL_FILMIC  : u32 = 5u;

const FLAG_ENCODE_SRGB : u32 = 1u;
const FLAG_DITHER      : u32 = 2u;

const MIDDLE_GREY : f32 = 0.18;

fn luminance(c: vec3f) -> f32 {
    return dot(c, vec3f(0.2126, 0.7152, 0.0722));
}

// ── Noise ──

// PCG-based 3D hash (Jarzynski & Olano, "Hash Functions for GPU Rendering", 2020).
fn pcg3d(v0: vec3u) -> vec3u {
    var v = v0 * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return v;
}

fn hash3(pix: vec2i, frame: u32) -> vec3f {
    return vec3f(pcg3d(vec3u(bitcast<vec2u>(pix), frame))) * (1.0 / 4294967296.0);
}

// Film grain: value noise on a lattice of `grainSize` pixels, a new pattern every frame.
// Each lattice value is triangular in [-1, 1] (two uniforms), which reads like film, not TV static.
fn grainNoise(pix: vec2f) -> f32 {
    let q = pix / max(p.grainSize, 1.0);
    let i = vec2i(floor(q));
    let f = fract(q);
    let s = f * f * (3.0 - 2.0 * f);
    let n00 = hash3(i, p.frame);
    let n10 = hash3(i + vec2i(1, 0), p.frame);
    let n01 = hash3(i + vec2i(0, 1), p.frame);
    let n11 = hash3(i + vec2i(1, 1), p.frame);
    let v = mix(mix(n00.x + n00.y, n10.x + n10.y, s.x), mix(n01.x + n01.y, n11.x + n11.y, s.x), s.y);
    return v - 1.0;
}

// ── Lens ──

// Lateral chromatic aberration: the image magnification varies with wavelength, so 7 taps from
// red (scaled out) to blue (scaled in), weighted by a coarse spectrum-to-RGB table whose columns
// each sum to one (white stays white).
fn sampleLens(coord: vec2u, uv: vec2f) -> vec3f {
    if (p.chromaticAberration <= 0.0) {
        return textureLoad(inputTex, coord, 0).rgb;
    }
    let weights = array<vec3f, 7>(
        vec3f(1.00, 0.00, 0.00) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.75, 0.25, 0.00) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.35, 0.65, 0.00) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.00, 1.00, 0.00) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.00, 0.65, 0.35) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.00, 0.25, 0.75) / vec3f(2.1, 2.8, 2.1),
        vec3f(0.00, 0.00, 1.00) / vec3f(2.1, 2.8, 2.1),
    );
    // at intensity 1, red and blue differ in magnification by 1 % (about 10 px at a 1080p edge)
    let amount = p.chromaticAberration * 0.005;
    let d = uv - 0.5;
    var acc = vec3f(0.0);
    for (var i = 0; i < 7; i++) {
        let scale = 1.0 + amount * (1.0 - f32(i) / 3.0);
        acc += textureSampleLevel(inputTex, linearSampler, 0.5 + d * scale, 0.0).rgb * weights[i];
    }
    return acc;
}

// Unreal's local exposure (its bilateral method, PostProcessHistogramCommon.ush's
// CalculateBaseLogLuminance and CalculateLocalExposure): the factor on this pixel's scene light
// that scales the contrast of its surroundings' luminance about middle grey (the highlight or
// shadow contrast) while keeping its detail against them (the detail strength). Its
// surroundings are the bilateral grid's mean log luminance of nearby pixels as bright as it,
// mixed with the blurred log luminance.
fn localExposure(scene: vec3f, uv: vec2f) -> f32 {
    let logL = log2(max(dot(scene, vec3f(1.0 / 3.0)), exp2(p.localExposure2.w)));
    let bin = logL * p.localExposure2.y + p.localExposure2.z;
    let g = textureSampleLevel(localGrid, linearSampler, vec3f(uv * p.localExposure3.xy, (bin * 31.0 + 0.5) / 32.0), 0.0).xy;
    let blurred = textureSampleLevel(localBlurred, linearSampler, uv, 0.0).r;
    // a grid cell with no pixels this bright falls back to the blurred luminance
    let bilateral = select(g.x / g.y, blurred, g.y * LOCAL_CELL_TEXELS < 0.001);
    let logExposure = log2(p.exposure);
    let base = mix(bilateral, blurred, p.localExposure.w) + logExposure;
    let y = logL + logExposure;
    let middleGrey = p.localExposure2.x;
    let contrast = select(p.localExposure.y, p.localExposure.x, base > middleGrey);
    return exp2(middleGrey + (base - middleGrey) * contrast + (y - base) * p.localExposure.z - y);
}

// Unreal's cosine-fourth vignette (PostProcessCommon.ush's VignetteSpace and
// ComputeVignetteMask): the position in the frame's [-1, 1] viewport, scaled so its corners lie
// on a circle of radius sqrt(2), times the intensity, is the tangent of the angle off the axis.
// A letterboxed picture is the centre crop of its frame (frameAspect), whose vignette it shows.
fn vignetteMask(uv: vec2f) -> f32 {
    let aspect = f32(p.width) / f32(p.height);
    let frame = select(aspect, p.frameAspect, p.frameAspect > 0.0);
    // the picture's height as a share of the frame's
    let band = frame / aspect;
    let pos = vec2f(uv.x * 2.0 - 1.0, (uv.y * 2.0 - 1.0) * band);
    let heightByWidth = 1.0 / frame;
    let circle = pos * vec2f(1.0, heightByWidth) * (sqrt(2.0) / sqrt(1.0 + heightByWidth * heightByWidth)) * p.vignette;
    let cos2 = 1.0 / (1.0 + dot(circle, circle));
    return cos2 * cos2;
}

// ── Grade (exposed scene-linear) ──

fn grade(c0: vec3f) -> vec3f {
    var c = max(p.whiteBalance * c0, vec3f(0.0));
    c = MIDDLE_GREY * pow(c / MIDDLE_GREY, vec3f(p.contrast));
    let luma = luminance(c);
    let wShadow = 1.0 - smoothstep(0.0, p.shadowsMax, luma);
    let wHighlight = smoothstep(p.highlightsMin, 1.0, luma) * (1.0 - wShadow);
    let wMid = 1.0 - wShadow - wHighlight;
    let sat = p.saturation * (wShadow * p.shadowSaturation + wMid + wHighlight * p.highlightSaturation);   // per channel
    c = max(mix(vec3f(luma), c, sat), vec3f(0.0));
    return c * p.gain * (wShadow * p.shadowGain + wMid + wHighlight * p.highlightGain);
}

// ── Tone curves (display-linear out) ──

// ACES RRT + sRGB ODT, Stephen Hill's fit (BakingLab, MIT). The matrices are written row by row,
// so they are applied as `v * M`.
fn acesFitted(c: vec3f) -> vec3f {
    let inputRows = mat3x3f(
        vec3f(0.59719, 0.35458, 0.04823),
        vec3f(0.07600, 0.90834, 0.01566),
        vec3f(0.02840, 0.13383, 0.83777),
    );
    let outputRows = mat3x3f(
        vec3f( 1.60475, -0.53108, -0.07367),
        vec3f(-0.10208,  1.10813, -0.00605),
        vec3f(-0.00327, -0.07276,  1.07602),
    );
    let v = c * inputRows;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate((a / b) * outputRows);
}

// AgX (Troy Sobotka), with the Rec.2020 inset/outset matrices from Filament and the 6th-order
// sigmoid fit of Blender's default contrast; `punchy` is Blender's Punchy look (ASC CDL power 1.35,
// saturation 1.4).
fn agx(c: vec3f, punchy: bool) -> vec3f {
    let srgbTo2020 = mat3x3f(
        vec3f(0.6274, 0.0691, 0.0164),
        vec3f(0.3293, 0.9195, 0.0880),
        vec3f(0.0433, 0.0113, 0.8956),
    );
    let rec2020ToSrgb = mat3x3f(
        vec3f( 1.6605, -0.1246, -0.0182),
        vec3f(-0.5876,  1.1329, -0.1006),
        vec3f(-0.0728, -0.0083,  1.1187),
    );
    let inset = mat3x3f(
        vec3f(0.856627153315983, 0.137318972929847, 0.11189821299995),
        vec3f(0.0951212405381588, 0.761241990602591, 0.0767994186031903),
        vec3f(0.0482516061458583, 0.101439036467562, 0.811302368396859),
    );
    let outset = mat3x3f(
        vec3f( 1.1271005818144368, -0.1413297634984383, -0.14132976349843826),
        vec3f(-0.11060664309660323, 1.157823702216272, -0.11060664309660294),
        vec3f(-0.016493938717834573, -0.016493938717834257, 1.2519364065950405),
    );
    let minEv = -12.47393;  // log2(2^-10 * 0.18)
    let maxEv = 4.026069;   // log2(2^6.5 * 0.18)

    var v = inset * (srgbTo2020 * c);
    v = saturate((log2(max(v, vec3f(1e-10))) - minEv) / (maxEv - minEv));
    let x2 = v * v;
    let x4 = x2 * x2;
    v = 15.5 * x4 * x2 - 40.14 * x4 * v + 31.96 * x4 - 6.868 * x2 * v + 0.4298 * x2 + 0.1191 * v - 0.00232;
    if (punchy) {
        let luma = luminance(v);
        v = pow(max(v, vec3f(0.0)), vec3f(1.35));
        v = luma + 1.4 * (v - luma);
    }
    v = pow(max(outset * v, vec3f(0.0)), vec3f(2.2));
    return saturate(rec2020ToSrgb * v);
}

// Khronos PBR Neutral: hue- and value-preserving up to 0.76, a smooth shoulder above (for
// product/material shots where colours must stay what they were authored as).
fn khronosNeutral(c0: vec3f) -> vec3f {
    let startCompression = 0.8 - 0.04;
    let desaturation = 0.15;
    var c = c0;
    let x = min(c.r, min(c.g, c.b));
    let offset = select(0.04, x - 6.25 * x * x, x < 0.08);
    c -= offset;
    let peak = max(c.r, max(c.g, c.b));
    if (peak < startCompression) {
        return saturate(c);
    }
    let d = 1.0 - startCompression;
    let newPeak = 1.0 - d * d / (peak + d - startCompression);
    c *= newPeak / peak;
    let g = 1.0 - 1.0 / (desaturation * (peak - newPeak) + 1.0);
    return saturate(mix(c, vec3f(newPeak), g));
}

fn toneCurve(c: vec3f) -> vec3f {
    switch (p.tonemapper) {
        case TONEMAP_ACES_FITTED: { return acesFitted(c); }
        case TONEMAP_AGX: { return agx(c, false); }
        case TONEMAP_AGX_PUNCHY: { return agx(c, true); }
        case TONEMAP_KHRONOS_NEUTRAL: { return khronosNeutral(c); }
        default: { return saturate(c); }
    }
}

// ── Unreal's display transform (PostProcessCombineLUTs.usf, TonemapCommon.ush, ACESCommon.ush) ──
// Its matrices are written row by row, so they are applied as `v * M`.

const AP1_Y = vec3f(0.2722287168, 0.6740817658, 0.0536895174);
const INV_LN10 : f32 = 0.4342944819;

fn srgbToAp1(c: vec3f) -> vec3f {
    return c * mat3x3f(vec3f(0.6130974024, 0.3395231461, 0.0473794514), vec3f(0.0701937225, 0.9163538791, 0.0134523986), vec3f(0.0206155929, 0.1095697729, 0.8698146341));
}
fn ap1ToSrgb(c: vec3f) -> vec3f {
    return c * mat3x3f(vec3f(1.7050509926, -0.6217921205, -0.0832588722), vec3f(-0.1302564175, 1.1408047365, -0.0105483190), vec3f(-0.0240033568, -0.1289689761, 1.1529723328));
}
fn ap1ToAp0(c: vec3f) -> vec3f {
    return c * mat3x3f(vec3f(0.6954522414, 0.1406786965, 0.1638690622), vec3f(0.0447945634, 0.8596711185, 0.0955343182), vec3f(-0.0055258826, 0.0040252103, 1.0015006723));
}
fn ap0ToAp1(c: vec3f) -> vec3f {
    return c * mat3x3f(vec3f(1.4514393161, -0.2365107469, -0.2149285693), vec3f(-0.0765537734, 1.1762296998, -0.0996759264), vec3f(0.0083161484, -0.0060324498, 0.9977163014));
}

// Nuke-style ColorCorrect in AP1: saturation about AP1 luma, contrast about middle grey, gain.
fn unrealColorCorrect(c: vec3f, saturation: vec3f, gain: vec3f) -> vec3f {
    let luma = dot(c, AP1_Y);
    var x = max(vec3f(0.0), luma + (c - luma) * saturation);
    x = pow(x / MIDDLE_GREY, vec3f(p.contrast)) * MIDDLE_GREY;
    return x * gain;
}

// ColorCorrectAll: the shadow, midtone and highlight grades blended by AP1 luma.
fn unrealColorCorrectAll(c: vec3f) -> vec3f {
    let luma = dot(c, AP1_Y);
    let shadows = unrealColorCorrect(c, p.shadowSaturation * p.saturation, p.shadowGain * p.gain);
    let wShadows = 1.0 - smoothstep(0.0, p.shadowsMax, luma);
    let highlights = unrealColorCorrect(c, p.highlightSaturation * p.saturation, p.highlightGain * p.gain);
    let wHighlights = smoothstep(p.highlightsMin, p.film2.w, luma);
    let midtones = unrealColorCorrect(c, p.saturation, p.gain);
    return shadows * wShadows + midtones * (1.0 - wShadows - wHighlights) + highlights * wHighlights;
}

fn rgbToSaturation(c: vec3f) -> f32 {
    let mn = min(c.r, min(c.g, c.b));
    let mx = max(c.r, max(c.g, c.b));
    return (max(mx, 1e-10) - max(mn, 1e-10)) / max(mx, 1e-2);
}

fn rgbToYc(c: vec3f) -> f32 {
    let chroma = sqrt(max(c.b * (c.b - c.g) + c.g * (c.g - c.r) + c.r * (c.r - c.b), 0.0));
    return (c.b + c.g + c.r + 1.75 * chroma) / 3.0;
}

fn sigmoidShaper(x: f32) -> f32 {
    let t = max(1.0 - abs(0.5 * x), 0.0);
    return 0.5 * (1.0 + sign(x) * (1.0 - t * t));
}

fn glowFwd(yc: f32, gain: f32, mid: f32) -> f32 {
    if (yc <= 2.0 / 3.0 * mid) { return gain; }
    if (yc >= 2.0 * mid) { return 0.0; }
    return gain * (mid / yc - 0.5);
}

fn rgbToHue(c: vec3f) -> f32 {
    var h = 0.0;
    if (!(c.r == c.g && c.g == c.b)) {
        h = degrees(atan2(sqrt(3.0) * (c.g - c.b), 2.0 * c.r - c.g - c.b));
    }
    if (h < 0.0) { h += 360.0; }
    return clamp(h, 0.0, 360.0);
}

fn centerHue(h: f32, center: f32) -> f32 {
    var x = h - center;
    if (x < -180.0) { x += 360.0; } else if (x > 180.0) { x -= 360.0; }
    return x;
}

// Unreal's FilmToneMap on AP1: ACES's glow and red modifier in AP0, a desaturation, the filmic
// curve per channel in log10 with a toe, a straight part and a shoulder, and a last desaturation.
fn unrealFilmToneMap(ap1: vec3f) -> vec3f {
    let slope = p.film.x;
    let toe = p.film.y;
    let shoulder = p.film.z;
    let blackClip = p.film.w;
    let whiteClip = p.film2.x;

    var ap0 = ap1ToAp0(ap1);
    let sat = rgbToSaturation(ap0);
    let s = sigmoidShaper((sat - 0.4) / 0.2);
    ap0 *= 1.0 + glowFwd(rgbToYc(ap0), 0.05 * s, 0.08);
    let hw = pow(smoothstep(0.0, 1.0, 1.0 - abs(2.0 * centerHue(rgbToHue(ap0), 0.0) / 135.0)), 2.0);
    ap0.r += hw * sat * (0.03 - ap0.r) * (1.0 - 0.82);

    var w = max(ap0ToAp1(ap0), vec3f(0.0));
    w = mix(vec3f(dot(w, AP1_Y)), w, 0.96);

    let toeScale = 1.0 + blackClip - toe;
    let shoulderScale = 1.0 + whiteClip - shoulder;
    let inMatch = 0.18;
    let outMatch = 0.18;
    var toeMatch : f32;
    if (toe > 0.8) {
        toeMatch = (1.0 - toe - outMatch) / slope + log(inMatch) * INV_LN10;
    } else {
        let bt = (outMatch + blackClip) / toeScale - 1.0;
        toeMatch = log(inMatch) * INV_LN10 - 0.5 * log((1.0 + bt) / (1.0 - bt)) * (toeScale / slope);
    }
    let straightMatch = (1.0 - toe) / slope - toeMatch;
    let shoulderMatch = shoulder / slope - straightMatch;

    let logColor = log(max(w, vec3f(1e-30))) * INV_LN10;
    let straight = slope * (logColor + straightMatch);
    var toeColor = -blackClip + 2.0 * toeScale / (1.0 + exp(-2.0 * slope / toeScale * (logColor - toeMatch)));
    var shoulderColor = 1.0 + whiteClip - 2.0 * shoulderScale / (1.0 + exp(2.0 * slope / shoulderScale * (logColor - shoulderMatch)));
    toeColor = select(straight, toeColor, logColor < vec3f(toeMatch));
    shoulderColor = select(straight, shoulderColor, logColor > vec3f(shoulderMatch));
    var t = saturate((logColor - toeMatch) / (shoulderMatch - toeMatch));
    if (shoulderMatch < toeMatch) { t = 1.0 - t; }
    t = (3.0 - 2.0 * t) * t * t;
    var tone = mix(toeColor, shoulderColor, t);
    tone = mix(vec3f(dot(tone, AP1_Y)), tone, 0.93);
    return max(tone, vec3f(0.0));
}

// Exposed scene-linear sRGB to display-linear sRGB, as Unreal's tonemapper LUT for an sRGB
// display (the white balance is applied by the caller, as `p.whiteBalance`).
fn unrealDisplay(balanced: vec3f) -> vec3f {
    var ap1 = srgbToAp1(balanced);
    // bright saturated colours pushed out toward a wider gamut
    let luma = dot(ap1, AP1_Y);
    let chroma = ap1 / max(luma, 1e-12);
    let chromaDist2 = dot(chroma - 1.0, chroma - 1.0);
    let expandAmount = (1.0 - exp2(-4.0 * chromaDist2)) * (1.0 - exp2(-4.0 * p.film2.z * luma * luma));
    let expanded = ap1 * mat3x3f(vec3f(1.3704123718, -0.3292921877, -0.0636831194), vec3f(-0.0834334917, 1.0970927480, -0.0108613795), vec3f(-0.0257933209, -0.0986257988, 1.2036949526));
    ap1 = mix(ap1, expanded, expandAmount);
    ap1 = unrealColorCorrectAll(ap1);
    let blueCorrect = mat3x3f(vec3f(0.9386393778, 0.0, 0.0613606221), vec3f(0.0, 0.8307941330, 0.1692058671), vec3f(0.0, 0.0, 1.0));
    let blueUncorrect = mat3x3f(vec3f(1.0653748755, 0.0000014467, -0.0653710053), vec3f(-0.0000003456, 1.2036635245, -0.2036677199), vec3f(0.0000000198, 0.0000000212, 0.9999996001));
    ap1 = mix(ap1, ap1 * blueCorrect, p.film2.y);
    ap1 = unrealFilmToneMap(ap1);
    ap1 = mix(ap1, ap1 * blueUncorrect, p.film2.y);
    return saturate(max(ap1ToSrgb(ap1), vec3f(0.0)));
}

// ── Encoding ──

fn srgbEncode(c: vec3f) -> vec3f {
    return select(1.055 * pow(c, vec3f(1.0 / 2.4)) - 0.055, c * 12.92, c <= vec3f(0.0031308));
}

fn srgbDecode(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= p.width || gid.y >= p.height) { return; }
    let pix = vec2f(gid.xy) + 0.5;
    let uv = pix / vec2f(f32(p.width), f32(p.height));

    let lens = sampleLens(gid.xy, uv);
    var local = 1.0;
    if ((p.flags & FLAG_LOCAL_EXPOSURE) != 0u) {
        local = localExposure(lens, uv);
    }
    let scene = lens * (p.exposure * vignetteMask(uv) * local);
    var display : vec3f;
    if (p.tonemapper == TONEMAP_UNREAL_FILMIC) {
        display = unrealDisplay(max(p.whiteBalance * scene, vec3f(0.0)));
    } else {
        display = toneCurve(grade(scene));
    }

    // film grain, strongest in the midtones of the encoded signal (where film shows it)
    var encoded = srgbEncode(display);
    if (p.grain > 0.0) {
        let l = luminance(encoded);
        encoded += p.grain * 0.12 * grainNoise(pix) * (4.0 * l * (1.0 - l));
    }
    // triangular dither of one 8-bit step, so dusk gradients don't band on 8-bit swapchains
    if ((p.flags & FLAG_DITHER) != 0u) {
        let h = hash3(vec2i(gid.xy), p.frame ^ 0x9e3779b9u);
        encoded += (h.x + h.y - 1.0) / 255.0;
    }
    encoded = saturate(encoded);

    let out = select(srgbDecode(encoded), encoded, (p.flags & FLAG_ENCODE_SRGB) != 0u);
    textureStore(outputTex, gid.xy, vec4f(out, 1.0));
}
