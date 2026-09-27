// Multiple-scattering LUT (Hillaire 2020 section 5.5): for each (cos sun zenith, altitude), the
// luminance of all scattering orders >= 2 as seen from that point, assuming the scattered light
// is isotropic and the same all around it. One 64-thread workgroup per texel, one thread per
// direction (an 8x8 stratified sphere), each marching to the ground or to space:
//   L2   = mean over directions of the second-order luminance (unit illuminance, isotropic phase)
//   f_ms = mean over directions of the transfer to the next order
//   Psi_ms = L2 / (1 - f_ms)   (the geometric series of all higher orders)
// Depends only on the atmosphere, so it is rebuilt only when the atmosphere changes.

@group(0) @binding(0) var<uniform> atm : Atmosphere;
@group(0) @binding(1) var transmittanceLut : texture_2d<f32>;
@group(0) @binding(2) var lutSampler : sampler;
@group(0) @binding(3) var msOut : texture_storage_2d<rgba16float, write>;

const SQRT_DIRS : u32 = 8u;
const DIRS : u32 = 64u;
const SAMPLES : u32 = 20u;

var<workgroup> sharedL : array<vec3f, 64>;
var<workgroup> sharedF : array<vec3f, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg : vec3u, @builtin(local_invocation_index) li : u32) {
    let sizeF = vec2f(textureDimensions(msOut));
    let uv = (vec2f(wg.xy) + 0.5) / sizeF;
    let muS = texelUvToUnit(uv.x, sizeF.x) * 2.0 - 1.0;
    let h = saturate(texelUvToUnit(uv.y, sizeF.y));
    let r = clamp(atm.bottomRadius + h * (atm.topRadius - atm.bottomRadius), atm.bottomRadius + 1e-3, atm.topRadius - 1e-3);
    let ro = vec3f(0.0, r, 0.0);
    let sunDir = vec3f(sqrt(max(1.0 - muS * muS, 0.0)), muS, 0.0);

    let a = (f32(li / SQRT_DIRS) + 0.5) / f32(SQRT_DIRS);
    let b = (f32(li % SQRT_DIRS) + 0.5) / f32(SQRT_DIRS);
    let phi = 2.0 * PI * a;
    let cosT = 1.0 - 2.0 * b;
    let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    let rd = vec3f(sinT * cos(phi), cosT, sinT * sin(phi));

    let tGround = rayGround(ro, rd);
    let tMax = select(rayTop(ro, rd), tGround, tGround > 0.0);
    let dt = tMax / f32(SAMPLES);
    let isotropicPhase = 1.0 / (4.0 * PI);
    var throughput = vec3f(1.0);
    var lum = vec3f(0.0);
    var transfer = vec3f(0.0);
    for (var i = 0u; i < SAMPLES; i++) {
        let p = ro + rd * ((f32(i) + 0.5) * dt);
        let rp = length(p);
        let med = sampleMedium(rp - atm.bottomRadius);
        let mu = dot(p, sunDir) / rp;
        let sun = horizonVisibility(rp, mu, 0.0) * transmittanceToTop(rp, mu);
        let s = sun * med.scattering * isotropicPhase;
        let segT = exp(-med.extinction * dt);
        let invExt = 1.0 / max(med.extinction, vec3f(1e-9));
        lum += throughput * (s - s * segT) * invExt;
        transfer += throughput * (med.scattering - med.scattering * segT) * invExt;
        throughput *= segT;
    }
    // light the sun puts on the ground, reflected diffusely back into the atmosphere
    if (tGround > 0.0) {
        let n = normalize(ro + rd * tGround);
        let mu = dot(n, sunDir);
        lum += throughput * transmittanceToTop(atm.bottomRadius, mu) * saturate(mu) * atm.groundAlbedo / PI;
    }

    sharedL[li] = lum;
    sharedF[li] = transfer;
    workgroupBarrier();
    for (var stride = DIRS / 2u; stride > 0u; stride = stride / 2u) {
        if (li < stride) {
            sharedL[li] += sharedL[li + stride];
            sharedF[li] += sharedF[li + stride];
        }
        workgroupBarrier();
    }
    if (li == 0u) {
        let l2 = sharedL[0] / f32(DIRS);
        let fms = sharedF[0] / f32(DIRS);
        let psi = l2 / max(vec3f(1.0) - fms, vec3f(1e-4));
        textureStore(msOut, wg.xy, vec4f(psi * atm.multiScatteringFactor, 1.0));
    }
}
