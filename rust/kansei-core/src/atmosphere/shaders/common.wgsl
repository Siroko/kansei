// Physically based sky and atmosphere, after Hillaire 2020 ("A Scalable and Production Ready Sky
// and Atmosphere Rendering Technique", EGSR) with Bruneton's transmittance parameterisation
// (Bruneton 2017, "Precomputed Atmospheric Scattering: a New Implementation").
//
// Units: kilometres, in a planet-centred frame whose +Y passes through the world origin; the
// scattering and absorption coefficients are per kilometre. Every shader that includes this file
// declares `atm : Atmosphere` as a uniform.

const PI : f32 = 3.14159265358979;

struct Atmosphere {
    bottomRadius          : f32,
    topRadius             : f32,
    rayleighExpScale      : f32,   // -1 / Rayleigh scale height
    mieExpScale           : f32,   // -1 / Mie scale height
    rayleighScattering    : vec3f,
    mieG                  : f32,
    mieScattering         : vec3f,
    absorptionTipAltitude : f32,   // the absorbing (ozone) layer is a tent around this altitude
    mieExtinction         : vec3f,
    absorptionTipValue    : f32,
    absorptionExtinction  : vec3f,
    absorptionWidth       : f32,   // half width of the tent
    groundAlbedo          : vec3f,
    multiScatteringFactor : f32,
}

struct Medium {
    rayleighScattering : vec3f,
    mieScattering      : vec3f,
    scattering         : vec3f,
    extinction         : vec3f,
}

fn sampleMedium(altitude: f32) -> Medium {
    let h = max(altitude, 0.0);
    let densityRayleigh = exp(atm.rayleighExpScale * h);
    let densityMie = exp(atm.mieExpScale * h);
    let densityAbsorption = max(0.0, atm.absorptionTipValue - abs(h - atm.absorptionTipAltitude) / atm.absorptionWidth);
    var m : Medium;
    m.rayleighScattering = atm.rayleighScattering * densityRayleigh;
    m.mieScattering = atm.mieScattering * densityMie;
    m.scattering = m.rayleighScattering + m.mieScattering;
    m.extinction = m.rayleighScattering + atm.mieExtinction * densityMie + atm.absorptionExtinction * densityAbsorption;
    return m;
}

fn rayleighPhase(cosTheta: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cosTheta * cosTheta);
}

// Cornette-Shanks: Henyey-Greenstein with the Rayleigh-like (1 + cos^2) term, normalised.
fn miePhase(cosTheta: f32, g: f32) -> f32 {
    let g2 = g * g;
    let denom = max(1.0 + g2 - 2.0 * g * cosTheta, 1e-4);
    return 3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + cosTheta * cosTheta) / ((2.0 + g2) * denom * sqrt(denom));
}

// Both distances along a unit ray from `ro` to the sphere of `radius` at the origin (near, far),
// or (-1, -1) when it misses. The product-of-roots form stays exact for rays that start a few
// metres above a 6000 km sphere, where the textbook formula cancels catastrophically.
fn raySphere(ro: vec3f, rd: vec3f, radius: f32) -> vec2f {
    let len = length(ro);
    let b = dot(ro, rd);
    let c = (len - radius) * (len + radius);
    let disc = b * b - c;
    if (disc < 0.0) { return vec2f(-1.0); }
    let s = sqrt(disc);
    let q = select(-b + s, -b - s, b > 0.0);
    if (abs(q) < 1e-12) { return vec2f(0.0); }
    let t0 = q;
    let t1 = c / q;
    return vec2f(min(t0, t1), max(t0, t1));
}

// Distance to the ground along the ray, or -1 if the ray escapes (for rays starting above it).
fn rayGround(ro: vec3f, rd: vec3f) -> f32 {
    let t = raySphere(ro, rd, atm.bottomRadius);
    return select(-1.0, t.x, t.x > 0.0);
}

// Distance to the top of the atmosphere along the ray (for rays starting inside it).
fn rayTop(ro: vec3f, rd: vec3f) -> f32 {
    return max(raySphere(ro, rd, atm.topRadius).y, 0.0);
}

// Cosine of the zenith angle of the geometric horizon seen from radius r (<= 0).
fn horizonCos(r: f32) -> f32 {
    let rho = sqrt(max((r - atm.bottomRadius) * (r + atm.bottomRadius), 0.0));
    return -rho / r;
}

// Map [0,1] onto the texel centres of an n-texel axis and back (Bruneton 2017, section 4).
fn unitToTexelUv(x: f32, n: f32) -> f32 {
    return 0.5 / n + x * (1.0 - 1.0 / n);
}

fn texelUvToUnit(u: f32, n: f32) -> f32 {
    return (u - 0.5 / n) / (1.0 - 1.0 / n);
}

// Transmittance LUT: (radius, cos zenith) -> unit coordinates (x_mu, x_r). Rays below the horizon
// clamp to the horizon; callers handle the ground themselves.
fn transmittanceParams(r: f32, mu: f32) -> vec2f {
    let top = atm.topRadius;
    let bottom = atm.bottomRadius;
    let H = sqrt(max((top - bottom) * (top + bottom), 0.0));
    let rho = sqrt(max((r - bottom) * (r + bottom), 0.0));
    let disc = (top - r) * (top + r) + r * r * mu * mu;
    let d = max(0.0, -r * mu + sqrt(max(disc, 0.0)));
    let dMin = top - r;
    let dMax = rho + H;
    return saturate(vec2f((d - dMin) / max(dMax - dMin, 1e-6), rho / H));
}

// Sky-view LUT latitude mapping: the zenith angle of the texel row, with rows concentrated at
// the horizon (a square-root mapping on each side of it, Hillaire 2020 section 5.3). Longitude
// is the world azimuth, so the sun and the moon share one LUT.
fn skyViewZenith(v: f32, r: f32) -> f32 {
    let thetaH = acos(horizonCos(r));
    if (v < 0.5) {
        let c = 1.0 - 2.0 * v;
        return thetaH * (1.0 - c * c);
    }
    let c = 2.0 * v - 1.0;
    return thetaH + (PI - thetaH) * c * c;
}

fn skyViewV(cosZenith: f32, r: f32) -> f32 {
    let thetaH = acos(horizonCos(r));
    let theta = acos(clamp(cosZenith, -1.0, 1.0));
    if (theta < thetaH) {
        return 0.5 * (1.0 - sqrt(max(1.0 - theta / thetaH, 0.0)));
    }
    return 0.5 + 0.5 * sqrt(max((theta - thetaH) / (PI - thetaH), 0.0));
}

// The tangent frame at the camera: y is the local up, x and z span the local horizon.
struct LocalFrame {
    x  : vec3f,
    up : vec3f,
    z  : vec3f,
}

fn localFrame(cameraPos: vec3f) -> LocalFrame {
    var f : LocalFrame;
    f.up = normalize(cameraPos);
    f.x = normalize(cross(f.up, vec3f(0.0, 0.0, 1.0)));
    f.z = cross(f.x, f.up);
    return f;
}
