// Curl noise (Bridson, Hourihan and Nordenstam 2007): the curl of a three-component noise
// potential, a velocity field with no divergence, so what it carries neither piles up nor thins
// out. Each component of the potential is gradient noise with analytic derivatives (after Inigo
// Quilez's "Gradient Noise Derivatives", MIT), two octaves, drifting with time along its own
// direction; the curl is taken from those derivatives, so it is divergence-free exactly, not only
// to a finite-difference stencil's order. Self-contained (simulations::smoke::CURL_NOISE_WGSL).

fn curlNoiseHash(p: vec3i) -> vec3f {
    // pcg3d (Jarzynski and Olano 2020), to a gradient in [-1, 1]^3
    var v = vec3u(p) * 1664525u + 1013904223u;
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z; v.y += v.z * v.x; v.z += v.x * v.y;
    return vec3f(v & vec3u(0xffffu)) / 32767.5 - 1.0;
}

// Gradient noise at p: (value, d/dx, d/dy, d/dz).
fn curlNoiseGradient(p: vec3f) -> vec4f {
    let i = vec3i(floor(p));
    let f = fract(p);
    // quintic fade and its derivative
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = curlNoiseHash(i + vec3i(0, 0, 0));
    let gb = curlNoiseHash(i + vec3i(1, 0, 0));
    let gc = curlNoiseHash(i + vec3i(0, 1, 0));
    let gd = curlNoiseHash(i + vec3i(1, 1, 0));
    let ge = curlNoiseHash(i + vec3i(0, 0, 1));
    let gf = curlNoiseHash(i + vec3i(1, 0, 1));
    let gg = curlNoiseHash(i + vec3i(0, 1, 1));
    let gh = curlNoiseHash(i + vec3i(1, 1, 1));
    let va = dot(ga, f - vec3f(0.0, 0.0, 0.0));
    let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va)
        + u.x * u.y * (va - vb - vc + vd) + u.y * u.z * (va - vc - ve + vg)
        + u.z * u.x * (va - vb - ve + vf) + u.x * u.y * u.z * (-va + vb + vc - vd + ve - vf - vg + vh);
    let grad = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga)
        + u.x * u.y * (ga - gb - gc + gd) + u.y * u.z * (ga - gc - ge + gg)
        + u.z * u.x * (ga - gb - ge + gf) + u.x * u.y * u.z * (-ga + gb + gc - gd + ge - gf - gg + gh)
        + du * (vec3f(vb - va, vc - va, ve - va)
            + u.yzx * vec3f(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
            + u.zxy * vec3f(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg)
            + u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));
    return vec4f(value, grad);
}

// Two octaves of one potential component, offset by `seed` and drifting with `time`: 2 n(p / 2)
// + n(p), (value, d/dp). The coarse octave's lattice is two units, so the eddies are about a
// unit across and the velocity about unit speed.
fn curlNoiseComponent(p: vec3f, time: f32, seed: vec3f, drift: vec3f) -> vec4f {
    let a = curlNoiseGradient(p * 0.5 + seed + drift * time);
    let b = curlNoiseGradient(p + seed.zxy + drift.yzx * (time * 1.7));
    // d/dp of 2 n(p / 2) is n'(p / 2)
    return vec4f(2.0 * a.x + b.x, a.yzw + b.yzw);
}

// The potential psi(p, time) whose curl curlNoise is.
fn curlNoisePotential(p: vec3f, time: f32) -> vec3f {
    return vec3f(
        curlNoiseComponent(p, time, vec3f(0.0, 0.0, 0.0), vec3f(0.31, 0.17, -0.23)).x,
        curlNoiseComponent(p, time, vec3f(31.4, -17.2, 9.1), vec3f(-0.19, 0.29, 0.13)).x,
        curlNoiseComponent(p, time, vec3f(-12.7, 41.3, -27.9), vec3f(0.23, -0.11, 0.27)).x,
    );
}

// The curl of curlNoisePotential at p: a divergence-free velocity of about unit speed, its
// eddies about a unit across.
fn curlNoise(p: vec3f, time: f32) -> vec3f {
    let a = curlNoiseComponent(p, time, vec3f(0.0, 0.0, 0.0), vec3f(0.31, 0.17, -0.23)).yzw;
    let b = curlNoiseComponent(p, time, vec3f(31.4, -17.2, 9.1), vec3f(-0.19, 0.29, 0.13)).yzw;
    let c = curlNoiseComponent(p, time, vec3f(-12.7, 41.3, -27.9), vec3f(0.23, -0.11, 0.27)).yzw;
    // curl (psi_x, psi_y, psi_z) = (dpz/dy - dpy/dz, dpx/dz - dpz/dx, dpy/dx - dpx/dy)
    return vec3f(c.y - b.z, a.z - c.x, b.x - a.y);
}
