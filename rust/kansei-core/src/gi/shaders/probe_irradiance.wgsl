// Reading the probes (gi::PROBES_WGSL): `kansei_gi_irradiance(p, n)`, the irradiance a surface at
// `p` facing `n` receives (scene units: a diffuse surface adds albedo / pi times it), from the
// eight probes around it. Needs probe_common.wgsl and these, declared by the includer at its own
// group and bindings (`SdfProbes::bindings_wgsl` writes them):
//   var<uniform> kansei_probe_grid : ProbeGrid;
//   var<storage, read> kansei_probe_sh : array<vec4f>;
//   var<storage, read> kansei_probe_state : array<vec4f>;
//   var<storage, read> kansei_probe_depth : array<vec2f>;
//
// The weights are DDGI's (Majercik et al. 2019): trilinear, times how much each probe lies in
// front of the surface, times the chance it sees the point (Chebyshev's bound on its depth
// moments toward the point), so a probe behind a wall does not light the room in front of it.

// A probe's irradiance toward `n`.
fn kanseiProbeShIrradiance(slot: u32, n: vec3f) -> vec3f {
    let y = probeShBasis(n);
    var e = vec3f(0.0);
    for (var i = 0u; i < PROBE_SH_WORDS; i++) {
        e += kansei_probe_sh[slot * PROBE_SH_WORDS + i].rgb * y[i];
    }
    return max(e, vec3f(0.0));
}

// A probe's depth moments toward `d`, bilinear over its octahedral map.
fn kanseiProbeMoments(slot: u32, d: vec3f) -> vec2f {
    let side = f32(PROBE_DEPTH_SIDE);
    let t = probeOctEncode(d) * side - 0.5;
    let t0 = floor(t);
    let f = t - t0;
    let lo = vec2i(clamp(t0, vec2f(0.0), vec2f(side - 1.0)));
    let hi = vec2i(clamp(t0 + 1.0, vec2f(0.0), vec2f(side - 1.0)));
    let base = slot * PROBE_DEPTH_TEXELS;
    let s = i32(PROBE_DEPTH_SIDE);
    let a = kansei_probe_depth[base + u32(lo.y * s + lo.x)];
    let b = kansei_probe_depth[base + u32(lo.y * s + hi.x)];
    let c = kansei_probe_depth[base + u32(hi.y * s + lo.x)];
    let e = kansei_probe_depth[base + u32(hi.y * s + hi.x)];
    return mix(mix(a, b, f.x), mix(c, e, f.x), f.y);
}

fn kansei_gi_irradiance(p: vec3f, n: vec3f) -> vec3f {
    return kanseiProbeIrradiance(p, n, n);
}

// kansei_gi_irradiance with the lookup moved off the surface along `biasDir` (the normal, or
// DDGI's mix of the normal and the direction to the viewer, which keeps lookups in creases
// clear of the probes' depth edges), `normalBias` metres times its length.
fn kanseiProbeIrradiance(p: vec3f, n: vec3f, biasDir: vec3f) -> vec3f {
    let grid = kansei_probe_grid;
    let biased = p + biasDir * grid.normalBias;
    let g = (biased - grid.origin) / grid.spacing;
    let top = vec3f(max(vec3i(grid.dims) - 2, vec3i(0)));
    let first = clamp(floor(g), vec3f(0.0), top);
    let alpha = clamp(g - first, vec3f(0.0), vec3f(1.0));
    var sum = vec3f(0.0);
    var total = 0.0;
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3u(i & 1u, (i >> 1u) & 1u, i >> 2u);
        let local = min(vec3u(first) + corner, grid.dims - 1u);
        let slot = probeSlot(grid, grid.base + vec3i(local));
        let state = kansei_probe_state[2u * slot];
        if (state.w > grid.backfaceLimit) { continue; }
        let probe = grid.origin + vec3f(local) * grid.spacing + state.xyz;
        let toProbe = probe - p;
        let dirToProbe = toProbe * inverseSqrt(max(dot(toProbe, toProbe), 1e-8));
        let tri = select(1.0 - alpha, alpha, corner == vec3u(1u));
        var weight = 1.0;
        // how much the probe lies in front of the surface (wrapped, so the ones beside it count)
        let front = (dot(dirToProbe, n) + 1.0) * 0.5;
        weight *= front * front + 0.2;
        if (grid.visibility != 0u) {
            let toPoint = biased - probe;
            let r = length(toPoint);
            let m = kanseiProbeMoments(slot, toPoint / max(r, 1e-6));
            if (r > m.x) {
                // (a floor of half the bias: depth edges a lookup grazes fade instead of cutting)
                let variance = max(abs(m.y - m.x * m.x), 0.25 * grid.normalBias * grid.normalBias);
                let chebyshev = variance / (variance + (r - m.x) * (r - m.x));
                weight *= max(chebyshev * chebyshev * chebyshev, 0.0);
            }
        }
        weight = max(weight, 1e-6);
        // crush the faint weights, so a probe that barely sees the point gives next to nothing
        let crush = 0.2;
        if (weight < crush) { weight *= weight * weight / (crush * crush); }
        weight *= tri.x * tri.y * tri.z;
        sum += weight * kanseiProbeShIrradiance(slot, n);
        total += weight;
    }
    return select(vec3f(0.0), sum / total, total > 0.0);
}
