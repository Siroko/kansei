// Caustics (rt::RtCausticsEffect), the photons: from each light toward each glass caster whose
// tile takes photons this frame (`tasks`, made on the CPU), a grid of photons over the caster's
// box as the light sees it (one a cell, jittered by the tile's history, so a moving caster's
// photons keep their pattern and a still one's fill in) is traced through the grid of triangles,
// each from a point of the light's emitter as RtShadowsEffect samples it (the sun's disk, a spot
// light's disk or rectangle) through its cell, so a photon stopped on its way is light those
// rays find stopped too.
// Those whose first hit is glass within the caster's box carry the light through it: at every
// glass surface they cross, reflected or refracted by its exact dielectric Fresnel (chosen at
// random, the photon keeping its flux: no light is lost to the choice), Beer-Lambert inside (the
// glass triangle's tint), to the first other surface, where they add their flux; the same photon
// taken straight on, through no glass (where the direct light, which passes through glass,
// lands), takes it away there. The tile thus holds what the glass changes of the direct light:
// the light it focuses, and the shadow it leaves. Only surfaces that face up keep it, outside
// every caster's box, at the height they landed at. Each photon splats a tent about as wide as
// the photons lie apart there, into its light's layers of the tile (the light added, then the
// light taken away), fixed-point atomics: rgb (lux * scale), the weight and the weighted height.
// With `unoccluded`, the composite weighs the light taken away by RtShadowsEffect's visibility at
// the surface, so the photons it counts pass whatever stands between the light and the caster
// (that visibility has it); the light added stays the photons' own, traced from the emitter
// through the glass to where it lands, whatever stops it on the way stopping it. Prefixed with the
// grid's library (its bindings in group 1), RT_OPAQUE_WGSL and rt_caustics_lookup.wgsl.

struct CausticsParams {
    taskCount    : u32,
    photons      : u32,
    interfaces   : u32,
    maxDistance  : f32,
    splatTexels  : f32,
    glassIor     : f32,
    // the light through the glass times focusGain, its splats focusSharpen as wide (1, 1:
    // physical)
    focusGain    : f32,
    focusSharpen : f32,
    // multiplies every glass triangle's tint
    glassTint    : vec3f,
    // metres the tints are the light left after
    tintDistance : f32,
    // 1: the light taken away is counted from the caster's box on (see above)
    unoccluded   : u32,
    _pad0        : u32,
    _pad1        : u32,
    _pad2        : u32,
}

// A light's photons at a caster: kind 0 a directional light (along axis through the plane origin
// + u e1 + v e2; color its illuminance; emitterX, emitterY the tangents of its angular radius
// along e1, e2), 1 or 2 a spot light (from origin, its emitter's centre, along axis + u e1 + v e2;
// color its intensity, cd; emitterX, emitterY a disk's radius along two tangents (1) or a
// rectangle's half sides (2)). `tile` the caster's, `layer` the light's added light's at it (the
// light it takes away the next), `seed` its jitter, `bias` how far rays leave a surface (m).
struct CausticTask {
    origin   : vec3f,
    kind     : u32,
    axis     : vec3f,
    side     : u32,
    e1       : vec3f,
    u0       : f32,
    e2       : vec3f,
    v0       : f32,
    color    : vec3f,
    du       : f32,
    emitterX : vec3f,
    dv       : f32,
    emitterY : vec3f,
    first    : u32,
    lightDir : vec3f,
    cosOuter : f32,
    cosInner : f32,
    range    : f32,
    tile     : u32,
    seed     : u32,
    light    : u32,
    bias     : f32,
    layer    : u32,
    _pad0    : u32,
}

// a tile's texel weight in fixed point
const CAUSTICS_WEIGHT : f32 = 4096.0;


@group(0) @binding(0) var<uniform> cp : CausticsParams;
@group(0) @binding(1) var<storage, read> tasks : array<CausticTask>;
@group(0) @binding(2) var<storage, read_write> acc : array<atomic<i32>>;
@group(0) @binding(3) var<uniform> caustics : KanseiRtCaustics;

fn hash4(i: u32, j: u32, seed: u32, k: u32) -> vec4f {
    var h = i * 1973u + j * 9277u + seed * 26699u + k * 7919u;
    var out = vec4f(0.0);
    for (var c = 0u; c < 4u; c++) {
        h = (h ^ (h >> 15u)) * 0x2c1b3c6du;
        h = (h ^ (h >> 12u)) * 0x297a2d39u;
        h = h ^ (h >> 15u);
        out[c] = f32(h >> 8u) / 16777216.0;
        h += 0x9e3779b9u;
    }
    return out;
}

fn fresnelDielectric(cosI: f32, etaI: f32, etaT: f32) -> f32 {
    let sinT = etaI / etaT * sqrt(max(1.0 - cosI * cosI, 0.0));
    if (sinT >= 1.0) {
        return 1.0;
    }
    let cosT = sqrt(max(1.0 - sinT * sinT, 0.0));
    let rs = (etaI * cosI - etaT * cosT) / (etaI * cosI + etaT * cosT);
    let rp = (etaT * cosI - etaI * cosT) / (etaT * cosI + etaI * cosT);
    return 0.5 * (rs * rs + rp * rp);
}

fn isGlass(id: u32) -> bool {
    return (kansei_rt_surface(id) & KANSEI_RT_GLASS) != 0u;
}

fn inBox(p: vec3f, lo: vec3f, hi: vec3f, grow: f32) -> bool {
    return all(p >= lo - grow) && all(p <= hi + grow);
}

// Whether p lies well inside a live caster's box: what lands there (an opaque core) is the
// caster's, not a receiver's (the table a marble rests on touches its box).
fn inCaster(p: vec3f) -> bool {
    for (var k = 0u; k < min(caustics.count, KANSEI_RT_CAUSTIC_TILES); k++) {
        let t = caustics.tiles[k];
        let inset = 0.05 * (t.boxMax - t.boxMin);
        if (t.live != 0u && all(p > t.boxMin + inset) && all(p < t.boxMax - inset)) {
            return true;
        }
    }
    return false;
}

// Splat irradiance (lux: flux over a texel's area, times the receiver's n.y) at p into `tile`'s
// `layer`, a tent of `radius` metres (at least a texel and a half, at most `splatTexels` texels).
fn deposit(tile: u32, layer: u32, p: vec3f, flux: vec3f, radius: f32) {
    let t = caustics.tiles[tile];
    let texel = t.size.x / f32(t.dims.x);
    let c = (p.xz - t.min) / texel - 0.5;
    let r = clamp(radius / texel, 2.0, max(cp.splatTexels, 2.0));
    let k = i32(ceil(r));
    let base = vec2i(round(c));
    if (any(base + k < vec2i(0)) || any(base - k >= vec2i(t.dims))) {
        return;
    }
    var sum = 0.0;
    for (var y = -k; y <= k; y++) {
        for (var x = -k; x <= k; x++) {
            sum += max(1.0 - length(vec2f(base + vec2i(x, y)) - c) / r, 0.0);
        }
    }
    if (sum <= 0.0) {
        return;
    }
    let e = flux / (texel * texel * sum) * caustics.scale;
    let base0 = layer * caustics.texels * caustics.texels;
    let h = p.y - t.refY;
    for (var y = -k; y <= k; y++) {
        for (var x = -k; x <= k; x++) {
            let q = base + vec2i(x, y);
            let w = max(1.0 - length(vec2f(q) - c) / r, 0.0);
            if (w <= 0.0 || any(q < vec2i(0)) || any(q >= vec2i(t.dims))) {
                continue;
            }
            let at = (base0 + u32(q.y) * caustics.texels + u32(q.x)) * 5u;
            let v = e * w;
            atomicAdd(&acc[at], i32(round(v.r)));
            atomicAdd(&acc[at + 1u], i32(round(v.g)));
            atomicAdd(&acc[at + 2u], i32(round(v.b)));
            let wn = w / sum * CAUSTICS_WEIGHT;
            atomicAdd(&acc[at + 3u], i32(round(wn)));
            atomicAdd(&acc[at + 4u], i32(round(wn * h)));
        }
    }
}

@compute @workgroup_size(64)
fn photons(@builtin(global_invocation_id) gid : vec3u, @builtin(num_workgroups) groups : vec3u) {
    let id = gid.x + gid.y * groups.x * 64u;
    if (id >= cp.photons) { return; }
    // the task whose photons these are: the last whose first is at most id
    var lo = 0u;
    var hi = cp.taskCount;
    while (hi - lo > 1u) {
        let mid = (lo + hi) / 2u;
        if (tasks[mid].first <= id) { lo = mid; } else { hi = mid; }
    }
    let task = tasks[lo];
    let local = id - task.first;
    if (local >= task.side * task.side) { return; }
    let tile = caustics.tiles[task.tile];
    let i = local % task.side;
    let j = local / task.side;
    let xi = hash4(i, j, task.seed, task.light);
    let u = task.u0 + (f32(i) + xi.x) * task.du;
    let v = task.v0 + (f32(j) + xi.y) * task.dv;
    // the photon from the emitter's centre through its cell (the sun's along its axis), and the
    // point where it meets the plane its caster stands on (the tile's height; past the caster's
    // centre at least): each photon leaves a point of the emitter (a uniform one, the cell's own)
    // toward that point, so the photons land there on the centre's even grid, from whichever
    // point they left, and the emitter's size softens the shadow and the caustics only through
    // which of them the glass stops or bends, as RtShadowsEffect's rays soften its shadows
    let mid = 0.5 * (tile.boxMin + tile.boxMax);
    var c : vec3f;
    var dc : vec3f;
    if (task.kind == 0u) {
        c = task.origin + task.e1 * u + task.e2 * v;
        dc = task.axis;
    } else {
        c = task.origin;
        dc = normalize(task.axis + task.e1 * u + task.e2 * v);
    }
    var reach = max(dot(mid - c, dc), 1e-3);
    if (dc.y < -1e-3) {
        reach = max((tile.refY - c.y) / dc.y, reach);
    }
    let aim = c + dc * reach;
    // the emitter's point, from the cell's place in a 16 x 16 ordered dither (a Bayer matrix's
    // bits, the coarsest split of the emitter first): every aligned block of 2^k x 2^k cells
    // leaves all 4^k of its strata, so the photons a splat gathers in a penumbra leave all of
    // the emitter. Jittered within the finest stratum, turned each frame (R2).
    var xe = vec2f(0.0);
    for (var b = 0u; b < 4u; b++) {
        xe += vec2f(f32(((i ^ j) >> b) & 1u), f32((j >> b) & 1u)) * exp2(-f32(b + 1u));
    }
    xe = fract(xe + hash4(i, j, task.seed, task.light + 211u).xy / 16.0 + fract(vec2f(0.7548776662, 0.5698402910) * f32(task.seed)));
    var onEmitter = xe * 2.0 - 1.0;
    if (task.kind != 2u) {
        let r = sqrt(xe.x);
        let a = 6.2831853 * xe.y;
        onEmitter = r * vec2f(cos(a), sin(a));
    }
    let offset = task.emitterX * onEmitter.x + task.emitterY * onEmitter.y;
    var o : vec3f;
    var d : vec3f;
    if (task.kind == 0u) {
        // a direction within the sun's disk, from as far before the caster
        d = normalize(dc + offset);
        o = aim - d * reach;
    } else {
        o = c + offset;
        d = normalize(aim - o);
    }
    // unoccluded: the light taken away from just before the caster's box, past what stands between
    // it and the light
    var tStart = 0.0;
    if (cp.unoccluded != 0u) {
        let grow = task.bias * 4.0;
        let inv = 1.0 / select(d, vec3f(1e-12), abs(d) < vec3f(1e-12));
        let ta = (tile.boxMin - grow - o) * inv;
        let tb = (tile.boxMax + grow - o) * inv;
        let near = max(max(min(ta.x, tb.x), min(ta.y, tb.y)), min(ta.z, tb.z));
        let far = min(min(max(ta.x, tb.x), max(ta.y, tb.y)), max(ta.z, tb.z));
        if (far < max(near, 0.0)) { return; }
        tStart = max(near - task.bias, 0.0);
    }
    // the first surface: this caster's glass, or the photon is the direct light's (left to the
    // lights) or another caster's
    let first = kansei_rt_trace(o, d, tStart, cp.maxDistance, KANSEI_RT_SOLID | KANSEI_RT_GLASS_HITS);
    if (!first.found || !isGlass(first.triangle)) { return; }
    let h0 = o + d * first.t;
    if (!inBox(h0, tile.boxMin, tile.boxMax, task.bias * 4.0)) { return; }
    // whether the photon's light reaches the glass: what it brings through the glass is its own,
    // stopped by whatever stands between the emitter and the glass (the light it takes away is
    // weighed by the visibility with `unoccluded`, which has that)
    var reaches = true;
    if (tStart > 0.0) {
        reaches = !kansei_rt_trace(o, d, 0.0, tStart, KANSEI_RT_ANY_HIT | KANSEI_RT_SOLID).found;
    }
    var flux : vec3f;
    var spread : f32;
    if (task.kind == 0u) {
        // the sun reaches the plane only where nothing stands between (a room's walls)
        if (reaches && kansei_rt_trace(o, -d, 0.0, 1e5, KANSEI_RT_ANY_HIT | KANSEI_RT_SOLID).found) { reaches = false; }
        flux = task.color * (task.du * task.dv);
        spread = 0.0;
    } else {
        let cosT = dot(dc, task.axis);
        let omega = task.du * task.dv * cosT * cosT * cosT;
        let cone = saturate((dot(d, task.lightDir) - task.cosOuter) / max(task.cosInner - task.cosOuter, 1e-4));
        let r = first.t * first.t / (task.range * task.range);
        let window = saturate(1.0 - r * r);
        flux = task.color * (window * window * cone * cone * omega);
        spread = sqrt(omega);
    }
    if (max(flux.r, max(flux.g, flux.b)) <= 0.0 || (!reaches && cp.unoccluded == 0u)) { return; }
    // the tent: twice as wide as the photons lie apart where they land (m; the sun's are
    // parallel, a spot light's spread with the distance), so a few overlap every texel
    let apart = select(2.0 * sqrt(task.du * task.dv), 0.0, task.kind != 0u);
    spread *= 2.0;
    // straight on through no glass: the direct light the glass changes, taken away
    let straight = kansei_rt_trace(h0, d, 0.0, cp.maxDistance, KANSEI_RT_SOLID);
    if (straight.found && straight.normal.y > 0.2) {
        let p = h0 + d * straight.t;
        if (!inCaster(p)) {
            deposit(task.tile, task.layer + 1u, p, -flux * straight.normal.y, apart + spread * (first.t + straight.t));
        }
    }
    if (!reaches) { return; }
    // through the glass: `depth` the glasses the photon is in (a core inside a shell: two), each
    // entered where the ray meets a triangle's front (its winding's normal) and left at a back
    var hit = first;
    var origin = o;
    var dir = d;
    var throughput = vec3f(1.0);
    var depth = 0u;
    var absorbs = array<vec3f, 2>(vec3f(0.0), vec3f(0.0));
    var travelled = 0.0;
    for (var n = 0u; n < cp.interfaces; n++) {
        if (n > 0u) {
            hit = kansei_rt_trace(origin, dir, 0.0, cp.maxDistance, KANSEI_RT_SOLID | KANSEI_RT_GLASS_HITS);
            if (!hit.found) { return; }
        }
        let h = origin + dir * hit.t;
        travelled += hit.t;
        if (depth > 0u) {
            throughput *= exp(-absorbs[min(depth, 2u) - 1u] * hit.t);
        }
        let tri = hit.triangle * 4u;
        let front = dot(dir, cross(kansei_rt_triangles[tri + 1u].xyz, kansei_rt_triangles[tri + 2u].xyz)) < 0.0;
        if (!isGlass(hit.triangle)) {
            // a surface's back, just off the glass: the step off it went through a surface the
            // glass touches (a marble's table), so the photon lands where it left the glass
            let through = !front && n > 0u;
            let at = select(h, origin, through);
            if (hit.normal.y > 0.2 && !inCaster(at)) {
                let far = select(travelled, travelled - hit.t, through);
                deposit(task.tile, task.layer, at, flux * throughput * hit.normal.y * cp.focusGain, (apart + spread * far) * cp.focusSharpen);
            }
            return;
        }
        let entering = front;
        let m = kansei_rt_shading_normal(hit.triangle, hit.bary, hit.normal);
        let etaI = select(1.0, cp.glassIor, depth > 0u);
        let etaT = select(select(cp.glassIor, 1.0, depth <= 1u), cp.glassIor, entering);
        let f = fresnelDielectric(clamp(-dot(dir, m), 0.0, 1.0), etaI, etaT);
        let refracted = refract(dir, m, etaI / etaT);
        // reflected with Fresnel's probability (always past the critical angle)
        if (f >= 1.0 || dot(refracted, refracted) < 0.5 || hash4(i ^ (n << 16u), j, task.seed, task.light + 101u).x < f) {
            dir = reflect(dir, m);
            origin = h + hit.normal * task.bias;
            continue;
        }
        dir = normalize(refracted);
        origin = h - hit.normal * task.bias;
        if (entering) {
            if (depth < 2u) {
                let tint = clamp(kansei_rt_albedo(hit.triangle) * cp.glassTint, vec3f(1e-4), vec3f(1.0));
                absorbs[depth] = -log(tint) / max(cp.tintDistance, 1e-6);
            }
            depth += 1u;
        } else {
            depth = max(depth, 1u) - 1u;
        }
    }
}
