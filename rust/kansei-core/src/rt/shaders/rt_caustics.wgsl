// Caustics (rt::RtCausticsEffect), the photons: from each light (`tasks`, made on the CPU), a grid
// of photons (one a cell of the light's view of the glass, jittered each frame) is traced through
// the grid of triangles; those whose first hit is glass (RtSurface::glass: the dragon, the water)
// carry the light through it, refracted at every surface they cross (exact dielectric Fresnel,
// Beer-Lambert inside; the water's surface normal from its height field and its ripples), to the
// first other surface, and add their flux there; the same photon taken straight on, through no
// glass (where the direct light, which passes through glass, lands), takes it away there. The map
// thus holds what the glass changes of the direct light: the light it focuses, and the shadow it
// leaves (its Fresnel losses and absorption, the light it bends elsewhere). Only surfaces that
// face up keep it (a horizontal map: the floor, the bed, the plinth's top), at the height they
// landed at. Each photon splats a tent about as wide as the photons lie apart there, into
// fixed-point atomics: rgb (lux * CAUSTICS_SCALE), the weight and the weighted height.
// Prefixed with rt_types.wgsl, RT_GRID_WGSL with its bindings in group 1, RT_OPAQUE_WGSL and
// WATER_RIPPLES_WGSL.

struct CausticsParams {
    mapMin          : vec2f,
    mapSize         : vec2f,
    mapDims         : vec2u,
    frame           : u32,
    taskCount       : u32,
    texel           : f32,
    blend           : f32,
    maxDistance     : f32,
    interfaces      : u32,
    waterMin        : vec2f,
    waterSize       : vec2f,
    waterDims       : vec2u,
    // the water's source in the grid (its renderable), or 0xffffffff
    waterSource     : u32,
    ripples         : f32,
    time            : f32,
    waterIor        : f32,
    glassIor        : f32,
    splatTexels     : f32,
    waterAbsorb     : vec3f,
    photons         : u32,
    glassAbsorb     : vec3f,
    // 1: photons through the water too (else the water's caustics are its ripples' focus)
    flags           : u32,
}

// A light's photons: kind 0 a directional light (parallel, from the plane origin + u e1 + v e2,
// along axis; color its illuminance), 1 a spot light (from origin, a point of its emitter
// rectangle emitterX, emitterY, toward axis + u e1 + v e2; color its intensity, cd).
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
    _pad0    : f32,
    _pad1    : f32,
}

const CAUSTICS_SCALE : f32 = 64.0;
const CAUSTICS_WEIGHT : f32 = 4096.0;

@group(0) @binding(0) var<uniform> cp : CausticsParams;
@group(0) @binding(1) var<storage, read> tasks : array<CausticTask>;
@group(0) @binding(2) var<storage, read_write> acc : array<atomic<i32>>;
@group(0) @binding(3) var waterHeights : texture_2d<f32>;

fn hash4(i: u32, j: u32, frame: u32, k: u32) -> vec4f {
    var h = i * 1973u + j * 9277u + frame * 26699u + k * 7919u;
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
    return (bitcast<u32>(kansei_rt_triangles[id * 4u].w) & KANSEI_RT_GLASS) != 0u;
}

fn waterHeight(c: vec2i, fallback: f32) -> f32 {
    let q = clamp(c, vec2i(0), vec2i(cp.waterDims) - 1);
    let h = textureLoad(waterHeights, vec2u(q), 0).r;
    return select(h, fallback, h < -1e8);
}

// The water's surface normal at world xz: its height field's slope, and the ripples.
fn waterNormal(xz: vec2f) -> vec3f {
    let c = vec2i(round((xz - cp.waterMin) / cp.waterSize * vec2f(cp.waterDims)));
    let h = waterHeight(c, 0.0);
    let s = cp.waterSize / vec2f(cp.waterDims);
    let gx = (waterHeight(c + vec2i(1, 0), h) - waterHeight(c - vec2i(1, 0), h)) / (2.0 * s.x);
    let gz = (waterHeight(c + vec2i(0, 1), h) - waterHeight(c - vec2i(0, 1), h)) / (2.0 * s.y);
    let r = kansei_water_ripples(xz, cp.time) * cp.ripples;
    return normalize(vec3f(-gx - r.x, 1.0, -gz - r.y));
}

// Splat irradiance `e` (lux: flux over a texel's area, times the receiver's n.y) at p, a tent of
// `radius` metres (at least a texel and a half, at most `splatTexels` texels).
fn deposit(p: vec3f, flux: vec3f, radius: f32) {
    let c = (p.xz - cp.mapMin) / cp.mapSize * vec2f(cp.mapDims) - 0.5;
    let r = clamp(radius / cp.texel, 1.0, cp.splatTexels);
    let k = i32(ceil(r));
    let base = vec2i(round(c));
    var sum = 0.0;
    for (var y = -k; y <= k; y++) {
        for (var x = -k; x <= k; x++) {
            sum += max(1.0 - length(vec2f(base + vec2i(x, y)) - c) / r, 0.0);
        }
    }
    if (sum <= 0.0) {
        return;
    }
    let e = flux / (cp.texel * cp.texel * sum);
    for (var y = -k; y <= k; y++) {
        for (var x = -k; x <= k; x++) {
            let q = base + vec2i(x, y);
            let w = max(1.0 - length(vec2f(q) - c) / r, 0.0);
            if (w <= 0.0 || any(q < vec2i(0)) || any(q >= vec2i(cp.mapDims))) {
                continue;
            }
            let at = (u32(q.y) * cp.mapDims.x + u32(q.x)) * 5u;
            let v = e * w * CAUSTICS_SCALE;
            atomicAdd(&acc[at], i32(round(v.r)));
            atomicAdd(&acc[at + 1u], i32(round(v.g)));
            atomicAdd(&acc[at + 2u], i32(round(v.b)));
            let wn = w / sum * CAUSTICS_WEIGHT;
            atomicAdd(&acc[at + 3u], i32(round(wn)));
            atomicAdd(&acc[at + 4u], i32(round(wn * p.y)));
        }
    }
}

@compute @workgroup_size(64)
fn photons(@builtin(global_invocation_id) gid : vec3u, @builtin(num_workgroups) groups : vec3u) {
    let id = gid.x + gid.y * groups.x * 64u;
    if (id >= cp.photons) { return; }
    var k = 0u;
    for (var t = 1u; t < cp.taskCount; t++) {
        if (id >= tasks[t].first) {
            k = t;
        }
    }
    let task = tasks[k];
    let local = id - task.first;
    if (local >= task.side * task.side) { return; }
    let i = local % task.side;
    let j = local / task.side;
    let xi = hash4(i, j, cp.frame, k);
    let u = task.u0 + (f32(i) + xi.x) * task.du;
    let v = task.v0 + (f32(j) + xi.y) * task.dv;
    var o : vec3f;
    var d : vec3f;
    if (task.kind == 0u) {
        o = task.origin + task.e1 * u + task.e2 * v;
        d = task.axis;
    } else {
        o = task.origin + task.emitterX * (xi.z * 2.0 - 1.0) + task.emitterY * (xi.w * 2.0 - 1.0);
        d = normalize(task.axis + task.e1 * u + task.e2 * v);
    }
    // the first surface: glass, or the photon is the direct light's (left to the lights)
    let first = kansei_rt_trace(o, d, 0.0, cp.maxDistance, KANSEI_RT_SOLID | KANSEI_RT_GLASS_HITS);
    if (!first.found || !isGlass(first.triangle)) { return; }
    if ((cp.flags & 1u) == 0u && kansei_rt_source(first.triangle) == cp.waterSource) { return; }
    var flux : vec3f;
    var spread : f32;
    if (task.kind == 0u) {
        // the sun reaches the plane only where nothing stands between (the room's walls)
        if (kansei_rt_trace(o, -d, 0.0, 1e5, KANSEI_RT_ANY_HIT | KANSEI_RT_SOLID).found) { return; }
        flux = task.color * (task.du * task.dv);
        spread = 0.0;
    } else {
        let cosT = dot(d, task.axis);
        let omega = task.du * task.dv * cosT * cosT * cosT;
        let cone = saturate((dot(d, task.lightDir) - task.cosOuter) / max(task.cosInner - task.cosOuter, 1e-4));
        let r = first.t * first.t / (task.range * task.range);
        let window = saturate(1.0 - r * r);
        flux = task.color * (window * window * cone * cone * omega);
        spread = sqrt(omega);
    }
    if (max(flux.r, max(flux.g, flux.b)) <= 0.0) { return; }
    // how far apart the photons land at a distance (m): the sun's are parallel
    let apart = sqrt(task.du * task.dv) * select(1.0, 0.0, task.kind != 0u);
    let h0 = o + d * first.t;
    // straight on through no glass: the direct light the glass changes, taken away
    let straight = kansei_rt_trace(h0, d, 0.0, cp.maxDistance, KANSEI_RT_SOLID);
    if (straight.found && straight.normal.y > 0.2) {
        deposit(h0 + d * straight.t, -flux * straight.normal.y, apart + spread * (first.t + straight.t));
    }
    // through the glass
    let step = 0.05 * kansei_rt_grid.cell;
    var hit = first;
    var origin = o;
    var dir = d;
    var throughput = vec3f(1.0);
    var inside = false;
    var ior = 1.0;
    var absorb = vec3f(0.0);
    var travelled = 0.0;
    for (var n = 0u; n < cp.interfaces; n++) {
        if (n > 0u) {
            hit = kansei_rt_trace(origin, dir, 0.0, cp.maxDistance, KANSEI_RT_SOLID | KANSEI_RT_GLASS_HITS);
            if (!hit.found) { return; }
        }
        let h = origin + dir * hit.t;
        travelled += hit.t;
        if (inside) {
            throughput *= exp(-absorb * hit.t);
        }
        if (!isGlass(hit.triangle)) {
            if (hit.normal.y > 0.2) {
                deposit(h, flux * throughput * hit.normal.y, apart + spread * travelled);
            }
            return;
        }
        let water = kansei_rt_source(hit.triangle) == cp.waterSource;
        var m : vec3f;
        if (water) {
            m = waterNormal(h.xz);
            if (dot(m, dir) > 0.0) {
                m = -m;
            }
        } else {
            m = kansei_rt_shading_normal(hit.triangle, hit.bary, hit.normal);
        }
        let surfaceIor = select(cp.glassIor, cp.waterIor, water);
        let etaI = select(1.0, ior, inside);
        let etaT = select(surfaceIor, 1.0, inside);
        let f = fresnelDielectric(clamp(-dot(dir, m), 0.0, 1.0), etaI, etaT);
        if (f >= 1.0) {
            dir = reflect(dir, m);
            origin = h + hit.normal * step;
            continue;
        }
        dir = refract(dir, m, etaI / etaT);
        if (dot(dir, dir) < 0.5) { return; }
        throughput *= 1.0 - f;
        origin = h - hit.normal * step;
        if (!inside) {
            ior = surfaceIor;
            absorb = select(cp.glassAbsorb, cp.waterAbsorb, water);
        }
        inside = !inside;
    }
}
