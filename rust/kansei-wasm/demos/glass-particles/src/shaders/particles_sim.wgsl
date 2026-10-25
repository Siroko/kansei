// The particles, one thread each: a dead one respawns at its emitter while the emitter's budget
// for the frame lasts; a live one is carried by curl noise (the curl of a noise potential, so the
// flow has vorticity and no sources or sinks), pushed off the repellers along their distance
// fields' gradients and kept out of them, then written as an instance the boxes draw: placed,
// turned to face where it moves and stretched with its speed, its albedo and glow packed as the
// ray tracing grid reads them (RT_EMISSION_WGSL's kansei_rt_pack_emission, prepended). The pointer
// pushes the particles near the ray from the eye through it along its motion, and the respawns
// and the hits on the repellers are counted for the page's sounds.

struct Emitter {
    // its rotation: its local axes in world space (it points along y)
    axisX    : vec4f,
    axisY    : vec4f,
    axisZ    : vec4f,
    // xyz, w: the shape (0 a point, 1 a ball, 2 a disc across y)
    position : vec4f,
    // the shape's radius (m), the launch speed (m/s), the spread (the cone's half angle,
    // radians), the lifetime (s)
    launch   : vec4f,
    // the size (m), how much the speed and the lifetime vary (0..1), the share that glows
    jitter   : vec4f,
    // its particles (first, count), how many respawn this frame, whether it runs
    pool     : vec4u,
}

struct Collider {
    // world to local: the rows of its rotation (xyz); row0.w its kind (0 a sphere, 1 a box,
    // 2 a pyramid)
    row0   : vec4f,
    row1   : vec4f,
    row2   : vec4f,
    center : vec4f,
    // a sphere's radius; a box's half extents; a pyramid's base side and height
    size   : vec4f,
}

struct Sim {
    dt            : f32,
    time          : f32,
    // the noise's frequency (1/m) and the speed of the flow it drives (m/s)
    curlFrequency : f32,
    curlSpeed     : f32,
    // how fast a particle takes up the flow's velocity (1/s), and gravity (m/s², up)
    drag          : f32,
    gravity       : f32,
    // how far the repellers reach past their surface (m), and how hard they push (m/s²)
    repelRange    : f32,
    repelStrength : f32,
    colliderCount : u32,
    frame         : u32,
    // the glowing particles' radiance (cd/m²)
    glow          : f32,
    // how much a box stretches along its velocity at 1 m/s
    stretch       : f32,
    // how fast the noise drifts (its field slides through itself, m/s)
    curlEvolve    : f32,
    // the share of a hit's normal speed it keeps
    bounce        : f32,
    // a hit counts as one (for the sounds) from this speed into the surface (m/s)
    hitSpeed      : f32,
    _pad0         : f32,
    // past these a particle dies; the floor is at boundsMin.y
    boundsMin     : vec4f,
    boundsMax     : vec4f,
    // the pointer: the ray from the eye through it (origin, w: its strength 0..1; direction,
    // w: the radius round it it reaches, m) and its velocity across the view (m/s; w: how fast
    // the particles near it take it up, 1/s)
    mouseOrigin   : vec4f,
    mouseRay      : vec4f,
    mouseMotion   : vec4f,
    // the palette: albedos, and the glowing particles' colours
    base          : array<vec4f, 4>,
    emissive      : array<vec4f, 2>,
    emitters      : array<Emitter, 2>,
    colliders     : array<Collider, 16>,
}

struct Particle {
    // xyz, w: age (s)
    p : vec4f,
    // xyz, w: lifetime (s); dead once the age reaches it
    v : vec4f,
}

// What the boxes draw (and the ray tracing grid gathers): 64 bytes.
struct Instance {
    // xyz, w: the size
    placement : vec4f,
    // a unit quaternion turning +z to the velocity
    rotation  : vec4f,
    // the velocity (for the motion vectors), w: the stretch along z
    motion    : vec4f,
    // the albedo (unorm4x8) and the glow (kansei_rt_pack_emission)
    surface   : vec4u,
}

@group(0) @binding(0) var<uniform> sim : Sim;
@group(0) @binding(1) var<storage, read_write> particles : array<Particle>;
@group(0) @binding(2) var<storage, read_write> instances : array<Instance>;
// how many of each emitter's particles took a respawn this frame, then the live particles
// written this frame (the instances are compacted: the live ones first, particles_clear.wgsl
// clearing the rest), and an unused word
@group(0) @binding(3) var<storage, read_write> spawned : array<atomic<u32>, 4>;
// what the sounds read back (lib.rs's `take_events`): the respawns of each emitter, the brushes
// with spheres, boxes and pyramids (into the inner half of their reach), the hits on the floor,
// their speed toward the surface in sum (cm/s), and the particles alive (summed over the frames
// between readbacks)
@group(0) @binding(4) var<storage, read_write> events : array<atomic<u32>, 8>;

// ---- hashing ----

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn hash01(v: u32) -> f32 {
    return f32(pcg(v) >> 8u) / 16777216.0;
}

// ---- noise ----

// A gradient at lattice point `c` (each component in -1..1).
fn latticeGradient(c: vec3i) -> vec3f {
    let h = pcg((u32(c.x) * 73856093u) ^ pcg((u32(c.y) * 19349663u) ^ pcg(u32(c.z) * 83492791u)));
    let g = vec3f(f32(h & 1023u), f32((h >> 10u) & 1023u), f32((h >> 20u) & 1023u)) / 511.5 - 1.0;
    return g;
}

// Gradient noise and its analytic gradient (Inigo Quilez, "gradient noise derivatives"):
// x the value, yzw d/dx, d/dy, d/dz.
fn noised(x: vec3f) -> vec4f {
    let i = vec3i(floor(x));
    let f = fract(x);
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let du = 30.0 * f * f * (f * (f - 2.0) + 1.0);
    let ga = latticeGradient(i + vec3i(0, 0, 0));
    let gb = latticeGradient(i + vec3i(1, 0, 0));
    let gc = latticeGradient(i + vec3i(0, 1, 0));
    let gd = latticeGradient(i + vec3i(1, 1, 0));
    let ge = latticeGradient(i + vec3i(0, 0, 1));
    let gf = latticeGradient(i + vec3i(1, 0, 1));
    let gg = latticeGradient(i + vec3i(0, 1, 1));
    let gh = latticeGradient(i + vec3i(1, 1, 1));
    let va = dot(ga, f - vec3f(0.0, 0.0, 0.0));
    let vb = dot(gb, f - vec3f(1.0, 0.0, 0.0));
    let vc = dot(gc, f - vec3f(0.0, 1.0, 0.0));
    let vd = dot(gd, f - vec3f(1.0, 1.0, 0.0));
    let ve = dot(ge, f - vec3f(0.0, 0.0, 1.0));
    let vf = dot(gf, f - vec3f(1.0, 0.0, 1.0));
    let vg = dot(gg, f - vec3f(0.0, 1.0, 1.0));
    let vh = dot(gh, f - vec3f(1.0, 1.0, 1.0));
    let value = va + u.x * (vb - va) + u.y * (vc - va) + u.z * (ve - va) + u.x * u.y * (va - vb - vc + vd)
        + u.y * u.z * (va - vc - ve + vg) + u.z * u.x * (va - vb - ve + vf) + (-va + vb + vc - vd + ve - vf - vg + vh) * u.x * u.y * u.z;
    let gradient = ga + u.x * (gb - ga) + u.y * (gc - ga) + u.z * (ge - ga) + u.x * u.y * (ga - gb - gc + gd)
        + u.y * u.z * (ga - gc - ge + gg) + u.z * u.x * (ga - gb - ge + gf) + (-ga + gb + gc - gd + ge - gf - gg + gh) * u.x * u.y * u.z
        + du * (vec3f(vb, vc, ve) - va + u.yzx * vec3f(va - vb - vc + vd, va - vc - ve + vg, va - vb - ve + vf)
        + u.zxy * vec3f(va - vb - ve + vf, va - vb - vc + vd, va - vc - ve + vg) + u.yzx * u.zxy * (-va + vb + vc - vd + ve - vf - vg + vh));
    return vec4f(value, gradient);
}

// The curl of a potential of three noise fields (Bridson et al. 2007, "Curl-noise for procedural
// fluid flow"), two octaves: divergence free, so the particles swirl without bunching up.
fn curlNoise(p: vec3f) -> vec3f {
    var curl = vec3f(0.0);
    var frequency = sim.curlFrequency;
    var amplitude = 1.0;
    for (var octave = 0u; octave < 2u; octave++) {
        let q = p * frequency + vec3f(0.31, 1.0, 0.17) * sim.time * sim.curlEvolve * frequency + f32(octave) * 7.13;
        let a = noised(q).yzw;
        let b = noised(q + vec3f(31.4, 17.9, 5.3)).yzw;
        let c = noised(q + vec3f(-11.2, 43.7, 23.1)).yzw;
        // potential (a, b, c): curl = (dc/dy - db/dz, da/dz - dc/dx, db/dx - da/dy)
        curl += amplitude * vec3f(c.y - b.z, a.z - c.x, b.x - a.y);
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return curl / 1.5;
}

// ---- the repellers' distance fields (Inigo Quilez's) ----

fn sdBox(p: vec3f, b: vec3f) -> f32 {
    let q = abs(p) - b;
    return length(max(q, vec3f(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
}

// A pyramid of base side 1 on y = 0 round the origin, apex at height h.
fn sdPyramid(p0: vec3f, h: f32) -> f32 {
    let m2 = h * h + 0.25;
    var p = p0;
    p = vec3f(abs(p.x), p.y, abs(p.z));
    if (p.z > p.x) {
        p = vec3f(p.z, p.y, p.x);
    }
    p = vec3f(p.x - 0.5, p.y, p.z - 0.5);
    let q = vec3f(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);
    let s = max(-q.x, 0.0);
    let t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);
    let a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    let b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t) + (q.y - m2 * t) * (q.y - m2 * t);
    var d2 = min(a, b);
    if (min(q.y, -q.x * m2 - q.y * 0.5) > 0.0) {
        d2 = 0.0;
    }
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}

// Collider `c`'s distance at local point `p`.
fn colliderDistance(c: Collider, p: vec3f) -> f32 {
    let kind = u32(c.row0.w);
    if (kind == 0u) {
        return length(p) - c.size.x;
    }
    if (kind == 1u) {
        return sdBox(p, c.size.xyz);
    }
    let side = c.size.x;
    return sdPyramid(p / side, c.size.y / side) * side;
}

fn toLocal(c: Collider, p: vec3f) -> vec3f {
    let d = p - c.center.xyz;
    return vec3f(dot(c.row0.xyz, d), dot(c.row1.xyz, d), dot(c.row2.xyz, d));
}

// The outward normal at local point `p`, in world space (the distance's gradient, by central
// differences).
fn colliderNormal(c: Collider, p: vec3f) -> vec3f {
    let e = 0.004;
    let g = vec3f(
        colliderDistance(c, p + vec3f(e, 0.0, 0.0)) - colliderDistance(c, p - vec3f(e, 0.0, 0.0)),
        colliderDistance(c, p + vec3f(0.0, e, 0.0)) - colliderDistance(c, p - vec3f(0.0, e, 0.0)),
        colliderDistance(c, p + vec3f(0.0, 0.0, e)) - colliderDistance(c, p - vec3f(0.0, 0.0, e)),
    );
    let n = c.row0.xyz * g.x + c.row1.xyz * g.y + c.row2.xyz * g.z;
    let len = length(n);
    if (len < 1e-8) {
        return vec3f(0.0, 1.0, 0.0);
    }
    return n / len;
}

// ---- the particle ----

// A unit quaternion turning +z to unit vector `d` (the shortest arc).
fn arcFromZ(d: vec3f) -> vec4f {
    if (d.z < -0.9999) {
        return vec4f(0.0, 1.0, 0.0, 0.0);
    }
    return normalize(vec4f(-d.y, d.x, 0.0, 1.0 + d.z));
}

fn emitterToWorld(e: Emitter, v: vec3f) -> vec3f {
    return e.axisX.xyz * v.x + e.axisY.xyz * v.y + e.axisZ.xyz * v.z;
}

fn spawn(i: u32, e: Emitter) -> Particle {
    let seed = i * 9u + sim.frame * 0x9E3779B9u;
    let u = vec4f(hash01(seed), hash01(seed + 1u), hash01(seed + 2u), hash01(seed + 3u));
    // where on the shape
    var offset = vec3f(0.0);
    let shape = u32(e.position.w);
    let phi = 6.2831853 * u.x;
    if (shape == 1u) {
        let z = 2.0 * u.y - 1.0;
        let r = sqrt(max(1.0 - z * z, 0.0));
        offset = vec3f(r * cos(phi), z, r * sin(phi)) * e.launch.x * pow(u.z, 1.0 / 3.0);
    } else if (shape == 2u) {
        offset = vec3f(cos(phi), 0.0, sin(phi)) * e.launch.x * sqrt(u.y);
    }
    // which way: uniform over the cone round y
    let w = vec2f(hash01(seed + 4u), hash01(seed + 5u));
    let cosT = mix(1.0, cos(e.launch.z), w.x);
    let sinT = sqrt(max(1.0 - cosT * cosT, 0.0));
    let a = 6.2831853 * w.y;
    let dir = vec3f(sinT * cos(a), cosT, sinT * sin(a));
    let speed = e.launch.y * (1.0 + e.jitter.y * (2.0 * u.w - 1.0));
    let life = e.launch.w * (1.0 + e.jitter.z * (2.0 * hash01(seed + 6u) - 1.0));
    var p: Particle;
    p.p = vec4f(e.position.xyz + emitterToWorld(e, offset), 0.0);
    p.v = vec4f(emitterToWorld(e, dir) * speed, max(life, 0.05));
    return p;
}

// A hit for the sounds: on what (events[slot]), how fast.
fn hit(slot: u32, speed: f32) {
    atomicAdd(&events[slot], 1u);
    atomicAdd(&events[6], u32(speed * 100.0));
}

fn advance(q: Particle, e: Emitter) -> Particle {
    let dt = sim.dt;
    var p = q.p.xyz;
    var v = q.v.xyz;
    let radius = 0.5 * e.jitter.x;
    // the flow takes the particle up, gravity pulls it
    let flow = curlNoise(p) * sim.curlSpeed;
    v += (sim.drag * (flow - v) + vec3f(0.0, sim.gravity, 0.0)) * dt;
    // the pointer: a push along its motion, fading out from the ray through it
    let toP = p - sim.mouseOrigin.xyz;
    let along = dot(toP, sim.mouseRay.xyz);
    let off = length(toP - sim.mouseRay.xyz * along);
    if (along > 0.0 && off < sim.mouseRay.w) {
        let w = 1.0 - off / sim.mouseRay.w;
        v += sim.mouseMotion.xyz * (sim.mouseMotion.w * sim.mouseOrigin.w * w * w * dt);
    }
    p += v * dt;
    // the repellers: a push off each within reach, growing toward its surface, and no way in
    for (var k = 0u; k < sim.colliderCount; k++) {
        let c = sim.colliders[k];
        let local = toLocal(c, p);
        let d = colliderDistance(c, local);
        if (d >= sim.repelRange) {
            continue;
        }
        let n = colliderNormal(c, local);
        // a brush for the sounds: crossing into the inner half of its reach, toward it
        let touch = 0.5 * sim.repelRange;
        if (d < touch && colliderDistance(c, toLocal(c, q.p.xyz)) >= touch && -dot(v, n) > sim.hitSpeed) {
            hit(min(u32(c.row0.w), 2u) + 2u, -dot(v, n));
        }
        let falloff = 1.0 - max(d, 0.0) / sim.repelRange;
        v += n * sim.repelStrength * falloff * falloff * dt;
        if (d < radius) {
            p += n * (radius - d);
            let vn = dot(v, n);
            if (vn < 0.0) {
                v -= (1.0 + sim.bounce) * vn * n;
            }
        }
    }
    // the floor
    let floorY = sim.boundsMin.y + radius;
    if (p.y < floorY) {
        if (-v.y > sim.hitSpeed) {
            hit(5u, -v.y);
        }
        p.y = floorY;
        v.y = abs(v.y) * sim.bounce;
    }
    var out: Particle;
    out.p = vec4f(p, q.p.w + dt);
    out.v = vec4f(v, q.v.w);
    // out of the box: dead
    if (any(p < sim.boundsMin.xyz - vec3f(0.0, 1.0, 0.0)) || any(p > sim.boundsMax.xyz)) {
        out.p.w = out.v.w;
    }
    return out;
}

fn instanceOf(i: u32, q: Particle, e: Emitter) -> Instance {
    var out: Instance;
    let v = q.v.xyz;
    let speed = length(v);
    let dir = select(vec3f(0.0, 1.0, 0.0), v / max(speed, 1e-6), speed > 1e-4);
    // grows in over a fifth of a second and shrinks away over the last half second
    let age = q.p.w;
    let life = q.v.w;
    let fade = smoothstep(0.0, 0.2, age) * (1.0 - smoothstep(life - 0.5, life, age));
    out.placement = vec4f(q.p.xyz, e.jitter.x * fade);
    out.rotation = arcFromZ(dir);
    out.motion = vec4f(v, 1.0 + sim.stretch * min(speed, 3.0));
    // the colour is the particle's for good: its index picks it
    let h = hash01(i * 3u + 101u);
    let glows = hash01(i * 3u + 102u) < e.jitter.w;
    let pick = hash01(i * 3u + 103u);
    var albedo = sim.base[min(u32(pick * 4.0), 3u)].rgb;
    var glow = vec3f(0.0);
    if (glows) {
        let colour = sim.emissive[select(0u, 1u, h > 0.5)].rgb;
        albedo = colour * 0.2;
        glow = colour * sim.glow;
    }
    out.surface = vec4u(pack4x8unorm(vec4f(albedo, 1.0)), kansei_rt_pack_emission(glow), 0u, 0u);
    return out;
}

// the workgroup's live particles (each one's place among them), and where they go in the
// compacted instances
var<workgroup> alive : atomic<u32>;
var<workgroup> base : u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u, @builtin(local_invocation_index) lane: u32) {
    let i = gid.x;
    var live = false;
    var instance: Instance;
    var at = 0u;
    if (i < arrayLength(&particles)) {
        let which = select(0u, 1u, i >= sim.emitters[1].pool.x);
        let e = sim.emitters[which];
        var q = particles[i];
        if (q.p.w >= q.v.w) {
            if (e.pool.w != 0u && atomicAdd(&spawned[which], 1u) < e.pool.z) {
                q = spawn(i, e);
                atomicAdd(&events[which], 1u);
            }
        } else {
            q = advance(q, e);
        }
        particles[i] = q;
        live = q.p.w < q.v.w;
        if (live) {
            instance = instanceOf(i, q, e);
            at = atomicAdd(&alive, 1u);
        }
    }
    workgroupBarrier();
    if (lane == 0u) {
        let n = atomicLoad(&alive);
        base = atomicAdd(&spawned[2], n);
        atomicAdd(&events[7], n);
    }
    workgroupBarrier();
    if (live) {
        instances[base + at] = instance;
    }
}
