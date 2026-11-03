// The smoke's Curl noise and Fluid levels (SmokeSimulation::step), appended to smoke_common.wgsl.
// One entry point per stage; each reads its inputs as textures and writes one storage texture
// (WebGPU has no read-write storage textures by default), so the grids ping-pong.
//
// Coordinates are in cells: cell (i, j, k) spans [i, i + 1] x [j, j + 1] x [k, k + 1], its centre
// at (i + 0.5, j + 0.5, k + 0.5). The velocity grid is a MAC grid (Fedkiw, Stam and Jensen 2001):
// texel (i, j, k) holds u on the face x = i, v on y = j and w on z = k, in m/s along the box's
// axes; the faces x = 0, y = 0, z = 0 are walls and hold 0, and so do the outer faces (x = dims.x
// ...), which no texel holds.
//
// A Fluid step (Stam 1999, "Stable Fluids"; Fedkiw et al. 2001, "Visual Simulation of Smoke"):
//   sources:                the sources' density and temperature (scalars a -> b)
//   advectVelocity:         semi-Lagrangian, the velocity by itself (v0 -> v1)
//   advectScalarsForward:   MacCormack's forward step (b -> c)
//   advectScalarsCorrect:   its backward step and correction, Selle et al. 2008's clamp, and the
//                           dissipation and cooling (b, c -> a)
//   vorticity, forces:      vorticity confinement, buoyancy, the sources' velocity, the mouse
//                           (v1 -> v0)
//   divergence, jacobi xN, project: the pressure projection with closed walls, and the speed
//                           cap (v0 -> v1)
// The scalars are advected by the velocity the step starts with (the last projection's,
// divergence-free). A Curl noise step runs sources and the scalars' advection only, with the
// velocity from curl noise (smokeVelocityAt).

@group(0) @binding(0) var<uniform> st : SmokeStep;
@group(0) @binding(1) var smokeLinear : sampler;
@group(0) @binding(2) var velocityIn : texture_3d<f32>;
@group(0) @binding(3) var scalarsIn : texture_3d<f32>;
@group(0) @binding(4) var auxIn : texture_3d<f32>;       // MacCormack's forward scalars; the vorticity
@group(0) @binding(5) var vectorOut : texture_storage_3d<rgba16float, write>;
@group(0) @binding(6) var pressureIn : texture_3d<f32>;
@group(0) @binding(7) var divergenceIn : texture_3d<f32>;
@group(0) @binding(8) var scalarOut : texture_storage_3d<r32float, write>;

fn dimsF() -> vec3f {
    return vec3f(st.volume.dims);
}

// A cell's edge along each of the box's axes, metres.
fn cellH() -> vec3f {
    return st.volume.size / dimsF();
}

// The world position of a point in cells.
fn cellsToWorld(c: vec3f) -> vec3f {
    return smokeLocalToWorld(st.volume, c / dimsF());
}

// The MAC grid's components at a point in cells, interpolated (each on its own staggered grid).
fn macU(c: vec3f) -> f32 {
    let d = dimsF();
    return textureSampleLevel(velocityIn, smokeLinear, vec3f((c.x + 0.5) / d.x, c.y / d.y, c.z / d.z), 0.0).x;
}

fn macV(c: vec3f) -> f32 {
    let d = dimsF();
    return textureSampleLevel(velocityIn, smokeLinear, vec3f(c.x / d.x, (c.y + 0.5) / d.y, c.z / d.z), 0.0).y;
}

fn macW(c: vec3f) -> f32 {
    let d = dimsF();
    return textureSampleLevel(velocityIn, smokeLinear, vec3f(c.x / d.x, c.y / d.y, (c.z + 0.5) / d.z), 0.0).z;
}

// The gas's velocity at a point in cells, m/s along the box's axes: the MAC grid's on Fluid;
// on Curl noise the noise's (smokeCurlVelocity) and the mouse's push.
fn smokeVelocityAt(c: vec3f) -> vec3f {
    if (st.volume.motion == SMOKE_CURL) {
        let p = cellsToWorld(c);
        return smokeVecToBox(st.volume, smokeCurlVelocity(st.volume, p, st.time) + smokeExtraVelocity(p));
    }
    return vec3f(macU(c), macV(c), macW(c));
}

// Velocity the Curl noise level adds to its noise's at a world point: the mouse's push, round its
// ray (the gas it moves stays moved: a trail).
fn smokeExtraVelocity(p: vec3f) -> vec3f {
    if (st.mouseActive == 0u) { return vec3f(0.0); }
    return st.mousePush * smokeMouseFalloff(p, st.mouseOrigin, st.mouseDirection, st.mouseRadius);
}

// A point in cells carried back over the step (or forward, `dt` < 0) by the velocity at it.
fn trace(c: vec3f, dt: f32) -> vec3f {
    return clamp(c - smokeVelocityAt(c) / cellH() * dt, vec3f(0.0), dimsF());
}

fn backtrace(c: vec3f) -> vec3f {
    return trace(c, st.dt);
}

fn scalarsAt(t: texture_3d<f32>, c: vec3f) -> vec4f {
    return textureSampleLevel(t, smokeLinear, c / dimsF(), 0.0);
}

// How much of a source covers a point: 1 at its centre, 0 past its radius.
fn sourceWeight(s: SmokeSourceData, p: vec3f) -> f32 {
    let r = saturate(length(p - s.position) / s.radius);
    let f = 1.0 - r * r;
    return f * f;
}

fn inGrid(gid: vec3u) -> bool {
    return all(gid < st.volume.dims);
}

// ── sources ──

@compute @workgroup_size(4, 4, 4)
fn sources(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    var s = textureLoad(scalarsIn, gid, 0);
    let p = cellsToWorld(vec3f(gid) + 0.5);
    let relax = 1.0 - exp(-20.0 * st.dt);
    for (var i = 0u; i < st.volume.numSources; i++) {
        let src = st.volume.sources[i];
        let w = sourceWeight(src, p);
        if (w <= 0.0) { continue; }
        s.x += src.rate * st.dt * w;
        s.y = mix(s.y, st.sourceTemperature[i], w * relax);
    }
    textureStore(vectorOut, gid, s);
}

// ── advection ──

@compute @workgroup_size(4, 4, 4)
fn advectVelocity(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let f = vec3f(gid);
    var u = macU(backtrace(f + vec3f(0.0, 0.5, 0.5)));
    var v = macV(backtrace(f + vec3f(0.5, 0.0, 0.5)));
    var w = macW(backtrace(f + vec3f(0.5, 0.5, 0.0)));
    // the walls
    if (gid.x == 0u) { u = 0.0; }
    if (gid.y == 0u) { v = 0.0; }
    if (gid.z == 0u) { w = 0.0; }
    textureStore(vectorOut, gid, vec4f(u, v, w, 0.0));
}

@compute @workgroup_size(4, 4, 4)
fn advectScalarsForward(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let c = vec3f(gid) + 0.5;
    textureStore(vectorOut, gid, scalarsAt(scalarsIn, backtrace(c)));
}

@compute @workgroup_size(4, 4, 4)
fn advectScalarsCorrect(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let c = vec3f(gid) + 0.5;
    let vel = smokeVelocityAt(c) / cellH() * st.dt;
    let back = clamp(c - vel, vec3f(0.0), dimsF());
    let ahead = clamp(c + vel, vec3f(0.0), dimsF());
    let forward = textureLoad(auxIn, gid, 0);
    // the forward step carried back again: half its error corrects it (MacCormack)
    let again = scalarsAt(auxIn, ahead);
    var s = forward + 0.5 * (textureLoad(scalarsIn, gid, 0) - again);
    // clamped to the values the forward step interpolated between (Selle et al. 2008)
    let dims = vec3i(st.volume.dims);
    let base = vec3i(floor(back - 0.5));
    var lo = vec4f(1e30);
    var hi = vec4f(-1e30);
    for (var n = 0; n < 8; n++) {
        let q = clamp(base + vec3i(n & 1, (n >> 1) & 1, n >> 2), vec3i(0), dims - 1);
        let v = textureLoad(scalarsIn, q, 0);
        lo = min(lo, v);
        hi = max(hi, v);
    }
    s = clamp(s, lo, hi);
    s.x *= exp(-st.volume.dissipation * st.dt);
    s.y *= exp(-st.cooling * st.dt);
    textureStore(vectorOut, gid, vec4f(max(s.x, 0.0), s.y, 0.0, 0.0));
}

// ── forces ──

// The velocity at a cell's centre, from its faces (the outer faces hold 0).
fn cellVelocity(q: vec3i) -> vec3f {
    let dims = vec3i(st.volume.dims);
    let c = clamp(q, vec3i(0), dims - 1);
    let here = textureLoad(velocityIn, c, 0).xyz;
    var u1 = 0.0;
    var v1 = 0.0;
    var w1 = 0.0;
    if (c.x + 1 < dims.x) { u1 = textureLoad(velocityIn, c + vec3i(1, 0, 0), 0).x; }
    if (c.y + 1 < dims.y) { v1 = textureLoad(velocityIn, c + vec3i(0, 1, 0), 0).y; }
    if (c.z + 1 < dims.z) { w1 = textureLoad(velocityIn, c + vec3i(0, 0, 1), 0).z; }
    return 0.5 * (here + vec3f(u1, v1, w1));
}

// The vorticity at each cell's centre (xyz) and its magnitude (w), by central differences.
@compute @workgroup_size(4, 4, 4)
fn vorticity(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let q = vec3i(gid);
    let h = cellH();
    let dx = (cellVelocity(q + vec3i(1, 0, 0)) - cellVelocity(q - vec3i(1, 0, 0))) / (2.0 * h.x);
    let dy = (cellVelocity(q + vec3i(0, 1, 0)) - cellVelocity(q - vec3i(0, 1, 0))) / (2.0 * h.y);
    let dz = (cellVelocity(q + vec3i(0, 0, 1)) - cellVelocity(q - vec3i(0, 0, 1))) / (2.0 * h.z);
    let w = vec3f(dy.z - dz.y, dz.x - dx.z, dx.y - dy.x);
    textureStore(vectorOut, gid, vec4f(w, length(w)));
}

fn vorticityAt(q: vec3i) -> vec4f {
    return textureLoad(auxIn, clamp(q, vec3i(0), vec3i(st.volume.dims) - 1), 0);
}

// Vorticity confinement's force at a cell's centre (Fedkiw et al. 2001): eps h (N x omega), N
// the unit gradient of |omega|, h the cell.
fn confinement(q: vec3i) -> vec3f {
    let h = cellH();
    let grad = vec3f(
        (vorticityAt(q + vec3i(1, 0, 0)).w - vorticityAt(q - vec3i(1, 0, 0)).w) / h.x,
        (vorticityAt(q + vec3i(0, 1, 0)).w - vorticityAt(q - vec3i(0, 1, 0)).w) / h.y,
        (vorticityAt(q + vec3i(0, 0, 1)).w - vorticityAt(q - vec3i(0, 0, 1)).w) / h.z,
    );
    let len = length(grad);
    if (len < 1e-5) { return vec3f(0.0); }
    return st.vorticity * smokeCellSize(st.volume) * cross(grad / len, vorticityAt(q).xyz);
}

// Buoyancy at a cell's centre: hot gas rises, dense gas sinks.
fn buoyancyAt(q: vec3i) -> f32 {
    let s = textureLoad(scalarsIn, clamp(q, vec3i(0), vec3i(st.volume.dims) - 1), 0);
    return st.buoyancy * s.y - st.weight * s.x;
}

// The forces on one face's component `axis` at the face position `p` (in cells), from the cells
// either side of it (`q` and the one before it along the axis).
fn faceForce(axis: u32, q: vec3i, p: vec3f) -> f32 {
    var before = q;
    before[axis] -= 1;
    var f = 0.5 * (confinement(q)[axis] + confinement(before)[axis]);
    if (axis == 1u) {
        f += 0.5 * (buoyancyAt(q) + buoyancyAt(before));
    }
    f += smokeExtraForce(cellsToWorld(p))[axis];
    return f;
}

// Forces beyond the gas's own, at a world point, along the box's axes: the ambient draft (a
// weak, slow curl-noise force: it stirs hanging smoke without piling it against a wall). The
// mouse drags the gas (`mouseDrag`).
fn smokeExtraForce(p: vec3f) -> vec3f {
    var f = vec3f(0.0);
    if (st.ambientDraft != 0.0) {
        f += st.ambientDraft * curlNoise(p / st.draftScale, st.time * 0.05);
    }
    return smokeVecToBox(st.volume, f);
}

// The sources' velocity blended in at a face (`axis`, at `p` in cells): each source's own, and a
// curl-noise jitter of about half its speed over twice its radius, so a plume leaves its source
// unsteady and breaks into wisps instead of rising as a still column.
fn sourceVelocity(axis: u32, value: f32, p: vec3f) -> f32 {
    let world = cellsToWorld(p);
    let relax = 1.0 - exp(-20.0 * st.dt);
    var v = value;
    for (var i = 0u; i < st.volume.numSources; i++) {
        let src = st.volume.sources[i];
        let w = sourceWeight(src, world);
        if (w <= 0.0) { continue; }
        let speed = length(src.velocity);
        let jitter = curlNoise((world - src.position) / (2.0 * src.radius) + f32(i) * 17.0, st.time * 3.0) * (0.5 * speed);
        v = mix(v, smokeVecToBox(st.volume, src.velocity + jitter)[axis], w * relax);
    }
    return v;
}

// The mouse on Fluid: round its ray, the gas is dragged toward the push's velocity (about 20 per
// second), so a stir leaves a wake the solver carries on, and a push never adds more speed than
// it has.
fn mouseDrag(axis: u32, value: f32, p: vec3f) -> f32 {
    if (st.mouseActive == 0u) { return value; }
    let w = smokeMouseFalloff(cellsToWorld(p), st.mouseOrigin, st.mouseDirection, st.mouseRadius);
    if (w <= 0.0) { return value; }
    return mix(value, smokeVecToBox(st.volume, st.mousePush)[axis], w * (1.0 - exp(-20.0 * st.dt)));
}

@compute @workgroup_size(4, 4, 4)
fn forces(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let q = vec3i(gid);
    let f = vec3f(gid);
    var vel = textureLoad(velocityIn, gid, 0).xyz;
    var faces = array<vec3f, 3>(f + vec3f(0.0, 0.5, 0.5), f + vec3f(0.5, 0.0, 0.5), f + vec3f(0.5, 0.5, 0.0));
    for (var a = 0u; a < 3u; a++) {
        if (gid[a] == 0u) { vel[a] = 0.0; continue; }   // the wall
        var v = vel[a] + st.dt * faceForce(a, q, faces[a]);
        vel[a] = mouseDrag(a, sourceVelocity(a, v, faces[a]), faces[a]);
    }
    textureStore(vectorOut, gid, vec4f(vel, 0.0));
}

// ── the pressure projection ──

@compute @workgroup_size(4, 4, 4)
fn divergence(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let q = vec3i(gid);
    let dims = vec3i(st.volume.dims);
    let h = cellH();
    let here = textureLoad(velocityIn, q, 0).xyz;
    var next = vec3f(0.0);   // the outer walls hold 0
    if (q.x + 1 < dims.x) { next.x = textureLoad(velocityIn, q + vec3i(1, 0, 0), 0).x; }
    if (q.y + 1 < dims.y) { next.y = textureLoad(velocityIn, q + vec3i(0, 1, 0), 0).y; }
    if (q.z + 1 < dims.z) { next.z = textureLoad(velocityIn, q + vec3i(0, 0, 1), 0).z; }
    // the walls at the near faces hold 0 too
    let near = select(here, vec3f(0.0), q == vec3i(0));
    let d = (next - near) / h;
    textureStore(scalarOut, gid, vec4f(d.x + d.y + d.z, 0.0, 0.0, 0.0));
}

// One Jacobi iteration of laplacian(q) = divergence, the walls closed (no gradient across them:
// a neighbour outside drops out of the stencil).
@compute @workgroup_size(4, 4, 4)
fn jacobi(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let q = vec3i(gid);
    let dims = vec3i(st.volume.dims);
    let inv = 1.0 / (cellH() * cellH());
    var sum = 0.0;
    var weight = 0.0;
    for (var a = 0; a < 3; a++) {
        for (var s = -1; s <= 1; s += 2) {
            var n = q;
            n[a] += s;
            if (n[a] < 0 || n[a] >= dims[a]) { continue; }
            sum += textureLoad(pressureIn, n, 0).x * inv[a];
            weight += inv[a];
        }
    }
    let p = (sum - textureLoad(divergenceIn, q, 0).x) / max(weight, 1e-12);
    textureStore(scalarOut, gid, vec4f(p, 0.0, 0.0, 0.0));
}

// The velocity less the pressure's gradient (divergence-free), the walls at 0, and no texel
// faster than maxSpeed.
@compute @workgroup_size(4, 4, 4)
fn project(@builtin(global_invocation_id) gid : vec3u) {
    if (!inGrid(gid)) { return; }
    let q = vec3i(gid);
    var vel = textureLoad(velocityIn, q, 0).xyz;
    if (st.pressureIterations > 0u) {
        let h = cellH();
        let p = textureLoad(pressureIn, q, 0).x;
        for (var a = 0; a < 3; a++) {
            if (q[a] == 0) { continue; }
            var n = q;
            n[a] -= 1;
            vel[a] -= (p - textureLoad(pressureIn, n, 0).x) / h[a];
        }
    }
    vel = select(vel, vec3f(0.0), q == vec3i(0));
    let speed = length(vel);
    if (speed > st.maxSpeed) { vel *= st.maxSpeed / speed; }
    textureStore(vectorOut, gid, vec4f(vel, 0.0));
}
