export const bodySdfHelpers = /* wgsl */`

struct BodyState {
    pos: vec3<f32>,
    _pad0: f32,
    vel: vec3<f32>,
    _pad1: f32,
    angle: f32,
    angVel: f32,
    mass: f32,
    inertia: f32,
    restitution: f32,
    primStart: u32,
    primCount: u32,
    reactionMultiplier: f32,
    maxPushDist: f32,
    forceClampFactor: f32,
    rightingStrength: f32,
    linearDamping: f32,
    angularDamping: f32,
    density: f32,
    mouseScale: f32,
    _pad4: f32,
};

struct BodyPrimitive {
    primType: u32,
    param1: f32,
    param2: f32,
    localX: f32,
    localY: f32,
    localAngle: f32,
};

fn sdfCircle(p: vec2<f32>, radius: f32) -> f32 {
    return length(p) - radius;
}

fn sdfBox(p: vec2<f32>, halfSize: vec2<f32>) -> f32 {
    let d = abs(p) - halfSize;
    return length(max(d, vec2<f32>(0.0))) + min(max(d.x, d.y), 0.0);
}

fn sdfCapsule(p: vec2<f32>, radius: f32, halfLen: f32) -> f32 {
    let px = clamp(p.x, -halfLen, halfLen);
    return length(p - vec2<f32>(px, 0.0)) - radius;
}

fn sdfEllipse(p: vec2<f32>, ab: vec2<f32>) -> f32 {
    let pa = abs(p);
    var q = pa;
    var e = ab;
    if (e.x > e.y) { q = q.yx; e = e.yx; }
    let l = e.y * e.y - e.x * e.x;
    let m = e.x * q.x / l;
    let m2 = m * m;
    let n = e.y * q.y / l;
    let n2 = n * n;
    let c = (m2 + n2 - 1.0) / 3.0;
    let c3 = c * c * c;
    let qq = c3 + m2 * n2 * 2.0;
    let d = c3 + m2 * n2;
    let g = m + m * n2;
    var co: f32;
    if (d < 0.0) {
        let h = acos(qq / c3) / 3.0;
        let s = cos(h);
        let t = sin(h) * sqrt(3.0);
        let rx = sqrt(max(-c * (s + t + 2.0) + m2, 0.0));
        let ry = sqrt(max(-c * (s - t + 2.0) + m2, 0.0));
        co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) / 2.0;
    } else {
        let h = 2.0 * m * n * sqrt(d);
        let s = sign(qq + h) * pow(abs(qq + h), 1.0 / 3.0);
        let u = sign(qq - h) * pow(abs(qq - h), 1.0 / 3.0);
        let rx = -s - u - c * 4.0 + 2.0 * m2;
        let ry = (s - u) * sqrt(3.0);
        let rm = sqrt(rx * rx + ry * ry);
        co = (ry / sqrt(rm - rx) + 2.0 * g / rm - m) / 2.0;
    }
    let r = e * vec2<f32>(co, sqrt(max(1.0 - co * co, 0.0)));
    return length(r - q) * sign(q.y - r.y);
}

fn evaluateBodySDF(
    queryPos: vec2<f32>,
    body: BodyState,
    primitives: ptr<storage, array<BodyPrimitive>, read_write>
) -> f32 {
    let cosA = cos(-body.angle);
    let sinA = sin(-body.angle);
    let rel = queryPos - body.pos.xy;
    let local = vec2<f32>(
        rel.x * cosA - rel.y * sinA,
        rel.x * sinA + rel.y * cosA
    );

    var minDist = 999999.0;
    for (var i = body.primStart; i < body.primStart + body.primCount; i++) {
        let prim = (*primitives)[i];
        var p = local - vec2<f32>(prim.localX, prim.localY);
        // Per-primitive rotation
        if (abs(prim.localAngle) > 0.0001) {
            let cp = cos(-prim.localAngle);
            let sp = sin(-prim.localAngle);
            p = vec2<f32>(p.x * cp - p.y * sp, p.x * sp + p.y * cp);
        }

        var d: f32;
        if (prim.primType == 0u) {
            d = sdfCircle(p, prim.param1);
        } else if (prim.primType == 1u) {
            d = sdfBox(p, vec2<f32>(prim.param1, prim.param2));
        } else if (prim.primType == 2u) {
            d = sdfCapsule(p, prim.param1, prim.param2);
        } else {
            d = sdfEllipse(p, vec2<f32>(prim.param1, prim.param2));
        }
        minDist = min(minDist, d);
    }
    return minDist;
}

fn sdfGradient(
    queryPos: vec2<f32>,
    body: BodyState,
    primitives: ptr<storage, array<BodyPrimitive>, read_write>
) -> vec2<f32> {
    let eps = 0.01;
    let dx = evaluateBodySDF(queryPos + vec2<f32>(eps, 0.0), body, primitives)
           - evaluateBodySDF(queryPos - vec2<f32>(eps, 0.0), body, primitives);
    let dy = evaluateBodySDF(queryPos + vec2<f32>(0.0, eps), body, primitives)
           - evaluateBodySDF(queryPos - vec2<f32>(0.0, eps), body, primitives);
    let g = vec2<f32>(dx, dy);
    let l = length(g);
    if (l < 0.0001) { return vec2<f32>(0.0, 1.0); }
    return g / l;
}

fn cross2D(a: vec2<f32>, b: vec2<f32>) -> f32 {
    return a.x * b.y - a.y * b.x;
}

fn atomicAddF32(addr: ptr<storage, atomic<u32>, read_write>, value: f32) {
    var old = atomicLoad(addr);
    for (var i = 0u; i < 128u; i++) {
        let newVal = bitcast<f32>(old) + value;
        let result = atomicCompareExchangeWeak(addr, old, bitcast<u32>(newVal));
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}
`;
