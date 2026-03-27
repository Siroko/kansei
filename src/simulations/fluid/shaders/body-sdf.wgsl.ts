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
    _pad2: f32,
};

struct BodyPrimitive {
    primType: u32,
    param1: f32,
    param2: f32,
    localX: f32,
    localY: f32,
    _pad: f32,
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
        let p = local - vec2<f32>(prim.localX, prim.localY);

        var d: f32;
        if (prim.primType == 0u) {
            d = sdfCircle(p, prim.param1);
        } else if (prim.primType == 1u) {
            d = sdfBox(p, vec2<f32>(prim.param1, prim.param2));
        } else {
            d = sdfCapsule(p, prim.param1, prim.param2);
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
    loop {
        let newVal = bitcast<f32>(old) + value;
        let result = atomicCompareExchangeWeak(addr, old, bitcast<u32>(newVal));
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}
`;
