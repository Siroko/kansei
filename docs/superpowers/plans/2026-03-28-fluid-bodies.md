# Fluid Bodies — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SDF-based rigid bodies to the fluid simulation that float, bob, and rotate with two-way fluid coupling, rendered via GPU-driven instanced transforms.

**Architecture:** Two new compute passes (body-collision + body-integrate) appended to the existing 10-pass SPH pipeline. Fluid particles are projected out of body SDFs with reaction forces accumulated via atomic float CAS. Body transforms are written to a STORAGE|VERTEX buffer for zero-readback instanced rendering.

**Tech Stack:** WebGPU compute shaders (WGSL), TypeScript

**Spec:** `docs/superpowers/specs/2026-03-28-fluid-bodies-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/simulations/fluid/FluidBody.ts` | FluidBody class, FluidBodyPrimitive/Options interfaces |
| Create | `src/simulations/fluid/shaders/body-sdf.wgsl.ts` | Shared SDF evaluation + gradient + atomicAddF32 helpers |
| Create | `src/simulations/fluid/shaders/body-collision.wgsl.ts` | Fluid-body collision pass (push particles, accumulate forces) |
| Create | `src/simulations/fluid/shaders/body-integrate.wgsl.ts` | Body rigid body integration + transform output |
| Modify | `src/simulations/fluid/FluidSimulation.ts` | Body management, buffers, 2 new compute passes in pipeline |
| Modify | `src/main.ts` | Export FluidBody |
| Modify | `examples/index_fluid_text.html` | Add a floating body demo |

---

## Task 1: Create FluidBody class

**Files:**
- Create: `src/simulations/fluid/FluidBody.ts`

- [ ] **Step 1: Create the FluidBody class and interfaces**

```typescript
import { Vector3 } from '../../math/Vector3';

export interface FluidBodyPrimitive {
    type: 'circle' | 'box' | 'capsule';
    radius?: number;       // circle: radius, capsule: radius
    halfW?: number;        // box: half width
    halfH?: number;        // box: half height
    halfLength?: number;   // capsule: half length
    localX?: number;
    localY?: number;
}

export interface FluidBodyOptions {
    primitives: FluidBodyPrimitive[];
    position?: Vector3;
    velocity?: Vector3;
    angle?: number;
    angularVelocity?: number;
    mass?: number;
    restitution?: number;
}

const PRIMITIVE_TYPE_MAP: Record<string, number> = {
    circle: 0,
    box: 1,
    capsule: 2,
};

class FluidBody {
    public position: Vector3;
    public velocity: Vector3;
    public angle: number;
    public angularVelocity: number;
    public mass: number;
    public restitution: number;
    public primitives: FluidBodyPrimitive[];

    readonly index: number;
    readonly primitiveStart: number;
    readonly primitiveCount: number;

    constructor(options: FluidBodyOptions, index: number, primitiveStart: number) {
        this.position = options.position ?? new Vector3();
        this.velocity = options.velocity ?? new Vector3();
        this.angle = options.angle ?? 0;
        this.angularVelocity = options.angularVelocity ?? 0;
        this.mass = options.mass ?? 1.0;
        this.restitution = options.restitution ?? 0.3;
        this.primitives = options.primitives;
        this.index = index;
        this.primitiveStart = primitiveStart;
        this.primitiveCount = options.primitives.length;
    }

    /**
     * Compute moment of inertia from primitives (2D, around Z axis).
     * Approximation: sum of each primitive's inertia about the body center.
     */
    computeInertia(): number {
        let inertia = 0;
        for (const prim of this.primitives) {
            const ox = prim.localX ?? 0;
            const oy = prim.localY ?? 0;
            const offsetSq = ox * ox + oy * oy;
            switch (prim.type) {
                case 'circle': {
                    const r = prim.radius ?? 1;
                    // I = 0.5 * m * r^2 + m * d^2 (parallel axis)
                    // Use area as proxy for mass contribution
                    const area = Math.PI * r * r;
                    inertia += area * (0.5 * r * r + offsetSq);
                    break;
                }
                case 'box': {
                    const w = (prim.halfW ?? 1) * 2;
                    const h = (prim.halfH ?? 1) * 2;
                    const area = w * h;
                    inertia += area * ((w * w + h * h) / 12 + offsetSq);
                    break;
                }
                case 'capsule': {
                    const r = prim.radius ?? 0.5;
                    const l = (prim.halfLength ?? 1) * 2;
                    const area = Math.PI * r * r + 2 * r * l;
                    inertia += area * (r * r * 0.5 + l * l / 12 + offsetSq);
                    break;
                }
            }
        }
        // Scale by mass / total area to get actual inertia
        let totalArea = 0;
        for (const prim of this.primitives) {
            switch (prim.type) {
                case 'circle': { const r = prim.radius ?? 1; totalArea += Math.PI * r * r; break; }
                case 'box': { totalArea += (prim.halfW ?? 1) * 2 * (prim.halfH ?? 1) * 2; break; }
                case 'capsule': { const r = prim.radius ?? 0.5; totalArea += Math.PI * r * r + 2 * r * (prim.halfLength ?? 1) * 2; break; }
            }
        }
        return totalArea > 0 ? inertia * (this.mass / totalArea) : this.mass;
    }

    /**
     * Pack this primitive into a Float32Array at the given offset.
     * Layout per primitive: [type(u32), param1, param2, localX, localY, _pad] = 6 floats
     */
    static packPrimitive(prim: FluidBodyPrimitive, f32: Float32Array, u32: Uint32Array, offset: number): void {
        u32[offset] = PRIMITIVE_TYPE_MAP[prim.type] ?? 0;
        switch (prim.type) {
            case 'circle':
                f32[offset + 1] = prim.radius ?? 1;
                f32[offset + 2] = 0;
                break;
            case 'box':
                f32[offset + 1] = prim.halfW ?? 1;
                f32[offset + 2] = prim.halfH ?? 1;
                break;
            case 'capsule':
                f32[offset + 1] = prim.radius ?? 0.5;
                f32[offset + 2] = prim.halfLength ?? 1;
                break;
        }
        f32[offset + 3] = prim.localX ?? 0;
        f32[offset + 4] = prim.localY ?? 0;
        f32[offset + 5] = 0; // pad
    }

    /**
     * Pack body state into a Float32Array at the given offset.
     * Layout: [pos.x, pos.y, pos.z, _pad, vel.x, vel.y, vel.z, _pad,
     *          angle, angVel, mass, inertia, restitution, primStart(u32), primCount(u32), _pad]
     * = 16 floats = 64 bytes
     */
    packState(f32: Float32Array, u32: Uint32Array, offset: number): void {
        f32[offset + 0] = this.position.x;
        f32[offset + 1] = this.position.y;
        f32[offset + 2] = this.position.z;
        f32[offset + 3] = 0; // pad
        f32[offset + 4] = this.velocity.x;
        f32[offset + 5] = this.velocity.y;
        f32[offset + 6] = this.velocity.z;
        f32[offset + 7] = 0; // pad
        f32[offset + 8] = this.angle;
        f32[offset + 9] = this.angularVelocity;
        f32[offset + 10] = this.mass;
        f32[offset + 11] = this.computeInertia();
        f32[offset + 12] = this.restitution;
        u32[offset + 13] = this.primitiveStart;
        u32[offset + 14] = this.primitiveCount;
        f32[offset + 15] = 0; // pad
    }
}

export { FluidBody, PRIMITIVE_TYPE_MAP };
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/FluidBody.ts
git commit -m "feat: add FluidBody class with SDF primitive packing and inertia computation"
```

---

## Task 2: Create shared SDF shader helpers

**Files:**
- Create: `src/simulations/fluid/shaders/body-sdf.wgsl.ts`

- [ ] **Step 1: Create the shared SDF + atomicAddF32 WGSL code**

```typescript
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
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/body-sdf.wgsl.ts
git commit -m "feat: add shared SDF evaluation, gradient, and atomicAddF32 helpers"
```

---

## Task 3: Create body-collision shader

**Files:**
- Create: `src/simulations/fluid/shaders/body-collision.wgsl.ts`

- [ ] **Step 1: Create the fluid-body collision shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';
import { bodySdfHelpers } from './body-sdf.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}
${bodySdfHelpers}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> bodyStates: array<BodyState>;
@group(0) @binding(3) var<storage, read_write> bodyPrimitives: array<BodyPrimitive>;
@group(0) @binding(4) var<storage, read_write> bodyForces: array<atomic<u32>>;
@group(0) @binding(5) var<uniform> params: SimParams;
@group(0) @binding(6) var<uniform> bodyCount: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var pos = positions[idx];
    var vel = velocities[idx];
    let particlePos = pos.xy;

    for (var b = 0u; b < bodyCount; b++) {
        let body = bodyStates[b];
        let sdfVal = evaluateBodySDF(particlePos, body, &bodyPrimitives);

        if (sdfVal < 0.0) {
            let gradient = sdfGradient(particlePos, body, &bodyPrimitives);
            let pushDist = -sdfVal;

            // Push particle out
            let newPos = particlePos + gradient * pushDist;
            pos.x = newPos.x;
            pos.y = newPos.y;

            // Reflect velocity
            let velXY = vel.xy;
            let dotVN = dot(velXY, gradient);
            let reflected = velXY - 2.0 * dotVN * gradient;
            vel.x = reflected.x * body.restitution;
            vel.y = reflected.y * body.restitution;

            // Accumulate reaction force on body (Newton's 3rd law)
            let reactionForce = -gradient * pushDist * 50.0;
            let r = newPos - body.pos.xy;
            let torque = cross2D(r, reactionForce);

            // 3 atomic floats per body: forceX, forceY, torque (indices b*4, b*4+1, b*4+2)
            atomicAddF32(&bodyForces[b * 4u + 0u], reactionForce.x);
            atomicAddF32(&bodyForces[b * 4u + 1u], reactionForce.y);
            atomicAddF32(&bodyForces[b * 4u + 2u], torque);
        }
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/body-collision.wgsl.ts
git commit -m "feat: add fluid-body collision shader with SDF projection and force accumulation"
```

---

## Task 4: Create body-integrate shader

**Files:**
- Create: `src/simulations/fluid/shaders/body-integrate.wgsl.ts`

- [ ] **Step 1: Create the body integration shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';
import { bodySdfHelpers } from './body-sdf.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}
${bodySdfHelpers}

@group(0) @binding(0) var<storage, read_write> bodyStates: array<BodyState>;
@group(0) @binding(1) var<storage, read_write> bodyForces: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> bodyTransforms: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: SimParams;
@group(0) @binding(4) var<uniform> bodyCount: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= bodyCount) { return; }

    var body = bodyStates[b];

    // Read accumulated forces (stored as u32 bits for atomic CAS)
    let forceX = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 0u]));
    let forceY = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 1u]));
    let torque = bitcast<f32>(atomicLoad(&bodyForces[b * 4u + 2u]));

    // Clear accumulators for next substep
    atomicStore(&bodyForces[b * 4u + 0u], 0u);
    atomicStore(&bodyForces[b * 4u + 1u], 0u);
    atomicStore(&bodyForces[b * 4u + 2u], 0u);

    // Total force = fluid reaction + gravity
    var totalForce = vec2<f32>(forceX, forceY);
    totalForce += vec2<f32>(params.gravity.x, params.gravity.y) * body.mass;

    // Linear integration
    body.vel = vec3<f32>(
        body.vel.x + (totalForce.x / body.mass) * params.dt,
        body.vel.y + (totalForce.y / body.mass) * params.dt,
        body.vel.z
    );
    body.pos = vec3<f32>(
        body.pos.x + body.vel.x * params.dt,
        body.pos.y + body.vel.y * params.dt,
        body.pos.z
    );

    // Angular integration
    let safeInertia = max(body.inertia, 0.001);
    body.angVel += (torque / safeInertia) * params.dt;
    body.angle += body.angVel * params.dt;

    // Damping
    body.vel = vec3<f32>(body.vel.x * 0.999, body.vel.y * 0.999, body.vel.z);
    body.angVel *= 0.998;

    // Boundary collision
    let bMin = params.worldBoundsMin;
    let bMax = params.worldBoundsMax;
    if (body.pos.x < bMin.x) { body.pos.x = bMin.x; body.vel.x *= -0.5; }
    if (body.pos.x > bMax.x) { body.pos.x = bMax.x; body.vel.x *= -0.5; }
    if (body.pos.y < bMin.y) { body.pos.y = bMin.y; body.vel.y *= -0.5; }
    if (body.pos.y > bMax.y) { body.pos.y = bMax.y; body.vel.y *= -0.5; }

    // Write back
    bodyStates[b] = body;

    // Write transform for rendering: vec4(posX, posY, posZ, angle)
    bodyTransforms[b] = vec4<f32>(body.pos.x, body.pos.y, body.pos.z, body.angle);
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/body-integrate.wgsl.ts
git commit -m "feat: add body rigid body integration shader with transform output"
```

---

## Task 5: Integrate bodies into FluidSimulation

**Files:**
- Modify: `src/simulations/fluid/FluidSimulation.ts`

This is the largest task. It adds body management, buffers, compute passes, and wires them into the pipeline.

- [ ] **Step 1: Add imports and constants**

At the top of `FluidSimulation.ts`, after the existing shader imports (line 24), add:

```typescript
import { shaderCode as bodyCollisionShader } from './shaders/body-collision.wgsl';
import { shaderCode as bodyIntegrateShader } from './shaders/body-integrate.wgsl';
import { FluidBody, FluidBodyOptions } from './FluidBody';
import { Vector3 } from '../../math/Vector3';
import { Float } from '../../math/Float';
```

After the existing constants (line 27), add:

```typescript
const MAX_BODIES = 64;
const MAX_PRIMITIVES = 256;
const BODY_STATE_FLOATS = 16;  // 64 bytes per body
const PRIMITIVE_FLOATS = 6;     // 24 bytes per primitive
```

- [ ] **Step 2: Add body-related fields to the class**

After the existing `private integratePass!: Compute;` (line 70), add:

```typescript
    // Body system
    private bodies: FluidBody[] = [];
    private bodyStatesF32!: Float32Array;
    private bodyStatesU32!: Uint32Array;
    private bodyStatesBuffer!: ComputeBuffer;
    private bodyForcesBuffer!: ComputeBuffer;
    private bodyPrimitivesF32!: Float32Array;
    private bodyPrimitivesU32!: Uint32Array;
    private bodyPrimitivesBuffer!: ComputeBuffer;
    public bodyTransformsBuffer!: ComputeBuffer;
    private bodyCountFloat!: Float;
    private totalPrimitives: number = 0;

    private bodyCollisionPass!: Compute;
    private bodyIntegratePass!: Compute;
```

- [ ] **Step 3: Add `createBodyBuffers()` method**

Add after `createBuffers()`:

```typescript
    private createBodyBuffers(): void {
        // Body states: 64 bodies × 16 floats
        this.bodyStatesF32 = new Float32Array(MAX_BODIES * BODY_STATE_FLOATS);
        this.bodyStatesU32 = new Uint32Array(this.bodyStatesF32.buffer);
        this.bodyStatesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyStatesF32,
        });

        // Body forces: 64 bodies × 4 u32 (forceX, forceY, torque, _pad)
        this.bodyForcesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(MAX_BODIES * 4),
        });

        // Body primitives: 256 × 6 floats
        this.bodyPrimitivesF32 = new Float32Array(MAX_PRIMITIVES * PRIMITIVE_FLOATS);
        this.bodyPrimitivesU32 = new Uint32Array(this.bodyPrimitivesF32.buffer);
        this.bodyPrimitivesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyPrimitivesF32,
        });

        // Body transforms: vec4(posX, posY, posZ, angle) per body, STORAGE + VERTEX
        this.bodyTransformsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_VERTEX | BufferBase.BUFFER_USAGE_COPY_SRC,
            buffer: new Float32Array(MAX_BODIES * 4),
            shaderLocation: 3,
            offset: 0,
            stride: 4 * 4,
            format: 'float32x4' as GPUVertexFormat,
        });

        // Body count uniform
        this.bodyCountFloat = new Float(0);
    }
```

- [ ] **Step 4: Add `createBodyComputePasses()` method**

Add after `createComputePasses()`:

```typescript
    private createBodyComputePasses(): void {
        const C = GPUShaderStage.COMPUTE;

        this.bodyCollisionPass = new Compute(bodyCollisionShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.bodyStatesBuffer },
            { binding: 3, visibility: C, value: this.bodyPrimitivesBuffer },
            { binding: 4, visibility: C, value: this.bodyForcesBuffer },
            { binding: 5, visibility: C, value: this.paramsBuffer },
            { binding: 6, visibility: C, value: this.bodyCountFloat },
        ]);

        this.bodyIntegratePass = new Compute(bodyIntegrateShader, [
            { binding: 0, visibility: C, value: this.bodyStatesBuffer },
            { binding: 1, visibility: C, value: this.bodyForcesBuffer },
            { binding: 2, visibility: C, value: this.bodyTransformsBuffer },
            { binding: 3, visibility: C, value: this.paramsBuffer },
            { binding: 4, visibility: C, value: this.bodyCountFloat },
        ]);
    }
```

- [ ] **Step 5: Call body init methods from `initialize()`**

In the `initialize()` method, after the line `this.createComputePasses();` add:

```typescript
        this.createBodyBuffers();
        this.createBodyComputePasses();
```

- [ ] **Step 6: Add `addBody()` and `removeBody()` methods**

Add after `setParams()`:

```typescript
    public get bodyCount(): number {
        return this.bodies.length;
    }

    public addBody(options: FluidBodyOptions): FluidBody {
        if (this.bodies.length >= MAX_BODIES) {
            throw new Error(`Max body count (${MAX_BODIES}) reached`);
        }
        if (this.totalPrimitives + options.primitives.length > MAX_PRIMITIVES) {
            throw new Error(`Max primitive count (${MAX_PRIMITIVES}) reached`);
        }

        const body = new FluidBody(options, this.bodies.length, this.totalPrimitives);
        this.bodies.push(body);

        // Pack primitives
        for (let i = 0; i < body.primitiveCount; i++) {
            const offset = (this.totalPrimitives + i) * PRIMITIVE_FLOATS;
            FluidBody.packPrimitive(body.primitives[i], this.bodyPrimitivesF32, this.bodyPrimitivesU32, offset);
        }
        this.totalPrimitives += body.primitiveCount;
        this.bodyPrimitivesBuffer.needsUpdate = true;

        // Pack body state
        this.syncBodyState(body);

        // Update count
        this.bodyCountFloat.value = this.bodies.length;

        return body;
    }

    public removeBody(body: FluidBody): void {
        const idx = this.bodies.indexOf(body);
        if (idx === -1) return;

        // Swap with last body
        const last = this.bodies[this.bodies.length - 1];
        if (idx !== this.bodies.length - 1) {
            // Copy last body's state into removed body's slot
            const srcOffset = last.index * BODY_STATE_FLOATS;
            const dstOffset = idx * BODY_STATE_FLOATS;
            this.bodyStatesF32.copyWithin(dstOffset, srcOffset, srcOffset + BODY_STATE_FLOATS);
            (last as any).index = idx;
        }

        this.bodies.splice(idx, 1);
        // Note: primitive compaction is not done for simplicity — removed body's primitives remain as dead entries
        this.bodyCountFloat.value = this.bodies.length;
        this.bodyStatesBuffer.needsUpdate = true;
    }

    private syncBodyState(body: FluidBody): void {
        const offset = body.index * BODY_STATE_FLOATS;
        body.packState(this.bodyStatesF32, this.bodyStatesU32, offset);
        this.bodyStatesBuffer.needsUpdate = true;
    }

    public syncBodyParams(): void {
        for (const body of this.bodies) {
            this.syncBodyState(body);
        }
    }
```

- [ ] **Step 7: Add body passes to the `update()` pipeline**

In the `update()` method, modify the `computeBatch` array to add body passes after the integrate pass. Replace the existing `computeBatch` call (lines 381-396) with:

```typescript
            const passes = [
                // Grid build
                { compute: this.gridClearCountsPass,    workgroupsX: gridWorkgroups },
                { compute: this.gridClearScatterPass,   workgroupsX: gridWorkgroups },
                { compute: this.gridAssignPass,         workgroupsX: particleWorkgroups },
                // Prefix sum
                { compute: this.prefixSumLocalPass,      workgroupsX: Math.max(prefixSumWorkgroups, 1) },
                { compute: this.prefixSumTopPass,        workgroupsX: 1 },
                { compute: this.prefixSumDistributePass, workgroupsX: Math.max(prefixSumWorkgroups, 1) },
                // Scatter
                { compute: this.scatterPass,             workgroupsX: particleWorkgroups },
                // SPH
                { compute: this.densityPass,             workgroupsX: particleWorkgroups },
                { compute: this.forcesPass,              workgroupsX: particleWorkgroups },
                { compute: this.integratePass,           workgroupsX: particleWorkgroups },
            ];

            // Body passes (only if bodies exist)
            if (this.bodies.length > 0) {
                passes.push(
                    { compute: this.bodyCollisionPass,   workgroupsX: particleWorkgroups },
                    { compute: this.bodyIntegratePass,    workgroupsX: 1 },
                );
            }

            await this.renderer.computeBatch(passes);
```

- [ ] **Step 8: Commit**

```bash
git add src/simulations/fluid/FluidSimulation.ts
git commit -m "feat: integrate fluid bodies into FluidSimulation with body management and pipeline"
```

---

## Task 6: Export FluidBody from main.ts

**Files:**
- Modify: `src/main.ts`

- [ ] **Step 1: Add export**

At the end of `src/main.ts`, add:

```typescript
export { FluidBody } from "./simulations/fluid/FluidBody";
export type { FluidBodyOptions, FluidBodyPrimitive } from "./simulations/fluid/FluidBody";
```

- [ ] **Step 2: Commit**

```bash
git add src/main.ts
git commit -m "feat: export FluidBody from main entry point"
```

---

## Task 7: Add floating body to the example

**Files:**
- Modify: `examples/index_fluid_text.html`

- [ ] **Step 1: Add body creation and rendering**

Read the current file first. After the `sim.initialize(...)` block (around line 115), add:

```javascript
      // ── Floating Body ─────────────────────────────────────
      import { PlaneGeometry } from '../src/geometries/PlaneGeometry'
      import { InstancedGeometry } from '../src/geometries/InstancedGeometry'
```

Wait — imports must be at the top of the module. Move the imports to the top import block, then add the body code after `sim.initialize(...)`.

Add these imports at the top with the other imports:

```javascript
      import { PlaneGeometry } from '../src/geometries/PlaneGeometry'
      import { InstancedGeometry } from '../src/geometries/InstancedGeometry'
```

Then after `sim.initialize(...)`:

```javascript
      // ── Floating Body ─────────────────────────────────────
      const body = sim.addBody({
        primitives: [
          { type: 'circle', radius: 3.0, localX: 0, localY: 0 },
        ],
        position: new Vector3(20, 20, 0),
        mass: 8.0,
        restitution: 0.3,
      });

      // Body renderable — instanced quad driven by bodyTransformsBuffer
      const bodyQuadGeo = new PlaneGeometry(6, 6);
      const bodyInstancedGeo = new InstancedGeometry(bodyQuadGeo, 1, [sim.bodyTransformsBuffer]);
      const bodyMaterial = new Material(/* wgsl */`
        struct VertexInput {
          @location(0) position: vec3<f32>,
          @location(1) normal: vec3<f32>,
          @location(2) uv: vec2<f32>,
          @location(3) bodyTransform: vec4<f32>,
        };

        struct VertexOutput {
          @builtin(position) position: vec4<f32>,
          @location(0) vUv: vec2<f32>,
        };

        @group(0) @binding(0) var<uniform> viewProjection: mat4x4<f32>;
        @group(0) @binding(1) var<uniform> worldMatrix: mat4x4<f32>;

        @vertex
        fn vertexMain(input: VertexInput) -> VertexOutput {
          var output: VertexOutput;
          let cosA = cos(input.bodyTransform.w);
          let sinA = sin(input.bodyTransform.w);
          let rotated = vec2<f32>(
            input.position.x * cosA - input.position.y * sinA,
            input.position.x * sinA + input.position.y * cosA
          );
          let worldPos = vec3<f32>(rotated + input.bodyTransform.xy, input.bodyTransform.z);
          output.position = viewProjection * worldMatrix * vec4<f32>(worldPos, 1.0);
          output.vUv = input.uv;
          return output;
        }

        @fragment
        fn fragmentMain(input: VertexOutput) -> @location(0) vec4<f32> {
          let dist = length(input.vUv - vec2<f32>(0.5));
          let alpha = 1.0 - smoothstep(0.35, 0.5, dist);
          return vec4<f32>(1.0, 0.8, 0.2, alpha);
        }
      `, { transparent: true });
      const bodyMesh = new Renderable(bodyInstancedGeo, bodyMaterial);
      scene.add(bodyMesh);
```

Note: The vertex shader approach depends on how `Material` handles the vertex/fragment split and how `Renderable` injects camera/world uniforms. Read the existing `Material` class and `TextRenderShader` to verify the exact shader structure before implementing. The shader above is a template — adapt to match the engine's actual vertex input layout and uniform binding conventions.

- [ ] **Step 2: Add body tweakpane controls**

In the tweakpane section, add a new folder after the Boundaries folder:

```javascript
      // Body folder
      const bodyFolder = pane.addFolder({ title: 'Floating Body', expanded: false });
      const bodyParams = { mass: 8.0, restitution: 0.3, radius: 3.0 };
      bodyFolder.addBinding(bodyParams, 'mass', { min: 0.5, max: 50, step: 0.5 }).on('change', () => {
        body.mass = bodyParams.mass;
        sim.syncBodyParams();
      });
      bodyFolder.addBinding(bodyParams, 'restitution', { min: 0.0, max: 1.0, step: 0.05 }).on('change', () => {
        body.restitution = bodyParams.restitution;
        sim.syncBodyParams();
      });
```

- [ ] **Step 3: Commit**

```bash
git add examples/index_fluid_text.html
git commit -m "feat: add floating body to fluid text example with tweakpane controls"
```

---

## Task 8: Visual verification

- [ ] **Step 1: Start dev server and test**

Run: `npx vite`
Open: `http://localhost:5173/examples/index_fluid_text.html`

Expected:
- Text particles render and behave as before (SPH fluid)
- A circular body (golden disc) appears at position (20, 20)
- The body falls under gravity
- When the body hits the fluid surface, it should slow down and float (buoyancy from particle collisions)
- The body should bob and rotate slightly from asymmetric fluid forces
- Moving mouse near the body should push fluid which pushes the body

- [ ] **Step 2: Check browser console for errors**

Common issues:
- WGSL compilation errors: check shader syntax, struct alignment, binding declarations
- Bind group layout mismatch: verify binding order matches between TypeScript and WGSL
- `atomicCompareExchangeWeak` requires `array<atomic<u32>>` — make sure `bodyForces` buffer type is `'storage'`
- `Float` uniform for bodyCount: verify `Float` implements `IBindable` with `type: 'uniform'`

- [ ] **Step 3: Fix any issues and commit**

```bash
git add -A
git commit -m "fix: address issues from visual verification"
```

---

## Dependency Graph

```
Task 1 (FluidBody class) ──────────┐
Task 2 (SDF shader helpers) ────────┤
Task 3 (body-collision shader) ─────┼──→ Task 5 (FluidSimulation integration) ──→ Task 6 (exports) ──→ Task 7 (example) ──→ Task 8 (verify)
Task 4 (body-integrate shader) ─────┘
```

Tasks 1–4 are independent. Task 5 depends on all of them. Tasks 6–8 are sequential.
