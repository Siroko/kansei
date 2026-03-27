# Fluid Bodies — Design Spec

## Overview

SDF-based rigid bodies that float on the SPH fluid simulation with two-way coupling. Fluid particles collide with body surfaces (defined by compound signed distance fields), get deflected, and exert reaction forces that create buoyancy, drag, and torque on the bodies. Bodies are rendered via instanced geometry with GPU-driven transforms — zero CPU readback.

Builds on the existing `FluidSimulation` package at `src/simulations/fluid/`.

## API

```typescript
import { FluidBody } from '../src/simulations/fluid/FluidBody';
import { Vector3 } from '../src/math/Vector3';

const body = sim.addBody({
    primitives: [
        { type: 'circle', radius: 1.0, localX: 0, localY: 0 },
        { type: 'box', halfW: 2.0, halfH: 0.3, localX: 0, localY: -1.2 },
    ],
    position: new Vector3(10, 5, 0),
    velocity: new Vector3(0, 0, 0),
    mass: 5.0,
    restitution: 0.3,
});

// The transforms buffer for rendering (STORAGE | VERTEX)
// Contains vec4(posX, posY, posZ, angle) per body
const bodyTransformsBuffer = sim.bodyTransformsBuffer;

// Use with InstancedGeometry for rendering
const bodyGeometry = new InstancedGeometry(quadGeometry, sim.bodyCount, [bodyTransformsBuffer]);

// Runtime
body.mass = 10.0;
body.restitution = 0.5;
sim.removeBody(body);
```

### FluidBody class

```typescript
class FluidBody {
    public position: Vector3;        // initial position (written to GPU on addBody)
    public velocity: Vector3;        // initial velocity
    public angle: number;            // initial Z rotation (radians)
    public angularVelocity: number;  // initial spin
    public mass: number;
    public restitution: number;      // bounce factor for fluid collision (0-1)

    readonly index: number;          // slot in GPU buffers
    readonly primitiveStart: number; // offset into primitives buffer
    readonly primitiveCount: number; // number of SDF primitives
}
```

CPU handle for setup and configuration. Live position/angle state lives on GPU only — rendering reads directly from the GPU transforms buffer via instanced vertex attributes.

### FluidSimulation additions

```typescript
// New methods
addBody(options: FluidBodyOptions): FluidBody;
removeBody(body: FluidBody): void;
syncBodyParams(): void;  // upload body mass/restitution changes to GPU

// New properties
bodyTransformsBuffer: ComputeBuffer;  // vec4(posX, posY, posZ, angle) per body, STORAGE | VERTEX
bodyCount: number;                     // current number of active bodies
```

## SDF Primitives (2D)

| Type | ID | Params | SDF formula |
|------|----|--------|-------------|
| circle | 0 | radius | `length(p) - radius` |
| box | 1 | halfW, halfH | `length(max(abs(p) - half, 0)) + min(max(abs(p).x - halfW, abs(p).y - halfH), 0)` |
| capsule | 2 | radius, halfLength | `length(p - vec2(clamp(p.x, -halfLen, halfLen), 0)) - radius` |

Compound shape = `min(sdf_0, sdf_1, ..., sdf_n)`.

Each primitive is queried in **body-local space**: the query point is rotated by `-bodyAngle` and translated by `-bodyPosition` before evaluating. Each primitive's `localX/localY` provides an additional offset within body space.

SDF gradient computed via central finite differences (4 extra SDF evaluations per colliding particle). The gradient gives the push direction and surface normal.

## GPU Buffer Layout

| Buffer | Type | Size | Usage |
|--------|------|------|-------|
| `bodyStates` | storage, read_write | 64 × 64 bytes | Per-body: pos(vec3), vel(vec3), angle, angVel, mass, inertia, restitution, primStart(u32), primCount(u32), _pad |
| `bodyForces` | storage, read_write | 64 × 16 bytes | Per-body: forceX, forceY, torque, _pad (u32, atomics for float CAS) |
| `bodyPrimitives` | storage, read | 256 × 24 bytes | Per-primitive: type(u32), param1, param2, localX, localY, _pad |
| `bodyTransforms` | storage + vertex | 64 × 16 bytes | Per-body: posX, posY, posZ, angle (written by body-integrate, read by vertex shader) |
| `bodyCount` | uniform | 4 bytes | u32 active body count |

### bodyStates struct (WGSL)

```wgsl
struct BodyState {
    pos: vec3<f32>,        // 0  (align 16)
    _pad0: f32,            // 12
    vel: vec3<f32>,        // 16 (align 16)
    _pad1: f32,            // 28
    angle: f32,            // 32
    angVel: f32,           // 36
    mass: f32,             // 40
    inertia: f32,          // 44
    restitution: f32,      // 48
    primStart: u32,        // 52
    primCount: u32,        // 56
    _pad2: f32,            // 60
};
// 64 bytes per body
```

### bodyPrimitives struct (WGSL)

```wgsl
struct BodyPrimitive {
    primType: u32,   // 0=circle, 1=box, 2=capsule
    param1: f32,     // radius or halfW
    param2: f32,     // halfH or halfLength (0 for circle)
    localX: f32,
    localY: f32,
    _pad: f32,
};
// 24 bytes per primitive
```

## Compute Pipeline

Two new passes added after the existing `integratePass` (pass 10), within the same `computeBatch` call per substep:

### Pass 11: Fluid-Body Collision (N/64 workgroups)

Per fluid particle:
1. Loop all active bodies (bodyCount is small, 1-64)
2. Evaluate compound SDF at particle position
3. If `sdf < 0` (particle inside body):
   - Compute SDF gradient via central differences (4 extra evals)
   - Push particle out: `pos += gradient * (-sdf)`
   - Reflect velocity: `vel = reflect(vel, gradient) * restitution`
   - Accumulate reaction force on body via atomic float CAS: `force -= gradient * (-sdf) * particleMassScale` (particleMassScale = tunable, controls coupling strength)
   - Accumulate torque: `torque += cross2D(r, force)` where `r = particlePos - bodyPos`

### Pass 12: Body Integrate (1 workgroup, max 64 threads)

Per body (1 thread per body):
1. Read accumulated force and torque from `bodyForces`
2. Add body gravity: `force += vec2(0, -gravity * mass)`
3. Integrate velocity: `vel += (force / mass) * dt`
4. Integrate position: `pos += vel * dt`
5. Integrate angular: `angVel += (torque / inertia) * dt`, `angle += angVel * dt`
6. Apply velocity damping
7. Boundary collision (clamp to worldBounds, reflect velocity)
8. Write updated state to `bodyStates`
9. Write transform to `bodyTransforms`: `vec4(pos.x, pos.y, pos.z, angle)`
10. Clear `bodyForces` accumulators for next substep

### Atomic Float CAS pattern

WGSL only supports atomic ops on `u32`/`i32`. Float atomicAdd is emulated:

```wgsl
fn atomicAddF32(addr: ptr<storage, atomic<u32>, read_write>, value: f32) {
    var old = atomicLoad(addr);
    loop {
        let newVal = bitcast<f32>(old) + value;
        let result = atomicCompareExchangeWeak(addr, old, bitcast<u32>(newVal));
        if (result.exchanged) { break; }
        old = result.old_value;
    }
}
```

Low contention (few bodies, moderate collision count) means typically 1-2 CAS iterations.

## Rendering Integration

The `bodyTransformsBuffer` is a `ComputeBuffer` with `STORAGE | VERTEX` usage. It contains `vec4(posX, posY, posZ, angle)` per body.

The user creates an `InstancedGeometry` with this buffer as an extra buffer (same pattern as `TextGeometry.extraBuffers`). The vertex shader reads the per-instance transform and applies rotation + translation:

```wgsl
// In the body's vertex shader
let bodyTransform = instanceTransform; // vec4 from extra buffer
let cosA = cos(bodyTransform.w);
let sinA = sin(bodyTransform.w);
let rotated = vec2(
    vertexPos.x * cosA - vertexPos.y * sinA,
    vertexPos.x * sinA + vertexPos.y * cosA
);
let worldPos = vec3(rotated + bodyTransform.xy, bodyTransform.z);
```

Zero CPU readback. GPU writes transforms, GPU reads them for rendering.

## File Structure

```
src/simulations/fluid/
  FluidBody.ts                        # FluidBody class + FluidBodyOptions interface
  FluidSimulation.ts                  # Modified: add body management + 2 new passes
  shaders/
    body-collision.wgsl.ts            # Fluid-body SDF collision + force accumulation
    body-integrate.wgsl.ts            # Body rigid body integration + transform output
    body-sdf.wgsl.ts                  # Shared SDF evaluation + gradient functions
```

## Limits

- Max 64 bodies (single workgroup in body-integrate pass)
- Max 256 total primitives across all bodies
- 2D physics only (Z-axis rotation, XY forces)
- 3D SDFs + quaternion rotation deferred to future enhancement

## Future Enhancements

- **3D rotation**: Replace angle/angVel with quaternion, upgrade SDFs to 3D (sphere, box, capsule, cylinder)
- **Body-body collision**: SDF overlap detection between bodies, contact response
- **Joints/constraints**: Distance constraints between bodies for articulated structures
- **Per-body textures**: Different textures per instance via texture array + instance index
