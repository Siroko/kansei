# Fluid Simulation Package — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a GPU-accelerated SPH fluid simulation package (`src/simulations/fluid/`) with spatial hash grid acceleration, consumable by any Kansei instanced geometry.

**Architecture:** 9 compute passes per substep (grid build, prefix sum, scatter, SPH density/forces, integration) batched into single GPU submits via a new `Renderer.computeBatch()` method. Simulation mutates positions buffer in-place (zero-copy vertex output). Configurable 2D/3D modes.

**Tech Stack:** WebGPU compute shaders (WGSL), TypeScript, Tweakpane v4

**Spec:** `docs/superpowers/specs/2026-03-27-fluid-simulation-design.md`

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `src/renderers/Renderer.ts` | Add `computeBatch()` method |
| Create | `src/simulations/fluid/FluidSimulationParams.ts` | Parameter interface, defaults, presets |
| Create | `src/simulations/fluid/shaders/sim-params.wgsl.ts` | Shared `SimParams` WGSL struct |
| Create | `src/simulations/fluid/shaders/grid-clear.wgsl.ts` | Zero cell counts |
| Create | `src/simulations/fluid/shaders/grid-assign.wgsl.ts` | Hash particles into cells |
| Create | `src/simulations/fluid/shaders/prefix-sum-local.wgsl.ts` | Per-block Blelloch scan |
| Create | `src/simulations/fluid/shaders/prefix-sum-top.wgsl.ts` | Scan block sums |
| Create | `src/simulations/fluid/shaders/prefix-sum-distribute.wgsl.ts` | Distribute block offsets |
| Create | `src/simulations/fluid/shaders/scatter.wgsl.ts` | Reorder particles by cell |
| Create | `src/simulations/fluid/shaders/density.wgsl.ts` | SPH density computation |
| Create | `src/simulations/fluid/shaders/forces.wgsl.ts` | Pressure + viscosity + external forces |
| Create | `src/simulations/fluid/shaders/integrate.wgsl.ts` | Position integration + boundaries |
| Create | `src/simulations/fluid/FluidSimulation.ts` | Main orchestrator class |
| Modify | `src/main.ts` | Export FluidSimulation |
| Create | `examples/index_fluid_text.html` | Text fluid demo with tweakpane |

---

## Task 1: Add `computeBatch()` to Renderer

**Files:**
- Modify: `src/renderers/Renderer.ts` (after the `compute()` method at ~line 952)

- [ ] **Step 1: Add the computeBatch method**

Add this method right after the existing `compute()` method in `src/renderers/Renderer.ts`:

```typescript
/**
 * Executes multiple compute shaders in a single command buffer submission.
 * Storage buffer writes in pass N are visible to pass N+1 (WebGPU guarantee).
 * @param passes Array of compute passes with workgroup dimensions
 * @returns Promise that resolves when all passes complete
 */
public async computeBatch(passes: { compute: Compute, workgroupsX: number, workgroupsY?: number, workgroupsZ?: number }[]): Promise<void> {
    const commandEncoder = this.device!.createCommandEncoder();
    for (const pass of passes) {
        if (!pass.compute.initialized) {
            pass.compute.initialize(this.device!);
        }
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setBindGroup(0, pass.compute.getBindGroup(this.device!));
        passEncoder.setPipeline(pass.compute.pipeline!);
        passEncoder.dispatchWorkgroups(pass.workgroupsX, pass.workgroupsY ?? 1, pass.workgroupsZ ?? 1);
        passEncoder.end();
    }
    this.device!.queue.submit([commandEncoder.finish()]);
    return this.device!.queue.onSubmittedWorkDone();
}
```

- [ ] **Step 2: Verify it compiles**

Run: `npx tsc --noEmit src/renderers/Renderer.ts`
Expected: no errors (the method uses the same types as `compute()`)

- [ ] **Step 3: Commit**

```bash
git add src/renderers/Renderer.ts
git commit -m "feat: add computeBatch() to Renderer for multi-pass compute dispatch"
```

---

## Task 2: Create FluidSimulationParams

**Files:**
- Create: `src/simulations/fluid/FluidSimulationParams.ts`

- [ ] **Step 1: Create the params interface and defaults**

```typescript
export interface FluidSimulationOptions {
    maxParticles: number;
    dimensions: 2 | 3;
    smoothingRadius: number;
    pressureMultiplier: number;
    nearPressureMultiplier: number;
    densityTarget: number;
    viscosity: number;
    damping: number;
    gravity: [number, number, number];
    returnToOriginStrength: number;
    mouseRadius: number;
    mouseForce: number;
    substeps: number;
    worldBoundsPadding: number;
}

export const DEFAULT_OPTIONS: FluidSimulationOptions = {
    maxParticles: 10000,
    dimensions: 2,
    smoothingRadius: 1.0,
    pressureMultiplier: 10.0,
    nearPressureMultiplier: 18.0,
    densityTarget: 1.5,
    viscosity: 0.3,
    damping: 0.998,
    gravity: [0, 0, 0],
    returnToOriginStrength: 0.002,
    mouseRadius: 0.5,
    mouseForce: 300.0,
    substeps: 3,
    worldBoundsPadding: 0.2,
};

export interface FluidSimulationPreset extends Partial<FluidSimulationOptions> {
    name: string;
}

export const PRESETS: Record<string, FluidSimulationPreset> = {
    water: {
        name: 'Water',
        pressureMultiplier: 10,
        nearPressureMultiplier: 18,
        viscosity: 0.3,
        damping: 0.998,
        densityTarget: 1.5,
        returnToOriginStrength: 0.002,
        gravity: [0, -9.8, 0],
    },
    honey: {
        name: 'Viscous Honey',
        pressureMultiplier: 3,
        nearPressureMultiplier: 8,
        viscosity: 0.9,
        damping: 0.99,
        densityTarget: 3.0,
        returnToOriginStrength: 0.005,
        gravity: [0, -2.0, 0],
    },
    gas: {
        name: 'Gas',
        pressureMultiplier: 20,
        nearPressureMultiplier: 30,
        viscosity: 0.05,
        damping: 0.995,
        densityTarget: 0.5,
        returnToOriginStrength: 0.001,
        gravity: [0, 0, 0],
    },
    zeroG: {
        name: 'Zero-G Blob',
        pressureMultiplier: 8,
        nearPressureMultiplier: 14,
        viscosity: 0.6,
        damping: 0.997,
        densityTarget: 2.0,
        returnToOriginStrength: 0.0,
        gravity: [0, 0, 0],
    },
};

// SimParams uniform buffer layout (160 bytes = 40 f32s)
// Fields marked [u32] must be written via Uint32Array view
export const PARAMS = {
    dt:                       0,  // f32
    particleCount:            1,  // [u32]
    dimensions:               2,  // [u32]
    smoothingRadius:          3,  // f32
    pressureMultiplier:       4,  // f32
    densityTarget:            5,  // f32
    nearPressureMultiplier:   6,  // f32
    viscosity:                7,  // f32
    damping:                  8,  // f32
    returnToOriginStrength:   9,  // f32
    mouseStrength:           10,  // f32
    mouseRadius:             11,  // f32
    // --- 16-byte aligned boundary (offset 48) ---
    gravityX:                12,  // vec3<f32> gravity
    gravityY:                13,
    gravityZ:                14,
    mouseForce:              15,  // f32 (packed after vec3)
    // --- 8-byte aligned boundary (offset 64) ---
    mousePosX:               16,  // vec2<f32> mousePos
    mousePosY:               17,
    mouseDirX:               18,  // vec2<f32> mouseDir
    mouseDirY:               19,
    // --- 16-byte aligned boundary (offset 80) ---
    gridDimsX:               20,  // [u32] vec3<u32> gridDims
    gridDimsY:               21,  // [u32]
    gridDimsZ:               22,  // [u32]
    cellSize:                23,  // f32
    // --- 16-byte aligned boundary (offset 96) ---
    gridOriginX:             24,  // vec3<f32> gridOrigin
    gridOriginY:             25,
    gridOriginZ:             26,
    totalCells:              27,  // [u32]
    // --- 16-byte aligned boundary (offset 112) ---
    worldBoundsMinX:         28,  // vec3<f32> worldBoundsMin
    worldBoundsMinY:         29,
    worldBoundsMinZ:         30,
    poly6Factor:             31,  // f32
    // --- 16-byte aligned boundary (offset 128) ---
    worldBoundsMaxX:         32,  // vec3<f32> worldBoundsMax
    worldBoundsMaxY:         33,
    worldBoundsMaxZ:         34,
    spikyPow2Factor:         35,  // f32
    // --- remaining kernel factors ---
    spikyPow3Factor:         36,  // f32
    spikyPow2DerivFactor:    37,  // f32
    spikyPow3DerivFactor:    38,  // f32
    _pad:                    39,  // f32 padding (struct size must be 16-byte multiple)
    BUFFER_SIZE:             40,  // total f32 count
} as const;

export function computeKernelFactors2D(h: number) {
    const pi = Math.PI;
    return {
        poly6:           4.0 / (pi * Math.pow(h, 8)),
        spikyPow2:       6.0 / (pi * Math.pow(h, 4)),
        spikyPow3:       10.0 / (pi * Math.pow(h, 5)),
        spikyPow2Deriv:  12.0 / (Math.pow(h, 4) * pi),
        spikyPow3Deriv:  30.0 / (Math.pow(h, 5) * pi),
    };
}

export function computeKernelFactors3D(h: number) {
    const pi = Math.PI;
    return {
        poly6:           315.0 / (64.0 * pi * Math.pow(h, 9)),
        spikyPow2:       15.0 / (pi * Math.pow(h, 6)),
        spikyPow3:       15.0 / (pi * Math.pow(h, 6)),
        spikyPow2Deriv:  45.0 / (pi * Math.pow(h, 6)),
        spikyPow3Deriv:  45.0 / (pi * Math.pow(h, 6)),
    };
}
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/FluidSimulationParams.ts
git commit -m "feat: add fluid simulation params interface, defaults, and presets"
```

---

## Task 3: Create shared SimParams WGSL struct

**Files:**
- Create: `src/simulations/fluid/shaders/sim-params.wgsl.ts`

- [ ] **Step 1: Create the shared struct definition**

This is imported by all shaders that need the params uniform. Must match the byte layout defined in `PARAMS` from Task 2.

```typescript
export const simParamsStruct = /* wgsl */`
struct SimParams {
    dt: f32,
    particleCount: u32,
    dimensions: u32,
    smoothingRadius: f32,

    pressureMultiplier: f32,
    densityTarget: f32,
    nearPressureMultiplier: f32,
    viscosity: f32,

    damping: f32,
    returnToOriginStrength: f32,
    mouseStrength: f32,
    mouseRadius: f32,

    gravity: vec3<f32>,
    mouseForce: f32,

    mousePos: vec2<f32>,
    mouseDir: vec2<f32>,

    gridDims: vec3<u32>,
    cellSize: f32,

    gridOrigin: vec3<f32>,
    totalCells: u32,

    worldBoundsMin: vec3<f32>,
    poly6Factor: f32,

    worldBoundsMax: vec3<f32>,
    spikyPow2Factor: f32,

    spikyPow3Factor: f32,
    spikyPow2DerivFactor: f32,
    spikyPow3DerivFactor: f32,
    _pad: f32,
};

fn getCellCoord(pos: vec3<f32>, params: SimParams) -> vec3<i32> {
    return vec3<i32>(floor((pos - params.gridOrigin) / params.cellSize));
}

fn cellHash(coord: vec3<i32>, params: SimParams) -> u32 {
    let c = clamp(coord, vec3<i32>(0), vec3<i32>(params.gridDims) - vec3<i32>(1));
    return u32(c.z) * params.gridDims.x * params.gridDims.y + u32(c.y) * params.gridDims.x + u32(c.x);
}

`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/sim-params.wgsl.ts
git commit -m "feat: add shared SimParams WGSL struct definition"
```

---

## Task 4: Create grid shaders (clear, assign, scatter)

**Files:**
- Create: `src/simulations/fluid/shaders/grid-clear.wgsl.ts`
- Create: `src/simulations/fluid/shaders/grid-assign.wgsl.ts`
- Create: `src/simulations/fluid/shaders/scatter.wgsl.ts`

- [ ] **Step 1: Create grid-clear shader**

```typescript
export const shaderCode = /* wgsl */`

@group(0) @binding(0) var<storage, read_write> cellCounts: array<u32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&cellCounts)) { return; }
    cellCounts[idx] = 0u;
}
`;
```

- [ ] **Step 2: Create grid-assign shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(2) var<storage, read_write> cellCounts: array<atomic<u32>>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    let coord = getCellCoord(pos, params);
    let cell = cellHash(coord, params);

    cellIndices[idx] = cell;
    atomicAdd(&cellCounts[cell], 1u);
}
`;
```

- [ ] **Step 3: Create scatter shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> cellIndices: array<u32>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> scatterCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(4) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let cell = cellIndices[idx];
    let offset = cellOffsets[cell];
    let slot = atomicAdd(&scatterCounters[cell], 1u);
    sortedIndices[offset + slot] = idx;
}
`;
```

- [ ] **Step 4: Commit**

```bash
git add src/simulations/fluid/shaders/grid-clear.wgsl.ts \
        src/simulations/fluid/shaders/grid-assign.wgsl.ts \
        src/simulations/fluid/shaders/scatter.wgsl.ts
git commit -m "feat: add spatial hash grid shaders (clear, assign, scatter)"
```

---

## Task 5: Create prefix sum shaders

**Files:**
- Create: `src/simulations/fluid/shaders/prefix-sum-local.wgsl.ts`
- Create: `src/simulations/fluid/shaders/prefix-sum-top.wgsl.ts`
- Create: `src/simulations/fluid/shaders/prefix-sum-distribute.wgsl.ts`

Two-level work-efficient Blelloch scan. Each workgroup processes 512 elements using 256 threads with shared memory.

- [ ] **Step 1: Create prefix-sum-local shader**

```typescript
export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

var<workgroup> shared: array<u32, 512>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let tid = lid.x;
    let blockOffset = wid.x * BLOCK_SIZE;
    let inputLen = arrayLength(&input);

    // Load two elements per thread into shared memory
    let ai = tid;
    let bi = tid + 256u;
    let globalAi = blockOffset + ai;
    let globalBi = blockOffset + bi;
    shared[ai] = select(0u, input[globalAi], globalAi < inputLen);
    shared[bi] = select(0u, input[globalBi], globalBi < inputLen);

    // Up-sweep (reduce) phase
    var offset = 1u;
    for (var d = BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai2 = offset * (2u * tid + 1u) - 1u;
            let bi2 = offset * (2u * tid + 2u) - 1u;
            shared[bi2] += shared[ai2];
        }
        offset <<= 1u;
    }

    // Save block total and clear last element
    if (tid == 0u) {
        blockSums[wid.x] = shared[BLOCK_SIZE - 1u];
        shared[BLOCK_SIZE - 1u] = 0u;
    }

    // Down-sweep phase
    for (var d2 = 1u; d2 < BLOCK_SIZE; d2 <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d2) {
            let ai3 = offset * (2u * tid + 1u) - 1u;
            let bi3 = offset * (2u * tid + 2u) - 1u;
            let temp = shared[ai3];
            shared[ai3] = shared[bi3];
            shared[bi3] += temp;
        }
    }

    workgroupBarrier();

    // Write results
    if (globalAi < inputLen) { output[globalAi] = shared[ai]; }
    if (globalBi < inputLen) { output[globalBi] = shared[bi]; }
}
`;
```

- [ ] **Step 2: Create prefix-sum-top shader**

Single workgroup scans the block sums (max 512 blocks = 256K cells).

```typescript
export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> blockSums: array<u32>;

var<workgroup> shared: array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let len = arrayLength(&blockSums);

    let ai = tid;
    let bi = tid + 256u;
    shared[ai] = select(0u, blockSums[ai], ai < len);
    shared[bi] = select(0u, blockSums[bi], bi < len);

    var offset = 1u;
    for (var d = BLOCK_SIZE >> 1u; d > 0u; d >>= 1u) {
        workgroupBarrier();
        if (tid < d) {
            let ai2 = offset * (2u * tid + 1u) - 1u;
            let bi2 = offset * (2u * tid + 2u) - 1u;
            shared[bi2] += shared[ai2];
        }
        offset <<= 1u;
    }

    if (tid == 0u) {
        shared[BLOCK_SIZE - 1u] = 0u;
    }

    for (var d2 = 1u; d2 < BLOCK_SIZE; d2 <<= 1u) {
        offset >>= 1u;
        workgroupBarrier();
        if (tid < d2) {
            let ai3 = offset * (2u * tid + 1u) - 1u;
            let bi3 = offset * (2u * tid + 2u) - 1u;
            let temp = shared[ai3];
            shared[ai3] = shared[bi3];
            shared[bi3] += temp;
        }
    }

    workgroupBarrier();

    if (ai < len) { blockSums[ai] = shared[ai]; }
    if (bi < len) { blockSums[bi] = shared[bi]; }
}
`;
```

- [ ] **Step 3: Create prefix-sum-distribute shader**

```typescript
export const shaderCode = /* wgsl */`

const BLOCK_SIZE: u32 = 512u;

@group(0) @binding(0) var<storage, read_write> blockSums: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;

@compute @workgroup_size(256)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>
) {
    let blockOffset = wid.x * BLOCK_SIZE;
    let blockSum = blockSums[wid.x];
    let outputLen = arrayLength(&output);

    let globalAi = blockOffset + lid.x;
    let globalBi = blockOffset + lid.x + 256u;

    if (globalAi < outputLen) { output[globalAi] += blockSum; }
    if (globalBi < outputLen) { output[globalBi] += blockSum; }
}
`;
```

- [ ] **Step 4: Commit**

```bash
git add src/simulations/fluid/shaders/prefix-sum-local.wgsl.ts \
        src/simulations/fluid/shaders/prefix-sum-top.wgsl.ts \
        src/simulations/fluid/shaders/prefix-sum-distribute.wgsl.ts
git commit -m "feat: add two-level Blelloch prefix sum shaders"
```

---

## Task 6: Create SPH shaders (density, forces, integrate)

**Files:**
- Create: `src/simulations/fluid/shaders/density.wgsl.ts`
- Create: `src/simulations/fluid/shaders/forces.wgsl.ts`
- Create: `src/simulations/fluid/shaders/integrate.wgsl.ts`

- [ ] **Step 1: Create density shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(4) var<uniform> params: SimParams;

fn densityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * v * params.spikyPow2Factor;
}

fn nearDensityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return v * v * v * params.spikyPow3Factor;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;

    var density = 0.0;
    var nearDensity = 0.0;

    let searchRange = select(1, 0, params.dimensions == 2u);
    let zStart = select(-1, 0, params.dimensions == 2u);
    let zEnd = select(1, 0, params.dimensions == 2u);

    for (var dz = zStart; dz <= zEnd; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = coord + vec3<i32>(dx, dy, dz);

                // Bounds check
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= vec3<i32>(params.gridDims))) {
                    continue;
                }

                let neighborCell = cellHash(neighborCoord, params);
                let cellStart = cellOffsets[neighborCell];
                let cellEnd = select(cellOffsets[neighborCell + 1u], params.particleCount, neighborCell + 1u >= params.totalCells);

                for (var j = cellStart; j < cellEnd; j++) {
                    let neighborIdx = sortedIndices[j];
                    if (neighborIdx == idx) { continue; }

                    let neighborPos = positions[neighborIdx].xyz;
                    let diff = pos - neighborPos;
                    let dist = length(diff);

                    density += densityKernel(dist, h);
                    nearDensity += nearDensityKernel(dist, h);
                }
            }
        }
    }

    densities[idx] = vec2<f32>(density, nearDensity);
}
`;
```

- [ ] **Step 2: Create forces shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> densities: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read_write> originalPositions: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> cellOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> sortedIndices: array<u32>;
@group(0) @binding(6) var<uniform> params: SimParams;
@group(0) @binding(7) var<uniform> viewMatrix: mat4x4<f32>;
@group(0) @binding(8) var<uniform> projectionMatrix: mat4x4<f32>;
@group(0) @binding(9) var<uniform> inverseViewMatrix: mat4x4<f32>;
@group(0) @binding(10) var<uniform> worldMatrix: mat4x4<f32>;

fn pressureFromDensity(density: f32) -> f32 {
    return (density - params.densityTarget) * params.pressureMultiplier;
}

fn nearPressureFromDensity(nearDensity: f32) -> f32 {
    return nearDensity * params.nearPressureMultiplier;
}

fn densityDerivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return -v * params.spikyPow2DerivFactor;
}

fn nearDensityDerivative(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h - dist;
    return -v * v * params.spikyPow3DerivFactor;
}

fn viscosityKernel(dist: f32, h: f32) -> f32 {
    if (dist >= h) { return 0.0; }
    let v = h * h - dist * dist;
    return v * v * v * params.poly6Factor;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let pos = positions[idx].xyz;
    var vel = velocities[idx].xyz;
    let myDensity = densities[idx];
    let coord = getCellCoord(pos, params);
    let h = params.smoothingRadius;

    let pressure = pressureFromDensity(myDensity.x);
    let nearPressure = nearPressureFromDensity(myDensity.y);

    var pressureForce = vec3<f32>(0.0);
    var viscosityForce = vec3<f32>(0.0);

    let zStart = select(-1, 0, params.dimensions == 2u);
    let zEnd = select(1, 0, params.dimensions == 2u);

    for (var dz = zStart; dz <= zEnd; dz++) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let neighborCoord = coord + vec3<i32>(dx, dy, dz);
                if (any(neighborCoord < vec3<i32>(0)) || any(neighborCoord >= vec3<i32>(params.gridDims))) {
                    continue;
                }

                let neighborCell = cellHash(neighborCoord, params);
                let cellStart = cellOffsets[neighborCell];
                let cellEnd = select(cellOffsets[neighborCell + 1u], params.particleCount, neighborCell + 1u >= params.totalCells);

                for (var j = cellStart; j < cellEnd; j++) {
                    let neighborIdx = sortedIndices[j];
                    if (neighborIdx == idx) { continue; }

                    let neighborPos = positions[neighborIdx].xyz;
                    let neighborVel = velocities[neighborIdx].xyz;
                    let neighborDensity = densities[neighborIdx];

                    let diff = pos - neighborPos;
                    let dist = length(diff);
                    if (dist < 0.0001) { continue; }

                    let dir = diff / dist;

                    // Pressure force
                    let neighborPressure = pressureFromDensity(neighborDensity.x);
                    let neighborNearPressure = nearPressureFromDensity(neighborDensity.y);
                    let sharedPressure = (pressure + neighborPressure) * 0.5;
                    let sharedNearPressure = (nearPressure + neighborNearPressure) * 0.5;

                    pressureForce += dir * (densityDerivative(dist, h) * sharedPressure
                                          + nearDensityDerivative(dist, h) * sharedNearPressure);

                    // Viscosity force
                    let viscWeight = viscosityKernel(dist, h);
                    viscosityForce += (neighborVel - vel) * viscWeight;
                }
            }
        }
    }

    // Apply pressure + viscosity
    let safeDensity = max(myDensity.x, 0.001);
    vel += (pressureForce / safeDensity + viscosityForce * params.viscosity) * params.dt;

    // Gravity
    vel += params.gravity * params.dt;

    // Return to original position
    let origPos = originalPositions[idx].xyz;
    let toOrigin = origPos - pos;
    vel += toOrigin * params.returnToOriginStrength;

    // Mouse interaction (NDC-space proximity)
    if (params.mouseStrength > 0.001) {
        let projected = projectionMatrix * viewMatrix * worldMatrix * vec4<f32>(pos, 1.0);
        let ndc = projected.xyz / projected.w;
        var ndcMouse = params.mousePos;
        ndcMouse.y *= -1.0;
        let distToMouse = distance(ndcMouse, ndc.xy);

        if (distToMouse < params.mouseRadius * 2.0) {
            let nDist = distToMouse / (params.mouseRadius * 2.0);
            let displaceNDC = vec2<f32>(
                params.mouseDir.x * params.mouseForce * (1.0 - nDist) * -1.0,
                params.mouseDir.y * params.mouseForce * (1.0 - nDist)
            );
            let worldDisplace = (inverseViewMatrix * vec4<f32>(displaceNDC, 0.0, 0.0)).xyz;
            vel += worldDisplace * params.mouseStrength;
        }
    }

    // Damping
    vel *= params.damping;

    // Velocity limit
    let maxVel = 1200.0;
    let speed = length(vel);
    if (speed > maxVel) {
        vel = vel / speed * maxVel;
    }

    velocities[idx] = vec4<f32>(vel, 0.0);
}
`;
```

- [ ] **Step 3: Create integrate shader**

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var pos = positions[idx];
    var vel = velocities[idx];

    // Integrate
    pos = vec4<f32>(pos.xyz + vel.xyz * params.dt, pos.w);

    // Boundary collision (reflect)
    let bMin = params.worldBoundsMin;
    let bMax = params.worldBoundsMax;
    let bounce = -0.5;

    if (pos.x < bMin.x) { pos.x = bMin.x; vel.x *= bounce; }
    if (pos.x > bMax.x) { pos.x = bMax.x; vel.x *= bounce; }
    if (pos.y < bMin.y) { pos.y = bMin.y; vel.y *= bounce; }
    if (pos.y > bMax.y) { pos.y = bMax.y; vel.y *= bounce; }

    if (params.dimensions == 3u) {
        if (pos.z < bMin.z) { pos.z = bMin.z; vel.z *= bounce; }
        if (pos.z > bMax.z) { pos.z = bMax.z; vel.z *= bounce; }
    } else {
        pos.z = 0.0;
        vel.z = 0.0;
    }

    positions[idx] = pos;
    velocities[idx] = vel;
}
`;
```

- [ ] **Step 4: Commit**

```bash
git add src/simulations/fluid/shaders/density.wgsl.ts \
        src/simulations/fluid/shaders/forces.wgsl.ts \
        src/simulations/fluid/shaders/integrate.wgsl.ts
git commit -m "feat: add SPH density, forces, and integration shaders"
```

---

## Task 7: Create FluidSimulation orchestrator class

**Files:**
- Create: `src/simulations/fluid/FluidSimulation.ts`

This is the main class that creates all buffers, Compute instances, and orchestrates the simulation pipeline.

- [ ] **Step 1: Create the FluidSimulation class**

```typescript
import { Renderer } from '../../renderers/Renderer';
import { Compute } from '../../materials/Compute';
import { ComputeBuffer } from '../../buffers/ComputeBuffer';
import { BufferBase } from '../../buffers/BufferBase';
import { Matrix4 } from '../../math/Matrix4';
import { IBindable } from '../../buffers/IBindable';
import {
    FluidSimulationOptions,
    DEFAULT_OPTIONS,
    PARAMS,
    PRESETS,
    computeKernelFactors2D,
    computeKernelFactors3D,
} from './FluidSimulationParams';

import { shaderCode as gridClearShader } from './shaders/grid-clear.wgsl';
import { shaderCode as gridAssignShader } from './shaders/grid-assign.wgsl';
import { shaderCode as prefixSumLocalShader } from './shaders/prefix-sum-local.wgsl';
import { shaderCode as prefixSumTopShader } from './shaders/prefix-sum-top.wgsl';
import { shaderCode as prefixSumDistributeShader } from './shaders/prefix-sum-distribute.wgsl';
import { shaderCode as scatterShader } from './shaders/scatter.wgsl';
import { shaderCode as densityShader } from './shaders/density.wgsl';
import { shaderCode as forcesShader } from './shaders/forces.wgsl';
import { shaderCode as integrateShader } from './shaders/integrate.wgsl';

const MAX_GRID_CELLS = 262144; // 256K
const PREFIX_SUM_BLOCK_SIZE = 512;

class FluidSimulation {
    public params: FluidSimulationOptions;

    private renderer: Renderer;
    private particleCount: number = 0;

    // Params uniform buffer (dual view for mixed f32/u32)
    private paramsF32!: Float32Array;
    private paramsU32!: Uint32Array;
    private paramsBuffer!: ComputeBuffer;

    // Internal simulation buffers
    private velocitiesBuffer!: ComputeBuffer;
    private densitiesBuffer!: ComputeBuffer;
    private cellIndicesBuffer!: ComputeBuffer;
    private cellCountsBuffer!: ComputeBuffer;
    private cellOffsetsBuffer!: ComputeBuffer;
    private scatterCountersBuffer!: ComputeBuffer;
    private sortedIndicesBuffer!: ComputeBuffer;
    private blockSumsBuffer!: ComputeBuffer;

    // External buffers (passed in)
    private positionsBuffer!: ComputeBuffer;
    private originalPositionsBuffer!: ComputeBuffer;

    // Camera matrices (for mouse interaction)
    private viewMatrix: IBindable;
    private projectionMatrix: IBindable;
    private inverseViewMatrix: IBindable;
    private worldMatrix: IBindable;

    // Compute passes
    private gridClearCountsPass!: Compute;
    private gridClearScatterPass!: Compute;
    private gridAssignPass!: Compute;
    private prefixSumLocalPass!: Compute;
    private prefixSumTopPass!: Compute;
    private prefixSumDistributePass!: Compute;
    private scatterPass!: Compute;
    private densityPass!: Compute;
    private forcesPass!: Compute;
    private integratePass!: Compute;

    // Grid dimensions
    private gridDims: [number, number, number] = [1, 1, 1];
    private gridOrigin: [number, number, number] = [0, 0, 0];
    private totalCells: number = 1;
    private worldBoundsMin: [number, number, number] = [0, 0, 0];
    private worldBoundsMax: [number, number, number] = [0, 0, 0];

    constructor(renderer: Renderer, options?: Partial<FluidSimulationOptions>) {
        this.renderer = renderer;
        this.params = { ...DEFAULT_OPTIONS, ...options };

        // Default identity matrices (overridden in initialize if camera provided)
        this.viewMatrix = new Matrix4();
        this.projectionMatrix = new Matrix4();
        this.inverseViewMatrix = new Matrix4();
        this.worldMatrix = new Matrix4();
    }

    public initialize(
        positionsBuffer: ComputeBuffer,
        originalPositionsBuffer: ComputeBuffer,
        cameraBindings?: {
            viewMatrix: IBindable;
            projectionMatrix: IBindable;
            inverseViewMatrix: IBindable;
            worldMatrix: IBindable;
        }
    ): void {
        this.positionsBuffer = positionsBuffer;
        this.originalPositionsBuffer = originalPositionsBuffer;
        this.particleCount = this.params.maxParticles;

        if (cameraBindings) {
            this.viewMatrix = cameraBindings.viewMatrix;
            this.projectionMatrix = cameraBindings.projectionMatrix;
            this.inverseViewMatrix = cameraBindings.inverseViewMatrix;
            this.worldMatrix = cameraBindings.worldMatrix;
        }

        this.computeGridFromPositions(positionsBuffer);
        this.createBuffers();
        this.createComputePasses();
    }

    private computeGridFromPositions(positionsBuffer: ComputeBuffer): void {
        // Scan initial positions to determine world bounds
        const data = (positionsBuffer as any).buffer as Float32Array;
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < this.particleCount; i++) {
            const x = data[i * 4];
            const y = data[i * 4 + 1];
            const z = data[i * 4 + 2];
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }

        // Add padding
        const pad = this.params.worldBoundsPadding;
        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;
        const rangeZ = this.params.dimensions === 3 ? ((maxZ - minZ) || 1) : 0.01;
        const padX = rangeX * pad;
        const padY = rangeY * pad;
        const padZ = rangeZ * pad;

        this.worldBoundsMin = [minX - padX, minY - padY, minZ - padZ];
        this.worldBoundsMax = [maxX + padX, maxY + padY, maxZ + padZ];

        const cs = this.params.smoothingRadius;
        const gx = Math.ceil((this.worldBoundsMax[0] - this.worldBoundsMin[0]) / cs);
        const gy = Math.ceil((this.worldBoundsMax[1] - this.worldBoundsMin[1]) / cs);
        const gz = this.params.dimensions === 3
            ? Math.ceil((this.worldBoundsMax[2] - this.worldBoundsMin[2]) / cs)
            : 1;

        // Clamp to max grid cells
        const maxPerAxis2D = Math.floor(Math.sqrt(MAX_GRID_CELLS));
        const maxPerAxis3D = Math.floor(Math.cbrt(MAX_GRID_CELLS));
        const maxPerAxis = this.params.dimensions === 3 ? maxPerAxis3D : maxPerAxis2D;

        this.gridDims = [
            Math.min(Math.max(gx, 1), maxPerAxis),
            Math.min(Math.max(gy, 1), maxPerAxis),
            Math.min(Math.max(gz, 1), this.params.dimensions === 3 ? maxPerAxis : 1),
        ];
        this.totalCells = this.gridDims[0] * this.gridDims[1] * this.gridDims[2];
        this.gridOrigin = [...this.worldBoundsMin];
    }

    private createBuffers(): void {
        const N = this.particleCount;

        // Params uniform
        this.paramsF32 = new Float32Array(PARAMS.BUFFER_SIZE);
        this.paramsU32 = new Uint32Array(this.paramsF32.buffer);
        this.paramsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.paramsF32,
        });

        // Velocities (vec4 per particle — w reserved for future angular vel)
        this.velocitiesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N * 4),
        });

        // Densities (vec2 per particle — density + nearDensity)
        this.densitiesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N * 2),
        });

        // Cell indices (u32 per particle)
        this.cellIndicesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N),
        });

        // Cell counts (u32 per cell — max capacity)
        this.cellCountsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Cell offsets (u32 per cell — prefix sum output)
        this.cellOffsetsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Scatter counters (u32 per cell — reused for scatter atomics)
        this.scatterCountersBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Sorted indices (u32 per particle)
        this.sortedIndicesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N),
        });

        // Block sums for prefix sum (max 512 blocks)
        const numBlocks = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);
        this.blockSumsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(Math.max(numBlocks, 1)),
        });
    }

    private createComputePasses(): void {
        const C = GPUShaderStage.COMPUTE;

        // Grid clear — two instances (one for cellCounts, one for scatterCounters)
        this.gridClearCountsPass = new Compute(gridClearShader, [
            { binding: 0, visibility: C, value: this.cellCountsBuffer },
        ]);

        this.gridClearScatterPass = new Compute(gridClearShader, [
            { binding: 0, visibility: C, value: this.scatterCountersBuffer },
        ]);

        // Grid assign
        this.gridAssignPass = new Compute(gridAssignShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.cellIndicesBuffer },
            { binding: 2, visibility: C, value: this.cellCountsBuffer },
            { binding: 3, visibility: C, value: this.paramsBuffer },
        ]);

        // Prefix sum
        this.prefixSumLocalPass = new Compute(prefixSumLocalShader, [
            { binding: 0, visibility: C, value: this.cellCountsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.blockSumsBuffer },
        ]);

        this.prefixSumTopPass = new Compute(prefixSumTopShader, [
            { binding: 0, visibility: C, value: this.blockSumsBuffer },
        ]);

        this.prefixSumDistributePass = new Compute(prefixSumDistributeShader, [
            { binding: 0, visibility: C, value: this.blockSumsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
        ]);

        // Scatter
        this.scatterPass = new Compute(scatterShader, [
            { binding: 0, visibility: C, value: this.cellIndicesBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.scatterCountersBuffer },
            { binding: 3, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 4, visibility: C, value: this.paramsBuffer },
        ]);

        // SPH density
        this.densityPass = new Compute(densityShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 3, visibility: C, value: this.densitiesBuffer },
            { binding: 4, visibility: C, value: this.paramsBuffer },
        ]);

        // SPH forces
        this.forcesPass = new Compute(forcesShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.densitiesBuffer },
            { binding: 3, visibility: C, value: this.originalPositionsBuffer },
            { binding: 4, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 5, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 6, visibility: C, value: this.paramsBuffer },
            { binding: 7, visibility: C, value: this.viewMatrix },
            { binding: 8, visibility: C, value: this.projectionMatrix },
            { binding: 9, visibility: C, value: this.inverseViewMatrix },
            { binding: 10, visibility: C, value: this.worldMatrix },
        ]);

        // Integration
        this.integratePass = new Compute(integrateShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.paramsBuffer },
        ]);
    }

    private packParams(dt: number, mouseStrength: number, mousePosition?: { x: number, y: number }, mouseDirection?: { x: number, y: number }): void {
        const p = this.params;
        const f = this.paramsF32;
        const u = this.paramsU32;

        f[PARAMS.dt] = dt / p.substeps;
        u[PARAMS.particleCount] = this.particleCount;
        u[PARAMS.dimensions] = p.dimensions;
        f[PARAMS.smoothingRadius] = p.smoothingRadius;
        f[PARAMS.pressureMultiplier] = p.pressureMultiplier;
        f[PARAMS.densityTarget] = p.densityTarget;
        f[PARAMS.nearPressureMultiplier] = p.nearPressureMultiplier;
        f[PARAMS.viscosity] = p.viscosity;
        f[PARAMS.damping] = p.damping;
        f[PARAMS.returnToOriginStrength] = p.returnToOriginStrength;
        f[PARAMS.mouseStrength] = mouseStrength;
        f[PARAMS.mouseRadius] = p.mouseRadius;
        f[PARAMS.gravityX] = p.gravity[0];
        f[PARAMS.gravityY] = p.gravity[1];
        f[PARAMS.gravityZ] = p.gravity[2];
        f[PARAMS.mouseForce] = p.mouseForce;
        f[PARAMS.mousePosX] = mousePosition?.x ?? 0;
        f[PARAMS.mousePosY] = mousePosition?.y ?? 0;
        f[PARAMS.mouseDirX] = mouseDirection?.x ?? 0;
        f[PARAMS.mouseDirY] = mouseDirection?.y ?? 0;
        u[PARAMS.gridDimsX] = this.gridDims[0];
        u[PARAMS.gridDimsY] = this.gridDims[1];
        u[PARAMS.gridDimsZ] = this.gridDims[2];
        f[PARAMS.cellSize] = p.smoothingRadius;
        f[PARAMS.gridOriginX] = this.gridOrigin[0];
        f[PARAMS.gridOriginY] = this.gridOrigin[1];
        f[PARAMS.gridOriginZ] = this.gridOrigin[2];
        u[PARAMS.totalCells] = this.totalCells;
        f[PARAMS.worldBoundsMinX] = this.worldBoundsMin[0];
        f[PARAMS.worldBoundsMinY] = this.worldBoundsMin[1];
        f[PARAMS.worldBoundsMinZ] = this.worldBoundsMin[2];
        f[PARAMS.worldBoundsMaxX] = this.worldBoundsMax[0];
        f[PARAMS.worldBoundsMaxY] = this.worldBoundsMax[1];
        f[PARAMS.worldBoundsMaxZ] = this.worldBoundsMax[2];

        // Kernel factors
        const kernels = p.dimensions === 3
            ? computeKernelFactors3D(p.smoothingRadius)
            : computeKernelFactors2D(p.smoothingRadius);
        f[PARAMS.poly6Factor] = kernels.poly6;
        f[PARAMS.spikyPow2Factor] = kernels.spikyPow2;
        f[PARAMS.spikyPow3Factor] = kernels.spikyPow3;
        f[PARAMS.spikyPow2DerivFactor] = kernels.spikyPow2Deriv;
        f[PARAMS.spikyPow3DerivFactor] = kernels.spikyPow3Deriv;

        this.paramsBuffer.needsUpdate = true;
    }

    public setPreset(presetName: string): void {
        const preset = PRESETS[presetName];
        if (!preset) { return; }
        Object.assign(this.params, preset);
    }

    public async update(
        dt: number,
        mousePosition?: { x: number; y: number },
        mouseDirection?: { x: number; y: number },
        mouseStrength: number = 0
    ): Promise<void> {
        const N = this.particleCount;
        const particleWorkgroups = Math.ceil(N / 64);
        const gridWorkgroups = Math.ceil(this.totalCells / 256);
        const prefixSumWorkgroups = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);

        for (let s = 0; s < this.params.substeps; s++) {
            this.packParams(dt, mouseStrength, mousePosition, mouseDirection);

            await this.renderer.computeBatch([
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
            ]);
        }
    }
}

export { FluidSimulation };
```

- [ ] **Step 2: Verify it compiles**

Run: `npx tsc --noEmit src/simulations/fluid/FluidSimulation.ts`
Expected: no errors. If there are import resolution issues, check relative paths match the file map.

- [ ] **Step 3: Commit**

```bash
git add src/simulations/fluid/FluidSimulation.ts
git commit -m "feat: add FluidSimulation orchestrator class with spatial hash grid SPH"
```

---

## Task 8: Export from main.ts

**Files:**
- Modify: `src/main.ts` (add export at bottom, after existing exports)

- [ ] **Step 1: Add export**

Add at the end of `src/main.ts`:

```typescript
export { FluidSimulation } from "./simulations/fluid/FluidSimulation";
export { FluidSimulationOptions, PRESETS as FluidPresets } from "./simulations/fluid/FluidSimulationParams";
```

- [ ] **Step 2: Commit**

```bash
git add src/main.ts
git commit -m "feat: export FluidSimulation from main entry point"
```

---

## Task 9: Create example `index_fluid_text.html`

**Files:**
- Create: `examples/index_fluid_text.html`

- [ ] **Step 1: Create the full example**

```html
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="/vite.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Fluid Text — Kansei</title>
    <style>
      body {
        margin: 0;
        padding: 0;
        overflow: hidden;
        background-color: black;
      }
      canvas {
        width: 100%;
        height: 100%;
      }
      #info {
        position: absolute;
        bottom: 10px;
        left: 50%;
        transform: translateX(-50%);
        color: white;
        font-family: sans-serif;
      }
      #info a { color: white; }
    </style>
  </head>
  <body>
    <script type="module">
      import { FontLoader } from '../src/sdf/text/FontLoader'
      import { Renderer } from '../src/renderers/Renderer'
      import { Material } from '../src/materials/Material'
      import { Renderable } from '../src/objects/Renderable'
      import { Scene } from '../src/objects/Scene'
      import { Camera } from '../src/cameras/Camera'
      import { CameraControls } from '../src/controls/CameraControls'
      import { shaderCode } from '../src/materials/shaders/rendering/msdf/TextRenderShader'
      import { Sampler } from '../src/buffers/Sampler'
      import { Vector4 } from '../src/math/Vector4'
      import { TextGeometry } from '../src/geometries/TextGeometry'
      import { MouseVectors } from '../src/controls/MouseVectors'
      import { FluidSimulation } from '../src/simulations/fluid/FluidSimulation'
      import { PRESETS } from '../src/simulations/fluid/FluidSimulationParams'
      import { poem } from './assets/copy/poem.js'
      import { Pane } from 'https://cdn.jsdelivr.net/npm/tweakpane@4/dist/tweakpane.min.js'

      // ── Renderer ──────────────────────────────────────────
      const renderer = new Renderer({
        alphaMode: 'premultiplied',
        clearColor: new Vector4(0.02, 0.02, 0.04, 1),
        width: window.innerWidth,
        height: window.innerHeight,
        devicePixelRatio: Math.min(window.devicePixelRatio, 2),
        antialias: true,
        sampleCount: 1,
      });
      await renderer.initialize();
      document.body.appendChild(renderer.canvas);

      // ── Scene ─────────────────────────────────────────────
      const scene = new Scene();
      const camera = new Camera(20, 0.1, 100000, window.innerWidth / window.innerHeight);
      const cameraControls = new CameraControls(camera, scene.position);
      const mouseVectors = new MouseVectors();

      // ── Text Geometry ─────────────────────────────────────
      const fontLoader = new FontLoader();
      const fontInfo = await fontLoader.load('assets/fonts/L10-medium.arfont');

      const geometry = new TextGeometry({
        text: poem,
        fontInfo: fontInfo,
        width: 40,
        height: 100,
        fontSize: 25,
        color: new Vector4(1, 1, 1, 1),
      });
      const sampler = new Sampler('linear', 'linear');
      const material = new Material(shaderCode, {
        bindings: [
          { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, value: fontInfo.sdfTexture },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, value: sampler },
        ],
        transparent: true,
      });
      const mesh = new Renderable(geometry, material);
      mesh.position.x = -20;
      mesh.position.y = 8;
      scene.add(mesh);

      // ── Fluid Simulation ──────────────────────────────────
      const simSettings = {
        maxParticles: poem.length,
        dimensions: 2,
        smoothingRadius: 1.0,
        pressureMultiplier: 10.0,
        nearPressureMultiplier: 18.0,
        densityTarget: 1.5,
        viscosity: 0.3,
        damping: 0.998,
        gravity: [0, 0, 0],
        returnToOriginStrength: 0.002,
        mouseRadius: 0.5,
        mouseForce: 300.0,
        substeps: 3,
        worldBoundsPadding: 0.2,
      };

      const sim = new FluidSimulation(renderer, simSettings);
      sim.initialize(
        geometry.extraBuffers[0],
        geometry.extraBuffers[0].clone(),
        {
          viewMatrix: camera.viewMatrix,
          projectionMatrix: camera.projectionMatrix,
          inverseViewMatrix: camera.inverseViewMatrix,
          worldMatrix: mesh.worldMatrix,
        }
      );

      // ── Tweakpane ─────────────────────────────────────────
      const pane = new Pane({ title: 'Fluid Simulation' });

      // Presets
      const presetObj = { preset: 'custom' };
      const presetOptions = { custom: 'custom' };
      for (const key of Object.keys(PRESETS)) {
        presetOptions[key] = key;
      }
      pane.addBinding(presetObj, 'preset', {
        label: 'Preset',
        options: presetOptions,
      }).on('change', (ev) => {
        if (ev.value === 'custom') return;
        sim.setPreset(ev.value);
        Object.assign(simSettings, sim.params);
        pane.refresh();
      });

      // Simulation folder
      const simFolder = pane.addFolder({ title: 'Simulation' });
      simFolder.addBinding(simSettings, 'dimensions', { options: { '2D': 2, '3D': 3 } }).on('change', () => { sim.params.dimensions = simSettings.dimensions; });
      simFolder.addBinding(simSettings, 'substeps', { min: 1, max: 8, step: 1 }).on('change', () => { sim.params.substeps = simSettings.substeps; });
      simFolder.addBinding(simSettings, 'smoothingRadius', { min: 0.1, max: 5.0, step: 0.1, label: 'radius' }).on('change', () => { sim.params.smoothingRadius = simSettings.smoothingRadius; });

      // SPH folder
      const sphFolder = pane.addFolder({ title: 'SPH Parameters' });
      sphFolder.addBinding(simSettings, 'pressureMultiplier', { min: 1, max: 100, step: 0.5, label: 'pressure' }).on('change', () => { sim.params.pressureMultiplier = simSettings.pressureMultiplier; });
      sphFolder.addBinding(simSettings, 'nearPressureMultiplier', { min: 1, max: 100, step: 0.5, label: 'near pressure' }).on('change', () => { sim.params.nearPressureMultiplier = simSettings.nearPressureMultiplier; });
      sphFolder.addBinding(simSettings, 'densityTarget', { min: 0.1, max: 10.0, step: 0.1, label: 'density target' }).on('change', () => { sim.params.densityTarget = simSettings.densityTarget; });
      sphFolder.addBinding(simSettings, 'viscosity', { min: 0.0, max: 1.0, step: 0.01 }).on('change', () => { sim.params.viscosity = simSettings.viscosity; });
      sphFolder.addBinding(simSettings, 'damping', { min: 0.9, max: 1.0, step: 0.001 }).on('change', () => { sim.params.damping = simSettings.damping; });

      // Forces folder
      const forcesFolder = pane.addFolder({ title: 'Forces' });
      const gravityObj = { x: 0, y: 0, z: 0 };
      forcesFolder.addBinding(gravityObj, 'y', { min: -20, max: 0, step: 0.1, label: 'gravity Y' }).on('change', () => { sim.params.gravity = [gravityObj.x, gravityObj.y, gravityObj.z]; });
      forcesFolder.addBinding(simSettings, 'returnToOriginStrength', { min: 0.0, max: 0.1, step: 0.001, label: 'return force' }).on('change', () => { sim.params.returnToOriginStrength = simSettings.returnToOriginStrength; });
      forcesFolder.addBinding(simSettings, 'mouseRadius', { min: 0.1, max: 2.0, step: 0.05, label: 'mouse radius' }).on('change', () => { sim.params.mouseRadius = simSettings.mouseRadius; });
      forcesFolder.addBinding(simSettings, 'mouseForce', { min: 0, max: 1000, step: 10, label: 'mouse force' }).on('change', () => { sim.params.mouseForce = simSettings.mouseForce; });

      // ── Render loop ───────────────────────────────────────
      const time = new Vector4(0, 0, 0, 0);

      async function animate() {
        time.x = performance.now() / 1000;
        time.y = time.x - time.z;
        time.z = time.x;

        const dt = Math.min(time.y, 1 / 30); // Cap dt
        mouseVectors.update(dt);
        cameraControls.update();

        await sim.update(dt, mouseVectors.mousePosition, mouseVectors.mouseDirection, mouseVectors.mouseStrength);

        renderer.render(scene, camera);
        requestAnimationFrame(animate);
      }
      animate();

      // ── Resize ────────────────────────────────────────────
      function onResize() {
        renderer.setSize(window.innerWidth, window.innerHeight);
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
      }
      window.addEventListener('resize', onResize);
    </script>
    <div id="info">
      Made with ❤️ and <a href="https://github.com/siroko/kansei">KANSEI</a> — Fluid Text Demo
    </div>
  </body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add examples/index_fluid_text.html
git commit -m "feat: add fluid text example with tweakpane controls and presets"
```

---

## Task 10: Visual Verification & Debugging

- [ ] **Step 1: Start the dev server**

Run: `npx vite` (or whatever dev server the project uses)
Open: `http://localhost:5173/examples/index_fluid_text.html`

Expected: Text renders with MSDF shader. Particles should respond to mouse interaction (displacement and return). Tweakpane panel visible with all controls.

- [ ] **Step 2: Test presets**

In the Tweakpane panel:
1. Select "Water" preset → particles should flow downward with gravity, splash when mouse moves through
2. Select "Viscous Honey" preset → slower, thicker movement
3. Select "Gas" preset → particles spread out quickly, lighter feel
4. Select "Zero-G Blob" preset → particles form a cohesive blob, no gravity

- [ ] **Step 3: Debug common issues**

If particles don't move: check browser console for WebGPU errors. Verify compute dispatches complete. Common issues:
- **Bind group layout mismatch**: shader binding types must match `ComputeBuffer.type`. All storage buffers must use `var<storage, read_write>`.
- **Buffer size alignment**: WebGPU requires buffer sizes to be multiples of 4 bytes. All our buffers use Float32Array which guarantees this.
- **Prefix sum incorrect**: verify `cellOffsets` via `renderer.readBackBuffer()` — should be monotonically increasing with last value ≤ particleCount.
- **Grid hash out of bounds**: if particles move beyond worldBounds, `getCellCoord` may produce negative values. The `cellHash` function clamps to valid range.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "fix: address any issues found during visual verification"
```

(Only if changes were needed)

---

## Dependency Graph

```
Task 1 (Renderer.computeBatch) ──┐
Task 2 (Params interface) ───────┤
Task 3 (SimParams struct) ───────┤
Task 4 (Grid shaders) ───────────┼──→ Task 7 (FluidSimulation) ──→ Task 8 (Exports) ──→ Task 9 (Example) ──→ Task 10 (Verify)
Task 5 (Prefix sum shaders) ─────┤
Task 6 (SPH shaders) ────────────┘
```

Tasks 1–6 are independent and can be implemented in parallel. Task 7 depends on all of them. Tasks 8–10 are sequential.
