# Fluid Simulation Package — Design Spec

## Overview

GPU-accelerated SPH (Smoothed Particle Hydrodynamics) fluid simulation for the Kansei WebGPU engine. Ported from the Three.js `GPUTextSimulation` in the felixmartinez project, rewritten as native WebGPU compute shaders with a spatial hash grid for O(n) neighbor search (replacing the original O(n^2) brute force).

Supports 2D and 3D modes. Outputs particle positions directly into a `ComputeBuffer` usable as vertex data — zero-copy integration with any instanced `Renderable`.

## Package Structure

```
src/simulations/fluid/
  FluidSimulation.ts          # Main orchestrator class
  FluidSimulationParams.ts    # Parameter interface + defaults
  shaders/
    grid-clear.wgsl.ts        # Zero cell counts
    grid-assign.wgsl.ts       # Hash particles into grid cells
    prefix-sum-local.wgsl.ts  # Per-block Blelloch scan
    prefix-sum-top.wgsl.ts    # Scan block sums
    prefix-sum-distribute.wgsl.ts  # Add block offsets
    scatter.wgsl.ts           # Reorder particles by cell
    density.wgsl.ts           # SPH density + near-density
    forces.wgsl.ts            # Pressure + viscosity + external forces
    integrate.wgsl.ts         # Position integration + boundaries
```

## Public API

```typescript
import { FluidSimulation } from '../src/simulations/fluid/FluidSimulation'

const sim = new FluidSimulation(renderer, {
  maxParticles: 10000,
  dimensions: 2,                // 2 | 3
  smoothingRadius: 1.0,
  pressureMultiplier: 10.0,
  densityTarget: 1.5,
  viscosity: 0.5,
  damping: 0.999,
  gravity: [0, -9.8, 0],
  returnToOriginStrength: 0.001,
  mouseRadius: 0.5,
  mouseForce: 300.0,
  substeps: 3,
  worldBounds: null,            // auto-inferred from positions + padding
});

// Initialize — pass positions buffer (will be mutated in-place)
sim.initialize(positionsBuffer, originalPositionsBuffer);

// Per-frame update — dispatches all compute passes
await sim.update(dt, mousePosition, mouseDirection, mouseStrength);

// Output: positionsBuffer is updated in-place, already bound as vertex data

// Runtime parameter changes
sim.params.viscosity = 0.9;
sim.params.gravity = [0, -2.0, 0];

// Preset application
sim.setParams({ pressureMultiplier: 3, viscosity: 0.9, damping: 0.99, ... });
```

### Key design decisions

- **Mutates input positions buffer in-place.** The `ComputeBuffer` from `TextGeometry.extraBuffers[0]` has `VERTEX | STORAGE` usage, so the simulation writes directly to what the renderer reads. Zero-copy.
- **Single `update()` call.** Internally dispatches all compute passes in one command encoder per substep (1 GPU submit per substep, not 1 per pass).
- **Parameters are mutable at runtime.** Tweakpane (or user code) binds directly to `sim.params.*`, written to a uniform buffer each frame via `writeBuffer`.

## Simulation Pipeline

9 compute dispatches per substep, batched into 1 GPU submit:

```
Pass 1: Grid Clear         (gridCells/256 workgroups)
  Zero cellCounts array

Pass 2: Grid Assign        (N/64 workgroups)
  hash(floor(pos / cellSize)) -> cellIndex per particle
  atomicAdd(cellCounts[cellIndex], 1)

Pass 3: Prefix Sum Local   (gridCells/512 workgroups)
  Work-efficient Blelloch scan per block of 512 cells
  Write block totals to blockSums auxiliary buffer

Pass 4: Prefix Sum Top     (1 workgroup)
  Scan blockSums (max 512 entries)

Pass 5: Prefix Sum Distribute (gridCells/512 workgroups)
  Add block offsets back to produce final global cellOffsets

Pass 6: Scatter            (N/64 workgroups)
  Write particle index into sortedIndices at
  cellOffsets[cellIndex] + atomicAdd(cellCounter[cellIndex], 1)

Pass 7: Density            (N/64 workgroups)
  For each particle, iterate 9 (2D) or 27 (3D) neighbor cells
  Look up sortedIndices[cellOffsets[cell]..cellOffsets[cell+1]]
  Compute SPH density + near-density using Spiky kernels
  Output: densities[i] = vec2(density, nearDensity)

Pass 8: Forces             (N/64 workgroups)
  Pressure forces from density gradients (Spiky derivatives)
  Viscosity from kernel-weighted neighbor velocity differences (Poly6)
  + gravity (uniform)
  + return-to-origin spring force
  + mouse interaction (NDC-space proximity, repulsion along mouse direction)
  + damping (velocity *= damping)
  Output: velocities[i] updated

Pass 9: Integrate          (N/64 workgroups)
  pos += vel * dt/substeps
  Boundary collision (clamp + reflect velocity)
  In 2D mode: zero Z position and velocity
  Output: positions[i] updated in-place
```

## Buffer Layout

| Buffer | Type | Size | Usage Flags |
|--------|------|------|-------------|
| `positions` | `storage, read_write` | N x vec4f | STORAGE \| VERTEX \| COPY_SRC |
| `originalPositions` | `storage, read` | N x vec4f | STORAGE \| COPY_SRC |
| `velocities` | `storage, read_write` | N x vec4f | STORAGE |
| `densities` | `storage, read_write` | N x vec2f | STORAGE |
| `cellIndices` | `storage, read_write` | N x u32 | STORAGE |
| `cellCounts` | `storage, read_write` | gridCells x u32 | STORAGE |
| `cellOffsets` | `storage, read_write` | gridCells x u32 | STORAGE |
| `sortedIndices` | `storage, read_write` | N x u32 | STORAGE |
| `blockSums` | `storage, read_write` | 512 x u32 | STORAGE |
| `params` | `uniform` | ~128 bytes | UNIFORM \| COPY_DST |

### Params uniform struct

```wgsl
struct SimParams {
  dt: f32,
  particleCount: u32,
  dimensions: u32,             // 2 or 3
  smoothingRadius: f32,
  pressureMultiplier: f32,
  densityTarget: f32,
  nearPressureMultiplier: f32,
  viscosity: f32,
  damping: f32,
  gravity: vec3<f32>,
  returnToOriginStrength: f32,
  mousePos: vec2<f32>,
  mouseDir: vec2<f32>,
  mouseStrength: f32,
  mouseRadius: f32,
  mouseForce: f32,
  gridDims: vec3<u32>,         // grid resolution per axis
  gridOrigin: vec3<f32>,       // world-space min corner
  cellSize: f32,
  worldBoundsMin: vec3<f32>,
  worldBoundsMax: vec3<f32>,
  // SPH kernel scaling factors (precomputed on CPU)
  poly6Factor: f32,
  spikyPow2Factor: f32,
  spikyPow3Factor: f32,
  spikyPow2DerivFactor: f32,
  spikyPow3DerivFactor: f32,
};
```

## Spatial Hash Grid

- `cellSize = smoothingRadius` — guarantees all neighbors within kernel range are in adjacent cells
- Grid dimensions auto-computed: `gridDims = ceil((worldBoundsMax - worldBoundsMin) / cellSize)`
- Total cells capped at 256K (fits two-level prefix sum: 512 blocks x 512 cells/block)
- Hash function: `cellIndex = cellZ * gridDims.x * gridDims.y + cellY * gridDims.x + cellX`
- Out-of-bounds particles clamped to boundary cells
- World bounds: auto-inferred from initial positions with 20% padding, or user-specified

## SPH Kernels

Matching the source implementation (2D SPH kernels, extended to 3D):

### 2D kernels
```
poly6Factor         = 4 / (pi * h^8)
spikyPow2Factor     = 6 / (pi * h^4)
spikyPow3Factor     = 10 / (pi * h^5)
spikyPow2DerivFactor = 12 / (h^4 * pi)
spikyPow3DerivFactor = 30 / (h^5 * pi)
```

### 3D kernels
```
poly6Factor         = 315 / (64 * pi * h^9)
spikyPow2Factor     = 15 / (pi * h^6)
spikyPow3Factor     = 15 / (pi * h^6)  [standard 3D spiky]
spikyPow2DerivFactor = 45 / (pi * h^6)
spikyPow3DerivFactor = 45 / (pi * h^6)
```

Kernel factors are precomputed on CPU when `smoothingRadius` changes and uploaded via the params uniform.

## 2D vs 3D Mode

Controlled by `params.dimensions` uniform (2 or 3):

| Aspect | 2D | 3D |
|--------|----|----|
| Grid hash | `(cellX, cellY)` | `(cellX, cellY, cellZ)` |
| Neighbor cells | 9 (3x3) | 27 (3x3x3) |
| Z velocity/position | Zeroed in integrate pass | Active |
| Z boundaries | Skipped | Enforced |
| SPH kernel factors | 2D formulas | 3D formulas |
| Grid dims Z | 1 | Auto from bounds |

## Renderer Extension

New method on `Renderer` to batch multiple compute dispatches:

```typescript
public async computeBatch(
  passes: { compute: Compute, workgroupsX: number, workgroupsY?: number, workgroupsZ?: number }[]
): Promise<void>
```

Encodes all passes into a single `GPUCommandEncoder` with sequential compute passes. WebGPU guarantees storage buffer writes in pass N are visible to pass N+1 within the same command buffer. Existing `renderer.compute()` remains untouched.

## Example: index_fluid_text.html

Follows the pattern of `index_text.html` and `index_particles.html`:

1. Load font via `FontLoader`, create `TextGeometry` with poem text
2. Create `FluidSimulation` with `geometry.extraBuffers[0]` (positions)
3. Render text with MSDF shader (instanced quads)
4. Mouse interaction displaces particles as fluid, spring forces return them

### Tweakpane presets

| Preset | pressure | viscosity | damping | densityTarget | returnToOrigin | gravity.y |
|--------|----------|-----------|---------|---------------|----------------|-----------|
| Water | 10 | 0.3 | 0.998 | 1.5 | 0.002 | -9.8 |
| Viscous Honey | 3 | 0.9 | 0.99 | 3.0 | 0.005 | -2.0 |
| Gas | 20 | 0.05 | 0.995 | 0.5 | 0.001 | 0.0 |
| Zero-G Blob | 8 | 0.6 | 0.997 | 2.0 | 0.0 | 0.0 |

### Tweakpane folders

- **Simulation** — dimensions, smoothingRadius, substeps, gravity
- **SPH Parameters** — pressureMultiplier, densityTarget, viscosity, damping
- **Forces** — returnToOriginStrength, mouseRadius, mouseForce
- **Appearance** — textColor, bgColor

## Future Enhancements (out of scope for v1)

- **Angular momentum**: Store orientation (angle for 2D, quaternion for 3D) in a separate buffer. Pressure gradients at particle surface produce torque. Feed angular velocity into visual rotation of text glyphs.
- **Per-word Verlet constraints**: Word index stored per particle at init time (from TextGeometry word layout). Distance constraints between characters in same word, solved iteratively after the integrate pass. Preserves word shape while allowing fluid deformation.
- **Adaptive grid**: Resize grid bounds dynamically based on particle extent each frame.
- **Multiple emitters/sinks**: Particle sources and drains for continuous flow effects.
