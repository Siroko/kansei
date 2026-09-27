# Fluid sim optimization for 500K particles + fluid_clock antialiasing

Date: 2026-09-12. Scope: `rust/kansei-core/src/simulations/fluid/` and `rust/kansei-wasm/examples/fluid_clock/`.

## Goal
Make the SPH fluid sim fast enough that the fluid clock can run 500K particles at
smoothing radius 0.543 (same fluid volume as the tuned 80K @ h=1.0), then measure how
many sim substeps fit in a 16 ms frame. Separately, make the presented frame antialiased.

## Measured starting point (clock_fill_test, M4 Pro, CPU wall incl. GPU wait)
- 80K @ h=1.0, SCALE=1.9: 13.5 ms/frame (~3.5 ms/substep). Pool p50=-4.70 p95=0.87.
- 500K @ h=0.543, TARGET=99, SCALE=0.5: ~144 ms/substep. Explodes at SCALE >= 0.95
  (stable dt shrinks ~h^1.5; baseline already explodes at 2x dt).

## Sim changes (in measurement order)
1. Sorted neighbor search. Scatter also writes cell-ordered copies of positions and
   velocities. Density and forces run one thread per sorted slot and read neighbors
   contiguously; densities are stored in sorted order; forces writes velocity back to
   the particle's original index. Original-order `positions`/`velocities` stay the
   public buffers (attractor, density field, billboards untouched).
2. Grid cell = smoothing radius. Top-level prefix sum loops over 512-entry chunks;
   MAX_GRID_CELLS raised to 2M.
3. Inner loop: one squared-distance test per pair, kernels without per-call range
   checks, workgroup size tuned (64/128/256).
4. Merge the two grid-clear dispatches.

Out of scope: timestep policy (tuning parameter), half-radius cells, shared-memory
tiling, f16.

## Antialiasing
1. fluid_clock: size the canvas backing store by devicePixelRatio (clamped to 2) and
   pass the scaled size to the renderer, camera aspect and offscreen textures.
2. (Only if still needed) MSAA on the post-processing GBuffer path.

## Verification
clock_fill_test before/after each step: 80K pool profile must match within noise;
500K @ h=0.543 TARGET=99 SCALE=0.5 must stay settled (p95 < ~5). Report ms/step per step.

## Results (2026-09-12)
| Step | 80K ms/frame | 500K ms/step (2 substeps) |
|---|---|---|
| start | 13.5 | 144 |
| 1 sorted neighbor search | 3.6 | 13 |
| 2 cell = radius | 3.6 | 11.6 |
| 3 inner loop | 3.5 | 9.8 |
| 4 merged clears, steady state | 3.6 | 8.3 |

Workgroup size 64/128/256: no difference. Attractor+retag at 500K: 1.4 ms/frame.
Stability must be judged at equal simulated time (~7 s): 500K @ h=0.543 settles at
SCALE 0.25 only; SCALE 0.5 boils even with viscosity 3 / damping 0.999. Page ships with
sim time scale 0.25, density_target 99, near 10.9, viscosity 1.0, per-slot 12500.

## Update 2026-09-13: stiffness sets the timestep ceiling
Slow motion (scale 0.25) is unusable with the clock: recruits teleport in at real-time
frame rate and pile up above the digits. The stable substep scales ~1/sqrt(pressure
multiplier) and near pressure, not the multiplier, provides incompressibility, so the
multiplier can drop without changing the pool. Measured clean configurations (≥7 s sim):

| count | radius | pressure | time scale | sim ms/frame |
|---|---|---|---|---|
| 80K | 1.0 | 46.5 | 1.9 | 3.6 |
| 200K | 0.737 | 12 | 1.9 | 8.3 |
| 300K | 0.644 | 12 | 1.4 | 8.7 |
| 500K | 0.543 | 6 | 1.0 | 9.8 (+1.9 attractor) |

The page takes `?n=<count>`; `tuning_for` in the example derives all parameters.
The surface density field is splatted at radius 1.0 with kernel scale ∝ 1/count.
