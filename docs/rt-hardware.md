# Hardware ray tracing for the RT grid

The ray tracing grid (`kansei_core::rt`, `Renderer::enable_rt_grid`) traces its rays in compute
shaders: `RT_GRID_WGSL`'s `kansei_rt_trace` walks a uniform grid of the scene's triangles round
the camera. On native, where the GPU has ray tracing hardware, the same triangles can go into an
acceleration structure instead, traced with hardware ray queries (wgpu's
`EXPERIMENTAL_RAY_QUERY`). Both backends sit behind the same `kansei_rt_trace` and return the
same `KanseiRtHit` for the same triangle ids, so the reflections, glass and hybrid GI effects
(and any pass of your own) run unchanged on either. The web keeps the grid: WebGPU has no ray
tracing.

## Which backend

`RtGridOptions::backend` picks it once, when the grid is made (`RtGrid::new`,
`Renderer::enable_rt_grid`), and `RtGrid::backend()` says which it got:

| `RtBackend` | Traces with |
|---|---|
| `Auto` (default) | hardware on Vulkan, and on Metal on Apple GPUs with ray tracing hardware (M3, A17 and later); the grid elsewhere: the web, M1 and M2 (which report ray queries without the hardware), DX12 |
| `Software` | the grid, always (for comparisons, and as the fallback) |
| `Hardware` | hardware wherever the device has ray queries, DX12 included; the grid where it hasn't |

On native, `KANSEI_RT_BACKEND=software|hardware|auto` overrides `Auto` without a rebuild. The
renderer asks for `EXPERIMENTAL_RAY_QUERY` (and the acceleration structure limits) wherever the
adapter offers it; DX12 offers it only with DXC (`WGPU_DX12_COMPILER=dynamicdxc` with
`dxcompiler.dll` on the path, or wgpu's `static-dxc` feature).

DX12 stays opt-in because NVIDIA's driver 617.42 miscompiles a ray query there: a sample traced
its coherent primary rays with another branch's flags and range (28-97% of them wrong), while
Vulkan on the same GPU and driver was exact. See [Measured](#measured) for what Kansei's own
traces did there.

## Tracing it from a pass of your own

Take the WGSL, the bindings and their layout from the grid's backend rather than naming the
software ones:

```rust
let backend = renderer.rt_grid().unwrap().grid().backend(); // or SceneRtGrid::handle().backend()
let code = rt::hoist_wgsl_enables(&format!(
    "{}\n{}\n{covered}\n{my_pass}",
    backend.wgsl(),                // RT_GRID_WGSL or RT_QUERY_WGSL
    backend.bindings_wgsl(1, 0),   // the grid's three bindings in group 1
));
let mut entries = backend.layout_entries(0, wgpu::ShaderStages::COMPUTE).to_vec();
// ... bind grid.bind_group_entries(0) (or RtGridHandle::bindings().0.entries(0))
```

- `RT_QUERY_WGSL` starts with `enable wgpu_ray_query;`, which must open the module:
  `rt::hoist_wgsl_enables` moves a module's `enable` directives to its start.
- The third binding is the grid's cells in software and its acceleration structure in
  hardware, so a stage binds as many buffers and acceleration structures as before. (Metal draws
  both from one table of 29 a stage; `rt_passes_fit_metals_buffer_slots` checks the effects.)
- After a rebuild, call `RtGrid::prepare_trace` (or `RtGridHandle::prepare_trace`) before the
  passes that trace it: the hardware backend builds its TLAS there. The renderer calls it before
  post-processing and the effects call it themselves; a standalone `RtGrid` needs it after
  `read_back`.
- `KanseiRtHit::cells` is 0 in hardware, and `tests` counts the candidates that came back to the
  shader (alpha-tested and glass triangles only).

## How the hardware backend works

The gather is the grid's own: the same pass culls, cuts cluster LOD and writes the 64-byte world
triangles every `kansei_rt_*(id)` accessor reads. The hardware backend then skips the grid's
count, scan and fill, and builds instead:

- **One BLAS of the gathered triangles**, primitive `id` being triangle `id`, so a hit names the
  same triangle the grid would. Opaque triangles go into an `OPAQUE` geometry; when alpha-tested
  or glass sources were gathered, those go into a second, non-opaque one, the only candidates
  that come back to the shader (for `kansei_rt_covered` and glass pass-through). A small pass
  (`Rt/BlasInput` in the profiler) writes each triangle's vertices from its record.
- **Sized from the last readback.** BLAS sizes are fixed on the CPU, and the gather's count is
  known only on the GPU, so the BLAS holds what the last readback said was gathered with a
  quarter more room (the grid's whole capacity before any readback, or what `RtScene` reserves);
  `SceneRtGrid` rebuilds when a gather outgrew it. Slots past the count hold filler: centimetre
  triangles under the grid's box, where no ray goes (the trace keeps to the box). Inactive (NaN)
  or degenerate triangles would be cheaper, but Metal's builds lost the real triangles among
  tens of thousands of them.
- **A TLAS of one instance**, rebuilt after each BLAS build.
- **Rebuilt only when the grid would be** (the box moved, what it holds changed, a `dynamic`
  renderable is in it), built for fast building while that happens every few frames and once
  more for fast tracing when the grid has rested 30 frames. A static scene builds once.

Metal needs two workarounds, both from wgpu 30's Metal backend having no acceleration structure
barrier (gfx-rs/wgpu#9215, fix open as #9645) and Metal tracking no hazard through a TLAS's
instances:

- The CPU waits for the BLAS's submit to finish before building the TLAS. Without it, rays
  missed most of the scene even with the two builds in submits of their own.
- Two BLASes alternate, so a rebuild never overwrites the BLAS the frames before are still
  tracing.

The wait costs a static scene nothing, but a grid rebuilt every frame pays a CPU round trip each
time (see *room, rebuilt every frame* below). Vulkan and DX12 need neither.

## Measured

`cargo run --release -p kansei-native --example rt_backends` renders headless scenes set up as
the RT examples set theirs (they run on the web only), the two backends alternating in blocks of
frames, at 1920 x 1080: gi-box's room with a dense mesh where its dragon stands, the room
rebuilt every frame, outdoor-gi's grid round a camera driving through a field of blocks, and
glass-particles' room of glass beads rebuilt every frame. Frame times are wall time with the CPU
waiting for each frame's GPU work; pass times are the profiler's.

Software → hardware, 2026-10-27 (NVIDIA driver 617.42; the full tables are in the pull request
that added the backend):

| 1920 x 1080 | Apple M4 Pro, Metal | RTX 5090, Vulkan |
|---|---|---|
| **room** (a 19k-triangle mesh), frame | 7.96 → 6.20 ms | 2.18 → 1.05 ms |
| GI trace (`RtGi/Trace`) | 3.00 → 1.37 ms | 0.63 → 0.16 ms |
| reflections (`Rt/Trace`) | 0.55 → 0.26 ms | 0.40 → 0.06 ms |
| glass (`Rt/Glass`) | 0.94 → 1.00 ms | 0.44 → 0.12 ms |
| **room, rebuilt every frame**, frame | 8.21 → 7.78 ms | 2.53 → 1.57 ms |
| **room with a 1.3 M-triangle mesh**, frame | 66.4 → 6.2 ms | 56.2 → 1.25 ms |
| GI trace | 36.8 → 1.38 ms | 21.3 → 0.16 ms |
| **the same, rebuilt every frame**, frame | 68.7 → 67.3 ms | 58.6 → 4.7 ms |
| **drive** (the box stepping along), frame | 6.15 → 5.19 ms | 2.08 → 1.90 ms |
| GI trace | 1.49 → 0.85 ms | 0.19 → 0.06 ms |
| **particles** (400 glass beads, rebuilt every frame), frame | 65.9 → 46.0 ms | 19.0 → 9.1 ms |
| glass | 60.8 → 35.9 ms | 12.5 → 2.9 ms |

- In the room the traces run about twice as fast on the M4 Pro and four to six times on the
  5090. Dense meshes, which crowd the grid's cells, gain the most: the GI trace of the
  1.3 M-triangle room, 27 and 130 times.
- A hardware rebuild costs more than the grid's on Metal: a fast BLAS build of the room's 39k
  slots and the CPU's wait add about 1.6 ms to a frame, 61 ms for 1.7 M triangles. On the 5090
  the same builds take about 0.5 and 3.5 ms. Static scenes build once.
- Metal's glass pass gains nothing in the room, unexplained so far (on the 5090 it runs 3.8
  times as fast); its rays meet the glass ball in the BLAS's non-opaque geometry, whose
  candidates come back to the shader.
- DX12 (`RtBackend::Hardware`, DXC from the Windows SDK) passed the parity tests on the 5090 as
  Vulkan did: Kansei's traces don't take the shape of the miscompiled one. It stays opt-in until
  the driver bug is understood.

The path tracer (`pathtracer`) has its own BVH and doesn't use the grid; moving it to hardware
ray queries is later work, as is an instanced TLAS over per-mesh BLASes that traces the whole
scene rather than the box round the camera.

## Tests

- `cargo test -p kansei-core --lib rt::` validates every trace module for both backends with naga
  (ray queries for the hardware one), the hardware BLAS input's uniform size, the `enable`
  hoisting and the effects' Metal buffer slots.
- `cargo test -p kansei-core --test rt_grid_gpu` runs `hardware_rays_find_what_the_reference_finds`
  (the hardware backend against the CPU reference, as the grid's own test) and
  `hardware_hits_match_the_grid` (both backends ray for ray on the same scene with a glass ball,
  for every trace flag: hits, distances and the triangles' records), where the adapter has ray
  queries.
- The renderer's RT tests (`rt_reflections_gpu`, `rt_diffuse_gi_gpu`, `rt_grid_scene_gpu`) run on
  whatever `Auto` picks; `KANSEI_RT_BACKEND=software` runs them on the grid.
