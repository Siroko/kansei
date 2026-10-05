# Ray-traced reflections over a GPU triangle grid: design

From the cluster RT scout (2026-10-05): a near-field uniform grid of world triangles, rebuilt on
the GPU, fed by cluster-LOD cuts, traced for sharp and glossy reflections; the voxels light the
hits and take the rays past the grid. Hector Arellano's two posts give the idea: triangles small
enough to act like his particles (miaumiau.cat/?p=1457, with its "big triangles" kept apart),
listed per voxel and walked with a 3D DDA (miaumiau.cat/?p=1476's ray tracing bonus).

Three milestones, each a PR:

1. **The grid core** (`kansei_core::rt`): the GPU build and the traversal library.
2. **The GPU feed**: the renderer gathers the grid's triangles from its own renderables, culled
   for the grid's box by `InstanceCulling` and cut by cluster LOD at about a cell of error.
3. **`RtReflectionsEffect`**: reflections traced through the grid, lit by the voxels, shown in
   outdoor-gi (a wet road) and gi-box (a polished floor).

## 1. The grid core

A box of `dims` cells of `cell` metres follows the camera (or stays put indoors). Each rebuild:

1. **Gather.** Sources append world triangles to one buffer, a workgroup's survivors at once (a
   workgroup-wide count, one atomic). A thread transforms one triangle by its source's placement
   (a WGSL `kansei_rt_place(record, p)`, generated from an `InstanceTransform` or given), and
   drops it when its box misses the grid's. A world triangle is 64 bytes: `v0`, `e1`, `e2`, the
   surface word (flags, alpha layer), the albedo (`unorm4x8`), the source and record, and three
   packed uvs (for the alpha test).
2. **Count.** A thread a triangle (an indirect dispatch sized from the gathered count) visits the
   cells near its plane: column by column along the plane's dominant axis, only the cells the
   plane crosses in each column, each confirmed by the triangle/box separating-axis test
   (Akenine-Möller 2001), and counts them. Cells are widened by `epsilon` (world units) in every
   test, so a triangle lying on a cell face is listed in both cells (the scout lost 0.75% of hits
   there).
3. **Wide and big triangles.** One thread walking a wall's thousands of cells made the scout's
   gi-box build cost 9.4 ms. A triangle spanning more than 32 columns goes to a wide list that a
   second dispatch scatters, a workgroup a triangle, its threads a column each in turn. One
   spanning more than `big_triangle_cells` columns (room-sized) goes to a short list every ray
   tests before walking the grid, up to `big_triangle_capacity` of them (64, as in the post),
   which saves its references and their build. Meshes can also be split at load
   (`split_large_triangles`, the post's midpoint split). Measured natively on this Mac (M4 Pro)
   for gi-box's grid (100 x 96 x 100 cells of 4.75 cm, a 4.6 m room): one thread a triangle,
   24.7 ms; the wide list alone, 0.7 ms; with the big list, 0.4 ms (the scan of a million cells
   is 0.13 ms of it).
4. **Scan, fill.** An exclusive prefix sum turns the counts into each cell's start; a second
   scatter writes each triangle's index into its cells and marks their 4³ macro cells. A cell's
   list ends where the next starts.

Capacities (triangles, references) grow from a readback of what the build needed, as cluster
cuts' draw lists do, so a frame that overflows loses triangles only until the readback lands.

Buffers, for WebGPU's 8 storage buffers a stage: the trace binds a uniform and two storage
buffers (`RtGrid::bindings_wgsl(group, first)`): the triangles, and one word buffer holding the
cells' ends, the macro cells, the big list and the references. The build binds four.

`RT_GRID_WGSL`'s `kansei_rt_trace(origin, dir, t_max, flags)`: closest hit, or any hit with
`KANSEI_RT_ANY_HIT`. Triangles marked alpha-tested call the includer's
`kansei_rt_covered(layer, uv) -> bool` (`RT_OPAQUE_WGSL` defines one that keeps every hit), so a
caller samples its own foliage texture: WebGPU has no bindless textures.

## 2. The GPU feed

`Renderer::enable_rt_grid(SceneRtGridOptions)`; renderables join with `Renderable::rt`
(`RtSurface`: an albedo, an optional alpha layer) and, when instanced, `Renderable::rt_placement`
(an `InstanceTransform`, or WGSL for a material whose vertex stage does more: outdoor-gi's spruces
are widened by their tint). The grid's box becomes one more cull view, after the clipmap's:
`InstanceCulling` compacts the instances meeting it (by `rt_lod_range`, the camera's bands by
default), and a cluster view cuts clustered renderables there, orthographic, at
`cluster_error_cells` cells of error. The gather then reads, on the GPU, each source's count
(a `prepare` dispatch writes its indirect dispatch):

- a clustered renderable's cut: its draw list of (record, cluster), a workgroup an entry;
- an instanced renderable's culled records times its mesh;
- a single mesh, when its world box meets the grid's (a CPU test of one box per renderable).

It rebuilds when the box moves, when the static renderables in it change (their transform,
visibility, surface), when a cut it gathered grew, on `invalidate`, and every frame while a
`dynamic` one is in it. In outdoor-gi (`rt=1`, the road camera, 1080p, this Mac) the grid holds
21k triangles from 14 sources; a rebuild records in 0.1 ms of CPU (the scout's CPU selection took
2-2.5 ms) and costs 0.7-1.1 ms of GPU (gather 0.1-0.2, count 0.2-0.4, scan 0.1, fill 0.25-0.35).

## 3. `RtReflectionsEffect`

- Reflectivity and roughness come from the material: `GBUFFER_OUT_WGSL`'s
  `kansei_gbuffer_out_specular` stores F0 in the normal target's alpha (`1 - F0`) and the
  roughness in the albedo's (`1 - roughness / 2`), both unused until now and both averaging
  sensibly under MSAA. Materials that don't call it reflect nothing, so a scene looks as before
  until a material opts in.
- The trace runs at half or quarter resolution, one pixel of each block a frame (a Bayer order);
  glossy surfaces jitter their ray over a cone of GGX's alpha.
- Hits are lit by the voxels (clipmap or volume): the light leaving the surface there, the voxels
  as the surface cache. Rays leaving the grid continue as a narrow voxel cone, then the sky.
- The resolve, at full resolution: the frame's traced pixels upsampled by depth and normal,
  blended with the history reprojected by the surface and clamped to their range, then the lit
  colour scaled by `1 - F` plus `F` times the reflection (Schlick, lessened on rough surfaces).
- The effect holds an `RtGridHandle` (`SceneRtGrid::handle`), which follows the grid's buffers
  when they grow; `trace_grid = false` keeps the voxel cone alone for comparison.
- Off by default in both examples: `reflect=1` and a panel folder, with stats.

Measured at 1920 x 1080 in headless Chrome on this Mac (M4 Pro), kansei's profiler, variants
alternated in one page (two rounds each, within 3% of each other):

| Scene | Rays (half res) | `Rt/Trace` | `Rt/Resolve` | Voxel cone alone |
|---|---|---|---|---|
| outdoor-gi road, wet road, cards alpha-tested | 102k, 61% hit, 38 cells + 30 triangles a ray | 0.61 ms | 0.30 ms | 0.48 ms |
| the same, cards solid | 102k, 74% hit | 0.40 ms | 0.30 ms | |
| the same, quarter resolution | 25k | 0.25 ms | 0.29 ms | |
| outdoor-gi forest, everything wet | 310k, 69% hit | 2.05 ms | 0.58 ms | 1.46 ms |
| outdoor-gi driving at 12 m/s (grid rebuilt about 10 times a second) | 163k | 1.40 ms | 0.53 ms | 1.03 ms |
| gi-box floor, F0 0.3, 19k dragon as its cut (1.3k triangles in the grid) | 67k, 81% hit | 0.16 ms | 0.25 ms | 0.12 ms |
| gi-box, the 871k dragon as its cut (29k triangles, 11.7 MiB) | 67k | 0.67 ms | 0.26 ms | |
| gi-box, the 871k dragon whole (871k triangles, 100 MiB) | 67k | 9.3 ms | 0.40 ms | |

With gi-box's dragon animated the grid rebuilds every frame for 0.27 ms of GPU (gather 0.07,
count 0.05, scan 0.09, fill 0.06), where the scout's build took 9.4 ms.
