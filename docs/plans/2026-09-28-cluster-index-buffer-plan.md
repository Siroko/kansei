# Cluster LOD: a compacted index buffer — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A cluster cut draws only its clusters' own triangles, with vertices reused inside a cluster, through one indexed indirect draw per cut.

Today each cut is a non-indexed draw of `3 × max_triangles` vertices per cluster: every corner is its own vertex, and every cluster is padded to the largest. On the film's trees, whose vertex shader is heavy (wind), that made cards 2–4× slower than the LODs at equal triangles (`data/kansei-culling/cluster-lod-cost.md`, firstmate).

**Architecture:**

**Indices, written by the cull.**
- A cluster that passes the cut takes a draw-list entry (as today, `claimed`) and reserves its triangles in the cut's index buffer (`atomicAdd` on the triangles claimed).
- When both fit their capacities, the thread writes the entry `(record, cluster)` and 3 indices per triangle, each `entry << 8 | local vertex`. A vertex shared by a cluster's triangles has one index, so the post-transform cache reuses it.
- When either capacity is full, the part of its reserved range below the index capacity is written as degenerate triangles (index 0 three times). The index range the draw reads is then always fully written: `3 × min(triangles claimed, triangle capacity)`.

**The draw's words** (48 bytes, `DRAW_ARGS_BYTES`):
- 0–4: `DrawIndexedIndirect` (index count, instance count 1, first index, base vertex, first instance);
- 5: the visible instances;
- 6: the clusters claimed;
- 7: the clusters listed (drawn);
- 8: the triangles claimed;
- 9: the triangles listed;
- 10–11: pad.

**The vertex stage** decodes `vertex_index`: its entry (`>> 8`), and its local vertex (`& 255`, since clusters have at most 256 vertices). It reads that entry's record and the cluster's vertex.

**Sizing.** The index buffer (`INDEX | STORAGE`) is sized like the draw list, from the triangles-claimed readback (word 8):
- 262,144 triangles at first (3 MB);
- grown at once to 1.5× the need, as a power of two;
- shrunk after 64 readbacks at a quarter or less;
- at most the device's storage binding size.

The feedback reads words 6 and 8.

**Stats:** clusters come from word 7, triangles from word 9. The stats readback copies each cut's 48 bytes.

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md` §2 and Risks ("a compacted index buffer … would recover vertex reuse and drop the padding"); firstmate 037 (the user's approval).

## Global Constraints
- The cut and the image are unchanged: every τ = 0 image test (camera, shadows, sky, reflections, velocity) still matches the mesh path.
- No index past what the draw reads is ever unwritten.
- `ClusterLod::capacity` stays the hard maximum on clusters.
- PRs go against `development`, with no AI attribution. The film's A/B/C is re-measured after the PR, headless, with contended rounds discarded.

## Review Focus
1. **Stale or garbage indices inside the drawn range** (a cluster reserving triangles but not listed): degenerate fill. Pinned by the overflow test.
2. **An index's entry pointing at an unwritten draw-list slot:** only listed entries' indices are real; degenerate ones point at entry 0's vertex 0 with no area.
3. **Clusters over 256 vertices:** `max_vertices` ≤ 256 is already the builder's limit, since local indices are bytes.
4. **Render bundles:** the camera's bundles record `set_index_buffer` plus `draw_indexed_indirect`. A grown index buffer invalidates them, as a grown draw list does.

### Task 1: The cull writes indices; the vertex stage reads them
- [ ] Failing test `cluster_draws_are_indexed_without_padding`: after a cull, the index count is 3 × the listed clusters' triangles, and the decoded indices give exactly each listed cluster's triangles (`ClusterMesh::triangles`), entry by entry.
- [ ] Failing test `an_overflowing_cut_draws_only_whole_written_triangles`: a triangle capacity smaller than the cut. Every index in the drawn range is either a listed entry's real vertex or a degenerate triangle.
- [ ] Implement the WGSL, the buffers, the args layout, the vertex stage, and the renderer's draws (`CameraClusterDraw`, `draw_cut`). Update the tests' word offsets.
- [ ] The whole suite is green; the τ = 0 image tests stay green.

### Task 2: Sizing and stats
- [ ] Failing test: the index buffer grows to the cut's triangles within 8 frames and draws all of them.
- [ ] Implement feedback of word 8, `Cut::sized` for triangles, and the stats (48-byte entries).
- [ ] Suite green; commit.

### Task 3: Docs, review, PR; the film's re-measure
