# Cluster LOD: draw lists sized by need — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Each cut's draw list (one per renderable and view, M3) is sized to what that cut actually draws, not to the worst case. For the film the worst case is instances × clusters, about 13 MB per view across roughly 8 views.

**Architecture:**
- **Feedback.** The cull already counts every cluster a cut claims, including those past its capacity (word 5 of its draw). After the cull, `ClusterCulling` copies each cut's word into one staging buffer and maps it asynchronously, with at most one copy in flight, as `StatsReadback` does. A few frames later each cut learns what it needed. Cuts are keyed by (`ClusterGpu` id, view): a process-wide counter gives each `ClusterGpu` an id, so scene reordering can't misroute a result.
- **Sizing policy** (in `Cut`). `max` is today's bound (`ClusterLod::capacity`, or every cluster of every instance, at most 4M).
  - **Before any feedback:** `min(max, INITIAL_DRAWN)`, with `INITIAL_DRAWN` = 65,536 entries (512 KB).
  - **Target:** `clamp(next_power_of_two(needed × 3/2), MIN_DRAWN, max)`, with `MIN_DRAWN` = 1,024.
  - **Grow at once** when the target exceeds the capacity (an overflowing cut always does).
  - **Shrink** to the target only after `SHRINK_AFTER` = 64 readbacks in a row with the target at most a quarter of the capacity. This hysteresis keeps a cut from thrashing, and keeps the largest recent need through shot changes.
- **What overflow costs.** When a cut suddenly needs more than 1.5× what it recently needed, the clusters past its capacity aren't drawn until the readback lands, 2–3 frames later. At load that's the first frames. After that it takes a jump past the headroom and past the largest need of the last 64 readbacks.
- **Growth invalidates the camera's bundles**, as today (the draw list is a new buffer).

**Spec:** `docs/plans/2026-09-28-cluster-lod-design.md` §2 "As built (M3)" (memory), and the M3 review's finding 3.

## Global Constraints
- The shader's cap stays `params.capacity`: a cut never writes past its buffer.
- An explicit `ClusterLod::capacity` stays a hard maximum.
- No readback blocks the frame (no `Maintain::Wait` outside tests).
- PRs go against `development`, with no AI attribution.

## Review Focus
1. **A result routed to the wrong cut** (renderables removed or reordered, views appearing and disappearing): ids, not scene indices.
2. **Thrash** (a cut whose need hovers near a boundary reallocating every frame): hysteresis. Pinned by the shrink test, whose need drops only once.
3. **An explicit small `capacity`** (the tests use 5): target ≤ max always, and `MIN_DRAWN` never raises past max.
4. **Stale feedback after a cut grows**: a reading from before the growth that says "overflowed" must not shrink it. It grows only, and shrinking needs 64 readings in a row.

### Task 1: Feedback and sizing (gpu.rs, gpu_tests.rs)
- [ ] Failing tests:
  - `draw_lists_grow_to_what_the_cut_needs`: 2,000 rock instances at τ = 0 need more than 65,536 entries.
    - The first frames overflow: claimed > drawn.
    - Within 8 frames the list has grown, drawn == claimed, and the capacity is at most next_power_of_two(1.5 × claimed).
  - `draw_lists_shrink_when_the_cut_needs_less`: after that, the view moves far away (few clusters).
    - The capacity holds for fewer than 64 readbacks.
    - It has shrunk within 64 + 8 frames, and it is still at least the need.
  - `an_explicit_capacity_is_a_hard_maximum`: capacity 5 stays 5.
- [ ] Implement:
  - `ClusterFeedback` in `ClusterCulling`;
  - the id on `ClusterGpu`;
  - `Cut::sized(max, needed)`;
  - `bind` sizes the list, and `encode` records the feedback.
- [ ] Suite green; commit.

### Task 2: Docs, review, PR
- [ ] Update design §2 "As built (M3)" memory note and the `ClusterLod::capacity` doc.
- [ ] Final review, fix pass, and a PR against `development`.
