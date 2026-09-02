# Fluid Clock Example (Plan 3 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `fluid_clock` WASM example — a water clock where fluid particles form the current time (`HH:MM:SS`) via the Plan-2 attractor, with digits reassembling from the fluid as time changes, plus a Tweakpane tuning UI and per-second audio beeps.

**Architecture:** Fork the existing fluid WASM example. Add a `GlyphAttractor` (Plan 2) alongside the sim, driven by a new **TDD'd core clock module** (`ClockState` + nearest-particle retag). Each frame: step the SPH solver, run the attractor pass (velocities only), and on a digit change re-tag particles (release the changed slot's particles → recruit the nearest untagged ones). The existing marching-cubes surface renders **all** particles unchanged, so glyphs and pool read as one liquid.

**Tech Stack:** Rust, wgpu/WGSL, wasm-bindgen, web-sys (WebAudio + Date), Tweakpane. Consumes `kansei_core::sdf` (Plan 1) and the `GlyphAttractor` (Plan 2).

**Scope note:** Plan 3 of 3. Plans 1–2 (merged on `feat/fluid-clock-sdf`) built the SDF module and the GPU attractor (native-verified PASS). This plan is the integration + UX. Part A (clock/retag logic) is strict TDD in `cargo test`; Part B (the WASM example) is grounded integration verified in the browser (no headless test for WASM+GPU+DOM).

---

## Verification Strategy (read first)

- **Part A (Tasks 1–3):** pure-Rust clock + retag logic in `kansei-core`, strict TDD with `cargo test`. This is the novel, bug-prone logic and it is fully unit-testable with synthetic positions.
- **Part B (Tasks 4–8):** the WASM example. WASM + GPU + DOM cannot run under `cargo test`, so verification is: (a) `cargo build --target wasm32-unknown-unknown` compiles; (b) `wasm-pack build` succeeds; (c) served in a WebGPU browser, the digits visibly form from fluid, change over time, and beep. The engineer must state which of these were confirmed and, if the browser can't be driven, hand the visual check to the user.

---

## Design Decisions

- **Retag on the CPU with a position shadow.** Nearest-particle selection needs positions, which live on the GPU. Maintain a CPU **position shadow** updated by an async GPU→CPU readback (WASM `map_async` is non-blocking; the shadow lags 1–2 frames, which is fine for selecting recruits). Retags are infrequent (≤ a few per second), so cost is negligible.
- **Budget-constant swap** (from brainstorming): a fixed `per_slot_count` particles are attracted per digit slot. On a digit change: set that slot's particles back to `-1` (ordinary fluid — they flow away naturally), then recruit the `per_slot_count` **nearest untagged** particles to the slot and tag them. Colons never change.
- **Attractor is additive, unchanged from Plan 2.** It runs in its own encoder after the solver step, modifying velocities only. The marching-cubes surface already renders all particles, so no rendering changes are needed for glyphs to appear as liquid.
- **Audio is inlined WebAudio** (no Tone.js CDN dependency): a short sine `OscillatorNode` per second, pitch tiers for seconds / minute-rollover / hour-rollover.

---

## File Structure

- Create `rust/kansei-core/src/simulations/fluid/clock.rs` — `ClockState`, `retag_on_change`, `initial_tags` (pure logic, TDD).
- Modify `rust/kansei-core/src/simulations/fluid/mod.rs` — declare + re-export the clock types.
- Create `rust/kansei-wasm/examples/fluid_clock/` — forked example crate (`Cargo.toml`, `src/lib.rs`, `www/index.html`, `www/assets/`).

---

# PART A — Clock & retag logic (TDD in kansei-core)

## Task 1: `ClockState` — wall-clock digits + change detection

**Files:**
- Create: `rust/kansei-core/src/simulations/fluid/clock.rs`
- Modify: `rust/kansei-core/src/simulations/fluid/mod.rs`

- [ ] **Step 1: Declare the module + re-export**

In `rust/kansei-core/src/simulations/fluid/mod.rs`, add after the existing `mod` lines:

```rust
mod clock;
```

and after the existing `pub use` lines:

```rust
pub use clock::{ClockState, retag_on_change, initial_tags};
```

- [ ] **Step 2: Write the failing test**

Create `rust/kansei-core/src/simulations/fluid/clock.rs`:

```rust
//! Clock logic for the fluid clock example: map wall-clock time to the 8 glyph
//! slots, detect which digit slots changed, and re-tag particles (release the
//! changed slot's particles, recruit the nearest untagged ones). Pure CPU logic.

use crate::simulations::fluid::{SlotLayout, NUM_SLOTS};

/// Tracks the currently-displayed time so digit changes can be detected.
pub struct ClockState {
    /// Glyph id currently shown in each slot (`-1` = uninitialized).
    current: [i32; NUM_SLOTS],
}

impl ClockState {
    pub fn new() -> Self {
        ClockState { current: [-1; NUM_SLOTS] }
    }

    /// Apply `h:m:s` to `layout` and return the slot indices whose glyph changed
    /// since the last call (digit slots only; colons never change).
    pub fn update(&mut self, layout: &mut SlotLayout, h: u32, m: u32, s: u32) -> Vec<usize> {
        layout.set_time(h, m, s);
        let mut changed = Vec::new();
        for i in 0..NUM_SLOTS {
            let g = layout.slots[i].glyph_id;
            if g != self.current[i] {
                // Only report digit slots (glyph_id 0..=9); colon slots (id 10) are static.
                if g >= 0 && g <= 9 {
                    changed.push(i);
                }
                self.current[i] = g;
            }
        }
        changed
    }
}

impl Default for ClockState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulations::fluid::SlotLayout;

    #[test]
    fn first_update_reports_all_six_digit_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        let changed = clock.update(&mut layout, 12, 34, 56);
        // 6 digit slots (0,1,3,4,6,7); colons excluded.
        assert_eq!(changed, vec![0, 1, 3, 4, 6, 7]);
    }

    #[test]
    fn only_changed_digit_is_reported() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 56);
        // 12:34:56 -> 12:34:57 : only the seconds-ones slot (index 7) changes.
        let changed = clock.update(&mut layout, 12, 34, 57);
        assert_eq!(changed, vec![7]);
    }

    #[test]
    fn minute_rollover_changes_multiple_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 59);
        // 12:34:59 -> 12:35:00 : minute-ones (4), seconds-tens (6), seconds-ones (7).
        let mut changed = clock.update(&mut layout, 12, 35, 0);
        changed.sort();
        assert_eq!(changed, vec![4, 6, 7]);
    }

    #[test]
    fn no_change_reports_empty() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 1, 2, 3);
        assert_eq!(clock.update(&mut layout, 1, 2, 3), Vec::<usize>::new());
    }
}
```

- [ ] **Step 3: Run to verify fail then pass**

Run (from `/Users/felixmartinez/Documents/dev/kansei/rust`): `cargo test -p kansei-core clock`
Expected: compiles and PASSES (4 tests). If `mod.rs` re-exports `retag_on_change`/`initial_tags` (not yet defined), temporarily narrow the `pub use` to only `ClockState` and re-widen in Task 3.

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-core/src/simulations/fluid/clock.rs rust/kansei-core/src/simulations/fluid/mod.rs
git commit -m "feat(fluid): ClockState maps time to slots and detects digit changes"
```

---

## Task 2: `initial_tags` — assign the starting attracted subset

**Files:**
- Modify: `rust/kansei-core/src/simulations/fluid/clock.rs`

- [ ] **Step 1: Write the failing test**

Add inside `clock.rs`'s `mod tests`:

```rust
    fn grid_positions(n: usize) -> Vec<f32> {
        // n particles on a line from x=0..n, y=z=0; each is [x,y,z,1].
        let mut v = Vec::with_capacity(n * 4);
        for i in 0..n {
            v.extend_from_slice(&[i as f32, 0.0, 0.0, 1.0]);
        }
        v
    }

    #[test]
    fn initial_tags_assigns_per_slot_count_to_active_digit_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        layout.set_time(11, 11, 11); // all digit slots show '1'; colons show ':'
        let positions = grid_positions(1000);
        let tags = initial_tags(&positions, &layout, 10);
        assert_eq!(tags.len(), 1000);
        // 8 slots × 10 recruited = 80 tagged (colons are attractors too), rest -1.
        let tagged = tags.iter().filter(|&&t| t >= 0).count();
        assert_eq!(tagged, 80);
        // Every tagged particle references a slot with glyph_id >= 0.
        for &t in &tags {
            if t >= 0 {
                assert!(layout.slots[t as usize].glyph_id >= 0);
            }
        }
    }

    #[test]
    fn initial_tags_leaves_disabled_slots_unfilled() {
        // A layout with no time set: digit slots are -1 (disabled), colons enabled.
        let layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let positions = grid_positions(1000);
        let tags = initial_tags(&positions, &layout, 10);
        // Only the 2 colon slots are active → 20 tagged.
        assert_eq!(tags.iter().filter(|&&t| t >= 0).count(), 20);
    }
```

- [ ] **Step 2: Run to verify fail**

Run: `cargo test -p kansei-core initial_tags`
Expected: FAIL — `initial_tags` not found.

- [ ] **Step 3: Implement `initial_tags`**

Add to `clock.rs` (above the `#[cfg(test)]` block):

```rust
/// Squared distance from particle `i` (in a flat `[x,y,z,w]` slice) to `p`.
fn dist2(positions: &[f32], i: usize, p: [f32; 3]) -> f32 {
    let x = positions[i * 4] - p[0];
    let y = positions[i * 4 + 1] - p[1];
    let z = positions[i * 4 + 2] - p[2];
    x * x + y * y + z * z
}

/// World-space center of a slot's box.
fn slot_center(layout: &SlotLayout, slot: usize) -> [f32; 3] {
    let s = &layout.slots[slot];
    [
        s.world_min[0] + 0.5 * s.world_size[0],
        s.world_min[1] + 0.5 * s.world_size[1],
        s.world_min[2] + 0.5 * s.world_size[2],
    ]
}

/// Recruit the `count` nearest currently-untagged particles to `slot`'s center,
/// tagging them with `slot`. Mutates `tags` in place.
fn recruit_nearest(positions: &[f32], tags: &mut [i32], layout: &SlotLayout, slot: usize, count: usize) {
    let center = slot_center(layout, slot);
    // Candidate = untagged particle indices, sorted by distance to the slot center.
    let mut candidates: Vec<usize> = (0..tags.len()).filter(|&i| tags[i] < 0).collect();
    candidates.sort_by(|&a, &b| {
        dist2(positions, a, center)
            .partial_cmp(&dist2(positions, b, center))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    for &i in candidates.iter().take(count) {
        tags[i] = slot as i32;
    }
}

/// Build the initial per-particle tag array: for each ACTIVE slot (glyph_id >= 0),
/// recruit `per_slot_count` nearest untagged particles. Returns `positions.len()/4`
/// tags (`-1` = ordinary fluid).
pub fn initial_tags(positions: &[f32], layout: &SlotLayout, per_slot_count: usize) -> Vec<i32> {
    let n = positions.len() / 4;
    let mut tags = vec![-1i32; n];
    for slot in 0..NUM_SLOTS {
        if layout.slots[slot].glyph_id >= 0 {
            recruit_nearest(positions, &mut tags, layout, slot, per_slot_count);
        }
    }
    tags
}
```

- [ ] **Step 4: Run to verify pass**

Run: `cargo test -p kansei-core initial_tags`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/simulations/fluid/clock.rs
git commit -m "feat(fluid): initial_tags recruits nearest particles per active slot"
```

---

## Task 3: `retag_on_change` — release + recruit on digit change

**Files:**
- Modify: `rust/kansei-core/src/simulations/fluid/clock.rs`

- [ ] **Step 1: Write the failing test**

Add inside `mod tests`:

```rust
    #[test]
    fn retag_releases_old_and_recruits_new() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        layout.set_time(11, 11, 11);
        let positions = grid_positions(1000);
        let mut tags = initial_tags(&positions, &layout, 10);
        let before_slot0 = tags.iter().filter(|&&t| t == 0).count();
        assert_eq!(before_slot0, 10);

        // Simulate slot 0's digit changing: retag slot 0.
        retag_on_change(&positions, &mut tags, &layout, &[0], 10);

        // Slot 0 still has exactly per_slot_count particles (budget constant)...
        assert_eq!(tags.iter().filter(|&&t| t == 0).count(), 10);
        // ...and the total tagged count is unchanged.
        assert_eq!(tags.iter().filter(|&&t| t >= 0).count(), 80);
    }

    #[test]
    fn retag_recruits_only_untagged_particles() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        layout.set_time(11, 11, 11);
        let positions = grid_positions(1000);
        let mut tags = initial_tags(&positions, &layout, 10);
        retag_on_change(&positions, &mut tags, &layout, &[3], 10);
        // No particle is tagged to two slots (each index has exactly one tag value);
        // and slots other than 3 keep their counts.
        assert_eq!(tags.iter().filter(|&&t| t == 1).count(), 10); // slot 1 untouched
        assert_eq!(tags.iter().filter(|&&t| t == 3).count(), 10);
    }
```

- [ ] **Step 2: Run to verify fail**

Run: `cargo test -p kansei-core retag`
Expected: FAIL — `retag_on_change` not found.

- [ ] **Step 3: Implement `retag_on_change`**

Add to `clock.rs` (above the `#[cfg(test)]` block):

```rust
/// Re-tag particles for slots whose digit just changed: first RELEASE every
/// particle tagged to a changed slot (set to `-1`, so it rejoins the fluid),
/// then RECRUIT `per_slot_count` nearest untagged particles to each changed
/// slot. Keeps the attracted-particle budget constant while rotating membership.
pub fn retag_on_change(
    positions: &[f32],
    tags: &mut [i32],
    layout: &SlotLayout,
    changed_slots: &[usize],
    per_slot_count: usize,
) {
    // Release all changed slots first, so their old particles become recruitable
    // candidates for the (possibly different) new digits.
    for &slot in changed_slots {
        for t in tags.iter_mut() {
            if *t == slot as i32 {
                *t = -1;
            }
        }
    }
    // Recruit nearest untagged particles for each changed, still-active slot.
    for &slot in changed_slots {
        if layout.slots[slot].glyph_id >= 0 {
            recruit_nearest(positions, tags, layout, slot, per_slot_count);
        }
    }
}
```

- [ ] **Step 4: Run to verify pass + full suite**

Run: `cargo test -p kansei-core clock`
Expected: PASS (all clock tests). Then `cargo test -p kansei-core` — fully green.

- [ ] **Step 5: Re-widen mod.rs re-exports (if narrowed in Task 1)**

Ensure `rust/kansei-core/src/simulations/fluid/mod.rs` has:

```rust
pub use clock::{ClockState, retag_on_change, initial_tags};
```

Run `cargo build -p kansei-core` — clean.

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/simulations/fluid/clock.rs rust/kansei-core/src/simulations/fluid/mod.rs
git commit -m "feat(fluid): retag_on_change releases and recruits on digit change"
```

---

# PART B — The WASM example (integration + UX)

## Task 4: Fork the fluid example into `fluid_clock`

**Files:**
- Create: `rust/kansei-wasm/examples/fluid_clock/` (copy of `rust/kansei-wasm/examples/fluid/`)

- [ ] **Step 1: Copy the example directory**

Run (from `/Users/felixmartinez/Documents/dev/kansei`):

```bash
cp -R rust/kansei-wasm/examples/fluid rust/kansei-wasm/examples/fluid_clock
rm -rf rust/kansei-wasm/examples/fluid_clock/pkg rust/kansei-wasm/examples/fluid_clock/target
```

- [ ] **Step 2: Rename the crate**

In `rust/kansei-wasm/examples/fluid_clock/Cargo.toml`, change the package name:

```toml
[package]
name = "kansei-wasm-fluid-clock"
```

Ensure `web-sys` features include what the audio + time code needs — add these to the existing `web-sys` `features` list if not present: `"AudioContext"`, `"AudioDestinationNode"`, `"OscillatorNode"`, `"GainNode"`, `"AudioParam"`.

- [ ] **Step 3: Confirm the fork builds unchanged**

Run (from `/Users/felixmartinez/Documents/dev/kansei/rust/kansei-wasm/examples/fluid_clock`):
`cargo build --target wasm32-unknown-unknown`
Expected: compiles (it is a byte copy of a working example with a new name).

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-wasm/examples/fluid_clock
git commit -m "chore(fluid-clock): fork the fluid wasm example as fluid_clock"
```

---

## Task 5: Add the attractor + position readback to the example

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

This task wires the attractor and a CPU position shadow into the example's `State` and render loop. The example's sim lives in a `FluidSurfaceEffect` accessed via the volume's effect (`fse.sim`, with `fse.sim.positions_buffer()` / `fse.sim.velocities_buffer()` public accessors from Plans 1–2).

- [ ] **Step 1: Add imports**

At the top of `lib.rs`, extend the `kansei_core::simulations::fluid` import to include the new types:

```rust
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{
    GlyphAttractor, SlotLayout, ClockState, initial_tags, retag_on_change,
};
```

- [ ] **Step 2: Add fields to `State`**

Add these fields to the `State` struct (near the other sim-related fields):

```rust
    attractor: GlyphAttractor,
    slot_layout: SlotLayout,
    clock: ClockState,
    tags: Vec<i32>,
    per_slot_count: usize,
    // CPU position shadow for retag (updated by async readback).
    pos_shadow: Vec<f32>,
    pos_staging: wgpu::Buffer,
    pos_shadow_ready: std::rc::Rc<std::cell::RefCell<Option<Vec<f32>>>>,
    // Attractor tuning.
    attr_stiffness: f32,
    attr_max_speed: f32,
    attr_basin: f32,
    // Audio + last-second for beep triggering.
    audio_ctx: Option<web_sys::AudioContext>,
    last_second: i32,
```

- [ ] **Step 3: Build the glyph set, attractor, layout, initial tags after the sim is created**

Include the font asset and, right after the `FluidSimulation` is created (before it is moved into the effect), build the attractor. IMPORTANT: `GlyphAttractor::new` needs `&renderer` and the particle `count`; construct it here. Add near the top-level shaders:

```rust
const FONT: &[u8] = include_bytes!("../../../kansei-core/tests/fixtures/L10-medium.arfont");
```

After the sim is created and `count` is known (the example already has a `count` for particles), add:

```rust
    let font = FontAtlas::parse(FONT).expect("parse font");
    let glyph_set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
    let attractor = GlyphAttractor::new(&renderer, &glyph_set, count as u32);

    // Clock slots sized to the sim's world; tune `cell` to the domain.
    let mut slot_layout = SlotLayout::hh_mm_ss(4.0, 1.5);
    let mut clock = ClockState::new();
    // Seed with current time so digit slots are active at startup.
    let (h0, m0, s0) = now_hms();
    let _ = clock.update(&mut slot_layout, h0, m0, s0);

    let per_slot_count = (count / 20).max(1) as usize; // ~5% of particles per slot
    // Initial tags need positions; the sim was seeded from `positions` (the CPU
    // array used to create it) — reuse that array as the initial shadow.
    let tags = initial_tags(&positions, &slot_layout, per_slot_count);
    attractor.set_tags(&tags);
    attractor.set_slots(&slot_layout);

    let pos_staging = renderer.device().create_buffer(&wgpu::BufferDescriptor {
        label: Some("fluid_clock/pos_readback"),
        size: (count as u64) * 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
```

Where `positions` is the flat `[x,y,z,1]` array the example already built to seed the sim (reuse it; do not rebuild). `now_hms()` is a helper added in Step 6.

- [ ] **Step 4: Initialize the new `State` fields** where `State { ... }` is constructed:

```rust
        attractor,
        slot_layout,
        clock,
        tags,
        per_slot_count,
        pos_shadow: positions.clone(),
        pos_staging,
        pos_shadow_ready: std::rc::Rc::new(std::cell::RefCell::new(None)),
        attr_stiffness: 40.0,
        attr_max_speed: 20.0,
        attr_basin: 5.0,
        audio_ctx: None,
        last_second: -1,
```

(Make sure `positions` is still in scope — clone it before it is consumed by `FluidSimulation::new` if necessary, i.e. move the `let positions = ...` clone earlier.)

- [ ] **Step 5: In `render_frame`, after the sim steps, run the attractor + retag**

Find where the sim steps each frame (the `while self.sim_accumulator >= step_dt ... fse.step_simulation(...)` loop). AFTER that loop (still inside `render_frame`, before the render/`self.renderer.render(...)` call), add:

```rust
        // ── Clock + attractor ──────────────────────────────────────────
        let (h, m, s) = now_hms();
        let changed = self.clock.update(&mut self.slot_layout, h, m, s);

        // Pull the latest async position readback into the shadow, if ready.
        if let Some(latest) = self.pos_shadow_ready.borrow_mut().take() {
            self.pos_shadow = latest;
        }

        if !changed.is_empty() {
            retag_on_change(&self.pos_shadow, &mut self.tags, &self.slot_layout, &changed, self.per_slot_count);
            self.attractor.set_tags(&self.tags);
            self.attractor.set_slots(&self.slot_layout);
            self.beep_for_change(h, m, s); // Task 7
        } else if s as i32 != self.last_second {
            self.beep_for_change(h, m, s); // per-second tick even without a digit-position change is rare; keep simple
        }
        self.last_second = s as i32;

        self.attractor.set_params(
            (1.0f32 / 60.0),
            self.attr_stiffness,
            self.attr_max_speed,
            self.attr_basin,
        );

        // Dispatch the attractor in its own encoder (velocities only).
        {
            let fse = self.volume.effects[0].as_any().downcast_ref::<FluidSurfaceEffect>().unwrap();
            let pos_buf = fse.sim.positions_buffer().unwrap();
            let vel_buf = fse.sim.velocities_buffer().unwrap();
            let mut enc = self.renderer.device().create_command_encoder(&Default::default());
            self.attractor.dispatch(&mut enc, pos_buf, vel_buf);
            self.renderer.queue().submit(std::iter::once(enc.finish()));

            // Kick off an async position readback to refresh the shadow (non-blocking).
            let mut enc2 = self.renderer.device().create_command_encoder(&Default::default());
            enc2.copy_buffer_to_buffer(pos_buf, 0, &self.pos_staging, 0, (self.count as u64) * 16);
            self.renderer.queue().submit(std::iter::once(enc2.finish()));
            let ready = self.pos_shadow_ready.clone();
            let n = (self.count as usize) * 4;
            let staging = self.pos_staging.slice(..);
            staging.map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    // The mapped range is read in the poll below; here we just flag readiness
                    // by storing nothing — actual copy happens in the closure body via a second borrow.
                }
            });
            // NOTE: see Step 5b for the correct non-blocking map handling.
            let _ = (ready, n);
        }
```

- [ ] **Step 5b: Correct the async readback (important)**

The `map_async` callback cannot borrow the buffer slice. Use this pattern instead: keep the staging buffer permanently, and each frame (1) if not currently mapped, issue the copy + `map_async` storing readiness in the `Rc<RefCell<Option<Vec<f32>>>>`; (2) in the callback, you cannot read the slice, so instead poll and read on the NEXT frame. Simplest robust approach — replace the readback block in Step 5 with:

```rust
        // Refresh position shadow via a mapped staging buffer, non-blocking.
        // On WASM, poll(Poll) processes completed maps; read + unmap when ready.
        self.renderer.device().poll(wgpu::Maintain::Poll);
        // (The map_async below is issued once per frame; the closure copies out
        // the data into pos_shadow_ready, then we unmap.)
        {
            let ready = self.pos_shadow_ready.clone();
            let staging = std::rc::Rc::new(self.pos_staging.slice(..));
            let staging2 = staging.clone();
            staging.map_async(wgpu::MapMode::Read, move |res| {
                if res.is_ok() {
                    let data = staging2.get_mapped_range();
                    let v: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
                    drop(data);
                    *ready.borrow_mut() = Some(v);
                }
            });
        }
        self.pos_staging.unmap();
```

> **Implementer note:** WASM readback timing is fiddly. If the `Rc<Slice>` clone pattern does not satisfy wgpu's lifetimes, fall back to the documented wgpu WASM readback example (staging buffer + `map_async` + `unmap` on the next frame). The retag only needs positions that are 1–3 frames stale, so exact timing is not critical — correctness of the pattern matters more than freshness. Verify in the browser that retags visibly recruit *nearby* particles (not random ones); if recruits look random, the shadow is not updating — debug the map/unmap ordering.

- [ ] **Step 6: Add the `now_hms` and `FluidSurfaceEffect` imports/helpers**

Add a helper (WASM local time via JS `Date`):

```rust
fn now_hms() -> (u32, u32, u32) {
    let d = js_sys::Date::new_0();
    (d.get_hours(), d.get_minutes(), d.get_seconds())
}
```

Ensure `FluidSurfaceEffect` is imported (the example already downcasts to it elsewhere, so the import exists — reuse it). Add `js-sys` to `Cargo.toml` deps if not present (it already is in the fluid example).

- [ ] **Step 7: Build**

Run: `cargo build --target wasm32-unknown-unknown` (from the example dir)
Expected: compiles. Fix borrow/lifetime issues around the effect downcast and the readback closure as needed (these are the likely friction points; the `as_any().downcast_ref::<FluidSurfaceEffect>()` pattern is already used elsewhere in the file — mirror it).

- [ ] **Step 8: Commit**

```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs rust/kansei-wasm/examples/fluid_clock/Cargo.toml
git commit -m "feat(fluid-clock): wire attractor, clock retag, and position readback"
```

---

## Task 6: Camera + scene framing for a readable clock

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

- [ ] **Step 1: Frame the clock**

The forked example's camera and world bounds are tuned for the generic fluid demo. Adjust so the 8-slot clock (total width ≈ `8 * 4.0 * 1.05 ≈ 34` world units from `SlotLayout::hh_mm_ss(4.0, ..)`) is visible and centered:

- Set the camera to look at the origin from a distance that frames ~40 units wide (increase the orbit radius / move the camera back). The example creates a `Camera` and `CameraControls` — increase the initial distance passed to `CameraControls::from_canvas(...)` (or the camera position) to ≈ 45–60.
- Ensure the sim's world bounds / particle spawn region are large enough to contain the clock band and a pool below it. If the fork spawns particles in a small box, widen the spawn to roughly `x ∈ [-18, 18]`, `y ∈ [-8, 8]`, `z ∈ [-2, 2]` so there are enough particles near every slot to recruit, plus a pool. Match `FluidSimulationOptions` world bounds accordingly (the example sets these; scale them up to fit).

- [ ] **Step 2: Increase particle count for legibility**

Raise the particle `count` in the fork from the demo default to ~120000 (per the design's tunable). If performance is poor in the browser, note it and leave a smaller default with a comment.

- [ ] **Step 3: Build + commit**

Run: `cargo build --target wasm32-unknown-unknown`
```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs
git commit -m "feat(fluid-clock): frame camera and scene for the HH:MM:SS clock"
```

---

## Task 7: Audio — per-second sine beeps with pitch tiers

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

- [ ] **Step 1: Implement `beep_for_change`**

Add these methods to `impl State`:

```rust
    fn ensure_audio(&mut self) {
        if self.audio_ctx.is_none() {
            self.audio_ctx = web_sys::AudioContext::new().ok();
        }
    }

    /// Beep once per second. Pitch tiers: hour rollover (m==0 && s==0) highest,
    /// minute rollover (s==0) higher, ordinary second base.
    fn beep_for_change(&mut self, _h: u32, m: u32, s: u32) {
        self.ensure_audio();
        let Some(ctx) = self.audio_ctx.as_ref() else { return; };
        let freq = if m == 0 && s == 0 {
            880.0 // hour
        } else if s == 0 {
            660.0 // minute
        } else {
            440.0 // second
        };
        let osc = match ctx.create_oscillator() { Ok(o) => o, Err(_) => return };
        let gain = match ctx.create_gain() { Ok(g) => g, Err(_) => return };
        osc.set_type(web_sys::OscillatorType::Sine);
        osc.frequency().set_value(freq as f32);
        let now = ctx.current_time();
        // Short pluck: 0.12 gain, decay over ~120 ms.
        gain.gain().set_value(0.0001);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.12, now + 0.01);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.0001, now + 0.13);
        let _ = osc.connect_with_audio_node(&gain);
        let _ = gain.connect_with_audio_node(&ctx.destination());
        let _ = osc.start_with_when(now);
        let _ = osc.stop_with_when(now + 0.14);
    }
```

> **Browser autoplay note:** `AudioContext` starts suspended until a user gesture. The Tweakpane UI (Task 8) or a one-time click handler must call `ctx.resume()`. Add a "Sound: on" toggle in Task 8 that resumes the context; until then beeps are silent (expected).

- [ ] **Step 2: Build + commit**

Run: `cargo build --target wasm32-unknown-unknown`
```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs
git commit -m "feat(fluid-clock): per-second sine beeps with minute/hour pitch tiers"
```

---

## Task 8: Tuning UI + build the final page

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/www/index.html`
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs` (wasm-bindgen setters)

- [ ] **Step 1: Export tuning setters from Rust**

Add `#[wasm_bindgen]` setters mirroring the existing example's pattern (`with_fluid`/`with_state`):

```rust
#[wasm_bindgen] pub fn set_attr_stiffness(v: f32) { with_state(|s| s.attr_stiffness = v); }
#[wasm_bindgen] pub fn set_attr_max_speed(v: f32) { with_state(|s| s.attr_max_speed = v); }
#[wasm_bindgen] pub fn set_attr_basin(v: f32) { with_state(|s| s.attr_basin = v); }
#[wasm_bindgen] pub fn set_per_slot_count(v: u32) { with_state(|s| s.per_slot_count = v as usize); }
#[wasm_bindgen] pub fn resume_audio() { with_state(|s| { s.ensure_audio(); if let Some(c) = s.audio_ctx.as_ref() { let _ = c.resume(); } }); }
```

(Use whatever the fork's global-state accessor is named — the fluid example uses a `with_fluid`/`with_state` thread-local pattern; mirror it. If only `with_fluid` exists, add a `with_state` equivalent or reuse it.)

- [ ] **Step 2: Update the HTML controls**

In `www/index.html`, keep the existing fluid controls and add a "Clock" Tweakpane folder wiring the new setters, plus a Sound toggle that calls `resume_audio()`:

```javascript
import init, {
  start,
  set_attr_stiffness, set_attr_max_speed, set_attr_basin, set_per_slot_count, resume_audio,
} from '../pkg/kansei_wasm_fluid_clock.js';

// ...after start():
const clockParams = { stiffness: 40, maxSpeed: 20, basin: 5, perSlot: 6000, sound: false };
const clock = pane.addFolder({ title: 'Clock' });
clock.addBinding(clockParams, 'stiffness', { min: 0, max: 200, step: 1 }).on('change', e => set_attr_stiffness(e.value));
clock.addBinding(clockParams, 'maxSpeed', { min: 1, max: 60, step: 1 }).on('change', e => set_attr_max_speed(e.value));
clock.addBinding(clockParams, 'basin', { min: 0, max: 30, step: 0.5 }).on('change', e => set_attr_basin(e.value));
clock.addBinding(clockParams, 'perSlot', { min: 500, max: 20000, step: 500 }).on('change', e => set_per_slot_count(e.value));
clock.addBinding(clockParams, 'sound').on('change', e => { if (e.value) resume_audio(); });
```

Update the JS `import` path/name to the fork's pkg name (`kansei_wasm_fluid_clock`). Update the page `<title>` to "Kansei — Fluid Clock". Ensure the `www/assets/` fonts/models the example loads still exist in the fork (they were copied in Task 4).

- [ ] **Step 3: Build the wasm package**

Run (from the example dir): `wasm-pack build --target web --release`
Expected: `pkg/` produced with `kansei_wasm_fluid_clock.js` + `_bg.wasm`, exporting `start`, `set_attr_stiffness`, `set_attr_max_speed`, `set_attr_basin`, `set_per_slot_count`, `resume_audio`.

- [ ] **Step 4: Serve and verify in a WebGPU browser**

Run (from the example dir): `python3 -m http.server 8788` (background), then open `http://localhost:8788/www/index.html` in Chrome/Safari-with-WebGPU.

Confirm visually:
1. Fluid particles assemble into the current time `HH:MM:SS` as a liquid surface.
2. When the seconds digit changes, its particles fall away and new ones form the next digit.
3. The Clock folder sliders visibly change attraction; enabling Sound produces a beep each second (higher on minute/hour rollover).

If you cannot drive a WebGPU browser from your environment, state that the wasm built and the exports are present, and hand the visual verification to the user with the exact command above.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-wasm/examples/fluid_clock/www/index.html rust/kansei-wasm/examples/fluid_clock/src/lib.rs
git commit -m "feat(fluid-clock): tuning UI, audio toggle, and final page"
```

---

## Self-Review Results

**Spec coverage (design doc):**
- Clock controller (wall-clock → HH:MM:SS, detect digit change) → Task 1 `ClockState`. ✓
- Reserved fixed subset / budget-constant swap → Tasks 2–3 (`initial_tags`, `retag_on_change`). ✓
- Nearest-particle recruitment on digit change → Task 2 `recruit_nearest` (nearest untagged), Task 3 uses it. ✓
- Released particles rejoin the fluid (no special handling) → Task 3 sets tags to `-1`; solver carries them. ✓
- Colons static → `ClockState::update` excludes glyph id 10; `SlotLayout` fixes colon slots. ✓
- Attractor integrated after solver, velocities only, base sim unchanged → Task 5 separate encoder. ✓
- Unified marching-cubes surface over all particles → inherited from the forked example (no change needed). ✓
- Audio: per-second sine, minute/hour pitch tiers → Task 7. ✓
- Tuning UI → Task 8. ✓

**Placeholder scan:** Part A (Tasks 1–3) is complete TDD code. Part B references the forked example's existing structure rather than reproducing its 1076 lines; the novel additions (attractor wiring, readback, audio, UI) are given as concrete code. The one genuinely fiddly spot — WASM async readback (Task 5b) — is called out explicitly with a fallback, because exact `map_async`/`unmap` lifetimes are environment-sensitive and must be adjusted against the compiler rather than pre-specified perfectly.

**Type consistency:** `ClockState::{new,update}`, `initial_tags(positions, layout, per_slot_count)`, `retag_on_change(positions, tags, layout, changed_slots, per_slot_count)`, `SlotLayout`, `NUM_SLOTS`, `GlyphAttractor::{new,set_tags,set_slots,set_params,dispatch}` are used consistently and match the Plan-1/Plan-2 public API.

**Known risks carried:**
- WASM position readback timing (Task 5b) — mitigated by tolerating 1–3 frame staleness and a documented fallback.
- Legibility/perf at 120k particles — tunable via UI; Task 6 notes lowering the default if the browser struggles.
- Retag cost is `O(untagged · log)` per changed slot on the CPU each digit change (≤ a few/sec) — negligible at these counts.
