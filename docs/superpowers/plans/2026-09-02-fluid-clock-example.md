# Fluid Clock Example (Plan 3 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `fluid_clock` WASM example — a water clock where fluid particles form the current time (`HH:MM:SS`), digits reassembling from the fluid as time changes, with a Tweakpane tuning UI and per-second audio beeps.

**Architecture:** Fork the existing fluid WASM example. All per-particle attraction state lives **on the GPU**: a persistent `tags` buffer plus a `cooldown` buffer, mutated entirely by compute passes (release → recruit → attract). The CPU only runs trivial clock arithmetic (`ClockState`) that updates a slots uniform and a `changed_mask`. **No GPU→CPU readback.** The existing marching-cubes surface renders all particles unchanged, so glyphs and pool read as one liquid.

**Tech Stack:** Rust, wgpu/WGSL (compute + atomics), wasm-bindgen, web-sys (WebAudio + Date), Tweakpane. Consumes `kansei_core::sdf` (Plan 1) and extends the `GlyphAttractor` (Plan 2).

**Scope note:** Plan 3 of 3. Plans 1–2 (merged on `feat/fluid-clock-sdf`) built the SDF module and the additive GPU attractor (native-verified PASS). This plan adds GPU-resident particle tagging + the clock UX. `ClockState` (CPU) is TDD'd; the GPU tagging passes are verified by a native readback example; the full example is verified in the browser.

---

## Why GPU-resident tags (design rationale)

An earlier draft kept tags on the CPU and read particle positions back each digit change to pick "nearest" recruits. That readback is unacceptable — it stalls the pipeline and is unnecessary. Instead:

- **Tags live on the GPU** (`array<i32>`, `-1` = ordinary fluid, `0..NUM_SLOTS` = attracted to that slot). A GPU **recruit** pass tags any untagged particle that sits inside an active slot's capture box, capped per slot by an atomic counter (the "budget"). "Inside the slot box" is inherently local, so this satisfies the "nearest to the slot" intent without sorting or readback.
- **Digit changes** are signalled from the CPU with a tiny `changed_mask: u32` uniform (bit `k` set = slot `k`'s digit changed). A GPU **release** pass sets `tag = -1` for particles in changed slots.
- **Fall-away effect:** a released particle is still physically in the box, so the recruit pass would re-grab it instantly and it would never fall. Fix: a per-particle `cooldown: u32`. On release, `cooldown = COOLDOWN_FRAMES`; the recruit pass ignores particles with `cooldown > 0` and decrements it each frame. During cooldown gravity/flow carries the particle out of the box, so it visibly falls to the pool before becoming eligible again elsewhere.

CPU per frame: `ClockState::update` → upload the slots uniform + `changed_mask`. That's it.

---

## Verification Strategy

- **Task 1 (`ClockState`):** pure CPU logic, strict TDD (`cargo test`).
- **Tasks 2–3 (GPU tagging passes):** WGSL cannot run under `cargo test`. Task 2 must compile; Task 3 is a **native readback example** that asserts the tag lifecycle on a real GPU (recruit tags in-box particles; release + cooldown un-tags them and prevents instant re-tag).
- **Tasks 4–8 (WASM example):** verified by `wasm-pack build` + a WebGPU browser (digits form from fluid, change over time, beep). If the browser can't be driven from the agent's environment, hand the visual check to the user with the exact serve command.

---

## File Structure

- Create `rust/kansei-core/src/simulations/fluid/clock.rs` — `ClockState` (CPU, TDD).
- Modify `rust/kansei-core/src/simulations/fluid/attractor.rs` — extend `GlyphAttractor` with the `cooldown`/`slot_fill` buffers, the tagging WGSL (clear/release/recruit), and a `retag(encoder, positions, changed_mask)` method.
- Modify `rust/kansei-core/src/simulations/fluid/mod.rs` — re-export `ClockState`.
- Modify `rust/kansei-native/examples/attractor_test.rs` OR create `rust/kansei-native/examples/tagger_test.rs` — native tag-lifecycle verification.
- Create `rust/kansei-wasm/examples/fluid_clock/` — forked example (Cargo.toml, src/lib.rs, www/).

---

## Task 1: `ClockState` — digits, change detection, and `changed_mask`

**Files:**
- Create: `rust/kansei-core/src/simulations/fluid/clock.rs`
- Modify: `rust/kansei-core/src/simulations/fluid/mod.rs`

- [ ] **Step 1: Declare + re-export**

In `mod.rs`, add `mod clock;` after the other `mod` lines and `pub use clock::ClockState;` after the `pub use` lines.

- [ ] **Step 2: Write the failing test**

Create `rust/kansei-core/src/simulations/fluid/clock.rs`:

```rust
//! CPU clock logic for the fluid clock: map wall-clock time onto the 8 glyph
//! slots, and report which digit slots changed (as indices and as a GPU bitmask).
//! All particle tagging happens on the GPU; this module only drives the slot
//! layout and the change signal.

use crate::simulations::fluid::{SlotLayout, NUM_SLOTS};

/// Tracks the currently-displayed glyph per slot so digit changes are detected.
pub struct ClockState {
    current: [i32; NUM_SLOTS],
}

impl ClockState {
    pub fn new() -> Self {
        ClockState { current: [-1; NUM_SLOTS] }
    }

    /// Apply `h:m:s` to `layout` and return the digit-slot indices whose glyph
    /// changed since the last call (colon slots are static and never reported).
    pub fn update(&mut self, layout: &mut SlotLayout, h: u32, m: u32, s: u32) -> Vec<usize> {
        layout.set_time(h, m, s);
        let mut changed = Vec::new();
        for i in 0..NUM_SLOTS {
            let g = layout.slots[i].glyph_id;
            if g != self.current[i] {
                if (0..=9).contains(&g) {
                    changed.push(i);
                }
                self.current[i] = g;
            }
        }
        changed
    }

    /// Bitmask form of a changed-slot list: bit `i` set = slot `i` changed.
    /// Uploaded to the GPU release pass.
    pub fn changed_mask(changed: &[usize]) -> u32 {
        let mut mask = 0u32;
        for &i in changed {
            mask |= 1u32 << i;
        }
        mask
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
        assert_eq!(changed, vec![0, 1, 3, 4, 6, 7]); // colons (2,5) excluded
    }

    #[test]
    fn only_changed_digit_is_reported() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 56);
        assert_eq!(clock.update(&mut layout, 12, 34, 57), vec![7]);
    }

    #[test]
    fn minute_rollover_changes_multiple_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 59);
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

    #[test]
    fn changed_mask_sets_the_right_bits() {
        assert_eq!(ClockState::changed_mask(&[0, 3, 7]), 0b1000_1001);
        assert_eq!(ClockState::changed_mask(&[]), 0);
    }
}
```

- [ ] **Step 3: Run to verify fail then pass**

Run (from `/Users/felixmartinez/Documents/dev/kansei/rust`): `cargo test -p kansei-core clock`
Expected: compiles and PASSES (5 tests).

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-core/src/simulations/fluid/clock.rs rust/kansei-core/src/simulations/fluid/mod.rs
git commit -m "feat(fluid): ClockState maps time to slots and emits a changed-slot mask"
```

---

## Task 2: Extend `GlyphAttractor` with GPU tagging (clear / release / recruit)

**Files:**
- Modify: `rust/kansei-core/src/simulations/fluid/attractor.rs`

GPU-only; no `cargo test`. Verified in Task 3. Extends the existing `GlyphAttractor` (which already owns the `tags` buffer, `slots` uniform, and the attractor pass from Plan 2).

- [ ] **Step 1: Add a tagging-params GPU struct**

Near the other `#[repr(C)]` structs in `attractor.rs`:

```rust
/// GPU params for the tagging passes (16 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuTagParams {
    /// Bit `k` set = slot `k`'s digit changed this frame (drives release).
    pub changed_mask: u32,
    /// Max particles a slot may hold (atomic budget cap).
    pub per_slot_count: u32,
    /// Frames a released particle stays ineligible for recruitment.
    pub cooldown_frames: u32,
    /// Capture-box scale (>=1.0 enlarges the slot box slightly for recruitment).
    pub capture_scale: f32,
}
```

- [ ] **Step 2: Add the tagging WGSL**

Add a second shader constant. It contains three entry points sharing one bind group:

```rust
const TAGGER_WGSL: &str = r#"
struct Slot {
    world_min: vec4<f32>,
    world_size: vec4<f32>,
    glyph_id: i32,
    _pad0: i32, _pad1: i32, _pad2: i32,
};
struct TagParams {
    changed_mask: u32,
    per_slot_count: u32,
    cooldown_frames: u32,
    capture_scale: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tags: array<i32>;
@group(0) @binding(2) var<storage, read_write> cooldown: array<u32>;
@group(0) @binding(3) var<storage, read_write> slot_fill: array<atomic<u32>, 8>;
@group(0) @binding(4) var<uniform> slots: array<Slot, 8>;
@group(0) @binding(5) var<uniform> params: TagParams;

// Reset the per-slot fill counters to the number already committed, so recruit
// only tops slots up to per_slot_count. Runs one thread per slot.
@compute @workgroup_size(8)
fn clear_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 8u) { atomicStore(&slot_fill[gid.x], 0u); }
}

// Count particles already committed to each slot (so recruit respects existing
// members against the budget). Runs one thread per particle.
@compute @workgroup_size(64)
fn count_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let t = tags[idx];
    if (t >= 0 && t < 8) { atomicAdd(&slot_fill[t], 1u); }
}

// Release particles whose slot changed; start their cooldown.
@compute @workgroup_size(64)
fn release(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let t = tags[idx];
    if (t >= 0 && t < 8) {
        if ((params.changed_mask & (1u << u32(t))) != 0u) {
            tags[idx] = -1;
            cooldown[idx] = params.cooldown_frames;
        }
    }
}

// Recruit untagged, off-cooldown particles that sit inside an active slot's box,
// up to the per-slot budget. Runs one thread per particle.
@compute @workgroup_size(64)
fn recruit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }

    // Tick down cooldown for everyone.
    let cd = cooldown[idx];
    if (cd > 0u) { cooldown[idx] = cd - 1u; return; }

    if (tags[idx] >= 0) { return; } // already committed

    let pos = positions[idx].xyz;
    // Find the (single, disjoint) active slot whose enlarged box contains pos.
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let slot = slots[k];
        if (slot.glyph_id < 0) { continue; }
        let half = 0.5 * slot.world_size.xyz * params.capture_scale;
        let center = slot.world_min.xyz + 0.5 * slot.world_size.xyz;
        let d = abs(pos - center);
        if (all(d <= half)) {
            // Claim a budget slot atomically.
            let n = atomicAdd(&slot_fill[k], 1u);
            if (n < params.per_slot_count) {
                tags[idx] = k;
            } else {
                atomicSub(&slot_fill[k], 1u); // give the slot back; stay untagged
            }
            return;
        }
    }
}
"#;
```

- [ ] **Step 3: Add the tagging buffers + pipelines to `GlyphAttractor`**

In `GlyphAttractor`'s struct, add fields:

```rust
    cooldown_buf: wgpu::Buffer,
    slot_fill_buf: wgpu::Buffer,
    tag_params_buf: wgpu::Buffer,
    tag_bgl: wgpu::BindGroupLayout,
    clear_fill_pipeline: wgpu::ComputePipeline,
    count_fill_pipeline: wgpu::ComputePipeline,
    release_pipeline: wgpu::ComputePipeline,
    recruit_pipeline: wgpu::ComputePipeline,
```

In `GlyphAttractor::new`, after the existing buffers/pipeline are built, create these. The `tags_buf` already exists (from Plan 2) — reuse it; it needs `STORAGE | COPY_DST` (it has that). Build:

```rust
        let cooldown_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/Cooldown"),
            size: (particle_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slot_fill_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/SlotFill"),
            size: (NUM_SLOTS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let tag_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/TagParams"),
            size: std::mem::size_of::<GpuTagParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tag_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlyphAttractor/TaggerShader"),
            source: wgpu::ShaderSource::Wgsl(TAGGER_WGSL.into()),
        });
        let tag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GlyphAttractor/TagBGL"),
            entries: &[
                entry(0, storage(true)),   // positions
                entry(1, storage(false)),  // tags
                entry(2, storage(false)),  // cooldown
                entry(3, storage(false)),  // slot_fill (atomic)
                entry(4, uniform),         // slots
                entry(5, uniform),         // tag params
            ],
        });
        let tag_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GlyphAttractor/TagPL"),
            bind_group_layouts: &[&tag_bgl],
            push_constant_ranges: &[],
        });
        let mk = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GlyphAttractor/TagPipeline"),
            layout: Some(&tag_layout),
            module: &tag_module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        });
        let clear_fill_pipeline = mk("clear_fill");
        let count_fill_pipeline = mk("count_fill");
        let release_pipeline = mk("release");
        let recruit_pipeline = mk("recruit");
```

Initialize `cooldown_buf` to zero (the buffer is created uninitialized; zero it once):

```rust
        queue.write_buffer(&cooldown_buf, 0, bytemuck::cast_slice(&vec![0u32; particle_count as usize]));
```

Add all new fields to the returned `GlyphAttractor { ... }`.

- [ ] **Step 4: Add the `retag` method**

```rust
    /// Run the GPU tagging passes: (re)count committed particles, release the
    /// changed slots, and recruit untagged in-box particles up to the budget.
    /// Call once per frame BEFORE `dispatch`. `changed_mask` bit `k` = slot `k`
    /// changed this frame (0 when nothing changed — release is then a no-op).
    pub fn retag(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        positions: &wgpu::Buffer,
        changed_mask: u32,
        per_slot_count: u32,
        cooldown_frames: u32,
        capture_scale: f32,
    ) {
        let p = GpuTagParams { changed_mask, per_slot_count, cooldown_frames, capture_scale };
        self.queue.write_buffer(&self.tag_params_buf, 0, bytemuck::bytes_of(&p));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GlyphAttractor/TagBG"),
            layout: &self.tag_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cooldown_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.slot_fill_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.slots_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.tag_params_buf.as_entire_binding() },
            ],
        });
        let pcount = self.particle_count;
        let pwg = (pcount + 63) / 64;
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GlyphAttractor/Tagging"), timestamp_writes: None,
        });
        cp.set_bind_group(0, &bg, &[]);
        // 1. zero fill counters, 2. count current members, 3. release changed, 4. recruit.
        cp.set_pipeline(&self.clear_fill_pipeline);  cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.count_fill_pipeline);  cp.dispatch_workgroups(pwg, 1, 1);
        cp.set_pipeline(&self.release_pipeline);     cp.dispatch_workgroups(pwg, 1, 1);
        // Recount after release so freed budget is available to recruit this frame.
        cp.set_pipeline(&self.clear_fill_pipeline);  cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.count_fill_pipeline);  cp.dispatch_workgroups(pwg, 1, 1);
        cp.set_pipeline(&self.recruit_pipeline);     cp.dispatch_workgroups(pwg, 1, 1);
    }
```

> **Note on the double count_fill:** release frees budget, so the fill counters must be recomputed before recruit or freed slots would appear full. The two clear+count sequences bracket the release. All six dispatches share one compute pass (no barrier needed between them on the same queue in wgpu — sequential pipeline sets are ordered).

- [ ] **Step 5: Compile**

Run: `cargo build -p kansei-core`
Expected: Rust compiles. (WGSL validated in Task 3.) Ensure `slot_fill` as `array<atomic<u32>, 8>` in WGSL matches an `NUM_SLOTS*4`-byte storage buffer.

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/simulations/fluid/attractor.rs
git commit -m "feat(fluid): GPU tagging passes (release/recruit with cooldown budget)"
```

---

## Task 3: Native verification of the tag lifecycle

**Files:**
- Create: `rust/kansei-native/examples/tagger_test.rs`

- [ ] **Step 1: Write the example**

Model window/renderer/readback boilerplate on `rust/kansei-native/examples/attractor_test.rs` (from Plan 2). It must, headlessly-on-first-frame then exit:

1. Build a `Renderer`, font, `GlyphVolumeSet`, and a `GlyphAttractor` for a small particle count (~2048).
2. Spawn particles: put a known cluster of ~200 particles INSIDE slot 0's box (read `layout.slots[0].world_min/size` after `SlotLayout::hh_mm_ss(4.0,1.5)` + `set_time(11,11,11)`), and the rest scattered outside all boxes.
3. Encode `attractor.retag(&mut enc, positions_buf, changed_mask=0, per_slot_count=100, cooldown_frames=30, capture_scale=1.0)` once, submit. Read back the `tags` buffer.
   - **Assert A (recruit):** ~100 particles are tagged to slot 0 (the budget cap), and they are all from the in-box cluster; particles outside all boxes are `-1`.
4. Encode `retag(... changed_mask = 1<<0 ...)` (slot 0 changed), submit, read back tags.
   - **Assert B (release + cooldown):** immediately after release, slot-0 tag count drops (released particles are on cooldown so are NOT instantly re-recruited in the same retag). Verify count tagged to slot 0 is < the budget (ideally 0 for the just-released set, though the recruit in the same retag may grab other in-box particles not on cooldown — so assert it did not simply refill to 100 with the SAME particles; simplest robust assertion: the set of slot-0-tagged indices after release differs from before, OR the count is 0 if the cluster was the only in-box population).

   To keep Assert B unambiguous: make the in-box cluster EXACTLY the budget (100 particles in box, per_slot_count=100). Then after recruit (step 3) all 100 are tagged. After release-with-cooldown (step 4), those 100 are released and on cooldown, and there are no other in-box untagged particles, so slot-0 count must be 0. Assert `slot0_count_after_release == 0`.
5. Print `TAGGER TEST: PASS` / `FAIL <numbers>`, then `std::process::exit`.

Use the same `MAP_READ` staging-buffer readback pattern as `attractor_test.rs`, but copy the `tags` buffer. The tags buffer needs `COPY_SRC`: it is created in `GlyphAttractor::new` with `STORAGE | COPY_DST` — add `| wgpu::BufferUsages::COPY_SRC` to the `tags_buf` descriptor in `attractor.rs` (small change; commit it with this task).

- [ ] **Step 2: Build + run**

Run (from `/Users/felixmartinez/Documents/dev/kansei/rust`): `cargo run -p kansei-native --example tagger_test`
Expected: prints `TAGGER TEST: PASS`. If it can't open a window in your environment, confirm it compiles and hand the run to the user. If it prints FAIL, debug the WGSL: most likely the atomic budget logic, the box-containment test (`capture_scale`/half-extent), or the double clear+count ordering.

- [ ] **Step 3: Commit**

```bash
git add rust/kansei-native/examples/tagger_test.rs rust/kansei-core/src/simulations/fluid/attractor.rs
git commit -m "test(fluid): native verification of GPU tag lifecycle (recruit/release/cooldown)"
```

---

## Task 4: Fork the fluid example into `fluid_clock`

**Files:**
- Create: `rust/kansei-wasm/examples/fluid_clock/` (copy of `rust/kansei-wasm/examples/fluid/`)

- [ ] **Step 1: Copy + clean**

```bash
cp -R rust/kansei-wasm/examples/fluid rust/kansei-wasm/examples/fluid_clock
rm -rf rust/kansei-wasm/examples/fluid_clock/pkg rust/kansei-wasm/examples/fluid_clock/target
```

- [ ] **Step 2: Rename the crate + add web-sys features**

In `rust/kansei-wasm/examples/fluid_clock/Cargo.toml`, set `name = "kansei-wasm-fluid-clock"`. Add to the `web-sys` `features` list (if absent): `"AudioContext"`, `"AudioDestinationNode"`, `"OscillatorNode"`, `"GainNode"`, `"AudioParam"`.

- [ ] **Step 3: Confirm the fork builds unchanged**

Run (from the fork dir): `cargo build --target wasm32-unknown-unknown`
Expected: compiles (byte copy with a new name).

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-wasm/examples/fluid_clock
git commit -m "chore(fluid-clock): fork the fluid wasm example as fluid_clock"
```

---

## Task 5: Wire the attractor + GPU tagging + clock into the example

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

No readback. The CPU only runs `ClockState` and uploads the slots uniform + `changed_mask`.

- [ ] **Step 1: Imports + font**

```rust
use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
use kansei_core::simulations::fluid::{GlyphAttractor, SlotLayout, ClockState};

const FONT: &[u8] = include_bytes!("../../../kansei-core/tests/fixtures/L10-medium.arfont");
```

- [ ] **Step 2: `State` fields**

```rust
    attractor: GlyphAttractor,
    slot_layout: SlotLayout,
    clock: ClockState,
    per_slot_count: u32,
    cooldown_frames: u32,
    capture_scale: f32,
    attr_stiffness: f32,
    attr_max_speed: f32,
    attr_basin: f32,
    audio_ctx: Option<web_sys::AudioContext>,
    last_second: i32,
```

- [ ] **Step 3: Build attractor + layout after the sim is created (before it moves into the effect)**

`count` is the particle count the example already computes. Add:

```rust
    let font = FontAtlas::parse(FONT).expect("parse font");
    let glyph_set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
    let attractor = GlyphAttractor::new(&renderer, &glyph_set, count as u32);

    let mut slot_layout = SlotLayout::hh_mm_ss(4.0, 1.5);
    let mut clock = ClockState::new();
    let (h0, m0, s0) = now_hms();
    let _ = clock.update(&mut slot_layout, h0, m0, s0);
    attractor.set_slots(&slot_layout);
    // Tags start all-untagged (buffer is created zeroed → all 0 == slot 0!).
    // IMPORTANT: initialize tags to -1 so nothing is attracted until recruited.
    attractor.set_tags(&vec![-1i32; count as usize]);
```

> The `tags` buffer is created without initialization; `set_tags(-1)` here guarantees a clean start (all ordinary fluid), and the GPU recruit pass fills the glyphs over the first frames.

- [ ] **Step 4: Initialize the new `State` fields**

```rust
        attractor,
        slot_layout,
        clock,
        per_slot_count: (count / 16).max(1) as u32,
        cooldown_frames: 45,
        capture_scale: 1.15,
        attr_stiffness: 40.0,
        attr_max_speed: 20.0,
        attr_basin: 5.0,
        audio_ctx: None,
        last_second: -1,
```

- [ ] **Step 5: In `render_frame`, after the sim steps and before render, run clock → tagging → attractor**

After the sim-step accumulator loop, add:

```rust
        // ── Clock → GPU tagging → attractor (all GPU; no readback) ──────
        let (h, m, s) = now_hms();
        let changed = self.clock.update(&mut self.slot_layout, h, m, s);
        let changed_mask = ClockState::changed_mask(&changed);
        if !changed.is_empty() {
            self.attractor.set_slots(&self.slot_layout);
        }
        if s as i32 != self.last_second {
            self.beep_for_change(h, m, s);
            self.last_second = s as i32;
        }
        self.attractor.set_params(1.0 / 60.0, self.attr_stiffness, self.attr_max_speed, self.attr_basin);

        {
            let fse = self.volume.effects[0].as_any().downcast_ref::<FluidSurfaceEffect>().unwrap();
            let pos_buf = fse.sim.positions_buffer().unwrap();
            let vel_buf = fse.sim.velocities_buffer().unwrap();
            let mut enc = self.renderer.device().create_command_encoder(&Default::default());
            // Tagging first (writes tags), then attraction (reads tags).
            self.attractor.retag(&mut enc, pos_buf, changed_mask, self.per_slot_count, self.cooldown_frames, self.capture_scale);
            self.attractor.dispatch(&mut enc, pos_buf, vel_buf);
            self.renderer.queue().submit(std::iter::once(enc.finish()));
        }
```

> **Borrow note:** `fse` borrows `self.volume`; `self.attractor`/`self.renderer` are separate fields, so the borrows don't conflict. If the compiler complains, capture `pos_buf`/`vel_buf` as `*const wgpu::Buffer` raw pointers the way the existing example does for the MC buffers (there is precedent in this file), or scope the `fse` borrow tightly.

- [ ] **Step 6: `now_hms` helper**

```rust
fn now_hms() -> (u32, u32, u32) {
    let d = js_sys::Date::new_0();
    (d.get_hours(), d.get_minutes(), d.get_seconds())
}
```

Ensure `FluidSurfaceEffect` is imported (the file already downcasts to it — reuse the import).

- [ ] **Step 7: Build + commit**

Run: `cargo build --target wasm32-unknown-unknown`
```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs rust/kansei-wasm/examples/fluid_clock/Cargo.toml
git commit -m "feat(fluid-clock): drive GPU tagging + attractor from the wall clock"
```

---

## Task 6: Camera + scene framing

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

- [ ] **Step 1: Frame the clock** — the 8-slot clock spans ≈ `8 * 4.0 * 1.05 ≈ 34` world units. Increase the camera orbit distance (`CameraControls::from_canvas(..)` distance / camera position) to ≈ 45–60 so ~40 units are visible, looking at the origin.

- [ ] **Step 2: Widen the spawn + world bounds** so particles populate every slot box plus a pool below: spawn roughly `x ∈ [-18, 18]`, `y ∈ [-8, 8]`, `z ∈ [-2, 2]`, and set `FluidSimulationOptions` world bounds to match. Ensure enough particles sit in/near each slot box for recruitment.

- [ ] **Step 3: Particle count** — raise `count` to ~120000 for legible digits (tunable; drop with a comment if the browser struggles).

- [ ] **Step 4: Build + commit**

Run: `cargo build --target wasm32-unknown-unknown`
```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs
git commit -m "feat(fluid-clock): frame camera and scene for the HH:MM:SS clock"
```

---

## Task 7: Audio — per-second sine beeps with pitch tiers

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`

- [ ] **Step 1: Add audio methods to `impl State`**

```rust
    fn ensure_audio(&mut self) {
        if self.audio_ctx.is_none() {
            self.audio_ctx = web_sys::AudioContext::new().ok();
        }
    }

    /// Beep once per second: hour rollover (m==0 && s==0) highest, minute
    /// rollover (s==0) higher, ordinary second base.
    fn beep_for_change(&mut self, _h: u32, m: u32, s: u32) {
        self.ensure_audio();
        let Some(ctx) = self.audio_ctx.as_ref() else { return; };
        let freq = if m == 0 && s == 0 { 880.0 } else if s == 0 { 660.0 } else { 440.0 };
        let (Ok(osc), Ok(gain)) = (ctx.create_oscillator(), ctx.create_gain()) else { return; };
        osc.set_type(web_sys::OscillatorType::Sine);
        osc.frequency().set_value(freq as f32);
        let now = ctx.current_time();
        gain.gain().set_value(0.0001);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.12, now + 0.01);
        let _ = gain.gain().exponential_ramp_to_value_at_time(0.0001, now + 0.13);
        let _ = osc.connect_with_audio_node(&gain);
        let _ = gain.connect_with_audio_node(&ctx.destination());
        let _ = osc.start_with_when(now);
        let _ = osc.stop_with_when(now + 0.14);
    }
```

> `AudioContext` starts suspended until a user gesture; the UI toggle in Task 8 calls `resume()`. Until then beeps are silent (expected).

- [ ] **Step 2: Build + commit**

Run: `cargo build --target wasm32-unknown-unknown`
```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs
git commit -m "feat(fluid-clock): per-second sine beeps with minute/hour pitch tiers"
```

---

## Task 8: Tuning UI + final page + browser verification

**Files:**
- Modify: `rust/kansei-wasm/examples/fluid_clock/src/lib.rs`
- Modify: `rust/kansei-wasm/examples/fluid_clock/www/index.html`

- [ ] **Step 1: Export setters** (mirror the fork's `with_state` thread-local pattern):

```rust
#[wasm_bindgen] pub fn set_attr_stiffness(v: f32) { with_state(|s| s.attr_stiffness = v); }
#[wasm_bindgen] pub fn set_attr_max_speed(v: f32) { with_state(|s| s.attr_max_speed = v); }
#[wasm_bindgen] pub fn set_attr_basin(v: f32) { with_state(|s| s.attr_basin = v); }
#[wasm_bindgen] pub fn set_per_slot_count(v: u32) { with_state(|s| s.per_slot_count = v); }
#[wasm_bindgen] pub fn set_capture_scale(v: f32) { with_state(|s| s.capture_scale = v); }
#[wasm_bindgen] pub fn set_cooldown_frames(v: u32) { with_state(|s| s.cooldown_frames = v); }
#[wasm_bindgen] pub fn resume_audio() { with_state(|s| { s.ensure_audio(); if let Some(c) = s.audio_ctx.as_ref() { let _ = c.resume(); } }); }
```

- [ ] **Step 2: Update `www/index.html`** — point the import at `kansei_wasm_fluid_clock.js`, set `<title>` to "Kansei — Fluid Clock", and add a Clock folder + Sound toggle:

```javascript
import init, {
  start, set_attr_stiffness, set_attr_max_speed, set_attr_basin,
  set_per_slot_count, set_capture_scale, set_cooldown_frames, resume_audio,
} from '../pkg/kansei_wasm_fluid_clock.js';

// after start():
const cp = { stiffness: 40, maxSpeed: 20, basin: 5, perSlot: 7500, capture: 1.15, cooldown: 45, sound: false };
const cf = pane.addFolder({ title: 'Clock' });
cf.addBinding(cp, 'stiffness', { min: 0, max: 200, step: 1 }).on('change', e => set_attr_stiffness(e.value));
cf.addBinding(cp, 'maxSpeed', { min: 1, max: 60, step: 1 }).on('change', e => set_attr_max_speed(e.value));
cf.addBinding(cp, 'basin', { min: 0, max: 30, step: 0.5 }).on('change', e => set_attr_basin(e.value));
cf.addBinding(cp, 'perSlot', { min: 500, max: 30000, step: 500 }).on('change', e => set_per_slot_count(e.value));
cf.addBinding(cp, 'capture', { min: 1.0, max: 2.0, step: 0.05 }).on('change', e => set_capture_scale(e.value));
cf.addBinding(cp, 'cooldown', { min: 0, max: 180, step: 5 }).on('change', e => set_cooldown_frames(e.value));
cf.addBinding(cp, 'sound').on('change', e => { if (e.value) resume_audio(); });
```

Keep the existing fluid controls. Confirm `www/assets/` (fonts/models the fork loads) survived the copy.

- [ ] **Step 3: Build the wasm package**

Run (from the fork dir): `wasm-pack build --target web --release`
Expected: `pkg/kansei_wasm_fluid_clock.js` + `_bg.wasm`, exporting `start` and all setters.

- [ ] **Step 4: Serve + verify in a WebGPU browser**

Run: `python3 -m http.server 8788` (background) → open `http://localhost:8788/www/index.html` in Chrome/Safari-with-WebGPU. Confirm:
1. Fluid assembles into the current `HH:MM:SS` as one liquid surface.
2. On a seconds change, the old digit's particles fall to the pool and new ones form the next digit (the cooldown makes the fall visible).
3. Clock sliders visibly affect attraction/recruitment; Sound toggle produces a beep per second (higher on minute/hour rollover).

If you cannot drive a WebGPU browser, report that the wasm built + exports are present, and hand the visual check to the user with the command above.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-wasm/examples/fluid_clock/src/lib.rs rust/kansei-wasm/examples/fluid_clock/www/index.html
git commit -m "feat(fluid-clock): tuning UI, audio toggle, and final page"
```

---

## Self-Review Results

**Spec coverage (design doc):**
- Clock controller (wall-clock → HH:MM:SS, digit-change detection) → Task 1 `ClockState`. ✓
- Reserved fixed subset / budget-constant swap → GPU recruit pass with per-slot atomic cap (Task 2). ✓
- Nearest-particle recruitment on digit change → GPU box-capture (local to the slot) + budget cap; satisfies "nearest to the slot" without readback (Task 2). ✓
- Released particles rejoin the fluid and fall away → release sets `tag=-1` + cooldown so gravity carries them out before re-eligibility (Task 2). ✓
- Colons static → excluded from `changed_mask`; always-active slots (Task 1 + `SlotLayout`). ✓
- Attractor additive, after solver, base sim unchanged → Task 5 separate encoder; tagging + attraction only touch tags/cooldown/velocities. ✓
- Unified marching-cubes surface over all particles → inherited from the fork. ✓
- Audio per-second sine with pitch tiers → Task 7. ✓
- Tuning UI → Task 8. ✓
- **No GPU→CPU readback** (the redesign) → all tagging on GPU. ✓

**Placeholder scan:** Task 1 is complete TDD code. Tasks 2/5 give complete Rust + WGSL. Task 3's assertion is made unambiguous by sizing the in-box cluster to exactly the budget. Part B references the forked example's structure rather than reproducing its 1076 lines; novel additions are concrete code.

**Type consistency:** `ClockState::{new,update,changed_mask}`, `GlyphAttractor::{new,set_tags,set_slots,set_params,retag,dispatch}`, `GpuTagParams`, `SlotLayout`, `NUM_SLOTS` are used consistently and extend the Plan-1/Plan-2 API. `retag` runs before `dispatch` every frame.

**Known risks carried:**
- WGSL atomic budget + double clear/count ordering — verified natively in Task 3 before the browser.
- Recruit is one-particle-per-thread with a per-slot atomic cap: recruits are in-box but not strictly the N nearest (acceptable per rationale). If a slot box spans a very dense region, `capture_scale`/`per_slot_count` tune the look.
- Legibility/perf at 120k particles — tunable; Task 6 notes lowering the default.
- First frames: tags start `-1`; glyphs fill in over the first ~cooldown frames as recruit runs. Acceptable startup transient.
