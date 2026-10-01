# Fluid Clock — SDF Glyph Attractor (Rust/WASM) — Design

**Date:** 2026-09-02
**Status:** Approved for planning (pending final spec review)

## Summary

A new Rust/WASM example that forks the existing fluid simulation (`rust/kansei-wasm/examples/fluid`)
and turns it into a **water clock**. A subset of the fluid particles are attracted into the shapes
of digit glyphs rendered as extruded 3D signed-distance fields, spelling the current time in
`HH:MM:SS`. When a digit changes, the particles forming it revert to ordinary fluid and an equal
number of nearby ordinary particles are recruited to form the new digit. The whole scene — glyphs
and the water pool they fall into — is rendered through the existing marching-cubes fluid surface as
one continuous liquid. A Tone.js layer beeps once per second, with distinct pitch tiers for
seconds / minute-rollover / hour-rollover.

The base fluid simulation is **unchanged in behavior**: attraction is an additive, opt-in per-particle
force. Particles not tagged as "attracted" behave exactly as they do today.

## Goals

- Reuse the existing MSDF font assets (`.arfont`) and the fluid sim + marching cubes rendering.
- Port glyph-SDF handling to **pure Rust** in `kansei-core` (no JS typography dependency in the example).
- Digits assemble from and dissolve back into the same shared fluid, with a fixed attracted-particle budget.
- Keep the fluid sim a drop-in: the attractor is purely additive.

## Non-Goals

- No general text layout engine — the glyph set is fixed to `0`–`9` and `:` (11 glyphs).
- No runtime SDF *generation* from font outlines — the `.arfont` atlas already contains a baked SDF.
- No changes to the fluid solver's core dynamics (pressure, viscosity, grid, integration).
- No rigid-body / `body-sdf` physics (that TS feature is out of scope here).

## Key Finding That Shapes the Design

The `.arfont` files (e.g. `examples/assets/fonts/L10-medium.arfont`) are **pre-baked MTSDF atlases**.
The existing loader only *parses* them; it never generates SDF at runtime. Each atlas provides:

- An RGBA atlas image where **alpha is a true single-channel SDF** (RGB is the multi-channel MSDF,
  which we do not need for attraction).
- Per-glyph metrics: `codepoint`, `advance`, `image_bounds` (pixel rect in the atlas),
  `plane_bounds` (em-space quad), and font `metrics` including `distance_range`.

So "port the MSDF pipeline to Rust" means **write a Rust parser for the artery-font binary format**,
not implement SDF generation. This substantially de-risks the effort.

## Architecture

### New module: `kansei-core/src/sdf/`

- **`arfont.rs`** — minimal parser for the artery-font binary format. Produces:
  - `AtlasImage { width, height, data: Vec<u8> /* RGBA */ }`
  - `Vec<GlyphMetrics>` where `GlyphMetrics { codepoint, advance, image_bounds:[f32;4], plane_bounds:[f32;4] }`
  - font `metrics` (em size, distance range) needed to interpret the SDF units.
  - Scope: single-image atlas, the tag types present in our `.arfont` files. Reuses the
    `GlyphMetrics` shape already defined in `steering_text/src/text_data.rs` as a reference.
- **`glyph_volume.rs`** — builds, once at init, an **extruded 3D SDF volume** per glyph:
  - Crop the glyph's SDF sub-rect from the atlas alpha channel using `image_bounds`.
  - Extrude to `N` layers along Z: `sdf3d(x,y,z) = max(sdf2d(x,y), abs(z) - halfDepth)`.
  - Because the atlas SDF has a **narrow distance range**, also derive a coarse inside/outside
    "attraction basin" (broad smooth falloff) so particles far from the glyph still feel a pull.
    The final field blends the precise near-field SDF with the coarse basin.
  - Output: a compact per-glyph 3D texture (or a shared 3D texture atlas keyed by glyph id) plus
    the glyph's local extent, uploaded to the GPU for sampling in the attractor pass.

### Fluid sim extension (net-new attractor pass) — `kansei-core/src/simulations/fluid/`

- **Per-particle tag + slot**: add a parallel GPU buffer with, per particle, an `attract_slot: i32`
  (`-1` = ordinary fluid; `0..8` = glyph slot index). No change to existing particle buffers.
- **Attractor compute pass** (new shader `attractor.wgsl`, dispatched before/with `forces.wgsl`):
  for each particle with `attract_slot >= 0`, look up that slot's active glyph id and world
  transform, sample the glyph's 3D SDF volume in the glyph's local space, and apply:
  - a spring force toward the interior (down the SDF gradient to the zero level-set / inside),
  - a small tangential noise term so particles spread to fill the glyph area (legibility),
  - a force clamp so attraction never destabilizes the SPH solver.
  Ordinary particles (`slot == -1`) are skipped entirely — the base sim is untouched.
- **Slot uniforms**: a small uniform/storage buffer of 8 slots, each holding
  `{ glyph_id, world_offset, scale, enabled }`, updated by the clock controller.

### Clock controller (Rust) — in the example crate

- 8 glyph slots laid out as `H H : M M : S S`. The two colon slots are **static** (never change;
  their particles are never released).
- Each frame: read wall-clock time (`js_sys::Date` in WASM) → format `HH:MM:SS` → per digit slot,
  if the digit changed since last frame, trigger a **retag** for that slot.
- **Retag on digit change** (the core mechanic):
  1. Particles currently tagged to that slot are flipped to `-1` (ordinary). They receive no
     special impulse — the fluid solver (gravity + pressure) carries them away naturally.
  2. An equal count of ordinary particles **nearest to that slot's world position** are flipped
     to the slot. Nearest-selection uses the fluid sim's existing spatial grid to gather local
     candidates (a small compute pass or a CPU readback of a coarse count — see Open Questions).
  - The attracted-particle **budget stays constant**; membership rotates. A just-released particle
    is immediately eligible for recruitment again (no settle gate).

### Rendering (unchanged)

- The existing marching-cubes surface runs over **all** particles. Glyph particles and pool
  particles form one continuous water surface — digits look sculpted from the same liquid.
- No separate visual treatment for text vs. pool.

### Audio (JS) — `www/index.html`

- Tone.js (or a minimal inlined WebAudio oscillator if we avoid the CDN dependency).
- One sine beep per second on the seconds tick. Pitch tiers:
  - seconds tick → base tone,
  - minute rollover → higher tone,
  - hour rollover → higher still.
- Driven from the JS animation loop reading the same wall-clock second, or via a Rust→JS callback
  on rollover. JS-side is simplest and keeps audio out of the WASM boundary.

## Data Flow

```
.arfont asset
   │  (fetch bytes in JS, hand &[u8] to WASM  — OR  include_bytes! at build time)
   ▼
Rust sdf::arfont parser ──► AtlasImage + Vec<GlyphMetrics>
   ▼
sdf::glyph_volume ──► 11 extruded 3D SDF volumes (built once) ──► GPU 3D texture(s)
   ▼
8 glyph slots positioned in world space  ◄── Clock controller (wall-clock, retag on change)
   ▼
Attractor compute pass: tagged particles pulled into active digit volumes
   ▼
Fluid solver (unchanged) + Marching cubes surface over ALL particles ──► water render
   ▼
Tone.js beep on each second tick (pitch tier by seconds/minute/hour rollover)
```

## Defaults (tunable)

- **Font:** `L10-medium.arfont` (already present in the Rust examples' assets).
- **Total particles:** 150,000 (base fluid example uses 50,000; digits need density to read).
- **Text budget:** ~55% of particles eligible as attracted, distributed across the 8 slots
  (colons get a small static share; the 6 digit slots share the rest). Final split tuned for legibility.
- **Extrusion layers `N`:** start at 8; tune for surface thickness vs. cost.
- **Attractor stiffness / clamp / tangential noise:** exposed via a Tweakpane folder for tuning.

## Risks & Mitigations

- **Narrow atlas SDF range** → weak long-range pull. *Mitigation:* blend precise near-field SDF with
  a coarse inside/outside basin (see `glyph_volume.rs`).
- **Digit legibility** as a fluid surface. *Mitigation:* tune particles-per-glyph, marching-cubes
  iso-level, extrusion depth, and attractor stiffness; all exposed in the debug UI.
- **Nearest-recruitment cost** (spatial query on digit change). *Mitigation:* reuse the sim's existing
  neighbor grid; retags are infrequent (≤8 per second, usually 1).
- **`.arfont` format surprises** in the parser. *Mitigation:* fall back to vendoring the existing
  artery-font Rust crate (the one compiled into the wasm loader) if the hand-written parser stalls.
- **Solver stability** under added forces. *Mitigation:* force clamp in the attractor pass; attraction
  is additive and bounded.

## Open Questions (to resolve during planning)

1. **`.arfont` ingestion:** `include_bytes!` at build time (fully self-contained WASM) vs. `fetch` +
   pass bytes into WASM at runtime. Leaning `include_bytes!` for a single fixed font.
2. **Nearest recruitment implementation:** GPU compute using the neighbor grid vs. a coarse CPU
   readback. Leaning GPU to avoid stalls; validate cost during implementation.
3. **Tone.js vs. inlined WebAudio:** CDN dependency vs. a few lines of inlined oscillator code.
   Leaning inlined WebAudio to keep the example self-contained.

## Implementation Order (high level)

1. `sdf::arfont` parser + unit test against `L10-medium.arfont` (assert `0`–`9`, `:` present).
2. `sdf::glyph_volume` extrusion + a headless test dumping a slice to verify shape.
3. Fork the fluid example; add per-particle tag buffer + slot uniforms (no forces yet — verify sim
   unchanged).
4. Attractor compute pass; statically tag one glyph and confirm particles assemble.
5. Clock controller + retag-on-change with nearest recruitment.
6. Marching-cubes over the combined set (should be automatic) + tuning UI.
7. Audio layer.
8. Polish: legibility tuning, defaults, README.
