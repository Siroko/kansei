# Steering Text — Wasm Example Design

> 3D boids simulation where each word is a vehicle and its letters trail behind via verlet constraints, rendered as MSDF instanced text in a single draw call.

## Goal

New Rust wasm example (`rust/kansei-wasm/examples/steering_text/`) that renders psychology/mindfulness-themed words as 3D boids. Each word's first letter is a steering-behavior vehicle; subsequent letters trail via verlet distance constraints. All letters rendered in one instanced draw call using the engine's MSDF text system. Mouse interaction via MouseVectors displaces nearby vehicles.

## Architecture

**Hybrid JS/Rust** — JS handles font loading + JSON parsing (reuses existing arfont wasm parser), Rust handles simulation + rendering.

### Init Flow

```
JS                                    Rust
──                                    ────
Load words.json ────────────────────→ start(canvas_id)
Load .arfont → glyph metrics ──────→ init_text(words_json, glyph_json, msdf_rgba, w, h)
                                      ├── Parse words → build particle arrays
                                      ├── Upload MSDF atlas texture
                                      ├── Create InstancedGeometry (PlaneGeo base)
                                      ├── Create spatial hash buffers
                                      └── Create compute pipelines
```

### Entity Model

- **N words** from JSON. Each word has L letters → total **P particles**.
- Letter 0 of each word = "vehicle" (receives steering forces).
- Letters 1..L = "trailing" (pulled by verlet constraints toward previous letter).
- Spatial hash operates on **vehicles only** (N entries) for separation.

### GPU Buffers

| Buffer | Size | Role | Vertex Location |
|--------|------|------|-----------------|
| `positions` | P × vec4 | xyz + 1.0 | @location(3) |
| `velocities` | P × vec4 | xyz + 0 | compute only |
| `image_bounds` | P × vec4 | MSDF UV rect per glyph | @location(4) |
| `plane_bounds` | P × vec4 | glyph pixel rect | @location(5) |
| `colors` | P × vec4 | rgba (uniform, tweakpane) | @location(6) |
| `word_meta` | P × vec4(u32) | word_id, letter_idx, word_len, particle_offset | compute only |
| `rest_lengths` | P × f32 | verlet rest distance to previous letter | compute only |

### Per-Frame Compute Pipeline (single command encoder)

1. `grid-clear` — zero cell counts + scatter counters
2. `grid-assign` — hash vehicle positions (letter_idx==0) into cells
3. `prefix-sum-local` → `prefix-sum-top` → `prefix-sum-distribute`
4. `scatter` — sort vehicles by cell
5. `steering` — for each vehicle: separation (from hash) + wander + mouse force + bounds
6. `verlet` — for each trailing letter: constrain to previous letter (2-3 iterations)
7. `integrate` — velocity damping + position update for all particles

### Steering Behavior (shader: `steering.wgsl`)

Dispatched over N vehicles only.

- **Separation**: query 27 neighboring cells in spatial hash, accumulate repulsion from nearby vehicles within `separation_radius`. Force = `separation_strength * normalize(away) / distance`.
- **Wander**: 3D random walk using `sin(time * wander_speed + seed)` on 3 axes. Seeded per-vehicle.
- **Mouse force**: `mouse_direction * mouse_strength * falloff(distance_to_mouse)`. Same approach as fluid sim — MouseVectors provides screen-space direction + strength, projected into world via inverse view-projection.
- **Bounds**: soft wall at `[-bounds_size, bounds_size]` per axis. Steer back when near edges.
- **Clamp**: final force clamped to `max_force`, velocity clamped to `max_speed`.

### Verlet Constraints (shader: `verlet.wgsl`)

Dispatched over P particles, runs 2-3 iterations per frame.

For `letter_idx > 0`:
```
dir = positions[i] - positions[i-1]
dist = length(dir)
if dist > rest_length:
    correction = dir * (1 - rest_length / dist) * 0.5
    positions[i] -= correction
    // Don't move the previous letter (it's either the vehicle or already constrained)
```

### Rendering

- **Base**: `PlaneGeometry` (1 quad)
- **Instanced**: `instance_count = P`, 4 extra vertex buffers
- **Material**: MSDF shader (port from TS TextRenderShader):
  - Vertex: offset quad by `plane_bounds`, translate by `position`, map UV to `image_bounds`
  - Fragment: sample MSDF atlas, compute median, smoothstep at edge, output with alpha
- **Draw**: single `drawIndexedInstanced(6, P)` — one call for all letters
- **Blend**: transparent, depth write off, premultiplied alpha

### Tweakpane

```
Steering (folder)
├── separation strength   [0..10, default 3.0]
├── separation radius     [0..20, default 5.0]
├── wander strength       [0..5, default 1.0]
├── wander speed          [0..5, default 1.5]
├── mouse force           [0..5000, default 1500]
├── max speed             [0..20, default 8.0]
├── max force             [0..10, default 4.0]
├── damping               [0.9..1.0, default 0.98]
├── bounds size            [10..200, default 50]
└── verlet iterations     [1..8, default 3]

Visual (folder)
├── text color            [color picker, default #ffffff]
├── background color      [color picker, default #050510]
└── font size             [0.5..5, default 1.0] (uniform scale)

Camera (folder)
├── radius                [10..200, default 80]
└── auto-rotate speed     [0..2, default 0.1]
```

### Words (JSON)

Psychology/mindfulness themed. ~40-60 words:
```json
[
  "awareness", "presence", "breathe", "mindful", "clarity",
  "serenity", "balance", "focus", "observe", "acceptance",
  "compassion", "gratitude", "resilience", "stillness", "intention",
  "letting go", "equanimity", "empathy", "insight", "harmony",
  "patience", "kindness", "surrender", "grounding", "wholeness",
  "consciousness", "meditation", "reflection", "vulnerability", "courage",
  "authenticity", "flow", "peace", "trust", "release",
  "transform", "nurture", "connect", "listen", "heal",
  "gentle", "calm", "open", "anchor", "center",
  "wisdom", "wonder", "rest", "renew", "bloom"
]
```

### File Structure

```
rust/kansei-wasm/examples/steering_text/
├── Cargo.toml
├── src/
│   ├── lib.rs              # wasm entry, init, render loop
│   ├── steering_sim.rs     # SteeringSimulation struct (grid + compute passes)
│   └── shaders/
│       ├── steering.wgsl
│       ├── verlet.wgsl
│       ├── integrate.wgsl
│       ├── grid-assign.wgsl
│       ├── grid-clear.wgsl
│       ├── scatter.wgsl
│       ├── prefix-sum-local.wgsl
│       ├── prefix-sum-top.wgsl
│       ├── prefix-sum-distribute.wgsl
│       └── msdf-text.wgsl   # vertex + fragment for instanced MSDF
└── www/
    ├── index.html           # JS: font loading, tweakpane, rAF loop
    ├── words.json
    └── assets/
        └── fonts/
            └── L10-medium.arfont
```

### Camera & Scene

- Perspective, orbit controls with touch (`CameraControls::from_canvas`)
- Gentle auto-rotate around Y axis (tweakpane-adjustable speed)
- MouseVectors from canvas
- Dark background (tweakpane color)
- Single directional light (not critical — text is unlit, just MSDF alpha)

### Spatial Hash Grid

Reuse the exact grid-assign → prefix-sum → scatter pattern from the fluid sim. Only vehicle particles (N, not P) are hashed. The steering shader queries the sorted vehicle array for separation. Grid cell size = `separation_radius`, grid dims auto-computed from `bounds_size`.
