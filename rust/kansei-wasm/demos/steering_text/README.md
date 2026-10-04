# Steering text

About 1 600 words and short phrases (14 487 letters) flock in 3D as boids. Each word's first letter
is a vehicle steered by separation, cohesion, alignment, an attractor, wander and the pointer; the
other letters trail it on distance constraints, spaced by their glyph advances, and letters of
different words push each other apart. All letters are MSDF glyph quads in one instanced draw that
reads the simulation's positions buffer directly; the camera turns slowly around the cloud.

Engine API: `kansei_wasm::{Canvas, run, Frame, fetch_bytes}`, `RendererConfig` (4x MSAA, clear
colour), `simulations::grid::{NeighbourGrid, NeighbourGridOptions, GridLayout::covering}` and
`NEIGHBOUR_GRID_WGSL` for the neighbour search, `sdf::FontAtlas` (`parse`, `glyph`,
`glyph_rects`), `Material::msdf_text` with `MsdfTextOptions`, `InstancedGeometry` over a
`PlaneGeometry` fed by `ComputeBuffer::from_external` / `from_slice` with `with_vertex_vec4`,
`CameraControls` (`azimuth`, `set_azimuth`) and `MouseVectors`.

Demo-local: the simulation in `src/steering_sim.rs` and its WGSL in `src/shaders/` (the
`SimParams` uniform; `steering.wgsl`, the boid, attractor, wander, pointer and soft-wall forces on
the vehicles; `integrate.wgsl`, velocity, damping and the bounds clamp; `verlet.wgsl`, which snaps
each trailing letter to its rest length from the previous one; `repulsion.wgsl`, the push between
letters of different words), its default parameters, and the passes' bind groups and dispatch
order (grid, steering, integrate, verlet iterations, repulsion). `src/text_data.rs` turns the word
list into per-letter buffers: spaces dropped, each word started at a seeded random point, rest
lengths from the glyph advances times 1.5, colours from a six-entry palette by word. `lib.rs` casts
the pointer into a world-space ray and motion vector for the steering shader. The panel is in the
page. The neighbour grid (cell sort, the WGSL cell helpers), the glyph metrics and the MSDF text
material are engine API.

| URL parameter | Effect |
|---|---|
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom; moving the pointer pushes the words whose leading
letter lies near the ray under it, along the pointer's motion. A Tweakpane panel has folders for
steering (wander, separation, cohesion, alignment, radius, speed and force limits, damping, pointer
force, bounds, verlet iterations), letter collision (strength, radius, letters read per cell), the
attractor (strength and position), visual (auto-rotate speed, background colour) and the six
palette colours.

Assets: `www/words.json`, the word list (1 598 entries; the page fetches it and passes it to
`init_text`), and `www/assets/fonts/L10-medium.arfont`, an MTSDF font atlas fetched at run time
(the same file as `kansei-core/tests/fixtures/L10-medium.arfont`); source and licence not recorded
for either. Nothing is compiled in except the WGSL in `src/shaders/`. The page loads Tweakpane 4
from cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
