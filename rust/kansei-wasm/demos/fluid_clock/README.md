# Fluid clock

The current time, HH / MM / SS in three stacked rows, formed out of an SPH fluid: particles are
recruited from a pool, pulled into extruded volumes of the digits' glyphs, and released back into
the pool when a digit changes. It renders as `fluid` does: a marching-cubes surface refracted by
`FluidSurfaceEffect`, then depth of field, under a striped dome.

Engine API: `kansei_wasm::{Canvas, run, fetch_bytes}`, `FluidSimulation` with
`FluidSimulationOptions::scaled_to_count`, `GlyphAttractor` (`set_slots`, `set_params`, `retag`
with `RetagParams`, `dispatch`), `SlotLayout::stacked_hh_mm_ss`, `ClockState`,
`sdf::FontAtlas` and `GlyphVolumeSet::for_clock_with_threshold`, `FluidSurfaceEffect`
(`splat_radius`, `surface_renderable`), `FluidDensityField`, `FluidMarchingCubes`,
`RaymarchingRenderable`, `DepthOfFieldEffect` in a `PostProcessingVolume`,
`GLTFLoader::load_glb`, `pacing::FixedStep`, `CameraControls` and `MouseVectors`.

Demo-local: `tuning_for`, which scales the sim to the particle count (radius, density target
and near pressure from `scaled_to_count`; pressure and time scale from per-count bands measured
with kansei-native's headless `clock_fill_test`; glyph budgets and emit rate in proportion to the
count), the rows' height and the colons' smaller budget, the dome's stripe shader, the beep each
second (Web Audio: 440 Hz, 660 Hz on the minute, 880 Hz on the hour) and the page's panel. The
glyph tagging, the attractor and the mapping of the time onto glyph slots are engine API.

| URL parameter | Effect |
|---|---|
| `n=<particles>` | particle count, default 500 000, clamped to 10 000..2 000 000; the sim parameters follow it (the page reads it and passes it to `start`) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom; moving the pointer over the fluid pushes it. A
Tweakpane panel has the clock's settings (stack height, attractor, recruiting, cooldown) and a
sound box that turns the beep on (off by default), plus the same folders as `fluid`.

Assets: `assets/L10-medium.arfont`, an MTSDF font atlas compiled in with `include_bytes!` (the
same file as `kansei-core/tests/fixtures/L10-medium.arfont`), and `www/assets/dome.glb`; source
not recorded for either. The page loads Tweakpane 4 from cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
