# Volumetric smoke

Steam rising off a cup in a dark room, a low sun through the one window in a shaft that faint
volumetric fog makes visible. The steam is lit and composited by the volumetric fog, with the
fog's own lights, shadow map and phase function, so the window cross's shadow cuts through it as
it does through the fog, and fog in front of it veils it. Hover to stir it. The panel switches the
level of motion:

- **Waves**: no simulation; the plume swayed by sums of sines, evaluated where the march samples it.
- **Curl noise**: a density grid advected by divergence-free curl noise and a rise.
- **Fluid** (the default): a grid smoke solver with buoyancy, vorticity confinement and a pressure
  projection; the mouse leaves a wake.

With `?layers=1` smoke hangs in the room in three soft sheets (a second, room-sized volume with
no source, which the plume rises into; the fog marches both in one pass). On Fluid a weak ambient
draft keeps them drifting.

Engine API: `simulations::smoke` (`SmokeSimulation`: `add`, `step`, `set_motion`, `set_enabled`,
`volume_mut`, `handle`; `SmokeVolume::rising_from`, `SmokeMotion`, `FluidSmokeSettings`,
`CurlSettings`, `SmokeSource`, `SmokeFill::Layers`, `SmokeLook`, `SmokeMouse::from_screen`),
`VolumetricFogEffect::set_smoke` and `SmokeRenderOptions` (`smoke_render`), `pacing::FixedStep`,
`MouseVectors`, `CameraControls` (`with_mouse_pan`, `is_dragging`: no stirring while the camera
is dragged), `Material::standard_lit`, `Material::emissive`, `TemporalAAEffect`, `ToneMapEffect`.

| URL parameter | Effect |
|---|---|
| `motion=waves\|curl\|fluid` | the level of motion (default `fluid`) |
| `layers=1` | smoke hanging in the room in sheets |
| `aa=taa\|off` | TAA (default) or no anti-aliasing (the smoke has its own temporal filter either way) |
| `res=<16..160>` | the plume's grid, cells along its 0.8 m height (default 96: 48 x 96 x 48); the layers take two thirds |
| `stats=1` | log the interval between frames, the sim steps a frame and the GPU profile every 240 frames |
| `perf=1` | the performance overlay (or the backquote key) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: hover to stir the smoke; drag to orbit, right-drag or shift-drag to pan, the wheel to
zoom. The panel: the level of motion, the steam on or off, the hanging layers, the smoke's density,
the mouse's strength and radius, and the fog's density (at 0 the smoke still shows, lit by the
sun). For scripted checks, `window.kansei.set(key, value)` and `window.kansei.view()` (the keys
are the panel's: `motion` 0 waves / 1 curl noise / 2 fluid, `steam`, `layers`, `density`,
`mouseStrength`, `mouseRadius`, `fogDensity`).

GPU time on an Apple M4 Pro at 1920 x 1080 (`?dpr=1&stats=1`, Chrome with vsync and the frame
cap off), the median of three 240-frame windows, ms. The simulation is the `Smoke/*` passes
(Sources, Advect, Forces, Pressure; one 1/60 s step a frame at 60 fps), the rendering the fog's
`VolumetricFog/SmokeMarch` and `SmokeTemporal`; the frame's GPU time includes the room, the fog,
TAA and tone mapping.

| Level | Simulation, a step | Smoke rendering, a frame | GPU, a frame | Frame interval |
|---|---|---|---|---|
| Waves | 0 | 2.32 | 3.98 | 10.66 |
| Curl noise | 0.69 | 1.56 | 3.56 | 10.82 |
| Fluid | 1.53 | 0.91 | 3.38 | 10.84 |
| Fluid, `?layers=1` | 2.30 (both volumes) | 9.55 | 12.66 | 13.94 |

The hanging layers add about 9.3 ms of GPU time: the room-sized volume covers the whole screen,
so every half-resolution texel marches about 90 lit steps through the haze. Waves costs nothing
to simulate but more than Fluid to render, as its plume is lit across most of its box where the
Fluid plume's gas is zero outside its wisps. The frames here are not GPU-bound (the interval stays
near 11 ms with 3.4 ms of GPU work).

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
