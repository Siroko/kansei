# SMAA

SMAA 1x on thin, high-contrast things moving fast, where aliasing crawls and a temporal resolve
smears: a wire fence in a ±45° lattice across the view, square-mesh fences down both sides, two
lattice towers against the sky and a rotor of eight thin bars turning twice a second, under a
camera that swings left and right at up to 90° a second. SMAA works on each frame alone, so the
pan stays crisp; the panel switches to temporal AA, or to none, on the same scene. Film grain is
on: under SMAA the tonemapper leaves its grain and dither to it, added after the blend, so the
grain isn't smoothed as edges. With neither TAA nor anything else reading motion vectors, the
renderer skips its velocity pass.

Engine API: `SmaaEffect`, `SmaaOptions`, `FilmGrain`, `ToneMapOptions::defer_film` /
`ToneMapEffect::later_film`, `TemporalAAEffect`, `PostProcessingVolume::set_enabled`,
`PostProcessingEffect::reads_velocity`, `Material::standard_lit` with `outputs_velocity`,
`Material::gradient_sky`, `Geometry::merged`, and `Renderer::set_profiling` / `take_profile`.

| URL parameter | Effect |
|---|---|
| `aa=off\|smaa\|taa` | the starting anti-aliasing (default `smaa`) |
| `threshold=<luma>` | SMAA's edge threshold (default 0.1) |
| `grain=<amount>` | film grain (default 0.3) |
| `dither=0` | no dither of one 8-bit step |
| `pan=<deg/s>` | the camera's top swing speed (default 90) |
| `t=<seconds>` | freeze the scene and the camera at that time |
| `size=<w>x<h>` | draw at that size whatever the window's (`Canvas::with_size`) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |
| `perf=1` | the performance overlay (or the backquote key) |
| `reach=<px>` | diagnostic: SMAA's search reach (default 16, iryoku High's is 32) |
| `bench=1` | diagnostic: SMAA and TAA in turn at 1920 x 1080 (unless `size=`), the scene held at `t` (default 2), 8 s each, 4 rounds after 3 s; reports each one's mean anti-aliasing GPU time, the frame's GPU time and interval, and the velocity pass's time |

Controls: the panel sets the anti-aliasing (Off, SMAA, TAA), SMAA's threshold, the pan and its
speed, and the grain; it shows SMAA's reach.

## Measured

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
