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

MacBook Pro, Apple M4 Pro (20-core GPU), headless Chrome 156 (WebGPU, vsync off, developer
timestamps), 2026-11-03. The GPU was shared with other work, so the modes are alternated inside
one page (`bench=1`) and compared only with each other. The final bench, with the grain applied
after SMAA's blend (and by the tonemapper under TAA):

```text
bench 1920x1080, 4 rounds: SMAA 1.036 ms | TAA 1.147 ms; frame GPU 1.874 / 2.365 ms; interval 9.881 / 10.277 ms; velocity pass -0.000 / 0.321 ms
round 1: SMAA 1.024 | TAA 0.960 ms
round 2: SMAA 0.896 | TAA 1.229 ms
round 3: SMAA 1.074 | TAA 1.238 ms
round 4: SMAA 1.149 | TAA 1.160 ms
SMAA cheaper than TAA: yes
```

- The frame's GPU time is 0.49 ms lower under SMAA (1.874 against 2.365 ms). Most of that is the
  renderer's velocity pass, 0.321 ms a frame under TAA and skipped under SMAA, which reads no
  motion vectors. (The report printed the skipped pass as -0.000, an empty sum's sign; it prints
  0.000 now.)
- The anti-aliasing passes alone: SMAA's three, 1.036 ms, against the TAA resolve's 1.147 ms,
  about 10 % less on the mean; the resolve won round 1.

SMAA was planned as one compute pass, and the checkpoint benches (before the grain and the
velocity skip) timed two single-pass designs against these three passes and TAA. Both were
slower than the three passes on this frame, so the three passes ship (`smaa.rs`'s module docs
have the native numbers too):

| single pass, in Chrome on this page | single pass | three passes | TAA resolve |
|---|---|---|---|
| 16 x 16 tiles, luma and edge bits in workgroup memory | 1.231 ms | 0.716 ms | 0.785 ms |
| 32 x 32 tiles, 4 px an invocation, edges straight from the input | 1.554 ms | 0.830 ms | 1.050 ms |

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
