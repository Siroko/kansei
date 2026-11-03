# SMAA in one pass: design

Date: 2026-11-03. Engine: Rust (`kansei-core`), shown by a new WebAssembly example.

## Intent

Kansei's only anti-aliasing on the GBuffer path is `TemporalAAEffect`. TAA accumulates jittered
frames, and in scenes where the action is fast (whip pans, fast objects, particles) that history
smears and ghosts. The owner wants a second anti-aliasing option to pick instead of TAA for those
scenes and demos: SMAA, which works on the current frame only, so fast motion stays crisp.

It is chosen for performance as much as for crispness: it must cost less than TAA. That is why
it is one pass: SMAA's three standard passes (edge detection, blending weights, neighbourhood
blending) fused into a single dispatch.

TAA and SMAA are complementary. A scene picks one of them. TAA stays the default and stays
unchanged.

## Goals

- `SmaaEffect`, a `PostProcessingEffect` in `kansei-core` (`postprocessing/effects/smaa.rs`,
  exported beside `TemporalAAEffect`), that anti-aliases the frame in **one compute dispatch**.
- Quality close to reference SMAA 1x (Jimenez et al., <https://github.com/iryoku/smaa>) at a
  preset the builder picks and states (expected: Medium or High): luma edge detection, orthogonal
  pattern search with the precomputed area and search textures, and diagonal handling if it fits
  the pass.
- Cost, measured on the owner's Mac (Apple GPU) at 1080p in the new example:
  - lower than `TemporalAAEffect` on the same frame
  - lower than a standard three-pass SMAA on the same frame
- A new example, `rust/kansei-wasm/examples/smaa/`, that shows it (below).

## Non-goals

- Changing `TemporalAAEffect`, or making SMAA the default anywhere.
- Running SMAA and TAA together in one scene (SMAA T2x and similar). This was not asked for.
- Upscaling. TAA is also the engine's temporal upscaler (`Renderer::set_render_scale` below 1). An
  SMAA scene renders at full resolution, so `SmaaEffect::upscales_to_display` is false.
- A TypeScript port (Rust only for now).
- Web Worker (`?worker=1`) support in the example.

## Anti-goals (failure even if it technically works)

- A "single pass" that is slower than three passes, or slower than TAA. If the fused pass can't
  beat both, stop and report to the owner with numbers. Don't ship it, and don't quietly ship
  three passes instead.
- Visible blur on text or texture detail beyond what reference SMAA does. SMAA should only touch
  geometric edges.
- Requesting camera jitter. `wants_jitter` must be false.

## Constraints

- Place it after the tonemapper (and colour grading), on display-referred colour. SMAA detects
  edges by luma contrast and expects perceptual values. Check what the tonemapper writes and what
  the chain's last effect hands to `PostProcessingVolume`'s blit
  (`postprocessing/volume.rs`, `effect.rs`), and document where `SmaaEffect` must go, the way
  `TemporalAAEffect`'s doc comment does.
- WebGPU limits: the pass must work under the default limits: workgroup storage of 16 KiB, and
  256 invocations per workgroup. wgpu 30 on the web cannot request more.
- WGSL that Chrome accepts. Validate with naga in `cargo test -p kansei-core` (the
  `shaders_validate_*` pattern), check each `#[repr(C)]` uniform against its WGSL size, then load
  it in Chrome (AGENTS.md lists WGSL that naga accepts and Chrome rejects).
- The area and search textures are MIT (SMAA's licence). Ship them in a form the crate embeds,
  either as bytes or generated at build time, and credit SMAA in the module docs and the README's
  licence line.
- Time it the way AGENTS.md describes: `Renderer::set_profiling(true)` and `profiling::gpu_pass`
  labels. On Apple GPUs, time a span across passes, not each pass's own timestamps. Because the
  GPU is shared, compare A and B inside one page (the example's `?bench=1`), never across runs.

## Decided

- One compute dispatch over workgroup tiles. Each workgroup loads its tile of colour plus a
  border into workgroup memory, finds edges for the tile and border, computes blending weights
  for the tile (plus the one-pixel ring the blend reads), and blends. The border width limits how
  far the edge search reaches, so the search distance is the preset's main lever. State the
  chosen distance and what it costs in quality against reference SMAA.
- A standard three-pass SMAA kept as a reference: in tests, and as a mode the bench can switch
  to (for example `SmaaEffect::with_mode(SmaaMode::ThreePass)`; the name is yours). It is the
  baseline the cost goal is measured against, and the image the GPU test compares with.
  *(My call)* It is three compute passes in this engine's style, not a port of iryoku's
  fragment passes with a stencil mask:
  - It uses the same texture formats and the same search reach and diagonal setting as the
    single pass, so what gets compared is fusing against splitting the same algorithm.
  - Skipping non-edge pixels with an early return in the weights pass stands in for the stencil.
  - Document in the module where it differs from iryoku's reference.
  - Its search reach is configurable, so it can also run at the full preset's reach.
- A GPU test (`tests/smaa_gpu.rs`, the `*_gpu.rs` pattern). Render synthetic edges (a rotated
  square, near-horizontal and near-vertical lines, a diagonal) and check two things:
  - the single pass matches the three-pass reference, at the same reach, within a small
    per-pixel tolerance (this proves the fusing is correct)
  - flat regions and texture-like noise come through unchanged
- *(My call)* The quality cost of the shorter reach is measured separately, not tested. Compare
  the single pass with the three-pass at the full preset's reach on the same synthetic edges and
  on the example's frame, and report the difference (mean and max per-pixel error, and where it
  shows) in the module docs and the PR.
- *(My call)* The number that decides whether to ship is the AA work's own GPU time: its passes'
  `profiling::gpu_pass` span, single pass against three-pass against `TAA/Resolve`, averaged
  over the bench's rounds. The whole-frame GPU time and frame interval are reported beside it.
  At 1080p the AA pass is a fraction of a millisecond, which the shared GPU's run-to-run noise
  can swamp at whole-frame level.

## The example: `rust/kansei-wasm/examples/smaa/`

Thin, high-contrast shapes moving fast: wire fences, spinning thin bars, a lattice, and a camera
that pans quickly (or orbit controls with an auto-pan the panel can stop). That is where aliasing
crawls and TAA smears, so the three modes look different.

- A Tweakpane panel (as the other examples have) switches **Off**, **SMAA** and **TAA**, and
  shows SMAA's preset or search distance if there is one.
- `?aa=off|smaa|taa` picks the starting mode.
- `?bench=1` alternates SMAA (single pass), SMAA (three-pass reference) and TAA inside the page,
  a few seconds each, several rounds, and reports each mode's mean GPU time and frame interval,
  the way `occlusion-culling`'s `bench=1` does.
- A README in the examples' format: what it shows, the engine API, its URL parameters, and the
  measured numbers.
- A row on `examples/index.html` under the post-processing / anti-aliasing area, with a WASM
  button and dashed variant buttons for `?aa=taa` and `?aa=off`.

## Decided with the owner after review

**Skip the motion-vector pass when nothing reads it.**
- Today `Renderer::render_scene_to_gbuffer` runs `draw_velocity` every frame, redrawing every
  material with `outputs_velocity`. TAA needs it; SMAA doesn't. Skipping it in SMAA scenes is
  part of the performance win.
- Add a `PostProcessingEffect` hook saying whether an effect reads the GBuffer's velocity
  (`reads_velocity`, default **true**, so no existing scene changes). Return false from
  effects that are known not to read it, including `SmaaEffect`, the tonemapper, colour grading,
  bloom and the fogs.
- The renderer skips the velocity pass when no active effect in the volume's chain reads it.
- Audit every reader of the velocity texture before marking anything false: `TemporalAAEffect`,
  `MotionBlurEffect`, the RT GI's SVGF, and any temporal reprojection. Then check in Chrome that
  `taa`, `gi-box` and `outdoor-gi` look unchanged.
- Report the frame time saved by the skip in the example's bench beside the AA pass's own time.
  The ship-or-not gate stays the AA pass's own time.

**Film grain keeps working with SMAA.**
- Run after the tonemapper, SMAA would treat grain as edges and smooth it.
- Add an option to `ToneMapEffect` to leave grain, vignette and dither to a later pass.
  `SmaaEffect` applies them after blending, in its single pass, with the tonemapper's
  settings, so the image matches what the tonemapper would have made, minus the smoothing.
- Chromatic aberration stays in the tonemapper. It reads colour up to about 1% of the image
  away, beyond the pass's tile, and it is smooth enough that SMAA barely touches it.
- The example has grain on by default, so the difference shows.

## Left to the builder

- Tile size, border width and preset; whether diagonal search fits the pass's memory and cost.
- How to load and store the area and search textures.
- How the effect is configured (`SmaaOptions` with a preset or thresholds, as you see fit).
- The example scene's exact contents, as long as it has fast thin edges and a fast pan.

## Done when

- `cargo test -p kansei-core` passes, including the new shader validation and the GPU test.
- The example builds with `wasm-pack build --target web --release`, runs in Chrome, and its
  `?bench=1` shows the single pass cheaper than both TAA and the three-pass reference at 1080p
  on the owner's Mac, with the numbers in the example's README and the PR.
- A PR against `development`, with no AI attribution in commits or the PR.
