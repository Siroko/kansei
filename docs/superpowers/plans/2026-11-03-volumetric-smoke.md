# Volumetric Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Smoke and steam in `kansei-core` at three levels of motion (Waves, Curl noise, Fluid), ray-marched and lit through `VolumetricFogEffect` so smoke and fog read as one medium. A new `smoke` WebAssembly example shows it and opens on Fluid.

**Architecture:** `simulations::smoke` holds the app-owned `SmokeSimulation`: the volumes, their grids, the Curl and Fluid steppers, and a `SmokeHandle` that tracks the current textures. `VolumetricFogEffect::set_smoke(handle)` makes the fog march every enabled smoke volume in one half-resolution pass. The pass uses the fog's own lights, shadow maps, sky and phase function, and reads the fog's froxel grid at each step. The fog's composite applies smoke and fog together. The march has its own temporal filter, separate from TAA.

**Tech Stack:** Rust 2021, wgpu 30 under WebGPU's default limits, WGSL checked by naga 30 in `cargo test -p kansei-core`. The example uses wasm-pack/wasm-bindgen and a Tweakpane 4 panel from the CDN, as `examples/depth-of-field` does.

**Spec:** `docs/superpowers/specs/2026-11-03-volumetric-smoke-design.md`. Read it before Task 1. It is approved, and this plan only adds the decisions it left to the builder.

## Global Constraints

- Branch off `development`, and open the PR against `development`. Commits and the PR carry no AI attribution. Commit subjects follow the log's style, `<area>: <what, in prose>`, for example `smoke: the Fluid level's pressure projection`.
- Rust only: no TypeScript port, no `?worker=1`. Don't change how the fog looks in a scene without smoke. The fog's existing tests (`volumetric_fog.rs`) must pass unchanged.
- The TS engine `?raw`-imports `volumetric_fog_inject.wgsl`, `_media`, `_spot`, `_composite`, `froxel_common.wgsl` and `compute_shadows.wgsl` (`src/postprocessing/effects/VolumetricFogEffect.ts`). It also text-patches `if (pl.shadowLayer != NO_SHADOW && params.hasPointShadows != 0u) {`.
  - Keep that line verbatim, along with the names it uses (`pl`, `visibility`, `worldPos`, `params`).
  - Keep `spotInScatter`'s signature, and the composite's `main`, its bindings 0-6 and its 32-byte `CompositeParams`.
  - Don't edit `froxel_common.wgsl` or `compute_shadows.wgsl`.
  - A task that edits a `?raw` file also runs `pnpm test` and `pnpm exec tsc`, and loads `examples/index_volumetric_fog.html` and `examples/index_spot_lights.html`.
- WebGPU default limits per stage: 16 sampled textures, 4 storage textures, 8 storage buffers, 12 uniform buffers, 16 samplers, 256 invocations per workgroup.
  - No optional features, so no `float32-filterable` and no read-write storage textures.
  - Filtered or stored grids are `rgba16float`, unfiltered scalars `r32float`, the march's aux target `rgba32float`.
- WGSL Chrome rejects although naga accepts it: reserved words as names (`patch`, `filter`), and unparenthesized comparisons in call arguments (`f(a < b, c > d)`). Load every new shader in Chrome.
- Every new WGSL module is validated with naga, and every `#[repr(C)]` GPU struct is checked against its WGSL struct's size (pattern: `volumetric_fog.rs::shaders_validate_and_uniform_layouts_match`).
- GPU tests (`rust/kansei-core/tests/*_gpu.rs`, or in-crate like the fog's) pass without running when no adapter exists: `let Ok(adapter) = … else { return eprintln!("no GPU adapter: skipping") }`.
- `queue.write_buffer` lands before the next submit.
  - `SmokeSimulation::step` makes one write (every volume's 256-byte slot) and one submit.
  - The fog uploads the smoke's frame data in one write.
  - Per-dispatch variations are entry points or slots, never rewrites.
- Label every pass with `crate::profiling::gpu_pass` (labels in decision 10).
- Depth is `[0, 1]`, cleared to 1.0 (sky). Front faces are counter-clockwise.
- Nothing in `simulations::smoke` depends on `wasm32`. `CameraControls`' DOM code stays `cfg(target_arch = "wasm32")`.
- The example:
  - uses kansei-core's public API only (`cargo test -p kansei-wasm` checks it);
  - takes its web plumbing from `kansei-wasm` (`Canvas`, `run`, `param`, `param_or`, `flag`, `perf`);
  - draws with `Material::standard_lit` through a `PostProcessingVolume` (`sample_count: 1`);
  - is a standalone crate with its own `[workspace]`, listed under `exclude` in `rust/Cargo.toml`.
- No new dependencies in kansei-core. Module docs credit the papers followed: Stam 1999; Fedkiw, Stam and Jensen 2001; Bridson, Hourihan and Nordenstam 2007; Selle et al. 2008; Wrenninge 2013.
- Browser checks follow R1-R4 below. Never kill or reuse another session's `chrome-devtools-axi` or server.

## Decisions this plan makes (the design left them to the builder)

**[interface]** marks a decision that shapes the API between tasks. The executor may change the others if a test or a checkpoint argues for it.

1. **[interface] Rendering lives inside `VolumetricFogEffect`.** `set_smoke(Some(&handle))` registers the volumes of an app-owned `SmokeSimulation`. Why:
   - The march needs the fog's private lighting inputs (`ComputeShadows`, sky lighting, sky occlusion, clipmap probes, `FogParams`), so a scene binds its lights once.
   - The seam needs the fog's froxel grid at every step, and the fog's composite in the same pass.
   - Smoke without visible fog is a fog with `base_density: 0`.
   - The GPU side sits in `postprocessing/effects/volumetric_smoke.rs` (`pub(crate) SmokeRenderer`), not in `volumetric_fog.rs`.
2. **[interface] One lighting code path.** `volumetric_fog_inject.wgsl` gains `fogDirLight` and `fogPointLight`, and `volumetric_fog_spot.wgsl` gains `fogSpotLight`. Each returns `FogLight { toLight, radiance }`: colour × falloff × shadow-map visibility, with no phase. The fog sums them with `henyeyGreenstein`, the smoke with `smokePhase` (decision 5).
3. **[interface] The seam.** One march over all volumes along each ray (decision 6) yields the smoke's transmittance `Ts` and a radiance `R`. The composite with smoke is `out = Ts * (scene * fog.a + fog.rgb + shafts) + R`.
   - `R` holds the fog in front of the smoke, the smoke veiled by it, and the removal of the fog the smoke hides.
   - With no smoke, `Ts = 1` and `R = 0`, which is exactly the fog's composite.
   - Raymarched spot shafts (`SpotScattering::Raymarched`) count as behind the smoke. That is approximate; froxel spots are exact.
4. **[interface] Grids** live in each volume's yawed box, with usages `TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST`.
   - `scalars`: `rgba16float` (r density, g temperature), three textures rotating for MacCormack.
   - `velocity`: `rgba16float` ×2 on a MAC grid (Fedkiw 2001). Texel `(i, j, k)` holds `x` = u on face `(i-½, j, k)`, `y` = v on `(i, j-½, k)`, `z` = w on `(i, j, k-½)`, in m/s along the box's local axes. The outer +faces are walls and hold 0.
   - `pressure`: `r32float` ×2; `divergence`: `r32float`; `vorticity`: `rgba16float`.
   - Fluid volumes have all of them, Curl volumes scalars only, Waves volumes none.
5. **Lighting.**
   - Self-shadowing is a short march (`shadow_steps`, default 6, steps ×1.5) toward each light through every volume's density, giving `tau`.
   - `smokePhase(c, g, tau) = HG(c, g) e^-tau + Σ_{o=1,2} 0.5^o HG(c, 0.5^o g) (e^{-0.5^o tau} - e^-tau)`. These are Wrenninge's octaves, written so `smokePhase(c, g, 0) == HG(c, g)`: thin smoke scatters as the fog does.
   - Ambient and sky are the fog's (`params.ambient + skyAmbient(viewDir, p)`), not self-shadowed.
   - Scattering is `density * density_scale * albedo`, extinction `density * density_scale * extinction`, with the fog's anisotropy.
6. **[interface] The march** runs at half resolution, each texel taking the front-most depth of its 2×2 pixels. Its targets:
   - `light`: `rgba16float` (R, Ts).
   - `aux`: `rgba32float` (scene linear depth, the representative point's previous uv x and y, its ray distance).

   The composite upsamples both by depth, as `upsampleShafts` does.
   ```
   segs = every enabled volume's smokeRayInterval clipped to [0, tScene], boundaries sorted; a gap is one smoke-free step
   dt per segment = step_scale * finest cell covering it; if the total passes max_steps, every dt grows by the same factor
   Ts = 1; Rs = 0; Rf = 0; Lprev = fogL(t_first)          // fogL/fogT: froxel accum .rgb/.a at slice(t * cosView), no screen jitter
   for each step [t0, t1], sample x jittered inside it:
       L1 = fogL(t1); Rf += (1 - Ts) * (L1 - Lprev); Lprev = L1
       m = smokeMedium(x)                                  // summed over the volumes at x
       if m.extinction > 0:
           e = exp(-m.extinction * (t1 - t0))
           Rs += Ts * fogT(t0) * m.scattering * smokeLight(x, rd) * (1 - e) / m.extinction
           Ts *= e
       if Ts < 0.003: Rf += (1 - Ts) * (fogL(t_last) - Lprev); Lprev = fogL(t_last); break
   R = Rs + (1 - Ts) * Lprev - Rf
   ```
   - The representative point P sits at the distance weighted by `Ts * (1 - e)`.
   - Its velocity v comes from the Fluid grid (sampled once), the Curl field, or is zero for Waves.
   - Its previous uv projects `P - v * advanced` (the sim seconds stepped since the last render) through the previous view-projection.
7. **Temporal.** A pass of its own, like the shafts':
   - Reproject by `aux.yz`.
   - Clamp to the 3×3 min/max, padded by a quarter of the range.
   - Drop history where the depth moved more than 10%.
   - Blend 0.25 new.
   - Reset on a camera-frame gap, a resize, `reset_history` or `set_smoke`.
   - With `temporal: false` the jitter is a fixed 0.5 (tests).
8. **Solver.**
   - Fluid:
     - semi-Lagrangian velocity advection;
     - MacCormack scalars with Selle's clamp (forward, then backward-and-correct);
     - Jacobi pressure, ping-ponged and warm-started, `pressure_iterations` default 40;
     - vorticity confinement scaled by the cell size;
     - buoyancy `(-weight * density + buoyancy * temperature) * up`;
     - a `max_speed` cap;
     - closed walls.
   - Curl runs the same sources and scalar advection with the velocity `curlNoise + rise * up + mouse` in place of a grid. This is the design's "Curl and Fluid share the density grid, its advection and its rendering". Curl noise is the curl of a three-component noise potential (`CURL_NOISE_WGSL`, public).
9. **[interface] Limits and defaults.**
   - Limits: `MAX_SMOKE_VOLUMES = 4`, `MAX_SMOKE_SOURCES = 4` per volume, `MAX_SMOKE_PUSHES = 8`, `resolution` clamped to `[16, 160]` cells along the box's longest side.
   - The march's group: 14 sampled textures (the fog's 4, depth, accumulation, 4 density, 4 velocity), 2 storage textures, 5 storage buffers, 7 uniforms, 3 samplers. Smoke bindings start at 30.
   - `SmokeRenderOptions` defaults: 96 steps, 6 shadow steps, step scale 0.75, temporal on, blend 0.25.
   - The plume defaults to resolution 96 (48 × 96 × 48 in its 0.4 × 0.8 × 0.4 m box), the layers to 64.
10. **Pass labels:**
    - `Smoke/Fill`, `Smoke/Sources`, `Smoke/Forces`, `Smoke/Advect`, `Smoke/Pressure`;
    - `VolumetricFog/SmokeMarch`, `VolumetricFog/SmokeTemporal`;
    - the composite stays `VolumetricFog/Composite`.
11. **Boxes are closed.** The plume's box is tall, and its default dissipation fades the steam before the lid (Review Focus 3). Every face fades density over `edge_fade` of the box, so no outline shows.
12. **Example additions** beyond the design's list: `?stats=1` logs the frame interval and the profile every 240 frames (as `spot-lights` and `taa` do), and `window.kansei.set/view` serve scripted checks.

## Review Focus

1. The camera inside a smoke box. The room-sized layers volume always contains it, and the user can zoom into the plume. Smoke should render from the camera outward, with no hole and no flipped interval. Task 3: `the_camera_inside_a_volume_sees_the_smoke_around_it`.
2. Geometry standing in the smoke (the cup and table), at half resolution. The march should stop at the surface, smoke behind the cup should be hidden, and no halo should show at the rim. Task 3: `smoke_behind_a_surface_is_hidden`.
3. Steam reaching the top of the plume's closed box. It should fade before the lid, not pile into a flat sheet under it. Task 5: `the_default_plume_fades_before_the_lid`.
4. A mouse button released outside the canvas or over the panel. `mouseup` is only heard on the canvas, so `is_dragging` would stick and the smoke would never stir again. The next mousemove with no button held should end the drag. Task 10: `a_mousemove_with_no_button_ends_the_drag`.
5. Violent stirring, large steps and long runs. The gas should stay finite, with velocity capped and the mouse still effective after minutes. Task 10: `violent_stirring_stays_finite_and_capped`, `the_mouse_still_stirs_after_a_long_run`.

---

## File Structure

All paths are under `rust/kansei-core/src/` unless shown in full.

- `simulations/smoke/`:
  - `mod.rs`: docs, credits, re-exports, `MAX_*`, `CURL_NOISE_WGSL`, `shaders_validate_and_the_structs_match`.
  - `volume.rs`: `SmokeVolume`, motion settings, look, sources, fill, `SmokeVolumeGpu`, box maths, `smoke_phase`.
  - `mouse.rs`: `SmokeMouse` and `SmokePushes`.
  - `simulation.rs`: `SmokeSimulation`, `SmokeHandle`, grids, fill.
  - `solver.rs`: the Fluid and Curl passes.
  - `waves.rs`: the Waves density's CPU mirror.
  - `shaders/`: `smoke_common.wgsl` (`SmokeVolumeData`, `SmokeStep`, box maths, edge fade, `smokeWavesDensity`, `smokePhase`), `curl_noise.wgsl`, `smoke_fill.wgsl`, `smoke_solver.wgsl`.
- `shaders/volumetric_smoke_{march,temporal,composite}.wgsl`. These are Rust only. The composite's entry `compositeSmoke` reuses bindings 0-6 and adds 7 (light) and 8 (aux).
- `postprocessing/effects/volumetric_smoke.rs`: `SmokeRenderOptions`, `SmokeRenderer`, `SmokeFrameGpu`.
- Modified:
  - `simulations/mod.rs`;
  - `shaders/volumetric_fog_inject.wgsl:52-128` and `volumetric_fog_spot.wgsl`;
  - `postprocessing/effects/volumetric_fog.rs`: struct 399-437, `new` 440, `reset_history` 479, `init_gpu` 607-864, `shader_sources` 928, `render` 948-1179 (smoke between steps 4 and 5), size asserts 1214;
  - `postprocessing/effects/mod.rs:18,39`;
  - `controls/camera_controls.rs:88-131`.
- Tests: `rust/kansei-core/tests/smoke_gpu.rs` (simulation) and `rust/kansei-core/tests/smoke_render_gpu.rs` (rendering and seam).
- Example: `rust/kansei-wasm/examples/smoke/{Cargo.toml, README.md, src/lib.rs, www/index.html}`, plus `rust/Cargo.toml:10-38` (`exclude`).
- Docs: `README.md:133-134` and `examples/index.html:169, 308-336`.

## Verification recipes

`$EX` is `rust/kansei-wasm/examples/smoke`.

- **R1 Build and serve.**
  - Build with `(cd $EX && wasm-pack build --target web --release)`.
  - Choose a port that `lsof -i :<port>` shows free (8791 if it is). Serve `$EX` from it with `Cache-Control: no-store`:
    ```bash
    (cd $EX && python3 -c 'import http.server as h, socketserver as s
    class H(h.SimpleHTTPRequestHandler):
        def end_headers(self): self.send_header("Cache-Control", "no-store"); super().end_headers()
    s.TCPServer.allow_reuse_address = True; s.TCPServer(("127.0.0.1", 8791), H).serve_forever()') &
    ```
  - Stop only that server when done.
- **R2 Browser.**
  - `export CHROME_DEVTOOLS_AXI_SESSION=smoke-<task> CHROME_DEVTOOLS_AXI_CHROME_ARGS="--enable-unsafe-webgpu --enable-gpu --ignore-gpu-blocklist"`.
  - Run `chrome-devtools-axi resize 1920 1080`, `open "http://127.0.0.1:8791/www/?dpr=1&…"`, `wait 4000`.
  - Save screenshots outside the repo. Read the `console`: any WGSL or validation error fails the check.
  - Run `open about:blank` after each capture and `chrome-devtools-axi stop` at the end.
- **R3 GPU timing.**
  - R2 with `--disable-gpu-vsync --disable-frame-rate-limit --enable-webgpu-developer-features` added and `&stats=1`.
  - Wait 4 s or more after any switch, then read three `smoke stats` windows (frame interval, drawing-buffer size, sim steps a frame, `take_profile().report()`).
  - Report their median and the GPU (`system_profiler SPDisplaysDataType | grep Chipset`).
  - Other sessions share this GPU: compare levels inside one page through `window.kansei.set('motion', n)`.
- **R4 Scripted stir.** `chrome-devtools-axi eval` a `setInterval` that dispatches canvas `mousemove` events (`clientX`/`clientY`) round a 200 px circle through the plume every 16 ms. Clear it afterwards.

## Checkpoints (stop rules)

- **Checkpoint B (seam), end of Task 4.** A static density, rendered and lit through the fog, shows no seam. The criteria are in Task 4.
- **Checkpoint A (performance), end of Task 6.** The Fluid level, default grid, fog on, layers off, TAA on, at 1920×1080 on this Mac. Under R3, every window's frame interval must be ≤ 15.0 ms (a 10% margin under 16.7), idle and under R4.
  - You may tune `max_steps`, `shadow_steps`, `step_scale`, `pressure_iterations`, and the default resolution down to 64, provided the plume still looks right: wisps finer than the froxels, no blocky cells or banding at `?aa=off`.
- **On failure:** stop, and do not start the next task. Report to the owner the numbers (test output, R3 windows, the per-pass report), the screenshots' paths, the GPU, and what you tried.

---

### Task 1: The fog's lights as per-light functions (the fog unchanged)

**Files:** Modify `shaders/volumetric_fog_inject.wgsl:52-128` and `shaders/volumetric_fog_spot.wgsl`. Test in `postprocessing/effects/volumetric_fog.rs` (`mod tests`).

**Interfaces:**
- Produces (WGSL, in the inject chain):
  - `struct FogLight { toLight: vec3f, radiance: vec3f }`;
  - `fn fogDirLight(i: u32, worldPos: vec3f) -> FogLight`, with `toLight = -normalize(direction)`;
  - `fn fogPointLight(i: u32, worldPos: vec3f) -> FogLight`, radiance 0 past the radius, holding the TS-patched line;
  - `fn fogSpotLight(i: u32, worldPos: vec3f, minDist: f32) -> FogLight`, with `volumetricScale` and the atlas visibility applied;
  - `fn fogInScatter(worldPos: vec3f, viewDir: vec3f, spotMinDist: f32) -> vec3f`, the fog's in-scatter per unit density before albedo: `params.ambient + Σ dir·HG + Σ point·HG + (spots unless params.skipSpots) + skyAmbient(viewDir, worldPos)`.
- `main` becomes `density * fogInScatter(…) * media.albedo`, and `spotInScatter` (same signature) is built on `fogSpotLight`.

- [ ] **Step 1: Write the characterization test** `the_injection_lights_a_froxel_as_the_formula_says`. It must pass both before and after the refactor.
  - Setup:
    - `FroxelGridOptions { grid_w: 16, grid_h: 16, grid_d: 32, near: 0.5, far: 60.0, temporal: false, blend_factor: 1.0 }`, `base_density: 0.02`, `height_falloff: 0.0`, `fog_height: 100.0`, `anisotropy: 0.6`, `ambient: (0.1, 0.1, 0.1)`;
    - an unshadowed sun with direction `(-0.6, -0.5, -0.62)`, colour `(1.0, 0.9, 0.8)` and intensity 2;
    - an unshadowed point light at `(1.5, 5.0, -6.0)` with radius 12;
    - the camera at `(0, 5, 0)` looking down −z.
  - The CPU computes `density * (ambient + Σ light · HG(dot(viewDir, toLight), 0.6) · falloff) * albedo` at the froxel centres of column (8, 8), integrated as `froxel_accumulate.wgsl` does.
  ```rust
  for z in [0u32, 6, 12] { let (got, want) = (accum_at(&v, 8, 8, z), expected_accum(z)); assert!((got - want).abs() <= 1e-3 * want.max(1e-4), "slice {z}: {got} vs {want}"); }
  ```
- [ ] **Step 2: Run** `cargo test -p kansei-core --lib the_injection_lights_a_froxel -- --nocapture` on the current shader. Expected: PASS. If it fails, fix the CPU formula, not the shader.
- [ ] **Step 3: Refactor** the two shaders into the functions above.
- [ ] **Step 4: Run** `cargo test -p kansei-core --lib volumetric_fog`. Expected: all PASS, including `raymarched_spot_shafts_keep_thin_shadows`.
- [ ] **Step 5: Check the TS side.**
  - `pnpm test` and `pnpm exec tsc` both PASS.
  - Serve `pnpm dev --port <free> --strictPort`, and with R2 load `examples/index_volumetric_fog.html` and `examples/index_spot_lights.html`.
  - Expected: no console error, and the same fog as a screenshot taken on `development`.
- [ ] **Step 6: Commit:** `volumetric fog: the injection's lights as per-light functions (fogDirLight, fogPointLight, fogSpotLight), for the smoke to share`

### Task 2: Smoke volumes, grids, fill and handle

**Files:**
- Create `simulations/smoke/{mod.rs, volume.rs, mouse.rs, simulation.rs}` and `shaders/{smoke_common.wgsl, smoke_fill.wgsl, curl_noise.wgsl}`. Curl noise is written here because `smoke_common` includes it; it is tested in Task 7.
- Modify `simulations/mod.rs`.
- Test: `volume.rs` units and `tests/smoke_gpu.rs`.

**Interfaces:** Produces (every later task uses these names):
```rust
pub const MAX_SMOKE_VOLUMES: usize = 4;  pub const MAX_SMOKE_SOURCES: usize = 4;  pub const MAX_SMOKE_PUSHES: usize = 8;
pub const MIN_SMOKE_RESOLUTION: u32 = 16;  pub const MAX_SMOKE_RESOLUTION: u32 = 160;
pub const CURL_NOISE_WGSL: &str; // fn curlNoisePotential(p: vec3f, time: f32) -> vec3f;  fn curlNoise(p: vec3f, time: f32) -> vec3f
pub struct SmokeVolume { pub center: Vec3, pub size: Vec3, pub yaw: f32, pub resolution: u32,
    pub motion: SmokeMotion, pub look: SmokeLook, pub sources: Vec<SmokeSource>, pub fill: SmokeFill }
impl SmokeVolume {
    pub fn new(center: Vec3, size: Vec3, motion: SmokeMotion) -> Self;  // resolution 96, SmokeLook::default(), no sources, SmokeFill::None
    pub fn rising_from(source: SmokeSource, height: f32, motion: SmokeMotion) -> Self; // box height x height/2 x height/2, source at the bottom centre + 2 cells
    pub fn grid_dims(&self) -> [u32; 3];  // longest side = resolution.clamp(16, 160), others in proportion, each >= 8
    pub fn cell_size(&self) -> f32;
    pub fn to_local(&self, world: Vec3) -> Vec3;  // [0,1]^3 inside; yaw about +Y as LocalFogVolume
    pub fn to_world(&self, local: Vec3) -> Vec3;
    pub fn ray_interval(&self, origin: Vec3, dir: Vec3) -> Option<(f32, f32)>; // t0 clamped to 0 (origin inside); None if missed
}
pub enum SmokeMotion { Waves(WavesSettings), CurlNoise(CurlSettings), Fluid(FluidSmokeSettings) }
impl SmokeMotion { pub fn waves() -> Self; pub fn curl_noise() -> Self; pub fn fluid() -> Self; }
// each settings struct and SmokeLook impl Default with these values:
pub struct WavesSettings { pub amplitude: f32, pub wavelength: f32, pub speed: f32, pub spread: f32, pub relax_time: f32 }  // 0.03, 0.25, 1.5, 0.25, 1.5
pub struct CurlSettings { pub strength: f32, pub scale: f32, pub speed: f32, pub rise: f32 }                           // 0.15, 0.15, 0.3, 0.25
pub struct FluidSmokeSettings { pub buoyancy: f32, pub weight: f32, pub cooling: f32, pub vorticity: f32,
    pub pressure_iterations: u32, pub max_speed: f32, pub ambient_draft: f32, pub draft_scale: f32 }                // 1.5, 0.1, 0.8, 2.0, 40, 3.0, 0.0, 1.0
pub struct SmokeLook { pub density_scale: f32, pub albedo: Vec3, pub extinction: f32, pub dissipation: f32, pub edge_fade: f32 } // 1, 0.95, 1, 0.6, 0.1
pub struct SmokeSource { pub position: Vec3, pub radius: f32, pub rate: f32, pub temperature: f32, pub velocity: Vec3 }
pub enum SmokeFill { None, Box { density: f32, temperature: f32 }, Sphere { center: Vec3, radius: f32, density: f32, temperature: f32 } } // + Layers, Task 9
pub struct SmokeMouse { pub origin: Vec3, pub direction: Vec3, pub push: Vec3, pub radius: f32 } // world ray; push in m/s
impl SmokeMouse { pub const NONE: SmokeMouse; }                                                   // from_screen: Task 10
#[derive(Clone, Copy, PartialEq, Eq, Debug)] pub struct SmokeVolumeId(usize);
pub struct SmokeGridTextures<'a> { pub dims: [u32; 3], pub scalars: &'a wgpu::Texture, pub velocity: Option<&'a wgpu::Texture> }
#[derive(Clone)] pub struct SmokeHandle { /* Arc<Mutex<…>>: each enabled volume's SmokeVolumeGpu and current views, pushes, time, sim seconds since the last render */ }
impl SmokeSimulation {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self;
    pub fn add(&mut self, volume: SmokeVolume) -> SmokeVolumeId;   // allocates, fills (Smoke/Fill), enabled; panics past MAX_SMOKE_VOLUMES
    pub fn volume(&self, id: SmokeVolumeId) -> &SmokeVolume;
    pub fn volume_mut(&mut self, id: SmokeVolumeId) -> &mut SmokeVolume; // look, sources, centre: read every step and frame
    pub fn set_motion(&mut self, id: SmokeVolumeId, motion: SmokeMotion); // reallocates, refills
    pub fn set_enabled(&mut self, id: SmokeVolumeId, enabled: bool);       // enabling refills
    pub fn is_enabled(&self, id: SmokeVolumeId) -> bool;
    pub fn reset(&mut self, id: SmokeVolumeId);                            // re-reads resolution and size, refills
    pub fn step(&mut self, dt: f32, mouse: &SmokeMouse);                   // dt <= 0: no simulation, only refreshes the handle (box or look edits)
    pub fn time(&self) -> f32;
    pub fn handle(&self) -> SmokeHandle;
    pub fn grid_textures(&self, id: SmokeVolumeId) -> Option<SmokeGridTextures<'_>>; // None for Waves
}
```
`pub(crate)` GPU structs: `SmokeVolumeGpu` ↔ WGSL `SmokeVolumeData` (box, dims, motion kind, look, waves and curl settings, sources), and `SmokeStepGpu` ↔ `SmokeStep` (one 256-byte slot per volume). Here `step` only advances time, ages pushes and refreshes the handle. Tasks 5, 7 and 10 fill it in.

- [ ] **Step 1: Write the CPU tests** (`volume.rs`):
  ```rust
  // the_box_maps_points_and_rays_with_its_yaw
  let v = SmokeVolume { yaw: FRAC_PI_2, ..SmokeVolume::new(Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.5), SmokeMotion::fluid()) };
  assert_close(v.to_local(v.center), Vec3::new(0.5, 0.5, 0.5));
  assert_close(v.to_world(v.to_local(Vec3::new(1.2, 1.3, 0.4))), Vec3::new(1.2, 1.3, 0.4));
  assert!(v.to_local(Vec3::new(1.0, 1.0, 0.9)).x.max(v.to_local(Vec3::new(1.0, 1.0, 0.9)).z) <= 1.0); // the long axis now along world z
  let (a, b) = v.ray_interval(Vec3::new(1.0, 1.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).unwrap(); assert_close2((a, b), (4.0, 6.0));
  assert_eq!(v.ray_interval(v.center, Vec3::new(0.0, 1.0, 0.0)).unwrap().0, 0.0);
  assert!(v.ray_interval(Vec3::new(9.0, 9.0, 9.0), Vec3::new(1.0, 0.0, 0.0)).is_none());
  // grid_dims_follow_the_box_and_clamp_the_resolution
  assert_eq!(SmokeVolume::new(Vec3::ZERO, Vec3::new(0.4, 0.8, 0.4), SmokeMotion::fluid()).grid_dims(), [48, 96, 48]);
  let mut c = SmokeVolume::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), SmokeMotion::fluid());
  c.resolution = 4096; assert_eq!(c.grid_dims(), [160; 3]);  c.resolution = 1; assert_eq!(c.grid_dims(), [16; 3]);
  ```
- [ ] **Step 2: Write the GPU test** `a_sphere_fill_fills_its_cells` (`tests/smoke_gpu.rs`). Add a helper `read_scalars` that decodes f16 as `volumetric_fog.rs::f16_to_f32` does.
  - Setup: a 32³ box of 1 m with `SmokeFill::Sphere { center: box centre, radius: 0.25, density: 2.0, temperature: 1.0 }`.
  - Assert: cell (16, 16, 16) is within 0.01 of (2, 1), cell (2, 2, 2) is (0, 0), and a Waves volume's `grid_textures` is `None`.
- [ ] **Step 3: Run** `cargo test -p kansei-core --lib smoke` and `cargo test -p kansei-core --test smoke_gpu`. Expected: FAIL (no module `smoke`).
- [ ] **Step 4: Implement:**
  - the types and their packing;
  - `smoke_fill.wgsl` (entry `fill`);
  - the grids, per decision 4;
  - the handle;
  - `shaders_validate_and_the_structs_match`, covering `SmokeVolumeData` and `SmokeStep`.
- [ ] **Step 5: Run** both commands. Expected: PASS.
- [ ] **Step 6: Commit:** `smoke: volumes, their grids and fill, and the handle the fog marches them through`

### Task 3: Smoke marched and composited through the fog

**Files:**
- Create `postprocessing/effects/volumetric_smoke.rs`, `shaders/volumetric_smoke_march.wgsl` and `shaders/volumetric_smoke_composite.wgsl`.
- Modify `volumetric_fog.rs`, `effects/mod.rs`, `smoke_common.wgsl` (`smokePhase`) and `volume.rs` (`smoke_phase`).
- Test: `tests/smoke_render_gpu.rs` and a `volume.rs` unit.

**Interfaces:**
- Consumes:
  - Task 1's per-light functions;
  - Task 2's `SmokeHandle` and `SmokeVolumeGpu`;
  - from the fog: `fog_params`, `media_params`, `ComputeShadows::layout_entries()`/`entries()`, `grid.accum_view()`, `accum_sampler`, `bind_groups`.
- Produces:
  ```rust
  pub struct SmokeRenderOptions { pub max_steps: u32, pub shadow_steps: u32, pub step_scale: f32, pub temporal: bool, pub blend: f32 } // Default: 96, 6, 0.75, true, 0.25
  impl VolumetricFogEffect { pub fn set_smoke(&mut self, smoke: Option<&SmokeHandle>); } // + pub field smoke_render: SmokeRenderOptions
  pub(crate) fn smoke_phase(cos_theta: f32, g: f32, tau: f32) -> f32; // volume.rs; mirrors WGSL smokePhase
  ```
  - The march module is the inject chain (as `SHAFTS_WGSL` is) + `smoke_common.wgsl` + `volumetric_smoke_march.wgsl`, with entry `smokeMarch`.
  - The smoke composite is `froxel_common` + `volumetric_fog_composite` + `volumetric_smoke_composite`, with entry `compositeSmoke`. It is used only while the handle has an enabled volume.
  - Add both modules to `shader_sources()` and `SmokeFrame` to the size asserts.
  - Until Task 4 the composite reads the march's own targets.

- [ ] **Step 1: Write the failing tests.**
  - `volume.rs`, `smoke_phase_is_the_fogs_where_thin`:
    ```rust
    for c in [-0.9f32, 0.0, 0.7] { assert_eq!(smoke_phase(c, 0.6, 0.0), hg(c, 0.6)); }
    assert!(smoke_phase(0.3, 0.6, 2.0) < smoke_phase(0.3, 0.6, 1.0));
    assert!(smoke_phase(0.3, 0.6, 2.0) > hg(0.3, 0.6) * (-2.0f32).exp());
    ```
  - `smoke_render_gpu.rs`. Every test uses `smoke_render.temporal = false`, 64×32, the camera at (0, 1, 0) looking down −z, a constant `input` of 0.3, and depth written by a helper like `spot_fog_scene`'s. It compares the centre 8×4 pixels' mean with a CPU march of 4000 steps. Unless a test says otherwise: `SmokeFill::Box` volumes, smoke `albedo` 1, `extinction` 1, `edge_fade: 0.0`, fog `base_density: 0`, fog `ambient` 1, no lights.
    - `smoke_with_no_density_leaves_the_fog_as_it_was`: fog plus a `SmokeFill::None` volume, against fog alone. Max abs difference < 1e-3 × the max value.
    - `thin_smoke_scatters_as_the_fog_does`: uniform fog of 0.02/m, against fog 0 plus a 200 m smoke box of 0.02/m. Same albedo, an unshadowed sun, ambient 0.05, a depth plane at 20 m. Relative difference < 3%.
    - `dense_smoke_hides_the_fog_behind_it`: ambient-lit fog of 0.05/m, and a black smoke box (`albedo` 0, 50/m) over z ∈ [−7, −5]. The centre equals the fog's in-scatter over 0-5 m within 3%.
    - `fog_in_front_veils_the_smoke`: fog of 0.1/m, a smoke box of 2/m over z ∈ [−9, −8], albedo 1, ambient 1. Within 3% of the CPU fog-and-smoke integral.
    - `the_smoke_is_darker_away_from_the_light`: no fog or ambient, a sun travelling −x, and a sphere fill of 20/m with radius 0.5 m at z = −3. The mean of its right third is > 2× its left third's.
    - `the_camera_inside_a_volume_sees_the_smoke_around_it` (Review Focus 1): a 4 m box of 0.5/m round the camera, sky beyond. Centre ≈ `1 − e^{−1} + 0.3·e^{−1}` within 3%, and every pixel finite.
    - `smoke_behind_a_surface_is_hidden` (Review Focus 2): a box of 1/m over z ∈ [−4.5, −0.5], and a depth plane at 1 m over the left half. Left centre ≈ `1 − e^{−0.5} + 0.3·e^{−0.5}`, right centre ≈ `1 − e^{−4}`, both within 3%. No NaN.
- [ ] **Step 2: Run** `cargo test -p kansei-core --test smoke_render_gpu`. Expected: FAIL (no `set_smoke`).
- [ ] **Step 3: Implement** decisions 3, 5 and 6, with the `VolumetricFog/SmokeMarch` pass.
  - `SmokeFrameGpu` is one write a frame.
  - Bind groups come from the fog's `BindGroupCache`, so swapping textures cost nothing to follow.
  - Missing grids bind 1³ stand-ins.
- [ ] **Step 4: Run** `cargo test -p kansei-core --test smoke_render_gpu` and `cargo test -p kansei-core --lib`. Expected: PASS.
- [ ] **Step 5: Commit:** `volumetric fog: smoke volumes marched in one pass through the fog's lights and froxels, composited with the fog`

### Task 4: The smoke's history, the example's room, Checkpoint B

**Files:**
- Create `shaders/volumetric_smoke_temporal.wgsl` and `rust/kansei-wasm/examples/smoke/{Cargo.toml, src/lib.rs, www/index.html}`.
- Modify `volumetric_smoke.rs`, `volumetric_fog.rs` (`reset_history`) and `rust/Cargo.toml` (`exclude`).
- Test: `tests/smoke_render_gpu.rs`.

**Interfaces:**
- Produces the `VolumetricFog/SmokeTemporal` pass (decision 7).
- Produces the crate `kansei-wasm-smoke` (from `examples/volumetric-fog/Cargo.toml`, renamed). It exports `start(canvas_id: &str)`, `set(key: &str, value: f32)` (only `"fogDensity"` for now) and `view() -> String`.
- The scene:
  - a dark room about 6 × 3 × 5 m, built from `BoxGeometry` with `Material::standard_lit`;
  - one window with a cross-shaped frame;
  - a low sun (`cast_shadow`, `renderer.enable_shadows(2048)`) shining through it;
  - a table, and a cup (`CylinderGeometry`).
- Fog: `base_density: 0.03`, `height_falloff: 0.0`, a temporal grid with `far: 20.0`, `set_shadow_map(renderer.shadow_map())`, `update_lights`.
- Smoke: `SmokeVolume::rising_from(<the cup's steam source>, 0.8, SmokeMotion::CurlNoise(CurlSettings { strength: 0.0, rise: 0.0, ..Default::default() }))` with a `Sphere` fill above the cup. It is static: nothing steps it yet.
- Effects: `[VolumetricFogEffect, TemporalAAEffect unless ?aa=off, ToneMapEffect]`, with `CameraControls::from_canvas(…).with_mouse_pan(…)`.

- [ ] **Step 1: Write the failing tests.**
  - `a_moved_volume_leaves_no_ghost`: temporal on, 30 frames of a sphere-filled box at x = −1. Then move `center` by +2 m, call `sim.step(0.0, &SmokeMouse::NONE)`, and render one frame. The old location's mean is within 5% of a background-only render.
  - `history_survives_a_resize`: 5 frames at 64×32, then one at 96×48. Every pixel is finite, and the result is within 5% of a `temporal: false` render.
- [ ] **Step 2: Run** `cargo test -p kansei-core --test smoke_render_gpu`. Expected: these two FAIL.
- [ ] **Step 3: Implement** the temporal pass and its resets. Then build the example scaffold. Its `index.html` is `examples/depth-of-field/www/index.html` without the panel, importing `init, { start, set, view }` from `'../pkg/kansei_wasm_smoke.js'` and setting `window.kansei = { set, view }`.
- [ ] **Step 4: Run** the tests (PASS), then R1's build (no warnings from the example).
- [ ] **Step 5: Checkpoint B.**
  - R1 and R2. Capture `?aa=off` and `?aa=taa`, each with the fog at its default, at `kansei.set('fogDensity', 0.2)` and at 0. Then orbit so the sun's shaft crosses the smoke, and capture again.
  - Pass only if all of these hold:
    - (a) no box outline or face is visible;
    - (b) where the smoke thins into the fog there is no halo or dark rim;
    - (c) in thick fog the smoke fades with distance, and dense smoke hides the shaft behind it;
    - (d) the window frame's shadow cuts through the smoke;
    - (e) at `?aa=off` there is no static noise or banding;
    - (f) the console is clean.
  - Otherwise stop (Checkpoints).
- [ ] **Step 6: Commit:** `volumetric fog: the smoke's own history, reprojected, clamped and reset on cuts and resizes; examples: smoke, a room with a static plume`

### Task 5: The Fluid level

**Files:**
- Create `simulations/smoke/solver.rs` and `shaders/smoke_solver.wgsl`.
- Modify `simulation.rs` and `mod.rs` (validation list).
- Test: `tests/smoke_gpu.rs`.

**Interfaces:**
- Consumes Task 2's grids, `SmokeStepGpu` slots and `SmokeMouse` (unused until Task 10).
- Produces `step` for Fluid volumes:
  - Passes: `Smoke/Sources`, `Smoke/Forces` (vorticity, buoyancy, ambient draft, mouse, speed cap), `Smoke/Advect` (scalar entry points `advectScalarsForward` and `advectScalarsCorrect` take the velocity from `smokeVelocityAt`, which Task 7 extends to Curl), and `Smoke/Pressure` (divergence, Jacobi, gradient subtraction, walls).
  - Sources add `rate * dt * w(r)` to density (with `w ≤ 1`), relax temperature toward theirs, and blend velocity toward theirs.
  - Dissipation is `density *= exp(-dissipation * dt)`.
  - `pressure_iterations: 0` skips the projection (tests).

- [ ] **Step 1: Write the failing tests:**
  ```rust
  // the_projection_leaves_no_divergence: 16^3 box 1 m, Fluid { buoyancy: 0, weight: 0, vorticity: 0, cooling: 0, pressure_iterations: 1000, .. },
  // dissipation 0; velocity written as v = (p - c) * exp(-|p - c|^2 / 0.04) on the MAC faces; one step of 1/60
  assert!(rms_mac_div(&after) < 0.02 * rms_mac_div(&before));
  // a_hot_blob_rises: box 1 x 1.5 x 1 at resolution 48, Sphere 0.3 m above the floor, radius 0.15, density 1, temperature 1; defaults; 90 steps of 1/60
  assert!(com.y - com0.y > 0.15);  assert!((com.x - com0.x).abs() < 0.02 && (com.z - com0.z).abs() < 0.02);
  // the_gas_stays_finite_and_in_its_box: the same blob, dissipation 0, 600 steps
  assert!(all_finite && min_density >= 0.0 && max_density <= max0 * 1.001);  assert!((mass / mass0 - 1.0).abs() < 0.10);
  // the_default_plume_fades_before_the_lid (Review Focus 3): rising_from(SmokeSource { radius: 0.03, rate: 4.0, temperature: 1.0, velocity: (0, 0.15, 0), .. }, 0.8, fluid()), 1200 steps
  assert!(mean_density(top 10% of rows) < 0.1 * mean_density(source column, lower third));
  assert!(max_density <= 1.05 * rate / dissipation && max_temperature <= 1.001);
  ```
- [ ] **Step 2: Run** `cargo test -p kansei-core --test smoke_gpu`. Expected: the four FAIL.
- [ ] **Step 3: Implement** `solver.rs` and `smoke_solver.wgsl`, with one entry point per stage and two bind groups per ping-pong.
- [ ] **Step 4: Run** `cargo test -p kansei-core`. Expected: PASS. If `fades_before_the_lid` needs `SmokeLook::default().dissipation` or the Fluid defaults tuned, tune them and say so in the commit.
- [ ] **Step 5: Commit:** `smoke: the Fluid level (MAC grid, buoyancy, vorticity confinement, MacCormack scalars, Jacobi projection, closed walls)`

### Task 6: Fluid in the example, reprojection by the gas's velocity, Checkpoint A

**Files:** Modify `shaders/volumetric_smoke_march.wgsl` and the example's `lib.rs`. Test: `tests/smoke_render_gpu.rs`.

**Interfaces:**
- Produces the plume on `SmokeMotion::fluid()`, emitting from the cup, stepped as `for _ in 0..fixed.advance(frame.dt as f64) { smoke.step(fixed.step as f32, &SmokeMouse::NONE) }` with `fixed = pacing::FixedStep::new(1.0 / 60.0)`.
- The example sets `fog.time` every frame and calls `kansei_wasm::perf::profile(&mut renderer)` every frame.
- `?stats=1` sets `renderer.set_profiling(true)` and logs every 240 frames: `smoke stats: {w}x{h}, {interval:.2} ms between frames, {steps:.2} sim steps a frame\n{take_profile().report()}`.

- [ ] **Step 1: Write the failing test** `the_history_follows_the_gas`.
  - Setup: a 2 m Fluid box with `pressure_iterations: 0`, `buoyancy: 0`, `weight: 0` and `vorticity: 0`. A closed box's projection would remove a uniform flow entirely, hence no projection. Write a uniform u = 0.5 m/s into its velocity grid, put a sphere fill of radius 0.2 m at the centre, and turn temporal on.
  - Step once per frame for 20 frames.
  - Assert: the blob's centroid column is within 1 px of a `temporal: false` render's.
- [ ] **Step 2: Run** it. Expected: FAIL (the history lags).
- [ ] **Step 3: Implement** the velocity reprojection (decision 6), and the Fluid plume and `?stats=1` in the example.
- [ ] **Step 4: Run** `cargo test -p kansei-core` (PASS) and R1's build.
- [ ] **Step 5: Checkpoint A.**
  - R3 idle and under R4, with `?dpr=1&stats=1`. Confirm the log says `1920x1080`.
  - Pass if every window is ≤ 15.0 ms and a close R2 view at `?aa=off` shows wisps with no blocky cells.
  - Keep the per-pass report for the README and PR. On failure, see Checkpoints.
- [ ] **Step 6: Commit:** `volumetric fog: the smoke's history follows the gas's velocity; examples: smoke on the Fluid level, ?stats=1`

### Task 7: The Curl noise level

**Files:** Modify `solver.rs`, `smoke_solver.wgsl` (`smokeVelocityAt` for Curl), `simulation.rs` and `volumetric_smoke_march.wgsl` (Curl velocity for reprojection). Test: `tests/smoke_gpu.rs`.

**Interfaces:** Produces `step` for Curl volumes: `Smoke/Sources` and `Smoke/Advect` with the velocity `curlNoise(p / scale, time * speed) * strength + rise * up` (the mouse comes in Task 10), then dissipation. There is no velocity grid.

- [ ] **Step 1: Write the failing tests:**
  ```rust
  // the_curl_noise_velocity_is_divergence_free: a test compute shader over CURL_NOISE_WGSL writes curlNoise, and as a control
  // the gradient of curlNoisePotential(p).x, on a 32^3 lattice of spacing h = 1/8 (noise units), time 0.37
  let ratio = |v: &[[f32; 3]]| mean_abs_central_div(v, h) * h / mean_len(v);
  assert!(ratio(&curl) < 0.02, "curl {}", ratio(&curl));  assert!(ratio(&gradient) > 0.2, "control {}", ratio(&gradient));
  // curl_noise_carries_the_gas_up_and_keeps_it: Curl { rise: 0.2, .. }, sphere fill, dissipation 0, 60 steps of 1/60
  assert!((com.y - com0.y - 0.2).abs() < 0.06);  assert!((mass / mass0 - 1.0).abs() < 0.10);
  ```
- [ ] **Step 2: Run** `cargo test -p kansei-core --test smoke_gpu curl`. Expected: the second FAILS. If the first fails too, fix Task 2's noise.
- [ ] **Step 3: Implement** the Curl branches.
- [ ] **Step 4: Run** `cargo test -p kansei-core`. Expected: PASS.
- [ ] **Step 5: Commit:** `smoke: the Curl noise level (the Fluid level's scalar advection, by curl noise and a rise)`

### Task 8: The Waves level

**Files:** Create `simulations/smoke/waves.rs`. Modify `smoke_common.wgsl` (`smokeWavesDensity`) and `volumetric_smoke_march.wgsl` (Waves in `smokeMedium` and the shadow march). Test: `waves.rs` units.

**Interfaces:**
- Produces `pub(crate) fn waves_density(volume: &SmokeVolume, p: Vec3, time: f32) -> f32`, the CPU mirror of WGSL `smokeWavesDensity(v: SmokeVolumeData, p: vec3f, time: f32) -> f32`.
- Per source, a column rises from it:
  - it widens by `spread` per metre;
  - its axis sways by sums of sin/cos of height and time (`amplitude`, `wavelength`, `speed`);
  - it fades with height at a rate set by `dissipation`;
  - it is zero outside the box.
- The exact formulas are the builder's. A Waves volume's step-length cell is `size / resolution`.

- [ ] **Step 1: Write the failing tests:**
  ```rust
  // waves_plume_rises_from_its_source_and_fades
  let d = |dy: f32, dx: f32| waves_density(&v, src + Vec3::new(dx, dy, 0.0), 0.0);
  assert!(d(0.05, 0.0) > d(0.5, 0.0) && d(0.5, 0.0) > 0.0);  assert!(d(0.05, 0.15) < 0.01 * d(0.05, 0.0));
  // waves_density_is_zero_outside_the_box_and_never_negative: 10_000 hashed points in a box twice the size
  // waves_sway_with_time: the density-weighted x of the slice 0.4 m up moves > 0.2 * amplitude between t = 0 and 1,
  //   and stays within amplitude + radius + spread * 0.4 of the source
  // waves_density_matches_its_wgsl (GPU, skipped without adapter): 64 points, |gpu - cpu| <= 1e-3 * cpu + 1e-5
  ```
- [ ] **Step 2: Run** `cargo test -p kansei-core --lib waves`. Expected: FAIL.
- [ ] **Step 3: Implement** both versions and the march's Waves branch.
- [ ] **Step 4: Run** `cargo test -p kansei-core`. Expected: PASS.
- [ ] **Step 5: Commit:** `smoke: the Waves level (a plume swayed by sums of sines, evaluated where the march samples it)`

### Task 9: Hanging layers, overlapping volumes, switching at runtime

**Files:** Modify `volume.rs` (`SmokeFill::Layers`), `smoke_fill.wgsl`, `smoke_common.wgsl` and `waves.rs` (layers on Waves), `smoke_solver.wgsl` (ambient draft) and the example's `lib.rs`. Tests: `tests/smoke_gpu.rs`, `tests/smoke_render_gpu.rs`, `waves.rs`.

**Interfaces:**
- Produces `SmokeFill::Layers { count: u32, thickness: f32, density: f32 }`: `count` soft horizontal sheets spread over the box's height, gently uneven in x and z. The same shape is the Waves density of a source-less volume with that fill.
- `ambient_draft` adds the force `ambient_draft * curlNoise(p / draft_scale, time * 0.05)` on Fluid.
- The example's layers volume:
  - room-sized and overlapping the plume;
  - resolution `(res * 2 / 3).max(16)`, no sources, `Layers { count: 3, thickness: 0.25, density: 0.6 }`;
  - `dissipation: 0.0`, `ambient_draft: 0.05`, `draft_scale: 1.5`;
  - enabled by `?layers=1`, and following the plume's motion level.

- [ ] **Step 1: Write the failing tests.**
  - `layers_fill_lies_in_horizontal_sheets` (GPU): the y-profile of the x-z mean density has exactly `count` local maxima, and the coefficient of variation along x in the middle sheet is < 0.3.
  - `waves_layers_lie_in_sheets` (CPU): the same check on `waves_density`.
  - `layers_drift_on_fluid_and_do_not_dissipate`: 300 steps with `ambient_draft: 0.05`. Mass is within 5%, and the grid's L1 change is > 5% of its L1.
  - `overlapping_volumes_march_as_one_medium` (render):
    - two coincident boxes of 0.25/m against one of 0.5/m: < 2%;
    - two boxes touching at x = 0 against one spanning both: no column of the middle row differs by > 2%.
  - `switching_levels_at_runtime_keeps_rendering` (render): 12 frames, each doing `set_motion` round Fluid → Waves → Curl and toggling a second volume with `set_enabled`, inside a validation error scope (`device.push_error_scope`, then pop). No error, and every pixel finite.
- [ ] **Step 2: Run** `cargo test -p kansei-core`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement**, then add the layers volume and `?layers=1` to the example.
- [ ] **Step 4: Run** `cargo test -p kansei-core` (PASS). Rebuild the example and check `?layers=1` with R2: the layers drift, and there is no seam where the plume meets them.
- [ ] **Step 5: Commit:** `smoke: hanging layers (no source, an ambient curl-noise draft on Fluid), overlapping volumes in one march; examples: smoke ?layers=1`

### Task 10: Mouse interaction

**Files:** Modify `mouse.rs`, `simulation.rs`, `smoke_solver.wgsl`, `volumetric_smoke_march.wgsl` (push displacement), `controls/camera_controls.rs` and the example's `lib.rs`. Tests: `mouse.rs` and `camera_controls.rs` units, and `tests/smoke_gpu.rs`.

**Interfaces:**
- Produces:
  ```rust
  impl SmokeMouse {
      /// `position`, `motion`: MouseVectors' `position` and `direction` (NDC, y down; direction = previous - current).
      /// push = the cursor's velocity in m/s at view depth `depth`, along the camera's right and up, times `strength`.
      pub fn from_screen(camera: &Camera, position: Vec2, motion: Vec2, dt: f32, strength: f32, radius: f32, depth: f32) -> SmokeMouse;
  }
  pub(crate) struct SmokePushes; // ring of MAX_SMOKE_PUSHES (origin, direction, push, radius, age); oldest dropped; aged by step's dt
  impl SmokePushes { pub(crate) fn displacement(&self, p: Vec3, relax_time: f32) -> Vec3; } // mirrors the march's
  impl CameraControls { pub fn is_dragging(&self) -> bool; } // orbit or pan drag, or touch, in progress (wasm); false natively
  pub(crate) fn drag_after_move(dragging: bool, panning: bool, buttons: u16) -> (bool, bool); // buttons & 3 == 0 ends both
  ```
- How each level takes the push:
  - Fluid: the force `push * falloff(distance to the ray / radius)`;
  - Curl: the same field added to the advecting velocity;
  - Waves: `step` records pushes, and the march displaces samples by `SmokePushes::displacement`.
- The example passes `SmokeMouse::NONE` while `controls.is_dragging()`, and otherwise `from_screen(&camera, m.position, m.direction, dt, strength, radius, <plume depth>)`, with strength 1.0 and radius 0.08 by default.

- [ ] **Step 1: Write the failing tests.**
  - Units:
    ```rust
    // the_mouse_maps_to_a_ray_and_a_push_in_the_scene: Camera::new(50.0, 0.1, 100.0, 16.0 / 9.0) at (0, 1, 3) looking at (0, 1, 0)
    let m = SmokeMouse::from_screen(&cam, Vec2::new(0.2, -0.1), Vec2::new(-0.1, 0.0), 1.0 / 60.0, 1.0, 0.08, 2.0);
    let ndc = view_proj.project_point3(m.origin + m.direction * 5.0); assert_close2((ndc.x, ndc.y), (0.2, 0.1)); // NDC y up
    assert_close3(m.push, right * (0.1 * 2.0 * 25f32.to_radians().tan() * (16.0 / 9.0) * 60.0));                   // 9.95 m/s
    assert!(SmokeMouse::from_screen(&cam, Vec2::ZERO, Vec2::new(0.0, 0.1), 1.0 / 60.0, 1.0, 0.08, 2.0).push.dot(up) > 0.0);
    // waves_pushes_relax: |displacement| at age 3 * relax_time < 0.05 of at age 0; a 9th push drops the oldest
    // a_mousemove_with_no_button_ends_the_drag (Review Focus 4)
    assert_eq!(drag_after_move(true, false, 0), (false, false));  assert_eq!(drag_after_move(true, false, 1), (true, false));
    assert_eq!(drag_after_move(false, true, 2), (false, true));
    ```
  - GPU:
    - `the_mouse_still_stirs_after_a_long_run` (Review Focus 5): Task 5's plume for 1200 steps. Then 10 steps with a 2 m/s +x push through the box centre (radius 0.05), against 10 without. The mean |Δu| within the radius is > 0.2 m/s.
    - `violent_stirring_stays_finite_and_capped` (Review Focus 5): 600 steps of 1/30 s with a 50 m/s push sweeping the box. All values finite, and max |v| ≤ `max_speed` × 1.01.
    - `curl_pushes_leave_a_trail`: a Curl blob of radius 0.1 m with `strength: 0` and `rise: 0`, and 30 steps of a 1 m/s +x push (radius 0.3) through it. Its COM moves +x by more than 0.15 m.
- [ ] **Step 2: Run** `cargo test -p kansei-core`. Expected: the new tests FAIL.
- [ ] **Step 3: Implement.** The wasm mousemove handler applies `drag_after_move(st.dragging, st.panning, e.buttons())` first.
- [ ] **Step 4: Run** `cargo test -p kansei-core` (PASS). Rebuild the example and run R2 with R4: on each level the smoke visibly moves with the stir, compared with no stir.
- [ ] **Step 5: Commit:** `smoke: the mouse pushes the gas on every level (SmokeMouse::from_screen); controls: CameraControls::is_dragging, a drag ended by a mousemove with no button`

### Task 11: The example's panel, URL parameters and README

**Files:** Modify the example's `src/lib.rs` and `www/index.html`. Create its `README.md`.

**Interfaces:**
- Produces `set` and `view` keys:
  - `motion` (0 waves, 1 curl, 2 fluid; `set_motion` on both volumes);
  - `steam` (0/1: the cup source's `rate`, 0 or the default);
  - `layers` (0/1: `set_enabled`);
  - `density` (`look.density_scale` of both);
  - `mouseStrength`, `mouseRadius`, `fogDensity`.
- Produces URL parameters:
  - `motion=waves|curl|fluid` (default fluid);
  - `layers=1`;
  - `aa=taa|off` (default taa);
  - `res=<16..160>` (default 96, the plume; the layers take 2/3);
  - `stats=1`, `perf=1`, `dpr=`.
- The panel binds the keys as `examples/depth-of-field/www/index.html` does, collapsed on narrow screens.

- [ ] **Step 1: Implement** the keys, the parameters and the panel. `view()` returns the current values as JSON.
- [ ] **Step 2: Verify (R1, R2, R4):**
  - It opens on Fluid, and every control and parameter acts.
  - All three levels follow the stir, and orbiting doesn't stir.
  - `?aa=off` is clean on every level, with no ghost after a fast stir or orbit.
  - A close view shows no blocky cells.
  - `?layers=1` works on every level, and the console is clean.
- [ ] **Step 3: Measure with R3**, for each level with layers off, and Fluid again with `?layers=1`: the simulation's GPU time (the `Smoke/*` passes), the rendering's (`VolumetricFog/SmokeMarch` + `SmokeTemporal`) and the frame interval.
- [ ] **Step 4: Write `README.md`** in the format of `examples/volumetric-fog/README.md`:
  - what it shows;
  - the engine API: `SmokeSimulation`, `SmokeVolume`, `SmokeMotion`, `SmokeMouse::from_screen`, `VolumetricFogEffect::set_smoke`, `SmokeRenderOptions`, `CameraControls::is_dragging`, `pacing::FixedStep`;
  - a URL-parameter table and the controls;
  - a GPU-time table per level from Step 3, naming the GPU and the layers' extra time;
  - the build line.
- [ ] **Step 5: Run** `cargo test -p kansei-wasm`. Expected: PASS.
- [ ] **Step 6: Commit:** `examples: smoke, its panel (motion level, steam, layers, density, mouse, fog), URL parameters, and README with each level's GPU time`

### Task 12: Docs, the examples list, the PR

**Files:** Modify `README.md:133-134` and `examples/index.html:169, 308-336`.

- [ ] **Step 1: Root README**, under **Lighting and shadows** after the fog line: `- Volumetric smoke and steam at three levels of motion (sine waves, curl noise, a grid fluid with buoyancy and vorticity), stirred by the mouse, lit and composited through the volumetric fog (Rust)`.
- [ ] **Step 2: `examples/index.html`.**
  - Drop `hidden` from the atmosphere nav item (line 169) and from `<section … id="atmosphere">`.
  - Add a visible `<li>` named `Volumetric smoke`, with:
    - a one-line description in the list's voice;
    - a WASM button to `/examples/smoke/`;
    - dashed `run-rust run-variant` buttons for `?motion=curl` ("Curl noise"), `?motion=waves` ("Waves") and `?layers=1` ("Layers").
  - Every other row there stays `<li hidden>`.
- [ ] **Step 3: Final checks:**
  - `cargo test -p kansei-core` and `cargo test -p kansei-wasm` pass;
  - `pnpm test` and `pnpm exec tsc` pass;
  - R1's build succeeds;
  - through `pnpm dev`, the atmosphere area shows only the smoke row.
- [ ] **Step 4: Commit:** `README and examples list: volumetric smoke (Rust); the atmosphere area shown with its smoke row`
- [ ] **Step 5: PR.** `git push -u origin HEAD`, then `gh pr create --base development`.
  - Say what was built and the decisions the owner may revisit (decisions 1, 4, 9 and 11).
  - Describe Checkpoint B's captures, and give Checkpoint A's R3 numbers and the per-level table.
  - No AI attribution.
