# Volumetric smoke: design

Date: 2026-11-03. Engine: Rust (`kansei-core`), shown by a new WebAssembly example.

## Intent

Kansei has froxel volumetric fog (`VolumetricFogEffect`, with `LocalFogVolume`s) for haze that
fills a scene. It has nothing for gas that moves and has shape: steam off a cup, smoke hanging in
a room, and similar effects. The owner wants a volumetric smoke system that pairs with the fog,
lit the same way so the two read as one medium.

Smoke needs different amounts of realism in different places, so the system offers three levels
of motion on the same smoke, picked per smoke volume:

1. **Waves:** simple sin/cos motion, close to free. For background wisps and many cheap sources.
2. **Curl noise:** swirly, turbulent motion with no physics pushing back.
3. **Fluid:** a real fluid simulation, with buoyancy, swirls that persist, and a wake behind the
   mouse. For the hero plume.

## Goals

- A smoke system in `kansei-core` with the three motion levels above, all lit and composited
  through the volumetric fog's lighting.
- Steam that rises from a source (the main case), and smoke left hanging in the air with no
  source (layers drifting through a room).
- Mouse interaction: the gas is pushed by the mouse using the `MouseVectors` behaviour (screen
  position, direction and strength, mapped into the scene the way the `fluid` example maps it
  for its particles).
- A new example, `rust/kansei-wasm/examples/smoke/`, that shows it (below). It opens on the
  **Fluid** level.

## Non-goals

- Smoke casting shadows onto scene surfaces. It is lit and shadowed by the scene, and shadows
  itself, but it doesn't darken the room.
- Fire and combustion (flames, emissive temperature colour).
- A TypeScript port (Rust only for now).
- Web Worker (`?worker=1`) support in the example.
- Changing how the existing fog looks in scenes with no smoke.

## Anti-goals (failure even if it technically works)

- A visible seam where smoke and fog meet: smoke that looks pasted on top of the fog, lit by
  different rules, or composited in the wrong order. Fog in front of the smoke must veil it, and
  the smoke must hide the fog behind it.
- Smoke that only looks clean under TAA. The owner also runs smoke and particle demos with SMAA
  or no anti-aliasing (the separate SMAA design), so ray-march noise must not rely on TAA's
  history to disappear.
- Blocky smoke. Wisps must stay finer than the fog's froxels. The froxel grid (160 x 90 x 64 by
  default) is far too coarse for a plume a few centimetres across.
- A Fluid level that blows up, leaks density out of a closed box, or stops reacting to the mouse
  after a while.

## Lighting (confirmed by the owner)

The smoke is lit like the fog: the same lights (sun, point and spot), the same shadow maps (the
window frame's shadow cuts through the steam), the same sky or ambient term, and the same
scattering phase function and albedo model. A scene that already binds its lights and shadows to
`VolumetricFogEffect` (`update_lights`, `set_shadow_map`, `set_cascaded_shadow_map`,
`set_point_shadows`, `set_spot_lights`, `set_sky_lighting`) shouldn't have to bind them a second
time for the smoke. The smoke also shadows itself: the side away from the light is darker where
the smoke is thick.

The smoke must still work, lit by those lights, when the fog's own density is zero.

*(My call)* Dense steam may add a multiple-scattering approximation, such as the Wrenninge
octaves `clouds_march.wgsl` uses, so thick parts don't go grey on the side away from the light.
It must reduce to the fog's single scattering (`henyeyGreenstein`, the same albedo) where the
smoke is thin, so the two still match where smoke fades into fog.

## Motion levels: my calls

Each level only differs in where the gas's motion comes from:

| Level | State on the GPU | Motion | Mouse |
|---|---|---|---|
| Waves | none: the density is evaluated where the ray march samples it | the source's plume (or a hanging layer) displaced by sums of sin/cos over height and time | a push around the mouse ray that relaxes back (a few recent pushes kept, decaying) |
| Curl noise | a density grid in the smoke's box, advected each step | curl-noise velocity (divergence-free; Bridson et al. 2007) plus a buoyant rise | pushes added to the velocity the advection uses, so it leaves trails |
| Fluid | velocity, density and temperature grids, plus the solver's scratch grids | a grid smoke solver: advection, buoyancy from temperature, vorticity confinement, pressure projection, dissipation, the box's walls as boundaries (Stam 1999; Fedkiw, Stam and Jensen 2001) | a force along the mouse ray, which the solver carries on as a wake |

Curl noise and Fluid share the density grid, its advection and its rendering. Waves shares only
the rendering. Switching a volume's level may restart its gas; that's fine.

## Shape of the API: my call, names are the builder's

- A smoke volume: a box in world space (centre, size, optional yaw like `LocalFogVolume`), a grid
  resolution for the levels that have one, its motion level and that level's settings, and its
  look: density scale, albedo, extinction, and dissipation over time.
- Sources: a position, radius, emission rate, temperature (for buoyancy), and initial velocity. A
  volume with no sources and an initial fill gives the hanging-smoke case.
- The simulation is owned by the app and stepped by it, as `FluidSimulation` is: with
  `pacing::FixedStep` in the example, and one call a step taking the mouse ray, direction and
  strength.
- Rendering goes through the fog: smoke volumes registered with `VolumetricFogEffect`, or a
  separate effect that reads the fog effect's lighting and froxel grid. The builder picks one.
  Whichever it is has to meet the lighting and seam requirements above.

## Constraints

- Ray-march the smoke at its own resolution inside its box, against the GBuffer's depth so
  geometry occludes it (the cup and table sit in the plume's base). Don't inject it into the
  froxels. Half resolution with a depth-aware upsample is allowed if edges against geometry hold.
- Several smoke volumes can be on screen, and they can overlap: steam rising into a hanging
  layer. Overlapping volumes are marched together in one pass, sorted along the ray, never
  composited one pass after another, which would make the seam the anti-goals forbid.
- *(My call)* The smoke may keep its own history from frame to frame (jittered march steps
  resolved over frames, as the fog's ray-marched spot shafts and the clouds do), separate from
  TAA. It must not smear: stirring the smoke fast with the mouse, or a fast orbit, leaves no
  ghost trail. Reproject with the gas's own velocity where there is a grid, and clamp or reject
  history that no longer fits.
- WebGPU's default limits (wgpu 30 cannot request more on the web), WGSL that Chrome accepts,
  naga validation and uniform-size tests in `cargo test -p kansei-core` (the `shaders_validate_*`
  pattern), and a browser check. AGENTS.md lists naga-accepts and Chrome-rejects cases.
- `queue.write_buffer` lands before the next submit, so per-step or per-dispatch parameters need
  their own slots (AGENTS.md). Upload a frame's parameters in one write.
- Label every pass with `profiling::gpu_pass`, so the perf overlay and `take_profile().report()`
  show the simulation and the rendering per level.
- The example uses kansei-core's public API only. What it needs from a private shader becomes a
  public constant (AGENTS.md).
- On native as on the web: nothing in the system depends on `wasm32`. The example is web-only.

## The example: `rust/kansei-wasm/examples/smoke/`

A dark room: walls, floor and ceiling, with one window. A low sun with a shadow map comes through
the window as a shaft, made visible by faint volumetric fog in the room. A table stands in the
shaft with a cup on it, and steam rises from the cup.

- Camera: orbit (drag), pan (right-drag or shift-drag, `CameraControls::with_mouse_pan`) and zoom
  (wheel). Hovering stirs the smoke through `MouseVectors`. *(My call)* While a camera drag is on,
  the mouse doesn't stir, so orbiting doesn't tear the plume. `CameraControls` has no public
  "drag in progress" flag yet; add one to kansei-core.
- *(My call)* The hanging smoke layers:
  - They are a second, room-sized smoke volume with no source, at a coarser grid than the cup's
    plume, filled with a few soft horizontal layers when switched on. They overlap the plume's
    volume, which the one-pass march above handles.
  - They follow the panel's motion level like the plume.
  - On Fluid, they keep drifting because of a weak, large-scale ambient draft (a slow
    curl-noise force, not a uniform wind, which would pile them against a wall) plus the mouse.
  - They don't dissipate.
  - The README states the extra GPU time with `?layers=1` on Fluid. The 60 fps target applies
    with them off.
- A Tweakpane panel:
  - the motion level: Waves, Curl noise or Fluid (default Fluid)
  - the cup's steam on or off
  - hanging smoke layers on or off (the no-source case, off by default)
  - density, mouse strength and radius
  - the fog's density, so the smoke can be seen with and without fog
- URL parameters: `?motion=waves|curl|fluid`, `?layers=1`, `?aa=taa|off` (to check the smoke
  without TAA), `?res=` for the grid resolution. `?perf=1` comes from `kansei-wasm`.
- A README in the examples' format: what it shows, the engine API, its URL parameters, and the
  GPU time of each level.
- A row on `examples/index.html` in the "Atmosphere and fog" section, with a WASM button and
  dashed variant buttons for `?motion=curl`, `?motion=waves` and `?layers=1`. That section and
  its nav link are hidden today (decided with the owner after review):
  - Show the section and its nav link again, with the smoke row visible.
  - Every other row in it (sky atmosphere, volumetric fog, and any others) stays hidden as
    `<li hidden>`.

## Performance target: my call, to confirm by measuring

At 1080p on the owner's Mac (Apple GPU), the example holds 60 fps on the Fluid level at the
default grid resolution, fog on. The README and the PR give the simulation's and the rendering's
GPU time for each level.

## Tests

- Naga validation and uniform sizes for every new shader.
- GPU tests (`tests/smoke_gpu.rs`, the `*_gpu.rs` pattern):
  - After the pressure projection, divergence is below a tolerance.
  - A hot blob rises: its density's centre of mass moves up.
  - Density stays finite and inside the box over many steps.
  - The curl-noise velocity's numerical divergence is near zero.
- CPU tests where the logic allows, such as mapping the mouse ray into the box and the Waves
  level's density function.

## Left to the builder

- Grid resolutions and their defaults, the advection scheme (semi-Lagrangian or MacCormack), the
  pressure solver (Jacobi, red-black Gauss-Seidel or multigrid) and its iteration count.
- How self-shadowing is computed (a light-transmittance volume per step, a march toward each
  light, or another way), within the performance target.
- Whether rendering lives inside `VolumetricFogEffect` or beside it, within the seam and lighting
  requirements.
- The noise functions and the Waves level's exact formulas.
- The example's exact geometry and materials (`Material::standard_lit` through a
  `PostProcessingVolume`).

## Done when

- `cargo test -p kansei-core` passes, including the new validation and GPU tests.
- The example builds with `wasm-pack build --target web --release` and runs in Chrome, opening
  on Fluid. All three levels react to the mouse. It looks clean with `?aa=off`, and has no
  visible seam with the fog.
- The README and the PR give the per-level GPU times measured at 1080p on the owner's Mac.
- A PR against `development`, with no AI attribution in commits or the PR.
