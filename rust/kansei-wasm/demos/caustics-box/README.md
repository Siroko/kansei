# Caustics

Ray-traced glass and water and the light they focus, in a plain white studio. Two scenes
(`?scene=`, also the panel's first control):

- **`box`** (the default): the engine's SPH (or PBF) water in a white tank, its surface drawn as
  ray-traced glass, its caustics on the tank's bed, and a glass sphere resting in it. **Drag in
  the view to push the water** along the mouse's motion.
- **`dragon`**: the glass Stanford dragon at full tessellation (the 871k-triangle scan) on a white
  or grey floor that reaches the horizon: its caustics, its soft shadow and the ambient occlusion
  under it, the key light moved with the panel or a shift drag.

Nothing private: the dragon is the gi-box example's CC-BY-NC-4.0 scan (credited on the page),
linked into `www/assets/`.

## Run it

```sh
wasm-pack build --target web --release
python3 -m http.server 8080      # any static server, from this folder
# open http://localhost:8080/www/            (the box)
#      http://localhost:8080/www/?scene=dragon
```

Serve with `Cache-Control: no-store` while rebuilding (see `AGENTS.md`). `www/assets/` links the
dragon's files from `../../examples/gi-box/www/assets/`; on the published site the page fetches
them from the gi-box example's folder beside its own.

## Controls

| | |
| --- | --- |
| left drag | push the water (box), orbit (dragon): the panel's *left drag* picks push, orbit or the light |
| right drag, alt + left drag | orbit |
| shift + left drag | move the key light (azimuth across, elevation up and down) |
| wheel | zoom |
| `R` | fill the tank again |
| `H` | the camera back home |
| `P` | hide the panel |
| `` ` `` | the performance overlay (`?perf=1`) |

## What it shows

- **The floor** (`floor=white|grey|checker`): white, an 18% grey, or a lab's checkerboard
  (`checker_size` squares of 5 cm to 1 m, light and mid grey, `checker_lines` thin metre lines,
  `checker_axes` the axes, x = 0 blue and z = 0 red), procedural in the floor's shader and box
  filtered over each pixel's footprint, so it greys out toward the horizon without moiré; the
  tank's bed carries it too. The reflections' and the glass's hits read the same pattern from a
  texture (`studio::checker_texture`, `RtSurface::with_albedo_layer`), the floor's uvs a period
  apart; the caustics' composite and the RT shadows read it from the GBuffer's albedo.
- **The studio** (`studio.rs`): a 800 m floor and a backdrop as bright as that floor lit far off,
  so the horizon disappears; auto exposure (`auto_ev`) puts the lit white floor at display white
  under the Khronos neutral tone map. Its surfaces are deferred: `RtShadowsEffect` adds the key
  light with ray-traced soft shadows (a sun's angular size, a spot's disk, `light_size`), the
  hybrid RT GI (`RtDiffuseGiEffect`) the white sky's light (`sky_radiance`, the fill), its
  occlusion and the floor's bounce (or the flat sky alone: `gi=rt|off`).
- **Ambient occlusion** (`ao.rs`, `ao.wgsl`): the white sky's fill is about a sixth of the floor's
  light under the key, so its physical occlusion (the RT GI's) barely shows (1/255 on average
  against the flat sky), and the engine's `SSAOEffect` runs before the direct light is added. This
  is a game's AO multiplied into the lit image after the lighting, the glass and the reflections:
  points in the hemisphere round the GBuffer's normal (`ao_samples`, rotated per pixel and frame
  for TAA), out to `ao_radius`, the near ones weighed up by `ao_falloff`, the unoccluded share to
  the power `ao_intensity`; the sky, glass and the water pass through. `ao_view` shows it alone.
  About 0.6-0.9 ms at 1600 x 813.
- **The water** (`water.rs`): about 41k particles 3.4 cm apart (`scale=16`, simulation units a
  metre) in the tank's bounds, the lake demo's tuning at a 16th of its size (time √16 times
  faster, so it moves as water does). Its surface is marching cubes over a surface field, drawn as
  glass (`FluidSurfaceEffect::surface_renderable_glass`), its height field (`FluidHeightfield`) in
  the ray tracing grid as a coarse mesh, which the caustics' photons refract at. The tank (walls
  and bed) is lit by its own material (the studio shader's forward mode, the sun through the
  cascades, as `standard_lit`): the glass
  pass reads what lies under the water from the GBuffer's background, which a deferred surface has
  no direct light in. The water rests when it is still (`fluid_rest`).
- **The mouse** (`push.rs`, `push.wgsl`): a `FluidSubstepPass` that makes the particles in a column
  round the point under the cursor (where its ray meets the still water) follow the cursor's
  motion over the water, each substep `push_strength` of the way, less toward the column's edge
  (`push_radius`), for either solver.
- **The caustics** (`RtCausticsEffect`): photons through the sphere's and the dragon's glass and
  through the water's waves (`caustics_water_photons`), the water's normal for them taken over
  `caustics_smooth` texels (`RtCausticsEffect::water_span`) so the particles' bumps don't speckle
  the bed, and over them the game's net of caustic lines (`caustics_lines`,
  `RtCausticsEffect::water_lines`): the photons bring the focus of the waves the mouse makes, the
  lines a pool's shimmer.
- **The dragon** (`add_dragon` in `lib.rs`): the full scan drawn as it is (`dragon_lod=1`: by its
  cluster LOD), and in the ray tracing grid a copy of its own, a cut of its cluster graph at
  `dragon_rt_mm` of error (about 56k triangles at 30 mm): the grid's own cut of the full scan cost
  4-8 ms of glass pass, and the 19k decimation's left 900 triangles. The grid's cells are 3.5 cm
  (`grid_cell`), 140 x 48 x 140 of them round the subject: the glass's cost is the triangles a ray
  meets in a cell.

## Settings

Every setting is a URL parameter (its starting value) and a control in the panel, live; the names
are `src/settings.rs`'s. A few take a reload: `scene`, `scale` (the particles' spacing),
`dragon` (`full` or `light`), `dragon_lod`, `dragon_rt_mm`, `grid_cell`. `quality=game` (the
default) or `high` sets the glass's, the caustics' and the rays' costs together
(`Settings::apply_quality`), per scene. `window.caustics` has `info()` (with `stats=1`, each
pass's GPU time), `set(key, value)`, `reset()` and `home()` for scripted captures.

## Timings

Headless Chrome on the test Mac (Apple GPU), the `game` preset, the water kept moving
(`fluid_rest=0`), `stats=1`'s frame and GPU times, with other sessions sharing the GPU (runs differ
by tens of percent):

| scene | canvas | scene pixels | frame | GPU |
| --- | --- | --- | --- | --- |
| box | 1600 x 813 | 1600 x 813 | 13 ms | 10.5 ms |
| box | 3200 x 1626 (pixel ratio 2) | auto, x 0.71 | 15 ms | 13.4 ms |
| dragon | 1600 x 813 | 1600 x 813 | 15 ms | 12.7 ms |
| dragon | 3200 x 1626 (pixel ratio 2) | auto, x 0.71 | 16 ms | 15.1 ms |

`render_scale=0` (the default) renders the scene at about 1.3 million pixels whatever the canvas
(TAA upscales); the ray-traced passes cost by the pixel, so a Retina canvas at its full size cost
the box 27 ms. The box's largest passes: the surface field's splat (1.4 ms), the water's glass
(1.3 ms), the GI's rays (1.1 ms) and the simulation (1 ms, 42k particles, 4 substeps); the
dragon's: its glass (4.5 ms) and the caustics' photons (1.6 ms, every other frame). `quality=high`
took the box to 21 ms at 1600 x 813. The water asleep (`fluid_rest`, the default) costs nothing
of the simulation and the surface.
