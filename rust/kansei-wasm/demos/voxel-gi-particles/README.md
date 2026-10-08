# Voxel GI on particles

Hector Arellano's "indirect lighting on particles" (miaumiau.cat, p=1476) on Kansei's `gi`
module. Each frame the particles of an SPH fluid splat their density and emission into a voxel
volume, with the walls as analytic boxes. Each particle then cone traces its incoming light (six
90-degree cones) and its visibility of the key light (one narrow cone), so the pile darkens inside,
glowing particles light their neighbours, and the walls take the pile's shadow and glow.

Two scenes. The lightbox (default): a closed white room on black, lit by an emissive panel in its
ceiling, over a glossy floor that reflects it; spheres of varied size, charcoal and brown with a
share glowing orange to yellow, rain down into a pile. With `rt=on` some spheres turn to mirrors
and glass, their rays walking the fluid's own neighbour grid. `scene=cornell`: the fluid
dam-breaks in an open-topped Cornell room (red and green side walls) under a sun and a blue sky,
drawn as camera-facing discs.

Engine API: `kansei_wasm::{Canvas, run, param, param_or, flag, is_phone}`, `FluidSimulation`
with `FluidSimulationOptions` (`rebuild_grid`, `reset_particles`, `positions_as_compute_buffer`,
and its sorted positions, cell offsets, sorted indices and `grid().params_buffer()` bound into the
spheres' material), `simulations::grid::NEIGHBOUR_GRID_WGSL`, `gi::ParticleGi`
(`ParticleGiOptions`, `ParticleEmission`, `GiBox`, `set_boxes`, `set_sky_gradient`, `sky_buffer`,
`lighting_buffer`, `lighting_instance_buffer`, `reset_history`), `VoxelGiQuality`,
`gi::VOXEL_CONES_WGSL` (`voxelConeTrace`, `voxelConeTraceSplit`, `voxelHemisphereCone`),
`atmosphere::SKY_LIGHTING_WGSL` and `direction_from_elevation_bearing`, `InstancedGeometry`,
`materials::Compute`, `pacing::FixedStep`, `CameraControls::with_mouse_pan`, `MouseVectors`,
`Renderer::set_profiling` / `take_profile` and `profiling::gpu_pass`.

Demo-local: the lightbox's sphere impostors (`src/shaders/room_spheres*.wgsl`): quads at each
sphere's nearest point with no `frag_depth`, a depth prepass and the shading on equal depth, both
`@builtin(position) @invariant`, so only the nearest sphere of a pile is shaded (see the Apple GPU
note in `AGENTS.md`); the ray walk for mirror and glass spheres, a DDA over the fluid's neighbour
grid with ray-sphere tests, capped by the pile's highest centre (`pile_top.wgsl`); the rect light,
an emissive ceiling panel whose irradiance is Lambert's polygon formula (`room_common.wgsl`, with
a CPU twin that sets the walls' emission in the volume); the per-particle radius in position w; the walls' and the Cornell room's shaders and
`SceneParams`. The fluid, its grid, the particle splat, the volume, the per-particle cones and the
cone-tracing WGSL are engine API.

| URL parameter | Effect |
|---|---|
| `scene=cornell` | the Cornell room; any other value (or none) is the lightbox |
| `gi=off` | no voxel volume: walls and particles lit by the light and the sky alone (default on; `0`, `false` and `no` also turn it off, but only `off` unticks the panel's checkbox) |
| `view=indirect` | only the light the volume brings, without the direct light, a stop brighter |
| `rt=on` | lightbox only: mirror and glass spheres, ray traced; also fewer, larger particles and no glowing share (`1`, `true` and `yes` also work) |
| `quality=low\|medium\|high` | voxel volume resolution (default medium; low on phones, which also keep the volume within 24 MiB); an unknown name is the default |
| `particles=<n>` | particle count, clamped to 1024..262 144 (default 12 288 in the lightbox, 8192 with `rt=on`, 32 768 in the Cornell room, half of each on phones) |
| `stats=1` | log the frame interval to the console every 240 frames |
| `profile=1` | time each pass; read it with `await window.kansei.profile_report()`, and `set_layers` to time the walls, particles and reflection apart |
| `ui=0` | hide the panel |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |
| `worker=1` | run the demo in a Web Worker on the canvas as an `OffscreenCanvas` (`kansei_wasm::launch`); the panel stays on the page and sends its changes as calls |

Controls: drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan;
moving the pointer over the fluid pushes it. A Tweakpane panel (collapsed on phones) switches the
scene and quality (both reload the page), voxel GI, indirect only, the ray-traced mirrors and glass
(lightbox), the volume's extinction, the cones' aperture and temporal blend, the panel or sun cone,
the wall occlusion (lightbox), the emission, particle size, the ceiling light and floor reflection
(lightbox) or the sun's elevation and bearing (Cornell room), exposure, and has a Pour again
button; it shows the volume's size, memory and the frame time. `window.kansei` exposes every
export (`info`, the setters, `set_positions`, `set_paused`, `set_cone_jitter`) for scripted
captures, each returning a promise of its result (with `worker=1` the call runs in the worker).
To frame a view by hand, orbit and pan (right drag), then `await window.kansei.camera()` gives
the orbit (`target`, `radius`, `azimuth`, `elevation` in radians) that `set_camera(x, y, z, radius,
azimuth, elevation)` and the default camera in `src/lib.rs` take.

Assets: none beyond the page; the shaders in `src/shaders/` are compiled in with `include_str!`.
The TS engine's twin of this demo, `examples/index_voxel_gi_particles.html` (same scenes, URL
parameters, panel and `window.kansei`), runs a copy of this demo's WGSL from
`examples/voxel-gi-particles-shaders.js`: change both together.
The page loads Tweakpane 4 from cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
