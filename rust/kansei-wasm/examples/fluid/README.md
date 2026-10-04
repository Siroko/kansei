# Fluid

50 000 SPH particles in a box under a striped dome, shown as a liquid: the particles are
splatted into a density field, meshed by marching cubes on the GPU, and the surface is refracted
by a post-processing effect, then blurred by depth of field. The panel switches to the particles,
a raymarched density field or the voxel-face shell. fluid.kansei.graphics serves this example
(its `vercel.json` proxies kansei.graphics/examples/fluid/).

Engine API: `kansei_wasm::{Canvas, run, fetch_bytes}`, `FluidSimulation` /
`FluidSimulationOptions` stepped with `pacing::FixedStep`, `FluidDensityField`,
`FluidMarchingCubes` (`set_use_classic`, `set_iso_level`, `MarchingCubesGridSizing`),
`FluidSurfaceEffect` (`step_simulation`, `surface_renderable`) and `DepthOfFieldEffect` in a
`PostProcessingVolume`, `RaymarchingRenderable`, the particles as an `InstancedGeometry` over
`FluidSimulation::positions_as_compute_buffer` drawn with `PARTICLE_BILLBOARD_WGSL`,
`GLTFLoader::load_glb`, `CameraControls` and `MouseVectors`.

| URL parameter | Effect |
|---|---|
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom; moving the pointer over the fluid pushes it. A
Tweakpane panel (collapsed) sets the render mode, SPH parameters, forces and radial gravity, the
simulation bounds, the light, depth of field, the surface's transmission and the dome's stripes.

Assets: `www/assets/dome.glb`, the dome, drawn with the example's own stripe shader (source not
recorded). The page loads Tweakpane 4 from cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
