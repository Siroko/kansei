# GI box

A Cornell box (red and green side walls, two white blocks, a rug half orange and half blue,
optionally the Stanford dragon) under one shadowed downlight, for comparing global illumination
methods. With GI off the ceiling and the shadows are black; with it the floor's light reaches
the ceiling and the walls bleed their colour onto the floor and the blocks.

The modes: screen-space GI (sees only what is on screen); voxel GI, which voxelizes the
renderables through their own vertex shaders, lights the voxels through the lamp's shadow map and
traces cones through them, so light from off screen arrives too (the rug's voxels take its texture
through its material's voxel entry); voxel GI with screen-space GI in front for contact detail; and
irradiance probes traced in the voxels' distance field in place of the per-pixel cones. The
distance field can also add AO to the GI and shadow the voxels and the direct light.

Engine API: `Renderer::enable_voxel_gi` (`SceneVoxelGiOptions`, `VoxelGiQuality`),
`Renderable::with_gi` (`GiSurface`), `MaterialOptions::voxel_fragment_entry` with
`gi::VOXEL_WRITE_WGSL`, `SceneVoxelGi::enable_sdf` / `enable_probes` (`SdfProbeOptions`) /
`invalidate`, `SdfShadows`, `VoxelGIEffect` (`VoxelGIOptions::near_field`, `set_sdf`,
`set_probes`), `ScreenSpaceGIEffect` (`GiQuality`), `gi::SDF_WGSL`, `SpotLight` with
`lights::SPOT_LIGHTS_WGSL` and `Renderer::enable_spot_shadows`, `materials::GBUFFER_OUT_WGSL`,
`GLTFLoader::load_glb` / `load_gltf_with_buffers`, `CameraControls::with_mouse_pan`,
`Renderer::set_profiling` / `take_profile`, `ToneMapEffect`.

| URL parameter | Effect |
|---|---|
| `preset=<name>` | `off`, `ssgi`, `voxel`, `best`, `indirect`, `voxels`, `phone`, `dragon`, `sdf`, `sdf-dragon`, `slice`, `probes`, `probe-view` or `probes-dragon` (`PRESETS` in `src/lib.rs`); applied first, the other parameters override it. Default `best` (voxel + SSGI) unless the URL has `gi=` |
| `gi=<mode>` | `off`; screen-space `low`, `medium`, `high` (or `ssgi`), `ultra`; `voxel`, `voxel+ssgi`, `probes`, `probes+ssgi` |
| `voxels=low\|medium\|high` | voxel volume resolution (default medium; low on phones, which also keep the volume within 24 MiB) |
| `view=<view>` | `lit` (default), `indirect` (only the light GI adds, 2 stops brighter), `voxels` (the lit voxels), `sdf` (a slice of the distance field), `probes` (the probes, lit by their own irradiance) |
| `slice=<metres>` | height of the `view=sdf` slice (default 0.6) |
| `sdf_ao=<0..1>` | strength of the distance field's AO on the GI (default 0) |
| `sdf_shadows=off\|fallback\|always` | the voxels' shadows through the distance field: never (default), where no shadow map covers them, or always |
| `shadows=map\|sdf` | the direct light's shadows from the shadow atlas (default) or the distance field |
| `cam=front\|corner\|low` | starting camera (default front) |
| `dragon=1\|full` | add the Stanford dragon: `1` (or `light`) the 19k-triangle `.glb`, `full` the 871k-triangle scan (24 MB) |
| `animate=1` | the dragon (or, without it, the tall block) turns and slides, revoxelized each frame |
| `albedo=constant` | the rug's voxels take its mean colour instead of its texture |
| `rug=off` | no rug |
| `ui=0` | hide the panel |
| `stats=1` | overlay: triangles, frame interval and each pass's GPU time |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Any distance-field use (`sdf_ao`, `sdf_shadows`, `shadows=sdf`, `view=sdf`, the probes) turns
voxel GI's volume on whatever the `gi` mode.

Controls: drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan. A
Tweakpane panel (loaded from jsDelivr; collapsed on phones) switches the same settings at run
time; `window.kansei` exposes its setters and `info()` for scripted captures.

Assets: the Stanford dragon, fetched only when shown: `www/assets/stanford_dragon_pbr.glb` and
`www/assets/scene.gltf` + `scene.bin`. "Stanford Dragon (Vrip)" by 3D graphics 101 on Sketchfab,
CC-BY-NC-4.0 (`www/assets/license.txt`); the page shows the credit while the dragon is on. The
rug's texture is generated in code.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`. A CPU
path-traced reference of the same room is
`cargo run --release -p kansei-native --example gi_box_reference -- out.ppm [samples] [indirect]`.
