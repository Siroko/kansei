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

The hybrid (`gi=rt`) traces the GI instead: one ray for each 2 x 2 pixels from the surface
through a grid of the room's triangles (below), the hits lit by the lamp (shadow rays through the
grid) and one cone through the voxels for the further bounces, denoised by SVGF and upsampled. It
comes closest to a path-traced reference: no banding, no light leaking onto the short block's
front, which faces the open side of the box. Its panel folder switches the resolution, the
denoiser, the hits' lighting and shadows, a reference path tracer through the grid
(`mode`, with `accumulate` for a converged image) and debug views (the indirect light, the signal,
SVGF's variance and history, the rays' cost).

A chrome ball stands on the short block and a glass ball on the rug, ray traced through a grid of
the room's triangles (below) whatever the GI: the chrome (`StandardLitOptions::mirror`) mirrors the
room, sharp or rough; the glass (`StandardLitOptions::glass`, `RtSurface::glass`) reflects it by
the exact dielectric Fresnel and refracts it through both of its surfaces, with total internal
reflection, its tint absorbing inside, frosted when rough. Their rays' hits take the lit image where
the camera sees the same point, and elsewhere the lamp (shadow rays through the grid) plus the
voxels' irradiance, so the room in them carries the GI on screen. The GI's own rays and the shadow
maps see through the glass; the chrome is dark to them. The panel's "Mirror & glass" folder turns
each on and off and sets the chrome's roughness and the glass's index of refraction and roughness.

With `reflect=1` the floor is polished and reflects the room: rays traced through a grid of the
room's triangles (96 cells of 4.8 cm each way, rebuilt on the GPU when something in it moves), the
hits lit by the voxels, the voxel cone past the grid. The dragon goes into the grid as the cut of
its cluster LOD at a cell of error (the 871k-triangle scan as 29k triangles).

Engine API: `Renderer::enable_voxel_gi` (`SceneVoxelGiOptions`, `VoxelGiQuality`),
`Renderable::with_gi` (`GiSurface`), `MaterialOptions::voxel_fragment_entry` with
`gi::VOXEL_WRITE_WGSL`, `SceneVoxelGi::enable_sdf` / `enable_probes` (`SdfProbeOptions`) /
`invalidate`, `SdfShadows`, `VoxelGIEffect` (`VoxelGIOptions::near_field`, `set_sdf`,
`set_probes`), `ScreenSpaceGIEffect` (`GiQuality`), `gi::SDF_WGSL`, `SpotLight` with
`lights::SPOT_LIGHTS_WGSL` and `Renderer::enable_spot_shadows`, `materials::GBUFFER_OUT_WGSL`,
`GLTFLoader::load_glb` / `load_gltf_with_buffers`, `CameraControls::with_mouse_pan`,
`Renderer::set_profiling` / `take_profile`, `ToneMapEffect`; with `gi=rt`,
`RtDiffuseGiEffect::with_volume` (`RtDiffuseGiOptions`, `set_spot_lights`, `update_lights`) over the
grid below; with `reflect=1`,
`Renderer::enable_rt_grid` (`SceneRtGridOptions`, a fixed box), `Renderable::rt` (`RtSurface`),
`ClusterLod` on the dragon, `RtReflectionsEffect::with_volume` and `GBUFFER_OUT_WGSL`'s
`kansei_gbuffer_out_specular` (the floor's F0 and roughness); for the balls,
`Material::standard_lit` with `StandardLitOptions::mirror` and `StandardLitOptions::glass`
(`set_standard_lit` for the panel's changes), `RtSurface::glass`, and `RtReflectionsEffect`'s
`set_glass` (`RtGlass`), `set_spot_lights`, `screen_hits` and `hit_indirect`.

| URL parameter | Effect |
|---|---|
| `preset=<name>` | `off`, `ssgi`, `voxel`, `best`, `indirect`, `voxels`, `phone`, `dragon`, `sdf`, `sdf-dragon`, `slice`, `probes`, `probe-view` or `probes-dragon` (`PRESETS` in `src/lib.rs`); applied first, the other parameters override it. Default `best` (voxel + SSGI) unless the URL has `gi=` |
| `gi=<mode>` | `off`; screen-space `low`, `medium`, `high` (or `ssgi`), `ultra`; `voxel`, `voxel+ssgi`, `probes`, `probes+ssgi`; `rt` (the hybrid; builds the grid of triangles, as `rt=1` does: the panel offers it only then, and reloads the page with `gi=rt` otherwise) |
| `rtgi_res=half\|full` | the hybrid's rays: one for each 2 x 2 pixels (default) or one a pixel |
| `rtgi_denoise=svgf\|temporal\|off` | SVGF (default), its temporal accumulation alone, or the raw 1 spp signal |
| `rtgi_kernel=3x3\|5x5` | SVGF's wavelet (default 3 x 3, half the cost) |
| `rtgi_hit=direct\|voxels` | what lights the hits: their direct light plus a voxel cone (default), or the voxels alone |
| `rtgi_shadows=rays\|maps` | the hits' shadows: rays through the grid (default) or the shadow atlas |
| `rtgi_mode=hybrid\|reference` | `reference`: a path tracer through the grid (4 bounces), for comparison |
| `rtgi_accum=1` | show the running mean of the raw signal (restarts when the camera or a setting changes) |
| `rtgi_view=lit\|indirect\|signal\|variance\|history\|cost` | the hybrid's debug views |
| `voxels=low\|medium\|high` | voxel volume resolution (default medium; low on phones, which also keep the volume within 24 MiB) |
| `view=<view>` | `lit` (default), `indirect` (only the light GI adds, 2 stops brighter), `voxels` (the lit voxels), `sdf` (a slice of the distance field), `probes` (the probes, lit by their own irradiance) |
| `slice=<metres>` | height of the `view=sdf` slice (default 0.6) |
| `sdf_ao=<0..1>` | strength of the distance field's AO on the GI (default 0) |
| `sdf_shadows=off\|fallback\|always` | the voxels' shadows through the distance field: never (default), where no shadow map covers them, or always |
| `shadows=map\|sdf` | the direct light's shadows from the shadow atlas (default) or the distance field |
| `cam=front\|corner\|low\|floor\|glass` | starting camera (default front; `floor` looks down onto the floor, `glass` close on the glass ball) |
| `mirror=0`, `glass=0` | no chrome ball, no glass ball (without both, `reflect`, `gi=rt` and `rt=1`, no grid is built and voxel GI runs only for its modes) |
| `mirror_rough=<0..1>` | the chrome's roughness (default 0, a mirror) |
| `glass_ior=<ior>` | the glass's index of refraction (default 1.5) |
| `glass_rough=<0..1>` | frosted glass (default 0, clear) |
| `glass_tint=<r,g,b>` | the light the glass leaves after a metre inside (default `0.82,0.93,0.88`, a faint green) |
| `glass_samples=<n>` | paths a frosted glass pixel traces a frame (default 4) |
| `rt_screen=0` | the balls' and the floor's rays' hits lit by the lamp and the voxels even where the camera sees them |
| `rt_direct=0` | the hits the camera doesn't see lit by the voxels' radiance alone |
| `dragon=1\|full` | add the Stanford dragon: `1` (or `light`) the 19k-triangle `.glb`, `full` the 871k-triangle scan (24 MB) |
| `animate=1` | the dragon (or, without it, the tall block) turns and slides, revoxelized each frame |
| `albedo=constant` | the rug's voxels take its mean colour instead of its texture |
| `rug=off` | no rug |
| `reflect=1` | the polished floor's ray-traced reflections (turns voxel GI's volume on whatever the `gi` mode) |
| `rt=1` | build the grid of the room's triangles without the reflections (the hybrid can then be picked at run time) |
| `floor_f0=<0..1>`, `floor_rough=<0..1>` | the floor's F0 (default 0.3) and roughness (default 0.05) |
| `rt_view=lit\|reflection\|mirror\|cost` | the lit image (default), the light the reflections add, what the rays see, their cost (cells and triangles a ray) |
| `rt_trace=voxels` | reflections from the voxel cone alone (for comparison) |
| `rt_res=quarter` | trace one pixel of each 4 x 4 a frame (default 2 x 2) |
| `rt_lod=0` | the dragon into the grid whole, not by its cluster cut (the full scan's graph takes about 4 s to build) |
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
