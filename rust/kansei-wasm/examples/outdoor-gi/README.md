# Outdoor GI

A forest valley 1.3 km across under a low sun, with voxel GI in a clipmap round the camera: the
sky's light reaches the forest floor and the road only where the canopy and the terrain let it
through, and the sunlit ground and trees light what lies in their shade. With GI off, the
materials' own sky light sees no trees.

- The terrain is 64 tiles of a height field.
- The spruces (17 576 by default) are instanced and culled on the GPU per view, with dithered
  crossfades between their LODs, as the film's forest is:
  - near ones are cards of needle sprays with cluster LOD (cut out by an alpha test, in their
    shadows too);
  - farther ones are two LODs of cone meshes.
- The sun shadows through four cascades out to 160 m.
- The sky is an atmosphere, with its aerial perspective.

The GI modes (`gi=`) compare what lights the shade:

| Mode | What lights the surfaces besides the sun |
|---|---|
| `off` | the materials' own sky light: the whole sky, as if no tree stood over them |
| `skyocc` | that sky light dimmed by the top-down sky occlusion (`Renderer::enable_sky_occlusion`), the film's technique today |
| `visibility` | that sky light dimmed by the sky visibility the clipmap's probes measure, in the material (`kansei_clipmap_sky_visibility`) |
| `cones` | voxel GI on screen, six cones a pixel through the clipmap: the sky past the canopy and the bounces |
| `probes` | the same light from the clipmap's irradiance probes, cheaper and smoother |

The voxel clipmap:
- 5 levels of 64 × 32 × 64 voxels, from 0.5 m voxels over 32 m to 8 m voxels over 512 m.
- Each level's window follows the camera, rewriting only the slab it moves into.
- What it voxelizes:
  - the terrain tiles near each slab;
  - the trees through the clipmap's own cull view: the cards by their cluster cut within 20 m, the
    middle cone LOD to 60 m, the coarsest past that.
- It is lit through the cascades, or cones through itself past them.

Engine API:
- `Renderer::enable_voxel_clipmap` (`SceneVoxelClipmapOptions`), `SceneVoxelClipmap`
  (`settings`: `ClipmapGiSettings`, `ConeShadows`; `use_sky_lighting`, `enable_probes`:
  `ClipmapProbeOptions`).
- `VoxelGIEffect::with_clipmap`, with `set_clipmap_probes`.
- `gi::CLIPMAP_PROBES_WGSL` with `ClipmapProbes::bindings_wgsl` / `bind_group_entries`.
- `Renderable::with_gi`: `GiSurface::with_opacity`.
- `MaterialOptions::voxel_fragment_entry` with `gi::VOXEL_WRITE_WGSL`'s
  `kansei_voxel_write_coverage` (the cards' alpha test).
- `InstanceCulling` (`with_crossfade`, `with_gi_lod_range`), `ClusterLod` (`ClusterOptions::cards`).
- `culling::LOD_FADE_WGSL`.
- `Renderer::enable_sky_occlusion` with `shadows::SKY_OCCLUSION_WGSL`.
- `Renderer::enable_cascaded_shadows` with `shadows::CASCADED_SHADOWS_WGSL`.
- `SkyAtmosphere` with `AtmosphereEffect`.
- `VolumetricFogEffect` (`set_clipmap_probes`, `set_sky_occlusion`).
- `TemporalAAEffect`, `ToneMapEffect`.
- `Renderer::set_profiling` / `take_profile`.
- `Renderer::enable_rt_grid` (`SceneRtGridOptions`), `Renderable::rt` (`RtSurface`) and
  `rt_placement` (`RtPlacement::Wgsl`: the spruces' records widen them by their tint, which
  `InstanceTransform` can't say).

| URL parameter | Effect |
|---|---|
| `gi=off\|skyocc\|visibility\|cones\|probes` | what lights the shade (default `cones`) |
| `view=lit\|indirect\|voxels` | the lit image (default); only the light the GI adds; the clipmap's voxels and their light |
| `cam=road\|clearing\|forest\|high\|fly` | starting camera (default `road`); `fly` drives along the road at 12 m/s, so the clipmap's windows move |
| `elevation=<degrees>` | the sun's elevation (default 16; the exposure follows) |
| `bearing=<degrees>` | the sun's bearing (default 160) |
| `ev=<EV100>` | exposure (default: the sun's, opened up 1.8 stops for the forest's shade) |
| `fog=<density>` | volumetric mist (0.01 is light), its ambient from the probes in the clipmap's modes |
| `trees=<n>` | at most this many spruces (default 20 000; the valley holds 17 576) |
| `cards=0` | cone meshes for the near spruces too, in place of the cards |
| `crown_opacity=<0..1>` | how much light the cone crowns' voxels stop for their area (default 0.35) |
| `levels=<n>` | clipmap levels (default 5) |
| `res=<voxels>` | voxels across each level (default 64; half as many in height) |
| `vox=<metres>` | the finest level's voxels (default 0.5) |
| `lit=<n>` | levels lit a frame (default 2; 0: all) |
| `probes=<n>` | probes traced a frame (default 8192) |
| `coneshadows=off\|fallback\|always` | the voxels' shadows from cones through the clipmap: never, where the cascades don't reach (default), always |
| `bounce=<share>` | share of last frame's light the voxels bounce again (default 1) |
| `shadowsteps=<n>` | steps of the voxels' shadow cones (default 48) |
| `intensity=<scale>` | scale of the light the on-screen GI adds (default 1) |
| `shadow_far=<metres>` | the cascades' reach (default 160) |
| `rt=1` | build a ray tracing grid of the scene round the camera (64 x 32 x 64 m): the trees culled for its box and cut at a cell of error, on the GPU; the stats show its triangles and build |
| `rt_cell=<metres>` | the grid's cells (default 0.5; the box stays 64 m across) |
| `rt_rebuild=1` | rebuild the grid every frame, not only when its box moves |
| `stats=1` | overlay: the clipmap, triangles, frame interval and each pass's GPU time |
| `ui=0` | hide the panel |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom, right-drag or shift-drag to pan. A Tweakpane
panel (loaded from jsDelivr) switches the GI mode, the view, the camera and the sun's
elevation. `window.kansei` exposes the setters and `info()` for scripted captures.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
