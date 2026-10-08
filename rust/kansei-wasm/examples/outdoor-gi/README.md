# Outdoor GI: the Raggare forest

The forest of the Raggare intro film ([raggare.kansei.graphics](https://raggare.kansei.graphics))
under a low sun, with voxel GI in a clipmap round the camera, screen-space GI or the hybrid
ray-traced GI: the sky's light reaches the forest floor and the road only where the canopy lets it
through, and the sunlit ground and trees light what lies in their shade.

The scene is the film's, its renderer's own procedural trees and materials on its exported data:
- Fetched at start from raggare.kansei.graphics (CORS open; `data=<base URL>` for another copy),
  about 25 MB, none of it in this repository:
  - the seed export: `data/scene.json`, the terrain's heights (1513 x 1009 at 1 m) and splat, the
    road's centre line, the 26 390 spruces and birches and the verge's grass, flowers and shrubs;
  - the CC0 ground and asphalt scans (`textures/ground`, `textures/road`, KTX2), whose licences
    are below.
- Generated here, ported from the film's renderer (raggare-web's `crates/intro`, the same
  author's):
  - the trees (`tree_meshes.rs`, `tree_textures.rs`, `canvas.rs`): Norway spruces in three crown
    styles and silver birches, three LODs each, and their foliage atlas and bark painted on the CPU;
  - the ground cover (`cover_meshes.rs`, `cover_textures.rs`);
  - the materials, in `examples/forest/*.wgsl` at the repository's root, shared with the TS
    page (`examples/index_outdoor_gi.html`, whose `forest/*.js` port the generators): the
    terrain's height-blended scans, the road's asphalts, lines, repairs and cracks, the trees and
    the cover, lit here by the sun (cascades) and the sky and written to the GBuffer and the GI's
    voxels in place of the film's dusk and headlights.
- Left out: the car, the lake's cottage and jetty, the roadside posts and poles, the forest-floor
  debris, the road's decals and the film's sway and breeze (so the shadow maps, the voxels and the
  ray tracing grid hold the trees the camera sees).

`forest.rs` builds it:
- the terrain as 96 tiles of the heights every 2 m (the film's resolution);
- per species three LODs, culled on the GPU per view with dithered crossfades; the spruces' foliage
  as card clusters (cluster LOD, cut at 16 px of error); an octahedral impostor per layer baked from
  LOD0 at start; their bands set each frame for the lens, as the film's (the screen-size thresholds
  of its median tree);
- the clipmap voxelizing LOD0 and the cards within 20 m, LOD1 to 60 m and LOD2 past that; the ray
  tracing grid LOD0's bark, the birches' leaves and the spruces' cards, the foliage alpha-tested by
  the atlas.

The sun shadows through four cascades of 1024² out to 160 m; foliage reads them through four
comparison taps rather than the cascades' PCSS (`foliage_sun_shadow` in forest.wgsl: alpha-tested,
every layer of it shades). The sky is an atmosphere, with its aerial perspective and a light mist.

The GI modes (`gi=`) compare what lights the shade:

| Mode | What lights the surfaces besides the sun |
|---|---|
| `off` | the materials' own sky light: the whole sky, as if no tree stood over them |
| `skyocc` | that sky light dimmed by the top-down sky occlusion (`Renderer::enable_sky_occlusion`), the film's technique |
| `visibility` | that sky light dimmed by the sky visibility the clipmap's probes measure, in the material (`kansei_clipmap_sky_visibility`) |
| `cones` | voxel GI on screen, six cones a pixel through the clipmap: the sky past the canopy and the bounces |
| `probes` | the same light from the clipmap's irradiance probes, cheaper and smoother |
| `ssgi` | screen-space GI (`ScreenSpaceGIEffect`): the bounces between what is on screen, the sky taken out where it is hidden |
| `rt` (default) | the hybrid (`RtDiffuseGiEffect`): one ray for each 2 x 2 pixels through the grid of the scene's triangles round the camera (below; the foliage alpha-tested), for 8 m, then the clipmap and the sky; the hits lit by the sun (shadow rays, the cascades past the grid) and a cone through the clipmap; denoised by SVGF. Closest to a path-traced reference: the cones are too bright under the canopy |

GPU time at 1920 x 1080 (device pixel ratio 1) on an Apple M4 Pro, Chrome, the default view
(`cam=departure`), passes summed by the profiler (`stats=1`), October 2026:

| `gi=` | Rust / WASM | TS |
|---|---|---|
| `rt` | ~35 ms | ~30 ms |
| `cones` | ~33 ms | ~27 ms |
| `ssgi` | ~36 ms | ~24 ms |
| `off` | ~31 ms | ~23 ms |

Most of it is the GBuffer (16-21 ms: the spruces' cards, many layers deep at the forest's edge,
lose the GPU's hidden-surface removal to their alpha test) and the cascades (4-5 ms); the hybrid
adds about 7 ms (trace 5, SVGF 2). Other sessions share this GPU, so runs differ by 10-20%.

The voxel clipmap:
- 5 levels of 64 × 32 × 64 voxels, from 0.5 m voxels over 32 m to 8 m voxels over 512 m.
- Each level's window follows the camera, rewriting only the slab it moves into.
- It is lit through the cascades, or cones through itself past them.

With `reflect=1` the road and the lake are wet and reflect the forest: rays traced through the grid
(64 x 32 x 64 m of 0.5 m cells), the hits lit by the clipmap, the voxel cone past the grid.

Engine API:
- `Renderer::enable_voxel_clipmap` (`SceneVoxelClipmapOptions`), `SceneVoxelClipmap`
  (`settings`: `ClipmapGiSettings`, `ConeShadows`; `use_sky_lighting`, `enable_probes`:
  `ClipmapProbeOptions`).
- `VoxelGIEffect::with_clipmap`, with `set_clipmap_probes`; `ScreenSpaceGIEffect`.
- `gi::CLIPMAP_PROBES_WGSL`, `gi::VOXEL_WRITE_WGSL`'s `kansei_voxel_write` and
  `kansei_voxel_write_coverage` (the foliage's alpha test).
- `InstanceCulling` (`with_crossfade`; `lod_range`, `shadow_lod_range`, `gi_lod_range`,
  `rt_lod_range` set per frame), `culling::LOD_FADE_WGSL`, `ClusterLod` (`ClusterOptions::cards`,
  `InstanceTransform::Placement`), `Renderer::set_cluster_error_threshold`.
- `Renderer::bake_impostor` with `impostors::IMPOSTOR_WGSL` and `billboard_geometry`.
- `loaders::ktx2::transcode` (2D and array textures).
- `Renderer::enable_sky_occlusion` with `shadows::SKY_OCCLUSION_WGSL`.
- `Renderer::enable_cascaded_shadows` with `shadows::CASCADED_SHADOWS_WGSL`.
- `SkyAtmosphere` with `AtmosphereEffect`; `VolumetricFogEffect` (`set_clipmap_probes`,
  `set_sky_occlusion`); `TemporalAAEffect`, `ToneMapEffect`.
- `Renderer::enable_rt_grid` (`SceneRtGridOptions`), `Renderable::rt` (`RtSurface`) and
  `rt_placement` (`RtPlacement::Wgsl`: the trees' records widen them by a hash of where they stand,
  which `InstanceTransform` can't say).
- `RtDiffuseGiEffect::with_clipmap` (`near_distance`; `set_alpha_texture`,
  `set_cascaded_shadow_map`, `set_sky_lighting`, `update_lights`).
- `RtReflectionsEffect::with_clipmap` and `GBUFFER_OUT_WGSL`'s `kansei_gbuffer_out_specular`.
- `Renderer::set_profiling` / `take_profile`.

| URL parameter | Effect |
|---|---|
| `gi=off\|skyocc\|visibility\|cones\|probes\|ssgi\|rt` | what lights the shade (default `rt`); `rt` builds the grid (as `rt=1` does), and the panel reloads the page with it when picked without |
| `rtgi_near=<metres>` | how far the hybrid's rays walk the grid before the clipmap takes over (default 8; 0: the whole 64 m box, about 1.5 times the cost) |
| `rtgi_res`, `rtgi_denoise`, `rtgi_kernel`, `rtgi_hit`, `rtgi_shadows`, `rtgi_mode`, `rtgi_accum`, `rtgi_view` | the hybrid's other settings, as in gi-box's README (`set_rtgi(key, value)` at run time) |
| `view=lit\|indirect\|voxels` | the lit image (default); only the light the GI adds; the clipmap's voxels and their light |
| `cam=canopy\|trunks\|branches\|lake\|headlights\|departure\|rise\|film\|drive` | a shot of the film as the starting view, halfway through it, on its lens (default `departure`), then orbit controls; `film` plays the shots in turn on the film's timeline; `drive` drives along the road at 12 m/s, so the clipmap's windows move |
| `elevation=<degrees>` | the sun's elevation (default 14; the exposure follows) |
| `bearing=<degrees>` | the sun's bearing (default 110) |
| `ev=<EV100>` | exposure (default: the sun's, opened up 2.3 stops for the forest's shade) |
| `fog=<density>` | volumetric mist (default 0.0015; 0: none), its ambient from the probes in the clipmap's modes |
| `lod=<scale>` | how many times nearer than the film's thresholds the trees' LODs switch (default 1.5, the film's Medium preset) |
| `shadow_lod=<scale>` | how many times nearer still they switch in the shadow maps (default 2) |
| `card_error=<pixels>` | the cards' cluster cut (default 16; the shadow maps four times that) |
| `shadow_res=<texels>` | the cascades' resolution (default 1024) |
| `impostor_shadows=0` | the impostors cast no shadows |
| `hide=<labels>` | leave parts out, by label (`Terrain`, `Road`, `Lake`, `Spruce`, `Birch`, `/Bark/`, `/Foliage/`, `/Cards`, `TreeImpostor`, `Grass`, `Flowers`, `Shrubs`; `set_hidden` at run time), to measure them |
| `levels=<n>` | clipmap levels (default 5) |
| `res=<voxels>` | voxels across each level (default 64; half as many in height) |
| `vox=<metres>` | the finest level's voxels (default 0.5) |
| `lit=<n>` | levels lit a frame (default 2; 0: all) |
| `probes=<n>` | probes traced a frame (default 8192) |
| `coneshadows=off\|fallback\|always` | the voxels' shadows from cones through the clipmap: never, where the cascades don't reach (default), always |
| `bounce=<share>` | share of last frame's light the voxels bounce again (default 1) |
| `shadowsteps=<n>` | steps of the voxels' shadow cones (default 48) |
| `intensity=<scale>` | scale of the light the voxel GI adds (default 1) |
| `ssgi_radius=<metres>` | the screen-space GI's search radius (default 6) |
| `shadow_far=<metres>` | the cascades' reach (default 160) |
| `reflect=1` | ray-traced reflections on the wet road and lake (builds the grid: `rt=1`) |
| `wet=all` | everything wet, not only the road and the lake |
| `wet_f0=<0..1>`, `wet_rough=<0..1>` | the wet surfaces' F0 (default 0.04) and roughness (default 0.1) |
| `rt_view=lit\|reflection\|mirror\|cost` | the lit image (default), the light the reflections add, what the rays see, their cost (cells and triangles a ray) |
| `rt_trace=voxels` | reflections from the voxel cone alone (for comparison) |
| `rt_alpha=0` | the foliage solid in the reflections (no alpha test) |
| `rt_res=quarter` | trace one pixel of each 4 x 4 a frame (default 2 x 2) |
| `rt=1` | build a ray tracing grid of the scene round the camera (64 x 32 x 64 m): the trees culled for its box and cut at a cell of error, on the GPU; the stats show its triangles and build |
| `rt_cell=<metres>` | the grid's cells (default 0.5; the box stays 64 m across) |
| `rt_rebuild=1` | rebuild the grid every frame, not only when its box moves |
| `data=<base URL>` | where the intro's `data/` and `textures/` are served (default `https://raggare.kansei.graphics/`) |
| `stats=1` | overlay: the clipmap, triangles, frame interval and each pass's GPU time |
| `ui=0` | hide the panel |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom, right-drag or shift-drag to pan. A Tweakpane
panel (loaded from jsDelivr) switches the GI mode, the view, the camera and the sun, and with the
grid the hybrid's settings. `window.kansei` exposes the setters and `info()` for scripted
captures.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.

## Licences of the fetched textures

The scans raggare.kansei.graphics serves (and this example fetches) are CC0; raggare-web's
`app/public/textures/LICENSES.md` records them. Its trees, ground cover and materials are the
film's own procedural ones; no Unreal, Fab, Megascans, Quixel or Starter Content asset is used.

- `textures/road/`: four [ambientCG](https://ambientcg.com) 1K-JPG sets, licensed
  [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/), downloaded 2026-09-27
  and repacked as KTX2 (colour; normal X and Y, roughness and occlusion):
  [Asphalt015](https://ambientcg.com/view?id=Asphalt015) (worn base),
  [Asphalt031](https://ambientcg.com/view?id=Asphalt031) (weathered variation),
  [Asphalt010](https://ambientcg.com/view?id=Asphalt010) (repair patches),
  [Gravel043](https://ambientcg.com/view?id=Gravel043) (gritty edges).
- `textures/ground/`: six ambientCG 1K-JPG sets, CC0 1.0, downloaded 2026-09-27, the layers of two
  KTX2 arrays in this order: [Grass004](https://ambientcg.com/view?id=Grass004) (meadow),
  [Ground048](https://ambientcg.com/view?id=Ground048) (needle litter),
  [Ground037](https://ambientcg.com/view?id=Ground037) (moss),
  [Ground081](https://ambientcg.com/view?id=Ground081) (gravel),
  [Ground103](https://ambientcg.com/view?id=Ground103) (dirt),
  [Ground054](https://ambientcg.com/view?id=Ground054) (sand). Each scan's colour is scaled to a
  mean albedo (`GROUND_SURFACES` in `forest.rs`).
