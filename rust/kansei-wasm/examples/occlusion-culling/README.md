# Occlusion culling

A dense procedural forest of spruces (60 000 by default, three mesh LODs) on terrain with a ridge
and a hill. Each LOD is culled on the GPU per view by frustum and LOD band, and for the camera
also by occlusion: two phases against a depth pyramid built from the terrain and the trees seen
last frame.

Engine API: `InstanceCulling` (`with_lod_range`, `with_occlusion`, `with_bounds_box`),
`Renderer::set_occlusion_culling` / `set_freeze_culling` / `culling_stats`, with `skyocc=`
`Renderer::enable_sky_occlusion` (`SkyOcclusionOptions`, `shadows::SKY_OCCLUSION_WGSL`),
`Material::gradient_sky`, `TemporalAAEffect`, `ToneMapEffect`, and for the diagnostics
`pacing::FrameTimer` and `profiling::AbBench`.

| URL parameter | Effect |
|---|---|
| `cam=valley\|forest\|ridge\|high\|fly\|sky\|edge` | starting camera (default `valley`); `fly` loops through the valley and over the ridge; `sky` looks straight up, so nothing is in view and only occlusion's overhead remains |
| `trees=<n>` | spruces (default 60 000) |
| `occlusion=0` | start with occlusion culling off |
| `freeze=1` | start with the camera's culling frozen |
| `bounds=sphere` | cull by spheres round the trees' bases instead of boxes |
| `scale=<ratio>` | render scale, 0.25 to 1 (default 1): the TAA upscales to the canvas |
| `taa=0` | no temporal anti-aliasing |
| `t=<seconds>` | freeze the `fly` path at that time |
| `size=<w>x<h>` | draw at exactly that many pixels whatever the window's size |
| `skyocc=1` | the trees occlude the sky's light: the sky ambient dims under the canopy |
| `skyocc=show` | show the sky visibility instead of the shading |
| `skyocc=rebuild` | diagnostic: as `skyocc=1`, starting a rebuild every frame, to time a rebuild's first frame |
| `bench=1` | diagnostic: alternate occlusion on and off every 3 s, 8 times, and report the mean GPU time and frame interval of each |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: O (or the panel's checkbox) toggles occlusion culling, F (or its checkbox) freezes the
camera's culling so a moving or switched camera shows what was culled, C cycles the camera. The
HUD shows the camera's culling stats (tested, outside the LOD band, outside the frustum,
occluded, drawn) and the GPU and CPU time.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
