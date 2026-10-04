# Cluster LOD

A field of rocks (24 x 24 by default, 81 920 triangles each) drawn three ways with one material.
`clusters`: one renderable with cluster LOD, each rock's cut picked per cluster every frame.
`lods`: four discrete LODs cut from the same cluster graph, each at the error budget from its
band's near edge, switched per rock by distance. `full`: the mesh as is.

Engine API: `ClusterMesh::build` (`ClusterOptions`, `cut_geometry` with `LodView`), `ClusterLod`
(`with_transform`, `InstanceTransform::Placement`) on `Renderable::clusters`,
`Renderer::set_cluster_error_threshold`, `InstanceCulling::with_lod_range`, `IcosphereGeometry`,
and for the diagnostics `Renderer::set_profiling` / `take_profile`, `pacing::FrameTimer` and
`profiling::AbBench`.

| URL parameter | Effect |
|---|---|
| `mode=clusters\|lods\|full` | how the rocks are drawn (default `clusters`) |
| `n=<rocks per side>` | default 24 |
| `sub=<subdivisions>` | icosphere subdivisions per rock (default 6: 81 920 triangles) |
| `tau=<pixels>` | the error budget (default 1); the discrete LODs are cut at it too |
| `t=<seconds>` | freeze the camera's loop at that time |
| `size=<w>x<h>` | draw at exactly that many pixels whatever the window's size |
| `profile=1` | diagnostic: log each pass's GPU time every 240 frames |
| `bench=lods\|full` | diagnostic: alternate `clusters` and that mode every 3 s, 8 times, with the camera still through each pair, and report the mean GPU time and frame interval of each |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera loops low over the field. The HUD shows the mode, the graph's build
time, the GPU time and the frame interval.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
