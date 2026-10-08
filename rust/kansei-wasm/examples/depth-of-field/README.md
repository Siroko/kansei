# Depth of field

Physical depth of field on a scrolling field of 150 × 150 noise-driven columns, some of whose tops
glow: small bright sources that scatter as bokeh. It is the Rust twin of the TS
`examples/index_dof_terrain.html`, with the same scene, camera, lens, panel and URL parameters; a
view copied on either page (`Copy view` or the V key) opens on both. The circle of confusion follows
from a `CameraLens` on Unreal's 23.76 mm filmback. A 150-unit terrain seen through a lens that frames
it would be nearly all in focus, so the lens is a long one (450 mm at f/1.4), set apart from the
view's 26° field of view: the shallow focus of a miniature.

Like the TS page, it draws through a 4× MSAA GBuffer at one pixel per CSS pixel, and the image goes
to the screen as the chain leaves it: no exposure, tone curve or sRGB encode.

Engine API: `CinematicDepthOfFieldEffect` (`CinematicDepthOfFieldOptions`, `CameraLens`,
`HighlightOptions`, `DofDebugView`), `DepthOfFieldEffect` (`dof=simple`), `SSAOEffect`,
`ToneMapEffect` with `ToneMapper::None`, `InstancedGeometry` with a per-instance `vec4`
(`ComputeBuffer::with_vertex_vec4`), `GBUFFER_OUT_WGSL`, `CameraControls`.

| URL parameter | Effect |
|---|---|
| `dof=simple` | the simple DoF (a focus range and a blur in pixels) instead of the cinematic one; `dof=0` none |
| `focal=<mm>` | focal length on the 23.76 mm filmback (default 450); the view's fov frames the picture |
| `fstop=<f-number>` | default 1.4 |
| `focus=<units>` | focus distance, view depth (default the view's, 101.5) |
| `blades=<n>` | aperture blades (default 0: round bokeh) |
| `samples=<n>` | gather samples per half-resolution pixel (default 72) |
| `scatter=0` | gather the highlights too instead of scattering them as bokeh sprites |
| `debug=<1..4>` | 1 background layer, 2 near layer, 3 near alpha, 4 CoC |
| `t=<seconds>` | fix the scroll at this time, for stills |
| `view=<JSON>` | the opening view (`Copy view` or the V key) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default 1, as the TS page) |

Controls: drag to orbit, scroll to zoom; the panel sets the lens (or, with `dof=simple`, the focus
range and blur). `window.kansei.set(key, value)` sets any view JSON key from the console.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
