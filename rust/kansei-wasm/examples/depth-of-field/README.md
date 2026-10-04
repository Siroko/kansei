# Depth of field

Physical depth of field: a forest-like depth set (trunks from 6 to 120 m, alpha-tested leaf cards
near and far, strings of small lights behind the subject) under a low sun, seen through a
`CameraLens` on Unreal's 23.76 mm filmback. The circle of confusion follows from the focal length,
f-stop and focus distance; bokeh keep their energy and the aperture's shape.

Engine API: `CinematicDepthOfFieldEffect` (`CinematicDepthOfFieldOptions`, `CameraLens`,
`HighlightOptions`, `DofDebugView`), `TemporalAAEffect`, `Renderer::set_render_scale`,
`SkyAtmosphere` with `AtmosphereEffect` and `SKY_LIGHTING_WGSL`, `Material::emissive`,
`Renderer::enable_shadows`, `ToneMapEffect`.

| URL parameter | Effect |
|---|---|
| `focal=<mm>` | focal length on the 23.76 mm filmback (default 50); sets the field of view |
| `fstop=<f-number>` | default 1.8 |
| `focus=<metres>` | focus distance (default 8, the subject) |
| `rack=1` | pull focus from 2 to 30 m and back every 8 s (any value turns it on) |
| `t=<seconds>` | fix the rack focus's time, for stills |
| `blades=<n>` | aperture blades (default 0: round bokeh) |
| `samples=<n>` | gather samples per half-resolution pixel (default 72) |
| `scatter=0` | gather the highlights too instead of scattering them as bokeh sprites |
| `ev=<EV100>` | exposure (default 10.8) |
| `dof=0` | no depth of field |
| `taa=0` | no temporal anti-aliasing |
| `order=after` | run the depth of field after the TAA instead of before it |
| `debug=<1..4>` | 1 background layer, 2 near layer, 3 near alpha, 4 CoC (3 and 4 shown unexposed) |
| `scale=<ratio>` | render scale, 0.25 to 1 (default 1): the TAA upscales to the canvas |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera is fixed. The page shows its URL parameters at the bottom left.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
