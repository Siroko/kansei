# Temporal AA

Temporal anti-aliasing on a scene that aliases: thin trunks and power lines, a meadow of 12 000
alpha-tested grass cards swaying in the wind, and a car crossing the frame, under a moving
camera. The surfaces write motion vectors and the sky is reprojected by depth. Optionally the
scene renders below the canvas resolution and the TAA upscales it, and motion blur follows.

Engine API: `TemporalAAEffect`, `MotionBlurEffect` (`set_frame_time`), `ToneMapEffect` with
`exposure_from_ev100`, `Renderer::set_render_scale`, `Material::standard_lit` with
`outputs_velocity`, a custom grass material on `MaterialOptions::outputs_velocity` composing
`cameras::MOTION_VECTORS_WGSL` and `lights::LIGHTS_WGSL`, `Material::gradient_sky`,
`InstancedGeometry` over `ComputeBuffer::with_vertex_vec4`, and `Renderer::set_profiling` /
`take_profile`.

| URL parameter | Effect |
|---|---|
| `taa=0` | no TAA |
| `vel=0` | no motion vectors: the car and the grass reproject by depth only |
| `scale=<0.25..1>` | render the scene at that fraction of the canvas and let the TAA upscale it (default 1) |
| `wind=<scale>` | wind strength on the grass (default 1) |
| `t=<seconds>` | freeze the camera at that time; the car and the wind keep moving |
| `step=<seconds>` | with `t`, alternate the camera, the car and the wind between t and t + step every frame: a still picture that is moving, for comparing the blur on and off |
| `pan=<deg/s>` | swing the camera left and right at up to that speed (default 0) |
| `car=<m/s>` | the car's speed (default 9) |
| `mblur=<amount>` | add a `MotionBlurEffect` after the TAA (default 0, off; 0.5 is a 180-degree shutter) |
| `mbfps=<fps>` | the frame rate the blur is scaled to (default 30; 0 blurs by each frame's motion) |
| `mbmax=<fraction>` | longest blur as a fraction of the width (default 0.04) |
| `stats=1` | log every 240 frames the interval between frames and the renderer's profile |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera moves on its own.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
