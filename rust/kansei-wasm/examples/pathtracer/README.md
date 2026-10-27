# Path tracer

A Cornell box (white floor, ceiling and back wall, red and green side walls) with two diffuse
boxes and a glass Stanford dragon, path traced on the GPU against a BVH. A low sun shines in
through the open front. One sample a pixel a frame goes through `PathTracerDenoise` (SVGF over
the tracer's own guide buffers, then a temporal anti-aliasing resolve), so the image stays clean
while the camera or objects move. With the denoiser off, samples accumulate while nothing
changes and moving the camera, an object or the light starts the accumulation again. Moved
objects only rebuild the top-level BVH.

Engine API: `kansei_wasm::{Canvas, run, fetch_bytes}`, `GLTFLoader::load_glb`,
`Renderable::path_tracer_material` with `PathTracerMaterial` and `PathTracerMaterial::glass`,
`BVHBuilder` (`build_full`, `refresh_transforms`, `scene_materials`), `TLASBuilder`,
`PathTracer` (`set_materials`, `set_lights_from_scene`, `set_spp`, `set_max_bounces`,
`set_use_blue_noise`, `set_jitter`, `set_guides`, `trace_frame`, `present`,
`reset_accumulation`), `PathTracerDenoise` (`denoise_frame`, `present`, `reset`, its mode,
view, kernel and strengths), `Keys`, `DirectionalLight`,
`PostProcessingVolume` with `ToneMapEffect` (`ToneMapper::KhronosNeutral`), and
`CameraControls`.

| URL parameter | Effect |
|---|---|
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |
| `denoise=svgf\|temporal\|off` | the denoiser's mode (default `svgf`; `off` accumulates instead) |
| `view=<name>` | what the screen shows: `final`, `noisy`, `temporal`, a guide buffer (`diffuse-albedo`, `specular-albedo`, `normal`, `depth`, `motion`, `diffuse`, `specular`), `variance` or `history` |
| `kernel=3x3\|5x5` | the wavelet's kernel |
| `iterations=<n>` | the wavelet's iterations (default 5) |

Controls: drag to orbit, wheel or pinch to zoom; `d` turns the denoiser on and off. The
Tweakpane panel's Denoiser folder sets the mode, the view (the guide buffers and the variance
to look at), the kernel, iterations, temporal alpha, the colour, normal and depth stops, a
history clamp, the TAA resolve and the debug views' scale. Below it the panel sets samples per
pixel, maximum bounces, blue-noise sampling and the sub-pixel jitter, has a button that resets
the accumulation, and shows the accumulated frame count. Its Light folder sets the sun's intensity, direction and colour.
Its Scene folder turns on an orbit for box B and moves box A, box B and the dragon (an offset
from where it loaded). An FPS counter sits at the top left.

Assets: `www/assets/stanford_dragon_pbr.glb`, "Stanford Dragon (Vrip)" by 3D graphics 101 on
Sketchfab, CC-BY-NC-4.0 (`www/assets/license.txt`). The page loads Tweakpane 4 from
cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
