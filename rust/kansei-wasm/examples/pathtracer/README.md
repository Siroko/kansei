# Path tracer

A Cornell box (white floor, ceiling and back wall, red and green side walls) with two diffuse
boxes and a glass Stanford dragon, path traced on the GPU against a BVH. A low sun shines in
through the open front. Samples accumulate while nothing changes; moving the camera, an object
or the light starts the accumulation again. Moved objects only rebuild the top-level BVH.

Engine API: `kansei_wasm::{Canvas, run, fetch_bytes}`, `GLTFLoader::load_glb`,
`Renderable::path_tracer_material` with `PathTracerMaterial` and `PathTracerMaterial::glass`,
`BVHBuilder` (`build_full`, `refresh_transforms`, `scene_materials`), `TLASBuilder`,
`PathTracer` (`set_materials`, `set_lights_from_scene`, `set_spp`, `set_max_bounces`,
`set_use_blue_noise`, `trace_frame`, `present`, `reset_accumulation`), `DirectionalLight`,
`PostProcessingVolume` with `ToneMapEffect` (`ToneMapper::KhronosNeutral`), and
`CameraControls`.

| URL parameter | Effect |
|---|---|
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom. The Tweakpane panel sets samples per pixel,
maximum bounces and blue-noise sampling, has a button that resets the accumulation, and shows
the accumulated frame count. Its Light folder sets the sun's intensity, direction and colour.
Its Scene folder turns on an orbit for box B and moves box A, box B and the dragon (an offset
from where it loaded). An FPS counter sits at the top left.

Assets: `www/assets/stanford_dragon_pbr.glb`, "Stanford Dragon (Vrip)" by 3D graphics 101 on
Sketchfab, CC-BY-NC-4.0 (`www/assets/license.txt`). The page loads Tweakpane 4 from
cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
