# Hello scene

The smallest complete Kansei scene: a floor, a spinning box and a sphere in the stock lit
material, a sun and two point lights, shadows, and an orbit camera.

Engine API: `kansei_wasm::{Canvas, run}`, `Material::basic_lit`, `DirectionalLight`,
`PointLight` (`cast_shadow`), `Renderer::enable_shadows` / `enable_point_shadows`,
`CameraControls`, and with `post=1` `PostProcessingVolume` with `BloomEffect` and
`ColorGradingEffect`.

| URL parameter | Effect |
|---|---|
| `shadows=0` | no shadow maps |
| `post=1` | render through a post-processing volume: bloom on the bright box, then a colour grade |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`. The native
twin is `cargo run -p kansei-native --example hello_scene`.
