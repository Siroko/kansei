# Specular factor

The same pieces twice, side by side under a low sun ahead of the camera: on the left with the
default specular factor (1: a dielectric's Fresnel, F0 0.04 rising to 1 at grazing angles, so a
highlight on the balls and a sheen on the board and the sloped plank), on the right with a factor
of 0, matte (Lambert alone). Each side has a board, three balls of roughness 0.35, 0.6 and 0.85,
and a plank.

Engine API: `StandardLitOptions::specular` (glTF's `KHR_materials_specular` `specularFactor`,
which the glTF loader reads into `GLTFMaterialInfo::specular`), and with `lighting=rt`
`RtShadowsEffect` reading it from the GBuffer, written by `kansei_gbuffer_with_specular`
(`materials::GBUFFER_OUT_WGSL`) round `kansei_gbuffer_out_rt_lit`.

| URL parameter | Effect |
|---|---|
| `lighting=rt` | the sun's light added by `RtShadowsEffect` with ray-traced shadows, instead of the stock material's own lighting and shadow map |
| `matte=<factor>` | the right side's factor (0 by default) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`. The native
twin is `cargo run -p kansei-native --example specular` (`--rt`, `--matte <factor>`; `--bench`
times the GBuffer pass and `RtShadowsEffect` headless).
