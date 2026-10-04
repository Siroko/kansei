# Cascaded shadows

A forest of 4000 instanced trees and a fence under a low afternoon sun, shadowed by four 2048²
cascades out to 250 m with contact-hardening (PCSS) penumbrae, through TAA and the tonemapper.
The trees are culled per cascade on the GPU. The camera walks down a path through the forest.

Engine API: `Renderer::enable_cascaded_shadows` (`CascadedShadowOptions`),
`Renderable::instanced_culled`, `Material::standard_lit` (`StandardInstancing::OffsetScale`),
`Material::gradient_sky`, `TemporalAAEffect`, `ToneMapEffect`, `exposure_from_ev100`; with `csm=0`
`Renderer::enable_shadows`, with `debug=1` `shadows::CASCADED_SHADOWS_WGSL`, with `fog=1`
`VolumetricFogEffect::set_cascaded_shadow_map`.

| URL parameter | Effect |
|---|---|
| `csm=0` | the single 2048² directional shadow map instead of the cascades, for comparison |
| `debug=1` | the sun's light only, each surface tinted by the cascade that shadows it |
| `fog=1` | volumetric fog, with light shafts shadowed by the cascades (or the single map with `csm=0`) |
| `far=<metres>` | camera far plane, which the single map is fitted to (default 1200) |
| `t=<seconds>` | freeze the camera at this point of its walk |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera moves on its own.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
