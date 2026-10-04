# Film look

A dusk path between tree trunks, lit by three lamps, in physical units (cd/m²) through the full
HDR chain: volumetric fog, physically based bloom, then the tonemapper with EV100 exposure, a
filmic curve, vignette, grain, chromatic aberration, sRGB encoding and dither. The camera walks
slowly toward the lamps.

Engine API: `ToneMapEffect` / `ToneMapOptions` (`tonemapper`, `exposure`, `vignette`, `grain`,
`chromatic_aberration`, `local_exposure`), `ToneMapper`, `exposure_from_ev100`,
`LocalExposure::unreal`, `BloomEffect::with_exposure`, `VolumetricFogEffect::set_point_shadows`,
`Material::emissive`, `Material::gradient_sky`, `Renderer::enable_point_shadows`.

| URL parameter | Effect |
|---|---|
| `tm=aces\|agx\|punchy\|neutral\|unreal\|none` | tone curve: ACES fitted (default), AgX, AgX punchy, Khronos neutral, Unreal filmic, none |
| `ev=<EV100>` | exposure (default 3.9) |
| `film=0` | no vignette, grain or chromatic aberration |
| `bloom=0` | no bloom |
| `le=<contrast>` | Unreal's local exposure at this highlight and shadow contrast, e.g. 0.8 (default off) |
| `t=<seconds>` | freeze the camera at this point of its walk |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera moves on its own.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
