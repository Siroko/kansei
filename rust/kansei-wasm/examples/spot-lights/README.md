# Spot lights

A car's dipped headlights, two shadowed 22 000 cd spot lights (10° inner and 30° outer cone,
70 m range), shine into an instanced forest at night. The trunks shadow the beams on the ground
and in the volumetric fog; they are culled per view on the GPU, so trunks outside the picture
still shadow the beams in it. Exposure is EV100 3.9.

Engine API: `SpotLight` (`cast_shadow`, `source_radius`, `volumetric_scale`, `look_at`),
`Renderer::enable_spot_shadows`, `Renderer::set_clustered_lights`, `Renderable::instanced_culled`
(GPU `culling::InstanceCulling`), `Material::standard_lit` with
`StandardInstancing::OffsetHeight`, `Material::emissive`, `VolumetricFogEffect` (`set_spot_lights`
with `Renderer::spot_lights_buffer` and `spot_shadow_atlas`, `SpotScattering`), `BloomEffect`,
`ToneMapEffect` with `exposure_from_ev100`, and `Renderer::set_profiling` / `take_profile`.

| URL parameter | Effect |
|---|---|
| `cam=<view>` | `behind` (high behind the car), `top` (above the canopy) or `wall` (a corridor to a pale wall, with a row of poles behind the camera whose shadows show that casters are culled per light); otherwise low in the forest, facing the lamps |
| `drive=1` | the car creeps forward and back |
| `t=<seconds>` | freeze the car, and the camera that follows it, at that time |
| `shadows=0` | no spot shadow maps |
| `fog=0` | no volumetric fog |
| `shafts=<steps>` | raymarch the beams per pixel with that many samples per light, instead of in the fog's froxels |
| `cull=main` | cull the trunks on the CPU against the camera only: trunks leaving the frame stop shadowing the beams, the bug per-view GPU culling avoids |
| `lamps=<n>` | n small unshadowed downlights in random colours over the forest (default 0) |
| `clusters=0` | no clustered light culling: every light shades every pixel |
| `casters=<n>` | n more renderables, one draw each, to measure per-draw CPU cost (default 0) |
| `stats=1` | log every 240 frames the interval between frames and the renderer's profile (each pass's GPU time, the CPU sections) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera is set by `cam`.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
