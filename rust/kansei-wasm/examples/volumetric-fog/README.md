# Volumetric fog

Froxel volumetric fog in a grove of pillars: a low shadowed sun whose light comes through in
shafts, a warm lamp in the clearing with cube shadows, a cool one without, and a dim ambient sky
term, drifting with the wind. The camera circles the grove.

Engine API: `VolumetricFogEffect` (`VolumetricFogOptions`, `FroxelGridOptions`, `set_shadow_map`,
`set_point_shadows`, `update_lights`, `time`), `Renderer::enable_shadows` /
`enable_point_shadows` / `shadow_map` / `cubemap_shadow_map`, `PostProcessingVolume`,
`DirectionalLight`, `PointLight` (`cast_shadow`), `Material::basic_lit`.

| URL parameter | Effect |
|---|---|
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera moves on its own.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
