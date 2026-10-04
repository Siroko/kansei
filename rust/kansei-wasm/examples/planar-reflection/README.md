# Planar reflection

A still lake at dusk mirrors the far shore's treeline, a red cottage with lit windows and the sky
through a `PlanarReflection`: a mirrored camera with an oblique clip plane at the water, the water
itself left out by its layer. The water material adds Fresnel, wind ripples that displace the
lookup and a roughness that picks the reflection's mips. A searchlight on the far bank throws a
beam through volumetric fog, and the fog is composited into the reflection too.

Engine API: `PlanarReflection` (`PlanarReflectionOptions`, `occlusion_culling`, `screen_space`,
`set_fog`), `Renderer::add_planar_reflection`, `reflections::PLANAR_REFLECTION_WGSL`,
`VolumetricFogEffect` (`reflection_fog`, `set_spot_lights`), `SpotLight`,
`Material::standard_lit`, `Renderable::instanced_culled`, `BloomEffect`, `ToneMapEffect`.

| URL parameter | Effect |
|---|---|
| `ripples=<strength>` | wind ripples (default 0.03; 0: a mirror) |
| `rough=<0..1>` | the water's roughness, which picks the reflection's mip (default 0.05) |
| `fogrefl=0` | no fog in the reflection: the water fogs the reflected path with a flat colour instead |
| `occlusion=1` | the treeline occlusion-culled, for the camera and in the mirror; the culling stats logged every 120 frames |
| `screen=1` | the reflection from the screen (`PlanarReflection::screen_space`) instead of the mirrored view |
| `t=<seconds>` | freeze the camera's pan at that time (the ripples and the fog keep moving) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the camera pans slowly along the far treeline.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
