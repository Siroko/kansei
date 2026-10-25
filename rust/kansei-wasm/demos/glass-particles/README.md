# Glass particles

Two emitters pour particles, each a small box turned to the way it moves (stretched with its
speed), through a scatter of glass spheres, boxes and pyramids over a polished floor with a
technical grid, under a spot light from above. Curl noise carries them, so the flow swirls; the
glass pushes them off by its distance fields; the pointer pushes them along its motion. A share
of them glow, and only those bloom. The glass refracts and reflects the particles, the floor's
grid, the lamp and the environment; the particles and the glass shadow the floor and streak the
beam in the mist; depth of field, TAA and SSAO finish the frame. Three lighting presets, six
palettes, and Tone.js sounds the emitters and the particles brushing the glass.

Engine API: `kansei_wasm::{Canvas, run, launch, param, param_or, flag, now}` and its performance
overlay (`kansei_wasm::perf`: `set_visible`, `profile` once a frame); `materials::Compute` and
`Material::new` (`MaterialOptions::outputs_velocity`, `transmissive`) with
`lights::SPOT_LIGHTS_WGSL`, `cameras::MOTION_VECTORS_WGSL` and `materials::GBUFFER_OUT_WGSL`;
`ComputeBuffer::from_external` with `with_vertex_layout` and `InstancedGeometry`;
`Material::standard_lit` with `StandardLitOptions::glass`, `Material::gradient_sky`; `SpotLight`
with `Renderer::enable_spot_shadows`; `Renderer::enable_voxel_gi` and `enable_rt_grid`
(`SceneRtGridOptions`, a fixed box), `set_render_scale`, `Renderable::rt` with `RtSurface::glass`,
`RtSurface::with_specular` (the floor's F0) and `RtSurface::with_record_surface` (each particle's
albedo and glow from its instance record, packed by `rt::RT_EMISSION_WGSL` and
`rt::pack_emission`), `Renderable::rt_placement` (`RtPlacement::Wgsl`); `RtReflectionsEffect`
(`set_glass`, `reflections`, `set_spot_lights`, `set_sky_lighting` with
`gi::gradient_sky_lighting`, `RtReflectionsOptions::albedo_wgsl` for the floor's grid where the
rays meet it); `ScreenSpaceGlassEffect`, `SSAOEffect`, `VolumetricFogEffect`
(`SpotScattering::Raymarched`), `TemporalAAEffect`, `CinematicDepthOfFieldEffect`, `BloomEffect`
with `BloomOptions::use_emissive`, `ToneMapEffect` in a `PostProcessingVolume`;
`CameraControls::with_mouse_pan`, `MouseVectors`.

Demo-local: the particle simulation (`src/shaders/particles_sim.wgsl`: the respawns against each
emitter's budget a frame, curl noise from analytic gradient noise, the repellers' distance fields
and their pushes, the pointer's push, the live particles compacted into the instance records, and
the events the sounds and the stats count; `particles_clear.wgsl` clears the records past the live
ones, so every pass draws as many as are alive, a bound read back a few frames late), the boxes'
material (`particles_draw.wgsl`), the floor's grid (`floor.wgsl`, `floor_grid.wgsl`, also baked
into the reflections' albedo hook), the lamp's face (one instance record the boxes' material draws,
as bright as the light it casts: I = L A), one environment (`sky_up`, `sky_down`) behind it all
that lights the particles' and the floor's ambient, the fog's and what the glass sees past the
grid, the lighting presets, the palettes, the page's panel and its sounds.

| URL parameter | Effect |
|---|---|
| `n=<particles>` | particle count, split between the emitters (default 32 768, 4096..262 144) |
| `palette=<name>` | `ember` (default), `glacier`, `neon`, `bauhaus`, `moss` or `mono` |
| `lighting=<name>` | `studio` (default), `spotlight` (the spot alone in a dark mist) or `overcast` |
| `ev=<EV100>` | exposure, over the preset's |
| `glassmode=1` | the glass in screen space (`ScreenSpaceGlassEffect`) instead of ray traced |
| `glass=0` | the repellers drawn grey instead of glass: no ray tracing grid, voxels or reflections |
| `fog=0`, `dof=0` | no mist, no depth of field |
| `scale=<0.25..1>` | render scale: the scene's passes at that share of the canvas, the TAA upscaling |
| `perf=0` | no performance overlay (shown by default; the backquote key toggles it) |
| `ui=0` | hide the panel |
| `worker=1` | run the demo in a Web Worker on an `OffscreenCanvas` (`kansei_wasm::launch`) |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 1.5) |

Controls: drag to orbit, wheel or pinch to zoom, right-drag, shift-drag or two fingers to pan;
moving the pointer over the particles pushes them. The Tweakpane panel has the palette and the
lighting preset; each emitter's transform (position, rotation), shape, radius, rate, speed,
spread, lifetime, size, glowing share and jitters; the curl flow and the pointer's push; the glass
(ray traced or screen space, reach and push of its fields, bounce, index of refraction, frost, its
shadows); the particles' look; the floor grid (its ray-traced reflections on or off, colours,
spacing, line widths); the light, the environment, the mist, AO and exposure; the lens (depth of
field on or off, focus, f-stop, bloom); the sound (off until ticked: audio starts on that gesture)
and its volume; and the stats: particles alive, emitted, glass brushes and floor hits a second,
the frame interval, the overlay and the render scale. `window.kansei` exposes every export (`info`,
`set`, `set_emitter`, `set_palette`, `set_lighting`, `set_grid_color`, `set_camera`, `set_perf`,
`take_events`), each returning a promise.

The sounds: the sim counts its respawns, the particles brushing each kind of glass (coming within
half its reach) and hitting the floor on the GPU, read back in bulk a few frames late
(`take_events`), and the page plays them on a sixteenth-note grid in A minor pentatonic: each
emitter an arpeggio while it emits, each kind of glass a bell, the floor a soft knock, at most
one note a kind a step.

The glass, two ways. Ray traced (the default): `RtReflectionsEffect`'s glass pass refracts at
every glass surface a path crosses, with total internal reflection, the tint absorbing over the
path; the paths meet the particles (their own albedo and glow), the floor (its grid and its
Fresnel reflection of the environment) and the lamp. Screen space (`ScreenSpaceGlassEffect`): each
pixel's glass taken for a ball touching it, the background bent through it from the GBuffer's copy
before the glass draws; a fraction of the cost, but one surface deep (flat faces bend too little, no
glass behind glass). `cargo run --release -p kansei-native --example glass_particles_reference --
out.ppm [samples] [width height]` path-traces the glass and the floor exactly (no particles, mist,
AO or depth of field: turn them off to compare, at the default camera).

Known limits: a glowing particle seen through the glass doesn't bloom (the bloom reads the
GBuffer's emissive target, where the glass is); hits the camera doesn't see take the lamp, the
spot light, the voxels' light and the environment's reflection, and pass through the glass's
shadows.

Assets: none. The page loads Tweakpane 4.0.5 and (when the sound is ticked) Tone.js 15.0.4 from
cdn.jsdelivr.net.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
