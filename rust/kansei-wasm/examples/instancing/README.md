# Instancing

A clone of [square.felixmartinez.dev](https://square.felixmartinez.dev/) (three.js) on Kansei: a
carpet of 128 × 128 cubes, one instanced draw, that an orange ball ploughs through. A compute
shader steps every cube at 60 Hz (pushed out of the ball's way within 6.5 m, drawn towards it
beyond, springing back home) and writes its position straight into the instance buffer the cubes
are drawn from, so the positions never leave the GPU. A shadowed spot light lights it; the cubes
shadow each other and the floor through their own vertex shader.

The look is the original's: a random palette row per load colours the cubes and the fog, lit in
three.js's unmanaged colour pipeline (sRGB numbers lit as they are, ACES-fitted, written without
encoding), then a linear distance fog and a vignette in display space (`FogVignetteEffect`, in
this crate). The camera's near and far planes are the original's (1 mm, 100 m): the woven grain
on the carpet is its overlapping cubes' faces fighting for depth.

Engine API: `ComputeBuffer::with_vertex_vec4` (one buffer that is both the compute shader's
storage and the instance attribute; clones share it), `ComputePass` with `Renderer::compute`,
`pacing::FixedStep`, `Material::standard_lit` with `StandardInstancing::OffsetScale`,
`SpotLight` with `Renderer::enable_spot_shadows`, `ToneMapEffect`, and a `PostProcessingEffect` of
its own. The TS twin is `examples/index_instancing.html`.

| URL parameter | Effect |
|---|---|
| `n=<cubes per side>` | default 128, 8 to 512 |
| `palette=<row>` | the palette row, 0 to 10 (default: random) |
| `shadows=0` | no spot shadow map |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`. The native
example of the same name (`cargo run -p kansei-native --example instancing`) is the plain
instancing demo this one replaced: a 10 × 10 × 10 grid of turning cubes, its matrices rewritten
by the CPU.
