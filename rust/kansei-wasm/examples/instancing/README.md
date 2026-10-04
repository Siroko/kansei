# Instancing

A grid of cubes drawn in one call: one `InstancedGeometry` whose per-instance model matrices
are a vertex buffer (`ComputeBuffer::with_vertex_mat4(3)`), drawn with the stock
`Material::basic_instanced`. The CPU rewrites the matrices every frame (`ComputeBuffer::write`)
to turn the cubes.

| URL parameter | Effect |
|---|---|
| `n=<cubes per side>` | default 10, at most 40 |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: drag to orbit, wheel or pinch to zoom.

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`. The native
twin is `cargo run -p kansei-native --example instancing`.
