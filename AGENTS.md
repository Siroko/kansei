# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Where work happens

- Active development is the Rust engine in `rust/` (workspace: `kansei-core`, `kansei-wasm`, `kansei-native`); the TypeScript library in `src/` is the older port. Open PRs against `development`, not `main`.
- `rust/kansei-wasm/examples/*` are standalone crates, each with its own `[workspace]` and listed under `exclude` in `rust/Cargo.toml`. Build one with `wasm-pack build --target web --release` in its directory, serve that directory, and open `www/`.
- Textures ship as KTX2 (Basis Universal), transcoded per device by `loaders::ktx2`; encode with `rust/tools/ktx2` (needs `basisu`). See `docs/ktx2.md`.

## Verifying

- `cargo test -p kansei-core` validates every WGSL module with naga and checks each `#[repr(C)]` uniform struct against its WGSL size (see the `shaders_validate_*` tests); keep that pattern for new shaders.
- `rust/kansei-core/tests/*_gpu.rs` run shaders on a real adapter and read results back; they pass without running when no adapter exists.
- Headless browser checks: run `chrome-devtools-axi` with its own `CHROME_DEVTOOLS_AXI_SESSION` and `CHROME_DEVTOOLS_AXI_CHROME_ARGS="--enable-unsafe-webgpu --enable-gpu --ignore-gpu-blocklist"`, and serve examples on a port nobody else uses.
- GPU cost in the browser: wgpu 24's WebGPU backend does not implement `queue.on_submitted_work_done`. Add `--disable-gpu-vsync --disable-frame-rate-limit` to the Chrome args and time the interval between frames. For a breakdown, `Renderer::set_profiling(true)` then `take_profile().report()`: every labelled pass's GPU time (`profiling::gpu_pass` on new passes) and the frame's CPU sections; add `--enable-webgpu-developer-features` for unquantized timestamps. Other sessions share this Mac's GPU, so separate runs differ by tens of percent: alternate A and B inside one page (the occlusion-culling example's `bench=1`). On Apple GPUs consecutive passes overlap, so time a span across passes rather than each pass's own timestamps. Leave no animating WebGPU page open between captures (load `about:blank`).

## Sharp edges

- `queue.write_buffer` lands before the next submit: several writes to one buffer inside a submit leave only the last one for every pass. Give per-dispatch parameters their own slots (or buffers).
- Each `queue.write_buffer` costs tens of microseconds in Chrome: upload a frame's data in one write, not one per dispatch or object (`culling::InstanceCulling` writes all its views once a frame, into one buffer its dispatches index).
- Depth is `[0, 1]` (`glam::Mat4::perspective_rh`), cleared to 1.0, so a depth of 1.0 means sky. Front faces are counter-clockwise and materials cull back faces by default.
- The GBuffer's four MRT targets fill WebGPU's default 32 bytes per sample (rgba8unorm counts 8), and wgpu 24's web backend cannot request more: another per-pixel output needs its own pass, as the velocity pass does.
- Shadow, reflection and velocity passes redraw each material through its own `vertex_main`, with the light or mirror as the camera (see `Material::get_depth_pipeline`). Vertex shaders must not read group 3. Mark `@builtin(position) @invariant` where a second pass depth-tests against the GBuffer.
- Group 3 (shadows and lights) only grows by additive bindings; `renderers/shared_layouts.rs` lists them.
- Instances culled on the CPU for the camera also drop out of shadow maps and reflections. Use `culling::InstanceCulling`, which culls per view on the GPU.
- The `target/` directories of a few older examples (spinning-box, lit-scene, shadow-scene, ...) are committed: building those examples modifies tracked files, so restore them with `git checkout -- <example>/target` before committing.
- Cluster LOD (`Renderable::clusters`) draws only the camera's pass until M3: other views draw the renderable's geometry, which must be the mesh its graph was built from.
- Skinned materials (`animation::SKINNING_WGSL`) skin in their own `vertex_main`, but the single directional shadow map (`enable_shadows`) draws casters with a shared depth shader: a skinned mesh casts its bind pose there, so use cascaded shadows.
- Never commit animation data, character meshes or anything derived from them (glTF exports, `.kmm` packs) unless their licence allows a public MIT repo: third-party sets such as Epic's GASP stay in a private folder, and the motion-matching example loads packs from a local path (`rust/kansei-anim-bake/README.md`).

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
