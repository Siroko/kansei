# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Where work happens

- Active development is the Rust engine in `rust/` (workspace: `kansei-core`, `kansei-wasm`, `kansei-native`); the TypeScript library in `src/` is the older port. Open PRs against `development`, not `main`.
- `rust/kansei-wasm/examples/*` are standalone crates, each with its own `[workspace]` and listed under `exclude` in `rust/Cargo.toml`. Build one with `wasm-pack build --target web --release` in its directory, serve that directory, and open `www/`.

## Verifying

- `cargo test -p kansei-core` validates every WGSL module with naga and checks each `#[repr(C)]` uniform struct against its WGSL size (see the `shaders_validate_*` tests); keep that pattern for new shaders.
- `rust/kansei-core/tests/*_gpu.rs` run shaders on a real adapter and read results back; they pass without running when no adapter exists.
- Headless browser checks: run `chrome-devtools-axi` with its own `CHROME_DEVTOOLS_AXI_SESSION` and `CHROME_DEVTOOLS_AXI_CHROME_ARGS="--enable-unsafe-webgpu --enable-gpu --ignore-gpu-blocklist"`, and serve examples on a port nobody else uses.

## Sharp edges

- `queue.write_buffer` lands before the next submit: several writes to one buffer inside a submit leave only the last one for every pass. Give per-dispatch parameters their own buffers.
- Depth is `[0, 1]` (`glam::Mat4::perspective_rh`), cleared to 1.0, so a depth of 1.0 means sky. Front faces are counter-clockwise and materials cull back faces by default.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
