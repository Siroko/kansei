# Fluid Integration Implementation Plan (Plan 3)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the fluid_3d example to use engine abstractions (CameraControls, blit, post-processing) and rebuild the WASM module to use the real engine, enabling TS vs Rust performance comparison.

**Architecture:** The fluid simulation and its custom rendering (particle billboards, surface raymarching) stay as-is — they're compute-shader driven and don't fit the Geometry/Material mesh pipeline. The integration replaces custom boilerplate (OrbitCamera, BlitPipeline) with engine equivalents, adds optional bloom/color grading post-processing, and rebuilds the WASM module to use kansei-core's Renderer for device management.

**Tech Stack:** Rust, wgpu 24, wasm-bindgen, web-sys

---

## File Structure

### Files to create:
- `rust/kansei-native/examples/fluid_engine.rs` — New fluid example using engine abstractions

### Files to modify:
- `rust/kansei-core/src/controls/camera_controls.rs` — Ensure it has scroll zoom support
- `rust/kansei-wasm/src/lib.rs` — Rebuild to use kansei-core Renderer
- `rust/kansei-wasm/Cargo.toml` — May need dependency updates

---

### Task 1: Rewrite fluid example using engine abstractions

**Files:**
- Create: `rust/kansei-native/examples/fluid_engine.rs`
- Read: `rust/kansei-native/examples/fluid_3d.rs` (reference for fluid setup)

The new example replaces:
- Custom `OrbitCamera` → uses engine's Camera + manual orbit math (CameraControls doesn't have full orbit camera yet, so keep orbit math but use engine Camera)
- Custom `BlitPipeline` → engine already handles this via Renderer surface management
- Custom offscreen textures → keep as-is (fluid rendering needs custom offscreen targets)
- Add optional post-processing (bloom + color grading on the fluid output)

The fluid rendering flow:
1. Fluid sim computes particle positions (compute shader)
2. Density field splatting (compute shader)
3. Either: particle billboard rendering OR surface raymarching (compute shader)
4. Blit result to canvas

The engine's Renderer manages device/queue/surface. The fluid's custom rendering pipelines (ParticleRenderer, FluidSurfaceRenderer) render to offscreen textures, then the result is blitted to the canvas.

- [ ] **Step 1: Create fluid_engine.rs**

This is a rewrite of fluid_3d.rs with these changes:
- Use `kansei_core::cameras::Camera` with `look_at()` 
- Use `kansei_core::renderers::Renderer` for device/queue/surface
- Keep the fluid sim, density field, surface renderer, particle renderer, egui as-is
- Keep the custom blit (the engine's blit is inside PostProcessingVolume, not standalone)
- Keep offscreen texture management
- Remove any redundant boilerplate

The implementer should:
1. Read the FULL current `fluid_3d.rs` (~477 lines)
2. Copy it to `fluid_engine.rs`
3. Replace the orbit camera eye/view computation to use `camera.set_position()` + `camera.look_at()`
4. Ensure the Renderer is used for device(), queue(), surface() access
5. The render flow stays the same (custom rendering, not through Renderer.render())
6. Add FPS counter logging

The key difference: this example proves the fluid sim works within the engine's infrastructure, even though the rendering itself is custom. This is the baseline for the TS vs Rust performance comparison.

- [ ] **Step 2: Build and test**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo build --example fluid_engine 2>&1 | tail -5
cargo run --example fluid_engine
```

Expected: same fluid sim as fluid_3d but using engine Camera.

- [ ] **Step 3: Commit**

```bash
git add kansei-native/examples/fluid_engine.rs
git commit -m "feat: add fluid_engine example using kansei engine infrastructure"
```

---

### Task 2: Rebuild WASM module using engine Renderer

**Files:**
- Modify: `rust/kansei-wasm/src/lib.rs`
- Modify: `rust/kansei-wasm/Cargo.toml` (if needed)

The current WASM module creates its own wgpu device/queue/surface directly. Replace this with using kansei-core's Renderer for device management while keeping the fluid-specific rendering.

- [ ] **Step 1: Update kansei-wasm to use Renderer for device management**

Read the current `rust/kansei-wasm/src/lib.rs` (400 lines). The key changes:

a) Replace the manual device/queue/surface creation in `start()` with Renderer:
```rust
let mut renderer = Renderer::new(RendererConfig {
    width, height, sample_count: 1,
    clear_color: Vec4::new(0.02, 0.02, 0.04, 1.0),
    ..Default::default()
});
renderer.initialize(surface, &adapter).await;
let device = renderer.device().clone(); // Note: may need Arc
let queue = renderer.queue().clone();
```

Actually, the WASM module stores device/queue in its State struct and uses them directly for custom rendering. The Renderer wraps device/queue but the fluid rendering needs raw access. The simplest approach: use Renderer for initialization, then extract device/queue references.

b) The State struct currently holds `device`, `queue`, `surface`, `format`. Replace with:
```rust
struct State {
    renderer: Renderer,
    sim: FluidSimulation,
    // ... rest stays the same
}
```

And access device/queue via `self.renderer.device()` and `self.renderer.queue()`.

c) The render_frame() method currently gets the surface texture directly. With Renderer, use `self.renderer.surface()` to get the surface.

The tricky part: the Renderer's `render()` method does its own pass setup, but the fluid example needs custom render passes. So we use Renderer only for device/surface management, not for rendering.

- [ ] **Step 2: Rebuild WASM**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust/kansei-wasm && wasm-pack build --target web 2>&1 | tail -10
```

- [ ] **Step 3: Test WASM in browser**

Open `rust/kansei-wasm/www/index.html` in a browser (needs a local server).
Verify: fluid sim renders and responds to mouse interaction.

- [ ] **Step 4: Commit**

```bash
git add kansei-wasm/src/lib.rs
git commit -m "feat: rebuild WASM module using kansei-core Renderer for device management"
```

---

### Task 3: Add performance timing to both native and WASM

**Files:**
- Modify: `rust/kansei-native/examples/fluid_engine.rs` — Add frame time measurement
- Modify: `rust/kansei-wasm/src/lib.rs` — Add frame time measurement via web_sys::Performance

For the performance comparison, both versions need to report frame times.

- [ ] **Step 1: Add FPS/frame time to native example**

In fluid_engine.rs, add to the RedrawRequested handler:
```rust
let frame_start = Instant::now();
// ... all rendering ...
let frame_time = frame_start.elapsed();
if frame_count % 60 == 0 {
    log::info!("Frame time: {:.2}ms ({:.0} FPS)", 
        frame_time.as_secs_f64() * 1000.0, 
        1.0 / frame_time.as_secs_f64());
}
frame_count += 1;
```

- [ ] **Step 2: Add frame time to WASM**

In the WASM render_frame(), use `web_sys::window().unwrap().performance()` for high-res timing:
```rust
let perf = web_sys::window().unwrap().performance().unwrap();
let start = perf.now();
// ... rendering ...
let elapsed = perf.now() - start;
// Log every 60 frames
```

Export a `get_frame_time() -> f64` function via wasm_bindgen for the JS side.

- [ ] **Step 3: Commit**

```bash
git add kansei-native/examples/fluid_engine.rs kansei-wasm/src/lib.rs
git commit -m "feat: add performance timing to native and WASM fluid examples"
```

---

### Task 4: Rebuild WASM and verify end-to-end

**Files:**
- Modify: None (just build and verify)

- [ ] **Step 1: Build everything**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust
# Native
cargo build --example fluid_engine --release 2>&1 | tail -5
# WASM
cd kansei-wasm && wasm-pack build --target web 2>&1 | tail -5
```

- [ ] **Step 2: Run native and note performance**

```bash
RUST_LOG=info cargo run --example fluid_engine --release
```

Note the frame time output.

- [ ] **Step 3: Run WASM and note performance**

Serve `kansei-wasm/www/` and open in browser. Note frame times in console.

- [ ] **Step 4: Commit final state**

```bash
git add -u
git commit -m "build: verify native + WASM fluid examples for performance comparison"
```

---

## Post-Plan Notes

### What this plan produces:
- `fluid_engine` native example using engine Camera/Renderer infrastructure
- WASM module rebuilt with engine Renderer for device management
- Performance timing in both native and WASM builds
- Ready for TS vs Rust performance comparison

### The performance comparison:
- **TS version**: The existing TypeScript fluid sim at `examples/index_postpro.html`
- **Rust native**: `cargo run --example fluid_engine --release`
- **Rust WASM**: The rebuilt `kansei-wasm/www/index.html`
- Compare: frame times at same particle count and sim parameters

### What's NOT integrated (by design):
- Particle rendering stays custom (reads from compute buffer, not Geometry)
- Surface raymarching stays custom (compute shader, not mesh Material)
- These are inherently compute-driven and would lose performance if forced through the mesh pipeline
