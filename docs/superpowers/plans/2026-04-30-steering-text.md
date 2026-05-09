# Steering Text Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust wasm example that renders psychology/mindfulness words as 3D boids — each word is a steering-behavior vehicle whose letters trail via verlet constraints, all drawn in one instanced MSDF draw call, with spatial-hash collision avoidance and mouse displacement.

**Architecture:** Hybrid JS/Rust. JS loads the `.arfont` font and `words.json`, extracts glyph metrics, passes them to Rust via wasm-bindgen. Rust creates an instanced PlaneGeometry (1 quad, P instances), builds compute pipelines for spatial hash + steering + verlet + integrate, and renders with an MSDF fragment shader. Tweakpane controls steering parameters and colors from JS.

**Tech Stack:** Rust + wgpu + wasm-bindgen, WGSL compute shaders, MSDF text rendering, Tweakpane v4.

**Spec:** `docs/superpowers/specs/2026-04-30-steering-text-design.md`

---

## File Structure

```
rust/kansei-wasm/examples/steering_text/
├── Cargo.toml                          # Package definition
├── src/
│   ├── lib.rs                          # wasm_bindgen entry, State, render_frame
│   ├── steering_sim.rs                 # SteeringSimulation struct, compute passes, buffers
│   ├── text_data.rs                    # GlyphData, parse words+glyphs → particle arrays
│   └── shaders/
│       ├── sim_params.wgsl             # SimParams struct shared across shaders
│       ├── grid_clear.wgsl             # Zero cell counts + scatter counters
│       ├── grid_assign.wgsl            # Hash vehicle positions → cells
│       ├── prefix_sum_local.wgsl       # Blelloch scan — local blocks
│       ├── prefix_sum_top.wgsl         # Blelloch scan — block sums
│       ├── prefix_sum_distribute.wgsl  # Blelloch scan — distribute
│       ├── scatter.wgsl                # Reorder vehicles by cell
│       ├── steering.wgsl               # Separation + wander + mouse + bounds
│       ├── verlet.wgsl                 # Letter-to-letter distance constraints
│       ├── integrate.wgsl              # Velocity damping + position update
│       └── msdf_text.wgsl              # Instanced vertex + MSDF fragment
└── www/
    ├── index.html                      # JS: font load, tweakpane, rAF
    ├── words.json                      # Psychology/mindfulness word list
    └── assets/
        └── fonts/
            └── (symlink or copy of L10-medium.arfont)
```

---

### Task 1: Project Scaffold

**Files:**
- Create: `rust/kansei-wasm/examples/steering_text/Cargo.toml`
- Create: `rust/kansei-wasm/examples/steering_text/src/lib.rs`
- Create: `rust/kansei-wasm/examples/steering_text/www/words.json`
- Create: `rust/kansei-wasm/examples/steering_text/www/index.html`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "kansei-wasm-steering-text"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
kansei-core = { path = "../../../kansei-core" }
wgpu = "24"
glam = { version = "0.29", features = ["bytemuck"] }
bytemuck = { version = "1", features = ["derive"] }
wasm-bindgen = "0.2"
wasm-bindgen-futures = "0.4"
web-sys = { version = "0.3", features = [
    "HtmlCanvasElement", "Window", "Document", "Element",
    "MouseEvent", "WheelEvent", "EventTarget",
    "TouchEvent", "Touch", "TouchList",
    "Performance", "Response",
] }
js-sys = "0.3"
log = "0.4"
console_log = "1"
console_error_panic_hook = "0.1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

- [ ] **Step 2: Create minimal lib.rs**

Skeleton: `init()`, `start(canvas_id)`, and an empty `State` struct with a `render_frame()` method. No simulation yet — just clear the canvas to the background color.

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use std::cell::RefCell;
use std::rc::Rc;

use kansei_core::cameras::Camera;
use kansei_core::controls::{CameraControls, MouseVectors};
use kansei_core::math::Vec3;
use kansei_core::renderers::{Renderer, RendererConfig};

mod steering_sim;
mod text_data;

#[wasm_bindgen(start)]
pub fn init() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).ok();
    log::info!("Steering Text WASM initialized");
}

#[wasm_bindgen]
pub async fn start(canvas_id: &str) -> Result<(), JsValue> {
    #[cfg(target_arch = "wasm32")]
    {
        let window = web_sys::window().unwrap();
        let document = window.document().unwrap();
        let canvas = document.get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?.dyn_into::<web_sys::HtmlCanvasElement>()?;

        let width = canvas.client_width() as u32;
        let height = canvas.client_height() as u32;
        canvas.set_width(width);
        canvas.set_height(height);

        let mut renderer = Renderer::new(RendererConfig {
            width, height, sample_count: 1,
            ..Default::default()
        });
        renderer.initialize_with_canvas(canvas.clone()).await;

        let mut camera = Camera::new(45.0, 0.1, 1000.0, width as f32 / height as f32);
        camera.set_position(0.0, 0.0, 80.0);
        camera.look_at(&Vec3::new(0.0, 0.0, 0.0));
        camera.update_projection_matrix();

        let controls = CameraControls::from_canvas(&canvas, Vec3::new(0.0, 0.0, 0.0), 80.0);
        let mouse = MouseVectors::from_canvas(&canvas);

        let state = Rc::new(RefCell::new(State {
            renderer, camera, controls, mouse,
            width, height,
            last_perf_time: window.performance().map(|p| p.now()).unwrap_or(0.0),
            bg_color: [0.02, 0.02, 0.06, 1.0],
        }));

        // Animation loop
        let f: Rc<RefCell<Option<Closure<dyn FnMut()>>>> = Rc::new(RefCell::new(None));
        let g = f.clone(); let s = state.clone();
        *g.borrow_mut() = Some(Closure::new(move || {
            s.borrow_mut().render_frame();
            request_animation_frame(f.borrow().as_ref().unwrap());
        }));
        request_animation_frame(g.borrow().as_ref().unwrap());

        log::info!("Steering Text — ready");
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    { let _ = canvas_id; Err(JsValue::from_str("wasm32 only")) }
}

fn request_animation_frame(f: &Closure<dyn FnMut()>) {
    web_sys::window().unwrap().request_animation_frame(f.as_ref().unchecked_ref()).unwrap();
}

struct State {
    renderer: Renderer,
    camera: Camera,
    controls: CameraControls,
    mouse: MouseVectors,
    width: u32,
    height: u32,
    last_perf_time: f64,
    bg_color: [f32; 4],
}

impl State {
    fn render_frame(&mut self) {
        let perf = web_sys::window().unwrap().performance().unwrap();
        let now = perf.now();
        let frame_ms = (now - self.last_perf_time).max(0.0);
        self.last_perf_time = now;
        let dt = (frame_ms * 0.001).min(0.1) as f32;

        self.controls.update(&mut self.camera, 0.0);
        self.mouse.update(dt);

        // For now just present a cleared canvas.
        let output = self.renderer.surface().unwrap().get_current_texture().unwrap();
        let view = output.texture.create_view(&Default::default());
        let mut encoder = self.renderer.device().create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("SteeringText/Frame") });
        {
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: self.bg_color[0] as f64,
                            g: self.bg_color[1] as f64,
                            b: self.bg_color[2] as f64,
                            a: self.bg_color[3] as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            });
        }
        self.renderer.submit(std::iter::once(encoder.finish()));
        output.present();
    }
}
```

- [ ] **Step 3: Create empty module stubs**

`src/steering_sim.rs`:
```rust
// SteeringSimulation — compute-based 3D boids with spatial hash.
// Populated in Task 4+.
```

`src/text_data.rs`:
```rust
// GlyphData parsing — convert JS glyph JSON into GPU buffer data.
// Populated in Task 3.
```

- [ ] **Step 4: Create words.json**

```json
[
  "awareness", "presence", "breathe", "mindful", "clarity",
  "serenity", "balance", "focus", "observe", "acceptance",
  "compassion", "gratitude", "resilience", "stillness", "intention",
  "equanimity", "empathy", "insight", "harmony", "patience",
  "kindness", "surrender", "grounding", "wholeness", "consciousness",
  "meditation", "reflection", "vulnerability", "courage", "authenticity",
  "flow", "peace", "trust", "release", "transform",
  "nurture", "connect", "listen", "heal", "gentle",
  "calm", "open", "anchor", "center", "wisdom",
  "wonder", "rest", "renew", "bloom", "grow"
]
```

- [ ] **Step 5: Create minimal index.html**

The HTML loads the wasm, calls `start('kansei')`, and sets up Tweakpane. Font loading and `init_text()` come in Task 3.

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Kansei — Steering Text</title>
    <style>
        body { margin: 0; background: #000; overflow: hidden; }
        canvas { width: 100vw; height: 100vh; display: block; }
    </style>
</head>
<body>
    <canvas id="kansei"></canvas>
    <script type="module">
        import init, { start } from './pkg/kansei_wasm_steering_text.js';
        await init();
        await start('kansei');
    </script>
</body>
</html>
```

- [ ] **Step 6: Symlink font asset**

```bash
mkdir -p rust/kansei-wasm/examples/steering_text/www/assets/fonts
cp examples/assets/fonts/L10-medium.arfont \
   rust/kansei-wasm/examples/steering_text/www/assets/fonts/
```

- [ ] **Step 7: Build and verify**

```bash
cd rust/kansei-wasm/examples/steering_text
cargo check --target wasm32-unknown-unknown
wasm-pack build --target web --release --out-dir www/pkg
```

Expected: compiles, wasm output in `www/pkg/`. Serving `www/` shows a dark canvas.

- [ ] **Step 8: Commit**

```bash
git add rust/kansei-wasm/examples/steering_text/
git commit -m "feat: scaffold steering_text wasm example"
```

---

### Task 2: Text Data Parsing (text_data.rs)

**Files:**
- Modify: `rust/kansei-wasm/examples/steering_text/src/text_data.rs`

This module receives serialized glyph data from JS and builds the flat particle arrays (positions, image_bounds, plane_bounds, colors, word_meta, rest_lengths).

- [ ] **Step 1: Define glyph + particle data structures**

```rust
use serde::Deserialize;

#[derive(Deserialize, Clone)]
pub struct GlyphMetrics {
    pub codepoint: u32,
    pub advance: f32,
    pub image_bounds: [f32; 4],  // [left, top, right, bottom] in UV space
    pub plane_bounds: [f32; 4],  // [left, top, right, bottom] in pixel space
}

/// All the CPU-side data needed to build GPU buffers for the instanced mesh
/// and the simulation.
pub struct ParticleData {
    pub total_particles: u32,
    pub total_words: u32,
    pub positions: Vec<f32>,       // P * 4
    pub velocities: Vec<f32>,      // P * 4
    pub image_bounds: Vec<f32>,    // P * 4
    pub plane_bounds: Vec<f32>,    // P * 4
    pub colors: Vec<f32>,          // P * 4
    pub word_meta: Vec<u32>,       // P * 4 (word_id, letter_idx, word_len, particle_offset)
    pub rest_lengths: Vec<f32>,    // P
}
```

- [ ] **Step 2: Implement build_particle_data**

```rust
use std::collections::HashMap;

pub fn build_particle_data(
    words: &[String],
    glyphs: &[GlyphMetrics],
    font_size: f32,
    text_color: [f32; 4],
    bounds_size: f32,
) -> ParticleData {
    // Build codepoint → glyph lookup
    let glyph_map: HashMap<u32, &GlyphMetrics> = glyphs.iter()
        .map(|g| (g.codepoint, g))
        .collect();
    let space_advance = glyph_map.get(&32)
        .map(|g| g.advance)
        .unwrap_or(0.5) * font_size;

    // Count total letters (skip spaces within multi-word phrases like "letting go")
    let word_chars: Vec<Vec<char>> = words.iter()
        .map(|w| w.chars().filter(|c| *c != ' ').collect())
        .collect();
    let total: usize = word_chars.iter().map(|wc| wc.len()).sum();
    let n_words = word_chars.len();

    let mut data = ParticleData {
        total_particles: total as u32,
        total_words: n_words as u32,
        positions: Vec::with_capacity(total * 4),
        velocities: vec![0.0; total * 4],
        image_bounds: Vec::with_capacity(total * 4),
        plane_bounds: Vec::with_capacity(total * 4),
        colors: Vec::with_capacity(total * 4),
        word_meta: Vec::with_capacity(total * 4),
        rest_lengths: Vec::with_capacity(total),
    };

    let mut rng: u64 = 42;
    let mut particle_offset = 0u32;

    for (word_id, chars) in word_chars.iter().enumerate() {
        // Random 3D starting position within bounds
        let rand3 = |rng: &mut u64| -> f32 {
            *rng ^= *rng << 13; *rng ^= *rng >> 7; *rng ^= *rng << 17;
            (*rng as f32 / u64::MAX as f32) * 2.0 - 1.0
        };
        let start_x = rand3(&mut rng) * bounds_size * 0.8;
        let start_y = rand3(&mut rng) * bounds_size * 0.8;
        let start_z = rand3(&mut rng) * bounds_size * 0.8;

        for (letter_idx, &ch) in chars.iter().enumerate() {
            let cp = ch as u32;
            let fallback = GlyphMetrics {
                codepoint: cp, advance: 0.5,
                image_bounds: [0.0; 4], plane_bounds: [0.0; 4],
            };
            let glyph = glyph_map.get(&cp).unwrap_or(&&fallback);

            // Position: all letters start at the word's random origin
            // (verlet will spread them on the first few frames)
            data.positions.extend_from_slice(&[start_x, start_y, start_z, 1.0]);

            // Image bounds (MSDF UV rect)
            data.image_bounds.extend_from_slice(&glyph.image_bounds);

            // Plane bounds (glyph pixel rect, scaled by font_size)
            data.plane_bounds.extend_from_slice(&[
                glyph.plane_bounds[0] * font_size,
                glyph.plane_bounds[1] * font_size,
                glyph.plane_bounds[2] * font_size,
                glyph.plane_bounds[3] * font_size,
            ]);

            // Color
            data.colors.extend_from_slice(&text_color);

            // Word metadata
            data.word_meta.extend_from_slice(&[
                word_id as u32,
                letter_idx as u32,
                chars.len() as u32,
                particle_offset,
            ]);

            // Rest length (distance to previous letter in the word chain)
            let rest = if letter_idx == 0 {
                0.0 // vehicles have no constraint to a predecessor
            } else {
                glyph.advance * font_size
            };
            data.rest_lengths.push(rest);

            particle_offset += 1;
        }
    }

    data
}
```

- [ ] **Step 3: Commit**

```bash
git add src/text_data.rs
git commit -m "feat(steering_text): text_data parser — words+glyphs → particle arrays"
```

---

### Task 3: JS Font Loading + init_text wasm-bindgen Bridge

**Files:**
- Modify: `rust/kansei-wasm/examples/steering_text/src/lib.rs`
- Modify: `rust/kansei-wasm/examples/steering_text/www/index.html`

- [ ] **Step 1: Add `init_text` wasm-bindgen function to lib.rs**

This receives serialized glyph metrics JSON, the MSDF atlas as raw RGBA bytes, atlas dimensions, and the words JSON. It builds particle arrays, creates GPU buffers, and sets up the instanced renderable.

Add this function after `start()`:

```rust
#[wasm_bindgen]
pub fn init_text(
    words_json: &str,
    glyphs_json: &str,
    msdf_rgba: &[u8],
    atlas_width: u32,
    atlas_height: u32,
) {
    // Parse
    let words: Vec<String> = serde_json::from_str(words_json)
        .expect("Failed to parse words JSON");
    let glyphs: Vec<text_data::GlyphMetrics> = serde_json::from_str(glyphs_json)
        .expect("Failed to parse glyphs JSON");

    let data = text_data::build_particle_data(
        &words, &glyphs, 1.0, [1.0, 1.0, 1.0, 1.0], 50.0,
    );

    log::info!("Steering Text — {} words, {} particles",
        data.total_words, data.total_particles);

    // Store in global state (will be wired to rendering in Task 5)
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() {
            let mut st = rc.borrow_mut();
            // TODO: create GPU buffers + renderable in Task 5
            st.particle_count = data.total_particles;
            st.word_count = data.total_words;
        }
    });
}
```

Add the global-state thread-local (same pattern as fluid example):
```rust
thread_local! {
    static GLOBAL_STATE: RefCell<Option<Rc<RefCell<State>>>> = RefCell::new(None);
}
```

Set it at the end of `start()`:
```rust
GLOBAL_STATE.with(|gs| { *gs.borrow_mut() = Some(state.clone()); });
```

Add `particle_count: u32` and `word_count: u32` fields to `State` (default 0).

- [ ] **Step 2: Update index.html with font loading + init_text call**

The JS side loads the `.arfont` via the existing wasm FontLoader (from the TS side — we import it), extracts glyph metrics, and passes them to Rust.

Since we're in a standalone wasm example that doesn't have access to the TS FontLoader, we'll load the arfont via a simpler approach: fetch the binary, use the artery-font wasm module directly, or — simplest — **pre-extract glyph metrics into a static JSON** shipped alongside `words.json`.

For the initial version, create a `glyphs.json` that contains pre-extracted metrics for all ASCII glyphs needed. The JS just fetches both JSONs and passes them to Rust.

```html
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Kansei — Steering Text</title>
    <style>
        body { margin: 0; background: #000; overflow: hidden; }
        canvas { width: 100vw; height: 100vh; display: block; }
    </style>
</head>
<body>
    <canvas id="kansei"></canvas>
    <script type="module">
        import init, { start, init_text } from './pkg/kansei_wasm_steering_text.js';
        import { Pane } from 'https://cdn.jsdelivr.net/npm/tweakpane@4/dist/tweakpane.min.js';

        await init();
        await start('kansei');

        // Load words + pre-extracted glyph metrics + MSDF atlas
        const [wordsResp, glyphsResp, atlasResp] = await Promise.all([
            fetch('words.json'),
            fetch('glyphs.json'),
            fetch('assets/fonts/L10-medium-atlas.png'),
        ]);
        const wordsJson = await wordsResp.text();
        const glyphsJson = await glyphsResp.text();

        // Decode MSDF atlas PNG → raw RGBA
        const atlasBlob = await atlasResp.blob();
        const atlasBitmap = await createImageBitmap(atlasBlob);
        const atlasCanvas = new OffscreenCanvas(atlasBitmap.width, atlasBitmap.height);
        const ctx = atlasCanvas.getContext('2d');
        ctx.drawImage(atlasBitmap, 0, 0);
        const atlasData = ctx.getImageData(0, 0, atlasBitmap.width, atlasBitmap.height);

        init_text(wordsJson, glyphsJson, atlasData.data, atlasBitmap.width, atlasBitmap.height);

        // Tweakpane (populated in Task 8)
        const pane = new Pane({ title: 'Steering Text', expanded: false });
    </script>
</body>
</html>
```

- [ ] **Step 3: Generate glyphs.json**

Write a small Node.js script (or do it manually) that extracts glyph metrics from the `.arfont` for all lowercase ASCII + space. Alternatively, use the existing TS FontLoader in a one-off script.

For now, create a placeholder `glyphs.json` with metrics for the characters used in `words.json`. The exact UV coordinates come from the font atlas. This step requires running the TS FontLoader once to dump the metrics.

Script approach (run from the kansei root with the TS dev server):
```js
// extract-glyphs.js — run once with: node --experimental-modules extract-glyphs.js
// Uses the FontLoader to read the .arfont and dump glyph metrics.
```

The output format:
```json
[
  {
    "codepoint": 97,
    "advance": 0.532,
    "image_bounds": [0.015, 0.421, 0.098, 0.515],
    "plane_bounds": [-0.024, -0.027, 0.556, 0.603]
  },
  ...
]
```

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs www/index.html www/glyphs.json
git commit -m "feat(steering_text): JS font loading bridge + init_text wasm function"
```

---

### Task 4: MSDF Shader + Instanced Rendering

**Files:**
- Create: `rust/kansei-wasm/examples/steering_text/src/shaders/msdf_text.wgsl`
- Modify: `rust/kansei-wasm/examples/steering_text/src/lib.rs`

- [ ] **Step 1: Write the MSDF text shader**

Port from the TS `TextRenderShader.ts`. The vertex shader maps a unit quad to glyph plane bounds, offsets by particle position, and projects. The fragment shader samples the MSDF atlas and thresholds at 0.5 with `smoothstep`.

```wgsl
// msdf_text.wgsl — Instanced MSDF text rendering.
// Base geometry: PlaneGeometry (unit quad, UVs 0..1).
// Per-instance: position (vec4), imageBounds (vec4), planeBounds (vec4), color (vec4).

// Group 0: Material (MSDF atlas + sampler)
@group(0) @binding(0) var atlas: texture_2d<f32>;
@group(0) @binding(1) var atlasSampler: sampler;

// Group 1: Camera
@group(1) @binding(0) var<uniform> viewMatrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projectionMatrix: mat4x4<f32>;

// Group 2: Mesh (world + normal matrices — required by engine pipeline layout)
@group(2) @binding(0) var<uniform> normalMatrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> worldMatrix: mat4x4<f32>;

struct VOut {
    @builtin(position) clipPos: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
};

fn remap(x: f32, lo: f32, hi: f32, oLo: f32, oHi: f32) -> f32 {
    return oLo + (x - lo) * (oHi - oLo) / (hi - lo);
}

@vertex
fn vertex_main(
    @location(0) position: vec4<f32>,     // quad vertex (PlaneGeometry: -0.5..0.5)
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) particlePos: vec4<f32>,  // instance: world position
    @location(4) imageBounds: vec4<f32>,  // instance: MSDF UV rect [l,t,r,b]
    @location(5) planeBounds: vec4<f32>,  // instance: glyph pixel rect [l,t,r,b]
    @location(6) color: vec4<f32>,        // instance: RGBA
) -> VOut {
    // Map quad vertex (0..1 via step) to glyph pixel bounds
    let gx = mix(planeBounds.x, planeBounds.z, step(0.0, position.x));
    let gy = mix(planeBounds.y, planeBounds.w, step(0.0, position.y));
    let worldPos = vec4<f32>(gx + particlePos.x, gy + particlePos.y, particlePos.z, 1.0);

    // Map UV to MSDF atlas sub-region
    let atlasUV = vec2<f32>(
        remap(uv.x, 0.0, 1.0, imageBounds.x, imageBounds.z),
        remap(uv.y, 0.0, 1.0, imageBounds.y, imageBounds.w),
    );

    var out: VOut;
    out.clipPos = projectionMatrix * viewMatrix * worldMatrix * worldPos;
    out.uv = atlasUV;
    out.color = color;
    return out;
}

fn median(r: f32, g: f32, b: f32) -> f32 {
    return max(min(r, g), min(max(r, g), b));
}

@fragment
fn fragment_main(v: VOut) -> @location(0) vec4<f32> {
    let sample = textureSample(atlas, atlasSampler, v.uv);
    let sd = median(sample.r, sample.g, sample.b);
    let d = fwidth(sd);
    let alpha = smoothstep(0.5 - d, 0.5 + d, sd);
    if (alpha < 0.01) { discard; }
    return vec4<f32>(v.color.rgb, v.color.a * alpha);
}
```

- [ ] **Step 2: Wire up instanced rendering in lib.rs**

In `init_text()`, create:
1. MSDF atlas as a wgpu Texture (from raw RGBA bytes)
2. Sampler
3. 4 ComputeBuffers for positions, imageBounds, planeBounds, colors (STORAGE | VERTEX)
4. InstancedGeometry wrapping PlaneGeometry(1,1) with the 4 extra buffers
5. Material with the MSDF shader + atlas texture + sampler bindings
6. Renderable added to a Scene
7. Store in State for rendering

This involves significant Rust code to create the texture, buffers, geometry, material, and scene. The exact implementation follows the patterns from the fluid example and the explored reference code.

Key buffer creation pattern:
```rust
let pos_buf = ComputeBuffer::from_slice(
    "SteeringText/Positions",
    BufferType::Storage,
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
    &data.positions,
).with_vertex_vec4(3);
```

Key material creation:
```rust
let mut mat = Material::new(
    "SteeringText/MSDF",
    MSDF_TEXT_WGSL,
    vec![
        Binding::texture_2d(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
        Binding::sampler(1, ShaderStages::FRAGMENT),
    ],
    MaterialOptions { transparent: true, depth_write: Some(false), ..Default::default() },
);
```

- [ ] **Step 3: Render the instanced mesh in render_frame**

Replace the bare clear-pass with the engine's `render_with_scene()` or manual scene rendering via the Renderer's standard path. The instanced mesh + material should produce visible MSDF text on screen (all clustered at random positions — no simulation yet).

- [ ] **Step 4: Build and verify**

```bash
cargo check --target wasm32-unknown-unknown
wasm-pack build --target web --release --out-dir www/pkg
```

Expected: text glyphs visible on screen at random 3D positions. No movement.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/msdf_text.wgsl src/lib.rs
git commit -m "feat(steering_text): MSDF instanced text rendering — single draw call"
```

---

### Task 5: Steering Simulation — Buffers + Compute Infrastructure

**Files:**
- Modify: `rust/kansei-wasm/examples/steering_text/src/steering_sim.rs`
- Create: `rust/kansei-wasm/examples/steering_text/src/shaders/sim_params.wgsl`

- [ ] **Step 1: Define SimParams struct and WGSL**

`sim_params.wgsl` — shared struct included in all compute shaders:
```wgsl
struct SimParams {
    dt: f32,
    particleCount: u32,
    vehicleCount: u32,
    separationStrength: f32,

    separationRadius: f32,
    wanderStrength: f32,
    wanderSpeed: f32,
    maxSpeed: f32,

    maxForce: f32,
    damping: f32,
    boundsSize: f32,
    time: f32,

    mouseStrength: f32,
    mousePosX: f32,
    mousePosY: f32,
    mouseDirX: f32,

    mouseDirY: f32,
    gridDimsX: u32,
    gridDimsY: u32,
    gridDimsZ: u32,

    cellSize: f32,
    gridOriginX: f32,
    gridOriginY: f32,
    gridOriginZ: f32,

    totalCells: u32,
    verletIterations: u32,
    _pad0: u32,
    _pad1: u32,
};

fn getCellCoord(pos: vec3<f32>) -> vec3<i32> {
    let origin = vec3<f32>(params.gridOriginX, params.gridOriginY, params.gridOriginZ);
    return vec3<i32>(floor((pos - origin) / params.cellSize));
}

fn cellHash(coord: vec3<i32>) -> u32 {
    let dims = vec3<i32>(i32(params.gridDimsX), i32(params.gridDimsY), i32(params.gridDimsZ));
    let c = clamp(coord, vec3<i32>(0), dims - vec3<i32>(1));
    return u32(c.z) * params.gridDimsX * params.gridDimsY
         + u32(c.y) * params.gridDimsX
         + u32(c.x);
}
```

- [ ] **Step 2: Define SteeringSimulation struct**

```rust
pub struct SteeringSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    // Simulation buffers
    pub positions_buffer: wgpu::Buffer,
    pub velocities_buffer: wgpu::Buffer,
    pub word_meta_buffer: wgpu::Buffer,
    pub rest_lengths_buffer: wgpu::Buffer,
    // Spatial hash buffers
    cell_indices_buffer: wgpu::Buffer,
    cell_counts_buffer: wgpu::Buffer,
    cell_offsets_buffer: wgpu::Buffer,
    scatter_counters_buffer: wgpu::Buffer,
    sorted_indices_buffer: wgpu::Buffer,
    block_sums_buffer: wgpu::Buffer,
    // Params
    params_buffer: wgpu::Buffer,
    // Compute pipelines + bind groups
    grid_clear_pipeline: wgpu::ComputePipeline,
    grid_clear_bg: wgpu::BindGroup,
    grid_assign_pipeline: wgpu::ComputePipeline,
    grid_assign_bg: wgpu::BindGroup,
    prefix_sum_local_pipeline: wgpu::ComputePipeline,
    prefix_sum_local_bg: wgpu::BindGroup,
    prefix_sum_top_pipeline: wgpu::ComputePipeline,
    prefix_sum_top_bg: wgpu::BindGroup,
    prefix_sum_distribute_pipeline: wgpu::ComputePipeline,
    prefix_sum_distribute_bg: wgpu::BindGroup,
    scatter_pipeline: wgpu::ComputePipeline,
    scatter_bg: wgpu::BindGroup,
    steering_pipeline: wgpu::ComputePipeline,
    steering_bg: wgpu::BindGroup,
    verlet_pipeline: wgpu::ComputePipeline,
    verlet_bg: wgpu::BindGroup,
    integrate_pipeline: wgpu::ComputePipeline,
    integrate_bg: wgpu::BindGroup,
    // Counts
    particle_count: u32,
    vehicle_count: u32,
    total_cells: u32,
    grid_dims: [u32; 3],
}
```

- [ ] **Step 3: Implement `new()` constructor**

Creates all buffers and pipelines. Each compute pipeline is created with an explicit `BindGroupLayout` and `PipelineLayout`. The shaders are loaded via `include_str!()`.

Key pattern (repeat for each pipeline):
```rust
let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
    label: Some("SteeringText/GridClear"),
    source: wgpu::ShaderSource::Wgsl(include_str!("shaders/grid_clear.wgsl").into()),
});
let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { ... });
let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    bind_group_layouts: &[&bgl], ..
});
let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    layout: Some(&layout), module: &shader, entry_point: Some("main"), ..
});
let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
    layout: &bgl, entries: &[...], ..
});
```

- [ ] **Step 4: Implement `update()` method**

Records all compute passes into a single command encoder:
```rust
pub fn update(&self, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, params: &SimParamsData) {
    // Upload params
    queue.write_buffer(&self.params_buffer, 0, bytemuck::bytes_of(params));

    let vehicle_wg = ((self.vehicle_count + 63) / 64) as u32;
    let particle_wg = ((self.particle_count + 63) / 64) as u32;
    let grid_wg = ((self.total_cells + 255) / 256) as u32;
    let prefix_wg = ((self.total_cells + 511) / 512).max(1) as u32;

    // 1. Grid clear
    self.dispatch(encoder, &self.grid_clear_pipeline, &self.grid_clear_bg, grid_wg);
    // 2. Grid assign (vehicles only)
    self.dispatch(encoder, &self.grid_assign_pipeline, &self.grid_assign_bg, vehicle_wg);
    // 3. Prefix sum (3 passes)
    self.dispatch(encoder, &self.prefix_sum_local_pipeline, &self.prefix_sum_local_bg, prefix_wg);
    self.dispatch(encoder, &self.prefix_sum_top_pipeline, &self.prefix_sum_top_bg, 1);
    self.dispatch(encoder, &self.prefix_sum_distribute_pipeline, &self.prefix_sum_distribute_bg, prefix_wg);
    // 4. Scatter
    self.dispatch(encoder, &self.scatter_pipeline, &self.scatter_bg, vehicle_wg);
    // 5. Steering
    self.dispatch(encoder, &self.steering_pipeline, &self.steering_bg, vehicle_wg);
    // 6. Verlet (run multiple iterations)
    for _ in 0..params.verlet_iterations {
        self.dispatch(encoder, &self.verlet_pipeline, &self.verlet_bg, particle_wg);
    }
    // 7. Integrate
    self.dispatch(encoder, &self.integrate_pipeline, &self.integrate_bg, particle_wg);
}

fn dispatch(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, workgroups: u32) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(workgroups, 1, 1);
}
```

- [ ] **Step 5: Commit**

```bash
git add src/steering_sim.rs src/shaders/sim_params.wgsl
git commit -m "feat(steering_text): SteeringSimulation struct + compute infrastructure"
```

---

### Task 6: Spatial Hash Shaders

**Files:**
- Create: `src/shaders/grid_clear.wgsl`
- Create: `src/shaders/grid_assign.wgsl`
- Create: `src/shaders/prefix_sum_local.wgsl`
- Create: `src/shaders/prefix_sum_top.wgsl`
- Create: `src/shaders/prefix_sum_distribute.wgsl`
- Create: `src/shaders/scatter.wgsl`

Port the 6 spatial hash shaders from the fluid simulation (`rust/kansei-core/src/simulations/fluid/shaders/`). The only differences:

1. `grid_assign.wgsl` — hashes **vehicle positions only** (filter by `word_meta[idx].y == 0` i.e. letter_idx == 0). Uses the vehicles' positions from the main `positions` buffer, indexed via a `vehicle_indices` buffer (or filtered inline).

2. All shaders reference the `SimParams` struct from `sim_params.wgsl` (same field names).

3. The prefix-sum shaders are identical (they're generic — operate on `cellCounts` → `cellOffsets`).

4. `scatter.wgsl` writes `sortedIndices` for vehicles only.

Each shader is ~30-80 lines, following the exact patterns from the fluid sim. The grid-clear, prefix-sum, and scatter shaders can be copied nearly verbatim from the fluid example with `SimParams` field name adjustments.

- [ ] **Step 1: Write grid_clear.wgsl** — identical to fluid's, clears both `cellCounts` and `scatterCounters`.

- [ ] **Step 2: Write grid_assign.wgsl** — dispatched over vehicles only. Reads `positions[vehicle_global_index]`, hashes to cell, increments `cellCounts[cell]`.

- [ ] **Step 3: Write prefix_sum_local.wgsl, prefix_sum_top.wgsl, prefix_sum_distribute.wgsl** — copy from fluid sim, adjust `SimParams` struct import.

- [ ] **Step 4: Write scatter.wgsl** — dispatched over vehicles. Reads `cellIndices[idx]`, `cellOffsets[cell]`, writes `sortedIndices[offset + slot] = idx`.

- [ ] **Step 5: Commit**

```bash
git add src/shaders/grid_*.wgsl src/shaders/prefix_sum_*.wgsl src/shaders/scatter.wgsl
git commit -m "feat(steering_text): spatial hash compute shaders (grid + prefix-sum + scatter)"
```

---

### Task 7: Steering + Verlet + Integrate Shaders

**Files:**
- Create: `src/shaders/steering.wgsl`
- Create: `src/shaders/verlet.wgsl`
- Create: `src/shaders/integrate.wgsl`

- [ ] **Step 1: Write steering.wgsl**

Dispatched over `vehicleCount`. For each vehicle (letter_idx == 0):

```wgsl
// steering.wgsl
// Bindings: positions, velocities, word_meta, sorted vehicle indices,
// cell offsets, cell counts, params

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let vid = gid.x;
    if (vid >= params.vehicleCount) { return; }

    // Find this vehicle's global particle index
    let pidx = /* lookup from word_meta or vehicle_indices */;
    let pos = positions[pidx].xyz;
    var force = vec3<f32>(0.0);

    // ── Separation ──
    let myCell = getCellCoord(pos);
    for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
        let neighbor = myCell + vec3<i32>(dx, dy, dz);
        let hash = cellHash(neighbor);
        let start = cellOffsets[hash];
        let count = cellCounts[hash];
        for (var j = start; j < start + count; j++) {
            let other = sortedIndices[j];
            if (other == vid) { continue; }
            let otherPidx = /* other vehicle's particle index */;
            let diff = pos - positions[otherPidx].xyz;
            let dist = length(diff);
            if (dist > 0.001 && dist < params.separationRadius) {
                force += normalize(diff) * params.separationStrength / dist;
            }
        }
    }}}

    // ── Wander ──
    let seed = f32(vid) * 1.618;
    force.x += sin(params.time * params.wanderSpeed + seed) * params.wanderStrength;
    force.y += sin(params.time * params.wanderSpeed * 0.7 + seed * 2.3) * params.wanderStrength;
    force.z += cos(params.time * params.wanderSpeed * 1.1 + seed * 0.9) * params.wanderStrength;

    // ── Mouse force ──
    let mouseDir = vec2<f32>(params.mouseDirX, params.mouseDirY);
    let mouseWorld = vec3<f32>(mouseDir * params.mouseStrength, 0.0);
    force += mouseWorld;

    // ── Bounds ──
    let bs = params.boundsSize;
    let wall = 2.0;
    if (pos.x < -bs) { force.x += wall * (-bs - pos.x); }
    if (pos.x >  bs) { force.x += wall * ( bs - pos.x); }
    if (pos.y < -bs) { force.y += wall * (-bs - pos.y); }
    if (pos.y >  bs) { force.y += wall * ( bs - pos.y); }
    if (pos.z < -bs) { force.z += wall * (-bs - pos.z); }
    if (pos.z >  bs) { force.z += wall * ( bs - pos.z); }

    // ── Clamp ──
    let fl = length(force);
    if (fl > params.maxForce) { force = force * (params.maxForce / fl); }

    // Apply to velocity
    var vel = velocities[pidx].xyz + force * params.dt;
    let sl = length(vel);
    if (sl > params.maxSpeed) { vel = vel * (params.maxSpeed / sl); }

    velocities[pidx] = vec4<f32>(vel, 0.0);
}
```

- [ ] **Step 2: Write verlet.wgsl**

Dispatched over all P particles. Only acts on `letter_idx > 0`:

```wgsl
// verlet.wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let meta = word_meta[idx]; // (word_id, letter_idx, word_len, particle_offset)
    let letterIdx = meta.y;
    if (letterIdx == 0u) { return; } // vehicles don't constrain

    let prevIdx = idx - 1u; // previous letter in the word
    let restLen = rest_lengths[idx];
    if (restLen <= 0.0) { return; }

    let myPos = positions[idx].xyz;
    let prevPos = positions[prevIdx].xyz;
    let dir = myPos - prevPos;
    let dist = length(dir);

    if (dist > restLen && dist > 0.001) {
        let correction = dir * (1.0 - restLen / dist) * 0.5;
        positions[idx] = vec4<f32>(myPos - correction, 1.0);
    }
}
```

- [ ] **Step 3: Write integrate.wgsl**

```wgsl
// integrate.wgsl
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    var vel = velocities[idx].xyz;
    vel *= params.damping;
    var pos = positions[idx].xyz + vel * params.dt;

    // Soft bounds clamp
    let bs = params.boundsSize;
    pos = clamp(pos, vec3<f32>(-bs), vec3<f32>(bs));

    positions[idx] = vec4<f32>(pos, 1.0);
    velocities[idx] = vec4<f32>(vel, 0.0);
}
```

- [ ] **Step 4: Commit**

```bash
git add src/shaders/steering.wgsl src/shaders/verlet.wgsl src/shaders/integrate.wgsl
git commit -m "feat(steering_text): steering + verlet + integrate compute shaders"
```

---

### Task 8: Wire Simulation into Render Loop

**Files:**
- Modify: `src/lib.rs`
- Modify: `src/steering_sim.rs`

- [ ] **Step 1: Create SteeringSimulation in init_text**

After building particle data, create the SteeringSimulation with all buffers and pipelines. Store it in State.

- [ ] **Step 2: Call sim.update() in render_frame**

```rust
fn render_frame(&mut self) {
    // ... dt, controls, mouse ...

    if let Some(ref sim) = self.simulation {
        let params = SimParamsData {
            dt, particle_count: self.particle_count,
            vehicle_count: self.word_count,
            // ... fill from self.steering_params ...
            mouse_strength: self.mouse.strength.min(1.0),
            mouse_pos: [self.mouse.position.x, self.mouse.position.y],
            mouse_dir: [self.mouse.direction.x, self.mouse.direction.y],
            time: self.elapsed_time,
            // ... grid dims, cell size, etc ...
        };
        let mut encoder = self.renderer.device().create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        sim.update(self.renderer.queue(), &mut encoder, &params);
        self.renderer.submit(std::iter::once(encoder.finish()));
    }

    // Render scene (instanced mesh)
    self.renderer.render(&mut self.scene, &mut self.camera);
}
```

- [ ] **Step 3: Build and verify**

Words should now move: vehicles wander, letters trail, words avoid each other.

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs src/steering_sim.rs
git commit -m "feat(steering_text): wire simulation into render loop — words move"
```

---

### Task 9: Tweakpane Controls

**Files:**
- Modify: `www/index.html`
- Modify: `src/lib.rs` (add wasm-bindgen setter functions)

- [ ] **Step 1: Add wasm-bindgen setters to lib.rs**

```rust
fn with_state<F: FnOnce(&mut State)>(f: F) {
    GLOBAL_STATE.with(|gs| {
        if let Some(ref rc) = *gs.borrow() { f(&mut rc.borrow_mut()); }
    });
}

#[wasm_bindgen] pub fn set_separation_strength(v: f32) { with_state(|s| s.steering_params.separation_strength = v); }
#[wasm_bindgen] pub fn set_separation_radius(v: f32)   { with_state(|s| s.steering_params.separation_radius = v); }
#[wasm_bindgen] pub fn set_wander_strength(v: f32)     { with_state(|s| s.steering_params.wander_strength = v); }
#[wasm_bindgen] pub fn set_wander_speed(v: f32)        { with_state(|s| s.steering_params.wander_speed = v); }
#[wasm_bindgen] pub fn set_mouse_force(v: f32)         { with_state(|s| s.steering_params.mouse_force = v); }
#[wasm_bindgen] pub fn set_max_speed(v: f32)           { with_state(|s| s.steering_params.max_speed = v); }
#[wasm_bindgen] pub fn set_max_force(v: f32)           { with_state(|s| s.steering_params.max_force = v); }
#[wasm_bindgen] pub fn set_damping(v: f32)             { with_state(|s| s.steering_params.damping = v); }
#[wasm_bindgen] pub fn set_bounds_size(v: f32)         { with_state(|s| s.steering_params.bounds_size = v); }
#[wasm_bindgen] pub fn set_verlet_iterations(v: u32)   { with_state(|s| s.steering_params.verlet_iterations = v); }
#[wasm_bindgen] pub fn set_text_color(r: f32, g: f32, b: f32) { with_state(|s| s.text_color = [r, g, b, 1.0]); }
#[wasm_bindgen] pub fn set_bg_color(r: f32, g: f32, b: f32)   { with_state(|s| s.bg_color = [r, g, b, 1.0]); }
```

- [ ] **Step 2: Wire tweakpane in index.html**

```js
const pane = new Pane({ title: 'Steering Text', expanded: false });

const steering = pane.addFolder({ title: 'Steering', expanded: false });
const p = {
    separationStrength: 3.0, separationRadius: 5.0,
    wanderStrength: 1.0, wanderSpeed: 1.5,
    mouseForce: 1500, maxSpeed: 8.0, maxForce: 4.0,
    damping: 0.98, boundsSize: 50, verletIterations: 3,
};
steering.addBinding(p, 'separationStrength', { min: 0, max: 10, step: 0.1 })
    .on('change', () => set_separation_strength(p.separationStrength));
steering.addBinding(p, 'separationRadius', { min: 0, max: 20, step: 0.5 })
    .on('change', () => set_separation_radius(p.separationRadius));
steering.addBinding(p, 'wanderStrength', { min: 0, max: 5, step: 0.1 })
    .on('change', () => set_wander_strength(p.wanderStrength));
steering.addBinding(p, 'wanderSpeed', { min: 0, max: 5, step: 0.1 })
    .on('change', () => set_wander_speed(p.wanderSpeed));
steering.addBinding(p, 'mouseForce', { min: 0, max: 5000, step: 10 })
    .on('change', () => set_mouse_force(p.mouseForce));
steering.addBinding(p, 'maxSpeed', { min: 0, max: 20, step: 0.5 })
    .on('change', () => set_max_speed(p.maxSpeed));
steering.addBinding(p, 'maxForce', { min: 0, max: 10, step: 0.5 })
    .on('change', () => set_max_force(p.maxForce));
steering.addBinding(p, 'damping', { min: 0.9, max: 1.0, step: 0.001 })
    .on('change', () => set_damping(p.damping));
steering.addBinding(p, 'boundsSize', { min: 10, max: 200, step: 5 })
    .on('change', () => set_bounds_size(p.boundsSize));
steering.addBinding(p, 'verletIterations', { min: 1, max: 8, step: 1 })
    .on('change', () => set_verlet_iterations(p.verletIterations));

const visual = pane.addFolder({ title: 'Visual', expanded: false });
const v = {
    textColor: { r: 1, g: 1, b: 1 },
    bgColor: { r: 0.02, g: 0.02, b: 0.06 },
};
visual.addBinding(v, 'textColor', { color: { type: 'float' }, label: 'text' })
    .on('change', () => set_text_color(v.textColor.r, v.textColor.g, v.textColor.b));
visual.addBinding(v, 'bgColor', { color: { type: 'float' }, label: 'background' })
    .on('change', () => set_bg_color(v.bgColor.r, v.bgColor.g, v.bgColor.b));
```

- [ ] **Step 3: Push defaults on load**

After tweakpane setup:
```js
// Push all defaults to Rust
set_separation_strength(p.separationStrength);
set_separation_radius(p.separationRadius);
// ... etc for all params
```

- [ ] **Step 4: Commit**

```bash
git add src/lib.rs www/index.html
git commit -m "feat(steering_text): tweakpane controls for steering + visual params"
```

---

### Task 10: Polish — Auto-Rotate, Mobile Detection, Color Update

**Files:**
- Modify: `src/lib.rs`
- Modify: `www/index.html`

- [ ] **Step 1: Add gentle auto-rotate** in `render_frame()`:

```rust
self.auto_rotate_angle += dt * self.auto_rotate_speed;
self.controls.set_azimuth(self.auto_rotate_angle);
```

- [ ] **Step 2: Update text color buffer on change**

When `set_text_color` is called, write the new color to all P entries of the colors buffer:
```rust
let color_data: Vec<f32> = (0..self.particle_count)
    .flat_map(|_| self.text_color.iter().copied())
    .collect();
self.renderer.queue().write_buffer(&self.colors_buffer, 0, bytemuck::cast_slice(&color_data));
```

- [ ] **Step 3: Mobile camera radius**

```js
const isMobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent)
    || (window.matchMedia && window.matchMedia('(pointer: coarse)').matches);
// Pass larger initial radius to Rust via a setter if mobile
```

- [ ] **Step 4: Final build + test**

```bash
wasm-pack build --target web --release --out-dir www/pkg
```

Serve `www/` and verify: words wander in 3D, separate from each other, letters trail smoothly, mouse displaces vehicles, tweakpane controls all work, colors update live.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(steering_text): polish — auto-rotate, color updates, mobile detection"
```

---

## Self-Review Checklist

| Spec Requirement | Task |
|---|---|
| Hybrid JS/Rust font loading | Task 3 |
| JSON words (psychology/mindfulness) | Task 1 (words.json) |
| MSDF instanced rendering (single draw) | Task 4 |
| Spatial hash (grid-assign + prefix-sum + scatter) | Task 5 + 6 |
| Steering behaviors (separation + wander + mouse + bounds) | Task 7 (steering.wgsl) |
| Verlet constraints (letter chains) | Task 7 (verlet.wgsl) |
| MouseVectors displacement | Task 8 (sim wiring) |
| Tweakpane controls | Task 9 |
| 3D simulation | All shaders operate in 3D |
| Camera orbit + touch | Task 1 (CameraControls::from_canvas) |
| Text/BG color tweakpane | Task 9 + 10 |
| Auto-rotate | Task 10 |
