# Remaining Engine Features Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining TS engine features to Rust: render bundles, MSAA depth-copy, shader includes, GPU readback, onChange callbacks, system pattern.

**Architecture:** Each feature is independent. Implemented in order of dependency — render bundles first (biggest perf impact), then smaller features.

**Tech Stack:** Rust, wgpu 24

---

### Task 1: Render Bundle Caching

**Files:**
- Modify: `kansei-core/src/renderers/renderer.rs`

Add two cached render bundles (`render_bundle` and `gbuffer_bundle`) that pre-record draw commands. Rebuild when object count changes, material swaps, or bind groups resize.

- [ ] **Step 1: Add bundle fields to Renderer**

```rust
render_bundle: Option<wgpu::RenderBundle>,
last_bundle_object_count: usize,
gbuffer_bundle: Option<wgpu::RenderBundle>,
gbuffer_last_object_count: usize,
gbuffer_last_sample_count: u32,
```

- [ ] **Step 2: Add invalidate_bundle() method**

```rust
pub fn invalidate_bundle(&mut self) {
    self.render_bundle = None;
    self.gbuffer_bundle = None;
}
```

- [ ] **Step 3: Add build_render_bundle() method**

Private method that creates a `RenderBundleEncoder`, iterates ordered objects, sets pipelines/bind groups/vertex buffers with state dedup (skip if same as previous), bakes dynamic offsets. Returns `wgpu::RenderBundle`.

Key: uses `device.create_render_bundle_encoder()` with color formats + depth format + sample count. Camera bind group set once at group 1. Shadow at group 3. Per-object: material BG at group 0, mesh BG at group 2 with dynamic offsets.

Track `current_pipeline`, `current_material_bg`, `current_index_buffer`, `current_vertex_buffer` to skip redundant calls.

- [ ] **Step 4: Integrate into render()**

In render(), after matrix upload:
1. Check material_dirty on each renderable → invalidate if any set
2. If bundle is None or object count changed → rebuild
3. In render pass: `pass.execute_bundles(&[bundle])` instead of inline draw loop

- [ ] **Step 5: Same for GBuffer path**

Separate gbuffer_bundle with MRT formats and GBuffer sample count.

- [ ] **Step 6: Invalidate on bind group resize**

In `_ensure_matrix_buffers()` when buffers grow, call `invalidate_bundle()`.

- [ ] **Step 7: Verify and commit**

---

### Task 2: MSAA Depth-Copy Pass

**Files:**
- Modify: `kansei-core/src/renderers/renderer.rs`
- Create: `kansei-core/src/shaders/depth_copy.wgsl`

Fullscreen triangle that resolves MSAA depth to non-MSAA for compute shader sampling. Used when GBuffer has MSAA and post-processing needs depth.

- [ ] **Step 1: Create depth_copy.wgsl**

```wgsl
@group(0) @binding(0) var depth_msaa: texture_depth_multisampled_2d;

@vertex fn vs(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 3>(vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
    return vec4<f32>(pos[vi], 0.0, 1.0);
}

@fragment fn fs(@builtin(position) frag: vec4<f32>) -> @builtin(frag_depth) f32 {
    return textureLoad(depth_msaa, vec2<i32>(frag.xy), 0);
}
```

- [ ] **Step 2: Add depth copy pipeline to Renderer**

Create pipeline + BGL during initialize(). After MSAA render pass, run depth copy pass to resolve depth.

- [ ] **Step 3: Verify and commit**

---

### Task 3: Shader Include Preprocessor

**Files:**
- Create: `kansei-core/src/materials/shader_utils.rs`
- Modify: `kansei-core/src/materials/mod.rs`
- Modify: `kansei-core/src/materials/material.rs`

- [ ] **Step 1: Create shader_utils.rs with parse_includes()**

```rust
use std::collections::HashMap;

pub type ShaderChunks = HashMap<String, String>;

pub fn parse_includes(code: &str, chunks: &ShaderChunks) -> String {
    let mut result = String::new();
    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("#include") {
            let name = trimmed
                .trim_start_matches("#include")
                .trim()
                .trim_matches('<').trim_matches('>')
                .trim_matches('"');
            if let Some(chunk) = chunks.get(name) {
                result.push_str(&parse_includes(chunk, chunks)); // recursive
            }
        } else {
            result.push_str(line);
        }
        result.push('\n');
    }
    result
}
```

- [ ] **Step 2: Add to Material initialization**

In `ensure_shared()`, before creating the shader module, run `parse_includes()` on the shader code if a chunks map is provided.

Add `shader_chunks: Option<ShaderChunks>` to Material or pass through the Renderer.

- [ ] **Step 3: Verify and commit**

---

### Task 4: GPU Readback Buffer

**Files:**
- Modify: `kansei-core/src/renderers/renderer.rs`

- [ ] **Step 1: Add read_back_buffer() method**

```rust
pub async fn read_back_buffer<T: bytemuck::Pod>(
    &self,
    buffer: &wgpu::Buffer,
    size: u64,
) -> Vec<T> {
    let device = self.device.as_ref().unwrap();
    let queue = self.queue.as_ref().unwrap();
    
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Readback/Staging"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit(std::iter::once(encoder.finish()));
    
    let slice = staging.slice(..);
    let (tx, rx) = futures_channel::oneshot::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| { tx.send(result).ok(); });
    device.poll(wgpu::Maintain::Wait);
    rx.await.unwrap().unwrap();
    
    let data = slice.get_mapped_range();
    bytemuck::cast_slice(&data).to_vec()
}
```

Note: needs `futures-channel` crate or use a simpler polling approach for native.

- [ ] **Step 2: Verify and commit**

---

### Task 5: onChange Dirty Callbacks

**Files:**
- Modify: `kansei-core/src/math/vector.rs`
- Modify: `kansei-core/src/objects/object3d.rs`

The TS engine auto-flags dirty when position/rotation/scale vectors are mutated. In Rust, we can't have callbacks on field mutation easily. Instead, we make Object3D check if values changed since last update.

- [ ] **Step 1: Add version tracking to Object3D**

```rust
pub struct Object3D {
    // ... existing fields ...
    last_position: Vec3,
    last_rotation: Vec3,
    last_scale: Vec3,
}
```

In `update_model_matrix()`, check if values changed:
```rust
pub fn update_model_matrix(&mut self) {
    // Auto-detect dirty from field changes
    if self.position != self.last_position || self.rotation != self.last_rotation || self.scale != self.last_scale {
        self.dirty = true;
    }
    if !self.dirty { return; }
    // ... build matrix ...
    self.last_position = self.position;
    self.last_rotation = self.rotation;
    self.last_scale = self.scale;
    self.dirty = false;
}
```

Add `PartialEq` derive to Vec3 if not already there.

- [ ] **Step 2: Verify and commit**

---

### Task 6: System Pattern (Trait-Based Compute)

**Files:**
- Create: `kansei-core/src/systems/mod.rs`
- Create: `kansei-core/src/systems/compute_system.rs`
- Modify: `kansei-core/src/lib.rs`

- [ ] **Step 1: Define ComputeSystem trait**

```rust
pub trait ComputeSystem {
    fn initialize(&mut self, device: &wgpu::Device, queue: &wgpu::Queue);
    fn update(&mut self, dt: f32);
    fn is_initialized(&self) -> bool;
}
```

- [ ] **Step 2: Add compute_system() to Renderer**

```rust
pub fn compute_system(&self, system: &mut dyn ComputeSystem) {
    let device = self.device.as_ref().unwrap();
    let queue = self.queue.as_ref().unwrap();
    if !system.is_initialized() {
        system.initialize(device, queue);
    }
    system.update(0.0); // dt passed separately
}
```

This is a stub for the future System pattern. FluidSimulation can implement ComputeSystem later.

- [ ] **Step 3: Verify and commit**

---

## Post-Plan Notes

### What this produces:
- Render bundle caching (biggest perf improvement)
- MSAA depth-copy pass for post-processing
- Shader include preprocessor
- GPU readback buffer
- Auto-dirty detection on transform changes
- ComputeSystem trait for future System pattern

### Next: Path Tracer (separate plan)
