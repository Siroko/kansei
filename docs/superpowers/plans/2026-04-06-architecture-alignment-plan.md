# Rust Architecture Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Align the Rust kansei-core engine architecture to match the TypeScript engine 1:1 — same bind group slots, same matrix semantics, same object lifecycle, device private in Renderer.

**Architecture:** Sequential refactor in 10 tasks matching the spec's migration order. Each task produces a compiling codebase. Bind group slot swap lands first (biggest cross-cutting change), followed by Object3D/Scene restructuring, then lifecycle changes (Camera/Material/FluidSim/PostProcessing own their GPU resources, Renderer drives initialization).

**Tech Stack:** Rust, wgpu 24, glam 0.29, bytemuck 1

---

## File Structure

### Files to modify (core):
- `kansei-core/src/renderers/shared_layouts.rs` — swap bind group order
- `kansei-core/src/renderers/renderer.rs` — device private, remove camera/light buffers, update bind group set calls
- `kansei-core/src/objects/object3d.rs` — add model_matrix, fix normal_matrix
- `kansei-core/src/objects/renderable.rs` — minor adjustments for SceneNode
- `kansei-core/src/objects/scene.rs` — SceneNode tree, typed light collections
- `kansei-core/src/objects/mod.rs` — export SceneNode
- `kansei-core/src/materials/material.rs` — initialize() stores shared_layouts, remove shared param from get_pipeline
- `kansei-core/src/cameras/camera.rs` — owns bind group, initialize(device), remove look_at_target
- `kansei-core/src/postprocessing/volume.rs` — stores renderer refs, drives render
- `kansei-core/src/simulations/fluid/simulation.rs` — stores device/queue from initialize

### Shaders to update (group swap):
- `kansei-core/src/shaders/basic.wgsl`
- `kansei-core/src/shaders/basic_lit.wgsl`
- `kansei-core/src/shaders/basic_instanced.wgsl`
- `kansei-core/src/shaders/shadow_vs.wgsl`

### Examples to update:
- `kansei-native/examples/spinning_box.rs`
- `kansei-native/examples/instanced_cubes.rs`
- `kansei-native/examples/lit_scene.rs`
- `kansei-native/examples/shadow_scene.rs`
- `kansei-native/examples/postprocess_scene.rs`
- `kansei-native/examples/gltf_viewer.rs`
- `kansei-native/examples/fluid_engine.rs`
- `kansei-wasm/src/lib.rs`

---

### Task 1: Swap bind group slot order in SharedLayouts + shaders

**Files:**
- Modify: `kansei-core/src/renderers/shared_layouts.rs`
- Modify: `kansei-core/src/materials/material.rs` (pipeline layout order)
- Modify: `kansei-core/src/renderers/renderer.rs` (set_bind_group calls)
- Modify: All `.wgsl` shaders (swap @group(1) ↔ @group(2))

This is the most cross-cutting change. Everything compiles but shaders break if groups aren't swapped atomically.

- [ ] **Step 1: Update BindGroupSlot enum**

In `shared_layouts.rs`, change:
```rust
pub enum BindGroupSlot {
    Material = 0,
    Camera = 1,   // was Mesh
    Mesh = 2,     // was Camera
    Shadow = 3,
}
```

- [ ] **Step 2: Swap BGL creation order in SharedLayouts::new()**

In `shared_layouts.rs`, reorder so `camera_bgl` is created before `mesh_bgl`. The struct field order changes too:
```rust
pub struct SharedLayouts {
    pub camera_bgl: wgpu::BindGroupLayout,  // group 1
    pub mesh_bgl: wgpu::BindGroupLayout,    // group 2
    pub shadow_bgl: wgpu::BindGroupLayout,  // group 3
}
```

- [ ] **Step 3: Update Material pipeline layout order**

In `material.rs`, in `ensure_shared()`, change:
```rust
bind_group_layouts: &[&material_bgl, &shared.camera_bgl, &shared.mesh_bgl, &shared.shadow_bgl],
```

- [ ] **Step 4: Update Renderer set_bind_group calls**

In `renderer.rs`, in `render()` and `render_scene_to_gbuffer()`, change:
```rust
pass.set_bind_group(1, self.camera_bind_group.as_ref().unwrap(), &[]);  // was group 2
pass.set_bind_group(2, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);  // was group 1
pass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);
```

- [ ] **Step 5: Swap @group in all WGSL shaders**

In `basic.wgsl`, `basic_lit.wgsl`, `basic_instanced.wgsl`:
- Camera bindings (view_matrix, projection_matrix, lights): change from `@group(2)` to `@group(1)`
- Mesh bindings (normal_matrix, world_matrix): change from `@group(1)` to `@group(2)`
- Shadow bindings stay `@group(3)`

In `shadow_vs.wgsl`:
- `@group(1)` mesh bindings → `@group(2)`
- `@group(0)` light_view_proj stays (shadow pass uses its own layout)

- [ ] **Step 6: Update inline WGSL in WASM and examples**

The cornell box shader, particle shader in `kansei-wasm/src/lib.rs` and `fluid_engine.rs` — these use `@group(0)` only so no changes needed for the group swap. Verify.

- [ ] **Step 7: Verify compilation**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5
cargo build --example spinning_box 2>&1 | tail -3
```

- [ ] **Step 8: Commit**

```bash
git add kansei-core/src/renderers/ kansei-core/src/materials/ kansei-core/src/shaders/
git commit -m "refactor: swap bind group slots to match TS — camera=1, mesh=2"
```

---

### Task 2: Object3D two-matrix system + world-space normal matrix

**Files:**
- Modify: `kansei-core/src/objects/object3d.rs`

- [ ] **Step 1: Add model_matrix field, update methods**

Replace Object3D with:
```rust
pub struct Object3D {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub model_matrix: Mat4,     // local T * Rz * Ry * Rx * S
    pub world_matrix: Mat4,     // parent.world * model (or just model if root)
    pub normal_matrix: Mat4,    // transpose(inverse(world_matrix))
    children: Vec<usize>,
    parent: Option<usize>,
    dirty: bool,
}
```

Update `new()` to initialize `model_matrix: Mat4::identity()`.

- [ ] **Step 2: Update update_model_matrix()**

```rust
pub fn update_model_matrix(&mut self) {
    let t = glam::Mat4::from_translation(self.position.to_glam());
    let rx = glam::Mat4::from_rotation_x(self.rotation.x);
    let ry = glam::Mat4::from_rotation_y(self.rotation.y);
    let rz = glam::Mat4::from_rotation_z(self.rotation.z);
    let s = glam::Mat4::from_scale(self.scale.to_glam());
    self.model_matrix = Mat4::from(t * rz * ry * rx * s);
    self.dirty = false;
}
```

- [ ] **Step 3: Add update_world_matrix()**

```rust
pub fn update_world_matrix(&mut self, parent_world: Option<&Mat4>) {
    match parent_world {
        Some(pw) => self.world_matrix = pw.mul(&self.model_matrix),
        None => self.world_matrix = self.model_matrix,
    }
}
```

- [ ] **Step 4: Fix update_normal_matrix() — world-space, no view param**

```rust
pub fn update_normal_matrix(&mut self) {
    self.normal_matrix = Mat4::from(self.world_matrix.to_glam().inverse().transpose());
}
```

Remove the `view_matrix: &Mat4` parameter.

- [ ] **Step 5: Fix look_at() — extract Euler angles**

```rust
pub fn look_at(&mut self, target: &Vec3) {
    let forward = (*target - self.position).normalize();
    self.rotation.y = forward.x.atan2(forward.z);
    self.rotation.x = (-forward.y).asin();
    self.dirty = true;
}
```

- [ ] **Step 6: Fix all callers of update_normal_matrix**

Search for `update_normal_matrix(` across the codebase and remove the view_matrix argument. Callers in examples (spinning_box, fluid_engine, etc.) that pass `&camera.view_matrix` — remove that param.

- [ ] **Step 7: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
git add kansei-core/src/objects/object3d.rs
git commit -m "refactor: Object3D two-matrix system + world-space normal matrix"
```

---

### Task 3: Scene as SceneNode tree with typed light collections

**Files:**
- Modify: `kansei-core/src/objects/scene.rs`
- Modify: `kansei-core/src/objects/mod.rs`
- Modify: `kansei-core/src/lights/mod.rs` (may need to export individual types)

- [ ] **Step 1: Define SceneNode enum**

In `scene.rs`:
```rust
pub enum SceneNode {
    Transform(Object3D),
    Renderable(Renderable),
    Light(Light),
}

impl SceneNode {
    pub fn object(&self) -> &Object3D {
        match self {
            SceneNode::Transform(o) => o,
            SceneNode::Renderable(r) => &r.object,
            SceneNode::Light(_) => panic!("Light has no Object3D"),
        }
    }
    pub fn object_mut(&mut self) -> &mut Object3D {
        match self {
            SceneNode::Transform(o) => o,
            SceneNode::Renderable(r) => &mut r.object,
            SceneNode::Light(_) => panic!("Light has no Object3D"),
        }
    }
}
```

- [ ] **Step 2: Rewrite Scene struct**

```rust
pub struct Scene {
    pub object: Object3D,
    children: Vec<SceneNode>,
    // Collected during prepare():
    collected_renderables: Vec<usize>,  // indices into children
    collected_opaque: Vec<usize>,
    collected_transparent: Vec<usize>,
    collected_dir_lights: Vec<usize>,
    collected_point_lights: Vec<usize>,
    collected_area_lights: Vec<usize>,
}
```

- [ ] **Step 3: Implement add() and prepare()**

```rust
impl Scene {
    pub fn new() -> Self { ... }

    pub fn add(&mut self, node: SceneNode) -> usize {
        let idx = self.children.len();
        self.children.push(node);
        idx
    }

    pub fn prepare(&mut self, camera_pos: &Vec3) {
        // Clear collections
        self.collected_renderables.clear();
        self.collected_opaque.clear();
        self.collected_transparent.clear();
        self.collected_dir_lights.clear();
        self.collected_point_lights.clear();
        self.collected_area_lights.clear();

        // Walk children, update world matrices, collect renderables and lights
        for i in 0..self.children.len() {
            // Update model matrix if dirty
            match &mut self.children[i] {
                SceneNode::Transform(o) | SceneNode::Renderable(Renderable { object: o, .. }) => {
                    if o.is_dirty() { o.update_model_matrix(); }
                    o.update_world_matrix(None); // root level — no parent
                    o.update_normal_matrix();
                }
                _ => {}
            }

            // Collect
            match &self.children[i] {
                SceneNode::Renderable(r) if r.visible => {
                    self.collected_renderables.push(i);
                    if r.is_transparent() {
                        self.collected_transparent.push(i);
                    } else {
                        self.collected_opaque.push(i);
                    }
                }
                SceneNode::Light(Light::Directional(_)) => self.collected_dir_lights.push(i),
                SceneNode::Light(Light::Point(_)) => self.collected_point_lights.push(i),
                SceneNode::Light(Light::Area(_)) => self.collected_area_lights.push(i),
                _ => {}
            }
        }

        // Sort transparent back-to-front
        let children = &self.children;
        self.collected_transparent.sort_by(|&a, &b| {
            let ra = match &children[a] { SceneNode::Renderable(r) => r, _ => unreachable!() };
            let rb = match &children[b] { SceneNode::Renderable(r) => r, _ => unreachable!() };
            if ra.render_order != rb.render_order {
                return ra.render_order.cmp(&rb.render_order);
            }
            let da = ra.position.distance_to_squared(camera_pos);
            let db = rb.position.distance_to_squared(camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn ordered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.collected_opaque.iter().copied().chain(self.collected_transparent.iter().copied())
    }

    pub fn get_renderable(&self, idx: usize) -> Option<&Renderable> {
        match &self.children[idx] {
            SceneNode::Renderable(r) => Some(r),
            _ => None,
        }
    }

    pub fn get_renderable_mut(&mut self, idx: usize) -> Option<&mut Renderable> {
        match &mut self.children[idx] {
            SceneNode::Renderable(r) => Some(r),
            _ => None,
        }
    }

    pub fn lights(&self) -> impl Iterator<Item = &Light> {
        self.children.iter().filter_map(|n| match n {
            SceneNode::Light(l) => Some(l),
            _ => None,
        })
    }

    pub fn len(&self) -> usize {
        self.collected_renderables.len()
    }
}
```

- [ ] **Step 4: Update mod.rs exports**

```rust
pub use scene::{Scene, SceneNode};
```

Remove `Object3DNode` trait export.

- [ ] **Step 5: Update Renderer to use new Scene API**

In `renderer.rs`, the render loop currently calls `scene.renderables()` and `scene.ordered_indices()`. Update to use `scene.get_renderable(idx)` and `scene.lights()`.

The light packing in `upload_all()` currently reads `scene.lights` directly. Change to `scene.lights().collect::<Vec<_>>()` or iterate.

- [ ] **Step 6: Update all examples**

Examples currently call `scene.add(renderable)` — change to `scene.add(SceneNode::Renderable(renderable))`.
Examples calling `scene.add_light(light)` — change to `scene.add(SceneNode::Light(light))`.
Examples calling `scene.get(idx)` / `scene.get_mut(idx)` — change to `scene.get_renderable(idx)` / `scene.get_renderable_mut(idx)`.

- [ ] **Step 7: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
cargo build --example spinning_box --example lit_scene 2>&1 | tail -3
git add kansei-core/src/objects/ kansei-core/src/renderers/ kansei-native/examples/
git commit -m "refactor: Scene as SceneNode tree with typed light collections"
```

---

### Task 4: Camera owns bind group

**Files:**
- Modify: `kansei-core/src/cameras/camera.rs`
- Modify: `kansei-core/src/renderers/renderer.rs`

- [ ] **Step 1: Add GPU fields + initialize() to Camera**

```rust
pub struct Camera {
    pub object: Object3D,
    pub fov: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
    pub view_matrix: Mat4,
    pub inverse_view_matrix: Mat4,
    pub projection_matrix: Mat4,
    // GPU resources (created during initialize)
    view_buf: Option<wgpu::Buffer>,
    proj_buf: Option<wgpu::Buffer>,
    bind_group: Option<wgpu::BindGroup>,
    initialized: bool,
}

impl Camera {
    pub fn initialize(&mut self, device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout) {
        if self.initialized { return; }
        self.view_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera/View"), size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.proj_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Camera/Proj"), size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Camera/BG"), layout: camera_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.view_buf.as_ref().unwrap().as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.proj_buf.as_ref().unwrap().as_entire_binding() },
            ],
        }));
        self.initialized = true;
    }

    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(ref buf) = self.view_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.view_matrix.as_slice()));
        }
        if let Some(ref buf) = self.proj_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.projection_matrix.as_slice()));
        }
    }

    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }
}
```

Note: The camera BGL has 3 bindings in current code (view, proj, lights). The light buffer stays in the Renderer for now — we'll add a third binding to Camera later or keep lights in a separate bind group. For this task, Camera's bind group has view + proj only. The light buffer binding moves to group 1 binding 2 as part of the camera bind group.

Actually — looking at the current SharedLayouts, the camera BGL already has 3 entries (view, proj, lights). Camera needs to own all three. But the light buffer is uploaded by the Renderer from scene.lights(). Solution: Camera creates the bind group with a light buffer that the Renderer writes to.

Simpler: Camera creates view + proj buffers. Renderer creates the light buffer. Camera's bind group includes all three. The Renderer passes the light buffer to Camera during initialize:

```rust
pub fn initialize(&mut self, device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout, light_buf: &wgpu::Buffer) {
    // ... create view_buf, proj_buf ...
    self.bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Camera/BG"), layout: camera_bgl,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: self.view_buf.as_ref().unwrap().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: self.proj_buf.as_ref().unwrap().as_entire_binding() },
            wgpu::BindGroupEntry { binding: 2, resource: light_buf.as_entire_binding() },
        ],
    }));
}
```

- [ ] **Step 2: Remove camera buffers/bind group from Renderer**

In `renderer.rs`, remove fields: `camera_view_buf`, `camera_proj_buf`, `camera_bind_group`. Remove their creation from `initialize()`.

In `render()`, change:
```rust
// Initialize camera if needed
if !camera.initialized {
    let device = self.device.as_ref().unwrap();
    let shared = self.shared_layouts.as_ref().unwrap();
    camera.initialize(device, &shared.camera_bgl, self.light_buf.as_ref().unwrap());
}
```

Upload camera matrices:
```rust
camera.upload(self.queue.as_ref().unwrap());
```

Set bind group:
```rust
pass.set_bind_group(1, camera.bind_group().unwrap(), &[]);
```

- [ ] **Step 3: Remove look_at_target from Camera**

Replace `look_at_target: Option<Vec3>` with direct Euler extraction in `look_at()`:
```rust
pub fn look_at(&mut self, target: &Vec3) {
    self.object.look_at(target);  // sets rotation via Euler extraction
    self.update_view_matrix();
}
```

`update_view_matrix()` always rebuilds from Object3D:
```rust
pub fn update_view_matrix(&mut self) {
    self.object.update_model_matrix();
    self.object.update_world_matrix(None);
    self.view_matrix = self.object.world_matrix.inverse();
    self.inverse_view_matrix = self.view_matrix.inverse();
}
```

- [ ] **Step 4: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
cargo build --example spinning_box 2>&1 | tail -3
git add kansei-core/src/cameras/ kansei-core/src/renderers/
git commit -m "refactor: Camera owns bind group, Renderer drives initialize"
```

---

### Task 5: Material initialize() stores SharedLayouts

**Files:**
- Modify: `kansei-core/src/materials/material.rs`
- Modify: `kansei-core/src/renderers/renderer.rs`

- [ ] **Step 1: Add initialize() method, store shared_layouts reference**

Material stores an `Option<SharedLayouts>` clone (SharedLayouts contains only wgpu BindGroupLayouts which are cheap Arc clones):

Actually, SharedLayouts fields are `wgpu::BindGroupLayout` which is `Clone`. But storing the whole struct means Material needs ownership. Better: Material stores just the pipeline layout (already does) and the material BGL. The `ensure_shared()` pattern already works — just remove `shared` from `get_pipeline()` signature.

```rust
pub fn initialize(&mut self, device: &wgpu::Device, shared: &SharedLayouts) {
    self.ensure_shared(device, shared);
}

pub fn get_pipeline(
    &mut self,
    device: &wgpu::Device,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    color_formats: &[wgpu::TextureFormat],
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
) -> &wgpu::RenderPipeline {
    // Pipeline layout already created during initialize/ensure_shared
    // ... rest of pipeline cache logic unchanged
}
```

- [ ] **Step 2: Update Renderer to call material.initialize()**

In render(), during Phase 0.5:
```rust
for r in scene.renderables_mut() {
    if !r.geometry.initialized { r.geometry.initialize(device); }
    if !r.material.initialized { r.material.initialize(device, shared); }
    r.material.get_pipeline(device, &layouts, &[format], depth_format, sample_count);
}
```

- [ ] **Step 3: Add outputs_emissive to MaterialOptions**

```rust
pub struct MaterialOptions {
    pub transparent: bool,
    pub depth_write: Option<bool>,
    pub depth_compare: wgpu::CompareFunction,
    pub cull_mode: CullMode,
    pub topology: wgpu::PrimitiveTopology,
    pub outputs_emissive: bool,  // NEW
}
```

Default to `false`.

- [ ] **Step 4: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
git add kansei-core/src/materials/ kansei-core/src/renderers/
git commit -m "refactor: Material.initialize() stores shared layouts, get_pipeline simplified"
```

---

### Task 6: Renderer device private + remove factory methods

**Files:**
- Modify: `kansei-core/src/renderers/renderer.rs`
- Modify: `kansei-core/src/renderers/mod.rs`

- [ ] **Step 1: Change device() and queue() to pub(crate)**

```rust
pub(crate) fn device(&self) -> &wgpu::Device { ... }
pub(crate) fn queue(&self) -> &wgpu::Queue { ... }
```

- [ ] **Step 2: Remove factory method wrappers**

Remove these public methods from Renderer:
- `create_shader_module`
- `create_buffer`
- `create_buffer_init`
- `create_texture`
- `create_bind_group_layout`
- `create_pipeline_layout`
- `create_render_pipeline`
- `create_bind_group`
- `create_sampler`
- `write_buffer`

Keep:
- `compute_batch()` — public, needed by FluidSim
- `submit()` — public, needed by FluidSim
- `create_command_encoder()` — public, needed by FluidSim

- [ ] **Step 3: Remove shared_layouts() from public API**

```rust
pub(crate) fn shared_layouts(&self) -> &SharedLayouts { ... }
```

- [ ] **Step 4: Fix compilation errors in examples**

Examples that call `renderer.device()` need to be updated. The spinning_box example creates material uniform buffers via `device.create_buffer()` — this should move to `material.set_uniform_bindable()` which creates the buffer during Material::initialize().

For each example:
- Remove `let device = renderer.device();`
- Replace `device.create_buffer(...)` with `material.set_uniform_bindable(binding, label, &data)`
- Remove `renderer.queue().write_buffer(...)` — material handles this
- Remove `material.create_bind_group(device, shared, ...)` — Renderer calls this internally

- [ ] **Step 5: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
cargo build --example spinning_box --example lit_scene 2>&1 | tail -3
git add kansei-core/src/renderers/ kansei-native/examples/
git commit -m "refactor: Renderer device private, remove factory methods"
```

---

### Task 7: FluidSimulation stores device/queue from initialize

**Files:**
- Modify: `kansei-core/src/simulations/fluid/simulation.rs`
- Modify: `kansei-native/examples/fluid_engine.rs`
- Modify: `kansei-wasm/src/lib.rs`

- [ ] **Step 1: Store device + queue in FluidSimulation**

Add fields:
```rust
pub struct FluidSimulation {
    // ... existing fields ...
    device: Option<wgpu::Device>,   // Clone (cheap Arc)
    queue: Option<wgpu::Queue>,     // Clone (cheap Arc)
}
```

- [ ] **Step 2: Change initialize() to take &Renderer**

```rust
pub fn initialize(&mut self, positions: &[f32], renderer: &Renderer) {
    let device = renderer.device();
    let queue = renderer.queue();
    self.device = Some(device.clone());
    self.queue = Some(queue.clone());
    // ... rest of existing initialize logic using device ...
}
```

- [ ] **Step 3: Change update() to use stored device/queue**

```rust
pub fn update(&mut self, dt: f32, mouse_strength: f32, mouse_pos: [f32; 2], mouse_dir: [f32; 2]) {
    let device = self.device.as_ref().unwrap();
    let queue = self.queue.as_ref().unwrap();
    for _s in 0..self.params.substeps {
        self.update_substep_internal(device, queue, dt, mouse_strength, mouse_pos, mouse_dir);
    }
}
```

Same pattern for `update_batched()`, `set_camera_matrices()`, `rebuild_grid()`.

- [ ] **Step 4: Update FluidDensityField and FluidSurfaceRenderer similarly**

These also take `&wgpu::Device` — change to take `&Renderer` in their constructors/init methods and store device/queue.

- [ ] **Step 5: Update examples and WASM**

In `fluid_engine.rs`: `sim.initialize(&positions, &renderer)` instead of `sim.initialize(&positions, device)`.
In `kansei-wasm/src/lib.rs`: same pattern.

Remove `device` and `queue` variables from example render loops — they're stored in sim.

- [ ] **Step 6: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
cargo build --example fluid_engine 2>&1 | tail -3
git add kansei-core/src/simulations/ kansei-native/examples/ kansei-wasm/src/
git commit -m "refactor: FluidSimulation stores device/queue from initialize"
```

---

### Task 8: PostProcessingVolume drives render

**Files:**
- Modify: `kansei-core/src/postprocessing/volume.rs`
- Modify: `kansei-core/src/renderers/renderer.rs`
- Modify: `kansei-native/examples/postprocess_scene.rs`

- [ ] **Step 1: PostProcessingVolume stores device/queue/surface refs**

```rust
pub struct PostProcessingVolume {
    pub effects: Vec<Box<dyn PostProcessingEffect>>,
    gbuffer: Option<GBuffer>,
    blit_pipeline: Option<wgpu::RenderPipeline>,
    blit_sampler: Option<wgpu::Sampler>,
    blit_bgl: Option<wgpu::BindGroupLayout>,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    presentation_format: wgpu::TextureFormat,
}
```

- [ ] **Step 2: Constructor takes &Renderer**

```rust
pub fn new(renderer: &Renderer, effects: Vec<Box<dyn PostProcessingEffect>>) -> Self {
    Self {
        effects,
        gbuffer: None,
        blit_pipeline: None, blit_sampler: None, blit_bgl: None,
        device: Some(renderer.device().clone()),
        queue: Some(renderer.queue().clone()),
        surface: None, // surface can't be cloned — volume needs another approach for present
        presentation_format: renderer.presentation_format(),
    }
}
```

Note: `wgpu::Surface` is not `Clone`. The volume needs access to the surface for `get_current_texture()`. Options: pass surface on each `render()` call, or store a reference. For now, `render()` takes `&wgpu::Surface`:

```rust
pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera, surface: &wgpu::Surface, width: u32, height: u32) {
    // ... GBuffer render, effects chain, blit to surface
}
```

- [ ] **Step 3: Remove render_with_postprocessing from Renderer**

Delete `Renderer::render_with_postprocessing()`.

- [ ] **Step 4: Update postprocess_scene.rs example**

```rust
let mut volume = PostProcessingVolume::new(&renderer, effects);
// In render loop:
let surface = renderer.surface().unwrap();
volume.render(&mut scene, &mut camera, surface, renderer.width(), renderer.height());
```

- [ ] **Step 5: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -10
cargo build --example postprocess_scene 2>&1 | tail -3
git add kansei-core/src/postprocessing/ kansei-core/src/renderers/ kansei-native/examples/
git commit -m "refactor: PostProcessingVolume drives render, removed render_with_postprocessing"
```

---

### Task 9: Update all examples for new API

**Files:**
- Modify: All example files in `kansei-native/examples/`
- Modify: `kansei-wasm/src/lib.rs`

This task catches any remaining compilation errors from the previous tasks. Each example needs:
- `SceneNode::Renderable(...)` for scene.add
- `SceneNode::Light(...)` for lights
- No `renderer.device()` calls
- Material uniforms via `set_uniform_bindable()`
- No `update_normal_matrix(&camera.view_matrix)` calls

- [ ] **Step 1: Fix each example one by one**

For each example, run `cargo build --example <name>`, fix errors, repeat.

- [ ] **Step 2: Build all examples**

```bash
cargo build --examples 2>&1 | tail -10
```

- [ ] **Step 3: Build WASM**

```bash
cd kansei-wasm && wasm-pack build --target web 2>&1 | tail -5
```

- [ ] **Step 4: Commit**

```bash
git add kansei-native/examples/ kansei-wasm/src/
git commit -m "refactor: update all examples for new architecture"
```

---

### Task 10: Final verification — run all examples

**Files:** None (just verify)

- [ ] **Step 1: Run spinning_box**

```bash
cargo run --example spinning_box --release
```
Verify: spinning box renders with correct lighting.

- [ ] **Step 2: Run lit_scene**

```bash
cargo run --example lit_scene --release
```
Verify: multiple lights, correct shading.

- [ ] **Step 3: Run shadow_scene**

```bash
cargo run --example shadow_scene --release
```
Verify: shadows render correctly.

- [ ] **Step 4: Run fluid_engine**

```bash
cargo run --example fluid_engine --release
```
Verify: fluid sim runs, particles render, cornell box visible.

- [ ] **Step 5: Build and test WASM**

```bash
cd kansei-wasm && wasm-pack build --release --target web
```
Verify in browser.

- [ ] **Step 6: Commit final state**

```bash
git add -A
git commit -m "feat: Rust architecture aligned 1:1 with TypeScript engine"
```

---

## Post-Plan Notes

### What this plan produces:
- Bind group slots match TS (camera=1, mesh=2)
- Object3D with model_matrix + world_matrix + world-space normal matrix
- Scene as SceneNode tree with typed light collections
- Camera owns its bind group
- Material initialized by Renderer
- Device private in Renderer
- FluidSimulation stores device/queue
- PostProcessingVolume drives render
- All shaders portable between TS and Rust
- All examples updated

### Follow-ups (not in this plan):
- Point shadow / cubemap shadows (5-binding shadow BGL)
- Render bundle caching
- MSAA depth-copy pass
- parseIncludes() shader preprocessor
- readBackBuffer GPU readback
- Path tracer integration
- onChange dirty callbacks on vectors
- System pattern (renderer.computeSystem)
