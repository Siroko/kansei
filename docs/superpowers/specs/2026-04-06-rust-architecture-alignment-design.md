# Rust Architecture Alignment — 1:1 TS Port

## Goal

Align the Rust kansei-core architecture to match the TypeScript engine 1:1. Same bind group slots, same matrix semantics, same object lifecycle, same shader portability. Device stays private in Renderer — subsystems receive it only via initialize() calls driven by the Renderer.

## Scope

### In scope (this refactor):
1. Bind group slot order matches TS
2. Object3D model_matrix/world_matrix split
3. Scene as Object3D with SceneNode tree
4. Renderer encapsulation (device private)
5. Material lifecycle matches TS
6. Camera owns bind group
7. FluidSimulation stores renderer refs
8. PostProcessingVolume drives render
9. Normal matrix world-space
10. All WGSL shaders updated for new bind group order

### Out of scope (follow-ups):
- Point shadow / cubemap shadows
- Render bundle caching
- MSAA depth-copy pass
- Path tracer integration
- parseIncludes() shader preprocessor
- readBackBuffer GPU readback
- onChange dirty callbacks on vectors

---

## 1. Bind Group Slot Order

**Current Rust:** material=0, mesh=1, camera=2, shadow=3

**TS (ground truth):** material=0, camera=1, mesh=2, shadow=3

### Changes:
- `BindGroupSlot` enum: `Material=0, Camera=1, Mesh=2, Shadow=3`
- `SharedLayouts::new()`: create `camera_bgl` before `mesh_bgl`, pipeline layout order = `[material_bgl, camera_bgl, mesh_bgl, shadow_bgl]`
- `Material::ensure_shared()`: pipeline layout uses `[material_bgl, shared.camera_bgl, shared.mesh_bgl, shared.shadow_bgl]`
- `Renderer::render()`: `pass.set_bind_group(1, camera_bg)`, `pass.set_bind_group(2, mesh_bg, &[offset, offset])`, `pass.set_bind_group(3, shadow_bg)`
- All WGSL shaders: swap `@group(1)` ↔ `@group(2)` for camera/mesh bindings

### Affected shaders:
- `shaders/basic.wgsl`
- `shaders/basic_lit.wgsl`
- `shaders/basic_instanced.wgsl`
- `shaders/shadow_vs.wgsl`
- Any inline WGSL in examples

---

## 2. Object3D — Two-Matrix System

**Current Rust:** Single `world_matrix` computed from T*R*S directly. Normal matrix is view-space.

**TS:** Separate `model_matrix` (local T*R*S) and `world_matrix` (parent.world * model). Normal matrix is world-space.

### Changes to Object3D:
```rust
pub struct Object3D {
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
    pub model_matrix: Mat4,     // local T * Rz * Ry * Rx * S
    pub world_matrix: Mat4,     // parent.world * model
    pub normal_matrix: Mat4,    // transpose(inverse(world_matrix))
    pub children: Vec<usize>,   // changes to Vec<SceneNode> in Section 3
    pub parent: Option<usize>,
    pub dirty: bool,
}
```

### Methods:
- `update_model_matrix()` — builds model_matrix from position/rotation/scale. Sets dirty=false.
- `update_world_matrix(parent_world: Option<&Mat4>)` — if parent, world = parent * model. Otherwise world = model.
- `update_normal_matrix()` — `normal_matrix = transpose(inverse(world_matrix))`. No view_matrix parameter.
- `look_at(target)` — computes look-at matrix, extracts Euler angles into `rotation` (matching TS), sets dirty.

---

## 3. Scene — Object3D Tree with SceneNode

**Current Rust:** Flat `Vec<Renderable>` with arena indices. Scene does not extend Object3D.

**TS:** `Scene extends Object3D`. Children are mixed Object3D/Renderable/Light. Recursive traversal in `prepare()`.

### SceneNode enum:
```rust
pub enum SceneNode {
    Transform(Object3D),
    Renderable(Renderable),
    Light(Light),
}
```

Each variant carries its own Object3D (Renderable and Light wrap one via composition).

### Scene struct:
```rust
pub struct Scene {
    pub object: Object3D,
    children: Vec<SceneNode>,
    // Extracted during prepare():
    opaque_objects: Vec<&Renderable>,
    transparent_objects: Vec<&Renderable>,
    directional_lights: Vec<&DirectionalLight>,
    point_lights: Vec<&PointLight>,
    area_lights: Vec<&AreaLight>,
}
```

### Methods:
- `add(node: SceneNode)` — adds to children
- `prepare(camera_pos: &Vec3)` — recursive traversal:
  1. Walk tree depth-first
  2. Propagate world matrices (parent.world * child.model)
  3. Collect renderables into opaque/transparent (sorted)
  4. Collect lights by type
- `ordered_objects()` — returns opaque + sorted transparent
- `directional_lights()`, `point_lights()`, `area_lights()` — typed accessors

### Hierarchy example:
```rust
let mut arm = Object3D::new();
arm.set_position(1.0, 0.0, 0.0);

let sword = Renderable::new(sword_geo, sword_mat);
scene.add(SceneNode::Transform(arm));
// arm.add(SceneNode::Renderable(sword)); — via tree API
```

---

## 4. Renderer — Device Private

**Current Rust:** `renderer.device()` is public. Examples create buffers directly.

**TS:** `renderer.device` is public but subsystems receive it via method calls. User code calls `renderer.render(scene, camera)`.

### Changes:
- `Renderer::device()` becomes `pub(crate)` — not accessible from examples
- `Renderer::queue()` becomes `pub(crate)`
- `Renderer::render()` internally calls:
  - `camera.initialize(device)` if not initialized
  - `scene.prepare(camera_pos)` — propagates world matrices, collects renderables/lights
  - For each renderable: `geometry.initialize(device)`, `material.initialize(device, shared_layouts)`
  - `material.get_pipeline(device, ...)` for pipeline cache
  - Upload matrices, set bind groups, draw
- `Renderer::compute_batch(passes)` — public, for user-driven compute (FluidSim calls this)
- Remove `renderer.shared_layouts()` from public API

### Factory methods removed:
The current `create_buffer`, `create_texture`, `create_shader_module` etc. wrappers are removed. Subsystems receive device via initialize() only.

---

## 5. Material — Initialize via Renderer

**Current Rust:** `Material::create_bind_group(device, shared, resources)` — user calls with device.

**TS:** `material.initialize(device, ...)` called by Renderer. User never touches device.

### Changes:
- `Material::new(label, shader_code, bindings, options)` — pure data, no device
- `Material::initialize(device: &wgpu::Device, shared: &SharedLayouts)` — creates shader module + pipeline layout. Called by Renderer.
- `Material::get_pipeline(device, vertex_layouts, color_formats, depth_format, sample_count)` — no `shared` param (stored from init)
- `Material::set_uniform(binding: u32, data: &[u8])` — stores uniform data. Buffer created during initialize.
- `Material::bind_group()` — returns `Option<&BindGroup>`
- Add `outputs_emissive: bool` to MaterialOptions

### Pipeline layout (matching TS order):
```rust
bind_group_layouts: &[material_bgl, shared.camera_bgl, shared.mesh_bgl, shared.shadow_bgl]
//                     group 0       group 1            group 2           group 3
```

---

## 6. Camera — Owns Bind Group

**Current Rust:** Renderer owns camera uniform buffers and bind group.

**TS:** Camera creates its own BindableGroup with view + projection matrices.

### Changes:
- `Camera::new(fov, near, far, aspect)` — no device, pure data
- `Camera::initialize(device: &wgpu::Device, camera_bgl: &wgpu::BindGroupLayout)` — creates view/projection uniform buffers + bind group. Called by Renderer.
- `Camera::bind_group()` — returns `&wgpu::BindGroup` for group 1
- `Camera::upload(queue: &wgpu::Queue)` — writes current view/projection matrices to GPU
- Remove camera buffers and bind group from Renderer
- `lookAt(target)` extracts Euler angles into `rotation` (no stored target)
- `update_view_matrix()` always rebuilds from position/rotation

---

## 7. FluidSimulation — Stores Device Refs

**Current Rust:** `sim.initialize(&positions, device)` + `sim.update(device, queue, dt, ...)`

**TS:** `new FluidSimulation(renderer, settings)` — stores renderer. `sim.update(dt, mouse)` — no renderer param.

### Changes:
- `FluidSimulation::new(options)` — no device
- `FluidSimulation::initialize(&mut self, positions: &[f32], renderer: &Renderer)` — stores device + queue (Clone — cheap Arc copies in wgpu). Creates all GPU resources.
- `FluidSimulation::update(&mut self, dt, mouse_strength, mouse_pos, mouse_dir)` — uses stored device/queue. No external params for GPU access.
- Same pattern for `FluidDensityField` and `FluidSurfaceRenderer`

---

## 8. PostProcessingVolume — Drives Render

**Current Rust:** `renderer.render_with_postprocessing(scene, camera, volume)`

**TS:** `volume.render(scene, camera)` — volume drives the render using stored renderer ref.

### Changes:
- `PostProcessingVolume::new(renderer: &Renderer, effects: Vec<Box<dyn PostProcessingEffect>>)` — stores device/queue/surface refs
- `PostProcessingVolume::render(&mut self, scene: &mut Scene, camera: &mut Camera)` — orchestrates: render to GBuffer, run effects, blit to canvas
- Remove `Renderer::render_with_postprocessing()`
- Volume calls Renderer methods internally for GBuffer rendering

---

## 9. Normal Matrix — World-Space

**Current Rust:** `update_normal_matrix(view_matrix)` computes `transpose(inverse(view * world))` — view-space normals.

**TS:** `transpose(inverse(world_matrix))` — world-space normals. No view matrix parameter.

### Change:
```rust
pub fn update_normal_matrix(&mut self) {
    self.normal_matrix = Mat4::from(self.world_matrix.to_glam().inverse().transpose());
}
```

All shaders already expect world-space normals (the TS shaders do). The Rust shaders will now match.

---

## 10. Shader Updates

All WGSL shaders updated for:
1. Group 1 = camera (was mesh), Group 2 = mesh (was camera)
2. Normal matrix is world-space (no shader change needed if shaders already treat it as world-space)

Files to update:
- `shaders/basic.wgsl`
- `shaders/basic_lit.wgsl`
- `shaders/basic_instanced.wgsl`
- `shaders/shadow_vs.wgsl`
- `shaders/blit.wgsl` (no groups 1/2, unchanged)
- `shaders/bloom_*.wgsl` (compute only, unchanged)
- `shaders/color_grading.wgsl` (compute only, unchanged)
- Inline WGSL in examples (fluid particle shader, cornell box shader)
- WASM inline shaders

---

## Migration Order

1. Bind group slot order (breaks everything, fix first)
2. Object3D two-matrix system + normal matrix fix
3. Scene as SceneNode tree
4. Camera owns bind group
5. Material initialize lifecycle
6. Renderer device private
7. FluidSimulation stores device refs
8. PostProcessingVolume drives render
9. Update all shaders
10. Update all examples
