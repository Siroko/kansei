# InstancedGeometry Implementation Plan (Plan 2a)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add instanced rendering support so the engine can efficiently draw thousands of objects with per-instance data (transforms, colors, etc.) using a single draw call.

**Architecture:** `InstanceBuffer` holds per-instance GPU data with vertex attribute descriptors. `Renderable` gains an optional `instance_buffers: Vec<InstanceBuffer>` field. The Renderer sets extra vertex buffers (slot 1+) with `step_mode: Instance` and passes the combined vertex layouts to Material for pipeline creation. `PipelineKey` gains `num_vertex_buffers` to differentiate instanced from non-instanced pipelines.

**Tech Stack:** Rust, wgpu 24, bytemuck 1

---

## File Structure

### Files to create:
- `rust/kansei-core/src/buffers/instance_buffer.rs` — InstanceBuffer struct with vertex attribute descriptors
- `rust/kansei-core/src/shaders/basic_instanced.wgsl` — Instanced material shader (reads per-instance world matrix from vertex attribute)
- `rust/kansei-native/examples/instanced_cubes.rs` — Validation example: 1000 instanced cubes

### Files to modify:
- `rust/kansei-core/src/buffers/mod.rs` — Export InstanceBuffer
- `rust/kansei-core/src/objects/renderable.rs` — Add instance_buffers field
- `rust/kansei-core/src/materials/material.rs` — Add num_vertex_buffers to PipelineKey
- `rust/kansei-core/src/renderers/renderer.rs` — Set extra vertex buffers, pass combined layouts to pipeline creation

---

### Task 1: Create InstanceBuffer

**Files:**
- Create: `rust/kansei-core/src/buffers/instance_buffer.rs`
- Modify: `rust/kansei-core/src/buffers/mod.rs`

An InstanceBuffer holds per-instance data on the GPU with vertex attribute descriptors for the shader. Matches the TS ComputeBuffer when used for instancing.

- [ ] **Step 1: Create instance_buffer.rs**

```rust
// rust/kansei-core/src/buffers/instance_buffer.rs

/// A single vertex attribute within an instance buffer.
#[derive(Debug, Clone)]
pub struct InstanceAttribute {
    pub shader_location: u32,
    pub offset: u64,
    pub format: wgpu::VertexFormat,
}

/// Per-instance GPU buffer with vertex attribute descriptors.
/// Used for instanced rendering — data steps once per instance, not per vertex.
pub struct InstanceBuffer {
    pub label: String,
    pub data: Vec<u8>,
    pub stride: u64,
    pub attributes: Vec<InstanceAttribute>,
    pub gpu_buffer: Option<wgpu::Buffer>,
    pub initialized: bool,
}

impl InstanceBuffer {
    /// Create an instance buffer from typed data.
    /// `stride` is bytes per instance. `attributes` describe how the shader reads each instance.
    pub fn new(label: &str, data: &[u8], stride: u64, attributes: Vec<InstanceAttribute>) -> Self {
        Self {
            label: label.to_string(),
            data: data.to_vec(),
            stride,
            attributes,
            gpu_buffer: None,
            initialized: false,
        }
    }

    /// Convenience: create from a slice of f32 data with a single vec4 attribute.
    pub fn from_f32_vec4(label: &str, data: &[f32], shader_location: u32) -> Self {
        Self::new(
            label,
            bytemuck::cast_slice(data),
            16, // vec4<f32> = 16 bytes
            vec![InstanceAttribute {
                shader_location,
                offset: 0,
                format: wgpu::VertexFormat::Float32x4,
            }],
        )
    }

    /// Convenience: create from a slice of mat4 data (4 x vec4 attributes at consecutive locations).
    pub fn from_mat4(label: &str, data: &[f32], base_shader_location: u32) -> Self {
        Self::new(
            label,
            bytemuck::cast_slice(data),
            64, // mat4 = 64 bytes
            vec![
                InstanceAttribute { shader_location: base_shader_location, offset: 0, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 1, offset: 16, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 2, offset: 32, format: wgpu::VertexFormat::Float32x4 },
                InstanceAttribute { shader_location: base_shader_location + 3, offset: 48, format: wgpu::VertexFormat::Float32x4 },
            ],
        )
    }

    /// Initialize the GPU buffer.
    pub fn initialize(&mut self, device: &wgpu::Device) {
        use wgpu::util::DeviceExt;
        self.gpu_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(&format!("{}/Buffer", self.label)),
            contents: &self.data,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        self.initialized = true;
    }

    /// Update GPU buffer contents.
    pub fn update(&self, queue: &wgpu::Queue, data: &[u8]) {
        if let Some(ref buf) = self.gpu_buffer {
            queue.write_buffer(buf, 0, data);
        }
    }

    /// Build the wgpu VertexBufferLayout for this instance buffer.
    /// Returns owned data since the layout references heap-allocated attributes.
    pub fn vertex_layout(&self) -> InstanceBufferLayout {
        let attrs: Vec<wgpu::VertexAttribute> = self.attributes.iter().map(|a| {
            wgpu::VertexAttribute {
                format: a.format,
                offset: a.offset,
                shader_location: a.shader_location,
            }
        }).collect();

        InstanceBufferLayout {
            stride: self.stride,
            attributes: attrs,
        }
    }
}

/// Owned vertex buffer layout data for an instance buffer.
/// Needed because wgpu::VertexBufferLayout borrows its attributes slice.
pub struct InstanceBufferLayout {
    pub stride: u64,
    pub attributes: Vec<wgpu::VertexAttribute>,
}

impl InstanceBufferLayout {
    /// Borrow as a wgpu::VertexBufferLayout. The returned value borrows self.
    pub fn as_layout(&self) -> wgpu::VertexBufferLayout<'_> {
        wgpu::VertexBufferLayout {
            array_stride: self.stride,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &self.attributes,
        }
    }
}
```

- [ ] **Step 2: Update buffers/mod.rs**

Add to `rust/kansei-core/src/buffers/mod.rs`:

```rust
mod instance_buffer;
pub use instance_buffer::{InstanceBuffer, InstanceAttribute, InstanceBufferLayout};
```

- [ ] **Step 3: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5`
Expected: compiles with existing warnings only

- [ ] **Step 4: Commit**

```bash
git add kansei-core/src/buffers/instance_buffer.rs kansei-core/src/buffers/mod.rs
git commit -m "feat: add InstanceBuffer for per-instance vertex data"
```

---

### Task 2: Add instance_buffers to Renderable

**Files:**
- Modify: `rust/kansei-core/src/objects/renderable.rs`

Add an `instance_buffers: Vec<InstanceBuffer>` field so renderables can carry per-instance data. The `Renderable::new` constructor keeps the field empty (non-instanced). A new `new_instanced` constructor takes instance buffers.

- [ ] **Step 1: Update renderable.rs**

In `rust/kansei-core/src/objects/renderable.rs`, add the import and field:

At the top, add:
```rust
use crate::buffers::InstanceBuffer;
```

Add field to the struct:
```rust
pub struct Renderable {
    pub object: Object3D,
    pub geometry: Geometry,
    pub material: Material,
    pub instance_count: u32,
    pub instance_buffers: Vec<InstanceBuffer>,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub render_order: i32,
    pub visible: bool,
    pub material_dirty: bool,
}
```

Update `new()` to initialize the field:
```rust
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Self {
            object: Object3D::new(),
            geometry,
            material,
            instance_count: 1,
            instance_buffers: Vec::new(),
            cast_shadow: true,
            receive_shadow: true,
            render_order: 0,
            visible: true,
            material_dirty: true,
        }
    }
```

Add instanced constructor:
```rust
    /// Create an instanced renderable with per-instance data buffers.
    pub fn new_instanced(
        geometry: Geometry,
        material: Material,
        instance_count: u32,
        instance_buffers: Vec<InstanceBuffer>,
    ) -> Self {
        Self {
            object: Object3D::new(),
            geometry,
            material,
            instance_count,
            instance_buffers,
            cast_shadow: true,
            receive_shadow: true,
            render_order: 0,
            visible: true,
            material_dirty: true,
        }
    }

    /// Whether this renderable uses instanced rendering.
    pub fn is_instanced(&self) -> bool {
        !self.instance_buffers.is_empty()
    }
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5`

- [ ] **Step 3: Commit**

```bash
git add kansei-core/src/objects/renderable.rs
git commit -m "feat: add instance_buffers to Renderable for instanced rendering"
```

---

### Task 3: Add num_vertex_buffers to PipelineKey and update Renderer

**Files:**
- Modify: `rust/kansei-core/src/materials/material.rs` — Add num_vertex_buffers to PipelineKey
- Modify: `rust/kansei-core/src/renderers/renderer.rs` — Build combined vertex layouts, set extra vertex buffers

This is the integration task. The Renderer needs to:
1. Build combined vertex layouts (base + instance buffers) per renderable
2. Pass them to Material.get_pipeline with the right PipelineKey
3. Set extra vertex buffers in the draw loop (slot 1+ for instance buffers)
4. Initialize instance buffers alongside geometry

- [ ] **Step 1: Add num_vertex_buffers to PipelineKey**

In `rust/kansei-core/src/materials/material.rs`, modify the PipelineKey struct:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub(crate) color_formats: Vec<wgpu::TextureFormat>,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
    pub(crate) num_vertex_buffers: usize,
}
```

- [ ] **Step 2: Update Renderer pre-warm phase**

In `rust/kansei-core/src/renderers/renderer.rs`, in the `render()` method, replace the Phase 0.5 block (currently lines ~382-398) with:

```rust
        // Phase 0.5: Initialize geometries, instance buffers, and pre-warm pipelines
        {
            let device = self.device.as_ref().unwrap();
            let shared = self.shared_layouts.as_ref().unwrap();
            let format = self.presentation_format;
            let sample_count = self.config.sample_count;
            let depth_format = wgpu::TextureFormat::Depth24Plus;

            for r in scene.renderables_mut() {
                if !r.geometry.initialized {
                    r.geometry.initialize(device);
                }
                for ib in &mut r.instance_buffers {
                    if !ib.initialized {
                        ib.initialize(device);
                    }
                }

                // Build combined vertex layouts
                let instance_layouts: Vec<_> = r.instance_buffers.iter()
                    .map(|ib| ib.vertex_layout())
                    .collect();
                let mut layouts = vec![Vertex::LAYOUT];
                for il in &instance_layouts {
                    layouts.push(il.as_layout());
                }

                r.material.get_pipeline(
                    device, shared, &layouts,
                    &[format], depth_format, sample_count,
                );
            }
        }
```

- [ ] **Step 3: Update Renderer draw loop**

In the draw loop (currently lines ~460-486), replace the pipeline lookup and vertex buffer binding:

```rust
            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = &scene.renderables()[scene_idx];
                if !r.visible || !r.geometry.initialized { continue; }

                // Build pipeline key with vertex buffer count
                let num_vb = 1 + r.instance_buffers.len();
                let key = crate::materials::PipelineKey {
                    color_formats: vec![format],
                    depth_format,
                    sample_count,
                    num_vertex_buffers: num_vb,
                };
                let pipeline = match r.material.pipeline_cache.get(&key) {
                    Some(p) => p,
                    None => continue,
                };

                pass.set_pipeline(pipeline);

                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }

                let offset = (draw_idx as u32) * alignment;
                pass.set_bind_group(1, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                // Slot 0: base geometry vertex buffer
                pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                // Slot 1+: instance buffers
                for (i, ib) in r.instance_buffers.iter().enumerate() {
                    if let Some(ref buf) = ib.gpu_buffer {
                        pass.set_vertex_buffer((i + 1) as u32, buf.slice(..));
                    }
                }

                pass.set_index_buffer(
                    r.geometry.index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
            }
```

- [ ] **Step 4: Also update the PipelineKey construction in render() where it was outside the loop**

The old code had a `let key = ...` outside the loop. Remove that — the key is now per-renderable (inside the loop).

- [ ] **Step 5: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -10`
Fix any errors. The spinning_box example should still build since non-instanced renderables have `instance_buffers: Vec::new()` and `num_vertex_buffers: 1`.

- [ ] **Step 6: Verify spinning_box still works**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo build --example spinning_box 2>&1 | tail -3`
Expected: builds successfully

- [ ] **Step 7: Commit**

```bash
git add kansei-core/src/materials/material.rs kansei-core/src/renderers/renderer.rs
git commit -m "feat: Renderer supports instanced rendering with extra vertex buffers"
```

---

### Task 4: Create instanced shader

**Files:**
- Create: `rust/kansei-core/src/shaders/basic_instanced.wgsl`

This shader extends basic.wgsl to read a per-instance world matrix from vertex attributes (locations 3-6, one vec4 per row of mat4). It replaces the uniform world_matrix from group 1 binding 1 with the per-instance attribute. The normal matrix from group 1 binding 0 is still used for the base object.

- [ ] **Step 1: Create basic_instanced.wgsl**

```wgsl
// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Mesh (dynamic offset) ──
// binding 0 unused for instanced (normal computed from instance matrix)
// binding 1 unused for instanced (world matrix comes from vertex attribute)
@group(1) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> _world_matrix: mat4x4<f32>;

// ── Group 2: Camera ──
@group(2) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Vertex I/O ──
struct VertexInput {
    // Per-vertex (slot 0)
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // Per-instance (slot 1) — mat4 as 4 vec4 rows
    @location(3) instance_mat_0: vec4<f32>,
    @location(4) instance_mat_1: vec4<f32>,
    @location(5) instance_mat_2: vec4<f32>,
    @location(6) instance_mat_3: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    let instance_matrix = mat4x4<f32>(
        input.instance_mat_0,
        input.instance_mat_1,
        input.instance_mat_2,
        input.instance_mat_3,
    );

    var out: VertexOutput;
    let world_pos = instance_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    // Compute normal matrix as inverse-transpose of upper-left 3x3
    // For uniform scale, the instance matrix itself works
    out.world_normal = (instance_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    out.uv = input.uv;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let n = normalize(input.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let ambient = 0.15;
    return vec4<f32>(material.color.rgb * (ambient + ndotl * 0.85), material.color.a);
}
```

- [ ] **Step 2: Commit**

```bash
git add kansei-core/src/shaders/basic_instanced.wgsl
git commit -m "feat: add basic_instanced.wgsl for instanced rendering"
```

---

### Task 5: Create instanced cubes example

**Files:**
- Create: `rust/kansei-native/examples/instanced_cubes.rs`

Renders 1000 instanced cubes in a grid using a single draw call. Each instance has its own world matrix passed via an InstanceBuffer. Uses the basic_instanced.wgsl shader.

- [ ] **Step 1: Create the example**

```rust
// rust/kansei-native/examples/instanced_cubes.rs
use std::sync::Arc;
use std::time::Instant;

use kansei_core::math::{Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Material, MaterialOptions, Binding, BindingResource};
use kansei_core::buffers::InstanceBuffer;
use kansei_core::objects::{Renderable, Scene};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::{WindowEvent, ElementState, MouseButton};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const INSTANCED_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic_instanced.wgsl");

const GRID_SIZE: usize = 10; // 10x10x10 = 1000 cubes
const INSTANCE_COUNT: usize = GRID_SIZE * GRID_SIZE * GRID_SIZE;
const SPACING: f32 = 3.0;

struct OrbitCamera {
    target: glam::Vec3,
    distance: f32,
    azimuth: f32,
    elevation: f32,
    dragging: bool,
    last_mouse: Option<(f64, f64)>,
}

impl OrbitCamera {
    fn new(target: glam::Vec3, distance: f32) -> Self {
        Self { target, distance, azimuth: 0.5, elevation: 0.4, dragging: false, last_mouse: None }
    }
    fn eye(&self) -> glam::Vec3 {
        self.target + glam::Vec3::new(
            self.distance * self.azimuth.sin() * self.elevation.cos(),
            self.distance * self.elevation.sin(),
            self.distance * self.azimuth.cos() * self.elevation.cos(),
        )
    }
    fn on_mouse_move(&mut self, x: f64, y: f64) {
        if self.dragging {
            if let Some((lx, ly)) = self.last_mouse {
                self.azimuth -= (x - lx) as f32 * 0.005;
                self.elevation = (self.elevation + (y - ly) as f32 * 0.005).clamp(-1.5, 1.5);
            }
        }
        self.last_mouse = Some((x, y));
    }
    fn on_scroll(&mut self, delta: f32) {
        self.distance = (self.distance - delta * 0.5).clamp(5.0, 200.0);
    }
}

fn build_instance_matrices(time: f32) -> Vec<f32> {
    let mut data = vec![0.0f32; INSTANCE_COUNT * 16];
    let offset = (GRID_SIZE as f32 - 1.0) * SPACING * 0.5;
    let mut idx = 0;
    for x in 0..GRID_SIZE {
        for y in 0..GRID_SIZE {
            for z in 0..GRID_SIZE {
                let px = x as f32 * SPACING - offset;
                let py = y as f32 * SPACING - offset;
                let pz = z as f32 * SPACING - offset;
                // Each cube rotates based on its position + time
                let angle = time * 0.5 + (x + y + z) as f32 * 0.3;
                let m = glam::Mat4::from_translation(glam::Vec3::new(px, py, pz))
                    * glam::Mat4::from_rotation_y(angle);
                data[idx..idx + 16].copy_from_slice(&m.to_cols_array());
                idx += 16;
            }
        }
    }
    data
}

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    orbit: OrbitCamera,
    start_time: Instant,
    color_buf: Option<wgpu::Buffer>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None, renderer: None, scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 500.0, 1.0),
            orbit: OrbitCamera::new(glam::Vec3::ZERO, 50.0),
            start_time: Instant::now(), color_buf: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() { return; }
        let window = Arc::new(el.create_window(
            Window::default_attributes()
                .with_title("Kansei — 1000 Instanced Cubes")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        ).unwrap());
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface), ..Default::default()
        })).unwrap();

        let mut renderer = Renderer::new(RendererConfig {
            width: size.width, height: size.height, sample_count: 4,
            clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0), ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));

        let device = renderer.device();
        let shared = renderer.shared_layouts();

        // Color uniform
        let color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CubeColor"), size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color: [f32; 4] = [0.4, 0.7, 0.9, 1.0];
        renderer.queue().write_buffer(&color_buf, 0, bytemuck::cast_slice(&color));

        // Material
        let mut material = Material::new(
            "InstancedMaterial", INSTANCED_WGSL,
            vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
            MaterialOptions::default(),
        );
        material.create_bind_group(device, shared, &[(
            0, BindingResource::Buffer { buffer: &color_buf, offset: 0, size: None },
        )]);

        // Geometry — small cube shared by all instances
        let geometry = BoxGeometry::new(1.0, 1.0, 1.0);

        // Instance buffer — 1000 world matrices
        let matrices = build_instance_matrices(0.0);
        let instance_buf = InstanceBuffer::from_mat4("InstanceMatrices", &matrices, 3);

        // Create instanced renderable
        let renderable = Renderable::new_instanced(
            geometry, material,
            INSTANCE_COUNT as u32,
            vec![instance_buf],
        );
        self.scene.add(renderable);

        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();
        self.color_buf = Some(color_buf);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();
        log::info!("Instanced cubes — {} instances", INSTANCE_COUNT);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::Resized(s) => {
                if let Some(ref mut r) = self.renderer {
                    r.resize(s.width, s.height);
                    self.camera.aspect = s.width as f32 / s.height as f32;
                    self.camera.update_projection_matrix();
                }
            }
            WindowEvent::MouseInput { button: MouseButton::Left, state: s, .. } => {
                self.orbit.dragging = s == ElementState::Pressed;
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.orbit.on_mouse_move(position.x, position.y);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let dy = match delta {
                    winit::event::MouseScrollDelta::LineDelta(_, y) => y,
                    winit::event::MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.1,
                };
                self.orbit.on_scroll(dy);
            }
            WindowEvent::RedrawRequested => {
                let t = self.start_time.elapsed().as_secs_f32();

                // Update instance matrices (rotate each cube)
                if let Some(r) = self.scene.get_mut(0) {
                    let matrices = build_instance_matrices(t);
                    if let Some(ref ib) = r.instance_buffers.get(0) {
                        if let Some(ref renderer) = self.renderer {
                            ib.update(renderer.queue(), bytemuck::cast_slice(&matrices));
                        }
                    }
                }

                // Update camera from orbit
                let eye = self.orbit.eye();
                self.camera.set_position(eye.x, eye.y, eye.z);
                self.camera.look_at(&Vec3::new(
                    self.orbit.target.x, self.orbit.target.y, self.orbit.target.z,
                ));

                if let Some(ref mut renderer) = self.renderer {
                    renderer.render(&mut self.scene, &mut self.camera);
                }

                if let Some(ref w) = self.window { w.request_redraw(); }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei — Instanced Cubes");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
```

- [ ] **Step 2: Build the example**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo build --example instanced_cubes 2>&1 | tail -5`
Fix any compilation errors.

- [ ] **Step 3: Run the example**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo run --example instanced_cubes`
Expected: window opens with 1000 cubes in a 10x10x10 grid, each rotating, with orbit camera controls.

- [ ] **Step 4: Commit**

```bash
git add kansei-native/examples/instanced_cubes.rs
git commit -m "feat: add instanced cubes example — 1000 cubes in one draw call"
```

---

## Post-Plan Notes

### What this plan produces:
- `InstanceBuffer` — per-instance GPU data with vertex attribute descriptors
- `InstanceBufferLayout` — owned layout data that can produce wgpu::VertexBufferLayout borrows
- `Renderable::new_instanced()` — constructor for instanced rendering
- `PipelineKey` differentiates instanced vs non-instanced pipelines
- Renderer sets extra vertex buffers (slot 1+) with step_mode: Instance
- basic_instanced.wgsl — shader reading per-instance mat4 from vertex attributes
- instanced_cubes example — 1000 cubes rendered in a single draw call

### What comes next:
- **Plan 2b: Lights** — DirectionalLight, PointLight, AreaLight + uniform packing
- **Plan 2c: Shadows** — Shadow maps, cubemap shadows
- **Plan 2d: Post-processing** — SSAO, bloom, DoF, god rays, color grading, volumetric fog
- **Plan 2e: Loaders** — Texture + glTF
