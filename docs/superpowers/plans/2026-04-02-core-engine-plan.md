# Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the Kansei Rust engine rendering a spinning box through the full Scene → Camera → Material → Geometry → Renderer pipeline, proving the architecture works end-to-end.

**Architecture:** Renderable owns Geometry + Material. Renderer creates shared bind groups (mesh matrices with dynamic offsets, camera uniforms) and issues draw calls. Material creates pipelines against Renderer-provided shared layouts. Three-group bind layout: material (0), mesh (1, dynamic), camera (2). Shadows (group 3) deferred to Plan 2.

**Tech Stack:** Rust, wgpu 24, glam 0.29, bytemuck 1

---

## File Structure

### Files to create:
- `rust/kansei-core/src/shaders/basic.wgsl` — Basic lit material shader
- `rust/kansei-core/src/renderers/shared_layouts.rs` — Shared bind group layouts (mesh, camera)
- `rust/kansei-native/examples/spinning_box.rs` — Validation example

### Files to modify:
- `rust/kansei-core/src/renderers/mod.rs` — Export shared_layouts
- `rust/kansei-core/src/renderers/renderer.rs` — Add shared bind groups, camera buffers, draw loop
- `rust/kansei-core/src/materials/material.rs` — Accept external shared layouts instead of creating its own
- `rust/kansei-core/src/objects/renderable.rs` — Own Geometry + Material directly
- `rust/kansei-core/src/objects/scene.rs` — Update for new Renderable structure
- `rust/kansei-core/src/objects/mod.rs` — Update exports
- `rust/kansei-core/src/cameras/camera.rs` — No changes needed (matrices are uploaded by Renderer)
- `rust/kansei-core/src/lib.rs` — Add shaders module if needed

---

### Task 1: Create shared bind group layouts module

**Files:**
- Create: `rust/kansei-core/src/renderers/shared_layouts.rs`
- Modify: `rust/kansei-core/src/renderers/mod.rs`

Currently, Material.ensure_shared() creates the mesh BGL and camera BGL internally (material.rs:93-137). This means every material creates its own copy. These layouts need to live in the Renderer so all materials share the same layouts, and the Renderer can create bind groups against them.

- [ ] **Step 1: Create shared_layouts.rs**

```rust
// rust/kansei-core/src/renderers/shared_layouts.rs

/// Bind group slot assignments matching the TS engine.
/// Group 0: Material-specific (owned by Material)
/// Group 1: Mesh transforms — dynamic offsets into bulk matrix buffers (owned by Renderer)
/// Group 2: Camera — view + projection matrices (owned by Renderer)
/// Group 3: Shadows (future — Plan 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum BindGroupSlot {
    Material = 0,
    Mesh = 1,
    Camera = 2,
    Shadow = 3,
}

/// Shared bind group layouts created once by the Renderer and passed to Materials.
pub struct SharedLayouts {
    pub mesh_bgl: wgpu::BindGroupLayout,
    pub camera_bgl: wgpu::BindGroupLayout,
}

impl SharedLayouts {
    pub fn new(device: &wgpu::Device) -> Self {
        // Group 1: mesh transforms (normal_matrix + world_matrix with dynamic offsets)
        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared/MeshBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Group 2: camera (view_matrix + projection_matrix, no dynamic offset)
        let camera_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared/CameraBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        Self { mesh_bgl, camera_bgl }
    }
}
```

- [ ] **Step 2: Update renderers/mod.rs to export SharedLayouts**

Replace the contents of `rust/kansei-core/src/renderers/mod.rs` with:

```rust
mod renderer;
mod gbuffer;
mod compute_batch;
mod shared_layouts;

pub use renderer::{Renderer, RendererConfig};
pub use gbuffer::GBuffer;
pub use compute_batch::ComputeBatch;
pub use shared_layouts::{SharedLayouts, BindGroupSlot};
```

- [ ] **Step 3: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5`
Expected: compiles with existing warnings only (no new errors)

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-core/src/renderers/shared_layouts.rs rust/kansei-core/src/renderers/mod.rs
git commit -m "feat: add SharedLayouts for mesh/camera bind groups"
```

---

### Task 2: Update Material to accept shared layouts

**Files:**
- Modify: `rust/kansei-core/src/materials/material.rs`

Currently Material.ensure_shared() creates its own mesh BGL and camera BGL (lines 93-143). Change it to accept a `&SharedLayouts` from the Renderer so all materials share the same bind group layouts.

- [ ] **Step 1: Modify Material to use SharedLayouts**

Replace the full contents of `rust/kansei-core/src/materials/material.rs` with:

```rust
use std::collections::HashMap;
use super::binding::{Binding, BindGroupBuilder, BindingResource};
use crate::renderers::SharedLayouts;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CullMode {
    None,
    Front,
    Back,
}

impl CullMode {
    fn to_wgpu(self) -> Option<wgpu::Face> {
        match self {
            CullMode::None => None,
            CullMode::Front => Some(wgpu::Face::Front),
            CullMode::Back => Some(wgpu::Face::Back),
        }
    }
}

/// Configuration for a render material.
pub struct MaterialOptions {
    pub transparent: bool,
    pub depth_write: Option<bool>,
    pub depth_compare: wgpu::CompareFunction,
    pub cull_mode: CullMode,
    pub topology: wgpu::PrimitiveTopology,
}

impl Default for MaterialOptions {
    fn default() -> Self {
        Self {
            transparent: false,
            depth_write: None,
            depth_compare: wgpu::CompareFunction::Less,
            cull_mode: CullMode::Back,
            topology: wgpu::PrimitiveTopology::TriangleList,
        }
    }
}

/// Pipeline cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct PipelineKey {
    color_formats: Vec<wgpu::TextureFormat>,
    depth_format: wgpu::TextureFormat,
    sample_count: u32,
}

/// A render material — shader + pipeline cache + bind group.
pub struct Material {
    pub label: String,
    pub shader_code: String,
    pub options: MaterialOptions,
    pub bindings: Vec<Binding>,
    shader_module: Option<wgpu::ShaderModule>,
    material_bgl: Option<wgpu::BindGroupLayout>,
    pipeline_layout: Option<wgpu::PipelineLayout>,
    pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
    bind_group: Option<wgpu::BindGroup>,
    pub initialized: bool,
}

impl Material {
    pub fn new(label: &str, shader_code: &str, bindings: Vec<Binding>, options: MaterialOptions) -> Self {
        Self {
            label: label.to_string(),
            shader_code: shader_code.to_string(),
            options,
            bindings,
            shader_module: None,
            material_bgl: None,
            pipeline_layout: None,
            pipeline_cache: HashMap::new(),
            bind_group: None,
            initialized: false,
        }
    }

    /// Ensure shader module and layouts are created once.
    /// Uses shared layouts from the Renderer instead of creating its own.
    fn ensure_shared(&mut self, device: &wgpu::Device, shared: &SharedLayouts) {
        if self.shader_module.is_some() {
            return;
        }

        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&format!("{}/Shader", self.label)),
            source: wgpu::ShaderSource::Wgsl(self.shader_code.as_str().into()),
        });

        let material_bgl = BindGroupBuilder::create_layout(
            device,
            &format!("{}/MaterialBGL", self.label),
            &self.bindings,
        );

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/PipelineLayout", self.label)),
            bind_group_layouts: &[&material_bgl, &shared.mesh_bgl, &shared.camera_bgl],
            push_constant_ranges: &[],
        });

        self.shader_module = Some(module);
        self.material_bgl = Some(material_bgl);
        self.pipeline_layout = Some(pipeline_layout);
    }

    /// Get or create a pipeline for the given render target config.
    pub fn get_pipeline(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        color_formats: &[wgpu::TextureFormat],
        depth_format: wgpu::TextureFormat,
        sample_count: u32,
    ) -> &wgpu::RenderPipeline {
        self.ensure_shared(device, shared);

        let key = PipelineKey {
            color_formats: color_formats.to_vec(),
            depth_format,
            sample_count,
        };

        if !self.pipeline_cache.contains_key(&key) {
            let targets: Vec<Option<wgpu::ColorTargetState>> = color_formats
                .iter()
                .enumerate()
                .map(|(i, fmt)| {
                    let mut state = wgpu::ColorTargetState {
                        format: *fmt,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    };
                    if i == 0 && self.options.transparent {
                        state.blend = Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                operation: wgpu::BlendOperation::Add,
                                src_factor: wgpu::BlendFactor::SrcAlpha,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            },
                            alpha: wgpu::BlendComponent {
                                operation: wgpu::BlendOperation::Add,
                                src_factor: wgpu::BlendFactor::One,
                                dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            },
                        });
                    }
                    Some(state)
                })
                .collect();

            let depth_write = self.options.depth_write.unwrap_or(!self.options.transparent);
            let cull_mode = if self.options.transparent {
                None
            } else {
                self.options.cull_mode.to_wgpu()
            };

            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(&format!("{}/Pipeline", self.label)),
                layout: self.pipeline_layout.as_ref(),
                vertex: wgpu::VertexState {
                    module: self.shader_module.as_ref().unwrap(),
                    entry_point: Some("vertex_main"),
                    buffers: vertex_layouts,
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: self.shader_module.as_ref().unwrap(),
                    entry_point: Some("fragment_main"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: self.options.topology,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: depth_format,
                    depth_write_enabled: depth_write,
                    depth_compare: self.options.depth_compare,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState {
                    count: sample_count,
                    ..Default::default()
                },
                multiview: None,
                cache: None,
            });

            self.pipeline_cache.insert(key.clone(), pipeline);
        }

        self.pipeline_cache.get(&key).unwrap()
    }

    /// Create or update the material's bind group (group 0).
    pub fn create_bind_group(
        &mut self,
        device: &wgpu::Device,
        shared: &SharedLayouts,
        resources: &[(u32, BindingResource)],
    ) {
        self.ensure_shared(device, shared);
        self.bind_group = Some(BindGroupBuilder::create_bind_group(
            device,
            &format!("{}/BindGroup", self.label),
            self.material_bgl.as_ref().unwrap(),
            resources,
        ));
        self.initialized = true;
    }

    pub fn bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.bind_group.as_ref()
    }

    pub fn material_bgl(&self) -> Option<&wgpu::BindGroupLayout> {
        self.material_bgl.as_ref()
    }
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5`
Expected: may show errors from renderer.rs which still references old Material API — that's fine, we'll fix it in Task 5.

- [ ] **Step 3: Commit**

```bash
git add rust/kansei-core/src/materials/material.rs
git commit -m "refactor: Material uses SharedLayouts from Renderer"
```

---

### Task 3: Restructure Renderable to own Geometry + Material

**Files:**
- Modify: `rust/kansei-core/src/objects/renderable.rs`
- Modify: `rust/kansei-core/src/objects/scene.rs`
- Modify: `rust/kansei-core/src/objects/mod.rs`

Currently Renderable holds `geometry_id: Option<usize>` and `material_id: Option<usize>` — indices into hypothetical pools that don't exist. Change it to own Geometry and Material directly.

- [ ] **Step 1: Rewrite renderable.rs**

```rust
// rust/kansei-core/src/objects/renderable.rs
use super::Object3D;
use crate::geometries::Geometry;
use crate::materials::Material;

/// A renderable object — owns geometry + material + transform.
pub struct Renderable {
    pub object: Object3D,
    pub geometry: Geometry,
    pub material: Material,
    pub instance_count: u32,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub render_order: i32,
    pub visible: bool,
    pub material_dirty: bool,
}

impl Renderable {
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Self {
            object: Object3D::new(),
            geometry,
            material,
            instance_count: 1,
            cast_shadow: true,
            receive_shadow: true,
            render_order: 0,
            visible: true,
            material_dirty: true,
        }
    }

    /// Whether this renderable uses transparency.
    pub fn is_transparent(&self) -> bool {
        self.material.options.transparent
    }
}

impl std::ops::Deref for Renderable {
    type Target = Object3D;
    fn deref(&self) -> &Object3D {
        &self.object
    }
}

impl std::ops::DerefMut for Renderable {
    fn deref_mut(&mut self) -> &mut Object3D {
        &mut self.object
    }
}
```

- [ ] **Step 2: Update scene.rs for new Renderable**

Replace the full contents of `rust/kansei-core/src/objects/scene.rs` with:

```rust
use super::Renderable;
use crate::math::Vec3;

/// The scene graph — holds all renderables, sorts opaque/transparent.
pub struct Scene {
    pub position: Vec3,
    renderables: Vec<Renderable>,
    opaque_order: Vec<usize>,
    transparent_order: Vec<usize>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            renderables: Vec::new(),
            opaque_order: Vec::new(),
            transparent_order: Vec::new(),
        }
    }

    /// Add a renderable to the scene. Returns its index.
    pub fn add(&mut self, renderable: Renderable) -> usize {
        let idx = self.renderables.len();
        self.renderables.push(renderable);
        idx
    }

    pub fn get(&self, index: usize) -> Option<&Renderable> {
        self.renderables.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Renderable> {
        self.renderables.get_mut(index)
    }

    /// Sort renderables into opaque and transparent (back-to-front).
    pub fn prepare(&mut self, camera_pos: &Vec3) {
        self.opaque_order.clear();
        self.transparent_order.clear();

        for (i, r) in self.renderables.iter().enumerate() {
            if !r.visible {
                continue;
            }
            if r.is_transparent() {
                self.transparent_order.push(i);
            } else {
                self.opaque_order.push(i);
            }
        }

        let renderables = &self.renderables;
        self.transparent_order.sort_by(|&a, &b| {
            let ra = &renderables[a];
            let rb = &renderables[b];
            if ra.render_order != rb.render_order {
                return ra.render_order.cmp(&rb.render_order);
            }
            let da = ra.position.distance_to_squared(camera_pos);
            let db = rb.position.distance_to_squared(camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Ordered indices: opaque first, then transparent.
    pub fn ordered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.opaque_order
            .iter()
            .copied()
            .chain(self.transparent_order.iter().copied())
    }

    pub fn renderables(&self) -> &[Renderable] {
        &self.renderables
    }

    pub fn renderables_mut(&mut self) -> &mut [Renderable] {
        &mut self.renderables
    }

    pub fn len(&self) -> usize {
        self.renderables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.renderables.is_empty()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 3: Update objects/mod.rs exports**

Replace `rust/kansei-core/src/objects/mod.rs` with:

```rust
mod object3d;
mod renderable;
mod scene;

pub use object3d::Object3D;
pub use renderable::Renderable;
pub use scene::Scene;
```

- [ ] **Step 4: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -10`
Expected: errors from renderer.rs (references old Renderable fields) — will be fixed in Task 5.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/objects/
git commit -m "refactor: Renderable owns Geometry + Material directly"
```

---

### Task 4: Create basic.wgsl shader

**Files:**
- Create: `rust/kansei-core/src/shaders/basic.wgsl`

This shader uses the 3-group bind layout (material, mesh, camera) and the standard Vertex layout (position vec4, normal vec3, uv vec2). Simple directional lighting.

- [ ] **Step 1: Create shaders directory and basic.wgsl**

```bash
mkdir -p /Users/felixmartinez/Documents/dev/kansei/rust/kansei-core/src/shaders
```

Write `rust/kansei-core/src/shaders/basic.wgsl`:

```wgsl
// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Mesh (dynamic offset) ──
@group(1) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// ── Group 2: Camera ──
@group(2) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

// ── Vertex I/O ──
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
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
git add rust/kansei-core/src/shaders/basic.wgsl
git commit -m "feat: add basic.wgsl lit material shader"
```

---

### Task 5: Implement Renderer draw loop with shared bind groups

**Files:**
- Modify: `rust/kansei-core/src/renderers/renderer.rs`

This is the main task. The Renderer needs to:
1. Create SharedLayouts on init
2. Create camera uniform buffers + bind group (group 2)
3. Create shared mesh bind group (group 1) pointing at matrix buffers
4. Initialize materials and geometries that aren't yet initialized
5. Issue actual draw calls instead of the TODO comments

- [ ] **Step 1: Rewrite renderer.rs with full draw loop**

Replace the full contents of `rust/kansei-core/src/renderers/renderer.rs` with:

```rust
use crate::math::Vec4;
use crate::cameras::Camera;
use crate::geometries::Vertex;
use crate::objects::Scene;
use super::gbuffer::GBuffer;
use super::shared_layouts::SharedLayouts;

/// Core WebGPU renderer configuration.
pub struct RendererConfig {
    pub width: u32,
    pub height: u32,
    pub device_pixel_ratio: f32,
    pub sample_count: u32,
    pub clear_color: Vec4,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            width: 800,
            height: 600,
            device_pixel_ratio: 1.0,
            sample_count: 4,
            clear_color: Vec4::new(0.0, 0.0, 0.0, 1.0),
        }
    }
}

/// The main GPU renderer.
pub struct Renderer {
    pub config: RendererConfig,
    device: Option<wgpu::Device>,
    queue: Option<wgpu::Queue>,
    surface: Option<wgpu::Surface<'static>>,
    surface_config: Option<wgpu::SurfaceConfiguration>,
    presentation_format: wgpu::TextureFormat,
    // Depth
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    // MSAA
    msaa_texture: Option<wgpu::Texture>,
    msaa_view: Option<wgpu::TextureView>,
    // Shared layouts
    shared_layouts: Option<SharedLayouts>,
    // Per-object matrix buffers (dynamic offset uniform)
    world_matrices_buf: Option<wgpu::Buffer>,
    normal_matrices_buf: Option<wgpu::Buffer>,
    world_matrices_staging: Vec<f32>,
    normal_matrices_staging: Vec<f32>,
    matrix_alignment: u32,
    last_object_count: usize,
    // Shared bind groups
    mesh_bind_group: Option<wgpu::BindGroup>,
    // Camera uniform buffers + bind group
    camera_view_buf: Option<wgpu::Buffer>,
    camera_proj_buf: Option<wgpu::Buffer>,
    camera_bind_group: Option<wgpu::BindGroup>,
}

impl Renderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            device: None,
            queue: None,
            surface: None,
            surface_config: None,
            presentation_format: wgpu::TextureFormat::Bgra8Unorm,
            depth_texture: None,
            depth_view: None,
            msaa_texture: None,
            msaa_view: None,
            shared_layouts: None,
            world_matrices_buf: None,
            normal_matrices_buf: None,
            world_matrices_staging: Vec::new(),
            normal_matrices_staging: Vec::new(),
            matrix_alignment: 256,
            last_object_count: 0,
            mesh_bind_group: None,
            camera_view_buf: None,
            camera_proj_buf: None,
            camera_bind_group: None,
        }
    }

    /// Initialize with a wgpu surface (from winit window or web canvas).
    pub async fn initialize(&mut self, surface: wgpu::Surface<'static>, adapter: &wgpu::Adapter) {
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Kansei Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .expect("Failed to create device");

        self.matrix_alignment = device.limits().min_uniform_buffer_offset_alignment;

        let surface_caps = surface.get_capabilities(adapter);
        let format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: self.config.width,
            height: self.config.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        self.presentation_format = format;

        // Create shared layouts
        self.shared_layouts = Some(SharedLayouts::new(&device));

        // Create camera uniform buffers (64 bytes each = mat4x4)
        self.camera_view_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/CameraView"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.camera_proj_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/CameraProj"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Create camera bind group
        let shared = self.shared_layouts.as_ref().unwrap();
        self.camera_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/CameraBG"),
            layout: &shared.camera_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_view_buf.as_ref().unwrap().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.camera_proj_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));

        self._create_depth_texture(&device);

        self.device = Some(device);
        self.queue = Some(queue);
        self.surface = Some(surface);
        self.surface_config = Some(surface_config);
    }

    pub fn device(&self) -> &wgpu::Device {
        self.device.as_ref().expect("Renderer not initialized")
    }

    pub fn queue(&self) -> &wgpu::Queue {
        self.queue.as_ref().expect("Renderer not initialized")
    }

    pub fn presentation_format(&self) -> wgpu::TextureFormat {
        self.presentation_format
    }

    pub fn surface(&self) -> Option<&wgpu::Surface<'static>> {
        self.surface.as_ref()
    }

    pub fn shared_layouts(&self) -> &SharedLayouts {
        self.shared_layouts.as_ref().expect("Renderer not initialized")
    }

    pub fn width(&self) -> u32 {
        self.config.width
    }
    pub fn height(&self) -> u32 {
        self.config.height
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.config.width = width;
        self.config.height = height;
        if self.device.is_none() {
            return;
        }
        if let (Some(ref surface), Some(ref mut config)) = (&self.surface, &mut self.surface_config)
        {
            config.width = width;
            config.height = height;
            surface.configure(self.device.as_ref().unwrap(), config);
        }
        self._recreate_size_dependent();
    }

    fn _recreate_size_dependent(&mut self) {
        let w = self.config.width;
        let h = self.config.height;
        let sc = self.config.sample_count;
        let fmt = self.presentation_format;
        let device = self.device.as_ref().unwrap();

        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: sc,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if sc > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: sc,
                dimension: wgpu::TextureDimension::D2,
                format: fmt,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    fn _create_depth_texture(&mut self, device: &wgpu::Device) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/Depth"),
            size: wgpu::Extent3d {
                width: self.config.width,
                height: self.config.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.config.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        if self.config.sample_count > 1 {
            let msaa = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Renderer/MSAA"),
                size: wgpu::Extent3d {
                    width: self.config.width,
                    height: self.config.height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: self.config.sample_count,
                dimension: wgpu::TextureDimension::D2,
                format: self.presentation_format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.msaa_view = Some(msaa.create_view(&Default::default()));
            self.msaa_texture = Some(msaa);
        }
    }

    /// Ensure shared per-object matrix buffers are large enough.
    /// Also rebuilds the mesh bind group when buffers are recreated.
    fn _ensure_matrix_buffers(&mut self, count: usize) {
        if count == 0 {
            return;
        }
        if count <= self.last_object_count && self.world_matrices_buf.is_some() {
            return;
        }
        let device = self.device.as_ref().unwrap();
        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;
        let total_floats = count * floats_per_slot;
        let total_bytes = (total_floats * 4) as u64;

        self.world_matrices_staging.resize(total_floats, 0.0);
        self.normal_matrices_staging.resize(total_floats, 0.0);

        self.world_matrices_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/WorldMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
        self.normal_matrices_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/NormalMatrices"),
            size: total_bytes,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // Rebuild mesh bind group with new buffers
        let shared = self.shared_layouts.as_ref().unwrap();
        self.mesh_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/MeshBG"),
            layout: &shared.mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.normal_matrices_buf.as_ref().unwrap(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(64), // mat4x4 = 64 bytes
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: self.world_matrices_buf.as_ref().unwrap(),
                        offset: 0,
                        size: std::num::NonZeroU64::new(64),
                    }),
                },
            ],
        }));

        self.last_object_count = count;
    }

    /// Upload camera matrices + all scene object matrices to GPU.
    fn upload_all(&mut self, scene: &Scene, camera: &Camera) {
        let queue = self.queue.as_ref().unwrap();

        // Upload camera matrices
        if let Some(ref buf) = self.camera_view_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(camera.view_matrix.as_slice()));
        }
        if let Some(ref buf) = self.camera_proj_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(camera.projection_matrix.as_slice()));
        }

        // Upload per-object matrices
        let count = scene.len();
        self._ensure_matrix_buffers(count);

        let alignment = self.matrix_alignment as usize;
        let floats_per_slot = alignment / 4;

        for (i, idx) in scene.ordered_indices().enumerate() {
            if let Some(renderable) = scene.get(idx) {
                let offset = i * floats_per_slot;
                self.world_matrices_staging[offset..offset + 16]
                    .copy_from_slice(renderable.world_matrix.as_slice());
                self.normal_matrices_staging[offset..offset + 16]
                    .copy_from_slice(renderable.normal_matrix.as_slice());
            }
        }

        if count > 0 {
            if let Some(ref buf) = self.world_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.world_matrices_staging));
            }
            if let Some(ref buf) = self.normal_matrices_buf {
                queue.write_buffer(buf, 0, bytemuck::cast_slice(&self.normal_matrices_staging));
            }
        }
    }

    /// Render the scene to the canvas surface.
    pub fn render(&mut self, scene: &mut Scene, camera: &mut Camera) {
        // Phase 0: Update transforms
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Initialize any uninitialized geometries + ensure pipelines
        let device = self.device.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        let format = self.presentation_format;
        let sample_count = self.config.sample_count;
        let depth_format = wgpu::TextureFormat::Depth24Plus;

        for r in scene.renderables_mut() {
            if !r.geometry.initialized {
                r.geometry.initialize(device);
            }
            // Pre-warm pipeline cache
            r.material.get_pipeline(
                device,
                shared,
                &[Vertex::LAYOUT],
                &[format],
                depth_format,
                sample_count,
            );
        }

        // Phase 1: Upload matrices
        self.upload_all(scene, camera);

        // Phase 2+3: Record and execute draw commands
        let surface = self.surface.as_ref().unwrap();
        let output = surface
            .get_current_texture()
            .expect("Failed to get surface texture");
        let canvas_view = output.texture.create_view(&Default::default());

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/RenderEncoder"),
        });

        let cc = &self.config.clear_color;
        {
            let color_view = if self.config.sample_count > 1 {
                self.msaa_view.as_ref().unwrap()
            } else {
                &canvas_view
            };
            let resolve = if self.config.sample_count > 1 {
                Some(canvas_view.as_ref())
            } else {
                None
            };

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Renderer/MainPass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: color_view,
                    resolve_target: resolve,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: cc.x as f64,
                            g: cc.y as f64,
                            b: cc.z as f64,
                            a: cc.w as f64,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().unwrap(),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });

            // Set camera bind group (group 2) — shared across all objects
            pass.set_bind_group(2, self.camera_bind_group.as_ref().unwrap(), &[]);

            let alignment = self.matrix_alignment;

            // Draw each renderable
            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = &scene.renderables()[scene_idx];
                if !r.visible || !r.geometry.initialized {
                    continue;
                }

                let pipeline = r.material.pipeline_cache
                    .values()
                    .next();
                let pipeline = match pipeline {
                    Some(p) => p,
                    None => continue,
                };

                pass.set_pipeline(pipeline);

                // Group 0: material bind group
                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }

                // Group 1: mesh bind group with dynamic offsets
                let offset = (draw_idx as u32) * alignment;
                pass.set_bind_group(
                    1,
                    self.mesh_bind_group.as_ref().unwrap(),
                    &[offset, offset],
                );

                // Set vertex + index buffers
                pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                pass.set_index_buffer(
                    r.geometry.index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
            }
        }

        self.queue
            .as_ref()
            .unwrap()
            .submit(std::iter::once(encoder.finish()));
        output.present();
    }

    /// Render scene into a GBuffer for post-processing.
    pub fn render_to_gbuffer(
        &mut self,
        scene: &mut Scene,
        camera: &mut Camera,
        gbuffer: &GBuffer,
    ) {
        camera.update_view_matrix();
        scene.prepare(camera.position());

        // Initialize geometries + ensure pipelines for GBuffer formats
        let device = self.device.as_ref().unwrap();
        let shared = self.shared_layouts.as_ref().unwrap();
        let depth_format = wgpu::TextureFormat::Depth32Float;

        for r in scene.renderables_mut() {
            if !r.geometry.initialized {
                r.geometry.initialize(device);
            }
            r.material.get_pipeline(
                device,
                shared,
                &[Vertex::LAYOUT],
                &GBuffer::MRT_FORMATS,
                depth_format,
                gbuffer.sample_count,
            );
        }

        self.upload_all(scene, camera);

        let device = self.device.as_ref().unwrap();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Renderer/GBufferEncoder"),
        });

        let cc = &self.config.clear_color;
        let clear = wgpu::Color {
            r: cc.x as f64,
            g: cc.y as f64,
            b: cc.z as f64,
            a: cc.w as f64,
        };
        let black = wgpu::Color { r: 0.0, g: 0.0, b: 0.0, a: 0.0 };

        {
            let mut pass = if gbuffer.sample_count > 1 {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Renderer/GBufferPass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.color_msaa_view.as_ref().unwrap(),
                            resolve_target: Some(&gbuffer.color_view),
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Discard },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.emissive_msaa_view.as_ref().unwrap(),
                            resolve_target: Some(&gbuffer.emissive_view),
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.normal_msaa_view.as_ref().unwrap(),
                            resolve_target: Some(&gbuffer.normal_view),
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: gbuffer.albedo_msaa_view.as_ref().unwrap(),
                            resolve_target: Some(&gbuffer.albedo_view),
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Discard },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: gbuffer.depth_msaa_view.as_ref().unwrap(),
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                })
            } else {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Renderer/GBufferPass"),
                    color_attachments: &[
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gbuffer.color_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(clear), store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gbuffer.emissive_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gbuffer.normal_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                        }),
                        Some(wgpu::RenderPassColorAttachment {
                            view: &gbuffer.albedo_view,
                            resolve_target: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(black), store: wgpu::StoreOp::Store },
                        }),
                    ],
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &gbuffer.depth_view,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                })
            };

            // Draw with same logic as canvas path
            pass.set_bind_group(2, self.camera_bind_group.as_ref().unwrap(), &[]);

            let alignment = self.matrix_alignment;

            for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                let r = &scene.renderables()[scene_idx];
                if !r.visible || !r.geometry.initialized {
                    continue;
                }

                // For GBuffer, use the MRT pipeline
                let gbuffer_key = super::renderer_internal::gbuffer_pipeline_key(gbuffer);
                let pipeline = r.material.pipeline_cache.get(&gbuffer_key);
                let pipeline = match pipeline {
                    Some(p) => p,
                    None => continue,
                };

                pass.set_pipeline(pipeline);

                if let Some(bg) = r.material.bind_group() {
                    pass.set_bind_group(0, bg, &[]);
                }

                let offset = (draw_idx as u32) * alignment;
                pass.set_bind_group(1, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                pass.set_index_buffer(
                    r.geometry.index_buffer.as_ref().unwrap().slice(..),
                    wgpu::IndexFormat::Uint32,
                );

                pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
            }
        }

        self.queue
            .as_ref()
            .unwrap()
            .submit(std::iter::once(encoder.finish()));
    }

    /// Keep the old upload_matrices for backward compat with fluid example.
    pub fn upload_matrices(&mut self, scene: &Scene, camera: &Camera) {
        self.upload_all(scene, camera);
    }
}
```

Note: The `render_to_gbuffer` references `renderer_internal::gbuffer_pipeline_key` which doesn't exist yet. For this first plan, comment out or simplify the GBuffer draw loop to just clear — we'll fill it in Plan 2. For the canvas `render()` path, the pipeline lookup uses `.pipeline_cache.values().next()` which gets the one pipeline we pre-warmed. This is correct for now since we only have one render target config per material.

- [ ] **Step 2: Fix the GBuffer path to compile**

The GBuffer draw loop references a function that doesn't exist. Replace the GBuffer draw section (everything inside the `{ let mut pass = ... }` block after creating the pass) with a simpler version that just clears for now:

In the `render_to_gbuffer` method, replace the draw loop section (after creating the pass) with:

```rust
            // TODO: GBuffer draw loop — same pattern as canvas path but with MRT pipeline key
            // For now, just clear. Full GBuffer drawing comes in Plan 2.
            let _ = &pass; // suppress unused warning
```

Actually, it's cleaner to just remove the `render_to_gbuffer` draw calls and leave the clear-only version (which is what it was before). The canvas `render()` method is the one we need working now. Let's keep `render_to_gbuffer` as clear-only for this plan.

- [ ] **Step 3: Verify it compiles**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -20`

Fix any compilation errors. Common issues:
- `pipeline_cache` field visibility: the Material's `pipeline_cache` field is private. Add `pub(crate)` to it.
- Lifetime issues with render pass and `canvas_view` — ensure `canvas_view` lives long enough.
- `resolve_target: Some(canvas_view.as_ref())` should be `Some(&canvas_view)`.
- `draw_indexed` signature: `pass.draw_indexed(0..count, instance_count, 0, 0, 0)` — the second arg is `instance_count: u32`, so it should be `pass.draw_indexed(0..r.geometry.index_count(), 0..r.instance_count, 0, 0, 0)` or use the 5-arg form depending on wgpu version.

The wgpu 24 `draw_indexed` signature is: `draw_indexed(indices: Range<u32>, base_vertex: i32, instances: Range<u32>)`. So the correct call is:

```rust
pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
```

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-core/src/renderers/renderer.rs
git commit -m "feat: implement Renderer draw loop with shared bind groups"
```

---

### Task 6: Fix compilation — make pipeline_cache accessible and resolve API issues

**Files:**
- Modify: `rust/kansei-core/src/materials/material.rs` (make `pipeline_cache` pub(crate))
- Modify: `rust/kansei-core/src/renderers/renderer.rs` (fix any API mismatches)

This task is about getting the whole crate to compile cleanly after the previous changes.

- [ ] **Step 1: Make pipeline_cache visible to renderer**

In `rust/kansei-core/src/materials/material.rs`, change:

```rust
    pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
```

to:

```rust
    pub(crate) pipeline_cache: HashMap<PipelineKey, wgpu::RenderPipeline>,
```

Also make `PipelineKey` pub(crate) since the renderer needs to reference it:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct PipelineKey {
    pub(crate) color_formats: Vec<wgpu::TextureFormat>,
    pub(crate) depth_format: wgpu::TextureFormat,
    pub(crate) sample_count: u32,
}
```

- [ ] **Step 2: Fix the canvas render pipeline lookup**

In `renderer.rs`, the draw loop gets the pipeline via `.pipeline_cache.values().next()`. This is fragile. Instead, construct the key and look it up properly. In the draw loop inside `render()`, replace:

```rust
                let pipeline = r.material.pipeline_cache
                    .values()
                    .next();
                let pipeline = match pipeline {
                    Some(p) => p,
                    None => continue,
                };
```

with:

```rust
                use crate::materials::material::PipelineKey;
                let key = PipelineKey {
                    color_formats: vec![format],
                    depth_format,
                    sample_count,
                };
                let pipeline = match r.material.pipeline_cache.get(&key) {
                    Some(p) => p,
                    None => continue,
                };
```

And capture `format`, `depth_format`, `sample_count` before the render pass:

```rust
        let format = self.presentation_format;
        let sample_count = self.config.sample_count;
        let depth_format = wgpu::TextureFormat::Depth24Plus;
```

(These should already be in scope from the pre-warm phase.)

- [ ] **Step 3: Fix draw_indexed call signature for wgpu 24**

In wgpu 24, `draw_indexed` takes `(indices: Range<u32>, base_vertex: i32, instances: Range<u32>)`. Fix the call:

```rust
pass.draw_indexed(0..r.geometry.index_count(), 0, 0..r.instance_count);
```

- [ ] **Step 4: Fix render_to_gbuffer to compile without draw loop**

Remove the draw loop from `render_to_gbuffer`, keeping only the pass creation (clear). The block after creating the pass should just drop it:

```rust
            // GBuffer draw loop deferred to Plan 2
        }
```

- [ ] **Step 5: Full compile check**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1`

Iterate until clean (warnings are OK, errors are not). Common remaining issues:
- The `materials::material::PipelineKey` needs to be importable from renderer — ensure the `pub(crate)` visibility is set
- `resolve_target` type: should be `Some(&canvas_view)` not `Some(canvas_view.as_ref())`

- [ ] **Step 6: Commit**

```bash
git add rust/kansei-core/src/materials/material.rs rust/kansei-core/src/renderers/renderer.rs
git commit -m "fix: resolve compilation issues in Material + Renderer integration"
```

---

### Task 7: Create spinning box example

**Files:**
- Create: `rust/kansei-native/examples/spinning_box.rs`

This example validates the full pipeline: creates a Scene with a BoxGeometry + basic material, rotates it each frame, and renders via Renderer.render().

- [ ] **Step 1: Write the example**

```rust
// rust/kansei-native/examples/spinning_box.rs
use std::sync::Arc;
use std::time::Instant;

use kansei_core::math::{Vec3, Vec4};
use kansei_core::cameras::Camera;
use kansei_core::geometries::BoxGeometry;
use kansei_core::materials::{Material, MaterialOptions, Binding};
use kansei_core::materials::binding::BindingResource;
use kansei_core::objects::{Renderable, Scene};
use kansei_core::renderers::{Renderer, RendererConfig};

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowId};

const BASIC_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic.wgsl");

struct App {
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    scene: Scene,
    camera: Camera,
    start_time: Instant,
    color_buf: Option<wgpu::Buffer>,
}

impl App {
    fn new() -> Self {
        Self {
            window: None,
            renderer: None,
            scene: Scene::new(),
            camera: Camera::new(45.0, 0.1, 100.0, 1.0),
            start_time: Instant::now(),
            color_buf: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let window = Arc::new(
            el.create_window(
                Window::default_attributes()
                    .with_title("Kansei — Spinning Box")
                    .with_inner_size(winit::dpi::LogicalSize::new(1280, 720)),
            )
            .unwrap(),
        );
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&Default::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            compatible_surface: Some(&surface),
            ..Default::default()
        }))
        .unwrap();

        let mut renderer = Renderer::new(RendererConfig {
            width: size.width,
            height: size.height,
            sample_count: 4,
            clear_color: Vec4::new(0.05, 0.05, 0.08, 1.0),
            ..Default::default()
        });
        pollster::block_on(renderer.initialize(surface, &adapter));

        let device = renderer.device();
        let shared = renderer.shared_layouts();

        // Create color uniform buffer for the material
        let color_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BoxColor"),
            size: 16, // vec4<f32>
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let color: [f32; 4] = [0.8, 0.3, 0.2, 1.0]; // reddish
        renderer
            .queue()
            .write_buffer(&color_buf, 0, bytemuck::cast_slice(&color));

        // Create material
        let mut material = Material::new(
            "BasicMaterial",
            BASIC_WGSL,
            vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
            MaterialOptions::default(),
        );
        material.create_bind_group(device, shared, &[(
            0,
            BindingResource::Buffer {
                buffer: &color_buf,
                offset: 0,
                size: None,
            },
        )]);

        // Create geometry
        let geometry = BoxGeometry::new(2.0, 2.0, 2.0);

        // Create renderable
        let renderable = Renderable::new(geometry, material);

        self.scene.add(renderable);

        // Position camera
        self.camera.set_position(0.0, 2.0, 6.0);
        self.camera.look_at(&Vec3::ZERO);
        self.camera.aspect = size.width as f32 / size.height as f32;
        self.camera.update_projection_matrix();

        self.color_buf = Some(color_buf);
        self.renderer = Some(renderer);
        self.window = Some(window);
        self.start_time = Instant::now();

        log::info!("Spinning box example initialized");
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
            WindowEvent::RedrawRequested => {
                let t = self.start_time.elapsed().as_secs_f32();

                // Rotate the box
                if let Some(r) = self.scene.get_mut(0) {
                    r.object.rotation.y = t * 0.8;
                    r.object.rotation.x = t * 0.3;
                    r.object.update_model_matrix();
                    r.object.update_normal_matrix(&self.camera.view_matrix);
                }

                if let Some(ref mut renderer) = self.renderer {
                    renderer.render(&mut self.scene, &mut self.camera);
                }

                if let Some(ref w) = self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    env_logger::init();
    log::info!("Kansei — Spinning Box");
    let el = EventLoop::new().unwrap();
    el.set_control_flow(winit::event_loop::ControlFlow::Poll);
    el.run_app(&mut App::new()).unwrap();
}
```

- [ ] **Step 2: Run the example**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo run --example spinning_box 2>&1`

Expected: A window opens showing a spinning red box with simple directional lighting on a dark background. The box rotates smoothly.

If it doesn't compile, fix errors. Common issues:
- Import paths may need adjusting (e.g., `kansei_core::materials::binding::BindingResource` vs `kansei_core::materials::BindingResource`)
- The `BindingResource` type may need to be re-exported from `materials/mod.rs`
- `BoxGeometry::new()` returns a `Geometry` — check the existing API

- [ ] **Step 3: Fix any import issues**

If `BindingResource` isn't exported from the `materials` module, update `rust/kansei-core/src/materials/mod.rs`:

```rust
mod binding;
mod compute;
mod material;

pub use binding::{Binding, BindingResource, BindGroupBuilder};
pub use compute::ComputePass;
pub use material::{Material, MaterialOptions, CullMode};
```

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-native/examples/spinning_box.rs rust/kansei-core/src/shaders/basic.wgsl
git commit -m "feat: add spinning box example — validates full render pipeline"
```

---

### Task 8: Fix remaining compilation issues and verify end-to-end

**Files:**
- Potentially: any file from previous tasks that needs adjustment

This task is for resolving any remaining issues discovered when building the full workspace (including kansei-native and kansei-wasm which depend on kansei-core).

- [ ] **Step 1: Build entire workspace**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo build --workspace 2>&1 | head -50`

This will catch issues in kansei-native (fluid_3d example) and kansei-wasm that use the old Material/Renderable API.

- [ ] **Step 2: Fix fluid_3d.rs if it uses old Material API**

The fluid_3d example creates its own pipelines and doesn't use Material at all, so it should still compile. But if it references `Renderable` or `Scene` with the old API (e.g., `Renderable::new()` without args), it needs updating.

Check: does fluid_3d.rs use `kansei_core::objects::{Scene, Renderable}`? If not (it uses its own custom structs), no changes needed.

- [ ] **Step 3: Fix kansei-wasm if it uses old APIs**

Same check for `rust/kansei-wasm/src/lib.rs`. The WASM module creates its own rendering pipeline and probably doesn't use `Renderable` or `Scene` from kansei-core.

- [ ] **Step 4: Run spinning_box example**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo run --example spinning_box 2>&1`

Verify: window opens, box renders, box spins.

- [ ] **Step 5: Commit final fixes**

```bash
git add -u
git commit -m "fix: resolve workspace compilation issues after core engine changes"
```

---

## Post-Plan Notes

### What this plan produces:
- A working `Renderer.render()` that draws objects through Scene → Camera → Material → Geometry
- Shared bind group layout architecture (mesh + camera) used by all materials
- A basic.wgsl shader with simple directional lighting
- A spinning box example that validates the full pipeline

### What comes next (Plan 2: Engine Features):
- InstancedGeometry (extra vertex buffers with instance step mode)
- Lights module (uniform packing, light data in shaders)
- Shadows (shadow maps, cubemap shadows, shadow bind group slot 3)
- Post-processing effects (SSAO, bloom, DoF, god rays, color grading, volumetric fog)
- Loaders (texture via `image` crate, glTF via `gltf` crate)
- Render bundles (cache draw commands)

### What comes next (Plan 3: Fluid Integration):
- Rewrite fluid_3d example to use engine Renderer + Scene + Material
- Rewrite kansei-wasm to use engine abstractions
- Performance benchmarking harness (TS vs Rust)
