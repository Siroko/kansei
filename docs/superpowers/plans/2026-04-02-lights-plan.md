# Lights Implementation Plan (Plan 2b)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add forward lighting with DirectionalLight, PointLight, and AreaLight support. Light data is packed into a GPU uniform buffer and made available to all fragment shaders via the camera bind group.

**Architecture:** Lights are added to the Scene. The Renderer collects them, packs up to 4 directional + 8 point + 4 area lights into a `LightUniforms` buffer (binding 2 of camera group). Shaders read this buffer for Blinn-Phong lighting. Camera bind group gains a third binding for lights. All existing shaders and pipeline layouts are updated.

**Tech Stack:** Rust, wgpu 24, bytemuck 1

---

## File Structure

### Files to create:
- `rust/kansei-core/src/lights/light_uniforms.rs` — LightUniforms GPU buffer packing
- `rust/kansei-core/src/shaders/lighting.wgsl` — Reusable lighting WGSL functions
- `rust/kansei-core/src/shaders/basic_lit.wgsl` — Basic material with multi-light support
- `rust/kansei-native/examples/lit_scene.rs` — Validation: scene with multiple lights

### Files to modify:
- `rust/kansei-core/src/lights/mod.rs` — Export Light enum, LightUniforms
- `rust/kansei-core/src/lights/directional.rs` — Add cast_shadow, volumetric fields
- `rust/kansei-core/src/lights/point.rs` — Add cast_shadow, volumetric fields
- `rust/kansei-core/src/objects/scene.rs` — Add lights collection
- `rust/kansei-core/src/renderers/shared_layouts.rs` — Add light uniform binding to camera BGL
- `rust/kansei-core/src/renderers/renderer.rs` — Create light buffer, upload per frame
- `rust/kansei-core/src/shaders/basic.wgsl` — Add empty light uniform declaration for compatibility
- `rust/kansei-core/src/shaders/basic_instanced.wgsl` — Same

---

### Task 1: Flesh out Light types and add Light enum

**Files:**
- Modify: `rust/kansei-core/src/lights/directional.rs`
- Modify: `rust/kansei-core/src/lights/point.rs`
- Create: `rust/kansei-core/src/lights/area.rs`
- Modify: `rust/kansei-core/src/lights/mod.rs`

- [ ] **Step 1: Update directional.rs**

Replace `rust/kansei-core/src/lights/directional.rs` with:

```rust
use crate::math::Vec3;

pub struct DirectionalLight {
    pub direction: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub cast_shadow: bool,
    pub volumetric: bool,
}

impl DirectionalLight {
    pub fn new(direction: Vec3, color: Vec3, intensity: f32) -> Self {
        Self { direction, color, intensity, cast_shadow: false, volumetric: true }
    }

    /// Color multiplied by intensity.
    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
```

- [ ] **Step 2: Update point.rs**

Replace `rust/kansei-core/src/lights/point.rs` with:

```rust
use crate::math::Vec3;

pub struct PointLight {
    pub position: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub radius: f32,
    pub cast_shadow: bool,
    pub volumetric: bool,
}

impl PointLight {
    pub fn new(position: Vec3, color: Vec3, intensity: f32, radius: f32) -> Self {
        Self { position, color, intensity, radius, cast_shadow: false, volumetric: true }
    }

    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
```

- [ ] **Step 3: Create area.rs**

Create `rust/kansei-core/src/lights/area.rs`:

```rust
use crate::math::Vec3;

pub struct AreaLight {
    pub position: Vec3,
    pub target: Vec3,
    pub color: Vec3,
    pub intensity: f32,
    pub width: f32,
    pub height: f32,
    pub radius: f32,
    pub cast_shadow: bool,
}

impl AreaLight {
    pub fn new(position: Vec3, target: Vec3, color: Vec3, intensity: f32, width: f32, height: f32) -> Self {
        Self {
            position, target, color, intensity,
            width, height, radius: 50.0, cast_shadow: false,
        }
    }

    pub fn direction(&self) -> Vec3 {
        (self.target - self.position).normalize()
    }

    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }
}
```

- [ ] **Step 4: Update lights/mod.rs with Light enum**

Replace `rust/kansei-core/src/lights/mod.rs` with:

```rust
mod directional;
mod point;
mod area;
mod light_uniforms;

pub use directional::DirectionalLight;
pub use point::PointLight;
pub use area::AreaLight;
pub use light_uniforms::LightUniforms;

/// A scene light — one of the supported light types.
pub enum Light {
    Directional(DirectionalLight),
    Point(PointLight),
    Area(AreaLight),
}
```

Note: `light_uniforms` module doesn't exist yet — it will be created in Task 2. For now, comment out the `mod light_uniforms` and `pub use light_uniforms::LightUniforms` lines, or create an empty file as a placeholder.

- [ ] **Step 5: Verify and commit**

Run: `cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5`

```bash
git add kansei-core/src/lights/
git commit -m "feat: flesh out Light types — DirectionalLight, PointLight, AreaLight + enum"
```

---

### Task 2: Create LightUniforms buffer packing

**Files:**
- Create: `rust/kansei-core/src/lights/light_uniforms.rs`

This struct packs scene lights into a fixed-size GPU uniform buffer. Max 4 directional + 8 point lights. Layout matches what the WGSL shader expects.

- [ ] **Step 1: Create light_uniforms.rs**

```rust
// rust/kansei-core/src/lights/light_uniforms.rs
use super::{Light, DirectionalLight, PointLight};

/// Maximum lights supported in the uniform buffer.
pub const MAX_DIRECTIONAL_LIGHTS: usize = 4;
pub const MAX_POINT_LIGHTS: usize = 8;

/// GPU layout for a single directional light (32 bytes, 8 floats).
/// Matches WGSL: struct DirLight { direction: vec3f, _pad0: f32, color: vec3f, intensity: f32 }
const DIR_LIGHT_FLOATS: usize = 8;

/// GPU layout for a single point light (32 bytes, 8 floats).
/// Matches WGSL: struct PointLight { position: vec3f, radius: f32, color: vec3f, intensity: f32 }
const POINT_LIGHT_FLOATS: usize = 8;

/// Total floats in the light uniform buffer:
/// 4 floats header (num_dir, num_point, pad, pad)
/// + MAX_DIRECTIONAL * 8
/// + MAX_POINT * 8
const HEADER_FLOATS: usize = 4;
pub const LIGHT_UNIFORM_FLOATS: usize = HEADER_FLOATS
    + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS
    + MAX_POINT_LIGHTS * POINT_LIGHT_FLOATS;
pub const LIGHT_UNIFORM_BYTES: usize = LIGHT_UNIFORM_FLOATS * 4;

/// Packs scene lights into a GPU-ready uniform buffer.
pub struct LightUniforms {
    pub data: Vec<f32>,
}

impl LightUniforms {
    pub fn new() -> Self {
        Self {
            data: vec![0.0; LIGHT_UNIFORM_FLOATS],
        }
    }

    /// Pack lights from the scene into the uniform buffer.
    pub fn pack(&mut self, lights: &[Light]) {
        self.data.fill(0.0);

        let mut num_dir: u32 = 0;
        let mut num_point: u32 = 0;

        let dir_offset = HEADER_FLOATS;
        let point_offset = HEADER_FLOATS + MAX_DIRECTIONAL_LIGHTS * DIR_LIGHT_FLOATS;

        for light in lights {
            match light {
                Light::Directional(dl) => {
                    if (num_dir as usize) < MAX_DIRECTIONAL_LIGHTS {
                        let i = num_dir as usize;
                        let o = dir_offset + i * DIR_LIGHT_FLOATS;
                        let ec = dl.effective_color();
                        self.data[o] = dl.direction.x;
                        self.data[o + 1] = dl.direction.y;
                        self.data[o + 2] = dl.direction.z;
                        // o+3 is padding
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = dl.intensity;
                        num_dir += 1;
                    }
                }
                Light::Point(pl) => {
                    if (num_point as usize) < MAX_POINT_LIGHTS {
                        let i = num_point as usize;
                        let o = point_offset + i * POINT_LIGHT_FLOATS;
                        let ec = pl.effective_color();
                        self.data[o] = pl.position.x;
                        self.data[o + 1] = pl.position.y;
                        self.data[o + 2] = pl.position.z;
                        self.data[o + 3] = pl.radius;
                        self.data[o + 4] = ec.x;
                        self.data[o + 5] = ec.y;
                        self.data[o + 6] = ec.z;
                        self.data[o + 7] = pl.intensity;
                        num_point += 1;
                    }
                }
                Light::Area(_) => {
                    // Area lights handled separately (deferred or future extension)
                }
            }
        }

        // Write header (as f32 reinterpreted from u32)
        self.data[0] = f32::from_bits(num_dir);
        self.data[1] = f32::from_bits(num_point);
        // data[2], data[3] = padding (0)
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.data)
    }
}

impl Default for LightUniforms {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 2: Update lights/mod.rs to export LightUniforms**

Ensure `mod light_uniforms` and `pub use light_uniforms::LightUniforms` are uncommented in `rust/kansei-core/src/lights/mod.rs`.

Also export the constants:
```rust
pub use light_uniforms::{LightUniforms, LIGHT_UNIFORM_BYTES};
```

- [ ] **Step 3: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -5
git add kansei-core/src/lights/
git commit -m "feat: add LightUniforms for packing scene lights into GPU buffer"
```

---

### Task 3: Add lights to Scene

**Files:**
- Modify: `rust/kansei-core/src/objects/scene.rs`

- [ ] **Step 1: Add lights field and methods to Scene**

Add import at top:
```rust
use crate::lights::Light;
```

Add field to Scene struct:
```rust
    pub lights: Vec<Light>,
```

Initialize in `new()`:
```rust
    lights: Vec::new(),
```

Add method:
```rust
    /// Add a light to the scene.
    pub fn add_light(&mut self, light: Light) {
        self.lights.push(light);
    }
```

- [ ] **Step 2: Verify and commit**

```bash
cargo check -p kansei-core 2>&1 | tail -5
git add kansei-core/src/objects/scene.rs
git commit -m "feat: add lights collection to Scene"
```

---

### Task 4: Add light buffer to camera bind group and Renderer

**Files:**
- Modify: `rust/kansei-core/src/renderers/shared_layouts.rs` — Add binding 2 for lights
- Modify: `rust/kansei-core/src/renderers/renderer.rs` — Create light buffer, upload, include in camera BG

- [ ] **Step 1: Add light uniform binding to camera BGL**

In `rust/kansei-core/src/renderers/shared_layouts.rs`, add a third entry to the camera BGL:

```rust
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
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
```

- [ ] **Step 2: Add light buffer to Renderer**

In `rust/kansei-core/src/renderers/renderer.rs`:

Add import: `use crate::lights::{LightUniforms, LIGHT_UNIFORM_BYTES};`

Add fields to Renderer struct:
```rust
    light_buf: Option<wgpu::Buffer>,
    light_uniforms: LightUniforms,
```

Initialize in `new()`:
```rust
    light_buf: None,
    light_uniforms: LightUniforms::new(),
```

In `initialize()`, create the light buffer (after camera buffers):
```rust
        self.light_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/Lights"),
            size: LIGHT_UNIFORM_BYTES as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
```

Update the camera bind group creation to include the light buffer (binding 2):
```rust
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.light_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));
```

In `upload_all()`, add light packing + upload (after camera matrix upload):
```rust
        // Upload lights
        self.light_uniforms.pack(&scene.lights);
        if let Some(ref buf) = self.light_buf {
            queue.write_buffer(buf, 0, self.light_uniforms.as_bytes());
        }
```

- [ ] **Step 3: Update existing shaders for new camera bind group**

Both `basic.wgsl` and `basic_instanced.wgsl` need to declare the light uniform binding even if they don't use it, so the pipeline layout matches.

Add to both shaders after the camera bindings:

```wgsl
// ── Group 2: Lights (part of camera group) ──
struct DirLight {
    direction: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct PtLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct LightUniforms {
    num_directional: u32,
    num_point: u32,
    _pad0: u32,
    _pad1: u32,
    directional: array<DirLight, 4>,
    point: array<PtLight, 8>,
};
@group(2) @binding(2) var<uniform> lights: LightUniforms;
```

- [ ] **Step 4: Verify existing examples still build**

```bash
cargo build --example spinning_box --example instanced_cubes 2>&1 | tail -5
```

Both should build. The shaders now declare the lights binding but don't use it — that's fine, wgpu allows unused bindings.

- [ ] **Step 5: Commit**

```bash
git add kansei-core/src/renderers/ kansei-core/src/shaders/
git commit -m "feat: add light uniform buffer to camera bind group (group 2, binding 2)"
```

---

### Task 5: Create lit shader with multi-light Blinn-Phong

**Files:**
- Create: `rust/kansei-core/src/shaders/basic_lit.wgsl`

Full Blinn-Phong shader that reads from the LightUniforms buffer. Supports multiple directional and point lights.

- [ ] **Step 1: Create basic_lit.wgsl**

```wgsl
// ── Group 0: Material ──
struct MaterialUniforms {
    color: vec4<f32>,
    specular: vec4<f32>,
};
@group(0) @binding(0) var<uniform> material: MaterialUniforms;

// ── Group 1: Mesh (dynamic offset) ──
@group(1) @binding(0) var<uniform> normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

// ── Group 2: Camera ──
@group(2) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;

struct DirLight {
    direction: vec3<f32>,
    _pad0: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct PtLight {
    position: vec3<f32>,
    radius: f32,
    color: vec3<f32>,
    intensity: f32,
};
struct LightUniforms {
    num_directional: u32,
    num_point: u32,
    _pad0: u32,
    _pad1: u32,
    directional: array<DirLight, 4>,
    point: array<PtLight, 8>,
};
@group(2) @binding(2) var<uniform> lights: LightUniforms;

// ── Vertex I/O ──
struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) world_position: vec3<f32>,
    @location(2) uv: vec2<f32>,
};

@vertex
fn vertex_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let world_pos = world_matrix * input.position;
    out.clip_position = projection_matrix * view_matrix * world_pos;
    out.world_normal = (normal_matrix * vec4<f32>(input.normal, 0.0)).xyz;
    out.world_position = world_pos.xyz;
    out.uv = input.uv;
    return out;
}

@fragment
fn fragment_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let n = normalize(input.world_normal);
    let inv_view = transpose(mat3x3<f32>(
        view_matrix[0].xyz, view_matrix[1].xyz, view_matrix[2].xyz
    ));
    let camera_pos = -(inv_view * view_matrix[3].xyz);
    let view_dir = normalize(camera_pos - input.world_position);

    let albedo = material.color.rgb;
    let spec_color = material.specular.rgb;
    let shininess = material.specular.a * 256.0;

    var total_diffuse = vec3<f32>(0.0);
    var total_specular = vec3<f32>(0.0);

    // Directional lights
    for (var i = 0u; i < lights.num_directional; i++) {
        let dl = lights.directional[i];
        let light_dir = normalize(-dl.direction);
        let ndotl = max(dot(n, light_dir), 0.0);
        total_diffuse += dl.color * ndotl;

        let half_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(n, half_dir), 0.0), shininess);
        total_specular += dl.color * spec;
    }

    // Point lights
    for (var i = 0u; i < lights.num_point; i++) {
        let pl = lights.point[i];
        let to_light = pl.position - input.world_position;
        let dist = length(to_light);
        let light_dir = to_light / dist;
        let attenuation = max(1.0 - dist / pl.radius, 0.0);
        let att2 = attenuation * attenuation;

        let ndotl = max(dot(n, light_dir), 0.0);
        total_diffuse += pl.color * ndotl * att2;

        let half_dir = normalize(light_dir + view_dir);
        let spec = pow(max(dot(n, half_dir), 0.0), shininess);
        total_specular += pl.color * spec * att2;
    }

    let ambient = vec3<f32>(0.08);
    let color = albedo * (ambient + total_diffuse) + spec_color * total_specular;
    return vec4<f32>(color, material.color.a);
}
```

- [ ] **Step 2: Commit**

```bash
git add kansei-core/src/shaders/basic_lit.wgsl
git commit -m "feat: add basic_lit.wgsl with multi-light Blinn-Phong shading"
```

---

### Task 6: Create lit scene validation example

**Files:**
- Create: `rust/kansei-native/examples/lit_scene.rs`

A scene with 3 objects (floor plane, sphere-ish, box) lit by 1 directional + 2 point lights (one warm, one cool). Orbit camera.

- [ ] **Step 1: Create the example**

Create `rust/kansei-native/examples/lit_scene.rs` with a scene that:
- Creates a Renderer with MSAA 4
- Adds a large PlaneGeometry (floor) at y=0
- Adds a BoxGeometry at the center
- Adds a SphereGeometry offset to the side
- Creates a Material using `basic_lit.wgsl` with a 32-byte material uniform (color vec4 + specular vec4)
- Adds 1 DirectionalLight (sun from above-right, warm white)
- Adds 2 PointLights (one warm orange near the box, one cool blue near the sphere)
- All lights added via `scene.add_light(Light::Directional(...))`
- Orbit camera, mouse controls
- Each frame: updates object rotations, calls `renderer.render()`

Key API:
```rust
const LIT_WGSL: &str = include_str!("../../kansei-core/src/shaders/basic_lit.wgsl");

// Material uniform is 32 bytes: color (vec4) + specular (vec4)
// specular.a controls shininess (0-1, multiplied by 256 in shader)
let mat_data: [f32; 8] = [
    0.8, 0.2, 0.2, 1.0,   // color: red
    1.0, 1.0, 1.0, 0.5,   // specular: white, shininess=0.5 (128)
];

// Material needs TWO uniform bindings? No — pack into one buffer:
let mut material = Material::new(
    "LitMaterial", LIT_WGSL,
    vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)],
    MaterialOptions::default(),
);
// The buffer is 32 bytes for the MaterialUniforms struct
material.create_bind_group(device, shared, &[(
    0, BindingResource::Buffer { buffer: &mat_buf, offset: 0, size: None },
)]);

// Lights
scene.add_light(Light::Directional(DirectionalLight::new(
    Vec3::new(-0.5, -1.0, -0.3).normalize(),
    Vec3::new(1.0, 0.95, 0.8),
    1.0,
)));
scene.add_light(Light::Point(PointLight::new(
    Vec3::new(3.0, 2.0, 0.0),
    Vec3::new(1.0, 0.6, 0.2),
    2.0, 15.0,
)));
```

- [ ] **Step 2: Build and run**

```bash
cargo build --example lit_scene 2>&1 | tail -5
cargo run --example lit_scene
```

Expected: window with a floor, box, and sphere lit by a directional sun and two colored point lights. Specular highlights visible on objects.

- [ ] **Step 3: Commit**

```bash
git add kansei-native/examples/lit_scene.rs
git commit -m "feat: add lit scene example with directional + point lights"
```

---

## Post-Plan Notes

### What this plan produces:
- Complete Light types (DirectionalLight, PointLight, AreaLight) with proper fields
- Light enum for Scene storage
- LightUniforms GPU buffer packing (4 dir + 8 point max)
- Light buffer integrated into camera bind group (group 2, binding 2)
- Renderer uploads light data each frame
- basic_lit.wgsl with multi-light Blinn-Phong (diffuse + specular + attenuation)
- Existing shaders updated for new bind group layout
- Lit scene validation example

### What comes next:
- **Plan 2c: Shadows** — Shadow maps for directional/point lights, shadow sampling in shaders
- **Plan 2d: Post-processing** — SSAO, bloom, DoF, god rays, color grading, volumetric fog
- **Plan 2e: Loaders** — Texture + glTF
