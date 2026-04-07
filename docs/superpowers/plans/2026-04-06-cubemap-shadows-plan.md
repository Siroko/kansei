# Cubemap Point Shadows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the TypeScript CubeMapShadowMap to Rust — 6-face distance-based point light shadows with r32float texture array, matching the TS implementation exactly.

**Architecture:** `CubeMapShadowMap` renders the scene from each point light's position into 6 cubemap faces (r32float distance encoding). Shadow BGL expands from 3 to 5 bindings (add cubemap texture array + non-filtering sampler). Fragment shaders sample via `calcPointShadow()` with major-axis face selection and distance comparison. Renderer drives the cubemap shadow pass per-light before the main render pass.

**Tech Stack:** Rust, wgpu 24, glam 0.29, bytemuck 1

---

## File Structure

### Files to create:
- `kansei-core/src/shadows/cubemap_shadow_map.rs` — CubeMapShadowMap struct
- `kansei-core/src/shaders/cubemap_shadow.wgsl` — Cubemap shadow vertex + fragment shaders

### Files to modify:
- `kansei-core/src/shadows/mod.rs` — Replace stub with real export
- `kansei-core/src/renderers/shared_layouts.rs` — Shadow BGL expands to 5 bindings
- `kansei-core/src/renderers/renderer.rs` — Cubemap shadow pass, shadow BG update, point shadow params upload
- `kansei-core/src/shaders/basic_lit.wgsl` — Add calcPointShadow() function
- `kansei-core/src/shaders/basic.wgsl` — Add group 3 binding 3+4 declarations
- `kansei-core/src/shaders/basic_instanced.wgsl` — Same
- `kansei-core/src/lights/point.rs` — Add shadow field

---

### Task 1: Create CubeMapShadowMap struct

**Files:**
- Create: `kansei-core/src/shadows/cubemap_shadow_map.rs`
- Modify: `kansei-core/src/shadows/mod.rs`

The CubeMapShadowMap owns:
- Distance texture (`r32float`, `[resolution, resolution, 6 * max_lights]`)
- Scratch depth texture (`depth32float`, `[resolution, resolution]`)
- Light uniform buffer (dynamic offset, per-face: mat4 + vec3 + pad)
- Light uniform BGL + BG
- Mesh matrix buffers + BGL + BG (same pattern as directional ShadowMap)
- Shadow render pipeline (vertex + fragment, outputs r32float)

- [ ] **Step 1: Create cubemap_shadow_map.rs**

Implement the struct with:
- `new(renderer: &Renderer, resolution: u32, max_lights: u32)` — creates all textures, buffers, pipeline
- Face direction/up constants matching TS exactly
- `compute_face_vp(light_pos, face_idx, shadow_far)` — perspective 90° FOV + lookAt per face
- `upload_uniforms(queue, lights, face_vps)` — write all face uniforms to buffer
- `render_pass(encoder, face_slot, objects, alignment)` — render one face
- Getters: `distance_view()`, `distance_texture()`

The cubemap shadow vertex shader outputs worldPos for distance computation. The fragment shader returns `length(worldPos - lightPos)`.

- [ ] **Step 2: Update shadows/mod.rs**

Replace `pub struct CubeMapShadowMap;` stub with real export from `cubemap_shadow_map.rs`.

- [ ] **Step 3: Create cubemap_shadow.wgsl**

```wgsl
struct LightUniform {
    light_view_proj: mat4x4<f32>,
    light_world_pos: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> light: LightUniform;
@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
};

@vertex
fn shadow_vs(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>, @location(2) uv: vec2<f32>) -> VSOut {
    let wp = world_matrix * position;
    var out: VSOut;
    out.position = light.light_view_proj * wp;
    out.world_pos = wp.xyz;
    return out;
}

@fragment
fn shadow_fs(in: VSOut) -> @location(0) f32 {
    return length(in.world_pos - light.light_world_pos);
}
```

Note: group 0 = light uniform, group 2 = mesh (matching current slot order: camera=1, mesh=2).

- [ ] **Step 4: Verify and commit**

```bash
cargo check -p kansei-core
```

---

### Task 2: Expand shadow BGL to 5 bindings

**Files:**
- Modify: `kansei-core/src/renderers/shared_layouts.rs`
- Modify: `kansei-core/src/shaders/basic.wgsl`
- Modify: `kansei-core/src/shaders/basic_lit.wgsl`
- Modify: `kansei-core/src/shaders/basic_instanced.wgsl`

- [ ] **Step 1: Add bindings 3 and 4 to shadow BGL**

In `shared_layouts.rs`, add to the shadow_bgl entries:

```rust
// Binding 3: cubemap distance texture array (point shadows)
wgpu::BindGroupLayoutEntry {
    binding: 3,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Texture {
        sample_type: wgpu::TextureSampleType::Float { filterable: false },
        view_dimension: wgpu::TextureViewDimension::D2Array,
        multisampled: false,
    },
    count: None,
},
// Binding 4: non-filtering sampler for cubemap
wgpu::BindGroupLayoutEntry {
    binding: 4,
    visibility: wgpu::ShaderStages::FRAGMENT,
    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
    count: None,
},
```

- [ ] **Step 2: Update all shaders with group 3 bindings 3+4**

Add to basic.wgsl, basic_lit.wgsl, basic_instanced.wgsl:

```wgsl
@group(3) @binding(3) var cube_shadow_tex: texture_2d_array<f32>;
@group(3) @binding(4) var cube_shadow_sampler: sampler;
```

- [ ] **Step 3: Update ShadowUniforms struct in shaders**

Expand the ShadowUniforms struct to include point shadow fields:

```wgsl
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
    shadow_enabled: f32,
    point_shadow_enabled: f32,
    point_light_pos: vec3<f32>,
    point_shadow_far: f32,
};
```

- [ ] **Step 4: Update Renderer shadow bind group creation**

In `renderer.rs`, the shadow bind group creation (in `initialize()` and `enable_shadows()`) needs to include bindings 3 and 4. Create a dummy 1x1x6 r32float texture for the cubemap and a non-filtering sampler when no cubemap shadows are active.

- [ ] **Step 5: Verify and commit**

---

### Task 3: Add calcPointShadow to basic_lit.wgsl

**Files:**
- Modify: `kansei-core/src/shaders/basic_lit.wgsl`

- [ ] **Step 1: Add calcPointShadow function**

Port the TS `calcPointShadow` exactly:

```wgsl
fn calcPointShadow(world_pos: vec3<f32>) -> f32 {
    if (shadow_uniforms.point_shadow_enabled < 0.5) {
        return 1.0;
    }
    
    let to_frag = world_pos - shadow_uniforms.point_light_pos;
    let dist = length(to_frag);
    let dir = to_frag / max(dist, 1e-6);
    
    let ax = abs(dir.x);
    let ay = abs(dir.y);
    let az = abs(dir.z);
    
    var face_index: i32;
    var uv: vec2<f32>;
    
    if (ax >= ay && ax >= az) {
        if (dir.x > 0.0) {
            face_index = 0; uv = vec2<f32>(-dir.z, -dir.y) / ax;
        } else {
            face_index = 1; uv = vec2<f32>(dir.z, -dir.y) / ax;
        }
    } else if (ay >= ax && ay >= az) {
        if (dir.y > 0.0) {
            face_index = 2; uv = vec2<f32>(dir.x, dir.z) / ay;
        } else {
            face_index = 3; uv = vec2<f32>(dir.x, -dir.z) / ay;
        }
    } else {
        if (dir.z > 0.0) {
            face_index = 4; uv = vec2<f32>(dir.x, -dir.y) / az;
        } else {
            face_index = 5; uv = vec2<f32>(-dir.x, -dir.y) / az;
        }
    }
    
    uv = uv * 0.5 + 0.5;
    
    let tex_size = vec2<f32>(textureDimensions(cube_shadow_tex));
    let tex_coord = vec2<i32>(clamp(uv * tex_size, vec2<f32>(0.0), tex_size - 1.0));
    let stored_dist = textureLoad(cube_shadow_tex, tex_coord, face_index, 0).r;
    
    let bias = 0.05;
    if (dist - bias > stored_dist) {
        return 0.0;
    }
    return 1.0;
}
```

- [ ] **Step 2: Integrate into fragment_main**

In fragment_main, multiply lighting by point shadow:

```wgsl
    let dir_shadow = calcDirectionalShadow(input.world_position, n);
    let point_shadow = calcPointShadow(input.world_position);
    let shadow = dir_shadow * point_shadow;
    let color = albedo * (ambient + total_diffuse * shadow) + spec_color * total_specular * shadow;
```

- [ ] **Step 3: Verify and commit**

---

### Task 4: Renderer cubemap shadow pass integration

**Files:**
- Modify: `kansei-core/src/renderers/renderer.rs`
- Modify: `kansei-core/src/lights/point.rs`

- [ ] **Step 1: Add cubemap shadow fields to Renderer**

```rust
cubemap_shadow_map: Option<CubeMapShadowMap>,
cube_shadow_sampler: Option<wgpu::Sampler>,
cube_dummy_tex: Option<wgpu::Texture>,
cube_dummy_view: Option<wgpu::TextureView>,
```

- [ ] **Step 2: Create dummy cubemap texture in initialize()**

1x1x6 r32float texture filled with large distance (1e10). Non-filtering sampler. Include in shadow bind group at bindings 3+4.

- [ ] **Step 3: Add enable_point_shadows() method**

```rust
pub fn enable_point_shadows(&mut self, resolution: u32, max_lights: u32) {
    let csm = CubeMapShadowMap::new(self, resolution, max_lights);
    // Rebuild shadow bind group with real cubemap texture
    self.rebuild_shadow_bind_group();
    self.cubemap_shadow_map = Some(csm);
}
```

- [ ] **Step 4: Add cubemap shadow pass to render()**

Before the main render pass, after directional shadow pass:

```rust
if let Some(ref mut csm) = self.cubemap_shadow_map {
    // Find point lights with cast_shadow=true
    let shadow_lights: Vec<_> = scene.lights()
        .filter_map(|l| match l { Light::Point(pl) if pl.cast_shadow => Some(pl), _ => None })
        .collect();
    
    if !shadow_lights.is_empty() {
        // Compute VPs for each light × 6 faces
        // Upload uniforms
        // For each light, for each face: render shadow pass
        // Upload point shadow params to shadow uniform buffer
    }
}
```

- [ ] **Step 5: Upload point shadow params**

Expand shadow uniform buffer to include point shadow fields:
```rust
shadow_data[19] = 1.0; // pointShadowEnabled
shadow_data[20] = point_light.position.x;
shadow_data[21] = point_light.position.y;
shadow_data[22] = point_light.position.z;
shadow_data[23] = shadow_far;
```

- [ ] **Step 6: Add cast_shadow to PointLight**

In `lights/point.rs`, ensure `cast_shadow: bool` exists and is used by the renderer to decide which lights get shadow passes.

- [ ] **Step 7: Verify and commit**

---

### Task 5: Validation example

**Files:**
- Modify: `kansei-native/examples/shadow_scene.rs` or create new example

- [ ] **Step 1: Update shadow_scene to include a point light with shadows**

Add a point light with `cast_shadow: true`. Call `renderer.enable_point_shadows(512, 4)`.

- [ ] **Step 2: Run and verify**

```bash
cargo run --example shadow_scene --release
```

Verify: point light casts shadows on the floor from the box and sphere.

- [ ] **Step 3: Commit**

---

## Post-Plan Notes

### What this produces:
- CubeMapShadowMap with 6-face r32float distance rendering
- 5-binding shadow BGL matching TS
- calcPointShadow() with cubemap face selection
- Renderer drives cubemap shadow passes per-light
- Point lights can cast shadows

### Next items in queue:
- Render bundle caching
- MSAA depth-copy pass
- parseIncludes() shader preprocessor
- readBackBuffer GPU readback
- Path tracer integration
- onChange dirty callbacks
- System pattern
