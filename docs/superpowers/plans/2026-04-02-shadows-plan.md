# Directional Shadows Implementation Plan (Plan 2c)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add directional light shadow mapping with PCF filtering. Objects cast and receive shadows from one directional light. Point light cubemap shadows deferred to a follow-up plan.

**Architecture:** `ShadowMap` renders the scene from the light's perspective into a depth texture. The Renderer runs a shadow pass before the main pass, computing a tight orthographic frustum from the camera. Shadow bind group (group 3) holds the depth texture, comparison sampler, and shadow uniforms. Fragment shaders sample shadows via `calcDirectionalShadow()` with 3x3 PCF.

**Tech Stack:** Rust, wgpu 24, bytemuck 1, glam 0.29

---

## File Structure

### Files to create:
- `rust/kansei-core/src/shadows/shadow_map.rs` — ShadowMap struct with depth texture + light VP computation
- `rust/kansei-core/src/shadows/shadow_pass.rs` — Shadow rendering pipeline (depth-only vertex shader)
- `rust/kansei-core/src/shaders/shadow_vs.wgsl` — Shadow pass vertex shader
- `rust/kansei-core/src/shaders/shadow_sampling.wgsl` — Shadow sampling functions for inclusion in lit shaders
- `rust/kansei-native/examples/shadow_scene.rs` — Validation example

### Files to modify:
- `rust/kansei-core/src/shadows/mod.rs` — Replace stubs with real exports
- `rust/kansei-core/src/renderers/shared_layouts.rs` — Add shadow_bgl (group 3)
- `rust/kansei-core/src/renderers/renderer.rs` — Shadow pass + shadow bind group + shadow uniforms
- `rust/kansei-core/src/materials/material.rs` — Include shadow_bgl in pipeline layout
- `rust/kansei-core/src/shaders/basic.wgsl` — Add shadow group 3 declarations
- `rust/kansei-core/src/shaders/basic_instanced.wgsl` — Same
- `rust/kansei-core/src/shaders/basic_lit.wgsl` — Add shadow sampling to lighting

---

### Task 1: Create ShadowMap struct with depth texture and light VP computation

**Files:**
- Create: `rust/kansei-core/src/shadows/shadow_map.rs`
- Modify: `rust/kansei-core/src/shadows/mod.rs`

The ShadowMap creates a depth texture and computes a tight orthographic light-view-projection matrix that fits the camera frustum.

- [ ] **Step 1: Create shadow_map.rs**

```rust
// rust/kansei-core/src/shadows/shadow_map.rs
use crate::math::{Vec3, Mat4};
use crate::cameras::Camera;

/// Directional light shadow map.
pub struct ShadowMap {
    pub resolution: u32,
    pub depth_texture: Option<wgpu::Texture>,
    pub depth_view: Option<wgpu::TextureView>,
    pub light_vp: Mat4,
    pub light_vp_buf: Option<wgpu::Buffer>,
    pub bias: f32,
    pub normal_bias: f32,
    initialized: bool,
}

impl ShadowMap {
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            depth_texture: None,
            depth_view: None,
            light_vp: Mat4::identity(),
            light_vp_buf: None,
            bias: 0.001,
            normal_bias: 0.02,
            initialized: false,
        }
    }

    /// Create GPU resources.
    pub fn initialize(&mut self, device: &wgpu::Device) {
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("ShadowMap/Depth"),
            size: wgpu::Extent3d {
                width: self.resolution,
                height: self.resolution,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.depth_view = Some(tex.create_view(&Default::default()));
        self.depth_texture = Some(tex);

        self.light_vp_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ShadowMap/LightVP"),
            size: 64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        self.initialized = true;
    }

    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Compute tight orthographic light-view-projection matrix from camera frustum.
    pub fn compute_light_vp(&mut self, camera: &Camera, light_dir: &Vec3) {
        let g_light_dir = glam::Vec3::new(light_dir.x, light_dir.y, light_dir.z).normalize();

        // Build camera inverse VP to get frustum corners in world space
        let view = camera.view_matrix.to_glam();
        let proj = camera.projection_matrix.to_glam();
        let inv_vp = (proj * view).inverse();

        // 8 frustum corners in NDC (WebGPU: z in [0,1])
        let ndc_corners = [
            glam::Vec3::new(-1.0, -1.0, 0.0),
            glam::Vec3::new( 1.0, -1.0, 0.0),
            glam::Vec3::new(-1.0,  1.0, 0.0),
            glam::Vec3::new( 1.0,  1.0, 0.0),
            glam::Vec3::new(-1.0, -1.0, 1.0),
            glam::Vec3::new( 1.0, -1.0, 1.0),
            glam::Vec3::new(-1.0,  1.0, 1.0),
            glam::Vec3::new( 1.0,  1.0, 1.0),
        ];

        // Transform to world space
        let mut world_corners = [glam::Vec3::ZERO; 8];
        let mut center = glam::Vec3::ZERO;
        for (i, ndc) in ndc_corners.iter().enumerate() {
            let clip = inv_vp * glam::Vec4::new(ndc.x, ndc.y, ndc.z, 1.0);
            world_corners[i] = clip.truncate() / clip.w;
            center += world_corners[i];
        }
        center /= 8.0;

        // Light view matrix
        let eye = center - g_light_dir * 100.0;
        let up = if g_light_dir.y.abs() >= 0.99 {
            glam::Vec3::X
        } else {
            glam::Vec3::Y
        };
        let light_view = glam::Mat4::look_at_rh(eye, center, up);

        // Compute AABB in light space
        let mut min_ls = glam::Vec3::splat(f32::MAX);
        let mut max_ls = glam::Vec3::splat(f32::MIN);
        for corner in &world_corners {
            let ls = (light_view * glam::Vec4::new(corner.x, corner.y, corner.z, 1.0)).truncate();
            min_ls = min_ls.min(ls);
            max_ls = max_ls.max(ls);
        }

        // Extend Z range to catch shadow casters behind camera
        let z_range = max_ls.z - min_ls.z;
        min_ls.z -= z_range * 2.0;

        // Orthographic projection (right-handed, Z [0,1] for WebGPU)
        let mut light_proj = glam::Mat4::orthographic_rh(
            min_ls.x, max_ls.x,
            min_ls.y, max_ls.y,
            -max_ls.z, -min_ls.z,
        );

        // glam's orthographic_rh already maps Z to [0,1] in WebGPU mode
        // but we need to verify — if it maps to [-1,1], remap:
        // light_proj.z_axis.z *= 0.5;
        // light_proj.w_axis.z = light_proj.w_axis.z * 0.5 + 0.5;

        self.light_vp = Mat4::from(light_proj * light_view);
    }

    /// Upload light VP matrix to GPU.
    pub fn upload(&self, queue: &wgpu::Queue) {
        if let Some(ref buf) = self.light_vp_buf {
            queue.write_buffer(buf, 0, bytemuck::cast_slice(self.light_vp.as_slice()));
        }
    }
}
```

- [ ] **Step 2: Update shadows/mod.rs**

Replace `rust/kansei-core/src/shadows/mod.rs` with:

```rust
mod shadow_map;

pub use shadow_map::ShadowMap;

/// Cubemap shadow map for point lights (future).
pub struct CubeMapShadowMap;
```

- [ ] **Step 3: Verify and commit**

```bash
cd /Users/felixmartinez/Documents/dev/kansei/rust && cargo check -p kansei-core 2>&1 | tail -5
git add kansei-core/src/shadows/
git commit -m "feat: add ShadowMap with depth texture and light VP computation"
```

---

### Task 2: Add shadow bind group (group 3) to SharedLayouts and Material

**Files:**
- Modify: `rust/kansei-core/src/renderers/shared_layouts.rs` — Add shadow_bgl
- Modify: `rust/kansei-core/src/materials/material.rs` — Include shadow_bgl in pipeline layout

This adds the 4th bind group slot. The shadow BGL has:
- Binding 0: depth texture (texture_depth_2d)
- Binding 1: comparison sampler
- Binding 2: shadow uniforms (96 bytes: lightVP mat4, bias, normalBias, shadowEnabled, pad)

- [ ] **Step 1: Update shared_layouts.rs**

Add `shadow_bgl` field to `SharedLayouts` and create it in `new()`:

```rust
pub struct SharedLayouts {
    pub mesh_bgl: wgpu::BindGroupLayout,
    pub camera_bgl: wgpu::BindGroupLayout,
    pub shadow_bgl: wgpu::BindGroupLayout,
}
```

In `new()`, after camera_bgl creation, add:

```rust
        // Group 3: shadows (depth texture + comparison sampler + shadow uniforms)
        let shadow_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shared/ShadowBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
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

        Self { mesh_bgl, camera_bgl, shadow_bgl }
```

- [ ] **Step 2: Update Material pipeline layout to include shadow_bgl**

In `rust/kansei-core/src/materials/material.rs`, in `ensure_shared()`, change the pipeline layout from 3 groups to 4:

```rust
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(&format!("{}/PipelineLayout", self.label)),
            bind_group_layouts: &[&material_bgl, &shared.mesh_bgl, &shared.camera_bgl, &shared.shadow_bgl],
            push_constant_ranges: &[],
        });
```

- [ ] **Step 3: Update ALL existing shaders to declare group 3 bindings**

Every shader must declare group 3 or the pipeline layout won't match. Add to `basic.wgsl`, `basic_instanced.wgsl`, and `basic_lit.wgsl`:

```wgsl
// ── Group 3: Shadows ──
struct ShadowUniforms {
    light_view_proj: mat4x4<f32>,
    bias: f32,
    normal_bias: f32,
    shadow_enabled: f32,
    _pad: f32,
};
@group(3) @binding(0) var shadow_depth_tex: texture_depth_2d;
@group(3) @binding(1) var shadow_sampler: sampler_comparison;
@group(3) @binding(2) var<uniform> shadow_uniforms: ShadowUniforms;
```

- [ ] **Step 4: Verify all examples still compile**

```bash
cargo check -p kansei-core 2>&1 | tail -5
cargo build --example spinning_box --example instanced_cubes --example lit_scene 2>&1 | tail -3
```

Note: examples will fail at runtime until the Renderer creates the shadow bind group (Task 3). But they should compile.

- [ ] **Step 5: Commit**

```bash
git add kansei-core/src/renderers/shared_layouts.rs kansei-core/src/materials/material.rs kansei-core/src/shaders/
git commit -m "feat: add shadow bind group (group 3) to pipeline layout"
```

---

### Task 3: Shadow pass rendering + shadow bind group in Renderer

**Files:**
- Create: `rust/kansei-core/src/shaders/shadow_vs.wgsl` — Shadow vertex shader
- Modify: `rust/kansei-core/src/renderers/renderer.rs` — Shadow pass, shadow bind group, shadow uniforms

This is the integration task. The Renderer needs to:
1. Create a dummy 1x1 depth texture (when no shadow map assigned)
2. Create comparison sampler + shadow uniform buffer
3. Create the shadow bind group (group 3)
4. Create a shadow render pipeline (depth-only)
5. Run the shadow pass before the main render pass
6. Upload shadow uniforms each frame
7. Set shadow bind group (group 3) in the main render pass

- [ ] **Step 1: Create shadow_vs.wgsl**

```wgsl
// Shadow pass — depth-only vertex shader
@group(0) @binding(0) var<uniform> light_view_proj: mat4x4<f32>;

@group(1) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> world_matrix: mat4x4<f32>;

@vertex
fn shadow_vs(
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
) -> @builtin(position) vec4<f32> {
    return light_view_proj * world_matrix * position;
}
```

Note: This shader uses group 0 for the light VP (not the material). The shadow pass uses a separate pipeline layout with:
- Group 0: light VP uniform
- Group 1: shared mesh matrices (same as main pass)

- [ ] **Step 2: Add shadow infrastructure to Renderer**

In `rust/kansei-core/src/renderers/renderer.rs`, add these fields to the Renderer struct:

```rust
    // Shadow resources
    shadow_map: Option<crate::shadows::ShadowMap>,
    shadow_uniform_buf: Option<wgpu::Buffer>,
    shadow_bind_group: Option<wgpu::BindGroup>,
    shadow_comparison_sampler: Option<wgpu::Sampler>,
    shadow_dummy_depth_view: Option<wgpu::TextureView>,
    shadow_pipeline: Option<wgpu::RenderPipeline>,
    shadow_light_vp_bgl: Option<wgpu::BindGroupLayout>,
    shadow_light_vp_bg: Option<wgpu::BindGroup>,
    shadows_enabled: bool,
```

In `initialize()`, after creating camera/light resources:

a) Create dummy 1x1 depth texture (used when no shadow map):
```rust
        let dummy_depth = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Renderer/DummyDepth"),
            size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            mip_level_count: 1, sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.shadow_dummy_depth_view = Some(dummy_depth.create_view(&Default::default()));
```

b) Create comparison sampler:
```rust
        self.shadow_comparison_sampler = Some(device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Renderer/ShadowSampler"),
            compare: Some(wgpu::CompareFunction::Less),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        }));
```

c) Create shadow uniform buffer (96 bytes: mat4 + 4 floats + vec3 + float):
```rust
        self.shadow_uniform_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Renderer/ShadowUniforms"),
            size: 96,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
```

d) Create shadow bind group with dummy depth texture:
```rust
        self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/ShadowBG"),
            layout: &shared.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        self.shadow_dummy_depth_view.as_ref().unwrap()
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        self.shadow_comparison_sampler.as_ref().unwrap()
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.shadow_uniform_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));
```

- [ ] **Step 3: Add enable_shadows method and shadow pass to render()**

Add public method:
```rust
    pub fn enable_shadows(&mut self, resolution: u32) {
        let device = self.device.as_ref().unwrap();
        let mut sm = crate::shadows::ShadowMap::new(resolution);
        sm.initialize(device);

        // Rebuild shadow bind group with real depth texture
        let shared = self.shared_layouts.as_ref().unwrap();
        self.shadow_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Renderer/ShadowBG"),
            layout: &shared.shadow_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(sm.depth_view.as_ref().unwrap()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(
                        self.shadow_comparison_sampler.as_ref().unwrap()
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.shadow_uniform_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        }));

        // Create shadow pipeline (depth-only, no fragment)
        // Light VP bind group layout (group 0 for shadow pass)
        let shadow_light_vp_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Shadow/LightVPBGL"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        let shadow_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Shadow/PipelineLayout"),
            bind_group_layouts: &[&shadow_light_vp_bgl, &shared.mesh_bgl],
            push_constant_ranges: &[],
        });

        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shadow/Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/shadow_vs.wgsl").into()),
        });

        self.shadow_pipeline = Some(device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Shadow/Pipeline"),
            layout: Some(&shadow_layout),
            vertex: wgpu::VertexState {
                module: &shadow_shader,
                entry_point: Some("shadow_vs"),
                buffers: &[crate::geometries::Vertex::LAYOUT],
                compilation_options: Default::default(),
            },
            fragment: None,
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        }));

        // Create light VP bind group
        self.shadow_light_vp_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Shadow/LightVPBG"),
            layout: &shadow_light_vp_bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: sm.light_vp_buf.as_ref().unwrap().as_entire_binding(),
            }],
        }));

        self.shadow_light_vp_bgl = Some(shadow_light_vp_bgl);
        self.shadow_map = Some(sm);
        self.shadows_enabled = true;
    }
```

In `render()`, after Phase 0.5 (pipeline pre-warm) and before Phase 1 (upload), add a shadow pass:

```rust
        // Shadow pass (if enabled)
        if self.shadows_enabled {
            if let Some(ref mut sm) = self.shadow_map {
                // Find first directional light
                if let Some(dir_light) = scene.lights.iter().find_map(|l| {
                    if let crate::lights::Light::Directional(dl) = l { Some(dl) } else { None }
                }) {
                    sm.compute_light_vp(camera, &dir_light.direction);
                    sm.upload(self.queue.as_ref().unwrap());

                    // Upload shadow uniforms
                    let mut shadow_data = [0.0f32; 24];
                    shadow_data[..16].copy_from_slice(sm.light_vp.as_slice());
                    shadow_data[16] = sm.bias;
                    shadow_data[17] = sm.normal_bias;
                    shadow_data[18] = 1.0; // shadowEnabled
                    if let Some(ref buf) = self.shadow_uniform_buf {
                        self.queue.as_ref().unwrap().write_buffer(
                            buf, 0, bytemuck::cast_slice(&shadow_data)
                        );
                    }

                    // Render shadow pass
                    let device = self.device.as_ref().unwrap();
                    let mut encoder = device.create_command_encoder(&Default::default());
                    {
                        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                            label: Some("Renderer/ShadowPass"),
                            color_attachments: &[],
                            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                                view: sm.depth_view.as_ref().unwrap(),
                                depth_ops: Some(wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(1.0),
                                    store: wgpu::StoreOp::Store,
                                }),
                                stencil_ops: None,
                            }),
                            ..Default::default()
                        });

                        pass.set_pipeline(self.shadow_pipeline.as_ref().unwrap());
                        pass.set_bind_group(0, self.shadow_light_vp_bg.as_ref().unwrap(), &[]);

                        let alignment = self.matrix_alignment;
                        for (draw_idx, scene_idx) in scene.ordered_indices().enumerate() {
                            let r = &scene.renderables()[scene_idx];
                            if !r.visible || !r.cast_shadow || !r.geometry.initialized {
                                continue;
                            }

                            let offset = (draw_idx as u32) * alignment;
                            pass.set_bind_group(1, self.mesh_bind_group.as_ref().unwrap(), &[offset, offset]);

                            pass.set_vertex_buffer(0, r.geometry.vertex_buffer.as_ref().unwrap().slice(..));
                            pass.set_index_buffer(
                                r.geometry.index_buffer.as_ref().unwrap().slice(..),
                                wgpu::IndexFormat::Uint32,
                            );
                            pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                        }
                    }
                    self.queue.as_ref().unwrap().submit(std::iter::once(encoder.finish()));
                }
            }
        } else {
            // Upload disabled shadow uniforms
            let shadow_data = [0.0f32; 24];
            if let Some(ref buf) = self.shadow_uniform_buf {
                self.queue.as_ref().unwrap().write_buffer(
                    buf, 0, bytemuck::cast_slice(&shadow_data)
                );
            }
        }
```

In the main render pass, add shadow bind group (group 3) after camera bind group:
```rust
            pass.set_bind_group(3, self.shadow_bind_group.as_ref().unwrap(), &[]);
```

- [ ] **Step 4: Verify all examples still compile and run**

```bash
cargo build --example spinning_box --example instanced_cubes --example lit_scene 2>&1 | tail -3
```

The examples should work — shadows are disabled by default (shadow_data is all zeros, shadowEnabled = 0.0).

- [ ] **Step 5: Commit**

```bash
git add kansei-core/src/renderers/ kansei-core/src/shaders/shadow_vs.wgsl
git commit -m "feat: shadow pass rendering + shadow bind group in Renderer"
```

---

### Task 4: Add shadow sampling to basic_lit.wgsl

**Files:**
- Modify: `rust/kansei-core/src/shaders/basic_lit.wgsl` — Add shadow sampling in fragment shader

- [ ] **Step 1: Add calcDirectionalShadow function and integrate into lighting**

In basic_lit.wgsl, add the shadow sampling function before `fragment_main`, and multiply the lighting result by the shadow factor:

```wgsl
fn calcDirectionalShadow(world_pos: vec3<f32>, world_normal: vec3<f32>) -> f32 {
    if (shadow_uniforms.shadow_enabled < 0.5) {
        return 1.0;
    }

    // Normal bias to reduce self-shadowing
    let biased_pos = world_pos + world_normal * shadow_uniforms.normal_bias;

    // Transform to light clip space
    let ls_pos = shadow_uniforms.light_view_proj * vec4<f32>(biased_pos, 1.0);
    let ndc = ls_pos.xyz / ls_pos.w;

    // NDC to UV (Y-flipped for WebGPU)
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

    // Bounds check
    let in_bounds = step(0.0, uv.x) * step(uv.x, 1.0)
                  * step(0.0, uv.y) * step(uv.y, 1.0)
                  * step(ndc.z, 1.0);
    if (in_bounds < 0.5) {
        return 1.0;
    }

    // 3x3 PCF
    let tex_size = vec2<f32>(textureDimensions(shadow_depth_tex));
    let texel_size = 1.0 / tex_size;
    let ref_depth = ndc.z - shadow_uniforms.bias;

    var shadow = 0.0;
    for (var x = -1; x <= 1; x++) {
        for (var y = -1; y <= 1; y++) {
            let sample_uv = clamp(
                uv + vec2<f32>(f32(x), f32(y)) * texel_size,
                vec2<f32>(0.0), vec2<f32>(1.0)
            );
            shadow += textureSampleCompare(
                shadow_depth_tex, shadow_sampler,
                sample_uv, ref_depth
            );
        }
    }
    return shadow / 9.0;
}
```

In `fragment_main`, add shadow factor to the lighting:

After computing `total_diffuse` and `total_specular`, before the final color:

```wgsl
    let shadow = calcDirectionalShadow(input.world_position, n);
    let color = albedo * (ambient + total_diffuse * shadow) + spec_color * total_specular * shadow;
```

- [ ] **Step 2: Verify it compiles**

```bash
cargo build --example lit_scene 2>&1 | tail -3
```

- [ ] **Step 3: Commit**

```bash
git add kansei-core/src/shaders/basic_lit.wgsl
git commit -m "feat: add directional shadow sampling with 3x3 PCF to lit shader"
```

---

### Task 5: Create shadow scene validation example

**Files:**
- Create: `rust/kansei-native/examples/shadow_scene.rs`

A scene similar to lit_scene but with shadows enabled. A floor plane receives shadows from a box and sphere lit by a directional light.

- [ ] **Step 1: Create the example**

Same as lit_scene but with these additions:

After creating the renderer:
```rust
renderer.enable_shadows(2048);
```

Use `basic_lit.wgsl` for all materials (it now has shadow sampling).

The floor should have `cast_shadow: false` (floors don't cast shadows on themselves):
```rust
if let Some(r) = scene.get_mut(floor_idx) {
    r.cast_shadow = false;
}
```

The directional light should point downward at an angle to create visible shadows on the floor:
```rust
scene.add_light(Light::Directional(DirectionalLight::new(
    Vec3::new(-0.3, -1.0, -0.5),
    Vec3::new(1.0, 0.95, 0.8),
    1.0,
)));
```

Orbit camera at distance 15, elevation 0.5, looking at (0, 0, 0).

- [ ] **Step 2: Build and run**

```bash
cargo build --example shadow_scene 2>&1 | tail -5
cargo run --example shadow_scene
```

Expected: floor with box and sphere casting shadows. Shadows should be soft (PCF) and follow the directional light angle.

- [ ] **Step 3: Commit**

```bash
git add kansei-native/examples/shadow_scene.rs
git commit -m "feat: add shadow scene example with directional shadow mapping"
```

---

## Post-Plan Notes

### What this plan produces:
- ShadowMap with depth texture and tight orthographic frustum fitting
- Shadow bind group (group 3) with depth texture, comparison sampler, uniforms
- Shadow pass: depth-only rendering before main pass
- 3x3 PCF directional shadow sampling in basic_lit.wgsl
- All existing shaders/examples updated for 4-group pipeline layout
- Shadow scene validation example

### Limitations (deferred):
- Point light cubemap shadows (CubeMapShadowMap) — follow-up plan
- Multiple directional shadow cascades — future optimization
- Custom shadow vertex code for instanced objects — future

### What comes next:
- **Plan 2d: Post-processing** — SSAO, bloom, DoF, god rays, color grading, volumetric fog
- **Plan 2e: Loaders** — Texture + glTF
