# Path Tracer Plan B: Core Tracing

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the core path tracing shaders and Rust orchestration — ray generation, BVH traversal, ray-triangle/AABB intersection, BSDF evaluation, basic path tracing with accumulation.

**Architecture:** 1:1 port of the TS trace/intersection/traversal shaders. PathTracerMaterial for PBR properties. TraceParams uniform for per-frame camera/settings. Single compute dispatch traces rays and writes to an rgba16float output texture.

**Tech Stack:** Rust, wgpu 24, WGSL

---

### Task 1: Port intersection.wgsl and traversal.wgsl

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/intersection.wgsl`
- Create: `kansei-core/src/pathtracer/shaders/traversal.wgsl`

Port from TS. These are the foundation all tracing depends on.

**intersection.wgsl** contains:
- `Ray` struct (origin, direction, tmin, tmax)
- `HitInfo` struct (t, uv, normal, tri_index, instance_index, material_index)
- `rayAABB(ray, bmin, bmax) -> f32` — slab test
- `rayTriangle(ray, v0, v1, v2) -> (t, u, v)` — Möller–Trumbore

**traversal.wgsl** contains:
- `Instance` struct (matching PackedInstance layout)
- `transformRayToLocal(ray, inv_transform) -> Ray`
- `transformNormalToWorld(normal, inv_transform) -> vec3f`
- `traverseBLAS(ray, bvh4_nodes, triangles, node_offset, tri_offset) -> HitInfo`
- `traceBVHInternal(ray, tlas_nodes, instances, bvh4_nodes, triangles) -> HitInfo`

Read the TS files at `src/pathtracer/shaders/intersection.wgsl.ts` and `src/pathtracer/shaders/traversal.wgsl.ts`. Extract the pure WGSL.

- [ ] **Step 1: Create intersection.wgsl**
- [ ] **Step 2: Create traversal.wgsl**
- [ ] **Step 3: Verify compilation** (no Rust changes, just shader files)
- [ ] **Step 4: Commit**

---

### Task 2: Port PathTracerMaterial

**Files:**
- Create: `kansei-core/src/pathtracer/material.rs`
- Modify: `kansei-core/src/pathtracer/mod.rs`

64-byte GPU struct matching TS exactly:
```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PathTracerMaterial {
    pub albedo: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub max_bounces: f32,  // negative if probe_invisible
    pub transmission: f32,
    pub absorption_color: [f32; 3],
    pub absorption_density: f32,
    pub emissive: [f32; 3],
    pub emissive_intensity: f32,
}
```

- [ ] **Step 1: Create material.rs**
- [ ] **Step 2: Commit**

---

### Task 3: Port trace.wgsl (main path tracing shader)

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/trace.wgsl`

This is the biggest shader — ray generation, path tracing loop, BSDF evaluation, light sampling, MIS.

Port from `src/pathtracer/shaders/trace.wgsl.ts`. Key components:
- TraceParams struct (192 bytes)
- MaterialData struct
- LightData struct (64 bytes)
- Ray generation from camera inverse VP
- Path tracing loop (bounce 0..maxBounces)
- GGX specular BSDF
- Diffuse cosine BSDF
- Fresnel (Schlick)
- Direct light sampling with MIS
- Transmission/refraction
- Blue noise sampling
- PCG RNG fallback

The shader concatenates intersection.wgsl + traversal.wgsl + trace.wgsl at runtime. In Rust, we'll use the shader include system (parse_includes) or concatenate the strings.

- [ ] **Step 1: Create trace.wgsl** (port from TS, extract WGSL)
- [ ] **Step 2: Commit**

---

### Task 4: Create PathTracer Rust struct

**Files:**
- Create: `kansei-core/src/pathtracer/path_tracer.rs`
- Modify: `kansei-core/src/pathtracer/mod.rs`

The PathTracer struct owns the trace compute pipeline and dispatches tracing.

```rust
pub struct PathTracer {
    trace_pipeline: wgpu::ComputePipeline,
    trace_bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    output_texture: Option<wgpu::Texture>,
    output_view: Option<wgpu::TextureView>,
    materials_buf: Option<wgpu::Buffer>,
    lights_buf: Option<wgpu::Buffer>,
    blue_noise_buf: Option<wgpu::Buffer>,
    frame_index: u32,
    width: u32,
    height: u32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
```

Methods:
- `new(renderer: &Renderer)` — create pipeline from concatenated shaders
- `resize(width, height)` — recreate output texture
- `set_materials(materials: &[PathTracerMaterial])`
- `set_lights(lights: &[LightData])`
- `trace(encoder, bvh_data: &GPUBVHData, camera, width, height)` — dispatch trace compute

The trace shader bind group:
```
binding 0: TraceParams uniform
binding 1: output texture (storage write)
binding 2: triangles buffer (storage read)
binding 3: bvh4_nodes buffer (storage read)
binding 4: tlas_nodes buffer (storage read)
binding 5: instances buffer (storage read)
binding 6: materials buffer (storage read)
binding 7: lights buffer (storage read)
binding 8: blue noise buffer (storage read)
```

- [ ] **Step 1: Create path_tracer.rs**
- [ ] **Step 2: Wire into mod.rs**
- [ ] **Step 3: Verify and commit**

---

### Task 5: Blue Noise Generation

**Files:**
- Create: `kansei-core/src/pathtracer/blue_noise.rs`

Port the 128x128 blue noise table from TS. This can be a precomputed constant or generated at init time.

- [ ] **Step 1: Create blue_noise.rs** with `generate_blue_noise() -> Vec<f32>`
- [ ] **Step 2: Commit**

---

### Task 6: Validation — basic trace test

**Files:**
- Create: `kansei-native/examples/pathtracer_test.rs`

Minimal example: create a scene with a few boxes, build BVH, trace 1 SPP, display result.

- [ ] **Step 1: Create example**
- [ ] **Step 2: Verify and commit**

---

## Post-Plan Notes

### What this produces:
- intersection.wgsl + traversal.wgsl (BVH traversal shaders)
- trace.wgsl (full path tracing with GGX + MIS + transmission)
- PathTracerMaterial (64-byte GPU struct)
- PathTracer struct (compute pipeline + dispatch)
- Blue noise sampling
- Basic validation example

### Next:
- **Plan C: Denoising + Advanced** — SVGF, ReSTIR, probes, voxels
- **Plan D: Integration** — PathTracerEffect as PostProcessingEffect
