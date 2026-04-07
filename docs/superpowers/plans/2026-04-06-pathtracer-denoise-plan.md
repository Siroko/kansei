# Path Tracer Plan C: Denoising + Advanced Features

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port SVGF temporal/spatial denoising, ReSTIR direct illumination, irradiance probe grid, and composite shader from TypeScript to Rust.

**Architecture:** Each denoiser stage is a separate compute shader. Temporal denoising uses dual-buffered history textures (ping-pong). Spatial denoising uses A-trous wavelet filter with variance guidance. ReSTIR uses reservoir resampling for direct lighting. Probes use L2 SH with dual-buffered accumulation. All shaders ported 1:1 from TS.

**Tech Stack:** Rust, wgpu 24, WGSL

---

### Task 1: Temporal Denoising (SVGF)

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/denoise-temporal.wgsl`
- Create: `kansei-core/src/pathtracer/temporal_denoise.rs`

Port from `src/pathtracer/shaders/denoise-temporal.wgsl.ts`.

The temporal denoiser reprojects the previous frame's result using motion vectors (camera delta), blends with the current frame, and tracks variance moments.

- [ ] **Step 1: Port denoise-temporal.wgsl**

Read the TS file and extract WGSL. Key:
- Reads current GI, depth, normal from current frame
- Reprojects pixel to previous frame position using previous VP matrix
- Blends: `output = lerp(history, current, blend_factor)`
- Updates moments (mean, variance) for spatial filter guidance
- Workgroup size 8x8

- [ ] **Step 2: Create temporal_denoise.rs**

```rust
pub struct TemporalDenoise {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    history_a: Option<wgpu::Texture>,
    history_b: Option<wgpu::Texture>,
    moments_a: Option<wgpu::Texture>,
    moments_b: Option<wgpu::Texture>,
    ping: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
```

Methods: `new(&Renderer)`, `resize(w, h)`, `denoise(encoder, input_view, depth_view, normal_view, prev_vp, current_vp)`, `output_view() -> &TextureView`

- [ ] **Step 3: Verify and commit**

---

### Task 2: Spatial Denoising (A-trous)

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/denoise-spatial.wgsl`
- Create: `kansei-core/src/pathtracer/spatial_denoise.rs`

Port from `src/pathtracer/shaders/denoise-spatial.wgsl.ts`.

A-trous wavelet filter with variance guidance. Multiple iterations with increasing step size.

- [ ] **Step 1: Port denoise-spatial.wgsl**

5x5 kernel, step size = 2^iteration. Weights based on:
- Color distance (luminance)
- Normal distance
- Depth distance
- Variance

- [ ] **Step 2: Create spatial_denoise.rs**

```rust
pub struct SpatialDenoise {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params_buf: wgpu::Buffer,
    scratch_tex: Option<wgpu::Texture>,
    iterations: u32,  // default 3
    device: wgpu::Device,
    queue: wgpu::Queue,
}
```

Methods: `new(&Renderer)`, `resize(w, h)`, `denoise(encoder, input_view, depth_view, normal_view, moments_view) -> &TextureView`

Runs N iterations, ping-ponging between input and scratch textures.

- [ ] **Step 3: Verify and commit**

---

### Task 3: ReSTIR Direct Illumination

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/restir-di.wgsl`
- Create: `kansei-core/src/pathtracer/restir.rs`

Port from `src/pathtracer/shaders/restir-di.wgsl.ts`.

Reservoir-based importance sampling for direct lighting. Two passes: generation + spatial reuse.

- [ ] **Step 1: Port restir-di.wgsl**

Two entry points:
- `restir_generate` — initial reservoir generation from light samples
- `restir_spatial` — spatial reuse from neighbor reservoirs + final shading

Reservoir struct stores: light index, weight, sample count, M (history length).

- [ ] **Step 2: Create restir.rs**

```rust
pub struct ReSTIR {
    generate_pipeline: wgpu::ComputePipeline,
    spatial_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    reservoir_a: Option<wgpu::Buffer>,  // ping
    reservoir_b: Option<wgpu::Buffer>,  // pong
    output_tex: Option<wgpu::Texture>,
    ping: bool,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
```

Methods: `new(&Renderer)`, `resize(w, h)`, `run(encoder, depth_view, normal_view, bvh_data, lights_buf, params)`

- [ ] **Step 3: Verify and commit**

---

### Task 4: Irradiance Probe Grid

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/probe-trace.wgsl`
- Create: `kansei-core/src/pathtracer/shaders/probe-update.wgsl`
- Create: `kansei-core/src/pathtracer/probes.rs`

Port from `src/pathtracer/shaders/probe-trace.wgsl.ts` and `probe-update.wgsl.ts`.

L2 Spherical Harmonics probe grid for indirect diffuse fallback.

- [ ] **Step 1: Port probe-trace.wgsl**

Traces rays from probe positions using spherical Fibonacci sampling. Stores ray hit radiance + direction.

- [ ] **Step 2: Port probe-update.wgsl**

Projects ray results onto SH basis (9 coefficients per probe). Temporal hysteresis blending with previous SH data.

- [ ] **Step 3: Create probes.rs**

```rust
pub struct ProbeGrid {
    trace_pipeline: wgpu::ComputePipeline,
    update_pipeline: wgpu::ComputePipeline,
    sh_buf_a: Option<wgpu::Buffer>,  // ping
    sh_buf_b: Option<wgpu::Buffer>,  // pong
    ray_results_buf: Option<wgpu::Buffer>,
    grid_dims: [u32; 3],
    grid_min: [f32; 3],
    grid_step: [f32; 3],
    probe_count: u32,
    rays_per_frame: u32,
    hysteresis: f32,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
```

Methods: `new(&Renderer)`, `configure(bounds_min, bounds_max, spacing)`, `update(encoder, bvh_data, tlas_buf, frame_index)`, `sh_buffer() -> &Buffer`

- [ ] **Step 4: Verify and commit**

---

### Task 5: Composite Shader

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/composite.wgsl`
- Create: `kansei-core/src/pathtracer/compositor.rs`

Port from `src/pathtracer/shaders/composite.wgsl.ts`.

Final compositing: GI × albedo + direct light + emissive.

- [ ] **Step 1: Port composite.wgsl**

Reads GI texture, albedo from GBuffer, direct light texture. Outputs final color.

- [ ] **Step 2: Create compositor.rs**

Simple compute pass with bind group for inputs + output.

- [ ] **Step 3: Verify and commit**

---

## Post-Plan Notes

### What this produces:
- SVGF temporal denoising with motion-compensated reprojection
- A-trous spatial denoising with variance guidance
- ReSTIR direct illumination (reservoir resampling)
- L2 SH irradiance probe grid
- Final composite shader

### Next:
- **Plan D: Integration** — PathTracerEffect wrapping everything as a PostProcessingEffect
