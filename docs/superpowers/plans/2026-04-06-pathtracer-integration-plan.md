# Path Tracer Plan D: Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create PathTracerEffect that wraps all path tracer components (BVH, trace, denoise, ReSTIR, probes, composite) into a single PostProcessingEffect, matching the TS PathTracerEffect class.

**Architecture:** PathTracerEffect implements PostProcessingEffect trait. On first render: builds BVH from scene. Each frame: updates TLAS, traces rays, denoises (temporal + spatial), composites with GBuffer. Configurable: SPP, max bounces, denoise passes, ReSTIR, probes.

**Tech Stack:** Rust, wgpu 24

---

### Task 1: Create PathTracerEffect

**Files:**
- Create: `kansei-core/src/pathtracer/effect.rs`
- Modify: `kansei-core/src/pathtracer/mod.rs`

```rust
pub struct PathTracerEffect {
    bvh: BVHBuilder,
    tlas: TLASBuilder,
    tracer: PathTracer,
    temporal: TemporalDenoise,
    spatial: SpatialDenoise,
    restir: Option<ReSTIR>,
    probes: Option<ProbeGrid>,
    compositor: Compositor,
    gpu_data: Option<GPUBVHData>,
    blas_built: bool,
    prev_vp: [f32; 16],
    // Settings
    pub spp: u32,
    pub max_bounces: u32,
    pub spatial_passes: u32,
    pub use_restir: bool,
    pub use_probes: bool,
    pub trace_scale: f32,
    pub temporal_blend: f32,
}
```

Implements `PostProcessingEffect`:
```rust
impl PostProcessingEffect for PathTracerEffect {
    fn initialize(&mut self, device: &wgpu::Device, gbuffer: &GBuffer, camera: &Camera) {
        // First-time BVH build happens here or is deferred
    }

    fn render(&mut self, device, queue, encoder, input, depth, output, camera, width, height) {
        // 1. Build TLAS (if scene changed)
        // 2. Update probes (if enabled)
        // 3. Run ReSTIR (if enabled)
        // 4. Trace rays
        // 5. Temporal denoise
        // 6. Spatial denoise
        // 7. Composite to output
        // 8. Update prev_vp for next frame
    }

    fn resize(&mut self, width, height, gbuffer: &GBuffer) {
        self.tracer.resize(width, height);
        self.temporal.resize(width, height);
        self.spatial.resize(width, height);
        if let Some(ref mut r) = self.restir { r.resize(width, height); }
    }

    fn destroy(&mut self) {}
}
```

Constructor:
```rust
pub fn new(renderer: &Renderer, scene: &Scene) -> Self {
    let mut bvh = BVHBuilder::new();
    let mut tlas = TLASBuilder::new(renderer);
    let gpu_data = bvh.build_full(renderer, scene, &mut tlas);
    let tracer = PathTracer::new(renderer);
    let temporal = TemporalDenoise::new(renderer);
    let spatial = SpatialDenoise::new(renderer);
    let compositor = Compositor::new(renderer);
    
    Self {
        bvh, tlas, tracer, temporal, spatial,
        restir: None, probes: None, compositor,
        gpu_data: Some(gpu_data),
        blas_built: true,
        prev_vp: [0.0; 16],
        spp: 1, max_bounces: 4, spatial_passes: 3,
        use_restir: false, use_probes: false,
        trace_scale: 1.0, temporal_blend: 0.1,
    }
}
```

- [ ] **Step 1: Create effect.rs**
- [ ] **Step 2: Update mod.rs**
- [ ] **Step 3: Verify and commit**

---

### Task 2: Update pathtracer_test example

**Files:**
- Modify: `kansei-native/examples/pathtracer_test.rs`

Update to use PathTracerEffect through PostProcessingVolume instead of manual pipeline.

```rust
let pt_effect = PathTracerEffect::new(&renderer, &scene);
let mut volume = PostProcessingVolume::new(&renderer, vec![Box::new(pt_effect)]);

// Per frame:
volume.render(&camera, &canvas_view, width, height);
```

- [ ] **Step 1: Update example**
- [ ] **Step 2: Verify and commit**

---

### Task 3: Final verification

- [ ] **Step 1: Build all examples**
```bash
cargo build --examples
```

- [ ] **Step 2: Build WASM**
```bash
cd kansei-wasm && wasm-pack build --target web
```

- [ ] **Step 3: Commit final state**

---

## What the full path tracer port produces:

**BVH Infrastructure (Plan A):**
- CPU SAH BLAS construction + BVH4 collapse
- GPU radix sort, Morton codes, TLAS BVH4 build

**Core Tracing (Plan B):**
- intersection.wgsl + traversal.wgsl
- trace.wgsl with GGX + MIS + transmission
- PathTracer compute pipeline

**Denoising + Advanced (Plan C):**
- SVGF temporal denoiser
- A-trous spatial denoiser
- ReSTIR direct illumination
- L2 SH irradiance probes
- Composite shader

**Integration (Plan D):**
- PathTracerEffect as PostProcessingEffect
- Full pipeline: BVH → trace → denoise → composite
