# Path Tracer Plan A: BVH Infrastructure

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the BVH construction system from TypeScript to Rust — triangle packing, SAH-based BLAS construction (CPU), binary-to-BVH4 conversion, radix sort (GPU), and TLAS building (GPU). This is the foundation all path tracing depends on.

**Architecture:** 1:1 port of BVHBuilder.ts. Triangle buffer (96 bytes/tri), binary BVH built via SAH on CPU, collapsed to BVH4 (128 bytes/node). Instance buffer (112 bytes/instance). TLAS built per-frame on GPU via Morton sort + bottom-up BVH4 construction. All buffer layouts match TS exactly for shader compatibility.

**Tech Stack:** Rust, wgpu 24, bytemuck 1

---

### Task 1: Triangle and Instance Buffer Structures

**Files:**
- Create: `kansei-core/src/pathtracer/mod.rs`
- Create: `kansei-core/src/pathtracer/buffers.rs`
- Modify: `kansei-core/src/lib.rs`

Define the GPU buffer layouts matching TS exactly.

- [ ] **Step 1: Create pathtracer module**

`kansei-core/src/pathtracer/mod.rs`:
```rust
mod buffers;
pub use buffers::*;
```

Add `pub mod pathtracer;` to `lib.rs`.

- [ ] **Step 2: Define buffer structs in buffers.rs**

Triangle: 24 floats = 96 bytes
```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PackedTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub n0: [f32; 3],
    pub n1: [f32; 3],
    pub n2: [f32; 3],
    pub material_index: u32,
    pub _pad: [u32; 5],
}
```

BVH4 Node: 32 floats = 128 bytes
```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct BVH4Node {
    pub child_min_x: [f32; 4],
    pub child_max_x: [f32; 4],
    pub child_min_y: [f32; 4],
    pub child_max_y: [f32; 4],
    pub child_min_z: [f32; 4],
    pub child_max_z: [f32; 4],
    pub children: [i32; 4],    // negative = leaf (-(triStart+1)), positive = node index
    pub tri_counts: [u32; 4],  // triangle count per leaf, 0 for internal
}
```

Instance: 28 floats = 112 bytes
```rust
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct PackedInstance {
    pub transform: [[f32; 4]; 3],       // 3x4 affine (row-major in GPU)
    pub inv_transform: [[f32; 4]; 3],   // inverse 3x4
    pub blas_node_offset: u32,
    pub blas_tri_offset: u32,
    pub blas_tri_count: u32,
    pub material_index: u32,
}
```

Binary BVH Node (CPU construction intermediate): 8 floats
```rust
#[repr(C)]
#[derive(Clone, Copy)]
pub struct BinaryBVHNode {
    pub bounds_min: [f32; 3],
    pub left: i32,      // left child index (or tri_start for leaf)
    pub bounds_max: [f32; 3],
    pub right: i32,     // right child index (or tri_count for leaf)
}
```

- [ ] **Step 3: Verify and commit**

---

### Task 2: SAH-Based BLAS Construction (CPU)

**Files:**
- Create: `kansei-core/src/pathtracer/bvh_builder.rs`
- Modify: `kansei-core/src/pathtracer/mod.rs`

CPU-side SAH BVH construction matching TS `_buildCPUBLAS`.

- [ ] **Step 1: Implement BVHBuilder struct**

```rust
pub struct BVHBuilder {
    pub triangles: Vec<PackedTriangle>,
    pub bvh4_nodes: Vec<BVH4Node>,
    pub instances: Vec<PackedInstance>,
    binary_nodes: Vec<BinaryBVHNode>,
    // Per-geometry BLAS info
    blas_entries: Vec<BLASEntry>,
}

struct BLASEntry {
    node_offset: u32,
    tri_offset: u32,
    tri_count: u32,
}
```

- [ ] **Step 2: Implement SAH build**

```rust
const SAH_BINS: usize = 12;
const MAX_LEAF_SIZE: usize = 4;
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 1.0;
```

Methods:
- `add_geometry(vertices, indices, normals, uvs, material_index)` — pack triangles
- `build_blas()` — for each geometry, build binary BVH via recursive SAH partition
- `sah_partition(tris, start, end, centroids, bounds)` — find best split axis/position using 12 bins
- `collapse_to_bvh4()` — convert binary tree to 4-wide BVH4 nodes
- `build_tlas_instances(scene_renderables)` — pack instance transforms

The SAH algorithm:
1. For each axis (X, Y, Z):
   - Compute centroid bounds
   - Assign triangles to bins based on centroid position
   - Sweep bins left-to-right accumulating counts and bounds
   - Compute SAH cost = traversal + (left_area * left_count + right_area * right_count) / parent_area
2. Pick axis + bin with lowest cost
3. If cost >= leaf cost (count * intersection), make leaf
4. Otherwise partition and recurse

- [ ] **Step 3: Implement BVH4 collapse**

Convert binary tree to 4-wide:
- For each internal node, absorb grandchildren to get up to 4 direct children
- Leaf nodes: store negative child index (-(triStart+1)) and tri_count
- Sentinel children: children=-1, tri_counts=0

- [ ] **Step 4: Verify and commit**

---

### Task 3: Radix Sort (GPU)

**Files:**
- Create: `kansei-core/src/pathtracer/radix_sort.rs`
- Create: `kansei-core/src/pathtracer/shaders/radix-sort.wgsl`
- Modify: `kansei-core/src/pathtracer/mod.rs`

GPU-based 4-bit radix sort for Morton code sorting of TLAS instances.

- [ ] **Step 1: Port radix-sort.wgsl**

Three entry points: `histogram`, `prefix_sum`, `scatter`.
- Workgroup size: 256
- 16 bins per 4-bit digit
- 8 passes for 32-bit keys
- Uses per-pass parameter buffer (bit_offset)

IMPORTANT: each of the 8 passes needs its own parameter buffer (writeBuffer overwrite issue from MEMORY.md).

- [ ] **Step 2: Create RadixSort struct**

```rust
pub struct RadixSort {
    histogram_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    param_buffers: Vec<wgpu::Buffer>,  // 8 separate buffers
    // ... bind groups, temp buffers
}
```

Methods:
- `new(renderer)` — create pipelines
- `sort(encoder, queue, keys_buf, values_buf, count)` — dispatch 8 passes

- [ ] **Step 3: Verify and commit**

---

### Task 4: Morton Code Generation (GPU)

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/morton.wgsl`
- Create: `kansei-core/src/pathtracer/shaders/tlas-sort.wgsl`

- [ ] **Step 1: Port morton.wgsl**

3D position to 30-bit Morton code (10 bits per axis). Bit-interleaving via expandBits().

- [ ] **Step 2: Port tlas-sort.wgsl**

Compute Morton codes for all instances based on their world-space AABB centroids, then prepare for radix sort.

- [ ] **Step 3: Verify and commit**

---

### Task 5: TLAS BVH4 Build (GPU)

**Files:**
- Create: `kansei-core/src/pathtracer/shaders/tlas-build.wgsl`
- Create: `kansei-core/src/pathtracer/tlas_builder.rs`

- [ ] **Step 1: Port tlas-build.wgsl**

Two entry points:
- `buildLeaves` — groups 4 consecutive Morton-sorted instances into BVH4 leaf nodes
- `buildInternal` — groups 4 consecutive child nodes into parent nodes, bottom-up

- [ ] **Step 2: Create TLASBuilder struct**

```rust
pub struct TLASBuilder {
    sort: RadixSort,
    morton_pipeline: wgpu::ComputePipeline,
    leaf_pipeline: wgpu::ComputePipeline,
    internal_pipeline: wgpu::ComputePipeline,
    // buffers: morton codes, sorted indices, TLAS BVH4 nodes
    tlas_nodes_buf: wgpu::Buffer,
}
```

Methods:
- `new(renderer)` — create pipelines
- `build(encoder, queue, instances_buf, count, bounds)` — full TLAS build pipeline:
  1. Compute Morton codes
  2. Radix sort by Morton code
  3. Build leaves
  4. Build internal levels (bottom-up)

- [ ] **Step 3: Verify and commit**

---

### Task 6: Integration + Validation

**Files:**
- Modify: `kansei-core/src/pathtracer/mod.rs`
- Create: `kansei-native/examples/bvh_test.rs` (optional)

- [ ] **Step 1: Wire up BVHBuilder with TLASBuilder**

```rust
impl BVHBuilder {
    pub fn build_full(&mut self, renderer: &Renderer, scene: &Scene) {
        // 1. Pack triangles from scene renderables
        // 2. Build BLAS (CPU SAH)
        // 3. Collapse to BVH4
        // 4. Pack instances
        // 5. Upload to GPU
        // 6. Build TLAS (GPU)
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cargo check -p kansei-core
```

- [ ] **Step 3: Commit**

---

## Post-Plan Notes

### What this produces:
- PackedTriangle, BVH4Node, PackedInstance buffer structs
- CPU SAH BLAS construction with BVH4 collapse
- GPU radix sort (8-pass, 4-bit)
- GPU Morton code generation
- GPU TLAS BVH4 bottom-up construction
- BVHBuilder orchestrating the full pipeline

### Next plans:
- **Plan B: Path Tracer Core** — Ray generation, traversal, intersection, BSDF, basic tracing
- **Plan C: Denoising + Advanced** — SVGF temporal/spatial, ReSTIR, probes, voxels
- **Plan D: Integration** — PathTracerEffect, material system, blue noise, compositing
