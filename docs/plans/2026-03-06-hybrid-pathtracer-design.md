# Hybrid Path Tracer Design

## Goals

- Full global illumination: diffuse + specular + AO + refraction
- Real-time 30+ fps at 1-2 SPP with temporal accumulation and spatial denoising
- General-purpose: works with any scene geometry
- Supports instanced meshes driven by compute simulations (e.g. 300K particles)
- Per-material refraction with configurable bounce count (default 2)

## Architecture Overview

Rasterization handles primary visibility (existing pipeline). A compute-based path tracer handles all indirect lighting, reflections, and refractions by tracing rays into a two-level BVH built on the GPU.

```
Per frame:

 Compute Sim ──▶ TLAS Build ──▶ GI + Refraction Trace
 (positions)     (GPU, full      (compute, 1-2 rays/pixel)
                  rebuild)                │
                                          ▼
 Rasterize ────▶ GBuffer ──────▶ Denoise (temporal + spatial)
 (opaque)        color, depth,            │
                 normal, albedo           ▼
                                  Composite (direct + indirect + refraction)
```

## Two-Level BVH (TLAS/BLAS)

### BLAS (Bottom-Level Acceleration Structure)

One BLAS per unique base geometry (e.g. SphereGeometry, BoxGeometry). Built on GPU via compute shaders. Rebuilt only when geometry changes.

**Build pipeline (compute):**
1. Calculate per-triangle AABBs and centroids
2. Compute Morton codes from centroids (normalized to scene AABB)
3. Radix sort Morton codes (parallel GPU radix sort)
4. Build binary radix tree (Karras 2012 algorithm)
5. Compute internal node AABBs (bottom-up pass)

**GPU data layout:**

```
BLASTriangleBuffer: storage<read>
  Per triangle (stride TBD, ~96 bytes):
    v0: vec3f, v1: vec3f, v2: vec3f    (positions)
    n0: vec3f, n1: vec3f, n2: vec3f    (normals)
    materialIndex: u32

BLASNodeBuffer: storage<read>
  Per node (32 bytes):
    boundsMin: vec3f
    leftOrTriOffset: i32    (>= 0: internal left child, < 0: leaf tri offset)
    boundsMax: vec3f
    rightOrTriCount: i32    (>= 0: internal right child, < 0: leaf tri count)
```

### TLAS (Top-Level Acceleration Structure)

One entry per renderable instance. Rebuilt fully every frame on GPU.

**Instance data:**
```
TLASInstanceBuffer: storage<read>
  Per instance (80 bytes):
    transformRow0: vec4f     (world matrix rows 0-2, affine 3x4)
    transformRow1: vec4f
    transformRow2: vec4f
    invTransformRow0: vec4f  (inverse, for transforming rays to local space)
    invTransformRow1: vec4f
    invTransformRow2: vec4f  (reduced: only need 3x3 inverse + translation)
    blasIndex: u32           (which BLAS this instance references)
    materialIndex: u32       (into MaterialBuffer)
    padding: vec2u
```

**TLAS node buffer:** Same format as BLAS nodes but indices reference instances, not triangles.

**Build pipeline:** Same as BLAS (Morton sort + radix tree), but operates on instance AABBs. Reads instance positions directly from compute simulation buffers (zero-copy for instanced geometry).

### Instance Transform Source

For standard Renderables: world matrix from the renderer's shared matrix buffer.

For instanced geometry (InstancedGeometry with ComputeBuffer extraBuffers): read instance positions from the same storage buffer the compute simulation writes to and the vertex shader reads from. The TLAS builder reads these positions, computes per-instance world-space AABBs (base geometry AABB transformed by instance position/transform), and feeds them into the sort+build pipeline.

## Ray Traversal

Two-level traversal in the trace compute shader:

```wgsl
fn traceRay(origin: vec3f, direction: vec3f) -> HitInfo {
    // 1. Traverse TLAS
    var closestT = FAR;
    var closestInstance = -1;
    var closestLocalHit: LocalHitInfo;

    // Stack-based TLAS traversal
    for each TLAS node hit by ray:
        if leaf:
            let inst = instances[node.instanceIndex];
            // 2. Transform ray to instance local space
            let localOrigin = inst.invTransform * origin;
            let localDir = inst.invTransform * direction;
            // 3. Traverse instance's BLAS
            let localHit = traverseBLAS(blas[inst.blasIndex], localOrigin, localDir);
            if localHit.t < closestT:
                closestT = localHit.t;
                closestInstance = node.instanceIndex;
                closestLocalHit = localHit;

    // 4. Transform hit back to world space
    if closestInstance >= 0:
        return transformHitToWorld(closestLocalHit, instances[closestInstance]);
    return miss;
}
```

Stack size: 32 entries for TLAS + 32 for BLAS (64 total) should handle scenes with millions of instances.

## GI Trace Pass (Compute)

**Input:** GBuffer (position from depth, world normal, albedo, roughness, metallic), BVH buffers, material table, light data.

**Output:** `giTexture` (rgba16float) — indirect radiance per pixel.

**Per pixel:**
1. Reconstruct world position and normal from GBuffer
2. Sample hemisphere direction (cosine-weighted for diffuse, GGX for specular)
3. Trace ray through TLAS/BLAS
4. At hit: evaluate direct lighting (shadow ray to lights) + material BRDF
5. For refractive hits: bend ray via Snell's law, continue tracing up to `material.maxBounces`
6. Accumulate radiance along the path

**Ray budget:** 1 indirect ray + 1 shadow ray per pixel for diffuse. 1 specular ray for metallic/glossy pixels. Refraction rays up to `maxBounces` for refractive materials.

**Workgroup:** 8x8 threads, dispatch (width/8, height/8).

## Refraction

Handled inline during the GI trace. When a ray hits a refractive surface:

```
for bounce = 0 to material.maxBounces:
    compute refracted direction (Snell's law using material.ior)
    handle total internal reflection (switch to reflection)
    trace refracted ray through BVH
    accumulate absorption: throughput *= exp(-material.absorptionColor * material.absorptionDensity * hitDistance)
    if hit another refractive surface: continue loop
    if hit opaque surface: shade and break
    if miss: sample environment and break
```

Refractive pixels are identified in the GBuffer by a material flag. The trace shader handles them with a variable-length bounce loop capped by `maxBounces`.

## Denoiser

### Temporal Accumulation

- Reproject previous frame's GI using depth + camera matrices (same pattern as FroxelGrid temporal blend)
- Blend factor: `mix(current, history, 0.9)` — 90% history reuse when camera/scene is stable
- Discard history on disocclusion: detected by depth difference > threshold or normal angle > threshold
- History stored in ping-pong textures (rgba16float)

### Spatial Filter (SVGF-inspired)

- Edge-aware a-trous wavelet filter
- Edge-stop functions: depth difference, normal angle, luminance variance
- 3-4 iterations with progressively wider kernel (5x5 base, doubling stride each iteration)
- Separate passes for diffuse and specular (specular uses tighter kernel)
- Refraction output is not spatially denoised (coherent enough at 1 ray/pixel)

### Denoiser data:
- `giHistoryTexture[2]` (ping-pong, rgba16float) — temporal accumulation
- `giDenoisedTexture` (rgba16float) — final denoised indirect light
- `momentTexture` (rg16float) — first and second moment for variance estimation

## Composite Pass (Compute)

Combines rasterized direct light with denoised indirect light and refraction:

```wgsl
let direct = textureLoad(colorTex, coord, 0).rgb;       // rasterized output
let indirect = textureLoad(giDenoisedTex, coord, 0).rgb; // denoised GI
let albedo = textureLoad(albedoTex, coord, 0).rgb;

// Direct already includes albedo*lighting from rasterizer.
// Indirect is raw incoming radiance — multiply by albedo here.
let final = direct + albedo * indirect;

// For refractive pixels, replace direct entirely with traced result.
if (isRefractive(coord)) {
    final = refractionResult;
}

textureStore(outputTex, coord, vec4f(final, 1.0));
```

## GBuffer Extensions

Current GBuffer has: `colorTexture` (rgba16float), `depthTexture` (depth32float), `emissiveTexture` (rgba16float).

**New textures:**

| Texture | Format | Content |
|---------|--------|---------|
| `normalTexture` | rgba16float | world normal.xyz + roughness in .w |
| `albedoTexture` | rgba8unorm | base color.rgb + metallic in .a |

These require extending the MRT output from 2 targets to 4 targets in `renderToGBuffer()`. Shaders that opt into path tracing write to all 4 locations. The GBuffer bundle and pipeline layout accommodate the extra targets.

## PathTracerMaterial (Data Class)

```typescript
class PathTracerMaterial {
    albedo: [number, number, number] = [1, 1, 1];
    roughness: number = 1.0;
    metallic: number = 0.0;
    ior: number = 1.0;
    maxBounces: number = 2;
    absorptionColor: [number, number, number] = [1, 1, 1];
    absorptionDensity: number = 0.0;
    emissive: [number, number, number] = [0, 0, 0];
    refractive: boolean = false;
}
```

Packed into a flat `MaterialBuffer` (storage<read>) for GPU access. Each Renderable references its material by index.

## New Files

| File | Type | Purpose |
|------|------|---------|
| `src/pathtracer/BVHBuilder.ts` | Class | GPU BLAS/TLAS construction and management |
| `src/pathtracer/PathTracerEffect.ts` | Class | PostProcessingEffect: trace + denoise + composite |
| `src/pathtracer/PathTracerMaterial.ts` | Class | PBR material data for GPU material table |
| `src/pathtracer/shaders/bvh-build.wgsl` | Shader | Morton codes, radix sort, tree build, AABB refit |
| `src/pathtracer/shaders/trace.wgsl` | Shader | Ray-BVH traversal, shading, refraction |
| `src/pathtracer/shaders/denoise-temporal.wgsl` | Shader | Temporal reprojection and accumulation |
| `src/pathtracer/shaders/denoise-spatial.wgsl` | Shader | A-trous wavelet spatial filter |
| `src/pathtracer/shaders/composite.wgsl` | Shader | Final direct + indirect + refraction merge |

## Modified Files

| File | Change |
|------|--------|
| `src/postprocessing/GBuffer.ts` | Add normalTexture, albedoTexture; extend resize/destroy |
| `src/renderers/Renderer.ts` | Extend renderToGBuffer MRT to 4 targets; extend bundle for 4 color targets |
| `src/objects/Renderable.ts` | Add optional `pathTracerMaterial` property |
| `src/main.ts` | Export new classes |

## Compute Pass Summary

| Pass | Workgroup | Dispatch | Frequency |
|------|-----------|----------|-----------|
| BLAS build (per geometry) | 256x1x1 | ceil(triCount/256) | On geometry change |
| TLAS instance AABB | 256x1x1 | ceil(instanceCount/256) | Every frame |
| TLAS Morton + sort | 256x1x1 | ceil(instanceCount/256) × sort passes | Every frame |
| TLAS tree build | 256x1x1 | ceil(instanceCount/256) | Every frame |
| TLAS AABB refit | 256x1x1 | ceil(nodeCount/256) | Every frame |
| GI trace | 8x8x1 | (width/8, height/8) | Every frame |
| Denoise temporal | 8x8x1 | (width/8, height/8) | Every frame |
| Denoise spatial (×3-4) | 8x8x1 | (width/8, height/8) | Every frame |
| Composite | 8x8x1 | (width/8, height/8) | Every frame |
