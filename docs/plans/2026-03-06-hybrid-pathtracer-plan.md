# Hybrid Path Tracer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a GPU-based hybrid path tracer to the kansei engine — rasterization for primary visibility, compute-shader path tracing for full GI (diffuse, specular, AO, refraction) with temporal + spatial denoising.

**Architecture:** Two-level BVH (TLAS/BLAS) built entirely on GPU via compute shaders. BLAS per unique geometry (built once), TLAS rebuilt every frame from instance transforms. PathTracerEffect is a PostProcessingEffect that reads the GBuffer and traces 1-2 indirect rays per pixel through the BVH, denoises temporally and spatially, then composites with the rasterized direct lighting.

**Tech Stack:** WebGPU compute shaders (WGSL), TypeScript, existing engine patterns (PostProcessingEffect, ComputeBuffer, GBuffer).

**Reference:** `docs/plans/2026-03-06-hybrid-pathtracer-design.md`

---

## Phase 1: Foundation

### Task 1: PathTracerMaterial data class

**Files:**
- Create: `src/pathtracer/PathTracerMaterial.ts`

**Step 1: Create the data class**

```typescript
export class PathTracerMaterial {
    public albedo: [number, number, number] = [1, 1, 1];
    public roughness: number = 1.0;
    public metallic: number = 0.0;
    public ior: number = 1.0;
    public maxBounces: number = 2;
    public absorptionColor: [number, number, number] = [1, 1, 1];
    public absorptionDensity: number = 0.0;
    public emissive: [number, number, number] = [0, 0, 0];
    public emissiveIntensity: number = 0.0;
    public refractive: boolean = false;

    /** Byte size of the packed GPU struct (std140-aligned). */
    static readonly GPU_STRIDE = 64; // see packInto()

    /**
     * Pack this material into a Float32Array at the given float offset.
     * GPU layout (64 bytes = 16 floats):
     *   [0-2]  albedo.rgb        [3]  roughness
     *   [4]    metallic          [5]  ior           [6] maxBounces(f32)  [7] flags
     *   [8-10] absorptionColor   [11] absorptionDensity
     *   [12-14] emissive         [15] emissiveIntensity
     */
    public packInto(target: Float32Array, offset: number): void {
        target[offset + 0]  = this.albedo[0];
        target[offset + 1]  = this.albedo[1];
        target[offset + 2]  = this.albedo[2];
        target[offset + 3]  = this.roughness;
        target[offset + 4]  = this.metallic;
        target[offset + 5]  = this.ior;
        target[offset + 6]  = this.maxBounces;
        target[offset + 7]  = (this.refractive ? 1 : 0);
        target[offset + 8]  = this.absorptionColor[0];
        target[offset + 9]  = this.absorptionColor[1];
        target[offset + 10] = this.absorptionColor[2];
        target[offset + 11] = this.absorptionDensity;
        target[offset + 12] = this.emissive[0];
        target[offset + 13] = this.emissive[1];
        target[offset + 14] = this.emissive[2];
        target[offset + 15] = this.emissiveIntensity;
    }
}
```

**Step 2: Add property to Renderable**

In `src/objects/Renderable.ts`, add after the `receiveShadow` property:

```typescript
import { PathTracerMaterial } from "../pathtracer/PathTracerMaterial";

/** Path tracer material properties. If null, defaults are derived at BVH build time. */
public pathTracerMaterial: PathTracerMaterial | null = null;
```

**Step 3: Export from main.ts**

Add to `src/main.ts`:
```typescript
export { PathTracerMaterial } from "./pathtracer/PathTracerMaterial";
```

**Step 4: Verify**

Run: `tsc --noEmit`
Expected: No errors.

**Step 5: Commit**

```bash
git add src/pathtracer/PathTracerMaterial.ts src/objects/Renderable.ts src/main.ts
git commit -m "feat(pathtracer): add PathTracerMaterial data class"
```

---

### Task 2: Extend GBuffer with normal and albedo textures

**Files:**
- Modify: `src/postprocessing/GBuffer.ts`

The GBuffer currently has `colorTexture`, `depthTexture`, `emissiveTexture` (+ MSAA variants), `outputTexture`, `pingPongTexture`. We add `normalTexture` (world normal.xyz + roughness) and `albedoTexture` (base color.rgb + metallic).

**Step 1: Add texture properties**

After `emissiveTexture` (line ~32), add:
```typescript
public normalTexture!: GPUTexture;
public albedoTexture!: GPUTexture;
public normalMSAATexture: GPUTexture | null = null;
public albedoMSAATexture: GPUTexture | null = null;
```

**Step 2: Create textures in `_create()`**

After the emissiveTexture creation block, add:
```typescript
this.normalTexture = this.device.createTexture({
    label: 'GBuffer/Normal',
    size: [this.width, this.height],
    format: 'rgba16float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});

this.albedoTexture = this.device.createTexture({
    label: 'GBuffer/Albedo',
    size: [this.width, this.height],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
});
```

In the MSAA block (where `colorMSAATexture` is created), add:
```typescript
this.normalMSAATexture = this.device.createTexture({
    label: 'GBuffer/NormalMSAA',
    size: [this.width, this.height],
    format: 'rgba16float',
    sampleCount: this.msaaSampleCount,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});

this.albedoMSAATexture = this.device.createTexture({
    label: 'GBuffer/AlbedoMSAA',
    size: [this.width, this.height],
    format: 'rgba8unorm',
    sampleCount: this.msaaSampleCount,
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
});
```

**Step 3: Destroy the new textures in `destroy()`**

Add to the destroy method:
```typescript
this.normalTexture?.destroy();
this.albedoTexture?.destroy();
this.normalMSAATexture?.destroy();
this.normalMSAATexture = null;
this.albedoMSAATexture?.destroy();
this.albedoMSAATexture = null;
```

**Step 4: Verify**

Run: `tsc --noEmit`
Expected: No errors.

**Step 5: Commit**

```bash
git add src/postprocessing/GBuffer.ts
git commit -m "feat(gbuffer): add normal and albedo render targets"
```

---

### Task 3: Extend Renderer MRT to support 4 color targets

**Files:**
- Modify: `src/renderers/Renderer.ts`
- Modify: `src/materials/Material.ts`

Currently `renderToGBuffer()` uses `colorTargetCount: 2` (color + emissive). We need to support 4 (color + emissive + normal + albedo).

**Step 1: Update `_buildPipeline` in Material.ts**

In `_buildPipeline()` (around line 127), the targets array currently handles 1 or 2 targets. Replace the target-building logic:

```typescript
const targets: GPUColorTargetState[] = [colorTarget];
for (let t = 1; t < colorTargetCount; t++) {
    targets.push({ format: colorFormat });
}
```

But the albedo target uses `rgba8unorm` while the others use `rgba16float`. To handle this, add an optional `colorFormats` parameter to `_buildPipeline` and `getPipelineForConfig`:

```typescript
private _buildPipeline(
    gpuDevice: GPUDevice,
    vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>,
    colorFormat: GPUTextureFormat,
    sampleCount: number,
    depthFormat: GPUTextureFormat,
    colorTargetCount: number = 1,
    colorFormats?: GPUTextureFormat[]  // NEW
): GPURenderPipeline {
```

Build targets using `colorFormats` when provided:
```typescript
const targets: GPUColorTargetState[] = [];
if (colorFormats && colorFormats.length > 0) {
    for (const fmt of colorFormats) {
        const target: GPUColorTargetState = { format: fmt };
        if (targets.length === 0 && this.transparent) {
            target.blend = { /* existing blend config */ };
        }
        targets.push(target);
    }
} else {
    targets.push(colorTarget);
    for (let t = 1; t < colorTargetCount; t++) {
        targets.push({ format: colorFormat });
    }
}
```

Add the `colorFormats` parameter to `getPipelineForConfig` and thread it through to `_buildPipeline`. Update the cache key to include the formats:
```typescript
const key = colorFormats
    ? `${colorFormats.join(',')}:${sampleCount}:${depthFormat}`
    : `${colorFormat}:${sampleCount}:${depthFormat}:${colorTargetCount}`;
```

**Step 2: Update `renderToGBuffer()` in Renderer.ts**

Change the `getPipelineForConfig` call to pass 4 MRT formats:
```typescript
const mrtFormats: GPUTextureFormat[] = [
    'rgba16float',  // color
    'rgba16float',  // emissive
    'rgba16float',  // normal
    'rgba8unorm',   // albedo
];
renderable.material.getPipelineForConfig(
    this.device!,
    renderable.geometry.vertexBuffersDescriptors,
    'rgba16float',
    gbuffer.msaaSampleCount,
    'depth32float',
    4,
    mrtFormats
);
```

**Step 3: Update render pass descriptor in `renderToGBuffer()`**

Add color attachments for normal and albedo. In the MSAA path:
```typescript
colorAttachments: [
    { /* color - existing */ },
    { /* emissive - existing */ },
    {
        view: gbuffer.normalMSAATexture!.createView(),
        resolveTarget: gbuffer.normalTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'discard',
    },
    {
        view: gbuffer.albedoMSAATexture!.createView(),
        resolveTarget: gbuffer.albedoTexture.createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 0 },
        loadOp: 'clear',
        storeOp: 'discard',
    },
],
```

Non-MSAA path similarly — 4 color attachments with direct views.

**Step 4: Update `_buildRenderBundle()` color formats**

The `_buildRenderBundle` method creates a `GPURenderBundleEncoder` with `colorFormats`. Add an optional `colorFormats` parameter:

```typescript
private _buildRenderBundle(
    orderedObjects: Renderable[],
    cameraBindGroup: GPUBindGroup,
    colorFormat: GPUTextureFormat = this._presentationFormat!,
    sampleCount: number = this.sampleCount,
    depthFormat: GPUTextureFormat = 'depth24plus',
    colorTargetCount: number = 1,
    colorFormats?: GPUTextureFormat[]  // NEW
): GPURenderBundle {
    const formats: (GPUTextureFormat | null)[] = colorFormats
        ? [...colorFormats]
        : Array.from({ length: colorTargetCount }, () => colorFormat);
```

Thread `colorFormats` through the call in `renderToGBuffer()`:
```typescript
this._gbufferBundle = this._buildRenderBundle(
    orderedObjects, cameraBindGroup, 'rgba16float', gbuffer.msaaSampleCount, 'depth32float', 4, mrtFormats
);
```

Also thread `colorFormats` through to each `renderable.material.getPipelineForConfig()` call inside `_buildRenderBundle`.

**Step 5: Verify**

Run: `tsc --noEmit`
Expected: No errors.

**Step 6: Commit**

```bash
git add src/renderers/Renderer.ts src/materials/Material.ts
git commit -m "feat(renderer): extend MRT to 4 color targets for pathtracer GBuffer"
```

---

### Task 4: Update example shaders to output normals and albedo

**Files:**
- Modify: `examples/index_postpro.html`

Shaders currently output to `@location(0)` (color) and `@location(1)` (emissive). Add `@location(2)` (world normal + roughness) and `@location(3)` (albedo + metallic).

**Step 1: Update FragmentOutput struct in both shaders**

```wgsl
struct FragmentOutput {
    @location(0) color    : vec4f,
    @location(1) emissive : vec4f,
    @location(2) normal   : vec4f,
    @location(3) albedo   : vec4f,
}
```

**Step 2: Update lambert fragment**

```wgsl
@fragment
fn fragment_main(frag : VertexOut) -> FragmentOutput {
    let shadow   = calcDirectionalShadow(frag.worldPos, frag.worldNormal);
    let L        = -normalize(lightDirection);
    let diffuse  = max(dot(frag.worldNormal, L), 0.0);
    let lighting = AMBIENT + (1.0 - AMBIENT) * diffuse * shadow;
    return FragmentOutput(
        vec4f(BASE_COLOR * lighting, 1.0),
        vec4f(0, 0, 0, 1),
        vec4f(frag.worldNormal * 0.5 + 0.5, 1.0),  // pack normal [0,1]
        vec4f(BASE_COLOR, 0.0)                        // albedo, metallic=0
    );
}
```

**Step 3: Update emissive fragment similarly**

Same pattern — add normal and albedo outputs.

**Step 4: Verify**

Open `examples/index_postpro.html` in browser. Scene should render identically — extra MRT targets are written but not yet consumed.

**Step 5: Commit**

```bash
git add examples/index_postpro.html
git commit -m "feat(example): output world normals and albedo to GBuffer MRT"
```

---

## Phase 2: GPU BVH Construction

### Task 5: BVH buffer manager (TypeScript scaffolding)

**Files:**
- Create: `src/pathtracer/BVHBuilder.ts`

This class manages GPU buffers for triangles, BVH nodes, instances, and materials. It orchestrates the compute passes for BLAS/TLAS construction.

**Step 1: Create the class skeleton**

```typescript
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { PathTracerMaterial } from "./PathTracerMaterial";

export interface BLASEntry {
    geometryId: string;           // unique geometry identifier
    triangleOffset: number;       // offset into global triangle buffer
    triangleCount: number;        // number of triangles
    nodeOffset: number;           // offset into BLAS node buffer
    nodeCount: number;            // number of BVH nodes
}

export class BVHBuilder {
    private _device: GPUDevice;

    // BLAS data
    private _triangleBuffer: GPUBuffer | null = null;
    private _blasNodeBuffer: GPUBuffer | null = null;
    private _blasEntries: Map<string, BLASEntry> = new Map();
    private _totalTriangles: number = 0;
    private _totalBLASNodes: number = 0;

    // TLAS data
    private _instanceBuffer: GPUBuffer | null = null;
    private _tlasNodeBuffer: GPUBuffer | null = null;
    private _totalInstances: number = 0;

    // Material data
    private _materialBuffer: GPUBuffer | null = null;
    private _materialCount: number = 0;

    // Sort scratch buffers
    private _mortonBuffer: GPUBuffer | null = null;
    private _mortonIndicesBuffer: GPUBuffer | null = null;
    private _sortScratchBuffer: GPUBuffer | null = null;

    // Compute pipelines
    private _mortonPipeline: GPUComputePipeline | null = null;
    private _sortPipelines: GPUComputePipeline[] = [];
    private _treeBuildPipeline: GPUComputePipeline | null = null;
    private _refitPipeline: GPUComputePipeline | null = null;
    private _instanceAABBPipeline: GPUComputePipeline | null = null;

    // Bind group layouts
    private _mortonBGL: GPUBindGroupLayout | null = null;
    private _sortBGL: GPUBindGroupLayout | null = null;
    private _treeBuildBGL: GPUBindGroupLayout | null = null;
    private _refitBGL: GPUBindGroupLayout | null = null;
    private _instanceAABBBGL: GPUBindGroupLayout | null = null;

    constructor(device: GPUDevice) {
        this._device = device;
    }

    // Public accessors for trace shader bind groups
    get triangleBuffer(): GPUBuffer | null { return this._triangleBuffer; }
    get blasNodeBuffer(): GPUBuffer | null { return this._blasNodeBuffer; }
    get instanceBuffer(): GPUBuffer | null { return this._instanceBuffer; }
    get tlasNodeBuffer(): GPUBuffer | null { return this._tlasNodeBuffer; }
    get materialBuffer(): GPUBuffer | null { return this._materialBuffer; }
    get totalInstances(): number { return this._totalInstances; }

    /**
     * Extract all unique geometries and build BLAS for each.
     * Called once on scene setup, or when geometry changes.
     */
    public buildBLAS(scene: Scene): void {
        // Implementation in Task 6
    }

    /**
     * Build TLAS from current instance transforms.
     * Called every frame.
     */
    public buildTLAS(commandEncoder: GPUCommandEncoder, scene: Scene): void {
        // Implementation in Task 11
    }

    /**
     * Pack all PathTracerMaterials into the GPU material buffer.
     */
    public updateMaterials(scene: Scene): void {
        // Implementation in Task 6
    }

    public destroy(): void {
        this._triangleBuffer?.destroy();
        this._blasNodeBuffer?.destroy();
        this._instanceBuffer?.destroy();
        this._tlasNodeBuffer?.destroy();
        this._materialBuffer?.destroy();
        this._mortonBuffer?.destroy();
        this._mortonIndicesBuffer?.destroy();
        this._sortScratchBuffer?.destroy();
    }
}
```

**Step 2: Export from main.ts**

```typescript
export { BVHBuilder } from "./pathtracer/BVHBuilder";
```

**Step 3: Verify**

Run: `tsc --noEmit`

**Step 4: Commit**

```bash
git add src/pathtracer/BVHBuilder.ts src/main.ts
git commit -m "feat(pathtracer): add BVHBuilder class skeleton"
```

---

### Task 6: BLAS triangle extraction and material packing

**Files:**
- Modify: `src/pathtracer/BVHBuilder.ts`

Extract triangles from each unique Geometry, pack into flat GPU storage buffers. Also pack PathTracerMaterials.

**Step 1: Implement `buildBLAS()`**

Key logic: iterate scene objects, deduplicate geometries by reference, extract vertex+index data, pack triangles.

**GPU Triangle struct (48 floats = 192 bytes per triangle):**
```
v0: vec3f, _pad0: f32,     // 16 bytes
v1: vec3f, _pad1: f32,     // 16 bytes
v2: vec3f, _pad2: f32,     // 16 bytes
n0: vec3f, _pad3: f32,     // 16 bytes
n1: vec3f, _pad4: f32,     // 16 bytes
n2: vec3f, materialIdx: f32 // 16 bytes
```
Total: 96 bytes = 24 floats per triangle.

Actually, let's use a more compact layout:
```
v0.xyz, v1.xyz, v2.xyz       (9 floats)
n0.xyz, n1.xyz, n2.xyz       (9 floats)
materialIndex, padding×3     (4 floats)
= 22 floats → round up to 24 for alignment = 96 bytes
```

```typescript
static readonly TRI_STRIDE_FLOATS = 24; // 96 bytes per triangle

public buildBLAS(scene: Scene): void {
    const objects = scene.getOrderedObjects();
    const geometryMap = new Map<object, { triangles: Float32Array; triCount: number }>();
    const materialMap = new Map<Renderable, number>();
    const materials: PathTracerMaterial[] = [];

    // Collect unique geometries and assign material indices
    for (const obj of objects) {
        const geom = obj.geometry.isInstancedGeometry
            ? (obj.geometry as InstancedGeometry).geometry ?? obj.geometry
            : obj.geometry;

        if (!geometryMap.has(geom)) {
            const tris = this._extractTriangles(geom, 0); // materialIdx filled later
            geometryMap.set(geom, tris);
        }

        if (!materialMap.has(obj)) {
            const ptMat = obj.pathTracerMaterial ?? new PathTracerMaterial();
            const idx = materials.length;
            materials.push(ptMat);
            materialMap.set(obj, idx);
        }
    }

    // Pack all triangles into one contiguous buffer
    let totalTris = 0;
    for (const [, data] of geometryMap) totalTris += data.triCount;

    const allTriangles = new Float32Array(totalTris * BVHBuilder.TRI_STRIDE_FLOATS);
    let offset = 0;
    this._blasEntries.clear();

    for (const [geom, data] of geometryMap) {
        const entry: BLASEntry = {
            geometryId: (geom as any).uuid ?? String(offset),
            triangleOffset: offset / BVHBuilder.TRI_STRIDE_FLOATS,
            triangleCount: data.triCount,
            nodeOffset: 0,  // filled after tree build
            nodeCount: 0,
        };
        allTriangles.set(data.triangles, offset);
        offset += data.triCount * BVHBuilder.TRI_STRIDE_FLOATS;
        this._blasEntries.set(entry.geometryId, entry);
    }

    this._totalTriangles = totalTris;

    // Upload triangle buffer
    this._triangleBuffer?.destroy();
    this._triangleBuffer = this._device.createBuffer({
        label: 'BVH/Triangles',
        size: allTriangles.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._device.queue.writeBuffer(this._triangleBuffer, 0, allTriangles);

    // Pack materials
    this._updateMaterialBuffer(materials);
}
```

**Step 2: Implement `_extractTriangles()`**

Reads `geometry.vertices` (Float32Array, stride 9 floats) and `geometry.indices` (Uint16Array):

```typescript
private _extractTriangles(
    geometry: Geometry,
    materialIndex: number
): { triangles: Float32Array; triCount: number } {
    const verts = geometry.vertices!;
    const indices = geometry.indices!;
    const triCount = Math.floor(indices.length / 3);
    const tris = new Float32Array(triCount * BVHBuilder.TRI_STRIDE_FLOATS);
    const VERT_STRIDE = 9; // 4 pos + 3 normal + 2 uv

    for (let t = 0; t < triCount; t++) {
        const i0 = indices[t * 3 + 0];
        const i1 = indices[t * 3 + 1];
        const i2 = indices[t * 3 + 2];
        const out = t * BVHBuilder.TRI_STRIDE_FLOATS;

        // Positions (skip w component at offset 3)
        tris[out + 0] = verts[i0 * VERT_STRIDE + 0];
        tris[out + 1] = verts[i0 * VERT_STRIDE + 1];
        tris[out + 2] = verts[i0 * VERT_STRIDE + 2];
        tris[out + 3] = verts[i1 * VERT_STRIDE + 0];
        tris[out + 4] = verts[i1 * VERT_STRIDE + 1];
        tris[out + 5] = verts[i1 * VERT_STRIDE + 2];
        tris[out + 6] = verts[i2 * VERT_STRIDE + 0];
        tris[out + 7] = verts[i2 * VERT_STRIDE + 1];
        tris[out + 8] = verts[i2 * VERT_STRIDE + 2];

        // Normals (offset 4 in vertex stride)
        tris[out + 9]  = verts[i0 * VERT_STRIDE + 4];
        tris[out + 10] = verts[i0 * VERT_STRIDE + 5];
        tris[out + 11] = verts[i0 * VERT_STRIDE + 6];
        tris[out + 12] = verts[i1 * VERT_STRIDE + 4];
        tris[out + 13] = verts[i1 * VERT_STRIDE + 5];
        tris[out + 14] = verts[i1 * VERT_STRIDE + 6];
        tris[out + 15] = verts[i2 * VERT_STRIDE + 4];
        tris[out + 16] = verts[i2 * VERT_STRIDE + 5];
        tris[out + 17] = verts[i2 * VERT_STRIDE + 6];

        // Material index + padding
        tris[out + 18] = materialIndex;
        // [19-23] = 0 (padding, already zero)
    }

    return { triangles: tris, triCount };
}
```

**Step 3: Implement `_updateMaterialBuffer()`**

```typescript
private _updateMaterialBuffer(materials: PathTracerMaterial[]): void {
    const floatsPerMat = PathTracerMaterial.GPU_STRIDE / 4; // 16
    const staging = new Float32Array(materials.length * floatsPerMat);
    for (let i = 0; i < materials.length; i++) {
        materials[i].packInto(staging, i * floatsPerMat);
    }
    this._materialBuffer?.destroy();
    this._materialBuffer = this._device.createBuffer({
        label: 'BVH/Materials',
        size: staging.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this._device.queue.writeBuffer(this._materialBuffer, 0, staging);
    this._materialCount = materials.length;
}
```

**Step 4: Verify**

Run: `tsc --noEmit`

**Step 5: Commit**

```bash
git add src/pathtracer/BVHBuilder.ts
git commit -m "feat(pathtracer): BLAS triangle extraction and material packing"
```

---

### Task 7: Morton code compute shader

**Files:**
- Create: `src/pathtracer/shaders/morton.wgsl.ts`
- Modify: `src/pathtracer/BVHBuilder.ts`

Compute Morton codes for triangle centroids (BLAS) or instance AABBs (TLAS). 30-bit Morton codes from 10-bit x/y/z.

**Step 1: Create the Morton code shader**

```typescript
// src/pathtracer/shaders/morton.wgsl.ts
export const mortonShader = /* wgsl */`
struct Params {
    count      : u32,
    sceneMinX  : f32,
    sceneMinY  : f32,
    sceneMinZ  : f32,
    sceneExtX  : f32,  // 1 / (max - min) for each axis
    sceneExtY  : f32,
    sceneExtZ  : f32,
    _pad       : u32,
}

@group(0) @binding(0) var<storage, read>       centroids  : array<vec4f>;
@group(0) @binding(1) var<storage, read_write> mortonKeys : array<u32>;
@group(0) @binding(2) var<storage, read_write> mortonVals : array<u32>;
@group(0) @binding(3) var<uniform>             params     : Params;

fn expandBits10(v: u32) -> u32 {
    var x = v & 0x3FFu;
    x = (x | (x << 16u)) & 0x030000FFu;
    x = (x | (x <<  8u)) & 0x0300F00Fu;
    x = (x | (x <<  4u)) & 0x030C30C3u;
    x = (x | (x <<  2u)) & 0x09249249u;
    return x;
}

fn morton3D(x: f32, y: f32, z: f32) -> u32 {
    let ix = u32(clamp(x * 1024.0, 0.0, 1023.0));
    let iy = u32(clamp(y * 1024.0, 0.0, 1023.0));
    let iz = u32(clamp(z * 1024.0, 0.0, 1023.0));
    return (expandBits10(ix) << 2u) | (expandBits10(iy) << 1u) | expandBits10(iz);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.count) { return; }

    let c = centroids[idx];
    let nx = (c.x - params.sceneMinX) * params.sceneExtX;
    let ny = (c.y - params.sceneMinY) * params.sceneExtY;
    let nz = (c.z - params.sceneMinZ) * params.sceneExtZ;

    mortonKeys[idx] = morton3D(nx, ny, nz);
    mortonVals[idx] = idx;
}
`;
```

**Step 2: Add pipeline creation to BVHBuilder**

Add method `_ensureMortonPipeline()` that creates the compute pipeline and BGL for the Morton code pass.

**Step 3: Add `_computeMortonCodes()` method**

Takes a centroids buffer and count, dispatches the Morton shader, returns morton keys + values buffers.

```typescript
private _computeMortonCodes(
    commandEncoder: GPUCommandEncoder,
    centroidsBuffer: GPUBuffer,
    count: number,
    sceneMin: [number, number, number],
    sceneMax: [number, number, number]
): void {
    this._ensureMortonPipeline();
    // Create/resize morton buffers if needed
    // Update params uniform (count, sceneMin, sceneExt)
    // Dispatch compute pass
    const pass = commandEncoder.beginComputePass({ label: 'BVH/Morton' });
    pass.setPipeline(this._mortonPipeline!);
    pass.setBindGroup(0, mortonBG);
    pass.dispatchWorkgroups(Math.ceil(count / 256));
    pass.end();
}
```

**Step 4: Verify**

Run: `tsc --noEmit`

**Step 5: Commit**

```bash
git add src/pathtracer/shaders/morton.wgsl.ts src/pathtracer/BVHBuilder.ts
git commit -m "feat(pathtracer): GPU Morton code compute shader"
```

---

### Task 8: GPU radix sort compute shader

**Files:**
- Create: `src/pathtracer/shaders/radix-sort.wgsl.ts`
- Modify: `src/pathtracer/BVHBuilder.ts`

4-bit radix sort (8 passes for 32-bit keys). Each pass: histogram → prefix sum → scatter.

**Step 1: Create radix sort shader**

This is the largest single shader. It sorts `mortonKeys` and `mortonVals` arrays in-place using a work-efficient parallel radix sort.

The shader has 3 entry points per pass:
- `histogram`: count occurrences of each 4-bit digit per workgroup
- `prefix_sum`: exclusive prefix sum over histograms
- `scatter`: scatter elements to sorted positions

Each 4-bit pass sorts by one nybble (0-15), requiring 8 passes total for 32-bit keys.

Key buffers:
- `mortonKeys` / `mortonVals` (input)
- `mortonKeysOut` / `mortonValsOut` (output, ping-pong)
- `histograms` (workgroup_count × 16 u32s)
- `prefixSums` (workgroup_count × 16 u32s)

**Step 2: Add radix sort orchestration to BVHBuilder**

```typescript
private _radixSort(
    commandEncoder: GPUCommandEncoder,
    count: number
): void {
    this._ensureSortPipelines();

    for (let bitOffset = 0; bitOffset < 32; bitOffset += 4) {
        // Update bit offset uniform
        // Pass 1: histogram
        // Pass 2: prefix sum
        // Pass 3: scatter
        // Swap input/output buffers
    }
}
```

**Step 3: Verify**

Run: `tsc --noEmit`

**Step 4: Commit**

```bash
git add src/pathtracer/shaders/radix-sort.wgsl.ts src/pathtracer/BVHBuilder.ts
git commit -m "feat(pathtracer): GPU parallel radix sort for BVH construction"
```

---

### Task 9: Tree build compute shader (Karras 2012)

**Files:**
- Create: `src/pathtracer/shaders/tree-build.wgsl.ts`
- Modify: `src/pathtracer/BVHBuilder.ts`

Build a binary radix tree from sorted Morton codes using the Karras 2012 algorithm. One thread per internal node.

**BVH Node layout (32 bytes = 8 floats):**
```wgsl
struct BVHNode {
    boundsMin  : vec3f,
    leftChild  : i32,   // >= 0: internal node index, < 0: -(leafOffset + 1)
    boundsMax  : vec3f,
    rightChild : i32,   // >= 0: internal node index, < 0: -(leafOffset + 1)
}
```

**Step 1: Create tree build shader**

```typescript
export const treeBuildShader = /* wgsl */`
// Karras 2012: "Maximizing Parallelism in the Construction of BVHs..."

struct Params {
    leafCount : u32,
    _pad      : vec3u,
}

@group(0) @binding(0) var<storage, read>       sortedMorton : array<u32>;
@group(0) @binding(1) var<storage, read_write> nodes        : array<BVHNode>;
@group(0) @binding(2) var<uniform>             params       : Params;

fn clz(v: u32) -> u32 {
    // Count leading zeros via binary search
    if (v == 0u) { return 32u; }
    var n = 0u;
    if ((v & 0xFFFF0000u) == 0u) { n += 16u; v <<= 16u; }
    if ((v & 0xFF000000u) == 0u) { n +=  8u; v <<=  8u; }
    if ((v & 0xF0000000u) == 0u) { n +=  4u; v <<=  4u; }
    if ((v & 0xC0000000u) == 0u) { n +=  2u; v <<=  2u; }
    if ((v & 0x80000000u) == 0u) { n +=  1u; }
    return n;
}

fn delta(i: i32, j: i32, n: i32) -> i32 {
    if (j < 0 || j >= n) { return -1; }
    let ki = sortedMorton[u32(i)];
    let kj = sortedMorton[u32(j)];
    if (ki == kj) {
        return i32(32u - clz(u32(i) ^ u32(j))) + 32;
    }
    return i32(32u - clz(ki ^ kj));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = i32(gid.x);
    let n = i32(params.leafCount);
    if (idx >= n - 1) { return; }

    // Determine direction of the range
    let dPlus  = delta(idx, idx + 1, n);
    let dMinus = delta(idx, idx - 1, n);
    let d = select(-1, 1, dPlus > dMinus);
    let dMin = select(dPlus, dMinus, dPlus > dMinus);

    // Compute upper bound for the length of the range
    var lMax = 2;
    while (delta(idx, idx + lMax * d, n) > dMin) {
        lMax *= 2;
    }

    // Find the other end using binary search
    var l = 0;
    var t = lMax / 2;
    while (t >= 1) {
        if (delta(idx, idx + (l + t) * d, n) > dMin) {
            l += t;
        }
        t /= 2;
    }
    let j = idx + l * d;

    // Find the split position
    let dNode = delta(idx, j, n);
    var s = 0;
    var divider = 2;
    t = (l + divider - 1) / divider;
    while (t >= 1) {
        if (delta(idx, idx + (s + t) * d, n) > dNode) {
            s += t;
        }
        divider *= 2;
        t = (l + divider - 1) / divider;
    }
    let split = idx + s * d + min(d, 0);

    // Output children
    let leftIdx  = select(split, split + n - 1, split == min(idx, j));
    let rightIdx = select(split + 1, split + 1 + n - 1, split + 1 == max(idx, j));

    nodes[u32(idx)].leftChild  = i32(leftIdx);
    nodes[u32(idx)].rightChild = i32(rightIdx);
}
`;
```

**Step 2: Add tree build dispatch to BVHBuilder**

**Step 3: Verify & Commit**

---

### Task 10: AABB refit compute shader

**Files:**
- Create: `src/pathtracer/shaders/refit.wgsl.ts`
- Modify: `src/pathtracer/BVHBuilder.ts`

Bottom-up pass to compute AABBs for each BVH node. Uses atomic counters — each leaf marks itself done, each internal node waits until both children are done.

**Step 1: Create refit shader**

```typescript
export const refitShader = /* wgsl */`
struct Params {
    leafCount : u32,
    triOffset : u32,  // offset into global triangle buffer for this BLAS
    _pad      : vec2u,
}

@group(0) @binding(0) var<storage, read_write> nodes      : array<BVHNode>;
@group(0) @binding(1) var<storage, read>       triangles  : array<f32>;
@group(0) @binding(2) var<storage, read_write> atomicFlags : array<atomic<u32>>;
@group(0) @binding(3) var<uniform>             params     : Params;
@group(0) @binding(4) var<storage, read>       parents    : array<i32>;

fn triAABB(triIdx: u32) -> vec2<vec3f> {
    let base = (params.triOffset + triIdx) * 24u;
    let v0 = vec3f(triangles[base+0u], triangles[base+1u], triangles[base+2u]);
    let v1 = vec3f(triangles[base+3u], triangles[base+4u], triangles[base+5u]);
    let v2 = vec3f(triangles[base+6u], triangles[base+7u], triangles[base+8u]);
    return vec2<vec3f>(min(min(v0, v1), v2), max(max(v0, v1), v2));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    let n = params.leafCount;
    if (idx >= n) { return; }

    // Start from leaf nodes (indices n-1 to 2n-2)
    let leafNodeIdx = idx + n - 1u;
    let aabb = triAABB(idx);
    nodes[leafNodeIdx].boundsMin = aabb[0];
    nodes[leafNodeIdx].boundsMax = aabb[1];

    // Walk up the tree
    var current = parents[leafNodeIdx];
    while (current >= 0) {
        // Atomic increment — first child to arrive exits, second does the work
        let old = atomicAdd(&atomicFlags[u32(current)], 1u);
        if (old == 0u) { return; } // first child — bail

        let left  = u32(nodes[u32(current)].leftChild);
        let right = u32(nodes[u32(current)].rightChild);
        nodes[u32(current)].boundsMin = min(nodes[left].boundsMin, nodes[right].boundsMin);
        nodes[u32(current)].boundsMax = max(nodes[left].boundsMax, nodes[right].boundsMax);

        current = parents[u32(current)];
    }
}
`;
```

**Step 2: Add parent array computation**

The tree-build pass (Task 9) produces child pointers. We need a `parents` array — add a second pass after tree-build that computes `parents[child] = parentIdx` for each internal node.

**Step 3: Add refit dispatch to BVHBuilder**

**Step 4: Verify & Commit**

---

### Task 11: TLAS builder — instance AABB + full rebuild pipeline

**Files:**
- Create: `src/pathtracer/shaders/instance-aabb.wgsl.ts`
- Modify: `src/pathtracer/BVHBuilder.ts`

**Step 1: Create instance AABB shader**

Reads instance transforms + BLAS root AABBs, computes world-space AABB per instance:

```wgsl
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let idx = gid.x;
    if (idx >= params.instanceCount) { return; }

    let inst = instances[idx];
    let blasRoot = blasNodes[inst.blasNodeOffset];
    let localMin = blasRoot.boundsMin;
    let localMax = blasRoot.boundsMax;

    // Transform 8 corners of local AABB to world space, take min/max
    var worldMin = vec3f(1e30);
    var worldMax = vec3f(-1e30);
    for (var i = 0u; i < 8u; i++) {
        let corner = vec3f(
            select(localMin.x, localMax.x, (i & 1u) != 0u),
            select(localMin.y, localMax.y, (i & 2u) != 0u),
            select(localMin.z, localMax.z, (i & 4u) != 0u),
        );
        let world = transformPoint(inst, corner);
        worldMin = min(worldMin, world);
        worldMax = max(worldMax, world);
    }

    centroids[idx] = vec4f((worldMin + worldMax) * 0.5, 0.0);
    instanceAABBs[idx] = vec2<vec4f>(vec4f(worldMin, 0), vec4f(worldMax, 0));
}
```

**Step 2: Implement `buildTLAS()`**

Orchestrates the full per-frame TLAS rebuild:

```typescript
public buildTLAS(commandEncoder: GPUCommandEncoder, scene: Scene): void {
    const objects = scene.getOrderedObjects();
    this._packInstances(objects); // upload instance data

    // 1. Compute instance AABBs + centroids
    // 2. Compute Morton codes from centroids
    // 3. Radix sort
    // 4. Build tree (Karras)
    // 5. Refit AABBs

    // All dispatched into the same commandEncoder
}
```

**Step 3: Implement `_packInstances()`**

For standard Renderables: use worldMatrix.
For InstancedGeometry: read from extraBuffers (the compute simulation position buffer). Each instance gets an entry with its blasIndex, materialIndex, and transform.

```typescript
private _packInstances(objects: Renderable[]): void {
    // Count total instances (regular objects = 1 each, instanced = instanceCount)
    let totalInstances = 0;
    for (const obj of objects) {
        if (obj.geometry.isInstancedGeometry) {
            totalInstances += (obj.geometry as InstancedGeometry).instanceCount;
        } else {
            totalInstances += 1;
        }
    }
    this._totalInstances = totalInstances;

    // For instanced geometry, we can't read positions from CPU —
    // they live in GPU storage buffers. Instead, we pass the GPU buffer
    // references directly to the instance AABB shader.
    // For non-instanced: pack world matrix into instance buffer from CPU.
}
```

**Step 4: Verify & Commit**

---

### Task 12: Integration with InstancedGeometry compute buffers

**Files:**
- Modify: `src/pathtracer/BVHBuilder.ts`
- Modify: `src/pathtracer/shaders/instance-aabb.wgsl.ts`

The instance AABB shader needs to handle two cases:
1. **Non-instanced**: Read world matrix from a packed instance buffer (CPU-uploaded)
2. **Instanced**: Read per-instance positions from the compute simulation's storage buffer (GPU-only)

**Step 1: Extend instance buffer layout**

```
Per instance (96 bytes = 24 floats):
  transform   : mat3x4f    (48 bytes, rows 0-2 of world matrix)
  invTransform: mat3x4f    (36 bytes, inverse for ray transform)
  blasIndex   : u32
  materialIndex: u32
  pad         : vec2u
```

For instanced geometry, the "transform" is just a translation from the position buffer. The instance AABB shader reads the position buffer directly and constructs the affine transform in-shader.

**Step 2: Add bind group entry for instance position buffer**

The TLAS builder needs to bind the InstancedGeometry's position ComputeBuffer. Add a reference to the buffer in the instance data:

```typescript
// When packing instances for instanced geometry:
// Store a reference to the GPU buffer so the AABB shader can read it
private _instancePositionBuffers: Map<number, GPUBuffer> = new Map();
```

**Step 3: Verify & Commit**

---

## Phase 3: Path Tracing Shaders

### Task 13: Ray-BVH intersection utilities (WGSL)

**Files:**
- Create: `src/pathtracer/shaders/intersection.wgsl.ts`

Core ray intersection functions used by the trace shader.

**Step 1: Create intersection utilities**

```typescript
export const intersectionShader = /* wgsl */`
struct Ray {
    origin : vec3f,
    dir    : vec3f,
}

struct HitInfo {
    t          : f32,
    u          : f32,
    v          : f32,
    triIndex   : u32,
    instanceId : u32,
    matIndex   : u32,
    worldPos   : vec3f,
    worldNorm  : vec3f,
    hit        : bool,
}

fn rayAABB(ray: Ray, bmin: vec3f, bmax: vec3f, tMax: f32) -> f32 {
    let invDir = 1.0 / ray.dir;
    let t1 = (bmin - ray.origin) * invDir;
    let t2 = (bmax - ray.origin) * invDir;
    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));
    if (tmax < 0.0 || tmin > tmax || tmin > tMax) { return -1.0; }
    return tmin;
}

fn rayTriangle(ray: Ray, v0: vec3f, v1: vec3f, v2: vec3f) -> vec3f {
    // Möller–Trumbore. Returns (t, u, v) or t < 0 on miss.
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    let h = cross(ray.dir, e2);
    let a = dot(e1, h);
    if (abs(a) < 1e-8) { return vec3f(-1.0, 0.0, 0.0); }
    let f = 1.0 / a;
    let s = ray.origin - v0;
    let u = f * dot(s, h);
    if (u < 0.0 || u > 1.0) { return vec3f(-1.0, 0.0, 0.0); }
    let q = cross(s, e1);
    let v = f * dot(ray.dir, q);
    if (v < 0.0 || u + v > 1.0) { return vec3f(-1.0, 0.0, 0.0); }
    let t = f * dot(e2, q);
    if (t < 1e-5) { return vec3f(-1.0, 0.0, 0.0); }
    return vec3f(t, u, v);
}
`;
```

**Step 2: Verify & Commit**

---

### Task 14: Two-level BVH traversal (WGSL)

**Files:**
- Create: `src/pathtracer/shaders/traversal.wgsl.ts`

Stack-based TLAS/BLAS traversal.

**Step 1: Create traversal code**

```wgsl
const STACK_SIZE = 32u;

fn traceBVH(ray: Ray) -> HitInfo {
    var hit: HitInfo;
    hit.t = 1e30;
    hit.hit = false;

    // TLAS traversal
    var tlasStack: array<u32, STACK_SIZE>;
    var tlasPtr = 0u;
    tlasStack[0] = 0u; // root
    tlasPtr = 1u;

    while (tlasPtr > 0u) {
        tlasPtr -= 1u;
        let nodeIdx = tlasStack[tlasPtr];
        let node = tlasNodes[nodeIdx];

        let tHit = rayAABB(ray, node.boundsMin, node.boundsMax, hit.t);
        if (tHit < 0.0) { continue; }

        if (node.leftChild < 0) {
            // Leaf — contains instance
            let instIdx = u32(-(node.leftChild + 1));
            let inst = instances[instIdx];

            // Transform ray to instance local space
            let localRay = transformRay(ray, inst);

            // BLAS traversal
            let blasHit = traverseBLAS(localRay, inst.blasNodeOffset, hit.t);
            if (blasHit.hit && blasHit.t < hit.t) {
                hit = blasHit;
                hit.instanceId = instIdx;
                hit.matIndex = inst.materialIndex;
                // Transform hit to world space
                hit.worldPos = transformPoint(inst, blasHit.worldPos);
                hit.worldNorm = transformNormal(inst, blasHit.worldNorm);
            }
        } else {
            // Internal — push children (nearer child last for early exit)
            tlasStack[tlasPtr] = u32(node.leftChild);
            tlasPtr += 1u;
            tlasStack[tlasPtr] = u32(node.rightChild);
            tlasPtr += 1u;
        }
    }
    return hit;
}

fn traverseBLAS(ray: Ray, nodeOffset: u32, tMax: f32) -> HitInfo {
    // Similar stack-based traversal but over triangle leaves
    // ...
}
```

**Step 2: Verify & Commit**

---

### Task 15: GI trace compute shader

**Files:**
- Create: `src/pathtracer/shaders/trace.wgsl.ts`

Main trace shader. Reads GBuffer, shoots indirect rays, handles refraction.

**Step 1: Create trace shader**

```wgsl
@group(0) @binding(0)  var          depthTex    : texture_depth_2d;
@group(0) @binding(1)  var          normalTex   : texture_2d<f32>;
@group(0) @binding(2)  var          albedoTex   : texture_2d<f32>;
@group(0) @binding(3)  var          giOutput    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4)  var<uniform> traceParams : TraceParams;
@group(0) @binding(5)  var<storage, read> triangles   : array<f32>;
@group(0) @binding(6)  var<storage, read> blasNodes   : array<BVHNode>;
@group(0) @binding(7)  var<storage, read> tlasNodes   : array<BVHNode>;
@group(0) @binding(8)  var<storage, read> instances   : array<Instance>;
@group(0) @binding(9)  var<storage, read> materials   : array<MaterialData>;

struct TraceParams {
    invViewProj : mat4x4f,
    frameIndex  : u32,
    width       : u32,
    height      : u32,
    _pad        : u32,
}

// PCG hash for random numbers
fn pcgHash(seed: u32) -> u32 { ... }
fn randomFloat(state: ptr<function, u32>) -> f32 { ... }

// Cosine-weighted hemisphere sampling
fn cosineSampleHemisphere(n: vec3f, r1: f32, r2: f32) -> vec3f { ... }

// GGX importance sampling for specular
fn sampleGGX(n: vec3f, roughness: f32, r1: f32, r2: f32) -> vec3f { ... }

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    if (gid.x >= traceParams.width || gid.y >= traceParams.height) { return; }

    let coord = vec2u(gid.x, gid.y);
    let depth = textureLoad(depthTex, coord, 0);
    if (depth >= 1.0) {
        textureStore(giOutput, coord, vec4f(0.0));
        return;
    }

    // Reconstruct world position from depth
    let uv = (vec2f(coord) + 0.5) / vec2f(f32(traceParams.width), f32(traceParams.height));
    let ndc = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
    let wp = traceParams.invViewProj * ndc;
    let worldPos = wp.xyz / wp.w;

    let normalData = textureLoad(normalTex, coord, 0);
    let worldNormal = normalize(normalData.xyz * 2.0 - 1.0);
    let roughness = normalData.w;

    let albedoData = textureLoad(albedoTex, coord, 0);
    let albedo = albedoData.rgb;
    let metallic = albedoData.a;

    // RNG seed from pixel + frame
    var rng = pcgHash(gid.x + gid.y * traceParams.width + traceParams.frameIndex * 1000003u);

    // Indirect diffuse ray
    let r1 = randomFloat(&rng);
    let r2 = randomFloat(&rng);
    let sampleDir = cosineSampleHemisphere(worldNormal, r1, r2);

    var ray: Ray;
    ray.origin = worldPos + worldNormal * 0.001;
    ray.dir = sampleDir;

    let hit = traceBVH(ray);
    var indirect = vec3f(0.0);

    if (hit.hit) {
        let mat = materials[hit.matIndex];
        // Direct lighting at hit point (shadow ray to sun)
        let hitLighting = evaluateDirectLight(hit, mat);
        indirect = hitLighting * mat.albedo.rgb;

        // Handle refraction
        if (mat.flags & 1u) != 0u {
            indirect = traceRefraction(ray, hit, mat, &rng);
        }
    }

    textureStore(giOutput, coord, vec4f(indirect, 1.0));
}
```

**Step 2: Add refraction loop**

```wgsl
fn traceRefraction(
    inRay: Ray,
    firstHit: HitInfo,
    mat: MaterialData,
    rng: ptr<function, u32>
) -> vec3f {
    var throughput = vec3f(1.0);
    var ray = inRay;
    var currentHit = firstHit;
    let maxBounces = u32(mat.maxBounces);

    for (var bounce = 0u; bounce < maxBounces; bounce++) {
        let n = currentHit.worldNorm;
        let entering = dot(ray.dir, n) < 0.0;
        let faceNorm = select(-n, n, entering);
        let eta = select(mat.ior, 1.0 / mat.ior, entering);

        let refracted = refract(ray.dir, faceNorm, eta);
        if (length(refracted) < 0.001) {
            // Total internal reflection
            ray.origin = currentHit.worldPos + faceNorm * 0.001;
            ray.dir = reflect(ray.dir, faceNorm);
        } else {
            ray.origin = currentHit.worldPos - faceNorm * 0.001;
            ray.dir = refracted;
        }

        let nextHit = traceBVH(ray);
        if (!nextHit.hit) { break; }

        // Beer's law absorption
        let dist = nextHit.t;
        throughput *= exp(-mat.absorptionColor * mat.absorptionDensity * dist);

        let nextMat = materials[nextHit.matIndex];
        if ((nextMat.flags & 1u) == 0u) {
            // Hit opaque surface — shade and return
            return throughput * evaluateDirectLight(nextHit, nextMat) * nextMat.albedo;
        }
        currentHit = nextHit;
    }
    return throughput * vec3f(0.0); // exhausted bounces
}
```

**Step 3: Verify & Commit**

---

## Phase 4: Denoiser

### Task 16: Temporal denoiser compute shader

**Files:**
- Create: `src/pathtracer/shaders/denoise-temporal.wgsl.ts`

Reproject previous frame GI, exponential blend with current frame, reject on disocclusion.

**Step 1: Create temporal shader**

```wgsl
@group(0) @binding(0) var currentGI   : texture_2d<f32>;
@group(0) @binding(1) var historyGI   : texture_2d<f32>;
@group(0) @binding(2) var outputGI    : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var depthTex    : texture_depth_2d;
@group(0) @binding(4) var normalTex   : texture_2d<f32>;
@group(0) @binding(5) var historySamp  : sampler;
@group(0) @binding(6) var<uniform> params : TemporalParams;

struct TemporalParams {
    currentInvViewProj : mat4x4f,
    prevViewProj       : mat4x4f,
    blendFactor        : f32,  // 0.1 = 90% history reuse
    width              : f32,
    height             : f32,
    frameIndex         : u32,
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    // Reconstruct world pos from current depth
    // Reproject to previous frame UV via prevViewProj
    // Sample history at reprojected UV
    // Reject if depth/normal discontinuity
    // Blend: output = mix(history, current, blendFactor)
}
```

**Step 2: Verify & Commit**

---

### Task 17: Spatial denoiser (a-trous wavelet filter)

**Files:**
- Create: `src/pathtracer/shaders/denoise-spatial.wgsl.ts`

Edge-aware bilateral filter with progressively wider kernels.

**Step 1: Create spatial shader**

```wgsl
// 5x5 a-trous kernel, parameterized by step size
@group(0) @binding(0) var inputGI    : texture_2d<f32>;
@group(0) @binding(1) var outputGI   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var depthTex   : texture_depth_2d;
@group(0) @binding(3) var normalTex  : texture_2d<f32>;
@group(0) @binding(4) var<uniform> params : SpatialParams;

struct SpatialParams {
    stepSize   : u32,   // 1, 2, 4, 8 for each iteration
    sigmaDepth : f32,
    sigmaNormal: f32,
    sigmaLum   : f32,
    width      : u32,
    height     : u32,
    _pad       : vec2u,
}

const KERNEL_OFFSETS = array<vec2i, 25>( /* 5x5 */ );
const KERNEL_WEIGHTS = array<f32, 25>( /* Gaussian */ );

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    // For each 5x5 kernel tap at stepSize distance:
    //   weight *= exp(-depthDiff / sigmaDepth)
    //   weight *= pow(max(dot(n1, n2), 0), sigmaNormal)
    //   weight *= exp(-lumDiff / sigmaLum)
    //   accumulate weighted sample
    // Normalize and store
}
```

Run 3-4 iterations with stepSize = 1, 2, 4, 8.

**Step 2: Verify & Commit**

---

## Phase 5: Integration

### Task 18: PathTracerEffect class

**Files:**
- Create: `src/pathtracer/PathTracerEffect.ts`

PostProcessingEffect subclass that orchestrates: BVH build → trace → temporal denoise → spatial denoise → composite.

**Step 1: Create the effect class**

```typescript
import { PostProcessingEffect } from "../postprocessing/PostProcessingEffect";
import { GBuffer } from "../postprocessing/GBuffer";
import { Camera } from "../cameras/Camera";
import { Scene } from "../objects/Scene";
import { BVHBuilder } from "./BVHBuilder";

export interface PathTracerOptions {
    /** Max indirect bounces. Default 2. */
    maxBounces?: number;
    /** Spatial denoise iterations. Default 3. */
    denoiseIterations?: number;
    /** Temporal blend factor (lower = more history). Default 0.1. */
    temporalBlend?: number;
}

export class PathTracerEffect extends PostProcessingEffect {
    private _bvhBuilder: BVHBuilder | null = null;
    private _device: GPUDevice | null = null;
    private _scene: Scene | null = null;
    private _bvhBuilt: boolean = false;

    // Trace pass
    private _tracePipeline: GPUComputePipeline | null = null;
    private _traceBGL: GPUBindGroupLayout | null = null;
    private _traceBG: GPUBindGroup | null = null;
    private _traceParamsBuffer: GPUBuffer | null = null;
    private _giTexture: GPUTexture | null = null;

    // Temporal denoise
    private _temporalPipeline: GPUComputePipeline | null = null;
    private _temporalBG: GPUBindGroup | null = null;
    private _temporalParamsBuffer: GPUBuffer | null = null;
    private _historyTextures: [GPUTexture | null, GPUTexture | null] = [null, null];
    private _historyIdx: number = 0;
    private _frameIndex: number = 0;

    // Spatial denoise
    private _spatialPipeline: GPUComputePipeline | null = null;
    private _spatialBG: GPUBindGroup | null = null;
    private _spatialParamsBuffer: GPUBuffer | null = null;
    private _spatialScratchTexture: GPUTexture | null = null;

    // Composite
    private _compositePipeline: GPUComputePipeline | null = null;
    private _compositeBG: GPUBindGroup | null = null;

    // Options
    public maxBounces: number;
    public denoiseIterations: number;
    public temporalBlend: number;

    constructor(options: PathTracerOptions = {}) {
        super();
        this.maxBounces = options.maxBounces ?? 2;
        this.denoiseIterations = options.denoiseIterations ?? 3;
        this.temporalBlend = options.temporalBlend ?? 0.1;
    }

    /**
     * Must be called before the first render to set the scene reference
     * and trigger initial BVH build.
     */
    public setScene(scene: Scene): void {
        this._scene = scene;
        this._bvhBuilt = false;
    }

    initialize(device: GPUDevice, gbuffer: GBuffer, camera: Camera): void {
        this._device = device;
        this._bvhBuilder = new BVHBuilder(device);
        // Create all pipelines, buffers, textures
        // ...
        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
        emissive?: GPUTexture
    ): void {
        if (!this._scene) return;

        // Build BLAS on first frame or scene change
        if (!this._bvhBuilt) {
            this._bvhBuilder!.buildBLAS(this._scene);
            this._bvhBuilt = true;
        }

        // Rebuild TLAS every frame
        this._bvhBuilder!.buildTLAS(commandEncoder, this._scene);

        // Update trace params (invViewProj, frame index, etc.)
        this._updateTraceParams(camera, width, height);
        this._rebuildBindGroupsIfNeeded(input, depth, output);

        // Pass 1: GI Trace
        const wg = (n: number) => Math.ceil(n / 8);
        const tracePass = commandEncoder.beginComputePass({ label: 'PathTracer/Trace' });
        tracePass.setPipeline(this._tracePipeline!);
        tracePass.setBindGroup(0, this._traceBG!);
        tracePass.dispatchWorkgroups(wg(width), wg(height));
        tracePass.end();

        // Pass 2: Temporal denoise
        const temporalPass = commandEncoder.beginComputePass({ label: 'PathTracer/Temporal' });
        temporalPass.setPipeline(this._temporalPipeline!);
        temporalPass.setBindGroup(0, this._temporalBG!);
        temporalPass.dispatchWorkgroups(wg(width), wg(height));
        temporalPass.end();
        this._historyIdx = 1 - this._historyIdx; // swap ping-pong

        // Pass 3: Spatial denoise (N iterations)
        for (let i = 0; i < this.denoiseIterations; i++) {
            this._updateSpatialParams(1 << i, width, height);
            const spatialPass = commandEncoder.beginComputePass({ label: `PathTracer/Spatial/${i}` });
            spatialPass.setPipeline(this._spatialPipeline!);
            spatialPass.setBindGroup(0, this._spatialBG!);
            spatialPass.dispatchWorkgroups(wg(width), wg(height));
            spatialPass.end();
            // Swap spatial input/output for next iteration
        }

        // Pass 4: Composite (direct + indirect)
        const compositePass = commandEncoder.beginComputePass({ label: 'PathTracer/Composite' });
        compositePass.setPipeline(this._compositePipeline!);
        compositePass.setBindGroup(0, this._compositeBG!);
        compositePass.dispatchWorkgroups(wg(width), wg(height));
        compositePass.end();

        this._frameIndex++;
    }

    resize(width: number, height: number, gbuffer: GBuffer): void {
        // Recreate size-dependent textures (giTexture, history, scratch)
        // Rebuild bind groups
    }

    destroy(): void {
        this._bvhBuilder?.destroy();
        this._giTexture?.destroy();
        this._historyTextures[0]?.destroy();
        this._historyTextures[1]?.destroy();
        this._spatialScratchTexture?.destroy();
        // ... destroy all buffers and pipelines
    }
}
```

**Step 2: Export from main.ts**

```typescript
export { PathTracerEffect } from "./pathtracer/PathTracerEffect";
export { BVHBuilder } from "./pathtracer/BVHBuilder";
```

**Step 3: Verify & Commit**

---

### Task 19: Composite compute shader

**Files:**
- Create: `src/pathtracer/shaders/composite.wgsl.ts`

Merges rasterized direct lighting with denoised indirect GI.

```wgsl
@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    let coord = vec2u(gid.x, gid.y);
    let direct   = textureLoad(inputTex, coord, 0).rgb;
    let indirect = textureLoad(denoisedGI, coord, 0).rgb;
    let albedo   = textureLoad(albedoTex, coord, 0).rgb;

    // Direct already has albedo baked in from rasterizer.
    // Indirect is raw incoming radiance — multiply by albedo.
    let final = direct + albedo * indirect;

    textureStore(outputTex, coord, vec4f(final, 1.0));
}
```

**Verify & Commit**

---

### Task 20: Example integration

**Files:**
- Modify: `examples/index_postpro.html`

**Step 1: Import PathTracerEffect**

```javascript
import { PathTracerEffect } from '../src/pathtracer/PathTracerEffect'
```

**Step 2: Create and add the effect**

```javascript
const pathTracerEffect = new PathTracerEffect({
    maxBounces: 2,
    denoiseIterations: 3,
    temporalBlend: 0.1,
});
pathTracerEffect.setScene(scene);

// Add to effect chain (before fog, bloom, etc. or as replacement)
```

**Step 3: Add Tweakpane controls**

```javascript
const ptFolder = pane.addFolder({ title: 'Path Tracer' });
ptFolder.addBinding(effectToggles, 'pathTracer', { label: 'enabled' }).on('change', rebuildEffectChain);
ptFolder.addBinding(pathTracerEffect, 'maxBounces', { min: 1, max: 8, step: 1 });
ptFolder.addBinding(pathTracerEffect, 'denoiseIterations', { min: 0, max: 5, step: 1 });
ptFolder.addBinding(pathTracerEffect, 'temporalBlend', { min: 0.01, max: 1.0, step: 0.01 });
```

**Step 4: Add a refractive sphere to test refraction**

```javascript
const glassSphere = new Renderable(sphereGeometry, material);
glassSphere.position.set(0, 30, 0);
glassSphere.scale.set(5, 5, 5);
glassSphere.pathTracerMaterial = new PathTracerMaterial();
glassSphere.pathTracerMaterial.refractive = true;
glassSphere.pathTracerMaterial.ior = 1.5;
glassSphere.pathTracerMaterial.maxBounces = 4;
glassSphere.pathTracerMaterial.absorptionColor = [0.8, 0.95, 1.0];
glassSphere.pathTracerMaterial.absorptionDensity = 0.5;
scene.add(glassSphere);
```

**Step 5: Verify visually**

Open in browser. Expected:
- Scene renders with rasterized direct light (as before)
- Indirect lighting appears (subtle color bleeding, soft shadows from GI)
- Temporal accumulation converges over several frames when camera is still
- Refractive sphere shows distorted background through it
- 30+ fps target

**Step 6: Commit**

```bash
git add examples/index_postpro.html
git commit -m "feat(example): integrate PathTracerEffect with refractive glass sphere"
```

---

### Task 21: Final exports and cleanup

**Files:**
- Modify: `src/main.ts`

Ensure all new public classes are exported:
```typescript
export { PathTracerEffect } from "./pathtracer/PathTracerEffect";
export { PathTracerMaterial } from "./pathtracer/PathTracerMaterial";
export { BVHBuilder } from "./pathtracer/BVHBuilder";
```

**Verify:** `tsc --noEmit` — no errors.

**Commit:**
```bash
git add -A
git commit -m "feat(pathtracer): final exports and cleanup"
```

---

## Implementation Notes

**Key patterns from existing codebase to follow:**
- `PostProcessingEffect` lifecycle: `initialize()` → `render()` per frame → `resize()` on viewport change → `destroy()`
- All compute passes share a single `commandEncoder` within `render()` (see VolumetricFogEffect)
- Bind groups rebuilt only when textures change (ping-pong tracking pattern)
- Uniform buffers updated via `device.queue.writeBuffer()` every frame
- Storage buffers for large data (triangles, nodes, instances)
- Workgroup sizes: 8×8 for screen-space, 256×1 for 1D operations

**Vertex data access:** Geometry stores interleaved vertices as Float32Array with stride 9 floats (pos4 + normal3 + uv2). Position offset = 0, normal offset = 4 (in floats). Index format = uint16.

**InstancedGeometry:** `extraBuffers` are ComputeBuffers with `STORAGE | VERTEX` usage. The TLAS builder reads instance positions directly from these GPU buffers — no CPU roundtrip.

**Testing approach:** No unit tests for GPU compute shaders. Verification is visual + `tsc --noEmit` for type checking. Each phase builds on the previous — verify the pipeline compiles and runs before moving to the next phase.

**Performance budget at 1080p:**
- TLAS rebuild (300K instances): ~1-2ms
- GI trace (1 SPP): ~4-8ms
- Temporal denoise: ~0.5ms
- Spatial denoise (3 iterations): ~1.5ms
- Composite: ~0.2ms
- Total: ~8-12ms → 80-120 fps headroom for 30fps target
