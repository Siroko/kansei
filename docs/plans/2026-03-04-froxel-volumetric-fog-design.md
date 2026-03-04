# Froxel Volumetric Fog — Design

## Goal

Add physically-based volumetric fog with directional light scattering, shadow map occlusion,
and temporal reprojection to the Kansei WebGPU postprocessing pipeline. The froxel grid and
shadow map are modular components reusable by future effects (clustered lighting, volumetric GI).

## Architecture

Three independent modules:

```
ShadowMap (raster)          FroxelGrid (compute, reusable)
     │                           │
     └──────┬────────────────────┘
            │
    VolumetricFogEffect (compute, PostProcessingEffect)
```

### ShadowMap — `src/shadows/ShadowMap.ts`

Shared resource. Renders the scene from the sun's POV into a depth-only texture.

- **Resolution:** 2048×2048 `depth32float`
- **Projection:** Orthographic, fit to camera frustum (single cascade)
- **Raster pass:** Depth-only (no color attachment). Shadow vertex shader transforms
  positions by `lightViewProj`. Supports instanced draws — reuses the same draw calls
  and render bundles as the main scene with a shadow-specific pipeline/shader.
- **Outputs:**
  - `shadowDepthTexture` (GPUTexture, depth32float, TEXTURE_BINDING)
  - `lightViewProjMatrix` (Float32Array, 16 floats, updated per-frame)
- **Per-frame update:** Recompute orthographic bounds from camera frustum + light direction,
  re-render shadow pass.

### FroxelGrid — `src/froxels/FroxelGrid.ts`

Reusable 3D spatial grid aligned to the camera frustum. Owns textures, depth slicing,
temporal reprojection, and front-to-back accumulation. Effect-specific injection is external.

**Grid dimensions:** 160 × 90 × 64 (configurable)

**Exponential depth distribution:**
```
sliceDepth(i) = near * (far / near) ^ (i / numSlices)
```
Near slices are dense (fog detail visible up close), far slices are coarse.

**3D textures (rgba16float):**

| Texture              | Purpose                                           |
|----------------------|---------------------------------------------------|
| `scatterExtinctionTex` | rgb = in-scattered light, a = extinction coefficient |
| `accumTex`           | rgb = accumulated light, a = transmittance         |
| `historyTex`         | Previous frame's scatter/extinction for temporal   |

**Shared WGSL helpers** (injected into consumer shaders via string concatenation):
- `froxelToWorld(coord, invViewProj, params)` — froxel (x,y,z) → world position
- `worldToFroxel(worldPos, viewProj, params)` — world position → froxel (x,y,z)
- `sliceDepth(i, params)` — slice index → linear depth
- `depthToSlice(linearDepth, params)` — linear depth → fractional slice index

**Temporal reprojection pass** (compute, 4×4×4 workgroups over full grid):
1. Reconstruct world position from current froxel
2. Reproject via `prevViewProjMatrix` → previous frame's froxel UV
3. Sample `historyTex` (trilinear)
4. Blend: `result = mix(current, history, 0.9)`
5. Reject if reprojected UV out of bounds
6. Write blended result to `scatterExtinctionTex`, copy to `historyTex`

**Front-to-back accumulation pass** (compute, 8×8 workgroups over x×y):
Each thread marches 64 slices serially (front-to-back dependency):
```
transmittance = 1.0
accumulatedLight = vec3(0)

for slice 0..63:
    read (scatter, extinction) from scatterExtinctionTex[x, y, slice]
    thickness = sliceDepth(slice+1) - sliceDepth(slice)
    sliceT = exp(-extinction * thickness)
    accumulatedLight += transmittance * scatter * (1 - sliceT) / max(extinction, 0.0001)
    transmittance *= sliceT
    write vec4(accumulatedLight, transmittance) to accumTex[x, y, slice]
```

**Public API:**
```typescript
class FroxelGrid {
    constructor(device: GPUDevice, options?: { gridW, gridH, gridD, near, far })

    // Textures for external injection passes to write into
    get scatterExtinctionTex(): GPUTexture

    // Shared WGSL code for injection shaders
    static readonly WGSL_HELPERS: string

    // Called by effects after their injection pass writes scatterExtinctionTex
    temporalReproject(encoder: GPUCommandEncoder, camera: Camera, prevVP: Float32Array): void
    accumulate(encoder: GPUCommandEncoder): void

    // For composite: sample this to get fog at a given depth
    get accumTex(): GPUTexture

    resize(gridW, gridH, gridD): void
    destroy(): void
}
```

### VolumetricFogEffect — `src/postprocessing/effects/VolumetricFogEffect.ts`

`PostProcessingEffect` subclass. Owns the fog injection pass and composite pass.
References `FroxelGrid` and `ShadowMap` as constructor dependencies.

**Constructor options:**
```typescript
interface VolumetricFogOptions {
    froxelGrid: FroxelGrid;
    shadowMap: ShadowMap;
    lightDirection?: [number, number, number];  // normalized, world space
    lightColor?: [number, number, number];      // linear RGB
    baseDensity?: number;       // default 0.02
    heightFalloff?: number;     // default 0.1
    extinctionCoeff?: number;   // default 1.0
    anisotropy?: number;        // Henyey-Greenstein g, default 0.6
    windDirection?: [number, number, number]; // world space offset per second
}
```

**Pass 1 — Fog injection** (compute, 4×4×4 workgroups over froxel grid):

Per froxel:
1. Compute world position via `froxelToWorld()`
2. Fog density: `baseDensity * exp(-heightFalloff * worldPos.y)`
3. Apply wind: offset world position by `windDirection * time` before density lookup
4. Extinction: `density * extinctionCoeff`
5. Shadow: transform world pos to light space, sample `shadowDepthTexture`, binary visibility
6. Phase function: `HG(dot(viewDir, lightDir), anisotropy)`
7. In-scattering: `density * lightColor * visibility * HG`
8. Write `vec4(scatter.rgb, extinction)` to `froxelGrid.scatterExtinctionTex`

**Pass 2 — Temporal + Accumulation** (delegated to FroxelGrid):
```typescript
this._froxelGrid.temporalReproject(encoder, camera, this._prevVP);
this._froxelGrid.accumulate(encoder);
```

**Pass 3 — Composite** (compute, 8×8 workgroups, full screen resolution):

Per pixel:
1. Read scene depth → linear depth
2. Convert to fractional slice index via `depthToSlice()`
3. Manual trilinear sample of `accumTex` at `(screenUV * gridSize, sliceFloat)`
   (interpolate between two adjacent slices)
4. Read `accumulatedLight` and `transmittance`
5. Composite: `finalColor = sceneColor * transmittance + accumulatedLight`
6. Write to output texture

## Params Buffer

Single uniform buffer updated per frame (fits one 256-byte aligned block):

```wgsl
struct FogParams {
    invViewProj      : mat4x4f,    // 64 bytes
    lightViewProj    : mat4x4f,    // 64 bytes
    lightDir         : vec3f,      // 12 bytes
    baseDensity      : f32,        //  4 bytes
    lightColor       : vec3f,      // 12 bytes
    heightFalloff    : f32,        //  4 bytes
    windOffset       : vec3f,      // 12 bytes
    extinctionCoeff  : f32,        //  4 bytes
    anisotropy       : f32,        //  4 bytes
    near             : f32,        //  4 bytes
    far              : f32,        //  4 bytes
    time             : f32,        //  4 bytes
    gridW            : u32,        //  4 bytes
    gridH            : u32,        //  4 bytes
    gridD            : u32,        //  4 bytes
    screenWidth      : f32,        //  4 bytes
    screenHeight     : f32,        //  4 bytes
    _pad             : vec3f,      // 12 bytes
}                                  // Total: 224 bytes
```

## Dispatch Summary

| Pass | Scope | Workgroup | Dispatches |
|------|-------|-----------|------------|
| Shadow map | Scene geometry | Raster | 1 render pass |
| Fog injection | 160×90×64 froxels | 4×4×4 | 40×23×16 = 14,720 |
| Temporal reprojection | 160×90×64 froxels | 4×4×4 | 40×23×16 = 14,720 |
| Accumulation | 160×90 columns | 8×8 | 20×12 = 240 |
| Composite | Full screen | 8×8 | ceil(W/8)×ceil(H/8) |

## VRAM Budget

| Resource | Size |
|----------|------|
| Shadow map (2048² depth32float) | 16 MB |
| scatterExtinctionTex (160×90×64 rgba16float) | 7 MB |
| historyTex (same) | 7 MB |
| accumTex (same) | 7 MB |
| **Total** | **~37 MB** |

## File Structure

```
src/
  shadows/
    ShadowMap.ts              — Shadow map class + shadow vertex shader
  froxels/
    FroxelGrid.ts             — 3D grid, temporal, accumulation, WGSL helpers
  postprocessing/
    effects/
      VolumetricFogEffect.ts  — Injection + composite (PostProcessingEffect)
```

## Integration in Example

```typescript
const shadowMap = new ShadowMap(renderer.gpuDevice, { resolution: 2048 });
const froxelGrid = new FroxelGrid(renderer.gpuDevice, {
    gridW: 160, gridH: 90, gridD: 64,
    near: camera.near, far: camera.far
});

const volume = new PostProcessingVolume(renderer, [
    new SSAOEffect({ ... }),
    new VolumetricFogEffect({
        froxelGrid,
        shadowMap,
        lightDirection: [0.5, -0.8, 0.3],
        lightColor: [1.0, 0.95, 0.85],
        baseDensity: 0.02,
        anisotropy: 0.6,
    }),
    new DepthOfFieldEffect({ ... }),
]);

function animate(now) {
    shadowMap.render(renderer, scene, camera, lightDir);
    volume.render(scene, camera);
    requestAnimationFrame(animate);
}
```

## Future Reuse

| Future effect | Reuses | New code needed |
|--------------|--------|-----------------|
| Clustered lighting | FroxelGrid (grid + helpers) | Light list injection, per-pixel light loop |
| Volumetric GI | FroxelGrid + ShadowMap | Irradiance injection from probes/RSM |
| Participating media (smoke) | FroxelGrid + temporal + accumulation | Density injection from 3D noise/simulation |
