# Fluid Density Field & Surface Rendering — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reusable 3D density field compute pass (particles → 3D texture) and a ray-march surface rendering effect, designed to plug into the PostProcessing pipeline and later the path tracer.

**Architecture:** `FluidDensityField` is a standalone compute object that reuses the sim's spatial grid to splat particle density into a 3D `rgba16float` storage texture. `FluidSurfaceEffect` extends `PostProcessingEffect` and ray-marches the density texture to render the fluid surface with normals and shading. Atomic accumulation uses a flat `r32uint` buffer (same pattern as `grid-assign.wgsl`) because `texture_storage_3d` doesn't support atomics.

**Tech Stack:** WebGPU compute shaders (WGSL), TypeScript

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `src/simulations/fluid/shaders/density-field-clear.wgsl.ts` | Compute shader: zero-fill the 3D density texture |
| Create | `src/simulations/fluid/shaders/density-field-splat.wgsl.ts` | Compute shader: scatter particle density into atomic buffer |
| Create | `src/simulations/fluid/shaders/density-field-copy.wgsl.ts` | Compute shader: convert atomic buffer → 3D texture |
| Create | `src/simulations/fluid/FluidDensityField.ts` | Class owning 3D texture + clear/splat/copy pipelines |
| Create | `src/postprocessing/effects/shaders/fluid-surface.wgsl.ts` | Compute shader: ray-march density field, shade surface |
| Create | `src/postprocessing/effects/FluidSurfaceEffect.ts` | PostProcessingEffect subclass for fluid rendering |
| Modify | `src/simulations/fluid/FluidSimulation.ts` | Expose `positionsBuffer` and `paramsBuffer` via public getters |
| Modify | `src/main.ts` | Export new classes |

---

## Task 1: Expose simulation buffers

**Files:**
- Modify: `src/simulations/fluid/FluidSimulation.ts`

- [ ] **Step 1: Add public getters for positionsBuffer and paramsBuffer**

These are needed by `FluidDensityField` to read particle positions and sim params without duplicating data.

```typescript
// Add after the existing public methods (around line 527)

public get positionsBufferRef(): ComputeBuffer {
    return this.positionsBuffer;
}

public get paramsBufferRef(): ComputeBuffer {
    return this.paramsBuffer;
}

public get cellOffsetsBufferRef(): ComputeBuffer {
    return this.cellOffsetsBuffer;
}

public get sortedIndicesBufferRef(): ComputeBuffer {
    return this.sortedIndicesBuffer;
}

public get gridDimsRef(): [number, number, number] {
    return this.gridDims;
}

public get gridOriginRef(): [number, number, number] {
    return this.gridOrigin;
}
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/FluidSimulation.ts
git commit -m "feat: expose fluid sim buffers for density field consumption"
```

---

## Task 2: Density field clear shader

**Files:**
- Create: `src/simulations/fluid/shaders/density-field-clear.wgsl.ts`

- [ ] **Step 1: Create the clear shader**

Zeroes out a 3D `rgba16float` storage texture. Workgroup `(4,4,4)` = 64 threads covering a 3D tile.

```typescript
export const shaderCode = /* wgsl */`

@group(0) @binding(0) var densityTex: texture_storage_3d<rgba16float, write>;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = textureDimensions(densityTex);
    if (any(gid >= dims)) { return; }
    textureStore(densityTex, gid, vec4<f32>(0.0));
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/density-field-clear.wgsl.ts
git commit -m "feat: add density field clear shader"
```

---

## Task 3: Density field splat shader

**Files:**
- Create: `src/simulations/fluid/shaders/density-field-splat.wgsl.ts`

- [ ] **Step 1: Create the splat shader**

One thread per particle. Maps particle world position into the 3D texture's voxel space, then scatters a smooth kernel into surrounding voxels using atomic u32 accumulation (same pattern as `grid-assign.wgsl` cell counts).

```typescript
import { simParamsStruct } from './sim-params.wgsl';

export const shaderCode = /* wgsl */`
${simParamsStruct}

struct DensityFieldParams {
    texDims       : vec3<u32>,
    particleCount : u32,
    boundsMin     : vec3<f32>,
    smoothingRadius: f32,
    boundsMax     : vec3<f32>,
    kernelScale   : f32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> accumBuffer: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> fieldParams: DensityFieldParams;

fn worldToVoxel(worldPos: vec3<f32>) -> vec3<f32> {
    let norm = (worldPos - fieldParams.boundsMin) / (fieldParams.boundsMax - fieldParams.boundsMin);
    return norm * vec3<f32>(fieldParams.texDims);
}

fn voxelIndex(coord: vec3<u32>) -> u32 {
    return coord.z * fieldParams.texDims.x * fieldParams.texDims.y
         + coord.y * fieldParams.texDims.x
         + coord.x;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= fieldParams.particleCount) { return; }

    let pos = positions[idx].xyz;
    let voxelPos = worldToVoxel(pos);
    let h = fieldParams.smoothingRadius;
    let dims = fieldParams.texDims;

    // Compute kernel radius in voxel units
    let voxelSize = (fieldParams.boundsMax - fieldParams.boundsMin) / vec3<f32>(dims);
    let kernelVoxels = vec3<i32>(ceil(vec3<f32>(h) / voxelSize));

    let centerVoxel = vec3<i32>(floor(voxelPos));

    for (var dz = -kernelVoxels.z; dz <= kernelVoxels.z; dz++) {
        for (var dy = -kernelVoxels.y; dy <= kernelVoxels.y; dy++) {
            for (var dx = -kernelVoxels.x; dx <= kernelVoxels.x; dx++) {
                let voxel = centerVoxel + vec3<i32>(dx, dy, dz);

                // Bounds check
                if (any(voxel < vec3<i32>(0)) || any(voxel >= vec3<i32>(dims))) {
                    continue;
                }

                // World-space distance from particle to voxel center
                let voxelCenter = (vec3<f32>(voxel) + 0.5) * voxelSize + fieldParams.boundsMin;
                let dist = length(pos - voxelCenter);

                if (dist >= h) { continue; }

                // Poly6-like kernel (smooth, non-negative)
                let t = 1.0 - dist / h;
                let weight = t * t * t * fieldParams.kernelScale;

                // Atomic accumulate (fixed-point: multiply by 1024, store as u32)
                let fixedPoint = u32(weight * 1024.0);
                if (fixedPoint > 0u) {
                    let vi = voxelIndex(vec3<u32>(voxel));
                    atomicAdd(&accumBuffer[vi], fixedPoint);
                }
            }
        }
    }
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/density-field-splat.wgsl.ts
git commit -m "feat: add density field splat shader with atomic accumulation"
```

---

## Task 4: Density field copy shader

**Files:**
- Create: `src/simulations/fluid/shaders/density-field-copy.wgsl.ts`

- [ ] **Step 1: Create the copy shader**

Converts the atomic `u32` accumulation buffer back to float and writes into the 3D texture. Also computes a running max for normalization.

```typescript
export const shaderCode = /* wgsl */`

struct DensityFieldParams {
    texDims       : vec3<u32>,
    particleCount : u32,
    boundsMin     : vec3<f32>,
    smoothingRadius: f32,
    boundsMax     : vec3<f32>,
    kernelScale   : f32,
};

@group(0) @binding(0) var<storage, read_write> accumBuffer: array<u32>;
@group(0) @binding(1) var densityTex: texture_storage_3d<rgba16float, write>;
@group(0) @binding(2) var<uniform> fieldParams: DensityFieldParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dims = fieldParams.texDims;
    if (any(gid >= dims)) { return; }

    let vi = gid.z * dims.x * dims.y + gid.y * dims.x + gid.x;
    let raw = f32(accumBuffer[vi]) / 1024.0;

    // Store density in R channel. G,B,A reserved for future (normal, material ID).
    textureStore(densityTex, gid, vec4<f32>(raw, 0.0, 0.0, 1.0));

    // Clear accumulator for next frame
    accumBuffer[vi] = 0u;
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/shaders/density-field-copy.wgsl.ts
git commit -m "feat: add density field copy shader (atomic buffer → 3D texture)"
```

---

## Task 5: FluidDensityField class

**Files:**
- Create: `src/simulations/fluid/FluidDensityField.ts`

- [ ] **Step 1: Create the class**

```typescript
import { FluidSimulation } from './FluidSimulation';
import { shaderCode as clearShader } from './shaders/density-field-clear.wgsl';
import { shaderCode as splatShader } from './shaders/density-field-splat.wgsl';
import { shaderCode as copyShader } from './shaders/density-field-copy.wgsl';

export interface FluidDensityFieldOptions {
    resolution?: number; // max voxels per axis (default 64)
    kernelScale?: number; // density kernel amplitude (default 1.0)
}

class FluidDensityField {
    private _device: GPUDevice;
    private _sim: FluidSimulation;

    private _densityTex!: GPUTexture;
    private _densityView!: GPUTextureView;
    private _accumBuffer!: GPUBuffer;
    private _paramsBuffer!: GPUBuffer;
    private _paramsData: Float32Array;
    private _paramsU32: Uint32Array;

    private _clearPipeline!: GPUComputePipeline;
    private _splatPipeline!: GPUComputePipeline;
    private _copyPipeline!: GPUComputePipeline;

    private _clearBG!: GPUBindGroup;
    private _splatBG!: GPUBindGroup;
    private _copyBG!: GPUBindGroup;

    private _texDims: [number, number, number];
    private _kernelScale: number;

    constructor(device: GPUDevice, sim: FluidSimulation, options?: FluidDensityFieldOptions) {
        this._device = device;
        this._sim = sim;

        const maxRes = options?.resolution ?? 64;
        this._kernelScale = options?.kernelScale ?? 1.0;

        // Derive texture dims from sim grid, clamped to maxRes
        const gd = sim.gridDimsRef;
        this._texDims = [
            Math.min(gd[0], maxRes),
            Math.min(gd[1], maxRes),
            Math.max(Math.min(gd[2], maxRes), 1),
        ];

        this._paramsData = new Float32Array(12); // 48 bytes = 3 × vec4
        this._paramsU32 = new Uint32Array(this._paramsData.buffer);

        this._createResources();
        this._createPipelines();
        this._buildBindGroups();
    }

    get densityTexture(): GPUTexture { return this._densityTex; }
    get densityView(): GPUTextureView { return this._densityView; }
    get texDims(): [number, number, number] { return this._texDims; }
    get boundsMin(): [number, number, number] { return [...this._sim.worldBoundsMin] as [number, number, number]; }
    get boundsMax(): [number, number, number] { return [...this._sim.worldBoundsMax] as [number, number, number]; }

    private _createResources(): void {
        const [w, h, d] = this._texDims;

        this._densityTex = this._device.createTexture({
            label: 'FluidDensityField/DensityTex',
            size: [w, h, d],
            dimension: '3d',
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this._densityView = this._densityTex.createView();

        this._accumBuffer = this._device.createBuffer({
            label: 'FluidDensityField/AccumBuffer',
            size: w * h * d * 4, // u32 per voxel
            usage: GPUBufferUsage.STORAGE,
        });

        this._paramsBuffer = this._device.createBuffer({
            label: 'FluidDensityField/Params',
            size: 48,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    private _createPipelines(): void {
        const device = this._device;

        // Clear pipeline
        const clearModule = device.createShaderModule({ label: 'DensityField/Clear', code: clearShader });
        const clearBGL = device.createBindGroupLayout({
            label: 'DensityField/Clear BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
            ],
        });
        this._clearPipeline = device.createComputePipeline({
            label: 'DensityField/ClearPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [clearBGL] }),
            compute: { module: clearModule, entryPoint: 'main' },
        });

        // Splat pipeline
        const splatModule = device.createShaderModule({ label: 'DensityField/Splat', code: splatShader });
        const splatBGL = device.createBindGroupLayout({
            label: 'DensityField/Splat BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this._splatPipeline = device.createComputePipeline({
            label: 'DensityField/SplatPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [splatBGL] }),
            compute: { module: splatModule, entryPoint: 'main' },
        });

        // Copy pipeline
        const copyModule = device.createShaderModule({ label: 'DensityField/Copy', code: copyShader });
        const copyBGL = device.createBindGroupLayout({
            label: 'DensityField/Copy BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        this._copyPipeline = device.createComputePipeline({
            label: 'DensityField/CopyPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [copyBGL] }),
            compute: { module: copyModule, entryPoint: 'main' },
        });
    }

    private _buildBindGroups(): void {
        const device = this._device;
        const posBuffer = (this._sim.positionsBufferRef as any)._resource as GPUBuffer;

        this._clearBG = device.createBindGroup({
            label: 'DensityField/Clear BG',
            layout: this._clearPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: this._densityView },
            ],
        });

        this._splatBG = device.createBindGroup({
            label: 'DensityField/Splat BG',
            layout: this._splatPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: posBuffer } },
                { binding: 1, resource: { buffer: this._accumBuffer } },
                { binding: 2, resource: { buffer: this._paramsBuffer } },
            ],
        });

        this._copyBG = device.createBindGroup({
            label: 'DensityField/Copy BG',
            layout: this._copyPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: this._accumBuffer } },
                { binding: 1, resource: this._densityView },
                { binding: 2, resource: { buffer: this._paramsBuffer } },
            ],
        });
    }

    private _uploadParams(): void {
        const p = this._paramsData;
        const u = this._paramsU32;
        const bMin = this._sim.worldBoundsMin;
        const bMax = this._sim.worldBoundsMax;

        u[0] = this._texDims[0]; u[1] = this._texDims[1]; u[2] = this._texDims[2];
        u[3] = this._sim.params.maxParticles; // particleCount
        p[4] = bMin[0]; p[5] = bMin[1]; p[6] = bMin[2];
        p[7] = this._sim.params.smoothingRadius;
        p[8] = bMax[0]; p[9] = bMax[1]; p[10] = bMax[2];
        p[11] = this._kernelScale;

        this._device.queue.writeBuffer(this._paramsBuffer, 0, this._paramsData);
    }

    update(commandEncoder: GPUCommandEncoder): void {
        this._uploadParams();

        const [w, h, d] = this._texDims;

        // Pass 1: Clear density texture
        const clearPass = commandEncoder.beginComputePass({ label: 'DensityField/Clear' });
        clearPass.setPipeline(this._clearPipeline);
        clearPass.setBindGroup(0, this._clearBG);
        clearPass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        clearPass.end();

        // Pass 2: Splat particles into atomic buffer
        const particleCount = this._sim.params.maxParticles;
        const splatPass = commandEncoder.beginComputePass({ label: 'DensityField/Splat' });
        splatPass.setPipeline(this._splatPipeline);
        splatPass.setBindGroup(0, this._splatBG);
        splatPass.dispatchWorkgroups(Math.ceil(particleCount / 64));
        splatPass.end();

        // Pass 3: Copy atomic buffer → 3D texture
        const copyPass = commandEncoder.beginComputePass({ label: 'DensityField/Copy' });
        copyPass.setPipeline(this._copyPipeline);
        copyPass.setBindGroup(0, this._copyBG);
        copyPass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        copyPass.end();
    }

    destroy(): void {
        this._densityTex.destroy();
        this._accumBuffer.destroy();
        this._paramsBuffer.destroy();
    }
}

export { FluidDensityField };
```

- [ ] **Step 2: Commit**

```bash
git add src/simulations/fluid/FluidDensityField.ts
git commit -m "feat: add FluidDensityField class (particles → 3D density texture)"
```

---

## Task 6: Ray-march surface shader

**Files:**
- Create: `src/postprocessing/effects/shaders/fluid-surface.wgsl.ts`

- [ ] **Step 1: Create the shader**

Screen-space ray march through the 3D density field. At threshold crossing, compute normal from density gradient, shade with lighting.

```typescript
export const shaderCode = /* wgsl */`

struct SurfaceParams {
    invViewProj : mat4x4<f32>,
    cameraPos   : vec3<f32>,
    densityThreshold: f32,
    boundsMin   : vec3<f32>,
    absorption  : f32,
    boundsMax   : vec3<f32>,
    densityScale: f32,
    fluidColor  : vec3<f32>,
    stepCount   : u32,
    screenWidth : u32,
    screenHeight: u32,
    _pad0       : u32,
    _pad1       : u32,
};

@group(0) @binding(0) var inputTex   : texture_2d<f32>;
@group(0) @binding(1) var depthTex   : texture_depth_2d;
@group(0) @binding(2) var outputTex  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(3) var densityTex : texture_3d<f32>;
@group(0) @binding(4) var densitySamp: sampler;
@group(0) @binding(5) var<uniform> params: SurfaceParams;

fn worldToUVW(worldPos: vec3<f32>) -> vec3<f32> {
    return (worldPos - params.boundsMin) / (params.boundsMax - params.boundsMin);
}

fn sampleDensity(worldPos: vec3<f32>) -> f32 {
    let uvw = worldToUVW(worldPos);
    if (any(uvw < vec3<f32>(0.0)) || any(uvw > vec3<f32>(1.0))) { return 0.0; }
    return textureSampleLevel(densityTex, densitySamp, uvw, 0.0).r * params.densityScale;
}

fn computeNormal(worldPos: vec3<f32>) -> vec3<f32> {
    let eps = length(params.boundsMax - params.boundsMin) / 128.0;
    let dx = sampleDensity(worldPos + vec3<f32>(eps, 0.0, 0.0)) - sampleDensity(worldPos - vec3<f32>(eps, 0.0, 0.0));
    let dy = sampleDensity(worldPos + vec3<f32>(0.0, eps, 0.0)) - sampleDensity(worldPos - vec3<f32>(0.0, eps, 0.0));
    let dz = sampleDensity(worldPos + vec3<f32>(0.0, 0.0, eps)) - sampleDensity(worldPos - vec3<f32>(0.0, 0.0, eps));
    let n = vec3<f32>(dx, dy, dz);
    let l = length(n);
    if (l < 0.0001) { return vec3<f32>(0.0, 1.0, 0.0); }
    return -n / l; // Points outward (toward decreasing density)
}

// Ray-AABB intersection (returns tMin, tMax)
fn intersectAABB(rayOrigin: vec3<f32>, rayDir: vec3<f32>, boxMin: vec3<f32>, boxMax: vec3<f32>) -> vec2<f32> {
    let invDir = 1.0 / rayDir;
    let t1 = (boxMin - rayOrigin) * invDir;
    let t2 = (boxMax - rayOrigin) * invDir;
    let tNear = min(t1, t2);
    let tFar  = max(t1, t2);
    let tMin = max(max(tNear.x, tNear.y), tNear.z);
    let tMax = min(min(tFar.x, tFar.y), tFar.z);
    return vec2<f32>(max(tMin, 0.0), tMax);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.screenWidth || gid.y >= params.screenHeight) { return; }

    let coord = vec2<i32>(gid.xy);
    let sceneColor = textureLoad(inputTex, coord, 0);
    let sceneDepth = textureLoad(depthTex, coord, 0);

    // Reconstruct world-space ray
    let uv = (vec2<f32>(gid.xy) + 0.5) / vec2<f32>(f32(params.screenWidth), f32(params.screenHeight));
    let ndc = vec2<f32>(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0);

    let nearClip = params.invViewProj * vec4<f32>(ndc, 0.0, 1.0);
    let farClip  = params.invViewProj * vec4<f32>(ndc, 1.0, 1.0);
    let nearWorld = nearClip.xyz / nearClip.w;
    let farWorld  = farClip.xyz / farClip.w;
    let rayDir = normalize(farWorld - nearWorld);
    let rayOrigin = params.cameraPos;

    // Intersect ray with density field bounds
    let tRange = intersectAABB(rayOrigin, rayDir, params.boundsMin, params.boundsMax);
    if (tRange.x >= tRange.y) {
        textureStore(outputTex, coord, sceneColor);
        return;
    }

    // Clamp march to scene depth
    let maxT = min(tRange.y, length(farWorld - nearWorld) * sceneDepth);
    let stepSize = (tRange.y - tRange.x) / f32(params.stepCount);

    // March through volume
    var transmittance = 1.0;
    var hitPos = vec3<f32>(0.0);
    var hit = false;

    for (var i = 0u; i < params.stepCount; i++) {
        let t = tRange.x + (f32(i) + 0.5) * stepSize;
        if (t > maxT) { break; }

        let samplePos = rayOrigin + rayDir * t;
        let density = sampleDensity(samplePos);

        if (density > params.densityThreshold) {
            hitPos = samplePos;
            hit = true;
            break;
        }

        transmittance *= exp(-density * params.absorption * stepSize);
        if (transmittance < 0.01) { break; }
    }

    var finalColor = sceneColor.rgb;

    if (hit) {
        let normal = computeNormal(hitPos);

        // Simple directional lighting
        let lightDir = normalize(vec3<f32>(0.3, 1.0, 0.5));
        let diffuse = max(dot(normal, lightDir), 0.0);
        let ambient = 0.15;
        let specDir = reflect(-lightDir, normal);
        let viewDir = normalize(params.cameraPos - hitPos);
        let spec = pow(max(dot(specDir, viewDir), 0.0), 32.0) * 0.5;

        let surfaceColor = params.fluidColor * (ambient + diffuse) + vec3<f32>(spec);
        finalColor = mix(surfaceColor, sceneColor.rgb, transmittance);
    } else {
        finalColor = mix(params.fluidColor * 0.1, sceneColor.rgb, transmittance);
    }

    textureStore(outputTex, coord, vec4<f32>(finalColor, sceneColor.a));
}
`;
```

- [ ] **Step 2: Commit**

```bash
git add src/postprocessing/effects/shaders/fluid-surface.wgsl.ts
git commit -m "feat: add fluid surface ray-march shader"
```

---

## Task 7: FluidSurfaceEffect class

**Files:**
- Create: `src/postprocessing/effects/FluidSurfaceEffect.ts`

- [ ] **Step 1: Create the class**

```typescript
import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { FluidDensityField } from '../../simulations/fluid/FluidDensityField';
import { shaderCode } from './shaders/fluid-surface.wgsl';
import { mat4 } from 'gl-matrix';

export interface FluidSurfaceOptions {
    densityField: FluidDensityField;
    fluidColor?: [number, number, number];
    absorption?: number;
    densityScale?: number;
    densityThreshold?: number;
    stepCount?: number;
}

class FluidSurfaceEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _densityField: FluidDensityField;

    fluidColor: [number, number, number];
    absorption: number;
    densityScale: number;
    densityThreshold: number;
    stepCount: number;

    private _pipeline: GPUComputePipeline | null = null;
    private _bgl: GPUBindGroupLayout | null = null;
    private _bg: GPUBindGroup | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private _sampler: GPUSampler | null = null;

    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    constructor(options: FluidSurfaceOptions) {
        super();
        this._densityField = options.densityField;
        this.fluidColor = options.fluidColor ?? [0.1, 0.4, 0.8];
        this.absorption = options.absorption ?? 2.0;
        this.densityScale = options.densityScale ?? 1.0;
        this.densityThreshold = options.densityThreshold ?? 0.5;
        this.stepCount = options.stepCount ?? 64;
    }

    initialize(device: GPUDevice, gbuffer: GBuffer, camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'FluidSurface/Params',
            size: 128, // 2 × mat4 = 128 bytes is safe upper bound
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._sampler = device.createSampler({
            label: 'FluidSurface/Sampler',
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'clamp-to-edge',
            addressModeV: 'clamp-to-edge',
            addressModeW: 'clamp-to-edge',
        });

        const module = device.createShaderModule({ label: 'FluidSurface/Shader', code: shaderCode });

        this._bgl = device.createBindGroupLayout({
            label: 'FluidSurface/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'FluidSurface/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildBindGroup(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        this._bg = this._device!.createBindGroup({
            label: 'FluidSurface/BG',
            layout: this._bgl!,
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: this._densityField.densityView },
                { binding: 4, resource: this._sampler! },
                { binding: 5, resource: { buffer: this._paramsBuffer! } },
            ],
        });
        this._currentInput = input;
        this._currentDepth = depth;
        this._currentOutput = output;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._pipeline) return;

        // Rebuild bind group if textures changed (ping-pong)
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildBindGroup(input, depth, output);
        }

        // Upload params
        const data = new Float32Array(32); // 128 bytes
        const u32 = new Uint32Array(data.buffer);
        const invViewProj = mat4.create();
        const viewProj = mat4.create();
        mat4.multiply(viewProj, camera.projectionMatrix.internalMat4 as unknown as mat4, camera.viewMatrix.internalMat4 as unknown as mat4);
        mat4.invert(invViewProj, viewProj);
        data.set(invViewProj as unknown as Float32Array, 0); // 0-15: invViewProj

        const camPos = camera.position;
        data[16] = camPos.x; data[17] = camPos.y; data[18] = camPos.z;
        data[19] = this.densityThreshold;

        const bMin = this._densityField.boundsMin;
        const bMax = this._densityField.boundsMax;
        data[20] = bMin[0]; data[21] = bMin[1]; data[22] = bMin[2];
        data[23] = this.absorption;
        data[24] = bMax[0]; data[25] = bMax[1]; data[26] = bMax[2];
        data[27] = this.densityScale;
        data[28] = this.fluidColor[0]; data[29] = this.fluidColor[1]; data[30] = this.fluidColor[2];
        u32[31] = this.stepCount;

        // Second row of u32s for screen dims
        // Reuse remaining space — we allocated 128 bytes = 32 floats
        // But SurfaceParams needs screenWidth/Height after stepCount
        // Let's extend the buffer slightly
        const data2 = new Float32Array(36);
        const u32_2 = new Uint32Array(data2.buffer);
        data2.set(data);
        u32_2[32] = width;
        u32_2[33] = height;
        u32_2[34] = 0;
        u32_2[35] = 0;

        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, data2.buffer, 0, 144);

        const pass = commandEncoder.beginComputePass({ label: 'FluidSurface/March' });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bg!);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(width: number, height: number, gbuffer: GBuffer): void {
        this._currentInput = null; // Force bind group rebuild on next render
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
    }
}

export { FluidSurfaceEffect };
```

- [ ] **Step 2: Fix params buffer size to match shader struct**

The `SurfaceParams` struct in the shader is:
- `invViewProj`: 64 bytes (mat4x4)
- `cameraPos + densityThreshold`: 16 bytes
- `boundsMin + absorption`: 16 bytes
- `boundsMax + densityScale`: 16 bytes
- `fluidColor + stepCount`: 16 bytes
- `screenWidth + screenHeight + pad`: 16 bytes
- **Total: 144 bytes**

Update `_paramsBuffer` size to 144:

```typescript
this._paramsBuffer = device.createBuffer({
    label: 'FluidSurface/Params',
    size: 144,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
```

- [ ] **Step 3: Commit**

```bash
git add src/postprocessing/effects/FluidSurfaceEffect.ts
git commit -m "feat: add FluidSurfaceEffect (ray-march PostProcessing effect)"
```

---

## Task 8: Exports

**Files:**
- Modify: `src/main.ts`

- [ ] **Step 1: Add exports**

```typescript
export { FluidDensityField } from "./simulations/fluid/FluidDensityField";
export type { FluidDensityFieldOptions } from "./simulations/fluid/FluidDensityField";
export { FluidSurfaceEffect } from "./postprocessing/effects/FluidSurfaceEffect";
export type { FluidSurfaceOptions } from "./postprocessing/effects/FluidSurfaceEffect";
```

- [ ] **Step 2: Commit**

```bash
git add src/main.ts
git commit -m "feat: export FluidDensityField and FluidSurfaceEffect"
```

---

## Task 9: Example — `index_fluid.html`

**Files:**
- Create: `examples/index_fluid.html`

- [ ] **Step 1: Create a 3D fluid blob example**

This example initializes particles in a sphere, runs the 3D fluid sim, feeds it into `FluidDensityField`, and renders with `FluidSurfaceEffect` through the `PostProcessingVolume`.

Key setup:
- Renderer with `sampleCount: 1` (compute effects don't use MSAA)
- GBuffer for the post-processing pipeline
- PostProcessingVolume with FluidSurfaceEffect
- Particles initialized in a sphere layout (custom buffer, no TextGeometry)
- FluidSimulation with `dimensions: 3`
- FluidDensityField updated each frame before rendering
- Tweakpane for fluid color, absorption, density scale, threshold

- [ ] **Step 2: Commit**

```bash
git add examples/index_fluid.html
git commit -m "feat: add 3D fluid blob example with density field rendering"
```

---

## Task 10: Build verification

- [ ] **Step 1: TypeScript check**

```bash
npx tsc --noEmit
```

- [ ] **Step 2: Vite build**

```bash
npx vite build
```

- [ ] **Step 3: Dev server test**

```bash
npx vite --open examples/index_fluid.html
```

Verify: particles form a blob, density field visualizes as a shaded surface, tweakpane controls work.

- [ ] **Step 4: Commit final**

```bash
git add -A
git commit -m "feat: complete fluid density field + surface rendering pipeline"
```
