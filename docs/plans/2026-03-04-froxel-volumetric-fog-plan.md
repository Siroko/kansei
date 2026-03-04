# Froxel Volumetric Fog — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add physically-based froxel volumetric fog with shadow-mapped light scattering and front-to-back accumulation to the Kansei WebGPU postprocessing pipeline.

**Architecture:** Three independent modules — `ShadowMap` (depth-only raster from light POV), `FroxelGrid` (reusable 3D frustum-aligned voxel grid with accumulation compute pass), `VolumetricFogEffect` (PostProcessingEffect subclass owning fog injection + composite compute passes). Shadow map and froxel grid are modular for future reuse (clustered lighting, volumetric GI).

**Tech Stack:** WebGPU compute shaders (WGSL), gl-matrix (already used by Matrix4), existing PostProcessingEffect / PostProcessingVolume infrastructure.

**Design doc:** `docs/plans/2026-03-04-froxel-volumetric-fog-design.md`

---

## Codebase Context for Implementers

### Existing PostProcessingEffect pattern
See `src/postprocessing/effects/SSAOEffect.ts` for the canonical example:
- Extends `PostProcessingEffect` (abstract class in `src/postprocessing/PostProcessingEffect.ts`)
- `initialize(device, gbuffer, camera)` — create pipeline, buffers, bind groups
- `render(commandEncoder, input, depth, output, camera, width, height)` — dispatch compute, rebuild bind group when input/output change (ping-pong)
- `resize(w, h, gbuffer)` — recreate size-dependent resources
- `destroy()` — release GPU resources

### PostProcessingVolume flow (`src/postprocessing/PostProcessingVolume.ts`)
1. `renderToGBuffer(scene, camera, gbuffer)` — uploads all object matrices, renders scene via render bundles into rgba16float color + depth32float depth
2. Effects chain — single command encoder, each effect reads previous output + depth, writes to ping-pong texture
3. Blit — final texture rendered to canvas

### Renderer matrix system (`src/renderers/Renderer.ts`)
- All object world + normal matrices packed into two large aligned GPU buffers
- `device.limits.minUniformBufferOffsetAlignment` (typically 256 bytes) stride per object
- Bind group 1 has two dynamic-offset bindings: `@binding(0)` normalMatrix, `@binding(1)` worldMatrix
- Each draw uses `setBindGroup(1, sharedMeshBG, [offset, offset])` where offset = objectIndex × alignment
- Exposed via: `renderer.sharedMeshBindGroupLayout`, `renderer.sharedMeshBindGroup`, `renderer.matrixAlignment`

### Vertex layout (all geometries)
- `@location(0) position : vec4f` (16 bytes)
- `@location(1) normal : vec3f` (12 bytes)
- `@location(2) uv : vec2f` (8 bytes)
- Stride: 36 bytes (packed), or 40 bytes with padding — check `geometry.vertexBuffersDescriptors`

### Matrix4 (`src/math/Matrix4.ts`)
Wraps gl-matrix `mat4`. Has: `perspective()`, `multiply()`, `invert()`, `identity()`, `copy()`, `translate()`, `scale()`, `rotate*()`. Does NOT have `lookAt()` or `ortho()` — Task 1 adds these.

### Camera (`src/cameras/Camera.ts`)
- `camera.viewMatrix`, `camera.inverseViewMatrix`, `camera.projectionMatrix` — all Matrix4
- `camera.near`, `camera.far`, `camera.fov`, `camera.aspect`
- Camera bind group (group 2): `@binding(0) viewMatrix`, `@binding(1) projectionMatrix`

---

## Task 1: Add lookAt and ortho to Matrix4

**Files:**
- Modify: `src/math/Matrix4.ts`

### Step 1: Add lookAt method

After the existing `perspective()` method (~line 85), add:

```typescript
/**
 * Sets this matrix to a look-at view matrix.
 * @param eye    Camera position.
 * @param center Target position to look at.
 * @param up     Up direction.
 */
lookAt(eye: vec3, center: vec3, up: vec3): Matrix4 {
    mat4.lookAt(this.internalMat4, eye, center, up);
    this.updateBuffer();
    return this;
}
```

### Step 2: Add ortho method

```typescript
/**
 * Sets this matrix to an orthographic projection matrix.
 */
ortho(left: number, right: number, bottom: number, top: number, near: number, far: number): Matrix4 {
    mat4.ortho(this.internalMat4, left, right, bottom, top, near, far);
    this.updateBuffer();
    return this;
}
```

### Step 3: Verify build compiles

Run: `npx tsc --noEmit`
Expected: No errors related to Matrix4.

### Step 4: Commit

```bash
git add src/math/Matrix4.ts
git commit -m "feat(math): add lookAt and ortho methods to Matrix4"
```

---

## Task 2: ShadowMap

**Files:**
- Create: `src/shadows/ShadowMap.ts`

### Step 1: Create the ShadowMap class

```typescript
import { vec3, mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { Matrix4 } from '../math/Matrix4';
import { InstancedGeometry } from '../geometries/InstancedGeometry';
import { Renderable } from '../objects/Renderable';
import { Renderer } from '../renderers/Renderer';
import { Scene } from '../objects/Scene';

export interface ShadowMapOptions {
    resolution?: number;
}

class ShadowMap {
    private _device: GPUDevice;
    private _resolution: number;
    private _depthTexture: GPUTexture;
    private _lightVP = new Float32Array(16);

    // Shadow render pipeline (depth-only, no fragment)
    private _pipeline: GPURenderPipeline | null = null;
    private _shaderModule: GPUShaderModule | null = null;

    // Light VP uniform
    private _lightVPBuffer: GPUBuffer;
    private _lightVPBGL: GPUBindGroupLayout;
    private _lightVPBG: GPUBindGroup;

    // Own mesh matrix buffers (same layout as Renderer's shared mesh)
    private _meshBGL: GPUBindGroupLayout;
    private _worldMatBuf: GPUBuffer | null = null;
    private _normalMatBuf: GPUBuffer | null = null;
    private _meshBG: GPUBindGroup | null = null;
    private _worldStaging: Float32Array | null = null;
    private _normalStaging: Float32Array | null = null;
    private _objectCapacity = 0;

    // Scratch matrices
    private _lightView = mat4.create();
    private _lightProj = mat4.create();
    private _lightVPMat = mat4.create();
    private _invViewProj = mat4.create();

    constructor(device: GPUDevice, options?: ShadowMapOptions) {
        this._device = device;
        this._resolution = options?.resolution ?? 2048;

        // Depth texture
        this._depthTexture = device.createTexture({
            label: 'ShadowMap/Depth',
            size: [this._resolution, this._resolution],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Light VP uniform buffer (64 bytes = mat4x4f)
        this._lightVPBuffer = device.createBuffer({
            label: 'ShadowMap/LightVP',
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Bind group layout for group 0: lightViewProj
        this._lightVPBGL = device.createBindGroupLayout({
            label: 'ShadowMap/LightVP BGL',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' },
            }],
        });

        this._lightVPBG = device.createBindGroup({
            label: 'ShadowMap/LightVP BG',
            layout: this._lightVPBGL,
            entries: [{ binding: 0, resource: { buffer: this._lightVPBuffer } }],
        });

        // Bind group layout for group 1: mesh matrices (dynamic offset, matches renderer)
        this._meshBGL = device.createBindGroupLayout({
            label: 'ShadowMap/Mesh BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
            ],
        });
    }

    get depthTexture(): GPUTexture { return this._depthTexture; }
    get lightViewProjMatrix(): Float32Array { return this._lightVP; }

    /**
     * Compute orthographic light view-projection matrix fitted to camera frustum.
     */
    private _computeLightVP(camera: Camera, lightDir: [number, number, number]): void {
        // 1. Build camera inverse view-projection
        const vp = mat4.create();
        mat4.multiply(vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);
        mat4.invert(this._invViewProj, vp);

        // 2. Compute 8 frustum corners in world space
        const ndcCorners = [
            [-1, -1, 0], [ 1, -1, 0], [-1,  1, 0], [ 1,  1, 0], // near
            [-1, -1, 1], [ 1, -1, 1], [-1,  1, 1], [ 1,  1, 1], // far
        ];

        const worldCorners: vec3[] = [];
        const temp = new Float32Array(4);
        for (const ndc of ndcCorners) {
            const clip = [ndc[0], ndc[1], ndc[2], 1.0];
            // Transform by invViewProj
            const x = this._invViewProj[0]*clip[0] + this._invViewProj[4]*clip[1] + this._invViewProj[8]*clip[2]  + this._invViewProj[12]*clip[3];
            const y = this._invViewProj[1]*clip[0] + this._invViewProj[5]*clip[1] + this._invViewProj[9]*clip[2]  + this._invViewProj[13]*clip[3];
            const z = this._invViewProj[2]*clip[0] + this._invViewProj[6]*clip[1] + this._invViewProj[10]*clip[2] + this._invViewProj[14]*clip[3];
            const w = this._invViewProj[3]*clip[0] + this._invViewProj[7]*clip[1] + this._invViewProj[11]*clip[2] + this._invViewProj[15]*clip[3];
            worldCorners.push(vec3.fromValues(x/w, y/w, z/w));
        }

        // 3. Frustum center
        const center = vec3.create();
        for (const c of worldCorners) vec3.add(center, center, c);
        vec3.scale(center, center, 1 / 8);

        // 4. Light view matrix (looking from light direction toward center)
        const ld = vec3.fromValues(lightDir[0], lightDir[1], lightDir[2]);
        vec3.normalize(ld, ld);
        const eye = vec3.create();
        vec3.scaleAndAdd(eye, center, ld, -100); // back up from center along -lightDir
        const up = Math.abs(ld[1]) < 0.99
            ? vec3.fromValues(0, 1, 0)
            : vec3.fromValues(1, 0, 0);
        mat4.lookAt(this._lightView, eye, center, up);

        // 5. Transform frustum to light space, find AABB
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        const lp = vec3.create();
        for (const wc of worldCorners) {
            vec3.transformMat4(lp, wc, this._lightView);
            minX = Math.min(minX, lp[0]); maxX = Math.max(maxX, lp[0]);
            minY = Math.min(minY, lp[1]); maxY = Math.max(maxY, lp[1]);
            minZ = Math.min(minZ, lp[2]); maxZ = Math.max(maxZ, lp[2]);
        }

        // Extend Z range to catch shadow casters behind the camera frustum
        const zRange = maxZ - minZ;
        minZ -= zRange * 2;

        // 6. Orthographic projection
        mat4.ortho(this._lightProj, minX, maxX, minY, maxY, minZ, maxZ);

        // 7. Combined lightVP
        mat4.multiply(this._lightVPMat, this._lightProj, this._lightView);
        this._lightVP.set(this._lightVPMat as unknown as Float32Array);
    }

    /**
     * Ensure mesh matrix buffers are large enough for objectCount.
     */
    private _ensureMeshBuffers(objectCount: number): void {
        if (objectCount <= this._objectCapacity && this._meshBG) return;

        const alignment = this._device.limits.minUniformBufferOffsetAlignment ?? 256;
        const bufferSize = Math.max(objectCount * alignment, alignment);
        const floatsPerSlot = alignment / 4;

        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();

        this._worldMatBuf = this._device.createBuffer({
            label: 'ShadowMap/WorldMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._normalMatBuf = this._device.createBuffer({
            label: 'ShadowMap/NormalMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._worldStaging = new Float32Array(objectCount * floatsPerSlot);
        this._normalStaging = new Float32Array(objectCount * floatsPerSlot);

        this._meshBG = this._device.createBindGroup({
            label: 'ShadowMap/Mesh BG',
            layout: this._meshBGL,
            entries: [
                { binding: 0, resource: { buffer: this._normalMatBuf, size: 64 } },
                { binding: 1, resource: { buffer: this._worldMatBuf, size: 64 } },
            ],
        });

        this._objectCapacity = objectCount;
    }

    /**
     * Build the shadow render pipeline lazily (needs vertex buffer descriptors from first geometry).
     */
    private _ensurePipeline(vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>): void {
        if (this._pipeline) return;

        const shaderCode = /* wgsl */`
            @group(0) @binding(0) var<uniform> lightViewProj : mat4x4f;
            @group(1) @binding(0) var<uniform> normalMatrix  : mat4x4f;
            @group(1) @binding(1) var<uniform> worldMatrix   : mat4x4f;

            @vertex
            fn shadow_vs(
                @location(0) position : vec4f,
                @location(1) normal   : vec3f,
                @location(2) uv       : vec2f,
            ) -> @builtin(position) vec4f {
                return lightViewProj * worldMatrix * position;
            }
        `;

        this._shaderModule = this._device.createShaderModule({
            label: 'ShadowMap/Shader',
            code: shaderCode,
        });

        const pipelineLayout = this._device.createPipelineLayout({
            label: 'ShadowMap/PipelineLayout',
            bindGroupLayouts: [this._lightVPBGL, this._meshBGL],
        });

        this._pipeline = this._device.createRenderPipeline({
            label: 'ShadowMap/Pipeline',
            layout: pipelineLayout,
            vertex: {
                module: this._shaderModule,
                entryPoint: 'shadow_vs',
                buffers: vertexBuffersDescriptors,
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });
    }

    /**
     * Render the shadow map from the light's POV.
     * Call BEFORE volume.render() each frame.
     */
    render(renderer: Renderer, scene: Scene, camera: Camera, lightDir: [number, number, number]): void {
        const device = this._device;

        // Prepare scene (computes world matrices)
        scene.prepare(camera);
        camera.updateViewMatrix();
        const objects = scene.getOrderedObjects();
        if (objects.length === 0) return;

        // Compute light VP
        this._computeLightVP(camera, lightDir);
        device.queue.writeBuffer(this._lightVPBuffer, 0, this._lightVP.buffer as ArrayBuffer);

        // Ensure pipeline exists (use first geometry's vertex layout)
        if (!objects[0].geometry.initialized) {
            objects[0].geometry.initialize(device);
        }
        this._ensurePipeline(objects[0].geometry.vertexBuffersDescriptors);

        // Ensure mesh buffers
        this._ensureMeshBuffers(objects.length);

        const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256;
        const floatsPerSlot = alignment / 4;

        // Upload world matrices
        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            if (!obj.geometry.initialized) obj.geometry.initialize(device);
            obj.updateModelMatrix();

            this._worldStaging!.set(obj.worldMatrix.internalMat4, i * floatsPerSlot);
            // Normal matrix not needed but buffer must exist for bind group layout
            this._normalStaging!.set(obj.normalMatrix.internalMat4, i * floatsPerSlot);
        }

        device.queue.writeBuffer(this._worldMatBuf!, 0, this._worldStaging!.buffer as ArrayBuffer);
        device.queue.writeBuffer(this._normalMatBuf!, 0, this._normalStaging!.buffer as ArrayBuffer);

        // Encode shadow render pass (depth-only)
        const commandEncoder = device.createCommandEncoder({ label: 'ShadowMap' });
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [],
            depthStencilAttachment: {
                view: this._depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        pass.setPipeline(this._pipeline!);
        pass.setBindGroup(0, this._lightVPBG);

        let currentVertexBuffer: GPUBuffer | null = null;
        let currentIndexBuffer: GPUBuffer | null = null;

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            if (!obj.geometry.initialized) continue;

            const offset = i * alignment;
            pass.setBindGroup(1, this._meshBG!, [offset, offset]);

            if (obj.geometry.vertexBuffer !== currentVertexBuffer) {
                pass.setVertexBuffer(0, obj.geometry.vertexBuffer!);
                currentVertexBuffer = obj.geometry.vertexBuffer!;
            }
            if (obj.geometry.indexBuffer !== currentIndexBuffer) {
                pass.setIndexBuffer(obj.geometry.indexBuffer!, obj.geometry.indexFormat!);
                currentIndexBuffer = obj.geometry.indexBuffer!;
            }

            if (obj.geometry.isInstancedGeometry) {
                const geo = obj.geometry as InstancedGeometry;
                let idx = 1;
                for (const extraBuf of geo.extraBuffers) {
                    if (!extraBuf.initialized) extraBuf.initialize(device);
                    pass.setVertexBuffer(idx++, extraBuf.resource.buffer);
                }
                pass.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
            } else {
                pass.drawIndexed(obj.geometry.vertexCount);
            }
        }

        pass.end();
        device.queue.submit([commandEncoder.finish()]);
    }

    destroy(): void {
        this._depthTexture?.destroy();
        this._lightVPBuffer?.destroy();
        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();
        this._pipeline = null;
    }
}

export { ShadowMap };
```

### Step 2: Verify build compiles

Run: `npx tsc --noEmit`
Expected: No errors.

### Step 3: Commit

```bash
git add src/shadows/ShadowMap.ts
git commit -m "feat(shadows): add ShadowMap class with frustum-fitted orthographic projection"
```

---

## Task 3: FroxelGrid

**Files:**
- Create: `src/froxels/FroxelGrid.ts`

### Step 1: Create the FroxelGrid class

```typescript
export interface FroxelGridOptions {
    gridW?: number;  // default 160
    gridH?: number;  // default 90
    gridD?: number;  // default 64
    near?: number;   // default 0.1
    far?: number;    // default 1000
}

class FroxelGrid {
    private _device: GPUDevice;
    private _gridW: number;
    private _gridH: number;
    private _gridD: number;
    private _near: number;
    private _far: number;

    private _scatterExtinctionTex!: GPUTexture;
    private _accumTex!: GPUTexture;

    private _accumPipeline: GPUComputePipeline | null = null;
    private _accumBG: GPUBindGroup | null = null;
    private _gridParamsBuffer: GPUBuffer;

    /**
     * WGSL helper functions for exponential depth slicing and coordinate conversion.
     * Inject into consumer shaders via string concatenation.
     */
    static readonly WGSL_HELPERS = /* wgsl */`
        fn sliceDepth(i: f32, near: f32, far: f32, numSlices: f32) -> f32 {
            return near * pow(far / near, i / numSlices);
        }

        fn depthToSlice(linearDepth: f32, near: f32, far: f32, numSlices: f32) -> f32 {
            return numSlices * log(linearDepth / near) / log(far / near);
        }

        fn froxelToWorld(coord: vec3f, invViewProj: mat4x4f, near: f32, far: f32, gridSize: vec3f) -> vec3f {
            let uv = (coord.xy + 0.5) / gridSize.xy;
            let linearD = sliceDepth(coord.z + 0.5, near, far, gridSize.z);
            // Reverse perspective: linear depth → NDC Z (WebGPU [0,1] range)
            let ndcZ = far * (linearD - near) / (linearD * (far - near));
            let ndcX = uv.x * 2.0 - 1.0;
            let ndcY = (1.0 - uv.y) * 2.0 - 1.0;
            let world = invViewProj * vec4f(ndcX, ndcY, ndcZ, 1.0);
            return world.xyz / world.w;
        }
    `;

    constructor(device: GPUDevice, options?: FroxelGridOptions) {
        this._device = device;
        this._gridW = options?.gridW ?? 160;
        this._gridH = options?.gridH ?? 90;
        this._gridD = options?.gridD ?? 64;
        this._near  = options?.near ?? 0.1;
        this._far   = options?.far ?? 1000;

        this._createTextures();

        // Grid params uniform (for accumulation shader)
        this._gridParamsBuffer = device.createBuffer({
            label: 'FroxelGrid/Params',
            size: 32, // near(4) + far(4) + gridW(4) + gridH(4) + gridD(4) + pad(12) = 32
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._uploadGridParams();
        this._createAccumPipeline();
    }

    get scatterExtinctionTex(): GPUTexture { return this._scatterExtinctionTex; }
    get accumTex(): GPUTexture { return this._accumTex; }
    get gridW(): number { return this._gridW; }
    get gridH(): number { return this._gridH; }
    get gridD(): number { return this._gridD; }
    get near(): number { return this._near; }
    get far(): number { return this._far; }

    private _createTextures(): void {
        const texUsage = GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING;
        const size: GPUExtent3D = [this._gridW, this._gridH, this._gridD];

        this._scatterExtinctionTex = this._device.createTexture({
            label: 'FroxelGrid/ScatterExtinction',
            size,
            dimension: '3d',
            format: 'rgba16float',
            usage: texUsage,
        });

        this._accumTex = this._device.createTexture({
            label: 'FroxelGrid/Accum',
            size,
            dimension: '3d',
            format: 'rgba16float',
            usage: texUsage,
        });
    }

    private _uploadGridParams(): void {
        const data = new Float32Array(8); // 32 bytes
        data[0] = this._near;
        data[1] = this._far;
        new Uint32Array(data.buffer, 8, 1)[0] = this._gridW;
        new Uint32Array(data.buffer, 12, 1)[0] = this._gridH;
        new Uint32Array(data.buffer, 16, 1)[0] = this._gridD;
        // [5..7] padding
        this._device.queue.writeBuffer(this._gridParamsBuffer, 0, data.buffer as ArrayBuffer);
    }

    private _createAccumPipeline(): void {
        const shaderCode = FroxelGrid.WGSL_HELPERS + /* wgsl */`

            struct GridParams {
                near  : f32,
                far   : f32,
                gridW : u32,
                gridH : u32,
                gridD : u32,
                _pad0 : f32,
                _pad1 : f32,
                _pad2 : f32,
            }

            @group(0) @binding(0) var scatterExtTex : texture_3d<f32>;
            @group(0) @binding(1) var accumOut       : texture_storage_3d<rgba16float, write>;
            @group(0) @binding(2) var<uniform> gp    : GridParams;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) gid : vec3u) {
                let x = gid.x;
                let y = gid.y;
                if (x >= gp.gridW || y >= gp.gridH) { return; }

                var transmittance = 1.0;
                var accLight = vec3f(0.0);

                for (var z = 0u; z < gp.gridD; z++) {
                    let data = textureLoad(scatterExtTex, vec3u(x, y, z), 0);
                    let scatter    = data.rgb;
                    let extinction = data.a;

                    let d0 = sliceDepth(f32(z), gp.near, gp.far, f32(gp.gridD));
                    let d1 = sliceDepth(f32(z + 1u), gp.near, gp.far, f32(gp.gridD));
                    let thickness = d1 - d0;

                    let sliceT = exp(-extinction * thickness);

                    // Integrate in-scattered light over this slice
                    accLight += transmittance * scatter * (1.0 - sliceT) / max(extinction, 0.0001);
                    transmittance *= sliceT;

                    textureStore(accumOut, vec3u(x, y, z), vec4f(accLight, transmittance));
                }
            }
        `;

        const module = this._device.createShaderModule({
            label: 'FroxelGrid/AccumShader',
            code: shaderCode,
        });

        const bgl = this._device.createBindGroupLayout({
            label: 'FroxelGrid/Accum BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._accumPipeline = this._device.createComputePipeline({
            label: 'FroxelGrid/AccumPipeline',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._accumBG = this._device.createBindGroup({
            label: 'FroxelGrid/Accum BG',
            layout: bgl,
            entries: [
                { binding: 0, resource: this._scatterExtinctionTex.createView() },
                { binding: 1, resource: this._accumTex.createView() },
                { binding: 2, resource: { buffer: this._gridParamsBuffer } },
            ],
        });
    }

    /**
     * Front-to-back accumulation pass.
     * Call after the injection pass has written to scatterExtinctionTex.
     */
    accumulate(encoder: GPUCommandEncoder): void {
        const pass = encoder.beginComputePass({ label: 'FroxelGrid/Accumulate' });
        pass.setPipeline(this._accumPipeline!);
        pass.setBindGroup(0, this._accumBG!);
        pass.dispatchWorkgroups(
            Math.ceil(this._gridW / 8),
            Math.ceil(this._gridH / 8)
        );
        pass.end();
    }

    resize(gridW: number, gridH: number, gridD: number): void {
        this._gridW = gridW;
        this._gridH = gridH;
        this._gridD = gridD;
        this._scatterExtinctionTex?.destroy();
        this._accumTex?.destroy();
        this._createTextures();
        this._uploadGridParams();
        // Rebuild accumulation bind group with new textures
        this._accumBG = null;
        this._createAccumPipeline();
    }

    destroy(): void {
        this._scatterExtinctionTex?.destroy();
        this._accumTex?.destroy();
        this._gridParamsBuffer?.destroy();
    }
}

export { FroxelGrid };
```

### Step 2: Verify build compiles

Run: `npx tsc --noEmit`

### Step 3: Commit

```bash
git add src/froxels/FroxelGrid.ts
git commit -m "feat(froxels): add FroxelGrid class with 3D textures and accumulation pass"
```

---

## Task 4: VolumetricFogEffect

**Files:**
- Create: `src/postprocessing/effects/VolumetricFogEffect.ts`

### Step 1: Create the VolumetricFogEffect class

This is the largest file. It creates two compute pipelines (injection + composite) and wires them together with the FroxelGrid and ShadowMap.

```typescript
import { Camera } from '../../cameras/Camera';
import { FroxelGrid } from '../../froxels/FroxelGrid';
import { ShadowMap } from '../../shadows/ShadowMap';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { mat4 } from 'gl-matrix';

export interface VolumetricFogOptions {
    froxelGrid: FroxelGrid;
    shadowMap: ShadowMap;
    lightDirection?: [number, number, number];
    lightColor?: [number, number, number];
    baseDensity?: number;
    heightFalloff?: number;
    extinctionCoeff?: number;
    anisotropy?: number;
    windDirection?: [number, number, number];
}

class VolumetricFogEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _froxelGrid: FroxelGrid;
    private _shadowMap: ShadowMap;

    private _lightDir: [number, number, number];
    private _lightColor: [number, number, number];
    private _baseDensity: number;
    private _heightFalloff: number;
    private _extinctionCoeff: number;
    private _anisotropy: number;
    private _windDir: [number, number, number];

    // Injection pass
    private _injectPipeline: GPUComputePipeline | null = null;
    private _injectBG: GPUBindGroup | null = null;
    private _fogParamsBuffer: GPUBuffer | null = null;

    // Composite pass
    private _compositePipeline: GPUComputePipeline | null = null;
    private _compositeBG: GPUBindGroup | null = null;
    private _compositeParamsBuffer: GPUBuffer | null = null;
    private _accumSampler: GPUSampler | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    // Time tracking for wind offset
    private _startTime = performance.now();

    // Scratch for invViewProj
    private _invVP = mat4.create();

    constructor(options: VolumetricFogOptions) {
        super();
        this._froxelGrid    = options.froxelGrid;
        this._shadowMap     = options.shadowMap;
        this._lightDir      = options.lightDirection ?? [0.5, -0.8, 0.3];
        this._lightColor    = options.lightColor ?? [1.0, 0.95, 0.85];
        this._baseDensity   = options.baseDensity ?? 0.02;
        this._heightFalloff = options.heightFalloff ?? 0.1;
        this._extinctionCoeff = options.extinctionCoeff ?? 1.0;
        this._anisotropy    = options.anisotropy ?? 0.6;
        this._windDir       = options.windDirection ?? [0, 0, 0];
    }

    // ── Injection shader ──────────────────────────────────────────────────

    private static _INJECT_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct FogParams {
            invViewProj     : mat4x4f,
            lightViewProj   : mat4x4f,
            cameraPos       : vec3f,
            baseDensity     : f32,
            lightDir        : vec3f,
            heightFalloff   : f32,
            lightColor      : vec3f,
            extinctionCoeff : f32,
            windOffset      : vec3f,
            anisotropy      : f32,
            near            : f32,
            far             : f32,
            time            : f32,
            gridW           : u32,
            gridH           : u32,
            gridD           : u32,
            _pad0           : f32,
            _pad1           : f32,
        }

        @group(0) @binding(0) var scatterExtTex  : texture_storage_3d<rgba16float, write>;
        @group(0) @binding(1) var shadowDepthTex  : texture_depth_2d;
        @group(0) @binding(2) var<uniform> params : FogParams;

        fn henyeyGreenstein(cosTheta: f32, g: f32) -> f32 {
            let g2 = g * g;
            return (1.0 - g2) / (4.0 * 3.14159265 * pow(1.0 + g2 - 2.0 * g * cosTheta, 1.5));
        }

        @compute @workgroup_size(4, 4, 4)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            if (gid.x >= params.gridW || gid.y >= params.gridH || gid.z >= params.gridD) { return; }

            let gridSize = vec3f(f32(params.gridW), f32(params.gridH), f32(params.gridD));
            let worldPos = froxelToWorld(
                vec3f(f32(gid.x), f32(gid.y), f32(gid.z)),
                params.invViewProj, params.near, params.far, gridSize
            );

            // Wind-displaced sample position for density
            let samplePos = worldPos + params.windOffset;

            // Height-exponential fog density
            let density = params.baseDensity * exp(-params.heightFalloff * max(samplePos.y, 0.0));
            let extinction = density * params.extinctionCoeff;

            // Shadow map lookup (binary visibility)
            let lightClip = params.lightViewProj * vec4f(worldPos, 1.0);
            let lightNDC  = lightClip.xyz / lightClip.w;
            let shadowUV  = vec2f(lightNDC.x * 0.5 + 0.5, 1.0 - (lightNDC.y * 0.5 + 0.5));

            var visibility = 1.0;
            if (shadowUV.x >= 0.0 && shadowUV.x <= 1.0 && shadowUV.y >= 0.0 && shadowUV.y <= 1.0) {
                let shadowDim = vec2f(textureDimensions(shadowDepthTex, 0));
                let shadowCoord = vec2i(vec2f(shadowUV * shadowDim));
                let shadowDepth = textureLoad(shadowDepthTex, shadowCoord, 0);
                // Bias to prevent shadow acne on fog voxels
                visibility = select(0.0, 1.0, lightNDC.z <= shadowDepth + 0.002);
            }

            // Phase function (Henyey-Greenstein)
            let viewDir  = normalize(worldPos - params.cameraPos);
            let cosTheta = dot(viewDir, -normalize(params.lightDir));
            let phase    = henyeyGreenstein(cosTheta, params.anisotropy);

            // In-scattered light
            let scatter = density * params.lightColor * visibility * phase;

            textureStore(scatterExtTex, gid, vec4f(scatter, extinction));
        }
    `;

    // ── Composite shader ──────────────────────────────────────────────────

    private static _COMPOSITE_SHADER = FroxelGrid.WGSL_HELPERS + /* wgsl */`

        struct CompositeParams {
            near         : f32,
            far          : f32,
            gridW        : f32,
            gridH        : f32,
            gridD        : f32,
            screenWidth  : f32,
            screenHeight : f32,
            _pad         : f32,
        }

        @group(0) @binding(0) var inputTex     : texture_2d<f32>;
        @group(0) @binding(1) var depthTex     : texture_depth_2d;
        @group(0) @binding(2) var outputTex    : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var accumTex     : texture_3d<f32>;
        @group(0) @binding(4) var accumSampler : sampler;
        @group(0) @binding(5) var<uniform> cp  : CompositeParams;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            if (f32(coord.x) >= cp.screenWidth || f32(coord.y) >= cp.screenHeight) { return; }

            let sceneColor = textureLoad(inputTex, coord, 0);
            let depth      = textureLoad(depthTex, coord, 0);

            // Sky — pass through unchanged
            if (depth >= 1.0) {
                textureStore(outputTex, coord, sceneColor);
                return;
            }

            // Reverse-perspective: NDC depth → linear depth
            let linearDepth = cp.near * cp.far / (cp.far - depth * (cp.far - cp.near));

            // Fractional slice in the froxel grid
            let sliceFloat = depthToSlice(linearDepth, cp.near, cp.far, cp.gridD);

            // UVW for 3D texture sampling (trilinear)
            let uv = vec2f(f32(coord.x) / cp.screenWidth, f32(coord.y) / cp.screenHeight);
            let gridUV = vec3f(uv.x, uv.y, clamp(sliceFloat / cp.gridD, 0.0, 1.0));

            let fogData = textureSampleLevel(accumTex, accumSampler, gridUV, 0.0);
            let accLight      = fogData.rgb;
            let transmittance = fogData.a;

            // Composite: scene * transmittance + accumulated in-scattered light
            let finalColor = sceneColor.rgb * transmittance + accLight;
            textureStore(outputTex, coord, vec4f(finalColor, sceneColor.a));
        }
    `;

    // ── PostProcessingEffect interface ───────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        // Fog params buffer (256 bytes — fits the FogParams struct with alignment)
        this._fogParamsBuffer = device.createBuffer({
            label: 'VolumetricFog/FogParams',
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Composite params buffer
        this._compositeParamsBuffer = device.createBuffer({
            label: 'VolumetricFog/CompositeParams',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Accumulation sampler (trilinear for smooth fog lookup)
        this._accumSampler = device.createSampler({
            label: 'VolumetricFog/AccumSampler',
            magFilter: 'linear',
            minFilter: 'linear',
        });

        // ── Injection pipeline ──
        const injectModule = device.createShaderModule({
            label: 'VolumetricFog/InjectShader',
            code: VolumetricFogEffect._INJECT_SHADER,
        });

        const injectBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/Inject BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float', viewDimension: '3d' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._injectPipeline = device.createComputePipeline({
            label: 'VolumetricFog/InjectPipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [injectBGL] }),
            compute: { module: injectModule, entryPoint: 'main' },
        });

        this._injectBG = device.createBindGroup({
            label: 'VolumetricFog/Inject BG',
            layout: injectBGL,
            entries: [
                { binding: 0, resource: this._froxelGrid.scatterExtinctionTex.createView() },
                { binding: 1, resource: this._shadowMap.depthTexture.createView() },
                { binding: 2, resource: { buffer: this._fogParamsBuffer } },
            ],
        });

        // ── Composite pipeline ──
        const compositeModule = device.createShaderModule({
            label: 'VolumetricFog/CompositeShader',
            code: VolumetricFogEffect._COMPOSITE_SHADER,
        });

        const compositeBGL = device.createBindGroupLayout({
            label: 'VolumetricFog/Composite BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, sampler: { type: 'filtering' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._compositePipeline = device.createComputePipeline({
            label: 'VolumetricFog/CompositePipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [compositeBGL] }),
            compute: { module: compositeModule, entryPoint: 'main' },
        });

        this._buildCompositeBG(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildCompositeBG(input: GPUTexture, depth: GPUTexture, output: GPUTexture): void {
        const bgl = this._compositePipeline!.getBindGroupLayout(0);
        this._compositeBG = this._device!.createBindGroup({
            label: 'VolumetricFog/Composite BG',
            layout: bgl,
            entries: [
                { binding: 0, resource: input.createView() },
                { binding: 1, resource: depth.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: this._froxelGrid.accumTex.createView() },
                { binding: 4, resource: this._accumSampler! },
                { binding: 5, resource: { buffer: this._compositeParamsBuffer! } },
            ],
        });
        this._currentInput  = input;
        this._currentDepth  = depth;
        this._currentOutput = output;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number
    ): void {
        if (!this._injectPipeline || !this._compositePipeline) return;

        const device = this._device!;
        const grid = this._froxelGrid;
        const time = (performance.now() - this._startTime) / 1000;

        // ── Update fog params ──
        const vp = mat4.create();
        mat4.multiply(vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);
        mat4.invert(this._invVP, vp);

        // Camera position from inverse view matrix
        const iv = camera.inverseViewMatrix.internalMat4;

        const fogParams = new Float32Array(64); // 256 bytes
        fogParams.set(this._invVP as unknown as Float32Array, 0);             // offset 0:  invViewProj (16 floats)
        fogParams.set(this._shadowMap.lightViewProjMatrix, 16);               // offset 64: lightViewProj (16 floats)
        fogParams[32] = iv[12]; fogParams[33] = iv[13]; fogParams[34] = iv[14]; // offset 128: cameraPos
        fogParams[35] = this._baseDensity;                                     // offset 140: baseDensity
        fogParams[36] = this._lightDir[0]; fogParams[37] = this._lightDir[1]; fogParams[38] = this._lightDir[2]; // offset 144: lightDir
        fogParams[39] = this._heightFalloff;                                   // offset 156: heightFalloff
        fogParams[40] = this._lightColor[0]; fogParams[41] = this._lightColor[1]; fogParams[42] = this._lightColor[2]; // offset 160: lightColor
        fogParams[43] = this._extinctionCoeff;                                 // offset 172: extinctionCoeff
        fogParams[44] = this._windDir[0] * time; fogParams[45] = this._windDir[1] * time; fogParams[46] = this._windDir[2] * time; // offset 176: windOffset
        fogParams[47] = this._anisotropy;                                      // offset 188: anisotropy
        fogParams[48] = grid.near;                                             // offset 192: near
        fogParams[49] = grid.far;                                              // offset 196: far
        fogParams[50] = time;                                                  // offset 200: time
        new Uint32Array(fogParams.buffer, 204, 1)[0] = grid.gridW;            // offset 204: gridW
        new Uint32Array(fogParams.buffer, 208, 1)[0] = grid.gridH;            // offset 208: gridH
        new Uint32Array(fogParams.buffer, 212, 1)[0] = grid.gridD;            // offset 212: gridD
        // [54..55] padding

        device.queue.writeBuffer(this._fogParamsBuffer!, 0, fogParams.buffer as ArrayBuffer);

        // ── Update composite params ──
        const compositeParams = new Float32Array(8); // 32 bytes
        compositeParams[0] = grid.near;
        compositeParams[1] = grid.far;
        compositeParams[2] = grid.gridW;
        compositeParams[3] = grid.gridH;
        compositeParams[4] = grid.gridD;
        compositeParams[5] = width;
        compositeParams[6] = height;
        device.queue.writeBuffer(this._compositeParamsBuffer!, 0, compositeParams.buffer as ArrayBuffer);

        // Rebuild composite bind group on texture change (ping-pong)
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildCompositeBG(input, depth, output);
        }

        // ── Pass 1: Fog injection ──
        const injectPass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Inject' });
        injectPass.setPipeline(this._injectPipeline!);
        injectPass.setBindGroup(0, this._injectBG!);
        injectPass.dispatchWorkgroups(
            Math.ceil(grid.gridW / 4),
            Math.ceil(grid.gridH / 4),
            Math.ceil(grid.gridD / 4)
        );
        injectPass.end();

        // ── Pass 2: Front-to-back accumulation ──
        grid.accumulate(commandEncoder);

        // ── Pass 3: Composite fog onto scene ──
        const compositePass = commandEncoder.beginComputePass({ label: 'VolumetricFog/Composite' });
        compositePass.setPipeline(this._compositePipeline!);
        compositePass.setBindGroup(0, this._compositeBG!);
        compositePass.dispatchWorkgroups(
            Math.ceil(width / 8),
            Math.ceil(height / 8)
        );
        compositePass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Composite params updated every frame — nothing persistent to recreate.
        // Bind group rebuilt on texture change in render().
    }

    destroy(): void {
        this._fogParamsBuffer?.destroy();
        this._compositeParamsBuffer?.destroy();
        this._fogParamsBuffer = null;
        this._compositeParamsBuffer = null;
        this._injectPipeline = null;
        this._compositePipeline = null;
        this._injectBG = null;
        this._compositeBG = null;
    }
}

export { VolumetricFogEffect };
```

### Step 2: Verify build compiles

Run: `npx tsc --noEmit`

### Step 3: Commit

```bash
git add src/postprocessing/effects/VolumetricFogEffect.ts
git commit -m "feat(volumetric): add VolumetricFogEffect with injection and composite passes"
```

---

## Task 5: Example Integration

**Files:**
- Modify: `examples/index_postpro.html`

### Step 1: Add imports

After the existing imports (line 31), add:

```javascript
import { ShadowMap } from '../src/shadows/ShadowMap'
import { FroxelGrid } from '../src/froxels/FroxelGrid'
import { VolumetricFogEffect } from '../src/postprocessing/effects/VolumetricFogEffect'
```

### Step 2: Create ShadowMap and FroxelGrid after renderer init

After `renderer.canvas.style.height = '100%';` (~line 125), add:

```javascript
// ── Volumetric fog resources ──────────────────────────────────────────
const lightDir = [0.5, -0.8, 0.3];

const shadowMap = new ShadowMap(renderer.gpuDevice, { resolution: 2048 });
const froxelGrid = new FroxelGrid(renderer.gpuDevice, {
    gridW: 160, gridH: 90, gridD: 64,
    near: 0.1, far: 1000
});
```

### Step 3: Add VolumetricFogEffect to the PostProcessingVolume

In the volume constructor, add VolumetricFogEffect between SSAO and DoF:

```javascript
const volume = new PostProcessingVolume(renderer, [
    new SSAOEffect({
        radius: 0.8,
        bias: 0.02,
        kernelSize: 64,
        strength: 0.5
    }),

    new VolumetricFogEffect({
        froxelGrid,
        shadowMap,
        lightDirection: lightDir,
        lightColor: [1.0, 0.95, 0.85],
        baseDensity: 0.02,
        heightFalloff: 0.1,
        anisotropy: 0.6,
    }),

    new DepthOfFieldEffect({
        focusDistance: 25,
        focusRange: 15,
        maxBlur: 24
    }),
]);
```

### Step 4: Call shadowMap.render() in the animation loop

At the start of `animate()`, before `volume.render()`:

```javascript
function animate(now) {
    const dt = (now - lastTime) / 1000;
    lastTime = now;

    // ... cube scroll code ...

    cameraControls.update();
    shadowMap.render(renderer, scene, camera, lightDir);
    volume.render(scene, camera);
    requestAnimationFrame(animate);
}
```

### Step 5: Visual verification

Run: `npx vite examples/`
Expected: Height-based volumetric fog visible in the scene, denser near ground level, with visible light scattering from the directional light and shadow occlusion from the cube geometry.

### Step 6: Commit

```bash
git add examples/index_postpro.html
git commit -m "feat(example): integrate volumetric fog with shadow map into postpro example"
```

---

## Task 6: Temporal Reprojection (Enhancement)

**Files:**
- Modify: `src/froxels/FroxelGrid.ts`
- Modify: `src/postprocessing/effects/VolumetricFogEffect.ts`

### Step 1: Add double-buffered history textures to FroxelGrid

In FroxelGrid, add two history textures and a temporal reprojection compute pipeline.

New fields:
```typescript
private _historyTex: [GPUTexture, GPUTexture]; // double-buffered
private _historyIdx = 0; // which history to read (write to the other)
private _temporalPipeline: GPUComputePipeline | null = null;
private _temporalBGs: [GPUBindGroup | null, GPUBindGroup | null] = [null, null];
private _temporalParamsBuffer: GPUBuffer;
```

Create history textures in `_createTextures()` with same size and usage as scatterExtinctionTex.

### Step 2: Write temporal reprojection shader

```wgsl
// WGSL_HELPERS injected

struct TemporalParams {
    invViewProj  : mat4x4f,
    prevViewProj : mat4x4f,
    near         : f32,
    far          : f32,
    gridW        : u32,
    gridH        : u32,
    gridD        : u32,
    blendWeight  : f32,
    _pad0        : f32,
    _pad1        : f32,
}

@group(0) @binding(0) var currentTex : texture_3d<f32>;       // scatterExtinctionTex
@group(0) @binding(1) var historyTex : texture_3d<f32>;       // previous blended
@group(0) @binding(2) var outputTex  : texture_storage_3d<rgba16float, write>; // new blended
@group(0) @binding(3) var historySampler : sampler;
@group(0) @binding(4) var<uniform> tp : TemporalParams;

@compute @workgroup_size(4, 4, 4)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    if (gid.x >= tp.gridW || gid.y >= tp.gridH || gid.z >= tp.gridD) { return; }

    let gridSize = vec3f(f32(tp.gridW), f32(tp.gridH), f32(tp.gridD));
    let current = textureLoad(currentTex, gid, 0);

    // Reconstruct world position from current froxel
    let worldPos = froxelToWorld(vec3f(f32(gid.x), f32(gid.y), f32(gid.z)),
                                  tp.invViewProj, tp.near, tp.far, gridSize);

    // Reproject to previous frame's froxel UV
    let prevClip = tp.prevViewProj * vec4f(worldPos, 1.0);
    let prevNDC  = prevClip.xyz / prevClip.w;
    let prevUV   = vec2f(prevNDC.x * 0.5 + 0.5, 1.0 - (prevNDC.y * 0.5 + 0.5));

    // Depth from previous clip
    let prevLinearD = tp.near * tp.far / (tp.far - prevNDC.z * (tp.far - tp.near));
    let prevSlice   = depthToSlice(prevLinearD, tp.near, tp.far, gridSize.z);
    let prevW       = clamp(prevSlice / gridSize.z, 0.0, 1.0);

    // Reject if out of bounds
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0 || prevW < 0.0 || prevW > 1.0) {
        textureStore(outputTex, gid, current);
        return;
    }

    let history = textureSampleLevel(historyTex, historySampler, vec3f(prevUV, prevW), 0.0);
    let blended = mix(current, history, tp.blendWeight);
    textureStore(outputTex, gid, blended);
}
```

### Step 3: Add temporalReproject() method to FroxelGrid

```typescript
temporalReproject(encoder: GPUCommandEncoder, invViewProj: Float32Array, prevViewProj: Float32Array): void {
    // Upload temporal params
    const params = new Float32Array(48); // 192 bytes
    params.set(invViewProj, 0);          // invViewProj
    params.set(prevViewProj, 16);        // prevViewProj
    params[32] = this._near;
    params[33] = this._far;
    new Uint32Array(params.buffer, 136, 1)[0] = this._gridW;
    new Uint32Array(params.buffer, 140, 1)[0] = this._gridH;
    new Uint32Array(params.buffer, 144, 1)[0] = this._gridD;
    params[37] = 0.9; // blend weight
    this._device.queue.writeBuffer(this._temporalParamsBuffer, 0, params.buffer as ArrayBuffer);

    const readIdx = this._historyIdx;
    const writeIdx = 1 - readIdx;

    // Dispatch temporal reprojection
    const pass = encoder.beginComputePass({ label: 'FroxelGrid/Temporal' });
    pass.setPipeline(this._temporalPipeline!);
    pass.setBindGroup(0, this._temporalBGs[readIdx]!);
    pass.dispatchWorkgroups(
        Math.ceil(this._gridW / 4),
        Math.ceil(this._gridH / 4),
        Math.ceil(this._gridD / 4)
    );
    pass.end();

    // Swap history index
    this._historyIdx = writeIdx;
}
```

### Step 4: Wire into VolumetricFogEffect.render()

Between injection and accumulation, add:

```typescript
// Pass 1.5: Temporal reprojection
this._froxelGrid.temporalReproject(commandEncoder, this._invVP as unknown as Float32Array, this._prevVP);

// Save current VP as previous for next frame
mat4.copy(this._prevVPMat, vp);
this._prevVP.set(this._prevVPMat as unknown as Float32Array);
```

The accumulation pass now reads from the latest history buffer instead of scatterExtinctionTex.

### Step 5: Verify visual smoothness

Run: `npx vite examples/`
Expected: Fog flickers less between frames. Moving the camera should show smooth fog without jitter.

### Step 6: Commit

```bash
git add src/froxels/FroxelGrid.ts src/postprocessing/effects/VolumetricFogEffect.ts
git commit -m "feat(froxels): add temporal reprojection with double-buffered history"
```

---

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
| accumTex (same) | 7 MB |
| historyTex ×2 (temporal, same) | 14 MB |
| **Total** | **~44 MB** |
