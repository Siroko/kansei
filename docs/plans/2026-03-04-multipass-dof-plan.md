# Multi-Pass Cinematic DoF Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the single-pass DoF with a 5-pass cinematic pipeline that blurs at half-resolution with proper near/far separation, CoC dilation, and disc bokeh.

**Architecture:** All 5 passes are self-contained within `DepthOfFieldEffect.ts`. The effect creates 6 internal textures and dispatches 6 compute passes (dilation = 2 dispatches) through the single `commandEncoder` it receives from `PostProcessingVolume`. No infrastructure changes needed.

**Tech Stack:** WebGPU compute shaders (WGSL), TypeScript, Vite dev server

**Design doc:** `docs/plans/2026-03-04-multipass-dof-design.md`

---

### Task 1: Scaffold internal textures and multi-pipeline infrastructure

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** The current effect has a single pipeline, single bind group, and no internal textures. We need to restructure it to support 5 pipelines (one per shader/pass) and 6 internal textures.

**Step 1: Replace single-pipeline fields with multi-pipeline structure**

Replace the current private fields:

```typescript
private _device: GPUDevice | null = null;
private _pipeline: GPUComputePipeline | null = null;
private _paramsBuffer: GPUBuffer | null = null;
private _bindGroup: GPUBindGroup | null = null;
private _currentInput: GPUTexture | null = null;
private _currentDepth: GPUTexture | null = null;
private _currentOutput: GPUTexture | null = null;
```

With:

```typescript
private _device: GPUDevice | null = null;
private _paramsBuffer: GPUBuffer | null = null;

// 5 pipelines (pass 2 uses same pipeline twice with different bind groups)
private _cocPipeline: GPUComputePipeline | null = null;
private _dilateHPipeline: GPUComputePipeline | null = null;
private _dilateVPipeline: GPUComputePipeline | null = null;
private _downsamplePipeline: GPUComputePipeline | null = null;
private _blurPipeline: GPUComputePipeline | null = null;
private _compositePipeline: GPUComputePipeline | null = null;

// Internal textures
private _cocTex: GPUTexture | null = null;           // r16float, full-res
private _cocDilTempTex: GPUTexture | null = null;     // r16float, full-res
private _nearHalfTex: GPUTexture | null = null;       // rgba16float, half-res
private _farHalfTex: GPUTexture | null = null;        // rgba16float, half-res
private _nearBlurTex: GPUTexture | null = null;       // rgba16float, half-res
private _farBlurTex: GPUTexture | null = null;        // rgba16float, half-res

// Bind groups (rebuilt on resize or input texture change)
private _cocBindGroup: GPUBindGroup | null = null;
private _dilateHBindGroup: GPUBindGroup | null = null;
private _dilateVBindGroup: GPUBindGroup | null = null;
private _downsampleBindGroup: GPUBindGroup | null = null;
private _blurBindGroup: GPUBindGroup | null = null;
private _compositeBindGroup: GPUBindGroup | null = null;

// Track current external textures for bind group invalidation
private _currentInput: GPUTexture | null = null;
private _currentDepth: GPUTexture | null = null;
private _currentOutput: GPUTexture | null = null;
private _halfWidth: number = 0;
private _halfHeight: number = 0;
```

**Step 2: Add internal texture creation method**

```typescript
private _createInternalTextures(width: number, height: number): void {
    this._destroyInternalTextures();
    const device = this._device!;
    const hw = Math.ceil(width / 2);
    const hh = Math.ceil(height / 2);
    this._halfWidth = hw;
    this._halfHeight = hh;

    const fullR16 = (label: string) => device.createTexture({
        label,
        size: [width, height],
        format: 'r16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });
    const halfRGBA16 = (label: string) => device.createTexture({
        label,
        size: [hw, hh],
        format: 'rgba16float',
        usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
    });

    this._cocTex        = fullR16('DoF/CoC');
    this._cocDilTempTex = fullR16('DoF/CoCDilTemp');
    this._nearHalfTex   = halfRGBA16('DoF/NearHalf');
    this._farHalfTex    = halfRGBA16('DoF/FarHalf');
    this._nearBlurTex   = halfRGBA16('DoF/NearBlur');
    this._farBlurTex    = halfRGBA16('DoF/FarBlur');
}

private _destroyInternalTextures(): void {
    this._cocTex?.destroy();
    this._cocDilTempTex?.destroy();
    this._nearHalfTex?.destroy();
    this._farHalfTex?.destroy();
    this._nearBlurTex?.destroy();
    this._farBlurTex?.destroy();
    this._cocTex = null;
    this._cocDilTempTex = null;
    this._nearHalfTex = null;
    this._farHalfTex = null;
    this._nearBlurTex = null;
    this._farBlurTex = null;
}
```

**Step 3: Update `resize()` to recreate internal textures**

```typescript
resize(w: number, h: number, _gbuffer: GBuffer): void {
    this._createInternalTextures(w, h);
    // Invalidate all bind groups so they're rebuilt next frame
    this._cocBindGroup = null;
    this._dilateHBindGroup = null;
    this._dilateVBindGroup = null;
    this._downsampleBindGroup = null;
    this._blurBindGroup = null;
    this._compositeBindGroup = null;
}
```

**Step 4: Update `destroy()` to clean up everything**

```typescript
destroy(): void {
    this._paramsBuffer?.destroy();
    this._destroyInternalTextures();
    this._paramsBuffer = null;
    this._cocPipeline = null;
    this._dilateHPipeline = null;
    this._dilateVPipeline = null;
    this._downsamplePipeline = null;
    this._blurPipeline = null;
    this._compositePipeline = null;
    this._cocBindGroup = null;
    this._dilateHBindGroup = null;
    this._dilateVBindGroup = null;
    this._downsampleBindGroup = null;
    this._blurBindGroup = null;
    this._compositeBindGroup = null;
}
```

**Step 5: Stub out the shader strings**

Replace the single `_SHADER` static with 5 shader stubs (empty entry points that just pass through). We will fill these in during tasks 2-6. For now, use the simplest possible shader for each pass so the pipeline compiles:

```typescript
private static readonly _COC_SHADER = /* wgsl */`/* pass 1 - filled in task 2 */`;
private static readonly _DILATE_H_SHADER = /* wgsl */`/* pass 2a - filled in task 3 */`;
private static readonly _DILATE_V_SHADER = /* wgsl */`/* pass 2b - filled in task 3 */`;
private static readonly _DOWNSAMPLE_SHADER = /* wgsl */`/* pass 3 - filled in task 4 */`;
private static readonly _BLUR_SHADER = /* wgsl */`/* pass 4 - filled in task 5 */`;
private static readonly _COMPOSITE_SHADER = /* wgsl */`/* pass 5 - filled in task 6 */`;
```

**Step 6: Stub `initialize()` to create all 6 pipelines**

Create the params buffer (same as current — 8 x f32 = 32 bytes) and all 6 compute pipelines. Each pipeline will have its own bind group layout matching the shader's bindings. For now, only create the CoC pipeline; the others will be added as we implement each shader.

**Step 7: Verify it compiles**

Run: `pnpm dev`
Expected: Vite builds without TypeScript errors. The page may crash at runtime since shaders are stubs — that's OK, we'll fix it pass by pass.

**Step 8: Commit**

```
feat(dof): scaffold multi-pass infrastructure with 6 internal textures
```

---

### Task 2: Implement Pass 1 — CoC computation shader

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** This is the simplest pass. Reads the depth texture, computes signed CoC in pixels, writes to the r16float CoC texture. The `DoFParams` uniform struct is shared by all passes.

**Step 1: Write the CoC shader**

```wgsl
struct DoFParams {
    focusDistance : f32,
    focusRange    : f32,
    maxBlur       : f32,
    near          : f32,
    far           : f32,
    screenWidth   : f32,
    screenHeight  : f32,
    _pad          : f32,
}

@group(0) @binding(0) var depthTex  : texture_depth_2d;
@group(0) @binding(1) var cocOut    : texture_storage_2d<r16float, write>;
@group(0) @binding(2) var<uniform> params : DoFParams;

fn linearDepth(d: f32) -> f32 {
    return (params.near * params.far) / (params.far - d * (params.far - params.near));
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }

    let depth = textureLoad(depthTex, coord, 0);
    let ld    = linearDepth(depth);
    let coc   = clamp((ld - params.focusDistance) / params.focusRange, -1.0, 1.0)
                * params.maxBlur;

    textureStore(cocOut, coord, vec4f(coc, 0.0, 0.0, 0.0));
}
```

**Step 2: Create the CoC pipeline and bind group layout in `initialize()`**

Bind group layout entries:
- binding 0: texture (depth, sampleType 'depth')
- binding 1: storageTexture (r16float, write-only)
- binding 2: buffer (uniform)

**Step 3: Write a `_buildCoCBindGroup()` method**

Reads `depth` (external), writes `cocTex` (internal), uses `paramsBuffer`.

**Step 4: Add CoC dispatch to `render()`**

```typescript
// Pass 1: CoC
const wgFull = (t: number) => Math.ceil(t / 8);
const cocPass = commandEncoder.beginComputePass({ label: 'DoF/CoC' });
cocPass.setPipeline(this._cocPipeline!);
cocPass.setBindGroup(0, this._cocBindGroup!);
cocPass.dispatchWorkgroups(wgFull(width), wgFull(height));
cocPass.end();
```

**Step 5: Temporarily output cocTex as a debug visualization**

For now, add a simple debug composite pass that reads `cocTex` and writes it as a red-blue gradient to the output texture, so we can visually verify the CoC values are correct:
- Red = far field (positive CoC)
- Blue = near field (negative CoC)
- Black = in focus

**Step 6: Verify visually**

Run: `pnpm dev`, open `http://localhost:5173/examples/index_postpro.html`
Expected: Scene renders with a red/blue CoC debug overlay. Objects near the camera should be blue, objects at focusDistance should be black, objects far away should be red.

**Step 7: Commit**

```
feat(dof): implement pass 1 — CoC computation with debug visualization
```

---

### Task 3: Implement Pass 2 — Separable CoC dilation

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** Two dispatches. The horizontal pass scans ±maxBlur pixels reading from `cocTex`, writing max near-field |CoC| to `cocDilTempTex`. The vertical pass reads `cocDilTempTex` and writes the final dilated result back to `cocTex` (the original signed CoC is recomputed from depth in pass 5, so we can safely reuse `cocTex`).

**Step 1: Write the horizontal dilation shader**

```wgsl
@group(0) @binding(0) var cocIn  : texture_2d<f32>;
@group(0) @binding(1) var cocOut : texture_storage_2d<r16float, write>;
@group(0) @binding(2) var<uniform> params : DoFParams;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }

    let center = textureLoad(cocIn, coord, 0).r;
    var maxNear = select(abs(center), 0.0, center >= 0.0);
    let radius = i32(params.maxBlur);

    for (var dx = -radius; dx <= radius; dx++) {
        let sx = clamp(i32(coord.x) + dx, 0, i32(w) - 1);
        let sc = vec2u(u32(sx), coord.y);
        let val = textureLoad(cocIn, sc, 0).r;
        // Only propagate near-field (negative CoC)
        if (val < 0.0) {
            // Weight by how close the sample is — farther samples contribute less
            // unless their CoC is large enough to reach us
            let sampleR = abs(val);
            let dist = abs(f32(dx));
            if (sampleR >= dist) {
                maxNear = max(maxNear, sampleR);
            }
        }
    }

    // Output: the dilated gather radius. Keep sign info:
    // store as positive value (it's the gather radius, always >= 0)
    // but also preserve the original center CoC for blending decisions
    let gatherR = max(abs(center), maxNear);
    textureStore(cocOut, coord, vec4f(gatherR, 0.0, 0.0, 0.0));
}
```

**Step 2: Write the vertical dilation shader**

Same as horizontal but scans in Y direction. Reads `cocDilTempTex` (horizontal result), writes to `cocTex`.

```wgsl
// Same structure, but:
// let sy = clamp(i32(coord.y) + dy, 0, i32(h) - 1);
// let sc = vec2u(coord.x, u32(sy));
```

**Step 3: Create pipelines and bind group layouts**

Horizontal:
- binding 0: texture_2d<f32> (cocTex — read as float since r16float)
- binding 1: texture_storage_2d<r16float, write> (cocDilTempTex)
- binding 2: uniform buffer

Vertical:
- binding 0: texture_2d<f32> (cocDilTempTex)
- binding 1: texture_storage_2d<r16float, write> (cocTex — reuse!)
- binding 2: uniform buffer

**Step 4: Add dispatches to `render()`**

```typescript
// Pass 2a: Horizontal dilation
const dilHPass = commandEncoder.beginComputePass({ label: 'DoF/DilateH' });
dilHPass.setPipeline(this._dilateHPipeline!);
dilHPass.setBindGroup(0, this._dilateHBindGroup!);
dilHPass.dispatchWorkgroups(wgFull(width), wgFull(height));
dilHPass.end();

// Pass 2b: Vertical dilation
const dilVPass = commandEncoder.beginComputePass({ label: 'DoF/DilateV' });
dilVPass.setPipeline(this._dilateVPipeline!);
dilVPass.setBindGroup(0, this._dilateVBindGroup!);
dilVPass.dispatchWorkgroups(wgFull(width), wgFull(height));
dilVPass.end();
```

**Step 5: Update debug visualization to show dilated CoC**

Show `cocTex` (now containing dilated values) as a heat map. Near foreground objects should have a visible "halo" of expanded gather radius around their silhouettes.

**Step 6: Verify visually**

Run: `pnpm dev`
Expected: Dilated CoC shows a bright halo around near-field object edges. The halo width should roughly match `maxBlur` pixels.

**Step 7: Commit**

```
feat(dof): implement pass 2 — separable near-field CoC dilation
```

---

### Task 4: Implement Pass 3 — Downsample + near/far separation

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** This pass reads full-res color + depth + dilated CoC, and outputs two half-res layers: near-field (pre-multiplied) and far-field.

**Step 1: Write the downsample shader**

```wgsl
@group(0) @binding(0) var colorTex : texture_2d<f32>;
@group(0) @binding(1) var depthTex : texture_depth_2d;
@group(0) @binding(2) var cocTex   : texture_2d<f32>;  // dilated CoC
@group(0) @binding(3) var nearOut  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var farOut   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params : DoFParams;

fn linearDepth(d: f32) -> f32 {
    return (params.near * params.far) / (params.far - d * (params.far - params.near));
}

fn computeCoC(d: f32) -> f32 {
    let ld = linearDepth(d);
    return clamp((ld - params.focusDistance) / params.focusRange, -1.0, 1.0) * params.maxBlur;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let halfCoord = gid.xy;
    let hw = u32(ceil(params.screenWidth / 2.0));
    let hh = u32(ceil(params.screenHeight / 2.0));
    if (halfCoord.x >= hw || halfCoord.y >= hh) { return; }

    let fullW = u32(params.screenWidth);
    let fullH = u32(params.screenHeight);

    // Sample 2x2 block from full-res
    let base = halfCoord * 2u;
    var nearColor = vec4f(0.0);
    var farColor  = vec4f(0.0);
    var nearW     = 0.0;
    var farW      = 0.0;
    var maxDilatedR = 0.0;

    for (var dy = 0u; dy < 2u; dy++) {
        for (var dx = 0u; dx < 2u; dx++) {
            let fc = min(base + vec2u(dx, dy), vec2u(fullW - 1u, fullH - 1u));
            let color = textureLoad(colorTex, fc, 0);
            let depth = textureLoad(depthTex, fc, 0);
            let coc   = computeCoC(depth);
            let dilR  = textureLoad(cocTex, fc, 0).r;  // dilated gather radius

            maxDilatedR = max(maxDilatedR, dilR);

            if (coc < 0.0) {
                // Near-field: pre-multiply by coverage
                let coverage = clamp(abs(coc) / max(params.maxBlur * 0.5, 1.0), 0.0, 1.0);
                nearColor += vec4f(color.rgb * coverage, coverage);
                nearW += 1.0;
            } else {
                // Far-field: store color + normalized CoC
                let normCoc = abs(coc) / max(params.maxBlur, 1.0);
                farColor += vec4f(color.rgb, normCoc);
                farW += 1.0;
            }
        }
    }

    // Average the contributions
    let near = select(vec4f(0.0), nearColor / max(nearW, 1.0), nearW > 0.0);
    var far  = select(vec4f(0.0), farColor  / max(farW, 1.0),  farW  > 0.0);

    // Pack dilated gather radius (in half-res pixels) into far.a
    // when there's no far contribution — store dilated radius anyway
    // We use a separate channel strategy: far.a = max(farCoC, dilatedR/maxBlur)
    far = vec4f(far.rgb, max(far.a, maxDilatedR / max(params.maxBlur, 1.0)));

    textureStore(nearOut, halfCoord, near);
    textureStore(farOut,  halfCoord, far);
}
```

**Step 2: Create pipeline and bind group layout**

6 bindings:
- 0: texture_2d<f32> (color input)
- 1: texture_depth_2d (depth)
- 2: texture_2d<f32> (cocTex — dilated)
- 3: texture_storage_2d<rgba16float, write> (nearHalfTex)
- 4: texture_storage_2d<rgba16float, write> (farHalfTex)
- 5: uniform buffer

**Step 3: Add dispatch to `render()` — dispatched at half-res workgroup count**

```typescript
const wgHalf = (t: number) => Math.ceil(Math.ceil(t / 2) / 8);
const dsPass = commandEncoder.beginComputePass({ label: 'DoF/Downsample' });
dsPass.setPipeline(this._downsamplePipeline!);
dsPass.setBindGroup(0, this._downsampleBindGroup!);
dsPass.dispatchWorkgroups(wgHalf(width), wgHalf(height));
dsPass.end();
```

**Step 4: Update debug visualization**

Show `nearHalfTex` and `farHalfTex` side by side (or toggle between them). Near layer should show pre-multiplied foreground objects fading out at edges. Far layer should show background objects.

**Step 5: Verify visually**

Run: `pnpm dev`
Expected: Near layer shows only foreground objects with soft edges (pre-multiplied alpha). Far layer shows only background objects. In-focus regions should be dark/empty in both layers.

**Step 6: Commit**

```
feat(dof): implement pass 3 — downsample with near/far layer separation
```

---

### Task 5: Implement Pass 4 — Vogel disk blur at half-res

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** The main bokeh blur. Runs at half-res. 64-tap Vogel disk with scatter-as-gather weighting. Blurs near and far layers separately in a single dispatch writing to two output textures.

**Step 1: Write the blur shader**

```wgsl
@group(0) @binding(0) var nearIn   : texture_2d<f32>;
@group(0) @binding(1) var farIn    : texture_2d<f32>;
@group(0) @binding(2) var cocTex   : texture_2d<f32>;  // dilated CoC, full-res
@group(0) @binding(3) var nearOut  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var farOut   : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params : DoFParams;

const GOLDEN_ANGLE : f32 = 2.39996323;
const NUM_SAMPLES : u32 = 64u;

fn vogelDisk(i: u32, n: u32) -> vec2f {
    let r     = sqrt((f32(i) + 0.5) / f32(n));
    let theta = f32(i) * GOLDEN_ANGLE;
    return vec2f(cos(theta), sin(theta)) * r;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let hw = u32(ceil(params.screenWidth / 2.0));
    let hh = u32(ceil(params.screenHeight / 2.0));
    if (coord.x >= hw || coord.y >= hh) { return; }

    let lim = vec2i(i32(hw) - 1, i32(hh) - 1);

    // Read dilated gather radius from full-res cocTex at corresponding position
    let fullCoord = min(coord * 2u, vec2u(u32(params.screenWidth) - 1u, u32(params.screenHeight) - 1u));
    let dilatedR  = textureLoad(cocTex, fullCoord, 0).r;
    let halfR     = dilatedR * 0.5;  // convert to half-res pixels

    let centerNear = textureLoad(nearIn, coord, 0);
    let centerFar  = textureLoad(farIn,  coord, 0);

    // If gather radius is tiny, pass through
    if (halfR < 0.5) {
        textureStore(nearOut, coord, centerNear);
        textureStore(farOut,  coord, centerFar);
        return;
    }

    // ── Vogel disk gather ──────────────────────────────────────────────
    var nearAcc  = vec4f(0.0);
    var farAcc   = vec3f(0.0);
    var farW     = 0.0;

    for (var i = 0u; i < NUM_SAMPLES; i++) {
        let uv  = vogelDisk(i, NUM_SAMPLES);
        let off = uv * halfR;
        let sc  = clamp(
            vec2i(coord) + vec2i(i32(off.x), i32(off.y)),
            vec2i(0), lim
        );
        let dist = length(off);

        // ── Near: gather pre-multiplied samples ──
        let sNear = textureLoad(nearIn, sc, 0);
        // Near samples are already weighted by coverage (pre-multiplied).
        // Just accumulate uniformly — the pre-multiplication handles edge fadeout.
        nearAcc += sNear;

        // ── Far: scatter-as-gather with disc coverage ──
        let sFar   = textureLoad(farIn, sc, 0);
        let sFarR  = sFar.a * params.maxBlur * 0.5;  // sample's half-res CoC
        // Coverage: does the sample's bokeh disc reach this pixel?
        let coverage = smoothstep(dist - 1.0, dist + 1.0, sFarR);
        farAcc += sFar.rgb * coverage;
        farW   += coverage;
    }

    // Near: divide by sample count (pre-multiplied average)
    let nearResult = nearAcc / f32(NUM_SAMPLES);

    // Far: weighted average
    let farResult = select(centerFar.rgb, farAcc / farW, farW > 0.001);

    textureStore(nearOut, coord, nearResult);
    textureStore(farOut,  coord, vec4f(farResult, centerFar.a));
}
```

**Step 2: Create pipeline and bind group layout**

6 bindings:
- 0: texture_2d<f32> (nearHalfTex)
- 1: texture_2d<f32> (farHalfTex)
- 2: texture_2d<f32> (cocTex — full-res dilated)
- 3: texture_storage_2d<rgba16float, write> (nearBlurTex)
- 4: texture_storage_2d<rgba16float, write> (farBlurTex)
- 5: uniform buffer

**Step 3: Add dispatch to `render()`**

Dispatched at half-res workgroup count, same as pass 3.

**Step 4: Update debug visualization**

Show `nearBlurTex` and `farBlurTex`. Near blur should show smooth, filled-in foreground bokeh. Far blur should show smooth background bokeh with disc-shaped highlights.

**Step 5: Verify visually**

Run: `pnpm dev`
Expected: Both blur layers show smooth, filled-in bokeh without visible sampling artifacts. Far-field bright highlights should have disc-shaped bokeh. Near-field objects should have smooth, wide blur that fades at edges.

**Step 6: Commit**

```
feat(dof): implement pass 4 — 64-tap Vogel disk blur at half-res
```

---

### Task 6: Implement Pass 5 — Full-res composite

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Context:** Final pass. Recomputes original CoC from depth, bilinearly upscales the half-res blur layers, and composites sharp + far blur + near blur.

**Step 1: Write the composite shader**

```wgsl
@group(0) @binding(0) var colorTex   : texture_2d<f32>;       // sharp input
@group(0) @binding(1) var depthTex   : texture_depth_2d;       // for recomputing CoC
@group(0) @binding(2) var nearBlurTex: texture_2d<f32>;        // half-res blurred near
@group(0) @binding(3) var farBlurTex : texture_2d<f32>;        // half-res blurred far
@group(0) @binding(4) var outputTex  : texture_storage_2d<rgba16float, write>;
@group(0) @binding(5) var<uniform> params : DoFParams;

fn linearDepth(d: f32) -> f32 {
    return (params.near * params.far) / (params.far - d * (params.far - params.near));
}

fn computeCoC(d: f32) -> f32 {
    let ld = linearDepth(d);
    return clamp((ld - params.focusDistance) / params.focusRange, -1.0, 1.0) * params.maxBlur;
}

// Manual bilinear interpolation for half-res texture
fn sampleBilinear(tex: texture_2d<f32>, fullCoord: vec2u, halfW: u32, halfH: u32) -> vec4f {
    // Map full-res coord to half-res continuous coordinate
    let hx = (f32(fullCoord.x) + 0.5) * 0.5 - 0.5;
    let hy = (f32(fullCoord.y) + 0.5) * 0.5 - 0.5;

    let x0 = u32(clamp(i32(floor(hx)), 0, i32(halfW) - 1));
    let y0 = u32(clamp(i32(floor(hy)), 0, i32(halfH) - 1));
    let x1 = min(x0 + 1u, halfW - 1u);
    let y1 = min(y0 + 1u, halfH - 1u);

    let fx = fract(hx);
    let fy = fract(hy);

    let s00 = textureLoad(tex, vec2u(x0, y0), 0);
    let s10 = textureLoad(tex, vec2u(x1, y0), 0);
    let s01 = textureLoad(tex, vec2u(x0, y1), 0);
    let s11 = textureLoad(tex, vec2u(x1, y1), 0);

    return mix(mix(s00, s10, fx), mix(s01, s11, fx), fy);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid : vec3u) {
    let coord = gid.xy;
    let w = u32(params.screenWidth);
    let h = u32(params.screenHeight);
    if (coord.x >= w || coord.y >= h) { return; }

    let hw = u32(ceil(params.screenWidth / 2.0));
    let hh = u32(ceil(params.screenHeight / 2.0));

    let sharp = textureLoad(colorTex, coord, 0);
    let depth = textureLoad(depthTex, coord, 0);
    let coc   = computeCoC(depth);
    let absCoc = abs(coc);

    // Bilinear upscale from half-res
    let nearBlur = sampleBilinear(nearBlurTex, coord, hw, hh);
    let farBlur  = sampleBilinear(farBlurTex,  coord, hw, hh);

    // ── Composite ──────────────────────────────────────────────────────

    // Start with sharp image
    var result = sharp;

    // Far field: blend from sharp to far blur based on CoC magnitude
    if (coc > 0.0) {
        let t = clamp(absCoc / 2.0, 0.0, 1.0);
        result = mix(sharp, vec4f(farBlur.rgb, sharp.a), t);
    }

    // Near field: alpha-composite blurred near on top
    // nearBlur is pre-multiplied: .rgb = color * alpha, .a = alpha
    let nearAlpha = clamp(nearBlur.a * 3.0, 0.0, 1.0);  // amplify for visibility
    result = mix(result, vec4f(nearBlur.rgb / max(nearBlur.a, 0.001), sharp.a), nearAlpha);

    textureStore(outputTex, coord, result);
}
```

**Step 2: Create pipeline and bind group layout**

6 bindings:
- 0: texture_2d<f32> (input color)
- 1: texture_depth_2d (depth)
- 2: texture_2d<f32> (nearBlurTex)
- 3: texture_2d<f32> (farBlurTex)
- 4: texture_storage_2d<rgba16float, write> (output)
- 5: uniform buffer

**Step 3: Add dispatch to `render()` and remove debug visualization**

All 6 dispatches in sequence:
1. CoC (full-res)
2. Dilate H (full-res)
3. Dilate V (full-res)
4. Downsample (half-res)
5. Blur (half-res)
6. Composite (full-res)

**Step 4: Verify visually**

Run: `pnpm dev`
Expected: Full cinematic DoF! Smooth, filled-in bokeh in both foreground and background. Foreground objects blur smoothly without ghosting artifacts. Background highlights show disc-shaped bokeh. In-focus region remains sharp with smooth transitions.

**Step 5: Commit**

```
feat(dof): implement pass 5 — full-res composite with bilinear upscale
```

---

### Task 7: Polish and tune parameters

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`
- Modify: `examples/index_postpro.html`

**Step 1: Tune the near-field alpha amplification**

The `nearAlpha * 3.0` multiplier in the composite may need adjustment. Test with:
- Objects very close to camera (strong near blur)
- Objects just barely in front of focus plane (subtle near blur)
- Adjust multiplier until near-field transitions look natural

**Step 2: Tune the far-field transition**

The `clamp(absCoc / 2.0, ...)` far-field blend may transition too quickly or slowly. Test with different focusDistance/focusRange values.

**Step 3: Verify edge cases**

- All objects in focus (focusRange very large): should look identical to no DoF
- Focus at near plane: everything should be far-blurred
- Focus at far plane: everything should be near-blurred
- maxBlur = 0: should be a no-op (pass-through)

**Step 4: Clean up any remaining debug code**

Remove debug visualization shaders/code if still present.

**Step 5: Commit**

```
fix(dof): tune composite blending and clean up debug code
```

---

### Task 8: Remove old single-pass shader code

**Files:**
- Modify: `src/postprocessing/effects/DepthOfFieldEffect.ts`

**Step 1: Remove the old `_SHADER` static string**

It should have been replaced by the 5 new shader strings. Verify no references remain.

**Step 2: Remove old `_buildBindGroup()` method**

Replaced by per-pass bind group builders.

**Step 3: Final verify**

Run: `pnpm dev`
Expected: Everything still works. No TypeScript errors. Clean code.

**Step 4: Commit**

```
refactor(dof): remove old single-pass shader and bind group code
```
