# Multi-Pass Cinematic Depth of Field

## Problem

The current single-pass DoF uses 80 Vogel disk taps at full resolution with maxBlur=24.
The disc area is ~1810 pixels but only 80 are sampled (~4.4% coverage), producing sparse,
box-blur-like results. Foreground objects show as ghosted duplicates due to crude 4-tap
CoC dilation.

## Solution

5-pass pipeline inspired by Unreal Engine's Cinematic DoF. Blur operates at half resolution
where 64 taps cover ~14% of the disc area plus inherent 2x2 pre-filtering from downsample.
Proper separable CoC dilation and near/far layer separation eliminate foreground ghosting.

## Internal Textures (owned by DepthOfFieldEffect)

| Texture         | Format      | Resolution | Purpose                                      |
|-----------------|-------------|------------|----------------------------------------------|
| `cocTex`        | r16float    | Full       | Signed CoC → dilated CoC (reused across passes) |
| `cocDilTempTex` | r16float    | Full       | Temp for separable dilation horizontal pass   |
| `nearHalfTex`   | rgba16float | Half       | Pre-multiplied near-field color + coverage α  |
| `farHalfTex`    | rgba16float | Half       | Far-field color + CoC in alpha                |
| `nearBlurTex`   | rgba16float | Half       | Blurred near-field                            |
| `farBlurTex`    | rgba16float | Half       | Blurred far-field                             |

~24 MB at 1080p.

## Pass 1: CoC Computation (full-res)

- Reads: depthTex
- Writes: cocTex (r16float)
- `coc = clamp((linearDepth - focusDistance) / focusRange, -1, 1) * maxBlur`
- 1 texture read per pixel, trivial ALU

## Pass 2: Near-Field CoC Dilation (full-res, 2 dispatches)

Expand near-field CoC outward so in-focus/far pixels near foreground edges gather correctly.

- **2a Horizontal**: read cocTex, write cocDilTempTex
  - For each pixel, scan ±maxBlur horizontally
  - Output = max(own |CoC|, max near-neighbor |CoC|) — only propagate negative (near) CoC
- **2b Vertical**: read cocDilTempTex, write cocTex (reuse — original CoC recomputed from depth in pass 5)
  - Same vertical scan on horizontal result
  - Result: square max-filter approximating circular near-field dilation

## Pass 3: Downsample + Near/Far Separation (full → half)

- Reads: input color (full-res), cocTex (now contains dilated CoC), depthTex
- Writes: nearHalfTex, farHalfTex
- Each half-res pixel samples 2x2 full-res block
- Computes original (un-dilated) CoC from depth for each texel in the block
- **Near layer** (original CoC < 0):
  - coverage = saturate(|CoC| / (maxBlur * 0.5))
  - rgb = color * coverage (pre-multiplied)
  - a = coverage
  - Takes max coverage across 2x2 block to preserve thin features
- **Far layer** (original CoC > 0):
  - rgb = color
  - a = |CoC| / maxBlur
- Stores dilated gather radius (from cocTex) into farHalfTex for pass 4 to read

## Pass 4: Vogel Disk Blur (half-res)

- Reads: nearHalfTex, farHalfTex, cocTex (at 2x coords for gather radius)
- Writes: nearBlurTex, farBlurTex
- 64-tap Vogel disk per pixel
- Half-res maxBlur = maxBlur / 2; disc area ≈ π*(maxBlur/2)² pixels
- **Near blur**: gather pre-multiplied near samples; weight = smoothstep coverage of sample's CoC disc
- **Far blur**: gather far samples; scatter-as-gather with hard-disc coverage weighting
- Single dispatch writes both outputs

## Pass 5: Composite (full-res)

- Reads: sharp input color, depthTex (recompute original CoC), nearBlurTex, farBlurTex
- Writes: output texture
- Bilinear upscale of half-res textures via manual 4-tap interpolation
- Compositing:
  - Recompute original signed CoC from depth
  - |CoC| < 1px → sharp
  - CoC > 0 → lerp(sharp, farBlur, saturate(CoC / 2.0))
  - Near-field → nearBlur alpha-composites on top using blurred near alpha

## Performance Comparison

| Metric              | Current (1-pass) | New (5-pass)                          |
|---------------------|-----------------|---------------------------------------|
| Dispatches          | 1               | 6 (dilation = 2 dispatches)           |
| Blur resolution     | Full            | Half (4x fewer pixels)                |
| Blur taps           | 80              | 64 (at half-res ≈ 256 full-res equiv) |
| Disc coverage       | ~4.4%           | ~14% + 2x2 pre-filter                 |
| Depth reads in blur | 80/pixel        | 0 (CoC pre-computed)                  |
| VRAM overhead       | 0               | ~24 MB at 1080p                       |
