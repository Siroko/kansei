import { mat4 } from 'gl-matrix';
import { gpuPass } from '../profiling/Profiler';
import depthPyramidWgsl from '../../rust/kansei-core/src/shaders/depth_pyramid.wgsl?raw';

/**
 * The pyramid's build shader, shared with the Rust engine (`shaders/depth_pyramid.wgsl`), before
 * its `MODE_VALUE` and `FORMAT` are substituted (`DepthPyramid` does).
 */
export const DEPTH_PYRAMID_WGSL: string = depthPyramidWgsl;

/**
 * What each texel of a `DepthPyramid` keeps of the depths it covers: the largest (`r32float`),
 * the farthest with the renderer's `[0, 1]` depth, which is what an occlusion test compares
 * against; the smallest (`r32float`), the farthest with reverse Z; or both (`rg32float`, r the
 * smallest and g the largest). Rust: `culling::DepthReduction`.
 */
export type DepthReduction = 'max' | 'min' | 'minMax';

/** The storage format of a pyramid keeping `reduction`. */
export function depthReductionFormat(reduction: DepthReduction): GPUTextureFormat {
    return reduction === 'minMax' ? 'rg32float' : 'r32float';
}

function shader(reduction: DepthReduction): string {
    const mode = { max: 0, min: 1, minMax: 2 }[reduction];
    return DEPTH_PYRAMID_WGSL.replace(/MODE_VALUE/g, `${mode}u`).replace(/FORMAT/g, depthReductionFormat(reduction));
}

/**
 * The size of each mip for a depth buffer of `width` x `height`: from half the buffer, rounded up
 * to a power of two (so that every mip halves exactly, as the texture's mip chain does), down to
 * 1 x 1.
 */
export function mipSizes(width: number, height: number): [number, number][] {
    const base = (n: number) => {
        const half = Math.max(Math.ceil(n / 2), 1);
        let p = 1;
        while (p < half) p *= 2;
        return p;
    };
    const sizes: [number, number][] = [[base(width), base(height)]];
    for (;;) {
        const [w, h] = sizes[sizes.length - 1];
        if (w === 1 && h === 1) return sizes;
        sizes.push([Math.max(w >> 1, 1), Math.max(h >> 1, 1)]);
    }
}

interface Mip {
    size: [number, number];
    storage: GPUTextureView;
    /** mips 1..: the bind group reading the mip before */
    bindGroup: GPUBindGroup | null;
}

/**
 * A hierarchical depth ("Hi-Z") pyramid of a depth buffer, built by compute. A port of the Rust
 * engine's `culling::DepthPyramid`, sharing its shader.
 *
 * Mip 0 is half the depth buffer's size, rounded up to a power of two in each axis, and each mip
 * halves the one before, down to 1 x 1. Texel `(x, y)` of mip `L` reduces exactly the depth
 * pixels `[x, x + 1) * 2^(L + 1)` in each axis (clipped to the buffer; texels wholly past it hold
 * edge values and are never needed), so a query maps a pixel rectangle to texels by shifting its
 * corners right by `L + 1`, for any buffer size: a rectangle whose extent in pixels is under
 * `2^(L + 1)` touches at most 2 x 2 texels of mip `L`.
 *
 * ```ts
 * const pyramid = new DepthPyramid(device, width, height, 'max');
 * pyramid.build(encoder, gbuffer.depthTexture.createView()); // after the depth is drawn
 * // bind pyramid.view as texture_2d<f32> and textureLoad(pyramid, texel, level)
 * ```
 */
export class DepthPyramid {
    readonly reduction: DepthReduction;
    private _sourceSize: [number, number];
    private _texture!: GPUTexture;
    private _view!: GPUTextureView;
    private _mips: Mip[] = [];
    private readonly fromDepth: GPUComputePipeline;
    private readonly fromMip: GPUComputePipeline;
    private readonly depthLayout: GPUBindGroupLayout;
    private readonly mipLayout: GPUBindGroupLayout;
    // `buildLinear`'s first pass, its layout and uniform
    private readonly linear: { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout; uniform: GPUBuffer };
    private readonly linearData = new Float32Array(20);

    /** A pyramid for a depth buffer of `width` x `height` (a `texture_depth_2d`, single-sampled). */
    constructor(private readonly device: GPUDevice, width: number, height: number, reduction: DepthReduction) {
        this.reduction = reduction;
        const format = depthReductionFormat(reduction);
        const compute = GPUShaderStage.COMPUTE;
        const depth: GPUBindGroupLayoutEntry = { binding: 0, visibility: compute, texture: { sampleType: 'depth' } };
        const dst: GPUBindGroupLayoutEntry = { binding: 2, visibility: compute, storageTexture: { access: 'write-only', format } };
        this.depthLayout = device.createBindGroupLayout({ label: 'DepthPyramid/FromDepthBGL', entries: [depth, dst] });
        const linearLayout = device.createBindGroupLayout({
            label: 'DepthPyramid/FromDepthLinearBGL',
            entries: [depth, dst, { binding: 3, visibility: compute, buffer: { type: 'uniform' } }],
        });
        this.mipLayout = device.createBindGroupLayout({
            label: 'DepthPyramid/FromMipBGL',
            entries: [{ binding: 1, visibility: compute, texture: { sampleType: 'unfilterable-float' } }, dst],
        });
        const module = device.createShaderModule({ label: 'DepthPyramid', code: shader(reduction) });
        const pipeline = (layout: GPUBindGroupLayout, entryPoint: string) => device.createComputePipeline({
            label: `DepthPyramid/${entryPoint}`,
            layout: device.createPipelineLayout({ label: 'DepthPyramid', bindGroupLayouts: [layout] }),
            compute: { module, entryPoint },
        });
        this.fromDepth = pipeline(this.depthLayout, 'from_depth');
        this.fromMip = pipeline(this.mipLayout, 'from_mip');
        this.linear = {
            pipeline: pipeline(linearLayout, 'from_depth_linear'),
            layout: linearLayout,
            uniform: device.createBuffer({ label: 'DepthPyramid/Linearize', size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        };
        this._sourceSize = [width, height];
        this.createMips(width, height);
    }

    private createMips(width: number, height: number): void {
        const sizes = mipSizes(width, height);
        this._texture?.destroy();
        const texture = this.device.createTexture({
            label: 'DepthPyramid',
            size: [sizes[0][0], sizes[0][1]],
            mipLevelCount: sizes.length,
            format: depthReductionFormat(this.reduction),
            // (COPY_SRC: readable for debugging)
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        const mipView = (level: number) => texture.createView({ label: 'DepthPyramid/Mip', baseMipLevel: level, mipLevelCount: 1 });
        const storage = sizes.map((_, level) => mipView(level));
        this._texture = texture;
        this._view = texture.createView({ label: 'DepthPyramid' });
        this._mips = sizes.map((size, level) => ({
            size,
            storage: storage[level],
            bindGroup: level === 0 ? null : this.device.createBindGroup({
                label: 'DepthPyramid/FromMip',
                layout: this.mipLayout,
                entries: [
                    { binding: 1, resource: mipView(level - 1) },
                    { binding: 2, resource: storage[level] },
                ],
            }),
        }));
    }

    /**
     * Resize for a depth buffer of `width` x `height` (a no-op at the current size). The texture
     * and its views are recreated: rebind them.
     */
    resize(width: number, height: number): void {
        if (this._sourceSize[0] === width && this._sourceSize[1] === height) return;
        this.createMips(width, height);
        this._sourceSize = [width, height];
    }

    /**
     * Record the build from `depth` (a single-sampled depth view of `sourceSize`): one compute
     * pass, one dispatch per mip.
     */
    build(encoder: GPUCommandEncoder, depth: GPUTextureView): void {
        const first = this.device.createBindGroup({
            label: 'DepthPyramid/FromDepth',
            layout: this.depthLayout,
            entries: [
                { binding: 0, resource: depth },
                { binding: 2, resource: this._mips[0].storage },
            ],
        });
        this.record(encoder, this.fromDepth, first);
    }

    /**
     * `build`, of view distances (-z in view space) instead of depths: each depth unprojected with
     * `inverseProjection` (the inverse of the projection it was rasterized with); where nothing
     * was drawn, infinitely far. For projections whose depth does not grow with the distance alike
     * on every pixel, as with an oblique near plane (planar reflections).
     */
    buildLinear(encoder: GPUCommandEncoder, depth: GPUTextureView, inverseProjection: mat4 | Float32Array): void {
        const data = this.linearData;
        data.set(inverseProjection, 0);
        data[16] = this._sourceSize[0];
        data[17] = this._sourceSize[1];
        this.device.queue.writeBuffer(this.linear.uniform, 0, data);
        const first = this.device.createBindGroup({
            label: 'DepthPyramid/FromDepthLinear',
            layout: this.linear.layout,
            entries: [
                { binding: 0, resource: depth },
                { binding: 2, resource: this._mips[0].storage },
                { binding: 3, resource: { buffer: this.linear.uniform } },
            ],
        });
        this.record(encoder, this.linear.pipeline, first);
    }

    private record(encoder: GPUCommandEncoder, firstPipeline: GPUComputePipeline, first: GPUBindGroup): void {
        const pass = encoder.beginComputePass({ label: 'DepthPyramid', timestampWrites: gpuPass('DepthPyramid') });
        this._mips.forEach((mip, level) => {
            if (level === 0) {
                pass.setPipeline(firstPipeline);
                pass.setBindGroup(0, first);
            } else {
                if (level === 1) pass.setPipeline(this.fromMip);
                pass.setBindGroup(0, mip.bindGroup!);
            }
            pass.dispatchWorkgroups(Math.ceil(mip.size[0] / 8), Math.ceil(mip.size[1] / 8), 1);
        });
        pass.end();
    }

    /** The pyramid's texture (`depthReductionFormat(reduction)`, `mipCount` mips). */
    get texture(): GPUTexture {
        return this._texture;
    }

    /** A view of every mip, to bind as `texture_2d<f32>` and read with `textureLoad`. */
    get view(): GPUTextureView {
        return this._view;
    }

    get mipCount(): number {
        return this._mips.length;
    }

    /** The size of mip `level`. */
    mipSize(level: number): [number, number] {
        return this._mips[level].size;
    }

    /** The size of the depth buffer it is built from. */
    get sourceSize(): [number, number] {
        return this._sourceSize;
    }

    /** Frees the texture and the linearize uniform. */
    destroy(): void {
        this._texture.destroy();
        this.linear.uniform.destroy();
    }
}
