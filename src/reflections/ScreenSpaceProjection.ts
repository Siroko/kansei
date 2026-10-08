import { gpuPass } from "../profiling/Profiler";
import { assemble } from "../materials/shaders/ShaderUtils";
import type { ReflectionFog } from "./PlanarReflection";
import froxelCommon from "../../rust/kansei-core/src/shaders/froxel_common.wgsl?raw";
import screenSpaceWgsl from "../../rust/kansei-core/src/shaders/planar_reflection_screen_space.wgsl?raw";

/**
 * The screen-space path of `PlanarReflection` (`PlanarReflection.screenSpace`): last frame's
 * GBuffer projected across the plane into the reflection texture. See the WGSL. Rust:
 * `reflections/screen_space.rs`.
 */
export const SCREEN_SPACE_WGSL: string = assemble([froxelCommon, screenSpaceWgsl]);

/**
 * Bytes of the WGSL `Params`: last frame's inverse view-projection (as drawn, jitter included),
 * this frame's view-projection, the plane, the camera position and the surface's height
 * tolerance, the source and reflection sizes, and the screen rectangle the reflection is needed
 * in (x0, y0, x1, y1; empty: nothing is projected).
 */
export const SCREEN_SPACE_PARAMS_BYTES = 192;

/** Pipelines and per-texel buffers of the projection, for a reflection texture of one size. */
export class ScreenSpaceProjection {
    private readonly _clear: GPUComputePipeline;
    private readonly _project: GPUComputePipeline;
    private readonly _own: GPUComputePipeline;
    private readonly _resolve: GPUComputePipeline;
    private readonly _bgl: GPUBindGroupLayout;
    private readonly _heights: GPUBuffer;
    private readonly _owners: GPUBuffer;
    private readonly _params: GPUBuffer;
    // the bind group and what it was made from
    private _bindGroup: GPUBindGroup | null = null;
    private _bound: unknown[] = [];

    constructor(private readonly device: GPUDevice, width: number, height: number) {
        const compute = GPUShaderStage.COMPUTE;
        this._bgl = device.createBindGroupLayout({
            label: 'PlanarReflection/ScreenSpaceBGL',
            entries: [
                { binding: 0, visibility: compute, texture: { sampleType: 'unfilterable-float' } },
                { binding: 1, visibility: compute, texture: { sampleType: 'depth' } },
                { binding: 2, visibility: compute, buffer: { type: 'storage' } },
                { binding: 3, visibility: compute, buffer: { type: 'storage' } },
                { binding: 4, visibility: compute, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 5, visibility: compute, buffer: { type: 'uniform' } },
                { binding: 6, visibility: compute, texture: { sampleType: 'float', viewDimension: '3d' } },
                { binding: 7, visibility: compute, sampler: { type: 'filtering' } },
                { binding: 8, visibility: compute, buffer: { type: 'uniform' } },
            ],
        });
        const module = device.createShaderModule({ label: 'PlanarReflection/ScreenSpace', code: SCREEN_SPACE_WGSL });
        const layout = device.createPipelineLayout({ label: 'PlanarReflection/ScreenSpace', bindGroupLayouts: [this._bgl] });
        const pipeline = (entryPoint: string) => device.createComputePipeline({
            label: 'PlanarReflection/ScreenSpace',
            layout,
            compute: { module, entryPoint },
        });
        this._clear = pipeline('clear');
        this._project = pipeline('project');
        this._own = pipeline('own');
        this._resolve = pipeline('resolve');
        const texels = width * height * 4;
        this._heights = device.createBuffer({ label: 'PlanarReflection/ScreenSpaceHeights', size: texels, usage: GPUBufferUsage.STORAGE });
        this._owners = device.createBuffer({ label: 'PlanarReflection/ScreenSpaceOwners', size: texels, usage: GPUBufferUsage.STORAGE });
        this._params = device.createBuffer({
            label: 'PlanarReflection/ScreenSpaceParams',
            size: SCREEN_SPACE_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * Project `color` and `depth` (last frame's, of the source size in `params`) into `dst` (mip 0
     * of the reflection, of the reflection's size), with `fog` composited over what it reflects.
     * `params` is the WGSL `Params` (`SCREEN_SPACE_PARAMS_BYTES`).
     */
    run(
        encoder: GPUCommandEncoder,
        color: GPUTexture,
        depth: GPUTexture,
        dst: GPUTextureView,
        fog: ReflectionFog,
        fogSampler: GPUSampler,
        params: ArrayBuffer,
    ): void {
        this.device.queue.writeBuffer(this._params, 0, params);
        const bound = [color, depth, dst, fog.volume, fog.params, fogSampler];
        if (!this._bindGroup || bound.some((b, i) => b !== this._bound[i])) {
            this._bindGroup = this.device.createBindGroup({
                label: 'PlanarReflection/ScreenSpaceBG',
                layout: this._bgl,
                entries: [
                    { binding: 0, resource: color.createView() },
                    { binding: 1, resource: depth.createView() },
                    { binding: 2, resource: { buffer: this._heights } },
                    { binding: 3, resource: { buffer: this._owners } },
                    { binding: 4, resource: dst },
                    { binding: 5, resource: { buffer: this._params } },
                    { binding: 6, resource: fog.volume.createView() },
                    { binding: 7, resource: fogSampler },
                    { binding: 8, resource: { buffer: fog.params } },
                ],
            });
            this._bound = bound;
        }
        const u32 = new Uint32Array(params);
        const [sw, sh, dw, dh] = [u32[40], u32[41], u32[42], u32[43]];
        const pass = encoder.beginComputePass({ label: 'PlanarReflection/ScreenSpace', timestampWrites: gpuPass('PlanarReflection/ScreenSpace') });
        pass.setBindGroup(0, this._bindGroup);
        pass.setPipeline(this._clear);
        pass.dispatchWorkgroups(Math.ceil(dw / 8), Math.ceil(dh / 8));
        pass.setPipeline(this._project);
        pass.dispatchWorkgroups(Math.ceil(sw / 8), Math.ceil(sh / 8));
        pass.setPipeline(this._own);
        pass.dispatchWorkgroups(Math.ceil(sw / 8), Math.ceil(sh / 8));
        pass.setPipeline(this._resolve);
        pass.dispatchWorkgroups(Math.ceil(dw / 8), Math.ceil(dh / 8));
        pass.end();
    }

    destroy(): void {
        this._heights.destroy();
        this._owners.destroy();
        this._params.destroy();
    }
}
