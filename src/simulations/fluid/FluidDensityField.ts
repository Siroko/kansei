import { Renderer } from '../../renderers/Renderer';
import { FluidSimulation } from './FluidSimulation';
import { shaderCode as clearShader } from './shaders/density-field-clear.wgsl';
import { shaderCode as splatShader } from './shaders/density-field-splat.wgsl';
import { shaderCode as copyShader } from './shaders/density-field-copy.wgsl';
import surfaceSplatShader from '../../../rust/kansei-core/src/simulations/fluid/shaders/surface-field-splat.wgsl?raw';
import surfaceCopyShader from '../../../rust/kansei-core/src/simulations/fluid/shaders/surface-field-copy.wgsl?raw';
import { gpuPass } from '../../profiling/Profiler';

export interface FluidDensityFieldOptions {
    resolution?: number;
    kernelScale?: number;
    /**
     * Unset: the field is the particles' splatted density (`kernelScale` scales it). Set: it is
     * a surface field instead (Zhu & Bridson), `1 + (r - d) / h` with `d` the distance to the
     * kernel-weighted mean of the particles within the splat radius `h`: its iso level is 1,
     * where a lone particle is a droplet of radius `r` and a flat layer of particles a flat
     * surface, however they are stacked. `kernelScale` is then 1 over the kernel weight the bulk
     * sums to (particles per unit volume × 0.638 h³ for its `(1 - d²/h²)³` kernel): a voxel
     * weighing past a third of that is inside, so walls and floors under the fluid make no
     * surface. The Rust engine's `DensityFieldOptions::particle_radius`.
     */
    particleRadius?: number;
    /**
     * The splat's kernel radius `h`; unset, the simulation's smoothing radius. Set it when the
     * smoothing radius is smaller than a field voxel, or the field is sparse and the surface
     * shatters into shards (the Rust `FluidSurfaceEffect::splat_radius`).
     */
    splatRadius?: number;
}

/**
 * The particles splatted into a 3D `rgba16float` texture over the simulation's bounds, for
 * `FluidMarchingCubes` or `FluidRaymarchEffect`: their density, or with `particleRadius` a
 * surface field whose iso level is 1. Runs on the simulation's live particles.
 */
class FluidDensityField {
    private _renderer: Renderer;
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
    /** Scales the splatted density, or normalises the surface field's weight (see the options). */
    public kernelScale: number;
    /** The splat's kernel radius; unset, the simulation's smoothing radius. */
    public splatRadius: number | undefined;
    private _particleRadius: number | undefined;

    constructor(renderer: Renderer, sim: FluidSimulation, options?: FluidDensityFieldOptions) {
        this._renderer = renderer;
        this._device = renderer.gpuDevice;
        this._sim = sim;

        const maxRes = options?.resolution ?? 64;
        this.kernelScale = options?.kernelScale ?? 1.0;
        this.splatRadius = options?.splatRadius;
        this._particleRadius = options?.particleRadius;

        // Compute texture dims proportional to bounds aspect ratio,
        // with the largest axis at `resolution` and others scaled proportionally.
        const bMin = sim.worldBoundsMin;
        const bMax = sim.worldBoundsMax;
        const extents = [bMax[0] - bMin[0], bMax[1] - bMin[1], bMax[2] - bMin[2]];
        const maxExtent = Math.max(...extents, 0.001);
        this._texDims = [
            Math.max(Math.round((extents[0] / maxExtent) * maxRes), 1),
            Math.max(Math.round((extents[1] / maxExtent) * maxRes), 1),
            Math.max(Math.round((extents[2] / maxExtent) * maxRes), 1),
        ];

        this._paramsData = new Float32Array(16); // 64 bytes: the surface field's params, the density's in the first 48
        this._paramsU32 = new Uint32Array(this._paramsData.buffer);

        this._createResources();
        this._createPipelines();
        // Bind groups are built lazily in update() because sim buffers
        // may not be GPU-initialized yet (BufferBase._resource is set lazily).
    }

    get densityTexture(): GPUTexture { return this._densityTex; }
    get densityView(): GPUTextureView { return this._densityView; }
    get texDims(): [number, number, number] { return this._texDims; }
    get boundsMin(): [number, number, number] { return [...this._sim.worldBoundsMin] as [number, number, number]; }
    get boundsMax(): [number, number, number] { return [...this._sim.worldBoundsMax] as [number, number, number]; }
    /** The surface field's particle radius (unset: a density field). */
    get particleRadius(): number | undefined { return this._particleRadius; }

    /** The surface field's particle radius (a field made with `particleRadius`; ignored for a density field). */
    set particleRadius(radius: number | undefined) {
        if (this._particleRadius !== undefined && radius !== undefined) {
            this._particleRadius = radius;
        }
    }

    /** Whether this is a surface field (made with `particleRadius`). */
    get isSurfaceField(): boolean { return this._particleRadius !== undefined; }

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
            // the surface field accumulates a weight and a weighted offset per voxel
            size: w * h * d * (this.isSurfaceField ? 16 : 4),
            usage: GPUBufferUsage.STORAGE,
        });

        this._paramsBuffer = this._device.createBuffer({
            label: 'FluidDensityField/Params',
            size: 64,
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
        const splatModule = device.createShaderModule({ label: 'DensityField/Splat', code: this.isSurfaceField ? surfaceSplatShader : splatShader });
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
        const copyModule = device.createShaderModule({ label: 'DensityField/Copy', code: this.isSurfaceField ? surfaceCopyShader : copyShader });
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
        u[3] = this._sim.particleCount;
        p[4] = bMin[0]; p[5] = bMin[1]; p[6] = bMin[2];
        p[7] = this.splatRadius ?? this._sim.params.smoothingRadius;
        p[8] = bMax[0]; p[9] = bMax[1]; p[10] = bMax[2];
        p[11] = this.kernelScale;
        p[12] = this._particleRadius ?? 0;

        this._device.queue.writeBuffer(this._paramsBuffer, 0, this._paramsData);
    }

    /**
     * Run the density field update (clear → splat → copy).
     * If no encoder is provided, creates a new one and submits it immediately.
     */
    update(commandEncoder?: GPUCommandEncoder): void {
        if (!commandEncoder) {
            const encoder = this._renderer.createCommandEncoder('DensityField/Encoder');
            this._record(encoder);
            this._renderer.submit([encoder.finish()]);
            return;
        }
        this._record(commandEncoder);
    }

    private _record(commandEncoder: GPUCommandEncoder): void {
        // Lazily build bind groups on first update (sim buffers are now GPU-initialized)
        if (!this._splatBG) {
            const posRef = this._sim.positionsBufferRef;
            if (!(posRef as any)._resource) {
                posRef.initialize(this._device);
            }
            this._buildBindGroups();
        }
        this._uploadParams();

        const [w, h, d] = this._texDims;

        const clearPass = commandEncoder.beginComputePass({ label: 'DensityField/Clear', timestampWrites: gpuPass('DensityField/Clear') });
        clearPass.setPipeline(this._clearPipeline);
        clearPass.setBindGroup(0, this._clearBG);
        clearPass.dispatchWorkgroups(Math.ceil(w / 4), Math.ceil(h / 4), Math.ceil(d / 4));
        clearPass.end();

        const particleCount = this._sim.particleCount;
        const splatPass = commandEncoder.beginComputePass({ label: 'DensityField/Splat', timestampWrites: gpuPass('DensityField/Splat') });
        splatPass.setPipeline(this._splatPipeline);
        splatPass.setBindGroup(0, this._splatBG);
        splatPass.dispatchWorkgroups(Math.ceil(particleCount / 64));
        splatPass.end();

        const copyPass = commandEncoder.beginComputePass({ label: 'DensityField/Copy', timestampWrites: gpuPass('DensityField/Copy') });
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
