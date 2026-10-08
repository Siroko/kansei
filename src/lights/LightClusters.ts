import { mat4 } from "gl-matrix";
import type { Camera } from "../cameras/Camera";
import { LIGHT_CLUSTERS_WGSL } from "../materials/shaders/SharedWGSL";
import { gpuPass } from "../profiling/Profiler";
import { CLUSTER_PARAMS_BYTES } from "../renderers/SharedLayouts";

/** Clusters: 16 x 9 screen tiles x 24 exponential depth slices. */
export const CLUSTER_GRID: readonly [number, number, number] = [16, 9, 24];
/** Per cluster: the light count, then up to 31 light indices (`KANSEI_CLUSTER_SLOTS`). */
export const CLUSTER_SLOTS = 32;
/** Depth the slices reach at most; beyond it, fragments use the last slice. */
const MAX_CLUSTER_FAR = 2000;
const CLUSTER_COUNT = CLUSTER_GRID[0] * CLUSTER_GRID[1] * CLUSTER_GRID[2];

/**
 * The renderer's clustered light lists (group 3 bindings 8-9; Rust `lights::light_clusters`),
 * rebuilt every frame for the camera by a compute pass, so materials shade each fragment with
 * the spot lights that reach it (`kansei_spot_lights_radiance`).
 */
export class LightClusters {
    /** `KanseiClusterParams` (group 3 binding 8). */
    readonly params: GPUBuffer;
    /** Per cluster, the light count then its light indices (group 3 binding 9). */
    readonly lights: GPUBuffer;
    private _pipeline: GPUComputePipeline | null = null;
    private _bindGroup: GPUBindGroup | null = null;
    private readonly _staging = new ArrayBuffer(CLUSTER_PARAMS_BYTES);
    private readonly _f32 = new Float32Array(this._staging);
    private readonly _u32 = new Uint32Array(this._staging);
    private readonly _invProj = mat4.create();
    // Whether the params buffer holds `disable`'s values (they need writing only once).
    private _disabled = false;

    constructor(private readonly device: GPUDevice, private readonly spotLights: GPUBuffer) {
        this.params = device.createBuffer({
            label: 'LightClusters/Params',
            size: CLUSTER_PARAMS_BYTES,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.lights = device.createBuffer({
            label: 'LightClusters/Lights',
            size: CLUSTER_COUNT * CLUSTER_SLOTS * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        this.disable();
    }

    /**
     * Fills the staging `KanseiClusterParams` for `camera` over a `width` x `height` target:
     * view, inverse projection, screen size, the depth range the slices span, the grid, enabled.
     */
    private _stageParams(camera: Camera, width: number, height: number): void {
        const f = this._f32;
        f.set(camera.viewMatrix.internalMat4 as Float32Array, 0);
        f.set(mat4.invert(this._invProj, camera.projectionMatrix.internalMat4) as Float32Array, 16);
        f[32] = width;
        f[33] = height;
        const near = Math.max(camera.near, 1e-3);
        f[34] = near;
        f[35] = Math.max(Math.min(camera.far, MAX_CLUSTER_FAR), camera.near * 2);
        this._u32.set(CLUSTER_GRID, 36);
        this._u32[39] = 1;
    }

    /**
     * Shades with every light: for views the clusters aren't built for. Takes effect for the
     * command buffers submitted after it.
     */
    disable(): void {
        if (this._disabled) return;
        this._f32.fill(0);
        this._f32[34] = 1;
        this._f32[35] = 2;
        this._u32.set(CLUSTER_GRID, 36);
        this.device.queue.writeBuffer(this.params, 0, this._staging);
        this._disabled = true;
    }

    /**
     * Builds the clusters for `camera` over a `width` x `height` target (the pixels the shading
     * pass covers), into `encoder`: after the frame's spot lights are uploaded, before the passes
     * that shade with them.
     */
    encode(encoder: GPUCommandEncoder, camera: Camera, width: number, height: number): void {
        this._stageParams(camera, width, height);
        this.device.queue.writeBuffer(this.params, 0, this._staging);
        this._disabled = false;

        if (!this._pipeline) {
            const device = this.device;
            const layout = device.createBindGroupLayout({
                label: 'LightClusters/BGL',
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                ],
            });
            this._pipeline = device.createComputePipeline({
                label: 'LightClusters/Build',
                layout: device.createPipelineLayout({ label: 'LightClusters', bindGroupLayouts: [layout] }),
                compute: { module: device.createShaderModule({ label: 'LightClusters', code: LIGHT_CLUSTERS_WGSL }), entryPoint: 'main' },
            });
            this._bindGroup = device.createBindGroup({
                label: 'LightClusters/BG',
                layout,
                entries: [
                    { binding: 0, resource: { buffer: this.params } },
                    { binding: 1, resource: { buffer: this.spotLights } },
                    { binding: 2, resource: { buffer: this.lights } },
                ],
            });
        }

        const pass = encoder.beginComputePass({ label: 'LightClusters/Build', timestampWrites: gpuPass('LightClusters/Build') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup!);
        pass.dispatchWorkgroups(Math.ceil(CLUSTER_COUNT / 64));
        pass.end();
    }

    destroy(): void {
        this.params.destroy();
        this.lights.destroy();
    }
}
