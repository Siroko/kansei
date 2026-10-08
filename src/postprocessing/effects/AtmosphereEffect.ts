import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { SKY_COMPOSITE_SOURCE } from '../../atmosphere/AtmosphereWGSL';
import { SkyAtmosphere, SkyAtmosphereBindings } from '../../atmosphere/SkyAtmosphere';
import { gpuPass } from '../../profiling/Profiler';

/**
 * Renders a `SkyAtmosphere`: the sky, the sun and the moon wherever the scene left the depth
 * buffer at the far plane, and aerial perspective (the atmosphere between the camera and each
 * surface) everywhere else. Put it first in the chain, before the fog and the tonemapper, and
 * call `SkyAtmosphere.update` every frame before rendering. The Rust engine's
 * `postprocessing/effects/atmosphere.rs`, on the same WGSL (`sky_composite.wgsl`).
 *
 * The sky is in physical units (cd/m², a 100 000 lux sun), so it needs a `ToneMapEffect` with an
 * EV100 exposure (`exposureFromEV100`) to be seen.
 */
export class AtmosphereEffect extends PostProcessingEffect {
    private readonly _sky: SkyAtmosphereBindings;
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _bindGroup: GPUBindGroup | null = null;
    private _bound: [GPUTexture | null, GPUTexture | null, GPUTexture | null] = [null, null, null];

    constructor(sky: SkyAtmosphere) {
        super();
        this._sky = sky.bindings;
    }

    initialize(device: GPUDevice, _gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        const C = GPUShaderStage.COMPUTE;
        const uniform = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, buffer: { type: 'uniform' } });
        const texture = (binding: number, viewDimension: GPUTextureViewDimension = '2d'): GPUBindGroupLayoutEntry =>
            ({ binding, visibility: C, texture: { sampleType: 'float', viewDimension } });
        const sampler = (binding: number): GPUBindGroupLayoutEntry => ({ binding, visibility: C, sampler: { type: 'filtering' } });
        const layout = device.createBindGroupLayout({
            label: 'Atmosphere/CompositeBGL',
            entries: [
                uniform(0), uniform(1), texture(2), texture(3), sampler(4), sampler(5),
                { binding: 6, visibility: C, texture: { sampleType: 'unfilterable-float' } },
                { binding: 7, visibility: C, texture: { sampleType: 'depth' } },
                { binding: 8, visibility: C, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                texture(9, '3d'), texture(10, '3d'),
            ],
        });
        this._pipeline = device.createComputePipeline({
            label: 'Atmosphere/Composite',
            layout: device.createPipelineLayout({ label: 'Atmosphere/Composite', bindGroupLayouts: [layout] }),
            compute: { module: device.createShaderModule({ label: 'Atmosphere/Composite', code: SKY_COMPOSITE_SOURCE }), entryPoint: 'main' },
        });
        this.initialized = true;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture,
        _camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._pipeline) return;
        const b = this._bound;
        if (!this._bindGroup || b[0] !== input || b[1] !== depth || b[2] !== output) {
            const s = this._sky;
            const resources: GPUBindingResource[] = [
                { buffer: s.atmosphere }, { buffer: s.frame }, s.transmittance, s.skyView, s.lutSampler, s.skyViewSampler,
                input.createView(), depth.createView(), output.createView(), s.apScattering, s.apTransmittance,
            ];
            this._bindGroup = this._device!.createBindGroup({
                label: 'Atmosphere/CompositeBG',
                layout: this._pipeline.getBindGroupLayout(0),
                entries: resources.map((resource, binding) => ({ binding, resource })),
            });
            this._bound = [input, depth, output];
        }
        const pass = commandEncoder.beginComputePass({ label: 'Atmosphere/Composite', timestampWrites: gpuPass('Atmosphere/Composite') });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bindGroup);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        // the GBuffer's textures are new: the bind group follows the next render's
        this._bindGroup = null;
    }

    destroy(): void {
        this._pipeline = null;
        this._bindGroup = null;
        this._bound = [null, null, null];
        this.initialized = false;
    }
}
