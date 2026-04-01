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

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'FluidSurface/Params',
            size: 144,
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
        const data = new Float32Array(36); // 144 bytes
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
        u32[32] = width;
        u32[33] = height;
        u32[34] = 0;
        u32[35] = 0;

        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, data.buffer, 0, 144);

        const pass = commandEncoder.beginComputePass({ label: 'FluidSurface/March' });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bg!);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(_width: number, _height: number, _gbuffer: GBuffer): void {
        this._currentInput = null; // Force bind group rebuild on next render
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
    }
}

export { FluidSurfaceEffect };
