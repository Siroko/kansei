import { Camera } from '../../cameras/Camera';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';
import { shaderCode } from './shaders/fluid-transmission.wgsl';

export interface FluidTransmissionOptions {
    color?: [number, number, number];
    ior?: number;
    chromaticAberration?: number;
    tintStrength?: number;
    fresnelPower?: number;
    roughness?: number;
    thickness?: number;
}

/**
 * FluidTransmissionEffect — screen-space refraction composite.
 *
 * Runs AFTER the scene has been rendered into the GBuffer (with any transmissive
 * objects drawn on top and `GBuffer.backgroundTexture` holding the opaque-only
 * result). Uses `GBuffer.normalTexture` as a mask to detect transmissive pixels
 * and samples `backgroundTexture` with a normal-driven offset to fake refraction.
 *
 * Use with a Material that has `transmissive: true`: the renderer automatically
 * snapshots the background between the opaque and transmissive sub-passes, so
 * this effect can consume that snapshot.
 */
class FluidTransmissionEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;

    public color: [number, number, number];
    public ior: number;
    public chromaticAberration: number;
    public tintStrength: number;
    public fresnelPower: number;
    public roughness: number;
    public thickness: number;

    private _pipeline: GPUComputePipeline | null = null;
    private _bgl: GPUBindGroupLayout | null = null;
    private _bg: GPUBindGroup | null = null;
    private _paramsBuffer: GPUBuffer | null = null;

    private _currentInput: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;
    private _currentBackground: GPUTexture | null = null;
    private _currentNormal: GPUTexture | null = null;

    constructor(options: FluidTransmissionOptions = {}) {
        super();
        this.color = options.color ?? [0.77, 0.96, 1.0];
        this.ior = options.ior ?? 1.41;
        this.chromaticAberration = options.chromaticAberration ?? 0.05;
        this.tintStrength = options.tintStrength ?? 0.30;
        this.fresnelPower = options.fresnelPower ?? 2.3;
        this.roughness = options.roughness ?? 0.28;
        this.thickness = options.thickness ?? 2.4;
    }

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;

        this._paramsBuffer = device.createBuffer({
            label: 'FluidTransmission/Params',
            size: 144,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        const module = device.createShaderModule({ label: 'FluidTransmission/Shader', code: shaderCode });

        this._bgl = device.createBindGroupLayout({
            label: 'FluidTransmission/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'unfilterable-float' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'FluidTransmission/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [this._bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.outputTexture, gbuffer);
        this.initialized = true;
    }

    private _buildBindGroup(input: GPUTexture, output: GPUTexture, gbuffer: GBuffer): void {
        this._bg = this._device!.createBindGroup({
            label: 'FluidTransmission/BG',
            layout: this._bgl!,
            entries: [
                { binding: 0, resource: { buffer: this._paramsBuffer! } },
                { binding: 1, resource: input.createView() },
                { binding: 2, resource: gbuffer.backgroundTexture.createView() },
                { binding: 3, resource: gbuffer.normalTexture.createView() },
                { binding: 4, resource: output.createView() },
            ],
        });
        this._currentInput = input;
        this._currentOutput = output;
        this._currentBackground = gbuffer.backgroundTexture;
        this._currentNormal = gbuffer.normalTexture;
    }

    render(
        commandEncoder: GPUCommandEncoder,
        input: GPUTexture,
        _depth: GPUTexture,
        output: GPUTexture,
        camera: Camera,
        width: number,
        height: number,
    ): void {
        if (!this._pipeline || !this._device) return;

        // The PostProcessingVolume passes depth/input/output but not the GBuffer itself.
        // Our stored background/normal refs came from initialize(). If the pingpong
        // flips input/output across frames we still need the SAME background+normal.
        if (input !== this._currentInput || output !== this._currentOutput) {
            this._bg = this._device.createBindGroup({
                label: 'FluidTransmission/BG',
                layout: this._bgl!,
                entries: [
                    { binding: 0, resource: { buffer: this._paramsBuffer! } },
                    { binding: 1, resource: input.createView() },
                    { binding: 2, resource: this._currentBackground!.createView() },
                    { binding: 3, resource: this._currentNormal!.createView() },
                    { binding: 4, resource: output.createView() },
                ],
            });
            this._currentInput = input;
            this._currentOutput = output;
        }

        // Upload params.
        const data = new Float32Array(36); // 144 bytes
        data.set(camera.viewMatrix.internalMat4 as unknown as Float32Array, 0); // 0..15 view matrix
        data[16] = this.color[0];
        data[17] = this.color[1];
        data[18] = this.color[2];
        data[19] = 1.0;
        data[20] = this.ior;
        data[21] = this.chromaticAberration;
        data[22] = this.tintStrength;
        data[23] = this.fresnelPower;
        data[24] = this.roughness;
        data[25] = this.thickness;
        data[26] = width;
        data[27] = height;
        // remaining entries stay zero
        this._device.queue.writeBuffer(this._paramsBuffer!, 0, data.buffer, 0, 144);

        const pass = commandEncoder.beginComputePass({ label: 'FluidTransmission/Composite' });
        pass.setPipeline(this._pipeline);
        pass.setBindGroup(0, this._bg!);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    resize(_width: number, _height: number, gbuffer: GBuffer): void {
        this._currentBackground = gbuffer.backgroundTexture;
        this._currentNormal = gbuffer.normalTexture;
        this._currentInput = null; // force rebuild on next render
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
    }
}

export { FluidTransmissionEffect };
