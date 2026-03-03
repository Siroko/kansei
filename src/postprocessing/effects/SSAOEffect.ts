import { Camera } from '../../cameras/Camera';
import { Matrix4 } from '../../math/Matrix4';
import { GBuffer } from '../GBuffer';
import { PostProcessingEffect } from '../PostProcessingEffect';

export interface SSAOOptions {
    /** World-space sampling radius around each fragment. Default 0.5 */
    radius?: number;
    /** Depth-comparison bias to prevent self-occlusion. Default 0.025 */
    bias?: number;
    /** Number of hemisphere samples (1–32). Default 16 */
    kernelSize?: number;
    /** Occlusion multiplier applied to the colour result. Default 1.0 */
    strength?: number;
}

/**
 * Screen-Space Ambient Occlusion (SSAO)
 * ======================================
 * Reconstructs view-space positions and normals entirely from the depth buffer
 * (no separate normals render pass required).  A hemisphere of random samples is
 * projected back to screen space and compared against the stored depth to
 * determine how much each fragment is occluded by surrounding geometry.
 *
 * The effect outputs the scene colour modulated by the AO factor.
 */
class SSAOEffect extends PostProcessingEffect {
    private _device: GPUDevice | null = null;
    private _pipeline: GPUComputePipeline | null = null;
    private _paramsBuffer: GPUBuffer | null = null;
    private _kernelBuffer: GPUBuffer | null = null;
    private _bindGroup: GPUBindGroup | null = null;
    private _currentInput: GPUTexture | null = null;
    private _currentDepth: GPUTexture | null = null;
    private _currentOutput: GPUTexture | null = null;

    private readonly _radius: number;
    private readonly _bias: number;
    private readonly _kernelSize: number;
    private readonly _strength: number;
    private readonly _kernelData: Float32Array;

    // Reusable scratch matrix for inverse-projection computation.
    private _invProj: Matrix4 = new Matrix4();

    constructor(options: SSAOOptions = {}) {
        super();
        this._radius     = options.radius     ?? 0.5;
        this._bias       = options.bias       ?? 0.025;
        this._kernelSize = Math.min(Math.max(options.kernelSize ?? 16, 1), 32);
        this._strength   = options.strength   ?? 1.0;
        this._kernelData = this._generateKernel(32); // always generate 32; shader caps at kernelSize
    }

    // ── Hemisphere sample kernel ─────────────────────────────────────────────

    private _generateKernel(size: number): Float32Array {
        const data = new Float32Array(size * 4);
        for (let i = 0; i < size; i++) {
            const phi      = Math.random() * Math.PI * 2;
            const cosTheta = Math.random();
            const sinTheta = Math.sqrt(1 - cosTheta * cosTheta);

            // Lerp(0.1, 1.0, (i/size)^2) — more samples near the origin.
            const t     = i / size;
            const scale = 0.1 + t * t * 0.9;

            data[i * 4 + 0] = Math.cos(phi) * sinTheta * scale;
            data[i * 4 + 1] = Math.sin(phi) * sinTheta * scale;
            data[i * 4 + 2] = cosTheta * scale;
            data[i * 4 + 3] = 0.0;
        }
        return data;
    }

    // ── WGSL compute shader ──────────────────────────────────────────────────

    private static readonly _SHADER = /* wgsl */`
        struct SSAOParams {
            projMatrix    : mat4x4f,
            invProjMatrix : mat4x4f,
            screenWidth   : f32,
            screenHeight  : f32,
            radius        : f32,
            bias          : f32,
            kernelSize    : u32,
            strength      : f32,
            _pad0         : f32,
            _pad1         : f32,
        }

        @group(0) @binding(0) var depthTex   : texture_depth_2d;
        @group(0) @binding(1) var colorTex   : texture_2d<f32>;
        @group(0) @binding(2) var outputTex  : texture_storage_2d<rgba16float, write>;
        @group(0) @binding(3) var<uniform>  params : SSAOParams;
        @group(0) @binding(4) var<uniform>  kernel : array<vec4f, 32>;

        // Reconstruct a view-space position from a UV coordinate and a raw depth value.
        fn viewPosFromDepth(uv: vec2f, depth: f32) -> vec3f {
            let ndc     = vec4f(uv.x * 2.0 - 1.0, (1.0 - uv.y) * 2.0 - 1.0, depth, 1.0);
            let viewPos = params.invProjMatrix * ndc;
            return viewPos.xyz / viewPos.w;
        }

        // Estimate the view-space normal at a pixel by finite differences on the depth buffer.
        fn viewNormalFromDepth(coord: vec2u) -> vec3f {
            let w = u32(params.screenWidth);
            let h = u32(params.screenHeight);

            let rCoord = vec2u(min(coord.x + 1u, w - 1u), coord.y);
            let lCoord = vec2u(select(0u, coord.x - 1u, coord.x > 0u), coord.y);
            let uCoord = vec2u(coord.x, select(0u, coord.y - 1u, coord.y > 0u));
            let dCoord = vec2u(coord.x, min(coord.y + 1u, h - 1u));

            let pR = viewPosFromDepth(vec2f(f32(rCoord.x) / params.screenWidth, f32(rCoord.y) / params.screenHeight), textureLoad(depthTex, rCoord, 0));
            let pL = viewPosFromDepth(vec2f(f32(lCoord.x) / params.screenWidth, f32(lCoord.y) / params.screenHeight), textureLoad(depthTex, lCoord, 0));
            let pU = viewPosFromDepth(vec2f(f32(uCoord.x) / params.screenWidth, f32(uCoord.y) / params.screenHeight), textureLoad(depthTex, uCoord, 0));
            let pD = viewPosFromDepth(vec2f(f32(dCoord.x) / params.screenWidth, f32(dCoord.y) / params.screenHeight), textureLoad(depthTex, dCoord, 0));

            return normalize(cross(pR - pL, pD - pU));
        }

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) gid : vec3u) {
            let coord = gid.xy;
            let w     = u32(params.screenWidth);
            let h     = u32(params.screenHeight);
            if (coord.x >= w || coord.y >= h) { return; }

            let uv    = vec2f(f32(coord.x) / params.screenWidth, f32(coord.y) / params.screenHeight);
            let depth = textureLoad(depthTex, coord, 0);

            // Background/sky — no occlusion; pass colour through unchanged.
            if (depth >= 1.0) {
                textureStore(outputTex, coord, textureLoad(colorTex, coord, 0));
                return;
            }

            let viewPos    = viewPosFromDepth(uv, depth);
            let viewNormal = viewNormalFromDepth(coord);

            // Build a tangent-space basis aligned with the surface normal.
            let up        = select(vec3f(0.0, 1.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(viewNormal.y) > 0.99);
            let tangent   = normalize(up - viewNormal * dot(up, viewNormal));
            let bitangent = cross(viewNormal, tangent);
            let tbn       = mat3x3f(tangent, bitangent, viewNormal);

            var occlusion : f32 = 0.0;

            for (var i = 0u; i < params.kernelSize; i++) {
                // Orient hemisphere sample along the surface normal.
                let sampleViewPos = viewPos + (tbn * kernel[i].xyz) * params.radius;

                // Project into clip / NDC / screen space.
                let clip      = params.projMatrix * vec4f(sampleViewPos, 1.0);
                let ndc       = clip.xyz / clip.w;
                let sampleUV  = vec2f(ndc.x * 0.5 + 0.5, 1.0 - (ndc.y * 0.5 + 0.5));

                if (sampleUV.x < 0.0 || sampleUV.x > 1.0 ||
                    sampleUV.y < 0.0 || sampleUV.y > 1.0) {
                    continue;
                }

                let sampleCoord = vec2u(
                    u32(sampleUV.x * params.screenWidth),
                    u32(sampleUV.y * params.screenHeight)
                );
                let sampleDepth   = textureLoad(depthTex, sampleCoord, 0);
                let sampleViewRef = viewPosFromDepth(sampleUV, sampleDepth);

                // Range check avoids bleeding occlusion over large depth discontinuities.
                let rangeCheck = smoothstep(0.0, 1.0, params.radius / abs(viewPos.z - sampleViewRef.z));
                occlusion += select(0.0, 1.0, sampleViewRef.z >= sampleViewPos.z + params.bias) * rangeCheck;
            }

            let ao    = 1.0 - (occlusion / f32(params.kernelSize)) * params.strength;
            let color = textureLoad(colorTex, coord, 0);
            textureStore(outputTex, coord, vec4f(color.rgb * ao, color.a));
        }
    `;

    // ── PostProcessingEffect interface ───────────────────────────────────────

    initialize(device: GPUDevice, gbuffer: GBuffer, _camera: Camera): void {
        this._device = device;
        // Params uniform buffer: 2×mat4 + 8 floats = 128 + 32 = 160 bytes → padded to 256.
        this._paramsBuffer = device.createBuffer({
            label: 'SSAO/Params',
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // Kernel uniform buffer: 32 × vec4f = 512 bytes.
        this._kernelBuffer = device.createBuffer({
            label: 'SSAO/Kernel',
            size: 512,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        device.queue.writeBuffer(this._kernelBuffer, 0, this._kernelData.buffer as ArrayBuffer);

        const module = device.createShaderModule({ code: SSAOEffect._SHADER });
        const bgl = device.createBindGroupLayout({
            label: 'SSAO/BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'depth' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, texture: { sampleType: 'float' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, storageTexture: { access: 'write-only', format: 'rgba16float' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        this._pipeline = device.createComputePipeline({
            label: 'SSAO/Pipeline',
            layout: device.createPipelineLayout({ bindGroupLayouts: [bgl] }),
            compute: { module, entryPoint: 'main' },
        });

        this._buildBindGroup(gbuffer.colorTexture, gbuffer.depthTexture, gbuffer.outputTexture);
        this.initialized = true;
    }

    private _buildBindGroup(
        input: GPUTexture,
        depth: GPUTexture,
        output: GPUTexture
    ): void {
        const bgl = this._pipeline!.getBindGroupLayout(0);
        this._bindGroup = this._device!.createBindGroup({
            label: 'SSAO/BindGroup',
            layout: bgl,
            entries: [
                { binding: 0, resource: depth.createView() },
                { binding: 1, resource: input.createView() },
                { binding: 2, resource: output.createView() },
                { binding: 3, resource: { buffer: this._paramsBuffer! } },
                { binding: 4, resource: { buffer: this._kernelBuffer! } },
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
        if (!this._pipeline) return;

        // Rebuild bind group when textures change (ping-pong).
        if (input !== this._currentInput || depth !== this._currentDepth || output !== this._currentOutput) {
            this._buildBindGroup(input, depth, output);
        }

        // Update params uniform.
        this._invProj.invert(camera.projectionMatrix);
        const params = new Float32Array(64); // 256-byte buffer as f32 view
        params.set(camera.projectionMatrix.internalMat4, 0);  // offset 0  (16 floats)
        params.set(this._invProj.internalMat4,            16); // offset 64 (16 floats)
        params[32] = width;
        params[33] = height;
        params[34] = this._radius;
        params[35] = this._bias;
        // kernelSize as uint — write raw bytes at byte offset 144 (float index 36)
        new Uint32Array(params.buffer, 36 * 4, 1)[0] = this._kernelSize;
        params[37] = this._strength;

        this._device!.queue.writeBuffer(this._paramsBuffer!, 0, params.buffer as ArrayBuffer);

        const wg = (t: number) => Math.ceil(t / 8);
        const pass = commandEncoder.beginComputePass({ label: 'SSAO' });
        pass.setPipeline(this._pipeline!);
        pass.setBindGroup(0, this._bindGroup!);
        pass.dispatchWorkgroups(wg(width), wg(height));
        pass.end();
    }

    resize(_w: number, _h: number, _gbuffer: GBuffer): void {
        // Params are recalculated every frame — nothing size-dependent to recreate.
    }

    destroy(): void {
        this._paramsBuffer?.destroy();
        this._kernelBuffer?.destroy();
        this._paramsBuffer = null;
        this._kernelBuffer = null;
        this._pipeline = null;
        this._bindGroup = null;
    }
}

export { SSAOEffect };
