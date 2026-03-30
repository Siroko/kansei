import { mat4, vec3 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { InstancedGeometry } from '../geometries/InstancedGeometry';
import { PointLight } from '../lights/PointLight';
import { AreaLight } from '../lights/AreaLight';
import { Renderer } from '../renderers/Renderer';
import { Scene } from '../objects/Scene';

/** Any light with a world-space position and radius, usable for cubemap shadow rendering. */
export type PositionalLight = PointLight | AreaLight;

export interface CubeMapShadowMapOptions {
    resolution?: number;
    maxLights?: number;
    near?: number;
    shadowFar?: number;
}

// Standard cubemap face directions and up vectors
const FACE_DIRS: [number, number, number][] = [
    [ 1,  0,  0], // +X
    [-1,  0,  0], // -X
    [ 0,  1,  0], // +Y
    [ 0, -1,  0], // -Y
    [ 0,  0,  1], // +Z
    [ 0,  0, -1], // -Z
];
const FACE_UPS: [number, number, number][] = [
    [0, -1,  0],
    [0, -1,  0],
    [0,  0,  1],
    [0,  0, -1],
    [0, -1,  0],
    [0, -1,  0],
];

class CubeMapShadowMap {
    private _device: GPUDevice;
    private _resolution: number;
    private _maxLights: number;
    private _near: number;
    private _shadowFar: number;

    private _distanceTexture: GPUTexture;
    private _scratchDepthTexture: GPUTexture;

    private _pipeline: GPURenderPipeline | null = null;
    private _customPipelines: Map<string, GPURenderPipeline> = new Map();

    // Light VP + light position uniform (group 0, dynamic offset)
    // One slot per face per light, aligned to minUniformBufferOffsetAlignment
    private _lightUniformBuffer: GPUBuffer | null = null;
    private _lightUniformBGL: GPUBindGroupLayout;
    private _lightUniformBG: GPUBindGroup | null = null;
    private _lightUniformCapacity = 0; // max face slots allocated
    private _uniformAlignment: number;

    // Own mesh matrix buffers (group 1, dynamic offset)
    private _meshBGL: GPUBindGroupLayout;
    private _worldMatBuf: GPUBuffer | null = null;
    private _normalMatBuf: GPUBuffer | null = null;
    private _meshBG: GPUBindGroup | null = null;
    private _worldStaging: Float32Array | null = null;
    private _normalStaging: Float32Array | null = null;
    private _objectCapacity = 0;

    // Scratch gl-matrix temporaries
    private _viewMat = mat4.create();
    private _projMat = mat4.create();
    private _vpMat = mat4.create();

    constructor(device: GPUDevice, options?: CubeMapShadowMapOptions) {
        this._device = device;
        this._resolution = options?.resolution ?? 512;
        this._maxLights = options?.maxLights ?? 8;
        this._near = options?.near ?? 0.1;
        this._shadowFar = options?.shadowFar ?? 500;
        this._uniformAlignment = device.limits.minUniformBufferOffsetAlignment ?? 256;

        // r32float 2D array: 6 faces per light
        this._distanceTexture = device.createTexture({
            label: 'CubeMapShadow/Distance',
            size: [this._resolution, this._resolution, 6 * this._maxLights],
            format: 'r32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        // Scratch depth for rasterization Z-testing (reused across all face renders)
        this._scratchDepthTexture = device.createTexture({
            label: 'CubeMapShadow/ScratchDepth',
            size: [this._resolution, this._resolution],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        // Light uniform BGL with dynamic offset — one slot per face
        this._lightUniformBGL = device.createBindGroupLayout({
            label: 'CubeMapShadow/LightUniform BGL',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT,
                buffer: { type: 'uniform', hasDynamicOffset: true },
            }],
        });

        this._meshBGL = device.createBindGroupLayout({
            label: 'CubeMapShadow/Mesh BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
            ],
        });
    }

    get distanceTexture(): GPUTexture { return this._distanceTexture; }
    get maxLights(): number { return this._maxLights; }

    private _ensureLightUniformBuffer(faceCount: number): void {
        if (faceCount <= this._lightUniformCapacity && this._lightUniformBG) return;

        this._lightUniformBuffer?.destroy();
        const bufferSize = faceCount * this._uniformAlignment;
        this._lightUniformBuffer = this._device.createBuffer({
            label: 'CubeMapShadow/LightUniform',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._lightUniformBG = this._device.createBindGroup({
            label: 'CubeMapShadow/LightUniform BG',
            layout: this._lightUniformBGL,
            entries: [{ binding: 0, resource: { buffer: this._lightUniformBuffer, size: 80 } }],
        });
        this._lightUniformCapacity = faceCount;
    }

    private _ensureMeshBuffers(objectCount: number): void {
        if (objectCount <= this._objectCapacity && this._meshBG) return;

        const alignment = this._uniformAlignment;
        const bufferSize = Math.max(objectCount * alignment, alignment);
        const floatsPerSlot = alignment / 4;

        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();

        this._worldMatBuf = this._device.createBuffer({
            label: 'CubeMapShadow/WorldMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._normalMatBuf = this._device.createBuffer({
            label: 'CubeMapShadow/NormalMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._worldStaging = new Float32Array(objectCount * floatsPerSlot);
        this._normalStaging = new Float32Array(objectCount * floatsPerSlot);

        this._meshBG = this._device.createBindGroup({
            label: 'CubeMapShadow/Mesh BG',
            layout: this._meshBGL,
            entries: [
                { binding: 0, resource: { buffer: this._normalMatBuf, size: 64 } },
                { binding: 1, resource: { buffer: this._worldMatBuf, size: 64 } },
            ],
        });

        this._objectCapacity = objectCount;
    }

    private _ensurePipeline(vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null>): void {
        if (this._pipeline) return;

        const shaderCode = /* wgsl */`
            struct LightUniform {
                lightViewProj : mat4x4f,
                lightWorldPos : vec3f,
                _pad          : f32,
            }

            @group(0) @binding(0) var<uniform> light       : LightUniform;
            @group(1) @binding(0) var<uniform> normalMatrix : mat4x4f;
            @group(1) @binding(1) var<uniform> worldMatrix  : mat4x4f;

            struct VSOut {
                @builtin(position) position : vec4f,
                @location(0)       worldPos : vec3f,
            }

            @vertex
            fn shadow_vs(
                @location(0) position : vec4f,
                @location(1) normal   : vec3f,
                @location(2) uv       : vec2f,
            ) -> VSOut {
                let wp = worldMatrix * position;
                var out : VSOut;
                out.position = light.lightViewProj * wp;
                out.worldPos = wp.xyz;
                return out;
            }

            @fragment
            fn shadow_fs(in : VSOut) -> @location(0) f32 {
                return length(in.worldPos - light.lightWorldPos);
            }
        `;

        const module = this._device.createShaderModule({
            label: 'CubeMapShadow/Shader',
            code: shaderCode,
        });

        const pipelineLayout = this._device.createPipelineLayout({
            label: 'CubeMapShadow/PipelineLayout',
            bindGroupLayouts: [this._lightUniformBGL, this._meshBGL],
        });

        this._pipeline = this._device.createRenderPipeline({
            label: 'CubeMapShadow/Pipeline',
            layout: pipelineLayout,
            vertex: {
                module,
                entryPoint: 'shadow_vs',
                buffers: vertexBuffersDescriptors,
            },
            fragment: {
                module,
                entryPoint: 'shadow_fs',
                targets: [{ format: 'r32float' }],
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });
    }

    private _getOrCreateCustomPipeline(
        shadowVertexCode: string,
        vertexBuffers: Iterable<GPUVertexBufferLayout | null>,
        extraBGL: GPUBindGroupLayout | null,
    ): GPURenderPipeline {
        let pipeline = this._customPipelines.get(shadowVertexCode);
        if (pipeline) return pipeline;

        const shaderCode = /* wgsl */`
            struct LightUniform {
                lightViewProj : mat4x4f,
                lightWorldPos : vec3f,
                _pad          : f32,
            }

            @group(0) @binding(0) var<uniform> light       : LightUniform;
            @group(1) @binding(0) var<uniform> normalMatrix : mat4x4f;
            @group(1) @binding(1) var<uniform> worldMatrix  : mat4x4f;

            ${shadowVertexCode}

            struct VSOut {
                @builtin(position) position : vec4f,
                @location(0)       worldPos : vec3f,
            }

            @vertex
            fn shadow_vs(
                @location(0) position : vec4f,
                @location(1) normal   : vec3f,
                @location(2) uv       : vec2f,
                @builtin(instance_index) instanceIdx : u32,
            ) -> VSOut {
                let wp = shadowWorldPos(position, instanceIdx);
                var out : VSOut;
                out.position = light.lightViewProj * wp;
                out.worldPos = wp.xyz;
                return out;
            }

            @fragment
            fn shadow_fs(in : VSOut) -> @location(0) f32 {
                return length(in.worldPos - light.lightWorldPos);
            }
        `;

        const module = this._device.createShaderModule({
            label: 'CubeMapShadow/CustomShader',
            code: shaderCode,
        });

        const layouts: GPUBindGroupLayout[] = [this._lightUniformBGL, this._meshBGL];
        if (extraBGL) layouts.push(extraBGL);

        pipeline = this._device.createRenderPipeline({
            label: 'CubeMapShadow/CustomPipeline',
            layout: this._device.createPipelineLayout({
                label: 'CubeMapShadow/CustomPipelineLayout',
                bindGroupLayouts: layouts,
            }),
            vertex: {
                module,
                entryPoint: 'shadow_vs',
                buffers: vertexBuffers,
            },
            fragment: {
                module,
                entryPoint: 'shadow_fs',
                targets: [{ format: 'r32float' }],
            },
            depthStencil: {
                format: 'depth32float',
                depthWriteEnabled: true,
                depthCompare: 'less',
            },
            primitive: {
                topology: 'triangle-list',
                cullMode: 'back',
            },
        });

        this._customPipelines.set(shadowVertexCode, pipeline);
        return pipeline;
    }

    render(_renderer: Renderer, scene: Scene, _camera: Camera, pointLights: readonly PositionalLight[]): void {
        const device = this._device;
        if (pointLights.length === 0) return;

        const objects = scene.getOrderedObjects();
        if (objects.length === 0) return;

        // Ensure pipeline (lazy — needs vertex layout from first geometry)
        if (!objects[0].geometry.initialized) {
            objects[0].geometry.initialize(device);
        }
        this._ensurePipeline(objects[0].geometry.vertexBuffersDescriptors);

        const alignment = this._uniformAlignment;
        const floatsPerSlot = alignment / 4;

        this._ensureMeshBuffers(objects.length);

        // Upload all object world matrices once
        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            if (!obj.geometry.initialized) obj.geometry.initialize(device);
            obj.updateModelMatrix();
            this._worldStaging!.set(obj.worldMatrix.internalMat4, i * floatsPerSlot);
            this._normalStaging!.set(obj.normalMatrix.internalMat4, i * floatsPerSlot);
        }

        device.queue.writeBuffer(this._worldMatBuf!, 0, this._worldStaging!.buffer as ArrayBuffer);
        device.queue.writeBuffer(this._normalMatBuf!, 0, this._normalStaging!.buffer as ArrayBuffer);

        const lightCount = Math.min(pointLights.length, this._maxLights);
        const totalFaces = lightCount * 6;

        // Ensure light uniform buffer is large enough for all faces
        this._ensureLightUniformBuffer(totalFaces);

        // Pre-compute all face VP matrices and upload in one writeBuffer
        const uniformStaging = new Float32Array(totalFaces * floatsPerSlot);

        for (let li = 0; li < lightCount; li++) {
            const light = pointLights[li];
            light.updateModelMatrix();
            const wm = light.worldMatrix.internalMat4;
            const lightPos: vec3 = [wm[12], wm[13], wm[14]];

            // Shadow far plane is decoupled from light radius so distant
            // occluders outside the attenuation volume are still captured.
            const shadowFar = Math.max(this._shadowFar, light.radius);
            mat4.perspective(this._projMat, Math.PI / 2, 1.0, this._near, shadowFar);
            // Remap Z from [-1,1] to [0,1] for WebGPU
            (this._projMat as unknown as Float32Array)[10] *= 0.5;
            (this._projMat as unknown as Float32Array)[14] =
                (this._projMat as unknown as Float32Array)[14] * 0.5 + 0.5;

            for (let face = 0; face < 6; face++) {
                const dir = FACE_DIRS[face];
                const up = FACE_UPS[face];
                const target: vec3 = [
                    lightPos[0] + dir[0],
                    lightPos[1] + dir[1],
                    lightPos[2] + dir[2],
                ];
                mat4.lookAt(this._viewMat, lightPos, target, up);
                mat4.multiply(this._vpMat, this._projMat, this._viewMat);

                const slotBase = (li * 6 + face) * floatsPerSlot;
                uniformStaging.set(this._vpMat as unknown as Float32Array, slotBase);
                uniformStaging[slotBase + 16] = lightPos[0];
                uniformStaging[slotBase + 17] = lightPos[1];
                uniformStaging[slotBase + 18] = lightPos[2];
            }
        }

        device.queue.writeBuffer(this._lightUniformBuffer!, 0, uniformStaging.buffer as ArrayBuffer);

        // Record all face render passes into one command encoder
        const commandEncoder = device.createCommandEncoder({ label: 'CubeMapShadow' });
        const scratchDepthView = this._scratchDepthTexture.createView();

        for (let li = 0; li < lightCount; li++) {
            const light = pointLights[li];
            const clearDist = Math.max(this._shadowFar, light.radius);

            for (let face = 0; face < 6; face++) {
                const faceSlot = li * 6 + face;
                const lightOffset = faceSlot * alignment;

                const colorView = this._distanceTexture.createView({
                    dimension: '2d',
                    baseArrayLayer: faceSlot,
                    arrayLayerCount: 1,
                });

                const pass = commandEncoder.beginRenderPass({
                    colorAttachments: [{
                        view: colorView,
                        clearValue: { r: clearDist, g: 0, b: 0, a: 0 },
                        loadOp: 'clear',
                        storeOp: 'store',
                    }],
                    depthStencilAttachment: {
                        view: scratchDepthView,
                        depthClearValue: 1.0,
                        depthLoadOp: 'clear',
                        depthStoreOp: 'discard',
                    },
                });

                pass.setBindGroup(0, this._lightUniformBG!, [lightOffset]);

                let activePipeline: GPURenderPipeline | null = null;
                let currentVertexBuffer: GPUBuffer | null = null;
                let currentIndexBuffer: GPUBuffer | null = null;

                for (let i = 0; i < objects.length; i++) {
                    const obj = objects[i];
                    if (!obj.castShadow) continue;
                    if (!obj.geometry.initialized) continue;

                    let targetPipeline: GPURenderPipeline;
                    if (obj.shadowVertexCode) {
                        targetPipeline = this._getOrCreateCustomPipeline(
                            obj.shadowVertexCode,
                            obj.geometry.vertexBuffersDescriptors,
                            obj.shadowExtraBGL,
                        );
                    } else {
                        targetPipeline = this._pipeline!;
                    }

                    if (targetPipeline !== activePipeline) {
                        pass.setPipeline(targetPipeline);
                        activePipeline = targetPipeline;
                        currentVertexBuffer = null;
                        currentIndexBuffer = null;
                    }

                    const offset = i * alignment;
                    pass.setBindGroup(1, this._meshBG!, [offset, offset]);

                    if (obj.shadowExtraBG) {
                        pass.setBindGroup(2, obj.shadowExtraBG);
                    }

                    if (obj.geometry.vertexBuffer !== currentVertexBuffer) {
                        pass.setVertexBuffer(0, obj.geometry.vertexBuffer!);
                        currentVertexBuffer = obj.geometry.vertexBuffer!;
                    }
                    if (obj.geometry.indexBuffer !== currentIndexBuffer) {
                        pass.setIndexBuffer(obj.geometry.indexBuffer!, obj.geometry.indexFormat!);
                        currentIndexBuffer = obj.geometry.indexBuffer!;
                    }

                    if (obj.geometry.isInstancedGeometry) {
                        const geo = obj.geometry as InstancedGeometry;
                        let idx = 1;
                        for (const extraBuf of geo.extraBuffers) {
                            if (!extraBuf.initialized) extraBuf.initialize(device);
                            pass.setVertexBuffer(idx++, extraBuf.resource.buffer);
                        }
                        pass.drawIndexed(geo.vertexCount, geo.instanceCount, 0, 0, 0);
                    } else {
                        pass.drawIndexed(obj.geometry.vertexCount);
                    }
                }

                pass.end();
            }
        }

        device.queue.submit([commandEncoder.finish()]);
    }

    destroy(): void {
        this._distanceTexture?.destroy();
        this._scratchDepthTexture?.destroy();
        this._lightUniformBuffer?.destroy();
        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();
        this._pipeline = null;
        this._customPipelines.clear();
    }
}

export { CubeMapShadowMap };
