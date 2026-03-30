import { vec3, mat4 } from 'gl-matrix';
import { Camera } from '../cameras/Camera';
import { InstancedGeometry } from '../geometries/InstancedGeometry';
import { Renderer } from '../renderers/Renderer';
import { Scene } from '../objects/Scene';
import { DirectionalLight } from '../lights/DirectionalLight';
import { PointLight } from '../lights/PointLight';
import { AreaLight } from '../lights/AreaLight';

export interface ShadowMapOptions {
    resolution?: number;
    /** Maximum distance from the camera that shadows cover.
     *  Tighter values give higher effective shadow resolution. */
    maxShadowDistance?: number;
    /** Field-of-view (degrees) for perspective shadow maps (area lights). */
    fov?: number;
    /** Near plane for perspective shadow maps (area lights). */
    near?: number;
    /** Far plane for perspective shadow maps (area lights). 0 = use light radius. */
    far?: number;
}

class ShadowMap {
    private _device: GPUDevice;
    private _resolution: number;
    private _maxShadowDistance: number;
    private _fov: number;
    private _near: number;
    private _far: number;
    private _depthTexture: GPUTexture;
    private _lightVP = new Float32Array(16);

    private _pipeline: GPURenderPipeline | null = null;
    private _customPipelines: Map<string, GPURenderPipeline> = new Map();

    // Light VP uniform (group 0)
    private _lightVPBuffer: GPUBuffer;
    private _lightVPBGL: GPUBindGroupLayout;
    private _lightVPBG: GPUBindGroup;

    // Own mesh matrix buffers (group 1, same layout as Renderer's shared mesh)
    private _meshBGL: GPUBindGroupLayout;
    private _worldMatBuf: GPUBuffer | null = null;
    private _normalMatBuf: GPUBuffer | null = null;
    private _meshBG: GPUBindGroup | null = null;
    private _worldStaging: Float32Array | null = null;
    private _normalStaging: Float32Array | null = null;
    private _objectCapacity = 0;

    // Scratch gl-matrix temporaries
    private _lightView = mat4.create();
    private _lightProj = mat4.create();
    private _lightVPMat = mat4.create();
    private _invViewProj = mat4.create();

    constructor(device: GPUDevice, options?: ShadowMapOptions) {
        this._device = device;
        this._resolution = options?.resolution ?? 2048;
        this._maxShadowDistance = options?.maxShadowDistance ?? 0;
        this._fov = options?.fov ?? 90;
        this._near = options?.near ?? 0.1;
        this._far = options?.far ?? 0;

        this._depthTexture = device.createTexture({
            label: 'ShadowMap/Depth',
            size: [this._resolution, this._resolution],
            format: 'depth32float',
            usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
        });

        this._lightVPBuffer = device.createBuffer({
            label: 'ShadowMap/LightVP',
            size: 64,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._lightVPBGL = device.createBindGroupLayout({
            label: 'ShadowMap/LightVP BGL',
            entries: [{
                binding: 0,
                visibility: GPUShaderStage.VERTEX,
                buffer: { type: 'uniform' },
            }],
        });

        this._lightVPBG = device.createBindGroup({
            label: 'ShadowMap/LightVP BG',
            layout: this._lightVPBGL,
            entries: [{ binding: 0, resource: { buffer: this._lightVPBuffer } }],
        });

        this._meshBGL = device.createBindGroupLayout({
            label: 'ShadowMap/Mesh BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
                { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'uniform', hasDynamicOffset: true } },
            ],
        });
    }

    get depthTexture(): GPUTexture { return this._depthTexture; }
    get lightViewProjMatrix(): Float32Array { return this._lightVP; }
    get maxShadowDistance(): number { return this._maxShadowDistance; }
    set maxShadowDistance(v: number) { this._maxShadowDistance = v; }
    /** Field-of-view in degrees (perspective shadow maps). */
    get fov(): number { return this._fov; }
    set fov(v: number) { this._fov = v; }
    /** Near plane (perspective shadow maps). */
    get near(): number { return this._near; }
    set near(v: number) { this._near = v; }
    /** Far plane (perspective shadow maps). 0 = use light radius. */
    get far(): number { return this._far; }
    set far(v: number) { this._far = v; }

    private _computeLightVP(camera: Camera, lightDir: [number, number, number]): void {
        // When maxShadowDistance is set, build a tighter projection so
        // the shadow frustum only covers nearby geometry.
        const useShadowFar = this._maxShadowDistance > 0
            ? Math.min(this._maxShadowDistance, camera.far)
            : camera.far;

        // Build a (possibly tighter) view-projection and invert
        const vp = mat4.create();
        if (useShadowFar < camera.far) {
            // Temporary projection with clamped far plane
            const tmpProj = mat4.create();
            mat4.perspective(tmpProj, camera.fov * Math.PI / 180, camera.aspect, camera.near, useShadowFar);
            mat4.multiply(vp, tmpProj, camera.viewMatrix.internalMat4);
        } else {
            mat4.multiply(vp, camera.projectionMatrix.internalMat4, camera.viewMatrix.internalMat4);
        }
        mat4.invert(this._invViewProj, vp);

        // 8 frustum corners in NDC (WebGPU depth [0,1])
        const ndcCorners = [
            [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
            [-1, -1, 1], [1, -1, 1], [-1, 1, 1], [1, 1, 1],
        ];

        const worldCorners: vec3[] = [];
        const center = vec3.create();
        const m = this._invViewProj;

        for (const ndc of ndcCorners) {
            const x = m[0] * ndc[0] + m[4] * ndc[1] + m[8] * ndc[2] + m[12];
            const y = m[1] * ndc[0] + m[5] * ndc[1] + m[9] * ndc[2] + m[13];
            const z = m[2] * ndc[0] + m[6] * ndc[1] + m[10] * ndc[2] + m[14];
            const w = m[3] * ndc[0] + m[7] * ndc[1] + m[11] * ndc[2] + m[15];
            const corner = vec3.fromValues(x / w, y / w, z / w);
            worldCorners.push(corner);
            vec3.add(center, center, corner);
        }
        vec3.scale(center, center, 1 / 8);

        // Light view: look from behind the center along light direction
        const ld = vec3.fromValues(lightDir[0], lightDir[1], lightDir[2]);
        vec3.normalize(ld, ld);
        const eye = vec3.create();
        vec3.scaleAndAdd(eye, center, ld, -100);
        const up = Math.abs(ld[1]) < 0.99
            ? vec3.fromValues(0, 1, 0)
            : vec3.fromValues(1, 0, 0);
        mat4.lookAt(this._lightView, eye, center, up);

        // AABB in light space
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
        const lp = vec3.create();
        for (const wc of worldCorners) {
            vec3.transformMat4(lp, wc, this._lightView);
            minX = Math.min(minX, lp[0]); maxX = Math.max(maxX, lp[0]);
            minY = Math.min(minY, lp[1]); maxY = Math.max(maxY, lp[1]);
            minZ = Math.min(minZ, lp[2]); maxZ = Math.max(maxZ, lp[2]);
        }

        // Extend Z to catch shadow casters behind the camera
        const zRange = maxZ - minZ;
        minZ -= zRange * 2;

        // gl-matrix ortho expects positive distances for near/far,
        // but minZ/maxZ are negative view-space Z coords (objects in
        // front of the light have Z < 0).  Negate & swap so that
        // near = -maxZ (closest) and far = -minZ (farthest).
        mat4.ortho(this._lightProj, minX, maxX, minY, maxY, -maxZ, -minZ);

        // gl-matrix v3 ortho maps Z to [-1,1]; remap to [0,1] for WebGPU
        (this._lightProj as unknown as Float32Array)[10] *= 0.5;
        (this._lightProj as unknown as Float32Array)[14] =
            (this._lightProj as unknown as Float32Array)[14] * 0.5 + 0.5;

        mat4.multiply(this._lightVPMat, this._lightProj, this._lightView);
        this._lightVP.set(this._lightVPMat as unknown as Float32Array);
    }

    /**
     * Compute a perspective light VP from an area light's position/target.
     */
    private _computeAreaLightVP(light: AreaLight): void {
        light.updateModelMatrix();
        const wm = light.worldMatrix.internalMat4;
        const lightPos: vec3 = [wm[12], wm[13], wm[14]];
        const target: vec3 = [light.target.x, light.target.y, light.target.z];

        const up: vec3 = Math.abs(light.direction[1]) < 0.99
            ? [0, 1, 0]
            : [1, 0, 0];
        mat4.lookAt(this._lightView, lightPos, target, up);

        const near = this._near;
        const far = this._far > 0 ? this._far : light.radius;
        mat4.perspective(this._lightProj, this._fov * Math.PI / 180, 1.0, near, far);

        // gl-matrix perspective maps Z to [-1,1]; remap to [0,1] for WebGPU.
        // For perspective P[11]=-1, the correct remap is:
        //   P'[10] = P[10]*0.5 + P[11]*0.5
        //   P'[14] = P[14]*0.5
        const P = this._lightProj as unknown as Float32Array;
        P[10] = P[10] * 0.5 + P[11] * 0.5;
        P[14] = P[14] * 0.5;

        mat4.multiply(this._lightVPMat, this._lightProj, this._lightView);
        this._lightVP.set(this._lightVPMat as unknown as Float32Array);
    }

    /**
     * Compute a perspective light VP from a point light's position,
     * looking toward the provided target (defaults to origin).
     */
    private _computePointLightVP(light: PointLight, target: vec3 = [0, 0, 0]): void {
        light.updateModelMatrix();
        const wm = light.worldMatrix.internalMat4;
        const lightPos: vec3 = [wm[12], wm[13], wm[14]];

        const dx = target[0] - lightPos[0];
        const dy = target[1] - lightPos[1];
        const dz = target[2] - lightPos[2];
        const len = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const dirY = len > 1e-8 ? dy / len : -1;

        const up: vec3 = Math.abs(dirY) < 0.99 ? [0, 1, 0] : [1, 0, 0];
        mat4.lookAt(this._lightView, lightPos, target, up);

        const near = this._near;
        const far = this._far > 0 ? this._far : light.radius;
        mat4.perspective(this._lightProj, this._fov * Math.PI / 180, 1.0, near, far);

        const P = this._lightProj as unknown as Float32Array;
        P[10] = P[10] * 0.5 + P[11] * 0.5;
        P[14] = P[14] * 0.5;

        mat4.multiply(this._lightVPMat, this._lightProj, this._lightView);
        this._lightVP.set(this._lightVPMat as unknown as Float32Array);
    }

    private _ensureMeshBuffers(objectCount: number): void {
        if (objectCount <= this._objectCapacity && this._meshBG) return;

        const alignment = this._device.limits.minUniformBufferOffsetAlignment ?? 256;
        const bufferSize = Math.max(objectCount * alignment, alignment);
        const floatsPerSlot = alignment / 4;

        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();

        this._worldMatBuf = this._device.createBuffer({
            label: 'ShadowMap/WorldMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._normalMatBuf = this._device.createBuffer({
            label: 'ShadowMap/NormalMatrices',
            size: bufferSize,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._worldStaging = new Float32Array(objectCount * floatsPerSlot);
        this._normalStaging = new Float32Array(objectCount * floatsPerSlot);

        this._meshBG = this._device.createBindGroup({
            label: 'ShadowMap/Mesh BG',
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
            @group(0) @binding(0) var<uniform> lightViewProj : mat4x4f;
            @group(1) @binding(0) var<uniform> normalMatrix  : mat4x4f;
            @group(1) @binding(1) var<uniform> worldMatrix   : mat4x4f;

            @vertex
            fn shadow_vs(
                @location(0) position : vec4f,
                @location(1) normal   : vec3f,
                @location(2) uv       : vec2f,
            ) -> @builtin(position) vec4f {
                return lightViewProj * worldMatrix * position;
            }
        `;

        const module = this._device.createShaderModule({
            label: 'ShadowMap/Shader',
            code: shaderCode,
        });

        const pipelineLayout = this._device.createPipelineLayout({
            label: 'ShadowMap/PipelineLayout',
            bindGroupLayouts: [this._lightVPBGL, this._meshBGL],
        });

        this._pipeline = this._device.createRenderPipeline({
            label: 'ShadowMap/Pipeline',
            layout: pipelineLayout,
            vertex: {
                module,
                entryPoint: 'shadow_vs',
                buffers: vertexBuffersDescriptors,
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
            @group(0) @binding(0) var<uniform> lightViewProj : mat4x4f;
            @group(1) @binding(0) var<uniform> normalMatrix  : mat4x4f;
            @group(1) @binding(1) var<uniform> worldMatrix   : mat4x4f;

            ${shadowVertexCode}

            @vertex
            fn shadow_vs(
                @location(0) position : vec4f,
                @location(1) normal   : vec3f,
                @location(2) uv       : vec2f,
                @builtin(instance_index) instanceIdx : u32,
            ) -> @builtin(position) vec4f {
                let wp = shadowWorldPos(position, instanceIdx);
                return lightViewProj * wp;
            }
        `;

        const module = this._device.createShaderModule({
            label: 'ShadowMap/CustomShader',
            code: shaderCode,
        });

        const layouts: GPUBindGroupLayout[] = [this._lightVPBGL, this._meshBGL];
        if (extraBGL) layouts.push(extraBGL);

        pipeline = this._device.createRenderPipeline({
            label: 'ShadowMap/CustomPipeline',
            layout: this._device.createPipelineLayout({
                label: 'ShadowMap/CustomPipelineLayout',
                bindGroupLayouts: layouts,
            }),
            vertex: {
                module,
                entryPoint: 'shadow_vs',
                buffers: vertexBuffers,
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

    render(_renderer: Renderer, scene: Scene, camera: Camera, lightDirOrLight: DirectionalLight | AreaLight | PointLight | [number, number, number], target?: [number, number, number]): void {
        const device = this._device;

        scene.prepare(camera);
        camera.updateViewMatrix();
        const objects = scene.getOrderedObjects();
        if (objects.length === 0) return;

        if (lightDirOrLight instanceof AreaLight) {
            this._computeAreaLightVP(lightDirOrLight);
        } else if (lightDirOrLight instanceof PointLight) {
            this._computePointLightVP(lightDirOrLight, target ?? [0, 0, 0]);
        } else {
            const lightDir: [number, number, number] = Array.isArray(lightDirOrLight)
                ? lightDirOrLight
                : lightDirOrLight.direction;
            this._computeLightVP(camera, lightDir);
        }
        device.queue.writeBuffer(this._lightVPBuffer, 0, this._lightVP.buffer as ArrayBuffer);

        // Ensure pipeline (lazy — needs vertex layout from first geometry)
        if (!objects[0].geometry.initialized) {
            objects[0].geometry.initialize(device);
        }
        this._ensurePipeline(objects[0].geometry.vertexBuffersDescriptors);

        this._ensureMeshBuffers(objects.length);

        const alignment = device.limits.minUniformBufferOffsetAlignment ?? 256;
        const floatsPerSlot = alignment / 4;

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            if (!obj.geometry.initialized) obj.geometry.initialize(device);
            obj.updateModelMatrix();
            this._worldStaging!.set(obj.worldMatrix.internalMat4, i * floatsPerSlot);
            this._normalStaging!.set(obj.normalMatrix.internalMat4, i * floatsPerSlot);
        }

        device.queue.writeBuffer(this._worldMatBuf!, 0, this._worldStaging!.buffer as ArrayBuffer);
        device.queue.writeBuffer(this._normalMatBuf!, 0, this._normalStaging!.buffer as ArrayBuffer);

        const commandEncoder = device.createCommandEncoder({ label: 'ShadowMap' });
        const pass = commandEncoder.beginRenderPass({
            colorAttachments: [],
            depthStencilAttachment: {
                view: this._depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        pass.setBindGroup(0, this._lightVPBG);

        let activePipeline: GPURenderPipeline | null = null;
        let currentVertexBuffer: GPUBuffer | null = null;
        let currentIndexBuffer: GPUBuffer | null = null;

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            if (!obj.castShadow) continue;
            if (!obj.geometry.initialized) continue;

            // Select default or custom shadow pipeline
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
        device.queue.submit([commandEncoder.finish()]);
    }

    destroy(): void {
        this._depthTexture?.destroy();
        this._lightVPBuffer?.destroy();
        this._worldMatBuf?.destroy();
        this._normalMatBuf?.destroy();
        this._pipeline = null;
        this._customPipelines.clear();
    }
}

export { ShadowMap };
