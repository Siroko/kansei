import { Renderer } from '../../renderers/Renderer';
import { Compute } from '../../materials/Compute';
import { ComputeBuffer } from '../../buffers/ComputeBuffer';
import { BufferBase } from '../../buffers/BufferBase';
import { Matrix4 } from '../../math/Matrix4';
import { IBindable } from '../../buffers/IBindable';
import {
    FluidSimulationOptions,
    DEFAULT_OPTIONS,
    PARAMS,
    PRESETS,
    computeKernelFactors2D,
    computeKernelFactors3D,
} from './FluidSimulationParams';

import { shaderCode as gridClearShader } from './shaders/grid-clear.wgsl';
import { shaderCode as gridAssignShader } from './shaders/grid-assign.wgsl';
import { shaderCode as prefixSumLocalShader } from './shaders/prefix-sum-local.wgsl';
import { shaderCode as prefixSumTopShader } from './shaders/prefix-sum-top.wgsl';
import { shaderCode as prefixSumDistributeShader } from './shaders/prefix-sum-distribute.wgsl';
import { shaderCode as scatterShader } from './shaders/scatter.wgsl';
import { shaderCode as densityShader } from './shaders/density.wgsl';
import { shaderCode as forcesShader } from './shaders/forces.wgsl';
import { shaderCode as integrateShader } from './shaders/integrate.wgsl';
import { shaderCode as bodyCollisionShader } from './shaders/body-collision.wgsl';
import { shaderCode as bodyIntegrateShader } from './shaders/body-integrate.wgsl';
import { FluidBody, FluidBodyOptions } from './FluidBody';

const MAX_GRID_CELLS = 262144; // 256K
const PREFIX_SUM_BLOCK_SIZE = 512;
const MAX_BODIES = 64;
const MAX_PRIMITIVES = 256;
const BODY_STATE_FLOATS = 24;
const PRIMITIVE_FLOATS = 6;

class FluidSimulation {
    public params: FluidSimulationOptions;

    private renderer: Renderer;
    private particleCount: number = 0;

    // Params uniform buffer (dual view for mixed f32/u32)
    private paramsF32!: Float32Array;
    private paramsU32!: Uint32Array;
    private paramsBuffer!: ComputeBuffer;

    // Internal simulation buffers
    private velocitiesBuffer!: ComputeBuffer;
    private densitiesBuffer!: ComputeBuffer;
    private cellIndicesBuffer!: ComputeBuffer;
    private cellCountsBuffer!: ComputeBuffer;
    private cellOffsetsBuffer!: ComputeBuffer;
    private scatterCountersBuffer!: ComputeBuffer;
    private sortedIndicesBuffer!: ComputeBuffer;
    private blockSumsBuffer!: ComputeBuffer;

    // External buffers (passed in)
    private positionsBuffer!: ComputeBuffer;
    private originalPositionsBuffer!: ComputeBuffer;

    // Camera matrices (for mouse interaction)
    private viewMatrix: IBindable;
    private projectionMatrix: IBindable;
    private inverseViewMatrix: IBindable;
    private worldMatrix: IBindable;

    // Compute passes
    private gridClearCountsPass!: Compute;
    private gridClearScatterPass!: Compute;
    private gridAssignPass!: Compute;
    private prefixSumLocalPass!: Compute;
    private prefixSumTopPass!: Compute;
    private prefixSumDistributePass!: Compute;
    private scatterPass!: Compute;
    private densityPass!: Compute;
    private forcesPass!: Compute;
    private integratePass!: Compute;

    // Body system
    private bodies: FluidBody[] = [];
    private bodyStatesF32!: Float32Array;
    private bodyStatesU32!: Uint32Array;
    private bodyStatesBuffer!: ComputeBuffer;
    private bodyForcesBuffer!: ComputeBuffer;
    private bodyPrimitivesF32!: Float32Array;
    private bodyPrimitivesU32!: Uint32Array;
    private bodyPrimitivesBuffer!: ComputeBuffer;
    private bodyTransformsF32!: Float32Array;
    public bodyTransformsBuffer!: ComputeBuffer;
    private bodyCountU32!: Uint32Array;
    private bodyCountBuffer!: ComputeBuffer;
    private totalPrimitives: number = 0;

    private bodyCollisionPass!: Compute;
    private bodyIntegratePass!: Compute;

    // Grid dimensions
    private gridDims: [number, number, number] = [1, 1, 1];
    private gridOrigin: [number, number, number] = [0, 0, 0];
    private totalCells: number = 1;
    public worldBoundsMin: [number, number, number] = [0, 0, 0];
    public worldBoundsMax: [number, number, number] = [0, 0, 0];

    constructor(renderer: Renderer, options?: Partial<FluidSimulationOptions>) {
        this.renderer = renderer;
        this.params = { ...DEFAULT_OPTIONS, ...options };

        // Default identity matrices (overridden in initialize if camera provided)
        this.viewMatrix = new Matrix4();
        this.projectionMatrix = new Matrix4();
        this.inverseViewMatrix = new Matrix4();
        this.worldMatrix = new Matrix4();
    }

    public initialize(
        positionsBuffer: ComputeBuffer,
        originalPositionsBuffer: ComputeBuffer,
        cameraBindings?: {
            viewMatrix: IBindable;
            projectionMatrix: IBindable;
            inverseViewMatrix: IBindable;
            worldMatrix: IBindable;
        }
    ): void {
        this.positionsBuffer = positionsBuffer;
        this.originalPositionsBuffer = originalPositionsBuffer;
        this.particleCount = this.params.maxParticles;

        if (cameraBindings) {
            this.viewMatrix = cameraBindings.viewMatrix;
            this.projectionMatrix = cameraBindings.projectionMatrix;
            this.inverseViewMatrix = cameraBindings.inverseViewMatrix;
            this.worldMatrix = cameraBindings.worldMatrix;
        }

        this.computeGridFromPositions(positionsBuffer);
        this.createBuffers();
        this.createComputePasses();
        this.createBodyBuffers();
        this.createBodyComputePasses();
    }

    private computeGridFromPositions(positionsBuffer: ComputeBuffer): void {
        // Scan initial positions to determine world bounds
        const data = (positionsBuffer as any).buffer as Float32Array;
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (let i = 0; i < this.particleCount; i++) {
            const x = data[i * 4];
            const y = data[i * 4 + 1];
            const z = data[i * 4 + 2];
            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }

        // Add padding
        const pad = this.params.worldBoundsPadding;
        const rangeX = (maxX - minX) || 1;
        const rangeY = (maxY - minY) || 1;
        const rangeZ = this.params.dimensions === 3 ? ((maxZ - minZ) || 1) : 0.01;
        const padX = rangeX * pad;
        const padY = rangeY * pad;
        const padZ = rangeZ * pad;

        this.worldBoundsMin = [minX - padX, minY - padY, minZ - padZ];
        this.worldBoundsMax = [maxX + padX, maxY + padY, maxZ + padZ];

        const cs = this.params.smoothingRadius;
        const gx = Math.ceil((this.worldBoundsMax[0] - this.worldBoundsMin[0]) / cs);
        const gy = Math.ceil((this.worldBoundsMax[1] - this.worldBoundsMin[1]) / cs);
        const gz = this.params.dimensions === 3
            ? Math.ceil((this.worldBoundsMax[2] - this.worldBoundsMin[2]) / cs)
            : 1;

        // Clamp to max grid cells
        const maxPerAxis2D = Math.floor(Math.sqrt(MAX_GRID_CELLS));
        const maxPerAxis3D = Math.floor(Math.cbrt(MAX_GRID_CELLS));
        const maxPerAxis = this.params.dimensions === 3 ? maxPerAxis3D : maxPerAxis2D;

        this.gridDims = [
            Math.min(Math.max(gx, 1), maxPerAxis),
            Math.min(Math.max(gy, 1), maxPerAxis),
            Math.min(Math.max(gz, 1), this.params.dimensions === 3 ? maxPerAxis : 1),
        ];
        this.totalCells = this.gridDims[0] * this.gridDims[1] * this.gridDims[2];
        this.gridOrigin = [...this.worldBoundsMin];
    }

    private createBuffers(): void {
        const N = this.particleCount;

        // Params uniform
        this.paramsF32 = new Float32Array(PARAMS.BUFFER_SIZE);
        this.paramsU32 = new Uint32Array(this.paramsF32.buffer);
        this.paramsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.paramsF32,
        });

        // Velocities (vec4 per particle — w reserved for future angular vel)
        this.velocitiesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N * 4),
        });

        // Densities (vec2 per particle — density + nearDensity)
        this.densitiesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N * 2),
        });

        // Cell indices (u32 per particle)
        this.cellIndicesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N),
        });

        // Cell counts (u32 per cell)
        this.cellCountsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Cell offsets (prefix sum output)
        this.cellOffsetsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Scatter counters
        this.scatterCountersBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });

        // Sorted indices (u32 per particle)
        this.sortedIndicesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(N),
        });

        // Block sums for prefix sum
        const numBlocks = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);
        this.blockSumsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(Math.max(numBlocks, 1)),
        });
    }

    private createBodyBuffers(): void {
        this.bodyStatesF32 = new Float32Array(MAX_BODIES * BODY_STATE_FLOATS);
        this.bodyStatesU32 = new Uint32Array(this.bodyStatesF32.buffer);
        this.bodyStatesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyStatesF32,
        });

        this.bodyForcesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(MAX_BODIES * 4),
        });

        this.bodyPrimitivesF32 = new Float32Array(MAX_PRIMITIVES * PRIMITIVE_FLOATS);
        this.bodyPrimitivesU32 = new Uint32Array(this.bodyPrimitivesF32.buffer);
        this.bodyPrimitivesBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyPrimitivesF32,
        });

        this.bodyTransformsF32 = new Float32Array(MAX_BODIES * 4);
        this.bodyTransformsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_VERTEX | BufferBase.BUFFER_USAGE_COPY_SRC | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyTransformsF32,
            shaderLocation: 3,
            offset: 0,
            stride: 4 * 4,
            format: 'float32x4' as GPUVertexFormat,
        });

        this.bodyCountU32 = new Uint32Array([0]);
        this.bodyCountBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this.bodyCountU32,
        });
    }

    private createComputePasses(): void {
        const C = GPUShaderStage.COMPUTE;

        this.gridClearCountsPass = new Compute(gridClearShader, [
            { binding: 0, visibility: C, value: this.cellCountsBuffer },
        ]);

        this.gridClearScatterPass = new Compute(gridClearShader, [
            { binding: 0, visibility: C, value: this.scatterCountersBuffer },
        ]);

        this.gridAssignPass = new Compute(gridAssignShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.cellIndicesBuffer },
            { binding: 2, visibility: C, value: this.cellCountsBuffer },
            { binding: 3, visibility: C, value: this.paramsBuffer },
        ]);

        this.prefixSumLocalPass = new Compute(prefixSumLocalShader, [
            { binding: 0, visibility: C, value: this.cellCountsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.blockSumsBuffer },
        ]);

        this.prefixSumTopPass = new Compute(prefixSumTopShader, [
            { binding: 0, visibility: C, value: this.blockSumsBuffer },
        ]);

        this.prefixSumDistributePass = new Compute(prefixSumDistributeShader, [
            { binding: 0, visibility: C, value: this.blockSumsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
        ]);

        this.scatterPass = new Compute(scatterShader, [
            { binding: 0, visibility: C, value: this.cellIndicesBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.scatterCountersBuffer },
            { binding: 3, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 4, visibility: C, value: this.paramsBuffer },
        ]);

        this.densityPass = new Compute(densityShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 2, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 3, visibility: C, value: this.densitiesBuffer },
            { binding: 4, visibility: C, value: this.paramsBuffer },
        ]);

        this.forcesPass = new Compute(forcesShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.densitiesBuffer },
            { binding: 3, visibility: C, value: this.originalPositionsBuffer },
            { binding: 4, visibility: C, value: this.cellOffsetsBuffer },
            { binding: 5, visibility: C, value: this.sortedIndicesBuffer },
            { binding: 6, visibility: C, value: this.paramsBuffer },
            { binding: 7, visibility: C, value: this.viewMatrix },
            { binding: 8, visibility: C, value: this.projectionMatrix },
            { binding: 9, visibility: C, value: this.inverseViewMatrix },
            { binding: 10, visibility: C, value: this.worldMatrix },
        ]);

        this.integratePass = new Compute(integrateShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.paramsBuffer },
        ]);
    }

    private createBodyComputePasses(): void {
        const C = GPUShaderStage.COMPUTE;

        this.bodyCollisionPass = new Compute(bodyCollisionShader, [
            { binding: 0, visibility: C, value: this.positionsBuffer },
            { binding: 1, visibility: C, value: this.velocitiesBuffer },
            { binding: 2, visibility: C, value: this.bodyStatesBuffer },
            { binding: 3, visibility: C, value: this.bodyPrimitivesBuffer },
            { binding: 4, visibility: C, value: this.bodyForcesBuffer },
            { binding: 5, visibility: C, value: this.paramsBuffer },
            { binding: 6, visibility: C, value: this.bodyCountBuffer },
        ]);

        this.bodyIntegratePass = new Compute(bodyIntegrateShader, [
            { binding: 0, visibility: C, value: this.bodyStatesBuffer },
            { binding: 1, visibility: C, value: this.bodyForcesBuffer },
            { binding: 2, visibility: C, value: this.bodyTransformsBuffer },
            { binding: 3, visibility: C, value: this.paramsBuffer },
            { binding: 4, visibility: C, value: this.bodyCountBuffer },
            { binding: 5, visibility: C, value: this.viewMatrix },
            { binding: 6, visibility: C, value: this.projectionMatrix },
            { binding: 7, visibility: C, value: this.inverseViewMatrix },
            { binding: 8, visibility: C, value: this.worldMatrix },
        ]);
    }

    private packParams(dt: number, mouseStrength: number, mousePosition?: { x: number, y: number }, mouseDirection?: { x: number, y: number }): void {
        const p = this.params;
        const f = this.paramsF32;
        const u = this.paramsU32;

        f[PARAMS.dt] = dt / p.substeps;
        u[PARAMS.particleCount] = this.particleCount;
        u[PARAMS.dimensions] = p.dimensions;
        f[PARAMS.smoothingRadius] = p.smoothingRadius;
        f[PARAMS.pressureMultiplier] = p.pressureMultiplier;
        f[PARAMS.densityTarget] = p.densityTarget;
        f[PARAMS.nearPressureMultiplier] = p.nearPressureMultiplier;
        f[PARAMS.viscosity] = p.viscosity;
        f[PARAMS.damping] = p.damping;
        f[PARAMS.returnToOriginStrength] = p.returnToOriginStrength;
        f[PARAMS.mouseStrength] = mouseStrength;
        f[PARAMS.mouseRadius] = p.mouseRadius;
        f[PARAMS.gravityX] = p.gravity[0];
        f[PARAMS.gravityY] = p.gravity[1];
        f[PARAMS.gravityZ] = p.gravity[2];
        f[PARAMS.mouseForce] = p.mouseForce;
        f[PARAMS.mousePosX] = mousePosition?.x ?? 0;
        f[PARAMS.mousePosY] = mousePosition?.y ?? 0;
        f[PARAMS.mouseDirX] = mouseDirection?.x ?? 0;
        f[PARAMS.mouseDirY] = mouseDirection?.y ?? 0;
        u[PARAMS.gridDimsX] = this.gridDims[0];
        u[PARAMS.gridDimsY] = this.gridDims[1];
        u[PARAMS.gridDimsZ] = this.gridDims[2];
        f[PARAMS.cellSize] = p.smoothingRadius;
        f[PARAMS.gridOriginX] = this.gridOrigin[0];
        f[PARAMS.gridOriginY] = this.gridOrigin[1];
        f[PARAMS.gridOriginZ] = this.gridOrigin[2];
        u[PARAMS.totalCells] = this.totalCells;
        f[PARAMS.worldBoundsMinX] = this.worldBoundsMin[0];
        f[PARAMS.worldBoundsMinY] = this.worldBoundsMin[1];
        f[PARAMS.worldBoundsMinZ] = this.worldBoundsMin[2];
        f[PARAMS.worldBoundsMaxX] = this.worldBoundsMax[0];
        f[PARAMS.worldBoundsMaxY] = this.worldBoundsMax[1];
        f[PARAMS.worldBoundsMaxZ] = this.worldBoundsMax[2];

        // Kernel factors
        const kernels = p.dimensions === 3
            ? computeKernelFactors3D(p.smoothingRadius)
            : computeKernelFactors2D(p.smoothingRadius);
        f[PARAMS.poly6Factor] = kernels.poly6;
        f[PARAMS.spikyPow2Factor] = kernels.spikyPow2;
        f[PARAMS.spikyPow3Factor] = kernels.spikyPow3;
        f[PARAMS.spikyPow2DerivFactor] = kernels.spikyPow2Deriv;
        f[PARAMS.spikyPow3DerivFactor] = kernels.spikyPow3Deriv;

        // Radial gravity
        const gc = p.gravityCenter ?? [0, 0, 0];
        f[PARAMS.gravityCenterX] = gc[0];
        f[PARAMS.gravityCenterY] = gc[1];
        f[PARAMS.gravityCenterZ] = gc[2];
        f[PARAMS.radialGravity] = p.radialGravity ? 1.0 : 0.0;

        this.paramsBuffer.needsUpdate = true;
    }

    public setPreset(presetName: string): void {
        const preset = PRESETS[presetName];
        if (!preset) { return; }
        Object.assign(this.params, preset);
    }

    public setParams(overrides: Partial<FluidSimulationOptions>): void {
        Object.assign(this.params, overrides);
    }

    public get bodyCount(): number {
        return this.bodies.length;
    }

    public addBody(options: FluidBodyOptions): FluidBody {
        if (this.bodies.length >= MAX_BODIES) {
            throw new Error(`Max body count (${MAX_BODIES}) reached`);
        }
        if (this.totalPrimitives + options.primitives.length > MAX_PRIMITIVES) {
            throw new Error(`Max primitive count (${MAX_PRIMITIVES}) reached`);
        }

        const body = new FluidBody(options, this.bodies.length, this.totalPrimitives);
        this.bodies.push(body);

        // Pack primitives
        for (let i = 0; i < body.primitiveCount; i++) {
            const offset = (this.totalPrimitives + i) * PRIMITIVE_FLOATS;
            FluidBody.packPrimitive(body.primitives[i], this.bodyPrimitivesF32, this.bodyPrimitivesU32, offset);
        }
        this.totalPrimitives += body.primitiveCount;
        this.bodyPrimitivesBuffer.needsUpdate = true;

        // Pack body state
        this.syncBodyState(body);

        // Initialize transform for rendering
        const tOff = body.index * 4;
        this.bodyTransformsF32[tOff] = body.position.x;
        this.bodyTransformsF32[tOff + 1] = body.position.y;
        this.bodyTransformsF32[tOff + 2] = body.position.z;
        this.bodyTransformsF32[tOff + 3] = body.angle;
        this.bodyTransformsBuffer.needsUpdate = true;

        // Update count
        this.bodyCountU32[0] = this.bodies.length;
        this.bodyCountBuffer.needsUpdate = true;

        return body;
    }

    public removeBody(body: FluidBody): void {
        const idx = this.bodies.indexOf(body);
        if (idx === -1) return;

        const last = this.bodies[this.bodies.length - 1];
        if (idx !== this.bodies.length - 1) {
            const srcOffset = last.index * BODY_STATE_FLOATS;
            const dstOffset = idx * BODY_STATE_FLOATS;
            this.bodyStatesF32.copyWithin(dstOffset, srcOffset, srcOffset + BODY_STATE_FLOATS);
            (last as any).index = idx;
        }

        this.bodies.splice(idx, 1);
        this.bodyCountU32[0] = this.bodies.length;
        this.bodyCountBuffer.needsUpdate = true;
        this.bodyStatesBuffer.needsUpdate = true;
    }

    private syncBodyState(body: FluidBody): void {
        const offset = body.index * BODY_STATE_FLOATS;
        body.packState(this.bodyStatesF32, this.bodyStatesU32, offset);
        this.bodyStatesBuffer.needsUpdate = true;
    }

    public syncBodyParams(): void {
        for (const body of this.bodies) {
            this.syncBodyState(body);
        }
    }

    public syncBodyPrimitives(body: FluidBody): void {
        for (let i = 0; i < body.primitiveCount; i++) {
            const offset = (body.primitiveStart + i) * PRIMITIVE_FLOATS;
            FluidBody.packPrimitive(body.primitives[i], this.bodyPrimitivesF32, this.bodyPrimitivesU32, offset);
        }
        this.bodyPrimitivesBuffer.needsUpdate = true;
    }

    /**
     * Sync only configurable body parameters (mass, damping, etc.)
     * without overwriting GPU-owned dynamic state (position, velocity, angle).
     */
    public syncBodyConfig(): void {
        for (const body of this.bodies) {
            const o = body.index * BODY_STATE_FLOATS;
            this.bodyStatesF32[o + 10] = body.mass;
            this.bodyStatesF32[o + 11] = body.computeInertia();
            this.bodyStatesF32[o + 12] = body.restitution;
            this.bodyStatesF32[o + 15] = body.reactionMultiplier;
            this.bodyStatesF32[o + 16] = body.maxPushDist;
            this.bodyStatesF32[o + 17] = body.forceClampFactor;
            this.bodyStatesF32[o + 18] = body.rightingStrength;
            this.bodyStatesF32[o + 19] = body.linearDamping;
            this.bodyStatesF32[o + 20] = body.angularDamping;
            this.bodyStatesF32[o + 21] = body.density;
            this.bodyStatesF32[o + 22] = body.mouseScale;
        }
        // Partial write: only the config region (skip first 10 floats = 40 bytes of dynamic state per body)
        const device = this.renderer.device;
        const gpuBuffer = (this.bodyStatesBuffer as any)._resource as GPUBuffer;
        if (device && gpuBuffer) {
            for (const body of this.bodies) {
                const byteOffset = (body.index * BODY_STATE_FLOATS + 10) * 4;
                const byteLength = (BODY_STATE_FLOATS - 10) * 4;
                device.queue.writeBuffer(gpuBuffer, byteOffset,
                    this.bodyStatesF32.buffer, this.bodyStatesF32.byteOffset + byteOffset, byteLength);
            }
        }
    }

    /**
     * Rebuild the spatial grid from current worldBoundsMin/Max.
     * Call after changing bounds at runtime.
     */
    public rebuildGrid(): void {
        const cs = this.params.smoothingRadius;
        const gx = Math.ceil((this.worldBoundsMax[0] - this.worldBoundsMin[0]) / cs);
        const gy = Math.ceil((this.worldBoundsMax[1] - this.worldBoundsMin[1]) / cs);
        const gz = this.params.dimensions === 3
            ? Math.ceil((this.worldBoundsMax[2] - this.worldBoundsMin[2]) / cs)
            : 1;

        const maxPerAxis2D = Math.floor(Math.sqrt(MAX_GRID_CELLS));
        const maxPerAxis3D = Math.floor(Math.cbrt(MAX_GRID_CELLS));
        const maxPerAxis = this.params.dimensions === 3 ? maxPerAxis3D : maxPerAxis2D;

        this.gridDims = [
            Math.min(Math.max(gx, 1), maxPerAxis),
            Math.min(Math.max(gy, 1), maxPerAxis),
            Math.min(Math.max(gz, 1), this.params.dimensions === 3 ? maxPerAxis : 1),
        ];
        this.totalCells = this.gridDims[0] * this.gridDims[1] * this.gridDims[2];
        this.gridOrigin = [...this.worldBoundsMin];

        // Reallocate grid-sized buffers
        this.cellCountsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });
        this.cellOffsetsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });
        this.scatterCountersBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(this.totalCells),
        });
        const numBlocks = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);
        this.blockSumsBuffer = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_STORAGE,
            usage: BufferBase.BUFFER_USAGE_STORAGE,
            buffer: new Float32Array(Math.max(numBlocks, 1)),
        });

        // Recreate compute passes with new buffers
        this.createComputePasses();
        if (this.bodies.length > 0) {
            this.createBodyComputePasses();
        }
    }

    public get positionsBufferRef(): ComputeBuffer {
        return this.positionsBuffer;
    }

    public get paramsBufferRef(): ComputeBuffer {
        return this.paramsBuffer;
    }

    public get cellOffsetsBufferRef(): ComputeBuffer {
        return this.cellOffsetsBuffer;
    }

    public get sortedIndicesBufferRef(): ComputeBuffer {
        return this.sortedIndicesBuffer;
    }

    public get gridDimsRef(): [number, number, number] {
        return this.gridDims;
    }

    public get gridOriginRef(): [number, number, number] {
        return this.gridOrigin;
    }

    /** Build the list of compute passes that make up one full SPH iteration.
     *  Shared by `update()` and `updateBatched()`. */
    private _buildSimPasses(): { compute: Compute, workgroupsX: number, workgroupsY?: number, workgroupsZ?: number }[] {
        const N = this.particleCount;
        const particleWorkgroups  = Math.ceil(N / 64);
        const gridWorkgroups      = Math.ceil(this.totalCells / 256);
        const prefixSumWorkgroups = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);
        const passes: { compute: Compute, workgroupsX: number, workgroupsY?: number, workgroupsZ?: number }[] = [
            // Grid build
            { compute: this.gridClearCountsPass,    workgroupsX: gridWorkgroups },
            { compute: this.gridClearScatterPass,   workgroupsX: gridWorkgroups },
            { compute: this.gridAssignPass,         workgroupsX: particleWorkgroups },
            // Prefix sum
            { compute: this.prefixSumLocalPass,      workgroupsX: Math.max(prefixSumWorkgroups, 1) },
            { compute: this.prefixSumTopPass,        workgroupsX: 1 },
            { compute: this.prefixSumDistributePass, workgroupsX: Math.max(prefixSumWorkgroups, 1) },
            // Scatter
            { compute: this.scatterPass,             workgroupsX: particleWorkgroups },
            // SPH
            { compute: this.densityPass,             workgroupsX: particleWorkgroups },
            { compute: this.forcesPass,              workgroupsX: particleWorkgroups },
            { compute: this.integratePass,           workgroupsX: particleWorkgroups },
        ];
        // Body passes (only if bodies exist)
        if (this.bodies.length > 0) {
            passes.push(
                { compute: this.bodyCollisionPass,   workgroupsX: particleWorkgroups },
                { compute: this.bodyIntegratePass,    workgroupsX: 1 },
            );
        }
        return passes;
    }

    /**
     * Standard sim update. Runs `params.substeps` iterations of the SPH
     * pipeline, each with `dt / substeps` as the integration timestep.
     * Each iteration is a separate submit + `onSubmittedWorkDone` sync — fine
     * for single-step-per-frame callers, not ideal for fixed-timestep loops.
     * For a framerate-independent loop prefer `updateBatched()`.
     */
    public async update(
        dt: number,
        mousePosition?: { x: number; y: number },
        mouseDirection?: { x: number; y: number },
        mouseStrength: number = 0
    ): Promise<void> {
        const passes = this._buildSimPasses();
        for (let s = 0; s < this.params.substeps; s++) {
            this.packParams(dt, mouseStrength, mousePosition, mouseDirection);
            await this.renderer.computeBatch(passes);
        }
    }

    /**
     * Framerate-independent update — runs `steps * substeps` iterations of the
     * SPH pipeline inside a **single command encoder** with one queue submit
     * and zero CPU-GPU sync. Each iteration integrates over a fixed `stepDt`
     * (the *total* advance passed to one call of `update()`), so the simulation
     * evolves deterministically regardless of how many steps are batched.
     *
     * Intended use: drive from an accumulator in the render loop so the sim
     * advances at a constant real-time rate even when the renderer drops
     * frames. Because there's no sync, batching 2 steps costs ~2× the GPU
     * work but almost zero extra CPU time.
     *
     * @param stepDt         Total advance per step (seconds). Equivalent to
     *                       what you would pass to `update()`. Packed once
     *                       and reused for all batched steps.
     * @param steps          Number of sim steps to run this frame.
     * @param mousePosition  Screen-space NDC position (or undefined).
     * @param mouseDirection Screen-space NDC direction (or undefined).
     * @param mouseStrength  Scalar impulse magnitude.
     */
    public updateBatched(
        stepDt: number,
        steps: number,
        mousePosition?: { x: number; y: number },
        mouseDirection?: { x: number; y: number },
        mouseStrength: number = 0
    ): void {
        if (steps <= 0) return;
        this.packParams(stepDt, mouseStrength, mousePosition, mouseDirection);

        const passes = this._buildSimPasses();
        const device = this.renderer.gpuDevice;
        const commandEncoder = this.renderer.createCommandEncoder('FluidSim/Batched');

        // Initialise compute passes on first use so we can skip the check
        // inside the hot inner loop.
        for (const p of passes) {
            if (!p.compute.initialized) p.compute.initialize(device);
        }

        // Each iteration of the outer loop is one `update()`-equivalent:
        // `substeps` inner SPH iterations, all integrating with the same dt.
        const totalIters = steps * this.params.substeps;
        for (let i = 0; i < totalIters; i++) {
            for (const pass of passes) {
                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setBindGroup(0, pass.compute.getBindGroup(device));
                passEncoder.setPipeline(pass.compute.pipeline!);
                passEncoder.dispatchWorkgroups(pass.workgroupsX, pass.workgroupsY ?? 1, pass.workgroupsZ ?? 1);
                passEncoder.end();
            }
        }

        // Single submit, no CPU-GPU sync. Pacing is left to the browser —
        // `requestAnimationFrame` naturally throttles us to the display's
        // vsync cadence, and the subsequent render commands implicitly
        // serialise behind these compute passes via WebGPU's queue order.
        this.renderer.submit(commandEncoder.finish());
    }
}

export { FluidSimulation };
