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

const MAX_GRID_CELLS = 262144; // 256K
const PREFIX_SUM_BLOCK_SIZE = 512;

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

    // Grid dimensions
    private gridDims: [number, number, number] = [1, 1, 1];
    private gridOrigin: [number, number, number] = [0, 0, 0];
    private totalCells: number = 1;
    private worldBoundsMin: [number, number, number] = [0, 0, 0];
    private worldBoundsMax: [number, number, number] = [0, 0, 0];

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

        this.paramsBuffer.needsUpdate = true;
    }

    public setPreset(presetName: string): void {
        const preset = PRESETS[presetName];
        if (!preset) { return; }
        Object.assign(this.params, preset);
    }

    public async update(
        dt: number,
        mousePosition?: { x: number; y: number },
        mouseDirection?: { x: number; y: number },
        mouseStrength: number = 0
    ): Promise<void> {
        const N = this.particleCount;
        const particleWorkgroups = Math.ceil(N / 64);
        const gridWorkgroups = Math.ceil(this.totalCells / 256);
        const prefixSumWorkgroups = Math.ceil(this.totalCells / PREFIX_SUM_BLOCK_SIZE);

        for (let s = 0; s < this.params.substeps; s++) {
            this.packParams(dt, mouseStrength, mousePosition, mouseDirection);

            await this.renderer.computeBatch([
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
            ]);
        }
    }
}

export { FluidSimulation };
