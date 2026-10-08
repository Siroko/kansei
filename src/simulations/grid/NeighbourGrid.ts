import { Compute } from '../../materials/Compute';
import { ComputeBuffer } from '../../buffers/ComputeBuffer';
import { BufferBase } from '../../buffers/BufferBase';
import { shaderCode as clearShader } from './shaders/grid-clear.wgsl';
import { shaderCode as assignShader } from './shaders/grid-assign.wgsl';
import { shaderCode as prefixSumLocalShader } from './shaders/prefix-sum-local.wgsl';
import { shaderCode as prefixSumTopShader } from './shaders/prefix-sum-top.wgsl';
import { shaderCode as prefixSumDistributeShader } from './shaders/prefix-sum-distribute.wgsl';
import { scatterShader } from './shaders/scatter.wgsl';

export { neighbourGridWgsl } from './shaders/neighbour-grid.wgsl';

const PREFIX_SUM_BLOCK_SIZE = 512;
/** WebGPU's default limit of storage buffers per shader stage leaves the scatter room for two. */
const MAX_SORTED_COPIES = 2;

/** Where a grid's cells lie: `dims` cells `cellSize` wide from `origin`. */
export interface GridLayout {
    origin: [number, number, number];
    cellSize: number;
    dims: [number, number, number];
}

/**
 * Cells `cellSize` wide over the box from `min` to `max` (at least one per axis; an axis with
 * no extent gets one), widened in steps of 25% until there are at most `maxCells`. A search of
 * the cells around a point reaches every neighbour within `cellSize`: pass the search radius.
 * Clamping the counts per axis instead would fold every point beyond them into the edge cells.
 */
export function gridLayoutCovering(
    min: [number, number, number],
    max: [number, number, number],
    cellSize: number,
    maxCells: number,
): GridLayout {
    if (!(cellSize > 0)) throw new Error('grid cells need a width');
    let cell = cellSize;
    for (;;) {
        const dims = [0, 1, 2].map(d => Math.max(Math.ceil((max[d] - min[d]) / cell), 1)) as [number, number, number];
        if (dims[0] * dims[1] * dims[2] <= Math.max(maxCells, 1)) {
            return { origin: [...min], cellSize: cell, dims };
        }
        cell *= 1.25;
    }
}

export function gridLayoutTotalCells(layout: GridLayout): number {
    return layout.dims[0] * layout.dims[1] * layout.dims[2];
}

export interface NeighbourGridOptions {
    /** The most points it holds. */
    capacity: number;
    layout: GridLayout;
    /** The points: `array<vec4<f32>>` with the position in `xyz`, at least `capacity` long. */
    positions: ComputeBuffer;
    /**
     * Per-point `array<vec4<f32>>` buffers (at most two) the grid also copies in cell order each
     * step, into `sorted(k)`: e.g. the positions and velocities, so a neighbour search reads them
     * contiguously.
     */
    sortedCopies?: ComputeBuffer[];
}

function storageBuffer(data: Float32Array | Uint32Array): ComputeBuffer {
    return new ComputeBuffer({
        type: BufferBase.BUFFER_TYPE_STORAGE,
        usage: BufferBase.BUFFER_USAGE_STORAGE | BufferBase.BUFFER_USAGE_COPY_SRC,
        buffer: data,
    });
}

function destroy(buffer: ComputeBuffer): void {
    if (buffer.initialized) buffer.resource.buffer.destroy();
}

/**
 * A counting-sort grid of points on the GPU, rebuilt by `encode` each step (the TypeScript side
 * of `rust/kansei-core/src/simulations/grid`): each cell's point count (`cellCounts`), where its
 * points start in the sorted order (`cellOffsets`), the sorted order itself (`sortedIndices`,
 * slot to point) and cell-ordered copies of chosen buffers (`sorted(k)`). A pass binds those
 * with `paramsBuffer` and `neighbourGridWgsl` to visit the points near one. Points outside the
 * grid go into its edge cells.
 *
 * The grid sorts the first `count` points. `setCount` and `setLayout` mark the uniform for
 * upload, which happens the next time a pass of the grid is encoded.
 */
export class NeighbourGrid {
    private _layout: GridLayout;
    private _capacity: number;
    private _count = 0;

    private _paramsF32 = new Float32Array(8);
    private _paramsU32 = new Uint32Array(this._paramsF32.buffer);
    private _params: ComputeBuffer;
    private _positions: ComputeBuffer;
    /** Each copied buffer and its cell-ordered copy. */
    private _copies: [ComputeBuffer, ComputeBuffer][];
    private _cellIndices: ComputeBuffer;
    private _sortedIndices: ComputeBuffer;
    // sized by the layout's cells: replaced when their number changes
    private _cellCounts!: ComputeBuffer;
    private _cellOffsets!: ComputeBuffer;
    private _scatterCounters!: ComputeBuffer;
    private _blockSums!: ComputeBuffer;

    private _passes: Compute[] = [];

    constructor(options: NeighbourGridOptions) {
        const copies = options.sortedCopies ?? [];
        if (copies.length > MAX_SORTED_COPIES) {
            throw new Error(`a NeighbourGrid copies at most ${MAX_SORTED_COPIES} buffers`);
        }
        const n = Math.max(options.capacity, 1);
        this._capacity = options.capacity;
        this._layout = options.layout;
        this._positions = options.positions;
        this._copies = copies.map(source => [source, storageBuffer(new Float32Array(n * 4))]);
        this._cellIndices = storageBuffer(new Uint32Array(n));
        this._sortedIndices = storageBuffer(new Uint32Array(n));
        this._params = new ComputeBuffer({
            type: BufferBase.BUFFER_TYPE_UNIFORM,
            usage: BufferBase.BUFFER_USAGE_UNIFORM | BufferBase.BUFFER_USAGE_COPY_DST,
            buffer: this._paramsF32,
        });
        this._makeCellBuffers();
        this._makePasses();
        this._writeParams();
    }

    private _makeCellBuffers(): void {
        const cells = gridLayoutTotalCells(this._layout);
        const blocks = Math.ceil(cells / PREFIX_SUM_BLOCK_SIZE);
        this._cellCounts = storageBuffer(new Uint32Array(Math.max(cells, 1)));
        this._cellOffsets = storageBuffer(new Uint32Array(Math.max(cells, 1)));
        this._scatterCounters = storageBuffer(new Uint32Array(Math.max(cells, 1)));
        this._blockSums = storageBuffer(new Uint32Array(Math.max(blocks, 1)));
    }

    private _makePasses(): void {
        const C = GPUShaderStage.COMPUTE;
        const make = (code: string, buffers: ComputeBuffer[]) =>
            new Compute(code, buffers.map((value, binding) => ({ binding, visibility: C, value })));
        const scatter = [this._cellIndices, this._cellOffsets, this._scatterCounters, this._sortedIndices, this._params];
        for (const [source, sorted] of this._copies) scatter.push(source, sorted);
        this._passes = [
            make(clearShader, [this._cellCounts, this._scatterCounters]),
            make(assignShader, [this._positions, this._cellIndices, this._cellCounts, this._params]),
            // an exclusive scan of the counts into the offsets (the counts stay as they were)
            make(prefixSumLocalShader, [this._cellCounts, this._cellOffsets, this._blockSums]),
            make(prefixSumTopShader, [this._blockSums]),
            make(prefixSumDistributeShader, [this._blockSums, this._cellOffsets]),
            make(scatterShader(this._copies.length), scatter),
        ];
    }

    private _writeParams(): void {
        const l = this._layout;
        this._paramsF32.set(l.origin, 0);
        this._paramsF32[3] = l.cellSize;
        this._paramsU32.set(l.dims, 4);
        this._paramsU32[7] = this._count;
        this._params.needsUpdate = true;
    }

    /** Sort the first `count` points (at most the capacity) from the next step on. */
    public setCount(count: number): void {
        if (count > this._capacity) {
            throw new Error(`${count} points in a grid for ${this._capacity}`);
        }
        if (count !== this._count) {
            this._count = count;
            this._writeParams();
        }
    }

    /**
     * Move or resize the cells. Returns whether that replaced the per-cell buffers (`cellCounts`,
     * `cellOffsets`): it does when the number of cells changes, and passes bound to them need new
     * bind groups.
     */
    public setLayout(layout: GridLayout): boolean {
        const old = this._layout;
        const same = old.cellSize === layout.cellSize
            && old.origin.every((v, d) => v === layout.origin[d])
            && old.dims.every((v, d) => v === layout.dims[d]);
        if (same) return false;
        const replaced = gridLayoutTotalCells(layout) !== gridLayoutTotalCells(old);
        this._layout = { origin: [...layout.origin], cellSize: layout.cellSize, dims: [...layout.dims] };
        if (replaced) {
            for (const b of [this._cellCounts, this._cellOffsets, this._scatterCounters, this._blockSums]) destroy(b);
            this._makeCellBuffers();
            this._makePasses();
        }
        this._writeParams();
        return replaced;
    }

    /** Rebuild the grid from the points' current positions: six dispatches in `pass`. */
    public encode(pass: GPUComputePassEncoder, device: GPUDevice): void {
        const cells = gridLayoutTotalCells(this._layout);
        const points = Math.ceil(this._count / 64);
        const blocks = Math.max(Math.ceil(cells / PREFIX_SUM_BLOCK_SIZE), 1);
        const workgroups = [Math.ceil(cells / 256), points, blocks, 1, blocks, points];
        this._passes.forEach((compute, k) => {
            if (!compute.initialized) compute.initialize(device);
            pass.setPipeline(compute.pipeline!);
            pass.setBindGroup(0, compute.getBindGroup(device));
            pass.dispatchWorkgroups(workgroups[k]);
        });
    }

    public get layout(): GridLayout { return this._layout; }
    /** The points sorted: the first `count` of the positions buffer. */
    public get count(): number { return this._count; }
    public get capacity(): number { return this._capacity; }
    /** The `NeighbourGrid` uniform of `neighbourGridWgsl`: the layout and the count. */
    public get paramsBuffer(): ComputeBuffer { return this._params; }
    /** Each cell's point count (`array<u32>`). */
    public get cellCounts(): ComputeBuffer { return this._cellCounts; }
    /** Each cell's first sorted slot (`array<u32>`). */
    public get cellOffsets(): ComputeBuffer { return this._cellOffsets; }
    /** Each sorted slot's point (`array<u32>`). */
    public get sortedIndices(): ComputeBuffer { return this._sortedIndices; }
    /** The cell-ordered copy of `sortedCopies[k]` (`array<vec4<f32>>`). */
    public sorted(k: number): ComputeBuffer { return this._copies[k][1]; }
}
