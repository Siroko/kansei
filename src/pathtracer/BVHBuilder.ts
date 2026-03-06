import { Geometry } from "../buffers/Geometry";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { PathTracerMaterial } from "./PathTracerMaterial";
import { mortonShader } from "./shaders/morton.wgsl";
import { radixSortShader } from "./shaders/radix-sort.wgsl";
import { treeBuildShader } from "./shaders/tree-build.wgsl";
import { refitShader } from "./shaders/refit.wgsl";

export interface BLASEntry {
    geometryId: string;
    triangleOffset: number;
    triangleCount: number;
    nodeOffset: number;
    nodeCount: number;
}

export class BVHBuilder {
    private _device: GPUDevice;

    // Triangle stride: 24 floats = 96 bytes per triangle
    // v0.xyz(3) v1.xyz(3) v2.xyz(3) n0.xyz(3) n1.xyz(3) n2.xyz(3) matIdx(1) pad(5)
    static readonly TRI_STRIDE_FLOATS = 24;

    // BLAS data
    private _triangleBuffer: GPUBuffer | null = null;
    private _blasNodeBuffer: GPUBuffer | null = null;
    private _blasEntries: Map<string, BLASEntry> = new Map();
    public totalTriangleCount: number = 0;
    public totalBLASNodes: number = 0;

    // TLAS data
    private _instanceBuffer: GPUBuffer | null = null;
    private _tlasNodeBuffer: GPUBuffer | null = null;
    private _totalInstances: number = 0;

    // Material data
    private _materialBuffer: GPUBuffer | null = null;
    public materialCount: number = 0;

    // Sort scratch buffers
    private _mortonBuffer: GPUBuffer | null = null;
    private _mortonIndicesBuffer: GPUBuffer | null = null;
    private _sortScratchBuffer: GPUBuffer | null = null;

    // Scratch buffers pending deferred cleanup (destroyed on next buildBLASTree call)
    private _pendingScratchBuffers: GPUBuffer[] = [];

    // GPU BVH construction pipelines
    private _mortonPipeline: GPUComputePipeline | null = null;
    private _mortonBGL: GPUBindGroupLayout | null = null;
    private _sortHistogramPipeline: GPUComputePipeline | null = null;
    private _sortPrefixPipeline: GPUComputePipeline | null = null;
    private _sortScatterPipeline: GPUComputePipeline | null = null;
    private _sortBGL: GPUBindGroupLayout | null = null;
    private _treeBuildPipeline: GPUComputePipeline | null = null;
    private _treeBuildBGL: GPUBindGroupLayout | null = null;
    private _refitPipeline: GPUComputePipeline | null = null;
    private _refitBGL: GPUBindGroupLayout | null = null;
    private _mortonParamsBuf: GPUBuffer | null = null;
    private _sortParamsBuf: GPUBuffer | null = null;
    private _treeBuildParamsBuf: GPUBuffer | null = null;
    private _refitParamsBuf: GPUBuffer | null = null;

    // Per-geometry centroid cache (vec4f per triangle: xyz = centroid, w = unused)
    private _centroidsMap: Map<string, Float32Array> = new Map();

    // Scene bounds computed from centroid data
    private _sceneMin: [number, number, number] = [0, 0, 0];
    private _sceneMax: [number, number, number] = [0, 0, 0];

    constructor(device: GPUDevice) {
        this._device = device;
    }

    // Public accessors for trace shader bind groups
    get triangleBuffer(): GPUBuffer | null { return this._triangleBuffer; }
    get blasNodeBuffer(): GPUBuffer | null { return this._blasNodeBuffer; }
    get instanceBuffer(): GPUBuffer | null { return this._instanceBuffer; }
    get tlasNodeBuffer(): GPUBuffer | null { return this._tlasNodeBuffer; }
    get materialBuffer(): GPUBuffer | null { return this._materialBuffer; }
    get totalInstances(): number { return this._totalInstances; }

    /**
     * Extract all unique geometries and build BLAS for each.
     * Called once on scene setup, or when geometry changes.
     */
    public buildBLAS(scene: Scene): void {
        const objects = scene.getOrderedObjects();
        const geometryMap = new Map<object, { triangles: Float32Array; triCount: number }>();
        const materials: PathTracerMaterial[] = [];
        const materialIndexMap = new Map<Renderable, number>();

        // Collect unique geometries and assign material indices
        for (const obj of objects) {
            const geom = obj.geometry.isInstancedGeometry
                ? (obj.geometry as InstancedGeometry).geometry ?? obj.geometry
                : obj.geometry;

            if (!geometryMap.has(geom)) {
                const tris = this._extractTriangles(geom, 0);
                geometryMap.set(geom, tris);
            }

            if (!materialIndexMap.has(obj)) {
                const ptMat = obj.pathTracerMaterial ?? new PathTracerMaterial();
                materialIndexMap.set(obj, materials.length);
                materials.push(ptMat);
            }
        }

        // Pack all triangles into one contiguous buffer
        let totalTris = 0;
        for (const [, data] of geometryMap) totalTris += data.triCount;

        const allTriangles = new Float32Array(totalTris * BVHBuilder.TRI_STRIDE_FLOATS);
        let offset = 0;
        this._blasEntries.clear();
        this._centroidsMap.clear();

        for (const [geom, data] of geometryMap) {
            const entry: BLASEntry = {
                geometryId: (geom as any).uuid ?? String(offset),
                triangleOffset: offset / BVHBuilder.TRI_STRIDE_FLOATS,
                triangleCount: data.triCount,
                nodeOffset: 0,
                nodeCount: 0,
            };
            allTriangles.set(data.triangles, offset);

            // Compute and cache centroids for this geometry
            const centroids = this._computeCentroids(data.triangles, data.triCount);
            this._centroidsMap.set(entry.geometryId, centroids);

            offset += data.triCount * BVHBuilder.TRI_STRIDE_FLOATS;
            this._blasEntries.set(entry.geometryId, entry);
        }

        this.totalTriangleCount = totalTris;

        // Compute scene bounds from all centroids
        this._computeSceneBounds();

        // Upload triangle buffer
        this._triangleBuffer?.destroy();
        this._triangleBuffer = this._device.createBuffer({
            label: 'BVH/Triangles',
            size: Math.max(allTriangles.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._triangleBuffer, 0, allTriangles);

        // Pack materials
        this._updateMaterialBuffer(materials);
    }

    /**
     * Build TLAS from current instance transforms.
     * Called every frame.
     */
    public buildTLAS(_commandEncoder: GPUCommandEncoder, _scene: Scene): void {
        // Implementation in Task 11
    }

    /**
     * Pack all PathTracerMaterials into the GPU material buffer.
     */
    public updateMaterials(scene: Scene): void {
        const objects = scene.getOrderedObjects();
        const materials: PathTracerMaterial[] = [];
        for (const obj of objects) {
            materials.push(obj.pathTracerMaterial ?? new PathTracerMaterial());
        }
        this._updateMaterialBuffer(materials);
    }

    /**
     * Run GPU BVH construction for all BLAS entries.
     * Dispatches morton codes -> radix sort -> tree build -> refit for each entry.
     */
    public buildBLASTree(commandEncoder: GPUCommandEncoder): void {
        // Destroy scratch buffers from previous build (safe now since prior commands have been submitted)
        for (const buf of this._pendingScratchBuffers) {
            buf.destroy();
        }
        this._pendingScratchBuffers.length = 0;

        this._ensureBLASPipelines();
        this.totalBLASNodes = 0;

        for (const [, entry] of this._blasEntries) {
            this._buildSingleBLAS(commandEncoder, entry);
        }
    }

    private _extractTriangles(
        geometry: Geometry,
        materialIndex: number
    ): { triangles: Float32Array; triCount: number } {
        const verts = geometry.vertices!;
        const indices = geometry.indices!;
        const triCount = Math.floor(indices.length / 3);
        const tris = new Float32Array(triCount * BVHBuilder.TRI_STRIDE_FLOATS);
        const VERT_STRIDE = 9; // 4 pos + 3 normal + 2 uv

        for (let t = 0; t < triCount; t++) {
            const i0 = indices[t * 3 + 0];
            const i1 = indices[t * 3 + 1];
            const i2 = indices[t * 3 + 2];
            const out = t * BVHBuilder.TRI_STRIDE_FLOATS;

            // Positions (x,y,z from each vertex, skip w at offset 3)
            tris[out + 0] = verts[i0 * VERT_STRIDE + 0];
            tris[out + 1] = verts[i0 * VERT_STRIDE + 1];
            tris[out + 2] = verts[i0 * VERT_STRIDE + 2];
            tris[out + 3] = verts[i1 * VERT_STRIDE + 0];
            tris[out + 4] = verts[i1 * VERT_STRIDE + 1];
            tris[out + 5] = verts[i1 * VERT_STRIDE + 2];
            tris[out + 6] = verts[i2 * VERT_STRIDE + 0];
            tris[out + 7] = verts[i2 * VERT_STRIDE + 1];
            tris[out + 8] = verts[i2 * VERT_STRIDE + 2];

            // Normals (offset 4 in vertex stride)
            tris[out + 9]  = verts[i0 * VERT_STRIDE + 4];
            tris[out + 10] = verts[i0 * VERT_STRIDE + 5];
            tris[out + 11] = verts[i0 * VERT_STRIDE + 6];
            tris[out + 12] = verts[i1 * VERT_STRIDE + 4];
            tris[out + 13] = verts[i1 * VERT_STRIDE + 5];
            tris[out + 14] = verts[i1 * VERT_STRIDE + 6];
            tris[out + 15] = verts[i2 * VERT_STRIDE + 4];
            tris[out + 16] = verts[i2 * VERT_STRIDE + 5];
            tris[out + 17] = verts[i2 * VERT_STRIDE + 6];

            // Material index + padding
            tris[out + 18] = materialIndex;
            // [19-23] = 0 (padding, already zero from Float32Array init)
        }

        return { triangles: tris, triCount };
    }

    /**
     * Compute triangle centroids from extracted triangle data.
     * Returns a Float32Array of vec4f (xyz = centroid, w = 0).
     */
    private _computeCentroids(triangles: Float32Array, triCount: number): Float32Array {
        const centroids = new Float32Array(triCount * 4);
        for (let t = 0; t < triCount; t++) {
            const base = t * BVHBuilder.TRI_STRIDE_FLOATS;
            // Average of 3 vertex positions
            const cx = (triangles[base + 0] + triangles[base + 3] + triangles[base + 6]) / 3;
            const cy = (triangles[base + 1] + triangles[base + 4] + triangles[base + 7]) / 3;
            const cz = (triangles[base + 2] + triangles[base + 5] + triangles[base + 8]) / 3;
            centroids[t * 4 + 0] = cx;
            centroids[t * 4 + 1] = cy;
            centroids[t * 4 + 2] = cz;
            centroids[t * 4 + 3] = 0;
        }
        return centroids;
    }

    /**
     * Compute scene AABB from all cached centroid data.
     */
    private _computeSceneBounds(): void {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (const [, centroids] of this._centroidsMap) {
            const count = centroids.length / 4;
            for (let i = 0; i < count; i++) {
                const cx = centroids[i * 4 + 0];
                const cy = centroids[i * 4 + 1];
                const cz = centroids[i * 4 + 2];
                minX = Math.min(minX, cx);
                minY = Math.min(minY, cy);
                minZ = Math.min(minZ, cz);
                maxX = Math.max(maxX, cx);
                maxY = Math.max(maxY, cy);
                maxZ = Math.max(maxZ, cz);
            }
        }

        // Handle degenerate case
        if (!isFinite(minX)) {
            minX = minY = minZ = 0;
            maxX = maxY = maxZ = 1;
        }

        this._sceneMin = [minX, minY, minZ];
        this._sceneMax = [maxX, maxY, maxZ];
    }

    /**
     * Lazily create all GPU compute pipelines for BVH construction.
     */
    private _ensureBLASPipelines(): void {
        if (this._mortonPipeline) return;

        // Morton pipeline
        this._mortonBGL = this._device.createBindGroupLayout({
            label: 'BVH/MortonBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const mortonModule = this._device.createShaderModule({ code: mortonShader });
        this._mortonPipeline = this._device.createComputePipeline({
            label: 'BVH/Morton',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._mortonBGL] }),
            compute: { module: mortonModule, entryPoint: 'main' },
        });

        // Sort pipelines (3 entry points sharing same BGL)
        this._sortBGL = this._device.createBindGroupLayout({
            label: 'BVH/SortBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const sortLayout = this._device.createPipelineLayout({ bindGroupLayouts: [this._sortBGL] });
        const sortModule = this._device.createShaderModule({ code: radixSortShader });
        this._sortHistogramPipeline = this._device.createComputePipeline({
            label: 'BVH/Sort/Histogram',
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: 'histogram' },
        });
        this._sortPrefixPipeline = this._device.createComputePipeline({
            label: 'BVH/Sort/Prefix',
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: 'prefix_sum' },
        });
        this._sortScatterPipeline = this._device.createComputePipeline({
            label: 'BVH/Sort/Scatter',
            layout: sortLayout,
            compute: { module: sortModule, entryPoint: 'scatter' },
        });

        // Tree build pipeline
        this._treeBuildBGL = this._device.createBindGroupLayout({
            label: 'BVH/TreeBuildBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const treeBuildModule = this._device.createShaderModule({ code: treeBuildShader });
        this._treeBuildPipeline = this._device.createComputePipeline({
            label: 'BVH/TreeBuild',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._treeBuildBGL] }),
            compute: { module: treeBuildModule, entryPoint: 'main' },
        });

        // Refit pipeline
        this._refitBGL = this._device.createBindGroupLayout({
            label: 'BVH/RefitBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            ],
        });
        const refitModule = this._device.createShaderModule({ code: refitShader });
        this._refitPipeline = this._device.createComputePipeline({
            label: 'BVH/Refit',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._refitBGL] }),
            compute: { module: refitModule, entryPoint: 'main' },
        });

        // Params buffers
        this._mortonParamsBuf = this._device.createBuffer({
            label: 'BVH/MortonParams', size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._sortParamsBuf = this._device.createBuffer({
            label: 'BVH/SortParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._treeBuildParamsBuf = this._device.createBuffer({
            label: 'BVH/TreeBuildParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this._refitParamsBuf = this._device.createBuffer({
            label: 'BVH/RefitParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
    }

    /**
     * Build BVH for a single BLAS entry:
     *   morton codes -> radix sort (8 passes) -> tree build -> refit
     */
    private _buildSingleBLAS(commandEncoder: GPUCommandEncoder, entry: BLASEntry): void {
        const count = entry.triangleCount;
        if (count === 0) return;
        const nodeCount = 2 * count - 1; // binary tree: n-1 internal + n leaf
        const workgroupCount = Math.ceil(count / 256);

        // Retrieve cached centroid data
        const centroidData = this._centroidsMap.get(entry.geometryId);
        if (!centroidData) return;

        // Upload centroids to GPU
        const centroidsBuffer = this._device.createBuffer({
            label: 'BVH/Centroids',
            size: centroidData.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(centroidsBuffer, 0, centroidData);

        // Double-buffered morton keys/vals for radix sort ping-pong
        const mortonKeysA = this._device.createBuffer({
            label: 'BVH/MortonKeysA', size: count * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const mortonValsA = this._device.createBuffer({
            label: 'BVH/MortonValsA', size: count * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const mortonKeysB = this._device.createBuffer({
            label: 'BVH/MortonKeysB', size: count * 4,
            usage: GPUBufferUsage.STORAGE,
        });
        const mortonValsB = this._device.createBuffer({
            label: 'BVH/MortonValsB', size: count * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // BVH node buffer (stored persistently for this entry)
        const nodeBuffer = this._device.createBuffer({
            label: 'BVH/BLASNodes',
            size: nodeCount * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const parentsBuffer = this._device.createBuffer({
            label: 'BVH/Parents',
            size: nodeCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const atomicFlagsBuffer = this._device.createBuffer({
            label: 'BVH/AtomicFlags',
            size: nodeCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        const histogramBuffer = this._device.createBuffer({
            label: 'BVH/Histograms',
            size: 16 * workgroupCount * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // --- Step 1: Morton code computation ---
        const extX = this._sceneMax[0] - this._sceneMin[0];
        const extY = this._sceneMax[1] - this._sceneMin[1];
        const extZ = this._sceneMax[2] - this._sceneMin[2];
        const mortonParams = new Float32Array(8);
        const mortonParamsU32 = new Uint32Array(mortonParams.buffer);
        mortonParamsU32[0] = count;                                     // count
        mortonParams[1] = this._sceneMin[0];                            // sceneMinX
        mortonParams[2] = this._sceneMin[1];                            // sceneMinY
        mortonParams[3] = this._sceneMin[2];                            // sceneMinZ
        mortonParams[4] = extX > 0 ? 1.0 / extX : 1.0;                 // sceneExtX (inverse)
        mortonParams[5] = extY > 0 ? 1.0 / extY : 1.0;                 // sceneExtY
        mortonParams[6] = extZ > 0 ? 1.0 / extZ : 1.0;                 // sceneExtZ
        mortonParamsU32[7] = 0;                                         // _pad
        this._device.queue.writeBuffer(this._mortonParamsBuf!, 0, mortonParams);

        const mortonBG = this._device.createBindGroup({
            layout: this._mortonBGL!,
            entries: [
                { binding: 0, resource: { buffer: centroidsBuffer } },
                { binding: 1, resource: { buffer: mortonKeysA } },
                { binding: 2, resource: { buffer: mortonValsA } },
                { binding: 3, resource: { buffer: this._mortonParamsBuf! } },
            ],
        });

        const mortonPass = commandEncoder.beginComputePass();
        mortonPass.setPipeline(this._mortonPipeline!);
        mortonPass.setBindGroup(0, mortonBG);
        mortonPass.dispatchWorkgroups(workgroupCount);
        mortonPass.end();

        // --- Step 2: Radix sort (8 passes, 4 bits each) ---
        const buffers = [
            { keys: mortonKeysA, vals: mortonValsA },
            { keys: mortonKeysB, vals: mortonValsB },
        ];

        for (let pass = 0; pass < 8; pass++) {
            const src = buffers[pass & 1];
            const dst = buffers[(pass + 1) & 1];
            const bitOffset = pass * 4;

            const sortParams = new Uint32Array([count, bitOffset, workgroupCount, 0]);
            this._device.queue.writeBuffer(this._sortParamsBuf!, 0, sortParams);

            const sortBG = this._device.createBindGroup({
                layout: this._sortBGL!,
                entries: [
                    { binding: 0, resource: { buffer: src.keys } },
                    { binding: 1, resource: { buffer: src.vals } },
                    { binding: 2, resource: { buffer: dst.keys } },
                    { binding: 3, resource: { buffer: dst.vals } },
                    { binding: 4, resource: { buffer: histogramBuffer } },
                    { binding: 5, resource: { buffer: this._sortParamsBuf! } },
                ],
            });

            // Histogram
            const histPass = commandEncoder.beginComputePass();
            histPass.setPipeline(this._sortHistogramPipeline!);
            histPass.setBindGroup(0, sortBG);
            histPass.dispatchWorkgroups(workgroupCount);
            histPass.end();

            // Prefix sum (single workgroup)
            const prefixPass = commandEncoder.beginComputePass();
            prefixPass.setPipeline(this._sortPrefixPipeline!);
            prefixPass.setBindGroup(0, sortBG);
            prefixPass.dispatchWorkgroups(1);
            prefixPass.end();

            // Scatter
            const scatterPass = commandEncoder.beginComputePass();
            scatterPass.setPipeline(this._sortScatterPipeline!);
            scatterPass.setBindGroup(0, sortBG);
            scatterPass.dispatchWorkgroups(workgroupCount);
            scatterPass.end();
        }

        // After 8 passes (even number), sorted result is back in buffers[0] (keysA/valsA)
        const sortedKeys = buffers[0].keys;
        const sortedVals = buffers[0].vals;

        // --- Step 3: Karras tree build ---
        const treeBuildParams = new Uint32Array([count, 0, 0, 0]);
        this._device.queue.writeBuffer(this._treeBuildParamsBuf!, 0, treeBuildParams);

        const treeBuildBG = this._device.createBindGroup({
            layout: this._treeBuildBGL!,
            entries: [
                { binding: 0, resource: { buffer: sortedKeys } },
                { binding: 1, resource: { buffer: nodeBuffer } },
                { binding: 2, resource: { buffer: parentsBuffer } },
                { binding: 3, resource: { buffer: this._treeBuildParamsBuf! } },
            ],
        });

        const treePass = commandEncoder.beginComputePass();
        treePass.setPipeline(this._treeBuildPipeline!);
        treePass.setBindGroup(0, treeBuildBG);
        treePass.dispatchWorkgroups(Math.ceil((count - 1) / 256));
        treePass.end();

        // --- Step 4: Bottom-up AABB refit ---
        const refitParams = new Uint32Array([
            count,
            entry.triangleOffset,
            BVHBuilder.TRI_STRIDE_FLOATS,
            0,
        ]);
        this._device.queue.writeBuffer(this._refitParamsBuf!, 0, refitParams);

        const refitBG = this._device.createBindGroup({
            layout: this._refitBGL!,
            entries: [
                { binding: 0, resource: { buffer: nodeBuffer } },
                { binding: 1, resource: { buffer: this._triangleBuffer! } },
                { binding: 2, resource: { buffer: atomicFlagsBuffer } },
                { binding: 3, resource: { buffer: this._refitParamsBuf! } },
                { binding: 4, resource: { buffer: parentsBuffer } },
                { binding: 5, resource: { buffer: sortedVals } },
            ],
        });

        const refitPass = commandEncoder.beginComputePass();
        refitPass.setPipeline(this._refitPipeline!);
        refitPass.setBindGroup(0, refitBG);
        refitPass.dispatchWorkgroups(workgroupCount);
        refitPass.end();

        // Store results on the entry
        entry.nodeOffset = 0;
        entry.nodeCount = nodeCount;
        this.totalBLASNodes += nodeCount;

        // Keep the node buffer as the BLAS node buffer
        // (For multi-geometry, these would be merged; for now store last)
        this._blasNodeBuffer?.destroy();
        this._blasNodeBuffer = nodeBuffer;

        // Defer scratch buffer destruction until after command submission
        this._pendingScratchBuffers.push(
            centroidsBuffer, mortonKeysA, mortonValsA, mortonKeysB, mortonValsB,
            parentsBuffer, atomicFlagsBuffer, histogramBuffer,
        );
    }

    private _updateMaterialBuffer(materials: PathTracerMaterial[]): void {
        const floatsPerMat = PathTracerMaterial.GPU_STRIDE / 4; // 16
        const staging = new Float32Array(Math.max(materials.length * floatsPerMat, 1));
        for (let i = 0; i < materials.length; i++) {
            materials[i].packInto(staging, i * floatsPerMat);
        }
        this._materialBuffer?.destroy();
        this._materialBuffer = this._device.createBuffer({
            label: 'BVH/Materials',
            size: Math.max(staging.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._materialBuffer, 0, staging);
        this.materialCount = materials.length;
    }

    public destroy(): void {
        this._triangleBuffer?.destroy();
        this._blasNodeBuffer?.destroy();
        this._instanceBuffer?.destroy();
        this._tlasNodeBuffer?.destroy();
        this._materialBuffer?.destroy();
        this._mortonBuffer?.destroy();
        this._mortonIndicesBuffer?.destroy();
        this._sortScratchBuffer?.destroy();
        this._mortonParamsBuf?.destroy();
        this._sortParamsBuf?.destroy();
        this._treeBuildParamsBuf?.destroy();
        this._refitParamsBuf?.destroy();
        for (const buf of this._pendingScratchBuffers) {
            buf.destroy();
        }
        this._pendingScratchBuffers.length = 0;
    }
}
