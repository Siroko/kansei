import { Geometry } from "../buffers/Geometry";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { PathTracerMaterial } from "./PathTracerMaterial";
import { mortonShader } from "./shaders/morton.wgsl";
import { radixSortShader } from "./shaders/radix-sort.wgsl";
import { treeBuildShader } from "./shaders/tree-build.wgsl";
import { refitLeavesShader, refitInternalShader } from "./shaders/refit.wgsl";
import { instanceAABBShader } from "./shaders/instance-aabb.wgsl";
import { tlasRefitShader } from "./shaders/tlas-refit.wgsl";

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
    private _refitLeavesPipeline: GPUComputePipeline | null = null;
    private _refitLeavesBGL: GPUBindGroupLayout | null = null;
    private _refitInternalPipeline: GPUComputePipeline | null = null;
    private _refitInternalBGL: GPUBindGroupLayout | null = null;
    private _mortonParamsBuf: GPUBuffer | null = null;
    private _sortParamsBuf: GPUBuffer | null = null;
    private _treeBuildParamsBuf: GPUBuffer | null = null;
    private _refitParamsBuf: GPUBuffer | null = null;

    // TLAS construction pipelines
    private _instanceAABBPipeline: GPUComputePipeline | null = null;
    private _instanceAABBBGL: GPUBindGroupLayout | null = null;
    private _instanceAABBParamsBuf: GPUBuffer | null = null;
    private _tlasRefitLeafPipeline: GPUComputePipeline | null = null;
    private _tlasRefitMergePipeline: GPUComputePipeline | null = null;
    private _tlasRefitBGL: GPUBindGroupLayout | null = null;
    private _tlasRefitParamsBuf: GPUBuffer | null = null;

    // Reusable staging array for instance packing
    private _instanceStaging: Float32Array | null = null;
    private _instanceCapacity: number = 0;

    // Track instance count for TLAS structure rebuild
    private _lastTLASInstanceCount: number = 0;

    // Instance stride: 28 floats/u32 = 112 bytes per instance
    // transform(3*vec4f) + invTransform(3*vec4f) + 4 u32 metadata
    static readonly INSTANCE_STRIDE = 28;

    // Map from geometry object to its BLAS entry key for instance packing
    private _geomToBLASKey: Map<object, string> = new Map();

    // Per-geometry centroid cache (vec4f per triangle: xyz = centroid, w = unused)
    private _centroidsMap: Map<string, Float32Array> = new Map();

    // Per-BLAS local-space AABB (for CPU TLAS construction)
    private _blasLocalBounds: Map<string, { min: Float32Array; max: Float32Array }> = new Map();

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
        this._geomToBLASKey.clear();

        for (const [geom, data] of geometryMap) {
            const geoId = (geom as any).uuid ?? String(offset);
            const entry: BLASEntry = {
                geometryId: geoId,
                triangleOffset: offset / BVHBuilder.TRI_STRIDE_FLOATS,
                triangleCount: data.triCount,
                nodeOffset: 0,
                nodeCount: 0,
            };
            allTriangles.set(data.triangles, offset);

            // Compute and cache centroids for this geometry
            const centroids = this._computeCentroids(data.triangles, data.triCount);
            this._centroidsMap.set(entry.geometryId, centroids);

            // Compute and cache local-space AABB for CPU TLAS construction
            this._blasLocalBounds.set(entry.geometryId, this._computeLocalBounds(data.triangles, data.triCount));

            offset += data.triCount * BVHBuilder.TRI_STRIDE_FLOATS;
            this._blasEntries.set(entry.geometryId, entry);
            this._geomToBLASKey.set(geom, geoId);
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
     * Called every frame to account for moving objects.
     *
     * Approach: CPU builds a balanced binary tree structure (child pointers only)
     * once when instance count changes. GPU fills leaf AABBs from instance
     * transforms, then merges internal AABBs bottom-up in ceil(log2(N)) passes.
     */
    public buildTLAS(commandEncoder: GPUCommandEncoder, scene: Scene): void {
        const objects = scene.getOrderedObjects();
        if (objects.length === 0) return;

        this._ensureTLASPipelines();

        // Pack instance data to GPU buffer (reuses buffer if same size)
        this._packInstances(objects);

        const count = this._totalInstances;
        if (count === 0) return;

        // Rebuild balanced tree structure when instance count changes
        if (count !== this._lastTLASInstanceCount) {
            this._buildBalancedTLASStructure(count);
            this._lastTLASInstanceCount = count;
        }

        // Upload params
        const tlasRefitParams = new Uint32Array([count, 0, 0, 0]);
        this._device.queue.writeBuffer(this._tlasRefitParamsBuf!, 0, tlasRefitParams);

        const tlasRefitBG = this._device.createBindGroup({
            layout: this._tlasRefitBGL!,
            entries: [
                { binding: 0, resource: { buffer: this._tlasNodeBuffer! } },
                { binding: 1, resource: { buffer: this._instanceBuffer! } },
                { binding: 2, resource: { buffer: this._blasNodeBuffer! } },
                { binding: 3, resource: { buffer: this._tlasRefitParamsBuf! } },
            ],
        });

        // GPU Pass 1: Fill leaf AABBs from instance transforms + BLAS root bounds
        const leafWG = Math.ceil(count / 256);
        const leafPass = commandEncoder.beginComputePass({ label: 'TLAS/InitLeaves' });
        leafPass.setPipeline(this._tlasRefitLeafPipeline!);
        leafPass.setBindGroup(0, tlasRefitBG);
        leafPass.dispatchWorkgroups(leafWG);
        leafPass.end();

        // GPU Pass 2+: Merge internal node AABBs bottom-up.
        // Each compute pass has an implicit memory barrier in WebGPU,
        // so child bounds are visible before parents read them.
        // ceil(log2(N)) passes covers the full balanced tree depth.
        if (count > 1) {
            const mergeCount = Math.ceil(Math.log2(count));
            const mergeWG = Math.ceil((count - 1) / 256);
            for (let i = 0; i < mergeCount; i++) {
                const mergePass = commandEncoder.beginComputePass({ label: `TLAS/Merge/${i}` });
                mergePass.setPipeline(this._tlasRefitMergePipeline!);
                mergePass.setBindGroup(0, tlasRefitBG);
                mergePass.dispatchWorkgroups(mergeWG);
                mergePass.end();
            }
        }
    }

    /**
     * Build a balanced binary tree structure for the TLAS on CPU.
     * Only sets child pointers — GPU fills AABBs via initLeaves + mergeNodes.
     * Layout: internal nodes [0..N-2], leaf nodes [N-1..2N-2].
     */
    private _buildBalancedTLASStructure(N: number): void {
        const nodeCount = 2 * N - 1;
        const buf = new ArrayBuffer(nodeCount * 32);
        const i32 = new Int32Array(buf);

        if (N === 1) {
            // Single leaf at index 0 — initLeaves handles it
        } else {
            let nextInternal = 0;
            const buildNode = (indices: number[]): number => {
                if (indices.length === 1) {
                    return (N - 1) + indices[0]; // leaf index
                }
                const myIdx = nextInternal++;
                const mid = Math.floor(indices.length / 2);
                const left = buildNode(indices.slice(0, mid));
                const right = buildNode(indices.slice(mid));
                const off = myIdx * 8; // 8 int32/float32 per node
                i32[off + 3] = left;   // leftChild
                i32[off + 7] = right;  // rightChild
                return myIdx;
            };
            buildNode(Array.from({ length: N }, (_, i) => i));
        }

        this._tlasNodeBuffer?.destroy();
        this._tlasNodeBuffer = this._device.createBuffer({
            label: 'TLAS/Nodes',
            size: nodeCount * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._tlasNodeBuffer, 0, buf);
    }

    /**
     * Pack renderable objects into the GPU instance buffer.
     * Each instance = 28 floats/u32 = 112 bytes.
     */
    private _packInstances(objects: Renderable[]): void {
        const STRIDE = BVHBuilder.INSTANCE_STRIDE;

        // Reuse staging array if capacity is sufficient
        const needed = objects.length * STRIDE;
        if (!this._instanceStaging || this._instanceStaging.length < needed) {
            this._instanceStaging = new Float32Array(needed);
        }
        const instanceData = this._instanceStaging;
        const instanceDataU32 = new Uint32Array(instanceData.buffer);

        let instanceCount = 0;

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];
            const m = obj.worldMatrix.internalMat4;
            const off = instanceCount * STRIDE;

            // Convert column-major 4x4 to row-major 3x4 (transform rows)
            // Col-major: m[0..3]=col0, m[4..7]=col1, m[8..11]=col2, m[12..15]=col3
            // Row 0: m[0], m[4], m[8], m[12]
            instanceData[off + 0] = m[0];  instanceData[off + 1] = m[4];
            instanceData[off + 2] = m[8];  instanceData[off + 3] = m[12];
            // Row 1: m[1], m[5], m[9], m[13]
            instanceData[off + 4] = m[1];  instanceData[off + 5] = m[5];
            instanceData[off + 6] = m[9];  instanceData[off + 7] = m[13];
            // Row 2: m[2], m[6], m[10], m[14]
            instanceData[off + 8]  = m[2];  instanceData[off + 9]  = m[6];
            instanceData[off + 10] = m[10]; instanceData[off + 11] = m[14];

            // Compute inverse of the 3x3 rotation/scale part
            const det = m[0] * (m[5] * m[10] - m[6] * m[9])
                      - m[4] * (m[1] * m[10] - m[2] * m[9])
                      + m[8] * (m[1] * m[6]  - m[2] * m[5]);
            const invDet = det !== 0 ? 1.0 / det : 1.0;

            // A^-1 = adj(A) / det — off-diagonals are cofactors transposed
            const inv00 = (m[5] * m[10] - m[6] * m[9])  * invDet;
            const inv01 = (m[6] * m[8]  - m[4] * m[10]) * invDet;
            const inv02 = (m[4] * m[9]  - m[5] * m[8])  * invDet;
            const inv10 = (m[2] * m[9]  - m[1] * m[10]) * invDet;
            const inv11 = (m[0] * m[10] - m[2] * m[8])  * invDet;
            const inv12 = (m[1] * m[8]  - m[0] * m[9])  * invDet;
            const inv20 = (m[1] * m[6]  - m[2] * m[5])  * invDet;
            const inv21 = (m[2] * m[4]  - m[0] * m[6])  * invDet;
            const inv22 = (m[0] * m[5]  - m[1] * m[4])  * invDet;

            const tx = m[12], ty = m[13], tz = m[14];

            // invTransform row 0
            instanceData[off + 12] = inv00;
            instanceData[off + 13] = inv01;
            instanceData[off + 14] = inv02;
            instanceData[off + 15] = -(inv00 * tx + inv01 * ty + inv02 * tz);
            // invTransform row 1
            instanceData[off + 16] = inv10;
            instanceData[off + 17] = inv11;
            instanceData[off + 18] = inv12;
            instanceData[off + 19] = -(inv10 * tx + inv11 * ty + inv12 * tz);
            // invTransform row 2
            instanceData[off + 20] = inv20;
            instanceData[off + 21] = inv21;
            instanceData[off + 22] = inv22;
            instanceData[off + 23] = -(inv20 * tx + inv21 * ty + inv22 * tz);

            // Find BLAS entry for this geometry
            const geom = obj.geometry.isInstancedGeometry
                ? (obj.geometry as InstancedGeometry).geometry ?? obj.geometry
                : obj.geometry;

            const blasKey = this._geomToBLASKey.get(geom);
            const blasEntry = blasKey ? this._blasEntries.get(blasKey) : undefined;

            // BLAS metadata (packed as u32)
            instanceDataU32[off + 24] = blasEntry ? blasEntry.nodeOffset : 0;
            instanceDataU32[off + 25] = blasEntry ? blasEntry.triangleOffset : 0;
            instanceDataU32[off + 26] = blasEntry ? blasEntry.triangleCount : 0;
            instanceDataU32[off + 27] = i; // materialIndex = object index

            instanceCount++;
        }

        this._totalInstances = instanceCount;

        // Reuse GPU buffer if capacity is sufficient, only recreate if grown
        const byteSize = Math.max(instanceCount * STRIDE * 4, 4);
        if (!this._instanceBuffer || this._instanceCapacity < instanceCount) {
            this._instanceBuffer?.destroy();
            this._instanceBuffer = this._device.createBuffer({
                label: 'TLAS/Instances',
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._instanceCapacity = instanceCount;
        }
        this._device.queue.writeBuffer(
            this._instanceBuffer, 0,
            instanceData.buffer, 0, instanceCount * STRIDE * 4,
        );
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
     * All entries' nodes are merged into a single contiguous BLAS node buffer.
     */
    /**
     * Destroy scratch buffers from previous frame. Call at the start of each frame
     * BEFORE recording new commands, so prior GPU work has completed.
     */
    public cleanupScratchBuffers(): void {
        for (const buf of this._pendingScratchBuffers) {
            buf.destroy();
        }
        this._pendingScratchBuffers.length = 0;
    }

    public buildBLASTree(_commandEncoder: GPUCommandEncoder): void {
        // Use CPU BLAS for correctness — bypasses GPU Morton/sort/tree/refit pipeline
        this._buildCPUBLAS();
    }

    /**
     * Build BLAS on CPU using a balanced binary tree (midpoint split).
     * Tree depth = O(log n), fits within BLAS_STACK_SIZE = 32.
     */
    private _buildCPUBLAS(): void {
        // Pre-calculate total node count
        let totalNodes = 0;
        for (const [, entry] of this._blasEntries) {
            if (entry.triangleCount > 0) {
                totalNodes += 2 * entry.triangleCount - 1;
            }
        }
        this.totalBLASNodes = totalNodes;

        // Allocate combined BLAS node buffer
        const combinedBuf = new ArrayBuffer(Math.max(totalNodes * 32, 4));
        const combinedF32 = new Float32Array(combinedBuf);
        const combinedI32 = new Int32Array(combinedBuf);

        let nodeOffset = 0;
        for (const [geoId, entry] of this._blasEntries) {
            const n = entry.triangleCount;
            if (n === 0) continue;
            const nodeCount = 2 * n - 1;
            entry.nodeOffset = nodeOffset;
            entry.nodeCount = nodeCount;

            // Compute per-triangle AABBs
            const triAABBs: { min: number[]; max: number[] }[] = [];
            let geom: any = null;
            for (const [g, key] of this._geomToBLASKey) {
                if (key === geoId) { geom = g; break; }
            }

            if (geom && geom.vertices && geom.indices) {
                const verts = geom.vertices;
                const indices = geom.indices;
                const VERT_STRIDE = 9;
                for (let t = 0; t < n; t++) {
                    const i0 = indices[t * 3 + 0];
                    const i1 = indices[t * 3 + 1];
                    const i2 = indices[t * 3 + 2];
                    const v0x = verts[i0 * VERT_STRIDE], v0y = verts[i0 * VERT_STRIDE + 1], v0z = verts[i0 * VERT_STRIDE + 2];
                    const v1x = verts[i1 * VERT_STRIDE], v1y = verts[i1 * VERT_STRIDE + 1], v1z = verts[i1 * VERT_STRIDE + 2];
                    const v2x = verts[i2 * VERT_STRIDE], v2y = verts[i2 * VERT_STRIDE + 1], v2z = verts[i2 * VERT_STRIDE + 2];
                    triAABBs.push({
                        min: [Math.min(v0x, v1x, v2x), Math.min(v0y, v1y, v2y), Math.min(v0z, v1z, v2z)],
                        max: [Math.max(v0x, v1x, v2x), Math.max(v0y, v1y, v2y), Math.max(v0z, v1z, v2z)],
                    });
                }
            } else {
                const lb = this._blasLocalBounds.get(geoId);
                for (let t = 0; t < n; t++) {
                    triAABBs.push({
                        min: lb ? [lb.min[0], lb.min[1], lb.min[2]] : [-1, -1, -1],
                        max: lb ? [lb.max[0], lb.max[1], lb.max[2]] : [1, 1, 1],
                    });
                }
            }

            // Build balanced binary tree via recursive midpoint split.
            // Layout: internal nodes [0..n-2], leaf nodes [n-1..2n-2].
            // triOrder maps sorted position → original triangle index.
            const triOrder = Array.from({ length: n }, (_, i) => i);
            let nextInternal = 0;

            // Recursively build the tree, returns the local node index.
            const buildNode = (triIndices: number[]): number => {
                if (triIndices.length === 1) {
                    // Leaf node
                    const triIdx = triIndices[0];
                    const leafLocalIdx = (n - 1) + triIdx;
                    const globalIdx = nodeOffset + leafLocalIdx;
                    const off = globalIdx * 8;
                    combinedF32[off + 0] = triAABBs[triIdx].min[0];
                    combinedF32[off + 1] = triAABBs[triIdx].min[1];
                    combinedF32[off + 2] = triAABBs[triIdx].min[2];
                    combinedI32[off + 3] = -(triIdx + 1);
                    combinedF32[off + 4] = triAABBs[triIdx].max[0];
                    combinedF32[off + 5] = triAABBs[triIdx].max[1];
                    combinedF32[off + 6] = triAABBs[triIdx].max[2];
                    combinedI32[off + 7] = -1;
                    return leafLocalIdx;
                }

                // Allocate internal node
                const myIdx = nextInternal++;
                const mid = Math.floor(triIndices.length / 2);
                const leftChild = buildNode(triIndices.slice(0, mid));
                const rightChild = buildNode(triIndices.slice(mid));

                // Compute merged AABB
                let imin = [Infinity, Infinity, Infinity];
                let imax = [-Infinity, -Infinity, -Infinity];
                for (const ti of triIndices) {
                    imin[0] = Math.min(imin[0], triAABBs[ti].min[0]);
                    imin[1] = Math.min(imin[1], triAABBs[ti].min[1]);
                    imin[2] = Math.min(imin[2], triAABBs[ti].min[2]);
                    imax[0] = Math.max(imax[0], triAABBs[ti].max[0]);
                    imax[1] = Math.max(imax[1], triAABBs[ti].max[1]);
                    imax[2] = Math.max(imax[2], triAABBs[ti].max[2]);
                }

                const globalIdx = nodeOffset + myIdx;
                const off = globalIdx * 8;
                combinedF32[off + 0] = imin[0];
                combinedF32[off + 1] = imin[1];
                combinedF32[off + 2] = imin[2];
                combinedI32[off + 3] = leftChild;
                combinedF32[off + 4] = imax[0];
                combinedF32[off + 5] = imax[1];
                combinedF32[off + 6] = imax[2];
                combinedI32[off + 7] = rightChild;
                return myIdx;
            };

            buildNode(triOrder);

            nodeOffset += nodeCount;
        }

        // Upload to GPU
        this._blasNodeBuffer?.destroy();
        this._blasNodeBuffer = this._device.createBuffer({
            label: 'BVH/BLASNodes/CPU',
            size: Math.max(combinedBuf.byteLength, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._blasNodeBuffer, 0, combinedBuf);
    }

    public buildBLASTreeGPU(commandEncoder: GPUCommandEncoder): void {
        this._ensureBLASPipelines();

        // Pre-calculate total node count for combined buffer
        let totalNodes = 0;
        for (const [, entry] of this._blasEntries) {
            if (entry.triangleCount > 0) {
                totalNodes += 2 * entry.triangleCount - 1;
            }
        }
        this.totalBLASNodes = totalNodes;

        // Allocate combined BLAS node buffer
        this._blasNodeBuffer?.destroy();
        this._blasNodeBuffer = this._device.createBuffer({
            label: 'BVH/BLASNodes',
            size: Math.max(totalNodes * 32, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Build each entry into its own temp buffer, then copy into combined buffer
        let nodeOffset = 0;
        for (const [, entry] of this._blasEntries) {
            if (entry.triangleCount === 0) continue;
            const nodeCount = 2 * entry.triangleCount - 1;
            entry.nodeOffset = nodeOffset;
            entry.nodeCount = nodeCount;

            const tempNodeBuffer = this._buildSingleBLAS(commandEncoder, entry);
            if (tempNodeBuffer) {
                commandEncoder.copyBufferToBuffer(
                    tempNodeBuffer, 0,
                    this._blasNodeBuffer, nodeOffset * 32,
                    nodeCount * 32,
                );
                this._pendingScratchBuffers.push(tempNodeBuffer);
            }
            nodeOffset += nodeCount;
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
     * Compute local-space AABB from triangle vertex positions.
     */
    private _computeLocalBounds(triangles: Float32Array, triCount: number): { min: Float32Array; max: Float32Array } {
        const bMin = new Float32Array([Infinity, Infinity, Infinity]);
        const bMax = new Float32Array([-Infinity, -Infinity, -Infinity]);
        for (let t = 0; t < triCount; t++) {
            const base = t * BVHBuilder.TRI_STRIDE_FLOATS;
            for (let v = 0; v < 3; v++) {
                const vb = base + v * 3;
                bMin[0] = Math.min(bMin[0], triangles[vb]);
                bMin[1] = Math.min(bMin[1], triangles[vb + 1]);
                bMin[2] = Math.min(bMin[2], triangles[vb + 2]);
                bMax[0] = Math.max(bMax[0], triangles[vb]);
                bMax[1] = Math.max(bMax[1], triangles[vb + 1]);
                bMax[2] = Math.max(bMax[2], triangles[vb + 2]);
            }
        }
        return { min: bMin, max: bMax };
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
     * Lazily create GPU compute pipelines for TLAS construction.
     */
    private _ensureTLASPipelines(): void {
        if (this._instanceAABBPipeline) return;

        // Instance AABB pipeline
        this._instanceAABBBGL = this._device.createBindGroupLayout({
            label: 'TLAS/InstanceAABB_BGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const aabbModule = this._device.createShaderModule({ code: instanceAABBShader });
        this._instanceAABBPipeline = this._device.createComputePipeline({
            label: 'TLAS/InstanceAABB',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._instanceAABBBGL] }),
            compute: { module: aabbModule, entryPoint: 'main' },
        });

        this._instanceAABBParamsBuf = this._device.createBuffer({
            label: 'TLAS/InstanceAABBParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // TLAS refit pipelines (two entry points: initLeaves + mergeNodes)
        this._tlasRefitBGL = this._device.createBindGroupLayout({
            label: 'TLAS/RefitBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },          // nodes
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // instances
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // blasNodes
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },          // params
            ],
        });
        const tlasRefitModule = this._device.createShaderModule({ code: tlasRefitShader });
        const refitLayout = this._device.createPipelineLayout({ bindGroupLayouts: [this._tlasRefitBGL] });

        this._tlasRefitLeafPipeline = this._device.createComputePipeline({
            label: 'TLAS/Refit/Leaves',
            layout: refitLayout,
            compute: { module: tlasRefitModule, entryPoint: 'initLeaves' },
        });
        this._tlasRefitMergePipeline = this._device.createComputePipeline({
            label: 'TLAS/Refit/Merge',
            layout: refitLayout,
            compute: { module: tlasRefitModule, entryPoint: 'mergeNodes' },
        });

        this._tlasRefitParamsBuf = this._device.createBuffer({
            label: 'TLAS/RefitParams', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
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

        // Refit leaves pipeline
        this._refitLeavesBGL = this._device.createBindGroupLayout({
            label: 'BVH/RefitLeavesBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },        // nodes
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // triangles
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },        // ready flags
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },        // params
                { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // sortedIndices
            ],
        });
        const refitLeavesModule = this._device.createShaderModule({ code: refitLeavesShader });
        this._refitLeavesPipeline = this._device.createComputePipeline({
            label: 'BVH/RefitLeaves',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._refitLeavesBGL] }),
            compute: { module: refitLeavesModule, entryPoint: 'main' },
        });

        // Refit internal pipeline
        this._refitInternalBGL = this._device.createBindGroupLayout({
            label: 'BVH/RefitInternalBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // nodes
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },  // ready flags
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },  // params
            ],
        });
        const refitInternalModule = this._device.createShaderModule({ code: refitInternalShader });
        this._refitInternalPipeline = this._device.createComputePipeline({
            label: 'BVH/RefitInternal',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._refitInternalBGL] }),
            compute: { module: refitInternalModule, entryPoint: 'main' },
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
     * Returns the per-entry node buffer (caller copies into combined buffer).
     */
    private _buildSingleBLAS(commandEncoder: GPUCommandEncoder, entry: BLASEntry): GPUBuffer | null {
        const count = entry.triangleCount;
        if (count === 0) return null;
        const nodeCount = 2 * count - 1; // binary tree: n-1 internal + n leaf
        const workgroupCount = Math.ceil(count / 256);

        // Retrieve cached centroid data
        const centroidData = this._centroidsMap.get(entry.geometryId);
        if (!centroidData) return null;

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

        // BVH node buffer (temporary — copied into combined buffer by caller)
        const nodeBuffer = this._device.createBuffer({
            label: 'BVH/BLASNodes/Entry',
            size: nodeCount * 32,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
        });
        const parentsBuffer = this._device.createBuffer({
            label: 'BVH/Parents',
            size: nodeCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        // Root (index 0) has no parent — set to -1 so refit walk-up terminates
        this._device.queue.writeBuffer(parentsBuffer, 0, new Int32Array([-1]));

        const histogramBuffer = this._device.createBuffer({
            label: 'BVH/Histograms',
            size: 16 * workgroupCount * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // --- Step 1: Morton code computation ---
        // Per-entry param buffers to avoid overwriting shared buffers
        const entryMortonParamsBuf = this._device.createBuffer({
            label: 'BVH/MortonParams/Entry', size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
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
        this._device.queue.writeBuffer(entryMortonParamsBuf, 0, mortonParams);

        const mortonBG = this._device.createBindGroup({
            layout: this._mortonBGL!,
            entries: [
                { binding: 0, resource: { buffer: centroidsBuffer } },
                { binding: 1, resource: { buffer: mortonKeysA } },
                { binding: 2, resource: { buffer: mortonValsA } },
                { binding: 3, resource: { buffer: entryMortonParamsBuf } },
            ],
        });

        const mortonPass = commandEncoder.beginComputePass();
        mortonPass.setPipeline(this._mortonPipeline!);
        mortonPass.setBindGroup(0, mortonBG);
        mortonPass.dispatchWorkgroups(workgroupCount);
        mortonPass.end();

        // --- Step 2: Radix sort (8 passes, 4 bits each) ---
        // Each pass needs its own param buffer because writeBuffer is immediate
        // (executes before the command buffer) — sharing one buffer means all
        // passes would see only the last-written bitOffset.
        const buffers = [
            { keys: mortonKeysA, vals: mortonValsA },
            { keys: mortonKeysB, vals: mortonValsB },
        ];

        const sortPassParamsBufs: GPUBuffer[] = [];
        for (let pass = 0; pass < 8; pass++) {
            const src = buffers[pass & 1];
            const dst = buffers[(pass + 1) & 1];
            const bitOffset = pass * 4;

            const passSortParamsBuf = this._device.createBuffer({
                label: `BVH/SortParams/Entry/Pass${pass}`, size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });
            const sortParams = new Uint32Array([count, bitOffset, workgroupCount, 0]);
            this._device.queue.writeBuffer(passSortParamsBuf, 0, sortParams);
            sortPassParamsBufs.push(passSortParamsBuf);

            const sortBG = this._device.createBindGroup({
                layout: this._sortBGL!,
                entries: [
                    { binding: 0, resource: { buffer: src.keys } },
                    { binding: 1, resource: { buffer: src.vals } },
                    { binding: 2, resource: { buffer: dst.keys } },
                    { binding: 3, resource: { buffer: dst.vals } },
                    { binding: 4, resource: { buffer: histogramBuffer } },
                    { binding: 5, resource: { buffer: passSortParamsBuf } },
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
        const entryTreeBuildParamsBuf = this._device.createBuffer({
            label: 'BVH/TreeBuildParams/Entry', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const treeBuildParams = new Uint32Array([count, 0, 0, 0]);
        this._device.queue.writeBuffer(entryTreeBuildParamsBuf, 0, treeBuildParams);

        const treeBuildBG = this._device.createBindGroup({
            layout: this._treeBuildBGL!,
            entries: [
                { binding: 0, resource: { buffer: sortedKeys } },
                { binding: 1, resource: { buffer: nodeBuffer } },
                { binding: 2, resource: { buffer: parentsBuffer } },
                { binding: 3, resource: { buffer: entryTreeBuildParamsBuf } },
            ],
        });

        const treePass = commandEncoder.beginComputePass();
        treePass.setPipeline(this._treeBuildPipeline!);
        treePass.setBindGroup(0, treeBuildBG);
        treePass.dispatchWorkgroups(Math.ceil((count - 1) / 256));
        treePass.end();

        // --- Step 4: Multi-pass bottom-up AABB refit ---
        const entryRefitParamsBuf = this._device.createBuffer({
            label: 'BVH/RefitParams/Entry', size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        const refitParams = new Uint32Array([
            count,
            entry.triangleOffset,
            BVHBuilder.TRI_STRIDE_FLOATS,
            0,
        ]);
        this._device.queue.writeBuffer(entryRefitParamsBuf, 0, refitParams);

        // Ready flags buffer (zero-initialized by WebGPU)
        const readyBuffer = this._device.createBuffer({
            label: 'BVH/ReadyFlags',
            size: nodeCount * 4,
            usage: GPUBufferUsage.STORAGE,
        });

        // Pass A: Initialize leaf AABBs
        const leavesBG = this._device.createBindGroup({
            layout: this._refitLeavesBGL!,
            entries: [
                { binding: 0, resource: { buffer: nodeBuffer } },
                { binding: 1, resource: { buffer: this._triangleBuffer! } },
                { binding: 2, resource: { buffer: readyBuffer } },
                { binding: 3, resource: { buffer: entryRefitParamsBuf } },
                { binding: 4, resource: { buffer: sortedVals } },
            ],
        });

        const leavesPass = commandEncoder.beginComputePass();
        leavesPass.setPipeline(this._refitLeavesPipeline!);
        leavesPass.setBindGroup(0, leavesBG);
        leavesPass.dispatchWorkgroups(workgroupCount);
        leavesPass.end();

        // Pass B: Converge internal nodes (each dispatch = implicit barrier)
        const internalBG = this._device.createBindGroup({
            layout: this._refitInternalBGL!,
            entries: [
                { binding: 0, resource: { buffer: nodeBuffer } },
                { binding: 1, resource: { buffer: readyBuffer } },
                { binding: 2, resource: { buffer: entryRefitParamsBuf } },
            ],
        });

        const internalWG = Math.ceil((count - 1) / 256);
        const maxDepth = Math.max(Math.ceil(Math.log2(Math.max(count, 2))) + 1, count - 1);
        for (let level = 0; level < maxDepth; level++) {
            const intPass = commandEncoder.beginComputePass();
            intPass.setPipeline(this._refitInternalPipeline!);
            intPass.setBindGroup(0, internalBG);
            intPass.dispatchWorkgroups(internalWG);
            intPass.end();
        }

        // Defer scratch buffer destruction until after command submission
        this._pendingScratchBuffers.push(
            centroidsBuffer, mortonKeysA, mortonValsA, mortonKeysB, mortonValsB,
            parentsBuffer, readyBuffer, histogramBuffer,
            entryMortonParamsBuf, ...sortPassParamsBufs, entryTreeBuildParamsBuf, entryRefitParamsBuf,
        );

        // Return the node buffer so the caller can copy it into the combined buffer
        return nodeBuffer;
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
        // Note: refitLeaves/Internal pipelines are not buffers, no destroy needed
        this._instanceAABBParamsBuf?.destroy();
        this._tlasRefitParamsBuf?.destroy();
        for (const buf of this._pendingScratchBuffers) {
            buf.destroy();
        }
        this._pendingScratchBuffers.length = 0;
    }
}
