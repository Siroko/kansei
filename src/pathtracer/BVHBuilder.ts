import { Geometry } from "../buffers/Geometry";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { PathTracerMaterial } from "./PathTracerMaterial";
import { mortonShader } from "./shaders/morton.wgsl";
import { radixSortShader } from "./shaders/radix-sort.wgsl";
import { treeBuildShader } from "./shaders/tree-build.wgsl";
import { refitLeavesShader, refitInternalShader } from "./shaders/refit.wgsl";
import { instanceExpandShader } from "./shaders/instance-expand.wgsl";
import { tlasSortShader, tlasGatherShader } from "./shaders/tlas-sort.wgsl";
import { tlasBuildShader } from "./shaders/tlas-build.wgsl";

export interface BLASEntry {
    geometryId: string;
    triangleOffset: number;
    triangleCount: number;
    nodeOffset: number;
    nodeCount: number;
}

export interface DynamicBLASInfo {
    copies: number;
    nodeCountPerCopy: number;
    triCountPerCopy: number;
    copyNodeOffsets: number[];   // root node offset for each copy
    copyTriOffsets: number[];    // triangle offset for each copy
    maxDepth: number;            // BVH4 tree depth for refit passes
    origTriangleData: Float32Array;  // reordered original triangle data (CPU-side)
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
    public totalInstanceTriangles: number = 0; // sum of blasTriCount across all instances (handles shared BLAS)

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

    // TLAS construction (legacy fields, kept for destroy() cleanup)
    private _instanceAABBParamsBuf: GPUBuffer | null = null;
    private _tlasRefitParamsBuf: GPUBuffer | null = null;

    // Instance expansion pipelines (GPU-driven instanced geometry)
    private _expandPosPipeline: GPUComputePipeline | null = null;
    private _expandXfmPipeline: GPUComputePipeline | null = null;
    private _expandBGL: GPUBindGroupLayout | null = null;
    private _expandParamBufs: GPUBuffer[] = [];  // one per expansion group (writeBuffer ordering)

    // TLAS spatial sort pipelines (Morton code sorting for BVH quality)
    private _tlasSortKeysPipeline: GPUComputePipeline | null = null;
    private _tlasSortKeysBGL: GPUBindGroupLayout | null = null;
    private _tlasSortParamsBuf: GPUBuffer | null = null;
    private _tlasMortonKeysA: GPUBuffer | null = null;
    private _tlasMortonKeysB: GPUBuffer | null = null;
    private _tlasMortonValsA: GPUBuffer | null = null;
    private _tlasMortonValsB: GPUBuffer | null = null;
    private _tlasHistogramBuf: GPUBuffer | null = null;
    private _tlasSortPassParamsBufs: GPUBuffer[] = [];
    private _tlasGatherPipeline: GPUComputePipeline | null = null;
    private _tlasGatherBGL: GPUBindGroupLayout | null = null;
    private _tlasGatherParamsBuf: GPUBuffer | null = null;
    private _instanceBufferSorted: GPUBuffer | null = null;  // sorted copy
    private _tlasSortCapacity: number = 0;

    // BVH4 TLAS build pipelines
    private _tlasBvh4Buffer: GPUBuffer | null = null;
    private _tlasBuildLeafPipeline: GPUComputePipeline | null = null;
    private _tlasBuildInternalPipeline: GPUComputePipeline | null = null;
    private _tlasBuildBGL: GPUBindGroupLayout | null = null;
    private _tlasBuildParamsBufs: GPUBuffer[] = [];
    private _tlasLevelLayout: { offset: number; count: number }[] = [];
    private _tlasBvh4NodeCount: number = 0;

    // Dynamic BLAS support (per-instance BLAS copies for deformable meshes)
    private _dynamicBLASCopies: Map<string, number> = new Map();
    private _dynamicBLASInfos: Map<string, DynamicBLASInfo> = new Map();
    private _origTriangleBuffer: GPUBuffer | null = null;

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

    // Per-geometry extracted triangle data (for reordering after BVH build)
    private _blasTriData: Map<string, Float32Array> = new Map();

    // BVH4 (4-wide) node buffer for trace shader
    private _bvh4NodeBuffer: GPUBuffer | null = null;
    public totalBVH4Nodes: number = 0;

    // Scene bounds computed from centroid data
    private _sceneMin: [number, number, number] = [0, 0, 0];
    private _sceneMax: [number, number, number] = [0, 0, 0];

    constructor(device: GPUDevice) {
        this._device = device;
    }

    // Public accessors for trace shader bind groups
    get triangleBuffer(): GPUBuffer | null { return this._triangleBuffer; }
    get blasNodeBuffer(): GPUBuffer | null { return this._blasNodeBuffer; }
    get bvh4NodeBuffer(): GPUBuffer | null { return this._bvh4NodeBuffer; }
    get instanceBuffer(): GPUBuffer | null { return this._instanceBuffer; }
    get tlasNodeBuffer(): GPUBuffer | null { return this._tlasNodeBuffer; }
    get tlasBvh4Buffer(): GPUBuffer | null { return this._tlasBvh4Buffer; }
    get materialBuffer(): GPUBuffer | null { return this._materialBuffer; }
    get origTriangleBuffer(): GPUBuffer | null { return this._origTriangleBuffer; }
    get totalInstances(): number { return this._totalInstances; }
    get sceneMin(): [number, number, number] { return this._sceneMin; }
    get sceneMax(): [number, number, number] { return this._sceneMax; }

    /** Compute world-space scene AABB by transforming per-BLAS local bounds with instance world matrices. */
    computeWorldBounds(scene: Scene): { min: [number, number, number]; max: [number, number, number] } {
        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        for (const obj of scene.getOrderedObjects()) {
            const geom = obj.geometry.isInstancedGeometry
                ? (obj.geometry as InstancedGeometry).geometry ?? obj.geometry
                : obj.geometry;
            const blasKey = this._geomToBLASKey.get(geom);
            if (!blasKey) continue;
            const lb = this._blasLocalBounds.get(blasKey);
            if (!lb) continue;

            const m = obj.worldMatrix.internalMat4;
            // Transform 8 corners of local AABB by world matrix
            for (let cz = 0; cz < 2; cz++) {
                for (let cy = 0; cy < 2; cy++) {
                    for (let cx = 0; cx < 2; cx++) {
                        const lx = cx === 0 ? lb.min[0] : lb.max[0];
                        const ly = cy === 0 ? lb.min[1] : lb.max[1];
                        const lz = cz === 0 ? lb.min[2] : lb.max[2];
                        // m is column-major (gl-matrix)
                        const wx = m[0] * lx + m[4] * ly + m[8]  * lz + m[12];
                        const wy = m[1] * lx + m[5] * ly + m[9]  * lz + m[13];
                        const wz = m[2] * lx + m[6] * ly + m[10] * lz + m[14];
                        minX = Math.min(minX, wx); maxX = Math.max(maxX, wx);
                        minY = Math.min(minY, wy); maxY = Math.max(maxY, wy);
                        minZ = Math.min(minZ, wz); maxZ = Math.max(maxZ, wz);
                    }
                }
            }
        }

        if (!isFinite(minX)) { return { min: [0, 0, 0], max: [1, 1, 1] }; }
        return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
    }

    getDynamicBLASInfo(renderable: Renderable): DynamicBLASInfo | undefined {
        const geom = renderable.geometry.isInstancedGeometry
            ? (renderable.geometry as InstancedGeometry).geometry ?? renderable.geometry
            : renderable.geometry;
        const key = this._geomToBLASKey.get(geom);
        return key ? this._dynamicBLASInfos.get(key) : undefined;
    }

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

        // Assign consistent geometry IDs (uuid or auto-generated)
        const geoIdMap = new Map<object, string>();
        let geoIdCounter = 0;
        for (const [geom] of geometryMap) {
            geoIdMap.set(geom, (geom as any).uuid ?? `geo_${geoIdCounter++}`);
        }

        // Detect dynamic BLAS geometries and compute copy counts
        this._dynamicBLASCopies.clear();
        this._dynamicBLASInfos.clear();
        for (const obj of objects) {
            if (obj.dynamicBLAS && obj.geometry.isInstancedGeometry) {
                const geom = (obj.geometry as InstancedGeometry).geometry ?? obj.geometry;
                const geoKey = geoIdMap.get(geom);
                if (geoKey && !this._dynamicBLASCopies.has(geoKey)) {
                    const ig = obj.geometry as InstancedGeometry;
                    const copies = obj.gpuInstanceCount || ig.instanceCount;
                    this._dynamicBLASCopies.set(geoKey, copies);
                }
            }
        }

        // Pack all triangles into one contiguous buffer (with space for dynamic copies)
        let totalTris = 0;
        for (const [geom, data] of geometryMap) {
            const geoKey = geoIdMap.get(geom)!;
            const copies = this._dynamicBLASCopies.get(geoKey) || 1;
            totalTris += data.triCount * copies;
        }

        const allTriangles = new Float32Array(totalTris * BVHBuilder.TRI_STRIDE_FLOATS);
        let offset = 0;
        this._blasEntries.clear();
        this._centroidsMap.clear();
        this._geomToBLASKey.clear();
        this._blasTriData.clear();

        for (const [geom, data] of geometryMap) {
            const geoId = geoIdMap.get(geom)!;
            const copies = this._dynamicBLASCopies.get(geoId) || 1;
            const entry: BLASEntry = {
                geometryId: geoId,
                triangleOffset: offset / BVHBuilder.TRI_STRIDE_FLOATS,
                triangleCount: data.triCount,
                nodeOffset: 0,
                nodeCount: 0,
            };
            allTriangles.set(data.triangles, offset);

            // Store per-geometry triangle data for reordering after BVH build
            this._blasTriData.set(geoId, data.triangles);

            // Compute and cache centroids for this geometry
            const centroids = this._computeCentroids(data.triangles, data.triCount);
            this._centroidsMap.set(entry.geometryId, centroids);

            // Compute and cache local-space AABB for CPU TLAS construction
            this._blasLocalBounds.set(entry.geometryId, this._computeLocalBounds(data.triangles, data.triCount));

            // Reserve space for all copies (only copy 0 is packed now; rest filled after BVH build)
            offset += data.triCount * copies * BVHBuilder.TRI_STRIDE_FLOATS;
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
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
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

        this._ensureTLASBvh4Pipelines();

        // Pack instance data to GPU buffer (CPU instances written, GPU instances reserved)
        this._packInstances(objects);

        const count = this._totalInstances;
        if (count === 0) return;

        // Rebuild BVH4 tree layout when instance count changes
        if (count !== this._lastTLASInstanceCount) {
            this._computeTLASBvh4Layout(count);
            this._lastTLASInstanceCount = count;
        }

        // Fill GPU-expanded instance slots (reads positions/transforms from ComputeBuffers)
        this._dispatchInstanceExpansion(commandEncoder);

        // Sort instances by Morton code for better TLAS spatial coherence
        this._dispatchTLASSort(commandEncoder, count);

        // Build BVH4 TLAS (leaves + internal levels)
        this._dispatchTLASBvh4Build(commandEncoder, count);
    }

    /** Temporary per-frame expansion group info, built by _packInstances. */
    private _expandGroupInfos: Array<{
        gpuBuffer: GPUBuffer;
        fullTransform: boolean;
        instanceOffset: number;
        count: number;
        params: Float32Array; // 32 floats = 128 bytes uniform data
    }> = [];

    /**
     * Pack renderable objects into the GPU instance buffer.
     * Each instance = 28 floats/u32 = 112 bytes.
     *
     * Objects with gpuInstanceBuffer are skipped on CPU — their slots are
     * reserved and filled later by a GPU compute pass (_dispatchInstanceExpansion).
     */
    private _packInstances(objects: Renderable[]): void {
        const STRIDE = BVHBuilder.INSTANCE_STRIDE;

        // Pre-calculate total instance count
        let totalInstances = 0;
        for (const obj of objects) {
            if (obj.dynamicBLAS && obj.geometry.isInstancedGeometry) {
                // Dynamic BLAS: one TLAS instance per BLAS copy (identity transform)
                const ig = obj.geometry as InstancedGeometry;
                totalInstances += obj.gpuInstanceCount || ig.instanceCount;
            } else if (obj.gpuInstanceBuffer && obj.geometry.isInstancedGeometry) {
                totalInstances += obj.gpuInstanceCount || (obj.geometry as InstancedGeometry).instanceCount;
            } else {
                totalInstances++;
            }
        }

        // Reuse staging array if capacity is sufficient
        const needed = totalInstances * STRIDE;
        if (!this._instanceStaging || this._instanceStaging.length < needed) {
            this._instanceStaging = new Float32Array(needed);
        }
        const instanceData = this._instanceStaging;
        const instanceDataU32 = new Uint32Array(instanceData.buffer);

        let instanceCount = 0;
        this._expandGroupInfos.length = 0;

        for (let i = 0; i < objects.length; i++) {
            const obj = objects[i];

            // Find BLAS entry for this geometry
            const geom = obj.geometry.isInstancedGeometry
                ? (obj.geometry as InstancedGeometry).geometry ?? obj.geometry
                : obj.geometry;
            const blasKey = this._geomToBLASKey.get(geom);
            const blasEntry = blasKey ? this._blasEntries.get(blasKey) : undefined;
            const blasNodeOff = blasEntry ? blasEntry.nodeOffset : 0;
            const blasTriOff = blasEntry ? blasEntry.triangleOffset : 0;
            const blasTriCnt = blasEntry ? blasEntry.triangleCount : 0;

            // ── Dynamic BLAS: identity transforms + per-copy BLAS offsets ──
            if (obj.dynamicBLAS && obj.geometry.isInstancedGeometry) {
                const ig = obj.geometry as InstancedGeometry;
                const count = obj.gpuInstanceCount || ig.instanceCount;
                const info = blasKey ? this._dynamicBLASInfos.get(blasKey) : undefined;
                if (info) {
                    for (let k = 0; k < count; k++) {
                        const off = instanceCount * STRIDE;
                        // Identity transform rows (row-major 3×4)
                        instanceData[off + 0] = 1; instanceData[off + 1] = 0; instanceData[off + 2] = 0; instanceData[off + 3] = 0;
                        instanceData[off + 4] = 0; instanceData[off + 5] = 1; instanceData[off + 6] = 0; instanceData[off + 7] = 0;
                        instanceData[off + 8] = 0; instanceData[off + 9] = 0; instanceData[off + 10] = 1; instanceData[off + 11] = 0;
                        // Identity inverse transform rows
                        instanceData[off + 12] = 1; instanceData[off + 13] = 0; instanceData[off + 14] = 0; instanceData[off + 15] = 0;
                        instanceData[off + 16] = 0; instanceData[off + 17] = 1; instanceData[off + 18] = 0; instanceData[off + 19] = 0;
                        instanceData[off + 20] = 0; instanceData[off + 21] = 0; instanceData[off + 22] = 1; instanceData[off + 23] = 0;
                        // Per-copy BLAS offsets
                        instanceDataU32[off + 24] = info.copyNodeOffsets[k];
                        instanceDataU32[off + 25] = info.copyTriOffsets[k];
                        instanceDataU32[off + 26] = info.triCountPerCopy;
                        instanceDataU32[off + 27] = i; // materialIndex
                        instanceCount++;
                    }
                    continue;
                }
            }

            // ── GPU-expanded instances: reserve slots, fill via compute ──
            if (obj.gpuInstanceBuffer && obj.geometry.isInstancedGeometry) {
                const ig = obj.geometry as InstancedGeometry;
                const count = obj.gpuInstanceCount || ig.instanceCount;
                const m = obj.worldMatrix.internalMat4;

                // Parent rotation/scale rows (row-major from column-major gl-matrix)
                const r00 = m[0], r01 = m[4], r02 = m[8];
                const r10 = m[1], r11 = m[5], r12 = m[9];
                const r20 = m[2], r21 = m[6], r22 = m[10];
                const ptx = m[12], pty = m[13], ptz = m[14];

                // Inverse of rotation/scale (adjugate / det)
                const det = r00 * (r11 * r22 - r12 * r21)
                          - r01 * (r10 * r22 - r12 * r20)
                          + r02 * (r10 * r21 - r11 * r20);
                const invDet = det !== 0 ? 1.0 / det : 1.0;
                const inv00 = (r11 * r22 - r12 * r21) * invDet;
                const inv01 = (r02 * r21 - r01 * r22) * invDet;
                const inv02 = (r01 * r12 - r02 * r11) * invDet;
                const inv10 = (r12 * r20 - r10 * r22) * invDet;
                const inv11 = (r00 * r22 - r02 * r20) * invDet;
                const inv12 = (r02 * r10 - r00 * r12) * invDet;
                const inv20 = (r10 * r21 - r11 * r20) * invDet;
                const inv21 = (r01 * r20 - r00 * r21) * invDet;
                const inv22 = (r00 * r11 - r01 * r10) * invDet;

                // Build ExpandParams uniform (32 floats = 128 bytes)
                const params = new Float32Array(32);
                const paramsU32 = new Uint32Array(params.buffer);
                // r0, r1, r2 (parent rotscale + translation)
                params[0] = r00; params[1] = r01; params[2] = r02; params[3] = ptx;
                params[4] = r10; params[5] = r11; params[6] = r12; params[7] = pty;
                params[8] = r20; params[9] = r21; params[10] = r22; params[11] = ptz;
                // ir0, ir1, ir2 (inverse rotscale)
                params[12] = inv00; params[13] = inv01; params[14] = inv02; params[15] = 0;
                params[16] = inv10; params[17] = inv11; params[18] = inv12; params[19] = 0;
                params[20] = inv20; params[21] = inv21; params[22] = inv22; params[23] = 0;
                // BLAS metadata (as u32)
                paramsU32[24] = blasNodeOff;
                paramsU32[25] = blasTriOff;
                paramsU32[26] = blasTriCnt;
                paramsU32[27] = i; // materialIndex = object index
                // Range
                paramsU32[28] = instanceCount;  // instanceOffset
                paramsU32[29] = count;
                paramsU32[30] = 0;
                paramsU32[31] = 0;

                this._expandGroupInfos.push({
                    gpuBuffer: obj.gpuInstanceBuffer.resource.buffer,
                    fullTransform: obj.gpuInstanceFullTransform,
                    instanceOffset: instanceCount,
                    count,
                    params,
                });

                instanceCount += count;
                continue;
            }

            // ── Single instance (standard path) ──
            const m = obj.worldMatrix.internalMat4;
            const off = instanceCount * STRIDE;

            instanceData[off + 0] = m[0];  instanceData[off + 1] = m[4];
            instanceData[off + 2] = m[8];  instanceData[off + 3] = m[12];
            instanceData[off + 4] = m[1];  instanceData[off + 5] = m[5];
            instanceData[off + 6] = m[9];  instanceData[off + 7] = m[13];
            instanceData[off + 8]  = m[2];  instanceData[off + 9]  = m[6];
            instanceData[off + 10] = m[10]; instanceData[off + 11] = m[14];

            const det = m[0] * (m[5] * m[10] - m[6] * m[9])
                      - m[4] * (m[1] * m[10] - m[2] * m[9])
                      + m[8] * (m[1] * m[6]  - m[2] * m[5]);
            const invDet = det !== 0 ? 1.0 / det : 1.0;

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

            instanceData[off + 12] = inv00; instanceData[off + 13] = inv01;
            instanceData[off + 14] = inv02; instanceData[off + 15] = -(inv00 * tx + inv01 * ty + inv02 * tz);
            instanceData[off + 16] = inv10; instanceData[off + 17] = inv11;
            instanceData[off + 18] = inv12; instanceData[off + 19] = -(inv10 * tx + inv11 * ty + inv12 * tz);
            instanceData[off + 20] = inv20; instanceData[off + 21] = inv21;
            instanceData[off + 22] = inv22; instanceData[off + 23] = -(inv20 * tx + inv21 * ty + inv22 * tz);

            instanceDataU32[off + 24] = blasNodeOff;
            instanceDataU32[off + 25] = blasTriOff;
            instanceDataU32[off + 26] = blasTriCnt;
            instanceDataU32[off + 27] = i;

            instanceCount++;
        }

        this._totalInstances = instanceCount;

        // Compute total instance-triangles (each shared BLAS counted per instance)
        let totalInstTris = 0;
        for (let k = 0; k < instanceCount; k++) {
            totalInstTris += instanceDataU32[k * STRIDE + 26];
        }
        this.totalInstanceTriangles = totalInstTris;

        // Reuse GPU buffer if capacity is sufficient, only recreate if grown
        const byteSize = Math.max(instanceCount * STRIDE * 4, 4);
        if (!this._instanceBuffer || this._instanceCapacity < instanceCount) {
            this._instanceBuffer?.destroy();
            this._instanceBuffer = this._device.createBuffer({
                label: 'TLAS/Instances',
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            });
            this._instanceCapacity = instanceCount;
        }
        this._device.queue.writeBuffer(
            this._instanceBuffer, 0,
            instanceData.buffer, 0, instanceCount * STRIDE * 4,
        );
    }

    /**
     * Dispatch GPU compute passes that expand per-instance position/transform
     * data from a ComputeBuffer into full TLAS instance entries.
     * Called between _packInstances (CPU data uploaded) and TLAS refit passes.
     */
    private _dispatchInstanceExpansion(commandEncoder: GPUCommandEncoder): void {
        if (this._expandGroupInfos.length === 0) return;

        this._ensureExpandPipelines();

        // Grow param buffer pool as needed (one buffer per group — writeBuffer ordering)
        while (this._expandParamBufs.length < this._expandGroupInfos.length) {
            this._expandParamBufs.push(this._device.createBuffer({
                label: `TLAS/ExpandParams/${this._expandParamBufs.length}`,
                size: 128, // 32 floats
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }

        // Upload ALL params before recording any compute passes (writeBuffer is immediate)
        for (let g = 0; g < this._expandGroupInfos.length; g++) {
            this._device.queue.writeBuffer(this._expandParamBufs[g], 0, this._expandGroupInfos[g].params);
        }

        // Dispatch one compute pass per expansion group
        for (let g = 0; g < this._expandGroupInfos.length; g++) {
            const group = this._expandGroupInfos[g];
            const pipeline = group.fullTransform ? this._expandXfmPipeline! : this._expandPosPipeline!;

            const bg = this._device.createBindGroup({
                layout: this._expandBGL!,
                entries: [
                    { binding: 0, resource: { buffer: group.gpuBuffer } },
                    { binding: 1, resource: { buffer: this._instanceBuffer! } },
                    { binding: 2, resource: { buffer: this._expandParamBufs[g] } },
                ],
            });

            const wg = Math.ceil(group.count / 256);
            const pass = commandEncoder.beginComputePass({ label: `TLAS/Expand/${g}` });
            pass.setPipeline(pipeline);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(wg);
            pass.end();
        }
    }

    /**
     * Lazily create GPU pipelines for instance expansion.
     */
    private _ensureExpandPipelines(): void {
        if (this._expandPosPipeline) return;

        this._expandBGL = this._device.createBindGroupLayout({
            label: 'TLAS/ExpandBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } }, // srcData
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },           // instances
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },           // params
            ],
        });

        const module = this._device.createShaderModule({ code: instanceExpandShader });
        const layout = this._device.createPipelineLayout({ bindGroupLayouts: [this._expandBGL] });

        this._expandPosPipeline = this._device.createComputePipeline({
            label: 'TLAS/Expand/Positions',
            layout,
            compute: { module, entryPoint: 'expandPositions' },
        });

        this._expandXfmPipeline = this._device.createComputePipeline({
            label: 'TLAS/Expand/Transforms',
            layout,
            compute: { module, entryPoint: 'expandTransforms' },
        });
    }

    /**
     * Lazily create GPU pipelines for TLAS spatial sorting.
     */
    private _ensureTLASSortPipelines(): void {
        if (this._tlasSortKeysPipeline) return;

        this._ensureSortPipelines();

        // Morton key computation: reads instances, writes morton keys + indices
        this._tlasSortKeysBGL = this._device.createBindGroupLayout({
            label: 'TLAS/SortKeysBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const keysModule = this._device.createShaderModule({ code: tlasSortShader });
        this._tlasSortKeysPipeline = this._device.createComputePipeline({
            label: 'TLAS/SortKeys',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._tlasSortKeysBGL] }),
            compute: { module: keysModule, entryPoint: 'computeKeys' },
        });

        // Gather/reorder: reads sorted indices + source instances → writes to dest instances
        this._tlasGatherBGL = this._device.createBindGroupLayout({
            label: 'TLAS/GatherBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });
        const gatherModule = this._device.createShaderModule({ code: tlasGatherShader });
        this._tlasGatherPipeline = this._device.createComputePipeline({
            label: 'TLAS/Gather',
            layout: this._device.createPipelineLayout({ bindGroupLayouts: [this._tlasGatherBGL] }),
            compute: { module: gatherModule, entryPoint: 'main' },
        });
    }

    /**
     * Allocate sort buffers for TLAS spatial sorting (Morton keys, indices, histogram).
     */
    private _ensureTLASSortBuffers(count: number): void {
        if (count <= this._tlasSortCapacity) return;

        // Destroy old
        this._tlasMortonKeysA?.destroy();
        this._tlasMortonKeysB?.destroy();
        this._tlasMortonValsA?.destroy();
        this._tlasMortonValsB?.destroy();
        this._tlasHistogramBuf?.destroy();
        this._tlasSortParamsBuf?.destroy();
        this._tlasGatherParamsBuf?.destroy();
        this._instanceBufferSorted?.destroy();
        for (const b of this._tlasSortPassParamsBufs) b.destroy();
        this._tlasSortPassParamsBufs = [];

        const byteSize = count * 4;
        const mk = (label: string) => this._device.createBuffer({
            label, size: byteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._tlasMortonKeysA = mk('TLAS/Sort/KeysA');
        this._tlasMortonKeysB = mk('TLAS/Sort/KeysB');
        this._tlasMortonValsA = mk('TLAS/Sort/ValsA');
        this._tlasMortonValsB = mk('TLAS/Sort/ValsB');

        const wgCount = Math.ceil(count / 256);
        this._tlasHistogramBuf = this._device.createBuffer({
            label: 'TLAS/Sort/Histogram',
            size: 16 * wgCount * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        this._tlasSortParamsBuf = this._device.createBuffer({
            label: 'TLAS/Sort/Params',
            size: 32,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        // 8 sort passes need separate param buffers (writeBuffer ordering)
        for (let i = 0; i < 8; i++) {
            this._tlasSortPassParamsBufs.push(this._device.createBuffer({
                label: `TLAS/Sort/PassParams/${i}`,
                size: 16,
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }

        this._tlasGatherParamsBuf = this._device.createBuffer({
            label: 'TLAS/Sort/GatherParams',
            size: 16,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });

        this._instanceBufferSorted = this._device.createBuffer({
            label: 'TLAS/InstancesSorted',
            size: Math.max(count * BVHBuilder.INSTANCE_STRIDE * 4, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this._tlasSortCapacity = count;
    }

    /**
     * Sort TLAS instances by Morton code for better spatial coherence.
     * Runs between instance expansion and TLAS refit.
     */
    private _dispatchTLASSort(commandEncoder: GPUCommandEncoder, count: number): void {
        if (count <= 1) return;

        this._ensureTLASSortPipelines();
        this._ensureTLASSortBuffers(count);

        const wgCount = Math.ceil(count / 256);

        // Scene bounds — use generous fixed bounds; Morton codes only need rough spatial partitioning
        const sortParams = new Float32Array(8);
        const sortParamsU32 = new Uint32Array(sortParams.buffer);
        sortParamsU32[0] = count;
        sortParams[1] = -50.0;   // sceneMinX
        sortParams[2] = -50.0;   // sceneMinY
        sortParams[3] = -50.0;   // sceneMinZ
        sortParams[4] = 1.0 / 100.0;  // sceneExtX = 1/(max-min)
        sortParams[5] = 1.0 / 100.0;  // sceneExtY
        sortParams[6] = 1.0 / 100.0;  // sceneExtZ
        this._device.queue.writeBuffer(this._tlasSortParamsBuf!, 0, sortParams);

        // Step 1: Compute Morton codes from instance centroids
        const keysBG = this._device.createBindGroup({
            layout: this._tlasSortKeysBGL!,
            entries: [
                { binding: 0, resource: { buffer: this._instanceBuffer! } },
                { binding: 1, resource: { buffer: this._tlasMortonKeysA! } },
                { binding: 2, resource: { buffer: this._tlasMortonValsA! } },
                { binding: 3, resource: { buffer: this._tlasSortParamsBuf! } },
            ],
        });
        const keysPass = commandEncoder.beginComputePass({ label: 'TLAS/Sort/Keys' });
        keysPass.setPipeline(this._tlasSortKeysPipeline!);
        keysPass.setBindGroup(0, keysBG);
        keysPass.dispatchWorkgroups(wgCount);
        keysPass.end();

        // Step 2: Radix sort (8 passes × 4 bits = 32-bit keys)
        const buffers = [
            { keys: this._tlasMortonKeysA!, vals: this._tlasMortonValsA! },
            { keys: this._tlasMortonKeysB!, vals: this._tlasMortonValsB! },
        ];

        // Upload all sort pass params (writeBuffer is immediate)
        for (let pass = 0; pass < 8; pass++) {
            const sortPassParams = new Uint32Array([count, pass * 4, wgCount, 0]);
            this._device.queue.writeBuffer(this._tlasSortPassParamsBufs[pass], 0, sortPassParams);
        }

        for (let pass = 0; pass < 8; pass++) {
            const src = buffers[pass & 1];
            const dst = buffers[(pass + 1) & 1];

            const sortBG = this._device.createBindGroup({
                layout: this._sortBGL!,
                entries: [
                    { binding: 0, resource: { buffer: src.keys } },
                    { binding: 1, resource: { buffer: src.vals } },
                    { binding: 2, resource: { buffer: dst.keys } },
                    { binding: 3, resource: { buffer: dst.vals } },
                    { binding: 4, resource: { buffer: this._tlasHistogramBuf! } },
                    { binding: 5, resource: { buffer: this._tlasSortPassParamsBufs[pass] } },
                ],
            });

            const histPass = commandEncoder.beginComputePass({ label: `TLAS/Sort/Hist/${pass}` });
            histPass.setPipeline(this._sortHistogramPipeline!);
            histPass.setBindGroup(0, sortBG);
            histPass.dispatchWorkgroups(wgCount);
            histPass.end();

            const prefixPass = commandEncoder.beginComputePass({ label: `TLAS/Sort/Prefix/${pass}` });
            prefixPass.setPipeline(this._sortPrefixPipeline!);
            prefixPass.setBindGroup(0, sortBG);
            prefixPass.dispatchWorkgroups(1);
            prefixPass.end();

            const scatterPass = commandEncoder.beginComputePass({ label: `TLAS/Sort/Scatter/${pass}` });
            scatterPass.setPipeline(this._sortScatterPipeline!);
            scatterPass.setBindGroup(0, sortBG);
            scatterPass.dispatchWorkgroups(wgCount);
            scatterPass.end();
        }

        // After 8 passes (even), sorted result is in buffers[0] = (keysA, valsA)

        // Step 3: Gather instances into sorted order
        const gatherParams = new Uint32Array([count, 0, 0, 0]);
        this._device.queue.writeBuffer(this._tlasGatherParamsBuf!, 0, gatherParams);

        const gatherBG = this._device.createBindGroup({
            layout: this._tlasGatherBGL!,
            entries: [
                { binding: 0, resource: { buffer: this._tlasMortonValsA! } },
                { binding: 1, resource: { buffer: this._instanceBuffer! } },
                { binding: 2, resource: { buffer: this._instanceBufferSorted! } },
                { binding: 3, resource: { buffer: this._tlasGatherParamsBuf! } },
            ],
        });
        const gatherPass = commandEncoder.beginComputePass({ label: 'TLAS/Sort/Gather' });
        gatherPass.setPipeline(this._tlasGatherPipeline!);
        gatherPass.setBindGroup(0, gatherBG);
        gatherPass.dispatchWorkgroups(wgCount);
        gatherPass.end();

        // Copy sorted instances back to the main instance buffer
        commandEncoder.copyBufferToBuffer(
            this._instanceBufferSorted!, 0,
            this._instanceBuffer!, 0,
            count * BVHBuilder.INSTANCE_STRIDE * 4,
        );
    }

    /**
     * Compute BVH4 tree level layout for the TLAS.
     * Levels stored top-down: root at offset 0, leaves at the end.
     */
    private _computeTLASBvh4Layout(N: number): void {
        const levels: number[] = [];
        let count = N;
        while (count > 1) {
            count = Math.ceil(count / 4);
            levels.push(count);
        }
        if (N === 1) levels.push(1);

        // Reverse for top-down layout (root first)
        levels.reverse();

        this._tlasLevelLayout = [];
        let offset = 0;
        for (const lvl of levels) {
            this._tlasLevelLayout.push({ offset, count: lvl });
            offset += lvl;
        }
        this._tlasBvh4NodeCount = offset;

        // Allocate BVH4 TLAS node buffer (128 bytes per node)
        this._tlasBvh4Buffer?.destroy();
        this._tlasBvh4Buffer = this._device.createBuffer({
            label: 'TLAS/BVH4',
            size: Math.max(this._tlasBvh4NodeCount * 128, 4),
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });

        // Ensure enough param buffers for all build levels
        while (this._tlasBuildParamsBufs.length < levels.length) {
            this._tlasBuildParamsBufs.push(this._device.createBuffer({
                label: `TLAS/BVH4/BuildParams/${this._tlasBuildParamsBufs.length}`,
                size: 32, // BuildParams: 8 × u32
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            }));
        }
    }

    /**
     * Lazily create GPU pipelines for BVH4 TLAS construction.
     */
    private _ensureTLASBvh4Pipelines(): void {
        if (this._tlasBuildLeafPipeline) return;

        this._tlasBuildBGL = this._device.createBindGroupLayout({
            label: 'TLAS/BVH4/BuildBGL',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            ],
        });

        const module = this._device.createShaderModule({ code: tlasBuildShader });
        const layout = this._device.createPipelineLayout({ bindGroupLayouts: [this._tlasBuildBGL] });

        this._tlasBuildLeafPipeline = this._device.createComputePipeline({
            label: 'TLAS/BVH4/BuildLeaves',
            layout,
            compute: { module, entryPoint: 'buildLeaves' },
        });

        this._tlasBuildInternalPipeline = this._device.createComputePipeline({
            label: 'TLAS/BVH4/BuildInternal',
            layout,
            compute: { module, entryPoint: 'buildInternal' },
        });
    }

    /**
     * Dispatch BVH4 TLAS build: leaf nodes first, then internal levels bottom-up.
     * Each compute pass has an implicit barrier so child data is visible to parents.
     */
    private _dispatchTLASBvh4Build(commandEncoder: GPUCommandEncoder, count: number): void {
        const levels = this._tlasLevelLayout;
        if (levels.length === 0) return;

        const leafLevel = levels.length - 1;

        // Upload all level params before dispatching (writeBuffer is immediate)
        for (let i = 0; i < levels.length; i++) {
            const params = new Uint32Array(8);
            if (i === leafLevel) {
                params[0] = count;              // instanceCount
                params[1] = levels[i].offset;   // nodeOffset
                params[2] = 0;                  // childOffset (unused for leaves)
                params[3] = levels[i].count;    // nodeCount
                params[4] = 0;                  // childCount (unused for leaves)
            } else {
                const childLevel = i + 1;
                params[0] = count;
                params[1] = levels[i].offset;
                params[2] = levels[childLevel].offset;
                params[3] = levels[i].count;
                params[4] = levels[childLevel].count;
            }
            this._device.queue.writeBuffer(this._tlasBuildParamsBufs[i], 0, params);
        }

        // Dispatch leaf build
        {
            const bg = this._device.createBindGroup({
                layout: this._tlasBuildBGL!,
                entries: [
                    { binding: 0, resource: { buffer: this._instanceBuffer! } },
                    { binding: 1, resource: { buffer: this._bvh4NodeBuffer! } },
                    { binding: 2, resource: { buffer: this._tlasBvh4Buffer! } },
                    { binding: 3, resource: { buffer: this._tlasBuildParamsBufs[leafLevel] } },
                ],
            });
            const wg = Math.ceil(levels[leafLevel].count / 256);
            const pass = commandEncoder.beginComputePass({ label: 'TLAS/BVH4/BuildLeaves' });
            pass.setPipeline(this._tlasBuildLeafPipeline!);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(wg);
            pass.end();
        }

        // Dispatch internal levels bottom-up (from leafLevel-1 down to 0)
        for (let i = leafLevel - 1; i >= 0; i--) {
            const bg = this._device.createBindGroup({
                layout: this._tlasBuildBGL!,
                entries: [
                    { binding: 0, resource: { buffer: this._instanceBuffer! } },
                    { binding: 1, resource: { buffer: this._bvh4NodeBuffer! } },
                    { binding: 2, resource: { buffer: this._tlasBvh4Buffer! } },
                    { binding: 3, resource: { buffer: this._tlasBuildParamsBufs[i] } },
                ],
            });
            const wg = Math.ceil(levels[i].count / 256);
            const pass = commandEncoder.beginComputePass({ label: `TLAS/BVH4/BuildInternal/${i}` });
            pass.setPipeline(this._tlasBuildInternalPipeline!);
            pass.setBindGroup(0, bg);
            pass.dispatchWorkgroups(wg);
            pass.end();
        }
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
     * Build BLAS on CPU using SAH (Surface Area Heuristic) with binned splits.
     * Produces much better BVH quality than midpoint split for complex geometry.
     */
    private _buildCPUBLAS(): void {
        const SAH_BINS = 12;
        const SAH_TRAVERSAL_COST = 1.0;
        const SAH_INTERSECT_COST = 1.0;
        const MAX_LEAF_SIZE = 4;

        // Pre-calculate max node count (2n - 1 per geometry)
        let maxTotalNodes = 0;
        for (const [, entry] of this._blasEntries) {
            if (entry.triangleCount > 0) {
                maxTotalNodes += 2 * entry.triangleCount - 1;
            }
        }

        // Allocate combined BLAS node buffer (may use fewer nodes than max)
        const combinedBuf = new ArrayBuffer(Math.max(maxTotalNodes * 32, 4));
        const combinedF32 = new Float32Array(combinedBuf);
        const combinedI32 = new Int32Array(combinedBuf);

        let globalNodeOffset = 0;

        for (const [geoId, entry] of this._blasEntries) {
            const n = entry.triangleCount;
            if (n === 0) continue;

            entry.nodeOffset = globalNodeOffset;

            // Compute per-triangle AABBs and centroids
            const triMinX = new Float32Array(n), triMinY = new Float32Array(n), triMinZ = new Float32Array(n);
            const triMaxX = new Float32Array(n), triMaxY = new Float32Array(n), triMaxZ = new Float32Array(n);
            const centX = new Float32Array(n), centY = new Float32Array(n), centZ = new Float32Array(n);

            let geom: any = null;
            for (const [g, key] of this._geomToBLASKey) {
                if (key === geoId) { geom = g; break; }
            }

            if (geom && geom.vertices && geom.indices) {
                const verts = geom.vertices;
                const indices = geom.indices;
                const VS = 9;
                for (let t = 0; t < n; t++) {
                    const i0 = indices[t * 3], i1 = indices[t * 3 + 1], i2 = indices[t * 3 + 2];
                    const ax = verts[i0 * VS], ay = verts[i0 * VS + 1], az = verts[i0 * VS + 2];
                    const bx = verts[i1 * VS], by = verts[i1 * VS + 1], bz = verts[i1 * VS + 2];
                    const cx = verts[i2 * VS], cy = verts[i2 * VS + 1], cz = verts[i2 * VS + 2];
                    triMinX[t] = Math.min(ax, bx, cx); triMinY[t] = Math.min(ay, by, cy); triMinZ[t] = Math.min(az, bz, cz);
                    triMaxX[t] = Math.max(ax, bx, cx); triMaxY[t] = Math.max(ay, by, cy); triMaxZ[t] = Math.max(az, bz, cz);
                    centX[t] = (triMinX[t] + triMaxX[t]) * 0.5;
                    centY[t] = (triMinY[t] + triMaxY[t]) * 0.5;
                    centZ[t] = (triMinZ[t] + triMaxZ[t]) * 0.5;
                }
            } else {
                const lb = this._blasLocalBounds.get(geoId);
                const mn = lb ? lb.min : [-1, -1, -1], mx = lb ? lb.max : [1, 1, 1];
                for (let t = 0; t < n; t++) {
                    triMinX[t] = mn[0]; triMinY[t] = mn[1]; triMinZ[t] = mn[2];
                    triMaxX[t] = mx[0]; triMaxY[t] = mx[1]; triMaxZ[t] = mx[2];
                    centX[t] = (mn[0] + mx[0]) * 0.5;
                    centY[t] = (mn[1] + mx[1]) * 0.5;
                    centZ[t] = (mn[2] + mx[2]) * 0.5;
                }
            }

            // Working index array — indices into per-geometry triangle arrays
            const indices = new Uint32Array(n);
            for (let i = 0; i < n; i++) indices[i] = i;

            let nextNode = 0;
            let nextTriSlot = 0;
            const reorderMap = new Uint32Array(n);

            const surfaceArea = (mnx: number, mny: number, mnz: number, mxx: number, mxy: number, mxz: number): number => {
                const dx = mxx - mnx, dy = mxy - mny, dz = mxz - mnz;
                return 2.0 * (dx * dy + dy * dz + dz * dx);
            };

            // Pre-allocated SAH scratch arrays (reused across all recursive calls — eliminates GC pressure)
            const sahBinCount = new Uint32Array(SAH_BINS);
            const sahBinMinX = new Float32Array(SAH_BINS), sahBinMinY = new Float32Array(SAH_BINS), sahBinMinZ = new Float32Array(SAH_BINS);
            const sahBinMaxX = new Float32Array(SAH_BINS), sahBinMaxY = new Float32Array(SAH_BINS), sahBinMaxZ = new Float32Array(SAH_BINS);
            const sahLeftCount = new Uint32Array(SAH_BINS - 1);
            const sahLeftSA = new Float32Array(SAH_BINS - 1);

            // Build SAH tree recursively. Operates on indices[start..start+count).
            const buildSAH = (start: number, count: number): number => {
                const nodeIdx = nextNode++;
                const globalIdx = globalNodeOffset + nodeIdx;
                const off = globalIdx * 8;

                // Compute bounds of all triangles in this subtree
                let bMinX = Infinity, bMinY = Infinity, bMinZ = Infinity;
                let bMaxX = -Infinity, bMaxY = -Infinity, bMaxZ = -Infinity;
                for (let i = start; i < start + count; i++) {
                    const t = indices[i];
                    bMinX = Math.min(bMinX, triMinX[t]); bMinY = Math.min(bMinY, triMinY[t]); bMinZ = Math.min(bMinZ, triMinZ[t]);
                    bMaxX = Math.max(bMaxX, triMaxX[t]); bMaxY = Math.max(bMaxY, triMaxY[t]); bMaxZ = Math.max(bMaxZ, triMaxZ[t]);
                }

                if (count <= MAX_LEAF_SIZE) {
                    // Multi-triangle leaf: store contiguous triangle range
                    const leafTriStart = nextTriSlot;
                    for (let i = start; i < start + count; i++) {
                        reorderMap[nextTriSlot] = indices[i];
                        nextTriSlot++;
                    }
                    combinedF32[off + 0] = bMinX; combinedF32[off + 1] = bMinY; combinedF32[off + 2] = bMinZ;
                    combinedI32[off + 3] = -(leafTriStart + 1); // negative start index
                    combinedF32[off + 4] = bMaxX; combinedF32[off + 5] = bMaxY; combinedF32[off + 6] = bMaxZ;
                    combinedI32[off + 7] = count; // triangle count in leaf
                    return nodeIdx;
                }

                // Find best SAH split across 3 axes using binned approach
                const parentSA = surfaceArea(bMinX, bMinY, bMinZ, bMaxX, bMaxY, bMaxZ);
                const leafCost = SAH_INTERSECT_COST * count;
                let bestCost = leafCost;
                let bestAxis = -1;
                let bestSplit = -1;

                // Cache per-axis centroid ranges to avoid redundant scans
                const axisCMin = [0, 0, 0], axisCMax = [0, 0, 0];

                for (let axis = 0; axis < 3; axis++) {
                    const cent = axis === 0 ? centX : axis === 1 ? centY : centZ;
                    let cMin = Infinity, cMax = -Infinity;
                    for (let i = start; i < start + count; i++) {
                        const c = cent[indices[i]];
                        if (c < cMin) cMin = c;
                        if (c > cMax) cMax = c;
                    }
                    axisCMin[axis] = cMin;
                    axisCMax[axis] = cMax;

                    if (cMax - cMin < 1e-10) continue;

                    const scale = SAH_BINS / (cMax - cMin);

                    // Clear bins
                    sahBinCount.fill(0);
                    sahBinMinX.fill(Infinity); sahBinMinY.fill(Infinity); sahBinMinZ.fill(Infinity);
                    sahBinMaxX.fill(-Infinity); sahBinMaxY.fill(-Infinity); sahBinMaxZ.fill(-Infinity);

                    // Fill bins
                    for (let i = start; i < start + count; i++) {
                        const t = indices[i];
                        let b = ((cent[t] - cMin) * scale) | 0;
                        if (b >= SAH_BINS) b = SAH_BINS - 1;
                        sahBinCount[b]++;
                        if (triMinX[t] < sahBinMinX[b]) sahBinMinX[b] = triMinX[t];
                        if (triMinY[t] < sahBinMinY[b]) sahBinMinY[b] = triMinY[t];
                        if (triMinZ[t] < sahBinMinZ[b]) sahBinMinZ[b] = triMinZ[t];
                        if (triMaxX[t] > sahBinMaxX[b]) sahBinMaxX[b] = triMaxX[t];
                        if (triMaxY[t] > sahBinMaxY[b]) sahBinMaxY[b] = triMaxY[t];
                        if (triMaxZ[t] > sahBinMaxZ[b]) sahBinMaxZ[b] = triMaxZ[t];
                    }

                    // Sweep from left, accumulating bounds and counts
                    let lMinX = Infinity, lMinY = Infinity, lMinZ = Infinity;
                    let lMaxX = -Infinity, lMaxY = -Infinity, lMaxZ = -Infinity;
                    let lCount = 0;
                    for (let i = 0; i < SAH_BINS - 1; i++) {
                        if (sahBinMinX[i] < lMinX) lMinX = sahBinMinX[i];
                        if (sahBinMinY[i] < lMinY) lMinY = sahBinMinY[i];
                        if (sahBinMinZ[i] < lMinZ) lMinZ = sahBinMinZ[i];
                        if (sahBinMaxX[i] > lMaxX) lMaxX = sahBinMaxX[i];
                        if (sahBinMaxY[i] > lMaxY) lMaxY = sahBinMaxY[i];
                        if (sahBinMaxZ[i] > lMaxZ) lMaxZ = sahBinMaxZ[i];
                        lCount += sahBinCount[i];
                        sahLeftCount[i] = lCount;
                        sahLeftSA[i] = lCount > 0 ? surfaceArea(lMinX, lMinY, lMinZ, lMaxX, lMaxY, lMaxZ) : 0;
                    }

                    // Sweep from right
                    let rMinX = Infinity, rMinY = Infinity, rMinZ = Infinity;
                    let rMaxX = -Infinity, rMaxY = -Infinity, rMaxZ = -Infinity;
                    let rCount = 0;
                    for (let i = SAH_BINS - 1; i >= 1; i--) {
                        if (sahBinMinX[i] < rMinX) rMinX = sahBinMinX[i];
                        if (sahBinMinY[i] < rMinY) rMinY = sahBinMinY[i];
                        if (sahBinMinZ[i] < rMinZ) rMinZ = sahBinMinZ[i];
                        if (sahBinMaxX[i] > rMaxX) rMaxX = sahBinMaxX[i];
                        if (sahBinMaxY[i] > rMaxY) rMaxY = sahBinMaxY[i];
                        if (sahBinMaxZ[i] > rMaxZ) rMaxZ = sahBinMaxZ[i];
                        rCount += sahBinCount[i];
                        const rightSA = rCount > 0 ? surfaceArea(rMinX, rMinY, rMinZ, rMaxX, rMaxY, rMaxZ) : 0;

                        const cost = SAH_TRAVERSAL_COST + SAH_INTERSECT_COST * (sahLeftCount[i - 1] * sahLeftSA[i - 1] + rCount * rightSA) / parentSA;
                        if (cost < bestCost) {
                            bestCost = cost;
                            bestAxis = axis;
                            bestSplit = i;
                        }
                    }
                }

                // No good split found — fallback: split along longest axis at midpoint
                if (bestAxis === -1) {
                    const dx = bMaxX - bMinX, dy = bMaxY - bMinY, dz = bMaxZ - bMinZ;
                    bestAxis = dx >= dy && dx >= dz ? 0 : dy >= dz ? 1 : 2;
                    const cent = bestAxis === 0 ? centX : bestAxis === 1 ? centY : centZ;
                    const mid = (axisCMin[bestAxis] + axisCMax[bestAxis]) * 0.5;

                    let l = start, r = start + count - 1;
                    while (l <= r) {
                        if (cent[indices[l]] < mid) { l++; }
                        else { const tmp = indices[l]; indices[l] = indices[r]; indices[r] = tmp; r--; }
                    }
                    let leftCount2 = l - start;
                    if (leftCount2 === 0 || leftCount2 === count) leftCount2 = count >> 1;

                    const leftChild = buildSAH(start, leftCount2);
                    const rightChild = buildSAH(start + leftCount2, count - leftCount2);
                    combinedF32[off + 0] = bMinX; combinedF32[off + 1] = bMinY; combinedF32[off + 2] = bMinZ;
                    combinedI32[off + 3] = leftChild;
                    combinedF32[off + 4] = bMaxX; combinedF32[off + 5] = bMaxY; combinedF32[off + 6] = bMaxZ;
                    combinedI32[off + 7] = rightChild;
                    return nodeIdx;
                }

                // Partition indices according to best bin split (use cached cMin/cMax)
                {
                    const cent = bestAxis === 0 ? centX : bestAxis === 1 ? centY : centZ;
                    const cMin = axisCMin[bestAxis], cMax = axisCMax[bestAxis];
                    const scale = SAH_BINS / (cMax - cMin);

                    let l = start, r = start + count - 1;
                    while (l <= r) {
                        let b = ((cent[indices[l]] - cMin) * scale) | 0;
                        if (b >= SAH_BINS) b = SAH_BINS - 1;
                        if (b < bestSplit) { l++; }
                        else { const tmp = indices[l]; indices[l] = indices[r]; indices[r] = tmp; r--; }
                    }
                    let leftCount2 = l - start;
                    if (leftCount2 === 0 || leftCount2 === count) leftCount2 = count >> 1;

                    const leftChild = buildSAH(start, leftCount2);
                    const rightChild = buildSAH(start + leftCount2, count - leftCount2);
                    combinedF32[off + 0] = bMinX; combinedF32[off + 1] = bMinY; combinedF32[off + 2] = bMinZ;
                    combinedI32[off + 3] = leftChild;
                    combinedF32[off + 4] = bMaxX; combinedF32[off + 5] = bMaxY; combinedF32[off + 6] = bMaxZ;
                    combinedI32[off + 7] = rightChild;
                    return nodeIdx;
                }
            };

            buildSAH(0, n);
            entry.nodeCount = nextNode;
            globalNodeOffset += nextNode;

            // Reorder triangle data so leaf triangles are contiguous in memory
            const triData = this._blasTriData.get(geoId);
            if (triData && this._triangleBuffer) {
                const stride = BVHBuilder.TRI_STRIDE_FLOATS;
                const reordered = new Float32Array(n * stride);
                for (let i = 0; i < n; i++) {
                    const oldIdx = reorderMap[i];
                    reordered.set(
                        triData.subarray(oldIdx * stride, oldIdx * stride + stride),
                        i * stride,
                    );
                }
                const byteOffset = entry.triangleOffset * stride * 4;
                this._device.queue.writeBuffer(this._triangleBuffer, byteOffset, reordered);

                // For dynamic BLAS: store reordered data and duplicate triangle copies
                if (this._dynamicBLASCopies.has(geoId)) {
                    this._blasTriData.set(geoId + '_reordered', reordered.slice());
                    const dynCopies = this._dynamicBLASCopies.get(geoId)!;
                    for (let k = 1; k < dynCopies; k++) {
                        const copyOffset = (entry.triangleOffset + k * n) * stride * 4;
                        this._device.queue.writeBuffer(this._triangleBuffer, copyOffset, reordered);
                    }
                }
            }
        }

        this.totalBLASNodes = globalNodeOffset;

        // Upload to GPU (trim to actual size used)
        this._blasNodeBuffer?.destroy();
        const actualSize = Math.max(globalNodeOffset * 32, 4);
        this._blasNodeBuffer = this._device.createBuffer({
            label: 'BVH/BLASNodes/CPU',
            size: actualSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._blasNodeBuffer, 0, combinedBuf, 0, actualSize);

        // ── Collapse binary BVH to BVH4 (4-wide) ──
        // Each BVH4 node = 128 bytes (32 floats = 8 vec4f):
        //   [0-3]   childMinX[0..3]   [4-7]   childMaxX[0..3]
        //   [8-11]  childMinY[0..3]   [12-15] childMaxY[0..3]
        //   [16-19] childMinZ[0..3]   [20-23] childMaxZ[0..3]
        //   [24-27] children[0..3] (i32: <0 = leaf, >=0 = BVH4 node)
        //   [28-31] triCounts[0..3] (u32: triangle count for leaves)
        const BVH4_STRIDE = 32;
        const bvh4Data = new Float32Array(Math.max(globalNodeOffset * BVH4_STRIDE, 4));
        const bvh4I32 = new Int32Array(bvh4Data.buffer);
        const bvh4U32 = new Uint32Array(bvh4Data.buffer);
        let bvh4TotalNodes = 0;

        for (const [, entry] of this._blasEntries) {
            if (entry.nodeCount === 0) continue;
            const binBase = entry.nodeOffset * 8; // binary node start offset (in floats)
            const bvh4Start = bvh4TotalNodes;
            let nextBVH4 = 0;

            const isLeaf = (li: number) => combinedI32[binBase + li * 8 + 3] < 0;
            const getLC  = (li: number) => combinedI32[binBase + li * 8 + 3];
            const getRC  = (li: number) => combinedI32[binBase + li * 8 + 7];
            const mnX = (li: number) => combinedF32[binBase + li * 8 + 0];
            const mnY = (li: number) => combinedF32[binBase + li * 8 + 1];
            const mnZ = (li: number) => combinedF32[binBase + li * 8 + 2];
            const mxX = (li: number) => combinedF32[binBase + li * 8 + 4];
            const mxY = (li: number) => combinedF32[binBase + li * 8 + 5];
            const mxZ = (li: number) => combinedF32[binBase + li * 8 + 6];

            const initBVH4 = (): number => {
                const idx = bvh4Start + nextBVH4++;
                const o = idx * BVH4_STRIDE;
                for (let i = 0; i < 4; i++) {
                    bvh4Data[o + i]      =  1e30;  // childMinX
                    bvh4Data[o + 4 + i]  = -1e30;  // childMaxX
                    bvh4Data[o + 8 + i]  =  1e30;  // childMinY
                    bvh4Data[o + 12 + i] = -1e30;  // childMaxY
                    bvh4Data[o + 16 + i] =  1e30;  // childMinZ
                    bvh4Data[o + 20 + i] = -1e30;  // childMaxZ
                    bvh4I32[o + 24 + i]  = -1;  // empty leaf sentinel (0 triangles, never traversed)
                    bvh4U32[o + 28 + i]  = 0;
                }
                return idx;
            };

            const setChild = (nodeIdx: number, slot: number, binLocalIdx: number, bvh4Idx: number) => {
                const o = nodeIdx * BVH4_STRIDE;
                bvh4Data[o + slot]      = mnX(binLocalIdx);
                bvh4Data[o + 4 + slot]  = mxX(binLocalIdx);
                bvh4Data[o + 8 + slot]  = mnY(binLocalIdx);
                bvh4Data[o + 12 + slot] = mxY(binLocalIdx);
                bvh4Data[o + 16 + slot] = mnZ(binLocalIdx);
                bvh4Data[o + 20 + slot] = mxZ(binLocalIdx);
                if (isLeaf(binLocalIdx)) {
                    bvh4I32[o + 24 + slot] = getLC(binLocalIdx);  // -(triStart+1)
                    bvh4U32[o + 28 + slot] = getRC(binLocalIdx);  // triCount
                } else {
                    bvh4I32[o + 24 + slot] = bvh4Idx;  // global BVH4 node index
                    bvh4U32[o + 28 + slot] = 0;
                }
            };

            const convert = (li: number): number => {
                const bvh4Idx = initBVH4();

                if (isLeaf(li)) {
                    setChild(bvh4Idx, 0, li, 0);
                    return bvh4Idx;
                }

                // Collect grandchildren by expanding one level
                // Use local array — shared scratch was corrupted by recursive calls
                const gc: number[] = [];
                const L = getLC(li), R = getRC(li);

                if (isLeaf(L)) { gc.push(L); }
                else { gc.push(getLC(L), getRC(L)); }

                if (isLeaf(R)) { gc.push(R); }
                else { gc.push(getLC(R), getRC(R)); }

                for (let i = 0; i < gc.length && i < 4; i++) {
                    const g = gc[i];
                    if (isLeaf(g)) {
                        setChild(bvh4Idx, i, g, 0);
                    } else {
                        const childBVH4 = convert(g);
                        setChild(bvh4Idx, i, g, childBVH4);
                    }
                }
                return bvh4Idx;
            };

            convert(0);
            entry.nodeOffset = bvh4Start;
            entry.nodeCount = nextBVH4;
            bvh4TotalNodes += nextBVH4;
        }

        // ── Duplicate BVH4 nodes for dynamic BLAS geometries ──
        let extraBvh4Nodes = 0;
        for (const [geoId, entry] of this._blasEntries) {
            const copies = this._dynamicBLASCopies.get(geoId) || 1;
            if (copies > 1) extraBvh4Nodes += (copies - 1) * entry.nodeCount;
        }

        const finalBvh4Total = bvh4TotalNodes + extraBvh4Nodes;
        let finalBvh4Data: Float32Array;
        let finalBvh4I32: Int32Array;
        if (extraBvh4Nodes > 0) {
            finalBvh4Data = new Float32Array(finalBvh4Total * BVH4_STRIDE);
            finalBvh4Data.set(bvh4Data.subarray(0, bvh4TotalNodes * BVH4_STRIDE));
            finalBvh4I32 = new Int32Array(finalBvh4Data.buffer);
        } else {
            finalBvh4Data = bvh4Data;
            finalBvh4I32 = bvh4I32;
        }

        // Helper: compute BVH4 tree depth from root
        const bvh4DepthFn = (rootIdx: number): number => {
            const o = rootIdx * BVH4_STRIDE;
            let maxD = 0;
            for (let c = 0; c < 4; c++) {
                const child = finalBvh4I32[o + 24 + c];
                if (child >= 0) maxD = Math.max(maxD, bvh4DepthFn(child) + 1);
            }
            return maxD;
        };

        let copyNodeOffset = bvh4TotalNodes;
        // Also prepare original triangle data for origTriangleBuffer
        let origTriDataAll: Float32Array | null = null;
        let origTriDataOffset = 0;

        for (const [geoId, entry] of this._blasEntries) {
            if (!this._dynamicBLASCopies.has(geoId)) continue;
            const copies = this._dynamicBLASCopies.get(geoId)!;

            const nodeCount = entry.nodeCount;
            const triCount = entry.triangleCount;
            const srcStart = entry.nodeOffset * BVH4_STRIDE;

            const copyNodeOffsets: number[] = [entry.nodeOffset];
            const copyTriOffsets: number[] = [entry.triangleOffset];

            for (let k = 1; k < copies; k++) {
                const dstStart = copyNodeOffset * BVH4_STRIDE;
                // Copy node data
                finalBvh4Data.set(
                    finalBvh4Data.subarray(srcStart, srcStart + nodeCount * BVH4_STRIDE),
                    dstStart,
                );
                // Remap internal children (positive child indices)
                for (let n = 0; n < nodeCount; n++) {
                    for (let c = 0; c < 4; c++) {
                        const off = dstStart + n * BVH4_STRIDE + 24 + c;
                        const child = finalBvh4I32[off];
                        if (child >= 0) {
                            finalBvh4I32[off] = child - entry.nodeOffset + copyNodeOffset;
                        }
                    }
                }

                copyNodeOffsets.push(copyNodeOffset);
                copyTriOffsets.push(entry.triangleOffset + k * triCount);
                copyNodeOffset += nodeCount;
            }

            // Get reordered original triangle data
            const reordered = this._blasTriData.get(geoId + '_reordered');

            this._dynamicBLASInfos.set(geoId, {
                copies,
                nodeCountPerCopy: nodeCount,
                triCountPerCopy: triCount,
                copyNodeOffsets,
                copyTriOffsets,
                maxDepth: bvh4DepthFn(entry.nodeOffset) + 1,
                origTriangleData: reordered || new Float32Array(0),
            });

            // Accumulate original triangle data for GPU buffer
            if (reordered) {
                if (!origTriDataAll) {
                    origTriDataAll = new Float32Array(reordered.length);
                }
                origTriDataAll.set(reordered, origTriDataOffset);
                origTriDataOffset += reordered.length;
            }
        }

        this.totalBVH4Nodes = finalBvh4Total;

        // Upload BVH4 buffer
        this._bvh4NodeBuffer?.destroy();
        const bvh4ByteSize = Math.max(finalBvh4Total * 128, 4);
        this._bvh4NodeBuffer = this._device.createBuffer({
            label: 'BVH/BVH4Nodes',
            size: bvh4ByteSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        });
        this._device.queue.writeBuffer(this._bvh4NodeBuffer, 0, finalBvh4Data.buffer, 0, bvh4ByteSize);

        // Upload original triangle data buffer for deform shader
        this._origTriangleBuffer?.destroy();
        this._origTriangleBuffer = null;
        if (origTriDataAll && origTriDataOffset > 0) {
            this._origTriangleBuffer = this._device.createBuffer({
                label: 'BVH/OrigTriangles',
                size: origTriDataOffset * 4,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
            this._device.queue.writeBuffer(this._origTriangleBuffer, 0, origTriDataAll.buffer, 0, origTriDataOffset * 4);
        }
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
     * Lazily create all GPU compute pipelines for BVH construction.
     */
    /**
     * Lazily create radix sort pipelines (shared by BLAS and TLAS sorting).
     */
    private _ensureSortPipelines(): void {
        if (this._sortBGL) return;

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
    }

    private _ensureBLASPipelines(): void {
        if (this._mortonPipeline) return;

        this._ensureSortPipelines();

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

    private _materialStaging: Float32Array | null = null;

    private _updateMaterialBuffer(materials: PathTracerMaterial[]): void {
        const floatsPerMat = PathTracerMaterial.GPU_STRIDE / 4; // 16
        const needed = Math.max(materials.length * floatsPerMat, 1);
        if (!this._materialStaging || this._materialStaging.length < needed) {
            this._materialStaging = new Float32Array(needed);
        }
        const staging = this._materialStaging;
        for (let i = 0; i < materials.length; i++) {
            materials[i].packInto(staging, i * floatsPerMat);
        }
        const byteSize = Math.max(materials.length * floatsPerMat * 4, 4);
        if (!this._materialBuffer || this.materialCount < materials.length) {
            this._materialBuffer?.destroy();
            this._materialBuffer = this._device.createBuffer({
                label: 'BVH/Materials',
                size: byteSize,
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });
        }
        this._device.queue.writeBuffer(this._materialBuffer, 0, staging.buffer, 0, byteSize);
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
        // TLAS sort buffers
        this._tlasMortonKeysA?.destroy();
        this._tlasMortonKeysB?.destroy();
        this._tlasMortonValsA?.destroy();
        this._tlasMortonValsB?.destroy();
        this._tlasHistogramBuf?.destroy();
        this._tlasSortParamsBuf?.destroy();
        this._tlasGatherParamsBuf?.destroy();
        this._instanceBufferSorted?.destroy();
        for (const b of this._tlasSortPassParamsBufs) b.destroy();
        // BVH4 TLAS buffers
        this._tlasBvh4Buffer?.destroy();
        for (const buf of this._tlasBuildParamsBufs) buf.destroy();
        for (const buf of this._pendingScratchBuffers) {
            buf.destroy();
        }
        this._pendingScratchBuffers.length = 0;
    }
}
