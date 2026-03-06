import { Geometry } from "../buffers/Geometry";
import { Renderable } from "../objects/Renderable";
import { Scene } from "../objects/Scene";
import { InstancedGeometry } from "../geometries/InstancedGeometry";
import { PathTracerMaterial } from "./PathTracerMaterial";

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

        for (const [geom, data] of geometryMap) {
            const entry: BLASEntry = {
                geometryId: (geom as any).uuid ?? String(offset),
                triangleOffset: offset / BVHBuilder.TRI_STRIDE_FLOATS,
                triangleCount: data.triCount,
                nodeOffset: 0,
                nodeCount: 0,
            };
            allTriangles.set(data.triangles, offset);
            offset += data.triCount * BVHBuilder.TRI_STRIDE_FLOATS;
            this._blasEntries.set(entry.geometryId, entry);
        }

        this.totalTriangleCount = totalTris;

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
    }
}
