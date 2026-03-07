import { Geometry } from "../buffers/Geometry";
import { Material } from "../materials/Material";
import { Renderable } from "../objects/Renderable";
import { Object3D } from "../objects/Object3D";
import { mat4 } from "gl-matrix";

interface GLTFAccessor {
    bufferView: number;
    byteOffset?: number;
    componentType: number;
    count: number;
    type: string;
    max?: number[];
    min?: number[];
}

interface GLTFBufferView {
    buffer: number;
    byteLength: number;
    byteOffset?: number;
    byteStride?: number;
    target?: number;
}

interface GLTFPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    material?: number;
}

interface GLTFMesh {
    name?: string;
    primitives: GLTFPrimitive[];
}

interface GLTFNode {
    name?: string;
    mesh?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

interface GLTFMaterialInfo {
    name?: string;
    baseColor: [number, number, number, number];
    metallic: number;
    roughness: number;
    doubleSided: boolean;
}

export interface GLTFResult {
    scene: Object3D;
    geometries: Geometry[];
    materials: GLTFMaterialInfo[];
}

const COMPONENT_TYPE_SIZES: Record<number, number> = {
    5120: 1, // BYTE
    5121: 1, // UNSIGNED_BYTE
    5122: 2, // SHORT
    5123: 2, // UNSIGNED_SHORT
    5125: 4, // UNSIGNED_INT
    5126: 4, // FLOAT
};

const TYPE_COUNTS: Record<string, number> = {
    SCALAR: 1,
    VEC2: 2,
    VEC3: 3,
    VEC4: 4,
    MAT2: 4,
    MAT3: 9,
    MAT4: 16,
};

class GLTFLoader {
    private gltf: any;
    private buffers: ArrayBuffer[] = [];
    private baseUrl: string = "";

    async load(url: string, defaultMaterial: Material): Promise<GLTFResult> {
        this.baseUrl = url.substring(0, url.lastIndexOf("/") + 1);

        const response = await fetch(url);
        this.gltf = await response.json();

        await this.loadBuffers();

        const materials = this.parseMaterials();
        const geometries = this.parseMeshes();
        const scene = this.buildScene(geometries, defaultMaterial);

        return { scene, geometries, materials };
    }

    private async loadBuffers(): Promise<void> {
        const bufferDefs = this.gltf.buffers || [];
        this.buffers = await Promise.all(
            bufferDefs.map(async (buf: any) => {
                const resp = await fetch(this.baseUrl + buf.uri);
                return resp.arrayBuffer();
            })
        );
    }

    private getAccessorData(accessorIndex: number): { data: Float32Array | Uint16Array | Uint32Array; count: number; componentCount: number } {
        const accessor: GLTFAccessor = this.gltf.accessors[accessorIndex];
        const bufferView: GLTFBufferView = this.gltf.bufferViews[accessor.bufferView];
        const buffer = this.buffers[bufferView.buffer];

        const componentSize = COMPONENT_TYPE_SIZES[accessor.componentType];
        const componentCount = TYPE_COUNTS[accessor.type];
        const byteOffset = (bufferView.byteOffset || 0) + (accessor.byteOffset || 0);
        const byteStride = bufferView.byteStride || 0;

        if (byteStride && byteStride !== componentSize * componentCount) {
            // Strided access — need to unpack
            const totalComponents = accessor.count * componentCount;
            const out = accessor.componentType === 5126
                ? new Float32Array(totalComponents)
                : accessor.componentType === 5125
                    ? new Uint32Array(totalComponents)
                    : new Uint16Array(totalComponents);

            const view = new DataView(buffer);
            for (let i = 0; i < accessor.count; i++) {
                const elementOffset = byteOffset + i * byteStride;
                for (let c = 0; c < componentCount; c++) {
                    const off = elementOffset + c * componentSize;
                    if (accessor.componentType === 5126) {
                        (out as Float32Array)[i * componentCount + c] = view.getFloat32(off, true);
                    } else if (accessor.componentType === 5125) {
                        (out as Uint32Array)[i * componentCount + c] = view.getUint32(off, true);
                    } else {
                        (out as Uint16Array)[i * componentCount + c] = view.getUint16(off, true);
                    }
                }
            }
            return { data: out, count: accessor.count, componentCount };
        }

        // Tightly packed — direct typed array view
        const elementCount = accessor.count * componentCount;
        let data: Float32Array | Uint16Array | Uint32Array;
        if (accessor.componentType === 5126) {
            data = new Float32Array(buffer, byteOffset, elementCount);
        } else if (accessor.componentType === 5125) {
            data = new Uint32Array(buffer, byteOffset, elementCount);
        } else {
            data = new Uint16Array(buffer, byteOffset, elementCount);
        }

        return { data, count: accessor.count, componentCount };
    }

    private parseMaterials(): GLTFMaterialInfo[] {
        const defs = this.gltf.materials || [];
        return defs.map((m: any) => {
            const pbr = m.pbrMetallicRoughness || {};
            const baseColor = pbr.baseColorFactor || [1, 1, 1, 1];
            return {
                name: m.name,
                baseColor: baseColor as [number, number, number, number],
                metallic: pbr.metallicFactor ?? 1.0,
                roughness: pbr.roughnessFactor ?? 1.0,
                doubleSided: m.doubleSided ?? false,
            };
        });
    }

    private parseMeshes(): Geometry[] {
        const meshDefs: GLTFMesh[] = this.gltf.meshes || [];
        const geometries: Geometry[] = [];

        for (const meshDef of meshDefs) {
            for (const prim of meshDef.primitives) {
                const geo = this.parsePrimitive(prim);
                geometries.push(geo);
            }
        }

        return geometries;
    }

    private parsePrimitive(prim: GLTFPrimitive): Geometry {
        const posAccessor = prim.attributes.POSITION;
        const posData = this.getAccessorData(posAccessor);
        const positions = posData.data as Float32Array;
        const vertCount = posData.count;

        let normals: Float32Array | null = null;
        if (prim.attributes.NORMAL !== undefined) {
            normals = this.getAccessorData(prim.attributes.NORMAL).data as Float32Array;
        }

        let uvs: Float32Array | null = null;
        if (prim.attributes.TEXCOORD_0 !== undefined) {
            uvs = this.getAccessorData(prim.attributes.TEXCOORD_0).data as Float32Array;
        }

        // Interleave into Kansei format: [x, y, z, 1, nx, ny, nz, u, v] per vertex
        const stride = 9; // 4 (pos) + 3 (normal) + 2 (uv)
        const vertices = new Float32Array(vertCount * stride);

        for (let i = 0; i < vertCount; i++) {
            const vi = i * stride;
            const pi = i * 3;

            // position (vec4)
            vertices[vi] = positions[pi];
            vertices[vi + 1] = positions[pi + 1];
            vertices[vi + 2] = positions[pi + 2];
            vertices[vi + 3] = 1.0;

            // normal (vec3)
            if (normals) {
                vertices[vi + 4] = normals[pi];
                vertices[vi + 5] = normals[pi + 1];
                vertices[vi + 6] = normals[pi + 2];
            } else {
                vertices[vi + 4] = 0;
                vertices[vi + 5] = 1;
                vertices[vi + 6] = 0;
            }

            // uv (vec2)
            if (uvs) {
                const ui = i * 2;
                vertices[vi + 7] = uvs[ui];
                vertices[vi + 8] = uvs[ui + 1];
            }
            // else stays 0,0
        }

        const geo = new Geometry();
        geo.vertices = vertices;

        // Indices
        if (prim.indices !== undefined) {
            const indexData = this.getAccessorData(prim.indices);
            geo.vertexCount = indexData.count;

            if (indexData.data instanceof Uint32Array) {
                // Check if indices fit in uint16
                const maxIndex = indexData.data.reduce((m, v) => Math.max(m, v), 0);
                if (maxIndex > 65535) {
                    geo.indices = new Uint32Array(indexData.data);
                    geo.indexFormat = "uint32";
                } else {
                    geo.indices = new Uint16Array(indexData.data);
                    geo.indexFormat = "uint16";
                }
            } else {
                geo.indices = new Uint16Array(indexData.data);
                geo.indexFormat = "uint16";
            }
        } else {
            // Non-indexed: generate sequential indices
            const indexCount = vertCount;
            if (vertCount > 65535) {
                geo.indices = new Uint32Array(indexCount);
                geo.indexFormat = "uint32";
            } else {
                geo.indices = new Uint16Array(indexCount);
            }
            for (let i = 0; i < indexCount; i++) geo.indices[i] = i;
            geo.vertexCount = indexCount;
        }

        return geo;
    }

    private buildScene(geometries: Geometry[], defaultMaterial: Material): Object3D {
        const root = new Object3D();
        const nodeDefs: GLTFNode[] = this.gltf.nodes || [];
        const sceneDef = (this.gltf.scenes || [])[this.gltf.scene ?? 0];

        if (!sceneDef) return root;

        // Track which geometry index corresponds to each mesh/primitive
        // Each mesh may have multiple primitives; geometries array is flat
        const meshDefs: GLTFMesh[] = this.gltf.meshes || [];
        const meshGeoStart: number[] = [];
        let geoIdx = 0;
        for (const meshDef of meshDefs) {
            meshGeoStart.push(geoIdx);
            geoIdx += meshDef.primitives.length;
        }

        const buildNode = (nodeIndex: number): Object3D => {
            const nodeDef = nodeDefs[nodeIndex];
            let node: Object3D;

            if (nodeDef.mesh !== undefined) {
                const startIdx = meshGeoStart[nodeDef.mesh];
                const primCount = meshDefs[nodeDef.mesh].primitives.length;

                if (primCount === 1) {
                    node = new Renderable(geometries[startIdx], defaultMaterial);
                } else {
                    node = new Object3D();
                    for (let p = 0; p < primCount; p++) {
                        const child = new Renderable(geometries[startIdx + p], defaultMaterial);
                        node.add(child);
                    }
                }
            } else {
                node = new Object3D();
            }

            // Apply transforms
            if (nodeDef.matrix) {
                const m = nodeDef.matrix;
                // GLTF matrices are column-major, gl-matrix is also column-major
                const mm = node.modelMatrix.internalMat4;
                for (let i = 0; i < 16; i++) mm[i] = m[i];

                // Extract TRS from matrix for position/rotation/scale
                const translation = [0, 0, 0] as [number, number, number];
                const scaling = [0, 0, 0] as [number, number, number];
                mat4.getTranslation(translation as any, mm);
                mat4.getScaling(scaling as any, mm);

                node.position.set(translation[0], translation[1], translation[2]);
                node.scale.set(scaling[0], scaling[1], scaling[2]);

                const [rx, ry, rz] = node.modelMatrix.extractEulerAngles();
                node.rotation.set(rx, ry, rz);
            } else {
                if (nodeDef.translation) {
                    const t = nodeDef.translation;
                    node.position.set(t[0], t[1], t[2]);
                }
                if (nodeDef.rotation) {
                    // GLTF uses quaternion [x, y, z, w] — convert to euler
                    const q = nodeDef.rotation;
                    const [rx, ry, rz] = quaternionToEuler(q[0], q[1], q[2], q[3]);
                    node.rotation.set(rx, ry, rz);
                }
                if (nodeDef.scale) {
                    const s = nodeDef.scale;
                    node.scale.set(s[0], s[1], s[2]);
                }
            }

            // Recurse children
            if (nodeDef.children) {
                for (const childIdx of nodeDef.children) {
                    node.add(buildNode(childIdx));
                }
            }

            return node;
        };

        for (const nodeIdx of sceneDef.nodes) {
            root.add(buildNode(nodeIdx));
        }

        return root;
    }
}

function quaternionToEuler(x: number, y: number, z: number, w: number): [number, number, number] {
    // YXZ order to match Object3D.updateModelMatrix (rotateZ, rotateY, rotateX)
    const sinr_cosp = 2 * (w * x + y * z);
    const cosr_cosp = 1 - 2 * (x * x + y * y);
    const rx = Math.atan2(sinr_cosp, cosr_cosp);

    const sinp = 2 * (w * y - z * x);
    const ry = Math.abs(sinp) >= 1 ? Math.sign(sinp) * Math.PI / 2 : Math.asin(sinp);

    const siny_cosp = 2 * (w * z + x * y);
    const cosy_cosp = 1 - 2 * (y * y + z * z);
    const rz = Math.atan2(siny_cosp, cosy_cosp);

    return [rx, ry, rz];
}

export { GLTFLoader };
