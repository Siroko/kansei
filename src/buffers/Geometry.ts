import { mat4, vec3 } from "gl-matrix";
import { Matrix4 } from "../math/Matrix4";

/** Floats per vertex in the standard layout: position (vec4), normal (vec3), uv (vec2); 36 bytes. */
const VERTEX_FLOATS = 9;

/** A column-major 4x4 matrix: a `Matrix4`, a gl-matrix `mat4` or 16 numbers. */
type MatrixLike = Matrix4 | ArrayLike<number>;

/** Axis-aligned bounds, lowest and highest corner. */
interface GeometryBounds {
    min: [number, number, number];
    max: [number, number, number];
}

/**
 * Represents a geometric mesh with vertex and index buffers for WebGPU rendering.
 */
class Geometry {
    /** Names the geometry's GPU buffers (in GPU captures); see `withLabel`. */
    public label: string = 'Geometry';

    /** Indicates if this geometry is used for instanced rendering */
    public isInstancedGeometry: boolean = false;

    /** WebGPU buffer containing vertex data */
    public vertexBuffer?: GPUBuffer;

    /** WebGPU buffer containing index data */
    public indexBuffer?: GPUBuffer;

    /** Format of the index buffer data: 32-bit unless set otherwise (as the Rust engine draws). */
    public indexFormat: GPUIndexFormat = "uint32";

    /** Collection of vertex buffer layout descriptors */
    public vertexBuffersDescriptors: Iterable<GPUVertexBufferLayout | null> = [];

    /** Indicates if the geometry has been initialized with GPU buffers */
    public initialized: boolean = false;

    /** Number of vertices in the geometry */
    public vertexCount: number = 0;

    /** Raw vertex data containing interleaved positions, normals, and UVs */
    public vertices?: Float32Array;

    /** Raw index data for defining triangles */
    public indices?: Uint16Array | Uint32Array;

    /** Optional indirect draw args buffer (DrawIndexedIndirect: 5 × u32).
     *  When set, the renderer uses drawIndexedIndirect() instead of drawIndexed(). */
    public indirectArgsBuffer?: GPUBuffer;

    /** True if vertex/index buffers are externally owned (not created by initialize()). */
    public isExternal: boolean = false;

    constructor() { }

    /**
     * A geometry from interleaved vertices (position vec4, normal vec3, uv vec2) and u32
     * indices: what the stock generators build.
     */
    public static fromArrays(label: string, vertices: Float32Array, indices: Uint32Array): Geometry {
        return new Geometry().setArrays(label, vertices, indices);
    }

    /** Take `vertices` and u32 `indices` as this geometry's data, before it is initialized. */
    protected setArrays(label: string, vertices: Float32Array, indices: Uint32Array): this {
        this.label = label;
        this.vertices = vertices;
        this.indices = indices;
        this.indexFormat = 'uint32';
        this.vertexCount = indices.length;
        return this;
    }

    /**
     * The same geometry named `label` (in GPU captures): one of the stock generators' meshes,
     * say, as `new SpruceGeometry(8, 1, 3).withLabel('Spruce/LOD2')`.
     */
    public withLabel(label: string): this {
        this.label = label;
        return this;
    }

    /**
     * A geometry whose vertices a compute pass writes into `vertexBuffer` (laid out as the
     * standard vertex, with VERTEX usage), drawn with `indices`. Nothing is uploaded for the
     * vertices: the geometry keeps the buffer (its CPU `vertices` stay empty).
     */
    public static fromGpuVertices(label: string, vertexBuffer: GPUBuffer, indices: Uint32Array): Geometry {
        const geometry = Geometry.fromArrays(label, new Float32Array(0), indices);
        geometry.vertexBuffer = vertexBuffer;
        return geometry;
    }

    /**
     * Several geometries as one, each moved by its matrix first (normals by its inverse
     * transpose): a spruce from cones, a model from its parts, props from boxes and cylinders.
     * Instancing is not carried over.
     */
    public static merged(label: string, parts: [Geometry, MatrixLike][]): Geometry {
        let vertexTotal = 0;
        let indexTotal = 0;
        for (const [geometry] of parts) {
            vertexTotal += (geometry.vertices?.length ?? 0) / VERTEX_FLOATS;
            indexTotal += geometry.indices?.length ?? 0;
        }
        const vertices = new Float32Array(vertexTotal * VERTEX_FLOATS);
        const indices = new Uint32Array(indexTotal);
        const normalMatrix = mat4.create();
        const p = vec3.create();
        let base = 0;
        let at = 0;
        for (const [geometry, matrix] of parts) {
            const m = (matrix instanceof Matrix4 ? matrix.internalMat4 : matrix) as mat4;
            mat4.transpose(normalMatrix, mat4.invert(normalMatrix, m) ?? mat4.create());
            const source = geometry.vertices ?? new Float32Array(0);
            const count = source.length / VERTEX_FLOATS;
            for (let v = 0; v < count; v++) {
                const i = v * VERTEX_FLOATS;
                const o = (base + v) * VERTEX_FLOATS;
                vec3.transformMat4(p, vec3.set(p, source[i], source[i + 1], source[i + 2]), m);
                vertices.set([p[0], p[1], p[2], 1], o);
                // the normal as a direction (w = 0) through the inverse transpose
                const [nx, ny, nz] = [source[i + 4], source[i + 5], source[i + 6]];
                const n = normalMatrix;
                vec3.set(p, n[0] * nx + n[4] * ny + n[8] * nz, n[1] * nx + n[5] * ny + n[9] * nz, n[2] * nx + n[6] * ny + n[10] * nz);
                vec3.normalize(p, p);
                vertices.set([p[0], p[1], p[2], source[i + 7], source[i + 8]], o + 4);
            }
            for (const index of geometry.indices ?? []) indices[at++] = base + index;
            base += count;
        }
        return Geometry.fromArrays(label, vertices, indices);
    }

    /** The axis-aligned bounds of the CPU vertices; zero for none. */
    public bounds(): GeometryBounds {
        const v = this.vertices;
        if (!v || v.length < VERTEX_FLOATS) return { min: [0, 0, 0], max: [0, 0, 0] };
        const min: [number, number, number] = [v[0], v[1], v[2]];
        const max: [number, number, number] = [v[0], v[1], v[2]];
        for (let i = VERTEX_FLOATS; i < v.length; i += VERTEX_FLOATS) {
            for (let k = 0; k < 3; k++) {
                min[k] = Math.min(min[k], v[i + k]);
                max[k] = Math.max(max[k], v[i + k]);
            }
        }
        return { min, max };
    }

    /**
     * Scale the vertices uniformly to fit inside a box of `size` (use `Infinity` for an axis
     * that may be any size) and move them so the bottom centre of their bounds sits at the
     * origin: a model ready to stand on the ground. Call it before the geometry is drawn.
     */
    public fit(size: [number, number, number]): this {
        const { min, max } = this.bounds();
        const scale = Math.min(...[0, 1, 2].map((k) => size[k] / Math.max(max[k] - min[k], 1e-9)));
        const anchor = [(min[0] + max[0]) * 0.5, min[1], (min[2] + max[2]) * 0.5];
        const v = this.vertices!;
        for (let i = 0; i < v.length; i += VERTEX_FLOATS) {
            for (let k = 0; k < 3; k++) v[i + k] = (v[i + k] - anchor[k]) * scale;
            v[i + 3] = 1;
        }
        return this;
    }

    /**
     * Create a Geometry that wraps externally-owned GPU buffers (zero readback).
     * Used for compute-generated meshes (e.g. marching cubes) where the vertex/index data
     * lives entirely on the GPU. The `indirectArgsBuffer` holds DrawIndexedIndirect args
     * so the triangle count is also GPU-driven.
     *
     * The vertex layout matches the standard engine Vertex format:
     *   position(vec4) + normal(vec3) + uv(vec2) — 36-byte stride.
     */
    public static fromGpuBuffers(
        vertexBuffer: GPUBuffer,
        indexBuffer: GPUBuffer,
        indirectArgsBuffer: GPUBuffer,
        indexFormat: GPUIndexFormat = 'uint32',
    ): Geometry {
        const geo = new Geometry();
        geo.vertexBuffer = vertexBuffer;
        geo.indexBuffer = indexBuffer;
        geo.indirectArgsBuffer = indirectArgsBuffer;
        geo.indexFormat = indexFormat;
        geo.isExternal = true;
        geo.initialized = true;
        (geo.vertexBuffersDescriptors as Array<GPUVertexBufferLayout>).push({
            attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x4' },
                { shaderLocation: 1, offset: 16, format: 'float32x3' },
                { shaderLocation: 2, offset: 28, format: 'float32x2' },
            ],
            arrayStride: 36,
            stepMode: 'vertex',
        });
        return geo;
    }

    /**
     * Initializes the geometry by creating GPU buffers and setting up vertex layouts.
     * @param gpuDevice - The WebGPU device to create buffers on
     */
    public initialize(gpuDevice: GPUDevice) {
        if (this.isExternal) { this.initialized = true; return; }
        // a buffer handed in (`fromGpuVertices`) is kept
        if (!this.vertexBuffer) {
            this.vertexBuffer = gpuDevice.createBuffer({
                label: `${this.label}/Vertices`,
                size: this.vertices!.byteLength,
                usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
                mappedAtCreation: true
            });
            new Float32Array(this.vertexBuffer.getMappedRange()).set(this.vertices!);
            this.vertexBuffer.unmap();
        }

        this.indexBuffer = gpuDevice.createBuffer({
            label: `${this.label}/Indices`,
            size: Math.ceil(this.indices!.byteLength / 4) * 4,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true
        });
        const IndexArray = this.indexFormat === "uint32" ? Uint32Array : Uint16Array;
        new IndexArray(this.indexBuffer.getMappedRange()).set(this.indices!);
        this.indexBuffer.unmap();

        (this.vertexBuffersDescriptors as Array<GPUVertexBufferLayout>).push({
            attributes: [
                {
                    shaderLocation: 0 as GPUIndex32,
                    offset: 0 as GPUSize64,
                    format: "float32x4" as GPUVertexFormat
                },
                {
                    shaderLocation: 1 as GPUIndex32,
                    offset: 4 * 4 as GPUSize64,
                    format: "float32x3" as GPUVertexFormat
                },
                {
                    shaderLocation: 2 as GPUIndex32,
                    offset: 4 * 4 + 4 * 3 as GPUSize64,
                    format: "float32x2" as GPUVertexFormat
                }
            ] as Iterable<GPUVertexAttribute>,
            arrayStride: 4 * 4 + 4 * 3 + 4 * 2 as GPUSize32,
            stepMode: "vertex" as GPUVertexStepMode
        });

        this.initialized = true;
    }
}

export { Geometry }
export type { GeometryBounds, MatrixLike }
