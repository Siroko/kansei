import { mat4, vec3 } from "gl-matrix";
import { Geometry } from "../buffers/Geometry";
import { Transform } from "./Transform";

/** Joint influences per vertex. */
const MAX_INFLUENCES = 4;

/** Floats per vertex of `SkinnedMesh.vertices`: position (vec4), normal (vec3), uv (vec2). */
const VERTEX_FLOATS = 9;

/**
 * A mesh deformed by a skeleton: its vertices at bind time, each vertex's joints and weights,
 * and the skin (which skeleton joints it uses, with their inverse bind matrices). Rust:
 * `animation::SkinnedMesh`.
 *
 * The vertex shader finds each vertex's skin record by `@builtin(vertex_index)`, the index
 * buffer's value, so the geometry must keep this vertex order: never merge or re-index it.
 */
class SkinnedMesh {
    constructor(
        public name: string,
        /** Interleaved bind-pose vertices in the standard layout (position vec4, normal vec3, uv vec2). */
        public vertices: Float32Array,
        public indices: Uint32Array,
        /** Per vertex, up to four skin joints (indices into `skinJoints`), 4 per vertex... */
        public joints: Uint16Array,
        /** ...and their weights, summing to 1, 4 per vertex. */
        public weights: Float32Array,
        /** The skeleton joint of each skin joint. */
        public skinJoints: number[],
        /** Model space to each skin joint's space at bind time (column-major). */
        public inverseBind: mat4[],
        /** Index of the source material, if any. */
        public material?: number,
    ) { }

    public get vertexCount(): number {
        return this.vertices.length / VERTEX_FLOATS;
    }

    /** The renderable geometry (bind-pose vertices; the vertex shader skins them). */
    public geometry(): Geometry {
        return Geometry.fromArrays(this.name, this.vertices.slice(), this.indices.slice());
    }

    /**
     * Joint matrices for skinning: each skin joint's model transform times its inverse bind
     * matrix, from the skeleton's model-space pose; 16 floats per skin joint, into `out`.
     */
    public palette(model: Transform[], out: Float32Array = new Float32Array(this.skinJoints.length * 16)): Float32Array {
        const m = mat4.create();
        this.skinJoints.forEach((j, i) => {
            model[j].toMat4(m);
            mat4.multiply(out.subarray(16 * i, 16 * i + 16), m, this.inverseBind[i]);
        });
        return out;
    }

    /** Skinned positions and normals on the CPU (the vertex shader's reference). */
    public skinCpu(palette: Float32Array): { position: vec3, normal: vec3 }[] {
        const out: { position: vec3, normal: vec3 }[] = [];
        const m = mat4.create();
        for (let v = 0; v < this.vertexCount; v++) {
            m.fill(0);
            for (let k = 0; k < MAX_INFLUENCES; k++) {
                const w = this.weights[4 * v + k];
                const j = this.joints[4 * v + k];
                for (let i = 0; i < 16; i++) m[i] += palette[16 * j + i] * w;
            }
            const base = v * VERTEX_FLOATS;
            const p = vec3.fromValues(this.vertices[base], this.vertices[base + 1], this.vertices[base + 2]);
            const n = this.vertices.subarray(base + 4, base + 7);
            const normal = vec3.fromValues(
                m[0] * n[0] + m[4] * n[1] + m[8] * n[2],
                m[1] * n[0] + m[5] * n[1] + m[9] * n[2],
                m[2] * n[0] + m[6] * n[1] + m[10] * n[2],
            );
            out.push({ position: vec3.transformMat4(p, p, m), normal });
        }
        return out;
    }

    /**
     * The per-vertex records the skinning shader reads (`SKINNING_WGSL`): the four joints as
     * u16 pairs, then the four weights as unorm16 pairs that sum to exactly 1; 4 words a vertex.
     */
    public skinWords(): Uint32Array {
        const count = this.vertexCount;
        const out = new Uint32Array(count * 4);
        for (let v = 0; v < count; v++) {
            const j = this.joints.subarray(4 * v, 4 * v + 4);
            const q = quantizeWeights(this.weights.subarray(4 * v, 4 * v + 4));
            out[4 * v] = (j[0] | (j[1] << 16)) >>> 0;
            out[4 * v + 1] = (j[2] | (j[3] << 16)) >>> 0;
            out[4 * v + 2] = (q[0] | (q[1] << 16)) >>> 0;
            out[4 * v + 3] = (q[2] | (q[3] << 16)) >>> 0;
        }
        return out;
    }
}

/** Weights as unorm16 that sum to exactly 65535 (the rounding error goes to the largest). */
function quantizeWeights(w: ArrayLike<number>): number[] {
    const q = Array.from({ length: MAX_INFLUENCES }, (_, i) => Math.round(Math.min(Math.max(w[i], 0), 1) * 65535));
    let largest = 0;
    for (let i = 1; i < MAX_INFLUENCES; i++) if (w[i] > w[largest]) largest = i;
    q[largest] += 65535 - q.reduce((a, b) => a + b, 0);
    return q.map((x) => Math.min(Math.max(x, 0), 65535));
}

/**
 * The `MAX_INFLUENCES` largest of a vertex's influences (a joint listed twice counts once, with
 * both weights), renormalized to sum to 1 (joint 0 with full weight when none has weight).
 */
function strongestInfluences(influences: Iterable<[number, number]>): { joints: number[], weights: number[] } {
    const all: [number, number][] = [];
    for (const [j, w] of influences) {
        if (!(w > 0)) continue;
        const same = all.find(([k]) => k === j);
        if (same) same[1] += w;
        else all.push([j, w]);
    }
    // A stable sort, strongest first, as Rust's `sort_by`.
    all.sort((a, b) => b[1] - a[1]);
    all.length = Math.min(all.length, MAX_INFLUENCES);
    const total = all.reduce((s, [, w]) => s + w, 0);
    const joints = [0, 0, 0, 0];
    const weights = [0, 0, 0, 0];
    if (total <= 0) {
        weights[0] = 1;
        return { joints, weights };
    }
    all.forEach(([j, w], k) => {
        joints[k] = j;
        weights[k] = w / total;
    });
    return { joints, weights };
}

export { SkinnedMesh, MAX_INFLUENCES, strongestInfluences, quantizeWeights };
