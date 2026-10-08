import { mat4, quat, vec3 } from "gl-matrix";
import type { Texture } from "../buffers/Texture";
import type { CompressionSupport } from "../renderers/Renderer";
import { GLTFImage, GLTFLoader, GLTFMaterialInfo, GLTFTextureRef, loadTexture } from "../loaders/GLTFLoader";
import { Clip } from "./Clip";
import { Pose } from "./Pose";
import { Skeleton } from "./Skeleton";
import { SkinnedMesh, strongestInfluences } from "./SkinnedMesh";
import { Transform } from "./Transform";

/**
 * A glTF file's skeleton, skinned meshes and animations. Rust: `animation::SkinnedGltf`.
 *
 * The skeleton holds every joint of every skin (or, in a file without skins, every node), parents
 * first. Non-joint nodes above a joint (an armature node, an axis conversion, a helper between
 * two joints) are folded into it, so the skeleton's model space is the file's world space, as glTF
 * skinning defines it. Animations are resampled at a fixed rate into `Clip`s over the skeleton's
 * joints; channels on other nodes are ignored.
 */
class SkinnedGltf {
    private constructor(
        public skeleton: Skeleton,
        public meshes: SkinnedMesh[],
        public clips: Clip[],
        /** With their texture references (`KHR_texture_basisu` KTX2 images included), as `GLTFLoader` reads them. */
        public materials: GLTFMaterialInfo[],
        public images: GLTFImage[],
        private baseUrl: string,
    ) { }

    /**
     * Load a .gltf or .glb file. `sampleRate` resamples the animations (frames per second);
     * unset, the rate of their keys.
     */
    public static async load(url: string, sampleRate?: number): Promise<SkinnedGltf> {
        const loader = new GLTFLoader();
        await loader.open(url);
        return SkinnedGltf.fromLoader(loader, sampleRate);
    }

    /**
     * From in-memory .glb bytes (or .gltf JSON with embedded buffers). Images and buffers stored
     * outside the file resolve against `baseUrl`.
     */
    public static async fromBytes(bytes: ArrayBuffer | Uint8Array, sampleRate?: number, baseUrl: string = ""): Promise<SkinnedGltf> {
        const loader = new GLTFLoader();
        await loader.openBytes(bytes, baseUrl);
        return SkinnedGltf.fromLoader(loader, sampleRate);
    }

    /** Decode or transcode a material texture, as `GLTFResult.loadTexture`. */
    public loadTexture(texture: GLTFTextureRef, support: CompressionSupport): Promise<Texture> {
        return loadTexture(this.images, this.baseUrl, texture, support);
    }

    private static fromLoader(loader: GLTFLoader, sampleRate?: number): SkinnedGltf {
        const json = loader.json;
        const { skeleton, joints } = buildSkeleton(json);
        if (skeleton.isEmpty()) throw new Error('SkinnedGltf: the glTF has no nodes to animate');
        const meshes: SkinnedMesh[] = [];
        const scene = (json.scenes ?? [])[json.scene ?? 0];
        const nodes: GltfNode[] = json.nodes ?? [];
        for (const n of (scene?.nodes ?? []).flatMap((root: number) => descendants(nodes, root))) {
            const node = nodes[n];
            if (node.mesh !== undefined && node.skin !== undefined) {
                const mesh = json.meshes[node.mesh];
                mesh.primitives.forEach((primitive: GltfPrimitive, p: number) => {
                    const m = skinnedPrimitive(loader, `${mesh.name ?? 'mesh'}/${p}`, primitive, json.skins[node.skin!], joints.nodeJoint);
                    if (m) meshes.push(m);
                });
            } else if (node.mesh !== undefined) {
                console.warn(`SkinnedGltf: node '${node.name ?? '?'}' has a mesh without a skin: not imported`);
            }
        }
        const clips = (json.animations ?? []).map((a: GltfAnimation, i: number) => buildClip(loader, a, i, joints, sampleRate));
        return new SkinnedGltf(skeleton, meshes, clips, loader.parseMaterials(), loader.parseImages(), loader.base);
    }
}

interface GltfNode {
    name?: string;
    mesh?: number;
    skin?: number;
    children?: number[];
    matrix?: number[];
    translation?: number[];
    rotation?: number[];
    scale?: number[];
}

interface GltfPrimitive {
    attributes: Record<string, number>;
    indices?: number;
    material?: number;
    mode?: number;
}

interface GltfAnimation {
    name?: string;
    channels: { sampler: number; target: { node?: number; path: string } }[];
    samplers: { input: number; output: number; interpolation?: 'LINEAR' | 'STEP' | 'CUBICSPLINE' }[];
}

/** A node and everything below it, depth first. */
function descendants(nodes: GltfNode[], node: number): number[] {
    return [node, ...(nodes[node].children ?? []).flatMap((c) => descendants(nodes, c))];
}

function nodeTransform(node: GltfNode): Transform {
    if (node.matrix) return Transform.fromMat4(mat4.clone(node.matrix as unknown as mat4));
    const t = node.translation ?? [0, 0, 0];
    const r = node.rotation ?? [0, 0, 0, 1];
    const s = node.scale ?? [1, 1, 1];
    const rotation = quat.fromValues(r[0], r[1], r[2], r[3]);
    return new Transform(vec3.fromValues(t[0], t[1], t[2]), quat.normalize(rotation, rotation), vec3.fromValues(s[0], s[1], s[2]));
}

/** How the glTF's nodes map onto the skeleton. */
interface JointNodes {
    /** Joint index of each joint node. */
    nodeJoint: Map<number, number>;
    /** Per joint, its node's own local transform (what animation channels replace)... */
    own: Transform[];
    /** ...and the non-joint nodes between it and its parent joint (or the scene root), folded in front of it. */
    above: Transform[];
}

/** The skeleton, and how the nodes map onto it. */
function buildSkeleton(json: any): { skeleton: Skeleton; joints: JointNodes } {
    const nodes: GltfNode[] = json.nodes ?? [];
    const skins: { joints: number[] }[] = json.skins ?? [];
    const isJoint = new Array<boolean>(nodes.length).fill(skins.length === 0);
    for (const skin of skins) for (const j of skin.joints) isJoint[j] = true;

    const names: string[] = [];
    const parents: (number | null)[] = [];
    const joints: JointNodes = { nodeJoint: new Map(), own: [], above: [] };
    // Depth first from the scene roots: parents come before children.
    const visit = (n: number, parentJoint: number | null, above: Transform) => {
        const node = nodes[n];
        const local = nodeTransform(node);
        let joint = parentJoint;
        if (isJoint[n]) {
            joint = names.length;
            names.push(node.name ?? `node_${n}`);
            parents.push(parentJoint);
            joints.own.push(local);
            // A joint takes the transform of the non-joint nodes since its parent joint.
            joints.above.push(above);
            joints.nodeJoint.set(n, joint);
            above = Transform.identity();
        } else {
            above = above.mul(local);
        }
        for (const child of node.children ?? []) visit(child, joint, above);
    };
    const scene = (json.scenes ?? [])[json.scene ?? 0];
    const roots: number[] = scene
        ? scene.nodes ?? []
        : nodes.map((_, i) => i).filter((i) => !nodes.some((p) => p.children?.includes(i)));
    for (const root of roots) visit(root, null, Transform.identity());
    const rest = joints.above.map((a, j) => a.mul(joints.own[j]));
    return { skeleton: new Skeleton(names, parents, rest), joints };
}

/** A skinned primitive: vertices at bind time, their four strongest influences, the skin. */
function skinnedPrimitive(
    loader: GLTFLoader,
    name: string,
    primitive: GltfPrimitive,
    skin: { joints: number[]; inverseBindMatrices?: number },
    nodeJoint: Map<number, number>,
): SkinnedMesh | null {
    if ((primitive.mode ?? 4) !== 4) {
        console.warn(`SkinnedGltf: primitive ${name} is not a triangle list: not imported`);
        return null;
    }
    const a = primitive.attributes;
    if (a.POSITION === undefined) return null;
    const positions = loader.accessorFloats(a.POSITION);
    const count = positions.length / 3;
    const normals = a.NORMAL !== undefined ? loader.accessorFloats(a.NORMAL) : null;
    const uvs = a.TEXCOORD_0 !== undefined ? loader.accessorFloats(a.TEXCOORD_0) : null;
    const indices = primitive.indices !== undefined
        ? loader.accessorUints(primitive.indices)
        : Uint32Array.from({ length: count }, (_, i) => i);

    // Every JOINTS_n / WEIGHTS_n set, reduced to the four strongest influences.
    const influences: [number, number][][] = Array.from({ length: count }, () => []);
    let set = 0;
    for (; a[`JOINTS_${set}`] !== undefined && a[`WEIGHTS_${set}`] !== undefined; set++) {
        const j = loader.accessorUints(a[`JOINTS_${set}`]);
        const w = loader.accessorFloats(a[`WEIGHTS_${set}`]);
        for (let v = 0; v < count; v++) {
            for (let k = 0; k < 4; k++) influences[v].push([j[4 * v + k], w[4 * v + k]]);
        }
    }
    if (set === 0) throw new Error(`SkinnedGltf: primitive ${name} is skinned but has no JOINTS_0/WEIGHTS_0`);
    const joints = new Uint16Array(count * 4);
    const weights = new Float32Array(count * 4);
    influences.forEach((list, v) => {
        const strongest = strongestInfluences(list);
        joints.set(strongest.joints, 4 * v);
        weights.set(strongest.weights, 4 * v);
    });

    const skinJoints = skin.joints.map((n) => {
        const j = nodeJoint.get(n);
        if (j === undefined) throw new Error(`SkinnedGltf: skin joint node ${n} is not in the skeleton`);
        return j;
    });
    let inverseBind: mat4[];
    if (skin.inverseBindMatrices !== undefined) {
        const m = loader.accessorFloats(skin.inverseBindMatrices);
        inverseBind = skinJoints.map((_, i) => mat4.clone(m.subarray(16 * i, 16 * i + 16) as unknown as mat4));
    } else {
        inverseBind = skinJoints.map(() => mat4.create());
    }
    if (joints.some((j) => j >= skinJoints.length)) {
        throw new Error(`SkinnedGltf: primitive ${name} references a joint beyond its skin's ${skinJoints.length}`);
    }
    const vertices = new Float32Array(count * 9);
    for (let i = 0; i < count; i++) {
        vertices.set(positions.subarray(3 * i, 3 * i + 3), 9 * i);
        vertices[9 * i + 3] = 1;
        if (normals) vertices.set(normals.subarray(3 * i, 3 * i + 3), 9 * i + 4);
        else vertices[9 * i + 5] = 1;
        if (uvs) vertices.set(uvs.subarray(2 * i, 2 * i + 2), 9 * i + 7);
    }
    return new SkinnedMesh(name, vertices, indices, joints, weights, skinJoints, inverseBind, primitive.material);
}

/** One sampler's keys, for evaluation at any time. */
class Track {
    constructor(
        private times: Float32Array,
        /** Per key: the value (xyz, or xyzw for rotations); for cubic splines, in-tangent, value, out-tangent. */
        private values: Float32Array,
        private width: number,
        private interpolation: 'LINEAR' | 'STEP' | 'CUBICSPLINE',
        private rotation: boolean,
    ) { }

    /** Element `e` (a key's value, or a tangent) as 4 numbers. */
    private element(e: number): number[] {
        const v = [0, 0, 0, 0];
        for (let i = 0; i < this.width; i++) v[i] = this.values[e * this.width + i];
        return v;
    }

    private value(k: number): number[] {
        return this.element(this.interpolation === 'CUBICSPLINE' ? 3 * k + 1 : k);
    }

    public sample(t: number): number[] {
        const n = this.times.length;
        if (n === 1 || t <= this.times[0]) return this.value(0);
        if (t >= this.times[n - 1]) return this.value(n - 1);
        // The last key at or before t (Rust's `partition_point(|x| x <= t) - 1`).
        let lo = 0, hi = n;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (this.times[mid] <= t) lo = mid + 1;
            else hi = mid;
        }
        const k = lo - 1;
        const t0 = this.times[k], t1 = this.times[k + 1];
        const dt = t1 - t0;
        const s = dt > 0 ? (t - t0) / dt : 0;
        switch (this.interpolation) {
            case 'STEP':
                return this.value(k);
            case 'CUBICSPLINE': {
                const p0 = this.element(3 * k + 1), m0 = this.element(3 * k + 2);
                const m1 = this.element(3 * (k + 1)), p1 = this.element(3 * (k + 1) + 1);
                const s2 = s * s, s3 = s2 * s;
                const v = [0, 1, 2, 3].map((i) =>
                    (2 * s3 - 3 * s2 + 1) * p0[i] + (s3 - 2 * s2 + s) * dt * m0[i] + (-2 * s3 + 3 * s2) * p1[i] + (s3 - s2) * dt * m1[i]);
                if (!this.rotation) return v;
                const q = quat.normalize(quat.create(), v as unknown as quat);
                return Array.from(q);
            }
            default: {
                const a = this.value(k), b = this.value(k + 1);
                if (this.rotation) {
                    const q = quat.slerp(quat.create(), a as unknown as quat, b as unknown as quat, s);
                    return Array.from(q);
                }
                return [0, 1, 2, 3].map((i) => a[i] + (b[i] - a[i]) * s);
            }
        }
    }
}

/** An animation resampled at `sampleRate` (or its keys' rate) into a clip over the skeleton. */
function buildClip(loader: GLTFLoader, animation: GltfAnimation, index: number, joints: JointNodes, sampleRate?: number): Clip {
    const name = animation.name ?? `animation_${index}`;
    // Per joint: translation, rotation, scale tracks.
    const tracks: (Track | undefined)[][] = joints.own.map(() => [undefined, undefined, undefined]);
    let start = Infinity, end = -Infinity, spacing = Infinity;
    for (const channel of animation.channels) {
        const node = channel.target.node;
        const joint = node === undefined ? undefined : joints.nodeJoint.get(node);
        if (joint === undefined) {
            console.warn(`SkinnedGltf: animation '${name}' animates node ${node} outside the skeleton: ignored`);
            continue;
        }
        const slot = { translation: 0, rotation: 1, scale: 2 }[channel.target.path];
        if (slot === undefined) continue;
        const sampler = animation.samplers[channel.sampler];
        const times = loader.accessorFloats(sampler.input);
        if (times.length === 0) continue;
        const values = loader.accessorFloats(sampler.output);
        start = Math.min(start, times[0]);
        end = Math.max(end, times[times.length - 1]);
        for (let i = 1; i < times.length; i++) {
            const d = times[i] - times[i - 1];
            if (d > 1e-6) spacing = Math.min(spacing, d);
        }
        tracks[joint][slot] = new Track(times, values, slot === 1 ? 4 : 3, sampler.interpolation ?? 'LINEAR', slot === 1);
    }
    if (start > end) {
        start = 0;
        end = 0;
    }
    // The keys' rate, rounded to a whole number of frames per second.
    const rate = sampleRate ?? (spacing < Infinity ? Math.min(Math.max(Math.round(1 / spacing), 1), 240) : 30);
    const frames = Math.round((end - start) * rate) + 1;
    const poses: Pose[] = [];
    for (let f = 0; f < frames; f++) {
        const t = start + f / rate;
        poses.push(new Pose(tracks.map(([translation, rotation, scale], j) => {
            // Channels replace the node's own transform; the folded nodes stay above it.
            const out = joints.own[j].clone();
            if (translation) {
                const v = translation.sample(t);
                vec3.set(out.translation, v[0], v[1], v[2]);
            }
            if (rotation) {
                const v = rotation.sample(t);
                quat.set(out.rotation, v[0], v[1], v[2], v[3]);
                quat.normalize(out.rotation, out.rotation);
            }
            if (scale) {
                const v = scale.sample(t);
                vec3.set(out.scale, v[0], v[1], v[2]);
            }
            return joints.above[j].mul(out);
        })));
    }
    return Clip.fromPoses(name, rate, poses);
}

export { SkinnedGltf };
