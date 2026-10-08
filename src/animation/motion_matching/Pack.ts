import { mat4, quat, vec3 } from "gl-matrix";
import { Skeleton } from "../Skeleton";
import { SkinnedMesh } from "../SkinnedMesh";
import { Transform } from "../Transform";
import { ActionClip, ActionKind } from "./ActionClip";
import { ClipInfo, Database, FEATURES, FeatureWeights, JointRoles, STRIDE, Vec3Tracks } from "./Database";

/**
 * `.kmm`: a motion-matching database and its skinned meshes in one little-endian binary file;
 * or, as a `CharacterPack`, a character to show that animation on (skeleton, meshes, images).
 * Rust: `motion_matching::pack`.
 *
 * `KMMP`, a u32 version, then sections: a 4-byte tag, a u64 length and the payload. Readers skip
 * tags they don't know. Sections: `SKEL` skeleton, `ROLE` joint roles, rate and feature weights,
 * `CLIP` clips, `ROTS` quantized rotations, `TRAN`/`SCAL` translation and scale tracks, `ROOT`
 * character root per frame, `CONT` foot contacts, `FEAT` feature normalization and rows, `MESH`
 * (repeated) skinned meshes with a colour, `IMAG` (repeated) named images (encoded bytes, e.g.
 * WebP), `ACTS` action clips (traversals, falls, landings: what `ActionClip.analyze` found),
 * `META` key/value strings (source, licence).
 *
 * No pack ships with Kansei: they are baked from licensed animation (`rust/kansei-anim-bake`)
 * and stay out of the repository and the site.
 */

const MAGIC = "KMMP";
const VERSION = 1;

/** A skinned mesh of a pack and the colour to draw it with. */
interface PackMesh {
    mesh: SkinnedMesh;
    color: [number, number, number, number];
}

/** A named image, encoded (PNG, JPEG, WebP...): a character's textures. */
interface PackImage {
    name: string;
    /** Its media type, e.g. `image/webp`. */
    mime: string;
    bytes: Uint8Array;
}

/** A `.kmm` that cannot be read: not a pack, another version, truncated or inconsistent. */
class PackError extends Error { }

/** Everything a motion-matched character needs, as one file. */
class MotionPack {
    constructor(
        public database: Database,
        public meshes: PackMesh[],
        /** Clips played on command, with their analysis. */
        public actions: ActionClip[],
        /** Free-form (key, value) notes: where the data comes from and under which licence. */
        public meta: [string, string][],
    ) { }

    public metaValue(key: string): string | undefined {
        return this.meta.find(([k]) => k === key)?.[1];
    }

    public toBytes(): Uint8Array {
        const db = this.database;
        const out = new Writer();
        out.header();
        out.section("META", (w) => w.meta(this.meta));
        out.section("SKEL", (w) => w.skeleton(db.skeleton));
        out.section("ROLE", (w) => {
            for (const j of [db.roles.root, db.roles.hips, db.roles.feet[0], db.roles.feet[1]]) w.u32(j);
            w.f32(db.sampleRate);
            const f = db.weights;
            for (const x of [f.footPosition, f.footVelocity, f.hipsVelocity, f.trajectoryPosition, f.trajectoryDirection]) w.f32(x);
        });
        out.section("CLIP", (w) => {
            w.u32(db.clips.length);
            for (const c of db.clips) {
                w.str(c.name);
                w.u32(c.start);
                w.u32(c.frames);
                w.u8(c.looping ? 1 : 0);
                w.u32(c.tags);
            }
        });
        out.section("ROTS", (w) => {
            w.u32(db.rotations.length / 4);
            w.i16s(db.rotations);
        });
        out.section("TRAN", (w) => w.tracks(db.translations));
        out.section("SCAL", (w) => w.tracks(db.scales));
        out.section("ROOT", (w) => {
            w.u32(db.frameCount);
            for (let f = 0; f < db.frameCount; f++) {
                w.f32s(db.rootTranslations.subarray(3 * f, 3 * f + 3));
                w.f32s(db.rootRotations.subarray(4 * f, 4 * f + 4));
            }
        });
        out.section("CONT", (w) => {
            w.u32(db.contactBits.length);
            w.bytes(db.contactBits);
        });
        out.section("FEAT", (w) => {
            w.f32s(db.featureOffset);
            w.f32s(db.featureScale);
            w.u32(db.frameCount);
            for (let f = 0; f < db.frameCount; f++) w.f32s(db.featureRows.subarray(f * STRIDE, f * STRIDE + FEATURES));
        });
        for (const m of this.meshes) out.section("MESH", (w) => w.mesh(m));
        if (this.actions.length > 0) {
            out.section("ACTS", (w) => {
                w.u32(this.actions.length);
                for (const a of this.actions) {
                    w.u32(a.clip);
                    w.u8(a.kind);
                    w.f32(a.height);
                    w.f32s(a.ledge);
                    w.f32s(a.forward);
                    w.f32s([a.rise, a.anchor, a.onTop, a.offTop, a.down, a.exit, a.span, a.lastEntry]);
                }
            });
        }
        return out.finish();
    }

    /** Read a `.kmm`; throws a `PackError` saying what is wrong with it. */
    public static fromBytes(bytes: ArrayBuffer | Uint8Array): MotionPack {
        const meta: [string, string][] = [];
        let skeleton: Skeleton | undefined;
        let roles: [JointRoles, number, FeatureWeights] | undefined;
        const clips: ClipInfo[] = [];
        let rotations = new Int16Array(0);
        let translations: Vec3Tracks | undefined, scales: Vec3Tracks | undefined;
        let roots: { translations: Float32Array, rotations: Float32Array } | undefined;
        let contacts = new Uint8Array(0);
        let features: [Float32Array, Float32Array, Float32Array] | undefined;
        const meshes: PackMesh[] = [];
        const actions: ActionClip[] = [];
        forEachSection(bytes, "motion pack", (tag, r) => {
            switch (tag) {
                case "META": meta.push(...r.meta()); break;
                case "SKEL": skeleton = r.skeleton(); break;
                case "ROLE": {
                    const j = [r.u32(), r.u32(), r.u32(), r.u32()];
                    const rate = r.f32();
                    const f = r.f32s(5);
                    roles = [{ root: j[0], hips: j[1], feet: [j[2], j[3]] }, rate, { footPosition: f[0], footVelocity: f[1], hipsVelocity: f[2], trajectoryPosition: f[3], trajectoryDirection: f[4] }];
                    break;
                }
                case "CLIP": {
                    const n = r.u32();
                    for (let i = 0; i < n; i++) clips.push(new ClipInfo(r.str(), r.u32(), r.u32(), r.u8() !== 0, r.u32()));
                    break;
                }
                case "ROTS": rotations = r.i16s(r.u32() * 4); break;
                case "TRAN": translations = r.tracks(); break;
                case "SCAL": scales = r.tracks(); break;
                case "ROOT": {
                    const n = r.u32();
                    const packed = r.f32s(n * 7);
                    const t = new Float32Array(n * 3), q = new Float32Array(n * 4);
                    for (let f = 0; f < n; f++) {
                        t.set(packed.subarray(7 * f, 7 * f + 3), 3 * f);
                        q.set(packed.subarray(7 * f + 3, 7 * f + 7), 4 * f);
                    }
                    roots = { translations: t, rotations: q };
                    break;
                }
                case "CONT": contacts = r.take(r.u32()).slice(); break;
                case "FEAT": {
                    const offset = r.f32s(FEATURES), scale = r.f32s(FEATURES);
                    const frames = r.u32();
                    const packed = r.f32s(frames * FEATURES);
                    const rows = new Float32Array(frames * STRIDE);
                    for (let f = 0; f < frames; f++) rows.set(packed.subarray(f * FEATURES, (f + 1) * FEATURES), f * STRIDE);
                    features = [offset, scale, rows];
                    break;
                }
                case "MESH": meshes.push(r.mesh()); break;
                case "ACTS": {
                    const n = r.u32();
                    for (let i = 0; i < n; i++) {
                        const clip = r.u32();
                        const kind = r.u8();
                        if (kind > ActionKind.Jump) throw new PackError("unknown action kind in the motion pack");
                        const height = r.f32();
                        const ledge = r.f32s(3) as vec3, forward = r.f32s(3) as vec3;
                        const [rise, anchor, onTop, offTop, down, exit, span, lastEntry] = r.f32s(8);
                        actions.push(new ActionClip({ clip, kind, height, ledge, forward, rise, anchor, onTop, offTop, down, exit, span, lastEntry }));
                    }
                    break;
                }
            }
        });
        if (!skeleton) throw new PackError("the motion pack has no skeleton");
        if (!roles) throw new PackError("the motion pack has no joint roles");
        if (!features) throw new PackError("the motion pack has no features");
        if (!translations) throw new PackError("the motion pack has no translations");
        if (!scales) throw new PackError("the motion pack has no scales");
        const joints = skeleton.length;
        const frames = roots?.translations.length ? roots.translations.length / 3 : 0;
        if (rotations.length !== frames * joints * 4 || contacts.length !== frames || features[2].length !== frames * STRIDE
            || translations.animated.length !== frames * translations.animatedJoints * 3 || scales.animated.length !== frames * scales.animatedJoints * 3) {
            throw new PackError("the motion pack's sections disagree on the frame count");
        }
        const [jointRoles, sampleRate, weights] = roles;
        if (clips.some((c) => c.start + c.frames > frames) || [jointRoles.root, jointRoles.hips, ...jointRoles.feet].some((j) => j >= joints)) {
            throw new PackError("the motion pack's clips or roles are out of range");
        }
        if (actions.some((a) => a.clip >= clips.length)) throw new PackError("the motion pack's actions name clips it lacks");
        for (const m of meshes) {
            if (m.mesh.skinJoints.some((j) => j >= joints)) throw new PackError(`mesh '${m.mesh.name}' is skinned to joints the skeleton lacks`);
        }
        const database = new Database(skeleton, jointRoles, sampleRate, weights, clips, rotations, translations, scales,
            roots ?? { translations: new Float32Array(0), rotations: new Float32Array(0) }, contacts, features[0], features[1], features[2]);
        return new MotionPack(database, meshes, actions, meta);
    }
}

/**
 * A character to show a motion pack's animation on: its skeleton (the same joint names and axes
 * as the motion pack's, its own proportions: see `Retarget`), skinned meshes and images.
 */
class CharacterPack {
    constructor(
        public skeleton: Skeleton,
        public meshes: PackMesh[],
        public images: PackImage[],
        public meta: [string, string][],
    ) { }

    public metaValue(key: string): string | undefined {
        return this.meta.find(([k]) => k === key)?.[1];
    }

    public image(name: string): PackImage | undefined {
        return this.images.find((i) => i.name === name);
    }

    public toBytes(): Uint8Array {
        const out = new Writer();
        out.header();
        out.section("META", (w) => w.meta(this.meta));
        out.section("SKEL", (w) => w.skeleton(this.skeleton));
        for (const m of this.meshes) out.section("MESH", (w) => w.mesh(m));
        for (const image of this.images) {
            out.section("IMAG", (w) => {
                w.str(image.name);
                w.str(image.mime);
                w.u32(image.bytes.length);
                w.bytes(image.bytes);
            });
        }
        return out.finish();
    }

    /** Read a character `.kmm`; throws a `PackError` saying what is wrong with it. */
    public static fromBytes(bytes: ArrayBuffer | Uint8Array): CharacterPack {
        let skeleton: Skeleton | undefined;
        const meshes: PackMesh[] = [], images: PackImage[] = [], meta: [string, string][] = [];
        forEachSection(bytes, "pack", (tag, r) => {
            switch (tag) {
                case "META": meta.push(...r.meta()); break;
                case "SKEL": skeleton = r.skeleton(); break;
                case "MESH": meshes.push(r.mesh()); break;
                case "IMAG": {
                    const name = r.str(), mime = r.str();
                    images.push({ name, mime, bytes: r.take(r.u32()).slice() });
                    break;
                }
            }
        });
        if (!skeleton) throw new PackError("the pack has no skeleton");
        if (meshes.length === 0) throw new PackError("the pack has no mesh");
        if (meshes.some((m) => m.mesh.skinJoints.some((j) => j >= skeleton!.length))) throw new PackError("a mesh is skinned to joints the skeleton lacks");
        return new CharacterPack(skeleton, meshes, images, meta);
    }
}

/** Check the header, then call `read` with each section's tag and a reader of its payload. */
function forEachSection(input: ArrayBuffer | Uint8Array, what: string, read: (tag: string, r: Reader) => void): void {
    const bytes = input instanceof Uint8Array ? input : new Uint8Array(input);
    if (bytes.length < 8 || String.fromCharCode(...bytes.subarray(0, 4)) !== MAGIC) throw new PackError(`not a Kansei ${what} (.kmm)`);
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    const version = view.getUint32(4, true);
    if (version !== VERSION) throw new PackError(`${what} version ${version}, this build reads ${VERSION}`);
    let at = 8;
    while (at < bytes.length) {
        const header = new Reader(bytes, at, `truncated ${what}`);
        const tag = String.fromCharCode(...header.take(4));
        const length = header.u64();
        const start = header.at;
        if (start + length > bytes.length) throw new PackError(`truncated ${what}`);
        read(tag, new Reader(bytes.subarray(start, start + length), 0, `truncated ${what}`));
        at = start + length;
    }
}

const decoder = new TextDecoder("utf-8", { fatal: true });
const encoder = new TextEncoder();

/** Little-endian reads over a section's payload; past its end throws a `PackError`. */
class Reader {
    private view: DataView;

    constructor(private bytes: Uint8Array, public at: number, private truncated: string) {
        this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }

    public take(n: number): Uint8Array {
        if (!(n >= 0) || this.at + n > this.bytes.length) throw new PackError(this.truncated);
        const out = this.bytes.subarray(this.at, this.at + n);
        this.at += n;
        return out;
    }

    private skip(n: number): number {
        if (this.at + n > this.bytes.length) throw new PackError(this.truncated);
        const at = this.at;
        this.at += n;
        return at;
    }

    public u8(): number {
        return this.view.getUint8(this.skip(1));
    }

    public u32(): number {
        return this.view.getUint32(this.skip(4), true);
    }

    public u64(): number {
        const at = this.skip(8);
        return this.view.getUint32(at, true) + this.view.getUint32(at + 4, true) * 2 ** 32;
    }

    public i32(): number {
        return this.view.getInt32(this.skip(4), true);
    }

    public f32(): number {
        return this.view.getFloat32(this.skip(4), true);
    }

    /** `n` floats, copied out (an aligned copy, read as one typed array). */
    public f32s(n: number): Float32Array {
        return new Float32Array(this.take(n * 4).slice().buffer);
    }

    public i16s(n: number): Int16Array {
        return new Int16Array(this.take(n * 2).slice().buffer);
    }

    public str(): string {
        try {
            return decoder.decode(this.take(this.u32()));
        } catch (e) {
            if (e instanceof PackError) throw e;
            throw new PackError("a motion pack string is not UTF-8");
        }
    }

    public meta(): [string, string][] {
        const n = this.u32();
        const out: [string, string][] = [];
        for (let i = 0; i < n; i++) out.push([this.str(), this.str()]);
        return out;
    }

    public transform(): Transform {
        const f = this.f32s(10);
        return new Transform(f.slice(0, 3) as vec3, f.slice(3, 7) as quat, f.slice(7, 10) as vec3);
    }

    public skeleton(): Skeleton {
        const n = this.u32();
        const names: string[] = [], parents: (number | null)[] = [], rest: Transform[] = [];
        for (let i = 0; i < n; i++) {
            names.push(this.str());
            const p = this.i32();
            if (p >= i) throw new PackError(`joint ${i} comes before its parent ${p}`);
            parents.push(p >= 0 ? p : null);
            rest.push(this.transform());
        }
        return new Skeleton(names, parents, rest);
    }

    public tracks(): Vec3Tracks {
        const joints = this.u32();
        const isConstant = new Uint8Array(joints);
        const constants = new Float32Array(joints * 3);
        const slot = new Int32Array(joints).fill(-1);
        let animatedJoints = 0;
        for (let j = 0; j < joints; j++) {
            if (this.u8() === 0) {
                isConstant[j] = 1;
                constants.set(this.f32s(3), 3 * j);
            } else {
                slot[j] = animatedJoints++;
            }
        }
        const center = new Float32Array(animatedJoints * 3), extent = new Float32Array(animatedJoints * 3);
        for (let k = 0; k < animatedJoints; k++) {
            center.set(this.f32s(3), 3 * k);
            extent.set(this.f32s(3), 3 * k);
        }
        const animated = this.i16s(this.u32() * 3);
        return new Vec3Tracks(joints, isConstant, constants, slot, animatedJoints, center, extent, animated);
    }

    public mesh(): PackMesh {
        const name = this.str();
        const color = Array.from(this.f32s(4)) as [number, number, number, number];
        const material = this.i32();
        const n = this.u32();
        const packed = this.f32s(n * 8);
        const vertices = new Float32Array(n * 9);
        for (let v = 0; v < n; v++) {
            const p = packed.subarray(8 * v, 8 * v + 8);
            vertices.set([p[0], p[1], p[2], 1, p[3], p[4], p[5], p[6], p[7]], 9 * v);
        }
        const words = new Uint32Array(this.take(n * 16).slice().buffer);
        const joints = new Uint16Array(n * 4);
        const weights = new Float32Array(n * 4);
        for (let v = 0; v < n; v++) {
            const w = words.subarray(4 * v, 4 * v + 4);
            joints.set([w[0] & 0xffff, w[0] >>> 16, w[1] & 0xffff, w[1] >>> 16], 4 * v);
            weights.set([(w[2] & 0xffff) / 65535, (w[2] >>> 16) / 65535, (w[3] & 0xffff) / 65535, (w[3] >>> 16) / 65535], 4 * v);
        }
        const indices = new Uint32Array(this.take(this.u32() * 4).slice().buffer);
        if (indices.some((i) => i >= n)) throw new PackError(`mesh '${name}' indexes past its vertices`);
        const skin = this.u32();
        const skinJoints: number[] = [];
        const inverseBind: mat4[] = [];
        for (let k = 0; k < skin; k++) {
            skinJoints.push(this.u32());
            inverseBind.push(this.f32s(16) as mat4);
        }
        if (joints.some((j) => j >= skin)) throw new PackError(`mesh '${name}' references a joint beyond its skin`);
        return { mesh: new SkinnedMesh(name, vertices, indices, joints, weights, skinJoints, inverseBind, material >= 0 ? material : undefined), color };
    }
}

/** Little-endian writes into growing sections. */
class Writer {
    private chunks: Uint8Array[] = [];
    private length = 0;

    public bytes(b: ArrayLike<number> | Uint8Array): void {
        const chunk = b instanceof Uint8Array ? b.slice() : Uint8Array.from(b);
        this.chunks.push(chunk);
        this.length += chunk.length;
    }

    private typed(buffer: ArrayBuffer): void {
        this.bytes(new Uint8Array(buffer));
    }

    public u8(x: number): void {
        this.bytes([x]);
    }

    public u32(x: number): void {
        this.typed(Uint32Array.of(x).buffer);
    }

    public i32(x: number): void {
        this.typed(Int32Array.of(x).buffer);
    }

    public f32(x: number): void {
        this.typed(Float32Array.of(x).buffer);
    }

    public f32s(x: ArrayLike<number>): void {
        this.typed(Float32Array.from(x).buffer);
    }

    public i16s(x: Int16Array): void {
        this.typed(x.slice().buffer);
    }

    public str(s: string): void {
        const b = encoder.encode(s);
        this.u32(b.length);
        this.bytes(b);
    }

    public meta(meta: [string, string][]): void {
        this.u32(meta.length);
        for (const [k, v] of meta) {
            this.str(k);
            this.str(v);
        }
    }

    public skeleton(skeleton: Skeleton): void {
        this.u32(skeleton.length);
        for (let j = 0; j < skeleton.length; j++) {
            this.str(skeleton.names[j]);
            this.i32(skeleton.parents[j] ?? -1);
            const t = skeleton.rest[j];
            this.f32s([...t.translation, ...t.rotation, ...t.scale]);
        }
    }

    public tracks(t: Vec3Tracks): void {
        this.u32(t.joints);
        for (let j = 0; j < t.joints; j++) {
            if (t.isConstant[j]) {
                this.u8(0);
                this.f32s(t.constants.subarray(3 * j, 3 * j + 3));
            } else {
                this.u8(1);
            }
        }
        for (let k = 0; k < t.animatedJoints; k++) {
            this.f32s(t.center.subarray(3 * k, 3 * k + 3));
            this.f32s(t.extent.subarray(3 * k, 3 * k + 3));
        }
        this.u32(t.animated.length / 3);
        this.i16s(t.animated);
    }

    public mesh(m: PackMesh): void {
        const mesh = m.mesh;
        this.str(mesh.name);
        this.f32s(m.color);
        this.i32(mesh.material ?? -1);
        const n = mesh.vertexCount;
        this.u32(n);
        const packed = new Float32Array(n * 8);
        for (let v = 0; v < n; v++) {
            const p = mesh.vertices.subarray(9 * v, 9 * v + 9);
            packed.set([p[0], p[1], p[2], p[4], p[5], p[6], p[7], p[8]], 8 * v);
        }
        this.f32s(packed);
        // joints and weights as the shader reads them (weights in unorm16)
        this.typed(mesh.skinWords().buffer as ArrayBuffer);
        this.u32(mesh.indices.length);
        this.typed(Uint32Array.from(mesh.indices).buffer);
        this.u32(mesh.skinJoints.length);
        mesh.skinJoints.forEach((j, k) => {
            this.u32(j);
            this.f32s(mesh.inverseBind[k]);
        });
    }

    /** The magic and version. */
    public header(): void {
        this.bytes(encoder.encode(MAGIC));
        this.u32(VERSION);
    }

    /** A section: its tag, its length (u64) and what `write` puts in it. */
    public section(tag: string, write: (w: Writer) => void): void {
        const w = new Writer();
        write(w);
        this.bytes(encoder.encode(tag));
        this.u32(w.length);
        this.u32(0);
        this.bytes(w.finish());
    }

    public finish(): Uint8Array {
        const out = new Uint8Array(this.length);
        let at = 0;
        for (const c of this.chunks) {
            out.set(c, at);
            at += c.length;
        }
        return out;
    }
}

export { MotionPack, CharacterPack, PackError, MAGIC, VERSION };
export type { PackMesh, PackImage };
