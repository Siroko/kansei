// rust/kansei-core/src/animation/motion_matching/tests.rs, ported.
import { mat4, quat, vec3 } from "gl-matrix";
import { assert, assertEq, test } from "../harness";
import { Clip } from "../../src/animation/Clip";
import { Pose } from "../../src/animation/Pose";
import { Retarget } from "../../src/animation/Retarget";
import { Skeleton } from "../../src/animation/Skeleton";
import { SkinnedMesh } from "../../src/animation/SkinnedMesh";
import { Transform } from "../../src/animation/Transform";
import {
    ActionClip, ActionKind, CharacterPack, Database, DatabaseBuilder, FEATURES, MotionMatcher, MotionPack, STRIDE, SearchFilter,
    defaultContactThresholds, defaultMotionMatchingSettings, defaultSearchFilter, findJointRoles, measureFootSlide, moveVelocity, yawOf,
} from "../../src/animation/motion_matching/index";
import type { Scenario } from "../../src/animation/motion_matching/index";

const RATE = 30;
const v = (x: number, y: number, z: number) => vec3.fromValues(x, y, z);
const rotX = (a: number) => quat.setAxisAngle(quat.create(), [1, 0, 0], a);
const rotY = (a: number) => quat.setAxisAngle(quat.create(), [0, 1, 0], a);

/**
 * A root on the ground, hips 1 m up, and two legs of a thigh, a calf and a foot (ankles 0.1 m
 * above the ground at rest).
 */
export function biped(): Skeleton {
    const t = (x: number, y: number) => Transform.fromTranslationRotation(v(x, y, 0), quat.create());
    return new Skeleton(
        ["root", "hips", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"],
        [null, 0, 1, 2, 3, 1, 5, 6],
        [t(0, 0), t(0, 1), t(0.1, 0), t(0, -0.45), t(0, -0.45), t(-0.1, 0), t(0, -0.45), t(0, -0.45)],
    );
}

/**
 * A clip of `frames` frames: the root travelling forward at `speed(t)` m/s while turning at
 * `turn` rad/s, the legs swinging in opposition once a second when moving.
 */
function locomotion(name: string, frames: number, speed: (t: number) => number, turn: number): Clip {
    const skeleton = biped();
    const position = vec3.create();
    let yaw = 0;
    const dt = 1 / RATE;
    const poses: Pose[] = [];
    for (let f = 0; f < frames; f++) {
        const t = f * dt;
        if (f > 0) {
            const s = speed(t - 0.5 * dt);
            vec3.add(position, position, vec3.transformQuat(vec3.create(), v(0, 0, s * dt), rotY(yaw + 0.5 * turn * dt)));
            yaw += turn * dt;
        }
        const swing = Math.sin(2 * Math.PI * t) * 0.4 * Math.min(speed(t) / 1.5, 1);
        const pose = Pose.rest(skeleton);
        pose.local[0] = Transform.fromTranslationRotation(position, rotY(yaw));
        pose.local[2].rotation = rotX(-swing);
        pose.local[5].rotation = rotX(swing);
        pose.local[3].rotation = rotX(0.3 * Math.abs(swing));
        pose.local[6].rotation = rotX(0.3 * Math.abs(swing));
        poses.push(pose);
    }
    return Clip.fromPoses(name, RATE, poses);
}

/** Idle and walk loops, a start, a stop and a left turn while walking. */
export function database(): Database {
    const skeleton = biped();
    const roles = findJointRoles(skeleton, "root", "hips", "foot_l", "foot_r");
    const builder = new DatabaseBuilder(skeleton, roles, RATE);
    builder.addClip(locomotion("idle", 61, () => 0, 0), true, 1);
    builder.addClip(locomotion("walk", 61, () => 1.5, 0), true, 2);
    builder.addClip(locomotion("start", 46, (t) => 1.5 * Math.min(t / 1, 1), 0), false, 2);
    builder.addClip(locomotion("stop", 46, (t) => 1.5 * Math.max(1 - t / 1, 0), 0), false, 2);
    builder.addClip(locomotion("turn_left", 61, () => 1.5, Math.PI / 2 / 1.5), false, 2);
    return builder.build();
}

export function clipIndex(db: Database, name: string): number {
    return db.clips.findIndex((c) => c.name === name);
}

const raw = (db: Database, frame: number) => Array.from(db.denormalize(db.features(frame)));

test("the database holds the clips, roots and trajectories", () => {
    const db = database();
    assertEq(db.clips.reduce((s, c) => s + c.frames, 0), db.frameCount);
    const walk = db.clips[clipIndex(db, "walk")];
    assertEq(db.clipOf(walk.start + 10), clipIndex(db, "walk"));
    // the root is taken out of the pose: its joint is the identity relative to the character
    assert(vec3.length(db.transform(walk.start + 20, 0).translation) < 1e-5);
    assert(vec3.distance(db.root(walk.start + 30).translation, v(0, 0, 1.5)) < 1e-4);
    // trajectory features: 1.5 m ahead in 1 s walking, still when idle, rotated when turning
    let r = raw(db, walk.start + 5);
    assert(Math.abs(r[20] - 1.5) < 1e-3 && Math.abs(r[19]) < 1e-3, `${r.slice(15, 21)}`);
    assert(Math.abs(r[26] - 1) < 1e-3, `${r.slice(21, 27)}`);
    const idle = raw(db, db.clips[clipIndex(db, "idle")].start + 5);
    assert(idle.slice(15, 21).every((x) => Math.abs(x) < 1e-4), `${idle.slice(15, 21)}`);
    const turn = raw(db, db.clips[clipIndex(db, "turn_left")].start);
    const heading = Math.atan2(turn[25], turn[26]);
    assert(Math.abs(heading - Math.PI / 2 / 1.5) < 0.02, `${heading}`);
    // a stop's trajectory ends where it stops; a clip's end goes on at its last velocity
    const stop = db.clips[clipIndex(db, "stop")];
    r = raw(db, stop.start + stop.frames - 1);
    assert(Math.abs(r[20]) < 1e-3, `${r.slice(15, 21)}`);
    const start = db.clips[clipIndex(db, "start")];
    r = raw(db, start.start + start.frames - 1);
    assert(Math.abs(r[20] - 1.5) < 0.02, `${r.slice(15, 21)}`);
    // idle feet are planted
    assertEq(db.contacts(db.clips[clipIndex(db, "idle")].start + 30), [true, true]);
    // normalized features average to zero
    for (let i = 0; i < FEATURES; i++) {
        let sum = 0;
        for (let f = 0; f < db.frameCount; f++) sum += db.features(f)[i];
        assert(Math.abs(sum / db.frameCount) < 1e-3, `feature ${i}: ${sum / db.frameCount}`);
    }
});

test("loops are seamless in velocity and root motion", () => {
    const db = database();
    const walk = clipIndex(db, "walk");
    const info = db.clips[walk];
    // a loop's first and last frames have the same features
    const first = db.features(info.start), last = db.features(info.start + info.frames - 1);
    assert(first.every((a, i) => Math.abs(a - last[i]) < 1e-3), `${first}\n${last}`);
    // playing across the seam moves the root on as if the clip went on
    let [moved, turned] = db.rootMotion(walk, 50, 70);
    assert(vec3.distance(moved, v(0, 0, 1)) < 1e-3, `${moved}`);
    assert(Math.abs(turned) < 1e-5);
    const turn = clipIndex(db, "turn_left");
    // a 1.5 m arc turning 60 degrees: its chord, bending left (+x from facing +z)
    [moved, turned] = db.rootMotion(turn, 0, 30);
    const angle = Math.PI / 2 / 1.5;
    assert(Math.abs(turned - angle) < 1e-3);
    const chord = 2 * (1.5 / angle) * Math.sin(0.5 * angle);
    assert(Math.abs(vec3.length(moved) - chord) < 0.01 && moved[0] > 0, `${moved}`);
});

function noise(i: number): number {
    const x = (((Math.imul(i, 747796405) + 2891336453) >>> 0) ^ (Math.imul(i >>> 7, 277803737) >>> 0)) >>> 0;
    return (x % 10007) / 10007 - 0.5;
}

test("the accelerated search agrees with brute force", () => {
    const db = database();
    for (let k = 0; k < 200; k++) {
        const frame = Math.trunc(Math.fround((noise(k) + 0.5) * (db.frameCount - 1)));
        const query = new Float32Array(STRIDE);
        query.set(db.features(frame).subarray(0, FEATURES));
        for (let i = 0; i < FEATURES; i++) query[i] += noise(k * 31 + i) * 0.8;
        const filters: SearchFilter[] = [defaultSearchFilter(), { ...defaultSearchFilter(), current: frame, ignoreNear: 5 }, { ...defaultSearchFilter(), tags: 1 }];
        for (const filter of filters) {
            const fast = db.search(query, filter)!;
            const slow = db.searchBruteForce(query, filter)!;
            assertEq(fast.frame, slow.frame, `query ${k}, ${JSON.stringify(filter)}`);
            assert(Math.abs(fast.cost - slow.cost) < 1e-4);
            if (filter.tags === 1) assertEq(db.clips[db.clipOf(fast.frame)].name, "idle");
            if (filter.current !== undefined) assert(Math.abs(fast.frame - filter.current) > 5 || db.clipOf(fast.frame) !== db.clipOf(filter.current));
        }
    }
    // an exact query finds its own frame, and nothing beats a cost it can't improve on
    const walk = db.clips[clipIndex(db, "walk")];
    const exact = new Float32Array(db.features(walk.start + 12));
    let found = db.search(exact, defaultSearchFilter())!;
    assert(found.cost < 1e-6 && db.clipOf(found.frame) === clipIndex(db, "walk"), JSON.stringify(found));
    assert(db.search(exact, defaultSearchFilter(), 0) === undefined);
    // frames at the end of clips that don't loop are never found
    const stop = db.clips[clipIndex(db, "stop")];
    exact.set(db.features(stop.start + stop.frames - 1));
    found = db.search(exact, defaultSearchFilter())!;
    assert(found.frame < stop.start + stop.frames - 10 || !stop.contains(found.frame));
});

function triangleMesh(): SkinnedMesh {
    const vertex = (x: number, y: number) => [x, y, 0, 1, 0, 0, 1, x, y];
    return new SkinnedMesh(
        "tri",
        new Float32Array([...vertex(0, 0), ...vertex(1, 0), ...vertex(0, 1)]),
        new Uint32Array([0, 1, 2]),
        new Uint16Array([1, 0, 0, 0, 4, 1, 0, 0, 7, 4, 1, 0]),
        new Float32Array([1, 0, 0, 0, 0.5, 0.5, 0, 0, 0.25, 0.25, 0.5, 0]),
        Array.from({ length: 8 }, (_, i) => i),
        Array.from({ length: 8 }, (_, i) => mat4.fromTranslation(mat4.create(), [i, i, i])),
        0,
    );
}

/** Rust's derived `PartialEq` on `Database`: every stored field. */
function sameDatabase(a: Database, b: Database): boolean {
    const same = (x: ArrayLike<number>, y: ArrayLike<number>) => x.length === y.length && Array.from(x).every((value, i) => value === y[i]);
    const sameTracks = (x: Database["translations"], y: Database["translations"]) => x.joints === y.joints && x.animatedJoints === y.animatedJoints
        && same(x.isConstant, y.isConstant) && same(x.constants, y.constants) && same(x.slot, y.slot)
        && same(x.center, y.center) && same(x.extent, y.extent) && same(x.animated, y.animated);
    return JSON.stringify([a.skeleton.names, a.skeleton.parents, a.roles, a.sampleRate, a.weights, a.clips]) === JSON.stringify([b.skeleton.names, b.skeleton.parents, b.roles, b.sampleRate, b.weights, b.clips])
        && a.skeleton.rest.every((t, i) => t.equals(b.skeleton.rest[i]))
        && same(a.rotations, b.rotations) && sameTracks(a.translations, b.translations) && sameTracks(a.scales, b.scales)
        && same(a.rootTranslations, b.rootTranslations) && same(a.rootRotations, b.rootRotations) && same(a.contactBits, b.contactBits)
        && same(a.featureOffset, b.featureOffset) && same(a.featureScale, b.featureScale) && same(a.featureRows, b.featureRows);
}

test("a pack round trips", () => {
    const action = new ActionClip({ clip: 2, kind: ActionKind.Vault, height: 1.1, ledge: v(0.1, 1.1, 0.4), forward: v(0, 0, 1), rise: 10, anchor: 14, onTop: 15, offTop: 18, down: 22, exit: 22, span: 0.4, lastEntry: 4 });
    const pack = new MotionPack(database(), [{ mesh: triangleMesh(), color: [0.5, 0.4, 0.3, 1] }], [action], [["source", "synthetic"]]);
    const bytes = pack.toBytes();
    const back = MotionPack.fromBytes(bytes);
    assert(sameDatabase(back.database, pack.database), "the database round trips");
    // (the pack stores f32s)
    const f32 = (a: ActionClip) => JSON.stringify(a, (_, x) => typeof x === "number" ? Math.fround(x) : ArrayBuffer.isView(x) ? Array.from(x as Float32Array) : x);
    assertEq(back.actions.map(f32), pack.actions.map(f32));
    assertEq(back.metaValue("source"), "synthetic");
    const a = back.meshes[0].mesh, b = pack.meshes[0].mesh;
    assertEq([Array.from(a.indices), Array.from(a.joints), a.skinJoints, a.inverseBind.map((m) => Array.from(m)), a.material],
        [Array.from(b.indices), Array.from(b.joints), b.skinJoints, b.inverseBind.map((m) => Array.from(m)), b.material]);
    assertEq(Array.from(a.skinWords()), Array.from(b.skinWords()));
    assertEq(back.meshes[0].color, [0.5, 0.4, 0.3, 1].map((x) => Math.fround(x)));
    assertEq(Array.from(a.vertices), Array.from(b.vertices));
    // a character pack: skeleton, mesh and images
    const character = new CharacterPack(pack.database.skeleton, pack.meshes, [{ name: "base_color", mime: "image/webp", bytes: Uint8Array.of(1, 2, 3, 4, 5) }], [["source", "synthetic"]]);
    const backCharacter = CharacterPack.fromBytes(character.toBytes());
    assert(backCharacter.skeleton.rest.every((t, i) => t.equals(character.skeleton.rest[i])) && backCharacter.skeleton.names.join() === character.skeleton.names.join());
    assertEq(Array.from(backCharacter.image("base_color")!.bytes), [1, 2, 3, 4, 5]);
    assertEq(Array.from(backCharacter.meshes[0].mesh.skinWords()), Array.from(character.meshes[0].mesh.skinWords()));
    const throws = (f: () => unknown) => {
        try {
            f();
            return false;
        } catch {
            return true;
        }
    };
    assert(throws(() => CharacterPack.fromBytes(character.toBytes().subarray(0, 40))));
    // not a pack, truncated, or corrupt: errors, not crashes
    assert(throws(() => MotionPack.fromBytes(new TextEncoder().encode("nope"))));
    assert(throws(() => MotionPack.fromBytes(bytes.subarray(0, bytes.length >> 1))));
    const otherVersion = bytes.slice();
    otherVersion[4] = 99;
    assert(throws(() => MotionPack.fromBytes(otherVersion)));
});

function run(matcher: MotionMatcher, db: Database, velocity: vec3, seconds: number): void {
    const dt = 1 / 60;
    for (let i = 0; i < Math.trunc(Math.fround(seconds / Math.fround(dt))); i++) {
        matcher.update(db, { velocity }, dt);
        const gap = vec3.subtract(vec3.create(), matcher.character.translation, matcher.simulation.position);
        assert(Math.hypot(gap[0], gap[2]) <= matcher.settings.clampDistance + 1e-4, `${gap}`);
    }
}

const playing = (matcher: MotionMatcher, db: Database) => db.clips[matcher.playing()[0]].name;

test("the character idles, walks when asked and stops when released", () => {
    const db = database();
    const matcher = new MotionMatcher(db, defaultMotionMatchingSettings(), v(0, 0, 0), 0);
    run(matcher, db, v(0, 0, 0), 1);
    assertEq(playing(matcher, db), "idle");
    assert(vec3.length(matcher.character.translation) < 1e-3);
    assertEq(matcher.feetLocked(), [true, true]);

    run(matcher, db, v(0, 0, 1.5), 3);
    let name = playing(matcher, db);
    assert(name === "walk" || name === "start", name);
    const z = matcher.character.translation[2];
    assert(z > 3 && z < 4.5, `${z}`);

    run(matcher, db, v(0, 0, 0), 3);
    name = playing(matcher, db);
    assert(name === "idle" || name === "stop", name);
    assert(vec3.length(matcher.simulation.velocity) < 0.01);
    const before = vec3.clone(matcher.character.translation);
    run(matcher, db, v(0, 0, 0), 1);
    assert(vec3.distance(matcher.character.translation, before) < 0.02, "stands still");
});

test("the character turns to face where it goes", () => {
    const db = database();
    const matcher = new MotionMatcher(db, defaultMotionMatchingSettings(), v(0, 0, 0), 0);
    run(matcher, db, v(0, 0, 1.5), 2);
    run(matcher, db, v(1.5, 0, 0), 3);
    // the database turns 60 degrees at most in one go, and the character is only turned toward
    // the simulation while its animation turns (adjustByVelocity): it turns at least that far
    const yaw = yawOf(matcher.character.rotation);
    assert(yaw > 0.9 && yaw < Math.PI / 2 + 0.1, `${yaw}`);
    assert(Math.abs(yawOf(matcher.simulation.rotation) - Math.PI / 2) < 0.01);
    assert(matcher.character.translation[0] > 2, `${matcher.character.translation}`);
});

test("searches run on their interval and switch with inertialization", () => {
    const db = database();
    const matcher = new MotionMatcher(db, defaultMotionMatchingSettings(), v(0, 0, 0), 0);
    const dt = 1 / 60;
    let searches = 0;
    for (let i = 0; i < 120; i++) {
        matcher.update(db, { velocity: v(0, 0, 0) }, dt);
        searches += matcher.lastSearch.searched ? 1 : 0;
    }
    // every 0.1 s over 2 s, plus the first
    assert(searches >= 19 && searches <= 22, `${searches}`);
    // asking to walk searches at once and switches
    matcher.update(db, { velocity: v(0, 0, 1.5) }, dt);
    assert(matcher.lastSearch.searched);
    const poseBefore = matcher.pose.clone();
    matcher.update(db, { velocity: v(0, 0, 1.5) }, dt);
    // no pop: consecutive output poses stay close even across the switch
    poseBefore.local.forEach((a, i) => assert(Math.abs(quat.dot(a.rotation, matcher.pose.local[i].rotation)) > 0.99, `joint ${i}`));
});

test("a display skeleton shows the pose with its own proportions", () => {
    const db = database();
    // the biped with 20% shorter legs and hips 20% lower
    const short = new Skeleton([...db.skeleton.names], [...db.skeleton.parents], db.skeleton.rest.map((t) => t.clone()));
    short.names.forEach((name, j) => {
        if (name.startsWith("calf") || name.startsWith("foot") || name === "hips") vec3.scale(short.rest[j].translation, short.rest[j].translation, 0.8);
    });
    const matcher = new MotionMatcher(db, defaultMotionMatchingSettings(), v(0, 0, 0), 0);
    matcher.setDisplay(db, { skeleton: short, retarget: new Retarget(db.skeleton, short, Retarget.UNREAL_KEEP) });
    run(matcher, db, v(0, 0, 0), 1);
    let model = matcher.model;
    const hips = short.find("hips")!;
    assert(Math.abs(model[hips].translation[1] - 0.8) < 0.02, `hips at ${model[hips].translation[1]}`);
    // standing, both feet of the short legs are planted on the ground they reach
    assertEq(matcher.feetLocked(), [true, true]);
    const foot = short.find("foot_l")!;
    assert(Math.abs(model[foot].translation[1] - 0.08) < 0.02, `foot at ${model[foot].translation[1]}`);
    run(matcher, db, v(0, 0, 1.5), 2);
    assertEq(matcher.pose.length, short.length);
    assert(matcher.character.translation[2] > 1.5);
    // and back to the database's own skeleton
    matcher.setDisplay(db);
    model = matcher.model;
    assert(Math.abs(model[1].translation[1] - 1) < 0.1);
});

test("a foot on its ball is planted while its ankle moves", () => {
    // feet straight under the root, each with a ball 0.15 m ahead: the translations place them
    const t = (x: number, y: number, z: number) => Transform.fromTranslationRotation(v(x, y, z), quat.create());
    const skeleton = new Skeleton(
        ["root", "hips", "foot_l", "ball_l", "foot_r", "ball_r"],
        [null, 0, 0, 2, 0, 4],
        [t(0, 0, 0), t(0, 1, 0), t(0.1, 0.1, 0), t(0, -0.07, 0.15), t(-0.1, 0.1, 0), t(0, -0.07, 0.15)],
    );
    const roles = findJointRoles(skeleton, "root", "hips", "foot_l", "foot_r");
    // standing still; frames 10-19 the left ankle circles 4 cm round its rest, fast (1.5 m/s),
    // its ball still; frames 20-29 both swing high
    const poses: Pose[] = [];
    for (let f = 0; f < 40; f++) {
        const pose = Pose.rest(skeleton);
        if (f >= 10 && f < 20) {
            const a = f * 1.25;
            const ankle = v(0.1, 0.1 + Math.sin(a) * 0.04, Math.cos(a) * 0.04);
            pose.local[2].translation = ankle;
            pose.local[3].translation = vec3.subtract(vec3.create(), v(0.1, 0.03, 0.15), ankle);
        } else if (f >= 20 && f < 30) {
            pose.local[2].translation[1] += 0.3;
            pose.local[4].translation[1] += 0.3;
        }
        poses.push(pose);
    }
    const builder = new DatabaseBuilder(skeleton, roles, RATE);
    builder.addClip(Clip.fromPoses("tap", RATE, poses), false, 1);
    const db = builder.build();
    const left = (db: Database) => Array.from({ length: db.frameCount }, (_, f) => db.contacts(f)[0]);
    assert(left(db).slice(11, 19).every((c) => c), `${left(db)}`);
    assert(left(db).slice(21, 29).every((c) => !c), `${left(db)}`);
    // the ankle alone would have it lifted
    assert(Math.hypot(...raw(db, 14).slice(6, 9)) > 1, `${raw(db, 14).slice(6, 9)}`);
    // found again from the poses, the same
    const baked = db.contactBits.slice();
    db.detectContacts(defaultContactThresholds());
    assertEq(Array.from(db.contactBits), Array.from(baked));
});

test("a pack from before the ball contacts gets them found again", () => {
    const db = database();
    const baked = Array.from(db.contactBits);
    const stale = database();
    stale.contactBits.fill(0);
    const bytes = new MotionPack(stale, [], [], []).toBytes();
    // a pack that says how its contacts were found keeps them
    assertEq(Array.from(MotionPack.fromBytes(bytes).database.contactBits), Array.from(stale.contactBits));
    // one without the CRUL section gets them from the poses
    const tag = Array.from("CRUL", (c) => c.charCodeAt(0));
    const at = bytes.findIndex((_, i) => tag.every((c, k) => bytes[i + k] === c));
    const length = Number(new DataView(bytes.buffer, bytes.byteOffset + at + 4, 8).getBigUint64(0, true));
    const old = new Uint8Array([...bytes.subarray(0, at), ...bytes.subarray(at + 12 + length)]);
    assertEq(Array.from(MotionPack.fromBytes(old).database.contactBits), baked);
});

test("foot locking keeps planted feet from sliding", () => {
    const db = database();
    const walk: Scenario = { name: "walk", move: { kind: "straight", speed: 1.5 }, run: false, settle: 2, seconds: 4 };
    const slide = (lock: boolean) => {
        const settings = { ...defaultMotionMatchingSettings(), footLock: lock };
        return measureFootSlide(db, db, new MotionMatcher(db, settings, vec3.create(), 0), walk);
    };
    const free = slide(false), locked = slide(true);
    assert(free.planted > 0.2, JSON.stringify(free));
    // the synthetic legs swing without planting: their feet slide unless locked
    assert(free.cmPerSecond > 5, JSON.stringify(free));
    assert(locked.cmPerSecond < 0.3 * free.cmPerSecond, `${JSON.stringify(locked)} vs ${JSON.stringify(free)}`);
    assert(locked.yawGap < 1, JSON.stringify(locked));
    // a circle's heading turns at speed / radius, to the left (toward +x from +z) for `left`
    const quarter = Math.PI / 4;
    const close = (a: vec3, b: vec3) => vec3.distance(a, b) < 1e-4;
    assert(close(moveVelocity({ kind: "circle", radius: 2, speed: 4, left: true }, quarter), v(4, 0, 0)));
    assert(close(moveVelocity({ kind: "circle", radius: 2, speed: 4, left: false }, quarter), v(-4, 0, 0)));
});
