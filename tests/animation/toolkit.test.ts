// The inline tests of rust/kansei-core/src/animation/{springs,inertialization,ik,retarget,warping}.rs, ported.
import { quat, vec3 } from "gl-matrix";
import { assert, assertClose, assertEq, sameRotation, test } from "../harness";
import {
    damperExact, decaySpringDamperExact, decaySpringDamperExactQuat, springCharacterUpdate, springDamperExact, springDamperExactQuat,
} from "../../src/animation/Springs";
import { Inertializer, poseVelocities } from "../../src/animation/Inertialization";
import { FootLock, twoJointIK } from "../../src/animation/IK";
import type { FootPose } from "../../src/animation/IK";
import { Retarget, TranslationMode } from "../../src/animation/Retarget";
import { Placement, Ramp, Root, RootWarp } from "../../src/animation/Warping";
import { Pose } from "../../src/animation/Pose";
import { Skeleton } from "../../src/animation/Skeleton";
import { Transform, quatToScaledAngleAxis } from "../../src/animation/Transform";

const v = (x: number, y: number, z: number) => vec3.fromValues(x, y, z);
const rotX = (a: number) => quat.setAxisAngle(quat.create(), [1, 0, 0], a);
const rotY = (a: number) => quat.setAxisAngle(quat.create(), [0, 1, 0], a);
const rotZ = (a: number) => quat.setAxisAngle(quat.create(), [0, 0, 1], a);
const tr = (t: vec3, r: quat = quat.create()) => Transform.fromTranslationRotation(t, r);

// ── springs.rs ──

test("springs: a damper covers half the way in a half-life", () => {
    const x = damperExact(v(0, 0, 0), v(1, 0, 0), 0.2, 0.2);
    assert(Math.abs(x[0] - 0.5) < 1e-3, `${x}`);
});

test("springs: settle on their goal and are step independent", () => {
    const run = (steps: number) => {
        const x = v(1, -2, 0.5), vel = v(0, 3, 0);
        const dt = 2 / steps;
        for (let i = 0; i < steps; i++) springDamperExact(x, vel, v(4, 0, 0), 0.3, dt);
        return [x, vel];
    };
    const [a, va] = run(10);
    const [b, vb] = run(1000);
    assertClose(a, b, 1e-3);
    assertClose(va, vb, 1e-3);
    assertClose(a, [4, 0, 0], 1e-2);
    // a decaying offset vanishes
    const x = v(0.3, 0, -0.2), vel = v(0, 0, 0);
    decaySpringDamperExact(x, vel, 0.1, 1);
    assert(vec3.length(x) < 1e-3 && vec3.length(vel) < 1e-2);
});

test("springs: a rotation spring turns to its goal", () => {
    const q = quat.create(), w = v(0, 0, 0);
    const goal = rotY(2);
    let lastAngle = 0;
    for (let i = 0; i < 120; i++) {
        springDamperExactQuat(q, w, goal, 0.2, 1 / 60);
        const angle = quatToScaledAngleAxis(q)[1];
        // critically damped: no overshoot
        assert(angle >= lastAngle - 1e-4 && angle <= 2 + 1e-3, `${angle}`);
        lastAngle = angle;
    }
    assert(Math.abs(quat.dot(q, goal)) > 0.9999);
    const r = rotX(0.5), wr = v(0, 0, 0);
    decaySpringDamperExactQuat(r, wr, 0.1, 1);
    assert(Math.abs(r[3]) > 0.99999);
});

test("springs: a character spring reaches its goal velocity and integrates position", () => {
    const x = v(0, 0, 0), vel = v(0, 0, 0), a = v(0, 0, 0);
    const goal = v(0, 0, 2);
    const dt = 1 / 60;
    const integrated = v(0, 0, 0);
    for (let i = 0; i < 180; i++) {
        const before = vec3.clone(vel);
        springCharacterUpdate(x, vel, a, goal, 0.25, dt);
        vec3.scaleAndAdd(integrated, integrated, vec3.add(before, before, vel), 0.5 * dt);
    }
    assertClose(vel, goal, 1e-3);
    assert(vec3.length(a) < 1e-2);
    // the closed-form position is the velocity's integral
    assertClose(x, integrated, 1e-3);
    // one big step lands where many small ones do
    const x1 = v(0, 0, 0), v1 = v(0, 0, 0), a1 = v(0, 0, 0);
    springCharacterUpdate(x1, v1, a1, goal, 0.25, 3);
    assertClose(x1, x, 1e-3);
});

// ── inertialization.rs ──

function pose(angle: number, x: number): Pose {
    return new Pose([tr(v(x, 0, 0), rotY(angle)), tr(v(x, 0, 0), rotY(angle))]);
}

function clonePose(p: Pose): Pose {
    return new Pose(p.local.map((t) => t.clone()));
}

test("inertialization: the output does not jump and then settles on the destination", () => {
    const source = pose(0.5, 1), destination = pose(-0.3, -0.5);
    const zero = [v(0, 0, 0), v(0, 0, 0)];
    const inertializer = new Inertializer(2);
    inertializer.transition(source, zero, zero, destination, zero, zero);
    // right after the switch the output is the source
    let out = clonePose(destination);
    inertializer.update(out, 0.1, 0);
    assertClose(out.local[0].translation, source.local[0].translation, 1e-5);
    assert(sameRotation(out.local[0].rotation, source.local[0].rotation, 1e-6));
    // it moves toward the destination without overshooting, and gets there
    let last = Infinity;
    for (let i = 0; i < 60; i++) {
        out = clonePose(destination);
        inertializer.update(out, 0.1, 1 / 60);
        const gap = vec3.distance(out.local[0].translation, destination.local[0].translation);
        assert(gap <= last + 1e-6, `${gap} after ${last}`);
        last = gap;
    }
    assert(last < 1e-3 && inertializer.largestAngle() < 1e-3, `${last} ${inertializer.largestAngle()}`);
});

test("inertialization: a second transition keeps the output continuous", () => {
    const zero = [v(0, 0, 0), v(0, 0, 0)];
    const inertializer = new Inertializer(2);
    inertializer.transition(pose(0, 0), zero, zero, pose(1, 2), zero, zero);
    const shown = pose(1, 2);
    inertializer.update(shown, 0.2, 0.05);
    // switch again mid-blend: the next frame starts from what was showing
    inertializer.transition(pose(1, 2), zero, zero, pose(-1, -3), zero, zero);
    const next = pose(-1, -3);
    inertializer.update(next, 0.2, 0);
    assertClose(next.local[1].translation, shown.local[1].translation, 1e-4);
    assert(sameRotation(next.local[1].rotation, shown.local[1].rotation, 1e-5));
});

test("inertialization: velocities carry across the switch", () => {
    // the source moves at +1 m/s, the destination stands: the output keeps moving for a while
    const inertializer = new Inertializer(1);
    const still = new Pose([Transform.identity()]);
    inertializer.transition(still, [v(1, 0, 0)], [v(0, 0, 0)], still, [v(0, 0, 0)], [v(0, 0, 0)]);
    const out = clonePose(still);
    inertializer.update(out, 0.2, 1 / 60);
    assert(out.local[0].translation[0] > 0.01, `${out.local[0].translation}`);
    const { linear, angular } = poseVelocities([Transform.identity()], [tr(v(1, 0, 0), rotZ(0.1))], 0.5);
    assertClose(linear[0], [2, 0, 0], 1e-6);
    assertClose(angular[0], [0, 0, 0.2], 1e-5);
});

// ── ik.rs ──

/** A leg: hip at 1 m, knee 0.5 m below, ankle 0.5 m below that, slightly bent forward. */
function leg(): [Skeleton, Pose] {
    const skeleton = new Skeleton(["hip", "knee", "ankle"], [null, 0, 1], [
        tr(v(0, 1, 0), rotX(-0.2)),
        tr(v(0, -0.5, 0), rotX(0.4)),
        tr(v(0, -0.5, 0), rotX(-0.2)),
    ]);
    return [skeleton, Pose.rest(skeleton)];
}

test("ik: two-joint IK reaches the target and keeps the foot rotation", () => {
    const [skeleton, p] = leg();
    const model = p.model(skeleton);
    const footRotation = quat.clone(model[2].rotation);
    const lengths = [vec3.distance(model[1].translation, model[0].translation), vec3.distance(model[2].translation, model[1].translation)];
    for (const target of [v(0.1, 0.2, 0.15), v(-0.2, 0.4, 0.3), v(0, 0.1, -0.2)]) {
        twoJointIK(skeleton, p, model, 0, 1, 2, target);
        // the model transforms are the pose's
        const again = p.model(skeleton);
        model.forEach((x, i) => assertClose(x.translation, again[i].translation, 1e-5));
        assertClose(model[2].translation, target, 1e-4);
        assert(sameRotation(model[2].rotation, footRotation, 1e-5));
        // bones keep their lengths
        assert(Math.abs(vec3.distance(model[1].translation, model[0].translation) - lengths[0]) < 1e-5);
        assert(Math.abs(vec3.distance(model[2].translation, model[1].translation) - lengths[1]) < 1e-5);
    }
    // out of reach: the leg straightens toward the target
    twoJointIK(skeleton, p, model, 0, 1, 2, v(0, -2, 0));
    assert(vec3.length(model[2].translation) < 1e-2, `${model[2].translation}`);
});

test("ik: a planted foot stays until released, then blends back", () => {
    const lock = new FootLock();
    const dt = 1 / 60;
    // swinging, then touching down at x = 0.3
    assertClose(lock.update(v(0, 0.1, 0), false, 0.2, 0.1, dt), v(0, 0.1, 0), 0);
    const planted = lock.update(v(0.3, 0, 0), true, 0.2, 0.1, dt);
    assert(lock.isLocked());
    // the animated foot slides 10 cm while planted: the output stays
    const still = lock.update(v(0.4, 0, 0), true, 0.2, 0.1, dt);
    assertClose(still, planted, 0);
    // past the radius it lets go, starting from where it was, and is pinned again where the
    // animation has it, the output fading over
    const released = lock.update(v(0.6, 0, 0), true, 0.2, 0.1, dt);
    assert(lock.isLocked());
    assert(vec3.distance(released, planted) < 0.05, `${released}`);
    let out = released;
    for (let i = 0; i < 60; i++) out = lock.update(v(0.65, 0, 0), true, 0.2, 0.1, dt);
    assert(vec3.distance(out, v(0.6, 0, 0)) < 1e-3, `${out}`);
    // lifted, it follows the animation
    for (let i = 0; i < 60; i++) out = lock.update(v(0.9, 0.1, 0), false, 0.2, 0.1, dt);
    assert(vec3.distance(out, v(0.9, 0.1, 0)) < 1e-3, `${out}`);
});

test("ik: a rolling foot stays on its ball once the heel lifts", () => {
    const lock = new FootLock();
    const dt = 1 / 60;
    const foot = (ankle: vec3, ball: vec3, ankleLift: number): FootPose => ({ ankle, ball, ankleLift, ballLift: 0 });
    const toe = (f: FootPose) => vec3.subtract(vec3.create(), f.ball, f.ankle);
    // flat on the ground, the ball 0.15 m ahead of the ankle: the ankle is pinned
    const flat = foot(v(0, 0.1, 0), v(0, 0.03, 0.15), 0);
    assertClose(lock.updateFoot(flat, true, 0.2, 0.1, dt), flat.ankle, 0);
    // the heel rises 4 cm while the animated foot slides 2 cm: the pin moves to the ball
    // where the output has it, the ankle staying put
    const slide = (z: number) => v(0, 0, z);
    const plus = (a: vec3, b: vec3) => vec3.add(vec3.create(), a, b);
    const rolled = foot(plus(v(0, 0.14, 0.02), slide(0.02)), plus(flat.ball, slide(0.02)), 0.04);
    let target = lock.updateFoot(rolled, true, 0.2, 0.1, dt);
    assert(vec3.distance(target, flat.ankle) < 1e-5, `${target}`);
    const ball = plus(target, toe(rolled));
    // it rolls further and slides on: the ball stays, the ankle above it where the roll puts it
    const further = foot(plus(v(0, 0.17, 0.05), slide(0.06)), plus(flat.ball, slide(0.06)), 0.07);
    target = lock.updateFoot(further, true, 0.2, 0.1, dt);
    assert(lock.isLocked());
    assert(vec3.distance(plus(target, toe(further)), ball) < 1e-5, `${target}`);
    // heel down again: back on the ankle, without a jump
    const again = lock.updateFoot({ ...further, ankleLift: 0 }, true, 0.2, 0.1, dt);
    assert(vec3.distance(again, target) < 1e-5, `${again} vs ${target}`);
});

// ── retarget.rs ──

const t = (x: number, y: number, z: number) => tr(v(x, y, z));

/** root > pelvis (1 m up) > thigh (0.1 m out) > calf (0.45 m down); root > ik_foot. */
function source(): Skeleton {
    return new Skeleton(["root", "pelvis", "thigh_l", "calf_l", "ik_foot_l"], [null, 0, 1, 2, 0],
        [t(0, 0, 0), t(0, 1, 0), t(0.1, 0, 0), t(0, -0.45, 0), t(0.1, 0.1, 0)]);
}

/** Shorter legs (pelvis at 0.8 m, calf 0.35 m, slightly forward), no IK joint, an extra head. */
function target(): Skeleton {
    return new Skeleton(["root", "pelvis", "thigh_l", "calf_l", "head"], [null, 0, 1, 2, 1],
        [t(0, 0, 0), t(0, 0.8, 0), t(0.12, 0, 0), t(0, -0.35, 0.02), t(0, 0.7, 0)]);
}

test("retarget: modes follow the names and the rest translations", () => {
    const r = new Retarget(source(), target(), Retarget.UNREAL_KEEP);
    const { Animation, OrientAndScale } = TranslationMode;
    assertEq(r.modes(), [Animation, OrientAndScale, OrientAndScale, OrientAndScale, undefined]);
});

test("retarget: rotations carry over and bones take the target lengths", () => {
    const s = source(), tg = target();
    const r = new Retarget(s, tg, Retarget.UNREAL_KEEP);
    const p = Pose.rest(s);
    // the root walked 2 m, the pelvis bobbed 5 cm down, the knee bent
    p.local[0].translation = v(0, 0, 2);
    p.local[1].translation = v(0, 0.95, 0);
    p.local[3].rotation = rotX(0.8);
    const out = new Pose([]);
    r.apply(p, out);
    assertEq(out.length, 5);
    // root: the animation's; pelvis: its motion scaled to the shorter legs (0.95 * 0.8)
    assertClose(out.local[0].translation, [0, 0, 2], 0);
    assertClose(out.local[1].translation, [0, 0.76, 0], 1e-5);
    // a still bone: exactly the target's, direction included
    assertClose(out.local[3].translation, tg.rest[3].translation, 1e-5);
    assertClose(out.local[3].rotation, rotX(0.8), 1e-6);
    // a joint the source lacks keeps the target's rest
    assert(out.local[4].equals(tg.rest[4]));
    // the rest pose maps onto the target's rest pose
    r.apply(Pose.rest(s), out);
    out.local.forEach((a, i) => assertClose(a.translation, tg.rest[i].translation, 1e-5));
});

// ── warping.rs ──

/**
 * A clip root walking +z at 2 m/s for 60 frames at 30 fps, rising 1 m between frames 20 and
 * 30 (onto a 1 m obstacle whose edge is at z = 0 in clip space).
 */
function clipRoot(frame: number): Root {
    const z = -2 + frame * 2 / 30;
    const y = Math.min(Math.max((frame - 20) / 10, 0), 1);
    return [v(0, y, z), 0];
}

test("warping: ramps ease between their frames", () => {
    const r = new Ramp(10, 20, 1, 3);
    assertEq([r.at(0), r.at(10), r.at(15), r.at(20), r.at(99)], [1, 1, 2, 3, 3]);
    assert(r.at(12) < 1 + 2 * 0.2, "eased, slow at first");
    assertEq(new Ramp(5, 5, 0, 1).at(4), 0);
    assertEq(new Ramp(5, 5, 0, 1).at(5), 1);
});

test("warping: placements map clip roots to world ones", () => {
    const p = Placement.between([v(1, 0, 2), 0.3], [v(-4, 0, 7), 1.8]);
    assertClose(p.apply(v(1, 0.5, 2)), [-4, 0.5, 7], 1e-5);
    assert(Math.abs(p.yaw - 1.5) < 1e-6);
});

test("warping: the warp starts at the character and lands the moment on the target", () => {
    // the character is 1 m to the side of where the clip would start, turned 20 degrees
    const start = 0;
    const character: Root = [v(5, 0.2, 3), 0.35];
    const warp = RootWarp.identity(clipRoot(start), character);
    // the real obstacle: 1.4 m high, its edge at (10, 1.6, 8); the world approach heading is 90 degrees
    const edge = v(10, 0.2 + 1.4, 8);
    const anchor = 25;
    warp.to = Placement.between([v(0, 0, 0), 0], [v(edge[0], 0, edge[2]), Math.PI / 2]);
    warp.window = new Ramp(start, anchor, 0, 1);
    warp.lift = [new Ramp(20, anchor, 0, 0.4)];
    // frame 0: exactly where the character is
    let [p, yaw] = warp.root(start, clipRoot(start));
    assertClose(p, character[0], 1e-5);
    assert(Math.abs(yaw - character[1]) < 1e-6, `${yaw}`);
    // at the anchor and after: in the target's frame, and 1.4 m up on the obstacle by frame 30
    [p, yaw] = warp.root(30, clipRoot(30));
    assertClose(p, [10, 1.6, 8], 1e-4);
    assert(Math.abs(yaw - Math.PI / 2) < 1e-5);
    [p] = warp.root(45, clipRoot(45));
    assertClose(p, [11, 1.6, 8], 1e-4);
    // continuous: no step between frames anywhere
    let last = warp.root(0, clipRoot(0))[0];
    for (let k = 1; k <= 600; k++) {
        const f = k * 0.1;
        [p] = warp.root(f, clipRoot(f));
        assert(vec3.distance(p, last) < 0.1, `frame ${f}: ${vec3.distance(p, last)} m`);
        last = p;
    }
});

test("warping: stretching lengthens the path along a direction", () => {
    const warp = RootWarp.identity(clipRoot(0), [v(0, 0, -2), 0]);
    warp.stretch = [new Ramp(30, 40, 0, 0.5)];
    warp.stretchDirection = v(0, 0, 1);
    assertClose(warp.root(30, clipRoot(30))[0], clipRoot(30)[0], 1e-5);
    assertClose(warp.root(50, clipRoot(50))[0], vec3.add(vec3.create(), clipRoot(50)[0], v(0, 0, 0.5)), 1e-5);
});
