// rust/kansei-core/src/gi/clipmap.rs and clipmap_scene.rs tests, ported.
import { assert, assertEq, test } from "../harness";
import { ClipmapLayout } from "../../src/gi/VoxelClipmap";
import { nextJob } from "../../src/gi/SceneVoxelClipmap";
import type { Vec3 } from "../../src/gi/VoxelVolume";

const add = (a: Vec3, b: Vec3): Vec3 => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];

test("levels double and centre on the eye", () => {
    const layout = new ClipmapLayout(4, [64, 30, 64], 0.5);
    assertEq(layout.dims, [64, 32, 64]);
    assertEq(layout.levelVoxelSize(3), 4);
    assertEq(layout.levelExtent(1), [64, 32, 64]);
    const eye: Vec3 = [10.3, 1.7, -40.2];
    for (let level = 0; level < 4; level++) {
        const origin = layout.centredOrigin(level, eye, 4);
        assert(origin.every((o) => o % 4 === 0), `snapped: ${origin}`);
        const size = layout.levelVoxelSize(level);
        const centre = [(origin[0] + 32) * size, (origin[1] + 16) * size, (origin[2] + 32) * size];
        // within half a snap of the eye
        const off = Math.max(...centre.map((c, a) => Math.abs(c - eye[a])));
        assert(off <= 2 * size + 1e-4, `level ${level}: centre ${centre} for eye ${eye}`);
    }
});

test("a window follows in whole steps without flicker", () => {
    const layout = new ClipmapLayout(2, [64, 64, 64], 1);
    const origin = layout.centredOrigin(0, [0, 0, 0], 4);
    assertEq(origin, [-32, -32, -32]);
    // less than a step from the centre: stays
    assertEq(layout.follow(0, origin, [3.9, -3.9, 0], 4), origin);
    // a step and a bit: one step
    assertEq(layout.follow(0, origin, [4.1, 0, -9], 4), add(origin, [4, 0, -8]));
    // back to just under a step the other way from the new centre: stays
    const moved = add(origin, [4, 0, 0]);
    assertEq(layout.follow(0, moved, [0.5, 0, 0], 4), moved);
});

test("a footprint picks the level as wide", () => {
    const layout = new ClipmapLayout(5, [64, 64, 64], 0.5);
    assertEq(layout.levelFor(0.1), 0);
    assertEq(layout.levelFor(0.99), 0);
    assertEq(layout.levelFor(1.0), 1);
    assertEq(layout.levelFor(3.9), 2);
    assertEq(layout.levelFor(1e3), 4);
});

test("empty levels fill finest first then slabs follow the eye", () => {
    const layout = new ClipmapLayout(3, [32, 16, 32], 1);
    const origins: (Vec3 | null)[] = [null, null, null];
    const stale = [false, false, false];
    const free = [false, false, false];
    const eye: Vec3 = [0.5, 1, 0.5];
    for (let expect = 0; expect < 3; expect++) {
        const job = nextJob(layout, origins, stale, free, eye, 4)!;
        assertEq(job.region.level, expect);
        assertEq(job.region.size, [32, 16, 32]);
        assertEq(job.region.lo, job.origin);
        origins[expect] = job.origin;
    }
    assert(nextJob(layout, origins, stale, free, eye, 4) === null, "nothing to do while the eye stays");
    // the eye moves 9 m along +x: level 0 (furthest out of its window) moves 8 voxels, its new
    // slab the 8 voxels past its old window
    const moved: Vec3 = add(eye, [9, 0, 0]);
    let job = nextJob(layout, origins, stale, free, moved, 4)!;
    const old = origins[0]!;
    assertEq(job.region.level, 0);
    assertEq(job.origin, add(old, [8, 0, 0]));
    assertEq(job.region.lo, [old[0] + 32, old[1], old[2]]);
    assertEq(job.region.size, [8, 16, 32]);
    // level 0 taken (another slot has it): level 1 (2 m voxels, 4.75 out) moves one step
    job = nextJob(layout, origins, stale, [true, false, false], moved, 4)!;
    assertEq([job.region.level, job.region.size], [1, [4, 16, 32]]);
    // the eye moves back past the start: the slab is on the low side
    const back: Vec3 = add(eye, [-6, 0, 0]);
    job = nextJob(layout, origins, stale, free, back, 4)!;
    assertEq(job.region.level, 0);
    assertEq(job.origin, add(old, [-4, 0, 0]));
    assertEq(job.region.lo, job.origin);
    assertEq(job.region.size, [4, 16, 32]);
    // a stale level is redone whole before any slab
    job = nextJob(layout, origins, [false, false, true], free, moved, 4)!;
    assertEq([job.region.level, job.region.size], [2, [32, 16, 32]]);
    // a jump past a window: the whole window at once
    const far: Vec3 = add(eye, [100, 0, 0]);
    job = nextJob(layout, origins, stale, free, far, 4)!;
    assertEq(job.region.size, [32, 16, 32]);
    assertEq(job.region.lo, job.origin);
});
