// rust/kansei-core/src/simulations/fluid/fill.rs tests (fill_box, lattice_density), ported.
import { assert, assertEq, test } from "../harness";
import { fillBox, latticeDensity } from "../../src/simulations/fluid/FluidSimulationParams";

test("a box fill holds the count inside the box", () => {
    const lo: [number, number, number] = [-1, 0, -2];
    const hi: [number, number, number] = [1, 3, 2];
    for (const count of [1, 1000, 4097]) {
        const p = fillBox(count, lo, hi, 0.3);
        assertEq(p.length, count * 4);
        for (let k = 0; k < count; k++) {
            const q = p.subarray(k * 4, k * 4 + 4);
            assert([0, 1, 2].every((i) => q[i] > lo[i] && q[i] < hi[i] + 1), `${[...q]} outside`);
            assertEq(q[3], 1);
        }
    }
    // the first row is at the box's +z end
    assert(fillBox(1000, lo, hi, 0)[2] > 1.5);
});

test("lattice density tends to one over the cell volume", () => {
    const fine = latticeDensity(0.1, 1);
    assert(Math.abs(fine * 0.001 - 1) < 0.01, `${fine}`);
    // coarse: the particle's own share dominates
    const coarse = latticeDensity(2, 1);
    assert(Math.abs(coarse - 315 / (64 * Math.PI)) < 1e-3, `${coarse}`);
});
