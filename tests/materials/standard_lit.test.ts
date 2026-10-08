// rust/kansei-core/src/materials/standard.rs's CPU test of the traced surfaces, ported (the WGSL
// validation stays in Rust: the TS engine imports the same WGSL).
import { assertEq, test } from "../harness";
import { glassOptions, mirrorOptions, standardLitUniform, StandardLitOptions } from "../../src/materials/StandardLit";

test("traced surfaces tell the shader their mode and ior", () => {
    const params = (o: StandardLitOptions) => Array.from(standardLitUniform(o).slice(16, 20));
    assertEq(params({ roughness: 0.6 }), [Math.fround(0.6), 0, 0, 1]);
    assertEq(params(mirrorOptions([0.95, 0.95, 0.95], 0.2)), [Math.fround(0.2), 1, 1, 1]);
    assertEq(params(glassOptions([0.9, 0.9, 0.9], 1.5, 0.1)), [Math.fround(0.1), 0, 2, 1.5]);
    // an index under air's is air's
    assertEq(params(glassOptions([1, 1, 1], 0.5, 0))[3], 1);
});
