// rust/kansei-core/src/materials/standard.rs's CPU tests of the traced surfaces and the specular factor, ported (the WGSL
// validation stays in Rust: the TS engine imports the same WGSL).
import { assertEq, test } from "../harness";
import { glassOptions, matteOptions, mirrorOptions, standardLitUniform, StandardLitOptions } from "../../src/materials/StandardLit";

test("traced surfaces tell the shader their mode and ior", () => {
    const params = (o: StandardLitOptions) => Array.from(standardLitUniform(o).slice(16, 20));
    assertEq(params({ roughness: 0.6 }), [Math.fround(0.6), 0, 0, 1]);
    assertEq(params(mirrorOptions([0.95, 0.95, 0.95], 0.2)), [Math.fround(0.2), 1, 1, 1]);
    assertEq(params(glassOptions([0.9, 0.9, 0.9], 1.5, 0.1)), [Math.fround(0.1), 0, 2, 1.5]);
    // an index under air's is air's
    assertEq(params(glassOptions([1, 1, 1], 0.5, 0))[3], 1);
});

test("the specular factor rides in the base colour's alpha", () => {
    // the default writes the 1 the uniform held there before the factor
    assertEq(standardLitUniform({})[3], 1);
    assertEq(standardLitUniform(mirrorOptions([0.95, 0.95, 0.95], 0.2))[3], 1);
    assertEq(Array.from(standardLitUniform(matteOptions([0.5, 0.5, 0.5], 0.7)).slice(0, 4)), [0.5, 0.5, 0.5, 0]);
    assertEq(standardLitUniform({ specular: 0.25 })[3], 0.25);
    // clamped to 0..1: over 1 would read as a reflective surface's F0 in the GBuffer
    assertEq(standardLitUniform({ specular: 3 })[3], 1);
    assertEq(standardLitUniform({ specular: -1 })[3], 0);
});
