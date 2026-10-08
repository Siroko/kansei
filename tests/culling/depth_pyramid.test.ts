// rust/kansei-core/src/culling/depth_pyramid.rs tests, ported (the shader's naga validation stays
// Rust's: the TS pyramid builds the same shader).
import { assertEq, test } from "../harness";
import { mipSizes } from "../../src/culling/DepthPyramid";

test("mips_halve_exactly_down_to_one_texel", () => {
    assertEq(mipSizes(8, 4), [[4, 2], [2, 1], [1, 1]]);
    assertEq(mipSizes(5, 3), [[4, 2], [2, 1], [1, 1]]);
    assertEq(mipSizes(1, 1), [[1, 1]]);
    assertEq(mipSizes(1920, 1080)[0], [1024, 1024]);
    assertEq(mipSizes(1440, 810)[0], [1024, 512]);
    assertEq(mipSizes(1920, 1080).length, 11);
    // the top mip covers the whole buffer: (size - 1) >> (levels) == 0
    for (const [w, h] of [[1920, 1080], [1287, 723], [37, 23], [2, 1], [3, 1]]) {
        const levels = mipSizes(w, h).length;
        assertEq([(w - 1) >> levels, (h - 1) >> levels], [0, 0], `${w} x ${h}`);
    }
});
