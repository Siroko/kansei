// rust/kansei-core/src/postprocessing/effects/ssgi.rs's CPU test, ported (its others run the
// effect on a GPU).
import { assert, assertEq, test } from "../harness";
import { GiQuality, SSGI_PARAMS_BYTES } from "../../src/postprocessing/effects/ScreenSpaceGIEffect";
import { SSGI_COMPOSITE_WGSL, SSGI_TEMPORAL_WGSL, SSGI_TRACE_WGSL } from "../../src/materials/shaders/SharedWGSL";

/** Size and alignment of the WGSL types `SsgiParams` uses. */
const LAYOUT: Record<string, [number, number]> = { mat4x4f: [64, 16], vec2f: [8, 8], f32: [4, 4], u32: [4, 4] };

/** The byte size of WGSL struct `name` in `code` (uniform layout, for the types in `LAYOUT`). */
function structSize(code: string, name: string): number {
    const body = code.match(new RegExp(`struct ${name}\\s*\\{([^}]*)\\}`))?.[1];
    assert(body !== undefined, `no struct ${name}`);
    let offset = 0, align = 1;
    for (const [, type] of body.replace(/\/\/[^\n]*/g, "").matchAll(/:\s*([\w<>]+)\s*,/g)) {
        const layout = LAYOUT[type];
        assert(layout !== undefined, `unknown type ${type}`);
        offset = Math.ceil(offset / layout[1]) * layout[1] + layout[0];
        align = Math.max(align, layout[1]);
    }
    return Math.ceil(offset / align) * align;
}

test("the params layout matches", () => {
    // the bytes `ScreenSpaceGIEffect` writes (Rust `SsgiParamsGpu`)
    for (const code of [SSGI_TRACE_WGSL, SSGI_TEMPORAL_WGSL, SSGI_COMPOSITE_WGSL]) {
        assertEq(structSize(code, "SsgiParams"), SSGI_PARAMS_BYTES);
    }
});

test("the quality presets are Rust's", () => {
    assertEq(GiQuality.settings("low"), [0.25, 2, 6]);
    assertEq(GiQuality.settings("medium"), [0.5, 2, 8]);
    assertEq(GiQuality.settings("high"), [0.5, 4, 12]);
    assertEq(GiQuality.settings("ultra"), [1.0, 4, 16]);
    assertEq(GiQuality.fromName("ssgi"), null);
});
