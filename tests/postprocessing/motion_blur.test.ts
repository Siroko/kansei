// rust/kansei-core/src/postprocessing/effects/motion_blur.rs's tests, ported.
import { assert, assertClose, assertEq, test } from "../harness";
import { MotionBlurEffect } from "../../src/postprocessing/effects/MotionBlurEffect";
import { MOTION_BLUR_GATHER_WGSL, MOTION_BLUR_NEIGHBOURS_WGSL, MOTION_BLUR_PREPARE_WGSL } from "../../src/materials/shaders/SharedWGSL";

/** Size and alignment of the WGSL types `MotionBlurParams` uses. */
const LAYOUT: Record<string, [number, number]> = { mat4x4f: [64, 16], vec2f: [8, 8], f32: [4, 4], u32: [4, 4] };

/** The byte size of WGSL struct `name` in `code` (uniform layout, for the types in `LAYOUT`). */
function structSize(code: string, name: string): number {
    const body = code.match(new RegExp(`struct ${name}\\s*\\{([^}]*)\\}`))?.[1];
    assert(body !== undefined, `no struct ${name}`);
    let offset = 0, align = 1;
    for (const [, type] of body.matchAll(/:\s*([\w<>]+)\s*,/g)) {
        const layout = LAYOUT[type];
        assert(layout !== undefined, `unknown type ${type}`);
        offset = Math.ceil(offset / layout[1]) * layout[1] + layout[0];
        align = Math.max(align, layout[1]);
    }
    return Math.ceil(offset / align) * align;
}

test("the params layout matches", () => {
    // the bytes `MotionBlurEffect` writes (Rust `MotionBlurParamsGpu`)
    for (const code of [MOTION_BLUR_PREPARE_WGSL, MOTION_BLUR_NEIGHBOURS_WGSL, MOTION_BLUR_GATHER_WGSL]) {
        assertEq(structSize(code, "MotionBlurParams"), 224);
    }
});

test("the blur scales like Unreal's", () => {
    // amount 0.5: a 180-degree shutter, the streak half the frame's motion, its radius a quarter
    const fx = new MotionBlurEffect({ amount: 0.5 });
    assertEq(fx.blurScale(), 0.25);
    // at 60 fps with a 30 fps target, each frame's (half as long) motion counts twice
    fx.options.targetFps = 30;
    fx.setFrameTime(1 / 60);
    assertClose(fx.blurScale(), 0.5, 1e-5);
    fx.setFrameTime(1 / 30);
    assertClose(fx.blurScale(), 0.25, 1e-5);
    // the cap is the same fraction of any picture
    fx.options.max = 0.04;
    assertClose(fx.maxRadiusPx(1920), 76.8, 1e-3);
    assertClose(fx.maxRadiusPx(1280), 51.2, 1e-3);
});
