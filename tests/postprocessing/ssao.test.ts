// rust/kansei-core/src/postprocessing/effects/ssao.rs's CPU tests, ported (tests/ssao_gpu.rs runs
// the effect on a GPU).
import { mat4, vec3, vec4 } from "gl-matrix";
import { assert, assertEq, test } from "../harness";
import { SSAO_PARAMS_BYTES, ssaoKernel } from "../../src/postprocessing/effects/SSAOEffect";
import { SSAO_BLUR_WGSL, SSAO_COMPOSITE_WGSL, SSAO_WGSL } from "../../src/materials/shaders/SharedWGSL";

/** Size and alignment of the WGSL types `SSAOParams` uses. */
const LAYOUT: Record<string, [number, number]> = { mat4x4f: [64, 16], vec4f: [16, 16], vec2f: [8, 8], f32: [4, 4], u32: [4, 4] };

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

test("shaders validate and the params match", () => {
    // the bytes `SSAOEffect` writes (Rust `SSAOParams`); naga validates the WGSL on the Rust side
    for (const code of [SSAO_WGSL, SSAO_BLUR_WGSL, SSAO_COMPOSITE_WGSL]) {
        assertEq(structSize(code, "SSAOParams"), SSAO_PARAMS_BYTES);
    }
    assertEq(SSAO_PARAMS_BYTES, 336);
});

test("kernel is a hemisphere", () => {
    const k = ssaoKernel();
    for (let i = 0; i < 32; i++) {
        const [x, y, z] = k.subarray(i * 4, i * 4 + 3);
        const len = Math.hypot(x, y, z);
        const t = i / 32;
        assert(z >= 0.15 * len - 1e-6, `${i}: ${x} ${y} ${z}`);
        assert(len <= 0.1 + t * t * 0.9 + 1e-5, `${i}: ${len}`);
    }
});

test("the rebuilt normal faces the camera", () => {
    const [w, h] = [64, 48];
    const proj = mat4.perspectiveZO(mat4.create(), (60 * Math.PI) / 180, w / h, 0.1, 100);
    const invProj = mat4.invert(mat4.create(), proj);
    const unproject = (px: [number, number], depth: number): vec3 => {
        const p = vec4.transformMat4(vec4.create(), [((px[0] + 0.5) / w) * 2 - 1, 1 - ((px[1] + 0.5) / h) * 2, depth, 1], invProj);
        return vec3.fromValues(p[0] / p[3], p[1] / p[3], p[2] / p[3]);
    };
    // the depth of a view-space plane (normal n through point q) under each pixel
    const depthOf = (n: vec3, q: vec3) => (px: [number, number]) => {
        const dir = vec3.normalize(vec3.create(), unproject(px, 1));
        const t = vec3.dot(n, q) / vec3.dot(n, dir);
        const clip = vec4.transformMat4(vec4.create(), [dir[0] * t, dir[1] * t, dir[2] * t, 1], proj);
        return clip[2] / clip[3];
    };
    const floorN = vec3.normalize(vec3.create(), [0, 1, 0.3]);
    const planes: [vec3, vec3][] = [[vec3.fromValues(0, 0, 1), vec3.fromValues(0, 0, -5)], [floorN, vec3.fromValues(0, -1, -5)]];
    for (const [n, q] of planes) {
        const depth = depthOf(n, q);
        const at = (px: [number, number]) => unproject(px, depth(px));
        const sub = (a: vec3, b: vec3) => vec3.subtract(vec3.create(), a, b);
        const cross = (a: vec3, b: vec3) => vec3.normalize(vec3.create(), vec3.cross(vec3.create(), a, b));
        const p = at([32, 24]);
        const rebuilt = cross(sub(at([33, 24]), p), sub(at([32, 23]), p));
        assert(vec3.dot(rebuilt, n) > 0.999, `${rebuilt} for ${n}`);
        assert(vec3.dot(rebuilt, vec3.negate(vec3.create(), p)) > 0);
        const old = cross(sub(at([33, 24]), at([31, 24])), sub(at([32, 25]), at([32, 23])));
        assert(vec3.dot(old, n) < -0.999, `the old reconstruction: ${old}`);
    }
});
