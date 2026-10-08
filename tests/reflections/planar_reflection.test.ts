// rust/kansei-core/src/reflections/planar_reflection.rs's CPU tests, ported (`shaders_validate`
// stays in Rust: the TS engine imports the same WGSL; the GPU tests of screen_space.rs too).
import { mat4, vec3, vec4 } from "gl-matrix";
import { assert, assertClose, assertEq, test } from "../harness";
import {
    crop, flipX, obliqueNearPlane, reflectionMatrix, screenRect, screenSpaceSource,
} from "../../src/reflections/PlanarReflection";
import { frustumPlanes } from "../../src/culling/Frustum";
import { Camera } from "../../src/cameras/Camera";

const lookAt = (eye: number[], center: number[]) => mat4.lookAt(mat4.create(), eye as vec3, center as vec3, [0, 1, 0]);
const perspective = (fovy: number, aspect: number, near: number, far: number) => mat4.perspectiveZO(mat4.create(), fovy, aspect, near, far);
const mul = (...ms: mat4[]) => ms.reduce((a, b) => mat4.multiply(mat4.create(), a, b));
const point = (m: mat4, p: number[]) => vec4.transformMat4(vec4.create(), [p[0], p[1], p[2], 1], m);
const inside = (planes: number[][], p: number[]) => planes.every((pl) => pl[0] * p[0] + pl[1] * p[1] + pl[2] * p[2] + pl[3] >= 0);

/**
 * Last frame's GBuffer was drawn jittered (TAA): its pixels are projected with that jittered
 * view, or the whole reflection slides by the jitter from frame to frame (a flicker TAA can't
 * settle).
 */
test("the_screen_is_projected_with_the_view_it_was_drawn_with", () => {
    // the camera's bind group layout names its stages (no WebGPU in Node)
    const g = globalThis as Record<string, unknown>;
    g.GPUShaderStage ??= { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };
    const camera = new Camera(60, 0.1, 100, 1.5);
    camera.updateProjectionMatrix();
    camera.updateViewMatrix();
    assert(screenSpaceSource(camera) === null, "no last frame");
    camera.jitter = [0.002, -0.003];
    const drawn = mul(camera.jitteredProjection(), camera.viewMatrix.internalMat4);
    camera.endFrame();
    camera.jitter = [-0.001, 0.004];
    assertClose(screenSpaceSource(camera)!, drawn, 1e-6);
});

/** The mirrored view cropped to a lake's screen rectangle keeps what the lake reflects, and culls what is reflected beside it. */
test("cropped_mirror_keeps_what_the_surface_reflects", () => {
    const view = lookAt([0, 4, 20], [0, 0, -10]);
    const proj = perspective(0.8, 1.5, 0.1, 500);
    // a lake at y = 0 from x -5..5, z -20..0, seen from above its near shore
    const rect = screenRect(mul(proj, view), [-5, 0, -20], [5, 0, 0], 0);
    assert(rect.kind === "rect", JSON.stringify(rect));
    const [u0, v0, u1, v1] = rect.rect;
    // the mirror as updateCamera builds it, cropped as cullViewProj does
    const mirrored = mul(view, reflectionMatrix([0, 1, 0], 0));
    const refl = mul(flipX(), proj, mirrored);
    const planes = frustumPlanes(mul(crop(-(2 * u1 - 1), -(2 * u0 - 1), 1 - 2 * v1, 1 - 2 * v0), refl));
    // the lake itself, and a tree top on the far shore whose reflection falls on the lake
    assert(inside(planes, [0, 0, -10]));
    assert(inside(planes, [2, 3, -24]));
    // a tree well off to the side, whose reflection falls beside the lake
    assert(!inside(planes, [40, 3, -10]));
});

/** A box's screen rectangle: on screen, beside the view, behind the eye, straddling it; and the crop of that rectangle bounds exactly its part of the view. */
test("surface_rectangle_and_crop", () => {
    // looking down -z, 90 degrees: at z = -10 the view spans x, y in [-10, 10]
    const viewProj = mul(perspective(Math.PI / 2, 1, 0.1, 100), lookAt([0, 0, 0], [0, 0, -1]));
    const rect = (lo: [number, number, number], hi: [number, number, number]) => screenRect(viewProj, lo, hi, 0);
    const r = rect([1, -1, -10], [3, 1, -10]);
    assert(r.kind === "rect", JSON.stringify(r));
    assertClose(r.rect, [0.55, 0.45, 0.65, 0.55], 1e-5);
    assertEq(rect([20, -1, -10], [30, 1, -10]).kind, "offscreen", "beside the view");
    assertEq(rect([-1, -1, 5], [1, 1, 10]).kind, "offscreen", "behind the eye");
    assertEq(rect([-1, -1, -10], [1, 1, 10]).kind, "unbounded", "round the eye");
    // the crop of ndc x in [0.1, 0.3], y in [-0.1, 0.1] keeps the box's part of the view only
    const planes = frustumPlanes(mul(crop(0.1, 0.3, -0.1, 0.1), viewProj));
    assert(inside(planes, [2, 0, -10]));
    assert(inside(planes, [4, 0, -20]), "farther along the same rays");
    assert(!inside(planes, [0, 0, -10]));
    assert(!inside(planes, [4, 0, -10]));
    assert(!inside(planes, [2, 2, -10]));
});

test("reflection_matrix_mirrors_across_the_plane", () => {
    // the plane y = 3
    const m = reflectionMatrix([0, 1, 0], -3);
    assertClose(vec3.transformMat4(vec3.create(), [1, 5, 2], m), [1, 1, 2], 1e-6);
    assertClose(vec3.transformMat4(vec3.create(), [4, 3, -1], m), [4, 3, -1], 1e-6);
    assertClose(mat4.determinant(m), -1, 1e-6);
    // a tilted plane: reflecting twice is the identity
    const n = vec3.normalize(vec3.create(), [0.3, 0.9, -0.2]);
    const r = reflectionMatrix(n, 1.7);
    assertClose(mul(r, r), mat4.create(), 1e-5);
});

/**
 * With the mirrored camera below the plane y = 0 looking up and ahead, the oblique projection
 * puts the plane at depth 0, keeps what is above it in [0, 1], clips what is below, and leaves
 * x, y and w (so the image) unchanged.
 */
test("oblique_projection_clips_at_the_plane", () => {
    const view = lookAt([0, 2, 5], [0, 0.5, -10]);
    const mirrored = mul(view, reflectionMatrix([0, 1, 0], 0));
    const proj = perspective(1, 16 / 9, 0.1, 500);
    const inverseTranspose = mat4.transpose(mat4.create(), mat4.invert(mat4.create(), mirrored));
    const planeView = vec4.transformMat4(vec4.create(), [0, 1, 0, 0], inverseTranspose);
    const oblique = obliqueNearPlane(proj, planeView);

    const ndc = (p: number[]) => {
        const c = point(mul(oblique, mirrored), p);
        return [c[0] / c[3], c[1] / c[3], c[2] / c[3]];
    };
    const onPlane = ndc([0.5, 0, -8]);
    assert(Math.abs(onPlane[2]) < 1e-4, `${onPlane}`);
    const above = ndc([0.5, 2, -20]);
    assert(above[2] > 0 && above[2] < 1, `${above}`);
    const below = ndc([0.5, -1, -8]);
    assert(below[2] < 0, `${below}`);
    // same x/y as the unmodified projection
    const plain = point(mul(proj, mirrored), [0.5, 2, -20]);
    assert(Math.hypot(plain[0] / plain[3] - above[0], plain[1] / plain[3] - above[1]) < 1e-4);
});

/** A point on the plane lands at the same screen position in the main view and, flipped left-right, in the mirrored view: the lookup the WGSL helper does. */
test("plane_points_project_to_mirrored_screen_positions", () => {
    const view = lookAt([3, 4, 10], [-2, 0, -10]);
    const proj = perspective(0.8, 1.5, 0.1, 500);
    const mirrored = mul(view, reflectionMatrix([0, 1, 0], 0));
    const p = [-1, 0, -4];
    const main = point(mul(proj, view), p);
    const refl = point(mul(flipX(), proj, mirrored), p);
    assert(Math.abs(main[0] / main[3] + refl[0] / refl[3]) < 1e-5);
    assert(Math.abs(main[1] / main[3] - refl[1] / refl[3]) < 1e-5);
});
