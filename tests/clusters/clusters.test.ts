// rust/kansei-core/src/clusters/gpu_tests.rs, ported where they need no GPU: the vertex stage's
// rewrite and errors, the sizing of draw lists, and the cull's parameters and views.
import { mat4 } from "gl-matrix";
import { assert, assertEq, test } from "../harness";
import { CLUSTER_VERTEX_ENTRY, clusterVertexStage } from "../../src/clusters/vertexStage";
import {
    CLUSTER_CULL_BYTES, INITIAL_DRAWN, MIN_DRAWN, SHRINK_AFTER, Sizer, clusterCullParams, clusterViewOf, packClusterCull, projectionNear, trianglesNeeded,
} from "../../src/clusters/ClusterLod";

const validStage = (code: string, instances: Parameters<typeof clusterVertexStage>[1] = null) => {
    const out = clusterVertexStage(code, instances);
    assert(out.includes(`fn ${CLUSTER_VERTEX_ENTRY}(`), "no generated entry");
    assert(!/@vertex\s*fn vertex_main/.test(out), "vertex_main is still an entry point");
    assert(out.includes("kansei_cluster_mesh"), "no cluster mesh binding");
    return out;
};

test("located and struct forms get a cluster stage", () => {
    const located = `
struct VOut { @builtin(position) clip: vec4<f32>, @location(0) n: vec3<f32> };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex
fn vertex_main(
    // per vertex
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>, /* unused */
) -> VOut {
    var out: VOut;
    out.clip = view_matrix * position;
    out.n = normal + vec3<f32>(uv, 0.0);
    return out;
}
@fragment fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.n, 1.0); }
`;
    const out = validStage(located);
    assert(out.includes("fn vertex_main(position: vec4<f32>, normal: vec3<f32>, uv: vec2<f32>) -> VOut"), "plain parameters");
    assert(out.includes("return vertex_main(kansei_cluster_attribute(kansei_vertex, 0u, 4u), kansei_cluster_attribute(kansei_vertex, 4u, 3u).xyz, kansei_cluster_attribute(kansei_vertex, 7u, 2u).xy);"), "arguments");
    assert(out.includes("const KANSEI_RECORD_WORDS: u32 = 0u;"), "no records");

    const builtinOut = `
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex fn vertex_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> { return view_matrix * position; }
@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }
`;
    const plain = validStage(builtinOut);
    assert(plain.includes("fn vertex_main(position: vec4<f32>) -> vec4<f32> "), "the plain function returns a plain type");
    assert(plain.includes("-> @builtin(position) vec4<f32> {\n    let kansei_draw"), "the entry returns the builtin");

    const instanced = `
struct VIn {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec3<f32>,
    @location(3) place: vec4<f32>,
    @location(4) yaw: f32,
    @location(5) ids: vec2<u32>,
    @location(6) offset: vec3i,
};
struct VOut { @builtin(position) @invariant clip: vec4<f32>, @location(0) @interpolate(flat) id: u32 };
@vertex
fn vertex_main(v: VIn) -> VOut {
    var out: VOut;
    out.clip = vec4<f32>(v.position.xyz * v.place.w + v.place.xyz + vec3<f32>(v.offset) + v.normal * v.yaw, 1.0);
    out.id = v.ids.x + v.ids.y;
    return out;
}
`;
    const records = { arrayStride: 48, attributes: [
        { shaderLocation: 3, offset: 0, format: "float32x4" as GPUVertexFormat },
        { shaderLocation: 4, offset: 16, format: "float32" as GPUVertexFormat },
        { shaderLocation: 5, offset: 20, format: "uint32x2" as GPUVertexFormat },
        { shaderLocation: 6, offset: 28, format: "sint32x3" as GPUVertexFormat },
    ] };
    const s = validStage(instanced, records);
    for (const fill of [
        "kansei_input.place = kansei_record_f32(kansei_record, 0u, 4u);",
        "kansei_input.yaw = kansei_record_f32(kansei_record, 4u, 1u).x;",
        "kansei_input.ids = kansei_record_u32(kansei_record, 5u, 2u).xy;",
        "kansei_input.offset = kansei_record_i32(kansei_record, 7u, 3u).xyz;",
        "const KANSEI_RECORD_WORDS: u32 = 12u;",
        "fn vertex_main(v: VIn) -> VOut ",
    ]) assert(s.includes(fill), `missing \`${fill}\``);
});

test("inputs the cluster path cannot feed are errors", () => {
    const error = (code: string, instances: Parameters<typeof clusterVertexStage>[1] = null): string => {
        try {
            clusterVertexStage(code, instances);
        } catch (e) {
            return (e as Error).message;
        }
        throw new Error(`no error for ${code}`);
    };
    assert(error("@vertex fn vertex_main(@location(0) p: vec4<f32>, @builtin(instance_index) i: u32) -> @builtin(position) vec4<f32> { return p; }").includes("instance_index"));
    assert(error("struct VIn { @location(0) p: vec4<f32>, @builtin(vertex_index) i: u32 };\n@vertex fn vertex_main(v: VIn) -> @builtin(position) vec4<f32> { return v.p; }").includes("vertex_index"));
    assert(error("@vertex fn vertex_main(@location(3) q: vec4<f32>) -> @builtin(position) vec4<f32> { return q; }").includes("location(3)"));
    const floats = { arrayStride: 16, attributes: [{ shaderLocation: 3, offset: 0, format: "float32x4" as GPUVertexFormat }] };
    assert(error("@vertex fn vertex_main(@location(3) q: vec4<u32>) -> @builtin(position) vec4<f32> { return vec4<f32>(q); }", floats).includes("location(3)"));
    assert(error("@fragment fn fragment_main() -> @location(0) vec4<f32> { return vec4<f32>(1.0); }").includes("vertex_main"));
});

test("draw lists start at their initial size, grow at once and shrink after a while", () => {
    const sizer = new Sizer();
    const max = 1 << 22;
    let length = sizer.sized(0, INITIAL_DRAWN, MIN_DRAWN, max, null);
    assertEq(length, INITIAL_DRAWN);
    // a need past the length: half again, rounded up to an eighth of its power of two
    length = sizer.sized(length, INITIAL_DRAWN, MIN_DRAWN, max, 20000);
    assertEq(length, 32768);
    // never past the maximum, but a smaller maximum alone doesn't shrink it
    assertEq(sizer.sized(INITIAL_DRAWN, INITIAL_DRAWN, MIN_DRAWN, 25000, 40000), 25000);
    assertEq(sizer.sized(length, INITIAL_DRAWN, MIN_DRAWN, 25000, 40000), length);
    // a small need keeps the length until it has lasted SHRINK_AFTER readings
    for (let k = 1; k < SHRINK_AFTER; k++) assertEq(sizer.sized(length, INITIAL_DRAWN, MIN_DRAWN, max, 10), length);
    assertEq(sizer.sized(length, INITIAL_DRAWN, MIN_DRAWN, max, 10), MIN_DRAWN);
    // no reading: as it is
    assertEq(sizer.sized(length, INITIAL_DRAWN, MIN_DRAWN, max, null), length);
});

test("a full draw list scales the triangles it saw", () => {
    assertEq(trianglesNeeded(100, 200, 5000), 5000);
    assertEq(trianglesNeeded(400, 200, 5000), 10000);
    assertEq(trianglesNeeded(400, 0, 5000), 5000);
});

test("cull parameters pack as cluster_cull.wgsl's ClusterCull", () => {
    const world = mat4.fromTranslation(mat4.create(), [1, 2, 3]);
    const params = clusterCullParams(world, { kind: "placement", position: 0, scale: 12, yaw: 16, yawScale: -1 }, 32,
        { kind: "all", records: {} as GPUBuffer, count: 576 }, 1000, 372, true, 1);
    params.view = 3;
    params.triangleCapacity = 4096;
    const bytes = packClusterCull(params);
    assertEq(bytes.byteLength, CLUSTER_CULL_BYTES);
    const u = new Uint32Array(bytes), f = new Float32Array(bytes);
    assertEq(Array.from(f.subarray(12, 15)), [1, 2, 3]);
    // kind, position, scale, yaw, rotation, stride, first record, instances, count word, capacity, vertices, flags
    assertEq(Array.from(u.subarray(16, 28)), [1, 0, 3, 4, 0xffffffff, 8, 0, 576, 0xffffffff, 1000, 372, 1]);
    assertEq([f[28], f[29], u[30], u[31]], [-1, 1, 3, 4096]);
    // a stretch turns the cones off; no instances ignore the transform
    assertEq(clusterCullParams(world, null, 0, { kind: "none" }, 1, 3, true, 1.2).flags, 0);
    assertEq(clusterCullParams(world, { kind: "matrix", offset: 16 }, 64, { kind: "none" }, 1, 3, true, 1).kind, 0);
});

test("cluster views read a projection's kind and scale", () => {
    const perspective = mat4.perspectiveZO(mat4.create(), Math.PI / 4, 16 / 9, 0.25, 100);
    const view = mat4.lookAt(mat4.create(), [0, 5, 10], [0, 0, 0], [0, 1, 0]);
    const inverse = mat4.invert(mat4.create(), view);
    const viewProj = mat4.multiply(mat4.create(), perspective, view);
    const v = clusterViewOf(viewProj, perspective, inverse, 720, projectionNear(perspective), 1);
    assert(!v.orthographic);
    assert(Math.abs(v.pixelsPerUnit - 720 / (2 * Math.tan(Math.PI / 8))) < 1e-3, `${v.pixelsPerUnit}`);
    assert(Math.abs(v.near - 0.25) < 1e-6, `${v.near}`);
    assert(Math.abs(v.eye[1] - 5) < 1e-5 && Math.abs(v.eye[2] - 10) < 1e-5, `${Array.from(v.eye)}`);
    const ortho = mat4.orthoZO(mat4.create(), -10, 10, -10, 10, 1, 50);
    const o = clusterViewOf(ortho, ortho, mat4.create(), 1024, projectionNear(ortho), 1);
    assert(o.orthographic);
    assert(Math.abs(o.pixelsPerUnit - 1024 / 20) < 1e-4, `${o.pixelsPerUnit}`);
});
