// The cluster debug view's generated stage (TS only, no Rust counterpart): the cluster vertex
// stage as a plain function the debug entry point calls, and its output's clip position.
import { assert, assertEq, test } from "../harness";
import { CLUSTER_VERTEX_ENTRY, CLUSTER_VERTEX_FN, clusterVertexFunction } from "../../src/clusters/vertexStage";
import { CLUSTER_DEBUG_FRAGMENT_ENTRY, CLUSTER_DEBUG_VERTEX_ENTRY, clusterDebugWgsl } from "../../src/clusters/ClusterDebug";

const STRUCT_OUT = `
struct VOut { @location(0) n: vec3<f32>, @builtin(position) @invariant clip_pos: vec4<f32> };
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@vertex
fn vertex_main(@location(0) position: vec4<f32>, @location(1) normal: vec3<f32>) -> VOut {
    var out: VOut;
    out.clip_pos = view_matrix * position;
    out.n = normal;
    return out;
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4<f32> { return vec4<f32>(in.n, 1.0); }
`;

test("the debug stage is a plain function, and finds the clip position", () => {
    const { code, position } = clusterVertexFunction(STRUCT_OUT, null);
    assertEq(position, "clip_pos");
    assert(code.includes(`fn ${CLUSTER_VERTEX_FN}(kansei_vertex_index: u32, kansei_instance_index: u32) -> VOut {`), "no plain function");
    assert(!code.includes(CLUSTER_VERTEX_ENTRY), "the entry point is still generated");
    assert(!/@vertex\s*fn vertex_main/.test(code), "vertex_main is still an entry point");
});

test("a bare position output has no member", () => {
    const bare = `
@vertex
fn vertex_main(@location(0) position: vec4<f32>) -> @builtin(position) vec4<f32> { return position; }
`;
    const { code, position } = clusterVertexFunction(bare, null);
    assertEq(position, null);
    assert(code.includes(`fn ${CLUSTER_VERTEX_FN}(kansei_vertex_index: u32, kansei_instance_index: u32) -> vec4<f32> {`), "no plain function");
});

test("the debug stages write each target", () => {
    for (const targets of [1, 4]) {
        const wgsl = clusterDebugWgsl("clip", targets);
        assert(wgsl.includes(`fn ${CLUSTER_DEBUG_VERTEX_ENTRY}(`) && wgsl.includes(`fn ${CLUSTER_DEBUG_FRAGMENT_ENTRY}(`), "no entry points");
        assert(wgsl.includes("out.clip = shaded.clip;"), "no clip position");
        const outputs = wgsl.slice(wgsl.indexOf("struct KanseiClusterDebugTargets"), wgsl.indexOf("@fragment"));
        assertEq(outputs.match(/@location\(\d+\)/g), Array.from({ length: targets }, (_, i) => `@location(${i})`));
    }
});
