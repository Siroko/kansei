// rust/kansei-core/src/rt/mesh.rs tests, ported (those of its packing).
import { assert, assertEq, test } from "../harness";
import { RtMesh, halfBits, packOctahedral } from "../../src/rt/RtMesh";
import { rtGlassSurface, rtSurfaceWord } from "../../src/rt/RtGrid";

test("normals pack as kansei_rt_unpack_normal reads them", () => {
    // a CPU twin of rt_types.wgsl's kansei_rt_unpack_normal
    const unpack = (w: number): number[] => {
        const snorm = (h: number) => Math.max(((h & 0xffff) << 16 >> 16) / 32767, -1);
        const [x, y] = [snorm(w), snorm(w >>> 16)];
        let v = [x, y, 1 - Math.abs(x) - Math.abs(y)];
        if (v[2] < 0) v = [(1 - Math.abs(v[1])) * (v[0] >= 0 ? 1 : -1), (1 - Math.abs(v[0])) * (v[1] >= 0 ? 1 : -1), v[2]];
        const l = Math.hypot(v[0], v[1], v[2]);
        return v.map((c) => c / l);
    };
    for (const m of [[0, 1, 0], [0, 0, -1], [0.3, -0.5, 0.81], [-0.7, 0.1, -0.7], [1, 1, 1]]) {
        const l = Math.hypot(m[0], m[1], m[2]);
        const n = m.map((c) => c / l);
        const back = unpack(packOctahedral(n[0], n[1], n[2]));
        const dot = n[0] * back[0] + n[1] * back[1] + n[2] * back[2];
        assert(dot > 0.9999, `${n} came back ${back}`);
    }
});

test("half bits round like pack2x16float", () => {
    assertEq(halfBits(1.0), 0x3c00);
    assertEq(halfBits(0.5), 0x3800);
    assertEq(halfBits(2.0), 0x4000);
    assertEq(halfBits(-2.0), 0xc000);
    assertEq(halfBits(0.0), 0);
    assertEq(halfBits(65504.0), 0x7bff);
    assertEq(halfBits(1e6), 0x7c00);
    // 1 + 2^-11 is halfway between 1 and the next half: to even (1)
    assertEq(halfBits(1.0 + 1.0 / 2048.0), 0x3c00);
    assertEq(halfBits(1.0 + 3.0 / 2048.0), 0x3c02);
    // the smallest subnormal half
    assertEq(halfBits(2 ** -24), 1);
});

test("a mesh's vertices are five words: position, uv, normal", () => {
    const mesh = new RtMesh(new Float32Array([1, 2, 3, 4, 5, 6]), new Float32Array([0.5, 1, 0, 0]), new Float32Array([0, 0, -1, 0, 1, 0]), new Uint32Array([0, 1, 1]), [1, 2, 3], [4, 5, 6]);
    const words = mesh.gpuWords();
    assertEq(Array.from(words.slice(0, 4)), [4, 14, 2, 1]);
    assertEq(Array.from(new Float32Array(words.buffer, 4 * 9, 3)), [4, 5, 6]);
    assertEq(words[4 + 3], (halfBits(0.5) | (halfBits(1) << 16)) >>> 0);
    assertEq(words[4 + 4], packOctahedral(0, 0, -1));
    assertEq(words[4 + 9], packOctahedral(0, 1, 0));
    assertEq(Array.from(words.slice(14)), [0, 1, 1]);
});

test("a smooth surface's word carries KANSEI_RT_SMOOTH", () => {
    assertEq(rtSurfaceWord({ albedo: [1, 1, 1] }), 0);
    assertEq(rtSurfaceWord({ albedo: [1, 1, 1], smoothNormals: true }), 4);
    assertEq(rtSurfaceWord({ albedo: [1, 1, 1], alphaLayer: 2, smoothNormals: true }), 1 | (2 << 8) | 4);
    // glass (KANSEI_RT_GLASS) is smooth too (Rust: rt::tests::surface_words_carry_their_flags)
    assertEq(rtSurfaceWord(rtGlassSurface([0.9, 0.9, 0.9])), 4 | 8);
});
