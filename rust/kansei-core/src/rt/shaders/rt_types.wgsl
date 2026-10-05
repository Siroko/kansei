// The ray tracing grid's uniform (rt/grid.rs `RtGridGpu`) and its world triangles, shared by the
// gather, the build and the traversal.
//
// A world triangle is four vec4 (64 bytes): v0 and the surface word (flags in the low byte, the
// alpha layer in the next; the build sets KANSEI_RT_BIG), e1 = v1 - v0 and the albedo
// (unorm4x8), e2 = v2 - v0 and where it came from (source << 20 | record), then the three
// vertices' uvs (pack2x16float).
//
// The cell words: each cell's end in the references (its start the previous cell's end), the
// 4^3 macro cells (non-zero when any of their cells lists a triangle), the big triangles (a
// count, then their indices) and the references (triangle indices).

struct KanseiRtGrid {
    origin           : vec3f,
    cell             : f32,
    dims             : vec3u,
    flags            : u32,
    macroDims        : vec3u,
    // world units every cell is widened by, in every test
    epsilon          : f32,
    cellCount        : u32,
    macroBase        : u32,
    bigBase          : u32,
    refsBase         : u32,
    refCapacity      : u32,
    triangleCapacity : u32,
    bigCapacity      : u32,
    // a triangle whose footprint spans more columns than this goes to the big list
    bigCells         : u32,
}

// surface flags
const KANSEI_RT_ALPHA : u32 = 1u;
// set by the build: the triangle is in the big list (not in the cells)
const KANSEI_RT_BIG : u32 = 0x80000000u;
// grid flags
const KANSEI_RT_MACRO_SKIP : u32 = 1u;
const KANSEI_RT_MACRO_SHIFT : u32 = 2u;
