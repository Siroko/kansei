// The ray tracing grid's uniform (rt/grid.rs `RtGridGpu`) and its world triangles, shared by the
// gather, the build and the traversal.
//
// A world triangle is four vec4 (64 bytes): v0 and the surface word (flags in the low byte, the
// alpha layer in the next; the build sets KANSEI_RT_BIG), e1 = v1 - v0 and the albedo
// (unorm4x8, its alpha 1 - the surface's F0), e2 = v2 - v0 and where it came from (source << 20 | record), then the three
// vertices' uvs (pack2x16float) and the light it emits (RGBE, as rt_emission.wgsl packs it; 0:
// none).
//
// The cell words: each cell's end in the references (its start the previous cell's end), the
// 4^3 macro cells (non-zero when any of their cells lists a triangle), the big triangles (a
// count, then their indices) and the references (triangle indices).
//
// The grid has two layers in the same buffers, each with its own box, cells and triangles: the
// static one, rebuilt only when what it holds changes, and the moving one, rebuilt every frame
// from what moves (`moving`, absent while its cellCount is 0). A trace tests both and keeps the
// nearer hit. `KanseiRtGrid`'s own fields are the static layer's, laid out as a `KanseiRtLayer`.

struct KanseiRtLayer {
    origin           : vec3f,
    cell             : f32,
    dims             : vec3u,
    flags            : u32,
    macroDims        : vec3u,
    epsilon          : f32,
    cellCount        : u32,
    macroBase        : u32,
    bigBase          : u32,
    refsBase         : u32,
    refCapacity      : u32,
    triangleCapacity : u32,
    bigCapacity      : u32,
    bigCells         : u32,
    // where its cell words and its triangles start
    cellsBase        : u32,
    triangleBase     : u32,
    pad0             : u32,
    pad1             : u32,
}

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
    // where the layer's cell words and its triangles start (0 for the static layer; the build's
    // uniform of the moving layer holds the moving layer here)
    cellsBase        : u32,
    triangleBase     : u32,
    pad0             : u32,
    pad1             : u32,
    moving           : KanseiRtLayer,
}

// The static layer, as a `KanseiRtLayer`.
fn kansei_rt_static_layer(g: KanseiRtGrid) -> KanseiRtLayer {
    return KanseiRtLayer(g.origin, g.cell, g.dims, g.flags, g.macroDims, g.epsilon, g.cellCount, g.macroBase, g.bigBase, g.refsBase, g.refCapacity, g.triangleCapacity, g.bigCapacity, g.bigCells, g.cellsBase, g.triangleBase, 0u, 0u);
}

// surface flags
const KANSEI_RT_ALPHA : u32 = 1u;
// its last four words hold its vertices' world normals, not their uvs (RtSurface::smooth_normals)
const KANSEI_RT_SMOOTH : u32 = 4u;
// glass (RtSurface::glass): rays pass through it unless they ask for it (KANSEI_RT_GLASS_HITS)
const KANSEI_RT_GLASS : u32 = 8u;
// its albedo is also a layer (bits 16-23) of the texture array kansei_rt_hit_albedo samples by uv
// (RtSurface::with_albedo_layer), times the albedo word
const KANSEI_RT_ALBEDO_TEX : u32 = 16u;
// its material lights it on screen itself, so the GBuffer's background image holds its light
// (RtSurface::with_self_lit)
const KANSEI_RT_SELF_LIT : u32 = 32u;
// a coarse copy of a surface the camera draws finely (a fluid's): the glass pass's paths leaving
// that surface pass through it near where they start (RtSurface::with_approximate)
const KANSEI_RT_APPROXIMATE : u32 = 64u;

// A unit normal, octahedral, two snorm16 (RtMesh's packing).
fn kansei_rt_pack_normal(n: vec3f) -> u32 {
    var v = n / max(abs(n.x) + abs(n.y) + abs(n.z), 1e-20);
    var o = v.xy;
    if (v.z < 0.0) {
        o = (1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v.xy >= vec2f(0.0));
    }
    return pack2x16snorm(o);
}

fn kansei_rt_unpack_normal(w: u32) -> vec3f {
    let o = unpack2x16snorm(w);
    var v = vec3f(o, 1.0 - abs(o.x) - abs(o.y));
    if (v.z < 0.0) {
        v = vec3f((1.0 - abs(v.yx)) * select(vec2f(-1.0), vec2f(1.0), v.xy >= vec2f(0.0)), v.z);
    }
    return normalize(v);
}
// set by the build: the triangle is in the big list (not in the cells)
const KANSEI_RT_BIG : u32 = 0x80000000u;
// grid flags
const KANSEI_RT_MACRO_SKIP : u32 = 1u;
// the layer's build keeps only the triangles that also meet the second layer's box (the moving
// layer's build, whose uniform holds the static layer second: the triangles the static layer
// would keep)
const KANSEI_RT_CLIP : u32 = 2u;
const KANSEI_RT_MACRO_SHIFT : u32 = 2u;
