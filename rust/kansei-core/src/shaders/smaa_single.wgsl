// SmaaEffect's single pass (SmaaMode::SinglePass): SMAA 1x for a 16 x 16 tile in one dispatch,
// from workgroup memory. Appended to smaa_common.wgsl, whose accessors it defines; smaa.rs
// replaces KANSEI_SMAA_REACH with SmaaEffect::REACH.
//
// A workgroup of 16 x 16 invocations serves one tile of 16 x 16 output pixels. The blend of the
// tile reads the weights of a 17 x 17 block: the tile and the ring right of it and below it.
//   Phase 1: the luma of the block's 20 x 20 surroundings, then the block's edges. A tile with
//            none comes through as it is (each pixel stores its input), and stops.
//   Phase 2: the apron the block's searches reach, in two arms: the horizontal arm along the
//            rows (runs of top edges), the vertical arm down the columns (runs of left edges),
//            each SMAA_REACH past the block both ways. Their luma, then their edges, then the
//            block's weights (quantized as the three passes' rgba8unorm stores them), then the
//            tile's blend.
// Coordinates below are tile-relative (r = p - smaaOrigin). Edge detection at r reads the luma
// from r - 2 to r + 1 on each axis. A run's search from the block reaches SMAA_REACH along it
// (SMAA_REACH + 1 for its far end's crossing edges), and 2 before and 1 after it across.

@group(0) @binding(4) var smaaOutput : texture_storage_2d<rgba16float, write>;

const SMAA_REACH : i32 = KANSEI_SMAA_REACH;
const SMAA_TILE  : i32 = 16;
const SMAA_BLOCK : i32 = SMAA_TILE + 1;
const SMAA_GROUP : i32 = SMAA_TILE * SMAA_TILE;

// The edges an arm holds: ACROSS a run, -2 .. 17; ALONG it, -REACH .. 16 + REACH + 1.
const SMAA_ACROSS0  : i32 = -2;
const SMAA_ACROSS1  : i32 = SMAA_BLOCK;
const SMAA_ALONG0   : i32 = -SMAA_REACH;
const SMAA_ALONG1   : i32 = SMAA_BLOCK + SMAA_REACH;
const SMAA_ACROSS_N : i32 = SMAA_ACROSS1 - SMAA_ACROSS0 + 1;
const SMAA_ALONG_N  : i32 = SMAA_ALONG1 - SMAA_ALONG0 + 1;
// Their luma: 2 more before, 1 more after, on each axis.
const SMAA_LUMA_ACROSS_N : i32 = SMAA_ACROSS_N + 3;
const SMAA_LUMA_ALONG_N  : i32 = SMAA_ALONG_N + 3;
// Phase 1's block of luma, the edges' block's -2 .. 17 square.
const SMAA_LUMA_BLOCK_N  : i32 = SMAA_BLOCK + 3;

// The horizontal arm keeps its whole extent: luma columns -18 .. 34 by rows -4 .. 18, edge
// columns -16 .. 33 by rows -2 .. 17 (2 bits a pixel: left, top; 16 pixels a word, along a row).
// The vertical arm keeps only its ends, the rows the horizontal arm lacks: luma rows -18 .. -5 and
// 19 .. 34 of columns -4 .. 18, edge rows -16 .. -3 and 18 .. 33 of columns -2 .. 17 (packed
// down a column).
const SMAA_LUMA_ENDS_N : i32 = SMAA_LUMA_ALONG_N - SMAA_LUMA_ACROSS_N;
const SMAA_EDGE_ENDS_N : i32 = SMAA_ALONG_N - SMAA_ACROSS_N;
const SMAA_H_WORDS     : i32 = (SMAA_ALONG_N + 15) / 16;
const SMAA_V_WORDS     : i32 = (SMAA_EDGE_ENDS_N + 15) / 16;

var<workgroup> smaaLumaH        : array<f32, SMAA_LUMA_ALONG_N * SMAA_LUMA_ACROSS_N>;
var<workgroup> smaaLumaV        : array<f32, SMAA_LUMA_ACROSS_N * SMAA_LUMA_ENDS_N>;
var<workgroup> smaaEdgesH       : array<atomic<u32>, SMAA_ACROSS_N * SMAA_H_WORDS>;
var<workgroup> smaaEdgesV       : array<atomic<u32>, SMAA_ACROSS_N * SMAA_V_WORDS>;
var<workgroup> smaaBlockWeights : array<u32, SMAA_BLOCK * SMAA_BLOCK>;
var<workgroup> smaaTileHasEdges : u32;
var<private> smaaOrigin : vec2i;

// ── Where a tile-relative pixel lives ──

// Rows -4 .. 18 are the horizontal arm's luma, the rest the vertical arm's ends.
fn smaaLumaInH(r: vec2i) -> bool {
    return r.y >= SMAA_ACROSS0 - 2 && r.y <= SMAA_ACROSS1 + 1;
}

fn smaaLumaIndexH(r: vec2i) -> i32 {
    return (r.y - (SMAA_ACROSS0 - 2)) * SMAA_LUMA_ALONG_N + (r.x - (SMAA_ALONG0 - 2));
}

fn smaaLumaIndexV(r: vec2i) -> i32 {
    var row = r.y - (SMAA_ALONG0 - 2);
    if (r.y > SMAA_ACROSS1 + 1) {
        row -= SMAA_LUMA_ACROSS_N;
    }
    return row * SMAA_LUMA_ACROSS_N + (r.x - (SMAA_ACROSS0 - 2));
}

// Rows -2 .. 17 are the horizontal arm's edges, the rest the vertical arm's ends. The word and
// the shift of r's two bits.
fn smaaEdgesInH(r: vec2i) -> bool {
    return r.y >= SMAA_ACROSS0 && r.y <= SMAA_ACROSS1;
}

fn smaaEdgeSlotH(r: vec2i) -> vec2i {
    let c = r.x - SMAA_ALONG0;
    return vec2i((r.y - SMAA_ACROSS0) * SMAA_H_WORDS + c / 16, (c % 16) * 2);
}

fn smaaEdgeSlotV(r: vec2i) -> vec2i {
    var c = r.y - SMAA_ALONG0;
    if (r.y > SMAA_ACROSS1) {
        c -= SMAA_ACROSS_N;
    }
    return vec2i((r.x - SMAA_ACROSS0) * SMAA_V_WORDS + c / 16, (c % 16) * 2);
}

// ── smaa_common.wgsl's accessors ──

fn smaaColorAt(p: vec2i) -> vec3f {
    return textureLoad(smaaInput, smaaClampToFrame(p), 0).rgb;
}

fn smaaLumaAt(p: vec2i) -> f32 {
    let r = p - smaaOrigin;
    if (smaaLumaInH(r)) {
        return smaaLumaH[smaaLumaIndexH(r)];
    }
    return smaaLumaV[smaaLumaIndexV(r)];
}

fn smaaEdgesAt(p: vec2i) -> vec2f {
    if (!smaaInFrame(p)) {
        return vec2f(0.0);
    }
    let r = p - smaaOrigin;
    var bits : u32;
    if (smaaEdgesInH(r)) {
        let slot = smaaEdgeSlotH(r);
        bits = atomicLoad(&smaaEdgesH[slot.x]) >> u32(slot.y);
    } else {
        let slot = smaaEdgeSlotV(r);
        bits = atomicLoad(&smaaEdgesV[slot.x]) >> u32(slot.y);
    }
    return vec2f(f32(bits & 1u), f32((bits >> 1u) & 1u));
}

// ── Filling the tile's memory ──

// The luma at r, of the input clamped to the frame.
fn smaaStoreLuma(r: vec2i) {
    let luma = smaaLumaOf(smaaColorAt(smaaOrigin + r));
    if (smaaLumaInH(r)) {
        smaaLumaH[smaaLumaIndexH(r)] = luma;
    } else {
        smaaLumaV[smaaLumaIndexV(r)] = luma;
    }
}

// The edges at r (none outside the frame); whether it has any.
fn smaaStoreEdges(r: vec2i) -> bool {
    let p = smaaOrigin + r;
    if (!smaaInFrame(p)) {
        return false;
    }
    let e = smaaDetectEdges(p);
    let bits = u32(e.x) | (u32(e.y) << 1u);
    if (bits == 0u) {
        return false;
    }
    if (smaaEdgesInH(r)) {
        let slot = smaaEdgeSlotH(r);
        atomicOr(&smaaEdgesH[slot.x], bits << u32(slot.y));
    } else {
        let slot = smaaEdgeSlotV(r);
        atomicOr(&smaaEdgesV[slot.x], bits << u32(slot.y));
    }
    return true;
}

// Phase 1's luma block, -2 .. 17 square.
fn smaaInLumaBlock(r: vec2i) -> bool {
    return all(r >= vec2i(-2)) && all(r <= vec2i(SMAA_BLOCK));
}

// The edges' block, 0 .. 16 square.
fn smaaInBlock(r: vec2i) -> bool {
    return all(r >= vec2i(0)) && all(r < vec2i(SMAA_BLOCK));
}

// The loops below run a uniform number of times (each invocation takes items i, i + 256, ...),
// so the barriers after them are in uniform control flow.

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(local_invocation_index) li: u32
) {
    smaaOrigin = vec2i(wid.xy) * SMAA_TILE;
    let i = i32(li);
    let pixel = smaaOrigin + vec2i(lid.xy);

    // Nothing else is read before it is written, so only the edge flags and the tile's flag are
    // cleared (smaa.rs turns off the backend's zeroing of the whole workgroup memory, which naga's
    // Metal backend does from one invocation)
    if (i < SMAA_ACROSS_N * SMAA_H_WORDS) {
        atomicStore(&smaaEdgesH[i], 0u);
    }
    if (i < SMAA_ACROSS_N * SMAA_V_WORDS) {
        atomicStore(&smaaEdgesV[i], 0u);
    }
    if (i == 0) {
        smaaTileHasEdges = 0u;
    }

    // Phase 1: the block's luma surroundings (each invocation its own pixel, kept for a
    // pass-through, then the 144 round the tile), then its edges (each its own pixel, then the
    // 33 of the ring right of and below the tile)
    let color = smaaColorAt(pixel);
    smaaLumaH[smaaLumaIndexH(vec2i(lid.xy))] = smaaLumaOf(color);
    const RING = SMAA_LUMA_BLOCK_N * SMAA_LUMA_BLOCK_N - SMAA_GROUP;
    if (i < RING) {
        // rows -2 and -1, rows 16 and 17, then columns -2, -1, 16 and 17 of rows 0 .. 15
        var r : vec2i;
        if (i < 2 * 2 * SMAA_LUMA_BLOCK_N) {
            let row = i / SMAA_LUMA_BLOCK_N;
            r = vec2i(i % SMAA_LUMA_BLOCK_N - 2, select(row - 2, row + SMAA_TILE - 2, row >= 2));
        } else {
            let n = i - 2 * 2 * SMAA_LUMA_BLOCK_N;
            let c = n % 4;
            r = vec2i(select(c - 2, c + SMAA_TILE - 2, c >= 2), n / 4);
        }
        smaaStoreLuma(r);
    }
    workgroupBarrier();
    if (smaaStoreEdges(vec2i(lid.xy))) {
        smaaTileHasEdges = 1u;
    }
    if (i < 2 * SMAA_TILE + 1) {
        // column 16 (rows 0 .. 16), then row 16 (columns 0 .. 15)
        let r = select(vec2i(i - SMAA_BLOCK, SMAA_TILE), vec2i(SMAA_TILE, i), i < SMAA_BLOCK);
        if (smaaStoreEdges(r)) {
            smaaTileHasEdges = 1u;
        }
    }
    if (workgroupUniformLoad(&smaaTileHasEdges) == 0u) {
        if (smaaInFrame(pixel)) {
            textureStore(smaaOutput, pixel, vec4f(color, 1.0));
        }
        return;
    }

    // Phase 2: the arms' luma (what phase 1 didn't load)
    const LUMA_H = SMAA_LUMA_ALONG_N * SMAA_LUMA_ACROSS_N;
    for (var j = 0; j < (LUMA_H + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        let r = vec2i(k % SMAA_LUMA_ALONG_N + SMAA_ALONG0 - 2, k / SMAA_LUMA_ALONG_N + SMAA_ACROSS0 - 2);
        if (k < LUMA_H && !smaaInLumaBlock(r)) {
            smaaStoreLuma(r);
        }
    }
    const LUMA_V = SMAA_LUMA_ACROSS_N * SMAA_LUMA_ENDS_N;
    for (var j = 0; j < (LUMA_V + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < LUMA_V) {
            // rows -18 .. -5, then 19 .. 34
            var y = k / SMAA_LUMA_ACROSS_N + SMAA_ALONG0 - 2;
            if (y >= SMAA_ACROSS0 - 2) {
                y += SMAA_LUMA_ACROSS_N;
            }
            smaaStoreLuma(vec2i(k % SMAA_LUMA_ACROSS_N + SMAA_ACROSS0 - 2, y));
        }
    }
    workgroupBarrier();

    // their edges
    const EDGES_H = SMAA_ALONG_N * SMAA_ACROSS_N;
    for (var j = 0; j < (EDGES_H + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        let r = vec2i(k % SMAA_ALONG_N + SMAA_ALONG0, k / SMAA_ALONG_N + SMAA_ACROSS0);
        if (k < EDGES_H && !smaaInBlock(r)) {
            smaaStoreEdges(r);
        }
    }
    const EDGES_V = SMAA_ACROSS_N * SMAA_EDGE_ENDS_N;
    for (var j = 0; j < (EDGES_V + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < EDGES_V) {
            // rows -16 .. -3, then 18 .. 33
            var y = k / SMAA_ACROSS_N + SMAA_ALONG0;
            if (y >= SMAA_ACROSS0) {
                y += SMAA_ACROSS_N;
            }
            smaaStoreEdges(vec2i(k % SMAA_ACROSS_N + SMAA_ACROSS0, y));
        }
    }
    workgroupBarrier();

    // the block's weights
    const BLOCK = SMAA_BLOCK * SMAA_BLOCK;
    for (var j = 0; j < (BLOCK + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < BLOCK) {
            let q = smaaOrigin + vec2i(k % SMAA_BLOCK, k / SMAA_BLOCK);
            smaaBlockWeights[k] = pack4x8unorm(smaaQuantizeWeights(smaaWeights(q, SMAA_REACH)));
        }
    }
    workgroupBarrier();

    // the tile's blend
    if (smaaInFrame(pixel)) {
        let b = i32(lid.y) * SMAA_BLOCK + i32(lid.x);
        let here = unpack4x8unorm(smaaBlockWeights[b]);
        let right = unpack4x8unorm(smaaBlockWeights[b + 1]);
        let below = unpack4x8unorm(smaaBlockWeights[b + SMAA_BLOCK]);
        textureStore(smaaOutput, pixel, vec4f(smaaBlend(pixel, here, right, below), 1.0));
    }
}
