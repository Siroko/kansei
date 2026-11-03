// SmaaEffect's single pass (SmaaMode::SinglePass): SMAA 1x for a 32 x 32 tile in one dispatch,
// its edges and weights kept in workgroup memory. Appended to smaa_common.wgsl, whose accessors
// it defines; smaa.rs replaces KANSEI_SMAA_REACH with SmaaEffect::REACH.
//
// A workgroup of 16 x 16 invocations serves one tile of 32 x 32 output pixels, each invocation
// the four pixels (x, y), (x + 16, y), (x, y + 16) and (x + 16, y + 16) of its local id. The
// tile's blend reads the weights of a 33 x 33 block: the tile and the ring right of it and below.
//   Phase 1: the block's edges, found from the input (each invocation its own four pixels,
//            whose colours it keeps, then the 65 of the ring). A tile with none comes through as
//            it is, and stops.
//   Phase 2: the edges of the apron the block's searches reach, in two arms: the horizontal arm
//            along the rows (runs of top edges), the vertical arm down the columns (runs of left
//            edges), each SMAA_REACH past the block both ways. Then the block's weights
//            (quantized as the three passes' rgba8unorm stores them), then the tile's blend.
// The edges are found straight from the input's texels (smaaLumaAt reads the texture), into
// bit masks: no luma is kept. Coordinates below are tile-relative (r = p - smaaOrigin). A run's
// search from the block reaches SMAA_REACH along it (SMAA_REACH + 1 for its far end's crossing
// edges), and 2 before and 1 after it across.

@group(0) @binding(4) var smaaOutput : texture_storage_2d<rgba16float, write>;

const SMAA_REACH      : i32 = KANSEI_SMAA_REACH;
const SMAA_GROUP_SIDE : i32 = 16;
const SMAA_GROUP      : i32 = SMAA_GROUP_SIDE * SMAA_GROUP_SIDE;
const SMAA_TILE       : i32 = 2 * SMAA_GROUP_SIDE;
const SMAA_BLOCK      : i32 = SMAA_TILE + 1;
// the ring right of and below the tile: column 32 (rows 0 .. 32), row 32 (columns 0 .. 31)
const SMAA_RING       : i32 = 2 * SMAA_TILE + 1;

// The edges an arm holds: ACROSS a run, -2 .. 33; ALONG it, -REACH .. 32 + REACH + 1.
const SMAA_ACROSS0  : i32 = -2;
const SMAA_ACROSS1  : i32 = SMAA_BLOCK;
const SMAA_ALONG0   : i32 = -SMAA_REACH;
const SMAA_ALONG1   : i32 = SMAA_BLOCK + SMAA_REACH;
const SMAA_ACROSS_N : i32 = SMAA_ACROSS1 - SMAA_ACROSS0 + 1;
const SMAA_ALONG_N  : i32 = SMAA_ALONG1 - SMAA_ALONG0 + 1;

// The horizontal arm keeps its whole extent, columns -16 .. 49 by rows -2 .. 33 (2 bits a
// pixel: left, top; 16 pixels a word, along a row). The vertical arm keeps only its ends, the rows
// the horizontal arm lacks: rows -16 .. -3 and 34 .. 49 of columns -2 .. 33 (packed down a column).
const SMAA_EDGE_ENDS_N : i32 = SMAA_ALONG_N - SMAA_ACROSS_N;
const SMAA_H_WORDS     : i32 = (SMAA_ALONG_N + 15) / 16;
const SMAA_V_WORDS     : i32 = (SMAA_EDGE_ENDS_N + 15) / 16;
const SMAA_H_SIZE      : i32 = SMAA_ACROSS_N * SMAA_H_WORDS;
const SMAA_V_SIZE      : i32 = SMAA_ACROSS_N * SMAA_V_WORDS;

var<workgroup> smaaEdgesH       : array<atomic<u32>, SMAA_H_SIZE>;
var<workgroup> smaaEdgesV       : array<atomic<u32>, SMAA_V_SIZE>;
var<workgroup> smaaBlockWeights : array<u32, SMAA_BLOCK * SMAA_BLOCK>;
var<workgroup> smaaTileHasEdges : u32;
var<private> smaaOrigin : vec2i;

// ── Where a tile-relative pixel's edges live ──

// Rows -2 .. 33 are the horizontal arm's, the rest the vertical arm's ends. The word and the
// shift of r's two bits.
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
    return smaaLumaOf(smaaColorAt(p));
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

// The invocation's q-th pixel (0 .. 3) of the tile.
fn smaaOwnPixel(own: vec2i, q: i32) -> vec2i {
    return own + SMAA_GROUP_SIDE * vec2i(q % 2, q / 2);
}

// The ring's i-th pixel: column 32 (rows 0 .. 32), then row 32 (columns 0 .. 31).
fn smaaRingPixel(i: i32) -> vec2i {
    return select(vec2i(i - SMAA_BLOCK, SMAA_TILE), vec2i(SMAA_TILE, i), i < SMAA_BLOCK);
}

// The block's weights at r, packed.
fn smaaStoreWeights(r: vec2i) {
    smaaBlockWeights[r.y * SMAA_BLOCK + r.x] = pack4x8unorm(smaaQuantizeWeights(smaaWeights(smaaOrigin + r, SMAA_REACH)));
}

// The loops below run a uniform number of times (each invocation takes items i, i + 256, ...)
// and never break, so the barriers after them are in uniform control flow.

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wid: vec3u,
    @builtin(local_invocation_id) lid: vec3u,
    @builtin(local_invocation_index) li: u32
) {
    smaaOrigin = vec2i(wid.xy) * SMAA_TILE;
    let i = i32(li);
    let own = vec2i(lid.xy);

    // Nothing else is read before it is written, so only the edge flags and the tile's flag are
    // cleared (smaa.rs turns off the backend's zeroing of the whole workgroup memory, which naga's
    // Metal backend does from one invocation)
    for (var j = 0; j < (SMAA_H_SIZE + SMAA_V_SIZE + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < SMAA_H_SIZE) {
            atomicStore(&smaaEdgesH[k], 0u);
        } else if (k < SMAA_H_SIZE + SMAA_V_SIZE) {
            atomicStore(&smaaEdgesV[k - SMAA_H_SIZE], 0u);
        }
    }
    if (i == 0) {
        smaaTileHasEdges = 0u;
    }
    workgroupBarrier();

    // Phase 1: the block's edges (each invocation its own four pixels, their colours kept for a
    // pass-through, then the ring's)
    var colors : array<vec3f, 4>;
    var found = false;
    for (var q = 0; q < 4; q++) {
        let r = smaaOwnPixel(own, q);
        colors[q] = smaaColorAt(smaaOrigin + r);
        if (smaaStoreEdges(r)) {
            found = true;
        }
    }
    if (i < SMAA_RING) {
        if (smaaStoreEdges(smaaRingPixel(i))) {
            found = true;
        }
    }
    if (found) {
        smaaTileHasEdges = 1u;
    }
    if (workgroupUniformLoad(&smaaTileHasEdges) == 0u) {
        for (var q = 0; q < 4; q++) {
            let p = smaaOrigin + smaaOwnPixel(own, q);
            if (smaaInFrame(p)) {
                textureStore(smaaOutput, p, vec4f(colors[q], 1.0));
            }
        }
        return;
    }

    // Phase 2: the arms' edges. The horizontal arm round the block: rows -2, -1 and 33 whole,
    // then columns -16 .. -1 and 33 .. 49 of rows 0 .. 32.
    const ROWS = 3 * SMAA_ALONG_N;
    const SIDE_N = SMAA_ALONG_N - SMAA_BLOCK;
    const SIDES = SMAA_BLOCK * SIDE_N;
    for (var j = 0; j < (ROWS + SIDES + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < ROWS) {
            let row = k / SMAA_ALONG_N;
            smaaStoreEdges(vec2i(k % SMAA_ALONG_N + SMAA_ALONG0, select(SMAA_BLOCK, row - 2, row < 2)));
        } else if (k < ROWS + SIDES) {
            let n = k - ROWS;
            let c = n % SIDE_N;
            smaaStoreEdges(vec2i(select(c + SMAA_BLOCK + SMAA_ALONG0, c + SMAA_ALONG0, c < -SMAA_ALONG0), n / SIDE_N));
        }
    }
    // the vertical arm's ends: rows -16 .. -3, then 34 .. 49
    const ENDS = SMAA_ACROSS_N * SMAA_EDGE_ENDS_N;
    for (var j = 0; j < (ENDS + SMAA_GROUP - 1) / SMAA_GROUP; j++) {
        let k = i + j * SMAA_GROUP;
        if (k < ENDS) {
            var y = k / SMAA_ACROSS_N + SMAA_ALONG0;
            if (y >= SMAA_ACROSS0) {
                y += SMAA_ACROSS_N;
            }
            smaaStoreEdges(vec2i(k % SMAA_ACROSS_N + SMAA_ACROSS0, y));
        }
    }
    workgroupBarrier();

    // the block's weights: each invocation its own four pixels, then the ring's
    for (var q = 0; q < 4; q++) {
        smaaStoreWeights(smaaOwnPixel(own, q));
    }
    if (i < SMAA_RING) {
        smaaStoreWeights(smaaRingPixel(i));
    }
    workgroupBarrier();

    // the tile's blend
    for (var q = 0; q < 4; q++) {
        let r = smaaOwnPixel(own, q);
        let p = smaaOrigin + r;
        if (smaaInFrame(p)) {
            let b = r.y * SMAA_BLOCK + r.x;
            let here = unpack4x8unorm(smaaBlockWeights[b]);
            let right = unpack4x8unorm(smaaBlockWeights[b + 1]);
            let below = unpack4x8unorm(smaaBlockWeights[b + SMAA_BLOCK]);
            textureStore(smaaOutput, p, vec4f(smaaBlend(p, here, right, below), 1.0));
        }
    }
}
