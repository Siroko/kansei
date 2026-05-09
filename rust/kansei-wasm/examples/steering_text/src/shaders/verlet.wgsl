// Verlet distance constraint: for each trailing letter (letter_idx > 0),
// snap it to rest_length distance from the previous letter.
//
// Only the CURRENT particle is moved — the previous one (closer to the
// vehicle) is treated as an anchor. This prevents trailing letters from
// pulling the vehicle backward and eliminates the parallel race condition
// where two constraints fighting over a shared particle cause stutter.
//
// Dispatched over particleCount, runs multiple iterations per frame.

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> wordMeta: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> restLengths: array<f32>;
@group(0) @binding(3) var<uniform> params: SimParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.particleCount) { return; }

    let wm = wordMeta[idx];
    let letterIdx = wm.y;

    // Only constrain trailing letters (not the vehicle)
    if (letterIdx == 0u) { return; }

    let particleOffset = wm.w;
    let prevIdx = particleOffset + letterIdx - 1u;
    let restLen = restLengths[idx];

    if (restLen <= 0.0) { return; }

    let anchor = positions[prevIdx].xyz;
    let pos = positions[idx].xyz;
    let delta = pos - anchor;
    let dist = length(delta);

    if (dist < 0.0001) {
        // Nudge apart to avoid zero-length division
        positions[idx] = vec4<f32>(anchor + vec3<f32>(restLen, 0.0, 0.0), 1.0);
        return;
    }

    // Snap current particle to exactly restLen from anchor.
    // Only the current particle moves — anchor stays fixed.
    let snapped = anchor + (delta / dist) * restLen;
    positions[idx] = vec4<f32>(snapped, 1.0);
}
