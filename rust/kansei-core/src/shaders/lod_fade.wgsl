// Dithered LOD crossfades for materials (culling::InstanceCulling::with_crossfade): a culled
// instance's fade (an f32 after its record) says which share of its pixels it keeps, and this
// drops the others: `if (kansei_lod_fade_discard(in.lod_fade, in.position.xy, frame)) { discard; }`
// in the fragment shader (and the shadow fragment), with `frame` the camera's frame
// (`kansei_camera_temporal.frame`). The fading-out LOD keeps the pixels whose threshold is below
// its share and the fading-in one the rest, so the two cover each pixel once. The threshold is
// interleaved gradient noise (Jimenez 2014) turned by the golden ratio each frame, so temporal
// antialiasing blends the two LODs by their shares.

fn kansei_lod_fade_threshold(pixel : vec2f, frame : u32) -> f32 {
    let ign = fract(52.9829189 * fract(dot(floor(pixel), vec2f(0.06711056, 0.00583715))));
    return fract(ign + f32(frame % 64u) * 0.61803399);
}

// Whether to drop this pixel: fade 1 keeps them all, a share f in (0, 1) keeps those below it
// (fading out), -f in (-1, 0) keeps those at or above 1 - f (fading in).
fn kansei_lod_fade_discard(fade : f32, pixel : vec2f, frame : u32) -> bool {
    if (fade >= 1.0) { return false; }
    let t = kansei_lod_fade_threshold(pixel, frame);
    return select(t >= fade, t < 1.0 + fade, fade < 0.0);
}
