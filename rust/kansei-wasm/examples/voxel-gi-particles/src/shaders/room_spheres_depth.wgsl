// The spheres' depth prepass (after room_spheres.wgsl): only where each sphere is, so the shading
// pass after it, testing for equal depth, shades the nearest sphere alone. Its colour is drawn
// over.
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    _ = eyeHit(in);
    return vec4f(0.0, 0.0, 0.0, 1.0);
}
