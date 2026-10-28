// The floor's technical grid (floor.wgsl, and the ray traced hits on the floor through lib.rs's
// albedo hook): fine minor lines, thicker major lines every few, and at every second major
// intersection a small stronger cross, in world x and z. Lines are anti-aliased over a point's
// footprint (the floor: its pixel's, by screen-space derivatives; a ray's hit: one estimated from
// its distance) and fade to their average where they get finer than it (Ben Golus, "The Best Darn
// Grid Shader (Yet)"), so they neither shimmer nor moire at grazing angles.

struct FloorGrid {
    base     : vec4f,   // rgb albedo between the lines
    minor    : vec4f,   // rgb, w: the minor lines' spacing (m)
    major    : vec4f,   // rgb, w: minor cells per major cell
    cross    : vec4f,   // rgb, w: the cross's arm, a share of a major cell
    // the minor and major lines' and the cross's widths (m), the floor's roughness
    widths   : vec4f,
    // the floor's F0, then the sky (up, down: the albedo times it is its ambient)
    specular : vec4f,
    skyUp    : vec4f,
    skyDown  : vec4f,
}

// Coverage of lines at the integers of `uv` (cells), `width` wide (cells), anti-aliased over the
// pixel's footprint `deriv` and faded to their mean coverage where they are thinner than it.
fn gridLines(uv: vec2f, deriv: vec2f, width: vec2f) -> f32 {
    let wanted = min(width, vec2f(0.5));
    let draw = clamp(wanted, deriv, vec2f(0.5));
    let aa = deriv * 1.5;
    let d = 1.0 - abs(fract(uv) * 2.0 - 1.0);
    var lines = smoothstep(draw + aa, draw - aa, d);
    lines *= clamp(wanted / draw, vec2f(0.0), vec2f(1.0));
    lines = mix(lines, wanted, clamp(deriv * 2.0 - 1.0, vec2f(0.0), vec2f(1.0)));
    return mix(lines.x, 1.0, lines.y);
}

// The crosses at every second major intersection (in major cells `uv`): two bars `arm` long each
// way and `width` thick (cells), anti-aliased over the footprint, fading out where a cross is
// under a few pixels.
fn crosses(uv: vec2f, deriv: vec2f, arm: f32, width: f32) -> f32 {
    let node = round(uv);
    let even = all(abs(fract(node * 0.5)) < vec2f(0.25));
    if (!even) {
        return 0.0;
    }
    let q = abs(uv - node);
    let aa = max(deriv, vec2f(1e-5)) * 1.5;
    let along = smoothstep(vec2f(arm) + aa, vec2f(arm) - aa, q);
    let across = smoothstep(vec2f(width * 0.5) + aa, vec2f(width * 0.5) - aa, q);
    let c = max(along.x * across.y, along.y * across.x);
    // a cross smaller than about four pixels fades away rather than shimmer
    return c * clamp(arm / (4.0 * max(deriv.x, deriv.y)) - 0.5, 0.0, 1.0);
}

// The grid's albedo at floor point `p` (world x, z), over a footprint of `footprint` metres.
fn floorGridAlbedo(grid: FloorGrid, p: vec2f, footprint: vec2f) -> vec3f {
    let minorCell = grid.minor.w;
    let majorCell = minorCell * grid.major.w;
    let minorUV = p / minorCell;
    let majorUV = p / majorCell;
    let minor = gridLines(minorUV, footprint / minorCell, vec2f(grid.widths.x / minorCell));
    let major = gridLines(majorUV, footprint / majorCell, vec2f(grid.widths.y / majorCell));
    let marks = crosses(majorUV, footprint / majorCell, grid.cross.w, grid.widths.z / majorCell);
    var albedo = mix(grid.base.rgb, grid.minor.rgb, minor);
    albedo = mix(albedo, grid.major.rgb, major);
    return mix(albedo, grid.cross.rgb, marks);
}
