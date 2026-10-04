// The lightbox's particles (after VOXEL_CONES_WGSL, SKY_LIGHTING_WGSL, SCENE_WGSL and
// room_common.wgsl): spheres of varied radius, ray cast on camera-facing quads at their nearest
// point, lit by what the GI gathered for them (`lighting`, two vec4 per particle) and the panel.
// `mirrored` draws them under the floor (the reflection). The quads keep their own depth rather
// than writing frag_depth, which would run the shading before the depth test, for every sphere of
// a pile dozens deep; the fragment entries follow, a depth prepass (room_spheres_depth.wgsl) and
// the shading on equal depth (room_spheres_shade.wgsl). Each particle's position w is its radius
// as a share of `size`.
//
// Ray traced (scene.rtOn), after the bonus of miaumiau.cat/?p=1476: a share of the particles are
// mirrors or glass, and their reflected and refracted rays walk the fluid's own neighbour grid
// (sorted positions, cell offsets) with ray-sphere tests, no higher than the pile's top
// (pile_top.wgsl). A ray leaving the particles meets the room, lit by the panel and by one cone
// through the volume.
struct Particles { albedo: vec4f, size: f32, mirrored: f32, _p0: f32, _p1: f32 };
// the fluid's neighbour grid (FluidSimulation::grid_dims, cell_size, grid_origin)
struct Grid { origin: vec3f, cellSize: f32, dims: vec3u, count: u32 };

@group(0) @binding(0) var<uniform> particles: Particles;
@group(0) @binding(1) var<uniform> scene: SceneParams;
@group(0) @binding(2) var<uniform> grid: Grid;
@group(0) @binding(3) var<storage, read> sortedPositions: array<vec4f>;
@group(0) @binding(4) var<storage, read> cellOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(6) var<storage, read> lighting: array<vec4f>;
@group(0) @binding(7) var<uniform> vol: VoxelVolume;
@group(0) @binding(8) var radiance: texture_3d<f32>;
@group(0) @binding(9) var linearClamp: sampler;
// the highest particle centre (pile_top.wgsl), as f32 bits
@group(0) @binding(10) var<storage, read> pileTop: u32;
// the sky past the volume (ParticleGi::sky_buffer)
@group(0) @binding(11) var<uniform> sky: SkyLighting;
@group(1) @binding(0) var<uniform> view_matrix: mat4x4<f32>;
@group(1) @binding(1) var<uniform> projection_matrix: mat4x4<f32>;
@group(2) @binding(0) var<uniform> _normal_matrix: mat4x4<f32>;
@group(2) @binding(1) var<uniform> _world_matrix: mat4x4<f32>;

const MATTE: u32 = 0u;
const MIRROR: u32 = 1u;
const GLASS: u32 = 2u;
const NO_HIT: u32 = 0xffffffffu;

fn hash01(x: u32) -> f32 {
    // PCG (Jarzynski & Olano 2020)
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return f32(((word >> 22u) ^ word) >> 8u) / 16777216.0;
}

// A particle's radius from its position's w (the share of `size` it is given, see
// `radius_share`), under half a grid cell, so the ray walk below finds every sphere.
fn particleRadius(w: f32) -> f32 {
    return min(0.5 * particles.size * w, 0.49 * grid.cellSize);
}

fn particleKind(i: u32) -> u32 {
    if (scene.rtOn < 0.5) { return MATTE; }
    let h = hash01(i * 8u + 2u);
    if (h < scene.glassShare) { return GLASS; }
    if (h < scene.glassShare + scene.mirrorShare) { return MIRROR; }
    return MATTE;
}

// charcoal to brown (grey when ray traced); the glowing ones lighter, toward their glow
fn particleAlbedo(i: u32, glows: bool) -> vec3f {
    let h = hash01(i * 8u + 3u);
    if (glows) { return mix(vec3f(0.45, 0.3, 0.15), vec3f(0.7, 0.55, 0.3), h); }
    if (scene.rtOn > 0.5) { return vec3f(mix(0.45, 0.8, h)); }
    return mix(vec3f(0.07, 0.06, 0.055), vec3f(0.3, 0.19, 0.1), h * h);
}

// orange to yellow around the GI's emission colour, dimmer or brighter per particle
fn glowTint(i: u32) -> vec3f {
    let h = hash01(i * 8u + 4u);
    let k = mix(0.03, 0.14, hash01(i * 8u + 5u));
    return k * vec3f(1.0, mix(0.8, 1.7, h), mix(0.6, 2.2, h * h));
}

// a matte particle at p facing n: the panel (shadowed by the GI's cone toward it), the light the
// cones gathered, its glow
// (seen from v, toward the eye: a glowing sphere is brightest face on)
fn shadeMatte(i: u32, p: vec3f, n: vec3f, v: vec3f) -> vec3f {
    let light = lighting[2u * i];
    let emission = lighting[2u * i + 1u].rgb;
    let glows = any(emission > vec3f(0.0));
    let albedo = particleAlbedo(i, glows);
    let direct = panelIrradiance(scene, p, n) * light.a;
    let face = 0.35 + 0.65 * max(dot(n, v), 0.0);
    return albedo * (direct / PI + light.rgb) + emission * glowTint(i) * face;
}

struct Hit { t: f32, index: u32, center: vec3f, r: f32 };

// The nearest sphere along o + t d (t < tEnd), skipping particle `skip`. A 3D DDA over the dual
// of the fluid's grid (cells between its cells' centres): a sphere under half a cell wide that
// reaches into a dual cell has its centre in the 2x2x2 grid cells around it, so testing those
// finds every hit inside the dual cell, and the walk stops at the first dual cell holding one.
fn traceParticles(o: vec3f, d: vec3f, skip: u32, tEnd: f32) -> Hit {
    var hit = Hit(tEnd, NO_HIT, vec3f(0.0), 0.0);
    let cs = grid.cellSize;
    let dims = vec3i(grid.dims);
    let total = grid.dims.x * grid.dims.y * grid.dims.z;
    let inv = select(vec3f(1e30), 1.0 / d, abs(d) > vec3f(1e-8));
    // clip to the grid's box
    let lo = (grid.origin - o) * inv;
    let hi = (grid.origin + vec3f(grid.dims) * cs - o) * inv;
    var t0 = max(max(max(min(lo.x, hi.x), min(lo.y, hi.y)), min(lo.z, hi.z)), 0.0);
    var t1 = min(min(min(max(lo.x, hi.x), max(lo.y, hi.y)), max(lo.z, hi.z)), tEnd);
    // and to the slab under the highest centre plus the largest radius: nothing is above it
    let top = bitcast<f32>(pileTop) + 0.625 * particles.size;
    if (d.y > 0.0) {
        t1 = min(t1, (top - o.y) * inv.y);
    } else if (o.y > top) {
        t0 = max(t0, (top - o.y) * inv.y);
    }
    if (t0 >= t1) { return hit; }
    // dual coordinates: dual cell k spans the centres of grid cells k and k + 1
    let g = (o + d * t0 - grid.origin) / cs - 0.5;
    var cell = vec3i(floor(g));
    let stride = vec3i(sign(d));
    let delta = abs(cs * inv);
    var tMax = t0 + (select(vec3f(cell), vec3f(cell + 1), d > vec3f(0.0)) - g) * cs * inv;
    tMax = select(tMax, vec3f(1e30), abs(d) <= vec3f(1e-8));
    // after a step along `axis`, the block shares 4 cells with the one before, whose spheres are
    // already tested: test the 4 on its leading side (`side`: 1 when the step was +1)
    var axis = 3u;
    var side = 0u;
    for (var s = 0u; s < 96u; s++) {
        // the block's cells as 4 rows along x: neighbours along x are neighbours in the sorted
        // order too, so each row's spheres are one range
        for (var row = 0u; row < 4u; row++) {
            let dy = row & 1u;
            let dz = row >> 1u;
            if ((axis == 1u && dy != side) || (axis == 2u && dz != side)) { continue; }
            let y = cell.y + i32(dy);
            let z = cell.z + i32(dz);
            if (y < 0 || z < 0 || y >= dims.y || z >= dims.z) { continue; }
            var x0 = cell.x;
            var x1 = cell.x + 1;
            if (axis == 0u) {
                x0 = cell.x + i32(side);
                x1 = x0;
            }
            x0 = max(x0, 0);
            x1 = min(x1, dims.x - 1);
            if (x0 > x1) { continue; }
            let base = grid.dims.x * (u32(y) + grid.dims.y * u32(z));
            let first = cellOffsets[base + u32(x0)];
            let end = base + u32(x1) + 1u;
            let last = select(cellOffsets[end], grid.count, end >= total);
            for (var j = first; j < last; j++) {
                let sphere = sortedPositions[j];
                let r = particleRadius(sphere.w);
                let oc = o - sphere.xyz;
                let b = dot(oc, d);
                let h = b * b - (dot(oc, oc) - r * r);
                if (h < 0.0) { continue; }
                // a ray starting inside an overlapping sphere passes through it
                let t = -b - sqrt(h);
                if (t > 1e-4 && t < hit.t) {
                    // the index only for a hit: the sphere the ray leaves is no hit
                    let index = sortedIndices[j];
                    if (index != skip) {
                        hit = Hit(t, index, sphere.xyz, r);
                    }
                }
            }
        }
        let exit = min(tMax.x, min(tMax.y, tMax.z));
        if (hit.index != NO_HIT && hit.t <= exit) { break; }
        if (exit >= t1) { break; }
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            cell.x += stride.x;
            tMax.x += delta.x;
            axis = 0u;
            side = u32(stride.x > 0);
        } else if (tMax.y <= tMax.z) {
            cell.y += stride.y;
            tMax.y += delta.y;
            axis = 1u;
            side = u32(stride.y > 0);
        } else {
            cell.z += stride.z;
            tMax.z += delta.z;
            axis = 2u;
            side = u32(stride.z > 0);
        }
    }
    return hit;
}

// Where o + t d leaves the room (t, and the wall's inward normal). The front, toward the camera,
// is open.
fn roomExit(o: vec3f, d: vec3f) -> vec4f {
    let inv = select(vec3f(1e30), 1.0 / d, abs(d) > vec3f(1e-8));
    let t = (select(scene.roomMin, scene.roomMax, d > vec3f(0.0)) - o) * inv;
    if (t.x <= t.y && t.x <= t.z) { return vec4f(-sign(d.x), 0.0, 0.0, t.x); }
    if (t.y <= t.z) { return vec4f(0.0, -sign(d.y), 0.0, t.y); }
    return vec4f(0.0, 0.0, -sign(d.z), t.z);
}

// The room seen along o + t d, t up to `exit` (roomExit): the panel, or a wall lit by the panel
// and by one wide cone through the volume; black out of the open front.
fn shadeRoom(o: vec3f, d: vec3f, exit: vec4f) -> vec3f {
    let n = exit.xyz;
    if (n.z < -0.5) { return vec3f(0.0); }
    let p = o + d * exit.w;
    if (n.y < -0.5) {
        let panel = panelSeen(scene, o, d);
        if (any(panel > vec3f(0.0))) { return panel; }
    }
    let q = p + n * vol.voxelSize;
    let c = voxelConeTrace(vol, radiance, linearClamp, q, n, 1.0, vol.voxelSize, 1e4, 12u);
    return scene.wallAlbedo * (panelIrradiance(scene, p, n) / PI + c.rgb + c.a * skyRadiance(sky, n));
}

// what a ray sees, mirrors and glass met on the way taken as matte (the last bounce)
fn traceLast(o: vec3f, d: vec3f, skip: u32) -> vec3f {
    let exit = roomExit(o, d);
    let hit = traceParticles(o, d, skip, exit.w);
    if (hit.index == NO_HIT) { return shadeRoom(o, d, exit); }
    let p = o + d * hit.t;
    return shadeMatte(hit.index, p, normalize(p - hit.center), -d);
}

// A glass sphere's light toward -d at p (normal n): Fresnel's share of the reflection, the rest
// refracted through it, out of its far side.
fn glassRays(p: vec3f, n: vec3f, d: vec3f, center: vec3f, r: f32) -> array<vec3f, 4> {
    let eta = 1.0 / scene.ior;
    let inside = refract(d, n, eta);
    // the chord to the far side, and out (reflected back in at total internal reflection)
    let q = p + inside * max(-2.0 * dot(p - center, inside), 0.0);
    let nq = normalize(q - center);
    var out = refract(inside, -nq, scene.ior);
    if (dot(out, out) < 1e-6) { out = reflect(inside, -nq); }
    return array<vec3f, 4>(reflect(d, n), q, out, vec3f(0.0));
}

// The rays a mirror or a glass sphere sends on from p (normal n, met along d), with their weights:
// a mirror its reflection; glass Fresnel's share of its reflection, and the rest refracted out of
// its far side.
struct Rays {
    count : u32,
    o0    : vec3f,
    d0    : vec3f,
    w0    : vec3f,
    o1    : vec3f,
    d1    : vec3f,
    w1    : vec3f,
};
fn sphereRays(kind: u32, p: vec3f, n: vec3f, d: vec3f, center: vec3f, r: f32) -> Rays {
    let cosV = dot(-d, n);
    if (kind == MIRROR) {
        return Rays(1u, p, reflect(d, n), mirrorTint(cosV), vec3f(0.0), vec3f(0.0), vec3f(0.0));
    }
    let g = glassRays(p, n, d, center, r);
    let f = fresnel(glassF0(), cosV);
    return Rays(2u, p, g[0], vec3f(f), g[1], g[2], (1.0 - f) * glassTint());
}

// What a ray sees; with `bounce`, the mirrors and glass it meets send their rays on once more
// (seen as matte past that). One call site per walk keeps the shader small.
fn trace(o: vec3f, d: vec3f, skip: u32) -> vec3f {
    let exit = roomExit(o, d);
    let hit = traceParticles(o, d, skip, exit.w);
    if (hit.index == NO_HIT) { return shadeRoom(o, d, exit); }
    let p = o + d * hit.t;
    let n = normalize(p - hit.center);
    let kind = particleKind(hit.index);
    if (scene.rtBounces < 1.5 || kind == MATTE) {
        return shadeMatte(hit.index, p, n, -d);
    }
    let rays = sphereRays(kind, p, n, d, hit.center, hit.r);
    var sum = vec3f(0.0);
    for (var k = 0u; k < rays.count; k++) {
        let first = k == 0u;
        sum += select(rays.w1, rays.w0, first) * traceLast(select(rays.o1, rays.o0, first), select(rays.d1, rays.d0, first), hit.index);
    }
    return sum;
}

fn glassF0() -> f32 {
    let k = (scene.ior - 1.0) / (scene.ior + 1.0);
    return k * k;
}
fn glassTint() -> vec3f { return vec3f(0.94, 0.97, 0.96); }
// glossy black: a dark metal's reflection, brighter at grazing angles
fn mirrorTint(cosTheta: f32) -> vec3f { return vec3f(mix(0.55, 1.0, pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0))); }


struct VIn {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) center: vec4f,
};
struct VOut {
    // invariant: the depth prepass and the shading must agree on depth to the bit
    @builtin(position) @invariant clip: vec4f,
    @location(0) viewPos: vec3f,
    @location(1) @interpolate(flat) sphere: vec4f,   // view-space centre (as drawn), radius
    @location(2) @interpolate(flat) world: vec3f,    // world centre (unmirrored)
    @location(3) @interpolate(flat) index: u32,
};

@vertex
fn vertex_main(v: VIn, @builtin(instance_index) index: u32) -> VOut {
    let r = particleRadius(v.center.w);
    var shown = v.center.xyz;
    if (particles.mirrored > 0.5) {
        shown.y = 2.0 * scene.mirrorY - shown.y;
    }
    let c = (view_matrix * vec4f(shown, 1.0)).xyz;
    // a quad facing the eye at the sphere's nearest point, as wide as the cone of rays grazing
    // the sphere is there
    let len = max(length(c), 1.001 * r);
    let w = c / len;
    let u = normalize(select(cross(w, vec3f(0.0, 1.0, 0.0)), cross(w, vec3f(1.0, 0.0, 0.0)), abs(w.y) > 0.99));
    let b = cross(u, w);
    let s = 1.02 * r * (len - r) / sqrt(len * len - r * r);
    let p = c - w * r + (u * v.position.x + b * v.position.y) * 2.0 * s;
    var out: VOut;
    out.clip = projection_matrix * vec4f(p, 1.0);
    out.viewPos = p;
    out.sphere = vec4f(c, r);
    out.world = v.center.xyz;
    out.index = index;
    return out;
}

// The eye ray's hit on the sphere, in view space; the quad's corners past it are discarded.
fn eyeHit(in: VOut) -> vec3f {
    let ray = normalize(in.viewPos);
    let c = in.sphere.xyz;
    let r = in.sphere.w;
    let b = dot(ray, c);
    let h = b * b - (dot(c, c) - r * r);
    if (h < 0.0) { discard; }
    return ray * (b - sqrt(h));
}
