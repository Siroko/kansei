// The lightbox's particles (after VOXEL_CONES_WGSL, SCENE_WGSL and room_common.wgsl): spheres of
// varied radius, ray cast on camera-facing quads at their nearest point, lit by what the GI
// gathered for them (`lighting`, two vec4 per particle) and the panel. `mirrored` draws them under
// the floor (the reflection). The quads keep their own depth rather than writing frag_depth, which
// would run the shading before the depth test, for every sphere of a pile dozens deep.
//
// Ray traced (scene.rtOn), after the bonus of miaumiau.cat/?p=1476: a share of the particles are
// mirrors or glass, and their reflected and refracted rays walk the fluid's own neighbour grid
// (sorted positions, cell offsets) with ray-sphere tests. A ray leaving the particles meets the
// room, lit by the panel and by one cone through the volume.
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

// under half a grid cell, so the ray walk below finds every sphere
fn particleRadius(i: u32) -> f32 {
    return min(0.5 * particles.size * (0.45 + 0.8 * hash01(i * 8u + 1u)), 0.49 * grid.cellSize);
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

struct Hit { t: f32, index: u32, center: vec3f };

// The nearest sphere along o + t d (t < tEnd), skipping particle `skip`. A 3D DDA over the dual
// of the fluid's grid (cells between its cells' centres): a sphere under half a cell wide that
// reaches into a dual cell has its centre in the 2x2x2 grid cells around it, so testing those
// finds every hit inside the dual cell, and the walk stops at the first dual cell holding one.
fn traceParticles(o: vec3f, d: vec3f, skip: u32, tEnd: f32) -> Hit {
    var hit = Hit(tEnd, NO_HIT, vec3f(0.0));
    let cs = grid.cellSize;
    let dims = vec3i(grid.dims);
    let total = grid.dims.x * grid.dims.y * grid.dims.z;
    let inv = select(vec3f(1e30), 1.0 / d, abs(d) > vec3f(1e-8));
    // clip to the grid's box
    let lo = (grid.origin - o) * inv;
    let hi = (grid.origin + vec3f(grid.dims) * cs - o) * inv;
    let t0 = max(max(max(min(lo.x, hi.x), min(lo.y, hi.y)), min(lo.z, hi.z)), 0.0);
    let t1 = min(min(min(max(lo.x, hi.x), max(lo.y, hi.y)), max(lo.z, hi.z)), tEnd);
    if (t0 >= t1) { return hit; }
    // dual coordinates: dual cell k spans the centres of grid cells k and k + 1
    let g = (o + d * t0 - grid.origin) / cs - 0.5;
    var cell = vec3i(floor(g));
    let stride = vec3i(sign(d));
    let delta = abs(cs * inv);
    var tMax = t0 + (select(vec3f(cell), vec3f(cell + 1), d > vec3f(0.0)) - g) * cs * inv;
    tMax = select(tMax, vec3f(1e30), abs(d) <= vec3f(1e-8));
    for (var s = 0u; s < 96u; s++) {
        for (var k = 0u; k < 8u; k++) {
            let c = cell + vec3i(i32(k & 1u), i32((k >> 1u) & 1u), i32((k >> 2u) & 1u));
            if (any(c < vec3i(0)) || any(c >= dims)) { continue; }
            let ci = u32(c.x) + grid.dims.x * (u32(c.y) + grid.dims.y * u32(c.z));
            let first = cellOffsets[ci];
            let last = select(cellOffsets[ci + 1u], grid.count, ci + 1u >= total);
            for (var j = first; j < last; j++) {
                let index = sortedIndices[j];
                if (index == skip) { continue; }
                let center = sortedPositions[j].xyz;
                let r = particleRadius(index);
                let oc = o - center;
                let b = dot(oc, d);
                let h = b * b - (dot(oc, oc) - r * r);
                if (h < 0.0) { continue; }
                // a ray starting inside an overlapping sphere passes through it
                let t = -b - sqrt(h);
                if (t > 1e-4 && t < hit.t) {
                    hit = Hit(t, index, center);
                }
            }
        }
        let exit = min(tMax.x, min(tMax.y, tMax.z));
        if (hit.index != NO_HIT && hit.t <= exit) { break; }
        if (exit >= t1) { break; }
        if (tMax.x <= tMax.y && tMax.x <= tMax.z) {
            cell.x += stride.x;
            tMax.x += delta.x;
        } else if (tMax.y <= tMax.z) {
            cell.y += stride.y;
            tMax.y += delta.y;
        } else {
            cell.z += stride.z;
            tMax.z += delta.z;
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
    return scene.wallAlbedo * (panelIrradiance(scene, p, n) / PI + c.rgb + c.a * skyRad(scene, n));
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

// what a ray sees with one more bounce: mirrors and glass it meets reflect and refract once more
fn traceBounce(o: vec3f, d: vec3f, skip: u32) -> vec3f {
    let exit = roomExit(o, d);
    let hit = traceParticles(o, d, skip, exit.w);
    if (hit.index == NO_HIT) { return shadeRoom(o, d, exit); }
    let p = o + d * hit.t;
    let n = normalize(p - hit.center);
    let kind = particleKind(hit.index);
    if (kind == MIRROR) {
        return mirrorTint(dot(-d, n)) * traceLast(p, reflect(d, n), hit.index);
    }
    if (kind == GLASS) {
        let rays = glassRays(p, n, d, hit.center, particleRadius(hit.index));
        let f = fresnel(glassF0(), dot(-d, n));
        return f * traceLast(p, rays[0], hit.index) + (1.0 - f) * glassTint() * traceLast(rays[1], rays[2], hit.index);
    }
    return shadeMatte(hit.index, p, n, -d);
}

fn glassF0() -> f32 {
    let k = (scene.ior - 1.0) / (scene.ior + 1.0);
    return k * k;
}
fn glassTint() -> vec3f { return vec3f(0.94, 0.97, 0.96); }
// glossy black: a dark metal's reflection, brighter at grazing angles
fn mirrorTint(cosTheta: f32) -> vec3f { return vec3f(mix(0.55, 1.0, pow(1.0 - clamp(cosTheta, 0.0, 1.0), 5.0))); }

fn trace(o: vec3f, d: vec3f, skip: u32) -> vec3f {
    if (scene.rtBounces > 1.5) { return traceBounce(o, d, skip); }
    return traceLast(o, d, skip);
}

struct VIn {
    @location(0) position: vec4f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) center: vec4f,
};
struct VOut {
    @builtin(position) clip: vec4f,
    @location(0) viewPos: vec3f,
    @location(1) @interpolate(flat) sphere: vec4f,   // view-space centre (as drawn), radius
    @location(2) @interpolate(flat) world: vec3f,    // world centre (unmirrored)
    @location(3) @interpolate(flat) index: u32,
};

@vertex
fn vertex_main(v: VIn, @builtin(instance_index) index: u32) -> VOut {
    let r = particleRadius(index);
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

@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    // the eye ray against the sphere, in view space
    let ray = normalize(in.viewPos);
    let c = in.sphere.xyz;
    let r = in.sphere.w;
    let b = dot(ray, c);
    let h = b * b - (dot(c, c) - r * r);
    if (h < 0.0) { discard; }
    let pv = ray * (b - sqrt(h));
    let nv = (pv - c) / r;

    // to the world, on the room's side of the floor
    let right = vec3f(view_matrix[0][0], view_matrix[1][0], view_matrix[2][0]);
    let up = vec3f(view_matrix[0][1], view_matrix[1][1], view_matrix[2][1]);
    let back = vec3f(view_matrix[0][2], view_matrix[1][2], view_matrix[2][2]);
    var n = normalize(right * nv.x + up * nv.y + back * nv.z);
    var d = normalize(right * ray.x + up * ray.y + back * ray.z);
    if (particles.mirrored > 0.5) {
        n.y = -n.y;
        d.y = -d.y;
    }
    let i = in.index;
    let p = in.world + n * r;
    let cosV = dot(-d, n);

    var color: vec3f;
    let kind = particleKind(i);
    if (scene.view > 0.5) {
        let emission = lighting[2u * i + 1u].rgb;
        color = particleAlbedo(i, any(emission > vec3f(0.0))) * lighting[2u * i].rgb;
    } else if (kind == MIRROR) {
        color = mirrorTint(cosV) * trace(p, reflect(d, n), i);
    } else if (kind == GLASS) {
        let rays = glassRays(p, n, d, in.world, r);
        let f = fresnel(glassF0(), cosV);
        color = f * trace(p, rays[0], i) + (1.0 - f) * glassTint() * trace(rays[1], rays[2], i);
    } else {
        // matte, with a soft sheen of the panel (shadowed as its light is)
        color = shadeMatte(i, p, n, -d) + fresnel(0.04, cosV) * panelSeen(scene, p, reflect(d, n)) * lighting[2u * i].a;
    }
    if (particles.mirrored > 0.5) {
        color *= floorReflection(scene, p.y);
    }
    return vec4f(tonemap(scene, color), 1.0);
}
