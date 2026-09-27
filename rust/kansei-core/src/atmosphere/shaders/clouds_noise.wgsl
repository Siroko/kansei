// Volumetric clouds, noise: the tileable textures the clouds are carved from, generated once
// (after Schneider 2015, "The Real-time Volumetric Cloudscapes of Horizon Zero Dawn").
// - shape (3D): r a Perlin-Worley noise, g b a Worley fBm at rising frequencies, which erode it
//   into billows;
// - detail (3D, finer): r g b Worley fBm, which frays the edges;
// - weather (2D): r where clouds form (a Perlin-Worley fBm), g how tall they grow.
// Every noise is periodic over its texture, so the textures tile seamlessly.

@group(0) @binding(0) var shapeOut   : texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(1) var detailOut  : texture_storage_3d<rgba8unorm, write>;
@group(0) @binding(2) var weatherOut : texture_storage_2d<rgba8unorm, write>;

fn hash3(p: vec3u) -> vec3f {
    // PCG-style integer hash (Jarzynski & Olano 2020)
    var v = p * 1664525u + 1013904223u;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> vec3u(16u);
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return vec3f(v & vec3u(0xffffffu)) / f32(0xffffff);
}

fn wrap(c: vec3i, period: i32) -> vec3u {
    return vec3u(((c % period) + period) % period);
}

// Worley (cellular) noise with `period` cells per unit, 1 at the feature points, falling to 0.
fn worley(p: vec3f, period: i32, seed: u32) -> f32 {
    let q = p * f32(period);
    let cell = vec3i(floor(q));
    let f = q - floor(q);
    var best = 1.0;
    for (var z = -1; z <= 1; z++) {
        for (var y = -1; y <= 1; y++) {
            for (var x = -1; x <= 1; x++) {
                let o = vec3i(x, y, z);
                let feature = hash3(wrap(cell + o, period) + vec3u(seed * 131u)) + vec3f(o);
                let d = feature - f;
                best = min(best, dot(d, d));
            }
        }
    }
    return 1.0 - saturate(sqrt(best));
}

// Three octaves of Worley, stretched over [0, 1] (the raw sum spans about 0.26 to 0.70).
fn worleyFbm(p: vec3f, period: i32, seed: u32) -> f32 {
    let w = worley(p, period, seed) * 0.625 + worley(p, period * 2, seed + 1u) * 0.25 + worley(p, period * 4, seed + 2u) * 0.125;
    return saturate((w - 0.26) / 0.44);
}

fn fade(t: vec3f) -> vec3f {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn gradient(c: vec3i, period: i32) -> vec3f {
    return normalize(hash3(wrap(c, period) + vec3u(977u)) * 2.0 - 1.0);
}

// Periodic gradient (Perlin) noise, about [-1, 1].
fn perlin(p: vec3f, period: i32) -> f32 {
    let q = p * f32(period);
    let c = vec3i(floor(q));
    let f = q - floor(q);
    let u = fade(f);
    var n = array<f32, 8>();
    for (var i = 0; i < 8; i++) {
        let o = vec3i(i & 1, (i >> 1) & 1, i >> 2);
        n[i] = dot(gradient(c + o, period), f - vec3f(o));
    }
    let x0 = mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y);
    let x1 = mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y);
    return mix(x0, x1, u.z);
}

fn perlinFbm(p: vec3f, period: i32, octaves: i32) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var per = period;
    for (var i = 0; i < octaves; i++) {
        sum += perlin(p, per) * amp;
        amp *= 0.5;
        per *= 2;
    }
    return sum;
}

fn remap(v: f32, lo: f32, hi: f32, newLo: f32, newHi: f32) -> f32 {
    return newLo + (v - lo) / (hi - lo) * (newHi - newLo);
}

// Four octaves of Perlin over [0, 1] (the raw sum spans about -0.19 to 0.19).
fn perlin01(p: vec3f, period: i32) -> f32 {
    return saturate(perlinFbm(p, period, 4) * 2.4 + 0.5);
}

// Perlin-Worley: Perlin dilated by Worley cells, so it billows
fn perlinWorley(p: vec3f, period: i32) -> f32 {
    let pn = perlin01(p, period);
    let w = worleyFbm(p, period, 0u);
    return saturate(remap(pn, w - 1.0, 1.0, 0.0, 1.0));
}

@compute @workgroup_size(4, 4, 4)
fn shape(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(shapeOut);
    if (any(gid >= size)) { return; }
    let p = (vec3f(gid) + 0.5) / vec3f(size);
    textureStore(shapeOut, gid, vec4f(perlinWorley(p, 4), worleyFbm(p, 4, 3u), worleyFbm(p, 8, 6u), worleyFbm(p, 16, 9u)));
}

@compute @workgroup_size(4, 4, 4)
fn detail(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(detailOut);
    if (any(gid >= size)) { return; }
    let p = (vec3f(gid) + 0.5) / vec3f(size);
    textureStore(detailOut, gid, vec4f(worleyFbm(p, 2, 12u), worleyFbm(p, 4, 15u), worleyFbm(p, 8, 18u), 1.0));
}

@compute @workgroup_size(8, 8)
fn weather(@builtin(global_invocation_id) gid : vec3u) {
    let size = textureDimensions(weatherOut);
    if (any(gid.xy >= size)) { return; }
    // a thin slab of the 3D noises, so the 2D map tiles too
    let p = vec3f((vec2f(gid.xy) + 0.5) / vec2f(size), 0.5);
    let coverage = perlinWorley(p, 4);
    let height = perlin01(p + vec3f(0.37, 0.11, 0.0), 3);
    textureStore(weatherOut, gid.xy, vec4f(coverage, height, 0.0, 1.0));
}
