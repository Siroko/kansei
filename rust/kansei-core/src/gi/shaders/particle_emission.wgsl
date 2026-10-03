// The light a particle emits (gi::ParticleEmission), shared by the splat (which puts it in the
// volume) and the cone shading (which hands it to the particle's material): a share of the
// particles, picked by a hash of their index (stable: Kansei's fluids keep particle order), glows
// with `color`, and every particle adds `speedColor` per unit of speed.
struct ParticleEmission {
    color      : vec3f,   // scene radiance of an emissive particle
    share      : f32,     // the share of particles that glow, 0..1
    speedColor : vec3f,   // scene radiance per unit of speed
    seed       : u32,
}

fn giHash(x: u32) -> u32 {
    // PCG (Jarzynski & Olano 2020)
    let state = x * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn giHash01(x: u32) -> f32 {
    return f32(giHash(x) >> 8u) / 16777216.0;
}

fn particleEmission(e: ParticleEmission, index: u32, velocity: vec3f) -> vec3f {
    let glows = giHash01(index ^ e.seed) < e.share;
    return select(vec3f(0.0), e.color, glows) + e.speedColor * length(velocity);
}
