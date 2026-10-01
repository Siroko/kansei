//! Adding fluid at runtime: a [`FluidNozzle`] turns a stream (a hose, a spout, a cannon's
//! muzzle) into particles, which [`FluidSimulation::emit`] appends into the simulation's spare
//! capacity ([`FluidSimulation::with_capacity`]).
//!
//! The nozzle lays the stream down in layers across its disc, a particle spacing apart both
//! across and along the stream, so the new water starts at about the density the rest of the
//! fluid is at: no burst from packed particles, no gaps. Each layer is placed as far along the
//! stream as it has travelled since it left within the step, and turned and jittered a little so
//! the stream does not read as a lattice.

use super::simulation::FluidSimulation;

/// A round nozzle: a stream of particles out of a disc, along its axis.
#[derive(Debug, Clone)]
pub struct FluidNozzle {
    /// The disc's centre and the stream's direction (normalised), in the simulation's space.
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    /// The disc's radius.
    pub radius: f32,
    /// How fast the stream leaves (the simulation's units per simulated second).
    pub speed: f32,
    /// The particles' spacing across and along the stream: the fluid's rest spacing.
    pub spacing: f32,
    /// Random offsets of each particle, as a fraction of the spacing.
    pub jitter: f32,
    /// Random deviation of each particle's velocity, as a fraction of `speed`: the stream
    /// spreading as it flies.
    pub spread: f32,
    /// How far the stream has run since its last layer.
    travelled: f32,
    seed: u32,
}

impl FluidNozzle {
    pub fn new(origin: [f32; 3], direction: [f32; 3], radius: f32, speed: f32, spacing: f32) -> Self {
        let mut nozzle = Self { origin, direction: [0.0, 1.0, 0.0], radius, speed, spacing, jitter: 0.1, spread: 0.02, travelled: 0.0, seed: 0x9e37_79b9 };
        nozzle.aim(origin, direction);
        nozzle
    }

    /// Move and turn it (`direction` need not be normalised).
    pub fn aim(&mut self, origin: [f32; 3], direction: [f32; 3]) {
        let d = glam::Vec3::from(direction).normalize_or(glam::Vec3::Y);
        self.origin = origin;
        self.direction = d.to_array();
    }

    /// The particles in one layer across the disc (before any is cut by the capacity).
    pub fn layer_size(&self) -> usize {
        self.disc(0.0).len()
    }

    /// Particles per simulated second at full flow.
    pub fn rate(&self) -> f32 {
        self.layer_size() as f32 * self.speed / self.spacing
    }

    /// Restart the stream: its next layer leaves at once.
    pub fn restart(&mut self) {
        self.travelled = self.spacing;
    }

    /// The disc's points, a spacing apart on a hexagonal pattern turned by `angle`, relative to
    /// the centre and across the axis.
    fn disc(&self, angle: f32) -> Vec<glam::Vec3> {
        let d = glam::Vec3::from(self.direction);
        let u = d.any_orthonormal_vector();
        let v = d.cross(u);
        let (s, c) = angle.sin_cos();
        let (u, v) = (u * c + v * s, v * c - u * s);
        let h = self.spacing * 0.866_025_4;
        let rows = (self.radius / h).floor() as i32;
        let cols = (self.radius / self.spacing).ceil() as i32 + 1;
        let mut points = Vec::new();
        for j in -rows..=rows {
            let shift = if j.rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            for i in -cols..=cols {
                let (x, y) = ((i as f32 + shift) * self.spacing, j as f32 * h);
                if x * x + y * y <= self.radius * self.radius + 1e-6 {
                    points.push(u * x + v * y);
                }
            }
        }
        points
    }

    fn random(&mut self) -> f32 {
        // xorshift32, in [-1, 1)
        let mut x = self.seed;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.seed = x;
        (x >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }

    /// The particles for `dt` simulated seconds of flow: positions and velocities, layer by
    /// layer, the first to leave the farthest along.
    pub fn flow(&mut self, dt: f32) -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
        let (mut positions, mut velocities) = (Vec::new(), Vec::new());
        let d = glam::Vec3::from(self.direction);
        let origin = glam::Vec3::from(self.origin);
        self.travelled += self.speed * dt.max(0.0);
        while self.travelled >= self.spacing {
            self.travelled -= self.spacing;
            let along = self.travelled;
            let angle = self.random() * std::f32::consts::PI;
            for p in self.disc(angle) {
                let jitter = glam::Vec3::new(self.random(), self.random(), self.random()) * self.jitter * self.spacing;
                let deviation = glam::Vec3::new(self.random(), self.random(), self.random()) * self.spread * self.speed;
                positions.push((origin + p + d * along + jitter).to_array());
                velocities.push((d * self.speed + deviation).to_array());
            }
        }
        (positions, velocities)
    }

    /// [`flow`](Self::flow) for `dt` simulated seconds into `sim`, as much as its spare capacity
    /// takes. Returns how many particles were added.
    pub fn emit_into(&mut self, sim: &mut FluidSimulation, dt: f32) -> u32 {
        let (positions, velocities) = self.flow(dt);
        sim.emit(&positions, &velocities)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_layer_fills_the_disc_a_spacing_apart() {
        let nozzle = FluidNozzle::new([0.0; 3], [1.0, 1.0, 0.0], 1.2, 10.0, 0.5);
        let disc = nozzle.disc(0.3);
        // about the disc's area over a hexagonal cell's
        let expected = std::f32::consts::PI * 1.2 * 1.2 / (0.5 * 0.5 * 0.866);
        assert!((disc.len() as f32 - expected).abs() < expected * 0.35, "{} vs {expected}", disc.len());
        let axis = glam::Vec3::new(1.0, 1.0, 0.0).normalize();
        for (k, p) in disc.iter().enumerate() {
            assert!(p.dot(axis).abs() < 1e-4, "off the disc: {p}");
            assert!(p.length() <= 1.2 + 1e-4);
            for q in &disc[k + 1..] {
                assert!(p.distance(*q) > 0.5 - 1e-3, "closer than a spacing");
            }
        }
    }

    #[test]
    fn the_flow_lays_a_layer_per_spacing_travelled() {
        let mut nozzle = FluidNozzle::new([1.0, 2.0, 3.0], [0.0, 0.0, 2.0], 0.6, 4.0, 0.5);
        let layer = nozzle.layer_size();
        assert!(layer >= 4);
        // 4 units/s for 0.3 s: 1.2 units, two layers (0.2 left over)
        let (p, v) = nozzle.flow(0.3);
        assert_eq!(p.len(), 2 * layer);
        assert_eq!(v.len(), p.len());
        // then 0.05 s more: 0.4 travelled, none; then 0.03: a third
        assert_eq!(nozzle.flow(0.05).0.len(), 0);
        assert_eq!(nozzle.flow(0.03).0.len(), layer);
        // a second's flow is the rate
        let mut nozzle = FluidNozzle::new([0.0; 3], [0.0, 0.0, 1.0], 0.6, 4.0, 0.5);
        let n: usize = (0..100).map(|_| nozzle.flow(0.01).0.len()).sum();
        assert!((n as f32 - nozzle.rate()).abs() <= layer as f32, "{n} vs {}", nozzle.rate());
    }

    #[test]
    fn the_first_layer_is_farthest_along_and_the_velocities_follow_the_axis() {
        let mut nozzle = FluidNozzle::new([0.0; 3], [0.0, 0.0, 1.0], 0.6, 4.0, 0.5);
        nozzle.jitter = 0.0;
        let layer = nozzle.layer_size();
        let (p, v) = nozzle.flow(0.3);
        let z = |k: usize| p[k * layer][2];
        assert!((z(0) - 0.7).abs() < 1e-4 && (z(1) - 0.2).abs() < 1e-4, "{} {}", z(0), z(1));
        for v in v {
            let v = glam::Vec3::from(v);
            assert!(v.z > 3.8 && v.length() < 4.0 * 1.06, "{v}");
        }
    }
}
