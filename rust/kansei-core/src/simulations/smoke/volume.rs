//! A smoke volume: its box, its grid, its motion level and its look, its sources and its fill.

use bytemuck::{Pod, Zeroable};

use super::{MAX_SMOKE_RESOLUTION, MAX_SMOKE_SOURCES, MIN_SMOKE_RESOLUTION};
use crate::math::Vec3;

fn g(v: Vec3) -> glam::Vec3 {
    glam::Vec3::new(v.x, v.y, v.z)
}

fn k(v: glam::Vec3) -> Vec3 {
    Vec3::new(v.x, v.y, v.z)
}

/// The Waves level: a plume from each source swayed by sums of sines over height and time, or
/// hanging layers rippled the same way, evaluated where the march samples it (no state).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavesSettings {
    /// How far the plume's axis sways, metres.
    pub amplitude: f32,
    /// The sway's wavelength up the plume, metres.
    pub wavelength: f32,
    /// How fast the sway's phase turns, radians per second.
    pub speed: f32,
    /// How much the plume widens per metre it rises.
    pub spread: f32,
    /// Seconds a mouse push takes to relax back (to about a third).
    pub relax_time: f32,
}

impl Default for WavesSettings {
    fn default() -> Self {
        Self { amplitude: 0.03, wavelength: 0.25, speed: 1.5, spread: 0.25, relax_time: 1.5 }
    }
}

/// The Curl noise level: the gas advected by curl noise (divergence-free) and a rise.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CurlSettings {
    /// The noise velocity's scale, m/s.
    pub strength: f32,
    /// The noise's feature size, metres.
    pub scale: f32,
    /// How fast the noise changes, per second.
    pub speed: f32,
    /// A steady rise, m/s.
    pub rise: f32,
}

impl Default for CurlSettings {
    fn default() -> Self {
        Self { strength: 0.15, scale: 0.15, speed: 0.3, rise: 0.25 }
    }
}

/// The Fluid level: a grid smoke solver (Stam 1999; Fedkiw, Stam and Jensen 2001).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidSmokeSettings {
    /// Upward acceleration per unit of temperature, m/s².
    pub buoyancy: f32,
    /// Downward acceleration per unit of density, m/s².
    pub weight: f32,
    /// Temperature lost per second (exponential rate).
    pub cooling: f32,
    /// Vorticity confinement strength (Fedkiw 2001's epsilon), scaled by the cell size.
    pub vorticity: f32,
    /// Jacobi iterations of the pressure projection; 0 skips it.
    pub pressure_iterations: u32,
    /// The speed no cell exceeds, m/s.
    pub max_speed: f32,
    /// A weak, large-scale ambient draft: a slow curl-noise force, m/s².
    pub ambient_draft: f32,
    /// The draft's feature size, metres.
    pub draft_scale: f32,
}

impl Default for FluidSmokeSettings {
    fn default() -> Self {
        Self { buoyancy: 1.5, weight: 0.1, cooling: 0.8, vorticity: 2.0, pressure_iterations: 40, max_speed: 3.0, ambient_draft: 0.0, draft_scale: 1.0 }
    }
}

/// Where a volume's motion comes from (see the module's docs).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SmokeMotion {
    Waves(WavesSettings),
    CurlNoise(CurlSettings),
    Fluid(FluidSmokeSettings),
}

impl SmokeMotion {
    pub fn waves() -> Self {
        Self::Waves(WavesSettings::default())
    }

    pub fn curl_noise() -> Self {
        Self::CurlNoise(CurlSettings::default())
    }

    pub fn fluid() -> Self {
        Self::Fluid(FluidSmokeSettings::default())
    }

    pub(crate) fn kind(&self) -> u32 {
        match self {
            Self::Waves(_) => super::MOTION_WAVES,
            Self::CurlNoise(_) => super::MOTION_CURL,
            Self::Fluid(_) => super::MOTION_FLUID,
        }
    }

    /// Whether the level keeps a density grid (Curl noise and Fluid).
    pub(crate) fn has_grid(&self) -> bool {
        !matches!(self, Self::Waves(_))
    }
}

/// How a volume's gas looks.
#[derive(Debug, Clone, Copy)]
pub struct SmokeLook {
    /// Scales the density into scattering and extinction per metre.
    pub density_scale: f32,
    /// Scattering albedo: scattering = density * density_scale * albedo.
    pub albedo: Vec3,
    /// Extinction = density * density_scale * extinction.
    pub extinction: f32,
    /// Density lost per second (exponential rate); on Waves, per metre the plume rises.
    pub dissipation: f32,
    /// The fraction of the box over which every face fades the density, so no outline shows.
    pub edge_fade: f32,
}

impl Default for SmokeLook {
    fn default() -> Self {
        Self { density_scale: 1.0, albedo: Vec3::new(0.95, 0.95, 0.95), extinction: 1.0, dissipation: 0.6, edge_fade: 0.1 }
    }
}

/// A source of gas: steam off a cup.
#[derive(Debug, Clone, Copy)]
pub struct SmokeSource {
    pub position: Vec3,
    /// Metres.
    pub radius: f32,
    /// Density added per second at the centre.
    pub rate: f32,
    /// The temperature the gas leaves it at (buoyancy on Fluid).
    pub temperature: f32,
    /// The velocity the gas leaves it with, m/s.
    pub velocity: Vec3,
}

impl Default for SmokeSource {
    fn default() -> Self {
        Self { position: Vec3::ZERO, radius: 0.03, rate: 4.0, temperature: 1.0, velocity: Vec3::new(0.0, 0.15, 0.0) }
    }
}

/// What a volume holds when it starts (and on `SmokeSimulation::reset`).
#[derive(Debug, Clone, Copy)]
pub enum SmokeFill {
    None,
    /// The whole box.
    Box { density: f32, temperature: f32 },
    /// A ball, `center` in world space, `radius` in metres.
    Sphere { center: Vec3, radius: f32, density: f32, temperature: f32 },
    /// `count` soft horizontal sheets spread over the box's height, each about `thickness` of the
    /// height between them, gently uneven in x and z: smoke left hanging in a room.
    Layers { count: u32, thickness: f32, density: f32 },
}

/// A box of gas in world space, with its grid, motion level, look, sources and fill.
#[derive(Debug, Clone)]
pub struct SmokeVolume {
    pub center: Vec3,
    /// Full extents, metres, along the box's own axes.
    pub size: Vec3,
    /// Rotation about +Y, radians (as `LocalFogVolume::yaw`).
    pub yaw: f32,
    /// Cells along the box's longest side (clamped to 16..=160); the others in proportion.
    pub resolution: u32,
    pub motion: SmokeMotion,
    pub look: SmokeLook,
    /// At most `MAX_SMOKE_SOURCES` are used.
    pub sources: Vec<SmokeSource>,
    pub fill: SmokeFill,
}

impl SmokeVolume {
    /// A box with resolution 96, the default look, no sources and no fill.
    pub fn new(center: Vec3, size: Vec3, motion: SmokeMotion) -> Self {
        Self { center, size, yaw: 0.0, resolution: 96, motion, look: SmokeLook::default(), sources: Vec::new(), fill: SmokeFill::None }
    }

    /// A plume `height` tall and half as wide and deep, its source at the bottom centre, two cells
    /// above the floor.
    pub fn rising_from(source: SmokeSource, height: f32, motion: SmokeMotion) -> Self {
        let mut v = Self::new(Vec3::ZERO, Vec3::new(height * 0.5, height, height * 0.5), motion);
        let cell = v.cell_size();
        v.center = source.position + Vec3::new(0.0, height * 0.5 - 2.0 * cell, 0.0);
        v.sources.push(source);
        v
    }

    /// The grid's cells along x, y and z: the longest side gets `resolution` (clamped), the others
    /// in proportion, at least 8.
    pub fn grid_dims(&self) -> [u32; 3] {
        let res = self.resolution.clamp(MIN_SMOKE_RESOLUTION, MAX_SMOKE_RESOLUTION);
        let longest = self.size.x.max(self.size.y).max(self.size.z).max(1e-6);
        let d = |s: f32| ((s.max(0.0) / longest * res as f32).round() as u32).clamp(8, res);
        [d(self.size.x), d(self.size.y), d(self.size.z)]
    }

    /// The edge of a cell along the longest side, metres.
    pub fn cell_size(&self) -> f32 {
        let res = self.resolution.clamp(MIN_SMOKE_RESOLUTION, MAX_SMOKE_RESOLUTION);
        self.size.x.max(self.size.y).max(self.size.z).max(1e-6) / res as f32
    }

    fn yaw_sin_cos(&self) -> (f32, f32) {
        self.yaw.sin_cos()
    }

    /// A world point in the box's unit cube ([0, 1]³ inside).
    pub fn to_local(&self, world: Vec3) -> Vec3 {
        let (s, c) = self.yaw_sin_cos();
        let d = g(world) - g(self.center);
        let q = glam::Vec3::new(c * d.x - s * d.z, d.y, s * d.x + c * d.z);
        k(q / g(self.size).max(glam::Vec3::splat(1e-6)) + 0.5)
    }

    /// A point of the box's unit cube in world space.
    pub fn to_world(&self, local: Vec3) -> Vec3 {
        let (s, c) = self.yaw_sin_cos();
        let q = (g(local) - 0.5) * g(self.size);
        k(g(self.center) + glam::Vec3::new(c * q.x + s * q.z, q.y, -s * q.x + c * q.z))
    }

    /// The part `[t0, t1]` of the ray `origin + t dir` inside the box, `t0` clamped to 0 (the
    /// origin inside); `None` if the ray misses it or it lies behind.
    pub fn ray_interval(&self, origin: Vec3, dir: Vec3) -> Option<(f32, f32)> {
        let (s, c) = self.yaw_sin_cos();
        let rot = |v: glam::Vec3| glam::Vec3::new(c * v.x - s * v.z, v.y, s * v.x + c * v.z);
        let o = rot(g(origin) - g(self.center));
        let d = rot(g(dir));
        let half = g(self.size) * 0.5;
        let (mut t0, mut t1) = (0.0f32, f32::INFINITY);
        for i in 0..3 {
            if d[i].abs() < 1e-12 {
                if o[i].abs() > half[i] {
                    return None;
                }
                continue;
            }
            let (a, b) = ((-half[i] - o[i]) / d[i], (half[i] - o[i]) / d[i]);
            t0 = t0.max(a.min(b));
            t1 = t1.min(a.max(b));
        }
        (t1 > t0).then_some((t0, t1))
    }

    pub(crate) fn gpu(&self, dims: [u32; 3]) -> SmokeVolumeGpu {
        let (s, c) = self.yaw_sin_cos();
        let mut sources = [SmokeSourceGpu::zeroed(); MAX_SMOKE_SOURCES];
        for (dst, src) in sources.iter_mut().zip(&self.sources) {
            *dst = SmokeSourceGpu {
                position: [src.position.x, src.position.y, src.position.z],
                radius: src.radius.max(1e-4),
                velocity: [src.velocity.x, src.velocity.y, src.velocity.z],
                rate: src.rate.max(0.0),
            };
        }
        let (waves, curl) = match self.motion {
            SmokeMotion::Waves(w) => (w, CurlSettings::default()),
            SmokeMotion::CurlNoise(c) => (WavesSettings::default(), c),
            SmokeMotion::Fluid(_) => (WavesSettings::default(), CurlSettings::default()),
        };
        let (layer_count, layer_thickness, layer_density) = match self.fill {
            SmokeFill::Layers { count, thickness, density } => (count, thickness, density),
            _ => (0, 0.0, 0.0),
        };
        let l = &self.look;
        SmokeVolumeGpu {
            center: [self.center.x, self.center.y, self.center.z],
            cos_yaw: c,
            size: [self.size.x.max(1e-4), self.size.y.max(1e-4), self.size.z.max(1e-4)],
            sin_yaw: s,
            dims,
            motion: self.motion.kind(),
            albedo: [l.albedo.x, l.albedo.y, l.albedo.z],
            density_scale: l.density_scale.max(0.0),
            extinction: l.extinction.max(0.0),
            dissipation: l.dissipation.max(0.0),
            edge_fade: l.edge_fade.clamp(0.0, 0.5),
            num_sources: self.sources.len().min(MAX_SMOKE_SOURCES) as u32,
            waves_amplitude: waves.amplitude,
            waves_wavelength: waves.wavelength.max(1e-3),
            waves_speed: waves.speed,
            waves_spread: waves.spread.max(0.0),
            waves_relax_time: waves.relax_time.max(1e-3),
            curl_strength: curl.strength,
            curl_scale: curl.scale.max(1e-3),
            curl_speed: curl.speed,
            curl_rise: curl.rise,
            layer_count,
            layer_thickness,
            layer_density,
            sources,
        }
    }
}

// ── GPU layouts (must match smoke_common.wgsl) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(crate) struct SmokeSourceGpu {
    pub(crate) position: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) velocity: [f32; 3],
    pub(crate) rate: f32,
}

/// WGSL `SmokeVolumeData`: what the march and the solver read of a volume.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq)]
pub(crate) struct SmokeVolumeGpu {
    pub(crate) center: [f32; 3],
    pub(crate) cos_yaw: f32,
    pub(crate) size: [f32; 3],
    pub(crate) sin_yaw: f32,
    pub(crate) dims: [u32; 3],
    pub(crate) motion: u32,
    pub(crate) albedo: [f32; 3],
    pub(crate) density_scale: f32,
    pub(crate) extinction: f32,
    pub(crate) dissipation: f32,
    pub(crate) edge_fade: f32,
    pub(crate) num_sources: u32,
    pub(crate) waves_amplitude: f32,
    pub(crate) waves_wavelength: f32,
    pub(crate) waves_speed: f32,
    pub(crate) waves_spread: f32,
    pub(crate) waves_relax_time: f32,
    pub(crate) curl_strength: f32,
    pub(crate) curl_scale: f32,
    pub(crate) curl_speed: f32,
    pub(crate) curl_rise: f32,
    pub(crate) layer_count: u32,
    pub(crate) layer_thickness: f32,
    pub(crate) layer_density: f32,
    pub(crate) sources: [SmokeSourceGpu; MAX_SMOKE_SOURCES],
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::FRAC_PI_2;

    fn assert_close(a: Vec3, b: Vec3) {
        assert!((g(a) - g(b)).abs().max_element() < 1e-5, "{a:?} vs {b:?}");
    }

    fn assert_close2(a: (f32, f32), b: (f32, f32)) {
        assert!((a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5, "{a:?} vs {b:?}");
    }

    #[test]
    fn the_box_maps_points_and_rays_with_its_yaw() {
        let v = SmokeVolume { yaw: FRAC_PI_2, ..SmokeVolume::new(Vec3::new(1.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.5), SmokeMotion::fluid()) };
        assert_close(v.to_local(v.center), Vec3::new(0.5, 0.5, 0.5));
        assert_close(v.to_world(v.to_local(Vec3::new(1.2, 1.3, 0.4))), Vec3::new(1.2, 1.3, 0.4));
        // the long axis now along world z
        let l = v.to_local(Vec3::new(1.0, 1.0, 0.9));
        assert!(l.x.max(l.z) <= 1.0 && l.x.min(l.z) >= 0.0, "{l:?}");
        let (a, b) = v.ray_interval(Vec3::new(1.0, 1.0, 5.0), Vec3::new(0.0, 0.0, -1.0)).unwrap();
        assert_close2((a, b), (4.0, 6.0));
        assert_eq!(v.ray_interval(v.center, Vec3::new(0.0, 1.0, 0.0)).unwrap().0, 0.0);
        assert!(v.ray_interval(Vec3::new(9.0, 9.0, 9.0), Vec3::new(1.0, 0.0, 0.0)).is_none());
    }

    #[test]
    fn grid_dims_follow_the_box_and_clamp_the_resolution() {
        assert_eq!(SmokeVolume::new(Vec3::ZERO, Vec3::new(0.4, 0.8, 0.4), SmokeMotion::fluid()).grid_dims(), [48, 96, 48]);
        let mut c = SmokeVolume::new(Vec3::ZERO, Vec3::new(1.0, 1.0, 1.0), SmokeMotion::fluid());
        c.resolution = 4096;
        assert_eq!(c.grid_dims(), [160; 3]);
        c.resolution = 1;
        assert_eq!(c.grid_dims(), [16; 3]);
    }

    #[test]
    fn a_plume_stands_on_its_source() {
        let source = SmokeSource { position: Vec3::new(0.2, 0.9, -0.1), ..Default::default() };
        let v = SmokeVolume::rising_from(source, 0.8, SmokeMotion::fluid());
        assert_eq!(v.grid_dims(), [48, 96, 48]);
        let local = v.to_local(source.position);
        assert_close(local, Vec3::new(0.5, 2.0 / 96.0, 0.5));
    }
}
