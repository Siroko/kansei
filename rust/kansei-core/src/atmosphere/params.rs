use bytemuck::{Pod, Zeroable};

use crate::math::Vec3;

/// The planet's atmosphere. The fields and their defaults follow Unreal's `SkyAtmosphereComponent`
/// (an Earth-like atmosphere from Bruneton 2017 and Hillaire 2020), so a scene exported from
/// Unreal can copy its values across: each coefficient is a colour times a scale, per kilometre.
#[derive(Debug, Clone, Copy)]
pub struct AtmosphereParams {
    /// Radius of the planet's surface, km.
    pub bottom_radius_km: f32,
    /// Thickness of the atmosphere above the surface, km.
    pub atmosphere_height_km: f32,
    /// Diffuse reflectance of the planet's surface; it bounces light back into the sky.
    pub ground_albedo: Vec3,
    /// Air molecules: scattering per km at sea level is `rayleigh_scattering * rayleigh_scattering_scale`.
    pub rayleigh_scattering_scale: f32,
    pub rayleigh_scattering: Vec3,
    /// Altitude at which Rayleigh density falls to 1/e, km.
    pub rayleigh_exponential_distribution_km: f32,
    /// Aerosols (haze): scattering and absorption per km at sea level.
    pub mie_scattering_scale: f32,
    pub mie_scattering: Vec3,
    pub mie_absorption_scale: f32,
    pub mie_absorption: Vec3,
    /// Cornette-Shanks g: 0 isotropic, toward 1 a tight halo around the sun.
    pub mie_anisotropy: f32,
    /// Altitude at which aerosol density falls to 1/e, km.
    pub mie_exponential_distribution_km: f32,
    /// An absorbing layer (ozone): absorption per km at the tent's tip.
    pub other_absorption_scale: f32,
    pub other_absorption: Vec3,
    /// The absorbing layer's density is a tent: `tip_value` at `tip_altitude`, falling linearly
    /// to zero `width` km above and below it.
    pub other_tent_tip_altitude_km: f32,
    pub other_tent_tip_value: f32,
    pub other_tent_width_km: f32,
    /// Scales the multiple-scattering contribution (1 is physical).
    pub multi_scattering_factor: f32,
    /// Artistic tint of the sky's luminance (not of the aerial perspective); 1 is physical.
    pub sky_luminance_factor: Vec3,
    /// Aerial perspective as if the scene were this many times farther away; 1 is physical.
    pub aerial_perspective_view_distance_scale: f32,
    /// Distance from the camera before which there is no aerial perspective, km.
    pub aerial_perspective_start_depth_km: f32,
}

impl Default for AtmosphereParams {
    fn default() -> Self {
        Self::earth()
    }
}

impl AtmosphereParams {
    /// Earth, with Unreal's `SkyAtmosphereComponent` defaults.
    pub fn earth() -> Self {
        Self {
            bottom_radius_km: 6360.0,
            atmosphere_height_km: 100.0,
            ground_albedo: Vec3::new(0.402, 0.402, 0.402),
            rayleigh_scattering_scale: 0.0331,
            rayleigh_scattering: Vec3::new(0.175287, 0.409607, 1.0),
            rayleigh_exponential_distribution_km: 8.0,
            mie_scattering_scale: 0.003996,
            mie_scattering: Vec3::new(1.0, 1.0, 1.0),
            mie_absorption_scale: 0.000444,
            mie_absorption: Vec3::new(1.0, 1.0, 1.0),
            mie_anisotropy: 0.8,
            mie_exponential_distribution_km: 1.2,
            other_absorption_scale: 0.001881,
            other_absorption: Vec3::new(0.345561, 1.0, 0.045188),
            other_tent_tip_altitude_km: 25.0,
            other_tent_tip_value: 1.0,
            other_tent_width_km: 15.0,
            multi_scattering_factor: 1.0,
            sky_luminance_factor: Vec3::new(1.0, 1.0, 1.0),
            aerial_perspective_view_distance_scale: 1.0,
            aerial_perspective_start_depth_km: 0.1,
        }
    }

    pub fn top_radius_km(&self) -> f32 {
        self.bottom_radius_km + self.atmosphere_height_km.max(1e-3)
    }

    pub(crate) fn gpu_layout(&self) -> AtmosphereGpu {
        let v = |c: Vec3, s: f32| [c.x * s, c.y * s, c.z * s];
        let rayleigh = v(self.rayleigh_scattering, self.rayleigh_scattering_scale);
        let mie_scattering = v(self.mie_scattering, self.mie_scattering_scale);
        let mie_absorption = v(self.mie_absorption, self.mie_absorption_scale);
        AtmosphereGpu {
            bottom_radius: self.bottom_radius_km,
            top_radius: self.top_radius_km(),
            rayleigh_exp_scale: -1.0 / self.rayleigh_exponential_distribution_km.max(1e-3),
            mie_exp_scale: -1.0 / self.mie_exponential_distribution_km.max(1e-3),
            rayleigh_scattering: rayleigh,
            mie_g: self.mie_anisotropy.clamp(-0.999, 0.999),
            mie_scattering,
            absorption_tip_altitude: self.other_tent_tip_altitude_km,
            mie_extinction: [
                mie_scattering[0] + mie_absorption[0],
                mie_scattering[1] + mie_absorption[1],
                mie_scattering[2] + mie_absorption[2],
            ],
            absorption_tip_value: self.other_tent_tip_value,
            absorption_extinction: v(self.other_absorption, self.other_absorption_scale),
            absorption_width: self.other_tent_width_km.max(1e-3),
            ground_albedo: [self.ground_albedo.x, self.ground_albedo.y, self.ground_albedo.z],
            multi_scattering_factor: self.multi_scattering_factor,
        }
    }

    /// Extinction per km at `altitude_km`, as the shaders' `sampleMedium`.
    pub fn extinction_at(&self, altitude_km: f32) -> glam::Vec3 {
        let g = self.gpu_layout();
        let h = altitude_km.max(0.0);
        let rayleigh = (g.rayleigh_exp_scale * h).exp();
        let mie = (g.mie_exp_scale * h).exp();
        let absorption = (g.absorption_tip_value - (h - g.absorption_tip_altitude).abs() / g.absorption_width).max(0.0);
        glam::Vec3::from(g.rayleigh_scattering) * rayleigh
            + glam::Vec3::from(g.mie_extinction) * mie
            + glam::Vec3::from(g.absorption_extinction) * absorption
    }

    /// Transmittance from `altitude_km` above the surface to space along a ray whose cosine with
    /// the local zenith is `cos_zenith`; zero if the ray hits the planet. The CPU twin of the
    /// transmittance LUT, for lighting that has to agree with the sky (the sun's colour at the
    /// ground, say).
    pub fn transmittance_to_space(&self, altitude_km: f32, cos_zenith: f32) -> glam::Vec3 {
        let (bottom, top) = (self.bottom_radius_km as f64, self.top_radius_km() as f64);
        let r = bottom + altitude_km.clamp(0.0, self.atmosphere_height_km) as f64;
        let mu = cos_zenith.clamp(-1.0, 1.0) as f64;
        let horizon = -((r - bottom) * (r + bottom)).max(0.0).sqrt() / r;
        if mu < horizon {
            return glam::Vec3::ZERO;
        }
        let d = -r * mu + ((top - r) * (top + r) + r * r * mu * mu).max(0.0).sqrt();
        const STEPS: usize = 256;
        let dt = d / STEPS as f64;
        let mut depth = glam::DVec3::ZERO;
        for i in 0..STEPS {
            let t = (i as f64 + 0.5) * dt;
            let h = (r * r + 2.0 * r * mu * t + t * t).sqrt() - bottom;
            depth += self.extinction_at(h as f32).as_dvec3() * dt;
        }
        (-depth).exp().as_vec3()
    }
}

/// A light far outside the atmosphere with a visible disk: the sun or the moon.
#[derive(Debug, Clone, Copy)]
pub struct CelestialLight {
    /// Unit vector from the scene toward the light, world space (Y up).
    pub direction: Vec3,
    /// Illuminance at the top of the atmosphere (colour times intensity). In lux for a physically
    /// exposed scene (the sun is about 100 000, a full moon about 0.25), or any unit the scene's
    /// other lights share. Zero switches the light off.
    pub illuminance: Vec3,
    /// Apparent diameter, degrees (the sun 0.5357 as Unreal's default, the moon about 0.52).
    pub angular_diameter_deg: f32,
    /// Scales the disk's luminance; 0 hides the disk but keeps the light in the sky.
    pub disk_luminance_scale: f32,
}

impl CelestialLight {
    pub fn sun() -> Self {
        Self {
            direction: direction_from_elevation_bearing(30.0, 180.0),
            illuminance: Vec3::new(1.0, 1.0, 1.0),
            angular_diameter_deg: 0.5357,
            disk_luminance_scale: 1.0,
        }
    }

    /// A moon that is off until given an illuminance.
    pub fn moon() -> Self {
        Self {
            direction: direction_from_elevation_bearing(20.0, 0.0),
            illuminance: Vec3::ZERO,
            angular_diameter_deg: 0.52,
            disk_luminance_scale: 1.0,
        }
    }

    pub(crate) fn angular_radius(&self) -> f32 {
        (self.angular_diameter_deg * 0.5).to_radians().max(1e-5)
    }

    /// Disk luminance per unit illuminance: 1 / (solid angle * mean limb darkening).
    pub(crate) fn disk_luminance(&self, limb_darkening: f32) -> f32 {
        let solid_angle = 2.0 * std::f32::consts::PI * (1.0 - self.angular_radius().cos());
        self.disk_luminance_scale.max(0.0) / (solid_angle * (1.0 - limb_darkening / 3.0))
    }
}

/// Unit vector toward a light at `elevation_deg` above the horizon (negative below it) and a
/// compass `bearing_deg` clockwise from north, in kansei's world frame (Y up, north = -Z, east =
/// +X). Unreal's sun rotator (pitch, yaw) maps to elevation = -pitch, bearing = yaw.
pub fn direction_from_elevation_bearing(elevation_deg: f32, bearing_deg: f32) -> Vec3 {
    let (e, b) = (elevation_deg.to_radians(), bearing_deg.to_radians());
    Vec3::new(b.sin() * e.cos(), e.sin(), -b.cos() * e.cos())
}

// ── GPU layouts (must match the WGSL structs) ──

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, PartialEq)]
pub(crate) struct AtmosphereGpu {
    pub bottom_radius: f32,
    pub top_radius: f32,
    pub rayleigh_exp_scale: f32,
    pub mie_exp_scale: f32,
    pub rayleigh_scattering: [f32; 3],
    pub mie_g: f32,
    pub mie_scattering: [f32; 3],
    pub absorption_tip_altitude: f32,
    pub mie_extinction: [f32; 3],
    pub absorption_tip_value: f32,
    pub absorption_extinction: [f32; 3],
    pub absorption_width: f32,
    pub ground_albedo: [f32; 3],
    pub multi_scattering_factor: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkyFrameGpu {
    pub inv_view_proj: [f32; 16],
    pub camera_pos: [f32; 3],
    pub ap_distance: f32,
    pub world_origin: [f32; 3],
    pub ap_start_depth: f32,
    pub camera_world: [f32; 3],
    pub ap_distance_scale: f32,
    pub sun_direction: [f32; 3],
    pub sun_angular_radius: f32,
    pub sun_illuminance: [f32; 3],
    pub sun_disk_luminance: f32,
    pub moon_direction: [f32; 3],
    pub moon_angular_radius: f32,
    pub moon_illuminance: [f32; 3],
    pub moon_disk_luminance: f32,
    pub sky_luminance_factor: [f32; 3],
    pub _pad0: f32,
    pub sky_light_ground_albedo: [f32; 3],
    pub _pad1: f32,
}

/// The WGSL `SkyLighting` struct: sky radiance SH and the lights at the camera.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SkyLightingGpu {
    pub sh: [[f32; 4]; 9],
    pub sun_illuminance: [f32; 4],
    pub sun_direction: [f32; 4],
    pub moon_illuminance: [f32; 4],
    pub moon_direction: [f32; 4],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zenith_transmittance_matches_the_analytic_optical_depth() {
        let a = AtmosphereParams::earth();
        let g = a.gpu_layout();
        // exponential layers integrate to coefficient * scale height (1 - e^-(top / H)); the tent
        // to tip value * width
        let layer = |h: f32| h * (1.0 - (-a.atmosphere_height_km / h).exp());
        let depth = glam::Vec3::from(g.rayleigh_scattering) * layer(a.rayleigh_exponential_distribution_km)
            + glam::Vec3::from(g.mie_extinction) * layer(a.mie_exponential_distribution_km)
            + glam::Vec3::from(g.absorption_extinction) * a.other_tent_width_km;
        let expected = (-depth).exp();
        let t = a.transmittance_to_space(0.0, 1.0);
        assert!((t - expected).abs().max_element() < 2e-3, "{t} vs {expected}");
        // blue is scattered most, so it is transmitted least
        assert!(t.x > t.y && t.y > t.z);
    }

    #[test]
    fn transmittance_is_zero_below_the_horizon_and_falls_toward_it() {
        let a = AtmosphereParams::earth();
        assert_eq!(a.transmittance_to_space(0.0, -0.01), glam::Vec3::ZERO);
        let low = a.transmittance_to_space(0.0, 0.05);
        let high = a.transmittance_to_space(0.0, 0.5);
        assert!(low.max_element() < high.min_element());
        // from 10 km up, the horizon has dropped: a sun 1 degree below the horizontal still shines
        assert!(a.transmittance_to_space(10.0, (-1.0f32).to_radians().sin()).max_element() > 0.0);
    }

    #[test]
    fn directions_follow_the_world_frame() {
        let north = direction_from_elevation_bearing(0.0, 0.0);
        assert!((north.z + 1.0).abs() < 1e-6 && north.x.abs() < 1e-6);
        let east = direction_from_elevation_bearing(0.0, 90.0);
        assert!((east.x - 1.0).abs() < 1e-6);
        let below = direction_from_elevation_bearing(-2.5, 140.0);
        assert!(below.y < 0.0 && (below.length() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn disk_luminance_integrates_to_the_illuminance() {
        let sun = CelestialLight::sun();
        let (k, r) = (0.6, sun.angular_radius());
        // integrate L0 * (1 - k (1 - mu)) over the disk, as the composite shader does
        let n = 2000;
        let mut e = 0.0f64;
        for i in 0..n {
            let theta = (i as f64 + 0.5) / n as f64 * r as f64;
            let s = (theta.sin() / (r as f64).sin()).min(1.0);
            let mu = (1.0 - s * s).sqrt();
            let d_omega = 2.0 * std::f64::consts::PI * theta.sin() * (r as f64 / n as f64);
            e += sun.disk_luminance(k) as f64 * (1.0 - k as f64 * (1.0 - mu)) * d_omega;
        }
        assert!((e - 1.0).abs() < 2e-3, "{e}");
    }
}
