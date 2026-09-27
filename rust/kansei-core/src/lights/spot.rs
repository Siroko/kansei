use crate::math::Vec3;

/// A spot light in photometric units: a point source whose luminous intensity (candela) is
/// full inside `inner_angle` of `direction` and fades to zero at `outer_angle`, falling off with
/// the inverse square of distance until `range`, where a smooth window takes it to zero.
///
/// Angles are half-angles from the axis in radians, as UE's inner/outer cone angles. The
/// illuminance it gives a surface facing it at distance d is `intensity / d²` lux, so lit
/// surfaces come out in cd/m², ready for `ToneMapEffect`'s EV100 exposure.
pub struct SpotLight {
    pub position: Vec3,
    /// The direction the light points (normalized when packed).
    pub direction: Vec3,
    /// Linear RGB tint; multiplied by `intensity`.
    pub color: Vec3,
    /// Luminous intensity on the axis, in candela.
    pub intensity: f32,
    /// Metres; the light reaches exactly zero here (UE's attenuation radius).
    pub range: f32,
    pub inner_angle: f32,
    pub outer_angle: f32,
    /// Render a perspective shadow map for this light (needs `Renderer::enable_spot_shadows`).
    pub cast_shadow: bool,
    /// Radius of the emitter in metres. Shadows are contact-hardening (PCSS): sharp where the
    /// caster touches the receiver, softer with distance. 0 gives a fixed small PCF kernel.
    pub source_radius: f32,
    /// Scattering in volumetric fog, relative to the surface lighting (UE's volumetric
    /// scattering intensity); 0 keeps the light out of the fog.
    pub volumetric_scale: f32,
    /// Receiver offset along the surface normal for shadow lookups, in shadow-map texels.
    pub shadow_normal_bias: f32,
}

impl SpotLight {
    pub fn new(position: Vec3, direction: Vec3, color: Vec3, intensity: f32, range: f32, inner_angle: f32, outer_angle: f32) -> Self {
        Self {
            position,
            direction,
            color,
            intensity,
            range,
            inner_angle,
            outer_angle,
            cast_shadow: false,
            source_radius: 0.05,
            volumetric_scale: 1.0,
            shadow_normal_bias: 1.5,
        }
    }

    /// The light's colour times its intensity.
    pub fn effective_color(&self) -> Vec3 {
        self.color * self.intensity
    }

    /// Point the light at `target`.
    pub fn look_at(&mut self, target: Vec3) {
        self.direction = (target - self.position).normalize();
    }

    /// Outer and inner half-angles clamped to a valid cone (outer in (0, 89°], inner <= outer).
    pub fn cone(&self) -> (f32, f32) {
        let outer = self.outer_angle.clamp(1e-3, 89f32.to_radians());
        (outer, self.inner_angle.clamp(0.0, outer))
    }

    /// The view matrix of the light's shadow map: at `position`, looking along `direction`.
    pub fn shadow_view(&self) -> glam::Mat4 {
        let pos = glam::Vec3::new(self.position.x, self.position.y, self.position.z);
        let dir = glam::Vec3::new(self.direction.x, self.direction.y, self.direction.z).normalize_or(glam::Vec3::NEG_Z);
        let up = if dir.y.abs() > 0.99 { glam::Vec3::Z } else { glam::Vec3::Y };
        glam::Mat4::look_to_rh(pos, dir, up)
    }

    /// The perspective projection of the light's shadow map: the outer cone plus a margin for
    /// the filter kernel, `[0, 1]` depth from `near` to `range`.
    pub fn shadow_projection(&self, near: f32) -> glam::Mat4 {
        let (outer, _) = self.cone();
        let fov = (outer * 2.0 * 1.05 + 2f32.to_radians()).min(178f32.to_radians());
        glam::Mat4::perspective_rh(fov, 1.0, near, self.range.max(near * 2.0))
    }

    pub fn shadow_view_projection(&self, near: f32) -> glam::Mat4 {
        self.shadow_projection(near) * self.shadow_view()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_view_projection_frames_the_cone() {
        let mut spot = SpotLight::new(Vec3::new(1.0, 2.0, 3.0), Vec3::new(0.0, 0.0, -1.0), Vec3::new(1.0, 1.0, 1.0), 1000.0, 50.0, 0.2, 0.5);
        let vp = spot.shadow_view_projection(0.1);
        // a point on the axis lands in the centre, at a depth inside [0, 1]
        let on_axis = vp.project_point3(glam::Vec3::new(1.0, 2.0, -7.0));
        assert!(on_axis.x.abs() < 1e-5 && on_axis.y.abs() < 1e-5 && on_axis.z > 0.0 && on_axis.z < 1.0, "{on_axis:?}");
        // a point on the outer cone lands inside the map, near its edge
        let edge = glam::Vec3::new(1.0 + 10.0 * 0.5f32.tan(), 2.0, -7.0);
        let p = vp.project_point3(edge);
        assert!(p.x > 0.85 && p.x < 1.0, "{p:?}");
        // straight down doesn't degenerate
        spot.direction = Vec3::new(0.0, -1.0, 0.0);
        let down = spot.shadow_view_projection(0.1).project_point3(glam::Vec3::new(1.0, -8.0, 3.0));
        assert!(down.x.abs() < 1e-4 && down.y.abs() < 1e-4, "{down:?}");
    }
}
