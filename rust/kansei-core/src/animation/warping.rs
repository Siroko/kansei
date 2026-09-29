//! Motion warping: bend a clip's root motion so a moment of it lands on a target, such as a hand
//! reaching the ledge the clip was captured against, wherever the real ledge is.
//!
//! The clip's root path is placed in the world through a frame (a heading and a ground-plane
//! offset) that eases from where the character is when the clip starts to where the target puts
//! the clip, over a window of frames; heights and lengths the target changes are eased in the same
//! way with ramps. Before the window the clip plays as captured from the character; after it, it
//! plays in the target's frame. After the idea of motion warping (Witkin and Popović, "Motion
//! Warping", SIGGRAPH 1995) as games use it for traversal.

use glam::{Quat, Vec2, Vec3};

use super::motion_matching::{wrap_angle, yaw_rotation};

/// Smooth step between 0 and 1.
fn ease(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// A value that eases from `from` to `to` between two clip frames.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ramp {
    pub start: f32,
    pub end: f32,
    pub from: f32,
    pub to: f32,
}

impl Ramp {
    pub fn new(start: f32, end: f32, from: f32, to: f32) -> Self {
        Self { start, end, from, to }
    }

    pub fn at(&self, frame: f32) -> f32 {
        if self.end <= self.start {
            return if frame < self.start { self.from } else { self.to };
        }
        self.from + (self.to - self.from) * ease((frame - self.start) / (self.end - self.start))
    }
}

/// A rigid placement on the ground: a heading (radians about +Y) and a ground-plane offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Placement {
    pub yaw: f32,
    pub offset: Vec2,
}

impl Placement {
    pub const IDENTITY: Self = Self { yaw: 0.0, offset: Vec2::ZERO };

    /// The placement that takes a clip root (position, heading) to a world one.
    pub fn between(clip: (Vec3, f32), world: (Vec3, f32)) -> Self {
        let yaw = wrap_angle(world.1 - clip.1);
        let rotated = yaw_rotation(yaw) * clip.0;
        Self { yaw, offset: Vec2::new(world.0.x - rotated.x, world.0.z - rotated.z) }
    }

    pub fn apply(&self, p: Vec3) -> Vec3 {
        let r = yaw_rotation(self.yaw) * p;
        Vec3::new(r.x + self.offset.x, r.y, r.z + self.offset.y)
    }

    fn lerp(&self, other: &Placement, t: f32) -> Placement {
        Placement { yaw: self.yaw + wrap_angle(other.yaw - self.yaw) * t, offset: self.offset.lerp(other.offset, t) }
    }
}

/// A clip's root path warped into the world.
#[derive(Debug, Clone, PartialEq)]
pub struct RootWarp {
    /// Where the clip is placed when it starts (so the character does not jump)...
    pub from: Placement,
    /// ...and where the target puts it, reached over `window` (0 to 1 over its frames).
    pub to: Placement,
    pub window: Ramp,
    /// World height of the clip's height 0.
    pub ground: f32,
    /// Heights added to the clip's root, summed (e.g. up by the difference between the real and
    /// the captured obstacle, then down again past it).
    pub lift: Vec<Ramp>,
    /// Distances added along `stretch_direction` (world), summed (a deeper obstacle).
    pub stretch: Vec<Ramp>,
    pub stretch_direction: Vec3,
}

impl RootWarp {
    /// A warp that plays the clip as captured from where the character is: the clip root at
    /// `start` (in clip space) maps to the character (world position and heading).
    pub fn identity(clip_start: (Vec3, f32), character: (Vec3, f32)) -> Self {
        let from = Placement::between((Vec3::new(clip_start.0.x, 0.0, clip_start.0.z), clip_start.1), (Vec3::new(character.0.x, 0.0, character.0.z), character.1));
        Self {
            from,
            to: from,
            window: Ramp::new(0.0, 0.0, 0.0, 1.0),
            ground: character.0.y - clip_start.0.y,
            lift: Vec::new(),
            stretch: Vec::new(),
            stretch_direction: Vec3::ZERO,
        }
    }

    /// The world root (position and heading) of the clip root (clip-space position and heading)
    /// at clip frame `frame`.
    pub fn root(&self, frame: f32, clip: (Vec3, f32)) -> (Vec3, f32) {
        let placement = self.from.lerp(&self.to, self.window.at(frame));
        let mut p = placement.apply(clip.0);
        p.y = self.ground + clip.0.y + self.lift.iter().map(|r| r.at(frame)).sum::<f32>();
        p += self.stretch_direction * self.stretch.iter().map(|r| r.at(frame)).sum::<f32>();
        (p, clip.1 + placement.yaw)
    }

    /// `root` as a heading rotation.
    pub fn root_rotation(&self, frame: f32, clip: (Vec3, f32)) -> (Vec3, Quat) {
        let (p, yaw) = self.root(frame, clip);
        (p, yaw_rotation(yaw))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clip root walking +z at 2 m/s for 60 frames at 30 fps, rising 1 m between frames 20 and
    /// 30 (onto a 1 m obstacle whose edge is at z = 0 in clip space).
    fn clip_root(frame: f32) -> (Vec3, f32) {
        let z = -2.0 + frame * 2.0 / 30.0;
        let y = ((frame - 20.0) / 10.0).clamp(0.0, 1.0);
        (Vec3::new(0.0, y, z), 0.0)
    }

    #[test]
    fn ramps_ease_between_their_frames() {
        let r = Ramp::new(10.0, 20.0, 1.0, 3.0);
        assert_eq!((r.at(0.0), r.at(10.0), r.at(15.0), r.at(20.0), r.at(99.0)), (1.0, 1.0, 2.0, 3.0, 3.0));
        assert!(r.at(12.0) < 1.0 + 2.0 * 0.2, "eased, slow at first");
        assert_eq!(Ramp::new(5.0, 5.0, 0.0, 1.0).at(4.0), 0.0);
        assert_eq!(Ramp::new(5.0, 5.0, 0.0, 1.0).at(5.0), 1.0);
    }

    #[test]
    fn placements_map_clip_roots_to_world_ones() {
        let p = Placement::between((Vec3::new(1.0, 0.0, 2.0), 0.3), (Vec3::new(-4.0, 0.0, 7.0), 1.8));
        let q = p.apply(Vec3::new(1.0, 0.5, 2.0));
        assert!(q.abs_diff_eq(Vec3::new(-4.0, 0.5, 7.0), 1e-5), "{q}");
        assert!((p.yaw - 1.5).abs() < 1e-6);
    }

    #[test]
    fn the_warp_starts_at_the_character_and_lands_the_moment_on_the_target() {
        // the character is 1 m to the side of where the clip would start, turned 20 degrees
        let start = 0.0;
        let character = (Vec3::new(5.0, 0.2, 3.0), 0.35);
        let mut warp = RootWarp::identity(clip_root(start), character);
        // the real obstacle: 1.4 m high, its edge at (10, 1.6, 8) facing -x (the clip runs +z into
        // it, so the clip's heading maps to +x... here the world approach heading is 90 degrees)
        let edge = Vec3::new(10.0, 0.2 + 1.4, 8.0);
        let anchor = 25.0; // the clip frame at which the root is at the edge (z = 0 at frame 30)
        let clip_edge = (Vec3::new(0.0, 0.0, 0.0), 0.0);
        warp.to = Placement::between(clip_edge, (Vec3::new(edge.x, 0.0, edge.z), std::f32::consts::FRAC_PI_2));
        warp.window = Ramp::new(start, anchor, 0.0, 1.0);
        warp.lift = vec![Ramp::new(20.0, anchor, 0.0, 0.4)];
        // frame 0: exactly where the character is
        let (p, yaw) = warp.root(start, clip_root(start));
        assert!(p.abs_diff_eq(character.0, 1e-5) && (yaw - character.1).abs() < 1e-6, "{p} {yaw}");
        // at the anchor and after: in the target's frame, and 1.4 m up on the obstacle by frame 30
        let (p, yaw) = warp.root(30.0, clip_root(30.0));
        assert!(p.abs_diff_eq(Vec3::new(10.0, 1.6, 8.0), 1e-4), "{p}");
        assert!((yaw - std::f32::consts::FRAC_PI_2).abs() < 1e-5);
        let (p, _) = warp.root(45.0, clip_root(45.0));
        assert!(p.abs_diff_eq(Vec3::new(11.0, 1.6, 8.0), 1e-4), "{p}");
        // continuous: no step between frames anywhere
        let mut last = warp.root(0.0, clip_root(0.0)).0;
        for k in 1..=600 {
            let f = k as f32 * 0.1;
            let (p, _) = warp.root(f, clip_root(f));
            assert!(p.distance(last) < 0.1, "frame {f}: {} m", p.distance(last));
            last = p;
        }
    }

    #[test]
    fn stretching_lengthens_the_path_along_a_direction() {
        let mut warp = RootWarp::identity(clip_root(0.0), (Vec3::new(0.0, 0.0, -2.0), 0.0));
        warp.stretch = vec![Ramp::new(30.0, 40.0, 0.0, 0.5)];
        warp.stretch_direction = Vec3::Z;
        assert!(warp.root(30.0, clip_root(30.0)).0.abs_diff_eq(clip_root(30.0).0, 1e-5));
        assert!(warp.root(50.0, clip_root(50.0)).0.abs_diff_eq(clip_root(50.0).0 + Vec3::new(0.0, 0.0, 0.5), 1e-5));
    }
}
