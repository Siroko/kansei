//! Input without a player, to compare packs side by side.
//!
//! - `drive=1`: a fixed route stands in for the stick: starts, walks, turns, stops, a run, a turn
//!   and a stop at a run, pivots, strafes and walking backwards, round and round. Every pack gets
//!   the same input, so two recordings line up (`drive_restart()` starts it over, from where the
//!   character started, and the clips of `play=` from the first).
//! - `play=<pattern>`: plays the pack's clips whose names start with the pattern (`*` matches any
//!   run of characters: `play=Parkour/*_00`) one after another,
//!   as they are (root motion included), each from the start point, with the clip's name on the
//!   HUD. For clips the search would never pick (a car, pushing, parkour).

use glam::Vec3;
use kansei_core::animation::motion_matching::{Action, Database, RootPath};
use kansei_core::animation::warping::RootWarp;

/// Seconds, the direction travelled (degrees about +Y, 0 is +Z; `None` stands), running, and
/// the facing held while strafing (`None` faces the way it moves).
const ROUTE: [(f32, Option<f32>, bool, Option<f32>); 14] = [
    (2.0, None, false, None),
    (4.0, Some(0.0), false, None),
    (3.0, Some(-90.0), false, None),
    (2.0, None, false, None),
    (3.0, Some(90.0), false, None),
    (4.0, Some(90.0), true, None),
    (3.0, Some(0.0), true, None),
    (2.5, None, false, None),
    (3.0, Some(180.0), true, None),
    (3.0, Some(0.0), false, None),
    (3.0, Some(90.0), false, Some(0.0)),
    (3.0, Some(-90.0), false, Some(0.0)),
    (2.0, Some(180.0), false, Some(0.0)),
    (2.0, None, false, None),
];

/// What the route asks for at a moment: a unit direction (world, on the ground) or none, running,
/// and a strafing facing (radians).
pub struct Step {
    pub direction: Option<Vec3>,
    pub run: bool,
    pub facing: Option<f32>,
}

pub struct Drive {
    pub start: f64,
    /// Where the character started, and its heading: the route turns with it.
    pub home: (Vec3, f32),
}

impl Drive {
    pub fn step(&self, now: f64) -> Step {
        let total: f32 = ROUTE.iter().map(|r| r.0).sum();
        let mut t = ((now - self.start) as f32).rem_euclid(total);
        for &(seconds, direction, run, facing) in &ROUTE {
            if t < seconds {
                let turn = |deg: f32| deg.to_radians() + self.home.1;
                return Step {
                    direction: direction.map(|d| Vec3::new(turn(d).sin(), 0.0, turn(d).cos())),
                    run,
                    facing: facing.map(turn),
                };
            }
            t -= seconds;
        }
        Step { direction: None, run: false, facing: None }
    }
}

/// Plays clips in turn from `home`.
pub struct ClipPlayer {
    pub clips: Vec<usize>,
    next: usize,
    /// When the clip playing ends (and the next starts).
    until: f64,
    pub home: (Vec3, f32),
}

impl ClipPlayer {
    /// The clips whose names start with `pattern`, in the pack's order.
    pub fn new(db: &Database, pattern: &str, home: (Vec3, f32)) -> Self {
        let clips = db.clips.iter().enumerate().filter(|(_, c)| starts_with(&c.name, pattern)).map(|(i, _)| i).collect();
        Self { clips, next: 0, until: 0.0, home }
    }

    /// From the first clip again, now.
    pub fn restart(&mut self) {
        self.next = 0;
        self.until = 0.0;
    }

    /// The next clip to start now, if the one playing has played out (with a moment's hold).
    pub fn due(&mut self, db: &Database, now: f64) -> Option<Action> {
        if self.clips.is_empty() || now < self.until {
            return None;
        }
        let clip = self.clips[self.next % self.clips.len()];
        self.next += 1;
        self.until = now + (db.clips[clip].frames as f64 - 1.0) / db.sample_rate as f64 + 0.8;
        Some(Action { clip, start: 0.0, exit: None, path: RootPath::Warp(RootWarp::identity(db.root_at(clip, 0.0), self.home)), collides: false, tag: 0 })
    }
}

/// Whether `name` starts with `pattern`, `*` in it matching any run of characters.
fn starts_with(name: &str, pattern: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else { return true };
    let Some(mut rest) = name.strip_prefix(first) else { return false };
    for part in parts {
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    true
}
