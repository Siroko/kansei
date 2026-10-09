//! Foot slide: how far planted feet travel across the ground, and a scripted course of moves to
//! measure it on (straight walks and runs, circles both ways, starts and stops, 180° turns).
//!
//! A foot counts as planted while the frame playing says so (`Database::contacts` of a reference
//! database: the one played, or the same frames with their contacts found by other thresholds, so
//! tuning the contacts foot locking pins on does not move the yardstick). Each frame its slide is whichever of the ankle and the toe moved less
//! across the ground (a foot rolling onto its toes moves its ankle, not its toe), counted only
//! while it was planted the frame before too. The report gives it per second planted, and per
//! plant; and how far the character (the mesh) faces from the simulation (the capsule).

use glam::Vec3;

use super::controller::{MotionInput, MotionMatcher};
use super::database::{wrap_angle, yaw_of, Database};
use crate::animation::Skeleton;

/// Collects foot slide while a `MotionMatcher` plays (`sample` after each update).
#[derive(Debug, Clone, Default)]
pub struct FootSlide {
    /// Ankle and toe joints of each foot on the output skeleton (left, right).
    joints: [(usize, Option<usize>); 2],
    /// Where each planted foot's ankle and toe were last frame (world).
    last: [Option<(Vec3, Vec3)>; 2],
    planted_time: [f32; 2],
    slide: [f32; 2],
    plants: [u32; 2],
    time: f32,
    yaw_gap: f32,
    yaw_gap_max: f32,
}

/// What a `FootSlide` measured.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct FootSlideReport {
    /// Planted-foot slide (cm per second planted, both feet together).
    pub cm_per_second: f32,
    /// Slide per plant (cm).
    pub cm_per_plant: f32,
    /// Share of the time each foot was planted, on average.
    pub planted: f32,
    /// Mean and largest angle between the character's facing and the simulation's (degrees).
    pub yaw_gap: f32,
    pub yaw_gap_max: f32,
}

impl FootSlide {
    /// A meter for `matcher`'s output skeleton: the database's feet, found there by name, and
    /// each foot's first child as its toe.
    pub fn new(matcher: &MotionMatcher, db: &Database) -> Self {
        Self::on(matcher.output_skeleton(db), db.roles.feet.map(|f| db.skeleton.names[f].as_str()))
    }

    /// A meter for the feet named `feet` (left, right) on `skeleton`.
    pub fn on(skeleton: &Skeleton, feet: [&str; 2]) -> Self {
        let joints = feet.map(|name| {
            let foot = skeleton.find(name).unwrap_or(0);
            (foot, skeleton.parents.iter().position(|p| *p == Some(foot)))
        });
        Self { joints, ..Default::default() }
    }

    /// Account for the frame `matcher` just played, `dt` seconds long, with whether
    /// each foot is `planted`.
    pub fn sample(&mut self, matcher: &MotionMatcher, planted: [bool; 2], dt: f32) {
        let contacts = planted;
        let (character, model) = (matcher.character(), matcher.model());
        self.time += dt;
        let gap = wrap_angle(yaw_of(character.rotation) - yaw_of(matcher.simulation().rotation)).abs().to_degrees();
        self.yaw_gap += gap * dt;
        self.yaw_gap_max = self.yaw_gap_max.max(gap);
        for side in 0..2 {
            if !contacts[side] {
                self.last[side] = None;
                continue;
            }
            let (ankle, toe) = self.joints[side];
            let ankle = character.transform_point(model[ankle].translation);
            let toe = toe.map_or(ankle, |t| character.transform_point(model[t].translation));
            let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z).length();
            match self.last[side] {
                Some((a, t)) => {
                    self.slide[side] += flat(ankle - a).min(flat(toe - t));
                    self.planted_time[side] += dt;
                }
                None => self.plants[side] += 1,
            }
            self.last[side] = Some((ankle, toe));
        }
    }

    pub fn report(&self) -> FootSlideReport {
        let planted = self.planted_time[0] + self.planted_time[1];
        let slide = self.slide[0] + self.slide[1];
        let plants = self.plants[0] + self.plants[1];
        FootSlideReport {
            cm_per_second: if planted > 0.0 { 100.0 * slide / planted } else { 0.0 },
            cm_per_plant: if plants > 0 { 100.0 * slide / plants as f32 } else { 0.0 },
            planted: if self.time > 0.0 { planted / (2.0 * self.time) } else { 0.0 },
            yaw_gap: if self.time > 0.0 { self.yaw_gap / self.time } else { 0.0 },
            yaw_gap_max: self.yaw_gap_max,
        }
    }
}

/// A move of the foot-slide course: the input, by the seconds since it started, for a character
/// starting out facing +Z.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Move {
    /// Straight ahead (+Z) at `speed` m/s.
    Straight { speed: f32 },
    /// Round a circle of `radius` m at `speed` m/s; `left` turns left (counter-clockwise from
    /// above).
    Circle { radius: f32, speed: f32, left: bool },
    /// Stand `still` seconds, go ahead at `speed` for `go` seconds, again and again.
    StartStop { speed: f32, go: f32, still: f32 },
    /// Go ahead at `speed` for `leg` seconds, then back the way it came, again and again.
    Reverse { speed: f32, leg: f32 },
}

/// One run of the course: a move, at a gait (`run` picks the run tags in `measure`), measured
/// for `seconds` after `settle` seconds of it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scenario {
    pub name: &'static str,
    pub motion: Move,
    pub run: bool,
    pub settle: f32,
    pub seconds: f32,
}

impl Move {
    /// The desired velocity `t` seconds into the move.
    pub fn velocity(&self, t: f32) -> Vec3 {
        let ahead = |speed: f32| Vec3::new(0.0, 0.0, speed);
        match *self {
            Move::Straight { speed } => ahead(speed),
            Move::Circle { radius, speed, left } => {
                // the heading turns at speed / radius (yaw grows to the left: +Z toward +X)
                let yaw = speed / radius * t * if left { 1.0 } else { -1.0 };
                Vec3::new(yaw.sin(), 0.0, yaw.cos()) * speed
            }
            Move::StartStop { speed, go, still } => {
                if (t % (go + still)) < still {
                    Vec3::ZERO
                } else {
                    ahead(speed)
                }
            }
            Move::Reverse { speed, leg } => ahead(if (t / leg) as u32 % 2 == 0 { speed } else { -speed }),
        }
    }
}

/// The course, at walk and run paces of 2 and 5 m/s: straight walk and run, run circles of 2,
/// 3.5 and 5 m and walk circles of 1.5 and 3 m both ways, starts and stops at a run, and 180°
/// turns at a run.
pub fn course(walk: f32, run: f32) -> Vec<Scenario> {
    let circle = |name, radius, speed, left, run| Scenario { name, motion: Move::Circle { radius, speed, left }, run, settle: 3.0, seconds: 12.0 };
    vec![
        Scenario { name: "walk straight", motion: Move::Straight { speed: walk }, run: false, settle: 2.0, seconds: 8.0 },
        Scenario { name: "run straight", motion: Move::Straight { speed: run }, run: true, settle: 2.0, seconds: 8.0 },
        circle("run circle 2 m left", 2.0, run, true, true),
        circle("run circle 2 m right", 2.0, run, false, true),
        circle("run circle 3.5 m left", 3.5, run, true, true),
        circle("run circle 3.5 m right", 3.5, run, false, true),
        circle("run circle 5 m left", 5.0, run, true, true),
        circle("run circle 5 m right", 5.0, run, false, true),
        circle("walk circle 1.5 m left", 1.5, walk, true, false),
        circle("walk circle 1.5 m right", 1.5, walk, false, false),
        circle("walk circle 3 m left", 3.0, walk, true, false),
        circle("walk circle 3 m right", 3.0, walk, false, false),
        Scenario { name: "start and stop", motion: Move::StartStop { speed: run, go: 3.0, still: 2.5 }, run: true, settle: 1.0, seconds: 22.0 },
        Scenario { name: "180 turn", motion: Move::Reverse { speed: run, leg: 3.0 }, run: true, settle: 3.0, seconds: 24.0 },
    ]
}

/// Play `scenario` on `matcher` (standing at the origin facing +Z, its filter set for the
/// scenario's gait) at a fixed 60 Hz step, and measure it, the feet planted where `reference`
/// (`db`'s frames) has its contacts. Within a few seconds a scenario settles into a cycle of the
/// same frames, whatever it started from: a change of tuning that moves one scenario a lot may
/// only have sent it round another cycle, so judge one over the whole course.
pub fn measure(db: &Database, reference: &Database, mut matcher: MotionMatcher, scenario: &Scenario) -> FootSlideReport {
    let dt = 1.0 / 60.0;
    let mut meter = FootSlide::new(&matcher, db);
    let frames = ((scenario.settle + scenario.seconds) / dt).round() as usize;
    for f in 0..frames {
        let t = f as f32 * dt;
        matcher.update(db, &MotionInput { velocity: scenario.motion.velocity(t), facing: None }, dt);
        if t >= scenario.settle {
            meter.sample(&matcher, reference.contacts(matcher.current_frame(db)), dt);
        }
    }
    meter.report()
}
