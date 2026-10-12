//! Parkour moves on top of the traversals, driven by the tagged geometry in `features`:
//!
//! - **Aimed jumps.** A jump toward a platform (or any top across a gap) takes off at the speed
//!   whose arc lands on it, its run-up turned toward it.
//! - **Hanging.** A character falling past a ledge's edge catches it; one climbing a tagged ledge
//!   stops hanging from it. The hang is the climb clip's own hanging frame, held (breathing a frame
//!   either way), placed from the edge; the hands are pinned to the edge and the feet to the wall
//!   with IK. The stick shimmies along the edge and round its corners, hands and feet stepping in
//!   turn (each pinned where it lands); Space climbs up (the climb clip on from the hang); drop lets
//!   go.
//! - **Wall runs.** Running or jumping along a `wallrun` face runs along it: the run loop at the
//!   wall-run speed, the body leaning away from the wall, the planted feet pinned flat on it; light
//!   gravity at first, then more; Space jumps off it, away from the wall.
//! - **Bars.** A jump or fall that brings the hands to a bar catches it: the hang pose swings
//!   about the bar as a damped pendulum, pumped by the stick, the hands pinned to it; Space lets go
//!   at the swing's speed, drop lets go still.
//!
//! The moves are scripted (the clips animate the pose, the controller places the character), as
//! the traversals are; motion matching takes over again on the ground.

use glam::{Quat, Vec3};

use super::features::ParkourFeatures;
use super::{ActionClip, ActionKind, CharacterController, CharacterState, Refusal};
use crate::animation::motion_matching::controller::{Action, LimbTarget, MotionInput, RootPath};
use crate::animation::motion_matching::database::{yaw_of, yaw_rotation, Database, FORWARD};
use crate::animation::warping::{Placement, Ramp, RootWarp};
use crate::collision::CollisionWorld;

/// Tuning of the parkour moves.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParkourSettings {
    /// Aimed jumps: the farthest horizontal take-off speed (m/s), the extra height over the
    /// target the arc clears (m), and the cone (radians either way of the input) a target is
    /// looked for in.
    pub max_jump_speed: f32,
    pub jump_clearance: f32,
    pub jump_cone: f32,
    /// Catching an edge: how far the root may be from its hanging place when it does (m), and the
    /// shortest height of the face under the edge for a hang (m).
    pub catch_reach: f32,
    pub min_hang_height: f32,
    /// Shimmy speed (m/s) and stride (m).
    pub shimmy_speed: f32,
    pub shimmy_stride: f32,
    /// Wall runs: the slowest speed along the wall that starts one (m/s), the run's speed range,
    /// its lean away from the wall (radians), the take-off's vertical speed, gravity's share for
    /// the first `wall_light_time` seconds and after, and the longest run (s).
    pub wall_min_speed: f32,
    pub wall_speed: (f32, f32),
    pub wall_lean: f32,
    pub wall_up_speed: f32,
    pub wall_gravity: (f32, f32),
    pub wall_light_time: f32,
    pub wall_max_time: f32,
    /// Jumping off a wall: speed out of it and up (m/s).
    pub wall_jump: (f32, f32),
    /// Bars: the pendulum's length (pivot to the body's centre, m), its damping (1/s), the stick's
    /// pump (rad/s²) and the widest swing (radians).
    pub swing_length: f32,
    pub swing_damping: f32,
    pub swing_pump: f32,
    pub swing_max: f32,
    /// Seconds after letting go of an edge or bar (or leaving a wall) before it can catch the same
    /// again.
    pub cooldown: f32,
}

impl Default for ParkourSettings {
    fn default() -> Self {
        Self {
            max_jump_speed: 7.5,
            jump_clearance: 0.45,
            jump_cone: 0.5,
            catch_reach: 0.55,
            min_hang_height: 1.9,
            shimmy_speed: 0.8,
            shimmy_stride: 0.34,
            wall_min_speed: 2.8,
            wall_speed: (4.5, 6.0),
            wall_lean: 0.7,
            wall_up_speed: 2.6,
            wall_gravity: (0.3, 0.85),
            wall_light_time: 0.45,
            wall_max_time: 1.3,
            wall_jump: (3.8, 4.4),
            swing_length: 1.05,
            swing_damping: 0.25,
            swing_pump: 3.0,
            swing_max: 1.3,
            cooldown: 0.5,
        }
    }
}

/// The hang: the climb clip's frame the character hangs at, and where its root, hands and feet
/// are then relative to the clip's ledge, in the edge's frame (x: along the edge to the
/// character's right... `side`, y: up, z: into the wall).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HangPose {
    pub climb: ActionClip,
    pub frame: f32,
    pub root: Vec3,
    /// The root's heading less the clip's approach heading.
    pub yaw: f32,
    pub hands: [Vec3; 2],
    pub feet: [Vec3; 2],
}

/// The forward run loop wall runs play: its clip and speed (m/s).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunLoop {
    pub clip: usize,
    pub speed: f32,
}

/// The pack's clips of hanging from a bar: still, and swinging (`ActionKind::BarHang`,
/// `BarSwing`). The swing clip is phase-locked to the pendulum: its frames over one swing and the
/// angle it swings by (read from its hips under its hands).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BarClips {
    pub hang: usize,
    /// The swing clip, its period and angle (radians), and the frame its swing passes the
    /// bottom going forward (phase 0).
    pub swing: Option<(usize, f32, f32, f32)>,
}

/// The hands' midpoint relative to the clip root (root-local) at `frame` of `clip`.
fn hands_mid(db: &Database, clip: usize, frame: f32, hands: [usize; 2]) -> Vec3 {
    let info = &db.clips[clip];
    let f = frame.clamp(0.0, (info.frames - 1) as f32);
    let a = f.floor() as usize;
    let b = (a + 1).min(info.frames - 1);
    let mut pose = crate::animation::Pose { local: Vec::new() };
    let mut model = Vec::new();
    db.pose(info.start + a, info.start + b, f - a as f32, &mut pose);
    pose.to_model(&db.skeleton, &mut model);
    (model[hands[0]].translation + model[hands[1]].translation) * 0.5
}

/// Database joint `joint` relative to the clip root (root-local) at `frame` of `clip`.
fn clip_joint(db: &Database, clip: usize, frame: f32, joint: usize) -> Vec3 {
    let info = &db.clips[clip];
    let a = (frame.max(0.0) as usize).min(info.frames - 1);
    let mut pose = crate::animation::Pose { local: Vec::new() };
    let mut model = Vec::new();
    db.pose(info.start + a, info.start + a, 0.0, &mut pose);
    pose.to_model(&db.skeleton, &mut model);
    model[joint].translation
}

/// Read the bar clips from the pack's actions: the first still hang and the first swing, its
/// period from the hips' swing under the hands (zero crossings of their offset along the facing)
/// and its angle.
pub fn bar_clips(db: &Database, actions: &[ActionClip], hands: [usize; 2]) -> Option<BarClips> {
    let hang = actions.iter().find(|c| c.kind == ActionKind::BarHang)?.clip;
    let swing = actions.iter().find(|c| c.kind == ActionKind::BarSwing).and_then(|c| {
        let info = &db.clips[c.clip];
        let mut pose = crate::animation::Pose { local: Vec::new() };
        let mut model = Vec::new();
        let offsets: Vec<(f32, f32)> = (0..info.frames)
            .map(|f| {
                db.pose(info.start + f, info.start + f, 0.0, &mut pose);
                pose.to_model(&db.skeleton, &mut model);
                let mid = (model[hands[0]].translation + model[hands[1]].translation) * 0.5;
                let hips = model[db.roles.hips].translation;
                (hips.z - mid.z, mid.y - hips.y)
            })
            .collect();
        // upward zero crossings of the hips' offset: whole swings
        let ups: Vec<usize> = (1..offsets.len()).filter(|&f| offsets[f - 1].0 < 0.0 && offsets[f].0 >= 0.0).collect();
        let period = if ups.len() >= 2 { (ups[1] - ups[0]) as f32 } else { return None };
        let reach = offsets.iter().map(|o| o.0.abs()).fold(0.0f32, f32::max);
        let length = offsets.iter().map(|o| o.1).sum::<f32>() / offsets.len() as f32;
        Some((c.clip, period, (reach / length.max(0.3)).clamp(0.05, 1.2).asin(), ups[0] as f32))
    });
    Some(BarClips { hang, swing })
}

/// A limb's step, for the shimmy's gait: where it is planted (world), and while stepping where
/// from, and its progress.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
struct Step {
    planted: Vec3,
    from: Vec3,
    stepping: bool,
    /// A step out of turn, its progress (0..1): the body moved too far from where the limb was
    /// planted for it to reach (round a corner).
    forced: Option<f32>,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Hang {
    pub edge: usize,
    pub s: f32,
    /// Seconds since the hang began, the shimmy's gait phase (0..1) and direction.
    time: f32,
    phase: f32,
    direction: f32,
    /// Hands, then feet.
    limbs: [Step; 4],
    /// Getting there (catching): the root's place and heading when the catch began, the seconds
    /// it takes and has taken.
    from: Option<(Vec3, f32)>,
    settle: f32,
    /// The face under the edge holds the feet (else they hang free).
    braced: bool,
    /// The stick's direction when the shimmy began, and which way along the edges it goes.
    latch: Option<(Vec3, f32)>,
    /// The clip frame the catch started at.
    start_frame: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct WallRun {
    pub face: usize,
    sign: f32,
    along: f32,
    y: f32,
    vy: f32,
    speed: f32,
    time: f32,
    /// Where the root was when the run began: its distance out of the face and heading; and the
    /// seconds it takes to get onto the wall.
    from_out: f32,
    entry: f32,
    from_yaw: f32,
    /// The feet's pins on the wall and how planted each is.
    pins: [Option<Vec3>; 2],
    weights: [f32; 2],
}

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Swing {
    pub bar: usize,
    s: f32,
    theta: f32,
    omega: f32,
    /// The heading the body faces (perpendicular to the bar).
    yaw: f32,
    time: f32,
    from: Option<(Vec3, Quat)>,
    /// Swinging (the swing clip playing) rather than hanging still.
    swinging: bool,
}

/// The parkour state a `CharacterController` keeps.
#[derive(Debug, Clone, Default)]
pub(super) struct Runtime {
    pub hang: Option<Hang>,
    pub wall: Option<WallRun>,
    pub swing: Option<Swing>,
    /// Where an aimed jump lands.
    pub jump_target: Option<Vec3>,
    /// A climb of a tagged ledge that stops hanging from it.
    pub hang_after: bool,
    /// What was let go of, and for how much longer it can't be caught.
    pub cooldown: Option<(Catchable, f32)>,
    /// The heading to turn to in the air, and whether to come upright.
    pub air_yaw: Option<f32>,
    pub hang_poses: Option<Vec<HangPose>>,
    /// The hang pose in use (the climb clip's own, or the stillest for a catch).
    pub hang_pose: Option<HangPose>,
    pub run_loop: Option<Option<RunLoop>>,
    /// The bar clips (`ActionKind::BarHang`, `BarSwing`), if the pack has them.
    pub bar_clips: Option<Option<BarClips>>,
    /// The hips' height over the root in the fall loop (m).
    pub fall_hips: f32,
    /// The limb targets set this frame (for `reach`, and for measuring).
    pub limbs: Vec<LimbTarget>,
    /// Each hand's and foot's push out of the level (`apply_limbs`): its weight and the push.
    pub limb_out: [(f32, Vec3); 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Catchable {
    Edge(usize),
    Bar(usize),
    Face(usize),
}

/// How far below a bar the hips hang (m), for aiming a jump at it.
const HANG_DROP: f32 = 0.92;

/// Added to a jump target's height to mark it as where the hips should pass (a bar to catch),
/// not where the feet land.
pub(super) const BAR_AIM: f32 = 1000.0;

/// An ankle's clearance from what the foot stands or pushes on (m).
const FOOT_RADIUS: f32 = 0.06;

/// A wrist's clearance from what it grips (m).
const HAND_RADIUS: f32 = 0.035;

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// The edge frame's axes in the world for an edge whose outward normal is `normal`: side (x),
/// up, into the wall (z).
fn edge_frame(normal: Vec3) -> (Vec3, Vec3) {
    let into = Vec3::new(-normal.x, 0.0, -normal.z).normalize_or(FORWARD);
    (Vec3::Y.cross(into), into)
}

fn from_edge_frame(origin: Vec3, normal: Vec3, local: Vec3) -> Vec3 {
    let (side, into) = edge_frame(normal);
    origin + side * local.x + Vec3::Y * local.y + into * local.z
}

/// The hands' and feet's database joints.
fn limb_joints(db: &Database, hands: [usize; 2]) -> [usize; 4] {
    [hands[0], hands[1], db.roles.feet[0], db.roles.feet[1]]
}

/// Read the hang from each climb clip: the frame after the hands reach the ledge where the root
/// moves least, its hands on the ledge (within 6 cm), the feet on the face under it. The best
/// (stillest) first.
pub fn hang_poses(db: &Database, actions: &[ActionClip], hands: [usize; 2]) -> Vec<HangPose> {
    let mut poses: Vec<(f32, HangPose)> = Vec::new();
    let mut pose = crate::animation::Pose { local: Vec::new() };
    let mut model = Vec::new();
    for c in actions.iter().filter(|c| c.kind == ActionKind::Climb) {
        let info = &db.clips[c.clip];
        let clip_yaw = c.forward.x.atan2(c.forward.z);
        let into = c.forward;
        let side = Vec3::Y.cross(into);
        let to_local = |p: Vec3| {
            let d = p - c.ledge;
            Vec3::new(d.dot(side), d.y, d.dot(into))
        };
        let first = c.anchor.ceil() as usize;
        let last = (c.on_top.floor() as usize).min(info.frames.saturating_sub(2));
        let mut best: Option<(f32, HangPose)> = None;
        for f in first.max(1)..last {
            let (p0, _) = db.root_at(c.clip, f as f32 - 1.0);
            let (p1, _) = db.root_at(c.clip, f as f32 + 1.0);
            let speed = p0.distance(p1) * db.sample_rate * 0.5;
            db.pose(info.start + f, info.start + f, 0.0, &mut pose);
            pose.to_model(&db.skeleton, &mut model);
            let (root, yaw) = db.root_at(c.clip, f as f32);
            let r = yaw_rotation(yaw);
            let world = |j: usize| root + r * model[j].translation;
            let hand = hands.map(|h| to_local(world(h)));
            // (the ledge is where the hands grip, a few centimetres short of them)
            let off = hand.iter().map(|h| h.y.abs() + (h.z - 0.05).abs()).fold(0.0f32, f32::max);
            if off > 0.06 || to_local(root).y > -1.0 {
                continue;
            }
            let cost = speed + 4.0 * off;
            if best.as_ref().is_none_or(|(b, _)| cost < *b) {
                let feet = db.roles.feet.map(|j| to_local(world(j)));
                best = Some((cost, HangPose { climb: *c, frame: f as f32, root: to_local(root), yaw: crate::animation::motion_matching::wrap_angle(yaw - clip_yaw), hands: hand, feet }));
            }
        }
        poses.extend(best);
    }
    poses.sort_by(|a, b| a.0.total_cmp(&b.0));
    poses.into_iter().map(|(_, p)| p).collect()
}

/// The forward run loop with the speed nearest `wanted`: looping clips whose root runs straight
/// ahead (little sideways drift or turn) at 3.5 m/s or more.
pub fn run_loop(db: &Database, wanted: f32) -> Option<RunLoop> {
    db.clips
        .iter()
        .enumerate()
        .filter(|(_, c)| c.looping && c.frames > 10)
        .filter_map(|(k, c)| {
            let end = (c.playable() as f32).max(1.0);
            let (moved, turned) = db.root_motion(k, 0.0, end);
            let seconds = end / db.sample_rate;
            let speed = moved.z / seconds;
            (speed >= 3.5 && moved.x.abs() < 0.1 * moved.z && turned.abs() < 0.15).then_some(RunLoop { clip: k, speed })
        })
        .min_by(|a, b| (a.speed - wanted).abs().total_cmp(&(b.speed - wanted).abs()))
}

/// Move each climb's and mantle's ledge back so the wrists, gripping it, stay `clearance` in
/// front of its face: `ActionClip::analyze` puts it just short of the planted hands, but a wrist
/// hanging below the top is in front of the face it grips, by more than that.
pub fn refine_grips(db: &Database, actions: &mut [ActionClip], hands: [usize; 2]) {
    let clearance = 0.03;
    let mut pose = crate::animation::Pose { local: Vec::new() };
    let mut model = Vec::new();
    for c in actions.iter_mut().filter(|c| matches!(c.kind, ActionKind::Climb | ActionKind::Mantle)) {
        let info = &db.clips[c.clip];
        let mut deepest = f32::MIN;
        let last = (c.on_top.ceil() as usize).min(info.frames - 1);
        for f in (c.anchor.floor() as usize)..=last {
            db.pose(info.start + f, info.start + f, 0.0, &mut pose);
            pose.to_model(&db.skeleton, &mut model);
            let (root, yaw) = db.root_at(c.clip, f as f32);
            for &h in &hands {
                let p = root + yaw_rotation(yaw) * model[h].translation;
                // a wrist below the top: in front of the face
                if p.y < c.ledge.y - 0.005 {
                    deepest = deepest.max((p - c.ledge).dot(c.forward));
                }
            }
        }
        if deepest > -clearance {
            c.ledge -= c.forward * (deepest + clearance);
        }
    }
}

impl CharacterController {
    /// Find the hand joints by their usual names (`hand_l`/`hand_r`, `LeftHand`/`RightHand`);
    /// without them the hang, the shimmy and the bars are off. With them, the climbs' and mantles'
    /// ledges are refined from the hands (`refine_grips`).
    pub fn find_hands(&mut self, db: &Database) {
        let pairs = [("hand_l", "hand_r"), ("LeftHand", "RightHand"), ("mixamorig:LeftHand", "mixamorig:RightHand"), ("Hand_L", "Hand_R")];
        self.hands = pairs.iter().find_map(|(l, r)| Some([db.skeleton.find(l)?, db.skeleton.find(r)?]));
        if let Some(hands) = self.hands {
            refine_grips(db, &mut self.actions, hands);
        }
        self.park.hang_poses = None;
        self.park.hang_pose = None;
        self.park.fall_hips = self.actions.iter().find(|c| c.kind == ActionKind::Fall).map_or(1.0, |c| clip_joint(db, c.clip, 0.0, db.roles.hips).y);
    }

    /// The limb targets set this frame (world): hands on an edge or bar, feet on a wall.
    pub fn limb_targets(&self) -> &[LimbTarget] {
        &self.park.limbs
    }

    /// The hang (edge and distance along it) while hanging.
    pub fn hanging(&self) -> Option<(usize, f32)> {
        self.park.hang.as_ref().map(|h| (h.edge, h.s))
    }

    /// The face run along while wall running, and the bar swung from while swinging.
    pub fn wall_face(&self) -> Option<usize> {
        self.park.wall.as_ref().map(|w| w.face)
    }

    pub fn bar(&self) -> Option<(usize, f32)> {
        self.park.swing.as_ref().map(|s| (s.bar, s.theta))
    }

    /// Where the aimed jump in progress lands.
    pub fn jump_target(&self) -> Option<Vec3> {
        self.park.jump_target
    }

    /// The hang poses of the climb clips (read once), the stillest first.
    fn hang_poses_cached(&mut self, db: &Database) -> &[HangPose] {
        if self.park.hang_poses.is_none() {
            self.park.hang_poses = Some(self.hands.map_or(Vec::new(), |h| hang_poses(db, &self.actions, h)));
        }
        self.park.hang_poses.as_deref().unwrap_or(&[])
    }

    /// The hang pose for a catch (the stillest), or for climb clip `clip`.
    fn hang_pose_for(&mut self, db: &Database, clip: Option<usize>) -> Option<HangPose> {
        let poses = self.hang_poses_cached(db);
        match clip {
            Some(c) => poses.iter().find(|p| p.climb.clip == c).copied(),
            None => poses.first().copied(),
        }
    }

    /// The hang pose in use.
    fn hang_pose_cached(&mut self, db: &Database) -> Option<HangPose> {
        if self.park.hang_pose.is_none() {
            self.park.hang_pose = self.hang_pose_for(db, None);
        }
        self.park.hang_pose
    }

    fn bar_clips_cached(&mut self, db: &Database) -> Option<BarClips> {
        if self.park.bar_clips.is_none() {
            self.park.bar_clips = Some(self.hands.and_then(|h| bar_clips(db, &self.actions, h)));
        }
        self.park.bar_clips.flatten()
    }

    fn run_loop_cached(&mut self, db: &Database) -> Option<RunLoop> {
        if self.park.run_loop.is_none() {
            self.park.run_loop = Some(run_loop(db, self.parkour.wall_speed.0));
        }
        self.park.run_loop.flatten()
    }

    fn cooling(&self, what: Catchable) -> bool {
        self.park.cooldown.is_some_and(|(c, t)| c == what && t > 0.0)
    }

    /// Space while hanging, wall running or swinging: climb up, jump off, let go. `None` when
    /// none of them is going on (or a wall run starts instead).
    pub(super) fn parkour_request(&mut self, db: &Database, world: &CollisionWorld) -> Option<Result<ActionKind, Refusal>> {
        match self.state {
            CharacterState::Hanging => Some(self.climb_from_hang(db, world)),
            CharacterState::WallRunning => {
                self.wall_jump(db);
                Some(Ok(ActionKind::Jump))
            }
            CharacterState::Swinging => {
                self.release_bar(db, true);
                Some(Ok(ActionKind::Jump))
            }
            CharacterState::Grounded => self.try_wall_run(db, true).then_some(Ok(ActionKind::Jump)),
            _ => None,
        }
    }

    /// Let go: of an edge or a bar (falling straight down), or a wall (falling off it).
    pub fn request_drop(&mut self, db: &Database) -> bool {
        match self.state {
            CharacterState::Hanging => {
                let Some(h) = self.park.hang.take() else { return false };
                let normal = self.features.edges[h.edge].normal;
                self.park.cooldown = Some((Catchable::Edge(h.edge), self.parkour.cooldown));
                self.park.limbs.clear();
                self.fall_with(db, normal * 0.6);
                self.air_top = self.matcher.character().translation.y;
                self.state = CharacterState::Falling(0.0);
                true
            }
            CharacterState::Swinging => {
                self.release_bar(db, false);
                true
            }
            CharacterState::WallRunning => {
                self.leave_wall(db, false);
                true
            }
            _ => false,
        }
    }

    // ---- aimed jumps ----

    /// Where a jump from `from` toward `direction` lands: the next platform of the chain stood on,
    /// a platform in the cone ahead, or a top across a gap. `None`: a plain jump.
    pub(super) fn jump_target_from(&self, world: &CollisionWorld, from: Vec3, direction: Vec3) -> Option<Vec3> {
        let dir = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
        if dir == Vec3::ZERO {
            return None;
        }
        let s = &self.parkour;
        let f = &self.features;
        let under = f.platform_under(from, 0.15);
        let reachable = |p: Vec3| {
            let d = Vec3::new(p.x - from.x, 0.0, p.z - from.z);
            let dist = d.length();
            let dh = p.y - from.y;
            (1.0..=5.5).contains(&dist) && (-2.5..=1.2).contains(&dh) && d.normalize_or_zero().dot(dir) >= s.jump_cone.cos()
        };
        let mut candidates: Vec<(f32, Vec3)> = Vec::new();
        // a ledge ahead to catch (above what a jump lands on): the arc brings the hips under its
        // edge, a hand's reach below it
        for e in &f.edges {
            if e.a.y - e.floor < self.parkour.min_hang_height || e.normal.dot(dir) > -0.5 {
                continue;
            }
            let s = e.project(from + dir * 1.5).clamp(0.4, (e.length() - 0.4).max(0.4));
            let point = e.point(s);
            // (high: the hands reach up from the jump's arms-down pose a moment later)
            let hips = point + e.normal * 0.3 - Vec3::Y * 0.6;
            let d = Vec3::new(hips.x - from.x, 0.0, hips.z - from.z);
            if (0.6..=3.5).contains(&d.length()) && hips.y - from.y > 0.3 && hips.y - from.y < 2.4 && d.normalize_or_zero().dot(dir) >= self.parkour.jump_cone.cos() {
                candidates.push((d.length() - 15.0, Vec3::new(hips.x, hips.y + BAR_AIM, hips.z)));
            }
        }
        // a bar ahead to catch: the arc brings the hips to where hanging from it puts them (the
        // hips as high over the root as the fall loop holds them)
        for b in &f.bars {
            let (point, along) = b.nearest(from + dir * 1.5);
            if along < 0.3 || along > b.length() - 0.3 {
                continue;
            }
            let hang = point - Vec3::Y * HANG_DROP - dir * 0.05;
            let root = hang - Vec3::Y * self.park.fall_hips;
            let d = Vec3::new(root.x - from.x, 0.0, root.z - from.z);
            if (0.4..=3.0).contains(&d.length()) && root.y - from.y < 1.2 && d.normalize_or_zero().dot(dir) >= s.jump_cone.cos() {
                // (marked as a hips target: the take-off aims the hips there)
                candidates.push((d.length() - 20.0, Vec3::new(hang.x, hang.y + BAR_AIM, hang.z)));
            }
        }
        for (k, p) in f.platforms.iter().enumerate() {
            if Some(k) == under {
                continue;
            }
            // the point of its top nearest the line ahead, a little in from its edges
            let centre = p.obb.center;
            let ahead = from + dir * Vec3::new(centre.x - from.x, 0.0, centre.z - from.z).dot(dir).max(0.0);
            let target = p.nearest_on_top(Vec3::new(ahead.x, from.y, ahead.z), 0.35);
            if reachable(target) {
                let next = under.and_then(|u| f.next_in_chain(u)) == Some(k);
                let cost = from.distance(target) - if next { 10.0 } else { 0.0 };
                candidates.push((cost, target));
            }
        }
        if let Some((_, t)) = candidates.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)) {
            return Some(t);
        }
        // a gap ahead, then a top to stand on
        let ground = |d: f32| world.ground_height(from + dir * d, 1.3, 3.0, self.layers);
        let mut gap = false;
        let mut d = 0.5;
        while d <= 5.0 {
            match ground(d) {
                Some(g) if g < from.y - 0.6 => gap = true,
                None => gap = true,
                Some(g) if gap && g >= from.y - 2.5 && g <= from.y + 1.2 => {
                    let target = from + dir * (d + 0.4);
                    let target = Vec3::new(target.x, g, target.z);
                    if super::stands_at(world, target, &self.rules, self.layers) && reachable(target) {
                        return Some(target);
                    }
                    return None;
                }
                Some(g) if g > from.y + 0.3 => return None,
                _ => {}
            }
            d += 0.25;
        }
        None
    }

    /// The take-off velocity whose arc lands on `target` from `from` (gravity `g`): up at `up` (the
    /// jump clip's own take-off, so the body's rise carries on), more only to clear the target by
    /// the clearance; across at what lands it, at most the cap.
    pub(super) fn aimed_velocity(&self, from: Vec3, target: Vec3, up: f32) -> Vec3 {
        let g = self.gravity;
        let dh = target.y - from.y;
        let needed = (2.0 * g * (dh + self.parkour.jump_clearance).max(0.05)).sqrt();
        let up = up.max(needed).min((2.0 * g * 1.8).sqrt());
        let t = (up + (up * up - 2.0 * g * dh).max(0.0).sqrt()) / g;
        let d = Vec3::new(target.x - from.x, 0.0, target.z - from.z);
        let speed = (d.length() / t.max(0.1)).min(self.parkour.max_jump_speed);
        d.normalize_or_zero() * speed + Vec3::Y * up
    }

    // ---- hanging ----

    /// The root's place and heading hanging from `edge` at `s`.
    fn hang_place(&self, pose: &HangPose, edge: usize, s: f32) -> (Vec3, f32) {
        let f = &self.features;
        let point = f.edges[edge].point(s);
        let normal = f.edge_normal(edge, s, 0.35);
        let root = from_edge_frame(point, normal, pose.root);
        let into = -normal;
        (root, into.x.atan2(into.z) + pose.yaw)
    }

    /// Whether the face under `edge` is tall enough to hang against and the feet have a wall.
    fn hangable(&self, edge: usize, pose: &HangPose) -> bool {
        let e = &self.features.edges[edge];
        e.a.y - e.floor >= self.parkour.min_hang_height.max(-pose.root.y + 0.25)
    }

    /// Catch an edge while falling: the root within reach of its hanging place below an edge, on
    /// the way down.
    pub(super) fn try_catch_edge(&mut self, db: &Database, world: &CollisionWorld, velocity: Vec3) -> bool {
        if self.features.edges.is_empty() || velocity.y > 1.0 {
            return false;
        }
        let Some(pose) = self.hang_pose_for(db, None) else { return false };
        let c = self.matcher.character().translation;
        let hands = c + Vec3::Y * (-pose.root.y);
        let Some((edge, s, _)) = self.features.nearest_edge(hands, 1.2) else { return false };
        if self.cooling(Catchable::Edge(edge)) || !self.hangable(edge, &pose) {
            return false;
        }
        let e = &self.features.edges[edge];
        let s = s.clamp(0.25, (e.length() - 0.25).max(0.25));
        let (root, _) = self.hang_place(&pose, edge, s);
        // in front of the face (not on top of the box), near the hanging place, and the hanging
        // place not below the floor
        let out = (c - e.point(s)).dot(e.normal);
        if out < -0.05 || c.distance(root) > self.parkour.catch_reach + velocity.length() * 0.05 || root.y < e.floor + 0.05 {
            return false;
        }
        // and the hands can take the edge from where the arms are
        let rest = self.rest_limbs(&pose, edge, s, world);
        if !self.arms_reach(db, [rest[0], rest[1]], -0.25) {
            return false;
        }
        self.start_hang(db, world, &pose, edge, s, true);
        true
    }

    /// Hang from `edge` at `s`: catching it (easing onto the hanging place), or straight on from
    /// a climb at its hanging frame.
    fn start_hang(&mut self, db: &Database, world: &CollisionWorld, pose: &HangPose, edge: usize, s: f32, catching: bool) {
        self.park.hang_pose = Some(*pose);
        let character = self.matcher.character();
        let start = if catching { (pose.frame - 4.0).max(pose.climb.anchor) } else { pose.frame };
        self.matcher.start_action(db, Action { clip: pose.climb.clip, start, exit: None, path: RootPath::External, collides: false, tag: ActionKind::Climb as u32, rate: 1.0, lock_feet: false });
        let braced = self.hangable(edge, pose);
        let mut hang = Hang { edge, s, time: 0.0, phase: 0.0, direction: 1.0, limbs: [Step::default(); 4], from: catching.then_some((character.translation, yaw_of(character.rotation))), settle: if catching { (pose.frame - start) / db.sample_rate } else { 0.0 }, braced, latch: None, start_frame: start };
        let rest = self.rest_limbs(pose, edge, s, world);
        for (k, l) in hang.limbs.iter_mut().enumerate() {
            l.planted = rest[k];
            l.from = rest[k];
        }
        if catching {
            // on the edge at once, the hips and the heading eased from where they show (the
            // catch's clip frames still play on to the hang)
            let (hips_before, rotation_before) = (self.matcher.joint_world(db, db.roles.hips), self.matcher.shown().rotation);
            let (root, yaw) = self.hang_place(pose, edge, s);
            hang.from = None;
            self.matcher.place(crate::animation::Transform::from_translation_rotation(root, yaw_rotation(yaw)));
            self.matcher.soften(0.15, 0.4);
            self.matcher.refresh(db);
            self.matcher.blend_position(hips_before - self.matcher.joint_world(db, db.roles.hips));
            self.matcher.blend_rotation(rotation_before);
        }
        self.park.hang = Some(hang);
        self.park.jump_target = None;
        self.state = CharacterState::Hanging;
        self.matcher.refresh(db);
    }

    /// `start_hang` for the tests: hang from `edge` at `s`, settled.
    #[cfg(test)]
    pub(super) fn start_hang_for_tests(&mut self, db: &Database, world: &CollisionWorld, pose: &HangPose, edge: usize, s: f32) {
        self.start_hang(db, world, pose, edge, s, false);
    }

    /// Hands and feet where the hang puts them with the root at `s` (world); the hands on the
    /// top, not in it (the clip's wrists sit at the ledge's height).
    fn rest_limbs(&self, pose: &HangPose, edge: usize, s: f32, world: &CollisionWorld) -> [Vec3; 4] {
        let f = &self.features;
        // (the frame's x is the character's left; the edge may run either way)
        let (side, _) = edge_frame(f.edges[edge].normal);
        let sign = f.edges[edge].direction().dot(side).signum();
        let at = |local: Vec3| {
            let (p, e) = f.edge_point(edge, s, local.x * sign);
            from_edge_frame(p, f.edges[e].normal, Vec3::new(0.0, local.y, local.z))
        };
        let hand = |p: Vec3| world.push_sphere_out(p, HAND_RADIUS, self.layers);
        [hand(at(pose.hands[0])), hand(at(pose.hands[1])), at(pose.feet[0]), at(pose.feet[1])]
    }

    pub(super) fn advance_hang(&mut self, db: &Database, world: &CollisionWorld, input: &MotionInput, dt: f32) {
        let Some(pose) = self.hang_pose_cached(db) else {
            self.state = CharacterState::Grounded;
            return;
        };
        let Some(mut h) = self.park.hang.take() else {
            self.state = CharacterState::Grounded;
            return;
        };
        h.time += dt;
        let settled = h.time >= h.settle;
        // the shimmy: the stick to the character's left or right, once settled. The way it goes
        // round the edges is latched when the stick is pushed, and kept while it stays pushed the
        // same way (round a corner the edge turns under the stick)
        let e = self.features.edges[h.edge];
        let stick = Vec3::new(input.velocity.x, 0.0, input.velocity.z);
        let (left, _) = edge_frame(self.features.edge_normal(h.edge, h.s, 0.35));
        let pushed = settled && stick.length() > 0.5;
        h.latch = match h.latch {
            Some((dir, sign)) if pushed && stick.normalize().dot(dir) > 0.5 => Some((dir, sign)),
            _ if pushed && stick.normalize().dot(left).abs() > 0.5 => {
                // which way along the edge the stick's side is
                let sign = (stick.dot(left) * e.direction().dot(left)).signum();
                Some((stick.normalize(), sign))
            }
            _ => None,
        };
        let wanted = h.latch.map_or(0.0, |(_, sign)| sign * (stick.length() / 2.0).min(1.0) * self.parkour.shimmy_speed);
        let stride = self.parkour.shimmy_stride;
        // the gait runs on while moving, and finishes a step under way once stopped
        let mid_step = h.limbs.iter().any(|l| l.stepping);
        let rate = if wanted.abs() > 0.05 { wanted.abs() / stride } else if mid_step { self.parkour.shimmy_speed / stride } else { 0.0 };
        if wanted.abs() > 0.05 && !mid_step {
            h.direction = wanted.signum();
        }
        let ds = if wanted.abs() > 0.05 { wanted * dt } else { 0.0 };
        let (edge, s) = self.features.advance_edge(h.edge, h.s, ds);
        // stopping at an end that turns no corner, short of the hands passing it
        let len = self.features.edges[edge].length();
        let (edge, s) = if (self.features.edges[edge].next.is_none() && s > len - 0.3) || (self.features.edges[edge].prev.is_none() && s < 0.3) { (h.edge, h.s.clamp(0.3, (len - 0.3).max(0.3))) } else { (edge, s) };
        h.edge = edge;
        h.s = s;
        h.phase = (h.phase + rate * dt).rem_euclid(1.0);
        let rest = self.rest_limbs(&pose, h.edge, h.s, world);
        // each limb steps in its quarter of the cycle: the leading hand, the trailing foot, the
        // trailing hand, the leading foot
        // (the edge frame's x, `side`, is the character's left, where hands[0] and feet[0] are)
        let (side, _) = edge_frame(self.features.edges[h.edge].normal);
        let left_ahead = side.dot(self.features.edges[h.edge].direction() * h.direction) > 0.0;
        let order = if left_ahead { [0usize, 3, 1, 2] } else { [1usize, 2, 0, 3] };
        // a planted limb the body has left out of its reach steps back under it at once
        let joints = self.hands.map(|hands| limb_joints(db, hands));
        for k in 0..4 {
            let reaches = joints.is_none_or(|j| self.limb_reaches(db, j[k], h.limbs[k].planted, 0.97));
            let l = &mut h.limbs[k];
            if l.forced.is_none() && !l.stepping && !reaches {
                l.forced = Some(0.0);
                l.from = l.planted;
            }
            if let Some(t) = l.forced.as_mut() {
                *t = (*t + dt / 0.22).min(1.0);
                let u = smooth(*t);
                let lift = (u * std::f32::consts::PI).sin() * if k < 2 { 0.05 } else { 0.08 };
                l.planted = l.from.lerp(rest[k], u) + Vec3::Y * lift + self.features.edges[h.edge].normal * lift * 0.6;
                if *t >= 1.0 {
                    l.forced = None;
                    l.planted = rest[k];
                }
            }
        }
        for (slot, &k) in order.iter().enumerate() {
            let window = (slot as f32 * 0.25, slot as f32 * 0.25 + 0.25);
            let inside = |p: f32| p >= window.0 && p < window.1;
            let l = &mut h.limbs[k];
            if l.forced.is_some() {
                continue;
            }
            // landing ahead of its rest by what the root moves while it stays planted, halved
            let target = rest[k] + self.features.edges[h.edge].direction() * h.direction * stride * 0.375 * (wanted.abs() > 0.05) as u32 as f32;
            if rate > 0.0 && inside(h.phase) {
                if !l.stepping {
                    l.stepping = true;
                    l.from = l.planted;
                }
                let t = smooth((h.phase - window.0) / 0.25);
                let lift = (t * std::f32::consts::PI).sin() * if k < 2 { 0.05 } else { 0.08 };
                let out = self.features.edges[h.edge].normal * lift * if k < 2 { 0.6 } else { 1.0 };
                l.planted = l.from.lerp(target, t) + Vec3::Y * lift + out;
            } else if l.stepping && (!inside(h.phase) || rate == 0.0) {
                l.stepping = false;
                l.planted = target;
            }
        }
        // the root: on the edge (eased from where a catch began), its breathing
        let (root, yaw) = self.hang_place(&pose, h.edge, h.s);
        let (root, yaw) = match h.from {
            Some((p, y)) if h.time < h.settle => {
                let t = smooth(h.time / h.settle.max(1e-3));
                (p.lerp(root, t), y + crate::animation::motion_matching::wrap_angle(yaw - y) * t)
            }
            _ => {
                h.from = None;
                (root, yaw)
            }
        };
        // swaying a little with the steps
        let sway = if rate > 0.0 { (h.phase * std::f32::consts::TAU * 2.0).sin() * 0.015 } else { 0.0 };
        let root = root + Vec3::Y * sway;
        let mut character = self.matcher.character();
        character.translation = root;
        character.rotation = yaw_rotation(yaw);
        self.matcher.place(character);
        // the clip: on to the hanging frame (catching), held there, then a frame either way of it,
        // slowly (breathing); the playback rate follows that frame
        let (_, frame) = self.matcher.playing();
        let wanted_frame = if h.time < h.settle {
            h.start_frame + (pose.frame - h.start_frame) * (h.time / h.settle.max(1e-3)).min(1.0)
        } else {
            let t = h.time - h.settle;
            pose.frame + (t * 1.3).sin() * 0.8 * smooth(t / 1.0)
        };
        let rate = ((wanted_frame - frame) / (dt * db.sample_rate)).clamp(-1.5, 1.5);
        if let Some(a) = self.matcher.action_mut() {
            a.rate = rate;
        }
        self.matcher.update(db, input, dt);
        // pinned: the hands always, the feet when braced (eased in as the catch settles)
        let w = if h.settle > 0.0 { smooth(h.time / h.settle) } else { 1.0 };
        let joints = self.hands.map(|hands| limb_joints(db, hands));
        self.park.limbs.clear();
        if let Some(joints) = joints {
            #[allow(clippy::needless_range_loop)]
            for k in 0..4 {
                if k >= 2 && !h.braced {
                    continue;
                }
                self.park.limbs.push(LimbTarget { joint: joints[k], target: h.limbs[k].planted, weight: w, rotation: None });
            }
        }
        self.park.hang = Some(h);
    }

    /// Climb up from the hang: the climb clip on from its hanging frame, warped onto the edge
    /// (refused at a corner, or with no room on top).
    fn climb_from_hang(&mut self, db: &Database, world: &CollisionWorld) -> Result<ActionKind, Refusal> {
        let (Some(pose), Some(h)) = (self.hang_pose_cached(db), self.park.hang.clone()) else { return Err(Refusal::Busy) };
        let e = self.features.edges[h.edge];
        if e.length() < 0.6 {
            return Err(Refusal::TooNarrow);
        }
        // (at a corner, onto the edge a little in from it: the warp eases there)
        let point = e.point(h.s.clamp(0.3, e.length() - 0.3));
        let into = -e.normal;
        let c = pose.climb;
        // room on top where the clip ends
        let character = self.matcher.character();
        let clip_heading = c.forward.x.atan2(c.forward.z);
        let (_, frame) = self.matcher.playing();
        let mut warp = RootWarp::identity(db.root_at(c.clip, frame), (character.translation, yaw_of(character.rotation)));
        warp.to = Placement::between((Vec3::new(c.ledge.x, 0.0, c.ledge.z), clip_heading), (Vec3::new(point.x, 0.0, point.z), into.x.atan2(into.z)));
        warp.window = Ramp::new(frame, frame + 6.0, 0.0, 1.0);
        // up to the edge's height by the end of the window, from where the hang has the root
        let lift = (point.y - c.height) - warp.ground;
        warp.lift.push(Ramp::new(frame, frame + 6.0, 0.0, lift));
        let end = warp.root(c.exit, db.root_at(c.clip, c.exit)).0;
        if !super::stands_at(world, end, &self.rules, self.layers) {
            return Err(Refusal::NoRoom);
        }
        if let Some(a) = self.matcher.action_mut() {
            *a = Action { clip: c.clip, start: frame, exit: Some(c.exit), path: RootPath::Warp(warp), collides: false, tag: ActionKind::Climb as u32, rate: 1.0, lock_feet: false };
        }
        self.park.hang = None;
        self.park.limbs.clear();
        self.state = CharacterState::Traversing(ActionKind::Climb);
        Ok(ActionKind::Climb)
    }

    /// A climb of a tagged ledge reaching the hanging frame: hang there.
    pub(super) fn climb_reached_hang(&mut self, db: &Database, world: &CollisionWorld) -> bool {
        if !self.park.hang_after {
            return false;
        }
        let (clip, frame) = self.matcher.playing();
        let Some(pose) = self.hang_pose_for(db, Some(clip)) else {
            self.park.hang_after = false;
            return false;
        };
        if frame < pose.frame {
            return false;
        }
        self.park.hang_after = false;
        let c = self.matcher.character().translation;
        let hands = c + Vec3::Y * (-pose.root.y);
        let Some((edge, s, _)) = self.features.nearest_edge(hands, 0.8) else { return false };
        let s = s.clamp(0.3, (self.features.edges[edge].length() - 0.3).max(0.3));
        // from the clip's own placement, eased onto the edge's
        self.start_hang(db, world, &pose, edge, s, false);
        if let Some(h) = self.park.hang.as_mut() {
            h.from = Some((c, yaw_of(self.matcher.character().rotation)));
            h.settle = 0.15;
        }
        true
    }

    /// The climb clip a climb of this obstacle uses ends in a hang (its ledge on a tagged edge
    /// tall enough).
    pub(super) fn ledge_is_hangable(&mut self, db: &Database, ledge: Vec3) -> bool {
        let Some(pose) = self.hang_pose_for(db, None) else { return false };
        self.features.nearest_edge(ledge, 0.25).is_some_and(|(e, _, _)| self.hangable(e, &pose))
    }

    // ---- wall runs ----

    /// A wall face to run along from here: on its side of it, within reach, moving along it fast
    /// enough and not straight at it. Starts the run when found.
    pub(super) fn try_wall_run(&mut self, db: &Database, from_ground: bool) -> bool {
        if self.features.faces.is_empty() {
            return false;
        }
        let c = self.matcher.character().translation;
        let v = self.matcher.simulation().velocity;
        let v = if from_ground && self.intent.length() > v.length() { self.intent.normalize_or_zero() * v.length().max(self.intent.length()) } else { v };
        let horizontal = Vec3::new(v.x, 0.0, v.z);
        let mut best: Option<(f32, usize, f32)> = None;
        for (k, f) in self.features.faces.iter().enumerate() {
            if self.cooling(Catchable::Face(k)) {
                continue;
            }
            let (out, along) = f.local(c);
            let speed = horizontal.dot(f.along);
            let reach = if from_ground { 1.1 } else { 0.8 };
            let angle_ok = horizontal.length() > 0.1 && speed.abs() >= horizontal.length() * 0.55;
            // on the wall's length (from its start), with room ahead to run along it
            let ahead = f.half_length - along * speed.signum();
            let on_it = along * speed.signum() >= -f.half_length + 0.1 && ahead >= 1.5;
            if out > 0.05 && out < reach && on_it && c.y >= f.bottom - 0.2 && c.y + 1.2 < f.top && speed.abs() >= self.parkour.wall_min_speed && angle_ok && best.is_none_or(|(o, _, _)| out < o) {
                best = Some((out, k, speed.signum()));
            }
        }
        let Some((out, face, sign)) = best else { return false };
        let Some(run) = self.run_loop_cached(db) else { return false };
        let f = self.features.faces[face];
        let (_, along) = f.local(c);
        let speed = (horizontal.dot(f.along).abs()).clamp(self.parkour.wall_speed.0, self.parkour.wall_speed.1);
        // the run loop at the frame whose pose is nearest the one showing
        let here = db.features(self.matcher.current_frame(db));
        let info = &db.clips[run.clip];
        let start = (0..info.playable()).min_by(|&a, &b| super::pose_distance(db.features(info.start + a), here).total_cmp(&super::pose_distance(db.features(info.start + b), here))).unwrap_or(0) as f32;
        self.matcher.start_action(db, Action { clip: run.clip, start, exit: None, path: RootPath::External, collides: false, tag: ActionKind::Jump as u32, rate: speed / run.speed, lock_feet: false });
        let vy = if from_ground { self.parkour.wall_up_speed } else { v.y.max(self.parkour.wall_up_speed * 0.6) };
        self.park.wall = Some(WallRun { face, sign, along, y: c.y, vy, speed, time: 0.0, from_out: out, entry: 0.15 + 0.15 * out, from_yaw: yaw_of(self.matcher.character().rotation), pins: [None; 2], weights: [0.0; 2] });
        self.park.jump_target = None;
        self.pending = 0.0;
        self.state = CharacterState::WallRunning;
        self.matcher.refresh(db);
        true
    }

    pub(super) fn advance_wall_run(&mut self, db: &Database, input: &MotionInput, dt: f32) {
        let Some(mut w) = self.park.wall.take() else {
            self.state = CharacterState::Grounded;
            return;
        };
        let s = self.parkour;
        let f = self.features.faces[w.face];
        w.time += dt;
        let gravity = self.gravity * if w.time < s.wall_light_time { s.wall_gravity.0 } else { s.wall_gravity.1 };
        w.along += w.sign * w.speed * dt;
        w.y += w.vy * dt - 0.5 * gravity * dt * dt;
        w.vy -= gravity * dt;
        // the ground under it ends the run (and so does the wall's end, or time)
        let out_of_wall = w.along.abs() > f.half_length - 0.35 || w.time > s.wall_max_time || w.y < f.bottom - 0.05 || (w.time > 0.3 && w.y <= f.bottom + 0.05);
        let entry = smooth(w.time / w.entry);
        let dir = f.along * w.sign;
        let yaw = dir.x.atan2(dir.z);
        let yaw = w.from_yaw + crate::animation::motion_matching::wrap_angle(yaw - w.from_yaw) * entry;
        // leaning out again before the run ends (the wall's end, its time, the ground), so the
        // body comes off it upright rather than unwinding in the air
        let to_end = ((f.half_length - 0.35 - w.along * w.sign) / w.speed).min(s.wall_max_time - w.time);
        let to_ground = if w.vy < 0.0 { (w.y - f.bottom) / -w.vy } else { f32::MAX };
        let exit = smooth(to_end.min(to_ground) / 0.25);
        let lean = s.wall_lean * entry * exit;
        let yaw_q = yaw_rotation(yaw);
        let n_local = yaw_q.conjugate() * f.normal;
        let rotation = yaw_q * Quat::from_rotation_z(-lean * n_local.x.signum());
        let out = w.from_out + (0.04 - w.from_out) * entry;
        let mut character = self.matcher.character();
        character.translation = f.point(w.along, w.y) + f.normal * out;
        character.rotation = rotation;
        self.matcher.place(character);
        self.matcher.update(db, input, dt);
        // the feet: planted ones pinned flat on the wall where they landed, swinging ones kept
        // out of it
        self.park.limbs.clear();
        let current = self.matcher.current_frame(db);
        let contacts = db.contacts(current);
        let up = rotation * Vec3::Y;
        let ankle = 0.1;
        #[allow(clippy::needless_range_loop)]
        for k in 0..2 {
            let joint = db.roles.feet[k];
            let animated = self.matcher.joint_world(db, joint);
            let (d, _) = f.local(animated);
            let on_wall = animated - f.normal * (d - ankle);
            let planted = contacts[k] && w.time > 0.12;
            w.weights[k] += ((planted as u32 as f32) - w.weights[k]) * (1.0 - (-dt / 0.02).exp());
            if planted {
                // pinned where it touched the wall, let go (and pinned again) once the leg would
                // have to stretch past the animation by more than a hand's width
                if w.pins[k].is_some_and(|p| p.distance(on_wall) > 0.12) {
                    w.pins[k] = None;
                }
                let pin = *w.pins[k].get_or_insert(on_wall);
                let flat = Quat::from_rotation_arc(up.normalize(), f.normal) * self.matcher.joint_world_rotation(db, joint);
                self.park.limbs.push(LimbTarget { joint, target: pin, weight: w.weights[k] * entry, rotation: Some(flat) });
            } else {
                w.pins[k] = None;
                let target = if d < ankle { on_wall } else { animated };
                let weight = if d < ankle { 1.0 } else { w.weights[k] };
                self.park.limbs.push(LimbTarget { joint, target, weight: weight * entry, rotation: None });
            }
        }
        // the arm on the wall's side swings into it, leaning: kept out of it
        if let Some(hands) = self.hands {
            for h in hands {
                let p = self.matcher.joint_world(db, h);
                let (d, _) = f.local(p);
                if d < HAND_RADIUS {
                    self.park.limbs.push(LimbTarget { joint: h, target: p + f.normal * (HAND_RADIUS - d), weight: entry, rotation: None });
                }
            }
        }
        self.park.wall = Some(w);
        if out_of_wall {
            self.leave_wall(db, false);
        }
    }

    /// Off the wall: falling on along it, or (`jump`) jumping away from it.
    fn leave_wall(&mut self, db: &Database, jump: bool) {
        let Some(w) = self.park.wall.take() else { return };
        let f = self.features.faces[w.face];
        let dir = f.along * w.sign;
        let s = self.parkour;
        let velocity = if jump { dir * w.speed * 0.7 + f.normal * s.wall_jump.0 + Vec3::Y * s.wall_jump.1 } else { dir * w.speed * 0.85 + f.normal * 1.0 + Vec3::Y * w.vy.max(-2.0) };
        self.park.cooldown = Some((Catchable::Face(w.face), s.cooldown));
        let flat = Vec3::new(velocity.x, 0.0, velocity.z);
        self.park.air_yaw = Some(flat.x.atan2(flat.z));
        self.park.limbs.clear();
        // the root (on the wall, at the feet) out to where the falling capsule clears the wall;
        // what shows stays put
        let mut character = self.matcher.character();
        let (out, _) = f.local(character.translation);
        let shift = f.normal * (self.radius + 0.05 - out).max(0.0);
        character.translation += shift;
        self.matcher.place(character);
        self.matcher.blend_position(-shift);
        self.fall_with(db, velocity);
        self.air_top = self.matcher.character().translation.y;
        self.state = CharacterState::Falling(0.0);
    }

    fn wall_jump(&mut self, db: &Database) {
        self.leave_wall(db, true);
    }

    // ---- bars ----

    /// Catch a bar in the air: the hands (the hang's height over the root) within reach of it.
    pub(super) fn try_catch_bar(&mut self, db: &Database, velocity: Vec3) -> bool {
        if self.features.bars.is_empty() {
            return false;
        }
        let Some(pose) = self.hang_pose_for(db, None) else { return false };
        self.park.hang_pose = Some(pose);
        let c = self.matcher.character().translation;
        let hands = c + Vec3::Y * (-pose.root.y);
        // with the bar clips: a bar the hands can reach from where the hips are (the clip's hips
        // hang that far under its hands), the hips under it; the swing starts at their angle
        // about it. Without them: the climb's hanging hands near a bar
        let clips = self.bar_clips_cached(db).zip(self.hands);
        let hips_now = self.matcher.joint_world(db, db.roles.hips);
        let mut found: Option<(usize, f32, f32)> = None;
        for (k, b) in self.features.bars.iter().enumerate() {
            let (d, s) = match clips {
                Some((clips, h)) => {
                    let (p, s) = b.nearest(hips_now);
                    let length = (clip_joint(db, clips.hang, 1.0, db.roles.hips) - hands_mid(db, clips.hang, 1.0, h)).length();
                    let r = hips_now - p;
                    let side = Vec3::Y.cross(yaw_rotation(yaw_of(self.matcher.character().rotation)) * FORWARD);
                    let grip = (side.dot(b.direction()) * 0.25).abs();
                    let fits = r.length() > length - 0.3 && r.length() < length + 0.15 && -r.y > 0.45 && self.arms_reach(db, [p - b.direction() * grip + Vec3::Y * 0.03, p + b.direction() * grip + Vec3::Y * 0.03], -0.15);
                    (if fits { (r.length() - length).abs() } else { f32::MAX }, s)
                }
                None => {
                    let (p, s) = b.nearest(hands);
                    (if p.distance(hands) < 0.45 && hands.y <= p.y + 0.35 { p.distance(hands) } else { f32::MAX }, s)
                }
            };
            if !self.cooling(Catchable::Bar(k)) && d < 1.0 && s > 0.25 && s < b.length() - 0.25 && found.is_none_or(|(_, _, best)| d < best) {
                found = Some((k, s, d));
            }
        }
        let Some((bar, s, _)) = found else { return false };
        let b = self.features.bars[bar];
        // facing across the bar, the way the character moves (else faces)
        let across = Vec3::Y.cross(b.direction());
        let facing = self.matcher.character().rotation * FORWARD;
        let moving = Vec3::new(velocity.x, 0.0, velocity.z);
        let ahead = if moving.length() > 0.5 { moving } else { facing };
        let into = if ahead.dot(across) >= 0.0 { across } else { -across };
        let yaw = into.x.atan2(into.z);
        // the swing's start: the angle of the root about the bar, and the speed across it
        let omega = (moving.dot(into) / self.parkour.swing_length).clamp(-4.0, 4.0);
        let start = (pose.frame - 3.0).max(pose.climb.anchor);
        self.matcher.start_action(db, Action { clip: pose.climb.clip, start, exit: None, path: RootPath::External, collides: false, tag: ActionKind::Jump as u32, rate: 1.0, lock_feet: false });
        let character = self.matcher.character();
        self.park.swing = Some(Swing { bar, s, theta: 0.0, omega, yaw, time: 0.0, from: Some((character.translation, character.rotation)), swinging: false });
        if let (Some(clips), Some(hands)) = (self.bar_clips_cached(db), self.hands) {
            self.matcher.start_action(db, Action { clip: clips.hang, start: 1.0, exit: None, path: RootPath::External, collides: false, tag: ActionKind::BarHang as u32, rate: 1.0, lock_feet: false });
            // placed on the bar at once; the hips eased there from where they show (the clip's
            // root is on the floor under the bar, the fall's at the feet)
            let hips_before = hips_now;
            let rotation_before = self.matcher.shown().rotation;
            let mut w = self.park.swing.take().unwrap();
            w.from = None;
            // the swing from the hips' angle about the bar, at their speed round it
            let pivot = b.a + b.direction() * s;
            let r = hips_now - pivot;
            let into = yaw_rotation(w.yaw) * FORWARD;
            w.theta = r.dot(into).atan2(-r.y);
            let tangent = into * w.theta.cos() + Vec3::Y * w.theta.sin();
            w.omega = (velocity.dot(tangent) / r.length().max(0.3)).clamp(-5.0, 5.0);
            let (root, rotation) = self.swing_clip_place(db, &clips, hands, &mut w, 1e-3);
            // (the clip placed with may have switched to the swing: the hips of what plays)
            self.matcher.reroot(db);
            let (clip, frame) = self.matcher.playing();
            let hips_after = root + rotation * clip_joint(db, clip, frame, db.roles.hips);
            let mut character = self.matcher.character();
            character.translation = root;
            character.rotation = rotation;
            self.matcher.place(character);
            self.matcher.soften(0.18, 0.5);
            self.matcher.refresh(db);
            self.matcher.blend_position(hips_before - hips_after);
            self.matcher.blend_rotation(rotation_before);
            self.park.swing = Some(w);
        }
        self.park.jump_target = None;
        self.state = CharacterState::Swinging;
        true
    }

    /// The root's place and rotation swinging at `theta` from bar `bar` at `s`, facing `yaw`.
    fn swing_place(&self, pose: &HangPose, bar: usize, s: f32, theta: f32, yaw: f32) -> (Vec3, Quat) {
        let b = self.features.bars[bar];
        let pivot = b.a + b.direction() * s;
        let into = yaw_rotation(yaw) * FORWARD;
        // the hang's root relative to the hands' line (no wall: the body hangs under the bar)
        let local = Vec3::new(0.0, pose.root.y, pose.root.z * 0.3);
        let (side, _) = edge_frame(-into);
        let pitch = Quat::from_axis_angle(side, -theta);
        let offset = pitch * (into * local.z + Vec3::Y * local.y);
        (pivot + offset, pitch * yaw_rotation(yaw + pose.yaw))
    }

    pub(super) fn advance_swing(&mut self, db: &Database, input: &MotionInput, dt: f32) {
        let (Some(pose), Some(mut w)) = (self.hang_pose_cached(db), self.park.swing.take()) else {
            self.state = CharacterState::Grounded;
            return;
        };
        let s = self.parkour;
        w.time += dt;
        let into = yaw_rotation(w.yaw) * FORWARD;
        let stick = Vec3::new(input.velocity.x, 0.0, input.velocity.z).dot(into) / 2.0;
        // the pendulum, substepped
        let steps = ((dt / (1.0 / 240.0)).ceil() as usize).clamp(1, 32);
        let h = dt / steps as f32;
        for _ in 0..steps {
            let pump = if w.omega.abs() > 0.05 { s.swing_pump * stick.clamp(-1.0, 1.0) * w.omega.signum() * (stick * w.omega).signum().max(0.0) } else { s.swing_pump * stick.clamp(-1.0, 1.0) * 0.5 };
            let accel = -(self.gravity / s.swing_length) * w.theta.sin() - s.swing_damping * w.omega + pump;
            w.omega += accel * h;
            w.theta += w.omega * h;
            if w.theta.abs() > s.swing_max {
                w.theta = w.theta.clamp(-s.swing_max, s.swing_max);
                w.omega = 0.0;
            }
        }
        let clips = self.bar_clips_cached(db);
        let (root, rotation) = match (clips, self.hands) {
            (Some(clips), Some(hands)) => self.swing_clip_place(db, &clips, hands, &mut w, dt),
            _ => self.swing_place(&pose, w.bar, w.s, w.theta, w.yaw),
        };
        let settle = 0.12;
        let (root, rotation) = match w.from {
            Some((p, r)) if w.time < settle => {
                let t = smooth(w.time / settle);
                (p.lerp(root, t), r.slerp(rotation, t))
            }
            _ => {
                w.from = None;
                (root, rotation)
            }
        };
        let mut character = self.matcher.character();
        character.translation = root;
        character.rotation = rotation;
        self.matcher.place(character);
        if clips.is_none() {
            let (_, frame) = self.matcher.playing();
            let rate = if frame < pose.frame { 1.0 } else { 0.0 };
            if let Some(a) = self.matcher.action_mut() {
                a.rate = rate;
            }
        }
        self.matcher.update(db, input, dt);
        self.park.limbs.clear();
        if let Some(hands) = self.hands {
            let b = self.features.bars[w.bar];
            let w8 = smooth(w.time / settle);
            let side = Vec3::Y.cross(into);
            #[allow(clippy::needless_range_loop)]
            for k in 0..2 {
                let along = (w.s + side.dot(b.direction()) * pose.hands[k].x).clamp(0.0, b.length());
                let target = b.a + b.direction() * along + Vec3::Y * 0.03;
                self.park.limbs.push(LimbTarget { joint: hands[k], target, weight: w8, rotation: None });
            }
        }
        self.park.swing = Some(w);
    }

    /// With bar clips: the clip that fits the swing (still or swinging, switched with some
    /// hysteresis), its frame from the pendulum's phase, the root placed so the clip's hands are
    /// on the bar, and turned about the bar by what the pendulum swings beyond the clip.
    fn swing_clip_place(&mut self, db: &Database, clips: &BarClips, hands: [usize; 2], w: &mut Swing, dt: f32) -> (Vec3, Quat) {
        let omega0 = (self.gravity / self.parkour.swing_length).sqrt();
        let amplitude = (w.theta * w.theta + (w.omega / omega0).powi(2)).sqrt();
        let phase = w.theta.atan2(w.omega / omega0);
        let swing = clips.swing.filter(|_| if w.swinging { amplitude > 0.08 } else { amplitude > 0.14 });
        if swing.is_some() != w.swinging {
            w.swinging = swing.is_some();
            let (clip, start, kind) = match swing {
                Some((clip, period, _, origin)) => (clip, origin + phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * period, ActionKind::BarSwing),
                None => (clips.hang, 1.0, ActionKind::BarHang),
            };
            self.matcher.start_action(db, Action { clip, start, exit: None, path: RootPath::External, collides: false, tag: kind as u32, rate: 1.0, lock_feet: false });
        }
        let (clip, frame) = self.matcher.playing();
        let info = &db.clips[clip];
        // the frame to be at: the pendulum's phase through one of the clip's swings, or (still)
        // to and fro over the clip's middle
        let (wanted, clip_angle) = match swing {
            Some((_, period, angle, origin)) => {
                let f = origin + phase.rem_euclid(std::f32::consts::TAU) / std::f32::consts::TAU * period;
                // the nearest copy of that phase to where it plays
                let k = ((frame - f) / period).round();
                (f + k * period, angle * phase.sin() * (amplitude / angle).min(1.0))
            }
            None => {
                let span = (info.frames as f32 - 2.0).max(1.0);
                let t = (w.time * db.sample_rate / span).rem_euclid(2.0);
                (1.0 + span * (1.0 - (t - 1.0).abs()), 0.0)
            }
        };
        let wanted = wanted.clamp(0.0, (info.frames - 1) as f32);
        let rate = ((wanted - frame) / (dt * db.sample_rate)).clamp(-2.0, 2.0);
        if let Some(a) = self.matcher.action_mut() {
            a.rate = rate;
        }
        let _ = clip_angle;
        let b = self.features.bars[w.bar];
        let pivot = b.a + b.direction() * w.s;
        let heading = yaw_rotation(w.yaw);
        let mid = hands_mid(db, clip, frame, hands);
        let root = pivot - heading * mid;
        // the angle the clip's own body hangs at, this frame: the pendulum turns it the rest
        let hips = clip_joint(db, clip, frame, db.roles.hips);
        let clip_angle = (hips.z - mid.z).atan2(mid.y - hips.y);
        let into = heading * FORWARD;
        let (side, _) = edge_frame(-into);
        let turn = Quat::from_axis_angle(side, -(w.theta - clip_angle));
        (pivot + turn * (root - pivot), turn * heading)
    }

    /// Let go of the bar: with the swing's speed (`fling`, a little up), or straight down.
    fn release_bar(&mut self, db: &Database, fling: bool) {
        let Some(w) = self.park.swing.take() else { return };
        let into = yaw_rotation(w.yaw) * FORWARD;
        let l = self.parkour.swing_length;
        // the body's velocity on the arc: tangent to it
        let tangent = into * w.theta.cos() + Vec3::Y * w.theta.sin();
        let velocity = if fling { tangent * (w.omega * l) + Vec3::Y * 1.5 + into * 1.0 } else { Vec3::ZERO };
        self.park.cooldown = Some((Catchable::Bar(w.bar), self.parkour.cooldown));
        let flat = Vec3::new(velocity.x, 0.0, velocity.z);
        self.park.air_yaw = Some(if flat.length() > 0.3 { flat.x.atan2(flat.z) } else { w.yaw });
        self.park.limbs.clear();
        self.fall_with(db, velocity);
        self.air_top = self.matcher.character().translation.y;
        self.state = CharacterState::Falling(0.0);
    }

    // ---- in the air ----

    /// While falling: come upright and turn toward the heading asked for; tick the cooldown.
    pub(super) fn air_turn(&mut self, db: &Database, dt: f32) {
        let mut character = self.matcher.character();
        let yaw = self.park.air_yaw.unwrap_or_else(|| yaw_of(character.rotation));
        let wanted = yaw_rotation(yaw);
        // (slow enough that a lean off a wall or a swing's pitch unwinds without a jerk)
        let t = 1.0 - (-dt / 0.18).exp();
        let rotation = character.rotation.slerp(wanted, t).normalize();
        if rotation.angle_between(character.rotation) > 1e-5 {
            // turned about the hips, not the feet: the body doesn't swing round its root
            let hips = self.hips_local(db);
            character.translation += character.rotation * hips - rotation * hips;
            character.rotation = rotation;
            self.matcher.place(character);
        }
    }

    /// Whether limb end `joint` (a hand or a foot) can reach `target` from where its root joint
    /// (shoulder, hip) is: within `fraction` of the limb's length.
    fn limb_reaches(&self, db: &Database, joint: usize, target: Vec3, fraction: f32) -> bool {
        let (Some(middle), Some(upper)) = (db.skeleton.parents[joint], db.skeleton.parents[joint].and_then(|m| db.skeleton.parents[m])) else { return true };
        let (e, m, u) = (self.matcher.joint_world(db, joint), self.matcher.joint_world(db, middle), self.matcher.joint_world(db, upper));
        target.distance(u) <= (e.distance(m) + m.distance(u)) * fraction
    }

    /// Whether both hands could reach these points from where the shoulders are (the arms'
    /// length less `slack`), either hand to either point.
    fn arms_reach(&self, db: &Database, points: [Vec3; 2], slack: f32) -> bool {
        let Some(hands) = self.hands else { return true };
        let reach = |hand: usize, p: Vec3| {
            let (Some(elbow), Some(shoulder)) = (db.skeleton.parents[hand], db.skeleton.parents[hand].and_then(|e| db.skeleton.parents[e])) else { return true };
            let (h, e, sh) = (self.matcher.joint_world(db, hand), self.matcher.joint_world(db, elbow), self.matcher.joint_world(db, shoulder));
            p.distance(sh) <= (h.distance(e) + e.distance(sh)) * 0.985 - slack
        };
        (reach(hands[0], points[0]) && reach(hands[1], points[1])) || (reach(hands[0], points[1]) && reach(hands[1], points[0]))
    }

    /// The hips relative to the root, as shown.
    fn hips_local(&self, db: &Database) -> Vec3 {
        let model = self.matcher.model();
        let hips = self.matcher.hips_joint(db);
        model.get(hips).map_or(Vec3::Y, |t| t.translation)
    }

    pub(super) fn tick_cooldown(&mut self, dt: f32) {
        if let Some((_, t)) = self.park.cooldown.as_mut() {
            *t -= dt;
        }
    }

    /// Pin the limbs this frame's move asked for, and keep every hand and foot out of the level
    /// whatever the character does off the ground (a vault's hand sinking into the box, a foot
    /// into the wall it leaves, a landing foot into a ledge's face): a limb that would end inside
    /// something is pushed out to a clearance, at once, and eased back onto the animation once
    /// clear.
    pub(super) fn apply_limbs(&mut self, db: &Database, world: &CollisionWorld, dt: f32) {
        if self.state == CharacterState::Grounded {
            self.park.limbs.clear();
            self.park.limb_out = Default::default();
            return;
        }
        if !matches!(self.state, CharacterState::Hanging | CharacterState::WallRunning | CharacterState::Swinging) {
            self.park.limbs.clear();
        }
        let mut targets = std::mem::take(&mut self.park.limbs);
        let ends = self.hands.map_or([None, None, Some(db.roles.feet[0]), Some(db.roles.feet[1])], |h| [Some(h[0]), Some(h[1]), Some(db.roles.feet[0]), Some(db.roles.feet[1])]);
        for (k, joint) in ends.into_iter().enumerate() {
            let Some(joint) = joint else { continue };
            let radius = if k < 2 { HAND_RADIUS } else { FOOT_RADIUS };
            let animated = self.matcher.joint_world(db, joint);
            let at = targets.iter().position(|t| t.joint == joint);
            let p = at.map_or(animated, |i| animated.lerp(targets[i].target, targets[i].weight.min(1.0)));
            let out = world.push_sphere_out(p, radius, self.layers);
            let push = out - p;
            let state = &mut self.park.limb_out[k];
            if push.length_squared() > 1e-8 {
                // what an eased release left still applies, plus this frame's push
                *state = (1.0, push);
            } else {
                state.0 *= (-dt / 0.06).exp();
            }
            if state.0 > 1e-3 {
                let target = p + state.1 * state.0;
                // (still clear of the level)
                let target = world.push_sphere_out(target, radius, self.layers);
                match at {
                    Some(i) => {
                        targets[i].target = target;
                        targets[i].weight = 1.0;
                    }
                    None => targets.push(LimbTarget { joint, target, weight: 1.0, rotation: None }),
                }
            }
        }
        // hanging or swinging, the body hangs from its hands: pulled up to them where the arms
        // would have to stretch past their length (a catch's eased body, a hand round a corner)
        let hanging = matches!(self.state, CharacterState::Hanging | CharacterState::Swinging);
        if hanging {
            let mut pull = Vec3::ZERO;
            for t in targets.iter().filter(|t| t.weight > 0.5 && self.hands.is_some_and(|h| h.contains(&t.joint))) {
                let (Some(elbow), Some(shoulder)) = (db.skeleton.parents[t.joint], db.skeleton.parents[t.joint].and_then(|e| db.skeleton.parents[e])) else { continue };
                let (h, e, sh) = (self.matcher.joint_world(db, t.joint), self.matcher.joint_world(db, elbow), self.matcher.joint_world(db, shoulder));
                let reach = (h.distance(e) + e.distance(sh)) * 0.985;
                let d = t.target.distance(sh);
                if d > reach {
                    let need = (t.target - sh).normalize_or_zero() * (d - reach);
                    // the larger of the hands' needs, along each axis
                    pull = Vec3::new(
                        if need.x.abs() > pull.x.abs() { need.x } else { pull.x },
                        if need.y.abs() > pull.y.abs() { need.y } else { pull.y },
                        if need.z.abs() > pull.z.abs() { need.z } else { pull.z },
                    );
                }
            }
            // what the arms still need, on top of the pull already showing; let go of slowly
            // (half a second's half-life) when they don't need it
            let held = self.matcher.reach_offset();
            // (no faster than the body can be seen to be pulled: a catch's last few centimetres
            // close over a few frames)
            let next = if pull.length_squared() > 1e-8 { held + pull.clamp_length_max(if self.state == CharacterState::Swinging { 2.0 } else { 6.0 } * dt) } else { held * (-dt / 0.15).exp() };
            self.matcher.set_reach_offset(next);
        } else if self.matcher.reach_offset() != Vec3::ZERO {
            let held = self.matcher.reach_offset();
            self.matcher.set_reach_offset(if held.length() < 1e-4 { Vec3::ZERO } else { held * (-dt / 0.1).exp() });
        }
        // feet braced on a face or running on a wall: a target past the leg's length comes in
        // toward the hip (the foot lifts off the wall a little rather than the leg failing to
        // reach)
        if hanging || self.state == CharacterState::WallRunning {
            for t in targets.iter_mut().filter(|t| db.roles.feet.contains(&t.joint)) {
                let (Some(knee), Some(hip)) = (db.skeleton.parents[t.joint], db.skeleton.parents[t.joint].and_then(|k| db.skeleton.parents[k])) else { continue };
                let (f, k, h) = (self.matcher.joint_world(db, t.joint), self.matcher.joint_world(db, knee), self.matcher.joint_world(db, hip));
                let reach = (f.distance(k) + k.distance(h)) * 0.97;
                if t.target.distance(h) > reach {
                    t.target = h + (t.target - h).normalize_or_zero() * reach;
                }
            }
        }
        if !targets.is_empty() {
            self.matcher.reach(db, &targets);
        }
        self.park.limbs = targets;
    }
}

/// The parkour geometry a controller uses (set `CharacterController::features`).
pub type Features = ParkourFeatures;
