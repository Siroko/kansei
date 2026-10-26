use glam::{Quat, Vec3};

use super::database::{wrap_angle, yaw_of, yaw_rotation, Database, FEATURES, FORWARD, STRIDE, TRAJECTORY_TIMES};
use super::search::{Match, SearchFilter};
use crate::animation::ik::{two_joint_ik, FootLock, FootPose};
use crate::animation::inertialization::Inertializer;
use crate::animation::springs::{damper_exact, negexp, spring_character_update, spring_damper_exact_quat};
use crate::animation::retarget::Retarget;
use crate::animation::warping::RootWarp;
use crate::animation::{quat_abs, quat_from_scaled_angle_axis, quat_to_scaled_angle_axis, Pose, Skeleton, Transform};

/// Tuning of a `MotionMatcher`. The defaults follow Holden's reference controller at 30 Hz data.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionMatchingSettings {
    /// Seconds between searches (a change of input searches at once).
    pub search_interval: f32,
    /// A search switches only to a frame that costs less than the frame playing minus
    /// `continuing_bias` (in squared normalized feature units). Databases cut from the same takes
    /// hold many copies of the same frames (a loop's cycle at the start of a stop, a turn, a
    /// pivot); without a bias the search hops between them.
    pub continuing_bias: f32,
    /// Search now when the desired velocity changed by this much (m/s) or the desired facing by
    /// this much (radians) since the last search.
    pub force_search_velocity: f32,
    pub force_search_turn: f32,
    pub filter: SearchFilter,
    /// Half-life of the transition offsets (seconds).
    pub inertialization_halflife: f32,
    /// Half-lives of the simulated character's velocity and facing springs (seconds).
    pub velocity_halflife: f32,
    pub rotation_halflife: f32,
    /// How the animated character is pulled toward the simulation: a damper of this half-life,
    /// limited (when `adjust_by_velocity`) to `max_adjustment_ratio` of the character's own speed,
    /// so a standing character does not slide; then clamped to `clamp_distance` metres and
    /// `clamp_angle` radians.
    pub adjustment_halflife: f32,
    pub adjust_by_velocity: bool,
    pub max_adjustment_ratio: f32,
    pub clamp_distance: f32,
    pub clamp_angle: f32,
    /// Foot locking: pin planted feet with two-joint IK. A pin lets go when the contact ends or
    /// the animated foot strays `foot_unlock_radius` metres from it; the foot then fades back
    /// onto the animation with `foot_lock_halflife`, and is pinned again there while the contact
    /// lasts.
    pub foot_lock: bool,
    pub foot_unlock_radius: f32,
    pub foot_lock_halflife: f32,
}

impl Default for MotionMatchingSettings {
    fn default() -> Self {
        Self {
            search_interval: 0.1,
            continuing_bias: 0.01,
            force_search_velocity: 0.5,
            force_search_turn: 0.35,
            filter: SearchFilter::default(),
            inertialization_halflife: 0.1,
            velocity_halflife: 0.27,
            rotation_halflife: 0.27,
            adjustment_halflife: 0.1,
            adjust_by_velocity: true,
            max_adjustment_ratio: 0.5,
            clamp_distance: 0.15,
            clamp_angle: std::f32::consts::FRAC_PI_2,
            foot_lock: true,
            foot_unlock_radius: 0.3,
            foot_lock_halflife: 0.1,
        }
    }
}

/// What the player asks for this frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionInput {
    /// Desired velocity in world space (m/s; y ignored).
    pub velocity: Vec3,
    /// Desired facing (yaw about +Y, radians) for strafing; `None` faces the way it moves.
    pub facing: Option<f32>,
}

/// The simulated character: where the input says it should be, as springs on velocity and facing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Simulation {
    pub position: Vec3,
    pub velocity: Vec3,
    pub acceleration: Vec3,
    pub rotation: Quat,
    pub angular_velocity: Vec3,
}

/// The last search: whether one ran, whether it switched, the frame it chose and its cost.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SearchInfo {
    pub searched: bool,
    pub switched: bool,
    pub frame: usize,
    pub cost: f32,
}

/// A character with its own proportions shown instead of the database's skeleton (see
/// `MotionMatcher::set_display`).
#[derive(Debug, Clone)]
struct Display {
    skeleton: Skeleton,
    retarget: Retarget,
    legs: [Leg; 2],
}

/// The horizontal direction a move from `from` toward `wanted` was blocked in, when it ended at
/// `allowed`: what was taken off the move, less any part of it along the move that was allowed
/// (a slide along a wall, stopped a hair short of it, loses a little of both).
fn blocked_direction(from: Vec3, wanted: Vec3, allowed: Vec3) -> Vec3 {
    let flat = |v: Vec3| Vec3::new(v.x, 0.0, v.z);
    let removed = flat(wanted - allowed);
    let along = flat(allowed - from).normalize_or_zero();
    (removed - along * removed.dot(along)).normalize_or(removed.normalize_or_zero())
}

/// A leg for foot locking: its (upper, middle, foot) joints, the foot's ball (its first child,
/// or the foot itself), and the heights of the foot and the ball at rest.
#[derive(Debug, Clone, Copy)]
struct Leg {
    joints: [usize; 3],
    ball: usize,
    rest: [f32; 2],
}

/// The leg ending at `foot`.
fn leg(skeleton: &Skeleton, foot: usize) -> Leg {
    let middle = skeleton.parents[foot].unwrap_or(foot);
    let ball = skeleton.parents.iter().position(|p| *p == Some(foot)).unwrap_or(foot);
    let rest = skeleton.rest_model();
    Leg { joints: [skeleton.parents[middle].unwrap_or(middle), middle, foot], ball, rest: [rest[foot].translation.y, rest[ball].translation.y] }
}

/// How the character moves while an action plays.
#[derive(Debug, Clone, PartialEq)]
pub enum RootPath {
    /// The clip's root motion placed in the world by a warp (traversal, landing).
    Warp(RootWarp),
    /// Momentum and gravity (jumping, falling); the clip only animates the pose.
    Ballistic { velocity: Vec3, gravity: f32 },
}

/// A clip played on command instead of found by the search: a traversal, a fall, a landing.
#[derive(Debug, Clone, PartialEq)]
pub struct Action {
    /// Database clip, and the clip frame it starts at.
    pub clip: usize,
    pub start: f32,
    /// Clip frame at which motion matching takes over again; `None` plays until
    /// `MotionMatcher::stop_action` (a looping clip goes round).
    pub exit: Option<f32>,
    pub path: RootPath,
    /// Whether the path is kept out of the world like the simulation (`update_constrained`):
    /// yes for jumps, falls and landings; not for traversals, which go over what they cross.
    pub collides: bool,
    /// For the caller: what the action is (e.g. a traversal kind).
    pub tag: u32,
}

/// A character driven by motion matching: every few frames it searches the database for the
/// frame whose pose and future trajectory best match its current pose and the trajectory the
/// input predicts, switches there with inertialization, and plays on, moved by the clips' root
/// motion, pulled toward a spring-simulated position, its planted feet pinned by IK.
pub struct MotionMatcher {
    pub settings: MotionMatchingSettings,
    clip: usize,
    /// Playhead in frames of `clip`.
    frame: f32,
    /// The database's pose at the playhead, the pose shown (inertialized, on the database's
    /// skeleton), and the output: that pose on the display skeleton if any, feet locked.
    sampled: Pose,
    matched: Pose,
    pose: Pose,
    model: Vec<Transform>,
    display: Option<Display>,
    /// The character root in the world: position and heading.
    character: Transform,
    simulation: Simulation,
    desired_yaw: f32,
    trajectory: [Transform; 3],
    inertializer: Inertializer,
    search_timer: f32,
    searched_input: Option<(Vec3, f32)>,
    feet: [FootLock; 2],
    legs: [Leg; 2],
    last_search: SearchInfo,
    /// The root motion's speed (m/s) and turn rate (rad/s) this frame.
    root_speed: (f32, f32),
    /// The clip playing on command, if any, instead of searching.
    action: Option<Action>,
    /// Height of the ground the character stands on while matching.
    ground: f32,
    scratch: (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Vec<Vec3>, Pose),
}

impl MotionMatcher {
    /// A character standing at `position` facing `yaw`, on the database's first frame until the
    /// first update searches.
    pub fn new(db: &Database, settings: MotionMatchingSettings, position: Vec3, yaw: f32) -> Self {
        let mut sampled = Pose { local: Vec::new() };
        db.pose(0, 0, 0.0, &mut sampled);
        let character = Transform::from_translation_rotation(position, yaw_rotation(yaw));
        let mut matcher = Self {
            settings,
            clip: 0,
            frame: 0.0,
            pose: sampled.clone(),
            matched: sampled.clone(),
            sampled,
            model: Vec::new(),
            display: None,
            character,
            simulation: Simulation { position, velocity: Vec3::ZERO, acceleration: Vec3::ZERO, rotation: character.rotation, angular_velocity: Vec3::ZERO },
            desired_yaw: yaw,
            trajectory: [character; 3],
            inertializer: Inertializer::new(db.joint_count()),
            search_timer: 0.0,
            searched_input: None,
            feet: Default::default(),
            legs: [leg(&db.skeleton, db.roles.feet[0]), leg(&db.skeleton, db.roles.feet[1])],
            last_search: SearchInfo::default(),
            root_speed: (0.0, 0.0),
            action: None,
            ground: position.y,
            scratch: (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Pose { local: Vec::new() }),
        };
        matcher.pose.to_model(&db.skeleton, &mut matcher.model);
        matcher
    }

    /// Show the animation on another skeleton with the same joint names and axes but its own
    /// proportions (`retarget` from the database's skeleton to `skeleton`); `None` shows the
    /// database's. Foot locking then works on that skeleton's legs.
    pub fn set_display(&mut self, db: &Database, display: Option<(Skeleton, Retarget)>) {
        self.display = display.map(|(skeleton, retarget)| {
            let foot = |j: usize| skeleton.find(&db.skeleton.names[j]).unwrap_or(0);
            let legs = [leg(&skeleton, foot(db.roles.feet[0])), leg(&skeleton, foot(db.roles.feet[1]))];
            Display { skeleton, retarget, legs }
        });
        self.feet.iter_mut().for_each(FootLock::reset);
        self.output(db);
    }

    /// The skeleton the output pose is on.
    pub fn output_skeleton<'a>(&'a self, db: &'a Database) -> &'a Skeleton {
        self.display.as_ref().map_or(&db.skeleton, |d| &d.skeleton)
    }

    /// The output pose (local, on `output_skeleton`; the root joint relative to `character`).
    pub fn pose(&self) -> &Pose {
        &self.pose
    }

    /// The output pose in model space (relative to `character`, on `output_skeleton`).
    pub fn model(&self) -> &[Transform] {
        &self.model
    }

    /// The character root in the world (position and heading): the mesh's placement.
    pub fn character(&self) -> Transform {
        self.character
    }

    pub fn simulation(&self) -> &Simulation {
        &self.simulation
    }

    /// The simulation's predicted root at `TRAJECTORY_TIMES` ahead (world).
    pub fn trajectory(&self) -> &[Transform; 3] {
        &self.trajectory
    }

    /// The clip playing and the playhead in its frames.
    pub fn playing(&self) -> (usize, f32) {
        (self.clip, self.frame)
    }

    /// The database frame at the playhead.
    pub fn current_frame(&self, db: &Database) -> usize {
        db.clips[self.clip].start + (self.frame.round() as usize).min(db.clips[self.clip].frames - 1)
    }

    pub fn last_search(&self) -> SearchInfo {
        self.last_search
    }

    /// Whether each foot is pinned.
    pub fn feet_locked(&self) -> [bool; 2] {
        [self.feet[0].is_locked(), self.feet[1].is_locked()]
    }

    /// The action playing, if any.
    pub fn action(&self) -> Option<&Action> {
        self.action.as_ref()
    }

    /// The action playing, to change its path or exit as it plays (a jump leaving the ground).
    pub fn action_mut(&mut self) -> Option<&mut Action> {
        self.action.as_mut()
    }

    /// Play `action` now instead of searching, inertializing from what shows.
    pub fn start_action(&mut self, db: &Database, action: Action) {
        let from = self.current_frame(db);
        let to = db.clips[action.clip].start + (action.start.round() as usize).min(db.clips[action.clip].frames - 1);
        self.transition(db, from, to);
        self.frame = action.start;
        self.feet.iter_mut().for_each(FootLock::reset);
        self.action = Some(action);
    }

    /// End the action: motion matching searches again at the next update.
    pub fn stop_action(&mut self) {
        if self.action.take().is_some() {
            self.searched_input = None;
            self.simulation.position = self.character.translation;
            self.simulation.rotation = self.character.rotation;
            self.simulation.angular_velocity = Vec3::ZERO;
            self.desired_yaw = yaw_of(self.character.rotation);
        }
    }

    /// Put the character here (keeping its pose and what plays), its simulation with it.
    pub fn place(&mut self, character: Transform) {
        self.character = character;
        self.simulation.position = character.translation;
    }

    /// The height of the ground under the character, which it stands on while matching (an
    /// action's root path decides its height itself).
    pub fn set_ground(&mut self, height: f32) {
        self.ground = height;
    }

    pub fn ground(&self) -> f32 {
        self.ground
    }

    /// Move the character (and its simulation) without blending, facing `yaw`.
    pub fn teleport(&mut self, position: Vec3, yaw: f32) {
        self.character = Transform::from_translation_rotation(position, yaw_rotation(yaw));
        self.simulation = Simulation { position, velocity: Vec3::ZERO, acceleration: Vec3::ZERO, rotation: self.character.rotation, angular_velocity: Vec3::ZERO };
        self.desired_yaw = yaw;
        self.inertializer.reset();
        self.feet.iter_mut().for_each(FootLock::reset);
        self.searched_input = None;
        self.action = None;
        self.ground = position.y;
    }

    /// Advance by `dt` seconds under `input`.
    pub fn update(&mut self, db: &Database, input: &MotionInput, dt: f32) {
        self.update_constrained(db, input, dt, &mut |_, to| to);
    }

    /// `update`, with `constrain(from, to)` saying where a move from `from` toward `to` may end
    /// (the world's collision): it shapes the simulated position, its predicted trajectory (so
    /// the search sees the character stopping at a wall), the character, and an action's path
    /// if it `collides`.
    pub fn update_constrained(&mut self, db: &Database, input: &MotionInput, dt: f32, constrain: &mut dyn FnMut(Vec3, Vec3) -> Vec3) {
        let dt = dt.max(1e-4);
        self.simulate(input, dt, constrain);
        if self.action.is_some() {
            self.play_action(db, dt, constrain);
        } else {
            self.search_if_due(db, input, dt);
            self.play(db, dt);
        }
        self.inertializer.update(&mut self.matched, self.settings.inertialization_halflife, dt);
        if self.action.is_some() {
            // the simulation waits where the action takes the character
            let s = &mut self.simulation;
            s.position = self.character.translation;
            s.rotation = self.character.rotation;
            s.acceleration = Vec3::ZERO;
            s.angular_velocity = Vec3::ZERO;
        } else {
            let before = self.character.translation;
            self.synchronize(dt);
            let moved = constrain(before, self.character.translation);
            self.character.translation = moved;
            // stand on the ground
            let y = &mut self.character.translation.y;
            *y = if (*y - self.ground).abs() > 0.5 { self.ground } else { *y + (self.ground - *y) * (1.0 - negexp(dt / 0.05)) };
            self.simulation.position.y = self.ground;
        }
        self.output(db);
        if self.settings.foot_lock && self.action.is_none() {
            self.lock_feet(db, dt);
        }
    }

    /// The output pose from the matched one: retargeted onto the display skeleton if any.
    fn output(&mut self, db: &Database) {
        match &self.display {
            Some(d) => {
                d.retarget.apply(&self.matched, &mut self.pose);
                self.pose.to_model(&d.skeleton, &mut self.model);
            }
            None => {
                self.pose.local.clone_from(&self.matched.local);
                self.pose.to_model(&db.skeleton, &mut self.model);
            }
        }
    }

    /// The simulated character follows the input with springs; predict its trajectory.
    fn simulate(&mut self, input: &MotionInput, dt: f32, constrain: &mut dyn FnMut(Vec3, Vec3) -> Vec3) {
        let goal = Vec3::new(input.velocity.x, 0.0, input.velocity.z);
        self.desired_yaw = match input.facing {
            Some(yaw) => yaw,
            None if goal.length() > 0.1 => goal.x.atan2(goal.z),
            None => self.desired_yaw,
        };
        let goal_rotation = yaw_rotation(self.desired_yaw);
        let s = &mut self.simulation;
        let before = s.position;
        spring_character_update(&mut s.position, &mut s.velocity, &mut s.acceleration, goal, self.settings.velocity_halflife, dt);
        let allowed = constrain(before, s.position);
        if allowed.distance(s.position) > 1e-5 {
            // blocked: drop the velocity into what blocked it (pushing out of an overlap is a
            // correction of the position, not speed)
            let blocked = blocked_direction(before, s.position, allowed);
            s.velocity -= blocked * s.velocity.dot(blocked).max(0.0);
            s.position = allowed;
        }
        spring_damper_exact_quat(&mut s.rotation, &mut s.angular_velocity, goal_rotation, self.settings.rotation_halflife, dt);
        for (k, t) in TRAJECTORY_TIMES.iter().enumerate() {
            let (mut p, mut v, mut a) = (s.position, s.velocity, s.acceleration);
            spring_character_update(&mut p, &mut v, &mut a, goal, self.settings.velocity_halflife, *t);
            let (mut r, mut w) = (s.rotation, s.angular_velocity);
            spring_damper_exact_quat(&mut r, &mut w, goal_rotation, self.settings.rotation_halflife, *t);
            let from = if k == 0 { s.position } else { self.trajectory[k - 1].translation };
            self.trajectory[k] = Transform::from_translation_rotation(constrain(from, p), r);
        }
    }

    /// The query: the playing frame's pose features, and the predicted trajectory relative to the
    /// character.
    fn query(&self, db: &Database, frame: usize) -> [f32; STRIDE] {
        let mut raw: [f32; FEATURES] = db.denormalize(db.features(frame));
        let to_local = self.character.rotation.conjugate();
        for (k, t) in self.trajectory.iter().enumerate() {
            let p = to_local * (t.translation - self.character.translation);
            let d = to_local * (t.rotation * FORWARD);
            raw[15 + 2 * k] = p.x;
            raw[16 + 2 * k] = p.z;
            raw[21 + 2 * k] = d.x;
            raw[22 + 2 * k] = d.z;
        }
        db.normalize_query(&raw)
    }

    fn search_if_due(&mut self, db: &Database, input: &MotionInput, dt: f32) {
        self.search_timer -= dt;
        self.last_search.searched = false;
        self.last_search.switched = false;
        let info = &db.clips[self.clip];
        let at_end = !info.looping && self.frame >= (info.frames - 1) as f32 - 1e-3;
        let changed = match self.searched_input {
            None => true,
            Some((velocity, yaw)) => velocity.distance(input.velocity) > self.settings.force_search_velocity || wrap_angle(yaw - self.desired_yaw).abs() > self.settings.force_search_turn,
        };
        // (a hair under zero counts: float steps of the frame time must not skip a frame)
        if self.search_timer > 1e-4 && !changed && !at_end {
            return;
        }
        self.search_timer = self.settings.search_interval;
        self.searched_input = Some((input.velocity, self.desired_yaw));
        let current = self.current_frame(db);
        let query = self.query(db, current);
        // staying costs what the playing frame does, unless its clip has run out
        let stay = if at_end { f32::MAX } else { super::database::distance(&query, db.features(current), f32::MAX) };
        let filter = SearchFilter { current: Some(current), ..self.settings.filter };
        let limit = if at_end { f32::MAX } else { stay - self.settings.continuing_bias };
        self.last_search = SearchInfo { searched: true, switched: false, frame: current, cost: stay };
        if let Some(Match { frame, cost }) = (limit > 0.0).then(|| db.search(&query, &filter, limit)).flatten() {
            self.transition(db, current, frame);
            self.last_search = SearchInfo { searched: true, switched: true, frame, cost };
        }
    }

    /// Switch playback to database frame `to`, inertializing from what plays now.
    fn transition(&mut self, db: &Database, from: usize, to: usize) {
        let (source_linear, source_angular, destination_linear, destination_angular, destination) = &mut self.scratch;
        db.velocities(from, source_linear, source_angular);
        db.velocities(to, destination_linear, destination_angular);
        db.pose(to, to, 0.0, destination);
        self.inertializer.transition(&self.sampled, source_linear, source_angular, destination, destination_linear, destination_angular);
        self.clip = db.clip_of(to);
        self.frame = (to - db.clips[self.clip].start) as f32;
    }

    /// Advance the playhead, moving the character by the root motion, and sample the pose.
    fn play(&mut self, db: &Database, dt: f32) {
        let info = &db.clips[self.clip];
        let next = self.frame + dt * db.sample_rate;
        let (moved, turned) = db.root_motion(self.clip, self.frame, next);
        self.character.translation += self.character.rotation * moved;
        self.character.rotation = (self.character.rotation * yaw_rotation(turned)).normalize();
        self.frame = if info.looping { next.rem_euclid(info.playable() as f32) } else { next.min((info.frames - 1) as f32) };
        let a = self.frame.floor() as usize;
        let b = (a + 1).min(info.frames - 1);
        db.pose(info.start + a, info.start + b, self.frame - a as f32, &mut self.sampled);
        self.matched.local.clone_from(&self.sampled.local);
        self.root_speed = (moved.length() / dt, turned.abs() / dt);
    }

    /// Play the action: its clip on, the character moved by its root path; hand back to the
    /// search at its exit frame.
    fn play_action(&mut self, db: &Database, dt: f32, constrain: &mut dyn FnMut(Vec3, Vec3) -> Vec3) {
        let Some((clip, exit, collides)) = self.action.as_ref().map(|a| (a.clip, a.exit, a.collides)) else { return };
        let info = &db.clips[clip];
        let before = self.character;
        let mut next = self.frame + dt * db.sample_rate;
        let exit = exit.map(|e| e.min((info.frames - 1) as f32));
        if info.looping && exit.is_none() {
            next = next.rem_euclid(info.playable() as f32);
        }
        let done = exit.is_some_and(|e| next >= e);
        self.frame = next.min((info.frames - 1) as f32);
        match &mut self.action.as_mut().unwrap().path {
            RootPath::Warp(warp) => {
                let (p, yaw) = warp.root(self.frame, db.root_at(clip, self.frame));
                let p = if collides { constrain(before.translation, p) } else { p };
                self.character = Transform::from_translation_rotation(p, yaw_rotation(yaw));
            }
            RootPath::Ballistic { velocity, gravity } => {
                let from = self.character.translation;
                let to = from + *velocity * dt - Vec3::Y * (0.5 * *gravity * dt * dt);
                let allowed = if collides { constrain(from, to) } else { to };
                // blocked: drop the velocity into what blocked it
                let blocked = if allowed.distance(to) > 1e-5 { blocked_direction(from, to, allowed) } else { Vec3::ZERO };
                *velocity -= blocked * velocity.dot(blocked).max(0.0);
                velocity.y -= *gravity * dt;
                self.character.translation = allowed;
            }
        }
        let a = self.frame.floor() as usize;
        let b = (a + 1).min(info.frames - 1);
        db.pose(info.start + a, info.start + b, self.frame - a as f32, &mut self.sampled);
        self.matched.local.clone_from(&self.sampled.local);
        let moved = self.character.translation - before.translation;
        self.root_speed = (Vec3::new(moved.x, 0.0, moved.z).length() / dt, wrap_angle(yaw_of(self.character.rotation) - yaw_of(before.rotation)).abs() / dt);
        self.simulation.velocity = Vec3::new(moved.x, 0.0, moved.z) / dt;
        if done {
            self.ground = self.character.translation.y;
            self.stop_action();
        }
    }

    /// Pull the animated character toward the simulation, then clamp it within reach of it.
    fn synchronize(&mut self, dt: f32) {
        let s = &self.settings;
        let (speed, turn_rate) = self.root_speed;
        let difference = self.simulation.position - self.character.translation;
        let mut adjustment = damper_exact(Vec3::ZERO, difference, s.adjustment_halflife, dt);
        if s.adjust_by_velocity {
            adjustment = adjustment.clamp_length_max(s.max_adjustment_ratio * speed * dt);
        }
        self.character.translation += Vec3::new(adjustment.x, 0.0, adjustment.z);
        let rotation_difference = quat_to_scaled_angle_axis(quat_abs(self.simulation.rotation * self.character.rotation.conjugate()));
        let mut rotation_adjustment = rotation_difference * (1.0 - negexp((std::f32::consts::LN_2 * dt) / (s.adjustment_halflife + 1e-5)));
        if s.adjust_by_velocity {
            rotation_adjustment = rotation_adjustment.clamp_length_max(s.max_adjustment_ratio * turn_rate * dt);
        }
        self.character.rotation = yaw_rotation(yaw_of(quat_from_scaled_angle_axis(rotation_adjustment) * self.character.rotation));
        // never farther than the clamps from the simulation
        let offset = self.character.translation - self.simulation.position;
        let flat = Vec3::new(offset.x, 0.0, offset.z);
        if flat.length() > s.clamp_distance {
            self.character.translation = self.simulation.position + flat.normalize() * s.clamp_distance + Vec3::new(0.0, offset.y, 0.0);
        }
        let yaw_gap = wrap_angle(yaw_of(self.character.rotation) - yaw_of(self.simulation.rotation));
        if yaw_gap.abs() > s.clamp_angle {
            self.character.rotation = yaw_rotation(yaw_of(self.simulation.rotation) + yaw_gap.signum() * s.clamp_angle);
        }
    }

    /// Pin planted feet where they touched down (two-joint IK on each leg): the ankle while the
    /// heel is down, the ball once it lifts.
    fn lock_feet(&mut self, db: &Database, dt: f32) {
        let contacts = db.contacts(self.current_frame(db));
        let s = &self.settings;
        let mut moved = false;
        let skeleton = self.display.as_ref().map_or(&db.skeleton, |d| &d.skeleton);
        let legs = self.display.as_ref().map_or(self.legs, |d| d.legs);
        for (side, leg) in legs.iter().enumerate() {
            let [upper, middle, foot] = leg.joints;
            if upper == middle || middle == foot {
                continue;
            }
            // the model is relative to the character, which stands on the ground
            let (ankle, ball) = (self.model[foot].translation, self.model[leg.ball].translation);
            let animated = FootPose { ankle: self.character.transform_point(ankle), ball: self.character.transform_point(ball), ankle_lift: ankle.y - leg.rest[0], ball_lift: ball.y - leg.rest[1] };
            let target = self.feet[side].update_foot(&animated, contacts[side], s.foot_unlock_radius, s.foot_lock_halflife, dt);
            if target.distance(animated.ankle) > 1e-5 {
                let local = self.character.inverse().transform_point(target);
                two_joint_ik(skeleton, &mut self.pose, &mut self.model, upper, middle, foot, local);
                moved = true;
            }
        }
        if moved {
            // the joints below the feet follow them
            self.pose.to_model(skeleton, &mut self.model);
        }
    }
}
