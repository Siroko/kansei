//! Traversal: hurdles, vaults, mantles and climbs over obstacles found in a collision world, and
//! the falls and landings around them, played as `Action`s with warped root motion between
//! stretches of motion matching.
//!
//! - `ActionClip::analyze` reads what a traversal clip was captured against from its animation
//!   alone: the root joint follows the surface the character is on (it rises onto the obstacle),
//!   and hands planted on the top mark its front edge. It finds the height, the ledge, and the
//!   frames the character leaves the ground, reaches the top, leaves it, lands and can hand back
//!   to motion matching.
//! - `detect_obstacle` probes the world in front of the character: casts at several heights for a
//!   face, a ray down for its top, then rays along the top and the edge for its depth, width and
//!   the floor behind.
//! - `traversal_kind` picks the kind from the obstacle's shape and the room to stand beyond its
//!   ledge. `plan_traversal` picks the clip from the character's speed, the frame to start at from
//!   the distance to the ledge and the pose, and skips clips that would end inside something; it
//!   warps the clip so its ledge lands on the real one, lifted to the real height and stretched to
//!   the real depth.
//! - `CharacterController` keeps a `MotionMatcher` out of the world's colliders, on its ground,
//!   falling off edges and landing, and runs traversals on request.

use glam::{Vec2, Vec3};

use super::controller::{Action, MotionInput, MotionMatcher, RootPath};
use super::database::{yaw_of, yaw_rotation, Database, FORWARD};
use crate::animation::warping::{Placement, Ramp, RootWarp};
use crate::animation::Pose;
use crate::collision::CollisionWorld;

/// What an action clip does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ActionKind {
    /// Over a thin obstacle, landing on the far side.
    Hurdle = 0,
    /// Over a deeper one, hands on top.
    Vault = 1,
    /// Up onto a top to stand on.
    Mantle = 2,
    /// Up a tall wall onto its top.
    Climb = 3,
    /// Falling (a loop).
    Fall = 4,
    /// Landing from a fall.
    Land = 5,
}

impl ActionKind {
    pub const ALL: [ActionKind; 6] = [ActionKind::Hurdle, ActionKind::Vault, ActionKind::Mantle, ActionKind::Climb, ActionKind::Fall, ActionKind::Land];

    pub fn from_u8(v: u8) -> Option<Self> {
        Self::ALL.get(v as usize).copied()
    }

    pub fn name(self) -> &'static str {
        match self {
            ActionKind::Hurdle => "hurdle",
            ActionKind::Vault => "vault",
            ActionKind::Mantle => "mantle",
            ActionKind::Climb => "climb",
            ActionKind::Fall => "fall",
            ActionKind::Land => "land",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|k| k.name() == name)
    }

    /// Goes over the obstacle (lands behind it) rather than onto it.
    pub fn crosses(self) -> bool {
        matches!(self, ActionKind::Hurdle | ActionKind::Vault)
    }
}

/// What an action clip was captured against, and its phases (clip frames).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActionClip {
    pub clip: usize,
    pub kind: ActionKind,
    /// Height of the obstacle above the clip's starting ground (for a landing: the drop).
    pub height: f32,
    /// The obstacle's front ledge (on its top), in the clip's space, and the direction the clip
    /// runs into it (horizontal, unit).
    pub ledge: Vec3,
    pub forward: Vec3,
    /// The last frame on the starting ground before the lift.
    pub rise: f32,
    /// When the ledge is reached (hands on it, or the root most of the way up).
    pub anchor: f32,
    /// When the root is up on the top, and (over an obstacle) when it leaves it and is back down.
    pub on_top: f32,
    pub off_top: f32,
    pub down: f32,
    /// When motion matching can take over (for a landing: the impact).
    pub exit: f32,
    /// Distance the root travels on the top (over an obstacle).
    pub span: f32,
    /// Latest frame a traversal can start at (time for the warp before the lift).
    pub last_entry: f32,
}

/// The analysis' view of a clip: root per frame, hands and hips in the clip's space.
struct Tracks {
    root: Vec<(Vec3, f32)>,
    hands: Vec<[Vec3; 2]>,
    hips_height: Vec<f32>,
}

fn tracks(db: &Database, clip: usize, hands: [usize; 2]) -> Tracks {
    let info = &db.clips[clip];
    let mut pose = Pose { local: Vec::new() };
    let mut model = Vec::new();
    let mut t = Tracks { root: Vec::new(), hands: Vec::new(), hips_height: Vec::new() };
    for f in 0..info.frames {
        let frame = info.start + f;
        db.pose(frame, frame, 0.0, &mut pose);
        pose.to_model(&db.skeleton, &mut model);
        let (p, yaw) = db.root_at(clip, f as f32);
        let r = yaw_rotation(yaw);
        t.root.push((p, yaw));
        t.hands.push(hands.map(|h| p + r * model[h].translation));
        t.hips_height.push(model[db.roles.hips].translation.y);
    }
    t
}

impl ActionClip {
    /// Read `clip`'s phases for `kind` from its animation (`hands`: the hand joints). `None` when
    /// the clip doesn't show the motion (a traversal that never leaves the ground, a landing
    /// with no fall).
    pub fn analyze(db: &Database, clip: usize, kind: ActionKind, hands: [usize; 2]) -> Option<ActionClip> {
        let info = &db.clips[clip];
        let n = info.frames;
        let rate = db.sample_rate;
        let t = tracks(db, clip, hands);
        let base = t.root[0].0.y;
        let y: Vec<f32> = t.root.iter().map(|(p, _)| p.y - base).collect();
        let last = (n - 1) as f32;
        let empty = ActionClip {
            clip,
            kind,
            height: 0.0,
            ledge: Vec3::ZERO,
            forward: FORWARD,
            rise: 0.0,
            anchor: 0.0,
            on_top: 0.0,
            off_top: 0.0,
            down: 0.0,
            exit: last,
            span: 0.0,
            last_entry: 0.0,
        };
        match kind {
            ActionKind::Fall => return Some(empty),
            ActionKind::Land => {
                // the impact: the root reaches its final height after being well above it
                let end = y[n - 1];
                let top = y.iter().cloned().fold(f32::MIN, f32::max);
                if top - end < 0.3 {
                    return None;
                }
                let impact = (0..n).find(|&f| y[f] - end <= 0.02 && (0..f).any(|g| y[g] - end > 0.3))?;
                return Some(ActionClip { height: top - end, anchor: impact as f32, exit: (impact as f32 + 0.5 * rate).min(last), ..empty });
            }
            _ => {}
        }
        let peak = if kind.crosses() { y.iter().cloned().fold(f32::MIN, f32::max) } else { y[n - 1] };
        if peak < 0.2 {
            return None;
        }
        let on_top = (0..n).find(|&f| y[f] >= 0.98 * peak)?;
        let rise = (0..on_top).rev().find(|&f| y[f] <= 0.02).unwrap_or(0);
        let halfway = (0..n).find(|&f| y[f] >= 0.9 * peak).unwrap_or(on_top);
        let yaw_at = t.root[on_top].1;
        let forward = yaw_rotation(yaw_at) * FORWARD;
        let forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or(FORWARD);
        // hands planted on the top near the lift: the ledge is just in front of them
        let top_y = base + peak;
        let mut planted: Option<(usize, f32)> = None;
        for f in rise.saturating_sub(10)..(on_top + 10).min(n - 1) {
            for h in 0..2 {
                let (p, q) = (t.hands[f][h], t.hands[f + 1][h]);
                if p.y > top_y - 0.15 && p.y < top_y + 0.25 && p.distance(q) * rate < 0.6 {
                    let along = p.dot(forward);
                    if planted.is_none_or(|(g, a)| f < g || (f == g && along < a)) {
                        planted = Some((f, along));
                    }
                }
            }
        }
        let (anchor, along) = match planted {
            Some((f, along)) => (f.min(on_top), along - 0.05),
            None => (halfway, t.root[halfway].0.dot(forward)),
        };
        let at = t.root[anchor].0;
        let ledge = Vec3::new(at.x, top_y, at.z) + forward * (along - at.dot(forward));
        let (off_top, down, span, exit) = if kind.crosses() {
            let off = (on_top..n).take_while(|&f| y[f] >= 0.9 * peak).last().unwrap_or(on_top);
            let down = (off..n).find(|&f| y[f] <= 0.02).unwrap_or(n - 1);
            let span = (t.root[off].0 - t.root[on_top].0).dot(forward);
            // a hurdle runs on from the landing; a vault hands over as it reaches the ground
            let exit = if kind == ActionKind::Hurdle { (down as f32 + 0.3 * rate).min(last) } else { down as f32 };
            (off as f32, down as f32, span, exit)
        } else {
            // up there once the hips are back to standing height
            let standing = t.hips_height[0] * 0.9;
            let up = (on_top..n).find(|&f| t.hips_height[f] >= standing).unwrap_or(n - 1);
            (last, last, 0.0, (up as f32 + 0.1 * rate).min(last))
        };
        Some(ActionClip {
            clip,
            kind,
            height: peak,
            ledge,
            forward,
            rise: rise as f32,
            anchor: anchor as f32,
            on_top: on_top as f32,
            off_top,
            down,
            exit,
            span,
            last_entry: (rise as f32 - 0.2 * rate).max(0.0),
        })
    }

    /// Distance from the clip root at `frame` to the ledge, along the approach.
    pub fn distance_at(&self, db: &Database, frame: f32) -> f32 {
        (self.ledge - db.root_at(self.clip, frame).0).dot(self.forward)
    }

    /// The clip root's speed (m/s) at `frame`.
    pub fn speed_at(&self, db: &Database, frame: f32) -> f32 {
        let a = db.root_at(self.clip, frame).0;
        let b = db.root_at(self.clip, frame + 1.0).0;
        Vec2::new(b.x - a.x, b.z - a.z).length() * db.sample_rate
    }
}

/// An obstacle in front of the character.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Obstacle {
    /// On the front edge of the top, where the character meets it.
    pub ledge: Vec3,
    /// The front face's normal (horizontal, toward the character).
    pub normal: Vec3,
    /// Top above the character's feet.
    pub height: f32,
    /// Length of the top away from the character; `None` when deeper than probed.
    pub depth: Option<f32>,
    /// Height (world) of the floor behind a shallow top.
    pub back_floor: Option<f32>,
    /// Free top along the edge on each side of the approach line, the smaller.
    pub half_width: f32,
    /// From the character's feet to the ledge, along the approach.
    pub distance: f32,
}

/// Probes for obstacle detection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DetectionSettings {
    /// Heights above the feet the forward sweeps run at, and their sphere radius.
    pub heights: [f32; 6],
    pub radius: f32,
    /// Highest top considered, above the feet.
    pub max_height: f32,
    /// How far along the top depth is measured, and the step.
    pub max_depth: f32,
    pub step: f32,
    pub layers: u32,
}

impl Default for DetectionSettings {
    fn default() -> Self {
        Self { heights: [0.35, 0.65, 1.0, 1.45, 1.95, 2.45], radius: 0.12, max_height: 3.0, max_depth: 2.0, step: 0.1, layers: u32::MAX }
    }
}

/// The obstacle ahead of `feet` along `direction` (horizontal) within `reach`, if any.
pub fn detect_obstacle(world: &CollisionWorld, feet: Vec3, direction: Vec3, reach: f32, settings: &DetectionSettings) -> Option<Obstacle> {
    let direction = Vec3::new(direction.x, 0.0, direction.z).normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    // the nearest steep face at any probe height
    let hit = settings
        .heights
        .iter()
        .filter_map(|h| world.sphere_cast(feet + Vec3::Y * h, settings.radius, direction, reach, settings.layers))
        .filter(|h| h.normal.y.abs() < 0.3 && h.distance > 0.0)
        .min_by(|a, b| a.distance.total_cmp(&b.distance))?;
    let normal = Vec3::new(hit.normal.x, 0.0, hit.normal.z).normalize_or_zero();
    if normal == Vec3::ZERO || normal.dot(direction) > -0.3 {
        return None;
    }
    let face = hit.point - normal * settings.radius;
    // its top, just behind the face
    let down = |p: Vec3| world.raycast(Vec3::new(p.x, feet.y + settings.max_height + 0.3, p.z), Vec3::NEG_Y, settings.max_height + 0.3 - 0.05, settings.layers);
    let top_hit = down(face - normal * 0.08)?;
    if top_hit.normal.y < 0.7 {
        return None;
    }
    let top = top_hit.point.y;
    let height = top - feet.y;
    if height < 0.25 || height > settings.max_height {
        return None;
    }
    let ledge = Vec3::new(face.x, top, face.z);
    let on_top = |p: Vec3| world.raycast(Vec3::new(p.x, top + 0.3, p.z), Vec3::NEG_Y, 0.45, settings.layers).is_some_and(|h| (h.point.y - top).abs() < 0.12);
    // the top along the approach
    let mut depth = None;
    let steps = (settings.max_depth / settings.step).round() as usize;
    for k in 1..=steps {
        let d = k as f32 * settings.step;
        if !on_top(ledge - normal * d) {
            depth = Some(d - settings.step * 0.5);
            break;
        }
    }
    // the floor behind a shallow top: none when the probe starts inside something (a wall right
    // behind) or finds nothing within 3 m below the feet
    let back_floor = depth.and_then(|d| {
        let behind = ledge - normal * (d + 0.5);
        world.raycast(Vec3::new(behind.x, top + 0.3, behind.z), Vec3::NEG_Y, top + 0.3 - (feet.y - 3.0), settings.layers).filter(|h| h.distance > 0.0).map(|h| h.point.y)
    });
    // the edge to each side
    let side = normal.cross(Vec3::Y).normalize();
    let reach_side = |s: f32| (1..=10).map(|k| k as f32 * 0.1).take_while(|&w| on_top(ledge - normal * 0.1 + side * s * w)).last().unwrap_or(0.0);
    let half_width = reach_side(1.0).min(reach_side(-1.0));
    let distance = (feet - ledge).dot(normal);
    Some(Obstacle { ledge, normal, height, depth, back_floor: back_floor.filter(|_| depth.is_some()), half_width, distance })
}

/// Why a traversal was not started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    Busy,
    NoObstacle,
    TooNarrow,
    TooHigh,
    NoRoom,
    NoClip,
    OutOfReach,
}

impl Refusal {
    pub fn describe(self) -> &'static str {
        match self {
            Refusal::Busy => "busy",
            Refusal::NoObstacle => "nothing to traverse ahead",
            Refusal::TooNarrow => "the ledge is too narrow",
            Refusal::TooHigh => "too high or too low",
            Refusal::NoRoom => "no room to land",
            Refusal::NoClip => "no clip for it",
            Refusal::OutOfReach => "too far or too close",
        }
    }
}

/// Clearances used when choosing a traversal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TraversalRules {
    /// Deepest top a hurdle, then a vault, goes over.
    pub hurdle_depth: f32,
    pub vault_depth: f32,
    /// Height ranges (above the feet) of each kind.
    pub cross_heights: (f32, f32),
    pub mantle_heights: (f32, f32),
    pub climb_heights: (f32, f32),
    /// Narrowest half-width of ledge to use.
    pub min_half_width: f32,
    /// A standing character's capsule, for landing room.
    pub radius: f32,
    pub height: f32,
    /// Weights of the entry frame's cost: distance to the ledge (per m²), speed (per (m/s)²) and
    /// pose (feature distance).
    pub distance_weight: f32,
    pub speed_weight: f32,
    pub pose_weight: f32,
    /// Largest distance error a warp may absorb (m), and fastest it may slide the root to do so
    /// before the ledge is reached (m/s).
    pub max_distance_error: f32,
    pub max_warp_speed: f32,
}

impl Default for TraversalRules {
    fn default() -> Self {
        Self {
            hurdle_depth: 0.55,
            vault_depth: 1.3,
            cross_heights: (0.3, 1.3),
            mantle_heights: (0.5, 1.8),
            climb_heights: (1.8, 2.9),
            min_half_width: 0.3,
            radius: 0.3,
            height: 1.75,
            distance_weight: 4.0,
            speed_weight: 0.5,
            pose_weight: 0.05,
            max_distance_error: 1.2,
            max_warp_speed: 1.5,
        }
    }
}

/// Whether a standing character (the rules' capsule) fits with its feet at `feet`.
pub fn stands_at(world: &CollisionWorld, feet: Vec3, rules: &TraversalRules, layers: u32) -> bool {
    let feet = feet + Vec3::Y * 0.05;
    !world.overlap_capsule(feet + Vec3::Y * rules.radius, feet + Vec3::Y * (rules.height - rules.radius), rules.radius, layers)
}

/// Which kind of traversal an obstacle calls for, or why none.
pub fn traversal_kind(world: &CollisionWorld, obstacle: &Obstacle, feet: Vec3, rules: &TraversalRules, layers: u32) -> Result<ActionKind, Refusal> {
    if obstacle.half_width < rules.min_half_width {
        return Err(Refusal::TooNarrow);
    }
    let h = obstacle.height;
    let within = |r: (f32, f32)| h >= r.0 && h <= r.1;
    let room_on_top = || stands_at(world, obstacle.ledge - obstacle.normal * (rules.radius + 0.35), rules, layers);
    let deep_enough = obstacle.depth.is_none_or(|d| d >= 2.0 * rules.radius + 0.2);
    // shallow tops: over them, onto a floor near the feet behind
    if let Some(depth) = obstacle.depth.filter(|d| *d <= rules.vault_depth) {
        if within(rules.cross_heights) {
            match obstacle.back_floor {
                Some(floor) if (floor - feet.y).abs() < 0.6 => {
                    let behind = obstacle.ledge - obstacle.normal * (depth + rules.radius + 0.3);
                    if !stands_at(world, Vec3::new(behind.x, floor, behind.z), rules, layers) {
                        return Err(Refusal::NoRoom);
                    }
                    return Ok(if depth <= rules.hurdle_depth { ActionKind::Hurdle } else { ActionKind::Vault });
                }
                // no floor to land on behind, and too shallow to stand on
                _ if !deep_enough => return Err(Refusal::NoRoom),
                _ => {}
            }
        }
    }
    if within(rules.mantle_heights) && deep_enough {
        return if room_on_top() { Ok(ActionKind::Mantle) } else { Err(Refusal::NoRoom) };
    }
    if within(rules.climb_heights) && deep_enough {
        return if room_on_top() { Ok(ActionKind::Climb) } else { Err(Refusal::NoRoom) };
    }
    Err(Refusal::TooHigh)
}

/// The best clip of `kind` and frame to start it at for a character at `feet` with this speed,
/// playing database frame `current`, and the action that warps it onto the obstacle. A clip
/// that ends on the top must leave the character where `stands` (feet position) allows: a
/// mantle that walks on before handing over needs more room than one that stands up by the edge.
#[allow(clippy::too_many_arguments)]
pub fn plan_traversal(db: &Database, table: &[ActionClip], kind: ActionKind, obstacle: &Obstacle, feet: Vec3, heading: f32, speed: f32, current: usize, rules: &TraversalRules, stands: impl Fn(Vec3) -> bool) -> Result<Action, Refusal> {
    let here = db.features(current);
    // each clip's best entry frame, cheapest first
    let mut candidates: Vec<(f32, &ActionClip, f32)> = Vec::new();
    for c in table.iter().filter(|c| c.kind == kind) {
        let mut best: Option<(f32, f32)> = None;
        let mut f = 0.0;
        while f <= c.last_entry {
            let d = c.distance_at(db, f);
            let error = d - obstacle.distance;
            let window = (c.anchor - f).max(1.0) / db.sample_rate;
            if error.abs() <= rules.max_distance_error.min(rules.max_warp_speed * window) {
                let s = c.speed_at(db, f) - speed;
                let frame = db.clips[c.clip].start + f as usize;
                let pose: f32 = db.features(frame)[..15].iter().zip(&here[..15]).map(|(a, b)| (a - b) * (a - b)).sum();
                let cost = rules.distance_weight * error * error + rules.speed_weight * s * s + rules.pose_weight * pose;
                if best.is_none_or(|(b, _)| cost < b) {
                    best = Some((cost, f));
                }
            }
            f += 1.0;
        }
        if let Some((cost, f)) = best {
            candidates.push((cost, c, f));
        }
    }
    if candidates.is_empty() {
        return Err(if table.iter().any(|c| c.kind == kind) { Refusal::OutOfReach } else { Refusal::NoClip });
    }
    candidates.sort_by(|a, b| a.0.total_cmp(&b.0));
    candidates
        .into_iter()
        .map(|(_, c, start)| (c, start, warp_onto(db, c, start, obstacle, feet, heading)))
        .find(|(c, _, warp)| c.kind.crosses() || stands(warp.root(c.exit, db.root_at(c.clip, c.exit)).0))
        .map(|(c, start, warp)| Action { clip: c.clip, start, exit: Some(c.exit), path: RootPath::Warp(warp), tag: kind as u32 })
        .ok_or(Refusal::NoRoom)
}

/// `c`'s root motion from `start`, warped from the character at `feet` onto the obstacle.
fn warp_onto(db: &Database, c: &ActionClip, start: f32, obstacle: &Obstacle, feet: Vec3, heading: f32) -> RootWarp {
    let clip_start = db.root_at(c.clip, start);
    let mut warp = RootWarp::identity(clip_start, (feet, heading));
    // the clip's ledge onto the real one, facing into it
    let clip_heading = c.forward.x.atan2(c.forward.z);
    let into = -obstacle.normal;
    warp.to = Placement::between((Vec3::new(c.ledge.x, 0.0, c.ledge.z), clip_heading), (Vec3::new(obstacle.ledge.x, 0.0, obstacle.ledge.z), into.x.atan2(into.z)));
    warp.window = Ramp::new(start, c.anchor.max(start + 1.0), 0.0, 1.0);
    // up by what the real obstacle has more than the captured one, by the time the ledge is
    // reached; over one, down again to the floor behind
    let clip_ground = clip_start.0.y;
    warp.ground = feet.y - clip_ground;
    let lift = obstacle.height - c.height;
    warp.lift.push(Ramp::new(c.rise.max(start), c.anchor.max(c.rise.max(start) + 1.0), 0.0, lift));
    if c.kind.crosses() {
        let floor = obstacle.back_floor.unwrap_or(feet.y) - feet.y;
        warp.lift.push(Ramp::new(c.off_top, c.down.max(c.off_top + 1.0), 0.0, floor - lift));
        let extra = (obstacle.depth.unwrap_or(c.span) - c.span).clamp(-0.3, 1.0);
        warp.stretch.push(Ramp::new(c.on_top, c.off_top.max(c.on_top + 1.0), 0.0, extra));
        warp.stretch_direction = into;
    }
    warp
}

/// What a `CharacterController` is doing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CharacterState {
    Grounded,
    Traversing(ActionKind),
    /// Seconds in the air.
    Falling(f32),
    Landing,
}

/// A motion-matched character in a collision world: kept out of colliders, on its ground,
/// falling off edges and landing, and traversing obstacles on request.
pub struct CharacterController {
    pub matcher: MotionMatcher,
    pub actions: Vec<ActionClip>,
    pub rules: TraversalRules,
    pub detection: DetectionSettings,
    /// Capsule: radius, height, and the step it walks up without blocking.
    pub radius: f32,
    pub height: f32,
    pub step: f32,
    pub gravity: f32,
    pub layers: u32,
    state: CharacterState,
    /// The last obstacle probed and what came of it (for debug views).
    pub last_obstacle: Option<Obstacle>,
    pub last_result: Option<Result<ActionKind, Refusal>>,
    /// Seconds a `request_traverse` keeps trying while the obstacle is still out of reach.
    pending: f32,
}

impl CharacterController {
    pub fn new(matcher: MotionMatcher, actions: Vec<ActionClip>) -> Self {
        Self {
            matcher,
            actions,
            rules: TraversalRules::default(),
            detection: DetectionSettings::default(),
            radius: 0.3,
            height: 1.75,
            step: 0.35,
            gravity: 9.81,
            layers: u32::MAX,
            state: CharacterState::Grounded,
            last_obstacle: None,
            last_result: None,
            pending: 0.0,
        }
    }

    pub fn state(&self) -> CharacterState {
        self.state
    }

    fn first(&self, kind: ActionKind) -> Option<&ActionClip> {
        self.actions.iter().find(|c| c.kind == kind)
    }

    /// Traverse the obstacle ahead as soon as it is in reach, trying for up to `patience` seconds
    /// (a button pressed a little early still vaults).
    pub fn request_traverse(&mut self, db: &Database, world: &CollisionWorld, patience: f32) -> Result<ActionKind, Refusal> {
        let result = self.traverse(db, world);
        // what may change as it comes closer: the obstacle in reach, a clip that fits
        self.pending = if matches!(result, Err(Refusal::OutOfReach | Refusal::NoObstacle | Refusal::NoRoom)) { patience } else { 0.0 };
        result
    }

    /// Advance by `dt` under `input`.
    pub fn update(&mut self, db: &Database, world: &CollisionWorld, input: &MotionInput, dt: f32) {
        if self.pending > 0.0 {
            self.pending -= dt;
            if self.state == CharacterState::Grounded && self.traverse(db, world).is_ok() {
                self.pending = 0.0;
            }
        }
        let (radius, height, step, layers) = (self.radius, self.height, self.step, self.layers);
        let mut constrain = |from: Vec3, to: Vec3| -> Vec3 {
            // sweep at the waist, stop short of what it meets and slide along it, then push out
            let waist = Vec3::Y * (height * 0.5);
            let delta = Vec3::new(to.x - from.x, 0.0, to.z - from.z);
            let length = delta.length();
            let mut end = Vec3::new(to.x, from.y, to.z);
            if length > 1e-6 {
                // walls it moves into (not what it moves away from, e.g. overlapped on landing)
                if let Some(hit) = world.sphere_cast(from + waist, radius, delta / length, length, layers).filter(|h| h.normal.y.abs() < 0.5 && h.normal.dot(delta) < 0.0) {
                    let n = Vec3::new(hit.normal.x, 0.0, hit.normal.z).normalize_or_zero();
                    let stop = from + delta / length * (hit.distance - 0.01).max(0.0);
                    let rest = delta * (1.0 - hit.distance / length);
                    end = stop + (rest - n * rest.dot(n));
                }
            }
            let resolved = world.resolve_capsule(end, height, radius, step, layers);
            Vec3::new(resolved.x, to.y, resolved.z)
        };
        match self.state {
            CharacterState::Grounded => {
                self.matcher.update_constrained(db, input, dt, &mut constrain);
                let c = self.matcher.character().translation;
                let ground = world.ground_height(c, step + 0.1, 4.0, layers);
                match ground {
                    Some(g) if g > self.matcher.ground() - 0.4 => self.matcher.set_ground(g),
                    _ => self.start_fall(db),
                }
            }
            CharacterState::Traversing(kind) => {
                self.matcher.update(db, input, dt);
                if self.matcher.action().is_none() {
                    // over an obstacle, land on the floor behind; else stand where it ended
                    let c = self.matcher.character().translation;
                    let ground = world.ground_height(c, 0.3, 4.0, layers);
                    if kind == ActionKind::Vault && ground.is_some_and(|g| (g - c.y).abs() < 0.3) {
                        self.matcher.set_ground(ground.unwrap());
                        self.land(db);
                    } else if ground.is_some_and(|g| g > c.y - 0.4) {
                        self.matcher.set_ground(ground.unwrap());
                        self.state = CharacterState::Grounded;
                    } else {
                        self.state = CharacterState::Grounded;
                        self.start_fall(db);
                    }
                }
            }
            CharacterState::Falling(time) => {
                self.matcher.update_constrained(db, input, dt, &mut constrain);
                let c = self.matcher.character().translation;
                let ground = world.ground_height(c, 1.0, 50.0, layers);
                if let Some(g) = ground.filter(|g| c.y <= *g) {
                    let mut t = self.matcher.character();
                    t.translation.y = g;
                    self.matcher.place(t);
                    self.matcher.set_ground(g);
                    if time > 0.35 {
                        self.land(db);
                    } else {
                        self.matcher.stop_action();
                        self.state = CharacterState::Grounded;
                    }
                } else {
                    self.state = CharacterState::Falling(time + dt);
                }
            }
            CharacterState::Landing => {
                self.matcher.update_constrained(db, input, dt, &mut constrain);
                if self.matcher.action().is_none() {
                    self.state = CharacterState::Grounded;
                }
            }
        }
    }

    fn start_fall(&mut self, db: &Database) {
        let velocity = self.matcher.simulation().velocity;
        let velocity = Vec3::new(velocity.x, 0.0, velocity.z);
        match self.first(ActionKind::Fall).copied() {
            Some(fall) => {
                let action = Action { clip: fall.clip, start: 0.0, exit: None, path: RootPath::Ballistic { velocity, gravity: self.gravity }, tag: ActionKind::Fall as u32 };
                self.matcher.start_action(db, action);
            }
            None => {
                // no fall clip: fall with the pose that plays
                let (clip, frame) = self.matcher.playing();
                let action = Action { clip, start: frame, exit: None, path: RootPath::Ballistic { velocity, gravity: self.gravity }, tag: ActionKind::Fall as u32 };
                self.matcher.start_action(db, action);
            }
        }
        self.state = CharacterState::Falling(0.0);
    }

    fn land(&mut self, db: &Database) {
        let speed = self.matcher.simulation().velocity.length();
        let character = self.matcher.character();
        // the landing whose run-out pace is nearest
        let best = self.actions.iter().filter(|c| c.kind == ActionKind::Land).min_by(|a, b| {
            let pace = |c: &ActionClip| (c.speed_at(db, (c.anchor + 5.0).min(c.exit)) - speed).abs();
            pace(a).total_cmp(&pace(b))
        });
        match best.copied() {
            Some(land) => {
                let clip_root = db.root_at(land.clip, land.anchor);
                // from the impact on, on the ground (whatever height the fall or vault ended at)
                let on_ground = Vec3::new(character.translation.x, self.matcher.ground(), character.translation.z);
                let warp = RootWarp::identity(clip_root, (on_ground, yaw_of(character.rotation)));
                self.matcher.start_action(db, Action { clip: land.clip, start: land.anchor, exit: Some(land.exit), path: RootPath::Warp(warp), tag: ActionKind::Land as u32 });
                self.state = CharacterState::Landing;
            }
            None => {
                self.matcher.stop_action();
                self.state = CharacterState::Grounded;
            }
        }
    }

    /// Look for an obstacle ahead (along the character's movement, else its facing) and
    /// traverse it if it can.
    pub fn traverse(&mut self, db: &Database, world: &CollisionWorld) -> Result<ActionKind, Refusal> {
        let result = self.try_traverse(db, world);
        self.last_result = Some(result);
        result
    }

    fn try_traverse(&mut self, db: &Database, world: &CollisionWorld) -> Result<ActionKind, Refusal> {
        if self.state != CharacterState::Grounded {
            return Err(Refusal::Busy);
        }
        let character = self.matcher.character();
        let velocity = self.matcher.simulation().velocity;
        let speed = Vec3::new(velocity.x, 0.0, velocity.z).length();
        let facing = character.rotation * FORWARD;
        let direction = if speed > 0.5 { velocity } else { facing };
        let reach = 1.5 + speed * 1.1;
        let obstacle = detect_obstacle(world, character.translation, direction, reach, &self.detection);
        self.last_obstacle = obstacle;
        let obstacle = obstacle.ok_or(Refusal::NoObstacle)?;
        let kind = traversal_kind(world, &obstacle, character.translation, &self.rules, self.layers)?;
        let stands = |feet: Vec3| stands_at(world, feet, &self.rules, self.layers);
        let action = plan_traversal(db, &self.actions, kind, &obstacle, character.translation, yaw_of(character.rotation), speed, self.matcher.current_frame(db), &self.rules, stands)?;
        self.matcher.start_action(db, action);
        self.state = CharacterState::Traversing(kind);
        Ok(kind)
    }
}

#[cfg(test)]
mod tests;
