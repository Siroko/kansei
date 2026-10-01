use glam::{Quat, Vec3};

use super::*;
use crate::animation::motion_matching::{DatabaseBuilder, JointRoles, MotionMatchingSettings, ACTION_TAG};
use crate::animation::{Clip, Pose, Skeleton, Transform};
use crate::collision::Obb;

const RATE: f32 = 30.0;
const HANDS: [usize; 2] = [2, 9];

/// A root, hips 1 m up, a hand each side, two legs.
fn skeleton() -> Skeleton {
    let t = |x: f32, y: f32| Transform::from_translation_rotation(Vec3::new(x, y, 0.0), Quat::IDENTITY);
    Skeleton::new(
        ["root", "hips", "hand_l", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r", "hand_r"].map(String::from).to_vec(),
        vec![None, Some(0), Some(1), Some(1), Some(3), Some(4), Some(1), Some(6), Some(7), Some(1)],
        vec![t(0.0, 0.0), t(0.0, 1.0), t(0.25, 0.0), t(0.1, 0.0), t(0.0, -0.45), t(0.0, -0.45), t(-0.1, 0.0), t(0.0, -0.45), t(0.0, -0.45), t(-0.25, 0.0)],
    )
}

/// A clip from per-frame root (position), hips height and optional world hand positions.
fn clip(name: &str, frames: usize, root: impl Fn(usize) -> Vec3, hips: impl Fn(usize) -> f32, hands: impl Fn(usize) -> Option<[Vec3; 2]>, swing: f32) -> Clip {
    let skeleton = skeleton();
    let poses: Vec<Pose> = (0..frames)
        .map(|f| {
            let mut pose = Pose::rest(&skeleton);
            let r = root(f);
            let h = hips(f);
            pose.local[0] = Transform::from_translation_rotation(r, Quat::IDENTITY);
            pose.local[1].translation = Vec3::new(0.0, h, 0.0);
            if let Some([l, rr]) = hands(f) {
                pose.local[2].translation = l - r - Vec3::new(0.0, h, 0.0);
                pose.local[9].translation = rr - r - Vec3::new(0.0, h, 0.0);
            }
            let s = (std::f32::consts::TAU * f as f32 / RATE).sin() * swing;
            pose.local[3].rotation = Quat::from_rotation_x(-s);
            pose.local[6].rotation = Quat::from_rotation_x(s);
            pose
        })
        .collect();
    Clip::from_poses(name, RATE, &poses)
}

fn smooth(x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
}

/// Walking into a 1 m block at 2 m/s: hands on its edge (z = 0) from frame 22, up from frame 26 to
/// 32, over the edge, standing again by frame 40 and walking on at 1 m/s.
fn mantle() -> Clip {
    mantle_standing_up_at("mantle", 32.0)
}

/// `mantle`, the hips back up to standing from frame `up` (the later, the farther it walks first).
fn mantle_standing_up_at(name: &str, up: f32) -> Clip {
    let z = |f: usize| {
        let f = f as f32;
        if f <= 25.0 { -2.0 + f * 2.0 / 30.0 } else if f <= 35.0 { -2.0 + 25.0 * 2.0 / 30.0 + (f - 25.0) * (0.2 + 2.0 - 25.0 * 2.0 / 30.0) / 10.0 } else { 0.2 + (f - 35.0) / 30.0 }
    };
    let y = |f: usize| smooth((f as f32 - 26.0) / 6.0);
    let hips = move |f: usize| if f < 26 { 1.0 } else if (f as f32) < up { 0.7 } else { 0.7 + 0.3 * smooth((f as f32 - up) / 8.0) };
    clip(name, 71, move |f| Vec3::new(0.0, y(f), z(f)), hips, |f| (22..=34).contains(&f).then_some([Vec3::new(0.2, 1.03, 0.05), Vec3::new(-0.2, 1.03, 0.05)]), 0.3)
}

/// Running at 5 m/s over a 1 m hurdle: the root up from frame 17 to 21, on top to 24, down by 28.
fn hurdle() -> Clip {
    hurdle_named("hurdle")
}

fn hurdle_named(name: &str) -> Clip {
    let y = |f: usize| {
        let f = f as f32;
        smooth((f - 17.0) / 4.0) - smooth((f - 24.0) / 4.0)
    };
    clip(name, 61, move |f| Vec3::new(0.0, y(f), -3.6 + f as f32 / 6.0), |_| 1.0, |_| None, 0.4)
}

/// Dropping 3 m (landing at frame 15), then walking.
fn land() -> Clip {
    clip("land", 50, |f| Vec3::new(0.0, (3.0 - f as f32 * 0.2).max(0.0), f as f32 * 0.05), |_| 1.0, |_| None, 0.2)
}

/// Dropping `drop` metres (landing at frame `drop / 0.2`), then walking.
fn land_from(name: &str, drop: f32) -> Clip {
    clip(name, (drop / 0.2) as usize + 35, move |f| Vec3::new(0.0, (drop - f as f32 * 0.2).max(0.0), f as f32 * 0.05), |_| 1.0, |_| None, 0.2)
}

/// A run-up at `speed` m/s, the take-off at frame 10, a 1 m apex, then falling on past the
/// take-off height.
fn jump_clip(name: &str, speed: f32) -> Clip {
    let up = (2.0f32 * 9.81).sqrt();
    let y = move |f: usize| {
        let t = (f as f32 - 10.0) / RATE;
        if t <= 0.0 { 0.0 } else { up * t - 0.5 * 9.81 * t * t }
    };
    clip(name, 61, move |f| Vec3::new(0.0, y(f), (f as f32 - 10.0) * speed / RATE), |f| if f < 10 { 0.9 } else { 1.0 }, |_| None, 0.1 + 0.2 * speed.min(1.0))
}

fn fall() -> Clip {
    clip("fall", 31, |_| Vec3::ZERO, |_| 1.0, |_| None, 0.6)
}

fn idle() -> Clip {
    clip("idle", 61, |_| Vec3::ZERO, |_| 1.0, |_| None, 0.0)
}

fn walk() -> Clip {
    clip("walk", 61, |f| Vec3::new(0.0, 0.0, f as f32 * 1.5 / 30.0), |_| 1.0, |_| None, 0.4)
}

fn database() -> (Database, Vec<ActionClip>) {
    database_with(vec![(mantle(), ActionKind::Mantle), (hurdle(), ActionKind::Hurdle), (land(), ActionKind::Land), (fall(), ActionKind::Fall)])
}

/// Idle and walk loops, then `actions` (their table in the same order).
fn database_with(actions: Vec<(Clip, ActionKind)>) -> (Database, Vec<ActionClip>) {
    let skeleton = skeleton();
    let roles = JointRoles::find(&skeleton, "root", "hips", "foot_l", "foot_r").unwrap();
    let mut b = DatabaseBuilder::new(skeleton, roles, RATE);
    b.add_clip(&idle(), true, 1).unwrap();
    b.add_clip(&walk(), true, 2).unwrap();
    for (c, _) in &actions {
        b.add_clip(c, c.name == "fall", ACTION_TAG).unwrap();
    }
    let db = b.build();
    let table = actions.iter().enumerate().map(|(i, (_, kind))| ActionClip::analyze(&db, 2 + i, *kind, HANDS).unwrap()).collect();
    (db, table)
}

#[test]
fn analysis_reads_heights_ledges_and_phases_from_the_animation() {
    let (db, table) = database();
    let mantle = table[0];
    assert!((mantle.height - 1.0).abs() < 1e-3, "{}", mantle.height);
    assert_eq!((mantle.rise, mantle.on_top, mantle.anchor), (26.0, 32.0, 22.0));
    // the planted hands at z = 0.05 put the ledge just in front of them, on the top
    assert!(mantle.ledge.abs_diff_eq(Vec3::new(0.0, 1.0, 0.0), 0.02), "{}", mantle.ledge);
    assert!(mantle.forward.abs_diff_eq(Vec3::Z, 1e-5));
    // hands back to the search once the hips stand again (0.9 m from frame 37)
    assert!(mantle.exit >= 37.0 && mantle.exit <= 41.0, "{}", mantle.exit);
    assert!((mantle.speed_at(&db, 5.0) - 2.0).abs() < 0.05);
    assert!((mantle.distance_at(&db, 0.0) - 2.0).abs() < 0.02);

    let hurdle = table[1];
    assert!((hurdle.height - 1.0).abs() < 1e-3 && hurdle.kind.crosses());
    assert!(hurdle.rise <= 17.0 && hurdle.on_top >= 20.0 && hurdle.off_top >= 24.0 && hurdle.down >= 27.0, "{hurdle:?}");
    assert!(hurdle.span > 0.3 && hurdle.span < 1.0, "{}", hurdle.span);
    assert!(hurdle.exit > hurdle.down);

    let land = table[2];
    assert_eq!(land.anchor, 15.0);
    assert!((land.height - 3.0).abs() < 1e-3);
    // clips that never leave the ground are no traversal
    assert!(ActionClip::analyze(&db, 1, ActionKind::Mantle, HANDS).is_none());
}

/// A floor, and boxes: a thin one (0.3 deep, 0.8 high), a deep one (2 m, 1.3 high), a wall
/// (2.4 m), a narrow post and a thin box with a wall right behind it.
fn course() -> CollisionWorld {
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 0.8, 0.3)));
    w.add_box(Obb::from_min_max(Vec3::new(9.0, 0.0, 0.0), Vec3::new(11.0, 1.3, 2.0)));
    w.add_box(Obb::from_min_max(Vec3::new(19.0, 0.0, 0.0), Vec3::new(21.0, 2.4, 3.0)));
    w.add_box(Obb::from_min_max(Vec3::new(29.9, 0.0, 0.0), Vec3::new(30.1, 1.0, 0.3)));
    w.add_box(Obb::from_min_max(Vec3::new(39.0, 0.0, 0.0), Vec3::new(41.0, 0.8, 0.3)));
    w.add_box(Obb::from_min_max(Vec3::new(39.0, 0.0, 0.5), Vec3::new(41.0, 3.0, 0.8)));
    w
}

#[test]
fn obstacles_are_measured_from_the_geometry() {
    let w = course();
    let s = DetectionSettings::default();
    let thin = detect_obstacle(&w, Vec3::new(0.0, 0.0, -3.0), Vec3::Z, 5.0, &s).unwrap();
    assert!(thin.ledge.abs_diff_eq(Vec3::new(0.0, 0.8, 0.0), 0.02), "{:?}", thin.ledge);
    assert!(thin.normal.abs_diff_eq(Vec3::NEG_Z, 1e-4));
    assert!((thin.height - 0.8).abs() < 1e-4 && (thin.distance - 3.0).abs() < 0.02);
    assert!((thin.depth.unwrap() - 0.3).abs() <= 0.06, "{:?}", thin.depth);
    assert_eq!(thin.back_floor, Some(0.0));
    assert!(thin.half_width >= 0.9, "{}", thin.half_width);

    let deep = detect_obstacle(&w, Vec3::new(10.0, 0.0, -2.0), Vec3::Z, 5.0, &s).unwrap();
    assert!((deep.height - 1.3).abs() < 1e-4 && deep.depth.is_none() && deep.back_floor.is_none());
    let wall = detect_obstacle(&w, Vec3::new(20.0, 0.0, -2.0), Vec3::Z, 5.0, &s).unwrap();
    assert!((wall.height - 2.4).abs() < 1e-4);
    // at an angle: the face's normal, not the approach
    let angled = detect_obstacle(&w, Vec3::new(9.0, 0.0, -2.0), Vec3::new(0.4, 0.0, 1.0), 5.0, &s).unwrap();
    assert!(angled.normal.abs_diff_eq(Vec3::NEG_Z, 1e-4) && (angled.distance - 2.0).abs() < 0.02);
    assert!(detect_obstacle(&w, Vec3::new(0.0, 0.0, -3.0), Vec3::NEG_Z, 5.0, &s).is_none(), "nothing behind");
    assert!(detect_obstacle(&w, Vec3::new(0.0, 0.0, -9.0), Vec3::Z, 5.0, &s).is_none(), "out of reach");

    let rules = TraversalRules::default();
    let kind = |feet: Vec3| traversal_kind(&w, &detect_obstacle(&w, feet, Vec3::Z, 5.0, &s).unwrap(), feet, &rules, u32::MAX);
    assert_eq!(kind(Vec3::new(0.0, 0.0, -3.0)), Ok(ActionKind::Hurdle));
    assert_eq!(kind(Vec3::new(10.0, 0.0, -2.0)), Ok(ActionKind::Mantle));
    assert_eq!(kind(Vec3::new(20.0, 0.0, -2.0)), Ok(ActionKind::Climb));
    assert_eq!(kind(Vec3::new(30.0, 0.0, -2.0)), Err(Refusal::TooNarrow));
    assert_eq!(kind(Vec3::new(40.0, 0.0, -2.0)), Err(Refusal::NoRoom));
}

fn controller(db: &Database, table: Vec<ActionClip>, at: Vec3) -> CharacterController {
    let mut matcher = MotionMatcher::new(db, MotionMatchingSettings::default(), at, 0.0);
    matcher.settings.foot_lock = false;
    CharacterController::new(matcher, table)
}

fn run(c: &mut CharacterController, db: &Database, w: &CollisionWorld, velocity: Vec3, seconds: f32) {
    let dt = 1.0 / 60.0;
    for _ in 0..(seconds / dt) as usize {
        c.update(db, w, &MotionInput { velocity, facing: None }, dt);
    }
}

#[test]
fn walls_stop_the_character_and_its_trajectory() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 1.0)));
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -3.0));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 4.0);
    let z = c.matcher.character().translation.z;
    assert!(z < -0.29 && z > -0.6, "stopped at the wall: {z}");
    assert!(c.matcher.trajectory().iter().all(|t| t.translation.z < -0.29));
    assert_eq!(c.state(), CharacterState::Grounded);
}

#[test]
fn the_character_mantles_up_a_block_and_stands_on_it() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 1.3, 3.0)));
    let mut c = controller(&db, table, Vec3::new(0.3, 0.0, -2.5));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.4);
    assert_eq!(c.traverse(&db, &w), Ok(ActionKind::Mantle));
    assert!(matches!(c.state(), CharacterState::Traversing(ActionKind::Mantle)));
    // mid-way: the warp puts the clip's ledge on the real one at the anchor frame
    let action = c.matcher.action().unwrap().clone();
    let RootPath::Warp(warp) = &action.path else { panic!() };
    let mantle = c.actions.iter().find(|a| a.kind == ActionKind::Mantle).unwrap();
    let placed = warp.to.apply(Vec3::new(mantle.ledge.x, 0.0, mantle.ledge.z));
    let ledge = c.last_obstacle.unwrap().ledge;
    assert!((placed.x - ledge.x).abs() < 1e-4 && (placed.z - ledge.z).abs() < 1e-4, "{placed} vs {ledge}");
    let (_, yaw) = warp.root(mantle.on_top, db.root_at(mantle.clip, mantle.on_top));
    assert!(yaw.abs() < 1e-3, "faces into the block: {yaw}");
    let (top, _) = warp.root(mantle.on_top + 2.0, db.root_at(mantle.clip, mantle.on_top + 2.0));
    assert!((top.y - 1.3).abs() < 1e-3, "lifted to the real top: {top}");
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 3.0);
    let p = c.matcher.character().translation;
    assert_eq!(c.state(), CharacterState::Grounded);
    assert!((p.y - 1.3).abs() < 0.02 && p.z > 0.2, "on the block: {p}");
}

#[test]
fn the_character_hurdles_a_thin_box_and_runs_on() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.7, 0.3)));
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -2.4));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.5);
    assert_eq!(c.traverse(&db, &w), Ok(ActionKind::Hurdle));
    let mut highest: f32 = 0.0;
    for _ in 0..120 {
        c.update(&db, &w, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, 1.0 / 60.0);
        let p = c.matcher.character().translation;
        highest = highest.max(p.y);
        // over the box, never through it
        if p.z > 0.0 && p.z < 0.3 {
            assert!(p.y > 0.6, "over the box at {p}");
        }
    }
    let p = c.matcher.character().translation;
    assert!((highest - 0.7).abs() < 0.05, "up to the real top: {highest}");
    assert!(p.z > 0.6 && p.y.abs() < 0.02, "beyond it on the floor: {p}");
    assert_eq!(c.state(), CharacterState::Grounded);
}

#[test]
fn walking_off_a_ledge_falls_and_lands() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, -4.0), Vec3::new(2.0, 1.5, 0.0)));
    let mut c = controller(&db, table, Vec3::new(0.0, 1.5, -1.5));
    c.matcher.set_ground(1.5);
    let mut states = Vec::new();
    for _ in 0..240 {
        c.update(&db, &w, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, 1.0 / 60.0);
        let s = std::mem::discriminant(&c.state());
        if states.last() != Some(&s) {
            states.push(s);
        }
    }
    let p = c.matcher.character().translation;
    assert!(p.y.abs() < 0.02 && p.z > 0.3, "down on the floor: {p}");
    assert_eq!(states, [CharacterState::Grounded, CharacterState::Falling(0.0), CharacterState::Landing, CharacterState::Grounded].map(|s| std::mem::discriminant(&s)));
}

#[test]
fn a_request_made_early_traverses_once_in_reach() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 1.3, 3.0)));
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -6.0));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.3);
    // far out of reach: refused now, but kept trying
    assert!(c.request_traverse(&db, &w, 3.0).is_err());
    let mut started = false;
    for _ in 0..180 {
        c.update(&db, &w, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, 1.0 / 60.0);
        started |= matches!(c.state(), CharacterState::Traversing(ActionKind::Mantle));
    }
    assert!(started, "mantled once in reach: {:?}", c.last_result);
}

#[test]
fn a_mantle_is_chosen_by_where_it_leaves_the_character() {
    // one mantle stands up by the edge, the other walks on a metre before handing over
    let (db, table) = database_with(vec![(mantle(), ActionKind::Mantle), (mantle_standing_up_at("mantle_on", 50.0), ActionKind::Mantle)]);
    let (near, far) = (table[0], table[1]);
    let past_ledge = |c: &ActionClip| (db.root_at(c.clip, c.exit).0 - c.ledge).dot(c.forward);
    assert!(past_ledge(&near) < 0.5 && past_ledge(&far) > 0.8, "{} {}", past_ledge(&near), past_ledge(&far));
    // a block with a wall on it 1.1 m back from its edge: room to stand by the edge, not beyond
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 1.3, 3.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 1.3, 1.1), Vec3::new(2.0, 3.3, 3.0)));
    let rules = TraversalRules::default();
    let feet = Vec3::new(0.0, 0.0, -2.0);
    let obstacle = detect_obstacle(&w, feet, Vec3::Z, 5.0, &DetectionSettings::default()).unwrap();
    assert_eq!(traversal_kind(&w, &obstacle, feet, &rules, u32::MAX), Ok(ActionKind::Mantle));
    let stands = |feet: Vec3| stands_at(&w, feet, &rules, u32::MAX);
    let plan = |table: &[ActionClip]| plan_traversal(&db, table, ActionKind::Mantle, &obstacle, feet, 0.0, 0.0, db.clips[0].start, &rules, stands);
    assert_eq!(plan(&[far]).err(), Some(Refusal::NoRoom));
    assert_eq!(plan(&table).map(|a| a.clip), Ok(near.clip));
    // without the wall, either fits
    let open = course();
    assert!(plan_traversal(&db, &[far], ActionKind::Mantle, &obstacle, feet, 0.0, 0.0, db.clips[0].start, &rules, |f| stands_at(&open, f, &rules, u32::MAX)).is_ok());
}

#[test]
fn pushing_the_character_out_of_a_wall_gives_it_no_speed() {
    let (db, table) = database();
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 1.0)));
    // standing 0.2 m into the wall (as an action can leave it)
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -0.1));
    run(&mut c, &db, &w, Vec3::ZERO, 1.0);
    let z = c.matcher.character().translation.z;
    assert!(z < -0.29 && z > -0.45, "out of the wall, and no farther: {z}");
    assert!(c.matcher.simulation().velocity.length() < 0.1, "{}", c.matcher.simulation().velocity);
}

/// Standing and walking jumps (and a running one if `run`), light and heavy landings, the fall
/// loop and a hurdle.
fn jump_database(run: bool) -> (Database, Vec<ActionClip>) {
    let mut actions = vec![
        (jump_clip("jump_stand", 0.0), ActionKind::Jump),
        (jump_clip("jump_walk", 1.5), ActionKind::Jump),
        (land(), ActionKind::Land),
        (land_from("land_heavy", 6.0), ActionKind::Land),
        (fall(), ActionKind::Fall),
        (hurdle(), ActionKind::Hurdle),
    ];
    if run {
        actions.push((jump_clip("jump_run", 4.0), ActionKind::Jump));
    }
    database_with(actions)
}

fn floor() -> CollisionWorld {
    let mut w = CollisionWorld::new();
    w.add_box(Obb::from_min_max(Vec3::new(-100.0, -1.0, -100.0), Vec3::new(100.0, 0.0, 100.0)));
    w
}

/// Runs `c` for `seconds`, walking along +Z, and returns each state and clip it went through (an
/// entry whenever either changes), and the highest point it reached.
fn fly(c: &mut CharacterController, db: &Database, w: &CollisionWorld, seconds: f32) -> (Vec<(CharacterState, String)>, f32) {
    let mut seen: Vec<(CharacterState, String)> = Vec::new();
    let mut highest = f32::MIN;
    for _ in 0..(seconds * 60.0) as usize {
        c.update(db, w, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, 1.0 / 60.0);
        highest = highest.max(c.matcher.character().translation.y);
        let state = match c.state() {
            CharacterState::Falling(_) => CharacterState::Falling(0.0),
            s => s,
        };
        let clip = db.clips[c.matcher.playing().0].name.clone();
        if seen.last().is_none_or(|(s, n)| *s != state || *n != clip) {
            seen.push((state, clip));
        }
    }
    (seen, highest)
}

#[test]
fn analysis_reads_the_take_off_and_apex_of_a_jump() {
    let (db, table) = jump_database(false);
    let walk = table.iter().find(|a| db.clips[a.clip].name == "jump_walk").unwrap();
    assert_eq!(walk.kind, ActionKind::Jump);
    assert_eq!(walk.rise, 10.0);
    assert!((walk.anchor - 23.5).abs() <= 0.5, "apex at {}", walk.anchor);
    assert!((walk.height - 1.0).abs() < 0.01, "{}", walk.height);
    assert!((walk.speed_at(&db, 5.0) - 1.5).abs() < 0.01);
}

#[test]
fn a_jump_takes_off_flies_as_high_as_asked_and_lands_light() {
    let (db, table) = jump_database(false);
    let w = floor();
    let mut c = controller(&db, table, Vec3::ZERO);
    c.jump_height = Some(0.8);
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 1.0);
    assert_eq!(c.jump(&db), Ok(ActionKind::Jump));
    // the run-up that goes with walking
    assert_eq!(db.clips[c.matcher.action().unwrap().clip].name, "jump_walk");
    let (seen, highest) = fly(&mut c, &db, &w, 3.0);
    assert!((highest - 0.8).abs() < 0.03, "as high as asked: {highest}");
    let mut states: Vec<_> = seen.iter().map(|(s, _)| *s).collect();
    states.dedup();
    assert_eq!(states, [CharacterState::Jumping, CharacterState::Falling(0.0), CharacterState::Landing, CharacterState::Grounded]);
    assert!(seen.contains(&(CharacterState::Landing, "land".into())), "a light landing for a 0.8 m fall: {seen:?}");
    let p = c.matcher.character().translation;
    assert!(p.y.abs() < 0.02 && p.z > 1.5, "down and on: {p}");
}

#[test]
fn a_running_jump_lands_on_top_of_a_box() {
    let (db, mut table) = jump_database(true);
    table.retain(|a| a.kind != ActionKind::Jump || db.clips[a.clip].name == "jump_run");
    // 0.5 m high (more than a step), its front 2.8 m ahead
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 2.8), Vec3::new(2.0, 0.5, 20.0)));
    let mut c = controller(&db, table, Vec3::ZERO);
    assert_eq!(c.jump(&db), Ok(ActionKind::Jump));
    let (seen, _) = fly(&mut c, &db, &w, 3.0);
    assert!(seen.iter().any(|(s, _)| *s == CharacterState::Landing), "{seen:?}");
    let p = c.matcher.character().translation;
    assert_eq!(c.state(), CharacterState::Grounded);
    assert!((p.y - 0.5).abs() < 0.02 && p.z > 2.8, "on the box: {p}");
}

#[test]
fn a_wall_stops_a_jump() {
    let (db, table) = jump_database(false);
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 1.0), Vec3::new(2.0, 3.0, 2.0)));
    let mut c = controller(&db, table, Vec3::ZERO);
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.2);
    assert_eq!(c.jump(&db), Ok(ActionKind::Jump));
    for _ in 0..180 {
        c.update(&db, &w, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, 1.0 / 60.0);
        let z = c.matcher.character().translation.z;
        assert!(z < 0.72, "kept out of the wall: {z}");
    }
    assert_eq!(c.state(), CharacterState::Grounded);
    assert!(c.matcher.character().translation.y.abs() < 0.02);
}

#[test]
fn a_jump_off_a_high_platform_goes_on_with_the_fall_loop_and_lands_heavy() {
    let (db, table) = jump_database(false);
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-5.0, 0.0, -5.0), Vec3::new(5.0, 10.0, 0.0)));
    let mut c = controller(&db, table, Vec3::new(0.0, 10.0, -2.6));
    c.matcher.set_ground(10.0);
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 1.0);
    assert_eq!(c.jump(&db), Ok(ActionKind::Jump));
    let (seen, _) = fly(&mut c, &db, &w, 4.0);
    // off the edge in the air; the jump clip runs out before the ground, the fall loop goes on
    let expected: Vec<(CharacterState, String)> = [
        (CharacterState::Jumping, "jump_walk"),
        (CharacterState::Falling(0.0), "jump_walk"),
        (CharacterState::Falling(0.0), "fall"),
        (CharacterState::Landing, "land_heavy"),
    ]
    .map(|(s, n)| (s, n.to_string()))
    .to_vec();
    assert_eq!(seen[..4], expected[..], "{seen:?}");
    assert_eq!(c.state(), CharacterState::Grounded);
    assert!(c.matcher.character().translation.y.abs() < 0.02);
}

#[test]
fn space_jumps_unless_there_is_something_to_traverse() {
    let (db, table) = jump_database(false);
    // open floor: a jump
    let w = floor();
    let mut c = controller(&db, table.clone(), Vec3::ZERO);
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.5);
    assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Ok(ActionKind::Jump));
    // a thin box ahead: a hurdle
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-2.0, 0.0, 0.0), Vec3::new(2.0, 0.7, 0.3)));
    let mut c = controller(&db, table.clone(), Vec3::new(0.0, 0.0, -2.4));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.5);
    assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Ok(ActionKind::Hurdle));
    // the end of a beam: too narrow to traverse, so a jump
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-0.175, 0.0, 1.0), Vec3::new(0.175, 0.6, 8.0)));
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -0.8));
    run(&mut c, &db, &w, Vec3::new(0.0, 0.0, 1.5), 0.3);
    assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Ok(ActionKind::Jump));
    assert_eq!(c.last_obstacle.map(|o| o.half_width < 0.3), Some(true));
}

/// A hurdle, a vault, a mantle and a climb (the synthetic hurdle and mantle, lifted by the warp),
/// the fall loop, a landing and a jump: every clip's run-up starts well back from its ledge (the
/// mantle 2 m, reaching 0.67 m by its `last_entry`), like captured ones.
fn contact_database() -> (Database, Vec<ActionClip>) {
    database_with(vec![
        (hurdle(), ActionKind::Hurdle),
        (hurdle_named("vault"), ActionKind::Vault),
        (mantle(), ActionKind::Mantle),
        (mantle_standing_up_at("climb", 32.0), ActionKind::Climb),
        (fall(), ActionKind::Fall),
        (land(), ActionKind::Land),
        (jump_clip("jump_walk", 1.5), ActionKind::Jump),
    ])
}

/// A box 20 m wide whose front face is at z = 0, `height` high and `depth` deep, on the floor.
fn block(height: f32, depth: f32) -> CollisionWorld {
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-10.0, 0.0, 0.0), Vec3::new(10.0, height, depth)));
    w
}

/// The boxes each kind of traversal is for.
const BLOCKS: [(ActionKind, f32, f32); 4] = [(ActionKind::Hurdle, 0.8, 0.3), (ActionKind::Vault, 1.0, 0.9), (ActionKind::Mantle, 1.3, 3.0), (ActionKind::Climb, 2.4, 3.0)];

/// Moves `c` under `input` until `stop` says so (or `seconds` run out); true if it stopped.
fn step_until(c: &mut CharacterController, db: &Database, w: &CollisionWorld, velocity: Vec3, seconds: f32, mut stop: impl FnMut(&CharacterController) -> bool) -> bool {
    for _ in 0..(seconds * 60.0) as usize {
        c.update(db, w, &MotionInput { velocity, facing: None }, 1.0 / 60.0);
        if stop(c) {
            return true;
        }
    }
    false
}

/// Plays out a traversal started just now, the input still pushing along `velocity`: the most
/// it slid backward (away from the face at z = 0) in a frame, and where it was once back on its
/// feet.
fn play_out(c: &mut CharacterController, db: &Database, w: &CollisionWorld, velocity: Vec3) -> (f32, Vec3) {
    let mut last = c.matcher.character().translation;
    let mut back: f32 = 0.0;
    step_until(c, db, w, velocity, 4.0, |c| {
        let p = c.matcher.character().translation;
        back = back.max(last.z - p.z);
        last = p;
        c.state() == CharacterState::Grounded
    });
    (back, last)
}

#[test]
fn pressed_against_an_obstacle_of_any_height_it_traverses() {
    let (db, table) = contact_database();
    for (kind, height, depth) in BLOCKS {
        let w = block(height, depth);
        let mut c = controller(&db, table.clone(), Vec3::new(0.0, 0.0, -3.0));
        // walked into it and still pushing
        let forward = Vec3::new(0.0, 0.0, 1.5);
        run(&mut c, &db, &w, forward, 3.0);
        let z = c.matcher.character().translation.z;
        assert!(z > -0.35 && z < -0.25, "{kind:?}: against it at {z}");
        assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Ok(kind), "{kind:?}: {:?}", c.last_obstacle);
        let (back, end) = play_out(&mut c, &db, &w, forward);
        // the clip starts late rather than sliding the character back to its run-up
        assert!(back < 0.01, "{kind:?}: slid back {back} m in a frame");
        if kind.crosses() {
            assert!(end.z > depth && end.y.abs() < 0.02, "{kind:?}: over it, on the floor: {end}");
        } else {
            assert!(end.z > 0.1 && (end.y - height).abs() < 0.02, "{kind:?}: on top: {end}");
        }
    }
}

#[test]
fn running_into_an_obstacle_it_traverses_the_moment_it_meets_it() {
    let (db, table) = contact_database();
    for (kind, height, depth) in BLOCKS {
        let w = block(height, depth);
        let mut c = controller(&db, table.clone(), Vec3::new(0.0, 0.0, -6.0));
        let forward = Vec3::new(0.0, 0.0, 4.0);
        // Space as it meets the face
        assert!(step_until(&mut c, &db, &w, forward, 3.0, |c| c.matcher.character().translation.z > -0.36), "{kind:?}: never reached it");
        let result = c.request_traverse_or_jump(&db, &w, 1.0);
        let started = result == Ok(kind) || step_until(&mut c, &db, &w, forward, 0.2, |c| c.state() == CharacterState::Traversing(kind));
        assert!(started, "{kind:?}: {result:?}, {:?}", c.last_result);
    }
}

#[test]
fn pushed_along_an_obstacle_at_an_angle_it_traverses_rather_than_jumps() {
    let (db, table) = contact_database();
    for (kind, height, depth) in BLOCKS {
        for degrees in [30.0f32, 50.0] {
            let w = block(height, depth);
            let mut c = controller(&db, table.clone(), Vec3::new(0.0, 0.0, -3.0));
            let a = degrees.to_radians();
            let heading = Vec3::new(a.sin(), 0.0, a.cos()) * 1.5;
            assert!(step_until(&mut c, &db, &w, heading, 4.0, |c| c.matcher.character().translation.z > -0.36));
            run(&mut c, &db, &w, heading, 0.5);
            // sliding along the face: the wall leaves it no speed into it
            let v = c.matcher.simulation().velocity;
            assert!(v.z.abs() < 0.2 && v.x > 0.2, "{kind:?} at {degrees}: sliding along it {v}");
            assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Ok(kind), "{kind:?} at {degrees}: {:?}", c.last_obstacle);
            // squared up to it
            step_until(&mut c, &db, &w, heading, 0.4, |_| false);
            let yaw = yaw_of(c.matcher.character().rotation);
            assert!(yaw.abs() < 0.05, "{kind:?} at {degrees}: facing into it, {yaw}");
        }
    }
}

#[test]
fn space_pressed_as_a_landing_ends_runs_once_it_is_back_on_its_feet() {
    let (db, table) = contact_database();
    let w = floor();
    let forward = Vec3::new(0.0, 0.0, 1.5);
    let jumping = || {
        let mut c = controller(&db, table.clone(), Vec3::ZERO);
        run(&mut c, &db, &w, forward, 0.5);
        assert_eq!(c.jump(&db), Ok(ActionKind::Jump));
        c
    };
    // the frame (after the jump) the landing ends
    let mut states = Vec::new();
    step_until(&mut jumping(), &db, &w, forward, 4.0, |c| {
        states.push(c.state());
        false
    });
    let end = states.iter().rposition(|s| *s == CharacterState::Landing).unwrap();
    // whether a Space pressed this many frames before then jumps again once it has landed
    let jump_again = |early: usize| -> bool {
        let mut c = jumping();
        for _ in 0..=end - early {
            c.update(&db, &w, &MotionInput { velocity: forward, facing: None }, 1.0 / 60.0);
        }
        assert_eq!(c.state(), CharacterState::Landing);
        assert_eq!(c.request_traverse_or_jump(&db, &w, 1.0), Err(Refusal::Busy));
        step_until(&mut c, &db, &w, forward, 0.5, |c| c.state() == CharacterState::Jumping)
    };
    assert!(jump_again(6), "pressed 0.1 s before the landing ends: jumps as it ends");
    assert!(!jump_again(24), "pressed 0.4 s before: dropped");
}

#[test]
fn from_right_against_an_obstacle_a_clip_starts_late_and_lands_its_ledge_just_past_the_edge() {
    let (db, table) = contact_database();
    let w = block(1.3, 3.0);
    let rules = TraversalRules::default();
    let mantle = *table.iter().find(|a| a.kind == ActionKind::Mantle).unwrap();
    let plan = |feet: Vec3| {
        let obstacle = detect_obstacle(&w, feet, Vec3::Z, 5.0, &DetectionSettings::default()).unwrap();
        let action = plan_traversal(&db, &[mantle], ActionKind::Mantle, &obstacle, feet, 0.0, 0.0, db.clips[0].start, &rules, |_| true).unwrap();
        let RootPath::Warp(warp) = &action.path else { panic!() };
        let placed = warp.to.apply(Vec3::new(mantle.ledge.x, 0.0, mantle.ledge.z));
        (action.start, placed.z - obstacle.ledge.z)
    };
    // from its run-up: within it, the ledge on the edge
    let (start, past) = plan(Vec3::new(0.0, 0.0, -1.5));
    assert!(start <= mantle.last_entry && past.abs() < 1e-3, "{start} {past}");
    // against it (0.3 m, closer than the run-up comes before the hands reach the ledge): after
    // `last_entry`, no later than the hands reach it, the ledge at most `ledge_tolerance` past
    let (start, past) = plan(Vec3::new(0.0, 0.0, -0.3));
    assert!(start > mantle.last_entry && start <= mantle.latest_entry(), "{start} ({} to {})", mantle.last_entry, mantle.latest_entry());
    assert!((-1e-3..=rules.ledge_tolerance + 1e-3).contains(&past), "{past}");
}

#[test]
fn pushed_into_a_wall_at_an_angle_it_slides_along_it() {
    let (db, table) = contact_database();
    let w = block(2.4, 3.0);
    let mut c = controller(&db, table, Vec3::new(0.0, 0.0, -3.0));
    let a = 30f32.to_radians();
    let heading = Vec3::new(a.sin(), 0.0, a.cos()) * 1.5;
    assert!(step_until(&mut c, &db, &w, heading, 4.0, |c| c.matcher.character().translation.z > -0.36));
    run(&mut c, &db, &w, heading, 0.5);
    let x = c.matcher.simulation().position.x;
    run(&mut c, &db, &w, heading, 1.0);
    // the input's pace along the wall (0.75 m/s), none into it
    let v = c.matcher.simulation().velocity;
    assert!((v.x - 0.75).abs() < 0.05 && v.z.abs() < 0.05, "{v}");
    let moved = c.matcher.simulation().position.x - x;
    assert!(moved > 0.65 && moved < 0.8, "{moved} m along the wall in a second");
}

#[test]
fn against_a_face_the_ledge_is_in_front_of_the_feet() {
    // a box 2 m wide; the feet against its face 0.35 m from one end, probing at 45 degrees toward
    // that end (where the probe meets the face only 0.1 m from the corner)
    let mut w = floor();
    w.add_box(Obb::from_min_max(Vec3::new(-1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.9)));
    let s = DetectionSettings::default();
    let feet = Vec3::new(-0.65, 0.0, -0.3);
    let o = detect_obstacle(&w, feet, Vec3::new(-1.0, 0.0, 1.0), 2.0, &s).unwrap();
    assert!(o.ledge.abs_diff_eq(Vec3::new(-0.65, 1.0, 0.0), 0.01), "{}", o.ledge);
    assert!(o.half_width >= 0.3, "{}", o.half_width);
    assert_eq!(traversal_kind(&w, &o, feet, &TraversalRules::default(), u32::MAX), Ok(ActionKind::Vault));
    // from farther away, where the probe meets it
    let o = detect_obstacle(&w, Vec3::new(0.5, 0.0, -1.5), Vec3::new(-1.0, 0.0, 1.0), 3.0, &s).unwrap();
    assert!(o.ledge.abs_diff_eq(Vec3::new(-0.88, 1.0, 0.0), 0.05), "{}", o.ledge);
}
