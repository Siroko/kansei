//! Measures the parkour moves (`motion_matching::traversal::parkour`) of a motion pack on a test
//! course: scripted runs over platforms, a ledge tower, a wall and bars, at several frame rates and
//! with frame hitches. Per scenario it prints how far the limbs pinned to the level miss their
//! targets, how far planted feet slide, limbs inside the level, and pops (the largest joint
//! acceleration at a change of state, against the run's typical).
//!
//! ```sh
//! cargo run -p kansei-core --release --example parkour_eval -- <pack.kmm> [only=<name part>] [csv=<dir>]
//! ```
//!
//! Packs are private: link them in (see the motion-matching demo's README), never commit them.
//! `csv=` writes each run's frames (state, limb errors, the largest joint acceleration) for plots.

#![allow(clippy::type_complexity, clippy::needless_range_loop, clippy::manual_is_multiple_of)]

use glam::{Quat, Vec3};
use kansei_core::animation::motion_matching::pack::MotionPack;
use kansei_core::animation::motion_matching::traversal::features::{ParkourFeatures, ParkourPiece, ParkourTag, PieceProps};
use kansei_core::animation::motion_matching::traversal::{CharacterController, CharacterState};
use kansei_core::animation::motion_matching::{yaw_of, Database, MotionInput, MotionMatcher, MotionMatchingSettings, ACTION_TAG};
use kansei_core::collision::{CollisionWorld, Obb};

/// The course the room's Blender scene tags (its south-east corner), and a floor.
fn course() -> (CollisionWorld, ParkourFeatures) {
    let mut world = CollisionWorld::new();
    world.add_box(Obb::from_min_max(Vec3::new(-30.0, -1.0, -30.0), Vec3::new(30.0, 0.0, 30.0)));
    let mut pieces = Vec::new();
    let mut add = |name: &str, tag: ParkourTag, center: Vec3, half: Vec3, props: &[(&str, &str)]| {
        let obb = Obb::new(center, half, Quat::IDENTITY);
        let mut p = PieceProps::default();
        for (k, v) in props {
            p.set(k, v);
        }
        pieces.push(ParkourPiece::new(name, tag, obb).with_props(p));
    };
    for (k, (x, h)) in [(5.2f32, 1.0f32), (7.8, 1.2), (10.6, 1.4), (13.6, 1.2)].into_iter().enumerate() {
        add(&format!("Course_Platform_{}", k + 1), ParkourTag::Platform, Vec3::new(x, h * 0.5, 11.6), Vec3::new(0.8, h * 0.5, 0.8), &[("kansei_chain", "south"), ("kansei_chain_index", &k.to_string())]);
    }
    add("Course_Tower", ParkourTag::Ledge, Vec3::new(14.4, 1.4, 8.0), Vec3::new(1.0, 1.4, 1.0), &[]);
    add("Course_Wall", ParkourTag::WallRun, Vec3::new(10.0, 1.6, 3.0), Vec3::new(2.5, 1.6, 0.12), &[("kansei_run_axis", "x"), ("kansei_run_faces", "both")]);
    for (k, z) in [8.4f32, 6.6, 4.8].into_iter().enumerate() {
        add(&format!("Course_Bar_{}", k + 1), ParkourTag::Bar, Vec3::new(7.6, 1.17, z), Vec3::new(1.14, 1.17, 0.04), &[("kansei_bar_height", "2.3"), ("kansei_bar_axis", "x")]);
    }
    for p in &pieces {
        match p.tag {
            // a bar collides as its posts
            ParkourTag::Bar => {
                let (lo, _) = p.obb.bounds();
                for sx in [-1.1f32, 1.1] {
                    world.add_box(Obb::new(Vec3::new(p.obb.center.x + sx, lo.y + 1.17, p.obb.center.z), Vec3::new(0.04, 1.17, 0.04), Quat::IDENTITY));
                }
                // (the tube is not a collider: a body swinging up to it must not be pushed off)
            }
            _ => {
                world.add_box(p.obb);
            }
        }
    }
    let features = ParkourFeatures::from_pieces(&pieces, &world);
    (world, features)
}

/// One scripted input from time `at` on: the stick (world direction and pace, m/s), and a Space
/// or drop press at that moment.
#[derive(Clone, Copy)]
struct Cue {
    at: f32,
    velocity: Vec3,
    space: bool,
    drop: bool,
}

fn cue(at: f32, velocity: Vec3) -> Cue {
    Cue { at, velocity, space: false, drop: false }
}

fn space(at: f32, velocity: Vec3) -> Cue {
    Cue { at, velocity, space: true, drop: false }
}

fn drop(at: f32) -> Cue {
    Cue { at, velocity: Vec3::ZERO, space: false, drop: true }
}

struct Scenario {
    name: &'static str,
    start: Vec3,
    yaw: f32,
    duration: f32,
    cues: Vec<Cue>,
    /// States the run must go through, in order.
    expect: &'static [&'static str],
}

fn label_is_60(dt: f32, hitch: bool) -> bool {
    (dt - 1.0 / 60.0).abs() < 1e-6 && !hitch
}

fn state_name(s: CharacterState) -> &'static str {
    match s {
        CharacterState::Grounded => "grounded",
        CharacterState::Traversing(_) => "traversing",
        CharacterState::Jumping => "jumping",
        CharacterState::Falling(_) => "falling",
        CharacterState::Landing => "landing",
        CharacterState::Hanging => "hanging",
        CharacterState::WallRunning => "wallrun",
        CharacterState::Swinging => "swinging",
    }
}

fn scenarios() -> Vec<Scenario> {
    let x = Vec3::X;
    let z = Vec3::Z;
    vec![
        Scenario {
            name: "baseline: walk, run, turn, stop (no parkour)",
            start: Vec3::new(-10.0, 0.0, -10.0),
            yaw: 0.0,
            duration: 8.0,
            cues: vec![cue(0.0, z * 2.0), cue(2.0, z * 5.0), cue(4.0, x * 5.0), cue(5.5, -z * 2.0), cue(7.0, Vec3::ZERO)],
            expect: &["grounded"],
        },
        Scenario {
            name: "gap jumps along the chain",
            start: Vec3::new(4.8, 1.0, 11.6),
            yaw: std::f32::consts::FRAC_PI_2,
            duration: 9.0,
            cues: vec![cue(0.0, x * 2.0), space(0.25, x * 2.0), cue(1.6, Vec3::ZERO), space(2.6, x * 2.0), cue(4.0, Vec3::ZERO), space(5.0, x * 2.0), cue(6.4, Vec3::ZERO)],
            expect: &["jumping", "falling", "landing", "grounded", "jumping", "falling", "grounded", "jumping", "falling", "grounded"],
        },
        Scenario {
            name: "climb to hang, shimmy round a corner, back, climb up",
            start: Vec3::new(14.4, 0.0, 10.8),
            yaw: std::f32::consts::PI,
            duration: 14.0,
            cues: vec![cue(0.0, -z * 2.0), space(0.3, -z * 2.0), cue(1.0, Vec3::ZERO), cue(3.0, x * 2.0), cue(7.0, Vec3::ZERO), cue(8.0, -x * 2.0), cue(10.5, Vec3::ZERO), space(11.5, Vec3::ZERO)],
            expect: &["traversing", "hanging", "traversing", "grounded"],
        },
        Scenario {
            name: "hang, drop, land",
            start: Vec3::new(14.0, 0.0, 10.8),
            yaw: std::f32::consts::PI,
            duration: 6.0,
            cues: vec![cue(0.0, -z * 2.0), space(0.3, -z * 2.0), cue(1.0, Vec3::ZERO), drop(3.5)],
            expect: &["traversing", "hanging", "falling", "grounded"],
        },
        Scenario {
            name: "jump from the last platform, catch the tower",
            start: Vec3::new(13.6, 1.2, 12.2),
            yaw: std::f32::consts::PI,
            duration: 6.0,
            cues: vec![cue(0.0, -z * 4.0), space(0.35, -z * 4.0), cue(1.2, Vec3::ZERO)],
            expect: &["jumping", "falling", "hanging"],
        },
        Scenario {
            name: "wall run along the wall's south face, jump off",
            start: Vec3::new(6.5, 0.0, 3.9),
            yaw: std::f32::consts::FRAC_PI_2,
            duration: 5.0,
            cues: vec![cue(0.0, x * 5.0), space(0.75, x * 5.0), space(1.5, x * 5.0), cue(2.6, x * 2.0)],
            expect: &["wallrun", "falling", "grounded"],
        },
        Scenario {
            name: "wall run to its end",
            start: Vec3::new(5.5, 0.0, 2.1),
            yaw: std::f32::consts::FRAC_PI_2,
            duration: 5.0,
            cues: vec![cue(0.0, x * 5.0), space(0.8, x * 5.0)],
            expect: &["wallrun", "falling", "grounded"],
        },
        Scenario {
            name: "jump to a bar, hang still, pull up, let go",
            start: Vec3::new(7.6, 0.0, 11.0),
            yaw: std::f32::consts::PI,
            duration: 9.0,
            cues: vec![cue(0.0, -z * 4.0), space(0.45, -z * 4.0), cue(1.2, Vec3::ZERO), space(4.0, Vec3::ZERO), drop(7.5)],
            expect: &["jumping", "falling", "swinging", "falling", "grounded"],
        },
        Scenario {
            name: "jump to a bar, swing, let go",
            start: Vec3::new(7.6, 0.0, 11.0),
            yaw: std::f32::consts::PI,
            duration: 7.0,
            cues: vec![cue(0.0, -z * 4.0), space(0.45, -z * 4.0), cue(1.2, -z * 2.0), cue(3.4, Vec3::ZERO), space(4.25, Vec3::ZERO)],
            expect: &["jumping", "falling", "swinging", "falling"],
        },
    ]
}

#[derive(Default, Clone)]
struct Report {
    states: Vec<&'static str>,
    limb_error_max: f32,
    limb_error_mean: f32,
    limb_frames: usize,
    slide_cm_s: f32,
    planted_frames: usize,
    inside: usize,
    pop_max: f32,
    pop_typical: f32,
    pop_at: f32,
    pop_state: String,
    root_max: f32,
    root_typical: f32,
    root_at: f32,
    root_state: String,
    end: Vec3,
}

#[allow(clippy::too_many_arguments)]
fn run(db: &Database, actions: &[kansei_core::animation::motion_matching::traversal::ActionClip], world: &CollisionWorld, features: &ParkourFeatures, s: &Scenario, dt: f32, hitch: bool, tags: (u32, u32), csv: Option<&mut String>, trace: bool) -> Report {
    let matcher = MotionMatcher::new(db, MotionMatchingSettings::default(), s.start, s.yaw);
    let mut c = CharacterController::new(matcher, actions.to_vec());
    c.features = features.clone();
    c.find_hands(db);
    let mut r = Report::default();
    let mut time = 0.0f32;
    let mut next_cue = 0;
    let mut stick = Vec3::ZERO;
    let joints = db.joint_count();
    // the body's core: a pop there is a pop (a foot striking the ground stops as fast as any pop)
    let body: Vec<usize> = (0..joints).filter(|&j| j == db.roles.hips || ["spine", "neck", "head"].iter().any(|n| db.skeleton.names[j].to_lowercase().contains(n))).collect();
    let balls: Vec<usize> = db.roles.feet.iter().map(|&f| db.skeleton.parents.iter().position(|p| *p == Some(f)).unwrap_or(f)).collect();
    let mut history: Vec<(f32, bool, Vec<Vec3>)> = Vec::new();
    let mut accel_log: Vec<(f32, f32, f32, usize, &'static str)> = Vec::new();
    let mut last_state = "";
    let mut planted_prev: [Option<[Vec3; 2]>; 2] = [None, None];
    let mut slide = 0.0f32;
    let mut csv = csv;
    let mut frame = 0usize;
    let mut sum_err = 0.0f32;
    let mut misses = 0;
    while time < s.duration {
        // a hitch: a 100 ms frame every 1.3 s
        let step = if hitch && frame % 78 == 77 { 0.1 } else { dt };
        let mut press = false;
        let mut drop_press = false;
        while next_cue < s.cues.len() && s.cues[next_cue].at <= time {
            stick = s.cues[next_cue].velocity;
            press |= s.cues[next_cue].space;
            drop_press |= s.cues[next_cue].drop;
            next_cue += 1;
        }
        c.matcher.settings.filter.tags = if stick.length() > 3.0 { tags.1 } else { tags.0 };
        c.update(db, world, &MotionInput { velocity: stick, facing: None }, step);
        if press {
            let result = c.request_traverse_or_jump(db, world, 1.0);
            if trace {
                println!("  {time:.2}s Space -> {:?} (state {}, at {:.2?}, obstacle {:?}, target {:?})", result.map(|k| k.name()).map_err(|e| e.describe()), state_name(c.state()), c.matcher.character().translation, c.last_obstacle.map(|o| (o.height, o.distance, o.depth)), c.jump_target());
            }
        }
        if drop_press {
            c.request_drop(db);
        }
        time += step;
        frame += 1;
        let state = state_name(c.state());
        let changed = state != last_state;
        if changed {
            if trace {
                let ch = c.matcher.character().translation;
                println!("  {time:.2}s {state} at ({:.2},{:.2},{:.2}) clip {}", ch.x, ch.y, ch.z, db.clips[c.matcher.playing().0].name);
            }
            r.states.push(state);
            last_state = state;
        }
        if trace && frame.is_multiple_of(15) && time > 1.4 && time < 3.0 {
            let ch = c.matcher.character();
            let (z, y, x) = ch.rotation.to_euler(glam::EulerRot::ZYX);
            println!("    {time:.2}s {state} root {:.2?} euler x {:.2} y {:.2} z {:.2} sim {:.2?} clip {} frame {:.1}", ch.translation, x, y, z, c.matcher.simulation().position, db.clips[c.matcher.playing().0].name, c.matcher.playing().1);
        }
        if std::env::var("PARKOUR_DEBUG").is_ok() && time > 0.98 && time < 1.2 && label_is_60(dt, hitch) {
            let head = db.skeleton.find("head").unwrap_or(0);
            let r = c.matcher.shown().rotation;
            eprintln!("  rot {:.3?} char {:.3?}", r, c.matcher.character().rotation);
            eprintln!("  {time:.3} {state} head {:.3?} hips {:.3?} shown {:.3?} char {:.3?} clip {} {:.1}", c.matcher.joint_world(db, head), c.matcher.joint_world(db, db.roles.hips), c.matcher.shown().translation, c.matcher.character().translation, db.clips[c.matcher.playing().0].name, c.matcher.playing().1);
        }
        // limbs on their targets
        if trace && (time - 1.5).abs() < dt * 0.5 {
            for t in c.limb_targets() {
                println!("    target {} {:.3?} w {:.2} -> at {:.3?}", db.skeleton.names[t.joint], t.target, t.weight, c.matcher.joint_world(db, t.joint));
            }
        }
        let mut frame_err = 0.0f32;
        for t in c.limb_targets().iter().filter(|t| t.weight > 0.99) {
            let e = c.matcher.joint_world(db, t.joint).distance(t.target);
            if trace && e > 0.04 && misses < 14 {
                misses += 1;
                println!("    {time:.2}s {} off its target by {:.1} cm (at {:.2?}, target {:.2?})", db.skeleton.names[t.joint], 100.0 * e, c.matcher.joint_world(db, t.joint), t.target);
            }
            frame_err = frame_err.max(e);
            sum_err += e;
            r.limb_frames += 1;
            r.limb_error_max = r.limb_error_max.max(e);
        }
        // planted feet on the ground: their slide (the ankle's or the toe's, whichever moved less:
        // a foot rolls onto its toe)
        let contacts = db.contacts(c.matcher.current_frame(db));
        for k in 0..2 {
            let p = [c.matcher.joint_world(db, db.roles.feet[k]), c.matcher.joint_world(db, balls[k])];
            let grounded = matches!(c.state(), CharacterState::Grounded) && contacts[k];
            if grounded {
                if let Some(q) = planted_prev[k] {
                    let d = |a: Vec3, b: Vec3| Vec3::new(a.x - b.x, 0.0, a.z - b.z).length();
                    slide += d(p[0], q[0]).min(d(p[1], q[1]));
                    r.planted_frames += 1;
                }
                planted_prev[k] = Some(p);
            } else {
                planted_prev[k] = None;
            }
        }
        // hands, feet and the body's core (their joints) inside the level
        let mut ends = db.roles.feet.to_vec();
        if let Some(h) = c.hands {
            ends.extend(h);
        }
        // and the body's core (hips, spine, head)
        ends.extend(body.iter().copied());
        for &j in &ends {
            let p = c.matcher.joint_world(db, j);
            if world.overlap_sphere(p, 0.001, u32::MAX) && p.y > 0.03 {
                r.inside += 1;
                if trace && r.inside % 15 == 1 {
                    println!("    {time:.2}s {} inside the level at {:.2?} ({})", db.skeleton.names[j], p, state);
                }
            }
        }
        // every joint's world position and the root's, kept for the last 1/15 s. The jump in
        // velocity over the data's own frame time (1/30 s windows, so a 30 fps clip's linear
        // interpolation between frames doesn't count, and every frame rate reads the same): a pop
        // is a jump the motion's own acceleration doesn't explain
        let root = c.matcher.character();
        let mut positions: Vec<Vec3> = (0..joints).map(|j| c.matcher.joint_world(db, j)).collect();
        positions.push(root.translation);
        history.push((time, step != dt, positions));
        let window = 1.0 / db.sample_rate;
        history.retain(|(t, _, _)| *t >= time - 2.0 * window - 1e-4);
        let mut accel = 0.0f32;
        let at = |t: f32| history.iter().find(|(u, _, _)| (u - t).abs() < 0.25 * dt.min(window));
        if let (Some(a0), Some(a1)) = (at(time - 2.0 * window), at(time - window)) {
            if !history.iter().any(|(_, h, _)| *h) {
                let a2 = history.last().unwrap();
                let jump = |j: usize| (a2.2[j] - 2.0 * a1.2[j] + a0.2[j]) / window;
                let root_jump = jump(joints);
                let (mut worst, mut worst_joint) = (0.0f32, 0usize);
                for &j in &body {
                    let v = jump(j).length();
                    if v > worst {
                        worst = v;
                        worst_joint = j;
                    }
                }
                accel = worst;
                accel_log.push((time, root_jump.length(), worst, worst_joint, state));
            }
        }
        if let Some(out) = csv.as_mut() {
            let ch = c.matcher.character().translation;
            let hips = c.matcher.joint_world(db, db.roles.hips);
            out.push_str(&format!("{time:.4},{state},{:.4},{:.4},{:.4},{frame_err:.4},{accel:.2},{:.3},{:.3},{:.3},{:.3}\n", ch.x, ch.y, ch.z, yaw_of(c.matcher.character().rotation), hips.x, hips.y, hips.z));
        }
    }
    r.limb_error_mean = if r.limb_frames > 0 { sum_err / r.limb_frames as f32 } else { 0.0 };
    r.slide_cm_s = if r.planted_frames > 0 { 100.0 * slide / (r.planted_frames as f32 * dt) } else { 0.0 };
    // pops: the largest velocity jump of the root and of the pose over the run, against the
    // run's 99th percentile
    let percentile = |f: &dyn Fn(&(f32, f32, f32, usize, &'static str)) -> f32| {
        let mut v: Vec<f32> = accel_log.iter().map(f).collect();
        v.sort_by(f32::total_cmp);
        v.get(v.len() * 99 / 100).copied().unwrap_or(0.0)
    };
    r.pop_typical = percentile(&|a| a.2);
    r.root_typical = percentile(&|a| a.1);
    for &(t, root, pose, joint, st) in &accel_log {
        if pose > r.pop_max {
            r.pop_max = pose;
            r.pop_at = t;
            r.pop_state = format!("{st} {}", db.skeleton.names[joint]);
        }
        if root > r.root_max {
            r.root_max = root;
            r.root_at = t;
            r.root_state = st.to_string();
        }
    }
    r.end = c.matcher.character().translation;
    r
}

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |key: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{key}=")).map(str::to_string));
    let path = args.iter().find(|a| !a.contains('=')).ok_or("usage: parkour_eval <pack.kmm> [only=<name part>] [csv=<dir>]")?;
    let pack = MotionPack::from_bytes(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)?;
    let tags: Vec<&str> = pack.meta("tags").unwrap_or("").split(',').collect();
    let bit = |name: &str| tags.iter().position(|t| *t == name).map_or(0, |b| 1u32 << b);
    let (idle, walk, run_bit) = (bit("idle"), bit("walk"), bit("run"));
    let gait = if walk != 0 && run_bit != 0 { (idle | walk, idle | run_bit) } else { (!ACTION_TAG, !ACTION_TAG) };
    let db = &pack.database;
    let (world, features) = course();
    println!("course: {} edges, {} bars, {} wall faces, {} platforms", features.edges.len(), features.bars.len(), features.faces.len(), features.platforms.len());
    let only = arg("only");
    // the data's own: each jump and landing clip as captured (its own root motion), the largest
    // jump in the hips' and head's velocity over 1/30 s windows: the floor a take-off or landing
    // can reach
    if arg("clips").is_some() {
        let mut pose = kansei_core::animation::Pose { local: Vec::new() };
        let mut model = Vec::new();
        let head = db.skeleton.find("head").unwrap_or(db.roles.hips);
        for a in &pack.actions {
            let c = &db.clips[a.clip];
            let world: Vec<[Vec3; 2]> = (0..c.frames)
                .map(|f| {
                    db.pose(c.start + f, c.start + f, 0.0, &mut pose);
                    pose.to_model(&db.skeleton, &mut model);
                    let (root, yaw) = db.root_at(a.clip, f as f32);
                    let r = kansei_core::animation::motion_matching::yaw_rotation(yaw);
                    [root + r * model[db.roles.hips].translation, root + r * model[head].translation]
                })
                .collect();
            let (mut worst, mut at) = (0.0f32, 0usize);
            for f in 2..world.len() {
                for k in 0..2 {
                    let v = (world[f][k] - 2.0 * world[f - 1][k] + world[f - 2][k]).length() * db.sample_rate;
                    if v > worst {
                        worst = v;
                        at = f;
                    }
                }
            }
            println!("data {:<7} {:<56} max {:.2} m/s at frame {} (rise {}, anchor {})", a.kind.name(), c.name, worst, at, a.rise, a.anchor);
        }
    }
    {
        let mut c = CharacterController::new(MotionMatcher::new(db, MotionMatchingSettings::default(), Vec3::ZERO, 0.0), pack.actions.clone());
        c.find_hands(db);
        let poses = c.hands.map_or(Vec::new(), |h| kansei_core::animation::motion_matching::traversal::parkour::hang_poses(db, &c.actions, h));
        for (a, b) in pack.actions.iter().zip(&c.actions).filter(|(a, b)| a.ledge != b.ledge) {
            println!("grip: {} ledge {:.3?} -> {:.3?}", db.clips[a.clip].name, a.ledge, b.ledge);
        }
        if let Some(h) = c.hands {
            println!("bar clips: {:?}", kansei_core::animation::motion_matching::traversal::parkour::bar_clips(db, &c.actions, h).map(|b| (db.clips[b.hang].name.clone(), b.swing.map(|(k, p, a, o)| (db.clips[k].name.clone(), p, a, o)))));
        }
        for p in &poses {
            println!("hang: {} frame {} root {:.2?} hands {:.2?} feet {:.2?}", db.clips[p.climb.clip].name, p.frame, p.root, p.hands, p.feet);
        }
    }
    let csv_dir = arg("csv");
    let trace = arg("trace").is_some();
    // (`trace=120` traces the 120 fps run; `trace=1` the 60 fps one)
    let trace_label = arg("trace").filter(|t| t != "1").unwrap_or_else(|| "60".to_string());
    let rates: [(f32, bool, &str); 4] = [(1.0 / 30.0, false, "30"), (1.0 / 60.0, false, "60"), (1.0 / 120.0, false, "120"), (1.0 / 60.0, true, "60+hitch")];
    println!("{:<52} {:>8} {:>4} {:>8} {:>8} {:>10} {:>6} {:>11} {:>11}  states", "scenario", "fps", "ok", "limb max", "limb avg", "slide", "inside", "root pop", "pose pop");
    let mut failures = 0;
    for s in scenarios().iter().filter(|s| only.as_ref().is_none_or(|o| s.name.contains(o.as_str()))) {
        for (dt, hitch, label) in rates {
            let mut csv = csv_dir.as_ref().map(|_| String::from("time,state,x,y,z,limb_error,accel,yaw,hips_x,hips_y,hips_z\n"));
            let r = run(db, &pack.actions, &world, &features, s, dt, hitch, gait, csv.as_mut(), trace && label == trace_label.as_str());
            // the expected states in order (others may come between)
            let mut want = s.expect.iter().peekable();
            for st in &r.states {
                if want.peek().is_some_and(|w| *w == st) {
                    want.next();
                }
            }
            let ok = want.peek().is_none();
            failures += !ok as u32;
            println!(
                "{:<52} {:>8} {:>4} {:>6.1}cm {:>6.1}cm {:>6.1}cm/s {:>6} {:>5.2}/{:<5.2} {:>5.2}/{:<5.2}  {} | root pop {:.2}s {} | pose pop {:.2}s {} | end ({:.2},{:.2},{:.2})",
                s.name, label, if ok { "yes" } else { "NO" }, 100.0 * r.limb_error_max, 100.0 * r.limb_error_mean, r.slide_cm_s, r.inside, r.root_max, r.root_typical, r.pop_max, r.pop_typical, r.states.join(">"), r.root_at, r.root_state, r.pop_at, r.pop_state, r.end.x, r.end.y, r.end.z
            );
            if let (Some(dir), Some(csv)) = (&csv_dir, csv) {
                let name: String = s.name.chars().map(|c| if c.is_alphanumeric() { c } else { '_' }).collect();
                std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                std::fs::write(format!("{dir}/{name}_{label}.csv"), csv).map_err(|e| e.to_string())?;
            }
        }
    }
    println!("{failures} runs missed a state they should have gone through");
    Ok(())
}
