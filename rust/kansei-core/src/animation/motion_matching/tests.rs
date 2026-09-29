use glam::{Mat4, Quat, Vec3};

use super::pack::{MotionPack, PackMesh};
use super::*;
use crate::animation::{Clip, Pose, Skeleton, SkinnedMesh, Transform};
use crate::geometries::Vertex;

const RATE: f32 = 30.0;

/// A root on the ground, hips 1 m up, and two legs of a thigh, a calf and a foot (ankles 0.1 m
/// above the ground at rest).
pub(crate) fn biped() -> Skeleton {
    let t = |x: f32, y: f32| Transform::from_translation_rotation(Vec3::new(x, y, 0.0), Quat::IDENTITY);
    Skeleton::new(
        ["root", "hips", "thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r"].map(String::from).to_vec(),
        vec![None, Some(0), Some(1), Some(2), Some(3), Some(1), Some(5), Some(6)],
        vec![t(0.0, 0.0), t(0.0, 1.0), t(0.1, 0.0), t(0.0, -0.45), t(0.0, -0.45), t(-0.1, 0.0), t(0.0, -0.45), t(0.0, -0.45)],
    )
}

/// A clip of `frames` frames: the root travelling forward at `speed(t)` m/s while turning at
/// `turn` rad/s, the legs swinging in opposition once a second when moving.
fn locomotion(name: &str, frames: usize, speed: impl Fn(f32) -> f32, turn: f32) -> Clip {
    let skeleton = biped();
    let (mut position, mut yaw) = (Vec3::ZERO, 0.0f32);
    let dt = 1.0 / RATE;
    let poses: Vec<Pose> = (0..frames)
        .map(|f| {
            let t = f as f32 * dt;
            if f > 0 {
                let v = speed(t - 0.5 * dt);
                position += Quat::from_rotation_y(yaw + 0.5 * turn * dt) * Vec3::new(0.0, 0.0, v * dt);
                yaw += turn * dt;
            }
            let swing = (std::f32::consts::TAU * t).sin() * 0.4 * (speed(t) / 1.5).min(1.0);
            let mut pose = Pose::rest(&skeleton);
            pose.local[0] = Transform::from_translation_rotation(position, Quat::from_rotation_y(yaw));
            pose.local[2].rotation = Quat::from_rotation_x(-swing);
            pose.local[5].rotation = Quat::from_rotation_x(swing);
            pose.local[3].rotation = Quat::from_rotation_x(0.3 * swing.abs());
            pose.local[6].rotation = Quat::from_rotation_x(0.3 * swing.abs());
            pose
        })
        .collect();
    Clip::from_poses(name, RATE, &poses)
}

/// Idle and walk loops, a start, a stop and a left turn while walking.
pub(crate) fn database() -> Database {
    let skeleton = biped();
    let roles = JointRoles::find(&skeleton, "root", "hips", "foot_l", "foot_r").unwrap();
    let mut builder = DatabaseBuilder::new(skeleton, roles, RATE);
    builder.add_clip(&locomotion("idle", 61, |_| 0.0, 0.0), true, 1).unwrap();
    builder.add_clip(&locomotion("walk", 61, |_| 1.5, 0.0), true, 2).unwrap();
    builder.add_clip(&locomotion("start", 46, |t| 1.5 * (t / 1.0).min(1.0), 0.0), false, 2).unwrap();
    builder.add_clip(&locomotion("stop", 46, |t| 1.5 * (1.0 - t / 1.0).max(0.0), 0.0), false, 2).unwrap();
    builder.add_clip(&locomotion("turn_left", 61, |_| 1.5, std::f32::consts::FRAC_PI_2 / 1.5), false, 2).unwrap();
    builder.build()
}

fn clip(db: &Database, name: &str) -> usize {
    db.clips.iter().position(|c| c.name == name).unwrap()
}

#[test]
fn the_database_holds_the_clips_roots_and_trajectories() {
    let db = database();
    assert_eq!(db.clips.iter().map(|c| c.frames).sum::<usize>(), db.frame_count());
    let walk = &db.clips[clip(&db, "walk")];
    assert_eq!(db.clip_of(walk.start + 10), clip(&db, "walk"));
    // the root is taken out of the pose: its joint is the identity relative to the character
    assert!(db.transform(walk.start + 20, 0).translation.length() < 1e-5);
    assert!(db.root(walk.start + 30).translation.abs_diff_eq(Vec3::new(0.0, 0.0, 1.5), 1e-4));
    // trajectory features: 1.5 m ahead in 1 s walking, still when idle, rotated when turning
    let raw = db.denormalize(db.features(walk.start + 5));
    assert!((raw[20] - 1.5).abs() < 1e-3 && raw[19].abs() < 1e-3, "{:?}", &raw[15..21]);
    assert!((raw[26] - 1.0).abs() < 1e-3, "{:?}", &raw[21..27]);
    let idle = db.denormalize(db.features(db.clips[clip(&db, "idle")].start + 5));
    assert!(idle[15..21].iter().all(|x| x.abs() < 1e-4), "{:?}", &idle[15..21]);
    let turn = db.denormalize(db.features(db.clips[clip(&db, "turn_left")].start));
    let heading = turn[25].atan2(turn[26]);
    assert!((heading - std::f32::consts::FRAC_PI_2 / 1.5).abs() < 0.02, "{heading}");
    // a stop's trajectory ends where it stops; a clip's end goes on at its last velocity
    let stop = &db.clips[clip(&db, "stop")];
    let raw = db.denormalize(db.features(stop.start + stop.frames - 1));
    assert!(raw[20].abs() < 1e-3, "{:?}", &raw[15..21]);
    let start = &db.clips[clip(&db, "start")];
    let raw = db.denormalize(db.features(start.start + start.frames - 1));
    assert!((raw[20] - 1.5).abs() < 0.02, "{:?}", &raw[15..21]);
    // idle feet are planted
    assert_eq!(db.contacts(db.clips[clip(&db, "idle")].start + 30), [true, true]);
    // normalized features average to zero
    for i in 0..FEATURES {
        let mean = (0..db.frame_count()).map(|f| db.features(f)[i]).sum::<f32>() / db.frame_count() as f32;
        assert!(mean.abs() < 1e-3, "feature {i}: {mean}");
    }
}

#[test]
fn loops_are_seamless_in_velocity_and_root_motion() {
    let db = database();
    let walk = clip(&db, "walk");
    let info = &db.clips[walk];
    // a loop's first and last frames have the same features
    let (first, last) = (db.features(info.start), db.features(info.start + info.frames - 1));
    assert!(first.iter().zip(last).all(|(a, b)| (a - b).abs() < 1e-3), "{first:?}\n{last:?}");
    // playing across the seam moves the root on as if the clip went on
    let (moved, turned) = db.root_motion(walk, 50.0, 70.0);
    assert!(moved.abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-3), "{moved}");
    assert!(turned.abs() < 1e-5);
    let turn = clip(&db, "turn_left");
    // a 1.5 m arc turning 60 degrees: its chord, bending left (+x from facing +z)
    let (moved, turned) = db.root_motion(turn, 0.0, 30.0);
    let angle = std::f32::consts::FRAC_PI_2 / 1.5;
    assert!((turned - angle).abs() < 1e-3);
    let chord = 2.0 * (1.5 / angle) * (0.5 * angle).sin();
    assert!((moved.length() - chord).abs() < 0.01 && moved.x > 0.0, "{moved}");
}

fn noise(i: u32) -> f32 {
    let x = (i.wrapping_mul(747796405).wrapping_add(2891336453)) ^ (i >> 7).wrapping_mul(277803737);
    (x % 10007) as f32 / 10007.0 - 0.5
}

#[test]
fn the_accelerated_search_agrees_with_brute_force() {
    let db = database();
    for k in 0..200u32 {
        let frame = (noise(k) + 0.5) as f32 * (db.frame_count() - 1) as f32;
        let mut query = [0.0; STRIDE];
        query[..FEATURES].copy_from_slice(&db.features(frame as usize)[..FEATURES]);
        for (i, q) in query[..FEATURES].iter_mut().enumerate() {
            *q += noise(k * 31 + i as u32) * 0.8;
        }
        for filter in [SearchFilter::default(), SearchFilter { current: Some(frame as usize), ignore_near: 5, ..Default::default() }, SearchFilter { tags: 1, ..Default::default() }] {
            let fast = db.search(&query, &filter, f32::MAX).unwrap();
            let slow = db.search_brute_force(&query, &filter, f32::MAX).unwrap();
            assert_eq!(fast.frame, slow.frame, "query {k}, {filter:?}");
            assert!((fast.cost - slow.cost).abs() < 1e-4);
            if filter.tags == 1 {
                assert_eq!(db.clips[db.clip_of(fast.frame)].name, "idle");
            }
            if let Some(current) = filter.current {
                assert!(fast.frame.abs_diff(current) > 5 || db.clip_of(fast.frame) != db.clip_of(current));
            }
        }
    }
    // an exact query finds its own frame, and nothing beats a cost it can't improve on
    let walk = &db.clips[clip(&db, "walk")];
    let mut exact = [0.0; STRIDE];
    exact.copy_from_slice(db.features(walk.start + 12));
    let found = db.search(&exact, &SearchFilter::default(), f32::MAX).unwrap();
    assert!(found.cost < 1e-6 && db.clip_of(found.frame) == clip(&db, "walk"), "{found:?}");
    assert!(db.search(&exact, &SearchFilter::default(), 0.0).is_none());
    // frames at the end of clips that don't loop are never found
    let stop = &db.clips[clip(&db, "stop")];
    exact.copy_from_slice(db.features(stop.start + stop.frames - 1));
    let found = db.search(&exact, &SearchFilter::default(), f32::MAX).unwrap();
    assert!(found.frame < stop.start + stop.frames - 10 || !stop.contains(found.frame));
}

fn triangle_mesh() -> SkinnedMesh {
    let v = |x: f32, y: f32| Vertex { position: [x, y, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [x, y] };
    SkinnedMesh {
        name: "tri".into(),
        vertices: vec![v(0.0, 0.0), v(1.0, 0.0), v(0.0, 1.0)],
        indices: vec![0, 1, 2],
        joints: vec![[1, 0, 0, 0], [4, 1, 0, 0], [7, 4, 1, 0]],
        weights: vec![[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.25, 0.25, 0.5, 0.0]],
        skin_joints: (0..8).collect(),
        inverse_bind: (0..8).map(|i| Mat4::from_translation(Vec3::splat(i as f32))).collect(),
        material: Some(0),
    }
}

#[test]
fn a_pack_round_trips() {
    use super::traversal::{ActionClip, ActionKind};
    let action = ActionClip { clip: 2, kind: ActionKind::Vault, height: 1.1, ledge: Vec3::new(0.1, 1.1, 0.4), forward: Vec3::Z, rise: 10.0, anchor: 14.0, on_top: 15.0, off_top: 18.0, down: 22.0, exit: 22.0, span: 0.4, last_entry: 4.0 };
    let pack = MotionPack { database: database(), meshes: vec![PackMesh { mesh: triangle_mesh(), color: [0.5, 0.4, 0.3, 1.0] }], actions: vec![action], meta: vec![("source".into(), "synthetic".into())] };
    let bytes = pack.to_bytes();
    let back = MotionPack::from_bytes(&bytes).unwrap();
    assert_eq!(back.database, pack.database);
    assert_eq!(back.actions, pack.actions);
    assert_eq!(back.meta("source"), Some("synthetic"));
    let (a, b) = (&back.meshes[0].mesh, &pack.meshes[0].mesh);
    assert_eq!((a.indices.clone(), a.joints.clone(), a.skin_joints.clone(), a.inverse_bind.clone(), a.material), (b.indices.clone(), b.joints.clone(), b.skin_joints.clone(), b.inverse_bind.clone(), b.material));
    assert_eq!(a.skin_words(), b.skin_words());
    assert_eq!(back.meshes[0].color, [0.5, 0.4, 0.3, 1.0]);
    assert!(a.vertices.iter().zip(&b.vertices).all(|(x, y)| x.position == y.position && x.normal == y.normal && x.uv == y.uv));
    // a character pack: skeleton, mesh and images
    use super::pack::{CharacterPack, PackImage};
    let character = CharacterPack {
        skeleton: pack.database.skeleton.clone(),
        meshes: pack.meshes.clone(),
        images: vec![PackImage { name: "base_color".into(), mime: "image/webp".into(), bytes: vec![1, 2, 3, 4, 5] }],
        meta: vec![("source".into(), "synthetic".into())],
    };
    let back_character = CharacterPack::from_bytes(&character.to_bytes()).unwrap();
    assert_eq!(back_character.skeleton, character.skeleton);
    assert_eq!(back_character.image("base_color").unwrap().bytes, vec![1, 2, 3, 4, 5]);
    assert_eq!(back_character.meshes[0].mesh.skin_words(), character.meshes[0].mesh.skin_words());
    assert!(CharacterPack::from_bytes(&character.to_bytes()[..40]).is_err());
    // not a pack, truncated, or corrupt: errors, not panics
    assert!(MotionPack::from_bytes(b"nope").is_err());
    assert!(MotionPack::from_bytes(&bytes[..bytes.len() / 2]).is_err());
    let mut other_version = bytes.clone();
    other_version[4] = 99;
    assert!(MotionPack::from_bytes(&other_version).is_err());
}

fn run(matcher: &mut MotionMatcher, db: &Database, velocity: Vec3, seconds: f32) {
    let dt = 1.0 / 60.0;
    for _ in 0..(seconds / dt) as usize {
        matcher.update(db, &MotionInput { velocity, facing: None }, dt);
        let gap = matcher.character().translation - matcher.simulation().position;
        assert!(Vec3::new(gap.x, 0.0, gap.z).length() <= matcher.settings.clamp_distance + 1e-4, "{gap}");
    }
}

fn playing(matcher: &MotionMatcher, db: &Database) -> String {
    db.clips[matcher.playing().0].name.clone()
}

#[test]
fn the_character_idles_walks_when_asked_and_stops_when_released() {
    let db = database();
    let mut matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), Vec3::ZERO, 0.0);
    run(&mut matcher, &db, Vec3::ZERO, 1.0);
    assert_eq!(playing(&matcher, &db), "idle");
    assert!(matcher.character().translation.length() < 1e-3);
    assert_eq!(matcher.feet_locked(), [true, true]);

    run(&mut matcher, &db, Vec3::new(0.0, 0.0, 1.5), 3.0);
    let name = playing(&matcher, &db);
    assert!(name == "walk" || name == "start", "{name}");
    let z = matcher.character().translation.z;
    assert!(z > 3.0 && z < 4.5, "{z}");

    run(&mut matcher, &db, Vec3::ZERO, 3.0);
    let name = playing(&matcher, &db);
    assert!(name == "idle" || name == "stop", "{name}");
    assert!(matcher.simulation().velocity.length() < 0.01);
    let before = matcher.character().translation;
    run(&mut matcher, &db, Vec3::ZERO, 1.0);
    assert!(matcher.character().translation.distance(before) < 0.02, "stands still");
}

#[test]
fn the_character_turns_to_face_where_it_goes() {
    let db = database();
    let mut matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), Vec3::ZERO, 0.0);
    run(&mut matcher, &db, Vec3::new(0.0, 0.0, 1.5), 2.0);
    run(&mut matcher, &db, Vec3::new(1.5, 0.0, 0.0), 3.0);
    // the database turns 60 degrees at most in one go, and the character is only turned toward
    // the simulation while its animation turns (adjust_by_velocity): it turns at least that far
    let yaw = yaw_of(matcher.character().rotation);
    assert!(yaw > 0.9 && yaw < std::f32::consts::FRAC_PI_2 + 0.1, "{yaw}");
    assert!((yaw_of(matcher.simulation().rotation) - std::f32::consts::FRAC_PI_2).abs() < 0.01);
    assert!(matcher.character().translation.x > 2.0, "{}", matcher.character().translation);
}

#[test]
fn searches_run_on_their_interval_and_switch_with_inertialization() {
    let db = database();
    let mut matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), Vec3::ZERO, 0.0);
    let dt = 1.0 / 60.0;
    let mut searches = 0;
    for _ in 0..120 {
        matcher.update(&db, &MotionInput { velocity: Vec3::ZERO, facing: None }, dt);
        searches += matcher.last_search().searched as usize;
    }
    // every 0.1 s over 2 s, plus the first
    assert!((19..=22).contains(&searches), "{searches}");
    // asking to walk searches at once and switches
    matcher.update(&db, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, dt);
    assert!(matcher.last_search().searched);
    let pose_before = matcher.pose().clone();
    matcher.update(&db, &MotionInput { velocity: Vec3::new(0.0, 0.0, 1.5), facing: None }, dt);
    // no pop: consecutive output poses stay close even across the switch
    for (a, b) in pose_before.local.iter().zip(&matcher.pose().local) {
        assert!(a.rotation.dot(b.rotation).abs() > 0.99, "{a:?} {b:?}");
    }
}

#[test]
fn a_display_skeleton_shows_the_pose_with_its_own_proportions() {
    use crate::animation::retarget::Retarget;
    let db = database();
    // the biped with 20% shorter legs and hips 20% lower
    let mut short = db.skeleton.clone();
    for (j, name) in short.names.clone().iter().enumerate() {
        if name.starts_with("calf") || name.starts_with("foot") || name == "hips" {
            short.rest[j].translation *= 0.8;
        }
    }
    let mut matcher = MotionMatcher::new(&db, MotionMatchingSettings::default(), Vec3::ZERO, 0.0);
    matcher.set_display(&db, Some((short.clone(), Retarget::new(&db.skeleton, &short, &Retarget::UNREAL_KEEP))));
    run(&mut matcher, &db, Vec3::ZERO, 1.0);
    let model = matcher.model();
    let hips = short.find("hips").unwrap();
    assert!((model[hips].translation.y - 0.8).abs() < 0.02, "hips at {}", model[hips].translation.y);
    // standing, both feet of the short legs are planted on the ground they reach
    assert_eq!(matcher.feet_locked(), [true, true]);
    let foot = short.find("foot_l").unwrap();
    assert!((model[foot].translation.y - 0.08).abs() < 0.02, "foot at {}", model[foot].translation.y);
    run(&mut matcher, &db, Vec3::new(0.0, 0.0, 1.5), 2.0);
    assert_eq!(matcher.pose().len(), short.len());
    assert!(matcher.character().translation.z > 1.5);
    // and back to the database's own skeleton
    matcher.set_display(&db, None);
    assert!((matcher.model()[1].translation.y - 1.0).abs() < 0.1);
}
