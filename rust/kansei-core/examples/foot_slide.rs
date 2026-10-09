//! Planted-foot slide of a motion-matching pack over a scripted course (straight walks and runs,
//! circles both ways, starts and stops, 180° turns): `animation::motion_matching::foot_slide`.
//!
//! ```sh
//! cargo run -p kansei-core --release --example foot_slide -- <pack.kmm> [hero=<character.kmm>] [walk=2] [run=5] [only=<name part>]
//! ```
//!
//! Packs are private: link them in (see the motion-matching demo's README), never commit them.
//! `hero=` measures on a character pack's body (the pose retargeted onto it), as the demo shows it.
//! Feet count as planted by the contacts `ContactThresholds::default()` finds. To try other
//! tunings: `contact_speed=` and `contact_height=` find the contacts foot locking pins on again
//! with those `ContactThresholds`; `lock=0` turns foot locking off; `unlock_radius=`, `lock_halflife=`,
//! `adjustment_halflife=`, `max_adjustment_ratio=`, `clamp_distance=` and `clamp_angle=` (degrees) set those
//! `MotionMatchingSettings`.

use glam::Vec3;
use kansei_core::animation::motion_matching::foot_slide::{course, measure, FootSlideReport};
use kansei_core::animation::motion_matching::pack::{CharacterPack, MotionPack};
use kansei_core::animation::motion_matching::{ContactThresholds, MotionMatcher, MotionMatchingSettings, ACTION_TAG};
use kansei_core::animation::retarget::Retarget;

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let arg = |key: &str| args.iter().find_map(|a| a.strip_prefix(&format!("{key}=")).map(str::to_string));
    let path = args.iter().find(|a| !a.contains('=')).ok_or("usage: foot_slide <pack.kmm> [hero=<character.kmm>] [walk=2] [run=5] [only=<name part>]")?;
    let read = |p: &str| std::fs::read(p).map_err(|e| format!("{p}: {e}"));
    let mut pack = MotionPack::from_bytes(&read(path)?)?;
    let hero = arg("hero").map(|p| read(&p).and_then(|b| CharacterPack::from_bytes(&b))).transpose()?;
    let (walk, run) = (arg("walk").map_or(Ok(2.0), |v| v.parse::<f32>()), arg("run").map_or(Ok(5.0), |v| v.parse::<f32>()));
    let (walk, run) = (walk.map_err(|e| e.to_string())?, run.map_err(|e| e.to_string())?);
    let only = arg("only");

    // the demo's gait filter: idle + walk or idle + run by the pack's tags
    let tags: Vec<&str> = pack.meta("tags").unwrap_or("").split(',').collect();
    let bit = |name: &str| tags.iter().position(|t| *t == name).map_or(0, |b| 1u32 << b);
    let (idle, walk_bit, run_bit) = (bit("idle"), bit("walk"), bit("run"));
    let (walk_tags, run_tags) = if walk_bit != 0 && run_bit != 0 { (idle | walk_bit, idle | run_bit) } else { (!ACTION_TAG, !ACTION_TAG) };
    let number = |key: &str| arg(key).map(|v| v.parse::<f32>().map_err(|e| format!("{key}: {e}"))).transpose();
    let mut reference = pack.database.clone();
    reference.detect_contacts(&ContactThresholds::default());
    if number("contact_speed")?.is_some() || number("contact_height")?.is_some() {
        let d = ContactThresholds::default();
        let thresholds = ContactThresholds { speed: number("contact_speed")?.unwrap_or(d.speed), height: number("contact_height")?.unwrap_or(d.height), ..d };
        pack.database.detect_contacts(&thresholds);
    }
    let db = &pack.database;
    let mut settings = MotionMatchingSettings::default();
    settings.foot_lock = arg("lock").as_deref() != Some("0");
    let s = &mut settings;
    for (key, field) in [("unlock_radius", &mut s.foot_unlock_radius), ("lock_halflife", &mut s.foot_lock_halflife), ("adjustment_halflife", &mut s.adjustment_halflife), ("max_adjustment_ratio", &mut s.max_adjustment_ratio), ("clamp_distance", &mut s.clamp_distance)] {
        if let Some(v) = number(key)? {
            *field = v;
        }
    }
    if let Some(v) = number("clamp_angle")? {
        settings.clamp_angle = v.to_radians();
    }

    let mut reports = Vec::new();
    println!("{:<26} {:>7} {:>9} {:>8} {:>9} {:>9}", "scenario", "cm/s", "cm/plant", "planted", "mesh off", "max off");
    for scenario in course(walk, run).iter().filter(|s| only.as_ref().is_none_or(|o| s.name.contains(o.as_str()))) {
        let mut matcher = MotionMatcher::new(db, settings.clone(), Vec3::ZERO, 0.0);
        matcher.settings.filter.tags = if scenario.run { run_tags } else { walk_tags };
        if let Some(h) = &hero {
            matcher.set_display(db, Some((h.skeleton.clone(), Retarget::new(&db.skeleton, &h.skeleton, &Retarget::UNREAL_KEEP))));
        }
        let r = measure(db, &reference, matcher, scenario);
        println!("{:<26} {:>7.1} {:>9.1} {:>7.0}% {:>8.1}° {:>8.1}°", scenario.name, r.cm_per_second, r.cm_per_plant, 100.0 * r.planted, r.yaw_gap, r.yaw_gap_max);
        reports.push(r);
    }
    // one scenario can swing a lot with a tuning (see `measure`): judge by the course
    let mean = |f: fn(&FootSlideReport) -> f32| reports.iter().map(f).sum::<f32>() / reports.len().max(1) as f32;
    println!("{:<26} {:>7.1} {:>9.1}", "mean", mean(|r| r.cm_per_second), mean(|r| r.cm_per_plant));
    Ok(())
}
