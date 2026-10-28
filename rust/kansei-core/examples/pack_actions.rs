//! Lists a motion pack's meta, skeleton and action clips (what `traversal::ActionClip::analyze`
//! found in each), for checking a pack before the parkour moves use it.
//!
//! ```sh
//! cargo run -p kansei-core --release --example pack_actions -- <pack.kmm>
//! ```
//!
//! Packs are private: link them in (see the motion-matching demo's README), never commit them.

use kansei_core::animation::motion_matching::pack::MotionPack;

fn main() -> Result<(), String> {
    let path = std::env::args().nth(1).ok_or("usage: pack_actions <pack.kmm>")?;
    let pack = MotionPack::from_bytes(&std::fs::read(&path).map_err(|e| format!("{path}: {e}"))?)?;
    let db = &pack.database;
    for (k, v) in &pack.meta {
        println!("meta {k}: {}", v.chars().take(160).collect::<String>());
    }
    println!("{} clips, {} frames at {} fps, {} joints", db.clips.len(), db.frame_count(), db.sample_rate, db.joint_count());
    println!("joints: {}", db.skeleton.names.join(" "));
    for a in &pack.actions {
        let c = &db.clips[a.clip];
        println!(
            "{:<7} {:<56} frames {:>3}  height {:.2}  rise {:>5.1} anchor {:>5.1} on_top {:>5.1} off_top {:>5.1} down {:>5.1} exit {:>5.1} span {:.2} ledge ({:.2},{:.2},{:.2})",
            a.kind.name(), c.name, c.frames, a.height, a.rise, a.anchor, a.on_top, a.off_top, a.down, a.exit, a.span, a.ledge.x, a.ledge.y, a.ledge.z
        );
    }
    // `track=<name part>`: per frame, the root and the hands and feet relative to the clip's ledge
    if let Some(part) = std::env::args().find_map(|a| a.strip_prefix("track=").map(str::to_string)) {
        let joint = |n: &str| db.skeleton.find(n).ok_or(format!("no joint {n}"));
        let ends = [joint("hand_l")?, joint("hand_r")?, joint("foot_l")?, joint("foot_r")?];
        let mut pose = kansei_core::animation::Pose { local: Vec::new() };
        let mut model = Vec::new();
        for a in pack.actions.iter().filter(|a| db.clips[a.clip].name.contains(part.as_str())) {
            let c = &db.clips[a.clip];
            println!("{} (ledge {:?})", c.name, a.ledge);
            for f in 0..c.frames {
                db.pose(c.start + f, c.start + f, 0.0, &mut pose);
                pose.to_model(&db.skeleton, &mut model);
                let (root, yaw) = db.root_at(a.clip, f as f32);
                let r = kansei_core::animation::motion_matching::yaw_rotation(yaw);
                let rel: Vec<String> = ends.iter().map(|&j| {
                    let p = root + r * model[j].translation - a.ledge;
                    format!("({:5.2},{:5.2},{:5.2})", p.x, p.y, p.z)
                }).collect();
                let contacts = db.contacts(c.start + f);
                println!("  {f:>3} root ({:5.2},{:5.2},{:5.2}) hands {} {} feet {} {} {:?}", root.x - a.ledge.x, root.y - a.ledge.y, root.z - a.ledge.z, rel[0], rel[1], rel[2], rel[3], contacts);
            }
        }
    }
    // `clip=<name part>`: per frame of any clip, the hips and feet in the clip root's frame
    if let Some(part) = std::env::args().find_map(|a| a.strip_prefix("clip=").map(str::to_string)) {
        let joint = |n: &str| db.skeleton.find(n).ok_or(format!("no joint {n}"));
        let ends = [joint("pelvis")?, joint("foot_l")?, joint("foot_r")?, joint("hand_l")?, joint("hand_r")?];
        let mut pose = kansei_core::animation::Pose { local: Vec::new() };
        let mut model = Vec::new();
        for (k, c) in db.clips.iter().enumerate().filter(|(_, c)| c.name.contains(part.as_str())) {
            println!("{} ({} frames)", c.name, c.frames);
            let (r0, _) = db.root_at(k, 0.0);
            for f in 0..c.frames {
                db.pose(c.start + f, c.start + f, 0.0, &mut pose);
                pose.to_model(&db.skeleton, &mut model);
                let (root, yaw) = db.root_at(k, f as f32);
                let cols: Vec<String> = ends.iter().map(|&j| { let p = model[j].translation; format!("({:5.2},{:5.2},{:5.2})", p.x, p.y, p.z) }).collect();
                println!("  {f:>3} root ({:6.2},{:6.2}) yaw {:6.1}  pelvis {} feet {} {} hands {} {} {:?}", root.x - r0.x, root.z - r0.z, yaw.to_degrees(), cols[0], cols[1], cols[2], cols[3], cols[4], db.contacts(c.start + f));
            }
        }
    }
    Ok(())
}
