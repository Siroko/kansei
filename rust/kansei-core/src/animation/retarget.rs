//! Retargeting a pose between skeletons that share joint names and joint axes but not
//! proportions (a stylised character rigged to an animation set's skeleton).
//!
//! Rotations carry over as they are (the axes agree). Translations, which hold the bone lengths,
//! are chosen per joint:
//! - `Animation`: the animation's own (the root, attachment and IK helper joints, whose
//!   positions mean something in the animation's space);
//! - `Skeleton`: the target's rest translation (fixed bone lengths);
//! - `OrientAndScale` (the default): the animation's translation, turned from the source rest
//!   direction onto the target's and scaled by the ratio of their lengths. A joint whose
//!   translation never moves gets the target's bone exactly; one that moves (the hips going up
//!   and down) moves in proportion to the target's size.

use glam::Quat;

use super::{Pose, Skeleton, Transform};

/// How one joint's translation is retargeted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum TranslationMode {
    Animation,
    Skeleton,
    OrientAndScale,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct Joint {
    source: usize,
    mode: TranslationMode,
    /// OrientAndScale: from the source rest direction onto the target's, and the length ratio.
    turn: Quat,
    scale: f32,
}

/// Maps poses of one skeleton onto another with the same joint names.
#[derive(Debug, Clone, PartialEq)]
pub struct Retarget {
    /// Per target joint, its source joint (`None`: not in the source; keeps the target's rest).
    joints: Vec<Option<Joint>>,
    target_rest: Vec<Transform>,
}

/// Whether `name` matches `pattern` (a trailing `*` matches any rest).
fn matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => pattern == name,
    }
}

impl Retarget {
    /// Retarget from `source` onto `target`, joints paired by name. Joints matching any of
    /// `keep_animation` (exact names, or prefixes ending in `*`) keep the animation's
    /// translation; every other joint orients and scales it.
    pub fn new(source: &Skeleton, target: &Skeleton, keep_animation: &[&str]) -> Self {
        let joints = target
            .names
            .iter()
            .enumerate()
            .map(|(j, name)| {
                let s = source.find(name)?;
                let (from, to) = (source.rest[s].translation, target.rest[j].translation);
                let mode = if keep_animation.iter().any(|p| matches(p, name)) {
                    TranslationMode::Animation
                } else if from.length() < 1e-5 || to.length() < 1e-5 {
                    TranslationMode::Skeleton
                } else {
                    TranslationMode::OrientAndScale
                };
                let (turn, scale) = if mode == TranslationMode::OrientAndScale {
                    (Quat::from_rotation_arc(from.normalize(), to.normalize()), to.length() / from.length())
                } else {
                    (Quat::IDENTITY, 1.0)
                };
                Some(Joint { source: s, mode, turn, scale })
            })
            .collect();
        Self { joints, target_rest: target.rest.clone() }
    }

    /// The joints Unreal-style skeletons keep in animation space: the root, `attach`, the IK
    /// targets, prop and virtual bones.
    pub const UNREAL_KEEP: [&'static str; 6] = ["root", "attach", "ik_*", "props_root", "prop_*", "VB *"];

    /// The translation mode chosen for each target joint (`None`: not in the source).
    pub fn modes(&self) -> Vec<Option<TranslationMode>> {
        self.joints.iter().map(|j| j.map(|j| j.mode)).collect()
    }

    /// `source` (a pose of the source skeleton) as a pose of the target, into `out`.
    pub fn apply(&self, source: &Pose, out: &mut Pose) {
        out.local.clear();
        out.local.extend(self.joints.iter().zip(&self.target_rest).map(|(joint, rest)| match joint {
            None => *rest,
            Some(j) => {
                let s = &source.local[j.source];
                let translation = match j.mode {
                    TranslationMode::Animation => s.translation,
                    TranslationMode::Skeleton => rest.translation,
                    TranslationMode::OrientAndScale => j.turn * s.translation * j.scale,
                };
                Transform { translation, rotation: s.rotation, scale: rest.scale }
            }
        }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec3;

    fn t(x: f32, y: f32, z: f32) -> Transform {
        Transform::from_translation_rotation(Vec3::new(x, y, z), Quat::IDENTITY)
    }

    /// root > pelvis (1 m up) > thigh (0.1 m out) > calf (0.45 m down); root > ik_foot.
    fn source() -> Skeleton {
        Skeleton::new(
            ["root", "pelvis", "thigh_l", "calf_l", "ik_foot_l"].map(String::from).to_vec(),
            vec![None, Some(0), Some(1), Some(2), Some(0)],
            vec![t(0.0, 0.0, 0.0), t(0.0, 1.0, 0.0), t(0.1, 0.0, 0.0), t(0.0, -0.45, 0.0), t(0.1, 0.1, 0.0)],
        )
    }

    /// Shorter legs (pelvis at 0.8 m, calf 0.35 m, slightly forward), no IK joint, an extra head.
    fn target() -> Skeleton {
        Skeleton::new(
            ["root", "pelvis", "thigh_l", "calf_l", "head"].map(String::from).to_vec(),
            vec![None, Some(0), Some(1), Some(2), Some(1)],
            vec![t(0.0, 0.0, 0.0), t(0.0, 0.8, 0.0), t(0.12, 0.0, 0.0), t(0.0, -0.35, 0.02), t(0.0, 0.7, 0.0)],
        )
    }

    #[test]
    fn modes_follow_the_names_and_the_rest_translations() {
        let r = Retarget::new(&source(), &target(), &Retarget::UNREAL_KEEP);
        use TranslationMode::*;
        assert_eq!(r.modes(), vec![Some(Animation), Some(OrientAndScale), Some(OrientAndScale), Some(OrientAndScale), None]);
    }

    #[test]
    fn rotations_carry_over_and_bones_take_the_target_lengths() {
        let (s, tg) = (source(), target());
        let r = Retarget::new(&s, &tg, &Retarget::UNREAL_KEEP);
        let mut pose = Pose::rest(&s);
        // the root walked 2 m, the pelvis bobbed 5 cm down, the knee bent
        pose.local[0].translation = Vec3::new(0.0, 0.0, 2.0);
        pose.local[1].translation = Vec3::new(0.0, 0.95, 0.0);
        pose.local[3].rotation = Quat::from_rotation_x(0.8);
        let mut out = Pose { local: Vec::new() };
        r.apply(&pose, &mut out);
        assert_eq!(out.len(), 5);
        // root: the animation's; pelvis: its motion scaled to the shorter legs (0.95 * 0.8)
        assert_eq!(out.local[0].translation, Vec3::new(0.0, 0.0, 2.0));
        assert!(out.local[1].translation.abs_diff_eq(Vec3::new(0.0, 0.76, 0.0), 1e-5), "{}", out.local[1].translation);
        // a still bone: exactly the target's, direction included
        assert!(out.local[3].translation.abs_diff_eq(tg.rest[3].translation, 1e-5), "{}", out.local[3].translation);
        assert!(out.local[3].rotation.abs_diff_eq(Quat::from_rotation_x(0.8), 1e-6));
        // a joint the source lacks keeps the target's rest
        assert_eq!(out.local[4], tg.rest[4]);
        // the rest pose maps onto the target's rest pose
        r.apply(&Pose::rest(&s), &mut out);
        for (a, b) in out.local.iter().zip(&tg.rest) {
            assert!(a.translation.abs_diff_eq(b.translation, 1e-5));
        }
    }
}
