use glam::{Quat, Vec3};

use super::{nlerp, Pose, Skeleton, Transform};

/// An animation sampled at a fixed rate: every joint's local transform at every frame.
///
/// Frame `i` is at time `i / sample_rate`; the clip lasts `(frames - 1) / sample_rate`, so a
/// looping clip's last frame is the pose it wraps back to (its first), as exported by DCC tools.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub name: String,
    /// Frames per second.
    pub sample_rate: f32,
    /// Sampling past the end wraps around instead of holding the last frame.
    pub looping: bool,
    joints: usize,
    /// Frame-major: `frame * joints + joint`.
    rotations: Vec<Quat>,
    translations: Vec<Vec3>,
    /// Empty when every joint keeps unit scale.
    scales: Vec<Vec3>,
}

impl Clip {
    /// A clip from its frames' local poses (each with the same joint count).
    pub fn from_poses(name: &str, sample_rate: f32, poses: &[Pose]) -> Self {
        assert!(!poses.is_empty(), "a clip needs at least one frame");
        assert!(sample_rate > 0.0);
        let joints = poses[0].len();
        let mut rotations = Vec::with_capacity(poses.len() * joints);
        let mut translations = Vec::with_capacity(poses.len() * joints);
        let mut scales = Vec::with_capacity(poses.len() * joints);
        for pose in poses {
            assert_eq!(pose.len(), joints, "every frame has the same joints");
            for t in &pose.local {
                rotations.push(t.rotation);
                translations.push(t.translation);
                scales.push(t.scale);
            }
        }
        if scales.iter().all(|s| s.abs_diff_eq(Vec3::ONE, 1e-5)) {
            scales = Vec::new();
        }
        Self { name: name.to_string(), sample_rate, looping: false, joints, rotations, translations, scales }
    }

    pub fn frame_count(&self) -> usize {
        self.rotations.len() / self.joints.max(1)
    }

    pub fn joint_count(&self) -> usize {
        self.joints
    }

    /// Seconds from the first frame to the last.
    pub fn duration(&self) -> f32 {
        (self.frame_count() - 1) as f32 / self.sample_rate
    }

    /// Joint `joint`'s local transform at frame `frame`.
    pub fn transform(&self, frame: usize, joint: usize) -> Transform {
        let k = frame * self.joints + joint;
        Transform {
            translation: self.translations[k],
            rotation: self.rotations[k],
            scale: if self.scales.is_empty() { Vec3::ONE } else { self.scales[k] },
        }
    }

    /// Frame `frame`'s pose into `out`.
    pub fn frame_pose(&self, frame: usize, out: &mut Pose) {
        out.local.clear();
        out.local.extend((0..self.joints).map(|j| self.transform(frame, j)));
    }

    /// The frame pair and blend weight at `time` seconds: clamped to the clip, or wrapped if it
    /// loops.
    pub fn frames_at(&self, time: f32) -> (usize, usize, f32) {
        let last = self.frame_count() - 1;
        if last == 0 {
            return (0, 0, 0.0);
        }
        let mut f = time * self.sample_rate;
        if self.looping {
            f = f.rem_euclid(last as f32);
        }
        let f = f.clamp(0.0, last as f32);
        let a = (f.floor() as usize).min(last);
        let b = (a + 1).min(last);
        (a, b, f - a as f32)
    }

    /// The pose at `time` seconds, blending the two nearest frames, into `out`.
    pub fn sample(&self, time: f32, out: &mut Pose) {
        let (a, b, t) = self.frames_at(time);
        out.local.clear();
        out.local.extend((0..self.joints).map(|j| {
            let (ka, kb) = (a * self.joints + j, b * self.joints + j);
            Transform {
                translation: self.translations[ka].lerp(self.translations[kb], t),
                rotation: nlerp(self.rotations[ka], self.rotations[kb], t),
                scale: if self.scales.is_empty() { Vec3::ONE } else { self.scales[ka].lerp(self.scales[kb], t) },
            }
        }));
    }

    /// This clip, authored on `from`, for skeleton `to`: joints matched by name, and `to`'s
    /// rest transform for the joints `from` lacks.
    pub fn retarget_by_name(&self, from: &Skeleton, to: &Skeleton) -> Clip {
        assert_eq!(from.len(), self.joints);
        let source = from.map_names(to);
        let frames = self.frame_count();
        let mut poses = Vec::with_capacity(frames);
        for f in 0..frames {
            poses.push(Pose {
                local: source.iter().enumerate().map(|(j, s)| s.map_or(to.rest[j], |s| self.transform(f, s))).collect(),
            });
        }
        let mut clip = Clip::from_poses(&self.name, self.sample_rate, &poses);
        clip.looping = self.looping;
        clip
    }

    /// A joint's local transform at every frame: for a root joint, its motion in model space.
    pub fn track(&self, joint: usize) -> Vec<Transform> {
        (0..self.frame_count()).map(|f| self.transform(f, joint)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A two-joint skeleton: a root, and a child 1 m up the root's y.
    fn chain() -> Skeleton {
        Skeleton::new(
            vec!["root".into(), "child".into()],
            vec![None, Some(0)],
            vec![Transform::IDENTITY, Transform::from_translation_rotation(Vec3::Y, Quat::IDENTITY)],
        )
    }

    /// The root walking +z at 1 m/s and turning about y at 1 rad/s, sampled at 10 Hz for 1 s.
    fn walk() -> Clip {
        let poses: Vec<Pose> = (0..=10)
            .map(|i| {
                let t = i as f32 / 10.0;
                Pose { local: vec![Transform::from_translation_rotation(Vec3::new(0.0, 0.0, t), Quat::from_rotation_y(t)), chain().rest[1]] }
            })
            .collect();
        Clip::from_poses("walk", 10.0, &poses)
    }

    #[test]
    fn sampling_blends_the_nearest_frames() {
        let clip = walk();
        assert_eq!((clip.frame_count(), clip.joint_count()), (11, 2));
        assert!((clip.duration() - 1.0).abs() < 1e-6);
        let mut pose = Pose::rest(&chain());
        clip.sample(0.25, &mut pose);
        assert!(pose.local[0].translation.abs_diff_eq(Vec3::new(0.0, 0.0, 0.25), 1e-6));
        assert!(pose.local[0].rotation.abs_diff_eq(Quat::from_rotation_y(0.25), 1e-4));
        // clamped past the end without looping
        clip.sample(3.0, &mut pose);
        assert!(pose.local[0].translation.abs_diff_eq(Vec3::new(0.0, 0.0, 1.0), 1e-6));
    }

    #[test]
    fn a_looping_clip_wraps_at_its_last_frame() {
        let mut clip = walk();
        clip.looping = true;
        let (a, b, t) = clip.frames_at(1.25);
        assert_eq!((a, b), (2, 3));
        assert!((t - 0.5).abs() < 1e-4);
        let (a, _, t) = clip.frames_at(-0.05);
        assert_eq!(a, 9);
        assert!((t - 0.5).abs() < 1e-4);
    }

    #[test]
    fn retargeting_matches_joints_by_name() {
        let clip = walk();
        // the target has an extra joint first and the chain's joints in another order
        let target = Skeleton::new(
            vec!["extra".into(), "root".into(), "child".into()],
            vec![None, None, Some(1)],
            vec![Transform::from_translation_rotation(Vec3::X, Quat::IDENTITY), Transform::IDENTITY, Transform::IDENTITY],
        );
        let moved = clip.retarget_by_name(&chain(), &target);
        assert_eq!(moved.joint_count(), 3);
        assert_eq!(moved.transform(4, 0), target.rest[0]);
        assert_eq!(moved.transform(4, 1), clip.transform(4, 0));
        assert_eq!(moved.transform(4, 2), clip.transform(4, 1));
        assert_eq!(clip.track(0)[10].translation, Vec3::new(0.0, 0.0, 1.0));
    }
}
