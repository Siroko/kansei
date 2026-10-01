use super::{Skeleton, Transform};

/// A skeleton's joints in local space (relative to their parents), as sampled from a clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    pub local: Vec<Transform>,
}

impl Pose {
    /// The skeleton's rest pose.
    pub fn rest(skeleton: &Skeleton) -> Self {
        Self { local: skeleton.rest.clone() }
    }

    pub fn len(&self) -> usize {
        self.local.len()
    }

    pub fn is_empty(&self) -> bool {
        self.local.is_empty()
    }

    /// Forward kinematics: each joint's transform in model space (the skeleton's root space),
    /// into `model` (resized to fit).
    pub fn to_model(&self, skeleton: &Skeleton, model: &mut Vec<Transform>) {
        debug_assert_eq!(self.local.len(), skeleton.len());
        model.clear();
        for (i, local) in self.local.iter().enumerate() {
            let m = match skeleton.parents[i] {
                Some(p) => model[p].mul(local),
                None => *local,
            };
            model.push(m);
        }
    }

    /// `to_model` into a new vector.
    pub fn model(&self, skeleton: &Skeleton) -> Vec<Transform> {
        let mut model = Vec::with_capacity(self.local.len());
        self.to_model(skeleton, &mut model);
        model
    }

    /// Blend towards `other` by `t` (0: self, 1: other), joint by joint.
    pub fn blend(&mut self, other: &Pose, t: f32) {
        for (a, b) in self.local.iter_mut().zip(&other.local) {
            *a = a.lerp(b, t);
        }
    }
}
