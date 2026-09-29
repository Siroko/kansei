use super::Transform;

/// A joint hierarchy: names, parents and the rest (bind) pose, in local space.
///
/// Joints are ordered parents first (`parents[i] < i`), so a single forward pass computes model
/// space (`Pose::to_model`).
#[derive(Debug, Clone, PartialEq)]
pub struct Skeleton {
    pub names: Vec<String>,
    /// Parent of each joint, `None` for a root.
    pub parents: Vec<Option<usize>>,
    /// Each joint's local transform at rest.
    pub rest: Vec<Transform>,
}

impl Skeleton {
    /// A skeleton from joints listed parents first. Panics if a parent comes after its child.
    pub fn new(names: Vec<String>, parents: Vec<Option<usize>>, rest: Vec<Transform>) -> Self {
        assert_eq!(names.len(), parents.len());
        assert_eq!(names.len(), rest.len());
        for (i, p) in parents.iter().enumerate() {
            if let Some(p) = p {
                assert!(*p < i, "joint {} ({}) comes before its parent {p}", i, names[i]);
            }
        }
        Self { names, parents, rest }
    }

    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Index of the joint called `name`.
    pub fn find(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// The rest pose in model space.
    pub fn rest_model(&self) -> Vec<Transform> {
        let mut model: Vec<Transform> = Vec::with_capacity(self.len());
        for (i, local) in self.rest.iter().enumerate() {
            model.push(match self.parents[i] {
                Some(p) => model[p].mul(local),
                None => *local,
            });
        }
        model
    }

    /// Whether `joint` is `ancestor` or below it.
    pub fn is_descendant(&self, mut joint: usize, ancestor: usize) -> bool {
        loop {
            if joint == ancestor {
                return true;
            }
            match self.parents[joint] {
                Some(p) => joint = p,
                None => return false,
            }
        }
    }

    /// For each of `other`'s joints, the index of the joint of this skeleton with its name.
    pub fn map_names(&self, other: &Skeleton) -> Vec<Option<usize>> {
        other.names.iter().map(|n| self.find(n)).collect()
    }
}
