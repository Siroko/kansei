use super::{Object3D, Renderable};
use crate::lights::Light;
use crate::math::Vec3;

/// A node in the scene graph.
pub enum SceneNode {
    Transform(Object3D),
    Renderable(Renderable),
    Light(Light),
}

/// The scene graph — holds a tree of SceneNode children, sorts opaque/transparent.
pub struct Scene {
    pub object: Object3D,
    children: Vec<SceneNode>,
    // Collected during prepare():
    collected_opaque: Vec<usize>,
    collected_transparent: Vec<usize>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            object: Object3D::new(),
            children: Vec::new(),
            collected_opaque: Vec::new(),
            collected_transparent: Vec::new(),
        }
    }

    /// Add a node to the scene. Returns its index.
    pub fn add(&mut self, node: SceneNode) -> usize {
        let idx = self.children.len();
        self.children.push(node);
        idx
    }

    /// Walk children, update matrices, collect renderables into opaque/transparent sorted lists.
    pub fn prepare(&mut self, camera_pos: &Vec3) {
        // Update transforms
        for node in &mut self.children {
            match node {
                SceneNode::Renderable(r) => {
                    if r.object.is_dirty() {
                        r.object.update_model_matrix();
                    }
                    r.object.update_world_matrix(None); // all root-level for now
                    r.object.update_normal_matrix();
                }
                SceneNode::Transform(o) => {
                    if o.is_dirty() {
                        o.update_model_matrix();
                    }
                    o.update_world_matrix(None);
                }
                SceneNode::Light(_) => {}
            }
        }

        // Collect renderables
        self.collected_opaque.clear();
        self.collected_transparent.clear();

        for (idx, node) in self.children.iter().enumerate() {
            if let SceneNode::Renderable(r) = node {
                if !r.visible {
                    continue;
                }
                if r.is_transparent() {
                    self.collected_transparent.push(idx);
                } else {
                    self.collected_opaque.push(idx);
                }
            }
        }

        // Sort transparent back-to-front
        let children = &self.children;
        self.collected_transparent.sort_by(|&a, &b| {
            let ra = match &children[a] {
                SceneNode::Renderable(r) => r,
                _ => unreachable!(),
            };
            let rb = match &children[b] {
                SceneNode::Renderable(r) => r,
                _ => unreachable!(),
            };
            if ra.render_order != rb.render_order {
                return ra.render_order.cmp(&rb.render_order);
            }
            let da = ra.position.distance_to_squared(camera_pos);
            let db = rb.position.distance_to_squared(camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Iterate opaque indices first, then sorted transparent indices.
    pub fn ordered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.collected_opaque
            .iter()
            .copied()
            .chain(self.collected_transparent.iter().copied())
    }

    /// Get a renderable by scene child index (immutable).
    pub fn get_renderable(&self, idx: usize) -> Option<&Renderable> {
        match self.children.get(idx)? {
            SceneNode::Renderable(r) => Some(r),
            _ => None,
        }
    }

    /// Get a renderable by scene child index (mutable).
    pub fn get_renderable_mut(&mut self, idx: usize) -> Option<&mut Renderable> {
        match self.children.get_mut(idx)? {
            SceneNode::Renderable(r) => Some(r),
            _ => None,
        }
    }

    /// Iterate all lights in the scene.
    pub fn lights(&self) -> impl Iterator<Item = &Light> {
        self.children.iter().filter_map(|node| match node {
            SceneNode::Light(l) => Some(l),
            _ => None,
        })
    }

    /// Count of collected renderables (after prepare).
    pub fn len(&self) -> usize {
        self.collected_opaque.len() + self.collected_transparent.len()
    }

    /// Whether there are no collected renderables.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total number of children (all node types).
    pub fn children_len(&self) -> usize {
        self.children.len()
    }

    // ── Backward-compat helpers used by CornellBox ──

    /// Internal: add a renderable directly (used by CornellBox).
    pub(crate) fn add_renderable_internal(&mut self, renderable: Renderable, _as_root: bool) -> usize {
        self.add(SceneNode::Renderable(renderable))
    }

    /// Internal: attach child (stub for CornellBox parent-child grouping, no-op for now).
    pub(crate) fn attach_child(&mut self, _parent_idx: usize, _child_idx: usize) {
        // TODO: hierarchical transforms for nested scene nodes
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
