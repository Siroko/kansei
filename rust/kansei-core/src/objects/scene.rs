use super::Renderable;
use crate::math::Vec3;

/// The scene graph — holds all renderables, sorts opaque/transparent.
pub struct Scene {
    pub position: Vec3,
    renderables: Vec<Renderable>,
    opaque_order: Vec<usize>,
    transparent_order: Vec<usize>,
}

impl Scene {
    pub fn new() -> Self {
        Self {
            position: Vec3::ZERO,
            renderables: Vec::new(),
            opaque_order: Vec::new(),
            transparent_order: Vec::new(),
        }
    }

    pub fn add(&mut self, renderable: Renderable) -> usize {
        let idx = self.renderables.len();
        self.renderables.push(renderable);
        idx
    }

    pub fn get(&self, index: usize) -> Option<&Renderable> {
        self.renderables.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut Renderable> {
        self.renderables.get_mut(index)
    }

    pub fn prepare(&mut self, camera_pos: &Vec3) {
        self.opaque_order.clear();
        self.transparent_order.clear();

        for (i, r) in self.renderables.iter().enumerate() {
            if !r.visible {
                continue;
            }
            if r.is_transparent() {
                self.transparent_order.push(i);
            } else {
                self.opaque_order.push(i);
            }
        }

        let renderables = &self.renderables;
        self.transparent_order.sort_by(|&a, &b| {
            let ra = &renderables[a];
            let rb = &renderables[b];
            if ra.render_order != rb.render_order {
                return ra.render_order.cmp(&rb.render_order);
            }
            let da = ra.position.distance_to_squared(camera_pos);
            let db = rb.position.distance_to_squared(camera_pos);
            db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    pub fn ordered_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.opaque_order.iter().copied().chain(self.transparent_order.iter().copied())
    }

    pub fn renderables(&self) -> &[Renderable] {
        &self.renderables
    }

    pub fn renderables_mut(&mut self) -> &mut [Renderable] {
        &mut self.renderables
    }

    pub fn len(&self) -> usize {
        self.renderables.len()
    }

    pub fn is_empty(&self) -> bool {
        self.renderables.is_empty()
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}
