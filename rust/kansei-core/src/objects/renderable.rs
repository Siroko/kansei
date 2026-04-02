use super::Object3D;
use crate::geometries::Geometry;
use crate::materials::Material;

/// A renderable object — owns geometry + material + transform.
pub struct Renderable {
    pub object: Object3D,
    pub geometry: Geometry,
    pub material: Material,
    pub instance_count: u32,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub render_order: i32,
    pub visible: bool,
    pub material_dirty: bool,
}

impl Renderable {
    pub fn new(geometry: Geometry, material: Material) -> Self {
        Self {
            object: Object3D::new(),
            geometry,
            material,
            instance_count: 1,
            cast_shadow: true,
            receive_shadow: true,
            render_order: 0,
            visible: true,
            material_dirty: true,
        }
    }

    /// Whether this renderable uses transparency.
    pub fn is_transparent(&self) -> bool {
        self.material.options.transparent
    }
}

impl std::ops::Deref for Renderable {
    type Target = Object3D;
    fn deref(&self) -> &Object3D {
        &self.object
    }
}

impl std::ops::DerefMut for Renderable {
    fn deref_mut(&mut self) -> &mut Object3D {
        &mut self.object
    }
}
