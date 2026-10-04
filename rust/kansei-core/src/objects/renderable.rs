use super::Object3D;
use crate::geometries::Geometry;
use crate::materials::Material;

/// A renderable object — owns geometry + material + transform.
/// Instance count and per-instance buffers live on `Geometry`
/// (populated by `InstancedGeometry::new`).
pub struct Renderable {
    pub object: Object3D,
    pub geometry: Geometry,
    pub material: Material,
    pub cast_shadow: bool,
    pub receive_shadow: bool,
    pub render_order: i32,
    pub visible: bool,
    pub material_dirty: bool,
    /// Cull the instances on the GPU per view (camera, each shadow map) instead of drawing the
    /// geometry's instance buffer as is; see `InstanceCulling`.
    pub instance_culling: Option<crate::culling::InstanceCulling>,
    /// Draw the camera's cut of a cluster graph instead of the geometry: see `ClusterLod`.
    pub clusters: Option<crate::clusters::ClusterLod>,
    /// Bitmask of the layers this renderable is on (bit 0 by default). Secondary views such
    /// as planar reflections draw only renderables whose layers intersect their mask.
    pub layers: u32,
    /// Its transform changes every frame, so it is drawn directly in the pass each frame rather
    /// than recorded into the cached render bundle (false by default). Mark anything the app
    /// moves while it is on screen: the bundle is re-recorded only when the set of renderables
    /// it holds changes.
    pub dynamic: bool,
    /// Its surface in voxel GI (`Renderer::enable_voxel_gi`): drawn into the scene's voxel
    /// volume, through its material's own `vertex_main`, with this albedo and emission (or what
    /// its material's `voxel_fragment_entry` gives). `None` (the default) leaves it out of the
    /// volume: it neither bounces nor blocks light there. A `dynamic` renderable is voxelized every
    /// frame; the others again only when they change (transform, visibility, surface).
    pub gi: Option<crate::gi::GiSurface>,
    /// World matrix the renderer uploaded last frame (for motion vectors); updated by it.
    pub(crate) previous_world_matrix: std::cell::Cell<Option<crate::math::Mat4>>,
}

impl Renderable {
    /// Layer mask of a new renderable: bit 0.
    pub const DEFAULT_LAYERS: u32 = 1;

    pub fn new(geometry: impl Into<Geometry>, material: Material) -> Self {
        Self {
            object: Object3D::new(),
            geometry: geometry.into(),
            material,
            cast_shadow: true,
            receive_shadow: true,
            render_order: 0,
            visible: true,
            material_dirty: true,
            instance_culling: None,
            clusters: None,
            layers: Self::DEFAULT_LAYERS,
            dynamic: false,
            gi: None,
            previous_world_matrix: std::cell::Cell::new(None),
        }
    }

    /// Forget last frame's transform, so the next frame has no motion from this object (after a
    /// teleport, or on a camera cut).
    pub fn reset_motion(&mut self) {
        self.previous_world_matrix.set(None);
    }

    /// Whether this renderable uses instanced rendering.
    pub fn is_instanced(&self) -> bool {
        self.geometry.is_instanced()
    }

    /// Put this renderable in voxel GI with `surface` (see `gi`).
    pub fn with_gi(mut self, surface: crate::gi::GiSurface) -> Self {
        self.gi = Some(surface);
        self
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
