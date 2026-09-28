//! Octahedral impostors: a far LOD of two triangles per instance for instanced renderables.
//!
//! `Renderer::bake_impostor` renders renderables of the scene (the parts of one object: bark and
//! foliage, say) with their own materials from N x N directions of an octahedral grid (or a
//! hemi-octahedral one, for things only seen from above), orthographically, into two atlases of
//! N x N frames: the albedo with the coverage, and the object-space normal with the depth. At
//! run time a material written with [`IMPOSTOR_WGSL`] draws [`billboard_geometry`] per instance,
//! facing the camera, and reads the three frames nearest the view direction with one step of
//! parallax: the albedo and normal to shade as the mesh's material does, and the surface point to
//! write depth from. Shadow maps and planar reflections draw it facing their own view.
//!
//! Its instances are culled like any instanced renderable's (`culling::InstanceCulling`, sharing
//! the meshes' instances, with the far LOD band).
//!
//! ```ignore
//! // bake from the tree's LOD0 renderables, with one instance record placing it at the origin
//! let impostor = renderer.bake_impostor(&mut scene, &[bark, foliage], &ImpostorOptions {
//!     instance: bytemuck::bytes_of(&[0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]).to_vec(),
//!     ..Default::default()
//! });
//! material.set_uniform_bindable(1, "Impostor", &[impostor.params()]);
//! material.set_bindable(2, impostor.albedo_texture());
//! material.set_bindable(3, impostor.normal_depth_texture());
//! let quad = InstancedGeometry::new(billboard_geometry("Trees/Impostor"), count, vec![instances]);
//! ```

mod bake;

use bytemuck::{Pod, Zeroable};

use crate::geometries::{Geometry, Vertex};

/// WGSL for drawing an [`Impostor`] in a material: `KanseiImpostor` (its `params`),
/// `kansei_impostor_corner` (the billboard, vertex stage), `kansei_impostor_sample` (albedo,
/// coverage, normal and surface point) and the octahedral mapping. See the file's header.
pub const IMPOSTOR_WGSL: &str = include_str!("../shaders/impostor.wgsl");

/// Which directions an impostor's frames are baked from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImpostorLayout {
    /// The whole sphere: for things seen from below too (in a mirror, from a slope).
    Octahedral,
    /// The upper hemisphere only: twice the frames' density there, for things only ever seen
    /// from above the horizon.
    HemiOctahedral,
}

/// How to bake an impostor.
#[derive(Clone, Debug)]
pub struct ImpostorOptions {
    /// Frames per side of the atlases (N x N views).
    pub frames: u32,
    /// Texels per side of a frame, a power of two (the atlases are `frames * frame_size` square,
    /// with a mip chain down to a texel per frame; two RGBA8 atlases of 12 x 128 take ~25 MB).
    pub frame_size: u32,
    pub layout: ImpostorLayout,
    /// Render texels per atlas texel side, averaged (antialiasing, and softer coverage).
    pub supersample: u32,
    /// The object's bounding box (min and max corners), object space; by default the one round
    /// the parts' vertex positions (right when `instance` leaves them in place). The frames are
    /// baked round the sphere through the box's corners (or round the vertices), and the
    /// billboard covers the box's outline.
    pub bounds: Option<(glam::Vec3, glam::Vec3)>,
    /// One instance record, in the layout of the parts' instance buffer, that places the object
    /// at the origin unrotated and unscaled (ignored for parts without instances).
    pub instance: Vec<u8>,
}

impl Default for ImpostorOptions {
    fn default() -> Self {
        Self { frames: 12, frame_size: 128, layout: ImpostorLayout::Octahedral, supersample: 2, bounds: None, instance: Vec::new() }
    }
}

/// `KanseiImpostor` in [`IMPOSTOR_WGSL`]: bind it as a uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct ImpostorParams {
    pub center: [f32; 3],
    pub radius: f32,
    pub extent: [f32; 3],
    pub frames: u32,
    pub hemi: u32,
    _pad: [u32; 3],
}

/// A baked impostor: its atlases and what `KanseiImpostor` needs to read them.
pub struct Impostor {
    pub frames: u32,
    pub frame_size: u32,
    pub layout: ImpostorLayout,
    /// The bounds' centre, the radius of the sphere the frames were baked round, and the box's
    /// half size (object space).
    pub center: glam::Vec3,
    pub radius: f32,
    pub extent: glam::Vec3,
    albedo: wgpu::Texture,
    normal_depth: wgpu::Texture,
}

impl Impostor {
    /// The atlases' format: albedo and coverage; normal (`n * 0.5 + 0.5`) and depth.
    pub const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    pub fn params(&self) -> ImpostorParams {
        ImpostorParams {
            center: self.center.to_array(),
            radius: self.radius,
            extent: self.extent.to_array(),
            frames: self.frames,
            hemi: (self.layout == ImpostorLayout::HemiOctahedral) as u32,
            _pad: [0; 3],
        }
    }

    /// Albedo (rgb) and coverage (a), to bind as a texture_2d<f32>.
    pub fn albedo_texture(&self) -> crate::buffers::Texture {
        crate::buffers::Texture::from_view("Impostor/Albedo", self.albedo.clone(), self.albedo.create_view(&Default::default()))
    }

    /// Object-space normal (rgb, `n * 0.5 + 0.5`) and depth (a: 0 at the frame's near side of
    /// the bounding sphere, 1 at its far side), to bind as a texture_2d<f32>.
    pub fn normal_depth_texture(&self) -> crate::buffers::Texture {
        crate::buffers::Texture::from_view("Impostor/NormalDepth", self.normal_depth.clone(), self.normal_depth.create_view(&Default::default()))
    }

    pub fn albedo_atlas(&self) -> &wgpu::Texture {
        &self.albedo
    }

    pub fn normal_depth_atlas(&self) -> &wgpu::Texture {
        &self.normal_depth
    }
}

/// The billboard an impostor material draws per instance: four corners at `position.xy` in
/// {-1, 1} (for `kansei_impostor_corner`), two triangles facing +z.
pub fn billboard_geometry(label: &str) -> Geometry {
    let corner = |x: f32, y: f32| Vertex { position: [x, y, 0.0, 1.0], normal: [0.0, 0.0, 1.0], uv: [x * 0.5 + 0.5, 0.5 - y * 0.5] };
    Geometry::new(label, vec![corner(-1.0, -1.0), corner(1.0, -1.0), corner(1.0, 1.0), corner(-1.0, 1.0)], vec![0, 1, 2, 0, 2, 3])
}

// The octahedral mapping and the frames' axes, as in IMPOSTOR_WGSL.

fn signs(v: glam::Vec2) -> glam::Vec2 {
    glam::Vec2::new(if v.x >= 0.0 { 1.0 } else { -1.0 }, if v.y >= 0.0 { 1.0 } else { -1.0 })
}

/// The grid position ([-1, 1] squared) of a direction, y up (`kansei_impostor_encode`).
#[cfg(test)]
pub(crate) fn encode(dir: glam::Vec3, hemi: bool) -> glam::Vec2 {
    let v = dir / (dir.x.abs() + dir.y.abs() + dir.z.abs());
    if hemi {
        let h = glam::Vec2::new(v.x, v.z) / (v.x.abs() + v.z.abs()).max(if v.y > 0.0 { 1.0 } else { 1e-12 });
        return glam::Vec2::new(h.x + h.y, h.x - h.y);
    }
    if v.y >= 0.0 {
        return glam::Vec2::new(v.x, v.z);
    }
    (glam::Vec2::ONE - glam::Vec2::new(v.z, v.x).abs()) * signs(glam::Vec2::new(v.x, v.z))
}

/// The unit direction at a grid position (`kansei_impostor_decode`).
pub(crate) fn decode(g: glam::Vec2, hemi: bool) -> glam::Vec3 {
    if hemi {
        let (x, z) = ((g.x + g.y) * 0.5, (g.x - g.y) * 0.5);
        return glam::Vec3::new(x, 1.0 - x.abs() - z.abs(), z).normalize();
    }
    let y = 1.0 - g.x.abs() - g.y.abs();
    if y < 0.0 {
        let xz = (glam::Vec2::ONE - glam::Vec2::new(g.y, g.x).abs()) * signs(g);
        return glam::Vec3::new(xz.x, y, xz.y).normalize();
    }
    glam::Vec3::new(g.x, y, g.y).normalize()
}

/// The up reference of a view from `dir`: y, or z looking straight down or up
/// (`kansei_impostor_basis` takes right = up x dir, up = dir x right).
pub(crate) fn up_reference(dir: glam::Vec3) -> glam::Vec3 {
    if dir.y.abs() > 0.999 { glam::Vec3::Z } else { glam::Vec3::Y }
}

/// The direction frame (column, row) of an N x N grid is baked from.
pub(crate) fn frame_direction(frames: u32, layout: ImpostorLayout, column: u32, row: u32) -> glam::Vec3 {
    let g = (glam::Vec2::new(column as f32, row as f32) + 0.5) / frames as f32 * 2.0 - 1.0;
    decode(g, layout == ImpostorLayout::HemiOctahedral)
}

#[cfg(test)]
mod tests;
