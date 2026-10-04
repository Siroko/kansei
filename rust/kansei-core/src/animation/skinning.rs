use glam::Mat4;

use super::{SkinnedMesh, Transform};
use crate::buffers::{BufferType, BufferUsage, ComputeBuffer, Sampler, Texture};
use crate::materials::{Binding, Material, MaterialOptions, ShaderStages};

/// WGSL for skinned materials: the bone palette (group 0 binding 1), the vertices' joints and
/// weights (group 0 binding 2) and `kansei_skin`. See the file's header for use.
pub const SKINNING_WGSL: &str = include_str!("../shaders/skinning.wgsl");

/// A skinned surface lit by a sun (cascade-shadowed) and a sky, writing motion vectors: the
/// shader `skinned_lit_material` builds on. Its uniform (group 0 binding 0) is
/// `SkinnedLitParams`.
pub const SKINNED_LIT_WGSL: &str = concat!(
    include_str!("../shaders/skinning.wgsl"),
    "\n",
    include_str!("../shaders/motion_vectors.wgsl"),
    "\n",
    include_str!("../shaders/cascaded_shadows.wgsl"),
    "\n",
    include_str!("../shaders/skinned_lit.wgsl"),
);

/// `SKINNED_LIT_WGSL` with textures: colour (sRGB), tangent-space normal (+Y up, no stored
/// tangents needed) and occlusion/roughness/metallic, read at the mesh's uv, with a GGX
/// specular. Its uniform is `SkinnedLitParams` (`base_color` tints the colour texture); see
/// `skinned_lit_textured_material`.
pub const SKINNED_LIT_TEXTURED_WGSL: &str = concat!(
    include_str!("../shaders/skinning.wgsl"),
    "\n",
    include_str!("../shaders/motion_vectors.wgsl"),
    "\n",
    include_str!("../shaders/cascaded_shadows.wgsl"),
    "\n",
    include_str!("../shaders/skinned_lit_textured.wgsl"),
);

/// Group 0 binding of the palette and of the per-vertex skin records.
pub const PALETTE_BINDING: u32 = 1;
pub const SKIN_BINDING: u32 = 2;

/// The joint matrices a skinned material reads, this frame's then last frame's (for motion
/// vectors), in one buffer that one `write_buffer` updates.
#[derive(Debug, Clone)]
pub struct BonePalette {
    joints: usize,
    /// `joints` current matrices, then `joints` previous ones.
    matrices: Vec<Mat4>,
    primed: bool,
    scratch: Vec<Mat4>,
}

impl BonePalette {
    /// A palette of `joints` identity matrices (no motion).
    pub fn new(joints: usize) -> Self {
        Self { joints, matrices: vec![Mat4::IDENTITY; 2 * joints.max(1)], primed: false, scratch: Vec::new() }
    }

    pub fn joints(&self) -> usize {
        self.joints
    }

    /// This frame's matrices; the previous frame's become last frame's. The first call (and the
    /// first after `reset_motion`) sets both, so a new or teleported mesh has no motion.
    pub fn set(&mut self, current: &[Mat4]) {
        assert_eq!(current.len(), self.joints);
        let (now, before) = self.matrices.split_at_mut(self.joints);
        if self.primed {
            before.copy_from_slice(now);
        } else {
            before.copy_from_slice(current);
            self.primed = true;
        }
        now.copy_from_slice(current);
    }

    /// `set` from a model-space pose of `mesh`'s skeleton.
    pub fn update(&mut self, mesh: &SkinnedMesh, model: &[Transform]) {
        let mut scratch = std::mem::take(&mut self.scratch);
        mesh.palette(model, &mut scratch);
        self.set(&scratch);
        self.scratch = scratch;
    }

    /// Forget last frame's matrices: the next `set` has no motion (after a teleport or cut).
    pub fn reset_motion(&mut self) {
        self.primed = false;
    }

    pub fn current(&self) -> &[Mat4] {
        &self.matrices[..self.joints]
    }

    pub fn previous(&self) -> &[Mat4] {
        &self.matrices[self.joints..2 * self.joints]
    }

    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.matrices)
    }

    /// A storage buffer holding the palette as it is now, for a material's binding 1.
    pub fn buffer(&self, label: &str) -> ComputeBuffer {
        ComputeBuffer::new(label, BufferType::ReadOnlyStorage, BufferUsage::STORAGE | BufferUsage::COPY_DST, self.as_bytes().to_vec())
    }

    /// Write the palette into its GPU buffer (the material's `bindable_buffer(PALETTE_BINDING)`).
    pub fn upload(&self, queue: &wgpu::Queue, buffer: &wgpu::Buffer) {
        queue.write_buffer(buffer, 0, self.as_bytes());
    }
}

/// A storage buffer of `mesh`'s per-vertex joints and weights, for a material's binding 2.
pub fn skin_buffer(label: &str, mesh: &SkinnedMesh) -> ComputeBuffer {
    ComputeBuffer::from_slice(label, BufferType::ReadOnlyStorage, BufferUsage::STORAGE, &mesh.skin_words())
}

/// A material drawing `mesh` skinned: `shader` (which includes `SKINNING_WGSL`), its uniform
/// `params` at binding 0 (visible to both stages), the palette's buffer at binding 1 and the
/// skin records at binding 2. Update the palette each frame with `BonePalette::upload` into
/// `material.bindable_buffer(PALETTE_BINDING)` (after the first render).
pub fn skinned_material<T: bytemuck::Pod>(label: &str, shader: &str, params: &[T], mesh: &SkinnedMesh, palette: &BonePalette, options: MaterialOptions) -> Material {
    assert_eq!(palette.joints(), mesh.skin_joints.len(), "the palette has one matrix per skin joint");
    let mut material = Material::new(
        label,
        shader,
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::storage(PALETTE_BINDING, ShaderStages::VERTEX, true),
            Binding::storage(SKIN_BINDING, ShaderStages::VERTEX, true),
        ],
        options,
    );
    material.set_uniform_bindable(0, &format!("{label}/Params"), params);
    material.set_bindable(PALETTE_BINDING, palette.buffer(&format!("{label}/Palette")));
    material.set_bindable(SKIN_BINDING, skin_buffer(&format!("{label}/Skin"), mesh));
    material
}

/// `SKINNED_LIT_WGSL`'s uniform: albedo, sun and sky in physical units.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedLitParams {
    /// Linear albedo (w unused).
    pub base_color: [f32; 4],
    /// The direction sunlight travels (w unused).
    pub sun_direction: [f32; 4],
    /// Colour times illuminance, lux.
    pub sun: [f32; 4],
    /// Zenith luminance, cd/m².
    pub sky: [f32; 4],
}

/// A `SKINNED_LIT_WGSL` material for `mesh`, writing motion vectors.
pub fn skinned_lit_material(label: &str, params: SkinnedLitParams, mesh: &SkinnedMesh, palette: &BonePalette) -> Material {
    skinned_material(label, SKINNED_LIT_WGSL, &[params], mesh, palette, MaterialOptions { outputs_velocity: true, ..Default::default() })
}

/// The textures of `skinned_lit_textured_material`: colour (sRGB), normal map and
/// occlusion/roughness/metallic (both linear), e.g. `Texture::from_image`.
pub struct SkinTextures {
    pub base_color: Texture,
    pub normal: Texture,
    pub orm: Texture,
}

/// A `SKINNED_LIT_TEXTURED_WGSL` material for `mesh`, writing motion vectors, its textures
/// sampled trilinearly with 8x anisotropy.
pub fn skinned_lit_textured_material(label: &str, params: SkinnedLitParams, mesh: &SkinnedMesh, palette: &BonePalette, textures: SkinTextures) -> Material {
    assert_eq!(palette.joints(), mesh.skin_joints.len(), "the palette has one matrix per skin joint");
    let mut material = Material::new(
        label,
        SKINNED_LIT_TEXTURED_WGSL,
        vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::storage(PALETTE_BINDING, ShaderStages::VERTEX, true),
            Binding::storage(SKIN_BINDING, ShaderStages::VERTEX, true),
            Binding::texture_2d(3, ShaderStages::FRAGMENT),
            Binding::texture_2d(4, ShaderStages::FRAGMENT),
            Binding::texture_2d(5, ShaderStages::FRAGMENT),
            Binding::sampler(6, ShaderStages::FRAGMENT),
        ],
        MaterialOptions { outputs_velocity: true, ..Default::default() },
    );
    material.set_uniform_bindable(0, &format!("{label}/Params"), &[params]);
    material.set_bindable(PALETTE_BINDING, palette.buffer(&format!("{label}/Palette")));
    material.set_bindable(SKIN_BINDING, skin_buffer(&format!("{label}/Skin"), mesh));
    material.set_bindable(3, textures.base_color);
    material.set_bindable(4, textures.normal);
    material.set_bindable(5, textures.orm);
    material.set_bindable(6, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_anisotropy(8));
    material
}
