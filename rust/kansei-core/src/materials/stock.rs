//! Stock materials: ready-made shaders for scenes that don't need their own.

use super::{Binding, Material, MaterialOptions};

/// Blinn-Phong under the scene's directional and point lights (camera group binding 2), with the
/// single directional shadow map and the point-light cube shadow (group 3). Forward, one colour
/// output. Group 0 binding 0: `color` then `specular` (rgb, shininess / 256 in `a`), two vec4s.
/// [`Material::basic_lit`] builds it.
pub const BASIC_LIT_WGSL: &str = include_str!("../shaders/basic_lit.wgsl");

/// A flat colour lit by a fixed light from above, for instanced geometry: the instance's model
/// matrix comes as four vec4 vertex attributes at locations 3-6
/// (`ComputeBuffer::with_vertex_mat4(3)`). Group 0 binding 0: `color` (vec4).
/// [`Material::basic_instanced`] builds it.
pub const BASIC_INSTANCED_WGSL: &str = include_str!("../shaders/basic_instanced.wgsl");

/// Camera-facing quads for particles, one per instance at a vec4 position (location 3), coloured
/// by height. Group 0 binding 0: `size`, `height_min`, `height_max`, a pad, then `color_low` and
/// `color_high` (vec4s), 48 bytes.
pub const PARTICLE_BILLBOARD_WGSL: &str = include_str!("../shaders/particle_billboard.wgsl");

/// WGSL for instances placed by a base point, a uniform scale and a yaw about +y (as
/// `glam::Mat4::from_rotation_y`, then scale, then the base): `kansei_place(local, inst, yaw)` takes a
/// point from the mesh to the world, with `inst` the base (xyz) and scale (w); `kansei_turn(v,
/// yaw)` turns a direction (a normal); `kansei_unplace(world, inst, yaw)` undoes `kansei_place`
/// (an impostor reads the camera in the tree's frame).
pub const INSTANCE_PLACEMENT_WGSL: &str = r#"
fn kansei_turn(v: vec3<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3<f32>(c * v.x + s * v.z, v.y, -s * v.x + c * v.z);
}

fn kansei_place(local: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    return kansei_turn(local, yaw) * inst.w + inst.xyz;
}

fn kansei_unplace(world: vec3<f32>, inst: vec4<f32>, yaw: f32) -> vec3<f32> {
    return kansei_turn((world - inst.xyz) / inst.w, -yaw);
}
"#;

impl Material {
    /// A [`BASIC_LIT_WGSL`] material: `color` (rgba, linear) under the scene's lights, with a
    /// Blinn-Phong highlight of `specular` (rgb; `a` is the shininess / 256). Change
    /// `options.cull_mode` and the like before the material is first drawn.
    pub fn basic_lit(label: &str, color: [f32; 4], specular: [f32; 4]) -> Material {
        let mut material = Material::new(label, BASIC_LIT_WGSL, vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)], MaterialOptions::default());
        let uniform: [f32; 8] = [color[0], color[1], color[2], color[3], specular[0], specular[1], specular[2], specular[3]];
        material.set_uniform_bindable(0, &format!("{label}/Color"), &uniform);
        material
    }

    /// A [`BASIC_INSTANCED_WGSL`] material of one `color` (rgba, linear), for an
    /// `InstancedGeometry` whose instance matrices sit at locations 3-6.
    pub fn basic_instanced(label: &str, color: [f32; 4]) -> Material {
        let mut material = Material::new(label, BASIC_INSTANCED_WGSL, vec![Binding::uniform(0, wgpu::ShaderStages::FRAGMENT)], MaterialOptions::default());
        material.set_uniform_bindable(0, &format!("{label}/Color"), &color);
        material
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The size of the WGSL struct `name` in `code`.
    fn struct_size(code: &str, name: &str) -> usize {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        let size = module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        });
        size.unwrap_or_else(|| panic!("no struct {name}"))
    }

    #[test]
    fn the_stock_uniforms_match_what_the_constructors_write() {
        assert_eq!(struct_size(BASIC_LIT_WGSL, "MaterialUniforms"), std::mem::size_of::<[f32; 8]>());
        assert_eq!(struct_size(BASIC_INSTANCED_WGSL, "MaterialUniforms"), std::mem::size_of::<[f32; 4]>());
        assert_eq!(struct_size(PARTICLE_BILLBOARD_WGSL, "ParticleParams"), 48);
    }

    #[test]
    fn the_placement_chunk_validates() {
        let module = naga::front::wgsl::parse_str(INSTANCE_PLACEMENT_WGSL).expect("parses");
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).expect("validates");
    }
}
