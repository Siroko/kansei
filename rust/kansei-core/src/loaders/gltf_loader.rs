use crate::geometries::{Geometry, Vertex};
use crate::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages};
use crate::math::Vec3;
use crate::objects::Renderable;

const BASIC_LIT_WGSL: &str = include_str!("../shaders/basic_lit.wgsl");

/// Material properties extracted from glTF PBR metallic-roughness.
pub struct GLTFMaterialInfo {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
}

/// A loaded renderable with its transform.
pub struct GLTFRenderable {
    pub geometry: Geometry,
    pub material_index: usize,
    pub position: Vec3,
    pub rotation: Vec3,
    pub scale: Vec3,
}

/// Result of loading a glTF file.
pub struct GLTFResult {
    pub renderables: Vec<GLTFRenderable>,
    pub materials: Vec<GLTFMaterialInfo>,
}

impl GLTFResult {
    /// Convert into engine `Renderable`s with basic lit materials derived from glTF PBR data.
    /// Applies position, rotation, scale, and an optional extra uniform scale multiplier.
    pub fn into_renderables(self, scale_multiplier: f32) -> Vec<Renderable> {
        let materials = self.materials;
        self.renderables
            .into_iter()
            .map(|gr| {
                let mat_info = materials.get(gr.material_index);
                let (color, double_sided) = match mat_info {
                    Some(m) => (m.base_color, m.double_sided),
                    None => ([0.8, 0.8, 0.8, 1.0], false),
                };

                let mut opts = MaterialOptions::default();
                if double_sided {
                    opts.cull_mode = CullMode::None;
                }

                let label = mat_info
                    .map(|m| m.name.as_str())
                    .unwrap_or("GLTF/Material");
                let uniform: [f32; 8] = [
                    color[0], color[1], color[2], color[3],
                    0.15, 0.15, 0.15, 0.5,
                ];
                let mut material = Material::new(
                    label,
                    BASIC_LIT_WGSL,
                    vec![Binding::uniform(0, ShaderStages::FRAGMENT)],
                    opts,
                );
                material.set_uniform_bindable(0, &format!("{label}/Color"), &uniform);

                let s = scale_multiplier;
                let mut r = Renderable::new(gr.geometry, material);
                r.object.position = gr.position;
                r.object.rotation = gr.rotation;
                r.object.scale = Vec3::new(gr.scale.x * s, gr.scale.y * s, gr.scale.z * s);
                r.object.update_model_matrix();
                r.object.update_world_matrix(None);
                r
            })
            .collect()
    }
}

/// Loads glTF 2.0 files into engine objects.
pub struct GLTFLoader;

impl GLTFLoader {
    /// Load a glTF or glb file from disk.
    pub fn load(path: &str) -> Result<GLTFResult, String> {
        let (document, buffers, _images) =
            gltf::import(path).map_err(|e| format!("Failed to load glTF '{}': {}", path, e))?;
        let materials = Self::parse_materials(&document);
        let renderables = Self::parse_scene(&document, &buffers);
        Ok(GLTFResult {
            renderables,
            materials,
        })
    }

    /// Load from in-memory glb bytes.
    pub fn load_glb(bytes: &[u8]) -> Result<GLTFResult, String> {
        let (document, buffers, _images) =
            gltf::import_slice(bytes).map_err(|e| format!("Failed to parse glb: {}", e))?;
        let materials = Self::parse_materials(&document);
        let renderables = Self::parse_scene(&document, &buffers);
        Ok(GLTFResult {
            renderables,
            materials,
        })
    }

    /// Load from in-memory glTF JSON + external binary buffer(s).
    /// Use this for WASM where .gltf + .bin are fetched separately via HTTP.
    pub fn load_gltf_with_buffers(
        gltf_json: &[u8],
        external_buffers: Vec<Vec<u8>>,
    ) -> Result<GLTFResult, String> {
        let gltf = gltf::Gltf::from_slice(gltf_json)
            .map_err(|e| format!("Failed to parse glTF JSON: {}", e))?;

        // Wrap external buffers as gltf::buffer::Data
        let buffers: Vec<gltf::buffer::Data> = external_buffers
            .into_iter()
            .map(gltf::buffer::Data)
            .collect();

        let materials = Self::parse_materials(&gltf.document);
        let renderables = Self::parse_scene(&gltf.document, &buffers);
        Ok(GLTFResult {
            renderables,
            materials,
        })
    }

    fn parse_materials(doc: &gltf::Document) -> Vec<GLTFMaterialInfo> {
        doc.materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();
                GLTFMaterialInfo {
                    name: mat.name().unwrap_or("Unnamed").to_string(),
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    double_sided: mat.double_sided(),
                }
            })
            .collect()
    }

    fn parse_scene(
        doc: &gltf::Document,
        buffers: &[gltf::buffer::Data],
    ) -> Vec<GLTFRenderable> {
        let mut renderables = Vec::new();
        let scene = doc.default_scene().or_else(|| doc.scenes().next());
        if let Some(scene) = scene {
            for node in scene.nodes() {
                Self::process_node(&node, buffers, &glam::Mat4::IDENTITY, &mut renderables);
            }
        }
        renderables
    }

    fn process_node(
        node: &gltf::Node,
        buffers: &[gltf::buffer::Data],
        parent_transform: &glam::Mat4,
        renderables: &mut Vec<GLTFRenderable>,
    ) {
        let local = glam::Mat4::from_cols_array_2d(&node.transform().matrix());
        let world = *parent_transform * local;

        if let Some(mesh) = node.mesh() {
            for primitive in mesh.primitives() {
                if let Some(geo) = Self::parse_primitive(&primitive, buffers) {
                    let (scale, rotation, translation) = world.to_scale_rotation_translation();
                    let euler = rotation.to_euler(glam::EulerRot::YXZ);

                    renderables.push(GLTFRenderable {
                        geometry: geo,
                        material_index: primitive.material().index().unwrap_or(0),
                        position: Vec3::new(translation.x, translation.y, translation.z),
                        rotation: Vec3::new(euler.1, euler.0, euler.2),
                        scale: Vec3::new(scale.x, scale.y, scale.z),
                    });
                }
            }
        }

        for child in node.children() {
            Self::process_node(&child, buffers, &world, renderables);
        }
    }

    fn parse_primitive(
        primitive: &gltf::Primitive,
        buffers: &[gltf::buffer::Data],
    ) -> Option<Geometry> {
        let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()]));

        let positions: Vec<[f32; 3]> = reader.read_positions()?.collect();
        let vertex_count = positions.len();

        let normals: Vec<[f32; 3]> = reader
            .read_normals()
            .map(|n| n.collect())
            .unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; vertex_count]);

        let uvs: Vec<[f32; 2]> = reader
            .read_tex_coords(0)
            .map(|tc| tc.into_f32().collect())
            .unwrap_or_else(|| vec![[0.0, 0.0]; vertex_count]);

        let vertices: Vec<Vertex> = (0..vertex_count)
            .map(|i| Vertex {
                position: [positions[i][0], positions[i][1], positions[i][2], 1.0],
                normal: normals[i],
                uv: uvs[i],
            })
            .collect();

        let indices: Vec<u32> = reader
            .read_indices()
            .map(|idx| idx.into_u32().collect())
            .unwrap_or_else(|| (0..vertex_count as u32).collect());

        Some(Geometry::new("GLTF/Primitive", vertices, indices))
    }
}
