use crate::geometries::{Geometry, Vertex};
use crate::math::Vec3;

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
