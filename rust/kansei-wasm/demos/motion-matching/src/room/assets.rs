//! The room's CC0 assets (`www/assets/room/`, made by `tools/room_assets.py`): surfaces from
//! ambientCG and models from Poly Haven, all textures KTX2. Each texture is uploaded once and
//! shared by every material that uses it.

use std::collections::HashMap;

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::buffers::Texture;
use kansei_core::geometries::Geometry;
use kansei_core::loaders::ktx2::CompressionSupport;
use kansei_core::loaders::{GLTFLoader, GLTFResult};
use kansei_core::renderers::Renderer;

use super::pbr::PbrMaps;
use crate::fetch_bytes;

/// The surfaces: the floors, the walls, the ceiling, wood, the rug, the plinth's marble, the pond's
/// gravel, metal.
pub const SURFACES: [&str; 10] = ["marble", "wood", "plaster", "concrete", "ceiling", "oak", "rug", "blackmarble", "gravel", "metal"];

/// The models (Poly Haven names).
pub const MODELS: [&str; 18] = [
    "sofa_02",
    "mid_century_lounge_chair",
    "modern_arm_chair_01",
    "Ottoman_01",
    "modern_coffee_table_01",
    "modern_wooden_cabinet",
    "wooden_display_shelves_01",
    "steel_frame_shelves_03",
    "round_wooden_table_01",
    "dining_chair_02",
    "potted_plant_02",
    "potted_plant_04",
    "ceramic_vase_01",
    "ceramic_vase_03",
    "throw_pillows_01",
    "hanging_picture_frame_02",
    "book_encyclopedia_set_01",
    "modern_ceiling_lamp_01",
];

/// A texture on the GPU, handed to any number of materials.
pub struct Shared {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Shared {
    fn upload(renderer: &Renderer, mut texture: Texture) -> Self {
        texture.initialize_with_data(renderer.device(), renderer.queue());
        Self { texture: texture.gpu_texture().unwrap().clone(), view: texture.view().unwrap().clone() }
    }

    pub fn texture(&self, label: &str) -> Texture {
        Texture::from_view(label, self.texture.clone(), self.view.clone())
    }
}

/// A surface's three maps, shared.
pub struct SharedMaps {
    pub color: Shared,
    pub normal: Shared,
    pub orm: Shared,
}

impl SharedMaps {
    pub fn maps(&self, label: &str) -> PbrMaps {
        PbrMaps { color: self.color.texture(label), normal: self.normal.texture(label), orm: self.orm.texture(label) }
    }

    fn from(renderer: &Renderer, maps: PbrMaps) -> Self {
        Self { color: Shared::upload(renderer, maps.color), normal: Shared::upload(renderer, maps.normal), orm: Shared::upload(renderer, maps.orm) }
    }
}

/// One part of a model: its geometry (in the model's space) and the material slot it uses.
pub struct ModelPart {
    pub geometry: Geometry,
    pub material: usize,
}

/// A model's material: its maps, colour factor, whether both sides draw and whether it is cut out.
pub struct ModelMaterial {
    pub maps: SharedMaps,
    pub color: [f32; 4],
    pub double_sided: bool,
    pub cutout: bool,
}

/// A model: its parts, its materials and its bounds (model space, after the parts' transforms).
pub struct Model {
    pub parts: Vec<ModelPart>,
    pub materials: Vec<ModelMaterial>,
    pub min: GVec3,
    pub max: GVec3,
}

pub struct Assets {
    pub surfaces: HashMap<&'static str, SharedMaps>,
    pub models: HashMap<&'static str, Model>,
}

/// The texture of a glTF slot, or a flat stand-in.
fn slot(result: &GLTFResult, texture: Option<&kansei_core::loaders::GLTFTextureRef>, support: CompressionSupport, flat: [u8; 4]) -> Texture {
    texture
        .and_then(|t| result.load_texture(t, support).map_err(|e| log::warn!("room: {e}")).ok())
        .map(|t| t.into_texture())
        .unwrap_or_else(|| Texture::from_rgba("Flat", 1, 1, &flat))
}

fn model(renderer: &Renderer, name: &str, bytes: &[u8]) -> Result<Model, String> {
    let result = GLTFLoader::load_glb(bytes)?;
    let support = renderer.compression_support();
    let materials = result
        .materials
        .iter()
        .map(|m| {
            let color = slot(&result, m.base_color_texture.as_ref(), support, [255, 255, 255, 255]);
            let normal = slot(&result, m.normal_texture.as_ref(), support, [128, 128, 255, 255]);
            // Poly Haven's "arm": occlusion, roughness, metallic in r, g, b
            let orm = slot(&result, m.metallic_roughness_texture.as_ref().or(m.occlusion_texture.as_ref()), support, [255, (m.roughness * 255.0) as u8, (m.metallic * 255.0) as u8, 255]);
            let lower = m.name.to_ascii_lowercase();
            let cutout = name.contains("plant") || lower.contains("leaf") || lower.contains("leaves");
            ModelMaterial { maps: SharedMaps::from(renderer, PbrMaps { color, normal, orm }), color: m.base_color, double_sided: m.double_sided || cutout, cutout }
        })
        .collect::<Vec<_>>();
    let (mut min, mut max) = (GVec3::splat(f32::MAX), GVec3::splat(f32::MIN));
    let parts = result
        .renderables
        .iter()
        .map(|part| {
            // the part's node transform baked in (as Object3D composes it)
            let (p, r, s) = (part.position, part.rotation, part.scale);
            let m = Mat4::from_translation(GVec3::new(p.x, p.y, p.z)) * Mat4::from_rotation_z(r.z) * Mat4::from_rotation_y(r.y) * Mat4::from_rotation_x(r.x) * Mat4::from_scale(GVec3::new(s.x, s.y, s.z));
            let geometry = Geometry::merged(&format!("{name}/Part"), &[(&part.geometry, m)]);
            let (a, b) = geometry.bounds();
            min = min.min(a);
            max = max.max(b);
            ModelPart { geometry, material: part.material_index.min(materials.len().saturating_sub(1)) }
        })
        .collect();
    Ok(Model { parts, materials, min, max })
}

impl Assets {
    /// Fetch and upload everything (each missing file logged and left out).
    pub async fn load(renderer: &Renderer) -> Self {
        let support = renderer.compression_support();
        let mut surfaces = HashMap::new();
        for name in SURFACES {
            let get = |suffix: &'static str| async move { fetch_bytes(&format!("assets/room/{name}_{suffix}.ktx2")).await };
            let (c, n, o) = (get("color").await, get("normal").await, get("orm").await);
            match (c, n, o) {
                (Ok(c), Ok(n), Ok(o)) => match PbrMaps::from_ktx2(name, &c, &n, &o, support) {
                    Ok(maps) => {
                        surfaces.insert(name, SharedMaps::from(renderer, maps));
                    }
                    Err(e) => log::warn!("room: surface {name}: {e}"),
                },
                _ => log::warn!("room: surface {name} missing (tools/room_assets.py)"),
            }
        }
        let mut models = HashMap::new();
        for name in MODELS {
            match fetch_bytes(&format!("assets/room/{name}.glb")).await.and_then(|bytes| model(renderer, name, &bytes)) {
                Ok(m) => {
                    models.insert(name, m);
                }
                Err(e) => log::warn!("room: model {name}: {e}"),
            }
        }
        log::info!("room: {} surfaces, {} models", surfaces.len(), models.len());
        Self { surfaces, models }
    }

    /// A surface's maps, or flat grey when it didn't load.
    pub fn surface(&self, name: &str, label: &str) -> PbrMaps {
        self.surfaces.get(name).map_or_else(|| PbrMaps::flat([180, 180, 180, 255], 200, 0), |s| s.maps(label))
    }
}
