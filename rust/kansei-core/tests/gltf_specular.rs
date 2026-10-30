//! `KHR_materials_specular`'s `specularFactor` reaches `GLTFMaterialInfo::specular`, the
//! `StandardLitOptions::specular` a material is drawn with: 1 without the extension, the factor
//! with it (0 for a matte surface).

use kansei_core::loaders::GLTFLoader;

#[test]
fn the_specular_factor_is_read_and_defaults_to_one() {
    let json = r#"{
        "asset": { "version": "2.0" },
        "extensionsUsed": ["KHR_materials_specular"],
        "materials": [
            { "name": "Plain" },
            { "name": "Matte", "extensions": { "KHR_materials_specular": { "specularFactor": 0.0 } } },
            { "name": "Half", "extensions": { "KHR_materials_specular": { "specularFactor": 0.5, "specularColorFactor": [1, 0.5, 0.5] } } },
            { "name": "ColourOnly", "extensions": { "KHR_materials_specular": { "specularColorFactor": [0.2, 0.2, 0.2] } } }
        ],
        "scenes": [{ "nodes": [] }],
        "scene": 0
    }"#;
    let result = GLTFLoader::load_gltf_with_buffers(json.as_bytes(), Vec::new()).unwrap();
    let specular: Vec<(&str, f32)> = result.materials.iter().map(|m| (m.name.as_str(), m.specular)).collect();
    assert_eq!(specular, [("Plain", 1.0), ("Matte", 0.0), ("Half", 0.5), ("ColourOnly", 1.0)]);
}
