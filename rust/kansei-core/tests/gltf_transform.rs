//! A glTF node's transform survives loading: the position, Euler rotation and scale the loader
//! hands out rebuild the node's world matrix through `Object3D::update_model_matrix`, including
//! rotations about several axes and nested nodes.

use glam::{Mat4, Quat, Vec3};
use kansei_core::loaders::GLTFLoader;
use kansei_core::objects::Object3D;

/// A triangle under a translated, scaled and rotated parent node, itself rotated about another
/// axis. Returns (JSON, buffer, the triangle's world matrix).
fn scene() -> (String, Vec<u8>, Mat4) {
    let bin: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let parent_rotation = Quat::from_euler(glam::EulerRot::XYZ, 0.7, -0.4, 1.1);
    let child_rotation = Quat::from_axis_angle(Vec3::new(1.0, 2.0, -0.5).normalize(), 0.9);
    let parent = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), parent_rotation, Vec3::new(1.0, -2.0, 3.0));
    let child = Mat4::from_rotation_translation(child_rotation, Vec3::new(0.5, 0.0, -1.0));
    let q = |q: Quat| format!("[{}, {}, {}, {}]", q.x, q.y, q.z, q.w);
    let json = format!(
        r#"{{
        "asset": {{ "version": "2.0" }},
        "buffers": [{{ "byteLength": 36 }}],
        "bufferViews": [{{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }}],
        "accessors": [{{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] }}],
        "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0 }} }}] }}],
        "nodes": [
            {{ "children": [1], "translation": [1, -2, 3], "rotation": {parent}, "scale": [2, 2, 2] }},
            {{ "mesh": 0, "translation": [0.5, 0, -1], "rotation": {child} }}
        ],
        "scenes": [{{ "nodes": [0] }}],
        "scene": 0
    }}"#,
        parent = q(parent_rotation),
        child = q(child_rotation),
    );
    (json, bin, parent * child)
}

#[test]
fn node_rotation_round_trips_through_object3d() {
    let (json, bin, world) = scene();
    let result = GLTFLoader::load_gltf_with_buffers(json.as_bytes(), vec![bin]).unwrap();
    assert_eq!(result.renderables.len(), 1);
    let node = &result.renderables[0];

    let mut object = Object3D::new();
    object.position = node.position;
    object.rotation = node.rotation;
    object.scale = node.scale;
    object.update_model_matrix();

    let got = object.model_matrix.to_glam();
    assert!(
        got.abs_diff_eq(world, 1e-4),
        "loaded transform rebuilds\n{got:?}\nnot the node's world matrix\n{world:?}"
    );
}
