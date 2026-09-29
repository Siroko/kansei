//! glTF materials whose textures use `KHR_texture_basisu`: the KTX2 image is found through the
//! extension (keeping the PNG fallback), read from a buffer view or a data URI, and transcoded in
//! the texture's colour space.

use kansei_core::loaders::ktx2::{CompressionSupport, GpuTarget};
use kansei_core::loaders::{GLTFLoader, GLTFResult, GLTFTextureRef};

mod ktx2_fixtures;
use ktx2_fixtures::fixture;

fn base64(bytes: &[u8]) -> String {
    const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, &b)| n | (b as u32) << (16 - 8 * i));
        for i in 0..4 {
            out.push(if i <= chunk.len() { A[(n >> (18 - 6 * i) & 63) as usize] as char } else { '=' });
        }
    }
    out
}

/// A triangle whose material has a KTX2 base colour (with a PNG fallback as a data URI) and a
/// KTX2-only normal map, both KTX2 images in the binary buffer. Returns (JSON, buffer).
fn scene() -> (String, Vec<u8>) {
    let mut bin: Vec<u8> = [0.0f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0].iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut view = |bytes: &[u8]| {
        let offset = bin.len();
        bin.extend(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        (offset, bytes.len())
    };
    let (color_at, color_len) = view(&fixture("uastc_rgba.ktx2"));
    let (normal_at, normal_len) = view(&fixture("uastc_normal.ktx2"));
    let json = format!(
        r#"{{
        "asset": {{ "version": "2.0" }},
        "extensionsUsed": ["KHR_texture_basisu"],
        "extensionsRequired": ["KHR_texture_basisu"],
        "buffers": [{{ "byteLength": {len}, "uri": "scene.bin" }}],
        "bufferViews": [
            {{ "buffer": 0, "byteOffset": 0, "byteLength": 36 }},
            {{ "buffer": 0, "byteOffset": {color_at}, "byteLength": {color_len} }},
            {{ "buffer": 0, "byteOffset": {normal_at}, "byteLength": {normal_len} }}
        ],
        "accessors": [{{ "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0,0,0], "max": [1,1,0] }}],
        "images": [
            {{ "name": "Color", "bufferView": 1, "mimeType": "image/ktx2" }},
            {{ "name": "ColorPng", "uri": "data:image/png;base64,{png}" }},
            {{ "name": "Normal", "bufferView": 2, "mimeType": "image/ktx2" }}
        ],
        "textures": [
            {{ "source": 1, "extensions": {{ "KHR_texture_basisu": {{ "source": 0 }} }} }},
            {{ "extensions": {{ "KHR_texture_basisu": {{ "source": 2 }} }} }}
        ],
        "materials": [{{
            "name": "Paint",
            "pbrMetallicRoughness": {{ "baseColorTexture": {{ "index": 0 }} }},
            "normalTexture": {{ "index": 1, "texCoord": 0 }}
        }}],
        "meshes": [{{ "primitives": [{{ "attributes": {{ "POSITION": 0 }}, "material": 0 }}] }}],
        "nodes": [{{ "mesh": 0 }}],
        "scenes": [{{ "nodes": [0] }}],
        "scene": 0
    }}"#,
        len = bin.len(),
        png = base64(&fixture("rgb.png")),
    );
    (json, bin)
}

fn glb(json: &str, bin: &[u8]) -> Vec<u8> {
    // a glb's buffer 0 is its binary chunk: no URI
    let mut json = json.replace(r#", "uri": "scene.bin""#, "").into_bytes();
    json.resize(json.len().next_multiple_of(4), b' ');
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = vec![];
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((json.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(&json);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(bin);
    out
}

fn check(result: &GLTFResult) {
    assert_eq!(result.renderables.len(), 1);
    let material = &result.materials[0];
    let color = material.base_color_texture.expect("base colour texture");
    assert_eq!(color, GLTFTextureRef { texture: 0, image: 0, fallback_image: Some(1), tex_coord: 0, srgb: true });
    let normal = material.normal_texture.expect("normal texture");
    assert_eq!((normal.image, normal.fallback_image, normal.srgb), (2, None, false));

    assert_eq!(result.images[0].mime_type.as_deref(), Some("image/ktx2"));
    assert_eq!(result.images[0].data.as_deref(), Some(fixture("uastc_rgba.ktx2").as_slice()));
    assert_eq!(result.images[1].mime_type.as_deref(), Some("image/png"));
    assert_eq!(result.images[1].data.as_deref(), Some(fixture("rgb.png").as_slice()));

    let bc = CompressionSupport { bc: true, ..CompressionSupport::NONE };
    let color_tex = result.load_texture(&color, bc).unwrap();
    assert_eq!((color_tex.target, color_tex.format), (GpuTarget::Bc7, wgpu::TextureFormat::Bc7RgbaUnormSrgb));
    assert_eq!(color_tex.levels.len(), 5);
    let normal_tex = result.load_texture(&normal, bc).unwrap();
    assert_eq!(normal_tex.format, wgpu::TextureFormat::Bc7RgbaUnorm);
    // the PNG fallback decodes to RGBA8 in the same colour space
    let png = result.load_texture(&GLTFTextureRef { image: 1, ..color }, bc).unwrap();
    assert_eq!((png.format, png.width, png.height), (wgpu::TextureFormat::Rgba8UnormSrgb, 20, 12));
}

#[test]
fn gltf_with_external_buffer_resolves_ktx2_images() {
    let (json, bin) = scene();
    check(&GLTFLoader::load_gltf_with_buffers(json.as_bytes(), vec![bin]).unwrap());
}

#[test]
fn glb_resolves_ktx2_images() {
    let (json, bin) = scene();
    check(&GLTFLoader::load_glb(&glb(&json, &bin)).unwrap());
}

#[test]
fn external_images_wait_for_their_bytes() {
    let (json, bin) = scene();
    let json = json.replace(r#""bufferView": 2, "mimeType": "image/ktx2""#, r#""uri": "normal%20map.ktx2""#);
    let mut result = GLTFLoader::load_gltf_with_buffers(json.as_bytes(), vec![bin]).unwrap();
    let normal = result.materials[0].normal_texture.unwrap();
    assert_eq!(result.images[2].uri.as_deref(), Some("normal%20map.ktx2"));
    assert!(result.load_texture(&normal, CompressionSupport::NONE).is_err());
    result.set_image_data(2, fixture("uastc_normal.ktx2"));
    assert_eq!(result.load_texture(&normal, CompressionSupport::NONE).unwrap().format, wgpu::TextureFormat::Rgba8Unorm);
}

/// The skinned-glTF importer reads the same materials and images, so a character's
/// `KHR_texture_basisu` textures load too (and `gltf::import`'s image decoding, which rejects
/// KTX2, is not in the way).
#[test]
fn skinned_import_resolves_ktx2_images() {
    let (json, bin) = scene();
    let skinned = kansei_core::animation::SkinnedGltf::from_slice(&glb(&json, &bin), None).unwrap();
    let color = skinned.materials[0].base_color_texture.expect("base colour texture");
    assert_eq!((color.image, color.fallback_image), (0, Some(1)));
    assert_eq!(skinned.images.len(), 3);
    let bc = CompressionSupport { bc: true, ..CompressionSupport::NONE };
    assert_eq!(skinned.load_texture(&color, bc).unwrap().format, wgpu::TextureFormat::Bc7RgbaUnormSrgb);
}
