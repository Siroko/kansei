use crate::geometries::{Geometry, Vertex};
use crate::materials::{CullMode, Material, MaterialOptions};
use crate::math::Vec3;
use crate::objects::Renderable;
use super::ktx2::{self, CompressionSupport, GpuTarget, Ktx2Options, TranscodedTexture};

/// Material properties extracted from glTF PBR metallic-roughness.
pub struct GLTFMaterialInfo {
    pub name: String,
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
    pub double_sided: bool,
    /// sRGB colour (x base colour factor).
    pub base_color_texture: Option<GLTFTextureRef>,
    /// Linear: roughness in G, metalness in B.
    pub metallic_roughness_texture: Option<GLTFTextureRef>,
    /// Linear tangent-space normals.
    pub normal_texture: Option<GLTFTextureRef>,
    /// Linear: occlusion in R.
    pub occlusion_texture: Option<GLTFTextureRef>,
    /// sRGB colour.
    pub emissive_texture: Option<GLTFTextureRef>,
}

/// A material's texture: which image to load and how its texels are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GLTFTextureRef {
    /// The glTF texture index.
    pub texture: usize,
    /// The image to load: the `KHR_texture_basisu` KTX2 image when the texture has one,
    /// otherwise the texture's own source.
    pub image: usize,
    /// The texture's PNG/JPEG source when a KTX2 image is preferred over it.
    pub fallback_image: Option<usize>,
    /// Which UV set it reads.
    pub tex_coord: u32,
    /// Colour (base colour, emissive) rather than linear data.
    pub srgb: bool,
}

/// An image of the glTF file, its encoded bytes resolved where the file holds them.
pub struct GLTFImage {
    pub name: Option<String>,
    /// "image/ktx2", "image/png", "image/jpeg" (from the file, or guessed from the bytes).
    pub mime_type: Option<String>,
    /// The URI of an image stored outside the file.
    pub uri: Option<String>,
    /// The encoded image; `None` for an external image not read yet (see
    /// `GLTFResult::set_image_data`).
    pub data: Option<Vec<u8>>,
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
    pub images: Vec<GLTFImage>,
}

impl GLTFResult {
    /// Provide the bytes of an external image (for WASM, where `.gltf` files' image URIs are
    /// fetched separately).
    pub fn set_image_data(&mut self, image: usize, bytes: Vec<u8>) {
        self.images[image].data = Some(bytes);
    }

    /// Decode or transcode a material texture for a device with `support`: KTX2 images (from
    /// `KHR_texture_basisu`) become the best block-compressed format it samples, PNG/JPEG images
    /// RGBA8 (one mip). Colour textures load as sRGB, the rest as linear.
    pub fn load_texture(&self, texture: &GLTFTextureRef, support: CompressionSupport) -> Result<TranscodedTexture, String> {
        load_texture(&self.images, texture, support)
    }

    /// Every part as one geometry, each moved by its node transform (as `Object3D` composes it):
    /// a model to draw with one material, voxelize, or path trace. `Geometry::fit` sizes it.
    pub fn merged_geometry(&self, label: &str) -> Geometry {
        let parts: Vec<(&Geometry, glam::Mat4)> = self
            .renderables
            .iter()
            .map(|part| {
                let (p, r, s) = (part.position, part.rotation, part.scale);
                let rotation = glam::Mat4::from_rotation_z(r.z) * glam::Mat4::from_rotation_y(r.y) * glam::Mat4::from_rotation_x(r.x);
                (&part.geometry, glam::Mat4::from_translation(glam::Vec3::new(p.x, p.y, p.z)) * rotation * glam::Mat4::from_scale(glam::Vec3::new(s.x, s.y, s.z)))
            })
            .collect();
        Geometry::merged(label, &parts)
    }

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
                let mut material = Material::basic_lit(label, color, [0.15, 0.15, 0.15, 0.5]);
                material.options = opts;

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
    /// Load a glTF or glb file from disk, with its external buffers and images.
    pub fn load(path: &str) -> Result<GLTFResult, String> {
        let (document, buffers, base) = open(path)?;
        Ok(Self::from_document(&document, &buffers, base.as_deref()))
    }

    /// Load from in-memory glb bytes.
    pub fn load_glb(bytes: &[u8]) -> Result<GLTFResult, String> {
        let (document, buffers) = open_slice(bytes)?;
        Ok(Self::from_document(&document, &buffers, None))
    }

    /// Load from in-memory glTF JSON + external binary buffer(s).
    /// Use this for WASM where .gltf + .bin are fetched separately via HTTP; images stored in
    /// the buffers or as data URIs are resolved, external ones are left for
    /// `GLTFResult::set_image_data`.
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

        Ok(Self::from_document(&gltf.document, &buffers, None))
    }

    fn from_document(doc: &gltf::Document, buffers: &[gltf::buffer::Data], base: Option<&std::path::Path>) -> GLTFResult {
        GLTFResult {
            renderables: Self::parse_scene(doc, buffers),
            materials: Self::parse_materials(doc),
            images: Self::parse_images(doc, buffers, base),
        }
    }

    /// Each image's encoded bytes, read from its buffer view, its data URI, or (native, given a
    /// base directory) its file. `gltf::import` would also decode them, which fails for KTX2.
    pub(crate) fn parse_images(doc: &gltf::Document, buffers: &[gltf::buffer::Data], base: Option<&std::path::Path>) -> Vec<GLTFImage> {
        doc.images()
            .map(|img| {
                let (uri, mime, data) = match img.source() {
                    gltf::image::Source::View { view, mime_type } => {
                        let data = buffers.get(view.buffer().index()).and_then(|b| b.get(view.offset()..view.offset() + view.length())).map(|d| d.to_vec());
                        (None, Some(mime_type.to_string()), data)
                    }
                    gltf::image::Source::Uri { uri, mime_type } => {
                        let data = if let Some(rest) = uri.strip_prefix("data:") {
                            rest.split_once(";base64,").and_then(|(_, b64)| decode_base64(b64))
                        } else {
                            base.and_then(|dir| std::fs::read(dir.join(percent_decode(uri))).ok())
                        };
                        let mime = mime_type.map(str::to_string).or_else(|| uri.strip_prefix("data:").and_then(|r| r.split(';').next()).map(str::to_string));
                        (Some(uri.to_string()).filter(|u| !u.starts_with("data:")), mime, data)
                    }
                };
                let mime = mime.or_else(|| data.as_deref().filter(|d| ktx2::is_ktx2(d)).map(|_| "image/ktx2".to_string()));
                GLTFImage { name: img.name().map(str::to_string), mime_type: mime, uri, data }
            })
            .collect()
    }

    fn texture_ref(texture: gltf::Texture, tex_coord: u32, srgb: bool) -> Option<GLTFTextureRef> {
        let basisu = texture
            .extension_value("KHR_texture_basisu")
            .and_then(|ext| ext.get("source"))
            .and_then(|s| s.as_u64())
            .map(|s| s as usize);
        let source = texture.source().map(|img| img.index());
        let image = basisu.or(source)?;
        Some(GLTFTextureRef {
            texture: texture.index(),
            image,
            fallback_image: source.filter(|&s| s != image),
            tex_coord,
            srgb,
        })
    }

    pub(crate) fn parse_materials(doc: &gltf::Document) -> Vec<GLTFMaterialInfo> {
        doc.materials()
            .map(|mat| {
                let pbr = mat.pbr_metallic_roughness();
                GLTFMaterialInfo {
                    name: mat.name().unwrap_or("Unnamed").to_string(),
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    double_sided: mat.double_sided(),
                    base_color_texture: pbr.base_color_texture().and_then(|t| Self::texture_ref(t.texture(), t.tex_coord(), true)),
                    metallic_roughness_texture: pbr.metallic_roughness_texture().and_then(|t| Self::texture_ref(t.texture(), t.tex_coord(), false)),
                    normal_texture: mat.normal_texture().and_then(|t| Self::texture_ref(t.texture(), t.tex_coord(), false)),
                    occlusion_texture: mat.occlusion_texture().and_then(|t| Self::texture_ref(t.texture(), t.tex_coord(), false)),
                    emissive_texture: mat.emissive_texture().and_then(|t| Self::texture_ref(t.texture(), t.tex_coord(), true)),
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
                    // Object3D composes Rz * Ry * Rx, so decompose in that order
                    let (rz, ry, rx) = rotation.to_euler(glam::EulerRot::ZYX);

                    renderables.push(GLTFRenderable {
                        geometry: geo,
                        material_index: primitive.material().index().unwrap_or(0),
                        position: Vec3::new(translation.x, translation.y, translation.z),
                        rotation: Vec3::new(rx, ry, rz),
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

/// `texture`'s image from `images`, transcoded (KTX2) or decoded (PNG/JPEG, one RGBA8 mip) in its
/// colour space (`GLTFResult::load_texture`, `SkinnedGltf::load_texture`).
pub(crate) fn load_texture(images: &[GLTFImage], texture: &GLTFTextureRef, support: CompressionSupport) -> Result<TranscodedTexture, String> {
    let image = images.get(texture.image).ok_or_else(|| format!("no image {}", texture.image))?;
    let label = image.name.clone().unwrap_or_else(|| format!("GLTF/Image{}", texture.image));
    let bytes = image.data.as_deref().ok_or_else(|| format!("{label}: image data not loaded ({:?})", image.uri))?;
    if ktx2::is_ktx2(bytes) {
        let options = if texture.srgb { Ktx2Options::color() } else { Ktx2Options::linear() };
        return ktx2::transcode(&label, bytes, &options, support).map_err(|e| format!("{label}: {e}"));
    }
    let rgba = image::load_from_memory(bytes).map_err(|e| format!("{label}: {e}"))?.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok(TranscodedTexture {
        label,
        codec: image.mime_type.clone().unwrap_or_else(|| "image".into()),
        target: GpuTarget::Rgba8,
        format: GpuTarget::Rgba8.format(texture.srgb),
        width,
        height,
        layers: None,
        levels: vec![rgba.into_raw()],
        file_bytes: bytes.len(),
    })
}

/// A glTF or glb file from disk with its buffers (external ones read from beside it), without
/// decoding images as `gltf::import` does (it rejects KTX2), and the directory external images
/// are read from.
pub(crate) fn open(path: &str) -> Result<(gltf::Document, Vec<gltf::buffer::Data>, Option<std::path::PathBuf>), String> {
    let gltf = gltf::Gltf::open(path).map_err(|e| format!("Failed to load glTF '{}': {}", path, e))?;
    let base = std::path::Path::new(path).parent().map(|p| p.to_path_buf());
    let buffers = gltf::import_buffers(&gltf.document, base.as_deref(), gltf.blob.clone())
        .map_err(|e| format!("Failed to load glTF '{}' buffers: {}", path, e))?;
    Ok((gltf.document, buffers, base))
}

/// In-memory glb bytes (or a .gltf with embedded buffers), as `open`.
pub(crate) fn open_slice(bytes: &[u8]) -> Result<(gltf::Document, Vec<gltf::buffer::Data>), String> {
    let gltf = gltf::Gltf::from_slice(bytes).map_err(|e| format!("Failed to parse glb: {}", e))?;
    let buffers = gltf::import_buffers(&gltf.document, None, gltf.blob.clone()).map_err(|e| format!("Failed to parse glb buffers: {}", e))?;
    Ok((gltf.document, buffers))
}

/// Standard base64 (data URIs), ignoring whitespace; `None` on anything else.
fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => Some(c - b'A'),
        b'a'..=b'z' => Some(c - b'a' + 26),
        b'0'..=b'9' => Some(c - b'0' + 52),
        b'+' | b'-' => Some(62),
        b'/' | b'_' => Some(63),
        _ => None,
    };
    let mut out = Vec::with_capacity(text.len() * 3 / 4);
    let (mut acc, mut bits) = (0u32, 0);
    for c in text.bytes().filter(|c| !c.is_ascii_whitespace() && *c != b'=') {
        acc = (acc << 6) | value(c)? as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
        }
    }
    Some(out)
}

/// `%XX` escapes in a relative URI.
fn percent_decode(uri: &str) -> String {
    let b = uri.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        match (b[i], b.get(i + 1..i + 3).and_then(|h| u8::from_str_radix(std::str::from_utf8(h).ok()?, 16).ok())) {
            (b'%', Some(v)) => {
                out.push(v);
                i += 3;
            }
            (c, _) => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}
