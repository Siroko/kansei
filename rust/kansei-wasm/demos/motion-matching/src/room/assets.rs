//! The room's CC0 assets (`www/assets/room/`, made by `tools/room_assets.py`): surfaces from
//! ambientCG and models from Poly Haven, all textures KTX2. Each texture is uploaded once and
//! shared by every material that uses it.

use std::collections::HashMap;

use glam::{Mat4, Vec3 as GVec3};

use kansei_core::buffers::Texture;
use kansei_core::geometries::{Geometry, Vertex};
use kansei_core::loaders::ktx2::CompressionSupport;
use kansei_core::loaders::{GLTFLoader, GLTFResult};
use kansei_core::renderers::Renderer;

use optimesh::simplifier::{simplify, simplify_sloppy, SimplifyTarget, VertexData, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_PRUNE};

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

/// A surface's layer in the albedo array (`albedo_array`): its place in `SURFACES`.
pub fn surface_layer(name: &str) -> Option<u8> {
    SURFACES.iter().position(|s| *s == name).map(|k| k as u8)
}

/// The hero's layer in the albedo array, after the surfaces.
pub const HERO_LAYER: u8 = SURFACES.len() as u8;

/// Each layer of the albedo array: 256 texels a side, its mips down to 1 (the last its mean).
const ALBEDO_SIZE: u32 = 256;

const BLIT_WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var linear: sampler;
struct VOut { @builtin(position) clip: vec4f, @location(0) uv: vec2f };
@vertex
fn vertex_main(@builtin(vertex_index) i: u32) -> VOut {
    let uv = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return VOut(vec4f(uv * vec2f(2.0, -2.0) + vec2f(-1.0, 1.0), 0.0, 1.0), uv);
}
@fragment
fn fragment_main(in: VOut) -> @location(0) vec4f {
    return vec4f(textureSample(source, linear, in.uv).rgb, 1.0);
}
"#;

/// The albedo texture array the reflections' hits read (`RtReflectionsEffect::set_albedo_texture`):
/// a layer for each of `SURFACES` (white where one didn't load) and then `extra` (the hero's base
/// colour), each drawn down from its colour map, with its mips.
pub fn albedo_array(renderer: &Renderer, assets: &Assets, extra: &[&wgpu::TextureView]) -> (wgpu::Texture, wgpu::TextureView) {
    let device = renderer.device();
    let layers = SURFACES.len() as u32 + extra.len() as u32;
    let levels = ALBEDO_SIZE.trailing_zeros() + 1;
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Room/Albedo"),
        size: wgpu::Extent3d { width: ALBEDO_SIZE, height: ALBEDO_SIZE, depth_or_array_layers: layers },
        mip_level_count: levels,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Room/AlbedoBlit"), source: wgpu::ShaderSource::Wgsl(BLIT_WGSL.into()) });
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Room/AlbedoBlit"),
        layout: None,
        vertex: wgpu::VertexState { module: &module, entry_point: Some("vertex_main"), buffers: &[], compilation_options: Default::default() },
        fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fragment_main"), targets: &[Some(format.into())], compilation_options: Default::default() }),
        primitive: Default::default(),
        depth_stencil: None,
        multisample: Default::default(),
        multiview: None,
        cache: None,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("Room/AlbedoBlit"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, mipmap_filter: wgpu::FilterMode::Linear, ..Default::default() });
    let white = Shared::upload(renderer, Texture::from_rgba("White", 1, 1, &[255, 255, 255, 255]));
    let level = |layer: u32, mip: u32| texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2), base_mip_level: mip, mip_level_count: Some(1), base_array_layer: layer, array_layer_count: Some(1), ..Default::default() });
    let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/Albedo") });
    for layer in 0..layers {
        let source = match SURFACES.get(layer as usize) {
            Some(name) => assets.surfaces.get(name).map_or(white.view(), |m| m.color.view()),
            None => extra[layer as usize - SURFACES.len()],
        };
        for mip in 0..levels {
            let from = (mip > 0).then(|| level(layer, mip - 1));
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(from.as_ref().unwrap_or(source)) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&sampler) },
                ],
            });
            let target = level(layer, mip);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Room/AlbedoBlit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment { view: &target, resolve_target: None, ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::WHITE), store: wgpu::StoreOp::Store } })],
                ..Default::default()
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.draw(0..3, 0..1);
        }
    }
    renderer.submit(std::iter::once(encoder.finish()));
    let view = texture.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
    (texture, view)
}

/// The mean of each layer of `albedo_array`'s texture (its last mip, read back), linear rgb.
pub async fn albedo_means(renderer: &Renderer, texture: &wgpu::Texture) -> Vec<[f32; 3]> {
    let layers = texture.depth_or_array_layers();
    let buffer = renderer.device().create_buffer(&wgpu::BufferDescriptor { label: Some("Room/AlbedoMeans"), size: 256 * layers as u64, usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
    let mut encoder = renderer.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Room/AlbedoMeans") });
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo { texture, mip_level: texture.mip_level_count() - 1, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
        wgpu::TexelCopyBufferInfo { buffer: &buffer, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(256), rows_per_image: Some(1) } },
        wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: layers },
    );
    renderer.submit(std::iter::once(encoder.finish()));
    let bytes: Vec<u8> = renderer.read_buffer_async(&buffer).await.unwrap_or_default();
    let linear = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    (0..layers as usize).map(|l| bytes.get(l * 256..l * 256 + 3).map_or([0.5; 3], |p| [linear(p[0]), linear(p[1]), linear(p[2])])).collect()
}

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

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
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

/// A model: its parts, its materials, its stand-in in the ray tracing grid and its bounds (model
/// space, after the parts' transforms).
pub struct Model {
    pub parts: Vec<ModelPart>,
    pub materials: Vec<ModelMaterial>,
    pub grid: Option<Geometry>,
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
    let parts: Vec<ModelPart> = result
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
    let solid: Vec<&Geometry> = parts.iter().filter(|p| !materials[p.material].cutout).map(|p| &p.geometry).collect();
    let grid = grid_stand_in(name, &solid);
    Ok(Model { parts, materials, grid, min, max })
}

/// How far (m) the grid's stand-in of a model strays from its surface, and the most triangles it
/// keeps.
const GRID_ERROR: f32 = 0.015;
const GRID_TRIANGLES: usize = 1500;

/// A model's solid parts as one mesh for the ray tracing grid: welded, simplified to `GRID_ERROR`
/// (small pieces pruned) and at most `GRID_TRIANGLES`. The grid's cost is the triangles a ray
/// meets in each 30 cm cell: Poly Haven's chairs put thousands there, which the rays' shadows,
/// bounces and reflections can't tell from a few hundred.
fn grid_stand_in(name: &str, parts: &[&Geometry]) -> Option<Geometry> {
    let sources: Vec<(&[Vertex], &[u32])> = parts.iter().map(|g| (g.vertices.as_slice(), g.indices.as_slice())).collect();
    let (kept, indices) = simplify_welded(&sources, 0, GRID_ERROR, GRID_TRIANGLES)?;
    let vertices = kept.iter().map(|&(part, v)| Vertex { position: parts[part].vertices[v].position, normal: [0.0, 1.0, 0.0], uv: [0.0, 0.0] }).collect();
    let total: usize = parts.iter().map(|g| g.indices.len() / 3).sum();
    log::info!("room: {name} in the grid as {} of {total} triangles", indices.len() / 3);
    Some(Geometry::new(&format!("{name}/Grid"), vertices, indices))
}

/// The vertices kept, as (mesh, vertex), and the triangles over them.
pub type Simplified = (Vec<(usize, usize)>, Vec<u32>);

/// Meshes (`vertices`, `indices` each) as one, for the ray tracing grid: welded by position (1 mm),
/// simplified toward `target_triangles` (0: as far as `error` allows) within `error` metres (small
/// pieces pruned), then cut down to at most `max_triangles`.
/// The vertices kept, as (mesh, vertex) of the input (each welded group's first), and the
/// triangles over them; `None` for nothing.
pub fn simplify_welded(meshes: &[(&[Vertex], &[u32])], target_triangles: usize, error: f32, max_triangles: usize) -> Option<Simplified> {
    let mut positions: Vec<f32> = Vec::new();
    let mut origin: Vec<(usize, usize)> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut welded: HashMap<[i32; 3], u32> = HashMap::new();
    for (m, (vertices, triangles)) in meshes.iter().enumerate() {
        let remap: Vec<u32> = vertices
            .iter()
            .enumerate()
            .map(|(k, v)| {
                let key = [0, 1, 2].map(|i| (v.position[i] * 1000.0).round() as i32);
                *welded.entry(key).or_insert_with(|| {
                    positions.extend_from_slice(&v.position[..3]);
                    origin.push((m, k));
                    (origin.len() - 1) as u32
                })
            })
            .collect();
        for t in triangles.chunks_exact(3) {
            let [a, b, c] = [0, 1, 2].map(|i| remap[t[i] as usize]);
            if a != b && b != c && a != c {
                indices.extend_from_slice(&[a, b, c]);
            }
        }
    }
    if indices.is_empty() {
        return None;
    }
    let vertices = VertexData { positions: &positions, count: positions.len() / 3, stride: 12 };
    let mut out = vec![0u32; indices.len()];
    let (mut count, _) = simplify(&mut out, &indices, &vertices, &SimplifyTarget { target_index_count: (target_triangles * 3).min(indices.len()), target_error: error, options: SIMPLIFY_ERROR_ABSOLUTE | SIMPLIFY_PRUNE });
    out.truncate(count);
    if count > max_triangles * 3 {
        let source = out.clone();
        let mut sloppy = vec![0u32; source.len()];
        count = simplify_sloppy(&mut sloppy, &source, &vertices, None, max_triangles * 3, f32::MAX).0;
        sloppy.truncate(count);
        out = sloppy;
    }
    // only the vertices kept
    let mut used: HashMap<u32, u32> = HashMap::new();
    let mut kept: Vec<(usize, usize)> = Vec::new();
    let out = out
        .iter()
        .map(|&i| {
            *used.entry(i).or_insert_with(|| {
                kept.push(origin[i as usize]);
                (kept.len() - 1) as u32
            })
        })
        .collect();
    Some((kept, out))
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
