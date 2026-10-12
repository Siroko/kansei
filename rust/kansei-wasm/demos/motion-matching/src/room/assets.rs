//! The room's textures on the GPU (each uploaded once and shared by every material that uses it),
//! the albedo array the reflections' hits read, and the mesh simplification the character's copy
//! in the ray tracing grid uses.

use std::collections::HashMap;

use kansei_core::buffers::Texture;
use kansei_core::geometries::Vertex;
use kansei_core::renderers::Renderer;

use optimesh::simplifier::{simplify, simplify_sloppy, simplify_with_attributes, Attributes, SimplifyTarget, VertexData, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_PERMISSIVE, SIMPLIFY_PRUNE};

use super::pbr::PbrMaps;

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
/// a layer for each of `sources` (the room's colour maps, then the hero's base colour), each drawn
/// down from its colour map, with its mips.
pub fn albedo_array(renderer: &Renderer, sources: &[&wgpu::TextureView]) -> (wgpu::Texture, wgpu::TextureView) {
    let device = renderer.device();
    let layers = sources.len().max(1) as u32;
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
        let source = sources.get(layer as usize).copied().unwrap_or(white.view());
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
    pub fn upload(renderer: &Renderer, mut texture: Texture) -> Self {
        texture.initialize_with_data(renderer.device(), renderer.queue());
        Self { texture: texture.gpu_texture().unwrap().clone(), view: texture.view().unwrap().clone() }
    }

    pub fn from_parts(texture: wgpu::Texture, view: wgpu::TextureView) -> Self {
        Self { texture, view }
    }

    pub fn parts(&self) -> (wgpu::Texture, wgpu::TextureView) {
        (self.texture.clone(), self.view.clone())
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

/// A textured mesh simplified for the ray tracing grid with its uvs intact (a copy welded by
/// position swaps uvs across the uv seams: the triangles there sample the texture across the
/// atlas, which mangled the hero's face and jacket in the mirrors). Welded only where vertices
/// share both position and uv (a hard edge's split normals), so the simplifier sees the seams
/// (vertices sharing a position, not a uv) and keeps every triangle on one uv island. Seams that
/// stay in place hold the hero at about 2250 of its 100k triangles, 1.5 ms more of tracing than
/// 1500; permissive collapse lets a seam move where the uvs on both sides agree, weighed by the
/// uvs (and a little the normals) in the error, which reaches the target with the texture intact.
/// Toward `target_triangles` within `error` metres; the vertices kept (indices into `vertices`)
/// and the triangles over them.
pub fn simplify_seamed(vertices: &[Vertex], indices: &[u32], target_triangles: usize, error: f32) -> Option<(Vec<usize>, Vec<u32>)> {
    let mut welded: HashMap<[i32; 5], u32> = HashMap::new();
    let mut origin: Vec<usize> = Vec::new();
    let remap: Vec<u32> = vertices
        .iter()
        .enumerate()
        .map(|(k, v)| {
            let p = [0, 1, 2].map(|i| (v.position[i] * 1000.0).round() as i32);
            let uv = v.uv.map(|c| (c * 8192.0).round() as i32);
            *welded.entry([p[0], p[1], p[2], uv[0], uv[1]]).or_insert_with(|| {
                origin.push(k);
                (origin.len() - 1) as u32
            })
        })
        .collect();
    let compact: Vec<Vertex> = origin.iter().map(|&k| vertices[k]).collect();
    let indices: Vec<u32> = indices.chunks_exact(3).map(|t| t.iter().map(|&i| remap[i as usize]).collect::<Vec<_>>()).filter(|t| t[0] != t[1] && t[1] != t[2] && t[0] != t[2]).flatten().collect();
    const FLOATS: usize = std::mem::size_of::<Vertex>() / 4;
    let flat: &[f32] = bytemuck::cast_slice(&compact);
    let data = VertexData { positions: flat, count: compact.len(), stride: FLOATS * 4 };
    // normal (3) and uv (2), after the position's four floats
    let weights = [0.05, 0.05, 0.05, 2.0, 2.0];
    let attributes = Attributes { data: &flat[4..], stride: FLOATS * 4, weights: &weights, count: 5 };
    let mut out = vec![0u32; indices.len()];
    let target = SimplifyTarget { target_index_count: (target_triangles * 3).min(indices.len()), target_error: error, options: SIMPLIFY_ERROR_ABSOLUTE | SIMPLIFY_PRUNE | SIMPLIFY_PERMISSIVE };
    let (count, _) = simplify_with_attributes(&mut out, &indices, &data, &attributes, None, &target);
    out.truncate(count);
    if out.is_empty() {
        return None;
    }
    let mut used: HashMap<u32, u32> = HashMap::new();
    let mut kept: Vec<usize> = Vec::new();
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

