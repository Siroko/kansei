//! The Raggare intro's forest on Kansei: its terrain, road, lake, spruces and birches and ground
//! cover, from the intro's served data (`data.rs`) and its procedural trees, cover and materials
//! (`tree_*.rs`, `cover_*.rs`, examples/forest/*.wgsl), lit by the sun and the sky for the GI
//! modes to compare. The trees follow the film's layers: per species three LODs, the spruces'
//! foliage as card clusters near the camera, and an impostor baked from LOD0 far off, their bands
//! set each frame for the lens (`TreeLayer::update`).

use glam::{Vec2, Vec3};
use kansei_core::atmosphere::SKY_LIGHTING_WGSL;
use kansei_core::buffers::{BufferType, BufferUsage, ComputeBuffer, InstanceAttribute, Sampler, Texture, VertexFormat};
use kansei_core::cameras::MOTION_VECTORS_WGSL;
use kansei_core::clusters::{ClusterLod, ClusterMesh, ClusterOptions, InstanceTransform};
use kansei_core::culling::{InstanceCulling, LOD_FADE_WGSL};
use kansei_core::geometries::{Geometry, InstancedGeometry, Vertex};
use kansei_core::gi::{ClipmapProbes, GiSurface, CLIPMAP_PROBES_WGSL, VOXEL_WRITE_WGSL};
use kansei_core::impostors::{billboard_geometry, ImpostorOptions, IMPOSTOR_WGSL};
use kansei_core::loaders::ktx2::{self, Ktx2Options};
use kansei_core::materials::{Binding, CullMode, Material, MaterialOptions, ShaderStages, GBUFFER_OUT_WGSL};
use kansei_core::objects::{Renderable, Scene, SceneNode};
use kansei_core::renderers::Renderer;
use kansei_core::rt::{RtPlacement, RtSurface};
use kansei_core::shadows::{SkyOcclusion, CASCADED_SHADOWS_WGSL, SKY_OCCLUSION_WGSL};
use kansei_core::atmosphere::SkyAtmosphere;
use wasm_bindgen::JsValue;

use crate::canvas::mip_chain;
use crate::data::{Heightfield, SceneData, ROAD_SURFACES};
use crate::tree_meshes::{self, Mesh, TreeMesh, SPRUCE_EDGE_STYLE, SPRUCE_STYLES};
use crate::{cover_meshes, cover_textures, tree_textures};

// The materials' WGSL, shared with the TS page (examples/forest/).
const AMBIENT_WGSL: &str = include_str!("../../../../../examples/forest/ambient.wgsl");
const FOREST_WGSL: &str = include_str!("../../../../../examples/forest/forest.wgsl");
const TERRAIN_WGSL: &str = include_str!("../../../../../examples/forest/terrain.wgsl");
const ROAD_WGSL: &str = include_str!("../../../../../examples/forest/road.wgsl");
const WATER_WGSL: &str = include_str!("../../../../../examples/forest/water.wgsl");
const TREE_WGSL: &str = include_str!("../../../../../examples/forest/tree.wgsl");
const TREE_IMPOSTOR_WGSL: &str = include_str!("../../../../../examples/forest/tree_impostor.wgsl");
const COVER_WGSL: &str = include_str!("../../../../../examples/forest/cover.wgsl");

/// The layer the trees are on besides the default one: the canopy the sky occlusion sees.
pub const TREE_LAYER: u32 = 1 << 1;

/// The terrain mesh: every 2nd heightmap sample (2 m), as the film's, in tiles of 63 x 63 quads
/// (126 m; the voxelizer skips those away from what it voxelizes).
pub const TERRAIN_STEP: usize = 2;
const TILE_QUADS: usize = 63;
/// Height of the road ribbon above the exported centre line.
const ROAD_LIFT: f32 = 0.06;
/// How far the lake's quad reaches past its outline (m); the terrain hides it on land.
const LAKE_MARGIN_M: f32 = 20.0;

/// The ground scans' calibration (the layers of ground/*.ktx2, in order): the mean linear albedo
/// each is scaled to (the flat colours the film matched against Unreal), its tile size (m), and the
/// served file's own mean (measured once from ground/color.ktx2's full-size level, as the film does
/// at load).
const GROUND_SURFACES: [([f32; 3], f32, [f32; 3]); 6] = [
    ([0.060, 0.075, 0.040], 1.6, [0.12575, 0.16025, 0.03476]),
    ([0.040, 0.036, 0.026], 2.0, [0.09792, 0.05576, 0.03901]),
    ([0.045, 0.060, 0.022], 2.2, [0.32821, 0.30291, 0.10932]),
    ([0.160, 0.150, 0.130], 1.8, [0.30594, 0.25340, 0.12932]),
    ([0.095, 0.075, 0.052], 2.0, [0.20074, 0.14436, 0.08645]),
    ([0.260, 0.235, 0.180], 2.5, [0.33282, 0.25815, 0.15878]),
];

// Constant albedos for what traces the scene (the ray tracing grid's hits) and stands in for the
// GI's voxels where a material has no voxel entry.
const GROUND_ALBEDO: [f32; 3] = [0.05, 0.05, 0.034];
const ROAD_ALBEDO: [f32; 3] = [0.07, 0.066, 0.062];
const WATER_ALBEDO: [f32; 3] = [0.02, 0.026, 0.03];
const NEEDLES: [f32; 3] = [0.035, 0.065, 0.032];
const LEAVES: [f32; 3] = [0.05, 0.09, 0.026];
const SPRUCE_BARK: [f32; 3] = [0.1, 0.075, 0.06];
const BIRCH_BARK: [f32; 3] = [0.5, 0.49, 0.45];

/// Screen-height fractions above which LOD 0 and LOD 1 are used (LOD 2 below), the film's.
const LOD_THRESHOLDS: [f32; 2] = [0.22, 0.06];
/// Screen-height fraction below which a layer's impostor takes over from LOD 2.
const IMPOSTOR_THRESHOLD: f32 = 0.03;
/// With the spruces' card clusters: where their impostor takes over from the cards.
const CARD_IMPOSTOR_THRESHOLD: f32 = 0.1;
/// Half the crossfades' width, as a share of the distance where LOD0 meets LOD1.
const LOD_FADE: f32 = 0.12;
/// How much wider than its height the tree material may make an instance, for the cards' cull.
const CARD_STRETCH: f32 = 1.15;
/// A band no tree is in (the crossfades cannot widen it into view).
const NOWHERE: (f32, f32) = (f32::MAX, f32::MAX);
/// Texels per side of an impostor's frames.
const IMPOSTOR_FRAME: u32 = 64;
/// The bands (m from the camera) in which the GI's clipmap voxelizes each LOD: the finest within
/// 20 m (its finest levels), the middle one to 60 m, the coarsest past that.
const GI_BANDS: [(f32, f32); 3] = [(0.0, 20.0), (20.0, 60.0), (60.0, f32::INFINITY)];
/// Spruces within this distance of the road keep the edge style's low, full crowns.
const ROADSIDE_M: f32 = 9.0;

/// Where a tree's record puts a point of its mesh, as tree.wgsl's `vertex_main` does (wider or
/// narrower by a hash of where it stands, turned by minus its bearing), for the ray tracing grid.
const TREE_PLACEMENT_WGSL: &str = r#"
fn tree_hash12(p: vec2f) -> f32 {
    var p3 = fract(vec3f(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn kansei_rt_place(record: u32, p: vec3f) -> vec3f {
    let place = kansei_rt_record_vec4(record, 0u);
    let extra = kansei_rt_record_vec4(record, 4u);
    let w = place.w * (0.9 + 0.2 * tree_hash12(place.xz));
    let q = p * vec3f(w, place.w, w);
    let c = cos(-extra.x);
    let s = sin(-extra.x);
    return vec3f(c * q.x + s * q.z, q.y, -s * q.x + c * q.z) + place.xyz;
}
"#;

/// What every material's sky light reads (ambient.wgsl): the sky, the page's GI mode, the sky
/// occlusion and the clipmap's probes.
pub struct AmbientSources {
    sky_lighting: wgpu::Buffer,
    mode: wgpu::Buffer,
    occlusion: (wgpu::Texture, wgpu::TextureView, wgpu::Buffer),
    probes: (wgpu::Buffer, wgpu::Buffer),
}

impl AmbientSources {
    pub fn new(sky: &SkyAtmosphere, mode: &wgpu::Buffer, occlusion: &SkyOcclusion, probes: &ClipmapProbes) -> Self {
        Self {
            sky_lighting: sky.bindings().sky_lighting.clone(),
            mode: mode.clone(),
            occlusion: (occlusion.volume_texture().clone(), occlusion.volume.clone(), occlusion.params.clone()),
            probes: (probes.grid_buffer().clone(), probes.probe_buffer().clone()),
        }
    }

    /// Bind them at group 0 bindings 1-7 of `m`.
    fn bind(&self, m: &mut Material) {
        m.set_bindable(1, ComputeBuffer::from_external("SkyLighting", self.sky_lighting.clone(), BufferType::Uniform));
        m.set_bindable(2, ComputeBuffer::from_external("AmbientMode", self.mode.clone(), BufferType::Uniform));
        m.set_bindable(3, Texture::from_view("SkyOcclusion", self.occlusion.0.clone(), self.occlusion.1.clone()));
        m.set_bindable(4, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
        m.set_bindable(5, ComputeBuffer::from_external("SkyOcclusionParams", self.occlusion.2.clone(), BufferType::Uniform));
        m.set_bindable(6, ComputeBuffer::from_external("ClipProbeGrid", self.probes.0.clone(), BufferType::Uniform));
        m.set_bindable(7, ComputeBuffer::from_external("ClipProbes", self.probes.1.clone(), BufferType::Storage));
    }

    /// A forest material: `body` (one of the WGSL files) after the shared prelude, its own uniform
    /// `uniform` at binding 0, the ambient's at 1-7 and `extra` bindings after them.
    fn material(&self, label: &str, body: &str, uniform: &[f32], extra: Vec<Binding>, options: MaterialOptions) -> Material {
        let f = ShaderStages::FRAGMENT;
        let mut bindings = vec![
            Binding::uniform(0, ShaderStages::VERTEX | ShaderStages::FRAGMENT),
            Binding::uniform(1, f),
            Binding::uniform(2, f),
            Binding::texture_3d(3, f),
            Binding::sampler(4, f),
            Binding::uniform(5, f),
            Binding::uniform(6, f),
            Binding::storage(7, f, true),
        ];
        bindings.extend(extra);
        let code = format!(
            "{SKY_LIGHTING_WGSL}\n{SKY_OCCLUSION_WGSL}\n{CLIPMAP_PROBES_WGSL}\n{AMBIENT_WGSL}\n{CASCADED_SHADOWS_WGSL}\n{GBUFFER_OUT_WGSL}\n{VOXEL_WRITE_WGSL}\n{MOTION_VECTORS_WGSL}\n{LOD_FADE_WGSL}\n{FOREST_WGSL}\n{body}"
        );
        let options = MaterialOptions { mrt_output_count: Some(4), ..options };
        let mut m = Material::new(label, &code, bindings, options);
        m.set_uniform_bindable(0, label, uniform);
        self.bind(&mut m);
        m
    }
}

/// A texture uploaded once and bound to many materials.
struct Shared {
    label: String,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl Shared {
    fn new(renderer: &Renderer, mut texture: Texture, label: &str) -> Self {
        texture.initialize_with_data(renderer.device(), renderer.queue());
        Self { label: label.into(), texture: texture.gpu_texture().unwrap().clone(), view: texture.view().unwrap().clone() }
    }

    /// A painted RGBA8 image with its mip chain (the alpha test's coverage kept down the chain
    /// with `alpha_test`).
    fn painted(renderer: &Renderer, label: &str, size: (usize, usize), rgba: Vec<u8>, srgb: bool, alpha_test: Option<f32>) -> Self {
        let levels = mip_chain(size.0, size.1, rgba, alpha_test).into_iter().map(|(_, _, data)| data).collect();
        let format = if srgb { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm };
        Self::new(renderer, Texture::from_levels(label, format, size.0 as u32, size.1 as u32, levels), label)
    }

    /// A served KTX2 scan, transcoded for the device.
    fn ktx2(renderer: &Renderer, label: &str, bytes: &[u8], options: Ktx2Options) -> Result<Self, JsValue> {
        let t = ktx2::transcode(label, bytes, &options, renderer.compression_support()).map_err(|e| JsValue::from_str(&format!("{label}: {e}")))?;
        log::info!("{}", t.summary());
        Ok(Self::new(renderer, t.into_texture(), label))
    }

    fn bind(&self) -> Texture {
        Texture::from_view(&self.label, self.texture.clone(), self.view.clone())
    }
}

fn repeat_sampler() -> Sampler {
    Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::Repeat).with_anisotropy(16)
}

fn vertex(p: Vec3, n: Vec3, uv: Vec2) -> Vertex {
    Vertex { position: [p.x, p.y, p.z, 1.0], normal: n.to_array(), uv: uv.to_array() }
}

/// One tile of the terrain grid: quads `c0..c0 + cols` and `r0..r0 + rows` of the grid every
/// `TERRAIN_STEP` samples, its uvs addressing the splat's texel centres.
fn terrain_tile(h: &Heightfield, splat: ([usize; 2], f32), c0: usize, r0: usize, cols: usize, rows: usize) -> Geometry {
    let step = TERRAIN_STEP;
    let mut vertices = Vec::with_capacity((cols + 1) * (rows + 1));
    for r in r0..=r0 + rows {
        for c in c0..=c0 + cols {
            let (ix, iz) = (c * step, r * step);
            let x = h.origin[0] + ix as f32 * h.spacing;
            let z = h.origin[1] + iz as f32 * h.spacing;
            let uv = Vec2::new(((x - h.origin[0]) / splat.1 + 0.5) / splat.0[0] as f32, ((z - h.origin[1]) / splat.1 + 0.5) / splat.0[1] as f32);
            vertices.push(vertex(Vec3::new(x, h.at(ix, iz), z), h.normal(ix, iz, step), uv));
        }
    }
    let w = (cols + 1) as u32;
    let mut indices = Vec::with_capacity(cols * rows * 6);
    for r in 0..rows as u32 {
        for c in 0..cols as u32 {
            let i00 = r * w + c;
            let (i10, i01) = (i00 + 1, i00 + w);
            indices.extend_from_slice(&[i00, i01, i10, i10, i01, i01 + 1]);
        }
    }
    Geometry::new("Terrain", vertices, indices)
}

/// The road as a flat ribbon over the carriageway and shoulders; uv = (lateral -1..1, station m).
fn road_ribbon(rows: &[f32], stride: usize, half_width: f32) -> Geometry {
    let rows: Vec<&[f32]> = rows.chunks_exact(stride).collect();
    let centre = |i: usize| Vec3::new(rows[i][1], rows[i][2] + ROAD_LIFT, rows[i][3]);
    let mut vertices = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        let (s, c, t) = (r[0], centre(i), Vec3::new(r[4], 0.0, r[5]).normalize());
        let right = t.cross(Vec3::Y);
        // the normal tilts with the grade (the ribbon is level across)
        let along = centre((i + 1).min(rows.len() - 1)) - centre(i.saturating_sub(1));
        let n = right.cross(along).try_normalize().filter(|n| n.y > 0.0).unwrap_or(Vec3::Y);
        vertices.push(vertex(c - right * half_width, n, Vec2::new(-1.0, s)));
        vertices.push(vertex(c + right * half_width, n, Vec2::new(1.0, s)));
    }
    let mut indices = Vec::new();
    for i in 0..(vertices.len() / 2).saturating_sub(1) as u32 {
        let (l0, r0, l1, r1) = (2 * i, 2 * i + 1, 2 * i + 2, 2 * i + 3);
        indices.extend_from_slice(&[l0, r0, l1, l1, r0, r1]);
    }
    Geometry::new("Road", vertices, indices)
}

/// A flat quad over the lake's outline (and a margin) at its level.
fn lake_quad(level: f32, outline: &[[f32; 2]]) -> Geometry {
    let (mut lo, mut hi) = (Vec2::splat(f32::MAX), Vec2::splat(f32::MIN));
    for p in outline {
        lo = lo.min(Vec2::from(*p));
        hi = hi.max(Vec2::from(*p));
    }
    lo -= LAKE_MARGIN_M;
    hi += LAKE_MARGIN_M;
    let v = |x: f32, z: f32| vertex(Vec3::new(x, level, z), Vec3::Y, Vec2::new(x, z));
    Geometry::new("Lake", vec![v(lo.x, lo.y), v(hi.x, lo.y), v(lo.x, hi.y), v(hi.x, hi.y)], vec![0, 2, 1, 1, 2, 3])
}

/// Whether points lie near the road, by its samples in cells.
struct RoadProximity {
    cell: f32,
    cells: std::collections::HashMap<(i32, i32), Vec<(f32, f32)>>,
}

impl RoadProximity {
    fn new(rows: &[f32], stride: usize, cell: f32) -> Self {
        let mut cells: std::collections::HashMap<(i32, i32), Vec<(f32, f32)>> = Default::default();
        for r in rows.chunks_exact(stride) {
            cells.entry(((r[1] / cell).floor() as i32, (r[3] / cell).floor() as i32)).or_default().push((r[1], r[3]));
        }
        Self { cell, cells }
    }

    /// True if (x, z) is within `radius` (at most the cell size) of a road sample.
    fn within(&self, x: f32, z: f32, radius: f32) -> bool {
        let (cx, cz) = ((x / self.cell).floor() as i32, (z / self.cell).floor() as i32);
        (-1..=1).any(|dx| (-1..=1).any(|dz| self.cells.get(&(cx + dx, cz + dz)).is_some_and(|pts| pts.iter().any(|(px, pz)| (px - x).powi(2) + (pz - z).powi(2) < radius * radius))))
    }
}

/// A 0..1 hash of a position for CPU-side choices, the same in the TS page (integer mixing of the
/// position in centimetres, exact in both).
fn hash_xz(x: f32, z: f32) -> f32 {
    let mut h = ((x * 100.0).round() as i32 as u32).wrapping_mul(0x9E37_79B1) ^ ((z * 100.0).round() as i32 as u32).wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h = h.wrapping_mul(0x2C1B_3C6D);
    h ^= h >> 12;
    (h >> 8) as f32 / (1u32 << 24) as f32
}

#[derive(Clone, Copy)]
enum Species {
    /// A spruce style (`SPRUCE_STYLES`).
    Spruce(usize),
    Birch,
}

/// One species: three LODs, each a bark and a foliage renderable sharing an instance buffer, the
/// spruces' foliage also as card clusters, and an impostor; their bands set each frame (`update`).
pub struct TreeLayer {
    pub count: usize,
    /// The layer's median tree height (m): its LOD distances are the screen-size thresholds' for a
    /// tree this tall.
    height: f32,
    /// Scene indices of the (bark, foliage) renderables per LOD, the cards and the impostor.
    nodes: [(usize, usize); 3],
    cards: Option<usize>,
    impostor: Option<usize>,
    source: ComputeBuffer,
    /// The tree's box round the 1 m tree at LOD0.
    bounds: (Vec3, Vec3),
    pub triangles: [usize; 3],
}

/// The textures every tree and cover material shares.
pub struct Atlases {
    foliage: (Shared, Shared),
    spruce_bark: (Shared, Shared),
    birch_bark: (Shared, Shared),
    cover: (Shared, Shared),
}

impl Atlases {
    fn new(renderer: &Renderer) -> Self {
        let t = tree_textures::generate();
        let pair = |label: &str, size: (usize, usize), (c, n): (Vec<u8>, Vec<u8>), alpha: Option<f32>| {
            (Shared::painted(renderer, label, size, c, true, alpha), Shared::painted(renderer, &format!("{label}/Normal"), size, n, false, None))
        };
        let atlas = (tree_textures::ATLAS, tree_textures::ATLAS);
        let bark = (tree_textures::BARK_W, tree_textures::BARK_H);
        Self {
            foliage: pair("Trees/Foliage", atlas, t.foliage, Some(0.5)),
            spruce_bark: pair("Trees/SpruceBark", bark, t.spruce_bark, None),
            birch_bark: pair("Trees/BirchBark", bark, t.birch_bark, None),
            cover: pair("Cover", (cover_textures::ATLAS, cover_textures::ATLAS), cover_textures::atlas(), Some(0.5)),
        }
    }

    /// The foliage atlas, which the ray tracing grid's alpha test reads.
    pub fn foliage_view(&self) -> &wgpu::TextureView {
        &self.foliage.0.view
    }
}

fn textured_bindings() -> Vec<Binding> {
    vec![Binding::texture_2d(8, ShaderStages::FRAGMENT), Binding::texture_2d(9, ShaderStages::FRAGMENT), Binding::sampler(10, ShaderStages::FRAGMENT)]
}

fn bind_pair(m: &mut Material, pair: &(Shared, Shared)) {
    m.set_bindable(8, pair.0.bind());
    m.set_bindable(9, pair.1.bind());
    m.set_bindable(10, repeat_sampler());
}

/// A record buffer's instances as culled: the 32 bytes, then the crossfade's fade.
fn culled_instances(source: &ComputeBuffer) -> ComputeBuffer {
    source.clone().with_vertex_layout(
        36,
        vec![
            InstanceAttribute { shader_location: 3, offset: 0, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: 4, offset: 16, format: VertexFormat::Float32x4 },
            InstanceAttribute { shader_location: 5, offset: 32, format: VertexFormat::Float32 },
        ],
    )
}

/// A layer's instance culling, the same for every LOD: a box round the tree (`bounds`, the 1 m
/// tree, x each instance's height; as wide on both axes as its widest reach, since instances turn
/// about y).
fn tree_culling(source: &ComputeBuffer, count: usize, bounds: (Vec3, Vec3)) -> InstanceCulling {
    let (lo, hi) = bounds;
    // the hashed width reaches 1.1x the mesh's (tree.wgsl)
    let reach = lo.x.abs().max(hi.x).max(lo.z.abs()).max(hi.z) * 1.1;
    InstanceCulling::from_buffer(source, count as u32, 32, 0, 1.05)
        .with_radius_scale(12)
        .with_bounds_shift(Vec3::new(0.0, (lo.y + hi.y) * 0.5, 0.0))
        .with_bounds_box(Vec3::new(reach, (hi.y - lo.y) * 0.5, reach))
        // the bands are set each frame for the lens (`TreeLayer::update`)
        .with_crossfade(1.0)
}

/// What the build needs besides the data.
pub struct Build {
    pub ambient: AmbientSources,
    /// Build the ray tracing grid's surfaces.
    pub rt: bool,
    /// The wet surfaces' F0 and roughness for the ray-traced reflections ([road F0, the rest's F0,
    /// roughness]); F0 0: dry.
    pub wet: [f32; 3],
}

/// The scene's parts that change after the build.
pub struct Forest {
    pub trees: Vec<TreeLayer>,
    pub atlases: Atlases,
    pub triangles: u64,
    pub tree_count: usize,
}

impl Forest {
    /// Add the forest to `scene`.
    pub fn build(renderer: &mut Renderer, scene: &mut Scene, data: &SceneData, b: &Build) -> Result<Forest, JsValue> {
        let h = Heightfield::new(data);
        let terrain = &data.scene.terrain;
        let mut triangles = 0u64;

        // the terrain, in tiles, textured by the exported splat weights
        let splat = Shared::new(renderer, Texture::from_rgba("Splat", terrain.splat.samples[0] as u32, terrain.splat.samples[1] as u32, &data.splat), "Splat");
        let ground_color = Shared::ktx2(renderer, "Ground/colour", &data.ground_color, Ktx2Options::color())?;
        let ground_nra = Shared::ktx2(renderer, "Ground/nra", &data.ground_nra, Ktx2Options::linear())?;
        let mut layers = [0.0f32; 24];
        for (i, (target, tile, mean)) in GROUND_SURFACES.iter().enumerate() {
            for c in 0..3 {
                layers[4 * i + c] = target[c] / mean[c];
            }
            layers[4 * i + 3] = *tile;
        }
        let (cols, rows) = ((h.nx - 1) / TERRAIN_STEP, (h.nz - 1) / TERRAIN_STEP);
        for r0 in (0..rows).step_by(TILE_QUADS) {
            for c0 in (0..cols).step_by(TILE_QUADS) {
                let geometry = terrain_tile(&h, (terrain.splat.samples, terrain.splat.spacing), c0, r0, TILE_QUADS.min(cols - c0), TILE_QUADS.min(rows - r0));
                triangles += geometry.indices.len() as u64 / 3;
                let f = ShaderStages::FRAGMENT;
                let bindings = vec![Binding::texture_2d(8, f), Binding::sampler(9, f), Binding::texture_2d_array(10, f), Binding::texture_2d_array(11, f), Binding::sampler(12, f)];
                let options = MaterialOptions { voxel_fragment_entry: Some("voxel_main"), ..Default::default() };
                let mut m = b.ambient.material("Terrain", TERRAIN_WGSL, &layers, bindings, options);
                m.set_bindable(8, splat.bind());
                m.set_bindable(9, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
                m.set_bindable(10, ground_color.bind());
                m.set_bindable(11, ground_nra.bind());
                m.set_bindable(12, repeat_sampler());
                let mut r = Renderable::new(geometry, m).with_gi(GiSurface::new(GROUND_ALBEDO));
                r.cast_shadow = true;
                // the hits shaded with the height field's vertex normals, not its facets
                r.rt = b.rt.then(|| RtSurface::new(GROUND_ALBEDO).with_smooth_normals());
                scene.add(SceneNode::Renderable(r));
            }
        }

        // the road, with the scanned asphalts
        let spec = &data.scene.road.spec;
        let road = road_ribbon(&data.road, data.scene.road.stride, spec.width * 0.5 + spec.shoulder);
        triangles += road.indices.len() as u64 / 3;
        let params = [
            spec.width * 0.5,
            // as M_IntroRoad: the edge lines 0.3 m inside the asphalt's edge
            spec.width * 0.5 - 0.3,
            0.75,
            spec.width * 0.5 + spec.shoulder,
            0.7,
            2.0,
            0.35,
            1.7,
            b.wet[0],
            b.wet[2],
            0.0,
            0.0,
        ];
        let mut bindings: Vec<Binding> = (8..16).map(|i| Binding::texture_2d(i, ShaderStages::FRAGMENT)).collect();
        bindings.push(Binding::sampler(16, ShaderStages::FRAGMENT));
        let mut m = b.ambient.material("Road", ROAD_WGSL, &params, bindings, MaterialOptions { voxel_fragment_entry: Some("voxel_main"), ..Default::default() });
        for (i, (surface, (colour, nra))) in ROAD_SURFACES.iter().zip(&data.road_textures).enumerate() {
            m.set_bindable(8 + 2 * i as u32, Shared::ktx2(renderer, &format!("Road/{surface}/colour"), colour, Ktx2Options::color())?.bind());
            m.set_bindable(9 + 2 * i as u32, Shared::ktx2(renderer, &format!("Road/{surface}/nra"), nra, Ktx2Options::linear())?.bind());
        }
        m.set_bindable(16, repeat_sampler());
        let mut r = Renderable::new(road, m).with_gi(GiSurface::new(ROAD_ALBEDO));
        r.cast_shadow = true;
        r.rt = b.rt.then(|| RtSurface::new(ROAD_ALBEDO));
        scene.add(SceneNode::Renderable(r));

        // the lake
        let water = [WATER_ALBEDO[0], WATER_ALBEDO[1], WATER_ALBEDO[2], 1.0, 0.02, if b.wet[0] > 0.0 { 0.02 } else { 0.05 }, if b.wet[0] > 0.0 { 1.0 } else { 0.0 }, 0.0];
        let m = b.ambient.material("Lake", WATER_WGSL, &water, vec![], MaterialOptions { voxel_fragment_entry: Some("voxel_main"), ..Default::default() });
        let mut lake = Renderable::new(lake_quad(terrain.lake.level, &terrain.lake.outline), m).with_gi(GiSurface::new(WATER_ALBEDO));
        lake.cast_shadow = false;
        lake.rt = b.rt.then(|| RtSurface::new(WATER_ALBEDO));
        scene.add(SceneNode::Renderable(lake));

        // the forest: spruces along the road keep low, full crowns; inside it a random mix of slim
        // and full crowns; the birches among them
        let atlases = Atlases::new(renderer);
        let near_road = RoadProximity::new(&data.road, data.scene.road.stride, 20.0);
        let mut groups: [Vec<[f32; 8]>; 4] = Default::default();
        for inst in &data.trees {
            let group = if inst[5] > 0.5 {
                3
            } else if near_road.within(inst[0], inst[2], ROADSIDE_M) {
                SPRUCE_EDGE_STYLE
            } else {
                (hash_xz(inst[0], inst[2]) * 2.0) as usize % 2
            };
            groups[group].push(*inst);
        }
        let mut trees = Vec::new();
        for (g, instances) in groups.into_iter().enumerate() {
            if instances.is_empty() {
                continue;
            }
            let species = if g == 3 { Species::Birch } else { Species::Spruce(g) };
            let layer = TreeLayer::new(scene, &atlases, &b.ambient, species, instances, b.rt);
            triangles += layer.triangles.iter().map(|&t| (t * layer.count) as u64).max().unwrap_or(0);
            trees.push(layer);
        }
        for layer in &mut trees {
            layer.add_impostor(renderer, scene, &b.ambient);
        }

        // the ground cover: verge grass, flowers and shrubs from the scatter, near the camera,
        // standing on the rendered ground
        let meshes = cover_meshes::meshes();
        for (label, records, meshes, regions, near, far, radius) in [
            ("Grass", &data.grass, meshes.grass, [cover_textures::GRASS, cover_textures::SEEDS], 25.0, 160.0, 0.8),
            ("Flowers", &data.flowers, meshes.flowers, [cover_textures::FLOWERS, cover_textures::FLOWERS], 20.0, 120.0, 0.6),
            ("Shrubs", &data.bushes, meshes.shrub, [cover_textures::SHRUB, cover_textures::SHRUB], 60.0, 220.0, 1.0),
        ] {
            let mut instances = records.clone();
            for inst in &mut instances {
                inst[1] = h.mesh_height(inst[0], inst[2], TERRAIN_STEP);
            }
            if instances.is_empty() {
                continue;
            }
            let count = instances.len();
            let flat: Vec<f32> = instances.iter().flatten().copied().collect();
            let source = ComputeBuffer::from_slice(label, BufferType::Storage, BufferUsage::VERTEX | BufferUsage::STORAGE, &flat);
            let [ra, rb] = regions;
            let uniform = [ra.0[0], ra.0[1], ra.0[2], ra.0[3], rb.0[0], rb.0[1], rb.0[2], rb.0[3]];
            let bands = [(0.0, near), (near, far)];
            for (lod, mesh) in meshes.into_iter().enumerate() {
                let options = MaterialOptions { cull_mode: CullMode::None, shadow_fragment_entry: Some("shadow_fragment"), ..Default::default() };
                let mut m = b.ambient.material(label, COVER_WGSL, &uniform, textured_bindings(), options);
                bind_pair(&mut m, &atlases.cover);
                triangles += (mesh.triangles() * count) as u64 / 4;
                let geometry = Geometry::new(&format!("{label}/LOD{lod}"), mesh.vertices, mesh.indices);
                let mut r = Renderable::new(InstancedGeometry::new(geometry, count as u32, vec![culled_instances(&source)]), m);
                r.instance_culling = Some(
                    InstanceCulling::from_buffer(&source, count as u32, 32, 0, radius)
                        .with_radius_scale(12)
                        .with_lod_range(bands[lod].0, bands[lod].1)
                        // out of the GI's voxels and the ray tracing grid: too small for either
                        .with_gi_lod_range(0.0, 0.0)
                        .with_crossfade(2.0 * LOD_FADE * near),
                );
                r.cast_shadow = label == "Shrubs";
                scene.add(SceneNode::Renderable(r));
            }
        }
        let tree_count = trees.iter().map(|t| t.count).sum();
        Ok(Forest { trees, atlases, triangles, tree_count })
    }

    /// Set the trees' LOD bands for this frame's lens, `scale` times nearer than the film's
    /// thresholds (the shadow maps' `shadow` times nearer still).
    pub fn update(&self, scene: &mut Scene, tan_half_vfov: f32, scale: f32, shadow: f32) {
        for layer in &self.trees {
            layer.update(scene, tan_half_vfov, scale, scale * shadow);
        }
    }
}

impl TreeLayer {
    fn new(scene: &mut Scene, atlases: &Atlases, ambient: &AmbientSources, species: Species, instances: Vec<[f32; 8]>, rt: bool) -> Self {
        let count = instances.len();
        let flat: Vec<f32> = instances.iter().flatten().copied().collect();
        let label = match species {
            Species::Spruce(s) => format!("Spruce{s}"),
            Species::Birch => "Birch".to_string(),
        };
        let source = ComputeBuffer::from_slice(&label, BufferType::Storage, BufferUsage::VERTEX | BufferUsage::STORAGE, &flat);
        let mut heights: Vec<f32> = instances.iter().map(|i| i[3]).collect();
        heights.sort_by(f32::total_cmp);
        let height = heights[heights.len() / 2];
        let meshes: Vec<TreeMesh> = (0..3)
            .map(|lod| match species {
                Species::Spruce(style) => tree_meshes::spruce(lod, &SPRUCE_STYLES[style]),
                Species::Birch => tree_meshes::birch(lod, 9),
            })
            .collect();
        let mut bounds = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in meshes[0].bark.vertices.iter().chain(&meshes[0].foliage.vertices) {
            let p = Vec3::new(v.position[0], v.position[1], v.position[2]);
            bounds = (bounds.0.min(p), bounds.1.max(p));
        }
        let (bark_albedo, foliage_albedo, bark_tex) = match species {
            Species::Spruce(_) => (SPRUCE_BARK, NEEDLES, &atlases.spruce_bark),
            Species::Birch => (BIRCH_BARK, LEAVES, &atlases.birch_bark),
        };
        let birch = matches!(species, Species::Birch) as u32 as f32;
        let renderable = |name: &str, mesh: Mesh, foliage: bool| {
            let options = MaterialOptions {
                cull_mode: if foliage { CullMode::None } else { CullMode::Back },
                shadow_fragment_entry: Some("shadow_fragment"),
                voxel_fragment_entry: Some("voxel_main"),
                ..Default::default()
            };
            let mut m = ambient.material(name, TREE_WGSL, &[1.0, 1.0, 1.0, 1.0, foliage as u32 as f32, birch, 0.0, 0.0], textured_bindings(), options);
            bind_pair(&mut m, if foliage { &atlases.foliage } else { bark_tex });
            let geometry = Geometry::new(name, mesh.vertices, mesh.indices);
            let mut r = Renderable::new(InstancedGeometry::new(geometry, count as u32, vec![culled_instances(&source)]), m)
                .with_gi(GiSurface::new(if foliage { foliage_albedo } else { bark_albedo }));
            r.instance_culling = Some(tree_culling(&source, count, bounds));
            r.cast_shadow = true;
            // in the scene, and the canopy the sky occlusion is built from
            r.layers = Renderable::DEFAULT_LAYERS | TREE_LAYER;
            r
        };
        let mut triangles = [0; 3];
        let mut card_foliage = None;
        let mut nodes = [(0, 0); 3];
        for (lod, TreeMesh { bark, foliage }) in meshes.into_iter().enumerate() {
            triangles[lod] = bark.triangles() + foliage.triangles();
            let spruce = matches!(species, Species::Spruce(_));
            if spruce && lod == 0 {
                card_foliage = Some(Mesh { vertices: foliage.vertices.clone(), indices: foliage.indices.clone() });
            }
            let mut bark_r = renderable(&format!("{label}/Bark/LOD{lod}"), bark, false);
            let mut foliage_r = renderable(&format!("{label}/Foliage/LOD{lod}"), foliage, true);
            // the ray tracing grid: LOD0's bark, and the birches' leaves (the spruces' needles are
            // their cards'), their foliage cut out by the atlas' alpha (layer 0)
            if rt && lod == 0 {
                bark_r.rt = Some(RtSurface::new(bark_albedo));
                bark_r.rt_placement = Some(RtPlacement::Wgsl(TREE_PLACEMENT_WGSL.into()));
                if !spruce {
                    foliage_r.rt = Some(RtSurface::new(foliage_albedo).with_alpha_layer(0));
                    foliage_r.rt_placement = Some(RtPlacement::Wgsl(TREE_PLACEMENT_WGSL.into()));
                }
            }
            nodes[lod] = (scene.add(SceneNode::Renderable(bark_r)), scene.add(SceneNode::Renderable(foliage_r)));
        }
        // the spruces' foliage as card clusters: one renderable over every instance, each tree's
        // cut picked per cluster and per view (the camera, the cascades, the clipmap, the grid)
        let cards = card_foliage.map(|mesh| {
            let name = format!("{label}/Foliage/Cards");
            let geometry = Geometry::new(&name, mesh.vertices.clone(), mesh.indices.clone());
            let clusters = ClusterMesh::build(&geometry, &ClusterOptions { cards: true, card_error_scale: 0.25, ..Default::default() });
            let mut r = renderable(&name, mesh, true);
            // tree.wgsl: the record's X, Y, Z, then the height as the scale and the bearing, which
            // turns the tree by minus itself
            let transform = InstanceTransform::Placement { position: 0, scale: Some(12), yaw: Some(16), yaw_scale: -1.0, rotation: None };
            r.clusters = Some(ClusterLod::new(clusters).with_transform(transform).with_stretch(CARD_STRETCH));
            if rt {
                r.rt = Some(RtSurface::new(foliage_albedo).with_alpha_layer(0));
                r.rt_placement = Some(RtPlacement::Wgsl(TREE_PLACEMENT_WGSL.into()));
            }
            scene.add(SceneNode::Renderable(r))
        });
        Self { count, height, nodes, cards, impostor: None, source, bounds, triangles }
    }

    /// Bake the layer's LOD0 into an octahedral impostor (12 x 12 frames of `IMPOSTOR_FRAME`
    /// texels) and add it as the far LOD: a billboard per tree, on the same instances, baked at the
    /// layer's median height.
    fn add_impostor(&mut self, renderer: &mut Renderer, scene: &mut Scene, ambient: &AmbientSources) {
        let (bark, foliage) = self.nodes[0];
        // the instance it is baked from: at the origin, unturned, the median height; tree.wgsl's
        // hash of the origin is 0, so its width factor is 0.9 and its tint 0.85
        let (width, tint) = (0.9, 0.85);
        let h = self.height;
        let scale = Vec3::new(width * h, h, width * h);
        let impostor = renderer.bake_impostor(scene, &[bark, foliage], &ImpostorOptions {
            frame_size: IMPOSTOR_FRAME,
            bounds: Some((self.bounds.0 * scale, self.bounds.1 * scale)),
            // and a fade of 1: drawn whole
            instance: bytemuck_floats(&[0.0, 0.0, 0.0, h, 0.0, 0.0, 0.0, 0.0, 1.0]),
            ..Default::default()
        });
        let label = "TreeImpostor";
        let f = ShaderStages::FRAGMENT;
        let options = MaterialOptions { cull_mode: CullMode::None, shadow_fragment_entry: Some("shadow_fragment"), ..Default::default() };
        // the bake's height, width and tint, then the impostor's parameters (one uniform: the
        // fragment stage has room for no more)
        let mut uniform = vec![h, width, tint, 0.0];
        uniform.extend_from_slice(bytemuck::cast_slice::<_, f32>(&[impostor.params()]));
        let mut m = ambient.material(label, &format!("{IMPOSTOR_WGSL}\n{TREE_IMPOSTOR_WGSL}"), &uniform, textured_bindings(), options);
        m.set_bindable(8, impostor.albedo_texture());
        m.set_bindable(9, impostor.normal_depth_texture());
        m.set_bindable(10, Sampler::new(wgpu::FilterMode::Linear, wgpu::FilterMode::Linear).with_address_mode(wgpu::AddressMode::ClampToEdge));
        let mut r = Renderable::new(InstancedGeometry::new(billboard_geometry(label), self.count as u32, vec![culled_instances(&self.source)]), m);
        // never in the GI's voxels nor the grid: the coarsest LOD stands for the far trees there
        r.instance_culling = Some(tree_culling(&self.source, self.count, self.bounds).with_gi_lod_range(0.0, 0.0));
        r.cast_shadow = true;
        r.layers = Renderable::DEFAULT_LAYERS | TREE_LAYER;
        self.impostor = Some(scene.add(SceneNode::Renderable(r)));
    }

    /// The film's LOD bands for this frame's lens: a tree of the layer's median height changes LOD
    /// where its height on screen crosses `LOD_THRESHOLDS` and becomes the impostor below
    /// `IMPOSTOR_THRESHOLD` (the spruces' cards reach `CARD_IMPOSTOR_THRESHOLD`), `camera` times
    /// nearer; the shadow maps `shadow` times nearer. The GI's voxels take fixed bands
    /// (`GI_BANDS`), the ray tracing grid LOD0 and the cards.
    fn update(&self, scene: &mut Scene, tan_half_vfov: f32, camera: f32, shadow: f32) {
        let d = LOD_THRESHOLDS.map(|t| self.height / (2.0 * t * camera * tan_half_vfov));
        let far = if self.impostor.is_some() { self.height / (2.0 * IMPOSTOR_THRESHOLD * camera * tan_half_vfov) } else { f32::INFINITY };
        let view = [(0.0, d[0]), (d[0], d[1]), (d[1], far)];
        let sd = [d[0] * camera / shadow, d[1] * camera / shadow, far * camera / shadow];
        let shadowed = [(0.0, sd[0]), (sd[0], sd[1]), (sd[1], sd[2])];
        let crossfade = 2.0 * LOD_FADE * d[0];
        // with the cards: they reach `hand`, then the impostor; no LOD2, and the bark stops there
        // too (the impostor is baked from both)
        let hand = self.cards.and(self.impostor).map(|_| (self.height / (2.0 * CARD_IMPOSTOR_THRESHOLD * camera * tan_half_vfov)).min(far));
        let clip = |band: (f32, f32), at: f32| if band.0 >= at { NOWHERE } else { (band.0, band.1.min(at)) };
        let view = match hand { Some(h) => view.map(|b| clip(b, h)), None => view };
        let shadow_hand = hand.map(|h| h * camera / shadow);
        let shadowed = match shadow_hand { Some(h) => shadowed.map(|b| clip(b, h)), None => shadowed };
        let set = |scene: &mut Scene, idx: usize, band: (f32, f32), shadowed: (f32, f32), gi: (f32, f32)| {
            if let Some(culling) = scene.get_renderable_mut(idx).and_then(|r| r.instance_culling.as_mut()) {
                culling.crossfade = crossfade;
                culling.lod_range = band;
                culling.shadow_lod_range = Some(shadowed);
                culling.gi_lod_range = Some(gi);
                culling.rt_lod_range = Some((0.0, f32::INFINITY));
            }
        };
        for (lod, &(bark, foliage)) in self.nodes.iter().enumerate() {
            set(scene, bark, view[lod], shadowed[lod], GI_BANDS[lod]);
            // with the cards, no view draws the spruces' foliage LODs 0 and 1 (nor 2 below an
            // impostor): the cards reach it, in the voxels too
            let carded = self.cards.is_some() && (lod < 2 || hand.is_some());
            let gi = if self.cards.is_some() && lod == 0 { NOWHERE } else { GI_BANDS[lod] };
            set(scene, foliage, if carded { NOWHERE } else { view[lod] }, if carded { NOWHERE } else { shadowed[lod] }, gi);
        }
        if let Some(idx) = self.cards {
            // up to the impostor (or LOD2), crossfading into it; in the voxels' finest band
            set(scene, idx, (0.0, hand.unwrap_or(d[1])), (0.0, shadow_hand.unwrap_or(sd[1])), GI_BANDS[0]);
        }
        if let Some(idx) = self.impostor {
            set(scene, idx, (hand.unwrap_or(far), f32::INFINITY), (shadow_hand.unwrap_or(sd[2]), f32::INFINITY), NOWHERE);
        }
    }
}

fn bytemuck_floats(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}
