//! The Raggare intro's scene data, fetched at load from where raggare.kansei.graphics serves it
//! (`data=<base URL>` for another copy): its seed export (scene.json, the terrain's heights and
//! splat, the road's centre line, the trees' and the ground cover's instances) and the CC0 ground
//! and asphalt scans (KTX2). Nothing of it is in this repository; see the README for the licences.

use serde::Deserialize;
use wasm_bindgen::JsValue;

/// Where raggare.kansei.graphics serves the intro (CORS open).
pub const DEFAULT_BASE: &str = "https://raggare.kansei.graphics/";

/// The road's scanned surfaces (`textures/road/<name>_{color,nra}.ktx2`), in the order road.wgsl
/// binds them: worn base, weathered variation, repair patches, gritty edges.
pub const ROAD_SURFACES: [&str; 4] = ["base", "var", "patch", "edge"];

#[derive(Debug, Deserialize)]
pub struct SceneJson {
    pub terrain: TerrainMeta,
    pub road: RoadMeta,
    pub instances: Instances,
    pub film: Film,
}

#[derive(Debug, Deserialize)]
pub struct TerrainMeta {
    pub samples: [usize; 2],
    pub origin: [f32; 2],
    pub spacing: f32,
    pub height_offset: f32,
    pub height_scale: f32,
    pub splat: SplatMeta,
    pub lake: Lake,
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct SplatMeta {
    pub samples: [usize; 2],
    pub spacing: f32,
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct Lake {
    pub level: f32,
    pub outline: Vec<[f32; 2]>,
}

#[derive(Debug, Deserialize)]
pub struct RoadMeta {
    pub stride: usize,
    pub spec: RoadSpec,
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct RoadSpec {
    pub width: f32,
    pub shoulder: f32,
}

#[derive(Debug, Deserialize)]
pub struct Instances {
    pub trees: InstanceFile,
    pub grass: InstanceFile,
    pub flowers: InstanceFile,
    pub bushes: InstanceFile,
}

#[derive(Debug, Deserialize)]
pub struct InstanceFile {
    pub file: String,
}

#[derive(Debug, Deserialize)]
pub struct Film {
    pub filmback_mm: [f32; 2],
    pub shots: Vec<Shot>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Shot {
    pub id: String,
    pub start: f32,
    pub duration: f32,
    pub lens_mm: f32,
    /// Keys of [t (shot-local), x, y, z, bearing, pitch].
    pub camera: Vec<[f32; 6]>,
}

/// Everything the forest is built from, as fetched.
pub struct SceneData {
    pub scene: SceneJson,
    /// Metres, row-major along +Z (`scene.terrain.samples`).
    pub heights: Vec<f32>,
    pub splat: Vec<u8>,
    /// `[s, X, Y, Z, tX, tZ]` rows.
    pub road: Vec<f32>,
    pub trees: Vec<[f32; 8]>,
    pub grass: Vec<[f32; 8]>,
    pub flowers: Vec<[f32; 8]>,
    pub bushes: Vec<[f32; 8]>,
    pub ground_color: Vec<u8>,
    pub ground_nra: Vec<u8>,
    /// (colour, nra) per `ROAD_SURFACES`.
    pub road_textures: Vec<(Vec<u8>, Vec<u8>)>,
    /// Bytes fetched.
    pub bytes: usize,
}

fn records(bytes: &[u8]) -> Vec<[f32; 8]> {
    bytes
        .chunks_exact(32)
        .map(|r| std::array::from_fn(|i| f32::from_le_bytes([r[4 * i], r[4 * i + 1], r[4 * i + 2], r[4 * i + 3]])))
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

/// Fetch the scene from `base` (ending in `/`): scene.json first, then every file it names and
/// the scans, all at once.
pub async fn load(base: &str) -> Result<SceneData, JsValue> {
    let json = kansei_wasm::fetch_bytes(&format!("{base}data/scene.json")).await?;
    let scene: SceneJson = serde_json::from_slice(&json).map_err(|e| JsValue::from_str(&format!("scene.json: {e}")))?;
    let mut urls: Vec<String> = [&scene.terrain.file, &scene.terrain.splat.file, &scene.road.file, &scene.instances.trees.file, &scene.instances.grass.file, &scene.instances.flowers.file, &scene.instances.bushes.file]
        .iter()
        .map(|f| format!("{base}data/{f}"))
        .collect();
    urls.push(format!("{base}textures/ground/color.ktx2"));
    urls.push(format!("{base}textures/ground/nra.ktx2"));
    for s in ROAD_SURFACES {
        urls.push(format!("{base}textures/road/{s}_color.ktx2"));
        urls.push(format!("{base}textures/road/{s}_nra.ktx2"));
    }
    let files = futures_util::future::join_all(urls.iter().map(|u| kansei_wasm::fetch_bytes(u))).await.into_iter().collect::<Result<Vec<_>, _>>()?;
    let bytes = json.len() + files.iter().map(Vec::len).sum::<usize>();
    let mut files = files.into_iter();
    let mut next = || files.next().unwrap();
    let raw = next();
    let [nx, nz] = scene.terrain.samples;
    if raw.len() != nx * nz * 2 {
        return Err(JsValue::from_str(&format!("height.u16 is {} bytes, expected {}", raw.len(), nx * nz * 2)));
    }
    let heights = raw.chunks_exact(2).map(|b| (u16::from_le_bytes([b[0], b[1]]) as f32 - scene.terrain.height_offset) * scene.terrain.height_scale).collect();
    let splat = next();
    let road = floats(&next());
    let trees = records(&next());
    let grass = records(&next());
    let flowers = records(&next());
    let bushes = records(&next());
    let ground_color = next();
    let ground_nra = next();
    let road_textures = ROAD_SURFACES.iter().map(|_| (next(), next())).collect();
    Ok(SceneData { scene, heights, splat, road, trees, grass, flowers, bushes, ground_color, ground_nra, road_textures, bytes })
}

/// The heights as a grid: (x, z) in metres to the rendered terrain's height.
pub struct Heightfield<'a> {
    pub nx: usize,
    pub nz: usize,
    pub origin: [f32; 2],
    pub spacing: f32,
    pub heights: &'a [f32],
}

impl<'a> Heightfield<'a> {
    pub fn new(data: &'a SceneData) -> Self {
        let t = &data.scene.terrain;
        Self { nx: t.samples[0], nz: t.samples[1], origin: t.origin, spacing: t.spacing, heights: &data.heights }
    }

    pub fn at(&self, ix: usize, iz: usize) -> f32 {
        self.heights[iz.min(self.nz - 1) * self.nx + ix.min(self.nx - 1)]
    }

    /// Height of the terrain mesh drawn every `step` samples (triangulated as the tiles are) at
    /// (x, z): what things standing on the ground must sit on.
    pub fn mesh_height(&self, x: f32, z: f32, step: usize) -> f32 {
        let cell = self.spacing * step as f32;
        let fx = ((x - self.origin[0]) / cell).max(0.0);
        let fz = ((z - self.origin[1]) / cell).max(0.0);
        let cols = (self.nx - 1) / step;
        let rows = (self.nz - 1) / step;
        let (c, r) = ((fx.floor() as usize).min(cols.saturating_sub(1)), (fz.floor() as usize).min(rows.saturating_sub(1)));
        let (tx, tz) = ((fx - c as f32).clamp(0.0, 1.0), (fz - r as f32).clamp(0.0, 1.0));
        let h = |cc: usize, rr: usize| self.at(cc * step, rr * step);
        let (h00, h10, h01, h11) = (h(c, r), h(c + 1, r), h(c, r + 1), h(c + 1, r + 1));
        // the quad's two triangles: (00, 01, 10) and (10, 01, 11), split along the 10-01 diagonal
        if tx + tz <= 1.0 {
            h00 + (h10 - h00) * tx + (h01 - h00) * tz
        } else {
            h11 + (h01 - h11) * (1.0 - tx) + (h10 - h11) * (1.0 - tz)
        }
    }

    /// The normal at a sample, from its neighbours `reach` samples away.
    pub fn normal(&self, ix: usize, iz: usize, reach: usize) -> glam::Vec3 {
        let (x0, x1) = (ix.saturating_sub(reach), (ix + reach).min(self.nx - 1));
        let (z0, z1) = (iz.saturating_sub(reach), (iz + reach).min(self.nz - 1));
        let dx = (self.at(x1, iz) - self.at(x0, iz)) / ((x1 - x0) as f32 * self.spacing);
        let dz = (self.at(ix, z1) - self.at(ix, z0)) / ((z1 - z0) as f32 * self.spacing);
        glam::Vec3::new(-dx, 1.0, -dz).normalize()
    }
}
