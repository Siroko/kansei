//! `RtCausticsEffect`: the light glass focuses (and the shadow it leaves), traced from the lights
//! through the ray tracing grid into a tile of texels round each glass caster.

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, Vec3, Vec4};

use super::{RtBackend, RtGridHandle, RT_OPAQUE_WGSL};
use crate::cameras::Camera;
use crate::lights::Light;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

/// The most glass casters one `RtCausticsEffect` keeps a tile for (`set_casters`).
pub const RT_CAUSTICS_MAX_CASTERS: usize = 8;

/// The caustics map read at a surface (`kansei_rt_caustics_lookup`, the uniform
/// `KanseiRtCaustics`), for passes that light surfaces themselves: bind `RtCausticsEffect::map`.
pub const RT_CAUSTICS_LOOKUP_WGSL: &str = include_str!("shaders/rt_caustics_lookup.wgsl");

/// The photons' WGSL, through a grid traced by `backend` (its bindings in group 1).
pub(crate) fn photons_wgsl(backend: RtBackend) -> String {
    super::hoist_wgsl_enables(&format!(
        "{}\n{}\n{RT_OPAQUE_WGSL}\n{RT_CAUSTICS_LOOKUP_WGSL}\n{}",
        backend.wgsl(),
        backend.bindings_wgsl(1, 0),
        include_str!("shaders/rt_caustics.wgsl")
    ))
}

pub(crate) fn resolve_wgsl() -> String {
    format!("{RT_CAUSTICS_LOOKUP_WGSL}\n{}", include_str!("shaders/rt_caustics_resolve.wgsl"))
}

/// The photons' bind group layouts (groups 0 and 1), through a grid traced by `backend`.
pub(crate) fn photon_layout_entries(backend: RtBackend) -> [Vec<wgpu::BindGroupLayoutEntry>; 2] {
    let compute = wgpu::ShaderStages::COMPUTE;
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
    let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
    let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
    [vec![entry(0, uniform), entry(1, storage(true)), entry(2, storage(false)), entry(3, uniform)], backend.layout_entries(0, compute).to_vec()]
}

/// What `RtCausticsEffect::new` sets up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtCausticsOptions {
    /// The finest texel (m). A tile's texel is about a pixel's width where its caster is on
    /// screen (in steps of two from this), so its size and its photons follow the glass's
    /// footprint on screen, not the scene's.
    pub texel: f32,
    /// A tile's most texels a side (a layer of the map): a caster's tile coarsens its texel to
    /// fit what its caustics and shadow cover.
    pub tile_texels: u32,
    /// Photons a light sends at a caster: this many per texel across its box (a side; 2: about
    /// four a texel where the light lands unfocused), at most `photons` a side.
    pub photons_per_texel: f32,
    pub photons: u32,
    /// Frames a still tile accumulates (each with its own jitter, averaged), after which it sends
    /// no photons until its caster or a light moves.
    pub max_history: u32,
    /// The most surfaces a photon crosses or reflects off.
    pub interfaces: u32,
    pub max_distance: f32,
    /// The widest splat, texels (a photon's tent is twice as wide as photons lie apart there).
    pub splat_texels: f32,
    /// A surface takes the map's light only within this height (m) of where the photons landed
    /// (and within half its caster's height).
    pub height_tolerance: f32,
    /// How far below its caster's box a tile reaches (m) for the lights' slant (0: half the box's
    /// height; a marble on its table needs none, glass over a floor further down more).
    pub reach: f32,
}

impl Default for RtCausticsOptions {
    fn default() -> Self {
        Self { texel: 0.0005, tile_texels: 256, photons_per_texel: 2.0, photons: 256, max_history: 64, interfaces: 8, max_distance: 60.0, splat_texels: 3.0, height_tolerance: 0.3, reach: 0.0 }
    }
}

/// The WGSL `CausticsParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
    task_count: u32,
    photons: u32,
    interfaces: u32,
    max_distance: f32,
    splat_texels: f32,
    glass_ior: f32,
    focus_gain: f32,
    focus_sharpen: f32,
    glass_tint: [f32; 3],
    tint_distance: f32,
}

/// The WGSL `CausticTask`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
struct TaskGpu {
    origin: [f32; 3],
    kind: u32,
    axis: [f32; 3],
    side: u32,
    e1: [f32; 3],
    u0: f32,
    e2: [f32; 3],
    v0: f32,
    color: [f32; 3],
    du: f32,
    emitter_x: [f32; 3],
    dv: f32,
    emitter_y: [f32; 3],
    first: u32,
    light_dir: [f32; 3],
    cos_outer: f32,
    cos_inner: f32,
    range: f32,
    tile: u32,
    seed: u32,
    light: u32,
    bias: f32,
    _pad: [u32; 2],
}

/// The WGSL `KanseiRtCausticTile`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
struct TileGpu {
    min: [f32; 2],
    size: [f32; 2],
    dims: [u32; 2],
    ref_y: f32,
    tolerance: f32,
    box_min: [f32; 3],
    live: u32,
    box_max: [f32; 3],
    blend: f32,
}

/// The WGSL `KanseiRtCaustics`: the tiles and how the map is read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct LookupGpu {
    enabled: u32,
    count: u32,
    intensity: f32,
    texels: u32,
    scale: f32,
    _pad: [u32; 3],
    tiles: [TileGpu; RT_CAUSTICS_MAX_CASTERS],
}

/// The WGSL `CompositeParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    size: [f32; 2],
    view: u32,
    receivers: u32,
    rect: [f32; 4],
}

/// The WGSL structs' sizes, for the tests.
#[cfg(test)]
pub(crate) fn struct_sizes() -> [(&'static str, usize); 5] {
    [
        ("CausticsParams", std::mem::size_of::<ParamsGpu>()),
        ("CausticTask", std::mem::size_of::<TaskGpu>()),
        ("KanseiRtCausticTile", std::mem::size_of::<TileGpu>()),
        ("KanseiRtCaustics", std::mem::size_of::<LookupGpu>()),
        ("CompositeParams", std::mem::size_of::<CompositeGpu>()),
    ]
}

/// The most light-and-caster pairs sending photons in a frame.
const MAX_TASKS: usize = 64;

/// A light as `update_lights` found it: a directional light (its direction, illuminance) or a spot
/// light (position, direction, colour times intensity (cd), range, the cone's outer and inner
/// half-angles, its emitter's radius).
#[derive(Clone, Copy, Debug, PartialEq)]
enum CausticLight {
    Sun { direction: Vec3, color: Vec3 },
    Spot { position: Vec3, direction: Vec3, color: Vec3, range: f32, cone: (f32, f32), radius: f32, rect: Option<[f32; 2]> },
}

/// What the screen shows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtCausticsView {
    /// The lit image with the caustics.
    #[default]
    Lit,
    /// The light the caustics add (brighter) and take (darker) over grey.
    Caustics,
}

/// The surfaces the caustics light.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RtCausticReceivers {
    /// Those `RtShadowsEffect` lights (`kansei_gbuffer_out_rt_lit`), whose shadow rays pass through
    /// glass: the map's change to the direct light (the light focused, minus the light that
    /// passed straight through) makes their direct light the glass's, counted once.
    #[default]
    RtLit,
    /// Every upward-facing surface, for a scene whose direct light passes through the glass by
    /// other means (no shadow map holds it). Where a shadow map does, the light passing straight
    /// through would be taken away twice.
    All,
}

/// A caster's tile, as last placed.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Tile {
    /// the caster's box
    lo: Vec3,
    hi: Vec3,
    /// the texel's level (texel = options.texel * 2^level), the rectangle's corner (world xz, a
    /// multiple of the texel) and its texels
    level: u32,
    min: [f32; 2],
    dims: [u32; 2],
    /// frames of photons blended in since it was last cleared
    history: u32,
    /// on screen: read and fed
    live: bool,
}

impl Tile {
    fn texel(&self, finest: f32) -> f32 {
        finest * (1u32 << self.level) as f32
    }
}

/// The xz rectangle (min, max) a caster's caustics and shadow may land on, under `lights`: its box
/// and its corners projected from each light onto the planes at its bottom and `reach` below it,
/// at most `spread` times its widest side out from its centre.
fn footprint(lo: Vec3, hi: Vec3, lights: &[CausticLight], reach: f32) -> ([f32; 2], [f32; 2]) {
    let (mut a, mut b) = ([lo.x, lo.z], [hi.x, hi.z]);
    let centre = (lo + hi) * 0.5;
    let widest = (hi.x - lo.x).max(hi.z - lo.z).max(hi.y - lo.y).max(1e-4);
    let spread = 8.0 * widest;
    let mut grow = |p: Vec3| {
        let x = p.x.clamp(centre.x - spread, centre.x + spread);
        let z = p.z.clamp(centre.z - spread, centre.z + spread);
        (a, b) = ([a[0].min(x), a[1].min(z)], [b[0].max(x), b[1].max(z)]);
    };
    for k in 0..8 {
        let q = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
        for plane in [lo.y, lo.y - reach] {
            for light in lights {
                match *light {
                    CausticLight::Sun { direction: d, .. } => {
                        if d.y < -1e-3 {
                            grow(q + d * ((plane - q.y) / d.y));
                        }
                    }
                    CausticLight::Spot { position: p, .. } => {
                        if p.y > q.y + 1e-4 {
                            grow(p + (q - p) * ((plane - p.y) / (q.y - p.y)));
                        }
                    }
                }
            }
        }
    }
    (a, b)
}

/// Whether any of `points` may be on screen (not all beyond one side of the clip volume).
fn on_screen(view_proj: Mat4, points: &[Vec3]) -> bool {
    let clip: Vec<Vec4> = points.iter().map(|p| view_proj * p.extend(1.0)).collect();
    let outside = |f: &dyn Fn(Vec4) -> bool| clip.iter().all(|&c| f(c));
    !(outside(&|c| c.x < -c.w) || outside(&|c| c.x > c.w) || outside(&|c| c.y < -c.w) || outside(&|c| c.y > c.w) || outside(&|c| c.z > c.w) || outside(&|c| c.w <= 0.0))
}

/// The pixels (min x, y, max x, y) the live tiles may cover: their rectangles at their height,
/// within their tolerance, and their casters' boxes, projected; the whole screen when one reaches
/// behind the eye, none when there are none.
fn screen_rect(view_proj: Mat4, lookup: &LookupGpu, size: [f32; 2]) -> [f32; 4] {
    let mut rect = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
    for t in lookup.tiles.iter().filter(|t| t.live != 0) {
        let (lo, hi) = (Vec3::new(t.min[0], (t.ref_y - t.tolerance).min(t.box_min[1]), t.min[1]), Vec3::new(t.min[0] + t.size[0], (t.ref_y + t.tolerance).max(t.box_max[1]), t.min[1] + t.size[1]));
        for k in 0..8 {
            let p = Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z });
            let c = view_proj * p.extend(1.0);
            if c.w <= 1e-4 {
                return [0.0, 0.0, size[0], size[1]];
            }
            let (x, y) = ((c.x / c.w * 0.5 + 0.5) * size[0], (0.5 - c.y / c.w * 0.5) * size[1]);
            rect = [rect[0].min(x), rect[1].min(y), rect[2].max(x), rect[3].max(y)];
        }
    }
    // a pixel's margin for the sampler
    [rect[0] - 1.0, rect[1] - 1.0, rect[2] + 1.0, rect[3] + 1.0]
}

/// The texel level a tile wants: a pixel's width at its caster (`pixel`, m) but no finer than
/// `finest`, coarse enough that `extent` (m) fits `texels`; kept at `previous` unless the pixel
/// left it by more than a quarter step, so a camera moving a little doesn't clear it.
fn texel_level(finest: f32, pixel: f32, extent: f32, texels: u32, previous: Option<u32>) -> u32 {
    let level = |x: f32| (x / finest).max(1.0).log2();
    let fit = level(extent / (texels.max(8) - 4) as f32).ceil() as u32;
    let want = level(pixel);
    let chosen = match previous {
        Some(p) if want > p as f32 - 1.25 && want <= p as f32 + 0.25 => p,
        _ => want.ceil() as u32,
    };
    chosen.max(fit).min(20)
}

/// Where the casters' tiles lie and what they hold, frame to frame.
#[derive(Clone, Debug, Default)]
struct Planner {
    /// the lights as the tiles were last fed: a change clears them
    fed_lights: Vec<CausticLight>,
    tiles: Vec<Option<Tile>>,
    frame: u32,
}

impl Planner {
    /// Place the casters' tiles for this frame's camera and lights, clearing those whose caster,
    /// lights or placement changed; the tiles that take photons this frame and their blend.
    fn plan(&mut self, options: &RtCausticsOptions, casters: &[(Vec3, Vec3)], lights: &[CausticLight], interval: u32, view_proj: Mat4, eye: Vec3, pixel_scale: f32) -> Vec<(usize, f32)> {
        let lights_changed = lights != self.fed_lights;
        self.fed_lights = lights.to_vec();
        let finest = options.texel;
        let texels = options.tile_texels;
        // casters that moved this frame, by their old and new footprints: their neighbours' tiles
        // (photons through both) clear with them
        let mut moved: Vec<([f32; 2], [f32; 2])> = Vec::new();
        for (k, &(lo, hi)) in casters.iter().enumerate() {
            if let Some(t) = self.tiles[k].filter(|t| t.lo != lo || t.hi != hi) {
                let reach = if options.reach > 0.0 { options.reach } else { 0.5 * (hi.y - lo.y) };
                moved.push(footprint(t.lo, t.hi, lights, reach));
                moved.push(footprint(lo, hi, lights, reach));
            }
        }
        let mut fed = Vec::new();
        for k in 0..RT_CAUSTICS_MAX_CASTERS {
            let Some(&(lo, hi)) = casters.get(k) else {
                self.tiles[k] = None;
                continue;
            };
            let reach = if options.reach > 0.0 { options.reach } else { 0.5 * (hi.y - lo.y) };
            let (a, b) = footprint(lo, hi, lights, reach);
            let corners: Vec<Vec3> = (0..8).map(|i| Vec3::new(if i & 1 == 0 { a[0] } else { b[0] }, if i & 2 == 0 { lo.y } else { hi.y.max(lo.y - reach) }, if i & 4 == 0 { a[1] } else { b[1] })).collect();
            let mut points = corners.clone();
            points.extend((0..4).map(|i| Vec3::new(if i & 1 == 0 { a[0] } else { b[0] }, lo.y - reach, if i & 2 == 0 { a[1] } else { b[1] })));
            if !on_screen(view_proj, &points) {
                self.tiles[k] = None;
                continue;
            }
            // a pixel's width at the caster
            let pixel = eye.distance((lo + hi) * 0.5) * pixel_scale;
            let extent = (b[0] - a[0]).max(b[1] - a[1]);
            let old = self.tiles[k];
            let level = texel_level(finest, pixel, extent, texels, old.map(|t| t.level));
            let texel = finest * (1u32 << level) as f32;
            // the splat's reach as margin
            let margin = (options.splat_texels.ceil() + 1.0) * texel;
            let contained = old.is_some_and(|t| {
                let size = [t.dims[0] as f32 * texel, t.dims[1] as f32 * texel];
                t.level == level && a[0] - margin >= t.min[0] && a[1] - margin >= t.min[1] && b[0] + margin <= t.min[0] + size[0] && b[1] + margin <= t.min[1] + size[1]
            });
            let mut tile = match old {
                Some(t) if contained => t,
                _ => {
                    // the footprint and its margin, a quarter wider for the caster to move in, on
                    // the texel's multiples
                    let pad = margin + 0.125 * extent;
                    let min = [((a[0] - pad) / texel).floor() * texel, ((a[1] - pad) / texel).floor() * texel];
                    let dims = [(((b[0] + pad - min[0]) / texel).ceil() as u32).clamp(8, texels), (((b[1] + pad - min[1]) / texel).ceil() as u32).clamp(8, texels)];
                    Tile { lo, hi, level, min, dims, history: 0, live: true }
                }
            };
            let neighbour_moved = moved.iter().any(|(ma, mb)| ma[0] <= b[0] && mb[0] >= a[0] && ma[1] <= b[1] && mb[1] >= a[1]);
            if lights_changed || neighbour_moved || tile.lo != lo || tile.hi != hi || !old.is_some_and(|t| t.live) {
                tile.history = 0;
            }
            (tile.lo, tile.hi, tile.live) = (lo, hi, true);
            let due = tile.history == 0 || self.frame.is_multiple_of(interval.max(1));
            if tile.history < options.max_history.max(1) && due {
                fed.push((k, 1.0 / (tile.history + 1) as f32));
            }
            self.tiles[k] = Some(tile);
        }
        fed
    }

}

struct Gpu {
    params: wgpu::Buffer,
    tasks: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    /// the resolve's target, copied into the map (which the composite reads)
    scratch: wgpu::Texture,
    grid_bgl: wgpu::BindGroupLayout,
    photons: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    photon_group: wgpu::BindGroup,
    resolve_group: wgpu::BindGroup,
    /// the grid's group, with its generation
    grid_group: Option<(u64, wgpu::BindGroup)>,
    /// the grid's backend the photons were made for
    backend: RtBackend,
}

/// Caustics: the light that glass (`RtSurface::glass`) focuses onto the surfaces under it, and
/// the shadow it leaves there, from the scene's directional and spot lights (`update_lights`).
///
/// Each glass caster (`set_casters`: its world box, a marble's or a dragon's, every frame it
/// moves) has a tile: a horizontal grid of texels over where its caustics and shadow land (its box
/// projected from each light onto the planes at its bottom and `reach` below), a texel about a
/// pixel wide where it is on screen (no finer than `RtCausticsOptions::texel`), placed on the
/// world's multiples of its texel. Each light sends a grid of photons at a tile's caster
/// (`photons_per_texel` across its box) through the renderer's ray tracing grid (either backend):
/// a photon whose first hit is the caster's glass is reflected or refracted at every glass surface
/// it crosses (exact dielectric Fresnel, chosen at random; Beer-Lambert inside by the triangles'
/// tint, `tint_distance`; total internal reflection) and adds its light where it lands, and the
/// same photon taken straight through no glass takes its light away where the direct light (which
/// passes through glass) lands. A tile thus holds the change the glass makes to the direct light:
/// dark where the light is bent away or absorbed, bright where it is focused, coloured by a tinted
/// core; in sum the light the glass reflects out of reach and absorbs, nothing more. It lights the
/// surfaces that face up at the height the photons landed (`RtCausticReceivers`).
///
/// A still caster's tile accumulates `max_history` frames of photons, each with its own jitter,
/// then sends none. One that moves, or whose lights move, is cleared and fed again every frame
/// with the same jitter: its photons keep their pattern round the glass as it goes, so the
/// caustics move with it without crawling. Tiles off screen take no photons.
///
/// Put it after `RtShadowsEffect` (and the GI), whose direct light it corrects; its `intensity`
/// matches the shadows'. A photon costs a ray to its caster and, when it hits the glass, one
/// straight on and one a surface it crosses; directional lights cost one more, back toward the
/// light.
pub struct RtCausticsEffect {
    enabled: bool,
    pub view: RtCausticsView,
    pub receivers: RtCausticReceivers,
    /// Scale of the light the caustics add and take (1 is physical; as `RtShadowsEffect`'s).
    pub intensity: f32,
    /// Frames between a tile's photons while it accumulates (1: every frame).
    pub interval: u32,
    /// The emitters' size (a spot light's disk or `set_rect_emitter`'s rectangle, the sun's
    /// `sun_angle`) blurs where the photons land by the penumbra it casts past the glass, as
    /// `RtShadowsEffect`'s rays reaching all of it soften its shadows (the photons themselves
    /// leave its centre: an even grid, smooth from the first frame); off, a point's crisp
    /// caustics.
    pub area_emitters: bool,
    /// The sun's angular radius (radians): `RtShadowsEffect::sun_angle`.
    pub sun_angle: f32,
    /// The light the glass brings (its photons through it) times this, and their splats this
    /// much narrower (1 and 1: physical; the shadow the glass leaves stays as it is): a game's
    /// caustics, bright, crisp lines and spots where a physical glass's light spreads faint.
    pub focus_gain: f32,
    pub focus_sharpen: f32,
    /// Metres the glass triangles' tints (`RtSurface::glass`, times `set_glass`'s) are the light
    /// left after: 1, as the glass's own (`kansei_gbuffer_out_glass`); a centimetre for a marble's
    /// coloured core to colour its light.
    pub tint_distance: f32,
    options: RtCausticsOptions,
    grid: RtGridHandle,
    casters: Vec<(Vec3, Vec3)>,
    glass: (f32, [f32; 3]),
    lights: Vec<CausticLight>,
    rects: Vec<(usize, [f32; 2])>,
    planner: Planner,
    sent: (usize, u32, usize),
    map: wgpu::Texture,
    map_view: wgpu::TextureView,
    lookup: wgpu::Buffer,
    sampler: wgpu::Sampler,
    gpu: Option<Gpu>,
}

impl RtCausticsEffect {
    /// Caustics traced through `grid` (`SceneRtGrid::handle`).
    pub fn new(device: &wgpu::Device, grid: RtGridHandle, options: RtCausticsOptions) -> Self {
        let texels = options.tile_texels.clamp(16, 2048).next_multiple_of(8);
        let options = RtCausticsOptions { tile_texels: texels, texel: options.texel.max(1e-5), ..options };
        let map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RtCaustics/Map"),
            size: wgpu::Extent3d { width: texels, height: texels, depth_or_array_layers: RT_CAUSTICS_MAX_CASTERS as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let map_view = map.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let lookup = device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtCaustics/Lookup"), size: std::mem::size_of::<LookupGpu>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("RtCaustics/Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        Self {
            enabled: true,
            view: RtCausticsView::Lit,
            receivers: RtCausticReceivers::RtLit,
            intensity: 1.0,
            interval: 1,
            area_emitters: true,
            sun_angle: 0.02,
            focus_gain: 1.0,
            focus_sharpen: 1.0,
            tint_distance: 1.0,
            options,
            grid,
            casters: Vec::new(),
            glass: (1.5, [1.0; 3]),
            lights: Vec::new(),
            rects: Vec::new(),
            planner: Planner { tiles: vec![None; RT_CAUSTICS_MAX_CASTERS], ..Default::default() },
            sent: (0, 0, 0),
            map,
            map_view,
            lookup,
            sampler,
            gpu: None,
        }
    }

    /// The boxes (world min, max) of the glass casters, at most [`RT_CAUSTICS_MAX_CASTERS`], each
    /// frame they may move: the `k`th keeps the `k`th tile, cleared when its box moves.
    pub fn set_casters(&mut self, boxes: &[(Vec3, Vec3)]) {
        self.casters = boxes.iter().take(RT_CAUSTICS_MAX_CASTERS).copied().collect();
    }

    /// The glass's index of refraction (every glass surface's) and a tint multiplying every glass
    /// triangle's own (`RtSurface::glass`): the light left after `tint_distance` inside.
    pub fn set_glass(&mut self, ior: f32, tint: [f32; 3]) {
        if (ior, tint) != self.glass {
            self.glass = (ior, tint);
            self.clear();
        }
    }

    /// The `index`th spot light (in `update_lights`' order, counting spot lights only) emits from a
    /// rectangle `width` x `height` (m) facing along its axis (`RtShadowsEffect::set_rect_emitter`'s
    /// axes), not from its disk.
    pub fn set_rect_emitter(&mut self, index: usize, width: f32, height: f32) {
        self.rects.retain(|r| r.0 != index);
        self.rects.push((index, [width, height]));
    }

    /// The scene's lights (`Scene::lights`), each frame they may change: its directional and spot
    /// lights send photons. A change clears the tiles.
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        self.lights.clear();
        let mut spots = 0;
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let c = l.effective_color();
                    self.lights.push(CausticLight::Sun { direction: Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or(Vec3::NEG_Y), color: Vec3::new(c.x, c.y, c.z) });
                }
                Light::Spot(l) => {
                    let c = l.effective_color();
                    self.lights.push(CausticLight::Spot {
                        position: Vec3::new(l.position.x, l.position.y, l.position.z),
                        direction: Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or(Vec3::NEG_Z),
                        color: Vec3::new(c.x, c.y, c.z),
                        range: l.range.max(1e-3),
                        cone: l.cone(),
                        radius: l.source_radius.max(0.0),
                        rect: self.rects.iter().find(|r| r.0 == spots).map(|r| r.1),
                    });
                    spots += 1;
                }
                Light::Point(_) | Light::Area(_) => {}
            }
        }
    }

    /// Clear every tile: their photons start over.
    pub fn clear(&mut self) {
        for tile in self.planner.tiles.iter_mut().flatten() {
            tile.history = 0;
        }
    }

    /// Turn it on or off (`is_active`). Off, what reads its map (`map`'s lookup) sees no caustics
    /// either: an inactive effect isn't rendered, so the lookup is cleared here.
    pub fn set_enabled(&mut self, queue: &wgpu::Queue, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            queue.write_buffer(&self.lookup, 0, bytemuck::bytes_of(&LookupGpu::default()));
            self.planner.tiles = vec![None; RT_CAUSTICS_MAX_CASTERS];
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    pub fn options(&self) -> &RtCausticsOptions {
        &self.options
    }

    /// The map (a layer a tile; rgb: the irradiance the caustics add, lux, signed; a: the height
    /// the photons landed at over the tile's) and its uniform (`KanseiRtCaustics`), for
    /// `RT_CAUSTICS_LOOKUP_WGSL`.
    pub fn map(&self) -> (&wgpu::TextureView, &wgpu::Buffer) {
        (&self.map_view, &self.lookup)
    }

    /// The map's texture (for reading it back).
    pub fn map_texture(&self) -> &wgpu::Texture {
        &self.map
    }

    /// The `k`th caster's tile as last rendered: its corner (world xz), texel (m), texels a side,
    /// and the frames of photons it holds; `None` when it has none (off screen, no caster).
    pub fn tile(&self, k: usize) -> Option<([f32; 2], f32, [u32; 2], u32)> {
        self.planner.tiles.get(k).copied().flatten().filter(|t| t.live).map(|t| (t.min, t.texel(self.options.texel), t.dims, t.history))
    }

    /// Last frame's light-and-caster pairs sending photons, the photons they sent, and the tiles
    /// that took them.
    pub fn photons_sent(&self) -> (usize, u32, usize) {
        self.sent
    }

    /// The lights' photon grids at the tiles fed this frame.
    fn tasks(&self, fed: &[(usize, f32)]) -> Vec<TaskGpu> {
        let mut tasks = Vec::new();
        let basis = |a: Vec3| {
            let any = if a.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let u = a.cross(any).normalize();
            (u, a.cross(u).normalize())
        };
        for &(k, _) in fed {
            let tile = self.planner.tiles[k].unwrap();
            let (lo, hi) = (tile.lo, tile.hi);
            let corners: Vec<Vec3> = (0..8).map(|i| Vec3::new(if i & 1 == 0 { lo.x } else { hi.x }, if i & 2 == 0 { lo.y } else { hi.y }, if i & 4 == 0 { lo.z } else { hi.z })).collect();
            let texel = tile.texel(self.options.texel);
            let bias = (1e-3 * (hi - lo).length()).max(1e-5);
            let side_for = |across: f32| ((across / texel * self.options.photons_per_texel).ceil() as u32).clamp(8, self.options.photons.clamp(8, 1024));
            for (index, light) in self.lights.iter().enumerate() {
                match *light {
                    CausticLight::Sun { direction: dir, color } => {
                        if color.max_element() <= 0.0 {
                            continue;
                        }
                        let (e1, e2) = basis(dir);
                        let (mut u, mut v, mut t) = ([f32::MAX, f32::MIN], [f32::MAX, f32::MIN], f32::MAX);
                        for c in &corners {
                            let (a, b) = (c.dot(e1), c.dot(e2));
                            u = [u[0].min(a), u[1].max(a)];
                            v = [v[0].min(b), v[1].max(b)];
                            t = t.min(c.dot(dir));
                        }
                        let side = side_for((u[1] - u[0]).max(v[1] - v[0]));
                        // the photons' plane just outside the box, toward the light
                        let origin = dir * (t - 0.05 * (hi - lo).length() - 1e-3);
                        let blur = if self.area_emitters { self.sun_angle.max(0.0).tan() } else { 0.0 };
                        tasks.push(TaskGpu {
                            origin: origin.to_array(),
                            kind: 0,
                            axis: dir.to_array(),
                            side,
                            e1: e1.to_array(),
                            u0: u[0],
                            e2: e2.to_array(),
                            v0: v[0],
                            color: color.to_array(),
                            du: (u[1] - u[0]) / side as f32,
                            emitter_x: (e1 * blur).to_array(),
                            dv: (v[1] - v[0]) / side as f32,
                            tile: k as u32,
                            seed: tile.history,
                            light: index as u32,
                            bias,
                            ..Default::default()
                        });
                    }
                    CausticLight::Spot { position: p, direction: d, color: c, range, cone: (outer, inner), radius, rect } => {
                        if c.max_element() <= 0.0 {
                            continue;
                        }
                        // aimed at the box's centre, the box's corners as tangents round that axis
                        let axis = ((lo + hi) * 0.5 - p).normalize_or(d);
                        let distance = ((lo + hi) * 0.5 - p).length();
                        if distance > range {
                            continue;
                        }
                        let (e1, e2) = basis(axis);
                        let (mut u, mut v) = ([f32::MAX, f32::MIN], [f32::MAX, f32::MIN]);
                        let mut behind = false;
                        for q in &corners {
                            let w = *q - p;
                            let z = w.dot(axis);
                            if z < 1e-3 {
                                behind = true;
                                break;
                            }
                            u = [u[0].min(w.dot(e1) / z), u[1].max(w.dot(e1) / z)];
                            v = [v[0].min(w.dot(e2) / z), v[1].max(w.dot(e2) / z)];
                        }
                        // the cone (a square round its axis, in the same tangents)
                        let dz = d.dot(axis);
                        if dz <= 0.05 {
                            continue;
                        }
                        let (cu, cv, r) = (d.dot(e1) / dz, d.dot(e2) / dz, outer.tan() / dz);
                        if behind {
                            (u, v) = ([cu - r, cu + r], [cv - r, cv + r]);
                        }
                        let u = [u[0].max(cu - r), u[1].min(cu + r)];
                        let v = [v[0].max(cv - r), v[1].min(cv + r)];
                        if u[0] >= u[1] || v[0] >= v[1] {
                            continue;
                        }
                        let side = side_for((u[1] - u[0]).max(v[1] - v[0]) * distance);
                        // its emitter: a rectangle (`set_rect_emitter`) or its disk, as a square
                        let (a, b) = basis(d);
                        let (ex, ey) = match (self.area_emitters, rect) {
                            (true, Some([w, h])) => (a * w * 0.5, b * h * 0.5),
                            (true, None) => (a * radius, b * radius),
                            (false, _) => (Vec3::ZERO, Vec3::ZERO),
                        };
                        tasks.push(TaskGpu {
                            origin: p.to_array(),
                            kind: 1,
                            axis: axis.to_array(),
                            side,
                            e1: e1.to_array(),
                            u0: u[0],
                            e2: e2.to_array(),
                            v0: v[0],
                            color: c.to_array(),
                            du: (u[1] - u[0]) / side as f32,
                            emitter_x: ex.to_array(),
                            dv: (v[1] - v[0]) / side as f32,
                            emitter_y: ey.to_array(),
                            light_dir: d.to_array(),
                            cos_outer: outer.cos(),
                            cos_inner: inner.cos(),
                            range,
                            tile: k as u32,
                            seed: tile.history,
                            light: index as u32,
                            bias,
                            ..Default::default()
                        });
                    }
                }
            }
        }
        tasks.truncate(MAX_TASKS);
        let mut first = 0;
        for t in &mut tasks {
            t.first = first;
            first += t.side * t.side;
        }
        tasks
    }

    /// The fixed point the photons add irradiance in: about 2^20 steps to the brightest direct
    /// light at a fed caster (focus up to two thousand times it fits an i32), a power of two.
    fn fixed_point(&self, tasks: &[TaskGpu]) -> f32 {
        let brightest = tasks
            .iter()
            .map(|t| {
                let c = Vec3::from(t.color).max_element();
                if t.kind == 0 {
                    c
                } else {
                    let tile = self.planner.tiles[t.tile as usize].unwrap();
                    c / (Vec3::from(t.origin) - (tile.lo + tile.hi) * 0.5).length_squared().max(1e-4)
                }
            })
            .fold(1e-6f32, f32::max);
        2f32.powi(((1u32 << 20) as f32 / brightest).log2().floor().clamp(-8.0, 30.0) as i32)
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let (d2, array) = (wgpu::TextureViewDimension::D2, wgpu::TextureViewDimension::D2Array);
        let texture = |filterable, view_dimension| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension, multisampled: false };
        let storage_tex = |view_dimension| wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension };
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let backend = self.grid.backend();
        let [photon_entries, grid_entries] = photon_layout_entries(backend);
        let photon_bgl = layout("RtCaustics/Photons", &photon_entries);
        let grid_bgl = layout("RtCaustics/Grid", &grid_entries);
        let resolve_bgl = layout("RtCaustics/Resolve", &[entry(0, uniform), entry(1, storage(false)), entry(2, texture(false, array)), entry(3, storage_tex(array))]);
        let depth = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false };
        let composite_bgl = layout(
            "RtCaustics/Composite",
            &[
                entry(0, uniform),
                entry(1, depth),
                entry(2, texture(false, d2)),
                entry(3, texture(false, d2)),
                entry(4, texture(false, d2)),
                entry(5, texture(false, d2)),
                entry(6, storage_tex(d2)),
                entry(7, texture(true, array)),
                entry(8, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(9, uniform),
            ],
        );
        let pipeline = |label: &str, code: &str, entry_point: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &layouts.iter().copied().map(Some).collect::<Vec<_>>(), immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let photons = pipeline("RtCaustics/Photons", &photons_wgsl(backend), "photons", &[&photon_bgl, &grid_bgl]);
        let resolve_code = resolve_wgsl();
        let resolve = pipeline("RtCaustics/Resolve", &resolve_code, "resolve", &[&resolve_bgl]);
        let composite = pipeline("RtCaustics/Composite", &resolve_code, "composite", &[&composite_bgl]);
        let buffer = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let texels = self.options.tile_texels;
        let layers = RT_CAUSTICS_MAX_CASTERS as u32;
        let accumulator = buffer("RtCaustics/Accumulator", (texels * texels * layers) as u64 * 5 * 4, wgpu::BufferUsages::STORAGE);
        let params = buffer("RtCaustics/Params", std::mem::size_of::<ParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let tasks = buffer("RtCaustics/Tasks", (std::mem::size_of::<TaskGpu>() * MAX_TASKS) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST);
        let scratch = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RtCaustics/Scratch"),
            size: wgpu::Extent3d { width: texels, height: texels, depth_or_array_layers: layers },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scratch_view = scratch.create_view(&wgpu::TextureViewDescriptor { dimension: Some(array), ..Default::default() });
        let resolve_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtCaustics/Resolve"),
            layout: &resolve_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.lookup.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accumulator.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.map_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&scratch_view) },
            ],
        });
        let photon_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtCaustics/Photons"),
            layout: &photon_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: tasks.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: accumulator.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.lookup.as_entire_binding() },
            ],
        });
        self.gpu = Some(Gpu {
            params,
            tasks,
            composite_params: buffer("RtCaustics/CompositeParams", std::mem::size_of::<CompositeGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            scratch,
            grid_bgl,
            photons,
            resolve,
            composite,
            photon_group,
            resolve_group,
            grid_group: None,
            backend,
        });
        // the accumulator starts at zero; the tiles start over
        self.planner.tiles = vec![None; RT_CAUSTICS_MAX_CASTERS];
    }
}

impl PostProcessingEffect for RtCausticsEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, gbuffer: &GBuffer, input: &wgpu::TextureView, depth: &wgpu::TextureView, output: &wgpu::TextureView, camera: &Camera, width: u32, height: u32) {
        // made again for a grid whose backend changed (the effect's handle follows its grid)
        if self.gpu.as_ref().is_none_or(|g| g.backend != self.grid.backend()) {
            self.init_gpu(device);
        }
        self.grid.prepare_trace(device, queue);
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let eye = view.inverse().w_axis.truncate();
        // a pixel's width a metre away
        let pixel_scale = 2.0 / (proj.y_axis.y.abs().max(1e-6) * height.max(1) as f32);
        let fed = self.planner.plan(&self.options, &self.casters, &self.lights, self.interval, proj * view, eye, pixel_scale);
        let tasks = self.tasks(&fed);
        let photons: u32 = tasks.iter().map(|t| t.side * t.side).sum();
        let mut fed_tiles: Vec<usize> = tasks.iter().map(|t| t.tile as usize).collect();
        fed_tiles.dedup();
        self.sent = (tasks.len(), photons, fed_tiles.len());
        let scale = self.fixed_point(&tasks);
        let finest = self.options.texel;
        let mut lookup = LookupGpu { enabled: self.enabled as u32, count: RT_CAUSTICS_MAX_CASTERS as u32, intensity: self.intensity.max(0.0), texels: self.options.tile_texels, scale, ..Default::default() };
        for (k, tile) in self.planner.tiles.iter_mut().enumerate() {
            let Some(t) = tile else { continue };
            let texel = finest * (1u32 << t.level) as f32;
            // a tile no light reached this frame keeps what it held
            let blend = if fed_tiles.contains(&k) { fed.iter().find(|f| f.0 == k).map_or(0.0, |f| f.1) } else { 0.0 };
            if blend > 0.0 {
                t.history += 1;
            }
            lookup.tiles[k] = TileGpu {
                min: t.min,
                size: [t.dims[0] as f32 * texel, t.dims[1] as f32 * texel],
                dims: t.dims,
                ref_y: t.lo.y,
                tolerance: self.options.height_tolerance.min(0.5 * (t.hi.y - t.lo.y)).max(2.0 * texel),
                box_min: t.lo.to_array(),
                // read once it holds photons
                live: (t.history > 0) as u32,
                box_max: t.hi.to_array(),
                blend,
            };
        }
        queue.write_buffer(&self.lookup, 0, bytemuck::bytes_of(&lookup));
        self.planner.frame = self.planner.frame.wrapping_add(1);
        let params = ParamsGpu {
            task_count: tasks.len() as u32,
            photons,
            interfaces: self.options.interfaces.max(1),
            max_distance: self.options.max_distance,
            splat_texels: self.options.splat_texels.max(1.0),
            glass_ior: self.glass.0.max(1.0),
            focus_gain: self.focus_gain.max(0.0),
            focus_sharpen: self.focus_sharpen.clamp(0.1, 1.0),
            glass_tint: self.glass.1,
            tint_distance: self.tint_distance.max(1e-6),
        };
        let gpu = self.gpu.as_mut().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        if !tasks.is_empty() {
            queue.write_buffer(&gpu.tasks, 0, bytemuck::cast_slice(&tasks));
        }
        let receivers = match self.receivers {
            RtCausticReceivers::RtLit => 0,
            RtCausticReceivers::All => 1,
        };
        let rect = screen_rect(proj * view, &lookup, [width as f32, height as f32]);
        queue.write_buffer(&gpu.composite_params, 0, bytemuck::bytes_of(&CompositeGpu { inv_proj: proj.inverse().to_cols_array(), inv_view: view.inverse().to_cols_array(), size: [width as f32, height as f32], view: self.view as u32, receivers, rect }));
        // the grid's group, made anew when its buffers (or its acceleration structure) are
        let (bindings, generation) = self.grid.bindings();
        if gpu.grid_group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("RtCaustics/Grid"), layout: &gpu.grid_bgl, entries: &bindings.entries(0) });
            gpu.grid_group = Some((generation, group));
        }
        if photons > 0 {
            let stamp = crate::profiling::gpu_pass("RtCaustics/Photons");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtCaustics/Photons"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            let groups = photons.div_ceil(64);
            pass.set_pipeline(&gpu.photons);
            pass.set_bind_group(0, &gpu.photon_group, &[]);
            pass.set_bind_group(1, &gpu.grid_group.as_ref().unwrap().1, &[]);
            pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
        }
        if !fed_tiles.is_empty() {
            let texels = self.options.tile_texels;
            let stamp = crate::profiling::gpu_pass("RtCaustics/Resolve");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtCaustics/Resolve"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.resolve);
            pass.set_bind_group(0, &gpu.resolve_group, &[]);
            // the widest fed tile's texels, every layer (the others return at once)
            let (w, h) = fed_tiles.iter().fold((0, 0), |(w, h), &k| {
                let d = lookup.tiles[k].dims;
                (w.max(d[0]), h.max(d[1]))
            });
            pass.dispatch_workgroups(w.min(texels).div_ceil(8), h.min(texels).div_ceil(8), RT_CAUSTICS_MAX_CASTERS as u32);
            drop(pass);
            for &k in &fed_tiles {
                let d = lookup.tiles[k].dims;
                let origin = wgpu::Origin3d { x: 0, y: 0, z: k as u32 };
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo { texture: &gpu.scratch, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                    wgpu::TexelCopyTextureInfo { texture: &self.map, mip_level: 0, origin, aspect: wgpu::TextureAspect::All },
                    wgpu::Extent3d { width: d[0], height: d[1], depth_or_array_layers: 1 },
                );
            }
        }
        let composite_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtCaustics/Composite"),
            layout: &gpu.composite.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: gpu.composite_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&gbuffer.normal_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&gbuffer.albedo_view) },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(&gbuffer.emissive_view) },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(input) },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(output) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(&self.map_view) },
                wgpu::BindGroupEntry { binding: 8, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 9, resource: self.lookup.as_entire_binding() },
            ],
        });
        let stamp = crate::profiling::gpu_pass("RtCaustics/Composite");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtCaustics/Composite"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&gpu.composite);
        pass.set_bind_group(0, &composite_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
    }

    fn resize(&mut self, _width: u32, _height: u32, _gbuffer: &GBuffer) {}

    fn destroy(&mut self) {
        self.gpu = None;
    }

    fn name(&self) -> &'static str {
        "RtCaustics"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spot(position: Vec3) -> CausticLight {
        CausticLight::Spot { position, direction: Vec3::NEG_Y, color: Vec3::splat(100.0), range: 10.0, cone: (0.6, 0.4), radius: 0.0, rect: None }
    }

    #[test]
    fn a_footprint_holds_the_shadow_under_each_light() {
        // a 3 cm marble on a table at y = 0.75, a spot light up and to the side
        let (lo, hi) = (Vec3::new(-0.015, 0.75, -0.015), Vec3::new(0.015, 0.78, 0.015));
        let light = Vec3::new(0.4, 1.5, 0.0);
        let (a, b) = footprint(lo, hi, &[spot(light)], 0.06);
        // the shadow of the marble's top on the table lies away from the light
        let top = Vec3::new(-0.015, 0.78, 0.0);
        let s = light + (top - light) * ((0.75 - light.y) / (top.y - light.y));
        assert!(s.x < lo.x && s.x >= a[0] && s.x <= b[0], "{s} in {a:?}..{b:?}");
        // and the box itself
        assert!(a[0] <= lo.x && a[1] <= lo.z && b[0] >= hi.x && b[1] >= hi.z);
        // a grazing sun is held to eight widths
        let sun = CausticLight::Sun { direction: Vec3::new(1.0, -0.001, 0.0).normalize(), color: Vec3::ONE };
        let (a, b) = footprint(lo, hi, &[sun], 0.06);
        assert!(b[0] - a[0] <= 16.0 * 0.03 + 0.031, "{a:?}..{b:?}");
    }

    #[test]
    fn the_texel_follows_the_pixel_and_holds_still() {
        let finest = 0.0005;
        // a pixel of 1.3 mm at the marble: the 2 mm level (2^2 the finest)
        assert_eq!(texel_level(finest, 0.0013, 0.05, 256, None), 2);
        // a little closer or further keeps it
        assert_eq!(texel_level(finest, 0.0011, 0.05, 256, Some(2)), 2);
        assert_eq!(texel_level(finest, 0.0021, 0.05, 256, Some(2)), 2);
        // much closer: finer; never finer than the finest
        assert_eq!(texel_level(finest, 0.0004, 0.05, 256, Some(2)), 0);
        assert_eq!(texel_level(finest, 0.00001, 0.05, 256, None), 0);
        // a footprint too wide for the tile coarsens it whatever the pixel
        let level = texel_level(finest, 0.0001, 1.0, 256, None);
        assert!(1.0 / (finest * (1 << level) as f32) <= 252.0, "{level}");
    }

    #[test]
    fn tiles_clear_when_their_caster_or_the_lights_move_and_stop_when_converged() {
        let view_proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 10.0) * Mat4::look_at_rh(Vec3::new(0.0, 1.2, 0.5), Vec3::new(0.0, 0.75, 0.0), Vec3::Y);
        let eye = Vec3::new(0.0, 1.2, 0.5);
        let options = RtCausticsOptions { max_history: 4, ..Default::default() };
        let marble = |x: f32| (Vec3::new(x - 0.015, 0.75, -0.015), Vec3::new(x + 0.015, 0.78, 0.015));
        let lights = vec![spot(Vec3::new(0.0, 1.5, 0.0))];
        let mut planner = Planner { tiles: vec![None; RT_CAUSTICS_MAX_CASTERS], ..Default::default() };
        let mut step = |casters: &[(Vec3, Vec3)], lights: &[CausticLight], frame: u32| {
            planner.frame = frame;
            let fed = planner.plan(&options, casters, lights, 1, view_proj, eye, 0.002);
            for &(k, _) in &fed {
                planner.tiles[k].as_mut().unwrap().history += 1;
            }
            (fed, planner.tiles[0])
        };
        // still: fed with blends 1, 1/2, 1/3, 1/4, then not at all
        let blends: Vec<f32> = (0..6).map(|f| step(&[marble(0.0)], &lights, f).0.first().map_or(0.0, |x| x.1)).collect();
        assert_eq!(blends, vec![1.0, 0.5, 1.0 / 3.0, 0.25, 0.0, 0.0]);
        // rolling: cleared every frame (blend 1, the same jitter)
        for f in 0..3 {
            assert_eq!(step(&[marble(0.001 * (f + 1) as f32)], &lights, 6 + f).0, vec![(0, 1.0)]);
        }
        // a light moves: cleared
        step(&[marble(0.003)], &lights, 9);
        let moved = vec![spot(Vec3::new(0.01, 1.5, 0.0))];
        assert_eq!(step(&[marble(0.003)], &moved, 10).0, vec![(0, 1.0)]);
        // the marble rolls a millimetre: its tile stays where it was (no new placement)
        let before = step(&[marble(0.003)], &moved, 11).1.unwrap();
        let after = step(&[marble(0.004)], &moved, 12).1.unwrap();
        assert_eq!((before.min, before.dims, before.level), (after.min, after.dims, after.level));
    }
}
