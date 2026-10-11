//! `RtCausticsEffect`: the light glass and water focus (and the shadow they leave), traced from
//! the lights through the ray tracing grid into a map.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::grid::{RtGrid, RtGridHandle};
use crate::cameras::Camera;
use crate::lights::Light;
use crate::postprocessing::PostProcessingEffect;
use crate::renderers::GBuffer;

/// The caustics map read at a surface (`kansei_rt_caustics_lookup`, `kansei_rt_caustics_focus`, the
/// uniform `KanseiRtCaustics`), for passes that light surfaces themselves; prefix it with
/// `simulations::fluid::WATER_RIPPLES_WGSL`.
pub const RT_CAUSTICS_LOOKUP_WGSL: &str = include_str!("shaders/rt_caustics_lookup.wgsl");

fn photons_wgsl() -> String {
    format!(
        "{}\n{}\n{}\n{}\n{}",
        super::RT_GRID_WGSL,
        RtGrid::bindings_wgsl(1, 0),
        super::RT_OPAQUE_WGSL,
        crate::simulations::fluid::WATER_RIPPLES_WGSL,
        include_str!("shaders/rt_caustics.wgsl")
    )
}

fn resolve_wgsl() -> String {
    format!("{}\n{RT_CAUSTICS_LOOKUP_WGSL}\n{}", crate::simulations::fluid::WATER_RIPPLES_WGSL, include_str!("shaders/rt_caustics_resolve.wgsl"))
}

/// What `RtCausticsEffect::new` sets up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RtCausticsOptions {
    /// The map's rectangle on the ground (world x, z): its corners.
    pub min: [f32; 2],
    pub max: [f32; 2],
    /// A texel's size, metres.
    pub texel: f32,
    /// Photons a light sends a frame: this many a side of its grid.
    pub photons: u32,
    /// Weight of each frame's photons in the map.
    pub blend: f32,
    /// The most surfaces a photon crosses or reflects off.
    pub interfaces: u32,
    pub max_distance: f32,
    /// The widest splat, texels (a photon's tent is about as wide as photons lie apart there).
    pub splat_texels: f32,
    /// A surface takes the map's light only within this height (m) of where the photons landed.
    pub height_tolerance: f32,
}

impl Default for RtCausticsOptions {
    fn default() -> Self {
        Self { min: [-5.0, -5.0], max: [5.0, 5.0], texel: 0.03, photons: 128, blend: 0.1, interfaces: 8, max_distance: 60.0, splat_texels: 3.0, height_tolerance: 0.3 }
    }
}

/// The water in the grid (`RtCausticsEffect::set_water`): its surface as a height field
/// (`simulations::fluid::FluidHeightfield`), the source its triangles carry in the grid (its
/// renderable's index), its material, and the ripples its glass material shows.
#[derive(Clone, Debug)]
pub struct RtCausticWater {
    pub heights: wgpu::TextureView,
    /// `FluidHeightfield::rect`: the corner and the size (world x, z) the heights span.
    pub min: [f32; 2],
    pub size: [f32; 2],
    pub dims: [u32; 2],
    pub source: u32,
    pub ior: f32,
    /// The light left after a metre through it.
    pub tint: [f32; 3],
    /// `FluidGlassSurface::ripples`.
    pub ripples: f32,
    /// The still water's height (m): what lies under it takes the ripples' focus
    /// (`RtCausticsEffect::water_photons` off).
    pub level: f32,
}

/// The WGSL `CausticsParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ParamsGpu {
    map_min: [f32; 2],
    map_size: [f32; 2],
    map_dims: [u32; 2],
    frame: u32,
    task_count: u32,
    texel: f32,
    blend: f32,
    max_distance: f32,
    interfaces: u32,
    water_min: [f32; 2],
    water_size: [f32; 2],
    water_dims: [u32; 2],
    water_source: u32,
    ripples: f32,
    time: f32,
    water_ior: f32,
    glass_ior: f32,
    splat_texels: f32,
    water_absorb: [f32; 3],
    photons: u32,
    glass_absorb: [f32; 3],
    /// 1: photons through the water too (`water_photons`)
    flags: u32,
    focus_gain: f32,
    focus_sharpen: f32,
    _pad: [f32; 2],
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
    _pad: [f32; 2],
}

/// The WGSL `KanseiRtCaustics`: the map's rectangle and how it is read.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub(crate) struct LookupGpu {
    map_min: [f32; 2],
    map_size: [f32; 2],
    enabled: u32,
    height_tolerance: f32,
    intensity: f32,
    focus: u32,
    water_min: [f32; 2],
    water_size: [f32; 2],
    water_level: f32,
    ripples: f32,
    time: f32,
    ior: f32,
}

/// The WGSL `ResolveParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ResolveGpu {
    map_dims: [u32; 2],
    blend: f32,
    _pad: u32,
}

/// The WGSL `CompositeParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CompositeGpu {
    inv_proj: [f32; 16],
    inv_view: [f32; 16],
    size: [f32; 2],
    view: u32,
    _pad: u32,
}

const MAX_TASKS: usize = 16;

/// A spot light as `update_lights` found it: position, direction, colour times intensity (cd),
/// range, the cone's outer and inner half-angles, its emitter's radius.
struct Spot {
    position: Vec3,
    direction: Vec3,
    color: Vec3,
    range: f32,
    cone: (f32, f32),
    radius: f32,
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

struct Gpu {
    params: wgpu::Buffer,
    tasks: wgpu::Buffer,
    accumulator: wgpu::Buffer,
    resolve_params: wgpu::Buffer,
    composite_params: wgpu::Buffer,
    /// the resolve's target, copied into the map (which the composite and the reflections read)
    scratch: wgpu::Texture,
    no_water: wgpu::TextureView,
    photons: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    composite: wgpu::ComputePipeline,
    resolve_group: wgpu::BindGroup,
    /// the photons' groups: theirs (made anew when the water changes) and the grid's (with its
    /// generation)
    photon_group: Option<(Option<wgpu::TextureView>, wgpu::BindGroup)>,
    grid_group: Option<(u64, wgpu::BindGroup)>,
}

/// Caustics: the light that glass (`RtSurface::glass`: a glass dragon) and water (its surface in
/// the grid, `set_water`) focus onto the surfaces under them, and the shadow they leave there,
/// from the scene's directional and spot lights (`update_lights`). Each frame every light sends a
/// grid of photons (`photons` a side, jittered) over the boxes the casters fill (`set_casters`)
/// through the renderer's ray tracing grid: a photon whose first hit is glass is refracted through
/// every glass surface it crosses (exact dielectric Fresnel, Beer-Lambert inside, total internal
/// reflection; the water's normal from its height field and ripples) and adds its light where it
/// lands, and the same photon taken straight through no glass takes its light away where the
/// direct light (which passes through glass) lands. The map is thus the change the glass makes to
/// the direct light, temporally blended (`blend`), on a horizontal grid of texels (`min`..`max`):
/// it lights the surfaces that face up (the floor, a pond's bed, a plinth's top) at the height
/// the photons landed. The effect adds it to the image where the GBuffer shows such a surface
/// (times its albedo / pi), and `RtReflectionsEffect::set_caustics` lights its hits with it (the
/// bed under ray-traced water, what the mirrors show). Put it after the GI and before the
/// reflections. A light's photons that hit no glass cost a ray each; directional lights cost one
/// more, toward the light, from each photon that does.
pub struct RtCausticsEffect {
    enabled: bool,
    pub view: RtCausticsView,
    /// Scale of the light the caustics add and take (1 is physical).
    pub intensity: f32,
    /// Photons a light sends a frame: this many a side of its grid.
    pub photons: u32,
    /// Weight of each frame's photons in the map (lower: smoother, slower to follow).
    pub blend: f32,
    /// Frames between the map's updates (1: every frame), the weight of each update raised to
    /// match: what lies under still glass changes slowly, so it can be fed photons every few
    /// frames, the composite reading the map every frame.
    pub interval: u32,
    /// The water's caustics from photons through its surface (its height field and ripples, the
    /// lights' emitters as they are): physical, faint under a large emitter, and as noisy as the
    /// photons are few. Off (the default), photons go through the glass alone, and the water's
    /// caustics are a game's net of lines over the direct light reaching what lies under it
    /// (`kansei_rt_caustics_focus`, `kansei_water_caustic_lines`): crisp, animated, cheap.
    pub water_photons: bool,
    /// Photons leave a spot light's emitter (`set_rect_emitter`, its disk) as it is; off, from its
    /// centre (sharper caustics, less physical).
    pub area_emitters: bool,
    /// The light the glass brings (its photons through it) times this, and their splats this
    /// much narrower (1 and 1: physical; the shadow the glass leaves stays as it is): a game's
    /// caustics, bright, crisp lines and spots where a physical glass's light spreads faint.
    pub focus_gain: f32,
    pub focus_sharpen: f32,
    options: RtCausticsOptions,
    map_dims: [u32; 2],
    grid: RtGridHandle,
    /// the boxes the glass and the water fill (photons are aimed at their union)
    casters: Vec<(Vec3, Vec3)>,
    glass: (f32, [f32; 3]),
    water: Option<RtCausticWater>,
    /// directional lights (direction, illuminance) and spot lights, as `update_lights` found them
    suns: Vec<(Vec3, Vec3)>,
    spots: Vec<Spot>,
    rects: Vec<(usize, [f32; 2])>,
    time: f32,
    frame: u32,
    map: wgpu::Texture,
    map_view: wgpu::TextureView,
    lookup: wgpu::Buffer,
    sampler: wgpu::Sampler,
    gpu: Option<Gpu>,
}

impl RtCausticsEffect {
    pub fn new(device: &wgpu::Device, grid: RtGridHandle, options: RtCausticsOptions) -> Self {
        let size = [(options.max[0] - options.min[0]).max(0.01), (options.max[1] - options.min[1]).max(0.01)];
        let map_dims = [((size[0] / options.texel).ceil() as u32).clamp(1, 4096), ((size[1] / options.texel).ceil() as u32).clamp(1, 4096)];
        let map = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RtCaustics/Map"),
            size: wgpu::Extent3d { width: map_dims[0], height: map_dims[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let map_view = map.create_view(&Default::default());
        let lookup = device.create_buffer(&wgpu::BufferDescriptor { label: Some("RtCaustics/Lookup"), size: std::mem::size_of::<LookupGpu>() as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor { label: Some("RtCaustics/Sampler"), mag_filter: wgpu::FilterMode::Linear, min_filter: wgpu::FilterMode::Linear, ..Default::default() });
        Self {
            enabled: true,
            view: RtCausticsView::Lit,
            intensity: 1.0,
            photons: options.photons,
            blend: options.blend,
            interval: 1,
            water_photons: false,
            area_emitters: false,
            focus_gain: 1.0,
            focus_sharpen: 1.0,
            options,
            map_dims,
            grid,
            casters: Vec::new(),
            glass: (1.5, [1.0; 3]),
            water: None,
            suns: Vec::new(),
            spots: Vec::new(),
            rects: Vec::new(),
            time: 0.0,
            frame: 0,
            map,
            map_view,
            lookup,
            sampler,
            gpu: None,
        }
    }

    /// The boxes (world min, max) the glass and the water fill: each light's photons are aimed at
    /// their union.
    pub fn set_casters(&mut self, boxes: &[(Vec3, Vec3)]) {
        self.casters = boxes.to_vec();
    }

    /// The glass's (every glass triangle but the water's) index of refraction and tint (the light
    /// left after a metre inside).
    pub fn set_glass(&mut self, ior: f32, tint: [f32; 3]) {
        self.glass = (ior, tint);
    }

    /// The water (or none).
    pub fn set_water(&mut self, water: Option<RtCausticWater>) {
        self.water = water;
    }

    /// Seconds, for the water's ripples.
    pub fn set_time(&mut self, seconds: f32) {
        self.time = seconds;
    }

    /// The `index`th spot light (in `update_lights`' order, counting spot lights only) emits from a
    /// rectangle `width` x `height` (m) facing along its axis (`RtShadowsEffect::set_rect_emitter`'s
    /// axes), not from its disk.
    pub fn set_rect_emitter(&mut self, index: usize, width: f32, height: f32) {
        self.rects.retain(|r| r.0 != index);
        self.rects.push((index, [width, height]));
    }

    /// The scene's lights (`Scene::lights`), each frame they may change: its directional and spot
    /// lights send photons.
    pub fn update_lights<'a>(&mut self, lights: impl IntoIterator<Item = &'a Light>) {
        self.suns.clear();
        self.spots.clear();
        for light in lights {
            match light {
                Light::Directional(l) => {
                    let c = l.effective_color();
                    self.suns.push((Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or(Vec3::NEG_Y), Vec3::new(c.x, c.y, c.z)));
                }
                Light::Spot(l) => {
                    let c = l.effective_color();
                    self.spots.push(Spot {
                        position: Vec3::new(l.position.x, l.position.y, l.position.z),
                        direction: Vec3::new(l.direction.x, l.direction.y, l.direction.z).normalize_or(Vec3::NEG_Z),
                        color: Vec3::new(c.x, c.y, c.z),
                        range: l.range.max(1e-3),
                        cone: l.cone(),
                        radius: l.source_radius.max(0.0),
                    });
                }
                Light::Point(_) | Light::Area(_) => {}
            }
        }
    }

    /// Turn it on or off (`is_active`). Off, what reads its map (`map`'s lookup,
    /// `RtReflectionsEffect::set_caustics`) sees no caustics either: an inactive effect isn't
    /// rendered, so the lookup is cleared here.
    pub fn set_enabled(&mut self, queue: &wgpu::Queue, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            queue.write_buffer(&self.lookup, 0, bytemuck::bytes_of(&LookupGpu::default()));
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// The map (rgb: the irradiance the caustics add, lux, signed; a: the height the photons
    /// landed at) and its uniform (`KanseiRtCaustics`), for `RT_CAUSTICS_LOOKUP_WGSL`.
    pub fn map(&self) -> (&wgpu::TextureView, &wgpu::Buffer) {
        (&self.map_view, &self.lookup)
    }

    /// The lights sending photons now (directional ones first) and the photons they send.
    pub fn photons_sent(&self) -> (usize, u32) {
        let tasks = self.tasks();
        (tasks.len(), tasks.iter().map(|t| t.side * t.side).sum())
    }

    /// The lights' photon grids this frame.
    fn tasks(&self) -> Vec<TaskGpu> {
        let mut tasks = Vec::new();
        let Some((lo, hi)) = self.casters.iter().copied().reduce(|(a, b), (c, d)| (a.min(c), b.max(d))) else { return tasks };
        let corners: Vec<Vec3> = (0..8).map(|k| Vec3::new(if k & 1 == 0 { lo.x } else { hi.x }, if k & 2 == 0 { lo.y } else { hi.y }, if k & 4 == 0 { lo.z } else { hi.z })).collect();
        let side = self.photons.clamp(1, 1024);
        let basis = |a: Vec3| {
            let any = if a.y.abs() < 0.9 { Vec3::Y } else { Vec3::X };
            let u = a.cross(any).normalize();
            (u, a.cross(u).normalize())
        };
        for &(dir, color) in &self.suns {
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
            // the photons' plane just outside the box, toward the light
            let origin = dir * (t - 0.5);
            tasks.push(TaskGpu { origin: origin.to_array(), kind: 0, axis: dir.to_array(), side, e1: e1.to_array(), u0: u[0], e2: e2.to_array(), v0: v[0], color: color.to_array(), du: (u[1] - u[0]) / side as f32, dv: (v[1] - v[0]) / side as f32, ..Default::default() });
        }
        for (index, l) in self.spots.iter().enumerate() {
            let (p, d, c) = (l.position, l.direction, l.color);
            if c.max_element() <= 0.0 {
                continue;
            }
            let (outer, inner) = l.cone;
            // aimed at the box's centre, the box's corners as tangents round that axis
            let axis = ((lo + hi) * 0.5 - p).normalize_or(d);
            let (e1, e2) = basis(axis);
            let (mut u, mut v) = ([f32::MAX, f32::MIN], [f32::MAX, f32::MIN]);
            let mut behind = false;
            for q in &corners {
                let w = *q - p;
                let z = w.dot(axis);
                if z < 0.05 {
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
            // its emitter: a rectangle (`set_rect_emitter`) or its disk, as a square
            let (ex, ey) = match self.rects.iter().find(|r| r.0 == index).filter(|_| self.area_emitters) {
                Some((_, [w, h])) => {
                    let (a, b) = basis(d);
                    (a * w * 0.5, b * h * 0.5)
                }
                None => {
                    let (a, b) = basis(d);
                    let r = if self.area_emitters { l.radius } else { 0.0 };
                    (a * r, b * r)
                }
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
                range: l.range,
                ..Default::default()
            });
        }
        tasks.truncate(MAX_TASKS);
        let mut first = 0;
        for t in &mut tasks {
            t.first = first;
            first += t.side * t.side;
        }
        tasks
    }

    fn init_gpu(&mut self, device: &wgpu::Device) {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let d2 = wgpu::TextureViewDimension::D2;
        let texture = |filterable| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable }, view_dimension: d2, multisampled: false };
        let storage_tex = wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: d2 };
        let layout = |label, entries: &[wgpu::BindGroupLayoutEntry]| device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries });
        let photon_bgl = layout("RtCaustics/Photons", &[entry(0, uniform), entry(1, storage(true)), entry(2, storage(false)), entry(3, texture(false))]);
        let grid_bgl = layout("RtCaustics/Grid", &RtGrid::layout_entries(0, compute));
        let resolve_bgl = layout("RtCaustics/Resolve", &[entry(0, uniform), entry(1, storage(false)), entry(2, texture(false)), entry(3, storage_tex)]);
        let depth = wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Depth, view_dimension: d2, multisampled: false };
        let composite_bgl = layout(
            "RtCaustics/Composite",
            &[
                entry(0, uniform),
                entry(1, depth),
                entry(2, texture(false)),
                entry(3, texture(false)),
                entry(4, texture(false)),
                entry(5, texture(false)),
                entry(6, storage_tex),
                entry(7, texture(true)),
                entry(8, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(9, uniform),
            ],
        );
        let pipeline = |label: &str, code: &str, entry_point: &str, layouts: &[&wgpu::BindGroupLayout]| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: layouts, push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(label), layout: Some(&pl), module: &module, entry_point: Some(entry_point), compilation_options: Default::default(), cache: None })
        };
        let photons = pipeline("RtCaustics/Photons", &photons_wgsl(), "photons", &[&photon_bgl, &grid_bgl]);
        let resolve_code = resolve_wgsl();
        let resolve = pipeline("RtCaustics/Resolve", &resolve_code, "resolve", &[&resolve_bgl]);
        let composite = pipeline("RtCaustics/Composite", &resolve_code, "composite", &[&composite_bgl]);
        let buffer = |label, size: u64, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let texels = (self.map_dims[0] * self.map_dims[1]) as u64;
        let accumulator = buffer("RtCaustics/Accumulator", texels * 5 * 4, wgpu::BufferUsages::STORAGE);
        let resolve_params = buffer("RtCaustics/ResolveParams", std::mem::size_of::<ResolveGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let scratch = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("RtCaustics/Scratch"),
            size: wgpu::Extent3d { width: self.map_dims[0], height: self.map_dims[1], depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let scratch_view = scratch.create_view(&Default::default());
        let resolve_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RtCaustics/Resolve"),
            layout: &resolve.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: resolve_params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accumulator.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&self.map_view) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&scratch_view) },
            ],
        });
        let no_water = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("RtCaustics/NoWater"),
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        self.gpu = Some(Gpu {
            params: buffer("RtCaustics/Params", std::mem::size_of::<ParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            tasks: buffer("RtCaustics/Tasks", (std::mem::size_of::<TaskGpu>() * MAX_TASKS) as u64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST),
            accumulator,
            resolve_params,
            composite_params: buffer("RtCaustics/CompositeParams", std::mem::size_of::<CompositeGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            scratch,
            no_water,
            photons,
            resolve,
            composite,
            resolve_group,
            photon_group: None,
            grid_group: None,
        });
    }
}

impl PostProcessingEffect for RtCausticsEffect {
    fn initialize(&mut self, _device: &wgpu::Device, _gbuffer: &GBuffer, _camera: &Camera) {}

    fn is_active(&self) -> bool {
        self.enabled
    }

    fn render(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, gbuffer: &GBuffer, input: &wgpu::TextureView, depth: &wgpu::TextureView, output: &wgpu::TextureView, camera: &Camera, width: u32, height: u32) {
        if self.gpu.is_none() {
            self.init_gpu(device);
        }
        let tasks = self.tasks();
        let photons: u32 = tasks.iter().map(|t| t.side * t.side).sum();
        let size = [self.options.max[0] - self.options.min[0], self.options.max[1] - self.options.min[1]];
        let absorb = |tint: [f32; 3]| tint.map(|t| -t.clamp(1e-4, 1.0).ln());
        let water = self.water.as_ref();
        let params = ParamsGpu {
            map_min: self.options.min,
            map_size: size,
            map_dims: self.map_dims,
            frame: self.frame,
            task_count: tasks.len() as u32,
            texel: size[0] / self.map_dims[0] as f32,
            blend: (self.blend * self.interval.max(1) as f32).clamp(0.01, 1.0),
            max_distance: self.options.max_distance,
            interfaces: self.options.interfaces.max(1),
            water_min: water.map_or([0.0; 2], |w| w.min),
            water_size: water.map_or([1.0; 2], |w| w.size),
            water_dims: water.map_or([1; 2], |w| w.dims),
            water_source: water.map_or(u32::MAX, |w| w.source),
            ripples: water.map_or(0.0, |w| w.ripples),
            time: self.time,
            water_ior: water.map_or(1.33, |w| w.ior),
            glass_ior: self.glass.0,
            splat_texels: self.options.splat_texels.max(1.0),
            water_absorb: absorb(water.map_or([1.0; 3], |w| w.tint)),
            photons,
            glass_absorb: absorb(self.glass.1),
            flags: self.water_photons as u32,
            focus_gain: self.focus_gain.max(0.0),
            focus_sharpen: self.focus_sharpen.clamp(0.1, 1.0),
            _pad: [0.0; 2],
        };
        let lookup = LookupGpu {
            map_min: self.options.min,
            map_size: size,
            enabled: self.enabled as u32,
            height_tolerance: self.options.height_tolerance,
            intensity: self.intensity.max(0.0),
            focus: (water.is_some() && !self.water_photons) as u32,
            water_min: water.map_or([0.0; 2], |w| w.min),
            water_size: water.map_or([1.0; 2], |w| w.size),
            water_level: water.map_or(-1e9, |w| w.level),
            ripples: water.map_or(0.0, |w| w.ripples),
            time: self.time,
            ior: water.map_or(1.33, |w| w.ior),
        };
        queue.write_buffer(&self.lookup, 0, bytemuck::bytes_of(&lookup));
        // (photons into the map every `interval` frames)
        let update = self.frame.is_multiple_of(self.interval.max(1));
        self.frame = self.frame.wrapping_add(1);
        let proj = camera.projection_matrix.to_glam();
        let view = camera.view_matrix.to_glam();
        let gpu = self.gpu.as_mut().unwrap();
        queue.write_buffer(&gpu.params, 0, bytemuck::bytes_of(&params));
        if !tasks.is_empty() {
            queue.write_buffer(&gpu.tasks, 0, bytemuck::cast_slice(&tasks));
        }
        queue.write_buffer(&gpu.resolve_params, 0, bytemuck::bytes_of(&ResolveGpu { map_dims: self.map_dims, blend: params.blend, _pad: 0 }));
        queue.write_buffer(&gpu.composite_params, 0, bytemuck::bytes_of(&CompositeGpu { inv_proj: proj.inverse().to_cols_array(), inv_view: view.inverse().to_cols_array(), size: [width as f32, height as f32], view: self.view as u32, _pad: 0 }));
        // the photons' groups, anew when the water's heights or the grid's buffers change
        let heights = water.map(|w| w.heights.clone());
        if gpu.photon_group.as_ref().is_none_or(|(h, _)| *h != heights) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtCaustics/Photons"),
                layout: &gpu.photons.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: gpu.params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: gpu.tasks.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: gpu.accumulator.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(heights.as_ref().unwrap_or(&gpu.no_water)) },
                ],
            });
            gpu.photon_group = Some((heights, group));
        }
        let (buffers, generation) = self.grid.buffers();
        if gpu.grid_group.as_ref().is_none_or(|(g, _)| *g != generation) {
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("RtCaustics/Grid"),
                layout: &gpu.photons.get_bind_group_layout(1),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: buffers[0].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: buffers[1].as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: buffers[2].as_entire_binding() },
                ],
            });
            gpu.grid_group = Some((generation, group));
        }
        if update {
            let stamp = crate::profiling::gpu_pass("RtCaustics/Photons");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtCaustics/Photons"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            if photons > 0 {
                let groups = photons.div_ceil(64);
                pass.set_pipeline(&gpu.photons);
                pass.set_bind_group(0, &gpu.photon_group.as_ref().unwrap().1, &[]);
                pass.set_bind_group(1, &gpu.grid_group.as_ref().unwrap().1, &[]);
                pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            }
        }
        if update {
            let stamp = crate::profiling::gpu_pass("RtCaustics/Resolve");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("RtCaustics/Resolve"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&gpu.resolve);
            pass.set_bind_group(0, &gpu.resolve_group, &[]);
            pass.dispatch_workgroups(self.map_dims[0].div_ceil(8), self.map_dims[1].div_ceil(8), 1);
            drop(pass);
            encoder.copy_texture_to_texture(gpu.scratch.as_image_copy(), self.map.as_image_copy(), wgpu::Extent3d { width: self.map_dims[0], height: self.map_dims[1], depth_or_array_layers: 1 });
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

    fn struct_span(module: &naga::Module, name: &str) -> Option<usize> {
        module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
            _ => None,
        })
    }

    fn validate(code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        module
    }

    #[test]
    fn shaders_validate_and_the_structs_match() {
        let photons = validate(&photons_wgsl());
        assert_eq!(struct_span(&photons, "CausticsParams"), Some(std::mem::size_of::<ParamsGpu>()));
        assert_eq!(struct_span(&photons, "CausticTask"), Some(std::mem::size_of::<TaskGpu>()));
        let resolve = validate(&resolve_wgsl());
        assert_eq!(struct_span(&resolve, "KanseiRtCaustics"), Some(std::mem::size_of::<LookupGpu>()));
        assert_eq!(struct_span(&resolve, "ResolveParams"), Some(std::mem::size_of::<ResolveGpu>()));
        assert_eq!(struct_span(&resolve, "CompositeParams"), Some(std::mem::size_of::<CompositeGpu>()));
    }
}
