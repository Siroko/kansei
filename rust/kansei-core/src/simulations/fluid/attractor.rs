//! Additive glyph attractor for the fluid sim: pulls tagged particles into the
//! extruded glyph SDF volumes (from `crate::sdf`). Runs as its own compute pass
//! after the SPH solver; modifies velocities only, leaving the solver untouched.

use crate::sdf::GlyphVolumeSet;
use bytemuck::{Pod, Zeroable};

/// GPU layout of one slot: 3 × vec4f = 48 bytes (std140-friendly).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuSlot {
    pub world_min: [f32; 4],  // xyz + pad
    pub world_size: [f32; 4], // xyz + pad
    pub glyph_id: i32,
    pub budget: u32,
    pub _pad: [i32; 2],
}

impl From<&AttractorSlot> for GpuSlot {
    fn from(s: &AttractorSlot) -> Self {
        GpuSlot {
            world_min: [s.world_min[0], s.world_min[1], s.world_min[2], 0.0],
            world_size: [s.world_size[0], s.world_size[1], s.world_size[2], 0.0],
            glyph_id: s.glyph_id,
            budget: s.budget,
            _pad: [0; 2],
        }
    }
}

/// GPU attractor parameters (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuAttractorParams {
    pub res_xy: u32,
    pub res_z: u32,
    pub glyph_count: u32,
    pub stiffness: f32,
    pub dt: f32,
    pub max_speed: f32,
    pub basin_strength: f32,
    /// Shell depth (in normalized SDF units, +inside) that in-box particles are
    /// pulled toward. Must be below the stroke half-width (~0.19–0.31 for the
    /// clock digits at res 64) or the stroke collapses onto its medial axis.
    pub target: f32,
    /// Velocity drag (per second) applied to tagged particles. Without it a
    /// sparse recruit has nothing to dissipate energy and oscillates through
    /// the glyph at `max_speed` forever; dense fluid gets this from viscosity.
    pub drag: f32,
    pub _pad: [f32; 3],
}

/// Per-frame inputs to [`GlyphAttractor::retag`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RetagParams {
    /// Bit `k` set = slot `k`'s digit changed this frame (its particles are released).
    pub changed_mask: u32,
    /// Recruitment cap for slots whose `budget` is 0.
    pub per_slot_count: u32,
    /// Frames a released particle stays ineligible for recruitment.
    pub cooldown_frames: u32,
    /// Capture-box scale (>= 1.0 enlarges the slot box for recruitment).
    pub capture_scale: f32,
    /// Extra reach of the recruit region below the slot box (world units).
    pub capture_below: f32,
    /// > 0: emit recruits this far above the box so they fall into the glyph.
    pub emit_height: f32,
    /// Vertical thickness of the emitter slab.
    pub emit_spread: f32,
    /// Max recruits per slot per frame (0 = unlimited).
    pub emit_rate: u32,
}

impl Default for RetagParams {
    fn default() -> Self {
        RetagParams {
            changed_mask: 0,
            per_slot_count: 0,
            cooldown_frames: 45,
            capture_scale: 1.15,
            capture_below: 0.0,
            emit_height: 0.0,
            emit_spread: 0.0,
            emit_rate: 0,
        }
    }
}

/// GPU params for the tagging passes (32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct GpuTagParams {
    pub changed_mask: u32,
    pub per_slot_count: u32,
    pub cooldown_frames: u32,
    pub capture_scale: f32,
    /// Extra reach of the recruit region *below* the slot box (world units),
    /// so recruits come from the bulk of the pool, not just its skin.
    pub capture_below: f32,
    /// > 0: recruits are teleported to an emitter slab this far above the
    /// slot box (world units) and fall into the glyph. 0: they stay where
    /// they were recruited and get pulled in from there.
    pub emit_height: f32,
    /// Vertical thickness of the emitter slab (world units).
    pub emit_spread: f32,
    /// Max recruits per slot per frame (0 = fill the budget in one frame).
    pub emit_rate: u32,
}

/// Number of glyphs packed into the atlas (`0`–`9` and `:`).
const ATLAS_GLYPHS: u32 = 11;

/// All 11 glyph SDF volumes flattened into a single 3D field of size
/// `(res_xy, res_xy, res_z * 11)`, stacked along Z. Glyph `g` occupies
/// depth `[g*res_z, (g+1)*res_z)`. Missing glyphs are filled with `-1.0`
/// (fully outside), so no particle is ever attracted to an absent glyph.
pub struct GlyphVolumeAtlas {
    pub res_xy: u32,
    pub res_z: u32,
    pub glyph_count: u32,
    /// `res_xy * res_xy * (res_z * glyph_count)` f32 SDF values, +inside.
    pub data: Vec<f32>,
}

impl GlyphVolumeAtlas {
    /// Total depth of the packed texture (`res_z * glyph_count`).
    pub fn depth(&self) -> u32 {
        self.res_z * self.glyph_count
    }

    /// Flatten a `GlyphVolumeSet` into the packed atlas.
    pub fn from_set(set: &GlyphVolumeSet) -> GlyphVolumeAtlas {
        let res_xy = set.res_xy;
        let res_z = set.res_z;
        let glyph_count = ATLAS_GLYPHS;
        let slab = (res_xy * res_xy * res_z) as usize;
        let mut data = vec![-1.0f32; slab * glyph_count as usize];

        for (g, vol) in set.volumes().iter().enumerate() {
            if let Some(v) = vol {
                let base = g * slab;
                // v.data is already ((z*res_xy)+y)*res_xy + x, same order as a slab.
                data[base..base + slab].copy_from_slice(&v.data);
            }
        }
        GlyphVolumeAtlas { res_xy, res_z, glyph_count, data }
    }
}

/// Number of glyph slots in the clock display: `H H : M M : S S`.
pub const NUM_SLOTS: usize = 8;

/// Slot indices that hold the fixed `:` separators.
const COLON_SLOTS: [usize; 2] = [2, 5];
/// Glyph id of the `:` glyph in the atlas (digits are 0..=9, colon is 10).
const COLON_GLYPH_ID: i32 = 10;

/// One glyph slot: which glyph it currently shows and the world-space box that
/// maps a particle position into the glyph's local voxel coordinates.
/// `glyph_id == -1` disables the slot (no attraction).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AttractorSlot {
    pub glyph_id: i32,
    /// World-space minimum corner of the slot's box.
    pub world_min: [f32; 3],
    /// World-space size of the slot's box (glyph maps into this).
    pub world_size: [f32; 3],
    /// Recruitment cap for this slot; 0 = use the global `per_slot_count`.
    /// A glyph only holds so many particles at SPH rest density (the colon
    /// has under half a digit's stroke area), so size this to the stroke.
    pub budget: u32,
}

impl AttractorSlot {
    fn empty() -> Self {
        AttractorSlot { glyph_id: -1, world_min: [0.0; 3], world_size: [1.0; 3], budget: 0 }
    }
}

/// The 8 clock slots in world space.
pub struct SlotLayout {
    pub slots: [AttractorSlot; NUM_SLOTS],
    /// Whether slots 2 and 5 show `:` (false = they stay disabled, e.g. in
    /// the stacked layout where rows replace the separators).
    pub colons: bool,
}

impl SlotLayout {
    /// Lay out `H H : M M : S S` centered on the origin in the XY plane.
    /// `cell` is the world width/height of each glyph box; `depth` its Z size.
    pub fn hh_mm_ss(cell: f32, depth: f32) -> SlotLayout {
        let spacing = cell * 1.05;
        let total_w = spacing * (NUM_SLOTS as f32);
        let x0 = -total_w * 0.5;
        let mut slots = [AttractorSlot::empty(); NUM_SLOTS];
        for i in 0..NUM_SLOTS {
            let x = x0 + spacing * (i as f32);
            slots[i] = AttractorSlot {
                glyph_id: if COLON_SLOTS.contains(&i) { COLON_GLYPH_ID } else { -1 },
                world_min: [x, -cell * 0.5, -depth * 0.5],
                world_size: [cell, cell, depth],
                budget: 0,
            };
        }
        SlotLayout { slots, colons: true }
    }

    /// Lay out the time as three stacked rows, `HH` on top, `MM` in the
    /// middle, `SS` at the bottom, centered on the origin in the XY plane.
    /// Colon slots (2, 5) stay disabled. `row_spacing` is the row pitch as a
    /// multiple of `cell` (e.g. 1.3 leaves a 0.3·cell gap between rows).
    pub fn stacked_hh_mm_ss(cell: f32, depth: f32, row_spacing: f32) -> SlotLayout {
        let col_spacing = cell * 1.05;
        let pitch = cell * row_spacing;
        let mut slots = [AttractorSlot::empty(); NUM_SLOTS];
        // (slot index, column 0/1, row 0=top..2=bottom)
        let placed = [(0usize, 0.0f32, 0.0f32), (1, 1.0, 0.0), (3, 0.0, 1.0), (4, 1.0, 1.0), (6, 0.0, 2.0), (7, 1.0, 2.0)];
        for (i, col, row) in placed {
            let x = -col_spacing + col * col_spacing + (col_spacing - cell) * 0.5;
            let cy = pitch - row * pitch;
            slots[i] = AttractorSlot {
                glyph_id: -1,
                world_min: [x, cy - cell * 0.5, -depth * 0.5],
                world_size: [cell, cell, depth],
                budget: 0,
            };
        }
        SlotLayout { slots, colons: false }
    }

    /// Set the displayed time. Digit slots get the time digits; colon slots
    /// show `:` when `colons` is set and stay disabled otherwise.
    pub fn set_time(&mut self, hours: u32, minutes: u32, seconds: u32) {
        let h = (hours % 100) as i32;
        let m = (minutes % 100) as i32;
        let s = (seconds % 100) as i32;
        let digits = [h / 10, h % 10, m / 10, m % 10, s / 10, s % 10];
        // Digit slots in display order (skipping the two colon slots).
        let digit_slots = [0usize, 1, 3, 4, 6, 7];
        for (k, &slot_idx) in digit_slots.iter().enumerate() {
            self.slots[slot_idx].glyph_id = digits[k];
        }
        for &c in COLON_SLOTS.iter() {
            self.slots[c].glyph_id = if self.colons { COLON_GLYPH_ID } else { -1 };
        }
    }
}

const ATTRACTOR_WGSL: &str = r#"
struct Slot {
    world_min: vec4<f32>,
    world_size: vec4<f32>,
    glyph_id: i32,
    budget: u32,
    _pad1: i32,
    _pad2: i32,
};

struct Params {
    res_xy: u32,
    res_z: u32,
    glyph_count: u32,
    stiffness: f32,
    dt: f32,
    max_speed: f32,
    basin_strength: f32,
    shell: f32, // (`target` is a reserved word in WGSL)
    drag: f32,
    _pad0: f32, _pad1: f32, _pad2: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tags: array<i32>;
// The array length (8) must equal NUM_SLOTS in the Rust module.
@group(0) @binding(3) var<uniform> slots: array<Slot, 8>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var sdf_tex: texture_3d<f32>;

// Load the SDF for glyph g at integer voxel (x,y,z), clamped into the glyph's Z band.
fn load_sdf(g: i32, x: i32, y: i32, z: i32) -> f32 {
    let rx = i32(params.res_xy);
    let rz = i32(params.res_z);
    let cx = clamp(x, 0, rx - 1);
    let cy = clamp(y, 0, rx - 1);
    let cz = clamp(z, 0, rz - 1);
    let abs_z = g * rz + cz;
    return textureLoad(sdf_tex, vec3<i32>(cx, cy, abs_z), 0).r;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let slot_id = tags[idx];
    // NUM_SLOTS (Rust) = 8; keep this literal in sync with it.
    let is_held = slot_id >= 0 && slot_id < 8 && slots[max(slot_id, 0)].glyph_id >= 0;
    if (!is_held) {
        // Not held: clear the flag so the solver applies gravity again.
        velocities[idx].w = 0.0;
        return;
    }
    let slot = slots[slot_id];

    let pos = positions[idx].xyz;
    let local = (pos - slot.world_min.xyz) / slot.world_size.xyz; // [0,1] inside box
    let inside_box = all(local >= vec3<f32>(0.0)) && all(local <= vec3<f32>(1.0));

    // Voxel coordinate in the glyph's local grid (clamped: outside the box
    // this samples the nearest face voxel).
    let rx = f32(params.res_xy);
    let rz = f32(params.res_z);
    let lc = clamp(local, vec3<f32>(0.0), vec3<f32>(1.0));
    let vx = i32(lc.x * (rx - 1.0));
    let vy = i32(lc.y * (rx - 1.0));
    let vz = i32(lc.z * (rz - 1.0));
    // SDF gradient via central differences (points toward increasing SDF = inside).
    let gx = load_sdf(slot.glyph_id, vx + 1, vy, vz) - load_sdf(slot.glyph_id, vx - 1, vy, vz);
    let gy = load_sdf(slot.glyph_id, vx, vy + 1, vz) - load_sdf(slot.glyph_id, vx, vy - 1, vz);
    let gz = load_sdf(slot.glyph_id, vx, vy, vz + 1) - load_sdf(slot.glyph_id, vx, vy, vz - 1);
    let grad = vec3<f32>(gx, gy, gz);
    let s = load_sdf(slot.glyph_id, vx, vy, vz);
    // Restoring force toward a shell at depth `params.shell` INSIDE the glyph.
    // (shell - s) pulls exterior particles inward (s<shell) AND pushes
    // over-packed interior particles back out (s>shell), so the stroke fills
    // its width instead of collapsing onto the medial axis. The old
    // max(-s,0) vanished at the boundary and left the interior force-free.
    // NOTE: a glyph only holds so many particles at SPH rest density; past
    // that the excess forms a skin outside the stroke whatever the force.
    let pull = params.stiffness * (params.shell - s);

    var force = vec3<f32>(0.0);
    if (inside_box) {
        if (length(grad) > 1e-5) {
            force = normalize(grad) * pull;
        }
    } else {
        // Basin: aim at the nearest point ON the box (not its center, which
        // piled everything into one ball). Magnitude = the face voxel's own
        // pull (continuous with the in-box force, so nothing hovers just
        // outside a face) plus a spring that grows with distance.
        let box_max = slot.world_min.xyz + slot.world_size.xyz;
        let nearest = clamp(pos, slot.world_min.xyz, box_max);
        let to_box = nearest - pos;
        let dist = length(to_box);
        if (dist > 1e-5) {
            force = (to_box / dist) * (max(pull, 0.0) + params.basin_strength * dist);
        }
    }

    var vel = velocities[idx].xyz + force * params.dt;
    // Drag so held particles settle instead of oscillating through the glyph.
    vel = vel * max(1.0 - params.drag * params.dt, 0.0);
    let sp = length(vel);
    if (sp > params.max_speed) { vel = vel / sp * params.max_speed; }
    // w = 1: "held" flag; the solver's forces pass skips gravity for us.
    // Only once inside the box: a recruit still outside (e.g. emitted above
    // the glyph) keeps falling under gravity, and the basin steers it in.
    velocities[idx] = vec4<f32>(vel, select(0.0, 1.0, inside_box));
}
"#;

const TAGGER_WGSL: &str = r#"
struct Slot {
    world_min: vec4<f32>,
    world_size: vec4<f32>,
    glyph_id: i32,
    budget: u32, // 0 = use params.per_slot_count
    _pad1: i32, _pad2: i32,
};
struct TagParams {
    changed_mask: u32,
    per_slot_count: u32,
    cooldown_frames: u32,
    capture_scale: f32,
    capture_below: f32,
    emit_height: f32,
    emit_spread: f32,
    emit_rate: u32,
};

@group(0) @binding(0) var<storage, read_write> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tags: array<i32>;
@group(0) @binding(2) var<storage, read_write> cooldown: array<u32>;
@group(0) @binding(3) var<storage, read_write> slot_fill: array<atomic<u32>, 8>;
@group(0) @binding(4) var<uniform> slots: array<Slot, 8>;
@group(0) @binding(5) var<uniform> params: TagParams;
@group(0) @binding(6) var<storage, read_write> velocities: array<vec4<f32>>;
// slot_fill as it was before this frame's recruit pass (for emit_rate).
@group(0) @binding(7) var<storage, read_write> slot_start: array<u32, 8>;

fn hash_u32(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}
fn rand01(seed: u32) -> f32 {
    return f32(hash_u32(seed) & 0xffffffu) / 16777216.0;
}

@compute @workgroup_size(8)
fn snapshot_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 8u) { slot_start[gid.x] = atomicLoad(&slot_fill[gid.x]); }
}

@compute @workgroup_size(8)
fn clear_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < 8u) { atomicStore(&slot_fill[gid.x], 0u); }
}

@compute @workgroup_size(64)
fn count_fill(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let t = tags[idx];
    if (t >= 0 && t < 8) { atomicAdd(&slot_fill[t], 1u); }
}

@compute @workgroup_size(64)
fn release(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }
    let t = tags[idx];
    if (t >= 0 && t < 8) {
        if ((params.changed_mask & (1u << u32(t))) != 0u) {
            tags[idx] = -1;
            cooldown[idx] = params.cooldown_frames;
        }
    }
}

@compute @workgroup_size(64)
fn recruit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&positions)) { return; }

    let cd = cooldown[idx];
    if (cd > 0u) { cooldown[idx] = cd - 1u; return; }

    if (tags[idx] >= 0) { return; }

    let pos = positions[idx].xyz;
    for (var k: i32 = 0; k < 8; k = k + 1) {
        let slot = slots[k];
        if (slot.glyph_id < 0) { continue; }
        let half = 0.5 * slot.world_size.xyz * params.capture_scale;
        let center = slot.world_min.xyz + 0.5 * slot.world_size.xyz;
        // Recruit region: the scaled box, extended downward into the pool.
        let lo = center - half - vec3<f32>(0.0, params.capture_below, 0.0);
        let hi = center + half;
        if (all(pos >= lo) && all(pos <= hi)) {
            var cap = select(params.per_slot_count, slot.budget, slot.budget > 0u);
            if (params.emit_rate > 0u) {
                cap = min(cap, slot_start[k] + params.emit_rate);
            }
            let n = atomicAdd(&slot_fill[k], 1u);
            if (n < cap) {
                tags[idx] = k;
                if (params.emit_height > 0.0) {
                    // Emit above the glyph: random column over the box, in a
                    // slab [top + emit_height, + emit_spread]; the basin +
                    // gravity bring it down into the glyph.
                    let r = vec3<f32>(rand01(idx * 3u + 1u), rand01(idx * 3u + 2u), rand01(idx * 3u + 3u));
                    let sz = slot.world_size.xyz;
                    let top = slot.world_min.y + sz.y;
                    let p = vec3<f32>(
                        center.x + (r.x - 0.5) * 0.8 * sz.x,
                        top + params.emit_height + r.y * params.emit_spread,
                        center.z + (r.z - 0.5) * 0.8 * sz.z);
                    positions[idx] = vec4<f32>(p, positions[idx].w);
                    velocities[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                }
                return;
            }
            // This slot is full: give it back and try the next slot whose
            // capture region also contains us (stacked rows share columns).
            atomicSub(&slot_fill[k], 1u);
        }
    }
}
"#;

use crate::buffers::Texture;
use crate::renderers::Renderer;

/// Additive glyph attractor compute pass. Owns the packed SDF 3D texture,
/// the per-particle tag buffer, the slot uniform buffer, and its params.
pub struct GlyphAttractor {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    tex_view: wgpu::TextureView,
    tags_buf: wgpu::Buffer,
    slots_buf: wgpu::Buffer,
    params_buf: wgpu::Buffer,
    particle_count: u32,
    res_xy: u32,
    res_z: u32,
    glyph_count: u32,
    cooldown_buf: wgpu::Buffer,
    slot_fill_buf: wgpu::Buffer,
    slot_start_buf: wgpu::Buffer,
    tag_params_buf: wgpu::Buffer,
    tag_bgl: wgpu::BindGroupLayout,
    clear_fill_pipeline: wgpu::ComputePipeline,
    count_fill_pipeline: wgpu::ComputePipeline,
    snapshot_fill_pipeline: wgpu::ComputePipeline,
    release_pipeline: wgpu::ComputePipeline,
    recruit_pipeline: wgpu::ComputePipeline,
}

impl GlyphAttractor {
    /// Build the attractor for `particle_count` particles using the packed
    /// glyph atlas derived from `set`.
    pub fn new(renderer: &Renderer, set: &GlyphVolumeSet, particle_count: u32) -> Self {
        let device = renderer.device().clone();
        let queue = renderer.queue().clone();
        let atlas = GlyphVolumeAtlas::from_set(set);

        // --- 3D SDF texture (R32Float, sampled via textureLoad) ---
        let mut tex = Texture::new_3d(
            "GlyphAttractor/SDF",
            atlas.res_xy,
            atlas.res_xy,
            atlas.depth(),
            wgpu::TextureFormat::R32Float,
            wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        );
        tex.initialize(&device);
        let gpu_tex = tex.gpu_texture().expect("texture created");
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: gpu_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&atlas.data),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(atlas.res_xy * 4),
                rows_per_image: Some(atlas.res_xy),
            },
            wgpu::Extent3d {
                width: atlas.res_xy,
                height: atlas.res_xy,
                depth_or_array_layers: atlas.depth(),
            },
        );
        let tex_view = gpu_tex.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D3),
            ..Default::default()
        });

        // --- buffers ---
        let tags_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/Tags"),
            size: (particle_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slots_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/Slots"),
            size: (std::mem::size_of::<GpuSlot>() * NUM_SLOTS) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/Params"),
            size: std::mem::size_of::<GpuAttractorParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // --- pipeline ---
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlyphAttractor/Shader"),
            source: wgpu::ShaderSource::Wgsl(ATTRACTOR_WGSL.into()),
        });
        let c = wgpu::ShaderStages::COMPUTE;
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding, visibility: c, ty, count: None,
        };
        let storage = |ro: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: ro },
            has_dynamic_offset: false, min_binding_size: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GlyphAttractor/BGL"),
            entries: &[
                entry(0, storage(true)),
                entry(1, storage(false)),
                entry(2, storage(true)),
                entry(3, uniform.clone()),
                entry(4, uniform.clone()),
                entry(5, wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D3,
                    multisampled: false,
                }),
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GlyphAttractor/PL"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GlyphAttractor/Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // --- tagging passes (release/recruit) ---
        let cooldown_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/Cooldown"),
            size: (particle_count as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let slot_fill_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/SlotFill"),
            size: (NUM_SLOTS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let slot_start_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/SlotStart"),
            size: (NUM_SLOTS as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tag_params_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GlyphAttractor/TagParams"),
            size: std::mem::size_of::<GpuTagParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&cooldown_buf, 0, bytemuck::cast_slice(&vec![0u32; particle_count as usize]));

        let tag_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GlyphAttractor/TaggerShader"),
            source: wgpu::ShaderSource::Wgsl(TAGGER_WGSL.into()),
        });
        let tag_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GlyphAttractor/TagBGL"),
            entries: &[
                entry(0, storage(false)), // positions: written when emitting
                entry(1, storage(false)),
                entry(2, storage(false)),
                entry(3, storage(false)),
                entry(4, uniform.clone()),
                entry(5, uniform.clone()),
                entry(6, storage(false)), // velocities: zeroed when emitting
                entry(7, storage(false)), // slot_start
            ],
        });
        let tag_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GlyphAttractor/TagPL"),
            bind_group_layouts: &[&tag_bgl],
            push_constant_ranges: &[],
        });
        let mk = |ep: &str| device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GlyphAttractor/TagPipeline"),
            layout: Some(&tag_layout),
            module: &tag_module,
            entry_point: Some(ep),
            compilation_options: Default::default(),
            cache: None,
        });
        let clear_fill_pipeline = mk("clear_fill");
        let count_fill_pipeline = mk("count_fill");
        let snapshot_fill_pipeline = mk("snapshot_fill");
        let release_pipeline = mk("release");
        let recruit_pipeline = mk("recruit");

        GlyphAttractor {
            device, queue, pipeline, bgl, tex_view,
            tags_buf, slots_buf, params_buf,
            particle_count,
            res_xy: atlas.res_xy, res_z: atlas.res_z, glyph_count: atlas.glyph_count,
            cooldown_buf, slot_fill_buf, slot_start_buf, tag_params_buf, tag_bgl,
            clear_fill_pipeline, count_fill_pipeline, snapshot_fill_pipeline, release_pipeline, recruit_pipeline,
        }
    }

    /// The per-particle tag buffer (`array<i32>`), for readback in tests/tools.
    pub fn tags_buffer(&self) -> &wgpu::Buffer {
        &self.tags_buf
    }

    /// Upload per-particle slot tags (`-1` = unattracted, else slot 0..NUM_SLOTS-1).
    pub fn set_tags(&self, tags: &[i32]) {
        assert_eq!(
            tags.len() as u32, self.particle_count,
            "set_tags: expected {} tags, got {}", self.particle_count, tags.len()
        );
        self.queue.write_buffer(&self.tags_buf, 0, bytemuck::cast_slice(tags));
    }

    /// Upload the current slot layout (glyph ids + world boxes).
    pub fn set_slots(&self, layout: &SlotLayout) {
        let gpu: Vec<GpuSlot> = layout.slots.iter().map(GpuSlot::from).collect();
        self.queue.write_buffer(&self.slots_buf, 0, bytemuck::cast_slice(&gpu));
    }

    /// Update per-frame parameters.
    /// `dt` is the sim time advanced this frame. Gravity for tagged particles
    /// is disabled inside the solver (velocity.w "held" flag), not here.
    pub fn set_params(&self, dt: f32, stiffness: f32, max_speed: f32, basin_strength: f32, target: f32, drag: f32) {
        let p = GpuAttractorParams {
            res_xy: self.res_xy,
            res_z: self.res_z,
            glyph_count: self.glyph_count,
            stiffness,
            dt,
            max_speed,
            basin_strength,
            target,
            drag,
            _pad: [0.0; 3],
        };
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&p));
    }

    /// Encode the attractor compute pass. Reads `positions`, read-writes `velocities`.
    /// Call AFTER `FluidSimulation::update_batched`, in a separate encoder.
    pub fn dispatch(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
    ) {
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GlyphAttractor/BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.tags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.slots_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: wgpu::BindingResource::TextureView(&self.tex_view) },
            ],
        });
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GlyphAttractor/Pass"),
            timestamp_writes: crate::profiling::gpu_pass("GlyphAttractor/Pass").as_ref().map(crate::profiling::PassStamp::compute),
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg = (self.particle_count + 63) / 64;
        cpass.dispatch_workgroups(wg, 1, 1);
    }

    /// Run the GPU tagging passes: (re)count committed particles, release the
    /// changed slots, and recruit untagged particles in each slot's capture
    /// region up to its budget (optionally emitting them above the glyph).
    /// Call once per frame BEFORE `dispatch`. `velocities` is only written
    /// when `p.emit_height > 0`.
    pub fn retag(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        positions: &wgpu::Buffer,
        velocities: &wgpu::Buffer,
        p: &RetagParams,
    ) {
        let gp = GpuTagParams {
            changed_mask: p.changed_mask,
            per_slot_count: p.per_slot_count,
            cooldown_frames: p.cooldown_frames,
            capture_scale: p.capture_scale,
            capture_below: p.capture_below,
            emit_height: p.emit_height,
            emit_spread: p.emit_spread,
            emit_rate: p.emit_rate,
        };
        self.queue.write_buffer(&self.tag_params_buf, 0, bytemuck::bytes_of(&gp));

        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GlyphAttractor/TagBG"),
            layout: &self.tag_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: positions.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.tags_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cooldown_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.slot_fill_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: self.slots_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.tag_params_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: velocities.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 7, resource: self.slot_start_buf.as_entire_binding() },
            ],
        });
        let pwg = (self.particle_count + 63) / 64;
        let mut cp = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("GlyphAttractor/Tagging"), timestamp_writes: crate::profiling::gpu_pass("GlyphAttractor/Tagging").as_ref().map(crate::profiling::PassStamp::compute),
        });
        cp.set_bind_group(0, &bg, &[]);
        // Deliberately ONE compute pass for all four dispatches: wgpu inserts the
        // needed memory barriers between consecutive writable-storage dispatches
        // within a pass, so release→clear→count→recruit is correctly ordered.
        // (This is intentionally different from the multi-pass ComputeBatch style.)
        // release keys off each particle's current tag (not slot_fill), so it runs
        // first; then clear+count recompute slot_fill for the remaining committed
        // particles so recruit only tops slots up to per_slot_count.
        cp.set_pipeline(&self.release_pipeline);     cp.dispatch_workgroups(pwg, 1, 1);
        cp.set_pipeline(&self.clear_fill_pipeline);  cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.count_fill_pipeline);  cp.dispatch_workgroups(pwg, 1, 1);
        cp.set_pipeline(&self.snapshot_fill_pipeline); cp.dispatch_workgroups(1, 1, 1);
        cp.set_pipeline(&self.recruit_pipeline);     cp.dispatch_workgroups(pwg, 1, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::{FontAtlas, GlyphVolumeSet};

    const FONT: &[u8] = include_bytes!("../../../tests/fixtures/L10-medium.arfont");

    #[test]
    fn atlas_dimensions_and_layout() {
        let font = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
        let atlas = GlyphVolumeAtlas::from_set(&set);
        assert_eq!(atlas.res_xy, 32);
        assert_eq!(atlas.res_z, 8);
        assert_eq!(atlas.glyph_count, 11);
        assert_eq!(atlas.depth(), 8 * 11);
        assert_eq!(atlas.data.len(), (32 * 32 * 8 * 11) as usize);
    }

    #[test]
    fn digit_slab_matches_source_volume() {
        let font = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&font, 32, 8, 0.5);
        let atlas = GlyphVolumeAtlas::from_set(&set);
        // Digit '3' is glyph index 3; its slab in the atlas must equal its source volume.
        let three = set.volume_for_digit(3).unwrap();
        let slab = (32 * 32 * 8) as usize;
        let base = 3 * slab;
        assert_eq!(&atlas.data[base..base + slab], &three.data[..]);
    }

    #[test]
    fn slot_layout_places_eight_slots_with_colons() {
        // HH:MM:SS → 8 slots; indices 2 and 5 are the colons.
        let layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        assert_eq!(layout.slots.len(), NUM_SLOTS);
        // Colons are fixed to the ':' glyph id (10) and never change.
        assert_eq!(layout.slots[2].glyph_id, 10);
        assert_eq!(layout.slots[5].glyph_id, 10);
        // Slots are laid out left-to-right in ascending world X.
        for i in 1..NUM_SLOTS {
            assert!(layout.slots[i].world_min[0] > layout.slots[i - 1].world_min[0]);
        }
    }

    #[test]
    fn set_time_maps_digits_to_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        layout.set_time(12, 34, 56);
        // Digit slots (0,1,3,4,6,7) carry the time digits; colons stay 10.
        assert_eq!(layout.slots[0].glyph_id, 1);
        assert_eq!(layout.slots[1].glyph_id, 2);
        assert_eq!(layout.slots[3].glyph_id, 3);
        assert_eq!(layout.slots[4].glyph_id, 4);
        assert_eq!(layout.slots[6].glyph_id, 5);
        assert_eq!(layout.slots[7].glyph_id, 6);
        assert_eq!(layout.slots[2].glyph_id, 10);
    }

    #[test]
    fn set_time_wraps_and_clamps() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        layout.set_time(9, 5, 0); // 09:05:00
        assert_eq!(layout.slots[0].glyph_id, 0);
        assert_eq!(layout.slots[1].glyph_id, 9);
        assert_eq!(layout.slots[3].glyph_id, 0);
        assert_eq!(layout.slots[4].glyph_id, 5);
    }

    #[test]
    fn gpu_slot_packing_is_std140_sized() {
        // Each GPU slot is 3 × vec4 = 48 bytes (world_min+pad, world_size+pad, glyph_id+pad).
        assert_eq!(std::mem::size_of::<GpuSlot>(), 48);
        let slot = AttractorSlot { glyph_id: 7, world_min: [1.0, 2.0, 3.0], world_size: [4.0, 5.0, 6.0], budget: 0 };
        let g = GpuSlot::from(&slot);
        assert_eq!(g.world_min, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(g.world_size, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(g.glyph_id, 7);
    }

    #[test]
    fn gpu_params_packing_is_sized() {
        // res_xy, res_z, glyph_count, stiffness | dt, max_speed, basin_strength, _pad
        assert_eq!(std::mem::size_of::<GpuAttractorParams>(), 48);
    }
}
