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
    pub _pad: [i32; 3],
}

impl From<&AttractorSlot> for GpuSlot {
    fn from(s: &AttractorSlot) -> Self {
        GpuSlot {
            world_min: [s.world_min[0], s.world_min[1], s.world_min[2], 0.0],
            world_size: [s.world_size[0], s.world_size[1], s.world_size[2], 0.0],
            glyph_id: s.glyph_id,
            _pad: [0; 3],
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
    pub _pad: f32,
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
}

impl AttractorSlot {
    fn empty() -> Self {
        AttractorSlot { glyph_id: -1, world_min: [0.0; 3], world_size: [1.0; 3] }
    }
}

/// The 8 clock slots laid out left-to-right in world space.
pub struct SlotLayout {
    pub slots: [AttractorSlot; NUM_SLOTS],
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
            };
        }
        SlotLayout { slots }
    }

    /// Set the displayed time. Digit slots get the time digits; colon slots stay `:`.
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
            self.slots[c].glyph_id = COLON_GLYPH_ID;
        }
    }
}

const ATTRACTOR_WGSL: &str = r#"
struct Slot {
    world_min: vec4<f32>,
    world_size: vec4<f32>,
    glyph_id: i32,
    _pad0: i32,
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
    _pad: f32,
};

@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> tags: array<i32>;
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
    if (slot_id < 0) { return; }

    let slot = slots[slot_id];
    if (slot.glyph_id < 0) { return; }

    let pos = positions[idx].xyz;
    let local = (pos - slot.world_min.xyz) / slot.world_size.xyz; // [0,1] inside box
    let inside_box = all(local >= vec3<f32>(0.0)) && all(local <= vec3<f32>(1.0));

    var force = vec3<f32>(0.0);
    if (inside_box) {
        // Voxel coordinate in the glyph's local grid.
        let rx = f32(params.res_xy);
        let rz = f32(params.res_z);
        let vx = i32(local.x * (rx - 1.0));
        let vy = i32(local.y * (rx - 1.0));
        let vz = i32(local.z * (rz - 1.0));
        // SDF gradient via central differences (points toward increasing SDF = inside).
        let gx = load_sdf(slot.glyph_id, vx + 1, vy, vz) - load_sdf(slot.glyph_id, vx - 1, vy, vz);
        let gy = load_sdf(slot.glyph_id, vx, vy + 1, vz) - load_sdf(slot.glyph_id, vx, vy - 1, vz);
        let gz = load_sdf(slot.glyph_id, vx, vy, vz + 1) - load_sdf(slot.glyph_id, vx, vy, vz - 1);
        let grad = vec3<f32>(gx, gy, gz);
        let s = load_sdf(slot.glyph_id, vx, vy, vz);
        // Pull toward the surface/interior: if outside (s<0) climb the gradient.
        if (length(grad) > 1e-5) {
            force = normalize(grad) * params.stiffness * max(-s, 0.0);
        }
    } else {
        // Coarse basin: pull toward the box center.
        let center = slot.world_min.xyz + 0.5 * slot.world_size.xyz;
        let to_center = center - pos;
        force = to_center * params.basin_strength;
    }

    var vel = velocities[idx].xyz + force * params.dt;
    let sp = length(vel);
    if (sp > params.max_speed) { vel = vel / sp * params.max_speed; }
    velocities[idx] = vec4<f32>(vel, velocities[idx].w);
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
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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
                entry(3, uniform),
                entry(4, uniform),
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

        GlyphAttractor {
            device, queue, pipeline, bgl, tex_view,
            tags_buf, slots_buf, params_buf,
            particle_count,
            res_xy: atlas.res_xy, res_z: atlas.res_z, glyph_count: atlas.glyph_count,
        }
    }

    /// Upload per-particle slot tags (`-1` = unattracted, else slot 0..NUM_SLOTS-1).
    pub fn set_tags(&self, tags: &[i32]) {
        debug_assert_eq!(tags.len() as u32, self.particle_count);
        self.queue.write_buffer(&self.tags_buf, 0, bytemuck::cast_slice(tags));
    }

    /// Upload the current slot layout (glyph ids + world boxes).
    pub fn set_slots(&self, layout: &SlotLayout) {
        let gpu: Vec<GpuSlot> = layout.slots.iter().map(GpuSlot::from).collect();
        self.queue.write_buffer(&self.slots_buf, 0, bytemuck::cast_slice(&gpu));
    }

    /// Update per-frame parameters.
    pub fn set_params(&self, dt: f32, stiffness: f32, max_speed: f32, basin_strength: f32) {
        let p = GpuAttractorParams {
            res_xy: self.res_xy,
            res_z: self.res_z,
            glyph_count: self.glyph_count,
            stiffness,
            dt,
            max_speed,
            basin_strength,
            _pad: 0.0,
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
            timestamp_writes: None,
        });
        cpass.set_pipeline(&self.pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        let wg = (self.particle_count + 63) / 64;
        cpass.dispatch_workgroups(wg, 1, 1);
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
        let slot = AttractorSlot { glyph_id: 7, world_min: [1.0, 2.0, 3.0], world_size: [4.0, 5.0, 6.0] };
        let g = GpuSlot::from(&slot);
        assert_eq!(g.world_min, [1.0, 2.0, 3.0, 0.0]);
        assert_eq!(g.world_size, [4.0, 5.0, 6.0, 0.0]);
        assert_eq!(g.glyph_id, 7);
    }

    #[test]
    fn gpu_params_packing_is_sized() {
        // res_xy, res_z, glyph_count, stiffness | dt, max_speed, basin_strength, _pad
        assert_eq!(std::mem::size_of::<GpuAttractorParams>(), 32);
    }
}
