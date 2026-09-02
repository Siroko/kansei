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
