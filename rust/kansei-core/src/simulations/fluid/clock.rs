//! CPU clock logic for the fluid clock: map wall-clock time onto the 8 glyph
//! slots, and report which digit slots changed (as indices and as a GPU bitmask).
//! All particle tagging happens on the GPU; this module only drives the slot
//! layout and the change signal.

use crate::simulations::fluid::{SlotLayout, NUM_SLOTS};

/// Tracks the currently-displayed glyph per slot so digit changes are detected.
pub struct ClockState {
    current: [i32; NUM_SLOTS],
}

impl ClockState {
    pub fn new() -> Self {
        ClockState { current: [-1; NUM_SLOTS] }
    }

    /// Apply `h:m:s` to `layout` and return the digit-slot indices whose glyph
    /// changed since the last call (colon slots are static and never reported).
    pub fn update(&mut self, layout: &mut SlotLayout, h: u32, m: u32, s: u32) -> Vec<usize> {
        layout.set_time(h, m, s);
        let mut changed = Vec::new();
        for i in 0..NUM_SLOTS {
            let g = layout.slots[i].glyph_id;
            if g != self.current[i] {
                if (0..=9).contains(&g) {
                    changed.push(i);
                }
                self.current[i] = g;
            }
        }
        changed
    }

    /// Bitmask form of a changed-slot list: bit `i` set = slot `i` changed.
    /// Uploaded to the GPU release pass.
    pub fn changed_mask(changed: &[usize]) -> u32 {
        let mut mask = 0u32;
        for &i in changed {
            mask |= 1u32 << i;
        }
        mask
    }
}

impl Default for ClockState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simulations::fluid::SlotLayout;

    #[test]
    fn first_update_reports_all_six_digit_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        let changed = clock.update(&mut layout, 12, 34, 56);
        assert_eq!(changed, vec![0, 1, 3, 4, 6, 7]); // colons (2,5) excluded
    }

    #[test]
    fn only_changed_digit_is_reported() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 56);
        assert_eq!(clock.update(&mut layout, 12, 34, 57), vec![7]);
    }

    #[test]
    fn minute_rollover_changes_multiple_slots() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 12, 34, 59);
        let mut changed = clock.update(&mut layout, 12, 35, 0);
        changed.sort();
        assert_eq!(changed, vec![4, 6, 7]);
    }

    #[test]
    fn no_change_reports_empty() {
        let mut layout = SlotLayout::hh_mm_ss(2.0, 1.0);
        let mut clock = ClockState::new();
        clock.update(&mut layout, 1, 2, 3);
        assert_eq!(clock.update(&mut layout, 1, 2, 3), Vec::<usize>::new());
    }

    #[test]
    fn changed_mask_sets_the_right_bits() {
        assert_eq!(ClockState::changed_mask(&[0, 3, 7]), 0b1000_1001);
        assert_eq!(ClockState::changed_mask(&[]), 0);
    }
}
