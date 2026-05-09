// GlyphData parsing — convert JS glyph JSON into GPU buffer data.

use std::collections::HashMap;
use serde::Deserialize;

/// Per-glyph metrics deserialized from MSDF font atlas JSON.
#[derive(Deserialize, Clone)]
pub struct GlyphMetrics {
    pub codepoint: u32,
    pub advance: f32,
    pub image_bounds: [f32; 4], // [left, top, right, bottom] in UV space
    pub plane_bounds: [f32; 4], // [left, top, right, bottom] in pixel space
}

/// Flat arrays ready for GPU buffer upload.
pub struct ParticleData {
    pub total_particles: u32,
    pub total_words: u32,
    pub positions: Vec<f32>,    // P * 4 (x, y, z, 1.0)
    pub velocities: Vec<f32>,   // P * 4 (vx, vy, vz, 0.0)
    pub image_bounds: Vec<f32>, // P * 4 (MSDF UV rect)
    pub plane_bounds: Vec<f32>, // P * 4 (glyph pixel rect, scaled by font_size)
    pub colors: Vec<f32>,       // P * 4 (rgba)
    pub word_meta: Vec<u32>,    // P * 4 (word_id, letter_idx, word_len, particle_offset)
    pub rest_lengths: Vec<f32>, // P (verlet rest distance to previous letter)
}

/// Build flat particle arrays from word list and glyph metrics.
pub fn build_particle_data(
    words: &[String],
    glyphs: &[GlyphMetrics],
    font_size: f32,
    palette: &[[f32; 4]],
    bounds_size: f32,
    letter_spacing: f32,
) -> ParticleData {
    // Build codepoint → glyph lookup
    let glyph_map: HashMap<u32, &GlyphMetrics> =
        glyphs.iter().map(|g| (g.codepoint, g)).collect();

    // Collect chars per word (filter out spaces for multi-word phrases)
    let word_chars: Vec<Vec<char>> = words
        .iter()
        .map(|w| w.chars().filter(|c| *c != ' ').collect())
        .collect();

    let total_particles: u32 = word_chars.iter().map(|wc| wc.len() as u32).sum();
    let total_words = words.len() as u32;
    let p = total_particles as usize;

    let mut positions = Vec::with_capacity(p * 4);
    let mut velocities = Vec::with_capacity(p * 4);
    let mut image_bounds = Vec::with_capacity(p * 4);
    let mut plane_bounds = Vec::with_capacity(p * 4);
    let mut colors = Vec::with_capacity(p * 4);
    let mut word_meta = Vec::with_capacity(p * 4);
    let mut rest_lengths = Vec::with_capacity(p);

    // Simple xorshift RNG
    let mut rng: u64 = 42;
    let rand_f = |rng: &mut u64| -> f32 {
        *rng ^= *rng << 13;
        *rng ^= *rng >> 7;
        *rng ^= *rng << 17;
        (*rng as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    let zeroed_bounds: [f32; 4] = [0.0; 4];
    let mut global_particle_idx: u32 = 0;

    for (word_id, chars) in word_chars.iter().enumerate() {
        let word_len = chars.len() as u32;
        // The first particle of this word — used by verlet to find the chain
        let word_start_offset = global_particle_idx;

        // Random starting position for this word
        let spread = bounds_size * 0.8;
        let ox = rand_f(&mut rng) * spread;
        let oy = rand_f(&mut rng) * spread;
        let oz = rand_f(&mut rng) * spread;

        for (letter_idx, ch) in chars.iter().enumerate() {
            let cp = *ch as u32;
            let glyph = glyph_map.get(&cp);

            // Position — all letters start at word origin
            positions.extend_from_slice(&[ox, oy, oz, 1.0]);

            // Velocity — zero
            velocities.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);

            // Image bounds (UV rect)
            let ib = glyph.map_or(&zeroed_bounds, |g| &g.image_bounds);
            image_bounds.extend_from_slice(ib);

            // Plane bounds scaled by font_size
            let pb = glyph.map_or(&zeroed_bounds, |g| &g.plane_bounds);
            plane_bounds.extend_from_slice(&[
                pb[0] * font_size,
                pb[1] * font_size,
                pb[2] * font_size,
                pb[3] * font_size,
            ]);

            // Color — cycle through palette per word
            let word_color = palette[word_id % palette.len()];
            colors.extend_from_slice(&word_color);

            // Word meta: word_start_offset is the WORD's first particle index
            // (not per-particle). Verlet uses: prevIdx = word_start_offset + letter_idx - 1.
            word_meta.extend_from_slice(&[
                word_id as u32,
                letter_idx as u32,
                word_len,
                word_start_offset,
            ]);

            // Rest length — distance constraint to previous letter.
            //
            // Computing proper spacing per pair:
            //   1. prev glyph's right edge (plane_bounds[2]) = how far it extends
            //   2. curr glyph's left edge (plane_bounds[0]) = where it starts (can be negative)
            //   3. The non-overlapping distance = prev_right - curr_left + gap
            //
            // This accounts for actual glyph shapes: 'm' needs more space than 'i',
            // and 'f' followed by 'i' can be tighter than 'w' followed by 'm'.
            // Multiplied by letter_spacing for user-tunable breathing room.
            if letter_idx == 0 {
                rest_lengths.push(0.0);
            } else {
                let prev_ch = chars[letter_idx - 1] as u32;
                let prev_glyph = glyph_map.get(&prev_ch);
                // Use previous glyph's advance — the typographically correct
                // distance between consecutive glyph origins.
                let prev_advance = prev_glyph.map_or(0.5, |g| g.advance);
                rest_lengths.push(prev_advance * font_size * letter_spacing);
            }

            global_particle_idx += 1;
        }
    }

    ParticleData {
        total_particles,
        total_words,
        positions,
        velocities,
        image_bounds,
        plane_bounds,
        colors,
        word_meta,
        rest_lengths,
    }
}
