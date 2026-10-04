// Particle data: one particle per letter, each word a chain whose first letter steers (the
// vehicle) and whose others follow it at their glyph advances.

use kansei_core::sdf::{FontAtlas, GlyphRects};

/// Flat arrays ready for GPU buffer upload.
pub struct ParticleData {
    pub total_particles: u32,
    pub total_words: u32,
    pub positions: Vec<f32>,    // P * 4 (x, y, z, 1.0)
    pub velocities: Vec<f32>,   // P * 4 (vx, vy, vz, 0.0)
    pub atlas_rects: Vec<f32>,  // P * 4 (GlyphRects::atlas)
    pub plane_rects: Vec<f32>,  // P * 4 (GlyphRects::plane, font_size world units per em)
    pub colors: Vec<f32>,       // P * 4 (rgba)
    pub word_meta: Vec<u32>,    // P * 4 (word_id, letter_idx, word_len, word's first particle)
    pub rest_lengths: Vec<f32>, // P (verlet rest distance to previous letter)
}

/// Per-particle colours: each word takes the palette entry of its id.
pub fn word_colors(word_meta: &[u32], palette: &[[f32; 4]]) -> Vec<f32> {
    word_meta.chunks_exact(4).flat_map(|meta| palette[meta[0] as usize % palette.len()]).collect()
}

/// Build flat particle arrays from the word list and the font atlas's metrics.
pub fn build_particle_data(
    words: &[String],
    atlas: &FontAtlas,
    font_size: f32,
    palette: &[[f32; 4]],
    bounds_size: f32,
    letter_spacing: f32,
) -> ParticleData {
    // Letters per word (spaces in multi-word phrases dropped)
    let word_chars: Vec<Vec<char>> = words
        .iter()
        .map(|w| w.chars().filter(|c| *c != ' ').collect())
        .collect();

    let total_particles: u32 = word_chars.iter().map(|wc| wc.len() as u32).sum();
    let total_words = words.len() as u32;
    let p = total_particles as usize;

    let mut positions = Vec::with_capacity(p * 4);
    let mut velocities = Vec::with_capacity(p * 4);
    let mut atlas_rects = Vec::with_capacity(p * 4);
    let mut plane_rects = Vec::with_capacity(p * 4);
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

    // a character the atlas lacks draws an empty quad
    let empty = GlyphRects { atlas: [0.0; 4], plane: [0.0; 4] };
    let mut global_particle_idx: u32 = 0;

    for (word_id, chars) in word_chars.iter().enumerate() {
        let word_len = chars.len() as u32;
        // The first particle of this word: verlet finds the chain from it
        let word_start_offset = global_particle_idx;

        // Random starting position for this word
        let spread = bounds_size * 0.8;
        let ox = rand_f(&mut rng) * spread;
        let oy = rand_f(&mut rng) * spread;
        let oz = rand_f(&mut rng) * spread;

        for (letter_idx, ch) in chars.iter().enumerate() {
            let glyph = atlas.glyph(*ch);

            // All letters start at the word's origin, at rest
            positions.extend_from_slice(&[ox, oy, oz, 1.0]);
            velocities.extend_from_slice(&[0.0, 0.0, 0.0, 0.0]);

            let rects = glyph.map_or(empty, |g| atlas.glyph_rects(g, font_size));
            atlas_rects.extend_from_slice(&rects.atlas);
            plane_rects.extend_from_slice(&rects.plane);

            // Verlet's anchor is word_start_offset + letter_idx - 1
            word_meta.extend_from_slice(&[word_id as u32, letter_idx as u32, word_len, word_start_offset]);

            // Rest length to the previous letter: its advance (the typographic distance between
            // consecutive glyph origins; half an em if the atlas lacks it) times letter_spacing
            if letter_idx == 0 {
                rest_lengths.push(0.0);
            } else {
                let prev_advance = atlas.glyph(chars[letter_idx - 1]).map_or(0.5, |g| g.advance);
                rest_lengths.push(prev_advance * font_size * letter_spacing);
            }

            global_particle_idx += 1;
        }
    }

    let colors = word_colors(&word_meta, palette);
    ParticleData {
        total_particles,
        total_words,
        positions,
        velocities,
        atlas_rects,
        plane_rects,
        colors,
        word_meta,
        rest_lengths,
    }
}
