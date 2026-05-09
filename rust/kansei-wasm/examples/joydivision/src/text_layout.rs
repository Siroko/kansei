// Text layout module — builds flat particle arrays from lyrics lines and glyph metrics.

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
pub struct LyricsParticleData {
    pub total_particles: u32,
    pub total_lines: u32,
    pub positions: Vec<f32>,      // P * 4 (x, y, z, 1.0)  — base positions
    pub image_bounds: Vec<f32>,   // P * 4 (MSDF UV rect)
    pub plane_bounds: Vec<f32>,   // P * 4 (glyph pixel rect, scaled by font_size)
    pub colors: Vec<f32>,         // P * 4 (rgba) — all white
    pub line_meta: Vec<u32>,      // P * 4 (line_idx, char_idx_in_line, chars_in_line, 0)
    pub line_timestamps: Vec<f32>, // per-line timestamps
    pub max_chars_per_line: u32,
    /// Y position of each line (for wave mesh placement)
    pub line_y_positions: Vec<f32>,
    /// Half-width of the widest line (for wave mesh extent)
    pub line_half_width: f32,
}

/// Build flat particle arrays for all lyrics lines.
/// Layout: each line is a horizontal row, lines stacked vertically downward.
/// Text is centered horizontally around x=0.
pub fn build_lyrics_particles(
    lines: &[String],
    timestamps: &[f32],
    glyphs: &[GlyphMetrics],
    font_size: f32,
    line_spacing: f32,
) -> LyricsParticleData {
    // Build codepoint -> glyph lookup
    let glyph_map: HashMap<u32, &GlyphMetrics> =
        glyphs.iter().map(|g| (g.codepoint, g)).collect();

    // Space glyph advance for spacing between words
    let space_advance = glyph_map.get(&(' ' as u32))
        .map(|g| g.advance)
        .unwrap_or(0.5);

    // Characters per line (no hyphens — geometric lines handle the visual frame)
    let line_chars: Vec<Vec<char>> = lines.iter()
        .map(|l| l.chars().collect())
        .collect();

    let total_visible: usize = line_chars.iter()
        .map(|chars| chars.iter().filter(|c| **c != ' ').count())
        .sum();

    let max_chars_per_line = line_chars.iter()
        .map(|chars| chars.iter().filter(|c| **c != ' ').count())
        .max()
        .unwrap_or(1) as u32;

    let total_particles = total_visible as u32;
    let total_lines = lines.len() as u32;
    let p = total_particles as usize;

    let mut positions = Vec::with_capacity(p * 4);
    let mut image_bounds_out = Vec::with_capacity(p * 4);
    let mut plane_bounds_out = Vec::with_capacity(p * 4);
    let mut colors = Vec::with_capacity(p * 4);
    let mut line_meta = Vec::with_capacity(p * 4);
    let mut line_y_positions = Vec::with_capacity(lines.len());
    let mut max_line_width: f32 = 0.0;

    let zeroed_bounds: [f32; 4] = [0.0; 4];

    // Vertical layout: reverse order — first JSON line at the bottom,
    // last JSON line at the top. Centered at origin.
    let total_height = (line_chars.len() as f32 - 1.0) * line_spacing;
    let y_offset = total_height * 0.5;
    let last_line = line_chars.len() as f32 - 1.0;

    for (line_idx, chars) in line_chars.iter().enumerate() {
        // Compute total width of this line for centering
        let mut line_width: f32 = 0.0;
        for ch in chars.iter() {
            let cp = *ch as u32;
            if cp == ' ' as u32 {
                line_width += space_advance * font_size;
            } else {
                let advance = glyph_map.get(&cp).map(|g| g.advance).unwrap_or(0.5);
                line_width += advance * font_size;
            }
        }

        // Start x position (centered horizontally)
        let start_x = -line_width * 0.5;
        // Y position: line 0 (first in JSON) at bottom, last at top
        let y = -(last_line - line_idx as f32) * line_spacing + y_offset;
        line_y_positions.push(y);
        max_line_width = max_line_width.max(line_width);

        // Count visible (non-space) characters in this line
        let visible_count = chars.iter().filter(|c| **c != ' ').count() as u32;

        let mut cursor_x = start_x;
        let mut char_idx_in_line: u32 = 0;

        for ch in chars.iter() {
            let cp = *ch as u32;
            if cp == ' ' as u32 {
                // Just advance cursor, don't emit a particle
                cursor_x += space_advance * font_size;
                continue;
            }

            let glyph = glyph_map.get(&cp);
            let advance = glyph.map(|g| g.advance).unwrap_or(0.5);

            // Position
            positions.extend_from_slice(&[cursor_x, y, 0.0, 1.0]);

            // Image bounds (UV rect)
            let ib = glyph.map_or(&zeroed_bounds, |g| &g.image_bounds);
            image_bounds_out.extend_from_slice(ib);

            // Plane bounds scaled by font_size
            let pb = glyph.map_or(&zeroed_bounds, |g| &g.plane_bounds);
            plane_bounds_out.extend_from_slice(&[
                pb[0] * font_size,
                pb[1] * font_size,
                pb[2] * font_size,
                pb[3] * font_size,
            ]);

            // Color — white
            colors.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);

            // Line meta: (line_idx, char_idx_in_line, chars_in_line, 0)
            line_meta.extend_from_slice(&[
                line_idx as u32,
                char_idx_in_line,
                visible_count,
                0,
            ]);

            cursor_x += advance * font_size;
            char_idx_in_line += 1;
        }
    }

    // Wave line half-width: extend beyond widest text + 40% padding
    let line_half_width = max_line_width * 0.5 + max_line_width * 0.4;

    LyricsParticleData {
        total_particles,
        total_lines,
        positions,
        image_bounds: image_bounds_out,
        plane_bounds: plane_bounds_out,
        colors,
        line_meta,
        line_timestamps: timestamps.to_vec(),
        max_chars_per_line,
        line_y_positions,
        line_half_width,
    }
}
