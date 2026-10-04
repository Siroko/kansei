// Text layout module — builds flat particle arrays from lyrics lines and the font atlas.

use kansei_core::sdf::{layout_line, FontAtlas};

/// Flat arrays ready for GPU buffer upload.
pub struct LyricsParticleData {
    pub total_particles: u32,
    pub total_lines: u32,
    pub positions: Vec<f32>,      // P * 4 (x, y, z, 1.0)  — base positions
    pub atlas_rects: Vec<f32>,    // P * 4 (GlyphRects::atlas)
    pub plane_rects: Vec<f32>,    // P * 4 (GlyphRects::plane, font_size world units per em)
    pub colors: Vec<f32>,         // P * 4 (rgba) — all white
    pub line_meta: Vec<u32>,      // P * 4 (line_idx, char_idx_in_line, chars_in_line, 0)
    pub line_timestamps: Vec<f32>, // per-line timestamps
    pub max_chars_per_line: u32,
    /// Half-width of the widest line (for wave mesh extent)
    pub line_half_width: f32,
}

/// Build flat particle arrays for all lyrics lines.
/// Layout: each line is a horizontal row, lines stacked vertically downward.
/// Text is centered horizontally around x=0.
pub fn build_lyrics_particles(
    lines: &[String],
    timestamps: &[f32],
    atlas: &FontAtlas,
    font_size: f32,
    line_spacing: f32,
) -> LyricsParticleData {
    // One glyph per visible character (spaces only advance; no hyphens: the geometric lines
    // frame the text)
    let placed: Vec<_> = lines.iter().map(|line| layout_line(atlas, line, font_size, 1.0)).collect();

    let total_particles = placed.iter().map(Vec::len).sum::<usize>() as u32;
    let max_chars_per_line = placed.iter().map(Vec::len).max().unwrap_or(1) as u32;
    let total_lines = lines.len() as u32;
    let p = total_particles as usize;

    let mut positions = Vec::with_capacity(p * 4);
    let mut atlas_rects = Vec::with_capacity(p * 4);
    let mut plane_rects = Vec::with_capacity(p * 4);
    let mut colors = Vec::with_capacity(p * 4);
    let mut line_meta = Vec::with_capacity(p * 4);
    let mut max_line_width: f32 = 0.0;

    // Vertical layout: reverse order — first JSON line at the bottom,
    // last JSON line at the top. Centered at origin.
    let total_height = (lines.len() as f32 - 1.0) * line_spacing;
    let y_offset = total_height * 0.5;
    let last_line = lines.len() as f32 - 1.0;

    for (line_idx, (line, glyphs)) in lines.iter().zip(&placed).enumerate() {
        // The line's full advance (what layout_line moves through), for centering
        let line_width: f32 = line.chars().map(|c| atlas.glyph(c).map_or(0.5, |g| g.advance)).sum::<f32>() * font_size;
        let start_x = -line_width * 0.5;
        // Y position: line 0 (first in JSON) at bottom, last at top
        let y = -(last_line - line_idx as f32) * line_spacing + y_offset;
        max_line_width = max_line_width.max(line_width);

        for (char_idx_in_line, glyph) in glyphs.iter().enumerate() {
            positions.extend_from_slice(&[start_x + glyph.x, y, 0.0, 1.0]);
            atlas_rects.extend_from_slice(&glyph.rects.atlas);
            plane_rects.extend_from_slice(&glyph.rects.plane);
            colors.extend_from_slice(&[1.0, 1.0, 1.0, 1.0]);
            line_meta.extend_from_slice(&[line_idx as u32, char_idx_in_line as u32, glyphs.len() as u32, 0]);
        }
    }

    // Wave line half-width: extend beyond widest text + 40% padding
    let line_half_width = max_line_width * 0.5 + max_line_width * 0.4;

    LyricsParticleData {
        total_particles,
        total_lines,
        positions,
        atlas_rects,
        plane_rects,
        colors,
        line_meta,
        line_timestamps: timestamps.to_vec(),
        max_chars_per_line,
        line_half_width,
    }
}
