//! Signed-distance-field typography: parse `.arfont` MTSDF atlases and build
//! extruded 3D SDF volumes for glyphs. CPU-only; no GPU or JS dependency.

mod arfont;
mod glyph_volume;

pub use arfont::{FontAtlas, GlyphMetrics, ArFontError};
// pub use glyph_volume::{GlyphVolume, GlyphVolumeSet}; // (defined in Task 6/7)
