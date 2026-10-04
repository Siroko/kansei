//! ```no_run
//! use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
//! let bytes: &[u8] = &[]; // load your .arfont
//! if let Ok(atlas) = FontAtlas::parse(bytes) {
//!     let set = GlyphVolumeSet::for_clock(&atlas, 32, 8, 0.5);
//!     let _five = set.volume_for_digit(5);
//! }
//! ```
//! Signed-distance-field typography: parse `.arfont` MTSDF atlases and build
//! extruded 3D SDF volumes for glyphs. CPU-only; no GPU or JS dependency.

mod arfont;
mod glyph_volume;
mod text;

pub use arfont::{FontAtlas, GlyphMetrics, ArFontError};
pub use text::{layout_line, GlyphRects, MsdfTextOptions, PlacedGlyph, MSDF_TEXT_WGSL};
pub use glyph_volume::{GlyphVolume, GlyphVolumeSet, GlyphSdf2d, crop_glyph_sdf};
