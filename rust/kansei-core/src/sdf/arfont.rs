//! Parser for the Artery Font Format (`.arfont`) MTSDF atlases.

/// Errors that can occur while parsing a `.arfont` file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ArFontError {
    TooShort,
    BadMagic,
    UnsupportedVersion(u32),
    UnsupportedRealType(u32),
    NoImage,
    ImageDecode,
}

/// Per-glyph metrics recovered from the atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphMetrics {
    pub codepoint: u32,
    pub advance: f32,
    /// [left, bottom, right, top] in atlas pixels.
    pub image_bounds: [f32; 4],
    /// [left, bottom, right, top] in em space.
    pub plane_bounds: [f32; 4],
}

/// Decoded atlas image plus glyph metrics.
pub struct FontAtlas {
    pub width: u32,
    pub height: u32,
    /// RGBA8, row-major, `width * height * 4` bytes. Alpha channel is the SDF.
    pub rgba: Vec<u8>,
    pub glyphs: Vec<GlyphMetrics>,
    pub distance_range: f32,
    pub em_size: f32,
}
