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

/// Little-endian u32 read helper.
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Little-endian f32 read helper.
fn rd_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Parsed `ArteryFontHeader` plus the computed block offsets.
#[derive(Debug, PartialEq)]
pub(crate) struct ArFontHeader {
    pub variant_count: u32,
    pub variants_length: u32,
    pub variants_offset: usize,
    pub image_count: u32,
    pub images_length: u32,
    pub images_offset: usize,
}

impl ArFontHeader {
    /// Header is 112 bytes; blocks follow in order metadata → variants → images.
    pub fn parse(buf: &[u8]) -> Result<ArFontHeader, ArFontError> {
        if buf.len() < 112 {
            return Err(ArFontError::TooShort);
        }
        if &buf[0..11] != b"ARTERY/FONT" {
            return Err(ArFontError::BadMagic);
        }
        if rd_u32(buf, 16) != 0x4D27_6A5C {
            return Err(ArFontError::BadMagic);
        }
        let version = rd_u32(buf, 20);
        if version != 1 {
            return Err(ArFontError::UnsupportedVersion(version));
        }
        let real_type = rd_u32(buf, 28);
        if real_type != 0x14 {
            return Err(ArFontError::UnsupportedRealType(real_type));
        }
        let metadata_length = rd_u32(buf, 0x34) as usize;
        let variant_count = rd_u32(buf, 0x38);
        let variants_length = rd_u32(buf, 0x3C);
        let image_count = rd_u32(buf, 0x40);
        let images_length = rd_u32(buf, 0x44);

        let variants_offset = 112 + metadata_length;
        let images_offset = variants_offset + variants_length as usize;

        Ok(ArFontHeader {
            variant_count,
            variants_length,
            variants_offset,
            image_count,
            images_length,
            images_offset,
        })
    }
}

#[cfg(test)]
mod header_tests {
    use super::*;

    fn synthetic_header() -> Vec<u8> {
        let mut b = vec![0u8; 112];
        b[0..12].copy_from_slice(b"ARTERY/FONT\0");
        b[16..20].copy_from_slice(&0x4D276A5Cu32.to_le_bytes()); // magicNo
        b[20..24].copy_from_slice(&1u32.to_le_bytes());          // version
        b[28..32].copy_from_slice(&0x14u32.to_le_bytes());       // realType (f32)
        b[0x38..0x3C].copy_from_slice(&1u32.to_le_bytes());      // variantCount
        b[0x3C..0x40].copy_from_slice(&0x1290u32.to_le_bytes()); // variantsLength
        b[0x40..0x44].copy_from_slice(&1u32.to_le_bytes());      // imageCount
        b[0x44..0x48].copy_from_slice(&0x1CCD0u32.to_le_bytes());// imagesLength
        b
    }

    #[test]
    fn parses_valid_header() {
        let h = ArFontHeader::parse(&synthetic_header()).unwrap();
        assert_eq!(h.variant_count, 1);
        assert_eq!(h.variants_length, 0x1290);
        assert_eq!(h.image_count, 1);
        assert_eq!(h.images_length, 0x1CCD0);
        assert_eq!(h.variants_offset, 112); // metadataLength == 0
        assert_eq!(h.images_offset, 112 + 0x1290);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut b = synthetic_header();
        b[16] ^= 0xFF;
        assert_eq!(ArFontHeader::parse(&b), Err(ArFontError::BadMagic));
    }

    #[test]
    fn rejects_short_input() {
        assert_eq!(ArFontHeader::parse(&[0u8; 8]), Err(ArFontError::TooShort));
    }
}
