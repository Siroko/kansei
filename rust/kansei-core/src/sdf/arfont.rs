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

/// Read a string block written by the upstream `artery-font-format` encoder:
/// when `len == 0` nothing is written; otherwise `len` bytes + 1 NUL byte,
/// padded to a 4-byte boundary. Returns the offset past the block.
fn skip_string(off: usize, len: usize) -> usize {
    if len == 0 {
        off
    } else {
        let total = len + 1; // + NUL terminator
        let padded = (total + 3) & !3usize; // 4-byte align
        off + padded
    }
}

/// One glyph record: 2×u32 + 10×f32 = 48 bytes.
/// Layout (from `artery_font::Glyph<REAL>`): codepoint, image,
/// planeBounds(l,b,r,t), imageBounds(l,b,r,t), advance(h,v).
const GLYPH_STRIDE: usize = 48;

/// Parse the glyph array out of the first variant block.
///
/// Layout (from upstream `artery-font-format`'s `FontVariantHeader`, verified
/// byte-for-byte against `tests/fixtures/L10-medium.arfont`):
/// ```text
/// u32 flags, weight, codepointType, imageType, fallbackVariant, fallbackGlyph;  // 24 bytes
/// u32 reserved[6];                                                              // 24 bytes
/// REAL metrics[32]; // fontSize, distanceRange, emSize, ascender, descender,
///                    // lineHeight, underlineY, underlineThickness,
///                    // distanceRangeMiddle, reserved[23]                       // 128 bytes
/// u32 nameLength, metadataLength, glyphCount, kernPairCount;                    // 16 bytes
/// // name bytes (if nameLength > 0): nameLength+1 bytes, padded to 4
/// // metadata bytes (if metadataLength > 0): metadataLength+1 bytes, padded to 4
/// // glyphCount × Glyph<REAL> (48 bytes each)
/// // kernPairCount × KernPair<REAL>
/// ```
fn parse_glyphs(
    buf: &[u8],
    variants_offset: usize,
) -> Result<(Vec<GlyphMetrics>, f32, f32), ArFontError> {
    let metrics_off = variants_offset + 48; // past the 6+6 u32 fixed fields
    let counts_off = metrics_off + 32 * 4; // metrics[32] REALs
    if counts_off + 16 > buf.len() {
        return Err(ArFontError::TooShort);
    }
    let distance_range = rd_f32(buf, metrics_off + 4);
    let em_size = rd_f32(buf, metrics_off + 8);

    let name_length = rd_u32(buf, counts_off) as usize;
    let metadata_length = rd_u32(buf, counts_off + 4) as usize;
    let glyph_count = rd_u32(buf, counts_off + 8) as usize;

    let mut off = counts_off + 16; // past nameLength,metadataLength,glyphCount,kernPairCount
    off = skip_string(off, name_length);
    off = skip_string(off, metadata_length);

    let mut glyphs = Vec::with_capacity(glyph_count);
    for i in 0..glyph_count {
        let g = off + i * GLYPH_STRIDE;
        if g + GLYPH_STRIDE > buf.len() {
            break;
        }
        let codepoint = rd_u32(buf, g);
        let plane_bounds = [
            rd_f32(buf, g + 8),
            rd_f32(buf, g + 12),
            rd_f32(buf, g + 16),
            rd_f32(buf, g + 20),
        ];
        let image_bounds = [
            rd_f32(buf, g + 24),
            rd_f32(buf, g + 28),
            rd_f32(buf, g + 32),
            rd_f32(buf, g + 36),
        ];
        let advance = rd_f32(buf, g + 40); // advance.horizontal
        glyphs.push(GlyphMetrics { codepoint, advance, image_bounds, plane_bounds });
    }

    Ok((glyphs, distance_range, em_size))
}

/// Decode the embedded atlas image (PNG, `encoding == 8`) to RGBA8.
fn decode_atlas_image(buf: &[u8], header: &ArFontHeader) -> Result<(u32, u32, Vec<u8>), ArFontError> {
    let img_off = header.images_offset;
    if img_off + 16 > buf.len() {
        return Err(ArFontError::ImageDecode);
    }
    // Image sub-header (verified layout, all u32):
    //   flags(+0) encoding(+4) width(+8) height(+12) channels(+16) pixelFormat(+20)
    //   imageType(+24) rowLength(+28) orientation(+32) childImages(+36) textureFlags(+40)
    //   reserved... metadataLength then dataLength immediately before the pixel data.
    let encoding = rd_u32(buf, img_off + 4);

    // The PNG stream begins at the `\x89PNG` magic within this image block. Locate it
    // robustly rather than hardcoding the sub-header size.
    const PNG_MAGIC: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
    let block_end = header.images_offset + header.images_length as usize;
    let search = &buf[img_off..block_end.min(buf.len())];
    let rel = search
        .windows(8)
        .position(|w| w == PNG_MAGIC)
        .ok_or(ArFontError::ImageDecode)?;
    let png_start = img_off + rel;

    debug_assert_eq!(encoding, 8, "expected PNG encoding");

    let dynimg = image::load_from_memory(&buf[png_start..block_end.min(buf.len())])
        .map_err(|_| ArFontError::ImageDecode)?;
    let rgba = dynimg.to_rgba8();
    let (width, height) = rgba.dimensions();
    Ok((width, height, rgba.into_raw()))
}

impl FontAtlas {
    /// Parse a `.arfont` byte buffer into a decoded atlas + glyph metrics.
    pub fn parse(buf: &[u8]) -> Result<FontAtlas, ArFontError> {
        let header = ArFontHeader::parse(buf)?;
        if header.image_count == 0 {
            return Err(ArFontError::NoImage);
        }
        let (glyphs, distance_range, em_size) = parse_glyphs(buf, header.variants_offset)?;
        let (width, height, rgba) = decode_atlas_image(buf, &header)?;
        Ok(FontAtlas { width, height, rgba, glyphs, distance_range, em_size })
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
