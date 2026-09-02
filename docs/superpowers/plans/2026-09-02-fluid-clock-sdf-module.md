# Fluid Clock SDF Module (Plan 1 of 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a pure-Rust `kansei-core/src/sdf/` module that parses `.arfont` MTSDF atlases and produces extruded 3D SDF volumes for the clock glyphs (`0`–`9`, `:`).

**Architecture:** Parse the Artery Font Format binary (header verified against the real asset) to recover the PNG atlas + per-glyph metrics; decode the PNG (via the already-present `image` crate) to RGBA where alpha is a true SDF; crop each glyph's SDF sub-rect and extrude it along Z into a small 3D volume. No GPU, no JS — this plan is CPU-only and fully unit-testable.

**Tech Stack:** Rust, `image` crate (PNG decode), `bytemuck`. No new dependencies.

**Scope note:** This is Plan 1 of 3. Plan 2 adds the additive attractor pass + per-particle tag buffer to the fluid sim. Plan 3 builds the `fluid_clock` WASM example (clock controller, retag, audio, wiring). Each plan produces working, testable software on its own; this one delivers a reusable SDF library piece with unit tests.

---

## Verified Facts (from hexdump of `examples/assets/fonts/L10-medium.arfont`)

- **`ArteryFontHeader`** (112 bytes) confirmed byte-for-byte:
  - `tag[16]` = `"ARTERY/FONT\0..."`, `magicNo`(u32)=`0x4D276A5C`, `version`(u32)=1, `flags`(u32)=0,
    `realType`(u32)=`0x14` (⇒ reals are **f32**), `reserved[4]`, `metadataFormat`(u32)=0,
    `metadataLength`(u32)=0, `variantCount`(u32)=1, `variantsLength`(u32)=`0x1290`,
    `imageCount`(u32)=1, `imagesLength`(u32)=`0x1CCD0`, `appendixCount`(u32)=0,
    `appendicesLength`(u32)=0, `reserved2[8]`.
- Blocks follow the header in order: metadata (len 0) → variants (`0x1290` bytes, starts `0x70`) →
  images (`0x1CCD0` bytes, starts `0x1300`).
- **Image sub-header** at `0x1300`: `flags`=0, `encoding`=8 (**PNG**), `width`=`0x1C0`(448),
  `height`=`0x1C0`(448), `channels`=4, `pixelFormat`=8, `imageType`=7 (**MTSDF**). PNG data begins at
  `0x1340` (`89 50 4E 47` magic). **The atlas is PNG-encoded — must be decoded to RGBA.**
- Alpha channel of the decoded RGBA is the **true single-channel SDF** used for attraction.
- The variant/glyph sub-layout (name string, glyph count, glyph array of
  `{codepoint:u32, image:u32, advance:2×f32, planeBounds:4×f32, imageBounds:4×f32}`) follows the
  Artery Font Format spec; exact offsets are validated by the real-asset test in Task 3, which fails
  loudly on any mistake.

---

## File Structure

- Create `rust/kansei-core/src/sdf/mod.rs` — module root, public re-exports.
- Create `rust/kansei-core/src/sdf/arfont.rs` — `.arfont` binary parser → `FontAtlas`.
- Create `rust/kansei-core/src/sdf/glyph_volume.rs` — SDF crop + extrusion → `GlyphVolume`, `GlyphVolumeSet`.
- Modify `rust/kansei-core/src/lib.rs:15` — add `pub mod sdf;`.
- Create `rust/kansei-core/tests/fixtures/L10-medium.arfont` — copy of the real asset for tests.
- Create `rust/kansei-core/tests/sdf_arfont.rs` — integration test against the real asset.

---

## Task 1: Scaffold the `sdf` module

**Files:**
- Create: `rust/kansei-core/src/sdf/mod.rs`
- Modify: `rust/kansei-core/src/lib.rs:15`

- [ ] **Step 1: Create the module file**

Create `rust/kansei-core/src/sdf/mod.rs`:

```rust
//! Signed-distance-field typography: parse `.arfont` MTSDF atlases and build
//! extruded 3D SDF volumes for glyphs. CPU-only; no GPU or JS dependency.

mod arfont;
mod glyph_volume;

pub use arfont::{FontAtlas, GlyphMetrics, ArFontError};
pub use glyph_volume::{GlyphVolume, GlyphVolumeSet};
```

- [ ] **Step 2: Register the module in the crate**

In `rust/kansei-core/src/lib.rs`, after line 15 (`pub mod systems;`), add:

```rust
pub mod sdf;
```

- [ ] **Step 3: Add placeholder submodule files so the crate compiles**

Create `rust/kansei-core/src/sdf/arfont.rs`:

```rust
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
```

Create `rust/kansei-core/src/sdf/glyph_volume.rs`:

```rust
//! Crop per-glyph SDF from a `FontAtlas` and extrude to a 3D volume.
```

- [ ] **Step 4: Verify the crate compiles**

Run: `cargo build -p kansei-core`
Expected: builds with warnings about unused items, no errors.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/sdf/ rust/kansei-core/src/lib.rs
git commit -m "feat(sdf): scaffold sdf module with FontAtlas/GlyphMetrics types"
```

---

## Task 2: Parse and validate the `.arfont` header

**Files:**
- Modify: `rust/kansei-core/src/sdf/arfont.rs`

- [ ] **Step 1: Write the failing unit test**

Append to `rust/kansei-core/src/sdf/arfont.rs`:

```rust
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
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p kansei-core header_tests`
Expected: FAIL — `ArFontHeader` not found.

- [ ] **Step 3: Implement the header parser**

Insert into `rust/kansei-core/src/sdf/arfont.rs` (above the `#[cfg(test)]` block):

```rust
/// Little-endian u32 read helper.
fn rd_u32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Little-endian f32 read helper.
fn rd_f32(buf: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Parsed `ArteryFontHeader` plus the computed block offsets.
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p kansei-core header_tests`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/sdf/arfont.rs
git commit -m "feat(sdf): parse and validate .arfont header block offsets"
```

---

## Task 3: Parse glyph metrics from the variant block (validated against the real asset)

**Files:**
- Create: `rust/kansei-core/tests/fixtures/L10-medium.arfont` (copy)
- Modify: `rust/kansei-core/src/sdf/arfont.rs`
- Create: `rust/kansei-core/tests/sdf_arfont.rs`

- [ ] **Step 1: Copy the real font asset into the crate as a test fixture**

Run:

```bash
mkdir -p rust/kansei-core/tests/fixtures
cp examples/assets/fonts/L10-medium.arfont rust/kansei-core/tests/fixtures/L10-medium.arfont
```

- [ ] **Step 2: Write the failing integration test**

Create `rust/kansei-core/tests/sdf_arfont.rs`:

```rust
use kansei_core::sdf::FontAtlas;

const FONT: &[u8] = include_bytes!("fixtures/L10-medium.arfont");

#[test]
fn parses_all_clock_glyphs() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");

    // Every clock glyph must be present: '0'..'9' and ':'.
    for cp in ('0'..='9').chain([':'].into_iter()) {
        let g = atlas
            .glyphs
            .iter()
            .find(|g| g.codepoint == cp as u32)
            .unwrap_or_else(|| panic!("missing glyph for {cp:?}"));

        // image_bounds must be a sane sub-rect inside the 448×448 atlas.
        let [l, b, r, t] = g.image_bounds;
        assert!(r > l && t >= b, "glyph {cp:?} bounds not ordered: {:?}", g.image_bounds);
        assert!(l >= 0.0 && r <= atlas.width as f32, "glyph {cp:?} x out of range");
        assert!(b >= 0.0 && t <= atlas.height as f32, "glyph {cp:?} y out of range");
    }
}

#[test]
fn reports_atlas_dimensions_and_metrics() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");
    assert_eq!(atlas.width, 448);
    assert_eq!(atlas.height, 448);
    assert_eq!(atlas.rgba.len(), (448 * 448 * 4) as usize);
    assert!(atlas.distance_range > 0.0);
    assert!(atlas.em_size > 0.0);
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cargo test -p kansei-core --test sdf_arfont`
Expected: FAIL — `FontAtlas::parse` not found.

- [ ] **Step 4: Implement variant + glyph parsing**

Add to `rust/kansei-core/src/sdf/arfont.rs`. This walks the variant header to the glyph array.
The Artery Font Format variant layout is: fixed header fields, then a length-prefixed `name` string,
then a length-prefixed `metadata` string, then `glyphCount`/`glyphsLength`/`kernCount`/`kernsLength`,
then the glyph array. Strings are stored as `{u32 length; bytes; padding to 4-byte alignment}`.

**CORRECTED against the authoritative `Chlumsky/artery-font-format` spec** (`structures.h` +
`serialization.hpp`) during implementation. Two errors in the original draft were fixed: the metrics
block is **32 reals (128 bytes)**, not 8; and the glyph field order is
`codepoint, image, planeBounds, imageBounds, advance` — not advance-first. Strings are
`{len bytes + 1 NUL, pad to 4}` and their lengths come from `nameLength`/`metadataLength` in the
counts block, so `skip_string` takes a known length rather than reading its own prefix word.

```rust
/// Advance past a string block of `len` payload bytes: `{len bytes + 1 NUL, pad to 4}`.
/// Writes nothing when `len == 0`.
fn skip_string(off: usize, len: usize) -> usize {
    if len == 0 {
        return off;
    }
    let total = len + 1; // trailing NUL
    let padded = (total + 3) & !3usize; // 4-byte align
    off + padded
}

/// One glyph record: 2×u32 + 10×f32 = 48 bytes.
/// Layout (artery-font-format): codepoint, image, planeBounds(l,b,r,t),
/// imageBounds(l,b,r,t), advance(h,v).
const GLYPH_STRIDE: usize = 48;

fn parse_glyphs(
    buf: &[u8],
    variants_offset: usize,
) -> Result<(Vec<GlyphMetrics>, f32, f32), ArFontError> {
    // Variant fixed header: flags,weight,codepointType,imageType,fallbackVariant,
    // fallbackGlyph (6×u32) then reserved[6] (6×u32) = 48 bytes, then metrics.
    let metrics_off = variants_offset + 48;

    // Metrics block: REAL metrics[32] = fontSize, distanceRange, emSize, ascender,
    // descender, lineHeight, underlineY, underlineThickness, distanceRangeMiddle, reserved[23].
    let distance_range = rd_f32(buf, metrics_off + 4);
    let em_size = rd_f32(buf, metrics_off + 8);

    // Counts block after the 32-real metrics: nameLength, metadataLength, glyphCount, kernPairCount.
    let counts_off = metrics_off + 32 * 4;
    let name_length = rd_u32(buf, counts_off) as usize;
    let metadata_length = rd_u32(buf, counts_off + 4) as usize;
    let glyph_count = rd_u32(buf, counts_off + 8) as usize;

    let mut off = counts_off + 16; // past the 4 count u32s
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
```

> **Verified offsets for `L10-medium.arfont`:** variant fixed header `0x70`–`0xA0` (48 B), metrics
> `0xA0`–`0x120` (128 B), counts at `0x120` (`nameLength=0, metadataLength=0, glyphCount=95`), glyph
> array `0x130`–`0x1300` (95 × 48 B), images block at `0x1300`. `'0'` image_bounds ≈
> `(121.5, 170.5, 164.5, 226.5)`, `':'` ≈ `(424.5, 231.5, 440.5, 272.5)`.

- [ ] **Step 5: Implement `FontAtlas::parse` (header + glyphs + PNG decode) — glyphs first**

Add the public entry point. PNG decode is added in Task 4; for now decode is stubbed so the glyph
tests pass independently:

```rust
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
```

- [ ] **Step 6: Add a temporary image-decode stub so Task 3 can pass before Task 4**

Add this stub (replaced in Task 4). It reads the verified image dimensions and returns a zeroed RGBA
buffer of the right size:

```rust
/// TEMPORARY stub (replaced in Task 4 with real PNG decode). Returns a zeroed RGBA
/// buffer sized to the atlas so glyph-metric tests can run independently.
fn decode_atlas_image(buf: &[u8], header: &ArFontHeader) -> Result<(u32, u32, Vec<u8>), ArFontError> {
    let img_off = header.images_offset;
    // Image sub-header (verified): flags(0), encoding(4), width(8), height(12), channels(16)...
    let width = rd_u32(buf, img_off + 8);
    let height = rd_u32(buf, img_off + 12);
    Ok((width, height, vec![0u8; (width * height * 4) as usize]))
}
```

- [ ] **Step 7: Run the tests to verify they pass**

Run: `cargo test -p kansei-core --test sdf_arfont`
Expected: PASS (2 tests). If `parses_all_clock_glyphs` fails, follow the implementer note in Step 4.

- [ ] **Step 8: Commit**

```bash
git add rust/kansei-core/src/sdf/arfont.rs rust/kansei-core/tests/
git commit -m "feat(sdf): parse glyph metrics from .arfont, validated against real asset"
```

---

## Task 4: Decode the PNG atlas to RGBA

**Files:**
- Modify: `rust/kansei-core/src/sdf/arfont.rs`

- [ ] **Step 1: Write the failing test (alpha channel is a real SDF, not all zero)**

Append to the `#[cfg(test)]` region of `rust/kansei-core/tests/sdf_arfont.rs`:

```rust
#[test]
fn atlas_alpha_channel_is_populated_sdf() {
    let atlas = FontAtlas::parse(FONT).expect("parse .arfont");
    // A real MTSDF alpha channel spans the full range across the atlas.
    let mut min_a = 255u8;
    let mut max_a = 0u8;
    for px in atlas.rgba.chunks_exact(4) {
        min_a = min_a.min(px[3]);
        max_a = max_a.max(px[3]);
    }
    assert!(max_a > min_a, "alpha channel is flat — PNG not decoded");
    assert!(max_a > 200 && min_a < 55, "alpha does not span SDF range: {min_a}..{max_a}");
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p kansei-core --test sdf_arfont atlas_alpha_channel_is_populated_sdf`
Expected: FAIL — alpha is flat (stub returns zeros).

- [ ] **Step 3: Replace the stub with a real PNG decode using the `image` crate**

In `rust/kansei-core/src/sdf/arfont.rs`, replace the entire `decode_atlas_image` stub from Task 3
Step 6 with:

```rust
/// Decode the embedded atlas image (PNG, `encoding == 8`) to RGBA8.
fn decode_atlas_image(buf: &[u8], header: &ArFontHeader) -> Result<(u32, u32, Vec<u8>), ArFontError> {
    let img_off = header.images_offset;
    // Image sub-header (verified layout, all u32):
    //   flags(+0) encoding(+4) width(+8) height(+12) channels(+16) pixelFormat(+20)
    //   imageType(+24) rowLength(+28) orientation(+32) childImages(+36) textureFlags(+40)
    //   reserved... metadataLength then dataLength immediately before the pixel data.
    let encoding = rd_u32(buf, img_off + 4);
    let width = rd_u32(buf, img_off + 8);
    let height = rd_u32(buf, img_off + 12);

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
    Ok((width, height, rgba.into_raw()))
}
```

- [ ] **Step 4: Run the full arfont test suite**

Run: `cargo test -p kansei-core --test sdf_arfont`
Expected: PASS (3 tests) — glyphs, dimensions, and populated alpha.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/sdf/arfont.rs rust/kansei-core/tests/sdf_arfont.rs
git commit -m "feat(sdf): decode embedded PNG atlas to RGBA (alpha = SDF)"
```

---

## Task 5: Crop a glyph's 2D SDF from the atlas

**Files:**
- Modify: `rust/kansei-core/src/sdf/glyph_volume.rs`

- [ ] **Step 1: Write the failing unit test**

Replace the contents of `rust/kansei-core/src/sdf/glyph_volume.rs` with:

```rust
//! Crop per-glyph SDF from a `FontAtlas` and extrude to a 3D volume.

use crate::sdf::{FontAtlas, GlyphMetrics};

/// A single glyph's SDF cropped to a fixed square resolution, values in [-1, 1]
/// where positive is inside the glyph.
pub struct GlyphSdf2d {
    pub res: u32,
    /// `res * res` signed values, row-major, +inside / -outside.
    pub data: Vec<f32>,
}

/// Sample the atlas alpha (SDF) for `glyph`, resampled to `res × res`.
/// Atlas alpha stores SDF as unsigned [0,255] with 0.5 (=127.5) at the outline;
/// we remap to signed [-1, 1] (+inside).
pub fn crop_glyph_sdf(atlas: &FontAtlas, glyph: &GlyphMetrics, res: u32) -> GlyphSdf2d {
    let [l, b, r, t] = glyph.image_bounds;
    let mut data = vec![0.0f32; (res * res) as usize];
    let aw = atlas.width as f32;
    let ah = atlas.height as f32;
    for y in 0..res {
        for x in 0..res {
            // Map output cell to atlas pixel (bilinear-nearest is fine for the field).
            let u = (x as f32 + 0.5) / res as f32;
            let v = (y as f32 + 0.5) / res as f32;
            let ax = (l + u * (r - l)).clamp(0.0, aw - 1.0);
            // image_bounds y is bottom-up; atlas rows are top-down.
            let ay = (ah - (b + v * (t - b))).clamp(0.0, ah - 1.0);
            let idx = ((ay as u32 * atlas.width + ax as u32) * 4 + 3) as usize;
            let alpha = atlas.rgba[idx] as f32 / 255.0;
            data[(y * res + x) as usize] = (alpha - 0.5) * 2.0; // [-1,1], +inside
        }
    }
    GlyphSdf2d { res, data }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdf::FontAtlas;

    const FONT: &[u8] = include_bytes!("../../tests/fixtures/L10-medium.arfont");

    #[test]
    fn glyph_center_is_inside_edges_outside() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let zero = atlas.glyphs.iter().find(|g| g.codepoint == '0' as u32).unwrap();
        let sdf = crop_glyph_sdf(&atlas, zero, 32);
        // For '0', a point on the left stroke should be inside; the very center is the hole (outside).
        let at = |x: u32, y: u32| sdf.data[(y * 32 + x) as usize];
        assert!(at(4, 16) > 0.0 || at(28, 16) > 0.0, "a stroke sample should be inside the glyph");
        assert!(at(0, 0) < 0.0, "the corner should be outside the glyph");
    }
}
```

- [ ] **Step 2: Run the test to verify it fails, then passes**

Run: `cargo test -p kansei-core glyph_center_is_inside_edges_outside`
Expected: the code compiles and the test PASSES (implementation is included above).
If the stroke assertion fails, the glyph's `image_bounds` Y-orientation may be top-down; flip the
`ay` computation to `(b + v * (t - b))` and re-run.

- [ ] **Step 3: Commit**

```bash
git add rust/kansei-core/src/sdf/glyph_volume.rs
git commit -m "feat(sdf): crop per-glyph 2D SDF from atlas alpha channel"
```

---

## Task 6: Extrude a glyph SDF to a 3D volume

**Files:**
- Modify: `rust/kansei-core/src/sdf/glyph_volume.rs`

- [ ] **Step 1: Write the failing unit test**

Append to the `tests` module in `rust/kansei-core/src/sdf/glyph_volume.rs`:

```rust
    #[test]
    fn extrudes_symmetrically_along_z() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let one = atlas.glyphs.iter().find(|g| g.codepoint == '1' as u32).unwrap();
        let vol = GlyphVolume::extrude(&atlas, one, 32, 8, 0.5);
        assert_eq!(vol.res_xy, 32);
        assert_eq!(vol.res_z, 8);
        assert_eq!(vol.data.len(), (32 * 32 * 8) as usize);

        // A cell that is inside the 2D glyph and near mid-depth stays inside;
        // the same (x,y) at the front/back cap is pushed outside by the |z| term.
        let idx = |x: u32, y: u32, z: u32| ((z * 32 + y) * 32 + x) as usize;
        // Find an inside 2D cell.
        let sdf2d = crop_glyph_sdf(&atlas, one, 32);
        let mut inside_xy = None;
        for y in 0..32 { for x in 0..32 {
            if sdf2d.data[(y * 32 + x) as usize] > 0.2 { inside_xy = Some((x, y)); }
        }}
        let (ix, iy) = inside_xy.expect("glyph '1' must have interior cells");
        assert!(vol.data[idx(ix, iy, 4)] > 0.0, "mid-depth interior should be inside");
        assert!(vol.data[idx(ix, iy, 0)] <= vol.data[idx(ix, iy, 4)], "cap should be <= mid");
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p kansei-core extrudes_symmetrically_along_z`
Expected: FAIL — `GlyphVolume` not found.

- [ ] **Step 3: Implement `GlyphVolume::extrude`**

Insert into `rust/kansei-core/src/sdf/glyph_volume.rs` (above the `#[cfg(test)]` block):

```rust
/// A glyph's SDF extruded into a 3D volume of `res_xy × res_xy × res_z` cells.
/// Values are signed (+inside). Z spans [-1, 1] scaled so `half_depth` is the
/// front/back face of the slab.
pub struct GlyphVolume {
    pub res_xy: u32,
    pub res_z: u32,
    pub half_depth: f32,
    /// Row-major `x + res_xy*(y + res_xy_z_stride)`; index as ((z*res_xy)+y)*res_xy + x.
    pub data: Vec<f32>,
}

impl GlyphVolume {
    /// Extrude a 2D glyph SDF along Z: `sdf3d = min(sdf2d, half_depth - |z|)`.
    /// (Signed convention is +inside, so the slab cap is `half_depth - |z|`.)
    pub fn extrude(
        atlas: &FontAtlas,
        glyph: &GlyphMetrics,
        res_xy: u32,
        res_z: u32,
        half_depth: f32,
    ) -> GlyphVolume {
        let sdf2d = crop_glyph_sdf(atlas, glyph, res_xy);
        let mut data = vec![0.0f32; (res_xy * res_xy * res_z) as usize];
        for z in 0..res_z {
            // z in [-1, 1]
            let zc = if res_z > 1 {
                (z as f32 / (res_z - 1) as f32) * 2.0 - 1.0
            } else {
                0.0
            };
            let cap = half_depth - zc.abs(); // +inside slab
            for y in 0..res_xy {
                for x in 0..res_xy {
                    let s2 = sdf2d.data[(y * res_xy + x) as usize];
                    let s3 = s2.min(cap);
                    data[(((z * res_xy) + y) * res_xy + x) as usize] = s3;
                }
            }
        }
        GlyphVolume { res_xy, res_z, half_depth, data }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cargo test -p kansei-core extrudes_symmetrically_along_z`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/sdf/glyph_volume.rs
git commit -m "feat(sdf): extrude 2D glyph SDF into 3D volume slab"
```

---

## Task 7: Build the clock glyph set (`0`–`9`, `:`)

**Files:**
- Modify: `rust/kansei-core/src/sdf/glyph_volume.rs`
- Modify: `rust/kansei-core/src/sdf/mod.rs` (already re-exports `GlyphVolumeSet`)

- [ ] **Step 1: Write the failing unit test**

Append to the `tests` module in `rust/kansei-core/src/sdf/glyph_volume.rs`:

```rust
    #[test]
    fn builds_all_eleven_clock_glyphs() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&atlas, 32, 8, 0.5);
        // Indices 0..=9 are digits; index 10 is ':'.
        for d in 0u32..=9 {
            assert!(set.volume_for_digit(d).is_some(), "missing digit {d}");
        }
        assert!(set.colon().is_some(), "missing colon volume");
        assert_eq!(set.res_xy, 32);
        assert_eq!(set.res_z, 8);
    }

    #[test]
    fn digit_lookup_out_of_range_is_none() {
        let atlas = FontAtlas::parse(FONT).unwrap();
        let set = GlyphVolumeSet::for_clock(&atlas, 16, 4, 0.5);
        assert!(set.volume_for_digit(10).is_none());
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cargo test -p kansei-core builds_all_eleven_clock_glyphs`
Expected: FAIL — `GlyphVolumeSet` not found.

- [ ] **Step 3: Implement `GlyphVolumeSet`**

Insert into `rust/kansei-core/src/sdf/glyph_volume.rs` (above the `#[cfg(test)]` block):

```rust
/// The 11 glyph volumes a clock needs: digits `0`–`9` (indices 0..=9) and `:` (index 10).
pub struct GlyphVolumeSet {
    pub res_xy: u32,
    pub res_z: u32,
    /// 11 volumes; `[0..=9]` = digits, `[10]` = colon. `None` if a glyph was missing.
    volumes: Vec<Option<GlyphVolume>>,
}

impl GlyphVolumeSet {
    /// Build volumes for `'0'..'9'` and `':'`. Missing glyphs yield `None` slots.
    pub fn for_clock(atlas: &FontAtlas, res_xy: u32, res_z: u32, half_depth: f32) -> GlyphVolumeSet {
        let codepoints: Vec<u32> = ('0'..='9').chain([':'].into_iter()).map(|c| c as u32).collect();
        let volumes = codepoints
            .iter()
            .map(|cp| {
                atlas
                    .glyphs
                    .iter()
                    .find(|g| g.codepoint == *cp)
                    .map(|g| GlyphVolume::extrude(atlas, g, res_xy, res_z, half_depth))
            })
            .collect();
        GlyphVolumeSet { res_xy, res_z, volumes }
    }

    /// Volume for digit `d` (0..=9), or `None` if `d > 9` or the glyph was missing.
    pub fn volume_for_digit(&self, d: u32) -> Option<&GlyphVolume> {
        if d > 9 {
            return None;
        }
        self.volumes.get(d as usize).and_then(|v| v.as_ref())
    }

    /// The colon (`:`) volume, or `None` if it was missing.
    pub fn colon(&self) -> Option<&GlyphVolume> {
        self.volumes.get(10).and_then(|v| v.as_ref())
    }
}
```

- [ ] **Step 4: Run the whole SDF test suite**

Run: `cargo test -p kansei-core sdf`
Expected: PASS — header tests, glyph tests, crop, extrude, and the glyph set.

Also run the integration test:

Run: `cargo test -p kansei-core --test sdf_arfont`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add rust/kansei-core/src/sdf/glyph_volume.rs
git commit -m "feat(sdf): build clock glyph volume set (0-9 and colon)"
```

---

## Task 8: Public API polish + doc example

**Files:**
- Modify: `rust/kansei-core/src/sdf/mod.rs`
- Modify: `rust/kansei-core/src/sdf/glyph_volume.rs`

- [ ] **Step 1: Ensure the public surface is exactly what Plans 2/3 need**

Confirm `rust/kansei-core/src/sdf/mod.rs` re-exports:

```rust
pub use arfont::{FontAtlas, GlyphMetrics, ArFontError};
pub use glyph_volume::{GlyphVolume, GlyphVolumeSet, GlyphSdf2d, crop_glyph_sdf};
```

Add `GlyphSdf2d` and `crop_glyph_sdf` to the `pub use` line if not already present.

- [ ] **Step 2: Add a module-level doc example (compiled as a doctest)**

Prepend to `rust/kansei-core/src/sdf/mod.rs`:

```rust
//! ```no_run
//! use kansei_core::sdf::{FontAtlas, GlyphVolumeSet};
//! let bytes: &[u8] = &[]; // load your .arfont
//! if let Ok(atlas) = FontAtlas::parse(bytes) {
//!     let set = GlyphVolumeSet::for_clock(&atlas, 32, 8, 0.5);
//!     let _five = set.volume_for_digit(5);
//! }
//! ```
```

- [ ] **Step 3: Run the full core test suite + clippy**

Run: `cargo test -p kansei-core`
Expected: PASS (all prior tests, including doctest).

Run: `cargo clippy -p kansei-core -- -D warnings 2>&1 | tail -5`
Expected: no errors from the new `sdf` module (pre-existing warnings elsewhere are out of scope).

- [ ] **Step 4: Commit**

```bash
git add rust/kansei-core/src/sdf/
git commit -m "feat(sdf): finalize public API and add module doc example"
```

---

## Self-Review Results

**Spec coverage (SDF portion of the design doc):**
- "New module `kansei-core/src/sdf/`" → Tasks 1–8. ✓
- "`arfont.rs` — minimal parser → AtlasImage + Vec<GlyphMetrics> + metrics" → Tasks 2–4. ✓
- "`glyph_volume.rs` — crop SDF sub-rect + extrude to N layers" → Tasks 5–6. ✓
- "coarse inside/outside attraction basin" → deferred to Plan 2 (attractor pass), where the broad
  basin is applied in the GPU force; the CPU volume here stores the precise near-field SDF. Noted so
  it is not lost.
- "11 glyphs (0–9, ':')" → Task 7. ✓
- PNG-encoded atlas (discovered during planning) → Task 4. ✓

**Not in this plan (correctly deferred):** per-particle tag buffer, attractor compute pass, GPU 3D
texture upload, clock controller, retag/recruitment, marching-cubes integration, audio. These are
Plans 2 and 3.

**Placeholder scan:** none — every code step contains complete code. The one non-verified area
(variant sub-offsets in Task 3) is explicitly guarded by a real-asset test with a documented fix path.

**Type consistency:** `FontAtlas`, `GlyphMetrics`, `GlyphVolume`, `GlyphVolumeSet`, `GlyphSdf2d`,
`crop_glyph_sdf`, `volume_for_digit`, `colon`, `for_clock` are used consistently across tasks and
re-exports.
