//! The KTX2 container: header, level index and the Data Format Descriptor's colour fields
//! (KTX 2.0 spec §3-§4, Khronos Data Format spec §5). The Basis payload itself is decoded by the
//! backend in `basis.rs`; this reads what the engine needs to decide how to upload it.

use super::Ktx2Error;

const IDENTIFIER: [u8; 12] = [0xAB, 0x4B, 0x54, 0x58, 0x20, 0x32, 0x30, 0xBB, 0x0D, 0x0A, 0x1A, 0x0A];
const HEADER_BYTES: usize = 80;
const LEVEL_INDEX_ENTRY_BYTES: usize = 24;
/// `KHR_DF_TRANSFER_SRGB`.
const TRANSFER_SRGB: u8 = 2;

/// Whether `bytes` start with the KTX2 identifier.
pub fn is_ktx2(bytes: &[u8]) -> bool {
    bytes.len() >= IDENTIFIER.len() && bytes[..IDENTIFIER.len()] == IDENTIFIER
}

/// KTX2 level supercompression (`supercompressionScheme`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Supercompression {
    None,
    /// Basis Universal's ETC1S codebooks + slices.
    BasisLz,
    Zstandard,
    Zlib,
    Other(u32),
}

impl Supercompression {
    fn from_raw(v: u32) -> Self {
        match v {
            0 => Self::None,
            1 => Self::BasisLz,
            2 => Self::Zstandard,
            3 => Self::Zlib,
            other => Self::Other(other),
        }
    }
}

/// What a KTX2 file declares about itself, read without decoding any level.
#[derive(Debug, Clone, PartialEq)]
pub struct Ktx2Header {
    /// `VK_FORMAT_UNDEFINED` (0) for ETC1S and UASTC LDR; UASTC HDR 4x4 files carry
    /// `VK_FORMAT_ASTC_4x4_SFLOAT_BLOCK`.
    pub vk_format: u32,
    pub width: u32,
    pub height: u32,
    /// 0 for 1D/2D textures.
    pub depth: u32,
    /// 0 for a texture that is not an array.
    pub layers: u32,
    /// 6 for a cubemap, otherwise 1.
    pub faces: u32,
    /// Mip levels stored in the file (a zero `levelCount`, "generate at load", reads as 1).
    pub levels: u32,
    pub supercompression: Supercompression,
    /// The DFD's colour model (`KHR_DF_MODEL_ETC1S` 163, `KHR_DF_MODEL_UASTC` 166, ...).
    pub color_model: u8,
    /// Whether the DFD's transfer function is sRGB (colour data) rather than linear.
    pub srgb: bool,
    /// Stored bytes of each level (after supercompression), level 0 first.
    pub level_bytes: Vec<u64>,
}

impl Ktx2Header {
    pub fn parse(bytes: &[u8]) -> Result<Self, Ktx2Error> {
        if !is_ktx2(bytes) {
            return Err(Ktx2Error::NotKtx2);
        }
        if bytes.len() < HEADER_BYTES {
            return Err(Ktx2Error::Truncated("header"));
        }
        let u32_at = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        let u64_at = |at: usize| u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let level_count = u32_at(40);
        let levels = level_count.max(1);
        let index_end = HEADER_BYTES + levels as usize * LEVEL_INDEX_ENTRY_BYTES;
        if bytes.len() < index_end {
            return Err(Ktx2Error::Truncated("level index"));
        }
        let mut level_bytes = Vec::with_capacity(levels as usize);
        for level in 0..levels as usize {
            let at = HEADER_BYTES + level * LEVEL_INDEX_ENTRY_BYTES;
            let (offset, length) = (u64_at(at), u64_at(at + 8));
            if offset.checked_add(length).is_none_or(|end| end > bytes.len() as u64) {
                return Err(Ktx2Error::Truncated("level data"));
            }
            level_bytes.push(length);
        }

        // DFD: dfdTotalSize, then the basic descriptor block (colour model, primaries,
        // transfer function, flags at bytes 8..12 of the block)
        let (dfd_offset, dfd_length) = (u32_at(48) as usize, u32_at(52) as usize);
        if dfd_length < 4 + 12 || dfd_offset.checked_add(dfd_length).is_none_or(|end| end > bytes.len()) {
            return Err(Ktx2Error::Truncated("data format descriptor"));
        }
        let block = dfd_offset + 4;
        Ok(Self {
            vk_format: u32_at(12),
            width: u32_at(20),
            height: u32_at(24).max(1),
            depth: u32_at(28),
            layers: u32_at(32),
            faces: u32_at(36).max(1),
            levels,
            supercompression: Supercompression::from_raw(u32_at(44)),
            color_model: bytes[block + 8],
            srgb: bytes[block + 10] == TRANSFER_SRGB,
            level_bytes,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal well-formed header: `levels` index entries pointing at `level_len` bytes each,
    /// then a 28-byte DFD with the given transfer function.
    fn file(levels: u32, transfer: u8, supercompression: u32) -> Vec<u8> {
        let index_end = HEADER_BYTES + levels.max(1) as usize * LEVEL_INDEX_ENTRY_BYTES;
        let dfd_len = 4 + 24;
        let data_at = index_end + dfd_len;
        let mut b = IDENTIFIER.to_vec();
        for v in [0u32, 1, 20, 12, 0, 0, 1, levels, supercompression, index_end as u32, dfd_len as u32, 0, 0] {
            b.extend(v.to_le_bytes());
        }
        b.extend(0u64.to_le_bytes());
        b.extend(0u64.to_le_bytes());
        for level in 0..levels.max(1) as u64 {
            b.extend((data_at as u64 + level * 16).to_le_bytes());
            b.extend(16u64.to_le_bytes());
            b.extend(16u64.to_le_bytes());
        }
        b.extend((dfd_len as u32).to_le_bytes());
        b.extend([0u8; 8]);
        b.extend([166, 1, transfer, 0]);
        b.extend([0u8; 12]);
        b.resize(data_at + 16 * levels.max(1) as usize, 0);
        b
    }

    #[test]
    fn reads_levels_supercompression_and_transfer() {
        let h = Ktx2Header::parse(&file(5, TRANSFER_SRGB, 2)).unwrap();
        assert_eq!((h.width, h.height, h.levels, h.faces), (20, 12, 5, 1));
        assert_eq!(h.supercompression, Supercompression::Zstandard);
        assert_eq!(h.color_model, 166);
        assert!(h.srgb);
        assert_eq!(h.level_bytes, vec![16; 5]);
        assert!(!Ktx2Header::parse(&file(1, 1, 0)).unwrap().srgb);
    }

    #[test]
    fn zero_level_count_reads_as_one_level() {
        assert_eq!(Ktx2Header::parse(&file(0, 1, 1)).unwrap().levels, 1);
    }

    #[test]
    fn rejects_other_files_and_truncation() {
        assert_eq!(Ktx2Header::parse(b"\x89PNG\r\n\x1a\n...."), Err(Ktx2Error::NotKtx2));
        let f = file(3, 1, 0);
        assert!(matches!(Ktx2Header::parse(&f[..60]), Err(Ktx2Error::Truncated(_))));
        assert!(matches!(Ktx2Header::parse(&f[..f.len() - 1]), Err(Ktx2Error::Truncated("level data"))));
    }
}
