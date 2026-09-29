//! Which GPU format a Basis texture becomes on this device: the device's block-compression
//! features, the source codec, the texture's channels and its size decide it.

/// The block-compressed formats a device can sample, from its enabled `wgpu::Features`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CompressionSupport {
    /// `texture-compression-bc`: BC1-7 (desktop GPUs, Apple Silicon).
    pub bc: bool,
    /// `texture-compression-astc`: ASTC LDR (Apple, most mobile).
    pub astc: bool,
    /// `texture-compression-etc2`: ETC2/EAC (mobile, Apple).
    pub etc2: bool,
    /// ASTC HDR (native only; WebGPU has no feature for it).
    pub astc_hdr: bool,
}

impl CompressionSupport {
    /// No compressed formats: everything decodes to uncompressed texels.
    pub const NONE: Self = Self { bc: false, astc: false, etc2: false, astc_hdr: false };

    /// Every compression feature the KTX2 loader can use; request the subset the adapter offers
    /// (`adapter.features() & CompressionSupport::FEATURES`) when creating the device.
    pub const FEATURES: wgpu::Features = wgpu::Features::TEXTURE_COMPRESSION_BC
        .union(wgpu::Features::TEXTURE_COMPRESSION_ASTC)
        .union(wgpu::Features::TEXTURE_COMPRESSION_ETC2)
        .union(wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR);

    pub fn from_features(features: wgpu::Features) -> Self {
        Self {
            bc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC),
            astc: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC),
            etc2: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ETC2),
            astc_hdr: features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_HDR),
        }
    }

    /// What `device` was created with.
    pub fn of_device(device: &wgpu::Device) -> Self {
        Self::from_features(device.features())
    }

    /// Whether the device can sample `target` (the uncompressed ones always).
    pub fn supports(self, target: GpuTarget) -> bool {
        use GpuTarget::*;
        match target {
            Bc1 | Bc4 | Bc5 | Bc6h | Bc7 => self.bc,
            Astc4x4 => self.astc,
            AstcHdr4x4 => self.astc_hdr,
            Etc2Rgb | Etc2Rgba | EacR11 | EacRg11 => self.etc2,
            Rgba8 | Rg8 | R8 | Rgba16Float => true,
        }
    }
}

/// The channels a texture carries, which decide whether one- and two-channel formats fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channels {
    /// One channel (roughness, occlusion, a mask), read from `.r`.
    R,
    /// Two channels encoded the Basis way, R in colour and G in alpha (`basisu
    /// -separate_rg_to_color_alpha`, for XY normal maps); sampled as `.rg` whatever the GPU format.
    Rg,
    Rgb,
    Rgba,
}

/// The Basis source codec, as far as target choice is concerned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BasisCodec {
    /// ETC1S: small, ETC1-quality; ETC2 is a lossless target.
    Etc1s,
    /// UASTC LDR 4x4: high quality; ASTC 4x4 is a lossless target.
    UastcLdr,
    /// UASTC HDR 4x4: BC6H, ASTC HDR or half floats.
    UastcHdr,
    /// Another Basis codec (XUASTC, raw ASTC): whatever of the generic LDR targets it supports.
    OtherLdr,
    /// Another HDR codec.
    OtherHdr,
}

/// The GPU format a Basis texture is transcoded to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuTarget {
    Bc1,
    Bc4,
    Bc5,
    Bc6h,
    Bc7,
    Astc4x4,
    AstcHdr4x4,
    /// ETC2 RGB8 (Basis writes ETC1 blocks, a subset of it).
    Etc2Rgb,
    Etc2Rgba,
    EacR11,
    EacRg11,
    /// Uncompressed fallbacks.
    Rgba8,
    Rg8,
    R8,
    Rgba16Float,
}

impl GpuTarget {
    /// The texture format, with the sRGB variant where the format has one and `srgb` is set.
    pub fn format(self, srgb: bool) -> wgpu::TextureFormat {
        use wgpu::{AstcBlock, AstcChannel, TextureFormat as F};
        match self {
            GpuTarget::Bc1 if srgb => F::Bc1RgbaUnormSrgb,
            GpuTarget::Bc1 => F::Bc1RgbaUnorm,
            GpuTarget::Bc4 => F::Bc4RUnorm,
            GpuTarget::Bc5 => F::Bc5RgUnorm,
            GpuTarget::Bc6h => F::Bc6hRgbUfloat,
            GpuTarget::Bc7 if srgb => F::Bc7RgbaUnormSrgb,
            GpuTarget::Bc7 => F::Bc7RgbaUnorm,
            GpuTarget::Astc4x4 => F::Astc {
                block: AstcBlock::B4x4,
                channel: if srgb { AstcChannel::UnormSrgb } else { AstcChannel::Unorm },
            },
            GpuTarget::AstcHdr4x4 => F::Astc { block: AstcBlock::B4x4, channel: AstcChannel::Hdr },
            GpuTarget::Etc2Rgb if srgb => F::Etc2Rgb8UnormSrgb,
            GpuTarget::Etc2Rgb => F::Etc2Rgb8Unorm,
            GpuTarget::Etc2Rgba if srgb => F::Etc2Rgba8UnormSrgb,
            GpuTarget::Etc2Rgba => F::Etc2Rgba8Unorm,
            GpuTarget::EacR11 => F::EacR11Unorm,
            GpuTarget::EacRg11 => F::EacRg11Unorm,
            GpuTarget::Rgba8 if srgb => F::Rgba8UnormSrgb,
            GpuTarget::Rgba8 => F::Rgba8Unorm,
            GpuTarget::Rg8 => F::Rg8Unorm,
            GpuTarget::R8 => F::R8Unorm,
            GpuTarget::Rgba16Float => F::Rgba16Float,
        }
    }

    pub fn is_compressed(self) -> bool {
        !matches!(self, GpuTarget::Rgba8 | GpuTarget::Rg8 | GpuTarget::R8 | GpuTarget::Rgba16Float)
    }

    /// Bytes one mip level of `width` x `height` texels occupies on the GPU (compressed formats
    /// round up to whole 4x4 blocks).
    pub fn level_bytes(self, width: u32, height: u32) -> u64 {
        let format = self.format(false);
        let (bw, bh) = format.block_dimensions();
        let block_bytes = format.block_copy_size(None).expect("colour format") as u64;
        width.div_ceil(bw) as u64 * height.div_ceil(bh) as u64 * block_bytes
    }

    /// Bytes of a full chain of `levels` mips from `width` x `height`.
    pub fn chain_bytes(self, width: u32, height: u32, levels: u32) -> u64 {
        (0..levels).map(|l| self.level_bytes((width >> l).max(1), (height >> l).max(1))).sum()
    }

    /// Short name for logs and HUDs ("BC7", "ASTC 4x4", ...).
    pub fn name(self) -> &'static str {
        match self {
            GpuTarget::Bc1 => "BC1",
            GpuTarget::Bc4 => "BC4",
            GpuTarget::Bc5 => "BC5",
            GpuTarget::Bc6h => "BC6H",
            GpuTarget::Bc7 => "BC7",
            GpuTarget::Astc4x4 => "ASTC 4x4",
            GpuTarget::AstcHdr4x4 => "ASTC 4x4 HDR",
            GpuTarget::Etc2Rgb => "ETC2 RGB",
            GpuTarget::Etc2Rgba => "ETC2 RGBA",
            GpuTarget::EacR11 => "EAC R11",
            GpuTarget::EacRg11 => "EAC RG11",
            GpuTarget::Rgba8 => "RGBA8",
            GpuTarget::Rg8 => "RG8",
            GpuTarget::R8 => "R8",
            GpuTarget::Rgba16Float => "RGBA16F",
        }
    }
}

/// Targets in order of preference for a codec and its channels. The lossless targets lead (ETC2
/// holds ETC1S exactly, ASTC 4x4 holds UASTC exactly); among the rest, the smallest format that
/// keeps the channels. Every list ends in an uncompressed format, which is always available.
pub fn preferences(codec: BasisCodec, channels: Channels) -> &'static [GpuTarget] {
    use GpuTarget::*;
    match (codec, channels) {
        (BasisCodec::UastcHdr | BasisCodec::OtherHdr, _) => &[Bc6h, AstcHdr4x4, Rgba16Float],
        (BasisCodec::Etc1s, Channels::R) => &[EacR11, Bc4, R8],
        (BasisCodec::Etc1s, Channels::Rg) => &[EacRg11, Bc5, Rg8],
        (BasisCodec::Etc1s, Channels::Rgb) => &[Etc2Rgb, Bc1, Astc4x4, Rgba8],
        (BasisCodec::Etc1s, Channels::Rgba) => &[Etc2Rgba, Bc7, Astc4x4, Rgba8],
        // ASTC keeps UASTC's own RRRG layout for two channels, so it is not offered for Rg
        (BasisCodec::UastcLdr | BasisCodec::OtherLdr, Channels::R) => &[Bc4, EacR11, Astc4x4, R8],
        (BasisCodec::UastcLdr | BasisCodec::OtherLdr, Channels::Rg) => &[Bc5, EacRg11, Rg8],
        (BasisCodec::UastcLdr | BasisCodec::OtherLdr, Channels::Rgb) => &[Astc4x4, Bc7, Etc2Rgb, Rgba8],
        (BasisCodec::UastcLdr | BasisCodec::OtherLdr, Channels::Rgba) => &[Astc4x4, Bc7, Etc2Rgba, Rgba8],
    }
}

/// The first preferred target this device supports, the file can produce (`transcodable`), and
/// the size allows: WebGPU needs a block-compressed texture's base size to be a whole number of
/// 4x4 blocks, so other sizes fall back to uncompressed texels.
pub fn select_target(
    codec: BasisCodec,
    channels: Channels,
    (width, height): (u32, u32),
    support: CompressionSupport,
    transcodable: impl Fn(GpuTarget) -> bool,
) -> GpuTarget {
    let block_aligned = width % 4 == 0 && height % 4 == 0;
    let prefs = preferences(codec, channels);
    prefs
        .iter()
        .copied()
        .find(|&t| (!t.is_compressed() || block_aligned) && support.supports(t) && transcodable(t))
        .unwrap_or(*prefs.last().unwrap())
}

#[cfg(test)]
mod tests {
    use super::*;
    use GpuTarget::*;

    const ALL: CompressionSupport = CompressionSupport { bc: true, astc: true, etc2: true, astc_hdr: true };
    const BC: CompressionSupport = CompressionSupport { bc: true, ..CompressionSupport::NONE };
    const ASTC: CompressionSupport = CompressionSupport { astc: true, ..CompressionSupport::NONE };
    const ETC2: CompressionSupport = CompressionSupport { etc2: true, ..CompressionSupport::NONE };
    const SIZE: (u32, u32) = (256, 128);

    fn pick(codec: BasisCodec, channels: Channels, support: CompressionSupport) -> GpuTarget {
        select_target(codec, channels, SIZE, support, |_| true)
    }

    #[test]
    fn uastc_prefers_lossless_astc_then_bc7_then_etc2() {
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgba, ALL), Astc4x4);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgba, BC), Bc7);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgb, BC), Bc7);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgba, ETC2), Etc2Rgba);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgb, ETC2), Etc2Rgb);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rgba, CompressionSupport::NONE), Rgba8);
    }

    #[test]
    fn etc1s_prefers_lossless_etc2_then_the_smallest_bc() {
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgb, ALL), Etc2Rgb);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgba, ALL), Etc2Rgba);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgb, BC), Bc1);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgba, BC), Bc7);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgb, ASTC), Astc4x4);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rgb, CompressionSupport::NONE), Rgba8);
    }

    #[test]
    fn one_and_two_channel_textures_use_bc4_bc5_or_eac() {
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::R, ALL), Bc4);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rg, ALL), Bc5);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rg, ETC2), EacRg11);
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::R, ASTC), Astc4x4);
        // ASTC would keep the RRRG layout, so two channels fall back to RG8 rather than use it
        assert_eq!(pick(BasisCodec::UastcLdr, Channels::Rg, ASTC), Rg8);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::R, ALL), EacR11);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::Rg, BC), Bc5);
        assert_eq!(pick(BasisCodec::Etc1s, Channels::R, CompressionSupport::NONE), R8);
    }

    #[test]
    fn hdr_uses_bc6h_astc_hdr_or_half_floats() {
        assert_eq!(pick(BasisCodec::UastcHdr, Channels::Rgb, ALL), Bc6h);
        assert_eq!(pick(BasisCodec::UastcHdr, Channels::Rgb, CompressionSupport { astc_hdr: true, ..ASTC }), AstcHdr4x4);
        // plain ASTC (LDR) support is not enough
        assert_eq!(pick(BasisCodec::UastcHdr, Channels::Rgb, ASTC), Rgba16Float);
    }

    #[test]
    fn sizes_off_the_block_grid_fall_back_to_uncompressed() {
        for size in [(20, 12), (4, 4), (4096, 8)] {
            assert_eq!(select_target(BasisCodec::UastcLdr, Channels::Rgba, size, ALL, |_| true), Astc4x4, "{size:?}");
        }
        for size in [(1001, 600), (256, 2), (1, 1)] {
            assert_eq!(select_target(BasisCodec::UastcLdr, Channels::Rgba, size, ALL, |_| true), Rgba8, "{size:?}");
            assert_eq!(select_target(BasisCodec::Etc1s, Channels::R, size, ALL, |_| true), R8, "{size:?}");
        }
    }

    #[test]
    fn targets_the_file_cannot_produce_are_skipped() {
        let no_astc = |t: GpuTarget| t != Astc4x4;
        assert_eq!(select_target(BasisCodec::OtherLdr, Channels::Rgba, SIZE, ALL, no_astc), Bc7);
    }

    #[test]
    fn formats_follow_the_colour_space_where_they_have_an_srgb_variant() {
        assert_eq!(Bc7.format(true), wgpu::TextureFormat::Bc7RgbaUnormSrgb);
        assert_eq!(Bc7.format(false), wgpu::TextureFormat::Bc7RgbaUnorm);
        assert_eq!(Etc2Rgb.format(true), wgpu::TextureFormat::Etc2Rgb8UnormSrgb);
        assert!(Astc4x4.format(true).is_srgb());
        assert!(Rgba8.format(true).is_srgb());
        // one- and two-channel formats are always linear
        assert_eq!(Bc5.format(true), wgpu::TextureFormat::Bc5RgUnorm);
        assert_eq!(R8.format(true), wgpu::TextureFormat::R8Unorm);
    }

    #[test]
    fn memory_rounds_up_to_blocks_and_sums_the_chain() {
        assert_eq!(Bc7.level_bytes(20, 12), 5 * 3 * 16);
        assert_eq!(Bc1.level_bytes(2, 1), 8);
        assert_eq!(Rgba8.level_bytes(20, 12), 20 * 12 * 4);
        // 20x12, 10x6, 5x3, 2x1, 1x1: 15 + 6 + 2 + 1 + 1 blocks
        assert_eq!(Astc4x4.chain_bytes(20, 12, 5), 25 * 16);
        assert_eq!(Rgba8.chain_bytes(2048, 2048, 12), (0..12).map(|l| 4 * (2048u64 >> l).pow(2)).sum::<u64>());
    }
}
