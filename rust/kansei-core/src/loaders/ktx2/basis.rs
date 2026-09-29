//! The Basis Universal transcoder backend: the only file that knows which transcoder decodes the
//! payload. It is the pure-Rust `basisu` crate, a port of the reference C++ transcoder whose
//! output the fixture tests (`tests/ktx2_transcode.rs`) check byte for byte against the official
//! one. It builds for wasm32 with no C toolchain. Swapping in another backend (the official C++
//! through `basisu_c_sys`, say) means reimplementing `BasisFile` here.

use super::select::{BasisCodec, GpuTarget};
use super::Ktx2Error;

/// An opened Basis Universal texture.
pub struct BasisFile<'a> {
    inner: basisu::Transcoder<'a>,
}

impl<'a> BasisFile<'a> {
    pub fn open(bytes: &'a [u8]) -> Result<Self, Ktx2Error> {
        basisu::Transcoder::new(bytes)
            .map(|inner| Self { inner })
            .map_err(|e| Ktx2Error::Transcoder(format!("{e:?}")))
    }

    pub fn codec(&self) -> BasisCodec {
        use basisu::SourceFormat as S;
        match self.inner.source_format() {
            S::Etc1s => BasisCodec::Etc1s,
            S::UastcLdr => BasisCodec::UastcLdr,
            S::UastcHdr4x4 => BasisCodec::UastcHdr,
            S::AstcHdr6x6 | S::UastcHdr6x6 => BasisCodec::OtherHdr,
            _ => BasisCodec::OtherLdr,
        }
    }

    /// A short name of the source codec ("ETC1S", "UASTC", ...).
    pub fn codec_name(&self) -> String {
        use basisu::SourceFormat as S;
        match self.inner.source_format() {
            S::Etc1s => "ETC1S".into(),
            S::UastcLdr => "UASTC".into(),
            S::UastcHdr4x4 => "UASTC HDR".into(),
            other => format!("{other:?}"),
        }
    }

    pub fn has_alpha(&self) -> bool {
        self.inner.has_alpha()
    }

    pub fn is_video(&self) -> bool {
        self.inner.is_video()
    }

    /// Whether the transcoder can produce `target` from this file (a direct transcode, or the
    /// uncompressed decode the R8/RG8/RGBA8/RGBA16F fallbacks are cut from).
    pub fn can_transcode(&self, target: GpuTarget) -> bool {
        self.inner.supports(transcoder_format(target))
    }

    /// Mip `level` of array layer `layer` (face 0) as `target`'s texels or blocks, tightly packed.
    pub fn transcode(&self, level: u32, layer: u32, target: GpuTarget) -> Result<Vec<u8>, Ktx2Error> {
        let raw = self
            .inner
            .transcode_image(level, layer, 0, transcoder_format(target), basisu::DecodeFlags::NONE)
            .map_err(|e| Ktx2Error::Transcoder(format!("level {level} layer {layer} to {}: {e:?}", target.name())))?;
        Ok(match target {
            // cut from RGBA: R in red; the Basis two-channel layout has G in alpha
            GpuTarget::R8 => raw.chunks_exact(4).map(|p| p[0]).collect(),
            GpuTarget::Rg8 => raw.chunks_exact(4).flat_map(|p| [p[0], p[3]]).collect(),
            _ => raw,
        })
    }
}

fn transcoder_format(target: GpuTarget) -> basisu::TargetFormat {
    use basisu::TargetFormat as T;
    match target {
        GpuTarget::Bc1 => T::Bc1Rgb,
        GpuTarget::Bc4 => T::Bc4R,
        GpuTarget::Bc5 => T::Bc5Rg,
        GpuTarget::Bc6h => T::Bc6h,
        GpuTarget::Bc7 => T::Bc7Rgba,
        GpuTarget::Astc4x4 => T::Astc4x4Rgba,
        GpuTarget::AstcHdr4x4 => T::AstcHdr4x4Rgba,
        GpuTarget::Etc2Rgb => T::Etc1Rgb,
        GpuTarget::Etc2Rgba => T::Etc2Rgba,
        GpuTarget::EacR11 => T::EacR11,
        GpuTarget::EacRg11 => T::EacRg11,
        GpuTarget::Rgba8 | GpuTarget::Rg8 | GpuTarget::R8 => T::Rgba32,
        GpuTarget::Rgba16Float => T::RgbaHalf,
    }
}
