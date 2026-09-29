//! KTX2 textures with Basis Universal supercompression (ETC1S, UASTC, UASTC HDR), transcoded at
//! load to the best block-compressed format the device supports: BC7/BC1/BC4/BC5/BC6H, ASTC 4x4,
//! ETC2/EAC, or uncompressed texels when none fits. See `docs/ktx2.md` for choosing codecs and
//! the encoding tool.
//!
//! ```ignore
//! let support = CompressionSupport::of_device(renderer.device());
//! let tex = ktx2::transcode("Car/BaseColor", &bytes, &Ktx2Options::color(), support)?;
//! log::info!("{}", tex.summary());
//! material.set_bindable(1, tex.into_texture());
//! ```
//!
//! The device must have been created with the compression features (`Renderer` requests
//! whichever of `CompressionSupport::FEATURES` the adapter offers).

mod basis;
mod container;
mod select;

pub use container::{is_ktx2, Ktx2Header, Supercompression};
pub use select::{preferences, select_target, BasisCodec, Channels, CompressionSupport, GpuTarget};

use crate::buffers::Texture;
use basis::BasisFile;

/// Why a KTX2 file could not be loaded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ktx2Error {
    NotKtx2,
    Truncated(&'static str),
    /// A valid file this loader does not handle (cubemaps, arrays, video, non-Basis payloads).
    Unsupported(String),
    /// The Basis transcoder rejected the payload.
    Transcoder(String),
}

impl std::fmt::Display for Ktx2Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Ktx2Error::NotKtx2 => write!(f, "not a KTX2 file"),
            Ktx2Error::Truncated(what) => write!(f, "KTX2 file truncated in its {what}"),
            Ktx2Error::Unsupported(what) => write!(f, "unsupported KTX2 file: {what}"),
            Ktx2Error::Transcoder(what) => write!(f, "Basis transcode failed: {what}"),
        }
    }
}

impl std::error::Error for Ktx2Error {}

/// How the texture is used, which the file cannot always say for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Ktx2Options {
    /// Sample as sRGB colour (`true`: base colour, emissive) or linear data (`false`: normals,
    /// roughness/metalness/occlusion). `None` follows the file's transfer function.
    pub srgb: Option<bool>,
    /// The channels to keep; `None` means RGB, or RGBA when the file has alpha. `R` and `Rg`
    /// let BC4/BC5/EAC hold one- and two-channel data at half the memory.
    pub channels: Option<Channels>,
}

impl Ktx2Options {
    /// Colour: sRGB, channels from the file.
    pub fn color() -> Self {
        Self { srgb: Some(true), channels: None }
    }

    /// Linear data (normal maps, ORM): channels from the file.
    pub fn linear() -> Self {
        Self { srgb: Some(false), channels: None }
    }

    pub fn with_channels(mut self, channels: Channels) -> Self {
        self.channels = Some(channels);
        self
    }
}

/// What a KTX2 file holds, read without transcoding.
#[derive(Debug, Clone, PartialEq)]
pub struct Ktx2Info {
    pub header: Ktx2Header,
    pub codec: BasisCodec,
    /// "ETC1S", "UASTC", "UASTC HDR", ...
    pub codec_name: String,
    pub has_alpha: bool,
}

impl Ktx2Info {
    /// The channels kept when the options name none.
    pub fn default_channels(&self) -> Channels {
        if self.has_alpha { Channels::Rgba } else { Channels::Rgb }
    }
}

/// Read a Basis KTX2 file's header and codec.
pub fn inspect(bytes: &[u8]) -> Result<Ktx2Info, Ktx2Error> {
    let header = Ktx2Header::parse(bytes)?;
    let file = BasisFile::open(bytes).map_err(|e| not_basis(&header, e))?;
    Ok(Ktx2Info { codec: file.codec(), codec_name: file.codec_name(), has_alpha: file.has_alpha(), header })
}

fn not_basis(header: &Ktx2Header, e: Ktx2Error) -> Ktx2Error {
    match header.vk_format {
        0 => e,
        vk => Ktx2Error::Unsupported(format!("vkFormat {vk} without a Basis payload ({e})")),
    }
}

/// The GPU target `bytes` would become on a device with `support`, without transcoding.
pub fn choose_target(bytes: &[u8], options: &Ktx2Options, support: CompressionSupport) -> Result<GpuTarget, Ktx2Error> {
    let header = Ktx2Header::parse(bytes)?;
    let file = BasisFile::open(bytes).map_err(|e| not_basis(&header, e))?;
    Ok(target_for(&header, &file, options, support))
}

fn target_for(header: &Ktx2Header, file: &BasisFile, options: &Ktx2Options, support: CompressionSupport) -> GpuTarget {
    let channels = options.channels.unwrap_or(if file.has_alpha() { Channels::Rgba } else { Channels::Rgb });
    select_target(file.codec(), channels, (header.width, header.height), support, |t| file.can_transcode(t))
}

/// Every mip transcoded to `target`, whatever the device supports (for tools, tests and forcing
/// a fallback): per level, each array layer's data in turn. Compressed levels are rounded up to
/// whole blocks.
pub fn transcode_levels(bytes: &[u8], target: GpuTarget) -> Result<Vec<Vec<u8>>, Ktx2Error> {
    let header = Ktx2Header::parse(bytes)?;
    (0..header.levels).map(|level| transcode_level(bytes, level, target)).collect()
}

/// One mip `level` transcoded to `target`, each array layer's data in turn (a small level
/// as `Rgba8` gives a texture's mean colour cheaply, say).
pub fn transcode_level(bytes: &[u8], level: u32, target: GpuTarget) -> Result<Vec<u8>, Ktx2Error> {
    let header = Ktx2Header::parse(bytes)?;
    let file = BasisFile::open(bytes).map_err(|e| not_basis(&header, e))?;
    if !file.can_transcode(target) {
        return Err(Ktx2Error::Unsupported(format!("{} cannot become {}", file.codec_name(), target.name())));
    }
    if level >= header.levels {
        return Err(Ktx2Error::Unsupported(format!("level {level} of {}", header.levels)));
    }
    level_data(&file, level, header.layers.max(1), target)
}

fn level_data(file: &BasisFile, level: u32, layers: u32, target: GpuTarget) -> Result<Vec<u8>, Ktx2Error> {
    let mut data = Vec::new();
    for layer in 0..layers {
        data.extend(file.transcode(level, layer, target)?);
    }
    Ok(data)
}

/// Transcode every mip of a 2D (or 2D array) Basis KTX2 texture for a device with `support`.
pub fn transcode(
    label: &str,
    bytes: &[u8],
    options: &Ktx2Options,
    support: CompressionSupport,
) -> Result<TranscodedTexture, Ktx2Error> {
    let header = Ktx2Header::parse(bytes)?;
    if header.faces > 1 || header.depth > 1 {
        return Err(Ktx2Error::Unsupported(format!(
            "{label}: {} faces, depth {} (2D textures and 2D arrays load so far)",
            header.faces, header.depth
        )));
    }
    let file = BasisFile::open(bytes).map_err(|e| not_basis(&header, e))?;
    if file.is_video() {
        return Err(Ktx2Error::Unsupported(format!("{label}: a Basis video")));
    }
    let hdr = matches!(file.codec(), BasisCodec::UastcHdr | BasisCodec::OtherHdr);
    let srgb = !hdr && options.srgb.unwrap_or(header.srgb);
    if options.srgb.is_some_and(|s| s != header.srgb) && !hdr {
        log::warn!(
            "{label}: encoded as {} but loaded as {} (check the encoder's -srgb/-linear)",
            if header.srgb { "sRGB" } else { "linear" },
            if srgb { "sRGB" } else { "linear" }
        );
    }
    let target = target_for(&header, &file, options, support);
    let layers = header.layers.max(1);
    let levels = (0..header.levels).map(|l| level_data(&file, l, layers, target)).collect::<Result<Vec<_>, _>>()?;
    Ok(TranscodedTexture {
        label: label.to_string(),
        codec: file.codec_name(),
        target,
        format: target.format(srgb),
        width: header.width,
        height: header.height,
        layers: (header.layers > 0).then_some(header.layers),
        levels,
        file_bytes: bytes.len(),
    })
}

/// A transcoded texture, ready to upload.
#[derive(Debug, Clone)]
pub struct TranscodedTexture {
    pub label: String,
    /// The file's source codec ("ETC1S", "UASTC", ...).
    pub codec: String,
    pub target: GpuTarget,
    pub format: wgpu::TextureFormat,
    pub width: u32,
    pub height: u32,
    /// The layer count of a 2D array texture (bound as `texture_2d_array`); `None` for a 2D
    /// texture.
    pub layers: Option<u32>,
    /// Each mip level's blocks (or texels), tightly packed, level 0 first; within a level, each
    /// array layer in turn.
    pub levels: Vec<Vec<u8>>,
    /// Size of the KTX2 file.
    pub file_bytes: usize,
}

impl TranscodedTexture {
    /// GPU memory the texture takes, all mips.
    pub fn gpu_bytes(&self) -> u64 {
        self.levels.iter().map(|l| l.len() as u64).sum()
    }

    /// GPU memory the same mips would take uncompressed (RGBA8, or RGBA16F for HDR).
    pub fn uncompressed_bytes(&self) -> u64 {
        let hdr = matches!(self.target, GpuTarget::Bc6h | GpuTarget::AstcHdr4x4 | GpuTarget::Rgba16Float);
        let base = if hdr { GpuTarget::Rgba16Float } else { GpuTarget::Rgba8 };
        base.chain_bytes(self.width, self.height, self.levels.len() as u32) * self.layers.unwrap_or(1) as u64
    }

    /// One line for logs and HUDs.
    pub fn summary(&self) -> String {
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        format!(
            "{}: {} {}x{}{} ({} mips, {:.2} MB file) -> {:?}, {:.2} MB on the GPU (uncompressed {:.2} MB)",
            self.label,
            self.codec,
            self.width,
            self.height,
            self.layers.map(|l| format!("x{l} layers")).unwrap_or_default(),
            self.levels.len(),
            mb(self.file_bytes as u64),
            self.format,
            mb(self.gpu_bytes()),
            mb(self.uncompressed_bytes()),
        )
    }

    /// A `Texture` with every mip (and layer), uploaded when first bound.
    pub fn into_texture(self) -> Texture {
        match self.layers {
            Some(layers) => Texture::from_array_levels(&self.label, self.format, self.width, self.height, layers, self.levels),
            None => Texture::from_levels(&self.label, self.format, self.width, self.height, self.levels),
        }
    }
}
