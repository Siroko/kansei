//! `kansei-ktx2`: encode images to Basis Universal KTX2 textures with presets per kind of
//! texture, by driving the official `basisu` encoder (`brew install basis_universal`, or a build
//! from https://github.com/BinomialLLC/basis_universal), then report what each device class will
//! hold on the GPU. See docs/ktx2.md.

use kansei_core::loaders::ktx2::{self, BasisCodec, Channels, CompressionSupport, GpuTarget};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: kansei-ktx2 [options] <image>...     encode PNG/JPEG/WebP/TGA (EXR/HDR for --kind hdr) to .ktx2
       kansei-ktx2 --info <file.ktx2>...    describe KTX2 files and their GPU memory per device

options:
  --kind <kind>      auto (default: from the file name), or one of
                       color   bulk colour: ETC1S, sRGB (small files)
                       hero    colour that must look its best: UASTC + RDO + zstd, sRGB
                       normal  normal maps (RGB): UASTC + RDO + zstd, linear, renormalized mips
                       data    linear data (ORM, roughness, masks): UASTC + RDO + zstd, linear
                       rg      two-channel XY normal maps: R and G kept as BC5/EAC RG11
                       hdr     HDR (EXR/HDR input): UASTC HDR 4x4
  --out <file>       output file (one input only; default: next to the input)
  --out-dir <dir>    output directory
  --no-mips          base level only (default: a full mip chain)
  --clamp            clamp instead of wrap at the borders when filtering mips (atlases)
  --q <1-255>        ETC1S quality (default 192)
  --lambda <x>       UASTC RDO lambda: higher is smaller and blurrier, 0 turns RDO off
  --basisu <path>    the basisu binary (default: basisu on PATH)
  -- <args>          further arguments passed to basisu as they are";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Color,
    Hero,
    Normal,
    Data,
    Rg,
    Hdr,
}

impl Kind {
    fn parse(s: &str) -> Option<Option<Self>> {
        Some(match s {
            "auto" => None,
            "color" | "colour" => Some(Kind::Color),
            "hero" => Some(Kind::Hero),
            "normal" => Some(Kind::Normal),
            "data" => Some(Kind::Data),
            "rg" => Some(Kind::Rg),
            "hdr" => Some(Kind::Hdr),
            _ => return None,
        })
    }

    /// Guess from the file name: `*normal*`/`*_n.*` normals, `*orm*`/roughness/metal/occlusion/
    /// masks linear data, EXR/HDR files HDR, everything else colour.
    fn guess(path: &Path) -> Self {
        let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
        let stem = name.rsplit_once('.').map(|(s, _)| s).unwrap_or(&name);
        let ext = path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
        let has = |words: &[&str]| words.iter().any(|w| stem.contains(w));
        if ext == "exr" || ext == "hdr" {
            Kind::Hdr
        } else if has(&["normal", "nrm"]) || stem.ends_with("_n") {
            Kind::Normal
        } else if has(&["orm", "rough", "metal", "occlusion", "_ao", "mask", "height", "specular"]) {
            Kind::Data
        } else {
            Kind::Color
        }
    }

    fn name(self) -> &'static str {
        match self {
            Kind::Color => "color",
            Kind::Hero => "hero",
            Kind::Normal => "normal",
            Kind::Data => "data",
            Kind::Rg => "rg",
            Kind::Hdr => "hdr",
        }
    }

    /// Channels the engine keeps for this kind (what the memory report assumes).
    fn channels(self) -> Option<Channels> {
        (self == Kind::Rg).then_some(Channels::Rg)
    }
}

struct Options {
    kind: Option<Kind>,
    out: Option<PathBuf>,
    out_dir: Option<PathBuf>,
    mips: bool,
    clamp: bool,
    etc1s_q: u32,
    lambda: Option<f32>,
    basisu: String,
    extra: Vec<String>,
}

impl Default for Options {
    fn default() -> Self {
        Self { kind: None, out: None, out_dir: None, mips: true, clamp: false, etc1s_q: 192, lambda: None, basisu: "basisu".into(), extra: vec![] }
    }
}

/// The basisu arguments for a kind (input and output excluded).
fn basisu_args(kind: Kind, o: &Options) -> Vec<String> {
    let uastc = |lambda: f32| {
        let mut a = vec!["-uastc".to_string(), "-uastc_level".into(), "2".into()];
        let lambda = o.lambda.unwrap_or(lambda);
        if lambda > 0.0 {
            a.extend(["-uastc_rdo_l".into(), lambda.to_string()]);
        }
        a
    };
    let mut args: Vec<String> = match kind {
        Kind::Color => vec!["-etc1s".into(), "-srgb".into(), "-q".into(), o.etc1s_q.to_string()],
        Kind::Hero => [uastc(1.0), vec!["-srgb".into()]].concat(),
        Kind::Normal => [uastc(0.5), vec!["-linear".into(), "-normal_map".into(), "-mip_renorm".into()]].concat(),
        Kind::Data => [uastc(1.0), vec!["-linear".into()]].concat(),
        Kind::Rg => [uastc(0.5), vec!["-linear".into(), "-normal_map".into(), "-separate_rg_to_color_alpha".into()]].concat(),
        Kind::Hdr => vec!["-hdr".into()],
    };
    if o.mips {
        args.push("-mipmap".into());
        if o.clamp {
            args.push("-mip_clamp".into());
        }
    }
    args.extend(o.extra.iter().cloned());
    args
}

fn mb(bytes: u64) -> String {
    format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
}

fn kb(bytes: u64) -> String {
    format!("{} KB", (bytes + 512) / 1024)
}

/// "BC 2.67 MB (BC1) · ASTC ... · none ..." for the file's codec and channels.
fn memory_report(bytes: &[u8], channels: Option<Channels>) -> Result<String, String> {
    let info = ktx2::inspect(bytes).map_err(|e| e.to_string())?;
    let channels = channels.unwrap_or(info.default_channels());
    let (w, h, levels) = (info.header.width, info.header.height, info.header.levels);
    let hdr = matches!(info.codec, BasisCodec::UastcHdr | BasisCodec::OtherHdr);
    let devices = [
        ("desktop (BC)", CompressionSupport { bc: true, ..CompressionSupport::NONE }),
        ("Apple (BC+ASTC+ETC2)", CompressionSupport { bc: true, astc: true, etc2: true, astc_hdr: false }),
        ("mobile (ASTC+ETC2)", CompressionSupport { astc: true, etc2: true, ..CompressionSupport::NONE }),
        ("none", CompressionSupport::NONE),
    ];
    let mut parts = vec![];
    for (device, support) in devices {
        let options = ktx2::Ktx2Options { srgb: None, channels: Some(channels) };
        let target = ktx2::choose_target(bytes, &options, support).map_err(|e| e.to_string())?;
        parts.push(format!("{device} {} {}", target.name(), mb(target.chain_bytes(w, h, levels))));
    }
    let uncompressed = if hdr { GpuTarget::Rgba16Float } else { GpuTarget::Rgba8 };
    Ok(format!(
        "GPU memory: {}; uncompressed {} {} with these mips, {} without",
        parts.join(" · "),
        uncompressed.name(),
        mb(uncompressed.chain_bytes(w, h, levels)),
        mb(uncompressed.level_bytes(w, h)),
    ))
}

fn describe(path: &Path) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let info = ktx2::inspect(&bytes).map_err(|e| format!("{}: {e}", path.display()))?;
    let h = &info.header;
    println!(
        "{}: {} {}x{}, {} mips, {}{}, {:?} supercompression, {}",
        path.display(),
        info.codec_name,
        h.width,
        h.height,
        h.levels,
        if h.srgb { "sRGB" } else { "linear" },
        if info.has_alpha { " with alpha" } else { "" },
        h.supercompression,
        kb(bytes.len() as u64),
    );
    println!("  {}", memory_report(&bytes, None)?);
    Ok(())
}

/// A PNG basisu can read: the input itself when it is PNG/JPEG/TGA/EXR/HDR, otherwise a decoded
/// copy in the output directory (removed afterwards).
fn readable_input(input: &Path, out: &Path) -> Result<(PathBuf, bool), String> {
    let ext = input.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    if matches!(ext.as_str(), "png" | "jpg" | "jpeg" | "tga" | "exr" | "hdr" | "qoi") {
        return Ok((input.to_path_buf(), false));
    }
    let img = image::open(input).map_err(|e| format!("{}: {e}", input.display()))?;
    let png = out.with_extension("kansei-ktx2-input.png");
    img.save(&png).map_err(|e| format!("{}: {e}", png.display()))?;
    Ok((png, true))
}

fn encode(input: &Path, o: &Options) -> Result<(), String> {
    let kind = o.kind.unwrap_or_else(|| Kind::guess(input));
    let out = match (&o.out, &o.out_dir) {
        (Some(out), _) => out.clone(),
        (None, Some(dir)) => dir.join(input.file_name().unwrap()).with_extension("ktx2"),
        (None, None) => input.with_extension("ktx2"),
    };
    if let Some(dir) = out.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    let (source, temporary) = readable_input(input, &out)?;
    let args = basisu_args(kind, o);
    let status = Command::new(&o.basisu)
        .arg("-quiet")
        .args(&args)
        .arg("-output_file")
        .arg(&out)
        .arg(&source)
        .stdout(std::process::Stdio::null())
        .status();
    if temporary {
        let _ = std::fs::remove_file(&source);
    }
    match status {
        Ok(s) if s.success() => {}
        Ok(s) => return Err(format!("{}: basisu {} failed ({s})", input.display(), args.join(" "))),
        Err(e) => return Err(format!("could not run {} ({e}); install it with `brew install basis_universal`", o.basisu)),
    }
    let bytes = std::fs::read(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let info = ktx2::inspect(&bytes).map_err(|e| format!("{}: {e}", out.display()))?;
    let input_bytes = std::fs::metadata(input).map(|m| m.len()).unwrap_or(0);
    println!(
        "{} ({}) -> {} [{}: {} {}, {} mips] {}",
        input.display(),
        kb(input_bytes),
        out.display(),
        kind.name(),
        info.codec_name,
        if info.header.srgb { "sRGB" } else { "linear" },
        info.header.levels,
        kb(bytes.len() as u64),
    );
    println!("  {}", memory_report(&bytes, kind.channels())?);
    Ok(())
}

fn parse(args: &[String]) -> Result<(Options, bool, Vec<PathBuf>), String> {
    let mut o = Options::default();
    let (mut info, mut inputs) = (false, vec![]);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or_else(|| format!("{name} needs a value"));
        match a.as_str() {
            "--kind" => o.kind = Kind::parse(&value("--kind")?).ok_or("unknown --kind")?,
            "--out" => o.out = Some(value("--out")?.into()),
            "--out-dir" => o.out_dir = Some(value("--out-dir")?.into()),
            "--no-mips" => o.mips = false,
            "--clamp" => o.clamp = true,
            "--q" => o.etc1s_q = value("--q")?.parse().map_err(|_| "--q takes 1-255")?,
            "--lambda" => o.lambda = Some(value("--lambda")?.parse().map_err(|_| "--lambda takes a number")?),
            "--basisu" => o.basisu = value("--basisu")?,
            "--info" => info = true,
            "-h" | "--help" => return Err(String::new()),
            "--" => {
                o.extra = it.by_ref().cloned().collect();
            }
            flag if flag.starts_with('-') => return Err(format!("unknown option {flag}")),
            path => inputs.push(PathBuf::from(path)),
        }
    }
    if inputs.is_empty() {
        return Err(String::new());
    }
    if o.out.is_some() && inputs.len() > 1 {
        return Err("--out takes one input; use --out-dir for several".into());
    }
    Ok((o, info, inputs))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (options, info, inputs) = match parse(&args) {
        Ok(p) => p,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("{e}");
            }
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    let mut failed = false;
    for input in &inputs {
        let result = if info { describe(input) } else { encode(input, &options) };
        if let Err(e) = result {
            eprintln!("{e}");
            failed = true;
        }
    }
    if failed { ExitCode::FAILURE } else { ExitCode::SUCCESS }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kinds_are_guessed_from_file_names() {
        let guess = |p: &str| Kind::guess(Path::new(p));
        assert_eq!(guess("car/basecolor.webp"), Kind::Color);
        assert_eq!(guess("car/normal.webp"), Kind::Normal);
        assert_eq!(guess("rock_n.png"), Kind::Normal);
        assert_eq!(guess("Rock_ORM.png"), Kind::Data);
        assert_eq!(guess("floor_roughness.jpg"), Kind::Data);
        assert_eq!(guess("sky.exr"), Kind::Hdr);
        // a name that merely contains "n" is colour
        assert_eq!(guess("lantern.png"), Kind::Color);
    }

    #[test]
    fn presets_pick_the_codec_colour_space_and_mips() {
        let o = Options::default();
        let args = |k| basisu_args(k, &o).join(" ");
        assert_eq!(args(Kind::Color), "-etc1s -srgb -q 192 -mipmap");
        assert_eq!(args(Kind::Hero), "-uastc -uastc_level 2 -uastc_rdo_l 1 -srgb -mipmap");
        assert_eq!(args(Kind::Normal), "-uastc -uastc_level 2 -uastc_rdo_l 0.5 -linear -normal_map -mip_renorm -mipmap");
        assert_eq!(args(Kind::Data), "-uastc -uastc_level 2 -uastc_rdo_l 1 -linear -mipmap");
        assert!(args(Kind::Rg).contains("-separate_rg_to_color_alpha"));
        let o = Options { mips: false, lambda: Some(0.0), extra: vec!["-y_flip".into()], ..Options::default() };
        assert_eq!(basisu_args(Kind::Hero, &o).join(" "), "-uastc -uastc_level 2 -srgb -y_flip");
        let o = Options { clamp: true, ..Options::default() };
        assert!(basisu_args(Kind::Color, &o).join(" ").ends_with("-mipmap -mip_clamp"));
    }

    #[test]
    fn options_parse() {
        let a = |s: &str| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let (o, info, inputs) = parse(&a("--kind normal --lambda 2 --out-dir out a.webp b.png -- -stats")).unwrap();
        assert_eq!((o.kind, o.lambda, info, inputs.len()), (Some(Kind::Normal), Some(2.0), false, 2));
        assert_eq!(o.extra, vec!["-stats"]);
        assert!(parse(&a("--out x.ktx2 a.png b.png")).is_err());
        assert!(parse(&a("--kind shiny a.png")).is_err());
        assert!(parse(&a("--info t.ktx2")).unwrap().1);
    }

    #[test]
    fn memory_report_lists_each_device_class() {
        let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/../../kansei-core/tests/fixtures/ktx2/etc1s_rgb.ktx2")).unwrap();
        let report = memory_report(&bytes, None).unwrap();
        assert!(report.contains("desktop (BC) BC1"), "{report}");
        assert!(report.contains("mobile (ASTC+ETC2) ETC2 RGB"), "{report}");
        // 20x12 is on the 4x4 grid; RGBA8 with 5 mips: 4 * (240 + 60 + 15 + 2 + 1) bytes
        assert!(report.contains("none RGBA8 0.00 MB"), "{report}");
    }
}
