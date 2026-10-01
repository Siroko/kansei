//! KTX2 transcodes checked byte for byte against the official Basis Universal transcoder, on the
//! fixtures `tests/fixtures/ktx2/generate.py` encodes with the official `basisu` tool. Each golden
//! holds, per target, what `basisu -unpack` wrote ("cli/") and what the official transcoder built
//! with strict IEEE float wrote ("oracle/"); see generate.py for why both.

use kansei_core::loaders::ktx2::{
    self, BasisCodec, Channels, CompressionSupport, GpuTarget, Ktx2Error, Ktx2Options, Supercompression,
};

mod ktx2_fixtures;
use ktx2_fixtures::{fixture, golden, target};

const FIXTURES: [&str; 7] = ["etc1s_rgb", "etc1s_rgba", "uastc_rgba", "uastc_normal", "uastc_hdr", "etc1s_array", "uastc_array"];

fn differing_blocks(a: &[Vec<u8>], b: &[Vec<u8>]) -> usize {
    a.iter().zip(b).map(|(x, y)| x.chunks(16).zip(y.chunks(16)).filter(|(p, q)| p != q).count()).sum()
}

#[test]
fn every_target_matches_the_official_transcoder_byte_for_byte() {
    let mut checked = 0;
    for name in FIXTURES {
        let bytes = fixture(&format!("{name}.ktx2"));
        let goldens = golden(name);
        for (key, oracle) in goldens.iter().filter(|(k, _)| k.starts_with("oracle/")) {
            let basis_name = &key["oracle/".len()..];
            let ours = ktx2::transcode_levels(&bytes, target(basis_name)).unwrap();
            assert_eq!(ours.len(), oracle.len(), "{name} {basis_name}: level count");
            for (level, (o, g)) in ours.iter().zip(oracle).enumerate() {
                assert!(o == g, "{name} {basis_name} level {level}: differs from the official transcoder");
            }
            if let Some(cli) = goldens.get(&format!("cli/{basis_name}")) {
                if cli == oracle {
                    checked += 1;
                } else {
                    // arm64 builds of basisu fuse multiply-adds into different BC7 p-bits
                    assert_eq!(basis_name, "BC7_RGBA", "{name}: `basisu -unpack` and the strict build differ");
                    eprintln!(
                        "{name} {basis_name}: identical to the strict-float official build; {} of {} blocks differ from this machine's basisu -unpack (FMA p-bits)",
                        differing_blocks(&ours, cli),
                        cli.iter().map(|l| l.len() / 16).sum::<usize>()
                    );
                }
            }
        }
    }
    eprintln!("{checked} (fixture, target) pairs identical to basisu -unpack");
    assert!(checked >= 60);
}

#[test]
fn one_and_two_channel_fallbacks_are_cut_from_the_rgba_decode() {
    for name in ["etc1s_rgba", "uastc_rgba"] {
        let bytes = fixture(&format!("{name}.ktx2"));
        let rgba = &golden(name)["cli/RGBA32"];
        let r8 = ktx2::transcode_levels(&bytes, GpuTarget::R8).unwrap();
        let rg8 = ktx2::transcode_levels(&bytes, GpuTarget::Rg8).unwrap();
        for (level, px) in rgba.iter().enumerate() {
            let want_r: Vec<u8> = px.chunks(4).map(|p| p[0]).collect();
            let want_rg: Vec<u8> = px.chunks(4).flat_map(|p| [p[0], p[3]]).collect();
            assert_eq!(r8[level], want_r, "{name} R8 level {level}");
            assert_eq!(rg8[level], want_rg, "{name} RG8 level {level}");
        }
    }
}

#[test]
fn inspect_reads_codec_levels_supercompression_and_colour_space() {
    let cases = [
        ("etc1s_rgb", BasisCodec::Etc1s, Supercompression::BasisLz, false, true, (20, 12), 5),
        ("etc1s_rgba", BasisCodec::Etc1s, Supercompression::BasisLz, true, true, (20, 12), 5),
        ("uastc_rgba", BasisCodec::UastcLdr, Supercompression::Zstandard, true, true, (20, 12), 5),
        ("uastc_normal", BasisCodec::UastcLdr, Supercompression::Zstandard, false, false, (8, 8), 4),
        ("uastc_hdr", BasisCodec::UastcHdr, Supercompression::Zstandard, false, false, (20, 12), 5),
    ];
    let array = ktx2::inspect(&fixture("etc1s_array.ktx2")).unwrap();
    assert_eq!((array.header.layers, array.header.levels, array.has_alpha), (2, 5, true));
    for (name, codec, sc, alpha, srgb, (w, h), levels) in cases {
        let info = ktx2::inspect(&fixture(&format!("{name}.ktx2"))).unwrap();
        assert_eq!(info.codec, codec, "{name}");
        assert_eq!(info.header.supercompression, sc, "{name}");
        assert_eq!(info.has_alpha, alpha, "{name}");
        assert_eq!(info.header.srgb, srgb, "{name}");
        assert_eq!((info.header.width, info.header.height, info.header.levels), (w, h, levels), "{name}");
    }
}

#[test]
fn transcode_picks_the_target_and_colour_space_for_the_device() {
    let all = CompressionSupport { bc: true, astc: true, etc2: true, astc_hdr: false };
    let bc = CompressionSupport { bc: true, ..CompressionSupport::NONE };
    let t = |name: &str, options: Ktx2Options, support| {
        ktx2::transcode(name, &fixture(&format!("{name}.ktx2")), &options, support).unwrap()
    };
    use wgpu::TextureFormat as F;

    // the file's transfer function by default
    assert_eq!(t("uastc_rgba", Ktx2Options::default(), bc).format, F::Bc7RgbaUnormSrgb);
    assert_eq!(t("uastc_normal", Ktx2Options::default(), bc).format, F::Bc7RgbaUnorm);
    // the usage overrides it
    assert_eq!(t("uastc_rgba", Ktx2Options::linear(), bc).format, F::Bc7RgbaUnorm);
    assert_eq!(t("etc1s_rgb", Ktx2Options::color(), bc).format, F::Bc1RgbaUnormSrgb);
    assert_eq!(t("etc1s_rgb", Ktx2Options::color(), all).format, F::Etc2Rgb8UnormSrgb);
    assert_eq!(t("etc1s_rgba", Ktx2Options::color(), all).format, F::Etc2Rgba8UnormSrgb);
    // no compression: uncompressed texels in the right colour space
    let rgba = t("uastc_rgba", Ktx2Options::color(), CompressionSupport::NONE);
    assert_eq!((rgba.target, rgba.format), (GpuTarget::Rgba8, F::Rgba8UnormSrgb));
    assert_eq!(rgba.levels[0].len(), 20 * 12 * 4);
    // one and two channels
    assert_eq!(t("uastc_normal", Ktx2Options::linear().with_channels(Channels::Rg), all).format, F::Bc5RgUnorm);
    assert_eq!(t("etc1s_rgb", Ktx2Options::linear().with_channels(Channels::R), all).format, F::EacR11Unorm);
    // HDR
    assert_eq!(t("uastc_hdr", Ktx2Options::default(), all).format, F::Bc6hRgbUfloat);
    assert_eq!(t("uastc_hdr", Ktx2Options::default(), CompressionSupport::NONE).format, F::Rgba16Float);
}

#[test]
fn arrays_hold_every_layer_of_every_level() {
    let bytes = fixture("uastc_array.ktx2");
    let bc = CompressionSupport { bc: true, ..CompressionSupport::NONE };
    let t = ktx2::transcode("Array", &bytes, &Ktx2Options::color(), bc).unwrap();
    assert_eq!((t.target, t.layers, t.levels.len()), (GpuTarget::Bc7, Some(2), 5));
    assert_eq!(t.gpu_bytes(), 2 * 25 * 16);
    assert_eq!(t.uncompressed_bytes(), 2 * 4 * (20 * 12 + 10 * 6 + 5 * 3 + 2 + 1));
    assert!(t.summary().contains("20x12x2 layers"), "{}", t.summary());
    // layer 1 is rgb.png: opaque, unlike layer 0
    let level0 = ktx2::transcode_level(&bytes, 0, GpuTarget::Rgba8).unwrap();
    let (layer0, layer1) = level0.split_at(20 * 12 * 4);
    assert!(layer0.chunks(4).any(|p| p[3] < 200));
    assert!(layer1.chunks(4).all(|p| p[3] == 255));
    // a single level, all layers
    assert_eq!(ktx2::transcode_level(&bytes, 4, GpuTarget::Rgba8).unwrap().len(), 2 * 4);
}

#[test]
fn gpu_memory_is_reported_against_the_uncompressed_chain() {
    let bytes = fixture("uastc_rgba.ktx2");
    let bc7 = ktx2::transcode("t", &bytes, &Ktx2Options::color(), CompressionSupport { bc: true, ..CompressionSupport::NONE }).unwrap();
    // 20x12 .. 1x1: 15 + 6 + 2 + 1 + 1 blocks of 16 bytes
    assert_eq!(bc7.gpu_bytes(), 25 * 16);
    assert_eq!(bc7.uncompressed_bytes(), 4 * (20 * 12 + 10 * 6 + 5 * 3 + 2 + 1));
    let hdr = ktx2::transcode("h", &fixture("uastc_hdr.ktx2"), &Ktx2Options::default(), CompressionSupport::NONE).unwrap();
    assert_eq!(hdr.gpu_bytes(), hdr.uncompressed_bytes());
    assert!(bc7.summary().contains("UASTC 20x12 (5 mips"), "{}", bc7.summary());
}

#[test]
fn other_files_are_refused_with_a_reason() {
    assert_eq!(ktx2::transcode("x", &fixture("rgb.png"), &Ktx2Options::default(), CompressionSupport::NONE).unwrap_err(), Ktx2Error::NotKtx2);
    let mut cut = fixture("uastc_rgba.ktx2");
    cut.truncate(100);
    assert!(matches!(ktx2::inspect(&cut), Err(Ktx2Error::Truncated(_))));
    assert!(matches!(ktx2::transcode_levels(&fixture("uastc_hdr.ktx2"), GpuTarget::Bc7), Err(Ktx2Error::Unsupported(_))));
}
