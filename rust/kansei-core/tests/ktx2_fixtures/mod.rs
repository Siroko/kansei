//! The KTX2 fixtures and their goldens (`fixtures/ktx2/generate.py` writes both).

use kansei_core::loaders::ktx2::GpuTarget;
use std::collections::BTreeMap;

pub fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(format!("{}/tests/fixtures/ktx2/{name}", env!("CARGO_MANIFEST_DIR"))).unwrap()
}

/// Entry name ("cli/BC7_RGBA", "oracle/BC7_RGBA", "decoded/BC7_RGBA") -> levels, from a KGOLDEN1 file.
pub fn golden(name: &str) -> BTreeMap<String, Vec<Vec<u8>>> {
    let b = fixture(&format!("{name}.golden"));
    assert_eq!(&b[..8], b"KGOLDEN1");
    let u32_at = |p: usize| u32::from_le_bytes(b[p..p + 4].try_into().unwrap()) as usize;
    let mut p = 12;
    let mut out = BTreeMap::new();
    for _ in 0..u32_at(8) {
        let n = b[p] as usize;
        let key = String::from_utf8(b[p + 1..p + 1 + n].to_vec()).unwrap();
        p += 1 + n;
        let count = u32_at(p);
        p += 4;
        let mut levels = vec![];
        for _ in 0..count {
            let len = u32_at(p);
            levels.push(b[p + 4..p + 4 + len].to_vec());
            p += 4 + len;
        }
        out.insert(key, levels);
    }
    out
}

/// The engine target for a Basis `transcoder_texture_format` name.
pub fn target(basis_name: &str) -> GpuTarget {
    match basis_name {
        "ETC1_RGB" => GpuTarget::Etc2Rgb,
        "ETC2_RGBA" => GpuTarget::Etc2Rgba,
        "BC1_RGB" => GpuTarget::Bc1,
        "BC4_R" => GpuTarget::Bc4,
        "BC5_RG" => GpuTarget::Bc5,
        "BC7_RGBA" => GpuTarget::Bc7,
        "ASTC_LDR_4X4_RGBA" => GpuTarget::Astc4x4,
        "RGBA32" => GpuTarget::Rgba8,
        "ETC2_EAC_R11" => GpuTarget::EacR11,
        "ETC2_EAC_RG11" => GpuTarget::EacRg11,
        "BC6H" => GpuTarget::Bc6h,
        "ASTC_HDR_4X4_RGBA" => GpuTarget::AstcHdr4x4,
        "RGBA_HALF" => GpuTarget::Rgba16Float,
        other => panic!("unknown golden target {other}"),
    }
}
