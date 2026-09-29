# KTX2 textures

Kansei (the Rust engine, `rust/kansei-core`) loads KTX2 files with Basis Universal
supercompression. That is the web's standard for GPU-compressed textures (glTF
`KHR_texture_basisu`, three.js, Babylon). One file is shipped to every device and transcoded
at load time to whatever block-compressed format that GPU samples:

| Device | Formats it samples | UASTC becomes | ETC1S becomes |
|---|---|---|---|
| Desktop (Windows, Linux, Intel Macs) | BC1-7 | BC7 | BC1 (opaque), BC7 (alpha) |
| Apple Silicon (BC, ASTC and ETC2) | all three | ASTC 4x4 (lossless) | ETC2 (lossless) |
| Android / mobile | ASTC and/or ETC2 | ASTC 4x4, else ETC2 | ETC2 |
| none of them | – | RGBA8 | RGBA8 |

Block-compressed textures stay compressed in GPU memory: 4 or 8 bits per texel against RGBA8's
32. A 2048² texture with mips takes 21.3 MB as RGBA8, 5.3 MB as BC7/ASTC, 2.7 MB as BC1/ETC2.

## Choosing a codec

- **ETC1S** for bulk colour: small files (often smaller than a WebP of the same texture), ETC1
  quality, and ETC2 holds it exactly, or BC1 at half BC7's memory on desktop.
- **UASTC** (with RDO and zstd) for textures that must look their best: hero colour, normal
  maps, and linear data such as ORM. ASTC 4x4 holds it exactly; BC7 is a close transcode. Files
  are larger (~4-8 bits per texel before zstd).
- **UASTC HDR** for HDR images: BC6H, ASTC HDR (native only), or RGBA16F.

Measured on the raggare-web Cadillac's 2048² textures (PSNR against the source):

| Texture | Encoding | File | PSNR |
|---|---|---|---|
| base colour | WebP (source) | 354 KB | – |
| base colour | ETC1S `-q 128` | 455 KB | 33.8 dB |
| base colour | ETC1S `-q 255` | 566 KB | 35.5 dB |
| base colour | UASTC, RDO λ 1 | 2.68 MB | 45.7 dB |
| base colour | UASTC, RDO λ 10 | 2.34 MB | 35.5 dB |
| normal map | ETC1S `-q 255` | 614 KB | 26.8 dB |
| normal map | UASTC, RDO λ 0.5 | 2.98 MB | 38.2 dB |

On noisy, photographic colour, ETC1S at high quality matches heavy-RDO UASTC at a quarter of the
size. Normal maps need UASTC.

## Encoding: `kansei-ktx2`

`rust/tools/ktx2` wraps the official `basisu` encoder (`brew install basis_universal`, or build
it from https://github.com/BinomialLLC/basis_universal) with a preset per kind of texture, and
reports the GPU memory each device class will use:

```sh
cd rust
cargo run --release -p kansei-ktx2-tool -- --out-dir out textures/car/*.webp
cargo run --release -p kansei-ktx2-tool -- --info out/basecolor.ktx2
```

| `--kind` | For | basisu |
|---|---|---|
| `color` | bulk colour | `-etc1s -srgb -q 192` |
| `hero` | colour that must look its best | `-uastc -uastc_level 2 -uastc_rdo_l 1 -srgb` |
| `normal` | RGB normal maps | `-uastc -uastc_rdo_l 0.5 -linear -normal_map -mip_renorm` |
| `data` | linear data (ORM, roughness, masks) | `-uastc -uastc_rdo_l 1 -linear` |

Every UASTC preset also passes `-uastc_rdo_d 65536 -ktx2_zstandard_level 22`: the largest RDO
dictionary and the strongest zstd level make files about 1.5% smaller with the same texels.
On noisy scans (photographed ground, asphalt) UASTC stays near 8 bits per texel whatever the
RDO strength; there, resolution is what trades size (a 512² map is a quarter of a 1K one).
| `rg` | two-channel XY normals (BC5/EAC RG11) | as `normal`, plus `-separate_rg_to_color_alpha` |
| `hdr` | EXR/HDR input | `-hdr` |

`--array --out <file>` encodes all inputs (of equal size) as the layers of one 2D array texture.
`auto` (the default) guesses the kind from the file name (`*normal*`, `*_n.*`, `*orm*`,
`*rough*`, …, `.exr`). Every preset writes a full mip chain (`--no-mips` to skip; `--clamp` for
atlases). `--q` and `--lambda` tune ETC1S quality and UASTC RDO strength, and arguments after `--`
go to basisu unchanged. PNG, JPEG, TGA and EXR/HDR go to basisu directly; the tool decodes WebP
and other formats to a temporary PNG first.

## Loading

```rust
use kansei_core::loaders::ktx2::{self, Ktx2Options};

let support = renderer.compression_support();
let texture = ktx2::transcode("Car/BaseColor", &bytes, &Ktx2Options::color(), support)?;
log::info!("{}", texture.summary()); // codec, chosen format, GPU memory vs uncompressed
material.set_bindable(1, texture.into_texture());
```

- `Renderer` requests whichever of `texture-compression-bc`, `-astc`, `-etc2` (and ASTC HDR,
  native only) the adapter offers. For a device you create yourself, request
  `adapter.features() & CompressionSupport::FEATURES`.
- **Colour space**: `Ktx2Options::color()` loads sRGB (base colour, emissive) and
  `Ktx2Options::linear()` loads linear data (normals, ORM). `Ktx2Options::default()` follows the
  file's DFD transfer function. One- and two-channel formats are always linear.
- **Channels**: `.with_channels(Channels::R)` or `Channels::Rg` allows BC4/BC5/EAC at half the
  memory. Two channels follow the Basis layout (encode with `--kind rg`) and are sampled as
  `.rg` in every format, including the RG8 fallback.
- **Sizes**: WebGPU requires a block-compressed texture's base size to be a multiple of 4. Other
  sizes load as uncompressed texels. Smaller mips are uploaded as whole blocks.
- **2D arrays**: a KTX2 with layers becomes a `texture_2d_array` (`TranscodedTexture::layers`;
  each level holds every layer in turn).
- `ktx2::inspect` reads a file's codec, levels, layers and supercompression without
  transcoding. `ktx2::choose_target` gives the target without transcoding.
  `ktx2::transcode_levels` forces a target. `ktx2::transcode_level` transcodes one level, for
  example a small level as RGBA8 for a texture's mean colour.
- Not yet supported: cubemaps, 3D textures and Basis video (the loader returns
  `Ktx2Error::Unsupported`).

### glTF (`KHR_texture_basisu`)

`GLTFLoader` fills each material's `base_color_texture`, `metallic_roughness_texture`,
`normal_texture`, `occlusion_texture` and `emissive_texture`. When a texture carries
`KHR_texture_basisu`, the reference points at the KTX2 image and keeps the PNG/JPEG source as
`fallback_image`. Images are read from buffer views, data URIs or (native) files.
`GLTFResult::load_texture(&tex, support)` transcodes or decodes one in its slot's colour space.
On WASM, images at external URIs are fetched by the caller and passed to
`GLTFResult::set_image_data`.

## The transcoder

Transcoding uses the pure-Rust [`basisu`](https://crates.io/crates/basisu) crate (Apache-2.0), a
port of the reference C++ transcoder. It builds for `wasm32-unknown-unknown` with no C toolchain.
The official C++ transcoder would need LLVM clang with a WebAssembly backend in every build,
which Apple's clang and the Vercel build image lack. Only `loaders/ktx2/basis.rs` touches the
crate, so another backend (the official C++ through `basisu_c_sys`, say) replaces that one file.

The crate is vendored in `rust/vendor/basisu` with the transcode targets Kansei never requests
removed (PVRTC, ATC, FXT1, BC3, the 16-bit and RGB half/9E5 formats; see its `PATCHES.md`), so
their code and lookup tables stay out of WASM builds.

The crate is young (0.1, July 2026), so its output is checked against the official transcoder
rather than trusted:

- `tests/fixtures/ktx2/generate.py` encodes tiny fixtures (20x12 with mips down to 1x1; ETC1S,
  UASTC with alpha, a UASTC normal map, UASTC HDR, and two-layer ETC1S and UASTC arrays) with
  the official `basisu`. It records
  `basisu -unpack`'s output, and the output of the official transcoder built at the same tag with
  strict IEEE float, for every target the engine uses.
- `tests/ktx2_transcode.rs` asserts that every target matches the official transcoder byte for
  byte.
- On arm64, clang fuses multiply-adds by default. The Homebrew `basisu` therefore picks
  different BC7 p-bits for a few UASTC blocks with alpha than the same source built without
  fused multiply-adds (the x86 and emscripten/WASM builds). The crate matches the latter; the
  test allows this difference for BC7 only.
- `tests/ktx2_gpu.rs` uploads every target the GPU supports with its full mip chain. It checks
  the sampled texels against the official CPU decode of the same blocks, sRGB decoding, and the
  HDR formats.

The trimmed transcoder is about 0.8 MB of WASM at opt-level "s" (364 KB gzipped), about 1 MB at
opt-level 3, in a build that loads KTX2.
