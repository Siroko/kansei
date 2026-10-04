# KTX2 textures

Three KTX2 (Basis Universal) textures on quads: an ETC1S colour texture, a UASTC colour texture
with alpha over a checkerboard, and a UASTC normal map lit by a circling light. Each is transcoded
to the best compressed format the device samples, and the page lists what each became and its GPU
memory against RGBA8.

Engine API: `loaders::ktx2::transcode` (`Ktx2Options::color` / `linear`, `CompressionSupport`),
`TranscodedTexture` (`into_texture`, `gpu_bytes`, `uncompressed_bytes`, `summary`),
`Renderer::compression_support`, `Sampler::with_anisotropy`, `kansei_wasm::fetch_bytes`.

| URL parameter | Effect |
|---|---|
| `support=none\|bc\|astc\|etc2` | transcode as if the device had only that format family (`none`: uncompressed), to show the fallbacks; default every format the device has |
| `dpr=<ratio>` | drawing-buffer pixels per CSS pixel (default: the screen's, at most 2) |

Controls: none; the links under the report switch `support=`.

Assets: `www/assets/card_etc1s.ktx2`, `badge_uastc.ktx2` and `bumps_normal.ktx2`, procedural
images generated and encoded by `www/assets/make_assets.py` (needs Pillow and `basisu`; see
`docs/ktx2.md`).

Build: `wasm-pack build --target web --release` here, serve this folder, open `www/`.
