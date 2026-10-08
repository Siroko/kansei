# Basis Universal transcoder (official build)

`basis_transcoder.js` and `basis_transcoder.wasm` are Binomial's prebuilt WebAssembly transcoder,
copied unchanged from https://github.com/BinomialLLC/basis_universal at tag **`v2_50`**
(`webgl/transcoder/build/`), Apache-2.0 (`LICENSE`). That is the tag whose `basisu` encoded the
Rust engine's KTX2 fixtures and whose transcoder `rust/kansei-core/tests/ktx2_transcode.rs`
compares against (`tests/fixtures/ktx2/generate.py`, `BASIS_TAG`).

| File | SHA-256 of the upstream file |
|---|---|
| `basis_transcoder.js` | `720dd9bd09c7cada6d87f1b7b70cec713df04da88cd641ac3212559353834dc8` |
| `basis_transcoder.wasm` | `a0f65d4a30ecb3269d01ead7d0a3477d2b0208146d083625a90623f473f6c139` |

Kansei patch: two lines appended to `basis_transcoder.js` (`export default BASIS;`) so it loads
as an ES module; `basis_transcoder.d.ts` types the factory. Only `../BasisModule.ts` imports them.

Binomial builds it with two options off (`webgl/transcoder/CMakeLists.txt`), so it differs from
the Rust engine's transcoder in two places, both checked against the Rust/wasm build in a browser
(every other target of the seven Rust fixtures and the example's textures is byte-identical):

- `BASISD_SUPPORT_ASTC_HIGHER_OPAQUE_QUALITY=0`: opaque ETC1S to ASTC 4x4 picks endpoints from
  the smaller [0, 47] table, so some blocks differ from Rust's (which uses the [0, 255] one).
  ETC1S with alpha to ASTC matches.
- `BASISD_SUPPORT_ETC2_EAC_RG11=0`: ETC1S cannot become EAC R11/RG11 (UASTC can), so one- and
  two-channel ETC1S textures on an ETC2-only device fall back to R8/RG8 where Rust picks EAC.

To update, copy the two files from the new tag's `webgl/transcoder/build/`, reapply the patch,
update the hashes, and check `examples/index_ktx2.html` with every `support=` value.
