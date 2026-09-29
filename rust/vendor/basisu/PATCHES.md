# Kansei's copy of `basisu` 0.1.0

This is the pure-Rust Basis Universal transcoder [`basisu`](https://crates.io/crates/basisu)
0.1.0 (Apache-2.0, see `LICENSE`; https://github.com/marcogomez/basisu), vendored so that Kansei
can leave out the transcode targets it never requests. The crate has no feature flags for that.

The change, marked "Kansei patch" in the source: the dispatch arms of BC3, PVRTC1/2, ATC, FXT1,
RGB565, BGR565, RGBA4444, RGB half and RGB9E5 are removed (`src/dispatch.rs`,
`src/xuastc/transcode.rs`), `support::is_format_supported` reports those targets unsupported,
and `src/lib.rs` allows the dead code that only they used. Nothing references their code or
lookup tables any more, so the linker drops them: a WASM build that transcodes carries
1,238,572 → 995,003 bytes of transcoder at opt-level 3, and 1,031,684 → 807,629 bytes (470 → 364 KB
gzipped) at opt-level "s". (Refusing the targets with an early return instead is not enough:
LLVM prunes the arms at "s" but not at 3.)

The targets Kansei uses (ETC1/ETC2, EAC R11/RG11, BC1, BC4, BC5, BC6H, BC7, ASTC 4x4 LDR and
HDR, RGBA32, RGBA half; `kansei-core/src/loaders/ktx2/basis.rs`) are untouched, and
`kansei-core/tests/ktx2_transcode.rs` checks them byte for byte against the official
transcoder.

To update, copy the new release over this directory and reapply the change.
