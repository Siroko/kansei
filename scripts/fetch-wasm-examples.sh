#!/usr/bin/env bash
# Copies the prebuilt Rust/WASM examples into dist/examples/<name>/ for the site build.
#
# The site-wasm GitHub Actions workflow builds them (scripts/build-wasm-examples.sh) and
# force-pushes the result to the orphan site-wasm branch; this only downloads that branch, so the
# Vercel build needs no Rust toolchain. WASM_EXAMPLES_DIR=<dir> copies a local build instead.
# Without either, the site still builds: it warns and leaves the Rust examples out.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
dest="$root/dist/examples"
mkdir -p "$dest"

src="${WASM_EXAMPLES_DIR:-}"
if [[ -z "$src" ]]; then
    url="${WASM_EXAMPLES_URL:-https://codeload.github.com/Siroko/kansei/tar.gz/refs/heads/site-wasm}"
    tmp="$(mktemp -d)"
    trap 'rm -rf "$tmp"' EXIT
    if ! curl -fsSL --retry 3 "$url" | tar -xz -C "$tmp"; then
        echo "warning: could not download the prebuilt Rust/WASM examples from $url; the site has none" >&2
        exit 0
    fi
    src="$(echo "$tmp"/*/)"
fi

copied=0
for dir in "$src"/*/; do
    name="$(basename "$dir")"
    rm -rf "${dest:?}/${name:?}"
    cp -R "$dir" "$dest/$name"
    copied=$((copied + 1))
done

built="$(sed -n 's/.*"rust_tree": "\([^"]*\)".*/\1/p' "$src/manifest.json" 2>/dev/null || true)"
here="$(git -C "$root" rev-parse HEAD:rust 2>/dev/null || true)"
echo "copied $copied Rust/WASM examples into dist/examples (built from rust/ tree ${built:-unknown})"
if [[ -n "$built" && -n "$here" && "$built" != "$here" ]]; then
    echo "warning: they were built from another rust/ tree than this commit's ($here); the site-wasm workflow is still building, or failed" >&2
fi
