#!/usr/bin/env bash
# Builds every Rust/WASM example under rust/kansei-wasm/examples into a static tree served at
# kansei.graphics/examples/<name>/: each example's www/ page and files next to its wasm-pack pkg/.
#
#   scripts/build-wasm-examples.sh [out-dir]    (default: build/wasm-examples)
#
# Needs rustup's wasm32-unknown-unknown target and wasm-pack. The site-wasm GitHub Actions
# workflow runs this and publishes the tree to the site-wasm branch; the Vercel build only copies
# it into dist/ (scripts/fetch-wasm-examples.sh), so no Rust toolchain runs on Vercel.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
examples="$root/rust/kansei-wasm/examples"
out="${1:-$root/build/wasm-examples}"
mkdir -p "$out"
out="$(cd "$out" && pwd)"

# Not published (space-separated example names).
#   joydivision: plays a commercial recording the viewer supplies; the page stays local-only.
#   motion-matching: animates a character from private animation packs (.kmm) that never ship;
#     its world without the character is the lake example, which is published.
skip=" joydivision motion-matching "

# One target dir for every example, so the engine and wgpu compile once.
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$root/rust/target/wasm-examples}"

# Examples with their own .cargo/config.toml (rustflags such as +simd128) last, so a shared
# dependency is not rebuilt back and forth between flag sets.
names=()
for dir in "$examples"/*/; do
    name="$(basename "$dir")"
    [[ -f "$dir/Cargo.toml" && -f "$dir/www/index.html" ]] || continue
    [[ "$skip" == *" $name "* ]] && { echo "skip $name"; continue; }
    [[ -d "$dir/.cargo" ]] || names+=("$name")
done
for dir in "$examples"/*/; do
    name="$(basename "$dir")"
    [[ -d "$dir/.cargo" && -f "$dir/www/index.html" && "$skip" != *" $name "* ]] && names+=("$name")
done

for name in "${names[@]}"; do
    src="$examples/$name"
    dest="$out/$name"
    echo "=== $name"
    start=$SECONDS
    rm -rf "$dest"
    mkdir -p "$dest"
    # the page and its files; never a pkg/ committed or left in www/, nor linked private packs
    rsync -a --exclude 'pkg/' --exclude 'pack/' --exclude '*.kmm' --exclude 'make_assets.py' "$src/www/" "$dest/"
    (cd "$src" && wasm-pack --log-level warn build --target web --release --no-pack --out-dir "$dest/pkg")
    rm -f "$dest/pkg/.gitignore" "$dest/pkg"/*.d.ts
    # served as <name>/index.html with pkg/ beside it: pages written for www/ import ../pkg/.
    # The base keeps every relative URL (pkg/, assets/, fetches from the wasm) under the
    # example's folder even when the page is opened without its trailing slash.
    perl -0pi -e "s{(['\"])\\.\\./pkg/}{\$1./pkg/}g; s{(<head[^>]*>)}{\$1\\n    <base href=\"/examples/$name/\">}i or die 'no <head>'" "$dest/index.html"
    echo "    $(du -sh "$dest" | cut -f1) in $((SECONDS - start)) s"
done

if find "$out" -name '*.kmm' | grep -q .; then
    echo "error: a motion pack (.kmm) is in $out; packs never ship" >&2
    exit 1
fi
# Music is someone else's recording: the site never hosts audio.
audio="$(find "$out" -type f \( -iname '*.mp3' -o -iname '*.ogg' -o -iname '*.oga' -o -iname '*.opus' -o -iname '*.wav' -o -iname '*.m4a' -o -iname '*.aac' -o -iname '*.flac' \))"
if [[ -n "$audio" ]]; then
    echo "error: audio files are in $out; the site never ships recordings:" >&2
    echo "$audio" >&2
    exit 1
fi

{
    echo '{'
    echo "  \"commit\": \"$(git -C "$root" rev-parse HEAD 2>/dev/null || echo unknown)\","
    echo "  \"rust_tree\": \"$(git -C "$root" rev-parse HEAD:rust 2>/dev/null || echo unknown)\","
    printf '  "examples": ['
    sep=''
    for name in "${names[@]}"; do printf '%s"%s"' "$sep" "$name"; sep=', '; done
    echo ']'
    echo '}'
} > "$out/manifest.json"
echo "built ${#names[@]} examples into $out ($(du -sh "$out" | cut -f1))"
