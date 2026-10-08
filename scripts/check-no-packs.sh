#!/usr/bin/env bash
# Fails the site build when a motion-matching pack (.kmm) or a linked packs folder made it into
# dist/: packs are baked from licensed animation and never ship (AGENTS.md). The TS motion-matching
# page (examples/index_motion_matching.html) loads its packs from examples/pack/, which
# copy-examples leaves out; scripts/build-wasm-examples.sh checks the Rust examples the same way.
set -euo pipefail

dist="$(cd "$(dirname "$0")/.." && pwd)/dist"
[[ -d "$dist" ]] || exit 0
found="$(find -L "$dist" -name '*.kmm' -print 2>/dev/null || true)"
if [[ -e "$dist/examples/pack" || -L "$dist/examples/pack" ]]; then
    found="$dist/examples/pack"$'\n'"$found"
fi
if [[ -n "$found" ]]; then
    echo "error: motion packs are in $dist; packs never ship:" >&2
    printf '%s\n' "$found" | sed -n 1,5p >&2
    exit 1
fi
