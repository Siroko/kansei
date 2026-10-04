#!/bin/sh
# Link the private motion-matching packs for www/kimodo.html into www/pack/ (gitignored; the packs
# stay out of git):
#   www/pack/gen  -> <data>/genanim/pack  (gen-dance, gen-all, gen-limp, gasp-plus-gen)
#   www/pack/gasp -> <data>/gasp/pack     (gasp-locomotion, hero: the GASP reference)
# Usage: link-packs.sh [data]   (default: ~/Documents/dev/kansei-private-data)
set -e
data="${1:-$HOME/Documents/dev/kansei-private-data}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/www/pack"
for pair in gen:genanim/pack gasp:gasp/pack; do
    name="${pair%%:*}"
    dir="$data/${pair#*:}"
    if [ -d "$dir" ]; then
        ln -sfn "$dir" "$here/www/pack/$name"
        echo "www/pack/$name -> $dir"
    else
        echo "no $dir: www/pack/$name left out" >&2
    fi
done
