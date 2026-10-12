#!/bin/sh
# Link the private data the pages load into gitignored places (none of it goes into git):
#   www/pack/gen  -> <data>/genanim/pack  (gen-dance, gen-all, gen-limp, gasp-plus-gen)
#   www/pack/gasp -> <data>/gasp/pack     (gasp-locomotion, hero: the GASP reference)
#   www/pack/locomotion.kmm     -> the best motion pack there is: <data>/gasp/pack/gasp-parkour-mixamo.kmm
#                                  (parkour, Mixamo hangs), else gasp-parkour.kmm, else gasp-locomotion.kmm
#   www/pack/hero.kmm           -> <data>/gasp/pack/hero.kmm                (the room's body)
#   www/pack/mannequin.kmm      -> <assets>/mannequin/export/mannequin.kmm  (the showroom's body)
#   www/assets/showroom/private -> <assets>/showroom/export/web             (the showroom's furniture)
# Usage: link-packs.sh [data] [assets]
#   (defaults: ~/Documents/dev/kansei-private-data, ~/Documents/dev/kansei-assets)
set -e
data="${1:-$HOME/Documents/dev/kansei-private-data}"
assets="${2:-$HOME/Documents/dev/kansei-assets}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$here/www/pack"
link() {
    if [ -e "$1" ]; then
        ln -sfn "$1" "$2"
        echo "${2#$here/} -> $1"
    else
        echo "no $1: ${2#$here/} left out" >&2
    fi
}
for pair in gen:genanim/pack gasp:gasp/pack; do
    link "$data/${pair#*:}" "$here/www/pack/${pair%%:*}"
done
# the packs the pages load with no parameters
for pack in gasp-parkour-mixamo gasp-parkour gasp-locomotion; do
    if [ -e "$data/gasp/pack/$pack.kmm" ]; then
        link "$data/gasp/pack/$pack.kmm" "$here/www/pack/locomotion.kmm"
        break
    fi
done
link "$data/gasp/pack/hero.kmm" "$here/www/pack/hero.kmm"
link "$assets/mannequin/export/mannequin.kmm" "$here/www/pack/mannequin.kmm"
link "$assets/showroom/export/web" "$here/www/assets/showroom/private"
