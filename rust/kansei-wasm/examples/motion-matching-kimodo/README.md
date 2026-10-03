# Motion matching on generated animation (Kimodo)

The [motion-matching example](../motion-matching/README.md) on packs of generated clips: NVIDIA
Kimodo, baked by [`genanim`](../../../kansei-anim-bake/genanim/README.md), to feel how they play
under the stick. It plays like the original (keyboard or gamepad, the course, the lake; see its
controls) and adds a panel (P hides it) to switch packs and to watch the custom clips.

The crate is a thin start over the motion-matching crate, used as a library: the scene, the
controller and the exports (`clip_names`, `play_clips`, `set_drive`) are that crate's. It differs
in one thing: it loads a character pack only when the URL names one (`hero=`), since the generated
packs carry their own SOMA body.

## The packs

They are private and stay out of git (`www/pack/` is gitignored). Link them in:

```sh
rust/kansei-wasm/examples/motion-matching-kimodo/link-packs.sh [data]   # default ~/Documents/dev/kansei-private-data
```

That links `<data>/genanim/pack` as `www/pack/gen` and `<data>/gasp/pack` as `www/pack/gasp`. The
panel marks the packs that aren't linked, and opening one shows how to link it.

| `?pack=` | pack | body | paces (walk, run m/s) |
|---|---|---|---|
| `dance` (the default) | `gen/gen-dance.kmm`: the dance card | SOMA | 2, 5 |
| `all` | `gen/gen-all.kmm`: the dance card and the custom clips | SOMA | 2, 5 |
| `limp` | `gen/gen-limp.kmm`: limping on the left leg | SOMA | 1.2, 3 |
| `gasp-gen` | `gen/gasp-plus-gen.kmm`: GASP with the generated clips retargeted | mannequin (C: hero) | 2, 5 |
| `gasp` | `gasp/gasp-locomotion.kmm`: GASP only, for reference | hero (C: mannequin) | 2, 5 |

A pack id is written out into the pack's URL and its parameters (`walk=`, `run=`, `hero=`,
`char=`); switching packs reloads the page with them and keeps the rest of the URL (`course=0`,
`lake=0`, `view=`, `at=`…). `pack=<url>` loads any pack, as in the original.

GASP packs (and their renders) are under GASP's licence: keep screenshots and recordings of them
private.

## The clip browser

Under **clips**, the custom clips (push, drag, car and parkour, in `all` and `gasp-gen`) by kind:
pick a clip and a sample (or all of its samples) and play it, or every clip of the kind. They play
as they are, root motion included, one after another from where the character stands, with the
clip's name on the HUD (`play_clips`, as `play=<pattern>` does at start). A `play=` field takes any
pattern (`*` any run, several separated by commas). **drive the route** drives `drive=1`'s route
from there; **back to the stick** (or Escape) gives the character back to the player.

## Build and run

```sh
cd rust/kansei-wasm/examples/motion-matching-kimodo
wasm-pack build --target web --release
./link-packs.sh
python3 -m http.server 8080   # then open http://localhost:8080/www/ (or www/?pack=limp)
```
