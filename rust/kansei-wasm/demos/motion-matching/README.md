# Motion matching

A skinned character driven by motion matching (`kansei_core::animation::motion_matching`):
walking, running, starting, stopping, turning and strafing under keyboard or gamepad control, on
a ground plane with cascaded shadows and TAA, and over a course of boxes: hurdling, vaulting,
mantling and climbing on request, falling off edges and landing, and wading into a lake.

**No animation data ships with Kansei.** The page loads a motion-matching pack (`.kmm`) from
`www/pack/locomotion.kmm`, or from the URL in `?pack=<url>`. Without one it shows how to make
one, and renders the empty scene. So the example is not on kansei.graphics: its world without
the character (the course, the lake, the cannon and the mill) is the [lake example](../lake/README.md),
which is. This crate depends on that one for the world (`kansei_wasm_lake::World`) and keeps the
character: the controller, the bodies and retargeting, the follow camera, the HUD, the packs and
the clip tools.

`www/kimodo.html` is a second page on the same module, for packs of generated clips (see
[Generated animation (Kimodo)](#generated-animation-kimodo)).

## A pack

Bake one from glTF clips and a skinned mesh with [`kansei-anim-bake`](../../../kansei-anim-bake/README.md).
It can export from an Unreal Engine project, headless.

Keep packs outside the repository and link them in. `*.kmm` is gitignored, and so is `www/pack/`:

```sh
ln -s /path/to/private/packs rust/kansei-wasm/demos/motion-matching/www/pack
```

An app that stores its packs another way (encrypted, say) can depend on this crate and call
`start_with_loader(canvas_id, load)` instead of `start`: `load` gets each pack's URL and returns
its `.kmm` bytes, so it can fetch and decode them itself.

A pack carries the licence of the animation it was baked from, in its `source` and `license`
notes. Don't put a pack built from licensed third-party data at a public URL unless
that licence allows it.

The clips should cover:
- idle;
- walk and run loops;
- starts, stops, turns and pivots;
- for the course: traversals (hurdles, vaults, mantles, climbs), a fall loop and landings, named
  as `actions` in the bake config. Without them the course only blocks the way.

Tag them `idle`, `walk` and `run` for the gait filter (see the bake config). The walk and run
paces default to 2 and 5 m/s; set `walk=` and `run=` in the URL to match your data.

## A second body

A character pack (`www/pack/hero.kmm`, or `?hero=<url>`) is optional. It holds a mesh rigged to the
same skeleton as the motion pack, with its own proportions and textures. `kansei-anim-bake` makes
one from a glTF mesh and its textures (`character` config).

- Loaded, it is shown by default. C switches between it and the motion pack's own mesh, and
  `?char=hero` or `?char=mannequin` picks one.
- A motion pack may then ship without a mesh of its own (no `MESH` section): the character pack's
  body is the only one.
- The pose is retargeted onto its skeleton (`animation::retarget`: rotations as they are,
  translations oriented and scaled to its bones), and foot locking works on its legs.
- Its colour, normal (tangent space, +Y up) and occlusion/roughness/metallic textures are
  decoded from the pack and mipmapped on load.

## The course

Boxes stand around the start, some turned so you can take them at an angle:
- low rails (0.5 to 1 m) to hurdle, and a thin wall;
- boxes (about 1 m high, 1 m deep) to vault;
- blocks (1.2 to 1.5 m) to mantle onto, and walls (2 and 2.4 m) to climb;
- long narrow beams, too narrow to stand along;
- two stacks of blocks, one on top of another: mantle onto the first, then onto the second;
- a 0.3 m step, walked up without a traversal;
- two 1.2 m platforms 1.5 m apart: a running jump crosses the gap, a walking one falls short.

Space (A) traverses what is ahead, along the way the character moves (else where it faces)
(`motion_matching::traversal`):
- the kind comes from the obstacle's shape: its height, its depth and whether there is room to
  stand on it or beyond it;
- the clip and its start frame come from the pace and the distance to the ledge, among the
  clips that leave room where they end;
- the clip's root motion is warped so its ledge lands on the real one, at the real height.

Pressed a little early, Space waits up to a second for the obstacle to come in reach. With
nothing to traverse ahead (nothing there, or something too high, too narrow or with no room on
it), Space jumps:
- a jump clip whose pace and pose fit plays up to its take-off;
- from there the flight is ballistic, with the run-up's momentum and the controller's gravity,
  kept out of the boxes and landing wherever it comes down, box tops included;
- the jump clip, then the fall loop, animate it in the air, and the landing is chosen by the
  height of the fall (light or heavy) and the pace (into a stand, a walk or a run).

Walking off a top falls the same way. The HUD's `state` line shows what the character is doing,
and why the last Space was refused ("too high", "no room to land"…).

## The lake

East of the course lies a small lake (`lake=0` leaves it out; `at=14,-1,90` starts on its
shore): the [lake example](../lake/README.md)'s SPH water, with its water cannon and water mill.
Everything about the water, the cannon, the mill and the P panel is in that README; here the
character is in it:

- The character wades: the bed is in the collision world, shelving from the waterline to about
  0.6 m deep, so it walks down into the water and out again.
- Its thighs, shins, feet and hips are capsules (`FluidColliders`). Walking pushes a wake and
  ripples ahead of the legs, running throws the water up, and water pushed onto the shore drains
  back.
- Landing in the water (a jump or a fall) throws a crown of spray, scaled by how fast the
  character came down: a sphere at its feet pushes the water out for a moment
  (`FluidCapsule::expansion`).
- Legs within 2 m of the lake wake its water from rest.
- **The cannon** (`at=14.4,-3.9,10` starts beside it) fires only with the character within 2.5 m
  of it: E, the gamepad's X, or a click on its prompt; R (Y) by it drains the lake.
- **The mill** (`at=18,2.6,110` looks along the shore at it) turns in the north shallows.

## Build and run

```sh
cd rust/kansei-wasm/demos/motion-matching
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/ (or www/kimodo.html)
```

The crate builds with WASM SIMD (`.cargo/config.toml`), since the search runs on the CPU.
`scripts/build-wasm-examples.sh` skips it: its packs never ship.

## Controls

| | keyboard / mouse | gamepad |
|---|---|---|
| move (relative to the camera) | WASD, arrows | left stick (tilt sets the pace) |
| run | Shift | B, right trigger |
| jump, or traverse what is ahead | Space | A |
| strafe (face the camera's way) | Q | left bumper |
| fire the water cannon (by it; hold to pour) | E, or click the prompt | X |
| drain the lake (by the cannon) | R | Y |
| orbit, zoom | drag, wheel | right stick |
| overlay (trajectory, feet, HUD) | B | |
| skeleton, mesh, foot locking | K, M, L | |
| character (with a character pack) | C | |

In the overlay:
- blue boxes are the simulated character now and at ⅓, ⅔ and 1 s ahead;
- the small boxes under the feet grow while a foot is planted and locked;
- the red bar and post mark the last ledge found, and its height.

URL parameters:
- `hero=<url>` loads that character pack, `hero=none` none (the default is `pack/hero.kmm`);
- `gait=0` searches every clip, instead of idle + walk or idle + run by tag;
- `taa=0` turns TAA off;
- `course=0` leaves the boxes out;
- `lake=0` leaves the lake out;
- `rest=0` never rests the lake's water (always stepped and drawn, as before resting);
- `mill=0` starts the water mill stopped;
- `debug=1` allows `lake_regions()` (see the lake's README);
- `at=<x>,<z>,<degrees>` starts the character there, facing that way (0 is +Z).
- `drive=1` drives a fixed route instead of the player (starts, walks, a turn, stops, a run, a
  turn and a stop at a run, pivots, strafes, walking backwards), round and round: two packs get the
  same input, to record them side by side. `drive_restart()` (on `window` as `driveRestart`) starts
  it over.
- `play=<pattern>` plays the pack's clips whose names start with the pattern (`*` any run of
  characters, e.g. `play=Parkour/*_00`; several patterns separated by commas) one after another, as they are, each from the start point:
  for clips the search never picks (generated ones, see
  [`genanim`](../../../kansei-anim-bake/genanim/README.md)). `drive_restart()` starts them over.
- A page can drive these at runtime: `clip_names()` lists the pack's clips, `play_clips(pattern)`
  plays them from where the character stands (`""` gives it back to the player), and
  `set_drive(on)` starts or stops the `drive=1` route there (`www/kimodo.html` builds its clip
  browser on them).
- `view=<degrees>` turns the camera round the character from behind it (90: its left side).
- `profile=1` logs each labelled GPU pass's time (the fluid's included) to the console every
  3 s (`Renderer::set_profiling`).
- The lake's exports (the P panel's, `cannon_prompt()`, `cannon_fire(down)`, `lake_fill()`,
  `lake_regions()`) come from the lake crate and are in this module too: see the
  [lake's README](../lake/README.md).

## Generated animation (Kimodo)

`www/kimodo.html` plays packs of generated clips: NVIDIA Kimodo, baked by
[`genanim`](../../../kansei-anim-bake/genanim/README.md), to feel how they play under the stick.
It plays like `www/index.html` (keyboard or gamepad, the course, the lake; see the controls) and
adds a panel (P hides it) to switch packs and to watch the custom clips. It differs in one thing:
it loads a character pack only when the URL names one (`hero=<url>`; else it adds `hero=none`),
since the generated packs carry their own SOMA body.

### The packs

They are private and stay out of git (`www/pack/` is gitignored). Link them in:

```sh
rust/kansei-wasm/demos/motion-matching/link-packs.sh [data]   # default ~/Documents/dev/kansei-private-data
```

That links `<data>/genanim/pack` as `www/pack/gen` and `<data>/gasp/pack` as `www/pack/gasp`
(inside `www/pack/`: if that is itself a link to your packs folder, the links land in that
folder). The panel marks the packs that aren't linked, and opening one shows how to link it.

| `?pack=` | pack | body | paces (walk, run m/s) |
|---|---|---|---|
| `dance` (the default) | `gen/gen-dance.kmm`: the dance card | SOMA | 2, 5 |
| `all` | `gen/gen-all.kmm`: the dance card and the custom clips | SOMA | 2, 5 |
| `limp` | `gen/gen-limp.kmm`: limping on the left leg | SOMA | 1.2, 3 |
| `gasp-gen` | `gen/gasp-plus-gen.kmm`: GASP with the generated clips retargeted | mannequin (C: hero) | 2, 5 |
| `gasp` | `gasp/gasp-locomotion.kmm`: GASP only, for reference | hero (C: mannequin) | 2, 5 |

A pack id is written out into the pack's URL and its parameters (`walk=`, `run=`, `hero=`,
`char=`); switching packs reloads the page with them and keeps the rest of the URL (`course=0`,
`lake=0`, `view=`, `at=`…). `pack=<url>` loads any pack, as `www/index.html` does.

GASP packs (and their renders) are under GASP's licence: keep screenshots and recordings of them
private.

### The clip browser

Under **clips**, the custom clips (push, drag, car and parkour, in `all` and `gasp-gen`) by kind:
pick a clip and a sample (or all of its samples) and play it, or every clip of the kind. They play
as they are, root motion included, one after another from where the character stands, with the
clip's name on the HUD (`play_clips`, as `play=<pattern>` does at start). A `play=` field takes any
pattern (`*` any run, several separated by commas). **drive the route** drives `drive=1`'s route
from there; **back to the stick** (or Escape) gives the character back to the player.
