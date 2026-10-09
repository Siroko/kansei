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

`www/room.html` is a third, the same character in a room lit by ray-traced global illumination,
with mirrors, a glass dragon in a pond, furniture to vault and climb, fog and dust (see
[The room](#the-room)). The character and its controls are shared by the pages (`src/character.rs`,
`src/player.rs`); the room is `src/room/`.

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

## The room

`www/room.html` (`start_room`) puts the character in a 40 x 40 m room, 9 m high (`src/room/`). It
loads the same packs as `www/index.html` (`www/pack/`, `pack=`, `hero=`). Without one a capsule
stands in for the character (drawn by the same skinned material, and in the mirrors) under the
HUD's "no motion pack" message, and the room renders as it does with one.

- **The light.** A 7 x 7 m panel in the ceiling lights the room from above. Kansei's Rust
  renderer doesn't rasterize area lights (`Light::Area` reaches only the ray tracer, as a point),
  so the panel is an emissive plane with a spot light under it: straight down, its cone covering
  the floor, and a PCSS emitter 0.9 m across, so its shadows are contact-hardening as an area
  light's are (`layout.rs`). Three floor lamps are warm, shadowed downlights under glowing shades.
  The four spot lights light everything through `SPOT_LIGHTS_WGSL`: the surfaces, the GI's hits,
  the reflections' hits, the fog and the dust.
- **One-sided walls.** The walls, the floor and the ceiling are planes facing in, their back
  faces culled, so from outside (O, or `cam=outside`) the camera looks through them into the
  room. The grid of triangles the rays trace hits a triangle from either side, so the GI's rays
  and the mirrors' from inside meet the walls as they should. They cast no shadows (the lights
  are inside), and thick boxes outside them keep the character in.
- **The GI.** The hybrid by default (`RtDiffuseGiEffect`: a ray for each 2 x 2 pixels through a
  grid of the room's triangles, 30 cm cells, the hits lit by the spot lights with shadow rays and a
  voxel cone, SVGF 3 x 3), over voxel GI's volume of the room. G (or `gi=`) switches to voxel
  cones, screen space or none.
- **Mirrors and glass.** A mirror on the north wall and a brushed one (roughness 0.1) on the east
  wall, and a glass Stanford dragon (index of refraction 1.5, in the grid by its cluster LOD's
  cut, with its vertex normals), through `RtReflectionsEffect` and its glass pass. Glass casts no
  shadow and the GI's rays pass through it. In a mirror, glass shows only where the camera sees the
  same point (the effect steps through glass it can't take from the screen).
- **The character in the mirrors.** Its skinned mesh can't go into the grid (the grid takes
  geometry as built, not as skinned), so capsules on its bones stand in for it there
  (`look::Proxy`): renderables no camera draws (`HIDDEN_WGSL`), moved when a bone has moved
  3 cm. Walking, they rebuild the grid every frame (about 0.8 ms); standing, never. `rt_body=0`
  leaves them out.
- **The pond.** In the middle, the lake's SPH water (`kansei_wasm_lake::lake`'s scale and tuning)
  round a stone plinth, the dragon's island. The character wades in as it does in the lake, and the
  water rests (culled out of view, asleep when settled). The plinth is a still capsule collider,
  not part of the container's floor: as a floor that steep it kept the water churning.
- **Furniture.** Sofas, a coffee table, a dining table and chairs, shelves, a sideboard, an island,
  crates, blocks, a two-block stack, a 2.2 m platform behind a step, two ledges with a gap, a bench.
  The solid ones are boxes in the collision world, so Space vaults, hurdles, mantles and climbs
  them as on the course. `room_check()` on the page probes each piece as the traversal does and
  says whether it gets the traversal meant (vault: the crate, the sideboard, the island, the
  dining table; hurdle: the rail, the bench, a sofa's back; mantle: the block, the stack, its top,
  the ledges; climb: the platform from its step).
- **Fog and dust.** A faint haze fills the room (a box `LocalFogVolume`, none outside), lit by the
  four lights through their shadow maps; F turns it off. 32768 dust motes drift in a slow curl
  flow round the camera (a compute shader), pushed aside by the character's legs and body, drawn as
  soft billboards a pixel or two across and lit by the same lights and shadow maps (scattering
  forward), so they catch the light in the beams and vanish in shadow; the fog veils them as it
  does the rest. N turns them off.

The dragon is the GI box's (`www/assets/` links to `examples/gi-box/www/assets/`): "Stanford
Dragon (Vrip)" by 3D graphics 101, CC-BY-NC-4.0, credited on the page (`license.txt`).

The character's body is drawn by `look::RoomLit` into the GBuffer (its normal and albedo for the
GI and the reflections), lit by the spot lights; the overlay's markers are dimmed to the room's
exposure.

Keys, on top of the shared ones: O inside / outside, G the GI, F the fog, N the dust.

URL parameters, on top of the character's (`pack`, `hero`, `char`, `gait`, `walk`, `run`, `at`,
`drive`, `play`, `view`, `taa`):
- `gi=rt|voxel|ssgi|off` (default `rt`);
- `cam=outside|dragon|mirror|parkour|living` starts at a fixed view (O, or any other name, follows
  the character);
- `fog=<extinction per metre>` (default 0.004), `fog=0` none;
- `dust=<count>` (default 32768, 0 none), `dust_size=` (m), `dust_bright=`, `dust_opacity=`,
  `dust_speed=` (m/s);
- `pond=0`, `rest=0` (the water never rests), `dragon=0`, `rt_body=0`;
- `ev=<EV100>` (exposure, default 4);
- `stats=1`: each pass's GPU time and the CPU sections on the page, and in `room.info()`;
  `profile=1`: the profile in the console every 3 s.

On `window.room`: `info()`, `set(key, value)` (`gi`, `fog`, `dust`, `outside`, `cam`) and
`check()`.

### What a frame costs

1080p (`dpr=1`), M4 Pro, headless Chrome without vsync (`stats=1`), the GASP pack's character;
other sessions share the GPU, so a pass's time varies by tens of percent between runs:

| | GPU, passes | GPU, span | frame interval |
|---|---|---|---|
| hybrid GI, fog, dust, the water asleep, standing | 11.7-12.3 ms | 12.8-13.2 ms | 15 ms |
| the same, walking (the capsules rebuild the grid) | 12.3 ms | 13.5 ms | 15 ms |
| the water running (`rest=0`) | 17.8 ms | 18.2 ms | 19 ms |
| voxel GI / screen-space GI / no GI | 8.0 / 6.9 / 6.4 ms | 9.4 / 8.2 / 7.8 ms | 13 ms |

- The hybrid's trace dominates: `RtGi/Trace` 2.6-4.7 ms (more while the grid is rebuilt the same
  frame), its SVGF and composite about 1.4 ms more. Then the glass pass (`Rt/Glass`, 1.1-1.4 ms,
  the dragon), the grid's rebuild while the character moves (`Rt/Gather`, 0.75-1.3 ms), TAA
  (0.6 ms), the mirrors' rays (`Rt/Trace` + `Rt/Resolve`, 0.6 ms).
- The water, while it runs: `FluidSim/Substep` 3.8 ms and the surface's extraction and mesh about
  2.5 ms more; it goes back to sleep once it settles after the character leaves it.
- The fog costs about 0.3 ms, the dust 0.01 ms of compute and a share of the GBuffer pass.
- On the CPU: the character's update (search, pose, IK, collision) 2.9-3.4 ms, the renderer
  about 1 ms. The frame interval stays near 13 ms even with no GI (6.4 ms of GPU) in this
  setup: there is headroom on the GPU.

## Build and run

```sh
cd rust/kansei-wasm/demos/motion-matching
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/ (or www/kimodo.html, www/room.html)
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
- `circle=<radius>` runs round a circle of that radius (metres; negative turns right) at the run
  pace instead, as the foot-slide course does (see [Foot slide](#foot-slide)).
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

## Foot slide

Foot locking (L) pins a planted foot where it touched down and bends the leg to it with two-joint
IK: on its ankle while the heel is down, on its ball once the heel lifts (a running foot lands
on its ball and rolls off it). A foot counts as planted while the pack's contacts say so: its
ankle or its ball low and under 1 m/s in the clip (`ContactThresholds`; packs baked before that
rule get their contacts found again on load). The pin lets go when the contact ends or the
animated foot strays 0.3 m from it, and pins again where the animation has the foot while the
contact lasts.

To measure it, headless, on a pack:

```sh
cd rust
cargo run -p kansei-core --release --example foot_slide -- <pack.kmm> [hero=<character.kmm>]
```

It plays a scripted course (straight walk and run, run circles of 2, 3.5 and 5 m and walk circles
of 1.5 and 3 m both ways, starts and stops, 180° turns; `animation::motion_matching::foot_slide`)
and prints each scenario's planted-foot slide (cm per second planted, cm per plant) and how far
the character faces from the simulation. `circle=<radius>` on this page runs the same circles,
to watch them. The tightest circles and 180° turns still slide where the database has no clip
for the turn: the character follows the simulation through clips that turn less.

## Timing against the TS port

The TS engine has a pure TypeScript port of this runtime (`src/animation/motion_matching`,
`src/collision`) and a page like this one on it, `examples/index_motion_matching.html` (packs in
the gitignored `examples/pack/`, or `?pack=<url>`). Both pages time the same things the same way
(`src/timing.rs` here), for the same pack:

- the HUD's `update` line: the character's update per frame (search, pose, inertialization,
  foot locking, collision), mean, 95th percentile and largest over the last 3000 frames;
  `window.motionTimings()` returns them as JSON (the TS page also splits off the searches);
- `window.benchSearch(step)`: milliseconds per `Database::search` over every `step`th frame's
  features, nudged, as queries;
- `window.benchPose(count)`: milliseconds to sample a pose between two frames, run its forward
  kinematics and fill the first body's bone palette, `count` times.

Run both with `drive=1` (the same route, `driveRestart()` starts it over) and `lake=0` here (the TS
page has the course and no lake), alternating the two pages: other sessions share the machine.
Serve them cross-origin isolated (COOP `same-origin`, COEP `credentialless`), or Chrome rounds
`performance.now()` to 0.1 ms.

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
