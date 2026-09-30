# Motion matching

A skinned character driven by motion matching (`kansei_core::animation::motion_matching`):
walking, running, starting, stopping, turning and strafing under keyboard or gamepad control, on
a ground plane with cascaded shadows and TAA, and over a course of boxes: hurdling, vaulting,
mantling and climbing on request, falling off edges and landing.

**No animation data ships with Kansei.** The page loads a motion-matching pack (`.kmm`) from
`www/pack/locomotion.kmm`, or from the URL in `?pack=<url>`. Without one it shows how to make
one, and renders the empty scene.

## A pack

Bake one from glTF clips and a skinned mesh with [`kansei-anim-bake`](../../../kansei-anim-bake/README.md).
It can export from an Unreal Engine project, headless.

Keep packs outside the repository and link them in. `*.kmm` is gitignored, and so is `www/pack/`:

```sh
ln -s /path/to/private/packs rust/kansei-wasm/examples/motion-matching/www/pack
```

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
shore). The water is the engine's SPH fluid, the one the fluid clock uses (about 128K particles,
5 cm apart, and less viscous than the clock's).
It is held by a `FluidContainer` whose walls follow the lake's irregular outline, a strip of shore
outside it, and whose floor is the lake bed. The character's legs push it through
`FluidColliders`.

- The character wades: the bed is in the collision world, shelving from the waterline to about
  0.6 m deep, so it walks down into the water and out again.
- Its thighs, shins, feet and hips are capsules. Walking pushes a wake and ripples ahead of the
  legs, running throws the water up, and water pushed onto the shore drains back.
- The simulation runs at 11 times the world's size and √11 times real time, so waves and splashes
  move at their real pace.
- The surface is marching cubes over a surface field (`DensityFieldOptions::particle_radius`):
  smooth over the bulk, with spray as small droplets. It refracts the bed and reflects the sky
  (`FluidSurfaceEffect`).
- Landing in the water (a jump or a fall) throws a crown of spray, scaled by how fast the
  character came down: a sphere at its feet pushes the water out for a moment
  (`FluidCapsule::expansion`).

### Tweaking the water

P shows a panel (Tweakpane), hidden at first:
- **water:** viscosity, the tensile correction (how much of the pull under the rest density
  acts), stiffness and near stiffness, rest density, substeps, time scale, the bed's friction,
  the legs' drag, the landing splash, and a reset;
- **surface:** presets (*Surface field, droplets*, the default; *Density iso, smooth*;
  *Performance*), and each setting: surface field or density, iso level, kernel radius, particle
  radius, grid resolution, interpolation.

## Build and run

```sh
cd rust/kansei-wasm/examples/motion-matching
wasm-pack build --target web --release
python3 -m http.server 8080   # then open http://localhost:8080/www/
```

The crate builds with WASM SIMD (`.cargo/config.toml`), since the search runs on the CPU.

## Controls

| | keyboard / mouse | gamepad |
|---|---|---|
| move (relative to the camera) | WASD, arrows | left stick (tilt sets the pace) |
| run | Shift | B, right trigger |
| jump, or traverse what is ahead | Space | A |
| strafe (face the camera's way) | Q | left bumper |
| orbit, zoom | drag, wheel | right stick |
| overlay (trajectory, feet, HUD) | B | |
| skeleton, mesh, foot locking | K, M, L | |
| character (with a character pack) | C | |

In the overlay:
- blue boxes are the simulated character now and at ⅓, ⅔ and 1 s ahead;
- the small boxes under the feet grow while a foot is planted and locked;
- the red bar and post mark the last ledge found, and its height.

URL parameters:
- `gait=0` searches every clip, instead of idle + walk or idle + run by tag;
- `taa=0` turns TAA off;
- `course=0` leaves the boxes out;
- `lake=0` leaves the lake out;
- `at=<x>,<z>,<degrees>` starts the character there, facing that way (0 is +Z).
- `profile=1` logs each labelled GPU pass's time (the fluid's included) to the console every
  3 s (`Renderer::set_profiling`).
- `lake_regions()` (a wasm export, from the console) counts the lake's particles in the lake, on the
  bank, against the walls and outside them, with their mean height: to check water drains back.
