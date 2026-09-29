# Motion matching

A skinned character driven by motion matching (`kansei_core::animation::motion_matching`):
walking, running, starting, stopping, turning and strafing under keyboard or gamepad control, on
a ground plane with cascaded shadows and TAA.

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

A pack carries the licence of the animation it was baked from. The HUD shows the pack's `source`
and `license` notes. Don't put a pack built from licensed third-party data at a public URL unless
that licence allows it.

The clips should cover:
- idle;
- walk and run loops;
- starts, stops, turns and pivots.

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
| run | Shift | A, right trigger |
| strafe (face the camera's way) | Q | left bumper |
| orbit, zoom | drag, wheel | right stick |
| overlay (trajectory, feet, HUD) | B | |
| skeleton, mesh, foot locking | K, M, L | |
| character (with a character pack) | C | |

In the overlay:
- blue boxes are the simulated character now and at ⅓, ⅔ and 1 s ahead;
- the small boxes under the feet grow while a foot is planted and locked.

URL parameters:
- `gait=0` searches every clip, instead of idle + walk or idle + run by tag;
- `taa=0` turns TAA off.
