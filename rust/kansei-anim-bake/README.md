# kansei-anim-bake

Bakes animation clips and a skinned mesh (glTF) into a **motion-matching pack** (`.kmm`) for
`kansei_core::animation::motion_matching`: the clips' poses, the character root's motion, foot
contacts and the search features, with the mesh, in one binary file the runtime loads as is.

## Data stays out of this repository

The tool is code; what it reads and writes is **your data**. A pack is derived from the clips and
mesh it was baked from, and carries their licence.

- Never commit third-party animation, meshes, their glTF exports or `.kmm` packs to Kansei (a
  public MIT repository). `*.kmm` is gitignored; keep exports and packs in a private folder
  outside any public repo.
- Ship a pack only as the licence of its sources allows. Store licensed store assets (Fab,
  Marketplace, Mixamo…) only as packed binaries inside a product, and only where their terms
  allow that.
- Record where a pack comes from in its `meta` (`source`, `license`): the runtime can show it.

## Pipeline

1. **Export glTF from Unreal Engine, headless** (no editor window). `unreal/export_gltf.py` runs
   in the editor commandlet with UE's own glTF exporter:

   ```sh
   UnrealEditor-Cmd /path/Project.uproject -run=pythonscript \
     -script="$PWD/unreal/export_gltf.py /path/export-config.json" \
     -EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities \
     -unattended -nullrhi -nosplash -nosound -nop4 -stdout
   ```

   The config names the skeletal mesh, a root folder of AnimSequences and include/exclude
   patterns. The script writes `mesh.glb`, `clips/<path>.glb` (skeleton and animation only) and
   `manifest.json` (frames, length, loop flag). It keeps each sequence's root motion by clearing
   "Force Root Lock" on the loaded asset (in memory; nothing is saved). The editor still writes
   caches (`Saved/`, `DerivedDataCache/`), so run it on a copy of the project; on APFS,
   `cp -cR` makes one instantly without using disk. Any other glTF source works too: skip this
   step and point `export` at a folder of `.glb`/`.gltf` clips.

2. **Bake**:

   ```sh
   cargo run --release -p kansei-anim-bake -- /path/bake-config.json
   ```

   The config's fields are listed in `src/main.rs`. `include`/`exclude` pick clips by path,
   `loop` marks loops, `tags` sets bits the search can filter on (e.g. one per gait), and
   `joints` names the root (a top joint on the ground carrying root motion and facing +Z), hips
   and feet. The defaults are Unreal's names. Joints nothing is weighted to (IK targets, virtual
   bones) are left out. The tool prints the database's size and contact coverage, then reads
   the pack back to check it.

3. **Play it**: `rust/kansei-wasm/demos/motion-matching` loads a pack from a local path.

`export` may also list several folders, baked into one pack (the mesh from the first): say, an
Unreal export and clips generated onto the same skeleton.

## Generated clips

[`genanim/`](genanim/README.md) makes clips with a text- and path-conditioned motion model (NVIDIA
Kimodo) on an NVIDIA GPU, converts them to glTF on its own SOMA skeleton and body or retargets them
onto another rig, ready for this tool. Its README says what may and may not go into the model.

## Characters

A character pack (`CharacterPack`) is another body for a motion pack's animation: a mesh rigged to
the same skeleton (same joint names and axes) with its own proportions, and its textures. Make one
with a config that has a `character` section (see `src/main.rs`).

- Textures are resized and encoded as lossy WebP. A DirectX-style normal map is flipped into
  glTF's convention (`normal_directx`).
- To keep the bone axes the animations expect, export the mesh through the same path as the
  clips. `export_gltf.py`'s `import` step brings an FBX onto the skeleton asset with Unreal's FBX
  importer, then exports it with the rest.
- At runtime `animation::retarget` maps the database's poses onto it.

## Actions

Traversals, jumps, falls and landings are played on command, not found by the search. Name them
in the config's `actions`, by kind (`hurdle`, `vault`, `mantle`, `climb`, `jump`, `fall`,
`land`). They get only `ACTION_TAG`, so the search skips them, and the pack stores what
`motion_matching::traversal::ActionClip::analyze` reads from each:
- the obstacle's height, from the root joint, which rides the surface the character is on;
- its front ledge, from the hands planted on its top (`joints.left_hand`/`right_hand`);
- the frames where the character leaves the ground, reaches the ledge, is on the top, leaves it,
  lands, and can hand back to motion matching;
- for a jump, the take-off and the apex's height; for a landing, the impact and the height it
  fell from (the controller picks harder landings for longer falls by it).

It reads the animation alone, no engine metadata. The tool prints each clip's analysis. A clip
it finds nothing in is left out, with a warning.

## What is baked

At the clips' rate (30 fps by default), per frame:

- joint rotations as 16-bit quaternions;
- translations, stored once per joint when they never change, else 16 bits per axis over the
  joint's range;
- the character root (position and heading);
- foot contacts (a foot low and slow for three frames or more);
- 27 normalized features: foot positions and velocities, hips velocity, and the root's future
  positions and facings at ⅓, ⅔ and 1 s.

Loops continue cycle after cycle past their end. Other clips go on at their last velocity.
