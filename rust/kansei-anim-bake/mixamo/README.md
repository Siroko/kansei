# Mixamo clips onto the mannequin

Mixamo (Adobe) animations, downloaded by hand as FBX, put on another skeleton (the UE5 mannequin
of the private GASP pack) and baked into a private motion-matching pack. Plain geometry: no
learned model. **Mixamo's terms forbid AI/ML use of its content, so these clips (and anything
derived from them) never go into a generator such as Kimodo (`../genanim`).** Their files and the
packs baked from them stay out of git and off the site, like GASP's.

## Download

Pick one character (Y Bot) for every file. Download the T-pose once (FBX Binary, **With Skin**)
as `YBot_TPose.fbx`; it gives the rest pose. Download each clip as FBX Binary, **Without Skin**,
30 fps, no keyframe reduction, In Place off. A Without Skin file has no bind pose, so the rest
pose it carries can be any pose. The animation itself is absolute, and the rest pose comes from
the T-pose file.

## Convert and bake

```sh
./convert.sh <fbx folder> <rig mesh.glb> <out>
```

1. `fbx_to_glb.py`, in headless Blender: each FBX to a glb (Blender's importer and glTF exporter
   do the axes and units; bone orientations as authored).
2. `mixamo_to_gltf.py` (numpy):
   - folds the armature node (its centimetre scale and axis turn) into the hips;
   - adds a ground `root` under the hips' path (floor height, smoothed 0.1 s, turned to the body's
     heading from the thighs), since Mixamo has no root joint;
   - resamples at 30 fps;
   - retargets with `../genanim/retarget.py` and `maps/mixamo_to_ue5_mannequin.json`.

   Clips are named `Mixamo/<file name>`; `Hang_Idle_*` and `Shimmy_*` are marked as loops
   (`--loop`). `report.json` lists each clip's travel and its hips' and hands' height range.
3. List `<out>/export` after the rig's export in a bake config's `export`, include `Mixamo/*`.

`uv run tests.py <rig mesh.glb>` builds a synthetic Mixamo-like FBX in Blender (`make_test_fbx.py`,
no Mixamo data) and checks the conventions: metres, Y up and facing +Z, the root on the floor under
the hips and turning with the body, and the T-pose driving the rig's rest pose.
