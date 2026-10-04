# Generated animation for motion matching

Text prompts and ground paths in, a motion-matching pack (`.kmm`) out, through NVIDIA's
[Kimodo](https://github.com/nv-tlabs/kimodo) (`Kimodo-SOMA-RP-v1.1`), a text- and
constraint-conditioned motion diffusion model, then [`kansei-anim-bake`](../README.md). Nothing is
trained here: the model is used as released.

```
prompts/*.json ──► kimodo_batch.py (Docker, NVIDIA GPU) ──► out/<set>/<clip>_<sample>.npz + .json
                                                              │
     kimodo_to_gltf.py (any machine, numpy) ◄──────────────────┘
       route A: SOMA skeleton + body   → export/{mesh.glb, clips/, manifest.json}
       route B: --target <rig> --map   → clips on that rig (bake with its mesh)
                                                              │
     kansei-anim-bake <config.json> ◄──────────────────────────┘ → pack.kmm
```

## Data and licences

- **No generated data lives in this repository**, like every other pack: the raw clips, the glTF
  exports and the packs stay in a private folder. Only the code, the prompt sets and the joint maps
  are here.
- **The model:** Kimodo-SOMA-RP-v1.1 is under the
  [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/):
  its card says the model is ready for commercial use, including animations for games and media,
  and NVIDIA claims no ownership of its outputs. Its text encoder needs Meta's gated Llama 3
  (`meta-llama/Meta-Llama-3-8B-Instruct`, Llama 3 Community License): a Hugging Face account that
  accepted it, its token in `HF_TOKEN`. Don't get around the model's guardrails: that ends the
  licence.
- **Provenance:** `kimodo_batch.py` writes, next to every clip, its prompt, path constraints, seed,
  the model and its Hugging Face revision, and the Kimodo commit; `provenance.json` sums up a run.
  Keep those files with the clips.
- **A pack made from them** carries a data notice in its `meta` (see `DATA-NOTICE.md`): AI-generated
  with NVIDIA Kimodo, the model's licence, the date. Whether pure AI output can be copyrighted is
  unsettled, so don't rely on this repository's MIT licence reaching the clips.
- **Never feed Epic GASP data into any generator.** The GASP listing is flagged NoAI: no GASP pose,
  clip, render or the mannequin rig may go into Kimodo, as a keyframe, an end-effector target, a
  reference or anything else. Generated clips meet GASP clips only *after* generation: retargeted
  by `retarget.py` (plain geometry) onto the mannequin skeleton, on your own machine, and baked
  into the same private pack (route B). The prompt sets here are text and paths only.

## 1. Generate (a Windows PC with an NVIDIA GPU, Docker Desktop on WSL2)

`pc/generate.cmd` keeps everything in one folder, `%USERPROFILE%\kansei-genanim` (`GENANIM` to
change it): `kimodo\` (a checkout pinned to the commit the scripts were written against),
`genanim\` (copy this folder there), `hf\` (the model weights and the 16 GB text encoder) and
`out\`.

```bat
set HF_TOKEN=...                       rem a token whose account accepted Llama 3
generate.cmd setup                     rem clone Kimodo, build the image (CUDA 12.8, for RTX 50xx)
generate.cmd dance_card                rem prompts\dance_card.json -> out\dance_card
generate.cmd custom --only "Parkour/*" --samples 4
```

The first run downloads about 17 GB and loads in minutes; after that, loading takes about a minute
and a clip a few seconds (an RTX 5090: 3 to 5 s for two 8 s samples). Runs resume: clips with a
`.json` are skipped. Delete the folder and the `kansei-genanim-kimodo` image to remove it all.

A prompt set (`prompts/*.json`) lists clips by name (`Walk/Walk_Loop_F`: the folder becomes the
bake's tag patterns), each with its prompt (or several, played in sequence), its samples and a
ground path as `legs` (`paths.py`): speed, direction of travel and facing, eased over each leg.
The path goes to Kimodo as a dense `root2d` constraint with heading, which is what makes turns,
pivots, starts, stops and strafes come out where motion matching needs them. Three sets:

- `dance_card`: idle, walk and run loops, circles, starts in four directions, stops, 90° turns,
  180° pivots, strafes and backwards, at the GASP walker's paces (walk 2.0, run 5.0 m/s);
- `custom`: pushing and dragging, a car (getting in, driving seated, getting out) and parkour
  (vaults, wall runs, rolls, climbs, flips);
- `limp`: one style, "a woman limping on her injured left leg", over a whole pack's locomotion
  and some parkour, at slower paces (walk 1.2, run 3.0 m/s: open the example with `walk=1.2&run=3`).

## 2. Convert

```sh
uv run kimodo_to_gltf.py --kimodo path/to/kimodo export/dance out/dance_card out/custom
```

- A ground joint `root` goes above SOMA's `Hips`: Kimodo's smoothed root path on the floor,
  turned to its heading. It carries the root motion and faces +Z, as the bake wants.
- Loops are cut where a pose comes back closest to an earlier one (the longest such cut), and the
  remaining difference is spread over the cycle so the last frame is the first. `report.json`
  lists each cut and how many degrees it bent.
- `--samples 00` keeps one sample per clip; `--trim START END` drops frames from non-loops;
  `--prefix Gen_` names clips `Walk/Gen_Walk_Loop_F_00`, to tell them apart in a mixed pack.

**Route A (SOMA):** `mesh.glb` is SOMA's skinned body from the Kimodo checkout
(`kimodo/assets/skeletons/somaskel77/skin_standard.npz`, Apache-2.0), on the same skeleton. Bake
with `joints: {"root": "root", "hips": "Hips", "left_foot": "LeftFoot", "right_foot":
"RightFoot", "left_hand": "LeftHand", "right_hand": "RightHand"}`.

**Route B (another rig):** `--target <that rig's mesh.glb> --map maps/soma_to_ue5_mannequin.json`
retargets each clip onto that skeleton (`retarget.py`: every source bone is first swung onto the
target's rest direction, so a T-pose source drives an A-pose target; then world rotations carry
over and the hips and root path scale by hip height). No mesh is written: list both exports in the
bake config, the rig's first:

```json
{ "export": ["gasp/export", "genanim/export-b"], "include": ["...", "*/Gen_*"],
  "loop": ["*_Loop_*", "*_Loop", "*/Gen_*_Circle_*"] }
```

`uv run tests.py` checks the path, quaternion, loop and retargeting conventions.

## 3. Bake and look

Bake as usual (`cargo run --release -p kansei-anim-bake -- config.json`) with `meta` `source`
and `license` filled in from `DATA-NOTICE.md`. In `rust/kansei-wasm/demos/motion-matching`:

- `?pack=<url>&drive=1` drives a fixed route (starts, turns, stops, a run, pivots, strafes) so two
  packs can be recorded with the same input; `drive_restart()` starts it over;
- `?pack=<url>&play=Parkour/` plays the clips whose names start with that, one after another, as
  generated (for clips the search would never pick).
