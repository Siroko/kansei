"""Generate a prompt set's clips with NVIDIA Kimodo, in one process (the model loads once).

    python kimodo_batch.py prompts/dance_card.json out/dance_card [--only 'Walk/*'] [--samples 2]

Runs where Kimodo is installed (the Docker image in `pc/`). For each clip of the set it writes
`<out>/<clip name>_<sample>.npz` (Kimodo's NPZ: SOMA 77-joint rotations, joint positions, root
path, heading and foot contacts, 30 fps) and `<out>/<clip name>.json`, its provenance: the
prompt, the constraints, the seed, the model and its revision, the Kimodo commit and the time it
took. `<out>/provenance.json` sums up the run. Clips already generated are skipped, so a run can
be resumed.

Never pass Epic GASP data (poses, clips, renders, the mannequin) into this: see README.md.
"""

import argparse
import fnmatch
import json
import os
import sys
import time
import zlib
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paths  # noqa: E402

from kimodo import load_model  # noqa: E402
from kimodo.constraints import load_constraints_lst  # noqa: E402
from kimodo.exports.motion_io import save_kimodo_npz  # noqa: E402
from kimodo.model.registry import get_model_info  # noqa: E402


def model_revision(repo_id):
    """The Hugging Face commit the model was loaded from (its snapshot folder's name)."""
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo_id, local_files_only=True)).name
    except Exception as error:  # pragma: no cover - only informative
        return f"unknown ({type(error).__name__})"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("set", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--only", default="*", help="generate only the clips whose name matches")
    parser.add_argument("--samples", type=int, default=None, help="samples per clip (default: the set's)")
    args = parser.parse_args()

    spec = json.loads(args.set.read_text())
    defaults = spec.get("defaults", {})
    model_name = spec.get("model", "kimodo-soma-rp-v1.1")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    started = time.time()
    model, resolved = load_model(model_name, device=device, return_resolved_name=True)
    info = get_model_info(resolved)
    loaded = time.time() - started
    provenance = {
        "generator": "NVIDIA Kimodo",
        "model": info.display_name if info else resolved,
        "model_repo": info.repo_id if info else None,
        "model_revision": model_revision(info.repo_id) if info else None,
        "model_license": "NVIDIA Open Model License (https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)",
        "kimodo_commit": os.environ.get("KIMODO_COMMIT", "unknown"),
        "text_encoder": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised (Meta Llama 3 Community License)",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "set": spec.get("set", args.set.stem),
        "set_file": args.set.name,
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_load_seconds": round(loaded, 1),
        "clips": [],
    }
    print(f"loaded {provenance['model']} ({provenance['model_revision']}) in {loaded:.0f} s on {provenance['gpu']}")

    args.out.mkdir(parents=True, exist_ok=True)
    for clip in spec["clips"]:
        name = clip["name"]
        if not fnmatch.fnmatchcase(name, args.only):
            continue
        record_path = args.out / f"{name}.json"
        if record_path.exists():
            provenance["clips"].append(json.loads(record_path.read_text()))
            print(f"{name}: done before, skipped")
            continue
        record_path.parent.mkdir(parents=True, exist_ok=True)

        prompts = clip["prompt"] if isinstance(clip["prompt"], list) else [clip["prompt"]]
        legs = clip.get("legs")
        total = clip.get("duration") or (paths.duration(legs) if legs else 5.0)
        durations = clip.get("durations", [total / len(prompts)] * len(prompts))
        num_frames = [int(round(d * model.fps)) for d in durations]
        frames = sum(num_frames)
        constraints = []
        if legs:
            constraints.append(paths.root2d_constraint(legs, frames, every=clip.get("every", defaults.get("every", 1)), heading=clip.get("heading", True), fps=model.fps))
        samples = args.samples or clip.get("samples", defaults.get("samples", 1))
        seed = clip.get("seed", defaults.get("seed", 0) + zlib.crc32(name.encode()) % 100000)
        cfg = clip.get("cfg", defaults.get("cfg", [2.0, 2.0]))
        steps = clip.get("diffusion_steps", defaults.get("diffusion_steps", 100))

        torch.manual_seed(seed)
        t0 = time.time()
        output = model(
            prompts,
            num_frames,
            constraint_lst=load_constraints_lst(constraints, model.skeleton) if constraints else [],
            num_denoising_steps=steps,
            num_samples=samples,
            multi_prompt=True,
            num_transition_frames=clip.get("transition_frames", 5),
            post_processing=clip.get("postprocess", True),
            return_numpy=True,
            cfg_type="separated",
            cfg_weight=cfg,
        )
        seconds = time.time() - t0
        files = []
        for i in range(samples):
            single = {k: (v[i] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == samples else v) for k, v in output.items()}
            path = args.out / f"{name}_{i:02d}.npz"
            save_kimodo_npz(str(path), single)
            files.append(path.name)
        record = {
            "name": name,
            "prompts": prompts,
            "durations": durations,
            "frames": frames,
            "loop": clip.get("loop", False),
            "legs": legs,
            "constraints": constraints,
            "samples": samples,
            "seed": seed,
            "cfg": cfg,
            "diffusion_steps": steps,
            "postprocess": clip.get("postprocess", True),
            "files": files,
            "seconds": round(seconds, 2),
            "model": provenance["model"],
            "model_revision": provenance["model_revision"],
            "kimodo_commit": provenance["kimodo_commit"],
        }
        record_path.write_text(json.dumps(record, indent=1))
        provenance["clips"].append(record)
        print(f"{name}: {samples} x {frames} frames in {seconds:.1f} s")

    provenance["finished"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (args.out / "provenance.json").write_text(json.dumps(provenance, indent=1))
    (args.out / args.set.name).write_text(args.set.read_text())


if __name__ == "__main__":
    main()
