#!/bin/sh
# Mixamo FBX downloads to clips on another rig, ready to bake (see README.md).
#   convert.sh <fbx folder> <rig mesh.glb> <out folder>
# The folder must hold YBot_TPose.fbx (the With Skin T-pose: the rest pose) beside the clips.
set -e
here=$(cd "$(dirname "$0")" && pwd)
fbx=$1; rig=$2; out=$3
[ -f "$fbx/YBot_TPose.fbx" ] || { echo "no $fbx/YBot_TPose.fbx (the With Skin T-pose download)"; exit 1; }
blender=${BLENDER:-/Applications/Blender.app/Contents/MacOS/Blender}
glb="$out/glb"
"$blender" -b --factory-startup -P "$here/fbx_to_glb.py" -- "$fbx" "$glb" | grep "kansei fbx_to_glb" || true
cd "$here"
uv run -q mixamo_to_gltf.py --target "$rig" --map maps/mixamo_to_ue5_mannequin.json --rest "$glb/YBot_TPose.glb" "$out/export" "$glb"
