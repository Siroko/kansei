"""Export a skeletal mesh and animation sequences from an Unreal Engine project to glTF.

Runs inside the editor commandlet, headless (no editor window):

    UnrealEditor-Cmd <Project>.uproject -run=pythonscript \
        -script="<path>/export_gltf.py <config.json>" \
        -EnablePlugins=PythonScriptPlugin,EditorScriptingUtilities \
        -unattended -nullrhi -nosplash -nosound -nop4 -stdout

The config (JSON) names what to export and where:

    {
      "output": "/abs/path/export",
      "mesh": "/Game/Characters/Hero/Meshes/SK_Hero",
      "animation_root": "/Game/Characters/Hero/Animations",
      "include": ["Walk/*_Loop_*", "Run/*"],
      "exclude": ["*_Additive_*"],
      "skip_existing": true
    }

`include` and `exclude` are fnmatch patterns over each AnimSequence's path relative to
`animation_root`. Optional `"import"` brings in other meshes rigged to the same skeleton, as
FBX, before exporting them too (to `<name>.glb`):

    "import": [{"fbx": "/abs/path/hero.fbx", "name": "hero",
                "skeleton": "/Game/Characters/Hero/Meshes/SK_Hero_Skeleton",
                "folder": "/Game/KanseiImport"}]

They go through the FBX importer onto that skeleton asset (its normals as authored, no
animation, no materials), so their bones have the skeleton's axes, as its animations expect.
This does write the imported assets into the project: use a copy. Writes `mesh.glb` (the skinned mesh and skeleton), one `clips/<path>.glb` per
sequence (skeleton and animation, no mesh) and `manifest.json` listing them with their frame
count, length and loop flag, for `kansei-anim-bake`.

Root motion: the exporter samples each sequence as the engine plays it, which locks the root bone
when the sequence has "Force Root Lock". The script clears that flag on the loaded sequence (in
memory only; nothing is saved) so the root bone keeps its motion.

Only reads the project: no asset is modified on disk. Run it on a copy of the project if the
editor writes caches you do not want in the original (Saved/, DerivedDataCache/).
"""

import fnmatch
import json
import os
import sys
import time

import unreal


def load_config():
    path = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KANSEI_EXPORT_CONFIG")
    if not path:
        raise RuntimeError("usage: export_gltf.py <config.json> (or set KANSEI_EXPORT_CONFIG)")
    with open(path) as f:
        return json.load(f)


def export_options(preview_mesh):
    o = unreal.GLTFExportOptions()
    # no material baking (it needs a renderer) and nothing but the skeleton, skin and animation
    o.set_editor_property("bake_material_inputs", unreal.GLTFMaterialBakeMode.DISABLED)
    o.set_editor_property("export_proxy_materials", False)
    o.set_editor_property("export_morph_targets", False)
    o.set_editor_property("export_cameras", False)
    o.set_editor_property("export_lights", False)
    o.set_editor_property("export_level_sequences", False)
    o.set_editor_property("export_vertex_skin_weights", True)
    o.set_editor_property("export_animation_sequences", True)
    o.set_editor_property("export_preview_mesh", preview_mesh)
    return o


def messages(result):
    if result is None:
        return []
    out = []
    for kind in ("warnings", "errors"):
        try:
            out += ["%s: %s" % (kind, m) for m in result.get_editor_property(kind)]
        except Exception:
            pass
    return out


def import_fbx(item):
    """An FBX rigged to an existing skeleton, as a skeletal mesh of that skeleton."""
    skeleton = unreal.load_asset(item["skeleton"])
    if skeleton is None:
        raise RuntimeError("no skeleton at %s" % item["skeleton"])
    options = unreal.FbxImportUI()
    options.set_editor_property("import_mesh", True)
    options.set_editor_property("import_as_skeletal", True)
    options.set_editor_property("mesh_type_to_import", unreal.FBXImportType.FBXIT_SKELETAL_MESH)
    options.set_editor_property("skeleton", skeleton)
    options.set_editor_property("import_animations", False)
    options.set_editor_property("import_materials", False)
    options.set_editor_property("import_textures", False)
    options.set_editor_property("create_physics_asset", False)
    data = options.get_editor_property("skeletal_mesh_import_data")
    data.set_editor_property("import_morph_targets", False)
    data.set_editor_property("update_skeleton_reference_pose", False)
    data.set_editor_property("use_t0_as_ref_pose", False)
    data.set_editor_property("convert_scene", True)
    data.set_editor_property("import_meshes_in_bone_hierarchy", True)
    data.set_editor_property("normal_import_method", unreal.FBXNormalImportMethod.FBXNIM_IMPORT_NORMALS)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", item["fbx"])
    task.set_editor_property("destination_path", item.get("folder", "/Game/KanseiImport"))
    task.set_editor_property("destination_name", item["name"])
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)
    task.set_editor_property("options", options)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    path = "%s/%s" % (item.get("folder", "/Game/KanseiImport"), item["name"])
    asset = unreal.load_asset(path)
    if asset is None:
        raise RuntimeError("importing %s made no asset at %s" % (item["fbx"], path))
    return asset


def main():
    config = load_config()
    out_dir = config["output"]
    os.makedirs(os.path.join(out_dir, "clips"), exist_ok=True)
    t0 = time.time()

    manifest = {"source": unreal.SystemLibrary.get_engine_version(), "clips": []}

    mesh_path = config.get("mesh")
    if mesh_path:
        mesh = unreal.load_asset(mesh_path)
        if mesh is None:
            raise RuntimeError("no asset at %s" % mesh_path)
        target = os.path.join(out_dir, "mesh.glb")
        issues = messages(unreal.GLTFExporter.export_to_gltf(mesh, target, export_options(True), set()))
        manifest["mesh"] = {"file": "mesh.glb", "asset": mesh_path, "issues": issues}
        unreal.log("kansei export: mesh %s -> %s" % (mesh_path, target))

    for item in config.get("import", []):
        asset = import_fbx(item)
        target = os.path.join(out_dir, item["name"] + ".glb")
        issues = messages(unreal.GLTFExporter.export_to_gltf(asset, target, export_options(True), set()))
        manifest.setdefault("imported", []).append({"file": item["name"] + ".glb", "fbx": item["fbx"], "issues": issues})
        unreal.log("kansei export: imported %s -> %s" % (item["fbx"], target))

    root = config["animation_root"].rstrip("/")
    include = config.get("include", ["*"])
    exclude = config.get("exclude", [])
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    query = unreal.ARFilter(
        package_paths=[root],
        recursive_paths=True,
        class_paths=[unreal.TopLevelAssetPath("/Script/Engine", "AnimSequence")],
    )
    assets = sorted(str(a.package_name) for a in registry.get_assets(query))
    selected = []
    for asset in assets:
        relative = asset[len(root) + 1:]
        if any(fnmatch.fnmatch(relative, p) for p in include) and not any(fnmatch.fnmatch(relative, p) for p in exclude):
            selected.append((asset, relative))
    unreal.log("kansei export: %d of %d sequences selected" % (len(selected), len(assets)))

    options = export_options(False)
    for i, (asset, relative) in enumerate(selected):
        target = os.path.join(out_dir, "clips", relative + ".glb")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        anim = unreal.load_asset(asset)
        entry = {
            "file": os.path.relpath(target, out_dir),
            "asset": asset,
            "name": os.path.basename(relative),
            "frames": unreal.AnimationLibrary.get_num_frames(anim) + 1,
            "duration": anim.get_play_length(),
            "looping": bool(anim.get_editor_property("loop")),
            "root_motion": bool(anim.get_editor_property("enable_root_motion")),
        }
        if not (config.get("skip_existing") and os.path.exists(target)):
            # keep the root bone's motion in the export (in memory only; not saved)
            anim.set_editor_property("force_root_lock", False)
            entry["issues"] = messages(unreal.GLTFExporter.export_to_gltf(anim, target, options, set()))
        manifest["clips"].append(entry)
        if i % 25 == 0:
            unreal.log("kansei export: %d/%d %s" % (i + 1, len(selected), relative))

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    unreal.log("kansei export: done, %d clips in %.1f s" % (len(manifest["clips"]), time.time() - t0))


main()
