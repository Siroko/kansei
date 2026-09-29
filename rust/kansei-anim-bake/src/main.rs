//! `kansei-anim-bake <config.json>`: bake glTF animation clips and a skinned mesh into a Kansei
//! motion-matching pack (`.kmm`, see `kansei_core::animation::motion_matching::pack`).
//!
//! The config (paths relative to it):
//!
//! ```json
//! {
//!   "export": "export",
//!   "mesh": "mesh.glb",
//!   "output": "pack/locomotion.kmm",
//!   "include": ["Walk/*_Loop_*", "Idle/*"],
//!   "exclude": [],
//!   "loop": ["*_Loop_*"],
//!   "tags": {"idle": ["Idle/*"], "walk": ["Walk/*"]},
//!   "joints": {"root": "root", "hips": "pelvis", "left_foot": "foot_l", "right_foot": "foot_r"},
//!   "sample_rate": 30,
//!   "keep_unskinned": false,
//!   "color": [0.55, 0.57, 0.6, 1.0],
//!   "meta": {"source": "where the clips come from", "license": "their licence"}
//! }
//! ```
//!
//! A config with a `character` instead bakes a `CharacterPack`: a mesh to show a motion pack's
//! animation on (rigged to the same skeleton, with its own proportions) and its textures, each
//! resized to at most `texture_size` and encoded as lossy WebP (`quality`, 0-100):
//!
//! ```json
//! {
//!   "character": {
//!     "mesh": "export/hero.glb",
//!     "textures": {"base_color": "hero_BaseColor.png", "normal": "hero_Normal.png", "orm": "hero_ORM.png"},
//!     "texture_size": 2048, "quality": 85, "normal_directx": true
//!   },
//!   "output": "pack/hero.kmm",
//!   "meta": {"source": "...", "license": "..."}
//! }
//! ```
//!
//! `normal_directx` flips the normal map's green channel into glTF's convention (+Y up). Joints
//! nothing is weighted to are left out, as for a motion pack.
//!
//! Clips come from `export/manifest.json` when there is one (written by `unreal/export_gltf.py`,
//! with each clip's loop flag), else from every `.glb`/`.gltf` under `export`. A clip is named by
//! its path under `export/clips` (or `export`) without the extension; `include`, `exclude`,
//! `loop` and `tags` are patterns over that name (`*` any run of characters, `?` one). `loop`
//! overrides the manifest's flag for the clips it matches. Each tag is a bit (in the order
//! given) the runtime search can filter on.
//!
//! Joints no vertex is weighted to, that the database doesn't read and that have no such joint
//! below them (IK targets, virtual bones, prop sockets) are left out, unless `keep_unskinned`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use kansei_core::animation::motion_matching::pack::{MotionPack, PackMesh};
use kansei_core::animation::motion_matching::{DatabaseBuilder, JointRoles, FEATURES};
use kansei_core::animation::{Skeleton, SkinnedGltf, SkinnedMesh};
use serde::Deserialize;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default = "here")]
    export: PathBuf,
    #[serde(default = "default_mesh")]
    mesh: PathBuf,
    output: PathBuf,
    #[serde(default = "everything")]
    include: Vec<String>,
    #[serde(default)]
    exclude: Vec<String>,
    #[serde(rename = "loop")]
    looping: Option<Vec<String>>,
    #[serde(default)]
    tags: BTreeMap<String, Vec<String>>,
    #[serde(default)]
    joints: Joints,
    #[serde(default = "default_rate")]
    sample_rate: f32,
    #[serde(default)]
    keep_unskinned: bool,
    #[serde(default = "default_color")]
    color: [f32; 4],
    #[serde(default)]
    meta: BTreeMap<String, String>,
    character: Option<CharacterConfig>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CharacterConfig {
    mesh: PathBuf,
    #[serde(default)]
    textures: BTreeMap<String, PathBuf>,
    #[serde(default = "default_texture_size")]
    texture_size: u32,
    #[serde(default = "default_quality")]
    quality: f32,
    #[serde(default)]
    normal_directx: bool,
    #[serde(default = "white")]
    color: [f32; 4],
}

fn here() -> PathBuf {
    ".".into()
}
fn default_texture_size() -> u32 {
    2048
}
fn default_quality() -> f32 {
    85.0
}
fn white() -> [f32; 4] {
    [1.0; 4]
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Joints {
    root: String,
    hips: String,
    left_foot: String,
    right_foot: String,
}

impl Default for Joints {
    /// Unreal's skeleton names.
    fn default() -> Self {
        Self { root: "root".into(), hips: "pelvis".into(), left_foot: "foot_l".into(), right_foot: "foot_r".into() }
    }
}

fn default_mesh() -> PathBuf {
    "mesh.glb".into()
}
fn everything() -> Vec<String> {
    vec!["*".into()]
}
fn default_rate() -> f32 {
    30.0
}
fn default_color() -> [f32; 4] {
    [0.55, 0.57, 0.6, 1.0]
}

#[derive(Deserialize)]
struct Manifest {
    clips: Vec<ManifestClip>,
}

#[derive(Deserialize)]
struct ManifestClip {
    file: PathBuf,
    looping: bool,
}

/// A clip to bake: its name (path without extension), file and loop flag.
struct Source {
    name: String,
    file: PathBuf,
    looping: bool,
}

/// The skeleton without the joints nothing needs: kept are the joints the meshes are weighted to,
/// `needed`, and every joint above them. The meshes' skins are remapped onto it.
fn prune(skeleton: &Skeleton, meshes: &mut [SkinnedMesh], needed: &[usize]) -> Skeleton {
    let mut keep = vec![false; skeleton.len()];
    let mark = |mut j: usize, keep: &mut Vec<bool>| loop {
        keep[j] = true;
        match skeleton.parents[j] {
            Some(p) => j = p,
            None => break,
        }
    };
    for &j in needed {
        mark(j, &mut keep);
    }
    for mesh in meshes.iter() {
        for (joints, weights) in mesh.joints.iter().zip(&mesh.weights) {
            for (j, w) in joints.iter().zip(weights) {
                if *w > 0.0 {
                    mark(mesh.skin_joints[*j as usize], &mut keep);
                }
            }
        }
    }
    let mut new_index = vec![None; skeleton.len()];
    let (mut names, mut parents, mut rest) = (Vec::new(), Vec::new(), Vec::new());
    for j in (0..skeleton.len()).filter(|&j| keep[j]) {
        new_index[j] = Some(names.len());
        names.push(skeleton.names[j].clone());
        parents.push(skeleton.parents[j].map(|p| new_index[p].expect("a kept joint's parent is kept")));
        rest.push(skeleton.rest[j]);
    }
    for mesh in meshes.iter_mut() {
        // skin joints of dropped joints carry no weight; point them at the root
        mesh.skin_joints = mesh.skin_joints.iter().map(|&j| new_index[j].unwrap_or(0)).collect();
    }
    Skeleton::new(names, parents, rest)
}

/// `*` matches any run of characters, `?` any one.
fn matches(pattern: &str, name: &str) -> bool {
    let (p, n): (Vec<char>, Vec<char>) = (pattern.chars().collect(), name.chars().collect());
    // dynamic programming over (pattern prefix, name prefix)
    let mut row = vec![false; n.len() + 1];
    row[0] = true;
    for &c in &p {
        let mut next = vec![false; n.len() + 1];
        next[0] = row[0] && c == '*';
        for j in 1..=n.len() {
            next[j] = match c {
                '*' => next[j - 1] || row[j],
                '?' => row[j - 1],
                c => row[j - 1] && n[j - 1] == c,
            };
        }
        row = next;
    }
    row[n.len()]
}

fn any_match(patterns: &[String], name: &str) -> bool {
    patterns.iter().any(|p| matches(p, name))
}

fn clip_name(path: &Path, under: &Path) -> String {
    let relative = path.strip_prefix(under).unwrap_or(path).with_extension("");
    relative.to_string_lossy().replace('\\', "/")
}

/// Every clip of the export: from its manifest, or every glTF file under it but the mesh.
fn sources(export: &Path, mesh: &Path) -> Result<Vec<Source>, String> {
    let manifest = export.join("manifest.json");
    if manifest.exists() {
        let text = std::fs::read_to_string(&manifest).map_err(|e| format!("{}: {e}", manifest.display()))?;
        let manifest: Manifest = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", manifest.display()))?;
        let clips_dir = export.join("clips");
        return Ok(manifest.clips.into_iter().map(|c| Source { name: clip_name(&export.join(&c.file), &clips_dir), file: export.join(c.file), looping: c.looping }).collect());
    }
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_dir() {
                walk(&path, out)?;
            } else if matches!(path.extension().and_then(|e| e.to_str()), Some("glb" | "gltf")) {
                out.push(path);
            }
        }
        Ok(())
    }
    let mut files = Vec::new();
    walk(export, &mut files).map_err(|e| format!("{}: {e}", export.display()))?;
    files.sort();
    Ok(files.into_iter().filter(|f| f != mesh).map(|f| Source { name: clip_name(&f, export), file: f, looping: false }).collect())
}

/// A texture resized to fit `size` and encoded as lossy WebP; a normal map's green flipped when
/// asked.
fn encode_texture(path: &Path, size: u32, quality: f32, flip_green: bool) -> Result<Vec<u8>, String> {
    let image = image::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let image = if image.width() > size || image.height() > size { image.resize(size, size, image::imageops::FilterType::Lanczos3) } else { image };
    let mut rgb = image.to_rgb8();
    if flip_green {
        for p in rgb.pixels_mut() {
            p[1] = 255 - p[1];
        }
    }
    let encoder = webp::Encoder::from_rgb(rgb.as_raw(), rgb.width(), rgb.height());
    Ok(encoder.encode(quality).to_vec())
}

fn bake_character(base: &Path, character: &CharacterConfig, output: &Path, meta: &BTreeMap<String, String>, keep_unskinned: bool) -> Result<(), String> {
    use kansei_core::animation::motion_matching::pack::{CharacterPack, PackImage};
    let mesh_path = base.join(&character.mesh);
    let mut model = SkinnedGltf::load(&mesh_path.to_string_lossy(), None)?;
    if model.meshes.is_empty() {
        return Err(format!("{} has no skinned mesh", mesh_path.display()));
    }
    let root = model.skeleton.parents.iter().position(|p| p.is_none()).unwrap_or(0);
    let skeleton = if keep_unskinned { model.skeleton.clone() } else { prune(&model.skeleton, &mut model.meshes, &[root]) };
    println!(
        "character: {} joints ({} left out), {} vertices, {} triangles",
        skeleton.len(),
        model.skeleton.len() - skeleton.len(),
        model.meshes.iter().map(|m| m.vertices.len()).sum::<usize>(),
        model.meshes.iter().map(|m| m.indices.len() / 3).sum::<usize>()
    );
    let mut images = Vec::new();
    for (name, path) in &character.textures {
        let bytes = encode_texture(&base.join(path), character.texture_size, character.quality, name == "normal" && character.normal_directx)?;
        println!("texture {name}: {:.1} MB", bytes.len() as f64 / 1e6);
        images.push(PackImage { name: name.clone(), mime: "image/webp".into(), bytes });
    }
    let meshes = model.meshes.into_iter().map(|mesh| PackMesh { mesh, color: character.color }).collect();
    let pack = CharacterPack { skeleton, meshes, images, meta: meta.clone().into_iter().collect() };
    let bytes = pack.to_bytes();
    if let Some(dir) = output.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(output, &bytes).map_err(|e| format!("{}: {e}", output.display()))?;
    CharacterPack::from_bytes(&bytes)?;
    println!("wrote {} ({:.1} MB)", output.display(), bytes.len() as f64 / 1e6);
    Ok(())
}

fn run(config_path: &Path) -> Result<(), String> {
    let text = std::fs::read_to_string(config_path).map_err(|e| format!("{}: {e}", config_path.display()))?;
    let config: Config = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", config_path.display()))?;
    let base = config_path.parent().unwrap_or(Path::new("."));
    if let Some(character) = &config.character {
        return bake_character(base, character, &base.join(&config.output), &config.meta, config.keep_unskinned);
    }
    let export = base.join(&config.export);
    let mesh_path = export.join(&config.mesh);
    let output = base.join(&config.output);

    let mut model = SkinnedGltf::load(&mesh_path.to_string_lossy(), Some(config.sample_rate))?;
    if model.meshes.is_empty() {
        return Err(format!("{} has no skinned mesh", mesh_path.display()));
    }
    let j = &config.joints;
    let full = JointRoles::find(&model.skeleton, &j.root, &j.hips, &j.left_foot, &j.right_foot)?;
    let skeleton = if config.keep_unskinned {
        model.skeleton.clone()
    } else {
        prune(&model.skeleton, &mut model.meshes, &[full.root, full.hips, full.feet[0], full.feet[1]])
    };
    let roles = JointRoles::find(&skeleton, &j.root, &j.hips, &j.left_foot, &j.right_foot)?;
    println!(
        "mesh: {} joints ({} left out), {} meshes, {} vertices",
        skeleton.len(),
        model.skeleton.len() - skeleton.len(),
        model.meshes.len(),
        model.meshes.iter().map(|m| m.vertices.len()).sum::<usize>()
    );

    let tag_names: Vec<&String> = config.tags.keys().collect();
    if tag_names.len() > 32 {
        return Err("at most 32 tags".into());
    }
    let mut builder = DatabaseBuilder::new(skeleton.clone(), roles, config.sample_rate);
    let mut baked = 0;
    for source in sources(&export, &mesh_path)? {
        if !any_match(&config.include, &source.name) || any_match(&config.exclude, &source.name) {
            continue;
        }
        let looping = config.looping.as_ref().map_or(source.looping, |p| any_match(p, &source.name));
        let mut tags = 0u32;
        for (bit, name) in tag_names.iter().enumerate() {
            if any_match(&config.tags[*name], &source.name) {
                tags |= 1 << bit;
            }
        }
        let file = SkinnedGltf::load(&source.file.to_string_lossy(), Some(config.sample_rate))?;
        for (k, clip) in file.clips.iter().enumerate() {
            let mut clip = clip.retarget_by_name(&file.skeleton, &skeleton);
            clip.name = if file.clips.len() == 1 { source.name.clone() } else { format!("{}#{k}", source.name) };
            if looping {
                // a loop's last frame should be its first again
                let (first, last) = (clip.transform(0, roles.hips), clip.transform(clip.frame_count() - 1, roles.hips));
                if first.rotation.dot(last.rotation).abs() < 0.999 {
                    eprintln!("warning: loop '{}' does not end on its first pose", clip.name);
                }
            }
            builder.add_clip(&clip, looping, tags)?;
            baked += 1;
        }
    }
    if baked == 0 {
        return Err("no clip matched `include`".into());
    }
    let database = builder.build();
    let frames = database.frame_count();
    let planted = (0..frames).map(|f| database.contacts(f)).fold([0, 0], |acc, c| [acc[0] + c[0] as usize, acc[1] + c[1] as usize]);
    println!(
        "database: {} clips ({} loops), {frames} frames ({:.1} min at {} fps), feet planted {:.0}% / {:.0}% of frames",
        database.clips.len(),
        database.clips.iter().filter(|c| c.looping).count(),
        frames as f32 / config.sample_rate / 60.0,
        config.sample_rate,
        100.0 * planted[0] as f32 / frames as f32,
        100.0 * planted[1] as f32 / frames as f32,
    );
    println!("feature spread (normalization / weight): {:?}", (0..FEATURES).step_by(3).map(|i| (database.feature_scale[i] * 1000.0).round() / 1000.0).collect::<Vec<_>>());
    for (bit, name) in tag_names.iter().enumerate() {
        println!("tag {bit} '{name}': {} clips", database.clips.iter().filter(|c| c.tags & (1 << bit) != 0).count());
    }

    let meshes = model.meshes.into_iter().map(|mesh| PackMesh { mesh, color: config.color }).collect();
    let mut meta: Vec<(String, String)> = config.meta.into_iter().collect();
    meta.push(("tags".into(), tag_names.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(",")));
    let pack = MotionPack { database, meshes, actions: Vec::new(), meta };
    let bytes = pack.to_bytes();
    if let Some(dir) = output.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    }
    std::fs::write(&output, &bytes).map_err(|e| format!("{}: {e}", output.display()))?;
    // read it back: the file must load as the runtime loads it
    MotionPack::from_bytes(&bytes)?;
    println!("wrote {} ({:.1} MB)", output.display(), bytes.len() as f64 / 1e6);
    Ok(())
}

fn main() {
    let Some(config) = std::env::args().nth(1) else {
        eprintln!("usage: kansei-anim-bake <config.json>");
        std::process::exit(2);
    };
    if let Err(e) = run(Path::new(&config)) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Quat, Vec3};
    use kansei_core::animation::Transform;
    use kansei_core::geometries::Vertex;

    #[test]
    fn pruning_keeps_weighted_and_needed_joints_and_their_ancestors() {
        // root > hips > (spine > head, ik_target), root > prop
        let t = Transform::from_translation_rotation(Vec3::Y, Quat::IDENTITY);
        let skeleton = Skeleton::new(
            ["root", "hips", "spine", "head", "ik_target", "prop"].map(String::from).to_vec(),
            vec![None, Some(0), Some(1), Some(2), Some(1), Some(0)],
            vec![t; 6],
        );
        let vertex = Vertex { position: [0.0, 0.0, 0.0, 1.0], normal: [0.0, 1.0, 0.0], uv: [0.0; 2] };
        // skin joints: prop, head, ik_target (weight 0)
        let mut meshes = vec![SkinnedMesh {
            name: "m".into(),
            vertices: vec![vertex; 2],
            indices: vec![0, 1, 0],
            joints: vec![[1, 2, 0, 0], [1, 0, 0, 0]],
            weights: vec![[0.5, 0.0, 0.5, 0.0], [1.0, 0.0, 0.0, 0.0]],
            skin_joints: vec![5, 3, 4],
            inverse_bind: vec![Mat4::IDENTITY; 3],
            material: None,
        }];
        let pruned = prune(&skeleton, &mut meshes, &[1]);
        assert_eq!(pruned.names, ["root", "hips", "spine", "head", "prop"]);
        assert_eq!(pruned.parents, vec![None, Some(0), Some(1), Some(2), Some(0)]);
        assert_eq!(meshes[0].skin_joints, vec![4, 3, 0]);
    }

    #[test]
    fn patterns_match_like_a_shell() {
        assert!(matches("Walk/*_Loop_*", "Walk/Hero_Walk_Loop_F"));
        assert!(!matches("Walk/*_Loop_*", "Run/Hero_Run_Loop_F"));
        assert!(matches("*", ""));
        assert!(matches("Idle/H?ro*", "Idle/Hero_Stand_Idle_Loop"));
        assert!(!matches("Idle", "Idle/x"));
        assert!(matches("*Loop*", "Loop"));
        assert!(!matches("a*b", "ac"));
    }
}
