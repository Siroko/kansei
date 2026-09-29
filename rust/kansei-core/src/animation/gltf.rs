//! glTF import of skeletons, skinned meshes and animations.

use std::collections::HashMap;

use glam::{Mat4, Quat, Vec3};

use super::{strongest_influences, Clip, Pose, Skeleton, SkinnedMesh, Transform};
use crate::geometries::Vertex;
use crate::loaders::GLTFMaterialInfo;

/// A glTF file's skeleton, skinned meshes and animations.
///
/// The skeleton holds every joint of every skin (or, in a file without skins, every node), parents
/// first. Non-joint nodes above a joint (an armature node, an axis conversion, a helper between
/// two joints) are folded into it, so the skeleton's model space is the file's world space, as
/// glTF skinning defines it. Animations are resampled at a fixed rate into `Clip`s over the skeleton's joints;
/// channels on other nodes are ignored.
pub struct SkinnedGltf {
    pub skeleton: Skeleton,
    pub meshes: Vec<SkinnedMesh>,
    pub clips: Vec<Clip>,
    pub materials: Vec<GLTFMaterialInfo>,
}

impl SkinnedGltf {
    /// Load a .gltf or .glb file from disk. `sample_rate` resamples the animations (frames per
    /// second); `None` uses the rate of their keys.
    pub fn load(path: &str, sample_rate: Option<f32>) -> Result<Self, String> {
        let (document, buffers, _) = gltf::import(path).map_err(|e| format!("failed to load glTF '{path}': {e}"))?;
        Self::from_document(&document, &buffers, sample_rate)
    }

    /// Load from in-memory .glb bytes (or .gltf with embedded buffers).
    pub fn from_slice(bytes: &[u8], sample_rate: Option<f32>) -> Result<Self, String> {
        let (document, buffers, _) = gltf::import_slice(bytes).map_err(|e| format!("failed to parse glTF: {e}"))?;
        Self::from_document(&document, &buffers, sample_rate)
    }

    pub fn from_document(document: &gltf::Document, buffers: &[gltf::buffer::Data], sample_rate: Option<f32>) -> Result<Self, String> {
        let (skeleton, joint_nodes) = skeleton(document);
        if skeleton.is_empty() {
            return Err("the glTF has no nodes to animate".into());
        }
        let mut meshes = Vec::new();
        let scene = document.default_scene().or_else(|| document.scenes().next());
        for node in scene.iter().flat_map(|s| s.nodes()).flat_map(descendants) {
            if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
                for (p, primitive) in mesh.primitives().enumerate() {
                    let name = format!("{}/{p}", mesh.name().unwrap_or("mesh"));
                    if let Some(m) = skinned_primitive(&name, &primitive, &skin, buffers, &joint_nodes.node_joint)? {
                        meshes.push(m);
                    }
                }
            } else if node.mesh().is_some() {
                log::warn!("glTF node '{}' has a mesh without a skin: not imported", node.name().unwrap_or("?"));
            }
        }
        let clips = document.animations().enumerate().map(|(i, a)| clip(&a, i, buffers, &joint_nodes, sample_rate)).collect::<Result<_, _>>()?;
        let materials = document
            .materials()
            .map(|m| {
                let pbr = m.pbr_metallic_roughness();
                GLTFMaterialInfo {
                    name: m.name().unwrap_or("Unnamed").to_string(),
                    base_color: pbr.base_color_factor(),
                    metallic: pbr.metallic_factor(),
                    roughness: pbr.roughness_factor(),
                    double_sided: m.double_sided(),
                }
            })
            .collect();
        Ok(Self { skeleton, meshes, clips, materials })
    }
}

/// A node and everything below it, depth first.
fn descendants(node: gltf::Node) -> Vec<gltf::Node> {
    let mut out = vec![node.clone()];
    for child in node.children() {
        out.extend(descendants(child));
    }
    out
}

fn node_transform(node: &gltf::Node) -> Transform {
    let (t, r, s) = node.transform().decomposed();
    Transform::new(Vec3::from(t), Quat::from_array(r).normalize(), Vec3::from(s))
}

/// How the glTF's nodes map onto the skeleton.
struct JointNodes {
    /// Joint index of each joint node.
    node_joint: HashMap<usize, usize>,
    /// Per joint, its node's own local transform (what animation channels replace)...
    own: Vec<Transform>,
    /// ...and the non-joint nodes between it and its parent joint (or the scene root), folded in
    /// front of it.
    above: Vec<Transform>,
}

/// The skeleton, and how the nodes map onto it.
fn skeleton(document: &gltf::Document) -> (Skeleton, JointNodes) {
    let mut is_joint = vec![document.skins().len() == 0; document.nodes().len()];
    for skin in document.skins() {
        for j in skin.joints() {
            is_joint[j.index()] = true;
        }
    }
    struct Out {
        names: Vec<String>,
        parents: Vec<Option<usize>>,
        joints: JointNodes,
    }
    // depth first from the scene roots: parents come before children
    fn visit(node: gltf::Node, parent_joint: Option<usize>, above: Transform, is_joint: &[bool], out: &mut Out) {
        let local = node_transform(&node);
        let (joint, above) = if is_joint[node.index()] {
            let j = out.names.len();
            out.names.push(node.name().map_or_else(|| format!("node_{}", node.index()), str::to_string));
            out.parents.push(parent_joint);
            out.joints.own.push(local);
            // a joint takes the transform of the non-joint nodes since its parent joint
            out.joints.above.push(above);
            out.joints.node_joint.insert(node.index(), j);
            (Some(j), Transform::IDENTITY)
        } else {
            (parent_joint, above.mul(&local))
        };
        for child in node.children() {
            visit(child, joint, above, is_joint, out);
        }
    }
    let mut out = Out { names: Vec::new(), parents: Vec::new(), joints: JointNodes { node_joint: HashMap::new(), own: Vec::new(), above: Vec::new() } };
    let roots: Vec<gltf::Node> = match document.default_scene().or_else(|| document.scenes().next()) {
        Some(scene) => scene.nodes().collect(),
        None => document.nodes().filter(|n| !document.nodes().any(|p| p.children().any(|c| c.index() == n.index()))).collect(),
    };
    for root in roots {
        visit(root, None, Transform::IDENTITY, &is_joint, &mut out);
    }
    let rest = out.joints.above.iter().zip(&out.joints.own).map(|(a, o)| a.mul(o)).collect();
    (Skeleton::new(out.names, out.parents, rest), out.joints)
}

/// A skinned primitive: vertices at bind time, their four strongest influences, the skin.
fn skinned_primitive(name: &str, primitive: &gltf::Primitive, skin: &gltf::Skin, buffers: &[gltf::buffer::Data], node_joint: &HashMap<usize, usize>) -> Result<Option<SkinnedMesh>, String> {
    if primitive.mode() != gltf::mesh::Mode::Triangles {
        log::warn!("glTF primitive {name} is not a triangle list: not imported");
        return Ok(None);
    }
    let reader = primitive.reader(|b| Some(&buffers[b.index()]));
    let Some(positions) = reader.read_positions() else { return Ok(None) };
    let positions: Vec<[f32; 3]> = positions.collect();
    let count = positions.len();
    let normals: Vec<[f32; 3]> = reader.read_normals().map(|n| n.collect()).unwrap_or_else(|| vec![[0.0, 1.0, 0.0]; count]);
    let uvs: Vec<[f32; 2]> = reader.read_tex_coords(0).map(|t| t.into_f32().collect()).unwrap_or_else(|| vec![[0.0; 2]; count]);
    let indices: Vec<u32> = reader.read_indices().map(|i| i.into_u32().collect()).unwrap_or_else(|| (0..count as u32).collect());

    // every JOINTS_n / WEIGHTS_n set, reduced to the four strongest influences
    let mut influences: Vec<Vec<(u16, f32)>> = vec![Vec::new(); count];
    let mut set = 0;
    while let (Some(joints), Some(weights)) = (reader.read_joints(set), reader.read_weights(set)) {
        for (v, (j, w)) in joints.into_u16().zip(weights.into_f32()).enumerate().take(count) {
            for k in 0..4 {
                influences[v].push((j[k], w[k]));
            }
        }
        set += 1;
    }
    if set == 0 {
        return Err(format!("glTF primitive {name} is skinned but has no JOINTS_0/WEIGHTS_0"));
    }
    let (joints, weights): (Vec<_>, Vec<_>) = influences.into_iter().map(strongest_influences).unzip();

    let skin_joints = skin
        .joints()
        .map(|j| node_joint.get(&j.index()).copied().ok_or_else(|| format!("skin joint node {} is not in the skeleton", j.index())))
        .collect::<Result<Vec<_>, _>>()?;
    let inverse_bind: Vec<Mat4> = match skin.reader(|b| Some(&buffers[b.index()])).read_inverse_bind_matrices() {
        Some(m) => m.map(|m| Mat4::from_cols_array_2d(&m)).collect(),
        None => vec![Mat4::IDENTITY; skin_joints.len()],
    };
    if joints.iter().flatten().any(|&j| j as usize >= skin_joints.len()) {
        return Err(format!("glTF primitive {name} references a joint beyond its skin's {}", skin_joints.len()));
    }
    let vertices = (0..count)
        .map(|i| Vertex { position: [positions[i][0], positions[i][1], positions[i][2], 1.0], normal: normals[i], uv: uvs[i] })
        .collect();
    Ok(Some(SkinnedMesh { name: name.to_string(), vertices, indices, joints, weights, skin_joints, inverse_bind, material: primitive.material().index() }))
}

/// One sampler's keys, for evaluation at any time.
struct Track {
    times: Vec<f32>,
    /// Per key: the value (xyz, or xyzw for rotations); for cubic splines, in-tangent, value,
    /// out-tangent.
    values: Vec<[f32; 4]>,
    interpolation: gltf::animation::Interpolation,
    rotation: bool,
}

impl Track {
    fn value(&self, k: usize) -> [f32; 4] {
        match self.interpolation {
            gltf::animation::Interpolation::CubicSpline => self.values[3 * k + 1],
            _ => self.values[k],
        }
    }

    fn sample(&self, t: f32) -> [f32; 4] {
        use gltf::animation::Interpolation::*;
        let n = self.times.len();
        if n == 1 || t <= self.times[0] {
            return self.value(0);
        }
        if t >= self.times[n - 1] {
            return self.value(n - 1);
        }
        let k = self.times.partition_point(|&x| x <= t) - 1;
        let (t0, t1) = (self.times[k], self.times[k + 1]);
        let dt = t1 - t0;
        let s = if dt > 0.0 { (t - t0) / dt } else { 0.0 };
        match self.interpolation {
            Step => self.value(k),
            Linear => {
                let (a, b) = (self.value(k), self.value(k + 1));
                if self.rotation {
                    Quat::from_array(a).slerp(Quat::from_array(b), s).to_array()
                } else {
                    [0, 1, 2, 3].map(|i| a[i] + (b[i] - a[i]) * s)
                }
            }
            CubicSpline => {
                let (p0, m0) = (self.values[3 * k + 1], self.values[3 * k + 2]);
                let (m1, p1) = (self.values[3 * (k + 1)], self.values[3 * (k + 1) + 1]);
                let (s2, s3) = (s * s, s * s * s);
                let v = [0, 1, 2, 3].map(|i| {
                    (2.0 * s3 - 3.0 * s2 + 1.0) * p0[i] + (s3 - 2.0 * s2 + s) * dt * m0[i] + (-2.0 * s3 + 3.0 * s2) * p1[i] + (s3 - s2) * dt * m1[i]
                });
                if self.rotation { Quat::from_array(v).normalize().to_array() } else { v }
            }
        }
    }
}

/// An animation resampled at `sample_rate` (or its keys' rate) into a clip over the skeleton.
fn clip(animation: &gltf::Animation, index: usize, buffers: &[gltf::buffer::Data], joints: &JointNodes, sample_rate: Option<f32>) -> Result<Clip, String> {
    use gltf::animation::util::ReadOutputs;
    use gltf::animation::Property;
    let name = animation.name().map_or_else(|| format!("animation_{index}"), str::to_string);
    // per joint: translation, rotation, scale tracks
    let mut tracks: Vec<[Option<Track>; 3]> = (0..joints.own.len()).map(|_| [None, None, None]).collect();
    let (mut start, mut end, mut spacing) = (f32::MAX, f32::MIN, f32::MAX);
    for channel in animation.channels() {
        let target = channel.target();
        let Some(&joint) = joints.node_joint.get(&target.node().index()) else {
            log::warn!("animation '{name}' animates node {} outside the skeleton: ignored", target.node().index());
            continue;
        };
        let reader = channel.reader(|b| Some(&buffers[b.index()]));
        let Some(times) = reader.read_inputs() else { continue };
        let times: Vec<f32> = times.collect();
        let (slot, values): (usize, Vec<[f32; 4]>) = match reader.read_outputs() {
            Some(ReadOutputs::Translations(v)) => (0, v.map(|p| [p[0], p[1], p[2], 0.0]).collect()),
            Some(ReadOutputs::Rotations(v)) => (1, v.into_f32().collect()),
            Some(ReadOutputs::Scales(v)) => (2, v.map(|p| [p[0], p[1], p[2], 0.0]).collect()),
            _ => continue,
        };
        if times.is_empty() {
            continue;
        }
        start = start.min(times[0]);
        end = end.max(times[times.len() - 1]);
        spacing = times.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 1e-6).fold(spacing, f32::min);
        let interpolation = channel.sampler().interpolation();
        debug_assert_eq!(target.property() == Property::Rotation, slot == 1);
        tracks[joint][slot] = Some(Track { times, values, interpolation, rotation: slot == 1 });
    }
    if start > end {
        (start, end) = (0.0, 0.0);
    }
    // the keys' rate, rounded to a whole number of frames per second
    let rate = sample_rate.unwrap_or(if spacing < f32::MAX { (1.0 / spacing).round().clamp(1.0, 240.0) } else { 30.0 });
    let frames = ((end - start) * rate).round() as usize + 1;
    let poses: Vec<Pose> = (0..frames)
        .map(|f| {
            let t = start + f as f32 / rate;
            Pose {
                local: tracks
                    .iter()
                    .enumerate()
                    .map(|(j, [translation, rotation, scale])| {
                        // channels replace the node's own transform; the folded nodes stay above it
                        let mut out = joints.own[j];
                        if let Some(tr) = translation {
                            let v = tr.sample(t);
                            out.translation = Vec3::new(v[0], v[1], v[2]);
                        }
                        if let Some(tr) = rotation {
                            out.rotation = Quat::from_array(tr.sample(t)).normalize();
                        }
                        if let Some(tr) = scale {
                            let v = tr.sample(t);
                            out.scale = Vec3::new(v[0], v[1], v[2]);
                        }
                        joints.above[j].mul(&out)
                    })
                    .collect(),
            }
        })
        .collect();
    Ok(Clip::from_poses(&name, rate, &poses))
}
