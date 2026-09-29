use glam::{Mat4, Quat, Vec3};

use super::*;

/// A .glb from its JSON and binary chunk.
pub(crate) fn glb(json: &str, bin: &[u8]) -> Vec<u8> {
    let mut json = json.as_bytes().to_vec();
    while json.len() % 4 != 0 {
        json.push(b' ');
    }
    let mut bin = bin.to_vec();
    while bin.len() % 4 != 0 {
        bin.push(0);
    }
    let total = 12 + 8 + json.len() + 8 + bin.len();
    let mut out = Vec::with_capacity(total);
    out.extend_from_slice(b"glTF");
    out.extend_from_slice(&2u32.to_le_bytes());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json);
    out.extend_from_slice(&(bin.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&bin);
    out
}

/// Builds a binary chunk of accessors.
#[derive(Default)]
pub(crate) struct Bin {
    pub(crate) bytes: Vec<u8>,
    pub(crate) views: Vec<String>,
    pub(crate) accessors: Vec<String>,
}

impl Bin {
    /// An accessor of `data` (component type, glTF type, count); returns its index.
    pub(crate) fn add(&mut self, data: &[u8], component: u32, ty: &str, count: usize, extra: &str) -> usize {
        while self.bytes.len() % 4 != 0 {
            self.bytes.push(0);
        }
        self.views.push(format!(r#"{{"buffer":0,"byteOffset":{},"byteLength":{}}}"#, self.bytes.len(), data.len()));
        self.bytes.extend_from_slice(data);
        self.accessors.push(format!(r#"{{"bufferView":{},"componentType":{component},"type":"{ty}","count":{count}{extra}}}"#, self.views.len() - 1));
        self.accessors.len() - 1
    }

    pub(crate) fn floats(&mut self, data: &[f32], ty: &str, extra: &str) -> usize {
        let width = match ty {
            "SCALAR" => 1,
            "VEC2" => 2,
            "VEC3" => 3,
            "VEC4" => 4,
            "MAT4" => 16,
            _ => unreachable!(),
        };
        self.add(bytemuck::cast_slice(data), 5126, ty, data.len() / width, extra)
    }

    pub(crate) fn json(&self) -> String {
        format!(r#""buffers":[{{"byteLength":{}}}],"bufferViews":[{}],"accessors":[{}]"#, self.bytes.len().div_ceil(4) * 4, self.views.join(","), self.accessors.join(","))
    }
}

/// The armature's transform: Z-up to Y-up (-90 degrees about x) at a hundredth scale.
fn armature() -> Transform {
    Transform::new(Vec3::new(0.0, 0.0, 0.5), Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2), Vec3::splat(0.01))
}

/// An armature node (Z-up, centimetres) over a hip joint and a knee 100 cm above it along z (via a
/// non-joint helper node 40 cm up, then 60 cm), a
/// skinned triangle (one vertex with a second influence set), and an animation turning the knee
/// 90 degrees about x over 1 s (linear) and stepping the hip up 10 cm at 1 s.
fn leg_glb() -> Vec<u8> {
    let mut bin = Bin::default();
    let a = armature();
    let hip_world = a.mul(&Transform::from_translation_rotation(Vec3::new(0.0, 0.0, 100.0), Quat::IDENTITY));
    let knee_world = hip_world.mul(&Transform::from_translation_rotation(Vec3::new(0.0, 0.0, 100.0), Quat::IDENTITY));
    // vertices in world space at bind time: at the hip, at the knee, above the knee
    let positions = [hip_world.translation, knee_world.translation, knee_world.translation + Vec3::new(0.0, 0.5, 0.0)];
    let pos: Vec<f32> = positions.iter().flat_map(|p| p.to_array()).collect();
    let p = bin.floats(&pos, "VEC3", r#","min":[-1,-1,-1],"max":[3,3,3]"#);
    let n = bin.floats(&[1.0, 0.0, 0.0].repeat(3), "VEC3", "");
    let idx = bin.add(bytemuck::cast_slice(&[0u16, 1, 2]), 5123, "SCALAR", 3, "");
    let j0 = bin.add(&[0u8, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], 5121, "VEC4", 3, "");
    let w0 = bin.floats(&[1.0, 0.0, 0.0, 0.0, 0.3, 0.3, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], "VEC4", "");
    // a second set: vertex 1 gets 0.4 more on the knee
    let j1 = bin.add(&[0u8; 4].into_iter().chain([1, 0, 0, 0]).chain([0; 4]).collect::<Vec<_>>(), 5121, "VEC4", 3, "");
    let w1 = bin.floats(&[0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], "VEC4", "");
    let ibm: Vec<f32> = [hip_world, knee_world].iter().flat_map(|t| t.to_mat4().inverse().to_cols_array()).collect();
    let ibm = bin.floats(&ibm, "MAT4", "");
    let times = bin.floats(&[0.0, 1.0], "SCALAR", r#","min":[0],"max":[1]"#);
    let knee_rot: Vec<f32> = [Quat::IDENTITY, Quat::from_rotation_x(std::f32::consts::FRAC_PI_2)].iter().flat_map(|q| q.to_array()).collect();
    let knee_rot = bin.floats(&knee_rot, "VEC4", "");
    let hip_t = bin.floats(&[0.0, 0.0, 100.0, 0.0, 0.0, 110.0], "VEC3", "");
    let r = armature().rotation.to_array();
    let json = format!(
        r#"{{"asset":{{"version":"2.0"}},"scene":0,"scenes":[{{"nodes":[0]}}],
        "nodes":[
          {{"name":"Armature","translation":[0,0,0.5],"rotation":[{},{},{},{}],"scale":[0.01,0.01,0.01],"children":[1,3]}},
          {{"name":"hip","translation":[0,0,100],"children":[4]}},
          {{"name":"knee","translation":[0,0,60]}},
          {{"name":"LegMesh","mesh":0,"skin":0}},
          {{"name":"KneeOffset","translation":[0,0,40],"children":[2]}}
        ],
        "meshes":[{{"name":"Leg","primitives":[{{"attributes":{{"POSITION":{p},"NORMAL":{n},"JOINTS_0":{j0},"WEIGHTS_0":{w0},"JOINTS_1":{j1},"WEIGHTS_1":{w1}}},"indices":{idx}}}]}}],
        "skins":[{{"joints":[1,2],"inverseBindMatrices":{ibm}}}],
        "animations":[{{"name":"Kick","samplers":[
            {{"input":{times},"output":{knee_rot},"interpolation":"LINEAR"}},
            {{"input":{times},"output":{hip_t},"interpolation":"STEP"}}],
          "channels":[{{"sampler":0,"target":{{"node":2,"path":"rotation"}}}},{{"sampler":1,"target":{{"node":1,"path":"translation"}}}}]}}],
        {}}}"#,
        r[0], r[1], r[2], r[3],
        bin.json()
    );
    glb(&json, &bin.bytes)
}

#[test]
fn gltf_import_reads_the_skeleton_skin_and_animation() {
    let gltf = SkinnedGltf::from_slice(&leg_glb(), Some(10.0)).unwrap();
    let skeleton = &gltf.skeleton;
    assert_eq!(skeleton.names, vec!["hip".to_string(), "knee".to_string()]);
    assert_eq!(skeleton.parents, vec![None, Some(0)]);
    // the armature is folded into the hip: model space is the file's world space
    let rest = skeleton.rest_model();
    let hip_world = armature().transform_point(Vec3::new(0.0, 0.0, 100.0));
    assert!(rest[0].translation.abs_diff_eq(hip_world, 1e-5), "{}", rest[0].translation);
    assert!(rest[1].translation.abs_diff_eq(Vec3::new(0.0, 2.0, 0.5), 1e-5), "{}", rest[1].translation);

    let mesh = &gltf.meshes[0];
    assert_eq!((mesh.vertices.len(), mesh.indices.clone()), (3, vec![0, 1, 2]));
    assert_eq!(mesh.skin_joints, vec![0, 1]);
    // vertex 1: 0.3 hip, 0.3 + 0.4 knee over both sets, renormalized
    assert_eq!(mesh.joints[1][..2], [1, 0]);
    assert!((mesh.weights[1][0] - 0.7).abs() < 1e-6 && (mesh.weights[1][1] - 0.3).abs() < 1e-6, "{:?}", mesh.weights[1]);
    // at rest the palette is the identity: the mesh stays as bound
    let mut palette = Vec::new();
    mesh.palette(&rest, &mut palette);
    assert!(palette.iter().all(|m| m.abs_diff_eq(Mat4::IDENTITY, 1e-5)));

    let clip = &gltf.clips[0];
    assert_eq!((clip.name.as_str(), clip.frame_count(), clip.joint_count()), ("Kick", 11, 2));
    let mut pose = Pose::rest(skeleton);
    clip.sample(0.5, &mut pose);
    // linear keys slerp: 45 degrees halfway
    assert!(pose.local[1].rotation.abs_diff_eq(Quat::from_rotation_x(std::f32::consts::FRAC_PI_4), 1e-5));
    // step keys hold until the next key; the hip's channel is under the folded armature
    assert!(pose.local[0].translation.abs_diff_eq(hip_world, 1e-5));
    clip.frame_pose(10, &mut pose);
    assert!(pose.local[0].translation.abs_diff_eq(armature().transform_point(Vec3::new(0.0, 0.0, 110.0)), 1e-5));
    // the knee turned 90 degrees about its (armature-rotated) x: the vertex above it swings
    let model = pose.model(skeleton);
    mesh.palette(&model, &mut palette);
    let skinned = mesh.skin_cpu(&palette);
    let knee = model[1].translation;
    assert!(knee.abs_diff_eq(Vec3::new(0.0, 2.1, 0.5), 1e-5), "{knee}");
    // the vertex 0.5 m above the knee is 50 cm along the knee's z; +90 degrees about x turns
    // that to its -y, which the armature's -90 degrees about x takes to world +z
    assert!(skinned[2].0.abs_diff_eq(Vec3::new(0.0, 2.1, 1.0), 1e-5), "{}", skinned[2].0);
}
