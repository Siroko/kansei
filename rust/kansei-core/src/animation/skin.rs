use glam::{Mat4, Vec3};

use super::Transform;
use crate::geometries::{Geometry, Vertex};

/// Joint influences per vertex.
pub const MAX_INFLUENCES: usize = 4;

/// A mesh deformed by a skeleton: its vertices at bind time, each vertex's joints and weights,
/// and the skin (which skeleton joints it uses, with their inverse bind matrices).
#[derive(Debug, Clone)]
pub struct SkinnedMesh {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    /// Per vertex, up to four skin joints (indices into `skin_joints`)...
    pub joints: Vec<[u16; MAX_INFLUENCES]>,
    /// ...and their weights, summing to 1.
    pub weights: Vec<[f32; MAX_INFLUENCES]>,
    /// The skeleton joint of each skin joint.
    pub skin_joints: Vec<usize>,
    /// Model space to each skin joint's space at bind time.
    pub inverse_bind: Vec<Mat4>,
    /// Index of the source material, if any.
    pub material: Option<usize>,
}

impl SkinnedMesh {
    /// The renderable geometry (bind-pose vertices; the vertex shader skins them).
    pub fn geometry(&self) -> Geometry {
        Geometry::new(&self.name, self.vertices.clone(), self.indices.clone())
    }

    /// Joint matrices for skinning: each skin joint's model transform times its inverse bind
    /// matrix, from the skeleton's model-space pose.
    pub fn palette(&self, model: &[Transform], out: &mut Vec<Mat4>) {
        out.clear();
        out.extend(self.skin_joints.iter().zip(&self.inverse_bind).map(|(&j, ibm)| model[j].to_mat4() * *ibm));
    }

    /// Skinned positions and normals on the CPU (the vertex shader's reference).
    pub fn skin_cpu(&self, palette: &[Mat4]) -> Vec<(Vec3, Vec3)> {
        self.vertices
            .iter()
            .enumerate()
            .map(|(v, vertex)| {
                let m = self.joints[v].iter().zip(&self.weights[v]).fold(Mat4::ZERO, |acc, (&j, &w)| acc + palette[j as usize] * w);
                let p = Vec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
                (m.transform_point3(p), m.transform_vector3(Vec3::from(vertex.normal)))
            })
            .collect()
    }

    /// The per-vertex records the skinning shader reads (`SKINNING_WGSL`): the four joints as
    /// u16 pairs, then the four weights as unorm16 pairs that sum to exactly 1.
    pub fn skin_words(&self) -> Vec<[u32; 4]> {
        self.joints
            .iter()
            .zip(&self.weights)
            .map(|(j, w)| {
                let q = quantize_weights(w);
                [
                    j[0] as u32 | (j[1] as u32) << 16,
                    j[2] as u32 | (j[3] as u32) << 16,
                    q[0] as u32 | (q[1] as u32) << 16,
                    q[2] as u32 | (q[3] as u32) << 16,
                ]
            })
            .collect()
    }
}

/// Weights as unorm16 that sum to exactly 65535 (the rounding error goes to the largest).
fn quantize_weights(w: &[f32; MAX_INFLUENCES]) -> [u16; MAX_INFLUENCES] {
    let mut q = w.map(|x| (x.clamp(0.0, 1.0) * 65535.0).round() as i32);
    let largest = (0..MAX_INFLUENCES).max_by(|&a, &b| w[a].total_cmp(&w[b])).unwrap();
    q[largest] += 65535 - q.iter().sum::<i32>();
    q.map(|x| x.clamp(0, 65535) as u16)
}

/// The `MAX_INFLUENCES` largest of a vertex's influences (a joint listed twice counts once, with
/// both weights), renormalized to sum to 1 (joint 0 with full weight when none has weight).
pub fn strongest_influences(influences: impl IntoIterator<Item = (u16, f32)>) -> ([u16; MAX_INFLUENCES], [f32; MAX_INFLUENCES]) {
    let mut all: Vec<(u16, f32)> = Vec::new();
    for (j, w) in influences.into_iter().filter(|(_, w)| *w > 0.0) {
        match all.iter_mut().find(|(k, _)| *k == j) {
            Some((_, total)) => *total += w,
            None => all.push((j, w)),
        }
    }
    all.sort_by(|a, b| b.1.total_cmp(&a.1));
    all.truncate(MAX_INFLUENCES);
    let total: f32 = all.iter().map(|(_, w)| w).sum();
    let mut joints = [0u16; MAX_INFLUENCES];
    let mut weights = [0.0f32; MAX_INFLUENCES];
    if total <= 0.0 {
        weights[0] = 1.0;
        return (joints, weights);
    }
    for (k, (j, w)) in all.into_iter().enumerate() {
        joints[k] = j;
        weights[k] = w / total;
    }
    (joints, weights)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::animation::{Pose, Skeleton};
    use glam::Quat;

    /// A bone 1 m long up +y with a child at its tip, and a vertex halfway weighted to both.
    pub(crate) fn two_bone_mesh() -> (Skeleton, SkinnedMesh) {
        let skeleton = Skeleton::new(
            vec!["hip".into(), "knee".into()],
            vec![None, Some(0)],
            vec![Transform::IDENTITY, Transform::from_translation_rotation(Vec3::Y, Quat::IDENTITY)],
        );
        let bind = skeleton.rest_model();
        let vertex = |p: [f32; 3]| Vertex { position: [p[0], p[1], p[2], 1.0], normal: [1.0, 0.0, 0.0], uv: [0.0; 2] };
        let mesh = SkinnedMesh {
            name: "leg".into(),
            vertices: vec![vertex([0.1, 0.5, 0.0]), vertex([0.1, 1.5, 0.0]), vertex([0.1, 1.0, 0.0])],
            indices: vec![0, 1, 2],
            joints: vec![[0, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]],
            weights: vec![[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
            skin_joints: vec![0, 1],
            inverse_bind: bind.iter().map(|t| t.to_mat4().inverse()).collect(),
            material: None,
        };
        (skeleton, mesh)
    }

    #[test]
    fn the_rest_pose_leaves_the_mesh_as_bound() {
        let (skeleton, mesh) = two_bone_mesh();
        let mut palette = Vec::new();
        mesh.palette(&skeleton.rest_model(), &mut palette);
        for ((p, n), v) in mesh.skin_cpu(&palette).iter().zip(&mesh.vertices) {
            assert!(p.abs_diff_eq(Vec3::new(v.position[0], v.position[1], v.position[2]), 1e-6));
            assert!(n.abs_diff_eq(Vec3::X, 1e-6));
        }
    }

    #[test]
    fn a_bent_knee_moves_its_vertices_and_blends_the_shared_one() {
        let (skeleton, mesh) = two_bone_mesh();
        let mut pose = Pose::rest(&skeleton);
        // bend the knee 90 degrees about z: the shin points along -x
        pose.local[1].rotation = Quat::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let mut palette = Vec::new();
        mesh.palette(&pose.model(&skeleton), &mut palette);
        let skinned = mesh.skin_cpu(&palette);
        // the thigh vertex stays, the shin vertex turns about the knee (0, 1, 0)
        assert!(skinned[0].0.abs_diff_eq(Vec3::new(0.1, 0.5, 0.0), 1e-6));
        assert!(skinned[1].0.abs_diff_eq(Vec3::new(-0.5, 1.1, 0.0), 1e-5), "{}", skinned[1].0);
        assert!(skinned[1].1.abs_diff_eq(Vec3::Y, 1e-5));
        // the knee vertex averages both joints' results: (0.1, 1, 0) and (0, 1.1, 0)
        assert!(skinned[2].0.abs_diff_eq(Vec3::new(0.05, 1.05, 0.0), 1e-5), "{}", skinned[2].0);
    }

    #[test]
    fn influences_keep_the_four_strongest_and_sum_to_one() {
        let (joints, weights) = strongest_influences([(3, 0.1), (7, 0.4), (1, 0.05), (9, 0.2), (2, 0.25)]);
        assert_eq!(joints, [7, 2, 9, 3]);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!((weights[0] - 0.4 / 0.95).abs() < 1e-6);
        let (joints, weights) = strongest_influences([(5, 0.25), (2, 0.5), (5, 0.25)]);
        assert_eq!((joints[..2].to_vec(), weights), (vec![5, 2], [0.5, 0.5, 0.0, 0.0]));
        let (joints, weights) = strongest_influences([]);
        assert_eq!((joints, weights), ([0; 4], [1.0, 0.0, 0.0, 0.0]));
    }

    #[test]
    fn packed_weights_sum_to_exactly_one() {
        for w in [[1.0 / 3.0; 3].into_iter().chain([0.0]).collect::<Vec<_>>(), vec![0.7, 0.2, 0.1, 0.0], vec![0.25; 4]] {
            let q = quantize_weights(&[w[0], w[1], w[2], w[3]]);
            assert_eq!(q.iter().map(|&x| x as u32).sum::<u32>(), 65535, "{w:?} -> {q:?}");
        }
        let (_, mesh) = two_bone_mesh();
        let words = mesh.skin_words();
        assert_eq!(words[2][0], 1 << 16); // joints 0 and 1
        let (a, b) = (words[2][2] & 0xffff, words[2][2] >> 16);
        assert_eq!(a + b, 65535);
        assert!(a.abs_diff(b) <= 1, "{a} {b}");
    }
}
