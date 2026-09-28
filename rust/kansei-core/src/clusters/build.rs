use glam::Vec3;
use optimesh::clusterizer::{build_meshlets, build_meshlets_bound, Meshlet, MeshletBuffers, Positions};
use optimesh::meshletutils::compute_cluster_bounds;
use optimesh::partition::partition_clusters;
use optimesh::simplifier::{simplify_with_attributes, Attributes, SimplifyTarget, VertexData, SIMPLIFY_ERROR_ABSOLUTE, SIMPLIFY_SPARSE, SIMPLIFY_VERTEX_LOCK};
use std::collections::HashMap;

use super::{Cluster, ClusterMesh, ClusterOptions, Sphere};
use crate::geometries::Geometry;

impl ClusterMesh {
    /// Split `geometry` into clusters and build the graph of coarser versions over them.
    pub fn build(geometry: &Geometry, options: &ClusterOptions) -> ClusterMesh {
        let mut positions: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.position[0], v.position[1], v.position[2]]).collect();
        let tolerance = WELD * extent(&positions);
        weld(&mut positions, tolerance);
        let attributes: Vec<f32> = geometry.vertices.iter().flat_map(|v| [v.normal[0], v.normal[1], v.normal[2], v.uv[0], v.uv[1]]).collect();
        let weights = [options.normal_weight, options.normal_weight, options.normal_weight, options.uv_weight, options.uv_weight];
        let position_ids = position_ids(&positions);
        // one vertex per position, so clusters on either side of a seam are neighbours
        let mut first = vec![u32::MAX; position_ids.len()];
        let canonical: Vec<u32> = position_ids
            .iter()
            .enumerate()
            .map(|(v, &p)| {
                if first[p as usize] == u32::MAX {
                    first[p as usize] = v as u32;
                }
                first[p as usize]
            })
            .collect();
        let mut vertices = geometry.vertices.clone();
        for (v, p) in vertices.iter_mut().zip(positions.chunks(3)) {
            v.position[..3].copy_from_slice(p);
        }
        let mut mesh = ClusterMesh { vertices, clusters: Vec::new(), cluster_vertices: Vec::new(), cluster_triangles: Vec::new() };
        // the clusters still without a parent, with their triangles
        let mut pending = mesh.split(&geometry.indices, &positions, 0.0, None, 0, options);
        let mut level = 0;
        while pending.len() > 1 {
            level += 1;
            let groups = partition(&pending, &positions, &canonical, options.group_size);
            let lock = shared_vertex_locks(&groups, &pending, &position_ids);
            let mut next = Vec::new();
            let mut progress = false;
            for group in &groups {
                let merged: Vec<u32> = group.iter().flat_map(|&i| pending[i].1.iter().copied()).collect();
                let target = ((merged.len() as f32 * options.simplify_ratio) as usize / 3) * 3;
                let mut simplified = vec![0u32; merged.len()];
                let (count, error) = simplify_with_attributes(
                    &mut simplified,
                    &merged,
                    &VertexData { positions: &positions, count: positions.len() / 3, stride: 12 },
                    &Attributes { data: &attributes, stride: 20, weights: &weights, count: 5 },
                    Some(&lock),
                    &SimplifyTarget { target_index_count: target, target_error: f32::MAX, options: SIMPLIFY_SPARSE | SIMPLIFY_ERROR_ABSOLUTE },
                );
                if count as f32 > merged.len() as f32 * options.stall_ratio {
                    // too little came off: next round, grouped with other neighbours
                    next.extend(group.iter().map(|&i| pending[i].clone()));
                    continue;
                }
                progress = true;
                // never less than a child's error, from a sphere round all of theirs
                let error = group.iter().map(|&i| mesh.clusters[pending[i].0].error).fold(error, f32::max);
                let bounds = Sphere::enclosing(group.iter().map(|&i| mesh.clusters[pending[i].0].lod_bounds));
                for &i in group {
                    let child = &mut mesh.clusters[pending[i].0];
                    child.parent_error = error;
                    child.parent_bounds = bounds;
                }
                next.extend(mesh.split(&simplified[..count], &positions, error, Some(bounds), level, options));
            }
            if !progress {
                break;
            }
            pending = next;
        }
        mesh
    }

    /// Clusters of `indices` (into `vertices`), carrying `error` and `lod_bounds` (their own
    /// bounds when `None`); per new cluster, its index and its triangles as indices into
    /// `vertices`.
    pub(super) fn split(&mut self, indices: &[u32], positions: &[f32], error: f32, lod_bounds: Option<Sphere>, level: u32, options: &ClusterOptions) -> Vec<(usize, Vec<u32>)> {
        if indices.is_empty() {
            return Vec::new();
        }
        let bound = build_meshlets_bound(indices.len(), options.max_vertices, options.max_triangles);
        let mut meshlets = vec![Meshlet::default(); bound];
        let mut vertices = vec![0u32; bound * options.max_vertices];
        let mut triangles = vec![0u8; bound * options.max_triangles * 3];
        let count = build_meshlets(
            &mut MeshletBuffers { meshlets: &mut meshlets, vertices: &mut vertices, triangles: &mut triangles },
            indices,
            &Positions { data: positions, count: positions.len() / 3, stride: 12 },
            options.max_vertices,
            options.max_triangles,
            options.cone_weight,
        );
        let mut out = Vec::with_capacity(count);
        for m in &meshlets[..count] {
            let local_vertices = &vertices[m.vertex_offset as usize..(m.vertex_offset + m.vertex_count) as usize];
            // (meshlet triangle offsets count indices, 3 per triangle)
            let local_triangles = &triangles[m.triangle_offset as usize..(m.triangle_offset + m.triangle_count * 3) as usize];
            let global: Vec<u32> = local_triangles.iter().map(|&t| local_vertices[t as usize]).collect();
            let b = compute_cluster_bounds(&global, positions, positions.len() / 3, 12);
            let bounds = Sphere { center: Vec3::from(b.center), radius: b.radius };
            let lod_bounds = lod_bounds.unwrap_or(bounds);
            out.push((self.clusters.len(), global));
            self.clusters.push(Cluster {
                vertex_offset: self.cluster_vertices.len() as u32,
                vertex_count: m.vertex_count,
                triangle_offset: (self.cluster_triangles.len() / 3) as u32,
                triangle_count: m.triangle_count,
                bounds,
                cone_apex: Vec3::from(b.cone_apex),
                cone_axis: Vec3::from(b.cone_axis),
                cone_cutoff: b.cone_cutoff,
                error,
                lod_bounds,
                parent_error: f32::INFINITY,
                parent_bounds: lod_bounds,
                level,
            });
            self.cluster_vertices.extend_from_slice(local_vertices);
            self.cluster_triangles.extend_from_slice(local_triangles);
        }
        out
    }
}

/// Positions closer than this share of the mesh's extent are one position (`weld`).
const WELD: f32 = 1e-6;

/// The diagonal of `positions`' bounding box.
fn extent(positions: &[f32]) -> f32 {
    let (lo, hi) = positions.chunks(3).fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), p| {
        let p = Vec3::from_slice(p);
        (lo.min(p), hi.max(p))
    });
    if lo.x > hi.x { 0.0 } else { lo.distance(hi) }
}

/// Each position within `tolerance` of an earlier one takes that one's value, bit for bit (and
/// -0.0 is 0.0): a seam's copies that differ by rounding or by the sign of zero become one
/// position to the simplifier and the locks, which match positions exactly.
fn weld(positions: &mut [f32], tolerance: f32) {
    let tolerance = tolerance.max(f32::MIN_POSITIVE);
    let mut cells: HashMap<[i64; 3], Vec<usize>> = HashMap::new();
    for i in 0..positions.len() / 3 {
        let p = Vec3::from_slice(&positions[i * 3..i * 3 + 3]) + Vec3::ZERO;
        let c = (p / tolerance).floor();
        let c = [c.x as i64, c.y as i64, c.z as i64];
        let mut near = None;
        'search: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    for &j in cells.get(&[c[0] + dx, c[1] + dy, c[2] + dz]).into_iter().flatten() {
                        if Vec3::from_slice(&positions[j * 3..j * 3 + 3]).distance(p) <= tolerance {
                            near = Some(j);
                            break 'search;
                        }
                    }
                }
            }
        }
        match near {
            Some(j) => positions.copy_within(j * 3..j * 3 + 3, i * 3),
            None => {
                positions[i * 3..i * 3 + 3].copy_from_slice(&p.to_array());
                cells.entry(c).or_default().push(i);
            }
        }
    }
}

/// One id per distinct position (seams split vertices, not positions).
/// `position_ids`, for the card tests.
#[cfg(test)]
pub(super) fn position_ids_for_tests(positions: &[f32]) -> Vec<u32> {
    position_ids(positions)
}

fn position_ids(positions: &[f32]) -> Vec<u32> {
    let mut ids = HashMap::new();
    positions
        .chunks(3)
        .map(|p| {
            let next = ids.len() as u32;
            *ids.entry([p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]).or_insert(next)
        })
        .collect()
}

/// Groups of about `size` neighbouring clusters (indices into `pending`); `canonical` maps each
/// vertex to one vertex per position, so a seam doesn't part neighbours.
fn partition(pending: &[(usize, Vec<u32>)], positions: &[f32], canonical: &[u32], size: usize) -> Vec<Vec<usize>> {
    let indices: Vec<u32> = pending.iter().flat_map(|(_, t)| t.iter().map(|&v| canonical[v as usize])).collect();
    let counts: Vec<u32> = pending.iter().map(|(_, t)| t.len() as u32).collect();
    let mut destination = vec![0u32; pending.len()];
    let groups = partition_clusters(&mut destination, &indices, &counts, Some(positions), positions.len() / 3, 12, size);
    let mut out = vec![Vec::new(); groups];
    for (i, &g) in destination.iter().enumerate() {
        out[g as usize].push(i);
    }
    out
}

/// `SIMPLIFY_VERTEX_LOCK` on every vertex whose position more than one group uses: groups meet
/// at the same vertices whatever level each is drawn at.
fn shared_vertex_locks(groups: &[Vec<usize>], pending: &[(usize, Vec<u32>)], position_ids: &[u32]) -> Vec<u8> {
    const NONE: u32 = u32::MAX;
    const SHARED: u32 = u32::MAX - 1;
    let mut owner = vec![NONE; position_ids.len()];
    for (g, group) in groups.iter().enumerate() {
        for &i in group {
            for &v in &pending[i].1 {
                let p = position_ids[v as usize] as usize;
                owner[p] = match owner[p] {
                    NONE => g as u32,
                    o if o == g as u32 => o,
                    _ => SHARED,
                };
            }
        }
    }
    position_ids.iter().map(|&p| if owner[p as usize] == SHARED { SIMPLIFY_VERTEX_LOCK } else { 0 }).collect()
}
