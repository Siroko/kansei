use wgpu::util::DeviceExt;

use super::buffers::{BVH4Node, BinaryBVHNode, PackedInstance, PackedTriangle};
use super::TLASBuilder;
use crate::math::Mat4;
use crate::objects::Scene;
use crate::renderers::Renderer;

// ── SAH constants ──────────────────────────────────────────────────────────

const SAH_BINS: usize = 12;
const MAX_LEAF_SIZE: usize = 4;
const TRAVERSAL_COST: f32 = 1.0;
const INTERSECTION_COST: f32 = 1.0;

/// Triangle stride in floats (24 floats = 96 bytes, matching PackedTriangle).
pub const TRI_STRIDE_FLOATS: usize = 24;

// ── Helpers ────────────────────────────────────────────────────────────────

/// AABB surface area (half-area would work too, but we match the TS engine).
#[inline]
fn aabb_surface_area(min: [f32; 3], max: [f32; 3]) -> f32 {
    let dx = max[0] - min[0];
    let dy = max[1] - min[1];
    let dz = max[2] - min[2];
    2.0 * (dx * dy + dy * dz + dz * dx)
}

/// Component-wise min of two AABB corners.
#[inline]
fn vec3_min(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].min(b[0]), a[1].min(b[1]), a[2].min(b[2])]
}

/// Component-wise max of two AABB corners.
#[inline]
fn vec3_max(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0].max(b[0]), a[1].max(b[1]), a[2].max(b[2])]
}

// ── Per-triangle AABB cache ────────────────────────────────────────────────

/// Pre-computed per-triangle bounds and centroids for SAH evaluation.
struct TriangleAABBs {
    min_x: Vec<f32>,
    min_y: Vec<f32>,
    min_z: Vec<f32>,
    max_x: Vec<f32>,
    max_y: Vec<f32>,
    max_z: Vec<f32>,
    cent_x: Vec<f32>,
    cent_y: Vec<f32>,
    cent_z: Vec<f32>,
}

#[allow(dead_code)]
impl TriangleAABBs {
    fn from_vertices(vertices: &[[f32; 3]], indices: &[u32]) -> Self {
        let n = indices.len() / 3;
        let mut aabbs = Self {
            min_x: vec![0.0; n],
            min_y: vec![0.0; n],
            min_z: vec![0.0; n],
            max_x: vec![0.0; n],
            max_y: vec![0.0; n],
            max_z: vec![0.0; n],
            cent_x: vec![0.0; n],
            cent_y: vec![0.0; n],
            cent_z: vec![0.0; n],
        };
        for t in 0..n {
            let i0 = indices[t * 3] as usize;
            let i1 = indices[t * 3 + 1] as usize;
            let i2 = indices[t * 3 + 2] as usize;
            let a = vertices[i0];
            let b = vertices[i1];
            let c = vertices[i2];
            let mn = vec3_min(vec3_min(a, b), c);
            let mx = vec3_max(vec3_max(a, b), c);
            aabbs.min_x[t] = mn[0];
            aabbs.min_y[t] = mn[1];
            aabbs.min_z[t] = mn[2];
            aabbs.max_x[t] = mx[0];
            aabbs.max_y[t] = mx[1];
            aabbs.max_z[t] = mx[2];
            aabbs.cent_x[t] = (mn[0] + mx[0]) * 0.5;
            aabbs.cent_y[t] = (mn[1] + mx[1]) * 0.5;
            aabbs.cent_z[t] = (mn[2] + mx[2]) * 0.5;
        }
        aabbs
    }

    #[inline]
    fn centroid_axis(&self, axis: usize) -> &[f32] {
        match axis {
            0 => &self.cent_x,
            1 => &self.cent_y,
            _ => &self.cent_z,
        }
    }
}

// ── SAH bin scratch (reused across recursive calls) ────────────────────────

struct SAHBins {
    count: [u32; SAH_BINS],
    min_x: [f32; SAH_BINS],
    min_y: [f32; SAH_BINS],
    min_z: [f32; SAH_BINS],
    max_x: [f32; SAH_BINS],
    max_y: [f32; SAH_BINS],
    max_z: [f32; SAH_BINS],
    left_count: [u32; SAH_BINS],  // SAH_BINS-1 used, but size SAH_BINS avoids const-generic pain
    left_sa: [f32; SAH_BINS],
}

impl SAHBins {
    fn new() -> Self {
        Self {
            count: [0; SAH_BINS],
            min_x: [f32::INFINITY; SAH_BINS],
            min_y: [f32::INFINITY; SAH_BINS],
            min_z: [f32::INFINITY; SAH_BINS],
            max_x: [f32::NEG_INFINITY; SAH_BINS],
            max_y: [f32::NEG_INFINITY; SAH_BINS],
            max_z: [f32::NEG_INFINITY; SAH_BINS],
            left_count: [0; SAH_BINS],
            left_sa: [0.0; SAH_BINS],
        }
    }

    fn clear(&mut self) {
        self.count = [0; SAH_BINS];
        self.min_x = [f32::INFINITY; SAH_BINS];
        self.min_y = [f32::INFINITY; SAH_BINS];
        self.min_z = [f32::INFINITY; SAH_BINS];
        self.max_x = [f32::NEG_INFINITY; SAH_BINS];
        self.max_y = [f32::NEG_INFINITY; SAH_BINS];
        self.max_z = [f32::NEG_INFINITY; SAH_BINS];
        self.left_count = [0; SAH_BINS];
        self.left_sa = [0.0; SAH_BINS];
    }
}

// ── GPU BVH output ────────────────────────────────────────────────────────

/// Handles to GPU buffers produced by `BVHBuilder::upload_to_gpu`.
pub struct GPUBVHData {
    pub triangles_buf: wgpu::Buffer,
    pub bvh4_nodes_buf: wgpu::Buffer,
    pub instances_buf: wgpu::Buffer,
    pub triangle_count: u32,
    pub node_count: u32,
    pub instance_count: u32,
}

// ── Transform helpers ─────────────────────────────────────────────────────

/// Extract 3x4 row-major rows from a column-major Mat4.
///
/// Column-major layout: `d[0..4]=col0, d[4..8]=col1, d[8..12]=col2, d[12..16]=col3`
/// Row-major 3x4: `[[row0], [row1], [row2]]`
fn mat4_to_rows(m: &Mat4) -> [[f32; 4]; 3] {
    let d = m.as_slice();
    [
        [d[0], d[4], d[8], d[12]],  // row 0
        [d[1], d[5], d[9], d[13]],  // row 1
        [d[2], d[6], d[10], d[14]], // row 2
    ]
}

// ── BLAS entry (per-geometry bookkeeping) ──────────────────────────────────

/// Tracks per-geometry triangle and node ranges.
#[derive(Clone, Debug)]
pub struct BLASEntry {
    pub tri_offset: usize,
    pub tri_count: usize,
    /// Binary BVH node offset (temporary, before BVH4 collapse).
    pub binary_node_offset: usize,
    pub binary_node_count: usize,
    /// BVH4 node offset and count (final, after collapse).
    pub bvh4_offset: usize,
    pub bvh4_count: usize,
}

// ── BVHBuilder ─────────────────────────────────────────────────────────────

/// CPU-side BVH builder: SAH binary BVH construction + collapse to BVH4.
///
/// Usage:
/// ```ignore
/// let mut builder = BVHBuilder::new();
/// builder.add_mesh(&vertices, &normals, &indices, 0);
/// builder.build_all_blas();
/// // builder.triangles and builder.bvh4_nodes are ready for GPU upload.
/// ```
pub struct BVHBuilder {
    /// Packed triangle buffer (reordered after build so leaf ranges are contiguous).
    pub triangles: Vec<PackedTriangle>,
    /// Final BVH4 node buffer (all BLASes concatenated).
    pub bvh4_nodes: Vec<BVH4Node>,
    /// Per-geometry BLAS metadata.
    pub blas_entries: Vec<BLASEntry>,
    /// Packed instances for TLAS (populated by `pack_instances`).
    pub packed_instances: Vec<PackedInstance>,
    /// Scene AABB minimum (computed during `pack_instances`).
    pub scene_bounds_min: [f32; 3],
    /// Scene AABB maximum (computed during `pack_instances`).
    pub scene_bounds_max: [f32; 3],

    // ── Temporaries ──
    /// Binary BVH nodes (cleared after collapse).
    binary_nodes: Vec<BinaryBVHNode>,
    /// Per-triangle AABB cache per geometry (only lives during build).
    aabb_cache: Option<TriangleAABBs>,
    /// Working index array for the current geometry build.
    index_buf: Vec<u32>,
    /// Reorder map: new_pos -> old_pos (for triangle reordering).
    reorder_map: Vec<u32>,
    /// SAH bin scratch (reused across recursive calls).
    sah_bins: SAHBins,
    /// Next binary node index (per-geometry, local).
    next_node: usize,
    /// Next triangle slot for leaf packing.
    next_tri_slot: usize,
}

impl BVHBuilder {
    /// Create an empty builder.
    pub fn new() -> Self {
        Self {
            triangles: Vec::new(),
            bvh4_nodes: Vec::new(),
            blas_entries: Vec::new(),
            packed_instances: Vec::new(),
            scene_bounds_min: [f32::INFINITY; 3],
            scene_bounds_max: [f32::NEG_INFINITY; 3],
            binary_nodes: Vec::new(),
            aabb_cache: None,
            index_buf: Vec::new(),
            reorder_map: Vec::new(),
            sah_bins: SAHBins::new(),
            next_node: 0,
            next_tri_slot: 0,
        }
    }

    /// Add a mesh (geometry) to the builder.
    ///
    /// `vertices` and `normals` are per-vertex position/normal arrays indexed by
    /// `indices` (standard triangle-list indexing: every 3 indices = 1 triangle).
    ///
    /// Returns the index of the `BLASEntry` for this mesh.
    pub fn add_mesh(
        &mut self,
        vertices: &[[f32; 3]],
        normals: &[[f32; 3]],
        indices: &[u32],
        material_index: u32,
    ) -> usize {
        assert!(indices.len() % 3 == 0, "index count must be a multiple of 3");
        let tri_count = indices.len() / 3;
        let tri_offset = self.triangles.len();

        for t in 0..tri_count {
            let i0 = indices[t * 3] as usize;
            let i1 = indices[t * 3 + 1] as usize;
            let i2 = indices[t * 3 + 2] as usize;
            self.triangles.push(PackedTriangle {
                v0: vertices[i0],
                v1: vertices[i1],
                v2: vertices[i2],
                n0: normals[i0],
                n1: normals[i1],
                n2: normals[i2],
                material_index,
                _pad: [0; 5],
            });
        }

        let entry_idx = self.blas_entries.len();
        self.blas_entries.push(BLASEntry {
            tri_offset,
            tri_count,
            binary_node_offset: 0,
            binary_node_count: 0,
            bvh4_offset: 0,
            bvh4_count: 0,
        });
        entry_idx
    }

    /// Build SAH BVH for all meshes and collapse to BVH4.
    pub fn build_all_blas(&mut self) {
        // Phase 1: Build binary SAH trees for each geometry.
        let entries: Vec<(usize, usize)> = self
            .blas_entries
            .iter()
            .map(|e| (e.tri_offset, e.tri_count))
            .collect();

        let mut global_binary_offset: usize = 0;

        for (entry_idx, &(tri_offset, tri_count)) in entries.iter().enumerate() {
            if tri_count == 0 {
                continue;
            }

            // Pre-compute per-triangle AABBs from the already-packed triangles.
            self.precompute_aabbs(tri_offset, tri_count);

            // Allocate working arrays.
            self.index_buf.clear();
            self.index_buf.extend(0..tri_count as u32);
            self.reorder_map.clear();
            self.reorder_map.resize(tri_count, 0);
            self.next_node = 0;
            self.next_tri_slot = 0;

            // Reserve worst-case binary nodes for this geometry.
            let max_nodes = 2 * tri_count - 1;
            let binary_start = self.binary_nodes.len();
            self.binary_nodes.resize(
                binary_start + max_nodes,
                BinaryBVHNode {
                    bounds_min: [0.0; 3],
                    bounds_max: [0.0; 3],
                    left: 0,
                    right: 0,
                },
            );

            // Build binary SAH tree (recursive).
            self.build_sah(binary_start, 0, tri_count);

            let node_count = self.next_node;
            self.blas_entries[entry_idx].binary_node_offset = global_binary_offset;
            self.blas_entries[entry_idx].binary_node_count = node_count;
            global_binary_offset += node_count;

            // Shrink binary_nodes to actual usage.
            self.binary_nodes.truncate(binary_start + node_count);

            // Reorder triangles so leaf ranges are contiguous.
            self.reorder_triangles(tri_offset, tri_count);

            self.aabb_cache = None;
        }

        // Phase 2: Collapse all binary BLASes to BVH4.
        self.collapse_all_to_bvh4();
    }

    // ── Internal: pre-compute AABBs from packed triangles ──────────────────

    fn precompute_aabbs(&mut self, tri_offset: usize, tri_count: usize) {
        let mut aabbs = TriangleAABBs {
            min_x: vec![0.0; tri_count],
            min_y: vec![0.0; tri_count],
            min_z: vec![0.0; tri_count],
            max_x: vec![0.0; tri_count],
            max_y: vec![0.0; tri_count],
            max_z: vec![0.0; tri_count],
            cent_x: vec![0.0; tri_count],
            cent_y: vec![0.0; tri_count],
            cent_z: vec![0.0; tri_count],
        };
        for t in 0..tri_count {
            let tri = &self.triangles[tri_offset + t];
            let mn = vec3_min(vec3_min(tri.v0, tri.v1), tri.v2);
            let mx = vec3_max(vec3_max(tri.v0, tri.v1), tri.v2);
            aabbs.min_x[t] = mn[0];
            aabbs.min_y[t] = mn[1];
            aabbs.min_z[t] = mn[2];
            aabbs.max_x[t] = mx[0];
            aabbs.max_y[t] = mx[1];
            aabbs.max_z[t] = mx[2];
            aabbs.cent_x[t] = (mn[0] + mx[0]) * 0.5;
            aabbs.cent_y[t] = (mn[1] + mx[1]) * 0.5;
            aabbs.cent_z[t] = (mn[2] + mx[2]) * 0.5;
        }
        self.aabb_cache = Some(aabbs);
    }

    // ── Internal: recursive SAH build ──────────────────────────────────────

    /// Build a binary BVH node for triangles index_buf[start..start+count].
    /// `binary_base` is the offset into self.binary_nodes for this geometry.
    /// Returns the local node index (relative to binary_base).
    fn build_sah(&mut self, binary_base: usize, start: usize, count: usize) -> usize {
        let node_local = self.next_node;
        self.next_node += 1;
        let node_global = binary_base + node_local;

        // Compute bounds of all triangles in this subtree.
        let aabbs = self.aabb_cache.as_ref().unwrap();
        let mut b_min = [f32::INFINITY; 3];
        let mut b_max = [f32::NEG_INFINITY; 3];
        for i in start..start + count {
            let t = self.index_buf[i] as usize;
            b_min[0] = b_min[0].min(aabbs.min_x[t]);
            b_min[1] = b_min[1].min(aabbs.min_y[t]);
            b_min[2] = b_min[2].min(aabbs.min_z[t]);
            b_max[0] = b_max[0].max(aabbs.max_x[t]);
            b_max[1] = b_max[1].max(aabbs.max_y[t]);
            b_max[2] = b_max[2].max(aabbs.max_z[t]);
        }

        // Small enough for a leaf?
        if count <= MAX_LEAF_SIZE {
            let leaf_tri_start = self.next_tri_slot;
            for i in start..start + count {
                self.reorder_map[self.next_tri_slot] = self.index_buf[i];
                self.next_tri_slot += 1;
            }
            self.binary_nodes[node_global] = BinaryBVHNode {
                bounds_min: b_min,
                bounds_max: b_max,
                left: -((leaf_tri_start as i32) + 1),
                right: count as i32,
            };
            return node_local;
        }

        // Find best SAH split across 3 axes using binned approach.
        let parent_sa = aabb_surface_area(b_min, b_max);
        let leaf_cost = INTERSECTION_COST * count as f32;
        let mut best_cost = leaf_cost;
        let mut best_axis: i32 = -1;
        let mut best_split: usize = 0;

        // Cache per-axis centroid ranges.
        let mut axis_c_min = [0.0f32; 3];
        let mut axis_c_max = [0.0f32; 3];

        for axis in 0..3 {
            let cent = match axis {
                0 => &aabbs.cent_x,
                1 => &aabbs.cent_y,
                _ => &aabbs.cent_z,
            };

            let mut c_min = f32::INFINITY;
            let mut c_max = f32::NEG_INFINITY;
            for i in start..start + count {
                let c = cent[self.index_buf[i] as usize];
                c_min = c_min.min(c);
                c_max = c_max.max(c);
            }
            axis_c_min[axis] = c_min;
            axis_c_max[axis] = c_max;

            if c_max - c_min < 1e-10 {
                continue;
            }

            let scale = SAH_BINS as f32 / (c_max - c_min);

            // Clear bins.
            self.sah_bins.clear();

            // Fill bins.
            for i in start..start + count {
                let t = self.index_buf[i] as usize;
                let mut b = ((cent[t] - c_min) * scale) as usize;
                if b >= SAH_BINS {
                    b = SAH_BINS - 1;
                }
                self.sah_bins.count[b] += 1;
                self.sah_bins.min_x[b] = self.sah_bins.min_x[b].min(aabbs.min_x[t]);
                self.sah_bins.min_y[b] = self.sah_bins.min_y[b].min(aabbs.min_y[t]);
                self.sah_bins.min_z[b] = self.sah_bins.min_z[b].min(aabbs.min_z[t]);
                self.sah_bins.max_x[b] = self.sah_bins.max_x[b].max(aabbs.max_x[t]);
                self.sah_bins.max_y[b] = self.sah_bins.max_y[b].max(aabbs.max_y[t]);
                self.sah_bins.max_z[b] = self.sah_bins.max_z[b].max(aabbs.max_z[t]);
            }

            // Sweep left-to-right: accumulate bounds + counts.
            let mut l_min = [f32::INFINITY; 3];
            let mut l_max = [f32::NEG_INFINITY; 3];
            let mut l_count: u32 = 0;
            for i in 0..SAH_BINS - 1 {
                l_min[0] = l_min[0].min(self.sah_bins.min_x[i]);
                l_min[1] = l_min[1].min(self.sah_bins.min_y[i]);
                l_min[2] = l_min[2].min(self.sah_bins.min_z[i]);
                l_max[0] = l_max[0].max(self.sah_bins.max_x[i]);
                l_max[1] = l_max[1].max(self.sah_bins.max_y[i]);
                l_max[2] = l_max[2].max(self.sah_bins.max_z[i]);
                l_count += self.sah_bins.count[i];
                self.sah_bins.left_count[i] = l_count;
                self.sah_bins.left_sa[i] = if l_count > 0 {
                    aabb_surface_area(l_min, l_max)
                } else {
                    0.0
                };
            }

            // Sweep right-to-left: evaluate SAH cost at each split.
            let mut r_min = [f32::INFINITY; 3];
            let mut r_max = [f32::NEG_INFINITY; 3];
            let mut r_count: u32 = 0;
            for i in (1..SAH_BINS).rev() {
                r_min[0] = r_min[0].min(self.sah_bins.min_x[i]);
                r_min[1] = r_min[1].min(self.sah_bins.min_y[i]);
                r_min[2] = r_min[2].min(self.sah_bins.min_z[i]);
                r_max[0] = r_max[0].max(self.sah_bins.max_x[i]);
                r_max[1] = r_max[1].max(self.sah_bins.max_y[i]);
                r_max[2] = r_max[2].max(self.sah_bins.max_z[i]);
                r_count += self.sah_bins.count[i];

                let right_sa = if r_count > 0 {
                    aabb_surface_area(r_min, r_max)
                } else {
                    0.0
                };

                let cost = TRAVERSAL_COST
                    + INTERSECTION_COST
                        * (self.sah_bins.left_count[i - 1] as f32 * self.sah_bins.left_sa[i - 1]
                            + r_count as f32 * right_sa)
                        / parent_sa;

                if cost < best_cost {
                    best_cost = cost;
                    best_axis = axis as i32;
                    best_split = i;
                }
            }
        }

        // No good SAH split found — fallback to longest-axis midpoint.
        if best_axis == -1 {
            let dx = b_max[0] - b_min[0];
            let dy = b_max[1] - b_min[1];
            let dz = b_max[2] - b_min[2];
            let axis = if dx >= dy && dx >= dz {
                0
            } else if dy >= dz {
                1
            } else {
                2
            };
            let cent = match axis {
                0 => &aabbs.cent_x,
                1 => &aabbs.cent_y,
                _ => &aabbs.cent_z,
            };
            let mid = (axis_c_min[axis] + axis_c_max[axis]) * 0.5;

            let mut l = start;
            let mut r = start + count - 1;
            while l <= r {
                if cent[self.index_buf[l] as usize] < mid {
                    l += 1;
                } else {
                    self.index_buf.swap(l, r);
                    if r == 0 {
                        break;
                    }
                    r -= 1;
                }
            }
            let mut left_count = l - start;
            if left_count == 0 || left_count == count {
                left_count = count >> 1;
            }

            let left_child = self.build_sah(binary_base, start, left_count);
            let right_child = self.build_sah(binary_base, start + left_count, count - left_count);
            self.binary_nodes[node_global] = BinaryBVHNode {
                bounds_min: b_min,
                bounds_max: b_max,
                left: left_child as i32,
                right: right_child as i32,
            };
            return node_local;
        }

        // Partition indices by best SAH bin split.
        {
            let aabbs = self.aabb_cache.as_ref().unwrap();
            let cent = match best_axis as usize {
                0 => &aabbs.cent_x,
                1 => &aabbs.cent_y,
                _ => &aabbs.cent_z,
            };
            let c_min = axis_c_min[best_axis as usize];
            let c_max = axis_c_max[best_axis as usize];
            let scale = SAH_BINS as f32 / (c_max - c_min);

            let mut l = start;
            let mut r = start + count - 1;
            while l <= r {
                let mut b = ((cent[self.index_buf[l] as usize] - c_min) * scale) as usize;
                if b >= SAH_BINS {
                    b = SAH_BINS - 1;
                }
                if b < best_split {
                    l += 1;
                } else {
                    self.index_buf.swap(l, r);
                    if r == 0 {
                        break;
                    }
                    r -= 1;
                }
            }
            let mut left_count = l - start;
            if left_count == 0 || left_count == count {
                left_count = count >> 1;
            }

            let left_child = self.build_sah(binary_base, start, left_count);
            let right_child =
                self.build_sah(binary_base, start + left_count, count - left_count);
            self.binary_nodes[node_global] = BinaryBVHNode {
                bounds_min: b_min,
                bounds_max: b_max,
                left: left_child as i32,
                right: right_child as i32,
            };
        }

        node_local
    }

    // ── Internal: triangle reordering ──────────────────────────────────────

    fn reorder_triangles(&mut self, tri_offset: usize, tri_count: usize) {
        let mut reordered = Vec::with_capacity(tri_count);
        for i in 0..tri_count {
            let old_idx = self.reorder_map[i] as usize;
            reordered.push(self.triangles[tri_offset + old_idx]);
        }
        for i in 0..tri_count {
            self.triangles[tri_offset + i] = reordered[i];
        }
    }

    // ── Internal: BVH4 collapse ────────────────────────────────────────────

    fn collapse_all_to_bvh4(&mut self) {
        self.bvh4_nodes.clear();

        let entries: Vec<(usize, usize, usize)> = self
            .blas_entries
            .iter()
            .map(|e| (e.binary_node_offset, e.binary_node_count, e.tri_offset))
            .collect();

        for (entry_idx, &(binary_offset, binary_count, _tri_offset)) in entries.iter().enumerate() {
            if binary_count == 0 {
                continue;
            }

            let bvh4_start = self.bvh4_nodes.len();
            self.collapse_node(binary_offset, 0);

            let bvh4_count = self.bvh4_nodes.len() - bvh4_start;
            self.blas_entries[entry_idx].bvh4_offset = bvh4_start;
            self.blas_entries[entry_idx].bvh4_count = bvh4_count;
        }
    }

    /// Collapse a binary BVH subtree rooted at `binary_base + local_idx` into
    /// BVH4 nodes. Returns the global BVH4 node index.
    fn collapse_node(&mut self, binary_base: usize, local_idx: usize) -> usize {
        let bvh4_idx = self.bvh4_nodes.len();
        self.bvh4_nodes.push(BVH4Node::sentinel());

        let node = self.binary_nodes[binary_base + local_idx];

        if node.is_leaf() {
            // Single-child BVH4 node wrapping the leaf.
            self.set_child_leaf(bvh4_idx, 0, &node);
            return bvh4_idx;
        }

        // Collect grandchildren by expanding one level of the binary tree.
        // Each of the two children contributes 1 (if leaf) or 2 (if internal) grandchildren.
        let mut grandchildren: Vec<(usize, bool)> = Vec::with_capacity(4);
        let left_idx = node.left as usize;
        let right_idx = node.right as usize;

        let left_node = self.binary_nodes[binary_base + left_idx];
        if left_node.is_leaf() {
            grandchildren.push((left_idx, true));
        } else {
            grandchildren.push((left_node.left as usize, false));
            grandchildren.push((left_node.right as usize, false));
        }

        let right_node = self.binary_nodes[binary_base + right_idx];
        if right_node.is_leaf() {
            grandchildren.push((right_idx, true));
        } else {
            grandchildren.push((right_node.left as usize, false));
            grandchildren.push((right_node.right as usize, false));
        }

        for (slot, &(gc_local_idx, is_parent_leaf)) in grandchildren.iter().enumerate() {
            if slot >= 4 {
                break;
            }
            let gc_node = self.binary_nodes[binary_base + gc_local_idx];
            if is_parent_leaf || gc_node.is_leaf() {
                // This grandchild is a leaf (or its parent was already a leaf).
                let leaf_node = if is_parent_leaf {
                    self.binary_nodes[binary_base + gc_local_idx]
                } else {
                    gc_node
                };
                self.set_child_leaf(bvh4_idx, slot, &leaf_node);
            } else {
                // Internal grandchild: recurse to create a new BVH4 node.
                let child_bvh4 = self.collapse_node(binary_base, gc_local_idx);
                self.set_child_internal(bvh4_idx, slot, &gc_node, child_bvh4);
            }
        }

        bvh4_idx
    }

    /// Set a BVH4 child slot to a leaf.
    fn set_child_leaf(&mut self, bvh4_idx: usize, slot: usize, node: &BinaryBVHNode) {
        let n = &mut self.bvh4_nodes[bvh4_idx];
        n.child_min_x[slot] = node.bounds_min[0];
        n.child_min_y[slot] = node.bounds_min[1];
        n.child_min_z[slot] = node.bounds_min[2];
        n.child_max_x[slot] = node.bounds_max[0];
        n.child_max_y[slot] = node.bounds_max[1];
        n.child_max_z[slot] = node.bounds_max[2];
        n.children[slot] = node.left; // -(tri_start + 1)
        n.tri_counts[slot] = node.right as u32; // tri_count
    }

    /// Set a BVH4 child slot to an internal BVH4 node reference.
    fn set_child_internal(
        &mut self,
        bvh4_idx: usize,
        slot: usize,
        node: &BinaryBVHNode,
        child_bvh4_idx: usize,
    ) {
        let n = &mut self.bvh4_nodes[bvh4_idx];
        n.child_min_x[slot] = node.bounds_min[0];
        n.child_min_y[slot] = node.bounds_min[1];
        n.child_min_z[slot] = node.bounds_min[2];
        n.child_max_x[slot] = node.bounds_max[0];
        n.child_max_y[slot] = node.bounds_max[1];
        n.child_max_z[slot] = node.bounds_max[2];
        n.children[slot] = child_bvh4_idx as i32;
        n.tri_counts[slot] = 0;
    }

    // ── Scene integration ─────────────────────────────────────────────────

    /// Pack triangles from all scene renderables into this builder.
    ///
    /// Iterates `scene.ordered_indices()`, extracts position/normal data from
    /// each renderable's `Geometry`, and calls `add_mesh()` for each.
    pub fn pack_scene(&mut self, scene: &Scene) {
        for idx in scene.ordered_indices() {
            let renderable = match scene.get_renderable(idx) {
                Some(r) => r,
                None => continue,
            };

            let geo = &renderable.geometry;
            if geo.indices.is_empty() || geo.vertices.is_empty() {
                continue;
            }

            // Extract position [f32;3] and normal [f32;3] from Vertex
            let positions: Vec<[f32; 3]> = geo
                .vertices
                .iter()
                .map(|v| [v.position[0], v.position[1], v.position[2]])
                .collect();
            let normals: Vec<[f32; 3]> = geo
                .vertices
                .iter()
                .map(|v| v.normal)
                .collect();

            // Material index: use the blas_entries index as a stand-in for now.
            // Each renderable gets its own BLAS entry; material_index can be refined later.
            let material_index = self.blas_entries.len() as u32;

            self.add_mesh(&positions, &normals, &geo.indices, material_index);
        }
    }

    /// Pack instance data for each renderable in the scene.
    ///
    /// Must be called **after** `build_all_blas()` (which includes BVH4 collapse)
    /// so that `blas_entries` have valid `bvh4_offset`, `bvh4_count`, `tri_offset`,
    /// and `tri_count`.
    pub fn pack_instances(&mut self, scene: &Scene) {
        self.packed_instances.clear();
        self.scene_bounds_min = [f32::INFINITY; 3];
        self.scene_bounds_max = [f32::NEG_INFINITY; 3];

        let mut entry_idx = 0usize;
        for idx in scene.ordered_indices() {
            let renderable = match scene.get_renderable(idx) {
                Some(r) => r,
                None => continue,
            };

            let geo = &renderable.geometry;
            if geo.indices.is_empty() || geo.vertices.is_empty() {
                continue;
            }

            if entry_idx >= self.blas_entries.len() {
                break;
            }
            let entry = &self.blas_entries[entry_idx];

            let world = &renderable.object.world_matrix;
            let inv_world = world.inverse();

            let transform = mat4_to_rows(world);
            let inv_transform = mat4_to_rows(&inv_world);

            self.packed_instances.push(PackedInstance {
                transform,
                inv_transform,
                blas_node_offset: entry.bvh4_offset as u32,
                blas_tri_offset: entry.tri_offset as u32,
                blas_tri_count: entry.tri_count as u32,
                material_index: entry_idx as u32,
            });

            // Expand scene AABB by transforming the BLAS root AABB corners to world space.
            // The BLAS root is at bvh4_offset; its child bounds define the object-space AABB.
            if entry.bvh4_count > 0 {
                let root = &self.bvh4_nodes[entry.bvh4_offset];
                // Compute the object-space AABB from all used child slots of the root node.
                let mut obj_min = [f32::INFINITY; 3];
                let mut obj_max = [f32::NEG_INFINITY; 3];
                for slot in 0..4 {
                    if root.children[slot] == -1 && root.tri_counts[slot] == 0 {
                        continue; // sentinel
                    }
                    obj_min[0] = obj_min[0].min(root.child_min_x[slot]);
                    obj_min[1] = obj_min[1].min(root.child_min_y[slot]);
                    obj_min[2] = obj_min[2].min(root.child_min_z[slot]);
                    obj_max[0] = obj_max[0].max(root.child_max_x[slot]);
                    obj_max[1] = obj_max[1].max(root.child_max_y[slot]);
                    obj_max[2] = obj_max[2].max(root.child_max_z[slot]);
                }

                // Transform the 8 AABB corners to world space and expand scene bounds.
                for cx in 0..2 {
                    for cy in 0..2 {
                        for cz in 0..2 {
                            let corner = crate::math::Vec3::new(
                                if cx == 0 { obj_min[0] } else { obj_max[0] },
                                if cy == 0 { obj_min[1] } else { obj_max[1] },
                                if cz == 0 { obj_min[2] } else { obj_max[2] },
                            );
                            let wc = world.transform_point(&corner);
                            self.scene_bounds_min[0] = self.scene_bounds_min[0].min(wc.x);
                            self.scene_bounds_min[1] = self.scene_bounds_min[1].min(wc.y);
                            self.scene_bounds_min[2] = self.scene_bounds_min[2].min(wc.z);
                            self.scene_bounds_max[0] = self.scene_bounds_max[0].max(wc.x);
                            self.scene_bounds_max[1] = self.scene_bounds_max[1].max(wc.y);
                            self.scene_bounds_max[2] = self.scene_bounds_max[2].max(wc.z);
                        }
                    }
                }
            }

            entry_idx += 1;
        }
    }

    /// Upload triangles, BVH4 nodes, and instances to GPU buffers.
    ///
    /// Must be called after `build_all_blas()` and `pack_instances()`.
    pub fn upload_to_gpu(&self, renderer: &Renderer) -> GPUBVHData {
        let device = renderer.device();

        let triangles_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BVH/Triangles"),
            contents: bytemuck::cast_slice(&self.triangles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let bvh4_nodes_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BVH/BVH4Nodes"),
            contents: bytemuck::cast_slice(&self.bvh4_nodes),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let instances_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("BVH/Instances"),
            contents: bytemuck::cast_slice(&self.packed_instances),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        GPUBVHData {
            triangles_buf,
            bvh4_nodes_buf,
            instances_buf,
            triangle_count: self.triangles.len() as u32,
            node_count: self.bvh4_nodes.len() as u32,
            instance_count: self.packed_instances.len() as u32,
        }
    }

    /// Orchestrate the full BVH build pipeline:
    /// 1. `pack_scene` — pack triangles from scene renderables
    /// 2. `build_all_blas` — CPU SAH build + BVH4 collapse
    /// 3. `pack_instances` — create instance data with transforms + BLAS offsets
    /// 4. `upload_to_gpu` — create GPU buffers for triangles, nodes, instances
    /// 5. `tlas.build` — GPU Morton sort + BVH4 TLAS construction
    ///
    /// Returns GPU buffer handles for use in the trace shader.
    pub fn build_full(
        &mut self,
        renderer: &Renderer,
        scene: &Scene,
        tlas: &mut TLASBuilder,
    ) -> GPUBVHData {
        // Step 1: Pack triangles from scene
        self.pack_scene(scene);

        // Step 2: CPU SAH build + BVH4 collapse
        self.build_all_blas();

        // Step 3: Pack instance data
        self.pack_instances(scene);

        // Step 4: Upload to GPU
        let gpu_data = self.upload_to_gpu(renderer);

        // Step 5: Build TLAS on GPU (sort + BVH4)
        if gpu_data.instance_count > 0 {
            tlas.build(
                renderer,
                &gpu_data.instances_buf,
                &gpu_data.bvh4_nodes_buf,
                gpu_data.instance_count,
                self.scene_bounds_min,
                self.scene_bounds_max,
            );
        }

        gpu_data
    }
}

impl Default for BVHBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a simple axis-aligned triangle for testing.
    fn make_quad_mesh() -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        // Two triangles forming a 1x1 quad on the XY plane.
        let vertices = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let normals = vec![
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ];
        let indices = vec![0, 1, 2, 0, 2, 3];
        (vertices, normals, indices)
    }

    /// Create a mesh with many triangles spread along X axis for SAH testing.
    fn make_line_mesh(count: usize) -> (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<u32>) {
        let mut vertices = Vec::new();
        let mut normals = Vec::new();
        let mut indices = Vec::new();
        for i in 0..count {
            let x = i as f32;
            let base = (i * 3) as u32;
            vertices.push([x, 0.0, 0.0]);
            vertices.push([x + 0.5, 1.0, 0.0]);
            vertices.push([x + 1.0, 0.0, 0.0]);
            normals.push([0.0, 0.0, 1.0]);
            normals.push([0.0, 0.0, 1.0]);
            normals.push([0.0, 0.0, 1.0]);
            indices.push(base);
            indices.push(base + 1);
            indices.push(base + 2);
        }
        (vertices, normals, indices)
    }

    #[test]
    fn test_packed_triangle_size() {
        assert_eq!(std::mem::size_of::<PackedTriangle>(), 96);
    }

    #[test]
    fn test_bvh4_node_size() {
        assert_eq!(std::mem::size_of::<BVH4Node>(), 128);
    }

    #[test]
    fn test_binary_bvh_node_leaf_convention() {
        let leaf = BinaryBVHNode {
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
            left: -1, // -(0 + 1)
            right: 3,
        };
        assert!(leaf.is_leaf());
        assert_eq!(leaf.tri_start(), 0);
        assert_eq!(leaf.tri_count(), 3);

        let internal = BinaryBVHNode {
            bounds_min: [0.0; 3],
            bounds_max: [1.0; 3],
            left: 1,
            right: 2,
        };
        assert!(!internal.is_leaf());
    }

    #[test]
    fn test_add_mesh() {
        let (verts, normals, indices) = make_quad_mesh();
        let mut builder = BVHBuilder::new();
        let entry_idx = builder.add_mesh(&verts, &normals, &indices, 0);
        assert_eq!(entry_idx, 0);
        assert_eq!(builder.triangles.len(), 2);
        assert_eq!(builder.blas_entries[0].tri_count, 2);
    }

    #[test]
    fn test_build_small_mesh() {
        let (verts, normals, indices) = make_quad_mesh();
        let mut builder = BVHBuilder::new();
        builder.add_mesh(&verts, &normals, &indices, 0);
        builder.build_all_blas();

        // 2 triangles <= MAX_LEAF_SIZE => single leaf => single BVH4 node.
        assert_eq!(builder.bvh4_nodes.len(), 1);
        let node = &builder.bvh4_nodes[0];
        // First child slot should be a leaf (children[0] < 0).
        assert!(node.children[0] < 0);
        assert!(node.tri_counts[0] > 0);
        // Remaining slots should be sentinels.
        assert_eq!(node.children[1], -1);
        assert_eq!(node.tri_counts[1], 0);
    }

    #[test]
    fn test_build_many_triangles() {
        let (verts, normals, indices) = make_line_mesh(100);
        let mut builder = BVHBuilder::new();
        builder.add_mesh(&verts, &normals, &indices, 0);
        builder.build_all_blas();

        // Should produce multiple BVH4 nodes.
        assert!(builder.bvh4_nodes.len() > 1);
        // All triangles should still be present.
        assert_eq!(builder.triangles.len(), 100);
        // BVH4 root should be valid.
        let root = &builder.bvh4_nodes[builder.blas_entries[0].bvh4_offset];
        // At least one child slot should be used.
        let used = (0..4).filter(|&i| root.children[i] != -1 || root.tri_counts[i] != 0).count();
        assert!(used >= 2);
    }

    #[test]
    fn test_multiple_meshes() {
        let (v1, n1, i1) = make_quad_mesh();
        let (v2, n2, i2) = make_line_mesh(20);
        let mut builder = BVHBuilder::new();
        builder.add_mesh(&v1, &n1, &i1, 0);
        builder.add_mesh(&v2, &n2, &i2, 1);
        builder.build_all_blas();

        assert_eq!(builder.blas_entries.len(), 2);
        assert_eq!(builder.triangles.len(), 22); // 2 + 20
        // Each BLAS should have its own BVH4 region.
        assert!(builder.blas_entries[0].bvh4_count > 0);
        assert!(builder.blas_entries[1].bvh4_count > 0);
        // Non-overlapping BVH4 ranges.
        let end0 = builder.blas_entries[0].bvh4_offset + builder.blas_entries[0].bvh4_count;
        assert!(end0 <= builder.blas_entries[1].bvh4_offset);
    }

    #[test]
    fn test_aabb_surface_area() {
        // Unit cube: 6 faces of area 1 each = 6.0.
        let sa = aabb_surface_area([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((sa - 6.0).abs() < 1e-6);

        // Degenerate (flat): 2 * (2*0 + 0*3 + 3*2) = 12.
        let sa2 = aabb_surface_area([0.0, 0.0, 0.0], [2.0, 0.0, 3.0]);
        assert!((sa2 - 12.0).abs() < 1e-6);
    }
}
