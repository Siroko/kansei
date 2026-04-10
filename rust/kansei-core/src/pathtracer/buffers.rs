use bytemuck::{Pod, Zeroable};

/// Packed triangle for GPU BVH traversal. 24 floats = 96 bytes.
///
/// Layout matches the WGSL struct:
///   v0.xyz(3) v1.xyz(3) v2.xyz(3) n0.xyz(3) n1.xyz(3) n2.xyz(3) matIdx(1) pad(5)
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PackedTriangle {
    pub v0: [f32; 3],
    pub v1: [f32; 3],
    pub v2: [f32; 3],
    pub n0: [f32; 3],
    pub n1: [f32; 3],
    pub n2: [f32; 3],
    pub material_index: u32,
    pub _pad: [u32; 5],
}

/// BVH4 node (4-wide). 32 floats = 128 bytes = 8 vec4f.
///
/// Layout:
///   [0-3]   child_min_x[0..3]   [4-7]   child_max_x[0..3]
///   [8-11]  child_min_y[0..3]   [12-15] child_max_y[0..3]
///   [16-19] child_min_z[0..3]   [20-23] child_max_z[0..3]
///   [24-27] children[0..3] (i32: <0 = leaf -(tri_start+1), >=0 = BVH4 node index)
///   [28-31] tri_counts[0..3] (u32: triangle count for leaves, 0 for internal)
///
/// Sentinel (unused child slot): children[i] = -1, tri_counts[i] = 0
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct BVH4Node {
    pub child_min_x: [f32; 4],
    pub child_max_x: [f32; 4],
    pub child_min_y: [f32; 4],
    pub child_max_y: [f32; 4],
    pub child_min_z: [f32; 4],
    pub child_max_z: [f32; 4],
    pub children: [i32; 4],
    pub tri_counts: [u32; 4],
}

impl BVH4Node {
    /// Create a sentinel node with all child slots empty.
    pub fn sentinel() -> Self {
        Self {
            child_min_x: [1e30; 4],
            child_max_x: [-1e30; 4],
            child_min_y: [1e30; 4],
            child_max_y: [-1e30; 4],
            child_min_z: [1e30; 4],
            child_max_z: [-1e30; 4],
            children: [-1; 4],
            tri_counts: [0; 4],
        }
    }
}

/// Packed instance for TLAS. 28 floats = 112 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PackedInstance {
    /// 3x4 row-major transform (object-to-world).
    pub transform: [[f32; 4]; 3],
    /// 3x4 row-major inverse transform (world-to-object).
    pub inv_transform: [[f32; 4]; 3],
    pub blas_node_offset: u32,
    pub blas_tri_offset: u32,
    pub blas_tri_count: u32,
    pub material_index: u32,
}

/// Binary BVH node used during CPU SAH construction (intermediate form before
/// collapse to BVH4). Not uploaded to GPU.
///
/// Convention:
///   - Leaf:     `left` = -(tri_start + 1) (negative), `right` = tri_count
///   - Internal: `left` = left child index (>= 0),     `right` = right child index (>= 0)
#[derive(Clone, Copy, Debug)]
pub struct BinaryBVHNode {
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
    pub left: i32,
    pub right: i32,
}

impl BinaryBVHNode {
    /// Returns true if this node is a leaf (left < 0 encodes -(tri_start+1)).
    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.left < 0
    }

    /// For a leaf node, return the triangle start index.
    #[inline]
    pub fn tri_start(&self) -> usize {
        debug_assert!(self.is_leaf());
        (-(self.left + 1)) as usize
    }

    /// For a leaf node, return the triangle count.
    #[inline]
    pub fn tri_count(&self) -> usize {
        debug_assert!(self.is_leaf());
        self.right as usize
    }
}
