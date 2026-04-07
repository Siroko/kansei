use bytemuck::{Pod, Zeroable};

use crate::renderers::Renderer;
use super::radix_sort::RadixSort;

// ── Uniform structs ──────────────────────────────────────────────────────────

/// Parameters for the TLAS sort shader (Morton code computation).
/// Matches `SortParams` in tlas-sort.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TlasSortParams {
    count: u32,
    scene_min_x: f32,
    scene_min_y: f32,
    scene_min_z: f32,
    scene_ext_x: f32, // 1.0 / (max - min) per axis
    scene_ext_y: f32,
    scene_ext_z: f32,
    _pad: u32,
}

/// Parameters for the TLAS build shader (leaf + internal node construction).
/// Matches `BuildParams` in tlas-build.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TlasBuildParams {
    instance_count: u32,
    node_offset: u32,
    child_offset: u32,
    node_count: u32,
    child_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ── Constants ────────────────────────────────────────────────────────────────

const WG_SIZE: u32 = 256;

/// Size of a BVH4 node in vec4f units (8 vec4f = 128 bytes).
const NODE_VEC4_STRIDE: u32 = 8;

// ── TLAS tree layout helpers ─────────────────────────────────────────────────

/// Compute the tree layout for a BVH4 TLAS built bottom-up from `n` instances.
///
/// Returns a vec of (node_offset, node_count) pairs, from leaves (last entry)
/// up to root (first entry). Also returns total_nodes.
fn compute_tree_levels(instance_count: u32) -> (Vec<(u32, u32)>, u32) {
    if instance_count == 0 {
        return (vec![], 0);
    }

    // Leaves: ceil(N / 4) nodes, each grouping up to 4 instances
    let mut levels = Vec::new();
    let leaf_count = (instance_count + 3) / 4;
    levels.push(leaf_count);

    // Internal levels: keep grouping by 4 until we reach a single root
    let mut child_count = leaf_count;
    while child_count > 1 {
        let parent_count = (child_count + 3) / 4;
        levels.push(parent_count);
        child_count = parent_count;
    }

    // Total nodes
    let total_nodes: u32 = levels.iter().sum();

    // Compute offsets: root at index 0, leaves at end.
    // levels[last] = root (1 node at offset 0), levels[0] = leaves at the end.
    let mut result = Vec::with_capacity(levels.len());
    let mut offset = 0u32;
    for &count in levels.iter().rev() {
        result.push((offset, count));
        offset += count;
    }
    // result is now [root, ..., leaves] — reverse to get [leaves, ..., root]
    result.reverse();

    (result, total_nodes)
}

// ── TLASBuilder ──────────────────────────────────────────────────────────────

/// GPU TLAS builder: Morton code generation + radix sort + BVH4 construction.
///
/// Orchestrates the full pipeline each frame:
/// 1. Compute Morton codes from instance centroids
/// 2. Radix sort instances by Morton code
/// 3. Build BVH4 leaf nodes (groups of 4 sorted instances)
/// 4. Build BVH4 internal nodes bottom-up (groups of 4 child nodes)
pub struct TLASBuilder {
    // Morton code computation
    morton_pipeline: wgpu::ComputePipeline,
    morton_bgl: wgpu::BindGroupLayout,
    // TLAS BVH4 build (shared layout for both leaf + internal)
    leaf_pipeline: wgpu::ComputePipeline,
    internal_pipeline: wgpu::ComputePipeline,
    build_bgl: wgpu::BindGroupLayout,
    // Buffers
    morton_keys_buf: Option<wgpu::Buffer>,
    morton_vals_buf: Option<wgpu::Buffer>,
    sort_params_buf: Option<wgpu::Buffer>,
    /// The TLAS BVH4 node buffer (output). Bound at binding 7 in trace shader.
    pub tlas_nodes_buf: Option<wgpu::Buffer>,
    /// Per-level build parameter buffers (one per internal level + one for leaves).
    build_params_bufs: Vec<wgpu::Buffer>,
    // Radix sort
    sort: RadixSort,
    // State
    capacity: u32,
}

impl TLASBuilder {
    /// Create a new TLASBuilder, compiling all compute pipelines.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        // ── Morton code pipeline ─────────────────────────────────────────
        let morton_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TLASBuilder/MortonShader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/tlas-sort.wgsl").into(),
            ),
        });

        let morton_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TLASBuilder/MortonBGL"),
            entries: &[
                // binding 0: instances (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: morton_keys (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: morton_vals (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let morton_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TLASBuilder/MortonPipelineLayout"),
                bind_group_layouts: &[&morton_bgl],
                push_constant_ranges: &[],
            });

        let morton_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TLASBuilder/MortonPipeline"),
                layout: Some(&morton_pipeline_layout),
                module: &morton_shader,
                entry_point: Some("compute_keys"),
                compilation_options: Default::default(),
                cache: None,
            });

        // ── TLAS BVH4 build pipeline ─────────────────────────────────────
        let build_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("TLASBuilder/BuildShader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/tlas-build.wgsl").into(),
            ),
        });

        let build_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("TLASBuilder/BuildBGL"),
            entries: &[
                // binding 0: instances (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 1: bvh4_nodes / BLAS nodes (read)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: tlas_bvh4 (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let build_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("TLASBuilder/BuildPipelineLayout"),
                bind_group_layouts: &[&build_bgl],
                push_constant_ranges: &[],
            });

        let leaf_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TLASBuilder/LeafPipeline"),
                layout: Some(&build_pipeline_layout),
                module: &build_shader,
                entry_point: Some("build_leaves"),
                compilation_options: Default::default(),
                cache: None,
            });

        let internal_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TLASBuilder/InternalPipeline"),
                layout: Some(&build_pipeline_layout),
                module: &build_shader,
                entry_point: Some("build_internal"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Radix sort instance
        let sort = RadixSort::new(renderer);

        Self {
            morton_pipeline,
            morton_bgl,
            leaf_pipeline,
            internal_pipeline,
            build_bgl,
            morton_keys_buf: None,
            morton_vals_buf: None,
            sort_params_buf: None,
            tlas_nodes_buf: None,
            build_params_bufs: Vec::new(),
            sort,
            capacity: 0,
        }
    }

    /// Ensure all buffers are large enough for `instance_count` instances.
    fn ensure_buffers(&mut self, device: &wgpu::Device, instance_count: u32) {
        if instance_count <= self.capacity && self.morton_keys_buf.is_some() {
            return;
        }

        let elem_size = 4u64; // u32

        // Morton code key/value buffers
        let kv_size = (instance_count as u64) * elem_size;
        let kv_usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        self.morton_keys_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TLASBuilder/MortonKeys"),
            size: kv_size,
            usage: kv_usage,
            mapped_at_creation: false,
        }));

        self.morton_vals_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TLASBuilder/MortonVals"),
            size: kv_size,
            usage: kv_usage,
            mapped_at_creation: false,
        }));

        // Sort params uniform buffer
        self.sort_params_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TLASBuilder/SortParams"),
            size: std::mem::size_of::<TlasSortParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));

        // TLAS BVH4 node buffer
        let (levels, total_nodes) = compute_tree_levels(instance_count);
        let node_buf_size =
            (total_nodes as u64) * (NODE_VEC4_STRIDE as u64) * 16; // 128 bytes per node
        self.tlas_nodes_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("TLASBuilder/TlasNodes"),
            size: node_buf_size.max(128), // at least one node
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        // Build param buffers: one for the leaf level + one per internal level
        let num_levels = levels.len();
        self.build_params_bufs = (0..num_levels)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("TLASBuilder/BuildParams/Level{}", i)),
                    size: std::mem::size_of::<TlasBuildParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        self.capacity = instance_count;
    }

    /// Build the TLAS BVH4 from the given instances.
    ///
    /// # Arguments
    /// * `renderer` - The renderer (provides device + queue)
    /// * `instances_buf` - GPU buffer containing `PackedInstance` data
    /// * `blas_nodes_buf` - GPU buffer containing BLAS BVH4 nodes (for world AABB computation)
    /// * `instance_count` - Number of instances
    /// * `scene_bounds_min` - Scene AABB minimum corner
    /// * `scene_bounds_max` - Scene AABB maximum corner
    pub fn build(
        &mut self,
        renderer: &Renderer,
        instances_buf: &wgpu::Buffer,
        blas_nodes_buf: &wgpu::Buffer,
        instance_count: u32,
        scene_bounds_min: [f32; 3],
        scene_bounds_max: [f32; 3],
    ) {
        if instance_count == 0 {
            return;
        }

        let device = renderer.device();
        let queue = renderer.queue();

        self.ensure_buffers(device, instance_count);

        let (levels, _total_nodes) = compute_tree_levels(instance_count);

        // ── Step 1: Compute Morton codes ─────────────────────────────────
        let extent = [
            scene_bounds_max[0] - scene_bounds_min[0],
            scene_bounds_max[1] - scene_bounds_min[1],
            scene_bounds_max[2] - scene_bounds_min[2],
        ];
        let sort_params = TlasSortParams {
            count: instance_count,
            scene_min_x: scene_bounds_min[0],
            scene_min_y: scene_bounds_min[1],
            scene_min_z: scene_bounds_min[2],
            scene_ext_x: if extent[0] > 0.0 { 1.0 / extent[0] } else { 0.0 },
            scene_ext_y: if extent[1] > 0.0 { 1.0 / extent[1] } else { 0.0 },
            scene_ext_z: if extent[2] > 0.0 { 1.0 / extent[2] } else { 0.0 },
            _pad: 0,
        };
        queue.write_buffer(
            self.sort_params_buf.as_ref().unwrap(),
            0,
            bytemuck::bytes_of(&sort_params),
        );

        let morton_keys = self.morton_keys_buf.as_ref().unwrap();
        let morton_vals = self.morton_vals_buf.as_ref().unwrap();

        let morton_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TLASBuilder/MortonBG"),
            layout: &self.morton_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: morton_keys.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: morton_vals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.sort_params_buf.as_ref().unwrap().as_entire_binding(),
                },
            ],
        });

        let wg_count = (instance_count + WG_SIZE - 1) / WG_SIZE;

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("TLASBuilder"),
        });

        // Dispatch Morton code computation
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TLASBuilder/MortonCodes"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.morton_pipeline);
            cpass.set_bind_group(0, &morton_bg, &[]);
            cpass.dispatch_workgroups(wg_count, 1, 1);
        }

        // ── Step 2: Radix sort by Morton code ────────────────────────────
        self.sort.sort(
            renderer,
            &mut encoder,
            morton_keys,
            morton_vals,
            instance_count,
        );

        // ── Step 3: Build BVH4 leaves ────────────────────────────────────
        // levels[0] = (leaf_offset, leaf_count)
        let (leaf_offset, leaf_count) = levels[0];
        let leaf_params = TlasBuildParams {
            instance_count,
            node_offset: leaf_offset,
            child_offset: 0,
            node_count: leaf_count,
            child_count: 0,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        // Ensure we have enough param buffers
        if self.build_params_bufs.is_empty() {
            // This shouldn't happen after ensure_buffers, but just in case
            self.build_params_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("TLASBuilder/BuildParams/Leaf"),
                size: std::mem::size_of::<TlasBuildParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }

        queue.write_buffer(
            &self.build_params_bufs[0],
            0,
            bytemuck::bytes_of(&leaf_params),
        );

        let tlas_nodes = self.tlas_nodes_buf.as_ref().unwrap();

        let build_bg_leaf = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TLASBuilder/BuildBG/Leaves"),
            layout: &self.build_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: instances_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: blas_nodes_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tlas_nodes.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.build_params_bufs[0].as_entire_binding(),
                },
            ],
        });

        let leaf_wg = (leaf_count + WG_SIZE - 1) / WG_SIZE;
        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TLASBuilder/BuildLeaves"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.leaf_pipeline);
            cpass.set_bind_group(0, &build_bg_leaf, &[]);
            cpass.dispatch_workgroups(leaf_wg, 1, 1);
        }

        // ── Step 4: Build BVH4 internal nodes bottom-up ─────────────────
        // levels[1..] are internal levels from bottom to root
        for level_idx in 1..levels.len() {
            let (node_offset, node_count) = levels[level_idx];
            let (child_offset, child_count) = levels[level_idx - 1];

            let internal_params = TlasBuildParams {
                instance_count,
                node_offset,
                child_offset,
                node_count,
                child_count,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };

            // Ensure param buffer exists for this level
            while self.build_params_bufs.len() <= level_idx {
                self.build_params_bufs.push(device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("TLASBuilder/BuildParams/Level{}", level_idx)),
                    size: std::mem::size_of::<TlasBuildParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }

            queue.write_buffer(
                &self.build_params_bufs[level_idx],
                0,
                bytemuck::bytes_of(&internal_params),
            );

            let build_bg_internal = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("TLASBuilder/BuildBG/Internal{}", level_idx)),
                layout: &self.build_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: instances_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: blas_nodes_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: tlas_nodes.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.build_params_bufs[level_idx].as_entire_binding(),
                    },
                ],
            });

            let internal_wg = (node_count + WG_SIZE - 1) / WG_SIZE;
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("TLASBuilder/BuildInternal/Level{}", level_idx)),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.internal_pipeline);
                cpass.set_bind_group(0, &build_bg_internal, &[]);
                cpass.dispatch_workgroups(internal_wg, 1, 1);
            }
        }

        // ── Submit ───────────────────────────────────────────────────────
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Returns the total number of BVH4 nodes for the given instance count.
    pub fn total_nodes(instance_count: u32) -> u32 {
        let (_levels, total) = compute_tree_levels(instance_count);
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_levels_small() {
        // 1 instance -> 1 leaf node, no internal nodes
        let (levels, total) = compute_tree_levels(1);
        assert_eq!(total, 1);
        assert_eq!(levels.len(), 1);
        assert_eq!(levels[0], (0, 1)); // leaf at offset 0, count 1
    }

    #[test]
    fn tree_levels_four() {
        // 4 instances -> 1 leaf node (groups all 4)
        let (levels, total) = compute_tree_levels(4);
        assert_eq!(total, 1);
        assert_eq!(levels.len(), 1);
    }

    #[test]
    fn tree_levels_five() {
        // 5 instances -> ceil(5/4) = 2 leaf nodes -> 1 root
        let (levels, total) = compute_tree_levels(5);
        assert_eq!(total, 3); // 2 leaves + 1 root
        assert_eq!(levels.len(), 2);
        // Root at offset 0 (1 node), leaves at offset 1 (2 nodes)
        assert_eq!(levels[0], (1, 2)); // leaves
        assert_eq!(levels[1], (0, 1)); // root
    }

    #[test]
    fn tree_levels_500() {
        // N=500: ceil(500/4)=125 leaves, ceil(125/4)=32, ceil(32/4)=8, ceil(8/4)=2, ceil(2/4)=1
        // Total = 125 + 32 + 8 + 2 + 1 = 168
        let (levels, total) = compute_tree_levels(500);
        assert_eq!(total, 168);
        assert_eq!(levels.len(), 5);
    }

    #[test]
    fn tree_levels_zero() {
        let (levels, total) = compute_tree_levels(0);
        assert_eq!(total, 0);
        assert!(levels.is_empty());
    }
}
