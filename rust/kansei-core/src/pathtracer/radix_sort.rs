use crate::renderers::Renderer;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Parameters uploaded per radix-sort pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SortParams {
    count: u32,
    bit_offset: u32,
    workgroup_count: u32,
    _pad: u32,
}

const RADIX: u32 = 16;
const WG_SIZE: u32 = 256;
const NUM_PASSES: usize = 8; // 4 bits × 8 = 32 bits

/// GPU radix sort (4-bit radix, 8 passes) for 32-bit key/value pairs.
///
/// Used to sort TLAS instances by Morton code before BVH4 construction.
pub struct RadixSort {
    histogram_pipeline: wgpu::ComputePipeline,
    prefix_sum_pipeline: wgpu::ComputePipeline,
    scatter_pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    /// One parameter buffer per pass to avoid writeBuffer overwrite issues.
    param_buffers: Vec<wgpu::Buffer>,
    /// Temporary key buffer (ping-pong target).
    temp_keys: Option<wgpu::Buffer>,
    /// Temporary value buffer (ping-pong target).
    temp_vals: Option<wgpu::Buffer>,
    /// Global histogram buffer (RADIX × workgroup_count entries).
    histogram_buf: Option<wgpu::Buffer>,
    /// Last allocated capacity (in elements) for temp/histogram buffers.
    allocated_count: u32,
}

impl RadixSort {
    /// Create pipelines and per-pass parameter buffers.
    pub fn new(renderer: &Renderer) -> Self {
        let device = renderer.device();

        let shader_src = include_str!("shaders/radix-sort.wgsl");
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RadixSort Shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RadixSort BGL"),
            entries: &[
                // binding 0: keys_in (read)
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
                // binding 1: vals_in (read)
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
                // binding 2: keys_out (read_write)
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
                // binding 3: vals_out (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 4: histograms (read_write)
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 5: params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RadixSort PipelineLayout"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str, label: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

        let histogram_pipeline = make_pipeline("histogram", "RadixSort Histogram");
        let prefix_sum_pipeline = make_pipeline("prefix_sum", "RadixSort PrefixSum");
        let scatter_pipeline = make_pipeline("scatter", "RadixSort Scatter");

        // Create 8 separate param buffers — one per pass.
        // CRITICAL: writeBuffer overwrites are immediate, so if we reused a single
        // buffer and called writeBuffer 8 times before submit, only the last write
        // would be visible to GPU. Each pass needs its own buffer.
        let param_buffers: Vec<wgpu::Buffer> = (0..NUM_PASSES)
            .map(|i| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("RadixSort Params Pass {i}")),
                    size: std::mem::size_of::<SortParams>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();

        Self {
            histogram_pipeline,
            prefix_sum_pipeline,
            scatter_pipeline,
            bgl,
            param_buffers,
            temp_keys: None,
            temp_vals: None,
            histogram_buf: None,
            allocated_count: 0,
        }
    }

    /// Ensure temp buffers are large enough for `count` elements.
    fn ensure_buffers(&mut self, device: &wgpu::Device, count: u32) {
        if count <= self.allocated_count && self.temp_keys.is_some() {
            return;
        }

        let buf_size = (count as u64) * 4; // u32 per element

        self.temp_keys = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RadixSort TempKeys"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        self.temp_vals = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RadixSort TempVals"),
            size: buf_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        let workgroup_count = (count + WG_SIZE - 1) / WG_SIZE;
        let hist_size = (RADIX as u64) * (workgroup_count as u64) * 4;
        self.histogram_buf = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RadixSort Histogram"),
            size: hist_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        }));

        self.allocated_count = count;
    }

    /// Sort `count` key/value pairs in-place using GPU radix sort.
    ///
    /// After sorting, results are in `keys_buf` / `vals_buf` (since 8 passes
    /// is even, the final scatter writes back into the original buffers).
    ///
    /// Both buffers must have `STORAGE | COPY_SRC | COPY_DST` usage and hold
    /// at least `count` u32 elements.
    pub fn sort(
        &mut self,
        renderer: &Renderer,
        encoder: &mut wgpu::CommandEncoder,
        keys_buf: &wgpu::Buffer,
        vals_buf: &wgpu::Buffer,
        count: u32,
    ) {
        if count <= 1 {
            return;
        }

        let device = renderer.device();
        let queue = renderer.queue();

        self.ensure_buffers(device, count);

        let workgroup_count = (count + WG_SIZE - 1) / WG_SIZE;

        let temp_keys = self.temp_keys.as_ref().unwrap();
        let temp_vals = self.temp_vals.as_ref().unwrap();
        let histogram_buf = self.histogram_buf.as_ref().unwrap();

        // Upload all 8 param buffers before encoding.
        for pass in 0..NUM_PASSES {
            let params = SortParams {
                count,
                bit_offset: (pass as u32) * 4,
                workgroup_count,
                _pad: 0,
            };
            queue.write_buffer(
                &self.param_buffers[pass],
                0,
                bytemuck::bytes_of(&params),
            );
        }

        // Even passes: keys_buf -> temp_keys; Odd passes: temp_keys -> keys_buf.
        // After 8 passes (even count), result is back in keys_buf/vals_buf.
        for pass in 0..NUM_PASSES {
            let (src_keys, src_vals, dst_keys, dst_vals) = if pass % 2 == 0 {
                (
                    keys_buf as &wgpu::Buffer,
                    vals_buf as &wgpu::Buffer,
                    temp_keys as &wgpu::Buffer,
                    temp_vals as &wgpu::Buffer,
                )
            } else {
                (
                    temp_keys as &wgpu::Buffer,
                    temp_vals as &wgpu::Buffer,
                    keys_buf as &wgpu::Buffer,
                    vals_buf as &wgpu::Buffer,
                )
            };

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("RadixSort BG Pass {pass}")),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: src_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src_vals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst_keys.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dst_vals.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: histogram_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: self.param_buffers[pass].as_entire_binding(),
                    },
                ],
            });

            // Histogram
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("RadixSort Histogram Pass {pass}")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.histogram_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroup_count, 1, 1);
            }

            // Prefix sum (single workgroup)
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("RadixSort PrefixSum Pass {pass}")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.prefix_sum_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(1, 1, 1);
            }

            // Scatter
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some(&format!("RadixSort Scatter Pass {pass}")),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&self.scatter_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(workgroup_count, 1, 1);
            }
        }
    }
}
