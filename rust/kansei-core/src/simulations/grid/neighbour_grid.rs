/// The grid's uniform and the helpers that walk it, for passes that search neighbours: `struct
/// NeighbourGrid { origin, cellSize, dims, count }` (bind [`NeighbourGrid::params_buffer`]),
/// `neighbourCell(p, grid) -> vec3<i32>`, `neighbourCellInside(cell, grid) -> bool` and
/// `neighbourCellIndex(cell, grid) -> u32`.
pub const NEIGHBOUR_GRID_WGSL: &str = include_str!("shaders/neighbour-grid.wgsl");

const CLEAR_WGSL: &str = include_str!("shaders/grid-clear.wgsl");
const ASSIGN_WGSL: &str = include_str!("shaders/grid-assign.wgsl");
const PREFIX_SUM_LOCAL_WGSL: &str = include_str!("shaders/prefix-sum-local.wgsl");
const PREFIX_SUM_TOP_WGSL: &str = include_str!("shaders/prefix-sum-top.wgsl");
const PREFIX_SUM_DISTRIBUTE_WGSL: &str = include_str!("shaders/prefix-sum-distribute.wgsl");
const SCATTER_WGSL: &str = include_str!("shaders/scatter.wgsl");
const PREFIX_SUM_BLOCK_SIZE: u32 = 512;
/// WebGPU's default limit of storage buffers per shader stage leaves the scatter room for two.
const MAX_SORTED_COPIES: usize = 2;

/// `NeighbourGrid` in [`NEIGHBOUR_GRID_WGSL`].
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuNeighbourGrid {
    pub origin: [f32; 3],
    pub cell_size: f32,
    pub dims: [u32; 3],
    pub count: u32,
}

/// Where a grid's cells lie: `dims` cells `cell_size` wide from `origin`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLayout {
    pub origin: [f32; 3],
    pub cell_size: f32,
    pub dims: [u32; 3],
}

impl GridLayout {
    /// Cells `cell_size` wide over the box from `min` to `max` (at least one per axis; an axis
    /// with no extent gets one), widened in steps of 25% until there are at most `max_cells`.
    /// A search of the cells around a point reaches every neighbour within `cell_size`: pass the
    /// search radius. Clamping the counts per axis instead would fold every point beyond them
    /// into the edge cells.
    pub fn covering(min: [f32; 3], max: [f32; 3], cell_size: f32, max_cells: u32) -> Self {
        assert!(cell_size > 0.0, "cells need a width");
        let mut cell = cell_size;
        loop {
            let dims: [u32; 3] = std::array::from_fn(|d| (((max[d] - min[d]) / cell).ceil() as u32).max(1));
            if dims.iter().map(|&n| n as u64).product::<u64>() <= max_cells.max(1) as u64 {
                return Self { origin: min, cell_size: cell, dims };
            }
            cell *= 1.25;
        }
    }

    pub fn total_cells(&self) -> u32 {
        self.dims[0] * self.dims[1] * self.dims[2]
    }
}

/// What a [`NeighbourGrid`] sorts.
pub struct NeighbourGridOptions<'a> {
    pub label: &'a str,
    /// The most points it holds.
    pub capacity: u32,
    pub layout: GridLayout,
    /// The points: `array<vec4<f32>>` with the position in `xyz`, at least `capacity` long.
    pub positions: &'a wgpu::Buffer,
    /// Per-point `array<vec4<f32>>` buffers (at most two) the grid also copies in cell order
    /// each step, into [`sorted`](NeighbourGrid::sorted): e.g. the positions and velocities, so a
    /// neighbour search reads them contiguously.
    pub sorted_copies: &'a [&'a wgpu::Buffer],
}

/// A counting-sort grid of points on the GPU, rebuilt by [`encode`](Self::encode) each step:
/// each cell's point count ([`cell_counts`](Self::cell_counts)), where its points start in the
/// sorted order ([`cell_offsets`](Self::cell_offsets)), the sorted order itself
/// ([`sorted_indices`](Self::sorted_indices), slot to point) and cell-ordered copies of chosen
/// buffers ([`sorted`](Self::sorted)). A pass binds those with
/// [`params_buffer`](Self::params_buffer) and [`NEIGHBOUR_GRID_WGSL`] to visit the points near
/// one. Points outside the grid go into its edge cells.
///
/// The grid sorts the first [`count`](Self::count) points. [`set_count`](Self::set_count) and
/// [`set_layout`](Self::set_layout) write the uniform, which lands before the next submit.
pub struct NeighbourGrid {
    label: String,
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: GridLayout,
    capacity: u32,
    count: u32,
    params: wgpu::Buffer,
    positions: wgpu::Buffer,
    /// Each copied buffer and its cell-ordered copy.
    copies: Vec<(wgpu::Buffer, wgpu::Buffer)>,
    cell_indices: wgpu::Buffer,
    sorted_indices: wgpu::Buffer,
    // sized by the layout's cells: replaced when their number changes
    cell_counts: wgpu::Buffer,
    cell_offsets: wgpu::Buffer,
    scatter_counters: wgpu::Buffer,
    block_sums: wgpu::Buffer,
    passes: [(wgpu::ComputePipeline, wgpu::BindGroupLayout); 6],
    bind_groups: Vec<wgpu::BindGroup>,
}

const CLEAR: usize = 0;
const ASSIGN: usize = 1;
const PREFIX_LOCAL: usize = 2;
const PREFIX_TOP: usize = 3;
const PREFIX_DISTRIBUTE: usize = 4;
const SCATTER: usize = 5;

impl NeighbourGrid {
    /// A grid over `options.positions` holding no points yet (see [`set_count`](Self::set_count)).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, options: &NeighbourGridOptions) -> Self {
        assert!(options.sorted_copies.len() <= MAX_SORTED_COPIES, "a NeighbourGrid copies at most {MAX_SORTED_COPIES} buffers");
        let label = options.label;
        let n = options.capacity.max(1) as u64;
        let storage = |name: &str, size: u64| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}/{name}")),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let copies = options.sorted_copies.iter().enumerate()
            .map(|(k, source)| ((*source).clone(), storage(&format!("Sorted{k}"), n * 16)))
            .collect();
        let cell_indices = storage("CellIndices", n * 4);
        let sorted_indices = storage("SortedIndices", n * 4);
        let [cell_counts, cell_offsets, scatter_counters, block_sums] = Self::cell_buffers(device, label, options.layout.total_cells());
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label}/Params")),
            size: std::mem::size_of::<GpuNeighbourGrid>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let with_grid = |code: &str| format!("{NEIGHBOUR_GRID_WGSL}\n{code}");
        let scatter = scatter_wgsl(options.sorted_copies.len());
        // each pass's bindings in order: read-write storage, read-only storage, or the uniform
        use Slot::{Rw, Ro, Uniform};
        let mut scatter_slots = vec![Ro, Ro, Rw, Rw, Uniform];
        for _ in options.sorted_copies {
            scatter_slots.extend([Ro, Rw]);
        }
        let make = |name: &str, code: &str, slots: &[Slot]| pipeline(device, &format!("{label}/{name}"), code, slots);
        let passes = [
            make("Clear", CLEAR_WGSL, &[Rw, Rw]),
            make("Assign", &with_grid(ASSIGN_WGSL), &[Ro, Rw, Rw, Uniform]),
            make("PrefixSumLocal", PREFIX_SUM_LOCAL_WGSL, &[Rw, Rw, Rw]),
            make("PrefixSumTop", PREFIX_SUM_TOP_WGSL, &[Rw]),
            make("PrefixSumDistribute", PREFIX_SUM_DISTRIBUTE_WGSL, &[Rw, Rw]),
            make("Scatter", &scatter, &scatter_slots),
        ];

        let mut grid = Self {
            label: label.to_string(),
            device: device.clone(),
            queue: queue.clone(),
            layout: options.layout,
            capacity: options.capacity,
            count: 0,
            params,
            positions: options.positions.clone(),
            copies,
            cell_indices,
            sorted_indices,
            cell_counts,
            cell_offsets,
            scatter_counters,
            block_sums,
            passes,
            bind_groups: Vec::new(),
        };
        grid.bind();
        grid.write_params();
        grid
    }

    /// Cell counts, cell offsets, scatter counters and the prefix sum's block sums for `cells`.
    fn cell_buffers(device: &wgpu::Device, label: &str, cells: u32) -> [wgpu::Buffer; 4] {
        let blocks = cells.div_ceil(PREFIX_SUM_BLOCK_SIZE);
        let make = |name: &str, len: u32| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("{label}/{name}")),
                size: len.max(1) as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        [make("CellCounts", cells), make("CellOffsets", cells), make("ScatterCounters", cells), make("BlockSums", blocks)]
    }

    fn bind(&mut self) {
        let mut scatter = vec![&self.cell_indices, &self.cell_offsets, &self.scatter_counters, &self.sorted_indices, &self.params];
        for (source, sorted) in &self.copies {
            scatter.extend([source, sorted]);
        }
        let buffers: [Vec<&wgpu::Buffer>; 6] = [
            vec![&self.cell_counts, &self.scatter_counters],
            vec![&self.positions, &self.cell_indices, &self.cell_counts, &self.params],
            // an exclusive scan of the counts into the offsets (the counts stay as they were)
            vec![&self.cell_counts, &self.cell_offsets, &self.block_sums],
            vec![&self.block_sums],
            vec![&self.block_sums, &self.cell_offsets],
            scatter,
        ];
        self.bind_groups = buffers.iter().zip(&self.passes).map(|(buffers, (_, layout))| {
            let entries: Vec<wgpu::BindGroupEntry> = buffers.iter().enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry { binding: binding as u32, resource: buffer.as_entire_binding() })
                .collect();
            self.device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some(&self.label), layout, entries: &entries })
        }).collect();
    }

    fn write_params(&self) {
        let gpu = GpuNeighbourGrid { origin: self.layout.origin, cell_size: self.layout.cell_size, dims: self.layout.dims, count: self.count };
        self.queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&gpu));
    }

    /// Sort the first `count` points (at most the capacity) from the next step on.
    pub fn set_count(&mut self, count: u32) {
        assert!(count <= self.capacity, "{count} points in a grid for {}", self.capacity);
        if count != self.count {
            self.count = count;
            self.write_params();
        }
    }

    /// Move or resize the cells. Returns whether that replaced the per-cell buffers
    /// ([`cell_counts`](Self::cell_counts), [`cell_offsets`](Self::cell_offsets)): it does when
    /// the number of cells changes, and passes bound to them need new bind groups.
    pub fn set_layout(&mut self, layout: GridLayout) -> bool {
        if layout == self.layout {
            return false;
        }
        let replaced = layout.total_cells() != self.layout.total_cells();
        self.layout = layout;
        if replaced {
            [self.cell_counts, self.cell_offsets, self.scatter_counters, self.block_sums] =
                Self::cell_buffers(&self.device, &self.label, layout.total_cells());
            self.bind();
        }
        self.write_params();
        replaced
    }

    /// Rebuild the grid from the points' current positions: six dispatches in `pass`.
    pub fn encode(&self, pass: &mut wgpu::ComputePass<'_>) {
        let cells = self.layout.total_cells();
        let points = self.count.div_ceil(64);
        let blocks = cells.div_ceil(PREFIX_SUM_BLOCK_SIZE).max(1);
        for (k, workgroups) in [(CLEAR, cells.div_ceil(256)), (ASSIGN, points), (PREFIX_LOCAL, blocks), (PREFIX_TOP, 1), (PREFIX_DISTRIBUTE, blocks), (SCATTER, points)] {
            pass.set_pipeline(&self.passes[k].0);
            pass.set_bind_group(0, &self.bind_groups[k], &[]);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
    }

    pub fn layout(&self) -> GridLayout { self.layout }
    /// The points sorted: the first `count` of the positions buffer.
    pub fn count(&self) -> u32 { self.count }
    pub fn capacity(&self) -> u32 { self.capacity }
    /// The `NeighbourGrid` uniform of [`NEIGHBOUR_GRID_WGSL`]: the layout and the count.
    pub fn params_buffer(&self) -> &wgpu::Buffer { &self.params }
    /// Each cell's point count (`array<u32>`).
    pub fn cell_counts(&self) -> &wgpu::Buffer { &self.cell_counts }
    /// Each cell's first sorted slot (`array<u32>`).
    pub fn cell_offsets(&self) -> &wgpu::Buffer { &self.cell_offsets }
    /// Each sorted slot's point (`array<u32>`).
    pub fn sorted_indices(&self) -> &wgpu::Buffer { &self.sorted_indices }
    /// The cell-ordered copy of `sorted_copies[k]` (`array<vec4<f32>>`).
    pub fn sorted(&self, k: usize) -> &wgpu::Buffer { &self.copies[k].1 }
}

/// The scatter, copying `copies` buffers into cell order (bindings from 5 on, source then copy).
fn scatter_wgsl(copies: usize) -> String {
    let (bindings, writes) = (0..copies).fold((String::new(), String::new()), |(mut b, mut w), k| {
        b += &format!(
            "@group(0) @binding({}) var<storage, read> source{k}: array<vec4<f32>>;\n@group(0) @binding({}) var<storage, read_write> sorted{k}: array<vec4<f32>>;\n",
            5 + 2 * k,
            6 + 2 * k
        );
        w += &format!("sorted{k}[slot] = source{k}[idx];\n    ");
        (b, w)
    });
    format!("{NEIGHBOUR_GRID_WGSL}\n{SCATTER_WGSL}").replace("//__COPY_BINDINGS__", &bindings).replace("//__COPIES__", &writes)
}

#[derive(Clone, Copy)]
enum Slot {
    Rw,
    Ro,
    Uniform,
}

fn pipeline(device: &wgpu::Device, label: &str, code: &str, slots: &[Slot]) -> (wgpu::ComputePipeline, wgpu::BindGroupLayout) {
    let entries: Vec<wgpu::BindGroupLayoutEntry> = slots.iter().enumerate().map(|(binding, slot)| wgpu::BindGroupLayoutEntry {
        binding: binding as u32,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: match slot {
                Slot::Rw => wgpu::BufferBindingType::Storage { read_only: false },
                Slot::Ro => wgpu::BufferBindingType::Storage { read_only: true },
                Slot::Uniform => wgpu::BufferBindingType::Uniform,
            },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }).collect();
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(code.into()) });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: &[&layout], push_constant_ranges: &[] });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (pipeline, layout)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn validate(name: &str, code: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{name}: {}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        module
    }

    #[test]
    fn shaders_validate_and_the_uniform_matches() {
        let with_grid = |code: &str| format!("{NEIGHBOUR_GRID_WGSL}\n{code}");
        for (name, code) in [
            ("clear", CLEAR_WGSL.to_string()),
            ("assign", with_grid(ASSIGN_WGSL)),
            ("prefix local", PREFIX_SUM_LOCAL_WGSL.to_string()),
            ("prefix top", PREFIX_SUM_TOP_WGSL.to_string()),
            ("prefix distribute", PREFIX_SUM_DISTRIBUTE_WGSL.to_string()),
            ("scatter", scatter_wgsl(0)),
            ("scatter with copies", scatter_wgsl(MAX_SORTED_COPIES)),
        ] {
            let module = validate(name, &code);
            for (_, ty) in module.types.iter() {
                if let (Some("NeighbourGrid"), naga::TypeInner::Struct { span, .. }) = (ty.name.as_deref(), &ty.inner) {
                    assert_eq!(*span as usize, std::mem::size_of::<GpuNeighbourGrid>(), "{name}");
                }
            }
        }
    }

    #[test]
    fn layouts_cover_the_box_within_the_cap() {
        let layout = GridLayout::covering([-1.0, 0.0, -1.0], [1.0, 3.0, 1.0], 0.5, 1 << 20);
        assert_eq!((layout.dims, layout.cell_size, layout.origin), ([4, 6, 4], 0.5, [-1.0, 0.0, -1.0]));
        // a flat box gets one cell across
        assert_eq!(GridLayout::covering([0.0; 3], [2.0, 2.0, 0.0], 1.0, 100).dims, [2, 2, 1]);
        // too many cells: wider cells, the same on every axis, still covering the box
        let capped = GridLayout::covering([0.0; 3], [100.0; 3], 1.0, 1000);
        assert!(capped.total_cells() <= 1000 && capped.cell_size > 1.0);
        assert!(capped.dims.iter().all(|&n| n as f32 * capped.cell_size >= 100.0));
    }
}
