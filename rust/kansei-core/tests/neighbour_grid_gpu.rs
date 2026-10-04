//! The neighbour grid on a real GPU: every point lands in its cell (points outside the grid in
//! its edge cells), the offsets are the exclusive prefix sum of the counts (over more than 512
//! scan blocks of cells, which the top-level scan takes in chunks), the sorted order lists each
//! cell's points in its slots, and the copies follow that order. A smaller count, and a layout
//! with more cells, sort again. Skipped (passes) when no adapter is available.

use kansei_core::simulations::grid::{GridLayout, NeighbourGrid, NeighbourGridOptions};
use wgpu::util::DeviceExt;

fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).ok()
}

fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<T> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buffer.size(), usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit(Some(encoder.finish()));
    staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device.poll(wgpu::Maintain::Wait);
    let data = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    data
}

fn sort(device: &wgpu::Device, queue: &wgpu::Queue, grid: &NeighbourGrid) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut pass = encoder.begin_compute_pass(&Default::default());
    grid.encode(&mut pass);
    drop(pass);
    queue.submit(Some(encoder.finish()));
}

/// The cell `p` falls in, clamped into the grid.
fn cell_of(p: [f32; 4], l: &GridLayout) -> u32 {
    let c: [u32; 3] = std::array::from_fn(|d| (((p[d] - l.origin[d]) / l.cell_size).floor() as i64).clamp(0, l.dims[d] as i64 - 1) as u32);
    c[0] + l.dims[0] * (c[1] + l.dims[1] * c[2])
}

fn check(device: &wgpu::Device, queue: &wgpu::Queue, grid: &NeighbourGrid, points: &[[f32; 4]], tags: &[[f32; 4]]) {
    let layout = grid.layout();
    let n = grid.count() as usize;
    let cells = layout.total_cells() as usize;
    let counts: Vec<u32> = read(device, queue, grid.cell_counts());
    let offsets: Vec<u32> = read(device, queue, grid.cell_offsets());
    let sorted: Vec<u32> = read(device, queue, grid.sorted_indices());
    let sorted_tags: Vec<[f32; 4]> = read(device, queue, grid.sorted(0));

    let mut expected = vec![0u32; cells];
    for p in &points[..n] {
        expected[cell_of(*p, &layout) as usize] += 1;
    }
    assert_eq!(&counts[..cells], &expected[..], "counts");
    let mut running = 0;
    for c in 0..cells {
        assert_eq!(offsets[c], running, "offset of cell {c}");
        running += counts[c];
    }
    // each cell's slots hold its own points, every point exactly once
    let mut seen = vec![false; n];
    for c in 0..cells {
        for slot in offsets[c]..offsets[c] + counts[c] {
            let i = sorted[slot as usize] as usize;
            assert!(i < n && !seen[i], "slot {slot}");
            seen[i] = true;
            assert_eq!(cell_of(points[i], &layout), c as u32);
            assert_eq!(sorted_tags[slot as usize], tags[i], "copy of point {i}");
        }
    }
    assert!(seen.iter().all(|&s| s));
}

#[test]
fn points_sort_into_their_cells() {
    let Some((device, queue)) = gpu() else { return };
    // 70 × 70 × 70 cells: 343,000, 670 scan blocks
    let layout = GridLayout::covering([-35.0; 3], [35.0; 3], 1.0, 1 << 20);
    assert_eq!(layout.dims, [70; 3]);
    let capacity = 20_000;
    // a cheap hash spreads points over the grid and a little past it on every side
    let mut state = 12345u32;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state as f32 / u32::MAX as f32
    };
    let points: Vec<[f32; 4]> = (0..capacity).map(|_| [next() * 80.0 - 40.0, next() * 80.0 - 40.0, next() * 80.0 - 40.0, 1.0]).collect();
    // some piled into one cell
    let points: Vec<[f32; 4]> = points.iter().enumerate().map(|(i, p)| if i % 10 == 0 { [0.5, 0.5, 0.5, 1.0] } else { *p }).collect();
    let tags: Vec<[f32; 4]> = (0..capacity).map(|i| [i as f32, 0.0, 0.0, 1.0]).collect();
    let storage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST;
    let positions = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&points), usage: storage });
    let tag_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&tags), usage: storage });

    let mut grid = NeighbourGrid::new(&device, &queue, &NeighbourGridOptions {
        label: "Test/Grid",
        capacity: capacity as u32,
        layout,
        positions: &positions,
        sorted_copies: &[&tag_buffer],
    });
    grid.set_count(capacity as u32);
    sort(&device, &queue, &grid);
    check(&device, &queue, &grid, &points, &tags);

    // fewer points: the rest are left out
    grid.set_count(7_000);
    sort(&device, &queue, &grid);
    check(&device, &queue, &grid, &points, &tags);

    // finer cells: new per-cell buffers, sorted again
    assert!(grid.set_layout(GridLayout::covering([-35.0; 3], [35.0; 3], 0.5, 1 << 22)));
    assert!(!grid.set_layout(grid.layout()));
    sort(&device, &queue, &grid);
    check(&device, &queue, &grid, &points, &tags);
}
