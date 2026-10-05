//! `RtScene`: meshes and instances placed on the CPU, gathered into an `RtGrid`.

use glam::{Mat4, Vec3};

use super::grid::{RtGrid, RtPlacement, RtSource, RtSurface};
use super::mesh::{transform_box, RtMesh};
use crate::clusters::InstanceTransform;

/// An instance of one of an `RtScene`'s meshes.
#[derive(Clone, Copy, Debug)]
pub struct RtInstance {
    pub mesh: usize,
    /// Mesh to world.
    pub transform: Mat4,
    pub surface: RtSurface,
}

struct SceneMesh {
    mesh: RtMesh,
    buffer: Option<wgpu::Buffer>,
}

/// Meshes and their instances for an `RtGrid`, with no renderer: each gather picks the instances
/// whose world box meets the grid's on the CPU and places their triangles on the GPU. A source per
/// mesh and surface; its id (`kansei_rt_source`) is the mesh's index, its record
/// (`kansei_rt_record`) the instance's rank among that source's in the box.
#[derive(Default)]
pub struct RtScene {
    meshes: Vec<SceneMesh>,
    pub instances: Vec<RtInstance>,
    records: Option<wgpu::Buffer>,
    /// Instances in the box at the last gather, and their triangles.
    pub in_box: usize,
    pub in_box_triangles: u64,
}

const MATRIX: RtPlacement = RtPlacement::Instance(InstanceTransform::Matrix { offset: 0 });

impl RtScene {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a mesh; its index.
    pub fn add_mesh(&mut self, mesh: RtMesh) -> usize {
        self.meshes.push(SceneMesh { mesh, buffer: None });
        self.meshes.len() - 1
    }

    pub fn mesh(&self, index: usize) -> &RtMesh {
        &self.meshes[index].mesh
    }

    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }

    /// Add an instance; its index.
    pub fn add_instance(&mut self, instance: RtInstance) -> usize {
        assert!(instance.mesh < self.meshes.len(), "no mesh {}", instance.mesh);
        self.instances.push(instance);
        self.instances.len() - 1
    }

    /// Gather the instances whose world box meets `grid`'s into it: between `RtGrid::begin` and
    /// `RtGrid::finish`, as its one gather this build.
    pub fn gather(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, grid: &mut RtGrid) {
        let (lo, hi) = grid.bounds();
        let eps = Vec3::splat(grid.options().effective_epsilon());
        // the instances in the box, by (mesh, surface): their matrices, a source's run each
        let mut chosen: Vec<(usize, u32, u32, usize)> = Vec::new();
        for (k, i) in self.instances.iter().enumerate() {
            let m = &self.meshes[i.mesh].mesh;
            let (a, b) = transform_box(&i.transform, m.min, m.max);
            if a.cmple(hi + eps).all() && b.cmpge(lo - eps).all() {
                chosen.push((i.mesh, i.surface.surface_word(), i.surface.albedo_word(), k));
            }
        }
        chosen.sort_unstable();
        self.in_box = chosen.len();
        self.in_box_triangles = chosen.iter().map(|c| self.meshes[c.0].mesh.triangle_count() as u64).sum();
        let records: Vec<[f32; 16]> = chosen.iter().map(|c| self.instances[c.3].transform.to_cols_array()).collect();
        let bytes = (records.len().max(1) * 64) as u64;
        if self.records.as_ref().is_none_or(|r| r.size() < bytes) {
            self.records = Some(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("RtScene/Records"),
                size: bytes.next_power_of_two(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }));
        }
        let records_buffer = self.records.as_ref().unwrap();
        if !records.is_empty() {
            queue.write_buffer(records_buffer, 0, bytemuck::cast_slice(&records));
        }
        for c in &chosen {
            let m = &mut self.meshes[c.0];
            if m.buffer.is_none() {
                m.buffer = Some(m.mesh.create_buffer(device));
            }
        }
        grid.reserve_triangles(device, queue, self.in_box_triangles.min(u32::MAX as u64) as u32);
        // a source per run of one mesh and surface
        let mut sources = Vec::new();
        let mut start = 0;
        while start < chosen.len() {
            let end = start + chosen[start..].iter().take_while(|c| (c.0, c.1, c.2) == (chosen[start].0, chosen[start].1, chosen[start].2)).count();
            let mesh = &self.meshes[chosen[start].0];
            sources.push(RtSource {
                mesh: mesh.buffer.as_ref().unwrap(),
                triangles: mesh.mesh.triangle_count(),
                records: Some((records_buffer, 64)),
                first_record: start as u32,
                record_count: (end - start) as u32,
                world: Mat4::IDENTITY,
                placement: &MATRIX,
                surface: self.instances[chosen[start].3].surface,
                id: chosen[start].0 as u32,
            });
            start = end;
        }
        grid.gather(device, queue, encoder, &sources);
    }
}
