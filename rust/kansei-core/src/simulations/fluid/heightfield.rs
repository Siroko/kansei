//! `FluidHeightfield`: a fluid's surface as a height field, and as a coarse mesh for the ray
//! tracing grid.

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use super::FluidDensityField;
use crate::rt::{RtGpuMesh, RtMesh};

const WGSL: &str = include_str!("shaders/fluid-heightfield.wgsl");

/// The ripples a still water's surface shows: `kansei_water_ripples(xz, t)`, the slope of six
/// small travelling waves (see the shader), shared by the water's glass material
/// (`FluidSurfaceEffect::surface_renderable_glass`) and the caustics' photons
/// (`rt::RtCausticsEffect`).
pub const WATER_RIPPLES_WGSL: &str = include_str!("shaders/water-ripples.wgsl");

/// What `FluidHeightfield::new` sets up.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FluidHeightfieldOptions {
    /// The density field's iso level (its surface: 1 for a surface field).
    pub iso: f32,
    /// World units per simulation unit (the surface renderable's scale).
    pub world_scale: f32,
    /// The ray tracing grid's mesh: metres between its vertices (about the grid's cell: the
    /// grid's cost is the triangles in a cell).
    pub rt_spacing: f32,
}

impl Default for FluidHeightfieldOptions {
    fn default() -> Self {
        Self { iso: 1.0, world_scale: 1.0, rt_spacing: 0.3 }
    }
}

/// The WGSL `HeightfieldParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    bounds_min: [f32; 3],
    iso: f32,
    bounds_max: [f32; 3],
    scale: f32,
    dims: [u32; 3],
    _pad0: u32,
    mesh_dims: [u32; 2],
    spacing: f32,
    _pad1: u32,
    mesh_min: [f32; 2],
    _pad2: [f32; 2],
}

/// A fluid's surface as a height field over its bounds, made from its density field on the GPU
/// (`update`, after the field): in each column of voxels the top of the body of fluid seen from
/// below, so the spray above it is left out (a lake's or a pond's surface, not a splash's). It
/// gives
/// - `height_view`: an r32float texture of world heights, a texel a column of the field
///   (-1e9 where a column holds no fluid), over `rect`, for passes that trace the water's
///   surface finely (the caustics' photons);
/// - `rt_mesh`: a grid of vertices `rt_spacing` apart over it, for `Renderable::rt_mesh`, so the
///   ray tracing grid holds the water (as glass, `RtSurface::glass`: mirrors and glass see it,
///   the GI's and the shadows' rays pass through), coarse because the grid's cost is triangles a
///   cell; the quads with a dry corner are left out. Bump the renderable's `Geometry::revision`
///   when it was updated.
pub struct FluidHeightfield {
    params: wgpu::Buffer,
    heights: wgpu::TextureView,
    mesh: RtGpuMesh,
    dims: [u32; 3],
    mesh_dims: [u32; 2],
    heights_pipeline: wgpu::ComputePipeline,
    mesh_pipeline: wgpu::ComputePipeline,
    quads_pipeline: wgpu::ComputePipeline,
    heights_group: wgpu::BindGroup,
    mesh_group: wgpu::BindGroup,
    rect: ([f32; 2], [f32; 2]),
}

impl FluidHeightfield {
    /// Over `field` (spanning `bounds_min` to `bounds_max`, simulation units).
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, field: &FluidDensityField, bounds_min: [f32; 3], bounds_max: [f32; 3], options: FluidHeightfieldOptions) -> Self {
        let dims = field.tex_dims();
        let s = options.world_scale;
        let (lo, hi) = (Vec3::from(bounds_min) * s, Vec3::from(bounds_max) * s);
        let spacing = options.rt_spacing.max(0.01);
        let mesh_dims = [(((hi.x - lo.x) / spacing).floor() as u32 + 1).max(2), (((hi.z - lo.z) / spacing).floor() as u32 + 1).max(2)];
        // the mesh: its vertices (rewritten by `rt_mesh`) and its quads' indices (by `rt_quads`)
        let vertices = (mesh_dims[0] * mesh_dims[1]) as usize;
        let quads = ((mesh_dims[0] - 1) * (mesh_dims[1] - 1)) as usize;
        let rt = RtMesh { positions: vec![[lo.x, lo.y, lo.z]; vertices], uvs: vec![[0.0; 2]; vertices], normals: vec![[0.0, 1.0, 0.0]; vertices], indices: vec![0; quads * 6], min: lo, max: hi };
        let mesh = RtGpuMesh::new(device, "FluidHeightfield/RtMesh", &rt, lo, hi);
        let heights = device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some("FluidHeightfield/Heights"),
                size: wgpu::Extent3d { width: dims[0], height: dims[2], depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
                view_formats: &[],
            })
            .create_view(&Default::default());
        let params = Params {
            bounds_min,
            iso: options.iso,
            bounds_max,
            scale: s,
            dims,
            _pad0: 0,
            mesh_dims,
            spacing,
            _pad1: 0,
            mesh_min: [lo.x, lo.z],
            _pad2: [0.0; 2],
        };
        use wgpu::util::DeviceExt;
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("FluidHeightfield/Params"), contents: bytemuck::bytes_of(&params), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let _ = queue;
        let code = format!("{}\n{WGSL}", include_str!("../../rt/shaders/rt_types.wgsl"));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("FluidHeightfield"), source: wgpu::ShaderSource::Wgsl(code.into()) });
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let texture = |view_dimension| wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension, multisampled: false };
        let heights_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidHeightfield/Heights"),
            entries: &[
                entry(0, uniform),
                entry(1, texture(wgpu::TextureViewDimension::D3)),
                entry(2, wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D2 }),
            ],
        });
        let mesh_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("FluidHeightfield/Mesh"),
            entries: &[entry(0, uniform), entry(3, texture(wgpu::TextureViewDimension::D2)), entry(4, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None })],
        });
        let pipeline = |entry: &str, bgl: &wgpu::BindGroupLayout| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(entry), bind_group_layouts: &[bgl], push_constant_ranges: &[] });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(entry), layout: Some(&layout), module: &module, entry_point: Some(entry), compilation_options: Default::default(), cache: None })
        };
        let heights_pipeline = pipeline("heights", &heights_bgl);
        let mesh_pipeline = pipeline("rt_mesh", &mesh_bgl);
        let quads_pipeline = pipeline("rt_quads", &mesh_bgl);
        let heights_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidHeightfield/Heights"),
            layout: &heights_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&field.density_view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&heights) },
            ],
        });
        let mesh_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidHeightfield/Mesh"),
            layout: &mesh_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&heights) },
                wgpu::BindGroupEntry { binding: 4, resource: mesh.buffer.as_entire_binding() },
            ],
        });
        Self { params, heights, mesh, dims, mesh_dims, heights_pipeline, mesh_pipeline, quads_pipeline, heights_group, mesh_group, rect: ([lo.x, lo.z], [hi.x - lo.x, hi.z - lo.z]) }
    }

    /// Find the heights in the density field as it is now, and write the grid's mesh over them.
    pub fn update(&self, encoder: &mut wgpu::CommandEncoder) {
        let stamp = crate::profiling::gpu_pass("FluidHeightfield");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FluidHeightfield"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.heights_pipeline);
        pass.set_bind_group(0, &self.heights_group, &[]);
        pass.dispatch_workgroups(self.dims[0].div_ceil(8), self.dims[2].div_ceil(8), 1);
        pass.set_pipeline(&self.mesh_pipeline);
        pass.set_bind_group(0, &self.mesh_group, &[]);
        pass.dispatch_workgroups(self.mesh_dims[0].div_ceil(8), self.mesh_dims[1].div_ceil(8), 1);
        pass.set_pipeline(&self.quads_pipeline);
        pass.dispatch_workgroups(self.mesh_dims[0].div_ceil(8), self.mesh_dims[1].div_ceil(8), 1);
    }

    /// The world heights (r32float, a texel a column of the density field, -1e9 where dry).
    pub fn height_view(&self) -> &wgpu::TextureView {
        &self.heights
    }

    /// The world rectangle (x, z) the height texture spans: its corner and size (texel i at
    /// corner + i / width * size).
    pub fn rect(&self) -> ([f32; 2], [f32; 2]) {
        self.rect
    }

    /// The height texture's size (texels along x, z).
    pub fn height_dims(&self) -> [u32; 2] {
        [self.dims[0], self.dims[2]]
    }

    /// The mesh for the ray tracing grid (`Renderable::rt_mesh`).
    /// The grid's mesh: its vertices along x and z (`rt_spacing` apart).
    pub fn mesh_dims(&self) -> [u32; 2] {
        self.mesh_dims
    }

    pub fn rt_mesh(&self) -> &RtGpuMesh {
        &self.mesh
    }

    /// The parameters' uniform (for passes that read the heights with the shader's layout).
    pub fn params_buffer(&self) -> &wgpu::Buffer {
        &self.params
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_shaders_validate_and_the_params_match() {
        let code = format!("{}\n{}\n{}", include_str!("../../rt/shaders/rt_types.wgsl"), super::WGSL, super::WATER_RIPPLES_WGSL);
        let module = naga::front::wgsl::parse_str(&code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == "HeightfieldParams" => Some(*span as usize),
            _ => None,
        });
        assert_eq!(span, Some(std::mem::size_of::<super::Params>()));
    }
}
