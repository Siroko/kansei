use bytemuck::{Pod, Zeroable};

use crate::cameras::Camera;

const WGSL: &str = concat!(include_str!("../shaders/spot_light_types.wgsl"), include_str!("../shaders/light_clusters.wgsl"));

/// Clusters: 16 x 9 screen tiles x 24 exponential depth slices.
pub const CLUSTER_GRID: [u32; 3] = [16, 9, 24];
/// Per cluster: the light count, then up to 31 light indices (`KANSEI_CLUSTER_SLOTS`).
pub(crate) const CLUSTER_SLOTS: u32 = 32;
/// Depth the slices reach at most; beyond it, fragments use the last slice.
const MAX_CLUSTER_FAR: f32 = 2000.0;

/// `KanseiClusterParams` in `spot_light_types.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ClusterParamsGpu {
    view: [f32; 16],
    inv_proj: [f32; 16],
    screen: [f32; 2],
    near: f32,
    far: f32,
    grid: [u32; 3],
    enabled: u32,
}

/// The renderer's clustered light lists (group 3, bindings 8-9), rebuilt every frame for the
/// camera by a compute pass, so materials shade each fragment with the lights that reach it.
pub(crate) struct LightClusters {
    pub params: wgpu::Buffer,
    pub lights: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl LightClusters {
    pub fn new(device: &wgpu::Device, spot_lights: &wgpu::Buffer) -> Self {
        let clusters = CLUSTER_GRID.iter().product::<u32>();
        let params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightClusters/Params"),
            size: std::mem::size_of::<ClusterParamsGpu>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let lights = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("LightClusters/Lights"),
            size: (clusters * CLUSTER_SLOTS * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("LightClusters/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, storage(true)),
                entry(2, storage(false)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("LightClusters"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("LightClusters"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("LightClusters/Build"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("LightClusters/BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: spot_lights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: lights.as_entire_binding() },
            ],
        });
        Self { params, lights, pipeline, bind_group }
    }

    pub(crate) fn params_for(camera: &Camera, width: u32, height: u32) -> ClusterParamsGpu {
        ClusterParamsGpu {
            view: camera.view_matrix.data,
            inv_proj: camera.jittered_projection().to_glam().inverse().to_cols_array(),
            screen: [width as f32, height as f32],
            near: camera.near.max(1e-3),
            far: camera.far.min(MAX_CLUSTER_FAR).max(camera.near * 2.0),
            grid: CLUSTER_GRID,
            enabled: 1,
        }
    }

    /// Shade with every light: for views the clusters aren't built for (planar reflections).
    /// Takes effect for the command buffers submitted after it.
    pub fn disable(&self, queue: &wgpu::Queue) {
        let params = ClusterParamsGpu { grid: CLUSTER_GRID, near: 1.0, far: 2.0, ..Zeroable::zeroed() };
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&params));
    }

    /// Build the clusters for `camera` (after the frame's lights are uploaded, before the passes
    /// that shade with them).
    pub fn build(&self, device: &wgpu::Device, queue: &wgpu::Queue, camera: &Camera, width: u32, height: u32) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&Self::params_for(camera, width, height)));
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("LightClusters") });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("LightClusters/Build"), timestamp_writes: crate::profiling::gpu_pass("LightClusters/Build").as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(CLUSTER_GRID.iter().product::<u32>().div_ceil(64), 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    #[cfg(test)]
    pub(crate) fn shader_source() -> &'static str {
        WGSL
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lights::spot_lights_gpu::SpotLightsGpu;
    use crate::lights::{Light, SpotLight};
    use crate::math::Vec3;

    #[test]
    fn shader_validates_and_params_layout_matches() {
        let code = LightClusters::shader_source();
        let module = naga::front::wgsl::parse_str(code).unwrap_or_else(|e| panic!("{}", e.emit_to_string(code)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "KanseiClusterParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<ClusterParamsGpu>());
    }

    /// The cluster a view-space point falls in, as `kansei_light_cluster` computes it.
    fn cluster_of(p: &ClusterParamsGpu, camera: &Camera, point: glam::Vec3) -> u32 {
        let clip = camera.jittered_projection().to_glam() * point.extend(1.0);
        let uv = glam::Vec2::new(clip.x / clip.w * 0.5 + 0.5, 0.5 - clip.y / clip.w * 0.5);
        let slice = ((-point.z).max(p.near) / p.near).ln() / (p.far / p.near).ln();
        let z = ((slice.max(0.0) * p.grid[2] as f32) as u32).min(p.grid[2] - 1);
        let tx = ((uv.x * p.grid[0] as f32) as u32).min(p.grid[0] - 1);
        let ty = ((uv.y * p.grid[1] as f32) as u32).min(p.grid[1] - 1);
        (z * p.grid[1] + ty) * p.grid[0] + tx
    }

    /// On a real GPU: a spot in front of the camera is listed in the cluster around it, one
    /// behind the camera nowhere, and one aiming away is not listed where it doesn't reach.
    #[test]
    fn gpu_builds_cluster_lists() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        let spot = |pos: Vec3, dir: Vec3, range: f32, outer: f32| {
            Light::Spot(SpotLight::new(pos, dir, Vec3::new(1.0, 1.0, 1.0), 100.0, range, outer * 0.5, outer))
        };
        let lights = [
            spot(Vec3::new(0.0, 0.0, -10.0), Vec3::new(0.0, 0.0, -1.0), 2.0, 0.8), // 0: ahead
            spot(Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, 1.0), 2.0, 0.8),   // 1: behind the camera
            spot(Vec3::new(5.0, 0.0, -10.0), Vec3::new(1.0, 0.0, 0.0), 8.0, 0.35), // 2: aiming away, +x
        ];
        let mut packed = SpotLightsGpu::new();
        packed.pack(lights.iter(), 0, 0);
        use wgpu::util::DeviceExt;
        let spot_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: packed.as_bytes(),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let clusters = LightClusters::new(&device, &spot_buf);
        let camera = Camera::new(60.0, 0.1, 100.0, 16.0 / 9.0); // at the origin, looking down -z
        clusters.build(&device, &queue, &camera, 1600, 900);

        let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: clusters.lights.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&clusters.lights, 0, &staging, 0, staging.size());
        queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let words: Vec<u32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
        let list = |c: u32| -> Vec<u32> {
            let base = (c * CLUSTER_SLOTS) as usize;
            words[base + 1..base + 1 + words[base] as usize].to_vec()
        };
        let params = LightClusters::params_for(&camera, 1600, 900);
        let clusters_total = CLUSTER_GRID.iter().product::<u32>();

        // just in front of light 0, inside its cone
        let ahead = cluster_of(&params, &camera, glam::Vec3::new(0.0, 0.0, -11.0));
        assert!(list(ahead).contains(&0), "{:?}", list(ahead));
        // light 1 is behind the camera: in no cluster
        assert!((0..clusters_total).all(|c| !list(c).contains(&1)));
        // light 2 reaches +x of itself only: listed in its cone, and not 4 m behind it, which is
        // inside its range sphere (clusters there are ~1.3 x 1.3 x 2.9 m, so only the cone test
        // can drop it)
        let in_cone = cluster_of(&params, &camera, glam::Vec3::new(8.0, 0.0, -10.0));
        assert!(list(in_cone).contains(&2), "{:?}", list(in_cone));
        let behind_it = cluster_of(&params, &camera, glam::Vec3::new(1.0, 0.0, -10.0));
        assert!(!list(behind_it).contains(&2), "{:?}", list(behind_it));
        // far from everything: empty
        let empty = cluster_of(&params, &camera, glam::Vec3::new(-20.0, 5.0, -60.0));
        assert!(list(empty).is_empty(), "{:?}", list(empty));
    }
}
