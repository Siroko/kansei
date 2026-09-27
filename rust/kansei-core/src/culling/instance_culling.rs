use bytemuck::{Pod, Zeroable};

const WGSL: &str = include_str!("../shaders/instance_cull.wgsl");
const NO_WORD: u32 = u32::MAX;

/// World-space frustum planes of a `[0, 1]`-depth view-projection (Gribb & Hartmann), as
/// `(normal, d)` with `normal · p + d >= 0` inside, normals unit length. Order: left, right,
/// bottom, top, near, far.
pub fn frustum_planes(view_proj: glam::Mat4) -> [glam::Vec4; 6] {
    let (r0, r1, r2, r3) = (view_proj.row(0), view_proj.row(1), view_proj.row(2), view_proj.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2].map(|p| p / p.truncate().length().max(1e-12))
}

/// Per-view GPU culling for an instanced renderable (set `Renderable::instance_culling`).
///
/// `source` holds every instance, never culled: the per-instance vertex data of the geometry's
/// (single) instance buffer, `stride` bytes each. Every frame the renderer culls it on the GPU
/// for each view that draws the renderable: the main camera, and each shadowed spot light's own
/// frustum, so things outside the picture still cast into it. An instance is kept when its
/// bounding sphere (centre at `center_offset`, `radius` times the f32 at
/// `radius_scale_offset` if set, both in the renderable's object space) is inside the view, and
/// its distance from the main camera is within `lod_range`. The survivors are compacted into a
/// buffer per view and drawn indirectly; the geometry's own instance buffer only lends its
/// vertex layout.
///
/// LOD: one renderable per LOD mesh, all sharing `source`, each with its distance band. Bands
/// are measured from the main camera in every view, so shadows match what is on screen.
///
/// ```ignore
/// // 32-byte instances: position xyz + height, then yaw, ...; spheres of 0.6 x height
/// let culling = InstanceCulling::new(all_trees.clone(), tree_count, 32, 0, 0.6)
///     .with_radius_scale(12)
///     .with_lod_range(0.0, 60.0);
/// lod0.instance_culling = Some(culling);
/// ```
pub struct InstanceCulling {
    /// All instances; needs `BufferUsages::STORAGE`.
    pub source: wgpu::Buffer,
    /// Instances in `source` (at most the count it was created with).
    pub count: u32,
    /// Bytes per instance, a multiple of 4.
    pub stride: u32,
    /// Byte offset of the bounding sphere's centre (3 x f32) within an instance.
    pub center_offset: u32,
    /// Bounding radius, object space.
    pub radius: f32,
    /// Byte offset of an f32 the radius is multiplied by (a per-instance scale or height).
    pub radius_scale_offset: Option<u32>,
    /// Distances from the main camera at which the instances draw here: [near, far).
    pub lod_range: (f32, f32),
    capacity: u32,
    views: Vec<ViewSlot>,
}

struct ViewSlot {
    instances: wgpu::Buffer,
    args: wgpu::Buffer,
    params: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct CullParamsGpu {
    planes: [[f32; 4]; 6],
    world: [f32; 16],
    lod_origin: [f32; 3],
    lod_near: f32,
    lod_far: f32,
    radius: f32,
    max_scale: f32,
    count: u32,
    stride_words: u32,
    center_word: u32,
    scale_word: u32,
    _pad: u32,
}

/// A view the renderer culls for: its view-projection, and whether it only draws shadow casters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CullView {
    pub view_proj: glam::Mat4,
    pub casters_only: bool,
}

impl InstanceCulling {
    pub fn new(source: wgpu::Buffer, count: u32, stride: u32, center_offset: u32, radius: f32) -> Self {
        assert!(stride.is_multiple_of(4) && center_offset.is_multiple_of(4) && center_offset + 12 <= stride, "instance layout must be 4-byte words, with the centre inside");
        Self {
            source,
            count,
            stride,
            center_offset,
            radius,
            radius_scale_offset: None,
            lod_range: (0.0, f32::INFINITY),
            capacity: count,
            views: Vec::new(),
        }
    }

    pub fn with_radius_scale(mut self, offset: u32) -> Self {
        assert!(offset.is_multiple_of(4) && offset + 4 <= self.stride);
        self.radius_scale_offset = Some(offset);
        self
    }

    pub fn with_lod_range(mut self, near: f32, far: f32) -> Self {
        self.lod_range = (near, far);
        self
    }

    /// The compacted instances and indirect draw of `view` (0 is the main camera), once culled.
    pub(crate) fn view(&self, view: usize) -> Option<(&wgpu::Buffer, &wgpu::Buffer)> {
        self.views.get(view).map(|v| (&v.instances, &v.args))
    }

    /// Make sure there are GPU slots for `count` views; true if buffers were (re)created (render
    /// bundles that recorded the old ones are stale).
    pub(crate) fn ensure_views(&mut self, device: &wgpu::Device, bgl: &wgpu::BindGroupLayout, count: usize) -> bool {
        let mut recreated = false;
        if self.count > self.capacity {
            self.capacity = self.count;
            self.views.clear();
            recreated = true;
        }
        while self.views.len() < count {
            let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| {
                device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false })
            };
            // (COPY_SRC: readable for debugging and tests)
            let instances = buffer("InstanceCulling/Instances", (self.capacity.max(1) * self.stride) as u64, wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let args = buffer("InstanceCulling/Args", 20, wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC);
            let params = buffer("InstanceCulling/Params", std::mem::size_of::<CullParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM);
            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("InstanceCulling/BG"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: params.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: self.source.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: instances.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: args.as_entire_binding() },
                ],
            });
            self.views.push(ViewSlot { instances, args, params, bind_group });
            recreated = true;
        }
        recreated
    }

    pub(crate) fn params(&self, view: &CullView, world: glam::Mat4, lod_origin: glam::Vec3) -> CullParamsGpu {
        let scale = glam::Vec3::new(world.x_axis.truncate().length(), world.y_axis.truncate().length(), world.z_axis.truncate().length());
        CullParamsGpu {
            planes: frustum_planes(view.view_proj).map(|p| p.to_array()),
            world: world.to_cols_array(),
            lod_origin: lod_origin.to_array(),
            lod_near: self.lod_range.0,
            lod_far: self.lod_range.1.min(f32::MAX),
            radius: self.radius,
            max_scale: scale.max_element(),
            count: self.count.min(self.capacity),
            stride_words: self.stride / 4,
            center_word: self.center_offset / 4,
            scale_word: self.radius_scale_offset.map_or(NO_WORD, |o| o / 4),
            _pad: 0,
        }
    }

    /// Cull for `view` into its slot: reset the draw, write the parameters, dispatch.
    pub(crate) fn dispatch(&self, queue: &wgpu::Queue, pass: &mut wgpu::ComputePass, slot: usize, params: &CullParamsGpu, index_count: u32) {
        let v = &self.views[slot];
        queue.write_buffer(&v.args, 0, bytemuck::cast_slice(&[index_count, 0, 0, 0, 0u32]));
        queue.write_buffer(&v.params, 0, bytemuck::bytes_of(params));
        pass.set_bind_group(0, &v.bind_group, &[]);
        pass.dispatch_workgroups(params.count.div_ceil(64).max(1), 1, 1);
    }
}

/// The cull compute pipeline, shared by every renderable.
pub(crate) struct CullPipeline {
    pub pipeline: wgpu::ComputePipeline,
    pub bgl: wgpu::BindGroupLayout,
}

impl CullPipeline {
    pub fn new(device: &wgpu::Device) -> Self {
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None };
        let storage = |read_only| wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("InstanceCulling/BGL"),
            entries: &[
                entry(0, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, storage(false)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("InstanceCulling"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("InstanceCulling"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("InstanceCulling"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self { pipeline, bgl }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_planes_classify_points() {
        let view = glam::Mat4::look_at_rh(glam::Vec3::new(0.0, 0.0, 10.0), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = glam::Mat4::perspective_rh(1.0, 1.0, 0.1, 100.0);
        let planes = frustum_planes(proj * view);
        let inside = |p: glam::Vec3, r: f32| planes.iter().all(|pl| pl.truncate().dot(p) + pl.w >= -r);
        assert!(inside(glam::Vec3::ZERO, 0.0));
        assert!(!inside(glam::Vec3::new(0.0, 0.0, 20.0), 0.0), "behind the camera");
        assert!(!inside(glam::Vec3::new(0.0, 0.0, -200.0), 0.0), "beyond the far plane");
        assert!(!inside(glam::Vec3::new(30.0, 0.0, 0.0), 1.0), "off to the side");
        // a sphere straddling the left plane is kept
        let edge_x = 10.0 * 0.5f32.tan();
        assert!(inside(glam::Vec3::new(-edge_x - 0.5, 0.0, 0.0), 1.0));
        assert!(!inside(glam::Vec3::new(-edge_x - 2.0, 0.0, 0.0), 1.0));
        // planes are normalized: distances are metres
        for p in planes {
            assert!((p.truncate().length() - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn shader_validates_and_params_layout_matches() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        let span = module
            .types
            .iter()
            .find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == "CullParams" => Some(*span as usize),
                _ => None,
            })
            .unwrap();
        assert_eq!(span, std::mem::size_of::<CullParamsGpu>());
    }

    /// On a real GPU: of a row of instances along x, a view keeps exactly those in its frustum
    /// and LOD band, copies their whole records, and counts them into the indirect draw.
    #[test]
    fn gpu_culls_and_compacts_per_view() {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();
        // 8-word instances: centre xyz, scale, then an id and padding
        let mut data: Vec<f32> = Vec::new();
        for i in 0..100 {
            data.extend_from_slice(&[i as f32 - 50.0, 0.0, 0.0, 1.0, i as f32, 0.0, 0.0, 0.0]);
        }
        use wgpu::util::DeviceExt;
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pipeline = CullPipeline::new(&device);
        let mut culling = InstanceCulling::new(source, 100, 32, 0, 0.5).with_radius_scale(12).with_lod_range(0.0, 30.0);
        culling.ensure_views(&device, &pipeline.bgl, 2);
        // view 0: looking down -z from z = 20, 90 degrees wide: sees |x| < 20 at z = 0
        // view 1: the same from x = 40, sees 20 < x < 60 but the LOD band (from the origin) stops at 30
        let proj = glam::Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let views = [0.0f32, 40.0].map(|x| CullView {
            view_proj: proj * glam::Mat4::look_at_rh(glam::Vec3::new(x, 0.0, 20.0), glam::Vec3::new(x, 0.0, 0.0), glam::Vec3::Y),
            casters_only: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline.pipeline);
            for (slot, view) in views.iter().enumerate() {
                let params = culling.params(view, glam::Mat4::IDENTITY, glam::Vec3::ZERO);
                culling.dispatch(&queue, &mut pass, slot, &params, 36);
            }
        }
        let readback = |buf: &wgpu::Buffer, encoder: &mut wgpu::CommandEncoder| {
            let staging = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: buf.size(), usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
            encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, buf.size());
            staging
        };
        let staged: Vec<_> = (0..2)
            .flat_map(|v| {
                let (instances, args) = culling.view(v).unwrap();
                [readback(args, &mut encoder), readback(instances, &mut encoder)]
            })
            .collect();
        queue.submit(Some(encoder.finish()));
        for s in &staged {
            s.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        }
        device.poll(wgpu::Maintain::Wait);
        let words = |b: &wgpu::Buffer| -> Vec<u32> { bytemuck::cast_slice(&b.slice(..).get_mapped_range()).to_vec() };
        let ids = |args: &[u32], inst: &[u32]| -> Vec<i32> {
            let mut v: Vec<i32> = (0..args[1] as usize).map(|k| f32::from_bits(inst[k * 8 + 4]) as i32 - 50).collect();
            v.sort();
            v
        };
        let (a0, i0, a1, i1) = (words(&staged[0]), words(&staged[1]), words(&staged[2]), words(&staged[3]));
        assert_eq!(a0[0], 36, "index count");
        // |x| <= 20 (spheres of 0.5 straddle the planes at +-20.x)
        assert_eq!(ids(&a0, &i0), (-20..=20).collect::<Vec<_>>());
        // 20 <= x < 30: in view 1's frustum and within 30 of the LOD origin
        assert_eq!(ids(&a1, &i1), (20..=29).collect::<Vec<_>>());
    }
}
