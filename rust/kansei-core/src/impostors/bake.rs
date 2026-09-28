use bytemuck::{Pod, Zeroable};

use super::{frame_direction, up_reference, Impostor, ImpostorOptions};
use crate::cameras::camera::CameraTemporalGpu;
use crate::materials::PipelineKey;
use crate::objects::Scene;
use crate::renderers::{GBuffer, Renderer};

const PACK_WGSL: &str = include_str!("../shaders/impostor_pack.wgsl");
const MIP_WGSL: &str = include_str!("../shaders/impostor_mip.wgsl");

/// `Pack` in impostor_pack.wgsl: one frame.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(super) struct PackGpu {
    right: [f32; 3],
    radius: f32,
    up: [f32; 3],
    frame_size: u32,
    dir: [f32; 3],
    supersample: u32,
    frame: [u32; 2],
    _pad: [u32; 2],
}

// Per-frame uniforms (the view and camera temporal data, the pack parameters) sit this far apart:
// a multiple of every device's uniform offset alignment.
const SLOT: u64 = 256;

impl Renderer {
    /// Bake an octahedral impostor of the renderables `parts` (scene indices), drawn together
    /// with their own materials' GBuffer pipelines, from `options.frames` squared directions
    /// (see `impostors`). Parts may be hidden; each has one instance buffer at most, drawn with
    /// `options.instance`. The lights and shadows bound are the renderer's (materials that
    /// light in the GBuffer pass only colour their colour output, which the bake keeps only where
    /// they write no albedo).
    pub fn bake_impostor(&mut self, scene: &mut Scene, parts: &[usize], options: &ImpostorOptions) -> Impostor {
        let _t = crate::profiling::cpu_scope("impostor/bake");
        assert!(options.frames > 0 && options.frame_size.is_power_of_two() && options.supersample > 0, "frames, a power-of-two frame size and supersampling");
        for &i in parts {
            let r = scene.get_renderable_mut(i).expect("impostor part: a renderable's scene index");
            assert!(r.geometry.instance_buffers.len() <= 1, "impostor parts have one instance buffer at most");
            self.prepare_for_gbuffer(r, 1);
        }
        let (center, radius, extent) = match options.bounds {
            Some((lo, hi)) => ((lo + hi) * 0.5, ((hi - lo) * 0.5).length().max(1e-6), (hi - lo) * 0.5),
            None => bounds(parts.iter().filter_map(|&i| scene.get_renderable(i)).flat_map(|r| r.geometry.vertices.iter())),
        };
        let device = self.device();
        let queue = self.queue();
        let shared = self.shared_layouts();
        let (n, size, ss) = (options.frames, options.frame_size, options.supersample);
        let frame_count = (n * n) as u64;

        // the atlases, with a mip chain down to a texel per frame
        let levels = size.trailing_zeros() + 1;
        let atlas = |label| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: n * size, height: n * size, depth_or_array_layers: 1 },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: Impostor::FORMAT,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        let (albedo, normal_depth) = (atlas("Impostor/Albedo"), atlas("Impostor/NormalDepth"));
        let mip = |t: &wgpu::Texture, level| t.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() });

        // a frame's render targets: the GBuffer's, `supersample` times the frame's size
        let render_size = size * ss;
        let target = |label, format| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: render_size, height: render_size, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let targets: Vec<_> = GBuffer::MRT_FORMATS.iter().map(|&f| target("Impostor/Target", f)).collect();
        let depth = target("Impostor/Depth", GBuffer::DEPTH_FORMAT);

        // per frame: an orthographic camera on the bounding sphere looking at its centre, depth
        // [0, 1] over its diameter; and the pack parameters
        let proj = glam::Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, 2.0 * radius);
        let mut cameras = vec![0u8; (frame_count * 2 * SLOT) as usize];
        let mut packs = vec![0u8; (frame_count * SLOT) as usize];
        for k in 0..frame_count as u32 {
            let (column, row) = (k % n, k / n);
            let dir = frame_direction(n, options.layout, column, row);
            let view = glam::Mat4::look_at_rh(center + dir * radius, center, up_reference(dir));
            let view_proj = (proj * view).to_cols_array();
            let temporal = CameraTemporalGpu { view_proj, prev_view_proj: view_proj, jitter: [0.0; 2], prev_jitter: [0.0; 2], frame: 0, _pad: [0; 3] };
            let at = (k as u64 * 2 * SLOT) as usize;
            cameras[at..at + 64].copy_from_slice(bytemuck::cast_slice(&view.to_cols_array()));
            cameras[at + SLOT as usize..][..std::mem::size_of::<CameraTemporalGpu>()].copy_from_slice(bytemuck::bytes_of(&temporal));
            let right = up_reference(dir).cross(dir).normalize();
            let pack = PackGpu {
                right: right.to_array(),
                radius,
                up: dir.cross(right).to_array(),
                frame_size: size,
                dir: dir.to_array(),
                supersample: ss,
                frame: [column, row],
                _pad: [0; 2],
            };
            packs[(k as u64 * SLOT) as usize..][..std::mem::size_of::<PackGpu>()].copy_from_slice(bytemuck::bytes_of(&pack));
        }
        let uniform = |label, contents: &[u8]| {
            use wgpu::util::DeviceExt;
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::UNIFORM })
        };
        let cameras = uniform("Impostor/Cameras", &cameras);
        let projection = uniform("Impostor/Projection", bytemuck::cast_slice(&proj.to_cols_array()));
        let packs = uniform("Impostor/Pack", &packs);
        // the object in place: identity normal and world matrices (this frame's and last)
        let identity = glam::Mat4::IDENTITY.to_cols_array();
        let normal_matrix = uniform("Impostor/NormalMatrix", bytemuck::cast_slice(&identity));
        let world = uniform("Impostor/World", bytemuck::cast_slice(&[identity, identity]));
        let range = |buffer, offset, size| wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer, offset, size: std::num::NonZeroU64::new(size) });
        let mesh_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Impostor/Mesh"),
            layout: &shared.mesh_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: range(&normal_matrix, 0, 64) }, wgpu::BindGroupEntry { binding: 1, resource: range(&world, 0, 128) }],
        });
        let camera_groups: Vec<_> = (0..frame_count)
            .map(|k| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Impostor/Camera"),
                    layout: &shared.camera_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: range(&cameras, k * 2 * SLOT, 64) },
                        wgpu::BindGroupEntry { binding: 1, resource: projection.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: self.light_buffer().as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: range(&cameras, k * 2 * SLOT + SLOT, std::mem::size_of::<CameraTemporalGpu>() as u64) },
                    ],
                })
            })
            .collect();
        // each part's instance: the given record, in a vertex buffer of its own
        let instances: Vec<Option<wgpu::Buffer>> = parts
            .iter()
            .map(|&i| {
                let r = scene.get_renderable(i).unwrap();
                r.geometry.instance_buffers.first().and_then(|cb| cb.vertex_layout()).map(|layout| {
                    assert!(options.instance.len() as u64 >= layout.stride, "ImpostorOptions::instance: one record of the parts' instance layout ({} bytes)", layout.stride);
                    use wgpu::util::DeviceExt;
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("Impostor/Instance"), contents: &options.instance, usage: wgpu::BufferUsages::VERTEX })
                })
            })
            .collect();

        let pack_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Impostor/Pack"), source: wgpu::ShaderSource::Wgsl(PACK_WGSL.into()) });
        let pack_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Impostor/Pack"),
            layout: None,
            module: &pack_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (albedo_mip0, normal_depth_mip0) = (mip(&albedo, 0), mip(&normal_depth, 0));
        let view = wgpu::BindingResource::TextureView;
        let pack_groups: Vec<_> = (0..frame_count)
            .map(|k| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Impostor/Pack"),
                    layout: &pack_pipeline.get_bind_group_layout(0),
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: view(&targets[0]) },
                        wgpu::BindGroupEntry { binding: 1, resource: view(&targets[2]) },
                        wgpu::BindGroupEntry { binding: 2, resource: view(&targets[3]) },
                        wgpu::BindGroupEntry { binding: 3, resource: view(&depth) },
                        wgpu::BindGroupEntry { binding: 4, resource: view(&albedo_mip0) },
                        wgpu::BindGroupEntry { binding: 5, resource: view(&normal_depth_mip0) },
                        wgpu::BindGroupEntry { binding: 6, resource: range(&packs, k * SLOT, std::mem::size_of::<PackGpu>() as u64) },
                    ],
                })
            })
            .collect();

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Impostor/Bake") });
        let clear = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);
        for k in 0..frame_count as usize {
            {
                let color_attachments: Vec<_> = targets
                    .iter()
                    .map(|view| Some(wgpu::RenderPassColorAttachment { view, resolve_target: None, ops: wgpu::Operations { load: clear, store: wgpu::StoreOp::Store } }))
                    .collect();
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Impostor/Frame"),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                        view: &depth,
                        depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                        stencil_ops: None,
                    }),
                    ..Default::default()
                });
                pass.set_bind_group(1, &camera_groups[k], &[]);
                pass.set_bind_group(2, &mesh_group, &[0, 0]);
                pass.set_bind_group(3, self.shadow_bind_group(), &[]);
                for (&i, instance) in parts.iter().zip(&instances) {
                    let r = scene.get_renderable(i).unwrap();
                    let key = PipelineKey {
                        color_formats: GBuffer::MRT_FORMATS.to_vec(),
                        depth_format: GBuffer::DEPTH_FORMAT,
                        sample_count: 1,
                        num_vertex_buffers: 1 + r.geometry.instance_buffers.len(),
                    };
                    let (Some(pipeline), Some(vertices), Some(indices)) = (r.material.pipeline_cache.get(&key), r.geometry.active_vertex_buffer(), r.geometry.active_index_buffer()) else { continue };
                    pass.set_pipeline(pipeline);
                    if let Some(bg) = r.material.bind_group() {
                        pass.set_bind_group(0, bg, &[]);
                    }
                    pass.set_vertex_buffer(0, vertices.slice(..));
                    if let Some(instance) = instance {
                        pass.set_vertex_buffer(1, instance.slice(..));
                    }
                    pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                    pass.draw_indexed(0..r.geometry.index_count(), 0, 0..1);
                }
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Impostor/Pack"), timestamp_writes: None });
            pass.set_pipeline(&pack_pipeline);
            pass.set_bind_group(0, &pack_groups[k], &[]);
            pass.dispatch_workgroups(size.div_ceil(8), size.div_ceil(8), 1);
        }

        // the mip chain
        let mip_module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Impostor/Mip"), source: wgpu::ShaderSource::Wgsl(MIP_WGSL.into()) });
        let mip_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Impostor/Mip"),
            layout: None,
            module: &mip_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        for level in 1..levels {
            let views = [mip(&albedo, level - 1), mip(&normal_depth, level - 1), mip(&albedo, level), mip(&normal_depth, level)];
            let entries: Vec<_> = views.iter().enumerate().map(|(binding, v)| wgpu::BindGroupEntry { binding: binding as u32, resource: wgpu::BindingResource::TextureView(v) }).collect();
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("Impostor/Mip"), layout: &mip_pipeline.get_bind_group_layout(0), entries: &entries });
            let side = (n * size) >> level;
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Impostor/Mip"), timestamp_writes: None });
            pass.set_pipeline(&mip_pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(side.div_ceil(8), side.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()));

        Impostor { frames: n, frame_size: size, layout: options.layout, center, radius, extent, albedo, normal_depth }
    }
}

/// The box round `vertices`' positions: its centre, the distance of the farthest from it, and its
/// half size.
fn bounds<'a>(vertices: impl Iterator<Item = &'a crate::geometries::Vertex> + Clone) -> (glam::Vec3, f32, glam::Vec3) {
    let point = |v: &crate::geometries::Vertex| glam::Vec3::new(v.position[0], v.position[1], v.position[2]);
    let (lo, hi) = vertices.clone().fold((glam::Vec3::splat(f32::MAX), glam::Vec3::splat(f32::MIN)), |(lo, hi), v| (lo.min(point(v)), hi.max(point(v))));
    assert!(lo.x <= hi.x, "impostor parts have vertices");
    let center = (lo + hi) * 0.5;
    let radius = vertices.map(|v| point(v).distance(center)).fold(0.0f32, f32::max);
    (center, radius.max(1e-6), (hi - lo) * 0.5)
}
