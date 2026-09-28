//! The screen-space path of `PlanarReflection` (`PlanarReflection::screen_space`): last frame's
//! GBuffer projected across the plane into the reflection texture. See the WGSL.

use bytemuck::{Pod, Zeroable};

use super::planar_reflection::ReflectionFog;

pub(crate) const WGSL: &str = concat!(include_str!("../shaders/froxel_common.wgsl"), include_str!("../shaders/planar_reflection_screen_space.wgsl"));

/// The WGSL `Params`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct ScreenSpaceParamsGpu {
    pub prev_inv_view_proj: [f32; 16],
    pub view_proj: [f32; 16],
    pub plane: [f32; 4],
    pub camera_pos: [f32; 3],
    pub min_height: f32,
    pub src_size: [u32; 2],
    pub dst_size: [u32; 2],
    /// Screen uv the reflection is needed in (x0, y0, x1, y1); empty: nothing is projected.
    pub rect: [f32; 4],
}

/// Pipelines and per-texel buffers of the projection, for a reflection texture of one size.
pub(crate) struct ScreenSpaceProjection {
    clear: wgpu::ComputePipeline,
    project: wgpu::ComputePipeline,
    own: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    heights: wgpu::Buffer,
    owners: wgpu::Buffer,
    params: wgpu::Buffer,
}

impl ScreenSpaceProjection {
    pub(crate) fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let texture = |sample_type, view_dimension| wgpu::BindingType::Texture { sample_type, view_dimension, multisampled: false };
        let storage = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("PlanarReflection/ScreenSpaceBGL"),
            entries: &[
                entry(0, texture(wgpu::TextureSampleType::Float { filterable: false }, wgpu::TextureViewDimension::D2)),
                entry(1, texture(wgpu::TextureSampleType::Depth, wgpu::TextureViewDimension::D2)),
                entry(2, storage),
                entry(3, storage),
                entry(4, wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                }),
                entry(5, uniform),
                entry(6, texture(wgpu::TextureSampleType::Float { filterable: true }, wgpu::TextureViewDimension::D3)),
                entry(7, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering)),
                entry(8, uniform),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("PlanarReflection/ScreenSpace"), source: wgpu::ShaderSource::Wgsl(WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("PlanarReflection/ScreenSpace"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PlanarReflection/ScreenSpace"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let buffer = |label, size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let texels = (width * height) as u64 * 4;
        Self {
            clear: pipeline("clear"),
            project: pipeline("project"),
            own: pipeline("own"),
            resolve: pipeline("resolve"),
            heights: buffer("PlanarReflection/ScreenSpaceHeights", texels, wgpu::BufferUsages::STORAGE),
            owners: buffer("PlanarReflection/ScreenSpaceOwners", texels, wgpu::BufferUsages::STORAGE),
            params: buffer("PlanarReflection/ScreenSpaceParams", std::mem::size_of::<ScreenSpaceParamsGpu>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST),
            bgl,
        }
    }

    /// Project `color` and `depth` (last frame's, `params.src_size`) into `dst` (mip 0 of the
    /// reflection, `params.dst_size`), with `fog` composited over what it reflects.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        dst: &wgpu::TextureView,
        fog: &ReflectionFog,
        fog_sampler: &wgpu::Sampler,
        params: &ScreenSpaceParamsGpu,
    ) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(params));
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PlanarReflection/ScreenSpaceBG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(color) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(depth) },
                wgpu::BindGroupEntry { binding: 2, resource: self.heights.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.owners.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::TextureView(dst) },
                wgpu::BindGroupEntry { binding: 5, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: wgpu::BindingResource::TextureView(&fog.volume) },
                wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::Sampler(fog_sampler) },
                wgpu::BindGroupEntry { binding: 8, resource: fog.params.as_entire_binding() },
            ],
        });
        let [sw, sh] = params.src_size;
        let [dw, dh] = params.dst_size;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("PlanarReflection/ScreenSpace"),
            timestamp_writes: crate::profiling::gpu_pass("PlanarReflection/ScreenSpace").as_ref().map(crate::profiling::PassStamp::compute),
        });
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_pipeline(&self.clear);
        pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
        pass.set_pipeline(&self.project);
        pass.dispatch_workgroups(sw.div_ceil(8), sh.div_ceil(8), 1);
        pass.set_pipeline(&self.own);
        pass.dispatch_workgroups(sw.div_ceil(8), sh.div_ceil(8), 1);
        pass.set_pipeline(&self.resolve);
        pass.dispatch_workgroups(dw.div_ceil(8), dh.div_ceil(8), 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shader_validates_and_the_params_layout_matches() {
        let module = naga::front::wgsl::parse_str(WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap_or_else(|e| panic!("{e:?}"));
        let size = |name: &str| {
            module.types.iter().find_map(|(_, ty)| match (&ty.name, &ty.inner) {
                (Some(n), naga::TypeInner::Struct { span, .. }) if n == name => Some(*span as usize),
                _ => None,
            })
        };
        assert_eq!(size("Params"), Some(std::mem::size_of::<ScreenSpaceParamsGpu>()));
        assert_eq!(size("ReflectionFogParams"), Some(std::mem::size_of::<crate::reflections::ReflectionFogParamsGpu>()));
    }

    fn f16_to_f32(h: u16) -> f32 {
        let (sign, exp, frac) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as u32, (h & 0x3ff) as u32);
        let bits = match exp {
            0 if frac == 0 => sign << 31,
            0 => return (if sign == 1 { -1.0 } else { 1.0 }) * frac as f32 * 2f32.powi(-24),
            31 => (sign << 31) | 0x7f80_0000 | (frac << 13),
            _ => (sign << 31) | ((exp + 112) << 23) | (frac << 13),
        };
        f32::from_bits(bits)
    }

    /// A camera over water (the plane y = 0) facing a far wall and a nearer, lower post. Last
    /// frame's colour and depth (ray-cast here) projected across the plane give, in every texel,
    /// what the mirror shows there: the first surface along the reflected ray, with its colour and
    /// the reflected path length, where the camera saw that surface; sky where the reflected ray
    /// meets nothing. The water never reflects itself.
    #[test]
    fn the_screen_projected_across_the_plane_is_the_mirror_image() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let (device, queue) = pollster::block_on(adapter.request_device(&Default::default(), None)).unwrap();

        let eye = glam::Vec3::new(0.0, 2.0, 12.0);
        let view = glam::Mat4::look_at_rh(eye, glam::Vec3::new(0.0, 0.5, 0.0), glam::Vec3::Y);
        let view_proj = glam::Mat4::perspective_rh(0.9, 2.0, 0.1, 200.0) * view;
        let inv = view_proj.inverse();
        let (far_wall, post, water) = ([1.0f32, 0.5, 0.25], [0.0f32, 1.0, 0.0], [0.0f32, 0.0, 1.0]);
        // the first surface along a ray: its point and colour
        let trace = |o: glam::Vec3, d: glam::Vec3, water_too: bool| -> Option<(glam::Vec3, [f32; 3])> {
            let mut best: Option<(f32, [f32; 3])> = None;
            let mut hit = |t: f32, ok: bool, c: [f32; 3]| {
                if t > 1e-3 && ok && best.is_none_or(|(b, _)| t < b) {
                    best = Some((t, c));
                }
            };
            let t = (-10.0 - o.z) / d.z;
            let q = o + d * t;
            hit(t, q.x.abs() <= 30.0 && (0.0..=5.0).contains(&q.y), far_wall);
            let t = (-5.0 - o.z) / d.z;
            let q = o + d * t;
            hit(t, q.x.abs() <= 2.0 && (0.0..=3.0).contains(&q.y), post);
            if water_too {
                hit(-o.y / d.y, true, water);
            }
            best.map(|(t, c)| (o + d * t, c))
        };
        let ray = |uv: glam::Vec2| {
            let ndc = glam::Vec2::new(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
            let far = inv.project_point3(ndc.extend(1.0));
            (far - eye).normalize()
        };

        // last frame (the same view): colour and depth
        let (sw, sh) = (320u32, 160u32);
        let (mut colors, mut depths) = (vec![0.0f32; (sw * sh * 4) as usize], vec![1.0f32; (sw * sh) as usize]);
        for y in 0..sh {
            for x in 0..sw {
                let i = (y * sw + x) as usize;
                if let Some((p, c)) = trace(eye, ray((glam::Vec2::new(x as f32, y as f32) + 0.5) / glam::Vec2::new(sw as f32, sh as f32)), true) {
                    colors[i * 4..i * 4 + 3].copy_from_slice(&c);
                    depths[i] = view_proj.project_point3(p).z;
                }
            }
        }
        let size = |w, h| wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 };
        let texture = |w, h, format, usage| device.create_texture(&wgpu::TextureDescriptor { label: None, size: size(w, h), mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D2, format, usage, view_formats: &[] });
        let color = texture(sw, sh, wgpu::TextureFormat::Rgba32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &color, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            bytemuck::cast_slice(&colors),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(sw * 16), rows_per_image: Some(sh) },
            size(sw, sh),
        );
        let depth = texture(sw, sh, wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let depth_view = depth.create_view(&Default::default());
        let buf = { use wgpu::util::DeviceExt; device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&depths), usage: wgpu::BufferUsages::STORAGE }) };
        let code = format!("@group(0) @binding(0) var<storage, read> d : array<f32>;\n\
            @vertex fn vs(@builtin(vertex_index) i : u32) -> @builtin(position) vec4f {{ let p = vec2f(f32((i << 1u) & 2u), f32(i & 2u)); return vec4f(p * 2.0 - 1.0, 0.0, 1.0); }}\n\
            @fragment fn fs(@builtin(position) pos : vec4f) -> @builtin(frag_depth) f32 {{ return d[u32(pos.y) * {sw}u + u32(pos.x)]; }}");
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let fill = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: None,
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(wgpu::FragmentState { module: &module, entry_point: Some("fs"), targets: &[], compilation_options: Default::default() }),
            primitive: Default::default(),
            depth_stencil: Some(wgpu::DepthStencilState { format: wgpu::TextureFormat::Depth32Float, depth_write_enabled: true, depth_compare: wgpu::CompareFunction::Always, stencil: Default::default(), bias: Default::default() }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let fill_bg = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &fill.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buf.as_entire_binding() }] });

        // the reflection, at half the size
        let (dw, dh) = (sw / 2, sh / 2);
        let dst = texture(dw, dh, wgpu::TextureFormat::Rgba16Float, wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC);
        let fog = ReflectionFog {
            volume: device
                .create_texture(&wgpu::TextureDescriptor { label: None, size: size(1, 1), mip_level_count: 1, sample_count: 1, dimension: wgpu::TextureDimension::D3, format: wgpu::TextureFormat::Rgba16Float, usage: wgpu::TextureUsages::TEXTURE_BINDING, view_formats: &[] })
                .create_view(&Default::default()),
            params: device.create_buffer(&wgpu::BufferDescriptor { label: None, size: std::mem::size_of::<crate::reflections::ReflectionFogParamsGpu>() as u64, usage: wgpu::BufferUsages::UNIFORM, mapped_at_creation: false }),
            drawn: Default::default(),
        };
        let sampler = device.create_sampler(&Default::default());
        let projection = ScreenSpaceProjection::new(&device, dw, dh);
        let params = ScreenSpaceParamsGpu {
            prev_inv_view_proj: inv.to_cols_array(),
            view_proj: view_proj.to_cols_array(),
            plane: [0.0, 1.0, 0.0, 0.0],
            camera_pos: eye.to_array(),
            min_height: 0.02,
            src_size: [sw, sh],
            dst_size: [dw, dh],
            rect: [0.0, 0.0, 1.0, 1.0],
        };
        let row = (dw * 8).div_ceil(256) * 256;
        let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: (row * dh) as u64, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
        let mut e = device.create_command_encoder(&Default::default());
        {
            let mut pass = e.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment { view: &depth_view, depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }), stencil_ops: None }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&fill);
            pass.set_bind_group(0, &fill_bg, &[]);
            pass.draw(0..3, 0..1);
        }
        projection.run(&device, &queue, &mut e, &color.create_view(&Default::default()), &depth_view, &dst.create_view(&Default::default()), &fog, &sampler, &params);
        e.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture: &dst, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            wgpu::TexelCopyBufferInfo { buffer: &read, layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: Some(dh) } },
            size(dw, dh),
        );
        queue.submit([e.finish()]);
        read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        let data = read.slice(..).get_mapped_range();
        let texel = |x: u32, y: u32| -> [f32; 4] {
            // stored mirrored left-right, as materials read it
            let o = (y * row + (dw - 1 - x) * 8) as usize;
            std::array::from_fn(|c| f16_to_f32(u16::from_le_bytes([data[o + 2 * c], data[o + 2 * c + 1]])))
        };

        let (mut geometry, mut geometry_ok, mut sky, mut sky_ok) = (0, 0, 0, 0);
        for y in 0..dh {
            for x in 0..dw {
                let got = texel(x, y);
                assert!(got[..3] != water, "texel ({x}, {y}) reflects the water");
                // what the mirror shows here: the reflected ray from where the view ray meets the
                // water, where the camera saw that surface too (the screen has nothing else)
                let d = ray((glam::Vec2::new(x as f32, y as f32) + 0.5) / glam::Vec2::new(dw as f32, dh as f32));
                if d.y >= 0.0 {
                    continue;
                }
                let w = eye + d * (-eye.y / d.y);
                // (where the camera sees something else than the water, nothing reads the texel)
                if trace(eye, d, true).is_none_or(|(q, _)| q.distance(w) > 0.05) {
                    continue;
                }
                let reflected = glam::Vec3::new(d.x, -d.y, d.z);
                let hit = trace(w, reflected, false);
                // (texels an edge of what is reflected crosses hold either side)
                let near_edge = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)].iter().any(|&(ox, oy)| {
                    let d = ray((glam::Vec2::new(x as f32 + ox, y as f32 + oy)) / glam::Vec2::new(dw as f32, dh as f32));
                    let w = eye + d * (-eye.y / d.y);
                    trace(w, glam::Vec3::new(d.x, -d.y, d.z), false).map(|(_, c)| c) != hit.map(|(_, c)| c)
                });
                if near_edge {
                    continue;
                }
                match hit {
                    Some((p, c)) => {
                        let seen = trace(eye, (p - eye).normalize(), false).is_some_and(|(q, _)| q.distance(p) < 0.05);
                        if !seen {
                            continue;
                        }
                        geometry += 1;
                        let path = w.distance(eye) + p.distance(w);
                        if got[..3] == c && (got[3] - path).abs() < 0.02 * path {
                            geometry_ok += 1;
                        }
                    }
                    None => {
                        sky += 1;
                        if got[3] >= 65504.0 {
                            sky_ok += 1;
                        }
                    }
                }
            }
        }
        assert!(geometry > 400 && sky > 400, "the scene covers both: {geometry} geometry texels, {sky} sky");
        assert!(geometry_ok as f32 >= 0.99 * geometry as f32, "{geometry_ok} of {geometry} reflected texels match");
        assert!(sky_ok as f32 >= 0.99 * sky as f32, "{sky_ok} of {sky} sky texels match");
    }
}
