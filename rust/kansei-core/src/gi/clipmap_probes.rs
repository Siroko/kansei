use bytemuck::{Pod, Zeroable};
use glam::{IVec3, UVec3, Vec3};

use super::clipmap::{clipmap_entries, clipmap_layout_entries, ClipLevelGpu, ClipmapLayout, VoxelClipmap, MAX_CLIPMAP_LEVELS};

pub(crate) const CLIPMAP_PROBE_UPDATE_WGSL: &str = concat!(
    include_str!("shaders/clipmap.wgsl"),
    include_str!("shaders/particle_emission.wgsl"),
    include_str!("shaders/clipmap_probes.wgsl"),
    include_str!("../atmosphere/shaders/sky_lighting.wgsl"),
    include_str!("shaders/clipmap_probe_update.wgsl"),
);

/// What `SceneVoxelClipmap::enable_probes` sets up, and how the probes update
/// (`ClipmapProbes::options`; `levels`, `spacing_voxels` take effect at creation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ClipmapProbeOptions {
    /// Levels of probes, the clipmap's finest first (at most its levels).
    pub levels: u32,
    /// Voxels of the matching clipmap level between probes: each probe level spans that level's
    /// window with `dims / spacing_voxels` probes.
    pub spacing_voxels: u32,
    /// Probes traced each frame over all levels, in turn (the probes of slabs a window moves into
    /// are traced at once besides).
    pub probes_per_frame: u32,
    /// Weight of the history in each update (0.9: about ten updates to settle).
    pub hysteresis: f32,
    /// Scale of the sky the cones that leave the clipmap see.
    pub sky_scale: f32,
    /// Probe spacings a lookup moves off its surface along the normal, so the probes it blends
    /// lie in front of the surface (1: none of them below a floor).
    pub normal_bias: f32,
    /// Most steps per cone.
    pub max_steps: u32,
    /// Levels finer than its width each cone reads (`clipConeTraceNear`): each becomes a
    /// sparsely sampled ray that sees through gaps narrower than it (a road between trees), the
    /// probes' history averaging the samples; 0 reads the levels as wide as the cones, which fill
    /// such gaps.
    pub level_bias: f32,
}

impl Default for ClipmapProbeOptions {
    fn default() -> Self {
        Self { levels: MAX_CLIPMAP_LEVELS as u32, spacing_voxels: 2, probes_per_frame: 8192, hysteresis: 0.9, sky_scale: 1.0, normal_bias: 1.0, max_steps: 32, level_bias: 4.0 }
    }
}

/// The WGSL `ClipProbeGrid` (clipmap_probes.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub(crate) struct ClipProbeGridGpu {
    dims: [u32; 3],
    level_count: u32,
    spacing: f32,
    normal_bias: f32,
    _pad: [f32; 2],
    levels: [ClipLevelGpu; MAX_CLIPMAP_LEVELS],
}

/// The WGSL `ClipProbeUpdate` (clipmap_probe_update.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub(crate) struct ClipProbeUpdateGpu {
    rotation: [f32; 16],
    level: u32,
    mode: u32,
    first: u32,
    count: u32,
    lo: [i32; 3],
    hysteresis: f32,
    size: [u32; 3],
    sky_scale: f32,
    max_steps: u32,
    voxel_level: u32,
    tan_half: f32,
    level_bias: f32,
    frame: u32,
    _pad: [u32; 3],
}

const MODE_ROUND: u32 = 0;
const MODE_REGION: u32 = 1;
const MODE_CLEAR: u32 = 2;
/// Dispatches a frame may record at most (a slot of parameters each).
const MAX_DISPATCHES: usize = 5 * MAX_CLIPMAP_LEVELS;
/// Probes a window moves in steps of.
const SNAP: u32 = 2;

/// Irradiance probes of a voxel clipmap (`SceneVoxelClipmap::enable_probes`): the far field of
/// the scene's light at a few probes a metre instead of cones per pixel, and the sky each point
/// sees past the forest. A level of probes per level of the clipmap (or fewer), each a window
/// round the camera of a lattice `spacing_voxels` of that level's voxels apart, stored toroidally
/// as the clipmap's voxels are. Each probe traces 16 cones through the clipmap, the light they
/// gather and the sky past it projected onto order-1 spherical harmonics with the share of each
/// direction that reaches the sky, blended into its history (`hysteresis`); a probe inside a
/// surface is left out. A window moves with the camera in steps of two probes, and the probes of
/// the slab it moved into are traced at once; the others update in turn (`probes_per_frame`).
///
/// Read them with `CLIPMAP_PROBES_WGSL`'s `kansei_clipmap_light(p, n)` (irradiance and sky
/// visibility) or `kansei_clipmap_sky_visibility(p, n)` in a material, with `bindings_wgsl` and
/// `bind_group_entries`; or with `VoxelGIEffect::set_clipmap_probes`, the far field on screen.
pub struct ClipmapProbes {
    pub options: ClipmapProbeOptions,
    layout: ClipmapLayout,
    /// The clipmap level whose voxels are half the probes' spacing, per probe level 0.
    voxel_level0: u32,
    grid: wgpu::Buffer,
    grid_data: ClipProbeGridGpu,
    written: Option<ClipProbeGridGpu>,
    params: wgpu::Buffer,
    params_stride: u64,
    probes: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    group: Option<(wgpu::BindGroup, wgpu::Buffer)>,
    cursors: Vec<u32>,
    frame: u32,
}

impl ClipmapProbes {
    pub(crate) fn new(device: &wgpu::Device, clipmap: &VoxelClipmap, options: ClipmapProbeOptions) -> Self {
        let voxels = clipmap.layout();
        let spacing = options.spacing_voxels.max(1);
        let dims = voxels.dims.map(|d| (d / spacing).max(8));
        let layout = ClipmapLayout::new(options.levels.clamp(1, voxels.levels), dims, voxels.voxel_size * spacing as f32);
        let compute = wgpu::ShaderStages::COMPUTE;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry { binding, visibility: compute, ty, count: None };
        let uniform = wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None };
        let mut entries = vec![
            entry(0, uniform),
            entry(1, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only: false }, has_dynamic_offset: false, min_binding_size: None }),
            entry(2, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: wgpu::BufferSize::new(std::mem::size_of::<ClipProbeUpdateGpu>() as u64) }),
            entry(3, uniform),
        ];
        entries.extend(clipmap_layout_entries(compute));
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("VoxelClipmap/ProbesBGL"), entries: &entries });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("VoxelClipmap/Probes"), source: wgpu::ShaderSource::Wgsl(CLIPMAP_PROBE_UPDATE_WGSL.into()) });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("VoxelClipmap/Probes"), bind_group_layouts: &[&bgl], push_constant_ranges: &[] });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VoxelClipmap/Probes"),
            layout: Some(&pl),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: usage | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let params_stride = (std::mem::size_of::<ClipProbeUpdateGpu>() as u64).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment as u64);
        let count = layout.levels as u64 * layout.voxel_count();
        let grid_data = ClipProbeGridGpu {
            dims: layout.dims,
            level_count: layout.levels,
            spacing: layout.voxel_size,
            normal_bias: options.normal_bias.max(0.0),
            _pad: [0.0; 2],
            levels: [ClipLevelGpu::default(); MAX_CLIPMAP_LEVELS],
        };
        Self {
            options,
            layout,
            // voxels half the spacing: level 0's for 2 voxels, level 1's for 4, ...
            voxel_level0: spacing.ilog2().saturating_sub(1).min(voxels.levels - 1),
            grid: buffer("VoxelClipmap/ProbeGrid", std::mem::size_of::<ClipProbeGridGpu>() as u64, wgpu::BufferUsages::UNIFORM),
            grid_data,
            written: None,
            params: buffer("VoxelClipmap/ProbeUpdate", params_stride * MAX_DISPATCHES as u64, wgpu::BufferUsages::UNIFORM),
            params_stride,
            // (COPY_SRC: readable in tests)
            probes: buffer("VoxelClipmap/Probes", count * 64, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC),
            pipeline,
            bgl,
            group: None,
            cursors: vec![0; layout.levels as usize],
            frame: 0,
        }
    }

    /// The probes' lattice: per level, `dims` probes `voxel_size * 2^level` metres apart.
    pub fn layout(&self) -> &ClipmapLayout {
        &self.layout
    }

    /// Level `level`'s window's first lattice point, once placed.
    pub fn origin(&self, level: u32) -> Option<IVec3> {
        let l = self.grid_data.levels.get(level as usize)?;
        (l.valid != 0).then(|| IVec3::from(l.origin))
    }

    /// The `ClipProbeGrid` uniform.
    pub fn grid_buffer(&self) -> &wgpu::Buffer {
        &self.grid
    }

    /// The probes: 4 `vec4f` each (clipmap_probes.wgsl), level after level.
    pub fn probe_buffer(&self) -> &wgpu::Buffer {
        &self.probes
    }

    pub fn memory_bytes(&self) -> u64 {
        self.probes.size()
    }

    /// The declarations `CLIPMAP_PROBES_WGSL` reads, at `group` and the two bindings from
    /// `first`: the grid and the probes (`bind_group_entries` binds them).
    pub fn bindings_wgsl(group: u32, first: u32) -> String {
        format!(
            "@group({group}) @binding({}) var<uniform> kansei_clip_probe_grid : ClipProbeGrid;\n\
             @group({group}) @binding({}) var<storage, read> kansei_clip_probes : array<vec4f>;\n",
            first,
            first + 1
        )
    }

    /// The buffers for `bindings_wgsl(_, first)`, in order.
    pub fn bind_group_entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 2] {
        [wgpu::BindGroupEntry { binding: first, resource: self.grid.as_entire_binding() }, wgpu::BindGroupEntry { binding: first + 1, resource: self.probes.as_entire_binding() }]
    }

    /// Start every probe over (after a cut, or a change of the lighting the history should not
    /// blend through): each level is placed anew round the camera next update.
    pub fn reset(&mut self) {
        for level in &mut self.grid_data.levels {
            *level = ClipLevelGpu::default();
        }
    }

    /// Record the probes' update for an eye at `eye`, after the clipmap's light: each level's
    /// window follows the eye (placed anew when it first is, or jumps past itself: its probes
    /// marked never traced; the slab it moved into traced at once), then the next probes of each
    /// level in turn.
    pub(crate) fn encode(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, clipmap: &VoxelClipmap, sky: &wgpu::Buffer, eye: Vec3) {
        let layout = self.layout;
        let dims = UVec3::from(layout.dims).as_ivec3();
        let per_level = layout.voxel_count() as u32;
        let o = self.options;
        let base = ClipProbeUpdateGpu {
            rotation: random_rotation(self.frame).to_cols_array(),
            level: 0,
            mode: MODE_ROUND,
            first: 0,
            count: 0,
            lo: [0; 3],
            hysteresis: o.hysteresis.clamp(0.0, 0.999),
            size: [0; 3],
            sky_scale: o.sky_scale.max(0.0),
            max_steps: o.max_steps.max(1),
            voxel_level: 0,
            tan_half: 0.6,
            level_bias: o.level_bias.max(0.0),
            frame: self.frame,
            _pad: [0; 3],
        };
        // (params, workgroups) per dispatch
        let mut dispatches: Vec<(ClipProbeUpdateGpu, u32)> = Vec::new();
        let per_frame = o.probes_per_frame.max(1).div_ceil(layout.levels).min(per_level);
        for level in 0..layout.levels {
            let voxel_level = (self.voxel_level0 + level).min(clipmap.layout().levels - 1);
            let at = ClipProbeUpdateGpu { level, voxel_level, ..base };
            let current = self.origin(level);
            let target = match current {
                Some(origin) => layout.follow(level, origin, eye, SNAP),
                None => layout.centred_origin(level, eye, SNAP),
            };
            match current {
                Some(origin) if (target - origin).abs().cmplt(dims).all() => {
                    // the slabs the window moved into, axis by axis, traced fresh
                    let mut from = origin;
                    for axis in 0..3 {
                        let d = target[axis] - from[axis];
                        if d == 0 {
                            continue;
                        }
                        let mut moved = from;
                        moved[axis] = target[axis];
                        let mut lo = moved;
                        let mut size = dims;
                        if d > 0 {
                            lo[axis] = from[axis] + dims[axis];
                        }
                        size[axis] = d.abs();
                        let count = size.as_uvec3().element_product();
                        dispatches.push((ClipProbeUpdateGpu { mode: MODE_REGION, lo: lo.to_array(), size: size.as_uvec3().to_array(), count, ..at }, count));
                        from = moved;
                    }
                }
                _ => dispatches.push((ClipProbeUpdateGpu { mode: MODE_CLEAR, first: 0, count: per_level, ..at }, per_level.div_ceil(16))),
            }
            self.grid_data.levels[level as usize] = ClipLevelGpu { origin: target.to_array(), valid: 1 };
            // and the next ones in turn
            let first = self.cursors[level as usize];
            dispatches.push((ClipProbeUpdateGpu { mode: MODE_ROUND, first, count: per_frame, ..at }, per_frame));
            self.cursors[level as usize] = (first + per_frame) % per_level;
        }
        if self.written != Some(self.grid_data) {
            queue.write_buffer(&self.grid, 0, bytemuck::bytes_of(&self.grid_data));
            self.written = Some(self.grid_data);
        }
        dispatches.truncate(MAX_DISPATCHES);
        let mut bytes = vec![0u8; (self.params_stride * dispatches.len() as u64) as usize];
        for (k, (params, _)) in dispatches.iter().enumerate() {
            let at = (k as u64 * self.params_stride) as usize;
            bytes[at..at + std::mem::size_of::<ClipProbeUpdateGpu>()].copy_from_slice(bytemuck::bytes_of(params));
        }
        queue.write_buffer(&self.params, 0, &bytes);
        self.frame = self.frame.wrapping_add(1);

        if self.group.as_ref().is_none_or(|(_, bound)| bound != sky) {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: self.grid.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.probes.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &self.params, offset: 0, size: wgpu::BufferSize::new(std::mem::size_of::<ClipProbeUpdateGpu>() as u64) }),
                },
                wgpu::BindGroupEntry { binding: 3, resource: sky.as_entire_binding() },
            ];
            entries.extend(clipmap_entries(clipmap));
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("VoxelClipmap/ProbesBG"), layout: &self.bgl, entries: &entries });
            self.group = Some((group, sky.clone()));
        }
        let stamp = crate::profiling::gpu_pass("VoxelClipmap/Probes");
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("VoxelClipmap/Probes"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
        pass.set_pipeline(&self.pipeline);
        // a workgroup per probe (or per 16 slots cleared)
        for (k, (_, probes)) in dispatches.iter().enumerate() {
            pass.set_bind_group(0, &self.group.as_ref().unwrap().0, &[(k as u64 * self.params_stride) as u32]);
            pass.dispatch_workgroups((*probes).min(65535), 1, 1);
        }
    }
}

/// A random rotation for frame `frame` (a uniform quaternion, Shoemake 1992).
fn random_rotation(frame: u32) -> glam::Mat4 {
    let h = |k: u32| {
        let mut x = frame.wrapping_mul(0x9e37_79b9).wrapping_add(k.wrapping_mul(0x85eb_ca6b));
        x ^= x >> 16;
        x = x.wrapping_mul(0x7feb_352d);
        x ^= x >> 15;
        x = x.wrapping_mul(0x846c_a68b);
        x ^= x >> 16;
        x as f32 / u32::MAX as f32
    };
    let (u1, u2, u3) = (h(1), h(2) * std::f32::consts::TAU, h(3) * std::f32::consts::TAU);
    glam::Mat4::from_quat(glam::Quat::from_xyzw((1.0 - u1).sqrt() * u2.sin(), (1.0 - u1).sqrt() * u2.cos(), u1.sqrt() * u3.sin(), u1.sqrt() * u3.cos()).normalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default()))?;
        pollster::block_on(adapter.request_device(&Default::default(), None)).ok()
    }

    /// The lookups read the probes as their SH say: under a uniform sky of radiance L (every
    /// probe's constant band L * 0.282095 * 4 pi, its sky share likewise), a surface facing any
    /// way receives pi L and sees the whole sky, and a medium scatters L toward the camera
    /// whatever the phase; under a sky only above (the bands of the upper hemisphere's), an
    /// upward surface receives pi L, a downward one none, and forward scattering along +y brings
    /// more of it.
    #[test]
    fn the_lookups_read_what_the_probes_hold() {
        let Some((device, queue)) = device() else { return eprintln!("no GPU adapter: skipping") };
        let dims = [8u32; 3];
        let grid = ClipProbeGridGpu {
            dims,
            level_count: 1,
            spacing: 1.0,
            normal_bias: 1.0,
            _pad: [0.0; 2],
            levels: std::array::from_fn(|k| if k == 0 { ClipLevelGpu { origin: [-4; 3], valid: 1 } } else { ClipLevelGpu::default() }),
        };
        let l = [1.0f32, 2.0, 3.0];
        let four_pi = 4.0 * std::f32::consts::PI;
        let (y0, y1) = (0.282095f32, 0.488603f32);
        let uniform: [[f32; 4]; 4] = [[l[0] * y0 * four_pi, l[1] * y0 * four_pi, l[2] * y0 * four_pi, y0 * four_pi], [0.0; 4], [0.0; 4], [0.0; 4]];
        // the upper hemisphere: c0 = L Y0 2 pi, c1y = L * 0.488603 * pi (the integral of y over it)
        let pi = std::f32::consts::PI;
        let upper: [[f32; 4]; 4] = [[l[0] * y0 * 2.0 * pi, l[1] * y0 * 2.0 * pi, l[2] * y0 * 2.0 * pi, y0 * 2.0 * pi], [0.0; 4], [l[0] * y1 * pi, l[1] * y1 * pi, l[2] * y1 * pi, y1 * pi], [0.0; 4]];
        let code = format!(
            "{}\n{}\n@group(0) @binding(2) var<storage, read_write> out: array<vec4f>;\n\
             @compute @workgroup_size(1) fn main() {{\n\
                 let p = vec3f(0.3, 0.2, -0.4);\n\
                 out[0] = kansei_clipmap_light(p, normalize(vec3f(0.3, 0.8, -0.2)));\n\
                 out[1] = kansei_clipmap_light(p, vec3f(0.0, -1.0, 0.0));\n\
                 out[2] = kansei_clipmap_inscatter(p, vec3f(0.0, 1.0, 0.0), 0.6);\n\
                 out[3] = kansei_clipmap_inscatter(p, vec3f(0.0, -1.0, 0.0), 0.6);\n\
                 out[4] = kansei_clipmap_light(p, vec3f(0.0, 1.0, 0.0));\n\
                 out[5] = kansei_clipmap_light(vec3f(40.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0));\n\
             }}",
            include_str!("shaders/clipmap_probes.wgsl"),
            ClipmapProbes::bindings_wgsl(0, 0)
        );
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: None, source: wgpu::ShaderSource::Wgsl(code.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        use wgpu::util::DeviceExt;
        let run = |probe: [[f32; 4]; 4]| -> Vec<[f32; 4]> {
            let probes: Vec<[f32; 4]> = (0..dims.iter().product::<u32>()).flat_map(|_| probe).collect();
            let grid_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::bytes_of(&grid), usage: wgpu::BufferUsages::UNIFORM });
            let probe_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: None, contents: bytemuck::cast_slice(&probes), usage: wgpu::BufferUsages::STORAGE });
            let out = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 6 * 16, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false });
            let read = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 6 * 16, usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, mapped_at_creation: false });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: grid_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: probe_buffer.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: out.as_entire_binding() },
                ],
            });
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&out, 0, &read, 0, 6 * 16);
            queue.submit(Some(encoder.finish()));
            read.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            device.poll(wgpu::Maintain::Wait);
            let values = bytemuck::cast_slice::<u8, [f32; 4]>(&read.slice(..).get_mapped_range()).to_vec();
            values
        };
        let close = |a: [f32; 4], b: [f32; 4]| (0..4).all(|c| (a[c] - b[c]).abs() <= 1e-3 * b[c].abs().max(1.0));
        let pi_l = [pi * l[0], pi * l[1], pi * l[2], 1.0];
        let got = run(uniform);
        eprintln!("uniform sky: {got:?}");
        assert!(close(got[0], pi_l), "irradiance {:?}", got[0]);
        assert!(close(got[1], pi_l), "irradiance facing down {:?}", got[1]);
        assert!(close(got[2], [l[0], l[1], l[2], 1.0]) && close(got[3], [l[0], l[1], l[2], 1.0]), "inscattered {:?} {:?}", got[2], got[3]);
        assert_eq!(got[5][3], -1.0, "no probe holds a point past the window");
        let got = run(upper);
        eprintln!("sky above: {got:?}");
        assert!(close(got[4], pi_l), "an upward surface {:?}", got[4]);
        assert!(got[1][0].abs() < 1e-3 && got[1][3].abs() < 1e-3, "a downward surface {:?}", got[1]);
        // forward scattering of the light from above, seen looking up: L/2 + g * 3L/4
        let up = [0.5 * l[0] + 0.6 * 0.75 * l[0], 0.5 * l[1] + 0.6 * 0.75 * l[1], 0.5 * l[2] + 0.6 * 0.75 * l[2], 1.0];
        assert!(close(got[2], up), "looking up {:?} of {up:?}", got[2]);
        assert!(got[3][0] < got[2][0], "looking down, less of it");
    }
}
