//! `SmokeSimulation`: the app's smoke volumes, their grids on the GPU, and the handle the fog
//! marches them through.

use std::sync::{Arc, Mutex};

use bytemuck::{Pod, Zeroable};

use super::mouse::SmokeMouse;
use super::solver::{Solver, VolumeStep};
use super::volume::{SmokeFill, SmokeMotion, SmokeVolume, SmokeVolumeGpu};
use super::{FILL_WGSL, MAX_SMOKE_VOLUMES};

/// Each volume's slot in the step buffer: its `SmokeStep`, at a stride every device's dynamic
/// offset alignment divides.
pub(crate) const STEP_SLOT: u64 = 512;

/// WGSL `SmokeStep`: one volume's step.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SmokeStepGpu {
    pub(crate) volume: SmokeVolumeGpu,
    pub(crate) dt: f32,
    pub(crate) time: f32,
    pub(crate) buoyancy: f32,
    pub(crate) weight: f32,
    pub(crate) cooling: f32,
    pub(crate) vorticity: f32,
    pub(crate) max_speed: f32,
    pub(crate) ambient_draft: f32,
    pub(crate) draft_scale: f32,
    pub(crate) mouse_active: u32,
    pub(crate) pressure_iterations: u32,
    pub(crate) _pad1: f32,
    pub(crate) source_temperature: [f32; 4],
    pub(crate) mouse_origin: [f32; 3],
    pub(crate) mouse_radius: f32,
    pub(crate) mouse_direction: [f32; 3],
    pub(crate) _pad2: f32,
    pub(crate) mouse_push: [f32; 3],
    pub(crate) _pad3: f32,
}

/// WGSL `SmokeFillParams`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub(crate) struct SmokeFillGpu {
    pub(crate) dims: [u32; 3],
    pub(crate) kind: u32,
    pub(crate) size: [f32; 3],
    pub(crate) density: f32,
    pub(crate) sphere_center: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) temperature: f32,
    pub(crate) layer_count: u32,
    pub(crate) layer_thickness: f32,
    pub(crate) _pad: f32,
}

/// A volume added to a [`SmokeSimulation`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct SmokeVolumeId(pub(crate) usize);

/// A volume's current grids (for tests and tools): `scalars` holds density in r and temperature
/// in g, `velocity` (Fluid only) the MAC grid's face velocities along the box's axes, m/s.
pub struct SmokeGridTextures<'a> {
    pub dims: [u32; 3],
    pub scalars: &'a wgpu::Texture,
    pub velocity: Option<&'a wgpu::Texture>,
}

/// A 3D texture and its view.
pub(crate) struct Grid3 {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Grid3 {
    pub(crate) fn new(device: &wgpu::Device, label: &str, dims: [u32; 3], format: wgpu::TextureFormat) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: dims[0], height: dims[1], depth_or_array_layers: dims[2] },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D3,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        Self { texture, view }
    }
}

/// The Fluid level's grids beyond the scalars.
pub(crate) struct FluidGrids {
    /// MAC velocities (x: u on the cell's -x face, y: v on -y, z: w on -z), ping-ponged.
    pub(crate) velocity: [Grid3; 2],
    pub(crate) current: usize,
    /// Jacobi's two pressures; `pressure_current` holds the last solution (the next warm start).
    pub(crate) pressure: [Grid3; 2],
    pub(crate) pressure_current: usize,
    pub(crate) divergence: Grid3,
    pub(crate) vorticity: Grid3,
}

/// A volume's grids (Curl noise and Fluid).
pub(crate) struct Grids {
    pub(crate) dims: [u32; 3],
    /// (density, temperature), three for MacCormack's forward and corrected steps.
    pub(crate) scalars: [Grid3; 3],
    pub(crate) current: usize,
    pub(crate) fluid: Option<FluidGrids>,
}

impl Grids {
    fn new(device: &wgpu::Device, dims: [u32; 3], fluid: bool) -> Self {
        let vector = wgpu::TextureFormat::Rgba16Float;
        let scalar = wgpu::TextureFormat::R32Float;
        Self {
            dims,
            scalars: [0, 1, 2].map(|_| Grid3::new(device, "Smoke/Scalars", dims, vector)),
            current: 0,
            fluid: fluid.then(|| FluidGrids {
                velocity: [0, 1].map(|_| Grid3::new(device, "Smoke/Velocity", dims, vector)),
                current: 0,
                pressure: [0, 1].map(|_| Grid3::new(device, "Smoke/Pressure", dims, scalar)),
                pressure_current: 0,
                divergence: Grid3::new(device, "Smoke/Divergence", dims, scalar),
                vorticity: Grid3::new(device, "Smoke/Vorticity", dims, vector),
            }),
        }
    }

    pub(crate) fn scalars(&self) -> &Grid3 {
        &self.scalars[self.current]
    }

    pub(crate) fn velocity(&self) -> Option<&Grid3> {
        self.fluid.as_ref().map(|f| &f.velocity[f.current])
    }
}

impl Drop for Grids {
    // wgpu's web backend frees nothing on drop: give the memory back now (AGENTS.md)
    fn drop(&mut self) {
        for g in &self.scalars {
            g.texture.destroy();
        }
        if let Some(f) = &self.fluid {
            for g in f.velocity.iter().chain(&f.pressure).chain([&f.divergence, &f.vorticity]) {
                g.texture.destroy();
            }
        }
    }
}

/// A volume's `SmokeStep` for a step of `dt` at `time`.
fn step_gpu(volume: &SmokeVolume, dims: [u32; 3], dt: f32, time: f32, mouse: &SmokeMouse) -> SmokeStepGpu {
    let mut step = SmokeStepGpu { volume: volume.gpu(dims), dt, time, ..Zeroable::zeroed() };
    if let SmokeMotion::Fluid(f) = volume.motion {
        step.buoyancy = f.buoyancy;
        step.weight = f.weight;
        step.cooling = f.cooling.max(0.0);
        step.vorticity = f.vorticity.max(0.0);
        step.max_speed = f.max_speed.max(1e-3);
        step.ambient_draft = f.ambient_draft;
        step.draft_scale = f.draft_scale.max(1e-3);
        step.pressure_iterations = f.pressure_iterations;
    } else {
        step.max_speed = 1e9;
    }
    for (t, s) in step.source_temperature.iter_mut().zip(&volume.sources) {
        *t = s.temperature;
    }
    if mouse.is_active() {
        step.mouse_active = 1;
        step.mouse_origin = [mouse.origin.x, mouse.origin.y, mouse.origin.z];
        step.mouse_direction = [mouse.direction.x, mouse.direction.y, mouse.direction.z];
        step.mouse_radius = mouse.radius;
        step.mouse_push = [mouse.push.x, mouse.push.y, mouse.push.z];
    }
    step
}

/// What the fog reads of a volume each frame.
#[derive(Clone)]
pub(crate) struct SmokeHandleVolume {
    pub(crate) data: SmokeVolumeGpu,
    pub(crate) scalars: Option<wgpu::TextureView>,
    pub(crate) velocity: Option<wgpu::TextureView>,
}

#[derive(Default)]
pub(crate) struct SmokeShared {
    /// The enabled volumes, in the order they were added.
    pub(crate) volumes: Vec<SmokeHandleVolume>,
    /// The simulation's clock, seconds.
    pub(crate) time: f32,
    /// Simulated seconds since the fog last rendered (it takes them).
    pub(crate) advanced: f32,
}

/// What `VolumetricFogEffect::set_smoke` marches: a [`SmokeSimulation`]'s enabled volumes and
/// their current grids, followed as the simulation steps and swaps them.
#[derive(Clone, Default)]
pub struct SmokeHandle {
    pub(crate) shared: Arc<Mutex<SmokeShared>>,
}

impl SmokeHandle {
    /// Whether any volume is enabled.
    pub fn has_volumes(&self) -> bool {
        !self.shared.lock().unwrap().volumes.is_empty()
    }
}

struct Slot {
    volume: SmokeVolume,
    enabled: bool,
    grids: Option<Grids>,
}

struct FillGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    /// Bound for the velocity of a volume without one.
    no_velocity: Grid3,
}

/// The app's smoke: volumes of gas at one of three levels of motion (`SmokeMotion`), stepped by
/// the app (`step`, at a fixed rate: `pacing::FixedStep`) and rendered by the volumetric fog
/// (`VolumetricFogEffect::set_smoke(Some(&sim.handle()))`).
///
/// Change volumes (`set_motion`, `set_enabled`, `reset`) between frames, not between a frame's
/// recording and its submit: they may replace the grids the frame reads.
pub struct SmokeSimulation {
    device: wgpu::Device,
    queue: wgpu::Queue,
    slots: Vec<Slot>,
    time: f32,
    handle: SmokeHandle,
    fill: FillGpu,
    /// every volume's `SmokeStep` slot, written once a step
    step_buffer: wgpu::Buffer,
    solver: Solver,
}

impl SmokeSimulation {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let compute = wgpu::ShaderStages::COMPUTE;
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: compute,
            ty: wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 },
            count: None,
        };
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Smoke/FillBGL"),
            entries: &[
                wgpu::BindGroupLayoutEntry { binding: 0, visibility: compute, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None }, count: None },
                storage(1),
                storage(2),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Smoke/Fill"), source: wgpu::ShaderSource::Wgsl(FILL_WGSL.into()) });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("Smoke/Fill"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Smoke/Fill"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("fill"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &str, size: u64| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        Self {
            device: device.clone(),
            queue: queue.clone(),
            slots: Vec::new(),
            time: 0.0,
            handle: SmokeHandle::default(),
            fill: FillGpu {
                pipeline,
                bgl,
                params: buffer("Smoke/FillParams", std::mem::size_of::<SmokeFillGpu>() as u64),
                no_velocity: Grid3::new(device, "Smoke/NoVelocity", [1, 1, 1], wgpu::TextureFormat::Rgba16Float),
            },
            step_buffer: buffer("Smoke/Steps", STEP_SLOT * MAX_SMOKE_VOLUMES as u64),
            solver: Solver::new(device),
        }
    }

    /// Add a volume, enabled, its grids allocated and filled. Panics past `MAX_SMOKE_VOLUMES`.
    pub fn add(&mut self, volume: SmokeVolume) -> SmokeVolumeId {
        assert!(self.slots.len() < MAX_SMOKE_VOLUMES, "SmokeSimulation: at most {MAX_SMOKE_VOLUMES} volumes");
        self.slots.push(Slot { volume, enabled: true, grids: None });
        let id = SmokeVolumeId(self.slots.len() - 1);
        self.reset(id);
        id
    }

    pub fn volume(&self, id: SmokeVolumeId) -> &SmokeVolume {
        &self.slots[id.0].volume
    }

    /// The volume to edit: its look, sources and centre are read every step and every frame;
    /// call `reset` after changing its resolution, size or fill, `set_motion` for its level.
    pub fn volume_mut(&mut self, id: SmokeVolumeId) -> &mut SmokeVolume {
        &mut self.slots[id.0].volume
    }

    /// Switch a volume's level of motion: its grids are reallocated for it and refilled.
    pub fn set_motion(&mut self, id: SmokeVolumeId, motion: SmokeMotion) {
        self.slots[id.0].volume.motion = motion;
        self.reset(id);
    }

    /// Enable or disable a volume; enabling refills it.
    pub fn set_enabled(&mut self, id: SmokeVolumeId, enabled: bool) {
        let was = std::mem::replace(&mut self.slots[id.0].enabled, enabled);
        if enabled && !was {
            self.reset(id);
        } else {
            self.refresh_handle();
        }
    }

    pub fn is_enabled(&self, id: SmokeVolumeId) -> bool {
        self.slots[id.0].enabled
    }

    /// Re-read the volume's resolution and size (reallocating its grids if they changed) and
    /// refill it.
    pub fn reset(&mut self, id: SmokeVolumeId) {
        let slot = &mut self.slots[id.0];
        let (dims, fluid) = (slot.volume.grid_dims(), matches!(slot.volume.motion, SmokeMotion::Fluid(_)));
        if !slot.volume.motion.has_grid() {
            slot.grids = None;
        } else if slot.grids.as_ref().is_none_or(|g| g.dims != dims || g.fluid.is_some() != fluid) {
            slot.grids = None;
            slot.grids = Some(Grids::new(&self.device, dims, fluid));
        }
        self.fill_volume(id);
        self.refresh_handle();
    }

    /// Advance the gas `dt` seconds with `mouse` pushing it. `dt <= 0` only refreshes what the
    /// fog reads (after editing a volume's box or look).
    pub fn step(&mut self, dt: f32, mouse: &SmokeMouse) {
        if dt > 0.0 {
            self.time += dt;
            self.simulate(dt, mouse);
            self.handle.shared.lock().unwrap().advanced += dt;
        }
        self.refresh_handle();
    }

    /// One step of every enabled volume with a grid: their parameters in one write, their
    /// passes in one submit.
    fn simulate(&mut self, dt: f32, mouse: &SmokeMouse) {
        let mut data = vec![0u8; STEP_SLOT as usize * MAX_SMOKE_VOLUMES];
        let mut any = false;
        for (i, slot) in self.slots.iter().enumerate() {
            let Some(grids) = slot.grids.as_ref().filter(|_| slot.enabled) else { continue };
            let step = step_gpu(&slot.volume, grids.dims, dt, self.time, mouse);
            let o = i * STEP_SLOT as usize;
            data[o..o + std::mem::size_of::<SmokeStepGpu>()].copy_from_slice(bytemuck::bytes_of(&step));
            any = true;
        }
        if !any {
            return;
        }
        self.queue.write_buffer(&self.step_buffer, 0, &data);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Smoke/Step") });
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if !slot.enabled {
                continue;
            }
            let Some(grids) = slot.grids.as_mut() else { continue };
            let pressure_iterations = match slot.volume.motion {
                SmokeMotion::Fluid(f) => f.pressure_iterations,
                _ => 0,
            };
            self.solver.step(&self.device, &mut encoder, &self.step_buffer, grids, VolumeStep { slot: i, pressure_iterations });
        }
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// The simulation's clock, seconds.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// What the fog marches (`VolumetricFogEffect::set_smoke`).
    pub fn handle(&self) -> SmokeHandle {
        self.handle.clone()
    }

    /// The volume's current grids; `None` on Waves, which keeps none.
    pub fn grid_textures(&self, id: SmokeVolumeId) -> Option<SmokeGridTextures<'_>> {
        let grids = self.slots[id.0].grids.as_ref()?;
        Some(SmokeGridTextures { dims: grids.dims, scalars: &grids.scalars().texture, velocity: grids.velocity().map(|v| &v.texture) })
    }

    /// The volume's fill, into its current scalars (and its velocity cleared).
    fn fill_volume(&mut self, id: SmokeVolumeId) {
        let slot = &self.slots[id.0];
        let Some(grids) = &slot.grids else { return };
        let v = &slot.volume;
        let size = [v.size.x, v.size.y, v.size.z];
        let mut params = SmokeFillGpu { dims: grids.dims, size, ..Zeroable::zeroed() };
        match v.fill {
            SmokeFill::None => {}
            SmokeFill::Box { density, temperature } => {
                (params.kind, params.density, params.temperature) = (1, density, temperature);
            }
            SmokeFill::Sphere { center, radius, density, temperature } => {
                let l = v.to_local(center);
                params.kind = 2;
                params.sphere_center = [(l.x - 0.5) * size[0], (l.y - 0.5) * size[1], (l.z - 0.5) * size[2]];
                (params.radius, params.density, params.temperature) = (radius, density, temperature);
            }
            SmokeFill::Layers { .. } => {}
        }
        self.queue.write_buffer(&self.fill.params, 0, bytemuck::bytes_of(&params));
        let velocity = grids.velocity().unwrap_or(&self.fill.no_velocity);
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Smoke/FillBG"),
            layout: &self.fill.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.fill.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&grids.scalars().view) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(&velocity.view) },
            ],
        });
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Smoke/Fill") });
        {
            let stamp = crate::profiling::gpu_pass("Smoke/Fill");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("Smoke/Fill"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&self.fill.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let [w, h, d] = grids.dims;
            pass.dispatch_workgroups(w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        }
        // its own submit: the next fill rewrites the parameters
        self.queue.submit(std::iter::once(encoder.finish()));
    }

    /// What the fog reads: every enabled volume's data and current grids, and the clock.
    fn refresh_handle(&mut self) {
        let volumes = self
            .slots
            .iter()
            .filter(|s| s.enabled)
            .map(|s| {
                let dims = s.grids.as_ref().map_or_else(|| s.volume.grid_dims(), |g| g.dims);
                SmokeHandleVolume {
                    data: s.volume.gpu(dims),
                    scalars: s.grids.as_ref().map(|g| g.scalars().view.clone()),
                    velocity: s.grids.as_ref().and_then(|g| g.velocity()).map(|v| v.view.clone()),
                }
            })
            .collect();
        let mut shared = self.handle.shared.lock().unwrap();
        shared.volumes = volumes;
        shared.time = self.time;
    }
}
