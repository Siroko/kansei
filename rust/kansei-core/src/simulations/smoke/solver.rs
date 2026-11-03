//! The Curl noise and Fluid levels' passes (`shaders/smoke_solver.wgsl`): one pipeline per stage,
//! bind groups made for each ping-pong of the grids (and kept, `BindGroupCache`).

use super::simulation::{Grid3, Grids, SmokeStepGpu, STEP_SLOT};
use super::SOLVER_WGSL;
use crate::renderers::BindGroupCache;

/// The bindings an entry point reads and writes (smoke_solver.wgsl's binding numbers).
#[derive(Clone, Copy)]
enum B {
    Step = 0,
    Sampler = 1,
    Velocity = 2,
    Scalars = 3,
    Aux = 4,
    VectorOut = 5,
    Pressure = 6,
    Divergence = 7,
    ScalarOut = 8,
}

struct Stage {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    bindings: &'static [B],
}

pub(crate) struct Solver {
    sources: Stage,
    advect_velocity: Stage,
    advect_forward: Stage,
    advect_correct: Stage,
    vorticity: Stage,
    forces: Stage,
    divergence: Stage,
    jacobi: Stage,
    project: Stage,
    sampler: wgpu::Sampler,
    /// bound for the velocity of a volume without a grid of it (Curl noise)
    no_velocity: Grid3,
    cache: BindGroupCache,
}

/// One volume's step: where its `SmokeStep` sits and how many Jacobi iterations it takes.
pub(crate) struct VolumeStep {
    pub(crate) slot: usize,
    pub(crate) pressure_iterations: u32,
}

impl Solver {
    pub(crate) fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("Smoke/Solver"), source: wgpu::ShaderSource::Wgsl(SOLVER_WGSL.into()) });
        let compute = wgpu::ShaderStages::COMPUTE;
        let stage = |entry: &str, bindings: &'static [B]| {
            let entries: Vec<_> = bindings
                .iter()
                .map(|b| {
                    let ty = match b {
                        B::Step => wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                        B::Sampler => wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        B::Velocity | B::Scalars | B::Aux => wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: true }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                        B::Pressure | B::Divergence => wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D3, multisampled: false },
                        B::VectorOut => wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::Rgba16Float, view_dimension: wgpu::TextureViewDimension::D3 },
                        B::ScalarOut => wgpu::BindingType::StorageTexture { access: wgpu::StorageTextureAccess::WriteOnly, format: wgpu::TextureFormat::R32Float, view_dimension: wgpu::TextureViewDimension::D3 },
                    };
                    wgpu::BindGroupLayoutEntry { binding: *b as u32, visibility: compute, ty, count: None }
                })
                .collect();
            let label = format!("Smoke/{entry}");
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(&label), entries: &entries });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(&label), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            });
            Stage { pipeline, bgl, bindings }
        };
        use B::*;
        Self {
            sources: stage("sources", &[Step, Scalars, VectorOut]),
            advect_velocity: stage("advectVelocity", &[Step, Sampler, Velocity, VectorOut]),
            advect_forward: stage("advectScalarsForward", &[Step, Sampler, Velocity, Scalars, VectorOut]),
            advect_correct: stage("advectScalarsCorrect", &[Step, Sampler, Velocity, Scalars, Aux, VectorOut]),
            vorticity: stage("vorticity", &[Step, Velocity, VectorOut]),
            forces: stage("forces", &[Step, Velocity, Scalars, Aux, VectorOut]),
            divergence: stage("divergence", &[Step, Velocity, ScalarOut]),
            jacobi: stage("jacobi", &[Step, Pressure, Divergence, ScalarOut]),
            project: stage("project", &[Step, Velocity, VectorOut, Pressure]),
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("Smoke/Linear"),
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            }),
            no_velocity: Grid3::new(device, "Smoke/NoVelocityGrid", [1, 1, 1], wgpu::TextureFormat::Rgba16Float),
            cache: BindGroupCache::default(),
        }
    }

    /// Record one volume's step: sources, advection, and on Fluid the forces and the projection.
    pub(crate) fn step(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, steps: &wgpu::Buffer, grids: &mut Grids, v: VolumeStep) {
        let step = wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: steps,
            offset: v.slot as u64 * STEP_SLOT,
            size: wgpu::BufferSize::new(std::mem::size_of::<SmokeStepGpu>() as u64),
        });
        let [w, h, d] = grids.dims;
        let groups = (w.div_ceil(4), h.div_ceil(4), d.div_ceil(4));
        // the resources a stage reads and writes, by binding
        struct Io<'a> {
            velocity: Option<&'a wgpu::TextureView>,
            scalars: Option<&'a wgpu::TextureView>,
            aux: Option<&'a wgpu::TextureView>,
            vector_out: Option<&'a wgpu::TextureView>,
            pressure: Option<&'a wgpu::TextureView>,
            divergence: Option<&'a wgpu::TextureView>,
            scalar_out: Option<&'a wgpu::TextureView>,
        }
        let none = Io { velocity: None, scalars: None, aux: None, vector_out: None, pressure: None, divergence: None, scalar_out: None };
        let bind = |stage: &Stage, io: Io| -> wgpu::BindGroup {
            let entries: Vec<_> = stage
                .bindings
                .iter()
                .map(|b| {
                    let resource = match b {
                        B::Step => step.clone(),
                        B::Sampler => wgpu::BindingResource::Sampler(&self.sampler),
                        B::Velocity => wgpu::BindingResource::TextureView(io.velocity.unwrap_or(&self.no_velocity.view)),
                        B::Scalars => wgpu::BindingResource::TextureView(io.scalars.expect("scalars")),
                        B::Aux => wgpu::BindingResource::TextureView(io.aux.expect("aux")),
                        B::VectorOut => wgpu::BindingResource::TextureView(io.vector_out.expect("vector out")),
                        B::Pressure => wgpu::BindingResource::TextureView(io.pressure.expect("pressure")),
                        B::Divergence => wgpu::BindingResource::TextureView(io.divergence.expect("divergence")),
                        B::ScalarOut => wgpu::BindingResource::TextureView(io.scalar_out.expect("scalar out")),
                    };
                    wgpu::BindGroupEntry { binding: *b as u32, resource }
                })
                .collect();
            self.cache.get(device, &wgpu::BindGroupDescriptor { label: Some("Smoke/StageBG"), layout: &stage.bgl, entries: &entries })
        };
        let run = |pass: &mut wgpu::ComputePass, stage: &Stage, bg: &wgpu::BindGroup| {
            pass.set_pipeline(&stage.pipeline);
            pass.set_bind_group(0, bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        };
        let begin = |encoder: &'_ mut wgpu::CommandEncoder, label: &'static str| -> (wgpu::ComputePass<'static>, Option<crate::profiling::PassStamp>) {
            let stamp = crate::profiling::gpu_pass(label);
            let pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(label), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) }).forget_lifetime();
            (pass, stamp)
        };

        // the scalars: a -> b (sources) -> c (forward) -> a (corrected)
        let a = grids.current;
        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
        let s = &grids.scalars;
        let vel_in = grids.fluid.as_ref().map(|f| f.velocity[f.current].view.clone());
        {
            let (mut pass, _stamp) = begin(encoder, "Smoke/Sources");
            run(&mut pass, &self.sources, &bind(&self.sources, Io { scalars: Some(&s[a].view), vector_out: Some(&s[b].view), ..none }));
        }
        {
            let (mut pass, _stamp) = begin(encoder, "Smoke/Advect");
            if let Some(f) = &grids.fluid {
                let (vc, vn) = (&f.velocity[f.current].view, &f.velocity[1 - f.current].view);
                run(&mut pass, &self.advect_velocity, &bind(&self.advect_velocity, Io { velocity: Some(vc), vector_out: Some(vn), ..none }));
            }
            let vel = vel_in.as_ref();
            run(&mut pass, &self.advect_forward, &bind(&self.advect_forward, Io { velocity: vel, scalars: Some(&s[b].view), vector_out: Some(&s[c].view), ..none }));
            run(&mut pass, &self.advect_correct, &bind(&self.advect_correct, Io { velocity: vel, scalars: Some(&s[b].view), aux: Some(&s[c].view), vector_out: Some(&s[a].view), ..none }));
        }
        let Some(f) = grids.fluid.as_mut() else { return };
        // the velocity: c -> n (advected) -> c (forces) -> n (projected)
        let (vc, vn) = (&f.velocity[f.current].view, &f.velocity[1 - f.current].view);
        {
            let (mut pass, _stamp) = begin(encoder, "Smoke/Forces");
            run(&mut pass, &self.vorticity, &bind(&self.vorticity, Io { velocity: Some(vn), vector_out: Some(&f.vorticity.view), ..none }));
            run(&mut pass, &self.forces, &bind(&self.forces, Io { velocity: Some(vn), scalars: Some(&s[a].view), aux: Some(&f.vorticity.view), vector_out: Some(vc), ..none }));
        }
        {
            let (mut pass, _stamp) = begin(encoder, "Smoke/Pressure");
            let mut p = f.pressure_current;
            if v.pressure_iterations > 0 {
                run(&mut pass, &self.divergence, &bind(&self.divergence, Io { velocity: Some(vc), scalar_out: Some(&f.divergence.view), ..none }));
                let ping = bind(&self.jacobi, Io { pressure: Some(&f.pressure[0].view), divergence: Some(&f.divergence.view), scalar_out: Some(&f.pressure[1].view), ..none });
                let pong = bind(&self.jacobi, Io { pressure: Some(&f.pressure[1].view), divergence: Some(&f.divergence.view), scalar_out: Some(&f.pressure[0].view), ..none });
                for _ in 0..v.pressure_iterations {
                    run(&mut pass, &self.jacobi, if p == 0 { &ping } else { &pong });
                    p = 1 - p;
                }
            }
            run(&mut pass, &self.project, &bind(&self.project, Io { velocity: Some(vc), pressure: Some(&f.pressure[p].view), vector_out: Some(vn), ..none }));
            f.pressure_current = p;
        }
        f.current = 1 - f.current;
    }
}
