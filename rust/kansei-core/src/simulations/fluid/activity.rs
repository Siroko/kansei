//! Letting a fluid rest: stop stepping it (and extracting its surface) while nobody can see it, or
//! while it has settled and nothing is touching it, and resume when that changes.
//!
//! - [`FluidSpeedProbe`] measures the particles' fastest speed (and how many move faster than a
//!   threshold) on the GPU and reads it back asynchronously, a few frames late: the frame never
//!   waits for the GPU.
//! - [`FluidSleep`] decides from that, from whether the fluid's box is in view
//!   ([`crate::culling::aabb_in_frustum`] with [`FluidSimulation::bounds`]) and from whether
//!   anything is disturbing it, whether to step it: [`FluidActivity`].
//!
//! The caller skips [`FluidSimulation::update_batched_with`] while it is not running, and turns
//! off the surface's extraction (`FluidSurfaceEffect::extract`, which keeps drawing the last
//! surface) or, out of view, the whole effect (`FluidSurfaceEffect::active`). The motion-matching
//! example's lake shows it.

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use bytemuck::{Pod, Zeroable};

use super::simulation::FluidSimulation;

/// Whether a fluid steps this frame, and why not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FluidActivity {
    /// Stepping, and its surface extracted, every frame.
    Running,
    /// Out of view long enough for its last waves to have died out: paused, and not drawn.
    Culled,
    /// In view but settled, with nothing near it: paused, its last surface drawn as it was.
    Asleep,
}

impl FluidActivity {
    pub fn name(self) -> &'static str {
        match self {
            Self::Running => "running",
            Self::Culled => "culled",
            Self::Asleep => "asleep",
        }
    }
}

/// When [`FluidSleep`] pauses a fluid.
#[derive(Debug, Clone, Copy)]
pub struct FluidSleepOptions {
    /// Seconds out of view before it is culled: it keeps stepping meanwhile, so waves started in
    /// view die out rather than freeze.
    pub cull_after: f32,
    /// The fastest a particle may move for the fluid to count as settled (the speeds given to
    /// [`FluidSleep::update`]; set it under what shows on screen).
    pub settle_speed: f32,
    /// Seconds every speed read must stay under `settle_speed` before it sleeps (and at least
    /// two reads: they may arrive seconds apart on a busy GPU).
    pub settle_after: f32,
}

impl Default for FluidSleepOptions {
    fn default() -> Self {
        Self { cull_after: 1.5, settle_speed: 0.05, settle_after: 1.0 }
    }
}

/// Decides each frame whether a fluid steps ([`FluidActivity`]): culled after `cull_after`
/// seconds out of view; asleep once in view, undisturbed, and every speed read (two at least)
/// over `settle_after` seconds of stepping has been under `settle_speed`; else running. A disturbance
/// (something near enough to touch it) or [`wake`](Self::wake) runs it at once (while in view);
/// coming back into view runs it again, unless it had settled before it was culled.
#[derive(Debug, Clone)]
pub struct FluidSleep {
    pub options: FluidSleepOptions,
    state: FluidActivity,
    out_of_view: f32,
    /// Seconds stepped since the speed was last over `settle_speed` (and since a disturbance)
    settled: f32,
    /// Speed reads since then (all under it)
    calm: u32,
}

impl FluidSleep {
    pub fn new(options: FluidSleepOptions) -> Self {
        Self { options, state: FluidActivity::Running, out_of_view: 0.0, settled: 0.0, calm: 0 }
    }

    pub fn state(&self) -> FluidActivity {
        self.state
    }

    /// Whether the fluid steps (and its surface is extracted) this frame.
    pub fn running(&self) -> bool {
        self.state == FluidActivity::Running
    }

    /// The fluid changed (a setting, a reset): run it until it settles again.
    pub fn wake(&mut self) {
        self.settled = 0.0;
        self.calm = 0;
        if self.state == FluidActivity::Asleep {
            self.state = FluidActivity::Running;
        }
    }

    /// Advance by `dt` seconds: whether the fluid's box is `in_view`, whether something is
    /// `disturbed`-ing it (near enough to touch it), and the fastest particle's `speed` if a new
    /// measurement arrived (taken while it was stepping). Returns this frame's state.
    pub fn update(&mut self, dt: f32, in_view: bool, disturbed: bool, speed: Option<f32>) -> FluidActivity {
        let o = self.options;
        self.out_of_view = if in_view { 0.0 } else { self.out_of_view + dt };
        if disturbed {
            self.settled = 0.0;
            self.calm = 0;
        } else if self.state == FluidActivity::Running {
            match speed {
                Some(s) if s > o.settle_speed || !s.is_finite() => {
                    self.settled = 0.0;
                    self.calm = 0;
                }
                Some(_) => self.calm = self.calm.saturating_add(1),
                None => {}
            }
            if self.calm > 0 {
                self.settled += dt;
            }
        }
        self.state = if self.out_of_view >= o.cull_after {
            FluidActivity::Culled
        } else if self.calm >= 2 && self.settled >= o.settle_after {
            FluidActivity::Asleep
        } else {
            FluidActivity::Running
        };
        self.state
    }
}

/// One measurement of the particles' speeds (the simulation's units per simulated second).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FluidSpeed {
    /// The fastest particle's speed.
    pub max: f32,
    /// How many particles moved faster than the probe's threshold.
    pub above: u32,
}

const PROBE_WGSL: &str = r#"
struct Params { count: u32, threshold: f32, _pad0: u32, _pad1: u32 };
@group(0) @binding(0) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> result: array<atomic<u32>, 2>;
@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> group_max: atomic<u32>;
var<workgroup> group_above: atomic<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
    if (li == 0u) {
        atomicStore(&group_max, 0u);
        atomicStore(&group_above, 0u);
    }
    workgroupBarrier();
    if (gid.x < params.count) {
        let s = length(velocities[gid.x].xyz);
        // non-negative floats order as their bits do
        atomicMax(&group_max, bitcast<u32>(s));
        if (s > params.threshold) {
            atomicAdd(&group_above, 1u);
        }
    }
    workgroupBarrier();
    if (li == 0u) {
        atomicMax(&result[0], atomicLoad(&group_max));
        atomicAdd(&result[1], atomicLoad(&group_above));
    }
}
"#;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ProbeParams {
    count: u32,
    threshold: f32,
    _pad: [u32; 2],
}

const MAPPING: u8 = 0;
const MAPPED: u8 = 1;
const FAILED: u8 = 2;

/// Measures a simulation's particle speeds on the GPU (a reduction to two words) and reads them
/// back without stalling: one measurement in flight at a time, collected by
/// [`take`](Self::take) once mapped (a frame or a few later); [`measure`](Self::measure) does
/// nothing while one is in flight.
pub struct FluidSpeedProbe {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params: wgpu::Buffer,
    result: wgpu::Buffer,
    staging: wgpu::Buffer,
    count: u32,
    threshold: f32,
    /// The measurement in flight, and whether it is still wanted
    pending: Option<(Arc<AtomicU8>, bool)>,
}

impl FluidSpeedProbe {
    /// A probe on `sim`'s velocities, counting the particles faster than `threshold`.
    pub fn new(sim: &FluidSimulation, threshold: f32) -> Self {
        let device = sim.gpu().0;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("FluidSpeedProbe"), source: wgpu::ShaderSource::Wgsl(PROBE_WGSL.into()) });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("FluidSpeedProbe"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let buffer = |label: &str, size: u64, usage: wgpu::BufferUsages| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false });
        let params = buffer("FluidSpeedProbe/Params", std::mem::size_of::<ProbeParams>() as u64, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST);
        let result = buffer("FluidSpeedProbe/Result", 8, wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST);
        let staging = buffer("FluidSpeedProbe/Readback", 8, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FluidSpeedProbe"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: sim.velocities_buffer().expect("FluidSimulation not initialized").as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: result.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: params.as_entire_binding() },
            ],
        });
        let probe = Self { pipeline, bind_group, params, result, staging, count: sim.particle_count(), threshold, pending: None };
        probe.upload(sim.gpu().1);
        probe
    }

    fn upload(&self, queue: &wgpu::Queue) {
        queue.write_buffer(&self.params, 0, bytemuck::bytes_of(&ProbeParams { count: self.count, threshold: self.threshold, _pad: [0; 2] }));
    }

    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Count the particles faster than `threshold` from the next measurement on.
    pub fn set_threshold(&mut self, queue: &wgpu::Queue, threshold: f32) {
        self.threshold = threshold;
        self.upload(queue);
    }

    /// Measure the speeds as the GPU has them after the work submitted so far (this frame's
    /// steps), unless a measurement is still in flight.
    pub fn measure(&mut self, sim: &FluidSimulation) {
        if self.pending.is_some() {
            return;
        }
        let (device, queue) = sim.gpu();
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FluidSpeedProbe") });
        encoder.clear_buffer(&self.result, 0, None);
        {
            let stamp = crate::profiling::gpu_pass("FluidSpeedProbe");
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("FluidSpeedProbe"), timestamp_writes: stamp.as_ref().map(crate::profiling::PassStamp::compute) });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(256), 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.result, 0, &self.staging, 0, 8);
        queue.submit(Some(encoder.finish()));
        let state = Arc::new(AtomicU8::new(MAPPING));
        let done = state.clone();
        self.staging.slice(..).map_async(wgpu::MapMode::Read, move |r| done.store(if r.is_ok() { MAPPED } else { FAILED }, Ordering::Release));
        self.pending = Some((state, true));
    }

    /// Drop the measurement in flight (taken before the fluid changed): `take` will not return it.
    pub fn forget(&mut self) {
        if let Some((_, wanted)) = &mut self.pending {
            *wanted = false;
        }
    }

    /// The measurement in flight, once it has arrived (each once); `None` until then.
    pub fn take(&mut self, device: &wgpu::Device) -> Option<FluidSpeed> {
        #[cfg(not(target_arch = "wasm32"))]
        if self.pending.is_some() {
            device.poll(wgpu::Maintain::Poll);
        }
        #[cfg(target_arch = "wasm32")]
        let _ = device;
        let (state, wanted) = self.pending.take_if(|(s, _)| s.load(Ordering::Acquire) != MAPPING)?;
        if state.load(Ordering::Acquire) == FAILED {
            return None;
        }
        let words: [u32; 2] = {
            let bytes = self.staging.slice(..).get_mapped_range();
            let w: &[u32] = bytemuck::cast_slice(&bytes);
            [w[0], w[1]]
        };
        self.staging.unmap();
        wanted.then(|| FluidSpeed { max: f32::from_bits(words[0]), above: words[1] })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_probe_shader_validates_and_its_params_match() {
        let module = naga::front::wgsl::parse_str(PROBE_WGSL).unwrap_or_else(|e| panic!("{}", e.emit_to_string(PROBE_WGSL)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all()).validate(&module).unwrap();
        let span = module.types.iter().find_map(|(_, t)| match (&t.name, &t.inner) {
            (Some(n), naga::TypeInner::Struct { span, .. }) if n == "Params" => Some(*span as usize),
            _ => None,
        });
        assert_eq!(span, Some(std::mem::size_of::<ProbeParams>()));
    }

    const DT: f32 = 1.0 / 60.0;

    /// Run `sleep` for `seconds` at 60 fps, a speed read every third frame.
    fn run(sleep: &mut FluidSleep, seconds: f32, in_view: bool, disturbed: bool, speed: f32) -> FluidActivity {
        let mut state = sleep.state();
        for frame in 0..(seconds / DT).round() as u32 {
            state = sleep.update(DT, in_view, disturbed, (frame % 3 == 0).then_some(speed));
        }
        state
    }

    #[test]
    fn out_of_view_it_runs_through_the_grace_then_is_culled_and_comes_back_at_once() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        assert_eq!(run(&mut s, 1.0, false, false, 1.0), FluidActivity::Running, "still in its grace period");
        assert_eq!(run(&mut s, 0.6, false, false, 1.0), FluidActivity::Culled);
        // culled whatever touches it
        assert_eq!(s.update(DT, false, true, None), FluidActivity::Culled);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Running, "back in view: running that frame");
        // a glance away shorter than the grace never culls it
        assert_eq!(run(&mut s, 1.0, false, false, 1.0), FluidActivity::Running);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Running);
    }

    #[test]
    fn it_sleeps_once_settled_and_wakes_when_disturbed() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        assert_eq!(run(&mut s, 3.0, true, false, 0.5), FluidActivity::Running, "moving");
        assert_eq!(run(&mut s, 0.9, true, false, 0.01), FluidActivity::Running, "not settled long enough");
        assert_eq!(run(&mut s, 0.2, true, false, 0.01), FluidActivity::Asleep);
        // asleep, no speed reads arrive, and it stays asleep
        assert_eq!(run(&mut s, 5.0, true, false, f32::NAN), FluidActivity::Asleep);
        // something comes near: running that frame, and while it stays
        assert_eq!(s.update(DT, true, true, None), FluidActivity::Running);
        assert_eq!(run(&mut s, 2.0, true, true, 0.0), FluidActivity::Running, "a still collider in the water keeps it awake");
        // it leaves: running until the water settles again
        assert_eq!(run(&mut s, 0.5, true, false, 0.01), FluidActivity::Running);
        assert_eq!(run(&mut s, 0.6, true, false, 0.01), FluidActivity::Asleep);
    }

    #[test]
    fn one_fast_read_restarts_the_settling() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        run(&mut s, 0.8, true, false, 0.01);
        s.update(DT, true, false, Some(0.2));
        assert_eq!(run(&mut s, 0.8, true, false, 0.01), FluidActivity::Running);
        assert_eq!(run(&mut s, 0.3, true, false, 0.01), FluidActivity::Asleep);
        // a NaN (a broken fluid) never counts as settled
        s.wake();
        assert_eq!(run(&mut s, 3.0, true, false, f32::NAN), FluidActivity::Running);
    }

    #[test]
    fn it_waits_for_a_second_read_however_late() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        assert_eq!(s.update(DT, true, false, Some(0.0)), FluidActivity::Running);
        for _ in 0..180 {
            assert_eq!(s.update(DT, true, false, None), FluidActivity::Running);
        }
        assert_eq!(s.update(DT, true, false, Some(0.0)), FluidActivity::Asleep);
    }

    #[test]
    fn a_wake_runs_it_until_it_settles_again() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        assert_eq!(run(&mut s, 1.2, true, false, 0.0), FluidActivity::Asleep);
        s.wake();
        assert_eq!(s.state(), FluidActivity::Running);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Running, "needs a fresh read");
        assert_eq!(run(&mut s, 1.1, true, false, 0.0), FluidActivity::Asleep);
    }

    #[test]
    fn settled_before_it_was_culled_it_comes_back_asleep_else_running() {
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        run(&mut s, 1.2, true, false, 0.0);
        assert_eq!(run(&mut s, 2.0, false, false, 0.0), FluidActivity::Culled);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Asleep);

        let mut s = FluidSleep::new(FluidSleepOptions::default());
        run(&mut s, 1.0, true, false, 1.0);
        assert_eq!(run(&mut s, 2.0, false, false, 1.0), FluidActivity::Culled);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Running);
        // a wake while culled (a reset) runs it when it is back in view
        let mut s = FluidSleep::new(FluidSleepOptions::default());
        run(&mut s, 1.2, true, false, 0.0);
        run(&mut s, 2.0, false, false, 0.0);
        s.wake();
        assert_eq!(s.state(), FluidActivity::Culled);
        assert_eq!(s.update(DT, true, false, None), FluidActivity::Running);
    }
}
