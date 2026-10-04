//! Stepping a fluid from the frame loop: fixed steps ([`crate::pacing::FixedStep`]) in the
//! simulation's own units and time ([`WorldScale`]), and rest ([`FluidSleep`] with a
//! [`FluidSpeedProbe`]) when the stepper is given it. The lake example shows all three.

use crate::pacing::FixedStep;

use super::activity::{FluidActivity, FluidSleep, FluidSleepOptions, FluidSpeed, FluidSpeedProbe};
use super::simulation::FluidSimulation;

/// How a simulation's units relate to the world's: `length` simulation units a metre, and
/// `time` simulated seconds a real second. A fluid tuned at one size (its smoothing radius,
/// its spacing) runs in a world of another this way: positions and lengths go in times
/// `length`, velocities times `length / time`, and each real second steps `time` simulated ones.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldScale {
    pub length: f32,
    pub time: f32,
}

impl Default for WorldScale {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl WorldScale {
    /// The simulation's units are metres and seconds.
    pub const IDENTITY: Self = Self { length: 1.0, time: 1.0 };

    /// `length` simulation units a metre, with simulated time running √`length` times faster
    /// than real time: a simulation whose gravity is 9.8 of its units per second² then falls at
    /// the real pace.
    pub fn with_real_gravity(length: f32) -> Self {
        Self { length, time: length.sqrt() }
    }

    /// A point (metres) in simulation units.
    pub fn point_to_sim(&self, p: [f32; 3]) -> [f32; 3] {
        p.map(|v| v * self.length)
    }

    /// A point in simulation units, in metres.
    pub fn point_to_world(&self, p: [f32; 3]) -> [f32; 3] {
        p.map(|v| v / self.length)
    }

    /// A length (metres) in simulation units.
    pub fn length_to_sim(&self, m: f32) -> f32 {
        m * self.length
    }

    /// A length in simulation units, in metres.
    pub fn length_to_world(&self, l: f32) -> f32 {
        l / self.length
    }

    /// A velocity (m/s) in simulation units per simulated second.
    pub fn velocity_to_sim(&self, v: [f32; 3]) -> [f32; 3] {
        v.map(|c| self.speed_to_sim(c))
    }

    /// A speed (m/s) in simulation units per simulated second.
    pub fn speed_to_sim(&self, s: f32) -> f32 {
        s * self.length / self.time
    }

    /// A speed in simulation units per simulated second, in m/s.
    pub fn speed_to_world(&self, s: f32) -> f32 {
        s * self.time / self.length
    }

    /// Real seconds as simulated seconds.
    pub fn sim_seconds(&self, dt: f32) -> f32 {
        dt * self.time
    }
}

struct Rest {
    sleep: FluidSleep,
    probe: FluidSpeedProbe,
    enabled: bool,
}

/// Fixed steps for a fluid, at a [`WorldScale`], resting when it may.
///
/// Each frame, [`advance`](Self::advance) says how many steps of [`step_dt`](Self::step_dt)
/// simulated seconds to run. With [`with_rest`](Self::with_rest), call
/// [`update_rest`](Self::update_rest) first: while the fluid is culled or asleep, `advance`
/// returns 0 (and carries no time over), and the caller shows the state through
/// `FluidSurfaceEffect::set_activity`. After the steps, [`stepped`](Self::stepped) measures the
/// speed it settles by.
///
/// ```ignore
/// let state = stepper.update_rest(&sim, dt, in_view, disturbed);
/// surface.set_activity(state);
/// let steps = stepper.advance(dt);
/// for _ in 0..steps {
///     sim.update_batched_with(stepper.step_dt(), 0.0, [0.0; 2], [0.0; 2], &passes);
/// }
/// stepper.stepped(&sim, steps);
/// ```
pub struct FluidStepper {
    fixed: FixedStep,
    scale: WorldScale,
    rest: Option<Rest>,
    speed: Option<FluidSpeed>,
}

impl FluidStepper {
    /// Steps of `step` real seconds, at most `max_steps` a frame (time beyond is dropped), each
    /// `step * scale.time` simulated seconds.
    pub fn new(step: f32, max_steps: u32, scale: WorldScale) -> Self {
        Self { fixed: FixedStep::new(step as f64).with_max_steps(max_steps), scale, rest: None, speed: None }
    }

    /// Let the fluid rest by `options`, whose speeds are in m/s (the stepper converts through
    /// its scale), measuring `sim`'s particles.
    pub fn with_rest(mut self, sim: &FluidSimulation, options: FluidSleepOptions) -> Self {
        let probe = FluidSpeedProbe::new(sim, self.scale.speed_to_sim(options.settle_speed));
        self.rest = Some(Rest { sleep: FluidSleep::new(options), probe, enabled: true });
        self
    }

    pub fn scale(&self) -> WorldScale {
        self.scale
    }

    /// Run simulated time `time` times as fast as real time from now on (the speed probe's
    /// threshold follows).
    pub fn set_time_scale(&mut self, sim: &FluidSimulation, time: f32) {
        self.scale.time = time;
        if let Some(rest) = &mut self.rest {
            let threshold = self.scale.speed_to_sim(rest.sleep.options.settle_speed);
            rest.probe.set_threshold(sim.gpu().1, threshold);
        }
    }

    /// Real seconds a step stands for.
    pub fn step(&self) -> f32 {
        self.fixed.step as f32
    }

    /// Simulated seconds a step advances: the `dt` to step the simulation (and emit) with.
    pub fn step_dt(&self) -> f32 {
        self.scale.sim_seconds(self.fixed.step as f32)
    }

    /// Whether the fluid may rest (with `with_rest`); off, it steps every frame, in view or not.
    pub fn set_rest_enabled(&mut self, enabled: bool) {
        if let Some(rest) = &mut self.rest {
            rest.enabled = enabled;
        }
    }

    pub fn rest_enabled(&self) -> bool {
        self.rest.as_ref().is_some_and(|r| r.enabled)
    }

    /// Decide this frame's state from whether the fluid's box is `in_view`, whether something
    /// is `disturbed`-ing it, and the latest speed read. Always running without `with_rest`.
    pub fn update_rest(&mut self, sim: &FluidSimulation, dt: f32, in_view: bool, disturbed: bool) -> FluidActivity {
        let Some(rest) = &mut self.rest else { return FluidActivity::Running };
        let scale = self.scale;
        let speed = rest.probe.take(sim.gpu().0).map(|s| {
            let s = FluidSpeed { max: scale.speed_to_world(s.max), above: s.above };
            self.speed = Some(s);
            s.max
        });
        let state = rest.sleep.update(dt, in_view || !rest.enabled, disturbed || !rest.enabled, speed);
        if state != FluidActivity::Running {
            self.fixed.reset();
        }
        state
    }

    /// The steps to run for a frame of `dt` real seconds: none while the fluid rests.
    pub fn advance(&mut self, dt: f32) -> u32 {
        if self.state() != FluidActivity::Running {
            return 0;
        }
        self.fixed.advance(dt as f64)
    }

    /// After the frame's `steps` were submitted: measure the speed they left (with rest, when
    /// any ran and no measurement is in flight).
    pub fn stepped(&mut self, sim: &FluidSimulation, steps: u32) {
        if let (Some(rest), true) = (&mut self.rest, steps > 0) {
            rest.probe.measure(sim);
        }
    }

    /// Running, culled or asleep (always running without rest).
    pub fn state(&self) -> FluidActivity {
        self.rest.as_ref().map_or(FluidActivity::Running, |r| r.sleep.state())
    }

    /// The last speed read, in m/s: the fastest particle, and how many moved faster than the
    /// settle speed. `None` before the first read, or without rest.
    pub fn speed(&self) -> Option<FluidSpeed> {
        self.speed
    }

    /// The fluid changed (a reset, a setting): run it until it settles again, and drop the speed
    /// read in flight (taken before the change).
    pub fn wake(&mut self) {
        if let Some(rest) = &mut self.rest {
            rest.sleep.wake();
            rest.probe.forget();
        }
    }

    /// Forget the time carried over (a pause, a reset).
    pub fn reset(&mut self) {
        self.fixed.reset();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_world_scale_converts_both_ways() {
        let s = WorldScale::with_real_gravity(11.0);
        assert!((s.time - 11f32.sqrt()).abs() < 1e-6);
        assert_eq!(s.point_to_sim([1.0, 2.0, -3.0]), [11.0, 22.0, -33.0]);
        assert!((s.speed_to_world(s.speed_to_sim(2.5)) - 2.5).abs() < 1e-6);
        // gravity of 9.8 units/s² in the simulation falls 9.8 m/s² in the world
        let g = 9.8 / s.length * s.time * s.time;
        assert!((g - 9.8).abs() < 1e-4);
    }

    #[test]
    fn without_rest_it_steps_by_the_clock() {
        let mut stepper = FluidStepper::new(1.0 / 60.0, 2, WorldScale { length: 1.0, time: 3.0 });
        assert!((stepper.step_dt() - 0.05).abs() < 1e-6);
        assert_eq!(stepper.state(), FluidActivity::Running);
        assert_eq!(stepper.advance(1.0 / 60.0), 1);
        assert_eq!(stepper.advance(0.5), 2, "capped");
    }
}
