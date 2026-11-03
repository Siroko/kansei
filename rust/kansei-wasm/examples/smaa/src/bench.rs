//! `bench=1`: SMAA and TAA in turn inside one page, the scene held still, each one's
//! anti-aliasing GPU time (the sum of its passes' `exclusive_ms`, a span rather than one pass's
//! timestamps, as Apple GPUs overlap passes) beside the frame's and the velocity pass's.
//!
//! After a 3 s warm-up, phases of 8 s alternate the two, `rounds` times. 4 s into a phase
//! the renderer's profile is taken and dropped (the switch, late readbacks); at its end it is
//! taken again: that is the phase's window.

use kansei_core::profiling::FrameProfile;
use kansei_core::renderers::Renderer;

use crate::Aa;

const WARMUP_MS: f64 = 3000.0;
const PHASE_MS: f64 = 8000.0;
const SETTLE_MS: f64 = 4000.0;

/// A phase's window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Window {
    /// The mode's anti-aliasing passes, ms a frame; `None` without GPU timestamps.
    pub aa_ms: Option<f64>,
    /// The frame's passes, ms.
    pub gpu_ms: f64,
    /// The renderer's velocity pass, ms (0 when it didn't run).
    pub velocity_ms: f64,
    /// Between frames, ms.
    pub interval_ms: f64,
}

impl Window {
    /// The window of `mode` from its profile, over `elapsed_ms` and `frames`.
    pub fn from_profile(profile: &FrameProfile, mode: Aa, elapsed_ms: f64, frames: u32) -> Self {
        let sum = |pick: &dyn Fn(&str) -> bool| profile.gpu.iter().filter(|p| pick(p.label)).map(|p| p.exclusive_ms).sum::<f64>();
        let aa = match mode {
            Aa::Taa => sum(&|l| l == "TAA/Resolve"),
            _ => sum(&|l| l.starts_with("SMAA")),
        };
        Self {
            aa_ms: (profile.gpu_frames > 0).then_some(aa),
            gpu_ms: profile.gpu_ms,
            velocity_ms: sum(&|l| l == "Renderer/VelocityPass"),
            interval_ms: elapsed_ms / frames.max(1) as f64,
        }
    }
}

pub struct Bench {
    start_ms: f64,
    rounds: u32,
    size: (u32, u32),
    /// The phase the last frame was in.
    phase: Option<u32>,
    /// The open window: when it began, and the frames since.
    window: Option<(f64, u32)>,
    /// By phase (round x 2 + mode).
    windows: Vec<Option<Window>>,
    done: bool,
}

impl Bench {
    pub const MODES: [Aa; 2] = [Aa::Smaa, Aa::Taa];
    const N: u32 = Self::MODES.len() as u32;

    /// A bench of `rounds` rounds, starting at `now_ms`, of a drawing buffer of `size`.
    pub fn new(now_ms: f64, rounds: u32, size: (u32, u32)) -> Self {
        Self { start_ms: now_ms, rounds, size, phase: None, window: None, windows: vec![None; (rounds * Self::N) as usize], done: false }
    }

    /// The phase at `now_ms`; `None` in the warm-up.
    fn phase_at(&self, now_ms: f64) -> Option<(u32, f64)> {
        let t = now_ms - self.start_ms - WARMUP_MS;
        (t >= 0.0).then(|| ((t / PHASE_MS) as u32, t % PHASE_MS))
    }

    /// The mode to draw at `now_ms` (the first through the warm-up); `None` once done.
    pub fn mode(&self, now_ms: f64) -> Option<Aa> {
        match self.phase_at(now_ms) {
            None => Some(Self::MODES[0]),
            Some((p, _)) if p < self.rounds * Self::N => Some(Self::MODES[(p % Self::N) as usize]),
            Some(_) => None,
        }
    }

    /// Call once a frame, before rendering it: closes and opens the windows. Returns the report,
    /// once, when the last phase is over.
    pub fn frame(&mut self, renderer: &Renderer, now_ms: f64) -> Option<String> {
        let (phase, within) = self.phase_at(now_ms)?;
        if let Some((_, frames)) = self.window.as_mut() {
            *frames += 1;
        }
        if self.phase != Some(phase) {
            if let (Some(previous), Some((start, frames))) = (self.phase, self.window.take()) {
                let mode = Self::MODES[(previous % Self::N) as usize];
                self.windows[previous as usize] = Some(Window::from_profile(&renderer.take_profile(), mode, now_ms - start, frames));
            }
            self.phase = Some(phase);
        }
        if phase >= self.rounds * Self::N {
            if self.done {
                return None;
            }
            self.done = true;
            return Some(self.report());
        }
        if within >= SETTLE_MS && self.window.is_none() {
            let _ = renderer.take_profile();
            self.window = Some((now_ms, 0));
        }
        None
    }

    /// The report: the means, each round's AA times, and the verdict.
    pub fn report(&self) -> String {
        report(self.size, self.rounds, &self.windows)
    }
}

fn mean(values: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let values: Vec<f64> = values.flatten().collect();
    (!values.is_empty()).then(|| values.iter().sum::<f64>() / values.len() as f64)
}

fn ms(value: Option<f64>) -> String {
    value.map_or("no GPU timestamps".to_string(), |v| format!("{v:.3}"))
}

/// `windows` by phase (round x 2 + mode: SMAA, then TAA).
pub fn report(size: (u32, u32), rounds: u32, windows: &[Option<Window>]) -> String {
    let at = |r: usize, mode: usize| windows.get(r * 2 + mode).copied().flatten();
    let of = |mode: usize| (0..rounds as usize).map(move |r| at(r, mode));
    let aa = |mode| mean(of(mode).map(|w| w.and_then(|w| w.aa_ms)));
    let field = |mode, f: fn(&Window) -> f64| mean(of(mode).map(|w| w.map(|w| f(&w))));
    let both = |f: fn(&Window) -> f64| format!("{} / {}", ms(field(0, f)), ms(field(1, f)));
    let mut out = format!(
        "bench {}x{}, {rounds} rounds: SMAA {} ms | TAA {} ms; frame GPU {} ms; interval {} ms; velocity pass {} ms\n",
        size.0,
        size.1,
        ms(aa(0)),
        ms(aa(1)),
        both(|w| w.gpu_ms),
        both(|w| w.interval_ms),
        both(|w| w.velocity_ms),
    );
    for r in 0..rounds as usize {
        out += &format!("round {}: SMAA {} | TAA {} ms\n", r + 1, ms(at(r, 0).and_then(|w| w.aa_ms)), ms(at(r, 1).and_then(|w| w.aa_ms)));
    }
    // cheaper: a lower mean, and lower in at least 3 of 4 paired rounds (every window timed)
    let pairs: Vec<Option<(f64, f64)>> = (0..rounds as usize).map(|r| Some((at(r, 0)?.aa_ms?, at(r, 1)?.aa_ms?))).collect();
    let verdict = if rounds == 0 || pairs.iter().any(Option::is_none) {
        "no (a window had no GPU timestamps)"
    } else {
        let wins = pairs.iter().flatten().filter(|(a, b)| a < b).count() as u32;
        let lower_mean = aa(0).zip(aa(1)).is_some_and(|(a, b)| a < b);
        if lower_mean && wins * 4 >= rounds * 3 { "yes" } else { "no" }
    };
    out += &format!("SMAA cheaper than TAA: {verdict}");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use kansei_core::profiling::PassTime;

    fn window(aa: f64) -> Option<Window> {
        Some(Window { aa_ms: Some(aa), gpu_ms: 10.0, velocity_ms: 0.5, interval_ms: 16.0 })
    }

    #[test]
    fn the_verdict_wants_a_lower_mean_and_three_rounds_in_four() {
        // SMAA 0.3 against TAA 0.5 in every round
        let all: Vec<_> = (0..4).flat_map(|_| [window(0.3), window(0.5)]).collect();
        let r = report((1920, 1080), 4, &all);
        assert!(r.starts_with("bench 1920x1080, 4 rounds: SMAA 0.300 ms | TAA 0.500 ms; frame GPU 10.000 / 10.000 ms; interval 16.000 / 16.000 ms; velocity pass 0.500 / 0.500 ms"), "{r}");
        assert!(r.contains("round 4: SMAA 0.300 | TAA 0.500 ms"), "{r}");
        assert!(r.ends_with("SMAA cheaper than TAA: yes"), "{r}");
        // a lower mean (0.305 against 0.5), but only 2 rounds of 4 lower
        let mut split = all.clone();
        split[0] = window(0.51);
        split[2] = window(0.51);
        split[4] = window(0.1);
        split[6] = window(0.1);
        assert!(report((1920, 1080), 4, &split).ends_with("SMAA cheaper than TAA: no"));
        // a window with no timestamps
        let mut missing = all;
        missing[3] = Some(Window { aa_ms: None, ..window(0.0).unwrap() });
        let r = report((1920, 1080), 4, &missing);
        assert!(r.contains("round 2: SMAA 0.300 | TAA no GPU timestamps ms"), "{r}");
        assert!(r.ends_with("SMAA cheaper than TAA: no (a window had no GPU timestamps)"), "{r}");
    }

    #[test]
    fn a_window_sums_its_mode_s_passes() {
        let pass = |label, exclusive_ms| PassTime { label, exclusive_ms, busy_ms: exclusive_ms, count: 1.0 };
        let profile = FrameProfile {
            gpu_frames: 30,
            gpu: vec![pass("SMAA/Edges", 0.1), pass("SMAA/Weights", 0.2), pass("SMAA/Blend", 0.05), pass("TAA/Resolve", 0.7), pass("Renderer/VelocityPass", 0.4)],
            gpu_ms: 9.0,
            ..Default::default()
        };
        let w = Window::from_profile(&profile, Aa::Smaa, 400.0, 50);
        assert!((w.aa_ms.unwrap() - 0.35).abs() < 1e-9 && w.velocity_ms == 0.4 && w.interval_ms == 8.0 && w.gpu_ms == 9.0);
        assert_eq!(Window::from_profile(&profile, Aa::Taa, 1.0, 1).aa_ms, Some(0.7));
        assert_eq!(Window::from_profile(&FrameProfile::default(), Aa::Smaa, 1.0, 1).aa_ms, None);
    }

    #[test]
    fn the_phases_go_round_the_modes_after_the_warm_up() {
        let b = Bench::new(1000.0, 4, (1920, 1080));
        assert_eq!(b.mode(1000.0), Some(Aa::Smaa));
        assert_eq!(b.mode(1000.0 + WARMUP_MS + 1.0), Some(Aa::Smaa));
        assert_eq!(b.mode(1000.0 + WARMUP_MS + PHASE_MS + 1.0), Some(Aa::Taa));
        assert_eq!(b.mode(1000.0 + WARMUP_MS + 2.0 * PHASE_MS + 1.0), Some(Aa::Smaa));
        assert_eq!(b.mode(1000.0 + WARMUP_MS + 7.0 * PHASE_MS + 1.0), Some(Aa::Taa));
        assert_eq!(b.mode(1000.0 + WARMUP_MS + 8.0 * PHASE_MS + 1.0), None);
    }
}
