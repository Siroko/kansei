//! Frame pacing: rendering on a steady share of the display's refreshes.
//!
//! A browser calls `requestAnimationFrame` once per display refresh (every 16.7 ms at 60 Hz, 8.3 ms
//! at 120 Hz). A frame whose GPU work takes a little longer than a refresh is shown a refresh late,
//! the next may not be, and the cadence alternates between one refresh and two: an uneven picture,
//! though the average frame rate looks fine. `FramePacer` renders on every k-th refresh instead
//! (every other one on a 120 Hz display is a steady 60 fps), k the fewest that keep up:
//! - it renders on fewer (k + 1) when refreshes are being missed (the browser calls a refresh
//!   late while the frames before it are still on the GPU): over two windows running, or many
//!   over one, and only while the frames' median GPU time does not fit k refreshes;
//! - and when frames keep spilling past k refreshes (one in ten or more, on nearly every refresh
//!   over two windows running), missed refreshes or not: the browser keeps a frame or two in
//!   flight, so the screen shows uneven intervals before any refresh is missed. Only where the
//!   slower cadence keeps `spill_floor_fps` (on a 120 Hz display, 60 fps goes to 40; on a 60 Hz
//!   one, 30 would be worse than the spills);
//! - a burst of slow frames (a cut, a rebuild), or frames one step slower would not hold either,
//!   is a hitch, not a slower cadence;
//! - it renders when k refresh intervals have passed since the last rendered frame, by the
//!   refreshes' timestamps: a late refresh does not push the next frame a refresh further;
//! - after slowing down from a cadence it held, it comes back to it as soon as the latest frames
//!   fit it with room (`fit_share`), whatever the wait, a step or several at a time;
//! - otherwise it tries more (k - 1) blind once k has held without a miss for `settle_ms` (8 s:
//!   a try that fails shows as a couple of seconds of uneven frames), and keeps them once they
//!   have held for `probation_ms` with frames that rarely spill; if they don't, it goes back and
//!   waits twice as long before trying that cadence again (up to `max_backoff_ms`), each cadence
//!   its own wait;
//! - `max_fps` caps it (60 on a 120 Hz display, where a steady 60 is the aim).
//!
//! The frames' GPU time (`FrameTimer`) is measured too. A GPU with fewer frames to draw lowers its
//! clocks (Apple's do), so each frame takes longer at a slower cadence. When the latest frames fit
//! a faster cadence with room even at the slower one's clocks, and their median is below what this
//! cadence cost just before that faster one last failed, the pacer tries it at once, whatever the
//! wait (the cost fell: a lighter shot after a heavier one).
//!
//! `FramePacer::take_report` says what it saw and did (the refresh and frame intervals, the GPU
//! time, each change and why): log it on a display where the pacing misbehaves.
//!
//! ```ignore
//! let mut pacer = FramePacer::new(renderer.device(), renderer.queue(), FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
//! // each requestAnimationFrame, with its timestamp:
//! if pacer.on_refresh(now_ms) {
//!     pacer.begin_frame();   // before the frame's first submit
//!     renderer.render_with_postprocessing(&mut scene, &mut camera, &mut volume);
//!     pacer.end_frame();     // after its last
//! }
//! ```
//!
//! Advance animation by the time between rendered frames, not by a fixed step per call.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

/// How `FramePacer` picks its cadence.
#[derive(Debug, Clone, Copy)]
pub struct FramePacerOptions {
    /// The fastest to render, frames per second (60 keeps a 120 Hz display at every other
    /// refresh); `None`: every refresh.
    pub max_fps: Option<f64>,
    /// The most refreshes per frame.
    pub max_divisor: u32,
    /// A refresh is missed when it comes this many refresh intervals after the one before.
    pub late_factor: f64,
    /// The share of the refreshes that may be missed over each of two `window_ms` running.
    pub late_share: f64,
    /// Missing this share of the refreshes over one `window_ms` slows down at once.
    pub late_share_at_once: f64,
    pub window_ms: f64,
    /// How long a cadence must hold without a miss before a faster one is tried, ms. Each try
    /// that fails shows as a second or so of uneven frames, so tries come rarely.
    pub settle_ms: f64,
    /// The longest wait before trying a faster cadence that failed again, ms.
    pub max_backoff_ms: f64,
    /// How long a faster cadence must hold before it is kept, ms: slowing down within it, or
    /// frames that don't fit it at its end, is a failed try (the next waits twice as long).
    pub probation_ms: f64,
    /// Frames that keep spilling past the cadence slow it down only while the slower cadence
    /// renders at least this many frames per second (40: on a 120 Hz display, 60 fps goes to 40
    /// rather than showing uneven intervals; on a 60 Hz display the next step down is 30, and
    /// spills stay hitches).
    pub spill_floor_fps: f64,
    /// How long after the first refresh misses are not counted (loading, warming up), ms.
    pub warmup_ms: f64,
    /// Missed refreshes slow a cadence only while recent frames' GPU time (nine in ten of the
    /// last `GPU_FRAMES`) is over its frame time; while it fits, they are hitches (a burst of
    /// slow frames, a stall elsewhere) that a slower cadence would not remove. A faster cadence
    /// is tried at once when the latest frames (nine in ten) are below this share of its frame
    /// time (and their median well below what it was when that cadence last failed).
    pub fit_share: f64,
}

impl Default for FramePacerOptions {
    fn default() -> Self {
        Self { max_fps: None, max_divisor: 4, late_factor: 1.5, late_share: 0.05, late_share_at_once: 0.2, window_ms: 1000.0, settle_ms: 8000.0, max_backoff_ms: 120000.0, probation_ms: 10000.0, spill_floor_fps: 40.0, warmup_ms: 1000.0, fit_share: 0.9 }
    }
}

/// The pacing decisions: the refresh interval from the refreshes' timestamps, and the divisor
/// from the refreshes missed.
#[derive(Debug, Clone)]
pub(crate) struct Cadence {
    options: FramePacerOptions,
    intervals: VecDeque<f64>,
    first_refresh: Option<f64>,
    last_refresh: Option<f64>,
    refresh_ms: f64,
    /// When an interval about as short as `refresh_ms` was last seen
    last_short: f64,
    /// The last two `window_ms` of refreshes: when, and whether missed
    recent: VecDeque<(f64, bool)>,
    /// Recent rendered frames' GPU time, ms
    gpu: VecDeque<f64>,
    /// The last two `window_ms` of refreshes: when, and whether frames were spilling then
    spills: VecDeque<(f64, bool)>,
    divisor: u32,
    /// When the divisor last changed, and the try at a faster cadence still on probation
    changed_at: f64,
    trying: Option<Try>,
    /// Per divisor: the wait before trying it (after it failed), and the slower cadence's median
    /// GPU time when a try at it last failed (∞: none failed since it was last kept)
    backoff_ms: Vec<f64>,
    failed_base: Vec<f64>,
    /// The divisor the pacer held before it last slowed down from a held cadence (not a failed
    /// try): it comes back to it as soon as frames fit it
    home: Option<u32>,
    /// When the last frame was rendered
    last_render: Option<f64>,
    /// What `take_report` reports: since when, the refresh and frame intervals (in refreshes:
    /// 1, 2, 3, 4, 5 or more), the refreshes missed and the divisor's changes
    report_start: Option<f64>,
    report_refreshes: [u32; 5],
    report_frames: [u32; 5],
    report_missed: u32,
    report_stalls: u32,
    report_changes: Vec<PacerChange>,
}

/// A faster cadence on probation: the slower cadence's median GPU time just before it, whether
/// it was blind (no sign the frames would fit), and on how many of its refreshes frames spilled.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Try {
    base: f64,
    blind: bool,
    over: u32,
    refreshes: u32,
}

/// A change of `FramePacer`'s divisor: when (ms, the refreshes' clock), to what, and why:
/// "floor" (`max_fps`), "misses" (refreshes missed while frames don't fit), "spilling" (one frame
/// in ten or more over the cadence), "try failed", "try" (held long enough), "lighter" (the
/// latest frames fit a faster cadence).
#[derive(Debug, Clone, PartialEq)]
pub struct PacerChange {
    pub at_ms: f64,
    pub divisor: u32,
    pub reason: &'static str,
}

/// What `FramePacer` saw and did since the last report (`FramePacer::take_report`), to log on a
/// display the pacing misbehaves on: `Display` prints it on a line.
#[derive(Debug, Clone)]
pub struct PacerReport {
    pub seconds: f64,
    /// The display's refresh interval as measured, the divisor now, and the one it holds (the
    /// same but during a blind try at a faster cadence).
    pub refresh_ms: f64,
    pub divisor: u32,
    pub holding: u32,
    /// The intervals between refreshes (requestAnimationFrame), and between rendered frames, in
    /// refreshes: 1, 2, 3, 4, 5 or more.
    pub refresh_intervals: [u32; 5],
    pub frame_intervals: [u32; 5],
    pub missed: u32,
    /// Gaps between refreshes of a quarter second or more (a stall, a hidden tab), not counted
    /// as intervals.
    pub stalls: u32,
    /// Recent frames' GPU time, ms (NaN before any).
    pub gpu_p50: f64,
    pub gpu_p90: f64,
    pub changes: Vec<PacerChange>,
}

impl std::fmt::Display for PacerReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let counts = |c: &[u32; 5]| c.iter().enumerate().map(|(i, n)| format!("{}{}:{n}", i + 1, if i == 4 { "+" } else { "" })).collect::<Vec<_>>().join(" ");
        write!(
            f,
            "pacer: {:.1} s of {:.2} ms refreshes, every {}{} · refreshes {} · frames {} · missed {} · stalls {} · GPU p50 {:.1} p90 {:.1} ms",
            self.seconds,
            self.refresh_ms,
            self.divisor,
            if self.holding != self.divisor { format!(" (a try; holds {})", self.holding) } else { String::new() },
            counts(&self.refresh_intervals),
            counts(&self.frame_intervals),
            self.missed,
            self.stalls,
            self.gpu_p50,
            self.gpu_p90
        )?;
        for c in &self.changes {
            write!(f, " · {:.1} s → {} ({})", c.at_ms / 1000.0, c.divisor, c.reason)?;
        }
        Ok(())
    }
}

impl Cadence {
    const RELEARN_MS: f64 = 30000.0;
    /// Rendered frames whose GPU time the median is taken over (a second at 60 fps).
    const GPU_FRAMES: usize = 60;
    /// The latest frames a lighter scene is judged by (a quarter second at 60 fps): coming back
    /// to a faster cadence should not wait for the heavier frames to leave `GPU_FRAMES`.
    const RECENT_FRAMES: usize = 15;
    /// Frames spill over a cadence when they do on this share of the refreshes over two windows.
    const SPILLING: f64 = 0.8;
    /// A try is kept when frames spilled on less than this share of its probation's refreshes.
    const KEEP: f64 = 0.2;
    /// Changes a report keeps while nobody takes it.
    const MAX_CHANGES: usize = 64;
    /// How much lighter than when a faster cadence last failed the median must be for it to be
    /// tried at once: frames at a slower cadence measure lighter by themselves (less of a shared
    /// GPU's work falls inside them), which is not the scene getting lighter.
    const LIGHTER: f64 = 0.75;

    pub(crate) fn new(options: FramePacerOptions) -> Self {
        Self {
            options,
            intervals: VecDeque::new(),
            first_refresh: None,
            last_refresh: None,
            refresh_ms: f64::INFINITY,
            last_short: 0.0,
            recent: VecDeque::new(),
            gpu: VecDeque::new(),
            spills: VecDeque::new(),
            divisor: 1,
            changed_at: 0.0,
            trying: None,
            backoff_ms: vec![options.settle_ms; options.max_divisor.max(1) as usize + 1],
            failed_base: vec![f64::INFINITY; options.max_divisor.max(1) as usize + 1],
            home: None,
            last_render: None,
            report_start: None,
            report_refreshes: [0; 5],
            report_frames: [0; 5],
            report_missed: 0,
            report_stalls: 0,
            report_changes: Vec::new(),
        }
    }

    /// An interval's bucket in refreshes: 1, 2, 3, 4, 5 or more.
    fn bucket(&self, interval: f64) -> usize {
        ((interval / self.refresh_ms()).round().max(1.0) as usize).min(5) - 1
    }

    /// What it saw and did since the last report.
    pub(crate) fn take_report(&mut self) -> PacerReport {
        let now = self.last_refresh.unwrap_or(0.0);
        let report = PacerReport {
            seconds: self.report_start.map_or(0.0, |start| (now - start) / 1000.0),
            refresh_ms: self.refresh_ms(),
            divisor: self.divisor,
            holding: self.held(),
            refresh_intervals: self.report_refreshes,
            frame_intervals: self.report_frames,
            missed: self.report_missed,
            stalls: self.report_stalls,
            gpu_p50: self.gpu_p50().unwrap_or(f64::NAN),
            gpu_p90: self.gpu_p90().unwrap_or(f64::NAN),
            changes: std::mem::take(&mut self.report_changes),
        };
        self.report_start = self.last_refresh;
        self.report_refreshes = [0; 5];
        self.report_frames = [0; 5];
        self.report_missed = 0;
        self.report_stalls = 0;
        report
    }

    /// The fewest refreshes per frame `max_fps` allows.
    fn min_divisor(&self) -> u32 {
        if !self.refresh_ms.is_finite() {
            return 1;
        }
        let floor = self.options.max_fps.map_or(1, |fps| ((1000.0 / fps / self.refresh_ms) - 0.25).ceil().max(1.0) as u32);
        floor.min(self.options.max_divisor.max(1))
    }

    /// A refresh at `now_ms`: whether to render on it.
    pub(crate) fn on_refresh(&mut self, now_ms: f64) -> bool {
        let first = *self.first_refresh.get_or_insert(now_ms);
        self.report_start.get_or_insert(now_ms);
        let mut late = false;
        if let Some(last) = self.last_refresh {
            let interval = now_ms - last;
            // (a hidden tab or a stall is not a refresh interval)
            if interval >= 250.0 {
                self.report_stalls += 1;
            }
            if interval > 0.0 && interval < 250.0 {
                self.intervals.push_back(interval);
                if self.intervals.len() > 240 {
                    self.intervals.pop_front();
                }
                // the refresh interval: the short end of the intervals (a missed refresh only makes
                // one longer). It is kept through stretches where every refresh is late (they show
                // no refresh interval), unless none as short is seen for `RELEARN_MS` (a slower
                // display)
                if interval <= self.refresh_ms * 1.2 {
                    self.last_short = now_ms;
                }
                let shortest = percentile(&self.intervals, 0.1);
                if shortest < self.refresh_ms || now_ms - self.last_short > Self::RELEARN_MS {
                    self.refresh_ms = shortest;
                    self.last_short = now_ms;
                }
                late = interval > self.refresh_ms * self.options.late_factor;
                let bucket = self.bucket(interval);
                self.report_refreshes[bucket] += 1;
                self.report_missed += late as u32;
            }
        }
        self.last_refresh = Some(now_ms);
        if now_ms - first < self.options.warmup_ms {
            self.changed_at = now_ms;
            if self.divisor < self.min_divisor() {
                self.divisor = self.min_divisor();
            }
        } else {
            self.recent.push_back((now_ms, late));
            while self.recent.front().is_some_and(|&(t, _)| now_ms - t > 2.0 * self.options.window_ms) {
                self.recent.pop_front();
            }
            self.decide(now_ms);
        }
        // render once `divisor` refresh intervals have passed since the last frame (half a
        // refresh early, for the timestamps' jitter)
        let due = self.last_render.is_none_or(|last| now_ms - last >= (self.divisor as f64 - 0.5) * self.refresh_ms());
        if due {
            if let Some(last) = self.last_render {
                let bucket = self.bucket(now_ms - last);
                self.report_frames[bucket] += 1;
            }
            self.last_render = Some(now_ms);
        }
        due
    }

    /// A rendered frame's GPU time, ms.
    pub(crate) fn on_gpu_time(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.gpu.push_back(ms);
            if self.gpu.len() > Self::GPU_FRAMES {
                self.gpu.pop_front();
            }
        }
    }

    fn set_divisor(&mut self, divisor: u32, now_ms: f64, reason: &'static str) {
        // (kept for a report nobody takes: the latest)
        if self.report_changes.len() >= Self::MAX_CHANGES {
            self.report_changes.remove(0);
        }
        self.report_changes.push(PacerChange { at_ms: now_ms, divisor, reason });
        self.divisor = divisor;
        self.changed_at = now_ms;
        self.recent.clear();
        self.gpu.clear();
        self.spills.clear();
    }

    fn decide(&mut self, now_ms: f64) {
        let o = self.options;
        let floor = self.min_divisor();
        if self.divisor < floor {
            self.set_divisor(floor, now_ms, "floor");
            return;
        }
        let k = self.divisor;
        let since = now_ms - self.changed_at;
        // the refreshes missed over the last window, and their share there and over the one before
        let share = |from: f64, to: f64| {
            let (missed, count) = self.recent.iter().filter(|&&(t, _)| t > from && t <= to).fold((0, 0), |(m, n), &(_, late)| (m + late as u32, n + 1));
            (missed as f64, missed as f64 / count.max(1) as f64)
        };
        let (missed, last) = share(now_ms - o.window_ms, now_ms);
        let (_, before) = share(now_ms - 2.0 * o.window_ms, now_ms - o.window_ms);
        // missing refreshes over two windows running, or many over one, while the median frame
        // does not fit: render on fewer; a try fails on one window's misses
        let misses = !self.median_fits(k)
            && (since >= o.window_ms && (last > o.late_share_at_once || (self.trying.is_some() && last > o.late_share))
                || since >= 2.0 * o.window_ms && last > o.late_share && before > o.late_share);
        // frames spilling past the cadence (one in ten or more) on nearly every refresh over two
        // windows: the screen shows uneven intervals whether or not refreshes are missed (the
        // browser keeps a frame or two in flight). Frames that one step slower would not hold
        // either (a cut, a rebuild) are hitches, left out; and only while the slower cadence keeps
        // `spill_floor_fps`
        let over = self.spill_p90(k).is_some_and(|gpu| gpu > k as f64 * self.refresh_ms());
        self.spills.push_back((now_ms, over));
        while self.spills.front().is_some_and(|&(t, _)| now_ms - t > 2.0 * o.window_ms) {
            self.spills.pop_front();
        }
        let over_share = self.spills.iter().filter(|&&(_, over)| over).count() as f64 / self.spills.len().max(1) as f64;
        if let Some(t) = self.trying.as_mut() {
            t.over += over as u32;
            t.refreshes += 1;
        }
        let steps_down = 1000.0 / ((k + 1) as f64 * self.refresh_ms()) >= o.spill_floor_fps - 0.5;
        let spilling = steps_down && since >= 2.0 * o.window_ms && over_share >= Self::SPILLING;
        if misses || spilling {
            let reason = if self.trying.is_some() { "try failed" } else if spilling { "spilling" } else { "misses" };
            match self.trying.take() {
                Some(t) => self.failed(k, t),
                // a held cadence: come back to it as soon as frames fit it
                None => self.home = Some(k),
            }
            if k < o.max_divisor.max(floor) {
                self.set_divisor(k + 1, now_ms, reason);
            }
            return;
        }
        // a try that held through its probation, frames spilling on few of its refreshes: keep
        // it; otherwise it failed (near the threshold, the percentile dips under it now and then)
        if let Some(t) = self.trying.filter(|_| since >= o.probation_ms) {
            if (t.over as f64) < Self::KEEP * t.refreshes.max(1) as f64 {
                self.trying = None;
                self.backoff_ms[k as usize] = o.settle_ms;
                self.failed_base[k as usize] = f64::INFINITY;
                self.home = None;
            } else if k < o.max_divisor.max(floor) {
                self.trying = None;
                self.failed(k, t);
                self.set_divisor(k + 1, now_ms, "try failed");
                return;
            }
        }
        if k <= floor || missed > 0.0 {
            return;
        }
        // the latest frames well within a faster cadence: try it at once (and on, a step at a
        // time, while they keep fitting): back home on that alone; elsewhere when they are also
        // well below what this cadence cost when a try at the faster one last failed
        let faster = k - 1;
        let lighter = self.gpu_recent().is_some_and(|(p50, p90)| {
            p90 < faster as f64 * self.refresh_ms() * o.fit_share && (self.home == Some(faster) || p50 < self.failed_base[faster as usize] * Self::LIGHTER)
        });
        // or, held without a miss for long enough: a blind try (the clocks may be what's slow)
        let blind = self.trying.is_none() && since >= self.backoff_ms[faster as usize];
        if lighter || blind {
            let base = self.gpu_p50().unwrap_or(f64::INFINITY);
            // (a lighter step during a probation keeps the lower base)
            let base = self.trying.map_or(base, |t| t.base.min(base));
            self.set_divisor(faster, now_ms, if lighter { "lighter" } else { "try" });
            self.trying = Some(Try { base, blind: !lighter, over: 0, refreshes: 0 });
        }
    }

    /// A try at divisor `k` failed: wait longer before the next, remember what the slower cadence
    /// cost just before it, and it is no longer a home to come back to on fit alone.
    fn failed(&mut self, k: u32, t: Try) {
        self.backoff_ms[k as usize] = (self.backoff_ms[k as usize] * 2.0).min(self.options.max_backoff_ms);
        self.failed_base[k as usize] = t.base;
        if self.home == Some(k) {
            self.home = None;
        }
    }

    /// The divisor it holds: a blind try still on probation doesn't count.
    pub(crate) fn held(&self) -> u32 {
        match self.trying {
            Some(t) if t.blind => self.divisor + 1,
            _ => self.divisor,
        }
    }

    fn gpu_p50(&self) -> Option<f64> {
        (self.gpu.len() >= 10).then(|| percentile(&self.gpu, 0.5))
    }

    /// The median and 90th percentile of the latest `RECENT_FRAMES` frames' GPU time.
    fn gpu_recent(&self) -> Option<(f64, f64)> {
        (self.gpu.len() >= Self::RECENT_FRAMES).then(|| {
            let recent: VecDeque<f64> = self.gpu.iter().rev().take(Self::RECENT_FRAMES).copied().collect();
            (percentile(&recent, 0.5), percentile(&recent, 0.9))
        })
    }

    /// Recent frames' 90th percentile GPU time at divisor `k`, leaving out the frames that one
    /// step slower would not hold either (hitches: a cut, a rebuild).
    fn spill_p90(&self, k: u32) -> Option<f64> {
        let limit = (k + 1) as f64 * self.refresh_ms();
        let held: VecDeque<f64> = self.gpu.iter().copied().filter(|&ms| ms <= limit).collect();
        (held.len() >= 10).then(|| percentile(&held, 0.9))
    }

    fn gpu_p90(&self) -> Option<f64> {
        (self.gpu.len() >= 10).then(|| percentile(&self.gpu, 0.9))
    }

    /// Whether recent frames' median GPU time fits `divisor` refreshes.
    fn median_fits(&self, divisor: u32) -> bool {
        self.gpu_p50().is_some_and(|gpu| gpu < divisor as f64 * self.refresh_ms())
    }

    pub(crate) fn divisor(&self) -> u32 {
        self.divisor
    }

    pub(crate) fn refresh_ms(&self) -> f64 {
        if self.refresh_ms.is_finite() { self.refresh_ms } else { 1000.0 / 60.0 }
    }
}

fn percentile(values: &VecDeque<f64>, p: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    sorted[((sorted.len() as f64 * p) as usize).min(sorted.len() - 1)]
}

/// Renders on a steady share of the display's refreshes (see the module).
pub struct FramePacer {
    cadence: Cadence,
    timer: FrameTimer,
}

impl FramePacer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, options: FramePacerOptions) -> Self {
        Self { cadence: Cadence::new(options), timer: FrameTimer::new(device, queue) }
    }

    /// At each display refresh (`requestAnimationFrame`'s timestamp, ms): whether to render on it.
    pub fn on_refresh(&mut self, now_ms: f64) -> bool {
        for ms in self.timer.take() {
            self.cadence.on_gpu_time(ms);
        }
        self.cadence.on_refresh(now_ms)
    }

    /// Before a rendered frame's first submit.
    pub fn begin_frame(&mut self) {
        self.timer.begin();
    }

    /// After a rendered frame's last submit.
    pub fn end_frame(&mut self) {
        self.timer.end();
    }

    /// Refreshes per rendered frame.
    pub fn divisor(&self) -> u32 {
        self.cadence.divisor()
    }

    /// The display's refresh interval, as measured, ms.
    pub fn refresh_ms(&self) -> f64 {
        self.cadence.refresh_ms()
    }

    /// The last measured frame's GPU time, ms (NaN until one arrives).
    pub fn gpu_ms(&self) -> f64 {
        self.timer.last_ms()
    }

    /// What it saw and did since the last report: log it every few seconds on a display where
    /// the pacing misbehaves (`to_string` prints it on a line).
    pub fn take_report(&mut self) -> PacerReport {
        self.cadence.take_report()
    }
}

/// A frame's GPU time: a timestamp written by a one-thread compute pass submitted before the
/// frame's work and another after it (on Metal an empty pass resolves its timestamps to zero,
/// hence the dispatch). Without `TIMESTAMP_QUERY`, the time from the first submit until a buffer
/// copied after the last becomes mappable: an upper bound, queueing included. A ring of readbacks
/// measures every frame although results arrive a few frames late; the ring shares one query set
/// (on Metal each set is a counter sample buffer, of which a browser tab gets few). A frame's
/// stamps are resolved at the next frame's start (or `flush`): wgpu's native Metal backend can
/// resolve them before they are written when asked right after; a frame whose stamps still read
/// back out of order is left unmeasured.
pub struct FrameTimer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    queries: Option<(wgpu::ComputePipeline, f64)>,
    /// `STAMPS` timestamps per slot, and where each slot's are resolved (`RESOLVE_STRIDE` apart)
    set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    slots: Vec<TimerSlot>,
    armed: Option<usize>,
    started_ms: f64,
    /// Frames ended whose stamps are not resolved yet: their slot and CPU start time
    unresolved: Vec<(usize, f64)>,
    results: Arc<Mutex<Vec<f64>>>,
    last_ms: Arc<Mutex<f64>>,
}

struct TimerSlot {
    readback: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

impl FrameTimer {
    const SLOTS: usize = 8;
    const STAMPS: u32 = 4;
    /// Query resolves land at multiples of 256 bytes.
    const RESOLVE_STRIDE: u64 = 256;

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Self {
        let timestamps = device.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let queries = timestamps.then(|| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("FrameTimer"), source: wgpu::ShaderSource::Wgsl("@compute @workgroup_size(1) fn main() {}".into()) });
            let noop = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FrameTimer"),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });
            (noop, queue.get_timestamp_period() as f64)
        });
        let buffer = |size, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some("FrameTimer"), size, usage, mapped_at_creation: false });
        let set = timestamps.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("FrameTimer"), ty: wgpu::QueryType::Timestamp, count: Self::SLOTS as u32 * Self::STAMPS }));
        let resolve = buffer(Self::SLOTS as u64 * Self::RESOLVE_STRIDE, wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC);
        let slots = (0..Self::SLOTS)
            .map(|_| TimerSlot {
                readback: buffer(8 * Self::STAMPS as u64, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
                busy: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            queries,
            set,
            resolve,
            slots,
            armed: None,
            started_ms: 0.0,
            unresolved: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
            last_ms: Arc::new(Mutex::new(f64::NAN)),
        }
    }

    fn stamp(&self, encoder: &mut wgpu::CommandEncoder, slot: usize, end: bool) {
        let (Some((noop, _)), Some(set)) = (&self.queries, &self.set) else { return };
        let first = slot as u32 * Self::STAMPS + if end { 2 } else { 0 };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FrameTimer"),
            // both stamps of both passes (the end of a pass is not always written on every GPU)
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: set,
                beginning_of_pass_write_index: Some(first),
                end_of_pass_write_index: Some(first + 1),
            }),
        });
        pass.set_pipeline(noop);
        pass.dispatch_workgroups(1, 1, 1);
    }

    /// Before the frame's first submit (a frame starts unmeasured when every readback is busy).
    pub fn begin(&mut self) {
        self.flush();
        self.armed = self.slots.iter().position(|s| !s.busy.load(Ordering::Acquire));
        let Some(slot) = self.armed else { return };
        self.started_ms = now_ms();
        if self.queries.is_some() {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FrameTimer/Start") });
            self.stamp(&mut encoder, slot, false);
            self.queue.submit(Some(encoder.finish()));
        }
    }

    /// After the frame's last submit.
    pub fn end(&mut self) {
        let Some(k) = self.armed.take() else { return };
        self.slots[k].busy.store(true, Ordering::Release);
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FrameTimer/End") });
        self.stamp(&mut encoder, k, true);
        self.queue.submit(Some(encoder.finish()));
        self.unresolved.push((k, self.started_ms));
        // without timestamps the readback is the measurement: at once
        if self.queries.is_none() {
            self.flush();
        }
    }

    /// Resolve the ended frames' stamps and read them back (`begin` does, for the frames before).
    pub fn flush(&mut self) {
        if self.unresolved.is_empty() {
            return;
        }
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("FrameTimer/Resolve") });
        for &(k, _) in &self.unresolved {
            let offset = k as u64 * Self::RESOLVE_STRIDE;
            if let Some(set) = &self.set {
                let first = k as u32 * Self::STAMPS;
                encoder.resolve_query_set(set, first..first + Self::STAMPS, &self.resolve, offset);
            }
            encoder.copy_buffer_to_buffer(&self.resolve, offset, &self.slots[k].readback, 0, 8 * Self::STAMPS as u64);
        }
        self.queue.submit(Some(encoder.finish()));
        let period = self.queries.as_ref().map(|(_, period)| *period);
        for (k, started) in std::mem::take(&mut self.unresolved) {
            let slot = &self.slots[k];
            let (busy, results, last, readback) = (slot.busy.clone(), self.results.clone(), self.last_ms.clone(), slot.readback.clone());
            slot.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    let ms = match period {
                        Some(period) => {
                            let t: [u64; 4] = bytemuck::pod_read_unaligned(&readback.slice(..).get_mapped_range()[..32]);
                            // from the start pass's first stamp to the end pass's last written one
                            let start = t[0];
                            let end = if t[3] > t[2] { t[3] } else { t[2] };
                            (start > 0 && end > start).then(|| (end - start) as f64 * period / 1.0e6)
                        }
                        None => Some(now_ms() - started),
                    };
                    readback.unmap();
                    if let Some(ms) = ms {
                        results.lock().unwrap().push(ms);
                        *last.lock().unwrap() = ms;
                    }
                }
                busy.store(false, Ordering::Release);
            });
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.device.poll(wgpu::Maintain::Poll);
    }

    /// The frames measured since the last call, ms.
    pub fn take(&self) -> Vec<f64> {
        std::mem::take(&mut *self.results.lock().unwrap())
    }

    /// Whether the GPU times come from timestamp queries (else they are each frame's CPU start to
    /// its readback, an upper bound).
    pub fn has_timestamps(&self) -> bool {
        self.queries.is_some()
    }

    /// The last frame measured, ms (NaN until one arrives).
    pub fn last_ms(&self) -> f64 {
        *self.last_ms.lock().unwrap()
    }
}

fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        web_sys::window().and_then(|w| w.performance()).map_or(0.0, |p| p.now())
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::sync::OnceLock;
        static START: OnceLock<std::time::Instant> = OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A browser and a GPU: requestAnimationFrame at each refresh (every `refresh` ms), unless two
    /// frames are still on the GPU, when it comes at the first refresh after the older is done;
    /// each rendered frame costs `gpu(t)` ms of GPU time, one after another. Runs the cadence for
    /// `seconds` from `start`; the divisor at the end and the rendered frames' times.
    fn simulate(cadence: &mut Cadence, start: f64, refresh: f64, seconds: f64, gpu: impl Fn(f64) -> f64) -> (u32, Vec<f64>) {
        let run = run(cadence, start, refresh, seconds, gpu, |_, _| {});
        (cadence.held(), run.rendered)
    }

    /// `simulate`, calling `seen` with each refresh's time and the divisor after it.
    fn simulate_seeing(cadence: &mut Cadence, start: f64, refresh: f64, seconds: f64, gpu: impl Fn(f64) -> f64, seen: impl FnMut(f64, u32)) -> (u32, Vec<f64>) {
        let run = run(cadence, start, refresh, seconds, gpu, seen);
        (cadence.held(), run.rendered)
    }

    /// A simulation's frames: when they were rendered, and when the screen showed
    /// them (each at the first refresh after its GPU work is done, one a refresh, in order).
    struct Run {
        rendered: Vec<f64>,
        shown: Vec<f64>,
    }

    fn run(cadence: &mut Cadence, start: f64, refresh: f64, seconds: f64, gpu: impl Fn(f64) -> f64, mut seen: impl FnMut(f64, u32)) -> Run {
        let mut in_flight: VecDeque<f64> = VecDeque::new(); // when each frame on the GPU is done
        let mut gpu_free = start;
        let (mut rendered, mut shown) = (Vec::new(), Vec::<f64>::new());
        let mut t = start;
        while t < start + seconds * 1000.0 {
            while in_flight.front().is_some_and(|&done| done <= t) {
                in_flight.pop_front();
            }
            if in_flight.len() >= 2 {
                // the browser waits: the next refresh after the oldest frame is done
                let done = in_flight[0];
                t = start + ((done - start) / refresh).ceil() * refresh;
                continue;
            }
            if cadence.on_refresh(t) {
                let ms = gpu(t);
                rendered.push(t);
                gpu_free = gpu_free.max(t) + ms;
                in_flight.push_back(gpu_free);
                cadence.on_gpu_time(ms);
                let at = start + ((gpu_free - start) / refresh).ceil() * refresh;
                shown.push(shown.last().map_or(at, |&last| at.max(last + refresh)));
            }
            seen(t, cadence.divisor());
            t += refresh;
        }
        Run { rendered, shown }
    }

    /// The seconds (from `from` ms) whose shown frames came at more than one interval, but for
    /// those with a divisor change in them or just before (in `changes`).
    fn uneven_seconds(shown: &[f64], refresh: f64, from: f64, changes: &[(f64, u32)]) -> Vec<u32> {
        let mut per_second: std::collections::BTreeMap<u32, std::collections::BTreeSet<i64>> = Default::default();
        for w in shown.windows(2).filter(|w| w[0] >= from) {
            per_second.entry((w[1] / 1000.0) as u32).or_default().insert(((w[1] - w[0]) / refresh).round() as i64);
        }
        per_second
            .into_iter()
            .filter(|(s, intervals)| intervals.len() > 1 && !changes.iter().any(|&(t, _)| ((t / 1000.0) as u32).abs_diff(*s) <= 1))
            .map(|(s, _)| s)
            .collect()
    }

    /// A deterministic noise in [-1, 1) for frame `i`.
    fn noise(i: u64) -> f64 {
        let x = (i.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (i >> 7).wrapping_mul(0xBF58_476D_1CE4_E5B9)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (x >> 11) as f64 / (1u64 << 52) as f64 - 1.0
    }

    /// Frame costs of `base` ms, ±`spread` (a share), a new draw each frame.
    fn near_the_budget(base: f64, spread: f64) -> impl Fn(f64) -> f64 {
        let i = std::cell::Cell::new(0u64);
        move |_| {
            let k = i.get();
            i.set(k + 1);
            base * (1.0 + spread * noise(k))
        }
    }

    #[test]
    fn a_scene_near_the_budget_does_not_flip_between_cadences() {
        // 16.9 ms frames (±4%) at 120 Hz capped at 60: every third refresh (40 fps) keeps up and
        // every other doesn't quite. Tries at every other refresh are what the screen shows as
        // jumps between 60 and 40: they come rarely, not every second or two
        let refresh = 1000.0 / 120.0;
        for (base, spread) in [(16.9, 0.04), (17.5, 0.03), (18.0, 0.1)] {
            let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
            let mut changes = Vec::new();
            let mut last = cadence.divisor();
            let run = run(&mut cadence, 0.0, refresh, 120.0, near_the_budget(base, spread), |t, k| {
                if k != last {
                    changes.push((t, k));
                    last = k;
                }
            });
            let after_start: Vec<&(f64, u32)> = changes.iter().filter(|(t, _)| *t > 3000.0).collect();
            assert!(after_start.len() <= 8, "{base} ms ±{spread}: {} changes in 2 minutes: {changes:?}", changes.len());
            let tries: Vec<f64> = after_start.iter().filter(|(_, k)| *k == 2).map(|(t, _)| *t).collect();
            assert!(tries.windows(2).all(|w| w[1] - w[0] >= 8000.0), "{base} ms ±{spread}: tries {tries:?}");
            // frames that don't straddle two refreshes at every third are shown evenly every
            // second but around the changes
            if base == 17.5 {
                let uneven = uneven_seconds(&run.shown, refresh, 3000.0, &changes);
                assert!(uneven.is_empty(), "{base} ms ±{spread}: uneven seconds {uneven:?}, changes {changes:?}");
            }
        }
    }

    #[test]
    fn frames_that_keep_spilling_over_the_budget_slow_it_down() {
        // 15.5 ms frames ±12% at 120 Hz capped at 60: the median fits every other refresh, but
        // nearly one frame in five spills past it, and the screen shows 1-, 2- and 3-refresh
        // intervals every second. Every third refresh (40 fps) is steady: it goes there and stays
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        let changes = divisors(&mut cadence, 0.0, refresh, 30.0, near_the_budget(15.5, 0.12));
        assert!(changes.iter().any(|&(t, k)| k == 3 && t < 5000.0), "{changes:?}");
        assert_eq!(cadence.held(), 3, "{changes:?}");
        // a steady 15.5 ms (the same median) keeps every other refresh
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        assert!(divisors(&mut cadence, 0.0, refresh, 30.0, near_the_budget(15.5, 0.02)).iter().all(|&(_, k)| k == 2));
    }

    #[test]
    fn the_report_says_what_it_saw_and_why_it_changed() {
        // 15.5 ms frames ±12% at 120 Hz capped at 60: it slows to every third refresh, spilling
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        simulate(&mut cadence, 0.0, refresh, 6.0, near_the_budget(15.5, 0.12));
        let report = cadence.take_report();
        assert!((report.seconds - 6.0).abs() < 0.1, "{report}");
        assert_eq!((report.divisor, report.holding), (3, 3), "{report}");
        assert!(report.changes.iter().any(|c| c.divisor == 3 && c.reason == "spilling"), "{report}");
        // every refresh came on time; frames were 2 then 3 refreshes apart
        assert!(report.refresh_intervals[0] > 600 && report.refresh_intervals[1..].iter().all(|&n| n == 0), "{report}");
        assert!(report.frame_intervals[1] > 0 && report.frame_intervals[2] > 0, "{report}");
        assert!(report.gpu_p90 > 16.7 && report.gpu_p50 < report.gpu_p90, "{report}");
        let text = report.to_string();
        assert!(text.contains("spilling") && text.contains("→ 3"), "{text}");
        // a report covers what came after the last
        simulate(&mut cadence, 6000.0, refresh, 2.0, near_the_budget(15.5, 0.12));
        let next = cadence.take_report();
        assert!((next.seconds - 2.0).abs() < 0.1 && next.changes.is_empty(), "{next}");
    }

    fn steady(frames: &[f64], interval: f64) -> bool {
        frames.windows(2).all(|w| (w[1] - w[0] - interval).abs() < 1e-6)
    }

    #[test]
    fn it_settles_on_the_fewest_refreshes_that_keep_up() {
        let (r60, r120) = (1000.0 / 60.0, 1000.0 / 120.0);
        for (refresh, gpu, divisor) in [(r60, 12.0, 1), (r60, 20.0, 2), (r120, 12.0, 2), (r120, 15.0, 2), (r120, 20.0, 3), (r120, 6.0, 1), (r120, 30.0, 4)] {
            let mut cadence = Cadence::new(FramePacerOptions::default());
            // (light frames at first, as while loading: the refresh is seen)
            let (k, frames) = simulate(&mut cadence, 0.0, refresh, 45.0, |t| if t < 500.0 { 2.0 } else { gpu });
            assert_eq!(k, divisor, "{gpu} ms frames at {:.0} Hz", 1000.0 / refresh);
            assert!((cadence.refresh_ms() - refresh).abs() < 0.01);
            // and the last seconds are evenly spaced (between the ever rarer tries at a faster
            // cadence)
            let tail: Vec<f64> = frames.into_iter().filter(|&t| t > 42000.0).collect();
            assert!(steady(&tail, divisor as f64 * refresh), "{gpu} ms frames at {:.0} Hz: {tail:?}", 1000.0 / refresh);
        }
    }

    #[test]
    fn max_fps_caps_it() {
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        let (k, frames) = simulate(&mut cadence, 0.0, refresh, 10.0, |_| 5.0);
        assert_eq!(k, 2, "light frames at 120 Hz, capped at 60 fps");
        assert!(steady(&frames[1..], 2.0 * refresh));
        // at 60 Hz the same cap is every refresh
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        assert_eq!(simulate(&mut cadence, 0.0, 1000.0 / 60.0, 10.0, |_| 5.0).0, 1);
    }

    #[test]
    fn it_follows_the_cost_up_at_once_and_down_after_settling() {
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        // 12 ms frames: every other refresh
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 20.0, |_| 12.0).0, 2);
        // heavier: every third within about a second
        assert_eq!(simulate(&mut cadence, 20000.0, refresh, 2.5, |_| 20.0).0, 3);
        // lighter again: back to every other, after the settling (and any failed tries' waits)
        assert_eq!(simulate(&mut cadence, 22500.0, refresh, 40.0, |_| 12.0).0, 2);
        // a rare slow frame does not change it
        let (k, _) = simulate(&mut cadence, 62500.0, refresh, 10.0, |t| if ((t / refresh) as u64).is_multiple_of(300) { 30.0 } else { 12.0 });
        assert_eq!(k, 2);
    }

    #[test]
    fn a_failed_try_waits_longer_before_the_next() {
        // 15 ms frames at 120 Hz fit every other refresh but not every one: the tries at every
        // refresh fail, and come further and further apart
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        let changes = divisors(&mut cadence, 0.0, refresh, 90.0, |_| 15.0);
        let tries: Vec<f64> = changes.iter().filter(|&&(_, k)| k == 1).map(|&(t, _)| t).collect();
        assert!(tries.len() >= 3, "{changes:?}");
        let gaps: Vec<f64> = tries.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.windows(2).all(|g| g[1] >= g[0] - 150.0), "the gaps grow: {gaps:?}");
        assert!(gaps.last().unwrap() > &10000.0, "{gaps:?}");
    }

    #[test]
    fn a_lighter_scene_is_tried_at_once_after_failed_tries() {
        // a heavy stretch (15 ms frames at 120 Hz) runs up the wait between tries; a light one
        // after it (4 ms) is back at every refresh within a couple of seconds, not the wait
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 40.0, |_| 15.0).0, 2);
        assert_eq!(simulate(&mut cadence, 40000.0, refresh, 2.5, |_| 4.0).0, 1);
    }

    #[test]
    fn missed_refreshes_while_warming_up_are_not_counted() {
        let refresh = 1000.0 / 60.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        // slow first frames (the first half second), then light ones
        let (k, _) = simulate(&mut cadence, 0.0, refresh, 10.0, |t| if t < 500.0 { 100.0 } else { 8.0 });
        assert_eq!(k, 1);
    }

    /// The divisor changes of a `simulate`: when, and to what.
    fn divisors(cadence: &mut Cadence, start: f64, refresh: f64, seconds: f64, gpu: impl Fn(f64) -> f64) -> Vec<(f64, u32)> {
        let mut changes = Vec::new();
        let mut last = cadence.divisor();
        simulate_seeing(cadence, start, refresh, seconds, gpu, |t, k| {
            if k != last {
                changes.push((t, k));
                last = k;
            }
        });
        changes
    }

    #[test]
    fn a_burst_of_slow_frames_is_not_a_slower_cadence() {
        // 12 ms frames every other refresh at 120 Hz, then a cut: 40 ms frames for a fifth of a
        // second, a few hitches
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 10.0, |_| 12.0).0, 2);
        let cut = 10000.0;
        let changes = divisors(&mut cadence, cut, refresh, 10.0, |t| if (cut..cut + 200.0).contains(&t) { 40.0 } else { 12.0 });
        assert!(changes.is_empty(), "{changes:?}");
    }

    #[test]
    fn misses_over_two_windows_slow_it_and_many_over_one_at_once() {
        // at 60 Hz, 18 ms frames miss about one refresh in twelve: after two windows of it,
        // every other refresh
        let refresh = 1000.0 / 60.0;
        let onset = 5000.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 5.0, |_| 12.0).0, 1);
        let changes = divisors(&mut cadence, onset, refresh, 4.0, |_| 18.0);
        assert!(changes.first().is_some_and(|&(t, k)| k == 2 && (1500.0..2600.0).contains(&(t - onset))), "{changes:?}");
        // 24 ms frames miss nearly half: within a window
        let mut cadence = Cadence::new(FramePacerOptions::default());
        simulate(&mut cadence, 0.0, refresh, 5.0, |_| 12.0);
        let changes = divisors(&mut cadence, onset, refresh, 2.0, |_| 24.0);
        assert!(changes.first().is_some_and(|&(t, k)| k == 2 && t - onset < 1200.0), "{changes:?}");
    }

    #[test]
    fn misses_while_the_median_frame_fits_are_hitches() {
        // at 60 Hz capped at 60: 12 ms frames with every third at 26 ms miss refreshes all along,
        // but the typical frame fits: it stays at 60
        let refresh = 1000.0 / 60.0;
        let gpu = |t: f64| if ((t / refresh) as u64).is_multiple_of(3) { 26.0 } else { 12.0 };
        let run = |gpu: &dyn Fn(f64) -> f64| {
            let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
            simulate(&mut cadence, 0.0, refresh, 2.0, |_| 12.0);
            divisors(&mut cadence, 2000.0, refresh, 20.0, gpu)
        };
        let changes = run(&gpu);
        assert!(changes.is_empty(), "{changes:?}");
        // with the typical frame over a refresh (18 ms), the misses slow it down
        let heavy = run(&|_| 18.0);
        assert!(heavy.first().is_some_and(|&(_, k)| k == 2), "{heavy:?}");
    }

    #[test]
    fn on_a_60_hz_display_spills_stay_hitches() {
        // one step down from 60 is 30 there (under `spill_floor_fps`): 15.5 ms ±10% frames (one
        // in eight over a refresh) stay at every refresh
        let refresh = 1000.0 / 60.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        let changes = divisors(&mut cadence, 0.0, refresh, 60.0, near_the_budget(15.5, 0.1));
        assert!(changes.iter().all(|&(_, k)| k == 1), "{changes:?}");
    }

    /// Milliseconds spent at a divisor above `k` over `changes`, until `end`.
    fn time_above(changes: &[(f64, u32)], start: (f64, u32), k: u32, end: f64) -> f64 {
        let mut time = 0.0;
        let (mut at, mut current) = start;
        for &(t, d) in changes.iter().chain(std::iter::once(&(end, 0))) {
            if current > k {
                time += t - at;
            }
            (at, current) = (t, d);
        }
        time
    }

    #[test]
    fn rebuilds_of_half_a_second_or_a_second_are_hitches() {
        // 120 Hz capped at 60, 12 ms frames (every other refresh), or 19 ms (every third): a
        // stretch of 40-45 ms frames (a rebuild) doesn't fit one step slower either: a hitch
        let refresh = 1000.0 / 120.0;
        for (base, k, slow) in [(12.0, 2, 40.0), (19.0, 3, 45.0)] {
            for length in [500.0, 1000.0] {
                let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
                assert_eq!(simulate(&mut cadence, 0.0, refresh, 10.0, |_| base).0, k);
                let at = 10000.0;
                let changes = divisors(&mut cadence, at, refresh, 20.0, move |t| if (at..at + length).contains(&t) { slow } else { base });
                let off = time_above(&changes, (at, k), k, at + 20000.0);
                assert!(off <= 1000.0, "{base} ms at every {k}, {length} ms of {slow} ms: {off} ms slower: {changes:?}");
            }
        }
        // two of them five seconds apart
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        simulate(&mut cadence, 0.0, refresh, 10.0, |_| 12.0);
        let bursts = |t: f64| (10000.0..10500.0).contains(&t) || (15000.0..15500.0).contains(&t);
        let changes = divisors(&mut cadence, 10000.0, refresh, 20.0, move |t| if bursts(t) { 40.0 } else { 12.0 });
        assert!(time_above(&changes, (10000.0, 2), 2, 30000.0) <= 1000.0, "{changes:?}");
    }

    #[test]
    fn a_cut_after_failed_tries_does_not_pin_a_slower_cadence() {
        // 120 Hz capped at 60: 19 ms frames settle at every third refresh after failed tries at
        // every other (long waits between them); a heavy half second (a cut) then must not hold
        // every fourth for longer than the cut
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 150.0, |_| 19.0).0, 3);
        let at = 150000.0;
        let changes = divisors(&mut cadence, at, refresh, 30.0, move |t| if (at..at + 500.0).contains(&t) { 28.0 } else { 19.0 });
        assert!(time_above(&changes, (at, 3), 3, at + 30000.0) <= 2500.0, "{changes:?}");
        // on a 60 Hz display, steady 30 fps (22 ms frames) and a burst of 60 ms frames
        let refresh = 1000.0 / 60.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 150.0, |_| 22.0).0, 2);
        let changes = divisors(&mut cadence, at, refresh, 30.0, move |t| if (at..at + 500.0).contains(&t) { 60.0 } else { 22.0 });
        assert!(time_above(&changes, (at, 2), 2, at + 30000.0) <= 2500.0, "{changes:?}");
    }

    #[test]
    fn it_comes_back_several_steps_at_once() {
        // 120 Hz uncapped: 5 ms frames (every refresh), 3 s of 30 ms (every fourth), then 5 ms:
        // back to every refresh within a few seconds, not a probation per step
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions::default());
        assert_eq!(simulate(&mut cadence, 0.0, refresh, 10.0, |_| 5.0).0, 1);
        assert_eq!(simulate(&mut cadence, 10000.0, refresh, 3.0, |_| 30.0).0, 4);
        let changes = divisors(&mut cadence, 13000.0, refresh, 10.0, |_| 5.0);
        assert!(changes.iter().any(|&(t, k)| k == 1 && t < 16000.0), "{changes:?}");
        assert_eq!(cadence.divisor(), 1, "{changes:?}");
    }

    #[test]
    fn near_the_spill_threshold_it_settles() {
        // 120 Hz capped at 60: about one frame in ten over 16.7 ms (15.2 ms ±12%): whatever it
        // settles on, it doesn't keep switching
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        let changes = divisors(&mut cadence, 0.0, refresh, 300.0, near_the_budget(15.2, 0.12));
        // tries at every other refresh come further and further apart, whether they fail on
        // spills or at the end of their probation
        let tries: Vec<f64> = changes.iter().filter(|&&(t, k)| k == 2 && t > 3000.0).map(|&(t, _)| t).collect();
        let gaps: Vec<f64> = tries.windows(2).map(|w| w[1] - w[0]).collect();
        assert!(gaps.iter().all(|&g| g >= 8000.0) && gaps.windows(2).all(|g| g[1] >= g[0]), "tries {tries:?}: {changes:?}");
        assert!(changes.iter().filter(|(t, _)| *t > 180000.0).count() <= 2, "{changes:?}");
    }

    #[test]
    fn a_frame_time_that_depends_on_the_cadence_does_not_make_it_thrash() {
        // frames that take 19 ms at every refresh but 14.5 at every other (a shared GPU, clocks):
        // the tries at every refresh fail, and come further apart, not every second or two
        let refresh = 1000.0 / 60.0;
        let last = std::cell::Cell::new((0.0f64, 0.0f64));
        let gpu = |t: f64| {
            let (prev, ms) = last.get();
            if t == prev {
                return ms;
            }
            let ms = if t - prev > 1.5 * refresh { 14.5 } else { 19.0 };
            last.set((t, ms));
            ms
        };
        let mut cadence = Cadence::new(FramePacerOptions::default());
        simulate(&mut cadence, 0.0, refresh, 2.0, |_| 12.0);
        let changes = divisors(&mut cadence, 2000.0, refresh, 30.0, gpu);
        let tries = changes.iter().filter(|&&(_, k)| k == 1).count();
        assert!(tries <= 5, "{tries} tries in 30 s: {changes:?}");
    }

    #[test]
    fn it_comes_back_within_about_a_second() {
        let refresh = 1000.0 / 60.0;
        for (lighter, within) in [(12.0, 800.0), (14.5, 800.0)] {
            let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
            simulate(&mut cadence, 0.0, refresh, 2.0, |_| 12.0);
            // a heavy stretch: every other refresh
            assert_eq!(simulate(&mut cadence, 2000.0, refresh, 3.0, |_| 24.0).0, 2);
            // lighter again: a median that fits a refresh is tried at once
            let changes = divisors(&mut cadence, 5000.0, refresh, 5.0, move |_| lighter);
            assert!(changes.first().is_some_and(|&(t, k)| k == 1 && t - 5000.0 < within), "{lighter} ms frames: {changes:?}");
            assert_eq!(changes.len(), 1, "{lighter} ms frames: {changes:?}");
        }
    }

    #[test]
    fn a_late_refresh_does_not_delay_the_next_frame() {
        // every other refresh at 120 Hz; the browser skips a refresh now and then (light frames:
        // not the GPU), and the frame due on it is rendered on the next one, not a refresh later
        let refresh = 1000.0 / 120.0;
        let mut cadence = Cadence::new(FramePacerOptions { max_fps: Some(60.0), ..Default::default() });
        let mut last = None;
        for i in 0..2400u32 {
            if i % 37 == 36 {
                continue;
            }
            let t = i as f64 * refresh;
            let render = cadence.on_refresh(t);
            cadence.on_gpu_time(4.0);
            if t > 2000.0 {
                // the first refresh two intervals or more after the last frame renders
                let due = last.is_some_and(|l: f64| t - l > 1.5 * refresh);
                assert_eq!(render, due, "refresh {i}: {:.1} refreshes after the last frame", (t - last.unwrap_or(t)) / refresh);
            }
            if render {
                last = Some(t);
            }
        }
        assert_eq!(cadence.divisor(), 2);
    }

    /// On a real GPU, frames of work are measured (a ring of readbacks), with other timers alive;
    /// on wgpu's native Metal backend some stamps read back stale, and those frames go unmeasured.
    #[test]
    fn the_frame_timer_measures_each_frame() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() }, None)).unwrap();
        // many timers alive on one device, as a browser tab's reloads leave them for its garbage
        // collector: each holds one query set (on Metal a counter sample buffer, of which a
        // device has few)
        let _others: Vec<FrameTimer> = (0..15).map(|_| FrameTimer::new(&device, &queue)).collect();
        let mut timer = FrameTimer::new(&device, &queue);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(
                "@group(0) @binding(0) var<storage, read_write> b : array<f32>;
                 @compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id : vec3u) {
                     var x = b[id.x];
                     for (var i = 0; i < 2000; i++) { x = sin(x) * 1.0001 + 0.5; }
                     b[id.x] = x;
                 }"
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: None, layout: None, module: &module, entry_point: Some("main"), compilation_options: Default::default(), cache: None });
        let buffer = device.create_buffer(&wgpu::BufferDescriptor { label: None, size: 4 * 65536, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout: &pipeline.get_bind_group_layout(0), entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }] });
        let mut measured = Vec::new();
        for _ in 0..5 {
            timer.begin();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1024, 1, 1);
            }
            queue.submit(Some(encoder.finish()));
            timer.end();
            device.poll(wgpu::Maintain::Wait);
            measured.extend(timer.take());
        }
        timer.flush();
        device.poll(wgpu::Maintain::Wait);
        measured.extend(timer.take());
        assert!(measured.len() >= 3, "most frames measured: {measured:?}");
        assert!(measured.iter().all(|&ms| ms > 0.0 && ms < 1000.0), "{measured:?}");
        assert!(timer.last_ms() > 0.0);
    }
}
