//! Frame pacing: rendering on a steady share of the display's refreshes.
//!
//! A browser calls `requestAnimationFrame` once per display refresh (every 16.7 ms at 60 Hz, 8.3 ms
//! at 120 Hz). A frame whose GPU work takes a little longer than a refresh is shown a refresh late,
//! the next may not be, and the cadence alternates between one refresh and two: an uneven picture,
//! though the average frame rate looks fine. `FramePacer` renders on every k-th refresh instead
//! (every other one on a 120 Hz display is a steady 60 fps), k the fewest that keep up:
//! - it renders on fewer (k + 1) as soon as refreshes are being missed: the browser calls a
//!   refresh late while the frames before it are still on the GPU;
//! - it tries more (k - 1) once k has held without a miss for `settle_ms`, and keeps them if
//!   nothing is missed; if something is, it goes back and waits twice as long before trying again
//!   (up to `max_backoff_ms`);
//! - `max_fps` caps it (60 on a 120 Hz display, where a steady 60 is the aim).
//!
//! The frames' GPU time (`FrameTimer`) is measured too, but only hints: a GPU with fewer frames to
//! draw lowers its clocks (Apple's do), so each frame takes longer at a slower cadence, and a pacer
//! slowing down by GPU time slows down further and further. When recent frames' GPU time is well
//! within a faster cadence even so (`try_below`), and below what it was when that cadence last
//! failed, the pacer tries it at once, whatever the wait (the cost fell: a lighter shot after a
//! heavier one); misses still decide whether it stays.
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
    /// The share of the refreshes over the last `window_ms` that may be missed.
    pub late_share: f64,
    pub window_ms: f64,
    /// How long a cadence must hold without a miss before a faster one is tried, ms.
    pub settle_ms: f64,
    /// The longest wait before trying a faster cadence that failed again, ms.
    pub max_backoff_ms: f64,
    /// How long after the first refresh misses are not counted (loading, warming up), ms.
    pub warmup_ms: f64,
    /// A faster cadence is tried at once when recent frames' GPU time (90th percentile) is below
    /// this share of its frame time.
    pub try_below: f64,
}

impl Default for FramePacerOptions {
    fn default() -> Self {
        Self { max_fps: None, max_divisor: 4, late_factor: 1.5, late_share: 0.05, window_ms: 1000.0, settle_ms: 2000.0, max_backoff_ms: 16000.0, warmup_ms: 1000.0, try_below: 0.85 }
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
    /// The last `window_ms` of refreshes: when, and whether missed
    recent: VecDeque<(f64, bool)>,
    /// Recent rendered frames' GPU time, ms, and what it was when the last try failed
    gpu: VecDeque<f64>,
    failed_gpu: f64,
    divisor: u32,
    /// When the divisor last changed, whether that was a try at a faster cadence, and the wait
    /// before the next try
    changed_at: f64,
    trying: bool,
    backoff_ms: f64,
    tick: u32,
}

impl Cadence {
    const RELEARN_MS: f64 = 30000.0;

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
            failed_gpu: f64::INFINITY,
            divisor: 1,
            changed_at: 0.0,
            trying: false,
            backoff_ms: options.settle_ms,
            tick: 0,
        }
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
        let mut late = false;
        if let Some(last) = self.last_refresh {
            let interval = now_ms - last;
            // (a hidden tab or a stall is not a refresh interval)
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
            while self.recent.front().is_some_and(|&(t, _)| now_ms - t > self.options.window_ms) {
                self.recent.pop_front();
            }
            self.decide(now_ms);
        }
        self.tick = (self.tick + 1) % self.divisor;
        self.tick == 0
    }

    /// A rendered frame's GPU time, ms.
    pub(crate) fn on_gpu_time(&mut self, ms: f64) {
        if ms.is_finite() && ms > 0.0 {
            self.gpu.push_back(ms);
            if self.gpu.len() > 30 {
                self.gpu.pop_front();
            }
        }
    }

    fn set_divisor(&mut self, divisor: u32, now_ms: f64) {
        self.divisor = divisor;
        self.changed_at = now_ms;
        self.tick = 0;
        self.recent.clear();
        self.gpu.clear();
    }

    fn decide(&mut self, now_ms: f64) {
        let o = self.options;
        let floor = self.min_divisor();
        if self.divisor < floor {
            self.set_divisor(floor, now_ms);
            return;
        }
        let since = now_ms - self.changed_at;
        let missed = self.recent.iter().filter(|(_, late)| *late).count() as f64;
        let share = missed / self.recent.len().max(1) as f64;
        // missing refreshes over a whole window: render on fewer (a failed try: wait longer)
        if since >= o.window_ms && share > o.late_share {
            if self.trying {
                self.backoff_ms = (self.backoff_ms * 2.0).min(o.max_backoff_ms);
                self.failed_gpu = self.gpu_p90().unwrap_or(f64::INFINITY);
            }
            self.trying = false;
            if self.divisor < o.max_divisor.max(floor) {
                self.set_divisor(self.divisor + 1, now_ms);
            }
            return;
        }
        // a try that held over a window: keep it
        if self.trying && since >= o.window_ms {
            self.trying = false;
            self.backoff_ms = o.settle_ms;
            self.failed_gpu = f64::INFINITY;
        }
        // held without a miss for long enough, or the GPU time well within a faster cadence: try it
        if !self.trying && self.divisor > floor && missed == 0.0 {
            let lighter = since >= o.window_ms
                && self.gpu_p90().is_some_and(|gpu| gpu < (self.divisor - 1) as f64 * self.refresh_ms * o.try_below && gpu < self.failed_gpu * 0.9);
            if lighter {
                self.backoff_ms = o.settle_ms;
            }
            if lighter || since >= self.backoff_ms {
                self.set_divisor(self.divisor - 1, now_ms);
                self.trying = true;
            }
        }
    }

    fn gpu_p90(&self) -> Option<f64> {
        (self.gpu.len() >= 10).then(|| percentile(&self.gpu, 0.9))
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
}

/// A frame's GPU time: a timestamp written by a one-thread compute pass submitted before the
/// frame's work and another after it (on Metal an empty pass resolves its timestamps to zero,
/// hence the dispatch). Without `TIMESTAMP_QUERY`, the time from the first submit until a buffer
/// copied after the last becomes mappable: an upper bound, queueing included. A ring of readbacks
/// measures every frame although results arrive a few frames late. A frame's stamps are resolved
/// at the next frame's start (or `flush`): wgpu's native Metal backend can resolve them before
/// they are written when asked right after; a frame whose stamps still read back out of order is
/// left unmeasured.
pub struct FrameTimer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    queries: Option<(wgpu::ComputePipeline, f64)>,
    slots: Vec<TimerSlot>,
    armed: Option<usize>,
    started_ms: f64,
    /// Frames ended whose stamps are not resolved yet: their slot and CPU start time
    unresolved: Vec<(usize, f64)>,
    results: Arc<Mutex<Vec<f64>>>,
    last_ms: Arc<Mutex<f64>>,
}

struct TimerSlot {
    set: Option<wgpu::QuerySet>,
    resolve: wgpu::Buffer,
    readback: wgpu::Buffer,
    busy: Arc<AtomicBool>,
}

impl FrameTimer {
    const SLOTS: usize = 8;

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
        let buffer = |usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some("FrameTimer"), size: 32, usage, mapped_at_creation: false });
        let slots = (0..Self::SLOTS)
            .map(|_| TimerSlot {
                set: timestamps.then(|| device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("FrameTimer"), ty: wgpu::QueryType::Timestamp, count: 4 })),
                resolve: buffer(wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                readback: buffer(wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ),
                busy: Arc::new(AtomicBool::new(false)),
            })
            .collect();
        Self {
            device: device.clone(),
            queue: queue.clone(),
            queries,
            slots,
            armed: None,
            started_ms: 0.0,
            unresolved: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
            last_ms: Arc::new(Mutex::new(f64::NAN)),
        }
    }

    fn stamp(&self, encoder: &mut wgpu::CommandEncoder, slot: usize, end: bool) {
        let (Some((noop, _)), Some(set)) = (&self.queries, &self.slots[slot].set) else { return };
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("FrameTimer"),
            // both stamps of both passes (the end of a pass is not always written on every GPU)
            timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                query_set: set,
                beginning_of_pass_write_index: Some(if end { 2 } else { 0 }),
                end_of_pass_write_index: Some(if end { 3 } else { 1 }),
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
            let slot = &self.slots[k];
            if let Some(set) = &slot.set {
                encoder.resolve_query_set(set, 0..4, &slot.resolve, 0);
            }
            encoder.copy_buffer_to_buffer(&slot.resolve, 0, &slot.readback, 0, 32);
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
        let mut in_flight: VecDeque<f64> = VecDeque::new(); // when each frame on the GPU is done
        let mut gpu_free = start;
        let mut rendered = Vec::new();
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
                rendered.push(t);
                gpu_free = gpu_free.max(t) + gpu(t);
                in_flight.push_back(gpu_free);
                cadence.on_gpu_time(gpu(t));
            }
            t += refresh;
        }
        (cadence.divisor(), rendered)
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
            let (k, frames) = simulate(&mut cadence, 0.0, refresh, 30.0, |t| if t < 500.0 { 2.0 } else { gpu });
            assert_eq!(k, divisor, "{gpu} ms frames at {:.0} Hz", 1000.0 / refresh);
            assert!((cadence.refresh_ms() - refresh).abs() < 0.01);
            // and the last seconds are evenly spaced
            let tail: Vec<f64> = frames.into_iter().filter(|&t| t > 27000.0).collect();
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
        let mut t = 0.0;
        let mut tries = Vec::new();
        let mut last = 0;
        while t < 90000.0 {
            let (k, _) = simulate(&mut cadence, t, refresh, 0.1, |_| 15.0);
            if k == 1 && last != 1 {
                tries.push(t);
            }
            last = k;
            t += 100.0;
        }
        assert!(tries.len() >= 3, "{tries:?}");
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

    /// On a real GPU, frames of work are measured (a ring of readbacks); on wgpu's native Metal
    /// backend some stamps read back stale, and those frames go unmeasured.
    #[test]
    fn the_frame_timer_measures_each_frame() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&Default::default())) else { return eprintln!("no GPU adapter: skipping") };
        let features = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor { required_features: features, ..Default::default() }, None)).unwrap();
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
