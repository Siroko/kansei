//! Frame profiling, opt-in (`Renderer::set_profiling`): the GPU time of each labelled pass, from
//! timestamp queries, and the CPU time of each labelled section of the frame.
//!
//! Passes label themselves: `timestamp_writes: stamp.as_ref().map(PassStamp::compute)` with
//! `let stamp = profiling::gpu_pass("fog/inject");` (`None` while profiling is off, or without the
//! `TIMESTAMP_QUERY` feature). Sections time themselves: `let _t = profiling::cpu_scope("upload");`
//! (nested sections count in each). Both cost a thread-local check while profiling is off.
//!
//! GPU passes overlap on tile-based GPUs (a render pass starts its vertex work while the passes
//! before it are still shading), so each pass is charged its *exclusive* time: how far it pushes
//! the frame's GPU timeline past the end of every pass submitted before it. A pass hidden behind
//! earlier work costs nothing, the overlap goes to the pass still running, and the exclusive
//! times sum to the time the GPU spent on the frame's passes (idle gaps aside); `busy` is each
//! pass's own start to end, overlaps included. Readbacks are asynchronous: `Renderer::take_profile`
//! averages the frames that have arrived.

use std::cell::RefCell;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};

/// Timestamps per frame (two per pass): passes past this many go untimed.
const CAPACITY: u32 = 512;
const READBACKS: usize = 8;

/// The timestamps of one pass: pass `compute()` or `render()` as the descriptor's
/// `timestamp_writes`.
pub struct PassStamp {
    set: wgpu::QuerySet,
    index: u32,
}

impl PassStamp {
    pub fn compute(&self) -> wgpu::ComputePassTimestampWrites<'_> {
        wgpu::ComputePassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: Some(self.index), end_of_pass_write_index: Some(self.index + 1) }
    }

    pub fn render(&self) -> wgpu::RenderPassTimestampWrites<'_> {
        wgpu::RenderPassTimestampWrites { query_set: &self.set, beginning_of_pass_write_index: Some(self.index), end_of_pass_write_index: Some(self.index + 1) }
    }
}

/// Timestamps for a pass labelled `label`, while profiling (and the device has timestamps).
pub fn gpu_pass(label: &'static str) -> Option<PassStamp> {
    PROFILER.with(|p| {
        let mut p = p.borrow_mut();
        let p = p.as_mut().filter(|p| p.enabled)?;
        let gpu = p.gpu.as_mut()?;
        if gpu.labels.len() as u32 * 2 >= CAPACITY {
            return None;
        }
        let index = gpu.labels.len() as u32 * 2;
        gpu.labels.push(label);
        Some(PassStamp { set: gpu.set.clone(), index })
    })
}

/// Times a section of the frame's CPU work until dropped.
pub struct CpuScope {
    label: &'static str,
    start: f64,
}

impl Drop for CpuScope {
    fn drop(&mut self) {
        let ms = now_ms() - self.start;
        PROFILER.with(|p| {
            if let Some(p) = p.borrow_mut().as_mut() {
                let entry = p.cpu.iter_mut().find(|(l, _)| *l == self.label);
                match entry {
                    Some((_, total)) => *total += ms,
                    None => p.cpu.push((self.label, ms)),
                }
            }
        });
    }
}

/// Time a section labelled `label` (until the returned guard drops), while profiling.
pub fn cpu_scope(label: &'static str) -> Option<CpuScope> {
    PROFILER.with(|p| p.borrow().as_ref().is_some_and(|p| p.enabled)).then(|| CpuScope { label, start: now_ms() })
}

/// A labelled pass's GPU time, per frame on average.
#[derive(Clone, Debug)]
pub struct PassTime {
    pub label: &'static str,
    /// How far it pushes the frame's GPU timeline past the passes submitted before it (see the
    /// module docs), ms.
    pub exclusive_ms: f64,
    /// Its own start to end, overlaps included, ms.
    pub busy_ms: f64,
    /// Passes with this label per frame.
    pub count: f64,
}

/// Recent frames' profile, averaged per frame.
#[derive(Clone, Debug, Default)]
pub struct FrameProfile {
    /// Frames the GPU times average (they arrive a few frames late).
    pub gpu_frames: u32,
    /// Passes by label, in the order they first ran.
    pub gpu: Vec<PassTime>,
    /// The sum of the passes' exclusive times, ms.
    pub gpu_ms: f64,
    /// The first pass's start to the last pass's end, idle gaps included, ms.
    pub gpu_span_ms: f64,
    /// Frames the CPU times average.
    pub cpu_frames: u32,
    /// CPU sections by label, ms per frame (nested sections count in each).
    pub cpu: Vec<(&'static str, f64)>,
}

impl FrameProfile {
    /// A table, one line per pass then per section, most expensive first.
    pub fn report(&self) -> String {
        let mut gpu = self.gpu.clone();
        gpu.sort_by(|a, b| b.exclusive_ms.total_cmp(&a.exclusive_ms));
        let mut cpu = self.cpu.clone();
        cpu.sort_by(|a, b| b.1.total_cmp(&a.1));
        let mut out = format!("GPU {:.2} ms of passes ({:.2} ms span), {} frames\n", self.gpu_ms, self.gpu_span_ms, self.gpu_frames);
        for p in &gpu {
            out += &format!("  {:<32} {:>7.3} ms  (busy {:.3}, x{:.1})\n", p.label, p.exclusive_ms, p.busy_ms, p.count);
        }
        out += &format!("CPU, {} frames\n", self.cpu_frames);
        for (label, ms) in &cpu {
            out += &format!("  {:<32} {:>7.3} ms\n", label, ms);
        }
        out
    }
}

struct Gpu {
    set: wgpu::QuerySet,
    period_ns: f64,
    resolve: wgpu::Buffer,
    readbacks: Vec<Readback>,
    // this frame's passes, in the order their stamps were handed out
    labels: Vec<&'static str>,
}

struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
}

const FREE: u8 = 0;
const MAPPING: u8 = 1;

/// A frame's passes: (label, begin ns, end ns).
type FramePasses = Vec<(&'static str, u64, u64)>;

struct Profiler {
    enabled: bool,
    gpu: Option<Gpu>,
    // frames whose timestamps have arrived
    arrived: Arc<Mutex<Vec<FramePasses>>>,
    cpu: Vec<(&'static str, f64)>,
    cpu_frames: u32,
}

thread_local! {
    static PROFILER: RefCell<Option<Profiler>> = const { RefCell::new(None) };
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

/// Turn profiling on or off (on this thread; the renderer's).
pub(crate) fn set_enabled(device: &wgpu::Device, queue: &wgpu::Queue, enabled: bool) {
    PROFILER.with(|p| {
        let mut p = p.borrow_mut();
        let p = p.get_or_insert_with(|| Profiler { enabled: false, gpu: None, arrived: Arc::new(Mutex::new(Vec::new())), cpu: Vec::new(), cpu_frames: 0 });
        p.enabled = enabled;
        if enabled && p.gpu.is_none() && device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            let buffer = |label, usage| device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: CAPACITY as u64 * 8, usage, mapped_at_creation: false });
            p.gpu = Some(Gpu {
                set: device.create_query_set(&wgpu::QuerySetDescriptor { label: Some("Profiler"), ty: wgpu::QueryType::Timestamp, count: CAPACITY }),
                period_ns: queue.get_timestamp_period() as f64,
                resolve: buffer("Profiler/Resolve", wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC),
                readbacks: (0..READBACKS)
                    .map(|_| Readback { buffer: buffer("Profiler/Readback", wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST), state: Arc::new(AtomicU8::new(FREE)) })
                    .collect(),
                labels: Vec::new(),
            });
        }
    });
}

/// End the frame's GPU timing: resolve its passes' timestamps and read them back (after the
/// frame's last submit). Frames that find every readback busy go untimed.
pub(crate) fn end_frame(device: &wgpu::Device, queue: &wgpu::Queue) {
    PROFILER.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut().filter(|p| p.enabled) else { return };
        p.cpu_frames += 1;
        let Some(gpu) = p.gpu.as_mut() else { return };
        let labels = std::mem::take(&mut gpu.labels);
        if labels.is_empty() {
            return;
        }
        let Some(readback) = gpu.readbacks.iter().find(|r| r.state.load(Ordering::Acquire) == FREE) else { return };
        readback.state.store(MAPPING, Ordering::Release);
        let count = labels.len() as u32 * 2;
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Profiler") });
        encoder.resolve_query_set(&gpu.set, 0..count, &gpu.resolve, 0);
        encoder.copy_buffer_to_buffer(&gpu.resolve, 0, &readback.buffer, 0, count as u64 * 8);
        queue.submit(Some(encoder.finish()));
        let (buffer, state, arrived, period) = (readback.buffer.clone(), readback.state.clone(), p.arrived.clone(), gpu.period_ns);
        readback.buffer.slice(..count as u64 * 8).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                let stamps: Vec<u64> = bytemuck::cast_slice(&buffer.slice(..count as u64 * 8).get_mapped_range()).to_vec();
                buffer.unmap();
                let to_ns = |t: u64| (t as f64 * period) as u64;
                let passes = labels.iter().enumerate().map(|(k, &label)| (label, to_ns(stamps[2 * k]), to_ns(stamps[2 * k + 1]))).collect();
                arrived.lock().unwrap().push(passes);
            }
            state.store(FREE, Ordering::Release);
        });
    });
}

/// The frames profiled since the last call, averaged (and forgotten).
pub(crate) fn take(device: &wgpu::Device) -> FrameProfile {
    #[cfg(not(target_arch = "wasm32"))]
    device.poll(wgpu::Maintain::Poll);
    #[cfg(target_arch = "wasm32")]
    let _ = device;
    PROFILER.with(|p| {
        let mut p = p.borrow_mut();
        let Some(p) = p.as_mut() else { return FrameProfile::default() };
        let frames: Vec<_> = std::mem::take(&mut *p.arrived.lock().unwrap());
        let mut profile = gpu_profile(&frames);
        profile.cpu_frames = p.cpu_frames;
        for (label, total) in std::mem::take(&mut p.cpu) {
            profile.cpu.push((label, total / p.cpu_frames.max(1) as f64));
        }
        p.cpu_frames = 0;
        profile
    })
}

/// Average frames of passes `(label, begin ns, end ns)` (the GPU side of a `FrameProfile`).
fn gpu_profile(frames: &[FramePasses]) -> FrameProfile {
    let mut profile = FrameProfile::default();
    {
        for passes in frames {
            // the passes that ran (Metal resolves an empty pass's timestamps to 0), in submission
            // order, which is the GPU's
            let ran: Vec<_> = passes.iter().filter(|(_, b, e)| *b > 0 && e >= b).collect();
            let Some(first) = ran.iter().map(|(_, b, _)| *b).min() else { continue };
            profile.gpu_frames += 1;
            profile.gpu_span_ms += (ran.iter().map(|(_, _, e)| *e).max().unwrap() - first) as f64 / 1e6;
            // the end of everything submitted so far
            let mut frontier = 0u64;
            for &&(label, begin, end) in &ran {
                let exclusive = end.saturating_sub(begin.max(frontier)) as f64 / 1e6;
                frontier = frontier.max(end);
                let busy = (end - begin) as f64 / 1e6;
                profile.gpu_ms += exclusive;
                match profile.gpu.iter_mut().find(|t| t.label == label) {
                    Some(t) => {
                        t.exclusive_ms += exclusive;
                        t.busy_ms += busy;
                        t.count += 1.0;
                    }
                    None => profile.gpu.push(PassTime { label, exclusive_ms: exclusive, busy_ms: busy, count: 1.0 }),
                }
            }
        }
        let n = profile.gpu_frames.max(1) as f64;
        profile.gpu_ms /= n;
        profile.gpu_span_ms /= n;
        for t in &mut profile.gpu {
            t.exclusive_ms /= n;
            t.busy_ms /= n;
            t.count /= n;
        }
    }
    profile
}

/// How [`AbBench`] alternates: a warm-up, then `phases` phases of `phase_ms` each, A first, each
/// measured after `settle_ms` (caches, temporal effects and the pacing catch up first).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbBenchOptions {
    pub warmup_ms: f64,
    pub phase_ms: f64,
    pub settle_ms: f64,
    pub phases: u32,
}

impl Default for AbBenchOptions {
    fn default() -> Self {
        Self { warmup_ms: 3000.0, phase_ms: 3000.0, settle_ms: 500.0, phases: 8 }
    }
}

/// An A/B benchmark inside one page: the page switches between two variants (`phase`), feeds
/// each frame's GPU times (`pacing::FrameTimer::take`) and the clock (`record`), and gets one
/// report line comparing the two when the phases are over. Alternating A and B in one session is
/// what AGENTS.md prescribes, because separate runs on a shared GPU differ by tens of percent.
#[derive(Debug, Clone)]
pub struct AbBench {
    labels: [String; 2],
    options: AbBenchOptions,
    start_ms: f64,
    last_ms: f64,
    /// Per variant: GPU ms summed, GPU samples, frame intervals ms summed, frames.
    sums: [(f64, u32, f64, u32); 2],
    report: Option<String>,
}

impl AbBench {
    /// A bench of variants `labels[0]` (A) and `labels[1]` (B), starting at `now_ms`.
    pub fn new(labels: [&str; 2], now_ms: f64, options: AbBenchOptions) -> Self {
        Self { labels: labels.map(str::to_string), options, start_ms: now_ms, last_ms: now_ms, sums: [(0.0, 0, 0.0, 0); 2], report: None }
    }

    /// The variant to draw at `now_ms` (0: A, 1: B, A through the warm-up) and whether this
    /// frame is measured; `None` once the phases are over.
    pub fn phase(&self, now_ms: f64) -> Option<(usize, bool)> {
        let t = now_ms - self.start_ms - self.options.warmup_ms;
        if t < 0.0 {
            return Some((0, false));
        }
        let phase = (t / self.options.phase_ms) as u32;
        (phase < self.options.phases).then_some(((phase % 2) as usize, t % self.options.phase_ms >= self.options.settle_ms))
    }

    /// Which A-then-B pair of phases `now_ms` falls in (0 through the warm-up): hold the view
    /// still per pair so both variants see the same frames.
    pub fn pair(&self, now_ms: f64) -> u32 {
        ((now_ms - self.start_ms - self.options.warmup_ms).max(0.0) / self.options.phase_ms) as u32 / 2
    }

    /// Record a frame at `now_ms` with the GPU times that arrived since the last one (ms, often
    /// several or none: they arrive late). Returns the report on the frame it completes.
    pub fn record(&mut self, gpu_ms: &[f64], now_ms: f64) -> Option<&str> {
        if let Some((variant, true)) = self.phase(now_ms) {
            let sum = &mut self.sums[variant];
            sum.0 += gpu_ms.iter().sum::<f64>();
            sum.1 += gpu_ms.len() as u32;
            sum.2 += now_ms - self.last_ms;
            sum.3 += 1;
        }
        self.last_ms = now_ms;
        if self.report.is_some() || self.phase(now_ms).is_some() {
            return None;
        }
        let side = |label: &str, (gpu, samples, interval, frames): (f64, u32, f64, u32)| {
            let gpu = if samples > 0 { format!("{:.2} ms GPU ({samples} samples)", gpu / samples as f64) } else { "no GPU timestamps".to_string() };
            format!("{label} {gpu}, {:.2} ms/frame ({frames} frames)", interval / frames.max(1) as f64)
        };
        self.report = Some(format!("bench: {} | {}", side(&self.labels[0], self.sums[0]), side(&self.labels[1], self.sums[1])));
        self.report.as_deref()
    }

    /// The report, once the phases are over.
    pub fn report(&self) -> Option<&str> {
        self.report.as_deref()
    }
}

impl FrameProfile {
    /// The `n` most expensive passes by exclusive time, (label, ms per frame), for an overlay.
    pub fn top_passes(&self, n: usize) -> Vec<(&'static str, f64)> {
        let mut passes: Vec<(&'static str, f64)> = self.gpu.iter().map(|p| (p.label, p.exclusive_ms)).collect();
        passes.sort_by(|a, b| b.1.total_cmp(&a.1));
        passes.truncate(n);
        passes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exclusive times partition the GPU timeline: an overlapped stretch goes to the pass still
    /// running (the earlier one), a pass hidden behind earlier work costs nothing, idle gaps go to
    /// none, passes that did not run (zero stamps) are skipped; same labels add up.
    #[test]
    fn exclusive_times_partition_the_timeline() {
        const MS: u64 = 1_000_000;
        const T: u64 = 1000 * MS; // the frame's start on the GPU clock
        let frame = vec![
            ("shadow", T, T + 2 * MS),
            // starts before the shadow pass ends (tile-based overlap)
            ("gbuffer", T + MS, T + 5 * MS),
            // starts and ends while the GBuffer pass is still running
            ("hidden", T + 3 * MS, T + 4 * MS),
            // an idle millisecond, then two passes with the same label
            ("post", T + 6 * MS, T + 7 * MS),
            ("post", T + 7 * MS, T + 9 * MS),
            // did not run
            ("empty", 0, 0),
        ];
        let p = gpu_profile(&[frame.clone(), frame]);
        assert_eq!(p.gpu_frames, 2);
        let get = |label| p.gpu.iter().find(|t| t.label == label).unwrap();
        assert_eq!((get("shadow").exclusive_ms, get("shadow").busy_ms), (2.0, 2.0));
        assert_eq!((get("gbuffer").exclusive_ms, get("gbuffer").busy_ms), (3.0, 4.0));
        assert_eq!((get("hidden").exclusive_ms, get("hidden").busy_ms), (0.0, 1.0));
        assert_eq!((get("post").exclusive_ms, get("post").count), (3.0, 2.0));
        assert!(p.gpu.iter().all(|t| t.label != "empty"));
        assert_eq!((p.gpu_ms, p.gpu_span_ms), (8.0, 9.0));
    }

    /// A warm-up on A, then A and B in turn, each measured after it settles; the report averages
    /// each side's GPU samples and frame intervals.
    #[test]
    fn the_ab_bench_alternates_and_averages_each_side() {
        let options = AbBenchOptions { warmup_ms: 100.0, phase_ms: 100.0, settle_ms: 20.0, phases: 4 };
        let mut bench = AbBench::new(["clusters", "lods"], 0.0, options);
        assert_eq!(bench.phase(50.0), Some((0, false)));
        assert_eq!(bench.phase(110.0), Some((0, false)));
        assert_eq!(bench.phase(130.0), Some((0, true)));
        assert_eq!(bench.phase(230.0), Some((1, true)));
        assert_eq!((bench.pair(250.0), bench.pair(350.0)), (0, 1));
        let mut report = None;
        let mut t = 0.0;
        while t <= 520.0 {
            // A costs 2 ms on the GPU, B 3 ms; one sample a frame, 10 ms apart
            let gpu = match bench.phase(t) { Some((1, _)) => 3.0, _ => 2.0 };
            if let Some(r) = bench.record(&[gpu], t) {
                report = Some(r.to_string());
            }
            t += 10.0;
        }
        let report = report.expect("a report once the phases are over");
        assert!(report.starts_with("bench: clusters 2.00 ms GPU"), "{report}");
        assert!(report.contains("| lods 3.00 ms GPU"), "{report}");
        assert!(report.contains("10.00 ms/frame"), "{report}");
        assert_eq!(bench.phase(600.0), None);
        assert_eq!(bench.report(), Some(report.as_str()));
    }
}
