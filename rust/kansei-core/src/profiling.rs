//! Frame profiling, opt-in (`Renderer::set_profiling`): the GPU time of each labelled pass, from
//! timestamp queries, and the CPU time of each labelled section of the frame.
//!
//! Passes label themselves: `timestamp_writes: stamp.as_ref().map(PassStamp::compute)` with
//! `let stamp = profiling::gpu_pass("fog/inject");` (`None` while profiling is off, or without the
//! `TIMESTAMP_QUERY` feature). Sections time themselves: `let _t = profiling::cpu_scope("upload");`
//! (nested sections count in each). Both cost a thread-local check while profiling is off.
//!
//! GPU passes overlap on tile-based GPUs, so each pass is charged its *exclusive* time: from its
//! start to the next pass's start (or its own end, whichever is first), along the frame's GPU
//! timeline. Those sum to the time the GPU spent on the frame's passes; `busy` is each pass's own
//! start to end, overlaps included. Readbacks are asynchronous: `Renderer::profile` averages the
//! frames that have arrived.

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
    /// Its share of the frame's GPU timeline (see the module docs), ms.
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
            // the passes that ran (Metal resolves an empty pass's timestamps to 0), by start
            let mut ran: Vec<_> = passes.iter().filter(|(_, b, e)| *b > 0 && e >= b).collect();
            ran.sort_by_key(|(_, b, _)| *b);
            let Some(first) = ran.first() else { continue };
            profile.gpu_frames += 1;
            profile.gpu_span_ms += (ran.iter().map(|(_, _, e)| *e).max().unwrap() - first.1) as f64 / 1e6;
            for (k, &&(label, begin, end)) in ran.iter().enumerate() {
                let next = ran.get(k + 1).map_or(end, |n| n.1);
                let exclusive = (end.min(next).max(begin) - begin) as f64 / 1e6;
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Exclusive times partition the GPU timeline: an overlapped stretch goes to the later pass,
    /// idle gaps to none, passes that did not run (zero stamps) are skipped; same labels add up.
    #[test]
    fn exclusive_times_partition_the_timeline() {
        const MS: u64 = 1_000_000;
        const T: u64 = 1000 * MS; // the frame's start on the GPU clock
        let frame = vec![
            ("shadow", T, T + 2 * MS),
            // starts before the shadow pass ends (tile-based overlap)
            ("gbuffer", T + MS, T + 5 * MS),
            // an idle millisecond, then two passes with the same label
            ("post", T + 6 * MS, T + 7 * MS),
            ("post", T + 7 * MS, T + 9 * MS),
            // did not run
            ("empty", 0, 0),
        ];
        let p = gpu_profile(&[frame.clone(), frame]);
        assert_eq!(p.gpu_frames, 2);
        let get = |label| p.gpu.iter().find(|t| t.label == label).unwrap();
        assert_eq!((get("shadow").exclusive_ms, get("shadow").busy_ms), (1.0, 2.0));
        assert_eq!(get("gbuffer").exclusive_ms, 4.0);
        assert_eq!((get("post").exclusive_ms, get("post").count), (3.0, 2.0));
        assert!(p.gpu.iter().all(|t| t.label != "empty"));
        assert_eq!((p.gpu_ms, p.gpu_span_ms), (8.0, 9.0));
    }
}
