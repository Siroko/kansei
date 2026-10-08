//! Timings to compare this crate's motion matching (Rust in wasm) with the TS engine's port
//! (`examples/index_motion_matching.html`): the character's update per frame, and two benches
//! both pages run the same way (`bench_search`, `bench_pose`).

use kansei_core::animation::motion_matching::{Database, SearchFilter, STRIDE};
use kansei_core::animation::{Pose, Transform};

/// Frames a `Window` keeps.
pub const WINDOW: usize = 3000;

/// The last `WINDOW` samples (ms).
#[derive(Default)]
pub struct Window {
    samples: Vec<f32>,
    next: usize,
}

impl Window {
    pub fn push(&mut self, ms: f32) {
        if self.samples.len() < WINDOW {
            self.samples.push(ms);
        } else {
            self.samples[self.next] = ms;
        }
        self.next = (self.next + 1) % WINDOW;
    }

    /// Mean, median, 95th percentile and largest, and the sample count.
    fn stats(&self) -> (f32, f32, f32, f32, usize) {
        if self.samples.is_empty() {
            return (0.0, 0.0, 0.0, 0.0, 0);
        }
        let mut sorted = self.samples.clone();
        sorted.sort_by(f32::total_cmp);
        let at = |q: f32| sorted[((sorted.len() - 1) as f32 * q).round() as usize];
        (sorted.iter().sum::<f32>() / sorted.len() as f32, at(0.5), at(0.95), sorted[sorted.len() - 1], sorted.len())
    }

    pub fn summary(&self) -> String {
        let (mean, _, p95, max, _) = self.stats();
        format!("{mean:.3} ms mean, {p95:.3} p95, {max:.3} max (Rust, wasm)")
    }

    pub fn json(&self) -> String {
        let (mean, p50, p95, max, n) = self.stats();
        format!("{{\"engine\":\"rust-wasm\",\"update\":{{\"mean\":{mean},\"p50\":{p50},\"p95\":{p95},\"max\":{max},\"n\":{n}}}}}")
    }
}

/// The search bench's queries: every `step`th frame's features, nudged off the frame itself.
fn queries(db: &Database, step: usize) -> Vec<[f32; STRIDE]> {
    (0..db.frame_count())
        .step_by(step.max(1))
        .map(|f| {
            let mut q = [0.0; STRIDE];
            q.copy_from_slice(db.features(f));
            q[0] += 0.3;
            q[20] -= 0.2;
            q
        })
        .collect()
}

/// Milliseconds per search over `queries(step)` with the default filter (a warm-up pass first).
pub fn bench_search(db: &Database, step: usize) -> f64 {
    let queries = queries(db, step);
    let filter = SearchFilter::default();
    let mut checksum = 0usize;
    for q in &queries {
        checksum += db.search(q, &filter, f32::MAX).map_or(0, |m| m.frame);
    }
    let t0 = kansei_wasm::now();
    for q in &queries {
        checksum += db.search(q, &filter, f32::MAX).map_or(0, |m| m.frame);
    }
    let ms = (kansei_wasm::now() - t0) * 1000.0 / queries.len().max(1) as f64;
    log::info!("bench_search: {ms:.4} ms per search over {} queries (checksum {checksum})", queries.len());
    ms
}

/// Milliseconds per pose: frames `f` and `f + 1` blended halfway, then `finish` (forward kinematics,
/// the palette), `count` times over the database (a warm-up pass first).
pub fn bench_pose(db: &Database, count: usize, mut finish: impl FnMut(&Pose, &mut Vec<Transform>)) -> f64 {
    let mut pose = Pose { local: Vec::new() };
    let mut model = Vec::new();
    let frames = db.frame_count().max(2) - 1;
    let mut run = |n: usize| {
        for i in 0..n {
            let f = (i * 7) % frames;
            db.pose(f, f + 1, 0.5, &mut pose);
            finish(&pose, &mut model);
        }
    };
    run(count.min(100));
    let t0 = kansei_wasm::now();
    run(count);
    let ms = (kansei_wasm::now() - t0) * 1000.0 / count.max(1) as f64;
    log::info!("bench_pose: {ms:.4} ms per pose over {count}");
    ms
}
