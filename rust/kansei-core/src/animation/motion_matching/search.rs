use super::database::{box_distance, distance, Database, BOUND_LARGE, BOUND_SMALL, STRIDE};

/// What a search may return.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchFilter {
    /// Frames at the end of a clip that doesn't loop that the search never lands on, so playback
    /// has time to search again before the clip runs out.
    pub ignore_end: usize,
    /// Only clips with one of these tag bits (every clip when `u32::MAX`).
    pub tags: u32,
    /// The frame playing now: frames of its clip within `ignore_near` of it are skipped (landing
    /// next to the playhead only restarts what is already playing).
    pub current: Option<usize>,
    pub ignore_near: usize,
}

impl Default for SearchFilter {
    fn default() -> Self {
        Self { ignore_end: 10, tags: u32::MAX, current: None, ignore_near: 3 }
    }
}

/// The best match found: a database frame and its squared feature distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Match {
    pub frame: usize,
    pub cost: f32,
}

impl Database {
    /// The frame whose features are nearest `query` (normalized, see `normalize_query`), if any is
    /// nearer than `best_cost`. Brute force over every allowed frame, skipping runs of frames
    /// whose bounding box is already too far (Holden et al., "Learned Motion Matching", 2020).
    pub fn search(&self, query: &[f32; STRIDE], filter: &SearchFilter, best_cost: f32) -> Option<Match> {
        let mut best: Option<Match> = None;
        let mut limit = best_cost;
        let current_clip = filter.current.map(|f| self.clip_of(f));
        for (c, clip) in self.clips.iter().enumerate() {
            if clip.tags & filter.tags == 0 {
                continue;
            }
            let end = clip.start + if clip.looping { clip.playable() } else { clip.frames.saturating_sub(filter.ignore_end).max(1) };
            // the frames near the playhead, in its clip
            let skip = match (current_clip, filter.current) {
                (Some(cc), Some(f)) if cc == c => f.saturating_sub(filter.ignore_near)..f + filter.ignore_near + 1,
                _ => 0..0,
            };
            let mut i = clip.start;
            while i < end {
                let large = i / BOUND_LARGE;
                let large_end = ((large + 1) * BOUND_LARGE).min(end);
                if box_distance(query, &self.bounds_large[large], limit) >= limit {
                    i = large_end;
                    continue;
                }
                while i < large_end {
                    let small = i / BOUND_SMALL;
                    let small_end = ((small + 1) * BOUND_SMALL).min(large_end);
                    if box_distance(query, &self.bounds_small[small], limit) >= limit {
                        i = small_end;
                        continue;
                    }
                    while i < small_end {
                        if !skip.contains(&i) {
                            let cost = distance(query, self.features(i), limit);
                            if cost < limit {
                                limit = cost;
                                best = Some(Match { frame: i, cost });
                            }
                        }
                        i += 1;
                    }
                }
            }
        }
        best
    }

    /// `search` without the bounding boxes: every allowed frame's full distance. The reference the
    /// accelerated search must agree with.
    pub fn search_brute_force(&self, query: &[f32; STRIDE], filter: &SearchFilter, best_cost: f32) -> Option<Match> {
        let mut best: Option<Match> = None;
        let mut limit = best_cost;
        let current_clip = filter.current.map(|f| self.clip_of(f));
        for (c, clip) in self.clips.iter().enumerate() {
            if clip.tags & filter.tags == 0 {
                continue;
            }
            let end = clip.start + if clip.looping { clip.playable() } else { clip.frames.saturating_sub(filter.ignore_end).max(1) };
            for i in clip.start..end {
                if let (Some(cc), Some(f)) = (current_clip, filter.current) {
                    if cc == c && i + filter.ignore_near >= f && i <= f + filter.ignore_near {
                        continue;
                    }
                }
                let cost = distance(query, self.features(i), f32::MAX);
                if cost < limit {
                    limit = cost;
                    best = Some(Match { frame: i, cost });
                }
            }
        }
        best
    }
}
