/**
 * How `AbBench` alternates: a warm-up, then `phases` phases of `phaseMs` each, A first, each
 * measured after `settleMs` (caches, temporal effects and the pacing catch up first).
 */
export interface AbBenchOptions {
    warmupMs: number;
    phaseMs: number;
    settleMs: number;
    phases: number;
}

export const DEFAULT_AB_BENCH_OPTIONS: AbBenchOptions = { warmupMs: 3000, phaseMs: 3000, settleMs: 500, phases: 8 };

/** Per variant: GPU ms summed, GPU samples, frame intervals ms summed, frames. */
type Sums = [number, number, number, number];

/**
 * An A/B benchmark inside one page: the page switches between two variants (`phase`), feeds each
 * frame's GPU times (`FrameTimer.take`) and the clock (`record`), and gets one report line
 * comparing the two when the phases are over. Alternating A and B in one session is what
 * AGENTS.md prescribes, because separate runs on a shared GPU differ by tens of percent.
 * A port of the Rust engine's `profiling::AbBench`.
 */
export class AbBench {
    private readonly labels: [string, string];
    private readonly options: AbBenchOptions;
    private readonly startMs: number;
    private lastMs: number;
    private readonly sums: [Sums, Sums] = [[0, 0, 0, 0], [0, 0, 0, 0]];
    private _report: string | null = null;

    /** A bench of variants `labels[0]` (A) and `labels[1]` (B), starting at `nowMs`. */
    constructor(labels: [string, string], nowMs: number, options: Partial<AbBenchOptions> = {}) {
        this.labels = labels;
        this.options = { ...DEFAULT_AB_BENCH_OPTIONS, ...options };
        this.startMs = nowMs;
        this.lastMs = nowMs;
    }

    /**
     * The variant to draw at `nowMs` (0: A, 1: B, A through the warm-up) and whether this frame
     * is measured; `null` once the phases are over.
     */
    phase(nowMs: number): [0 | 1, boolean] | null {
        const t = nowMs - this.startMs - this.options.warmupMs;
        if (t < 0) return [0, false];
        const phase = Math.floor(t / this.options.phaseMs);
        if (phase >= this.options.phases) return null;
        return [(phase % 2) as 0 | 1, t % this.options.phaseMs >= this.options.settleMs];
    }

    /**
     * Which A-then-B pair of phases `nowMs` falls in (0 through the warm-up): hold the view still
     * per pair so both variants see the same frames.
     */
    pair(nowMs: number): number {
        return Math.floor(Math.floor(Math.max(nowMs - this.startMs - this.options.warmupMs, 0) / this.options.phaseMs) / 2);
    }

    /**
     * Record a frame at `nowMs` with the GPU times that arrived since the last one (ms, often
     * several or none: they arrive late). Returns the report on the frame it completes.
     */
    record(gpuMs: number[], nowMs: number): string | null {
        const phase = this.phase(nowMs);
        if (phase && phase[1]) {
            const sum = this.sums[phase[0]];
            sum[0] += gpuMs.reduce((a, b) => a + b, 0);
            sum[1] += gpuMs.length;
            sum[2] += nowMs - this.lastMs;
            sum[3] += 1;
        }
        this.lastMs = nowMs;
        if (this._report !== null || phase !== null) return null;
        const side = (label: string, [gpu, samples, interval, frames]: Sums) => {
            const g = samples > 0 ? `${(gpu / samples).toFixed(2)} ms GPU (${samples} samples)` : 'no GPU timestamps';
            return `${label} ${g}, ${(interval / Math.max(frames, 1)).toFixed(2)} ms/frame (${frames} frames)`;
        };
        this._report = `bench: ${side(this.labels[0], this.sums[0])} | ${side(this.labels[1], this.sums[1])}`;
        return this._report;
    }

    /** The report, once the phases are over. */
    report(): string | null {
        return this._report;
    }
}
