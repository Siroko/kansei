/**
 * Frame profiling, opt-in (`Renderer.setProfiling`): the GPU time of each labelled pass, from
 * timestamp queries, and the CPU time of each labelled section of the frame. A port of the Rust
 * engine's `profiling` module (`rust/kansei-core/src/profiling.rs`), with the same pass labels.
 *
 * Passes label themselves: `beginComputePass({ label, timestampWrites: gpuPass('Fog/Inject') })`
 * (`undefined` while profiling is off, or without the `timestamp-query` feature). Sections time
 * themselves: `const t = cpuScope('upload'); ...; t?.end();` (nested sections count in each).
 * Both cost a null check while profiling is off.
 *
 * GPU passes overlap on tile-based GPUs (a render pass starts its vertex work while the passes
 * before it are still shading), so each pass is charged its *exclusive* time: how far it pushes
 * the frame's GPU timeline past the end of every pass submitted before it. A pass hidden behind
 * earlier work costs nothing, the overlap goes to the pass still running, and the exclusive times
 * sum to the time the GPU spent on the frame's passes (idle gaps aside); `busyMs` is each pass's
 * own start to end, overlaps included. Readbacks are asynchronous: `Renderer.takeProfile`
 * averages the frames that have arrived.
 *
 * The profiler is one per page (Rust keeps one per thread): profile one renderer at a time.
 * Chrome quantizes timestamps to 100 µs unless it runs with `--enable-webgpu-developer-features`.
 */

/** Timestamps per frame (two per pass): passes past this many go untimed. */
const CAPACITY = 512;
const READBACKS = 8;

/** The timestamp writes of one pass, for a render or a compute pass descriptor. */
export type PassTimestampWrites = GPURenderPassTimestampWrites & GPUComputePassTimestampWrites;

/** A labelled pass's GPU time, per frame on average. */
export interface PassTime {
    label: string;
    /** How far it pushes the frame's GPU timeline past the passes submitted before it (see the module docs), ms. */
    exclusiveMs: number;
    /** Its own start to end, overlaps included, ms. */
    busyMs: number;
    /** Passes with this label per frame. */
    count: number;
}

/** Recent frames' profile, averaged per frame. */
export class FrameProfile {
    /** Frames the GPU times average (they arrive a few frames late). */
    gpuFrames = 0;
    /** Passes by label, in the order they first ran. */
    gpu: PassTime[] = [];
    /** The sum of the passes' exclusive times, ms. */
    gpuMs = 0;
    /** The first pass's start to the last pass's end, idle gaps included, ms. */
    gpuSpanMs = 0;
    /** Frames the CPU times average. */
    cpuFrames = 0;
    /** CPU sections by label, ms per frame (nested sections count in each). */
    cpu: [string, number][] = [];

    /** A table, one line per pass then per section, most expensive first. */
    report(): string {
        const gpu = [...this.gpu].sort((a, b) => b.exclusiveMs - a.exclusiveMs);
        const cpu = [...this.cpu].sort((a, b) => b[1] - a[1]);
        let out = `GPU ${this.gpuMs.toFixed(2)} ms of passes (${this.gpuSpanMs.toFixed(2)} ms span), ${this.gpuFrames} frames\n`;
        for (const p of gpu) {
            out += `  ${p.label.padEnd(32)} ${p.exclusiveMs.toFixed(3).padStart(7)} ms  (busy ${p.busyMs.toFixed(3)}, x${p.count.toFixed(1)})\n`;
        }
        out += `CPU, ${this.cpuFrames} frames\n`;
        for (const [label, ms] of cpu) {
            out += `  ${label.padEnd(32)} ${ms.toFixed(3).padStart(7)} ms\n`;
        }
        return out;
    }

    /** The `n` most expensive passes by exclusive time, [label, ms per frame], for an overlay. */
    topPasses(n: number): [string, number][] {
        return this.gpu
            .map((p): [string, number] => [p.label, p.exclusiveMs])
            .sort((a, b) => b[1] - a[1])
            .slice(0, n);
    }
}

/** A frame's passes: [label, begin ns, end ns] (0 for a pass that did not run). */
export type FramePasses = [string, number, number][];

/** Average frames of passes (the GPU side of a `FrameProfile`). */
export function gpuProfile(frames: FramePasses[]): FrameProfile {
    const profile = new FrameProfile();
    for (const passes of frames) {
        // the passes that ran (Metal resolves an empty pass's timestamps to 0), in submission
        // order, which is the GPU's
        const ran = passes.filter(([, b, e]) => b > 0 && e >= b);
        if (ran.length === 0) continue;
        const first = Math.min(...ran.map(([, b]) => b));
        profile.gpuFrames++;
        profile.gpuSpanMs += (Math.max(...ran.map(([, , e]) => e)) - first) / 1e6;
        // the end of everything submitted so far
        let frontier = 0;
        for (const [label, begin, end] of ran) {
            const exclusive = Math.max(0, end - Math.max(begin, frontier)) / 1e6;
            frontier = Math.max(frontier, end);
            const busy = (end - begin) / 1e6;
            profile.gpuMs += exclusive;
            const entry = profile.gpu.find((t) => t.label === label);
            if (entry) {
                entry.exclusiveMs += exclusive;
                entry.busyMs += busy;
                entry.count += 1;
            } else {
                profile.gpu.push({ label, exclusiveMs: exclusive, busyMs: busy, count: 1 });
            }
        }
    }
    const n = Math.max(profile.gpuFrames, 1);
    profile.gpuMs /= n;
    profile.gpuSpanMs /= n;
    for (const t of profile.gpu) {
        t.exclusiveMs /= n;
        t.busyMs /= n;
        t.count /= n;
    }
    return profile;
}

interface Readback {
    buffer: GPUBuffer;
    free: boolean;
}

interface Gpu {
    device: GPUDevice;
    set: GPUQuerySet;
    resolve: GPUBuffer;
    readbacks: Readback[];
    // this frame's passes, in the order their stamps were handed out
    labels: string[];
}

interface ProfilerState {
    enabled: boolean;
    gpu: Gpu | null;
    // frames whose timestamps have arrived
    arrived: FramePasses[];
    cpu: Map<string, number>;
    cpuFrames: number;
}

let profiler: ProfilerState | null = null;

/** Timestamp writes for a pass labelled `label`, while profiling (and the device has timestamps). */
export function gpuPass(label: string): PassTimestampWrites | undefined {
    const gpu = profiler?.enabled ? profiler.gpu : null;
    if (!gpu || gpu.labels.length * 2 >= CAPACITY) return undefined;
    const index = gpu.labels.length * 2;
    gpu.labels.push(label);
    return { querySet: gpu.set, beginningOfPassWriteIndex: index, endOfPassWriteIndex: index + 1 };
}

/** Times a section of the frame's CPU work until `end()`. */
export class CpuScope {
    private readonly start = performance.now();
    constructor(private readonly label: string) {}

    end(): void {
        const ms = performance.now() - this.start;
        if (profiler) profiler.cpu.set(this.label, (profiler.cpu.get(this.label) ?? 0) + ms);
    }
}

/** Time a section labelled `label` (until the returned scope's `end()`), while profiling. */
export function cpuScope(label: string): CpuScope | undefined {
    return profiler?.enabled ? new CpuScope(label) : undefined;
}

/** Turn profiling on or off (see `Renderer.setProfiling`). */
export function setProfilingEnabled(device: GPUDevice, enabled: boolean): void {
    profiler ??= { enabled: false, gpu: null, arrived: [], cpu: new Map(), cpuFrames: 0 };
    profiler.enabled = enabled;
    // passes of a frame that profiling was turned off in are not the next frame's
    if (enabled && profiler.gpu) profiler.gpu.labels = [];
    if (enabled && profiler.gpu?.device !== device) {
        profiler.gpu = null;
        if (device.features.has('timestamp-query')) {
            const buffer = (label: string, usage: number) => device.createBuffer({ label, size: CAPACITY * 8, usage });
            profiler.gpu = {
                device,
                set: device.createQuerySet({ label: 'Profiler', type: 'timestamp', count: CAPACITY }),
                resolve: buffer('Profiler/Resolve', GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC),
                readbacks: Array.from({ length: READBACKS }, () => ({
                    buffer: buffer('Profiler/Readback', GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
                    free: true,
                })),
                labels: [],
            };
        }
    }
}

/**
 * End the frame's GPU timing: resolve its passes' timestamps and read them back (after the
 * frame's last submit). Frames that find every readback busy go untimed.
 */
export function endProfiledFrame(): void {
    const p = profiler;
    if (!p?.enabled) return;
    p.cpuFrames++;
    const gpu = p.gpu;
    if (!gpu) return;
    const labels = gpu.labels;
    gpu.labels = [];
    if (labels.length === 0) return;
    const readback = gpu.readbacks.find((r) => r.free);
    if (!readback) return;
    readback.free = false;
    const count = labels.length * 2;
    const encoder = gpu.device.createCommandEncoder({ label: 'Profiler' });
    encoder.resolveQuerySet(gpu.set, 0, count, gpu.resolve, 0);
    encoder.copyBufferToBuffer(gpu.resolve, 0, readback.buffer, 0, count * 8);
    gpu.device.queue.submit([encoder.finish()]);
    readback.buffer.mapAsync(GPUMapMode.READ, 0, count * 8).then(
        () => {
            const stamps = new BigUint64Array(readback.buffer.getMappedRange(0, count * 8).slice(0));
            readback.buffer.unmap();
            readback.free = true;
            // nanoseconds from the frame's first stamp, plus one so a pass that ran never reads 0
            // (0 is a pass that did not run): f64 keeps every nanosecond only that close
            let base = 0n;
            for (const t of stamps) if (t > 0n && (base === 0n || t < base)) base = t;
            const rel = (t: bigint) => (t === 0n ? 0 : Number(t - base) + 1);
            p.arrived.push(labels.map((label, k): [string, number, number] => [label, rel(stamps[2 * k]), rel(stamps[2 * k + 1])]));
        },
        () => { readback.free = true; },
    );
}

/** The frames profiled since the last call, averaged (and forgotten). */
export function takeProfile(): FrameProfile {
    const p = profiler;
    if (!p) return new FrameProfile();
    const profile = gpuProfile(p.arrived);
    p.arrived = [];
    profile.cpuFrames = p.cpuFrames;
    for (const [label, total] of p.cpu) profile.cpu.push([label, total / Math.max(p.cpuFrames, 1)]);
    p.cpu.clear();
    p.cpuFrames = 0;
    return profile;
}
