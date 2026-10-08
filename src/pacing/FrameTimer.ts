interface TimerSlot {
    readback: GPUBuffer;
    busy: boolean;
}

/**
 * Each frame's GPU time: a timestamp before the frame's first submit (`begin`) and after its last
 * (`end`), where the device has `timestamp-query`; otherwise the CPU time from `begin` until the
 * frame's readback, copied after the last submit, becomes mappable: an upper bound, queueing
 * included. A ring of readbacks measures every frame although results arrive a few frames late;
 * the ring shares one query set (on Metal each set is a counter sample buffer, of which a browser
 * tab gets few). A frame's stamps are resolved at the next frame's start (or `flush`); a frame
 * whose stamps read back out of order is left unmeasured. A port of the Rust engine's
 * `pacing::FrameTimer`.
 *
 * ```ts
 * const timer = new FrameTimer(renderer.gpuDevice);
 * // each frame:
 * timer.begin();
 * volume.render(scene, camera);
 * timer.end();
 * const gpuMs = timer.take(); // the frames measured since the last call
 * ```
 */
export class FrameTimer {
    private static readonly SLOTS = 8;
    private static readonly STAMPS = 4;
    /** Query resolves land at multiples of 256 bytes. */
    private static readonly RESOLVE_STRIDE = 256;

    private readonly device: GPUDevice;
    private readonly noop: GPUComputePipeline | null = null;
    /** `STAMPS` timestamps per slot, and where each slot's are resolved (`RESOLVE_STRIDE` apart). */
    private readonly set: GPUQuerySet | null = null;
    private readonly resolve: GPUBuffer;
    private readonly slots: TimerSlot[];
    private armed: number | null = null;
    private startedMs = 0;
    /** Frames ended whose stamps are not resolved yet: their slot and CPU start time. */
    private unresolved: [number, number][] = [];
    private results: number[] = [];
    private _lastMs = NaN;

    constructor(device: GPUDevice) {
        this.device = device;
        if (device.features.has('timestamp-query')) {
            const module = device.createShaderModule({ label: 'FrameTimer', code: '@compute @workgroup_size(1) fn main() {}' });
            this.noop = device.createComputePipeline({ label: 'FrameTimer', layout: 'auto', compute: { module, entryPoint: 'main' } });
            this.set = device.createQuerySet({ label: 'FrameTimer', type: 'timestamp', count: FrameTimer.SLOTS * FrameTimer.STAMPS });
        }
        this.resolve = device.createBuffer({
            label: 'FrameTimer',
            size: FrameTimer.SLOTS * FrameTimer.RESOLVE_STRIDE,
            usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
        });
        this.slots = Array.from({ length: FrameTimer.SLOTS }, () => ({
            readback: device.createBuffer({ label: 'FrameTimer', size: 8 * FrameTimer.STAMPS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }),
            busy: false,
        }));
    }

    private stamp(encoder: GPUCommandEncoder, slot: number, end: boolean): void {
        if (!this.noop || !this.set) return;
        const first = slot * FrameTimer.STAMPS + (end ? 2 : 0);
        const pass = encoder.beginComputePass({
            label: 'FrameTimer',
            // both stamps of both passes (the end of a pass is not always written on every GPU)
            timestampWrites: { querySet: this.set, beginningOfPassWriteIndex: first, endOfPassWriteIndex: first + 1 },
        });
        pass.setPipeline(this.noop);
        pass.dispatchWorkgroups(1);
        pass.end();
    }

    /** Before the frame's first submit (a frame starts unmeasured when every readback is busy). */
    begin(): void {
        this.flush();
        const slot = this.slots.findIndex((s) => !s.busy);
        this.armed = slot >= 0 ? slot : null;
        if (this.armed === null) return;
        this.startedMs = performance.now();
        if (this.noop) {
            const encoder = this.device.createCommandEncoder({ label: 'FrameTimer/Start' });
            this.stamp(encoder, this.armed, false);
            this.device.queue.submit([encoder.finish()]);
        }
    }

    /** After the frame's last submit. */
    end(): void {
        const k = this.armed;
        if (k === null) return;
        this.armed = null;
        this.slots[k].busy = true;
        const encoder = this.device.createCommandEncoder({ label: 'FrameTimer/End' });
        this.stamp(encoder, k, true);
        this.device.queue.submit([encoder.finish()]);
        this.unresolved.push([k, this.startedMs]);
        // without timestamps the readback is the measurement: at once
        if (!this.noop) this.flush();
    }

    /** Resolve the ended frames' stamps and read them back (`begin` does, for the frames before). */
    flush(): void {
        if (this.unresolved.length === 0) return;
        const encoder = this.device.createCommandEncoder({ label: 'FrameTimer/Resolve' });
        for (const [k] of this.unresolved) {
            const offset = k * FrameTimer.RESOLVE_STRIDE;
            if (this.set) {
                encoder.resolveQuerySet(this.set, k * FrameTimer.STAMPS, FrameTimer.STAMPS, this.resolve, offset);
            }
            encoder.copyBufferToBuffer(this.resolve, offset, this.slots[k].readback, 0, 8 * FrameTimer.STAMPS);
        }
        this.device.queue.submit([encoder.finish()]);
        const timestamps = this.noop !== null;
        for (const [k, started] of this.unresolved) {
            const slot = this.slots[k];
            slot.readback.mapAsync(GPUMapMode.READ).then(
                () => {
                    let ms: number | null;
                    if (timestamps) {
                        const t = new BigUint64Array(slot.readback.getMappedRange().slice(0));
                        // from the start pass's first stamp to the end pass's last written one
                        const start = t[0];
                        const end = t[3] > t[2] ? t[3] : t[2];
                        ms = start > 0n && end > start ? Number(end - start) / 1e6 : null;
                    } else {
                        ms = performance.now() - started;
                    }
                    slot.readback.unmap();
                    if (ms !== null) {
                        this.results.push(ms);
                        this._lastMs = ms;
                    }
                    slot.busy = false;
                },
                () => { slot.busy = false; },
            );
        }
        this.unresolved = [];
    }

    /** The frames measured since the last call, ms. */
    take(): number[] {
        const results = this.results;
        this.results = [];
        return results;
    }

    /**
     * Whether the GPU times come from timestamp queries (else they are each frame's CPU start to
     * its readback, an upper bound).
     */
    get hasTimestamps(): boolean {
        return this.noop !== null;
    }

    /** The last frame measured, ms (NaN until one arrives). */
    get lastMs(): number {
        return this._lastMs;
    }

    /** Release the timer's GPU resources. */
    destroy(): void {
        this.set?.destroy();
        this.resolve.destroy();
        for (const s of this.slots) s.readback.destroy();
    }
}
