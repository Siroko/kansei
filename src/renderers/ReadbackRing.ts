/**
 * Reads a small GPU buffer back without ever stalling the frame: a ring of `MAP_READ` staging
 * buffers, each copied into at the end of the caller's encoder and mapped once that work is
 * submitted. A copy goes into a free staging buffer, and `copy` does nothing (returns false) while
 * every one is still on its way. `take` returns the newest result that has arrived, once, a frame
 * or a few after its copy. The Rust engine's probe (`FluidSpeedProbe`) keeps one staging buffer
 * in flight; `Renderer.readBackBuffer` is the awaited, one-off form.
 *
 * ```ts
 * const enc = device.createCommandEncoder();
 * // ... a pass writing `result` ...
 * const copied = ring.copy(enc, result);
 * device.queue.submit([enc.finish()]);
 * if (copied) ring.submitted();
 * const words = ring.take(); // a Uint32Array, or null until one arrives
 * ```
 */
export class ReadbackRing {
    private readonly slots: {
        buffer: GPUBuffer;
        /** 0 free, 1 copied (awaiting its submit), 2 mapping, 3 mapped */
        state: number;
        /** The order its copy was made in */
        serial: number;
        /** Whether its result is still wanted (`forget` clears it) */
        wanted: boolean;
    }[] = [];
    private serial = 0;
    /** The serial of the newest result handed out, so an older one arriving late is dropped */
    private taken = -1;

    /** `depth` staging buffers of `size` bytes (a multiple of 4). */
    constructor(device: GPUDevice, readonly size: number, depth: number = 2, label: string = 'ReadbackRing') {
        for (let i = 0; i < Math.max(Math.floor(depth), 1); i++) {
            this.slots.push({
                buffer: device.createBuffer({ label: `${label}/Staging${i}`, size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
                state: 0,
                serial: 0,
                wanted: false,
            });
        }
    }

    /** Whether a copy would find a free staging buffer. */
    get free(): boolean {
        return this.slots.some((s) => s.state === 0);
    }

    /** Whether any copy is on its way. */
    get inFlight(): boolean {
        return this.slots.some((s) => s.state !== 0);
    }

    /**
     * Copy `size` bytes of `source` (from `offset`) into a free staging buffer at the end of
     * `encoder`; false (and nothing recorded) when none is free. Call `submitted` once the encoder
     * is submitted.
     */
    copy(encoder: GPUCommandEncoder, source: GPUBuffer, offset: number = 0): boolean {
        const slot = this.slots.find((s) => s.state === 0);
        if (!slot) return false;
        encoder.copyBufferToBuffer(source, offset, slot.buffer, 0, this.size);
        slot.state = 1;
        slot.serial = this.serial++;
        slot.wanted = true;
        return true;
    }

    /** The encoder holding the copies made since the last call was submitted: map them. */
    submitted(): void {
        for (const slot of this.slots) {
            if (slot.state !== 1) continue;
            slot.state = 2;
            slot.buffer.mapAsync(GPUMapMode.READ).then(
                () => { slot.state = 3; },
                // a lost device or a destroyed buffer: free the slot, its result never comes
                () => { slot.state = 0; },
            );
        }
    }

    /** Drop the results on their way (taken before something changed): `take` will not return them. */
    forget(): void {
        for (const slot of this.slots) slot.wanted = false;
    }

    /**
     * The newest result that has arrived since the last call, as 32-bit words, or null. Older
     * results arriving with it, or after it, are dropped.
     */
    take(): Uint32Array | null {
        let newest: (typeof this.slots)[number] | null = null;
        for (const slot of this.slots) {
            if (slot.state === 3 && slot.wanted && slot.serial > this.taken && (!newest || slot.serial > newest.serial)) newest = slot;
        }
        const words = newest ? new Uint32Array(newest.buffer.getMappedRange().slice(0)) : null;
        if (newest) this.taken = newest.serial;
        for (const slot of this.slots) {
            if (slot.state !== 3) continue;
            slot.buffer.unmap();
            slot.state = 0;
        }
        return words;
    }

    destroy(): void {
        for (const slot of this.slots) slot.buffer.destroy();
    }
}
