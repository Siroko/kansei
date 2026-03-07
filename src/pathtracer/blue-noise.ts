/**
 * Generate a tileable blue noise texture using the void-and-cluster algorithm
 * (Ulichney 1993, with sigma=1.9 per demofox recommendations).
 *
 * The algorithm produces a progressive blue noise pattern: thresholding the
 * output at ANY value produces a well-distributed point set. This is critical
 * for path tracer denoising — the spatial filter sees uniform noise at every scale.
 *
 * Three phases:
 *   1. Create + stabilize an initial binary pattern (~10% seed density)
 *   2. Rank seed points (remove tightest clusters) + fill to 50% (add at largest voids)
 *   3. Fill 50%→100% with inverted energy (track 0-pixels, remove tightest 0-clusters)
 *
 * Returns Float32Array of size×size values in [0, 1).
 */
export function generateBlueNoise(size: number = 128): Float32Array {
    const N = size * size;
    const sigma = 1.9;
    const sigma2x2 = 2.0 * sigma * sigma;
    const radius = Math.ceil(3.0 * sigma); // 6 pixels

    const binary = new Uint8Array(N);
    const energy = new Float32Array(N);
    const ranks = new Float32Array(N);

    // ── Helpers ──────────────────────────────────────────────────────────

    // Add or subtract a pixel's Gaussian contribution to the energy LUT
    function updateEnergy(idx: number, sign: number): void {
        const cx = idx % size;
        const cy = (idx / size) | 0;
        for (let dy = -radius; dy <= radius; dy++) {
            for (let dx = -radius; dx <= radius; dx++) {
                const nx = ((cx + dx) % size + size) % size;
                const ny = ((cy + dy) % size + size) % size;
                energy[ny * size + nx] += sign * Math.exp(-(dx * dx + dy * dy) / sigma2x2);
            }
        }
    }

    // Find pixel with binary[i]==val that has max or min energy
    function findExtreme(val: number, findMax: boolean): number {
        let bestIdx = 0;
        let bestE = findMax ? -Infinity : Infinity;
        for (let i = 0; i < N; i++) {
            if (binary[i] !== val) continue;
            if (findMax ? energy[i] > bestE : energy[i] < bestE) {
                bestE = energy[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // Simple xorshift32 RNG
    let seed = 2463534242;
    const rng = (): number => {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        return (seed >>> 0) / 4294967296;
    };

    // ── Phase 1: Create initial binary pattern ──────────────────────────
    // Seed ~10% of pixels randomly
    const initialCount = Math.max(1, Math.floor(N * 0.1));
    let onesCount = 0;
    while (onesCount < initialCount) {
        const idx = Math.floor(rng() * N);
        if (binary[idx]) continue;
        binary[idx] = 1;
        updateEnergy(idx, 1);
        onesCount++;
    }

    // Stabilize: repeatedly move tightest cluster → largest void until convergent
    for (let iter = 0; iter < N; iter++) {
        const clusterIdx = findExtreme(1, true); // 1-pixel with max energy
        binary[clusterIdx] = 0;
        updateEnergy(clusterIdx, -1);

        const voidIdx = findExtreme(0, false); // 0-pixel with min energy
        if (voidIdx === clusterIdx) {
            // Converged: re-insert at same position and stop
            binary[clusterIdx] = 1;
            updateEnergy(clusterIdx, 1);
            break;
        }

        binary[voidIdx] = 1;
        updateEnergy(voidIdx, 1);
    }

    // ── Rank initial seed points (remove tightest clusters) ─────────────
    // Save the stabilized initial pattern
    const savedBinary = new Uint8Array(binary);
    const savedEnergy = new Float32Array(energy);
    let remaining = onesCount;

    while (remaining > 0) {
        const clusterIdx = findExtreme(1, true);
        remaining--;
        ranks[clusterIdx] = remaining; // most clustered → highest rank among seeds
        binary[clusterIdx] = 0;
        updateEnergy(clusterIdx, -1);
    }

    // ── Phase 2: Fill from initial pattern to 50% density ───────────────
    // Restore initial binary pattern
    binary.set(savedBinary);
    energy.set(savedEnergy);
    let currentRank = onesCount;
    const halfN = Math.floor(N / 2);

    while (currentRank < halfN) {
        const voidIdx = findExtreme(0, false); // 0-pixel with min energy = largest void
        ranks[voidIdx] = currentRank;
        binary[voidIdx] = 1;
        updateEnergy(voidIdx, 1);
        currentRank++;
    }

    // ── Phase 3: Fill from 50% to 100% (inverted energy) ────────────────
    // Rebuild energy from 0-pixels (invert meaning of 0s and 1s)
    energy.fill(0);
    for (let i = 0; i < N; i++) {
        if (binary[i] === 0) {
            updateEnergy(i, 1);
        }
    }

    while (currentRank < N) {
        // Tightest cluster of 0s = 0-pixel with max energy
        const clusterIdx = findExtreme(0, true);
        ranks[clusterIdx] = currentRank;
        binary[clusterIdx] = 1;
        updateEnergy(clusterIdx, -1); // remove its 0-contribution
        currentRank++;
    }

    // Normalize ranks to [0, 1)
    for (let i = 0; i < N; i++) {
        ranks[i] /= N;
    }

    return ranks;
}
