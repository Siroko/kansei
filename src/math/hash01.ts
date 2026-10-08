/**
 * A deterministic pseudo-random number in 0..1 for `i` (an integer hash, 10 007 steps): the same
 * scatter of trees, rocks or tints on every run and platform, and the same as the Rust engine's
 * `math::hash01` for the same `i`.
 */
export function hash01(i: number): number {
    const u = i >>> 0;
    const x = ((Math.imul(u, 747796405) + 2891336453) ^ Math.imul(u >>> 7, 277803737)) >>> 0;
    return (x % 10007) / 10007;
}
