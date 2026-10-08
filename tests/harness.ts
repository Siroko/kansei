/**
 * A minimal test harness for the engine's CPU-only code (no GPU, no DOM): Rust `#[test]`s ported
 * one for one. `pnpm test` runs every `tests/**\/*.test.ts` in Node through Vite
 * (`scripts/run-tests.mjs`); `tsc` type-checks them with `src`.
 */

type Vector = ArrayLike<number>;

interface TestCase {
    file: string;
    name: string;
    run: () => void | Promise<void>;
}

const cases: TestCase[] = [];
let currentFile = "";

/** Register a test; it fails when it throws. */
export function test(name: string, run: () => void | Promise<void>): void {
    cases.push({ file: currentFile, name, run });
}

/** Throw `message` unless `condition`. */
export function assert(condition: unknown, message: string = "assertion failed"): asserts condition {
    if (!condition) throw new Error(message);
}

export function assertEq<T>(actual: T, expected: T, message: string = ""): void {
    const a = JSON.stringify(actual), e = JSON.stringify(expected);
    if (a !== e) throw new Error(`${message ? message + ": " : ""}${a} != ${e}`);
}

/** Every component of `a` within `eps` of `b`'s (and the same length). */
export function close(a: Vector, b: Vector, eps: number): boolean {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (!(Math.abs(a[i] - b[i]) <= eps)) return false;
    return true;
}

export function assertClose(a: Vector | number, b: Vector | number, eps: number, message: string = ""): void {
    const x = typeof a === "number" ? [a] : a, y = typeof b === "number" ? [b] : b;
    if (!close(x, y, eps)) throw new Error(`${message ? message + ": " : ""}[${Array.from(x)}] vs [${Array.from(y)}] (eps ${eps})`);
}

/** The same rotation (q or -q) within `eps`. */
export function sameRotation(a: Vector, b: Vector, eps: number): boolean {
    const dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
    return Math.abs(dot) > 1 - eps;
}

/** Run the tests registered by `load` (each test file's import), print, and return the failures. */
export async function runAll(files: [string, () => Promise<unknown>][]): Promise<number> {
    for (const [file, load] of files) {
        currentFile = file;
        await load();
    }
    let failed = 0;
    for (const c of cases) {
        try {
            await c.run();
            console.log(`ok    ${c.file} › ${c.name}`);
        } catch (e) {
            failed++;
            console.log(`FAIL  ${c.file} › ${c.name}\n      ${(e as Error).message}`);
        }
    }
    console.log(`\n${cases.length - failed} passed, ${failed} failed`);
    return failed;
}
