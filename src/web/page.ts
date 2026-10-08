/**
 * The page around the canvas: the clock, the query string, fetching files and a HUD's helpers.
 * The TS side of `rust/kansei-wasm/src/page.rs`.
 */

/** Seconds since the page started (`performance.now()`), at sub-millisecond resolution. */
export function now(): number {
    return performance.now() / 1000;
}

/**
 * The query string's value for `name`, percent-decoded (`?gi=voxel%2Bssgi` reads `voxel+ssgi`;
 * a literal `+` reads as a space, as in any form-encoded URL). `null` when the page's URL has
 * no `name`.
 */
export function param(name: string): string | null {
    return new URLSearchParams(location.search).get(name);
}

/**
 * The query string's value for `name` read as a number, or `fallback` when it is missing or
 * not a number: `const ev = paramOr('ev', 12)`. With a string fallback the raw value is
 * returned; with a boolean one, `flag`.
 */
export function paramOr(name: string, fallback: number): number;
export function paramOr(name: string, fallback: string): string;
export function paramOr(name: string, fallback: boolean): boolean;
export function paramOr(name: string, fallback: number | string | boolean): number | string | boolean {
    if (typeof fallback === 'boolean') return flag(name, fallback);
    const value = param(name);
    if (value === null) return fallback;
    if (typeof fallback === 'string') return value;
    const trimmed = value.trim();
    const parsed = Number(trimmed);
    return trimmed === '' || Number.isNaN(parsed) ? fallback : parsed;
}

/**
 * A switch in the query string: `1`, `true`, `on` or `yes` turn it on, `0`, `false`, `off` or
 * `no` off; anything else (or no `name`) leaves `fallback`.
 */
export function flag(name: string, fallback: boolean): boolean {
    const value = param(name)?.trim();
    if (value === '1' || value === 'true' || value === 'on' || value === 'yes') return true;
    if (value === '0' || value === 'false' || value === 'off' || value === 'no') return false;
    return fallback;
}

/**
 * Whether the browser says it is a phone or tablet (by its user agent), for examples that pick
 * a lighter quality tier there.
 */
export function isPhone(): boolean {
    return ['Mobi', 'Android', 'iPhone', 'iPad'].some((k) => navigator.userAgent.includes(k));
}

/** Show `text` in the page element with id `id` (a HUD), if there is one. */
export function setText(id: string, text: string): void {
    const element = document.getElementById(id);
    if (element) element.textContent = text;
}

/** The page's checkbox with id `id`, if there is one (a HUD's toggles). */
export function checkbox(id: string): HTMLInputElement | null {
    const element = document.getElementById(id);
    return element instanceof HTMLInputElement ? element : null;
}

/** `n` with its thousands apart, for a HUD: `40000` reads `40 000`. */
export function thousands(n: number): string {
    return String(Math.trunc(n)).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
}

/** Fetch `url` (relative to the page) as bytes; an HTTP error status is an error too. */
export async function fetchBytes(url: string): Promise<Uint8Array> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return new Uint8Array(await response.arrayBuffer());
}
