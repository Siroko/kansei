/**
 * Kansei on the web: the plumbing every example shares, so an example's code is about the
 * engine feature it shows. The TS side of the Rust `kansei-wasm` crate (`rust/kansei-wasm/src`).
 *
 * ```ts
 * const canvas = Canvas.fill();
 * const renderer = await canvas.renderer({ sampleCount: 4 });
 * const camera = new Camera(45, 0.1, 100, canvas.aspect);
 * const exposure = paramOr('ev', 12);
 * run(canvas, (frame) => {
 *     frame.resize(renderer, camera);
 *     renderer.render(scene, camera);
 * });
 * ```
 *
 * - `Canvas` sizes the drawing buffer to the canvas's CSS box times `devicePixelRatio` (capped
 *   at 2, or `?dpr=` on the page's URL) and follows it when the page resizes.
 * - `run` drives the `requestAnimationFrame` loop and hands each frame its `Frame`: time, delta
 *   time and any new canvas size.
 * - `param`, `paramOr` and `flag` read the page's query string, percent-decoded.
 * - `now`, `fetchBytes` and `isPhone` cover timing, loading and picking a tier; `setText`,
 *   `checkbox` and `thousands` serve a HUD.
 * - `Keys` and `Gamepad` are the input of pages that play: keys held and pressed, sticks and
 *   buttons.
 */
export { Canvas } from "./Canvas";
export { Frame, run } from "./run";
export { Keys, Gamepad, deadZone } from "./input";
export { now, param, paramOr, flag, isPhone, setText, checkbox, thousands, fetchBytes } from "./page";
