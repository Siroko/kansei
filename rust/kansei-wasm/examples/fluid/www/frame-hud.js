// A frame-time overlay shared by the Rust fluid example and its TypeScript original
// (examples/index_fluid.html), so the two can be watched side by side: load either with ?hud=1.
//
// A rendered frame is a call to getCurrentTexture (each page calls it once a frame, whichever
// engine draws it), so refreshes a frame loop skips are not frames. Over the last two seconds
// it shows the frame rate, the median and p95 interval between frames, the share of intervals
// that span 1, 2, 3 and 4 or more display refreshes (?hz=, 120 by default), and the sim steps
// each frame ran (from the page's `steps` callback, read when the frame starts drawing).

export function installFrameHud({ steps = () => NaN } = {}) {
  const hz = Number(new URLSearchParams(location.search).get('hz')) || 120;
  const refresh = 1000 / hz;
  const frames = []; // [time ms, sim steps]
  const getCurrentTexture = GPUCanvasContext.prototype.getCurrentTexture;
  GPUCanvasContext.prototype.getCurrentTexture = function () {
    frames.push([performance.now(), steps()]);
    return getCurrentTexture.call(this);
  };

  const stats = () => {
    const now = performance.now();
    while (frames.length && now - frames[0][0] > 2000) frames.shift();
    const intervals = [];
    for (let i = 1; i < frames.length; i++) intervals.push(frames[i][0] - frames[i - 1][0]);
    const sorted = [...intervals].sort((a, b) => a - b);
    const at = (p) => (sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))] : NaN);
    const refreshes = [0, 0, 0, 0];
    for (const v of intervals) refreshes[Math.min(4, Math.max(1, Math.round(v / refresh))) - 1]++;
    const stepCounts = {};
    for (const [, s] of frames) stepCounts[s] = (stepCounts[s] || 0) + 1;
    const span = frames.length > 1 ? (frames[frames.length - 1][0] - frames[0][0]) / 1000 : 0;
    return {
      fps: span > 0 ? intervals.length / span : 0,
      medianMs: at(0.5),
      p95Ms: at(0.95),
      refreshShare: refreshes.map((n) => (intervals.length ? n / intervals.length : 0)),
      steps: frames.length ? frames[frames.length - 1][1] : NaN,
      stepCounts,
    };
  };

  const box = document.createElement('pre');
  box.style.cssText = 'position:fixed;left:8px;top:8px;margin:0;padding:6px 8px;z-index:10;'
    + 'font:12px/1.35 ui-monospace,Menlo,monospace;color:#e8f0ff;background:rgba(0,0,0,.65);pointer-events:none';
  document.body.appendChild(box);
  setInterval(() => {
    const s = stats();
    const pct = s.refreshShare.map((x) => `${Math.round(100 * x)}%`).join(' / ');
    const stepShare = Object.entries(s.stepCounts).map(([k, n]) => `${k}:${n}`).join(' ');
    box.textContent = `${s.fps.toFixed(1)} fps  (last 2 s, ${hz} Hz)\n`
      + `interval median ${s.medianMs.toFixed(1)} ms  p95 ${s.p95Ms.toFixed(1)} ms\n`
      + `1 / 2 / 3 / 4+ refreshes  ${pct}\n`
      + `sim steps this frame ${s.steps}   (frames by steps ${stepShare})`;
  }, 250);
  window.frameHud = { stats };
}
