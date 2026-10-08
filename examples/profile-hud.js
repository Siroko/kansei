// GPU profiling for an example page, as the Rust examples do it (rust/kansei-core/src/profiling.rs):
//   ?profile=1  Renderer.setProfiling: every 2 s, logs FrameProfile.report() (each labelled pass's
//               GPU time and the frame's CPU sections) and shows the costliest passes and the
//               frame's GPU time (FrameTimer) in an overlay. Run Chrome with
//               --enable-webgpu-developer-features for unquantized timestamps.
//   ?bench=1    AbBench: alternates the page's two variants (`bench.labels`, applied by
//               `bench.set(0 | 1)`) and logs one line comparing their GPU time and frame interval.
// The page passes the engine classes in (examples are bundled with their imports rewritten, this
// file is not), and wraps every frame's submits in hud.begin(now) / hud.end(now).

export function installProfileHud({ renderer, FrameTimer, AbBench, bench } = {}) {
  const query = new URLSearchParams(location.search);
  const profiling = query.get('profile') === '1';
  const ab = query.get('bench') === '1' && bench ? new AbBench(bench.labels, performance.now()) : null;
  if (!profiling && !ab) return { begin() {}, end() {} };

  const timer = new FrameTimer(renderer.gpuDevice);
  if (profiling) renderer.setProfiling(true);

  const box = document.createElement('pre');
  box.style.cssText = 'position:fixed;left:8px;bottom:8px;margin:0;padding:6px 8px;z-index:10;'
    + 'font:12px/1.35 ui-monospace,Menlo,monospace;color:#e8f0ff;background:rgba(0,0,0,.65);pointer-events:none';
  document.body.appendChild(box);

  let gpuSamples = [];
  let variant = -1;
  let benchLine = ab ? `bench: ${bench.labels[0]} / ${bench.labels[1]} running...` : '';
  const show = (lines) => { box.textContent = [...lines, benchLine].filter(Boolean).join('\n'); };
  show([profiling ? 'profiling...' : '']);

  if (profiling) {
    setInterval(() => {
      const profile = renderer.takeProfile();
      console.info(profile.report());
      const sorted = [...gpuSamples].sort((a, b) => a - b);
      gpuSamples = [];
      const median = sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
      const frame = timer.hasTimestamps ? `frame GPU ${median.toFixed(2)} ms` : `frame ${median.toFixed(2)} ms to readback (no timestamps)`;
      show([
        `${frame}  passes ${profile.gpuMs.toFixed(2)} ms (span ${profile.gpuSpanMs.toFixed(2)}), ${profile.gpuFrames} frames`,
        ...profile.topPasses(8).map(([label, ms]) => `  ${label.padEnd(30)} ${ms.toFixed(3)} ms`),
      ]);
    }, 2000);
  }

  window.profileHud = { timer, bench: ab };
  return {
    begin(now) {
      if (ab) {
        const phase = ab.phase(now);
        const next = phase ? phase[0] : 0;
        if (next !== variant) bench.set((variant = next));
      }
      timer.begin();
    },
    end(now) {
      timer.end();
      const gpu = timer.take();
      gpuSamples.push(...gpu);
      if (ab) {
        const report = ab.record(gpu, now);
        if (report) {
          console.info(report);
          benchLine = report;
          if (!profiling) show([]);
        }
      }
    },
  };
}
